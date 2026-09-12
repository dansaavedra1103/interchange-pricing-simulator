"""End to end elasticity: calibrate the acceptance curve, then race the link-prediction ladder.

Two halves that meet in the simulator. The curve says **how much** volume walks away when a
merchant's burden rises; the ladder says **where it goes**. Feature 6 chains them; this module
produces and stores both, together with the sentences that state how each comparison came out.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl

from ips.elasticity.calibration import calibrate_acceptance
from ips.elasticity.cardholder_response import segment_response
from ips.elasticity.gnn import fit_graphsage, graphsage_scores
from ips.elasticity.link_prediction import (
    GRAPHSAGE,
    GROUP_POPULARITY,
    POPULARITY,
    RANDOM,
    SVD,
    VERDICT_PAIRS,
    LinkSplit,
    evaluate_rankings,
    merchant_substitutes,
    ranking_gaps,
    redistribute_volume,
    redistribution_leakage,
    restrict_candidates,
    split_by_time,
    verdict,
)
from ips.elasticity.merchant_acceptance import (
    AcceptanceCurve,
    abandonment_probability,
    acceptance_response,
)
from ips.elasticity.scorers import (
    embedding_scores,
    group_popularity_scores,
    popularity_scores,
    random_scores,
)
from ips.graph.baseline_kmeans import feature_matrix
from ips.graph.build_graph import build_bipartite, load_edges
from ips.graph.features import (
    clustered_population,
    load_merchants,
    load_priced_transactions,
    merchant_profile,
)
from ips.graph.spectral_embed import embed_merchants
from ips.utils.config import ProjectConfig, load_config
from ips.utils.io import artifacts_dir, read_table
from ips.utils.logging import get_logger

ARTIFACTS = {
    "curve": Path("elasticity") / "acceptance_curve.parquet",
    "acceptance": Path("elasticity") / "merchant_acceptance.parquet",
    "response": Path("elasticity") / "acceptance_response.parquet",
    "cardholder": Path("elasticity") / "cardholder_response.parquet",
    "comparison": Path("elasticity") / "link_prediction.parquet",
    "gaps": Path("elasticity") / "link_prediction_gaps.parquet",
    "substitutes": Path("elasticity") / "merchant_substitutes.parquet",
    "redistribution": Path("elasticity") / "redistribution.parquet",
    "losses": Path("elasticity") / "graphsage_losses.parquet",
}
# Recorte de recompensas con el que se reporta la respuesta del titular: un punto porcentual.
REWARDS_SHOCK = -0.01


@dataclass(frozen=True)
class ElasticityRun:
    """Everything one run produces, on both halves of the feature."""

    curve: AcceptanceCurve
    acceptance: pl.DataFrame
    response: pl.DataFrame
    cardholder: pl.DataFrame
    comparison: pl.DataFrame
    gaps: pl.DataFrame
    substitutes: pl.DataFrame
    redistribution: pl.DataFrame
    leaked_cop: float
    losses: pl.DataFrame
    substitute_source: str
    split_stats: dict[str, float]

    def verdicts(self) -> list[str]:
        """How each comparison came out, written straight from the numbers."""
        return [
            (
                "The acceptance curve is an assumption, not an estimate: it reproduces an annual "
                f"abandonment of {self.curve.achieved_abandonment:.3f} (anchor "
                f"{self.curve.target_abandonment:.3f}) and a semi-elasticity of "
                f"{self.curve.achieved_semi_elasticity:.4f} (anchor "
                f"{self.curve.target_semi_elasticity:.4f})."
            ),
            *(verdict(self.gaps, baseline, challenger) for baseline, challenger in VERDICT_PAIRS),
        ]

    def conserves_volume(self, tolerance: float = 1e-9) -> bool:
        """Redistributed plus leakage equals what was lost, as the P&L identity does for fees."""
        lost = float(self.acceptance["gdv_at_risk_cop"].sum())
        moved = float(self.redistribution["gdv_moved_cop"].sum())
        return abs(moved + self.leaked_cop - lost) <= tolerance * max(lost, 1.0)


def _link_prediction(
    edges: pl.DataFrame,
    profile: pl.DataFrame,
    cardholders: pl.DataFrame,
    cfg: ProjectConfig,
) -> tuple[LinkSplit, pl.DataFrame, pl.DataFrame, np.ndarray, np.ndarray, pl.DataFrame, str]:
    """Run the whole ladder on one temporal split and return it with the winner's vectors."""
    logger = get_logger(__name__)
    split = split_by_time(edges, cfg)

    graph = build_bipartite(split.train, cfg)
    embedding = embed_merchants(graph, cfg)
    usable = np.intersect1d(embedding.merchant_ids, profile["merchant_id"].to_numpy())
    split = restrict_candidates(split, usable, cfg)
    logger.info("link prediction split: %s", split.stats())

    merchants = profile.join(
        pl.DataFrame({"merchant_id": usable}), on="merchant_id", how="inner"
    ).sort("merchant_id")
    merchant_ids = merchants["merchant_id"].to_numpy()
    merchant_matrix, _ = feature_matrix(merchants, cfg)

    trained = fit_graphsage(split, cardholders, merchant_ids, merchant_matrix, cfg)
    scores = {
        RANDOM: random_scores(split, cfg),
        POPULARITY: popularity_scores(split),
        GROUP_POPULARITY: group_popularity_scores(split, profile, cfg),
        SVD: embedding_scores(split, embedding),
        GRAPHSAGE: graphsage_scores(split, trained),
    }
    comparison = evaluate_rankings(scores, split, cfg)
    gaps = ranking_gaps(scores, split, cfg)

    # Los sustitutos salen de la representación que ganó entre las dos que tienen vectores de
    # comercio; cuál fue queda registrado, para que el artifact no esconda de dónde viene.
    ranking = dict(zip(comparison["method"], comparison["recall_at_k"], strict=True))
    source = GRAPHSAGE if ranking[GRAPHSAGE] >= ranking[SVD] else SVD
    order = np.searchsorted(embedding.merchant_ids, merchant_ids)
    vectors = trained.merchant_vectors if source == GRAPHSAGE else embedding.vectors[order]
    return split, comparison, gaps, merchant_ids, vectors, trained.frame(), source


def run_elasticity(
    cfg: ProjectConfig | None = None,
    months: int | None = None,
    warehouse: Path | None = None,
) -> ElasticityRun:
    """Read the warehouse, calibrate the curve and race the link-prediction ladder."""
    config = cfg if cfg is not None else load_config()
    logger = get_logger(__name__)
    start = time.perf_counter()

    edges = load_edges(months, config, warehouse)
    profile = merchant_profile(
        load_priced_transactions(warehouse), load_merchants(warehouse), config
    )
    cardholders = read_table("dim_cardholder", warehouse).select(
        "cardholder_id", "spend_segment", "product"
    )
    population = clustered_population(profile, config)
    logger.info(
        "%s edges, %s merchants with purchases, %s in the population",
        f"{edges.height:,}",
        f"{profile.height:,}",
        f"{population.height:,}",
    )

    run = elasticity_run(edges, profile, population, cardholders, config)
    logger.info("elasticity finished in %.1f s", time.perf_counter() - start)
    return run


def elasticity_run(
    edges: pl.DataFrame,
    profile: pl.DataFrame,
    population: pl.DataFrame,
    cardholders: pl.DataFrame,
    cfg: ProjectConfig,
) -> ElasticityRun:
    """The whole feature on frames already in memory, so the tests can skip the warehouse."""
    curve = calibrate_acceptance(population, cfg)
    acceptance = abandonment_probability(population, cfg, curve)
    response = acceptance_response(
        population, cfg, curve, cfg.elasticity.acceptance.semi_elasticity_shock
    )

    split, comparison, gaps, merchant_ids, vectors, losses, source = _link_prediction(
        edges, profile, cardholders, cfg
    )
    substitutes = merchant_substitutes(vectors, merchant_ids, cfg)
    lost = acceptance.select("merchant_id", pl.col("gdv_at_risk_cop").alias("gdv_lost_cop"))
    redistribution = redistribute_volume(lost, substitutes, cfg)
    return ElasticityRun(
        curve=curve,
        acceptance=acceptance,
        response=response,
        cardholder=segment_response(cfg, REWARDS_SHOCK),
        comparison=comparison,
        gaps=gaps,
        substitutes=substitutes,
        redistribution=redistribution,
        leaked_cop=redistribution_leakage(lost, substitutes, cfg),
        losses=losses,
        substitute_source=source,
        split_stats=split.stats(),
    )


def write_artifacts(run: ElasticityRun, cfg: ProjectConfig | None = None) -> dict[str, Path]:
    """Write the curve, the acceptance table, the ladder and the substitutes under artifacts."""
    base = artifacts_dir(cfg)
    paths = {name: base / relative for name, relative in ARTIFACTS.items()}
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    run.curve.frame().write_parquet(paths["curve"])
    run.acceptance.write_parquet(paths["acceptance"])
    run.response.write_parquet(paths["response"])
    run.cardholder.write_parquet(paths["cardholder"])
    run.comparison.with_columns(
        pl.lit(run.substitute_source).alias("substitute_source")
    ).write_parquet(paths["comparison"])
    run.gaps.write_parquet(paths["gaps"])
    run.substitutes.write_parquet(paths["substitutes"])
    run.redistribution.with_columns(pl.lit(run.leaked_cop).alias("gdv_leaked_cop")).write_parquet(
        paths["redistribution"]
    )
    run.losses.write_parquet(paths["losses"])
    return paths
