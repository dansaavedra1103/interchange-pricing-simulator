"""End to end segmentation: graph, embeddings, the ladder of methods, and the artifacts.

Every method labels exactly the same merchants — those with enough purchases in the window —
so the comparison is about the representation and the objective, not about who was included.
The ladder runs from the benchmark that needs no model (the merchant's own MCC group), through
the unsupervised methods, to the supervised one that targets the business number directly.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl

from ips.graph.baseline_kmeans import BENCHMARK, feature_matrix, fit_baseline, segment_by_mcc_group
from ips.graph.build_graph import build_bipartite, load_edges, project_merchants
from ips.graph.clustering import Segmentation, cluster_merchants, combine_spaces
from ips.graph.communities import leiden_communities
from ips.graph.evaluate import (
    compare_segmentations,
    marginal_information,
    marginal_verdict,
    verdict,
)
from ips.graph.features import (
    clustered_population,
    load_merchants,
    load_priced_transactions,
    merchant_profile,
)
from ips.graph.segment_profiles import profile_segments
from ips.graph.spectral_embed import MerchantEmbedding, embed_merchants
from ips.graph.supervised import TARGET, fit_supervised, segment_rules
from ips.utils.config import ProjectConfig, load_config
from ips.utils.io import artifacts_dir, raw_dir
from ips.utils.logging import get_logger

BASELINE, GRAPH, HYBRID, LEIDEN = "baseline", "graph", "hybrid", "leiden"
SUPERVISED, SUPERVISED_GRAPH = "supervised", "supervised_graph"
SPACES = ("attributes", "graph", "attributes + graph")
ARTIFACTS = {
    "embeddings": Path("embeddings") / "merchant_embeddings.parquet",
    "segments": Path("segments") / "merchant_segments.parquet",
    "comparison": Path("segments") / "segmentation_comparison.parquet",
    "profiles": Path("segments") / "segment_profiles.parquet",
    "rules": Path("segments") / "segment_rules.parquet",
    "marginal": Path("segments") / "marginal_information.parquet",
}


@dataclass(frozen=True)
class SegmentationRun:
    """What one run produces: the population, every method, the comparison and the profiles."""

    population: pl.DataFrame
    embedding: MerchantEmbedding
    segmentations: tuple[Segmentation, ...]
    comparison: pl.DataFrame
    marginal: pl.DataFrame
    profiles: pl.DataFrame
    rules: pl.DataFrame
    graph_stats: dict[str, float]

    def method(self, name: str) -> Segmentation:
        """The segmentation produced by one method."""
        for segmentation in self.segmentations:
            if segmentation.name == name:
                return segmentation
        raise KeyError(f"No segmentation named {name!r}")

    @property
    def verdict(self) -> str:
        """The Feature 4 question: does the graph beat the attribute baseline?"""
        return verdict(self.comparison, BASELINE, GRAPH)

    def verdicts(self) -> list[str]:
        """The three sentences that close the comparison, whichever way they come out."""
        return [
            verdict(self.comparison, BASELINE, GRAPH),
            verdict(self.comparison, BENCHMARK, SUPERVISED),
            marginal_verdict(self.marginal),
        ]

    def segments_frame(self) -> pl.DataFrame:
        """One row per merchant with the segment each method gave it."""
        frame = self.population.select("merchant_id", "mcc_group", "size_tier", "gdv_cop")
        for segmentation in self.segmentations:
            frame = frame.join(
                segmentation.labels.rename({"segment": f"{segmentation.name}_segment"}),
                on="merchant_id",
                how="left",
            )
        return frame


def segment_merchants(
    edges: pl.DataFrame,
    profile: pl.DataFrame,
    cfg: ProjectConfig,
    latent: pl.DataFrame | None = None,
) -> SegmentationRun:
    """Run every method over the same population and compare them."""
    graph = build_bipartite(edges, cfg)
    in_graph = pl.DataFrame({"merchant_id": pl.Series(graph.merchant_ids, dtype=pl.UInt32)})
    population = (
        clustered_population(profile, cfg)
        .join(in_graph, on="merchant_id", how="inner")
        .sort("merchant_id")
    )
    if population.height <= max(cfg.segmentation.k_values):
        raise ValueError("Too few merchants with purchases in the window to segment")

    embedding = embed_merchants(graph, cfg)
    position = {int(merchant): row for row, merchant in enumerate(embedding.merchant_ids)}
    rows = np.array([position[int(merchant)] for merchant in population["merchant_id"]])
    merchant_ids = population["merchant_id"].to_numpy()

    vectors = embedding.vectors[rows]
    tabular, names = feature_matrix(population, cfg)
    both = combine_spaces(tabular, vectors)
    supervised, tree = fit_supervised(population, tabular, cfg, SUPERVISED)
    supervised_graph, _ = fit_supervised(population, both, cfg, SUPERVISED_GRAPH)
    segmentations = (
        segment_by_mcc_group(population),
        fit_baseline(population, cfg),
        cluster_merchants(GRAPH, merchant_ids, vectors, cfg),
        cluster_merchants(HYBRID, merchant_ids, both, cfg),
        leiden_communities(project_merchants(edges, cfg), merchant_ids, cfg),
        supervised,
        supervised_graph,
    )
    spaces = dict(zip(SPACES, (tabular, vectors, both), strict=True))
    return SegmentationRun(
        population=population,
        embedding=embedding,
        segmentations=segmentations,
        comparison=compare_segmentations(segmentations, population, cfg, latent),
        marginal=marginal_information(spaces, population[TARGET].to_numpy(), cfg),
        # El perfil es el de la segmentación supervisada: es la que un equipo llevaría al memo.
        profiles=profile_segments(supervised, population),
        rules=segment_rules(tree, names),
        graph_stats=graph.stats(),
    )


def load_latent(cfg: ProjectConfig | None = None) -> pl.DataFrame | None:
    """Planted clientele of each merchant, when the generator wrote it.

    Es ground truth del generador: sirve como diagnóstico, nunca como criterio de éxito.
    """
    path = raw_dir(cfg) / "latent" / "merchants_latent.parquet"
    if not path.exists():
        return None
    frame = pl.read_parquet(path)
    if "clientele" not in frame.columns:
        return None
    return frame.select("merchant_id", "clientele")


def run_segmentation(
    cfg: ProjectConfig | None = None,
    months: int | None = None,
    warehouse: Path | None = None,
) -> SegmentationRun:
    """Read the warehouse, segment the merchants and compare the methods."""
    config = cfg if cfg is not None else load_config()
    logger = get_logger(__name__)
    start = time.perf_counter()
    edges = load_edges(months, config, warehouse)
    profile = merchant_profile(
        load_priced_transactions(warehouse), load_merchants(warehouse), config
    )
    logger.info("%s edges, %s merchants with purchases", f"{edges.height:,}", f"{profile.height:,}")
    run = segment_merchants(edges, profile, config, load_latent(config))
    logger.info(
        "segmented %s merchants in %.1f s",
        f"{run.population.height:,}",
        time.perf_counter() - start,
    )
    return run


def write_artifacts(run: SegmentationRun, cfg: ProjectConfig | None = None) -> dict[str, Path]:
    """Write embeddings, segments, comparison, profiles and rules under ``data/artifacts``."""
    base = artifacts_dir(cfg)
    paths = {name: base / relative for name, relative in ARTIFACTS.items()}
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    run.embedding.write_parquet(paths["embeddings"])
    run.segments_frame().write_parquet(paths["segments"])
    run.comparison.write_parquet(paths["comparison"])
    run.profiles.write_parquet(paths["profiles"])
    run.rules.write_parquet(paths["rules"])
    run.marginal.write_parquet(paths["marginal"])
    return paths
