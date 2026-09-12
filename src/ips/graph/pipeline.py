"""End to end segmentation: graph, embeddings, three methods, comparison and artifacts.

The three methods label exactly the same merchants — those with enough purchases in the
window — so the comparison is about the space they live in, not about who was included.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl

from ips.graph.baseline_kmeans import feature_matrix, fit_baseline
from ips.graph.build_graph import build_bipartite, load_edges, project_merchants
from ips.graph.clustering import Segmentation, cluster_merchants, combine_spaces
from ips.graph.communities import leiden_communities
from ips.graph.evaluate import compare_segmentations, verdict
from ips.graph.features import (
    clustered_population,
    load_merchants,
    load_priced_transactions,
    merchant_profile,
)
from ips.graph.segment_profiles import profile_segments
from ips.graph.spectral_embed import MerchantEmbedding, embed_merchants
from ips.utils.config import ProjectConfig, load_config
from ips.utils.io import artifacts_dir, raw_dir
from ips.utils.logging import get_logger

BASELINE, GRAPH, HYBRID, LEIDEN = "baseline", "graph", "hybrid", "leiden"
ARTIFACTS = {
    "embeddings": Path("embeddings") / "merchant_embeddings.parquet",
    "segments": Path("segments") / "merchant_segments.parquet",
    "comparison": Path("segments") / "segmentation_comparison.parquet",
    "profiles": Path("segments") / "segment_profiles.parquet",
}


@dataclass(frozen=True)
class SegmentationRun:
    """What one run produces: the population, the three methods, the comparison and profiles."""

    population: pl.DataFrame
    embedding: MerchantEmbedding
    segmentations: tuple[Segmentation, ...]
    comparison: pl.DataFrame
    profiles: pl.DataFrame
    graph_stats: dict[str, float]

    def method(self, name: str) -> Segmentation:
        """The segmentation produced by one method."""
        for segmentation in self.segmentations:
            if segmentation.name == name:
                return segmentation
        raise KeyError(f"No segmentation named {name!r}")

    @property
    def verdict(self) -> str:
        """The sentence that closes the comparison: graph against baseline."""
        return verdict(self.comparison, BASELINE, GRAPH)

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
    """Segment the merchants three ways over the same population and compare the results."""
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

    # El híbrido responde la pregunta que sigue: ¿el grafo agrega algo sobre lo que el banco
    # ya sabe del comercio, o repite esa información?
    vectors = embedding.vectors[rows]
    tabular, _ = feature_matrix(population, cfg)
    segmentations = (
        fit_baseline(population, cfg),
        cluster_merchants(GRAPH, merchant_ids, vectors, cfg),
        cluster_merchants(HYBRID, merchant_ids, combine_spaces(tabular, vectors), cfg),
        leiden_communities(project_merchants(edges, cfg), merchant_ids, cfg),
    )
    return SegmentationRun(
        population=population,
        embedding=embedding,
        segmentations=segmentations,
        comparison=compare_segmentations(segmentations, population, cfg, latent),
        profiles=profile_segments(segmentations[1], population),  # the graph segmentation
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
    """Write embeddings, segments, comparison and profiles under ``data/artifacts``."""
    base = artifacts_dir(cfg)
    paths = {name: base / relative for name, relative in ARTIFACTS.items()}
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    run.embedding.write_parquet(paths["embeddings"])
    run.segments_frame().write_parquet(paths["segments"])
    run.comparison.write_parquet(paths["comparison"])
    run.profiles.write_parquet(paths["profiles"])
    return paths
