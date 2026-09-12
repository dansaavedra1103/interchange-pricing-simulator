"""Leiden communities on the merchant projection: the third segmentation of the comparison.

Unlike k-means, Leiden is not told how many segments to find; the count it returns is part of
what the comparison reports. Merchants left without a strong tie keep ``UNASSIGNED``.
"""

from __future__ import annotations

import random

import numpy as np
import polars as pl

from ips.graph.clustering import UNASSIGNED, Segmentation
from ips.utils.config import ProjectConfig


def leiden_communities(
    projection: pl.DataFrame, merchant_ids: np.ndarray, cfg: ProjectConfig
) -> Segmentation:
    """Run Leiden over the pruned merchant graph, restricted to the merchants being segmented."""
    import igraph as ig

    nodes = pl.DataFrame({"merchant_id": pl.Series(merchant_ids, dtype=pl.UInt32)}).sort(
        "merchant_id"
    )
    indexed = projection.join(
        nodes.with_row_index("node_a"), left_on="merchant_a", right_on="merchant_id", how="inner"
    ).join(
        nodes.with_row_index("node_b"), left_on="merchant_b", right_on="merchant_id", how="inner"
    )
    graph = ig.Graph(
        n=nodes.height,
        edges=list(zip(indexed["node_a"].to_list(), indexed["node_b"].to_list(), strict=True)),
    )
    # igraph toma números del módulo random: se siembra y se restaura, como en diagnostics.
    ig.set_random_number_generator(random.Random(cfg.seed))
    try:
        clustering = graph.community_leiden(
            objective_function="modularity",
            weights=indexed["weight"].to_list(),
            resolution=cfg.segmentation.leiden_resolution,
            n_iterations=-1,
        )
    finally:
        ig.set_random_number_generator(random)

    membership = np.asarray(clustering.membership)
    # Un comercio aislado forma una comunidad de uno: eso no es un segmento, es falta de señal.
    isolated = np.asarray(graph.degree()) == 0
    segments = np.where(isolated, UNASSIGNED, membership)
    assigned = np.unique(segments[segments != UNASSIGNED])
    relabel = {int(old): index for index, old in enumerate(assigned)}
    segments = np.array([relabel.get(int(s), UNASSIGNED) for s in segments], dtype=np.int64)
    labels = pl.DataFrame(
        {
            "merchant_id": nodes["merchant_id"],
            "segment": pl.Series(segments, dtype=pl.Int32),
        }
    )
    return Segmentation(
        name="leiden",
        labels=labels,
        k=int(len(assigned)),
        silhouette=float("nan"),
        scores=pl.DataFrame(schema={"k": pl.Int64, "score": pl.Float64}),
    )
