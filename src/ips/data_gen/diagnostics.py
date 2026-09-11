"""Diagnostics of the generated data: concentration and community structure.

Used by the tests and the validation notebook. Modularity is computed with polars on the
weighted cardholder-merchant edge list, NMI with plain counts, and community detection uses
igraph's Leiden (imported lazily: it is only a development dependency).
"""

from __future__ import annotations

import math
import random

import numpy as np
import polars as pl


def top_share(values: pl.Series | np.ndarray, fraction: float) -> float:
    """Share of the total held by the top ``fraction`` of entities (e.g. top 10% merchants)."""
    array = np.sort(np.asarray(values, dtype=np.float64))[::-1]
    k = max(1, math.ceil(fraction * len(array)))
    return float(array[:k].sum() / array.sum())


def edge_list(transactions: pl.DataFrame) -> pl.DataFrame:
    """Weighted bipartite edges: one row per (cardholder, merchant) with its purchase count."""
    return transactions.group_by("cardholder_id", "merchant_id").agg(
        pl.len().cast(pl.Float64).alias("weight")
    )


def bipartite_modularity(
    edges: pl.DataFrame, cardholder_labels: pl.DataFrame, merchant_labels: pl.DataFrame
) -> float:
    """Newman modularity of a partition of the cardholder-merchant graph.

    ``cardholder_labels`` has columns (cardholder_id, community) and ``merchant_labels``
    (merchant_id, community); an edge is internal when both ends share a community.
    Q = sum_c [ L_c / m - (D_c / 2m)^2 ], with m the total edge weight.
    """
    labelled = edges.join(
        cardholder_labels.rename({"community": "c_ch"}), on="cardholder_id", how="inner"
    ).join(merchant_labels.rename({"community": "c_m"}), on="merchant_id", how="inner")
    if labelled.height != edges.height:
        raise ValueError("Every edge endpoint needs a community label")
    m = labelled["weight"].sum()
    internal = labelled.filter(pl.col("c_ch") == pl.col("c_m"))["weight"].sum()
    degree = pl.concat(
        [
            labelled.select(pl.col("c_ch").alias("community"), "weight"),
            labelled.select(pl.col("c_m").alias("community"), "weight"),
        ]
    )
    degree_by_community = degree.group_by("community").agg(pl.col("weight").sum())["weight"]
    return float(internal / m - ((degree_by_community / (2 * m)) ** 2).sum())


def shuffled_modularity(
    edges: pl.DataFrame,
    cardholder_labels: pl.DataFrame,
    merchant_labels: pl.DataFrame,
    rng: np.random.Generator,
    n_shuffles: int = 20,
) -> np.ndarray:
    """Modularity of the same labels randomly permuted among nodes (the null model)."""
    values = []
    for _ in range(n_shuffles):
        ch = cardholder_labels.with_columns(
            pl.Series("community", rng.permutation(cardholder_labels["community"].to_numpy()))
        )
        mer = merchant_labels.with_columns(
            pl.Series("community", rng.permutation(merchant_labels["community"].to_numpy()))
        )
        values.append(bipartite_modularity(edges, ch, mer))
    return np.array(values)


def normalized_mutual_info(a: pl.Series | np.ndarray, b: pl.Series | np.ndarray) -> float:
    """Normalised mutual information between two labellings (arithmetic normalisation)."""
    counts = (
        pl.DataFrame({"a": pl.Series(a).cast(pl.Utf8), "b": pl.Series(b).cast(pl.Utf8)})
        .group_by("a", "b")
        .len()
    )
    n = counts["len"].sum()
    p_ab = counts["len"].to_numpy() / n
    p_a = counts.group_by("a").agg(pl.col("len").sum())["len"].to_numpy() / n
    p_b = counts.group_by("b").agg(pl.col("len").sum())["len"].to_numpy() / n
    marginal_a = counts.join(counts.group_by("a").agg(pl.col("len").sum().alias("na")), on="a")
    marginal = marginal_a.join(counts.group_by("b").agg(pl.col("len").sum().alias("nb")), on="b")
    expected = marginal["na"].to_numpy() * marginal["nb"].to_numpy() / n**2
    mutual = float(np.sum(p_ab * np.log(p_ab / expected)))
    entropy_a = float(-np.sum(p_a * np.log(p_a)))
    entropy_b = float(-np.sum(p_b * np.log(p_b)))
    if entropy_a + entropy_b == 0:
        return 1.0
    return 2 * mutual / (entropy_a + entropy_b)


def leiden_partition(
    edges: pl.DataFrame, seed: int, resolution: float = 1.0
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Leiden communities (modularity objective) of the bipartite graph, via igraph.

    Returns (cardholder_id, community) and (merchant_id, community) frames.
    """
    import igraph as ig

    cardholders = edges.select("cardholder_id").unique().sort("cardholder_id")
    merchants = edges.select("merchant_id").unique().sort("merchant_id")
    n_ch = cardholders.height
    indexed = edges.join(cardholders.with_row_index("src"), on="cardholder_id").join(
        merchants.with_row_index("dst"), on="merchant_id"
    )
    pairs = np.column_stack(
        [indexed["src"].to_numpy(), indexed["dst"].to_numpy().astype(np.int64) + n_ch]
    )
    graph = ig.Graph(n=n_ch + merchants.height, edges=pairs.tolist())
    # igraph draws from Python's random module: use a seeded generator, then restore it.
    ig.set_random_number_generator(random.Random(seed))
    try:
        clustering = graph.community_leiden(
            objective_function="modularity",
            weights=indexed["weight"].to_list(),
            resolution=resolution,
            n_iterations=-1,
        )
    finally:
        ig.set_random_number_generator(random)
    membership = np.asarray(clustering.membership)
    return (
        cardholders.with_columns(pl.Series("community", membership[:n_ch])),
        merchants.with_columns(pl.Series("community", membership[n_ch:])),
    )
