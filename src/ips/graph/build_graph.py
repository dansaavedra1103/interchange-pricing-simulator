"""The bipartite cardholder-merchant graph the segmentation learns from.

An edge is a pair that transacted in the window, weighted by purchases. Two structures come
out of it:

* the sparse cardholder x merchant matrix that feeds the embeddings;
* the merchant-merchant projection that feeds Leiden, built only from each card's habitual
  merchants — without that cap the projection has ~10^8 pairs and says little, because a card
  that buys everywhere links everything to everything.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import polars as pl

from ips.utils.config import ProjectConfig, load_config
from ips.utils.io import read_table

if TYPE_CHECKING:  # pragma: no cover - typing only, scipy is an optional extra
    from scipy.sparse import csr_matrix

EDGE_COLUMNS = ("cardholder_id", "merchant_id", "n_txns", "amount_cop")


@dataclass(frozen=True)
class BipartiteGraph:
    """Sparse cardholder x merchant matrix and the ids behind its rows and columns."""

    matrix: csr_matrix
    cardholder_ids: np.ndarray
    merchant_ids: np.ndarray

    @property
    def shape(self) -> tuple[int, int]:
        """(cardholders, merchants)."""
        return self.matrix.shape

    def merchant_purchases(self) -> np.ndarray:
        """Purchases of each merchant inside the window (column sums)."""
        return np.asarray(self.matrix.sum(axis=0)).ravel()

    def merchant_customers(self) -> np.ndarray:
        """Distinct cardholders of each merchant (column non-zeros)."""
        return np.bincount(self.matrix.indices, minlength=self.shape[1])

    def cardholder_merchants(self) -> np.ndarray:
        """Distinct merchants of each cardholder (row non-zeros)."""
        return np.diff(self.matrix.indptr)

    def stats(self) -> dict[str, float]:
        """Size and density of the graph, for the notebook and the logs."""
        cardholders, merchants = self.shape
        return {
            "cardholders": float(cardholders),
            "merchants": float(merchants),
            "edges": float(self.matrix.nnz),
            "purchases": float(self.matrix.sum()),
            "density": float(self.matrix.nnz / (cardholders * merchants)),
            "mean_merchants_per_cardholder": float(self.cardholder_merchants().mean()),
            "median_customers_per_merchant": float(np.median(self.merchant_customers())),
        }


def edges_from_transactions(transactions: pl.DataFrame) -> pl.DataFrame:
    """Monthly edges from transactions: the Python mirror of ``mart_graph_edges``."""
    return transactions.group_by(
        pl.col("txn_date").dt.truncate("1mo").alias("txn_month"), "cardholder_id", "merchant_id"
    ).agg(
        pl.len().cast(pl.Int64).alias("n_txns"),
        pl.col("amount_cop").sum().cast(pl.Float64).alias("amount_cop"),
    )


def window_edges(edges: pl.DataFrame, months: int) -> pl.DataFrame:
    """Keep the last ``months`` of monthly edges, counted from the latest month with purchases."""
    last_month = edges["txn_month"].max()
    first_month = pl.lit(last_month).dt.offset_by(f"-{months - 1}mo")
    return edges.filter(pl.col("txn_month") >= first_month)


def load_edges(
    months: int | None = None, cfg: ProjectConfig | None = None, path: Path | None = None
) -> pl.DataFrame:
    """Monthly edges from ``mart_graph_edges``, restricted to the window."""
    config = cfg if cfg is not None else load_config()
    window = months if months is not None else config.segmentation.window_months
    return window_edges(read_table("mart_graph_edges", path), window)


def aggregate_edges(edges: pl.DataFrame) -> pl.DataFrame:
    """Collapse the monthly edges into one row per (cardholder, merchant)."""
    return edges.group_by("cardholder_id", "merchant_id").agg(
        pl.col("n_txns").sum().cast(pl.Int64), pl.col("amount_cop").sum().cast(pl.Float64)
    )


def build_bipartite(edges: pl.DataFrame, cfg: ProjectConfig) -> BipartiteGraph:
    """Build the sparse graph, dropping cards that buy almost everywhere.

    Una tarjeta con cientos de comercios distintos no señala clientela: conecta todo con todo
    y solo agrega ruido al embedding.
    """
    from scipy.sparse import csr_matrix

    if edges.is_empty():
        raise ValueError("No edges in the window")
    pairs = aggregate_edges(edges)
    degree = pairs.group_by("cardholder_id").len()
    keep = degree.filter(pl.col("len") <= cfg.segmentation.max_cardholder_degree)
    pairs = pairs.join(keep.select("cardholder_id"), on="cardholder_id", how="inner")
    if pairs.is_empty():
        raise ValueError("Every cardholder was dropped by max_cardholder_degree")

    cardholders = pairs.select("cardholder_id").unique().sort("cardholder_id")
    merchants = pairs.select("merchant_id").unique().sort("merchant_id")
    indexed = pairs.join(cardholders.with_row_index("row"), on="cardholder_id").join(
        merchants.with_row_index("column"), on="merchant_id"
    )
    matrix = csr_matrix(
        (
            indexed["n_txns"].to_numpy().astype(np.float64),
            (indexed["row"].to_numpy(), indexed["column"].to_numpy()),
        ),
        shape=(cardholders.height, merchants.height),
    )
    return BipartiteGraph(
        matrix=matrix,
        cardholder_ids=cardholders["cardholder_id"].to_numpy(),
        merchant_ids=merchants["merchant_id"].to_numpy(),
    )


def habitual_edges(edges: pl.DataFrame, cfg: ProjectConfig) -> pl.DataFrame:
    """Each cardholder's most-visited merchants, the ones that carry its clientele.

    Los empates se rompen por ``merchant_id``: sin ese desempate el resultado dependería del
    orden de las filas, que no está garantizado, y la misma semilla daría comunidades distintas.
    """
    pairs = aggregate_edges(edges).sort(
        ["cardholder_id", "n_txns", "merchant_id"], descending=[False, True, False]
    )
    position = pl.int_range(pl.len()).over("cardholder_id")
    return pairs.filter(position < cfg.segmentation.projection_top_merchants)


def project_merchants(edges: pl.DataFrame, cfg: ProjectConfig) -> pl.DataFrame:
    """Merchant-merchant graph: weight = cards that have both among their habitual merchants.

    Returns (merchant_a, merchant_b, weight) with a < b, pruned by ``projection_min_weight``
    and to ``projection_max_neighbors`` per merchant, so Leiden sees the strong ties only.
    """
    habitual = habitual_edges(edges, cfg).select("cardholder_id", "merchant_id")
    pairs = (
        habitual.join(habitual, on="cardholder_id", suffix="_b")
        .filter(pl.col("merchant_id") < pl.col("merchant_id_b"))
        .group_by(
            pl.col("merchant_id").alias("merchant_a"), pl.col("merchant_id_b").alias("merchant_b")
        )
        .agg(pl.len().cast(pl.Int64).alias("weight"))
        .filter(pl.col("weight") >= cfg.segmentation.projection_min_weight)
    )
    # Poda por vecino: cada comercio conserva sus aristas más pesadas, en ambos sentidos. El
    # desempate por el otro comercio mantiene el resultado igual entre corridas.
    top = cfg.segmentation.projection_max_neighbors
    ranked = (
        pairs.sort(["merchant_a", "weight", "merchant_b"], descending=[False, True, False])
        .with_columns(pl.int_range(pl.len()).over("merchant_a").alias("rank_a"))
        .sort(["merchant_b", "weight", "merchant_a"], descending=[False, True, False])
        .with_columns(pl.int_range(pl.len()).over("merchant_b").alias("rank_b"))
    )
    return (
        ranked.filter((pl.col("rank_a") < top) | (pl.col("rank_b") < top))
        .drop("rank_a", "rank_b")
        .sort("merchant_a", "merchant_b")
    )
