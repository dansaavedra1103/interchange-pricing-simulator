"""Business profile of each segment: the table a pricing memo can carry.

A segment is only useful if someone can say what it is. Each row answers that: how big it is,
what it sells, where, what it pays and how heavy that payment is against its margin.
"""

from __future__ import annotations

import polars as pl

from ips.graph.clustering import UNASSIGNED, Segmentation

TOP_VALUES = 3


def _top_by_volume(frame: pl.DataFrame, column: str, top_n: int) -> pl.DataFrame:
    """The ``top_n`` values of ``column`` with the most volume in each segment."""
    return (
        frame.group_by("segment", column)
        .agg(pl.col("gdv_cop").sum())
        .sort("gdv_cop", descending=True)
        .group_by("segment", maintain_order=True)
        .agg(pl.col(column).cast(pl.Utf8).head(top_n).str.join(", ").alias(f"top_{column}"))
    )


def profile_segments(
    segmentation: Segmentation, profile: pl.DataFrame, top_n: int = TOP_VALUES
) -> pl.DataFrame:
    """One row per segment: size, volume, what it sells and what it pays."""
    joined = profile.join(segmentation.labels, on="merchant_id", how="inner").filter(
        pl.col("segment") != UNASSIGNED
    )
    total_volume = joined["gdv_cop"].sum()
    segments = (
        joined.group_by("segment")
        .agg(
            pl.len().cast(pl.Int64).alias("merchants"),
            (pl.col("gdv_cop").sum() / total_volume).alias("volume_share"),
            (pl.col("gdv_cop").sum() / pl.col("n_txns").sum()).alias("avg_ticket_cop"),
            (pl.col("interchange_cop").sum() / pl.col("gdv_cop").sum()).alias(
                "effective_interchange_rate"
            ),
            (pl.col("mdr_cop").sum() / pl.col("gdv_cop").sum()).alias("effective_mdr_rate"),
            pl.col("relative_burden").median().alias("median_relative_burden"),
            pl.col("relative_burden").quantile(0.9).alias("p90_relative_burden"),
            (pl.col("cnp_share") * pl.col("gdv_cop")).sum().alias("_cnp_volume"),
            (pl.col("cross_border_share") * pl.col("gdv_cop")).sum().alias("_cross_border_volume"),
            pl.col("gdv_cop").sum().alias("_volume"),
        )
        .with_columns(
            (pl.col("_cnp_volume") / pl.col("_volume")).alias("cnp_share"),
            (pl.col("_cross_border_volume") / pl.col("_volume")).alias("cross_border_share"),
        )
        .drop("_cnp_volume", "_cross_border_volume", "_volume")
    )
    for column in ("mcc_group", "city", "size_tier"):
        segments = segments.join(_top_by_volume(joined, column, top_n), on="segment", how="left")
    return segments.sort("volume_share", descending=True)
