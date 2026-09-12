"""Merchant profile: what every segmentation is built on and judged by.

One definition, two paths into it. The production run projects the columns it needs out of
``fct_transactions``; the tests build the same frame from ``compute_pnl`` on the sample. The
profile carries both the attributes a bank already has about a merchant and the two business
targets of the comparison: effective interchange and relative burden.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from ips.utils.config import ProjectConfig
from ips.utils.io import connect, read_table

FCT_COLUMNS = (
    "merchant_id",
    "cardholder_id",
    "product",
    "channel",
    "cross_border",
    "amount_cop",
    "interchange_cop",
    "mdr_cop",
)
MERCHANT_COLUMNS = (
    "merchant_id",
    "mcc",
    "mcc_group",
    "country",
    "city",
    "channel",
    "size_tier",
    "size_weight",
    "acquirer_id",
    "pricing_model",
    "sector_margin",
)
TARGETS = ("effective_interchange_rate", "relative_burden")


def load_priced_transactions(path: Path | None = None) -> pl.DataFrame:
    """The columns of ``fct_transactions`` the profile needs: a projection, no logic in SQL."""
    columns = ", ".join(FCT_COLUMNS)
    with connect(path) as connection:
        return connection.sql(f"select {columns} from fct_transactions").pl()


def load_merchants(path: Path | None = None) -> pl.DataFrame:
    """Merchant dimension from the warehouse."""
    return read_table("dim_merchant", path).select(MERCHANT_COLUMNS)


def merchant_profile(
    transactions: pl.DataFrame, merchants: pl.DataFrame, cfg: ProjectConfig
) -> pl.DataFrame:
    """One row per merchant with purchases: volume, customer mix, effective rates and burden."""
    missing = [c for c in FCT_COLUMNS if c not in transactions.columns]
    if missing:
        raise ValueError(f"transactions is missing required columns: {missing}")
    attributes = merchants.select(
        *(c for c in MERCHANT_COLUMNS if c != "channel"),
        pl.col("channel").alias("merchant_channel"),
    )
    volume = transactions.group_by("merchant_id").agg(
        pl.len().cast(pl.Int64).alias("n_txns"),
        pl.col("cardholder_id").n_unique().cast(pl.Int64).alias("n_customers"),
        pl.col("amount_cop").sum().cast(pl.Float64).alias("gdv_cop"),
        pl.col("interchange_cop").sum().alias("interchange_cop"),
        pl.col("mdr_cop").sum().alias("mdr_cop"),
        (pl.col("channel").cast(pl.Utf8) == "cnp").mean().alias("cnp_share"),
        pl.col("cross_border").mean().alias("cross_border_share"),
        *(
            (pl.col("product").cast(pl.Utf8) == product).mean().alias(f"share_{product}")
            for product in cfg.products
        ),
    )
    profiled = volume.join(attributes, on="merchant_id", how="inner", validate="1:1")
    if profiled.height != volume.height:
        missing = volume.height - profiled.height
        raise ValueError(f"{missing} merchants with purchases are missing from the dimension")
    return (
        profiled.with_columns(
            (pl.col("gdv_cop") / pl.col("n_txns")).alias("avg_ticket_cop"),
            (pl.col("interchange_cop") / pl.col("gdv_cop")).alias("effective_interchange_rate"),
            (pl.col("mdr_cop") / pl.col("gdv_cop")).alias("effective_mdr_rate"),
        )
        .with_columns(
            (pl.col("effective_mdr_rate") / pl.col("sector_margin")).alias("relative_burden")
        )
        .sort("merchant_id")
    )


def clustered_population(profile: pl.DataFrame, cfg: ProjectConfig) -> pl.DataFrame:
    """Merchants with enough purchases to segment; below that a vector is mostly noise."""
    return profile.filter(pl.col("n_txns") >= cfg.segmentation.min_purchases)
