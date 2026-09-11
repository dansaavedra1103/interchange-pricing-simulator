"""Business metrics on top of the P&L: yield, effective interchange, contribution and burden.

Each function takes a ``PnLResult`` and returns a frame by any segment (``by``), so the same
definitions feed the notebooks and, later, the simulator's reports.
"""

from __future__ import annotations

from collections.abc import Sequence

import polars as pl

from ips.economics.pnl import NETWORK_DRIVER_COLUMNS, PnLResult, pnl_by

_BPS = 10_000
_ISSUER_LINES = (
    "interchange_cop",
    "issuer_network_fee_cop",
    "issuer_gross_cop",
    "rewards_cop",
    "funding_cop",
    "credit_loss_cop",
    "issuer_processing_cop",
    "fraud_loss_cop",
    "issuer_contribution_cop",
)


def _names(by: Sequence[str | pl.Expr]) -> list[str]:
    return [key if isinstance(key, str) else key.meta.output_name() for key in by]


def net_revenue_yield(result: PnLResult, by: Sequence[str | pl.Expr] = ()) -> pl.DataFrame:
    """Network revenue over volume in bps, with its drivers (assessment, authorization,
    cross-border)."""
    return pnl_by(result, by).select(
        *_names(by),
        "n_txns",
        "gdv_cop",
        "network_revenue_cop",
        *NETWORK_DRIVER_COLUMNS,
        (pl.col("net_revenue_yield") * _BPS).alias("net_revenue_yield_bps"),
    )


def effective_interchange(result: PnLResult, by: Sequence[str | pl.Expr] = ()) -> pl.DataFrame:
    """Interchange over volume: what really reaches the issuer once the fixed component and
    the small-ticket tier apply."""
    return pnl_by(result, by).select(
        *_names(by), "n_txns", "gdv_cop", "interchange_cop", "effective_interchange_rate"
    )


def issuer_contribution(
    result: PnLResult, by: Sequence[str | pl.Expr] = ("product",)
) -> pl.DataFrame:
    """Issuer P&L lines and contribution margin over volume (default: by card product)."""
    return pnl_by(result, by).select(
        *_names(by), "n_txns", "gdv_cop", *_ISSUER_LINES, "issuer_contribution_rate"
    )


def relative_burden(result: PnLResult, merchants: pl.DataFrame) -> pl.DataFrame:
    """Per merchant with purchases: effective MDR over its sector margin (spec §2.4).

    La carga relativa es el principal predictor de abandono de la aceptación: qué fracción
    del margen del comercio se va en MDR.
    """
    volume = result.transactions.group_by("merchant_id").agg(
        pl.len().cast(pl.Int64).alias("n_txns"),
        pl.col("amount_cop").sum().cast(pl.Float64).alias("gdv_cop"),
        pl.col("mdr_cop").sum(),
    )
    attributes = merchants.select(
        "merchant_id", "mcc_group", "size_tier", "pricing_model", "sector_margin"
    )
    return (
        volume.join(attributes, on="merchant_id", how="inner")
        .with_columns((pl.col("mdr_cop") / pl.col("gdv_cop")).alias("effective_mdr_rate"))
        .with_columns(
            (pl.col("effective_mdr_rate") / pl.col("sector_margin")).alias("relative_burden")
        )
        .sort("merchant_id")
    )
