"""Transitional P&L: how the MDR paid by merchants is split, on Feature 1 data.

Every peso of MDR goes to exactly one of three receivers:

* issuer   -> interchange (set by the network, paid by the acquirer to the issuer)
* network  -> scheme fee (the network's actual revenue)
* acquirer -> the residual margin, ``MDR - interchange - scheme fee``

These are gross flows out of the MDR. Every merchant pays its group's blended rate; IC++,
cross-border and authorisation fees and issuer costs arrive with Feature 3, which replaces
this module.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import polars as pl

from ips.data_gen.interchange_table import apply_interchange
from ips.utils.config import PricingConfig

NETWORK_ACTOR_ID = "NETWORK"
MONEY_COLUMNS = ("interchange_cop", "scheme_fee_cop", "acquirer_net_cop", "mdr_cop")
REQUIRED_COLUMNS = (
    "txn_id",
    "issuer_id",
    "acquirer_id",
    "product",
    "mcc_group",
    "channel",
    "cross_border",
    "amount_cop",
)
_BPS = 10_000

# Quién recibe qué parte del MDR y por qué id se agrega: el emisor cobra el interchange,
# el adquirente el residuo y la red el scheme fee (un único actor).
_RECEIVERS: tuple[tuple[str, str | None, str], ...] = (
    ("issuer", "issuer_id", "interchange_cop"),
    ("acquirer", "acquirer_id", "acquirer_net_cop"),
    ("network", None, "scheme_fee_cop"),
)


@dataclass(frozen=True)
class PnLResult:
    """P&L of a set of transactions.

    Attributes:
        transactions: Input transactions plus the four money columns in ``MONEY_COLUMNS``.
        by_actor: One row per receiving actor with columns ``actor_type``, ``actor_id``,
            ``revenue_cop``, ``gdv_cop`` and ``n_txns``.
    """

    transactions: pl.DataFrame
    by_actor: pl.DataFrame

    @property
    def network_revenue(self) -> float:
        """Scheme fees collected by the network (COP)."""
        network = self.by_actor.filter(pl.col("actor_type") == "network")
        return float(network["revenue_cop"].sum())

    def totals(self) -> dict[str, float]:
        """Network-wide GDV, each money flow, and the net revenue yield in bps of GDV."""
        totals = self.transactions.select(
            pl.col("amount_cop").sum().cast(pl.Float64).alias("gdv_cop"),
            *(pl.col(column).sum() for column in MONEY_COLUMNS),
        ).row(0, named=True)
        totals["net_revenue_yield_bps"] = totals["scheme_fee_cop"] / totals["gdv_cop"] * _BPS
        return totals


def compute_pnl(
    transactions: pl.DataFrame, table: pl.DataFrame, pricing: PricingConfig
) -> PnLResult:
    """Price every transaction and split its MDR among issuer, network and acquirer.

    Args:
        transactions: Transactions with at least ``REQUIRED_COLUMNS``. Existing money
            columns are dropped and recomputed, so an already-priced frame can be repriced.
        table: Expanded interchange table (``build_interchange_table``). Separate from
            ``pricing`` because it is the lever that scenarios and the optimizer change.
        pricing: Scheme fee and blended MDR rate per MCC group.

    Raises:
        ValueError: If required columns are missing, the table has duplicate or missing
            cells, or a group has no MDR rate (which would silently leak P&L).
    """
    missing_columns = [c for c in REQUIRED_COLUMNS if c not in transactions.columns]
    if missing_columns:
        raise ValueError(f"transactions is missing required columns: {missing_columns}")

    priced = apply_interchange(transactions.drop(MONEY_COLUMNS, strict=False), table)
    mdr_rate = (
        pl.col("mcc_group")
        .cast(pl.Utf8)
        .replace_strict(pricing.blended_mdr_rate, default=None, return_dtype=pl.Float64)
    )
    priced = priced.with_columns(mdr_rate.alias("_mdr_rate"))
    unpriced = priced.filter(pl.col("_mdr_rate").is_null())
    if unpriced.height:
        groups = sorted(set(unpriced["mcc_group"].cast(pl.Utf8).to_list()))
        raise ValueError(f"No blended MDR rate for groups: {groups}")

    amount = pl.col("amount_cop").cast(pl.Float64)
    scheme_fee = pricing.scheme_fee
    priced = (
        priced.with_columns(
            # Scheme fee: lo que la red cobra al adquirente; es su ingreso real.
            (amount * scheme_fee.rate + scheme_fee.fixed_cop).alias("scheme_fee_cop"),
            # MDR blended: el comercio paga la tasa de su grupo sin importar la tarjeta.
            (amount * pl.col("_mdr_rate")).alias("mdr_cop"),
        )
        .with_columns(
            # El adquirente se queda con el residuo, que puede ser negativo en tarjetas premium
            # cuando interchange + scheme fee superan la tarifa blended.
            (pl.col("mdr_cop") - pl.col("interchange_cop") - pl.col("scheme_fee_cop")).alias(
                "acquirer_net_cop"
            )
        )
        .drop("_mdr_rate")
    )
    return PnLResult(transactions=priced, by_actor=_by_actor(priced))


def _by_actor(priced: pl.DataFrame) -> pl.DataFrame:
    frames = []
    for actor_type, id_column, revenue_column in _RECEIVERS:
        aggregations = [
            pl.col(revenue_column).sum().alias("revenue_cop"),
            pl.col("amount_cop").sum().cast(pl.Float64).alias("gdv_cop"),
            pl.len().cast(pl.Int64).alias("n_txns"),
        ]
        if id_column is None:
            frame = priced.select(pl.lit(NETWORK_ACTOR_ID).alias("actor_id"), *aggregations)
        else:
            frame = (
                priced.group_by(pl.col(id_column).cast(pl.Utf8).alias("actor_id"))
                .agg(aggregations)
                .sort("actor_id")
            )
        frames.append(
            frame.select(
                pl.lit(actor_type).alias("actor_type"),
                "actor_id",
                "revenue_cop",
                "gdv_cop",
                "n_txns",
            )
        )
    return pl.concat(frames)


def pnl_by(result: PnLResult, keys: Sequence[str | pl.Expr]) -> pl.DataFrame:
    """Aggregate volume and money flows by any segment; empty ``keys`` gives the total.

    Adds effective rates over GDV for interchange, MDR and the acquirer margin.
    """
    aggregations = [
        pl.len().cast(pl.Int64).alias("n_txns"),
        pl.col("amount_cop").sum().cast(pl.Float64).alias("gdv_cop"),
        *(pl.col(column).sum() for column in MONEY_COLUMNS),
    ]
    if keys:
        names = [key if isinstance(key, str) else key.meta.output_name() for key in keys]
        frame = result.transactions.group_by(list(keys)).agg(aggregations).sort(names)
    else:
        frame = result.transactions.select(aggregations)
    gdv = pl.col("gdv_cop")
    return frame.with_columns(
        (pl.col("interchange_cop") / gdv).alias("effective_interchange_rate"),
        (pl.col("mdr_cop") / gdv).alias("effective_mdr_rate"),
        (pl.col("acquirer_net_cop") / gdv).alias("acquirer_margin_rate"),
    )
