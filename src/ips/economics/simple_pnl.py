"""Minimal P&L for the vertical slice: how the MDR paid by merchants is split.

Every peso of MDR goes to exactly one of three receivers:

* issuer   -> interchange (set by the network, paid by the acquirer to the issuer)
* network  -> scheme fee (the network's actual revenue)
* acquirer -> the residual margin, ``MDR - interchange - scheme fee``

These are gross flows out of the MDR. Issuer costs (rewards, fraud, funding) and
issuer-side assessments are out of scope until Feature 3.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import polars as pl

from ips.utils.config import PricingConfig

NETWORK_ACTOR_ID = "NETWORK"
MONEY_COLUMNS = ("interchange_cop", "scheme_fee_cop", "acquirer_net_cop", "mdr_cop")
REQUIRED_COLUMNS = ("txn_id", "issuer_id", "acquirer_id", "product", "mcc", "amount_cop")
TABLE_KEYS = ("product", "mcc")
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
    """P&L of the vertical slice.

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


def interchange_table_frame(pricing: PricingConfig) -> pl.DataFrame:
    """Flatten the configured interchange table into (product, mcc, ic_rate, ic_fixed_cop)."""
    return pl.DataFrame(
        [
            (product, mcc, fee.rate, fee.fixed_cop)
            for product, cells in pricing.interchange_table.items()
            for mcc, fee in cells.items()
        ],
        schema={
            "product": pl.Utf8,
            "mcc": pl.Utf8,
            "ic_rate": pl.Float64,
            "ic_fixed_cop": pl.Float64,
        },
        orient="row",
    )


def compute_pnl(
    transactions: pl.DataFrame, table: pl.DataFrame, pricing: PricingConfig
) -> PnLResult:
    """Price every transaction and split its MDR among issuer, network and acquirer.

    Args:
        transactions: Raw transactions with at least ``REQUIRED_COLUMNS``. Existing money
            columns are dropped and recomputed, so an already-priced frame can be repriced.
        table: Interchange table with one row per (product, mcc) and columns ``ic_rate``
            and ``ic_fixed_cop``. Separate from ``pricing`` because it is the lever that
            scenarios and the optimizer change.
        pricing: Scheme fee and blended MDR rate per MCC.

    Returns:
        The priced transactions and the revenue of every receiving actor.

    Raises:
        ValueError: If required columns are missing, the table has duplicate cells, or any
            transaction has no interchange or MDR rate (which would silently leak P&L).
    """
    missing_columns = [c for c in REQUIRED_COLUMNS if c not in transactions.columns]
    if missing_columns:
        raise ValueError(f"transactions is missing required columns: {missing_columns}")

    keys = list(TABLE_KEYS)
    duplicated = table.filter(table.select(keys).is_duplicated())
    if duplicated.height:
        cells = sorted(set(duplicated.select(keys).iter_rows()))
        raise ValueError(f"Interchange table has duplicate cells (product, mcc): {cells}")

    mdr_rates = pl.DataFrame(
        {
            "mcc": list(pricing.blended_mdr_rate),
            "mdr_rate": list(pricing.blended_mdr_rate.values()),
        },
        schema={"mcc": pl.Utf8, "mdr_rate": pl.Float64},
    )
    priced = (
        transactions.drop(MONEY_COLUMNS, strict=False)
        .join(
            table.select(*keys, "ic_rate", "ic_fixed_cop"),
            on=keys,
            how="left",
            validate="m:1",
            maintain_order="left",
        )
        .join(mdr_rates, on="mcc", how="left", validate="m:1", maintain_order="left")
    )

    unpriced = priced.filter(
        pl.any_horizontal(pl.col("ic_rate", "ic_fixed_cop", "mdr_rate").is_null())
    )
    if unpriced.height:
        cells = sorted(set(unpriced.select(keys).iter_rows()))
        raise ValueError(
            f"{unpriced.height} transactions have no interchange or MDR rate; "
            f"missing cells (product, mcc): {cells}"
        )

    amount = pl.col("amount_cop").cast(pl.Float64)
    scheme_fee = pricing.scheme_fee
    priced = (
        priced.with_columns(
            # Interchange: porcentual + fijo según la celda producto × MCC; lo paga el
            # adquirente al emisor. La red lo fija pero no se lo queda.
            (amount * pl.col("ic_rate") + pl.col("ic_fixed_cop")).alias("interchange_cop"),
            # Scheme fee: lo que la red cobra al adquirente; es su ingreso real.
            (amount * scheme_fee.rate + scheme_fee.fixed_cop).alias("scheme_fee_cop"),
            # MDR blended: el comercio paga la misma tasa sin importar qué tarjeta le presentan.
            (amount * pl.col("mdr_rate")).alias("mdr_cop"),
        )
        .with_columns(
            # El adquirente se queda con el residuo, que puede ser negativo en tarjetas premium
            # cuando interchange + scheme fee superan la tarifa blended.
            (pl.col("mdr_cop") - pl.col("interchange_cop") - pl.col("scheme_fee_cop")).alias(
                "acquirer_net_cop"
            )
        )
        .drop("ic_rate", "ic_fixed_cop", "mdr_rate")
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
                priced.group_by(pl.col(id_column).alias("actor_id"))
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
