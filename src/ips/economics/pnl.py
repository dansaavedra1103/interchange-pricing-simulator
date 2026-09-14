"""Unit economics of every actor in a card purchase.

Two layers:

* **Fee waterfall** (the conservation identity). Out of the MDR the merchant pays, the issuer
  keeps the interchange net of its network fees, the network collects the fees of both sides
  and the acquirer keeps the rest: issuer_gross + network_revenue + acquirer_gross = MDR.
* **Operating costs**, outside the identity: rewards, net fraud, funding, expected loss and
  processing for the issuer; processing for the acquirer; chargebacks for the merchant.

The issuer's contribution is the economics of payments, not of lending: interest income and
losses on revolving balances are out of scope (docs/assumptions.md).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import polars as pl

from ips.data_gen.interchange_table import apply_interchange
from ips.utils.config import Fee, IssuerProductCosts, NetworkFeesConfig, ProjectConfig

NETWORK_ACTOR_ID = "NETWORK"
WATERFALL_COLUMNS = (
    "interchange_cop",
    "issuer_network_fee_cop",
    "acquirer_network_fee_cop",
    "network_revenue_cop",
    "mdr_cop",
    "issuer_gross_cop",
    "acquirer_gross_cop",
)
NETWORK_DRIVER_COLUMNS = (
    "network_assessment_cop",
    "network_authorization_cop",
    "network_cross_border_cop",
)
COST_COLUMNS = (
    "rewards_cop",
    "funding_cop",
    "credit_loss_cop",
    "issuer_processing_cop",
    "fraud_loss_cop",
    "acquirer_processing_cop",
    "chargeback_cop",
)
RESULT_COLUMNS = ("issuer_contribution_cop", "acquirer_contribution_cop", "merchant_cost_cop")
MONEY_COLUMNS = WATERFALL_COLUMNS + NETWORK_DRIVER_COLUMNS + COST_COLUMNS + RESULT_COLUMNS
REQUIRED_COLUMNS = (
    "txn_id",
    "merchant_id",
    "issuer_id",
    "acquirer_id",
    "product",
    "mcc_group",
    "channel",
    "cross_border",
    "amount_cop",
    "fraud_flag",
    "chargeback_flag",
)
_SIDES = ("issuer", "acquirer")
_COMPONENTS = ("assessment", "authorization", "cross_border")
_SIDE_COLUMNS = tuple(f"{side}_{part}_cop" for side in _SIDES for part in _COMPONENTS)
_AMOUNT = pl.col("amount_cop").cast(pl.Float64)
_BPS = 10_000

# Quién recibe qué parte del MDR (ingreso bruto) y qué le queda tras sus costos operativos.
_RECEIVERS: tuple[tuple[str, str | None, str, str], ...] = (
    ("issuer", "issuer_id", "issuer_gross_cop", "issuer_contribution_cop"),
    ("acquirer", "acquirer_id", "acquirer_gross_cop", "acquirer_contribution_cop"),
    ("network", None, "network_revenue_cop", "network_revenue_cop"),
)


def _by(column: str, mapping: dict[str, float]) -> pl.Expr:
    """Per-category parameter (e.g. a rate by product); unknown categories give null."""
    return (
        pl.col(column).cast(pl.Utf8).replace_strict(mapping, default=None, return_dtype=pl.Float64)
    )


@dataclass(frozen=True)
class NetworkPnL:
    """What the network collects: assessment, authorization and cross-border, on both sides."""

    fees: NetworkFeesConfig

    @classmethod
    def from_config(cls, cfg: ProjectConfig) -> NetworkPnL:
        return cls(cfg.economics.network)

    def components(self) -> list[pl.Expr]:
        """Fee of each side and component, per transaction."""
        expressions = []
        for side, fees in (("issuer", self.fees.issuer), ("acquirer", self.fees.acquirer)):
            expressions += [
                (_AMOUNT * fees.assessment_rate).alias(f"{side}_assessment_cop"),
                pl.lit(float(fees.authorization_fee_cop)).alias(f"{side}_authorization_cop"),
                # Recargo cross-border: solo cuando el comercio está en otro país.
                pl.when(pl.col("cross_border"))
                .then(_AMOUNT * fees.cross_border_rate)
                .otherwise(0.0)
                .alias(f"{side}_cross_border_cop"),
            ]
        return expressions

    def totals(self) -> list[pl.Expr]:
        """Fees by side, network revenue and its drivers, added in the same order as the SQL."""
        side_total = {
            side: pl.col(f"{side}_assessment_cop")
            + pl.col(f"{side}_authorization_cop")
            + pl.col(f"{side}_cross_border_cop")
            for side in _SIDES
        }
        return [
            side_total["issuer"].alias("issuer_network_fee_cop"),
            side_total["acquirer"].alias("acquirer_network_fee_cop"),
            (side_total["issuer"] + side_total["acquirer"]).alias("network_revenue_cop"),
            *(
                (pl.col(f"issuer_{part}_cop") + pl.col(f"acquirer_{part}_cop")).alias(
                    f"network_{part}_cop"
                )
                for part in _COMPONENTS
            ),
        ]


@dataclass(frozen=True)
class MerchantCost:
    """What the merchant pays: the MDR of its pricing model, plus chargebacks."""

    blended_mdr_rate: dict[str, float]
    blended_cross_border_rate: dict[str, float]
    icpp_markup_rate: dict[str, float]
    icpp_markup_fixed_cop: dict[str, float]

    @classmethod
    def from_config(cls, cfg: ProjectConfig) -> MerchantCost:
        terms = cfg.economics.acquirer
        acquirers = [a.acquirer_id for a in cfg.acquirers] + [cfg.cross_border.acquirer.acquirer_id]
        return cls(
            blended_mdr_rate={g: c.blended_mdr_rate for g, c in cfg.mcc_groups.groups.items()},
            blended_cross_border_rate=dict.fromkeys(
                acquirers, terms.blended_cross_border_surcharge
            ),
            icpp_markup_rate=dict.fromkeys(acquirers, terms.icpp_markup.rate),
            icpp_markup_fixed_cop=dict.fromkeys(acquirers, terms.icpp_markup.fixed_cop),
        )

    def mdr(self) -> pl.Expr:
        # Blended: tasa plana del grupo sin importar la tarjeta, más un recargo cuando la tarjeta
        # es extranjera para el comercio. IC++: el comercio paga el interchange y los fees de
        # red del adquirente tal cual, más el margen del adquirente.
        surcharge = (
            pl.when(pl.col("cross_border"))
            .then(_by("acquirer_id", self.blended_cross_border_rate))
            .otherwise(0.0)
        )
        blended = _AMOUNT * (_by("mcc_group", self.blended_mdr_rate) + surcharge)
        markup = _AMOUNT * _by("acquirer_id", self.icpp_markup_rate) + _by(
            "acquirer_id", self.icpp_markup_fixed_cop
        )
        icpp = pl.col("interchange_cop") + pl.col("acquirer_network_fee_cop") + markup
        is_icpp = pl.col("pricing_model").cast(pl.Utf8) == "icpp"
        return pl.when(is_icpp).then(icpp).otherwise(blended).alias("mdr_cop")

    def chargeback(self) -> pl.Expr:
        # Un contracargo revierte la venta: el comercio pierde el monto.
        return (
            pl.when(pl.col("chargeback_flag")).then(_AMOUNT).otherwise(0.0).alias("chargeback_cop")
        )


@dataclass(frozen=True)
class IssuerPnL:
    """Interchange net of network fees, minus the costs of running the card (payments only)."""

    costs: dict[str, IssuerProductCosts]

    @classmethod
    def from_config(cls, cfg: ProjectConfig) -> IssuerPnL:
        return cls(dict(cfg.economics.issuer.products))

    def _rate(self, field: str) -> pl.Expr:
        return _by("product", {product: getattr(c, field) for product, c in self.costs.items()})

    def columns(self) -> list[pl.Expr]:
        return [
            (pl.col("interchange_cop") - pl.col("issuer_network_fee_cop")).alias(
                "issuer_gross_cop"
            ),
            (_AMOUNT * self._rate("rewards_rate")).alias("rewards_cop"),
            (_AMOUNT * self._rate("funding_rate")).alias("funding_cop"),
            (_AMOUNT * self._rate("credit_loss_rate")).alias("credit_loss_cop"),
            (_AMOUNT * self._rate("processing_rate")).alias("issuer_processing_cop"),
            # Fraude neto: sin contracargo la pérdida queda en el emisor; con contracargo pasa
            # al comercio (liability shift).
            pl.when(pl.col("fraud_flag") & ~pl.col("chargeback_flag"))
            .then(_AMOUNT)
            .otherwise(0.0)
            .alias("fraud_loss_cop"),
        ]

    def contribution(self) -> pl.Expr:
        return (
            pl.col("issuer_gross_cop")
            - pl.col("rewards_cop")
            - pl.col("funding_cop")
            - pl.col("credit_loss_cop")
            - pl.col("issuer_processing_cop")
            - pl.col("fraud_loss_cop")
        ).alias("issuer_contribution_cop")


@dataclass(frozen=True)
class AcquirerPnL:
    """MDR minus interchange and acquirer-side network fees, minus processing."""

    processing: Fee

    @classmethod
    def from_config(cls, cfg: ProjectConfig) -> AcquirerPnL:
        return cls(cfg.economics.acquirer.processing)

    def columns(self) -> list[pl.Expr]:
        return [
            # El adquirente se queda con el residuo; en blended puede ser negativo (premium,
            # cross-border), en IC++ es exactamente su margen.
            (
                pl.col("mdr_cop") - pl.col("interchange_cop") - pl.col("acquirer_network_fee_cop")
            ).alias("acquirer_gross_cop"),
            (_AMOUNT * self.processing.rate + self.processing.fixed_cop).alias(
                "acquirer_processing_cop"
            ),
        ]

    def contribution(self) -> pl.Expr:
        return (pl.col("acquirer_gross_cop") - pl.col("acquirer_processing_cop")).alias(
            "acquirer_contribution_cop"
        )


@dataclass(frozen=True)
class PnLResult:
    """P&L of a set of transactions.

    Attributes:
        transactions: Input transactions plus ``pricing_model`` and every ``MONEY_COLUMNS``.
        by_actor: One row per actor: ``actor_type``, ``actor_id``, ``revenue_cop`` (gross,
            from the waterfall), ``contribution_cop`` (after operating costs), ``gdv_cop``
            and ``n_txns``.
    """

    transactions: pl.DataFrame
    by_actor: pl.DataFrame

    @property
    def network_revenue(self) -> float:
        """Fees collected by the network (COP)."""
        return float(self.transactions["network_revenue_cop"].sum())

    def totals(self) -> dict[str, float]:
        """Network-wide GDV, every money column and the net revenue yield in bps of GDV."""
        totals = self.transactions.select(
            pl.col("amount_cop").sum().cast(pl.Float64).alias("gdv_cop"),
            *(pl.col(column).sum() for column in MONEY_COLUMNS),
        ).row(0, named=True)
        totals["net_revenue_yield_bps"] = totals["network_revenue_cop"] / totals["gdv_cop"] * _BPS
        return totals


def compute_pnl(
    transactions: pl.DataFrame,
    table: pl.DataFrame,
    cfg: ProjectConfig,
    merchants: pl.DataFrame | None = None,
    interchange_adjustment: pl.Expr | None = None,
) -> PnLResult:
    """Price every transaction and compute the P&L of issuer, network, acquirer and merchant.

    Args:
        transactions: Transactions with at least ``REQUIRED_COLUMNS``. Money columns already
            present are recomputed, so a priced frame can be repriced.
        table: Expanded interchange table (``build_interchange_table``). Separate from the
            configuration because it is the lever that scenarios and the optimizer change.
        cfg: Project configuration: blended MDR by group and ``economics`` parameters.
        merchants: Needed when ``transactions`` has no ``pricing_model`` column.
        interchange_adjustment: Expression for a new ``interchange_cop``, applied right after
            the table prices each transaction and before anything that depends on it. It is how
            a scenario caps or discounts interchange on some transactions without a new table.

    Raises:
        ValueError: If columns or prices are missing (a gap would silently leak P&L).
    """
    frame = transactions.drop(MONEY_COLUMNS, strict=False)
    missing_columns = [c for c in REQUIRED_COLUMNS if c not in frame.columns]
    if missing_columns:
        raise ValueError(f"transactions is missing required columns: {missing_columns}")
    if "pricing_model" not in frame.columns:
        if merchants is None:
            raise ValueError("transactions has no pricing_model: pass the merchants table")
        frame = frame.join(
            merchants.select("merchant_id", "pricing_model"),
            on="merchant_id",
            how="left",
            validate="m:1",
            maintain_order="left",
        )
    unknown = frame["pricing_model"].null_count()
    if unknown:
        raise ValueError(f"{unknown} transactions have no pricing_model (unknown merchant_id)")

    network = NetworkPnL.from_config(cfg)
    merchant = MerchantCost.from_config(cfg)
    issuer = IssuerPnL.from_config(cfg)
    acquirer = AcquirerPnL.from_config(cfg)

    interchanged = apply_interchange(frame, table)
    if interchange_adjustment is not None:
        # El ajuste entra antes de todo lo que depende del interchange (MDR en IC++, bruto del
        # emisor, residuo del adquirente), así que la identidad se sigue cumpliendo fila a fila.
        interchanged = interchanged.with_columns(interchange_adjustment.alias("interchange_cop"))
    priced = (
        interchanged.with_columns(network.components())
        .with_columns(network.totals())
        .with_columns(merchant.mdr(), merchant.chargeback())
    )
    unpriced = priced.filter(pl.col("mdr_cop").is_null())
    if unpriced.height:
        cells = sorted(set(unpriced.select("pricing_model", "mcc_group", "acquirer_id").rows()))
        raise ValueError(f"{unpriced.height} transactions have no MDR; missing prices for {cells}")
    priced = (
        priced.with_columns(*issuer.columns(), *acquirer.columns())
        .with_columns(
            issuer.contribution(),
            acquirer.contribution(),
            (pl.col("mdr_cop") + pl.col("chargeback_cop")).alias("merchant_cost_cop"),
        )
        .drop(_SIDE_COLUMNS)
    )
    missing_costs = priced.select(COST_COLUMNS).null_count().sum_horizontal().item()
    if missing_costs:
        raise ValueError("Some products have no issuer cost parameters")
    return PnLResult(transactions=priced, by_actor=_by_actor(priced))


def _by_actor(priced: pl.DataFrame) -> pl.DataFrame:
    frames = []
    for actor_type, id_column, revenue, contribution in _RECEIVERS:
        aggregations = [
            pl.col(revenue).sum().alias("revenue_cop"),
            pl.col(contribution).sum().alias("contribution_cop"),
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
                "contribution_cop",
                "gdv_cop",
                "n_txns",
            )
        )
    return pl.concat(frames)


def pnl_by(result: PnLResult, keys: Sequence[str | pl.Expr]) -> pl.DataFrame:
    """Aggregate volume and every money column by any segment; empty ``keys`` is the total.

    Adds rates over GDV: effective interchange and MDR, network yield, acquirer margin and
    issuer contribution.
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
        (pl.col("network_revenue_cop") / gdv).alias("net_revenue_yield"),
        (pl.col("acquirer_gross_cop") / gdv).alias("acquirer_margin_rate"),
        (pl.col("issuer_contribution_cop") / gdv).alias("issuer_contribution_rate"),
    )
