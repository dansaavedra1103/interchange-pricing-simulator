"""Unit economics: the conservation invariant, hand-computed P&L and the business metrics."""

from __future__ import annotations

import datetime as dt
from typing import Any

import numpy as np
import polars as pl
import pytest

from ips.data_gen.generate import GeneratedData, generate
from ips.data_gen.interchange_table import apply_interchange, build_interchange_table
from ips.economics.metrics import (
    effective_interchange,
    issuer_contribution,
    net_revenue_yield,
    relative_burden,
)
from ips.economics.pnl import (
    MONEY_COLUMNS,
    NETWORK_ACTOR_ID,
    NETWORK_DRIVER_COLUMNS,
    PnLResult,
    compute_pnl,
    pnl_by,
)
from ips.utils.config import (
    AcquirerEconomics,
    Adjustment,
    Adjustments,
    EconomicsConfig,
    Fee,
    InterchangeTableConfig,
    IssuerEconomics,
    IssuerProductCosts,
    NetworkFeesConfig,
    ProjectConfig,
    SideFees,
    SmallTicketConfig,
)

REL_TOL = 1e-9
ISSUER_COSTS = ("rewards_cop", "funding_cop", "credit_loss_cop", "issuer_processing_cop")


@pytest.fixture(scope="module")
def table(cfg: ProjectConfig) -> pl.DataFrame:
    return build_interchange_table(cfg)


@pytest.fixture(scope="module")
def pnl(cfg: ProjectConfig, data: GeneratedData, table: pl.DataFrame) -> PnLResult:
    return compute_pnl(data.tables["transactions"], table, cfg, merchants=data.tables["merchants"])


# ---------------------------------------------------------------------------
# Conservación del P&L: emisor + red + adquirente = MDR pagado por los comercios.
# ---------------------------------------------------------------------------


def _merchants_paid(cfg: ProjectConfig, data: GeneratedData, table: pl.DataFrame) -> float:
    """MDR recomputed from the facts and the configuration, without compute_pnl."""
    network = cfg.economics.network.acquirer
    terms = cfg.economics.acquirer
    blended = {group: c.blended_mdr_rate for group, c in cfg.mcc_groups.groups.items()}
    cells = (
        apply_interchange(data.tables["transactions"], table)
        .join(data.tables["merchants"].select("merchant_id", "pricing_model"), on="merchant_id")
        .group_by(
            pl.col("mcc_group").cast(pl.Utf8),
            pl.col("pricing_model").cast(pl.Utf8),
            "cross_border",
        )
        .agg(
            pl.col("amount_cop").sum().cast(pl.Float64).alias("gdv"),
            pl.col("interchange_cop").sum().alias("interchange"),
            pl.len().alias("n"),
        )
    )
    paid = 0.0
    for group, model, cross_border, gdv, interchange, n in cells.iter_rows():
        if model == "icpp":
            # IC++: interchange + fees de red del adquirente + margen del adquirente.
            rate = network.assessment_rate + terms.icpp_markup.rate
            rate += network.cross_border_rate if cross_border else 0.0
            fixed = network.authorization_fee_cop + terms.icpp_markup.fixed_cop
            paid += interchange + gdv * rate + n * fixed
        else:
            # Blended: tasa del grupo, más el recargo si la tarjeta es extranjera.
            surcharge = terms.blended_cross_border_surcharge if cross_border else 0.0
            paid += gdv * (blended[group] + surcharge)
    return paid


def test_conservation_total_against_independent_mdr(
    cfg: ProjectConfig, data: GeneratedData, table: pl.DataFrame, pnl: PnLResult
) -> None:
    paid = _merchants_paid(cfg, data, table)
    # Lo que reciben los actores, agregado por separado por issuer_id, acquirer_id y red.
    received = pnl.by_actor["revenue_cop"].sum()
    assert paid > 0
    assert received == pytest.approx(paid, rel=REL_TOL)
    assert pnl.totals()["mdr_cop"] == pytest.approx(paid, rel=REL_TOL)


SEGMENTS: dict[str, list[str | pl.Expr]] = {
    "total": [],
    "month": [pl.col("txn_date").dt.truncate("1mo").alias("month")],
    "mcc_group": ["mcc_group"],
    "product": ["product"],
    "channel": ["channel"],
    "cross_border": ["cross_border"],
    "issuer": ["issuer_id"],
    "acquirer": ["acquirer_id"],
    "pricing_model": ["pricing_model"],
    "on_us": ["on_us"],
    "group_x_product": ["mcc_group", "product"],
}


@pytest.mark.parametrize("keys", list(SEGMENTS.values()), ids=list(SEGMENTS))
def test_conservation_in_every_aggregation(pnl: PnLResult, keys: list[str | pl.Expr]) -> None:
    segments = pnl_by(pnl, keys)
    received = segments.select(
        pl.sum_horizontal("issuer_gross_cop", "network_revenue_cop", "acquirer_gross_cop")
    ).to_series()
    assert received.to_list() == pytest.approx(segments["mdr_cop"].to_list(), rel=REL_TOL)
    # El ingreso de la red es exactamente la suma de sus conceptos.
    drivers = segments.select(pl.sum_horizontal(NETWORK_DRIVER_COLUMNS)).to_series()
    network = segments["network_revenue_cop"].to_list()
    assert drivers.to_list() == pytest.approx(network, rel=REL_TOL)
    assert segments["mdr_cop"].sum() == pytest.approx(pnl.totals()["mdr_cop"], rel=REL_TOL)
    assert segments["n_txns"].sum() == pnl.transactions.height


def test_by_actor_matches_row_level(cfg: ProjectConfig, pnl: PnLResult) -> None:
    receivers = {
        "issuer": ("issuer_gross_cop", "issuer_contribution_cop"),
        "acquirer": ("acquirer_gross_cop", "acquirer_contribution_cop"),
        "network": ("network_revenue_cop", "network_revenue_cop"),
    }
    tx = pnl.transactions
    assert set(pnl.by_actor["actor_type"]) == set(receivers)
    for actor_type, (revenue, contribution) in receivers.items():
        actors = pnl.by_actor.filter(pl.col("actor_type") == actor_type)
        assert actors["revenue_cop"].sum() == pytest.approx(tx[revenue].sum(), rel=REL_TOL)
        assert actors["contribution_cop"].sum() == pytest.approx(
            tx[contribution].sum(), rel=REL_TOL
        )
        # Cada transacción, on-us y cross-border incluidas, va a un solo actor de cada tipo.
        assert actors["n_txns"].sum() == tx.height
    issuers = pnl.by_actor.filter(pl.col("actor_type") == "issuer")["actor_id"]
    assert set(issuers) == {issuer.issuer_id for issuer in cfg.issuers}
    network = pnl.by_actor.filter(pl.col("actor_type") == "network")
    assert network["actor_id"].to_list() == [NETWORK_ACTOR_ID]
    assert network["revenue_cop"].item() == pytest.approx(pnl.network_revenue, rel=REL_TOL)


def test_compute_pnl_preserves_rows(transactions: pl.DataFrame, pnl: PnLResult) -> None:
    assert pnl.transactions.height == transactions.height
    assert (pnl.transactions["txn_id"] == transactions["txn_id"]).all()
    assert pnl.transactions.select(MONEY_COLUMNS).null_count().sum_horizontal().item() == 0


def test_mdr_follows_the_pricing_model(cfg: ProjectConfig, pnl: PnLResult) -> None:
    terms = cfg.economics.acquirer
    icpp = pnl.transactions.filter(pl.col("pricing_model") == "icpp")
    blended = pnl.transactions.filter(pl.col("pricing_model") == "blended")
    assert icpp.height > 0
    assert blended.height > 0
    # En IC++ el adquirente se queda exactamente su margen: nunca pierde en una transacción.
    markup = icpp["amount_cop"].cast(pl.Float64) * terms.icpp_markup.rate
    np.testing.assert_allclose(
        icpp["acquirer_gross_cop"], markup + terms.icpp_markup.fixed_cop, rtol=REL_TOL
    )
    # En blended paga la tasa de su grupo (más el recargo por tarjeta extranjera).
    rates = {group: c.blended_mdr_rate for group, c in cfg.mcc_groups.groups.items()}
    rate = blended["mcc_group"].cast(pl.Utf8).replace_strict(rates, return_dtype=pl.Float64)
    surcharge = blended["cross_border"].cast(pl.Float64) * terms.blended_cross_border_surcharge
    expected = blended["amount_cop"].cast(pl.Float64) * (rate + surcharge)
    np.testing.assert_allclose(blended["mdr_cop"], expected, rtol=REL_TOL)


def test_priced_frame_can_be_repriced(
    cfg: ProjectConfig, table: pl.DataFrame, pnl: PnLResult
) -> None:
    again = compute_pnl(pnl.transactions, table, cfg)
    assert again.transactions.columns == pnl.transactions.columns
    assert again.totals() == pytest.approx(pnl.totals(), rel=1e-12)


def test_missing_inputs_raise(cfg: ProjectConfig, data: GeneratedData, table: pl.DataFrame) -> None:
    transactions, merchants = data.tables["transactions"], data.tables["merchants"]
    groups = {name: c for name, c in cfg.mcc_groups.groups.items() if name != "travel"}
    no_rate = cfg.model_copy(
        update={"mcc_groups": cfg.mcc_groups.model_copy(update={"groups": groups})}
    )
    with pytest.raises(ValueError, match="have no MDR"):
        compute_pnl(transactions, table, no_rate, merchants=merchants)

    no_cell = table.filter(~((pl.col("product") == "debit") & (pl.col("mcc_group") == "grocery")))
    with pytest.raises(ValueError, match="no interchange cell"):
        compute_pnl(transactions, no_cell, cfg, merchants=merchants)

    products = {p: c for p, c in cfg.economics.issuer.products.items() if p != "commercial"}
    economics = cfg.economics.model_copy(update={"issuer": IssuerEconomics(products=products)})
    no_costs = cfg.model_copy(update={"economics": economics})
    with pytest.raises(ValueError, match="no issuer cost"):
        compute_pnl(transactions, table, no_costs, merchants=merchants)

    with pytest.raises(ValueError, match="pass the merchants table"):
        compute_pnl(transactions, table, cfg)
    with pytest.raises(ValueError, match="no pricing_model"):
        compute_pnl(transactions, table, cfg, merchants=merchants.head(10))
    with pytest.raises(ValueError, match="missing required columns"):
        compute_pnl(transactions.drop("fraud_flag"), table, cfg, merchants=merchants)


def test_same_seed_same_totals(cfg: ProjectConfig, table: pl.DataFrame, pnl: PnLResult) -> None:
    # The data are bit-identical; float totals are compared with a tolerance because polars
    # parallelises float reductions, so the accumulation order can change the last bit.
    again = generate(cfg)
    result = compute_pnl(
        again.tables["transactions"], table, cfg, merchants=again.tables["merchants"]
    )
    assert result.totals() == pytest.approx(pnl.totals(), rel=1e-12)


# ---------------------------------------------------------------------------
# Ejemplos calculados a mano, con parámetros redondos independientes del YAML calibrado.
# ---------------------------------------------------------------------------


def _hand_config(cfg: ProjectConfig) -> ProjectConfig:
    table = InterchangeTableConfig(
        version="hand",
        valid_from=dt.date(2024, 1, 1),
        base={
            # Ejemplo §1.2 de la spec: crédito en supermercado al 1,70 % sin fijo.
            "credit_standard": {"grocery": Fee(rate=0.017)},
            "debit": {"grocery": Fee(rate=0.004)},
            "credit_premium": {
                "travel": Fee(rate=0.025, fixed_cop=150),
                "retail": Fee(rate=0.023, fixed_cop=150),
            },
        },
        adjustments=Adjustments(
            channel={"cnp": Adjustment(rate_add=0.003)},
            region={"cross_border": Adjustment(rate_add=0.008)},
        ),
        small_ticket=SmallTicketConfig(
            max_amount_cop=20_000, products=("debit",), groups=("grocery",), rate=0.002
        ),
    )
    costs = {
        "debit": (0.0, 0.0, 0.0, 0.001),
        "credit_standard": (0.004, 0.002, 0.003, 0.0015),
        "credit_premium": (0.012, 0.0025, 0.0015, 0.002),
    }
    economics = EconomicsConfig(
        network=NetworkFeesConfig(
            issuer=SideFees(
                assessment_rate=0.0005, authorization_fee_cop=40, cross_border_rate=0.008
            ),
            acquirer=SideFees(
                assessment_rate=0.0013, authorization_fee_cop=80, cross_border_rate=0.0045
            ),
        ),
        acquirer=AcquirerEconomics(
            icpp_markup=Fee(rate=0.0035, fixed_cop=200),
            blended_cross_border_surcharge=0.015,
            processing=Fee(rate=0.001, fixed_cop=100),
        ),
        issuer=IssuerEconomics(
            products={
                product: IssuerProductCosts(
                    rewards_rate=rewards,
                    funding_rate=funding,
                    credit_loss_rate=credit_loss,
                    processing_rate=processing,
                )
                for product, (rewards, funding, credit_loss, processing) in costs.items()
            }
        ),
    )
    groups = {
        name: group.model_copy(update={"blended_mdr_rate": 0.025})
        for name, group in cfg.mcc_groups.groups.items()
    }
    return cfg.model_copy(
        update={
            "interchange_table": table,
            "mcc_groups": cfg.mcc_groups.model_copy(update={"groups": groups}),
            "economics": economics,
        }
    )


HAND_COLUMNS = (
    "product",
    "mcc_group",
    "channel",
    "cross_border",
    "amount_cop",
    "pricing_model",
    "fraud_flag",
    "chargeback_flag",
)
HAND_ROWS = {
    "spec_1_2": ("credit_standard", "grocery", "cp", False, 100_000, "blended", False, False),
    "icpp": ("credit_standard", "grocery", "cp", False, 100_000, "icpp", False, False),
    "micropayment": ("debit", "grocery", "cp", False, 15_000, "blended", False, False),
    "cross_border": ("credit_premium", "travel", "cnp", True, 1_000_000, "blended", False, False),
    "fraud": ("credit_premium", "retail", "cnp", False, 200_000, "blended", True, False),
    "fraud_chargeback": ("credit_premium", "retail", "cnp", False, 200_000, "blended", True, True),
}


@pytest.fixture(scope="module")
def hand(cfg: ProjectConfig) -> dict[str, dict[str, Any]]:
    hand_cfg = _hand_config(cfg)
    rows = [dict(zip(HAND_COLUMNS, values, strict=True)) for values in HAND_ROWS.values()]
    transactions = pl.DataFrame(rows).with_columns(
        pl.int_range(0, pl.len(), dtype=pl.UInt32).alias("txn_id"),
        pl.lit(0, dtype=pl.UInt32).alias("merchant_id"),
        pl.lit("ISS").alias("issuer_id"),
        pl.lit(cfg.acquirers[0].acquirer_id).alias("acquirer_id"),
    )
    result = compute_pnl(transactions, build_interchange_table(hand_cfg), hand_cfg)
    return dict(
        zip(HAND_ROWS, result.transactions.sort("txn_id").iter_rows(named=True), strict=True)
    )


def test_hand_examples_conserve(hand: dict[str, dict[str, Any]]) -> None:
    for name, row in hand.items():
        received = row["issuer_gross_cop"] + row["network_revenue_cop"] + row["acquirer_gross_cop"]
        assert received == pytest.approx(row["mdr_cop"]), name


def test_spec_example_blended(hand: dict[str, dict[str, Any]]) -> None:
    # Ejemplo §1.2 de la spec: 100.000 COP con crédito en supermercado, MDR 2,5 % e interchange
    # 1,70 %. Además del scheme fee (0,13 %), la red cobra la autorización a ambos lados y un
    # assessment al emisor: se queda 300 COP, no los 130 del ejemplo simplificado.
    row = hand["spec_1_2"]
    assert row["mdr_cop"] == pytest.approx(2_500)
    assert row["interchange_cop"] == pytest.approx(1_700)
    assert row["issuer_network_fee_cop"] == pytest.approx(50 + 40)
    assert row["acquirer_network_fee_cop"] == pytest.approx(130 + 80)
    assert row["network_revenue_cop"] == pytest.approx(300)
    assert row["network_assessment_cop"] == pytest.approx(50 + 130)
    assert row["network_authorization_cop"] == pytest.approx(40 + 80)
    assert row["network_cross_border_cop"] == pytest.approx(0)
    assert row["issuer_gross_cop"] == pytest.approx(1_610)
    assert row["acquirer_gross_cop"] == pytest.approx(590)
    # Crédito estándar: recompensas 0,4 %, fondeo 0,2 %, pérdida esperada 0,3 % y
    # procesamiento 0,15 %; el adquirente procesa a 0,10 % + 100 COP.
    assert row["issuer_contribution_cop"] == pytest.approx(1_610 - 400 - 200 - 300 - 150)
    assert row["acquirer_contribution_cop"] == pytest.approx(590 - 100 - 100)
    assert row["merchant_cost_cop"] == pytest.approx(2_500)


def test_icpp_example(hand: dict[str, dict[str, Any]]) -> None:
    # La misma compra en un comercio IC++: paga el interchange y los fees de red del adquirente
    # tal cual, más el margen del adquirente (0,35 % + 200 COP), que es lo único que este gana.
    row, blended = hand["icpp"], hand["spec_1_2"]
    markup = 350 + 200
    assert row["mdr_cop"] == pytest.approx(1_700 + 210 + markup)
    assert row["acquirer_gross_cop"] == pytest.approx(markup)
    assert row["issuer_gross_cop"] == pytest.approx(blended["issuer_gross_cop"])
    assert row["network_revenue_cop"] == pytest.approx(blended["network_revenue_cop"])


def test_micropayment_example(hand: dict[str, dict[str, Any]]) -> None:
    # Micropago de 15.000 COP en débito: tramo reducido de interchange (0,20 % = 30 COP). El
    # fee fijo de autorización del emisor (40 COP) supera al interchange: el emisor pierde en
    # la transacción antes de sus costos.
    row = hand["micropayment"]
    assert row["interchange_cop"] == pytest.approx(30)
    assert row["issuer_gross_cop"] == pytest.approx(30 - (7.5 + 40))
    assert row["issuer_gross_cop"] < 0
    assert row["acquirer_gross_cop"] == pytest.approx(375 - 30 - (19.5 + 80))
    assert row["network_revenue_cop"] == pytest.approx(47.5 + 99.5)


def test_cross_border_example(hand: dict[str, dict[str, Any]]) -> None:
    # Premium no presente en un comercio del exterior: interchange 2,50 % + 0,30 pp (cnp)
    # + 0,80 pp (cross-border) + 150 COP. La red cobra el recargo cross-border a ambos lados y
    # el comercio paga su tarifa blended más 1,5 pp por tarjeta extranjera.
    row = hand["cross_border"]
    assert row["interchange_cop"] == pytest.approx(36_150)
    assert row["issuer_network_fee_cop"] == pytest.approx(500 + 40 + 8_000)
    assert row["acquirer_network_fee_cop"] == pytest.approx(1_300 + 80 + 4_500)
    assert row["network_cross_border_cop"] == pytest.approx(8_000 + 4_500)
    assert row["mdr_cop"] == pytest.approx(1_000_000 * (0.025 + 0.015))
    # Aun con el recargo, la tarifa plana no cubre una premium en viajes: el adquirente pierde.
    assert row["acquirer_gross_cop"] == pytest.approx(40_000 - 36_150 - 5_880)
    assert row["acquirer_gross_cop"] < 0


def test_premium_issuer_contribution_with_fraud(hand: dict[str, dict[str, Any]]) -> None:
    # Premium no presente de 200.000 COP: interchange 2,60 % + 150 COP = 5.350 COP. Costos
    # premium: recompensas 1,2 %, fondeo 0,25 %, pérdida esperada 0,15 % y procesamiento 0,2 %.
    before_fraud = 5_350 - (100 + 40) - (2_400 + 500 + 300 + 400)
    fraud, charged_back = hand["fraud"], hand["fraud_chargeback"]
    # Sin contracargo el emisor absorbe el fraude completo.
    assert fraud["fraud_loss_cop"] == pytest.approx(200_000)
    assert fraud["issuer_contribution_cop"] == pytest.approx(before_fraud - 200_000)
    assert fraud["merchant_cost_cop"] == pytest.approx(5_000)
    # Con contracargo la pérdida pasa al comercio (liability shift).
    assert charged_back["fraud_loss_cop"] == pytest.approx(0)
    assert charged_back["issuer_contribution_cop"] == pytest.approx(before_fraud)
    assert charged_back["chargeback_cop"] == pytest.approx(200_000)
    assert charged_back["merchant_cost_cop"] == pytest.approx(5_000 + 200_000)


# ---------------------------------------------------------------------------
# Métricas de negocio y rangos de calibración
# ---------------------------------------------------------------------------


def test_yield_and_interchange_metrics(pnl: PnLResult) -> None:
    totals = pnl.totals()
    overall = net_revenue_yield(pnl)
    bps = overall["net_revenue_yield_bps"].item()
    assert bps == pytest.approx(totals["net_revenue_yield_bps"], rel=REL_TOL)
    drivers = overall.select(pl.sum_horizontal(NETWORK_DRIVER_COLUMNS)).item()
    assert drivers == pytest.approx(totals["network_revenue_cop"], rel=REL_TOL)
    # El yield total es el promedio de los segmentos ponderado por volumen.
    by_product = net_revenue_yield(pnl, by=["product"])
    weighted = by_product.select(
        (pl.col("net_revenue_yield_bps") * pl.col("gdv_cop")).sum() / pl.col("gdv_cop").sum()
    ).item()
    assert weighted == pytest.approx(bps, rel=REL_TOL)

    interchange = effective_interchange(pnl, by=["product"])
    assert interchange["interchange_cop"].sum() == pytest.approx(
        totals["interchange_cop"], rel=REL_TOL
    )
    rate = dict(
        zip(
            interchange["product"].cast(pl.Utf8),
            interchange["effective_interchange_rate"],
            strict=True,
        )
    )
    # Orden de la tabla (spec §1.4): débito < crédito estándar < premium.
    assert rate["debit"] < rate["credit_standard"] < rate["credit_premium"]


def test_issuer_contribution_lines(pnl: PnLResult) -> None:
    lines = issuer_contribution(pnl)
    np.testing.assert_allclose(
        lines["issuer_gross_cop"],
        lines["interchange_cop"] - lines["issuer_network_fee_cop"],
        rtol=REL_TOL,
    )
    costs = lines.select(pl.sum_horizontal(*ISSUER_COSTS, "fraud_loss_cop")).to_series()
    np.testing.assert_allclose(
        lines["issuer_contribution_cop"], lines["issuer_gross_cop"] - costs, rtol=REL_TOL
    )
    issuers = pnl.by_actor.filter(pl.col("actor_type") == "issuer")
    assert lines["issuer_contribution_cop"].sum() == pytest.approx(
        issuers["contribution_cop"].sum(), rel=REL_TOL
    )


def test_relative_burden(pnl: PnLResult, data: GeneratedData) -> None:
    burden = relative_burden(pnl, data.tables["merchants"])
    assert burden.height == pnl.transactions["merchant_id"].n_unique()
    assert burden["merchant_id"].is_unique().all()
    assert burden["n_txns"].sum() == pnl.transactions.height
    assert burden["mdr_cop"].sum() == pytest.approx(pnl.totals()["mdr_cop"], rel=REL_TOL)
    expected = burden["mdr_cop"] / burden["gdv_cop"] / burden["sector_margin"]
    np.testing.assert_allclose(burden["relative_burden"], expected, rtol=REL_TOL)


def test_calibrated_ranges(pnl: PnLResult) -> None:
    # Rangos objetivo de la calibración (docs/assumptions.md, F3). Si un cambio de parámetros
    # los rompe, se recalibra y se documenta; no se amplía el rango.
    # Red: en 2025 Visa y Mastercard ingresaron ~29-31 pb del volumen netos de incentivos (con
    # servicios de valor agregado) y ~37-38 pb brutos por la red de pagos. El modelo no tiene
    # incentivos ni servicios de valor agregado.
    assert 25 <= pnl.totals()["net_revenue_yield_bps"] <= 40
    # Adquirente doméstico en blended: margen bruto de 0,2-0,7 % del monto (spec §1.6).
    segments = pnl_by(pnl, ["pricing_model", "cross_border"])
    domestic_blended = segments.filter(
        (pl.col("pricing_model") == "blended") & ~pl.col("cross_border")
    )
    assert 0.002 <= domestic_blended["acquirer_margin_rate"].item() <= 0.007
    # En IC++ el adquirente cobra solo su margen negociado, también dentro del rango.
    by_model = pnl_by(pnl, ["pricing_model"])
    icpp = by_model.filter(pl.col("pricing_model") == "icpp")
    assert 0.002 <= icpp["acquirer_margin_rate"].item() <= 0.007
    # El emisor gana en cada producto: el interchange paga la operación de la tarjeta.
    assert (issuer_contribution(pnl)["issuer_contribution_rate"] > 0).all()
