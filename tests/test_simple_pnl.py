"""Transitional P&L on Feature 1 data: the conservation invariant and hand-computed fees."""

from __future__ import annotations

import polars as pl
import pytest

from ips.data_gen.generate import generate
from ips.data_gen.interchange_table import build_interchange_table
from ips.economics.simple_pnl import MONEY_COLUMNS, PnLResult, compute_pnl, pnl_by
from ips.utils.config import Fee, OverrideConfig, PricingConfig, ProjectConfig

REL_TOL = 1e-9


@pytest.fixture(scope="module")
def table(cfg: ProjectConfig) -> pl.DataFrame:
    return build_interchange_table(cfg)


@pytest.fixture(scope="module")
def pnl(cfg: ProjectConfig, transactions: pl.DataFrame, table: pl.DataFrame) -> PnLResult:
    return compute_pnl(transactions, table, cfg.pricing)


# ---------------------------------------------------------------------------
# Conservación del P&L: emisor + red + adquirente = MDR pagado por los comercios.
# ---------------------------------------------------------------------------


def test_conservation_total_against_independent_mdr(
    cfg: ProjectConfig, transactions: pl.DataFrame, pnl: PnLResult
) -> None:
    # Lo que pagan los comercios se recalcula desde los hechos y el config, sin usar ninguna
    # columna producida por compute_pnl.
    gdv_by_group = transactions.group_by("mcc_group").agg(pl.col("amount_cop").sum()).iter_rows()
    merchants_paid = sum(gdv * cfg.pricing.blended_mdr_rate[group] for group, gdv in gdv_by_group)

    # Lo que reciben los actores, agregado por separado por issuer_id, acquirer_id y red.
    received = pnl.by_actor["revenue_cop"].sum()

    assert merchants_paid > 0
    assert received == pytest.approx(merchants_paid, rel=REL_TOL)


SEGMENTS: dict[str, list[str | pl.Expr]] = {
    "total": [],
    "month": [pl.col("txn_date").dt.truncate("1mo").alias("month")],
    "mcc_group": ["mcc_group"],
    "product": ["product"],
    "channel": ["channel"],
    "cross_border": ["cross_border"],
    "issuer": ["issuer_id"],
    "acquirer": ["acquirer_id"],
    "on_us": ["on_us"],
    "group_x_product": ["mcc_group", "product"],
}


@pytest.mark.parametrize("keys", list(SEGMENTS.values()), ids=list(SEGMENTS))
def test_conservation_in_every_aggregation(pnl: PnLResult, keys: list[str | pl.Expr]) -> None:
    segments = pnl_by(pnl, keys)
    received = (
        segments["interchange_cop"] + segments["scheme_fee_cop"] + segments["acquirer_net_cop"]
    )
    assert received.to_list() == pytest.approx(segments["mdr_cop"].to_list(), rel=REL_TOL)
    assert segments["mdr_cop"].sum() == pytest.approx(pnl.totals()["mdr_cop"], rel=REL_TOL)
    assert segments["n_txns"].sum() == pnl.transactions.height


def test_by_actor_matches_row_level(pnl: PnLResult) -> None:
    receivers = {
        "issuer": "interchange_cop",
        "acquirer": "acquirer_net_cop",
        "network": "scheme_fee_cop",
    }
    tx = pnl.transactions
    assert set(pnl.by_actor["actor_type"].unique()) == set(receivers)
    for actor_type, column in receivers.items():
        actors = pnl.by_actor.filter(pl.col("actor_type") == actor_type)
        assert actors["revenue_cop"].sum() == pytest.approx(tx[column].sum(), rel=REL_TOL)
        # Cada transacción, on-us y cross-border incluidas, va a un solo actor de cada tipo.
        assert actors["n_txns"].sum() == tx.height


def test_compute_pnl_preserves_rows(transactions: pl.DataFrame, pnl: PnLResult) -> None:
    assert pnl.transactions.height == transactions.height
    assert pnl.transactions["txn_id"].n_unique() == transactions.height
    assert pnl.transactions.select(MONEY_COLUMNS).null_count().sum_horizontal().item() == 0


def test_missing_prices_raise(
    cfg: ProjectConfig, transactions: pl.DataFrame, table: pl.DataFrame
) -> None:
    rates = {g: r for g, r in cfg.pricing.blended_mdr_rate.items() if g != "travel"}
    pricing = PricingConfig(scheme_fee=cfg.pricing.scheme_fee, blended_mdr_rate=rates)
    with pytest.raises(ValueError, match="No blended MDR rate"):
        compute_pnl(transactions, table, pricing)
    without_cell = table.filter(
        ~((pl.col("product") == "debit") & (pl.col("mcc_group") == "grocery"))
    )
    with pytest.raises(ValueError, match="no interchange cell"):
        compute_pnl(transactions, without_cell, cfg.pricing)


def test_hand_computed_examples(cfg: ProjectConfig) -> None:
    # Ejemplo §1.2 de la spec: 100.000 COP con crédito en supermercado, MDR 2,5 %,
    # interchange 1,70 % y scheme fee 0,13 %. Se fija esa celda con un override.
    override = OverrideConfig(
        product="credit_standard",
        mcc_group="grocery",
        channel="cp",
        region="domestic",
        rate=0.017,
        fixed_cop=0,
    )
    spec = cfg.interchange_table.model_copy(update={"overrides": (override,)})
    table = build_interchange_table(cfg.model_copy(update={"interchange_table": spec}))
    pricing = PricingConfig(
        scheme_fee=Fee(rate=0.0013), blended_mdr_rate=dict.fromkeys(cfg.group_names, 0.025)
    )
    transactions = pl.DataFrame(
        {
            "txn_id": [0, 1, 2],
            "issuer_id": ["ISS"] * 3,
            "acquirer_id": ["ACQ"] * 3,
            "product": ["credit_standard", "debit", "credit_premium"],
            "mcc_group": ["grocery", "grocery", "travel"],
            "channel": ["cp", "cp", "cnp"],
            "cross_border": [False, False, True],
            "amount_cop": [100_000, 15_000, 1_000_000],
        }
    )
    result = compute_pnl(transactions, table, pricing)
    credit, small, premium = (
        result.transactions.sort("txn_id").select(MONEY_COLUMNS).iter_rows(named=True)
    )

    assert credit["interchange_cop"] == pytest.approx(1_700)
    assert credit["scheme_fee_cop"] == pytest.approx(130)
    assert credit["acquirer_net_cop"] == pytest.approx(670)
    assert credit["mdr_cop"] == pytest.approx(2_500)

    # Micropago en débito: tramo reducido de 0,20 % sin fijo.
    assert small["interchange_cop"] == pytest.approx(15_000 * 0.002)
    assert small["acquirer_net_cop"] == pytest.approx(375 - 30 - 19.5)

    # Premium no presente cross-border: 2,50 % + 0,30 + 0,80 pp + 150 COP. La tarifa blended
    # no alcanza a cubrirlo y el adquirente pierde en esa transacción.
    assert premium["interchange_cop"] == pytest.approx(1_000_000 * 0.036 + 150)
    assert premium["acquirer_net_cop"] == pytest.approx(25_000 - 36_150 - 1_300)
    assert result.network_revenue == pytest.approx(130 + 19.5 + 1_300)


def test_same_seed_same_totals(cfg: ProjectConfig, table: pl.DataFrame, pnl: PnLResult) -> None:
    # The data are bit-identical; float totals are compared with a tolerance because polars
    # parallelises float reductions, so the accumulation order can change the last bit.
    again = compute_pnl(generate(cfg).tables["transactions"], table, cfg.pricing)
    assert again.totals() == pytest.approx(pnl.totals(), rel=1e-12)
