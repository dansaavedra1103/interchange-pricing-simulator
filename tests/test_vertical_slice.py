"""Vertical slice tests: P&L conservation, determinism and generator structure."""

from __future__ import annotations

import os
import time

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from ips.data_gen.simple_generator import TRANSACTION_SCHEMA, SliceData, generate_all
from ips.economics.simple_pnl import (
    MONEY_COLUMNS,
    PnLResult,
    compute_pnl,
    interchange_table_frame,
    pnl_by,
)
from ips.utils.config import Fee, PricingConfig, ProjectConfig, load_config, project_root

REL_TOL = 1e-9
NOTEBOOK = project_root() / "notebooks" / "00_vertical_slice.ipynb"


@pytest.fixture(scope="module")
def cfg() -> ProjectConfig:
    return load_config()


@pytest.fixture(scope="module")
def data(cfg: ProjectConfig) -> SliceData:
    return generate_all(cfg)


@pytest.fixture(scope="module")
def pnl(cfg: ProjectConfig, data: SliceData) -> PnLResult:
    return compute_pnl(data.transactions, interchange_table_frame(cfg.pricing), cfg.pricing)


# ---------------------------------------------------------------------------
# Conservación del P&L: emisor + red + adquirente = MDR pagado por los comercios.
# ---------------------------------------------------------------------------


def test_conservation_total_against_independent_mdr(
    cfg: ProjectConfig, data: SliceData, pnl: PnLResult
) -> None:
    # Lo que pagan los comercios se recalcula desde los hechos crudos y el config, sin usar
    # ninguna columna producida por compute_pnl.
    gdv_by_mcc = data.transactions.group_by("mcc").agg(pl.col("amount_cop").sum()).iter_rows()
    merchants_paid = sum(gdv * cfg.pricing.blended_mdr_rate[mcc] for mcc, gdv in gdv_by_mcc)

    # Lo que reciben los actores, agregado por separado por issuer_id, acquirer_id y red.
    received = pnl.by_actor["revenue_cop"].sum()

    assert merchants_paid > 0
    assert received == pytest.approx(merchants_paid, rel=REL_TOL)


SEGMENTS: dict[str, list[str | pl.Expr]] = {
    "total": [],
    "month": [pl.col("txn_date").dt.truncate("1mo").alias("month")],
    "mcc": ["mcc"],
    "product": ["product"],
    "issuer": ["issuer_id"],
    "acquirer": ["acquirer_id"],
    "on_us": ["on_us"],
    "mcc_x_product": ["mcc", "product"],
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
        # Cada transacción, on-us incluidas, se atribuye a exactamente un actor de cada tipo.
        assert actors["n_txns"].sum() == tx.height
        assert actors["gdv_cop"].sum() == pytest.approx(float(tx["amount_cop"].sum()), rel=REL_TOL)


def test_compute_pnl_preserves_rows(data: SliceData, pnl: PnLResult) -> None:
    assert pnl.transactions.height == data.transactions.height
    assert pnl.transactions["txn_id"].n_unique() == data.transactions.height
    assert pnl.transactions.select(MONEY_COLUMNS).null_count().sum_horizontal().item() == 0


def test_missing_interchange_cell_raises(cfg: ProjectConfig, data: SliceData) -> None:
    table = interchange_table_frame(cfg.pricing)
    broken = table.filter(~((pl.col("product") == "credit_premium") & (pl.col("mcc") == "5411")))
    with pytest.raises(ValueError, match="missing cells"):
        compute_pnl(data.transactions, broken, cfg.pricing)


def test_duplicate_interchange_cell_raises(cfg: ProjectConfig, data: SliceData) -> None:
    table = interchange_table_frame(cfg.pricing)
    with pytest.raises(ValueError, match="duplicate cells"):
        compute_pnl(data.transactions, pl.concat([table, table.head(1)]), cfg.pricing)


def test_hand_computed_example_from_spec() -> None:
    # Ejemplo §1.2 de la spec: compra de 100.000 COP con crédito en un supermercado con MDR
    # de 2,5%, interchange de 1,70% y scheme fee de 0,13%. Una segunda compra en débito
    # ejercita el componente fijo del interchange.
    pricing = PricingConfig(
        scheme_fee=Fee(rate=0.0013),
        blended_mdr_rate={"5411": 0.025},
        interchange_table={
            "credit_standard": {"5411": Fee(rate=0.017)},
            "debit": {"5411": Fee(rate=0.004, fixed_cop=150)},
        },
    )
    transactions = pl.DataFrame(
        {
            "txn_id": [0, 1],
            "issuer_id": ["ISS", "ISS"],
            "acquirer_id": ["ACQ", "ACQ"],
            "product": ["credit_standard", "debit"],
            "mcc": ["5411", "5411"],
            "amount_cop": [100_000, 100_000],
        }
    )
    result = compute_pnl(transactions, interchange_table_frame(pricing), pricing)
    credit, debit = result.transactions.sort("txn_id").select(MONEY_COLUMNS).iter_rows(named=True)

    assert credit["interchange_cop"] == pytest.approx(1_700)
    assert credit["scheme_fee_cop"] == pytest.approx(130)
    assert credit["acquirer_net_cop"] == pytest.approx(670)
    assert credit["mdr_cop"] == pytest.approx(2_500)

    assert debit["interchange_cop"] == pytest.approx(0.004 * 100_000 + 150)
    assert debit["acquirer_net_cop"] == pytest.approx(2_500 - 550 - 130)
    assert result.network_revenue == pytest.approx(260)


# ---------------------------------------------------------------------------
# Determinismo con semilla fija.
# ---------------------------------------------------------------------------


def test_same_seed_same_data(cfg: ProjectConfig, data: SliceData, pnl: PnLResult) -> None:
    again = generate_all(cfg)
    for name, frame in data.tables().items():
        assert_frame_equal(frame, again.tables()[name], check_exact=True)
    # The data are bit-identical; float totals are compared with a tolerance because polars
    # parallelises float reductions, so the accumulation order can change the last bit.
    repriced = compute_pnl(again.transactions, interchange_table_frame(cfg.pricing), cfg.pricing)
    assert repriced.totals() == pytest.approx(pnl.totals(), rel=1e-12)


def test_different_seed_different_data(cfg: ProjectConfig, data: SliceData) -> None:
    other = generate_all(cfg.model_copy(update={"seed": cfg.seed + 1}))
    assert not other.transactions.equals(data.transactions)


# ---------------------------------------------------------------------------
# Estructura del generador.
# ---------------------------------------------------------------------------


def test_acquirer_independent_of_issuer(data: SliceData) -> None:
    issuer_group = dict(data.issuers.select("issuer_id", "bank_group").iter_rows())
    acquirer_group = dict(data.acquirers.select("acquirer_id", "bank_group").iter_rows())
    tx = data.transactions.with_columns(
        pl.col("issuer_id").replace_strict(issuer_group, return_dtype=pl.Utf8).alias("iss_group"),
        pl.col("acquirer_id")
        .replace_strict(acquirer_group, return_dtype=pl.Utf8)
        .alias("acq_group"),
    )

    # on_us sigue exactamente la regla: mismo grupo financiero, ambos no nulos.
    rule = (pl.col("iss_group") == pl.col("acq_group")).fill_null(False)
    assert tx.select((pl.col("on_us") == rule).all()).item()
    assert not tx.filter(pl.col("iss_group").is_null() | pl.col("acq_group").is_null())[
        "on_us"
    ].any()

    # Si adquirente y emisor son independientes, P(on-us) = Σ_g P(emisor ∈ g)·P(adquirente ∈ g).
    def shares(column: str) -> dict[str | None, float]:
        counts = tx.group_by(column).len()
        return {group: n / tx.height for group, n in counts.iter_rows()}

    issuer_shares, acquirer_shares = shares("iss_group"), shares("acq_group")
    expected = sum(
        share * acquirer_shares.get(group, 0.0)
        for group, share in issuer_shares.items()
        if group is not None
    )
    observed = tx["on_us"].mean()
    assert observed > 0
    assert abs(observed - expected) < 0.01


def test_generated_shapes(cfg: ProjectConfig, data: SliceData) -> None:
    tx = data.transactions
    assert tx.schema == TRANSACTION_SCHEMA
    assert tx.height == cfg.sizes.n_transactions
    assert data.merchants.height == cfg.sizes.n_merchants
    assert data.cardholders.height == cfg.sizes.n_cardholders
    assert tx["txn_id"].to_list() == list(range(tx.height))
    assert tx["txn_date"].is_sorted()
    assert tx["txn_date"].min() >= cfg.dates.start
    assert tx["txn_date"].max() <= cfg.dates.end
    assert tx["amount_cop"].min() >= cfg.distributions.min_amount_cop
    assert set(tx["product"].unique()) == set(cfg.products)
    assert set(tx["mcc"].unique()) == set(cfg.mcc_codes)
    assert tx.null_count().sum_horizontal().item() == 0


# ---------------------------------------------------------------------------
# Criterio de terminado: el notebook corre end-to-end en menos de un minuto.
# ---------------------------------------------------------------------------


def test_notebook_runs_under_one_minute(monkeypatch: pytest.MonkeyPatch) -> None:
    nbformat = pytest.importorskip("nbformat")
    nbclient = pytest.importorskip("nbclient")

    # El kernel es un subproceso: se le pasa el src de este checkout para que no importe
    # otra copia instalada del paquete.
    src = str(project_root() / "src")
    monkeypatch.setenv(
        "PYTHONPATH", os.pathsep.join(filter(None, [src, os.environ.get("PYTHONPATH")]))
    )

    notebook = nbformat.read(NOTEBOOK, as_version=4)
    start = time.perf_counter()
    nbclient.NotebookClient(
        notebook,
        timeout=60,
        kernel_name="python3",
        resources={"metadata": {"path": str(NOTEBOOK.parent)}},
    ).execute()
    elapsed = time.perf_counter() - start

    outputs = "".join(
        output.get("text", "")
        for cell in notebook.cells
        if cell.cell_type == "code"
        for output in cell.get("outputs", [])
    )
    assert elapsed < 60
    assert "Network revenue (scheme fees) =" in outputs
