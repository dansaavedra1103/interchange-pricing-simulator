"""dbt pipeline on the sample: it builds green and agrees with the Python implementation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pytest

from ips.data_gen.diagnostics import edge_list
from ips.data_gen.generate import GeneratedData, write_outputs
from ips.data_gen.interchange_table import build_interchange_table
from ips.economics.simple_pnl import MONEY_COLUMNS, PnLResult, compute_pnl
from ips.utils.config import ProjectConfig
from ips.utils.io import read_table, run_dbt

pytest.importorskip("dbt", reason="needs the pipeline extra: pip install -e '.[pipeline]'")


@pytest.fixture(scope="module")
def warehouse(
    tmp_path_factory: pytest.TempPathFactory, cfg: ProjectConfig, data: GeneratedData
) -> Path:
    base = tmp_path_factory.mktemp("pipeline")
    write_outputs(data, base / "raw", cfg, elapsed_seconds=0.0)
    path = base / "warehouse" / "ips.duckdb"
    result = run_dbt(
        ["build"],
        raw=base / "raw",
        warehouse=path,
        target_path=base / "target",
        log_path=base / "logs",
        capture=True,
    )
    assert result.returncode == 0, result.stdout[-6000:]
    return path


@pytest.fixture(scope="module")
def python_pnl(cfg: ProjectConfig, transactions: pl.DataFrame) -> PnLResult:
    return compute_pnl(transactions, build_interchange_table(cfg), cfg.pricing)


def test_fct_matches_python_row_by_row(warehouse: Path, python_pnl: PnLResult) -> None:
    # La misma tabla aplicada en SQL y en Python da el mismo P&L en cada transacción.
    fct = read_table("fct_transactions", warehouse).sort("txn_id")
    python = python_pnl.transactions.sort("txn_id")
    assert fct.height == python.height
    assert (fct["txn_id"] == python["txn_id"]).all()
    for column in MONEY_COLUMNS:
        np.testing.assert_allclose(
            fct[column].to_numpy(), python[column].to_numpy(), rtol=1e-9, atol=1e-6, err_msg=column
        )


def test_marts_add_up_to_the_fct(warehouse: Path) -> None:
    fct = read_table("fct_transactions", warehouse)
    pnl = read_table("mart_pnl_by_actor", warehouse)
    mdr = fct["mdr_cop"].sum()
    assert pnl["revenue_cop"].sum() == pytest.approx(mdr, rel=1e-9)
    for mart in ("mart_effective_interchange", "mart_merchant_relative_burden"):
        assert read_table(mart, warehouse)["mdr_cop"].sum() == pytest.approx(mdr, rel=1e-9), mart
    network = pnl.filter(pl.col("actor_type") == "network")["revenue_cop"].sum()
    assert network == pytest.approx(fct["scheme_fee_cop"].sum(), rel=1e-9)


def test_graph_edges_match_diagnostics(warehouse: Path, transactions: pl.DataFrame) -> None:
    keys = ["cardholder_id", "merchant_id"]
    mart = read_table("mart_graph_edges", warehouse).group_by(keys).agg(pl.col("n_txns").sum())
    expected = edge_list(transactions)
    joined = expected.join(mart, on=keys, how="full", coalesce=True)
    assert joined.height == expected.height == mart.height
    assert (joined["weight"] == joined["n_txns"].cast(pl.Float64)).all()


def test_relative_burden_matches_python(
    warehouse: Path, data: GeneratedData, python_pnl: PnLResult
) -> None:
    # Carga relativa = MDR efectivo del comercio / margen de su sector (spec §2.4).
    expected = (
        python_pnl.transactions.group_by("merchant_id")
        .agg(pl.col("mdr_cop").sum(), pl.col("amount_cop").sum())
        .join(data.tables["merchants"].select("merchant_id", "sector_margin"), on="merchant_id")
        .with_columns(
            (pl.col("mdr_cop") / pl.col("amount_cop") / pl.col("sector_margin")).alias("expected")
        )
    )
    burden = read_table("mart_merchant_relative_burden", warehouse).join(
        expected.select("merchant_id", "expected"), on="merchant_id"
    )
    assert burden.height == expected.height
    np.testing.assert_allclose(burden["relative_burden"], burden["expected"], rtol=1e-9)


def test_dim_date_has_no_gaps(warehouse: Path, transactions: pl.DataFrame) -> None:
    days = read_table("dim_date", warehouse)["date_day"].sort()
    first, last = transactions["txn_date"].min(), transactions["txn_date"].max()
    assert days.to_list() == pl.date_range(first, last, eager=True).to_list()
