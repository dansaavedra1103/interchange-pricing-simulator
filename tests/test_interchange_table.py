"""Interchange table: full grid, hand-computed fees, vectorised = scalar, business ordering."""

from __future__ import annotations

import itertools

import polars as pl
import pytest

from ips.data_gen.interchange_table import (
    TABLE_KEYS,
    InterchangeTable,
    apply_interchange,
    build_interchange_table,
)
from ips.utils.config import CHANNELS, REGIONS, OverrideConfig, ProjectConfig


@pytest.fixture(scope="module")
def table(cfg: ProjectConfig) -> InterchangeTable:
    return InterchangeTable.from_config(cfg)


def test_full_grid(cfg: ProjectConfig, table: InterchangeTable) -> None:
    frame = table.frame
    assert frame.height == len(cfg.products) * len(cfg.group_names) * 2 * 2 == 192
    assert not frame.select(TABLE_KEYS).is_duplicated().any()
    assert frame.select("rate", "fixed_cop").null_count().sum_horizontal().item() == 0
    assert (frame["valid_from"] == cfg.interchange_table.valid_from).all()


def test_cells_by_hand(table: InterchangeTable) -> None:
    cells = {tuple(row[k] for k in TABLE_KEYS): row for row in table.frame.iter_rows(named=True)}
    # Premium, viajes, no presente, cross-border: base 2,50 % + 0,30 pp + 0,80 pp, fijo 150.
    premium = cells[("credit_premium", "travel", "cnp", "cross_border")]
    assert premium["rate"] == pytest.approx(0.036)
    assert premium["fixed_cop"] == 150
    # El tramo de micropagos solo existe en celdas domésticas de los grupos configurados.
    assert cells[("debit", "grocery", "cp", "domestic")]["small_ticket_max_cop"] == 20_000
    assert cells[("debit", "grocery", "cp", "cross_border")]["small_ticket_max_cop"] is None
    assert cells[("credit_premium", "grocery", "cp", "domestic")]["small_ticket_max_cop"] is None


@pytest.mark.parametrize(
    "product, group, channel, region, amount, fee",
    [
        ("credit_standard", "restaurants", "cp", "domestic", 100_000, 100_000 * 0.017 + 100),
        ("debit", "grocery", "cp", "domestic", 15_000, 15_000 * 0.002),
        ("debit", "grocery", "cp", "domestic", 25_000, 25_000 * 0.004),
        ("debit", "retail", "cnp", "domestic", 100_000, 100_000 * 0.013),
        ("credit_premium", "travel", "cnp", "cross_border", 1_000_000, 1_000_000 * 0.036 + 150),
    ],
)
def test_lookup_by_hand(
    table: InterchangeTable,
    product: str,
    group: str,
    channel: str,
    region: str,
    amount: int,
    fee: float,
) -> None:
    assert table.lookup(product, group, channel, region, amount) == pytest.approx(fee)


def test_vectorised_matches_scalar(table: InterchangeTable, transactions: pl.DataFrame) -> None:
    sample = table.apply(transactions.sample(2_000, seed=1))
    for row in sample.iter_rows(named=True):
        region = "cross_border" if row["cross_border"] else "domestic"
        expected = table.lookup(
            row["product"], row["mcc_group"], row["channel"], region, row["amount_cop"]
        )
        assert row["interchange_cop"] == pytest.approx(expected)


def test_product_ordering(cfg: ProjectConfig, table: InterchangeTable) -> None:
    # Débito < crédito estándar < premium <= comercial en cada celda (spec §1.4).
    rates = {
        tuple(row[k] for k in TABLE_KEYS): row["rate"] for row in table.frame.iter_rows(named=True)
    }
    for group, channel, region in itertools.product(cfg.group_names, CHANNELS, REGIONS):
        debit, standard, premium, commercial = (
            rates[(p, group, channel, region)] for p in cfg.products
        )
        assert debit < standard < premium <= commercial, (group, channel, region)


def test_missing_and_duplicate_cells_raise(
    table: InterchangeTable, transactions: pl.DataFrame
) -> None:
    sample = transactions.head(1_000)
    grocery_debit = (pl.col("product") == "debit") & (pl.col("mcc_group") == "grocery")
    with pytest.raises(ValueError, match="no interchange cell"):
        apply_interchange(sample, table.frame.filter(~grocery_debit))
    with pytest.raises(ValueError, match="duplicate cells"):
        apply_interchange(sample, pl.concat([table.frame, table.frame.head(1)]))


def test_override_replaces_cell(cfg: ProjectConfig) -> None:
    override = OverrideConfig(
        product="debit", mcc_group="fuel", channel="cp", region="domestic", rate=0.001
    )
    spec = cfg.interchange_table.model_copy(update={"overrides": (override,)})
    frame = build_interchange_table(cfg.model_copy(update={"interchange_table": spec}))
    cell = frame.filter(
        (pl.col("product") == "debit")
        & (pl.col("mcc_group") == "fuel")
        & (pl.col("channel") == "cp")
        & (pl.col("region") == "domestic")
    )
    assert cell["rate"].item() == pytest.approx(0.001)
