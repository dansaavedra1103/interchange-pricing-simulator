"""Revenue bridge: an exact, order-independent split of the change in network revenue."""

from __future__ import annotations

import polars as pl
import pytest

from ips.data_gen.generate import GeneratedData
from ips.data_gen.interchange_table import build_interchange_table
from ips.economics.pnl import compute_pnl
from ips.economics.revenue_bridge import DRIVERS, revenue_bridge
from ips.utils.config import ProjectConfig


def _frame(rows: list[tuple[str, bool, float, float]]) -> pl.DataFrame:
    """Minimal priced frame from (product, cross_border, volume, network revenue) rows."""
    return pl.DataFrame(
        {
            "product": [row[0] for row in rows],
            "mcc_group": ["retail"] * len(rows),
            "channel": ["cp"] * len(rows),
            "cross_border": [row[1] for row in rows],
            "amount_cop": [row[2] for row in rows],
            "network_revenue_cop": [row[3] for row in rows],
        }
    )


BASE_ROWS = [
    ("debit", False, 600.0, 1.2),
    ("credit", False, 300.0, 1.5),
    ("credit", True, 100.0, 2.0),
]
BASE = _frame(BASE_ROWS)


def _only(driver: str, effects: dict[str, float], change: float) -> None:
    assert effects[driver] == pytest.approx(change, rel=1e-9)
    for other in DRIVERS:
        if other != driver:
            assert effects[other] == pytest.approx(0.0, abs=1e-9), other


def test_effects_add_up_to_the_change() -> None:
    new = _frame(
        [("debit", False, 700.0, 1.3), ("credit", False, 250.0, 1.6), ("credit", True, 180.0, 3.1)]
    )
    bridge = revenue_bridge(BASE, new)
    assert bridge.base_revenue == pytest.approx(4.7)
    assert bridge.new_revenue == pytest.approx(6.0)
    assert sum(bridge.effects.values()) == pytest.approx(bridge.change, rel=1e-12)
    assert bridge.frame()["step"].to_list() == ["base", *DRIVERS, "new"]


def test_no_change_gives_zero_effects() -> None:
    bridge = revenue_bridge(BASE, BASE)
    assert all(effect == pytest.approx(0.0, abs=1e-12) for effect in bridge.effects.values())


def test_pure_volume_growth() -> None:
    # Todo crece 10 % con la misma mezcla y las mismas tarifas: solo hay efecto volumen.
    new = BASE.with_columns(pl.col("amount_cop") * 1.1, pl.col("network_revenue_cop") * 1.1)
    bridge = revenue_bridge(BASE, new)
    _only("volume", bridge.effects, bridge.change)


def test_pure_rate_change() -> None:
    new = BASE.with_columns(pl.col("network_revenue_cop") * 1.2)
    bridge = revenue_bridge(BASE, new)
    _only("rate", bridge.effects, bridge.change)


def test_pure_cross_border_shift() -> None:
    # Mismo volumen total, mezcla doméstica y tarifas; solo sube la participación cross-border.
    new = _frame(
        [
            ("debit", False, 600.0 * 8 / 9, 1.2 * 8 / 9),
            ("credit", False, 300.0 * 8 / 9, 1.5 * 8 / 9),
            ("credit", True, 200.0, 4.0),
        ]
    )
    bridge = revenue_bridge(BASE, new)
    assert bridge.change == pytest.approx(1.7)
    _only("cross_border", bridge.effects, bridge.change)


def test_pure_mix_shift() -> None:
    # Dentro de lo doméstico se pasa volumen de débito a crédito, que rinde más a la red.
    new = _frame(
        [("debit", False, 500.0, 1.0), ("credit", False, 400.0, 2.0), ("credit", True, 100.0, 2.0)]
    )
    bridge = revenue_bridge(BASE, new)
    assert bridge.change == pytest.approx(0.3)
    _only("mix", bridge.effects, bridge.change)


def test_segment_present_in_one_period_only() -> None:
    new = _frame([*BASE_ROWS, ("commercial", False, 150.0, 0.9)])
    bridge = revenue_bridge(BASE, new)
    assert all(effect == effect for effect in bridge.effects.values())  # no NaN
    assert sum(bridge.effects.values()) == pytest.approx(bridge.change, rel=1e-12)


def test_bridge_on_generated_data(cfg: ProjectConfig, data: GeneratedData) -> None:
    priced = compute_pnl(
        data.tables["transactions"],
        build_interchange_table(cfg),
        cfg,
        merchants=data.tables["merchants"],
    ).transactions
    year = pl.col("txn_date").dt.year()
    bridge = revenue_bridge(priced.filter(year == 2024), priced.filter(year == 2025))
    assert sum(bridge.effects.values()) == pytest.approx(bridge.change, rel=1e-9)
    # La red creció por volumen (+19 % anual en compras); las tarifas no cambiaron.
    assert bridge.effects["volume"] > 0
    assert abs(bridge.effects["rate"]) < 0.25 * bridge.effects["volume"]
