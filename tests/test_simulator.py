"""Scenario simulator: the null scenario is the base year, every lever does exactly what it says,
and both the fees and the moved volume balance in every scenario."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest
from pydantic import ValidationError

from ips.data_gen.generate import GeneratedData
from ips.data_gen.interchange_table import build_interchange_table
from ips.economics.pnl import compute_pnl
from ips.simulator.cli import build_parser
from ips.simulator.engine import Simulator, passed_through
from ips.simulator.result import SimulationResult
from ips.simulator.scenario import Scenario, check_scenario, load_scenario, scenario_paths
from ips.simulator.sensitivity import sweep, with_parameter
from ips.simulator.state import NetworkState, build_state
from ips.utils.config import ProjectConfig

STANDARD = {
    "regulatory_cap_debit",
    "regulatory_cap_credit",
    "instant_payments_entry",
    "strategic_merchant_deal",
    "premium_migration",
    "cnp_tokenized_discount",
}
DEBIT_CAP = {"kind": "interchange_cap", "products": ["debit"], "max_rate": 0.003}
CREDIT_CAP = {
    "kind": "interchange_cap",
    "products": ["credit_standard", "credit_premium"],
    "max_rate": 0.005,
}


def _scenario(name: str, *levers: dict) -> Scenario:
    return Scenario.model_validate(
        {"name": name, "description": f"{name}.", "levers": list(levers)}
    )


def _with_simulator(cfg: ProjectConfig, **updates: float) -> ProjectConfig:
    return cfg.model_copy(update={"simulator": cfg.simulator.model_copy(update=updates)})


def _volume(frame: pl.DataFrame, condition: pl.Expr, weight: str | None = None) -> float:
    amount = pl.col("amount_cop").cast(pl.Float64)
    if weight is not None:
        amount = amount * pl.col(weight)
    return float(frame.filter(condition).select(amount.sum()).item())


def _changed(result: SimulationResult) -> pl.DataFrame:
    """Transactions whose interchange moved (without migrations, rows stay aligned)."""
    new = result.priced["interchange_cop"].to_numpy()
    old = result.baseline["interchange_cop"].to_numpy()
    return result.priced.filter(pl.Series(new != old))


@pytest.fixture(scope="module")
def substitutes(data: GeneratedData) -> pl.DataFrame:
    """Same-group neighbours by volume: enough to exercise redistribution without the GNN."""
    volume = (
        data.tables["transactions"]
        .group_by("merchant_id")
        .agg(pl.col("amount_cop").sum().cast(pl.Float64).alias("gdv"))
        .join(
            data.tables["merchants"].select("merchant_id", pl.col("mcc_group").cast(pl.Utf8)),
            on="merchant_id",
        )
    )
    pairs = (
        volume.join(volume, on="mcc_group", suffix="_substitute")
        .filter(pl.col("merchant_id") != pl.col("merchant_id_substitute"))
        .sort(
            ["merchant_id", "gdv_substitute", "merchant_id_substitute"],
            descending=[False, True, False],
        )
        .with_columns(pl.int_range(pl.len()).over("merchant_id").alias("position"))
        .filter(pl.col("position") < 5)
    )
    return pairs.select(
        "merchant_id",
        pl.col("merchant_id_substitute").alias("substitute_id"),
        (pl.col("gdv_substitute") / pl.col("gdv_substitute").max()).alias("similarity"),
        (pl.col("position") + 1).alias("rank"),
    )


@pytest.fixture(scope="module")
def state(cfg: ProjectConfig, data: GeneratedData, substitutes: pl.DataFrame) -> NetworkState:
    return build_state(
        data.tables["transactions"],
        data.tables["merchants"],
        data.tables["cardholders"],
        substitutes,
        cfg,
    )


@pytest.fixture(scope="module")
def simulator(state: NetworkState) -> Simulator:
    return Simulator(state)


@pytest.fixture(scope="module")
def debit_cap(simulator: Simulator) -> SimulationResult:
    return simulator.run(_scenario("debit_cap", DEBIT_CAP))


@pytest.fixture(scope="module")
def standard_results(cfg: ProjectConfig, simulator: Simulator) -> dict[str, SimulationResult]:
    return {path.stem: simulator.run(load_scenario(path, cfg)) for path in scenario_paths(cfg)}


# ---------------------------------------------------------------------------
# The base year and the scenario files
# ---------------------------------------------------------------------------


def test_the_null_scenario_is_the_base_year(simulator: Simulator) -> None:
    result = simulator.run(_scenario("nothing"))
    assert (result.totals()["change"] == 0.0).all()
    assert (result.priced["weight"] == 1.0).all()
    assert (result.acceptance["delta_p"] == 0.0).all()


def test_the_standard_scenarios_load(cfg: ProjectConfig) -> None:
    scenarios = {path.stem: load_scenario(path, cfg) for path in scenario_paths(cfg)}
    assert set(scenarios) == STANDARD | {"baseline"}
    assert all(scenario.name == stem for stem, scenario in scenarios.items())


def test_every_scenario_balances_fees_and_volume(
    standard_results: dict[str, SimulationResult],
) -> None:
    for name, result in standard_results.items():
        assert result.fee_gap() < 1e-9, name
        assert result.volume_gap() < 1e-9, name
        assert result.headline().startswith(result.scenario.description), name


def test_a_scenario_runs_the_same_every_time(cfg: ProjectConfig, simulator: Simulator) -> None:
    # Seis corridas y no dos: el ruido en el orden de las sumas no aparecía en todas.
    path = next(path for path in scenario_paths(cfg) if path.stem == "regulatory_cap_credit")
    scenario = load_scenario(path, cfg)
    first, *others = (simulator.run(scenario) for _ in range(6))
    for other in others:
        assert other.priced.equals(first.priced)
        assert other.acceptance.equals(first.acceptance)
        assert other.bridge == first.bridge


# ---------------------------------------------------------------------------
# Levers
# ---------------------------------------------------------------------------


def test_a_debit_cap_cuts_issuer_revenue_and_nothing_pays_more(
    debit_cap: SimulationResult,
) -> None:
    debit = (debit_cap.priced["product"].cast(pl.Utf8) == "debit").to_numpy()
    new = debit_cap.priced["interchange_cop"].to_numpy()
    old = debit_cap.baseline["interchange_cop"].to_numpy()
    assert np.all(new[debit] <= old[debit] + 1e-9)
    assert np.array_equal(new[~debit], old[~debit])
    change = {row["metric"]: row["change"] for row in debit_cap.totals().iter_rows(named=True)}
    assert change["interchange_cop"] < 0
    assert change["issuer_gross_cop"] < 0


def test_cheaper_interchange_never_raises_abandonment(debit_cap: SimulationResult) -> None:
    assert debit_cap.acceptance["delta_p"].max() <= 1e-12


def test_substitution_takes_exactly_its_share(simulator: Simulator) -> None:
    lever = {
        "kind": "volume_substitution",
        "where": {"products": ["debit"], "max_amount_cop": 50000},
        "share": 0.2,
    }
    result = simulator.run(_scenario("instant", lever))
    eligible = (pl.col("product").cast(pl.Utf8) == "debit") & (pl.col("amount_cop") < 50000)
    assert _volume(result.priced, eligible, "volume_weight") == pytest.approx(
        0.8 * _volume(result.baseline, eligible)
    )
    assert _volume(result.priced, ~eligible, "volume_weight") == pytest.approx(
        _volume(result.baseline, ~eligible)
    )


def test_migration_moves_exactly_its_share(simulator: Simulator) -> None:
    lever = {
        "kind": "product_migration",
        "from_product": "credit_standard",
        "to_product": "credit_premium",
        "share": 0.1,
    }
    result = simulator.run(_scenario("upgrade", lever))
    standard = pl.col("product").cast(pl.Utf8) == "credit_standard"
    premium = pl.col("product").cast(pl.Utf8) == "credit_premium"
    base_standard = _volume(result.baseline, standard)
    base_premium = _volume(result.baseline, premium)
    assert _volume(result.priced, standard, "volume_weight") == pytest.approx(0.9 * base_standard)
    assert _volume(result.priced, premium, "volume_weight") == pytest.approx(
        base_premium + 0.1 * base_standard
    )


def test_the_strategic_discount_touches_only_that_merchant(simulator: Simulator) -> None:
    lever = {"kind": "interchange_discount", "where": {"top_merchants_by_gdv": 1}, "share": 0.25}
    result = simulator.run(_scenario("strategic", lever))
    changed = _changed(result)
    assert changed.height > 0
    assert set(changed["merchant_id"].to_list()) == set(simulator.top_merchants(1))


def test_the_tokenized_discount_touches_only_tokenized_card_not_present(
    simulator: Simulator,
) -> None:
    lever = {
        "kind": "interchange_discount",
        "where": {"channels": ["cnp"], "tokenized": True},
        "rate_cut": 0.003,
    }
    result = simulator.run(_scenario("tokens", lever))
    changed = _changed(result)
    assert changed.height > 0
    assert changed["tokenized"].all()
    assert (changed["channel"].cast(pl.Utf8) == "cnp").all()
    assert result.priced["interchange_cop"].min() >= 0.0


# ---------------------------------------------------------------------------
# Pass-through
# ---------------------------------------------------------------------------


def test_blended_merchants_see_only_what_acquirers_pass_on(
    cfg: ProjectConfig, state: NetworkState
) -> None:
    cap = _scenario("cap", DEBIT_CAP)
    blended = pl.col("pricing_model").cast(pl.Utf8) == "blended"

    frozen = Simulator(state.with_config(_with_simulator(cfg, blended_pass_through=0.0))).run(cap)
    mask = frozen.baseline.select(blended).to_series().to_numpy()
    assert np.array_equal(
        frozen.priced["mdr_cop"].to_numpy()[mask], frozen.baseline["mdr_cop"].to_numpy()[mask]
    )

    full = Simulator(state.with_config(_with_simulator(cfg, blended_pass_through=1.0))).run(cap)
    rates = full.prices.filter(pl.col("price") == "blended_mdr_rate")
    for row in rates.iter_rows(named=True):
        group = blended & (pl.col("mcc_group").cast(pl.Utf8) == row["key"])
        before = full.baseline.filter(group)
        after = full.priced.filter(group)
        change = after["interchange_cop"].sum() / after["amount_cop"].sum() - (
            before["interchange_cop"].sum() / before["amount_cop"].sum()
        )
        assert row["scenario"] - row["base"] == pytest.approx(change, abs=1e-12), row["key"]


def test_rewards_never_go_below_zero(cfg: ProjectConfig, state: NetworkState) -> None:
    capped = Simulator(state.with_config(_with_simulator(cfg, rewards_pass_through=1.0))).run(
        _scenario("credit_cap", CREDIT_CAP)
    )
    rewards = capped.prices.filter(pl.col("price") == "rewards_rate")
    assert (rewards["scenario"] >= 0.0).all()
    assert rewards.filter(pl.col("key") == "credit_standard")["scenario"].item() == 0.0


def test_a_product_without_rewards_does_not_start_paying_them(cfg: ProjectConfig) -> None:
    products = cfg.economics.issuer.products
    assert products["debit"].rewards_rate == 0.0
    changes = {"debit": 0.01, "credit_standard": 0.01}
    repriced, _ = passed_through(_with_simulator(cfg, rewards_pass_through=0.5), {}, changes)
    after = repriced.economics.issuer.products
    assert after["debit"].rewards_rate == 0.0
    assert after["credit_standard"].rewards_rate == pytest.approx(
        products["credit_standard"].rewards_rate + 0.005
    )


def test_passing_a_credit_cap_on_to_rewards_softens_the_hit_to_issuers(
    state: NetworkState,
) -> None:
    table = sweep(
        state, _scenario("credit_cap", CREDIT_CAP), "simulator.rewards_pass_through", [0.0, 1.0]
    )
    contribution = dict(zip(table["value"], table["issuer_contribution_change_pct"], strict=True))
    assert contribution[1.0] > contribution[0.0]


# ---------------------------------------------------------------------------
# Validation, the pricing hook and the command line
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "levers, message",
    [
        ([{"kind": "price_hike"}], "expected tags"),
        ([{"kind": "volume_substitution", "where": {}, "share": 1.5}], "less than or equal to 1"),
        (
            [{"kind": "interchange_discount", "where": {}, "rate_cut": 0.001, "share": 0.1}],
            "exactly one",
        ),
        (
            [
                {
                    "kind": "product_migration",
                    "from_product": "debit",
                    "to_product": "debit",
                    "share": 0.1,
                }
            ],
            "different products",
        ),
    ],
)
def test_invalid_scenarios_fail(levers: list[dict], message: str) -> None:
    with pytest.raises(ValidationError, match=message):
        Scenario.model_validate({"name": "bad", "description": "Bad.", "levers": levers})


def test_scenarios_cannot_name_products_the_network_does_not_issue(cfg: ProjectConfig) -> None:
    scenario = _scenario(
        "prepaid_cap", {"kind": "interchange_cap", "products": ["prepaid"], "max_rate": 0.003}
    )
    with pytest.raises(ValueError, match="unknown products"):
        check_scenario(scenario, cfg)


def test_an_interchange_adjustment_keeps_the_fee_identity(
    cfg: ProjectConfig, data: GeneratedData
) -> None:
    table = build_interchange_table(cfg)
    transactions, merchants = data.tables["transactions"], data.tables["merchants"]
    plain = compute_pnl(transactions, table, cfg, merchants=merchants)
    same = compute_pnl(transactions, table, cfg, merchants=merchants, interchange_adjustment=None)
    assert plain.transactions.equals(same.transactions)
    halved = compute_pnl(
        transactions,
        table,
        cfg,
        merchants=merchants,
        interchange_adjustment=pl.col("interchange_cop") * 0.5,
    ).transactions
    identity = (
        pl.col("issuer_gross_cop")
        + pl.col("network_revenue_cop")
        + pl.col("acquirer_gross_cop")
        - pl.col("mdr_cop")
    )
    assert halved.select(identity.abs().max()).item() < 1e-6
    assert halved["interchange_cop"].sum() == pytest.approx(
        0.5 * plain.transactions["interchange_cop"].sum()
    )


def test_with_parameter_changes_one_field_only(cfg: ProjectConfig) -> None:
    changed = with_parameter(cfg, "simulator.rewards_pass_through", 0.0)
    assert changed.simulator.rewards_pass_through == 0.0
    assert changed.simulator.blended_pass_through == cfg.simulator.blended_pass_through
    with pytest.raises(KeyError):
        with_parameter(cfg, "simulator.no_such_parameter", 1.0)


def test_the_command_line_needs_a_scenario_or_all() -> None:
    parser = build_parser()
    assert parser.parse_args(["run", "--all"]).all
    with pytest.raises(SystemExit):
        parser.parse_args(["run"])
