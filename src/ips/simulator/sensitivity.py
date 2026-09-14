"""Sensitivity: how a scenario's result moves when one assumption does.

Two tools, both built on re-running the engine:

* ``sweep`` walks one parameter through a list of values;
* ``tornado`` pushes each of several parameters to a low and a high value and ranks them by how
  far the result swings — the chart that shows which assumption a conclusion leans on.

Parameters are dotted paths into the configuration, for example
``simulator.rewards_pass_through`` or ``elasticity.acceptance.target_semi_elasticity``. The
state is rebuilt for every value, so assumptions that shape the base year — the acceptance curve
among them — are recalibrated rather than reused.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import polars as pl
from pydantic import BaseModel

from ips.simulator.engine import Simulator
from ips.simulator.scenario import Scenario
from ips.simulator.state import NetworkState
from ips.utils.config import ProjectConfig


def with_parameter(cfg: ProjectConfig, path: str, value: object) -> ProjectConfig:
    """A copy of the configuration with one dotted parameter replaced."""

    def replace(model: BaseModel, keys: list[str]) -> BaseModel:
        head, *rest = keys
        if head not in type(model).model_fields:
            raise KeyError(f"Unknown parameter {path!r}")
        if not rest:
            return model.model_copy(update={head: value})
        child = getattr(model, head)
        if not isinstance(child, BaseModel):
            raise KeyError(f"Unknown parameter {path!r}")
        return model.model_copy(update={head: replace(child, rest)})

    return replace(cfg, path.split("."))


def _summary(
    state: NetworkState, scenario: Scenario, parameter: str, value: float
) -> dict[str, float | str]:
    cfg = with_parameter(state.cfg, parameter, value)
    return Simulator(state.with_config(cfg)).run(scenario).summary()


def sweep(
    state: NetworkState, scenario: Scenario, parameter: str, values: Sequence[float]
) -> pl.DataFrame:
    """The scenario's summary at every value of one parameter."""
    rows = [
        {
            "parameter": parameter,
            "value": float(value),
            **_summary(state, scenario, parameter, value),
        }
        for value in values
    ]
    return pl.DataFrame(rows)


def tornado(
    state: NetworkState,
    scenario: Scenario,
    ranges: Mapping[str, tuple[float, float]],
    metric: str = "network_revenue_change_pct",
) -> pl.DataFrame:
    """How far ``metric`` swings as each parameter goes from its low to its high value."""
    center = Simulator(state).run(scenario).summary()[metric]
    rows = []
    for parameter, (low, high) in ranges.items():
        at_low = _summary(state, scenario, parameter, low)[metric]
        at_high = _summary(state, scenario, parameter, high)[metric]
        rows.append(
            {
                "parameter": parameter,
                "low": float(low),
                "high": float(high),
                "at_low": at_low,
                "at_high": at_high,
                "center": center,
                "swing": abs(at_high - at_low),
            }
        )
    return pl.DataFrame(rows).sort("swing", descending=True)
