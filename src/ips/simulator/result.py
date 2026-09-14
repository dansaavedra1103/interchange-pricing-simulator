"""What a scenario did, next to the network without it.

A result keeps the priced base year and the priced scenario, every transaction with its weight,
so any total, segment or actor can be read after the fact. The two checks that make a result
trustworthy live here too: the fees still add up to the MDR, and the volume the merchants lost
equals what moved to substitutes plus what left the card rails.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import polars as pl

from ips.economics.pnl import MONEY_COLUMNS
from ips.economics.revenue_bridge import BridgeResult
from ips.simulator.scenario import Scenario
from ips.utils.frames import stable_group_sums

WEIGHT = "weight"
METRICS = (
    "gdv_cop",
    "n_txns",
    "interchange_cop",
    "mdr_cop",
    "network_revenue_cop",
    "issuer_gross_cop",
    "acquirer_gross_cop",
    "rewards_cop",
    "issuer_contribution_cop",
    "acquirer_contribution_cop",
    "merchant_cost_cop",
)
SEGMENT_COLUMNS = (
    "gdv_cop",
    "interchange_cop",
    "mdr_cop",
    "network_revenue_cop",
    "issuer_contribution_cop",
)
_ACTORS = (
    ("issuer", "issuer_gross_cop", "issuer_contribution_cop"),
    ("acquirer", "acquirer_gross_cop", "acquirer_contribution_cop"),
    ("network", "network_revenue_cop", "network_revenue_cop"),
    ("merchant", "merchant_cost_cop", "merchant_cost_cop"),
)
_AMOUNT = pl.col("amount_cop").cast(pl.Float64)


def weighted_sums(frame: pl.DataFrame, keys: Sequence[str] = ()) -> pl.DataFrame:
    """Volume, weighted transaction count and every money column, in total or by segment."""
    weight = pl.col(WEIGHT)
    values = [
        (_AMOUNT * weight).alias("gdv_cop"),
        weight.alias("n_txns"),
        *((pl.col(column) * weight).alias(column) for column in MONEY_COLUMNS),
    ]
    if not keys:
        return frame.select([value.sum() for value in values])
    return stable_group_sums(frame, [pl.col(key).cast(pl.Utf8) for key in keys], *values)


@dataclass(frozen=True)
class SimulationResult:
    """One scenario's run: priced transactions with weights, next to the base year."""

    scenario: Scenario
    baseline: pl.DataFrame
    priced: pl.DataFrame
    acceptance: pl.DataFrame
    prices: pl.DataFrame
    lost_cop: float
    placed_cop: float
    leaked_cop: float
    bridge: BridgeResult
    seconds: float

    def totals(self) -> pl.DataFrame:
        """Every metric in the base year and in the scenario, with the change."""
        base = weighted_sums(self.baseline).row(0, named=True)
        new = weighted_sums(self.priced).row(0, named=True)
        return (
            pl.DataFrame(
                {
                    "metric": list(METRICS),
                    "baseline": [float(base[metric]) for metric in METRICS],
                    "scenario": [float(new[metric]) for metric in METRICS],
                }
            )
            .with_columns((pl.col("scenario") - pl.col("baseline")).alias("change"))
            .with_columns(
                pl.when(pl.col("baseline") != 0)
                .then(pl.col("change") / pl.col("baseline").abs())
                .otherwise(None)
                .alias("change_pct")
            )
        )

    def by_actor(self) -> pl.DataFrame:
        """Gross revenue and contribution of issuers, acquirers and the network; merchant cost."""
        totals = {row["metric"]: row for row in self.totals().iter_rows(named=True)}
        rows = [
            {
                "actor": actor,
                "revenue_base": totals[revenue]["baseline"],
                "revenue_scenario": totals[revenue]["scenario"],
                "contribution_base": totals[contribution]["baseline"],
                "contribution_scenario": totals[contribution]["scenario"],
            }
            for actor, revenue, contribution in _ACTORS
        ]
        return pl.DataFrame(rows).with_columns(
            (pl.col("revenue_scenario") - pl.col("revenue_base")).alias("revenue_change"),
            (pl.col("contribution_scenario") - pl.col("contribution_base")).alias(
                "contribution_change"
            ),
        )

    def by_segment(self, keys: Sequence[str] = ("product", "mcc_group")) -> pl.DataFrame:
        """Volume, fees and issuer contribution by segment, base year next to the scenario."""
        base = weighted_sums(self.baseline, keys).select(*keys, *SEGMENT_COLUMNS)
        new = weighted_sums(self.priced, keys).select(*keys, *SEGMENT_COLUMNS)
        return (
            base.join(new, on=list(keys), how="full", coalesce=True, suffix="_scenario")
            .with_columns(pl.col(pl.Float64).fill_null(0.0))
            .sort(list(keys))
        )

    def fee_gap(self) -> float:
        """Relative gap between the fees each side receives and the MDR merchants pay."""
        row = self.priced.select(
            (
                (
                    pl.col("issuer_gross_cop")
                    + pl.col("network_revenue_cop")
                    + pl.col("acquirer_gross_cop")
                    - pl.col("mdr_cop")
                )
                * pl.col(WEIGHT)
            )
            .sum()
            .alias("gap"),
            (pl.col("mdr_cop") * pl.col(WEIGHT)).sum().alias("mdr"),
        ).row(0, named=True)
        return abs(row["gap"]) / max(abs(row["mdr"]), 1.0)

    def volume_gap(self) -> float:
        """Relative gap between the volume lost and the volume moved plus leaked."""
        return abs(self.placed_cop + self.leaked_cop - self.lost_cop) / max(abs(self.lost_cop), 1.0)

    def conserves(self, tolerance: float = 1e-9) -> bool:
        """Both identities hold: fees add up to the MDR, and lost volume is accounted for."""
        return self.fee_gap() <= tolerance and self.volume_gap() <= tolerance

    def headline(self) -> str:
        """The scenario in one sentence, written from the numbers."""
        change = {row["metric"]: row["change_pct"] for row in self.totals().iter_rows(named=True)}

        def pct(metric: str) -> str:
            value = change[metric]
            return "n/a" if value is None else f"{value:+.2%}"

        return (
            f"{self.scenario.description} Network revenue {pct('network_revenue_cop')}, issuer "
            f"contribution {pct('issuer_contribution_cop')}, acquirer contribution "
            f"{pct('acquirer_contribution_cop')}, merchant cost {pct('merchant_cost_cop')} and "
            f"card volume {pct('gdv_cop')}."
        )

    def summary(self) -> dict[str, float | str]:
        """One row for the table that compares scenarios."""
        totals = {row["metric"]: row for row in self.totals().iter_rows(named=True)}
        return {
            "scenario": self.scenario.name,
            "network_revenue_change_cop": totals["network_revenue_cop"]["change"],
            "network_revenue_change_pct": totals["network_revenue_cop"]["change_pct"],
            "issuer_contribution_change_pct": totals["issuer_contribution_cop"]["change_pct"],
            "acquirer_contribution_change_pct": totals["acquirer_contribution_cop"]["change_pct"],
            "merchant_cost_change_pct": totals["merchant_cost_cop"]["change_pct"],
            "gdv_change_pct": totals["gdv_cop"]["change_pct"],
            "net_merchants_abandoning": float(self.acceptance["delta_p"].sum()),
            "volume_lost_cop": self.lost_cop,
            "volume_moved_cop": self.placed_cop,
            "volume_leaked_cop": self.leaked_cop,
            "seconds": self.seconds,
            "headline": self.headline(),
        }

    def write(self, directory: Path) -> dict[str, Path]:
        """Write the result's tables under ``directory/<scenario name>/``."""
        target = Path(directory) / self.scenario.name
        target.mkdir(parents=True, exist_ok=True)
        frames = {
            "totals": self.totals(),
            "by_actor": self.by_actor(),
            "by_segment": self.by_segment(),
            "acceptance": self.acceptance,
            "prices": self.prices,
            "bridge": self.bridge.frame(),
        }
        paths = {}
        for name, frame in frames.items():
            paths[name] = target / f"{name}.parquet"
            frame.write_parquet(paths[name])
        return paths
