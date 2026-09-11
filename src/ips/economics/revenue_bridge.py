"""Revenue bridge: why network revenue changed between two periods or two scenarios.

Network revenue is R = V * sum_r x_r * sum_{s in r} w_{s|r} * y_s, where V is total volume,
x the share of each region (domestic or cross-border), w the mix of segments within a region
and y the network yield of each segment. The change R1 - R0 is split into four drivers with
Shapley values over the 16 combinations of old and new factors: the effects add up exactly to
the change and do not depend on the order in which the drivers are considered.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import polars as pl

DRIVERS = ("volume", "cross_border", "mix", "rate")
SEGMENT_KEYS = ("product", "mcc_group", "channel")


@dataclass(frozen=True)
class BridgeResult:
    """Network revenue in both periods and the effect of each driver (COP)."""

    base_revenue: float
    new_revenue: float
    effects: dict[str, float]

    @property
    def change(self) -> float:
        """Total change, equal to the sum of the effects."""
        return self.new_revenue - self.base_revenue

    def frame(self) -> pl.DataFrame:
        """Waterfall rows: the base, one row per driver, and the new revenue."""
        return pl.DataFrame(
            {
                "step": ["base", *DRIVERS, "new"],
                "value_cop": [
                    self.base_revenue,
                    *(self.effects[driver] for driver in DRIVERS),
                    self.new_revenue,
                ],
            }
        )


def _cells(transactions: pl.DataFrame, keys: list[str]) -> pl.DataFrame:
    return transactions.group_by("cross_border", *keys).agg(
        pl.col("amount_cop").sum().cast(pl.Float64).alias("volume"),
        pl.col("network_revenue_cop").sum().alias("revenue"),
    )


def revenue_bridge(
    base: pl.DataFrame, new: pl.DataFrame, keys: Sequence[str] = SEGMENT_KEYS
) -> BridgeResult:
    """Split the change in network revenue from ``base`` to ``new`` into its drivers.

    Both frames are priced transactions (``compute_pnl(...).transactions``), for example two
    years of the same dataset or the baseline and a scenario.
    """
    keys = list(keys)
    cells = (
        _cells(base, keys)
        .join(
            _cells(new, keys), on=["cross_border", *keys], how="full", coalesce=True, suffix="_new"
        )
        .fill_null(0.0)
    )
    cross_border = cells["cross_border"].to_numpy().astype(bool)
    volume = {0: cells["volume"].to_numpy(), 1: cells["volume_new"].to_numpy()}
    revenue = {0: cells["revenue"].to_numpy(), 1: cells["revenue_new"].to_numpy()}
    total = {p: float(volume[p].sum()) for p in (0, 1)}
    if total[0] <= 0 or total[1] <= 0:
        raise ValueError("Both periods need purchases to build a bridge")

    region_volume = {
        p: np.where(cross_border, volume[p][cross_border].sum(), volume[p][~cross_border].sum())
        for p in (0, 1)
    }
    share = {p: region_volume[p] / total[p] for p in (0, 1)}
    with np.errstate(divide="ignore", invalid="ignore"):
        mix = {
            p: np.where(region_volume[p] > 0, volume[p] / region_volume[p], np.nan) for p in (0, 1)
        }
        rate = {p: np.where(volume[p] > 0, revenue[p] / volume[p], np.nan) for p in (0, 1)}
    # Un segmento (o una región) que falta en un período toma el valor del otro: ahí su peso
    # es cero, así que solo importa en las combinaciones mixtas del cálculo.
    for mine, other in ((0, 1), (1, 0)):
        mix[mine] = np.where(np.isnan(mix[mine]), mix[other], mix[mine])
        rate[mine] = np.where(np.isnan(rate[mine]), rate[other], rate[mine])

    def evaluate(volume_p: int, share_p: int, mix_p: int, rate_p: int) -> float:
        return float(total[volume_p] * np.sum(share[share_p] * mix[mix_p] * rate[rate_p]))

    n = len(DRIVERS)
    effects = {}
    for i, driver in enumerate(DRIVERS):
        others = [j for j in range(n) if j != i]
        effect = 0.0
        for size in range(n):
            weight = math.factorial(size) * math.factorial(n - size - 1) / math.factorial(n)
            for subset in itertools.combinations(others, size):
                with_driver = [int(j in subset or j == i) for j in range(n)]
                without_driver = [int(j in subset) for j in range(n)]
                effect += weight * (evaluate(*with_driver) - evaluate(*without_driver))
        effects[driver] = effect
    return BridgeResult(evaluate(0, 0, 0, 0), evaluate(1, 1, 1, 1), effects)
