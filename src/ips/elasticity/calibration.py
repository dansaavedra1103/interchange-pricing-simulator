"""Solving the acceptance curve against two anchor figures, with one level per MCC group.

There are no abandonment events in the data to fit, so the curve is not estimated. What can be
done honestly is **moment matching**: choose the parameters so that, evaluated over the
population's real distribution of relative burden, the curve reproduces two target figures —
both assumptions of our own until a published source replaces them:

1. the share of merchants that stop accepting cards in a year — this fixes the **levels**;
2. the acceptance lost when the MDR rises by ``semi_elasticity_shock`` — this fixes the
   **slope**.

**One intercept per MCC group, one slope for everyone.** With a single intercept, the slope that
meets the second anchor comes out so steep that the curve behaves like a threshold: most groups
sit at the probability floor and do not respond to price at all, which an optimiser would read
as free room to raise their rates. Anchoring each group at its own level spreads the response
across the population. The levels are not free parameters either: each group's target scales
with its median relative burden raised to ``level_burden_exponent``, normalised so that the
population as a whole still sits on the first anchor. Heavier burden then means more abandonment
between sectors as well as within them; an exponent of zero puts every group on the same level.

Each group's level is monotone in its intercept, and the population's sensitivity grows with
the slope, so nested bisection solves them: the intercepts in the inner loop keep every group on
its level for whatever slope the outer loop is trying. The result is deterministic — no seed, no
random start — and the achieved moments travel with the parameters so the notebook can show the
calibration landed.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import polars as pl

from ips.elasticity.merchant_acceptance import (
    BURDEN,
    AcceptanceCurve,
    burden_term,
    logit_offset,
    probability_from_terms,
)
from ips.utils.config import ProjectConfig

_ALPHA_START = 1.0
_ALPHA_CAP = 1e9
_BETA_START = 1.0
_BETA_CAP = 1e6
_TOLERANCE = 1e-10
_MAX_ITERATIONS = 200


def _bisect(function: Callable[[float], float], low: float, high: float, target: float) -> float:
    """Solve ``function(x) == target`` for an increasing function, by bisection.

    Iteración fija y sin aleatoriedad: dos corridas dan el mismo número bit a bit.
    """
    for _ in range(_MAX_ITERATIONS):
        middle = 0.5 * (low + high)
        value = function(middle)
        if abs(value - target) < _TOLERANCE or high - low < _TOLERANCE:
            return middle
        if value < target:
            low = middle
        else:
            high = middle
    return 0.5 * (low + high)


def level_targets(profile: pl.DataFrame, cfg: ProjectConfig) -> dict[str, float]:
    """Abandonment level each MCC group is anchored to, before anything is fitted.

    El nivel de cada grupo escala con la mediana de su carga relativa: con exponente 1, un
    sector que entrega el doble de su margen en MDR abandona el doble. Los niveles se normalizan
    con el peso de cada grupo en la población, así que la tasa agregada sigue siendo la del ancla.
    """
    acceptance = cfg.elasticity.acceptance
    if profile.is_empty():
        raise ValueError("Cannot set group levels on an empty population")
    overall = float(profile[BURDEN].median())
    if overall <= 0:
        raise ValueError("The population's median relative burden must be positive")
    groups = (
        profile.group_by(pl.col("mcc_group").cast(pl.Utf8))
        .agg(pl.len().alias("merchants"), pl.col(BURDEN).median().alias("median_burden"))
        .sort("mcc_group")
    )
    ratio = (groups["median_burden"].to_numpy() / overall) ** acceptance.level_burden_exponent
    share = groups["merchants"].to_numpy() / profile.height
    targets = acceptance.target_annual_abandonment * ratio / float((share * ratio).sum())
    names = groups["mcc_group"].to_list()
    outside = [
        name
        for name, target in zip(names, targets, strict=True)
        if not acceptance.min_probability < target < acceptance.max_probability
    ]
    if outside:
        raise ValueError(
            f"Group levels fall outside the probability bounds for {outside}; lower "
            "acceptance.level_burden_exponent or widen the bounds in config/elasticity.yaml."
        )
    return {name: float(target) for name, target in zip(names, targets, strict=True)}


def _mean_probability(
    term: np.ndarray,
    offset: np.ndarray,
    alpha: float | np.ndarray,
    beta: float,
    cfg: ProjectConfig,
) -> float:
    """Mean abandonment of the **public** curve, bounds included.

    Se calibra sobre la curva recortada y no sobre la sigmoide cruda porque es la recortada la
    que usa todo lo demás: si el ancla se cumpliera solo antes de aplicar las cotas, el número
    que el simulador consume no sería el calibrado. El recorte es monótono, así que la bisección
    sigue siendo válida.
    """
    return float(probability_from_terms(term, offset, alpha, beta, cfg).mean())


def _solve_alpha(
    term: np.ndarray, offset: np.ndarray, beta: float, target_rate: float, cfg: ProjectConfig
) -> float:
    """The intercept that puts one group's mean abandonment on its level, for a given slope.

    El intervalo se amplía hasta encerrar el nivel: con pendientes grandes el intercepto
    necesario crece en proporción, y un intervalo fijo devolvería uno de sus extremos en
    silencio, con un grupo que no cumple su nivel.
    """

    def mean(alpha: float) -> float:
        return _mean_probability(term, offset, alpha, beta, cfg)

    low, high = -_ALPHA_START, _ALPHA_START
    while mean(low) > target_rate:
        low *= 2.0
        if low < -_ALPHA_CAP:
            raise ValueError("No intercept brings a group's abandonment down to its level")
    while mean(high) < target_rate:
        high *= 2.0
        if high > _ALPHA_CAP:
            raise ValueError("No intercept brings a group's abandonment up to its level")
    return _bisect(mean, low, high, target_rate)


def _group_rows(profile: pl.DataFrame, names: list[str]) -> dict[str, np.ndarray]:
    """Row positions of each MCC group inside the profile."""
    labels = profile["mcc_group"].cast(pl.Utf8).to_numpy()
    return {name: np.flatnonzero(labels == name) for name in names}


def _intercepts(
    term: np.ndarray,
    offset: np.ndarray,
    rows: dict[str, np.ndarray],
    targets: dict[str, float],
    beta: float,
    cfg: ProjectConfig,
) -> np.ndarray:
    """Per merchant: the intercept of its group, each solved for that group's own level."""
    alpha = np.empty(term.size, dtype=np.float64)
    for name, positions in rows.items():
        alpha[positions] = _solve_alpha(
            term[positions], offset[positions], beta, targets[name], cfg
        )
    return alpha


def _semi_elasticity(
    term: np.ndarray,
    offset: np.ndarray,
    alpha: np.ndarray,
    beta: float,
    shock: float,
    cfg: ProjectConfig,
) -> float:
    """Acceptance lost when every merchant's MDR rises by ``shock`` (relative)."""
    base = _mean_probability(term, offset, alpha, beta, cfg)
    after = _mean_probability(term * (1.0 + shock), offset, alpha, beta, cfg)
    return after - base


def _bracket_beta(
    term: np.ndarray,
    offset: np.ndarray,
    rows: dict[str, np.ndarray],
    targets: dict[str, float],
    target_semi: float,
    shock: float,
    cfg: ProjectConfig,
) -> float:
    """Grow the slope until it produces at least the target sensitivity.

    En beta = 0 la curva no depende de la carga y la semi-elasticidad es exactamente 0, así que
    el extremo inferior siempre sirve; el superior se busca duplicando.
    """
    high = _BETA_START
    while high <= _BETA_CAP:
        alpha = _intercepts(term, offset, rows, targets, high, cfg)
        if _semi_elasticity(term, offset, alpha, high, shock, cfg) >= target_semi:
            return high
        high *= 2.0
    raise ValueError(
        f"No slope reproduces acceptance.target_semi_elasticity ({target_semi}) with these "
        "group levels. The two anchors are inconsistent with the population's burden "
        "distribution; review them in config/elasticity.yaml."
    )


def calibrate_acceptance(profile: pl.DataFrame, cfg: ProjectConfig) -> AcceptanceCurve:
    """Solve the group intercepts and the shared slope against both anchors on this population."""
    acceptance = cfg.elasticity.acceptance
    term = burden_term(profile, cfg)
    offset = logit_offset(profile, cfg)
    if term.size == 0:
        raise ValueError("Cannot calibrate the acceptance curve on an empty population")

    targets = level_targets(profile, cfg)
    rows = _group_rows(profile, list(targets))
    target_semi = acceptance.target_semi_elasticity
    shock = acceptance.semi_elasticity_shock

    def sensitivity(beta: float) -> float:
        alpha = _intercepts(term, offset, rows, targets, beta, cfg)
        return _semi_elasticity(term, offset, alpha, beta, shock, cfg)

    high = _bracket_beta(term, offset, rows, targets, target_semi, shock, cfg)
    beta = _bisect(sensitivity, 0.0, high, target_semi)
    alpha = _intercepts(term, offset, rows, targets, beta, cfg)

    base = probability_from_terms(term, offset, alpha, beta, cfg)
    after = probability_from_terms(term * (1.0 + shock), offset, alpha, beta, cfg)
    return AcceptanceCurve(
        intercepts={name: float(alpha[positions[0]]) for name, positions in rows.items()},
        beta=beta,
        level_targets=targets,
        achieved_levels={name: float(base[positions].mean()) for name, positions in rows.items()},
        achieved_abandonment=float(base.mean()),
        achieved_semi_elasticity=float(after.mean() - base.mean()),
        target_abandonment=acceptance.target_annual_abandonment,
        target_semi_elasticity=target_semi,
    )
