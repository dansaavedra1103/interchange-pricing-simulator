"""Solving the acceptance curve's two parameters against two anchor figures.

There are no abandonment events in the data to fit, so the curve is not estimated. What can be
done honestly is **moment matching**: choose ``alpha`` and ``beta`` so that, evaluated over the
population's real distribution of relative burden, the curve reproduces two target figures —
both assumptions of our own until a published source replaces them:

1. the share of merchants that stop accepting cards in a year — this fixes the level;
2. the acceptance lost when the MDR rises by ``semi_elasticity_shock`` — this fixes the slope.

Both moments are monotone in their parameter, so nested bisection solves them: ``alpha`` in the
inner loop keeps the level right for whatever ``beta`` the outer loop is trying. The result is
deterministic — no seed, no random start — and the achieved moments travel with the parameters
so the notebook can show the calibration landed.

A single pair is solved for the whole population, not one pair per group: the anchors are
population-level, and forcing them group by group would give every group the same abandonment
rate, erasing exactly the heterogeneity the curve exists to express. Groups differ through
``base_elasticity`` (see :func:`ips.elasticity.params.group_sensitivity`) and through where
their merchants actually sit in the burden distribution.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import polars as pl

from ips.elasticity.merchant_acceptance import (
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


def _mean_probability(
    term: np.ndarray, offset: np.ndarray, alpha: float, beta: float, cfg: ProjectConfig
) -> float:
    """Population abandonment rate of the **public** curve, bounds included.

    Se calibra sobre la curva recortada y no sobre la sigmoide cruda porque es la recortada la
    que usa todo lo demás: si el ancla se cumpliera solo antes de aplicar las cotas, el número
    que el simulador consume no sería el calibrado. El recorte es monótono, así que la bisección
    sigue siendo válida.
    """
    return float(probability_from_terms(term, offset, alpha, beta, cfg).mean())


def _solve_alpha(
    term: np.ndarray, offset: np.ndarray, beta: float, target_rate: float, cfg: ProjectConfig
) -> float:
    """The intercept that puts the mean abandonment on its anchor, for a given slope.

    El intervalo se amplía hasta encerrar el ancla: con pendientes grandes el intercepto
    necesario crece en proporción, y un intervalo fijo devolvería uno de sus extremos en
    silencio, con una curva que no cumple la tasa base.
    """

    def mean(alpha: float) -> float:
        return _mean_probability(term, offset, alpha, beta, cfg)

    low, high = -_ALPHA_START, _ALPHA_START
    while mean(low) > target_rate:
        low *= 2.0
        if low < -_ALPHA_CAP:
            raise ValueError("No intercept brings the abandonment rate down to its anchor")
    while mean(high) < target_rate:
        high *= 2.0
        if high > _ALPHA_CAP:
            raise ValueError("No intercept brings the abandonment rate up to its anchor")
    return _bisect(mean, low, high, target_rate)


def _semi_elasticity(
    term: np.ndarray,
    offset: np.ndarray,
    alpha: float,
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
    target_rate: float,
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
        alpha = _solve_alpha(term, offset, high, target_rate, cfg)
        if _semi_elasticity(term, offset, alpha, high, shock, cfg) >= target_semi:
            return high
        high *= 2.0
    raise ValueError(
        "No slope reproduces acceptance.target_semi_elasticity "
        f"({target_semi}) at a base rate of {target_rate}. The two anchors are inconsistent "
        "with the population's burden distribution; review them in config/elasticity.yaml."
    )


def calibrate_acceptance(profile: pl.DataFrame, cfg: ProjectConfig) -> AcceptanceCurve:
    """Solve the curve against its two anchors on this population."""
    acceptance = cfg.elasticity.acceptance
    term = burden_term(profile, cfg)
    offset = logit_offset(profile, cfg)
    if term.size == 0:
        raise ValueError("Cannot calibrate the acceptance curve on an empty population")

    target_rate = acceptance.target_annual_abandonment
    target_semi = acceptance.target_semi_elasticity
    shock = acceptance.semi_elasticity_shock

    high = _bracket_beta(term, offset, target_rate, target_semi, shock, cfg)
    beta = _bisect(
        lambda candidate: _semi_elasticity(
            term,
            offset,
            _solve_alpha(term, offset, candidate, target_rate, cfg),
            candidate,
            shock,
            cfg,
        ),
        0.0,
        high,
        target_semi,
    )
    alpha = _solve_alpha(term, offset, beta, target_rate, cfg)

    base = probability_from_terms(term, offset, alpha, beta, cfg)
    after = probability_from_terms(term * (1.0 + shock), offset, alpha, beta, cfg)
    return AcceptanceCurve(
        alpha=alpha,
        beta=beta,
        achieved_abandonment=float(base.mean()),
        achieved_semi_elasticity=float(after.mean() - base.mean()),
        target_abandonment=target_rate,
        target_semi_elasticity=target_semi,
    )
