"""Parametrised samplers shared by the generator modules.

All functions are vectorised over numpy arrays and take an explicit ``numpy.random.Generator``
so that every draw is reproducible from the project seed.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def normalize(weights: Sequence[float] | np.ndarray) -> np.ndarray:
    """Scale non-negative weights so they sum to 1."""
    array = np.asarray(weights, dtype=np.float64)
    total = array.sum()
    if total <= 0:
        raise ValueError("weights must have a positive sum")
    return array / total


def categorical(
    rng: np.random.Generator, weights: Sequence[float] | np.ndarray, size: int
) -> np.ndarray:
    """Draw ``size`` category indices with probabilities proportional to ``weights``."""
    return rng.choice(len(weights), size=size, p=normalize(weights))


def lognormal_by_median(
    rng: np.random.Generator, median: float | np.ndarray, sigma: float | np.ndarray, size: int
) -> np.ndarray:
    """Lognormal draws parametrised by median and log-scale dispersion."""
    return np.asarray(median) * np.exp(np.asarray(sigma) * rng.standard_normal(size))


def dirichlet_rows(rng: np.random.Generator, alpha: np.ndarray) -> np.ndarray:
    """One Dirichlet draw per row of ``alpha`` (shape ``n x k``), via normalised gammas."""
    draws = rng.gamma(shape=alpha)
    return draws / draws.sum(axis=1, keepdims=True)


def row_categorical(
    rng: np.random.Generator, probabilities: np.ndarray, rows: np.ndarray
) -> np.ndarray:
    """For each entry of ``rows``, draw a column index from ``probabilities[row]``.

    Uses one ``searchsorted`` over the row-wise CDFs shifted by the row index, so it never
    materialises a ``len(rows) x k`` matrix (6M draws over 150k distributions stay cheap).
    """
    n_rows, n_cols = probabilities.shape
    cdf = np.cumsum(probabilities / probabilities.sum(axis=1, keepdims=True), axis=1)
    cdf[:, -1] = 1.0
    shifted = (cdf + np.arange(n_rows)[:, None]).ravel()
    targets = rows + rng.random(len(rows))
    flat = np.searchsorted(shifted, targets, side="right")
    return np.minimum(flat - rows * n_cols, n_cols - 1)


def quantile_tiers(weights: np.ndarray, tier_shares: Sequence[float]) -> np.ndarray:
    """Tier index for each weight: the top ``tier_shares[0]`` fraction gets tier 0, and so on."""
    order = np.argsort(-weights, kind="stable")
    bounds = np.round(np.cumsum(tier_shares) * len(weights)).astype(np.int64)
    tiers = np.empty(len(weights), dtype=np.int64)
    tiers[order] = np.searchsorted(bounds, np.arange(len(weights)), side="right")
    return np.minimum(tiers, len(tier_shares) - 1)


def allocate_counts(
    rng: np.random.Generator, total: int, weights: Sequence[float] | np.ndarray
) -> np.ndarray:
    """Split ``total`` events among entities proportionally to ``weights`` (multinomial)."""
    return rng.multinomial(total, normalize(weights))
