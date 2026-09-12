"""Cardholder side of the loop: what happens to spend when the issuer moves rewards.

The chain the simulator will close (spec §2.6) runs: interchange falls -> the issuer earns
less -> it funds fewer rewards -> the cardholder spends less on the card. This module is the
last link, as a log-linear response:

    spend_after / spend_before = exp( epsilon_segment * change_in_rewards_rate )

``epsilon`` is a semi-elasticity in the units the literature reports it: points of spend per
point of rewards rate. High-spend cardholders carry the largest one because they are the ones
who actually optimise rewards.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from ips.utils.config import ProjectConfig

REQUIRED_COLUMNS = ("cardholder_id", "spend_segment")


def spend_multiplier(
    rewards_rate_change: float | np.ndarray, epsilon: float | np.ndarray
) -> np.ndarray:
    """Spend after over spend before, elementwise over semi-elasticities.

    Exponencial y no lineal: mantiene el multiplicador positivo por grande que sea el recorte,
    y hace que subir y bajar la misma cantidad se compongan de vuelta a 1.
    """
    return np.exp(
        np.asarray(epsilon, dtype=np.float64) * np.asarray(rewards_rate_change, dtype=np.float64)
    )


def _bounded(multiplier: np.ndarray, cfg: ProjectConfig) -> np.ndarray:
    """Keep the response inside its documented bound: some spend does not follow incentives."""
    limit = cfg.elasticity.cardholder.max_spend_change
    return np.clip(multiplier, 1.0 - limit, 1.0 + limit)


def cardholder_response(
    cardholders: pl.DataFrame, rewards_rate_change: float, cfg: ProjectConfig
) -> pl.DataFrame:
    """Per cardholder: its semi-elasticity and the multiplier its spend takes."""
    missing = [c for c in REQUIRED_COLUMNS if c not in cardholders.columns]
    if missing:
        raise ValueError(f"cardholders is missing required columns: {missing}")
    elasticities = dict(cfg.elasticity.cardholder.rewards_semi_elasticity)
    frame = cardholders.select(
        "cardholder_id",
        "spend_segment",
        pl.col("spend_segment")
        .cast(pl.Utf8)
        .replace_strict(elasticities, return_dtype=pl.Float64)
        .alias("rewards_semi_elasticity"),
    )
    multiplier = spend_multiplier(rewards_rate_change, frame["rewards_semi_elasticity"].to_numpy())
    return frame.with_columns(
        pl.lit(rewards_rate_change).alias("rewards_rate_change"),
        pl.Series("spend_multiplier", _bounded(multiplier, cfg)),
    )


def segment_response(cfg: ProjectConfig, rewards_rate_change: float) -> pl.DataFrame:
    """The same response by spend segment, for the notebook: one row per segment."""
    elasticities = cfg.elasticity.cardholder.rewards_semi_elasticity
    segments = list(elasticities)
    multiplier = _bounded(
        spend_multiplier(
            rewards_rate_change, np.array([elasticities[s] for s in segments], dtype=np.float64)
        ),
        cfg,
    )
    return pl.DataFrame(
        {
            "spend_segment": segments,
            "rewards_semi_elasticity": [elasticities[s] for s in segments],
            "rewards_rate_change": [rewards_rate_change] * len(segments),
            "spend_multiplier": multiplier,
        }
    )
