"""Elasticity parameters: the models live in ``ips.utils.config``, the derived ones here.

The curve's two free parameters are *not* configured — they are solved in
:mod:`ips.elasticity.calibration` so the curve reproduces two anchor figures. What is
configured is everything that shapes it around those anchors: the modifiers, the bounds, and
how much each MCC group reacts relative to the others.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from ips.utils.config import (
    AcceptanceConfig,
    CardholderResponseConfig,
    ElasticityConfig,
    GnnConfig,
    LinkPredictionConfig,
    ProjectConfig,
    RedistributionConfig,
)

__all__ = [
    "AcceptanceConfig",
    "CardholderResponseConfig",
    "ElasticityConfig",
    "GnnConfig",
    "LinkPredictionConfig",
    "RedistributionConfig",
    "group_sensitivity",
    "sensitivity_column",
]


def group_sensitivity(cfg: ProjectConfig) -> dict[str, float]:
    """How much each MCC group reacts to the same relative burden, relative to the median group.

    Sale de ``base_elasticity`` en ``mcc_groups.yaml``, normalizada por su mediana: la curva
    tiene un solo par de parámetros calibrados para toda la población, y la heterogeneidad
    entre grupos viene de un parámetro de negocio ya documentado, no de anclas que no tenemos
    por grupo.
    """
    elasticities = {name: group.base_elasticity for name, group in cfg.mcc_groups.groups.items()}
    if not cfg.elasticity.acceptance.use_group_elasticity:
        return dict.fromkeys(elasticities, 1.0)
    median = float(np.median(list(elasticities.values())))
    if median <= 0:
        raise ValueError("Median base_elasticity must be positive")
    return {name: value / median for name, value in elasticities.items()}


def sensitivity_column(cfg: ProjectConfig) -> pl.Expr:
    """``mcc_group`` mapped to its sensitivity multiplier."""
    return (
        pl.col("mcc_group")
        .cast(pl.Utf8)
        .replace_strict(group_sensitivity(cfg), return_dtype=pl.Float64)
        .alias("group_sensitivity")
    )
