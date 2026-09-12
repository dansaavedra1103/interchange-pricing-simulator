"""The industry baseline: k-means on the attributes a bank already has about a merchant.

MCC group, size tier, channel, volume, ticket and the card mix of its customers. No graph
anywhere: this is what the segmentation on the graph has to beat on business metrics.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from ips.graph.clustering import Segmentation, cluster_merchants
from ips.utils.config import MERCHANT_CHANNELS, SIZE_TIERS, ProjectConfig


def feature_matrix(profile: pl.DataFrame, cfg: ProjectConfig) -> tuple[np.ndarray, list[str]]:
    """Standardised tabular features, plus one-hot categories (left unscaled).

    Escalar los indicadores 0/1 daría peso enorme a las categorías raras, así que solo se
    estandarizan las variables continuas.
    """
    from sklearn.preprocessing import StandardScaler

    numeric = profile.select(
        pl.col("gdv_cop").log10().alias("log_gdv"),
        pl.col("avg_ticket_cop").log10().alias("log_avg_ticket"),
        pl.col("n_customers").log10().alias("log_customers"),
        pl.col("cnp_share"),
        pl.col("cross_border_share"),
        *(pl.col(f"share_{product}") for product in cfg.products),
    )
    categories = {
        "mcc_group": cfg.group_names,
        "size_tier": SIZE_TIERS,
        "merchant_channel": MERCHANT_CHANNELS,
    }
    one_hot = profile.select(
        [
            (pl.col(column).cast(pl.Utf8) == value).cast(pl.Float64).alias(f"{column}_{value}")
            for column, values in categories.items()
            for value in values
        ]
    )
    matrix = np.hstack([StandardScaler().fit_transform(numeric.to_numpy()), one_hot.to_numpy()])
    return matrix, [*numeric.columns, *one_hot.columns]


def fit_baseline(profile: pl.DataFrame, cfg: ProjectConfig, k: int | None = None) -> Segmentation:
    """k-means on the tabular features of the merchants being segmented."""
    matrix, _ = feature_matrix(profile, cfg)
    return cluster_merchants("baseline", profile["merchant_id"].to_numpy(), matrix, cfg, k)
