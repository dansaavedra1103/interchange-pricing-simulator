"""The industry baseline, and the benchmark that needs no model at all.

The baseline is k-means on the attributes a bank already has about a merchant: MCC group, size
tier, channel, volume, ticket and the card mix of its customers. No graph anywhere.

The benchmark is simpler still: leave every merchant in its own MCC group. It costs nothing,
and on this data it predicts a merchant's relative burden better than either k-means, so every
segmentation is read against it.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from ips.graph.clustering import Segmentation, cluster_merchants, combine_spaces
from ips.utils.config import MERCHANT_CHANNELS, SIZE_TIERS, ProjectConfig

BENCHMARK = "mcc_group"


def feature_matrix(profile: pl.DataFrame, cfg: ProjectConfig) -> tuple[np.ndarray, list[str]]:
    """Standardised continuous features beside one-hot categories, both blocks weighing the same.

    Sin equilibrar los bloques, las variables continuas estandarizadas dominan la distancia y la
    categoría del comercio casi no cuenta, que es justo la que manda en su economía.
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
            (pl.col(column).cast(pl.Utf8) == value).cast(pl.Float64).alias(f"{column}={value}")
            for column, values in categories.items()
            for value in values
        ]
    )
    matrix = combine_spaces(StandardScaler().fit_transform(numeric.to_numpy()), one_hot.to_numpy())
    return matrix, [*numeric.columns, *one_hot.columns]


def fit_baseline(profile: pl.DataFrame, cfg: ProjectConfig, k: int | None = None) -> Segmentation:
    """k-means on the tabular features of the merchants being segmented."""
    matrix, _ = feature_matrix(profile, cfg)
    return cluster_merchants("baseline", profile["merchant_id"].to_numpy(), matrix, cfg, k)


def segment_by_mcc_group(profile: pl.DataFrame) -> Segmentation:
    """The benchmark with no model: every merchant keeps its own MCC group as its segment."""
    groups = profile["mcc_group"].cast(pl.Utf8)
    labels = pl.DataFrame(
        {
            "merchant_id": profile["merchant_id"],
            "segment": groups.cast(pl.Categorical).to_physical().cast(pl.Int32),
        }
    ).sort("merchant_id")
    return Segmentation(
        name=BENCHMARK,
        labels=labels,
        k=int(groups.n_unique()),
        silhouette=float("nan"),
        scores=pl.DataFrame(schema={"k": pl.Int64, "score": pl.Float64}),
    )
