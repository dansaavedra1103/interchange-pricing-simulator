"""The honest comparison: does the graph segmentation beat the attribute baseline?

Three criteria, from spec §2.5:

* **Homogeneity** — how much of the variation in effective interchange a segmentation
  explains, weighted by volume, because that is where the network's revenue sits.
* **Out-of-sample power** — relative burden is the spec's predictor of acceptance
  abandonment, so the business question is whether knowing a merchant's segment predicts the
  burden of a merchant the model has never seen. Measured per merchant, since dropping
  acceptance is a per-merchant decision. Feature 5 adds the churn curve itself as one more
  column here.
* **Interpretability** — segment sizes, silhouette, and how much the segmentation is just the
  MCC group relabelled.

In-sample homogeneity always improves with more segments, so it is read next to the
out-of-sample column, never alone.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import polars as pl

from ips.data_gen.diagnostics import normalized_mutual_info
from ips.graph.clustering import UNASSIGNED, Segmentation
from ips.utils.config import ProjectConfig

COMPARISON_COLUMNS = (
    "method",
    "segments",
    "coverage",
    "largest_segment_share",
    "eta2_interchange",
    "eta2_burden",
    "burden_r2_oos",
    "burden_mae_ratio",
    "burden_top_decile_lift",
    "silhouette",
    "nmi_mcc_group",
    "nmi_clientele",
)


def eta_squared(values: np.ndarray, labels: np.ndarray, weights: np.ndarray | None) -> float:
    """Share of the weighted variance of ``values`` that falls between segments."""
    weights = np.ones_like(values) if weights is None else weights
    total = np.sum(weights * (values - np.average(values, weights=weights)) ** 2)
    if total <= 0:
        return float("nan")
    within = 0.0
    for label in np.unique(labels):
        mask = labels == label
        mean = np.average(values[mask], weights=weights[mask])
        within += float(np.sum(weights[mask] * (values[mask] - mean) ** 2))
    return float(1.0 - within / total)


def _segment_means(values: np.ndarray, labels: np.ndarray) -> dict[int, float]:
    return {int(label): float(values[labels == label].mean()) for label in np.unique(labels)}


def out_of_sample(values: np.ndarray, labels: np.ndarray, cfg: ProjectConfig) -> dict[str, float]:
    """Predict held-out merchants with their segment's training mean.

    R² is against predicting the training mean for everyone: 0 means the segmentation adds
    nothing, 1 means it explains the held-out merchants exactly.
    """
    rng = np.random.default_rng(cfg.seed)
    order = rng.permutation(len(values))
    n_test = max(1, int(round(cfg.segmentation.test_fraction * len(values))))
    test, train = order[:n_test], order[n_test:]
    if len(train) == 0:
        raise ValueError("Not enough merchants to hold out a test set")

    baseline = float(values[train].mean())
    means = _segment_means(values[train], labels[train])
    predicted = np.array([means.get(int(label), baseline) for label in labels[test]])
    truth = values[test]

    residual = float(np.sum((truth - predicted) ** 2))
    reference = float(np.sum((truth - baseline) ** 2))
    mae = float(np.mean(np.abs(truth - predicted)))
    mae_reference = float(np.mean(np.abs(truth - baseline)))
    # ¿Sirve para encontrar a los comercios más cargados, que son los que dejan de aceptar?
    threshold = float(np.quantile(values[train], 0.9))
    heavy = truth >= threshold
    top = np.argsort(-predicted)[: max(1, int(round(0.1 * n_test)))]
    base_rate = float(heavy.mean())
    return {
        "burden_r2_oos": 1.0 - residual / reference if reference > 0 else float("nan"),
        "burden_mae_ratio": mae / mae_reference if mae_reference > 0 else float("nan"),
        "burden_top_decile_lift": float(heavy[top].mean() / base_rate)
        if base_rate > 0
        else float("nan"),
    }


def evaluate_segmentation(
    segmentation: Segmentation,
    profile: pl.DataFrame,
    cfg: ProjectConfig,
    latent: pl.DataFrame | None = None,
) -> dict[str, float | str]:
    """Every metric of one segmentation, on the merchants it actually labelled."""
    joined = profile.join(segmentation.labels, on="merchant_id", how="inner").filter(
        pl.col("segment") != UNASSIGNED
    )
    if joined.is_empty():
        raise ValueError(f"Segmentation {segmentation.name!r} labelled no merchant of the profile")
    labels = joined["segment"].to_numpy()
    volume = joined["gdv_cop"].to_numpy()
    interchange = joined["effective_interchange_rate"].to_numpy()
    burden = joined["relative_burden"].to_numpy()
    sizes = np.bincount(labels - labels.min())

    row: dict[str, float | str] = {
        "method": segmentation.name,
        "segments": float(len(np.unique(labels))),
        "coverage": float(joined.height / profile.height),
        "largest_segment_share": float(sizes.max() / sizes.sum()),
        "eta2_interchange": eta_squared(interchange, labels, volume),
        "eta2_burden": eta_squared(burden, labels, None),
        **out_of_sample(burden, labels, cfg),
        "silhouette": float(segmentation.silhouette),
        "nmi_mcc_group": normalized_mutual_info(joined["mcc_group"], joined["segment"]),
        "nmi_clientele": float("nan"),
    }
    if latent is not None:
        with_latent = joined.join(latent, on="merchant_id", how="inner")
        if not with_latent.is_empty():
            row["nmi_clientele"] = normalized_mutual_info(
                with_latent["clientele"], with_latent["segment"]
            )
    return row


def compare_segmentations(
    segmentations: Sequence[Segmentation],
    profile: pl.DataFrame,
    cfg: ProjectConfig,
    latent: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """One row per method, in the order given."""
    rows = [evaluate_segmentation(s, profile, cfg, latent) for s in segmentations]
    return pl.DataFrame(rows).select(COMPARISON_COLUMNS)


def verdict(comparison: pl.DataFrame, baseline: str = "baseline", challenger: str = "graph") -> str:
    """The sentence the notebook has to print: did the graph beat the baseline, or not?

    The criterion is the out-of-sample business metric, not the in-sample fit.
    """
    scores = dict(zip(comparison["method"], comparison["burden_r2_oos"], strict=True))
    if baseline not in scores or challenger not in scores:
        raise ValueError(f"Both {baseline!r} and {challenger!r} must be in the comparison")
    gap = scores[challenger] - scores[baseline]
    direction = "beats" if gap > 0 else "does not beat"
    return (
        f"The {challenger} segmentation {direction} the {baseline} on out-of-sample relative "
        f"burden: R² {scores[challenger]:.3f} vs {scores[baseline]:.3f} ({gap:+.3f})."
    )
