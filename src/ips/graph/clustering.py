"""k-means over any merchant representation, with k chosen by silhouette.

The tabular baseline and the graph embeddings share this module, so the only difference
between them in the comparison is the space the merchants live in.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

from ips.utils.config import ProjectConfig

UNASSIGNED = -1


@dataclass(frozen=True)
class Segmentation:
    """Merchants labelled by one method, with how the number of segments was chosen."""

    name: str
    labels: pl.DataFrame  # merchant_id, segment
    k: int
    silhouette: float
    scores: pl.DataFrame  # k, score of the search that chose k (empty when k was given)

    def segment_of(self) -> dict[int, int]:
        """Segment by merchant id."""
        return dict(self.labels.iter_rows())

    @property
    def coverage(self) -> float:
        """Share of merchants that got a segment (Leiden can leave isolates out)."""
        return float((self.labels["segment"] != UNASSIGNED).mean())


def combine_spaces(*blocks: np.ndarray) -> np.ndarray:
    """Stack representations side by side, each scaled to the same average row norm.

    Sin escalar, el bloque con más columnas domina la distancia y el otro deja de contar.
    """
    scaled = []
    for block in blocks:
        norm = float(np.linalg.norm(block, axis=1).mean())
        scaled.append(block / norm if norm > 0 else block)
    return np.hstack(scaled)


def _labels_frame(merchant_ids: np.ndarray, segments: np.ndarray) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "merchant_id": pl.Series(merchant_ids, dtype=pl.UInt32),
            "segment": pl.Series(segments, dtype=pl.Int32),
        }
    ).sort("merchant_id")


def _silhouette(matrix: np.ndarray, labels: np.ndarray, cfg: ProjectConfig) -> float:
    from sklearn.metrics import silhouette_score

    if len(set(labels.tolist())) < 2:
        return float("nan")
    sample = min(cfg.segmentation.silhouette_sample, len(labels))
    return float(silhouette_score(matrix, labels, sample_size=sample, random_state=cfg.seed))


def _fit(matrix: np.ndarray, k: int, cfg: ProjectConfig) -> np.ndarray:
    from sklearn.cluster import KMeans

    model = KMeans(n_clusters=k, random_state=cfg.seed, n_init=10)
    return model.fit_predict(matrix)


def select_k(matrix: np.ndarray, cfg: ProjectConfig) -> tuple[int, pl.DataFrame]:
    """Try every configured k and keep the one with the best silhouette."""
    rows = []
    for k in cfg.segmentation.k_values:
        if k >= matrix.shape[0]:
            continue
        rows.append({"k": k, "score": _silhouette(matrix, _fit(matrix, k, cfg), cfg)})
    if not rows:
        raise ValueError("No candidate k fits this population")
    scores = pl.DataFrame(rows)
    best = scores.sort("score", descending=True)["k"].first()
    return int(best), scores


def cluster_merchants(
    name: str,
    merchant_ids: np.ndarray,
    matrix: np.ndarray,
    cfg: ProjectConfig,
    k: int | None = None,
) -> Segmentation:
    """Cluster merchants in the given space; ``k`` defaults to the silhouette choice."""
    if matrix.shape[0] != len(merchant_ids):
        raise ValueError("One row per merchant is required")
    scores = pl.DataFrame(schema={"k": pl.Int64, "score": pl.Float64})
    if k is None:
        k, scores = select_k(matrix, cfg)
    segments = _fit(matrix, k, cfg)
    return Segmentation(
        name=name,
        labels=_labels_frame(merchant_ids, segments),
        k=int(k),
        silhouette=_silhouette(matrix, segments, cfg),
        scores=scores,
    )
