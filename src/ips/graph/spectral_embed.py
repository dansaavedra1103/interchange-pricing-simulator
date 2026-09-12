"""Merchant embeddings: PPMI over the bipartite graph, then a truncated SVD.

Random-walk embeddings (DeepWalk, node2vec) are implicit factorisations of a shifted pointwise
mutual information matrix: Levy and Goldberg (2014) for word2vec, Qiu et al. (2018, *Network
Embedding as Matrix Factorization*) for the graph case. Factorising it directly is
deterministic, needs no walk sampling and runs in seconds on the 6M purchases, which is what a
reviewer can reproduce.

Two merchants end up close when the same cards buy from both, whatever their MCC says.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import polars as pl

from ips.graph.build_graph import BipartiteGraph
from ips.utils.config import ProjectConfig

if TYPE_CHECKING:  # pragma: no cover - typing only, scipy is an optional extra
    from scipy.sparse import csr_matrix

_DIMENSION = "dim_{:03d}"


@dataclass(frozen=True)
class MerchantEmbedding:
    """L2-normalised merchant vectors, aligned with ``merchant_ids``."""

    merchant_ids: np.ndarray
    vectors: np.ndarray

    @property
    def dims(self) -> int:
        """Number of dimensions."""
        return self.vectors.shape[1]

    def frame(self) -> pl.DataFrame:
        """Long-lived form: one row per merchant, one column per dimension."""
        columns = {
            _DIMENSION.format(i): self.vectors[:, i].astype(np.float64) for i in range(self.dims)
        }
        return pl.DataFrame(
            {"merchant_id": pl.Series(self.merchant_ids, dtype=pl.UInt32)} | columns
        )

    @classmethod
    def from_frame(cls, frame: pl.DataFrame) -> MerchantEmbedding:
        """Rebuild an embedding written by :meth:`frame`."""
        dimensions = [c for c in frame.columns if c.startswith("dim_")]
        return cls(frame["merchant_id"].to_numpy(), frame.select(dimensions).to_numpy())

    def write_parquet(self, path: Path) -> Path:
        """Write the embedding as parquet, creating the directory if needed."""
        path.parent.mkdir(parents=True, exist_ok=True)
        self.frame().write_parquet(path)
        return path

    @classmethod
    def read_parquet(cls, path: Path) -> MerchantEmbedding:
        """Read an embedding written by :meth:`write_parquet`."""
        return cls.from_frame(pl.read_parquet(path))


def ppmi_matrix(graph: BipartiteGraph, shift: float) -> csr_matrix:
    """Positive pointwise mutual information of the cardholder-merchant counts.

    PMI = log( p(card, merchant) / (p(card) p(merchant)) ). El desplazamiento es el número de
    muestras negativas de word2vec: sube el umbral bajo el cual una coincidencia se considera
    casualidad y se descarta.
    """
    from scipy.sparse import csr_matrix

    coordinates = graph.matrix.tocoo()
    total = float(coordinates.data.sum())
    per_cardholder = np.asarray(graph.matrix.sum(axis=1)).ravel()
    per_merchant = np.asarray(graph.matrix.sum(axis=0)).ravel()
    expected = per_cardholder[coordinates.row] * per_merchant[coordinates.col] / total
    scores = np.log(coordinates.data / expected) - np.log(shift)
    keep = scores > 0
    return csr_matrix(
        (scores[keep], (coordinates.row[keep], coordinates.col[keep])), shape=graph.shape
    )


def embed_merchants(graph: BipartiteGraph, cfg: ProjectConfig) -> MerchantEmbedding:
    """Merchant vectors from the truncated SVD of the PPMI matrix."""
    from sklearn.utils.extmath import randomized_svd

    weights = ppmi_matrix(graph, cfg.segmentation.ppmi_shift)
    dims = min(cfg.segmentation.embedding_dims, min(graph.shape) - 1)
    _, singular, right = randomized_svd(weights, n_components=dims, random_state=cfg.seed)
    vectors = right.T * np.sqrt(singular)
    # Signo canónico: la componente mayor de cada eje queda positiva, para que dos corridas con
    # la misma semilla den vectores idénticos bit a bit.
    leading = np.abs(vectors).argmax(axis=0)
    signs = np.sign(vectors[leading, np.arange(dims)])
    vectors = vectors * np.where(signs == 0, 1.0, signs)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return MerchantEmbedding(graph.merchant_ids, vectors / np.maximum(norms, 1e-12))
