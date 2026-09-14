"""The rungs of the link-prediction ladder that need no neural network.

Each scorer returns the same thing: a (evaluated cardholders x candidate merchants) matrix of
scores, so :func:`ips.elasticity.link_prediction.evaluate_rankings` can judge them all the same
way. Nothing here sees the held-out months — every input is derived from ``split.train``.

The rung that matters is ``group_popularity``. It is the honest baseline: it knows only which
categories a cardholder already shops in and which merchants are big inside those categories.
Anything the graph claims has to beat that, not just beat global popularity.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from ips.elasticity.link_prediction import LinkSplit
from ips.graph.spectral_embed import MerchantEmbedding
from ips.utils.config import ProjectConfig


def random_scores(split: LinkSplit, cfg: ProjectConfig) -> np.ndarray:
    """The floor of the ladder: a seeded random ranking, identical between runs."""
    rng = np.random.default_rng(cfg.seed)
    return rng.random(split.shape)


def _candidate_purchases(split: LinkSplit) -> np.ndarray:
    """Training purchases of each candidate merchant, aligned with ``split.candidates``."""
    counts = (
        pl.DataFrame({"merchant_id": split.candidates})
        .join(
            split.train.group_by("merchant_id").agg(pl.col("n_txns").sum()),
            on="merchant_id",
            how="left",
        )
        .with_columns(pl.col("n_txns").fill_null(0))
        .sort("merchant_id")
    )
    return counts["n_txns"].to_numpy().astype(np.float64)


def popularity_scores(split: LinkSplit) -> np.ndarray:
    """Global popularity: every cardholder gets the same ranking."""
    scores = np.log1p(_candidate_purchases(split))
    return np.broadcast_to(scores, split.shape).copy()


def group_popularity_scores(
    split: LinkSplit, merchants: pl.DataFrame, cfg: ProjectConfig
) -> np.ndarray:
    """Popularity inside the categories the cardholder already shops in.

    Es el baseline fuerte y sin grafo: la mezcla de grupos revelada por el historial pesa a los
    comercios grandes de cada categoría. Un embedding que no le gane a esto no está aportando
    estructura, solo popularidad disfrazada.
    """
    groups = list(cfg.mcc_groups.groups)
    position = {name: index for index, name in enumerate(groups)}
    catalogue = merchants.select("merchant_id", pl.col("mcc_group").cast(pl.Utf8))

    candidate_groups = (
        pl.DataFrame({"merchant_id": split.candidates})
        .join(catalogue, on="merchant_id", how="left")
        .with_columns(pl.col("mcc_group").fill_null(groups[0]))
        .sort("merchant_id")
    )
    column_group = np.array(
        [position[name] for name in candidate_groups["mcc_group"]], dtype=np.int64
    )

    purchases = _candidate_purchases(split)
    within = np.zeros_like(purchases)
    for index in range(len(groups)):
        block = column_group == index
        total = purchases[block].sum()
        if total > 0:
            within[block] = purchases[block] / total

    mix = (
        split.train.join(catalogue, on="merchant_id", how="inner")
        .group_by("cardholder_id", "mcc_group")
        .agg(pl.col("n_txns").sum())
        .with_columns(
            (pl.col("n_txns") / pl.col("n_txns").sum().over("cardholder_id")).alias("share")
        )
    )
    rows = np.zeros((len(split.cardholders), len(groups)), dtype=np.float64)
    inside = mix.join(
        pl.DataFrame({"cardholder_id": split.cardholders}), on="cardholder_id", how="inner"
    )
    row_index = np.searchsorted(split.cardholders, inside["cardholder_id"].to_numpy())
    group_index = np.array([position[name] for name in inside["mcc_group"]], dtype=np.int64)
    rows[row_index, group_index] = inside["share"].to_numpy()
    return rows[:, column_group] * within[None, :]


def cardholder_profiles(
    split: LinkSplit, merchant_ids: np.ndarray, vectors: np.ndarray
) -> np.ndarray:
    """Each evaluated cardholder as the purchase-weighted mean of the merchants it bought from."""
    from scipy.sparse import csr_matrix

    order = np.argsort(merchant_ids)
    sorted_ids = merchant_ids[order]
    history = split.train.join(
        pl.DataFrame({"cardholder_id": split.cardholders}), on="cardholder_id", how="inner"
    ).join(pl.DataFrame({"merchant_id": sorted_ids}), on="merchant_id", how="inner")
    row = np.searchsorted(split.cardholders, history["cardholder_id"].to_numpy())
    column = order[np.searchsorted(sorted_ids, history["merchant_id"].to_numpy())]
    counts = csr_matrix(
        (history["n_txns"].to_numpy().astype(np.float64), (row, column)),
        shape=(len(split.cardholders), len(merchant_ids)),
    )
    profiles = counts @ vectors
    weight = np.asarray(counts.sum(axis=1)).ravel()
    return profiles / np.maximum(weight, 1.0)[:, None]


def embedding_scores(split: LinkSplit, embedding: MerchantEmbedding) -> np.ndarray:
    """Cosine between the cardholder's profile and each candidate merchant's vector."""
    present = np.isin(split.candidates, embedding.merchant_ids)
    if not present.all():
        missing = int((~present).sum())
        raise ValueError(f"{missing} candidate merchants are absent from the embedding")
    profiles = cardholder_profiles(split, embedding.merchant_ids, embedding.vectors)
    profiles = profiles / np.maximum(np.linalg.norm(profiles, axis=1, keepdims=True), 1e-12)
    order = np.argsort(embedding.merchant_ids)
    columns = order[np.searchsorted(embedding.merchant_ids[order], split.candidates)]
    candidates = embedding.vectors[columns]
    candidates = candidates / np.maximum(np.linalg.norm(candidates, axis=1, keepdims=True), 1e-12)
    return profiles @ candidates.T
