"""Segmentation that aims at the business question instead of at the shape of the data.

k-means groups merchants that *look* alike; a pricing team needs groups that *behave* alike in
the number the decision turns on. A shallow regression tree over the same attributes gives
exactly that, and every segment comes with the rule that defines it, which is what a memo can
carry: "card-not-present above 40% and grocery" is a segment someone can act on.

The tree only ever sees the training half of ``evaluate.train_test_split``, and the number of
leaves is chosen inside that half, so the out-of-sample numbers stay honest.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
import polars as pl

from ips.graph.clustering import Segmentation
from ips.graph.evaluate import train_test_split
from ips.utils.config import ProjectConfig

if TYPE_CHECKING:  # pragma: no cover - typing only, scikit-learn is an optional extra
    from sklearn.tree import DecisionTreeRegressor

TARGET = "relative_burden"


def _fit_tree(
    matrix: np.ndarray, values: np.ndarray, leaves: int, cfg: ProjectConfig
) -> DecisionTreeRegressor:
    from sklearn.tree import DecisionTreeRegressor

    return DecisionTreeRegressor(max_leaf_nodes=leaves, random_state=cfg.seed).fit(matrix, values)


def _r2(truth: np.ndarray, predicted: np.ndarray, mean: float) -> float:
    reference = float(np.sum((truth - mean) ** 2))
    if reference <= 0:
        return float("nan")
    return 1.0 - float(np.sum((truth - predicted) ** 2)) / reference


def select_leaves(
    matrix: np.ndarray, values: np.ndarray, cfg: ProjectConfig
) -> tuple[int, pl.DataFrame]:
    """Choose the number of segments inside the training half, never on the test set."""
    train, _ = train_test_split(len(values), cfg)
    inner, validation = train_test_split(len(train), cfg)
    fit_rows, check_rows = train[inner], train[validation]
    mean = float(values[fit_rows].mean())
    rows = [
        {
            "k": leaves,
            "score": _r2(
                values[check_rows],
                _fit_tree(matrix[fit_rows], values[fit_rows], leaves, cfg).predict(
                    matrix[check_rows]
                ),
                mean,
            ),
        }
        for leaves in cfg.segmentation.k_values
    ]
    scores = pl.DataFrame(rows)
    return int(scores.sort("score", descending=True)["k"].first()), scores


def fit_supervised(
    profile: pl.DataFrame,
    matrix: np.ndarray,
    cfg: ProjectConfig,
    name: str = "supervised",
    leaves: int | None = None,
    target: str = TARGET,
) -> tuple[Segmentation, DecisionTreeRegressor]:
    """Segments are the leaves of a regression tree fitted on the training half only."""
    if matrix.shape[0] != profile.height:
        raise ValueError("One row per merchant is required")
    values = profile[target].to_numpy()
    scores = pl.DataFrame(schema={"k": pl.Int64, "score": pl.Float64})
    if leaves is None:
        leaves, scores = select_leaves(matrix, values, cfg)
    train, _ = train_test_split(len(values), cfg)
    model = _fit_tree(matrix[train], values[train], leaves, cfg)

    # La hoja en la que cae cada comercio es su segmento; se renumeran de 0 en adelante.
    reached = model.apply(matrix)
    codes = {int(leaf): index for index, leaf in enumerate(np.unique(reached))}
    labels = pl.DataFrame(
        {
            "merchant_id": profile["merchant_id"],
            "segment": pl.Series([codes[int(leaf)] for leaf in reached], dtype=pl.Int32),
        }
    ).sort("merchant_id")
    segmentation = Segmentation(
        name=name, labels=labels, k=len(codes), silhouette=float("nan"), scores=scores
    )
    return segmentation, model


def _condition(name: str, threshold: float, goes_left: bool) -> str:
    """Readable form of one split; one-hot columns are named ``column=value``."""
    if "=" in name and abs(threshold - 0.5) < 0.5:
        column, value = name.split("=", 1)
        return f"{column} {'≠' if goes_left else '='} {value}"
    return f"{name} {'<=' if goes_left else '>'} {threshold:.3g}"


def segment_rules(model: DecisionTreeRegressor, feature_names: Sequence[str]) -> pl.DataFrame:
    """One row per segment with the conditions that define it, in the tree's leaf order."""
    tree = model.tree_
    leaves: list[tuple[int, list[str]]] = []

    def walk(node: int, conditions: list[str]) -> None:
        if tree.children_left[node] == -1:
            leaves.append((node, conditions))
            return
        name = feature_names[tree.feature[node]]
        threshold = float(tree.threshold[node])
        walk(tree.children_left[node], [*conditions, _condition(name, threshold, True)])
        walk(tree.children_right[node], [*conditions, _condition(name, threshold, False)])

    walk(0, [])
    order = {node: index for index, node in enumerate(sorted(node for node, _ in leaves))}
    return pl.DataFrame(
        [
            {
                "segment": order[node],
                "merchants_in_fit": int(tree.n_node_samples[node]),
                "predicted": float(tree.value[node].ravel()[0]),
                "rule": " and ".join(conditions) if conditions else "every merchant",
            }
            for node, conditions in leaves
        ]
    ).sort("segment")
