"""Frame helpers for guarantees polars does not give on its own."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import polars as pl

GroupKey = str | pl.Expr


def stable_group_sums(
    frame: pl.DataFrame, keys: GroupKey | Sequence[GroupKey], *values: GroupKey
) -> pl.DataFrame:
    """Sums of ``values`` by ``keys``, identical to the last bit on every call.

    ``group_by().agg(pl.sum(...))`` may add a group's floats in a different order on each call
    (polars 1.44 does it on groups of a few rows), and the order moves the last bits of a sum.
    Here every group adds its rows one by one, in the order the frame lists them, so the same
    frame always gives the same sums however the rows of other groups are interleaved. Keys
    cannot be null; sums come out as Float64 with nulls counted as zero, and groups sorted by
    their keys.
    """
    key_list = [keys] if isinstance(keys, str | pl.Expr) else list(keys)
    selected = frame.select(*key_list, *values)
    names = selected.columns[: len(key_list)]
    columns = selected.columns[len(key_list) :]
    if selected.select(pl.any_horizontal(pl.col(names).is_null()).any()).item():
        raise ValueError(f"Group keys cannot be null: {names}")
    groups = selected.select(names).unique().sort(names).with_row_index("_group")
    rows = selected.join(groups, on=names, how="left", maintain_order="left")["_group"]
    positions = rows.to_numpy().astype(np.int64)
    return groups.drop("_group").with_columns(
        pl.Series(
            column,
            np.bincount(
                positions,
                weights=selected[column].cast(pl.Float64).fill_null(0.0).to_numpy(),
                minlength=groups.height,
            ),
            dtype=pl.Float64,
        )
        for column in columns
    )
