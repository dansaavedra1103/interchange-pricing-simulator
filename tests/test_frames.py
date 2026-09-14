"""Frame helpers: grouped sums that give the same bits on every call."""

from __future__ import annotations

import polars as pl
import pytest

from ips.utils.frames import stable_group_sums


def test_sums_match_a_group_by_and_come_out_sorted() -> None:
    frame = pl.DataFrame(
        {
            "product": ["credit", "debit", "credit", "debit", "credit"],
            "channel": ["cp", "cnp", "cp", "cp", "cnp"],
            "amount": [1.5, 2.0, 2.5, 4.0, None],
            "fee": [0.1, 0.2, 0.3, 0.4, 0.5],
        }
    )
    sums = stable_group_sums(
        frame, ["product", pl.col("channel")], "amount", (pl.col("fee") * 2).alias("double_fee")
    )
    expected = (
        frame.group_by("product", "channel")
        .agg(pl.col("amount").sum(), (pl.col("fee") * 2).sum().alias("double_fee"))
        .sort("product", "channel")
    )
    assert sums.select("product", "channel").equals(expected.select("product", "channel"))
    assert sums["amount"].to_list() == pytest.approx(expected["amount"].to_list())
    assert sums["double_fee"].to_list() == pytest.approx(expected["double_fee"].to_list())
    assert sums.schema["amount"] == pl.Float64


def test_a_group_adds_its_rows_in_the_order_the_frame_lists_them() -> None:
    # 1e16 + 1 se redondea a 1e16: en el grupo a ese 1 se pierde y en el b sobrevive. Intercalar
    # las filas de otro grupo no puede cambiar ningún bit.
    frame = pl.DataFrame(
        {"key": ["a", "b", "a", "b", "a", "b"], "value": [1.0, 1e16, 1e16, -1e16, -1e16, 1.0]}
    )
    sums = stable_group_sums(frame, "key", "value")
    assert sums["value"].to_list() == [0.0, 1.0]
    interleaved = frame.select(pl.all().gather([1, 3, 0, 5, 2, 4]))
    assert stable_group_sums(interleaved, "key", "value").equals(sums)


def test_empty_frames_and_null_keys() -> None:
    empty = stable_group_sums(
        pl.DataFrame(schema={"merchant_id": pl.Int64, "amount": pl.Int64}), "merchant_id", "amount"
    )
    assert empty.is_empty()
    assert dict(empty.schema) == {"merchant_id": pl.Int64, "amount": pl.Float64}
    with pytest.raises(ValueError, match="cannot be null"):
        stable_group_sums(pl.DataFrame({"key": [1, None], "value": [1.0, 2.0]}), "key", "value")
