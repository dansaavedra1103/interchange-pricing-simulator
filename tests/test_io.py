"""I/O layer: paths from the config, raw reads and a read-only warehouse."""

from __future__ import annotations

from pathlib import Path

import duckdb
import pytest

from ips.utils import io
from ips.utils.config import project_root


def test_paths_come_from_the_config() -> None:
    root = project_root()
    assert io.raw_dir() == root / "data" / "raw"
    assert io.raw_path("transactions") == root / "data" / "raw" / "transactions.parquet"
    assert io.warehouse_path() == root / "data" / "warehouse" / "ips.duckdb"
    assert io.dbt_dir() == root / "dbt"


def test_dbt_environment_uses_absolute_posix_paths(tmp_path: Path) -> None:
    environment = io.dbt_environment(tmp_path / "raw", tmp_path / "warehouse.duckdb")
    assert set(environment) == {"IPS_RAW_DIR", "IPS_WAREHOUSE"}
    for value in environment.values():
        assert "\\" not in value
        assert Path(value).is_absolute()


def test_read_table_is_read_only(tmp_path: Path) -> None:
    path = tmp_path / "warehouse.duckdb"
    with duckdb.connect(str(path)) as connection:
        connection.execute("create table mart_example as select 1 as a, 'x' as b")
    assert io.read_table("mart_example", path).to_dicts() == [{"a": 1, "b": "x"}]
    with io.connect(path) as connection, pytest.raises(duckdb.Error):
        connection.execute("create table another as select 1 as a")


def test_read_table_rejects_anything_but_a_name(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Not a table name"):
        io.read_table("mart_example; drop table mart_example", tmp_path / "warehouse.duckdb")
