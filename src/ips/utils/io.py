"""I/O layer: project paths, raw parquet files and the DuckDB warehouse built by dbt.

Paths always come from ``config/base.yaml``. The warehouse opens read-only unless a caller
asks otherwise: DuckDB lets one process write a file or several read it, never both.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

import duckdb
import polars as pl

from ips.utils.config import ProjectConfig, load_config, project_root, resolve_path

_TABLE_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)?$")


def _config(cfg: ProjectConfig | None) -> ProjectConfig:
    return cfg if cfg is not None else load_config()


def dbt_dir() -> Path:
    """The dbt project directory (``dbt/`` at the repository root)."""
    return project_root() / "dbt"


def raw_dir(cfg: ProjectConfig | None = None) -> Path:
    """Directory holding the generated parquet files."""
    return resolve_path(_config(cfg).output.raw_dir)


def raw_path(name: str, cfg: ProjectConfig | None = None) -> Path:
    """Path of one raw table, e.g. ``raw_path("transactions")``."""
    return raw_dir(cfg) / f"{name}.parquet"


def read_raw(name: str, cfg: ProjectConfig | None = None) -> pl.DataFrame:
    """Read one raw parquet table."""
    return pl.read_parquet(raw_path(name, cfg))


def warehouse_path(cfg: ProjectConfig | None = None) -> Path:
    """Path of the DuckDB warehouse file."""
    return resolve_path(_config(cfg).output.warehouse_path)


def connect(path: Path | None = None, *, read_only: bool = True) -> duckdb.DuckDBPyConnection:
    """Open the warehouse, read-only by default."""
    target = path if path is not None else warehouse_path()
    return duckdb.connect(str(target), read_only=read_only)


def read_table(name: str, path: Path | None = None) -> pl.DataFrame:
    """Read a table or view built by dbt, e.g. ``read_table("mart_pnl_by_actor")``."""
    if not _TABLE_NAME.match(name):
        raise ValueError(f"Not a table name: {name!r}")
    with connect(path) as connection:
        return connection.sql(f"select * from {name}").pl()


def dbt_environment(raw: Path | None = None, warehouse: Path | None = None) -> dict[str, str]:
    """Variables the dbt project reads, as absolute POSIX paths (valid in SQL on any OS)."""
    raw_location = raw if raw is not None else raw_dir()
    warehouse_location = warehouse if warehouse is not None else warehouse_path()
    return {
        "IPS_RAW_DIR": raw_location.resolve().as_posix(),
        "IPS_WAREHOUSE": warehouse_location.resolve().as_posix(),
    }


def _dbt_executable() -> str:
    beside_python = Path(sys.executable).with_name("dbt.exe" if os.name == "nt" else "dbt")
    if beside_python.exists():
        return str(beside_python)
    found = shutil.which("dbt")
    if found is None:
        raise FileNotFoundError(
            "dbt not found: install the extra with pip install -e '.[pipeline]'"
        )
    return found


def run_dbt(
    args: Sequence[str],
    *,
    raw: Path | None = None,
    warehouse: Path | None = None,
    target_path: Path | None = None,
    log_path: Path | None = None,
    capture: bool = False,
) -> subprocess.CompletedProcess[str]:
    """Run a dbt command (e.g. ``["build"]``) against the project.

    dbt runs in a subprocess so that its DuckDB connection never stays open in the caller's
    process, which can then read the warehouse right away.
    """
    environment = dbt_environment(raw, warehouse)
    Path(environment["IPS_WAREHOUSE"]).parent.mkdir(parents=True, exist_ok=True)
    project = str(dbt_dir())
    command = [_dbt_executable(), *args, "--project-dir", project, "--profiles-dir", project]
    if target_path is not None:
        command += ["--target-path", str(target_path)]
    if log_path is not None:
        command += ["--log-path", str(log_path)]
    return subprocess.run(
        command,
        env={**os.environ, **environment},
        capture_output=capture,
        text=True,
        check=False,
    )
