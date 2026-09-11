"""Shared fixtures: a small configuration and one generated dataset for the whole session."""

from __future__ import annotations

import polars as pl
import pytest

from ips.data_gen.generate import GeneratedData, generate
from ips.utils.config import ProjectConfig, load_config


def small_config() -> ProjectConfig:
    """``base.yaml`` with its ``sample_sizes``: the same parameters, built in seconds."""
    cfg = load_config()
    return cfg.model_copy(update={"sizes": cfg.sample_sizes})


@pytest.fixture(scope="session")
def cfg() -> ProjectConfig:
    return small_config()


@pytest.fixture(scope="session")
def data(cfg: ProjectConfig) -> GeneratedData:
    return generate(cfg)


@pytest.fixture(scope="session")
def transactions(data: GeneratedData) -> pl.DataFrame:
    return data.tables["transactions"]
