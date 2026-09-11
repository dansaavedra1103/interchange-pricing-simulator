"""Shared fixtures: a small configuration and one generated dataset for the whole session."""

from __future__ import annotations

import polars as pl
import pytest

from ips.data_gen.generate import GeneratedData, generate
from ips.utils.config import ProjectConfig, Sizes, load_config

SMALL_SIZES = Sizes(
    n_cardholders=6_000, n_merchants=1_500, n_foreign_merchants=60, n_transactions=200_000
)


def small_config() -> ProjectConfig:
    """``base.yaml`` with small sizes: the same parameters, a dataset built in seconds."""
    return load_config().model_copy(update={"sizes": SMALL_SIZES})


@pytest.fixture(scope="session")
def cfg() -> ProjectConfig:
    return small_config()


@pytest.fixture(scope="session")
def data(cfg: ProjectConfig) -> GeneratedData:
    return generate(cfg)


@pytest.fixture(scope="session")
def transactions(data: GeneratedData) -> pl.DataFrame:
    return data.tables["transactions"]
