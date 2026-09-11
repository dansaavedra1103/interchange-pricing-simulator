"""Fraud and chargeback flags for generated transactions."""

from __future__ import annotations

import numpy as np
import polars as pl

from ips.utils.config import ProjectConfig

_REQUIRED = ("channel", "mcc_group", "cross_border", "tokenized")


def fraud_probability(transactions: pl.DataFrame, cfg: ProjectConfig) -> np.ndarray:
    """Fraud probability of each transaction.

    p = base rate of the channel x risk of the MCC group x cross-border multiplier x
    tokenisation multiplier, capped at 1.
    """
    missing = [c for c in _REQUIRED if c not in transactions.columns]
    if missing:
        raise ValueError(f"transactions is missing required columns: {missing}")
    groups = cfg.mcc_groups.groups
    fraud = cfg.fraud
    probability = transactions.select(
        (
            pl.col("channel").cast(pl.Utf8).replace_strict(fraud.base_rate, return_dtype=pl.Float64)
            * pl.col("mcc_group")
            .cast(pl.Utf8)
            .replace_strict({g: c.fraud_risk for g, c in groups.items()}, return_dtype=pl.Float64)
            # El fraude se concentra en no presente y cross-border (BCE 2023); el token lo reduce.
            * pl.when(pl.col("cross_border")).then(fraud.cross_border_multiplier).otherwise(1.0)
            * pl.when(pl.col("tokenized")).then(fraud.tokenized_multiplier).otherwise(1.0)
        )
        .clip(upper_bound=1.0)
        .alias("p")
    )
    return probability["p"].to_numpy()


def add_fraud_flags(
    rng: np.random.Generator, transactions: pl.DataFrame, cfg: ProjectConfig
) -> pl.DataFrame:
    """Add ``fraud_flag`` and ``chargeback_flag``.

    Casi todo fraude termina en contracargo; en compras legítimas el contracargo es una
    disputa de servicio, más frecuente en viajes y bienes digitales.
    """
    n = transactions.height
    fraud_flag = rng.random(n) < fraud_probability(transactions, cfg)
    dispute_rate = (
        transactions["mcc_group"]
        .cast(pl.Utf8)
        .replace_strict(
            {g: c.dispute_rate for g, c in cfg.mcc_groups.groups.items()}, return_dtype=pl.Float64
        )
        .to_numpy()
    )
    p_chargeback = np.where(fraud_flag, cfg.fraud.p_chargeback_given_fraud, dispute_rate)
    chargeback_flag = rng.random(n) < p_chargeback
    return transactions.with_columns(
        pl.Series("fraud_flag", fraud_flag), pl.Series("chargeback_flag", chargeback_flag)
    )
