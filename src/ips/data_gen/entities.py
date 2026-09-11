"""Entity contracts of the synthetic network: one pydantic model per output table.

The generator works on columnar polars frames for speed. These models document each table's
fields and allowed values; ``table_schemas`` gives the matching polars schemas (categorical
columns as ``Enum`` with the categories taken from the configuration), and the tests check
that both stay aligned.
"""

from __future__ import annotations

import datetime as dt
from typing import Literal

import polars as pl
from pydantic import BaseModel, ConfigDict, Field

from ips.utils.config import (
    CHANNELS,
    MERCHANT_CHANNELS,
    OTHER_CITIES,
    SIZE_TIERS,
    SPEND_SEGMENTS,
    ProjectConfig,
)


class _Row(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class Issuer(_Row):
    """Card issuer (bank or fintech)."""

    issuer_id: str
    issuer_type: Literal["bank", "fintech"]
    size: Literal["large", "medium", "small"]
    country: str
    bank_group: str | None
    market_share: float = Field(ge=0.0, le=1.0)


class Acquirer(_Row):
    """Merchant acquirer; the foreign one serves every merchant abroad."""

    acquirer_id: str
    acquirer_type: Literal["bank", "aggregator", "foreign"]
    country: str
    bank_group: str | None
    pricing_model: Literal["blended", "icpp", "mixed"]


class Merchant(_Row):
    """Merchant. ``city`` is null for online-only and foreign merchants."""

    merchant_id: int = Field(ge=0)
    mcc: str
    mcc_group: str
    country: str
    city: str | None
    channel: Literal["cp", "cnp", "mixed"]
    size_tier: Literal["large", "medium", "small", "micro"]
    size_weight: float = Field(gt=0.0)
    acquirer_id: str
    sector_margin: float = Field(gt=0.0, lt=1.0)


class Cardholder(_Row):
    """A card (a person with two cards appears twice)."""

    cardholder_id: int = Field(ge=0)
    issuer_id: str
    product: str
    country: str
    city: str
    spend_segment: Literal["low", "mid", "high"]
    cross_border_propensity: float = Field(ge=0.0, le=1.0)


class Transaction(_Row):
    """A purchase: facts only, fees are derived by the P&L."""

    txn_id: int = Field(ge=0)
    txn_date: dt.date
    cardholder_id: int = Field(ge=0)
    merchant_id: int = Field(ge=0)
    issuer_id: str
    acquirer_id: str
    product: str
    mcc: str
    mcc_group: str
    channel: Literal["cp", "cnp"]
    tokenized: bool
    cross_border: bool
    on_us: bool
    amount_cop: int = Field(ge=1)
    fraud_flag: bool
    chargeback_flag: bool


ENTITY_MODELS: dict[str, type[_Row]] = {
    "issuers": Issuer,
    "acquirers": Acquirer,
    "merchants": Merchant,
    "cardholders": Cardholder,
    "transactions": Transaction,
}


def enum_dtypes(cfg: ProjectConfig) -> dict[str, pl.Enum]:
    """Enum dtypes for the categorical columns, with categories in configuration order."""
    acquirer_ids = [a.acquirer_id for a in cfg.acquirers] + [cfg.cross_border.acquirer.acquirer_id]
    return {
        "issuer_id": pl.Enum([i.issuer_id for i in cfg.issuers]),
        "acquirer_id": pl.Enum(acquirer_ids),
        "product": pl.Enum(list(cfg.products)),
        "mcc": pl.Enum(list(cfg.mcc_codes)),
        "mcc_group": pl.Enum(list(cfg.group_names)),
        "channel": pl.Enum(list(CHANNELS)),
        "merchant_channel": pl.Enum(list(MERCHANT_CHANNELS)),
        "size_tier": pl.Enum(list(SIZE_TIERS)),
        "spend_segment": pl.Enum(list(SPEND_SEGMENTS)),
        "city": pl.Enum([*cfg.geography.cities, OTHER_CITIES]),
    }


def table_schemas(cfg: ProjectConfig) -> dict[str, pl.Schema]:
    """Polars schema of every observable table, keyed like ``ENTITY_MODELS``."""
    e = enum_dtypes(cfg)
    return {
        "issuers": pl.Schema(
            {
                "issuer_id": e["issuer_id"],
                "issuer_type": pl.Utf8,
                "size": pl.Utf8,
                "country": pl.Utf8,
                "bank_group": pl.Utf8,
                "market_share": pl.Float64,
            }
        ),
        "acquirers": pl.Schema(
            {
                "acquirer_id": e["acquirer_id"],
                "acquirer_type": pl.Utf8,
                "country": pl.Utf8,
                "bank_group": pl.Utf8,
                "pricing_model": pl.Utf8,
            }
        ),
        "merchants": pl.Schema(
            {
                "merchant_id": pl.UInt32,
                "mcc": e["mcc"],
                "mcc_group": e["mcc_group"],
                "country": pl.Utf8,
                "city": e["city"],
                "channel": e["merchant_channel"],
                "size_tier": e["size_tier"],
                "size_weight": pl.Float64,
                "acquirer_id": e["acquirer_id"],
                "sector_margin": pl.Float64,
            }
        ),
        "cardholders": pl.Schema(
            {
                "cardholder_id": pl.UInt32,
                "issuer_id": e["issuer_id"],
                "product": e["product"],
                "country": pl.Utf8,
                "city": e["city"],
                "spend_segment": e["spend_segment"],
                "cross_border_propensity": pl.Float64,
            }
        ),
        "transactions": pl.Schema(
            {
                "txn_id": pl.UInt32,
                "txn_date": pl.Date,
                "cardholder_id": pl.UInt32,
                "merchant_id": pl.UInt32,
                "issuer_id": e["issuer_id"],
                "acquirer_id": e["acquirer_id"],
                "product": e["product"],
                "mcc": e["mcc"],
                "mcc_group": e["mcc_group"],
                "channel": e["channel"],
                "tokenized": pl.Boolean,
                "cross_border": pl.Boolean,
                "on_us": pl.Boolean,
                "amount_cop": pl.Int64,
                "fraud_flag": pl.Boolean,
                "chargeback_flag": pl.Boolean,
            }
        ),
    }
