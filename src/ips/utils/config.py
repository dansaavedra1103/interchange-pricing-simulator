"""Typed configuration for the interchange pricing simulator.

All business parameters live in ``config/*.yaml``. This module loads them into frozen
pydantic models and validates their internal consistency, so that a typo or a gap in the
interchange table fails at load time instead of silently leaking P&L downstream.
"""

from __future__ import annotations

import datetime as dt
import math
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Annotated, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

_SUM_TOLERANCE = 1e-6
# Sanity bound: catches rates written in percent (1.6) instead of as a fraction (0.016).
_MAX_RATE = 0.1

Rate = Annotated[float, Field(ge=0.0, lt=_MAX_RATE)]
Share = Annotated[float, Field(ge=0.0, le=1.0)]


def project_root() -> Path:
    """Return the repository root: the closest parent of this file holding ``pyproject.toml``."""
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").is_file():
            return parent
    raise FileNotFoundError("Could not locate the project root (no pyproject.toml found).")


def default_config_path() -> Path:
    """Path to ``config/base.yaml`` inside the repository."""
    return project_root() / "config" / "base.yaml"


def _check_sums_to_one(values: Iterable[float], label: str) -> None:
    total = math.fsum(values)
    if not math.isclose(total, 1.0, abs_tol=_SUM_TOLERANCE):
        raise ValueError(f"{label} must sum to 1, got {total:.6f}")


def _check_unique(values: Iterable[str], label: str) -> None:
    duplicates = sorted(value for value, count in Counter(values).items() if count > 1)
    if duplicates:
        raise ValueError(f"Duplicate {label}: {duplicates}")


def _check_same_keys(actual: Iterable[str], expected: Iterable[str], label: str) -> None:
    actual_set, expected_set = set(actual), set(expected)
    if actual_set != expected_set:
        missing, extra = sorted(expected_set - actual_set), sorted(actual_set - expected_set)
        raise ValueError(
            f"{label} must cover exactly {sorted(expected_set)}; missing={missing}, extra={extra}"
        )


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class DateRange(_Frozen):
    """Inclusive date range of the simulated transactions."""

    start: dt.date
    end: dt.date

    @model_validator(mode="after")
    def _check_order(self) -> DateRange:
        if self.start >= self.end:
            raise ValueError(f"dates.start ({self.start}) must be before dates.end ({self.end})")
        return self

    @property
    def n_days(self) -> int:
        """Number of calendar days in the range, both ends included."""
        return (self.end - self.start).days + 1


class Sizes(_Frozen):
    """Number of entities to generate."""

    n_merchants: int = Field(gt=0)
    n_cardholders: int = Field(gt=0)
    n_transactions: int = Field(gt=0)


class OutputConfig(_Frozen):
    """Output locations, relative to the project root unless absolute."""

    raw_dir: Path


class DistributionConfig(_Frozen):
    """Shape parameters of the generator's distributions."""

    merchant_size_sigma: float = Field(gt=0)
    cardholder_activity_sigma: float = Field(gt=0)
    min_amount_cop: int = Field(ge=1)


class IssuerConfig(_Frozen):
    """A card issuer. ``bank_group`` links it to an acquirer of the same group (on-us)."""

    issuer_id: str
    issuer_type: Literal["bank", "fintech"]
    bank_group: str | None = None
    market_share: Share


class AcquirerConfig(_Frozen):
    """A merchant acquirer. ``bank_group`` links it to an issuer of the same group (on-us)."""

    acquirer_id: str
    acquirer_type: Literal["bank", "aggregator"]
    bank_group: str | None = None
    market_share: Share


class MccConfig(_Frozen):
    """Merchant category: share of merchants and lognormal ticket-size parameters."""

    mcc: str = Field(pattern=r"^\d{4}$")
    name: str
    merchant_share: Share
    median_ticket_cop: float = Field(gt=0)
    ticket_sigma: float = Field(gt=0)


class Fee(_Frozen):
    """Percentage-plus-fixed fee: ``amount * rate + fixed_cop``."""

    rate: Rate
    fixed_cop: float = Field(default=0.0, ge=0.0)


class PricingConfig(_Frozen):
    """Scheme fee, blended MDR per MCC and the interchange table (product -> MCC -> fee)."""

    scheme_fee: Fee
    blended_mdr_rate: dict[str, Rate]
    interchange_table: dict[str, dict[str, Fee]]

    @model_validator(mode="after")
    def _check_grid(self) -> PricingConfig:
        for product, cells in self.interchange_table.items():
            _check_same_keys(cells, self.blended_mdr_rate, f"interchange_table[{product!r}]")
        return self


class ProjectConfig(_Frozen):
    """Root configuration loaded from ``config/base.yaml``."""

    seed: int
    dates: DateRange
    sizes: Sizes
    output: OutputConfig
    products: tuple[str, ...] = Field(min_length=1)
    distributions: DistributionConfig
    issuers: tuple[IssuerConfig, ...] = Field(min_length=1)
    acquirers: tuple[AcquirerConfig, ...] = Field(min_length=1)
    product_mix: dict[str, dict[str, Share]]
    mccs: tuple[MccConfig, ...] = Field(min_length=1)
    pricing: PricingConfig

    @model_validator(mode="after")
    def _check_consistency(self) -> ProjectConfig:
        _check_unique(self.products, "products")
        _check_unique((i.issuer_id for i in self.issuers), "issuer_id")
        _check_unique((a.acquirer_id for a in self.acquirers), "acquirer_id")
        _check_unique(self.mcc_codes, "mcc")

        _check_sums_to_one((i.market_share for i in self.issuers), "issuers market_share")
        _check_sums_to_one((a.market_share for a in self.acquirers), "acquirers market_share")
        _check_sums_to_one((m.merchant_share for m in self.mccs), "mccs merchant_share")

        missing_types = {i.issuer_type for i in self.issuers} - set(self.product_mix)
        if missing_types:
            raise ValueError(f"product_mix has no row for issuer types {sorted(missing_types)}")
        for issuer_type, mix in self.product_mix.items():
            _check_same_keys(mix, self.products, f"product_mix[{issuer_type!r}]")
            _check_sums_to_one(mix.values(), f"product_mix[{issuer_type!r}]")

        _check_same_keys(self.pricing.blended_mdr_rate, self.mcc_codes, "pricing.blended_mdr_rate")
        _check_same_keys(self.pricing.interchange_table, self.products, "pricing.interchange_table")
        return self

    @property
    def mcc_codes(self) -> tuple[str, ...]:
        """MCC codes in configuration order."""
        return tuple(m.mcc for m in self.mccs)

    def resolve_path(self, path: Path) -> Path:
        """Resolve a configured path against the project root (absolute paths pass through)."""
        return path if path.is_absolute() else project_root() / path


def load_config(path: Path | None = None) -> ProjectConfig:
    """Load and validate a project configuration (defaults to ``config/base.yaml``)."""
    config_path = Path(path) if path is not None else default_config_path()
    with config_path.open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    return ProjectConfig.model_validate(raw)
