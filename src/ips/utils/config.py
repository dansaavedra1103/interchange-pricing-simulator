"""Typed configuration for the interchange pricing simulator.

``config/base.yaml`` holds the global and generator parameters and includes
``config/mcc_groups.yaml`` and ``config/interchange_table.yaml``. This module loads the three
files into frozen pydantic models and validates their consistency, so that a typo or a gap
fails at load time instead of silently distorting the generated data or the P&L.
"""

from __future__ import annotations

import datetime as dt
import math
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Annotated, Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

CHANNELS = ("cp", "cnp")
REGIONS = ("domestic", "cross_border")
MERCHANT_CHANNELS = ("cp", "cnp", "mixed")
SIZE_TIERS = ("large", "medium", "small", "micro")
SPEND_SEGMENTS = ("low", "mid", "high")
OTHER_CITIES = "other"

_SUM_TOLERANCE = 1e-6
# Sanity bound: catches rates written in percent (1.6) instead of as a fraction (0.016).
_MAX_RATE = 0.1

Rate = Annotated[float, Field(ge=0.0, lt=_MAX_RATE)]
Share = Annotated[float, Field(ge=0.0, le=1.0)]
Positive = Annotated[float, Field(gt=0.0)]
Channel = Literal["cp", "cnp"]
Region = Literal["domestic", "cross_border"]
MerchantChannel = Literal["cp", "cnp", "mixed"]
SizeTier = Literal["large", "medium", "small", "micro"]
SpendSegment = Literal["low", "mid", "high"]
PricingModel = Literal["blended", "icpp", "mixed"]


def project_root() -> Path:
    """Return the repository root: the closest parent of this file holding ``pyproject.toml``."""
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").is_file():
            return parent
    raise FileNotFoundError("Could not locate the project root (no pyproject.toml found).")


def default_config_path() -> Path:
    """Path to ``config/base.yaml`` inside the repository."""
    return project_root() / "config" / "base.yaml"


def resolve_path(path: Path) -> Path:
    """Resolve a configured path against the project root (absolute paths pass through)."""
    return path if path.is_absolute() else project_root() / path


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


def _check_subset(actual: Iterable[str], allowed: Iterable[str], label: str) -> None:
    unknown = sorted(set(actual) - set(allowed))
    if unknown:
        raise ValueError(f"{label} references unknown keys: {unknown}")


def _check_distribution(mapping: dict[str, float], keys: Iterable[str], label: str) -> None:
    _check_same_keys(mapping, keys, label)
    _check_sums_to_one(mapping.values(), label)


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


# ---------------------------------------------------------------------------
# base.yaml
# ---------------------------------------------------------------------------


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

    n_cardholders: int = Field(gt=0)
    n_merchants: int = Field(gt=0)
    n_foreign_merchants: int = Field(ge=0)
    n_transactions: int = Field(gt=0)


class OutputConfig(_Frozen):
    """Output locations, relative to the project root unless absolute."""

    raw_dir: Path
    warehouse_path: Path
    artifacts_dir: Path


class Includes(_Frozen):
    """Paths of the configuration files included by ``base.yaml``."""

    mcc_groups: Path
    interchange_table: Path
    economics: Path
    elasticity: Path
    simulator: Path


class Fee(_Frozen):
    """Percentage-plus-fixed fee: ``amount * rate + fixed_cop``."""

    rate: Rate
    fixed_cop: float = Field(default=0.0, ge=0.0)


class SideFees(_Frozen):
    """Fees the network charges one side of a transaction."""

    assessment_rate: Rate
    authorization_fee_cop: float = Field(ge=0.0)
    cross_border_rate: Rate


class NetworkFeesConfig(_Frozen):
    """The network's fee schedule for issuers and acquirers."""

    issuer: SideFees
    acquirer: SideFees


class AcquirerEconomics(_Frozen):
    """Pricing terms and processing cost of the acquirers.

    ``blended_cross_border_surcharge`` is added to the blended rate when the card is foreign
    to the merchant (a cross-border purchase).
    """

    icpp_markup: Fee
    blended_cross_border_surcharge: Rate
    processing: Fee


class IssuerProductCosts(_Frozen):
    """Costs of running a card of one product, as fractions of the purchase amount."""

    rewards_rate: Rate
    funding_rate: Rate
    credit_loss_rate: Rate
    processing_rate: Rate


class IssuerEconomics(_Frozen):
    """Issuer costs by card product."""

    products: dict[str, IssuerProductCosts]


class EconomicsConfig(_Frozen):
    """Contents of ``config/economics.yaml``."""

    network: NetworkFeesConfig
    acquirer: AcquirerEconomics
    issuer: IssuerEconomics


class GeographyConfig(_Frozen):
    """Home country, metro populations (millions) and travel behaviour."""

    country: str = Field(pattern=r"^[A-Z]{2}$")
    cities: dict[str, Positive] = Field(min_length=1)
    other_cities_share: Share
    p_home_city: Share

    def city_weights(self) -> dict[str, float]:
        """Share of cards and merchants per city; ``other`` collects the rest of the country."""
        total = math.fsum(self.cities.values())
        weights = {
            city: (1.0 - self.other_cities_share) * population / total
            for city, population in self.cities.items()
        }
        weights[OTHER_CITIES] = self.other_cities_share
        return weights


class IssuerConfig(_Frozen):
    """A card issuer. ``bank_group`` links it to an acquirer of the same group (on-us)."""

    issuer_id: str
    issuer_type: Literal["bank", "fintech"]
    size: Literal["large", "medium", "small"]
    bank_group: str | None = None
    product_mix: str
    market_share: Share


class SpendSegmentsConfig(_Frozen):
    """Spend segment mix by product and lognormal activity by segment."""

    mix: dict[str, dict[SpendSegment, Share]]
    activity_median: dict[SpendSegment, Positive]
    activity_sigma: Positive


class LifestyleConfig(_Frozen):
    """Latent lifestyle segments: mix by product and preferences over MCC groups."""

    mix: dict[str, dict[str, Share]]
    base_group_mix: dict[str, Share]
    boosts: dict[str, dict[str, Positive]]
    preference_concentration: Positive

    @property
    def segments(self) -> tuple[str, ...]:
        """Segment names, in the order of the first product row."""
        return tuple(next(iter(self.mix.values())))


class AffinityConfig(_Frozen):
    """Strength of the planted structure: clientele matching and habitual merchants."""

    clientele_boost: float = Field(ge=0.0)
    p_habit: Share
    favorite_weights: tuple[Positive, ...] = Field(min_length=1)


class MerchantConfig(_Frozen):
    """Merchant size and margin heterogeneity."""

    size_sigma: Positive
    size_tiers: dict[SizeTier, Share]
    sector_margin_sigma: float = Field(ge=0.0)
    icpp_tiers: tuple[SizeTier, ...] = Field(min_length=1)


class AcquirerConfig(_Frozen):
    """A domestic acquirer and its share of merchants in each size tier."""

    acquirer_id: str
    acquirer_type: Literal["bank", "aggregator"]
    bank_group: str | None = None
    pricing_model: PricingModel
    tier_shares: dict[SizeTier, Share]


class ForeignAcquirerConfig(_Frozen):
    """The acquirer of every merchant outside the home country."""

    acquirer_id: str
    acquirer_type: Literal["foreign"]
    country: str = Field(pattern=r"^[A-Z]{2}$")
    pricing_model: PricingModel


class ChannelConfig(_Frozen):
    """Security mix of card-not-present purchases."""

    p_tokenized_cnp: Share


class CrossBorderConfig(_Frozen):
    """Propensity to buy abroad and the population of foreign merchants."""

    propensity: dict[str, Share]
    segment_multiplier: dict[str, Positive]
    online_share: Share
    groups: dict[str, Share]
    countries: dict[str, Share]
    merchant_channel: dict[MerchantChannel, Share]
    acquirer: ForeignAcquirerConfig


class SeasonalityConfig(_Frozen):
    """Month factors, annual growth and payday boost of daily activity."""

    month_factors: tuple[Positive, ...] = Field(min_length=12, max_length=12)
    annual_growth: float = Field(gt=-1.0)
    payday_days: tuple[Annotated[int, Field(ge=1, le=31)], ...]
    payday_boost: Positive


class AmountsConfig(_Frozen):
    """Ticket floor and ticket multiplier by product."""

    min_amount_cop: int = Field(ge=1)
    product_multiplier: dict[str, Positive]


class SegmentationConfig(_Frozen):
    """Hyperparameters of the merchant segmentation (Feature 4)."""

    window_months: int = Field(gt=0)
    min_purchases: int = Field(gt=0)
    embedding_dims: int = Field(gt=1)
    ppmi_shift: Positive
    k_range: tuple[int, int]
    silhouette_sample: int = Field(gt=0)
    projection_top_merchants: int = Field(gt=1)
    projection_max_neighbors: int = Field(gt=0)
    projection_min_weight: int = Field(gt=0)
    leiden_resolution: Positive
    test_fraction: float = Field(gt=0.0, lt=1.0)
    max_cardholder_degree: int = Field(gt=0)

    @model_validator(mode="after")
    def _check_k_range(self) -> SegmentationConfig:
        low, high = self.k_range
        if not 2 <= low <= high:
            raise ValueError(f"segmentation.k_range must satisfy 2 <= low <= high: {self.k_range}")
        return self

    @property
    def k_values(self) -> tuple[int, ...]:
        """Candidate numbers of segments, both ends included."""
        return tuple(range(self.k_range[0], self.k_range[1] + 1))


class FraudConfig(_Frozen):
    """Fraud and chargeback probabilities."""

    base_rate: dict[Channel, Share]
    cross_border_multiplier: Positive
    tokenized_multiplier: Positive
    p_chargeback_given_fraud: Share


# ---------------------------------------------------------------------------
# mcc_groups.yaml
# ---------------------------------------------------------------------------


class GroupConfig(_Frozen):
    """Business parameters of an MCC group."""

    sector_margin: Annotated[float, Field(gt=0.0, lt=1.0)]
    base_elasticity: Positive
    blended_mdr_rate: Rate
    online_share: Share
    weekday_profile: str
    fraud_risk: Positive
    dispute_rate: Share


class MccConfig(_Frozen):
    """A merchant category: group, merchant share, ticket and channel mix."""

    mcc: str = Field(pattern=r"^\d{4}$")
    name: str
    group: str
    merchant_share: Share
    median_ticket_cop: Positive
    ticket_sigma: Positive
    channel_mix: dict[MerchantChannel, Share]


class MccGroupsConfig(_Frozen):
    """Contents of ``config/mcc_groups.yaml``."""

    groups: dict[str, GroupConfig] = Field(min_length=1)
    weekday_profiles: dict[str, tuple[Positive, ...]]
    mccs: tuple[MccConfig, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _check_consistency(self) -> MccGroupsConfig:
        _check_unique((m.mcc for m in self.mccs), "mcc")
        _check_sums_to_one((m.merchant_share for m in self.mccs), "mccs merchant_share")
        for name, profile in self.weekday_profiles.items():
            if len(profile) != 7:
                raise ValueError(f"weekday_profiles[{name!r}] needs 7 values (Monday..Sunday)")
        _check_subset(
            (g.weekday_profile for g in self.groups.values()),
            self.weekday_profiles,
            "groups weekday_profile",
        )
        _check_subset((m.group for m in self.mccs), self.groups, "mccs group")
        for mcc in self.mccs:
            _check_distribution(mcc.channel_mix, MERCHANT_CHANNELS, f"mcc {mcc.mcc} channel_mix")
        for name, group in self.groups.items():
            mccs = [m for m in self.mccs if m.group == name]
            if not mccs:
                raise ValueError(f"Group {name!r} has no MCC")
            online = sum(m.channel_mix["cnp"] + m.channel_mix["mixed"] for m in mccs)
            physical = sum(m.channel_mix["cp"] + m.channel_mix["mixed"] for m in mccs)
            if group.online_share > 0 and online == 0:
                raise ValueError(f"Group {name!r} sells online but has no cnp/mixed merchants")
            if group.online_share < 1 and physical == 0:
                raise ValueError(f"Group {name!r} sells in person but has no cp/mixed merchants")
        return self


# ---------------------------------------------------------------------------
# elasticity.yaml
# ---------------------------------------------------------------------------


class AcceptanceModifiers(_Frozen):
    """Shifts on the logit of the abandonment curve, one per driver."""

    cnp_share: float
    surcharge_allowed_shift: float
    icpp_sensitivity: Positive


class InstantPaymentsConfig(_Frozen):
    """Pressure from instant rails on the merchants they can substitute (spec §1.9)."""

    enabled: bool
    max_shift: float = Field(ge=0.0)
    small_ticket_cop: Positive


class AcceptanceConfig(_Frozen):
    """Merchant acceptance curve: its two anchor figures, the group levels and the modifiers."""

    target_annual_abandonment: Annotated[float, Field(gt=0.0, lt=1.0)]
    target_semi_elasticity: Annotated[float, Field(gt=0.0, lt=1.0)]
    semi_elasticity_shock: Positive
    use_group_elasticity: bool
    level_burden_exponent: float = Field(ge=0.0)
    modifiers: AcceptanceModifiers
    instant_payments: InstantPaymentsConfig
    surcharge_allowed: bool
    min_probability: Share
    max_probability: Share

    @model_validator(mode="after")
    def _check_bounds(self) -> AcceptanceConfig:
        if not self.min_probability < self.target_annual_abandonment < self.max_probability:
            raise ValueError(
                "acceptance.target_annual_abandonment must lie strictly between "
                f"min_probability and max_probability: {self.target_annual_abandonment}"
            )
        return self


class CardholderResponseConfig(_Frozen):
    """Spend response to a change in the rewards rate, by spend segment."""

    rewards_semi_elasticity: dict[SpendSegment, Positive]
    max_spend_change: Annotated[float, Field(gt=0.0, lt=1.0)]


class GnnConfig(_Frozen):
    """Hyperparameters of the GraphSAGE link-prediction model."""

    hidden_dims: int = Field(gt=1)
    layers: int = Field(gt=0)
    learning_rate: Positive
    negatives_per_positive: int = Field(gt=0)
    train_edges: int = Field(gt=0)
    max_epochs: int = Field(gt=0)
    patience: int = Field(gt=0)
    min_delta: float = Field(ge=0.0)


class LinkPredictionConfig(_Frozen):
    """Protocol and budget of the link-prediction ladder."""

    holdout_months: int = Field(gt=0)
    top_k: int = Field(gt=0)
    eval_cardholders: int = Field(gt=0)
    min_new_links: int = Field(gt=0)
    candidate_merchants: int = Field(gt=0)
    train_cardholders: int = Field(gt=0)
    bootstrap_samples: int = Field(gt=0)
    confidence_level: float = Field(gt=0.0, lt=1.0)
    gnn: GnnConfig

    @model_validator(mode="after")
    def _check_candidates(self) -> LinkPredictionConfig:
        if self.candidate_merchants <= self.top_k:
            raise ValueError("link_prediction.candidate_merchants must exceed top_k")
        return self


class RedistributionConfig(_Frozen):
    """How a dropped merchant's volume is spread over its substitutes."""

    leakage_share: Share
    top_substitutes: int = Field(gt=0)


class ElasticityConfig(_Frozen):
    """Contents of ``config/elasticity.yaml``."""

    acceptance: AcceptanceConfig
    cardholder: CardholderResponseConfig
    link_prediction: LinkPredictionConfig
    redistribution: RedistributionConfig


# ---------------------------------------------------------------------------
# simulator.yaml
# ---------------------------------------------------------------------------


class SimulatorConfig(_Frozen):
    """Scenario simulator: the base year and how interchange changes pass through."""

    base_months: int = Field(gt=0)
    blended_pass_through: Share
    rewards_pass_through: Share
    scenarios_dir: Path


# ---------------------------------------------------------------------------
# interchange_table.yaml
# ---------------------------------------------------------------------------


class Adjustment(_Frozen):
    """Add-on to the base cell: ``rate + rate_add`` and ``fixed_cop + fixed_add_cop``."""

    rate_add: float = Field(default=0.0, gt=-_MAX_RATE, lt=_MAX_RATE)
    fixed_add_cop: float = 0.0


class Adjustments(_Frozen):
    """Channel and region add-ons."""

    channel: dict[Channel, Adjustment] = Field(default_factory=dict)
    region: dict[Region, Adjustment] = Field(default_factory=dict)


class SmallTicketConfig(_Frozen):
    """Reduced tier for small domestic amounts."""

    max_amount_cop: Positive
    products: tuple[str, ...]
    groups: tuple[str, ...]
    rate: Rate
    fixed_cop: float = Field(default=0.0, ge=0.0)


class OverrideConfig(_Frozen):
    """An explicit cell that replaces base + add-ons."""

    product: str
    mcc_group: str
    channel: Channel
    region: Region
    rate: Rate
    fixed_cop: float = Field(default=0.0, ge=0.0)


class InterchangeTableConfig(_Frozen):
    """Contents of ``config/interchange_table.yaml``."""

    version: str
    valid_from: dt.date
    valid_to: dt.date | None = None
    base: dict[str, dict[str, Fee]]
    adjustments: Adjustments = Field(default_factory=Adjustments)
    small_ticket: SmallTicketConfig | None = None
    overrides: tuple[OverrideConfig, ...] = ()

    @model_validator(mode="after")
    def _check_validity(self) -> InterchangeTableConfig:
        if self.valid_to is not None and self.valid_to <= self.valid_from:
            raise ValueError("interchange_table.valid_to must be after valid_from")
        return self


# ---------------------------------------------------------------------------
# Root
# ---------------------------------------------------------------------------


class ProjectConfig(_Frozen):
    """Root configuration: ``base.yaml`` plus its included files."""

    seed: int
    dates: DateRange
    sizes: Sizes
    sample_sizes: Sizes
    output: OutputConfig
    includes: Includes
    products: tuple[str, ...] = Field(min_length=1)
    geography: GeographyConfig
    issuers: tuple[IssuerConfig, ...] = Field(min_length=1)
    product_mix: dict[str, dict[str, Share]]
    spend_segments: SpendSegmentsConfig
    lifestyle: LifestyleConfig
    affinity: AffinityConfig
    merchants: MerchantConfig
    acquirers: tuple[AcquirerConfig, ...] = Field(min_length=1)
    channel: ChannelConfig
    cross_border: CrossBorderConfig
    seasonality: SeasonalityConfig
    amounts: AmountsConfig
    fraud: FraudConfig
    segmentation: SegmentationConfig
    mcc_groups: MccGroupsConfig
    interchange_table: InterchangeTableConfig
    economics: EconomicsConfig
    elasticity: ElasticityConfig
    simulator: SimulatorConfig

    @property
    def group_names(self) -> tuple[str, ...]:
        """MCC group names in configuration order."""
        return tuple(self.mcc_groups.groups)

    @property
    def mcc_codes(self) -> tuple[str, ...]:
        """MCC codes in configuration order."""
        return tuple(m.mcc for m in self.mcc_groups.mccs)

    @model_validator(mode="after")
    def _check_consistency(self) -> ProjectConfig:
        products, groups = self.products, self.group_names
        segments = self.lifestyle.segments
        _check_unique(products, "products")
        _check_unique((i.issuer_id for i in self.issuers), "issuer_id")
        _check_unique(
            [a.acquirer_id for a in self.acquirers] + [self.cross_border.acquirer.acquirer_id],
            "acquirer_id",
        )

        _check_sums_to_one((i.market_share for i in self.issuers), "issuers market_share")
        _check_subset((i.product_mix for i in self.issuers), self.product_mix, "issuer product_mix")
        for name, mix in self.product_mix.items():
            _check_distribution(mix, products, f"product_mix[{name!r}]")

        _check_same_keys(self.spend_segments.mix, products, "spend_segments.mix")
        for product, mix in self.spend_segments.mix.items():
            _check_distribution(mix, SPEND_SEGMENTS, f"spend_segments.mix[{product!r}]")
        _check_same_keys(
            self.spend_segments.activity_median, SPEND_SEGMENTS, "spend_segments.activity_median"
        )

        _check_same_keys(self.lifestyle.mix, products, "lifestyle.mix")
        for product, mix in self.lifestyle.mix.items():
            _check_distribution(mix, segments, f"lifestyle.mix[{product!r}]")
        _check_distribution(self.lifestyle.base_group_mix, groups, "lifestyle.base_group_mix")
        _check_subset(self.lifestyle.boosts, segments, "lifestyle.boosts")
        for segment, boosts in self.lifestyle.boosts.items():
            _check_subset(boosts, groups, f"lifestyle.boosts[{segment!r}]")

        _check_distribution(self.merchants.size_tiers, SIZE_TIERS, "merchants.size_tiers")
        for acquirer in self.acquirers:
            _check_same_keys(acquirer.tier_shares, SIZE_TIERS, f"{acquirer.acquirer_id} tiers")
        for tier in SIZE_TIERS:
            _check_sums_to_one((a.tier_shares[tier] for a in self.acquirers), f"tier {tier!r}")

        xb = self.cross_border
        _check_same_keys(xb.propensity, products, "cross_border.propensity")
        _check_subset(xb.segment_multiplier, segments, "cross_border.segment_multiplier")
        _check_subset(xb.groups, groups, "cross_border.groups")
        _check_sums_to_one(xb.groups.values(), "cross_border.groups")
        _check_sums_to_one(xb.countries.values(), "cross_border.countries")
        if self.geography.country in xb.countries:
            raise ValueError("cross_border.countries must not include the home country")
        _check_distribution(xb.merchant_channel, MERCHANT_CHANNELS, "cross_border.merchant_channel")

        _check_same_keys(self.amounts.product_multiplier, products, "amounts.product_multiplier")
        _check_same_keys(self.fraud.base_rate, CHANNELS, "fraud.base_rate")

        table = self.interchange_table
        _check_same_keys(table.base, products, "interchange_table.base")
        for product, cells in table.base.items():
            _check_same_keys(cells, groups, f"interchange_table.base[{product!r}]")
        if table.small_ticket is not None:
            _check_subset(table.small_ticket.products, products, "small_ticket.products")
            _check_subset(table.small_ticket.groups, groups, "small_ticket.groups")
        for override in table.overrides:
            _check_subset([override.product], products, "override product")
            _check_subset([override.mcc_group], groups, "override mcc_group")

        _check_same_keys(self.economics.issuer.products, products, "economics.issuer.products")
        return self


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_config(path: Path | None = None) -> ProjectConfig:
    """Load ``base.yaml`` (default: ``config/base.yaml``) with its included files."""
    config_path = Path(path) if path is not None else default_config_path()
    data = _read_yaml(config_path)
    includes = Includes.model_validate(data.get("includes", {}))
    data["mcc_groups"] = _read_yaml(resolve_path(includes.mcc_groups))
    data["interchange_table"] = _read_yaml(resolve_path(includes.interchange_table))
    data["economics"] = _read_yaml(resolve_path(includes.economics))
    data["elasticity"] = _read_yaml(resolve_path(includes.elasticity))
    data["simulator"] = _read_yaml(resolve_path(includes.simulator))
    return ProjectConfig.model_validate(data)
