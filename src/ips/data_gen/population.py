"""Populations of the synthetic network: issuers, acquirers, cardholders and merchants.

Observable attributes go to the output tables. Latent ones (lifestyle segment, activity,
preferences, clientele) are returned separately: the transaction process needs them, and
they are written under ``data/raw/latent/`` as ground truth, never as model features.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import polars as pl

from ips.data_gen.affinity import CNP, MerchantPools, draw_clientele, draw_preferences
from ips.data_gen.distributions import (
    categorical,
    lognormal_by_median,
    quantile_tiers,
    row_categorical,
)
from ips.data_gen.entities import table_schemas
from ips.utils.config import MERCHANT_CHANNELS, SIZE_TIERS, SPEND_SEGMENTS, ProjectConfig

_MIN_MARGIN, _MAX_MARGIN = 0.005, 0.9


def _pick(name: str, values: Sequence[str | None], index: np.ndarray) -> pl.Series:
    return pl.Series(name, list(values), dtype=pl.Utf8).gather(index)


@dataclass(frozen=True)
class CardholderLatent:
    """Per-card arrays used by the transaction process."""

    issuer_idx: np.ndarray
    product_idx: np.ndarray
    city_idx: np.ndarray
    segment_idx: np.ndarray
    activity: np.ndarray
    preferences: np.ndarray

    def to_frame(self, cfg: ProjectConfig) -> pl.DataFrame:
        """Latent table: segment, activity weight and preference over each MCC group."""
        segments = pl.Enum(list(cfg.lifestyle.segments))
        return pl.DataFrame(
            {
                "cardholder_id": np.arange(len(self.segment_idx), dtype=np.uint32),
                "segment": pl.Series(list(cfg.lifestyle.segments), dtype=segments).gather(
                    self.segment_idx
                ),
                "activity_weight": self.activity,
                **{f"pref_{g}": self.preferences[:, i] for i, g in enumerate(cfg.group_names)},
            }
        )


@dataclass(frozen=True)
class MerchantLatent:
    """Per-merchant arrays used by the sampler and the transaction process."""

    pools: MerchantPools
    mcc_idx: np.ndarray
    acquirer_idx: np.ndarray

    def to_frame(self, cfg: ProjectConfig) -> pl.DataFrame:
        """Latent table: clientele segment of each merchant."""
        segments = pl.Enum(list(cfg.lifestyle.segments))
        return pl.DataFrame(
            {
                "merchant_id": np.arange(len(self.mcc_idx), dtype=np.uint32),
                "clientele": pl.Series(list(cfg.lifestyle.segments), dtype=segments).gather(
                    self.pools.clientele
                ),
            }
        )


def build_issuers(cfg: ProjectConfig) -> pl.DataFrame:
    """Issuer dimension, taken directly from the configuration."""
    rows = [
        {
            "issuer_id": i.issuer_id,
            "issuer_type": i.issuer_type,
            "size": i.size,
            "country": cfg.geography.country,
            "bank_group": i.bank_group,
            "market_share": i.market_share,
        }
        for i in cfg.issuers
    ]
    return pl.DataFrame(rows, schema=table_schemas(cfg)["issuers"])


def build_acquirers(cfg: ProjectConfig) -> pl.DataFrame:
    """Acquirer dimension: the domestic acquirers plus the one serving merchants abroad."""
    foreign = cfg.cross_border.acquirer
    rows = [
        {
            "acquirer_id": a.acquirer_id,
            "acquirer_type": a.acquirer_type,
            "country": cfg.geography.country,
            "bank_group": a.bank_group,
            "pricing_model": a.pricing_model,
        }
        for a in cfg.acquirers
    ]
    rows.append(
        {
            "acquirer_id": foreign.acquirer_id,
            "acquirer_type": foreign.acquirer_type,
            "country": foreign.country,
            "bank_group": None,
            "pricing_model": foreign.pricing_model,
        }
    )
    return pl.DataFrame(rows, schema=table_schemas(cfg)["acquirers"])


def draw_cardholders(
    rng: np.random.Generator, cfg: ProjectConfig
) -> tuple[pl.DataFrame, CardholderLatent]:
    """Cards with issuer, product, city, spend segment and latent lifestyle."""
    n = cfg.sizes.n_cardholders
    products, segments = list(cfg.products), list(cfg.lifestyle.segments)
    cities = cfg.geography.city_weights()

    issuer_idx = categorical(rng, [i.market_share for i in cfg.issuers], n)
    # El producto depende del emisor: bancos grandes con más premium, fintech casi solo débito.
    product_rows = np.array(
        [[cfg.product_mix[i.product_mix][p] for p in products] for i in cfg.issuers]
    )
    product_idx = row_categorical(rng, product_rows, issuer_idx)
    city_idx = categorical(rng, list(cities.values()), n)
    spend_rows = np.array(
        [[cfg.spend_segments.mix[p][s] for s in SPEND_SEGMENTS] for p in products]
    )
    spend_idx = row_categorical(rng, spend_rows, product_idx)
    segment_rows = np.array([[cfg.lifestyle.mix[p][k] for k in segments] for p in products])
    segment_idx = row_categorical(rng, segment_rows, product_idx)
    medians = np.array([cfg.spend_segments.activity_median[s] for s in SPEND_SEGMENTS])
    activity = lognormal_by_median(rng, medians[spend_idx], cfg.spend_segments.activity_sigma, n)
    preferences = draw_preferences(rng, cfg, segment_idx)
    # Propensión cross-border: baja y concentrada en premium y comercial; el viajero la duplica.
    base = np.array([cfg.cross_border.propensity[p] for p in products])[product_idx]
    multiplier = np.array([cfg.cross_border.segment_multiplier.get(k, 1.0) for k in segments])
    propensity = np.minimum(base * multiplier[segment_idx], 1.0)

    frame = pl.DataFrame(
        {
            "cardholder_id": np.arange(n, dtype=np.uint32),
            "issuer_id": _pick("issuer_id", [i.issuer_id for i in cfg.issuers], issuer_idx),
            "product": _pick("product", products, product_idx),
            "country": [cfg.geography.country] * n,
            "city": _pick("city", list(cities), city_idx),
            "spend_segment": _pick("spend_segment", SPEND_SEGMENTS, spend_idx),
            "cross_border_propensity": propensity,
        }
    ).cast(dict(table_schemas(cfg)["cardholders"]))
    latent = CardholderLatent(issuer_idx, product_idx, city_idx, segment_idx, activity, preferences)
    return frame, latent


def draw_merchants(
    rng: np.random.Generator, cfg: ProjectConfig
) -> tuple[pl.DataFrame, MerchantLatent]:
    """Domestic merchants followed by foreign ones (for cross-border purchases)."""
    groups, mccs = cfg.group_names, cfg.mcc_groups.mccs
    group_of_mcc = np.array([groups.index(m.group) for m in mccs])
    n_dom, n_for = cfg.sizes.n_merchants, cfg.sizes.n_foreign_merchants
    n = n_dom + n_for
    cities = cfg.geography.city_weights()

    mcc_dom = categorical(rng, [m.merchant_share for m in mccs], n_dom)
    channel_rows = np.array([[m.channel_mix[c] for c in MERCHANT_CHANNELS] for m in mccs])
    channel_dom = row_categorical(rng, channel_rows, mcc_dom)
    city_dom = np.where(channel_dom == CNP, -1, categorical(rng, list(cities.values()), n_dom))

    # Comercios del exterior: pocos grupos (digital, viajes, retail) y casi todos online.
    xb = cfg.cross_border
    xb_groups = np.array([groups.index(g) for g in xb.groups])
    group_for = xb_groups[categorical(rng, list(xb.groups.values()), n_for)]
    mcc_by_group = np.array([[m.merchant_share * (m.group == g) for m in mccs] for g in groups])
    mcc_for = row_categorical(rng, mcc_by_group, group_for)
    country_for = categorical(rng, list(xb.countries.values()), n_for)
    fully_online = np.array([cfg.mcc_groups.groups[g].online_share >= 1.0 for g in groups])
    channel_draw = categorical(rng, [xb.merchant_channel[c] for c in MERCHANT_CHANNELS], n_for)
    channel_for = np.where(fully_online[group_for], CNP, channel_draw)

    mcc_idx = np.concatenate([mcc_dom, mcc_for])
    group_idx = group_of_mcc[mcc_idx]
    channel = np.concatenate([channel_dom, channel_for])
    city_idx = np.concatenate([city_dom, np.full(n_for, -1)])
    foreign = np.concatenate([np.zeros(n_dom, dtype=bool), np.ones(n_for, dtype=bool)])

    size_weight = lognormal_by_median(rng, 1.0, cfg.merchants.size_sigma, n)
    tier_idx = quantile_tiers(size_weight, [cfg.merchants.size_tiers[t] for t in SIZE_TIERS])
    # El adquirente depende solo del tamaño del comercio, nunca del emisor de sus clientes.
    tier_rows = np.array([[a.tier_shares[t] for a in cfg.acquirers] for t in SIZE_TIERS])
    acquirer_idx = np.where(foreign, len(cfg.acquirers), row_categorical(rng, tier_rows, tier_idx))
    margins = np.array([cfg.mcc_groups.groups[g].sector_margin for g in groups])[group_idx]
    sector_margin = np.clip(
        lognormal_by_median(rng, margins, cfg.merchants.sector_margin_sigma, n),
        _MIN_MARGIN,
        _MAX_MARGIN,
    )
    clientele = draw_clientele(rng, cfg, group_idx)

    acquirer_ids = [a.acquirer_id for a in cfg.acquirers] + [xb.acquirer.acquirer_id]
    country = np.concatenate([np.zeros(n_dom, dtype=np.int64), country_for + 1])
    frame = pl.DataFrame(
        {
            "merchant_id": np.arange(n, dtype=np.uint32),
            "mcc": _pick("mcc", cfg.mcc_codes, mcc_idx),
            "mcc_group": _pick("mcc_group", groups, group_idx),
            "country": _pick("country", [cfg.geography.country, *xb.countries], country),
            "city": _pick("city", [*cities, None], np.where(city_idx < 0, len(cities), city_idx)),
            "channel": _pick("channel", MERCHANT_CHANNELS, channel),
            "size_tier": _pick("size_tier", SIZE_TIERS, tier_idx),
            "size_weight": size_weight,
            "acquirer_id": _pick("acquirer_id", acquirer_ids, acquirer_idx),
            "sector_margin": sector_margin,
        }
    ).cast(dict(table_schemas(cfg)["merchants"]))
    pools = MerchantPools(group_idx, city_idx, channel, foreign, size_weight, clientele)
    return frame, MerchantLatent(pools, mcc_idx, acquirer_idx)


__all__ = [
    "CardholderLatent",
    "MerchantLatent",
    "build_acquirers",
    "build_issuers",
    "draw_cardholders",
    "draw_merchants",
]
