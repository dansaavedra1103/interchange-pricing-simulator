"""Transaction process: who buys, where, when, how much and through which channel.

For each purchase: the card comes from a multinomial allocation by activity; the MCC group
from the card's preferences; the place (home city, another city, online or abroad) from the
group's online share, the card's travel and cross-border behaviour; the merchant from the
card's habitual merchants or from the affinity sampler; the date from a seasonal profile of
the group; and the amount from a lognormal ticket of the MCC scaled by the card product.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from ips.data_gen.affinity import CNP, CP, AffinitySampler
from ips.data_gen.distributions import allocate_counts, categorical, row_categorical
from ips.data_gen.population import CardholderLatent, MerchantLatent
from ips.utils.config import CHANNELS, ProjectConfig


def day_weights(cfg: ProjectConfig) -> np.ndarray:
    """Relative activity of each day for each MCC group (G x n_days)."""
    days = pl.date_range(cfg.dates.start, cfg.dates.end, eager=True)
    month = days.dt.month().to_numpy() - 1
    weekday = days.dt.weekday().to_numpy() - 1  # Monday = 0
    day_of_month = days.dt.day().to_numpy()
    season = cfg.seasonality
    elapsed_years = np.arange(len(days)) / 365.25
    # Mes (pico en diciembre), tendencia de crecimiento anual y quincena, comunes a todos.
    common = (
        np.asarray(season.month_factors)[month]
        * (1.0 + season.annual_growth) ** elapsed_years
        * np.where(np.isin(day_of_month, season.payday_days), season.payday_boost, 1.0)
    )
    profiles = cfg.mcc_groups.weekday_profiles
    by_group = np.array(
        [profiles[cfg.mcc_groups.groups[g].weekday_profile] for g in cfg.group_names]
    )
    return common[None, :] * by_group[:, weekday]


def generate_transactions(
    rng: np.random.Generator,
    cfg: ProjectConfig,
    cardholders: pl.DataFrame,
    card_latent: CardholderLatent,
    merchants: pl.DataFrame,
    merchant_latent: MerchantLatent,
    sampler: AffinitySampler,
    favorites: np.ndarray,
) -> pl.DataFrame:
    """Generate every purchase (without fraud flags), sorted by date."""
    groups = cfg.mcc_groups.groups
    card = np.repeat(
        np.arange(cfg.sizes.n_cardholders),
        allocate_counts(rng, cfg.sizes.n_transactions, card_latent.activity),
    )
    n = len(card)
    segment = card_latent.segment_idx[card]

    # Cross-border: la compra ocurre en el exterior con la propensión de la tarjeta, y el
    # grupo sale de sus preferencias restringidas a los grupos que se compran afuera.
    abroad = rng.random(n) < cardholders["cross_border_propensity"].to_numpy()[card]
    group = row_categorical(rng, card_latent.preferences, card)
    xb_groups = np.array([cfg.group_names.index(g) for g in cfg.cross_border.groups])
    group[abroad] = xb_groups[
        row_categorical(rng, card_latent.preferences[:, xb_groups], card[abroad])
    ]

    online_share = np.array([groups[g].online_share for g in cfg.group_names])
    online = np.where(
        abroad,
        rng.random(n) < cfg.cross_border.online_share,
        rng.random(n) < online_share[group],
    )
    # Compras presenciales: casi siempre en la ciudad de residencia; el resto, de viaje.
    home = card_latent.city_idx[card]
    at_home = rng.random(n) < cfg.geography.p_home_city
    travel_city = categorical(rng, list(cfg.geography.city_weights().values()), n)
    city = np.where(at_home, home, travel_city)
    location = np.select(
        [abroad & online, abroad, online],
        [sampler.abroad_online, sampler.abroad_physical, sampler.online],
        default=city,
    )

    # Comercio: habitual con p_habit (en su ciudad o online), si no uno nuevo por afinidad.
    habitual = ~abroad & (online | (city == home)) & (rng.random(n) < cfg.affinity.p_habit)
    weights = cfg.affinity.favorite_weights
    favorite = categorical(rng, weights, int(habitual.sum()))
    merchant = np.empty(n, dtype=np.int64)
    merchant[habitual] = favorites[
        card[habitual], group[habitual], online[habitual].astype(np.int64), favorite
    ]
    explore = ~habitual
    merchant[explore] = sampler.draw(rng, location[explore], group[explore], segment[explore])

    # El canal lo decide la compra, salvo comercios que solo venden de una forma.
    merchant_channel = merchant_latent.pools.channel[merchant]
    is_cnp = np.where(
        merchant_channel == CNP, True, np.where(merchant_channel == CP, False, online)
    )
    tokenized = is_cnp & (rng.random(n) < cfg.channel.p_tokenized_cnp)

    day = row_categorical(rng, day_weights(cfg), merchant_latent.pools.group_idx[merchant])
    mccs = cfg.mcc_groups.mccs
    mcc = merchant_latent.mcc_idx[merchant]
    median = np.array([m.median_ticket_cop for m in mccs])[mcc]
    sigma = np.array([m.ticket_sigma for m in mccs])[mcc]
    product_multiplier = np.array([cfg.amounts.product_multiplier[p] for p in cfg.products])
    # Ticket log-normal por MCC; premium y comercial compran más caro que débito.
    amount = (
        median
        * product_multiplier[card_latent.product_idx[card]]
        * np.exp(sigma * rng.standard_normal(n))
    )
    amount = np.maximum(np.round(amount), cfg.amounts.min_amount_cop).astype(np.int64)

    # On-us: emisor y adquirente del mismo grupo financiero (fintech y agregadores no tienen).
    bank_groups = sorted({i.bank_group for i in cfg.issuers if i.bank_group})
    issuer_group = np.array(
        [bank_groups.index(i.bank_group) if i.bank_group else -1 for i in cfg.issuers]
    )
    acquirer_group = np.array(
        [
            bank_groups.index(a.bank_group) if a.bank_group in bank_groups else -2
            for a in cfg.acquirers
        ]
        + [-2]
    )
    txn_issuer_group = issuer_group[card_latent.issuer_idx[card]]
    on_us = txn_issuer_group == acquirer_group[merchant_latent.acquirer_idx[merchant]]

    channel_enum = pl.Enum(list(CHANNELS))
    frame = pl.DataFrame(
        {
            "txn_date": np.datetime64(cfg.dates.start, "D") + day,
            "cardholder_id": cardholders["cardholder_id"].gather(card),
            "merchant_id": merchants["merchant_id"].gather(merchant),
            "issuer_id": cardholders["issuer_id"].gather(card),
            "acquirer_id": merchants["acquirer_id"].gather(merchant),
            "product": cardholders["product"].gather(card),
            "mcc": merchants["mcc"].gather(merchant),
            "mcc_group": merchants["mcc_group"].gather(merchant),
            "channel": pl.Series(list(CHANNELS), dtype=channel_enum).gather(
                is_cnp.astype(np.int64)
            ),
            "tokenized": tokenized,
            "cross_border": merchant_latent.pools.foreign[merchant],
            "on_us": on_us,
            "amount_cop": amount,
        }
    )
    # Orden aleatorio dentro de cada día y txn_id secuencial por fecha.
    return frame[rng.permutation(n)].sort("txn_date", maintain_order=True).with_row_index("txn_id")


__all__ = ["day_weights", "generate_transactions"]
