"""Latent structure that gives the cardholder-merchant graph its communities.

Each card belongs to a latent lifestyle segment with its own preference over MCC groups, and
each merchant has a latent clientele segment. A purchase picks a group from the card's
preferences and then a merchant of that group: either one of the card's habitual merchants,
or a new one with weight size x (1 + boost if the merchant's clientele matches the card's
segment). The implied cardholder-merchant affinity matrix is never materialised: sampling
works on one CDF per (location, group, segment) cell.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ips.data_gen.distributions import dirichlet_rows, normalize, row_categorical
from ips.utils.config import ProjectConfig

CP, CNP, MIXED = 0, 1, 2


def segment_prototypes(cfg: ProjectConfig) -> np.ndarray:
    """Share of purchases per MCC group for each lifestyle segment (K x G, rows sum to 1)."""
    groups = cfg.group_names
    base = np.array([cfg.lifestyle.base_group_mix[g] for g in groups])
    rows = []
    for segment in cfg.lifestyle.segments:
        boosts = cfg.lifestyle.boosts.get(segment, {})
        rows.append(normalize(base * np.array([boosts.get(g, 1.0) for g in groups])))
    return np.vstack(rows)


def segment_shares(cfg: ProjectConfig) -> np.ndarray:
    """Expected share of cards in each lifestyle segment, from issuer and product mixes."""
    product_share = dict.fromkeys(cfg.products, 0.0)
    for issuer in cfg.issuers:
        for product, share in cfg.product_mix[issuer.product_mix].items():
            product_share[product] += issuer.market_share * share
    return np.array(
        [
            sum(product_share[p] * cfg.lifestyle.mix[p][segment] for p in cfg.products)
            for segment in cfg.lifestyle.segments
        ]
    )


def draw_preferences(
    rng: np.random.Generator, cfg: ProjectConfig, segment_idx: np.ndarray
) -> np.ndarray:
    """Per-card preference over MCC groups (n x G).

    Cada tarjeta reparte su gasto alrededor del prototipo de su segmento; la concentración de
    la Dirichlet fija cuánto se parecen entre sí las tarjetas de un mismo segmento.
    """
    alpha = cfg.lifestyle.preference_concentration * segment_prototypes(cfg)[segment_idx]
    return dirichlet_rows(rng, alpha)


def clientele_probabilities(cfg: ProjectConfig) -> np.ndarray:
    """P(clientele = k | group g) proportional to the demand of segment k in group g (G x K)."""
    demand = segment_shares(cfg)[:, None] * segment_prototypes(cfg)
    return (demand / demand.sum(axis=0, keepdims=True)).T


def draw_clientele(
    rng: np.random.Generator, cfg: ProjectConfig, group_idx: np.ndarray
) -> np.ndarray:
    """Latent clientele segment of each merchant, given its MCC group.

    Comercios de grupos distintos comparten clientela, y por eso las comunidades del grafo
    cruzan los MCC: sustitutos y complementarios que el MCC por sí solo no revela.
    """
    return row_categorical(rng, clientele_probabilities(cfg), group_idx)


@dataclass(frozen=True)
class MerchantPools:
    """Merchant attributes the sampler needs, one entry per merchant row."""

    group_idx: np.ndarray
    city_idx: np.ndarray  # -1 when the merchant has no physical store
    channel: np.ndarray  # CP, CNP or MIXED
    foreign: np.ndarray
    size_weight: np.ndarray
    clientele: np.ndarray


class AffinitySampler:
    """Draws merchants for (location, group, segment) triples with one ``searchsorted``.

    Locations are the home-country cities (0..C-1), then domestic online, abroad in person
    and abroad online. An empty cell falls back to the national in-person pool of the group,
    then to its online pool (or, abroad, to the other foreign pool).
    """

    def __init__(
        self, pools: MerchantPools, n_cities: int, n_groups: int, n_segments: int, boost: float
    ) -> None:
        self.n_cities = n_cities
        self.online = n_cities
        self.abroad_physical = n_cities + 1
        self.abroad_online = n_cities + 2
        n_locations = n_cities + 3

        physical = pools.channel != CNP
        online = pools.channel != CP
        domestic = ~pools.foreign
        lookup = np.full((n_locations, n_groups, n_segments), -1, dtype=np.int64)
        cdf_parts: list[np.ndarray] = []
        member_parts: list[np.ndarray] = []
        combo = 0
        for g in range(n_groups):
            in_group = pools.group_idx == g
            national = np.flatnonzero(in_group & domestic & physical)
            online_pool = np.flatnonzero(in_group & domestic & online)
            abroad_physical = np.flatnonzero(in_group & pools.foreign & physical)
            abroad_online = np.flatnonzero(in_group & pools.foreign & online)
            candidates = {
                c: [
                    np.flatnonzero(in_group & domestic & physical & (pools.city_idx == c)),
                    national,
                    online_pool,
                ]
                for c in range(n_cities)
            }
            candidates[self.online] = [online_pool, national]
            candidates[self.abroad_physical] = [abroad_physical, abroad_online]
            candidates[self.abroad_online] = [abroad_online, abroad_physical]
            for location, options in candidates.items():
                members = next((m for m in options if len(m)), None)
                if members is None:
                    continue
                for k in range(n_segments):
                    # Clientela que coincide con el segmento: más probable de ser elegida.
                    weights = pools.size_weight[members] * (
                        1.0 + boost * (pools.clientele[members] == k)
                    )
                    cdf = np.cumsum(weights) / weights.sum()
                    cdf[-1] = 1.0
                    cdf_parts.append(cdf + combo)
                    member_parts.append(members)
                    lookup[location, g, k] = combo
                    combo += 1
        self._lookup = lookup
        self._cdf = np.concatenate(cdf_parts)
        self._members = np.concatenate(member_parts)

    def draw(
        self,
        rng: np.random.Generator,
        location: np.ndarray,
        group: np.ndarray,
        segment: np.ndarray,
    ) -> np.ndarray:
        """Merchant row index for each (location, group, segment) triple."""
        combos = self._lookup[location, group, segment]
        if (combos < 0).any():
            raise ValueError("No merchant available for some (location, group) cells")
        positions = np.searchsorted(self._cdf, combos + rng.random(len(combos)), side="right")
        return self._members[np.minimum(positions, len(self._members) - 1)]


def draw_favorites(
    rng: np.random.Generator,
    sampler: AffinitySampler,
    home_city: np.ndarray,
    segment: np.ndarray,
    n_groups: int,
    n_favorites: int,
) -> np.ndarray:
    """Habitual merchants per card: shape (n_cards, n_groups, 2, n_favorites).

    Mode 0 is in person in the home city, mode 1 is online. Las compras repetidas en los
    mismos comercios producen aristas pesadas y la cola larga del grafo.
    """
    n = len(home_city)
    per_card = n_groups * 2 * n_favorites
    card = np.repeat(np.arange(n, dtype=np.int64), per_card)
    group = np.tile(np.repeat(np.arange(n_groups), 2 * n_favorites), n)
    mode = np.tile(np.repeat(np.array([0, 1]), n_favorites), n * n_groups)
    location = np.where(mode == 0, home_city[card], sampler.online)
    merchants = sampler.draw(rng, location, group, segment[card])
    return merchants.reshape(n, n_groups, 2, n_favorites)
