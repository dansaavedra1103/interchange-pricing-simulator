"""Configuration validation: every consistency check fails where it should."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import yaml
from pydantic import ValidationError

from ips.utils.config import ProjectConfig, default_config_path, load_config, resolve_path


def raw_config() -> dict[str, Any]:
    """The YAML files merged as ``load_config`` does, as a mutable dict."""
    data = yaml.safe_load(default_config_path().read_text(encoding="utf-8"))
    for key in ("mcc_groups", "interchange_table", "economics", "elasticity"):
        path = resolve_path(Path(data["includes"][key]))
        data[key] = yaml.safe_load(path.read_text(encoding="utf-8"))
    return data


def test_base_config_loads() -> None:
    cfg = load_config()
    assert len(cfg.products) == 4
    assert len(cfg.group_names) == 12
    assert len(cfg.mcc_codes) == 33
    assert set(cfg.interchange_table.base) == set(cfg.products)
    # 6M compras / 150k tarjetas / 2 años = 20 compras por tarjeta al año (Superfinanciera).
    years = cfg.dates.n_days / 365.25
    assert 18 <= cfg.sizes.n_transactions / cfg.sizes.n_cardholders / years <= 22


def _set(path: list[Any], value: Any) -> Callable[[dict[str, Any]], None]:
    def mutate(data: dict[str, Any]) -> None:
        node = data
        for key in path[:-1]:
            node = node[key]
        node[path[-1]] = value

    return mutate


def _delete(path: list[Any]) -> Callable[[dict[str, Any]], None]:
    def mutate(data: dict[str, Any]) -> None:
        node = data
        for key in path[:-1]:
            node = node[key]
        del node[path[-1]]

    return mutate


CASES = {
    "issuer_shares": (_set(["issuers", 0, "market_share"], 0.5), "issuers market_share"),
    "incomplete_grid": (
        _delete(["interchange_table", "base", "debit", "grocery"]),
        r"interchange_table.base\['debit'\] must cover",
    ),
    "unknown_key": (_set(["seed_typo"], 1), "Extra inputs are not permitted"),
    "rate_in_percent": (
        _set(["interchange_table", "base", "debit", "grocery", "rate"], 0.4),
        "less than 0.1",
    ),
    "unknown_group": (_set(["mcc_groups", "mccs", 0, "group"], "groceries"), "mccs group"),
    "unknown_mix": (_set(["issuers", 0, "product_mix"], "bank_xl"), "issuer product_mix"),
    "tier_shares": (_set(["acquirers", 0, "tier_shares", "large"], 0.5), "tier 'large'"),
    "home_abroad": (_set(["cross_border", "countries"], {"CO": 1.0}), "home country"),
    "weekday_length": (_set(["mcc_groups", "weekday_profiles", "flat"], [1, 1, 1]), "7 values"),
    "online_without_merchants": (
        _set(["mcc_groups", "groups", "fuel", "online_share"], 0.5),
        "sells online",
    ),
    "reversed_dates": (_set(["dates", "end"], "2023-01-01"), "must be before"),
    "issuer_costs_missing_product": (
        _delete(["economics", "issuer", "products", "debit"]),
        "economics.issuer.products must cover",
    ),
    "network_fee_in_percent": (
        _set(["economics", "network", "acquirer", "assessment_rate"], 0.13),
        "less than 0.1",
    ),
    "unknown_icpp_tier": (_set(["merchants", "icpp_tiers"], ["huge"]), "icpp_tiers"),
    "k_range_backwards": (_set(["segmentation", "k_range"], [12, 4]), "k_range"),
    "test_fraction_too_high": (_set(["segmentation", "test_fraction"], 1.5), "less than 1"),
    "small_ticket_product": (
        _set(["interchange_table", "small_ticket", "products"], ["prepaid"]),
        "small_ticket.products",
    ),
    "abandonment_outside_bounds": (
        _set(["elasticity", "acceptance", "target_annual_abandonment"], 0.9),
        "must lie strictly between",
    ),
    "candidates_below_top_k": (
        _set(["elasticity", "link_prediction", "candidate_merchants"], 5),
        "must exceed top_k",
    ),
    "unknown_spend_segment": (
        _set(["elasticity", "cardholder", "rewards_semi_elasticity", "premium"], 1.0),
        "premium",
    ),
}


@pytest.mark.parametrize("mutate, message", list(CASES.values()), ids=list(CASES))
def test_invalid_config_fails(mutate: Callable[[dict[str, Any]], None], message: str) -> None:
    data = raw_config()
    mutate(data)
    with pytest.raises(ValidationError, match=message):
        ProjectConfig.model_validate(data)
