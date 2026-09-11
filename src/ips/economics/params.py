"""Economic parameters by actor and product, and the reference tables the warehouse reads.

The parameter models live in ``ips.utils.config`` (loaded from ``config/economics.yaml``).
This module turns them, together with the interchange table and the MCC groups, into the
small reference tables dbt joins. They are rewritten from the configuration before every
``dbt build``, so changing a fee never requires regenerating the transactions.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from ips.data_gen.entities import enum_dtypes
from ips.data_gen.interchange_table import build_interchange_table
from ips.utils.config import (
    EconomicsConfig,
    IssuerProductCosts,
    NetworkFeesConfig,
    ProjectConfig,
    SideFees,
)

__all__ = [
    "EconomicsConfig",
    "IssuerProductCosts",
    "NetworkFeesConfig",
    "SideFees",
    "mcc_dimension",
    "network_fees_table",
    "pricing_terms_table",
    "reference_tables",
    "write_reference_tables",
]


def mcc_dimension(cfg: ProjectConfig) -> pl.DataFrame:
    """One row per MCC with its group and the group's business parameters."""
    enums = enum_dtypes(cfg)
    rows = []
    for mcc in cfg.mcc_groups.mccs:
        group = cfg.mcc_groups.groups[mcc.group]
        rows.append(
            {
                "mcc": mcc.mcc,
                "name": mcc.name,
                "mcc_group": mcc.group,
                "sector_margin": group.sector_margin,
                "base_elasticity": group.base_elasticity,
                "blended_mdr_rate": group.blended_mdr_rate,
            }
        )
    return pl.DataFrame(rows).cast({"mcc": enums["mcc"], "mcc_group": enums["mcc_group"]})


def network_fees_table(cfg: ProjectConfig) -> pl.DataFrame:
    """The network's fee schedule for both sides, one row."""
    issuer, acquirer = cfg.economics.network.issuer, cfg.economics.network.acquirer
    return pl.DataFrame(
        {
            f"{side}_{field}": [float(getattr(fees, field))]
            for side, fees in (("issuer", issuer), ("acquirer", acquirer))
            for field in ("assessment_rate", "authorization_fee_cop", "cross_border_rate")
        }
    )


def pricing_terms_table(cfg: ProjectConfig) -> pl.DataFrame:
    """Pricing terms of each acquirer: the IC++ markup on top of interchange and network fees,
    and the surcharge on its blended rate for foreign cards."""
    terms = cfg.economics.acquirer
    acquirers = [a.acquirer_id for a in cfg.acquirers] + [cfg.cross_border.acquirer.acquirer_id]
    n = len(acquirers)
    return pl.DataFrame(
        {
            "acquirer_id": acquirers,
            "icpp_markup_rate": [terms.icpp_markup.rate] * n,
            "icpp_markup_fixed_cop": [terms.icpp_markup.fixed_cop] * n,
            "blended_cross_border_rate": [terms.blended_cross_border_surcharge] * n,
        }
    ).cast({"acquirer_id": enum_dtypes(cfg)["acquirer_id"]})


def reference_tables(cfg: ProjectConfig) -> dict[str, pl.DataFrame]:
    """Every configuration-derived table the warehouse joins, keyed by parquet stem."""
    return {
        "mccs": mcc_dimension(cfg),
        "interchange_table": build_interchange_table(cfg),
        "network_fees": network_fees_table(cfg),
        "pricing_terms": pricing_terms_table(cfg),
    }


def write_reference_tables(cfg: ProjectConfig, out_dir: Path) -> list[Path]:
    """Write the reference tables as parquet into ``out_dir`` and return their paths."""
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, frame in reference_tables(cfg).items():
        path = out_dir / f"{name}.parquet"
        frame.write_parquet(path)
        paths.append(path)
    return paths
