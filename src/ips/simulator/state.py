"""The network at rest: the base year, priced, with everything a scenario is compared to.

Built once and shared by every scenario, so a run prices only what the scenario changes. The
acceptance curve is calibrated on the base year's own population — the same merchants, at the
same prices, that the scenarios reprice.
"""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl

from ips.data_gen.interchange_table import build_interchange_table
from ips.economics.pnl import PnLResult, compute_pnl
from ips.elasticity.calibration import calibrate_acceptance
from ips.elasticity.merchant_acceptance import AcceptanceCurve, abandonment_probability
from ips.elasticity.pipeline import ARTIFACTS as ELASTICITY_ARTIFACTS
from ips.graph.features import clustered_population, merchant_profile
from ips.utils.config import ProjectConfig, load_config
from ips.utils.frames import stable_group_sums
from ips.utils.io import artifacts_dir, read_raw

_AMOUNT = pl.col("amount_cop").cast(pl.Float64)


@dataclass(frozen=True)
class NetworkState:
    """The priced base year and what the engine needs to compare a scenario against it."""

    cfg: ProjectConfig
    transactions: pl.DataFrame
    merchants: pl.DataFrame
    cardholders: pl.DataFrame
    substitutes: pl.DataFrame
    table: pl.DataFrame
    baseline: PnLResult
    profile: pl.DataFrame
    curve: AcceptanceCurve
    base_abandonment: pl.DataFrame

    def with_config(self, cfg: ProjectConfig) -> NetworkState:
        """The same base year under another configuration: prices, curve and abandonment again.

        El año base no cambia (sería otro conjunto de compras), así que un cambio de
        ``simulator.base_months`` exige construir el estado desde cero.
        """
        if cfg.simulator.base_months != self.cfg.simulator.base_months:
            raise ValueError("A different base year needs build_state, not with_config")
        return _assemble(
            self.transactions, self.merchants, self.cardholders, self.substitutes, cfg, self.table
        )


def base_year(transactions: pl.DataFrame, cfg: ProjectConfig) -> pl.DataFrame:
    """The last ``simulator.base_months`` months of purchases, counted from the latest month."""
    last = transactions["txn_date"].max()
    if last is None:
        raise ValueError("No transactions to build a base year from")
    months = cfg.simulator.base_months
    first = pl.lit(last).dt.truncate("1mo").dt.offset_by(f"-{months - 1}mo")
    return transactions.filter(pl.col("txn_date") >= first)


def merchant_burden(priced: pl.DataFrame, profile: pl.DataFrame, weight: pl.Expr) -> pl.DataFrame:
    """Relative burden and volume of every merchant, from priced transactions and their weights.

    La misma función mide la carga de la base (peso 1) y la de cada escenario, así que un
    escenario sin cambios reproduce la probabilidad de abandono de la base bit a bit. Un
    comercio que se quedó sin volumen conserva la carga de la base: no hay precio que leer.
    """
    rates = stable_group_sums(
        priced,
        "merchant_id",
        (pl.col("mdr_cop") * weight).alias("_mdr"),
        (_AMOUNT * weight).alias("gdv_cop"),
    )
    return (
        profile.select("merchant_id", "sector_margin", pl.col("relative_burden").alias("_base"))
        .join(rates, on="merchant_id", how="left", maintain_order="left")
        .with_columns(pl.col("gdv_cop").fill_null(0.0))
        .with_columns(
            pl.when(pl.col("gdv_cop") > 0)
            .then(pl.col("_mdr") / pl.col("gdv_cop") / pl.col("sector_margin"))
            .otherwise(pl.col("_base"))
            .alias("relative_burden")
        )
        .select("merchant_id", "relative_burden", "gdv_cop")
    )


def with_burden(profile: pl.DataFrame, burden: pl.DataFrame) -> pl.DataFrame:
    """The profile with its relative burden replaced by another one."""
    return profile.drop("relative_burden").join(
        burden.select("merchant_id", "relative_burden"),
        on="merchant_id",
        how="left",
        maintain_order="left",
    )


def _assemble(
    year: pl.DataFrame,
    merchants: pl.DataFrame,
    cardholders: pl.DataFrame,
    substitutes: pl.DataFrame,
    cfg: ProjectConfig,
    table: pl.DataFrame,
) -> NetworkState:
    baseline = compute_pnl(year, table, cfg)
    profile = merchant_profile(baseline.transactions, merchants, cfg)
    profile = with_burden(profile, merchant_burden(baseline.transactions, profile, pl.lit(1.0)))
    curve = calibrate_acceptance(clustered_population(profile, cfg), cfg)
    abandonment = abandonment_probability(profile, cfg, curve).select(
        "merchant_id", pl.col("mcc_group").cast(pl.Utf8), "p_abandonment"
    )
    return NetworkState(
        cfg=cfg,
        transactions=year,
        merchants=merchants,
        cardholders=cardholders,
        substitutes=substitutes,
        table=table,
        baseline=baseline,
        profile=profile,
        curve=curve,
        base_abandonment=abandonment,
    )


def build_state(
    transactions: pl.DataFrame,
    merchants: pl.DataFrame,
    cardholders: pl.DataFrame,
    substitutes: pl.DataFrame,
    cfg: ProjectConfig,
    table: pl.DataFrame | None = None,
) -> NetworkState:
    """Price the base year and calibrate everything a scenario is read against."""
    year = base_year(transactions, cfg)
    if "pricing_model" not in year.columns:
        year = year.join(
            merchants.select("merchant_id", "pricing_model"),
            on="merchant_id",
            how="left",
            validate="m:1",
            maintain_order="left",
        )
    return _assemble(
        year,
        merchants,
        cardholders,
        substitutes,
        cfg,
        table if table is not None else build_interchange_table(cfg),
    )


def load_state(cfg: ProjectConfig | None = None) -> NetworkState:
    """The state from the raw parquet and the substitutes Feature 5 learned."""
    config = cfg if cfg is not None else load_config()
    path = artifacts_dir(config) / ELASTICITY_ARTIFACTS["substitutes"]
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is missing. Run `python -m ips.tasks elasticity` first: the simulator moves "
            "lost volume to the substitutes it learns."
        )
    return build_state(
        read_raw("transactions", config),
        read_raw("merchants", config),
        read_raw("cardholders", config),
        pl.read_parquet(path),
        config,
    )
