"""The scenario engine: one change in prices, carried through every actor of the network.

``Simulator.run`` walks the chain of spec §2.6 once, on the base year:

1. **Price.** The scenario's levers change interchange, split migrated volume between products
   and take substituted purchases off the rails; every transaction is priced again.
2. **Blended pass-through.** Acquirers move each group's blended rate by a share of the change
   in that group's effective interchange. IC++ merchants already see the whole change.
3. **Abandonment.** Each merchant's new relative burden goes through the calibrated curve; a
   scenario costs a merchant only the extra abandonment its prices cause, delta p.
4. **Redistribution.** The volume delta p removes — or keeps, when it is negative — moves to the
   merchant's substitutes, after a share leaks off the card rails.
5. **Rewards.** Issuers move each product's rewards rate by a share of the change in its
   effective interchange, never below zero; a product without rewards does not start paying them.
6. **Spend.** Cardholders answer the new rewards rate, by spend segment.
7. **P&L.** Every transaction keeps its new prices and carries a weight: substitution and
   migration, its merchant's abandonment and redistribution, its cardholder's spend. Every line
   of the P&L is linear per transaction, so issuer + network + acquirer = MDR holds row by row
   whatever the weight.

One pass, no equilibrium: second-order effects, such as the abandonment of a substitute that
received volume, are left out and documented in docs/assumptions.md.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import polars as pl

from ips.economics.pnl import compute_pnl
from ips.economics.revenue_bridge import revenue_bridge
from ips.elasticity.cardholder_response import segment_response
from ips.elasticity.link_prediction import redistribute_volume, redistribution_leakage
from ips.elasticity.merchant_acceptance import abandonment_probability
from ips.simulator.result import WEIGHT, SimulationResult
from ips.simulator.scenario import (
    VOLUME_WEIGHT,
    Scenario,
    TransactionFilter,
    apply_volume_levers,
    interchange_adjustment,
)
from ips.simulator.state import NetworkState, merchant_burden, with_burden
from ips.utils.config import ProjectConfig
from ips.utils.frames import stable_group_sums

BRIDGE_KEYS = ("product", "mcc_group", "channel")
_AMOUNT = pl.col("amount_cop").cast(pl.Float64)


def effective_interchange_change(
    base: pl.DataFrame,
    scenario: pl.DataFrame,
    key: str,
    subset: pl.Expr | None = None,
) -> dict[str, float]:
    """Change in effective interchange by one key: scenario (volume-weighted) minus base."""
    if subset is not None:
        base, scenario = base.filter(subset), scenario.filter(subset)
    group = pl.col(key).cast(pl.Utf8)
    before = stable_group_sums(
        base, group, pl.col("interchange_cop").alias("interchange"), _AMOUNT.alias("volume")
    )
    after = stable_group_sums(
        scenario,
        group,
        (pl.col("interchange_cop") * pl.col(VOLUME_WEIGHT)).alias("interchange"),
        (_AMOUNT * pl.col(VOLUME_WEIGHT)).alias("volume"),
    )
    rows = before.join(after, on=key, how="inner", suffix="_new").filter(
        (pl.col("volume") > 0) & (pl.col("volume_new") > 0)
    )
    return {
        row[key]: row["interchange_new"] / row["volume_new"] - row["interchange"] / row["volume"]
        for row in rows.iter_rows(named=True)
    }


def passed_through(
    cfg: ProjectConfig, blended_change: dict[str, float], rewards_change: dict[str, float]
) -> tuple[ProjectConfig, pl.DataFrame]:
    """The configuration with the new blended and rewards rates, and a frame of what moved."""
    simulator = cfg.simulator
    rows = []
    groups = {}
    for name, group in cfg.mcc_groups.groups.items():
        shift = simulator.blended_pass_through * blended_change.get(name, 0.0)
        rate = max(0.0, group.blended_mdr_rate + shift)
        groups[name] = group.model_copy(update={"blended_mdr_rate": rate})
        rows.append(("blended_mdr_rate", name, group.blended_mdr_rate, rate))
    products = {}
    for name, costs in cfg.economics.issuer.products.items():
        # El traslado solo mueve recompensas que ya existen: un producto sin programa (débito) no
        # lo crea porque su interchange promedio suba. Y no pueden quedar negativas: un tope
        # agresivo las lleva a cero, no a un cobro al tarjetahabiente.
        shift = simulator.rewards_pass_through * rewards_change.get(name, 0.0)
        rate = costs.rewards_rate
        if rate > 0:
            rate = max(0.0, rate + shift)
        products[name] = costs.model_copy(update={"rewards_rate": rate})
        rows.append(("rewards_rate", name, costs.rewards_rate, rate))
    issuer = cfg.economics.issuer.model_copy(update={"products": products})
    updated = cfg.model_copy(
        update={
            "mcc_groups": cfg.mcc_groups.model_copy(update={"groups": groups}),
            "economics": cfg.economics.model_copy(update={"issuer": issuer}),
        }
    )
    prices = pl.DataFrame(
        rows,
        schema={"price": pl.Utf8, "key": pl.Utf8, "base": pl.Float64, "scenario": pl.Float64},
        orient="row",
    )
    return updated, prices


@dataclass(frozen=True)
class Simulator:
    """Runs scenarios against one network state, built once."""

    state: NetworkState

    def top_merchants(self, count: int) -> list[int]:
        """The merchants with the most volume in the base year, largest first."""
        volume = (
            self.state.baseline.transactions.group_by("merchant_id")
            .agg(_AMOUNT.sum().alias("gdv"))
            .sort(["gdv", "merchant_id"], descending=[True, False])
        )
        return volume.head(count)["merchant_id"].to_list()

    def _resolve(self, where: TransactionFilter) -> list[int] | None:
        if where.top_merchants_by_gdv is None:
            return None
        return self.top_merchants(where.top_merchants_by_gdv)

    def run(self, scenario: Scenario) -> SimulationResult:
        """Carry one scenario through the whole chain and compare it with the base year."""
        start = time.perf_counter()
        state, cfg = self.state, self.state.cfg
        base = state.baseline.transactions

        # 1. Tarifa: las palancas de volumen y de interchange, y una primera tarificación.
        frame = apply_volume_levers(state.transactions, scenario, self._resolve)
        adjustment = interchange_adjustment(scenario, self._resolve)
        first = compute_pnl(frame, state.table, cfg, interchange_adjustment=adjustment)

        # 2 y 5. Traslados: la tasa blended por grupo, medida sobre comercios blended, y la de
        # recompensas por producto. El interchange no depende de ninguna de las dos, así que
        # ambas salen de la misma primera tarificación.
        blended = pl.col("pricing_model").cast(pl.Utf8) == "blended"
        repriced_cfg, prices = passed_through(
            cfg,
            effective_interchange_change(base, first.transactions, "mcc_group", blended),
            effective_interchange_change(base, first.transactions, "product"),
        )
        rows = compute_pnl(
            frame, state.table, repriced_cfg, interchange_adjustment=adjustment
        ).transactions

        # 3. Abandono: solo el adicional que causan los precios del escenario.
        burden = merchant_burden(rows, state.profile, pl.col(VOLUME_WEIGHT))
        probability = abandonment_probability(with_burden(state.profile, burden), cfg, state.curve)
        acceptance = (
            state.base_abandonment.rename({"p_abandonment": "p_base"})
            .join(
                probability.select("merchant_id", pl.col("p_abandonment").alias("p_scenario")),
                on="merchant_id",
                how="left",
                maintain_order="left",
            )
            .join(
                burden.select("merchant_id", "gdv_cop"),
                on="merchant_id",
                how="left",
                maintain_order="left",
            )
            .with_columns((pl.col("p_scenario") - pl.col("p_base")).alias("delta_p"))
            .with_columns((pl.col("delta_p") * pl.col("gdv_cop")).alias("gdv_lost_cop"))
        )

        # 4. Redistribución con signo: lo que un comercio pierde, o retiene, pasa a sus
        # sustitutos o vuelve de ellos, y una parte sale del riel de tarjetas.
        lost = acceptance.select("merchant_id", "gdv_lost_cop")
        moved = redistribute_volume(lost, state.substitutes, cfg)
        leaked = redistribution_leakage(lost, state.substitutes, cfg)
        receivable = acceptance.filter(pl.col("gdv_cop") > 0).select("merchant_id")
        # Un sustituto sin compras en el escenario no tiene filas donde recibir volumen: ese
        # volumen cuenta como fuga, así la cuenta sigue cerrando.
        unplaced = float(
            moved.join(receivable, on="merchant_id", how="anti")["gdv_moved_cop"].sum()
        )
        acceptance = (
            acceptance.join(
                moved.join(receivable, on="merchant_id", how="semi"),
                on="merchant_id",
                how="left",
                maintain_order="left",
            )
            .with_columns(pl.col("gdv_moved_cop").fill_null(0.0).alias("gdv_moved_in_cop"))
            .drop("gdv_moved_cop")
            .with_columns(
                pl.when(pl.col("gdv_cop") > 0)
                .then(1.0 - pl.col("delta_p") + pl.col("gdv_moved_in_cop") / pl.col("gdv_cop"))
                .otherwise(1.0)
                .alias("merchant_weight")
            )
        )

        # 6. Gasto: la nueva tasa de recompensas de cada producto, por segmento de gasto.
        spend = pl.concat(
            [
                segment_response(cfg, row["scenario"] - row["base"]).select(
                    pl.lit(row["key"]).alias("_product"),
                    pl.col("spend_segment").cast(pl.Utf8),
                    pl.col("spend_multiplier").alias("spend_weight"),
                )
                for row in prices.filter(pl.col("price") == "rewards_rate").iter_rows(named=True)
            ]
        )

        # 7. Pesos y P&L: cada fila conserva sus precios nuevos y carga el producto de sus pesos.
        weighted = (
            rows.join(
                state.cardholders.select("cardholder_id", pl.col("spend_segment").cast(pl.Utf8)),
                on="cardholder_id",
                how="left",
                validate="m:1",
                maintain_order="left",
            )
            .with_columns(pl.col("product").cast(pl.Utf8).alias("_product"))
            .join(
                spend,
                on=["_product", "spend_segment"],
                how="left",
                validate="m:1",
                maintain_order="left",
            )
            .join(
                acceptance.select("merchant_id", "merchant_weight"),
                on="merchant_id",
                how="left",
                validate="m:1",
                maintain_order="left",
            )
            .with_columns(
                pl.col("spend_weight").fill_null(1.0), pl.col("merchant_weight").fill_null(1.0)
            )
            .with_columns(
                (pl.col(VOLUME_WEIGHT) * pl.col("merchant_weight") * pl.col("spend_weight")).alias(
                    WEIGHT
                )
            )
            .drop("_product", "spend_segment")
        )

        bridge = revenue_bridge(
            base.select("cross_border", *BRIDGE_KEYS, "amount_cop", "network_revenue_cop"),
            weighted.select(
                "cross_border",
                *BRIDGE_KEYS,
                (_AMOUNT * pl.col(WEIGHT)).alias("amount_cop"),
                (pl.col("network_revenue_cop") * pl.col(WEIGHT)).alias("network_revenue_cop"),
            ),
            BRIDGE_KEYS,
        )
        return SimulationResult(
            scenario=scenario,
            baseline=base.with_columns(pl.lit(1.0).alias(WEIGHT)),
            priced=weighted,
            acceptance=acceptance,
            prices=prices,
            lost_cop=float(acceptance["gdv_lost_cop"].sum()),
            placed_cop=float(acceptance["gdv_moved_in_cop"].sum()),
            leaked_cop=leaked + unplaced,
            bridge=bridge,
            seconds=time.perf_counter() - start,
        )
