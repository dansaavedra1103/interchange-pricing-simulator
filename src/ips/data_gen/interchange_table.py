"""Network interchange table: build the full grid and price transactions with it.

The table is product x MCC group x channel x region (192 cells for 4 x 12 x 2 x 2), built
from ``config/interchange_table.yaml``: a card-present domestic base grid, channel and region
add-ons, a reduced tier for small domestic tickets, and explicit overrides.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

import polars as pl

from ips.utils.config import CHANNELS, REGIONS, Adjustment, ProjectConfig

TABLE_KEYS = ("product", "mcc_group", "channel", "region")
TABLE_SCHEMA = pl.Schema(
    {
        "product": pl.Utf8,
        "mcc_group": pl.Utf8,
        "channel": pl.Utf8,
        "region": pl.Utf8,
        "rate": pl.Float64,
        "fixed_cop": pl.Float64,
        "small_ticket_max_cop": pl.Float64,
        "small_ticket_rate": pl.Float64,
        "small_ticket_fixed_cop": pl.Float64,
        "valid_from": pl.Date,
        "valid_to": pl.Date,
    }
)
_REQUIRED = ("product", "mcc_group", "channel", "cross_border", "amount_cop")
_NO_ADJUSTMENT = Adjustment()


def build_interchange_table(cfg: ProjectConfig) -> pl.DataFrame:
    """Expand the configured table into one row per (product, group, channel, region)."""
    spec = cfg.interchange_table
    tier = spec.small_ticket
    overrides = {(o.product, o.mcc_group, o.channel, o.region): o for o in spec.overrides}
    rows = []
    for product, cells in spec.base.items():
        for group, base in cells.items():
            for channel in CHANNELS:
                for region in REGIONS:
                    # Recargos: el no presente y el cross-border cargan más riesgo (spec §1.4).
                    channel_add = spec.adjustments.channel.get(channel, _NO_ADJUSTMENT)
                    region_add = spec.adjustments.region.get(region, _NO_ADJUSTMENT)
                    rate = base.rate + channel_add.rate_add + region_add.rate_add
                    fixed = base.fixed_cop + channel_add.fixed_add_cop + region_add.fixed_add_cop
                    override = overrides.get((product, group, channel, region))
                    if override is not None:
                        rate, fixed = override.rate, override.fixed_cop
                    if rate < 0 or fixed < 0:
                        raise ValueError(
                            f"Negative fee in cell {(product, group, channel, region)}"
                        )
                    # Tramo de micropagos: solo doméstico, para los productos y grupos configurados.
                    small = (
                        tier is not None
                        and region == "domestic"
                        and product in tier.products
                        and group in tier.groups
                    )
                    rows.append(
                        {
                            "product": product,
                            "mcc_group": group,
                            "channel": channel,
                            "region": region,
                            "rate": rate,
                            "fixed_cop": fixed,
                            "small_ticket_max_cop": tier.max_amount_cop if small else None,
                            "small_ticket_rate": tier.rate if small else None,
                            "small_ticket_fixed_cop": tier.fixed_cop if small else None,
                            "valid_from": spec.valid_from,
                            "valid_to": spec.valid_to,
                        }
                    )
    return pl.DataFrame(rows, schema=TABLE_SCHEMA)


def _fee(amount: pl.Expr) -> pl.Expr:
    small = pl.col("small_ticket_max_cop").is_not_null() & (
        amount <= pl.col("small_ticket_max_cop")
    )
    return (
        pl.when(small)
        .then(amount * pl.col("small_ticket_rate") + pl.col("small_ticket_fixed_cop"))
        .otherwise(amount * pl.col("rate") + pl.col("fixed_cop"))
    )


def apply_interchange(transactions: pl.DataFrame, table: pl.DataFrame) -> pl.DataFrame:
    """Add ``interchange_cop`` to every transaction.

    Raises:
        ValueError: If required columns are missing, the table has duplicate cells, or a
            transaction has no cell (which would silently leak P&L downstream).
    """
    missing_columns = [c for c in _REQUIRED if c not in transactions.columns]
    if missing_columns:
        raise ValueError(f"transactions is missing required columns: {missing_columns}")
    keys = list(TABLE_KEYS)
    duplicated = table.filter(table.select(keys).is_duplicated())
    if duplicated.height:
        cells = sorted(set(duplicated.select(keys).iter_rows()))
        raise ValueError(f"Interchange table has duplicate cells: {cells}")

    cell_columns = [c for c in TABLE_SCHEMA if c not in TABLE_KEYS and not c.startswith("valid")]
    keyed = transactions.with_columns(
        pl.when(pl.col("cross_border"))
        .then(pl.lit("cross_border"))
        .otherwise(pl.lit("domestic"))
        .alias("region"),
        *(pl.col(k).cast(pl.Utf8).alias(f"_{k}") for k in ("product", "mcc_group", "channel")),
    )
    priced = keyed.join(
        table.select(*keys, *cell_columns).rename({k: f"_{k}" for k in keys[:3]}),
        on=["_product", "_mcc_group", "_channel", "region"],
        how="left",
        validate="m:1",
        maintain_order="left",
    )
    unpriced = priced.filter(pl.col("rate").is_null())
    if unpriced.height:
        cells = sorted(
            set(unpriced.select("_product", "_mcc_group", "_channel", "region").iter_rows())
        )
        raise ValueError(
            f"{unpriced.height} transactions have no interchange cell; missing {cells}"
        )
    return priced.with_columns(
        _fee(pl.col("amount_cop").cast(pl.Float64)).alias("interchange_cop")
    ).drop("region", "_product", "_mcc_group", "_channel", *cell_columns)


@dataclass(frozen=True)
class InterchangeTable:
    """The expanded table with scalar and vectorised pricing."""

    frame: pl.DataFrame

    @classmethod
    def from_config(cls, cfg: ProjectConfig) -> InterchangeTable:
        """Build the table from the project configuration."""
        return cls(build_interchange_table(cfg))

    @cached_property
    def _cells(self) -> dict[tuple[str, str, str, str], dict[str, float | None]]:
        return {tuple(row[k] for k in TABLE_KEYS): row for row in self.frame.iter_rows(named=True)}

    def lookup(
        self, product: str, mcc_group: str, channel: str, region: str, amount: float
    ) -> float:
        """Interchange fee in COP for one transaction (the amount selects the small-ticket tier)."""
        try:
            cell = self._cells[(product, mcc_group, channel, region)]
        except KeyError:
            raise KeyError(
                f"No interchange cell for {(product, mcc_group, channel, region)}"
            ) from None
        tier_max = cell["small_ticket_max_cop"]
        if tier_max is not None and amount <= tier_max:
            return amount * cell["small_ticket_rate"] + cell["small_ticket_fixed_cop"]
        return amount * cell["rate"] + cell["fixed_cop"]

    def apply(self, transactions: pl.DataFrame) -> pl.DataFrame:
        """Vectorised pricing: ``apply_interchange(transactions, self.frame)``."""
        return apply_interchange(transactions, self.frame)
