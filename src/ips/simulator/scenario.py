"""Scenarios: what changes in the network, written as YAML and validated before anything runs.

A scenario is a name, one sentence and a list of levers. There are four kinds of lever, and
each knows how to express itself in the pricing chain:

* ``interchange_cap``: no transaction of the given products pays more interchange than a share
  of its amount, fixed fee included — the regulatory cap of spec §2.6;
* ``interchange_discount``: interchange falls, by an absolute rate or by a share, on the
  transactions a filter selects — a channel, tokenization, the largest merchants;
* ``volume_substitution``: a share of the selected purchases leaves the card rails;
* ``product_migration``: a share of one product's volume moves to another product.

Levers apply in the order the file lists them.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Annotated, Literal

import polars as pl
import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from ips.utils.config import Channel, ProjectConfig, Rate, Share, resolve_path

VOLUME_WEIGHT = "volume_weight"
_AMOUNT = pl.col("amount_cop").cast(pl.Float64)


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class TransactionFilter(_Frozen):
    """Which transactions a lever touches; every condition given has to hold."""

    products: tuple[str, ...] | None = None
    channels: tuple[Channel, ...] | None = None
    tokenized: bool | None = None
    max_amount_cop: float | None = Field(default=None, gt=0.0)
    top_merchants_by_gdv: int | None = Field(default=None, gt=0)


class InterchangeCap(_Frozen):
    """No transaction of these products pays more interchange than ``max_rate`` of its amount."""

    kind: Literal["interchange_cap"]
    products: tuple[str, ...] = Field(min_length=1)
    max_rate: Rate


class InterchangeDiscount(_Frozen):
    """Interchange falls on the selected transactions, by an absolute rate or by a share."""

    kind: Literal["interchange_discount"]
    where: TransactionFilter
    rate_cut: Rate | None = None
    share: Share | None = None

    @model_validator(mode="after")
    def _check_one_cut(self) -> InterchangeDiscount:
        if (self.rate_cut is None) == (self.share is None):
            raise ValueError("interchange_discount needs exactly one of rate_cut or share")
        return self


class VolumeSubstitution(_Frozen):
    """A share of the selected purchases leaves the card rails for another way to pay."""

    kind: Literal["volume_substitution"]
    where: TransactionFilter
    share: Share


class ProductMigration(_Frozen):
    """A share of one product's volume moves to another product."""

    kind: Literal["product_migration"]
    from_product: str
    to_product: str
    share: Share

    @model_validator(mode="after")
    def _check_products(self) -> ProductMigration:
        if self.from_product == self.to_product:
            raise ValueError("product_migration needs two different products")
        return self


Lever = Annotated[
    InterchangeCap | InterchangeDiscount | VolumeSubstitution | ProductMigration,
    Field(discriminator="kind"),
]
Resolver = Callable[[TransactionFilter], Sequence[int] | None]


class Scenario(_Frozen):
    """A named change to the network: one sentence and the levers that implement it."""

    name: str = Field(pattern=r"^[a-z0-9_]+$")
    description: str = Field(min_length=1)
    levers: tuple[Lever, ...] = ()


def check_scenario(scenario: Scenario, cfg: ProjectConfig) -> Scenario:
    """Fail on products the network does not issue, before any pricing runs."""
    named: set[str] = set()
    for lever in scenario.levers:
        if isinstance(lever, InterchangeCap):
            named.update(lever.products)
        elif isinstance(lever, ProductMigration):
            named.update((lever.from_product, lever.to_product))
        elif lever.where.products is not None:
            named.update(lever.where.products)
    unknown = sorted(named - set(cfg.products))
    if unknown:
        raise ValueError(f"Scenario {scenario.name!r} references unknown products: {unknown}")
    return scenario


def load_scenario(path: Path, cfg: ProjectConfig) -> Scenario:
    """Read and validate one scenario file."""
    with Path(path).open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    return check_scenario(Scenario.model_validate(data), cfg)


def scenario_paths(cfg: ProjectConfig) -> list[Path]:
    """Every scenario file in ``simulator.scenarios_dir``, in name order."""
    directory = resolve_path(cfg.simulator.scenarios_dir)
    paths = sorted(directory.glob("*.yaml"))
    if not paths:
        raise FileNotFoundError(f"No scenario files in {directory}")
    return paths


def matches(where: TransactionFilter, merchants: Sequence[int] | None = None) -> pl.Expr:
    """True on the transactions a filter selects."""
    condition = pl.lit(True)
    if where.products is not None:
        condition = condition & pl.col("product").cast(pl.Utf8).is_in(list(where.products))
    if where.channels is not None:
        condition = condition & pl.col("channel").cast(pl.Utf8).is_in(list(where.channels))
    if where.tokenized is not None:
        condition = condition & (pl.col("tokenized") == where.tokenized)
    if where.max_amount_cop is not None:
        # "Bajo 50.000 COP": estrictamente por debajo del umbral, como lo dice la spec.
        condition = condition & _AMOUNT.lt(where.max_amount_cop)
    if where.top_merchants_by_gdv is not None:
        if merchants is None:
            raise ValueError("top_merchants_by_gdv needs the merchants resolved on the base year")
        condition = condition & pl.col("merchant_id").is_in(list(merchants))
    return condition


def interchange_adjustment(scenario: Scenario, resolve: Resolver) -> pl.Expr | None:
    """The new ``interchange_cop`` the scenario's price levers produce, or None if none apply.

    Devolver None cuando ninguna palanca toca el interchange no es un atajo: deja que el motor
    tarifique exactamente igual que la base.
    """
    expression = pl.col("interchange_cop")
    touched = False
    for lever in scenario.levers:
        if isinstance(lever, InterchangeCap):
            capped = pl.min_horizontal(expression, _AMOUNT * lever.max_rate)
            inside = pl.col("product").cast(pl.Utf8).is_in(list(lever.products))
            expression = pl.when(inside).then(capped).otherwise(expression)
            touched = True
        elif isinstance(lever, InterchangeDiscount):
            if lever.rate_cut is not None:
                reduced = (expression - _AMOUNT * lever.rate_cut).clip(lower_bound=0.0)
            else:
                reduced = expression * (1.0 - lever.share)
            selected = matches(lever.where, resolve(lever.where))
            expression = pl.when(selected).then(reduced).otherwise(expression)
            touched = True
    return expression if touched else None


def apply_volume_levers(
    transactions: pl.DataFrame, scenario: Scenario, resolve: Resolver
) -> pl.DataFrame:
    """Add ``volume_weight``: migrations split rows between products, substitutions shrink them.

    Una migración no borra filas: parte cada compra del producto de origen en dos, una que se
    queda y una copia con el producto de destino, cada una con su peso. Así el volumen que se
    mueve es exactamente la participación del escenario, sin sortear tarjetas.
    """
    frame = transactions.with_columns(pl.lit(1.0).alias(VOLUME_WEIGHT))
    for lever in scenario.levers:
        if isinstance(lever, ProductMigration):
            source = pl.col("product").cast(pl.Utf8) == lever.from_product
            moving = frame.filter(source).with_columns(
                pl.lit(lever.to_product).cast(frame.schema["product"]).alias("product"),
                (pl.col(VOLUME_WEIGHT) * lever.share).alias(VOLUME_WEIGHT),
            )
            staying = frame.with_columns(
                pl.when(source)
                .then(pl.col(VOLUME_WEIGHT) * (1.0 - lever.share))
                .otherwise(pl.col(VOLUME_WEIGHT))
                .alias(VOLUME_WEIGHT)
            )
            frame = pl.concat([staying, moving])
        elif isinstance(lever, VolumeSubstitution):
            selected = matches(lever.where, resolve(lever.where))
            frame = frame.with_columns(
                pl.when(selected)
                .then(pl.col(VOLUME_WEIGHT) * (1.0 - lever.share))
                .otherwise(pl.col(VOLUME_WEIGHT))
                .alias(VOLUME_WEIGHT)
            )
    return frame
