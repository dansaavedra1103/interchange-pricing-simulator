"""Simple synthetic data generator for the vertical slice (Feature 0).

Produces issuers, acquirers, merchants, cardholders and transactions from
``config/base.yaml``. Deliberately minimal: uniform dates, one interchange table
(product x MCC), no channel, no cross-border and no fraud. Every modelling assumption
is recorded in ``docs/assumptions.md``.

Transactions carry facts only (who, where, when, how much). Fees are derived later by
``ips.economics.simple_pnl.compute_pnl`` so the same facts can be repriced under any
interchange table.

Usage::

    python -m ips.data_gen.simple_generator --config config/base.yaml
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl

from ips.utils.config import ProjectConfig, default_config_path, load_config

ISSUER_SCHEMA = pl.Schema(
    {
        "issuer_id": pl.Utf8,
        "issuer_type": pl.Utf8,
        "bank_group": pl.Utf8,
        "market_share": pl.Float64,
    }
)
ACQUIRER_SCHEMA = pl.Schema(
    {
        "acquirer_id": pl.Utf8,
        "acquirer_type": pl.Utf8,
        "bank_group": pl.Utf8,
        "market_share": pl.Float64,
    }
)
MERCHANT_SCHEMA = pl.Schema(
    {
        "merchant_id": pl.UInt32,
        "mcc": pl.Utf8,
        "acquirer_id": pl.Utf8,
        "size_weight": pl.Float64,
    }
)
CARDHOLDER_SCHEMA = pl.Schema(
    {
        "cardholder_id": pl.UInt32,
        "issuer_id": pl.Utf8,
        "product": pl.Utf8,
        "activity_weight": pl.Float64,
    }
)
TRANSACTION_SCHEMA = pl.Schema(
    {
        "txn_id": pl.UInt32,
        "txn_date": pl.Date,
        "cardholder_id": pl.UInt32,
        "merchant_id": pl.UInt32,
        "issuer_id": pl.Utf8,
        "acquirer_id": pl.Utf8,
        "product": pl.Utf8,
        "mcc": pl.Utf8,
        "amount_cop": pl.Int64,
        "on_us": pl.Boolean,
    }
)


@dataclass(frozen=True)
class SliceData:
    """All tables produced by the simple generator."""

    issuers: pl.DataFrame
    acquirers: pl.DataFrame
    merchants: pl.DataFrame
    cardholders: pl.DataFrame
    transactions: pl.DataFrame

    def tables(self) -> dict[str, pl.DataFrame]:
        """Tables keyed by their parquet file stem."""
        return {
            "issuers": self.issuers,
            "acquirers": self.acquirers,
            "merchants": self.merchants,
            "cardholders": self.cardholders,
            "transactions": self.transactions,
        }


def _probabilities(weights: Sequence[float] | np.ndarray) -> np.ndarray:
    array = np.asarray(weights, dtype=np.float64)
    return array / array.sum()


def _pick(name: str, values: Sequence[str], index: np.ndarray) -> pl.Series:
    return pl.Series(name, values, dtype=pl.Utf8).gather(index)


def generate_issuers(cfg: ProjectConfig) -> pl.DataFrame:
    """Issuer dimension, taken directly from the configuration."""
    return pl.from_dicts([i.model_dump() for i in cfg.issuers], schema=ISSUER_SCHEMA)


def generate_acquirers(cfg: ProjectConfig) -> pl.DataFrame:
    """Acquirer dimension, taken directly from the configuration."""
    return pl.from_dicts([a.model_dump() for a in cfg.acquirers], schema=ACQUIRER_SCHEMA)


def generate_merchants(cfg: ProjectConfig, rng: np.random.Generator) -> pl.DataFrame:
    """Merchants with an MCC, an acquirer and a lognormal volume weight.

    The acquirer is drawn from acquirer market shares alone, without looking at anything
    on the cardholder side, which makes acquirer and issuer independent by construction.
    """
    n = cfg.sizes.n_merchants
    mcc_idx = rng.choice(
        len(cfg.mccs), size=n, p=_probabilities([m.merchant_share for m in cfg.mccs])
    )
    # El comercio elige adquirente sin saber con qué banco tienen tarjeta sus clientes
    # (spec §1.1): el sorteo usa solo las participaciones de mercado de los adquirentes.
    acquirer_idx = rng.choice(
        len(cfg.acquirers), size=n, p=_probabilities([a.market_share for a in cfg.acquirers])
    )
    # Tamaño log-normal: pocos comercios grandes concentran buena parte del volumen.
    size_weight = rng.lognormal(mean=0.0, sigma=cfg.distributions.merchant_size_sigma, size=n)
    return pl.DataFrame(
        {
            "merchant_id": np.arange(n, dtype=np.uint32),
            "mcc": _pick("mcc", cfg.mcc_codes, mcc_idx),
            "acquirer_id": _pick(
                "acquirer_id", [a.acquirer_id for a in cfg.acquirers], acquirer_idx
            ),
            "size_weight": size_weight,
        },
        schema=MERCHANT_SCHEMA,
    )


def generate_cardholders(cfg: ProjectConfig, rng: np.random.Generator) -> pl.DataFrame:
    """Cardholders with an issuer, a card product and a lognormal activity weight."""
    n = cfg.sizes.n_cardholders
    issuer_idx = rng.choice(
        len(cfg.issuers), size=n, p=_probabilities([i.market_share for i in cfg.issuers])
    )
    issuer_types = np.array([i.issuer_type for i in cfg.issuers])[issuer_idx]

    # El mix de productos depende del tipo de emisor: los bancos emiten más premium y las
    # fintech más débito (spec §2.3). Primero se sortea el emisor y luego el producto.
    products = list(cfg.products)
    product_idx = np.empty(n, dtype=np.int64)
    for issuer_type, mix in cfg.product_mix.items():
        mask = issuer_types == issuer_type
        product_idx[mask] = rng.choice(
            len(products), size=int(mask.sum()), p=_probabilities([mix[p] for p in products])
        )

    activity_weight = rng.lognormal(
        mean=0.0, sigma=cfg.distributions.cardholder_activity_sigma, size=n
    )
    return pl.DataFrame(
        {
            "cardholder_id": np.arange(n, dtype=np.uint32),
            "issuer_id": _pick("issuer_id", [i.issuer_id for i in cfg.issuers], issuer_idx),
            "product": _pick("product", products, product_idx),
            "activity_weight": activity_weight,
        },
        schema=CARDHOLDER_SCHEMA,
    )


def generate_transactions(
    cfg: ProjectConfig,
    rng: np.random.Generator,
    merchants: pl.DataFrame,
    cardholders: pl.DataFrame,
    issuers: pl.DataFrame,
    acquirers: pl.DataFrame,
) -> pl.DataFrame:
    """Transactions as independent (cardholder, merchant) draws with lognormal amounts.

    Issuer and product are denormalised from the cardholder, MCC and acquirer from the
    merchant, and ``on_us`` flags pairs whose issuer and acquirer share a bank group.
    """
    n = cfg.sizes.n_transactions
    # Dos sorteos independientes por transacción: quién paga (tarjetahabiente, ponderado por
    # actividad) y dónde paga (comercio, ponderado por tamaño). Como el adquirente viene del
    # comercio y el emisor del tarjetahabiente, adquirente y emisor quedan independientes.
    cardholder_idx = rng.choice(
        cardholders.height, size=n, p=_probabilities(cardholders["activity_weight"].to_numpy())
    )
    merchant_idx = rng.choice(
        merchants.height, size=n, p=_probabilities(merchants["size_weight"].to_numpy())
    )
    # Fecha uniforme en el rango: sin estacionalidad en la Feature 0.
    day_offset = rng.integers(0, cfg.dates.n_days, size=n)
    z = rng.standard_normal(n)

    frame = pl.concat(
        [
            cardholders.select(
                pl.col("cardholder_id", "issuer_id", "product").gather(pl.Series(cardholder_idx))
            ),
            merchants.select(
                pl.col("merchant_id", "mcc", "acquirer_id").gather(pl.Series(merchant_idx))
            ),
            pl.DataFrame({"txn_date": np.datetime64(cfg.dates.start, "D") + day_offset, "_z": z}),
        ],
        how="horizontal",
        strict=True,
    )

    median_ticket = {m.mcc: m.median_ticket_cop for m in cfg.mccs}
    ticket_sigma = {m.mcc: m.ticket_sigma for m in cfg.mccs}
    # Monto log-normal por MCC: la mediana es el ticket típico del sector y la cola larga a la
    # derecha recoge las compras grandes. Se redondea a pesos y se aplica un piso.
    amount = (
        pl.col("mcc").replace_strict(median_ticket, return_dtype=pl.Float64)
        * (pl.col("mcc").replace_strict(ticket_sigma, return_dtype=pl.Float64) * pl.col("_z")).exp()
    )

    issuer_group = pl.col("issuer_id").replace_strict(
        dict(issuers.select("issuer_id", "bank_group").iter_rows()), return_dtype=pl.Utf8
    )
    acquirer_group = pl.col("acquirer_id").replace_strict(
        dict(acquirers.select("acquirer_id", "bank_group").iter_rows()), return_dtype=pl.Utf8
    )

    return (
        frame.with_columns(
            amount.round(0)
            .clip(lower_bound=cfg.distributions.min_amount_cop)
            .cast(pl.Int64)
            .alias("amount_cop"),
            # On-us: emisor y adquirente pertenecen al mismo grupo financiero. Fintech y
            # agregadores no tienen grupo (null), así que nunca son on-us.
            (issuer_group == acquirer_group).fill_null(False).alias("on_us"),
        )
        .sort("txn_date", maintain_order=True)
        .with_row_index("txn_id")
        .select(TRANSACTION_SCHEMA.names())
    )


def generate_all(cfg: ProjectConfig) -> SliceData:
    """Generate every table of the vertical slice, reproducibly from ``cfg.seed``."""
    # One independent RNG stream per entity: changing n_transactions leaves merchants and
    # cardholders untouched.
    merchant_seed, cardholder_seed, transaction_seed = np.random.SeedSequence(cfg.seed).spawn(3)
    issuers = generate_issuers(cfg)
    acquirers = generate_acquirers(cfg)
    merchants = generate_merchants(cfg, np.random.default_rng(merchant_seed))
    cardholders = generate_cardholders(cfg, np.random.default_rng(cardholder_seed))
    transactions = generate_transactions(
        cfg, np.random.default_rng(transaction_seed), merchants, cardholders, issuers, acquirers
    )
    return SliceData(issuers, acquirers, merchants, cardholders, transactions)


def write_parquet(data: SliceData, out_dir: Path) -> list[Path]:
    """Write every table to ``out_dir/<table>.parquet`` and return the written paths."""
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, frame in data.tables().items():
        path = out_dir / f"{name}.parquet"
        frame.write_parquet(path)
        paths.append(path)
    return paths


def main(argv: Sequence[str] | None = None) -> None:
    """CLI entry point: generate the dataset and write it as parquet."""
    parser = argparse.ArgumentParser(description="Generate the vertical-slice synthetic dataset.")
    parser.add_argument("--config", type=Path, default=default_config_path())
    parser.add_argument(
        "--out", type=Path, default=None, help="Output directory (default: output.raw_dir)."
    )
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    out_dir = args.out if args.out is not None else cfg.resolve_path(cfg.output.raw_dir)
    data = generate_all(cfg)
    for path in write_parquet(data, out_dir):
        print(f"{path}  rows={data.tables()[path.stem].height:,}")


if __name__ == "__main__":
    main()
