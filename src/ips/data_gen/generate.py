"""Command-line entry point: generate the full synthetic dataset into ``data/raw/``.

Usage::

    python -m ips.data_gen.generate --config config/base.yaml
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from collections.abc import Sequence
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path

import numpy as np
import polars as pl

from ips.data_gen.affinity import AffinitySampler, draw_favorites
from ips.data_gen.entities import table_schemas
from ips.data_gen.fraud import add_fraud_flags
from ips.data_gen.population import (
    build_acquirers,
    build_issuers,
    draw_cardholders,
    draw_merchants,
)
from ips.data_gen.transactions import generate_transactions
from ips.economics.params import reference_tables
from ips.utils.config import ProjectConfig, default_config_path, load_config, resolve_path
from ips.utils.logging import get_logger

STREAMS = ("cardholders", "merchants", "favorites", "transactions", "fraud")


@dataclass(frozen=True)
class GeneratedData:
    """Observable tables and latent ground truth of one generation run."""

    tables: dict[str, pl.DataFrame]
    latent: dict[str, pl.DataFrame]


def generate(cfg: ProjectConfig) -> GeneratedData:
    """Generate every table reproducibly from ``cfg.seed``."""
    # One independent RNG stream per stage: resizing one stage leaves the others untouched.
    seeds = np.random.SeedSequence(cfg.seed).spawn(len(STREAMS))
    rngs = {name: np.random.default_rng(seed) for name, seed in zip(STREAMS, seeds, strict=True)}

    cardholders, card_latent = draw_cardholders(rngs["cardholders"], cfg)
    merchants, merchant_latent = draw_merchants(rngs["merchants"], cfg)
    n_groups = len(cfg.group_names)
    sampler = AffinitySampler(
        merchant_latent.pools,
        n_cities=len(cfg.geography.city_weights()),
        n_groups=n_groups,
        n_segments=len(cfg.lifestyle.segments),
        boost=cfg.affinity.clientele_boost,
    )
    favorites = draw_favorites(
        rngs["favorites"],
        sampler,
        card_latent.city_idx,
        card_latent.segment_idx,
        n_groups,
        len(cfg.affinity.favorite_weights),
    )
    transactions = generate_transactions(
        rngs["transactions"],
        cfg,
        cardholders,
        card_latent,
        merchants,
        merchant_latent,
        sampler,
        favorites,
    )
    transactions = add_fraud_flags(rngs["fraud"], transactions, cfg)
    schema = table_schemas(cfg)["transactions"]
    tables = {
        "issuers": build_issuers(cfg),
        "acquirers": build_acquirers(cfg),
        "merchants": merchants,
        "cardholders": cardholders,
        "transactions": transactions.select(list(schema)).cast(dict(schema)),
        **reference_tables(cfg),
    }
    latent = {
        "cardholders_latent": card_latent.to_frame(cfg),
        "merchants_latent": merchant_latent.to_frame(cfg),
    }
    return GeneratedData(tables, latent)


def config_hash(cfg: ProjectConfig) -> str:
    """Short fingerprint of the full configuration, for the manifest."""
    return hashlib.sha256(cfg.model_dump_json().encode("utf-8")).hexdigest()[:16]


def _package_version() -> str:
    try:
        return metadata.version("ips")
    except metadata.PackageNotFoundError:
        return "unknown"


def write_outputs(
    data: GeneratedData, out_dir: Path, cfg: ProjectConfig, elapsed_seconds: float
) -> Path:
    """Write every table as parquet plus ``manifest.json``; return the manifest path."""
    latent_dir = out_dir / "latent"
    latent_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in data.tables.items():
        frame.write_parquet(out_dir / f"{name}.parquet")
    for name, frame in data.latent.items():
        frame.write_parquet(latent_dir / f"{name}.parquet")
    manifest = {
        "seed": cfg.seed,
        "config_hash": config_hash(cfg),
        "date_range": [cfg.dates.start.isoformat(), cfg.dates.end.isoformat()],
        "rows": {name: frame.height for name, frame in data.tables.items()},
        "latent_rows": {name: frame.height for name, frame in data.latent.items()},
        "elapsed_seconds": round(elapsed_seconds, 1),
        "versions": {
            "ips": _package_version(),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "polars": pl.__version__,
        },
    }
    path = out_dir / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return path


def main(argv: Sequence[str] | None = None) -> None:
    """CLI: generate the dataset and write it to ``output.raw_dir`` (or ``--out``)."""
    parser = argparse.ArgumentParser(description="Generate the synthetic payments dataset.")
    parser.add_argument("--config", type=Path, default=default_config_path())
    parser.add_argument("--out", type=Path, default=None, help="Output directory.")
    parser.add_argument(
        "--n-transactions", type=int, default=None, help="Override sizes.n_transactions."
    )
    parser.add_argument(
        "--sample", action="store_true", help="Use sample_sizes from the config (tests, CI)."
    )
    args = parser.parse_args(argv)
    logger = get_logger("ips.data_gen.generate")

    cfg = load_config(args.config)
    if args.sample:
        cfg = cfg.model_copy(update={"sizes": cfg.sample_sizes})
    if args.n_transactions is not None:
        sizes = cfg.sizes.model_copy(update={"n_transactions": args.n_transactions})
        cfg = cfg.model_copy(update={"sizes": sizes})
    out_dir = args.out if args.out is not None else resolve_path(cfg.output.raw_dir)

    start = time.perf_counter()
    data = generate(cfg)
    elapsed = time.perf_counter() - start
    manifest = write_outputs(data, out_dir, cfg, elapsed)
    for name, frame in data.tables.items():
        logger.info("%-18s %12s rows", name, f"{frame.height:,}")
    logger.info("generated in %.1f s; manifest at %s", elapsed, manifest)


if __name__ == "__main__":
    main()
