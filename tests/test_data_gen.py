"""Feature 1 generator: integrity, determinism, calibration ranges and community structure."""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from polars.testing import assert_frame_equal

from ips.data_gen.diagnostics import (
    bipartite_modularity,
    edge_list,
    leiden_partition,
    normalized_mutual_info,
    shuffled_modularity,
    top_share,
)
from ips.data_gen.entities import ENTITY_MODELS, table_schemas
from ips.data_gen.fraud import fraud_probability
from ips.data_gen.generate import GeneratedData, generate, main
from ips.utils.config import ProjectConfig

# Con resolución 1, Leiden solo separa ciudades; con 2 aparece la clientela dentro de ellas.
LEIDEN_RESOLUTION = 2.0


# ---------------------------------------------------------------------------
# Integridad y determinismo
# ---------------------------------------------------------------------------


def test_tables_match_entity_contracts(cfg: ProjectConfig, data: GeneratedData) -> None:
    schemas = table_schemas(cfg)
    for name, model in ENTITY_MODELS.items():
        frame = data.tables[name]
        assert frame.schema == schemas[name], name
        assert frame.columns == list(model.model_fields), name
        for row in frame.head(50).iter_rows(named=True):
            model.model_validate(row)


def test_sizes_keys_and_consistency(cfg: ProjectConfig, data: GeneratedData) -> None:
    tables, tx = data.tables, data.tables["transactions"]
    merchants, cardholders = tables["merchants"], tables["cardholders"]
    assert tx.height == cfg.sizes.n_transactions
    assert cardholders.height == cfg.sizes.n_cardholders
    assert merchants.height == cfg.sizes.n_merchants + cfg.sizes.n_foreign_merchants
    assert tx.select((pl.col("txn_id") == pl.int_range(pl.len())).all()).item()
    assert tx["txn_date"].is_sorted()
    assert cfg.dates.start <= tx["txn_date"].min() and tx["txn_date"].max() <= cfg.dates.end
    assert tx.null_count().sum_horizontal().item() == 0
    assert tx["amount_cop"].min() >= cfg.amounts.min_amount_cop

    # Los atributos desnormalizados coinciden con las tablas de entidades.
    joined = tx.join(
        merchants.select("merchant_id", "mcc", "acquirer_id", "country", mchannel="channel"),
        on="merchant_id",
        suffix="_m",
    ).join(
        cardholders.select("cardholder_id", "issuer_id", "product"), on="cardholder_id", suffix="_c"
    )
    assert joined.height == tx.height
    for column, other in [
        ("mcc", "mcc_m"),
        ("acquirer_id", "acquirer_id_m"),
        ("issuer_id", "issuer_id_c"),
        ("product", "product_c"),
    ]:
        assert (joined[column] == joined[other]).all(), column
    assert (joined["cross_border"] == (joined["country"] != cfg.geography.country)).all()
    # El canal respeta lo que el comercio vende; el token solo existe en no presente.
    assert not joined.filter((pl.col("mchannel") == "cnp") & (pl.col("channel") == "cp")).height
    assert not joined.filter((pl.col("mchannel") == "cp") & (pl.col("channel") == "cnp")).height
    assert not tx.filter((pl.col("channel") == "cp") & pl.col("tokenized")).height


def test_same_seed_same_data(cfg: ProjectConfig, data: GeneratedData) -> None:
    again = generate(cfg)
    for name, frame in {**data.tables, **data.latent}.items():
        assert_frame_equal(frame, {**again.tables, **again.latent}[name], check_exact=True)


def test_different_seed_different_data(cfg: ProjectConfig, data: GeneratedData) -> None:
    other = generate(cfg.model_copy(update={"seed": cfg.seed + 1}))
    assert not other.tables["transactions"].equals(data.tables["transactions"])


def test_resizing_transactions_keeps_entities(cfg: ProjectConfig, data: GeneratedData) -> None:
    # Un stream RNG por etapa: cambiar el número de compras no toca comercios ni tarjetas.
    sizes = cfg.sizes.model_copy(update={"n_transactions": 20_000})
    other = generate(cfg.model_copy(update={"sizes": sizes}))
    for name in ("merchants", "cardholders"):
        assert_frame_equal(other.tables[name], data.tables[name])


# ---------------------------------------------------------------------------
# Rangos de calibración
# ---------------------------------------------------------------------------


def test_merchant_mcc_mix(cfg: ProjectConfig, data: GeneratedData) -> None:
    domestic = data.tables["merchants"].filter(pl.col("country") == cfg.geography.country)
    observed = dict(domestic["mcc"].value_counts(normalize=True).iter_rows())
    for mcc in cfg.mcc_groups.mccs:
        p = mcc.merchant_share
        tolerance = 4 * math.sqrt(p * (1 - p) / domestic.height) + 1e-9
        assert abs(observed.get(mcc.mcc, 0.0) - p) < tolerance, mcc.mcc


def test_card_and_purchase_mix(data: GeneratedData, transactions: pl.DataFrame) -> None:
    cards = data.tables["cardholders"]
    # Superfinanciera 2022: ~74 % de las tarjetas y ~68 % de las compras son débito.
    assert 0.70 <= (cards["product"] == "debit").mean() <= 0.78
    assert 0.64 <= (transactions["product"] == "debit").mean() <= 0.72
    by_type = cards.join(data.tables["issuers"].select("issuer_id", "issuer_type"), on="issuer_id")
    debit_share = by_type.group_by("issuer_type").agg((pl.col("product") == "debit").mean())
    shares = dict(debit_share.iter_rows())
    assert shares["fintech"] > shares["bank"]


def test_average_ticket(transactions: pl.DataFrame) -> None:
    # Superfinanciera 2022 ajustado por inflación: ~131k débito y ~244k crédito (±25 %).
    debit = transactions.filter(pl.col("product") == "debit")["amount_cop"].mean()
    credit = transactions.filter(pl.col("product") != "debit")["amount_cop"].mean()
    assert 131_000 * 0.75 <= debit <= 131_000 * 1.25
    assert 244_000 * 0.75 <= credit <= 244_000 * 1.25


def test_seasonality(cfg: ProjectConfig, transactions: pl.DataFrame) -> None:
    years = dict(transactions.group_by(pl.col("txn_date").dt.year()).len().iter_rows())
    # BanRep: +19 % en número de compras con tarjeta en 2025.
    assert 1.12 <= years[2025] / years[2024] <= 1.26
    months = transactions.group_by(pl.col("txn_date").dt.month().alias("m")).len()
    assert months.sort("len")["m"][-1] == 12
    daily = (
        transactions.group_by("txn_date")
        .len()
        .with_columns(
            pl.col("txn_date").dt.day().is_in(cfg.seasonality.payday_days).alias("payday")
        )
    )
    means = dict(daily.group_by("payday").agg(pl.col("len").mean()).iter_rows())
    assert means[True] > 1.07 * means[False]
    restaurants = transactions.filter(pl.col("mcc_group") == "restaurants")
    by_weekday = dict(restaurants.group_by(pl.col("txn_date").dt.weekday()).len().iter_rows())
    weekend = (by_weekday[5] + by_weekday[6]) / 2
    weekdays = sum(by_weekday[d] for d in (1, 2, 3, 4)) / 4
    assert weekend > 1.3 * weekdays


def test_cross_border(transactions: pl.DataFrame) -> None:
    assert 0.01 <= transactions["cross_border"].mean() <= 0.05
    rates = dict(transactions.group_by("product").agg(pl.col("cross_border").mean()).iter_rows())
    assert rates["credit_premium"] > 5 * rates["debit"]


def test_fraud_structure(cfg: ProjectConfig, transactions: pl.DataFrame) -> None:
    # Se usa el valor esperado (probabilidad x monto): con pocos fraudes en la muestra, el
    # valor realizado es demasiado ruidoso para calibrar.
    frame = transactions.with_columns(
        pl.Series("p", fraud_probability(transactions, cfg)),
        pl.col("amount_cop").cast(pl.Float64).alias("v"),
    ).with_columns((pl.col("p") * pl.col("v")).alias("fv"))

    def rate(df: pl.DataFrame) -> float:
        return df["fv"].sum() / df["v"].sum()

    assert 0.0005 <= rate(frame) <= 0.0015  # spec §1.6
    cnp_share = frame.filter(pl.col("channel") == "cnp")["fv"].sum() / frame["fv"].sum()
    assert 0.75 <= cnp_share <= 0.92  # BCE 2023: ~84 %
    ratio = rate(frame.filter(pl.col("cross_border"))) / rate(frame.filter(~pl.col("cross_border")))
    assert 8 <= ratio <= 20  # BCE 2023: ~14x
    realized = dict(transactions.group_by("channel").agg(pl.col("fraud_flag").mean()).iter_rows())
    assert realized["cnp"] > realized["cp"]
    frauds = transactions.filter(pl.col("fraud_flag"))
    assert 0.6 <= frauds["chargeback_flag"].mean() <= 0.95


def test_on_us_consistent_with_independence(
    data: GeneratedData, transactions: pl.DataFrame
) -> None:
    issuer_group = dict(data.tables["issuers"].select("issuer_id", "bank_group").iter_rows())
    acquirer_group = dict(data.tables["acquirers"].select("acquirer_id", "bank_group").iter_rows())
    tx = transactions.with_columns(
        pl.col("issuer_id")
        .cast(pl.Utf8)
        .replace_strict(issuer_group, return_dtype=pl.Utf8)
        .alias("ig"),
        pl.col("acquirer_id")
        .cast(pl.Utf8)
        .replace_strict(acquirer_group, return_dtype=pl.Utf8)
        .alias("ag"),
    )
    rule = (pl.col("ig") == pl.col("ag")).fill_null(False)
    assert tx.select((pl.col("on_us") == rule).all()).item()
    # Con adquirente y emisor independientes, P(on-us) = suma_g P(emisor en g) P(adquirente en g).
    shares = {
        c: dict(tx.group_by(c).len().with_columns(pl.col("len") / tx.height).iter_rows())
        for c in ("ig", "ag")
    }
    expected = sum(p * shares["ag"].get(g, 0.0) for g, p in shares["ig"].items() if g is not None)
    assert tx["on_us"].mean() > 0
    assert abs(tx["on_us"].mean() - expected) < 0.01


def test_merchant_concentration(transactions: pl.DataFrame) -> None:
    volume = transactions.group_by("merchant_id").agg(pl.col("amount_cop").sum())["amount_cop"]
    assert top_share(volume, 0.10) > 0.40


def test_pricing_model_follows_acquirer_and_tier(cfg: ProjectConfig, data: GeneratedData) -> None:
    # IC++ solo para comercios grandes y medianos de adquirentes bancarios (mixed); el resto,
    # incluidos los comercios del exterior, paga blended.
    acquirers = data.tables["acquirers"].select(
        "acquirer_id", pl.col("pricing_model").alias("acquirer_model")
    )
    merchants = data.tables["merchants"].join(acquirers, on="acquirer_id")
    icpp_tier = pl.col("size_tier").cast(pl.Utf8).is_in(list(cfg.merchants.icpp_tiers))
    expected = (pl.col("acquirer_model") == "icpp") | (
        (pl.col("acquirer_model") == "mixed") & icpp_tier
    )
    assert merchants.select(((pl.col("pricing_model") == "icpp") == expected).all()).item()
    counts = dict(merchants["pricing_model"].cast(pl.Utf8).value_counts().iter_rows())
    assert counts.get("icpp", 0) > 0
    assert counts.get("blended", 0) > 0


# ---------------------------------------------------------------------------
# Comunidades en el grafo tarjetahabiente-comercio
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def graph(data: GeneratedData) -> dict[str, pl.DataFrame]:
    tables, latent = data.tables, data.latent

    def planted(city: pl.Expr, segment: pl.Expr) -> pl.Expr:
        return pl.concat_str(
            [city.cast(pl.Utf8).fill_null("online"), segment.cast(pl.Utf8)], separator="|"
        )

    cardholders = latent["cardholders_latent"].join(tables["cardholders"], on="cardholder_id")
    merchants = latent["merchants_latent"].join(tables["merchants"], on="merchant_id")
    return {
        "edges": edge_list(tables["transactions"]),
        "cardholders": cardholders.select(
            "cardholder_id", planted(pl.col("city"), pl.col("segment")).alias("community")
        ),
        "merchants": merchants.select(
            "merchant_id",
            planted(pl.col("city"), pl.col("clientele")).alias("community"),
            "clientele",
            "mcc_group",
        ),
    }


def test_planted_communities_beat_null(graph: dict[str, pl.DataFrame]) -> None:
    edges, cardholders = graph["edges"], graph["cardholders"]
    merchants = graph["merchants"].select("merchant_id", "community")
    q = bipartite_modularity(edges, cardholders, merchants)
    null = shuffled_modularity(edges, cardholders, merchants, np.random.default_rng(0), 20)
    assert q >= 0.25
    assert q > null.mean() + 10 * null.std()


def test_leiden_finds_communities_across_mcc(graph: dict[str, pl.DataFrame]) -> None:
    pytest.importorskip("igraph")
    cardholders, merchants = leiden_partition(graph["edges"], seed=1, resolution=LEIDEN_RESOLUTION)
    assert bipartite_modularity(graph["edges"], cardholders, merchants) >= 0.25
    labelled = merchants.join(graph["merchants"].rename({"community": "planted"}), on="merchant_id")

    def nmi(column: str) -> float:
        return normalized_mutual_info(labelled["community"], labelled[column])

    # Las comunidades recuperan la estructura plantada (ciudad x clientela) y cruzan los MCC:
    # se parecen mucho más a la clientela que al grupo de MCC.
    assert nmi("planted") >= 0.4
    assert nmi("clientele") > 2 * nmi("mcc_group")


# ---------------------------------------------------------------------------
# Escala completa (criterio de terminado de la spec): `pytest -m slow`
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_full_scale_generation(tmp_path: Path) -> None:
    start = time.perf_counter()
    main(["--out", str(tmp_path)])
    elapsed = time.perf_counter() - start
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    rows = pl.scan_parquet(tmp_path / "transactions.parquet").select(pl.len()).collect().item()
    assert rows == manifest["rows"]["transactions"] >= 5_000_000
    assert elapsed < 180


@pytest.mark.slow
def test_validation_notebook_runs(monkeypatch: pytest.MonkeyPatch) -> None:
    import os

    from ips.utils.config import load_config, project_root, resolve_path

    nbformat = pytest.importorskip("nbformat")
    nbclient = pytest.importorskip("nbclient")
    if not (resolve_path(load_config().output.raw_dir) / "manifest.json").exists():
        pytest.skip("run `python -m ips.data_gen.generate` first")
    root = project_root()
    # The kernel is a subprocess: point it at this checkout's src, not another install.
    paths = [str(root / "src"), os.environ.get("PYTHONPATH")]
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join(p for p in paths if p))
    notebook = nbformat.read(root / "notebooks" / "01_data_validation.ipynb", as_version=4)
    nbclient.NotebookClient(
        notebook,
        timeout=900,
        kernel_name="python3",
        resources={"metadata": {"path": str(root / "notebooks")}},
    ).execute()
