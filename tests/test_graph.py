"""Merchant segmentation: the graph is well formed, the embeddings reproduce, the comparison
measures what it claims."""

from __future__ import annotations

import datetime as dt
from dataclasses import replace

import numpy as np
import polars as pl
import pytest

from ips.data_gen.diagnostics import normalized_mutual_info
from ips.data_gen.generate import GeneratedData
from ips.data_gen.interchange_table import build_interchange_table
from ips.economics.pnl import compute_pnl
from ips.graph.baseline_kmeans import BENCHMARK, feature_matrix
from ips.graph.build_graph import (
    build_bipartite,
    edges_from_transactions,
    project_merchants,
    window_edges,
)
from ips.graph.clustering import UNASSIGNED, cluster_merchants
from ips.graph.evaluate import (
    COMPARISON_COLUMNS,
    compare_segmentations,
    eta_squared,
    marginal_information,
    marginal_verdict,
    out_of_sample,
    train_test_split,
    verdict,
)
from ips.graph.features import clustered_population, merchant_profile
from ips.graph.pipeline import (
    ARTIFACTS,
    BASELINE,
    GRAPH,
    SUPERVISED,
    SegmentationRun,
    segment_merchants,
    write_artifacts,
)
from ips.graph.spectral_embed import MerchantEmbedding, embed_merchants
from ips.graph.supervised import fit_supervised
from ips.utils.config import ProjectConfig

pytest.importorskip("sklearn", reason="needs the graph extra: pip install -e '.[graph]'")


@pytest.fixture(scope="module")
def priced(cfg: ProjectConfig, data: GeneratedData) -> pl.DataFrame:
    return compute_pnl(
        data.tables["transactions"],
        build_interchange_table(cfg),
        cfg,
        merchants=data.tables["merchants"],
    ).transactions


@pytest.fixture(scope="module")
def edges(priced: pl.DataFrame) -> pl.DataFrame:
    return edges_from_transactions(priced)


@pytest.fixture(scope="module")
def profile(cfg: ProjectConfig, priced: pl.DataFrame, data: GeneratedData) -> pl.DataFrame:
    return merchant_profile(priced, data.tables["merchants"], cfg)


@pytest.fixture(scope="module")
def run(
    cfg: ProjectConfig, edges: pl.DataFrame, profile: pl.DataFrame, data: GeneratedData
) -> SegmentationRun:
    latent = data.latent["merchants_latent"].select("merchant_id", "clientele")
    return segment_merchants(edges, profile, cfg, latent)


TOY_BLOCKS = ((40, range(0, 12)), (25, range(12, 24)))


def _toy_edges() -> pl.DataFrame:
    """Two clienteles with their own merchants, plus a few crossings so the graph connects.

    Los bloques son de distinto tamaño y densidad a propósito: dos bloques idénticos dejan el
    espectro degenerado y los vectores de uno y otro se pueden intercambiar.
    """
    rng = np.random.default_rng(7)
    rows = []
    card = 0
    for cards, merchants in TOY_BLOCKS:
        others = [m for block in TOY_BLOCKS for m in block[1] if m not in merchants]
        for _ in range(cards):
            for merchant in merchants:
                if rng.random() < 0.75:
                    rows.append((card, merchant, int(rng.integers(1, 6))))
            if rng.random() < 0.15:
                rows.append((card, int(rng.choice(others)), 1))
            card += 1
    return pl.DataFrame(
        {
            "txn_month": [dt.date(2025, 1, 1)] * len(rows),
            "cardholder_id": pl.Series([r[0] for r in rows], dtype=pl.UInt32),
            "merchant_id": pl.Series([r[1] for r in rows], dtype=pl.UInt32),
            "n_txns": pl.Series([r[2] for r in rows], dtype=pl.Int64),
            "amount_cop": [50_000.0] * len(rows),
        }
    )


# ---------------------------------------------------------------------------
# El grafo
# ---------------------------------------------------------------------------


def test_graph_is_well_formed(cfg: ProjectConfig, edges: pl.DataFrame) -> None:
    graph = build_bipartite(edges, cfg)
    pairs = edges.group_by("cardholder_id", "merchant_id").agg(pl.col("n_txns").sum())
    assert graph.shape == (graph.cardholder_ids.size, graph.merchant_ids.size)
    assert graph.matrix.nnz == pairs.height
    assert (graph.matrix.data > 0).all()
    assert np.array_equal(graph.merchant_ids, np.sort(graph.merchant_ids))
    assert graph.merchant_ids.size == pairs["merchant_id"].n_unique()
    # Los grados salen de la matriz y deben coincidir con contarlos en polars.
    degree = pairs.group_by("cardholder_id").len().sort("cardholder_id")
    assert np.array_equal(graph.cardholder_merchants(), degree["len"].to_numpy())
    assert graph.merchant_purchases().sum() == pytest.approx(pairs["n_txns"].sum())
    assert 0 < graph.stats()["density"] < 1


def test_degree_cap_drops_the_cards_that_buy_everywhere(
    cfg: ProjectConfig, edges: pl.DataFrame
) -> None:
    full = build_bipartite(edges, cfg)
    segmentation = cfg.segmentation.model_copy(update={"max_cardholder_degree": 5})
    capped = build_bipartite(edges, cfg.model_copy(update={"segmentation": segmentation}))
    assert capped.shape[0] < full.shape[0]
    assert capped.cardholder_merchants().max() <= 5


def test_window_keeps_only_the_last_months(edges: pl.DataFrame) -> None:
    months = edges["txn_month"].n_unique()
    recent = window_edges(edges, 6)
    assert months > 6
    assert recent["txn_month"].n_unique() == 6
    assert recent["txn_month"].max() == edges["txn_month"].max()
    assert recent.height < edges.height


def test_projection_is_pruned(cfg: ProjectConfig, edges: pl.DataFrame) -> None:
    projection = project_merchants(edges, cfg)
    assert projection.height > 0
    assert (projection["merchant_a"] < projection["merchant_b"]).all()
    assert projection.select("merchant_a", "merchant_b").is_duplicated().sum() == 0
    assert projection["weight"].min() >= cfg.segmentation.projection_min_weight


def test_projection_does_not_depend_on_row_order(cfg: ProjectConfig, edges: pl.DataFrame) -> None:
    # Misma semilla, mismo grafo: si los empates se rompieran por orden de filas, Leiden
    # devolvería comunidades distintas en cada corrida.
    shuffled = edges.sample(fraction=1.0, shuffle=True, seed=11)
    assert project_merchants(shuffled, cfg).equals(project_merchants(edges, cfg))


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------


def test_embeddings_are_deterministic_and_normalised(
    cfg: ProjectConfig, edges: pl.DataFrame
) -> None:
    graph = build_bipartite(edges, cfg)
    first = embed_merchants(graph, cfg)
    again = embed_merchants(graph, cfg)
    assert np.array_equal(first.vectors, again.vectors)
    assert first.vectors.shape == (graph.merchant_ids.size, first.dims)
    np.testing.assert_allclose(np.linalg.norm(first.vectors, axis=1), 1.0, atol=1e-9)


def test_embedding_separates_planted_clienteles(cfg: ProjectConfig) -> None:
    # Comercios que comparten tarjetas quedan cerca, aunque nadie le diga al modelo qué bloque
    # es cuál: esa es toda la apuesta del grafo frente a los atributos.
    graph = build_bipartite(_toy_edges(), cfg)
    embedding = embed_merchants(graph, cfg)
    similarity = embedding.vectors @ embedding.vectors.T
    block = (embedding.merchant_ids >= 12)[:, None]
    same_block = block == block.T
    np.fill_diagonal(same_block, False)
    assert similarity[same_block].mean() > similarity[~same_block].mean()

    np.fill_diagonal(similarity, -np.inf)
    nearest = similarity.argmax(axis=1)
    assert (block.ravel() == block.ravel()[nearest]).all()


def test_embedding_round_trips(cfg: ProjectConfig, edges: pl.DataFrame, tmp_path) -> None:
    embedding = embed_merchants(build_bipartite(edges, cfg), cfg)
    path = embedding.write_parquet(tmp_path / "embeddings" / "merchants.parquet")
    restored = MerchantEmbedding.read_parquet(path)
    assert np.array_equal(restored.merchant_ids, embedding.merchant_ids)
    np.testing.assert_allclose(restored.vectors, embedding.vectors)


# ---------------------------------------------------------------------------
# Perfil y clustering
# ---------------------------------------------------------------------------


def test_profile_matches_a_manual_aggregate(
    cfg: ProjectConfig, profile: pl.DataFrame, priced: pl.DataFrame
) -> None:
    merchant = int(profile.sort("gdv_cop", descending=True)["merchant_id"].first())
    rows = priced.filter(pl.col("merchant_id") == merchant)
    row = profile.filter(pl.col("merchant_id") == merchant).row(0, named=True)
    assert row["n_txns"] == rows.height
    assert row["gdv_cop"] == pytest.approx(rows["amount_cop"].sum())
    assert row["effective_mdr_rate"] == pytest.approx(
        rows["mdr_cop"].sum() / rows["amount_cop"].sum()
    )
    assert row["relative_burden"] == pytest.approx(row["effective_mdr_rate"] / row["sector_margin"])
    assert row["n_customers"] == rows["cardholder_id"].n_unique()


def test_population_respects_min_purchases(cfg: ProjectConfig, profile: pl.DataFrame) -> None:
    population = clustered_population(profile, cfg)
    assert population["n_txns"].min() >= cfg.segmentation.min_purchases
    assert 0 < population.height <= profile.height


def test_clustering_is_deterministic(cfg: ProjectConfig, run: SegmentationRun) -> None:
    graph_segments = run.method("graph")
    ids = run.population["merchant_id"].to_numpy()
    position = {int(m): i for i, m in enumerate(run.embedding.merchant_ids)}
    vectors = run.embedding.vectors[np.array([position[int(m)] for m in ids])]
    again = cluster_merchants("graph", ids, vectors, cfg, k=graph_segments.k)
    assert again.labels.equals(graph_segments.labels)
    assert graph_segments.k in cfg.segmentation.k_values


# ---------------------------------------------------------------------------
# Evaluación
# ---------------------------------------------------------------------------


def testeta_squared_bounds() -> None:
    values = np.array([1.0, 1.0, 2.0, 2.0])
    perfect = np.array([0, 0, 1, 1])
    assert eta_squared(values, perfect, None) == pytest.approx(1.0)
    assert eta_squared(values, np.array([0, 1, 0, 1]), None) == pytest.approx(0.0)


def test_out_of_sample_rewards_a_real_signal(cfg: ProjectConfig) -> None:
    rng = np.random.default_rng(cfg.seed)
    labels = np.repeat([0, 1, 2, 3], 100)
    values = labels * 1.0 + rng.normal(scale=0.05, size=labels.size)
    informative = out_of_sample(values, labels, cfg)
    noise = out_of_sample(values, rng.permutation(labels), cfg)
    assert informative["burden_r2_oos"] > 0.95
    assert noise["burden_r2_oos"] < 0.1
    assert informative["burden_mae_ratio"] < 0.2
    assert informative["burden_top_decile_lift"] > noise["burden_top_decile_lift"]


def test_comparison_covers_every_method(run: SegmentationRun) -> None:
    comparison = run.comparison
    assert comparison.columns == list(COMPARISON_COLUMNS)
    assert set(comparison["method"]) == {
        BENCHMARK,
        "baseline",
        "graph",
        "hybrid",
        "leiden",
        "supervised",
        "supervised_graph",
    }
    assert (comparison["segments"] >= 2).all()
    assert ((comparison["coverage"] > 0) & (comparison["coverage"] <= 1)).all()
    assert ((comparison["eta2_interchange"] <= 1) & (comparison["eta2_interchange"] >= -1e-9)).all()
    assert "R²" in run.verdict


def test_verdict_reads_the_out_of_sample_column() -> None:
    comparison = pl.DataFrame({"method": ["baseline", "graph"], "burden_r2_oos": [0.10, 0.25]})
    assert "beats" in verdict(comparison)
    assert "does not beat" in verdict(comparison.with_columns(burden_r2_oos=pl.Series([0.3, 0.1])))


def test_leiden_finds_the_planted_clientele(run: SegmentationRun, data: GeneratedData) -> None:
    # Diagnóstico del generador, no criterio de éxito: las comunidades deben parecerse a la
    # clientela plantada más que unas etiquetas barajadas.
    latent = data.latent["merchants_latent"].select("merchant_id", "clientele")
    labels = run.method("leiden").labels.filter(pl.col("segment") != UNASSIGNED)
    joined = labels.join(latent, on="merchant_id", how="inner")
    rng = np.random.default_rng(run.population.height)
    observed = normalized_mutual_info(joined["clientele"], joined["segment"])
    shuffled = normalized_mutual_info(
        joined["clientele"], pl.Series(rng.permutation(joined["segment"].to_numpy()))
    )
    assert observed > shuffled


# ---------------------------------------------------------------------------
# Perfiles y artefactos
# ---------------------------------------------------------------------------


def test_segment_profiles_add_up(run: SegmentationRun) -> None:
    profiles = run.profiles
    assert profiles.height == run.method(SUPERVISED).k
    assert profiles["volume_share"].sum() == pytest.approx(1.0)
    assert profiles["merchants"].sum() == run.population.height
    assert profiles["top_mcc_group"].null_count() == 0
    assert ((profiles["cnp_share"] >= 0) & (profiles["cnp_share"] <= 1)).all()


def test_blocks_weigh_the_same(cfg: ProjectConfig, profile: pl.DataFrame) -> None:
    # Sin equilibrar, las continuas estandarizadas dominan y la categoría deja de contar.
    matrix, names = feature_matrix(clustered_population(profile, cfg), cfg)
    numeric = len([n for n in names if "=" not in n])
    for block in (matrix[:, :numeric], matrix[:, numeric:]):
        assert float(np.linalg.norm(block, axis=1).mean()) == pytest.approx(1.0, rel=1e-9)


def test_benchmark_without_a_model_beats_kmeans(run: SegmentationRun) -> None:
    # La referencia honesta: dejar cada comercio en su grupo de MCC, sin analítica.
    scores = dict(zip(run.comparison["method"], run.comparison["burden_r2_oos"], strict=True))
    assert run.method(BENCHMARK).k == run.population["mcc_group"].n_unique()
    assert run.method(BENCHMARK).coverage == 1.0
    # Es el grupo de MCC: su información compartida con el grupo de MCC tiene que ser exacta.
    benchmark = run.comparison.filter(pl.col("method") == BENCHMARK)
    assert benchmark["nmi_mcc_group"].item() == pytest.approx(1.0)
    assert scores[BENCHMARK] > scores[BASELINE]
    assert scores[BENCHMARK] > scores[GRAPH]


def test_supervised_segmentation_leads_the_ladder(cfg: ProjectConfig, run: SegmentationRun) -> None:
    scores = dict(zip(run.comparison["method"], run.comparison["burden_r2_oos"], strict=True))
    assert scores[SUPERVISED] > scores[BASELINE]
    assert scores[SUPERVISED] > scores[GRAPH]
    assert run.method(SUPERVISED).k in cfg.segmentation.k_values
    assert run.method(SUPERVISED).coverage == 1.0


def test_supervised_never_sees_the_held_out_merchants(
    cfg: ProjectConfig, run: SegmentationRun
) -> None:
    # Si el árbol hubiera mirado la mitad de prueba, cambiarle el objetivo movería los segmentos.
    population = run.population
    matrix, _ = feature_matrix(population, cfg)
    _, test = train_test_split(population.height, cfg)
    values = population["relative_burden"].to_numpy().copy()
    values[test] = np.random.default_rng(0).permutation(values[test]) * 3.0
    altered = population.with_columns(pl.Series("relative_burden", values))
    again, _ = fit_supervised(altered, matrix, cfg, SUPERVISED, leaves=run.method(SUPERVISED).k)
    assert again.labels.equals(run.method(SUPERVISED).labels)


def test_evaluation_does_not_depend_on_the_row_order_of_the_labels(
    cfg: ProjectConfig, run: SegmentationRun
) -> None:
    # La partición de prueba es posicional: si la evaluación siguiera el orden que devuelve el
    # join, el método supervisado se mediría sobre comercios que vio al entrenarse.
    supervised = run.method(SUPERVISED)
    shuffled = replace(
        supervised, labels=supervised.labels.sample(fraction=1.0, shuffle=True, seed=3)
    )
    straight = compare_segmentations([supervised], run.population, cfg)
    mixed = compare_segmentations([shuffled], run.population, cfg)
    assert mixed["burden_r2_oos"].item() == pytest.approx(straight["burden_r2_oos"].item())
    assert mixed["eta2_burden"].item() == pytest.approx(straight["eta2_burden"].item())


def test_segment_rules_describe_every_segment(cfg: ProjectConfig, run: SegmentationRun) -> None:
    rules = run.rules
    train, _ = train_test_split(run.population.height, cfg)
    assert rules.height == run.method(SUPERVISED).k
    assert rules["rule"].str.len_chars().min() > 0
    assert rules["merchants_in_fit"].sum() == len(train)
    _, names = feature_matrix(run.population, cfg)
    mentioned = " ".join(rules["rule"].to_list())
    assert any(name.split("=")[0] in mentioned for name in names)


def test_marginal_information_separates_signal_from_noise(cfg: ProjectConfig) -> None:
    rng = np.random.default_rng(cfg.seed)
    n = 2_000
    signal = rng.normal(size=(n, 3))
    noise = rng.normal(size=(n, 3))
    values = np.where(signal[:, 0] > 0, 2.0, 1.0) + rng.normal(scale=0.05, size=n)
    spaces = {
        "attributes": signal,
        "graph": noise,
        "attributes + graph": np.hstack([signal, noise]),
    }
    marginal = marginal_information(spaces, values, cfg)
    scores = dict(zip(marginal["space"], marginal["r2_oos"], strict=True))
    assert scores["attributes"] > 0.9
    assert scores["graph"] < 0.1
    assert scores["attributes + graph"] <= scores["attributes"] + 0.02
    assert "adds nothing measurable" in marginal_verdict(marginal)


def test_artifacts_round_trip(cfg: ProjectConfig, run: SegmentationRun, tmp_path) -> None:
    output = cfg.output.model_copy(update={"artifacts_dir": tmp_path})
    paths = write_artifacts(run, cfg.model_copy(update={"output": output}))
    assert set(paths) == set(ARTIFACTS)
    assert all(path.exists() for path in paths.values())
    segments = pl.read_parquet(paths["segments"])
    assert segments.height == run.population.height
    expected = {"baseline_segment", "graph_segment", "hybrid_segment", "leiden_segment"}
    assert expected <= set(segments.columns)
    assert segments["graph_segment"].null_count() == 0
