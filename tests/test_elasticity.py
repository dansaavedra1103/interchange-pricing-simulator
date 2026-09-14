"""Elasticity: the acceptance curve behaves like one, the calibration lands on its anchors, and
the link-prediction ladder is measured without leakage."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from ips.data_gen.generate import GeneratedData
from ips.data_gen.interchange_table import build_interchange_table
from ips.economics.pnl import compute_pnl
from ips.elasticity import link_prediction
from ips.elasticity.calibration import calibrate_acceptance, level_targets
from ips.elasticity.cardholder_response import (
    cardholder_response,
    segment_response,
    spend_multiplier,
)
from ips.elasticity.link_prediction import (
    EVALUATION_COLUMNS,
    GAP_COLUMNS,
    GRAPHSAGE,
    LADDER,
    POPULARITY,
    RANDOM,
    SVD,
    VERDICT_PAIRS,
    LinkSplit,
    merchant_substitutes,
    ranking_gaps,
    redistribute_volume,
    redistribution_leakage,
    split_by_time,
    verdict,
)
from ips.elasticity.merchant_acceptance import (
    AcceptanceCurve,
    abandonment_probability,
    acceptance_response,
    burden_term,
    logit_offset,
)
from ips.elasticity.params import group_sensitivity
from ips.elasticity.pipeline import ARTIFACTS, ElasticityRun, elasticity_run, write_artifacts
from ips.elasticity.scorers import popularity_scores, random_scores
from ips.graph.build_graph import edges_from_transactions
from ips.graph.features import clustered_population, merchant_profile
from ips.utils.config import ProjectConfig


def _with_acceptance(cfg: ProjectConfig, **updates: object) -> ProjectConfig:
    acceptance = cfg.elasticity.acceptance.model_copy(update=updates)
    return cfg.model_copy(
        update={"elasticity": cfg.elasticity.model_copy(update={"acceptance": acceptance})}
    )


def _with_gnn(cfg: ProjectConfig, **updates: object) -> ProjectConfig:
    gnn = cfg.elasticity.link_prediction.gnn.model_copy(update=updates)
    link = cfg.elasticity.link_prediction.model_copy(update={"gnn": gnn})
    return cfg.model_copy(
        update={"elasticity": cfg.elasticity.model_copy(update={"link_prediction": link})}
    )


def _requires_gnn() -> None:
    for module in ("sklearn", "scipy", "torch", "torch_geometric"):
        pytest.importorskip(module, reason="needs the graph and gnn extras")


def _one_merchant(population: pl.DataFrame, cfg: ProjectConfig, n: int) -> pl.DataFrame:
    """One real merchant copied ``n`` times and moved to the most sensitive group.

    Así varía un solo driver a la vez, y la curva queda lejos de su cota inferior, donde el
    recorte aplanaría cualquier diferencia.
    """
    sensitivity = group_sensitivity(cfg)
    group = max(sensitivity, key=sensitivity.__getitem__)
    return pl.concat([population.head(1)] * n).with_columns(pl.lit(group).alias("mcc_group"))


@pytest.fixture(scope="module")
def priced(cfg: ProjectConfig, data: GeneratedData) -> pl.DataFrame:
    return compute_pnl(
        data.tables["transactions"],
        build_interchange_table(cfg),
        cfg,
        merchants=data.tables["merchants"],
    ).transactions


@pytest.fixture(scope="module")
def profile(cfg: ProjectConfig, priced: pl.DataFrame, data: GeneratedData) -> pl.DataFrame:
    return merchant_profile(priced, data.tables["merchants"], cfg)


@pytest.fixture(scope="module")
def population(cfg: ProjectConfig, profile: pl.DataFrame) -> pl.DataFrame:
    return clustered_population(profile, cfg)


@pytest.fixture(scope="module")
def curve(cfg: ProjectConfig, population: pl.DataFrame) -> AcceptanceCurve:
    return calibrate_acceptance(population, cfg)


@pytest.fixture(scope="module")
def edges(priced: pl.DataFrame) -> pl.DataFrame:
    return edges_from_transactions(priced)


@pytest.fixture(scope="module")
def cardholders(data: GeneratedData) -> pl.DataFrame:
    return data.tables["cardholders"].select("cardholder_id", "spend_segment", "product")


@pytest.fixture(scope="module")
def fast_cfg(cfg: ProjectConfig) -> ProjectConfig:
    """The same protocol with a short training budget: the tests check behaviour, not quality."""
    return _with_gnn(cfg, max_epochs=30, patience=10)


@pytest.fixture(scope="module")
def split(cfg: ProjectConfig, edges: pl.DataFrame) -> LinkSplit:
    return split_by_time(edges, cfg)


@pytest.fixture(scope="module")
def run(
    fast_cfg: ProjectConfig,
    edges: pl.DataFrame,
    profile: pl.DataFrame,
    population: pl.DataFrame,
    cardholders: pl.DataFrame,
) -> ElasticityRun:
    _requires_gnn()
    return elasticity_run(edges, profile, population, cardholders, fast_cfg)


@pytest.fixture(scope="module")
def trained_twice(
    fast_cfg: ProjectConfig, split: LinkSplit, profile: pl.DataFrame, cardholders: pl.DataFrame
):
    _requires_gnn()
    from ips.elasticity.gnn import fit_graphsage
    from ips.graph.baseline_kmeans import feature_matrix

    merchant_ids = np.intersect1d(
        split.train["merchant_id"].unique().to_numpy(), profile["merchant_id"].to_numpy()
    )
    merchants = profile.join(
        pl.DataFrame({"merchant_id": merchant_ids}), on="merchant_id", how="inner"
    ).sort("merchant_id")
    matrix, _ = feature_matrix(merchants, fast_cfg)
    first = fit_graphsage(split, cardholders, merchant_ids, matrix, fast_cfg)
    second = fit_graphsage(split, cardholders, merchant_ids, matrix, fast_cfg)
    return first, second


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


def test_config_carries_the_elasticity_section(cfg: ProjectConfig) -> None:
    acceptance = cfg.elasticity.acceptance
    assert (
        acceptance.min_probability
        < acceptance.target_annual_abandonment
        < acceptance.max_probability
    )
    assert set(cfg.elasticity.cardholder.rewards_semi_elasticity) == {"low", "mid", "high"}


def test_group_sensitivity_is_relative_to_the_median_group(cfg: ProjectConfig) -> None:
    sensitivity = group_sensitivity(cfg)
    assert set(sensitivity) == set(cfg.group_names)
    assert np.median(list(sensitivity.values())) == pytest.approx(1.0)
    flat = group_sensitivity(_with_acceptance(cfg, use_group_elasticity=False))
    assert set(flat.values()) == {1.0}


# ---------------------------------------------------------------------------
# The acceptance curve
# ---------------------------------------------------------------------------


def test_abandonment_rises_with_relative_burden(
    cfg: ProjectConfig, population: pl.DataFrame, curve: AcceptanceCurve
) -> None:
    grid = np.quantile(population["relative_burden"].to_numpy(), np.linspace(0.05, 1.0, 25))
    frame = _one_merchant(population, cfg, len(grid)).with_columns(
        pl.Series("relative_burden", grid)
    )
    probability = abandonment_probability(frame, cfg, curve)["p_abandonment"].to_numpy()
    assert np.all(np.diff(probability) >= 0)
    assert probability[-1] > probability[0]


def test_abandonment_falls_with_sector_margin(
    cfg: ProjectConfig, population: pl.DataFrame, curve: AcceptanceCurve
) -> None:
    # Misma tarifa efectiva, márgenes crecientes: la carga relativa baja y con ella el abandono.
    margins = np.array([0.03, 0.06, 0.12, 0.20])
    mdr_rate = float(population["effective_mdr_rate"].quantile(0.9))
    frame = _one_merchant(population, cfg, len(margins)).with_columns(
        pl.Series("sector_margin", margins),
        pl.Series("relative_burden", mdr_rate / margins),
    )
    probability = abandonment_probability(frame, cfg, curve)["p_abandonment"].to_numpy()
    assert np.all(np.diff(probability) <= 0)
    assert probability[0] > probability[-1]


def test_probabilities_stay_inside_their_bounds(
    cfg: ProjectConfig, population: pl.DataFrame, curve: AcceptanceCurve
) -> None:
    acceptance = cfg.elasticity.acceptance
    probability = abandonment_probability(population, cfg, curve)["p_abandonment"]
    assert probability.min() >= acceptance.min_probability
    assert probability.max() <= acceptance.max_probability


def test_card_not_present_raises_the_logit(cfg: ProjectConfig, population: pl.DataFrame) -> None:
    frame = _one_merchant(population, cfg, 2).with_columns(pl.Series("cnp_share", [0.0, 1.0]))
    offset = logit_offset(frame, cfg)
    assert offset[1] - offset[0] == pytest.approx(cfg.elasticity.acceptance.modifiers.cnp_share)
    assert offset[1] > offset[0]


def test_permitted_surcharge_lowers_the_logit(cfg: ProjectConfig, population: pl.DataFrame) -> None:
    frame = _one_merchant(population, cfg, 1)
    allowed = _with_acceptance(cfg, surcharge_allowed=True)
    shift = logit_offset(frame, allowed)[0] - logit_offset(frame, cfg)[0]
    assert shift == pytest.approx(cfg.elasticity.acceptance.modifiers.surcharge_allowed_shift)
    assert shift < 0


def test_instant_payments_press_on_small_debit_tickets(
    cfg: ProjectConfig, population: pl.DataFrame
) -> None:
    small = cfg.elasticity.acceptance.instant_payments.small_ticket_cop
    frame = _one_merchant(population, cfg, 3).with_columns(
        pl.Series("cnp_share", [0.0, 0.0, 0.0]),
        pl.Series("share_debit", [1.0, 1.0, 0.0]),
        pl.Series("avg_ticket_cop", [0.1 * small, 2.0 * small, 0.1 * small]),
    )
    offset = logit_offset(frame, cfg)
    # Débito de ticket bajo: sustituible. Ticket alto o sin débito: el pago instantáneo no pesa.
    assert offset[0] > 0
    assert offset[1] == pytest.approx(0.0)
    assert offset[2] == pytest.approx(0.0)


def test_icpp_merchants_react_more_to_the_same_burden(
    cfg: ProjectConfig, population: pl.DataFrame
) -> None:
    frame = _one_merchant(population, cfg, 3).with_columns(
        pl.Series("pricing_model", ["blended", "mixed", "icpp"])
    )
    term = burden_term(frame, cfg)
    assert term[0] < term[1] < term[2]


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------


def test_calibration_lands_on_both_anchors(curve: AcceptanceCurve) -> None:
    assert curve.achieved_abandonment == pytest.approx(curve.target_abandonment, abs=1e-6)
    assert curve.achieved_semi_elasticity == pytest.approx(curve.target_semi_elasticity, abs=1e-6)
    assert curve.beta > 0
    for group, target in curve.level_targets.items():
        assert curve.achieved_levels[group] == pytest.approx(target, abs=1e-6), group


def test_calibration_is_deterministic(
    cfg: ProjectConfig, population: pl.DataFrame, curve: AcceptanceCurve
) -> None:
    again = calibrate_acceptance(population, cfg)
    assert (again.intercepts, again.beta) == (curve.intercepts, curve.beta)


def test_group_levels_follow_their_burden(cfg: ProjectConfig, population: pl.DataFrame) -> None:
    rate = cfg.elasticity.acceptance.target_annual_abandonment
    targets = level_targets(population, cfg)
    groups = (
        population.group_by(pl.col("mcc_group").cast(pl.Utf8))
        .agg(pl.len().alias("merchants"), pl.col("relative_burden").median())
        .sort("relative_burden")
    )
    ordered = [targets[group] for group in groups["mcc_group"]]
    assert ordered == sorted(ordered)
    # Normalizados por el peso de cada grupo, los niveles devuelven la tasa del ancla.
    weighted = sum(
        targets[group] * merchants
        for group, merchants in zip(groups["mcc_group"], groups["merchants"], strict=True)
    )
    assert weighted / population.height == pytest.approx(rate)
    flat = level_targets(population, _with_acceptance(cfg, level_burden_exponent=0.0))
    assert all(level == pytest.approx(rate) for level in flat.values())


def test_group_levels_must_fit_inside_the_bounds(
    cfg: ProjectConfig, population: pl.DataFrame
) -> None:
    extreme = _with_acceptance(cfg, level_burden_exponent=12.0)
    with pytest.raises(ValueError, match="outside the probability bounds"):
        level_targets(population, extreme)


def test_calibration_refuses_anchors_the_population_cannot_meet(
    cfg: ProjectConfig, population: pl.DataFrame
) -> None:
    impossible = _with_acceptance(cfg, target_semi_elasticity=0.5)
    with pytest.raises(ValueError, match="inconsistent"):
        calibrate_acceptance(population, impossible)


def test_no_change_in_mdr_changes_nothing(
    cfg: ProjectConfig, population: pl.DataFrame, curve: AcceptanceCurve
) -> None:
    response = acceptance_response(population, cfg, curve, 0.0)
    assert response["p_abandonment_delta"].abs().max() == 0.0


def test_a_higher_mdr_never_lowers_abandonment(
    cfg: ProjectConfig, population: pl.DataFrame, curve: AcceptanceCurve
) -> None:
    shock = cfg.elasticity.acceptance.semi_elasticity_shock
    response = acceptance_response(population, cfg, curve, shock)
    assert response["p_abandonment_delta"].min() >= 0.0
    assert response["p_abandonment_delta"].mean() == pytest.approx(curve.achieved_semi_elasticity)


def test_every_group_pays_an_acceptance_cost_for_a_higher_mdr(
    cfg: ProjectConfig, population: pl.DataFrame, curve: AcceptanceCurve
) -> None:
    # Con un solo nivel para toda la población, la mayoría de los grupos quedaba en el piso y un
    # MDR más alto no les costaba aceptación: un optimizador lo habría leído como margen gratis.
    shock = cfg.elasticity.acceptance.semi_elasticity_shock
    response = acceptance_response(population, cfg, curve, shock).join(
        population.select("merchant_id", pl.col("mcc_group").cast(pl.Utf8)), on="merchant_id"
    )
    by_group = response.group_by("mcc_group").agg(pl.col("p_abandonment_delta").mean())
    assert by_group.height == len(curve.intercepts)
    assert (by_group["p_abandonment_delta"] > 0).all()


def test_an_mdr_cannot_fall_below_zero(
    cfg: ProjectConfig, population: pl.DataFrame, curve: AcceptanceCurve
) -> None:
    with pytest.raises(ValueError, match="greater than -1"):
        acceptance_response(population, cfg, curve, -1.0)


# ---------------------------------------------------------------------------
# Cardholder response
# ---------------------------------------------------------------------------


def test_unchanged_rewards_leave_spend_unchanged(cfg: ProjectConfig) -> None:
    assert segment_response(cfg, 0.0)["spend_multiplier"].to_list() == [1.0, 1.0, 1.0]


def test_a_rewards_cut_hits_high_spenders_hardest(cfg: ProjectConfig) -> None:
    response = segment_response(cfg, -0.01)
    multiplier = dict(zip(response["spend_segment"], response["spend_multiplier"], strict=True))
    assert multiplier["high"] < multiplier["mid"] < multiplier["low"] < 1.0


def test_spend_response_is_monotone_and_bounded(cfg: ProjectConfig) -> None:
    changes = np.linspace(-0.5, 0.5, 41)
    assert np.all(np.diff(spend_multiplier(changes, 1.6)) > 0)
    limit = cfg.elasticity.cardholder.max_spend_change
    extreme = segment_response(cfg, -1.0)["spend_multiplier"].to_numpy()
    assert extreme.min() == pytest.approx(1.0 - limit)


def test_cardholder_response_covers_every_cardholder(
    cfg: ProjectConfig, cardholders: pl.DataFrame
) -> None:
    response = cardholder_response(cardholders, -0.005, cfg)
    assert response.height == cardholders.height
    assert response["spend_multiplier"].max() < 1.0


# ---------------------------------------------------------------------------
# Link-prediction protocol
# ---------------------------------------------------------------------------


def test_split_holds_out_the_last_months(
    cfg: ProjectConfig, edges: pl.DataFrame, split: LinkSplit
) -> None:
    months = edges["txn_month"].unique().sort()
    cutoff = months[-cfg.elasticity.link_prediction.holdout_months]
    assert split.train["txn_month"].max() < cutoff


def test_test_links_never_appear_in_training(split: LinkSplit) -> None:
    seen = split.train.select("cardholder_id", "merchant_id").unique()
    leaked = split.test_pairs.join(seen, on=["cardholder_id", "merchant_id"], how="inner")
    assert leaked.is_empty()


def test_every_positive_is_rankable(split: LinkSplit) -> None:
    assert split.test_pairs["merchant_id"].is_in(split.candidates.tolist()).all()
    evaluated = set(split.cardholders.tolist())
    assert set(split.test_pairs["cardholder_id"].unique().to_list()) == evaluated


def test_split_is_deterministic(cfg: ProjectConfig, edges: pl.DataFrame, split: LinkSplit) -> None:
    again = split_by_time(edges, cfg)
    assert np.array_equal(again.cardholders, split.cardholders)
    assert np.array_equal(again.candidates, split.candidates)
    assert again.test_pairs.equals(split.test_pairs)


def test_ladder_has_one_row_per_method(run: ElasticityRun) -> None:
    assert tuple(run.comparison.columns) == EVALUATION_COLUMNS
    assert tuple(run.comparison["method"]) == LADDER


def test_every_method_beats_a_random_ranking(run: ElasticityRun) -> None:
    recall = dict(zip(run.comparison["method"], run.comparison["recall_at_k"], strict=True))
    for method in LADDER:
        if method != RANDOM:
            assert recall[method] > recall[RANDOM], method


def test_graphsage_is_deterministic(trained_twice) -> None:
    first, second = trained_twice
    assert np.array_equal(first.merchant_vectors, second.merchant_vectors)
    assert np.array_equal(first.cardholder_vectors, second.cardholder_vectors)
    assert first.losses == second.losses


def test_graphsage_stops_on_its_own_validation(fast_cfg: ProjectConfig, trained_twice) -> None:
    result, _ = trained_twice
    settings = fast_cfg.elasticity.link_prediction.gnn
    assert 1 <= result.best_epoch <= len(result.losses) <= settings.max_epochs
    assert len(result.validation_losses) == len(result.losses)
    if len(result.losses) < settings.max_epochs:
        assert len(result.losses) - result.best_epoch == settings.patience


def test_a_method_cannot_be_told_apart_from_itself(cfg: ProjectConfig, split: LinkSplit) -> None:
    scores = {POPULARITY: popularity_scores(split)}
    gaps = ranking_gaps(scores, split, cfg, pairs=((POPULARITY, POPULARITY),))
    row = gaps.row(0, named=True)
    assert (row["gap"], row["ci_low"], row["ci_high"]) == (0.0, 0.0, 0.0)
    assert "cannot be told apart" in verdict(gaps, POPULARITY, POPULARITY)


def test_gap_intervals_bracket_the_gap_and_reproduce(cfg: ProjectConfig, split: LinkSplit) -> None:
    scores = {RANDOM: random_scores(split, cfg), POPULARITY: popularity_scores(split)}
    pairs = ((RANDOM, POPULARITY),)
    gaps = ranking_gaps(scores, split, cfg, pairs=pairs)
    assert tuple(gaps.columns) == GAP_COLUMNS
    assert gaps.equals(ranking_gaps(scores, split, cfg, pairs=pairs))
    row = gaps.row(0, named=True)
    assert row["ci_low"] <= row["gap"] <= row["ci_high"]
    # Popularidad contra azar en la muestra: una ventaja que no se confunde con ruido.
    assert "beats" in verdict(gaps, RANDOM, POPULARITY)


# ---------------------------------------------------------------------------
# Redistribution and the run
# ---------------------------------------------------------------------------


def test_redistribution_conserves_volume(cfg: ProjectConfig) -> None:
    substitutes = pl.DataFrame(
        {
            "merchant_id": [1, 1, 1, 2, 2, 2],
            "substitute_id": [2, 3, 4, 1, 3, 4],
            "similarity": [0.9, 0.5, -0.2, 0.7, 0.7, 0.1],
            "rank": [1, 2, 3, 1, 2, 3],
        }
    )
    # El comercio 5 no tiene sustitutos: todo su volumen tiene que contarse como fuga.
    lost = pl.DataFrame({"merchant_id": [1, 2, 5], "gdv_lost_cop": [1_000.0, 250.0, 80.0]})
    moved = redistribute_volume(lost, substitutes, cfg)
    leaked = redistribution_leakage(lost, substitutes, cfg)
    assert float(moved["gdv_moved_cop"].sum()) + leaked == pytest.approx(1_330.0)
    assert leaked >= 80.0


def test_substitutes_do_not_depend_on_the_block_size(
    cfg: ProjectConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    rng = np.random.default_rng(7)
    vectors = rng.normal(size=(50, 8))
    ids = np.arange(100, 150, dtype=np.int64)
    whole = merchant_substitutes(vectors, ids, cfg)
    monkeypatch.setattr(link_prediction, "_SUBSTITUTE_BLOCK", 7)
    blocked = merchant_substitutes(vectors, ids, cfg)
    assert whole["substitute_id"].to_list() == blocked["substitute_id"].to_list()
    assert np.allclose(whole["similarity"].to_numpy(), blocked["similarity"].to_numpy())

    # El mejor sustituto de cada comercio es el de mayor similitud, y nunca él mismo.
    normalised = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    full = normalised @ normalised.T
    np.fill_diagonal(full, -np.inf)
    best = whole.filter(pl.col("rank") == 1)["substitute_id"].to_numpy()
    assert np.array_equal(best, ids[full.argmax(axis=1)])
    assert (whole["merchant_id"] != whole["substitute_id"]).all()


def test_the_run_conserves_the_volume_it_moves(run: ElasticityRun) -> None:
    assert run.conserves_volume()


def test_verdicts_state_every_comparison(run: ElasticityRun) -> None:
    sentences = run.verdicts()
    assert len(sentences) == 1 + len(VERDICT_PAIRS)
    assert "assumption, not an estimate" in sentences[0]
    assert all("recall@k" in sentence and "CI" in sentence for sentence in sentences[1:])
    assert run.substitute_source in {SVD, GRAPHSAGE}
    assert tuple(run.gaps.columns) == GAP_COLUMNS


def test_artifacts_are_written(run: ElasticityRun, cfg: ProjectConfig, tmp_path) -> None:
    local = cfg.model_copy(
        update={"output": cfg.output.model_copy(update={"artifacts_dir": tmp_path})}
    )
    paths = write_artifacts(run, local)
    assert set(paths) == set(ARTIFACTS)
    assert all(path.exists() for path in paths.values())
