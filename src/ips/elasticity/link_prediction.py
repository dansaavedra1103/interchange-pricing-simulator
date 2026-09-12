"""Where the volume goes: ranking the merchants a cardholder would move to.

Feature 4 asked the graph whether it explained what a merchant *pays*, and the answer was no.
This is the other question, and the one the graph was built for: if a merchant stops accepting
cards, which merchants pick up that spend? The ladder is judged the same way — against
baselines that use no graph at all, on a split the methods cannot see.

**Protocol.** The window is cut in time: the last ``holdout_months`` are held out and only the
earlier months train. A positive is a (cardholder, merchant) pair that appears in the held-out
months and *not* in training — a new link. Repeat purchases are excluded because predicting
them is just reading the history back.

Every method ranks the same candidate merchants for the same cardholders, with each
cardholder's own training history masked out, so the comparison is about the representation
and nothing else.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

from ips.utils.config import ProjectConfig

RANDOM, POPULARITY = "random", "popularity"
GROUP_POPULARITY, SVD, GRAPHSAGE = "group_popularity", "graph_svd", "graphsage"
LADDER = (RANDOM, POPULARITY, GROUP_POPULARITY, SVD, GRAPHSAGE)
EVALUATION_COLUMNS = (
    "method",
    "recall_at_k",
    "hit_rate_at_k",
    "mrr",
    "coverage",
    "cardholders",
    "positives",
)
GAP_COLUMNS = (
    "baseline",
    "challenger",
    "recall_baseline",
    "recall_challenger",
    "gap",
    "ci_low",
    "ci_high",
    "confidence_level",
)
# Las tres preguntas de la escalera: ¿le gana cada representación del grafo al baseline fuerte
# sin grafo, y le gana el GNN a la factorización?
VERDICT_PAIRS = ((GROUP_POPULARITY, SVD), (GROUP_POPULARITY, GRAPHSAGE), (SVD, GRAPHSAGE))
# Filas de similitud por bloque al buscar sustitutos: la memoria queda en bloque x comercios.
_SUBSTITUTE_BLOCK = 1024


@dataclass(frozen=True)
class LinkSplit:
    """A temporal split of the bipartite graph, plus everything the ladder is scored on."""

    train: pl.DataFrame
    test_pairs: pl.DataFrame
    candidates: np.ndarray
    cardholders: np.ndarray

    @property
    def shape(self) -> tuple[int, int]:
        """(evaluated cardholders, candidate merchants)."""
        return len(self.cardholders), len(self.candidates)

    def stats(self) -> dict[str, float]:
        """Size of the split, for the logs and the notebook."""
        cardholders, candidates = self.shape
        return {
            "train_edges": float(self.train.height),
            "eval_cardholders": float(cardholders),
            "candidate_merchants": float(candidates),
            "positives": float(self.test_pairs.height),
            "positives_per_cardholder": float(self.test_pairs.height / max(cardholders, 1)),
        }


def _index_of(values: np.ndarray, universe: np.ndarray) -> np.ndarray:
    """Position of each value inside a sorted universe."""
    return np.searchsorted(universe, values)


def split_by_time(edges: pl.DataFrame, cfg: ProjectConfig) -> LinkSplit:
    """Cut the window in time and assemble the evaluation set.

    El corte es temporal y no aleatorio a propósito: predecir un enlace del pasado con
    información del futuro no es la pregunta que el simulador necesita responder.
    """
    settings = cfg.elasticity.link_prediction
    rng = np.random.default_rng(cfg.seed)
    months = edges["txn_month"].unique().sort()
    if months.len() <= settings.holdout_months:
        raise ValueError(
            f"The window has {months.len()} months, not enough to hold out "
            f"{settings.holdout_months}"
        )
    cutoff = months[-settings.holdout_months]

    train = edges.filter(pl.col("txn_month") < cutoff)
    held_out = edges.filter(pl.col("txn_month") >= cutoff)

    # Un tope de titulares acota el costo del GNN en CPU. Se aplica antes de todo lo demás,
    # así que los cuatro métodos entrenan exactamente sobre el mismo grafo.
    population = train.select("cardholder_id").unique().sort("cardholder_id")
    if population.height > settings.train_cardholders:
        chosen = rng.choice(population.height, settings.train_cardholders, replace=False)
        population = population[np.sort(chosen)]
    train = train.join(population, on="cardholder_id", how="inner")
    if train.is_empty():
        raise ValueError("No training edges left after the temporal split")

    merchants = (
        train.group_by("merchant_id")
        .agg(pl.col("n_txns").sum().alias("train_txns"))
        .sort(["train_txns", "merchant_id"], descending=[True, False])
        .head(settings.candidate_merchants)
    )
    candidates = np.sort(merchants["merchant_id"].to_numpy())

    train_pairs = train.select("cardholder_id", "merchant_id").unique()
    positives = (
        held_out.select("cardholder_id", "merchant_id")
        .unique()
        .join(population, on="cardholder_id", how="inner")
        .join(pl.DataFrame({"merchant_id": candidates}), on="merchant_id", how="inner")
        .join(train_pairs, on=["cardholder_id", "merchant_id"], how="anti")
    )

    counts = positives.group_by("cardholder_id").len()
    eligible = (
        counts.filter(pl.col("len") >= settings.min_new_links)
        .select("cardholder_id")
        .sort("cardholder_id")
    )
    if eligible.is_empty():
        raise ValueError("No cardholder has enough new links to evaluate")
    if eligible.height > settings.eval_cardholders:
        chosen = rng.choice(eligible.height, settings.eval_cardholders, replace=False)
        eligible = eligible[np.sort(chosen)]
    cardholders = eligible["cardholder_id"].to_numpy()

    return LinkSplit(
        train=train,
        test_pairs=positives.join(eligible, on="cardholder_id", how="inner").sort(
            ["cardholder_id", "merchant_id"]
        ),
        candidates=candidates,
        cardholders=cardholders,
    )


def restrict_candidates(split: LinkSplit, keep: np.ndarray, cfg: ProjectConfig) -> LinkSplit:
    """Drop candidates no representation can score, and the positives that pointed at them.

    Un comercio que el embedding no cubre no es rankeable por nadie: dejarlo dentro castigaría a
    todos los métodos por igual, pero con ruido y no con información.
    """
    candidates = np.sort(np.intersect1d(split.candidates, keep))
    if len(candidates) <= cfg.elasticity.link_prediction.top_k:
        raise ValueError("Too few candidate merchants survive the intersection")
    survivors = pl.DataFrame({"merchant_id": candidates})
    positives = split.test_pairs.join(survivors, on="merchant_id", how="inner")
    counts = positives.group_by("cardholder_id").len()
    cardholders = np.sort(counts["cardholder_id"].to_numpy())
    cardholders = np.intersect1d(cardholders, split.cardholders)
    if len(cardholders) == 0:
        raise ValueError("No cardholder keeps a positive after restricting the candidates")
    return LinkSplit(
        train=split.train,
        test_pairs=positives.join(
            pl.DataFrame({"cardholder_id": cardholders}), on="cardholder_id", how="inner"
        ).sort(["cardholder_id", "merchant_id"]),
        candidates=candidates,
        cardholders=cardholders,
    )


def _pair_matrix(pairs: pl.DataFrame, split: LinkSplit) -> np.ndarray:
    """Boolean (cardholder x candidate) matrix from a frame of pairs."""
    rows, columns = split.shape
    matrix = np.zeros((rows, columns), dtype=bool)
    inside = pairs.join(
        pl.DataFrame({"cardholder_id": split.cardholders}), on="cardholder_id"
    ).join(pl.DataFrame({"merchant_id": split.candidates}), on="merchant_id")
    if inside.is_empty():
        return matrix
    row = _index_of(inside["cardholder_id"].to_numpy(), split.cardholders)
    column = _index_of(inside["merchant_id"].to_numpy(), split.candidates)
    matrix[row, column] = True
    return matrix


def history_mask(split: LinkSplit) -> np.ndarray:
    """Merchants each evaluated cardholder already bought from in training."""
    return _pair_matrix(split.train.select("cardholder_id", "merchant_id").unique(), split)


def positives_mask(split: LinkSplit) -> np.ndarray:
    """The new links each evaluated cardholder actually made in the held-out months."""
    return _pair_matrix(split.test_pairs, split)


def _cardholder_metrics(
    score: np.ndarray, seen: np.ndarray, positives: np.ndarray, k: int
) -> dict[str, np.ndarray]:
    """Per evaluated cardholder: recall, whether anything was found, reciprocal rank, top k."""
    ranked = np.where(seen, -np.inf, score.astype(np.float64))
    top = np.argsort(-ranked, axis=1, kind="stable")[:, :k]
    hits = np.take_along_axis(positives, top, axis=1)
    found = hits.any(axis=1)
    # El primer acierto de cada titular da su recíproco del rango; sin acierto, cero.
    first = np.argmax(hits, axis=1)
    return {
        "recall": hits.sum(axis=1) / np.maximum(positives.sum(axis=1), 1),
        "found": found,
        "reciprocal": np.where(found, 1.0 / (first + 1.0), 0.0),
        "top": top,
    }


def evaluate_rankings(
    scores: dict[str, np.ndarray], split: LinkSplit, cfg: ProjectConfig
) -> pl.DataFrame:
    """Score every method on the same split: recall@k, hit rate, MRR and coverage."""
    k = cfg.elasticity.link_prediction.top_k
    seen, positives = history_mask(split), positives_mask(split)
    if not positives.any():
        raise ValueError("No positives left to evaluate")

    rows = []
    for method in LADDER:
        if method not in scores:
            continue
        metrics = _cardholder_metrics(scores[method], seen, positives, k)
        rows.append(
            {
                "method": method,
                "recall_at_k": float(metrics["recall"].mean()),
                "hit_rate_at_k": float(metrics["found"].mean()),
                "mrr": float(metrics["reciprocal"].mean()),
                "coverage": float(len(np.unique(metrics["top"])) / len(split.candidates)),
                "cardholders": len(split.cardholders),
                "positives": int(positives.sum()),
            }
        )
    return pl.DataFrame(rows).select(EVALUATION_COLUMNS)


def ranking_gaps(
    scores: dict[str, np.ndarray],
    split: LinkSplit,
    cfg: ProjectConfig,
    pairs: tuple[tuple[str, str], ...] = VERDICT_PAIRS,
) -> pl.DataFrame:
    """Recall@k gap of each challenger over its baseline, with a paired bootstrap interval.

    Pareado: cada remuestreo toma los mismos titulares para los dos métodos, así que el
    intervalo mide la incertidumbre de la diferencia y no la de cada método por separado. Todos
    los pares usan los mismos remuestreos, sembrados con la semilla del proyecto.
    """
    settings = cfg.elasticity.link_prediction
    methods = sorted({method for pair in pairs for method in pair})
    missing = [method for method in methods if method not in scores]
    if missing:
        raise KeyError(f"scores are missing methods: {missing}")
    seen, positives = history_mask(split), positives_mask(split)
    recall = {
        method: _cardholder_metrics(scores[method], seen, positives, settings.top_k)["recall"]
        for method in methods
    }
    n = len(split.cardholders)
    draws = np.random.default_rng(cfg.seed).integers(0, n, size=(settings.bootstrap_samples, n))
    tail = (1.0 - settings.confidence_level) / 2.0
    rows = []
    for baseline, challenger in pairs:
        difference = recall[challenger] - recall[baseline]
        low, high = np.quantile(difference[draws].mean(axis=1), [tail, 1.0 - tail])
        rows.append(
            {
                "baseline": baseline,
                "challenger": challenger,
                "recall_baseline": float(recall[baseline].mean()),
                "recall_challenger": float(recall[challenger].mean()),
                "gap": float(difference.mean()),
                "ci_low": float(low),
                "ci_high": float(high),
                "confidence_level": settings.confidence_level,
            }
        )
    return pl.DataFrame(rows).select(GAP_COLUMNS)


def verdict(gaps: pl.DataFrame, baseline: str, challenger: str) -> str:
    """One sentence on whether the challenger earns its place, whichever way it comes out.

    Misma forma que ``ips.graph.evaluate.verdict``, con una regla más: una ventaja cuyo
    intervalo cruza el cero no cuenta como victoria. La conclusión se escribe sola desde los
    números, para que no dependa de cómo se quiera contar el resultado.
    """
    row = gaps.filter((pl.col("baseline") == baseline) & (pl.col("challenger") == challenger))
    if row.height != 1:
        raise KeyError(f"gaps must hold exactly one row for {challenger!r} over {baseline!r}")
    values = row.row(0, named=True)
    if values["ci_low"] > 0:
        relation = "beats"
    elif values["ci_high"] < 0:
        relation = "falls short of"
    else:
        relation = "cannot be told apart from"
    challenger_recall, baseline_recall = values["recall_challenger"], values["recall_baseline"]
    return (
        f"The {challenger} ranking {relation} {baseline} at finding the merchants a cardholder "
        f"moves to: recall@k {challenger_recall:.3f} vs {baseline_recall:.3f} "
        f"(gap {values['gap']:+.3f}, {values['confidence_level']:.0%} CI "
        f"{values['ci_low']:+.3f} to {values['ci_high']:+.3f})."
    )


def merchant_substitutes(
    vectors: np.ndarray, merchant_ids: np.ndarray, cfg: ProjectConfig
) -> pl.DataFrame:
    """Top substitutes of every merchant by similarity of their vectors.

    Es la pieza que consume el simulador: cuando un comercio deja de aceptar, su volumen se
    reparte entre estos. No se restringe al mismo grupo de MCC a propósito — si el grafo sirve,
    debe encontrar sustitutos que la categoría no ve, y eso se mide, no se impone.

    La similitud se calcula por bloques de filas: la matriz completa de 28.000 comercios pesa
    ~6 GB, y de cada fila solo hacen falta los primeros.
    """
    top = cfg.elasticity.redistribution.top_substitutes
    n = len(merchant_ids)
    if n <= top:
        raise ValueError("Not enough merchants to build a substitute list")
    normalised = vectors / np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)
    chosen = np.empty((n, top), dtype=np.int64)
    similarity = np.empty((n, top), dtype=np.float64)
    for start in range(0, n, _SUBSTITUTE_BLOCK):
        stop = min(start + _SUBSTITUTE_BLOCK, n)
        block = normalised[start:stop] @ normalised.T
        block[np.arange(stop - start), np.arange(start, stop)] = -np.inf
        candidates = np.argpartition(-block, top - 1, axis=1)[:, :top]
        values = np.take_along_axis(block, candidates, axis=1)
        # Dentro de los elegidos: similitud descendente y, a igual similitud, el índice menor.
        order = np.lexsort((candidates, -values), axis=1)
        chosen[start:stop] = np.take_along_axis(candidates, order, axis=1)
        similarity[start:stop] = np.take_along_axis(values, order, axis=1)
    return pl.DataFrame(
        {
            "merchant_id": np.repeat(merchant_ids, top),
            "substitute_id": merchant_ids[chosen.ravel()],
            "similarity": similarity.ravel(),
            "rank": np.tile(np.arange(1, top + 1), n),
        }
    ).sort(["merchant_id", "rank"])


def redistribute_volume(
    lost: pl.DataFrame, substitutes: pl.DataFrame, cfg: ProjectConfig
) -> pl.DataFrame:
    """Spread the volume of merchants that drop the card over their substitutes.

    Conservación: lo repartido más la fuga iguala lo perdido. Parte del volumen no pasa a otro
    comercio de la red — se va al efectivo o al pago instantáneo — y esa fuga es un parámetro,
    no un residuo.
    """
    if "gdv_lost_cop" not in lost.columns:
        raise ValueError("lost must carry a 'gdv_lost_cop' column")
    leakage = cfg.elasticity.redistribution.leakage_share
    weights = (
        substitutes.join(lost.select("merchant_id", "gdv_lost_cop"), on="merchant_id", how="inner")
        # Las similitudes pueden ser negativas: se desplazan a positivo antes de normalizar,
        # porque un peso negativo no significa "volumen negativo".
        .with_columns(
            (pl.col("similarity") - pl.col("similarity").min().over("merchant_id") + 1e-6).alias(
                "weight"
            )
        )
        .with_columns(
            (pl.col("weight") / pl.col("weight").sum().over("merchant_id")).alias("share")
        )
    )
    return (
        weights.with_columns(
            (pl.col("gdv_lost_cop") * (1.0 - leakage) * pl.col("share")).alias("gdv_moved_cop")
        )
        .group_by(pl.col("substitute_id").alias("merchant_id"))
        .agg(pl.col("gdv_moved_cop").sum())
        .sort("merchant_id")
    )


def redistribution_leakage(
    lost: pl.DataFrame, substitutes: pl.DataFrame, cfg: ProjectConfig
) -> float:
    """Volume that leaves the card rails instead of moving to another merchant.

    Dos fuentes: la fracción configurada del volumen que sí tiene sustitutos, y **todo** el
    volumen de los comercios que no tienen ninguno. Si no hay a quién pasárselo no se reparte,
    y contarlo como movido rompería la conservación.
    """
    share = cfg.elasticity.redistribution.leakage_share
    covered = substitutes.select("merchant_id").unique()
    placed = lost.join(covered, on="merchant_id", how="semi")
    unplaced = lost.join(covered, on="merchant_id", how="anti")
    return float(placed["gdv_lost_cop"].sum()) * share + float(unplaced["gdv_lost_cop"].sum())
