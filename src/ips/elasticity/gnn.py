"""GraphSAGE over the cardholder-merchant graph, for link prediction.

This is the rung of the ladder that justifies bringing torch in. The PPMI+SVD embedding of
Feature 4 factorises the adjacency and nothing else: two merchants are close when the same
cards buy from both. GraphSAGE also reads **node attributes** — what the merchant sells, how big
it is, how its customers pay; what segment the cardholder is in, how active it is — and passes
them along the edges. If substitution is driven by anything beyond co-purchase, this is what
can see it.

Three decisions keep it honest and keep it runnable on a laptop CPU:

* **Message passing is separated from supervision, and supervision from validation.** The
  training edges are split three ways: one part builds the graph the encoder sees, one supplies
  the positives it is fitted on, and one decides when to stop. A model asked to predict an edge
  it was also handed as input learns to copy, and a model whose epochs are chosen on the
  held-out months is being tuned on its own exam.
* **Full-batch message passing, minibatched supervision.** One forward pass per step over the
  sparse graph, so PyG needs neither ``pyg-lib`` nor ``torch-sparse`` — both of which have to be
  compiled on Windows. The install stays a single wheel.
* **Single-threaded and seeded.** Thread count changes the order of floating-point reduction in
  the scatter operations, and with it the result. The project's invariant is that the same seed
  gives the same output, so the threads are capped during the fit and restored afterwards.
"""

from __future__ import annotations

import contextlib
import copy
from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
import polars as pl

from ips.elasticity.link_prediction import LinkSplit
from ips.utils.config import SPEND_SEGMENTS, ProjectConfig
from ips.utils.logging import get_logger

CARDHOLDER, MERCHANT, BUYS = "cardholder", "merchant", "buys"
_MESSAGE_PASSING_SHARE = 0.8
_VALIDATION_SHARE = 0.2


@dataclass(frozen=True)
class GraphSageResult:
    """Trained node vectors on both sides of the graph, plus the training record."""

    cardholder_ids: np.ndarray
    merchant_ids: np.ndarray
    cardholder_vectors: np.ndarray
    merchant_vectors: np.ndarray
    losses: tuple[float, ...]
    validation_losses: tuple[float, ...]
    best_epoch: int

    def frame(self) -> pl.DataFrame:
        """Both loss curves and where training stopped, for the notebook."""
        epochs = len(self.losses)
        return pl.DataFrame(
            {
                "epoch": list(range(1, epochs + 1)),
                "loss": list(self.losses),
                "validation_loss": list(self.validation_losses),
                "is_best": [epoch == self.best_epoch for epoch in range(1, epochs + 1)],
            }
        )


@contextlib.contextmanager
def _deterministic(cfg: ProjectConfig) -> Iterator[None]:
    """Seed torch and pin it to one thread for the duration of the fit."""
    import torch

    threads = torch.get_num_threads()
    torch.manual_seed(cfg.seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(threads)
        torch.use_deterministic_algorithms(False)


def cardholder_features(
    split: LinkSplit, cardholders: pl.DataFrame
) -> tuple[np.ndarray, list[str]]:
    """What the model knows about a cardholder: who they are, and how they shop in training.

    Nada de esto viene de la ventana de prueba: la actividad se mide solo sobre ``split.train``.
    """
    activity = split.train.group_by("cardholder_id").agg(
        pl.col("n_txns").sum().alias("txns"),
        pl.col("amount_cop").sum().alias("spend"),
        pl.len().alias("merchants"),
    )
    known = cardholders.select(
        "cardholder_id", pl.col("spend_segment").cast(pl.Utf8), pl.col("product").cast(pl.Utf8)
    )
    products = sorted(known["product"].unique().to_list())
    frame = (
        activity.join(known, on="cardholder_id", how="left")
        .with_columns(pl.col("spend_segment").fill_null(SPEND_SEGMENTS[0]))
        .sort("cardholder_id")
    )
    numeric = frame.select(
        pl.col("txns").log1p(),
        pl.col("spend").log1p(),
        pl.col("merchants").log1p(),
        (pl.col("spend") / pl.col("txns")).log1p().alias("log_avg_ticket"),
    )
    one_hot = frame.select(
        *(
            (pl.col("spend_segment") == segment).cast(pl.Float64).alias(f"segment={segment}")
            for segment in SPEND_SEGMENTS
        ),
        *(
            (pl.col("product") == product).cast(pl.Float64).alias(f"product={product}")
            for product in products
        ),
    )
    matrix = np.hstack([numeric.to_numpy(), one_hot.to_numpy()]).astype(np.float32)
    # Estandarizar solo el bloque continuo: las indicadoras ya viven en la misma escala.
    block = matrix[:, : numeric.width]
    matrix[:, : numeric.width] = (block - block.mean(axis=0)) / np.maximum(block.std(axis=0), 1e-6)
    return matrix, [*numeric.columns, *one_hot.columns]


def _build_data(
    split: LinkSplit,
    cardholder_matrix: np.ndarray,
    merchant_ids: np.ndarray,
    merchant_matrix: np.ndarray,
):
    """The ``HeteroData`` the encoder runs on, and the supervision edges held out of it."""
    import torch
    from torch_geometric.data import HeteroData

    cardholder_ids = np.sort(split.train["cardholder_id"].unique().to_numpy())
    pairs = (
        split.train.select("cardholder_id", "merchant_id")
        .unique()
        .sort(["cardholder_id", "merchant_id"])
    )
    inside = pairs.join(pl.DataFrame({"merchant_id": merchant_ids}), on="merchant_id", how="inner")
    rows = np.searchsorted(cardholder_ids, inside["cardholder_id"].to_numpy())
    columns = np.searchsorted(merchant_ids, inside["merchant_id"].to_numpy())

    data = HeteroData()
    data[CARDHOLDER].x = torch.from_numpy(cardholder_matrix)
    data[MERCHANT].x = torch.from_numpy(merchant_matrix.astype(np.float32))
    edges = torch.from_numpy(np.vstack([rows, columns]).astype(np.int64))
    data[CARDHOLDER, BUYS, MERCHANT].edge_index = edges
    return data, cardholder_ids, edges


def _link_loss(vectors, positive, negatives, negatives_per_positive: int):
    """Binary cross-entropy of real links against random merchants for the same cardholders."""
    import torch

    sources = positive[0].repeat(negatives_per_positive)
    cardholders, merchants = vectors[CARDHOLDER], vectors[MERCHANT]
    positive_score = (cardholders[positive[0]] * merchants[positive[1]]).sum(dim=-1)
    negative_score = (cardholders[sources] * merchants[negatives]).sum(dim=-1)
    scores = torch.cat([positive_score, negative_score])
    labels = torch.cat([torch.ones_like(positive_score), torch.zeros_like(negative_score)])
    return torch.nn.functional.binary_cross_entropy_with_logits(scores, labels)


def _encoder(cfg: ProjectConfig):
    """Two SAGE layers, turned into a heterogeneous encoder by ``to_hetero``."""
    import torch
    from torch_geometric.nn import SAGEConv, to_hetero

    settings = cfg.elasticity.link_prediction.gnn

    class Encoder(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layers = torch.nn.ModuleList(
                SAGEConv((-1, -1), settings.hidden_dims) for _ in range(settings.layers)
            )

        def forward(self, x, edge_index):
            for index, layer in enumerate(self.layers):
                x = layer(x, edge_index)
                if index < len(self.layers) - 1:
                    x = x.relu()
            return x

    metadata = (
        [CARDHOLDER, MERCHANT],
        [(CARDHOLDER, BUYS, MERCHANT), (MERCHANT, f"rev_{BUYS}", CARDHOLDER)],
    )
    return to_hetero(Encoder(), metadata, aggr="mean")


def fit_graphsage(
    split: LinkSplit,
    cardholders: pl.DataFrame,
    merchant_ids: np.ndarray,
    merchant_matrix: np.ndarray,
    cfg: ProjectConfig,
) -> GraphSageResult:
    """Train the encoder on the training graph and return the node vectors."""
    import torch
    import torch_geometric.transforms as T

    settings = cfg.elasticity.link_prediction.gnn
    logger = get_logger(__name__)
    cardholder_matrix, _ = cardholder_features(split, cardholders)

    with _deterministic(cfg):
        data, cardholder_ids, edges = _build_data(
            split, cardholder_matrix, merchant_ids, merchant_matrix
        )
        generator = torch.Generator().manual_seed(cfg.seed)

        # Las aristas de mensaje y las de supervisión son disjuntas: si el modelo ve como
        # entrada la misma arista que debe predecir, aprende a copiarla.
        order = torch.randperm(edges.size(1), generator=generator)
        cut = max(1, int(_MESSAGE_PASSING_SHARE * edges.size(1)))
        message, supervision = edges[:, order[:cut]], edges[:, order[cut:]]
        if supervision.size(1) == 0:
            raise ValueError("Not enough training edges to hold out supervision links")
        data[CARDHOLDER, BUYS, MERCHANT].edge_index = message
        data = T.ToUndirected()(data)

        # Tercera división, dentro de la supervisión: una parte ajusta y otra valida. El número
        # de épocas lo decide esta validación, nunca los meses reservados para evaluar.
        shuffled = supervision[:, torch.randperm(supervision.size(1), generator=generator)]
        held = max(1, int(_VALIDATION_SHARE * shuffled.size(1)))
        validation, fitting = shuffled[:, :held], shuffled[:, held:]
        if fitting.size(1) == 0:
            raise ValueError("Not enough supervision edges to hold out a validation set")

        n_merchants = len(merchant_ids)
        negatives_per_positive = settings.negatives_per_positive
        # Negativos de validación fijos: la pérdida de una época debe ser comparable con la de
        # la siguiente, así que no pueden cambiar entre épocas.
        validation_negatives = torch.randint(
            0, n_merchants, (negatives_per_positive * validation.size(1),), generator=generator
        )

        model = _encoder(cfg)
        # Las capas perezosas fijan sus dimensiones en la primera pasada: se hace antes de crear
        # el optimizador para que registre parámetros ya materializados.
        with torch.no_grad():
            model(data.x_dict, data.edge_index_dict)
        optimizer = torch.optim.Adam(model.parameters(), lr=settings.learning_rate)
        budget = min(settings.train_edges, fitting.size(1))

        losses: list[float] = []
        validation_losses: list[float] = []
        best_loss, best_epoch, best_state = float("inf"), 0, None
        for epoch in range(1, settings.max_epochs + 1):
            model.train()
            optimizer.zero_grad()
            vectors = model(data.x_dict, data.edge_index_dict)

            with torch.no_grad():
                detached = {kind: tensor.detach() for kind, tensor in vectors.items()}
                validation_loss = float(
                    _link_loss(detached, validation, validation_negatives, negatives_per_positive)
                )
            # Los vectores de esta pasada son los de los parámetros antes del paso: si la
            # validación mejoró, el mejor estado es el actual y se guarda antes de moverlo.
            if validation_loss < best_loss - settings.min_delta:
                best_loss, best_epoch = validation_loss, epoch
                best_state = copy.deepcopy(model.state_dict())

            picked = torch.randperm(fitting.size(1), generator=generator)[:budget]
            negatives = torch.randint(
                0, n_merchants, (negatives_per_positive * budget,), generator=generator
            )
            loss = _link_loss(vectors, fitting[:, picked], negatives, negatives_per_positive)
            loss.backward()
            optimizer.step()

            losses.append(float(loss.detach()))
            validation_losses.append(validation_loss)
            logger.debug(
                "epoch %d loss %.4f validation %.4f", epoch, losses[-1], validation_losses[-1]
            )
            if epoch - best_epoch >= settings.patience:
                break

        if best_state is not None:
            model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            final = model(data.x_dict, data.edge_index_dict)
        logger.info(
            "graphsage stopped after %d epochs; best validation loss %.4f at epoch %d",
            len(losses),
            best_loss,
            best_epoch,
        )
        return GraphSageResult(
            cardholder_ids=cardholder_ids,
            merchant_ids=merchant_ids,
            cardholder_vectors=final[CARDHOLDER].numpy().astype(np.float64),
            merchant_vectors=final[MERCHANT].numpy().astype(np.float64),
            losses=tuple(losses),
            validation_losses=tuple(validation_losses),
            best_epoch=best_epoch,
        )


def graphsage_scores(split: LinkSplit, result: GraphSageResult) -> np.ndarray:
    """Score every candidate for every evaluated cardholder with the trained vectors."""
    rows = np.searchsorted(result.cardholder_ids, split.cardholders)
    if not np.array_equal(result.cardholder_ids[rows], split.cardholders):
        raise ValueError("Evaluated cardholders are missing from the trained graph")
    columns = np.searchsorted(result.merchant_ids, split.candidates)
    if not np.array_equal(result.merchant_ids[columns], split.candidates):
        raise ValueError("Candidate merchants are missing from the trained graph")
    return result.cardholder_vectors[rows] @ result.merchant_vectors[columns].T
