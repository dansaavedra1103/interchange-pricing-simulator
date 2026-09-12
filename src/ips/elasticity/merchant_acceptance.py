"""Merchant acceptance: the probability of dropping the card, given what it costs to take it.

    P(abandono) = sigma( alpha + beta * s_g * carga_relativa + modificadores )

The driver is the **relative burden** (MDR over the merchant's sector margin, spec §2.4): what
fraction of its margin the merchant hands over. ``s_g`` is the group's sensitivity from
``mcc_groups.yaml``; ``alpha`` and ``beta`` are the two numbers
:mod:`ips.elasticity.calibration` solves for.

Nothing here is estimated. The synthetic data contains no abandonment events — no merchant
ever stopped accepting — so this curve is a **structural assumption** anchored to two figures
of our own (``config/elasticity.yaml``), and every modifier carries the sign the spec implies.
Treating its output as an empirical finding would be wrong.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

from ips.elasticity.params import sensitivity_column
from ips.utils.config import ProjectConfig

BURDEN = "relative_burden"
REQUIRED_COLUMNS = (
    "merchant_id",
    "mcc_group",
    BURDEN,
    "cnp_share",
    "avg_ticket_cop",
    "pricing_model",
    "share_debit",
    "gdv_cop",
)


@dataclass(frozen=True)
class AcceptanceCurve:
    """The two calibrated parameters of the logistic, plus the moments they achieved."""

    alpha: float
    beta: float
    achieved_abandonment: float
    achieved_semi_elasticity: float
    target_abandonment: float
    target_semi_elasticity: float

    def frame(self) -> pl.DataFrame:
        """One row: what was solved for and what came out, for the artifact and the notebook."""
        return pl.DataFrame(
            {
                "alpha": [self.alpha],
                "beta": [self.beta],
                "target_abandonment": [self.target_abandonment],
                "achieved_abandonment": [self.achieved_abandonment],
                "target_semi_elasticity": [self.target_semi_elasticity],
                "achieved_semi_elasticity": [self.achieved_semi_elasticity],
            }
        )


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Logistic function, written so large negative inputs do not overflow."""
    out = np.empty_like(x, dtype=np.float64)
    positive = x >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-x[positive]))
    exponential = np.exp(x[~positive])
    out[~positive] = exponential / (1.0 + exponential)
    return out


def _check_columns(profile: pl.DataFrame) -> None:
    missing = [c for c in REQUIRED_COLUMNS if c not in profile.columns]
    if missing:
        raise ValueError(f"profile is missing required columns: {missing}")


def burden_term(profile: pl.DataFrame, cfg: ProjectConfig) -> np.ndarray:
    """What ``beta`` multiplies: the relative burden weighted by the group's sensitivity.

    El modelo de precio entra aquí y no en el desplazamiento: un comercio en IC++ ve el cambio
    de interchange directo, así que reacciona más a la misma carga (spec §1.5).
    """
    _check_columns(profile)
    icpp = cfg.elasticity.acceptance.modifiers.icpp_sensitivity
    # `mixed` queda a mitad de camino: parte de su volumen se cotiza en cada modelo.
    pricing = {"blended": 1.0, "icpp": icpp, "mixed": (1.0 + icpp) / 2.0}
    weighted = profile.select(
        (
            pl.col(BURDEN)
            * sensitivity_column(cfg)
            * pl.col("pricing_model").cast(pl.Utf8).replace_strict(pricing, return_dtype=pl.Float64)
        ).alias("term")
    )
    return weighted["term"].to_numpy().astype(np.float64)


def logit_offset(profile: pl.DataFrame, cfg: ProjectConfig) -> np.ndarray:
    """The additive part of the logit: channel, surcharge regime and instant-payment pressure."""
    _check_columns(profile)
    acceptance = cfg.elasticity.acceptance
    modifiers = acceptance.modifiers
    instant = acceptance.instant_payments

    offset = pl.col("cnp_share") * modifiers.cnp_share
    if acceptance.surcharge_allowed:
        # Donde se permite el recargo, el comercio traslada el costo en vez de rechazar la
        # tarjeta: la misma carga produce menos abandono (spec §1.8).
        offset = offset + modifiers.surcharge_allowed_shift
    if instant.enabled:
        # Los pagos instantáneos sustituyen sobre todo débito de ticket bajo (spec §1.9): el
        # empuje crece con la parte del volumen del comercio que es sustituible.
        small_ticket = (1.0 - pl.col("avg_ticket_cop") / instant.small_ticket_cop).clip(0.0, 1.0)
        offset = offset + instant.max_shift * pl.col("share_debit") * small_ticket
    return profile.select(offset.alias("offset"))["offset"].to_numpy().astype(np.float64)


def probability_from_terms(
    term: np.ndarray, offset: np.ndarray, alpha: float, beta: float, cfg: ProjectConfig
) -> np.ndarray:
    """Abandonment probability from the pre-computed logit pieces, clipped to its bounds."""
    acceptance = cfg.elasticity.acceptance
    probability = sigmoid(alpha + beta * term + offset)
    return np.clip(probability, acceptance.min_probability, acceptance.max_probability)


def abandonment_probability(
    profile: pl.DataFrame, cfg: ProjectConfig, curve: AcceptanceCurve
) -> pl.DataFrame:
    """Per merchant: its burden, its probability of dropping the card and the volume at risk."""
    term = burden_term(profile, cfg)
    offset = logit_offset(profile, cfg)
    probability = probability_from_terms(term, offset, curve.alpha, curve.beta, cfg)
    return profile.select(
        "merchant_id",
        "mcc_group",
        "size_tier",
        "pricing_model",
        BURDEN,
        "gdv_cop",
        pl.Series("p_abandonment", probability),
    ).with_columns((pl.col("p_abandonment") * pl.col("gdv_cop")).alias("gdv_at_risk_cop"))


def acceptance_response(
    profile: pl.DataFrame, cfg: ProjectConfig, curve: AcceptanceCurve, mdr_change: float
) -> pl.DataFrame:
    """The curve's whole point: what a proportional change in MDR does to acceptance.

    ``mdr_change`` is relative (0.10 = the merchant pays 10 % more), so it scales the burden.
    """
    if mdr_change <= -1.0:
        raise ValueError("mdr_change must be greater than -1 (the MDR cannot go negative)")
    term = burden_term(profile, cfg)
    offset = logit_offset(profile, cfg)
    base = probability_from_terms(term, offset, curve.alpha, curve.beta, cfg)
    shocked = probability_from_terms(
        term * (1.0 + mdr_change), offset, curve.alpha, curve.beta, cfg
    )
    return profile.select(
        "merchant_id",
        "gdv_cop",
        pl.Series("p_abandonment_base", base),
        pl.Series("p_abandonment_after", shocked),
    ).with_columns(
        (pl.col("p_abandonment_after") - pl.col("p_abandonment_base")).alias("p_abandonment_delta")
    )
