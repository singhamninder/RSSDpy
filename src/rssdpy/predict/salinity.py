"""Spatial prediction of ln(ECe) and derived salinity statistics.

Theory basis: Lesch, Strauss, and Rhoades (1995) Papers 1 and 2 — Confirmed.
  https://doi.org/10.1029/94WR02179  (Paper 1: prediction framework)
  https://doi.org/10.1029/94WR02180  (Paper 2: RSSD + APVE)

Prediction at site j (Confirmed: Paper 1):
  ŷⱼ = b' xⱼ

Prediction variance (Confirmed: Paper 1):
  v²ⱼ = s² (1 + xⱼ' (X'X)⁻¹ xⱼ)

Conditional probability (Confirmed: Paper 1):
  P(a ≤ ECe ≤ b) via t-distribution with n - p - 1 df.

Field average (Confirmed: Paper 1):
  G = (n/N) ȳ_cal + ((N-n)/N) Ḡ_pred
"""

import logging
from dataclasses import dataclass, field

import numpy as np
from scipy.stats import t as t_dist

from rssdpy.calibration.mlr import (
    MLRResult,
    _build_pred_matrix,
)

logger = logging.getLogger(__name__)


@dataclass
class PredictionResult:
    """Salinity prediction results at all survey sites.

    Attributes
    ----------
    pred_ln_ece : np.ndarray
        Shape ``(n_pred,)`` — predicted ln(ECe) at prediction sites.
    pred_variance : np.ndarray
        Shape ``(n_pred,)`` — prediction variance v²ⱼ at prediction sites.
    pred_indices : np.ndarray
        Integer indices of prediction sites in the original survey array.
    cal_indices : np.ndarray
        Integer indices of calibration sites in the original survey array.
    ln_ece_cal : np.ndarray
        Observed ln(ECe) at calibration sites.
    field_mean : float
        Field-average ln(ECe) estimate (Confirmed: Paper 1).
    field_mean_se : float
        Approximate standard error of the field mean.
    n_sites : int
        Total number of survey sites N.
    """

    pred_ln_ece: np.ndarray
    pred_variance: np.ndarray
    pred_indices: np.ndarray
    cal_indices: np.ndarray
    ln_ece_cal: np.ndarray
    field_mean: float
    field_mean_se: float
    n_sites: int

    def probability_interval(
        self,
        a: float,
        b: float,
    ) -> np.ndarray:
        """Compute P(a ≤ ECe ≤ b) at each prediction site.

        Parameters
        ----------
        a : float
            Lower ECe bound (dS/m).  Pass ``0.0`` for a left-open interval.
        b : float
            Upper ECe bound (dS/m).  Pass ``float('inf')`` for open right end.

        Returns
        -------
        np.ndarray
            Shape ``(n_pred,)`` — probability values in [0, 1].

        Notes
        -----
        Uses posterior predictive t-distribution with ``n_cal - p - 1`` df,
        where p is inferred from the model used to generate predictions
        (Confirmed: Paper 1).
        """
        df = self._df
        s = np.sqrt(self.pred_variance)

        ln_a = np.log(a) if a > 0 else -np.inf
        ln_b = np.log(b) if np.isfinite(b) else np.inf

        # Standardise bounds relative to predicted mean
        t_a = (ln_a - self.pred_ln_ece) / s if np.isfinite(ln_a) else np.full(len(s), -np.inf)
        t_b = (ln_b - self.pred_ln_ece) / s if np.isfinite(ln_b) else np.full(len(s), np.inf)

        return np.clip(t_dist.cdf(t_b, df=df) - t_dist.cdf(t_a, df=df), 0.0, 1.0)

    def range_proportions(
        self,
        boundaries: list[float],
    ) -> np.ndarray:
        """Estimate fraction of field in each salinity class.

        Parameters
        ----------
        boundaries : list[float]
            Salinity class boundaries in dS/m.  E.g. ``[0, 2, 4, 8, 16]``
            produces four classes.  First element should be 0; last can be
            ``float('inf')``.

        Returns
        -------
        np.ndarray
            Shape ``(n_classes,)`` — estimated proportion of field in each
            interval.  Values sum to 1.

        Raises
        ------
        ValueError
            If ``boundaries`` has fewer than 2 elements.
        """
        if len(boundaries) < 2:
            raise ValueError("boundaries must have at least 2 elements.")

        n_classes = len(boundaries) - 1
        proportions = np.zeros(n_classes)
        for k in range(n_classes):
            probs = self.probability_interval(boundaries[k], boundaries[k + 1])
            proportions[k] = float(probs.mean())

        # Normalise to sum to 1
        total = proportions.sum()
        if total > 0:
            proportions /= total
        return proportions

    # Private: degrees of freedom stored after construction
    _df: int = field(default=10, init=False, repr=False)


def predict_salinity(
    model: MLRResult,
    scores: np.ndarray,
    coords: np.ndarray,
    cal_indices: np.ndarray,
    ln_ece: np.ndarray,
) -> PredictionResult:
    """Predict ln(ECe) at all non-calibration survey sites.

    Parameters
    ----------
    model : MLRResult
        Best fitted MLR model from :func:`~rssdpy.calibration.mlr.fit_mlr_models`.
    scores : np.ndarray
        Shape ``(N, n_components)`` — standardised PC scores for all survey sites.
    coords : np.ndarray
        Shape ``(N, 2)`` — projected geographic coordinates in metres.
    cal_indices : np.ndarray
        Integer indices of calibration sites.
    ln_ece : np.ndarray
        Shape ``(n_cal,)`` — observed ln(ECe) at calibration sites.

    Returns
    -------
    PredictionResult
        Predictions, variances, and field statistics.

    Raises
    ------
    ValueError
        If input dimensions are inconsistent.
    """
    scores = np.asarray(scores, dtype=float)
    coords = np.asarray(coords, dtype=float)
    cal_indices = np.asarray(cal_indices, dtype=int)
    ln_ece = np.asarray(ln_ece, dtype=float)

    n_all = len(scores)
    n_cal = len(cal_indices)

    if len(coords) != n_all:
        raise ValueError(f"scores has {n_all} rows but coords has {len(coords)} rows.")
    if n_cal != len(ln_ece):
        raise ValueError(f"cal_indices has {n_cal} entries but ln_ece has {len(ln_ece)}.")

    pred_mask = np.ones(n_all, dtype=bool)
    pred_mask[cal_indices] = False
    pred_indices = np.where(pred_mask)[0]
    n_pred = len(pred_indices)

    if n_pred == 0:
        raise ValueError("All survey sites are calibration sites; no prediction sites remain.")

    x_pred = _build_pred_matrix(scores, coords, pred_indices, model.pc_flags, model.trend_flags)

    ols = model.ols_result
    coeffs = np.asarray(ols.params)  # type: ignore[union-attr]

    # Point predictions: ŷⱼ = b' xⱼ (Confirmed: Paper 1)
    pred_ln_ece = x_pred @ coeffs

    # Prediction variance: v²ⱼ = s²(1 + xⱼ'(X'X)⁻¹xⱼ) (Confirmed: Paper 1)
    s2 = float(ols.mse_resid)  # type: ignore[union-attr]
    xtx_inv = np.asarray(ols.normalized_cov_params)  # type: ignore[union-attr]
    leverages = np.einsum("ij,jk,ik->i", x_pred, xtx_inv, x_pred)
    pred_variance = s2 * (1.0 + leverages)

    # Field-average estimate (Confirmed: Paper 1)
    # G = (n/N) ȳ_cal + ((N-n)/N) Ḡ_pred
    cal_mean = float(ln_ece.mean())
    pred_mean = float(pred_ln_ece.mean())
    n_total = n_all
    field_avg = (n_cal / n_total) * cal_mean + ((n_total - n_cal) / n_total) * pred_mean

    # Approximate SE of field mean (Derived)
    se_pred_mean = float(np.sqrt(pred_variance.mean() / n_pred))
    field_mean_se = ((n_total - n_cal) / n_total) * se_pred_mean

    df = n_cal - model.n_params

    result = PredictionResult(
        pred_ln_ece=pred_ln_ece,
        pred_variance=pred_variance,
        pred_indices=pred_indices,
        cal_indices=cal_indices,
        ln_ece_cal=ln_ece,
        field_mean=field_avg,
        field_mean_se=field_mean_se,
        n_sites=n_all,
    )
    # Attach degrees of freedom without re-defining dataclass (uses object attribute)
    object.__setattr__(result, "_df", max(1, df))
    return result
