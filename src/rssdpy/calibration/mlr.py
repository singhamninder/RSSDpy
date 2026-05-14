"""MLR calibration with PRESS and APVE model selection.

Theory basis: Lesch, Strauss, and Rhoades (1995) Paper 2 — Confirmed.
  https://doi.org/10.1029/94WR02180

Model structure (Confirmed: Paper 2, Lesch 2005):
  ln(ECe) = β₀ + β₁κ₁ + β₂κ₂ + β₃κ₁κ₂ + β₄κ₃ + trend + ε

PC variable subsets (5 total):
  0: κ₁
  1: κ₁, κ₂
  2: κ₁, κ₂, κ₁κ₂
  3: κ₁, κ₂, κ₃
  4: κ₁, κ₂, κ₁κ₂, κ₃

Trend surface subsets (10 total, one per combination of {x, y, xy, x², y²}
  that sum to 0–5 terms, but in practice the 10 subsets used in ESAP are the
  power-set of {x, y} × {xy, x², y²} filtered to sensible combinations):
  0: no trend
  1: x
  2: y
  3: x, y
  4: x, y, xy
  5: x, y, x²
  6: x, y, y²
  7: x, y, xy, x²
  8: x, y, xy, y²
  9: x, y, xy, x², y²

PRESS (Confirmed: Paper 2, citing Myers 1986):
  PRESS = Σ (eᵢ / (1 − hᵢ))²
  where hᵢ = xᵢ' (X'X)⁻¹ xᵢ (leverage).

APVE (Confirmed: Paper 2):
  APVE = s² × mean(1 + xⱼ' (X'X)⁻¹ xⱼ) over all N−n prediction sites.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from itertools import product as iproduct
from typing import Any

import numpy as np
import statsmodels.api as sm

logger = logging.getLogger(__name__)

# PC variable subsets as (use_kappa1, use_kappa2, use_interaction, use_kappa3).
# kappa1 is always included (intercept-normalised).
_PC_SUBSETS: list[tuple[bool, bool, bool, bool]] = [
    (True, False, False, False),  # κ₁ only
    (True, True, False, False),  # κ₁, κ₂
    (True, True, True, False),  # κ₁, κ₂, κ₁κ₂
    (True, True, False, True),  # κ₁, κ₂, κ₃
    (True, True, True, True),  # κ₁, κ₂, κ₁κ₂, κ₃
]

# Trend subsets as column flags: (use_x, use_y, use_xy, use_x2, use_y2)
_TREND_SUBSETS: list[tuple[bool, bool, bool, bool, bool]] = [
    (False, False, False, False, False),  # no trend
    (True, False, False, False, False),  # x
    (False, True, False, False, False),  # y
    (True, True, False, False, False),  # x, y
    (True, True, True, False, False),  # x, y, xy
    (True, True, False, True, False),  # x, y, x²
    (True, True, False, False, True),  # x, y, y²
    (True, True, True, True, False),  # x, y, xy, x²
    (True, True, True, False, True),  # x, y, xy, y²
    (True, True, True, True, True),  # x, y, xy, x², y²
]


@dataclass
class MLRResult:
    """Result for a single fitted MLR candidate model.

    Attributes
    ----------
    model_id : int
        Zero-based index into the 50 (or fewer) candidate models.
    pc_flags : tuple[bool, bool, bool, bool]
        PC variable inclusion flags: (κ₁, κ₂, κ₁κ₂, κ₃).
    trend_flags : tuple[bool, bool, bool, bool, bool]
        Trend variable flags: (x, y, xy, x², y²).
    ols_result : statsmodels RegressionResultsWrapper
        Full statsmodels OLS result.
    press : float
        Leave-one-out prediction sum of squares.
    apve : float
        Average prediction variance estimate over prediction sites.
    r2 : float
        R-squared.
    adj_r2 : float
        Adjusted R-squared.
    mse : float
        Mean squared error (residual variance estimate s²).
    n_params : int
        Number of model parameters including intercept.
    feature_names : list[str]
        Names of predictor columns in the model matrix.
    """

    model_id: int
    pc_flags: tuple[bool, bool, bool, bool]
    trend_flags: tuple[bool, bool, bool, bool, bool]
    ols_result: Any
    press: float
    apve: float
    r2: float
    adj_r2: float
    mse: float
    n_params: int
    feature_names: list[str] = field(default_factory=list)


def _build_model_matrix(
    scores: np.ndarray,
    coords: np.ndarray,
    cal_indices: np.ndarray,
    pc_flags: tuple[bool, bool, bool, bool],
    trend_flags: tuple[bool, bool, bool, bool, bool],
) -> tuple[np.ndarray, list[str]]:
    """Assemble the design matrix X for calibration sites.

    Returns
    -------
    X : np.ndarray
        Shape ``(n_cal, p)``, including a leading intercept column of ones.
    names : list[str]
        Column names corresponding to X.
    """
    n_components = scores.shape[1]
    use_k1, use_k2, use_inter, use_k3 = pc_flags
    use_x, use_y, use_xy, use_x2, use_y2 = trend_flags

    cols: list[np.ndarray] = [np.ones(len(cal_indices))]
    names: list[str] = ["intercept"]

    s = scores[cal_indices]
    c = coords[cal_indices]

    if use_k1 and n_components >= 1:
        cols.append(s[:, 0])
        names.append("k1")
    if use_k2 and n_components >= 2:
        cols.append(s[:, 1])
        names.append("k2")
    if use_inter and n_components >= 2:
        cols.append(s[:, 0] * s[:, 1])
        names.append("k1k2")
    if use_k3 and n_components >= 3:
        cols.append(s[:, 2])
        names.append("k3")

    if use_x:
        cols.append(c[:, 0])
        names.append("x")
    if use_y:
        cols.append(c[:, 1])
        names.append("y")
    if use_xy:
        cols.append(c[:, 0] * c[:, 1])
        names.append("xy")
    if use_x2:
        cols.append(c[:, 0] ** 2)
        names.append("x2")
    if use_y2:
        cols.append(c[:, 1] ** 2)
        names.append("y2")

    return np.column_stack(cols), names


def _build_pred_matrix(
    scores: np.ndarray,
    coords: np.ndarray,
    pred_indices: np.ndarray,
    pc_flags: tuple[bool, bool, bool, bool],
    trend_flags: tuple[bool, bool, bool, bool, bool],
) -> np.ndarray:
    """Assemble the design matrix for prediction sites."""
    n_components = scores.shape[1]
    use_k1, use_k2, use_inter, use_k3 = pc_flags
    use_x, use_y, use_xy, use_x2, use_y2 = trend_flags

    cols: list[np.ndarray] = [np.ones(len(pred_indices))]
    s = scores[pred_indices]
    c = coords[pred_indices]

    if use_k1 and n_components >= 1:
        cols.append(s[:, 0])
    if use_k2 and n_components >= 2:
        cols.append(s[:, 1])
    if use_inter and n_components >= 2:
        cols.append(s[:, 0] * s[:, 1])
    if use_k3 and n_components >= 3:
        cols.append(s[:, 2])
    if use_x:
        cols.append(c[:, 0])
    if use_y:
        cols.append(c[:, 1])
    if use_xy:
        cols.append(c[:, 0] * c[:, 1])
    if use_x2:
        cols.append(c[:, 0] ** 2)
    if use_y2:
        cols.append(c[:, 1] ** 2)

    return np.column_stack(cols)


def _compute_press(ols_result: Any) -> float:
    """Compute PRESS from an OLS result via leverage.

    PRESS = Σ (eᵢ / (1 − hᵢ))²  (Confirmed: Paper 2, citing Myers 1986)
    """
    residuals = np.asarray(ols_result.resid)  # type: ignore[union-attr]
    influence = ols_result.get_influence()  # type: ignore[union-attr]
    hat = np.asarray(influence.hat_matrix_diag)
    # Avoid division by zero if any leverage equals 1
    safe_denom = np.where(hat >= 1.0 - 1e-12, np.nan, 1.0 - hat)
    loo_errors = residuals / safe_denom
    return float(np.nansum(loo_errors**2))


def _compute_apve(
    ols_result: Any,
    x_pred: np.ndarray,
) -> float:
    """Compute APVE over prediction sites.

    APVE = s² × mean(1 + xⱼ' (X'X)⁻¹ xⱼ)  (Confirmed: Paper 2)
    """
    s2 = float(ols_result.mse_resid)  # type: ignore[union-attr]
    xtx_inv = np.asarray(ols_result.normalized_cov_params)  # (X'X)⁻¹
    # Leverage-like term for each prediction site
    leverages = np.einsum("ij,jk,ik->i", x_pred, xtx_inv, x_pred)
    return float(s2 * np.mean(1.0 + leverages))


def fit_mlr_models(
    scores: np.ndarray,
    coords: np.ndarray,
    cal_indices: np.ndarray,
    ln_ece: np.ndarray,
    n_components: int | None = None,
) -> list[MLRResult]:
    """Fit all candidate MLR models and rank by PRESS then APVE.

    Parameters
    ----------
    scores : np.ndarray
        Shape ``(N, n_components)`` — standardised PC scores for all survey
        sites (calibration + prediction).
    coords : np.ndarray
        Shape ``(N, 2)`` — projected geographic coordinates in metres.
    cal_indices : np.ndarray
        Integer indices into ``scores``/``coords`` for the calibration sites.
    ln_ece : np.ndarray
        Shape ``(n_cal,)`` — natural log of ECe at calibration sites (dS/m).
    n_components : int | None
        Number of PC components available.  If ``None``, inferred from
        ``scores.shape[1]``.

    Returns
    -------
    list[MLRResult]
        Candidate models sorted by PRESS (ascending), with APVE as a
        tiebreaker.  Length ≤ 50 (fewer if ``n_components < 3``).

    Raises
    ------
    ValueError
        If ``len(cal_indices) != len(ln_ece)`` or any calibration index is
        out of range.

    Notes
    -----
    Models with fewer parameters than available calibration sites are
    skipped — statsmodels will raise a perfect-fit or rank-deficiency error
    otherwise.
    """
    scores = np.asarray(scores, dtype=float)
    coords = np.asarray(coords, dtype=float)
    cal_indices = np.asarray(cal_indices, dtype=int)
    ln_ece = np.asarray(ln_ece, dtype=float)

    if len(cal_indices) != len(ln_ece):
        raise ValueError(
            f"cal_indices has {len(cal_indices)} entries but ln_ece has {len(ln_ece)}."
        )
    n_cal = len(cal_indices)
    n_all = len(scores)
    if cal_indices.min() < 0 or cal_indices.max() >= n_all:
        raise ValueError("cal_indices contains out-of-range index values.")

    if n_components is None:
        n_components = scores.shape[1]

    pred_mask = np.ones(n_all, dtype=bool)
    pred_mask[cal_indices] = False
    pred_indices = np.where(pred_mask)[0]
    has_pred = len(pred_indices) > 0

    results: list[MLRResult] = []
    model_id = 0

    for pc_flags, trend_flags in iproduct(_PC_SUBSETS, _TREND_SUBSETS):
        x_cal, names = _build_model_matrix(scores, coords, cal_indices, pc_flags, trend_flags)
        n_params = x_cal.shape[1]

        # Skip under-determined models
        if n_params >= n_cal:
            model_id += 1
            continue

        try:
            ols = sm.OLS(ln_ece, x_cal).fit()
        except Exception as exc:
            logger.debug("Model %d failed to fit: %s", model_id, exc)
            model_id += 1
            continue

        press = _compute_press(ols)

        if has_pred:
            x_pred = _build_pred_matrix(scores, coords, pred_indices, pc_flags, trend_flags)
            apve = _compute_apve(ols, x_pred)
        else:
            apve = float("nan")

        results.append(
            MLRResult(
                model_id=model_id,
                pc_flags=pc_flags,
                trend_flags=trend_flags,
                ols_result=ols,
                press=press,
                apve=apve,
                r2=float(ols.rsquared),
                adj_r2=float(ols.rsquared_adj),
                mse=float(ols.mse_resid),
                n_params=n_params,
                feature_names=names,
            )
        )
        model_id += 1

    if not results:
        raise ValueError(
            "No valid MLR models could be fitted.  "
            "Increase the number of calibration sites or reduce n_components."
        )

    # Rank by PRESS ascending, APVE as tiebreaker (Confirmed: Paper 2)
    results.sort(key=lambda r: (r.press, r.apve if not np.isnan(r.apve) else float("inf")))
    logger.info("fit_mlr_models: %d models fitted; best PRESS=%.4f", len(results), results[0].press)
    return results
