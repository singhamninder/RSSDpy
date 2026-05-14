"""Residual diagnostics: Shapiro-Wilk, Moran's I, and lack-of-fit F test.

Theory basis:
  - Shapiro-Wilk and Moran's I — Confirmed: Lesch et al. 1995 Papers 1 and 2.
  - Lack-of-fit F test with Moran correction — Confirmed: Paper 2.
    Requires duplicate soil cores at ≥ 4–6 calibration sites.

Moran's I proximity matrix W: inverse-distance-squared, row-normalised
(Confirmed: Paper 1).

Lack-of-fit F statistic (Confirmed: Paper 2):
  F = [m / (m - p - 1)] × (1 + I_M) / (1 - I_M)
  where m = number of calibration sites, p = number of predictors,
  I_M = modified Moran's I on residuals.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import shapiro

logger = logging.getLogger(__name__)


@dataclass
class DiagnosticsResult:
    """Residual diagnostic statistics for a fitted MLR model.

    Attributes
    ----------
    shapiro_stat : float
        Shapiro-Wilk W statistic.
    shapiro_pvalue : float
        Shapiro-Wilk p-value.  p < 0.05 suggests non-normality.
    moran_i : float
        Moran's I statistic on residuals.
    moran_z : float
        Standardised Moran score (compared to standard normal).
    moran_pvalue : float
        Two-sided p-value for Moran's I under normality approximation.
    lof_f : float | None
        Lack-of-fit F statistic.  ``None`` if no duplicate data provided.
    lof_pvalue : float | None
        Approximate p-value for the lack-of-fit test.  ``None`` if no
        duplicate data provided.
    """

    shapiro_stat: float
    shapiro_pvalue: float
    moran_i: float
    moran_z: float
    moran_pvalue: float
    lof_f: float | None = None
    lof_pvalue: float | None = None


def _moran_proximity_matrix(coords: np.ndarray) -> np.ndarray:
    """Compute row-normalised inverse-distance-squared proximity matrix.

    Parameters
    ----------
    coords : np.ndarray
        Shape ``(n, 2)`` — calibration site coordinates.

    Returns
    -------
    np.ndarray
        Shape ``(n, n)`` — row-normalised weight matrix (diagonal = 0).
    """
    dist = cdist(coords, coords, metric="euclidean")
    # Avoid division by zero on diagonal
    with np.errstate(divide="ignore", invalid="ignore"):
        w_mat = np.where(dist == 0, 0.0, 1.0 / dist**2)
    np.fill_diagonal(w_mat, 0.0)
    # Row-normalise
    row_sums = w_mat.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1.0, row_sums)
    return w_mat / row_sums


def _compute_moran_i(
    residuals: np.ndarray,
    w_mat: np.ndarray,
) -> tuple[float, float, float]:
    """Compute Moran's I, standardised z, and two-sided p-value.

    Parameters
    ----------
    residuals : np.ndarray
        Shape ``(n,)``.
    w_mat : np.ndarray
        Shape ``(n, n)`` — row-normalised proximity matrix.

    Returns
    -------
    moran_stat, z_score, p_value : float, float, float
    """
    from scipy.stats import norm

    n = len(residuals)
    e = residuals - residuals.mean()
    numerator = float(e @ w_mat @ e)
    denominator = float(e @ e)
    if denominator == 0:
        return 0.0, 0.0, 1.0

    # Moran's I statistic (Confirmed: Paper 1 & 2)
    moran_stat = (n / w_mat.sum()) * (numerator / denominator)

    # Expected value and variance under normality (standard approximation)
    e_moran = -1.0 / (n - 1)
    s1 = 0.5 * np.sum((w_mat + w_mat.T) ** 2)
    s2 = np.sum((w_mat.sum(axis=1) + w_mat.sum(axis=0)) ** 2)
    s0 = w_mat.sum()
    n_sq = n * n
    var_moran = (n * (n_sq - 3 * n + 3) * s1 - n * s2 + 3 * s0**2) / (
        (n - 1) * (n_sq - n) * s0**2
    ) - e_moran**2

    if var_moran <= 0:
        return float(moran_stat), 0.0, 1.0

    z = (moran_stat - e_moran) / np.sqrt(var_moran)
    p = float(2 * norm.sf(abs(z)))
    return float(moran_stat), float(z), p


def run_diagnostics(
    residuals: np.ndarray,
    coords: np.ndarray,
    n_params: int,
    duplicate_residuals: np.ndarray | None = None,
) -> DiagnosticsResult:
    """Run residual diagnostic tests on a fitted MLR model.

    Parameters
    ----------
    residuals : np.ndarray
        Shape ``(n_cal,)`` — OLS residuals at calibration sites.
    coords : np.ndarray
        Shape ``(n_cal, 2)`` — calibration site coordinates in metres.
    n_params : int
        Number of model parameters including intercept.
    duplicate_residuals : np.ndarray | None
        Optional.  Shape ``(n_dup, 2)`` — paired residuals at duplicate-core
        sites, where each row is ``[resid_primary, resid_duplicate]``.
        Required for the lack-of-fit F test.  See source notes Section 8.

    Returns
    -------
    DiagnosticsResult
        Populated diagnostic statistics.

    Raises
    ------
    ValueError
        If ``residuals`` and ``coords`` have incompatible lengths.
    """
    residuals = np.asarray(residuals, dtype=float)
    coords = np.asarray(coords, dtype=float)

    if len(residuals) != len(coords):
        raise ValueError(
            f"residuals has {len(residuals)} entries but coords has {len(coords)} rows."
        )

    # Shapiro-Wilk normality test (Confirmed: Paper 2)
    sw_stat, sw_p = shapiro(residuals)

    # Moran's I (Confirmed: Paper 1 & 2)
    w_mat = _moran_proximity_matrix(coords)
    moran_i, moran_z, moran_p = _compute_moran_i(residuals, w_mat)

    # Lack-of-fit F test (Confirmed: Paper 2); requires duplicate core data
    lof_f: float | None = None
    lof_p: float | None = None
    if duplicate_residuals is not None:
        from scipy.stats import f as f_dist

        dup = np.asarray(duplicate_residuals, dtype=float)
        if dup.ndim == 2 and dup.shape[1] == 2:
            m = len(residuals)
            p = n_params - 1  # exclude intercept for df calculation
            if m > p + 1 and abs(moran_i) < 1.0 - 1e-12:
                lof_f = float((m / (m - p - 1)) * (1 + moran_i) / (1 - moran_i))
                df1 = m - p - 1
                df2 = dup.shape[0]
                lof_p = float(f_dist.sf(lof_f, df1, df2))
            else:
                logger.warning("Lack-of-fit test skipped: insufficient df or |I_M| ≈ 1.")

    return DiagnosticsResult(
        shapiro_stat=float(sw_stat),
        shapiro_pvalue=float(sw_p),
        moran_i=moran_i,
        moran_z=moran_z,
        moran_pvalue=moran_p,
        lof_f=lof_f,
        lof_pvalue=lof_p,
    )
