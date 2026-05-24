"""PC-space outlier detection using Mahalanobis-equivalent distance.

Theory basis: Lesch, Strauss, and Rhoades (1995) Paper 2 — Confirmed.
  ESAP-RSSD manual (σ thresholds 3.5 / 4.5) — Confirmed for ESAP workflows.

Because PC scores are standardised (mean 0, variance 1), the sum of squared
scores for a site, ``d² = κ₁² + κ₂² + … + κₚ²``, follows a chi-squared
distribution with df = n_components under multivariate normality (chi-square
mode).  ESAP uses fixed σ cutoffs on the same distance in decorrelated space.

See also: ``docs/theory/rssd-source-notes.md`` — Section 3.
"""

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import chi2

from rssdpy.features.pca import ECaPCA

logger = logging.getLogger(__name__)

ESAP_DEFAULT_MASKING_STD: float = 3.5
ESAP_DEFAULT_OUTLIER_STD: float = 4.5


@dataclass(frozen=True)
class ESAPQCResult:
    """ESAP-style signal validation result in standardised PC space.

    Attributes
    ----------
    distances : np.ndarray
        PC-space distance ``sqrt(Σ κ²)`` per site.
    outlier_mask : np.ndarray
        Sites to remove from the survey (distance > ``outlier_std``).
    masking_mask : np.ndarray
        Sites to exclude from sampling (distance > ``masking_std``).
    eligible_mask : np.ndarray
        Sites that may be selected for RSSD (not masked and not outlier).
    masking_std : float
        Masking threshold used.
    outlier_std : float
        Outlier threshold used.
    """

    distances: np.ndarray
    outlier_mask: np.ndarray
    masking_mask: np.ndarray
    eligible_mask: np.ndarray
    masking_std: float
    outlier_std: float


def pc_distances(scores: np.ndarray) -> np.ndarray:
    """Compute PC-space distance for each site.

    Parameters
    ----------
    scores : np.ndarray
        Shape ``(N, n_components)`` — standardised PC scores.

    Returns
    -------
    np.ndarray
        Distance ``sqrt(Σ κ²)`` per site, shape ``(N,)``.

    Raises
    ------
    ValueError
        If ``scores`` is not 2-D.
    """
    scores = np.asarray(scores, dtype=float)
    if scores.ndim != 2:
        raise ValueError(f"scores must be 2-D, got shape {scores.shape}")
    return np.sqrt(np.sum(scores**2, axis=1))


def outlier_threshold(n_components: int, alpha: float = 0.001) -> float:
    """Compute the PC-distance threshold for chi-square outlier detection.

    The threshold is ``sqrt(chi2.ppf(1 - alpha, df=n_components))``.

    Parameters
    ----------
    n_components : int
        Number of PC dimensions (= df for chi-squared test).
    alpha : float
        Significance level.  Sites with p-value < alpha are flagged.

    Returns
    -------
    float
        Distance threshold value.

    Raises
    ------
    ValueError
        If ``n_components < 1`` or ``alpha`` is outside ``(0, 1)``.
    """
    if n_components < 1:
        raise ValueError(f"n_components must be ≥ 1, got {n_components}")
    if not (0 < alpha < 1):
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")

    return float(np.sqrt(chi2.ppf(1 - alpha, df=n_components)))


def detect_outliers(
    scores: np.ndarray,
    alpha: float = 0.001,
) -> tuple[np.ndarray, np.ndarray]:
    """Flag sites whose PC-space distance exceeds the chi-squared threshold.

    Parameters
    ----------
    scores : np.ndarray
        Shape ``(N, n_components)`` — standardised PC scores (mean 0, variance 1).
    alpha : float
        Significance level for the chi-squared distance test.

    Returns
    -------
    mask : np.ndarray
        Boolean array of shape ``(N,)``.  ``True`` indicates an outlier.
    distances : np.ndarray
        Float array of shape ``(N,)`` — PC-space distance for every site.

    Raises
    ------
    ValueError
        If ``scores`` is not a 2-D array.
    """
    scores = np.asarray(scores, dtype=float)
    if scores.ndim != 2:
        raise ValueError(f"scores must be 2-D, got shape {scores.shape}")

    distances = pc_distances(scores)
    threshold = outlier_threshold(scores.shape[1], alpha)
    mask = distances > threshold
    n_flagged = int(mask.sum())
    if n_flagged:
        logger.info(
            "Chi-square outlier detection (alpha=%.4f, threshold=%.3f): flagged %d/%d sites.",
            alpha,
            threshold,
            n_flagged,
            len(distances),
        )
    return mask, distances


def detect_outliers_esap(
    scores: np.ndarray,
    *,
    masking_std: float = ESAP_DEFAULT_MASKING_STD,
    outlier_std: float = ESAP_DEFAULT_OUTLIER_STD,
) -> ESAPQCResult:
    """Flag masked and outlier sites using ESAP σ thresholds on PC distance.

    Parameters
    ----------
    scores : np.ndarray
        Shape ``(N, n_components)`` — standardised PC scores.
    masking_std : float
        Sites with distance above this value are masked from sampling.
        ESAP default 3.5.
    outlier_std : float
        Sites with distance above this value should be removed from the survey.
        ESAP default 4.5.

    Returns
    -------
    ESAPQCResult
        Distance array plus outlier, masking, and eligible masks.

    Raises
    ------
    ValueError
        If thresholds are non-positive or ``outlier_std < masking_std``.
    """
    if masking_std <= 0 or outlier_std <= 0:
        raise ValueError("masking_std and outlier_std must be positive.")
    if outlier_std < masking_std:
        raise ValueError("outlier_std must be ≥ masking_std.")

    distances = pc_distances(scores)
    masking_mask = distances > masking_std
    outlier_mask = distances > outlier_std
    eligible_mask = ~(masking_mask | outlier_mask)

    logger.info(
        "ESAP σ QC (mask=%.2f, outlier=%.2f): %d masked, %d outliers, %d eligible.",
        masking_std,
        outlier_std,
        int(masking_mask.sum()),
        int(outlier_mask.sum()),
        int(eligible_mask.sum()),
    )
    return ESAPQCResult(
        distances=distances,
        outlier_mask=outlier_mask,
        masking_mask=masking_mask,
        eligible_mask=eligible_mask,
        masking_std=masking_std,
        outlier_std=outlier_std,
    )


def iterative_esap_validation(
    eca: pd.DataFrame,
    pca: ECaPCA,
    *,
    masking_std: float = ESAP_DEFAULT_MASKING_STD,
    outlier_std: float = ESAP_DEFAULT_OUTLIER_STD,
    max_iterations: int = 10,
) -> tuple[pd.DataFrame, np.ndarray, ESAPQCResult, np.ndarray]:
    """Iteratively decorrelate, drop ESAP outliers, and re-fit PCA.

    Matches the ESAP practice of deleting outlier sites and re-running
    decorrelation until no outliers remain.

    Parameters
    ----------
    eca : pd.DataFrame
        Raw ECa survey (positive dS/m).
    pca : ECaPCA
        PCA configuration (not pre-fitted).
    masking_std : float
        ESAP masking threshold.
    outlier_std : float
        ESAP outlier deletion threshold.
    max_iterations : int
        Maximum delete-and-refit cycles.

    Returns
    -------
    eca_clean : pd.DataFrame
        ECa table after outlier sites are removed.
    scores : np.ndarray
        Standardised PC scores for ``eca_clean``.
    qc : ESAPQCResult
        Final QC result on ``scores``.
    original_indices : np.ndarray
        Row indices into the input ``eca`` for each row of ``eca_clean``.

    Raises
    ------
    ValueError
        If no sites remain after outlier removal.
    RuntimeError
        If ``max_iterations`` is exceeded without convergence.
    """
    keep_mask = np.ones(len(eca), dtype=bool)

    for iteration in range(max_iterations):
        if not keep_mask.any():
            raise ValueError("All survey sites were removed as ESAP outliers.")
        eca_work = eca.loc[keep_mask].copy()
        original_indices = np.flatnonzero(keep_mask)
        scores, _, _ = pca.fit_transform(eca_work)
        qc = detect_outliers_esap(scores, masking_std=masking_std, outlier_std=outlier_std)
        if not qc.outlier_mask.any():
            return eca_work, scores, qc, original_indices
        outlier_global = original_indices[qc.outlier_mask]
        keep_mask[outlier_global] = False
        logger.info(
            "ESAP iterative validation iteration %d: removed %d outliers.",
            iteration + 1,
            len(outlier_global),
        )

    raise RuntimeError(f"ESAP validation did not converge within {max_iterations} iterations.")
