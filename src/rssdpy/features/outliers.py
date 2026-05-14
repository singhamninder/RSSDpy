"""PC-space outlier detection using Mahalanobis-equivalent distance.

Theory basis: Lesch, Strauss, and Rhoades (1995) Paper 2 — Confirmed.
  https://doi.org/10.1029/94WR02180

Because PC scores are standardised (mean 0, variance 1), the sum of squared
scores for a site, ``d² = κ₁² + κ₂² + … + κₚ²``, follows a chi-squared
distribution with df = n_components under multivariate normality.  Sites are
flagged when their distance ``d = sqrt(d²)`` exceeds ``sqrt(chi2.ppf(1-alpha,
df=n_components))``.

See also: ``docs/theory/rssd-source-notes.md`` — Section 3.
"""

from __future__ import annotations

import logging

import numpy as np
from scipy.stats import chi2

logger = logging.getLogger(__name__)


def outlier_threshold(n_components: int, alpha: float = 0.001) -> float:
    """Compute the PC-distance threshold for outlier detection.

    The threshold is ``sqrt(chi2.ppf(1 - alpha, df=n_components))``.

    For 3 components at alpha=0.001:
      ``sqrt(chi2.ppf(0.999, df=3)) ≈ 4.03``  (Confirmed: Paper 2 description)

    Parameters
    ----------
    n_components : int
        Number of PC dimensions (= df for chi-squared test).
    alpha : float
        Significance level.  Sites with p-value < alpha are flagged.
        Default 0.001 matches the ESAP convention described in Paper 2.

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

    # Derived from confirmed claim: standardised PC scores ∼ chi-squared(df=n_components)
    return float(np.sqrt(chi2.ppf(1 - alpha, df=n_components)))


def detect_outliers(
    scores: np.ndarray,
    alpha: float = 0.001,
) -> tuple[np.ndarray, np.ndarray]:
    """Flag survey sites whose PC-space distance exceeds the chi-squared threshold.

    Parameters
    ----------
    scores : np.ndarray
        Shape ``(N, n_components)`` — standardised PC scores (mean 0, variance 1).
    alpha : float
        Significance level for the chi-squared distance test.  Default 0.001
        matches ESAP behaviour (flags the most extreme 0.1% of observations).

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

    Notes
    -----
    Distance is computed in standardised PC space:
    ``d_i = sqrt(κ_{i,1}² + κ_{i,2}² + … + κ_{i,p}²)``.

    Examples
    --------
    >>> import numpy as np
    >>> scores = np.array([[0.5, 0.3, 0.1], [5.0, 3.0, 2.0]])
    >>> mask, dist = detect_outliers(scores)
    >>> mask  # second site is far out
    array([False,  True])
    """
    scores = np.asarray(scores, dtype=float)
    if scores.ndim != 2:
        raise ValueError(f"scores must be 2-D, got shape {scores.shape}")

    n_components = scores.shape[1]
    distances = np.sqrt(np.sum(scores**2, axis=1))
    threshold = outlier_threshold(n_components, alpha)

    mask = distances > threshold
    n_flagged = int(mask.sum())
    if n_flagged:
        logger.info(
            "Outlier detection (alpha=%.4f, threshold=%.3f): flagged %d/%d sites.",
            alpha,
            threshold,
            n_flagged,
            len(distances),
        )
    return mask, distances
