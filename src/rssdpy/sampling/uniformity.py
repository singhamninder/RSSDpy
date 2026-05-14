"""Spatial uniformity metric for calibration site sets.

Theory basis: Lesch, Strauss, and Rhoades (1995) Paper 2 — Confirmed.
  https://doi.org/10.1029/94WR02180

The Average Distance (AD) is:

    AD(β) = (1/N) Σᵢ min_{j ∈ β} d(site_i, site_j)

where ``d`` is Euclidean distance in projected geographic space (metres).
AD is minimised when calibration sites form an equilateral triangular grid
(McBratney et al. 1981, cited in Paper 2).
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import KDTree


def average_distance(
    coords: np.ndarray,
    cal_indices: np.ndarray,
) -> float:
    """Compute the average nearest-calibration distance AD(β).

    For each of the N survey sites, finds the nearest calibration site in β
    and returns the mean of those distances.

    Parameters
    ----------
    coords : np.ndarray
        Shape ``(N, 2)`` — projected geographic coordinates in metres
        (easting, northing).
    cal_indices : np.ndarray
        Integer indices into ``coords`` identifying the calibration sites.

    Returns
    -------
    float
        AD(β) in metres.  Lower values indicate more uniform spatial coverage.

    Raises
    ------
    ValueError
        If ``cal_indices`` is empty, or any index is out of range.

    Notes
    -----
    Uses ``scipy.spatial.cKDTree`` for O(N log k) query performance.

    Examples
    --------
    >>> import numpy as np
    >>> coords = np.array([[0., 0.], [1., 0.], [2., 0.], [0., 1.], [2., 1.]])
    >>> ad = average_distance(coords, np.array([0, 4]))
    >>> round(ad, 4)
    1.2361
    """
    coords = np.asarray(coords, dtype=float)
    cal_indices = np.asarray(cal_indices, dtype=int)

    if cal_indices.size == 0:
        raise ValueError("cal_indices must not be empty.")
    if coords.ndim != 2 or coords.shape[1] < 2:
        raise ValueError("coords must be shape (N, 2) or (N, ≥2).")
    n = len(coords)
    if cal_indices.min() < 0 or cal_indices.max() >= n:
        raise ValueError("cal_indices contains out-of-range index values.")

    cal_coords = coords[cal_indices]
    tree = KDTree(cal_coords)
    distances, _ = tree.query(coords, k=1, workers=1)
    return float(distances.mean())
