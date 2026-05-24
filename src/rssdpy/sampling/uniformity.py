"""Spatial uniformity metric for calibration site sets.

Theory basis: Lesch, Strauss, and Rhoades (1995) Paper 2 — Confirmed.
  ESAP Opt-Criteria — Derived (dimensionless uniformity index; see source notes).

The Average Distance (AD) is:

    AD(β) = (1/N) Σᵢ min_{j ∈ β} d(site_i, site_j)

where ``d`` is Euclidean distance in projected geographic space (metres).
"""

import numpy as np
from scipy.spatial import KDTree


def average_distance(
    coords: np.ndarray,
    cal_indices: np.ndarray,
) -> float:
    """Compute the average nearest-calibration distance AD(β).

    Parameters
    ----------
    coords : np.ndarray
        Shape ``(N, 2)`` — projected geographic coordinates in metres.
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


def characteristic_spacing(coords: np.ndarray) -> float:
    """Estimate a characteristic survey spacing from coordinate extent.

    Uses ``sqrt(bounding_box_area / N)`` so the value has units of metres and
    scales like mean inter-site spacing on a roughly uniform grid.

    Parameters
    ----------
    coords : np.ndarray
        Shape ``(N, 2)`` — projected coordinates.

    Returns
    -------
    float
        Characteristic spacing in metres.

    Raises
    ------
    ValueError
        If fewer than two sites are provided.
    """
    coords = np.asarray(coords, dtype=float)
    if len(coords) < 2:
        raise ValueError("Need at least 2 sites to estimate characteristic spacing.")
    x_span = float(coords[:, 0].max() - coords[:, 0].min())
    y_span = float(coords[:, 1].max() - coords[:, 1].min())
    area = max(x_span * y_span, 1.0)
    return float(np.sqrt(area / len(coords)))


def opt_criteria_from_ad(ad: float, coords: np.ndarray) -> float:
    """Convert AD (metres) to an ESAP-like dimensionless Opt-Criteria value.

      Parameters
      ----------
      ad : float
          Average nearest-calibration distance in metres.
      coords : np.ndarray
          Full survey coordinates used for the AD calculation.

      Returns
      -------
      float
          Dimensionless uniformity index (lower is better; ESAP targets ≤1.30).

      Notes
      -----
      **Derived / Extension:** ESAP does not publish a closed formula. This
      implementation uses ``Opt-Criteria ≈ AD / characteristic_spacing`` so
      results are comparable across field sizes. Calibrate against legacy
    ``rsd#.txt`` outputs when available.
    """
    spacing = characteristic_spacing(coords)
    return float(ad / spacing)
