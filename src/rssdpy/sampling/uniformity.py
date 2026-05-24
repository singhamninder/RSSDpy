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


def opt_criteria_esap(
    ad: float,
    coords: np.ndarray,
    n_cal: int,
) -> float:
    """Convert AD to an ESAP-calibrated Opt-Criteria value.

    Parameters
    ----------
    ad : float
        Average nearest-calibration distance in coordinate units.
    coords : np.ndarray
        Full survey coordinates used for the AD calculation.
    n_cal : int
        Number of selected calibration sites.

    Returns
    -------
    float
        Dimensionless ESAP Opt-Criteria index.

    Notes
    -----
    **Derived:** Calibrated against legacy ``106Frsd1.txt`` (Opt-Criteria ≈ 1.26).
    Uses ``3 × AD / (spacing × sqrt(N / n_cal))`` where
    ``spacing`` is :func:`characteristic_spacing`.
    """
    if n_cal < 1:
        raise ValueError(f"n_cal must be ≥ 1, got {n_cal}")
    n_sites = len(coords)
    if n_sites < n_cal:
        raise ValueError(f"n_cal={n_cal} exceeds survey size {n_sites}.")
    spacing = characteristic_spacing(coords)
    return float(3.0 * ad / (spacing * np.sqrt(n_sites / n_cal)))


def compute_opt_criteria(
    ad: float,
    coords: np.ndarray,
    n_cal: int,
    *,
    mode: str = "derived",
) -> float:
    """Compute Opt-Criteria using the selected reporting mode.

    Parameters
    ----------
    ad : float
        Average nearest-calibration distance.
    coords : np.ndarray
        Survey coordinates.
    n_cal : int
        Number of selected calibration sites.
    mode : {"derived", "esap"}
        ``"derived"`` uses ``AD / characteristic_spacing``; ``"esap"`` uses
        :func:`opt_criteria_esap`.

    Returns
    -------
    float
        Dimensionless uniformity index.
    """
    if mode == "esap":
        return opt_criteria_esap(ad, coords, n_cal)
    if mode == "derived":
        return opt_criteria_from_ad(ad, coords)
    raise ValueError(f"opt_criteria mode must be 'derived' or 'esap', got {mode!r}")
