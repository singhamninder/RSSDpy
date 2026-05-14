"""Central Composite Design (CCD) generation for RSSD site selection.

Theory basis: Lesch, Strauss, and Rhoades (1995) Paper 2 — Confirmed for
3-component CCD without center (14 levels).  2- and 4-component variants
are Extensions; see ``docs/theory/rssd-source-notes.md``.

Design constant ``radius_squared = 3.84``:
  Equals ``scipy.stats.chi2.ppf(0.95, df=1)`` and ``1.96**2``.
  It is the Lesch/ESAP design constant, NOT ``chi2.ppf(0.95, df=3) ≈ 7.815``.
  See source notes Section 4 for full derivation note.

Layout:
  - Axial points: ``(±α, 0, …)`` etc., where ``α = sqrt(radius_squared)``.
  - Cube points: ``(±c, ±c, …)`` for all sign combinations, where
    ``c = sqrt(radius_squared / n_components)``.
"""

from __future__ import annotations

import itertools

import numpy as np

# Confirmed: Lesch/ESAP design constant (see source notes Section 4)
ESAP_RADIUS_SQUARED: float = 3.84


def central_composite_design(
    n_components: int = 3,
    radius_squared: float = ESAP_RADIUS_SQUARED,
    include_center: bool = False,
) -> np.ndarray:
    """Generate a Central Composite Design (CCD) in standardised PC space.

    The design consists of axial points (one non-zero coordinate) and cube
    points (all coordinates ±c).  A center point at the origin may optionally
    be included.

    For the ESAP-faithful 3-component case (n_components=3, include_center=False):
    returns exactly 14 levels — 6 axial + 8 cube.

    Parameters
    ----------
    n_components : int
        Number of PC dimensions (= number of design factors).  Default 3
        matches Lesch et al. 1995 Paper 2.
    radius_squared : float
        The squared radius of the design ellipsoid.  Default ``3.84`` is the
        Lesch/ESAP design constant.  See module docstring for its derivation.
    include_center : bool
        If ``True``, append the origin ``(0, 0, …)`` as a final row.
        ESAP-RSSD omits the center point; set ``False`` for ESAP-compatible
        behaviour.

    Returns
    -------
    np.ndarray
        Shape ``(n_levels, n_components)`` — CCD design matrix.
        - Row order: axial points first, cube points second, center last
          (if included).
        - All non-center levels lie exactly on the ellipsoid defined by
          ``radius_squared`` only for the axial points.  Cube point radius
          equals ``sqrt(radius_squared)`` as well (each coordinate is
          ``c = sqrt(radius_squared / n_components)``, so
          ``c² × n_components = radius_squared``).

    Raises
    ------
    ValueError
        If ``n_components < 2`` or ``radius_squared <= 0``.

    Notes
    -----
    Axial radius ``α = sqrt(radius_squared)``; cube-point half-extent
    ``c = sqrt(radius_squared / n_components)``.  Both types of points lie
    on the same hypersphere of radius ``sqrt(radius_squared)``.

    Examples
    --------
    >>> D = central_composite_design(n_components=3, include_center=False)
    >>> D.shape
    (14, 3)
    >>> import numpy as np
    >>> np.allclose(np.sum(D**2, axis=1), 3.84)
    True
    """
    if n_components < 2:
        raise ValueError(f"n_components must be ≥ 2, got {n_components}")
    if radius_squared <= 0:
        raise ValueError(f"radius_squared must be positive, got {radius_squared}")

    alpha = np.sqrt(radius_squared)  # axial extent
    c = np.sqrt(radius_squared / n_components)  # cube-point half-extent

    # Axial points: one non-zero coordinate at ±alpha, rest zero
    axial = []
    for dim in range(n_components):
        for sign in (+1, -1):
            point = np.zeros(n_components)
            point[dim] = sign * alpha
            axial.append(point)

    # Cube points: all ±c combinations
    cube = [np.array(combo) * c for combo in itertools.product([-1, 1], repeat=n_components)]

    parts: list[np.ndarray] = [np.array(axial), np.array(cube)]
    if include_center:
        parts.append(np.zeros((1, n_components)))

    design = np.vstack(parts)
    return design
