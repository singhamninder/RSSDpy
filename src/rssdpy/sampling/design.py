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

import itertools
import logging
from dataclasses import dataclass
from typing import Literal

import numpy as np

logger = logging.getLogger(__name__)

# Confirmed: Lesch/ESAP design constant (see source notes Section 4)
ESAP_RADIUS_SQUARED: float = 3.84
ESAP_TARGET_SAMPLE_SIZES: tuple[int, int, int] = (6, 12, 20)
ESAP_DESIGN_FACTOR_MIN: float = 0.90
ESAP_DESIGN_FACTOR_MAX: float = 1.10

# ESAP 2-signal SRS template magnitudes at design_factor=1.0 (Confirmed via 106Frsd1)
ESAP_TWO_SIGNAL_AXIAL_OUTER: float = 2.5
ESAP_TWO_SIGNAL_CUBE: float = 1.75
ESAP_TWO_SIGNAL_AXIAL_INNER: float = 0.75

DesignMode = Literal["ccd", "esap_two_signal"]


@dataclass(frozen=True)
class ESAPSamplePlan:
    """ESAP-oriented configuration for a single RSSD run.

    Attributes
    ----------
    n_components : int
        Number of principal-component dimensions used in the CCD.
    target_size : int
        Requested total RSSD sample size.
    n_levels : int
        Number of core CCD design levels.
    n_extra : int
        Number of extra sites to add after core-level assignment.
    include_center : bool
        Whether the center design level is included.
    """

    n_components: int
    target_size: int
    n_levels: int
    n_extra: int
    include_center: bool
    design_factor: float
    design_mode: DesignMode


def _validate_design_factor(design_factor: float) -> None:
    if design_factor <= 0:
        raise ValueError(f"design_factor must be positive, got {design_factor}")
    if not ESAP_DESIGN_FACTOR_MIN <= design_factor <= ESAP_DESIGN_FACTOR_MAX:
        logger.warning(
            "design_factor=%.3f is outside ESAP-recommended range [%.2f, %.2f].",
            design_factor,
            ESAP_DESIGN_FACTOR_MIN,
            ESAP_DESIGN_FACTOR_MAX,
        )


def central_composite_design(
    n_components: int = 3,
    radius_squared: float = ESAP_RADIUS_SQUARED,
    include_center: bool = False,
    design_factor: float = 1.0,
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
    design_factor : float
        Multiplier applied to all design levels (ESAP “D-Factor Val”).
        Recommended range 0.90–1.10 per ESAP manual §3.5.2.

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
    _validate_design_factor(design_factor)

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
    if design_factor != 1.0:
        center_rows = 1 if include_center else 0
        if center_rows:
            design[:-center_rows] *= design_factor
        else:
            design *= design_factor
    return design


def esap_two_signal_design(design_factor: float = 1.0) -> np.ndarray:
    """Generate the ESAP 2-signal SRS template in standardised PC space.

    Returns 10 core design levels matching the level family observed in ESAP
    ``rsd#.txt`` outputs (outer axial ``±2.5``, cube ``±1.75``, inner axial
    ``±0.75`` at ``design_factor=1.0``), scaled by ``design_factor``.

    Row order follows the ESAP reference listing in ``106Frsd1.txt`` so
    candidate-set construction matches desktop ESAP.

    Parameters
    ----------
    design_factor : float
        ESAP D-Factor multiplier (recommended 0.90–1.10).

    Returns
    -------
    np.ndarray
        Shape ``(10, 2)`` design matrix.
    """
    _validate_design_factor(design_factor)
    outer = ESAP_TWO_SIGNAL_AXIAL_OUTER * design_factor
    cube = ESAP_TWO_SIGNAL_CUBE * design_factor
    inner = ESAP_TWO_SIGNAL_AXIAL_INNER * design_factor
    return np.array(
        [
            [inner, 0.0],
            [cube, cube],
            [-cube, -cube],
            [cube, -cube],
            [-cube, cube],
            [outer, 0.0],
            [-outer, 0.0],
            [0.0, outer],
            [0.0, -outer],
            [-inner, 0.0],
        ],
        dtype=float,
    )


def esap_sampling_design(
    n_components: int,
    *,
    design_mode: DesignMode = "ccd",
    include_center: bool = False,
    radius_squared: float = ESAP_RADIUS_SQUARED,
    design_factor: float = 1.0,
) -> np.ndarray:
    """Return a PC-space design matrix for RSSD site matching.

    Parameters
    ----------
    n_components : int
        Number of principal-component dimensions.
    design_mode : {"ccd", "esap_two_signal"}
        ``"ccd"`` uses the generic central composite design. ``"esap_two_signal"``
        uses the strict ESAP 2-signal SRS template (10 levels).
    include_center : bool
        Passed to :func:`central_composite_design` when ``design_mode="ccd"``.
    radius_squared : float
        CCD radius-squared constant when ``design_mode="ccd"``.
    design_factor : float
        ESAP D-Factor multiplier.

    Returns
    -------
    np.ndarray
        Design matrix of shape ``(n_levels, n_components)``.

    Raises
    ------
    ValueError
        If ``design_mode="esap_two_signal"`` with ``n_components != 2``.
    """
    if design_mode == "esap_two_signal":
        if n_components != 2:
            raise ValueError(
                f"design_mode='esap_two_signal' requires n_components=2, got {n_components}."
            )
        return esap_two_signal_design(design_factor=design_factor)
    return central_composite_design(
        n_components=n_components,
        include_center=include_center,
        radius_squared=radius_squared,
        design_factor=design_factor,
    )


def esap_sample_plan(
    n_components: int,
    target_size: int,
    *,
    design_mode: DesignMode = "ccd",
    include_center: bool = False,
    radius_squared: float = ESAP_RADIUS_SQUARED,
    design_factor: float = 1.0,
) -> ESAPSamplePlan:
    """Create an ESAP-style sample-size plan for RSSD.

    Parameters
    ----------
    n_components : int
        Number of principal-component dimensions for the CCD.
    target_size : int
        Desired total sample size. Must be one of ``(6, 12, 20)``.
    include_center : bool
        Whether to include the center point in the CCD.
    radius_squared : float
        CCD radius-squared constant passed through to
        :func:`central_composite_design`.
    design_mode : {"ccd", "esap_two_signal"}
        Design generator to use. For ESAP 2-signal parity, use
        ``"esap_two_signal"``.
    design_factor : float
        ESAP design-factor multiplier for CCD levels.

    Returns
    -------
    ESAPSamplePlan
        Plan containing the number of core levels and required extra sites.

    Raises
    ------
    ValueError
        If ``target_size`` is unsupported or smaller than the CCD core size.
    """
    if target_size not in ESAP_TARGET_SAMPLE_SIZES:
        raise ValueError(
            f"target_size must be one of {ESAP_TARGET_SAMPLE_SIZES}, got {target_size}."
        )
    n_levels = len(
        esap_sampling_design(
            n_components,
            design_mode=design_mode,
            include_center=include_center,
            radius_squared=radius_squared,
            design_factor=design_factor,
        )
    )
    if target_size < n_levels:
        raise ValueError(
            f"target_size={target_size} is smaller than CCD core size {n_levels} "
            f"for n_components={n_components}."
        )
    return ESAPSamplePlan(
        n_components=n_components,
        target_size=target_size,
        n_levels=n_levels,
        n_extra=target_size - n_levels,
        include_center=include_center,
        design_factor=design_factor,
        design_mode=design_mode,
    )
