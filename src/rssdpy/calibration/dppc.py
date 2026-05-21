"""Dual Pathway Parallel Conductance (DPPC) model utilities.

Extension: Rhoades et al. (1989) deterministic model relating ECa to ECe,
water content, saturation percentage, bulk density, and temperature.
This module is a reference implementation — it is NOT required for the
RSSD statistical workflow, which uses MLR (see ``calibration/mlr.py``).

Theory basis: Rhoades et al. (1989). SSSAJ 53(2): 433–439.
  https://doi.org/10.2136/sssaj1989.03615995005300020020x
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)


def dppc_eca(
    ece: np.ndarray,
    theta_w: np.ndarray,
    sp: np.ndarray,
    theta_ws: np.ndarray,
    ecs: float = 0.023,
) -> np.ndarray:
    """Estimate ECa from soil properties via the simplified DPPC model.

    This implements the two-pathway conductance relationship described in
    Rhoades et al. (1989), Equation 6 (simplified form).

    Extension: this function is provided as a reference and is not used in
    the core RSSD sampling algorithm.

    Parameters
    ----------
    ece : np.ndarray
        Electrical conductivity of the saturation extract (dS/m).
    theta_w : np.ndarray
        Volumetric water content (m³/m³).
    sp : np.ndarray
        Saturation percentage (dimensionless fraction, not percent).
    theta_ws : np.ndarray
        Volumetric water content of the series-coupled pathway (m³/m³).
        Must satisfy ``theta_ws ≤ theta_w``.
    ecs : float
        Surface conductivity of soil particles (dS/m).  Default 0.023.

    Returns
    -------
    np.ndarray
        Estimated ECa (dS/m).

    Raises
    ------
    ValueError
        If any input array has a different length, or if ``theta_ws > theta_w``.

    Notes
    -----
    ECw is estimated from ECe and SP via: ECw ≈ ECe × SP / 100.
    The simplified DPPC equation is:

        ECa ≈ (θ_ws² × ECw × ECs) / (θ_s × ECw + θ_ws × ECs)
              + (θ_w − θ_ws) × ECw
    """
    ece = np.asarray(ece, dtype=float)
    theta_w = np.asarray(theta_w, dtype=float)
    sp = np.asarray(sp, dtype=float)
    theta_ws = np.asarray(theta_ws, dtype=float)

    shapes = {ece.shape, theta_w.shape, sp.shape, theta_ws.shape}
    if len(shapes) > 1:
        raise ValueError("All input arrays must have the same shape.")

    if np.any(theta_ws > theta_w):
        raise ValueError("theta_ws must be ≤ theta_w everywhere.")

    # ECw from ECe and saturation percentage
    ecw = ece * sp

    theta_s = theta_w  # saturated water content ≈ theta_w at saturation

    denom = theta_s * ecw + theta_ws * ecs
    # Series-coupled pathway (avoid divide by zero)
    safe_denom = np.where(denom == 0, np.nan, denom)
    pathway1 = (theta_ws**2 * ecw * ecs) / safe_denom

    # Continuous liquid pathway
    pathway2 = (theta_w - theta_ws) * ecw

    return pathway1 + pathway2
