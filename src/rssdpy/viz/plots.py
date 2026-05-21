"""Lightweight diagnostic plots for ECa PCA and MLR residuals.

Only produces publication-ready matplotlib figures.  Does not write files;
the caller is responsible for saving.
"""

import logging
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from matplotlib.axes import Axes

logger = logging.getLogger(__name__)


def plot_pc_scatter(
    scores: np.ndarray,
    outlier_mask: np.ndarray | None = None,
    selected_indices: np.ndarray | None = None,
    ax: Axes | None = None,
    pc_x: int = 0,
    pc_y: int = 1,
) -> Axes:
    """Bivariate scatter plot of PC scores.

    Parameters
    ----------
    scores : np.ndarray
        Shape ``(N, n_components)`` — standardised PC scores.
    outlier_mask : np.ndarray | None
        Boolean array of shape ``(N,)``.  Flagged sites shown in red.
    selected_indices : np.ndarray | None
        Indices of calibration sites to highlight.
    ax : matplotlib.axes.Axes | None
        Axes to plot on.  If ``None``, a new figure and axes are created.
    pc_x : int
        Component index for the x-axis (0-based).  Default 0.
    pc_y : int
        Component index for the y-axis (0-based).  Default 1.

    Returns
    -------
    matplotlib.axes.Axes
        The axes object with the scatter plot.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        _fig, ax = plt.subplots(figsize=(6, 5))
    assert ax is not None

    normal_mask = np.ones(len(scores), dtype=bool)
    if outlier_mask is not None:
        normal_mask = ~np.asarray(outlier_mask, dtype=bool)

    ax.scatter(
        scores[normal_mask, pc_x],
        scores[normal_mask, pc_y],
        c="steelblue",
        alpha=0.5,
        s=20,
        label="Survey sites",
        zorder=2,
    )

    if outlier_mask is not None and outlier_mask.any():
        ax.scatter(
            scores[~normal_mask, pc_x],
            scores[~normal_mask, pc_y],
            c="crimson",
            s=40,
            marker="x",
            label="Outliers",
            zorder=3,
        )

    if selected_indices is not None and len(selected_indices) > 0:
        sel = np.asarray(selected_indices)
        ax.scatter(
            scores[sel, pc_x],
            scores[sel, pc_y],
            c="gold",
            edgecolors="black",
            s=80,
            marker="^",
            label="Selected",
            zorder=4,
        )

    ax.set_xlabel(f"PC{pc_x + 1}")
    ax.set_ylabel(f"PC{pc_y + 1}")
    ax.legend(fontsize=8)
    ax.set_aspect("equal", adjustable="box")
    return ax


def plot_residuals(
    fitted: np.ndarray,
    residuals: np.ndarray,
    ax: Axes | None = None,
) -> Axes:
    """Residuals-vs-fitted plot with a horizontal zero line.

    Parameters
    ----------
    fitted : np.ndarray
        Shape ``(n_cal,)`` — fitted values (ŷ).
    residuals : np.ndarray
        Shape ``(n_cal,)`` — OLS residuals.
    ax : matplotlib.axes.Axes | None
        Axes to plot on.  If ``None``, a new figure and axes are created.

    Returns
    -------
    matplotlib.axes.Axes
    """
    import matplotlib.pyplot as plt

    if ax is None:
        _fig, ax = plt.subplots(figsize=(6, 4))
    assert ax is not None

    ax.scatter(fitted, residuals, c="steelblue", s=40, alpha=0.7, zorder=2)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Fitted ln(ECe)")
    ax.set_ylabel("Residual")
    ax.set_title("Residuals vs Fitted")
    return ax
