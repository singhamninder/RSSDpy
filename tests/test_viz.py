"""Smoke tests for viz/plots.py using the non-interactive Agg backend."""

import matplotlib

matplotlib.use("Agg")

import numpy as np
from matplotlib.axes import Axes

from rssdpy.viz import plot_pc_scatter, plot_residuals


class TestPlotPCScatter:
    def test_returns_axes_minimal(self) -> None:
        rng = np.random.default_rng(0)
        scores = rng.standard_normal((20, 2))
        ax = plot_pc_scatter(scores)
        assert isinstance(ax, Axes)

    def test_with_outliers_and_selected(self) -> None:
        rng = np.random.default_rng(1)
        scores = rng.standard_normal((20, 2))
        outlier_mask = np.zeros(20, dtype=bool)
        outlier_mask[:3] = True
        selected = np.array([5, 6, 7])
        ax = plot_pc_scatter(scores, outlier_mask=outlier_mask, selected_indices=selected)
        assert isinstance(ax, Axes)
        assert ax.get_xlabel() == "PC1"
        assert ax.get_ylabel() == "PC2"

    def test_custom_pc_axes(self) -> None:
        rng = np.random.default_rng(2)
        scores = rng.standard_normal((10, 3))
        ax = plot_pc_scatter(scores, pc_x=1, pc_y=2)
        assert ax.get_xlabel() == "PC2"
        assert ax.get_ylabel() == "PC3"


class TestPlotResiduals:
    def test_returns_axes(self) -> None:
        rng = np.random.default_rng(3)
        fitted = rng.uniform(0, 5, size=15)
        residuals = rng.normal(0, 0.5, size=15)
        ax = plot_residuals(fitted, residuals)
        assert isinstance(ax, Axes)
        assert ax.get_title() == "Residuals vs Fitted"
