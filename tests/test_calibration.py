"""Tests for calibration/mlr.py, calibration/diagnostics.py, and predict/salinity.py."""

import numpy as np
import pandas as pd
import pytest

from rssdpy.calibration.diagnostics import DiagnosticsResult, run_diagnostics
from rssdpy.calibration.mlr import (
    MLRResult,
    _build_model_matrix,
    _compute_press,
    fit_mlr_models,
)
from rssdpy.predict.salinity import predict_salinity

# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def simple_data() -> dict:
    """Minimal 3-PC, 20-site dataset with 16 calibration + 4 prediction sites."""
    rng = np.random.default_rng(99)
    n = 20
    n_cal = 16
    scores = rng.standard_normal((n, 3))
    coords = rng.uniform(0, 300, (n, 2))
    cal_idx = np.arange(n_cal)
    # ln(ECe) is a noisy linear function of the first PC score
    ln_ece = 1.5 + 0.8 * scores[cal_idx, 0] + rng.normal(0, 0.3, n_cal)
    return {
        "scores": scores,
        "coords": coords,
        "cal_indices": cal_idx,
        "ln_ece": ln_ece,
        "n": n,
        "n_cal": n_cal,
    }


# ── MLR model matrix tests ────────────────────────────────────────────────────


class TestBuildModelMatrix:
    def test_intercept_is_always_included(self, simple_data: dict) -> None:
        x_mat, names = _build_model_matrix(
            simple_data["scores"],
            simple_data["coords"],
            simple_data["cal_indices"],
            pc_flags=(True, False, False, False),
            trend_flags=(False, False, False, False, False),
        )
        assert "intercept" in names
        np.testing.assert_array_equal(x_mat[:, 0], 1.0)

    def test_correct_column_count_pc_only(self, simple_data: dict) -> None:
        # κ₁, κ₂, κ₁κ₂, κ₃ → 4 PC cols + 1 intercept = 5
        x_mat, names = _build_model_matrix(
            simple_data["scores"],
            simple_data["coords"],
            simple_data["cal_indices"],
            pc_flags=(True, True, True, True),
            trend_flags=(False, False, False, False, False),
        )
        assert x_mat.shape[1] == 5
        assert len(names) == 5

    def test_full_trend_columns(self, simple_data: dict) -> None:
        # intercept + k1 + k2 + k1k2 + k3 + x + y + xy + x2 + y2 = 10
        x_mat, names = _build_model_matrix(
            simple_data["scores"],
            simple_data["coords"],
            simple_data["cal_indices"],
            pc_flags=(True, True, True, True),
            trend_flags=(True, True, True, True, True),
        )
        assert x_mat.shape[1] == 10
        assert "xy" in names

    def test_interaction_term_correct(self, simple_data: dict) -> None:
        s = simple_data["scores"]
        cal = simple_data["cal_indices"]
        x_mat, names = _build_model_matrix(
            s,
            simple_data["coords"],
            cal,
            pc_flags=(True, True, True, False),
            trend_flags=(False, False, False, False, False),
        )
        k1k2_col = x_mat[:, names.index("k1k2")]
        expected = s[cal, 0] * s[cal, 1]
        np.testing.assert_allclose(k1k2_col, expected)


# ── PRESS and APVE ────────────────────────────────────────────────────────────


class TestPressApve:
    def test_press_is_positive(self, simple_data: dict) -> None:
        models = fit_mlr_models(
            simple_data["scores"],
            simple_data["coords"],
            simple_data["cal_indices"],
            simple_data["ln_ece"],
        )
        assert models[0].press > 0

    def test_apve_is_positive(self, simple_data: dict) -> None:
        models = fit_mlr_models(
            simple_data["scores"],
            simple_data["coords"],
            simple_data["cal_indices"],
            simple_data["ln_ece"],
        )
        assert models[0].apve > 0

    def test_models_sorted_by_press(self, simple_data: dict) -> None:
        models = fit_mlr_models(
            simple_data["scores"],
            simple_data["coords"],
            simple_data["cal_indices"],
            simple_data["ln_ece"],
        )
        presses = [m.press for m in models]
        assert presses == sorted(presses)

    def test_press_identity(self, simple_data: dict) -> None:
        """PRESS via hat matrix must approximately equal brute-force LOO error."""
        import statsmodels.api as sm

        x_mat, _ = _build_model_matrix(
            simple_data["scores"],
            simple_data["coords"],
            simple_data["cal_indices"],
            pc_flags=(True, True, False, False),
            trend_flags=(False, False, False, False, False),
        )
        y = simple_data["ln_ece"]
        ols = sm.OLS(y, x_mat).fit()

        # Hat-matrix PRESS
        press_hat = _compute_press(ols)

        # Brute-force LOO PRESS
        n = len(y)
        loo_sq = 0.0
        for i in range(n):
            mask = np.ones(n, dtype=bool)
            mask[i] = False
            ols_i = sm.OLS(y[mask], x_mat[mask]).fit()
            pred_i = float(x_mat[i] @ ols_i.params)
            loo_sq += (y[i] - pred_i) ** 2

        # Should agree to within ~5% for typical OLS
        assert abs(press_hat - loo_sq) / max(abs(loo_sq), 1e-10) < 0.05


class TestFitMLRModels:
    def test_returns_list(self, simple_data: dict) -> None:
        models = fit_mlr_models(
            simple_data["scores"],
            simple_data["coords"],
            simple_data["cal_indices"],
            simple_data["ln_ece"],
        )
        assert isinstance(models, list)
        assert len(models) > 0

    def test_model_result_type(self, simple_data: dict) -> None:
        models = fit_mlr_models(
            simple_data["scores"],
            simple_data["coords"],
            simple_data["cal_indices"],
            simple_data["ln_ece"],
        )
        assert all(isinstance(m, MLRResult) for m in models)

    def test_r2_in_range(self, simple_data: dict) -> None:
        models = fit_mlr_models(
            simple_data["scores"],
            simple_data["coords"],
            simple_data["cal_indices"],
            simple_data["ln_ece"],
        )
        for m in models[:5]:
            assert 0.0 <= m.r2 <= 1.0 + 1e-10

    def test_mismatched_cal_raises(self, simple_data: dict) -> None:
        with pytest.raises(ValueError, match="cal_indices"):
            fit_mlr_models(
                simple_data["scores"],
                simple_data["coords"],
                simple_data["cal_indices"],
                simple_data["ln_ece"][:5],  # wrong length
            )


# ── Diagnostics ───────────────────────────────────────────────────────────────


class TestRunDiagnostics:
    def test_returns_dataclass(self, simple_data: dict) -> None:
        models = fit_mlr_models(
            simple_data["scores"],
            simple_data["coords"],
            simple_data["cal_indices"],
            simple_data["ln_ece"],
        )
        best = models[0]
        residuals = np.asarray(best.ols_result.resid)
        cal_coords = simple_data["coords"][simple_data["cal_indices"]]
        result = run_diagnostics(residuals, cal_coords, best.n_params)
        assert isinstance(result, DiagnosticsResult)

    def test_shapiro_pvalue_range(self, simple_data: dict) -> None:
        models = fit_mlr_models(
            simple_data["scores"],
            simple_data["coords"],
            simple_data["cal_indices"],
            simple_data["ln_ece"],
        )
        residuals = np.asarray(models[0].ols_result.resid)
        cal_coords = simple_data["coords"][simple_data["cal_indices"]]
        result = run_diagnostics(residuals, cal_coords, models[0].n_params)
        assert 0.0 <= result.shapiro_pvalue <= 1.0

    def test_moran_i_range(self, simple_data: dict) -> None:
        models = fit_mlr_models(
            simple_data["scores"],
            simple_data["coords"],
            simple_data["cal_indices"],
            simple_data["ln_ece"],
        )
        residuals = np.asarray(models[0].ols_result.resid)
        cal_coords = simple_data["coords"][simple_data["cal_indices"]]
        result = run_diagnostics(residuals, cal_coords, models[0].n_params)
        assert -1.0 <= result.moran_i <= 1.0

    def test_length_mismatch_raises(self) -> None:
        residuals = np.ones(10)
        coords = np.ones((8, 2))
        with pytest.raises(ValueError, match="residuals"):
            run_diagnostics(residuals, coords, 3)


# ── Prediction ────────────────────────────────────────────────────────────────


class TestPredictSalinity:
    def test_basic_shape(self, simple_data: dict) -> None:
        models = fit_mlr_models(
            simple_data["scores"],
            simple_data["coords"],
            simple_data["cal_indices"],
            simple_data["ln_ece"],
        )
        result = predict_salinity(
            models[0],
            simple_data["scores"],
            simple_data["coords"],
            simple_data["cal_indices"],
            simple_data["ln_ece"],
        )
        n_pred = simple_data["n"] - simple_data["n_cal"]
        assert result.pred_ln_ece.shape == (n_pred,)
        assert result.pred_variance.shape == (n_pred,)

    def test_variance_is_positive(self, simple_data: dict) -> None:
        models = fit_mlr_models(
            simple_data["scores"],
            simple_data["coords"],
            simple_data["cal_indices"],
            simple_data["ln_ece"],
        )
        result = predict_salinity(
            models[0],
            simple_data["scores"],
            simple_data["coords"],
            simple_data["cal_indices"],
            simple_data["ln_ece"],
        )
        assert (result.pred_variance > 0).all()

    def test_field_mean_between_extremes(self, simple_data: dict) -> None:
        models = fit_mlr_models(
            simple_data["scores"],
            simple_data["coords"],
            simple_data["cal_indices"],
            simple_data["ln_ece"],
        )
        result = predict_salinity(
            models[0],
            simple_data["scores"],
            simple_data["coords"],
            simple_data["cal_indices"],
            simple_data["ln_ece"],
        )
        lo = min(simple_data["ln_ece"].min(), result.pred_ln_ece.min())
        hi = max(simple_data["ln_ece"].max(), result.pred_ln_ece.max())
        assert lo <= result.field_mean <= hi

    def test_probability_interval_sums_to_one(self, simple_data: dict) -> None:
        models = fit_mlr_models(
            simple_data["scores"],
            simple_data["coords"],
            simple_data["cal_indices"],
            simple_data["ln_ece"],
        )
        result = predict_salinity(
            models[0],
            simple_data["scores"],
            simple_data["coords"],
            simple_data["cal_indices"],
            simple_data["ln_ece"],
        )
        boundaries = [0, 2, 4, 8, 16, float("inf")]
        props = result.range_proportions(boundaries)
        assert abs(props.sum() - 1.0) < 1e-6

    def test_no_pred_sites_raises(self, simple_data: dict) -> None:
        """All sites as calibration → no prediction sites → ValueError."""
        models = fit_mlr_models(
            simple_data["scores"],
            simple_data["coords"],
            simple_data["cal_indices"],
            simple_data["ln_ece"],
        )
        with pytest.raises(ValueError, match="no prediction sites"):
            predict_salinity(
                models[0],
                simple_data["scores"][:16],
                simple_data["coords"][:16],
                np.arange(16),
                simple_data["ln_ece"],
            )

    def test_end_to_end_with_synthetic_fixture(
        self,
        synthetic_eca_df: pd.DataFrame,
        synthetic_coords: np.ndarray,
        synthetic_ece: np.ndarray,
    ) -> None:
        """Integration test: PCA → RSSD → MLR → prediction chain."""
        from rssdpy.features.pca import ECaPCA
        from rssdpy.sampling.design import central_composite_design
        from rssdpy.sampling.rssd import run_rssd

        pca = ECaPCA(n_components=2)
        scores, _, _ = pca.fit_transform(synthetic_eca_df)
        design = central_composite_design(n_components=2, include_center=False)
        run_rssd(scores, synthetic_coords, design, n_extra=8)

        # Use fixture cal indices 0..15 for deterministic calibration data alignment
        cal_idx_fixed = np.arange(16)
        models = fit_mlr_models(scores, synthetic_coords, cal_idx_fixed, synthetic_ece)
        pred = predict_salinity(models[0], scores, synthetic_coords, cal_idx_fixed, synthetic_ece)
        assert pred.pred_ln_ece.shape[0] == len(synthetic_eca_df) - 16
        assert np.isfinite(pred.field_mean)
