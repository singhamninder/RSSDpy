"""Tests for sampling/design.py, sampling/uniformity.py, and sampling/rssd.py."""

import numpy as np
import pandas as pd
import pytest

from rssdpy.sampling.design import ESAP_RADIUS_SQUARED, central_composite_design, esap_sample_plan
from rssdpy.sampling.rssd import RSSDesign, run_rssd
from rssdpy.sampling.uniformity import average_distance, opt_criteria_from_ad


class TestCentralCompositeDesign:
    def test_3d_no_center_shape(self) -> None:
        """3D CCD without center must have exactly 14 levels (confirmed Paper 2)."""
        des = central_composite_design(n_components=3, include_center=False)
        assert des.shape == (14, 3)

    def test_3d_with_center_shape(self) -> None:
        des = central_composite_design(n_components=3, include_center=True)
        assert des.shape == (15, 3)

    def test_2d_no_center_shape(self) -> None:
        """2D CCD: 4 axial + 4 cube = 8 levels."""
        des = central_composite_design(n_components=2, include_center=False)
        assert des.shape == (8, 2)

    def test_4d_no_center_shape(self) -> None:
        """4D CCD: 8 axial + 16 cube = 24 levels."""
        des = central_composite_design(n_components=4, include_center=False)
        assert des.shape == (24, 4)

    def test_all_points_on_ellipsoid(self) -> None:
        """Every non-center point must satisfy sum(x²) == radius_squared (Confirmed)."""
        for n in (2, 3, 4):
            des = central_composite_design(n_components=n, include_center=False)
            radii_sq = np.sum(des**2, axis=1)
            np.testing.assert_allclose(
                radii_sq,
                ESAP_RADIUS_SQUARED,
                rtol=1e-10,
                err_msg=f"Ellipsoid radius mismatch for n_components={n}",
            )

    def test_center_point_is_zero(self) -> None:
        des = central_composite_design(n_components=3, include_center=True)
        np.testing.assert_array_equal(des[-1], np.zeros(3))

    def test_axial_count(self) -> None:
        """There must be exactly 2*n_components axial points."""
        for n in (2, 3, 4):
            des = central_composite_design(n_components=n, include_center=False)
            n_axial = 2 * n
            n_cube = 2**n
            assert des.shape[0] == n_axial + n_cube

    def test_custom_radius(self) -> None:
        des = central_composite_design(n_components=3, radius_squared=7.0, include_center=False)
        radii_sq = np.sum(des**2, axis=1)
        np.testing.assert_allclose(radii_sq, 7.0, rtol=1e-10)

    def test_invalid_n_components_raises(self) -> None:
        with pytest.raises(ValueError, match="n_components"):
            central_composite_design(n_components=1)

    def test_invalid_radius_raises(self) -> None:
        with pytest.raises(ValueError, match="radius_squared"):
            central_composite_design(radius_squared=-1.0)

    def test_esap_constant_value(self) -> None:
        """3.84 ≈ chi2.ppf(0.95, df=1) ≈ 1.96^2 (source notes Section 4).

        3.84 is a rounded constant; exact values are ~3.8415 and ~3.8416.
        Tolerance is set to 2e-3 to accommodate the rounding.
        """
        from scipy.stats import chi2

        assert abs(ESAP_RADIUS_SQUARED - chi2.ppf(0.95, df=1)) < 2e-3
        assert abs(ESAP_RADIUS_SQUARED - 1.96**2) < 2e-3

    def test_esap_sample_plan_2d_target_12(self) -> None:
        plan = esap_sample_plan(n_components=2, target_size=12)
        assert plan.n_levels == 8
        assert plan.n_extra == 4

    def test_esap_sample_plan_target_too_small_raises(self) -> None:
        with pytest.raises(ValueError, match="smaller than CCD core size"):
            esap_sample_plan(n_components=3, target_size=12)

    def test_design_factor_scales_levels(self) -> None:
        base = central_composite_design(n_components=2, include_center=False)
        scaled = central_composite_design(n_components=2, include_center=False, design_factor=0.96)
        np.testing.assert_allclose(scaled, base * 0.96, rtol=1e-10)


class TestAverageDistance:
    def test_single_cal_site_is_max_distance(self) -> None:
        """With one calibration site, every survey site contributes its distance."""
        coords = np.array([[0.0, 0.0], [3.0, 4.0]])  # dist = 5
        ad = average_distance(coords, np.array([0]))
        assert abs(ad - 2.5) < 1e-10  # (0 + 5) / 2

    def test_all_sites_cal_gives_zero(self) -> None:
        """When all sites are calibration sites, nearest distance is 0."""
        coords = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
        ad = average_distance(coords, np.array([0, 1, 2]))
        assert ad == 0.0

    def test_symmetric(self) -> None:
        coords = np.array([[0.0, 0.0], [4.0, 0.0], [2.0, 0.0]])
        ad1 = average_distance(coords, np.array([0, 1]))
        ad2 = average_distance(coords, np.array([1, 0]))
        assert abs(ad1 - ad2) < 1e-10

    def test_empty_cal_raises(self) -> None:
        coords = np.array([[0.0, 0.0], [1.0, 1.0]])
        with pytest.raises(ValueError, match="empty"):
            average_distance(coords, np.array([], dtype=int))

    def test_out_of_range_raises(self) -> None:
        coords = np.array([[0.0, 0.0], [1.0, 1.0]])
        with pytest.raises(ValueError):
            average_distance(coords, np.array([5]))


class TestRunRSSD:
    def _make_scores_coords(self, n: int = 100) -> tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(7)
        scores = rng.standard_normal((n, 3))
        coords = rng.uniform(0, 500, (n, 2))
        return scores, coords

    def test_output_type(self) -> None:
        scores, coords = self._make_scores_coords()
        design = central_composite_design(n_components=3, include_center=False)
        result = run_rssd(scores, coords, design, n_extra=0)
        assert isinstance(result, RSSDesign)
        assert len(result.selected_original_indices) == len(result.selected_indices)

    def test_selected_indices_count(self) -> None:
        scores, coords = self._make_scores_coords()
        design = central_composite_design(n_components=3, include_center=False)
        n_extra = 2
        result = run_rssd(scores, coords, design, n_extra=n_extra)
        assert len(result.selected_indices) == len(design) + n_extra

    def test_unique_selected_indices(self) -> None:
        """No site may appear twice in the final selection."""
        scores, coords = self._make_scores_coords()
        design = central_composite_design(n_components=3, include_center=False)
        result = run_rssd(scores, coords, design, n_extra=2)
        assert len(result.selected_indices) == len(np.unique(result.selected_indices))

    def test_ad_non_increasing_trace(self) -> None:
        """AD trace must be monotonically non-increasing."""
        scores, coords = self._make_scores_coords()
        design = central_composite_design(n_components=3, include_center=False)
        result = run_rssd(scores, coords, design, n_extra=3)
        trace = result.ad_trace
        for i in range(1, len(trace)):
            assert trace[i] <= trace[i - 1] + 1e-12, f"AD increased at step {i}"

    def test_ad_final_leq_initial(self) -> None:
        scores, coords = self._make_scores_coords()
        design = central_composite_design(n_components=3, include_center=False)
        result = run_rssd(scores, coords, design, n_extra=0)
        assert result.ad_final <= result.ad_initial + 1e-12

    def test_dls_values_correct(self) -> None:
        """DLS values must match squared Euclidean distances for the final assignment.

        After swapping, selected_indices[:n_levels] gives the final site for each
        design level; dls_values is computed from those final assignments.
        """
        scores, coords = self._make_scores_coords()
        design = central_composite_design(n_components=3, include_center=False)
        n_levels = len(design)
        result = run_rssd(scores, coords, design, n_extra=0)
        for j in range(n_levels):
            site_idx = result.selected_indices[j]
            expected_dls = float(np.sum((scores[site_idx] - design[j]) ** 2))
            assert abs(result.dls_values[j] - expected_dls) < 1e-10

    def test_incompatible_lengths_raise(self) -> None:
        scores, coords = self._make_scores_coords(n=50)
        design = central_composite_design(n_components=3, include_center=False)
        with pytest.raises(ValueError, match="rows"):
            run_rssd(scores, coords[:30], design)

    def test_component_mismatch_raises(self) -> None:
        scores, coords = self._make_scores_coords()
        design = central_composite_design(n_components=2, include_center=False)
        with pytest.raises(ValueError, match="components"):
            run_rssd(scores, coords, design)

    def test_extra_indices_subset_of_remaining(self) -> None:
        """Extra site indices must not overlap with core design-level assignments."""
        scores, coords = self._make_scores_coords()
        design = central_composite_design(n_components=3, include_center=False)
        result = run_rssd(scores, coords, design, n_extra=3)
        core_set = set(result.design_level_indices.tolist())
        extra_set = set(result.extra_indices.tolist())
        assert core_set.isdisjoint(extra_set)

    def test_cube_extra_mode_runs(self) -> None:
        scores, coords = self._make_scores_coords()
        design = central_composite_design(n_components=3, include_center=False)
        result = run_rssd(scores, coords, design, n_extra=2, extra_mode="cube")
        assert len(result.extra_indices) <= 2
        assert len(result.selected_indices) >= len(design)

    def test_validation_pass_returns_indices(self) -> None:
        scores, coords = self._make_scores_coords(n=180)
        design = central_composite_design(n_components=3, include_center=False)
        result = run_rssd(scores, coords, design, n_extra=2, n_validation=8)
        assert len(result.validation_indices) >= 8
        assert len(result.validation_original_indices) == len(result.validation_indices)

    def test_opt_criteria_positive(self) -> None:
        scores, coords = self._make_scores_coords()
        design = central_composite_design(n_components=3, include_center=False)
        result = run_rssd(scores, coords, design, n_extra=1)
        assert result.opt_criteria > 0
        expected = opt_criteria_from_ad(result.ad_final, coords)
        assert abs(result.opt_criteria - expected) < 1e-10

    def test_eligible_mask_excludes_sites(self) -> None:
        scores, coords = self._make_scores_coords(n=80)
        design = central_composite_design(n_components=3, include_center=False)
        eligible = np.ones(len(scores), dtype=bool)
        eligible[0] = False
        result = run_rssd(scores, coords, design, eligible_mask=eligible)
        assert 0 not in result.selected_indices

    def test_original_index_mapping(self) -> None:
        scores, coords = self._make_scores_coords(n=130)
        design = central_composite_design(n_components=3, include_center=False)
        original = np.arange(1000, 1130)
        result = run_rssd(scores, coords, design, original_indices=original)
        np.testing.assert_array_equal(
            result.selected_original_indices, original[result.selected_indices]
        )

    def test_full_pipeline_with_synthetic_fixture(
        self,
        synthetic_eca_df: pd.DataFrame,
        synthetic_coords: np.ndarray,
    ) -> None:
        """Integration smoke test using shared synthetic survey fixtures."""
        from rssdpy.features.pca import ECaPCA

        pca = ECaPCA(n_components=2)
        scores, _, _ = pca.fit_transform(synthetic_eca_df)
        design = central_composite_design(n_components=2, include_center=False)
        result = run_rssd(scores, synthetic_coords, design, n_extra=4)
        assert len(result.selected_indices) == len(design) + 4
        assert result.ad_final > 0
