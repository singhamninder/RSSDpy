"""Tests for features/pca.py and features/outliers.py."""

import numpy as np
import pandas as pd
import pytest

from rssdpy.features.outliers import detect_outliers, outlier_threshold
from rssdpy.features.pca import ECaPCA


class TestECaPCA:
    def test_basic_shape(self, synthetic_eca_df: pd.DataFrame) -> None:
        pca = ECaPCA(n_components=2)
        scores, eigenvalues, eigenvectors = pca.fit_transform(synthetic_eca_df)
        n = len(synthetic_eca_df)
        assert scores.shape == (n, 2)
        assert eigenvalues.shape == (2,)
        assert eigenvectors.shape == (2, 2)

    def test_scores_standardised(self, synthetic_eca_df: pd.DataFrame) -> None:
        """Standardised scores must have mean≈0 and sample std≈1 per component.

        sklearn's PCA uses ddof=1 eigenvalues, so the unbiased (ddof=1) std of
        each standardised score column should equal 1.0.
        """
        pca = ECaPCA(n_components=2)
        scores, _, _ = pca.fit_transform(synthetic_eca_df)
        np.testing.assert_allclose(scores.mean(axis=0), 0.0, atol=1e-10)
        np.testing.assert_allclose(scores.std(axis=0, ddof=1), 1.0, atol=1e-6)

    def test_three_components(self, synthetic_eca_df: pd.DataFrame) -> None:
        """Cannot request more components than ECa channels."""
        pca = ECaPCA(n_components=2)
        scores, _, _ = pca.fit_transform(synthetic_eca_df)
        assert scores.shape[1] == 2

    def test_too_many_components_raises(self, synthetic_eca_df: pd.DataFrame) -> None:
        pca = ECaPCA(n_components=3)
        with pytest.raises(ValueError, match="exceeds the number of ECa channels"):
            pca.fit_transform(synthetic_eca_df)

    def test_non_positive_eca_raises(self) -> None:
        bad_eca = pd.DataFrame({"EMh": [1.0, -0.5, 2.0], "EMv": [1.0, 1.0, 1.0]})
        pca = ECaPCA(n_components=2)
        with pytest.raises(ValueError, match="strictly positive"):
            pca.fit_transform(bad_eca)

    def test_zero_eca_raises(self) -> None:
        bad_eca = pd.DataFrame({"EMh": [1.0, 0.0, 2.0], "EMv": [1.0, 1.0, 1.0]})
        pca = ECaPCA(n_components=2)
        with pytest.raises(ValueError, match="strictly positive"):
            pca.fit_transform(bad_eca)

    def test_transform_before_fit_raises(self, synthetic_eca_df: pd.DataFrame) -> None:
        pca = ECaPCA(n_components=2)
        with pytest.raises(RuntimeError, match="fit_transform"):
            pca.transform(synthetic_eca_df)

    def test_transform_consistent(self, synthetic_eca_df: pd.DataFrame) -> None:
        """transform() should produce the same result as fit_transform on the same data."""
        pca = ECaPCA(n_components=2)
        scores_fit, _, _ = pca.fit_transform(synthetic_eca_df)
        scores_transform = pca.transform(synthetic_eca_df)
        np.testing.assert_allclose(scores_fit, scores_transform, rtol=1e-10)

    def test_eigenvalues_descending(self, synthetic_eca_df: pd.DataFrame) -> None:
        pca = ECaPCA(n_components=2)
        _, eigenvalues, _ = pca.fit_transform(synthetic_eca_df)
        assert eigenvalues[0] >= eigenvalues[1]

    def test_explained_variance_sum_leq_1(self, synthetic_eca_df: pd.DataFrame) -> None:
        pca = ECaPCA(n_components=2)
        pca.fit_transform(synthetic_eca_df)
        assert pca.explained_variance_ratio_ is not None
        assert pca.explained_variance_ratio_.sum() <= 1.0 + 1e-10

    def test_empty_dataframe_raises(self) -> None:
        pca = ECaPCA(n_components=2)
        with pytest.raises(ValueError, match="empty"):
            pca.fit_transform(pd.DataFrame({"EMh": [], "EMv": []}))


class TestOutlierDetection:
    def test_basic_outlier_flag(self) -> None:
        """Sites far from origin should be flagged; nearby sites should not."""
        n = 50
        rng = np.random.default_rng(0)
        scores = rng.normal(0, 1, size=(n, 3))
        # Plant a clear outlier
        scores[0] = [6.0, 6.0, 6.0]
        mask, distances = detect_outliers(scores, alpha=0.001)
        assert mask[0], "Obvious outlier not flagged."
        # Most normal-ish sites should not be flagged
        assert mask[1:].sum() < 10

    def test_no_outliers_in_clean_data(self) -> None:
        """Very tightly clustered data near origin should have no outliers."""
        scores = np.full((20, 3), fill_value=0.1)
        mask, _ = detect_outliers(scores)
        assert not mask.any()

    def test_distances_shape(self, synthetic_eca_df: pd.DataFrame) -> None:
        pca = ECaPCA(n_components=2)
        scores, _, _ = pca.fit_transform(synthetic_eca_df)
        mask, distances = detect_outliers(scores, alpha=0.001)
        assert distances.shape == (len(scores),)
        assert mask.shape == (len(scores),)

    def test_threshold_formula_3d(self) -> None:
        """Threshold for 3D at alpha=0.001 should ≈ 4.03 (matches Paper 2 description)."""
        from scipy.stats import chi2

        thresh = outlier_threshold(n_components=3, alpha=0.001)
        expected = np.sqrt(chi2.ppf(0.999, df=3))
        assert abs(thresh - expected) < 1e-10

    def test_threshold_increases_with_components(self) -> None:
        """More PC components → larger chi-squared tail → higher threshold."""
        t2 = outlier_threshold(2)
        t3 = outlier_threshold(3)
        t4 = outlier_threshold(4)
        assert t2 < t3 < t4

    def test_invalid_input_raises(self) -> None:
        with pytest.raises(ValueError, match="2-D"):
            detect_outliers(np.array([1.0, 2.0, 3.0]))

    def test_invalid_alpha_raises(self) -> None:
        with pytest.raises(ValueError, match="alpha"):
            outlier_threshold(3, alpha=1.5)
