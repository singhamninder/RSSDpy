"""Regression tests for USDA Field 10-6 ESAP parity (101710A → 106Frsd1)."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from rssdpy.features import ECaPCA, iterative_esap_validation
from rssdpy.io import read_em_survey
from rssdpy.sampling import (
    esap_sample_plan,
    esap_sampling_design,
    esap_two_signal_design,
    run_rssd,
)
from rssdpy.sampling.uniformity import average_distance, opt_criteria_esap

DATA_DIR = Path(__file__).resolve().parents[1] / "examples" / "data" / "field_10_6"
SURVEY_PATH = DATA_DIR / "101710A_for_esap.txt"
ESAP_EXPECTED_PATH = DATA_DIR / "106Frsd1_expected.txt"

ESAP_REFERENCE_SITE_IDS = [
    5,
    384,
    683,
    1902,
    2226,
    2295,
    2454,
    3574,
    3840,
    3940,
    4617,
    5064,
]
ESAP_OPT_CRITERIA = 1.26
DESIGN_FACTOR = 0.96


def _load_field_10_6() -> tuple[pd.DataFrame, np.ndarray]:
    eca, coords, _ = read_em_survey(
        SURVEY_PATH,
        eca_columns=["EMv", "EMh"],
        delimiter=",",
        has_header=False,
        column_names=["x", "y", "EMv", "EMh", "row"],
        crs="EPSG:6339",
        require_projected_crs=True,
    )
    return eca.iloc[:5101], coords[:5101]


class TestEsapTwoSignalDesign:
    def test_template_has_ten_levels(self) -> None:
        design = esap_two_signal_design(design_factor=DESIGN_FACTOR)
        assert design.shape == (10, 2)

    def test_template_magnitudes(self) -> None:
        design = esap_two_signal_design(design_factor=DESIGN_FACTOR)
        magnitudes = np.unique(np.round(np.abs(design[design != 0.0]), 2))
        np.testing.assert_allclose(magnitudes, [0.72, 1.68, 2.4], rtol=1e-2)

    def test_sample_plan_target_12(self) -> None:
        plan = esap_sample_plan(
            n_components=2,
            target_size=12,
            design_factor=DESIGN_FACTOR,
            design_mode="esap_two_signal",
        )
        assert plan.n_levels == 10
        assert plan.n_extra == 2
        assert plan.design_mode == "esap_two_signal"

    def test_sampling_design_router(self) -> None:
        design = esap_sampling_design(
            2,
            design_mode="esap_two_signal",
            design_factor=DESIGN_FACTOR,
        )
        assert design.shape == (10, 2)


class TestField106Parity:
    def test_em_log_stats_match_esap_info(self) -> None:
        eca, _ = _load_field_10_6()
        assert abs(float(np.log(eca["EMv"]).mean()) - 5.693) < 0.01
        assert abs(float(np.log(eca["EMh"]).mean()) - 5.521) < 0.01

    def test_esap_reference_opt_criteria_formula(self) -> None:
        _, coords = _load_field_10_6()
        beta = np.array(ESAP_REFERENCE_SITE_IDS, dtype=int) - 1
        ad = average_distance(coords, beta)
        opt = opt_criteria_esap(ad, coords, n_cal=len(beta))
        assert opt == pytest.approx(ESAP_OPT_CRITERIA, rel=0.02)

    def test_strict_template_site_overlap(self) -> None:
        eca, coords = _load_field_10_6()
        pca = ECaPCA(n_components=2)
        _, scores, qc, original_idx = iterative_esap_validation(eca, pca)
        coords_clean = coords[original_idx]

        plan = esap_sample_plan(
            n_components=2,
            target_size=12,
            design_factor=DESIGN_FACTOR,
            design_mode="esap_two_signal",
        )
        design = esap_sampling_design(
            2,
            design_mode="esap_two_signal",
            design_factor=DESIGN_FACTOR,
        )
        result = run_rssd(
            scores,
            coords_clean,
            design,
            n_extra=plan.n_extra,
            extra_mode="cube",
            eligible_mask=qc.eligible_mask,
            original_indices=original_idx,
            design_factor=DESIGN_FACTOR,
            opt_criteria_mode="esap",
        )

        selected_ids = set((result.selected_original_indices + 1).tolist())
        overlap = selected_ids & set(ESAP_REFERENCE_SITE_IDS)
        assert len(overlap) >= 6
        assert result.opt_criteria == pytest.approx(ESAP_OPT_CRITERIA, rel=0.15)
