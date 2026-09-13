"""Regression tests for USDA Field 10-6 ESAP parity (101710A -> 106Frsd1).

Acceptance criterion (see docs/theory/rssd-source-notes.md Section 14 for full status and
rationale): stage-wise parity through PCA/QC/candidate pools, plus AD/Opt-Criteria dominance
over the ESAP reference set, plus a minimum exact site-ID overlap. Not exact 12/12 IDs --
ESAP's proprietary swap/greedy tie-breaking order is not published.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from rssdpy.io import read_em_survey
from rssdpy.pipeline import ESAPRunResult, run_esap_rssd
from rssdpy.sampling.rssd import _build_candidate_sets
from rssdpy.sampling.uniformity import average_distance, opt_criteria_esap

SURVEY_PATH = Path(__file__).resolve().parents[1] / "examples" / "data" / "101710A_for_esap.txt"

# ESAP-reported (site_id, Ds1-STD, Ds2-STD) for the 10 core + 2 support sites
# (examples/data/106Frsd1_expected.txt).
ESAP_ROWS = [
    (1902, 0.68, 0.00),
    (384, 1.13, 1.50),
    (3840, -1.77, -1.62),
    (2295, 1.31, -1.52),
    (5064, -1.43, 1.31),
    (5, 1.48, 0.02),
    (4617, -2.36, 0.05),
    (3940, -0.01, 1.95),
    (2226, -0.01, -2.10),
    (683, -0.68, 0.01),
    (2454, -0.16, 0.15),
    (3574, 0.02, -0.14),
]
ESAP_CORE_SITE_IDS = [row[0] for row in ESAP_ROWS[:10]]
ESAP_ALL_SITE_IDS = [row[0] for row in ESAP_ROWS]
ESAP_AD = 70.30
ESAP_OPT_CRITERIA = 1.26
DESIGN_FACTOR = 0.96
TARGET_SIZE = 12


@pytest.fixture(scope="module")
def survey() -> tuple[pd.DataFrame, np.ndarray]:
    eca, coords, _ = read_em_survey(
        SURVEY_PATH,
        eca_columns=["EMv", "EMh"],
        delimiter=",",
        has_header=False,
        column_names=["x", "y", "EMv", "EMh", "row"],
        crs="EPSG:6339",
        require_projected_crs=True,
    )
    return eca, coords


@pytest.fixture(scope="module")
def result(survey: tuple[pd.DataFrame, np.ndarray]) -> ESAPRunResult:
    eca, coords = survey
    return run_esap_rssd(
        eca,
        coords,
        n_components=2,
        target_size=TARGET_SIZE,
        design_factor=DESIGN_FACTOR,
        design_mode="esap_two_signal",
        extra_mode="global",
        opt_criteria_mode="esap",
    )


class TestSignalStatistics:
    """ESAP's 106Finfo_reference.txt reports these ln-signal summary statistics."""

    def test_log_signal_means(self, survey: tuple[pd.DataFrame, np.ndarray]) -> None:
        eca, _ = survey
        assert float(np.log(eca["EMv"]).mean()) == pytest.approx(5.693, abs=0.001)
        assert float(np.log(eca["EMh"]).mean()) == pytest.approx(5.521, abs=0.002)

    def test_log_signal_stds(self, survey: tuple[pd.DataFrame, np.ndarray]) -> None:
        eca, _ = survey
        assert float(np.log(eca["EMv"]).std()) == pytest.approx(0.372, abs=0.001)
        assert float(np.log(eca["EMh"]).std()) == pytest.approx(0.452, abs=0.001)

    def test_log_signal_correlation(self, survey: tuple[pd.DataFrame, np.ndarray]) -> None:
        eca, _ = survey
        corr = np.corrcoef(np.log(eca["EMv"]), np.log(eca["EMh"]))[0, 1]
        assert corr == pytest.approx(0.9828, abs=0.001)


class TestESAPQualityControl:
    """106Finfo_reference.txt: 72 sites > 3.5 STD, 1 deleted, 5029 active."""

    def test_masked_count(self, result: ESAPRunResult) -> None:
        assert int(result.qc.masking_mask.sum()) == 72

    def test_deleted_count(self, result: ESAPRunResult) -> None:
        # Bundled survey has 5102 rows; one is deleted at 4.5 sigma (source notes Section 14
        # covers the one-row discrepancy vs ESAP's reported "Survey Size: 5101").
        assert len(result.eca_clean) == len(result.qc.eligible_mask) == 5101

    def test_eligible_count(self, result: ESAPRunResult) -> None:
        assert int(result.qc.eligible_mask.sum()) == 5029


class TestPCScoreParity:
    """Standardised PC scores at the 12 ESAP-selected sites match Ds1/Ds2-STD, sign included."""

    def test_scores_match_esap_to_two_decimals(self, result: ESAPRunResult) -> None:
        position = {int(o): i for i, o in enumerate(result.original_indices)}
        for site_id, ds1, ds2 in ESAP_ROWS:
            score = result.scores[position[site_id - 1]]
            # ESAP prints two decimals, so the true deviation is at most 0.005.
            assert score[0] == pytest.approx(ds1, abs=0.01)
            assert score[1] == pytest.approx(ds2, abs=0.01)


class TestCandidatePools:
    """Each of ESAP's 10 core picks must lie inside our psi1-psi3 pool for its design level."""

    def test_all_core_picks_in_pool(self, result: ESAPRunResult) -> None:
        candidates, _ = _build_candidate_sets(
            result.scores, result.design, n_candidates=3, eligible_mask=result.qc.eligible_mask
        )
        position = {int(o): i for i, o in enumerate(result.original_indices)}
        pool_indices = {i for level in candidates for i in level}
        for site_id in ESAP_CORE_SITE_IDS:
            assert position[site_id - 1] in pool_indices, (
                f"ESAP core pick site_id={site_id} not found in any psi1-3 candidate pool"
            )


class TestADAndOptCriteriaDominance:
    """Our selection must be at least as spatially uniform as ESAP's own reference set."""

    def test_esap_reference_reproduces_its_own_opt_criteria(
        self, survey: tuple[pd.DataFrame, np.ndarray]
    ) -> None:
        """Sanity check for opt_criteria_esap itself, not our algorithm."""
        _, coords = survey
        beta = np.array(ESAP_ALL_SITE_IDS, dtype=int) - 1
        ad = average_distance(coords, beta)
        assert ad == pytest.approx(ESAP_AD, abs=0.1)
        opt = opt_criteria_esap(ad, coords, n_cal=len(beta))
        assert opt == pytest.approx(ESAP_OPT_CRITERIA, rel=0.02)

    def test_ad_final_leq_esap(self, result: ESAPRunResult) -> None:
        assert result.rssd.ad_final <= ESAP_AD

    def test_opt_criteria_leq_esap(self, result: ESAPRunResult) -> None:
        assert result.rssd.opt_criteria <= ESAP_OPT_CRITERIA * 1.02

    def test_minimum_exact_site_overlap(self, result: ESAPRunResult) -> None:
        selected_ids = set((result.rssd.selected_original_indices + 1).tolist())
        overlap = selected_ids & set(ESAP_ALL_SITE_IDS)
        assert len(overlap) >= 6, f"Only {len(overlap)}/12 exact site_id matches: {overlap}"
