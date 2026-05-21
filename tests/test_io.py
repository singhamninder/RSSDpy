"""Tests for survey ingestion and RSSD export utilities."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from rssdpy.io import (
    export_selected_sites_csv,
    export_selected_sites_geopackage,
    read_em_survey,
    selected_sites_table,
    validate_canonical_survey,
    write_esap_style_report,
)
from rssdpy.sampling.design import central_composite_design
from rssdpy.sampling.rssd import run_rssd


class TestCanonicalSurveyValidation:
    def test_validate_canonical_survey_success(self) -> None:
        frame = pd.DataFrame(
            {
                "site_id": [1, 2, 3],
                "x": [100.0, 101.0, 102.0],
                "y": [200.0, 201.0, 202.0],
                "EMh": [12.0, 13.0, 14.0],
                "EMv": [8.0, 9.0, 10.0],
            }
        )
        validated = validate_canonical_survey(
            frame,
            eca_columns=["EMh", "EMv"],
            crs="EPSG:32611",
            require_projected_crs=True,
        )
        assert list(validated.columns) == list(frame.columns)

    def test_duplicate_site_id_raises(self) -> None:
        frame = pd.DataFrame(
            {
                "site_id": [1, 1],
                "x": [100.0, 101.0],
                "y": [200.0, 201.0],
                "EMh": [12.0, 13.0],
            }
        )
        with pytest.raises(ValueError, match="unique"):
            validate_canonical_survey(frame, eca_columns=["EMh"])

    def test_non_positive_eca_raises(self) -> None:
        frame = pd.DataFrame(
            {
                "site_id": [1, 2],
                "x": [100.0, 101.0],
                "y": [200.0, 201.0],
                "EMh": [12.0, 0.0],
            }
        )
        with pytest.raises(ValueError, match="strictly positive"):
            validate_canonical_survey(frame, eca_columns=["EMh"])


class TestSurveyParsers:
    def test_read_em_survey_xyz(self, tmp_path: Path) -> None:
        path = tmp_path / "demo.xyz"
        path.write_text("100,200,12,8\n101,201,13,9\n", encoding="utf-8")
        eca, coords, metadata = read_em_survey(
            path,
            eca_columns=["EMh", "EMv"],
            format_hint="xyz",
            column_names=["x", "y", "EMh", "EMv"],
        )
        assert eca.shape == (2, 2)
        np.testing.assert_allclose(coords, np.array([[100.0, 200.0], [101.0, 201.0]]))
        assert metadata["extension"] == ".xyz"

    def test_read_em_survey_dat_veris_profile(self, tmp_path: Path) -> None:
        path = tmp_path / "veris.dat"
        path.write_text("100\t200\t12.1\t8.4\n101\t201\t12.9\t8.8\n", encoding="utf-8")
        eca, coords, metadata = read_em_survey(
            path,
            eca_columns=["EMh", "EMv"],
            format_hint="dat",
            profile="veris",
        )
        assert eca["EMh"].iloc[0] == pytest.approx(12.1)
        assert coords.shape == (2, 2)
        assert metadata["profile"] == "veris"


class TestExports:
    def _make_result(self) -> tuple:
        rng = np.random.default_rng(11)
        scores = rng.standard_normal((80, 2))
        coords = rng.uniform(0, 500, size=(80, 2))
        design = central_composite_design(n_components=2, include_center=False)
        result = run_rssd(scores, coords, design, n_extra=2, n_validation=8)
        return result, coords

    def test_selected_sites_table_columns(self) -> None:
        result, coords = self._make_result()
        table = selected_sites_table(result, coords)
        assert {"site_id", "x", "y", "selection_type"}.issubset(set(table.columns))
        assert len(table) == len(result.selected_indices)

    def test_export_selected_sites_csv(self, tmp_path: Path) -> None:
        result, coords = self._make_result()
        output = tmp_path / "selected_sites.csv"
        export_selected_sites_csv(result, coords, output)
        loaded = pd.read_csv(output)
        assert len(loaded) == len(result.selected_indices)

    def test_write_esap_style_report(self, tmp_path: Path) -> None:
        result, coords = self._make_result()
        _ = coords
        output = tmp_path / "report.txt"
        write_esap_style_report(result, output)
        content = output.read_text(encoding="utf-8")
        assert "RSSD Selection Report" in content
        assert "Core sites" in content

    def test_export_selected_sites_geopackage(self, tmp_path: Path) -> None:
        result, coords = self._make_result()
        output = tmp_path / "selected_sites.gpkg"
        export_selected_sites_geopackage(result, coords, output)
        assert output.exists()
