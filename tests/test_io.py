"""Tests for survey ingestion and RSSD export utilities."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from rssdpy.io import (
    export_selected_sites_csv,
    load_eca_csv,
    read_em_survey,
    selected_sites_table,
    validate_canonical_survey,
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

    def test_missing_eca_values_raises(self) -> None:
        frame = pd.DataFrame(
            {
                "site_id": [1, 2],
                "x": [100.0, 101.0],
                "y": [200.0, 201.0],
                "EMh": [12.0, np.nan],
            }
        )
        with pytest.raises(ValueError, match="missing values"):
            validate_canonical_survey(frame, eca_columns=["EMh"])

    def test_non_finite_coords_raises(self) -> None:
        frame = pd.DataFrame(
            {
                "site_id": [1, 2],
                "x": [100.0, np.inf],
                "y": [200.0, 201.0],
                "EMh": [12.0, 13.0],
            }
        )
        with pytest.raises(ValueError, match="finite"):
            validate_canonical_survey(frame, eca_columns=["EMh"])

    def test_missing_site_id_raises(self) -> None:
        frame = pd.DataFrame(
            {
                "site_id": [1, np.nan],
                "x": [100.0, 101.0],
                "y": [200.0, 201.0],
                "EMh": [12.0, 13.0],
            }
        )
        with pytest.raises(ValueError, match="site_id column contains missing"):
            validate_canonical_survey(frame, eca_columns=["EMh"])

    def test_non_projected_crs_raises(self) -> None:
        frame = pd.DataFrame(
            {
                "site_id": [1, 2],
                "x": [100.0, 101.0],
                "y": [200.0, 201.0],
                "EMh": [12.0, 13.0],
            }
        )
        with pytest.raises(ValueError, match="Projected CRS"):
            validate_canonical_survey(
                frame, eca_columns=["EMh"], crs="EPSG:4326", require_projected_crs=True
            )


class TestSurveyParsers:
    def test_read_em_survey_csv(self, tmp_path: Path) -> None:
        path = tmp_path / "survey.csv"
        path.write_text(
            "site_id,x,y,EMh,EMv\n1,100,200,12,8\n2,101,201,13,9\n",
            encoding="utf-8",
        )
        eca, coords, metadata = read_em_survey(path, eca_columns=["EMh", "EMv"])
        assert eca.shape == (2, 2)
        np.testing.assert_allclose(coords, np.array([[100.0, 200.0], [101.0, 201.0]]))
        assert metadata["extension"] == ".csv"

    def test_read_em_survey_txt_headerless(self, tmp_path: Path) -> None:
        path = tmp_path / "survey.txt"
        path.write_text("1 100 200 12 8\n2 101 201 13 9\n", encoding="utf-8")
        eca, coords, metadata = read_em_survey(
            path,
            eca_columns=["EMh", "EMv"],
            column_names=["site_id", "x", "y", "EMh", "EMv"],
        )
        assert eca["EMh"].iloc[0] == pytest.approx(12.0)
        assert coords.shape == (2, 2)
        assert metadata["extension"] == ".txt"

    def test_unsupported_extension_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "survey.xyz"
        path.write_text("100,200,12,8\n", encoding="utf-8")
        with pytest.raises(ValueError, match="Unsupported survey file"):
            read_em_survey(
                path,
                eca_columns=["EMh", "EMv"],
                column_names=["x", "y", "EMh", "EMv"],
            )

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="EM survey file not found"):
            read_em_survey(tmp_path / "nope.csv", eca_columns=["EMh"])

    def test_format_hint_overrides_missing_extension(self, tmp_path: Path) -> None:
        """A file with no recognizable extension can still be parsed via format_hint."""
        path = tmp_path / "survey_no_ext"
        path.write_text("1,100,200,12,8\n2,101,201,13,9\n", encoding="utf-8")
        eca, coords, metadata = read_em_survey(
            path,
            eca_columns=["EMh", "EMv"],
            format_hint="csv",
            has_header=False,
            column_names=["site_id", "x", "y", "EMh", "EMv"],
        )
        assert eca.shape == (2, 2)
        assert metadata["extension"] == ".csv"

    def test_explicit_delimiter_overrides_default(self, tmp_path: Path) -> None:
        """A .txt file that is actually comma-delimited needs an explicit delimiter."""
        path = tmp_path / "survey.txt"
        path.write_text("100,200,12,8\n101,201,13,9\n", encoding="utf-8")
        eca, coords, _ = read_em_survey(
            path,
            eca_columns=["EMh", "EMv"],
            delimiter=",",
            has_header=False,
            column_names=["x", "y", "EMh", "EMv"],
        )
        assert eca["EMh"].iloc[0] == pytest.approx(12.0)

    def test_headerless_without_column_names_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "survey.txt"
        path.write_text("1 100 200 12 8\n", encoding="utf-8")
        with pytest.raises(ValueError, match="column_names are required"):
            read_em_survey(path, eca_columns=["EMh"], has_header=False)


class TestLoadEcaCsv:
    def test_success(self, tmp_path: Path) -> None:
        path = tmp_path / "survey.csv"
        path.write_text("x,y,EMh,EMv\n100,200,12,8\n101,201,13,9\n", encoding="utf-8")
        eca, coords = load_eca_csv(path, eca_columns=["EMh", "EMv"])
        assert eca.shape == (2, 2)
        np.testing.assert_allclose(coords, np.array([[100.0, 200.0], [101.0, 201.0]]))

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="ECa CSV not found"):
            load_eca_csv(tmp_path / "nope.csv", eca_columns=["EMh"])

    def test_missing_columns_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "survey.csv"
        path.write_text("x,y,EMh\n100,200,12\n", encoding="utf-8")
        with pytest.raises(ValueError, match="missing required columns"):
            load_eca_csv(path, eca_columns=["EMh", "EMv"])


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
        assert {
            "site_id",
            "x",
            "y",
            "selection_type",
            "opt_criteria",
            "design_factor",
        }.issubset(set(table.columns))
        assert "support" in table["selection_type"].to_numpy()
        assert len(table) == len(result.selected_indices)

    def test_export_selected_sites_csv(self, tmp_path: Path) -> None:
        result, coords = self._make_result()
        output = tmp_path / "selected_sites.csv"
        export_selected_sites_csv(result, coords, output)
        loaded = pd.read_csv(output)
        assert len(loaded) == len(result.selected_indices)

    def test_explicit_site_ids_used(self) -> None:
        result, coords = self._make_result()
        custom_ids = np.arange(9000, 9000 + len(result.selected_indices))
        table = selected_sites_table(result, coords, site_ids=custom_ids)
        np.testing.assert_array_equal(table["site_id"].to_numpy(), custom_ids)

    def test_design_annotation_adds_pc_columns(self) -> None:
        result, coords = self._make_result()
        design = central_composite_design(n_components=2, include_center=False)
        table = selected_sites_table(result, coords, design=design)
        assert "design_pc1" in table.columns and "design_pc2" in table.columns
        n_core = len(result.design_level_indices)
        assert table["design_pc1"].notna().sum() == n_core
