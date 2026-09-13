"""Smoke test for the Streamlit app using AppTest (no browser required).

Requires the ``app`` extra (streamlit); skipped otherwise so a minimal
``uv sync --group dev`` install (no ``--all-extras``) isn't broken.
"""

from pathlib import Path

import pytest

pytest.importorskip("streamlit")

from streamlit.testing.v1 import AppTest  # noqa: E402

APP_PATH = Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py"
SURVEY_PATH = Path(__file__).resolve().parents[1] / "examples" / "data" / "101710A_for_esap.txt"


def test_full_run_on_bundled_survey() -> None:
    at = AppTest.from_file(str(APP_PATH), default_timeout=60)
    at.run()
    assert not at.exception

    at.file_uploader[0].set_value((SURVEY_PATH.name, SURVEY_PATH.read_bytes(), "text/plain"))
    at.run()
    assert not at.exception

    # Headerless, comma-delimited (matches the bundled fixture: x,y,EMv,EMh,row).
    # "Delimiter" selectbox already defaults to "Comma"; "has_header" defaults to
    # False for a .txt upload. Column pickers default to col_0/col_1 (x/y) and
    # eca_columns defaults to columns[2:4] == ["col_2", "col_3"] (EMv/EMh).
    at.checkbox[0].set_value(False)
    at.run()

    at.button[0].click()
    at.run()

    assert not at.exception
    assert at.success
    assert "Selected" in at.success[0].value
    assert len(at.metric) == 4
    assert len(at.dataframe) >= 2  # preview table + selected-sites table
