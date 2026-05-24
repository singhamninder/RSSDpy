"""Load and normalize ECa survey data for RSSD workflows."""

import logging
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from pyproj import CRS

logger = logging.getLogger(__name__)

_SUPPORTED_SURVEY_SUFFIXES: frozenset[str] = frozenset({".csv", ".txt"})

_SURVEY_DEFAULTS: dict[str, dict[str, str | bool]] = {
    ".csv": {"delimiter": ",", "has_header": True},
    ".txt": {"delimiter": r"\s+", "has_header": False},
}


def _is_projected_crs(crs_value: str | None) -> bool:
    """Return True when CRS string describes a projected CRS."""
    if crs_value is None:
        return False
    return bool(CRS.from_user_input(crs_value).is_projected)


def _validate_canonical_columns(
    frame: pd.DataFrame,
    *,
    eca_columns: list[str],
    x_col: str,
    y_col: str,
    site_id_col: str,
) -> None:
    required = {site_id_col, x_col, y_col, *eca_columns}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Survey data is missing required columns: {sorted(missing)}")


def _validate_positive_eca(frame: pd.DataFrame, eca_columns: list[str]) -> None:
    if frame[eca_columns].isna().any().any():
        raise ValueError("ECa columns contain missing values.")
    if not np.isfinite(frame[eca_columns].to_numpy(dtype=float)).all():
        raise ValueError("ECa columns must be finite numeric values.")
    if (frame[eca_columns].to_numpy(dtype=float) <= 0.0).any():
        raise ValueError("All ECa values must be strictly positive.")


def _validate_coords(frame: pd.DataFrame, *, x_col: str, y_col: str) -> None:
    coords = frame[[x_col, y_col]].to_numpy(dtype=float)
    if not np.isfinite(coords).all():
        raise ValueError("Coordinate columns must be finite numeric values.")


def validate_canonical_survey(
    survey: pd.DataFrame,
    eca_columns: list[str],
    *,
    x_col: str = "x",
    y_col: str = "y",
    site_id_col: str = "site_id",
    crs: str | None = None,
    require_projected_crs: bool = False,
) -> pd.DataFrame:
    """Validate and normalize a canonical survey table.

    Parameters
    ----------
    survey : pd.DataFrame
        Input table containing coordinates, site identifier, and ECa channels.
    eca_columns : list[str]
        ECa column names to validate.
    x_col : str
        Easting/x column name. Default ``"x"``.
    y_col : str
        Northing/y column name. Default ``"y"``.
    site_id_col : str
        Stable survey site identifier column name. Default ``"site_id"``.
    crs : str | None
        Optional CRS string (EPSG or WKT).
    require_projected_crs : bool
        If ``True``, raises when ``crs`` is not projected.

    Returns
    -------
    pd.DataFrame
        Copy of ``survey`` with validated numeric coordinate/ECa columns.

    Raises
    ------
    ValueError
        If required columns are missing, values are invalid, or CRS requirements
        are not satisfied.
    """
    frame = survey.copy()
    _validate_canonical_columns(
        frame,
        eca_columns=eca_columns,
        x_col=x_col,
        y_col=y_col,
        site_id_col=site_id_col,
    )

    if frame[site_id_col].isna().any():
        raise ValueError("site_id column contains missing values.")
    if frame[site_id_col].duplicated().any():
        raise ValueError("site_id values must be unique and stable.")

    _validate_positive_eca(frame, eca_columns)
    _validate_coords(frame, x_col=x_col, y_col=y_col)

    if require_projected_crs and not _is_projected_crs(crs):
        raise ValueError(
            f"Projected CRS is required for RSSD distance calculations; received {crs!r}."
        )

    frame[x_col] = frame[x_col].astype(float)
    frame[y_col] = frame[y_col].astype(float)
    for column in eca_columns:
        frame[column] = frame[column].astype(float)
    return frame


def _read_delimited(
    path: Path,
    *,
    delimiter: str,
    has_header: bool,
    column_names: list[str] | None,
    encoding: str,
) -> pd.DataFrame:
    header = 0 if has_header else None
    return pd.read_csv(
        path,
        sep=delimiter,
        header=header,
        names=column_names if not has_header else None,
        encoding=encoding,
        engine="python" if delimiter == r"\s+" else None,
    )


def load_eca_csv(
    path: str | Path,
    eca_columns: list[str],
    x_col: str = "x",
    y_col: str = "y",
    delimiter: str = ",",
) -> tuple[pd.DataFrame, np.ndarray]:
    """Load an ECa survey from a comma-separated CSV file.

    Parameters
    ----------
    path : str | Path
        Path to the CSV file.
    eca_columns : list[str]
        Names of the ECa reading columns (e.g. ``["EMh", "EMv"]``).
        All values must be positive floats after loading.
    x_col : str
        Name of the easting / x column.  Default ``"x"``.
    y_col : str
        Name of the northing / y column.  Default ``"y"``.
    delimiter : str
        CSV column delimiter.  Default ``","``.

    Returns
    -------
    eca : pd.DataFrame
        Columns are the names listed in ``eca_columns``.
    coords : np.ndarray
        Shape ``(N, 2)`` — ``[[x, y], …]`` in the original CRS units.

    Raises
    ------
    ValueError
        If required columns are missing from the file.
    FileNotFoundError
        If the file does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"ECa CSV not found: {path}")

    df = pd.read_csv(path, sep=delimiter)

    required = {x_col, y_col, *eca_columns}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {sorted(missing)}")

    eca = df[eca_columns].copy()
    coords = df[[x_col, y_col]].to_numpy(dtype=float)
    logger.info("Loaded %d survey sites from %s", len(df), path)
    return eca, coords


def read_em_survey(
    path: str | Path,
    eca_columns: list[str],
    *,
    x_col: str = "x",
    y_col: str = "y",
    site_id_col: str = "site_id",
    format_hint: Literal["txt", "csv"] | None = None,
    delimiter: str | None = None,
    has_header: bool | None = None,
    column_names: list[str] | None = None,
    crs: str | None = None,
    require_projected_crs: bool = False,
    encoding: str = "utf-8",
) -> tuple[pd.DataFrame, np.ndarray, dict[str, str | int | None]]:
    """Parse a CSV or whitespace-delimited TXT survey into canonical columns.

    Only ``.csv`` and ``.txt`` inputs are supported. TXT files are read into a
    tabular form and validated like CSV.

    Parameters
    ----------
    path : str | Path
        Input EM survey file path (``.csv`` or ``.txt``).
    eca_columns : list[str]
        ECa channel names to keep in the normalized frame.
    x_col : str
        X/easting column name in the parsed table.
    y_col : str
        Y/northing column name in the parsed table.
    site_id_col : str
        Site identifier column name in the normalized table.
    format_hint : {'txt', 'csv'} | None
        Optional parser hint when the file extension is missing or ambiguous.
    delimiter : str | None
        Optional explicit delimiter override (default: ``,`` for CSV, whitespace
        for TXT).
    has_header : bool | None
        Optional explicit header flag override.
    column_names : list[str] | None
        Column names for headerless files.
    crs : str | None
        Optional source CRS string used for projected CRS validation.
    require_projected_crs : bool
        If ``True``, requires ``crs`` to be projected.
    encoding : str
        Text encoding for file parsing.

    Returns
    -------
    tuple[pd.DataFrame, np.ndarray, dict[str, str | int | None]]
        ``eca_df`` (ECa channels only), ``coords`` array of shape ``(N, 2)``,
        and parsing metadata.

    Raises
    ------
    ValueError
        If the file type is unsupported, parser options are inconsistent, or
        required columns are missing.
    FileNotFoundError
        If ``path`` does not exist.
    """
    source_path = Path(path)
    if not source_path.exists():
        raise FileNotFoundError(f"EM survey file not found: {source_path}")

    if format_hint is not None:
        extension = f".{format_hint.lower()}"
    else:
        extension = source_path.suffix.lower()

    if extension not in _SUPPORTED_SURVEY_SUFFIXES:
        supported = ", ".join(sorted(_SUPPORTED_SURVEY_SUFFIXES))
        raise ValueError(
            f"Unsupported survey file {source_path.name!r} ({extension!r}). "
            f"Supported inputs: {supported}. "
            "Convert other formats to CSV or TXT with canonical columns."
        )

    defaults = _SURVEY_DEFAULTS[extension]
    parse_delimiter = delimiter or str(defaults["delimiter"])
    parse_has_header = has_header if has_header is not None else bool(defaults["has_header"])

    if not parse_has_header and column_names is None:
        raise ValueError("column_names are required when has_header=False.")

    raw = _read_delimited(
        source_path,
        delimiter=parse_delimiter,
        has_header=parse_has_header,
        column_names=column_names,
        encoding=encoding,
    )
    if site_id_col not in raw.columns:
        raw.insert(0, site_id_col, np.arange(1, len(raw) + 1))

    canonical = validate_canonical_survey(
        raw,
        eca_columns=eca_columns,
        x_col=x_col,
        y_col=y_col,
        site_id_col=site_id_col,
        crs=crs,
        require_projected_crs=require_projected_crs,
    )

    eca = canonical[eca_columns].copy()
    coords = canonical[[x_col, y_col]].to_numpy(dtype=float)
    metadata: dict[str, str | int | None] = {
        "source_path": str(source_path),
        "extension": extension,
        "n_sites": int(len(canonical)),
        "site_id_col": site_id_col,
        "x_col": x_col,
        "y_col": y_col,
        "crs": crs,
    }
    logger.info("Loaded %d survey sites from %s", len(canonical), source_path)
    return eca, coords, metadata
