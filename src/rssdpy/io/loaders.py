"""Load and normalize ECa survey data for RSSD workflows."""

import logging
from pathlib import Path
from typing import Literal

import geopandas as gpd
import numpy as np
import pandas as pd
from pyproj import CRS

logger = logging.getLogger(__name__)

_DELIMITED_PROFILES: dict[str, dict[str, str | list[str] | bool]] = {
    "veris": {
        "delimiter": "\t",
        "column_names": ["x", "y", "EMh", "EMv"],
        "has_header": False,
    },
    "hol31": {
        "delimiter": r"\s+",
        "column_names": ["x", "y", "EM"],
        "has_header": False,
    },
    "bwd101p": {
        "delimiter": r"\s+",
        "column_names": ["y", "x", "EMh", "EMv", "flag"],
        "has_header": False,
    },
}

_EM_EXTENSION_DEFAULTS: dict[str, dict[str, str | bool]] = {
    ".txt": {"delimiter": r"\s+", "has_header": True},
    ".xyz": {"delimiter": ",", "has_header": False},
    ".dat": {"delimiter": r"\s+", "has_header": False},
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
    """Load an ECa survey from a CSV file.

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


def load_eca_geodataframe(
    path: str | Path,
    eca_columns: list[str],
    layer: str | None = None,
) -> tuple[pd.DataFrame, np.ndarray, str | None]:
    """Load an ECa survey from a vector file (Shapefile, GeoJSON, GPKG, …).

    Parameters
    ----------
    path : str | Path
        Path to the vector file.
    eca_columns : list[str]
        Names of the ECa reading columns.
    layer : str | None
        Layer name for multi-layer formats (e.g. GeoPackage).

    Returns
    -------
    eca : pd.DataFrame
        Columns are the names listed in ``eca_columns``.
    coords : np.ndarray
        Shape ``(N, 2)`` — ``[[x, y], …]`` extracted from geometry centroids.
    crs : str | None
        CRS string (WKT or EPSG code).  ``None`` if undefined.

    Raises
    ------
    ValueError
        If required columns are missing, if the CRS is geographic (not
        projected), or if geometry is not point-like.
    FileNotFoundError
        If the file does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Vector file not found: {path}")

    kwargs: dict = {}
    if layer is not None:
        kwargs["layer"] = layer

    gdf = gpd.read_file(path, **kwargs)

    missing = set(eca_columns) - set(gdf.columns)
    if missing:
        raise ValueError(f"Vector file is missing required ECa columns: {sorted(missing)}")

    if gdf.crs is None:
        logger.warning("GeoDataFrame has no CRS defined.  Distance calculations may be wrong.")
        crs_str = None
    elif gdf.crs.is_geographic:
        raise ValueError(
            "CRS is geographic (degrees).  Reproject to a projected CRS (e.g. UTM) "
            "before running RSSD — distance calculations require metres."
        )
    else:
        crs_str = gdf.crs.to_wkt()
        logger.info("Loaded %d sites with CRS: %s", len(gdf), gdf.crs.name)

    coords = np.column_stack([gdf.geometry.x.to_numpy(), gdf.geometry.y.to_numpy()])
    eca = gdf[eca_columns].copy()
    return eca, coords, crs_str


def read_em_survey(
    path: str | Path,
    eca_columns: list[str],
    *,
    x_col: str = "x",
    y_col: str = "y",
    site_id_col: str = "site_id",
    format_hint: Literal["txt", "xyz", "dat"] | None = None,
    profile: str | None = None,
    delimiter: str | None = None,
    has_header: bool | None = None,
    column_names: list[str] | None = None,
    crs: str | None = None,
    require_projected_crs: bool = False,
    encoding: str = "utf-8",
) -> tuple[pd.DataFrame, np.ndarray, dict[str, str | int | None]]:
    """Parse EM survey files and normalize to RSSD canonical columns.

    Parameters
    ----------
    path : str | Path
        Input EM survey file path.
    eca_columns : list[str]
        ECa channel names to keep in the normalized frame.
    x_col : str
        X/easting column name in the parsed table.
    y_col : str
        Y/northing column name in the parsed table.
    site_id_col : str
        Site identifier column name in the normalized table.
    format_hint : {'txt', 'xyz', 'dat'} | None
        Optional parser hint when extension is ambiguous.
    profile : str | None
        Optional parser profile for known vendor formats (e.g. ``"veris"``).
    delimiter : str | None
        Optional explicit delimiter override.
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
        If parser options are inconsistent or required columns are missing.
    FileNotFoundError
        If ``path`` does not exist.
    """
    source_path = Path(path)
    if not source_path.exists():
        raise FileNotFoundError(f"EM survey file not found: {source_path}")

    extension = source_path.suffix.lower()
    if format_hint is not None:
        extension = f".{format_hint.lower()}"
    defaults = _EM_EXTENSION_DEFAULTS.get(extension, {})
    if not defaults and profile is None:
        raise ValueError(
            f"Unsupported EM survey extension {source_path.suffix!r}. "
            "Use format_hint or profile for custom parsing."
        )

    profile_settings: dict[str, str | list[str] | bool] = (
        _DELIMITED_PROFILES.get(profile, {}) if profile is not None else {}
    )
    parse_delimiter = delimiter or profile_settings.get("delimiter") or defaults.get("delimiter")
    parse_has_header = (
        has_header
        if has_header is not None
        else profile_settings.get("has_header", defaults.get("has_header"))
    )
    parse_column_names: list[str] | None = column_names
    if parse_column_names is None:
        profile_names = profile_settings.get("column_names")
        if isinstance(profile_names, list):
            parse_column_names = [str(name) for name in profile_names]

    if parse_delimiter is None or parse_has_header is None:
        raise ValueError("Could not infer delimiter/header settings; provide explicit options.")
    if not parse_has_header and parse_column_names is None:
        raise ValueError("column_names are required when has_header=False.")

    raw = _read_delimited(
        source_path,
        delimiter=str(parse_delimiter),
        has_header=bool(parse_has_header),
        column_names=parse_column_names,
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
        "profile": profile,
        "n_sites": int(len(canonical)),
        "site_id_col": site_id_col,
        "x_col": x_col,
        "y_col": y_col,
        "crs": crs,
    }
    return eca, coords, metadata
