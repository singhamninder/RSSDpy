"""CSV and GeoDataFrame loaders with boundary validation.

Validates required columns and projected CRS at load time (system boundary).
Does not validate between internal functions (project coding standard).
"""

from __future__ import annotations

import logging
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Columns that every ECa survey must contain after loading
_REQUIRED_ECA_COLS: set[str] = {"x", "y"}


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
