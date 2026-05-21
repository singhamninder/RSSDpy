"""Export utilities for RSSD selected-site products."""

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point

from rssdpy.sampling.rssd import RSSDesign


def selected_sites_table(
    result: RSSDesign,
    coords,
    *,
    site_ids=None,
) -> pd.DataFrame:
    """Build a tabular representation of RSSD selected sites.

    Parameters
    ----------
    result : RSSDesign
        RSSD run output.
    coords : array-like
        Coordinate array of shape ``(N, 2)`` for the same survey used by RSSD.
    site_ids : array-like | None
        Optional stable site identifiers. If omitted, uses original indices + 1.

    Returns
    -------
    pd.DataFrame
        Table with selected-site rows and export-friendly metadata columns.
    """
    xy = pd.DataFrame(np.asarray(coords, dtype=float), columns=pd.Index(["x", "y"]))
    if site_ids is None:
        site_ids = result.selected_original_indices + 1
    selected = pd.DataFrame(
        {
            "selected_index": result.selected_indices,
            "site_id": site_ids,
            "selected_original_index": result.selected_original_indices,
            "selection_order": range(1, len(result.selected_indices) + 1),
            "selection_type": ["core"] * len(result.design_level_indices)
            + ["extra"] * len(result.extra_indices),
        }
    )
    selected["x"] = xy.loc[result.selected_indices, "x"].to_numpy()
    selected["y"] = xy.loc[result.selected_indices, "y"].to_numpy()
    return selected


def export_selected_sites_csv(
    result: RSSDesign,
    coords,
    path: str | Path,
    *,
    site_ids=None,
) -> Path:
    """Export selected sites to CSV for field crews and QA workflows."""
    output_path = Path(path)
    table = selected_sites_table(result, coords, site_ids=site_ids)
    table.to_csv(output_path, index=False)
    return output_path


def export_selected_sites_geopackage(
    result: RSSDesign,
    coords,
    path: str | Path,
    *,
    site_ids=None,
    crs: str = "EPSG:32611",
    layer: str = "rssd_selected",
) -> Path:
    """Export selected sites to GeoPackage for GIS field operations."""
    output_path = Path(path)
    table = selected_sites_table(result, coords, site_ids=site_ids)
    geometry = [Point(xy) for xy in zip(table["x"], table["y"], strict=True)]
    gdf = gpd.GeoDataFrame(table, geometry=geometry, crs=crs)
    gdf.to_file(output_path, layer=layer, driver="GPKG")
    return output_path


def write_esap_style_report(
    result: RSSDesign,
    path: str | Path,
) -> Path:
    """Write a compact human-readable RSSD summary report."""
    output_path = Path(path)
    lines = [
        "RSSD Selection Report",
        f"Core sites: {len(result.design_level_indices)}",
        f"Extra sites: {len(result.extra_indices)}",
        f"Swaps accepted: {result.swap_count}",
        f"AD initial: {result.ad_initial:.6f}",
        f"AD final: {result.ad_final:.6f}",
        "",
        "Selected original indices:",
        ", ".join(str(idx) for idx in result.selected_original_indices),
    ]
    if len(result.validation_original_indices):
        lines.extend(
            [
                "",
                "Validation original indices:",
                ", ".join(str(idx) for idx in result.validation_original_indices),
            ]
        )
    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path
