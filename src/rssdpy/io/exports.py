"""Export utilities for RSSD selected-site products."""

from pathlib import Path

import numpy as np
import pandas as pd

from rssdpy.sampling.rssd import RSSDesign


def selected_sites_table(
    result: RSSDesign,
    coords,
    *,
    site_ids=None,
    design: np.ndarray | None = None,
) -> pd.DataFrame:
    """Build a tabular representation of RSSD selected sites.

    Parameters
    ----------
    result : RSSDesign
        RSSD run output.
    coords : array-like
        Coordinate array of shape ``(N, 2)`` for the same survey used by RSSD.
    site_ids : array-like | None
        Site identifiers aligned to each selected row. If omitted, uses
        ``selected_original_index + 1``.
    design : np.ndarray | None
        Optional CCD design matrix for annotating core design-level coordinates.

    Returns
    -------
    pd.DataFrame
        Table with selected-site rows and export-friendly metadata columns.
    """
    xy = pd.DataFrame(np.asarray(coords, dtype=float), columns=pd.Index(["x", "y"]))
    n_core = len(result.design_level_indices)
    if site_ids is None:
        site_id_values = result.selected_original_indices + 1
    else:
        site_id_values = np.asarray(site_ids)

    selection_types = ["core"] * n_core + ["support"] * len(result.extra_indices)
    selected = pd.DataFrame(
        {
            "selected_index": result.selected_indices,
            "site_id": site_id_values,
            "selected_original_index": result.selected_original_indices,
            "selection_order": range(1, len(result.selected_indices) + 1),
            "selection_type": selection_types,
            "ad_final_m": result.ad_final,
            "opt_criteria": result.opt_criteria,
            "design_factor": result.design_factor,
        }
    )
    selected["x"] = xy.loc[result.selected_indices, "x"].to_numpy()
    selected["y"] = xy.loc[result.selected_indices, "y"].to_numpy()

    if design is not None and n_core > 0:
        design = np.asarray(design, dtype=float)
        for dim in range(design.shape[1]):
            col = f"design_pc{dim + 1}"
            matched = np.full(len(result.selected_indices), np.nan)
            for j in range(min(n_core, len(design))):
                matched[j] = design[j, dim]
            selected[col] = matched
    return selected


def export_selected_sites_csv(
    result: RSSDesign,
    coords,
    path: str | Path,
    *,
    site_ids=None,
    design: np.ndarray | None = None,
) -> Path:
    """Export selected sites to CSV for field crews and QA workflows."""
    output_path = Path(path)
    table = selected_sites_table(result, coords, site_ids=site_ids, design=design)
    table.to_csv(output_path, index=False)
    return output_path


def write_esap_style_report(
    result: RSSDesign,
    path: str | Path,
    *,
    sample_size: int | None = None,
    active_survey_size: int | None = None,
) -> Path:
    """Write a compact human-readable RSSD summary report."""
    output_path = Path(path)
    n_total = sample_size if sample_size is not None else len(result.selected_indices)
    lines = [
        "RSSD Selection Report",
        f"Sample size: {n_total}",
        f"Core sites: {len(result.design_level_indices)}",
        f"Support sites: {len(result.extra_indices)}",
        f"Design factor (D-Factor): {result.design_factor:.2f}",
        f"Opt-Criteria: {result.opt_criteria:.3f}",
        f"AD initial (m): {result.ad_initial:.6f}",
        f"AD final (m): {result.ad_final:.6f}",
        f"Swaps accepted: {result.swap_count}",
    ]
    if active_survey_size is not None:
        lines.insert(2, f"Active survey size: {active_survey_size}")
    lines.extend(
        [
            "",
            "Selected original indices:",
            ", ".join(str(idx) for idx in result.selected_original_indices),
        ]
    )
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
