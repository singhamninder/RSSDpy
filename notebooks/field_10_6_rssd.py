"""Field 10-6 RSSD workflow — load survey, run RSSD, map sites, compare to ESAP."""

import marimo

__generated_with = "0.23.8"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _(mo):
    mo.md("""
    # Field 10-6 RSSD Notebook

    End-to-end **Response Surface Sampling Design** for the USDA Field 10-6 transect
    survey (fall 2016). This notebook:

    1. Loads the comma-delimited EM survey
    2. Runs ESAP σ validation (3.5σ mask / 4.5σ outlier delete)
    3. Selects 12 calibration sites via RSSD (D-Factor 0.96)
    4. Exports results and compares to desktop ESAP reference output
    5. Maps survey and selected sites on a satellite basemap

    ---

    ## How to run (Marimo basics)

    **One-time setup** (from the repo root):

    ```bash
    uv sync --locked --all-extras --dev
    ```

    **Interactive editing** (opens a browser — best for learning):

    ```bash
    uv run marimo edit notebooks/field_10_6_rssd.py
    ```

    **Run as a read-only app:**

    ```bash
    uv run marimo run notebooks/field_10_6_rssd.py
    ```

    **Export static HTML:**

    ```bash
    uv run marimo export html notebooks/field_10_6_rssd.py -o notebooks/field_10_6_rssd.html
    ```

    **Tips:** Each `@app.cell` is reactive — changing an upstream cell re-runs
    downstream cells. Press **Shift+Enter** to run the focused cell in edit mode.
    """)
    return


@app.cell
def _():
    from pathlib import Path

    import contextily as ctx_basemap
    import geopandas as gpd
    import matplotlib.pyplot as plt_mpl
    import numpy as np

    repo_root = Path(__file__).resolve().parents[1]
    data_dir = repo_root / "examples" / "data" / "field_10_6"

    survey_path = data_dir / "101710A_for_esap.txt"
    esap_expected_path = data_dir / "106Frsd1_expected.txt"
    output_csv = data_dir / "selected_sites.csv"
    output_report = data_dir / "rssd_report.txt"

    crs = "EPSG:6339"
    n_components = 2
    target_size = 12
    design_factor = 0.96
    return (
        crs,
        ctx_basemap,
        design_factor,
        esap_expected_path,
        gpd,
        n_components,
        np,
        output_csv,
        output_report,
        plt_mpl,
        survey_path,
        target_size,
    )


@app.cell
def _(crs, mo, np, survey_path):
    from rssdpy.io import read_em_survey

    eca, coords, meta = read_em_survey(
        survey_path,
        eca_columns=["EMv", "EMh"],
        delimiter=",",
        has_header=False,
        column_names=["x", "y", "EMv", "EMh", "row"],
        crs=crs,
        require_projected_crs=True,
    )

    n_sites = len(eca)
    log_emv_mean = float(np.log(eca["EMv"]).mean())
    log_emh_mean = float(np.log(eca["EMh"]).mean())
    x_min, x_max = float(coords[:, 0].min()), float(coords[:, 0].max())
    y_min, y_max = float(coords[:, 1].min()), float(coords[:, 1].max())

    mo.md(
        f"""
        ### Load summary

        | Metric | Value |
        |--------|-------|
        | Sites loaded | {n_sites} |
        | X range (m) | {x_min:.1f} – {x_max:.1f} |
        | Y range (m) | {y_min:.1f} – {y_max:.1f} |
        | mean(ln EMv) | {log_emv_mean:.4f} (ESAP ref ≈ 5.693) |
        | mean(ln EMh) | {log_emh_mean:.4f} (ESAP ref ≈ 5.521) |
        """
    )
    mo.ui.table([{"key": k, "value": v} for k, v in meta.items()])
    mo.ui.table(eca.describe().round(2))
    return coords, eca


@app.cell
def _(eca, mo):
    from rssdpy.features import ECaPCA, iterative_esap_validation

    pca = ECaPCA(n_components=2)
    eca_clean, scores, qc, original_idx = iterative_esap_validation(eca, pca)

    n_masked = int(qc.masking_mask.sum())
    n_eligible = int(qc.eligible_mask.sum())
    n_removed = len(eca) - len(eca_clean)

    mo.md(
        f"""
        ### ESAP σ validation

        | Step | Count |
        |------|-------|
        | Masked (> 3.5σ) | {n_masked} (ESAP ref: 72) |
        | Outliers deleted (> 4.5σ) | {n_removed} (ESAP ref: 1) |
        | Eligible for RSSD | {n_eligible} (ESAP ref: 5029) |
        | Sites after QC | {len(eca_clean)} |
        """
    )
    return n_eligible, original_idx, qc, scores


@app.cell
def _(
    coords,
    design_factor,
    mo,
    n_components,
    original_idx,
    qc,
    scores,
    target_size,
):
    from rssdpy.sampling import esap_sample_plan, esap_sampling_design, run_rssd

    coords_clean = coords[original_idx]

    plan = esap_sample_plan(
        n_components=n_components,
        target_size=target_size,
        design_factor=design_factor,
        design_mode="esap_two_signal",
    )
    design = esap_sampling_design(
        n_components,
        design_mode="esap_two_signal",
        design_factor=design_factor,
    )
    result = run_rssd(
        scores,
        coords_clean,
        design,
        n_extra=plan.n_extra,
        extra_mode="cube",
        eligible_mask=qc.eligible_mask,
        original_indices=original_idx,
        design_factor=design_factor,
        opt_criteria_mode="esap",
    )

    selected_site_ids = sorted((result.selected_original_indices + 1).tolist())

    mo.md(
        f"""
        ### RSSD results

        | Metric | Value |
        |--------|-------|
        | Core CCD levels | {len(result.design_level_indices)} (ESAP 2-signal template) |
        | Support sites | {len(result.extra_indices)} |
        | Total selected | {len(result.selected_indices)} |
        | D-Factor | {result.design_factor:.2f} |
        | AD initial (m) | {result.ad_initial:.2f} |
        | AD final (m) | {result.ad_final:.2f} |
        | Opt-Criteria | {result.opt_criteria:.3f} (ESAP ref ≈ 1.26) |
        | Swaps accepted | {result.swap_count} |

        **Selected site IDs:** {selected_site_ids}
        """
    )
    return coords_clean, design, result, selected_site_ids


@app.cell
def _(coords_clean, design, mo, n_eligible, output_csv, output_report, result):
    from rssdpy.io import (
        export_selected_sites_csv,
        selected_sites_table,
        write_esap_style_report,
    )

    table_sorted = (
        selected_sites_table(result, coords_clean, design=design)
        .sort_values("selection_order")
        .reset_index(drop=True)
    )

    export_selected_sites_csv(result, coords_clean, output_csv, design=design)
    write_esap_style_report(
        result,
        output_report,
        sample_size=len(result.selected_indices),
        active_survey_size=n_eligible,
    )

    mo.md(
        f"""
        ### Exported outputs

        - CSV: `{output_csv}`
        - Report: `{output_report}`
        """
    )
    mo.ui.table(table_sorted)
    return


@app.cell
def _(esap_expected_path, mo, selected_site_ids):
    import re

    def parse_esap_site_ids(path) -> list[int]:
        """Extract site IDs from the ESAP 'Target Information' block."""
        text = path.read_text(encoding="utf-8")
        in_target = False
        site_ids: list[int] = []
        for line in text.splitlines():
            if "Target Information" in line:
                in_target = True
                continue
            if in_target and "Ordered Listing" in line:
                break
            if not in_target:
                continue
            match = re.match(r"^\s*(\d+)\s+", line)
            if match and "Site ID" not in line:
                site_ids.append(int(match.group(1)))
        return site_ids

    expected_site_ids = sorted(parse_esap_site_ids(esap_expected_path))
    our_ids = sorted(selected_site_ids)
    matches = set(our_ids) & set(expected_site_ids)
    only_ours = sorted(set(our_ids) - set(expected_site_ids))
    only_esap = sorted(set(expected_site_ids) - set(our_ids))

    comparison_rows = [
        {"source": "RSSDpy", "site_ids": ", ".join(str(i) for i in our_ids)},
        {
            "source": "ESAP (106Frsd1)",
            "site_ids": ", ".join(str(i) for i in expected_site_ids),
        },
    ]

    mo.md(
        f"""
        ### Comparison to ESAP reference

        | Check | Result |
        |-------|--------|
        | Matches | {len(matches)} / {len(expected_site_ids)} |
        | Only in RSSDpy | {only_ours or "—"} |
        | Only in ESAP | {only_esap or "—"} |

        Full ESAP parity is a regression goal. Tie-breaking or implementation
        details may produce a different site set while the underlying algorithm
        (CCD matching + AD swapping) remains the same.
        """
    )
    mo.ui.table(comparison_rows)
    return


@app.cell
def _(
    coords,
    crs,
    ctx_basemap,
    gpd,
    mo,
    np,
    original_idx,
    plt_mpl,
    qc,
    result,
):
    n_total = len(coords)
    site_ids = np.arange(1, n_total + 1)
    selected_original_idx = result.selected_original_indices

    eligible_full = np.zeros(n_total, dtype=bool)
    eligible_full[original_idx] = qc.eligible_mask
    masked_full = ~eligible_full

    gdf = gpd.GeoDataFrame(
        {
            "site_id": site_ids,
            "masked": masked_full,
            "selected": np.isin(np.arange(n_total), selected_original_idx),
        },
        geometry=gpd.points_from_xy(coords[:, 0], coords[:, 1]),
        crs=crs,
    )
    gdf_web = gdf.to_crs(3857)

    fig_full, ax_full = plt_mpl.subplots(figsize=(10, 8))
    background = gdf_web[~gdf_web["selected"] & ~gdf_web["masked"]]
    masked_pts = gdf_web[gdf_web["masked"]]
    selected_pts = gdf_web[gdf_web["selected"]]

    if len(background):
        background.plot(ax=ax_full, color="#cccccc", markersize=1, alpha=0.5, label="Survey")
    if len(masked_pts):
        masked_pts.plot(
            ax=ax_full, color="#f39c12", markersize=2, alpha=0.6, label="Masked (>3.5σ)"
        )
    if len(selected_pts):
        selected_pts.plot(ax=ax_full, color="#e74c3c", markersize=40, label="Selected", zorder=5)

    for x_full, y_full, sid_full in zip(
        selected_pts.geometry.x,
        selected_pts.geometry.y,
        selected_pts["site_id"],
        strict=True,
    ):
        ax_full.annotate(
            str(int(sid_full)),
            (x_full, y_full),
            fontsize=7,
            ha="center",
            va="bottom",
            xytext=(0, 4),
            textcoords="offset points",
        )

    ctx_basemap.add_basemap(
        ax_full,
        source=ctx_basemap.providers.Esri.WorldImagery,
        attribution=False,
    )
    ax_full.set_axis_off()
    ax_full.set_title("Field 10-6 — full survey and RSSD sites")
    ax_full.legend(loc="upper right", fontsize=8)
    plt_mpl.tight_layout()

    mo.vstack([mo.md("### Map — full survey"), fig_full])
    return


@app.cell
def _(mo, plt_mpl, qc, result, scores):
    from rssdpy.viz import plot_pc_scatter

    fig_pc, ax_pc = plt_mpl.subplots(figsize=(6, 5))
    plot_pc_scatter(
        scores,
        outlier_mask=~qc.eligible_mask,
        selected_indices=result.selected_indices,
        ax=ax_pc,
    )
    ax_pc.set_title("PC1 vs PC2 — QC and selected sites")
    plt_mpl.tight_layout()

    mo.vstack([mo.md("### PC scatter (QC diagnostic)"), fig_pc])
    return


if __name__ == "__main__":
    app.run()
