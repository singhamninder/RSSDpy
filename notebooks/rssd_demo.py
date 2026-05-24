"""RSSD demo — load an EMI survey, run ESAP-style RSSD, export and map selected sites."""

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
    # RSSD demo

    End-to-end **Response Surface Sampling Design** for a 2-signal EMI transect survey.
    This notebook follows the ESAP-RSSD workflow:

    1. Load survey data (projected coordinates + EM channels)
    2. Natural-log transform, standardize, and PCA on ECa
    3. ESAP σ validation (mask > 3.5σ, delete outliers > 4.5σ)
    4. Match design levels in PC space and swap for spatial uniformity (AD)
    5. Export selected calibration sites and review maps

    Adjust the controls below (survey path, EPSG, sample size), then scroll through
    interfaces for load summary, QC, RSSD metrics, exports, and plots.

    ---

    ## How to run

    ```bash
    uv sync --locked --all-extras --dev
    uv run marimo edit notebooks/rssd_demo.py   # interactive
    uv run marimo run notebooks/rssd_demo.py    # read-only app
    uv run marimo export html notebooks/rssd_demo.py -o notebooks/rssd_demo.html
    ```
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
    default_survey_path = repo_root / "examples" / "data" / "field_10_6" / "101710A_for_esap.txt"

    n_components = 2
    design_factor = 0.96
    return (
        Path,
        ctx_basemap,
        default_survey_path,
        design_factor,
        gpd,
        n_components,
        np,
        plt_mpl,
        repo_root,
    )


@app.cell
def _(default_survey_path, mo):
    survey_path_ctrl = mo.ui.text(
        value=str(default_survey_path),
        label="Survey file path",
        full_width=True,
    )
    crs_ctrl = mo.ui.text(value="EPSG:6339", label="EPSG code")
    target_size_ctrl = mo.ui.number(
        start=6,
        stop=20,
        value=12,
        step=1,
        label="Calibration sites (n)",
    )

    mo.vstack(
        [
            mo.md("### Settings"),
            survey_path_ctrl,
            crs_ctrl,
            target_size_ctrl,
        ]
    )
    return crs_ctrl, survey_path_ctrl, target_size_ctrl


@app.cell
def _(Path, crs_ctrl, mo, np, survey_path_ctrl):
    from rssdpy.io import read_em_survey

    survey_path = Path(survey_path_ctrl.value)
    crs = crs_ctrl.value.strip()

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

    mo.vstack(
        [
            mo.md(
                f"""
                ### Load summary

                | Metric | Value |
                |--------|-------|
                | Survey file | `{survey_path.name}` |
                | CRS | {crs} |
                | Sites loaded | {n_sites} |
                | X range (m) | {x_min:.1f} – {x_max:.1f} |
                | Y range (m) | {y_min:.1f} – {y_max:.1f} |
                | mean(ln EMv) | {log_emv_mean:.4f} |
                | mean(ln EMh) | {log_emh_mean:.4f} |
                """
            ),
            mo.ui.table([{"key": k, "value": v} for k, v in meta.items()]),
            mo.ui.table(eca.head(10)),
            mo.ui.table(eca.describe().round(2)),
        ]
    )
    return coords, crs, eca, survey_path


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

        Natural-log ECa → center/scale → PCA; sites beyond 3.5σ are masked and
        beyond 4.5σ are removed before RSSD.

        | Step | Count |
        |------|-------|
        | Masked (> 3.5σ) | {n_masked} |
        | Outliers deleted (> 4.5σ) | {n_removed} |
        | Eligible for RSSD | {n_eligible} |
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
    target_size_ctrl,
):
    from rssdpy.sampling import esap_sample_plan, esap_sampling_design, run_rssd

    target_size = int(target_size_ctrl.value)
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
        | Target sample size | {target_size} |
        | Core design levels | {len(result.design_level_indices)} |
        | Support sites | {len(result.extra_indices)} |
        | Total selected | {len(result.selected_indices)} |
        | D-Factor | {result.design_factor:.2f} |
        | AD initial (m) | {result.ad_initial:.2f} |
        | AD final (m) | {result.ad_final:.2f} |
        | Opt-Criteria | {result.opt_criteria:.3f} |
        | Swaps accepted | {result.swap_count} |

        **Selected site IDs:** {selected_site_ids}
        """
    )
    return coords_clean, design, result


@app.cell
def _(coords_clean, design, mo, n_eligible, result, survey_path):
    from rssdpy.io import (
        export_selected_sites_csv,
        selected_sites_table,
        write_esap_style_report,
    )

    output_csv = survey_path.parent / "selected_sites.csv"
    output_report = survey_path.parent / "rssd_report.txt"

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

    mo.vstack(
        [
            mo.md(
                f"""
                ### Selected calibration sites

                - CSV: `{output_csv}`
                - Report: `{output_report}`
                """
            ),
            mo.ui.table(table_sorted),
        ]
    )
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
    ax_full.set_title("Survey and RSSD-selected calibration sites")
    ax_full.legend(loc="upper right", fontsize=8)
    plt_mpl.tight_layout()

    mo.vstack([mo.md("### Map"), fig_full])
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

    mo.vstack([mo.md("### PC scatter"), fig_pc])
    return


if __name__ == "__main__":
    app.run()
