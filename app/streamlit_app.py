"""RSSDpy interactive app: upload an EMI survey, run ESAP-faithful RSSD site selection.

Run locally with:  uv run --extra app streamlit run app/streamlit_app.py
"""

import io

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from pyproj import CRS, Transformer

from rssdpy import run_esap_rssd
from rssdpy.io.exports import selected_sites_table
from rssdpy.io.loaders import validate_canonical_survey
from rssdpy.viz import plot_pc_scatter

st.set_page_config(page_title="RSSDpy", layout="wide")
st.title("RSSDpy: ESAP-RSSD calibration-site selection")
st.caption(
    "Upload a 2-signal EMI survey (CSV/TXT), map its columns, and run the "
    "ESAP-faithful Response Surface Sampling Design (Lesch et al. 1995)."
)

uploaded = st.file_uploader("Survey file", type=["csv", "txt"])
if uploaded is None:
    st.info("Upload a survey file to begin.")
    st.stop()

col1, col2 = st.columns(2)
with col1:
    has_header = st.checkbox("File has a header row", value=uploaded.name.lower().endswith(".csv"))
with col2:
    delimiter = "," if st.selectbox("Delimiter", ["Comma", "Whitespace"]) == "Comma" else r"\s+"

try:
    preview = pd.read_csv(
        io.BytesIO(uploaded.getvalue()),
        sep=delimiter,
        header=0 if has_header else None,
        engine="python",
    )
except Exception as exc:  # user-supplied file: any parse failure is their input, not a bug
    st.error(f"Could not parse file: {exc}")
    st.stop()
if not has_header:
    preview.columns = [f"col_{i}" for i in range(preview.shape[1])]

st.write(f"Preview ({len(preview)} rows):")
st.dataframe(preview.head(10))
columns = list(preview.columns)

with st.form("rssd_config"):
    st.subheader("Column mapping")
    c1, c2, c3 = st.columns(3)
    x_col = c1.selectbox("X / easting", columns, index=0)
    y_col = c2.selectbox("Y / northing", columns, index=min(1, len(columns) - 1))
    site_id_col = c3.selectbox("Site ID (optional)", ["(auto: row number)", *columns])
    eca_columns = st.multiselect(
        "ECa channels — exactly 2 (the library's ESAP-parity design template requires 2 signals)",
        columns,
        default=columns[2:4] if len(columns) >= 4 else [],
    )
    epsg = st.text_input(
        "Projected CRS", value="EPSG:6339", help="EPSG code; must be projected (metres)."
    )

    st.subheader("RSSD parameters")
    p1, p2, p3, p4 = st.columns(4)
    target_size = p1.selectbox("Target sample size", [6, 12, 20], index=1)
    design_factor = p2.slider("D-Factor", 0.90, 1.10, 1.00, 0.01)
    masking_std = p3.number_input("Masking sigma", value=3.5, step=0.1)
    outlier_std = p4.number_input("Outlier sigma", value=4.5, step=0.1)

    submitted = st.form_submit_button("Run RSSD")

if not submitted:
    st.stop()

if len(eca_columns) != 2:
    st.error("Select exactly 2 ECa channel columns.")
    st.stop()
try:
    if not CRS.from_user_input(epsg).is_projected:
        st.error(f"{epsg} is not a projected CRS; RSSD distances require metres.")
        st.stop()
except Exception as exc:  # user-typed CRS string: pyproj raises several error types
    st.error(f"Invalid CRS {epsg!r}: {exc}")
    st.stop()

work = preview.copy()
if site_id_col == "(auto: row number)":
    work.insert(0, "site_id", np.arange(1, len(work) + 1))
else:
    work = work.rename(columns={site_id_col: "site_id"})

try:
    canonical = validate_canonical_survey(
        work,
        eca_columns=eca_columns,
        x_col=x_col,
        y_col=y_col,
        crs=epsg,
        require_projected_crs=True,
    )
    run = run_esap_rssd(
        canonical[eca_columns],
        canonical[[x_col, y_col]].to_numpy(dtype=float),
        n_components=2,
        target_size=target_size,
        design_factor=design_factor,
        design_mode="esap_two_signal",
        masking_std=masking_std,
        outlier_std=outlier_std,
    )
except ValueError as exc:
    st.error(f"RSSD run failed: {exc}")
    st.stop()

st.success(f"Selected {len(run.rssd.selected_indices)} calibration sites.")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Masked (> sigma)", int(run.qc.masking_mask.sum()))
m2.metric("Eligible sites", int(run.qc.eligible_mask.sum()))
m3.metric("AD final (m)", f"{run.rssd.ad_final:.1f}")
m4.metric("Opt-Criteria", f"{run.rssd.opt_criteria:.3f}")

table = selected_sites_table(run.rssd, run.coords_clean, design=run.design).sort_values(
    "selection_order"
)
st.dataframe(table)
st.download_button(
    "Download selected sites (CSV)",
    table.to_csv(index=False),
    file_name="selected_sites.csv",
    mime="text/csv",
)

left, right = st.columns(2)
with left:
    st.subheader("PC-space diagnostics")
    fig, ax = plt.subplots(figsize=(5, 5))
    plot_pc_scatter(
        run.scores,
        outlier_mask=~run.qc.eligible_mask,
        selected_indices=run.rssd.selected_indices,
        ax=ax,
    )
    st.pyplot(fig)

with right:
    st.subheader("Map")
    transformer = Transformer.from_crs(epsg, "EPSG:4326", always_xy=True)
    lon, lat = transformer.transform(run.coords_clean[:, 0], run.coords_clean[:, 1])
    selected_mask = np.zeros(len(run.coords_clean), dtype=bool)
    selected_mask[run.rssd.selected_indices] = True
    map_df = pd.DataFrame(
        {
            "lat": lat,
            "lon": lon,
            "color": np.where(selected_mask, "#e74c3c", "#999999"),
            "size": np.where(selected_mask, 30, 5),
        }
    )
    st.map(map_df, latitude="lat", longitude="lon", color="color", size="size")
