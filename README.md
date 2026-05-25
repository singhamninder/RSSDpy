# RSSDpy

A Python library implementing the **Response Surface Sampling Design (RSSD)** for
ECa-directed soil salinity sampling, faithful to the ESAP-RSSD methodology of
Lesch, Strauss, and Rhoades (1995) and the USDA-ARS ESAP software.

## What it does

RSSDpy selects a minimal set of calibration sites (typically 6–20) from a large
electromagnetic induction (EMI) survey, so that:

1. The sites are **statistically optimal** for fitting a multiple linear regression
   (MLR) model of `ln(ECe)` from `ln(ECa)` principal components.
2. The sites are **spatially representative** of the entire field area.

This is achieved by matching survey sites to a Central Composite Design (CCD) in
PC-score space, then iteratively swapping candidates to minimise the average
nearest-calibration distance (AD).

## Scope and limitations

- Core algorithms are faithful to the ESAP-RSSD methodology (`[P2]` Lesch et al. 1995;
  `[L05]` Lesch 2005).
- ESAP handles 1 or 2 EMI signal readings; the 3-component CCD extension matches
  the Lesch 1995 Paper 2 description.
- 4-component variants, simulated annealing, cLHS, and DPPC deterministic models are
  clearly labelled as **Extensions** — see `docs/theory/rssd-source-notes.md`.
- This library predicts at the survey sites **only**; raster interpolation requires
  an external kriging or IDW step.

## Installation

```bash
# Using uv (recommended)
uv add rssdpy

# Development install
git clone <repo-url>
cd RSSDpy
uv sync --locked --all-extras --dev
```

## Canonical survey schema

RSSDpy uses a single tabular schema loaded from **CSV** or whitespace-delimited **TXT**
(parsed to the same column layout). Required columns:

| Column | Notes |
|--------|--------|
| `site_id` | Stable integer site identifier |
| `x`, `y` | Projected coordinates (metres) |
| `EMh`, `EMv` (or your channel names) | Positive ECa in dS/m |
| `row` | Optional; transect surveys only |

Legacy ESAP files (`.svy`, `.pro`) are not required at runtime; use them only for
regression checks against desktop ESAP output. Convert vendor exports to CSV or TXT
before loading.

## Quick start

```python
import numpy as np
from rssdpy.features import ECaPCA, detect_outliers_esap, iterative_esap_validation
from rssdpy.io import export_selected_sites_csv, read_em_survey
from rssdpy.sampling import esap_sample_plan, central_composite_design, run_rssd
from rssdpy.calibration import fit_mlr_models
from rssdpy.predict import predict_salinity

# 1. Parse raw EM survey (ESAP transect example: site_id, x, y, EMv, EMh, row)
eca, coords, _meta = read_em_survey(
    "survey.txt",
    eca_columns=["EMv", "EMh"],
    has_header=False,
    column_names=["site_id", "x", "y", "EMv", "EMh", "row"],
    crs="EPSG:32611",
    require_projected_crs=True,
)

# 2. PCA + ESAP σ validation (mask 3.5σ, delete outliers > 4.5σ)
pca = ECaPCA(n_components=2)
eca_clean, scores, qc, original_idx = iterative_esap_validation(eca, pca)
coords_clean = coords[original_idx]

# 3. ESAP-style SRS design (n=12, D-Factor 1)
design_factor = 1
plan = esap_sample_plan(n_components=2, target_size=12, design_factor=design_factor)
design = central_composite_design(
    n_components=2, include_center=False, design_factor=design_factor
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
)
print(f"Selected {len(result.selected_indices)} sites")
print(f"Opt-Criteria = {result.opt_criteria:.3f}  (AD = {result.ad_final:.1f} m)")
export_selected_sites_csv(result, coords_clean, "selected_sites.csv", design=design)

```

## Demo notebook (Jupyter)

The Jupyter demo walks through the RSSD workflow using local file paths and plain
Python cells.

1. Open `notebooks/rssd_demo.ipynb`
2. Edit the config cell (`survey_path`, `crs`, `target_size`, `design_factor`)
3. Run cells top-to-bottom for load summary, ESAP QC, RSSD selection, CSV export,
   map, and PC diagnostics

## Development

```bash
uv sync --locked --all-extras --dev
uv run pre-commit install          # one-time per clone
uv run pre-commit run --all-files  # optional: verify before first commit
```

Each commit runs ruff (lint + format) and ty via pre-commit. Run tests manually:

```bash
uv run pytest
```

## References

- Lesch, S.M., Strauss, D.J., and Rhoades, J.D. (1995). Spatial Prediction of Soil
  Salinity Using Electromagnetic Induction Techniques: 2. *Water Resources Research*,
  31(2), 387–398. https://doi.org/10.1029/94WR02180
- Lesch, S.M., Strauss, D.J., and Rhoades, J.D. (1995). Spatial Prediction of Soil
  Salinity Using Electromagnetic Induction Techniques: 1. *Water Resources Research*,
  31(2), 373–386. https://doi.org/10.1029/94WR02179
- Lesch, S.M. (2005). Sensor-Directed Response Surface Sampling Designs.
  *Computers and Electronics in Agriculture*, 46, 153–179.
- Corwin, D.L. and Lesch, S.M. (2005). Characterizing Soil Spatial Variability with
  Apparent Soil Electrical Conductivity. *Computers and Electronics in Agriculture*,
  46, 103–133.

