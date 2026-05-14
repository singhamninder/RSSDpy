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

## Quick start

```python
import pandas as pd
import numpy as np
from rssdpy.features import ECaPCA, detect_outliers
from rssdpy.sampling import central_composite_design, run_rssd
from rssdpy.calibration import fit_mlr_models
from rssdpy.predict import predict_salinity

# 1. Load ECa survey (dS/m, projected CRS)
eca = pd.DataFrame({"EMh": [...], "EMv": [...]})
coords = np.array([[easting, northing], ...])  # metres, UTM

# 2. PCA on log-ECa
pca = ECaPCA(n_components=3)
scores, eigenvalues, _ = pca.fit_transform(eca)

# 3. Detect and remove outliers
mask, distances = detect_outliers(scores, alpha=0.001)
scores_clean = scores[~mask]
coords_clean = coords[~mask]

# 4. Run RSSD site selection
design = central_composite_design(n_components=3, radius_squared=3.84, include_center=False)
result = run_rssd(scores_clean, coords_clean, design, n_extra=2)
print(f"Selected {len(result.selected_indices)} calibration sites")
print(f"Final AD = {result.ad_final:.2f} m")

# 5. Calibrate (after collecting soil cores at selected sites)
cal_indices = result.selected_indices
ln_ece = np.log(ece_measurements)  # measured ECe in dS/m at cal sites
models = fit_mlr_models(scores_clean, coords_clean, cal_indices, ln_ece)
best = models[0]  # ranked by PRESS

# 6. Predict at all survey sites
predictions = predict_salinity(best, scores_clean, coords_clean, cal_indices, ln_ece)
print(f"Field-average ECe estimate: {np.exp(predictions.field_mean):.2f} dS/m")
```

## Development

```bash
uv run ruff format .
uv run ruff check --fix .
uv run ty check .
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

See `docs/theory/rssd-source-notes.md` for a detailed register of confirmed vs.
extended theory claims.
