# RSSD Source Notes

This document registers each theory claim, formula, and constant used in RSSDpy,
labelled as one of:

- **Confirmed** — directly stated or demonstrated in a primary/authoritative source.
- **Derived** — mathematically derivable from a confirmed claim with standard algebra or
  linear algebra identities; no independent citation needed.
- **Extension** — a generalisation or adaptation not guaranteed to reproduce the original
  ESAP software behaviour. Must be clearly named, documented, and tested as separate from
  ESAP-faithful code paths.

Primary sources:

| Key | Full reference |
|-----|----------------|
| `[P2]` | Lesch, S.M., Strauss, D.J., and Rhoades, J.D. (1995). Spatial Prediction of Soil Salinity Using Electromagnetic Induction Techniques: 2. An Efficient Spatial Sampling Algorithm Suitable for Multiple Linear Regression Model Identification and Estimation. *Water Resources Research*, 31(2), 387–398. https://doi.org/10.1029/94WR02180 |
| `[P1]` | Lesch, S.M., Strauss, D.J., and Rhoades, J.D. (1995). Spatial Prediction of Soil Salinity Using Electromagnetic Induction Techniques: 1. Statistical Prediction Models: A Comparison of Multiple Linear Regression and Cokriging. *Water Resources Research*, 31(2), 373–386. https://doi.org/10.1029/94WR02179 |
| `[L05]` | Lesch, S.M. (2005). Sensor-Directed Response Surface Sampling Designs for Characterizing Spatial Variation in Soil Properties. *Computers and Electronics in Agriculture*, 46, 153–179. |
| `[CL05]` | Corwin, D.L. and Lesch, S.M. (2005). Characterizing Soil Spatial Variability with Apparent Soil Electrical Conductivity: Part I. Survey Protocols. *Computers and Electronics in Agriculture*, 46, 103–133. |
| `[R89]` | Rhoades, J.D., Manteghi, N.A., Shouse, P.J., and Alves, W.J. (1989). Soil Electrical Conductivity and Soil Salinity: New Formulations and Calibrations. *Soil Science Society of America Journal*, 53(2), 433–439. https://doi.org/10.2136/sssaj1989.03615995005300020020x |
| `[ESAP]` | USDA ARS Salinity Laboratory. ESAP-RSSD software description. https://www.ars.usda.gov/pacific-west-area/riverside-ca/agricultural-water-efficiency-and-salinity-research-unit/docs/model/esap-model/ |
| `[SDK24]` | Banik, J. (2024). Sensor-Directed Sampling. UC Riverside MS Thesis. https://github.com/jayanta-banik/Sensor-Directed-Sampling |

---

## 1. EMI Measurement & Physical Foundation

| Claim | Label | Source | Notes |
|-------|-------|--------|-------|
| EM-38 horizontal mode effective depth ≈ 0–0.75 m | Confirmed | `[CL05]`, `[L05]` | |
| EM-38 vertical mode effective depth ≈ 0–1.5 m | Confirmed | `[CL05]`, `[L05]` | |
| ECa is the depth-integral of σ(z) weighted by instrument response φ(z) | Confirmed | `[CL05]` | Forward model identity |
| EMh/EMv ratio > 1 → upward salt accumulation | Confirmed | `[CL05]` | Heuristic interpretation |
| DPPC model relates ECa to ECe, θ_w, SP, ρ_b, T via two parallel pathways | Confirmed | `[R89]` | |

---

## 2. PCA Pre-processing

| Claim | Label | Source | Notes |
|-------|-------|--------|-------|
| Log-transform ECa before PCA (natural log) | Confirmed | `[P2]`, `[L05]` | Makes distributions more symmetric |
| Center and scale each log-ECa variable before PCA | Confirmed | `[P2]` | |
| Divide raw PCA scores by `sqrt(eigenvalue)` to obtain standardised scores | Confirmed | `[P2]`, `[L05]` | Each standardised score has mean 0 and variance 1 |
| PC1 typically represents overall ECa magnitude | Confirmed | `[P2]` | Interpretive label; may differ for specific datasets |
| PC2 represents shallow/deep contrast | Confirmed | `[P2]` | |

---

## 3. Outlier Detection

| Claim | Label | Source | Notes |
|-------|-------|--------|-------|
| Flag sites where `d = sqrt(Σ κᵢ²) > threshold` | Confirmed | `[P2]` | Mahalanobis-equivalent using standardised PC scores |
| Default threshold 4.03 cited in documentation notes for 3 PCs | **Needs verification** | Derived from chi-square? | `sqrt(chi2.ppf(0.999, df=3)) ≈ 4.03`. Must verify df in source. |
| Threshold is `sqrt(chi2.ppf(1 - alpha, df=n_components))` | Derived | Standard chi-square result | `alpha=0.001` by convention; df depends on n_components used |
| ESAP masking at 3.5σ and outliers at 4.5σ on PC distance | Confirmed | `[ESAP]` manual §3.4.2 | Same distance `d = sqrt(Σ κ²)` on standardised scores |
| Chi-square mode (`detect_outliers`) vs ESAP σ mode (`detect_outliers_esap`) | Extension / Derived | Both use PC distance; different cutoffs | Prefer σ mode for ESAP regression parity |

### Threshold derivation note

The value `sqrt(chi2.ppf(0.999, df=3)) ≈ 4.03` comes from assuming the standardised PC
scores are approximately i.i.d. standard normal, so their sum of squares follows chi-squared
with df = number of components. The code must compute this from `scipy.stats.chi2` with the
correct df rather than hardcoding 4.03.

---

## 4. Central Composite Design (CCD)

| Claim | Label | Source | Notes |
|-------|-------|--------|-------|
| ESAP uses 1 or 2 signal readings per site | Confirmed | `[ESAP]` | "1 or 2 signal readings per survey site" |
| 3-component CCD without center → 14 levels (6 axial + 8 cube) | Confirmed | `[P2]` | Core of the ESAP-RSSD algorithm |
| Axial points: `(±α, 0, 0)`, `(0, ±α, 0)`, `(0, 0, ±α)` where `α = sqrt(radius_squared)` | Confirmed | `[P2]` | |
| Cube points: all `(±c, ±c, ±c)` where `c = sqrt(radius_squared / n_components)` | Confirmed | `[P2]` | For 3 components: `c = sqrt(3.84/3) ≈ 1.131` |
| Design constant `radius_squared = 3.84` | Confirmed | `[P2]`, `[L05]` | See note below |
| ESAP design factor 0.90–1.10 scales all CCD levels | Confirmed | `[ESAP]` manual §3.5.2 | Implemented as `design *= design_factor` |
| 2-component CCD without center → 8 levels | Extension | Generic CCD | Use `design_mode="esap_two_signal"` for ESAP 2-signal SRS (10 levels) |
| ESAP 2-signal SRS template (±2.5, ±1.75, ±0.75 at D=1) | Confirmed | `[ESAP]` `106Frsd1.txt` | 10 core levels; 2 support for n=12 |
| 4-component CCD variants | Extension | — | Not present in `[P2]` or `[ESAP]` |

### The `3.84` design constant

`3.84` is used by Lesch et al. as the radius-squared of the CCD ellipsoid. It equals:

```python
from scipy.stats import chi2, norm
chi2.ppf(0.95, df=1)   # 3.8415 ≈ 3.84 — chi-squared with 1 df
norm.ppf(0.975)**2     # 1.96² = 3.8416 ≈ 3.84 — squared z-score at 97.5th percentile
```

It does **NOT** equal `chi2.ppf(0.95, df=3)` ≈ 7.815. The 3D chi-square 95% quantile is
a different value. The 3.84 constant comes from the univariate chi-square (or equivalently,
the squared critical z-value for a two-sided 95% interval). Its use in the 3D CCD means
the design points are NOT on the 95% probability contour of a trivariate normal; they are
within it. The interpretation in the document context is that 3.84 defines the scale factor
for the design, not a chi-square probability statement in 3D.

---

## 5. PRS Candidate Selection

| Claim | Label | Source | Notes |
|-------|-------|--------|-------|
| DLS_ij = Σ (κ_ik − T_jk)² for site i and design level j | Confirmed | `[P2]` | Squared Euclidean distance in standardised PC space |
| Select top-3 closest survey sites per design level → ψ₁, ψ₂, ψ₃ | Confirmed | `[P2]` | |
| No site may appear in more than one ψ set | Confirmed | `[P2]` | Uniqueness constraint |
| No site matched to more than one design level | Confirmed | `[P2]` | |

---

## 6. Spatial Uniformity and Swapping

| Claim | Label | Source | Notes |
|-------|-------|--------|-------|
| `AD(β) = (1/N) Σ min_{j∈β} d(site_i, site_j)` (average nearest-calibration distance) | Confirmed | `[P2]` | Physical (geographic) Euclidean distance, not PC-space distance |
| Initialise β = ψ₁ | Confirmed | `[P2]` | |
| For each design level try ψ₂ replacement; keep if AD decreases | Confirmed | `[P2]` | |
| For each design level try ψ₃ replacement; keep if AD decreases | Confirmed | `[P2]` | |
| Iterate swaps until convergence (no AD reduction) | Confirmed | `[P2]` | |
| Add extra sites greedily (maximise AD reduction) up to target n₀ | Confirmed | `[P2]` | |
| Extra-site pool is the 8 cube design levels using remaining N − n₀ sites | Confirmed | `[P2]` | For monitoring/validation sites |
| AD is theoretically minimised by equilateral triangular grid spacing | Confirmed | `[P2]` (cites McBratney et al. 1981) | Theoretical lower bound |
| ESAP Opt-Criteria is a dimensionless uniformity index | Confirmed | `[ESAP]` manual §3.5.2 | Lower is better; ≤1.30 “reasonable” on rectangular fields |
| `opt_criteria ≈ AD / characteristic_spacing` | Derived | Extension | Cross-field normalisation |
| `opt_criteria_esap ≈ 3 × AD / (spacing × sqrt(N / n_cal))` | Derived | Calibrated vs `106Frsd1.txt` | ESAP desktop parity |
| `characteristic_spacing = sqrt(bbox_area / N)` | Derived | Extension | Metres; normalises AD across field sizes |

---

## 7. MLR Calibration

| Claim | Label | Source | Notes |
|-------|-------|--------|-------|
| Dependent variable: `ln(ECe)` | Confirmed | `[P1]`, `[P2]` | |
| Independent variables: standardised PC scores + optional interaction κ₁κ₂ + optional trend surface | Confirmed | `[P1]`, `[P2]`, `[L05]` | |
| "Spatially homogeneous" model: no trend surface terms | Confirmed | `[L05]` | Appropriate when texture/moisture are uniform |
| 50 candidate model parameterizations (5 PC subsets × 10 trend subsets) | Confirmed | `[P2]` | |
| PRESS = Σ (e_i / (1 − h_i))² where h_i is the leverage | Confirmed | `[P2]` (cites Myers 1986) | Leave-one-out CV statistic |
| APVE = s² × mean(1 + x_j'(X'X)⁻¹x_j) over prediction sites | Confirmed | `[P2]` | Average prediction variance |
| Select model minimising both PRESS and APVE | Confirmed | `[P2]` | Use residual diagnostics to break ties |

---

## 8. Residual Diagnostics

| Claim | Label | Source | Notes |
|-------|-------|--------|-------|
| Shapiro-Wilk normality test on residuals | Confirmed | `[P2]` | |
| Modified Moran test statistic I_M on residuals | Confirmed | `[P1]`, `[P2]` | W matrix: inverse-distance-squared, row-normalised |
| Lack-of-fit F test using duplicate soil cores | Confirmed | `[P1]`, `[P2]` | Requires duplicate samples at ≥ 4–6 calibration sites |
| F = [m/(m−p−1)] × (1 + I_M) / (1 − I_M) | Confirmed | `[P2]` | Combines lack-of-fit and Moran statistic |

---

## 9. Spatial Prediction

| Claim | Label | Source | Notes |
|-------|-------|--------|-------|
| Point prediction: ŷ_j = b' x_j | Confirmed | `[P1]` | Standard OLS prediction |
| Prediction variance: v_j² = s²(1 + x_j'(X'X)⁻¹x_j) | Confirmed | `[P1]` | |
| Conditional probability via t-distribution with n−p−1 df | Confirmed | `[P1]` | |
| Field average: G = (n/N)ȳ_cal + ((N−n)/N)Ḡ_pred | Confirmed | `[P1]` | |
| Range interval proportions using individual P(a ≤ ECe ≤ b) | Confirmed | `[P1]` | Adjusts for regression "shrinkage" |
| Temporal change detection via t-test comparing re-survey to model | Confirmed | `[P1]` | |

---

## 10. Extensions (not in original ESAP)

These are clearly labelled extensions that may be implemented in later phases:

| Extension | Basis | Reference |
|-----------|-------|-----------|
| 2-component CCD without center (9 levels) | Adapted from RSSD for 2 readings | `[SDK24]` |
| 4-component CCD (24+1 levels) | Generalisation | `[SDK24]` |
| Weighted variance-space / geo-space tradeoff parameter W | SDSampling | `[SDK24]` |
| Simulated annealing for large combinatorial search | Generic optimisation | — |
| cLHS integration | Minasny & McBratney 2006 | — |
| DPPC deterministic model | Rhoades 1989 | `[R89]`, `[L05]` |
| Kriging on MLR residuals / kriging with external drift | Geostatistical extension | `[P1]` (contrast case) |
| Multi-sensor fusion (EM + gamma-ray + L-band) | SDSampling | `[SDK24]` |

---

## 11. RSSD workflow parity contract

This section defines operational parity checks for the original ESAP RSSD workflow.
Each item is labelled as **Confirmed** parity behavior or **Extension**.

| Workflow checkpoint | Label | Acceptance criterion |
|---------------------|-------|----------------------|
| Canonical survey schema (`site_id`, `x`, `y`, `eca_*`) | Extension | All ingest paths normalize to this schema before PCA/RSSD. |
| Candidate ranking per design level (`ψ1–ψ3`) | Confirmed | DLS = squared Euclidean distance in standardised PC space. |
| Global uniqueness of candidate assignments | Confirmed | A site cannot appear in more than one design-level candidate set. |
| Initial assignment | Confirmed | Core set starts from the `ψ1` match for each design level. |
| Swapping loop | Confirmed | Trial `ψ2`, then `ψ3`, retain only if AD decreases; stop when no improvement. |
| Extra-site selection (`extra_mode="global"`) | Extension | Greedy AD reduction over all remaining survey sites. |
| Extra-site selection (`extra_mode="cube"`) | Confirmed | Extra candidates are constrained by remaining cube-level matches. |
| Validation/monitoring pass | Confirmed | Optional second RSSD pass on remaining sites using cube levels only. |
| Original index mapping | Extension | Output includes filtered and original indices to avoid ID drift. |
| Export compatibility | Extension | Field-ready outputs use CSV, optional ESAP-like text summary. |
| ESAP σ QC (3.5 mask / 4.5 outlier) | Confirmed | `detect_outliers_esap`; `eligible_mask` on `run_rssd` |
| Design factor on CCD | Confirmed | `central_composite_design(..., design_factor=...)` |
| Opt-Criteria on `RSSDesign` | Derived | `opt_criteria_from_ad`; exported in CSV/report |

### RSSD workflow compatibility scope

- The package targets ESAP RSSD site-selection parity, not SaltMapper prediction file parity.
- Native ESAP text/binary outputs (`*.rsd1.txt`, `*.gps1.txt`, `*.xrs1.asc`) are treated as
  compatibility references rather than mandatory interchange formats.
- EM survey imports accept **`.csv`** and **`.txt`** only; once normalized, the internal
  workflow is format-agnostic.

---

## 12. Canonical CSV survey schema (modern interchange)

| Column | Required | Notes |
|--------|----------|-------|
| `site_id` | yes | Stable integer identifier (ESAP transect surveys) |
| `x`, `y` | yes | Projected coordinates (metres, e.g. UTM) |
| `EMh`, `EMv`, … | 1–2 channels | Positive ECa in dS/m; `ln` applied in `ECaPCA` |
| `row` | transect only | Optional row number for transect layouts |

Legacy `.svy` / `.pro` files are **not** required for RSSD; they remain useful as
regression references when validating against ESAP desktop output.
