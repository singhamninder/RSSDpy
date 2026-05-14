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
| 2-component CCD variants | Extension | — | Not the primary ESAP target for 2-reading surveys per `[ESAP]`; see note |
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
