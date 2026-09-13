# AGENTS.md — RSSDpy

Cross-tool brief for coding agents. Keep this file short. Deep domain rules live
under `.cursor/rules/` and `docs/theory/` — read those on demand; do not copy
full algorithms or formulas into this file.

## Project

Python library for **Response Surface Sampling Design (RSSD)** — ECa-directed
soil salinity sampling faithful to Lesch et al. (1995) and USDA-ARS ESAP-RSSD.

Audience: geospatial data scientists working with EMI (EM-38, DUALEM) survey data.

## Dev commands

```bash
uv sync --locked --all-extras --dev
uv run ruff format .
uv run ruff check --fix .
uv run ty check .
uv run pytest
```

Or as one chain: `uv sync --locked --all-extras --dev && uv run ruff format . && uv run ruff check --fix . && uv run ty check . && uv run pytest`

## Hard invariants

- **Source-truth labels** on theory claims in code: Confirmed / Derived / Extension.
- **Design radius `3.84`**: ESAP/Lesch constant = `chi2.ppf(0.95, df=1)` = `1.96**2`.
  Never treat it as `chi2.ppf(0.95, df=3)`.
- **Log transform**: `np.log` (natural log) only — never `log10`.
- **ECa**: positive floats; coordinates in a **projected CRS** for distances.
- **Canonical I/O**: CSV/TXT via `read_em_survey`. Legacy `.svy` / `.pro` / `rsd#.txt`
  are regression references only — convert before load.
- **ESAP QC**: `iterative_esap_validation` + pass `eligible_mask` into `run_rssd`.
- **ESAP parity metric**: compare `opt_criteria`, not `ad_final` (metres).
- **Index mapping**: after QC filtering, thread `original_indices` through `run_rssd`.

## Module map

| Path | Role |
|------|------|
| `src/rssdpy/features/` | PCA, log-ECa, outliers / ESAP σ QC |
| `src/rssdpy/sampling/` | CCD, RSSD pipeline, AD / swaps |
| `src/rssdpy/calibration/` | MLR, PRESS/APVE, diagnostics |
| `src/rssdpy/predict/` | Salinity predictions & probabilities |
| `src/rssdpy/io/` | Survey loaders, CSV/GeoPackage exports |
| `src/rssdpy/viz/` | Plots (non-core) |
| `src/rssdpy/pipeline.py` | `run_esap_rssd` — bundles QC + sample plan + RSSD in one call |
| `app/` | Streamlit interactive app (`app` extra) |
| `tests/` | pytest fixtures and coverage |

## Deep truth (read on demand)

- [`.cursor/rules/project_RSSD.mdc`](.cursor/rules/project_RSSD.mdc) — algorithms, architecture, source-truth policy
- [`.cursor/rules/esap-modernization.mdc`](.cursor/rules/esap-modernization.mdc) — parity workflow & anti-patterns
- [`.cursor/rules/python-standards.mdc`](.cursor/rules/python-standards.mdc) — typing, docstrings, ruff/ty
- [`docs/theory/rssd-source-notes.md`](docs/theory/rssd-source-notes.md) — source-truth register

## Non-goals for this file

Do not restate CCD levels, ψ₁–ψ₃ matching, AD swaps, PRESS/APVE, or MLR formulas
here. Those belong in the deep rules and source notes above.
