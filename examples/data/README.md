# Example transect fixture (USDA Field 10-6, fall 2016)

Bundled 2-signal EMI transect survey used for demos, development, and the ESAP parity
regression test (`tests/test_esap_parity.py`).

| File | Description |
|------|-------------|
| `101710A_for_esap.txt` | Survey input — comma-delimited, no header: `x, y, EMv, EMh, row` (5102 rows bundled; ESAP reports 5101 — see source notes Section 14 for the unresolved one-row discrepancy) |
| `106Frsd1_expected.txt` | ESAP desktop RSSD output — n=12, D-Factor 0.96, Opt-Criteria ~1.26 |
| `106Finfo_reference.txt` | ESAP decorrelation/QC summary — 72 masked, 1 deleted, 5029 active |

Coordinates in the bundled survey use **EPSG:6339** (NAD83(2011) / California zone 5,
US survey feet). Site IDs are assigned as row numbers (1..N) when loading via
`read_em_survey`.

Used by the demo notebook ([`notebooks/rssd_demo.ipynb`](../../notebooks/rssd_demo.ipynb)) and
the parity test (`tests/test_esap_parity.py`); see
`docs/theory/rssd-source-notes.md` Section 14 for the full parity status against
`106Frsd1_expected.txt`.
