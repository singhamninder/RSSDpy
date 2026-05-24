# Field 10-6 example data (USDA ESAP regression)

Transect EMI survey from USDA Field 10-6 (fall 2016), used to regression-test RSSDpy
against desktop ESAP-RSSD output.

| File | Description |
|------|-------------|
| `101710A_for_esap.txt` | Survey input — comma-delimited, no header: `x, y, EMv, EMh, row` (~5101 sites) |
| `106Frsd1_expected.txt` | ESAP reference design — n=12, D-Factor 0.96, Opt-Criteria ~1.26 |
| `106Finfo_reference.txt` | ESAP decorrelation/QC summary (72 masked, 5029 active) |

Coordinates use **EPSG:6339** (NAD83(2011) / California zone 5, US survey feet). Site IDs are assigned as row numbers
(1..N) when loading via `read_em_survey`.

Notebook workflow: [`notebooks/field_10_6_rssd.py`](../../../notebooks/field_10_6_rssd.py).
