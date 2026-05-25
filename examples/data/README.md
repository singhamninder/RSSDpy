# Example transect fixture

Bundled 2-signal EMI transect survey fixture for demos and development.

| File | Description |
|------|-------------|
| `101710A_for_esap.txt` | Survey input - comma-delimited, no header: `x, y, EMv, EMh, row` (~5101 sites) |

Coordinates in the bundled survey use **EPSG:6339** (NAD83(2011) / California zone 5,
US survey feet). Site IDs are assigned as row numbers (1..N) when loading via
`read_em_survey`.

Used by the demo notebook: [`notebooks/rssd_demo.ipynb`](../../notebooks/rssd_demo.ipynb).
