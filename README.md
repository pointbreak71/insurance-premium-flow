# suyana-colombia-mvp-pilot

Parametric agricultural insurance portfolio optimisation pilot for Colombia.

## Overview

This project constructs a minimum variance reinsurer portfolio (MVP) for a
parametric rainfall index insurance product covering Colombian agriculture.

**Land use data**: SPAM2017 crop allocation model (IFPRI) or synthetic fallback
based on known Colombian agricultural geography.

**Rainfall data**: Synthetic — gamma-distributed seasonal accumulations with
ENSO-correlated structure calibrated to Colombian climatology. Not real CHIRPS.

## Methodology

1. Define agricultural exposure universe (cropland extent, crop types, zones)
2. Assign crop calendars and growing seasons
3. Simulate 30 years of seasonal accumulated rainfall per plot (synthetic)
4. Fit gamma distributions and compute percentile thresholds (p1, p10, p90, p99)
5. Compute payout schedule: linear interpolation between attachment (p10/p90) and exhaustion (p1/p99)
6. Assign USD/ha maximum payout rates by crop type
7. Compute full USD payout history per plot per year
8. Aggregate to zone-season-peril portfolio slots
9. Estimate dependence structure (Spearman + Ledoit-Wolf covariance)
10. Solve minimum variance portfolio optimisation (cvxpy)
11. Stress test against historical ENSO years

## Outputs

- `outputs/excel/` — full Excel workbook with all intermediate steps
- `outputs/maps/` — four PNG maps (grid, cropland, crop types, growing seasons)

## Structure

    data/
      raw/          # downloaded source files (not committed if large)
      processed/    # intermediate processed data
    outputs/
      maps/         # PNG map outputs
      excel/        # Excel workbook output
    src/            # source scripts

## Usage

    pip install -r requirements.txt
    python src/run_pipeline.py

## Notes

Rainfall data is synthetic. All outputs are for research and portfolio
structuring purposes only. Not for redistribution as primary data.

---
*Suyana — June 2026*
