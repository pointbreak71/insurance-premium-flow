"""
Smoke-test the visualization code using synthetic Colombia rasters.
Verifies the plotting path without needing a Dataverse connection.
"""
import sys
sys.path.insert(0, ".")

import numpy as np
import pathlib
import geopandas as gpd
import rasterio
from rasterio.transform import from_bounds
from rasterio.crs import CRS
import tempfile, os

# Patch the module so it doesn't need an internet connection
import colombia_crop_exposure as cce

BBOX = cce.COLOMBIA_BBOX  # (-79, -4, -66, 13)
COLS, ROWS = 130, 170
transform = from_bounds(*BBOX, COLS, ROWS)

rng = np.random.default_rng(42)

def make_fake_array(seed_scale=1.0):
    """Sparse positive array mimicking harvested area (lots of NaNs)."""
    arr = np.zeros((ROWS, COLS), dtype=np.float32)
    # Put random values in ~30% of cells
    mask = rng.random((ROWS, COLS)) < 0.30
    arr[mask] = rng.exponential(scale=1000 * seed_scale, size=mask.sum()).astype(np.float32)
    arr[arr == 0] = np.nan
    return arr

# Build synthetic crop arrays
crop_arrays = {
    "MAIZ": make_fake_array(2.0),
    "SOYB": make_fake_array(0.5),
    "RICE": make_fake_array(1.0),
    "COFF": make_fake_array(0.8),
    "SUGC": make_fake_array(0.4),
}

print("Generating synthetic crop arrays …")
for crop, arr in crop_arrays.items():
    valid = arr[~np.isnan(arr)]
    print(f"  {crop}: shape={arr.shape}, valid={len(valid)}, max={valid.max():.0f}")

# Load Colombia boundary
print("\nLoading Colombia boundary …")
colombia_gdf = cce.get_colombia_boundary()
print(f"  Got {len(colombia_gdf)} feature(s)")

# Print summary
print("\nSummary table:")
cce.print_summary(crop_arrays)

# Render maps
print("Rendering dominant-crop map …")
cce.plot_dominant_map(
    crop_arrays, transform, colombia_gdf,
    cce.OUT_DIR / "colombia_crop_exposure.png",
)

print("Rendering 5-panel map …")
cce.plot_panel_map(
    crop_arrays, transform, colombia_gdf,
    cce.OUT_DIR / "colombia_crop_exposure_panels.png",
)

print("\nTest complete. Check outputs/")
