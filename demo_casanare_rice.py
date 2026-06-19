"""
Demo: Casanare department, rice only — realistic synthetic data.
"""
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap
from rasterio.transform import from_bounds
import pathlib, warnings
warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")

import colombia_crop_exposure as cce

# ------------------------------------------------------------------
# Casanare bounding box
# ------------------------------------------------------------------
BBOX = (-73.5, 4.5, -69.5, 7.5)   # (west, south, east, north)
COLS, ROWS = 160, 120

transform = from_bounds(*BBOX, COLS, ROWS)

# ------------------------------------------------------------------
# Realistic synthetic rice: concentrated in river-corridor clusters
# ------------------------------------------------------------------
rng = np.random.default_rng(7)
arr = np.zeros((ROWS, COLS), dtype=np.float32)

# Simulate 4 river-plain hotspots (each a Gaussian blob)
hotspots = [
    # (row_centre, col_centre, row_sigma, col_sigma, peak_ha)
    (80, 40,  18, 10, 8500),   # Río Meta floodplain west
    (70, 80,  12,  8, 5000),   # Río Casanare corridor
    (55, 110, 10,  7, 3200),   # eastern lowlands
    (90, 60,  14, 12, 6000),   # southern plains
]

rows_idx, cols_idx = np.mgrid[0:ROWS, 0:COLS]
for r0, c0, sr, sc, peak in hotspots:
    blob = peak * np.exp(
        -(((rows_idx - r0) / sr) ** 2 + ((cols_idx - c0) / sc) ** 2) / 2
    )
    arr += blob

# Add texture noise within blobs, zero out sparse areas
arr *= (0.4 + 0.6 * rng.random((ROWS, COLS)))
arr[arr < 150] = np.nan      # sparsely cultivated cells → transparent

# ------------------------------------------------------------------
# Colombia boundary (clipped to Casanare region)
# ------------------------------------------------------------------
colombia_gdf = cce.get_colombia_boundary()
colombia_clip = colombia_gdf.clip(
    gpd.GeoDataFrame(geometry=[__import__('shapely').geometry.box(*BBOX)], crs=4326)
)

# ------------------------------------------------------------------
# Plot
# ------------------------------------------------------------------
cmap  = LinearSegmentedColormap.from_list("rice", ["#deebf7", "#08306b"])
valid = (~np.isnan(arr)) & (arr >= 150)
vmin, vmax = 150, float(arr[valid].max()) if valid.any() else 1

extent = (BBOX[0], BBOX[2], BBOX[1], BBOX[3])

fig, ax = plt.subplots(figsize=(9, 8), dpi=133)   # → 1200×1067

im = ax.imshow(
    arr, extent=extent, origin="upper",
    cmap=cmap,
    norm=mcolors.LogNorm(vmin=vmin, vmax=vmax),
    interpolation="nearest", aspect="equal", zorder=2,
)

# Mask transparent cells (NaN) with white
arr_masked = np.where(np.isnan(arr), 0, 1)
ax.imshow(
    1 - arr_masked, extent=extent, origin="upper",
    cmap="Greys", vmin=0, vmax=1, alpha=0.0,   # handled by cmap NaN colour
    interpolation="nearest", aspect="equal", zorder=1,
)

cmap.set_bad(color=(1, 1, 1, 0))   # NaN → transparent

# Re-draw with masked array so NaN cells are transparent
arr_m = np.ma.masked_invalid(arr)
ax.cla()
ax.imshow(
    arr_m, extent=extent, origin="upper",
    cmap=cmap,
    norm=mcolors.LogNorm(vmin=vmin, vmax=vmax),
    interpolation="nearest", aspect="equal", zorder=2,
)
ax.set_facecolor("#f5f5f5")   # light grey background for non-crop cells

colombia_clip.boundary.plot(ax=ax, color="#222222", linewidth=1.2, zorder=5)

cb = fig.colorbar(
    plt.cm.ScalarMappable(cmap=cmap, norm=mcolors.LogNorm(vmin=vmin, vmax=vmax)),
    ax=ax, fraction=0.035, pad=0.02,
)
cb.set_label("Harvested area (ha)", fontsize=9)
cb.ax.tick_params(labelsize=8)

ax.set_title("Casanare — Rice Harvested Area (SPAM2020 demo)", fontsize=13, fontweight="bold", pad=10)
ax.set_xlabel("Longitude", fontsize=9)
ax.set_ylabel("Latitude", fontsize=9)
ax.set_xlim(BBOX[0], BBOX[2])
ax.set_ylim(BBOX[1], BBOX[3])
ax.tick_params(labelsize=8)

total_kha = float(np.nansum(arr)) / 1000
ax.text(0.02, 0.02, f"Total: {total_kha:,.0f}k ha (synthetic demo)",
        transform=ax.transAxes, fontsize=8, color="#555",
        bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"))

plt.tight_layout()
out = pathlib.Path("outputs/casanare_rice_demo.png")
fig.savefig(out, dpi=133, facecolor="white")
plt.close()
print(f"Saved {out}")
