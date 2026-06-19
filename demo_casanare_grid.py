"""
Casanare — Rice grid map
10 km × 10 km cells coloured by harvested area (ha), municipality
boundaries and department border overlaid.
"""
import json, pathlib, warnings
import numpy as np
import requests
import geopandas as gpd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.ticker
import matplotlib.lines as mlines
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch
from rasterio.transform import from_bounds
from shapely.geometry import box, shape, mapping
import warnings
warnings.filterwarnings("ignore")
matplotlib.use("Agg")

# ── Config ────────────────────────────────────────────────────────────────────
CACHE = pathlib.Path("spam2020_cache")
OUT   = pathlib.Path("outputs")
CACHE.mkdir(exist_ok=True); OUT.mkdir(exist_ok=True)

# ── Download helpers ──────────────────────────────────────────────────────────

def fetch_geojson(url: str, cache_file: str) -> dict:
    p = CACHE / cache_file
    if p.exists():
        return json.loads(p.read_bytes())
    print(f"  Downloading {cache_file} …")
    r = requests.get(url, timeout=60, headers={"User-Agent": "research/1.0"})
    r.raise_for_status()
    p.write_bytes(r.content)
    return r.json()

# ── Load boundaries ───────────────────────────────────────────────────────────
print("Loading boundaries …")

adm1_gj = fetch_geojson(
    "https://github.com/wmgeolab/geoBoundaries/raw/main/releaseData/gbOpen/COL/ADM1/geoBoundaries-COL-ADM1_simplified.geojson",
    "col_adm1.geojson",
)
adm2_gj = fetch_geojson(
    "https://github.com/wmgeolab/geoBoundaries/raw/main/releaseData/gbOpen/COL/ADM2/geoBoundaries-COL-ADM2_simplified.geojson",
    "col_adm2.geojson",
)

adm1 = gpd.GeoDataFrame.from_features(adm1_gj["features"], crs=4326)
adm2 = gpd.GeoDataFrame.from_features(adm2_gj["features"], crs=4326)

casanare_dept = adm1[adm1["shapeName"] == "Casanare"].copy()
assert len(casanare_dept) == 1, "Casanare department not found"

# Clip municipalities to Casanare
casanare_munis = adm2[adm2.intersects(casanare_dept.union_all())].copy()
casanare_munis = gpd.clip(casanare_munis, casanare_dept)
print(f"  Casanare municipalities: {len(casanare_munis)}")

bbox = casanare_dept.total_bounds   # (minx, miny, maxx, maxy)
print(f"  Casanare bbox: {bbox.round(3)}")

# ── Build 10 km grid ─────────────────────────────────────────────────────────
# 10 km ≈ 0.0898° at ~6°N (use exact value for the centroid latitude)
import math
lat_c = (bbox[1] + bbox[3]) / 2
km_per_deg_lat = 111.32
km_per_deg_lon = 111.32 * math.cos(math.radians(lat_c))
cell_deg_lat = 10.0 / km_per_deg_lat
cell_deg_lon = 10.0 / km_per_deg_lon
print(f"  Cell size: {cell_deg_lon:.4f}° lon × {cell_deg_lat:.4f}° lat at {lat_c:.1f}°N")

west, south, east, north = bbox
lons = np.arange(west,  east  + cell_deg_lon, cell_deg_lon)
lats = np.arange(south, north + cell_deg_lat, cell_deg_lat)
COLS = len(lons) - 1
ROWS = len(lats) - 1
print(f"  Grid: {COLS} cols × {ROWS} rows")

# ── Synthetic rice data (Gaussian hotspots on this exact grid) ────────────────
rng = np.random.default_rng(7)

# Convert hotspot positions from lon/lat to grid indices
def ll_to_ij(lon, lat):
    col = int((lon - west)  / cell_deg_lon)
    row = int((north - lat) / cell_deg_lat)   # row 0 = top
    return row, col

hotspots = [
    # (lon,   lat,  lon_sigma_deg, lat_sigma_deg, peak_ha)
    (-72.3,  5.8,  0.5,  0.4,  7500),   # Yopal / Río Cravo Sur plain
    (-71.6,  6.2,  0.35, 0.30, 4500),   # Río Casanare corridor
    (-70.8,  6.0,  0.28, 0.25, 3000),   # Eastern lowlands
    (-72.8,  5.2,  0.40, 0.35, 5500),   # Southern plains
    (-71.2,  5.5,  0.30, 0.28, 2500),   # Central Llanos
]

arr = np.zeros((ROWS, COLS), dtype=np.float32)
cols_c = (lons[:-1] + lons[1:]) / 2    # cell centre longitudes
rows_c = (lats[:-1] + lats[1:]) / 2    # cell centre latitudes (south→north)

for lon0, lat0, slon, slat, peak in hotspots:
    lon_grid, lat_grid = np.meshgrid(cols_c, rows_c[::-1])  # flip lat: row0=north
    blob = peak * np.exp(
        -(((lon_grid - lon0) / slon) ** 2 + ((lat_grid - lat0) / slat) ** 2) / 2
    )
    arr += blob

arr *= 0.5 + 0.5 * rng.random((ROWS, COLS))   # noise texture
arr[arr < 200] = np.nan                         # mask negligible cells

# Mask cells outside Casanare department
casanare_geom = casanare_dept.union_all()
for ri in range(ROWS):
    for ci in range(COLS):
        cell = box(lons[ci], lats[ri], lons[ci+1], lats[ri+1])
        if not cell.intersects(casanare_geom):
            arr[ROWS - 1 - ri, ci] = np.nan     # row 0 = north in image

valid = arr[~np.isnan(arr)]
print(f"\nRice stats: {len(valid)} cells, "
      f"total={valid.sum()/1000:.0f}k ha, max={valid.max():.0f} ha/cell")

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 9), dpi=120)

cmap = LinearSegmentedColormap.from_list("rice", ["#d0e8f5", "#08306b"])
cmap.set_bad(color=(0.96, 0.96, 0.96, 1.0))    # light grey for masked cells

vmin, vmax = 0, float(np.nanmax(arr))
norm = mcolors.PowerNorm(gamma=0.45, vmin=vmin, vmax=vmax)

# Raster extent: (left, right, bottom, top) for imshow
extent = (west, east, south, north)
im = ax.imshow(
    np.ma.masked_invalid(arr),
    extent=extent, origin="upper",
    cmap=cmap, norm=norm,
    interpolation="nearest",
    aspect="equal",
    zorder=2,
)

# ── Grid lines (10 km cells) ──────────────────────────────────────────────────
for lon in lons:
    ax.axvline(lon, color="#bbbbbb", linewidth=0.25, zorder=3, alpha=0.7)
for lat in lats:
    ax.axhline(lat, color="#bbbbbb", linewidth=0.25, zorder=3, alpha=0.7)

# ── Municipality borders ──────────────────────────────────────────────────────
casanare_munis.boundary.plot(
    ax=ax, color="#555555", linewidth=0.7, zorder=4, label="Municipalities"
)

# Municipality name labels (only the largest ones to avoid clutter)
for _, row in casanare_munis.iterrows():
    centroid = row.geometry.centroid
    name = row.get("shapeName", "")
    if name and row.geometry.area > 0.15:   # ~roughly >1500 km² in degrees²
        ax.annotate(
            name, xy=(centroid.x, centroid.y),
            fontsize=5.5, ha="center", va="center",
            color="#222222", zorder=6,
            fontweight="normal",
        )

# ── Department (Casanare) outer border ───────────────────────────────────────
casanare_dept.boundary.plot(
    ax=ax, color="#111111", linewidth=2.0, zorder=5
)

# ── Colorbar — plain hectare labels, no scientific notation ──────────────────
cb = fig.colorbar(im, ax=ax, fraction=0.032, pad=0.03, extend="max")
cb.set_label("Harvested area per cell (ha)", fontsize=9)

# Place ticks at round hectare values that span the data range
tick_vals = [0, 500, 1_000, 2_000, 3_000, 5_000, 8_000]
tick_vals = [t for t in tick_vals if t <= vmax]
cb.set_ticks(tick_vals)
cb.set_ticklabels([f"{t:,}" for t in tick_vals])
cb.ax.tick_params(labelsize=8)

# ── Legend ────────────────────────────────────────────────────────────────────
legend_elements = [
    mlines.Line2D([], [], color="#111111", linewidth=2.0, label="Casanare dept. border"),
    mlines.Line2D([], [], color="#555555", linewidth=0.9, label="Municipality border"),
    mlines.Line2D([], [], color="#bbbbbb", linewidth=0.5, label="10 km grid"),
    Patch(facecolor="#f5f5f5", edgecolor="#aaa", label="< 200 ha (not shown)"),
]
ax.legend(handles=legend_elements, loc="lower left", fontsize=8,
          framealpha=0.93, edgecolor="#ccc")

# ── Labels ────────────────────────────────────────────────────────────────────
ax.set_title("Casanare — Rice Harvested Area\n10 km × 10 km grid  (SPAM2020 demo)",
             fontsize=13, fontweight="bold", pad=10)
ax.set_xlabel("Longitude", fontsize=9)
ax.set_ylabel("Latitude",  fontsize=9)
ax.set_xlim(west  - 0.05, east  + 0.05)
ax.set_ylim(south - 0.05, north + 0.05)
ax.tick_params(labelsize=8)
fig.text(0.5, 0.005,
         "Boundaries: geoBoundaries (CC-BY)  |  Values: synthetic demo — replace with SPAM2020",
         ha="center", fontsize=7, color="#777")

plt.tight_layout(rect=[0, 0.02, 1, 1])
out_path = OUT / "casanare_rice_grid.png"
fig.savefig(out_path, dpi=120, facecolor="white")
plt.close()
print(f"\nSaved {out_path}  ({out_path.stat().st_size//1024} KB)")
