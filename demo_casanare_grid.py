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

# ── Aggregate rice ha per municipality ───────────────────────────────────────
from shapely.geometry import box as sbox

muni_ha = {}
for _, mrow in casanare_munis.iterrows():
    muni_geom = mrow.geometry
    name = mrow.get("shapeName", "Unknown")
    total = 0.0
    for ri in range(ROWS):
        for ci in range(COLS):
            cell = sbox(lons[ci], lats[ri], lons[ci+1], lats[ri+1])
            val  = arr[ROWS - 1 - ri, ci]
            if np.isnan(val):
                continue
            if cell.intersects(muni_geom):
                overlap = cell.intersection(muni_geom).area / cell.area
                total += float(val) * overlap
    muni_ha[name] = total

muni_names  = sorted(muni_ha, key=muni_ha.get, reverse=True)
muni_values = [muni_ha[n] / 1_000 for n in muni_names]   # → thousands of ha

# ── Plot ──────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(18, 10), dpi=110)
ax  = fig.add_axes([0.03, 0.06, 0.40, 0.88])   # map (left)
ax2 = fig.add_axes([0.52, 0.06, 0.44, 0.88])   # bar chart (right)

cmap = LinearSegmentedColormap.from_list("rice", ["#d0e8f5", "#08306b"])
cmap.set_bad(color=(0.96, 0.96, 0.96, 1.0))    # light grey for masked cells

vmin, vmax = 200, float(np.nanmax(arr))
norm = mcolors.LogNorm(vmin=vmin, vmax=vmax)

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

# ── Colorbar (map) ────────────────────────────────────────────────────────────
cb = fig.colorbar(im, ax=ax, fraction=0.040, pad=0.02, extend="both")
cb.set_label("Rice ha per 10×10 km cell", fontsize=8)
cb.ax.tick_params(labelsize=7)

# ── Legend (map) ──────────────────────────────────────────────────────────────
legend_elements = [
    mlines.Line2D([], [], color="#111111", linewidth=2.0, label="Dept. border"),
    mlines.Line2D([], [], color="#555555", linewidth=0.9, label="Municipality"),
    mlines.Line2D([], [], color="#bbbbbb", linewidth=0.5, label="10 km grid"),
    Patch(facecolor="#f5f5f5", edgecolor="#aaa", label="<200 ha (masked)"),
]
ax.legend(handles=legend_elements, loc="lower left", fontsize=7,
          framealpha=0.92, edgecolor="#ccc")

ax.set_title("Rice harvested area\n10 km × 10 km grid", fontsize=11,
             fontweight="bold", pad=8)
ax.set_xlabel("Longitude", fontsize=8)
ax.set_ylabel("Latitude",  fontsize=8)
ax.set_xlim(west  - 0.05, east  + 0.05)
ax.set_ylim(south - 0.05, north + 0.05)
ax.tick_params(labelsize=7)

# ── Bar chart: ha per municipality ───────────────────────────────────────────
bar_colors = [plt.cm.YlGnBu(0.3 + 0.7 * v / max(muni_values)) for v in muni_values]
bars = ax2.barh(
    range(len(muni_names)), muni_values,
    color=bar_colors, edgecolor="#aaaaaa", linewidth=0.4,
)

ax2.set_yticks(range(len(muni_names)))
ax2.set_yticklabels(muni_names, fontsize=8)
ax2.invert_yaxis()                             # largest at top
ax2.set_xlabel("Harvested area (thousand ha)", fontsize=10)
ax2.set_title("Total rice per municipality\n(summed from grid cells)",
              fontsize=11, fontweight="bold", pad=8)
ax2.xaxis.grid(True, linestyle="--", alpha=0.5, zorder=0)
ax2.set_axisbelow(True)

# Value labels on bars
for i, v in enumerate(muni_values):
    ax2.text(v + max(muni_values) * 0.01, i, f"{v:.1f}k",
             va="center", fontsize=7, color="#333")

ax2.spines[["top", "right"]].set_visible(False)
ax2.tick_params(labelsize=8)

# ── Figure title + footnote ───────────────────────────────────────────────────
fig.suptitle("Casanare — Rice Harvested Area (SPAM2020 demo)",
             fontsize=14, fontweight="bold", y=1.01)
fig.text(0.5, -0.01,
         "Boundaries: geoBoundaries (CC-BY)  |  Values: synthetic demo — replace with SPAM2020",
         ha="center", fontsize=6.5, color="#777")

out_path = OUT / "casanare_rice_grid.png"
fig.savefig(out_path, dpi=110, bbox_inches="tight", facecolor="white")
plt.close()
print(f"\nSaved {out_path}  ({out_path.stat().st_size//1024} KB)")
