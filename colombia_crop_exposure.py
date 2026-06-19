"""
Colombia Agricultural Exposure Map — SPAM2020 Data
====================================================
Downloads SPAM2020 harvested-area GeoTIFFs from Harvard Dataverse,
clips them to Colombia's bounding box, and renders:
  1. A dominant-crop choropleth (outputs/colombia_crop_exposure.png)
  2. A 5-panel per-crop figure   (outputs/colombia_crop_exposure_panels.png)

Run with:  python colombia_crop_exposure.py
Requires:  rasterio, geopandas, matplotlib, numpy, requests, scipy
"""

import io
import os
import sys
import json
import zipfile
import tempfile
import pathlib
import warnings
import textwrap

import numpy as np
import requests
import rasterio
import rasterio.mask
import rasterio.windows
import geopandas as gpd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap, Normalize
from rasterio.crs import CRS
from rasterio.transform import from_bounds
from shapely.geometry import box

warnings.filterwarnings("ignore", category=UserWarning)
matplotlib.use("Agg")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATAVERSE_DOI = "doi:10.7910/DVN/SWPENT"
DATAVERSE_API  = "https://dataverse.harvard.edu"

# Colombia bounding box  [west, south, east, north]
COLOMBIA_BBOX = (-79.0, -4.0, -66.0, 13.0)

# Crops to map: {code: display_name}
CROPS = {
    "MAIZ": "Maize",
    "SOYB": "Soybean",
    "RICE": "Rice",
    "COFF": "Coffee",
    "SUGC": "Sugarcane",
}

# Colour ramps for each crop (light → dark)
CROP_COLORS = {
    "MAIZ": ["#e5f5e0", "#31a354"],        # greens
    "SOYB": ["#fff7bc", "#d95f0e"],        # yellows → oranges
    "RICE": ["#deebf7", "#2171b5"],        # blues
    "COFF": ["#f2f0f7", "#6a3d9a"],        # purple
    "SUGC": ["#fee5d9", "#cb181d"],        # reds
}

# Solid representative colour for each crop (used in legend / dominant map)
CROP_SOLID = {
    "MAIZ": "#31a354",
    "SOYB": "#d95f0e",
    "RICE": "#2171b5",
    "COFF": "#6a3d9a",
    "SUGC": "#cb181d",
}

MIN_TOTAL_HA = 100          # cells below this threshold rendered as transparent
CACHE_DIR    = pathlib.Path("spam2020_cache")
OUT_DIR      = pathlib.Path("outputs")
OUT_DIR.mkdir(exist_ok=True)
CACHE_DIR.mkdir(exist_ok=True)

NODATA_VALUE = -9999.0

# ---------------------------------------------------------------------------
# Step 1 — Discover SPAM2020 files via Dataverse API
# ---------------------------------------------------------------------------

def get_dataverse_file_list(doi: str) -> list[dict]:
    """Return list of file-metadata dicts from Harvard Dataverse."""
    url = (
        f"{DATAVERSE_API}/api/datasets/:persistentId/versions/:latest/files"
        f"?persistentId={doi}&perPage=500"
    )
    headers = {
        "Accept": "application/json",
        "User-Agent": "SPAM2020-downloader/1.0 (agricultural research)",
    }
    print(f"  Querying Dataverse file list …")
    r = requests.get(url, headers=headers, timeout=60)
    r.raise_for_status()
    return r.json().get("data", [])


def find_crop_file(file_list: list[dict], crop: str) -> dict | None:
    """
    Locate the harvested-area 'all systems' file for a given crop.
    SPAM2020 naming: spam2020_v1r0_global_A_{CROP}_A.tif
    Falls back to summing I / H / L / S files if the combined _A file is absent.
    """
    target = f"_A_{crop}_A".upper()
    candidates = []
    for entry in file_list:
        name: str = entry.get("dataFile", {}).get("filename", "")
        if crop.upper() in name.upper() and name.lower().endswith(".tif"):
            candidates.append(entry)
            if target in name.upper():
                return entry          # exact _A (all-systems) match
    # Fall back: return all tech variants so caller can sum them
    tech_files = [
        e for e in candidates
        if any(f"_A_{crop}_{t}".upper() in e["dataFile"]["filename"].upper()
               for t in ("I", "H", "L", "S"))
    ]
    if tech_files:
        return tech_files             # caller receives a list → will be summed
    if candidates:
        return candidates[0]
    return None


# ---------------------------------------------------------------------------
# Step 2 — Download a single file (streaming, with local cache)
# ---------------------------------------------------------------------------

def download_file(file_id: int, filename: str) -> pathlib.Path:
    cache_path = CACHE_DIR / filename
    if cache_path.exists():
        print(f"  Cache hit : {filename}")
        return cache_path

    url = f"{DATAVERSE_API}/api/access/datafile/{file_id}"
    print(f"  Downloading {filename} (id={file_id}) …")
    headers = {"User-Agent": "SPAM2020-downloader/1.0"}
    tmp_path = cache_path.with_suffix(".download")
    with requests.get(url, stream=True, timeout=300, headers=headers) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        downloaded = 0
        with open(tmp_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
                downloaded += len(chunk)
                if total:
                    pct = downloaded / total * 100
                    print(f"    {pct:5.1f}%  ({downloaded>>20} / {total>>20} MB)", end="\r")
        print()

    # Dataverse sometimes wraps single-file downloads in a zip
    if zipfile.is_zipfile(tmp_path):
        print(f"    Extracting ZIP …")
        with zipfile.ZipFile(tmp_path, "r") as zf:
            tif_names = [n for n in zf.namelist() if n.lower().endswith(".tif")]
            if not tif_names:
                raise RuntimeError(f"No .tif found inside ZIP for {filename}")
            # Pick the file that best matches the expected name
            best = next(
                (n for n in tif_names if filename.replace(".tif", "").upper() in n.upper()),
                tif_names[0],
            )
            with zf.open(best) as src, open(cache_path, "wb") as dst:
                dst.write(src.read())
        tmp_path.unlink()
    else:
        tmp_path.rename(cache_path)

    return cache_path


# ---------------------------------------------------------------------------
# Step 3 — Read GeoTIFF clipped to Colombia bounding box
# ---------------------------------------------------------------------------

def read_clipped_array(tif_path: pathlib.Path, bbox: tuple) -> tuple[np.ndarray, object]:
    """
    Returns (2D float32 array, affine transform) clipped to bbox.
    NoData cells are set to NaN.
    """
    west, south, east, north = bbox
    with rasterio.open(tif_path) as src:
        # Convert bbox to pixel window
        win = rasterio.windows.from_bounds(
            west, south, east, north, transform=src.transform
        )
        win = win.intersection(
            rasterio.windows.Window(0, 0, src.width, src.height)
        )
        data = src.read(1, window=win).astype(np.float32)
        transform = src.window_transform(win)
        nodata = src.nodata if src.nodata is not None else NODATA_VALUE
        data[data == nodata] = np.nan
        data[data < 0] = np.nan
    return data, transform


def load_crop_array(
    crop: str,
    file_list: list[dict],
    bbox: tuple,
) -> tuple[np.ndarray, object]:
    """
    Download (if needed) and read the harvested-area array for one crop.
    If only per-technology files are available, sum them.
    """
    result = find_crop_file(file_list, crop)

    if result is None:
        raise RuntimeError(f"No file found for crop {crop} in Dataverse dataset.")

    # Single file  vs  list of tech files
    if isinstance(result, list):
        arrays = []
        for entry in result:
            df   = entry["dataFile"]
            path = download_file(df["id"], df["filename"])
            arr, transform = read_clipped_array(path, bbox)
            arrays.append(arr)
        combined = np.nansum(np.stack(arrays, axis=0), axis=0)
        combined[np.all(np.isnan(np.stack(arrays, axis=0)), axis=0)] = np.nan
        return combined, transform
    else:
        df   = result["dataFile"]
        path = download_file(df["id"], df["filename"])
        return read_clipped_array(path, bbox)


# ---------------------------------------------------------------------------
# Step 4 — Fetch Colombia boundary from Natural Earth via GitHub mirror
# ---------------------------------------------------------------------------

NATURALEARTH_COUNTRIES_URL = (
    "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/"
    "master/geojson/ne_10m_admin_0_countries.geojson"
)
NATURALEARTH_LOW_URL = (
    "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/"
    "master/geojson/ne_110m_admin_0_countries.geojson"
)


def get_colombia_boundary() -> gpd.GeoDataFrame:
    cache_path = CACHE_DIR / "ne_countries.geojson"
    if not cache_path.exists():
        print("  Downloading Natural Earth country boundaries …")
        for url in (NATURALEARTH_LOW_URL, NATURALEARTH_COUNTRIES_URL):
            try:
                r = requests.get(url, timeout=60,
                                 headers={"User-Agent": "SPAM2020-downloader/1.0"})
                r.raise_for_status()
                cache_path.write_bytes(r.content)
                break
            except Exception as e:
                print(f"    Warn: {url} failed ({e}), trying next …")
    world = gpd.read_file(cache_path)
    # Try common field names for country identification
    for col in ("NAME", "ADMIN", "NAME_EN", "name"):
        if col in world.columns:
            mask = world[col].str.contains("Colombia", case=False, na=False)
            if mask.any():
                return world[mask].to_crs(epsg=4326)
    raise RuntimeError("Could not find Colombia in Natural Earth dataset.")


# ---------------------------------------------------------------------------
# Step 5 — Build dominant-crop arrays for mapping
# ---------------------------------------------------------------------------

def build_dominant_arrays(
    crop_arrays: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
      dominant_idx  — index into sorted crop list for dominant crop (or -1 if below threshold)
      total_ha      — total harvested area across all crops per cell
      alpha         — normalised [0,1] intensity (NaN where below threshold)
    """
    crop_list = list(crop_arrays.keys())
    n = len(crop_list)
    shape = next(iter(crop_arrays.values())).shape
    stack = np.stack(
        [np.where(np.isnan(v), 0.0, v) for v in crop_arrays.values()], axis=0
    )  # (n_crops, rows, cols)

    total_ha = stack.sum(axis=0)
    dominant_idx = np.argmax(stack, axis=0).astype(np.float32)
    dominant_idx[total_ha < MIN_TOTAL_HA] = np.nan     # mask low-exposure cells

    # Alpha scales with log(total_ha) for visual clarity
    valid = total_ha >= MIN_TOTAL_HA
    alpha = np.full(shape, np.nan)
    if valid.any():
        log_vals = np.log1p(total_ha[valid])
        vmin, vmax = log_vals.min(), log_vals.max()
        if vmax > vmin:
            alpha[valid] = (log_vals - vmin) / (vmax - vmin)
        else:
            alpha[valid] = 1.0

    return dominant_idx, total_ha, alpha


# ---------------------------------------------------------------------------
# Step 6 — Plotting helpers
# ---------------------------------------------------------------------------

def make_rgba_dominant(
    dominant_idx: np.ndarray,
    alpha: np.ndarray,
    crop_list: list[str],
) -> np.ndarray:
    """Convert dominant-crop index + alpha into an RGBA image array."""
    rows, cols = dominant_idx.shape
    rgba = np.zeros((rows, cols, 4), dtype=np.float32)
    rgba[:, :, 3] = 0.0   # transparent by default

    for i, crop in enumerate(crop_list):
        r, g, b = mcolors.to_rgb(CROP_SOLID[crop])
        mask = (~np.isnan(dominant_idx)) & (dominant_idx == i)
        if not mask.any():
            continue
        a = alpha[mask]
        # Blend solid colour toward white for low values
        rgba[mask, 0] = r * a + 1.0 * (1 - a)
        rgba[mask, 1] = g * a + 1.0 * (1 - a)
        rgba[mask, 2] = b * a + 1.0 * (1 - a)
        rgba[mask, 3] = np.clip(0.2 + 0.8 * a, 0, 1)

    return rgba


def make_rgba_single_crop(arr: np.ndarray, color_pair: list[str]) -> np.ndarray:
    """Convert a single-crop harvested-area array to RGBA."""
    rows, cols = arr.shape
    rgba = np.zeros((rows, cols, 4), dtype=np.float32)
    valid = (~np.isnan(arr)) & (arr >= MIN_TOTAL_HA)
    if not valid.any():
        return rgba

    cmap = LinearSegmentedColormap.from_list("crop", color_pair)
    log_vals = np.log1p(arr[valid])
    vmin, vmax = log_vals.min(), log_vals.max()
    if vmax == vmin:
        norm_vals = np.ones_like(log_vals)
    else:
        norm_vals = (log_vals - vmin) / (vmax - vmin)

    colours = cmap(norm_vals)           # (N, 4)
    colours[:, 3] = np.clip(0.2 + 0.8 * norm_vals, 0, 1)
    rgba[valid] = colours
    return rgba


def get_extent(transform, shape):
    """Return (left, right, bottom, top) from affine transform + array shape."""
    rows, cols = shape
    left   = transform.c
    top    = transform.f
    right  = left + cols * transform.a
    bottom = top  + rows * transform.e
    return left, right, bottom, top


def plot_boundary(ax, colombia_gdf):
    colombia_gdf.boundary.plot(ax=ax, color="#333333", linewidth=0.8, zorder=5)


def style_ax(ax, title: str):
    ax.set_title(title, fontsize=11, fontweight="bold", pad=6)
    ax.set_xlabel("Longitude", fontsize=8)
    ax.set_ylabel("Latitude", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.set_aspect("equal")
    # Tight bbox around Colombia
    ax.set_xlim(COLOMBIA_BBOX[0], COLOMBIA_BBOX[2])
    ax.set_ylim(COLOMBIA_BBOX[1], COLOMBIA_BBOX[3])


# ---------------------------------------------------------------------------
# Step 7 — Main figure: dominant-crop choropleth
# ---------------------------------------------------------------------------

def plot_dominant_map(
    crop_arrays: dict[str, np.ndarray],
    transform,
    colombia_gdf: gpd.GeoDataFrame,
    out_path: pathlib.Path,
):
    crop_list = list(crop_arrays.keys())
    dominant_idx, total_ha, alpha = build_dominant_arrays(crop_arrays)
    rgba = make_rgba_dominant(dominant_idx, alpha, crop_list)
    extent = get_extent(transform, rgba.shape[:2])

    fig, ax = plt.subplots(figsize=(12, 10), dpi=100)
    ax.imshow(rgba, extent=extent, origin="upper", interpolation="nearest",
              aspect="equal", zorder=2)
    plot_boundary(ax, colombia_gdf)

    # Legend
    patches = [
        mpatches.Patch(facecolor=CROP_SOLID[c], edgecolor="#555", label=CROPS[c])
        for c in crop_list
    ]
    ax.legend(
        handles=patches,
        title="Dominant crop",
        loc="lower right",
        fontsize=9,
        title_fontsize=9,
        framealpha=0.9,
    )

    style_ax(ax, "Colombia — Dominant Agricultural Crop by Harvested Area (SPAM2020)")
    fig.text(
        0.5, 0.01,
        "Source: SPAM2020, doi:10.7910/DVN/SWPENT  |  Colour intensity ∝ log(total ha)",
        ha="center", fontsize=7, color="#666",
    )
    plt.tight_layout(rect=[0, 0.02, 1, 1])
    fig.savefig(out_path, dpi=100, facecolor="white")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Step 8 — 5-panel per-crop figure
# ---------------------------------------------------------------------------

def plot_panel_map(
    crop_arrays: dict[str, np.ndarray],
    transform,
    colombia_gdf: gpd.GeoDataFrame,
    out_path: pathlib.Path,
):
    crop_list = list(crop_arrays.keys())
    fig, axes = plt.subplots(1, 5, figsize=(24, 7), dpi=100)

    for ax, (crop, name) in zip(axes, CROPS.items()):
        arr   = crop_arrays.get(crop)
        if arr is None:
            ax.set_visible(False)
            continue
        rgba   = make_rgba_single_crop(arr, CROP_COLORS[crop])
        extent = get_extent(transform, rgba.shape[:2])
        ax.imshow(rgba, extent=extent, origin="upper", interpolation="nearest",
                  aspect="equal", zorder=2)
        plot_boundary(ax, colombia_gdf)

        # Colour bar
        cmap = LinearSegmentedColormap.from_list("crop", CROP_COLORS[crop])
        valid = (~np.isnan(arr)) & (arr >= MIN_TOTAL_HA)
        vmax  = float(arr[valid].max()) if valid.any() else 1.0
        sm    = plt.cm.ScalarMappable(
            cmap=cmap, norm=mcolors.LogNorm(vmin=MIN_TOTAL_HA, vmax=max(vmax, MIN_TOTAL_HA + 1))
        )
        sm.set_array([])
        cb = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
        cb.set_label("Harvested area (ha)", fontsize=7)
        cb.ax.tick_params(labelsize=6)

        style_ax(ax, name)

    fig.suptitle(
        "Colombia — Harvested Area by Crop (SPAM2020)",
        fontsize=14, fontweight="bold", y=1.01,
    )
    fig.text(
        0.5, -0.01,
        "Source: SPAM2020, doi:10.7910/DVN/SWPENT",
        ha="center", fontsize=7, color="#666",
    )
    plt.tight_layout()
    fig.savefig(out_path, dpi=100, facecolor="white")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Step 9 — Console summary table
# ---------------------------------------------------------------------------

def print_summary(crop_arrays: dict[str, np.ndarray]):
    print("\n" + "=" * 55)
    print(f"  {'Crop':<12}  {'Total area (1000 ha)':>22}  {'Max cell (ha)':>14}")
    print("-" * 55)
    grand = 0.0
    for crop, arr in crop_arrays.items():
        valid = arr[~np.isnan(arr)]
        total_kha = valid.sum() / 1_000
        max_ha    = valid.max() if len(valid) else 0
        grand += total_kha
        print(f"  {CROPS[crop]:<12}  {total_kha:>22,.1f}  {max_ha:>14,.0f}")
    print("-" * 55)
    print(f"  {'TOTAL':<12}  {grand:>22,.1f}")
    print("=" * 55 + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("\n=== Colombia Crop Exposure Map — SPAM2020 ===\n")

    # --- Discover files on Dataverse ---
    print("[1/5] Fetching file list from Harvard Dataverse …")
    try:
        file_list = get_dataverse_file_list(DATAVERSE_DOI)
        print(f"      Found {len(file_list)} files in dataset.")
    except requests.HTTPError as e:
        print(f"\n  ERROR: Could not reach Harvard Dataverse API.\n  {e}")
        print(
            "\n  Possible causes:\n"
            "  • Running inside a restricted network environment\n"
            "  • Harvard Dataverse is temporarily unavailable\n"
            "\n  Solution: run this script on a machine with unrestricted internet\n"
            "  access to dataverse.harvard.edu, or manually download the files\n"
            "  and place them in ./spam2020_cache/  then re-run.\n"
            "\n  Manual download URL pattern:\n"
            "    https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/SWPENT\n"
            "  Expected files:\n"
        )
        for crop in CROPS:
            print(f"    spam2020_v1r0_global_A_{crop}_A.tif")
        sys.exit(1)

    # --- Load crop arrays ---
    print("\n[2/5] Downloading and reading crop rasters (Colombia clip) …")
    crop_arrays  = {}
    crop_transforms = {}
    for crop in CROPS:
        print(f"  Crop: {CROPS[crop]} ({crop})")
        try:
            arr, transform = load_crop_array(crop, file_list, COLOMBIA_BBOX)
            crop_arrays[crop]    = arr
            crop_transforms[crop] = transform
            valid = arr[~np.isnan(arr)]
            print(f"    Clipped shape: {arr.shape}  |  valid cells: {len(valid)}  "
                  f"|  max: {valid.max() if len(valid) else 0:.0f} ha")
        except Exception as e:
            print(f"    WARNING: could not load {crop}: {e}")

    if not crop_arrays:
        print("ERROR: No crop data loaded. Exiting.")
        sys.exit(1)

    # Use the transform from the first successful crop (all share the same grid)
    ref_transform = next(iter(crop_transforms.values()))

    # Align all arrays to the same shape (handle off-by-one from windowed reads)
    shapes = [a.shape for a in crop_arrays.values()]
    min_rows = min(s[0] for s in shapes)
    min_cols = min(s[1] for s in shapes)
    crop_arrays = {k: v[:min_rows, :min_cols] for k, v in crop_arrays.items()}

    # --- Print summary ---
    print("\n[3/5] Summary of harvested area in Colombia …")
    print_summary(crop_arrays)

    # --- Colombia boundary ---
    print("[4/5] Loading Colombia boundary …")
    colombia_gdf = get_colombia_boundary()

    # --- Generate maps ---
    print("[5/5] Rendering maps …")
    plot_dominant_map(
        crop_arrays, ref_transform, colombia_gdf,
        OUT_DIR / "colombia_crop_exposure.png",
    )
    plot_panel_map(
        crop_arrays, ref_transform, colombia_gdf,
        OUT_DIR / "colombia_crop_exposure_panels.png",
    )

    print("\nDone.  Output files:")
    for p in sorted(OUT_DIR.glob("colombia_crop_exposure*.png")):
        size_kb = p.stat().st_size // 1024
        print(f"  {p}  ({size_kb} KB)")
    print()


if __name__ == "__main__":
    main()
