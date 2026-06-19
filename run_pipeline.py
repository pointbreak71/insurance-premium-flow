#!/usr/bin/env python3
"""
Minimum Variance Parametric Insurance Bundle for Colombia
Complete end-to-end pipeline
"""

import os
import sys
import logging
import warnings
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy import stats
import requests
from tqdm import tqdm

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
log = logging.getLogger(__name__)

# ─── CONFIG ──────────────────────────────────────────────────────────────────
COLOMBIA_BB = {'lon_min': -79.0, 'lon_max': -66.0, 'lat_min': 1.0, 'lat_max': 13.0}
CROPS = ['RICE', 'MAIZ', 'SUGC', 'ACOF', 'POTA']
YEARS = list(range(2005, 2026))
MONTHS = list(range(1, 13))

GROWING_SEASONS = {
    'RICE': (4, 9),
    'MAIZ': (3, 8),
    'SUGC': (3, 9),
    'ACOF': (5, 8),
    'POTA': (2, 6),
}

BENCHMARKS = {
    'RICE':  {'yield_t_ha': 5.5,  'price_usd_t': 250},
    'MAIZ':  {'yield_t_ha': 3.2,  'price_usd_t': 180},
    'SUGC':  {'yield_t_ha': 72.0, 'price_usd_t': 35},
    'ACOF':  {'yield_t_ha': 0.9,  'price_usd_t': 2800},
    'POTA':  {'yield_t_ha': 20.0, 'price_usd_t': 220},
}

CROP_PRODUCT_MAP = {
    'RICE': 'RICE',
    'MAIZ': 'MAIZE',
    'SUGC': 'SUGC',
    'ACOF': 'ACOF',
    'POTA': 'POTA',
}

PERILS = ['FLOOD', 'DROUGHT', 'HEAT', 'COLD']

DATA_DIR = Path('./data')
CHIRPS_DIR = DATA_DIR / 'chirps'
CHIRTS_DIR = DATA_DIR / 'chirts'
SPAM_DIR = DATA_DIR / 'spam'
OUT_DIR = Path('./outputs')


# ─── STAGE 1: DATA ───────────────────────────────────────────────────────────

def download_file(url, dest_path, timeout=60, max_mb=200):
    try:
        dest_path = Path(dest_path)
        if dest_path.exists():
            return True
        log.info(f"Downloading {url}")
        r = requests.get(url, stream=True, timeout=timeout)
        if r.status_code != 200:
            log.warning(f"HTTP {r.status_code} for {url}")
            return False
        total = int(r.headers.get('content-length', 0))
        if total > max_mb * 1024 * 1024:
            log.warning(f"File too large ({total/1e6:.0f}MB > {max_mb}MB limit), skipping")
            return False
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        with open(dest_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=1024*1024):
                if chunk:
                    f.write(chunk)
        log.info(f"Downloaded {dest_path.name} ({dest_path.stat().st_size/1e6:.1f}MB)")
        return True
    except Exception as e:
        log.warning(f"Download failed {url}: {e}")
        try:
            if Path(dest_path).exists():
                Path(dest_path).unlink()
        except Exception:
            pass
        return False


def get_synthetic_exposure_zones():
    log.info("Using synthetic exposure zones (fallback)")
    dept_centers = {
        'Cordoba':        (-75.9, 8.3),
        'Antioquia':      (-75.5, 6.5),
        'Cundinamarca':   (-74.3, 4.7),
        'Boyaca':         (-73.4, 5.5),
        'Valle_del_Cauca':(-76.5, 3.8),
        'Tolima':         (-75.2, 4.0),
        'Meta':           (-73.6, 3.5),
        'Huila':          (-75.5, 2.5),
        'Narino':         (-77.5, 1.5),
        'Cesar':          (-73.5, 9.5),
    }
    crop_dept_ha = {
        'RICE': [12000, 8000, 5000, 6000, 9000, 7000, 15000, 4000, 3000, 11000],
        'MAIZ': [15000, 12000, 8000, 10000, 6000, 9000, 20000, 5000, 7000, 13000],
        'SUGC': [2000,  3000,  1000, 1500,  45000,3000,  4000, 2500, 1000, 2000],
        'ACOF': [3000,  20000, 15000,18000, 5000, 12000, 8000, 25000,10000, 6000],
        'POTA': [1000,  8000,  25000,30000, 5000, 6000,  2000, 4000, 3000, 2000],
    }
    np.random.seed(42)
    zones = []
    depts = list(dept_centers.items())
    for crop in CROPS:
        has = crop_dept_ha[crop]
        total_ha = sum(has)
        cumulative = 0
        for i, (dept, (lon, lat)) in enumerate(depts):
            if cumulative / total_ha >= 0.80:
                break
            ha = has[i]
            lon_j = lon + np.random.uniform(-0.2, 0.2)
            lat_j = lat + np.random.uniform(-0.2, 0.2)
            zones.append({'crop': crop, 'dept': dept, 'lon': lon_j, 'lat': lat_j, 'ha': ha})
            cumulative += ha
    return pd.DataFrame(zones)


def load_spam_exposure_zones():
    try:
        import rasterio
        from rasterio.windows import from_bounds
        zones = []
        for crop in CROPS:
            tif_path = SPAM_DIR / f'spam2017v2r1_global_H_{crop}_A.tif'
            url = f'https://files.mapspam.info/data/v2.0.0/2017/global/geotiff/spam2017v2r1_global_H_{crop}_A.tif'
            ok = download_file(url, tif_path, timeout=60, max_mb=200)
            if not ok or not tif_path.exists():
                log.warning(f"SPAM download failed for {crop}")
                continue
            with rasterio.open(tif_path) as src:
                bb = COLOMBIA_BB
                window = from_bounds(bb['lon_min'], bb['lat_min'], bb['lon_max'], bb['lat_max'], src.transform)
                data = src.read(1, window=window, boundless=True, fill_value=0)
                transform = src.window_transform(window)
            data = np.where(data < 0, 0, data)
            total = data.sum()
            if total == 0:
                continue
            flat = data.flatten()
            idx_sorted = np.argsort(flat)[::-1]
            cumulative = 0
            selected = []
            for idx in idx_sorted:
                if flat[idx] <= 0:
                    break
                row, col = divmod(idx, data.shape[1])
                lon = transform.c + (col + 0.5) * transform.a
                lat = transform.f + (row + 0.5) * transform.e
                selected.append({'crop': crop, 'lon': float(lon), 'lat': float(lat), 'ha': float(flat[idx])})
                cumulative += flat[idx]
                if cumulative >= 0.8 * total:
                    break
            zones.extend(selected)
            log.info(f"SPAM {crop}: {len(selected)} zones, {total:.0f} ha total")
        if len(zones) == 0:
            return get_synthetic_exposure_zones()
        df = pd.DataFrame(zones)
        missing = [c for c in CROPS if c not in df['crop'].values]
        if missing:
            syn = get_synthetic_exposure_zones()
            df = pd.concat([df, syn[syn['crop'].isin(missing)]], ignore_index=True)
        return df
    except Exception as e:
        log.warning(f"SPAM failed: {e}")
        return get_synthetic_exposure_zones()


def extract_raster_value_at_point(tif_path, lon, lat):
    try:
        import rasterio
        with rasterio.open(tif_path) as src:
            row, col = src.index(lon, lat)
            row = max(0, min(row, src.height - 1))
            col = max(0, min(col, src.width - 1))
            data = src.read(1, window=((row, row+1), (col, col+1)))
            val = float(data[0, 0])
            nodata = src.nodata
            if nodata is not None and val == nodata:
                return np.nan
            if val < -9000:
                return np.nan
            return val
    except Exception:
        return np.nan


def download_chirps_monthly(year, month):
    fname = f'chirps-v2.0.{year}.{month:02d}.tif'
    dest = CHIRPS_DIR / fname
    if dest.exists():
        return dest
    url = f'https://data.chc.ucsb.edu/products/CHIRPS-2.0/global_monthly/tifs/{fname}'
    ok = download_file(url, dest, timeout=120, max_mb=150)
    return dest if (ok and dest.exists()) else None


def download_chirts_monthly(year, month):
    fname = f'Tmax.{year}.{month:02d}.tif'
    dest = CHIRTS_DIR / fname
    if dest.exists():
        return dest
    url = f'https://data.chc.ucsb.edu/products/CHIRTSdaily/v1.0/global_monthly_tifs/Tmax/{fname}'
    ok = download_file(url, dest, timeout=120, max_mb=150)
    return dest if (ok and dest.exists()) else None


def interpolate_missing(arr, missing_list):
    n_zones, n_years, n_months = arr.shape
    for year, month in missing_list:
        if year not in YEARS:
            continue
        yi = YEARS.index(year)
        mi = month - 1
        vals = []
        for dyi, dmi in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nyi = yi + dyi
            nmi = mi + dmi
            if 0 <= nyi < n_years and 0 <= nmi < n_months:
                v = arr[:, nyi, nmi]
                if not np.all(np.isnan(v)):
                    vals.append(v)
        if vals:
            arr[:, yi, mi] = np.nanmean(vals, axis=0)
    return arr


def build_climate_series(exposure_df):
    zones = exposure_df.to_dict('records')
    n_zones = len(zones)
    chirps = np.full((n_zones, len(YEARS), 12), np.nan)
    chirts = np.full((n_zones, len(YEARS), 12), np.nan)
    missing_chirps = []
    missing_chirts = []

    total = len(YEARS) * 12
    with tqdm(total=total, desc="Climate data") as pbar:
        for yi, year in enumerate(YEARS):
            for mi, month in enumerate(MONTHS):
                path_c = download_chirps_monthly(year, month)
                if path_c:
                    for zi, z in enumerate(zones):
                        chirps[zi, yi, mi] = extract_raster_value_at_point(path_c, z['lon'], z['lat'])
                else:
                    missing_chirps.append((year, month))

                path_t = download_chirts_monthly(year, month)
                if path_t:
                    for zi, z in enumerate(zones):
                        chirts[zi, yi, mi] = extract_raster_value_at_point(path_t, z['lon'], z['lat'])
                else:
                    missing_chirts.append((year, month))

                pbar.update(1)

    if missing_chirps:
        chirps = interpolate_missing(chirps, missing_chirps)
    if missing_chirts:
        chirts = interpolate_missing(chirts, missing_chirts)

    return chirps, chirts


def generate_synthetic_climate(exposure_df):
    log.info("Generating synthetic climate data")
    np.random.seed(42)
    n_zones = len(exposure_df)
    base_rain = np.array([80, 90, 130, 180, 200, 160, 130, 140, 180, 210, 160, 100], dtype=float)
    base_tmax = np.array([30, 31, 31, 30, 29, 28, 28, 29, 29, 29, 29, 30], dtype=float)

    # Known ENSO signal (stronger interannual variability)
    enso_years = {
        2005: -0.1, 2006: 0.5, 2007: -0.8, 2008: -0.5, 2009: 0.4,
        2010: -1.2, 2011: -1.0, 2012: 0.1, 2013: -0.2, 2014: 0.5,
        2015: 1.4, 2016: 0.6, 2017: -0.4, 2018: 0.5, 2019: 0.3,
        2020: -0.8, 2021: -0.8, 2022: -1.0, 2023: 1.2, 2024: 0.8,
        2025: -0.3,
    }

    chirps = np.zeros((n_zones, len(YEARS), 12))
    chirts = np.zeros((n_zones, len(YEARS), 12))

    for zi in range(n_zones):
        lat = float(exposure_df.iloc[zi]['lat'])
        lon = float(exposure_df.iloc[zi]['lon'])
        lat_factor = 1 + 0.04 * (lat - 7)
        lon_factor = 1 + 0.02 * (lon - (-72))
        for yi, year in enumerate(YEARS):
            enso = enso_years.get(year, 0)
            np.random.seed(42 + zi * 100 + yi)
            for mi in range(12):
                noise_r = np.random.normal(0, 0.12)
                rain = base_rain[mi] * lat_factor * lon_factor * (1 - 0.15 * enso + noise_r)
                chirps[zi, yi, mi] = max(0, rain)
                noise_t = np.random.normal(0, 0.4)
                chirts[zi, yi, mi] = base_tmax[mi] + 0.8 * enso + noise_t

    return chirps, chirts


# ─── STAGE 3: INDICES ────────────────────────────────────────────────────────

def compute_rainfall_index(chirps, exposure_df):
    records = []
    zones = exposure_df.reset_index(drop=True)
    for crop in CROPS:
        idx_list = zones.index[zones['crop'] == crop].tolist()
        if not idx_list:
            continue
        s_month, e_month = GROWING_SEASONS[crop]
        season_months = list(range(s_month - 1, e_month))
        bench = BENCHMARKS[crop]
        pname = CROP_PRODUCT_MAP[crop]

        season_rain = chirps[np.ix_(idx_list, range(len(YEARS)), season_months)].sum(axis=2)
        annual_flood = np.zeros(len(YEARS))
        annual_drought = np.zeros(len(YEARS))

        for li, gi in enumerate(idx_list):
            ha = float(zones.loc[gi, 'ha'])
            rain = season_rain[li, :]
            rain = np.where(np.isnan(rain), np.nanmean(rain) if not np.all(np.isnan(rain)) else 100, rain)
            rain = np.maximum(rain, 1e-3)

            try:
                a, loc, scale = stats.gamma.fit(rain, floc=0)
                pct = stats.gamma.cdf(rain, a, loc=loc, scale=scale) * 100
            except Exception:
                pct = stats.rankdata(rain) / (len(rain) + 1) * 100

            p1 = np.percentile(pct, 1)
            p10 = np.percentile(pct, 10)
            p90 = np.percentile(pct, 90)
            p99 = np.percentile(pct, 99)
            max_pay = bench['yield_t_ha'] * ha * bench['price_usd_t']

            drought_share = np.where(pct <= 10, np.clip((p10 - pct) / max(p10 - p1, 1e-6), 0, 1), 0)
            flood_share   = np.where(pct >= 90, np.clip((pct - p90) / max(p99 - p90, 1e-6), 0, 1), 0)

            annual_drought += drought_share * max_pay
            annual_flood   += flood_share   * max_pay

        for yi, year in enumerate(YEARS):
            records.append({'crop': crop, 'product_name': pname, 'year': year,
                            'payout_flood': annual_flood[yi], 'payout_drought': annual_drought[yi]})
    return pd.DataFrame(records)


def compute_temperature_index(chirts, exposure_df):
    records = []
    zones = exposure_df.reset_index(drop=True)
    for crop in CROPS:
        idx_list = zones.index[zones['crop'] == crop].tolist()
        if not idx_list:
            continue
        s_month, e_month = GROWING_SEASONS[crop]
        season_months = list(range(s_month - 1, e_month))
        bench = BENCHMARKS[crop]
        pname = CROP_PRODUCT_MAP[crop]

        season_t = chirts[np.ix_(idx_list, range(len(YEARS)), season_months)]
        annual_heat = np.zeros(len(YEARS))
        annual_cold = np.zeros(len(YEARS))

        for li, gi in enumerate(idx_list):
            ha = float(zones.loc[gi, 'ha'])
            T = season_t[li]  # (n_years, n_season_months)
            mu_month = np.nanmean(T, axis=0)
            anom = T - mu_month[np.newaxis, :]
            mean_anom = np.nanmean(anom, axis=1)
            max_pay = bench['yield_t_ha'] * ha * bench['price_usd_t']

            heat_share = np.where(mean_anom > 2, np.clip((mean_anom - 2) / (4 - 2), 0, 1), 0)
            cold_share = np.where(mean_anom < -2, np.clip((-2 - mean_anom) / (-2 - (-4)), 0, 1), 0)

            annual_heat += heat_share * max_pay
            annual_cold += cold_share * max_pay

        for yi, year in enumerate(YEARS):
            records.append({'crop': crop, 'product_name': pname, 'year': year,
                            'payout_heat': annual_heat[yi], 'payout_cold': annual_cold[yi]})
    return pd.DataFrame(records)


def build_payout_matrix(rain_df, temp_df):
    payout_dict = {}
    for crop in CROPS:
        pname = CROP_PRODUCT_MAP[crop]
        r = rain_df[rain_df['crop'] == crop].sort_values('year')
        t = temp_df[temp_df['crop'] == crop].sort_values('year')
        payout_dict[f'COL-{pname}-FLOOD']   = r['payout_flood'].values
        payout_dict[f'COL-{pname}-DROUGHT'] = r['payout_drought'].values
        payout_dict[f'COL-{pname}-HEAT']    = t['payout_heat'].values
        payout_dict[f'COL-{pname}-COLD']    = t['payout_cold'].values
    return pd.DataFrame(payout_dict, index=YEARS)


# ─── STAGE 4: MVB ────────────────────────────────────────────────────────────

def build_mvb(payout_df):
    from sklearn.covariance import LedoitWolf

    products = list(payout_df.columns)
    n = len(products)

    pure_premium = payout_df.mean(axis=0)
    gross_premium = pure_premium / 0.65
    # Guard against zeros
    min_gp = gross_premium[gross_premium > 0].min() if (gross_premium > 0).any() else 1.0
    gross_premium = gross_premium.clip(lower=min_gp * 0.001)

    lr_df = payout_df.div(gross_premium, axis=1)

    lw = LedoitWolf()
    lw.fit(lr_df.values)
    Sigma_hat = lw.covariance_

    try:
        import cvxpy as cp
        w = cp.Variable(n)
        obj = cp.Minimize(cp.quad_form(w, Sigma_hat))
        constraints = [cp.sum(w) == 1, w >= 0, w <= 0.15]
        prob = cp.Problem(obj, constraints)
        prob.solve(solver=cp.SCS, verbose=False)
        if prob.status in ['optimal', 'optimal_inaccurate'] and w.value is not None:
            weights = np.maximum(np.array(w.value).flatten(), 0)
            weights /= weights.sum()
            log.info(f"CVX succeeded (status={prob.status})")
        else:
            raise ValueError(f"CVX status: {prob.status}")
    except Exception as e:
        log.warning(f"CVX failed ({e}), using equal weights")
        weights = np.ones(n) / n

    metrics = []
    for i, prod in enumerate(products):
        lr = lr_df[prod].values
        metrics.append({
            'product': prod,
            'mean_payout_usd': float(pure_premium[prod]),
            'gross_premium_usd': float(gross_premium[prod]),
            'mean_lr': float(np.mean(lr)),
            'std_lr': float(np.std(lr)),
            'cv_lr': float(np.std(lr) / max(np.mean(lr), 1e-9)),
            'p95_lr': float(np.percentile(lr, 95)),
            'weight': float(weights[i]),
        })
    metrics_df = pd.DataFrame(metrics)

    portfolio_lr = lr_df.values @ weights
    weighted_avg_cv = float((metrics_df['cv_lr'] * metrics_df['weight']).sum())
    portfolio_cv = float(np.std(portfolio_lr) / max(np.mean(portfolio_lr), 1e-9))
    variance_reduction = (weighted_avg_cv - portfolio_cv) / max(weighted_avg_cv, 1e-9)

    bundle_row = {
        'product': 'MVB_PORTFOLIO',
        'mean_payout_usd': float((payout_df.values @ weights).mean()),
        'gross_premium_usd': float((gross_premium.values * weights).sum()),
        'mean_lr': float(np.mean(portfolio_lr)),
        'std_lr': float(np.std(portfolio_lr)),
        'cv_lr': portfolio_cv,
        'p95_lr': float(np.percentile(portfolio_lr, 95)),
        'weight': 1.0,
    }
    summary_df = pd.concat([metrics_df, pd.DataFrame([bundle_row])], ignore_index=True)

    return {
        'weights': weights, 'products': products, 'lr_df': lr_df,
        'portfolio_lr': portfolio_lr, 'gross_premium': gross_premium,
        'pure_premium': pure_premium, 'metrics_df': metrics_df,
        'summary_df': summary_df, 'weighted_avg_cv': weighted_avg_cv,
        'portfolio_cv': portfolio_cv, 'variance_reduction': variance_reduction,
    }


# ─── STAGE 5: CHARTS ─────────────────────────────────────────────────────────

CROP_COLORS = {
    'RICE': '#2196F3', 'MAIZ': '#FF9800', 'SUGC': '#4CAF50',
    'ACOF': '#795548', 'POTA': '#9C27B0',
}

def crop_color(prod):
    for k in CROPS:
        if CROP_PRODUCT_MAP[k] in prod or k in prod:
            return CROP_COLORS[k]
    return '#999999'


def plot_exposure_map(exposure_df):
    fig, ax = plt.subplots(figsize=(8, 8))
    for crop in CROPS:
        sub = exposure_df[exposure_df['crop'] == crop]
        if sub.empty: continue
        max_ha = sub['ha'].max()
        sizes = (sub['ha'] / max_ha * 200).clip(10, 200)
        ax.scatter(sub['lon'], sub['lat'], s=sizes, c=CROP_COLORS[crop], alpha=0.75,
                   label=CROP_PRODUCT_MAP[crop], edgecolors='white', linewidths=0.5)
    bb = COLOMBIA_BB
    ax.set_xlim(bb['lon_min'] - 0.5, bb['lon_max'] + 0.5)
    ax.set_ylim(bb['lat_min'] - 0.5, bb['lat_max'] + 0.5)
    # Simple Colombia outline
    col_pts = [(-77.2,8.6),(-76.9,8.1),(-77.4,7.7),(-77.2,6.5),(-77.5,5.6),
               (-78.0,4.8),(-77.9,2.7),(-76.5,1.4),(-75.7,0.8),(-74.5,1.5),
               (-74.0,1.0),(-72.0,2.0),(-70.0,2.1),(-67.9,2.4),(-67.5,2.5),
               (-67.1,3.8),(-67.4,6.0),(-67.8,6.4),(-68.2,7.5),(-69.4,7.5),
               (-70.1,7.7),(-71.1,7.5),(-72.5,8.0),(-73.0,9.3),(-74.0,11.1),
               (-74.5,11.0),(-74.6,10.6),(-75.6,10.9),(-76.9,10.3),(-76.5,9.5),(-77.2,8.6)]
    xs, ys = zip(*col_pts)
    ax.plot(xs, ys, 'k-', linewidth=1.5, alpha=0.5)
    ax.set_xlabel('Longitude'); ax.set_ylabel('Latitude')
    ax.set_title('Colombia Crop Exposure Zones\n(dot size proportional to harvested area ha)')
    ax.legend(title='Crop', loc='lower right')
    ax.grid(True, alpha=0.3); ax.set_facecolor('#E8F5E9')
    plt.tight_layout()
    plt.savefig(OUT_DIR / 'exposure_map.png', dpi=150, bbox_inches='tight')
    plt.close()
    log.info("Saved exposure_map.png")


def plot_payout_heatmap(payout_df):
    fig, ax = plt.subplots(figsize=(14, 8))
    data = payout_df.T / 1e6
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list('wr', ['white', '#B71C1C'])
    sns.heatmap(data, ax=ax, cmap=cmap, linewidths=0.3, linecolor='#ddd',
                cbar_kws={'label': 'Payout (USD millions)'})
    ax.set_title('Annual Payouts by Product (2005–2025)\nWhite = Zero Payout')
    ax.set_xlabel('Year'); ax.set_ylabel('Product')
    ax.tick_params(axis='x', rotation=45); ax.tick_params(axis='y', rotation=0)
    plt.tight_layout()
    plt.savefig(OUT_DIR / 'payout_heatmap.png', dpi=150, bbox_inches='tight')
    plt.close()
    log.info("Saved payout_heatmap.png")


def plot_loss_ratio_heatmap(lr_df):
    fig, ax = plt.subplots(figsize=(14, 8))
    data = lr_df.T
    sns.heatmap(data, ax=ax, cmap='RdYlGn_r', center=0.65, vmin=0, vmax=2,
                linewidths=0.3, linecolor='#ddd', cbar_kws={'label': 'Loss Ratio'})
    ax.set_title('Annual Loss Ratios by Product (2005–2025)')
    ax.set_xlabel('Year'); ax.set_ylabel('Product')
    ax.tick_params(axis='x', rotation=45); ax.tick_params(axis='y', rotation=0)
    plt.tight_layout()
    plt.savefig(OUT_DIR / 'loss_ratio_heatmap.png', dpi=150, bbox_inches='tight')
    plt.close()
    log.info("Saved loss_ratio_heatmap.png")


def plot_correlation_matrix(lr_df):
    corr = lr_df.corr()
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(corr, ax=ax, cmap='RdBu_r', center=0, vmin=-1, vmax=1,
                annot=True, fmt='.2f', annot_kws={'size': 6},
                linewidths=0.5, square=True, cbar_kws={'label': 'Correlation'})
    ax.set_title('Loss Ratio Pairwise Correlation Matrix')
    ax.tick_params(axis='x', rotation=90); ax.tick_params(axis='y', rotation=0)
    plt.tight_layout()
    plt.savefig(OUT_DIR / 'correlation_matrix.png', dpi=150, bbox_inches='tight')
    plt.close()
    log.info("Saved correlation_matrix.png")


def plot_mvb_weights(mvb):
    df = pd.DataFrame({'product': mvb['products'], 'weight': mvb['weights']})
    df = df.sort_values('weight', ascending=True)
    colors = [crop_color(p) for p in df['product']]
    fig, ax = plt.subplots(figsize=(8, 10))
    bars = ax.barh(df['product'], df['weight'], color=colors, edgecolor='white', height=0.7)
    for bar, w in zip(bars, df['weight']):
        ax.text(bar.get_width() + 0.0005, bar.get_y() + bar.get_height()/2,
                f'{w:.4f}', va='center', ha='left', fontsize=8)
    ax.axvline(1/20, color='gray', linestyle='--', alpha=0.6, linewidth=1.5)
    patches = [mpatches.Patch(color=CROP_COLORS[c], label=CROP_PRODUCT_MAP[c]) for c in CROPS]
    eq_line = plt.Line2D([0],[0], color='gray', linestyle='--', label='Equal weight (0.05)')
    ax.legend(handles=patches + [eq_line], loc='lower right', fontsize=8)
    ax.set_xlabel('Optimal Weight')
    ax.set_title('Minimum Variance Bundle — Optimal Weights')
    plt.tight_layout()
    plt.savefig(OUT_DIR / 'mvb_weights.png', dpi=150, bbox_inches='tight')
    plt.close()
    log.info("Saved mvb_weights.png")


def plot_cv_comparison(mvb):
    metrics_df = mvb['metrics_df']
    portfolio_cv = mvb['portfolio_cv']
    var_red = mvb['variance_reduction']
    lr_df = mvb['lr_df']
    eq_w = np.ones(len(lr_df.columns)) / len(lr_df.columns)
    eq_lr = lr_df.values @ eq_w
    eq_cv = float(np.std(eq_lr) / max(np.mean(eq_lr), 1e-9))

    x = list(metrics_df['product']) + ['EQ_PORTFOLIO', 'MVB_PORTFOLIO']
    y = list(metrics_df['cv_lr']) + [eq_cv, portfolio_cv]
    colors = [crop_color(p) for p in metrics_df['product']] + ['#607D8B', '#F44336']

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(x, y, color=colors, edgecolor='white', alpha=0.85)
    ax.axhline(portfolio_cv, color='#F44336', linestyle='--', linewidth=1.5,
               label=f'MVB CV = {portfolio_cv:.3f}')
    ax.set_xticks(range(len(x)))
    ax.set_xticklabels(x, rotation=90, fontsize=7)
    ax.set_ylabel('Coefficient of Variation (Loss Ratio)')
    ax.set_title(f'CV Comparison — MVB Variance Reduction vs Weighted-Avg Individual: {var_red*100:.1f}%')
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(OUT_DIR / 'cv_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    log.info("Saved cv_comparison.png")


def plot_portfolio_lr_ts(mvb):
    plr = mvb['portfolio_lr']
    mu = np.mean(plr); sigma = np.std(plr)
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(YEARS, plr, 'b-o', linewidth=2, markersize=5, label='MVB Loss Ratio')
    ax.axhline(0.65, color='red', linestyle='--', linewidth=1.5, label='L* = 0.65')
    ax.fill_between(YEARS, mu - sigma, mu + sigma, alpha=0.15, color='blue', label='±1 Std Dev')
    ax.axhline(mu, color='blue', linestyle=':', linewidth=1, alpha=0.6, label=f'Mean = {mu:.3f}')
    ax.set_xlabel('Year'); ax.set_ylabel('Loss Ratio')
    ax.set_title('MVB Portfolio Annual Loss Ratio (2005–2025)')
    ax.legend(); ax.grid(True, alpha=0.3)
    ax.set_xlim(YEARS[0]-0.5, YEARS[-1]+0.5)
    plt.tight_layout()
    plt.savefig(OUT_DIR / 'portfolio_loss_ratio_ts.png', dpi=150, bbox_inches='tight')
    plt.close()
    log.info("Saved portfolio_loss_ratio_ts.png")


def download_oni():
    url = 'https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt'
    try:
        r = requests.get(url, timeout=30)
        if r.status_code != 200:
            raise ValueError(f"HTTP {r.status_code}")
        records = []
        for line in r.text.strip().split('\n'):
            parts = line.split()
            if len(parts) >= 3:
                try:
                    records.append({'year': int(parts[0]), 'month': int(parts[1]), 'oni': float(parts[2])})
                except ValueError:
                    continue
        df = pd.DataFrame(records)
        return df.groupby('year')['oni'].mean().reset_index()
    except Exception as e:
        log.warning(f"ONI download failed ({e}), using synthetic")
        data = {2005:-0.1,2006:0.5,2007:-0.8,2008:-0.5,2009:0.4,2010:-1.2,2011:-1.0,
                2012:0.1,2013:-0.2,2014:0.5,2015:1.4,2016:0.6,2017:-0.4,2018:0.5,
                2019:0.3,2020:-0.8,2021:-0.8,2022:-1.0,2023:1.2,2024:0.8,2025:-0.3}
        return pd.DataFrame([{'year':k,'oni':v} for k,v in data.items()])


def plot_enso_overlay(mvb):
    plr = mvb['portfolio_lr']
    mu = np.mean(plr); sigma = np.std(plr)
    oni_df = download_oni()
    oni_dict = dict(zip(oni_df['year'], oni_df['oni']))
    oni_vals = np.array([oni_dict.get(y, 0) for y in YEARS])

    fig, ax1 = plt.subplots(figsize=(14, 6))
    ax2 = ax1.twinx()

    for yi, year in enumerate(YEARS):
        oni = oni_dict.get(year, 0)
        if oni > 0.5:
            ax1.axvspan(year-0.5, year+0.5, alpha=0.12, color='red', zorder=0)
        elif oni < -0.5:
            ax1.axvspan(year-0.5, year+0.5, alpha=0.12, color='blue', zorder=0)

    ax1.plot(YEARS, plr, 'k-o', linewidth=2, markersize=5, label='MVB Loss Ratio', zorder=5)
    ax1.axhline(0.65, color='red', linestyle='--', linewidth=1.5, label='L* = 0.65')
    ax1.fill_between(YEARS, mu-sigma, mu+sigma, alpha=0.12, color='gray', label='±1 Std Dev')

    ax2.plot(YEARS, oni_vals, 'g--', linewidth=1.5, alpha=0.7, label='ONI')
    ax2.axhline(0.5, color='red', linestyle=':', alpha=0.3, linewidth=1)
    ax2.axhline(-0.5, color='blue', linestyle=':', alpha=0.3, linewidth=1)
    ax2.set_ylabel('ONI Index', color='green')
    ax2.tick_params(axis='y', labelcolor='green')

    ax1.set_xlabel('Year'); ax1.set_ylabel('Loss Ratio')
    ax1.set_title('MVB Portfolio Loss Ratio with ENSO Overlay\n(Red shading = El Niño ONI>0.5 | Blue = La Niña ONI<-0.5)')
    ax1.set_xlim(YEARS[0]-0.5, YEARS[-1]+0.5)
    ax1.grid(True, alpha=0.3)

    l1, lb1 = ax1.get_legend_handles_labels()
    l2, lb2 = ax2.get_legend_handles_labels()
    el_patch = mpatches.Patch(color='red', alpha=0.3, label='El Niño')
    la_patch = mpatches.Patch(color='blue', alpha=0.3, label='La Niña')
    ax1.legend(handles=l1+l2+[el_patch, la_patch], loc='upper left', fontsize=8)

    plt.tight_layout()
    plt.savefig(OUT_DIR / 'enso_overlay.png', dpi=150, bbox_inches='tight')
    plt.close()
    log.info("Saved enso_overlay.png")


def print_summary_table(mvb):
    summary = mvb['summary_df']
    print("\n" + "="*105)
    print("  COLOMBIA MINIMUM VARIANCE PARAMETRIC INSURANCE BUNDLE — SUMMARY")
    print("="*105)
    hdr = f"{'Product':<25} {'Mean Payout':>14} {'Gross Premium':>14} {'Mean LR':>9} {'Std LR':>9} {'CV':>8} {'P95 LR':>9} {'Weight':>8}"
    print(hdr)
    print("-"*105)
    for _, row in summary.iterrows():
        if row['product'] == 'MVB_PORTFOLIO':
            print("─"*105)
        print(f"{row['product']:<25} ${row['mean_payout_usd']:>13,.0f} ${row['gross_premium_usd']:>13,.0f} "
              f"{row['mean_lr']:>9.4f} {row['std_lr']:>9.4f} {row['cv_lr']:>8.4f} "
              f"{row['p95_lr']:>9.4f} {row['weight']:>8.4f}")
    print("="*105)
    print(f"\nVariance Reduction (MVB vs weighted-avg individual): {mvb['variance_reduction']*100:.1f}%")
    print(f"Weighted-avg individual CV : {mvb['weighted_avg_cv']:.4f}")
    print(f"MVB Portfolio CV           : {mvb['portfolio_cv']:.4f}")
    print()


# ─── MAIN ────────────────────────────────────────────────────────────────────

def main():
    np.random.seed(42)
    log.info("="*60)
    log.info("COLOMBIA PARAMETRIC INSURANCE BUNDLE PIPELINE START")
    log.info("="*60)

    # Stage 1: Exposure zones
    log.info("\n--- STAGE 1: Exposure Zones ---")
    try:
        exposure_df = load_spam_exposure_zones()
    except Exception as e:
        log.error(f"Exposure stage failed: {e}")
        exposure_df = get_synthetic_exposure_zones()
    log.info(f"Zones: {len(exposure_df)} across {exposure_df['crop'].nunique()} crops")

    # Stage 1b: Climate
    log.info("\n--- STAGE 1b: Climate Data ---")
    try:
        chirps, chirts = build_climate_series(exposure_df)
        nan_c = np.isnan(chirps).mean()
        nan_t = np.isnan(chirts).mean()
        log.info(f"NaN fraction — CHIRPS: {nan_c:.1%}, CHIRTS: {nan_t:.1%}")
        if nan_c > 0.7 and nan_t > 0.7:
            raise ValueError("Too many NaNs, switching to synthetic")
    except Exception as e:
        log.warning(f"Climate data failed ({e}), using synthetic")
        chirps, chirts = generate_synthetic_climate(exposure_df)

    # Fill any remaining NaNs
    if np.isnan(chirps).any() or np.isnan(chirts).any():
        syn_c, syn_t = generate_synthetic_climate(exposure_df)
        chirps = np.where(np.isnan(chirps), syn_c, chirps)
        chirts = np.where(np.isnan(chirts), syn_t, chirts)

    # Stage 3: Payouts
    log.info("\n--- STAGE 3: Indices & Payouts ---")
    try:
        rain_df = compute_rainfall_index(chirps, exposure_df)
        temp_df = compute_temperature_index(chirts, exposure_df)
        payout_df = build_payout_matrix(rain_df, temp_df)
        log.info(f"Payout matrix: {payout_df.shape}, total products: {len(payout_df.columns)}")
    except Exception as e:
        log.error(f"Payout stage failed: {e}\n{traceback.format_exc()}")
        sys.exit(1)

    # Stage 4: MVB
    log.info("\n--- STAGE 4: Minimum Variance Bundle ---")
    try:
        mvb = build_mvb(payout_df)
        log.info(f"Variance reduction: {mvb['variance_reduction']*100:.1f}%")
    except Exception as e:
        log.error(f"MVB failed: {e}\n{traceback.format_exc()}")
        sys.exit(1)

    # Stage 5: Charts
    log.info("\n--- STAGE 5: Charts ---")
    for fn, args in [
        (plot_exposure_map, (exposure_df,)),
        (plot_payout_heatmap, (payout_df,)),
        (plot_loss_ratio_heatmap, (mvb['lr_df'],)),
        (plot_correlation_matrix, (mvb['lr_df'],)),
        (plot_mvb_weights, (mvb,)),
        (plot_cv_comparison, (mvb,)),
        (plot_portfolio_lr_ts, (mvb,)),
        (plot_enso_overlay, (mvb,)),
    ]:
        try:
            fn(*args)
        except Exception as e:
            log.error(f"{fn.__name__} failed: {e}\n{traceback.format_exc()}")

    print_summary_table(mvb)
    pngs = list(OUT_DIR.glob('*.png'))
    log.info(f"Done. {len(pngs)} charts in ./outputs/: {[f.name for f in pngs]}")


if __name__ == '__main__':
    main()
