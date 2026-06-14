"""
Reinsurer presentation maps — Drought vs Flood peril comparison
Reads from the already-run pipeline outputs.
"""

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import box, Point, Polygon
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

OUT = "outputs/maps"

# ── Reproduce grid and zone assignments (same as pipeline) ───────────────────
RES = 0.1
ZONES = {
    'A': dict(lat_min=3, lat_max=7,  lon_min=-73, lon_max=-68, crops={'maize':0.60,'rice':0.40},      frac=0.25, penetration=0.03),
    'B': dict(lat_min=3, lat_max=5,  lon_min=-77, lon_max=-75, crops={'sugarcane':0.55,'maize':0.45}, frac=0.45, penetration=0.08),
    'C': dict(lat_min=8, lat_max=11, lon_min=-76, lon_max=-73, crops={'maize':0.50,'sorghum':0.30,'cotton':0.20}, frac=0.30, penetration=0.04),
    'D': dict(lat_min=1, lat_max=4,  lon_min=-76, lon_max=-73, crops={'coffee':0.50,'maize':0.30,'potato':0.20}, frac=0.20, penetration=0.03),
    'E': dict(lat_min=5, lat_max=9,  lon_min=-75, lon_max=-73, crops={'rice':0.45,'maize':0.35,'sorghum':0.20}, frac=0.25, penetration=0.04),
}
ZONE_NAMES = {'A':'Llanos Orientales','B':'Cauca Valley','C':'Atlantic Coast',
              'D':'Andean Foothills','E':'Magdalena Valley'}

def is_valid_zone(z):
    return z is not None and not (isinstance(z, float) and np.isnan(z))

def assign_zone(lat, lon):
    for zname, z in ZONES.items():
        if z['lat_min'] <= lat < z['lat_max'] and z['lon_min'] <= lon < z['lon_max']:
            return zname
    return None

# Build grid
lons = np.arange(-79.0, -67.0, RES)
lats = np.arange(0.0,   12.0,  RES)
cells = []
for lat in lats:
    for lon in lons:
        cells.append({'centroid_lon': lon+RES/2, 'centroid_lat': lat+RES/2,
                      'geometry': box(lon, lat, lon+RES, lat+RES)})
grid = gpd.GeoDataFrame(cells, crs="EPSG:4326")

# Colombia bounding polygon (fallback — same as pipeline)
colombia_poly = Polygon([(-79,0),(-67,0),(-67,12),(-79,12),(-79,0)])
colombia_gdf  = gpd.GeoDataFrame({'geometry':[colombia_poly]}, crs="EPSG:4326")

grid['zone'] = [assign_zone(r.centroid_lat, r.centroid_lon) for _, r in grid.iterrows()]
grid = grid[grid['zone'].apply(is_valid_zone)].reset_index(drop=True)
grid['grid_id'] = ['G%04d'%i for i in range(len(grid))]
grid['primary_crop'] = grid['zone'].apply(
    lambda z: max(ZONES[z]['crops'], key=ZONES[z]['crops'].get))
grid['cell_area_ha'] = (111320*np.cos(np.radians(grid['centroid_lat']))*RES/1000) * (110570*RES/1000)*100
grid['cropland_ha']  = grid['zone'].apply(lambda z: ZONES[z]['frac']) * grid['cell_area_ha']
grid['uninsured_ha'] = grid['cropland_ha'] * grid['zone'].apply(lambda z: 1 - ZONES[z]['penetration'])

# ── Reproduce rainfall + payouts (same RNG seed = same numbers) ──────────────
MAX_PAYOUT = {'maize':280,'rice':320,'sugarcane':350,'sorghum':240,'cotton':300,'coffee':420,'potato':380}
SEASONS = [
    dict(season_id='S1', zones=['B','C','E'], crops=['maize','sorghum','cotton','rice']),
    dict(season_id='S2', zones=['B','C','E'], crops=['maize','sorghum','sugarcane']),
    dict(season_id='S3', zones=['A'],         crops=['maize','rice']),
    dict(season_id='S4', zones=['D'],         crops=['coffee','maize','potato']),
]
GAMMA_PARAMS = {
    ('A','S3'):dict(mean=1850,cv=0.22), ('B','S1'):dict(mean=920,cv=0.28),
    ('B','S2'):dict(mean=880,cv=0.30),  ('C','S1'):dict(mean=680,cv=0.32),
    ('C','S2'):dict(mean=620,cv=0.35),  ('D','S4'):dict(mean=750,cv=0.25),
    ('E','S1'):dict(mean=800,cv=0.29),  ('E','S2'):dict(mean=760,cv=0.31),
}
ENSO_YEARS = {'El Niño':{1997,1998,2002,2003,2009,2010,2015,2016,2019},
              'La Niña':{1995,1996,1999,2000,2007,2008,2010,2011,2020,2021,2022}}
ENSO_MOD = {'El Niño':{'A':0.75,'B':1.10,'C':0.75,'D':1.10,'E':0.75},
            'La Niña':{'A':1.20,'B':0.85,'C':1.20,'D':0.85,'E':1.20},
            'Neutral': {z:1.00 for z in 'ABCDE'}}
def enso_phase(yr):
    for p,ys in ENSO_YEARS.items():
        if yr in ys: return p
    return 'Neutral'

YEARS = list(range(1994,2024))
N_YEARS = 30

# Build plots
plots = []
for _, cell in grid.iterrows():
    for s in SEASONS:
        if cell.zone in s['zones'] and cell.primary_crop in s['crops']:
            plots.append({'grid_id':cell.grid_id,'zone':cell.zone,'season_id':s['season_id'],
                          'crop_type':cell.primary_crop,'uninsured_ha':cell.uninsured_ha,
                          'max_payout_usd_per_ha':MAX_PAYOUT[cell.primary_crop]})
plots_df = pd.DataFrame(plots)
plots_df.insert(0,'plot_id',['P%05d'%i for i in range(len(plots_df))])
n_plots = len(plots_df)

# Zone shocks
zone_year_shocks = {}
for z in 'ABCDE':
    for yr in YEARS:
        matching = [v for (zz,ss),v in GAMMA_PARAMS.items() if zz==z]
        zone_mean = matching[0]['mean'] if matching else 800
        zone_year_shocks[(z,yr)] = np.random.normal(0, 0.08*zone_mean)

# Rainfall
rainfall = np.zeros((n_plots, N_YEARS))
for pidx, plot in plots_df.iterrows():
    z,sid = plot.zone, plot.season_id
    params = GAMMA_PARAMS.get((z,sid), dict(mean=700,cv=0.30))
    alpha = 1.0/params['cv']**2
    for yidx,yr in enumerate(YEARS):
        phase = enso_phase(yr)
        mod   = ENSO_MOD[phase][z]
        beta  = params['mean']*mod/alpha
        raw   = np.random.gamma(alpha,beta) + zone_year_shocks[(z,yr)]
        rainfall[pidx,yidx] = max(10.0, raw)

# Gamma fit → percentiles
from scipy.stats import gamma as gamma_dist
percentiles = np.zeros((n_plots,4))
for pidx in range(n_plots):
    s = rainfall[pidx]
    try:
        a,loc,sc = gamma_dist.fit(s,floc=0)
        from scipy import stats
        _,ksp = stats.kstest(s,'gamma',args=(a,loc,sc))
        if ksp >= 0.05:
            percentiles[pidx] = [gamma_dist.ppf(q,a,loc,sc) for q in [0.01,0.10,0.90,0.99]]
        else:
            percentiles[pidx] = [np.percentile(s,q) for q in [1,10,90,99]]
    except:
        percentiles[pidx] = [np.percentile(s,q) for q in [1,10,90,99]]

# Payouts
deficit_pct = np.zeros((n_plots,N_YEARS))
excess_pct  = np.zeros((n_plots,N_YEARS))
for pidx in range(n_plots):
    p1,p10,p90,p99 = percentiles[pidx]
    for yidx in range(N_YEARS):
        r = rainfall[pidx,yidx]
        deficit_pct[pidx,yidx] = max(0, min(1, (p10-r)/(p10-p1))) if p10>p1 else (1 if r<=p1 else 0)
        excess_pct[pidx,yidx]  = max(0, min(1, (r-p90)/(p99-p90))) if p99>p90 else (1 if r>=p99 else 0)

max_usd = plots_df['max_payout_usd_per_ha'].values * plots_df['uninsured_ha'].values
usd_def = deficit_pct * max_usd[:,None]
usd_exc = excess_pct  * max_usd[:,None]

# ── Per-cell aggregates ───────────────────────────────────────────────────────
# For each grid cell: mean annual payout, std, loss ratio (assuming 1.3x premium on mean)
cell_stats = []
for gid in grid['grid_id']:
    mask = plots_df['grid_id'] == gid
    if not mask.any():
        cell_stats.append({'grid_id':gid,'def_mean':0,'def_std':0,'def_cv':0,
                           'exc_mean':0,'exc_std':0,'exc_cv':0,
                           'def_lr':0,'exc_lr':0,'def_max_usd':0,'exc_max_usd':0})
        continue
    d_annual = usd_def[mask].sum(axis=0)   # total USD per year across plots in cell
    e_annual = usd_exc[mask].sum(axis=0)
    d_mean = d_annual.mean(); d_std = d_annual.std()
    e_mean = e_annual.mean(); e_std = e_annual.std()
    cell_stats.append({
        'grid_id':   gid,
        'def_mean':  d_mean,
        'def_std':   d_std,
        'def_cv':    d_std/d_mean if d_mean>0 else 0,
        'exc_mean':  e_mean,
        'exc_std':   e_std,
        'exc_cv':    e_std/e_mean if e_mean>0 else 0,
        'def_lr':    d_mean/(d_mean*1.3) if d_mean>0 else 0,  # always 76.9% — use payout density instead
        'exc_lr':    e_mean/(e_mean*1.3) if e_mean>0 else 0,
        'def_max_usd': usd_def[mask].sum(axis=0).max(),
        'exc_max_usd': usd_exc[mask].sum(axis=0).max(),
    })

cs = pd.DataFrame(cell_stats)
# Merge onto grid
grid = grid.merge(cs, on='grid_id', how='left')
grid[['def_mean','def_std','def_cv','exc_mean','exc_std','exc_cv',
      'def_max_usd','exc_max_usd']] = grid[
    ['def_mean','def_std','def_cv','exc_mean','exc_std','exc_cv',
     'def_max_usd','exc_max_usd']].fillna(0)

# ── Portfolio-level year-by-year for the bottom panel ────────────────────────
portfolio_def = usd_def.sum(axis=0)
portfolio_exc = usd_exc.sum(axis=0)
# At 1.3x premium on mean
prem_def = portfolio_def.mean() * 1.3
prem_exc = portfolio_exc.mean() * 1.3
lr_def_annual = portfolio_def / prem_def
lr_exc_annual = portfolio_exc / prem_exc

# ── ENSO-phase portfolio stats ────────────────────────────────────────────────
enso_phases = [enso_phase(yr) for yr in YEARS]

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 1 — Geographic maps: mean annual payout density ($/ha insured)
# ══════════════════════════════════════════════════════════════════════════════
print("Drawing Figure 1 — payout density maps …")

# Payout per insured ha (intensity metric that's comparable across cell sizes)
grid['def_usd_per_ha'] = np.where(grid['uninsured_ha']>0,
                                   grid['def_mean']/grid['uninsured_ha'], 0)
grid['exc_usd_per_ha'] = np.where(grid['uninsured_ha']>0,
                                   grid['exc_mean']/grid['uninsured_ha'], 0)

ZONE_LABEL_POSITIONS = {
    'A': (-70.5, 5.0, 'Llanos\nOrientales'),
    'B': (-76.2, 4.0, 'Cauca\nValley'),
    'C': (-74.5, 9.5, 'Atlantic\nCoast'),
    'D': (-74.8, 2.5, 'Andean\nFoothills'),
    'E': (-74.0, 7.0, 'Magdalena\nValley'),
}

SUYANA_BLUE  = '#1B3A5C'
SUYANA_TEAL  = '#1D7A8C'
DROUGHT_CMAP = 'YlOrRd'
FLOOD_CMAP   = 'Blues'
GREY_BG      = '#F0F0F0'

fig = plt.figure(figsize=(18, 22), facecolor='white')
fig.patch.set_facecolor('white')

# Title
fig.text(0.5, 0.97, 'Colombia Parametric Rainfall Insurance',
         ha='center', va='top', fontsize=22, fontweight='bold', color=SUYANA_BLUE)
fig.text(0.5, 0.945, 'Mean Annual Payout Intensity by Peril  •  Synthetic Rainfall 1994–2023  •  5,600 plots',
         ha='center', va='top', fontsize=13, color='#444444')

gs = GridSpec(3, 2, figure=fig, left=0.05, right=0.95,
              top=0.93, bottom=0.32, hspace=0.25, wspace=0.08)

ax_def = fig.add_subplot(gs[0:2, 0])
ax_exc = fig.add_subplot(gs[0:2, 1])

non_crop = grid[grid['def_mean']==0]

for ax, col, cmap, title, unit in [
    (ax_def, 'def_usd_per_ha', DROUGHT_CMAP, 'DROUGHT  (Deficit Rainfall)', '$/ha insured/yr'),
    (ax_exc, 'exc_usd_per_ha', FLOOD_CMAP,   'FLOOD  (Excess Rainfall)',     '$/ha insured/yr'),
]:
    ax.set_facecolor('#D6EAF8')  # ocean
    colombia_gdf.plot(ax=ax, color=GREY_BG, edgecolor='none', zorder=0)

    # Non-agricultural background
    non_crop.plot(ax=ax, color='#E8E8E8', edgecolor='none', zorder=1, linewidth=0)

    crop_cells = grid[grid[col]>0]
    if len(crop_cells):
        vmax = np.percentile(crop_cells[col], 95)
        norm = mcolors.Normalize(vmin=0, vmax=vmax)
        crop_cells.plot(ax=ax, column=col, cmap=cmap, norm=norm,
                        edgecolor='none', zorder=2)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, shrink=0.7, pad=0.02, aspect=25)
        cbar.set_label(unit, fontsize=10)
        cbar.ax.tick_params(labelsize=9)

    colombia_gdf.boundary.plot(ax=ax, color=SUYANA_BLUE, linewidth=1.5, zorder=4)

    # Zone boundary boxes
    for zcode, z in ZONES.items():
        rect = plt.Rectangle((z['lon_min'], z['lat_min']),
                               z['lon_max']-z['lon_min'], z['lat_max']-z['lat_min'],
                               linewidth=1.2, edgecolor='#555555',
                               facecolor='none', linestyle='--', zorder=5)
        ax.add_patch(rect)

    # Zone labels
    for zcode, (lx, ly, lbl) in ZONE_LABEL_POSITIONS.items():
        ax.text(lx, ly, lbl, fontsize=7.5, color='#222222',
                ha='center', va='center', zorder=6,
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.6, edgecolor='none'))

    ax.set_xlim(-79.5, -66.5); ax.set_ylim(-0.5, 12.5)
    ax.set_title(title, fontsize=14, fontweight='bold', color=SUYANA_BLUE, pad=10)
    ax.set_xlabel('Longitude', fontsize=10); ax.set_ylabel('Latitude', fontsize=10)
    ax.tick_params(labelsize=9)

# ── Row 3: payout variance (CV) maps ─────────────────────────────────────────
ax_dcv = fig.add_subplot(gs[2, 0])
ax_ecv = fig.add_subplot(gs[2, 1])

for ax, col, cmap, title in [
    (ax_dcv, 'def_cv', 'Oranges', 'Drought — Coefficient of Variation'),
    (ax_ecv, 'exc_cv', 'PuBu',    'Flood — Coefficient of Variation'),
]:
    ax.set_facecolor('#D6EAF8')
    colombia_gdf.plot(ax=ax, color=GREY_BG, edgecolor='none', zorder=0)
    non_crop.plot(ax=ax, color='#E8E8E8', edgecolor='none', zorder=1)
    crop_cv = grid[grid[col]>0]
    if len(crop_cv):
        norm = mcolors.Normalize(vmin=0, vmax=min(2.5, crop_cv[col].quantile(0.95)))
        crop_cv.plot(ax=ax, column=col, cmap=cmap, norm=norm, edgecolor='none', zorder=2)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, shrink=0.7, pad=0.02, aspect=20)
        cbar.set_label('CV (std/mean)', fontsize=9)
        cbar.ax.tick_params(labelsize=8)
    colombia_gdf.boundary.plot(ax=ax, color=SUYANA_BLUE, linewidth=1.2, zorder=4)
    ax.set_xlim(-79.5,-66.5); ax.set_ylim(-0.5,12.5)
    ax.set_title(title, fontsize=11, fontweight='bold', color=SUYANA_BLUE, pad=6)
    ax.set_xlabel('Longitude', fontsize=9); ax.set_ylabel('Latitude', fontsize=9)
    ax.tick_params(labelsize=8)

# ══════════════════════════════════════════════════════════════════════════════
# BOTTOM PANEL — year-by-year loss ratios + ENSO phase, side by side
# ══════════════════════════════════════════════════════════════════════════════
ax_lr = fig.add_axes([0.07, 0.05, 0.86, 0.23])

ENSO_COLORS = {'El Niño':'#E74C3C','La Niña':'#2980B9','Neutral':'#BDC3C7'}
bar_w = 0.38
x = np.arange(N_YEARS)

bars_def = ax_lr.bar(x - bar_w/2, lr_def_annual*100, bar_w,
                     color='#E67E22', alpha=0.85, label='Drought LR', zorder=3)
bars_exc = ax_lr.bar(x + bar_w/2, lr_exc_annual*100, bar_w,
                     color='#2980B9', alpha=0.85, label='Flood LR', zorder=3)

# ENSO phase shading behind bars
for yidx, (yr, phase) in enumerate(zip(YEARS, enso_phases)):
    if phase != 'Neutral':
        ax_lr.axvspan(yidx-0.5, yidx+0.5,
                      color=ENSO_COLORS[phase], alpha=0.12, zorder=1)

ax_lr.axhline(76.9, color='#E67E22', linewidth=1.4, linestyle='--', alpha=0.7, zorder=4)
ax_lr.axhline(76.9, color='#2980B9', linewidth=1.4, linestyle=':',  alpha=0.7, zorder=4)
ax_lr.axhline(100,  color='#C0392B', linewidth=1.2, linestyle='-',  alpha=0.5, zorder=4,
              label='Break-even (100%)')

ax_lr.set_xticks(x)
ax_lr.set_xticklabels([str(yr) for yr in YEARS], rotation=45, fontsize=8, ha='right')
ax_lr.set_ylabel('Loss Ratio (%)', fontsize=11)
ax_lr.set_ylim(0, 170)
ax_lr.set_xlim(-0.6, N_YEARS-0.4)
ax_lr.set_title('Annual Loss Ratio by Peril  —  Drought vs Flood  |  ENSO phase shading: '
                '▪ red = El Niño  ▪ blue = La Niña',
                fontsize=11, fontweight='bold', color=SUYANA_BLUE)
ax_lr.yaxis.grid(True, alpha=0.3, zorder=0)
ax_lr.set_axisbelow(True)

# Legend
leg_handles = [
    mpatches.Patch(color='#E67E22', alpha=0.85, label=f'Drought  (mean LR {lr_def_annual.mean()*100:.0f}%)'),
    mpatches.Patch(color='#2980B9', alpha=0.85, label=f'Flood  (mean LR {lr_exc_annual.mean()*100:.0f}%)'),
    Line2D([0],[0], color='#C0392B', linewidth=1.5, linestyle='-', label='Break-even 100%'),
    mpatches.Patch(color='#E74C3C', alpha=0.25, label='El Niño year'),
    mpatches.Patch(color='#2980B9', alpha=0.25, label='La Niña year'),
]
ax_lr.legend(handles=leg_handles, loc='upper left', fontsize=9, ncol=5,
             framealpha=0.9, edgecolor='#CCCCCC')

# Annotate worst years
top_def = sorted(range(N_YEARS), key=lambda i: lr_def_annual[i], reverse=True)[:3]
top_exc = sorted(range(N_YEARS), key=lambda i: lr_exc_annual[i], reverse=True)[:3]
for i in set(top_def+top_exc):
    yr = YEARS[i]
    y_val = max(lr_def_annual[i], lr_exc_annual[i])*100
    ax_lr.annotate(str(yr), xy=(i, y_val+1), fontsize=7, ha='center', color='#333333', fontweight='bold')

# Footnote
fig.text(0.5, 0.01,
         'Synthetic data only — not real CHIRPS. Premium = mean annual payout × 1.3. '
         'Loss ratio = payout / premium. Suyana — June 2026.',
         ha='center', fontsize=8, color='#888888', style='italic')

path1 = f"{OUT}/fig1_drought_vs_flood_maps.png"
plt.savefig(path1, dpi=180, bbox_inches='tight', facecolor='white')
plt.close()
print(f"Saved: {path1}")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 2 — Scatter: per-zone mean LR vs LR volatility + combined portfolio
# ══════════════════════════════════════════════════════════════════════════════
print("Drawing Figure 2 — risk-return scatter …")

fig2, axes2 = plt.subplots(1, 3, figsize=(18, 7), facecolor='white')
fig2.patch.set_facecolor('white')
fig2.suptitle('Colombia Parametric Insurance — Peril Comparison\nRisk vs Return by Zone',
              fontsize=16, fontweight='bold', color=SUYANA_BLUE, y=1.02)

ZONE_COLORS_SCATTER = {'A':'#E74C3C','B':'#2ECC71','C':'#3498DB','D':'#9B59B6','E':'#F39C12'}
ZONE_MARKERS = {'A':'o','B':'s','C':'^','D':'D','E':'P'}

# Build zone-level stats
zone_stats_def, zone_stats_exc, zone_stats_both = [], [], []
for zcode in 'ABCDE':
    mask = plots_df['zone']==zcode
    if not mask.any(): continue
    d_ann = usd_def[mask].sum(axis=0)
    e_ann = usd_exc[mask].sum(axis=0)
    b_ann = d_ann + e_ann
    d_prem = d_ann.mean()*1.3; e_prem = e_ann.mean()*1.3; b_prem = b_ann.mean()*1.3

    zone_stats_def.append({'zone':zcode,'name':ZONE_NAMES[zcode],
        'mean_lr': d_ann.mean()/d_prem if d_prem>0 else 0,
        'std_lr':  d_ann.std()/d_prem  if d_prem>0 else 0,
        'max_lr':  d_ann.max()/d_prem  if d_prem>0 else 0,
        'mean_payout': d_ann.mean(), 'std_payout': d_ann.std()})
    zone_stats_exc.append({'zone':zcode,'name':ZONE_NAMES[zcode],
        'mean_lr': e_ann.mean()/e_prem if e_prem>0 else 0,
        'std_lr':  e_ann.std()/e_prem  if e_prem>0 else 0,
        'max_lr':  e_ann.max()/e_prem  if e_prem>0 else 0,
        'mean_payout': e_ann.mean(), 'std_payout': e_ann.std()})
    zone_stats_both.append({'zone':zcode,'name':ZONE_NAMES[zcode],
        'mean_lr': b_ann.mean()/b_prem if b_prem>0 else 0,
        'std_lr':  b_ann.std()/b_prem  if b_prem>0 else 0,
        'max_lr':  b_ann.max()/b_prem  if b_prem>0 else 0,
        'mean_payout': b_ann.mean(), 'std_payout': b_ann.std()})

for ax, stats_list, title, color in [
    (axes2[0], zone_stats_def,  'Drought Only',        '#E67E22'),
    (axes2[1], zone_stats_exc,  'Flood Only',          '#2980B9'),
    (axes2[2], zone_stats_both, 'Drought + Flood\n(combined)', '#27AE60'),
]:
    ax.set_facecolor('#FAFAFA')
    for zs in stats_list:
        z = zs['zone']
        ax.scatter(zs['std_lr']*100, zs['mean_lr']*100,
                   s=max(50, zs['mean_payout']/800000),
                   color=ZONE_COLORS_SCATTER[z],
                   marker=ZONE_MARKERS[z],
                   edgecolors='#333333', linewidths=0.8,
                   zorder=4, alpha=0.9)
        ax.annotate(f"  {ZONE_NAMES[z]}\n  max LR {zs['max_lr']*100:.0f}%",
                    (zs['std_lr']*100, zs['mean_lr']*100),
                    fontsize=8.5, color='#222222', va='center')

    # Portfolio point
    all_ann = np.array([s['mean_payout'] for s in stats_list])
    all_std = np.array([s['std_payout']  for s in stats_list])
    port_mean = sum(s['mean_payout'] for s in stats_list)
    port_std  = np.sqrt(sum(s['std_payout']**2 for s in stats_list))  # simplified
    port_prem = port_mean * 1.3
    if port_prem > 0:
        ax.scatter(port_std/port_prem*100, port_mean/port_prem*100,
                   s=300, color='black', marker='*', zorder=5, label='Portfolio')
        ax.annotate('  Portfolio', (port_std/port_prem*100, port_mean/port_prem*100),
                    fontsize=9, fontweight='bold', va='center')

    ax.axhline(76.9, color='grey', linestyle='--', linewidth=1, alpha=0.6, label='Mean LR 76.9%')
    ax.axhline(100,  color='#C0392B', linestyle='-', linewidth=1, alpha=0.5, label='Break-even 100%')
    ax.set_xlabel('Loss Ratio Volatility\n(std dev of annual LR, %)', fontsize=10)
    ax.set_ylabel('Mean Annual Loss Ratio (%)', fontsize=10)
    ax.set_title(title, fontsize=13, fontweight='bold', color=color, pad=8)
    ax.yaxis.grid(True, alpha=0.3); ax.xaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)

# Shared legend
legend_handles = [
    mpatches.Patch(color=ZONE_COLORS_SCATTER[z], label=f"Zone {z} — {ZONE_NAMES[z]}")
    for z in 'ABCDE'
] + [
    Line2D([0],[0], color='grey', linestyle='--', label='Mean LR 76.9%'),
    Line2D([0],[0], color='#C0392B', linestyle='-', label='Break-even 100%'),
    Line2D([0],[0], marker='*', color='black', linestyle='None', markersize=12, label='Portfolio total'),
]
fig2.legend(handles=legend_handles, loc='lower center', ncol=4, fontsize=9,
            framealpha=0.9, edgecolor='#CCCCCC', bbox_to_anchor=(0.5,-0.06))

fig2.text(0.5,-0.09,
          'Bubble size ∝ mean annual payout. Portfolio std simplified (sum of variances). '
          'Synthetic data only. Suyana — June 2026.',
          ha='center', fontsize=8, color='#888888', style='italic')

path2 = f"{OUT}/fig2_peril_risk_return_scatter.png"
plt.savefig(path2, dpi=180, bbox_inches='tight', facecolor='white')
plt.close()
print(f"Saved: {path2}")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 3 — ENSO decomposition: what drives bad years?
# ══════════════════════════════════════════════════════════════════════════════
print("Drawing Figure 3 — ENSO decomposition …")

fig3, axes3 = plt.subplots(2, 2, figsize=(16, 12), facecolor='white')
fig3.patch.set_facecolor('white')
fig3.suptitle('ENSO Phase Decomposition\nDrought vs Flood Loss Ratios 1994–2023',
              fontsize=16, fontweight='bold', color=SUYANA_BLUE)

# 3a — Stacked bar: annual payout by peril
ax3a = axes3[0,0]
x = np.arange(N_YEARS)
ax3a.bar(x, portfolio_def/1e6, color='#E67E22', alpha=0.85, label='Drought payout', zorder=3)
ax3a.bar(x, portfolio_exc/1e6, bottom=portfolio_def/1e6,
         color='#2980B9', alpha=0.85, label='Flood payout', zorder=3)
for yidx, phase in enumerate(enso_phases):
    if phase != 'Neutral':
        ax3a.axvspan(yidx-0.5,yidx+0.5, color=ENSO_COLORS[phase], alpha=0.12, zorder=1)
ax3a.axhline((portfolio_def+portfolio_exc).mean()/1e6, color='black',
             linestyle='--', linewidth=1.2, alpha=0.7, label='Mean total')
ax3a.set_xticks(x[::2]); ax3a.set_xticklabels([str(YEARS[i]) for i in range(0,N_YEARS,2)],
                                                rotation=45, fontsize=8)
ax3a.set_ylabel('Annual Payout ($M)', fontsize=11)
ax3a.set_title('Total Payout by Peril', fontsize=12, fontweight='bold', color=SUYANA_BLUE)
ax3a.legend(fontsize=9); ax3a.yaxis.grid(True, alpha=0.3); ax3a.set_axisbelow(True)

# 3b — Box plots by ENSO phase × peril
ax3b = axes3[0,1]
data_boxes = []
labels_b = []
colors_b = []
for phase in ['El Niño','Neutral','La Niña']:
    idx = [i for i,p in enumerate(enso_phases) if p==phase]
    data_boxes.append(lr_def_annual[idx]*100)
    data_boxes.append(lr_exc_annual[idx]*100)
    labels_b += [f'{phase}\nDrought', f'{phase}\nFlood']
    colors_b  += ['#E67E22','#2980B9']

bp = ax3b.boxplot(data_boxes, patch_artist=True, medianprops=dict(color='black',linewidth=2))
for patch, color in zip(bp['boxes'], colors_b):
    patch.set_facecolor(color); patch.set_alpha(0.7)
ax3b.set_xticklabels(labels_b, fontsize=8)
ax3b.axhline(100, color='#C0392B', linestyle='-', linewidth=1.2, alpha=0.6)
ax3b.axhline(76.9, color='grey', linestyle='--', linewidth=1, alpha=0.5)
ax3b.set_ylabel('Loss Ratio (%)', fontsize=11)
ax3b.set_title('Loss Ratio Distribution by ENSO Phase', fontsize=12, fontweight='bold', color=SUYANA_BLUE)
ax3b.yaxis.grid(True, alpha=0.3); ax3b.set_axisbelow(True)

# 3c — Scatter: drought LR vs flood LR (each year a dot, coloured by ENSO)
ax3c = axes3[1,0]
for yidx, (yr, phase) in enumerate(zip(YEARS, enso_phases)):
    ax3c.scatter(lr_def_annual[yidx]*100, lr_exc_annual[yidx]*100,
                 color=ENSO_COLORS[phase], s=80, zorder=3,
                 edgecolors='#333333', linewidths=0.5)
    ax3c.annotate(str(yr)[2:], (lr_def_annual[yidx]*100, lr_exc_annual[yidx]*100),
                  fontsize=6.5, ha='left', color='#444444')
ax3c.axhline(100, color='#C0392B', linestyle='-', alpha=0.4)
ax3c.axvline(100, color='#C0392B', linestyle='-', alpha=0.4)
ax3c.axhline(76.9, color='grey', linestyle='--', alpha=0.35)
ax3c.axvline(76.9, color='grey', linestyle='--', alpha=0.35)
ax3c.set_xlabel('Drought Loss Ratio (%)', fontsize=11)
ax3c.set_ylabel('Flood Loss Ratio (%)', fontsize=11)
ax3c.set_title('Drought vs Flood LR Correlation\n(each dot = 1 year)', fontsize=12, fontweight='bold', color=SUYANA_BLUE)
corr = np.corrcoef(lr_def_annual, lr_exc_annual)[0,1]
ax3c.text(0.05,0.93, f'Pearson r = {corr:.2f}', transform=ax3c.transAxes,
          fontsize=10, color=SUYANA_BLUE, fontweight='bold',
          bbox=dict(boxstyle='round', facecolor='white', edgecolor='#CCCCCC'))
# Quadrant labels
xr = ax3c.get_xlim(); yr_ = ax3c.get_ylim()
ax3c.yaxis.grid(True,alpha=0.3); ax3c.xaxis.grid(True,alpha=0.3); ax3c.set_axisbelow(True)
for ph, col in ENSO_COLORS.items():
    ax3c.scatter([],[],color=col,s=60,label=ph,edgecolors='#333333',linewidths=0.5)
ax3c.legend(fontsize=9, loc='lower right')

# 3d — Premium adequacy by ENSO phase (bar chart)
ax3d = axes3[1,1]
phases_ordered = ['El Niño','Neutral','La Niña']
def_by_phase = {p: np.array([portfolio_def[i] for i,ph in enumerate(enso_phases) if ph==p]).mean()
                for p in phases_ordered}
exc_by_phase  = {p: np.array([portfolio_exc[i] for i,ph in enumerate(enso_phases) if ph==p]).mean()
                 for p in phases_ordered}

bw = 0.35
xp = np.arange(3)
b1 = ax3d.bar(xp-bw/2, [def_by_phase[p]/1e6 for p in phases_ordered], bw,
              color='#E67E22', alpha=0.85, label='Drought')
b2 = ax3d.bar(xp+bw/2, [exc_by_phase[p]/1e6  for p in phases_ordered], bw,
              color='#2980B9', alpha=0.85, label='Flood')
ax3d.axhline(portfolio_def.mean()/1e6, color='#E67E22', linestyle='--', linewidth=1.5, alpha=0.8)
ax3d.axhline(portfolio_exc.mean()/1e6, color='#2980B9', linestyle=':',  linewidth=1.5, alpha=0.8)
ax3d.set_xticks(xp); ax3d.set_xticklabels(phases_ordered, fontsize=11)
ax3d.set_ylabel('Mean Annual Payout ($M)', fontsize=11)
ax3d.set_title('Mean Payout by ENSO Phase\n(dashed = 30-yr average)',
               fontsize=12, fontweight='bold', color=SUYANA_BLUE)
ax3d.legend(fontsize=10); ax3d.yaxis.grid(True, alpha=0.3); ax3d.set_axisbelow(True)

for bars in [b1, b2]:
    for bar in bars:
        h = bar.get_height()
        ax3d.text(bar.get_x()+bar.get_width()/2, h+0.3, f'${h:.1f}M',
                  ha='center', va='bottom', fontsize=9, fontweight='bold')

fig3.text(0.5, -0.02,
          'Synthetic data only — not real CHIRPS. Suyana — June 2026.',
          ha='center', fontsize=8, color='#888888', style='italic')
plt.tight_layout()
path3 = f"{OUT}/fig3_enso_decomposition.png"
plt.savefig(path3, dpi=180, bbox_inches='tight', facecolor='white')
plt.close()
print(f"Saved: {path3}")

print("\nAll figures saved.")
print(f"  Drought mean LR: {lr_def_annual.mean()*100:.1f}%  |  std: {lr_def_annual.std()*100:.1f}pp")
print(f"  Flood  mean LR: {lr_exc_annual.mean()*100:.1f}%  |  std: {lr_exc_annual.std()*100:.1f}pp")
print(f"  Drought-Flood LR correlation: {np.corrcoef(lr_def_annual,lr_exc_annual)[0,1]:.2f}")
