"""
fig7_mvp_table.py
Multi-Objective Portfolio Optimisation table for Colombia Parametric Insurance
"""

import numpy as np
import cvxpy as cp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import os

# ── 1. Synthetic Data Generation ──────────────────────────────────────────────
np.random.seed(42)

PERILS = ["Drought", "Flood", "Heatwave", "Cold Spell", "Hail", "Sargassum"]
N_YEARS = 30
N_PERILS = 6
LOSS_RATIO = 0.769  # premium = payout / 0.769

# Target stats (in $M)
target_means = np.array([239.6, 224.3, 210.0, 219.5, 334.4, 4.6])
target_stds  = np.array([158.5, 139.3, 269.0, 199.0, 282.8, 47.3])

# Correlation matrix
# Drought/Flood +0.3, Heatwave/Cold Spell -0.2, others ~0
corr = np.eye(N_PERILS)
corr[0, 1] = corr[1, 0] = 0.30   # Drought-Flood
corr[2, 3] = corr[3, 2] = -0.20  # Heatwave-ColdSpell
corr[0, 2] = corr[2, 0] = 0.10
corr[1, 3] = corr[3, 1] = 0.10
corr[4, 0] = corr[0, 4] = 0.05
corr[4, 1] = corr[1, 4] = 0.05
corr[5, 2] = corr[2, 5] = -0.05

# Build covariance from correlation and stds
cov = np.outer(target_stds, target_stds) * corr

# Generate multivariate normal (mean=0, then shift/scale)
Z = np.random.multivariate_normal(np.zeros(N_PERILS), corr, size=N_YEARS)
# Scale to match target stds and shift to match target means
# Use lognormal-like transformation to keep values positive
# Simple approach: shift Z by target_means + target_stds * Z
raw = target_means + target_stds * Z

# Clip negative payouts to 0 (insurance can't be negative)
payouts = np.maximum(raw, 0.0)  # shape (30, 6)

# Recompute actual means/stds from generated data
actual_means = payouts.mean(axis=0)
actual_stds  = payouts.std(axis=0, ddof=1)
actual_cov   = np.cov(payouts.T)  # 6×6

# ── 2. Premium calculation ────────────────────────────────────────────────────
premiums = actual_means / LOSS_RATIO  # per-peril premium ($M) at weight=1

# ── 3. Portfolio definitions ──────────────────────────────────────────────────

def portfolio_metrics(weights, payouts, premiums):
    """Compute metrics for a portfolio given weights vector (length 6)."""
    w = np.array(weights)
    # Annual portfolio payout
    annual_loss = payouts @ w           # shape (30,)
    # Portfolio premium
    total_premium = premiums @ w
    expected_loss = actual_means @ w
    loss_ratio_series = annual_loss / total_premium * 100  # %
    mean_lr = loss_ratio_series.mean()
    std_loss = annual_loss.std(ddof=1)
    cv = std_loss / expected_loss if expected_loss > 0 else np.nan
    worst_year_lr = loss_ratio_series.max()
    years_over_100 = (loss_ratio_series > 100).sum()
    return {
        "total_premium": total_premium,
        "expected_loss": expected_loss,
        "mean_lr": mean_lr,
        "std_loss": std_loss,
        "cv": cv,
        "worst_year_lr": worst_year_lr,
        "years_over_100": years_over_100,
    }

def peril_metrics(weights, payouts, premiums):
    """Compute per-peril metrics for a portfolio."""
    w = np.array(weights)
    total_premium = premiums @ w
    total_variance = w @ actual_cov @ w
    results = []
    for i, peril in enumerate(PERILS):
        book_pct = w[i] * 100
        prem_i = premiums[i] * w[i]
        prem_share = prem_i / total_premium * 100 if total_premium > 0 else 0
        exp_loss_i = actual_means[i] * w[i]
        # Variance contribution = w_i * (Σw)_i / total_variance
        sigma_w = actual_cov @ w
        var_contrib = (w[i] * sigma_w[i] / total_variance * 100) if total_variance > 0 else 0
        results.append({
            "book_pct": book_pct,
            "premium": prem_i,
            "prem_share": prem_share,
            "exp_loss": exp_loss_i,
            "var_contrib": var_contrib,
        })
    return results

# ── Full Book ──────────────────────────────────────────────────────────────────
w_full = np.ones(N_PERILS)
metrics_full = portfolio_metrics(w_full, payouts, premiums)
peril_full   = peril_metrics(w_full, payouts, premiums)

# ── Efficient Frontier: pick max premium/std portfolio ─────────────────────────
# Sweep target returns, compute min-variance portfolio for each
n_points = 200
w_var = cp.Variable(N_PERILS)
param_ret = cp.Parameter()
constraints_ef = [w_var >= 0, w_var <= 1]
objective_ef = cp.Minimize(cp.quad_form(w_var, actual_cov))
prob_ef = cp.Problem(objective_ef, constraints_ef + [actual_means @ w_var >= param_ret])

min_ret = actual_means @ np.zeros(N_PERILS)
max_ret = actual_means @ w_full

best_ratio = -np.inf
w_optimal = None

for ret_target in np.linspace(actual_means.min() * 0.5, max_ret, n_points):
    param_ret.value = ret_target
    try:
        prob_ef.solve(solver=cp.CLARABEL, warm_start=True)
        if w_var.value is not None:
            w_try = np.clip(w_var.value, 0, 1)
            prem = premiums @ w_try
            std  = np.sqrt(w_try @ actual_cov @ w_try)
            ratio = prem / std if std > 0 else 0
            if ratio > best_ratio:
                best_ratio = ratio
                w_optimal = w_try.copy()
    except Exception:
        pass

if w_optimal is None:
    w_optimal = w_full.copy()

metrics_opt  = portfolio_metrics(w_optimal, payouts, premiums)
peril_opt    = peril_metrics(w_optimal, payouts, premiums)

# ── Min Variance with premium floor ───────────────────────────────────────────
floor_premium = 0.20 * (premiums @ w_full)
w_mv = cp.Variable(N_PERILS)
objective_mv = cp.Minimize(cp.quad_form(w_mv, actual_cov))
constraints_mv = [
    w_mv >= 0,
    w_mv <= 1,
    premiums @ w_mv >= floor_premium,
]
prob_mv = cp.Problem(objective_mv, constraints_mv)
prob_mv.solve(solver=cp.CLARABEL)

if w_mv.value is not None:
    w_minvar = np.clip(w_mv.value, 0, 1)
else:
    w_minvar = w_full.copy()

metrics_mv   = portfolio_metrics(w_minvar, payouts, premiums)
peril_mv     = peril_metrics(w_minvar, payouts, premiums)

# ── 4. Build Table Data ───────────────────────────────────────────────────────

port_rows = [
    ("Total Premium $M",         f"{metrics_full['total_premium']:.1f}",   f"{metrics_opt['total_premium']:.1f}",   f"{metrics_mv['total_premium']:.1f}"),
    ("Expected Annual Loss $M",  f"{metrics_full['expected_loss']:.1f}",   f"{metrics_opt['expected_loss']:.1f}",   f"{metrics_mv['expected_loss']:.1f}"),
    ("Mean Loss Ratio %",        f"{metrics_full['mean_lr']:.1f}%",        f"{metrics_opt['mean_lr']:.1f}%",        f"{metrics_mv['mean_lr']:.1f}%"),
    ("Std Dev Annual Loss $M",   f"{metrics_full['std_loss']:.1f}",        f"{metrics_opt['std_loss']:.1f}",        f"{metrics_mv['std_loss']:.1f}"),
    ("CV (Std/Mean Loss)",       f"{metrics_full['cv']:.3f}",              f"{metrics_opt['cv']:.3f}",              f"{metrics_mv['cv']:.3f}"),
    ("Worst Year Loss Ratio %",  f"{metrics_full['worst_year_lr']:.1f}%",  f"{metrics_opt['worst_year_lr']:.1f}%",  f"{metrics_mv['worst_year_lr']:.1f}%"),
    ("Years LR > 100%",          f"{metrics_full['years_over_100']}",      f"{metrics_opt['years_over_100']}",      f"{metrics_mv['years_over_100']}"),
]

# Peril rows: each peril has 5 sub-metrics
peril_section = []
for i, peril in enumerate(PERILS):
    pf = peril_full[i]
    po = peril_opt[i]
    pm = peril_mv[i]
    peril_section.append((
        peril,
        f"{pf['book_pct']:.0f}%",        f"{po['book_pct']:.0f}%",        f"{pm['book_pct']:.0f}%",
        f"{pf['premium']:.1f}",           f"{po['premium']:.1f}",           f"{pm['premium']:.1f}",
        f"{pf['prem_share']:.1f}%",       f"{po['prem_share']:.1f}%",       f"{pm['prem_share']:.1f}%",
        f"{pf['exp_loss']:.1f}",          f"{po['exp_loss']:.1f}",           f"{pm['exp_loss']:.1f}",
        f"{pf['var_contrib']:.1f}%",      f"{po['var_contrib']:.1f}%",      f"{pm['var_contrib']:.1f}%",
    ))

# ── 5. Draw the Table ─────────────────────────────────────────────────────────
fig = plt.figure(figsize=(16, 12), dpi=150, facecolor="white")
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")

# Colors
HDR_BG    = "#1a2744"
HDR_FG    = "white"
SEC_BG    = "#2c3e6b"
SEC_FG    = "white"
ALT1_BG   = "#f0f3fa"
ALT2_BG   = "white"
PERIL_BG  = "#e8ecf5"
SUB_BG1   = "#f7f9fd"
SUB_BG2   = "white"
TEXT_COL  = "#1a1a2e"
NUM_COL   = "#1a2744"
BORDER    = "#c5cde8"

# Layout
LEFT   = 0.02
RIGHT  = 0.98
TOP    = 0.96
BOTTOM = 0.03

total_w = RIGHT - LEFT
col_widths = [0.24, 0.245, 0.245, 0.245]  # label + 3 portfolios
col_starts = [LEFT]
for cw in col_widths[:-1]:
    col_starts.append(col_starts[-1] + cw * total_w / sum(col_widths))
# Recalculate proportionally
cum = 0
col_starts = [LEFT]
for cw in col_widths[:-1]:
    cum += cw / sum(col_widths) * (RIGHT - LEFT)
    col_starts.append(LEFT + cum)
col_widths_abs = [cw / sum(col_widths) * (RIGHT - LEFT) for cw in col_widths]

def draw_cell(ax, x, y, w, h, text, bg, fg, fontsize=9, bold=False, align="center", valign="center", pad=0.005):
    rect = plt.Rectangle((x, y), w, h, facecolor=bg, edgecolor="none", transform=ax.transAxes, zorder=1)
    ax.add_patch(rect)
    tx = x + w/2 if align == "center" else x + pad
    ty = y + h/2
    ha = align
    ax.text(tx, ty, text, transform=ax.transAxes,
            ha=ha, va=valign,
            fontsize=fontsize, color=fg,
            fontweight="bold" if bold else "normal",
            zorder=2, clip_on=False)

def draw_hline(ax, y, x0, x1, color=BORDER, lw=0.5):
    ax.plot([x0, x1], [y, y], color=color, lw=lw, transform=ax.transAxes, zorder=3)

# Title
ax.text(0.5, 0.975, "Multi-Objective Portfolio Optimisation: Colombia Parametric Insurance",
        transform=ax.transAxes, ha="center", va="top",
        fontsize=14, fontweight="bold", color=HDR_BG)
ax.text(0.5, 0.957, "Synthetic data, 30-year simulation, 6 perils",
        transform=ax.transAxes, ha="center", va="top",
        fontsize=10, color="#555577", style="italic")

# Available vertical space
y_start = 0.935
row_h = 0.033

# ── Header row ──
y = y_start - row_h
headers = ["", "Full Book", "Optimal Portfolio", "Min Variance Portfolio"]
for j, (hdr, xs, cw) in enumerate(zip(headers, col_starts, col_widths_abs)):
    draw_cell(ax, xs, y, cw, row_h, hdr, HDR_BG, HDR_FG, fontsize=9.5, bold=True)
draw_hline(ax, y + row_h, LEFT, RIGHT, color=HDR_BG, lw=1.5)
draw_hline(ax, y,         LEFT, RIGHT, color=HDR_BG, lw=1.5)

y_cur = y

# ── Portfolio Summary Section header ──
y_cur -= 0.005
y = y_cur - row_h * 0.7
draw_cell(ax, LEFT, y, RIGHT - LEFT, row_h * 0.7, "PORTFOLIO SUMMARY", SEC_BG, SEC_FG, fontsize=8.5, bold=True)
draw_hline(ax, y + row_h * 0.7, LEFT, RIGHT, color=SEC_BG, lw=0.5)
y_cur = y

# ── Portfolio summary rows ──
for k, (label, v_full, v_opt, v_mv) in enumerate(port_rows):
    bg = ALT1_BG if k % 2 == 0 else ALT2_BG
    y = y_cur - row_h
    values = [label, v_full, v_opt, v_mv]
    for j, (val, xs, cw) in enumerate(zip(values, col_starts, col_widths_abs)):
        align = "left" if j == 0 else "center"
        bold_val = (j == 0)
        draw_cell(ax, xs, y, cw, row_h, val, bg, TEXT_COL if j == 0 else NUM_COL,
                  fontsize=9, bold=bold_val, align=align)
    draw_hline(ax, y, LEFT, RIGHT, color=BORDER, lw=0.4)
    y_cur = y

# ── Peril Breakdown Section header ──
y_cur -= 0.005
y = y_cur - row_h * 0.7
draw_cell(ax, LEFT, y, RIGHT - LEFT, row_h * 0.7, "PERIL BREAKDOWN", SEC_BG, SEC_FG, fontsize=8.5, bold=True)
draw_hline(ax, y + row_h * 0.7, LEFT, RIGHT, color=SEC_BG, lw=0.5)
y_cur = y

# Sub-header for peril section
y = y_cur - row_h * 0.75
sub_labels = ["Metric", "Full Book", "Optimal", "Min Var"]
for j, (sl, xs, cw) in enumerate(zip(sub_labels, col_starts, col_widths_abs)):
    draw_cell(ax, xs, y, cw, row_h * 0.75, sl, "#3a4f82", HDR_FG, fontsize=8, bold=True)
draw_hline(ax, y, LEFT, RIGHT, color="#3a4f82", lw=0.5)
y_cur = y

peril_sub_labels = ["Book %", "Premium $M", "Premium Share %", "Expected Loss $M", "Variance Contrib %"]

for i, peril_data in enumerate(peril_section):
    peril_name = peril_data[0]
    # Peril name row
    y = y_cur - row_h * 0.75
    draw_cell(ax, LEFT, y, RIGHT - LEFT, row_h * 0.75, peril_name, PERIL_BG, "#1a2744",
              fontsize=9, bold=True, align="left")
    draw_hline(ax, y + row_h * 0.75, LEFT, RIGHT, color="#8899cc", lw=0.6)
    draw_hline(ax, y, LEFT, RIGHT, color=BORDER, lw=0.3)
    y_cur = y

    # 5 sub-metric rows per peril
    # peril_data layout: (name, bk_f, bk_o, bk_m, pr_f, pr_o, pr_m, ps_f, ps_o, ps_m, el_f, el_o, el_m, vc_f, vc_o, vc_m)
    offsets = [1, 4, 7, 10, 13]  # start indices in peril_data for each sub-metric
    for si, (sub_lbl, off) in enumerate(zip(peril_sub_labels, offsets)):
        bg = SUB_BG1 if si % 2 == 0 else SUB_BG2
        y = y_cur - row_h * 0.72
        row_vals = [f"  {sub_lbl}", peril_data[off], peril_data[off+1], peril_data[off+2]]
        for j, (val, xs, cw) in enumerate(zip(row_vals, col_starts, col_widths_abs)):
            align = "left" if j == 0 else "center"
            draw_cell(ax, xs, y, cw, row_h * 0.72, val, bg, TEXT_COL if j == 0 else NUM_COL,
                      fontsize=8.2, bold=False, align=align)
        draw_hline(ax, y, LEFT, RIGHT, color=BORDER, lw=0.3)
        y_cur = y

# Outer border
rect = plt.Rectangle((LEFT, y_cur), RIGHT - LEFT, y_start - y_cur,
                      facecolor="none", edgecolor=HDR_BG, lw=1.5,
                      transform=ax.transAxes, zorder=5)
ax.add_patch(rect)

# Vertical dividers
for xs in col_starts[1:]:
    ax.plot([xs, xs], [y_cur, y_start], color=BORDER, lw=0.5, transform=ax.transAxes, zorder=3)

# Footer note
ax.text(0.5, 0.01, f"Loss ratio floor: {LOSS_RATIO*100:.1f}%  |  Premium floor for Min-Var: 20% of Full Book  |  Optimisation: CVXPY / CLARABEL",
        transform=ax.transAxes, ha="center", va="bottom",
        fontsize=7.5, color="#777799", style="italic")

# Save
out_path = "/home/user/insurance-premium-flow/outputs/maps/fig7_mvp_table.png"
os.makedirs(os.path.dirname(out_path), exist_ok=True)
plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
plt.close()
print(f"Saved to {out_path}")

# Print summary
print("\n=== PORTFOLIO SUMMARY ===")
for label, vf, vo, vm in port_rows:
    print(f"  {label:35s}  Full={vf:>10}  Optimal={vo:>10}  MinVar={vm:>10}")

print(f"\nOptimal weights: {np.round(w_optimal, 3)}")
print(f"MinVar weights:  {np.round(w_minvar, 3)}")
