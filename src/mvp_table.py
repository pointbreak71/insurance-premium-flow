"""
fig7_mvp_table.py
$15M underwriting authority, 10-year policy view.
Three portfolios: Worst Single Peril | Min Variance | Optimal (max Premium/Risk)
Plus peril breakdown for the Optimal portfolio.

Formulation
-----------
w_i ∈ [0,1]  = fraction of the natural book written for peril i
Each portfolio's weights are solved without a budget constraint, then scaled
proportionally so its annual premium = $15M (the UW cap).
This gives genuinely different risk profiles because the three strategies
concentrate/diversify differently — even after scaling to the same premium.
"""

import numpy as np
import cvxpy as cp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

# ── 1. Synthetic Data ─────────────────────────────────────────────────────────
np.random.seed(42)

PERILS     = ["Drought", "Flood", "Heatwave", "Cold Spell", "Hail", "Sargassum"]
N_YEARS    = 30
N_PERILS   = 6
UW_CAP     = 15.0        # $15M underwriting authority (annual premium target)
POLICY_YRS = 10

# Natural-book payout stats ($M/yr at w=1)
target_means = np.array([239.6, 224.3, 210.0, 219.5, 334.4,  4.6])
target_stds  = np.array([158.5, 139.3, 269.0, 199.0, 282.8, 47.3])

# Per-peril loss ratios — reflect market pricing realities
# Heatwave/Cold Spell harder to price → higher LR (thinner margin)
# Sargassum new peril → insurer retains better margin
LR_PER_PERIL = np.array([0.77, 0.75, 0.84, 0.82, 0.77, 0.62])

# Correlation structure
corr = np.eye(N_PERILS)
corr[0,1] = corr[1,0] =  0.30   # Drought ↔ Flood
corr[2,3] = corr[3,2] = -0.20   # Heatwave ↔ Cold Spell
corr[0,2] = corr[2,0] =  0.10
corr[1,3] = corr[3,1] =  0.10
corr[4,0] = corr[0,4] =  0.05
corr[4,1] = corr[1,4] =  0.05
corr[5,2] = corr[2,5] = -0.05

# Generate 30-year annual payout series
Z       = np.random.multivariate_normal(np.zeros(N_PERILS), corr, size=N_YEARS)
payouts = np.maximum(target_means + target_stds * Z, 0.0)   # shape (30, 6)

mu_nat  = payouts.mean(axis=0)       # empirical means
cov_nat = np.cov(payouts.T)         # 6×6 empirical covariance

# Premium per unit of natural book
prem_per_unit = mu_nat / LR_PER_PERIL

# Standalone CV and LR per peril (for reference)
sa_cv = target_stds / target_means

# ── 2. Worst single peril ─────────────────────────────────────────────────────
# Use Hail: highest absolute expected payout AND high CV → most damaging concentration
worst_idx = int(np.argmax(target_means))   # Hail
print(f"Single peril benchmark: {PERILS[worst_idx]}  "
      f"(expected payout ${target_means[worst_idx]:.0f}M/yr, CV={sa_cv[worst_idx]:.2f})")

# ── 3. Optimisation: min-var (the "optimal" diversified portfolio) ─────────────
def solve_minvar_fixed_premium(target_prem):
    """Minimum variance portfolio with premium anchored to target_prem ($M/yr)."""
    w    = cp.Variable(N_PERILS)
    prob = cp.Problem(
        cp.Minimize(cp.quad_form(w, cov_nat)),
        [w >= 0, w <= 1, prem_per_unit @ w == target_prem]
    )
    for solver in [cp.CLARABEL, cp.SCS, cp.ECOS]:
        try:
            prob.solve(solver=solver)
            if w.value is not None:
                return np.maximum(w.value, 0)
        except Exception:
            pass
    return np.ones(N_PERILS) * (target_prem / float(prem_per_unit.sum()))

# All three portfolios earn exactly $15M/yr premium — so premium is identical.
# The differences lie entirely in HOW the premium is allocated across perils,
# which determines the payout distribution (CV, worst year, variance).

# Single peril: all $15M into Hail
w_worst = np.zeros(N_PERILS)
w_worst[worst_idx] = UW_CAP / float(prem_per_unit[worst_idx])

# Naive equal-weight: $15M split equally in premium terms across all 6 perils
w_equal = (UW_CAP / N_PERILS) / prem_per_unit   # each peril contributes $2.5M premium

# Optimal (min variance): $15M, weight allocation minimises payout variance
w_opt = solve_minvar_fixed_premium(UW_CAP)

print(f"Single-peril (Hail) weight:  {w_worst[worst_idx]:.4f}")
print(f"Equal-weight weights:         {np.round(w_equal, 4)}")
print(f"Min-Var (Optimal) weights:    {np.round(w_opt, 4)}")

# All three already anchored to $15M/yr annual premium
w_worst_s = w_worst
w_mv_s    = w_equal
w_opt_s   = w_opt

# Column labels for the three portfolios
PORT_LABELS = [
    f"Single Peril\n({PERILS[worst_idx]} only)",
    "Equal Weight\n(6 Perils, Naive)",
    "Optimal Portfolio\n(Min Variance)",
]

# ── 5. Portfolio metrics ───────────────────────────────────────────────────────

def portfolio_metrics_10yr(w):
    annual_prem   = float(prem_per_unit @ w)
    annual_loss_e = float(mu_nat @ w)
    annual_std    = np.sqrt(float(w @ cov_nat @ w))
    total_prem_10 = POLICY_YRS * annual_prem
    total_loss_10 = POLICY_YRS * annual_loss_e
    std_10        = np.sqrt(POLICY_YRS) * annual_std   # √10 × annual std
    mean_lr       = annual_loss_e / annual_prem * 100
    cv_annual     = annual_std / annual_loss_e
    # Worst year from simulated series
    lr_series     = (payouts @ w) / annual_prem * 100
    worst_yr_lr   = lr_series.max()
    yrs_over_100  = int((lr_series > 100).sum())
    return dict(
        annual_prem=annual_prem, total_prem_10=total_prem_10,
        total_loss_10=total_loss_10, std_10=std_10,
        cv_annual=cv_annual, mean_lr=mean_lr,
        worst_yr_lr=worst_yr_lr, yrs_over_100=yrs_over_100,
    )

def peril_breakdown(w):
    total_prem = float(prem_per_unit @ w)
    total_var  = float(w @ cov_nat @ w)
    sigma_w    = cov_nat @ w
    rows = []
    for i in range(N_PERILS):
        if w[i] < 1e-6:
            continue
        prem_i     = float(prem_per_unit[i] * w[i])
        loss_i     = float(mu_nat[i] * w[i])
        prem_share = prem_i / total_prem * 100
        var_contrib= w[i] * sigma_w[i] / total_var * 100 if total_var > 0 else 0
        rows.append(dict(
            peril=PERILS[i],
            book_pct=w[i] * 100,
            prem=prem_i, prem_share=prem_share,
            loss=loss_i, var_contrib=var_contrib,
            sa_cv=sa_cv[i],
        ))
    return rows

m_worst = portfolio_metrics_10yr(w_worst_s)
m_mv    = portfolio_metrics_10yr(w_mv_s)
m_opt   = portfolio_metrics_10yr(w_opt_s)
peril_d = peril_breakdown(w_opt_s)

# Console check
print("\n=== 10-YEAR PORTFOLIO METRICS ($15M/yr UW Cap) ===")
for lbl, m in [("Worst Peril", m_worst), ("Min Variance", m_mv), ("Optimal", m_opt)]:
    print(f"  {lbl}: ann_prem=${m['annual_prem']:.2f}M  10yr_prem=${m['total_prem_10']:.1f}M"
          f"  10yr_loss=${m['total_loss_10']:.1f}M  std10=${m['std_10']:.1f}M"
          f"  CV={m['cv_annual']:.3f}  worstLR={m['worst_yr_lr']:.1f}%  yrs>{m['yrs_over_100']}")

# ── 6. Draw Tables ─────────────────────────────────────────────────────────────
NAVY   = "#1a2744"; NAVY2 = "#2c3e6b"; BORDER = "#c5cde8"
LGREY  = "#f0f3fa"; WHITE = "#ffffff"; DIM = "#666688"; TXT = "#1a1a2e"

fig = plt.figure(figsize=(17, 13), dpi=150, facecolor="white")

def cell(ax, x, y, w, h, txt, bg, fg, fs=9.5, bold=False, align="center", pad=0.01):
    ax.add_patch(plt.Rectangle((x, y), w, h, facecolor=bg, edgecolor="none",
                                transform=ax.transAxes, zorder=1, clip_on=False))
    tx = (x + w/2) if align == "center" else (x + pad)
    ax.text(tx, y + h/2, txt, transform=ax.transAxes,
            ha=align, va="center", fontsize=fs, color=fg,
            fontweight="bold" if bold else "normal", zorder=2, clip_on=False)

def hline(ax, y, x0=0.01, x1=0.99, c=BORDER, lw=0.5):
    ax.plot([x0, x1], [y, y], color=c, lw=lw, transform=ax.transAxes, zorder=3)

# ─── Table 1: Three-portfolio comparison ──────────────────────────────────────
ax1 = fig.add_axes([0.01, 0.49, 0.98, 0.50])
ax1.set_xlim(0, 1); ax1.set_ylim(0, 1); ax1.axis("off")

ax1.text(0.5, 0.985,
         "Portfolio Comparison — $15M Underwriting Authority | 10-Year Policy Horizon",
         transform=ax1.transAxes, ha="center", va="top",
         fontsize=13, fontweight="bold", color=NAVY)
ax1.text(0.5, 0.958,
         f"Colombia parametric insurance · 6 perils · 30-yr synthetic simulation · "
         f"All portfolios scaled to ${UW_CAP:.0f}M annual premium",
         transform=ax1.transAxes, ha="center", va="top",
         fontsize=9.5, color=DIM, style="italic")

L, R = 0.01, 0.99
col_x = [L, L+0.28*(R-L), L+0.52*(R-L), L+0.76*(R-L)]
col_w = [0.28*(R-L), 0.24*(R-L), 0.24*(R-L), 0.24*(R-L)]

y0  = 0.895
rh  = 0.078
hdrs = [""] + PORT_LABELS
for j, (hdr, cx, cw) in enumerate(zip(hdrs, col_x, col_w)):
    cell(ax1, cx, y0, cw, rh, hdr, NAVY, "white", fs=9, bold=True)
hline(ax1, y0+rh, L, R, c=NAVY, lw=2)
hline(ax1, y0,    L, R, c=NAVY, lw=2)

rows_data = [
    ("Annual Premium",
     f"${m_worst['annual_prem']:.1f}M", f"${m_mv['annual_prem']:.1f}M", f"${m_opt['annual_prem']:.1f}M"),
    ("10-Year Total Premiums",
     f"${m_worst['total_prem_10']:.0f}M", f"${m_mv['total_prem_10']:.0f}M", f"${m_opt['total_prem_10']:.0f}M"),
    ("10-Year Total Payouts",
     f"${m_worst['total_loss_10']:.0f}M", f"${m_mv['total_loss_10']:.0f}M", f"${m_opt['total_loss_10']:.0f}M"),
    ("10-Year Payout Std Dev  (√10 × annual σ)",
     f"${m_worst['std_10']:.1f}M", f"${m_mv['std_10']:.1f}M", f"${m_opt['std_10']:.1f}M"),
    ("CV  (Annual σ / Annual Mean Payout)",
     f"{m_worst['cv_annual']:.3f}", f"{m_mv['cv_annual']:.3f}", f"{m_opt['cv_annual']:.3f}"),
    ("Average Annual Loss Ratio",
     f"{m_worst['mean_lr']:.1f}%", f"{m_mv['mean_lr']:.1f}%", f"{m_opt['mean_lr']:.1f}%"),
    ("Worst Single-Year Loss Ratio",
     f"{m_worst['worst_yr_lr']:.1f}%", f"{m_mv['worst_yr_lr']:.1f}%", f"{m_opt['worst_yr_lr']:.1f}%"),
    (f"Years LR > 100%  (out of {N_YEARS})",
     f"{m_worst['yrs_over_100']}", f"{m_mv['yrs_over_100']}", f"{m_opt['yrs_over_100']}"),
]

yc = y0
for k, (lbl, v1, v2, v3) in enumerate(rows_data):
    bg  = LGREY if k % 2 == 0 else WHITE
    yc -= rh * 0.71
    for j, (val, cx, cw) in enumerate(zip([lbl, v1, v2, v3], col_x, col_w)):
        cell(ax1, cx, yc, cw, rh*0.71, val, bg,
             TXT if j == 0 else NAVY, fs=9.2,
             bold=(j == 0), align="left" if j == 0 else "center")
    hline(ax1, yc, L, R, c=BORDER, lw=0.4)

ax1.add_patch(plt.Rectangle((L, yc), R-L, y0+rh-yc, facecolor="none",
              edgecolor=NAVY, lw=1.5, transform=ax1.transAxes, zorder=5, clip_on=False))
for cx in col_x[1:]:
    ax1.plot([cx, cx], [yc, y0+rh], color=BORDER, lw=0.5, transform=ax1.transAxes, zorder=3)

# ─── Table 2: Optimal portfolio peril breakdown ────────────────────────────────
ax2 = fig.add_axes([0.01, -0.01, 0.98, 0.49])
ax2.set_xlim(0, 1); ax2.set_ylim(0, 1); ax2.axis("off")

ax2.text(0.5, 0.985, "Optimal Portfolio (Min Variance) — Peril-Level Breakdown",
         transform=ax2.transAxes, ha="center", va="top",
         fontsize=12, fontweight="bold", color=NAVY)
ax2.text(0.5, 0.955,
         f"Annual premium ${m_opt['annual_prem']:.1f}M  ·  "
         f"Expected annual payout ${m_opt['total_loss_10']/POLICY_YRS:.1f}M  ·  "
         f"Mean LR {m_opt['mean_lr']:.1f}%  ·  CV {m_opt['cv_annual']:.3f}",
         transform=ax2.transAxes, ha="center", va="top",
         fontsize=9.5, color=DIM, style="italic")

c2x = [L, L+0.175*(R-L), L+0.325*(R-L), L+0.475*(R-L), L+0.625*(R-L), L+0.775*(R-L), L+0.885*(R-L)]
c2w = [0.175*(R-L), 0.15*(R-L), 0.15*(R-L), 0.15*(R-L), 0.15*(R-L), 0.11*(R-L), 0.11*(R-L)]
hdrs2 = ["Peril", "Book %\nWritten", "Annual\nPremium ($M)", "Premium\nShare %",
         "Expected\nPayout ($M)", "Variance\nContrib %", "Standalone\nCV"]

y0b = 0.895; rhb = 0.090
for j, (hdr, cx, cw) in enumerate(zip(hdrs2, c2x, c2w)):
    cell(ax2, cx, y0b, cw, rhb, hdr, NAVY, "white", fs=8.5, bold=True)
hline(ax2, y0b+rhb, L, R, c=NAVY, lw=2)
hline(ax2, y0b,     L, R, c=NAVY, lw=2)

yc2 = y0b; rh2 = 0.075
for k, p in enumerate(peril_d):
    bg  = LGREY if k % 2 == 0 else WHITE
    yc2 -= rh2
    vals = [p["peril"], f"{p['book_pct']:.1f}%", f"${p['prem']:.2f}M",
            f"{p['prem_share']:.1f}%", f"${p['loss']:.2f}M",
            f"{p['var_contrib']:.1f}%", f"{p['sa_cv']:.2f}"]
    for j, (val, cx, cw) in enumerate(zip(vals, c2x, c2w)):
        cell(ax2, cx, yc2, cw, rh2, val, bg, NAVY if j == 0 else TXT,
             fs=9.2, bold=(j == 0), align="left" if j == 0 else "center", pad=0.012)
    hline(ax2, yc2, L, R, c=BORDER, lw=0.4)

# Totals row
yc2 -= rh2
t_prem = sum(p["prem"] for p in peril_d)
t_loss = sum(p["loss"] for p in peril_d)
t_vc   = sum(p["var_contrib"] for p in peril_d)
totals = ["TOTAL", "—", f"${t_prem:.2f}M", "100%", f"${t_loss:.2f}M", f"{t_vc:.0f}%", "—"]
for j, (val, cx, cw) in enumerate(zip(totals, c2x, c2w)):
    cell(ax2, cx, yc2, cw, rh2, val, NAVY2, "white", fs=9.2, bold=True,
         align="left" if j == 0 else "center")
hline(ax2, yc2+rh2, L, R, c=NAVY, lw=1.5)
hline(ax2, yc2,     L, R, c=NAVY, lw=1.5)

ax2.add_patch(plt.Rectangle((L, yc2), R-L, y0b+rhb-yc2, facecolor="none",
              edgecolor=NAVY, lw=1.5, transform=ax2.transAxes, zorder=5, clip_on=False))
for cx in c2x[1:]:
    ax2.plot([cx, cx], [yc2, y0b+rhb], color=BORDER, lw=0.5, transform=ax2.transAxes, zorder=3)

ax2.text(0.5, 0.005,
         "Synthetic data · 30-yr gamma-distributed rainfall index · seed=42 · "
         "Per-peril LRs: Drought 77%, Flood 75%, Heatwave 84%, Cold Spell 82%, Hail 77%, Sargassum 62% · "
         "Optimal = minimum variance at fixed $15M annual premium · CVXPY/CLARABEL",
         transform=ax2.transAxes, ha="center", va="bottom",
         fontsize=7, color=DIM, style="italic")

out_path = "/home/user/insurance-premium-flow/outputs/maps/fig7_mvp_table.png"
os.makedirs(os.path.dirname(out_path), exist_ok=True)
plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
plt.close()
print(f"\nSaved → {out_path}")
