#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
regenerate_fig1_fig_a1.py
Regenerates:
  - Figure 1: HAC Bandwidth Sensitivity (fixed overlap, correct stats)
  - Figure A1: ICGDF pipeline flowchart (clean, publication-quality)

Outputs saved to results/figures/revision_audit/ and figures/
"""
import sys
import os
os.environ["PYTHONIOENCODING"] = "utf-8"
if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

import os, sys
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
from scipy import stats, optimize
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import warnings
warnings.filterwarnings("ignore")

OUT_DIR = os.path.join(os.path.dirname(__file__), "../../results/figures/revision_audit")
FIG_DIR = os.path.join(os.path.dirname(__file__), "../../figures")
os.makedirs(OUT_DIR, exist_ok=True)

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 1  — HAC Bandwidth Sensitivity
# ══════════════════════════════════════════════════════════════════════════════
def make_fig1():
    # ── Inputs (canonical paper values) ──────────────────────────────────────
    IC_mean = -0.0005
    IC_std  =  0.2204
    T       =  1512

    def nw_var(rho, T, L):
        m = 1.0
        for k in range(1, L + 1):
            m += 2.0 * (1.0 - k / (L + 1)) * (rho ** k)
        return (IC_std ** 2 / T) * m

    # back-calculate rho from reported t = -0.09 at L=9
    HAC_t_paper = -0.09
    V_obs = (IC_mean / HAC_t_paper) ** 2
    try:
        rho = optimize.brentq(lambda r: nw_var(r, T, 9) - V_obs, -0.9999, 0.9999)
    except ValueError:
        rho = -0.022

    # rule-of-thumb and Andrews selector
    L_rot     = int(np.floor(4.0 * (T / 100.0) ** (2.0 / 9.0)))  # 7
    alpha_hat = 4.0 * rho**2 / (1.0 - rho)**4 if abs(rho) > 1e-10 else 0.0
    L_and     = max(1, int(np.ceil(1.1447 * (alpha_hat * T) ** (1.0 / 3.0))))  # 2
    L_paper   = 9

    lag_range = np.arange(1, 21)
    t_vals, p_vals = [], []
    for L in lag_range:
        V_L = nw_var(rho, T, L)
        t_L = IC_mean / np.sqrt(V_L)
        p_L = 1.0 - stats.norm.cdf(t_L)
        t_vals.append(t_L)
        p_vals.append(p_L)
    t_vals = np.array(t_vals)
    p_vals = np.array(p_vals)

    # ── Plot ──────────────────────────────────────────────────────────────────
    TEAL  = "#2A9D8F"
    CORAL = "#E76F51"
    SLATE = "#64748B"
    AMBER = "#B45309"

    plt.rcParams.update({
        "font.family":      "serif",
        "font.serif":       ["Times New Roman", "DejaVu Serif"],
        "font.size":        10,
        "axes.labelsize":   10,
        "legend.fontsize":  8.5,
        "legend.frameon":   True,
        "legend.framealpha": 0.92,
        "legend.edgecolor": "#CBD5E1",
        "figure.dpi":       300,
        "savefig.dpi":      300,
        "axes.spines.top":  False,
    })

    fig, ax1 = plt.subplots(figsize=(7.5, 4.2))
    fig.subplots_adjust(top=0.82, bottom=0.18, left=0.10, right=0.88)

    # ── Left axis: t-statistic ────────────────────────────────────────────────
    ax1.plot(lag_range, t_vals, color=TEAL, lw=2.0,
             marker="o", ms=4.5, zorder=4, label=r"HAC $t$-statistic")
    ax1.axhline(1.645, color=TEAL, lw=1.2, ls="--", alpha=0.7,
                label=r"Critical value ($t = 1.645$)")
    ax1.axhline(0.0, color="#94A3B8", lw=0.7, ls=":", alpha=0.5)

    ax1.set_xlabel(r"Newey–West Bandwidth $L$", labelpad=8)
    ax1.set_ylabel(r"HAC $t$-statistic", color=TEAL, labelpad=6)
    ax1.tick_params(axis="y", labelcolor=TEAL)
    t_pad = 0.12
    ax1.set_ylim(min(t_vals) - t_pad, 2.2)
    ax1.set_xlim(0.5, 20.5)
    ax1.set_xticks(np.arange(1, 21))
    ax1.grid(axis="y", alpha=0.12)
    ax1.grid(axis="x", alpha=0.06)

    # ── Right axis: p-value ───────────────────────────────────────────────────
    ax2 = ax1.twinx()
    ax2.plot(lag_range, p_vals, color=CORAL, lw=2.0, ls="--",
             marker="s", ms=4.5, zorder=4, label=r"One-tailed $p$-value")
    ax2.axhline(0.05, color=CORAL, lw=1.0, ls=":", alpha=0.6,
                label=r"$\alpha = 0.05$")
    ax2.set_ylabel(r"One-tailed $p$-value", color=CORAL, labelpad=6)
    ax2.tick_params(axis="y", labelcolor=CORAL)
    ax2.set_ylim(0.0, 1.05)
    ax2.spines["top"].set_visible(False)

    # ── Vertical markers: staggered labels to avoid overlap ───────────────────
    # L_and=2, L_rot=7, L_paper=9  →  7 and 9 close → stagger heights
    marker_cfg = [
        # (L, label_top_line, label_bot_line, y_frac_in_axes, x_offset)
        (L_and,   "Andrews (1991)",   f"$L = {L_and}$",  0.68, -0.25),
        (L_rot,   "Rule-of-thumb",    f"$L = {L_rot}$",  0.52,  0.20),
        (L_paper, "Paper bandwidth",  f"$L = {L_paper}$",0.36,  0.20),
    ]
    y_lim    = ax1.get_ylim()
    y_range  = y_lim[1] - y_lim[0]

    for (L_m, top_label, bot_label, y_frac, xoff) in marker_cfg:
        ax1.axvline(L_m, color=SLATE, lw=1.0, ls=":", alpha=0.55, zorder=1)
        y_pos = y_lim[0] + y_frac * y_range
        ax1.annotate(
            f"{top_label}\n{bot_label}",
            xy=(L_m, y_pos),
            xytext=(L_m + xoff, y_pos),
            fontsize=7.5,
            color=SLATE,
            va="center",
            ha="left" if xoff >= 0 else "right",
            arrowprops=dict(arrowstyle="-", color=SLATE, lw=0.7, alpha=0.6)
            if abs(xoff) > 0.1 else None,
        )

    # ── Title ─────────────────────────────────────────────────────────────────
    t_at_L9 = t_vals[L_paper - 1]
    p_at_L9 = p_vals[L_paper - 1]
    fig.text(
        0.5, 0.94,
        "Figure 1.  HAC Bandwidth Sensitivity ($L = 1$ to $20$)",
        ha="center", va="top",
        fontsize=11, fontweight="bold", fontfamily="serif",
    )
    fig.text(
        0.5, 0.88,
        (f"IC mean $= {IC_mean}$,  "
         f"$t(L=9) = {t_at_L9:.4f}$,  "
         f"$p = {p_at_L9:.4f}$ (upper tail) — gate CLOSED at all bandwidths"),
        ha="center", va="top",
        fontsize=8.5, color="#475569", fontfamily="serif",
    )

    # ── Combined legend (both axes) ───────────────────────────────────────────
    lines1, labs1 = ax1.get_legend_handles_labels()
    lines2, labs2 = ax2.get_legend_handles_labels()
    ax1.legend(
        lines1 + lines2, labs1 + labs2,
        loc="upper right",
        fontsize=8.0,
        bbox_to_anchor=(0.995, 0.99),
        ncol=1,
    )

    out1 = os.path.join(OUT_DIR, "figure_hac_sensitivity.png")
    out2 = os.path.join(FIG_DIR,  "fig01_hac_bandwidth.png")
    fig.savefig(out1, dpi=300, bbox_inches="tight")
    fig.savefig(out2, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure 1 saved → {out1}")
    print(f"Figure 1 saved → {out2}")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE A1  — ICGDF Pipeline Flowchart  (clean matplotlib version)
# ══════════════════════════════════════════════════════════════════════════════
def make_fig_a1():
    fig, ax = plt.subplots(figsize=(10, 6.8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7)
    ax.axis("off")

    # ── Colour palette ─────────────────────────────────────────────────────────
    C_DATA   = "#DBEAFE"  # blue-100
    C_PROC   = "#D1FAE5"  # green-100
    C_GATE   = "#FEF3C7"  # amber-100
    C_CLOSED = "#FEE2E2"  # red-100
    C_OPEN   = "#D1FAE5"  # green-100
    C_BORDER = "#1E3A5F"
    C_GREEN  = "#065F46"
    C_RED    = "#991B1B"
    C_AMBER  = "#78350F"
    FONT     = "DejaVu Sans"

    def box(ax, cx, cy, w, h, text, fill, border, tcolor="#1E293B",
            fs=8.5, bold=False, radius=0.12):
        """Draw a rounded rectangle with centred multi-line text."""
        rect = FancyBboxPatch(
            (cx - w/2, cy - h/2), w, h,
            boxstyle=f"round,pad=0.02,rounding_size={radius}",
            linewidth=1.4, edgecolor=border, facecolor=fill, zorder=2,
        )
        ax.add_patch(rect)
        ax.text(
            cx, cy, text,
            ha="center", va="center",
            fontsize=fs, color=tcolor,
            fontweight="bold" if bold else "normal",
            fontfamily=FONT,
            multialignment="center",
            zorder=3,
            linespacing=1.35,
        )

    def arrow(ax, x1, y1, x2, y2, color="#334155", lw=1.6):
        ax.annotate(
            "", xy=(x2, y2), xytext=(x1, y1),
            arrowprops=dict(
                arrowstyle="-|>",
                color=color,
                lw=lw,
                mutation_scale=14,
            ),
            zorder=1,
        )

    def label_arrow(ax, x, y, text, color="#64748B", fs=7.8):
        ax.text(x, y, text, ha="center", va="center",
                fontsize=fs, color=color, fontfamily=FONT,
                bbox=dict(fc="white", ec="none", pad=1.0), zorder=4)

    # ── Row 1: Data → Features → Ensemble → Calibration → Scores ─────────────
    row1_y = 5.8
    nodes_r1 = [
        (1.0,  "OHLCV Data\n(30 NASDAQ stocks\n2015–2024)",          C_DATA,   C_BORDER),
        (2.85, "Feature\nEngineering\n(49 indicators)",               C_DATA,   C_BORDER),
        (4.70, "Ensemble\nML Model\n(RF + MLP + CatBoost)",           C_PROC,   C_GREEN),
        (6.55, "Isotonic\nCalibration\n(held-out window)",            C_PROC,   C_GREEN),
        (8.40, "Conviction\nScores\n(calibrated probs)",              C_PROC,   C_GREEN),
    ]
    box_w, box_h = 1.60, 1.00
    for cx, txt, fill, border in nodes_r1:
        box(ax, cx, row1_y, box_w, box_h, txt, fill, border, fs=8.0)

    for i in range(len(nodes_r1) - 1):
        x1 = nodes_r1[i][0] + box_w/2
        x2 = nodes_r1[i+1][0] - box_w/2
        arrow(ax, x1, row1_y, x2, row1_y)

    # ── Row 2: Walk-forward validation (left) → IC Gate (centre) ─────────────
    row2_y = 3.90
    box(ax, 1.55, row2_y, 2.6, 1.05,
        "Walk-Forward Validation\n12 expanding folds × 126 OOS days\n"
        "[Train | MLP-val | Cal | Test] — no lookahead",
        "#F0FDF4", C_GREEN, fs=7.8)

    # Gate box (wider, prominent)
    box(ax, 5.30, row2_y, 3.20, 1.05,
        "ICGDF IC Gate\n① HAC Newey–West $t$-test (lag = 9)\n② Permutation test ($B$ = 1,000)",
        C_GATE, C_AMBER, tcolor=C_AMBER, fs=8.2, bold=True)

    # Arrow: walk-forward → gate
    arrow(ax, 2.85, row2_y, 3.70, row2_y, color=C_GREEN, lw=1.6)
    # Arrow: row1 scores → gate (vertical down)
    arrow(ax, 8.40, row1_y - box_h/2, 8.40, row2_y + 0.52, color="#334155", lw=1.4)
    ax.annotate("", xy=(6.90, row2_y + 0.52), xytext=(8.40, row2_y + 0.52),
                arrowprops=dict(arrowstyle="-|>", color="#334155", lw=1.4, mutation_scale=14))

    # ── Row 3: Gate CLOSED (left) & Gate OPEN (right) ─────────────────────────
    row3_y = 2.10
    box(ax, 2.90, row3_y, 3.20, 1.00,
        "Gate  CLOSED\n($p > 0.05$) → Capital Preserved\nNo ML deployment",
        C_CLOSED, C_RED, tcolor=C_RED, fs=8.2, bold=True)

    box(ax, 7.40, row3_y, 3.20, 1.00,
        "Gate  OPEN  (hypothetical)\n($p \\leq 0.05$) → TopK1 Ranking\n→ Backtest & Deploy",
        C_OPEN, C_GREEN, tcolor=C_GREEN, fs=8.2, bold=True)

    # Arrows from gate
    # Gate → CLOSED (actual result)
    ax.annotate("", xy=(2.90, row3_y + 0.50), xytext=(4.20, row2_y - 0.52),
                arrowprops=dict(arrowstyle="-|>", color=C_RED, lw=2.0, mutation_scale=15))
    label_arrow(ax, 3.20, 3.00, "Either condition\nfails", color=C_RED, fs=7.5)

    # Gate → OPEN
    ax.annotate("", xy=(7.40, row3_y + 0.50), xytext=(6.40, row2_y - 0.52),
                arrowprops=dict(arrowstyle="-|>", color=C_GREEN, lw=2.0, mutation_scale=15))
    label_arrow(ax, 7.05, 3.00, "Both conditions\nmet", color=C_GREEN, fs=7.5)

    # ── "Actual result" badge ─────────────────────────────────────────────────
    ax.text(2.90, 1.50, "← Actual result in this study",
            ha="center", va="top", fontsize=8.0, color=C_RED,
            fontweight="bold", fontfamily=FONT,
            bbox=dict(fc="#FFF1F2", ec=C_RED, pad=3, lw=0.8, boxstyle="round,pad=0.2"))

    # ── Title + caption ────────────────────────────────────────────────────────
    ax.text(5.0, 6.70,
            "Figure A1.  IC-Gated Walk-Forward Conviction Ranking Framework (ICGDF)",
            ha="center", va="top",
            fontsize=10.5, fontweight="bold", fontfamily=FONT, color="#0F172A")
    ax.text(5.0, 6.35,
            "The IC gate prevents capital deployment unless both HAC $t$-test and "
            "permutation test confirm IC > 0 at $\\alpha = 0.05$.",
            ha="center", va="top",
            fontsize=8.5, color="#475569", fontfamily=FONT)

    out1 = os.path.join(OUT_DIR, "figure_pipeline_a1.png")
    out2 = os.path.join(FIG_DIR,  "figA1_pipeline.png")
    fig.savefig(out1, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(out2, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Figure A1 saved → {out1}")
    print(f"Figure A1 saved → {out2}")


if __name__ == "__main__":
    print("Generating Figure 1...")
    make_fig1()
    print("\nGenerating Figure A1...")
    make_fig_a1()
    print("\nDone.")
