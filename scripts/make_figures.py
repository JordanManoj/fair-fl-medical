"""Figures for report/report.html, computed from the committed result CSVs.

  report/fig_tradeoff.svg  milestone 8 operating points: balanced accuracy vs
                           SPD and vs EOD (mean and 95% CI over 10 seeds)
  report/fig_stress.svg    milestone 9: hypervolume gain of the pooled over the
                           per-hospital penalty, by segregation level

Palette: validated categorical slots 1-3 (blue, orange, aqua) plus a neutral
for FedAvg; each series also has its own marker shape and a direct label.

Usage:
  python scripts/make_figures.py
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from fairfl.metrics import mean_ci  # noqa: E402
from run_stress import per_seed_scores  # noqa: E402

RES, OUT = ROOT / "results", ROOT / "report"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
NEUTRAL = "#7d7c78"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8.5, "axes.edgecolor": INK2,
    "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.8,
    "svg.fonttype": "none",
})


def ci(values):
    mean, half, _ = mean_ci(values)
    return mean, half


def tradeoff():
    dp = pd.read_csv(RES / "fairtrade_summary_dp_local.csv")
    eo_l = pd.read_csv(RES / "fairtrade_summary_eo_local.csv")
    eo_g = pd.read_csv(RES / "fairtrade_summary_eo_global.csv")
    series = [
        ("FedAvg", NEUTRAL, "D", dp.fedavg_test_bacc, dp.fedavg_test_spd, dp.fedavg_test_eod),
        ("FairTrade-DP, local", BLUE, "o", dp.ft10_test_bacc, dp.ft10_test_spd, dp.ft10_test_eod),
        ("FairTrade-EO, local", ORANGE, "s", eo_l.ft10_test_bacc, eo_l.ft10_test_spd,
         eo_l.ft10_test_eod),
        ("FairTrade-EO, pooled", AQUA, "^", eo_g.ft10_test_bacc, eo_g.ft10_test_spd,
         eo_g.ft10_test_eod),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.75), sharey=True)
    offsets = {"FedAvg": (-8, 7, "right"), "FairTrade-DP, local": (7, -12, "left"),
               "FairTrade-EO, local": (7, 7, "left"),
               "FairTrade-EO, pooled": (-8, -13, "right")}
    for ax, gap_idx, label in ((axes[0], 4, "SPD, women − men (0 = parity)"),
                               (axes[1], 5, "EOD, women − men (0 = equal detection)")):
        ax.axvline(0, color=INK2, lw=0.8, ls=(0, (3, 3)), zorder=1)
        ax.grid(axis="y", color=GRID, lw=0.6, zorder=0)
        for s in series:
            name, color, marker, bacc, gap = s[0], s[1], s[2], s[3], s[gap_idx]
            ax.scatter(gap, bacc, s=10, color=color, alpha=0.25, lw=0, zorder=2)
            (mx, hx), (my, hy) = ci(gap), ci(bacc)
            ax.errorbar(mx, my, xerr=hx, yerr=hy, fmt=marker, ms=6.5, color=color,
                        mec="white", mew=0.9, elinewidth=1.4, capsize=0, zorder=3)
            short = name.replace("FairTrade-", "").replace(",", "").replace("DP local", "DP")
            dx, dy, ha = offsets[name]
            ax.annotate(short, (mx, my), xytext=(dx, dy), textcoords="offset points", ha=ha,
                        fontsize=7.5, color=INK, zorder=4)
        ax.set_xlabel(label)
        ax.set_xlim(-0.55, 0.25)
    axes[0].set_ylabel("Balanced accuracy (test)")
    axes[0].set_ylim(0.5, 0.86)
    handles = [plt.Line2D([], [], marker=m, color=c, ls="", ms=6, mec="white", label=n)
               for n, c, m, *_ in series]
    fig.legend(handles=handles, loc="upper center", ncol=4, frameon=False, fontsize=7.5,
               bbox_to_anchor=(0.5, 1.02), handletextpad=0.3, columnspacing=1.2)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(OUT / "fig_tradeoff.svg")
    plt.close(fig)


def stress():
    df = pd.concat(pd.read_csv(f) for f in sorted(RES.glob("stress_s*.csv")))
    sc = per_seed_scores(df)
    fig, ax = plt.subplots(figsize=(4.6, 2.6))
    ax.axhline(0, color=INK2, lw=0.8, zorder=1)
    ax.grid(axis="y", color=GRID, lw=0.6, zorder=0)
    levels = sorted(sc.segregation.unique())
    for notion, color, marker, dx, name in (("dp", BLUE, "o", -0.012, "Demographic parity"),
                                            ("eo", ORANGE, "s", 0.012, "Equal opportunity")):
        means, halves = [], []
        for s in levels:
            g = sc[sc.segregation == s]
            d = (g[g.variant == f"{notion}_global"].set_index("seed").hv
                 - g[g.variant == f"{notion}_local"].set_index("seed").hv)
            m, h = ci(d)
            means.append(m)
            halves.append(h)
        x = np.array(levels) + dx
        ax.plot(x, means, color=color, lw=1.6, zorder=2)
        ax.errorbar(x, means, yerr=halves, fmt=marker, ms=6, color=color, mec="white",
                    mew=0.9, elinewidth=1.4, capsize=0, zorder=3, label=name)
        ax.annotate(f"{means[-1]:+.3f}", (x[-1], means[-1]), xytext=(7, -3),
                    textcoords="offset points", fontsize=7.5, color=INK)
    ax.set_xticks(levels, [f"{int(s * 100)}%" for s in levels])
    ax.set_xlim(-0.06, 1.12)
    ax.set_xlabel("Sex segregation of the 10 clients (100% = single-sex clients)")
    ax.set_ylabel("Hypervolume gain,\npooled − per-hospital")
    ax.legend(frameon=False, fontsize=7.5, loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT / "fig_stress.svg")
    plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    tradeoff()
    stress()
    print("Written report/fig_tradeoff.svg, report/fig_stress.svg")
