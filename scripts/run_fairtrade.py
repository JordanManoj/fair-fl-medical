"""FairTrade vs FedAvg on Fed-Heart-Disease (milestones 7 and 8).

Per seed: a fresh validation split (25% of each hospital's train split),
federated-impute normalization fit on the remaining 75%, then
  FedAvg     alpha = 0, lr tuned on validation balanced accuracy
  FairTrade  MOBO over (alpha, lr), 8 Sobol + 16 BO evaluations,
             objectives = validation balanced accuracy and -|soft gap|
--notion dp|eo picks the fairness gap that is penalized, optimized and used
for selection (dp: SPD, eo: EOD). --scope local|global picks FairTrade's
per-hospital penalty or the penalty on globally aggregated group statistics.
Both use the same trainer (fairfl/fairtrade.py). Every selection uses
validation metrics only; test metrics are only reported.

Reported per seed:
  * FairTrade operating points chosen on validation: the most accurate
    candidate with |val soft gap| <= tau, for tau in {0.10, 0.05}
    (soft gap = difference in mean predicted risk, see metrics.all_metrics);
  * the validation Pareto front, re-scored on test, as a hypervolume over
    (balanced acc, -|gap|) with reference point (0.5, -0.5).

Usage:
  python scripts/run_fairtrade.py                      # dp, local, seeds 42..51
  python scripts/run_fairtrade.py --notion eo --scope global
  python scripts/run_fairtrade.py --seeds 42 --n-iter 4  # quick check
"""
import argparse
import os
import sys
import time
import warnings
from pathlib import Path

os.environ.setdefault("TQDM_DISABLE", "1")
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fairfl.data import NORM_MODES, build_sites  # noqa: E402
from fairfl.fairtrade import (  # noqa: E402
    GAP_METRIC, NOTIONS, REF_POINT, SCOPES, fedavg_same_trainer, run_fairtrade,
)
from fairfl.metrics import mean_ci  # noqa: E402

TAUS = (0.10, 0.05)


def pareto_mask(points):
    """Non-dominated rows of an (n, 2) array, both columns maximized."""
    pts = np.asarray(points, dtype=float)
    keep = np.ones(len(pts), dtype=bool)
    for i, p in enumerate(pts):
        dominated = np.all(pts >= p, axis=1) & np.any(pts > p, axis=1)
        keep[i] = not dominated.any()
    return keep


def hypervolume_2d(points, ref=REF_POINT):
    """Area dominated by `points` (both maximized) above the reference point."""
    pts = [p for p in np.asarray(points, dtype=float) if p[0] > ref[0] and p[1] > ref[1]]
    pts = sorted(pts, key=lambda p: -p[0])
    area, best_y = 0.0, ref[1]
    for x, y in pts:
        if y > best_y:
            area += (x - ref[0]) * (y - best_y)
            best_y = y
    return area


def select(cands, tau, gap):
    """Most accurate candidate (validation) with |val soft gap| <= tau."""
    ok = cands[cands[f"val_{gap}_soft"].abs() <= tau]
    if ok.empty:  # fall back to the fairest candidate
        return cands.loc[cands[f"val_{gap}_soft"].abs().idxmin()], False
    return ok.loc[ok.val_bacc.idxmax()], True


def fmt(values, signed=False):
    mean, half, n = mean_ci(values)
    if n == 0:
        return "n/a"
    f = "+.3f" if signed else ".3f"
    return f"{mean:{f}}" + ("" if np.isnan(half) else f" +/- {half:.3f}")


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--seeds", type=int, nargs="+", default=list(range(42, 52)))
    p.add_argument("--norm", choices=NORM_MODES, default="federated-impute")
    p.add_argument("--n-init", type=int, default=8)
    p.add_argument("--n-iter", type=int, default=16)
    p.add_argument("--notion", choices=NOTIONS, default="dp")
    p.add_argument("--scope", choices=SCOPES, default="local")
    p.add_argument("--tag", default="", help="suffix for output files")
    args = p.parse_args()
    gap = GAP_METRIC[args.notion]
    torch.set_num_threads(1)  # tiny model; avoids thread contention

    all_cands, rows = [], []
    for seed in args.seeds:
        t0 = time.time()
        data = build_sites(args.norm, val_frac=0.25, split_seed=seed)
        base = fedavg_same_trainer(data, seed)
        cands = pd.DataFrame(run_fairtrade(data, seed, args.n_init, args.n_iter,
                                           notion=args.notion, scope=args.scope))
        cands["val_pareto"] = pareto_mask(
            np.c_[cands.val_bacc.fillna(0), -cands[f"val_{gap}_soft"].abs().fillna(1)])
        all_cands.append(cands)

        front = cands[cands.val_pareto]
        row = {"seed": seed,
               "fedavg_lr": base["lr"],
               "fedavg_test_bacc": base["test_bacc"],
               "fedavg_test_acc": base["test_acc"],
               "fedavg_test_spd": base["test_spd"],
               "fedavg_test_eod": base["test_eod"],
               "fedavg_test_hv": hypervolume_2d([[base["test_bacc"], -abs(base[f"test_{gap}"])]]),
               "ft_front_size": len(front),
               "ft_test_hv": hypervolume_2d(
                   np.c_[front.test_bacc, -front[f"test_{gap}"].abs().fillna(1)])}
        for tau in TAUS:
            pick, met = select(cands, tau, gap)
            key = f"ft{int(tau * 100):02d}"
            row.update({f"{key}_alpha": pick.alpha, f"{key}_lr": pick.lr,
                        f"{key}_met": met, f"{key}_test_bacc": pick.test_bacc,
                        f"{key}_test_acc": pick.test_acc, f"{key}_test_spd": pick.test_spd,
                        f"{key}_test_eod": pick.test_eod})
        rows.append(row)
        print(f"seed {seed} ({time.time() - t0:.0f}s): FedAvg bacc={base['test_bacc']:.3f} "
              f"{gap}={base[f'test_{gap}']:+.3f} | FairTrade(tau=.10) bacc={row['ft10_test_bacc']:.3f} "
              f"{gap}={row[f'ft10_test_{gap}']:+.3f} alpha={row['ft10_alpha']:.1f} | "
              f"front={len(front)} HV {row['fedavg_test_hv']:.4f} -> {row['ft_test_hv']:.4f}")

    tag = f"_{args.notion}_{args.scope}" + (f"_{args.tag}" if args.tag else "")
    out_c = ROOT / "results" / f"fairtrade_candidates{tag}.csv"
    out_s = ROOT / "results" / f"fairtrade_summary{tag}.csv"
    pd.concat(all_cands).to_csv(out_c, index=False)
    df = pd.DataFrame(rows)
    df.to_csv(out_s, index=False)

    print(f"\nnotion={args.notion} scope={args.scope}: test metrics, all-centers (n=254), "
          f"{len(df)} seeds, mean +/- 95% CI; hypervolume over (balanced acc, -|{gap}|)\n")
    print("| method | balanced acc | acc | SPD (F-M) | EOD (F-M) | test hypervolume |")
    print("|---|---|---|---|---|---|")
    print(f"| FedAvg (same trainer) | {fmt(df.fedavg_test_bacc)} | {fmt(df.fedavg_test_acc)} | "
          f"{fmt(df.fedavg_test_spd, True)} | {fmt(df.fedavg_test_eod, True)} | {fmt(df.fedavg_test_hv)} |")
    for tau in TAUS:
        k = f"ft{int(tau * 100):02d}"
        print(f"| FairTrade, val |{gap}_soft| <= {tau:.2f} | {fmt(df[f'{k}_test_bacc'])} | {fmt(df[f'{k}_test_acc'])} | "
              f"{fmt(df[f'{k}_test_spd'], True)} | {fmt(df[f'{k}_test_eod'], True)} | "
              f"{'' if tau != TAUS[0] else fmt(df.ft_test_hv)} |")
    for tau in TAUS:
        k = f"ft{int(tau * 100):02d}"
        print(f"\nPaired FairTrade(tau={tau:.2f}) - FedAvg: "
              f"d balanced acc {fmt(df[f'{k}_test_bacc'] - df.fedavg_test_bacc, True)}, "
              f"d |{gap}| {fmt(df[f'{k}_test_{gap}'].abs() - df[f'fedavg_test_{gap}'].abs(), True)} "
              f"(constraint met on val in {int(df[f'{k}_met'].sum())}/{len(df)} seeds)")
    print(f"\nWritten {out_c.relative_to(ROOT)} and {out_s.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
