"""FairTrade vs FedAvg on Fed-Heart-Disease (milestone 7).

Per seed: a fresh validation split (25% of each hospital's train split),
federated-impute normalization fit on the remaining 75%, then
  FedAvg     alpha = 0, lr tuned on validation balanced accuracy
  FairTrade  MOBO over (alpha, lr), 8 Sobol + 16 BO evaluations,
             objectives = validation balanced accuracy and -|SPD|
Both use the same trainer (fairfl/fairtrade.py). Every selection uses
validation metrics only; test metrics are only reported.

Reported per seed:
  * FairTrade operating points chosen on validation: the most accurate
    candidate with |val SPD| <= tau, for tau in {0.10, 0.05};
  * the validation Pareto front, re-scored on test, as a hypervolume over
    (balanced acc, -|SPD|) with reference point (0.5, -0.5).

Usage:
  python scripts/run_fairtrade.py                      # seeds 42..51
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
from fairfl.fairtrade import REF_POINT, fedavg_same_trainer, run_fairtrade  # noqa: E402
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


def select(cands, tau):
    """Most accurate candidate (validation) with |val SPD| <= tau."""
    ok = cands[cands.val_spd.abs() <= tau]
    if ok.empty:  # fall back to the fairest candidate
        return cands.loc[cands.val_spd.abs().idxmin()], False
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
    p.add_argument("--tag", default="", help="suffix for output files")
    args = p.parse_args()
    torch.set_num_threads(1)  # tiny model; avoids thread contention

    all_cands, rows = [], []
    for seed in args.seeds:
        t0 = time.time()
        data = build_sites(args.norm, val_frac=0.25, split_seed=seed)
        base = fedavg_same_trainer(data, seed)
        cands = pd.DataFrame(run_fairtrade(data, seed, args.n_init, args.n_iter))
        cands["val_pareto"] = pareto_mask(
            np.c_[cands.val_bacc.fillna(0), -cands.val_spd.abs().fillna(1)])
        all_cands.append(cands)

        front = cands[cands.val_pareto]
        row = {"seed": seed,
               "fedavg_lr": base["lr"],
               "fedavg_test_bacc": base["test_bacc"],
               "fedavg_test_acc": base["test_acc"],
               "fedavg_test_spd": base["test_spd"],
               "fedavg_test_eod": base["test_eod"],
               "fedavg_test_hv": hypervolume_2d([[base["test_bacc"], -abs(base["test_spd"])]]),
               "ft_front_size": len(front),
               "ft_test_hv": hypervolume_2d(np.c_[front.test_bacc, -front.test_spd.abs()])}
        for tau in TAUS:
            pick, met = select(cands, tau)
            key = f"ft{int(tau * 100):02d}"
            row.update({f"{key}_alpha": pick.alpha, f"{key}_lr": pick.lr,
                        f"{key}_met": met, f"{key}_test_bacc": pick.test_bacc,
                        f"{key}_test_acc": pick.test_acc, f"{key}_test_spd": pick.test_spd,
                        f"{key}_test_eod": pick.test_eod})
        rows.append(row)
        print(f"seed {seed} ({time.time() - t0:.0f}s): FedAvg bacc={base['test_bacc']:.3f} "
              f"SPD={base['test_spd']:+.3f} | FairTrade(tau=.10) bacc={row['ft10_test_bacc']:.3f} "
              f"SPD={row['ft10_test_spd']:+.3f} alpha={row['ft10_alpha']:.1f} | "
              f"front={len(front)} HV {row['fedavg_test_hv']:.4f} -> {row['ft_test_hv']:.4f}")

    tag = f"_{args.tag}" if args.tag else ""
    out_c = ROOT / "results" / f"fairtrade_candidates{tag}.csv"
    out_s = ROOT / "results" / f"fairtrade_summary{tag}.csv"
    pd.concat(all_cands).to_csv(out_c, index=False)
    df = pd.DataFrame(rows)
    df.to_csv(out_s, index=False)

    print(f"\nTest metrics, all-centers (n=254), {len(df)} seeds, mean +/- 95% CI\n")
    print("| method | balanced acc | acc | SPD (F-M) | EOD (F-M) | test hypervolume |")
    print("|---|---|---|---|---|---|")
    print(f"| FedAvg (same trainer) | {fmt(df.fedavg_test_bacc)} | {fmt(df.fedavg_test_acc)} | "
          f"{fmt(df.fedavg_test_spd, True)} | {fmt(df.fedavg_test_eod, True)} | {fmt(df.fedavg_test_hv)} |")
    for tau in TAUS:
        k = f"ft{int(tau * 100):02d}"
        print(f"| FairTrade, val SPD <= {tau:.2f} | {fmt(df[f'{k}_test_bacc'])} | {fmt(df[f'{k}_test_acc'])} | "
              f"{fmt(df[f'{k}_test_spd'], True)} | {fmt(df[f'{k}_test_eod'], True)} | "
              f"{'' if tau != TAUS[0] else fmt(df.ft_test_hv)} |")
    for tau in TAUS:
        k = f"ft{int(tau * 100):02d}"
        print(f"\nPaired FairTrade(tau={tau:.2f}) - FedAvg: "
              f"d balanced acc {fmt(df[f'{k}_test_bacc'] - df.fedavg_test_bacc, True)}, "
              f"d |SPD| {fmt(df[f'{k}_test_spd'].abs() - df.fedavg_test_spd.abs(), True)} "
              f"(constraint met on val in {int(df[f'{k}_met'].sum())}/{len(df)} seeds)")
    print(f"\nWritten {out_c.relative_to(ROOT)} and {out_s.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
