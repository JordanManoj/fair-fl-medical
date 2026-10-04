"""Milestone 9: stress test of FairTrade's per-client penalty under sex segregation.

The fit patients of all four hospitals are re-split into 10 clients with
segregation level s (fairfl/partition.py). Validation and test sets are the
original hospital splits; normalization uses global statistics, which do not
depend on the partition.

To isolate the effect of the penalty scope, MOBO is replaced by the same
alpha grid for every variant, at the learning rate FedAvg selects on
validation. For each (notion, scope) the grid plus FedAvg (alpha = 0) forms
the candidate set; the validation Pareto front (balanced acc, -|soft gap|) is
re-scored on test as a hypervolume, and an operating point is chosen as the
most accurate candidate with |val soft gap| <= 0.10.

Usage:
  python scripts/run_stress.py --segregation 1.0      # run one level, 10 seeds
  python scripts/run_stress.py --summarize            # tables over all levels
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
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from flamby.datasets.fed_heart_disease import Baseline  # noqa: E402

from fairfl.data import build_sites  # noqa: E402
from fairfl.fairtrade import GAP_METRIC, evaluate_union, train_federated  # noqa: E402
from fairfl.metrics import mean_ci  # noqa: E402
from fairfl.partition import segregate  # noqa: E402
from run_fairtrade import hypervolume_2d, pareto_mask  # noqa: E402

ALPHAS = (1, 3, 10, 30, 100, 300, 1000)
LRS = (0.01, 0.03, 0.1, 0.3)
N_CLIENTS, N_WOMEN_CLIENTS = 10, 2
TAU = 0.10


def run_level(segregation, seeds, out):
    """Run the given seeds, appending each finished seed to `out` (resumable)."""
    done = set(pd.read_csv(out).seed) if out.exists() else set()
    for seed in seeds:
        if seed in done:
            continue
        rows = []
        t0 = time.time()
        data = build_sites("federated-impute", val_frac=0.25, split_seed=seed)
        clients = segregate(data["fit"], N_CLIENTS, segregation, N_WOMEN_CLIENTS, seed)
        torch.manual_seed(seed)
        init = Baseline().state_dict()

        def record(variant, alpha, lr, model):
            val, test = evaluate_union(model, data["val"]), evaluate_union(model, data["test"])
            rows.append({"segregation": segregation, "seed": seed, "variant": variant,
                         "alpha": alpha, "lr": lr,
                         **{f"val_{k}": v for k, v in val.items()},
                         **{f"test_{k}": v for k, v in test.items()}})

        best = None
        for lr in LRS:
            m = train_federated(clients, 0.0, lr, init)
            bacc = evaluate_union(m, data["val"])["bacc"]
            if best is None or bacc > best[0]:
                best = (bacc, lr, m)
        lr = best[1]
        record("fedavg", 0.0, lr, best[2])
        for notion in ("dp", "eo"):
            for scope in ("local", "global"):
                for alpha in ALPHAS:
                    m = train_federated(clients, alpha, lr, init, notion=notion, scope=scope)
                    record(f"{notion}_{scope}", alpha, lr, m)
        pd.DataFrame(rows).to_csv(out, mode="a", header=not out.exists(), index=False)
        women = [int((c[2] == 0).sum()) for c in clients]
        print(f"s={segregation} seed {seed} ({time.time() - t0:.0f}s) lr={lr} women/client={women}",
              flush=True)


def per_seed_scores(df):
    """Per (segregation, seed, variant): test HV in the own space and the operating point."""
    out = []
    for (s, seed), g in df.groupby(["segregation", "seed"]):
        fed = g[g.variant == "fedavg"]
        for variant in sorted(set(g.variant) - {"fedavg"}):
            gap = GAP_METRIC[variant.split("_")[0]]
            cands = pd.concat([fed, g[g.variant == variant]])
            mask = pareto_mask(np.c_[cands.val_bacc.fillna(0),
                                     -cands[f"val_{gap}_soft"].abs().fillna(1)])
            front = cands[mask]
            hv = hypervolume_2d(np.c_[front.test_bacc, -front[f"test_{gap}"].abs().fillna(1)])
            ok = cands[cands[f"val_{gap}_soft"].abs() <= TAU]
            pick = ok.loc[ok.val_bacc.idxmax()] if not ok.empty else \
                cands.loc[cands[f"val_{gap}_soft"].abs().idxmin()]
            out.append({"segregation": s, "seed": seed, "variant": variant, "gap": gap,
                        "hv": hv, "bacc": pick.test_bacc, "spd": pick.test_spd,
                        "eod": pick.test_eod, "alpha": pick.alpha, "met": not ok.empty})
        out.append({"segregation": s, "seed": seed, "variant": "fedavg", "gap": None,
                    "hv_spd": hypervolume_2d([[fed.test_bacc.iloc[0], -abs(fed.test_spd.iloc[0])]]),
                    "hv_eod": hypervolume_2d([[fed.test_bacc.iloc[0], -abs(fed.test_eod.iloc[0])]]),
                    "bacc": fed.test_bacc.iloc[0], "spd": fed.test_spd.iloc[0],
                    "eod": fed.test_eod.iloc[0], "alpha": 0.0})
    return pd.DataFrame(out)


def fmt(v, signed=False):
    mean, half, n = mean_ci(v)
    if n == 0:
        return "n/a"
    return f"{mean:{'+.3f' if signed else '.3f'}}" + ("" if np.isnan(half) else f" ± {half:.3f}")


def summarize():
    sys.stdout.reconfigure(encoding="utf-8")
    df = pd.concat(pd.read_csv(f) for f in sorted((ROOT / "results").glob("stress_s*.csv")))
    sc = per_seed_scores(df)
    for gap, notion in (("eod", "eo"), ("spd", "dp")):
        print(f"\n### {notion.upper()} (hypervolume over balanced acc, −|{gap.upper()}|; "
              f"operating point |val soft {gap.upper()}| ≤ {TAU})\n")
        print(f"| segregation | FedAvg {gap.upper()} | local: bacc | local: {gap.upper()} | "
              f"global: bacc | global: {gap.upper()} | HV FedAvg | HV local | HV global | "
              f"HV global − local |")
        print("|---|---|---|---|---|---|---|---|---|---|")
        for s, g in sc.groupby("segregation"):
            fed = g[g.variant == "fedavg"].set_index("seed")
            loc = g[g.variant == f"{notion}_local"].set_index("seed")
            glo = g[g.variant == f"{notion}_global"].set_index("seed")
            d = glo.hv - loc.hv
            p = stats.ttest_rel(glo.hv, loc.hv).pvalue if d.abs().sum() > 0 else float("nan")
            print(f"| {s:.1f} | {fmt(fed[gap], True)} | {fmt(loc.bacc)} | {fmt(loc[gap], True)} | "
                  f"{fmt(glo.bacc)} | {fmt(glo[gap], True)} | {fmt(fed[f'hv_{gap}'])} | "
                  f"{fmt(loc.hv)} | {fmt(glo.hv)} | {fmt(d, True)} (p={p:.3f}, "
                  f"{int((d > 0).sum())}/{len(d)} up) |")


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--segregation", type=float)
    p.add_argument("--seeds", type=int, nargs="+", default=list(range(42, 52)))
    p.add_argument("--summarize", action="store_true")
    args = p.parse_args()
    torch.set_num_threads(1)
    if args.summarize:
        summarize()
        return
    out = ROOT / "results" / f"stress_s{args.segregation:.1f}.csv"
    run_level(args.segregation, args.seeds, out)
    print(f"Written {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
