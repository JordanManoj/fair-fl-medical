"""Phase-1 close-out: every method under one protocol on the real hospitals.

Methods (all on the same trainer, splits and federated-impute normalization):
  fedavg               lr tuned on validation balanced accuracy
  fairtrade_dp_local   FairTrade penalty, demographic parity, per-client
  fairtrade_eo_local   FairTrade penalty, equal opportunity, per-client
  fairtrade_eo_global  equal opportunity on pooled group statistics (ours)
  fedfb                FedFB, demographic parity (fairfl/baselines.py)
  fedfuel_dp           Fed-FUEL, discrimination = SPD
  fedfuel_eo           Fed-FUEL, discrimination = EOD
Each fair method sweeps its own fairness knob on a fixed grid at FedAvg's
selected learning rate (as in run_stress.py); FedAvg is always a candidate.
Selection uses validation only: the operating point is the most accurate
candidate whose |val soft gap| (the method's notion) is at most half of
FedAvg's on the same seed. A fixed threshold would not do: soft EOD runs on a
smaller scale than soft SPD, so FedAvg itself often passes |soft EOD| <= 0.10.
Hypervolume needs no threshold and is the headline metric.

Two kinds of uncertainty are reported (--summarize):
  * over seeds: mean and 95% t-interval (training variability);
  * two-level bootstrap over seeds and test patients (adds test-sampling
    noise; the test set has 56 women, 15 of them sick), as percentile
    intervals for each method and for its paired difference to FedAvg.

Usage:
  python scripts/run_compare.py                 # 10 seeds, resumable per seed
  python scripts/run_compare.py --summarize
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

from fairfl.baselines import train_fedfb, train_fedfuel  # noqa: E402
from fairfl.data import build_sites  # noqa: E402
from fairfl.fairtrade import train_federated  # noqa: E402
from fairfl.metrics import all_metrics, mean_ci  # noqa: E402
from run_fairtrade import hypervolume_2d, pareto_mask  # noqa: E402

OUT_DIR = ROOT / "results" / "compare"
LRS = (0.01, 0.03, 0.1, 0.3)
GRIDS = {
    "fairtrade_dp_local": (1, 3, 10, 30, 100, 300, 1000),
    "fairtrade_eo_local": (1, 3, 10, 30, 100, 300, 1000),
    "fairtrade_eo_global": (1, 3, 10, 30, 100, 300, 1000),
    "fedfb": (0.003, 0.01, 0.03, 0.1, 0.3, 1, 3),
    "fedfuel_dp": (0.025, 0.05, 0.1, 0.2, 0.4),
    "fedfuel_eo": (0.025, 0.05, 0.1, 0.2, 0.4),
}
NOTION = {"fairtrade_dp_local": "dp", "fairtrade_eo_local": "eo", "fairtrade_eo_global": "eo",
          "fedfb": "dp", "fedfuel_dp": "dp", "fedfuel_eo": "eo"}
GAP = {"dp": "spd", "eo": "eod"}
REL_TAU = 0.5  # operating point must at least halve FedAvg's validation soft gap


@torch.no_grad()
def predict(model, sites):
    model.eval()
    return torch.cat([model(s[0]).squeeze(1) for s in sites]).numpy()


def union(sites, i):
    return torch.cat([s[i] for s in sites]).numpy()


def run_seed(seed):
    data = build_sites("federated-impute", val_frac=0.25, split_seed=seed)
    yv, sv = union(data["val"], 1), union(data["val"], 2)
    yt, st = union(data["test"], 1), union(data["test"], 2)
    torch.manual_seed(seed)
    init = Baseline().state_dict()
    rows, preds = [], {}

    def record(method, knob, lr, model):
        pv, pt = predict(model, data["val"]), predict(model, data["test"])
        key = f"{method}|{knob}"
        preds[key] = pt.astype(np.float32)
        rows.append({"seed": seed, "method": method, "knob": knob, "lr": lr, "key": key,
                     **{f"val_{k}": v for k, v in all_metrics(yv, pv, sv).items()},
                     **{f"test_{k}": v for k, v in all_metrics(yt, pt, st).items()}})

    best = None
    for lr in LRS:
        m = train_federated(data["fit"], 0.0, lr, init)
        bacc = all_metrics(yv, predict(m, data["val"]), sv)["bacc"]
        if best is None or bacc > best[0]:
            best = (bacc, lr, m)
    lr = best[1]
    record("fedavg", 0.0, lr, best[2])

    for method, grid in GRIDS.items():
        for knob in grid:
            if method.startswith("fairtrade"):
                _, notion, scope = method.split("_")
                m = train_federated(data["fit"], knob, lr, init, notion=notion, scope=scope)
            elif method == "fedfb":
                m = train_fedfb(data["fit"], knob, lr, init)
            else:
                m = train_fedfuel(data["fit"], data["val"], knob, lr, init,
                                  notion=NOTION[method], seed=seed)
            record(method, knob, lr, m)
    return pd.DataFrame(rows), preds, yt, st


def operating_points(df):
    """Per (seed, method): the validation-selected candidate key and its HV."""
    out = []
    for seed, g in df.groupby("seed"):
        fed = g[g.method == "fedavg"]
        out.append({"seed": seed, "method": "fedavg", "key": fed.key.iloc[0],
                    "hv_spd": hypervolume_2d([[fed.test_bacc.iloc[0], -abs(fed.test_spd.iloc[0])]]),
                    "hv_eod": hypervolume_2d([[fed.test_bacc.iloc[0], -abs(fed.test_eod.iloc[0])]])})
        for method in GRIDS:
            gap = GAP[NOTION[method]]
            c = pd.concat([fed, g[g.method == method]])
            soft = c[f"val_{gap}_soft"].abs()
            tau = REL_TAU * abs(fed[f"val_{gap}_soft"].iloc[0])
            ok = c[soft <= tau]
            pick = ok.loc[ok.val_bacc.idxmax()] if not ok.empty else c.loc[soft.idxmin()]
            front = c[pareto_mask(np.c_[c.val_bacc.fillna(0), -soft.fillna(1)])]
            row = {"seed": seed, "method": method, "key": pick.key, "met": not ok.empty}
            for space in ("spd", "eod"):
                row[f"hv_{space}"] = hypervolume_2d(
                    np.c_[front.test_bacc, -front[f"test_{space}"].abs().fillna(1)])
            out.append(row)
    return pd.DataFrame(out)


def bootstrap(op, preds_by_seed, y, s, B=2000, rng_seed=0):
    """Two-level percentile bootstrap (seeds, then test patients) per method + vs FedAvg."""
    rng = np.random.default_rng(rng_seed)
    seeds = sorted(preds_by_seed)
    methods = list(dict.fromkeys(op.method))
    pick = {(r.seed, r.method): r.key for r in op.itertuples()}
    draws = {m: [] for m in methods}
    for _ in range(B):
        ss = rng.choice(seeds, len(seeds))
        idx = rng.integers(0, len(y), len(y))
        for m in methods:
            vals = [all_metrics(y[idx], preds_by_seed[sd][pick[(sd, m)]][idx], s[idx])
                    for sd in ss]
            draws[m].append([np.nanmean([v[k] for v in vals]) for k in ("bacc", "spd", "eod")])
    draws = {m: np.array(v) for m, v in draws.items()}
    rows = []
    for m in methods:
        d, diff = draws[m], draws[m] - draws["fedavg"]
        row = {"method": m}
        for j, k in enumerate(("bacc", "spd", "eod")):
            row[k] = tuple(np.nanpercentile(d[:, j], [2.5, 97.5]))
            row[f"d_{k}"] = tuple(np.nanpercentile(diff[:, j], [2.5, 97.5]))
        rows.append(row)
    return rows


def fmt(v, signed=False):
    mean, half, n = mean_ci(v)
    f = "+.3f" if signed else ".3f"
    return "n/a" if n == 0 else f"{mean:{f}} ± {half:.3f}"


def summarize(B):
    sys.stdout.reconfigure(encoding="utf-8")
    df = pd.read_csv(OUT_DIR / "candidates.csv")
    preds, y, s = {}, None, None
    for f in sorted(OUT_DIR.glob("seed*.npz")):
        z = np.load(f)
        preds[int(f.stem[4:])] = {k: z[k] for k in z.files if "|" in k}
        y, s = z["y"], z["s"]
    op = operating_points(df)
    sel = op.merge(df, on=["seed", "key"], suffixes=("", "_c"))
    print(f"Operating points, all-centers test, {df.seed.nunique()} seeds (mean ± 95% CI over seeds)\n")
    print("| method | balanced acc | SPD (F−M) | EOD (F−M) | HV (bacc, −\\|SPD\\|) | HV (bacc, −\\|EOD\\|) |")
    print("|---|---|---|---|---|---|")
    for m, g in sel.groupby("method", sort=False):
        print(f"| {m} | {fmt(g.test_bacc)} | {fmt(g.test_spd, True)} | {fmt(g.test_eod, True)} | "
              f"{fmt(g.hv_spd)} | {fmt(g.hv_eod)} |")
    base = op[op.method == "fedavg"].set_index("seed")
    print("\nPaired over seeds vs FedAvg, hypervolume in the method's own space:")
    for m in GRIDS:
        o = op[op.method == m].set_index("seed")
        sp = f"hv_{GAP[NOTION[m]]}"
        d = o[sp] - base[sp]
        p = stats.ttest_rel(o[sp], base[sp]).pvalue
        print(f"- {m}: {fmt(d, True)} (p={p:.4f}, {int((d > 0).sum())}/{len(d)} up)")
    print(f"\nTwo-level bootstrap (seeds × test patients, B={B}), 95% percentile intervals:\n")
    print("| method | balanced acc | SPD | EOD | Δ bacc vs FedAvg | Δ SPD vs FedAvg | Δ EOD vs FedAvg |")
    print("|---|---|---|---|---|---|---|")
    for r in bootstrap(op, preds, y, s, B=B):
        c = lambda t: f"[{t[0]:+.3f}, {t[1]:+.3f}]"  # noqa: E731
        print(f"| {r['method']} | {c(r['bacc'])} | {c(r['spd'])} | {c(r['eod'])} | "
              f"{c(r['d_bacc'])} | {c(r['d_spd'])} | {c(r['d_eod'])} |")
    op.to_csv(OUT_DIR / "operating_points.csv", index=False)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--seeds", type=int, nargs="+", default=list(range(42, 52)))
    p.add_argument("--summarize", action="store_true")
    p.add_argument("--bootstrap", type=int, default=2000)
    args = p.parse_args()
    torch.set_num_threads(1)
    if args.summarize:
        summarize(args.bootstrap)
        return
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cand = OUT_DIR / "candidates.csv"
    done = set(pd.read_csv(cand).seed) if cand.exists() else set()
    for seed in args.seeds:
        if seed in done:
            continue
        t0 = time.time()
        df, preds, y, s = run_seed(seed)
        np.savez_compressed(OUT_DIR / f"seed{seed}.npz", y=y, s=s, **preds)
        df.to_csv(cand, mode="a", header=not cand.exists(), index=False)
        print(f"seed {seed} done ({time.time() - t0:.0f}s, {len(df)} candidates)", flush=True)


if __name__ == "__main__":
    main()
