"""Compare the FairTrade variants of milestone 8 (notion x scope), paired by seed.

Reads results/fairtrade_{candidates,summary}_<notion>_<scope>.csv written by
run_fairtrade.py. For every variant and seed, the validation Pareto front
(selected with that variant's own soft gap) is re-scored on test and its
hypervolume is computed in both fairness spaces, (balanced acc, -|SPD|) and
(balanced acc, -|EOD|), so variants optimized for different notions can be
compared on the same scale.

Usage:
  python scripts/compare_fairtrade.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from fairfl.fairtrade import GAP_METRIC  # noqa: E402
from fairfl.metrics import mean_ci  # noqa: E402
from run_fairtrade import hypervolume_2d  # noqa: E402

VARIANTS = [("dp", "local"), ("dp", "global"), ("eo", "local"), ("eo", "global")]


def fmt(v, signed=False):
    mean, half, n = mean_ci(v)
    if n == 0:
        return "n/a"
    f = "+.3f" if signed else ".3f"
    return f"{mean:{f}}" + ("" if np.isnan(half) else f" ± {half:.3f}")


def paired(a, b, signed=True):
    d = np.asarray(b) - np.asarray(a)
    p = stats.ttest_rel(b, a).pvalue
    return f"{fmt(d, signed)} (p={p:.3f}, {int((d > 0).sum())}/{len(d)} up)"


def load(notion, scope):
    c = pd.read_csv(ROOT / "results" / f"fairtrade_candidates_{notion}_{scope}.csv")
    s = pd.read_csv(ROOT / "results" / f"fairtrade_summary_{notion}_{scope}.csv").set_index("seed")
    for space in ("spd", "eod"):
        s[f"hv_{space}"] = [
            hypervolume_2d(np.c_[f.test_bacc, -f[f"test_{space}"].abs().fillna(1)])
            for _, f in c[c.val_pareto].groupby("seed")
        ]
    return s


def main():
    sys.stdout.reconfigure(encoding="utf-8")  # Windows consoles default to cp1252
    runs ={v: load(*v) for v in VARIANTS}
    base = runs[VARIANTS[0]]
    fedavg_hv = {space: [hypervolume_2d([[b, -abs(g)]]) for b, g in
                         zip(base.fedavg_test_bacc, base[f"fedavg_test_{space}"])]
                 for space in ("spd", "eod")}

    print("Test, all-centers, 10 seeds, mean ± 95% CI. Operating point = most accurate "
          "candidate with |val soft gap| <= 0.10 (gap = the variant's own notion).\n")
    print("| variant | balanced acc | SPD (F−M) | EOD (F−M) | HV (bacc, −\\|SPD\\|) | HV (bacc, −\\|EOD\\|) |")
    print("|---|---|---|---|---|---|")
    print(f"| FedAvg (same trainer) | {fmt(base.fedavg_test_bacc)} | {fmt(base.fedavg_test_spd, True)} | "
          f"{fmt(base.fedavg_test_eod, True)} | {fmt(fedavg_hv['spd'])} | {fmt(fedavg_hv['eod'])} |")
    for (n, sc), s in runs.items():
        print(f"| FairTrade {n.upper()}, {sc} | {fmt(s.ft10_test_bacc)} | {fmt(s.ft10_test_spd, True)} | "
              f"{fmt(s.ft10_test_eod, True)} | {fmt(s.hv_spd)} | {fmt(s.hv_eod)} |")

    print("\nPaired comparisons (B − A):\n")
    for n in ("dp", "eo"):
        a, b = runs[(n, "local")], runs[(n, "global")]
        g = GAP_METRIC[n]
        print(f"- {n.upper()}: global − local | HV in own space {paired(a[f'hv_{g}'], b[f'hv_{g}'])} | "
              f"bacc {paired(a.ft10_test_bacc, b.ft10_test_bacc)} | "
              f"|{g}| {paired(a[f'ft10_test_{g}'].abs(), b[f'ft10_test_{g}'].abs())}")
    for sc in ("local", "global"):
        a, b = runs[("dp", sc)], runs[("eo", sc)]
        print(f"- {sc}: EO − DP | bacc {paired(a.ft10_test_bacc, b.ft10_test_bacc)} | "
              f"|EOD| {paired(a.ft10_test_eod.abs(), b.ft10_test_eod.abs())} | "
              f"HV(EOD space) {paired(a.hv_eod, b.hv_eod)}")


if __name__ == "__main__":
    main()
