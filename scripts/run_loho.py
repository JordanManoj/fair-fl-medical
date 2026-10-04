"""Leave-one-hospital-out (LOHO) FedAvg on Fed-Heart-Disease.

Tests whether the accuracy gain from federated normalization transfers to a
hospital the model has never seen. If the gain came from the model recognizing
which site a patient is from, it should shrink or vanish under LOHO.

For each held-out hospital: FedAvg on the other 3 (FLamby hyperparameters),
evaluate on all of the held-out hospital's patients, for each normalization
mode and seed. Seeds are shared across modes, so differences are paired.

Usage:
  python scripts/run_loho.py
  python scripts/run_loho.py --seeds 42 43 44 --norms local federated-impute
"""
import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("TQDM_DISABLE", "1")

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from fairfl.data import CENTER_NAMES, NORM_MODES, build_loho_splits  # noqa: E402
from fairfl.metrics import mean_ci  # noqa: E402
from run_fedavg import run_seed  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--seeds", type=int, nargs="+", default=list(range(42, 52)))
    p.add_argument("--norms", nargs="+", choices=NORM_MODES, default=NORM_MODES)
    p.add_argument("--out", type=Path, default=ROOT / "results" / "loho_heart.csv")
    args = p.parse_args()

    rows = []
    for holdout, name in enumerate(CENTER_NAMES):
        for norm in args.norms:
            train, test = build_loho_splits(norm, holdout)
            for seed in args.seeds:
                for r in run_seed(seed, train, {name: test}, norm):
                    rows.append({**r, "holdout": name})
            accs = [r["acc"] for r in rows if r["holdout"] == name and r["norm"] == norm]
            mean, half, _ = mean_ci(accs)
            print(f"held out {name:12s} norm={norm:17s} acc={mean:.3f} +/- {half:.3f}")

    df = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"Per-seed rows written to {args.out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
