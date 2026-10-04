"""FedAvg baseline on Fed-Heart-Disease with accuracy and sex-fairness metrics.

Uses FLamby's own FedAvg implementation and benchmark hyperparameters
(SGD, lr=1e-3, batch 4, 100 local updates/round, 15 rounds, seeds 42-46), so
the accuracy can be checked against FLamby's published results.

Feature normalization (--norm, see fairfl/data.py): local (FLamby default),
federated (global stats from per-hospital sufficient statistics), or
federated-impute (federated + zero-coded missing chol/trestbps imputed).

Evaluation sets:
  all-centers   union of the 4 hospital test sets, each normalized exactly as
                its hospital's training data was (what a deployed FL model
                sees); this is the headline number for accuracy and fairness.
  <hospital>    each hospital's test set on its own.
  flamby-pooled FLamby's "Pooled Test" (pooled-train normalization), reported
                with --norm local only, to sanity-check against the paper.

Usage:
  python scripts/run_fedavg.py                       # local norm, seeds 42..46
  python scripts/run_fedavg.py --norm federated-impute
  python scripts/run_fedavg.py --seeds 42 43 44 45 46 47 48 49 50 51
"""
import argparse
import os
import random
import sys
from pathlib import Path

os.environ.setdefault("TQDM_DISABLE", "1")

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from flamby.datasets.fed_heart_disease import (  # noqa: E402
    NUM_CLIENTS, Baseline, BaselineLoss, get_nb_max_rounds,
)
from flamby.strategies.fed_avg import FedAvg  # noqa: E402

from fairfl.data import (  # noqa: E402
    CENTER_NAMES, NORM_MODES, build_splits, load_flamby_pooled_test, train_loaders,
)
from fairfl.metrics import all_metrics, mean_ci  # noqa: E402

# FLamby benchmark config for FedAvg on heart (flamby/config_heart_disease.json)
LR = 1e-3
NUM_UPDATES = 100
NROUNDS = get_nb_max_rounds(NUM_UPDATES)  # 15
FLAMBY_PUBLISHED_POOLED_ACC = (0.748, 0.027)  # FedAvg100, mean / std, 5 seeds


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@torch.no_grad()
def predict(model, X):
    model.eval()
    return model(X).squeeze(1).numpy()


def run_seed(seed, train_datasets, test_sets, norm):
    set_seed(seed)
    strategy = FedAvg(
        training_dataloaders=train_loaders(train_datasets),
        model=Baseline(),
        loss=BaselineLoss(),
        optimizer_class=torch.optim.SGD,
        learning_rate=LR,
        num_updates=NUM_UPDATES,
        nrounds=NROUNDS,
        seed=seed,
    )
    model = strategy.run()[0]

    rows = []
    for name, (X, y, sex) in test_sets.items():
        m = all_metrics(y.numpy(), predict(model, X), sex.numpy())
        rows.append({"method": "FedAvg", "norm": norm, "seed": seed, "test": name, **m})
    return rows


def build_test_sets(per_center, norm):
    union = tuple(torch.cat([s[i] for s in per_center.values()]) for i in range(3))
    sets = {"all-centers": union, **per_center}
    if norm == "local":
        sets["flamby-pooled"] = load_flamby_pooled_test()
    return sets


def summarize(df):
    order = ["all-centers", *CENTER_NAMES, "flamby-pooled"]
    lines = []
    for test in order:
        sub = df[df.test == test]
        if sub.empty:
            continue
        cells = []
        for metric in ["acc", "bacc", "spd", "eod"]:
            mean, half, n = mean_ci(sub[metric])
            if n == 0:
                cells.append("n/a")
            elif np.isnan(half):
                cells.append(f"{mean:+.3f} (1 seed)" if metric in ("spd", "eod") else f"{mean:.3f} (1 seed)")
            else:
                fmt = "+.3f" if metric in ("spd", "eod") else ".3f"
                cells.append(f"{mean:{fmt}} +/- {half:.3f}")
        n_f = int(sub.n_female.iloc[0])
        lines.append(f"| {test} | {int(sub.n.iloc[0])} | {n_f} | " + " | ".join(cells) + " |")
    header = ("| test set | n | n female | acc | balanced acc | SPD (F-M) | EOD (F-M) |\n"
              "|---|---|---|---|---|---|---|")
    return header + "\n" + "\n".join(lines)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    p.add_argument("--norm", choices=NORM_MODES, default="local")
    p.add_argument("--out", type=Path, default=None,
                   help="per-seed CSV (default: results/fedavg_heart_<norm>.csv)")
    args = p.parse_args()
    out = args.out or ROOT / "results" / f"fedavg_heart_{args.norm}.csv"

    train_datasets, per_center = build_splits(args.norm)
    test_sets = build_test_sets(per_center, args.norm)
    rows = []
    for seed in args.seeds:
        seed_rows = run_seed(seed, train_datasets, test_sets, args.norm)
        rows += seed_rows
        overall = next(r for r in seed_rows if r["test"] == "all-centers")
        per_site = "  ".join(f"{r['test'][:5]}={r['acc']:.3f}" for r in seed_rows
                             if r["test"] in CENTER_NAMES)
        print(f"seed {seed}: acc={overall['acc']:.3f} SPD={overall['spd']:+.3f} "
              f"EOD={overall['eod']:+.3f} | {per_site}")

    df = pd.DataFrame(rows)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    print(f"\nFedAvg (norm={args.norm}), {len(args.seeds)} seeds, "
          f"mean +/- 95% CI (t-interval)\n")
    print(summarize(df))

    if args.norm == "local":
        acc = df[df.test == "flamby-pooled"].acc
        pub_mean, pub_std = FLAMBY_PUBLISHED_POOLED_ACC
        print(f"\nSanity check vs FLamby paper (FedAvg100, pooled test): ours "
              f"{acc.mean():.3f} +/- {acc.std(ddof=1):.3f} std  |  published "
              f"{pub_mean:.3f} +/- {pub_std:.3f} std")
    print(f"Per-seed rows written to {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
