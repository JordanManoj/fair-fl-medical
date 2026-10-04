"""Local (one model per hospital) and pooled baselines on Fed-Heart-Disease.

Follows FLamby's benchmark: one shared Baseline() init per seed, Adam lr=1e-3,
batch 4, 50 epochs (flamby/benchmarks/fed_benchmark.py, train_single_centric).

Normalization matches how each model is trained:
  Local i  per-hospital stats; evaluated on every hospital's test set, each
           normalized with that hospital's own stats (FLamby convention).
  Pooled   pooled-train stats (= federated stats); evaluated on all hospitals'
           test sets normalized the same way.
Every model is also evaluated on FLamby's "Pooled Test" to check against the
published results.

Usage:
  python scripts/run_baselines.py                  # seeds 42..51
"""
import argparse
import copy
import os
import sys
from pathlib import Path

os.environ.setdefault("TQDM_DISABLE", "1")

import pandas as pd
import torch
from torch.utils.data import ConcatDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from flamby.datasets.fed_heart_disease import (  # noqa: E402
    LR, NUM_CLIENTS, NUM_EPOCHS_POOLED, Baseline, BaselineLoss,
)

from fairfl.data import (  # noqa: E402
    CENTER_NAMES, build_splits, load_flamby_pooled_test, train_loaders,
)
from fairfl.metrics import all_metrics  # noqa: E402
from run_fedavg import build_test_sets, predict, set_seed, summarize  # noqa: E402


def train_single(model, dataset, seed):
    set_seed(seed)
    loader = train_loaders([dataset])[0]
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = BaselineLoss()
    model.train()
    for _ in range(NUM_EPOCHS_POOLED):
        for X, y in loader:
            opt.zero_grad()
            loss_fn(model(X), y).backward()
            opt.step()
    return model


def evaluate(model, method, seed, test_sets):
    return [{"method": method, "seed": seed, "test": name,
             **all_metrics(y.numpy(), predict(model, X), sex.numpy())}
            for name, (X, y, sex) in test_sets.items()]


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--seeds", type=int, nargs="+", default=list(range(42, 52)))
    p.add_argument("--out", type=Path, default=ROOT / "results" / "baselines_heart.csv")
    args = p.parse_args()

    local_train, local_test = build_splits("local")
    fed_train, fed_test = build_splits("federated")
    local_sets = build_test_sets(local_test, "local")  # includes flamby-pooled
    pooled_sets = {**build_test_sets(fed_test, "federated"),
                   "flamby-pooled": load_flamby_pooled_test()}
    pooled_train = ConcatDataset(fed_train)

    rows = []
    for seed in args.seeds:
        set_seed(seed)
        init = Baseline()
        for c in range(NUM_CLIENTS):
            m = train_single(copy.deepcopy(init), local_train[c], seed)
            rows += evaluate(m, f"Local {CENTER_NAMES[c]}", seed, local_sets)
        m = train_single(copy.deepcopy(init), pooled_train, seed)
        rows += evaluate(m, "Pooled", seed, pooled_sets)
        print(f"seed {seed} done")

    df = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    for method in df.method.unique():
        print(f"\n{method}, {len(args.seeds)} seeds, mean +/- 95% CI\n")
        print(summarize(df[df.method == method]))
    print(f"\nPer-seed rows written to {args.out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
