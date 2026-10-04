# Fair Federated Learning on Cross-Silo Medical Data

Studies the accuracy–fairness trade-off of federated learning on real
multi-hospital data with the [FLamby](https://github.com/owkin/FLamby) benchmark,
starting with **Fed-Heart-Disease**. FedAvg is compared against
[FairTrade](https://github.com/JordanManoj/HiWi-FairTrade-Challenge)-style
Pareto-optimal fair FL.

Status: baselines reproduced exactly against FLamby's published results;
FairTrade ported and evaluated (removes the sex gap in prediction rates at a
cost of 7 balanced-accuracy points). See [results/LOG.md](results/LOG.md) and
[docs/roadmap.html](docs/roadmap.html).

## Setup

```bash
bash scripts/setup.sh            # venv at ./.venv, FLamby pinned to edacf54, data → data/heart
```

The script clones FLamby into `external/` (git-ignored), installs only the
`heart` extra, re-pins dependencies from `requirements.txt`, and downloads the
data. The download asks you to accept the dataset's CC BY 4.0 terms.
If you move `data/heart`, run FLamby's
`dataset_creation_scripts/update_config.py --new-path <dir>`.

## Dataset: Fed-Heart-Disease (UCI, 1988)

| center | id | train | test | female train/test | positive rate (train) |
|---|---|---|---|---|---|
| Cleveland | 0 | 199 | 104 | 63 / 34 | 0.46 |
| Hungary | 1 | 172 | 89 | 50 / 19 | 0.38 |
| Switzerland | 2 | 30 | 16 | 3 / **0** | 1.00 |
| Long Beach VA | 3 | 85 | 45 | 2 / 3 | 0.78 |

740 patients in total. FLamby's split is fixed (`random_state=43`).

**Features.** The model sees 13 inputs, although FLamby's README says 16.
FLamby drops UCI columns 10–12 (slope, ca, thal) and one-hot encodes `cp` and
`restecg`. The resulting order is: `age, sex, trestbps, chol, fbs, thalach,
exang, oldpeak, cp_2, cp_3, cp_4, restecg_1, restecg_2`.
**Sex is index 1** (1 = male, 0 = female). It is a model input and is
z-scored by default, so group labels must come from a `normalize=False` copy
of the dataset.

**Fairness caveats that shape the analysis:**
- Switzerland has no women in its test set, so SPD and EOD are undefined
  there. Long Beach has 3 women in its test set, so its per-site fairness
  numbers are noise.
- Switzerland's training labels are 100% positive, so its local models are
  degenerate. Its test set has 1 negative out of 16, so balanced accuracy is
  close to meaningless.
- The headline fairness metrics will therefore be computed on the union of
  all test sets. Per-hospital fairness is reported only where both groups
  exist.

## Run

```bash
python scripts/run_fedavg.py                          # FedAvg, local norm, seeds 42-46, ~45 s on CPU
python scripts/run_fedavg.py --norm federated-impute  # federated normalization
python scripts/run_loho.py                            # leave-one-hospital-out, ~15 min
python scripts/run_baselines.py                       # local + pooled baselines
python scripts/run_fairtrade.py --notion eo --scope global  # FairTrade variants, ~10 min each
python scripts/compare_fairtrade.py                    # paired comparison of the 4 variants
python -m pytest tests                                # unit tests
```

## Layout

```
fairfl/      reusable code: data loaders with sex labels, fairness metrics, CIs, FairTrade
tests/       unit tests (incl. equivalence with the original FairTrade penalty)
scripts/     setup and experiment entry points
results/     LOG.md, the run log (mean ± 95% CI over ≥5 seeds), per-seed CSVs
docs/        roadmap.html, the project roadmap
external/    FLamby + FairTrade clones (ignored)
data/        downloaded data (ignored)
```

## Data credit

Janosi, A., Steinbrunn, W., Pfisterer, M., Detrano, R. (1988). *Heart Disease*.
UCI Machine Learning Repository. CC BY 4.0.
Benchmark: Ogier du Terrail et al., *FLamby*, NeurIPS 2022 Datasets & Benchmarks.
