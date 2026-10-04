"""Fed-Heart-Disease loaders that also expose the sensitive attribute (sex).

FLamby z-scores every feature, including sex, so group membership is read from
the raw (``normalize=False``) features before normalizing them ourselves.

Normalization modes (all statistics are fit on training data only):
  local             each hospital uses its own train mean/std (FLamby default)
  federated         every hospital uses the global train mean/std, computed by
                    the server from per-hospital sufficient statistics
                    (count, sum, sum of squares) - no patient rows are shared
  federated-impute  as federated, but physiologically impossible zeros
                    (UCI's missing-value code for chol and trestbps) are first
                    replaced by the global train mean of the observed values
"""
import torch
from torch.utils.data import DataLoader, TensorDataset

from flamby.datasets.fed_heart_disease import BATCH_SIZE, NUM_CLIENTS, FedHeartDisease

CENTER_NAMES = ["Cleveland", "Hungary", "Switzerland", "Long Beach"]
FEATURES = ["age", "sex", "trestbps", "chol", "fbs", "thalach", "exang",
            "oldpeak", "cp_2", "cp_3", "cp_4", "restecg_1", "restecg_2"]
SEX_IDX = FEATURES.index("sex")
ZERO_MEANS_MISSING = [FEATURES.index("trestbps"), FEATURES.index("chol")]
NORM_MODES = ["local", "federated", "federated-impute"]
FEMALE, MALE = 0, 1
EPS = 1e-9  # same as FLamby


def _stack(ds):
    X = torch.stack([ds[i][0] for i in range(len(ds))])
    y = torch.cat([ds[i][1] for i in range(len(ds))])
    return X, y


def load_raw(center, train):
    """Un-normalized (X, y) for one hospital split, in FLamby's row order."""
    return _stack(FedHeartDisease(center=center, train=train, normalize=False))


def sufficient_stats(X):
    """What a hospital sends to the server: count, sum, sum of squares."""
    return len(X), X.sum(0), (X ** 2).sum(0)


def aggregate_mean_std(stats):
    """Server side: global mean and (ddof=1) std from per-hospital stats."""
    n = sum(s[0] for s in stats)
    s1 = sum(s[1] for s in stats)
    s2 = sum(s[2] for s in stats)
    mean = s1 / n
    var = (s2 - n * mean ** 2) / (n - 1)
    return mean, var.clamp_min(0).sqrt()


def federated_impute_means(raw_train):
    """Global mean of observed (non-zero) values, from per-hospital sums/counts."""
    means = {}
    for j in ZERO_MEANS_MISSING:
        total = sum(X[X[:, j] != 0, j].sum() for X, _ in raw_train)
        count = sum((X[:, j] != 0).sum() for X, _ in raw_train)
        means[j] = total / count
    return means


def impute(X, means):
    X = X.clone()
    for j, m in means.items():
        X[X[:, j] == 0, j] = m
    return X


def build_splits(norm="local"):
    """Return (train_datasets, test_sets).

    train_datasets: list of per-hospital TensorDatasets (normalized X, y[:,None]).
    test_sets: {hospital: (X normalized, y, sex)} for each hospital's test split.
    """
    assert norm in NORM_MODES, norm
    raw_train = [load_raw(c, True) for c in range(NUM_CLIENTS)]
    raw_test = [load_raw(c, False) for c in range(NUM_CLIENTS)]
    sex_test = [X[:, SEX_IDX].long() for X, _ in raw_test]

    if norm == "federated-impute":
        means = federated_impute_means(raw_train)
        raw_train = [(impute(X, means), y) for X, y in raw_train]
        raw_test = [(impute(X, means), y) for X, y in raw_test]

    if norm == "local":
        stats = [(X.mean(0), X.std(0)) for X, _ in raw_train]
    else:
        g = aggregate_mean_std([sufficient_stats(X) for X, _ in raw_train])
        stats = [g] * NUM_CLIENTS

    def norm_x(X, c):
        mean, std = stats[c]
        return (X - mean) / (std + EPS)

    train = [TensorDataset(norm_x(X, c), y[:, None]) for c, (X, y) in enumerate(raw_train)]
    test = {CENTER_NAMES[c]: (norm_x(X, c), y, sex_test[c])
            for c, (X, y) in enumerate(raw_test)}
    return train, test


def build_loho_splits(norm, holdout):
    """Leave-one-hospital-out: FL on the other 3 hospitals, test on `holdout`.

    The held-out hospital contributes nothing to training or to the federated
    statistics; all of its patients (train + test split) form the test set.
    With norm='local' it z-scores with the mean/std of its own unlabeled
    features (its FLamby train split), as a new site deploying the model would.

    Returns (train_datasets, (X, y, sex)) for the held-out hospital.
    """
    assert norm in NORM_MODES, norm
    sites = [c for c in range(NUM_CLIENTS) if c != holdout]
    raw_train = [load_raw(c, True) for c in sites]
    ho_parts = [load_raw(holdout, True), load_raw(holdout, False)]
    ho_X = torch.cat([p[0] for p in ho_parts])
    ho_y = torch.cat([p[1] for p in ho_parts])
    ho_sex = ho_X[:, SEX_IDX].long()
    ho_fit = ho_parts[0][0]  # held-out site's own (unlabeled) stats source

    if norm == "federated-impute":
        means = federated_impute_means(raw_train)
        raw_train = [(impute(X, means), y) for X, y in raw_train]
        ho_X, ho_fit = impute(ho_X, means), impute(ho_fit, means)

    if norm == "local":
        stats = [(X.mean(0), X.std(0)) for X, _ in raw_train]
        ho_stats = (ho_fit.mean(0), ho_fit.std(0))
    else:
        g = aggregate_mean_std([sufficient_stats(X) for X, _ in raw_train])
        stats, ho_stats = [g] * len(sites), g

    def norm_x(X, s):
        return (X - s[0]) / (s[1] + EPS)

    train = [TensorDataset(norm_x(X, s), y[:, None])
             for (X, y), s in zip(raw_train, stats)]
    return train, (norm_x(ho_X, ho_stats), ho_y, ho_sex)


def build_sites(norm="federated-impute", val_frac=0.25, split_seed=0):
    """Per-hospital (X, y, sex) tensors for fit / val / test, for tuned methods.

    Each hospital's FLamby train split is divided into a fit part and a
    validation part (stratified on y when both classes have >= 2 patients;
    Switzerland is all-positive, so it is split unstratified). Normalization
    and imputation statistics are fit on the fit parts only, so neither the
    validation nor the test data influence preprocessing.

    Returns {"fit": [...], "val": [...], "test": [...]}, each a list of
    (X, y, sex) per hospital in CENTER_NAMES order.
    """
    from sklearn.model_selection import train_test_split

    assert norm in NORM_MODES, norm
    fit_raw, val_raw = [], []
    for c in range(NUM_CLIENTS):
        X, y = load_raw(c, True)
        counts = torch.bincount(y.long(), minlength=2)
        strat = y.numpy() if (counts >= 2).all() else None
        i_fit, i_val = train_test_split(range(len(y)), test_size=val_frac,
                                        random_state=split_seed, stratify=strat)
        fit_raw.append((X[i_fit], y[i_fit]))
        val_raw.append((X[i_val], y[i_val]))
    test_raw = [load_raw(c, False) for c in range(NUM_CLIENTS)]

    def sex_of(parts):
        return [X[:, SEX_IDX].long() for X, _ in parts]

    sexes = {"fit": sex_of(fit_raw), "val": sex_of(val_raw), "test": sex_of(test_raw)}
    parts = {"fit": fit_raw, "val": val_raw, "test": test_raw}

    if norm == "federated-impute":
        means = federated_impute_means(fit_raw)
        parts = {k: [(impute(X, means), y) for X, y in v] for k, v in parts.items()}

    if norm == "local":
        stats = [(X.mean(0), X.std(0)) for X, _ in parts["fit"]]
    else:
        g = aggregate_mean_std([sufficient_stats(X) for X, _ in parts["fit"]])
        stats = [g] * NUM_CLIENTS

    return {
        k: [((X - stats[c][0]) / (stats[c][1] + EPS), y, sexes[k][c])
            for c, (X, y) in enumerate(v)]
        for k, v in parts.items()
    }


def train_loaders(train_datasets, batch_size=BATCH_SIZE):
    """Per-hospital shuffled train loaders (the FL clients)."""
    return [DataLoader(ds, batch_size=batch_size, shuffle=True) for ds in train_datasets]


def load_flamby_pooled_test():
    """FLamby's 'Pooled Test' (pooled-train normalization) as (X, y, sex)."""
    X, y = _stack(FedHeartDisease(train=False, pooled=True))
    X_raw, _ = _stack(FedHeartDisease(train=False, pooled=True, normalize=False))
    return X, y, X_raw[:, SEX_IDX].long()
