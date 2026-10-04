"""Accuracy and group-fairness metrics for binary predictions.

Sign convention: every gap is female minus male, so a negative SPD means women
are predicted to have heart disease less often than men. A metric that is
undefined for a split (a group or class is missing) is returned as NaN, never 0.
"""
import numpy as np
from scipy import stats

FEMALE, MALE = 0, 1


def _rate(mask, values):
    return values[mask].mean() if mask.any() else np.nan


def accuracy(y, yhat):
    return (y == yhat).mean()


def balanced_accuracy(y, yhat):
    """Mean of TPR and TNR; NaN if one class is absent."""
    tpr = _rate(y == 1, yhat == 1)
    tnr = _rate(y == 0, yhat == 0)
    return np.nan if np.isnan(tpr) or np.isnan(tnr) else (tpr + tnr) / 2


def spd(yhat, sex):
    """Statistical parity difference: P(yhat=1 | F) - P(yhat=1 | M)."""
    return _rate(sex == FEMALE, yhat == 1) - _rate(sex == MALE, yhat == 1)


def eod(y, yhat, sex):
    """Equal opportunity difference: TPR_F - TPR_M."""
    pos = y == 1
    return (_rate(pos & (sex == FEMALE), yhat == 1)
            - _rate(pos & (sex == MALE), yhat == 1))


def all_metrics(y, prob, sex, threshold=0.5):
    y, sex = np.asarray(y).astype(int), np.asarray(sex).astype(int)
    yhat = (np.asarray(prob) > threshold).astype(int)
    return {
        "acc": accuracy(y, yhat),
        "bacc": balanced_accuracy(y, yhat),
        "spd": spd(yhat, sex),
        "eod": eod(y, yhat, sex),
        "n": len(y),
        "n_female": int((sex == FEMALE).sum()),
    }


def mean_ci(values, confidence=0.95):
    """Mean and t-interval half-width over seeds, ignoring NaNs.

    Returns (mean, half_width, n_defined). The half-width is NaN when fewer
    than two seeds give a defined value.
    """
    v = np.asarray(values, dtype=float)
    v = v[~np.isnan(v)]
    if len(v) == 0:
        return np.nan, np.nan, 0
    if len(v) == 1:
        return v[0], np.nan, 1
    half = stats.t.ppf((1 + confidence) / 2, len(v) - 1) * stats.sem(v)
    return v.mean(), half, len(v)
