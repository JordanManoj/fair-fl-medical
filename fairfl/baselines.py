"""FedFB and Fed-FUEL ported to the FLamby hospitals, on the same trainer.

Both use FLamby's Baseline (logistic regression), full-batch local Adam,
size-weighted aggregation and the fit/val/test splits of fairfl.data, so they
compare one-to-one with FedAvg and FairTrade in fairfl/fairtrade.py.

FedFB (Zeng, Chen & Lee, "Improving Fairness via Federated Learning", 2021;
reference code github.com/yzeng58/Improving-Fairness-via-Federated-Learning,
FedFB/DP_server.py, method FedFB + Client.fb2_update + Client.inference).
Demographic-parity version. The server keeps one weight lambda per
(label, group); clients train on a lambda-weighted loss and report per-
(label, group) loss sums, from which the server takes a signed step on lambda.
Deviations from the reference code, each a fix:
  * aggregation is a proper weighted average (the reference adds client 0's
    weights unweighted);
  * lambda for group z is clipped to [0, 2 n_z / n] (the reference uses group
    0's bound for every group);
  * the constant term of the update is added once per round (the reference
    adds it once per client).

Fed-FUEL (Badar, Younis, Sikdar, Nejdl & Fisichella, DMKD 2025; reference code
github.com/badarm/Fed-FUEL, fed_fuel-main.ipynb + utilities.py).
Pre-processing: each client measures its discrimination score on its own
validation split and, while it exceeds a threshold, rewrites its local data:
synthetic positives for the disadvantaged group plus downsampling of that
group's negatives (or synthetic negatives / downsampled positives for the
advantaged group when the model over-predicts), retraining after each step and
keeping the data with the best balanced-accuracy/fairness trade-off.
Deviations:
  * the discrimination score is measured on the client's validation split
    (the reference uses the client's test data);
  * the SMOTE interpolation weight is drawn from U(0, 1) (the reference's
    random.sample(range(0, 1), 1) always returns 0, so its synthetic points
    are copies of real ones);
  * downsampling removes random members of the cell (the reference picks them
    via nearest neighbours).
"""
import copy

import numpy as np
import torch

from flamby.datasets.fed_heart_disease import Baseline

from fairfl.metrics import all_metrics

FEMALE, MALE = 0, 1


def _weighted_average(states, weights):
    total = float(sum(weights))
    return {k: sum(w * s[k] for s, w in zip(states, weights)) / total for k in states[0]}


def _bce(prob, y):
    return torch.nn.functional.binary_cross_entropy(prob, y, reduction="none")


# ------------------------------------------------------------------ FedFB

def train_fedfb(sites, alpha, lr, init_state, rounds=30, local_steps=10):
    """FedFB (demographic parity) on `sites` = list of (X, y, sex)."""
    y_all = torch.cat([s[1] for s in sites])
    z_all = torch.cat([s[2] for s in sites])
    n = len(y_all)
    m = {(yy, z): int(((y_all == yy) & (z_all == z)).sum()) for yy in (0, 1) for z in (0, 1)}
    n_z = {z: m[(0, z)] + m[(1, z)] for z in (0, 1)}
    lbd = {(yy, z): n_z[z] / n for yy in (0, 1) for z in (0, 1)}

    model = Baseline()
    model.load_state_dict(init_state)
    for r in range(rounds):
        global_state = copy.deepcopy(model.state_dict())
        states, nc = [], []
        for X, y, z in sites:
            v = torch.zeros(len(y))
            for (yy, zz), lam in lbd.items():
                v[(y == yy) & (z == zz)] = lam / n_z[zz]
            local = Baseline()
            local.load_state_dict(global_state)
            opt = torch.optim.Adam(local.parameters(), lr=lr, weight_decay=1e-4)
            for _ in range(local_steps):
                opt.zero_grad()
                (v * _bce(local(X).squeeze(1), y)).sum().backward()
                opt.step()
            states.append(local.state_dict())
            nc.append(float(v.sum()))
        model.load_state_dict(_weighted_average(states, nc))

        # clients report per-(label, group) loss sums under the new global model
        model.eval()
        with torch.no_grad():
            loss_yz = {k: 0.0 for k in lbd}
            for X, y, z in sites:
                l = _bce(model(X).squeeze(1), y)
                for (yy, zz) in loss_yz:
                    loss_yz[(yy, zz)] += float(l[(y == yy) & (z == zz)].sum())
        f = (-loss_yz[(0, 0)] / n_z[0] + loss_yz[(1, 0)] / n_z[0]
             + loss_yz[(0, 1)] / n_z[1] - loss_yz[(1, 1)] / n_z[1]
             + m[(0, 0)] / n_z[0] - m[(0, 1)] / n_z[1])
        step = alpha / (r + 1) ** 0.5 * f
        for z, sign in ((0, -1), (1, +1)):
            hi = 2 * n_z[z] / n
            lbd[(0, z)] = min(max(lbd[(0, z)] + sign * step, 0.0), hi)
            lbd[(1, z)] = hi - lbd[(0, z)]
        model.train()
    return model


# ------------------------------------------------------------------ Fed-FUEL

def _smote(points, n_new, k, gen):
    """n_new synthetic rows by interpolating each sampled row towards its k-NN."""
    if n_new <= 0 or len(points) < 2:
        return points[:0]
    d = torch.cdist(points, points)
    nn = d.argsort(dim=1)[:, 1:k + 1]
    base = torch.randint(len(points), (n_new,), generator=gen)
    nbr = nn[base, torch.randint(nn.shape[1], (n_new,), generator=gen)]
    w = torch.rand(n_new, 1, generator=gen)
    return points[base] + w * (points[nbr] - points[base])


def _rewrite(X, y, z, disadvantaged, over_predicts, lam, k, gen, sex_col):
    """One Fed-FUEL data step on a client's (X, y, z)."""
    if not over_predicts:  # add positives / remove negatives of the disadvantaged group
        g, add_label = disadvantaged, 1.0
    else:                  # add negatives / remove positives of the advantaged group
        g, add_label = 1 - disadvantaged, 0.0
    add_cell = (y == add_label) & (z == g)
    cut_cell = (y == 1 - add_label) & (z == g)
    n_add = int(lam * int(cut_cell.sum()))
    new = _smote(X[add_cell], n_add, k, gen)
    if len(new):
        new[:, sex_col] = X[add_cell][0, sex_col]  # keep the group's (normalized) sex value
    cut_idx = torch.where(cut_cell)[0]
    drop = cut_idx[torch.randperm(len(cut_idx), generator=gen)[:int(lam * len(cut_idx))]]
    keep = torch.ones(len(y), dtype=torch.bool)
    keep[drop] = False
    X2 = torch.cat([X[keep], new])
    y2 = torch.cat([y[keep], torch.full((len(new),), add_label)])
    z2 = torch.cat([z[keep], torch.full((len(new),), g, dtype=z.dtype)])
    return X2, y2, z2


def _class_weighted_fit(model, X, y, lr, steps):
    pos = float(y.sum())
    w = torch.where(y == 1, (len(y) - pos) / max(pos, 1.0), 1.0)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        (w * _bce(model(X).squeeze(1), y)).mean().backward()
        opt.step()


@torch.no_grad()
def _local_scores(model, val, gap):
    X, y, z = val
    model.eval()
    prob = model(X).squeeze(1).numpy()
    model.train()
    m = all_metrics(y.numpy(), prob, z.numpy())
    return m[gap], m["bacc"], int((prob > 0.5).sum()), int(y.sum())


def train_fedfuel(sites, val_sites, lam0, lr, init_state, notion="dp", rounds=30,
                  local_steps=10, thresh=0.005, tol=0.25, k=5, max_iter=5, sex_col=1,
                  seed=0):
    """Fed-FUEL with discrimination measured as SPD (dp) or EOD (eo)."""
    gap = {"dp": "spd", "eo": "eod"}[notion]
    gen = torch.Generator().manual_seed(seed)
    data = [tuple(t.clone() for t in s) for s in sites]  # persistent, rewritten data
    sizes = [len(s[1]) for s in sites]
    model = Baseline()
    model.load_state_dict(init_state)

    def trade_off(disc, bacc):  # (1+e^2) * B * F / (e * B + F), e = 1, F = 1 - |disc|
        f = 1 - abs(disc)
        return 2 * bacc * f / (bacc + f) if bacc + f > 0 else 0.0

    for _ in range(rounds):
        global_state = copy.deepcopy(model.state_dict())
        states = []
        for c, val in enumerate(val_sites):
            local = Baseline()
            local.load_state_dict(global_state)
            X, y, z = data[c]
            _class_weighted_fit(local, X, y, lr, local_steps)
            disc, bacc, assigned, total = _local_scores(local, val, gap)
            if not np.isnan(disc) and not np.isnan(bacc) and abs(disc) > thresh:
                best = (trade_off(disc, bacc), copy.deepcopy(local.state_dict()), data[c])
                prev, worse = best[0], 0
                for _ in range(max_iter):
                    lam = lam0 * (1 + abs(disc) / tol)
                    disadvantaged = FEMALE if disc < 0 else MALE
                    X, y, z = _rewrite(X, y, z, disadvantaged, assigned > total, lam, k, gen,
                                       sex_col)
                    _class_weighted_fit(local, X, y, lr, local_steps)
                    disc, bacc, assigned, total = _local_scores(local, val, gap)
                    if np.isnan(disc) or np.isnan(bacc):
                        break
                    t = trade_off(disc, bacc)
                    if t > best[0]:
                        best = (t, copy.deepcopy(local.state_dict()), (X, y, z))
                    worse = worse + 1 if t < prev else worse
                    prev = t
                    if abs(disc) <= thresh or worse > 2:
                        break
                local.load_state_dict(best[1])
                data[c] = best[2]
            states.append(local.state_dict())
        model.load_state_dict(_weighted_average(states, sizes))
    return model
