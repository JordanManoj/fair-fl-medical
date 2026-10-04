"""FairTrade (Badar et al., AAAI 2024) ported to FLamby hospitals.

FairTrade has two loops:
  inner  federated training where every client minimizes
         BCE + alpha * demographic-parity penalty on its own data,
         aggregated with FedAvg;
  outer  multi-objective Bayesian optimization (MOBO) over (alpha, lr) that
         maximizes balanced accuracy and fairness, returning a Pareto front.

Deliberate differences from the HiWi challenge implementation
(external/HiWi-FairTrade-Challenge/FairTrade.py), each fixing a defect:
  * MOBO objectives are computed on a validation split, never the test set.
  * Each candidate (alpha, lr) trains a fresh federation from the same
    initialization; candidates do not mutate one shared global model.
  * Hypervolume reference point lies below all attainable objectives
    (the original used [0.001, 0.001] with objective -|SPD| <= 0, which made
    every hypervolume improvement zero, so the search was effectively random).
  * Probabilities pass through one sigmoid (the original model ended in
    Sigmoid and was then fed to BCEWithLogitsLoss and to a penalty that
    applied sigmoid again).
  * FedAvg is weighted by client size (the original averaged clients equally).
  * No cost-sensitive positive weight: the heart labels are balanced overall.
Kept from the original: full-batch local Adam steps, the demographic-parity
penalty form, and MOBO over (alpha, lr) with a hypervolume-improvement
acquisition. The model is FLamby's Baseline logistic regression, the same as
every other method here (the original used a 64-32 MLP on Adult).
"""
import copy
import math

import numpy as np
import torch

from flamby.datasets.fed_heart_disease import Baseline

from fairfl.metrics import all_metrics

FEMALE, MALE = 0, 1


NOTIONS = ("dp", "eo")       # demographic parity, equal opportunity
SCOPES = ("local", "global")  # FairTrade's per-client penalty, or ours


def group_stats(prob, y, sex, notion):
    """Per-group (sum of probabilities, count) that the penalty is built from.

    dp: over all patients; eo: over truly sick patients only (y = 1), so the
    group means are soft true-positive rates. Shape (2, 2): rows = groups
    (female, male), columns = (sum, count). These four numbers are all a
    client shares for the global penalty.
    """
    mask = torch.ones_like(y, dtype=torch.bool) if notion == "dp" else y == 1
    rows = []
    for g in (FEMALE, MALE):
        m = mask & (sex == g)
        rows.append(torch.stack([prob[m].sum(), m.sum().to(prob.dtype)]))
    return torch.stack(rows)


def fairness_penalty(prob, y, sex, notion="dp", others=None):
    """FairTrade's constraint: sum_g (m_g - m)^2.

    m_g is group g's mean probability and m the overall mean, over all
    patients (dp) or sick patients (eo). With `others` = None the means use
    this client's data only (FairTrade). With `others` = the other clients'
    group_stats summed, the means are global; only this client's part is
    differentiated, the rest is a constant for the round. A group with no
    members contributes nothing.
    """
    s = group_stats(prob, y, sex, notion)
    if others is not None:
        s = s + others
    present = s[:, 1] > 0
    if not present.any():
        return prob.new_zeros(())
    mean_all = s[present, 0].sum() / s[present, 1].sum()
    means = s[present, 0] / s[present, 1]
    return ((means - mean_all) ** 2).sum()


def dp_penalty(prob, sex):
    """FairTrade's local demographic-parity penalty (the original's form).

    Equals the original ConstraintLoss with its +/- constraint pairs (relu
    keeps one side of each pair); see tests/test_fairtrade.py.
    """
    return fairness_penalty(prob, torch.zeros_like(prob), sex, "dp")


def train_federated(sites, alpha, lr, init_state, rounds=30, local_steps=10,
                    notion="dp", scope="local"):
    """FedAvg with a fairness-regularized local objective.

    sites: list of (X, y, sex) per client (fit split).
    scope="local": each client penalizes unfairness on its own data
        (FairTrade). A client with no women has no fairness signal.
    scope="global": at the start of each round every client reports
        group_stats under the current global model (4 numbers, securely
        aggregatable); each client then penalizes unfairness of the global
        group means, with the other clients' statistics held fixed.
    Returns the global model after `rounds` rounds.
    """
    assert notion in NOTIONS and scope in SCOPES
    model = Baseline()
    model.load_state_dict(init_state)
    sizes = torch.tensor([len(y) for _, y, _ in sites], dtype=torch.float32)
    weights = sizes / sizes.sum()
    bce = torch.nn.BCELoss()

    for _ in range(rounds):
        global_state = copy.deepcopy(model.state_dict())
        others = [None] * len(sites)
        if scope == "global" and alpha > 0:
            with torch.no_grad():
                model.eval()
                stats = [group_stats(model(X).squeeze(1), y, sex, notion) for X, y, sex in sites]
            total = torch.stack(stats).sum(0)
            others = [total - st for st in stats]
        new_state = {k: torch.zeros_like(v) for k, v in global_state.items()}
        for w, (X, y, sex), other in zip(weights, sites, others):
            local = Baseline()
            local.load_state_dict(global_state)
            opt = torch.optim.Adam(local.parameters(), lr=lr)
            for _ in range(local_steps):
                opt.zero_grad()
                prob = local(X).squeeze(1)
                loss = bce(prob, y)
                if alpha > 0:
                    loss = loss + alpha * fairness_penalty(prob, y, sex, notion, other)
                loss.backward()
                opt.step()
            for k, v in local.state_dict().items():
                new_state[k] += w * v
        model.load_state_dict(new_state)
    return model


@torch.no_grad()
def evaluate_union(model, sites):
    """Metrics on the union of the sites' data.

    Every metric here is a ratio of per-site counts (predicted positives per
    group, true positives per group, class counts), so a server can compute
    it from securely aggregated counts without seeing patient rows.
    """
    model.eval()
    X = torch.cat([s[0] for s in sites])
    y = torch.cat([s[1] for s in sites])
    sex = torch.cat([s[2] for s in sites])
    return all_metrics(y.numpy(), model(X).squeeze(1).numpy(), sex.numpy())


# ---------------------------------------------------------------- MOBO outer loop

LOG_ALPHA = (-1.0, 3.5)   # alpha in [0.1, ~3162]
LOG_LR = (-3.0, -0.5)     # lr in [0.001, ~0.32]
REF_POINT = [0.5, -0.5]   # (balanced acc, -|gap|): chance accuracy, |gap| = 0.5
GAP_METRIC = {"dp": "spd", "eo": "eod"}


def decode(u):
    """Unit-cube point -> (alpha, lr)."""
    a = LOG_ALPHA[0] + u[0] * (LOG_ALPHA[1] - LOG_ALPHA[0])
    b = LOG_LR[0] + u[1] * (LOG_LR[1] - LOG_LR[0])
    return 10 ** a, 10 ** b


def run_fairtrade(data, seed, n_init=8, n_iter=16, rounds=30, local_steps=10,
                  notion="dp", scope="local"):
    """MOBO over (alpha, lr) for one seed.

    Objectives (both maximized) are measured on the validation split:
    balanced accuracy and -|soft gap| (see metrics.all_metrics), where the
    gap is SPD (notion "dp") or EOD
    (notion "eo"), matching the penalty. Test metrics are recorded for every
    candidate but never used for any decision.
    Returns a list of dicts, one per evaluated candidate.
    """
    from botorch.acquisition.multi_objective.logei import (
        qLogNoisyExpectedHypervolumeImprovement,
    )
    from botorch.fit import fit_gpytorch_mll
    from botorch.models import ModelListGP, SingleTaskGP
    from botorch.models.transforms.outcome import Standardize
    from botorch.optim import optimize_acqf
    from gpytorch.mlls import SumMarginalLogLikelihood
    from torch.quasirandom import SobolEngine

    torch.manual_seed(seed)
    init_state = Baseline().state_dict()

    def evaluate(u):
        alpha, lr = decode(u.tolist())
        model = train_federated(data["fit"], alpha, lr, init_state, rounds, local_steps,
                                notion, scope)
        val, test = evaluate_union(model, data["val"]), evaluate_union(model, data["test"])
        return {"alpha": alpha, "lr": lr,
                **{f"val_{k}": v for k, v in val.items()},
                **{f"test_{k}": v for k, v in test.items()}}

    def objectives(r):
        return [r["val_bacc"], -abs(r[f"val_{GAP_METRIC[notion]}_soft"])]

    U = SobolEngine(2, scramble=True, seed=seed).draw(n_init).double()
    records = [evaluate(u) for u in U]
    bounds = torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.double)

    for _ in range(n_iter):
        Y = torch.tensor([objectives(r) for r in records], dtype=torch.double)
        Y = torch.nan_to_num(Y, nan=0.0)
        gp = ModelListGP(*[SingleTaskGP(U, Y[:, i:i + 1], outcome_transform=Standardize(1))
                           for i in range(2)])
        fit_gpytorch_mll(SumMarginalLogLikelihood(gp.likelihood, gp))
        acq = qLogNoisyExpectedHypervolumeImprovement(
            model=gp, ref_point=REF_POINT, X_baseline=U, prune_baseline=True)
        u_new, _ = optimize_acqf(acq, bounds=bounds, q=1, num_restarts=8, raw_samples=256)
        U = torch.cat([U, u_new])
        records.append(evaluate(u_new[0]))

    for i, r in enumerate(records):
        r["seed"], r["eval"], r["notion"], r["scope"] = seed, i, notion, scope
    return records


def fedavg_same_trainer(data, seed, lrs=(1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1),
                        rounds=30, local_steps=10):
    """Unconstrained baseline (alpha = 0) through the same trainer.

    The learning rate is chosen by validation balanced accuracy, so the
    baseline gets the same tuning budget logic as FairTrade's lr dimension.
    """
    torch.manual_seed(seed)
    init_state = Baseline().state_dict()
    best = None
    for lr in lrs:
        model = train_federated(data["fit"], 0.0, lr, init_state, rounds, local_steps)
        val = evaluate_union(model, data["val"])
        score = -math.inf if np.isnan(val["bacc"]) else val["bacc"]
        if best is None or score > best[0]:
            best = (score, lr, val, evaluate_union(model, data["test"]))
    _, lr, val, test = best
    return {"seed": seed, "alpha": 0.0, "lr": lr,
            **{f"val_{k}": v for k, v in val.items()},
            **{f"test_{k}": v for k, v in test.items()}}
