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


def dp_penalty(prob, sex):
    """FairTrade's demographic-parity constraint on one client's data.

    sum_g (mean(prob | group g) - mean(prob))^2 over groups present in the
    batch. This equals the original ConstraintLoss with its +/- constraint
    pairs (relu keeps one side of each pair). A group absent from the client
    contributes nothing: there is no local signal about it.
    """
    mu = prob.mean()
    terms = [(prob[sex == g].mean() - mu) ** 2 for g in (FEMALE, MALE) if (sex == g).any()]
    return torch.stack(terms).sum() if terms else prob.new_zeros(())


def train_federated(sites, alpha, lr, init_state, rounds=30, local_steps=10):
    """FedAvg with FairTrade's fairness-regularized local objective.

    sites: list of (X, y, sex) per client (fit split).
    Returns the global model after `rounds` rounds.
    """
    model = Baseline()
    model.load_state_dict(init_state)
    sizes = torch.tensor([len(y) for _, y, _ in sites], dtype=torch.float32)
    weights = sizes / sizes.sum()
    bce = torch.nn.BCELoss()

    for _ in range(rounds):
        global_state = copy.deepcopy(model.state_dict())
        new_state = {k: torch.zeros_like(v) for k, v in global_state.items()}
        for w, (X, y, sex) in zip(weights, sites):
            local = Baseline()
            local.load_state_dict(global_state)
            opt = torch.optim.Adam(local.parameters(), lr=lr)
            for _ in range(local_steps):
                opt.zero_grad()
                prob = local(X).squeeze(1)
                loss = bce(prob, y)
                if alpha > 0:
                    loss = loss + alpha * dp_penalty(prob, sex)
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
REF_POINT = [0.5, -0.5]   # (balanced acc, -|SPD|): chance accuracy, |SPD| = 0.5


def decode(u):
    """Unit-cube point -> (alpha, lr)."""
    a = LOG_ALPHA[0] + u[0] * (LOG_ALPHA[1] - LOG_ALPHA[0])
    b = LOG_LR[0] + u[1] * (LOG_LR[1] - LOG_LR[0])
    return 10 ** a, 10 ** b


def run_fairtrade(data, seed, n_init=8, n_iter=16, rounds=30, local_steps=10):
    """MOBO over (alpha, lr) for one seed.

    Objectives (both maximized) are measured on the validation split:
    balanced accuracy and -|SPD|. Test metrics are recorded for every
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
        model = train_federated(data["fit"], alpha, lr, init_state, rounds, local_steps)
        val, test = evaluate_union(model, data["val"]), evaluate_union(model, data["test"])
        return {"alpha": alpha, "lr": lr,
                **{f"val_{k}": v for k, v in val.items()},
                **{f"test_{k}": v for k, v in test.items()}}

    def objectives(r):
        return [r["val_bacc"], -abs(r["val_spd"])]

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
        r["seed"], r["eval"] = seed, i
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
