import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fairfl.fairtrade import fairness_penalty  # noqa: E402
from fairfl.partition import segregate  # noqa: E402


def _sites():
    torch.manual_seed(0)
    X = torch.randn(200, 3)
    y = torch.randint(0, 2, (200,)).float()
    sex = (torch.rand(200) > 0.25).long()  # ~25% women
    return [(X[:120], y[:120], sex[:120]), (X[120:], y[120:], sex[120:])]


def test_partition_keeps_every_patient_once():
    sites = _sites()
    clients = segregate(sites, n_clients=10, segregation=0.5, seed=1)
    X_all = torch.cat([s[0] for s in sites])
    X_new = torch.cat([c[0] for c in clients])
    assert len(X_new) == len(X_all)
    # same multiset of rows
    key = lambda X: sorted(map(tuple, X.tolist()))  # noqa: E731
    assert key(X_new) == key(X_all)


def test_full_segregation_gives_single_sex_clients_and_zero_local_penalty():
    clients = segregate(_sites(), n_clients=10, segregation=1.0, seed=1)
    for X, y, sex in clients:
        assert len(sex.unique()) == 1
        prob = torch.rand(len(y))
        assert fairness_penalty(prob, y, sex, "dp") == 0
        assert fairness_penalty(prob, y, sex, "eo") == 0


def test_no_segregation_mixes_sexes():
    clients = segregate(_sites(), n_clients=10, segregation=0.0, seed=1)
    assert sum(len(c[2].unique()) == 2 for c in clients) >= 8
