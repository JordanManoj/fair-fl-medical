"""Re-partition patients into synthetic clients with controlled sex segregation.

Motivated by real hospitals that serve mostly one sex (Long Beach VA is a
veterans' hospital; 1 woman in its fit split). Each patient goes, with
probability 1 - s, to a uniformly random client, and with probability s to a
sex-specific client: women to clients [0, n_women_clients), men to the rest.
  s = 0  every client has the population mix (~25% women)
  s = 1  every client holds a single sex, so a per-client fairness penalty
         has no second group to compare with and is identically zero
"""
import torch

FEMALE = 0


def segregate(sites, n_clients=10, segregation=0.0, n_women_clients=2, seed=0):
    """Pool `sites` (list of (X, y, sex)) and split into `n_clients` clients."""
    assert 0 < n_women_clients < n_clients and 0.0 <= segregation <= 1.0
    X = torch.cat([s[0] for s in sites])
    y = torch.cat([s[1] for s in sites])
    sex = torch.cat([s[2] for s in sites])
    g = torch.Generator().manual_seed(seed)

    n = len(y)
    random_client = torch.randint(0, n_clients, (n,), generator=g)
    women_client = torch.randint(0, n_women_clients, (n,), generator=g)
    men_client = torch.randint(n_women_clients, n_clients, (n,), generator=g)
    sex_client = torch.where(sex == FEMALE, women_client, men_client)
    to_sex_client = torch.rand(n, generator=g) < segregation
    client = torch.where(to_sex_client, sex_client, random_client)

    return [(X[client == k], y[client == k], sex[client == k])
            for k in range(n_clients) if (client == k).any()]
