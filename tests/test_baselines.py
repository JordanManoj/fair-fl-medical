import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fairfl.baselines import _rewrite, _smote, _weighted_average  # noqa: E402


def test_weighted_average_weights_every_client():
    states = [{"w": torch.tensor([1.0])}, {"w": torch.tensor([4.0])}]
    assert torch.allclose(_weighted_average(states, [3, 1])["w"], torch.tensor([1.75]))


def test_smote_points_lie_between_cell_members():
    gen = torch.Generator().manual_seed(0)
    pts = torch.rand(20, 3)
    new = _smote(pts, 50, k=5, gen=gen)
    assert new.shape == (50, 3)
    assert (new >= pts.min(0).values - 1e-6).all() and (new <= pts.max(0).values + 1e-6).all()


def test_rewrite_adds_disadvantaged_positives_and_drops_its_negatives():
    gen = torch.Generator().manual_seed(0)
    torch.manual_seed(0)
    z = torch.tensor([0] * 10 + [1] * 10)
    y = torch.tensor([1.0] * 4 + [0.0] * 6 + [1.0] * 5 + [0.0] * 5)
    X = torch.rand(20, 3)
    X[:, 1] = z.float()  # sex column
    X2, y2, z2 = _rewrite(X, y, z, disadvantaged=0, over_predicts=False, lam=0.5, k=3,
                          gen=gen, sex_col=1)
    women = z2 == 0
    assert int((women & (y2 == 1)).sum()) == 4 + 3  # + int(0.5 * 6 negatives)
    assert int((women & (y2 == 0)).sum()) == 6 - 3
    assert int((z2 == 1).sum()) == 10                # men untouched
    assert (X2[women, 1] == 0).all()                 # synthetic rows keep the group's sex value
