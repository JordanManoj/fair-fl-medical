import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from fairfl.fairtrade import dp_penalty  # noqa: E402
from fairfl.metrics import all_metrics  # noqa: E402
from run_fairtrade import hypervolume_2d, pareto_mask  # noqa: E402

ORIGINAL = ROOT / "external" / "HiWi-FairTrade-Challenge"


@pytest.mark.skipif(not (ORIGINAL / "constraint.py").exists(),
                    reason="original FairTrade repo not cloned")
def test_dp_penalty_matches_original_constraint_loss():
    sys.path.insert(0, str(ORIGINAL))
    from constraint import DemographicParityLoss

    torch.manual_seed(0)
    logits = torch.randn(64)
    sex = torch.randint(0, 2, (64,)).float()
    # the original applies a sigmoid inside the loss, so it receives logits
    original = DemographicParityLoss(alpha=1.0)(None, logits, sex)
    ours = dp_penalty(torch.sigmoid(logits), sex.long())
    assert torch.allclose(original, ours, atol=1e-6)


def test_dp_penalty_ignores_absent_group():
    prob = torch.tensor([0.2, 0.9, 0.4])
    assert dp_penalty(prob, torch.tensor([1, 1, 1])) == 0
    assert dp_penalty(prob, torch.tensor([0, 1, 1])) > 0


def test_spd_undefined_without_women_is_nan():
    m = all_metrics(np.array([1, 0, 1]), np.array([0.9, 0.1, 0.8]), np.array([1, 1, 1]))
    assert np.isnan(m["spd"]) and np.isnan(m["eod"])


def test_pareto_mask_and_hypervolume():
    pts = np.array([[0.8, -0.3], [0.7, -0.1], [0.6, -0.2], [0.75, -0.05]])
    assert pareto_mask(pts).tolist() == [True, False, False, True]
    # ref (0.5, -0.5): (0.8-0.5)*(0.2) + (0.75-0.5)*(0.25) = 0.06 + 0.0625
    assert hypervolume_2d(pts[pareto_mask(pts)]) == pytest.approx(0.1225)
    assert hypervolume_2d(pts) == pytest.approx(0.1225)  # dominated points add nothing


def test_global_penalty_equals_pooled_penalty():
    from fairfl.fairtrade import fairness_penalty, group_stats

    torch.manual_seed(1)
    prob, y = torch.rand(40), torch.randint(0, 2, (40,)).float()
    sex = torch.randint(0, 2, (40,))
    for notion in ("dp", "eo"):
        pooled = fairness_penalty(prob, y, sex, notion)
        # client = first 10 rows, others = the remaining 30 rows' statistics
        others = group_stats(prob[10:], y[10:], sex[10:], notion)
        split = fairness_penalty(prob[:10], y[:10], sex[:10], notion, others)
        assert torch.allclose(pooled, split, atol=1e-6)


def test_global_penalty_gives_signal_to_client_without_women():
    from fairfl.fairtrade import fairness_penalty, group_stats

    prob = torch.tensor([0.9, 0.8, 0.7], requires_grad=True)
    y, sex = torch.ones(3), torch.ones(3, dtype=torch.long)  # men only
    assert fairness_penalty(prob, y, sex, "eo") == 0  # local: no signal
    women_elsewhere = group_stats(torch.tensor([0.2, 0.3]), torch.ones(2),
                                  torch.zeros(2, dtype=torch.long), "eo")
    pen = fairness_penalty(prob, y, sex, "eo", women_elsewhere)
    pen.backward()
    assert pen > 0 and prob.grad.abs().sum() > 0


def test_eo_ignores_healthy_patients():
    from fairfl.fairtrade import fairness_penalty

    prob = torch.tensor([0.9, 0.1, 0.5, 0.5])
    y = torch.tensor([0.0, 0.0, 1.0, 1.0])
    sex = torch.tensor([0, 1, 0, 1])
    assert fairness_penalty(prob, y, sex, "eo") == 0  # equal TPR proxy among sick
    assert fairness_penalty(prob, y, sex, "dp") > 0
