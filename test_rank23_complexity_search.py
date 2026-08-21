from pathlib import Path

import torch

from geometry_flow import naive_theta, residual_vector
from rank23_complexity_search import (
    canonical_checkpoint_to_theta,
    hard_support_metrics,
    smooth_support_objective,
)
from rankflow import mm_tensor


def test_schoolbook_2x2_raw_addition_count():
    theta = naive_theta(2, "cpu")
    m = hard_support_metrics(theta, 2, 8, 1e-12)
    assert m["u_support"] == 8
    assert m["v_support"] == 8
    assert m["output_support"] == 8
    assert m["hard_additions"] == 4


def test_smooth_support_is_finite():
    theta = naive_theta(2, "cpu")
    value = smooth_support_objective(theta, 2, 8, 0.05)
    assert torch.isfinite(value)
    assert float(value) > 0


def test_canonical_checkpoint_loader_preserves_tensor(tmp_path: Path):
    theta = naive_theta(2, "cpu")
    # Make an explicit, non-unit multilinear gauge.
    from geometry_flow import unpack
    U, V, W, a = unpack(theta, 2, 8)
    su = torch.linspace(0.5, 1.2, 8)
    sv = torch.linspace(0.7, 1.4, 8)
    sw = torch.linspace(0.8, 1.5, 8)
    Uc = U * su
    Vc = V * sv
    Wc = W * sw
    c = a / (su * sv * sw)
    path = tmp_path / "canonical.pt"
    torch.save({"U": Uc, "V": Vc, "W": Wc, "c": c}, path)
    loaded, rank = canonical_checkpoint_to_theta(path, n=2)
    assert rank == 8
    assert float(residual_vector(loaded, mm_tensor(2), 2, 8).norm()) < 1e-12
