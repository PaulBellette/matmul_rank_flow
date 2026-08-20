import math

import torch

from analytic_ansatz import SQRT2, branch_a_constraints, branch_a_full_residual
from closed_form_homotopy import (
    S_FUSION,
    collision_metrics,
    factors_from_s,
    full_homotopy,
    reduced_from_collision_p,
    reduced_from_s,
    transfer_from_fusion,
)
from rankflow import mm_tensor


def test_closed_branch_endpoints():
    q0 = reduced_from_collision_p(0.0)
    q1 = reduced_from_collision_p(0.5)
    assert torch.allclose(q0, torch.tensor([1,1,0,0,0,0,0,0,1,0,1,1], dtype=torch.float64))
    assert abs(float(q1[10]) - SQRT2) < 1e-12
    assert abs(float(q1[11]) - 2.0) < 1e-12


def test_closed_branch_is_exact_on_grid():
    for s in torch.linspace(0.0, S_FUSION, 21):
        q = reduced_from_s(float(s))
        assert float(branch_a_constraints(q).norm()) < 2e-12
        assert float(branch_a_full_residual(q).norm()) < 3e-12


def test_collision_coordinate_is_exact():
    for s in (0.0, 0.1, 0.3, 0.5, S_FUSION):
        m = collision_metrics(s)
        assert abs(m["u_cosine"] - 2.0 * s * s) < 1e-12
        assert abs(m["v_cosine"] - 2.0 * s * s) < 1e-12
        assert abs(m["w_cosine"] - 2.0 * s * s) < 1e-12
        assert abs(m["rank1_cosine"] - (2.0 * s * s) ** 3) < 1e-12


def test_channels_zero_and_seven_fuse_at_endpoint():
    U, V, W, a = factors_from_s(S_FUSION)
    assert torch.allclose(U[:, 0], U[:, 7], atol=1e-12, rtol=0)
    assert torch.allclose(V[:, 0], V[:, 7], atol=1e-12, rtol=0)
    assert torch.allclose(W[:, 0], W[:, 7], atol=1e-12, rtol=0)
    assert abs(float(a[0]) - SQRT2) < 1e-12
    assert abs(float(a[7]) - SQRT2) < 1e-12


def test_transfer_is_exact_and_kills_one_channel():
    target = mm_tensor(2, "cpu")
    for t in torch.linspace(0.0, 1.0, 11):
        U, V, W, a = transfer_from_fusion(float(t))
        residual = torch.einsum("ir,jr,kr,r->ijk", U, V, W, a) - target
        assert float(residual.norm()) < 2e-12
    _, _, _, afinal = transfer_from_fusion(1.0)
    assert abs(float(afinal[7])) < 1e-12
    assert abs(float(afinal[0]) - 2.0 * SQRT2) < 1e-12


def test_piecewise_full_homotopy_endpoints_and_continuity():
    points = torch.stack([full_homotopy(float(t)) for t in torch.linspace(0.0, 1.0, 101)])
    steps = (points[1:] - points[:-1]).norm(dim=1)
    assert torch.isfinite(points).all()
    assert float(steps.max()) < 0.5
