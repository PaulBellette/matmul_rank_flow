import torch

from curvature_flow import (
    amplitude_curvature_operator,
    constraint_jacobian,
    curvature_escape,
    tangent_basis,
)
from geometry_flow import naive_theta
from rankflow import mm_tensor


def test_schoolbook_physical_tangent_and_curvature():
    torch.set_default_dtype(torch.float64)
    n = 2
    rank = 8
    target = mm_tensor(n)
    theta = naive_theta(n, "cpu")

    J = constraint_jacobian(theta, target, n, rank)
    N, info = tangent_basis(J, 1.0e-10)
    assert J.shape == (88, 104)
    assert info.rank == 80
    assert N.shape == (104, 24)

    for ch in range(rank):
        _, evals, _, _, _, _ = amplitude_curvature_operator(
            theta, target, n, rank, ch, rcond=1.0e-10
        )
        assert int((evals > 1.0e-8).sum()) == 3
        assert int((evals < -1.0e-8).sum()) == 0
        positive = evals[evals > 1.0e-8]
        assert torch.allclose(positive, torch.ones(3), atol=1.0e-8, rtol=1.0e-8)


def test_curvature_escape_creates_killability():
    torch.set_default_dtype(torch.float64)
    n = 2
    rank = 8
    target = mm_tensor(n)
    theta = naive_theta(n, "cpu")
    _, row = curvature_escape(
        theta,
        target,
        n,
        rank,
        channel=0,
        size=0.4,
        rcond=1.0e-10,
        tol=1.0e-10,
    )
    assert row["converged"]
    assert row["constraint_residual"] < 1.0e-9
    assert row["amplitude"] > 1.05
    assert row["killability"] > 0.30
