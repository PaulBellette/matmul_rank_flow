import torch

from geometry_flow import jacobian, killability, naive_theta, residual_vector
from rankflow import mm_tensor


def test_schoolbook_geometry():
    torch.set_default_dtype(torch.float64)
    n = 2
    rank = 8
    target = mm_tensor(n)
    theta = naive_theta(n, "cpu")

    assert float(residual_vector(theta, target, n, rank).norm()) < 1.0e-12

    J = jacobian(theta, target, n, rank)
    assert J.shape == (64, 104)

    kills, info = killability(theta, J, n, rank, 1.0e-10)
    assert info.rank == 56
    assert info.nullity == 48
    assert max(row["killability"] for row in kills) < 1.0e-10


def test_robust_svd_falls_back_when_torch_svd_fails(monkeypatch):
    from geometry_flow import robust_svd, svd_info

    # Clustered spectrum with an exact null direction: representative of the
    # controller states where divide-and-conquer SVD can become temperamental.
    A = torch.diag(torch.tensor([3.0, 1.0, 1.0, 1.0e-8, 1.0e-8, 0.0], dtype=torch.float64))
    real_svd = torch.linalg.svd

    def fail_once(*args, **kwargs):
        raise torch._C._LinAlgError("synthetic gesdd convergence failure")

    monkeypatch.setattr(torch.linalg, "svd", fail_once)
    U, S, Vh = robust_svd(A, full_matrices=True)
    recon = U[:, : len(S)] @ torch.diag(S) @ Vh[: len(S), :]
    assert torch.allclose(recon, A, atol=1e-12, rtol=1e-12)
    assert torch.all(S[:-1] >= S[1:])

    _, _, _, info = svd_info(A, rcond=1e-10, full=True)
    assert info.rank == 5
    assert info.nullity == 1
