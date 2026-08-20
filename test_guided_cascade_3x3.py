import torch
from guided_cascade_3x3 import checkpoint_report, load_theta, REF
from geometry_flow import residual_vector
from rankflow import mm_tensor


def test_guided_checkpoints_are_finite_low_residual():
    rows = checkpoint_report()
    assert [r[0] for r in rows] == [26, 25, 24, 23]
    limits = {26: 1e-12, 25: 1e-8, 24: 1e-9, 23: 1e-12}
    for rank, residual, amin, amax in rows:
        assert residual < limits[rank]
        assert amax < 10.0
        assert amin > 0.0


def test_rank23_flow_is_exact_numerically():
    q = load_theta(REF / "rank23_flow.pt", 23)
    assert float(residual_vector(q, mm_tensor(3), 3, 23).norm()) < 1e-12
