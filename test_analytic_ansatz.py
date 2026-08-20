import math

import torch

from analytic_ansatz import (
    SQRT2,
    branch_a_constraints,
    branch_a_full_residual,
    complete_homotopy,
    fusion_reduced,
    rank7_family,
    schoolbook_reduced,
    schoolbook_to_fusion,
)
from rankflow import mm_tensor


def test_schoolbook_and_fusion_are_exact():
    for q in (schoolbook_reduced(), fusion_reduced(0.0)):
        assert float(branch_a_constraints(q).norm()) < 1e-12
        assert float(branch_a_full_residual(q).norm()) < 1e-12


def test_closed_form_rank7_family_is_exact():
    target = mm_tensor(2, "cpu")
    for theta in (0.0, 0.05, 0.1, 0.2):
        for split in (0.0, SQRT2, 2.0 * SQRT2):
            U, V, W, a = rank7_family(theta, split)
            r = torch.einsum("ir,jr,kr,r->ijk", U, V, W, a) - target
            assert float(r.norm()) < 1e-11


def test_projected_schoolbook_to_fusion_path_is_exact_and_smooth():
    qs, rows = schoolbook_to_fusion(31)
    assert max(r["full_tensor_residual"] for r in rows) < 1e-9
    steps = (qs[1:] - qs[:-1]).norm(dim=1)
    assert float(steps.max()) < 0.1


def test_complete_homotopy_ends_with_zero_channel():
    _, rows = complete_homotopy(21, 11)
    assert max(r["full_tensor_residual"] for r in rows) < 1e-9
    assert abs(rows[-1]["x"] - 2.0 * SQRT2) < 1e-12
