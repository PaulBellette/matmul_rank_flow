from __future__ import annotations

from pathlib import Path
import tempfile

import torch

from exactify_rank23 import (
    canonical_channel_gauge,
    direct_reference_match,
    exact_verify,
    rational_recognise,
    recognise_arrays,
)
from rank23_reference import reference_theta
from rankflow import mm_tensor
from geometry_flow import residual_vector


def test_reference_exact_after_canonical_gauge():
    q = reference_theta()
    U, V, W, c, _ = canonical_channel_gauge(q, 3, 23)
    exprs, ok, total, _ = recognise_arrays([U, V, W, c], mode="rational", max_denominator=64, tol=1e-12)
    assert ok == total
    Ue, Ve, We, ce = exprs
    exact, nonzero, failures = exact_verify(Ue, Ve, We, ce, 3)
    assert exact, failures
    assert nonzero == 0


def test_direct_reference_match_self_is_zero():
    U, V, W, c, _ = canonical_channel_gauge(reference_theta(), 3, 23)
    m = direct_reference_match(U, V, W, c)
    assert m.max_cost < 1e-12


def test_rational_recognise_simple_fraction():
    expr, err, ok = rational_recognise(1.0 / 3.0 + 1e-10, 16, 1e-8)
    assert ok
    assert str(expr) == "1/3"
    assert err < 1e-8


def test_high_precision_repivot_can_replace_singular_square_subset():
    import numpy as np
    from sparse_family_exactify import selected_independent_rows_array

    # Rows 0 and 1 are collinear, so that particular 2x2 subsystem is singular.
    # The full 3x2 Jacobian is nevertheless rank two; QR row pivoting must find
    # a nonsingular pair rather than diagnosing the whole system as singular.
    J = np.array([
        [1.0, 0.0],
        [2.0, 0.0],
        [0.0, 1.0],
    ])
    rows, min_diag, rank = selected_independent_rows_array(J, 2, rcond=1e-12)
    assert rank == 2
    assert min_diag > 0
    assert abs(np.linalg.det(J[rows, :])) > 0.5


def test_high_precision_repivot_reports_true_rank_deficiency():
    import numpy as np
    from sparse_family_exactify import selected_independent_rows_array

    J = np.array([
        [1.0, 0.0],
        [2.0, 0.0],
        [3.0, 0.0],
    ])
    _rows, _min_diag, rank = selected_independent_rows_array(J, 2, rcond=1e-12)
    assert rank == 1
