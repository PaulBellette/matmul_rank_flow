import math

import torch

from collision_search import collision_ascent, choose_hidden_pair, scan_schoolbook


def test_scan_finds_exactly_opposite_corner_pairs():
    rows, _ = scan_schoolbook()
    hidden = [r for r in rows if r["tangent_grad_norm"] < 1e-8]
    best = max(r["constrained_collision_curvature"] for r in hidden)
    tied = {
        (r["r"], r["s"])
        for r in hidden
        if best - r["constrained_collision_curvature"] < 1e-10
    }
    assert tied == {(0, 7), (1, 6), (2, 5), (3, 4)}
    assert abs(best - 1.0 / 3.0) < 1e-10


def test_autonomous_search_fuses_to_exact_rank7(tmp_path):
    summary = collision_ascent(seed=0, max_steps=70, out=tmp_path)
    assert tuple(summary["chosen_pair"]) in {(0, 7), (1, 6), (2, 5), (3, 4)}
    assert summary["pre_force_collision_value"] > 0.9999
    assert summary["collision_corrector_ok"]
    assert summary["fused_rank"] == 7
    assert summary["fused_tensor_residual"] < 1e-10
    a, b = summary["exact_pair_amplitudes"]
    assert abs(a - math.sqrt(2.0)) < 1e-5
    assert abs(b - math.sqrt(2.0)) < 1e-5
