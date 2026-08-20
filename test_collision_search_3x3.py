import math

from collision_search_3x3 import (
    all_best_pairs,
    fuse_cube_to_rank_minus_one,
    pair_mask,
    schoolbook_orbit_scan,
)


def test_3x3_orbit_scan_prefers_all_three_indices_different():
    rows, meta = schoolbook_orbit_scan(n=3)
    assert meta["total_pairs"] == 351
    assert meta["physical_tangent_dimension"] == 162
    assert rows[0]["mask"] == "111"
    assert abs(rows[0]["constrained_collision_curvature"] - 1.0 / 3.0) < 1e-10
    assert rows[0]["tangent_gradient_norm"] < 1e-10
    others = [r for r in rows if r["mask"] != "111"]
    assert all(abs(r["constrained_collision_curvature"] - 1.0 / 6.0) < 1e-10 for r in others)


def test_3x3_best_pair_class_has_108_opposite_pairs():
    rows, _ = schoolbook_orbit_scan(n=3)
    pairs = all_best_pairs(3, rows)
    assert len(pairs) == 108
    assert all(pair_mask(pair, 3) == (1, 1, 1) for pair in pairs)


def test_embed_local_strassen_reduces_27_to_26_exactly():
    U, V, W, a, residual, meta = fuse_cube_to_rank_minus_one((0, 13), n=3)
    assert a.numel() == 26
    assert residual < 1e-12
    assert len(meta["removed_schoolbook_channels"]) == 8
