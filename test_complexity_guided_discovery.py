from pathlib import Path

from autonomous_state_machine_3x3 import (
    ControllerConfig,
    beam_priority_order,
    complexity_policy_weight,
    discovery_complexity_metrics,
    load_theta,
    make_basin_state,
    specialist_expansion_schedule,
)


def test_complexity_policy_weight_modes():
    cfg = ControllerConfig(goal_rank=23, complexity_weight=0.8)
    assert complexity_policy_weight(26, cfg) == 0.0

    cfg.complexity_mode = "weak"
    assert complexity_policy_weight(26, cfg) == 0.8
    assert complexity_policy_weight(24, cfg) == 0.8

    cfg.complexity_mode = "delayed"
    cfg.complexity_delayed_rank = 24
    assert complexity_policy_weight(26, cfg) == 0.0
    assert complexity_policy_weight(25, cfg) == 0.0
    assert complexity_policy_weight(24, cfg) == 0.8

    cfg.complexity_mode = "adaptive"
    assert abs(complexity_policy_weight(26, cfg) - 0.2) < 1e-15
    assert abs(complexity_policy_weight(25, cfg) - 0.4) < 1e-15
    assert abs(complexity_policy_weight(24, cfg) - 0.8) < 1e-15


def test_reference_rank26_complexity_metric_is_sparse_and_finite():
    theta, rank = load_theta(Path("reference_guided_cascade/rank26.pt"))
    cfg = ControllerConfig(complexity_mode="weak")
    smooth, additions = discovery_complexity_metrics(theta, rank, cfg)
    assert smooth > 0
    # This is intentionally a coarse discovery metric; the embedded
    # schoolbook/Strassen rank-26 endpoint is extremely sparse.
    assert additions == 32


def _states_for_policy(mode: str):
    theta, rank = load_theta(Path("reference_guided_cascade/rank26.pt"))
    cfg = ControllerConfig(
        complexity_mode=mode,
        susceptibility_channels=1,
        susceptibility_steps=1,
        beam_width=4,
        beam_expand=3,
        beam_explore_every=3,
    )
    # Re-use the same physical point so the test only checks policy plumbing.
    states = [
        make_basin_state(theta, rank, cfg, basin_id=i, parent_id=-1, generation=0,
                         source=f"s{i}", tau=1e-6, seed=i, probe_susceptibility=False)
        for i in range(4)
    ]
    # Synthetic complexity ordering while all core metrics are tied.
    for i, s in enumerate(states):
        s.complexity_additions = 40 - 5 * i
        s.complexity_smooth = 100.0 - i
    return cfg, states


def test_off_mode_priority_ignores_complexity():
    cfg, states = _states_for_policy("off")
    order = beam_priority_order(states)
    assert [s.basin_id for s in order] == [0, 1, 2, 3]
    assert all(s.complexity_weight == 0.0 for s in states)


def test_guided_mode_uses_existing_explore_slot_for_complexity():
    cfg, states = _states_for_policy("weak")
    # generation 3 is an existing baseline exploration generation.
    sched = specialist_expansion_schedule(states, 3, cfg)
    roles = {s.basin_id: set(rr) for s, rr in sched}
    simplest = min(states, key=lambda s: (s.complexity_additions, s.complexity_smooth)).basin_id
    assert "complexity" in roles.get(simplest, set())
    assert len(sched) <= cfg.beam_width
