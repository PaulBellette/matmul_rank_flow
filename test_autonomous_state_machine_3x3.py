from pathlib import Path

import torch

from autonomous_state_machine_3x3 import ControllerConfig, analyse, load_theta
from geometry_flow import residual_vector
from rankflow import mm_tensor

ROOT = Path(__file__).resolve().parent


def test_rank26_checkpoint_is_exact_and_analyzable():
    theta, rank = load_theta(ROOT / "reference_guided_cascade" / "rank26.pt")
    assert rank == 26
    cfg = ControllerConfig(max_cycles=1)
    a = analyse(theta, rank, cfg)
    assert a.residual < 1e-10
    assert 0 <= a.best_channel < rank
    assert a.best_killability > 0.1
    assert a.max_abs_amplitude < cfg.coefficient_cap


def test_checkpoint_length_inference():
    theta, rank = load_theta(ROOT / "reference_guided_cascade" / "rank26.pt")
    assert theta.numel() == 28 * rank
    assert float(residual_vector(theta, mm_tensor(3), 3, rank).norm()) < 1e-10


def test_off_manifold_hop_smoke_and_mixed_csv(tmp_path):
    from autonomous_state_machine_3x3 import off_manifold_basin_hop, write_csv

    theta, rank = load_theta(ROOT / "reference_guided_cascade" / "rank26.pt")
    cfg = ControllerConfig(
        offhop_trials=1,
        offhop_steps_per_stage=2,
        offhop_shell_weights=(1.0,),
        offhop_exact_max_iters=2,
        susceptibility_channels=1,
        susceptibility_steps=2,
    )
    landed, rows = off_manifold_basin_hop(theta, rank, cfg, radius=1.0, seed=17)
    assert rows
    # Smoke only: a two-step tunnel need not find a new basin, but it must be a
    # recoverable search event and its heterogeneous diagnostics must serialize.
    out = tmp_path / "offhop.csv"
    write_csv(out, rows)
    assert out.exists()
    assert "constraint_residual" in out.read_text().splitlines()[0]
    assert landed is None or landed.numel() == theta.numel()


def test_deletion_susceptibility_smoke():
    from autonomous_state_machine_3x3 import deletion_susceptibility

    theta, rank = load_theta(ROOT / "reference_guided_cascade" / "rank26.pt")
    cfg = ControllerConfig(susceptibility_channels=1, susceptibility_steps=2)
    score, channel, rows = deletion_susceptibility(theta, rank, cfg, seed=3)
    assert 0 <= channel < rank
    assert score >= 0.0
    assert rows
    assert all("sus_residual" in row for row in rows)


def test_genericity_spectrum_detects_rank_generic_basin_opening():
    from autonomous_state_machine_3x3 import genericity_spectrum

    home, rank = load_theta(ROOT / "reference_guided_cascade" / "rank26.pt")
    # This checkpoint was produced by an endpoint-free basin hop in the earlier
    # exploratory state-machine probe; it is still rank 26 but substantially
    # more generic than the embedded-Strassen start.
    opened, rank2 = load_theta(ROOT / "reference_autonomous_state_machine_probe" / "probe1" / "cycle_01_hop.pt")
    assert rank2 == rank
    cfg = ControllerConfig(max_cycles=1)
    h = genericity_spectrum(home, rank, cfg)
    c = genericity_spectrum(opened, rank, cfg, tau=h.tau)
    assert h.hard_nullity > c.hard_nullity
    assert h.soft_nullity > c.soft_nullity + 10.0


def test_generic_basin_score_is_rank_agnostic_and_prefers_genericisation():
    from autonomous_state_machine_3x3 import generic_basin_score

    cfg = ControllerConfig()
    same = generic_basin_score(
        candidate_soft_nullity=54.0,
        home_soft_nullity=54.0,
        susceptibility=1.0,
        death_distance=2.0,
        max_abs_amplitude=3.0,
        cfg=cfg,
    )
    opened = generic_basin_score(
        candidate_soft_nullity=45.0,
        home_soft_nullity=54.0,
        susceptibility=1.0,
        death_distance=2.0,
        max_abs_amplitude=3.0,
        cfg=cfg,
    )
    nearly_deleted = generic_basin_score(
        candidate_soft_nullity=54.0,
        home_soft_nullity=54.0,
        susceptibility=1.0e-3,
        death_distance=0.1,
        max_abs_amplitude=3.0,
        cfg=cfg,
    )
    assert opened < same
    # A genuinely nearby lower-rank basin is allowed to beat genericisation;
    # this is how the controller switches modes without a hard-coded phase.
    assert nearly_deleted < same



def _dummy_basin(basin_id, soft, sus, death, amp):
    from autonomous_state_machine_3x3 import Analysis, BasinState
    theta = torch.tensor([float(basin_id)])
    analysis = Analysis(
        rank=1,
        residual=0.0,
        jacobian_rank=0,
        tangent_dim=int(round(soft)),
        sigma_min_positive=1.0,
        condition_positive=1.0,
        min_abs_amplitude=amp,
        max_abs_amplitude=amp,
        best_channel=0,
        best_amplitude=amp,
        best_killability=max(amp / max(death, 1e-9), 1e-9),
        best_death_distance=death,
    )
    return BasinState(
        basin_id=basin_id,
        theta=theta,
        rank=1,
        parent_id=-1,
        generation=0,
        source="test",
        analysis=analysis,
        soft_nullity=soft,
        susceptibility=sus,
        susceptibility_channel=0,
        max_abs_amplitude=amp,
        fingerprint=torch.tensor([float(basin_id)]),
    )


def test_pareto_beam_retains_metric_specialists():
    from autonomous_state_machine_3x3 import pareto_beam_prune

    states = [
        _dummy_basin(0, 40.0, 0.9, 2.0, 3.0),   # best genericity
        _dummy_basin(1, 52.0, 0.02, 2.2, 3.0),  # best susceptibility
        _dummy_basin(2, 50.0, 0.8, 0.5, 3.0),   # best death distance
        _dummy_basin(3, 51.0, 0.7, 1.8, 1.5),   # best coefficient scale
        _dummy_basin(4, 55.0, 1.0, 3.0, 4.0),   # dominated junk
    ]
    beam = pareto_beam_prune(states, width=4)
    ids = {s.basin_id for s in beam}
    assert ids == {0, 1, 2, 3}


def test_specialist_schedule_expands_metric_champions_and_explorer():
    from autonomous_state_machine_3x3 import ControllerConfig, specialist_expansion_schedule

    states = [
        _dummy_basin(0, 40.0, 0.9, 2.0, 3.0),   # genericity
        _dummy_basin(1, 52.0, 0.02, 2.2, 3.0),  # deletion
        _dummy_basin(2, 50.0, 0.8, 0.5, 3.0),   # death
        _dummy_basin(3, 51.0, 0.7, 1.8, 1.5),   # explorer
    ]
    states[0].expansion_count = 4
    states[1].expansion_count = 3
    states[2].expansion_count = 2
    states[3].expansion_count = 0
    states[3].generation = 3

    cfg = ControllerConfig(beam_width=4, beam_expand=3, beam_explore_every=3)
    schedule = specialist_expansion_schedule(states, generation=3, cfg=cfg)
    roles = {s.basin_id: set(r) for s, r in schedule}
    assert roles[0] == {"genericity"}
    assert roles[1] == {"deletion"}
    assert roles[2] == {"death"}
    assert roles[3] == {"explore"}


def test_specialist_schedule_merges_roles_and_fills_spare_slot():
    from autonomous_state_machine_3x3 import ControllerConfig, specialist_expansion_schedule

    # id0 owns genericity and deletion; the freed specialist slot should be
    # allocated to a lightly explored state instead of wasting the budget.
    states = [
        _dummy_basin(0, 40.0, 0.02, 2.0, 3.0),
        _dummy_basin(1, 50.0, 0.8, 0.5, 3.0),
        _dummy_basin(2, 51.0, 0.7, 1.8, 1.5),
        _dummy_basin(3, 52.0, 0.9, 1.9, 1.6),
    ]
    states[2].expansion_count = 0
    states[3].expansion_count = 5
    cfg = ControllerConfig(beam_width=4, beam_expand=3, beam_explore_every=999)
    schedule = specialist_expansion_schedule(states, generation=1, cfg=cfg)
    roles = {s.basin_id: set(r) for s, r in schedule}
    assert roles[0] == {"genericity", "deletion"}
    assert roles[1] == {"death"}
    assert roles[2] == {"explore"}
    assert len(schedule) == 3
