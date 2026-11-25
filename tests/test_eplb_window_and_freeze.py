# SPDX-License-Identifier: MIT
# Tests for gate-permute EPLB's operating point (the window's decode rows a
# step), which evaluations count as rebalances, its freeze, the window dump
# and the host transport.

import logging
import os
from types import SimpleNamespace

import pytest
from import_guard import skip_if_dependency_missing

torch = pytest.importorskip("torch")

try:
    import atom.config  # noqa: F401
    from atom.model_ops import eplb
    from atom.model_ops import eplb_gate_permute as gp
except ImportError as _e:  # aiter/triton absent under bare non-GPU pytest
    skip_if_dependency_missing(_e, "requires full atom import env")

from test_eplb_gate_permute import (
    EXPERTS,
    LAYERS,
    RANKS,
    TOP_K,
    _Gate,
    _layer,
    _Model,
    _setup,
    _skewed,
    clean_switches,  # noqa: F401  (autouse: every switch unset)
)

INTERVAL = 8


def _manager(monkeypatch, *, rank=0, policy="swap", **env):
    """A gate-permute manager on EP rank ``rank`` of 4 over a 3-layer model,
    its migration stubbed out (the maps and the gate still move)."""
    monkeypatch.setattr(gp, "_HAS_TRITON", False)  # count with torch ops
    monkeypatch.setattr(eplb, "SWAP_EPS", 0.0)  # any gain is a swap
    env = {"ATOM_EPLB_GATE_PERMUTE": 1, **env}
    for name, value in env.items():
        monkeypatch.setenv(name, str(value))
    manager = eplb.EPLBManager(
        enabled=True,
        monitor=eplb.ExpertLoadMonitor(enabled=True, window_size=4),
        rebalance_interval=INTERVAL,
        rebalance_min_balancedness=2.0,
        rebalance_balancedness_agg="min",
        placement_policy=policy,
    )
    meta = eplb.ExpertLocationMetadata.from_trivial(
        num_layers=LAYERS, num_logical_experts=EXPERTS, ep_size=RANKS, ep_rank=rank
    )
    gates = {i: _Gate(seed=i) for i in range(LAYERS)}
    layers = {i: _layer(i, gates[i]) for i in range(LAYERS)}
    manager.live_metadata = meta
    manager._moe_layers = layers
    manager._ep_group = object()
    manager._ep_rank = rank
    manager._dp_is_migration_group = True
    manager.monitor.initialize_for_metadata(meta)
    manager.monitor.set_load_group(SimpleNamespace(world_size=1))
    manager._init_gate_permute(
        _Model([SimpleNamespace(gate=gates[i], experts=layers[i]) for i in layers])
    )
    monkeypatch.setattr(
        eplb, "migrate_experts_chunk", lambda **kw: {kw["layer_ids"][0]: []}
    )
    monkeypatch.setattr(
        eplb,
        "commit_experts_chunk",
        lambda **kw: kw["live_meta"].update(kw["new_meta"], kw["layer_ids"]),
    )
    return manager


def _record_plans(monkeypatch):
    """Every call of the swap policy: (rows a step, budget)."""
    plans = []
    solve = gp.swap_rebalance

    def recording(pf, dec, rows, p2l, **kw):
        plans.append((rows, kw["budget"]))
        return solve(pf, dec, rows, p2l, **kw)

    monkeypatch.setattr(gp, "swap_rebalance", recording)
    return plans


def _step(manager, steps=1, dummy=False):
    for _ in range(steps):
        manager.on_forward_pass_end(True, True, is_dummy_run=dummy)


def _lines(caplog, part):
    return [r.getMessage() for r in caplog.records if part in r.getMessage()]


def _decode(runtime, rows, replays):
    """``replays`` decode replays of ``rows`` live rows on this rank."""
    for _ in range(replays):
        runtime.after_decode_replay("bucket", rows)


# ---------------------------------------------------------------- operating point
def test_sampler_keeps_the_window_beside_the_sums(monkeypatch):
    env = _setup(monkeypatch, ATOM_EPLB_DECODE_SAMPLE_EVERY=2)
    with env.gp.decode_capture("bucket"):
        env.gp.place(env.layers[0], torch.zeros((8, TOP_K), dtype=torch.int32))
    _decode(env.gp, 6, 4)  # two of four sampled
    _, _, cum, steps, window, window_steps = env.gp.dump_global(None)
    assert (cum, steps, window, window_steps) == (6.0, 2, 6.0, 2)
    env.gp.start_window()
    _decode(env.gp, 2, 8)
    _, _, cum, steps, window, window_steps = env.gp.dump_global(None)
    assert (steps, window, window_steps) == (6, 2.0, 4)
    assert cum == pytest.approx((2 * 6 + 4 * 2) / 6)
    env.gp.start_window()
    assert env.gp.dump_global(None)[4:] == (0.0, 0)
    assert (env.gp.dec_rows, env.gp.dec_steps) == (20, 6)


def test_rebalance_is_planned_for_the_window_not_the_mean_since_start(
    monkeypatch, caplog
):
    manager = _manager(monkeypatch)
    plans = _record_plans(monkeypatch)
    gpr = manager.gate_permute
    gpr.pf_total.copy_(_skewed(1))
    gpr.dec_load.copy_(_skewed(2))
    gpr.dec_rows, gpr.dec_steps = 480, 20  # 24 rows a step since the start
    gpr.window_rows, gpr.window_steps = 400, 4  # 100 in this window
    with caplog.at_level(logging.INFO, logger="atom"):
        for _ in manager._rebalance():
            pass
        assert plans[-1][0] == 100.0 and manager._rows_at_last_solve == 100.0
        assert (gpr.window_rows, gpr.window_steps) == (0, 0)
        assert (gpr.dec_rows, gpr.dec_steps) == (480, 20)  # the sums stay
        # a window without a sample keeps the last operating point
        gpr.pf_total[0, 0] += 1
        for _ in manager._rebalance():
            pass
        assert plans[-1][0] == 100.0
        # and the next one with samples is the new operating point
        gpr.pf_total[0, 0] += 1
        gpr.dec_rows, gpr.dec_steps = 480 + 30, 20 + 10
        gpr.window_rows, gpr.window_steps = 30, 10
        for _ in manager._rebalance():
            pass
        assert plans[-1][0] == 3.0 and manager._rows_at_last_solve == 3.0
    first, second, third = _lines(caplog, ": policy=swap")
    assert "rows/step=100.0 cum=24.0 window samples/rank=4" in first
    assert "rows/step=100.0 cum=24.0 window samples/rank=0" in second
    assert "rows/step=3.0 cum=17.0 window samples/rank=10" in third


def test_first_rebalance_without_any_window_sample_uses_what_there_is(monkeypatch):
    manager = _manager(monkeypatch)
    plans = _record_plans(monkeypatch)
    manager.gate_permute.pf_total.copy_(_skewed(1))
    for _ in manager._rebalance():
        pass
    assert plans == [(0.0, 16)] and manager._rows_at_last_solve == 0.0


# ---------------------------------------------------------------- frozen
class _NoTorch:
    def __getattr__(self, name):
        raise AssertionError(f"torch.{name} used while frozen")


def test_frozen_replay_does_nothing_at_all(monkeypatch):
    env = _setup(monkeypatch)
    env.manager.rebalancing_stopped = True
    with monkeypatch.context() as patched:
        patched.setattr(gp, "torch", _NoTorch())
        env.gp.after_decode_replay("bucket", 5)
    assert (env.gp.dec_steps, env.gp.window_steps, env.gp._replays) == (0, 0, 0)


def test_rows_per_step_is_the_groups():
    # eight ranks, each sampled the same 10 steps: 720 rows in 80 rank-steps
    assert gp.rows_per_step(720, 80, 8) == 72.0
    assert gp.rows_per_step(0, 0, 8) == 0.0


def test_stopped_for_good_no_step_reaches_the_schedule(monkeypatch):
    manager = _manager(monkeypatch, ATOM_EPLB_MAX_REBALANCES=1)
    gpr = manager.gate_permute
    gpr.pf_total.copy_(_skewed(1))
    for seed in range(100):
        if not manager.rebalancing_stopped:
            gpr.pf_total.add_(_skewed(seed))
            _step(manager)
    assert manager.rebalancing_stopped
    manager._gen = iter(lambda: pytest.fail("the schedule advanced"), None)
    _step(manager, 20)


# ---------------------------------------------------------------- counting
def _scripted_policy(monkeypatch, script):
    """A swap policy that moves two experts of layer 0 across ranks when the
    next entry of ``script`` is true, and nothing otherwise."""
    calls = []

    def policy(pf, dec, rows, old_p2l, **kw):
        move = script.pop(0)
        calls.append(move)
        p2l = old_p2l.clone()
        if move:
            p2l[0, 0], p2l[0, EXPERTS - 1] = old_p2l[0, EXPERTS - 1], old_p2l[0, 0]
        l2p = torch.empty_like(p2l)
        for layer in range(p2l.shape[0]):
            l2p[layer, p2l[layer].long()] = torch.arange(EXPERTS, dtype=torch.int32)
        return p2l, l2p.unsqueeze(-1), torch.ones_like(p2l), {0: int(move)}

    monkeypatch.setattr(gp, "swap_rebalance", policy)
    return calls


def _evaluate(manager, picks=1):
    """One evaluation with ``picks`` new picks; -> the chunk steps it took."""
    manager.gate_permute.pf_total[0, 0] += picks
    return sum(1 for _ in manager._rebalance())


def test_dummy_and_pure_decode_steps_reach_no_evaluation(monkeypatch):
    manager = _manager(monkeypatch, ATOM_EPLB_MAX_REBALANCES=1)
    calls = _scripted_policy(monkeypatch, [True] * 9)
    manager.gate_permute.pf_total.fill_(5)
    _step(manager, 100, dummy=True)
    for _ in range(100):
        manager.on_forward_pass_end(False, False)
    assert calls == [] and manager._evaluations == 0 and manager.rebalance_count == 0
    assert not manager.rebalancing_stopped and manager._steps == 0


def test_window_without_a_new_pick_is_no_evaluation(monkeypatch, caplog):
    manager = _manager(monkeypatch, ATOM_EPLB_MAX_REBALANCES=2)
    calls = _scripted_policy(monkeypatch, [True] * 9)
    with caplog.at_level(logging.INFO, logger="atom"):
        assert _evaluate(manager) == 1 and manager.rebalance_count == 1
        for _ in range(5):
            assert _evaluate(manager, picks=0) == 0
    assert calls == [True] and manager.rebalance_count == 1
    assert manager._evaluations == 1 and manager._idle_evaluations == 0
    assert not manager._freeze_due()
    skipped = _lines(caplog, "skipped")
    assert len(skipped) == 5 and "no pick since the last evaluation" in skipped[0]
    assert "rebalance #2 skipped" in skipped[0] and "(not counted)" in skipped[0]


def test_empty_load_is_no_evaluation(monkeypatch, caplog):
    manager = _manager(monkeypatch, ATOM_EPLB_MAX_REBALANCES=1)
    calls = _scripted_policy(monkeypatch, [True] * 9)
    with caplog.at_level(logging.INFO, logger="atom"):
        for _ in range(4):
            assert _evaluate(manager, picks=0) == 0
    assert calls == [] and manager.rebalance_count == 0
    assert manager._idle_evaluations == 0 and not manager._freeze_due()
    assert "0 picks < ATOM_EPLB_MIN_PICKS=1" in _lines(caplog, "skipped")[0]


def test_min_picks_skip_is_neither_a_rebalance_nor_idle(monkeypatch):
    manager = _manager(monkeypatch, ATOM_EPLB_MAX_REBALANCES=1, ATOM_EPLB_MIN_PICKS=50)
    calls = _scripted_policy(monkeypatch, [True] * 9)
    for _ in range(4):
        _evaluate(manager, picks=10)  # 10, 20, 30, 40 picks
    assert calls == [] and manager.rebalance_count == 0
    assert manager._evaluations == 0 and manager._idle_evaluations == 0
    assert not manager._freeze_due()
    _evaluate(manager, picks=10)
    assert calls == [True] and manager.rebalance_count == 1 and manager._freeze_due()


def test_evaluation_that_moves_nothing_is_not_counted(monkeypatch):
    manager = _manager(monkeypatch, ATOM_EPLB_MAX_REBALANCES=2)
    monkeypatch.setattr(eplb, "FREEZE_IDLE", 99)  # the idle rule out of the way
    calls = _scripted_policy(monkeypatch, [False] * 6 + [True, False, True])
    for _ in range(6):
        assert _evaluate(manager) == 0  # no chunk step either
    assert manager.rebalance_count == 0 and manager._evaluations == 6
    assert manager._idle_evaluations == 6 and not manager._freeze_due()
    _evaluate(manager)
    assert manager.rebalance_count == 1 and manager._idle_evaluations == 0
    _evaluate(manager)
    assert manager.rebalance_count == 1 and not manager._freeze_due()
    _evaluate(manager)
    assert manager.rebalance_count == 2 and manager._freeze_due()
    assert len(calls) == 9 and manager._experts_moved_total == 4


def test_rebalance_counts_when_its_migration_is_through(monkeypatch):
    manager = _manager(monkeypatch, ATOM_EPLB_MAX_REBALANCES=1)
    _scripted_policy(monkeypatch, [True])
    manager.gate_permute.pf_total[0, 0] += 1
    rebalance = manager._rebalance()
    next(rebalance)  # planned; the chunk is not committed yet
    assert manager._evaluations == 1 and manager.rebalance_count == 0
    assert not manager._freeze_due()
    assert next(rebalance, "done") == "done"
    assert manager.rebalance_count == 1 and manager._freeze_due()


def test_freezes_exactly_after_the_kth_rebalance_that_moved(monkeypatch, caplog):
    manager = _manager(monkeypatch, ATOM_EPLB_MAX_REBALANCES=3)
    monkeypatch.setattr(eplb, "FREEZE_IDLE", 99)  # the idle rule out of the way
    script = [True, False, False, True, False, True, True]
    calls = _scripted_policy(monkeypatch, script)
    counts = []
    with caplog.at_level(logging.INFO, logger="atom"):
        for _ in range(200):
            if manager.rebalancing_stopped:
                break
            manager.gate_permute.pf_total[0, 0] += 1
            _step(manager)
            counts.append(manager.rebalance_count)
    assert manager.rebalancing_stopped and manager.rebalance_count == 3
    assert calls == [True, False, False, True, False, True]  # the 7th never ran
    assert counts[-1] == 3 and counts[-2] == 2  # frozen in the step of the 3rd
    assert manager._evaluations == 6
    (frozen,) = _lines(caplog, "frozen after rebalance")
    assert (
        "frozen after rebalance #6: reason=max executed=3 experts_moved_total=6 "
        in frozen
    )
    # frozen for good: no step reaches the schedule, nothing more is planned
    _step(manager, 5 * INTERVAL)
    assert len(calls) == 6


# ---------------------------------------------------------------- idle freeze
def test_idle_evaluations_in_a_row_freeze_as_converged(monkeypatch, caplog):
    manager = _manager(monkeypatch, ATOM_EPLB_MAX_REBALANCES=5)
    calls = _scripted_policy(monkeypatch, [False, True, False, False, True])
    with caplog.at_level(logging.INFO, logger="atom"):
        for _ in range(200):
            if manager.rebalancing_stopped:
                break
            manager.gate_permute.pf_total[0, 0] += 1
            _step(manager)
    # idle, moved (the run starts again), idle, idle: frozen
    assert calls == [False, True, False, False]
    assert manager.rebalancing_stopped and manager.rebalance_count == 1
    (frozen,) = _lines(caplog, "frozen after rebalance")
    assert "frozen after rebalance #4: reason=converged executed=1 " in frozen
    assert "experts_moved_total=2 " in frozen


def test_idle_rule_needs_a_freeze_to_be_asked_for(monkeypatch):
    # no limit on the rebalances: they go on, idle or not
    manager = _manager(monkeypatch, ATOM_EPLB_MAX_REBALANCES=0)
    calls = _scripted_policy(monkeypatch, [False] * 50)
    for _ in range(12 * INTERVAL):
        manager.gate_permute.pf_total[0, 0] += 1
        _step(manager)
    assert len(calls) >= 10 and not manager.rebalancing_stopped


def test_default_policy_counts_as_before(monkeypatch):
    manager = _manager(
        monkeypatch,
        policy="naive",
        ATOM_EPLB_GATE_PERMUTE=0,
        ATOM_EPLB_MAX_REBALANCES=2,
    )
    assert not manager._places_by_plan()
    manager._execute_runtime_rebalance = lambda: iter(())
    for _ in manager._execute_rebalance():
        pass
    assert manager.rebalance_count == 1 and not manager._freeze_due()
    manager._idle_evaluations = 99  # no idle rule outside the plan modes
    assert not manager._freeze_due()
    for _ in manager._execute_rebalance():
        pass
    assert manager._freeze_due()


# ---------------------------------------------------------------- dump
def test_window_dump_is_written_per_evaluation_and_loads(monkeypatch, tmp_path):
    manager = _manager(
        monkeypatch, ATOM_EPLB_MAX_REBALANCES=0, ATOM_EPLB_DUMP_DIR=tmp_path / "dump"
    )
    _scripted_policy(monkeypatch, [True, False])
    gpr = manager.gate_permute
    gpr.pf_total[1, 2] = 7
    gpr.dec_load[2, 3] = 5
    gpr.window_rows, gpr.window_steps = 80, 4
    before = manager.live_metadata.physical_to_logical_map.clone().long()
    for _ in manager._rebalance():
        pass
    after = manager.live_metadata.physical_to_logical_map.clone().long()
    gpr.pf_total[1, 2] += 3
    for _ in manager._rebalance():
        pass
    files = sorted((tmp_path / "dump").glob("window_*.pt"))
    assert [f.name for f in files] == ["window_00001.pt", "window_00002.pt"]
    first, second = (torch.load(f) for f in files)
    assert (first["kind"], first["seq"], first["executed"]) == ("rebalance", 1, True)
    assert (second["kind"], second["seq"], second["executed"]) == (
        "rebalance",
        2,
        False,
    )
    assert first["rows_per_step_window"] == 20.0 and first["t"] <= second["t"]
    for record in (first, second):
        for key in (
            "prefill_counts_window",
            "decode_counts_window",
            "prefill_counts_cum",
            "decode_counts_cum",
            "p2l_before",
            "p2l_after",
        ):
            assert record[key].shape == (LAYERS, EXPERTS)
            assert record[key].dtype == torch.int64
    assert torch.equal(first["p2l_before"], before)
    assert torch.equal(first["p2l_after"], after) and not torch.equal(before, after)
    assert torch.equal(second["p2l_before"], after)
    assert torch.equal(second["p2l_after"], after)
    # the first window is everything so far, the second what came since
    assert int(first["prefill_counts_window"][1, 2]) == 7
    assert int(second["prefill_counts_window"].sum()) == 3
    assert int(second["prefill_counts_cum"][1, 2]) == 10
    assert int(second["decode_counts_window"].sum()) == 0
    assert int(second["decode_counts_cum"][2, 3]) == 5


def test_no_dump_dir_no_file(monkeypatch, tmp_path):
    manager = _manager(monkeypatch)
    _scripted_policy(monkeypatch, [True])
    manager.gate_permute.pf_total[0, 0] = 9
    for _ in manager._rebalance():
        pass
    assert list(tmp_path.iterdir()) == [] and manager._dumps == 0


def test_frozen_counts_nothing(monkeypatch):
    manager = _manager(monkeypatch, ATOM_EPLB_MAX_REBALANCES=1)
    _scripted_policy(monkeypatch, [True])
    gpr = manager.gate_permute
    while not manager.rebalancing_stopped:
        gpr.pf_total[0, 0] += 1
        _step(manager)
    assert not gpr.counting
    manager.monitor.on_forward_start()
    gpr.place(manager._moe_layers[1], torch.full((4, TOP_K), 2, dtype=torch.int32))
    assert int(manager.monitor._cur_pass_count.sum()) == 0


def test_host_transport_moves_experts_between_ranks_through_files(
    tmp_path, monkeypatch
):
    """ATOM_EPLB_MIGRATE_VIA=shm: `_migrate_single_layer` with a
    `HostExpertTransport` in place of the process group gives every rank's
    temp buffers the bytes of the experts it takes, for whole-layer plans
    with moves inside and across ranks, and removes its files."""
    ranks, per, experts = 4, 4, 16
    gen = torch.Generator().manual_seed(5)
    # one weight tensor in a packed dtype view and one scale tensor, per rank
    full = [
        torch.randint(0, 255, (experts, 6, 8), generator=gen, dtype=torch.uint8),
        torch.randn(experts, 3, generator=gen),
    ]
    old = torch.arange(experts)
    new = torch.tensor([5, 1, 2, 0, 4, 3, 6, 15, 8, 9, 10, 11, 12, 13, 14, 7])
    barriers = []
    monkeypatch.setattr(
        torch.distributed, "barrier", lambda group: barriers.append(group)
    )
    weights = [[t[r * per : (r + 1) * per].clone() for t in full] for r in range(ranks)]
    buffers = [[torch.zeros_like(w) for w in weights[r]] for r in range(ranks)]
    transports = [
        eplb.HostExpertTransport("cpu", str(tmp_path), r) for r in range(ranks)
    ]
    assert eplb.layer_changes_rank(old, new, per)
    assert not eplb.layer_changes_rank(old, torch.tensor([1, 0, *range(2, 16)]), per)

    # the ranks run the layer one after another here; each one's sends are on
    # the host before any reads because the plan is run in two passes
    plans, pending = {}, []
    for r in range(ranks):
        plan, local, sends, recvs = eplb._plan_single_layer_migration(
            old_p2l_layer=old,
            new_p2l_layer=new,
            num_local_physical_experts=per,
            num_gpu_per_node=ranks,
            rank=r,
            world_size=ranks,
        )
        plans[r] = plan
        for action in local:
            for w, b in zip(weights[r], buffers[r]):
                b[action.dst_slot].copy_(w[action.src_slot])
        pending.append((sends, recvs))
    for r, (sends, _) in enumerate(pending):  # every rank writes ...
        transports[r].seq = 1
        for action in sends:
            for part, w in enumerate(weights[r]):
                eplb._as_p2p_bytes(w[action.local_slot]).reshape(-1).numpy().tofile(
                    transports[r]._path(int(action.logical_expert_id), part)
                )
    for r, (_, recvs) in enumerate(pending):  # ... then every rank reads
        transports[r].seq = 0
        transports[r].exchange(weights[r], buffers[r], [], recvs)
    for r in range(ranks):
        for dst_primary, dst in plans[r]:
            for part in range(2):
                logical = int(new[r * per + dst])
                assert torch.equal(buffers[r][part][dst_primary], full[part][logical])
    assert len(barriers) == 2 * ranks

    # one rank end to end through `exchange`: writes once, cleans up
    t = eplb.HostExpertTransport("cpu", str(tmp_path / "solo"), 0)
    send = [type("A", (), {"logical_expert_id": 3, "peer_rank": 1, "local_slot": 3})()]
    seen = []
    monkeypatch.setattr(
        torch.distributed,
        "barrier",
        lambda group: seen.append(sorted(os.listdir(t.directory))),
    )
    t.exchange(weights[0], buffers[0], send + send, [])
    assert seen[0] == ["x1_e3_0.bin", "x1_e3_1.bin"] and os.listdir(t.directory) == []
    assert t.bytes_moved == 6 * 8 + 3 * 4
    assert t.take_stats()[2:] == (1, 0) and t.take_stats() == (0.0, 0.0, 0, 0)


def test_host_transport_stages_through_buffers_it_keeps(tmp_path, monkeypatch):
    """A blob goes device <-> file through one staging buffer of its size:
    made at the first expert, reused by every later one and every later
    exchange, on the sending and on the receiving side."""
    monkeypatch.setattr(torch.distributed, "barrier", lambda group: None)
    gen = torch.Generator().manual_seed(9)
    full = [
        torch.randint(0, 255, (6, 5, 8), generator=gen, dtype=torch.uint8),
        torch.randn(6, 3, generator=gen),
        torch.randn(6, 10, generator=gen),  # the same 40 bytes as part 0
    ]
    sender = eplb.HostExpertTransport("cpu", str(tmp_path), 0)
    taker = eplb.HostExpertTransport("cpu", str(tmp_path), 1)
    buffers = [torch.zeros_like(t[:3]) for t in full]

    def action(logical, slot):
        return type(
            "A", (), {"logical_expert_id": logical, "peer_rank": 0, "local_slot": slot}
        )()

    def move(pairs):
        """Experts `logical` of the sender's slots into the taker's slots."""
        sends = [action(logical, logical) for logical, _ in pairs]
        recvs = [action(logical, slot) for logical, slot in pairs]
        # the sender's files must outlive its own exchange for the taker here
        kept = []
        monkeypatch.setattr(os, "unlink", kept.append)
        sender.exchange(full, [torch.zeros_like(t) for t in full], sends, [])
        taker.seq = sender.seq - 1
        monkeypatch.undo()
        monkeypatch.setattr(torch.distributed, "barrier", lambda group: None)
        taker.exchange([t[:3] for t in full], buffers, [], recvs)
        for path in kept:
            os.unlink(path)
        for logical, slot in pairs:
            for part in range(3):
                assert torch.equal(buffers[part][slot], full[part][logical])

    move([(4, 0), (5, 2)])
    # two blob sizes (40 and 12 bytes), one buffer each, on either side
    assert sorted(sender._stages) == [12, 40] and sender.stages_made == 2
    assert sorted(taker._stages) == [12, 40] and taker.stages_made == 2
    addresses = {
        side: {n: stage.data_ptr() for n, stage in side._stages.items()}
        for side in (sender, taker)
    }
    for round_ in range(3):
        move([(round_, 1), (3, 0), (5, 2)])
    for side in (sender, taker):
        assert side.stages_made == 2
        assert {n: s.data_ptr() for n, s in side._stages.items()} == addresses[side]
    assert os.listdir(tmp_path) == []
    write_s, _read_s, sent, received = sender.take_stats()
    assert sent == 2 + 3 * 3 and received == 0 and write_s > 0
    assert taker.take_stats()[2:] == (0, 2 + 3 * 3)
    assert sender.bytes_moved == (2 + 9) * (40 + 12 + 40)


@pytest.mark.parametrize("size", [0, 39, 41])
def test_host_transport_refuses_a_file_of_another_size(tmp_path, monkeypatch, size):
    monkeypatch.setattr(torch.distributed, "barrier", lambda group: None)
    t = eplb.HostExpertTransport("cpu", str(tmp_path), 0)
    buffers = [torch.zeros(2, 5, 8, dtype=torch.uint8)]
    t.seq = 0
    path = os.path.join(str(tmp_path), "x1_e7_0.bin")
    with open(path, "wb") as f:
        f.write(bytes(range(size)))
    recv = [type("A", (), {"logical_expert_id": 7, "peer_rank": 1, "local_slot": 1})()]
    with pytest.raises(
        RuntimeError, match=f"expert 7 part 0: {size} bytes on the host"
    ):
        t.exchange(buffers, buffers, [], recv)
    assert int(buffers[0].sum()) == 0  # nothing of it reached the temp buffer
