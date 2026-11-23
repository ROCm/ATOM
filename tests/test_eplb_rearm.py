# SPDX-License-Identifier: MIT
# Tests for gate-permute EPLB's operating point (the window's decode rows a
# step), its frozen state and the drift watch that re-arms the rebalances
# (ATOM_EPLB_REARM_DRIFT), and the one-off swap budgets.

import logging
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

from test_eplb_gate_permute import (  # noqa: E402
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

INTERVAL, CHECK = 8, 3


def _manager(monkeypatch, *, rank=0, policy="swap", **env):
    """A gate-permute manager on EP rank ``rank`` of 4 over a 3-layer model,
    its migration stubbed out (the maps and the gate still move)."""
    env = {
        "ATOM_EPLB_GP_COUNT": "torch",
        "ATOM_EPLB_GATE_PERMUTE": 1,
        "ATOM_EPLB_SWAP_EPS": 0,
        "ATOM_EPLB_REARM_CHECK_STEPS": CHECK,
        **env,
    }
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


def _step_until(manager, caplog, part, limit=200):
    """Steps of the schedule until one more log line holding ``part``."""
    seen = len(_lines(caplog, part))
    for taken in range(1, limit + 1):
        _step(manager)
        if len(_lines(caplog, part)) > seen:
            return taken
    raise AssertionError(f"no {part!r} within {limit} steps")


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


def test_frozen_replay_adds_two_host_integers_and_nothing_else(monkeypatch):
    env = _setup(monkeypatch, ATOM_EPLB_DECODE_SAMPLE_EVERY=1)
    held = torch.full((8, TOP_K), 3, dtype=torch.int32)
    with env.gp.decode_capture("bucket"):
        env.gp.place(env.layers[0], held)
    env.manager.rebalancing_stopped = True
    env.gp.watching = True
    before = env.gp.dec_load.clone()
    with monkeypatch.context() as patched:
        patched.setattr(gp, "torch", _NoTorch())  # any torch call would raise
        patched.setattr(
            env.gp, "count_decode", lambda *a: pytest.fail("sampled while frozen")
        )
        for rows in (5, 0, 7):
            for _ in range(100):
                env.gp.after_decode_replay("bucket", rows)
        env.gp.after_decode_replay("another bucket", 4)
    assert (env.gp.watch_rows, env.gp.watch_steps) == (1204, 301)
    assert type(env.gp.watch_rows) is int and type(env.gp.watch_steps) is int
    assert torch.equal(env.gp.dec_load, before)
    assert (env.gp.dec_steps, env.gp.window_steps, env.gp._replays) == (0, 0, 0)


def test_frozen_without_a_watch_does_nothing_at_all(monkeypatch):
    env = _setup(monkeypatch)
    env.manager.rebalancing_stopped = True
    with monkeypatch.context() as patched:
        patched.setattr(gp, "torch", _NoTorch())
        env.gp.after_decode_replay("bucket", 5)
    assert (env.gp.watch_rows, env.gp.watch_steps, env.gp.dec_steps) == (0, 0, 0)


# ---------------------------------------------------------------- watch
@pytest.mark.parametrize(
    "now,last,outside",
    [
        (100.0, 100.0, False),
        (130.0, 100.0, False),  # the band's edges are inside
        (130.1, 100.0, True),
        (100.0, 130.0, False),
        (100.0, 130.1, True),
        (77.0, 100.0, False),
        (76.9, 100.0, True),
        (0.0, 100.0, False),  # no decode load now: nothing to place for
        (5.0, 0.0, True),  # none at the last solve, some now
        (0.0, 0.0, False),
    ],
)
def test_drift_band(now, last, outside):
    assert gp.drifted(now, last, 0.3) is outside


def test_rows_per_step_is_the_groups():
    # eight ranks, each sampled the same 10 steps: 720 rows in 80 rank-steps
    assert gp.rows_per_step(720, 80, 8) == 72.0
    assert gp.rows_per_step(0, 0, 8) == 0.0


def _freeze(manager, caplog, rows=100):
    """Run the schedule to the frozen state, the rebalances planned for
    ``rows`` decode rows a step."""
    gpr = manager.gate_permute
    gpr.dec_load.copy_(_skewed(2))
    for seed in range(1000):
        if manager.rebalancing_stopped:
            break
        _traffic(gpr, rows, seed)
        _step(manager)
    assert manager.rebalancing_stopped and manager._rows_at_last_solve == rows
    return gpr


def _traffic(gpr, rows, seed):
    """Fresh picks (so every evaluation has something to move) and a window
    of ``rows`` decode rows a step."""
    gpr.pf_total.add_(_skewed(100 + seed))
    gpr.window_rows, gpr.window_steps = rows * 4, 4
    gpr.dec_rows += rows * 4
    gpr.dec_steps += 4


def _check(manager, caplog, rows, replays=50):
    """One watch check over ``replays`` replays of ``rows`` live rows."""
    _decode(manager.gate_permute, rows, replays)
    _step_until(manager, caplog, "watch:")
    return _lines(caplog, "watch:")[-1]


def test_watch_rearms_on_two_checks_in_a_row_only(monkeypatch, caplog):
    manager = _manager(
        monkeypatch, ATOM_EPLB_MAX_REBALANCES=2, ATOM_EPLB_REARM_DRIFT=0.3
    )
    plans = _record_plans(monkeypatch)
    with caplog.at_level(logging.INFO, logger="atom"):
        gpr = _freeze(manager, caplog)
        assert manager.rebalance_count == 2 and len(plans) == 2
        (frozen,) = _lines(caplog, "frozen after")
        assert frozen.startswith(gp.TAG)
        assert "frozen after rebalance #2: reason=max executed=2 " in frozen
        assert "rows/step=100.0 (window) cum=100.0" in frozen
        assert "experts_moved_total=" in frozen and "t_since_first_step=" in frozen
        assert len(_lines(caplog, "EPLB stopped after 2 rebalances")) == 1

        # inside the band: nothing
        line = _check(manager, caplog, 120)
        assert "rows/step now 120.0 (last solve 100.0, drift +20%) ok" in line
        # one outlier, then back: a burst does not trigger
        assert _check(manager, caplog, 300).endswith("drift +200%) 1/2")
        assert _check(manager, caplog, 90).endswith("drift -10%) ok")
        assert _check(manager, caplog, 20).endswith("drift -80%) 1/2")
        # a check without a decode step neither counts nor clears
        _step_until(manager, caplog, "watch:")
        assert _lines(caplog, "watch:")[-1].endswith("no decode step")
        assert manager.rebalancing_stopped and gpr.watching
        assert manager.rebalance_count == 2 and len(plans) == 2
        # the second in a row re-arms
        assert _check(manager, caplog, 300).endswith("drift +200%) re-arm")
        (rearmed,) = _lines(caplog, "re-armed #1")
        assert "rows/step now 300.0, last solve 100.0" in rearmed
        assert "frozen again after 2 rebalances" in rearmed
        assert "budget of 16 swaps" in rearmed
    assert not manager.rebalancing_stopped and not gpr.watching
    assert manager.rebalance_count == 0 and manager._limit == 2
    assert manager._budget_once == ("re-arm", 16)  # the normal one by default
    assert (gpr.window_rows, gpr.window_steps) == (0, 0)
    assert (gpr.watch_rows, gpr.watch_steps) == (0, 0)
    assert len(plans) == 2  # no rebalance yet: the new window is empty
    assert int(gpr.pf_total.sum()) > 0  # the sums are kept


def test_checks_come_every_check_steps_of_real_prefill_steps(monkeypatch, caplog):
    manager = _manager(
        monkeypatch, ATOM_EPLB_MAX_REBALANCES=1, ATOM_EPLB_REARM_DRIFT=0.3
    )
    with caplog.at_level(logging.INFO, logger="atom"):
        gpr = _freeze(manager, caplog)
        _step(manager, 50, dummy=True)  # warm-up forwards are no steps
        manager.on_forward_pass_end(False, False)  # nor is a pure-decode step
        assert _lines(caplog, "watch:") == []
        _decode(gpr, 100, 5)
        assert _step_until(manager, caplog, "watch:") == CHECK
        _decode(gpr, 100, 5)
        assert _step_until(manager, caplog, "watch:") == CHECK


def test_rearmed_rebalance_waits_for_samples_and_takes_the_rearm_budget_once(
    monkeypatch, caplog
):
    manager = _manager(
        monkeypatch,
        ATOM_EPLB_MAX_REBALANCES=2,
        ATOM_EPLB_REARM_DRIFT=0.3,
        ATOM_EPLB_REARM_MIN_SAMPLES=6,
        ATOM_EPLB_FIRST_BUDGET=5,
        ATOM_EPLB_SWAP_BUDGET=2,
        ATOM_EPLB_REARM_BUDGET=0,
    )
    plans = _record_plans(monkeypatch)
    with caplog.at_level(logging.INFO, logger="atom"):
        gpr = _freeze(manager, caplog)
        assert plans == [(100.0, 5), (100.0, 2)]  # the first, then the normal
        _check(manager, caplog, 300)
        _check(manager, caplog, 300)
        assert not manager.rebalancing_stopped
        # the new window: 5 samples a rank at the next check is not enough
        gpr.window_rows, gpr.window_steps = 5 * 300, 5
        _step(manager, CHECK)
        assert len(plans) == 2
        # 6 are: the rebalance runs in that check step, at the new point
        gpr.pf_total.add_(_skewed(500))
        gpr.window_rows, gpr.window_steps = 6 * 300, 6
        _step(manager, CHECK - 1)
        assert len(plans) == 2
        _step(manager)
        assert plans[2] == (300.0, None)  # unlimited, once
        assert manager._budget_once is None
        assert manager._rows_at_last_solve == 300.0
        # the next one after the interval, with the normal budget; then frozen
        # again: a re-arm's round is ATOM_EPLB_REARM_REBALANCES (2) long
        gpr.pf_total.add_(_skewed(501))
        gpr.window_rows, gpr.window_steps = 4 * 300, 4
        for _ in range(INTERVAL + 3):
            if not manager.rebalancing_stopped:
                _step(manager)
        assert plans[3] == (300.0, 2) and len(plans) == 4
        assert manager.rebalancing_stopped and gpr.watching
    budgets = [
        line.split("budget=")[1].split(" ")[0]
        for line in _lines(caplog, ": policy=swap")
    ]
    assert budgets == ["5(first)", "2(normal)", "unlimited(re-arm)", "2(normal)"]
    frozen = _lines(caplog, "frozen after")
    assert len(frozen) == 2 and "rows/step=300.0 (window)" in frozen[1]
    assert len(_lines(caplog, "EPLB stopped after")) == 2


def test_rearm_budget_and_first_budget_values(monkeypatch):
    manager = _manager(monkeypatch, ATOM_EPLB_FIRST_BUDGET=0)
    assert manager._budget_once == ("first", None)  # 0: no bound
    assert _manager(monkeypatch, ATOM_EPLB_FIRST_BUDGET=40)._budget_once == (
        "first",
        40,
    )
    monkeypatch.delenv("ATOM_EPLB_FIRST_BUDGET")
    assert _manager(monkeypatch)._budget_once is None  # unset: the normal one


def test_skipped_rebalance_keeps_the_one_off_budget(monkeypatch):
    manager = _manager(monkeypatch, ATOM_EPLB_FIRST_BUDGET=7, ATOM_EPLB_MIN_PICKS=10**9)
    plans = _record_plans(monkeypatch)
    manager.gate_permute.pf_total.copy_(_skewed(1))
    manager.gate_permute.window_rows, manager.gate_permute.window_steps = 40, 4
    for _ in manager._rebalance():
        pass
    assert plans == [] and manager._budget_once == ("first", 7)
    assert manager._rows_at_last_solve is None
    assert manager.gate_permute.window_steps == 4  # the window goes on


def test_fatal_stop_ends_the_watch(monkeypatch, caplog):
    manager = _manager(
        monkeypatch, ATOM_EPLB_MAX_REBALANCES=1, ATOM_EPLB_REARM_DRIFT=0.3
    )
    with caplog.at_level(logging.INFO, logger="atom"):
        gpr = _freeze(manager, caplog)
        manager._fatal_stop("test")
        assert not gpr.watching and manager.rebalancing_stopped
        _decode(gpr, 300, 10)
        _step(manager, 10 * CHECK)
    assert _lines(caplog, "watch:") == [] and (gpr.watch_rows, gpr.watch_steps) == (
        0,
        0,
    )


# ---------------------------------------------------------------- group
def test_every_rank_takes_the_same_decision_in_the_same_step(monkeypatch, caplog):
    world = 8
    managers = [
        _manager(
            monkeypatch,
            rank=r % RANKS,
            ATOM_EPLB_MAX_REBALANCES=1,
            ATOM_EPLB_REARM_DRIFT=0.3,
            ATOM_EPLB_REARM_MIN_SAMPLES=4,
        )
        for r in range(world)
    ]
    for r, manager in enumerate(managers):
        manager._ep_rank = r
    runtimes = [m.gate_permute for m in managers]
    snapshot = {}

    def group_sum(rank, rows, steps):
        # What the collective returns: the sums of what every rank held when
        # the step began. Each rank must hand in exactly its own part.
        kind = "watch" if runtimes[rank].watching else "window"
        assert (rows, steps) == snapshot[kind][rank]
        parts = snapshot[kind]
        return sum(p[0] for p in parts), sum(p[1] for p in parts), world

    def group_dump(local):
        # `dump_global` with the operating points reduced over the group.
        dec, pf, *_ = local(None)
        rows = {
            kind: (sum(p[0] for p in snapshot[kind]), sum(p[1] for p in snapshot[kind]))
            for kind in ("sums", "window")
        }
        return (
            dec,
            pf,
            gp.rows_per_step(*rows["sums"], world),
            rows["sums"][1] // world,
            gp.rows_per_step(*rows["window"], world),
            rows["window"][1] // world,
        )

    for r, runtime in enumerate(runtimes):
        monkeypatch.setattr(
            runtime,
            "reduce_counts",
            lambda group, rows, steps, r=r: group_sum(r, rows, steps),
        )
        monkeypatch.setattr(
            runtime,
            "dump_global",
            lambda group, local=runtime.dump_global: group_dump(local),
        )
        runtime.pf_total.copy_(_skewed(1))
        runtime.dec_load.copy_(_skewed(2))
    plans = _record_plans(monkeypatch)

    def state(m):
        g = m.gate_permute
        return (
            m.rebalancing_stopped,
            g.watching,
            m.rebalance_count,
            m._budget_once,
            m._rearms,
            m._rows_at_last_solve,
        )

    def step():
        snapshot["watch"] = [(g.watch_rows, g.watch_steps) for g in runtimes]
        snapshot["window"] = [(g.window_rows, g.window_steps) for g in runtimes]
        snapshot["sums"] = [(g.dec_rows, g.dec_steps) for g in runtimes]
        for m in managers:
            _step(m)
        states = {state(m) for m in managers}
        assert len(states) == 1, states  # after every single step
        return states.pop()

    # The window the first rebalance sees is the same everywhere (the sums are
    # the group's in a real run); the rows each rank decodes are not.
    for g in runtimes:
        g.window_rows, g.window_steps = 400, 4
        g.dec_rows, g.dec_steps = 400, 4
    while not step()[0]:
        pass
    assert all(m._rows_at_last_solve == 800.0 for m in managers)  # 8 x 100
    solved = len(plans)

    def decode(rows_of_rank, replays=10):
        for r, g in enumerate(runtimes):
            _decode(g, rows_of_rank(r), replays)

    with caplog.at_level(logging.INFO, logger="atom"):
        # 800 rows a step for the group, unevenly spread over the ranks
        for _ in range(2 * CHECK):
            decode(lambda r: 200 if r % 2 else 0)
            assert step()[:2] == (True, True)
        # the group's rows rise to 2350 a step (rank 7 alone has most of them)
        rearmed_at = None
        for i in range(4 * CHECK):
            decode(lambda r: 2000 if r == 7 else 50)
            if not step()[0]:
                rearmed_at = i
                break
        assert rearmed_at == 2 * CHECK - 1  # the second check, on every rank
        assert state(managers[0])[2:5] == (0, ("re-arm", 16), 1)
        assert len(plans) == solved
        # sampling resumed everywhere; rank 0 has no decode step of its own,
        # the group has 4 samples a rank only once the others bring theirs
        for i in range(3 * CHECK):
            for r, g in enumerate(runtimes):
                g.pf_total[0, 0] += 1
                if r:
                    g.window_rows += 80 * (i % 2)
                    g.window_steps += 2 * (i % 2)
            step()
            if len(plans) > solved:
                break
    assert len(plans) == solved + world  # every rank planned, in that step
    # ... for the group's new operating point, with the re-arm budget
    assert set(plans[solved:]) == {(320.0, 16)}
    # one rank logs
    assert len(_lines(caplog, "re-armed #1")) == 1
    assert len(_lines(caplog, "watch:")) == 2 + 2


# ---------------------------------------------------------------- default
def _old_entrypoint(self):
    """`EPLBManager._entrypoint` as it was before the drift watch."""
    for _ in range(self._first_rebalance_steps()):
        yield
    while not self._max_rebalances_reached():
        yield from self._rebalance()
        if self._max_rebalances_reached():
            break
        for _ in range(self.rebalance_interval):
            yield
    self._stop_rebalancing()
    while True:
        yield


def _trace(manager, generator, steps=120):
    """Per step of the schedule: (rebalances so far, stopped)."""

    def rebalance():
        manager._rebalance_count += 1
        if manager._rebalance_count % 2:
            yield  # a rebalance that takes a chunk step

    manager._rebalance = rebalance
    manager._gen = generator(manager)
    trace = []
    for _ in range(steps):
        _step(manager)
        trace.append((manager.rebalance_count, manager.rebalancing_stopped))
    return trace


@pytest.mark.parametrize(
    "env",
    [
        {"ATOM_EPLB_MAX_REBALANCES": 3},
        {"ATOM_EPLB_MAX_REBALANCES": 3, "ATOM_EPLB_REARM_DRIFT": 0},
        {"ATOM_EPLB_MAX_REBALANCES": 0, "ATOM_EPLB_REARM_DRIFT": 0.3},
        {
            "ATOM_EPLB_MAX_REBALANCES": 3,
            "ATOM_EPLB_GATE_PERMUTE": 0,
            "ATOM_EPLB_REARM_DRIFT": 0.3,
        },
    ],
)
def test_without_a_drift_watch_the_schedule_is_the_old_one(monkeypatch, caplog, env):
    with caplog.at_level(logging.INFO, logger="atom"):
        new = _trace(_manager(monkeypatch, **env), eplb.EPLBManager._entrypoint)
        old = _trace(_manager(monkeypatch, **env), _old_entrypoint)
    assert new == old
    stops = env["ATOM_EPLB_MAX_REBALANCES"] > 0
    assert new[-1] == ((3, True) if stops else (new[-1][0], False))
    assert _lines(caplog, "watch:") == [] and _lines(caplog, "observe #") == []
    assert len(_lines(caplog, "EPLB stopped after")) == (2 if stops else 0)
    # the new schedule says why it froze, where a rebalance is an evaluation
    assert len(_lines(caplog, "frozen after rebalance")) == (1 if stops else 0)


def test_stopped_for_good_no_step_reaches_the_schedule(monkeypatch):
    manager = _manager(monkeypatch, ATOM_EPLB_MAX_REBALANCES=1)
    gpr = manager.gate_permute
    gpr.pf_total.copy_(_skewed(1))
    for seed in range(100):
        if not manager.rebalancing_stopped:
            gpr.pf_total.add_(_skewed(seed))
            _step(manager)
    assert manager.rebalancing_stopped and not gpr.watching
    manager._gen = iter(lambda: pytest.fail("the schedule advanced"), None)
    _step(manager, 20)
    _decode(gpr, 300, 20)
    assert (gpr.watch_rows, gpr.watch_steps) == (0, 0)


def test_switch_defaults():
    from atom.utils import envs

    assert envs.ATOM_EPLB_REARM_DRIFT == 0.0
    assert envs.ATOM_EPLB_REARM_CHECK_STEPS is None
    assert envs.ATOM_EPLB_REARM_MIN_SAMPLES == 16
    assert envs.ATOM_EPLB_REARM_BUDGET is None
    assert envs.ATOM_EPLB_REARM_REBALANCES == 2
    assert envs.ATOM_EPLB_FREEZE_IDLE == 2
    assert envs.ATOM_EPLB_FROZEN_OBSERVE_STEPS == 0
    assert envs.ATOM_EPLB_DUMP_DIR is None
    assert envs.ATOM_EPLB_FIRST_BUDGET is None


def test_check_steps_default_to_four_intervals(monkeypatch):
    manager = _manager(monkeypatch)
    monkeypatch.delenv("ATOM_EPLB_REARM_CHECK_STEPS")
    assert manager._rearm_check_steps() == 4 * INTERVAL


# ---------------------------------------------------------------- collective
def test_reduce_counts_uses_the_cpu_group(monkeypatch):
    env = _setup(monkeypatch)
    calls = []

    def all_reduce(tensor, op=None, group=None):
        calls.append((tensor.device.type, tuple(tensor.tolist()), group))
        tensor.mul_(8)  # as if eight ranks held the same

    monkeypatch.setattr(torch.distributed, "all_reduce", all_reduce)
    group = SimpleNamespace(world_size=8, cpu_group="gloo", device_group="rccl")
    assert env.gp.reduce_counts(group, 30, 4) == (240, 32, 8)
    assert calls == [("cpu", (30, 4), "gloo")]
    env.gp.watch_rows, env.gp.watch_steps = 30, 4
    assert env.gp.take_watch(group) == (60.0, 4)
    assert (env.gp.watch_rows, env.gp.watch_steps) == (0, 0)
    env.gp.window_rows, env.gp.window_steps = 99, 5
    assert env.gp.window_samples(group) == 5
    assert env.gp.window_steps == 5  # read, not restarted
    # one rank: no collective at all
    calls.clear()
    assert env.gp.reduce_counts(SimpleNamespace(world_size=1), 3, 2) == (3, 2, 1)
    assert env.gp.reduce_counts(None, 3, 2) == (3, 2, 1) and calls == []


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
    manager = _manager(monkeypatch, ATOM_EPLB_MAX_REBALANCES=2, ATOM_EPLB_FREEZE_IDLE=0)
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
    manager = _manager(monkeypatch, ATOM_EPLB_MAX_REBALANCES=3, ATOM_EPLB_FREEZE_IDLE=0)
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
    # and switched off it never freezes on idle evaluations
    manager = _manager(monkeypatch, ATOM_EPLB_MAX_REBALANCES=5, ATOM_EPLB_FREEZE_IDLE=0)
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


# ---------------------------------------------------------------- observe / dump
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


def test_frozen_placement_is_observed_on_each_windows_picks(
    monkeypatch, caplog, tmp_path
):
    observe = 4
    manager = _manager(
        monkeypatch,
        ATOM_EPLB_MAX_REBALANCES=1,
        ATOM_EPLB_FROZEN_OBSERVE_STEPS=observe,
        ATOM_EPLB_DUMP_DIR=tmp_path,
        ATOM_EPLB_DECODE_SAMPLE_EVERY=1,
    )
    calls = _scripted_policy(monkeypatch, [True])
    gpr = manager.gate_permute
    held = torch.full((8, TOP_K), 3, dtype=torch.int32)
    with gpr.decode_capture("bucket"):
        gpr.place(manager._moe_layers[0], held)
    with caplog.at_level(logging.INFO, logger="atom"):
        while not manager.rebalancing_stopped:
            gpr.pf_total[0, 0] += 1
            _step(manager)
        frozen = manager.live_metadata.physical_to_logical_map.clone()
        assert gpr.observing and gpr.counting and not gpr.watching
        base = int(gpr.pf_total.sum())
        for window in (1, 2):
            # counting goes on while frozen: a prefill pass and decode samples
            manager.monitor.on_forward_start()
            gpr.place(
                manager._moe_layers[1],
                torch.full((10 * window, TOP_K), 2, dtype=torch.int32),
            )
            manager.monitor.on_forward_end(is_dummy_run=False, is_pure_prefill=True)
            _decode(gpr, 4 * window, 3)
            _step(manager, observe)
            # a head line and two metrics lines (no reference table here)
            assert len(_lines(caplog, "observe #")) == 3 * window
    assert calls == [True]  # nothing solved, nothing moved
    assert torch.equal(manager.live_metadata.physical_to_logical_map, frozen)
    assert manager.rebalancing_stopped
    heads = [line for line in _lines(caplog, "observe #") if "metrics" not in line]
    # 30 counted in the pass, and the one pick of the step the chunk took
    assert (
        "observe #1: frozen since rebalance #1 | window prefill picks=31 decode picks=36"
        in heads[0]
    )
    assert "decode samples/rank=3 rows/step=4.0" in heads[0]
    assert (
        "observe #2: frozen since rebalance #1 | window prefill picks=60 decode picks=72"
        in heads[1]
    )
    assert "rows/step=8.0" in heads[1]
    names = [
        line.split("metrics ")[1].split(":")[0].strip()
        for line in _lines(caplog, "observe #1 metrics")
    ]
    assert names == ["frozen", "identity"]
    assert int(gpr.pf_total.sum()) == base + 90
    dumps = [torch.load(f) for f in sorted(tmp_path.glob("window_*.pt"))]
    assert [d["kind"] for d in dumps] == ["rebalance", "observe", "observe"]
    assert int(dumps[2]["prefill_counts_window"].sum()) == 60
    assert dumps[2]["executed"] is False
    assert torch.equal(dumps[2]["p2l_before"], dumps[2]["p2l_after"])


def test_frozen_without_observer_or_watch_counts_nothing(monkeypatch):
    manager = _manager(monkeypatch, ATOM_EPLB_MAX_REBALANCES=1)
    _scripted_policy(monkeypatch, [True])
    gpr = manager.gate_permute
    while not manager.rebalancing_stopped:
        gpr.pf_total[0, 0] += 1
        _step(manager)
    assert not gpr.observing and not gpr.watching and not gpr.counting
    manager.monitor.on_forward_start()
    gpr.place(manager._moe_layers[1], torch.full((4, TOP_K), 2, dtype=torch.int32))
    assert int(manager.monitor._cur_pass_count.sum()) == 0


def test_freeze_at_step_freezes_without_a_rebalance(monkeypatch):
    """ATOM_EPLB_FREEZE_AT_STEP (diagnostic): after that many steps the
    placement is frozen as it is -- no evaluation, nothing moved, and the
    schedule is not advanced again."""
    manager = _manager(monkeypatch, ATOM_EPLB_FREEZE_AT_STEP=3)
    before = manager.live_metadata.physical_to_logical_map.clone()
    monkeypatch.setattr(
        manager, "_execute_rebalance", lambda: pytest.fail("a rebalance ran")
    )
    gen = manager._entrypoint()
    for step in range(3):
        next(gen)
        assert not manager.rebalancing_stopped, step
    next(gen)
    assert manager.rebalancing_stopped and manager.rebalance_count == 0
    assert torch.equal(manager.live_metadata.physical_to_logical_map, before)
    next(gen)  # idles


def test_lazy_migration_group_is_made_at_the_first_rebalance_and_destroyed_at_the_freeze(
    monkeypatch,
):
    """ATOM_EPLB_LAZY_MIGRATION_GROUP: no group (no collective) until a
    rebalance needs one; the loads are reduced over it; the freeze destroys
    it, and a later rebalance (a re-arm) makes a new one."""
    manager = _manager(monkeypatch, ATOM_EPLB_MAX_REBALANCES=1)
    calls = []
    monkeypatch.setattr(
        torch.distributed, "new_group", lambda ranks: calls.append("new") or object()
    )
    monkeypatch.setattr(
        torch.distributed, "barrier", lambda group: calls.append("barrier")
    )
    monkeypatch.setattr(
        torch.distributed,
        "destroy_process_group",
        lambda group: calls.append("destroy"),
    )
    manager._lazy_group = True
    manager._ep_global_ranks = [0, 1, 2, 3]
    assert manager._migration_group is None and calls == []
    group = manager._reduce_group()
    assert calls == ["new", "barrier"] and group.device_group is manager._migration_group
    assert manager._ensure_migration_group() is manager._migration_group
    assert calls == ["new", "barrier"]  # made once
    manager._release_migration_group()
    assert calls == ["new", "barrier", "destroy"] and manager._migration_group is None
    manager._release_migration_group()  # nothing left to destroy
    assert calls == ["new", "barrier", "destroy"]
    manager._ensure_migration_group()
    assert calls[-2:] == ["new", "barrier"] and manager._groups_made == 2
    # not lazy: the group made at init is used as it is and never destroyed
    manager._lazy_group = False
    kept = manager._migration_group
    manager._release_migration_group()
    assert manager._migration_group is kept and manager._ensure_migration_group() is kept
