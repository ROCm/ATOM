# SPDX-License-Identifier: MIT
"""Execute the real DP loop on independent CPU threads with paired collectives.

The transport and model forward are faked; the loop, interval check, state
reduction and sync gate are the repository's code. This covers control-flow
agreement without importing the GPU engine or replacing the loop with a model.
"""

import ast
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from atom.model_engine.dp_sync_gate import MAX_DP_SYNC_INTERVAL, DPSyncGate

SOURCE = Path(__file__).resolve().parents[1] / "atom/model_engine/engine_core.py"


class _Scalar:
    def __init__(self, value):
        self.value = value

    def item(self):
        return self.value


class _Tensor:
    def __init__(self, values):
        self.values = list(values)

    def __getitem__(self, index):
        return _Scalar(self.values[index])

    def tolist(self):
        return list(self.values)


class _Group:
    def __init__(self, size):
        self.barrier = threading.Barrier(size, timeout=5)
        self.values = [None] * size

    def exchange(self, rank, kind, tick, values):
        self.values[rank] = (kind, tick, list(values))
        self.barrier.wait()
        entries = list(self.values)
        self.barrier.wait()
        assert len({(kind, tick) for kind, tick, _ in entries}) == 1, entries
        return [max(column) for column in zip(*(x[2] for x in entries))]


def _loop_methods(scope):
    cls = next(
        node
        for node in ast.parse(SOURCE.read_text(encoding="utf-8")).body
        if isinstance(node, ast.ClassDef) and node.name == "DPEngineCoreProc"
    )
    names = {"busy_loop", "_validate_dp_sync_interval", "_sync_dp_state"}
    methods = [
        node
        for node in cls.body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    exec(  # noqa: S102 - only the named methods from repository source.
        compile(ast.Module(methods, []), str(SOURCE), "exec"), scope
    )
    return {name: scope[name] for name in names}


def _run(intervals, work, commands=None, shutdown_at=32, startup_error=None):
    """A command enters on its specified round and is consumed on a sync round."""
    group = _Group(len(intervals))
    engines, failures = [], []
    commands = commands or {}
    for rank, interval in enumerate(intervals):
        engine = SimpleNamespace(
            rank=rank,
            tick=-1,
            events=[],
            pending=[],
            _report_backlog=False,
            _idle_when_blocked=False,
            _dp_group_hidden_work=False,
            _shutting_down=False,
            _is_rl_weights_offloaded=False,
            _weight_sync_fenced=False,
            _generation_paused=False,
            label=f"rank-{rank}",
            dp_group=group,
            utility_queue=None,
            cleaned=False,
        )

        def group_max_each(values, group, e=engine):
            if startup_error is not None:
                raise startup_error
            # The real payload is int64.
            assert all(-(1 << 63) <= v < 1 << 63 for v in values), values
            values = group.exchange(e.rank, "bounds", e.tick, values)
            e.events.append(("bounds", e.tick))
            return tuple(values)

        def all_reduce(tensor, *, op, group, e=engine):
            kind = "state"
            tensor.values = group.exchange(e.rank, kind, e.tick, tensor.values)
            e.events.append((kind, e.tick))

        scope = {
            "DPSyncGate": DPSyncGate,
            "MAX_DP_SYNC_INTERVAL": MAX_DP_SYNC_INTERVAL,
            "group_max_each": group_max_each,
            "envs": SimpleNamespace(
                ATOM_DP_SYNC_INTERVAL=interval, ATOM_METRICS_UPDATE_INTERVAL_S=1e9
            ),
            "torch": SimpleNamespace(
                tensor=lambda values, **kw: _Tensor(values),
                int64=None,
                distributed=SimpleNamespace(
                    all_reduce=all_reduce, ReduceOp=SimpleNamespace(MAX=None)
                ),
            ),
            "time": SimpleNamespace(monotonic=lambda: 0, sleep=lambda _: None),
            "logger": SimpleNamespace(
                info=lambda *a: None, warning=lambda *a: None, exception=lambda *a: None
            ),
            "tracing": SimpleNamespace(shutdown=lambda: None),
        }
        for name, method in _loop_methods(scope).items():
            setattr(engine, name, method.__get__(engine))

        def pull(e=engine):
            if (e.rank, e.tick) in commands:
                e.pending.append(commands[e.rank, e.tick])
            return e.tick >= shutdown_at

        def process_queue(_queue, _engine, e=engine):
            for attr, value in e.pending:
                setattr(e, attr, value)
                e.events.append(("utility", e.tick + 1))
            e.pending.clear()

        def step(dummy=False, e=engine):
            assert not (
                e._is_rl_weights_offloaded
                or e._weight_sync_fenced
                or e._generation_paused
            )
            group.exchange(e.rank, "step", e.tick, [1])
            e.events.append(("dummy" if dummy else "forward", e.tick))

        def process_step(e=engine, do_step=step):
            # schedule() polls the delayer on locally idle ranks too, before
            # deciding whether the forward will be real or dummy.
            e.scheduler.prefill_delayer.should_allow_prefill()
            if not work(e.rank, e.tick):
                return False
            do_step()
            return True

        engine.pull_and_process_input_queue = pull
        engine.utility_handler = SimpleNamespace(
            process_queue=process_queue, push_metrics=lambda: None
        )
        engine.scheduler = SimpleNamespace(
            heartbeat_throughput=lambda now, e=engine: setattr(e, "tick", e.tick + 1),
            is_finished=lambda e=engine: not work(e.rank, e.tick),
            engine_stats=SimpleNamespace(
                note_busy=lambda now: None,
                note_idle=lambda now: None,
                bump_step=lambda: None,
            ),
            prefill_delayer=SimpleNamespace(
                should_allow_prefill=lambda e=engine: (
                    group.exchange(e.rank, "prefill", e.tick, [1]),
                    e.events.append(("prefill", e.tick)),
                ),
                notify_idle=lambda: None,
            ),
            block_manager=SimpleNamespace(
                complete_previous_checkpoint_batch=lambda: None
            ),
            publish_kv_events=lambda: None,
            shutdown_kv_events=lambda e=engine: setattr(e, "cleaned", True),
            close_kv_connector=lambda: None,
        )
        engine._process_engine_step = process_step
        engine._execute_dummy_batch = lambda do_step=step: do_step(dummy=True)
        engine.has_pending_kv_work = lambda: False
        engine._drain_kv_work_at_exit = lambda: None
        engine._report_held_kv_work = lambda: None
        engines.append(engine)

    def run(engine):
        try:
            engine.busy_loop()
        except (ValueError, RuntimeError, AssertionError) as exc:
            failures.append((engine.rank, exc))

    threads = [threading.Thread(target=run, args=(e,), daemon=True) for e in engines]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)
    assert not any(thread.is_alive() for thread in threads), "DP loop did not finish"
    assert all(e.cleaned for e in engines), "startup/loop failure bypassed cleanup"
    return engines, failures


@pytest.mark.parametrize("interval", [1, 2, 8, 16])
def test_uneven_work_keeps_real_and_dummy_steps_paired(interval):
    engines, failures = _run([interval] * 4, lambda r, t: r == 0 and t < 20)
    assert not failures
    steps = [
        [t for kind, t in e.events if kind in ("dummy", "forward")] for e in engines
    ]
    assert all(x == steps[0] for x in steps)
    assert len(steps[0]) >= 20
    assert any(kind == "dummy" for kind, _ in engines[1].events)
    for engine in engines:
        assert [t for kind, t in engine.events if kind == "prefill"] == steps[0]
    syncs = [[t for kind, t in e.events if kind == "state"] for e in engines]
    assert all(x == syncs[0] for x in syncs)
    assert syncs[0][: 20 // interval] == list(range(0, 20, interval))[: 20 // interval]


@pytest.mark.parametrize(
    "attribute",
    ["_generation_paused", "_is_rl_weights_offloaded", "_weight_sync_fenced"],
)
def test_one_rank_hold_releases_and_rearms_on_a_sync_round(attribute):
    engines, failures = _run(
        [8] * 4,
        lambda r, t: t < 24,
        {(1, 1): (attribute, True), (1, 12): (attribute, False)},
    )
    assert not failures
    assert ("utility", 8) in engines[1].events
    assert ("utility", 13) in engines[1].events
    for engine in engines:
        syncs = [t for kind, t in engine.events if kind == "state"]
        assert list(range(8, 14)) == [t for t in syncs if 8 <= t <= 13]
        steps = [t for kind, t in engine.events if kind in ("forward", "dummy")]
        assert not any(8 <= t < 13 for t in steps)
        assert 13 in steps and 21 in syncs


@pytest.mark.parametrize("intervals", [[1, 8], [2, 8, 16, 2]])
def test_mismatched_intervals_fail_on_all_ranks_before_state_or_forward(intervals):
    engines, failures = _run(intervals, lambda r, t: True)
    assert len(failures) == len(intervals)
    assert all(isinstance(exc, ValueError) for _, exc in failures)
    assert all("ATOM_DP_SYNC_INTERVAL must match" in str(exc) for _, exc in failures)
    assert all(e.events == [("bounds", -1)] for e in engines)


@pytest.mark.parametrize(
    "intervals",
    [[1, None], [None, 8, 16, None], [1, 17], [17, 17], [1 << 70, 1 << 70]],
)
def test_invalid_interval_fails_on_all_ranks_before_any_other_collective(intervals):
    engines, failures = _run(intervals, lambda r, t: True)
    assert len(failures) == len(intervals)
    assert all(
        isinstance(exc, ValueError) and "invalid value" in str(exc)
        for _, exc in failures
    )
    assert all(e.events == [("bounds", -1)] for e in engines)


def test_idle_then_new_work_restarts_with_a_sync():
    engines, failures = _run([8] * 2, lambda r, t: 4 <= t < 10 or 20 <= t < 24)
    assert not failures
    for engine in engines:
        assert ("state", 4) in engine.events and ("forward", 4) in engine.events
        assert ("state", 20) in engine.events and ("forward", 20) in engine.events


def test_shutdown_drains_existing_work_with_sync_on_every_remaining_round():
    engines, failures = _run([8] * 2, lambda r, t: t < 18 + r, shutdown_at=3)
    assert not failures
    for engine in engines:
        syncs = [t for kind, t in engine.events if kind == "state"]
        # A locally latched shutdown becomes global at the next sync (round 8).
        assert list(range(8, 20)) == [t for t in syncs if 8 <= t <= 19]
        assert any(kind == "forward" and t == 17 for kind, t in engine.events)


def test_a_lost_peer_at_startup_shuts_down_like_a_state_sync():
    """The startup agreement failing (a dead peer) shuts the loop down as
    `_sync_dp_state` does, instead of raising out of `busy_loop`."""
    engines, failures = _run(
        [1, 1], lambda r, t: False, startup_error=RuntimeError("peer lost")
    )
    assert not failures
    assert all(e._shutting_down and e.cleaned for e in engines)
    assert all(
        kind not in ("bounds", "forward") for e in engines for kind, _ in e.events
    )
