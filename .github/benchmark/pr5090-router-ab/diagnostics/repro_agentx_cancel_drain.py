"""Replay AgentX 1.0.14 cancellation bookkeeping using its unchanged real methods.

Only IO, metadata and publication are replaced. The fixture starts at the
observed drain boundary; it does not reproduce how the future join was created.
"""

import ast
import asyncio
from collections import defaultdict
from dataclasses import dataclass
import hashlib
import inspect
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace as NS

ROOT = Path(__file__).resolve().parent
if len(sys.argv) > 3:
    raise SystemExit(
        "Usage: repro_agentx_cancel_drain.py [extracted-aiperf-directory] [output.json]"
    )
SOURCE = (
    Path(sys.argv[1]).resolve()
    if len(sys.argv) > 1
    else ROOT / "agentx-1.0.14-source/aiperf"
)
OUTPUT = (
    Path(sys.argv[2]).resolve()
    if len(sys.argv) > 2
    else ROOT / "agentx-cancel-drain-repro-20261125.json"
)
EXPECTED = {
    "timing/phase/credit_counter.py": "fe0197bde6d107427164c9f6937c61384a3c77800c36ebda399800f1e8c0c900",
    "timing/phase/progress_tracker.py": "9e5d676bfe8dab3a7d5254c018585d88ff0a2ca108d7dd7cfcced591d4bf8bbd",
    "credit/callback_handler.py": "2114eb180ce47821c7047a4fcab219ef6010b97c9972633d73230e6090e2a9f3",
    "timing/branch_orchestrator.py": "9a88cc8a42e75b044e5925fdd41becb54ecc9fe092efbb955c2cee53060dca6e",
    "timing/phase/runner.py": "82469e54d38578adb27ce2fc3c15c32f27e403a8e9babcd0cc8f61685e0c639a",
}
PHASE = NS(WARMUP="warmup", PROFILING="profiling")


class Log:
    def __init__(self):
        self.rows = []

    def __getattr__(self, level):
        def record(message, *args):
            if callable(message):
                message = message()
            self.rows.append((level, str(message)))

        return record


log = Log()


class Stats(NS):
    @property
    def in_flight_requests(self):
        return self.requests_sent - self.requests_completed - self.requests_cancelled


env = dict(
    asyncio=asyncio,
    inspect=inspect,
    dataclass=dataclass,
    _logger=log,
    logger=log,
    CreditPhase=PHASE,
    CreditPhaseStats=Stats,
)
hashes, selected = {}, {}


def load_class(file, name, members=None):
    path = SOURCE / file
    text = path.read_text(encoding="utf-8")
    hashes[file] = hashlib.sha256(path.read_bytes()).hexdigest()
    assert (
        hashes[file] == EXPECTED[file]
    ), f"Source differs from the pinned wheel: {file}"
    cls = next(
        n
        for n in ast.parse(text).body
        if isinstance(n, ast.ClassDef) and n.name == name
    )
    if members is not None:
        cls.body = [n for n in cls.body if getattr(n, "name", None) in members]
        assert {n.name for n in cls.body} == set(members)
        cls.bases = []
        selected[name] = sorted(members)
    else:
        selected[name] = "entire class"
    module = ModuleType("agentx_repro_" + name)
    module.__dict__.update(env)
    sys.modules[module.__name__] = module
    tree = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            cls,
        ],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(tree), str(path), "exec"), module.__dict__)
    env[name] = getattr(module, name)
    return env[name]


Counter = load_class("timing/phase/credit_counter.py", "CreditCounter")
Progress = load_class("timing/phase/progress_tracker.py", "PhaseProgressTracker")
Context = load_class("credit/callback_handler.py", "PhaseCallbackContext")
Callback = load_class("credit/callback_handler.py", "CreditCallbackHandler")
Branch = load_class(
    "timing/branch_orchestrator.py",
    "BranchOrchestrator",
    {
        "has_pending_branch_work",
        "get_branch_ids",
        "intercept",
        "_maybe_suspend_parent",
        "_pop_future_join",
        "set_drain_observer",
        "_notify_drain",
    },
)
env["Environment"] = NS(TIMING=NS(CANCEL_DRAIN_TIMEOUT=0.05))
Runner = load_class(
    "timing/phase/runner.py",
    "PhaseRunner",
    {
        "_wait_for_returning_complete",
        "_wait_for_event_with_timeout",
    },
)


async def noop(*args, **kwargs):
    return None


async def scenario(
    name, future_join=False, synchronous=False, clear_after=False, missing_ack=False
):
    config = NS(
        phase="profiling",
        phase_index=0,
        profiling_index=0,
        phase_name="profiling",
        phase_kind="profiling",
        total_expected_requests=None,
        expected_duration_sec=3600,
        expected_num_sessions=None,
        grace_period_sec=30,
    )
    progress = Progress(config)
    for _ in range(6):
        progress.increment_sent(
            NS(
                agent_depth=0,
                turn_index=1,
                is_session_start=False,
                num_turns=100,
                counts_toward_phase_target=False,
            )
        )
    progress.increment_returned(False, False)
    progress.freeze_sent_counts()
    lifecycle = NS(
        is_complete=False,
        is_sending_complete=True,
        started_at_ns=1,
        sending_complete_at_ns=2,
        complete_at_ns=None,
        timeout_triggered=True,
        grace_period_triggered=False,
        was_cancelled=False,
        time_left_in_seconds=lambda **kw: 0,
    )

    def complete(grace_period_triggered):
        lifecycle.is_complete = True
        lifecycle.complete_at_ns = 3
        lifecycle.grace_period_triggered = grace_period_triggered

    lifecycle.mark_complete = complete
    branch = Branch()
    branch._active_joins = {}
    branch._future_joins = {"future-parent": {74: object()}} if future_join else {}
    branch._child_to_join = {}
    branch._descendant_counts = {}
    branch._cleaning_up = False
    branch._accelerated_warmup_started = False
    branch._parent_locks = defaultdict(asyncio.Lock)
    branch._drain_observer = None
    branch._cs = NS(
        get_metadata=lambda _: NS(turns=[NS(branch_ids=[]), NS(branch_ids=[])])
    )
    concurrency = NS(
        release_session_slot=lambda *args: None, release_prefill_slot=lambda *args: None
    )
    callback = Callback(concurrency)
    callback.set_branch_orchestrator(branch)
    strategy = NS()
    callback.register_phase(
        phase="profiling",
        phase_index=0,
        progress=progress,
        lifecycle=lifecycle,
        stop_checker=NS(
            can_send_any_turn=lambda: False, can_send_child_turn=lambda: False
        ),
        strategy=strategy,
    )
    snapshots, tasks = [], []

    async def acknowledgements():
        for i in range(4 if missing_ack else 5):
            credit = NS(
                id=i,
                phase="profiling",
                phase_index=0,
                agent_depth=0,
                is_final_turn=False,
                turn_index=1,
                conversation_id=str(i),
                x_correlation_id=str(i),
                effective_root_correlation_id=str(i),
            )
            await callback.on_credit_return(
                "worker",
                NS(credit=credit, cancelled=True, error=None, first_token_sent=True),
            )
        snapshots.append(
            {
                "cancelled": progress.counter.requests_cancelled,
                "in_flight": progress.in_flight,
                "event_set": progress.all_credits_returned_event.is_set(),
            }
        )
        if clear_after:
            branch._future_joins.clear()
            branch._notify_drain()

    async def cancel_all():
        if synchronous:
            await acknowledgements()
        else:
            tasks.append(asyncio.create_task(acknowledgements()))

    runner = Runner()
    runner._config = config
    runner._progress = progress
    runner._lifecycle = lifecycle
    runner._branch_orchestrator = branch
    runner._credit_router = NS(cancel_all_credits=cancel_all)
    runner._phase_publisher = NS(publish_progress=noop, publish_phase_complete=noop)
    runner._release_stuck_slots = lambda: None
    runner._create_final_stats = lambda: progress.create_stats(lifecycle)
    runner._format_phase_complete = lambda stats: "complete"
    runner._snapshot_branch_stats = lambda: None
    messages = []
    at_timeout = []
    for level in ("debug", "info", "warning", "notice", "error"):

        def record(message, level=level):
            value = message() if callable(message) else message
            messages.append((level, value))
            if level == "error":
                at_timeout.append(
                    {
                        "in_flight": progress.in_flight,
                        "cancelled": progress.counter.requests_cancelled,
                        "branch_pending": branch.has_pending_branch_work(),
                    }
                )

        setattr(runner, level, record)
    await runner._wait_for_returning_complete(strategy)
    await asyncio.gather(*tasks)
    return {
        "case": name,
        "acknowledgement_snapshots": snapshots,
        "timeout_snapshots": at_timeout,
        "final_cancelled": progress.counter.final_requests_cancelled,
        "final_in_flight": progress.in_flight,
        "cancel_drain_timed_out": any(
            "cancelled credits to return" in str(m)
            for level, m in messages
            if level == "error"
        ),
    }


async def main():
    definitions = [
        ("no_future_join", {}),
        ("future_join_async_ack", {"future_join": True}),
        ("future_join_drains_after_ack", {"future_join": True, "clear_after": True}),
        ("future_join_sync_ack", {"future_join": True, "synchronous": True}),
        ("one_ack_really_missing", {"missing_ack": True}),
    ]
    results = [await scenario(name, **args) for name, args in definitions]
    assert [r["cancel_drain_timed_out"] for r in results] == [
        False,
        True,
        False,
        False,
        True,
    ]
    assert results[1]["timeout_snapshots"] == [
        {"in_flight": 0, "cancelled": 5, "branch_pending": True}
    ]
    assert results[4]["timeout_snapshots"][0]["in_flight"] == 1
    for _ in range(9):
        assert [await scenario(name, **args) for name, args in definitions] == results
    result = {
        "pinned_version": "agentx-harness1.0.14",
        "source_sha256": hashes,
        "python_version": sys.version,
        "platform": sys.platform,
        "unchanged_classes_and_methods": selected,
        "repetitions_per_case": 10,
        "cpu_only": True,
        "gpu_experiments": 0,
        "results": results,
        "limits": "Uses real counting, callbacks, DAG predicate/intercept and phase-drain methods. Starts from an observed future-join state; no HTTP/IPC, no reconstruction of the full DAG creation history. Drain timeout shortened from10s to50ms only for the local test.",
    }
    OUTPUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
