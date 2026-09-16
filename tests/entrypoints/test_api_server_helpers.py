# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Unit tests for helpers in ``atom.entrypoints.openai.api_server`` that do
not require a GPU or a running engine.

The ``api_server`` module pulls in transformers + uvicorn + fastapi + an
engine-ready ``atom`` package at import time. The repo's ``tests/conftest.py``
already stubs several heavy imports; here we only test small pure-python
helpers, so if any transitive dependency is unavailable we skip the module
rather than block the rest of the suite.
"""

from __future__ import annotations

import ast
import asyncio
import inspect
import sys
import textwrap
import threading
import types
from types import SimpleNamespace

import pytest
from import_guard import skip_if_dependency_missing


def _install_api_server_stubs() -> list[str]:
    """Ensure attribute access ``atom.SamplingParams`` works under the stubbed
    ``atom`` package that ``tests/conftest.py`` installs, and stub any heavy
    transitive deps (``aiter``-backed engine core manager and its argparse
    helper) that ``api_server`` would otherwise drag in at import time.

    Stubs are only installed when the corresponding real module cannot be
    imported in this environment (e.g. Windows without ``aiter``). Any
    module we inject here is recorded and torn down in a module-level
    fixture so we don't leak stubs into tests that run later and expect
    the real implementation (notably ``tests/test_arg_utils_spec.py``).
    """
    import importlib

    from atom.sampling_params import SamplingParams  # real implementation

    atom_pkg = sys.modules.get("atom")
    if atom_pkg is not None and not hasattr(atom_pkg, "SamplingParams"):
        atom_pkg.SamplingParams = SamplingParams  # type: ignore[attr-defined]

    injected: list[str] = []

    def _try_import_else_stub(mod_name: str, attr_name: str, stub_cls) -> None:
        if mod_name in sys.modules:
            return
        try:
            importlib.import_module(mod_name)
        except Exception:
            stub = types.ModuleType(mod_name)
            setattr(stub, attr_name, stub_cls)
            sys.modules[mod_name] = stub
            injected.append(mod_name)

    class _StubCoreManager:
        def __init__(self, *a, **kw):
            pass

        def add_request(self, reqs):
            return None

    class _StubEngineArgs:
        @classmethod
        def add_cli_args(cls, parser):
            return parser

        @classmethod
        def from_cli_args(cls, args):
            return cls()

        def create_engine(self, tokenizer=None):
            return None

    _try_import_else_stub(
        "atom.model_engine.engine_core_mgr", "CoreManager", _StubCoreManager
    )
    _try_import_else_stub("atom.model_engine.arg_utils", "EngineArgs", _StubEngineArgs)
    return injected


_injected_modules: list[str] = []  # set in try; kept defined for `finally`
try:
    _injected_modules = _install_api_server_stubs()
    import importlib

    api_server = importlib.import_module("atom.entrypoints.openai.api_server")
except ImportError as exc:  # pragma: no cover - environment-dependent skip
    # Re-raises unless a third-party dependency is what is missing. It used to
    # skip on anything, so a syntax error in `api_server.py` silenced this
    # whole module and the suite still reported a clean run.
    skip_if_dependency_missing(exc, "api_server import unavailable")
    api_server = None  # type: ignore[assignment]
    _import_error = exc
    # NB: do NOT reset _injected_modules here. When api_server import fails
    # (e.g. PIL absent on the non-GPU runner), the stubs injected by
    # _install_api_server_stubs() must still be torn down in `finally`;
    # clearing the list here would leak them into sys.modules and pollute
    # tests collected later (notably tests/test_arg_utils_spec.py, which then
    # sees a stub EngineArgs instead of the real one).
else:
    _import_error = None
finally:
    # Remove any stubs we injected so tests collected *after* this module
    # (notably ``tests/test_arg_utils_spec.py``) can still import the real
    # ``atom.model_engine.arg_utils`` / ``engine_core_mgr``. ``api_server``
    # already bound the names it needed at module import time.
    for _mod_name in list(_injected_modules):
        sys.modules.pop(_mod_name, None)
    _injected_modules = []


pytestmark = pytest.mark.skipif(
    api_server is None,
    reason=f"api_server import unavailable: {_import_error!r}",
)


def _run_bounded(coro, timeout=5.0):
    """Drive a coroutine that must end by cancellation, and fail if it does not.

    Bare `asyncio.run` on a `while True` turns "the guard swallowed the
    cancellation" into a suite that hangs and a CI job that reports a timeout
    with no failing assertion. That is the same defect these tests exist to
    pin, so the bound is not optional here.
    """
    try:
        asyncio.run(asyncio.wait_for(coro, timeout))
    except asyncio.CancelledError:
        return
    except TimeoutError:
        pytest.fail(f"still running after {timeout}s; a guard swallowed the cancel")
    pytest.fail("returned on its own; this must run until cancelled")


def _boom(*_args, **_kwargs):
    """A step that fails. Named so the loop's warning names something real."""
    raise RuntimeError("boom")


class TestCoerceN:
    """``_coerce_n`` normalizes the request ``n`` before engine fan-out."""

    def test_none_becomes_one(self):
        assert api_server._coerce_n(None, 0.8) == 1

    def test_zero_becomes_one(self):
        assert api_server._coerce_n(0, 0.8) == 1

    def test_negative_becomes_one(self):
        assert api_server._coerce_n(-2, 0.8) == 1

    def test_non_int_string_becomes_one(self):
        assert api_server._coerce_n("not-a-number", 0.8) == 1  # type: ignore[arg-type]

    def test_n_passes_through_when_temperature_positive(self):
        assert api_server._coerce_n(4, 0.7) == 4

    def test_n_collapses_to_one_under_greedy_sampling(self):
        # temperature==0 => greedy, so n>1 would produce identical siblings.
        assert api_server._coerce_n(4, 0.0) == 1

    def test_n_collapses_to_one_when_temperature_missing(self):
        assert api_server._coerce_n(4, None) == 1

    def test_n_one_with_greedy_stays_one(self):
        assert api_server._coerce_n(1, 0.0) == 1


class TestBuildSamplingParams:
    """``_build_sampling_params`` threads ``n`` into SamplingParams."""

    def test_default_n_is_one(self):
        sp = api_server._build_sampling_params(
            temperature=0.8,
            max_tokens=16,
            stop_strings=None,
            ignore_eos=False,
        )
        assert sp.n == 1

    def test_n_greater_than_one_propagates(self):
        sp = api_server._build_sampling_params(
            temperature=0.8,
            max_tokens=16,
            stop_strings=None,
            ignore_eos=False,
            n=4,
        )
        assert sp.n == 4

    def test_invalid_n_rejected_by_sampling_params(self):
        with pytest.raises(ValueError, match="n must be >= 1"):
            api_server._build_sampling_params(
                temperature=0.8,
                max_tokens=16,
                stop_strings=None,
                ignore_eos=False,
                n=0,
            )


class TestDPSessionAffinityHeaders:
    def test_disabled_drops_session_metadata(self, monkeypatch):
        monkeypatch.setenv("ATOM_DP_SESSION_AFFINITY", "0")
        request = SimpleNamespace(
            headers={
                "x-dynamo-session-id": "child",
                "x-dynamo-parent-session-id": "parent",
            }
        )
        assert api_server._get_dp_session_affinity_ids(request) == (None, None)

    def test_extracts_dynamo_session_lineage(self, monkeypatch):
        monkeypatch.setenv("ATOM_DP_SESSION_AFFINITY", "1")
        request = SimpleNamespace(
            headers={
                "x-dynamo-session-id": "child",
                "x-dynamo-parent-session-id": "parent",
                "x-correlation-id": "fallback",
            }
        )
        assert api_server._get_dp_session_affinity_ids(request) == (
            "child",
            "parent",
        )

    def test_correlation_id_is_session_fallback(self, monkeypatch):
        monkeypatch.setenv("ATOM_DP_SESSION_AFFINITY", "true")
        request = SimpleNamespace(headers={"x-correlation-id": "session"})
        assert api_server._get_dp_session_affinity_ids(request) == (
            "session",
            None,
        )


class TestAnthropicSamplingParams:
    def test_request_overrides_model_then_neutral_defaults(self, monkeypatch):
        captured = {}
        build_sampling_params = api_server._build_sampling_params

        def capture_sampling_params(**kwargs):
            captured.update(kwargs)
            return build_sampling_params(**kwargs)

        async def fake_nonstream(*_args, **_kwargs):
            return {"text": "", "num_cached_tokens": 0}

        monkeypatch.setattr(
            api_server,
            "engine",
            SimpleNamespace(
                config=SimpleNamespace(
                    generation_config=SimpleNamespace(
                        temperature=1.0,
                        top_p=0.95,
                        top_k=None,
                    ),
                    max_model_len=4096,
                )
            ),
        )
        monkeypatch.setattr(
            api_server, "tokenizer", SimpleNamespace(encode=lambda _text: [1])
        )
        monkeypatch.setattr(api_server, "model_name", "test")
        monkeypatch.setattr(api_server, "apply_chat_template", lambda *_a, **_kw: "")
        monkeypatch.setattr(
            api_server, "_build_sampling_params", capture_sampling_params
        )
        monkeypatch.setattr(
            api_server, "_run_nonstream_with_disconnect", fake_nonstream
        )

        request = api_server.AnthropicMessagesRequest(
            model="test",
            messages=[{"role": "user", "content": "Hi"}],
            temperature=0.0,
        )
        asyncio.run(api_server.anthropic_messages(request, None))

        assert captured["temperature"] == 0.0
        assert captured["top_p"] == 0.95
        assert captured["top_k"] == -1


class TestValidateContextLength:
    """Oversized OpenAI requests should fail before entering the scheduler."""

    def test_equal_to_max_model_len_is_allowed(self):
        api_server._validate_context_length(
            num_prompt_tokens=120,
            max_tokens=8,
            max_model_len=128,
        )

    def test_total_over_max_model_len_is_rejected(self):
        with pytest.raises(ValueError, match="maximum context length is 128"):
            api_server._validate_context_length(
                num_prompt_tokens=121,
                max_tokens=8,
                max_model_len=128,
            )

    def test_prompt_alone_over_max_model_len_is_rejected(self):
        with pytest.raises(ValueError, match="prompt contains at least 129"):
            api_server._validate_context_length(
                num_prompt_tokens=129,
                max_tokens=0,
                max_model_len=128,
            )

    def test_missing_max_model_len_skips_validation(self):
        api_server._validate_context_length(
            num_prompt_tokens=129,
            max_tokens=8,
            max_model_len=None,
        )


class TestARequestIsNotSerialisedForALogNobodyKeeps:
    """Building the log entry is the callee's job, not the caller's.

    ``_log_request_event`` returns immediately when request logging is off,
    but Python evaluates its arguments first, so
    ``_log_request_event("request", rid, request.model_dump())`` dumped every
    message and every tool schema on the event loop and threw the result away
    -- 20-26 us per request on an agent-shaped one, at all four call sites.
    The guard has to run before the dump, which is what
    ``_log_request_model`` is for.

    Asserted on whether ``model_dump`` ran, not on how long it took: the
    defect is an evaluation order, and an order is exactly observable.
    """

    class _Model:
        def __init__(self) -> None:
            self.dumps = 0

        def model_dump(self) -> dict:
            self.dumps += 1
            return {"big": "payload"}

    def test_nothing_is_dumped_while_request_logging_is_off(self, monkeypatch):
        monkeypatch.setattr(api_server, "_request_logger", None)
        model = self._Model()
        api_server._log_request_model("request", "req-1", model)
        assert model.dumps == 0, "the request was serialised for a log that is off"

    def test_it_is_dumped_once_when_request_logging_is_on(self, monkeypatch):
        written: list[str] = []
        monkeypatch.setattr(
            api_server, "_request_logger", SimpleNamespace(info=written.append)
        )
        model = self._Model()
        api_server._log_request_model("request", "req-1", model)
        assert model.dumps == 1
        assert written and "payload" in written[0], "the entry never reached the log"

    @staticmethod
    def _module_ast():
        return ast.parse(inspect.getsource(api_server))

    def _eager_sites(self):
        """Every ``_log_request_event(..., x.model_dump())`` outside the helper.

        ``_log_request_model``'s own body is that call, and there it is
        correct -- it runs after the guard. Excluding it by name rather than
        by weakening the pattern, because the pattern is the whole test.
        """
        tree = self._module_ast()
        helper = next(
            (
                n
                for n in ast.walk(tree)
                if isinstance(n, ast.FunctionDef) and n.name == "_log_request_model"
            ),
            None,
        )
        exempt = {id(n) for n in ast.walk(helper)} if helper is not None else set()
        return [
            f"line {node.lineno}"
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and id(node) not in exempt
            and getattr(node.func, "id", None) == "_log_request_event"
            for arg in node.args
            if isinstance(arg, ast.Call)
            and getattr(arg.func, "attr", None) == "model_dump"
        ]

    def test_the_scan_sees_the_endpoints(self):
        """The positive control: the endpoints must be using the helper.

        Without this the test passes on a module that stopped calling either
        function -- green, and reading nothing. The same silent retirement
        already happened once on this branch to the seeding scan.
        """
        tree = self._module_ast()
        used = [
            n.lineno
            for n in ast.walk(tree)
            if isinstance(n, ast.Call)
            and getattr(n.func, "id", None) == "_log_request_model"
        ]
        assert len(used) >= 4, f"only {len(used)} call sites use the helper: {used}"

    def test_no_call_site_dumps_before_the_guard(self):
        """The four sites this was written for, checked where they live.

        A helper nothing calls fixes nothing, and these sites sit in async
        route handlers no unit test reaches -- so the source is read instead,
        and an endpoint added later that spells it the old way is caught the
        moment it is written.
        """
        eager = self._eager_sites()
        assert not eager, (
            "_log_request_event is being handed a model_dump() built before "
            f"the guard can decline it: {eager}. Use _log_request_model."
        )


class TestPeriodicOutlivesItsStep:
    """`_periodic` is the one implementation of "a background task must not die
    silently", which is the invariant a second copy is how you lose.

    Nothing awaits these tasks until shutdown, so an escaping exception leaves
    only asyncio's "never retrieved": `/metrics` answers from the last
    snapshot, `metrics_snapshot_available` still reads 1, and the dashboard
    looks alive while every engine-derived series is frozen.
    """

    @staticmethod
    def _drive(step, rounds=4):
        """Run the real loop until `step` has been called `rounds` times.

        `CancelledError` is the stop signal rather than a timer, and the
        interval is zero, so this waits on nothing -- and it doubles as the
        check that shutdown still works, since a guard wide enough to swallow
        it would hang here instead of returning.
        """
        calls = []

        async def counted():
            calls.append(1)
            if len(calls) >= rounds:
                raise asyncio.CancelledError
            await step()

        counted.__name__ = getattr(step, "__name__", "step")
        _run_bounded(api_server._periodic(0, counted))
        return len(calls)

    def test_a_raising_step_does_not_end_the_loop(self):
        """A logging handler raising while it formats a warning is enough, and
        so is `MemoryError` under exactly the load this exists to watch."""

        async def boom():
            raise RuntimeError("handler raised while formatting")

        assert self._drive(boom) == 4

    def test_no_handler_here_is_wide_enough_to_swallow_cancellation(self):
        """Shutdown cancels these tasks and awaits them in the lifespan's
        `finally`, so a handler that caught the cancellation would hang the
        process rather than fail anything.

        Read from the source, because there is nothing to exercise:
        `CancelledError` is a BaseException, so `except Exception` cannot see
        it and a runtime check passes against any guard correct today. What can
        change is someone widening the clause, and that is a fact about text.
        """
        tree = ast.parse(textwrap.dedent(inspect.getsource(api_server._periodic)))
        caught = [
            ast.unparse(handler.type) if handler.type else "bare except"
            for node in ast.walk(tree)
            if isinstance(node, ast.Try)
            for handler in node.handlers
        ]

        assert caught == [
            "Exception"
        ], f"a handler here would swallow CancelledError and hang shutdown: {caught}"


class TestTheStepsOwnTheirOwnFailures:
    """One counter, one meaning. `metrics_refresh_errors` says the exported
    series are stale, so only the refresh may raise it -- riding both jobs on
    one loop is what conflated them, and they are two tasks now."""

    def test_a_failed_refresh_is_counted_and_not_only_logged(self, monkeypatch):
        """A log line is not a signal; nobody greps the frontend log to find
        out why a Grafana panel went flat."""
        errors = []
        monkeypatch.setattr(
            api_server, "engine", SimpleNamespace(get_metrics_statistics=_boom)
        )
        monkeypatch.setattr(
            api_server._metrics_exporter,
            "record_refresh_error",
            lambda: errors.append(1),
        )

        asyncio.run(api_server._refresh_metrics_once())

        assert errors == [1]

    def test_publishing_is_inside_the_guard_the_read_is(self, monkeypatch):
        """`update()` used to sit in an `else:` outside it, so a malformed
        snapshot -- the likely case when the engine is unhealthy -- escaped to
        the loop uncounted."""
        errors = []
        monkeypatch.setattr(
            api_server, "engine", SimpleNamespace(get_metrics_statistics=dict)
        )
        monkeypatch.setattr(api_server._metrics_exporter, "update", _boom)
        monkeypatch.setattr(
            api_server._metrics_exporter,
            "record_refresh_error",
            lambda: errors.append(1),
        )

        asyncio.run(api_server._refresh_metrics_once())

        assert errors == [1]

    def test_a_failing_gc_watch_does_not_claim_the_snapshot_went_stale(
        self, monkeypatch
    ):
        """The two answer to different readers. A gc watch says nothing about
        the exported snapshot, so counting its failure there pages whoever owns
        that alert for the one event the counter cannot name."""
        errors = []
        monkeypatch.setattr(api_server, "reclaim_watch", _boom)
        monkeypatch.setattr(
            api_server._metrics_exporter,
            "record_refresh_error",
            lambda: errors.append(1),
        )

        with pytest.raises(RuntimeError):
            asyncio.run(api_server._reclaim_watch_once())

        assert errors == [], "a gc failure incremented the stale-snapshot counter"

    def test_the_refresh_step_does_only_the_refresh(self, monkeypatch):
        """The boundary itself. Asserting the counter from `_reclaim_watch_once`
        cannot see the watch being called from inside the refresh, which is the
        arrangement that conflated them -- so this asks the refresh directly."""
        watched = []
        monkeypatch.setattr(
            api_server, "engine", SimpleNamespace(get_metrics_statistics=dict)
        )
        monkeypatch.setattr(api_server._metrics_exporter, "update", lambda _s: None)
        monkeypatch.setattr(
            api_server, "reclaim_watch", lambda ctx: watched.append(ctx)
        )

        asyncio.run(api_server._refresh_metrics_once())

        assert watched == [], "the refresh is running the gc watch again"


class TestLifespanOwnsTheBackgroundTasks:
    """Nothing else starts or stops them, and until now nothing checked that it
    does -- the tasks above are only reachable through here."""

    @staticmethod
    def _run(monkeypatch):
        started = []
        for name in ("tune_gc", "maybe_attach_gc_debug_callback", "freeze_gc_heap"):
            monkeypatch.setattr(api_server, name, lambda *a, **k: 0)
        monkeypatch.setattr(api_server, "arm_reclaim_watch", lambda: None)
        monkeypatch.setattr(api_server, "engine", None)
        # Long enough that neither step runs: this is about the tasks existing
        # and being cancelled, not about what they do.
        monkeypatch.setattr(api_server, "_METRICS_REFRESH_INTERVAL_SECONDS", 3600)
        monkeypatch.setattr(api_server, "_GC_WATCH_INTERVAL_SECONDS", 3600)

        async def drive():
            async with api_server.lifespan(api_server.app):
                started.extend(api_server._background_tasks)
                assert not any(t.done() for t in started)
            raise asyncio.CancelledError  # the loops above only end that way

        _run_bounded(drive())
        return started

    def test_both_jobs_get_a_task_of_their_own(self, monkeypatch):
        """Two cadences and two failure meanings, so two tasks. Riding one loop
        is what conflated the error counter."""
        tasks = self._run(monkeypatch)

        assert [t.get_name() for t in tasks] == [
            "_refresh_metrics_once",
            "_reclaim_watch_once",
        ]

    def test_shutdown_leaves_nothing_running(self, monkeypatch):
        """An uncancelled task keeps a reference to the whole app alive and
        goes on scraping an engine that is closing."""
        tasks = self._run(monkeypatch)

        assert all(t.cancelled() for t in tasks)
        assert api_server._background_tasks == []


class TestTheCensusEndpointKeepsTheLoopFree:
    """`/debug/gc_census` walks every tracked object -- seconds at a serving
    heap -- on the one loop that delivers every open SSE stream."""

    def test_the_walk_does_not_run_on_the_event_loop(self, monkeypatch):
        """Inline, a single unauthenticated GET stalls token delivery for every
        stream in flight, on a server whose product is inter-token latency.

        The thread identity is the assertion, not the presence of the call: a
        later refactor that awaits the walk directly would keep the executor
        import and lose the property.
        """
        seen = {}

        def fake_census(top, types_per_owner):
            seen["walk_thread"] = threading.current_thread()
            seen["args"] = (top, types_per_owner)
            return {"by_type": []}

        monkeypatch.setattr(api_server, "gc_census", fake_census)

        async def call():
            seen["loop_thread"] = threading.current_thread()
            return await api_server.get_gc_census(top=7, types_per_owner=3)

        assert asyncio.run(call()) == {"by_type": []}
        assert (
            seen["walk_thread"] is not seen["loop_thread"]
        ), "the census ran on the event loop"
        # Named, not merely "some other thread": the default pool would satisfy
        # that too, and the default pool is the one this must not borrow.
        assert seen["walk_thread"].name.startswith("gc-census"), seen[
            "walk_thread"
        ].name
        assert seen["args"] == (7, 3), "the query parameters did not reach the walk"

    def test_the_census_thread_is_not_a_thread_that_serves_requests(self):
        """`run_in_executor(None, ...)` takes the loop's default pool, which is
        where every request's prompt preprocessing runs. Handing a seconds-long
        heap walk to those threads moves the stall from the loop into admission
        rather than out of the way, and this endpoint has no authentication and
        no rate limit, so a curl loop is enough to fill them.
        """

        async def call():
            loop = asyncio.get_running_loop()
            census = await loop.run_in_executor(
                api_server._gc_census_executor, threading.current_thread
            )
            default = await loop.run_in_executor(None, threading.current_thread)
            return census, default

        census_thread, default_thread = asyncio.run(call())

        assert census_thread is not default_thread
        assert census_thread.name.startswith("gc-census")

    def test_concurrent_walks_do_not_each_pin_a_snapshot(self):
        """One worker is the single-flight. Each walk holds a full gen-2
        snapshot alive for its duration, so N concurrent GETs must be N walks
        in sequence rather than N snapshots at once -- and a lock on top of
        `max_workers=1` would add nothing, which is why there is not one.

        A barrier rather than a clock: with two workers the pair meets and the
        wait returns, with one it can only ever time out. Both outcomes are
        decided by the pool, not by how fast this machine is.
        """
        barrier = threading.Barrier(2)

        def met() -> bool:
            try:
                barrier.wait(timeout=0.15)
            except threading.BrokenBarrierError:
                return False
            return True

        async def call():
            loop = asyncio.get_running_loop()
            return await asyncio.gather(
                *[
                    loop.run_in_executor(api_server._gc_census_executor, met)
                    for _ in range(2)
                ]
            )

        assert asyncio.run(call()) == [False, False], "two walks ran at once"
