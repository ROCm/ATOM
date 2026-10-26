# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Token-bucket graphs that break at declared eager regions, and
request-bucket graphs that need no break, without torch.compile.

A forward is graphed one of two ways. A `TokenGraph` buckets it by its token
count, which leaves launches that depend on the requests to the host: those
are its breaks. A `ReqGraph` buckets it by its requests and their rows, which
fixes those launches too, so it records each bucket as one graph.

A module's forward is recorded once per token bucket as a `TokenGraph`
entry: the forward's Python runs under one stream capture, and every call
into an `eager_region` ends the current graph segment, runs the region for
real and starts the next segment. A replay runs no model Python at all -- it
walks the recorded steps: segment graphs, region calls, and the stream forks
and joins between them.

What decides where a forward breaks is a launch whose shape depends on more
than the bucket: a region that reads request lengths, context, or any other
host value it cannot take from the device. Such a region is called at replay
with the tensors it was recorded with -- graph-owned addresses -- and with
its per-step arguments (``live``) read afresh, so it serves whatever step is
running. Its result reaches the next segment through the address that segment
was recorded reading: either it wrote one of its inputs in place (`out=`), or
the replay copies it there.

A region declared with a stream runs beside the main one: the replay forks to
that stream before the call and the model's `join_stream` joins it back. Until
the join the region's inputs stay referenced, so no segment recorded in
between can be handed their memory.
"""

import functools
import inspect
import threading
import warnings
import weakref
from collections.abc import Callable
from typing import Any, ClassVar

import torch

from atom.distributed.nccl_watchdog import drain_nccl_works
from atom.utils import forward_context, weak_ref_tensor, weak_ref_tensors
from atom.utils.graph_holders import register_graph_holder

__all__ = [
    "ReqGraph",
    "TokenGraph",
    "eager_region",
    "forks",
    "graph_pool",
    "join_stream",
    "token_graph",
]

_local = threading.local()


def _recorder() -> "_Recorder | None":
    return getattr(_local, "recorder", None)


def _storage(t: torch.Tensor) -> int:
    return t.untyped_storage().data_ptr()


def _tensors(args, kwargs):
    return [a for a in (*args, *kwargs.values()) if isinstance(a, torch.Tensor)]


def _weak(value):
    """A region argument as a replay holds it: tensors (also inside a tuple,
    a named one keeping its type) without their storage, everything else as
    is."""
    if isinstance(value, tuple):
        items = [_weak(v) for v in value]
        return type(value)(*items) if hasattr(value, "_fields") else tuple(items)
    return weak_ref_tensor(value)


class _EagerStep:
    """One region call in a recording: same tensors, live context, result
    delivered to the address the next segment reads."""

    def __init__(self, fn, arguments, live, stream, target):
        self._fn, self._arguments, self._live = fn, arguments, live
        self._stream, self._target = stream, target

    def __call__(self):
        if self._stream is None:
            self._run()
            return
        with torch.cuda.stream(self._stream):
            self._run()

    def _run(self):
        arguments = self._arguments
        if self._live is not None:
            arguments = {**arguments, **self._live()}
        result = self._fn(**arguments)
        target = self._target
        if target is not None and result.data_ptr() != target.data_ptr():
            target.copy_(result)


class _Segments:
    """A recorded forward: a list of zero-argument steps and the output the
    last one leaves behind."""

    def __init__(self, steps: list[Callable[[], Any]], output):
        self._steps = steps
        self.output = output

    @property
    def num_segments(self) -> int:
        return sum(isinstance(s, _SegmentReplay) for s in self._steps)

    def replay(self):
        for step in self._steps:
            step()
        return self.output


class _SegmentReplay:
    def __init__(self, graph: torch.cuda.CUDAGraph):
        self.graph = graph

    def __call__(self):
        self.graph.replay()


# A replay runs on whatever stream the step runs on, not the one it was
# recorded on, so forks and joins name the current stream when they run.
def _fork(stream: torch.cuda.Stream) -> None:
    stream.wait_stream(torch.cuda.current_stream())


def _join(stream: torch.cuda.Stream) -> None:
    torch.cuda.current_stream().wait_stream(stream)


class _Recorder:
    def __init__(self, pool):
        self._pool = pool
        self._steps: list[Callable[[], Any]] = []
        self._graph: torch.cuda.CUDAGraph | None = None
        # Side stream -> the inputs and results of regions issued on it and not
        # joined yet. Held so no segment recorded meanwhile reuses that memory.
        self._pending: dict[torch.cuda.Stream, list] = {}

    def begin(self):
        self._graph = torch.cuda.CUDAGraph()
        self._graph.capture_begin(pool=self._pool, capture_error_mode="thread_local")

    def end(self):
        graph, self._graph = self._graph, None
        with warnings.catch_warnings():
            # Two breaks back to back leave an empty segment; it replays as
            # nothing, so it is not worth the warning.
            warnings.filterwarnings("ignore", "The CUDA Graph is empty")
            graph.capture_end()
        self._steps.append(_SegmentReplay(graph))

    def abort(self):
        if self._graph is not None:
            graph, self._graph = self._graph, None
            graph.capture_end()

    def eager(self, fn, args, kwargs, stream, live):
        self.end()
        if stream is not None:
            fork = functools.partial(_fork, stream)
            fork()
            self._steps.append(fork)
        _local.recorder = None
        try:
            with torch.cuda.stream(stream or torch.cuda.current_stream()):
                result = fn(*args, **kwargs)
        finally:
            _local.recorder = self
        self.begin()
        delivered = self._deliver(result, args, kwargs)
        bound = inspect.signature(fn).bind(*args, **kwargs)
        self._steps.append(
            _EagerStep(
                fn,
                {k: _weak(v) for k, v in bound.arguments.items()},
                live,
                stream,
                None if delivered is None else weak_ref_tensor(delivered),
            )
        )
        if stream is not None:
            self._pending.setdefault(stream, []).append((args, kwargs, delivered))
        return delivered

    @staticmethod
    def _deliver(result, args, kwargs):
        """The tensor the next segment reads the region's result from, which a
        replay copies into. A result written over one of the region's inputs is
        already in the pool. One the region allocated is not: the next
        segment's pool hands out an address in its place, so the pool can reuse
        it once nothing reads it, like any other activation. The recording
        reads it unwritten, as it reads every segment's output: segments are
        recorded, not run."""
        if result is None:
            return None
        if not isinstance(result, torch.Tensor):
            raise TypeError(
                f"an eager region returns a tensor or None, got {type(result)}"
            )
        inputs = {_storage(t) for t in _tensors(args, kwargs)}
        return result if _storage(result) in inputs else torch.empty_like(result)

    def join(self, streams):
        self.end()
        for stream in streams:
            join = functools.partial(_join, stream)
            join()
            self._steps.append(join)
            self._pending.pop(stream, None)
        self.begin()

    def finish(self, output) -> _Segments:
        self.end()
        if self._pending:
            raise RuntimeError(
                "a recorded forward left side-stream regions unjoined; "
                "call join_stream before the forward returns"
            )
        weak = output if output is None else weak_ref_tensors(output)
        return _Segments(self._steps, weak)


def _record_segments(fn: Callable, *args, pool=None, **kwargs) -> tuple[_Segments, Any]:
    """Run ``fn(*args, **kwargs)`` once under capture and return its segments
    and the output of that run. Call on a capture stream (not the default
    one), after a warmup run of the same shapes."""
    if _recorder() is not None:
        raise RuntimeError("recordings do not nest")
    recorder = _Recorder(pool)
    _local.recorder = recorder
    try:
        recorder.begin()
        output = fn(*args, **kwargs)
        return recorder.finish(output), output
    except BaseException:
        recorder.abort()
        raise
    finally:
        _local.recorder = None


def eager_region(
    stream: Callable[[Any], torch.cuda.Stream | None] | None = None,
    live: Callable[[], dict[str, Any]] | None = None,
):
    """Declare a method as a break in a recorded forward.

    ``stream`` maps the method's owner to the stream it runs on beside the
    main one, or None for the main stream. A caller joins that stream back
    with `join_stream` before it reads what the region wrote.

    ``live`` returns the arguments, by parameter name, that a replay must not
    take from the recording -- the step being served, not the one recorded.

    Not recording, the method is called directly; a side stream forks exactly
    as `side_stream` does, so a whole-forward graph captures the same fork.
    """

    def decorate(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            if torch.compiler.is_compiling():
                return fn(*args, **kwargs)
            side = None if stream is None else stream(args[0])
            recorder = _recorder()
            if recorder is not None:
                return recorder.eager(fn, args, kwargs, side, live)
            with forward_context.side_stream(side):
                return fn(*args, **kwargs)

        return wrapper

    return decorate


def forks(stream: torch.cuda.Stream | None) -> bool:
    """Whether a region declared on ``stream`` runs beside the main stream
    in this forward -- recorded, or inside a whole-forward capture."""
    if stream is None:
        return False
    return _recorder() is not None or forward_context.get_forward_context().in_hipgraph


def join_stream(*streams: torch.cuda.Stream | None) -> None:
    """The main stream waits for everything issued on ``streams``; only those
    a region forked to (`forks`) are joined."""
    joined = [s for s in streams if forks(s)]
    if not joined:
        return
    recorder = _recorder()
    if recorder is not None:
        recorder.join(joined)
        return
    main = forward_context.get_forward_context().main_stream
    for stream in joined:
        main.wait_stream(stream)


def _record_whole(fn: Callable, *args, pool=None, **kwargs) -> tuple[_Segments, Any]:
    """`_record_segments` with no break: every region is recorded inline, as a
    whole-forward capture records it."""
    if _recorder() is not None:
        raise RuntimeError("recordings do not nest")
    graph = torch.cuda.CUDAGraph()
    # thread_local, as `_Recorder.begin` captures: "global" invalidates a
    # capture when any thread makes an unsafe HIP call, and the NCCL watchdog
    # polls `hipEventQuery` every ~100 ms.
    with torch.cuda.graph(
        graph,
        pool,
        stream=torch.cuda.current_stream(),
        capture_error_mode="thread_local",
    ):
        output = fn(*args, **kwargs)
    weak = output if output is None else weak_ref_tensors(output)
    return _Segments([_SegmentReplay(graph)], weak), output


def graph_pool():
    """The one pool every `TokenGraph` and `ReqGraph` records into."""
    if TokenGraph._pool is None:
        TokenGraph._pool = torch.cuda.graph_pool_handle()
    return TokenGraph._pool


def _retire_pool(pool) -> None:
    """A holder that recorded into ``pool`` is gone, its recordings with it.
    Should they have been the pool's last while memory it handed out is still
    referenced, the pool cannot be recorded into again: later recordings take
    a fresh one. The holders still alive keep theirs."""
    if TokenGraph._pool == pool:
        TokenGraph._pool = None


class TokenGraph:
    """A module's forward recorded once per token bucket.

    The bucket is the shapes of the tensor arguments, and for a forward that
    records differently in prefill and decode (`token_graph(per_phase=True)`)
    the phase too; a recording without a phase serves both. Tensor arguments are
    copied into inputs this holds, so a caller may hand any tensor of the
    bucket's shape; one already at the recorded address is not copied. Every
    bucket's input is a leading slice of one buffer per argument, sized by the
    first bucket recorded -- so record the largest first.

    Every bucket of every module records into one pool (`graph_pool`), a
    `ReqGraph`'s included. Recordings never overlap -- one module's run in
    stream order, a module's output consumed before the next one replays --
    the same as two replays of one bucket, so the pool is the largest
    recording's, not their sum.
    """

    _pool = None
    # Each declared region is a break (`eager_region`).
    _record = staticmethod(_record_segments)
    # The live graphs of this class, not of a subclass (`captured_sizes`).
    _live: ClassVar["weakref.WeakSet[TokenGraph]"] = weakref.WeakSet()
    # phase -> `captured_sizes`, read every step: whatever changes a live
    # graph's `_captured`, or which graphs live, clears it.
    _sizes: ClassVar[dict[str | None, tuple[int, ...]]] = {}

    def __init__(self):
        # (phase or None, bucket) -> recording.
        self._buckets: dict[tuple, tuple[_Segments, tuple, dict]] = {}
        # (phase or None, bucket) -> the phases whose record pass asked for it:
        # a recording without a phase serves both, but only a phase that
        # records counts it among its own (`captured_sizes`).
        self._captured: dict[tuple, set[str]] = {}
        # Argument position or name -> the buffer every bucket's input slices.
        self._buffers: dict[int | str, torch.Tensor] = {}
        # The pools this records into, each retired when this goes.
        self._pools: set = set()
        type(self)._live.add(self)
        weakref.finalize(self, TokenGraph._sizes.clear)
        register_graph_holder(self)

    @staticmethod
    def captured_sizes(phase: str | None = None) -> tuple[int, ...]:
        """The token buckets a step may pad to, ascending: the rows of every
        bucket a live `TokenGraph` holds a recording of -- of a ``phase`` step,
        those a ``phase`` record pass asked for."""
        sizes = TokenGraph._sizes.get(phase)
        if sizes is None:
            sizes = TokenGraph._sizes[phase] = tuple(
                sorted(
                    {
                        key[0][0]
                        for graph in TokenGraph._live
                        for (_, key), phases in graph._captured.items()
                        if phase is None or phase in phases
                    }
                )
            )
        return sizes

    @staticmethod
    def bucket(tokens: int, phase: str, multiple_of: int = 1) -> int | None:
        """The smallest ``phase`` bucket (`captured_sizes`) that holds
        ``tokens`` rows and is a multiple of ``multiple_of``, None when none
        is recorded."""
        return next(
            (
                n
                for n in TokenGraph.captured_sizes(phase)
                if n >= tokens and n % multiple_of == 0
            ),
            None,
        )

    def mark_captured(self, key, phase: str | None = None) -> None:
        """Count ``key``'s recording under ``phase`` among the record pass's
        own, as a pass of the step's phase that finds it already held."""
        self._captured[(phase, key)].add(_phase())
        TokenGraph._sizes.clear()

    def release_graphs(self) -> int:
        dropped = len(self._buckets)
        self._buckets.clear()
        self._captured.clear()
        TokenGraph._sizes.clear()
        TokenGraph._pool = None
        self._buffers.clear()
        return dropped

    @staticmethod
    def key(args, kwargs) -> tuple:
        return tuple(tuple(t.shape) for t in _tensors(args, kwargs))

    def has(self, key, phase: str | None = None) -> bool:
        return (phase, key) in self._buckets

    def __contains__(self, key) -> bool:
        return self.has(key)

    def keys(self):
        return list(dict.fromkeys(key for _, key in self._buckets))

    def inputs(self, key, phase: str | None = None) -> tuple:
        """The tensors the recording of ``key`` reads its positional arguments
        from. A caller that writes them in place and hands them to `replay`
        spares it the copy in."""
        return self._buckets[(phase, key)][1]

    def record(self, forward: Callable, args=(), kwargs=None, key=None, phase=None):
        """``forward`` recorded under the bucket of ``args``, or under ``key``
        for a caller that names its own bucket, and under ``phase``."""
        kwargs = kwargs or {}
        key = self.key(args, kwargs) if key is None else key
        inputs = tuple(self._input(i, a) for i, a in enumerate(args))
        kw = {k: self._input(k, v) for k, v in kwargs.items()}
        pool = graph_pool()
        if pool not in self._pools:
            self._pools.add(pool)
            weakref.finalize(self, _retire_pool, pool)
        drain_nccl_works()
        segments, output = self._record(forward, *inputs, pool=pool, **kw)
        self._buckets[(phase, key)] = (segments, inputs, kw)
        self._captured[(phase, key)] = {_phase()}
        TokenGraph._sizes.clear()
        return output

    def _input(self, slot, value):
        """``value`` copied into a leading slice of ``slot``'s buffer."""
        if not isinstance(value, torch.Tensor):
            return value
        buffer = self._buffers.get(slot)
        if (
            buffer is None
            or buffer.dtype != value.dtype
            or buffer.shape[1:] != value.shape[1:]
            or buffer.shape[0] < value.shape[0]
        ):
            buffer = self._buffers[slot] = value.clone()
        view = buffer[: value.shape[0]]
        view.copy_(value)
        return view

    def replay(self, args=(), kwargs=None, key=None, phase=None):
        """The bucket of ``args`` replayed, or the one ``key`` names, as
        recorded under ``phase``."""
        kwargs = kwargs or {}
        segments, inputs, kw = self._buckets[
            (phase, self.key(args, kwargs) if key is None else key)
        ]
        for src, dst in zip(
            (*args, *kwargs.values()), (*inputs, *(kw[k] for k in kwargs))
        ):
            if isinstance(src, torch.Tensor) and src.data_ptr() != dst.data_ptr():
                dst.copy_(src)
        return segments.replay()


class ReqGraph(TokenGraph):
    """A forward recorded once per request bucket, the ``key`` its caller
    names (`request_bucket`): the step's request count and the draft tokens
    each verifies.

    A request bucket fixes what a token bucket leaves to the host -- the
    requests and their rows -- so a region declared as a token graph's break
    is recorded inline here: a bucket is one graph.
    """

    _record = staticmethod(_record_whole)
    # Its own set, so request buckets stay out of `captured_sizes`.
    _live: ClassVar["weakref.WeakSet[ReqGraph]"] = weakref.WeakSet()


def token_graph(cls: type | None = None, *, per_phase: bool = False):
    """Declare ``cls`` a model's recording unit: its calls are recorded once
    per token bucket and replayed (`_run_token_graph`).

    Everything the forward runs outside an `eager_region` lands in its
    segments, so the unit is the outermost module whose forward is all the
    step's per-token work -- a collective or an ``input_ids`` rewrite left in
    a caller above it runs eagerly on every replay, and whatever the recording
    read in its place is what the replay reads.

    A recording serves prefill and decode steps alike unless ``per_phase``:
    a forward that records different work in each (a tensor one phase
    computes and the other does not) is recorded once per phase, and a phase
    replays only its own."""

    def wrap(cls: type) -> type:
        call = cls.__call__

        @functools.wraps(call)
        def __call__(self, *args, **kwargs):
            if torch.compiler.is_compiling():
                return call(self, *args, **kwargs)
            return _run_token_graph(
                self, functools.partial(call, self), args, kwargs, per_phase
            )

        cls.__call__ = __call__
        return cls

    return wrap if cls is None else wrap(cls)


def _phase() -> str | None:
    """The phase a recording of this step is keyed under (`token_graph`), and
    the phase its record pass captures for (`captured_sizes`); None off a step.
    A rank decoding beside a prefilling DP peer replays the prefill recording
    the group settled on (`ForwardMode.prefill_token_graph`)."""
    context = getattr(forward_context.get_forward_context(), "context", None)
    if context is None:
        return None
    mode = getattr(context, "forward_mode", None)
    if context.is_prefill or (mode is not None and mode.prefill_token_graph):
        return "prefill"
    return "decode"


def _run_token_graph(owner, forward: Callable, args, kwargs, per_phase=False):
    """``owner``'s forward under the step's graph mode
    (`ForwardContext.tr_graph`): recorded once per token bucket (and phase,
    when ``per_phase``) in a "record" pass, replayed where its bucket was
    recorded in a "replay" one, and run directly otherwise."""
    mode = getattr(forward_context.get_forward_context(), "tr_graph", None)
    if mode is None or _recorder() is not None:
        return forward(*args, **kwargs)
    graph = owner.__dict__.get("_token_graph")
    if graph is None:
        graph = owner.__dict__["_token_graph"] = TokenGraph()
    key = TokenGraph.key(args, kwargs)
    phase = _phase() if per_phase else None
    if graph.has(key, phase):
        if mode == "replay":
            return graph.replay(args, kwargs, phase=phase)
        if mode == "record":
            graph.mark_captured(key, phase)
    elif mode == "record":
        return graph.record(forward, args, kwargs, phase=phase)
    return forward(*args, **kwargs)
