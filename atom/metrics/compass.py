"""Per-rank forward and per-op GPU timing records for ATOM Compass.

Enabled by ``ATOM_COMPASS_DIR``. Each model-runner rank writes, under
``$ATOM_COMPASS_DIR/dp{d}_pp{p}_tp{t}/``:

- ``forward.<pid>.jsonl``: one row per step (``phase`` target) and per
  drafter pass (``draft_kv``, ``draft``) -- device-event duration, absolute
  GPU start/end (wall-clock ns, via an anchor event), and the batch shape
  (graph padding, prefill/decode split, context length sum/max; per-sequence
  query/context lengths on tp_rank 0 only). A target row's ``scope`` says what
  its interval covers: ``forward`` is the whole ``ModelRunner.forward`` (input
  preparation, model, drafter passes, sampling), with the ``run_model`` part
  in ``model_gpu_*``; ``call`` is the timed call alone, for a ``run_model``
  reached outside ``forward`` (RapidServe ``prefill_forward``). Drafter rows
  inside a forward share its ``step``.
- ``ops.<pid>.jsonl``: during sampled profiler windows, one row per op that
  launched GPU work inside a step: op name, input shapes/dtypes, CPU and
  GPU start/end (wall-clock ns) and its kernels. ``gpu_kernel_ns`` sums the
  kernels' durations; ``gpu_busy_ns`` is the union of their intervals, so
  kernels overlapping on several streams count once. A CUDA graph replay
  launches its kernels without ops, so each replayed kernel is named after
  the op that launched a kernel of the same name in eager runs (the kernel
  map, below); consecutive kernels of one op form one row, with the graph
  launch's CPU times and no shapes. ``op_from`` says where the name came
  from: ``kernel_map``, ``kernel_rule`` (graph-only kernels such as the
  custom all-reduce) or ``kernel`` (unknown: the kernel's own name);
  ``op_alternatives`` lists the other ops seen launching that kernel.
- ``windows.<pid>.jsonl``: one row per op window -- why it opened, the batch
  bucket, the steps it holds ops for, and how many replayed graph kernels got
  an op name.
- ``graphs.<pid>.jsonl``: one row per graph whose eager warmup fed the
  kernel map: the graph's ``(bs, max_q_len)`` name and its kernel counts.
- ``meta.<pid>.json``: rank identity and collection settings.

Op windows are scheduled per batch bucket (prefill/decode, power-of-two batch
size, prefill tokens and max context length): a bucket gets a window once it
has been seen more than ``ATOM_COMPASS_OP_MIN_SEEN`` times (so first-call JIT
and warmup stay out), until it has ``ATOM_COMPASS_OP_PER_BUCKET`` windows;
every ``ATOM_COMPASS_OP_EVERY``-th step also opens one. No window opens
within ``ATOM_COMPASS_OP_MIN_GAP_S`` seconds of the previous one closing. A
window profiles one step more than ``ATOM_COMPASS_OP_STEPS`` and drops that
tail step's ops: the profiler loses most of the last step's kernels when it
stops.

The kernel map counts, per kernel name, the ops that launched it: in the
eager warmup forward before a few graph captures (power-of-two batch sizes
and the largest of each ``max_q_len``; GEMM tile choice follows the batch
size) and in every eager op of an op window.

Forward rows never synchronize the device except for re-anchoring the GPU
clock every ``ATOM_COMPASS_ANCHOR_S`` seconds. Op windows run the torch
profiler with ``record_shapes`` and synchronize once when each window closes.
"""

import bisect
import json
import logging
import os
import socket
import time
from collections import deque
from contextlib import contextmanager, nullcontext
from functools import wraps

import torch
from torch.profiler import ProfilerActivity, profile, record_function

from atom.utils import envs
from atom.utils.forward_context import get_forward_context

logger = logging.getLogger("atom")

SCHEMA_VERSION = 3

_STEP_PREFIX = "compass_step:"
_GRAPH_PREFIX = "compass_graph:"
_WARMUP_MARK = "compass_graph_warmup"
_RUNTIME_PREFIXES = ("hip", "cuda", "rocprof")
_MAX_PENDING = 1024
_FLUSH_INTERVAL_S = 1.0
# An op window still open after this long without a forward is closed by the
# engine's idle poll, so its rows land without waiting for more traffic or a
# graceful shutdown.
_WINDOW_IDLE_CLOSE_S = 2.0
# The profiler drops the trailing GPU activity it has not received when it
# stops; a warmup profile idles this long after its last kernel first.
_WARMUP_DRAIN_S = 0.05
# Ops for kernels that only graph replays launch, so no eager run maps them.
_KERNEL_RULES = (
    ("all_reduce", ("all_reduce", "allreduce", "ar_ll", "cross_device_reduce")),
    ("memcpy", ("copyBuffer", "Memcpy", "Memset")),
)


def _pow2(n):
    return 1 << max(0, int(n) - 1).bit_length()


def batch_bucket(batch):
    """The op-window bucket of a scheduled batch, or None for an empty one."""
    if batch is None or not batch.total_seqs_num:
        return None
    prefill = batch.total_tokens_num_prefill > 0
    return (
        "prefill" if prefill else "decode",
        _pow2(batch.total_seqs_num),
        _pow2(batch.total_tokens_num) if prefill else 0,
        _pow2(batch.context_lens.max(initial=0)),
    )


def graph_name(key):
    bs, max_q_len = key
    return f"bs{bs}_q{max_q_len}"


def _batch_fields(batch, context, seq_lists):
    row = {
        "is_prefill": bool(context.is_prefill),
        "is_dummy": bool(context.is_dummy_run),
        "use_cudagraph": bool(
            context.forward_mode is not None and context.forward_mode.use_cudagraph
        ),
        "scheduled_bs": int(context.scheduled_bs),
        "scheduled_tokens": int(context.scheduled_tokens),
        "running_bs": int(context.running_bs),
        "running_tokens": int(context.running_tokens),
    }
    if batch is None:
        return row
    row.update(
        num_prefill_seqs=int(batch.total_seqs_num_prefill),
        num_decode_seqs=int(batch.total_seqs_num_decode),
        num_prefill_tokens=int(batch.total_tokens_num_prefill),
        num_decode_tokens=int(batch.total_tokens_num_decode),
        num_tokens=int(batch.total_tokens_num),
        num_spec_step=int(batch.num_spec_step),
        sum_context_len=int(batch.context_lens.sum()),
        max_context_len=int(batch.context_lens.max(initial=0)),
    )
    if seq_lists:
        row.update(
            query_lens=batch.num_scheduled_tokens.tolist(),
            context_lens=batch.context_lens.tolist(),
            cached_lens=[int(n) for n in batch.num_cached_tokens],
        )
    return row


class CompassRecorder:
    def __init__(
        self,
        out_dir,
        *,
        dp_rank=0,
        pp_rank=0,
        tp_rank=0,
        engine_role="default",
        is_profiler_busy=lambda: False,
        op_steps=None,
        op_every=None,
        op_max_windows=None,
        op_min_seen=None,
        op_per_bucket=None,
        op_min_gap_s=None,
        graph_ops=None,
        anchor_interval_s=None,
    ):
        self.rank = {
            "dp_rank": dp_rank,
            "pp_rank": pp_rank,
            "tp_rank": tp_rank,
            "engine_role": engine_role,
        }
        self.op_steps = envs.ATOM_COMPASS_OP_STEPS if op_steps is None else op_steps
        self.op_every = envs.ATOM_COMPASS_OP_EVERY if op_every is None else op_every
        self.op_max_windows = (
            envs.ATOM_COMPASS_OP_MAX_WINDOWS
            if op_max_windows is None
            else op_max_windows
        )
        self.op_min_seen = (
            envs.ATOM_COMPASS_OP_MIN_SEEN if op_min_seen is None else op_min_seen
        )
        self.op_per_bucket = (
            envs.ATOM_COMPASS_OP_PER_BUCKET if op_per_bucket is None else op_per_bucket
        )
        self.op_min_gap_s = (
            envs.ATOM_COMPASS_OP_MIN_GAP_S if op_min_gap_s is None else op_min_gap_s
        )
        self.graph_ops = envs.ATOM_COMPASS_GRAPH_OPS if graph_ops is None else graph_ops
        self.anchor_interval_s = (
            envs.ATOM_COMPASS_ANCHOR_S
            if anchor_interval_s is None
            else anchor_interval_s
        )
        self.is_profiler_busy = is_profiler_busy

        rank_dir = os.path.join(out_dir, f"dp{dp_rank}_pp{pp_rank}_tp{tp_rank}")
        os.makedirs(rank_dir, exist_ok=True)
        pid = os.getpid()
        # Open for the recorder's lifetime; close() releases them.
        path = os.path.join(rank_dir, f"forward.{pid}.jsonl")
        self.forward_file = open(path, "a")  # noqa: SIM115
        path = os.path.join(rank_dir, f"ops.{pid}.jsonl")
        self.ops_file = open(path, "a")  # noqa: SIM115
        path = os.path.join(rank_dir, f"windows.{pid}.jsonl")
        self.windows_file = open(path, "a")  # noqa: SIM115
        path = os.path.join(rank_dir, f"graphs.{pid}.jsonl")
        self.graphs_file = open(path, "a")  # noqa: SIM115
        # One newline-terminated line, so /compass/records serves it whole.
        with open(os.path.join(rank_dir, f"meta.{pid}.json"), "w") as f:
            meta = {
                **self.rank,
                "schema_version": SCHEMA_VERSION,
                "host": socket.gethostname(),
                "pid": pid,
                "device": torch.cuda.get_device_name(),
                "torch": torch.__version__,
                "op_steps": self.op_steps,
                "op_every": self.op_every,
                "op_max_windows": self.op_max_windows,
                "op_min_seen": self.op_min_seen,
                "op_per_bucket": self.op_per_bucket,
                "op_min_gap_s": self.op_min_gap_s,
                "graph_ops": self.graph_ops,
                "anchor_interval_s": self.anchor_interval_s,
            }
            f.write(json.dumps(meta) + "\n")
        self.last_flush = time.monotonic()

        self.step = 0
        self.active = None  # the step being timed: row, nested spans, dropped
        self.pending = deque()
        self.free = []
        self.anchors = deque()  # (event, wall ns), oldest first
        self._reanchor()

        # TP peers run the same batch; only tp0 of each group lists every
        # sequence, the rest keep the scalar summary.
        self.seq_lists = tp_rank == 0
        self.last_measure = time.monotonic()
        self.window = None  # active torch profiler
        self.window_info = None  # reason, bucket and steps of the active window
        self.windows_done = 0
        self.bucket_seen = {}
        self.bucket_windows = {}
        self.window_closed_at = float("-inf")
        self.kernel_ops = {}  # kernel name -> {op name: kernels seen}
        self.warmup_q_lens = set()  # max_q_len whose largest graph was profiled
        logger.info(
            "Compass recording to %s (op window %d steps every %d, "
            "%d per bucket after %d sightings, %.0fs apart, max %s)",
            rank_dir,
            self.op_steps,
            self.op_every,
            self.op_per_bucket,
            self.op_min_seen,
            self.op_min_gap_s,
            self.op_max_windows or "unlimited",
        )

    # -- GPU clock anchor -------------------------------------------------

    def _reanchor(self):
        # The only device synchronization outside op windows: with an idle
        # device the anchor runs as soon as it is recorded, so its GPU time is
        # the wall clock read right before. Event elapsed times are float32
        # milliseconds, so short re-anchor intervals keep microsecond precision.
        torch.cuda.synchronize()
        event = torch.cuda.Event(enable_timing=True)
        wall_ns = time.time_ns()
        event.record()
        self.anchors.append((event, wall_ns))
        self.anchor_at = time.monotonic()

    def _wall_ns(self, anchor, event):
        anchor_event, anchor_ns = anchor
        return anchor_ns + int(anchor_event.elapsed_time(event) * 1e6)

    # -- forward rows -----------------------------------------------------

    def _event_pair(self):
        if self.free:
            return self.free.pop()
        return (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )

    def poll(self):
        while self.pending:
            row, anchor, spans = self.pending[0]
            if not all(end.query() for _, _, end in spans):
                break
            self.pending.popleft()
            for prefix, start, end in spans:
                row[f"{prefix}gpu_start_ns"] = self._wall_ns(anchor, start)
                row[f"{prefix}gpu_end_ns"] = self._wall_ns(anchor, end)
                row[f"{prefix}gpu_ms"] = start.elapsed_time(end)
                self.free.append((start, end))
            self.forward_file.write(json.dumps(row) + "\n")
        # Pending rows hold anchors oldest first; drop the ones none refer to.
        oldest = self.pending[0][1] if self.pending else self.anchors[-1]
        while self.anchors[0] is not oldest:
            self.anchors.popleft()
        if (
            self.window is not None
            and time.monotonic() - self.last_measure > _WINDOW_IDLE_CLOSE_S
        ):
            self._close_window()
        if time.monotonic() - self.last_flush > _FLUSH_INTERVAL_S:
            self.flush()

    def flush(self):
        self.forward_file.flush()
        self.ops_file.flush()
        self.windows_file.flush()
        self.graphs_file.flush()
        self.last_flush = time.monotonic()

    @contextmanager
    def measure_forward(self, batch=None):
        """Time one ``ModelRunner.forward`` as the step. Its batch fields come
        from the ``run_model`` inside, once the forward context exists;
        ``batch`` only picks the step's op-window bucket."""
        if torch.cuda.is_current_stream_capturing() or self.active is not None:
            yield
            return
        with self._step("target", "forward", batch):
            yield

    @contextmanager
    def measure(self, batch, phase="target"):
        """Time one GPU phase: the target ``run_model``, or a drafter's
        ``draft_kv`` / ``draft`` pass (timed at its runner call site). Inside a
        timed forward it is a span of that step; otherwise a step of its own."""
        if torch.cuda.is_current_stream_capturing():
            # Events recorded into a graph would time replays, not this call.
            yield
            return
        if self.active is None:
            with self._step(phase, "call", batch):
                yield
            return
        active = self.active
        if active["dropped"]:
            yield
            return
        context = get_forward_context().context
        start, end = self._event_pair()
        cpu_start_ns = time.time_ns()
        start.record()
        try:
            yield
        except BaseException:
            self.free.append((start, end))
            raise
        end.record()
        fields = _batch_fields(batch, context, self.seq_lists)
        if phase == "target":
            active["row"].update(fields)
            active["spans"].append(("model_", start, end))
            return
        row = {
            "step": active["row"]["step"],
            "phase": phase,
            "scope": "call",
            **self.rank,
            **fields,
            "cpu_start_ns": cpu_start_ns,
            "cpu_end_ns": time.time_ns(),
            "op_window": self.window is not None,
        }
        self.pending.append((row, self.anchors[-1], [("", start, end)]))

    @contextmanager
    def _step(self, phase, scope, batch):
        self.poll()
        if time.monotonic() - self.anchor_at > self.anchor_interval_s:
            self._reanchor()
        self.last_measure = time.monotonic()
        step = self.step
        self.step += 1
        self._maybe_open_window(step, batch)
        if self.window is not None:
            self.window_info["steps"].append(step)

        row = {"step": step, "phase": phase, "scope": scope, **self.rank}
        if scope == "call":
            context = get_forward_context().context
            row.update(_batch_fields(batch, context, self.seq_lists))
        active = self.active = {"row": row, "spans": [], "dropped": False}
        if len(self.pending) >= _MAX_PENDING:
            # Never block the forward on the recorder; drop the row instead.
            logger.warning(
                "Compass: %d pending forward rows, dropping step %d",
                len(self.pending),
                step,
            )
            active["dropped"] = True
            try:
                yield
            finally:
                self.active = None
            self._maybe_close_window(step)
            return
        start, end = self._event_pair()
        annotate = (
            record_function(f"{_STEP_PREFIX}{step}")
            if self.window is not None
            else nullcontext()
        )
        row["cpu_start_ns"] = time.time_ns()
        start.record()
        try:
            with annotate:
                yield
        except BaseException:
            self.free.append((start, end))
            self.free.extend((s, e) for _, s, e in active["spans"])
            raise
        finally:
            self.active = None
        end.record()
        row["cpu_end_ns"] = time.time_ns()
        row["op_window"] = self.window is not None
        self.pending.append(
            (row, self.anchors[-1], [("", start, end), *active["spans"]])
        )
        self._maybe_close_window(step)

    # -- op windows -------------------------------------------------------

    def _window_reason(self, step, bucket):
        if bucket is None:
            return None
        seen = self.bucket_seen[bucket] = self.bucket_seen.get(bucket, 0) + 1
        if (
            self.window is not None
            or self.op_steps <= 0
            or seen <= self.op_min_seen
            or (self.op_max_windows and self.windows_done >= self.op_max_windows)
            or time.monotonic() - self.window_closed_at < self.op_min_gap_s
            or self.is_profiler_busy()
        ):
            return None
        if self.bucket_windows.get(bucket, 0) < self.op_per_bucket:
            return "bucket"
        if step % self.op_every == 0:
            return "periodic"
        return None

    def _maybe_open_window(self, step, batch):
        bucket = batch_bucket(batch)
        reason = self._window_reason(step, bucket)
        if reason is None:
            return
        self.bucket_windows[bucket] = self.bucket_windows.get(bucket, 0) + 1
        self.window = profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=True,
        )
        self.window.__enter__()
        self.window_info = {"reason": reason, "bucket": list(bucket), "steps": []}

    def _maybe_close_window(self, step):
        if self.window is not None and len(self.window_info["steps"]) > self.op_steps:
            self._close_window(tail=step)

    def _close_window(self, tail=None):
        began = time.perf_counter()
        torch.cuda.synchronize()  # kernels still in flight would be lost
        window, self.window = self.window, None
        info, self.window_info = self.window_info, None
        window.__exit__(None, None, None)
        rows = [
            r
            for r in window_op_rows(
                window.profiler.kineto_results.events(), self.kernel_ops
            )
            if r["step"] != tail
        ]
        for row in rows:
            self.ops_file.write(json.dumps({**self.rank, **row}) + "\n")
        graph_rows = [r for r in rows if r["graph_replay"]]
        summary = {
            **self.rank,
            "window": self.windows_done,
            "reason": info["reason"],
            "bucket": info["bucket"],
            "steps": [s for s in info["steps"] if s != tail],
            "tail_step": tail,
            "op_rows": len(rows),
            "graph_kernels": sum(len(r["kernels"]) for r in graph_rows),
            "graph_kernels_named": sum(
                len(r["kernels"]) for r in graph_rows if r["op_from"] != "kernel"
            ),
            "close_s": time.perf_counter() - began,
        }
        self.windows_file.write(json.dumps(summary) + "\n")
        self.windows_done += 1
        self.window_closed_at = time.monotonic()
        self.flush()
        logger.info(
            "Compass op window %d (%s %s, steps %s): %d op rows, "
            "%d/%d graph kernels named, in %.2fs",
            summary["window"],
            summary["reason"],
            summary["bucket"],
            summary["steps"][:1],
            len(rows),
            summary["graph_kernels_named"],
            summary["graph_kernels"],
            summary["close_s"],
        )

    # -- CUDA graph kernel map ----------------------------------------------

    @contextmanager
    def graph_warmup(self, key):
        """Profile the eager warmup forward before ``key``'s graph capture into
        the kernel map, for power-of-two batch sizes and the first (largest)
        graph of each ``max_q_len``."""
        bs, max_q_len = key
        if (
            not self.graph_ops
            or self.op_steps <= 0
            or (bs & (bs - 1) and max_q_len in self.warmup_q_lens)
            or self.window is not None
            or self.is_profiler_busy()
            or torch.cuda.is_current_stream_capturing()
        ):
            yield
            return
        self.warmup_q_lens.add(max_q_len)
        torch.cuda.synchronize()
        prof = profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA])
        prof.__enter__()
        try:
            with record_function(_WARMUP_MARK):
                yield
            torch.cuda.synchronize()
            time.sleep(_WARMUP_DRAIN_S)
        finally:
            prof.__exit__(None, None, None)
        known = len(self.kernel_ops)
        kernels = with_op = 0
        for k, _launch, op, marks in _attributed_kernels(
            prof.profiler.kineto_results.events(), (_WARMUP_MARK,)
        ):
            if marks[_WARMUP_MARK] is None:
                continue
            kernels += 1
            if op is not None:
                with_op += 1
                learn_kernel_op(self.kernel_ops, k.name(), op.name())
        self.graphs_file.write(
            json.dumps(
                {
                    **self.rank,
                    "graph": graph_name(key),
                    "kernels": kernels,
                    "kernels_with_op": with_op,
                    "new_kernel_names": len(self.kernel_ops) - known,
                }
            )
            + "\n"
        )
        self.graphs_file.flush()

    def graph_replay(self, key):
        """Mark a graph replay inside an op window with the graph's name."""
        if self.window is None:
            return nullcontext()
        return record_function(f"{_GRAPH_PREFIX}{graph_name(key)}")

    def close(self):
        try:
            if self.window is not None:
                self._close_window()
            torch.cuda.synchronize()
            self.poll()
            self.flush()
        finally:
            self.forward_file.close()
            self.ops_file.close()
            self.windows_file.close()
            self.graphs_file.close()


class _Spans:
    """CPU annotation intervals with one prefix, looked up by a launch time."""

    def __init__(self, spans):
        self.spans = sorted(spans)
        self.starts = [s[0] for s in self.spans]

    def find(self, t):
        i = bisect.bisect_right(self.starts, t) - 1
        if i < 0 or t > self.spans[i][1]:
            return None
        return self.spans[i][2]


def _attributed_kernels(events, prefixes):
    """Each kernel with its runtime launch, its launching op (or None) and the
    innermost-starting annotation of each prefix that holds the launch.

    A kernel shares its correlation id with the runtime launch call and links
    (``linked_correlation_id``) to the innermost CPU op enclosing that launch,
    which carries the input shapes.
    """
    runtime, ops, kernels = {}, {}, []
    spans = {p: [] for p in prefixes}
    for e in events:
        if e.device_type() != torch.autograd.DeviceType.CPU:
            if not e.is_user_annotation():
                kernels.append(e)
            continue
        name = e.name()
        prefix = next((p for p in prefixes if name.startswith(p)), None)
        if prefix is not None:
            spans[prefix].append((e.start_ns(), e.end_ns(), name[len(prefix) :]))
        elif name.startswith(_RUNTIME_PREFIXES):
            runtime[e.correlation_id()] = e
        elif not e.is_user_annotation():
            ops[e.correlation_id()] = e
    lookup = {p: _Spans(s) for p, s in spans.items()}
    out = []
    for k in kernels:
        launch = runtime.get(k.correlation_id())
        if launch is None:
            continue
        op = ops.get(k.linked_correlation_id()) if k.linked_correlation_id() else None
        marks = {p: lookup[p].find(launch.start_ns()) for p in prefixes}
        out.append((k, launch, op, marks))
    out.sort(key=lambda a: a[0].start_ns())
    return out


def learn_kernel_op(kernel_ops, kernel, op):
    ops = kernel_ops.setdefault(kernel, {})
    ops[op] = ops.get(op, 0) + 1


def kernel_op(kernel_ops, kernel):
    """``(op, op_from, alternatives)`` for a kernel replayed without its op:
    the op most often seen launching it, else a rule, else its own name."""
    ops = kernel_ops.get(kernel)
    if ops:
        ranked = sorted(ops, key=lambda o: (-ops[o], o))
        return ranked[0], "kernel_map", ranked[1:]
    for op, needles in _KERNEL_RULES:
        if any(n in kernel for n in needles):
            return op, "kernel_rule", []
    return kernel, "kernel", []


def _busy_ns(kernels):
    busy, end = 0, None
    for k in sorted(kernels, key=lambda k: k.start_ns()):
        if end is None or k.start_ns() >= end:
            busy += k.end_ns() - k.start_ns()
            end = k.end_ns()
        elif k.end_ns() > end:
            busy += k.end_ns() - end
            end = k.end_ns()
    return busy


def _op_row(step, op, shapes, dtypes, cpu, kernels, **extra):
    return {
        "step": step,
        "op": op,
        **extra,
        "shapes": shapes,
        "dtypes": dtypes,
        "cpu_start_ns": cpu.start_ns(),
        "cpu_end_ns": cpu.end_ns(),
        "gpu_start_ns": min(k.start_ns() for k in kernels),
        "gpu_end_ns": max(k.end_ns() for k in kernels),
        "gpu_kernel_ns": sum(k.duration_ns() for k in kernels),
        "gpu_busy_ns": _busy_ns(kernels),
        "kernels": [
            {
                "name": k.name(),
                "start_ns": k.start_ns(),
                "end_ns": k.end_ns(),
                "stream": k.device_resource_id(),
            }
            for k in kernels
        ],
    }


def _graph_rows(step, launch, kernels, graph, kernel_ops):
    """Split one graph launch's replayed kernels, in GPU start order, into
    rows of consecutive kernels mapped to the same op."""
    rows, run, key = [], [], None
    for k in sorted(kernels, key=lambda k: k.start_ns()):
        op, op_from, alternatives = kernel_op(kernel_ops, k.name())
        if run and (op, op_from) != key[:2]:
            rows.append(_named_graph_row(step, launch, run, graph, *key))
            run = []
        key = (op, op_from, alternatives)
        run.append(k)
    if run:
        rows.append(_named_graph_row(step, launch, run, graph, *key))
    return rows


def _named_graph_row(step, launch, kernels, graph, op, op_from, alternatives):
    extra = {"graph_replay": True, "graph": graph, "op_from": op_from}
    if alternatives:
        extra["op_alternatives"] = alternatives
    return _op_row(step, op, [], [], launch, kernels, **extra)


def window_op_rows(events, kernel_ops=None):
    """Group a profiler window's kernels by the op that launched them.

    The launch's CPU time places a kernel inside a ``compass_step:<n>``
    annotation, which spans the whole forward (input preparation and sampling
    included); kernels outside any step are dropped. Eager kernels with an op
    are added to ``kernel_ops`` (updated in place) before graph replay kernels
    are named from it.
    """
    kernel_ops = {} if kernel_ops is None else kernel_ops
    eager, graphs = {}, {}
    for k, launch, op, marks in _attributed_kernels(
        events, (_STEP_PREFIX, _GRAPH_PREFIX)
    ):
        step = marks[_STEP_PREFIX]
        if step is None:
            continue
        step = int(step)
        if op is not None:
            eager.setdefault((step, op.correlation_id()), (op, []))[1].append(k)
        elif "Graph" in launch.name():
            graph = marks[_GRAPH_PREFIX]
            graphs.setdefault((step, k.correlation_id()), (launch, graph, []))[
                2
            ].append(k)
        else:
            eager.setdefault((step, ("launch", k.correlation_id())), (launch, []))[
                1
            ].append(k)
    rows = []
    for (step, key), (src, ks) in eager.items():
        has_op = not isinstance(key, tuple)
        if has_op:
            for k in ks:
                learn_kernel_op(kernel_ops, k.name(), src.name())
        rows.append(
            _op_row(
                step,
                src.name(),
                src.shapes() if has_op else [],
                src.dtypes() if has_op else [],
                src,
                ks,
                graph_replay=False,
            )
        )
    for (step, _), (launch, graph, ks) in graphs.items():
        rows.extend(_graph_rows(step, launch, ks, graph, kernel_ops))
    return sorted(rows, key=lambda r: (r["step"], r["gpu_start_ns"]))


def record_compass(func):
    @wraps(func)
    def wrapped(self, input_ids, batch=None):
        recorder = getattr(self, "compass_recorder", None)
        if recorder is None:
            return func(self, input_ids, batch)
        with recorder.measure(batch):
            return func(self, input_ids, batch)

    return wrapped


def record_compass_forward(func):
    @wraps(func)
    def wrapped(self, batch):
        recorder = getattr(self, "compass_recorder", None)
        if recorder is None:
            return func(self, batch)
        with recorder.measure_forward(batch):
            return func(self, batch)

    return wrapped
