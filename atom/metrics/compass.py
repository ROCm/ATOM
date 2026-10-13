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
  kernels overlapping on several streams count once. Kernels replayed from a
  CUDA graph are attributed through that graph's template (below): matched
  ones get the op, shapes and dtypes of the template kernel at the same place
  (``shapes_from: "graph_template"``; their CPU times are the graph launch's;
  ``kernels_renamed`` counts those paired with a template kernel of another
  name). Unmatched replay kernels stay grouped under the graph launch with no
  shapes.
- ``windows.<pid>.jsonl``: one row per op window -- why it opened, the batch
  bucket, the steps it holds ops for, and how many replayed graph kernels the
  templates matched.
- ``graphs.<pid>.jsonl``: one row per captured CUDA graph template: the
  graph's ``(bs, max_q_len)`` name and its kernel and op counts; and, the
  first time an op window replays a graph, a row with ``sample_step`` holding
  the template's and that replay's kernel names and streams.
- ``meta.<pid>.json``: rank identity and collection settings.

Op windows are scheduled per batch bucket (prefill/decode, power-of-two batch
size, prefill tokens and max context length): a bucket gets a window once it
has been seen more than ``ATOM_COMPASS_OP_MIN_SEEN`` times (so first-call JIT
and warmup stay out), until it has ``ATOM_COMPASS_OP_PER_BUCKET`` windows;
every ``ATOM_COMPASS_OP_EVERY``-th step also opens one. A window profiles one
step more than ``ATOM_COMPASS_OP_STEPS`` and drops that tail step's ops: the
profiler loses most of the last step's kernels when it stops.

A graph template is the kernel sequence of the eager warmup forward that runs
right before each graph capture, with the op that launched each kernel. Replay
kernels are aligned to it in GPU start order by kernel name.

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

SCHEMA_VERSION = 2

_STEP_PREFIX = "compass_step:"
_GRAPH_PREFIX = "compass_graph:"
_TEMPLATE_MARK = "compass_graph_template"
_RUNTIME_PREFIXES = ("hip", "cuda", "rocprof")
_MAX_PENDING = 1024
_FLUSH_INTERVAL_S = 1.0
# An op window still open after this long without a forward is closed by the
# engine's idle poll, so its rows land without waiting for more traffic or a
# graceful shutdown.
_WINDOW_IDLE_CLOSE_S = 2.0
# The profiler drops the trailing GPU activity it has not received when it
# stops; a template's capture idles this long after its last kernel first.
_TEMPLATE_DRAIN_S = 0.05
# Replay-to-template kernel alignment: scores, band half-width, and how many
# distinct replay sequences per graph keep their alignment cached.
_ALIGN_MATCH, _ALIGN_RENAME, _ALIGN_GAP = 2, -1, -1
_ALIGN_BAND = 32
_ALIGN_CACHE = 8
_DIAG, _UP, _LEFT = 0, 1, 2


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
        self.graph_templates = {}  # graph name -> kernel/op template
        self.graph_samples = set()  # graphs whose names went to graphs.jsonl
        logger.info(
            "Compass recording to %s (op window %d steps every %d, "
            "%d per bucket after %d sightings, max %s)",
            rank_dir,
            self.op_steps,
            self.op_every,
            self.op_per_bucket,
            self.op_min_seen,
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
                window.profiler.kineto_results.events(), self.graph_templates
            )
            if r["step"] != tail
        ]
        for row in rows:
            self.ops_file.write(json.dumps({**self.rank, **row}) + "\n")
        graph_rows = [r for r in rows if r["graph_replay"]]
        self._write_graph_samples(graph_rows)
        summary = {
            **self.rank,
            "window": self.windows_done,
            "reason": info["reason"],
            "bucket": info["bucket"],
            "steps": [s for s in info["steps"] if s != tail],
            "tail_step": tail,
            "op_rows": len(rows),
            "graph_kernels": sum(len(r["kernels"]) for r in graph_rows),
            "graph_kernels_matched": sum(
                len(r["kernels"])
                for r in graph_rows
                if r.get("shapes_from") == "graph_template"
            ),
            "graph_kernels_renamed": sum(r.get("kernels_renamed", 0) for r in rows),
            "close_s": time.perf_counter() - began,
        }
        self.windows_file.write(json.dumps(summary) + "\n")
        self.windows_done += 1
        self.flush()
        logger.info(
            "Compass op window %d (%s %s, steps %s): %d op rows, "
            "%d/%d graph kernels matched, in %.2fs",
            summary["window"],
            summary["reason"],
            summary["bucket"],
            summary["steps"][:1],
            len(rows),
            summary["graph_kernels_matched"],
            summary["graph_kernels"],
            summary["close_s"],
        )

    def _write_graph_samples(self, graph_rows):
        """Once per graph, its template and one replay's kernel names, so the
        alignment can be checked offline."""
        replays = {}
        for r in graph_rows:
            if r["graph"] in self.graph_templates:
                replays.setdefault((r["graph"], r["step"]), []).extend(r["kernels"])
        for (graph, step), kernels in sorted(replays.items()):
            if graph in self.graph_samples:
                continue
            self.graph_samples.add(graph)
            kernels.sort(key=lambda k: k["start_ns"])
            sample = {
                **self.rank,
                "graph": graph,
                "sample_step": step,
                "template_kernels": self.graph_templates[graph]["kernels"],
                "template_streams": self.graph_templates[graph]["streams"],
                "replay_kernels": [k["name"] for k in kernels],
                "replay_streams": [k["stream"] for k in kernels],
            }
            self.graphs_file.write(json.dumps(sample) + "\n")

    # -- CUDA graph templates -------------------------------------------------

    @contextmanager
    def graph_template(self, key):
        """Profile the eager warmup forward before ``key``'s graph capture."""
        if (
            not self.graph_ops
            or self.op_steps <= 0
            or self.window is not None
            or self.is_profiler_busy()
            or torch.cuda.is_current_stream_capturing()
        ):
            yield
            return
        name = graph_name(key)
        torch.cuda.synchronize()
        prof = profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=True,
        )
        prof.__enter__()
        try:
            with record_function(_TEMPLATE_MARK):
                yield
            torch.cuda.synchronize()
            time.sleep(_TEMPLATE_DRAIN_S)
        finally:
            prof.__exit__(None, None, None)
        template = graph_template_entries(prof.profiler.kineto_results.events())
        self.graph_templates[name] = template
        self.graphs_file.write(
            json.dumps(
                {
                    **self.rank,
                    "graph": name,
                    "kernels": len(template["kernels"]),
                    "ops": len(template["ops"]),
                    "kernels_without_op": template["op_index"].count(None),
                }
            )
            + "\n"
        )
        self.graphs_file.flush()

    def graph_replay(self, key):
        """Mark a graph replay inside an op window so its kernels can be
        matched to the graph's template."""
        if self.window is None or graph_name(key) not in self.graph_templates:
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


def graph_template_entries(events):
    """The kernel sequence of a graph's eager warmup forward, in GPU start
    order, each with the index of its launching op (None if no op)."""
    kernels, streams, op_index, ops, index_of = [], [], [], [], {}
    for k, _launch, op, marks in _attributed_kernels(events, (_TEMPLATE_MARK,)):
        if marks[_TEMPLATE_MARK] is None:
            continue
        kernels.append(k.name())
        streams.append(k.device_resource_id())
        if op is None:
            op_index.append(None)
            continue
        i = index_of.get(op.correlation_id())
        if i is None:
            i = index_of[op.correlation_id()] = len(ops)
            ops.append({"op": op.name(), "shapes": op.shapes(), "dtypes": op.dtypes()})
        op_index.append(i)
    return {"kernels": kernels, "streams": streams, "op_index": op_index, "ops": ops}


def _align(names, template_names, band=_ALIGN_BAND):
    """For each replay kernel, the index of its template kernel or None.

    A global alignment restricted to a band around the diagonal: layers repeat
    the same kernel names, so an unbanded longest-match pairs layer i of the
    replay with layer j of the template and loses everything after. Pairing
    two different names beats leaving both unmatched: a graph replays some ops
    with other kernels than their eager run (custom all-reduce).
    """
    n, m = len(names), len(template_names)
    if names == template_names:
        return list(range(n))
    if not n or not m:
        return [None] * n
    ids = {}
    a = [ids.setdefault(x, len(ids)) for x in names]
    b = [ids.setdefault(x, len(ids)) for x in template_names]
    neg = float("-inf")
    band = max(band, m // n + 2)

    def span(i):
        c = i * m // n
        return max(0, c - band), min(m, c + band)

    lo, hi = span(0)
    prev = [neg] * (m + 1)
    for j in range(hi + 1):
        prev[j] = j * _ALIGN_GAP
    trace = [(0, bytearray([_LEFT] * (hi + 1)))]
    for i in range(1, n + 1):
        lo, hi = span(i)
        cur = [neg] * (m + 1)
        moves = bytearray(hi - lo + 1)
        ai = a[i - 1]
        for j in range(lo, hi + 1):
            best, move = prev[j] + _ALIGN_GAP, _UP
            if j:
                diag = prev[j - 1] + (_ALIGN_MATCH if ai == b[j - 1] else _ALIGN_RENAME)
                if diag > best:
                    best, move = diag, _DIAG
                left = cur[j - 1] + _ALIGN_GAP
                if left > best:
                    best, move = left, _LEFT
            cur[j] = best
            moves[j - lo] = move
        trace.append((lo, moves))
        prev = cur
    match = [None] * n
    i, j = n, m
    while i or j:
        lo, moves = trace[i]
        move = moves[j - lo]
        if move == _DIAG:
            match[i - 1] = j - 1
            i, j = i - 1, j - 1
        elif move == _UP:
            i -= 1
        else:
            j -= 1
    return match


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


def _graph_rows(step, launch, kernels, graph, template):
    """Split one graph launch's replayed kernels into rows per template op."""
    extra = {"graph_replay": True, "graph": graph}
    if template is None:
        return [_op_row(step, launch.name(), [], [], launch, kernels, **extra)]
    names = tuple(k.name() for k in kernels)
    # Every replay of a graph launches the same kernels: align once.
    cache = template.setdefault("aligned", {})
    match = cache.get(names)
    if match is None:
        match = _align(list(names), template["kernels"])
        if len(cache) < _ALIGN_CACHE:
            cache[names] = match
    by_op, renamed, rest = {}, {}, []
    for k, m in zip(kernels, match):
        i = None if m is None else template["op_index"][m]
        if i is None:
            rest.append(k)
            continue
        by_op.setdefault(i, []).append(k)
        if k.name() != template["kernels"][m]:
            renamed[i] = renamed.get(i, 0) + 1
    rows = []
    for i, ks in by_op.items():
        op = template["ops"][i]
        rows.append(
            _op_row(
                step,
                op["op"],
                op["shapes"],
                op["dtypes"],
                launch,
                ks,
                **extra,
                shapes_from="graph_template",
                kernels_renamed=renamed.get(i, 0),
            )
        )
    if rest:
        rows.append(_op_row(step, launch.name(), [], [], launch, rest, **extra))
    return rows


def window_op_rows(events, graph_templates=None):
    """Group a profiler window's kernels by the op that launched them.

    The launch's CPU time places a kernel inside a ``compass_step:<n>``
    annotation, which spans the whole forward (input preparation and sampling
    included); kernels outside any step are dropped. A graph launch inside a
    ``compass_graph:<name>`` annotation has its kernels matched to that
    graph's template.
    """
    graph_templates = graph_templates or {}
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
        rows.extend(_graph_rows(step, launch, ks, graph, graph_templates.get(graph)))
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
