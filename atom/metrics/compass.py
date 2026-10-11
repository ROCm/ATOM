"""Per-rank forward and per-op GPU timing records for ATOM Compass.

Enabled by ``ATOM_COMPASS_DIR``. Each model-runner rank writes, under
``$ATOM_COMPASS_DIR/dp{d}_pp{p}_tp{t}/``:

- ``forward.<pid>.jsonl``: one row per ``run_model`` call (``phase`` target)
  and per drafter pass (``draft_kv``, ``draft``) -- device-event
  forward duration, absolute GPU start/end (wall-clock ns, via an anchor
  event), and the batch shape (graph padding, prefill/decode split, context
  length sum/max; per-sequence query/context lengths on tp_rank 0 only).
- ``ops.<pid>.jsonl``: during sampled profiler windows, one row per op that
  launched GPU work inside a forward: op name, input shapes/dtypes, CPU and
  GPU start/end (wall-clock ns) and its kernels. Kernels replayed from a CUDA
  graph carry no op shapes; they are grouped under their graph launch and the
  step's batch shape is in the forward row with the same ``step``.
- ``meta.<pid>.json``: rank identity and collection settings.

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

_STEP_PREFIX = "compass_step:"
_RUNTIME_PREFIXES = ("hip", "cuda", "rocprof")
_MAX_PENDING = 1024
_FLUSH_INTERVAL_S = 1.0
# An op window still open after this long without a forward is closed by the
# engine's idle poll, so its rows land without waiting for more traffic or a
# graceful shutdown.
_WINDOW_IDLE_CLOSE_S = 2.0


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
        # One newline-terminated line, so /compass/records serves it whole.
        with open(os.path.join(rank_dir, f"meta.{pid}.json"), "w") as f:
            meta = {
                **self.rank,
                "host": socket.gethostname(),
                "pid": pid,
                "device": torch.cuda.get_device_name(),
                "torch": torch.__version__,
                "op_steps": self.op_steps,
                "op_every": self.op_every,
                "op_max_windows": self.op_max_windows,
                "anchor_interval_s": self.anchor_interval_s,
            }
            f.write(json.dumps(meta) + "\n")
        self.last_flush = time.monotonic()

        self.step = 0
        self.pending = deque()
        self.free = []
        self.anchors = deque()  # (event, wall ns), oldest first
        self._reanchor()

        # TP peers run the same batch; only tp0 of each group lists every
        # sequence, the rest keep the scalar summary.
        self.seq_lists = tp_rank == 0
        self.last_measure = time.monotonic()
        self.window = None  # active torch profiler
        self.window_first_step = 0
        self.windows_done = 0
        logger.info(
            "Compass recording to %s (op window %d steps every %d, max %s)",
            rank_dir,
            self.op_steps,
            self.op_every,
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

    def poll(self):
        while self.pending:
            row, anchor, start, end = self.pending[0]
            if not end.query():
                break
            self.pending.popleft()
            row["gpu_start_ns"] = self._wall_ns(anchor, start)
            row["gpu_end_ns"] = self._wall_ns(anchor, end)
            row["gpu_ms"] = start.elapsed_time(end)
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
        self.last_flush = time.monotonic()

    @contextmanager
    def measure(self, batch, phase="target"):
        """Time one GPU phase of a step: the target ``run_model`` forward, or a
        drafter's ``draft_kv`` / ``draft`` pass (timed at its runner call site)."""
        if torch.cuda.is_current_stream_capturing():
            # Events recorded into a graph would time replays, not this call.
            yield
            return
        self.poll()
        if time.monotonic() - self.anchor_at > self.anchor_interval_s:
            self._reanchor()
        self.last_measure = time.monotonic()
        step = self.step
        self.step += 1
        self._maybe_open_window(step)

        context = get_forward_context().context
        row = {
            "step": step,
            "phase": phase,
            **self.rank,
            **_batch_fields(batch, context, self.seq_lists),
        }
        if len(self.pending) >= _MAX_PENDING:
            # Never block the forward on the recorder; drop the row instead.
            logger.warning(
                "Compass: %d pending forward rows, dropping step %d",
                len(self.pending),
                step,
            )
            yield
            self._maybe_close_window(step)
            return
        start, end = (
            self.free.pop()
            if self.free
            else (
                torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True),
            )
        )
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
            raise
        end.record()
        row["cpu_end_ns"] = time.time_ns()
        row["op_window"] = self.window is not None
        self.pending.append((row, self.anchors[-1], start, end))
        self._maybe_close_window(step)

    # -- op windows -------------------------------------------------------

    def _maybe_open_window(self, step):
        if (
            self.window is not None
            or self.op_steps <= 0
            or step % self.op_every != 0
            or (self.op_max_windows and self.windows_done >= self.op_max_windows)
            or self.is_profiler_busy()
        ):
            return
        self.window = profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=True,
        )
        self.window.__enter__()
        self.window_first_step = step

    def _maybe_close_window(self, step):
        if (
            self.window is not None
            and step - self.window_first_step + 1 >= self.op_steps
        ):
            self._close_window()

    def _close_window(self):
        began = time.perf_counter()
        torch.cuda.synchronize()  # kernels still in flight would be lost
        window, self.window = self.window, None
        window.__exit__(None, None, None)
        rows = window_op_rows(window.profiler.kineto_results.events())
        for row in rows:
            self.ops_file.write(json.dumps({**self.rank, **row}) + "\n")
        self.windows_done += 1
        self.flush()
        logger.info(
            "Compass op window %d (steps %d..): %d op rows in %.2fs",
            self.windows_done,
            self.window_first_step,
            len(rows),
            time.perf_counter() - began,
        )

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


def window_op_rows(events):
    """Group a profiler window's kernels by the op that launched them.

    A kernel shares its correlation id with the runtime launch call and links
    (``linked_correlation_id``) to the innermost CPU op enclosing that launch,
    which carries the input shapes. The launch's CPU time places it inside a
    ``compass_step:<n>`` annotation. Kernels outside any step (input
    preparation, sampling) are dropped.
    """
    runtime, ops, steps, kernels = {}, {}, [], []
    for e in events:
        if e.device_type() != torch.autograd.DeviceType.CPU:
            if not e.is_user_annotation():
                kernels.append(e)
            continue
        name = e.name()
        if name.startswith(_STEP_PREFIX):
            steps.append((e.start_ns(), e.end_ns(), int(name[len(_STEP_PREFIX) :])))
        elif name.startswith(_RUNTIME_PREFIXES):
            runtime[e.correlation_id()] = e
        elif not e.is_user_annotation():
            ops[e.correlation_id()] = e
    steps.sort()
    step_starts = [s[0] for s in steps]

    grouped = {}
    for k in kernels:
        launch = runtime.get(k.correlation_id())
        if launch is None:
            continue
        i = bisect.bisect_right(step_starts, launch.start_ns()) - 1
        if i < 0 or launch.start_ns() > steps[i][1]:
            continue
        op = ops.get(k.linked_correlation_id()) if k.linked_correlation_id() else None
        key = (
            ("op", op.correlation_id())
            if op is not None
            else ("launch", k.correlation_id())
        )
        row = grouped.get(key)
        if row is None:
            src = op if op is not None else launch
            row = grouped[key] = {
                "step": steps[i][2],
                "op": src.name(),
                "graph_replay": op is None and "Graph" in launch.name(),
                "shapes": op.shapes() if op is not None else [],
                "dtypes": op.dtypes() if op is not None else [],
                "cpu_start_ns": src.start_ns(),
                "cpu_end_ns": src.end_ns(),
                "gpu_start_ns": k.start_ns(),
                "gpu_end_ns": k.end_ns(),
                "gpu_kernel_ns": 0,
                "kernels": [],
            }
        row["gpu_start_ns"] = min(row["gpu_start_ns"], k.start_ns())
        row["gpu_end_ns"] = max(row["gpu_end_ns"], k.end_ns())
        row["gpu_kernel_ns"] += k.duration_ns()
        row["kernels"].append(
            {
                "name": k.name(),
                "start_ns": k.start_ns(),
                "end_ns": k.end_ns(),
                "stream": k.device_resource_id(),
            }
        )
    return sorted(grouped.values(), key=lambda r: (r["step"], r["gpu_start_ns"]))


def record_compass(func):
    @wraps(func)
    def wrapped(self, input_ids, batch=None):
        recorder = getattr(self, "compass_recorder", None)
        if recorder is None:
            return func(self, input_ids, batch)
        with recorder.measure(batch):
            return func(self, input_ids, batch)

    return wrapped
