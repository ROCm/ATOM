# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Record which aiter ops a run makes, with what arguments and how often,
graph replays included (`ATOM_OP_RECORD_DIR`).

An aiter op call is seen through aiter's call hook
(`aiter.jit.utils.call_hook`), which fires wherever the op's Python runs --
on an eager call, or once while a CUDA graph records it. A replay runs no
Python, so the count of an op recorded into a graph is the graph's replays
times its calls in the recording. `torch.cuda.CUDAGraph` is patched to tell
the two apart: a call made while a graph captures is charged to that graph,
a replay charges the graph once to the running step, and any other call --
prefill, an eager region between graph segments, sampling -- is charged to
the running step directly. Which ATOM graph type recorded the graph does not
matter, and an eager region is never counted twice.

One directory per rank holds:

- ``meta.json``: the run's configuration and the code it ran.
- ``sigs.jsonl``: one line per distinct call signature -- op name, nesting
  depth (0 is a call from outside aiter), the ops it is nested in
  (``parents``, outermost first; an ATOM op registered through aiter's
  `torch_compile_guard` counts as one), and each argument's shape, dtype and
  stride, or its value for a scalar.
- ``graphs.jsonl``: one line per graph recording, the signatures it captured
  and how many times each.
- ``steps.jsonl``: one line per forward, with its wall-clock window, its
  batch (per-request query and context lengths) and the signatures and graph
  replays it ran; a line of ``"kind": "outside"`` collects calls made between
  forwards (startup warmup, graph recording passes).

`tools/op_record/export.py` turns these into per-op counts for a time window.
"""

import atexit
import enum
import functools
import json
import os
import subprocess
import sys
import time
from collections import Counter
from typing import Any

import torch

_FLUSH_SECONDS = 1.0


def _arg_signature(x: Any) -> Any:
    """A hashable stand-in for one argument: shape, dtype and stride for a
    tensor, the value for a scalar, the type name for anything else."""
    if isinstance(x, torch.Tensor):
        try:
            stride = x.stride()
        except RuntimeError:  # sparse and other strideless layouts
            stride = ()
        return ("T", tuple(x.shape), str(x.dtype).removeprefix("torch."), stride)
    if x is None or isinstance(x, (bool, int, float, str)):
        return x
    if isinstance(x, (list, tuple)):
        return tuple(_arg_signature(v) for v in x)
    if isinstance(x, enum.Enum):
        return f"{type(x).__name__}.{x.name}"
    if isinstance(x, torch.dtype):
        return str(x).removeprefix("torch.")
    return f"<{type(x).__name__}>"


def _arg_json(sig: Any) -> Any:
    if isinstance(sig, tuple) and len(sig) == 4 and sig[0] == "T":
        return {"shape": list(sig[1]), "dtype": sig[2], "stride": list(sig[3])}
    if isinstance(sig, tuple):
        return [_arg_json(v) for v in sig]
    return sig


def _git_head(path: str) -> str:
    """The commit of the checkout holding ``path``, or why it is unknown: an
    empty answer would read as "no commit" instead of "could not ask"."""
    try:
        out = subprocess.run(
            ["git", "-C", path, "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as e:
        return f"unknown: {e}"
    if out.returncode != 0 or not out.stdout.strip():
        return f"unknown: git rc={out.returncode} {out.stderr.strip()[:200]}"
    return out.stdout.strip()


class OpRecorder:
    def __init__(self, out_dir: str, meta: dict):
        os.makedirs(out_dir, exist_ok=True)
        self.out_dir = out_dir
        with open(os.path.join(out_dir, "meta.json"), "w") as f:
            json.dump(meta, f, indent=1, default=str)
        self._sigs_file = self._open(out_dir, "sigs.jsonl")
        self._graphs_file = self._open(out_dir, "graphs.jsonl")
        self._steps_file = self._open(out_dir, "steps.jsonl")
        self._last_flush = time.monotonic()
        # (op, parents, args, kwargs) signature -> id.
        self._sigs: dict[tuple, int] = {}
        # The ops the current call is nested in, outermost first: a call at
        # depth d ends every call at depth d or deeper.
        self._stack: list[str] = []
        # id(torch.cuda.CUDAGraph) -> graph id; a new recording into a reused
        # object takes a new id.
        self._graph_ids: dict[int, int] = {}
        self._next_graph = 0
        # (graph id, its calls) for each recording in progress, innermost last.
        self._capturing: list[tuple[int, Counter]] = []
        # Since the last step line: signature -> calls, graph id -> replays.
        self._calls: Counter = Counter()
        self._replays: Counter = Counter()
        self._outside_since = time.time()
        self._in_step = False

    @staticmethod
    def _open(out_dir: str, name: str):
        # Open for the whole run; `close` (registered with atexit) ends it.
        return open(os.path.join(out_dir, name), "a")

    # aiter call hook
    def on_call(self, name: str, depth: int, args: tuple, kwargs: dict) -> None:
        del self._stack[depth:]
        parents = tuple(self._stack)
        self._stack.append(name)
        key = (
            name,
            parents,
            tuple(_arg_signature(a) for a in args),
            tuple((k, _arg_signature(v)) for k, v in kwargs.items()),
        )
        sid = self._sigs.get(key)
        if sid is None:
            sid = self._sigs[key] = len(self._sigs)
            line = {
                "id": sid,
                "op": name,
                "depth": depth,
                "parents": list(parents),
                "args": [_arg_json(a) for a in key[2]],
                "kwargs": {k: _arg_json(v) for k, v in key[3]},
            }
            self._sigs_file.write(json.dumps(line, default=str) + "\n")
        if self._capturing:
            self._capturing[-1][1][sid] += 1
        else:
            self._calls[sid] += 1

    # torch.cuda.CUDAGraph
    def on_capture_begin(self, graph: torch.cuda.CUDAGraph) -> None:
        gid = self._next_graph
        self._next_graph += 1
        self._graph_ids[id(graph)] = gid
        self._capturing.append((gid, Counter()))

    def on_capture_end(self, graph: torch.cuda.CUDAGraph) -> None:
        gid = self._graph_ids.get(id(graph))
        for i in range(len(self._capturing) - 1, -1, -1):
            if self._capturing[i][0] == gid:
                _, calls = self._capturing.pop(i)
                break
        else:
            return
        line = {"gid": gid, "t": time.time(), "calls": dict(calls)}
        self._graphs_file.write(json.dumps(line) + "\n")

    def on_replay(self, graph: torch.cuda.CUDAGraph) -> None:
        # -1: a graph recorded before the recorder was installed.
        self._replays[self._graph_ids.get(id(graph), -1)] += 1

    # ModelRunner.forward
    def step(self, forward, batch, *args, **kwargs):
        if self._in_step:
            return forward(batch, *args, **kwargs)
        t0 = time.time()
        if self._calls or self._replays:
            self._write_step({"kind": "outside", "t0": self._outside_since, "t1": t0})
        self._in_step = True
        try:
            out = forward(batch, *args, **kwargs)
        finally:
            self._in_step = False
            t1 = time.time()
            self._write_step(self._batch_fields(batch) | {"t0": t0, "t1": t1})
            self._outside_since = t1
            if time.monotonic() - self._last_flush > _FLUSH_SECONDS:
                self.flush()
        return out

    @staticmethod
    def _batch_fields(batch) -> dict:
        def ints(name):
            value = getattr(batch, name, None)
            return None if value is None else [int(v) for v in value]

        return {
            "kind": "step",
            "dummy": bool(getattr(batch, "is_dummy_run", False)),
            "tokens": int(getattr(batch, "total_tokens_num", 0) or 0),
            "prefill_seqs": int(getattr(batch, "total_seqs_num_prefill", 0) or 0),
            "decode_seqs": int(getattr(batch, "total_seqs_num_decode", 0) or 0),
            "spec": int(getattr(batch, "num_spec_step", 0) or 0),
            "q": ints("num_scheduled_tokens"),
            "ctx": ints("context_lens"),
        }

    def _write_step(self, fields: dict) -> None:
        fields["calls"] = dict(self._calls)
        fields["replays"] = dict(self._replays)
        self._calls.clear()
        self._replays.clear()
        self._steps_file.write(json.dumps(fields) + "\n")

    def flush(self) -> None:
        for f in (self._sigs_file, self._graphs_file, self._steps_file):
            f.flush()
        self._last_flush = time.monotonic()

    def close(self) -> None:
        if self._calls or self._replays:
            self._write_step(
                {"kind": "outside", "t0": self._outside_since, "t1": time.time()}
            )
        for f in (self._sigs_file, self._graphs_file, self._steps_file):
            if not f.closed:
                f.close()


def _patch_cuda_graph(recorder: OpRecorder) -> None:
    graph_cls = torch.cuda.CUDAGraph
    begin, end, replay = (
        graph_cls.capture_begin,
        graph_cls.capture_end,
        graph_cls.replay,
    )

    @functools.wraps(begin)
    def capture_begin(self, *args, **kwargs):
        out = begin(self, *args, **kwargs)
        recorder.on_capture_begin(self)
        return out

    @functools.wraps(end)
    def capture_end(self, *args, **kwargs):
        try:
            return end(self, *args, **kwargs)
        finally:
            recorder.on_capture_end(self)

    @functools.wraps(replay)
    def replay_(self, *args, **kwargs):
        recorder.on_replay(self)
        return replay(self, *args, **kwargs)

    graph_cls.capture_begin = capture_begin
    graph_cls.capture_end = capture_end
    graph_cls.replay = replay_


def _meta(runner, out_dir: str) -> dict:
    config = runner.config
    parallel = getattr(config, "parallel_config", None)
    fields = (
        "model",
        "tensor_parallel_size",
        "kv_cache_dtype",
        "enforce_eager",
        "max_num_seqs",
        "max_num_batched_tokens",
        "max_model_len",
        "torch_dtype",
    )
    meta = {
        "rank": runner.rank,
        "pid": os.getpid(),
        "started": time.time(),
        "out_dir": out_dir,
        "argv": sys.argv,
        "config": {f: getattr(config, f, None) for f in fields},
        "data_parallel_size": getattr(parallel, "data_parallel_size", None),
        "compilation_config": repr(getattr(config, "compilation_config", None)),
    }
    import atom

    meta["atom_path"] = os.path.dirname(atom.__file__)
    meta["atom_commit"] = _git_head(meta["atom_path"])
    try:
        import aiter

        meta["aiter_path"] = os.path.dirname(aiter.__file__)
        meta["aiter_commit"] = _git_head(meta["aiter_path"])
    except ImportError as e:
        meta["aiter_path"] = f"unknown: {e}"
    return meta


def install(runner, out_dir: str) -> OpRecorder:
    """Record ``runner``'s aiter calls and graph replays into ``out_dir``:
    one line per ``runner.forward`` call."""
    try:
        from aiter.jit.utils.call_hook import register_call_hook
    except ImportError as e:
        raise RuntimeError(
            "ATOM_OP_RECORD_DIR needs an aiter with aiter.jit.utils.call_hook"
        ) from e
    recorder = OpRecorder(out_dir, _meta(runner, out_dir))
    register_call_hook(recorder.on_call)
    _patch_cuda_graph(recorder)
    forward = runner.forward

    @functools.wraps(forward)
    def recorded_forward(batch, *args, **kwargs):
        return recorder.step(forward, batch, *args, **kwargs)

    runner.forward = recorded_forward
    atexit.register(recorder.close)
    return recorder
