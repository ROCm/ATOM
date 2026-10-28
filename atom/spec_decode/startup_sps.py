# SPDX-License-Identifier: MIT
"""Measure DSpark's verification cost curve at startup, as DeepInfer does.

Right after the decode token graphs are captured -- before the server accepts a
request -- every (batch, q) graph the step can run is replayed and timed. The
synthetic metadata is the one capture itself used, so the replays touch only
what capture already wrote. The per-shape medians are reduced by MAX over TP
and DP (the slowest rank sets the step), interpolated into the dense
`sps_table` the scheduler indexes by total query rows, and cached on disk keyed
by the deployment identity, so a restart of the same deployment skips the
sweep. Like the offline profile it replaces, this covers the target forward at
a short context only; `online_sps` adds what serving adds on top.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import logging
import os
import statistics
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

logger = logging.getLogger("atom")

# Bump when what a cached point means changes.
_FORMAT = 1
_WARMUP = 3
_ITERS = 15


def sps_table_from_points(points, max_rows):
    """Dense steps/s per total query rows from measured (rows, ms) points.

    Several shapes can share a row count: keep the slower median, then a
    monotone latency envelope before interpolation. Past the largest measured
    row count -- graphs a DP cap never records -- latency continues along the
    last measured slope.
    """
    grouped = {}
    for rows, ms in points:
        rows = int(rows)
        if rows < 1 or not np.isfinite(ms) or ms <= 0:
            raise ValueError("Invalid measured SPS point")
        grouped.setdefault(rows, []).append(float(ms))
    if not grouped:
        raise ValueError("No measured SPS points")
    xs = sorted(grouped)
    ys = np.maximum.accumulate([max(grouped[x]) for x in xs])
    grid = np.arange(max_rows + 1)
    latency = np.interp(grid, xs, ys)
    if xs[-1] < max_rows:
        slope = (ys[-1] - ys[-2]) / (xs[-1] - xs[-2]) if len(xs) > 1 else 0.0
        tail = grid > xs[-1]
        latency[tail] = ys[-1] + max(slope, 0.0) * (grid[tail] - xs[-1])
    return (1000.0 / latency).tolist()


def _groups():
    from aiter.dist.parallel_state import get_dp_group, get_tp_group

    return get_tp_group().cpu_group, get_dp_group().cpu_group


def _all_max(value: float) -> float:
    t = torch.tensor([value], dtype=torch.float64)
    for group in _groups():
        dist.all_reduce(t, op=dist.ReduceOp.MAX, group=group)
    return float(t.item())


def _all_max_list(values) -> list[float]:
    t = torch.tensor(values, dtype=torch.float64)
    for group in _groups():
        dist.all_reduce(t, op=dist.ReduceOp.MAX, group=group)
    return t.tolist()


def _all_true(flag: bool) -> bool:
    t = torch.tensor([int(flag)], dtype=torch.int64)
    for group in _groups():
        dist.all_reduce(t, op=dist.ReduceOp.MIN, group=group)
    return bool(t.item())


def _dynamic_q(runner) -> bool:
    """Whether the backend's capture builder takes ``max_q_len``. One that does
    not (Kimi-K3's) captured, and so can rebuild, the full width only."""
    builder = runner.attn_metadata_builder.build_for_cudagraph_capture
    return "max_q_len" in inspect.signature(builder).parameters


def _shapes(runner, qs):
    from atom.utils.region_graph import TokenGraph

    if not _dynamic_q(runner):
        qs = [max(qs)]
    captured = set(TokenGraph.captured_sizes("decode"))
    return [
        (bs, q) for bs in sorted(runner.capture_sizes) for q in qs if bs * q in captured
    ]


def _cache_path(identity, shapes, qs):
    from atom.utils import envs

    root = envs.ATOM_DSPARK_SPS_CACHE_DIR
    if root is None:
        root = str(Path.home() / ".cache" / "atom" / "dspark_sps")
    if not root:
        return None
    key = json.dumps(
        {"format": _FORMAT, "identity": identity, "shapes": shapes, "q": qs},
        sort_keys=True,
    )
    return Path(root) / f"{hashlib.sha256(key.encode()).hexdigest()}.json"


def _read(path, identity, shapes):
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if data.get("identity") != identity or [
        [s["batch"], s["q"]] for s in data.get("shapes", [])
    ] != [list(s) for s in shapes]:
        return None
    return data


@torch.inference_mode()
def _measure(runner, shapes):
    from atom.utils.forward_context import (
        get_forward_context,
        reset_forward_context,
        set_forward_context,
    )

    builder = runner.attn_metadata_builder.build_for_cudagraph_capture
    dynamic = _dynamic_q(runner)
    result = []
    for bs, q in shapes:
        rows = bs * q
        kwargs = {"bs": bs, "max_q_len": q} if dynamic else {"bs": bs}
        metadata, context = runner._build_capture(builder, **kwargs)
        runner.running_tokens_per_dp_rank.fill(rows)
        set_forward_context(
            attn_metadata=metadata,
            atom_config=runner.config,
            context=context,
            running_tokens_per_dp_rank=runner.running_tokens_per_dp_rank,
            in_hipgraph=True,
        )
        get_forward_context().tr_graph = "replay"
        try:
            input_ids = runner.forward_vars["input_ids"].gpu[:rows]
            positions = context.positions

            def forward(input_ids=input_ids, positions=positions):
                hidden = runner.model(input_ids, positions)
                runner.model.compute_logits(hidden)

            for _ in range(_WARMUP):
                forward()
            torch.cuda.synchronize()
            times = []
            for _ in range(_ITERS):
                start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
                start.record()
                forward()
                end.record()
                end.synchronize()
                times.append(start.elapsed_time(end))
        finally:
            reset_forward_context()
        ms = _all_max(statistics.median(times))
        result.append({"batch": bs, "q": q, "rows": rows, "target_ms": ms})
    return result


def startup_sps_table(runner, qs, max_rows):
    """The deployment's SPS table: cached, or measured now on every rank."""
    from atom.spec_decode.calibration import calibration_identity

    identity = calibration_identity(runner.config, runner.device)
    shapes = _shapes(runner, qs)
    path = _cache_path(identity, shapes, qs)
    # Every rank must take the same branch: the sweep runs collectives.
    cached = _read(path, identity, shapes) if path is not None and shapes else None
    if not _all_true(cached is not None):
        cached = None
    if cached is not None:
        # Nodes can hold caches of different runs under one key: reduce the
        # cached timings by MAX over TP and DP, as a fresh sweep does, so every
        # worker builds the same table.
        records = [dict(r) for r in cached["shapes"]]
        for r, ms in zip(records, _all_max_list([r["target_ms"] for r in records])):
            r["target_ms"] = ms
        source = f"cache {path}"
    else:
        if not shapes:
            raise RuntimeError(
                "DSpark startup SPS found no captured decode token graph to time; "
                "use --cudagraph-mode TR_HYBRID or supply a calibration_profile"
            )
        started = time.perf_counter()
        records = _measure(runner, shapes)
        source = f"measured in {time.perf_counter() - started:.1f}s"
        # Every worker writes the (MAX-reduced, identical) result: the default
        # cache directory is node-local, and every worker must find it before
        # any may reuse it. A unique temporary name and an atomic replace keep
        # concurrent writers of one directory safe.
        if path is not None:
            data = {
                "identity": identity,
                "max_num_seqs": runner.config.max_num_seqs,
                "points": [[r["rows"], r["target_ms"]] for r in records],
                "shapes": records,
                "scope": "startup: max TP/DP rank median per captured decode token "
                "graph at the capture context; target incl. eager regions and LM "
                "head; no sampler/draft",
            }
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                tmp = path.with_suffix(f".{os.getpid()}.{dist.get_rank()}.tmp")
                tmp.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
                tmp.replace(path)
            except OSError as e:
                logger.warning("DSpark SPS cache not written to %s: %s", path, e)
    table = sps_table_from_points(
        [(r["rows"], r["target_ms"]) for r in records], max_rows
    )
    if dist.get_rank() == 0:
        logger.info(
            "DSpark SPS from %d decode graphs (%s): %s",
            len(records),
            source,
            ", ".join(f"{r['rows']}:{r['target_ms']:.2f}ms" for r in records),
        )
    return torch.tensor(table, dtype=torch.float32, device=runner.device)
