#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Replay recorded calls of ATOM's own GDN kernels and time them, for aiter's
`aiter/utility/op_ab.py` (``driver: replay``, ``--package atom``): the two
sides of an A/B are ATOM checkouts, this file is the harness both run.

Each CSV row (`export.py` writes them) is one call: ``op``, its recorded
signature (``sig``), ``work`` -- the query and context length of each
sequence it served, zeros for a CUDA graph's padded rows -- and ``count``.
A replayer rebuilds every argument at its recorded shape, dtype and stride,
then the ones the lengths decide: offsets, state-slot indices (a padded row's
is -1, as the graph's are), state pools holding the slots the call touches,
the conv kernel's program map. Decays are negative and betas in (0, 1), as a
model's are. It times the call with `aiter.test_common.run_perftest`.

    python tools/op_record/replay.py ROWS.csv
"""

import csv
import inspect
import itertools
import json
import math
import sys
from collections.abc import Callable
from types import SimpleNamespace

import torch


def _is_tensor(spec) -> bool:
    return isinstance(spec, dict) and "shape" in spec


def _as_dtype(name):
    found = getattr(torch, str(name).removeprefix("torch."), None)
    return found if isinstance(found, torch.dtype) else None


def tensor(
    spec: dict, *, first: int | None = None, fill: str = "randn"
) -> torch.Tensor:
    """A tensor of the recorded shape, dtype and stride -- a view of a buffer
    just big enough -- with ``first`` replacing the leading dimension."""
    shape = list(spec["shape"])
    stride = list(spec.get("stride") or [])
    if first is not None:
        shape[0] = first
    if len(stride) != len(shape):
        stride = [math.prod(shape[i + 1 :]) for i in range(len(shape))]
    size = 0 if 0 in shape else 1 + sum((d - 1) * s for d, s in zip(shape, stride))
    dtype = _as_dtype(spec["dtype"])
    if fill == "zeros" or not dtype.is_floating_point:
        base = torch.zeros(size, dtype=dtype, device="cuda")
    elif fill == "pos":
        base = (torch.rand(size, device="cuda") * 0.9 + 0.05).to(dtype)
    elif fill == "neg":
        base = (-torch.rand(size, device="cuda") * 0.1).to(dtype)
    else:
        base = (torch.randn(size, device="cuda") * 0.5).to(dtype)
    return base.as_strided(shape, stride)


def value(arg):
    if _is_tensor(arg):
        return tensor(arg)
    if isinstance(arg, list):
        return tuple(value(a) for a in arg)
    if isinstance(arg, str):
        if arg.startswith("<") and arg.endswith(">"):
            return None
        dtype = _as_dtype(arg)
        return dtype if dtype is not None else arg
    return arg


class Call:
    """A recorded call bound to ``fn``'s parameter names."""

    def __init__(self, fn: Callable, sig: dict):
        self.fn = fn
        names = list(inspect.signature(fn).parameters)
        self.positional = names[: len(sig["args"])]
        self.keywords = list(sig.get("kwargs", {}))
        self.recorded = dict(zip(names, sig["args"])) | sig.get("kwargs", {})
        self.values = {n: value(a) for n, a in self.recorded.items()}

    def has(self, name: str) -> bool:
        return self.recorded.get(name) is not None

    def set(self, name: str, v) -> None:
        if name in self.recorded:
            self.values[name] = v

    def args(self) -> tuple[Callable, tuple, dict]:
        return (
            self.fn,
            tuple(self.values[n] for n in self.positional),
            {n: self.values[n] for n in self.keywords},
        )


def cu_seqlens(lengths: list[int]) -> torch.Tensor:
    offsets = [0, *itertools.accumulate(lengths)]
    return torch.tensor(offsets, dtype=torch.int32, device="cuda")


def slots(q: list[int], spec: dict) -> torch.Tensor:
    """One state slot per served sequence, -1 for a padded row, as the
    recorded index tensor's shape and dtype."""
    ids = [-1] * len(q)
    live = 0
    for i, n in enumerate(q):
        if n:
            ids[i] = live
            live += 1
    return torch.tensor(ids, dtype=_as_dtype(spec["dtype"]), device="cuda")


def _live(q: list[int]) -> int:
    return max(1, sum(1 for n in q if n))


def _check_tokens(call: Call, name: str, dim: int, q: list[int]) -> None:
    rows = call.recorded[name]["shape"][dim]
    if sum(q) > rows:
        raise ValueError(f"served {sum(q)} tokens, recorded {name} has {rows}")


# ------------------------------------------------------------------ replayers


def fused_recurrent_gated_delta_rule(sig: dict, work: dict):
    from atom.model_ops.fla_ops.fused_recurrent import (
        fused_recurrent_gated_delta_rule as fn,
    )

    call = Call(fn, sig)
    q = work["q"]
    _check_tokens(call, "q", 1, q)
    call.set("cu_seqlens", cu_seqlens(q))
    call.set("initial_state", tensor(call.recorded["initial_state"], first=_live(q)))
    if call.has("ssm_state_indices"):
        ids = slots(q, call.recorded["ssm_state_indices"])
        call.set("ssm_state_indices", ids)
        call.set(
            "ssm_state_indices_in", ids if call.has("ssm_state_indices_in") else None
        )
    call.set("g", tensor(call.recorded["g"], fill="neg"))
    if call.has("beta"):
        call.set("beta", tensor(call.recorded["beta"], fill="pos"))
    return call.args()


def causal_conv1d_update(sig: dict, work: dict):
    from atom.model_ops.mamba_ops.causal_conv1d import causal_conv1d_update as fn

    call = Call(fn, sig)
    q = work["q"]
    call.set("conv_state", tensor(call.recorded["conv_state"], first=_live(q)))
    if call.has("conv_state_indices"):
        ids = slots(q, call.recorded["conv_state_indices"])
        call.set("conv_state_indices", ids)
        call.set(
            "conv_state_indices_in", ids if call.has("conv_state_indices_in") else None
        )
    return call.args()


def fused_gdn_gating(sig: dict, work: dict):
    from atom.model_ops.attention_gdn import fused_gdn_gating as fn

    call = Call(fn, sig)
    return call.args()


def causal_conv1d_fn(sig: dict, work: dict):
    """Prefill conv; its program map built beforehand, as ATOM's metadata
    builder does, not inside the timed call."""
    from atom.model_ops.mamba_ops.causal_conv1d import (
        causal_conv1d_fn as fn,
    )
    from atom.model_ops.mamba_ops.causal_conv1d import (
        compute_causal_conv1d_metadata,
    )

    call = Call(fn, sig)
    q, ctx = work["q"], work["ctx"]
    _check_tokens(call, "x", 1, q)
    n = len(q)
    starts = cu_seqlens(q)
    call.set("query_start_loc", starts)
    call.set("conv_states", tensor(call.recorded["conv_states"], first=n))
    ids = torch.arange(n, dtype=torch.int32, device="cuda")
    call.set("cache_indices", ids)
    call.set("cache_indices_in", ids if call.has("cache_indices_in") else None)
    if call.has("has_initial_state"):
        has = [c > t for t, c in zip(q, ctx)]
        call.set(
            "has_initial_state", torch.tensor(has, dtype=torch.bool, device="cuda")
        )
    nums, batch_ptr, offsets = compute_causal_conv1d_metadata(
        starts, call.recorded["x"]["shape"][1]
    )
    if "metadata" not in call.positional and "metadata" not in call.keywords:
        call.keywords.append("metadata")
    call.values["metadata"] = SimpleNamespace(
        nums_dict=nums, batch_ptr=batch_ptr, token_chunk_offset_ptr=offsets
    )
    return call.args()


def chunk_gated_delta_rule(sig: dict, work: dict):
    from atom.model_ops.fla_ops.chunk import chunk_gated_delta_rule as fn

    call = Call(fn, sig)
    q = work["q"]
    if sum(q) != call.recorded["q"]["shape"][1]:
        raise ValueError(
            f"served {sum(q)} tokens, recorded q has {call.recorded['q']['shape'][1]}"
        )
    call.set("cu_seqlens", cu_seqlens(q))
    if call.has("initial_state"):
        call.set("initial_state", tensor(call.recorded["initial_state"], first=len(q)))
    call.set("g", tensor(call.recorded["g"], fill="neg"))
    call.set("beta", tensor(call.recorded["beta"], fill="pos"))
    return call.args()


REPLAYERS: dict[str, Callable[[dict, dict], tuple[Callable, tuple, dict]]] = {
    "fused_recurrent_gated_delta_rule": fused_recurrent_gated_delta_rule,
    "causal_conv1d_update": causal_conv1d_update,
    "fused_gdn_gating": fused_gdn_gating,
    "causal_conv1d_fn": causal_conv1d_fn,
    "chunk_gated_delta_rule": chunk_gated_delta_rule,
}


def replay(rows_csv: str, extra_args: list[str] | None = None) -> tuple[list, list]:
    """(rows, results) for `op_ab.py`, as aiter's `op_replay.replay`."""
    from aiter.test_common import run_perftest

    iters, warmup = 101, 5
    extra = list(extra_args or [])
    for flag in ("--iters", "--warmup"):
        if flag in extra:
            v = int(extra[extra.index(flag) + 1])
            iters, warmup = (v, warmup) if flag == "--iters" else (iters, v)

    with open(rows_csv, newline="") as f:
        rows = list(csv.DictReader(f))
    results = []
    for row in rows:
        label = row.get("label", row["op"])
        try:
            fn, args, kwargs = REPLAYERS[row["op"]](
                json.loads(row["sig"]), json.loads(row["work"])
            )
            _, us = run_perftest(
                fn, *args, num_iters=iters, num_warmup=warmup, **kwargs
            )
            status = "ok" if math.isfinite(us) and us > 0 else f"error:time {us}"
            results.append({"shape": label, "e2e_us": us, "status": status})
        except Exception as e:  # noqa: BLE001  one row's failure is that row's result
            results.append({"shape": label, "e2e_us": -1, "status": f"error:{e}"})
        finally:
            torch.cuda.empty_cache()
    return rows, results


if __name__ == "__main__":
    for row, result in zip(*replay(sys.argv[1], sys.argv[2:])):
        print(
            f"{result['e2e_us']:10.2f} us  {result['status']:<10} {row.get('label', '')}"
        )
