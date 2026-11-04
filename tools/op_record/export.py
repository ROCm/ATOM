#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Turn an `ATOM_OP_RECORD_DIR` recording into per-op call counts over a time
window, and into aiter tuner CSVs (the tuner's key columns plus ``count``)
for `aiter/utility/op_ab.py`.

A signature's count over the window is its calls in the window's steps plus,
for each graph, the graph's replays in those steps times the signature's
calls in the graph's recording. Only steps wholly inside the window count.

    python tools/op_record/export.py REC_DIR --aiperf-export profile_export.jsonl --out OUT

writes, for one rank (``--rank``, default the first):

- ``ops_all.csv``: every signature with a count in the window, every depth;
- ``<family>.csv``: one per tuner family whose production entry was called
  (`is_family_entry`), and one per replayed op (`REPLAY_OPS`: attention and
  GDN, each call at the query/context lengths its step ran with, binned by
  total tokens);
- ``summary.json``: the window, its steps, the mean step time, the families'
  coverage, and the recording's meta.
"""

import argparse
import csv
import json
import math
import os
import statistics
import sys
from collections import Counter, defaultdict
from collections.abc import Callable
from dataclasses import dataclass


def _read_jsonl(path: str) -> list[dict]:
    out = []
    if not os.path.exists(path):
        return out
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                # The last line of a killed run may be cut short.
                continue
    return out


@dataclass
class Recording:
    rank_dir: str
    meta: dict
    sigs: dict[int, dict]
    graphs: dict[int, Counter]
    steps: list[dict]


def load(rank_dir: str) -> Recording:
    with open(os.path.join(rank_dir, "meta.json")) as f:
        meta = json.load(f)
    sigs = {s["id"]: s for s in _read_jsonl(os.path.join(rank_dir, "sigs.jsonl"))}
    graphs = {
        g["gid"]: Counter({int(k): v for k, v in g["calls"].items()})
        for g in _read_jsonl(os.path.join(rank_dir, "graphs.jsonl"))
    }
    steps = _read_jsonl(os.path.join(rank_dir, "steps.jsonl"))
    return Recording(rank_dir, meta, sigs, graphs, steps)


def rank_dirs(rec_dir: str) -> list[str]:
    if os.path.exists(os.path.join(rec_dir, "meta.json")):
        return [rec_dir]
    return sorted(
        os.path.join(rec_dir, d)
        for d in os.listdir(rec_dir)
        if os.path.exists(os.path.join(rec_dir, d, "meta.json"))
    )


def aiperf_window(path: str) -> tuple[float, float]:
    """The first start and last end, in seconds, of the profiling-phase
    requests of an aiperf ``profile_export.jsonl``."""
    starts, ends = [], []
    for record in _read_jsonl(path):
        meta = record.get("metadata", record)
        phase = str(meta.get("benchmark_phase", "profiling")).lower()
        if "profil" not in phase:
            continue
        start, end = meta.get("request_start_ns"), meta.get("request_end_ns")
        if start is None or end is None:
            continue
        starts.append(start)
        ends.append(end)
    if not starts:
        raise SystemExit(f"{path}: no profiling-phase request with start and end")
    return min(starts) / 1e9, max(ends) / 1e9


def window_steps(rec: Recording, t0: float, t1: float) -> list[dict]:
    return [
        s
        for s in rec.steps
        if s.get("kind") == "step" and s["t0"] >= t0 and s["t1"] <= t1
    ]


def count_calls(rec: Recording, steps: list[dict]) -> tuple[Counter, Counter]:
    """(signature id -> calls, graph id -> replays) over ``steps``."""
    replays: Counter = Counter()
    calls: Counter = Counter()
    for s in steps:
        for sid, n in s.get("calls", {}).items():
            calls[int(sid)] += n
        for gid, n in s.get("replays", {}).items():
            replays[int(gid)] += n
    for gid, n in replays.items():
        for sid, m in rec.graphs.get(gid, {}).items():
            calls[sid] += n * m
    return calls, replays


# Tuner families: a production entry's signature -> the tuner's key columns.


def _bind(sig: dict, params: list[str]) -> dict:
    bound = dict(zip(params, sig["args"]))
    bound.update(sig.get("kwargs", {}))
    return bound


def _is_tensor(x) -> bool:
    return isinstance(x, dict) and "shape" in x


def _rows(t: dict) -> int:
    return math.prod(t["shape"][:-1])


def _torch_dtype(name) -> str:
    if name is None:
        return "None"
    name = str(name)
    return name if name.startswith("torch.") else f"torch.{name}"


_BLOCKSCALE = ["XQ", "WQ", "x_scale", "w_scale", "dtype", "isBpreshuffled", "split_k"]
_BLOCKSCALE_BPRESHUFFLE = ["XQ", "WQ", "x_scale", "w_scale", "dtype", "out"]
_A8W8 = ["XQ", "WQ", "x_scale", "w_scale", "bias", "dtype", "splitK"]
_A16W16 = ["A", "B", "bias", "otype", "scale_a", "scale_b", "scale_c"]
_A16W16_OUT = ["A", "B", "out", "bias", "otype", "scale_a", "scale_b", "scale_c"]
_FUSED_MOE = [
    "hidden_states",
    "w1",
    "w2",
    "topk_weight",
    "topk_ids",
    "expert_mask",
    "activation",
    "quant_type",
    "doweight_stage1",
    "w1_scale",
    "w2_scale",
    "a1_scale",
    "a2_scale",
    "block_size_M",
    "num_local_tokens",
    "moe_sorting_dispatch_policy",
    "dtype",
]
# csrc/include/aiter_enum.h ActivationType / QuantType values, for the
# names the tuner CSVs use.
_ACTIVATION = {
    -1: "No",
    0: "Silu",
    1: "Gelu",
    2: "Swiglu",
    3: "Situv2",
    4: "GeluTanh",
    5: "Relu2",
}
_QUANT = {
    0: "No",
    1: "per_Tensor",
    2: "per_Token",
    3: "per_1x32",
    4: "per_1x128",
    5: "per_128x128",
    6: "per_256x128",
    7: "per_1024x128",
    8: "iq2r_2bit",
}


def _gemm_mnk(x: dict, w: dict) -> dict:
    return {"M": _rows(x), "N": w["shape"][0], "K": x["shape"][-1]}


def _blockscale(sig: dict) -> tuple[str, dict] | None:
    a = _bind(sig, _BLOCKSCALE)
    if not (_is_tensor(a["XQ"]) and _is_tensor(a["WQ"])):
        return None
    if _is_tensor(a.get("x_scale")) and a["x_scale"]["dtype"] != "float32":
        return None  # E8M0 group32 scales: another config, no tuner here
    family = "a8w8_blockscale"
    if a.get("isBpreshuffled"):
        family = "a8w8_blockscale_bpreshuffle"
    return family, _gemm_mnk(a["XQ"], a["WQ"])


def _blockscale_bpreshuffle(sig: dict) -> tuple[str, dict] | None:
    a = _bind(sig, _BLOCKSCALE_BPRESHUFFLE)
    if not (_is_tensor(a["XQ"]) and _is_tensor(a["WQ"])):
        return None
    return "a8w8_blockscale_bpreshuffle", _gemm_mnk(a["XQ"], a["WQ"])


def _a8w8(family: str) -> Callable[[dict], tuple[str, dict] | None]:
    def keys(sig: dict):
        a = _bind(sig, _A8W8)
        if not (_is_tensor(a["XQ"]) and _is_tensor(a["WQ"])):
            return None
        return family, _gemm_mnk(a["XQ"], a["WQ"]) | {
            "q_dtype_w": _torch_dtype(a["WQ"]["dtype"])
        }

    return keys


def _a16w16(params: list[str]) -> Callable[[dict], tuple[str, dict] | None]:
    def keys(sig: dict):
        a = _bind(sig, params)
        A, B = a["A"], a["B"]
        if not (_is_tensor(A) and _is_tensor(B)):
            return None
        return "bf16", _gemm_mnk(A, B) | {
            "bias": _is_tensor(a.get("bias")),
            "dtype": _torch_dtype(A["dtype"]),
            "outdtype": _torch_dtype(a.get("otype") or A["dtype"]),
            "scaleAB": _is_tensor(a.get("scale_a")),
            "bpreshuffle": False,
        }

    return keys


def _fused_moe(sig: dict) -> tuple[str, dict] | None:
    a = _bind(sig, _FUSED_MOE)
    x, w1, w2, ids = a["hidden_states"], a["w1"], a["w2"], a["topk_ids"]
    if not all(_is_tensor(t) for t in (x, w1, w2, ids)):
        return None
    # aiter.fused_moe.get_inter_dim
    experts, model_dim, inter_dim = w2["shape"]
    inter_dim *= model_dim // w1["shape"][-1]
    activation = a.get("activation", 0)
    quant = a.get("quant_type", 0)
    q_dtype_w = _torch_dtype(w1["dtype"])
    return "fmoe", {
        "token": ids["shape"][0],
        "model_dim": model_dim,
        "inter_dim": inter_dim,
        "expert": experts,
        "topk": ids["shape"][1],
        "act_type": f"ActivationType.{_ACTIVATION.get(activation, activation)}",
        "dtype": _torch_dtype(a.get("dtype") or x["dtype"]),
        # The activation's quant dtype follows the weight's for the quantized
        # paths a tuner covers; aiter.fused_moe.resolve_activation_dtype has the
        # exceptions (bf16 x MXFP4), which keep the hidden dtype.
        "q_dtype_a": q_dtype_w if quant else _torch_dtype(x["dtype"]),
        "q_dtype_w": q_dtype_w,
        "q_type": f"QuantType.{_QUANT.get(quant, quant)}",
        "use_g1u1": int(w1["shape"][1] == 2 * inter_dim),
        "doweight_stage1": int(bool(a.get("doweight_stage1", False))),
    }


@dataclass(frozen=True)
class Family:
    tuner: str  # relative to the aiter checkout
    tuner_args: tuple[str, ...]
    keys: tuple[str, ...]
    size_key: str  # the per-call size column, bucketed when too many differ


FAMILIES = {
    "a8w8_blockscale": Family(
        "csrc/ck_gemm_a8w8_blockscale/gemm_a8w8_blockscale_tune.py",
        (),
        ("M", "N", "K"),
        "M",
    ),
    "a8w8_blockscale_bpreshuffle": Family(
        "csrc/ck_gemm_a8w8_blockscale/gemm_a8w8_blockscale_tune.py",
        ("--preshuffle",),
        ("M", "N", "K"),
        "M",
    ),
    "a8w8": Family(
        "csrc/ck_gemm_a8w8/gemm_a8w8_tune.py", (), ("M", "N", "K", "q_dtype_w"), "M"
    ),
    "a8w8_bpreshuffle": Family(
        "csrc/ck_gemm_a8w8_bpreshuffle/gemm_a8w8_bpreshuffle_tune.py",
        (),
        ("M", "N", "K", "q_dtype_w"),
        "M",
    ),
    "bf16": Family(
        "csrc/gemm_a16w16/gemm_a16w16_tune.py",
        (),
        ("M", "N", "K", "bias", "dtype", "outdtype", "scaleAB", "bpreshuffle"),
        "M",
    ),
    "fmoe": Family(
        "csrc/ck_gemm_moe_2stages_codegen/gemm_moe_tune.py",
        (),
        (
            "token",
            "model_dim",
            "inter_dim",
            "expert",
            "topk",
            "act_type",
            "dtype",
            "q_dtype_a",
            "q_dtype_w",
            "q_type",
            "use_g1u1",
            "doweight_stage1",
        ),
        "token",
    ),
}

OPS: dict[str, Callable[[dict], tuple[str, dict] | None]] = {
    "gemm_a8w8_blockscale": _blockscale,
    "gemm_a8w8_blockscale_bpreshuffle": _blockscale_bpreshuffle,
    "gemm_a8w8": _a8w8("a8w8"),
    "gemm_a8w8_bpreshuffle": _a8w8("a8w8_bpreshuffle"),
    "gemm_a16w16": _a16w16(_A16W16),
    "gemm_a16w16_out": _a16w16(_A16W16_OUT),
    "fused_moe_": _fused_moe,
}


def is_family_entry(sig: dict) -> bool:
    """The outermost call with a tuner family: the production entry a tuner's
    ``--run_config`` times. An ATOM wrapper registered through aiter's
    `torch_compile_guard` may sit above it; another family's op may not, or
    a call would be counted twice."""
    return sig["op"] in OPS and not any(p in OPS for p in sig.get("parents", []))


def family_rows(rec: Recording, calls: Counter) -> dict[str, Counter]:
    """family -> (key columns as a tuple of (name, value)) -> count, over the
    signatures that are a family's entry (`is_family_entry`)."""
    rows: dict[str, Counter] = defaultdict(Counter)
    for sid, n in calls.items():
        sig = rec.sigs.get(sid)
        if sig is None or not is_family_entry(sig):
            continue
        found = OPS[sig["op"]](sig)
        if found is None:
            continue
        family, keys = found
        rows[family][tuple(keys.items())] += n
    return rows


def bucket(rows: Counter, size_key: str, max_unique: int) -> tuple[Counter, bool]:
    """``rows`` with each group (all keys but ``size_key``) cut to at most
    ``max_unique`` sizes: equal-count buckets over the size, each kept as its
    most called size, weighted with the bucket's calls."""
    groups: dict[tuple, Counter] = defaultdict(Counter)
    for key, n in rows.items():
        rest = tuple(kv for kv in key if kv[0] != size_key)
        groups[rest][dict(key)[size_key]] += n
    out: Counter = Counter()
    bucketed = False
    for rest, sizes in groups.items():
        if len(sizes) <= max_unique:
            for size, n in sizes.items():
                out[_with_size(rest, size_key, size, rows)] += n
            continue
        bucketed = True
        total = sum(sizes.values())
        ordered = sorted(sizes.items())
        per_bucket = total / max_unique
        current: Counter = Counter()
        filled = 0
        for i, (size, n) in enumerate(ordered):
            current[size] = n
            filled += n
            last = i == len(ordered) - 1
            if filled >= per_bucket or last:
                rep = max(current.items(), key=lambda kv: (kv[1], -kv[0]))[0]
                out[_with_size(rest, size_key, rep, rows)] += sum(current.values())
                current, filled = Counter(), 0
    return out, bucketed


def _with_size(rest: tuple, size_key: str, size: int, rows: Counter) -> tuple:
    order = [k for k, _ in next(iter(rows))]
    values = dict(rest) | {size_key: size}
    return tuple((k, values[k]) for k in order)


# Replay families: ops whose cost follows values no recorded shape carries --
# attention's per-sequence query and context lengths, which a graph's
# recording holds as whatever the capture ran with -- or that no tuner times.
# Each call is replayed by its owner's harness: aiter's
# `aiter/utility/op_replay.py` for aiter's kernels, ATOM's
# `tools/op_record/replay.py` for ATOM's own (the GDN ops), at the lengths its
# step's batch carried.


@dataclass(frozen=True)
class ReplayOp:
    family: str
    owner: str  # "aiter" or "atom": whose two checkouts an A/B compares
    # (parameter name, position, minus): the tensor whose leading dimension,
    # less ``minus`` (1 for a cu_seqlens), counts the sequences a call serves;
    # None for an op no per-sequence value prices (`fused_gdn_gating`).
    seqs: tuple[str, int, int] | None
    # Which of a mixed step's sequences it serves: "decode", "prefill", or
    # (parameter name, position) of its max query length -- 1 means decode.
    phase: str | tuple[str, int] = "decode"


REPLAY_SCRIPTS = {
    "aiter": "aiter/utility/op_replay.py",
    "atom": "tools/op_record/replay.py",
}
REPLAY_OPS = {
    # decode attention
    "pa_decode_gluon": ReplayOp(
        "attn_pa_decode_gluon", "aiter", ("context_lengths", 4, 0)
    ),
    "pa_fwd_asm": ReplayOp("attn_pa_fwd_asm", "aiter", ("context_lens", 4, 0)),
    "pa_persistent_fwd": ReplayOp(
        "attn_pa_persistent", "aiter", ("context_lens", 8, 0)
    ),
    # decode or prefill attention
    "unified_attention": ReplayOp(
        "attn_unified", "aiter", ("seqused_k", 6, 0), ("max_seqlen_q", 5)
    ),
    # prefill attention: the public entry, and the one ATOM's fp8 path calls
    "flash_attn_varlen_func": ReplayOp(
        "attn_prefill_varlen", "aiter", ("cu_seqlens_q", 3, 1), "prefill"
    ),
    "_flash_attn_varlen_forward": ReplayOp(
        "attn_prefill_varlen_fwd", "aiter", ("cu_seqlens_q", 3, 1), "prefill"
    ),
    # GDN linear attention, ATOM's own Triton kernels
    "fused_recurrent_gated_delta_rule": ReplayOp(
        "gdn_fused_recurrent", "atom", ("cu_seqlens", 8, 1)
    ),
    "causal_conv1d_update": ReplayOp("gdn_conv1d_update", "atom", ("x", 0, 0)),
    "causal_conv1d_fn": ReplayOp(
        "gdn_conv1d_fn", "atom", ("query_start_loc", 4, 1), "prefill"
    ),
    "chunk_gated_delta_rule": ReplayOp(
        "gdn_chunk", "atom", ("cu_seqlens", 8, 1), "prefill"
    ),
    "fused_gdn_gating": ReplayOp("gdn_gating", "atom", None),
}


def _arg(sig: dict, name: str, position: int):
    if name in sig.get("kwargs", {}):
        return sig["kwargs"][name]
    args = sig["args"]
    return args[position] if position < len(args) else None


def is_replay_entry(sig: dict) -> bool:
    return sig["op"] in REPLAY_OPS and not any(
        p in OPS or p in REPLAY_OPS for p in sig.get("parents", [])
    )


def _served(
    steps_q: list[int], steps_ctx: list[int], n: int, phase: str = "decode"
) -> dict:
    """The (q, ctx) of the ``n`` sequences a call served: its ``phase``'s
    sequences of the step, or all of them -- whichever count is ``n`` -- and a
    graph's padded rows as zeros (they read and write nothing)."""
    seqs = list(zip(steps_q, steps_ctx))
    own = [x for x in seqs if (x[0] > 1) == (phase == "prefill")]
    for chosen in (own, seqs):
        if len(chosen) == n:
            break
    else:
        chosen = seqs[:n]
    chosen = chosen + [(0, 0)] * (n - len(chosen))
    return {"q": [q for q, _ in chosen], "ctx": [c for _, c in chosen]}


def replay_calls(rec: Recording, steps: list[dict]) -> dict[int, list[tuple[str, int]]]:
    """signature id -> [(work, calls)] over ``steps``: a call's work is the
    JSON of the (q, ctx) of the sequences it served (`_served`), "{}" for an
    op no per-sequence value prices. A graph's calls count once per replay."""
    out: dict[int, list] = defaultdict(list)
    for s in steps:
        here: Counter = Counter()
        for sid, n in s.get("calls", {}).items():
            sig = rec.sigs.get(int(sid))
            if sig is not None and is_replay_entry(sig):
                here[int(sid)] += n
        for gid, n in s.get("replays", {}).items():
            for sid, m in rec.graphs.get(int(gid), {}).items():
                sig = rec.sigs.get(sid)
                if sig is not None and is_replay_entry(sig):
                    here[sid] += n * m
        for sid, n in here.items():
            sig = rec.sigs[sid]
            op = REPLAY_OPS[sig["op"]]
            work = {}
            if op.seqs is not None:
                tensor = _arg(sig, op.seqs[0], op.seqs[1])
                rows = tensor["shape"][0] - op.seqs[2]
                phase = op.phase
                if not isinstance(phase, str):
                    phase = "decode" if _arg(sig, *phase) == 1 else "prefill"
                work = _served(s.get("q") or [], s.get("ctx") or [], rows, phase)
            out[sid].append((json.dumps(work, separators=(",", ":")), n))
    return out


def _work_size(work: str) -> int:
    w = json.loads(work)
    return sum(w.get("q", [])) + sum(w.get("ctx", []))


def replay_bins(
    instances: list[tuple[str, int]], bins: int
) -> tuple[list[tuple[str, int]], bool]:
    """``instances`` cut to at most ``bins`` (work, calls): equal-call bins
    over the work's total tokens (query + context), each kept as its
    call-weighted median member and weighted with the bin's calls."""
    merged: Counter = Counter()
    for work, n in instances:
        merged[work] += n
    if len(merged) <= bins:
        return sorted(merged.items(), key=lambda kv: _work_size(kv[0])), False
    ordered = sorted(merged.items(), key=lambda kv: (_work_size(kv[0]), kv[0]))
    per_bin = sum(merged.values()) / bins
    out, group, filled = [], [], 0
    for i, (work, n) in enumerate(ordered):
        group.append((work, n))
        filled += n
        if filled >= per_bin or i == len(ordered) - 1:
            seen = 0
            for rep, m in group:
                seen += m
                if seen >= filled / 2:
                    break
            out.append((rep, filled))
            group, filled = [], 0
    return out, True


def _replay_label(sig: dict, work: str) -> str:
    first = next((a for a in sig["args"] if _is_tensor(a)), None)
    shape = "x".join(map(str, first["shape"])) if first else "-"
    w = json.loads(work)
    if "ctx" not in w:
        return f"{sig['op']} {shape}"
    live = [c for c in w["ctx"] if c]
    return (
        f"{sig['op']} {shape} seqs={len(live)}/{len(w['ctx'])} "
        f"q_sum={sum(w['q'])} ctx_sum={sum(w['ctx'])} ctx_max={max(w['ctx'], default=0)}"
    )


def write_replay_csv(path: str, rows: list[tuple[dict, str, int]]) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["label", "op", "sig", "work", "count"])
        for sig, work, n in rows:
            call = {"args": sig["args"], "kwargs": sig.get("kwargs", {})}
            writer.writerow(
                [
                    _replay_label(sig, work),
                    sig["op"],
                    json.dumps(call, separators=(",", ":")),
                    work,
                    n,
                ]
            )


def write_family_csv(path: str, family: Family, rows: Counter) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([*family.keys, "count"])
        for key, n in sorted(rows.items(), key=lambda kv: -kv[1]):
            values = dict(key)
            writer.writerow([*(values[k] for k in family.keys), n])


def write_ops_all(path: str, rec: Recording, calls: Counter) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["sig", "op", "depth", "count", "args", "kwargs"])
        for sid, n in sorted(calls.items(), key=lambda kv: -kv[1]):
            sig = rec.sigs.get(sid, {"op": "?", "depth": -1, "args": [], "kwargs": {}})
            writer.writerow(
                [
                    sid,
                    sig["op"],
                    sig["depth"],
                    n,
                    json.dumps(sig["args"], separators=(",", ":")),
                    json.dumps(sig["kwargs"], separators=(",", ":")),
                ]
            )


def export(
    rec_dir: str,
    out_dir: str,
    window: tuple[float, float] | None,
    rank: str | None = None,
    max_unique: int = 2000,
    replay_bin_count: int = 16,
) -> dict:
    dirs = rank_dirs(rec_dir)
    if not dirs:
        raise SystemExit(f"{rec_dir}: no recording (no meta.json)")
    if rank is not None:
        dirs = [d for d in dirs if os.path.basename(d) == rank]
        if not dirs:
            raise SystemExit(f"{rec_dir}: no rank {rank!r}")
    rec = load(dirs[0])
    if window is None:
        window = _after_last_recording(rec)
    t0, t1 = window
    steps = window_steps(rec, t0, t1)
    if not steps:
        raise SystemExit(
            f"{rec.rank_dir}: no step inside the window [{t0:.3f}, {t1:.3f}]"
        )
    calls, replays = count_calls(rec, steps)
    os.makedirs(out_dir, exist_ok=True)
    write_ops_all(os.path.join(out_dir, "ops_all.csv"), rec, calls)

    families = {}
    for name, rows in sorted(family_rows(rec, calls).items()):
        family = FAMILIES[name]
        rows, bucketed = bucket(rows, family.size_key, max_unique)
        write_family_csv(os.path.join(out_dir, f"{name}.csv"), family, rows)
        families[name] = {
            "csv": f"{name}.csv",
            "tuner": family.tuner,
            "tuner_args": list(family.tuner_args),
            "rows": len(rows),
            "calls": sum(rows.values()),
            "bucketed": bucketed,
        }

    replay_rows: dict[str, list] = defaultdict(list)
    replay_bucketed: dict[str, bool] = defaultdict(bool)
    replay_owner: dict[str, str] = {}
    for sid, instances in sorted(replay_calls(rec, steps).items()):
        sig = rec.sigs[sid]
        op = REPLAY_OPS[sig["op"]]
        reps, bucketed = replay_bins(instances, replay_bin_count)
        replay_bucketed[op.family] |= bucketed
        replay_owner[op.family] = op.owner
        replay_rows[op.family] += [(sig, work, n) for work, n in reps]
    for name, rows in sorted(replay_rows.items()):
        write_replay_csv(os.path.join(out_dir, f"{name}.csv"), rows)
        families[name] = {
            "csv": f"{name}.csv",
            "driver": "replay",
            "owner": replay_owner[name],
            "replay": REPLAY_SCRIPTS[replay_owner[name]],
            "rows": len(rows),
            "calls": sum(r[2] for r in rows),
            "bucketed": replay_bucketed[name],
            "bins_per_signature": replay_bin_count,
        }

    # A depth-0 op is covered when it is a family's entry or wraps one.
    depth0, covered = Counter(), set()
    for sid, n in calls.items():
        sig = rec.sigs.get(sid)
        if sig is None:
            continue
        if sig["depth"] == 0:
            depth0[sig["op"]] += n
        if is_family_entry(sig) or is_replay_entry(sig):
            covered.add((sig.get("parents") or [sig["op"]])[0])
    step_times = [(s["t1"] - s["t0"]) * 1e6 for s in steps]
    summary = {
        "rank_dir": rec.rank_dir,
        "ranks_recorded": [os.path.basename(d) for d in rank_dirs(rec_dir)],
        "window": [t0, t1],
        "steps": len(steps),
        "dummy_steps": sum(1 for s in steps if s.get("dummy")),
        "prefill_steps": sum(1 for s in steps if s.get("prefill_seqs")),
        "decode_only_steps": sum(
            1 for s in steps if not s.get("prefill_seqs") and s.get("decode_seqs")
        ),
        "step_time_us_mean": statistics.fmean(step_times),
        "step_time_us_median": statistics.median(step_times),
        "graph_replays": sum(replays.values()),
        "unknown_graph_replays": replays.get(-1, 0),
        "families": families,
        "depth0_calls_by_op": dict(depth0.most_common()),
        "depth0_ops_without_family": sorted(set(depth0) - covered),
        "meta": rec.meta,
    }
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1, default=str)
    return summary


def _after_last_recording(rec: Recording) -> tuple[float, float]:
    """No window given: every step after the last graph recording."""
    graph_lines = _read_jsonl(os.path.join(rec.rank_dir, "graphs.jsonl"))
    start = max((g["t"] for g in graph_lines), default=0.0)
    return start, math.inf


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("rec_dir", help="ATOM_OP_RECORD_DIR, or one rank's directory")
    p.add_argument("--out", required=True)
    w = p.add_mutually_exclusive_group()
    w.add_argument(
        "--aiperf-export",
        help="aiperf profile_export.jsonl; the window is its profiling requests",
    )
    w.add_argument("--window", help="t0,t1 in seconds since the epoch")
    p.add_argument("--rank", help="rank directory name (default: the first)")
    p.add_argument("--max-unique-m", type=int, default=2000)
    p.add_argument(
        "--attn-bins",
        type=int,
        default=16,
        help="context-length bins per replayed signature (attention)",
    )
    args = p.parse_args(argv)

    window = None
    if args.aiperf_export:
        window = aiperf_window(args.aiperf_export)
    elif args.window:
        t0, t1 = (float(v) for v in args.window.split(","))
        window = (t0, t1)
    summary = export(
        args.rec_dir, args.out, window, args.rank, args.max_unique_m, args.attn_bins
    )
    print(
        f"{summary['steps']} steps in window "
        f"(mean {summary['step_time_us_mean']:.0f} us); "
        f"families: {', '.join(summary['families']) or 'none'}; "
        f"depth-0 ops without a family: "
        f"{len(summary['depth0_ops_without_family'])}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
