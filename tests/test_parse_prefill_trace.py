"""CPU tests for `tools/parse_prefill_trace.py` on a synthetic two-layer trace.

Each op's duration is a distinct power of two, so a half's summed time names
exactly which ops were attributed to it.
"""

import gzip
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import parse_prefill_trace

CPU, MAIN, SIDE = 1, 7, 9

# (module tag or None, kernel) in launch order. Layer 0 is KDA, layer 1 DSA.
OPS = [
    (None, "embedding"),
    ("model.layers.0.self_attn.in_proj", "gemm"),
    ("rmsnorm_gated", "norm"),
    ("model.layers.0.self_attn.o_proj", "gemm"),
    ("apply_attn_res", "attn_res"),
    ("model.layers.0.mlp.down_proj", "gemm"),
    (None, "all_reduce"),
    ("apply_attn_res", "attn_res"),
    ("model.layers.1.self_attn.fused_qkv_a_proj", "gemm"),
    ("dcp_sparse_prefill", "mla"),
    ("model.layers.1.self_attn.o_proj", "gemm"),
    ("apply_attn_res", "attn_res"),
    ("model.layers.1.block_sparse_moe.experts.fused_moe", "moe"),
    (None, "all_reduce"),
    ("apply_attn_res", "attn_res"),
]


def event(cat, name, ts, dur, tid, **args):
    return {"ph": "X", "cat": cat, "name": name, "ts": ts, "dur": dur} | {
        "pid": 0,
        "tid": tid,
        "args": args,
    }


def write_trace(tmp_path, ops):
    events = [event("user_annotation", "prefill[bs=1 tok=4 ctx=4]", 0.0, 100.0, CPU)]
    ts = 1000.0
    for i, (tag, kernel) in enumerate(ops):
        dur = float(2**i)
        ts += i == len(ops) - 1  # one idle microsecond inside the span
        events.append(
            event("cuda_runtime", "hipLaunchKernel", 1.0 + i, 0.5, CPU, correlation=i)
        )
        events.append(
            event("kernel", kernel, ts, dur, MAIN, correlation=i, stream=MAIN)
        )
        if tag:
            events.append(event("gpu_user_annotation", tag, ts, dur, MAIN))
        ts += dur
    # Off the main stream, and launched after the span: neither is the forward's.
    events.append(
        event("cuda_runtime", "hipMemcpyAsync", 50.0, 0.5, CPU, correlation=100)
    )
    events.append(
        event("gpu_memcpy", "Memcpy", 1000.0, 3.0, SIDE, correlation=100, stream=SIDE)
    )
    events.append(
        event("cuda_runtime", "hipLaunchKernel", 200.0, 0.5, CPU, correlation=200)
    )
    events.append(
        event("kernel", "decode", ts, 5.0, MAIN, correlation=200, stream=MAIN)
    )

    trace = tmp_path / "rank0.pt.trace.json.gz"
    with gzip.open(trace, "wt") as f:
        json.dump({"traceEvents": events}, f)
    model = tmp_path / "model"
    model.mkdir()
    config = {"linear_attn_config": {"kda_layers": [1]}, "num_hidden_layers": 2}
    (model / "config.json").write_text(json.dumps({"text_config": config}))
    return str(trace), str(model)


def test_assigns_every_op_to_a_layer_half(tmp_path):
    [row] = parse_prefill_trace.breakdown(*write_trace(tmp_path, OPS))

    assert row["forward"] == "bs=1 tok=4 ctx=4"
    # Layer 0 (KDA): ops 0-3 attention, 4-6 MLP. Layer 1 (DSA): the attention
    # residual opening it and ops 8-10, then 11-14 with the output residual.
    assert row["kda"] == (1 + 2 + 4 + 8, 16 + 32 + 64)
    assert row["dsa"] == (128 + 256 + 512 + 1024, 2048 + 4096 + 8192 + 16384)
    total = 2 ** len(OPS) - 1
    assert row["span_us"] == total + 1
    assert row["covered"] == total / (total + 1)


def test_rejects_a_forward_missing_a_layer(tmp_path):
    ops = [(tag and tag.replace("layers.1.", "layers.0."), k) for tag, k in OPS]
    with pytest.raises(SystemExit, match="expected 0..1"):
        parse_prefill_trace.breakdown(*write_trace(tmp_path, ops))
