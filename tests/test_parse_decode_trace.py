"""CPU tests for `tools/parse_decode_trace.py` on a synthetic two-layer decode.

The run trace holds two ``decode[`` graph replays, whose kernels carry no module
tag, and the capture trace the eager warmup forward of their graph. Each
replayed kernel's duration is a distinct power of two, scaled by the forward's
number, so a half's time names exactly which kernels were attributed to it.
"""

import gzip
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import parse_decode_trace

CPU, MAIN, SIDE = 1, 7, 9
LABEL = "decode[bs=4 tok=32 d=4 spec=7 sk=644000]"

# (module tag or None, kernel, stream). Layer 0 is KDA, layer 1 DSA. The
# warmup's all-reduce is the FillFunctor aiter returns outside capture, and the
# first op is attention metadata the replay builds before the graph.
OPS = [
    (None, "FillFunctor", MAIN),
    (None, "embedding", MAIN),
    ("model.layers.0.self_attn.in_proj", "gemm", MAIN),
    ("model.layers.0.self_attn.kda", "kda_recurrence", MAIN),
    ("apply_attn_res", "attn_res", MAIN),
    ("model.layers.0.mlp.down_proj", "gemm", MAIN),
    (None, "FillFunctor", MAIN),
    ("apply_attn_res", "attn_res", MAIN),
    ("model.layers.1.self_attn.indexer", "topk", SIDE),
    ("model.layers.1.self_attn.fused_qkv_a_proj", "gemm", MAIN),
    ("dcp_sparse_decode", "mla_decode", MAIN),
    ("model.layers.1.self_attn.o_proj", "gemm", MAIN),
    ("apply_attn_res", "attn_res", MAIN),
    ("model.layers.1.block_sparse_moe.experts", "moe", MAIN),
    (None, "FillFunctor", MAIN),
]


def event(cat, name, ts, dur, tid, **args):
    return {"ph": "X", "cat": cat, "name": name, "ts": ts, "dur": dur, "tid": tid} | {
        "pid": 0,
        "args": args,
    }


def kernel(name, ts, dur, stream, correlation):
    return event(
        "kernel", name, ts, dur, stream, stream=stream, correlation=correlation
    )


def write_gz(path, events):
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt") as f:
        json.dump({"traceEvents": events}, f)


def write_traces(tmp_path, replayed=lambda name: name):
    warmup = []
    for i, (tag, name, stream) in enumerate(OPS):
        warmup.append(
            event("cuda_runtime", "hipLaunchKernel", i, 0.5, CPU, correlation=i)
        )
        warmup.append(kernel(name, 1000.0 + i, 1.0, stream, i))
        if tag:
            warmup.append(event("gpu_user_annotation", tag, 1000.0 + i, 1.0, stream))
    # The capture opens while the GPU still runs the warmup's tail.
    warmup.append(event("user_annotation", "capture_graph_bs_4", 1007.0, 50.0, CPU))
    write_gz(tmp_path / "capture_traces" / "bs_4_q_8_rank0.json.gz", warmup)

    run = []
    for forward in (1, 2):
        ts, graph = 100000.0 * forward, 100 + forward
        run.append(event("user_annotation", LABEL, ts - 50, 10.0, CPU))
        run.append(
            event(
                "cuda_runtime", "hipGraphLaunch", ts - 45, 1.0, CPU, correlation=graph
            )
        )
        for i, (_, name, stream) in enumerate(OPS[1:], start=1):
            if i == len(OPS) - 1:  # a graph memset node
                run.append(
                    kernel("__amd_rocclr_fillBufferAligned", ts, 1.0, MAIN, graph)
                )
                ts += 1.0
            run.append(kernel(replayed(name), ts, forward * 2.0**i, stream, graph))
            ts += 0.0 if stream == SIDE else forward * 2.0**i
        # The head's GEMM, launched eagerly after the graph.
        run.append(kernel("gemm", ts, 3.0, MAIN, 0))
    trace = tmp_path / "rank0.pt.trace.json.gz"
    write_gz(trace, run)
    model = tmp_path / "model"
    model.mkdir()
    config = {"linear_attn_config": {"kda_layers": [1]}, "num_hidden_layers": 2}
    (model / "config.json").write_text(json.dumps({"text_config": config}))
    return str(trace), str(model)


def all_reduce_in_graph(name):
    return "cross_device_reduce_1stage" if name == "FillFunctor" else name


def test_attributes_every_replayed_kernel(tmp_path):
    trace, model = write_traces(tmp_path, all_reduce_in_graph)
    result = parse_decode_trace.breakdown(trace, model, batch_size=4, skip=0, count=2)

    assert result["capture"].endswith("bs_4_q_8_rank0.json.gz")
    # Layer 0 (KDA): ops 1-3 attention, 4-6 MLP. Layer 1 (DSA): the residual
    # opening it and 9-11 (8 is off the main stream), then 12-14. Mean scale 1.5.
    assert [row["kda"] for row in result["rows"]] == [
        (f * (2 + 4 + 8), f * (16 + 32 + 64)) for f in (1, 2)
    ]
    assert [row["dsa"] for row in result["rows"]] == [
        (f * (128 + 512 + 1024 + 2048), f * (4096 + 8192 + 16384)) for f in (1, 2)
    ]
    for row in result["rows"]:
        assert 0.98 <= row["covered"] < 1.0
    layer, kernels = result["medians"]["dsa"]
    assert layer == 1
    assert [(mlp, kernel, us) for mlp, _, kernel, us in kernels][-1] == (
        True,
        "cross_device_reduce_1stage",
        1.5 * 16384,
    )


def test_rejects_a_kernel_the_warmup_did_not_run(tmp_path):
    trace, model = write_traces(tmp_path, lambda n: "fused_" + n if n == "moe" else n)
    with pytest.raises(SystemExit, match="1 kernels not in .*first fused_moe"):
        parse_decode_trace.breakdown(trace, model, batch_size=4, skip=0, count=2)
