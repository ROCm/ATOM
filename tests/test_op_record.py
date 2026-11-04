"""`atom.utils.op_record` and `tools/op_record/export.py` without a GPU or
AITER: a call recorded into a graph is counted once per replay of that graph
in the step that replayed it, a call outside any recording once in its own
step, and calls between forwards stay out of every window."""

import importlib.util
import json
import sys
import types
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from atom.utils import op_record

SCRIPT = Path(__file__).resolve().parents[1] / "tools/op_record/export.py"
SPEC = importlib.util.spec_from_file_location("op_record_export", SCRIPT)
export = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(export)


class FakeGraph:
    def capture_begin(self, *args, **kwargs):
        pass

    def capture_end(self):
        pass

    def replay(self):
        pass


def _batch(q, ctx, prefill=0):
    return SimpleNamespace(
        is_dummy_run=False,
        total_tokens_num=sum(q),
        total_seqs_num_prefill=prefill,
        total_seqs_num_decode=len(q) - prefill,
        num_spec_step=0,
        num_scheduled_tokens=q,
        context_lens=ctx,
    )


def _x(rows, cols=128, dtype=torch.bfloat16):
    return torch.empty(rows, cols, dtype=dtype)


@pytest.fixture
def patched_graph(monkeypatch):
    monkeypatch.setattr(torch.cuda, "CUDAGraph", FakeGraph)
    for name in ("capture_begin", "capture_end", "replay"):
        monkeypatch.setattr(FakeGraph, name, getattr(FakeGraph, name))
    return FakeGraph


def _counts(rank_dir, window=(0, float("inf"))):
    rec = export.load(str(rank_dir))
    steps = export.window_steps(rec, *window)
    calls, replays = export.count_calls(rec, steps)
    by_op = Counter()
    for sid, n in calls.items():
        by_op[rec.sigs[sid]["op"], tuple(rec.sigs[sid]["args"][0]["shape"])] += n
    return rec, steps, by_op, replays


def test_replays_multiply_the_recording_into_the_replaying_step(
    tmp_path, patched_graph
):
    rec = op_record.OpRecorder(str(tmp_path), {"test": True})
    op_record._patch_cuda_graph(rec)

    rec.on_call("warmup_op", 0, (_x(7),), {})  # between forwards
    graph = FakeGraph()
    graph.capture_begin()
    rec.on_call("gemm", 0, (_x(8),), {})
    rec.on_call("gemm", 0, (_x(8),), {})
    rec.on_call("gemm_inner", 1, (_x(8),), {})
    graph.capture_end()

    def prefill(batch):
        rec.on_call("gemm", 0, (_x(300),), {})

    def decode(batch):
        for _ in range(3):
            graph.replay()
        rec.on_call("eager_region_attn", 0, (_x(8),), {})

    rec.step(prefill, _batch([300], [300], prefill=1))
    rec.step(decode, _batch([1] * 8, [301] * 8))
    rec.step(decode, _batch([1] * 8, [302] * 8))
    rec.close()

    _, steps, by_op, replays = _counts(tmp_path)
    assert len(steps) == 3
    assert by_op == {
        ("gemm", (300, 128)): 1,
        ("gemm", (8, 128)): 2 * 3 * 2,
        ("gemm_inner", (8, 128)): 1 * 3 * 2,
        ("eager_region_attn", (8, 128)): 2,
    }
    assert sum(replays.values()) == 6
    assert steps[1]["q"] == [1] * 8 and steps[1]["ctx"] == [301] * 8

    lines = [json.loads(s) for s in (tmp_path / "steps.jsonl").read_text().splitlines()]
    outside = [s for s in lines if s["kind"] == "outside"]
    assert len(outside) == 1 and sum(outside[0]["calls"].values()) == 1


def test_window_keeps_only_steps_wholly_inside(tmp_path, patched_graph):
    rec = op_record.OpRecorder(str(tmp_path), {})
    for rows in (1, 2, 3):
        rec.step(
            lambda b, r=rows: rec.on_call("op", 0, (_x(r),), {}),
            _batch([rows], [rows]),
        )
    rec.close()
    loaded = export.load(str(tmp_path))
    middle = loaded.steps[1]
    _, steps, by_op, _ = _counts(tmp_path, (middle["t0"], middle["t1"]))
    assert [s["q"] for s in steps] == [[2]]
    assert by_op == {("op", (2, 128)): 1}


def test_install_wraps_forward_and_registers_the_hook(
    tmp_path, monkeypatch, patched_graph
):
    hooks = []
    modules = {
        "aiter": types.ModuleType("aiter"),
        "aiter.jit": types.ModuleType("aiter.jit"),
        "aiter.jit.utils": types.ModuleType("aiter.jit.utils"),
        "aiter.jit.utils.call_hook": types.ModuleType("aiter.jit.utils.call_hook"),
    }
    modules["aiter"].__file__ = str(tmp_path / "no_aiter" / "__init__.py")
    modules["aiter.jit.utils.call_hook"].register_call_hook = hooks.append
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)

    runner = SimpleNamespace(
        rank=0,
        config=SimpleNamespace(
            model="m", parallel_config=SimpleNamespace(data_parallel_size=1)
        ),
    )
    runner.forward = lambda batch: hooks[0]("op", 0, (_x(4),), {}) or "out"
    rec = op_record.install(runner, str(tmp_path / "rank_0"))
    assert runner.forward(_batch([4], [4])) == "out"
    rec.close()

    meta = json.loads((tmp_path / "rank_0" / "meta.json").read_text())
    assert meta["aiter_commit"].startswith("unknown")
    _, steps, by_op, _ = _counts(tmp_path / "rank_0")
    assert len(steps) == 1 and by_op == {("op", (4, 128)): 1}


def _sig(op, args, kwargs=None, depth=0):
    return {
        "op": op,
        "depth": depth,
        "args": [op_record._arg_json(op_record._arg_signature(a)) for a in args],
        "kwargs": {
            k: op_record._arg_json(op_record._arg_signature(v))
            for k, v in (kwargs or {}).items()
        },
    }


def test_family_entry_is_the_outermost_family_op(tmp_path, patched_graph):
    """An ATOM wrapper above the aiter entry is looked through; an aiter
    family op nested in another is not counted again."""
    fp8 = torch.float8_e4m3fn
    x, w = torch.empty(4, 512, dtype=fp8), torch.empty(256, 512, dtype=fp8)
    xs, ws = torch.empty(4, 4), torch.empty(2, 4)
    rec = op_record.OpRecorder(str(tmp_path), {})

    def forward(batch):
        rec.on_call("atom_wrapper_impl", 0, (x, w, xs, ws), {})
        rec.on_call("gemm_a8w8_blockscale_bpreshuffle", 1, (x, w, xs, ws), {})
        rec.on_call("gemm_a8w8_blockscale", 0, (x, w, xs, ws), {"isBpreshuffled": True})
        rec.on_call("gemm_a8w8_blockscale_bpreshuffle", 1, (x, w, xs, ws), {})

    rec.step(forward, _batch([4], [4]))
    rec.close()
    loaded = export.load(str(tmp_path))
    calls, _ = export.count_calls(loaded, export.window_steps(loaded, 0, float("inf")))
    rows = export.family_rows(loaded, calls)
    assert dict(rows) == {
        "a8w8_blockscale_bpreshuffle": Counter({(("M", 4), ("N", 256), ("K", 512)): 2})
    }
    parents = {s["op"]: s["parents"] for s in loaded.sigs.values() if s["depth"] == 1}
    assert parents["gemm_a8w8_blockscale_bpreshuffle"] in (
        ["atom_wrapper_impl"],
        ["gemm_a8w8_blockscale"],
    )


def test_blockscale_family_follows_preshuffle_flag():
    fp8 = torch.float8_e4m3fn
    x, w = torch.empty(2, 3, 512, dtype=fp8), torch.empty(1024, 512, dtype=fp8)
    xs, ws = torch.empty(6, 4), torch.empty(8, 4)
    sig = _sig("gemm_a8w8_blockscale", (x, w, xs, ws), {"isBpreshuffled": True})
    assert export.OPS["gemm_a8w8_blockscale"](sig) == (
        "a8w8_blockscale_bpreshuffle",
        {"M": 6, "N": 1024, "K": 512},
    )
    sig = _sig("gemm_a8w8_blockscale", (x, w, xs, ws, torch.bfloat16, False))
    assert export.OPS["gemm_a8w8_blockscale"](sig)[0] == "a8w8_blockscale"


def test_fused_moe_keys():
    fp8 = torch.float8_e4m3fn
    x = torch.empty(16, 4096, dtype=torch.bfloat16)
    w1 = torch.empty(128, 2 * 1024, 4096, dtype=fp8)
    w2 = torch.empty(128, 4096, 1024, dtype=fp8)
    weights, ids = torch.empty(16, 8), torch.empty(16, 8, dtype=torch.int32)
    sig = _sig("fused_moe_", (x, w1, w2, weights, ids, None, 0, 4, False))
    assert export.OPS["fused_moe_"](sig) == (
        "fmoe",
        {
            "token": 16,
            "model_dim": 4096,
            "inter_dim": 1024,
            "expert": 128,
            "topk": 8,
            "act_type": "ActivationType.Silu",
            "dtype": "torch.bfloat16",
            "q_dtype_a": "torch.float8_e4m3fn",
            "q_dtype_w": "torch.float8_e4m3fn",
            "q_type": "QuantType.per_1x128",
            "use_g1u1": 1,
            "doweight_stage1": 0,
        },
    )


def test_a16w16_keys():
    a, b = torch.empty(5, 256, dtype=torch.bfloat16), torch.empty(64, 256)
    sig = _sig("gemm_a16w16", (a, b, None, torch.float32))
    family, keys = export.OPS["gemm_a16w16"](sig)
    assert family == "bf16"
    assert keys == {
        "M": 5,
        "N": 64,
        "K": 256,
        "bias": False,
        "dtype": "torch.bfloat16",
        "outdtype": "torch.float32",
        "scaleAB": False,
        "bpreshuffle": False,
    }


def test_bucketing_keeps_every_call():
    rows = Counter(
        {(("M", m), ("N", 64), ("K", 32)): m for m in range(1, 21)}
        | {(("M", 5), ("N", 128), ("K", 32)): 7}
    )
    out, bucketed = export.bucket(rows, "M", 4)
    assert bucketed
    assert sum(out.values()) == sum(rows.values())
    assert sum(1 for k in out if dict(k)["N"] == 64) <= 4
    assert out[(("M", 5), ("N", 128), ("K", 32))] == 7


def test_aiperf_window_skips_warmup(tmp_path):
    lines = [
        {
            "metadata": {
                "benchmark_phase": "warmup",
                "request_start_ns": 1e9,
                "request_end_ns": 2e9,
            }
        },
        {
            "metadata": {
                "benchmark_phase": "profiling",
                "request_start_ns": 5e9,
                "request_end_ns": 7e9,
            }
        },
        {
            "metadata": {
                "benchmark_phase": "profiling",
                "request_start_ns": 6e9,
                "request_end_ns": 9e9,
            }
        },
    ]
    path = tmp_path / "profile_export.jsonl"
    path.write_text("\n".join(json.dumps(line) for line in lines))
    assert export.aiperf_window(str(path)) == (5.0, 9.0)


def _pa_decode_args(rows=16, kv_heads=4, blocks=1000, width=64):
    fp8 = torch.float8_e4m3fn
    return (
        torch.empty(rows, 32, 128, dtype=torch.bfloat16),  # output
        torch.empty(rows, 32, 128, dtype=fp8),  # query
        torch.empty(blocks, kv_heads, 16, 16, 8, dtype=fp8),  # key_cache
        torch.empty(blocks, kv_heads, 2, 128, 8, dtype=fp8),  # value_cache
        torch.empty(rows, dtype=torch.int32),  # context_lengths
        torch.empty(rows, width, dtype=torch.int32),  # block_tables
        0.088,
        1,  # query_length
        4,
        256,
        torch.float8_e4m3fn,
        None,
        torch.empty(blocks, kv_heads, 16, 1),
        torch.empty(blocks, kv_heads, 16, 1),
        None,
        None,
        None,
        None,
        None,
        0,
        True,
    )


def test_attention_calls_take_their_steps_context_lengths(tmp_path, patched_graph):
    """A decode graph records attention once with dummy lengths; each replay
    is charged its step's lengths, padded to the graph's rows, once per layer."""
    rec = op_record.OpRecorder(str(tmp_path), {})
    op_record._patch_cuda_graph(rec)
    graph = FakeGraph()
    graph.capture_begin()
    for _ in range(10):  # ten attention layers
        rec.on_call("pa_decode_gluon", 0, _pa_decode_args(), {})
    rec.on_call("fused_gdn_gating", 0, (torch.empty(8), torch.empty(16, 8)), {})
    graph.capture_end()

    def decode(batch):
        graph.replay()

    rec.step(decode, _batch([1] * 13, [100 + i for i in range(13)]))
    rec.step(decode, _batch([1] * 13, [101 + i for i in range(13)]))
    rec.step(decode, _batch([1] * 16, [5000] * 16))
    rec.close()

    loaded = export.load(str(tmp_path))
    steps = export.window_steps(loaded, 0, float("inf"))
    by_op = {
        loaded.sigs[sid]["op"]: v
        for sid, v in export.replay_calls(loaded, steps).items()
    }
    attention = by_op["pa_decode_gluon"]
    assert [n for _, n in attention] == [10, 10, 10]
    first = json.loads(attention[0][0])
    assert first["ctx"] == [100 + i for i in range(13)] + [0, 0, 0]
    assert first["q"] == [1] * 13 + [0, 0, 0]
    assert json.loads(attention[2][0])["ctx"] == [5000] * 16
    assert by_op["fused_gdn_gating"] == [("{}", 1)] * 3

    reps, bucketed = export.replay_bins(attention, 2)
    assert bucketed and sum(n for _, n in reps) == 30
    assert json.loads(reps[-1][0])["ctx"] == [5000] * 16

    out = tmp_path / "export"
    summary = export.export(str(tmp_path), str(out), None, replay_bin_count=16)
    family = summary["families"]["attn_pa_decode_gluon"]
    assert (family["driver"], family["owner"], family["calls"]) == (
        "replay",
        "aiter",
        30,
    )
    assert family["replay"] == "aiter/utility/op_replay.py"
    gating = summary["families"]["gdn_gating"]
    assert (gating["owner"], gating["replay"], gating["calls"]) == (
        "atom",
        "tools/op_record/replay.py",
        3,
    )
    assert "pa_decode_gluon" not in summary["depth0_ops_without_family"]
    import csv

    with open(out / "attn_pa_decode_gluon.csv") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 3
    assert json.loads(rows[0]["work"])["ctx"][-3:] == [0, 0, 0]
    assert rows[0]["label"].startswith("pa_decode_gluon 16x32x128 seqs=13/16 ")


def test_a_call_serves_the_sequences_its_row_count_matches():
    """A mixed step's prefill call serves its prefill sequences, its decode
    call the decode ones; a graph's extra rows are zero."""
    q, ctx = [1, 300, 1, 512], [900, 300, 901, 600]
    assert export._served(q, ctx, 2, "prefill") == {"q": [300, 512], "ctx": [300, 600]}
    assert export._served(q, ctx, 2, "decode") == {"q": [1, 1], "ctx": [900, 901]}
    assert export._served(q, ctx, 4, "prefill") == {"q": q, "ctx": ctx}
    assert export._served([1, 1], [5, 6], 4) == {"q": [1, 1, 0, 0], "ctx": [5, 6, 0, 0]}


def test_collectives_export_with_the_recorded_tp_size(tmp_path, patched_graph):
    """A collective is priced by its shape alone and replayed on as many
    ranks as the recorded TP group."""
    rec = op_record.OpRecorder(
        str(tmp_path), {"config": {"tensor_parallel_size": 4}}
    )
    op_record._patch_cuda_graph(rec)
    graph = FakeGraph()
    graph.capture_begin()
    x, w = torch.empty(64, 4096, dtype=torch.bfloat16), torch.empty(4096)
    for _ in range(2):
        rec.on_call("fused_allreduce_rmsnorm_", 0, (x, x, w, 1e-6, "tp:0"), {})
    rec.on_call("fused_allreduce_rmsnorm", 1, (0, x, x, x, x), {})
    graph.capture_end()
    rec.step(lambda b: graph.replay(), _batch([1] * 64, [100] * 64))
    rec.step(lambda b: graph.replay(), _batch([1] * 64, [101] * 64))
    rec.close()

    rec_steps = export.window_steps(export.load(str(tmp_path)), 0, float("inf"))
    (calls,) = export.replay_calls(export.load(str(tmp_path)), rec_steps).values()
    assert calls == [('{"graph":true}', 2), ('{"graph":true}', 2)]

    summary = export.export(str(tmp_path), str(tmp_path / "out"), None)
    family = summary["families"]["comm_allreduce_rmsnorm"]
    assert (family["owner"], family["ranks"], family["calls"], family["rows"]) == (
        "aiter",
        4,
        4,
        1,
    )
