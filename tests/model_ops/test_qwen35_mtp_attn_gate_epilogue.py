# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""ATOM_MTP_ATTN_GATE_IN_REDUCE on one gfx1250 GPU: the Qwen3.5 attention
output gate (and the step-0 row selection) in the epilogue of aiter's
unified-attention reduce_segments, against the attention followed by ATOM's
output_gate_mul / output_gate_rows, at the drafter's A8K4V8 decode shapes (one
rank of TP4: 4 q heads, 1 kv head of 256, 128-token pages, the token split).

Run as a script with --bench for graph-replay timings (us a pass, the attention
included: the two sequences differ only after the attention kernel).
"""

import argparse
import functools
import os
import sys

import pytest
import torch

# The recipe's decode attention split, read when aiter's module is imported.
os.environ.setdefault("AITER_GFX12_TOKEN_SPLIT", "1")
os.environ.setdefault("AITER_GFX12_TOKEN_SPLIT_FUSED", "1")
os.environ.setdefault("AITER_GFX12_TOKEN_SPLIT_MIN_TILES", "16")
os.environ.setdefault("AITER_GFX12_TOKEN_SPLIT_MAX_TILES", "128")

HEADS, HEAD, PAGE = 4, 256, 128
CONTEXT = 12000  # tokens of KV a sequence


def _skip_reason():
    try:
        from aiter.jit.utils.chip_info import get_gfx
    except ImportError:
        return "needs aiter"
    if not torch.cuda.is_available() or get_gfx() != "gfx1250":
        return "needs gfx1250"
    try:
        import op_tests.op_benchmarks.triton.bench_unified_attention  # noqa: F401
    except ImportError:
        return "needs aiter's op_tests on the path"
    return None


class _Case:
    """One pass of the drafter: `seqs` sequences of `q_len` rows each."""

    def __init__(self, seqs, q_len, context=CONTEXT):
        from op_tests.op_benchmarks.triton import bench_unified_attention as bench
        from op_tests.triton_tests.attention.test_unified_attention import (
            shuffle_kv_cache,
        )

        lens = [(q_len, context - 37 * i) for i in range(seqs)]
        blocks = seqs * ((context + PAGE - 1) // PAGE) + 16
        inp = bench.make_inputs(
            seq_lens=lens,
            num_heads=(HEADS, 1),
            head_size_qk=HEAD,
            head_size_v=HEAD,
            block_size=PAGE,
            num_blocks=blocks,
            fp8_q=True,
            fp8_kv=True,
            fp8_output=False,
            out_scale_value=1.0,
            nvfp4_kv=True,
            nvfp4_ref=False,
            nvfp4_v=False,
        )
        self.inp = inp
        self.v = shuffle_kv_cache(inp["k_fp8"], inp["v_fp8"])[1]
        self.tokens = inp["q_fp8"].shape[0]
        self.q_len = q_len
        gen = torch.Generator(device="cuda").manual_seed(seqs * q_len)
        width = HEADS * HEAD
        self.qkvg = 3 * torch.randn(
            self.tokens, 2 * width + 512, generator=gen, device="cuda"
        ).to(torch.bfloat16)
        self.gate = self.qkvg[:, :width]
        # The last row of each sequence (step 0's head rows).
        self.rows = (
            (inp["cu_query_lens"][1:] - 1).to(torch.int64) if q_len > 1 else None
        )

    def attention(self, **epilogue):
        from aiter.ops.triton.unified_attention import unified_attention

        inp = self.inp
        rows = epilogue.get("output_rows")
        n = self.tokens if rows is None else rows.shape[0]
        out = torch.empty(n, HEADS, HEAD, dtype=torch.bfloat16, device="cuda")
        unified_attention(
            q=inp["q_fp8"],
            k=inp["k_nvfp4"],
            v=self.v,
            out=out,
            cu_seqlens_q=inp["cu_query_lens"],
            seqused_k=inp["kv_lens"],
            max_seqlen_q=inp["max_query_len"],
            max_seqlen_k=inp["max_kv_len"],
            softmax_scale=inp["scale"],
            causal=True,
            window_size=(-1, -1),
            block_table=inp["block_tables"],
            softcap=0,
            q_descale=inp["q_descale"],
            k_descale=inp["k_descale"],
            v_descale=inp["v_descale"],
            shuffled_kv_cache=True,
            equal_seqlen_q=self.q_len > 1,
            backend="gluon",
            **epilogue,
        )
        return out.view(n, HEADS * HEAD)

    def unfused(self):
        """What the drafter runs today: the attention, then the gate (and, at
        step 0 with ATOM_MTP_SELECT_ROWS_BEFORE_O_PROJ, the row gather in it)."""
        from atom.model_ops.sigmoid_gate import output_gate_mul, output_gate_rows

        out = self.attention()
        if self.rows is None:
            return output_gate_mul(out, self.gate, True)
        return output_gate_rows(out, self.gate, self.rows)

    def unfused_full(self):
        """Step 0 without the row selection: the gate over every row."""
        from atom.model_ops.sigmoid_gate import output_gate_mul

        return output_gate_mul(self.attention(), self.gate, True)

    def fused(self):
        return self.attention(output_gate=self.gate, output_rows=self.rows)


CASES = [(16, 1), (64, 1), (13, 5), (16, 5), (64, 5)]


@pytest.mark.skipif(_skip_reason() is not None, reason=str(_skip_reason()))
@pytest.mark.parametrize("seqs, q_len", CASES)
def test_gate_in_reduce_is_bitwise_the_separate_kernel(seqs, q_len):
    case = _Case(seqs, q_len, context=3000)
    ref = case.unfused()
    out = case.fused()
    assert torch.equal(out, ref)
    if case.rows is not None:
        # And the gathered rows are the full gate's rows.
        assert torch.equal(case.unfused_full()[case.rows], ref)


def _time(fn, iters=50, replays=40):
    for _ in range(3):
        fn()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(iters):
            fn()
    graph.replay()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(replays):
        graph.replay()
    stop.record()
    torch.cuda.synchronize()
    return start.elapsed_time(stop) * 1e3 / (iters * replays)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bench", action="store_true")
    args = parser.parse_args()
    if _skip_reason() is not None:
        sys.exit(f"skipped: {_skip_reason()}")
    print("| seqs x rows | attention + sequence | us |")
    print("|---|---|---:|")
    for seqs, q_len in CASES:
        case = _Case(seqs, q_len)
        assert torch.equal(case.fused(), case.unfused())
        if not args.bench:
            continue
        label = f"{seqs} x {q_len}"
        if q_len > 1:
            print(
                f"| {label} | attn + gate (all rows) | {_time(case.unfused_full):.2f} |"
            )
            print(f"| {label} | attn + gate_rows | {_time(case.unfused):.2f} |")
            print(f"| {label} | attn(gate+rows in reduce) | {_time(case.fused):.2f} |")
        else:
            print(f"| {label} | attn + gate | {_time(case.unfused):.2f} |")
            print(f"| {label} | attn(gate in reduce) | {_time(case.fused):.2f} |")
        print(f"| {label} | attn alone | {_time(case.attention):.2f} |")
        from atom.model_ops.sigmoid_gate import output_gate_mul, output_gate_rows

        out = case.attention()
        if q_len > 1:
            fn = functools.partial(output_gate_rows, out, case.gate, case.rows)
            print(f"| {label} | gate_rows alone | {_time(fn):.2f} |")
        else:
            fn = functools.partial(output_gate_mul, out, case.gate, True)
            print(f"| {label} | gate alone | {_time(fn):.2f} |")
    print("bitwise: ok")
