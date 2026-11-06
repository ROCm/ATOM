# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""ATOM_UA_DECODE_TOKEN_SPLIT: a decode forward's gfx1250 token split (aiter's
tiles per segment, AITER_GFX12_TOKEN_SPLIT=1) computed once on the host by the
attention builder and handed to every layer's unified attention, instead of a
gfx12_token_split_tiles launch before each.

The host value is aiter's int32 on any lengths; only the attentions over the
lengths it was computed from take it; prepare_decode publishes it through the
real H2D owners (RUN_H2D_GPU_TESTS=1); and on gfx1250 the Qwen3.5 verify and
draft attentions (16 q / 2 kv heads of 256, A8K4V8, 128-token pages) given it
are bitwise the attentions that compute their own, without computing one.

Run as a script with --bench for graph-replay timings (us an attention call).
"""

import argparse
import os
import sys
from types import SimpleNamespace as NS

import numpy as np
import pytest
import torch

# The recipe's decode attention split, read when aiter's module is imported
# (the benchmark also takes its segment bounds; under pytest they are left to
# the environment, as other suites read aiter's defaults).
os.environ.setdefault("AITER_GFX12_TOKEN_SPLIT", "1")
os.environ.setdefault("AITER_GFX12_TOKEN_SPLIT_FUSED", "1")
if __name__ == "__main__":
    os.environ.setdefault("AITER_GFX12_TOKEN_SPLIT_MIN_TILES", "16")
    os.environ.setdefault("AITER_GFX12_TOKEN_SPLIT_MAX_TILES", "128")

ua = pytest.importorskip(
    "aiter.ops.triton.attention.unified_attention",
    reason="needs aiter's unified attention",
    exc_type=ImportError,
)

HEADS, KV_HEADS, HEAD, PAGE = 16, 2, 256, 128
MAX_KV = 262144  # a decode graph's max_seqlen_k: the served max_model_len
# MLPerf agentic context lengths: the 9-quantiles of the prefills in the c=64
# reference trace (rank 0, 178 sequences, median 97k tokens).
CONTEXTS = [10065, 21528, 34298, 69120, 97191, 116045, 122782, 129160, 181586]


def _host_tiles():
    from atom.model_ops.attentions.aiter_attention import decode_token_split_tiles

    return decode_token_split_tiles


@pytest.mark.parametrize("programs, lo, hi", [(2048, 16, 128), (2048, 32, 256)])
def test_the_host_value_is_aiters(monkeypatch, programs, lo, hi):
    """`decode_token_split_tiles` against aiter's torch expression on the same
    lengths (pad rows of 0 included), from the lengths and from the page sum
    prepare_decode already has."""
    monkeypatch.setattr(ua, "_GFX12_TOKEN_SPLIT_PROGRAMS", programs)
    monkeypatch.setattr(ua, "_GFX12_TOKEN_SPLIT_MIN_TILES", lo)
    monkeypatch.setattr(ua, "_GFX12_TOKEN_SPLIT_MAX_TILES", hi)
    host = _host_tiles()
    g = np.random.default_rng(programs + lo)
    for rows, top, tile, heads in [
        (1, 3, 128, 2),
        (4, 300, 128, 2),
        (9, 190000, 128, 2),
        (12, 262144, 128, 2),
        (20, 262144, 64, 1),
        (36, 5000, 16, 4),
        (128, 300000, 128, 8),
    ]:
        for _ in range(20):
            lens = g.integers(0, top, rows, dtype=np.int32)
            lens[rows // 2 :] *= g.integers(0, 2)  # sometimes a zero pad tail
            ref = ua._gfx12_token_split_tiles_torch(torch.from_numpy(lens), heads, tile)
            constants = (tile, heads, programs, lo, hi)
            pages = int(np.cumsum((lens + tile - 1) // tile)[-1])
            assert host(lens, constants) == int(ref)
            assert host(lens, constants, pages) == int(ref)


def _layer():
    return NS(
        _reject_per_token_scales_on_unified=lambda *a: None,
        kv_cache_dtype="fp8_k4v8",
        _take_output_gate=lambda rows: None,
        sliding_window=0,
        scale=1.0,
        k4v8=True,
        kv_scale=None,
        sinks=None,
        use_flash_layout=False,
    )


@pytest.mark.parametrize(
    "is_draft, q, advance, same_lens, takes",
    [
        (False, 4, False, True, "decode"),  # a target layer of a verify
        (False, 1, False, True, "decode"),  # plain decode
        (True, 4, False, True, "decode"),  # draft step 0: the verify's lengths
        (True, 1, True, True, "advance"),  # a later pass: the advance's
        (True, 1, False, True, None),  # ... or aiter's own, never the verify's
        (False, 4, False, False, None),  # lengths not the ones it was made for
    ],
)
def test_which_attentions_take_it(monkeypatch, is_draft, q, advance, same_lens, takes):
    from atom.model_ops import attention_mha as mha
    from atom.model_ops.attentions.aiter_attention import DecodeTokenSplit

    seen = {}
    monkeypatch.setattr(mha, "unified_attention", lambda *a, **kw: seen.update(kw))
    monkeypatch.setattr(mha, "current_attention_output_epilogue", lambda: None)
    monkeypatch.setattr(mha.envs, "ATOM_UA_SPLIT_REDUCE", False)
    lens = torch.tensor([5000, 900], dtype=torch.int32)
    tiles = torch.tensor([16], dtype=torch.int32)
    md = NS(
        context_lens=lens if same_lens else lens.clone(),
        cu_seqlens_q=None,
        max_seqlen_q=q,
        max_seqlen_k=MAX_KV,
        block_tables=None,
        equal_seqlen_q=q > 1,
        decode_token_split_tiles=DecodeTokenSplit(tiles, PAGE, KV_HEADS, lens),
    )
    if advance:
        md.gfx12_token_split_tiles = ("advance's", PAGE, KV_HEADS)
    fwd_ctx = NS(attn_metadata=md, context=NS(is_draft=is_draft))
    query = torch.zeros(2 * q, HEADS, HEAD, dtype=torch.bfloat16)
    mha.PagedAttentionImpl.paged_attention_unified(
        _layer(), query, None, None, None, None, None, None, fwd_ctx
    )
    want = {
        "decode": (tiles, PAGE, KV_HEADS),
        "advance": md.__dict__.get("gfx12_token_split_tiles"),
        None: None,
    }[takes]
    assert seen.get("gfx12_token_split_tiles") == want


@pytest.mark.skipif(
    os.environ.get("RUN_H2D_GPU_TESTS") != "1", reason="set RUN_H2D_GPU_TESTS=1"
)
@pytest.mark.parametrize("transport", ["direct", "packed"])
@pytest.mark.parametrize("pp", [1, 2])
@pytest.mark.parametrize("q", [1, 4])
@pytest.mark.parametrize("tile", [16, 32])
def test_prepare_decode_publishes_the_split(monkeypatch, transport, pp, q, tile):
    """Through the real H2D owners: each decode's [1] int32 on the GPU is what
    aiter computes from that decode's seqused_k, the metadata's own tensor; at
    a tile of one page (16) from the page sum, at another from the lengths."""
    import importlib.util

    if ua.gfx12_token_split_constants() is None:
        pytest.skip("needs aiter's gfx1250 token split")
    here = os.path.dirname(__file__)
    path = os.path.join(here, "..", "test_h2d_attention_publication.py")
    spec = importlib.util.spec_from_file_location("_h2d_attention_publication", path)
    h2d = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(h2d)

    monkeypatch.setenv("ATOM_UA_DECODE_TOKEN_SPLIT", "1")
    runner, builder = h2d.make_runner(
        monkeypatch, "mha", q=q, pp=pp, tbo=False, transport=transport
    )
    assert builder._decode_token_split[:2] == (16, 1)  # the page, 1 KV head
    # Constants the test's short lengths do not all clamp to one value.
    builder._decode_token_split = (tile, 1, 2, 1, 64)
    monkeypatch.setattr(ua, "_GFX12_TOKEN_SPLIT_PROGRAMS", 2)
    monkeypatch.setattr(ua, "_GFX12_TOKEN_SPLIT_MIN_TILES", 1)
    monkeypatch.setattr(ua, "_GFX12_TOKEN_SPLIT_MAX_TILES", 64)
    values = set()
    for step, count in enumerate((3, 1, 4, 2)):
        batch = h2d.decode_batch(count, q, 30 * step)
        h2d.start_decode(runner, builder, batch, q)
        md, _ = builder.prepare_decode(batch, 4, 4 * q, q)
        split = md.decode_token_split_tiles
        assert split.seqused_k is md.context_lens
        assert split.metadata()[1:] == (tile, 1)
        assert split.tiles.data_ptr() == (
            runner.forward_vars["decode_token_split_tiles"].gpu.data_ptr()
        )
        h2d.finish(runner)
        torch.cuda.synchronize()
        ref = ua._gfx12_token_split_tiles_torch(md.context_lens, 1, tile)
        assert torch.equal(split.tiles, ref), (step, split.tiles, ref)
        values.add(int(ref))
    assert len(values) > 2
    runner.h2d_owner.begin()
    capture, _ = builder.build_for_cudagraph_capture(4)
    h2d.finish(runner)
    assert capture.decode_token_split_tiles.seqused_k is capture.context_lens


def _gfx1250_reason():
    if not torch.cuda.is_available() or ua.DEVICE_ARCH != "gfx1250":
        return "needs gfx1250"
    if ua.gfx12_token_split_constants() is None:
        return "needs AITER_GFX12_TOKEN_SPLIT=1"
    try:
        import op_tests.op_benchmarks.triton.bench_unified_attention  # noqa: F401
    except ImportError:
        return "needs aiter's op_tests on the path"
    return None


class _Case:
    """One unified-attention call of a Qwen3.5 decode step: `seqs` sequences of
    `q_len` rows (4: the verify and draft step 0; 1: a later draft pass)."""

    def __init__(self, seqs, q_len):
        from op_tests.op_benchmarks.triton import bench_unified_attention as bench
        from op_tests.triton_tests.attention.test_unified_attention import (
            shuffle_kv_cache,
        )

        self.lens = np.array(
            [CONTEXTS[(4 + 2 * i) % len(CONTEXTS)] for i in range(seqs)],
            dtype=np.int32,
        )
        pages = -(-int(self.lens.max()) // PAGE)
        inp = bench.make_inputs(
            seq_lens=[(q_len, int(n)) for n in self.lens],
            num_heads=(HEADS, KV_HEADS),
            head_size_qk=HEAD,
            head_size_v=HEAD,
            block_size=PAGE,
            num_blocks=seqs * pages + 16,
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
        self.q_len = q_len
        self.tokens = inp["q_fp8"].shape[0]
        # What prepare_decode publishes: the builder's host value, its tile.
        tile = 1 << (PAGE - 1).bit_length()
        constants = (tile, KV_HEADS, *ua.gfx12_token_split_constants())
        value = _host_tiles()(self.lens, constants)
        self.split = (
            torch.tensor([value], dtype=torch.int32, device="cuda"),
            tile,
            KV_HEADS,
        )

    def attention(self, out, precomputed=False):
        inp = self.inp
        split = {"gfx12_token_split_tiles": self.split} if precomputed else {}
        ua.unified_attention(
            q=inp["q_fp8"],
            k=inp["k_nvfp4"],
            v=self.v,
            out=out,
            cu_seqlens_q=inp["cu_query_lens"],
            seqused_k=inp["kv_lens"],
            max_seqlen_q=inp["max_query_len"],
            max_seqlen_k=MAX_KV,
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
            split_reduce=True,
            **split,
        )
        return out

    def new_out(self):
        return torch.empty(
            self.tokens, HEADS, HEAD, dtype=torch.bfloat16, device="cuda"
        )


CASES = [(1, 4), (2, 4), (3, 4), (4, 4), (5, 4), (8, 4), (9, 4), (2, 1), (4, 1)]


@pytest.mark.skipif(_gfx1250_reason() is not None, reason=str(_gfx1250_reason()))
@pytest.mark.parametrize("seqs, q_len", CASES)
@torch.inference_mode()
def test_the_attention_is_bitwise_and_computes_no_split(monkeypatch, seqs, q_len):
    case = _Case(seqs, q_len)
    calls = []
    fused = ua._gfx12_token_split_tiles_fused
    monkeypatch.setattr(
        ua, "_gfx12_token_split_tiles_fused", lambda *a: calls.append(1) or fused(*a)
    )
    ref = case.attention(case.new_out())
    assert calls, "the reference took no token split: not the 3d kernel"
    assert torch.equal(
        case.split[0],
        ua._gfx12_token_split_tiles_torch(case.inp["kv_lens"], KV_HEADS, case.split[1]),
    )

    def computed(*args):
        raise AssertionError("the attention computed its own token split")

    monkeypatch.setattr(ua, "_gfx12_token_split_tiles_torch", computed)
    monkeypatch.setattr(ua, "_gfx12_token_split_tiles_fused", computed)
    out = case.attention(case.new_out(), precomputed=True)
    torch.cuda.synchronize()
    assert torch.equal(out, ref)


def _bench(layers, replays, rounds, prologue):
    """us an attention call in a graph of `layers` calls (a step's
    full-attention layers), the split computed per call (aiter) vs given.
    ``prologue``: a small kernel before each call, as the qk-norm/RoPE is."""
    torch.backends.cuda.matmul.allow_tf32 = False
    print(
        f"{'seqs':>4} {'q':>2} {'ctx (k tokens)':<34} {'aiter':>7} {'given':>7} "
        f"{'saved':>6}  bitwise"
    )
    for seqs, q_len in CASES:
        case = _Case(seqs, q_len)
        graphs, outs = {}, {}
        scratch = torch.zeros(case.tokens, HEADS * HEAD, device="cuda")
        for pre in (False, True):
            outs[pre] = [case.new_out() for _ in range(layers)]
            for o in outs[pre]:
                case.attention(o, pre)  # compile and warm outside the capture
            torch.cuda.synchronize()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                for o in outs[pre]:
                    if prologue:
                        scratch.add_(1.0)
                    case.attention(o, pre)
            graphs[pre] = g
        times = {False: [], True: []}
        start, end = torch.cuda.Event(True), torch.cuda.Event(True)
        for _ in range(rounds):
            for pre in (False, True):
                for _ in range(10):
                    graphs[pre].replay()
                start.record()
                for _ in range(replays):
                    graphs[pre].replay()
                end.record()
                end.synchronize()
                times[pre].append(start.elapsed_time(end) * 1e3 / replays / layers)
        same = all(torch.equal(a, b) for a, b in zip(outs[False], outs[True]))
        a, b = (float(np.median(times[p])) for p in (False, True))
        ctx = ",".join(str(round(n / 1000)) for n in case.lens)
        print(f"{seqs:>4} {q_len:>2} {ctx:<34} {a:7.2f} {b:7.2f} {a - b:6.2f}  {same}")
        del case, graphs, outs, scratch
        torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bench", action="store_true")
    parser.add_argument("--layers", type=int, default=10)
    parser.add_argument("--replays", type=int, default=200)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--prologue", action="store_true")
    args, rest = parser.parse_known_args()
    if args.bench:
        reason = _gfx1250_reason()
        if reason:
            sys.exit(reason)
        with torch.inference_mode():
            _bench(args.layers, args.replays, args.rounds, args.prologue)
    else:
        sys.exit(pytest.main([__file__, *rest]))
