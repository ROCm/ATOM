# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""The Qwen3.5-MoE mono kernel's weight layouts (against aiter's shuffles) and its
build-time mailbox layout (no GPU)."""

import pytest
import torch

from atom.models.qwen3_5.mono import layout, packing
from atom.models.qwen3_5.mono.config import MAX_TOKENS, Q35Config

TOKEN_COUNTS = range(1, MAX_TOKENS + 1)


def _bytes(shape, lo=0, hi=256):
    return torch.randint(lo, hi, shape, dtype=torch.uint8)


@pytest.mark.parametrize("shape", [(64, 256), (3, 32, 4096)])
def test_fp8_matrices_read_in_atoms_shuffle(shape):
    shuffle = pytest.importorskip("aiter.ops.shuffle")
    q = _bytes(shape).view(torch.float8_e4m3fn)
    want = shuffle.shuffle_weight(q, layout=(16, 16)).view(torch.uint8).reshape(-1)
    assert torch.equal(packing.shuffle_fp8(q), want)


# the per-rank banks at TP 4: [gate | up] (2 x 256 rows, K = hidden), down (K = 256)
@pytest.mark.parametrize("gate_up, n, k", [(True, 512, 4096), (False, 4096, 256)])
def test_mxfp4_bank_reads_in_atoms_gguu_shuffle(gate_up, n, k):
    shuffle = pytest.importorskip("aiter.ops.shuffle")
    e = 3
    w = _bytes((e, n, k // 2))
    s = _bytes((e, n, k // 32), 100, 140)
    want_w = shuffle.shuffle_weight(
        w.clone(), layout=(16, 16), is_guinterleave=False, gate_up=gate_up
    )
    want_s = shuffle.shuffle_scale(
        s.clone().view(e * n, k // 32),
        experts_cnt=e,
        is_guinterleave=False,
        gate_up=gate_up,
    )
    assert torch.equal(packing.pack_mxfp4(w), want_w.view(torch.uint8).view(e, n, -1))
    assert torch.equal(
        packing.pack_mxfp4_scales(s).reshape(-1), want_s.view(torch.uint8).reshape(-1)
    )


def _dims(weights):
    return layout.shape_dims(Q35Config(weights=weights))


def _regions(table):
    """(name, start, end) of a layout table in address order."""
    starts = sorted((off, name) for name, off in table.items() if name != "_bytes")
    ends = [off for off, _ in starts[1:]] + [table["_bytes"]]
    return [(name, off, end) for (off, name), end in zip(starts, ends)]


@pytest.mark.parametrize("weights", ["fp8", "mxfp4"])
@pytest.mark.parametrize("full", [False, True], ids=["gdn", "full"])
@pytest.mark.parametrize("tokens", TOKEN_COUNTS)
def test_regions_are_disjoint_nonempty_and_aligned(weights, full, tokens):
    scratch, sym = layout.layout(tokens, 4, full, **_dims(weights))
    for table in (scratch, sym):
        for name, start, end in _regions(table):
            assert start % 256 == 0, name
            assert end > start, name


@pytest.mark.parametrize("npes", [1, 2, 4, 8])
def test_symmetric_regions_hold_every_rank_and_token(npes):
    dims = _dims("fp8")
    _, sym = layout.layout(MAX_TOKENS, npes, False, **dims)
    sizes = {name: end - start for name, start, end in _regions(sym)}
    # (value, tag) pairs of two bf16 each: 8 bytes per two elements of a row
    row_bytes = dims["hidden"] * 4
    assert sizes["attn"] >= npes * MAX_TOKENS * row_bytes
    assert sizes["ffn"] >= npes * MAX_TOKENS * row_bytes


@pytest.mark.parametrize("tokens", TOKEN_COUNTS)
def test_router_takes_the_fewest_samples_a_task_that_fit_the_grid(tokens):
    n_experts = Q35Config().n_experts
    tasks = layout.router_rows(n_experts) // layout.ROUTER_TILE
    spt = layout.router_spt(tokens, n_experts)
    assert spt <= layout.ROUTER_TILE
    assert tokens * tasks <= layout.BLOCKS * spt or spt == layout.ROUTER_TILE
    assert spt == 1 or tokens * tasks > layout.BLOCKS * (spt // 2)


@pytest.mark.parametrize("tokens", TOKEN_COUNTS)
def test_down_tiles_cover_hidden_in_whole_row_groups(tokens):
    hidden = Q35Config().hidden
    tile = layout.dn_tile(tokens, hidden)
    assert tile % 16 == 0 and hidden % tile == 0


@pytest.mark.parametrize("weights", ["fp8", "mxfp4"])
@pytest.mark.parametrize("tokens_per_seq", [1, 2, 4])
def test_every_stage_has_tasks_and_gdn_runs_a_task_per_sequence_head(
    weights, tokens_per_seq
):
    dims = _dims(weights)
    for tokens in range(tokens_per_seq, MAX_TOKENS + 1, tokens_per_seq):
        for full in (False, True):
            tasks = dict(
                layout.stage_tasks(tokens, full, **dims, tokens_per_seq=tokens_per_seq)
            )
            assert all(n > 0 for n in tasks.values()), tasks
            assert tasks["ug"] == tokens * dims["top_k"] * dims["inter"] // layout.UG8
            if not full:
                assert tasks["gdn"] == tokens // tokens_per_seq * dims["hv"]


@pytest.mark.parametrize("weights", ["fp8", "mxfp4"])
@pytest.mark.parametrize("full", [False, True], ids=["gdn", "full"])
def test_mixer_output_mailboxes_follow_the_out_proj_input(weights, full):
    """The FP8 checkpoint's GDN output is FP8 already (its gated norm quantizes, as aiter's does): four
    bytes a pair and a scale per 128; every other mixer output is bf16 pairs. ``a`` holds the f32 stream.
    """
    dims = _dims(weights)
    tokens = 4
    scratch, _ = layout.layout(tokens, 4, full, **dims)
    sizes = {name: end - start for name, start, end in _regions(scratch)}
    out_k = layout.out_k(full, dims)
    if full or weights == "mxfp4":
        assert "gq" not in sizes and sizes["gated"] >= tokens * out_k // 2 * 8
    else:
        assert "gated" not in sizes
        assert sizes["gq"] >= tokens * out_k // 4 * 8
        assert sizes["gqs"] >= tokens * out_k // 128 * 8
    assert sizes["a"] >= tokens * dims["hidden"] * 8


# the BF16 GEMMs whose baseline partial-sum rounding the kernel takes: TP 4 per-rank (N, K)
_BF16_GEMMS = {"out": (4096, 2048), "shared_ug": (512, 4096), "router": (513, 4096)}


def _rule_of_aiter_pick(cfg: dict) -> tuple:
    """The partial-sum rounding of an aiter tuned_gemm pick (see ``layout.BF16_GEMM_RULES``)."""
    name = cfg.get("kernelName") or ""
    if cfg.get("libtype") == "flydsl":
        params = name.split("_")
        tile = next(p for p in params if p.startswith("t") and "x" in p)
        ks = next(p for p in params if p.startswith("ks"))
        waves = next(p for p in params if p.startswith("w") and "x" in p)
        k_waves = int(waves[1:].split("x")[2])
        split_k = 1 if ks == "ks1" else int(cfg["splitK"])
        assert (split_k == 1) or (k_waves == 1), name
        if split_k > 1:
            return ("splitk", split_k)
        if k_waves == 2 and tile.split("x")[2] == "256":
            return ("halves",)
        assert k_waves == 1, name
    return ("single",)


@pytest.mark.parametrize("name", sorted(_BF16_GEMMS))
def test_bf16_gemm_rules_match_aiters_tuned_picks(name):
    """A re-tune of aiter's BF16 GEMM configs can change which partial sums baseline ATOM rounds: the
    pinned table must follow the installed aiter (needs its gfx950 tuned configs, so a GPU).
    """
    tg = pytest.importorskip("aiter.tuned_gemm")
    if not torch.cuda.is_available():
        pytest.skip("aiter's tuned config lookup reads the GPU's CU count")
    N, K = _BF16_GEMMS[name]
    for M in range(1, 17):
        cfg = tg.get_GEMM_A16W16_config(
            M, N, K, False, "torch.bfloat16", "torch.bfloat16", False, False
        )
        assert layout.BF16_GEMM_RULES[name][M] == _rule_of_aiter_pick(cfg), (M, cfg)


@pytest.mark.parametrize("tokens", TOKEN_COUNTS)
def test_baseline_m_is_the_graph_padded_decode_batch(tokens):
    m = layout.baseline_m(tokens)
    assert m == (tokens if tokens <= 8 else 16)
    assert layout.baseline_m(tokens, tokens_per_seq=4) == tokens
