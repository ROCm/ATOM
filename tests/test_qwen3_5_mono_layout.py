# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""The Qwen3.5-MoE mono kernel's weight layouts (against aiter's shuffles) and its
build-time mailbox layout (no GPU)."""

import pytest
import torch

from atom.models.qwen3_5.mono import layout
from atom.models.qwen3_5.mono.config import MAX_TOKENS, Q35Config

TOKEN_COUNTS = range(1, MAX_TOKENS + 1)


# The layouts the kernel reads ATOM's FP8 and MXFP4 weights in, written out to check against aiter's
# own shuffles.


def shuffle_fp8(q: torch.Tensor) -> torch.Tensor:
    """FP8 ``[..., N, K]`` in aiter's ``shuffle_weight(layout=(16, 16))`` order, the layout ATOM holds its
    FP8 GEMM / MoE weights in: 16-row, 32-K tiles of two 16-K halves, each 16 rows x 16 contiguous K.
    """

    *_lead, rows, k = q.shape
    if rows % 16 or k % 64:
        raise ValueError(
            f"FP8 matrix dimensions must be divisible by (16, 64), got {(rows, k)}"
        )
    w8 = q.view(torch.uint8).reshape(-1, rows // 16, 16, k // 32, 2, 16)
    return w8.permute(0, 1, 3, 4, 2, 5).contiguous().view(-1)


def pack_mxfp4(q: torch.Tensor) -> torch.Tensor:
    """An MXFP4 bank [E, N, K / 2] in ATOM's gfx950 layout for gate / up NOT interleaved (aiter
    ``moe_shuffle_weight(is_guinterleave=False)``: a [gate | up] bank shuffles as plain rows).
    """
    q = q.view(torch.uint8)
    e, n, kp = q.shape
    if n % 16 or kp % 64:
        raise ValueError(
            f"MXFP4 bank dimensions must be divisible by (16, 128), got {(n, kp * 2)}"
        )
    return (
        q.view(e, n // 16, 16, kp // 64, 4, 16)
        .permute(0, 1, 3, 4, 2, 5)
        .contiguous()
        .view(e, n, kp)
    )


def pack_mxfp4_scales(s: torch.Tensor) -> torch.Tensor:
    """E8M0 scales [E, N, K / 32] in ATOM's gfx950 layout (aiter ``moe_shuffle_scale``,
    ``is_guinterleave=False``); K / 32 padded to a multiple of 8 with 1.0 (0x7F)."""
    s = s.view(torch.uint8)
    e, n, kb = s.shape
    kb8 = -(-kb // 8) * 8
    if kb8 != kb:
        s = torch.cat(
            [s, torch.full((e, n, kb8 - kb), 0x7F, dtype=torch.uint8, device=s.device)],
            dim=-1,
        )
    return (
        s.view(e, n // 32, 2, 16, kb8 // 8, 2, 4)
        .permute(0, 1, 4, 6, 3, 5, 2)
        .contiguous()
        .view(e, n, kb8)
    )


def _bytes(shape, lo=0, hi=256):
    return torch.randint(lo, hi, shape, dtype=torch.uint8)


@pytest.mark.parametrize("shape", [(64, 256), (3, 32, 4096)])
def test_fp8_matrices_read_in_atoms_shuffle(shape):
    shuffle = pytest.importorskip("aiter.ops.shuffle")
    q = _bytes(shape).view(torch.float8_e4m3fn)
    want = shuffle.shuffle_weight(q, layout=(16, 16)).view(torch.uint8).reshape(-1)
    assert torch.equal(shuffle_fp8(q), want)


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
    assert torch.equal(pack_mxfp4(w), want_w.view(torch.uint8).view(e, n, -1))
    assert torch.equal(
        pack_mxfp4_scales(s).reshape(-1), want_s.view(torch.uint8).reshape(-1)
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


@pytest.mark.parametrize(
    "pick, rule",
    [
        (None, ("single",)),
        ({"libtype": "hipblaslt", "kernelName": "Cijk_x"}, ("single",)),
        (
            {
                "libtype": "flydsl",
                "kernelName": "hgemm_t16x16x256_ks1_w1x1x2",
                "splitK": 0,
            },
            ("halves",),
        ),
        (
            {
                "libtype": "flydsl",
                "kernelName": "hgemm_t16x64x128_ks4_w1x1x1",
                "splitK": 4,
            },
            ("splitk", 4),
        ),
        (
            {
                "libtype": "flydsl",
                "kernelName": "hgemm_t16x64x128_ks1_w1x1x1",
                "splitK": 0,
            },
            ("single",),
        ),
        # a pick the kernel has no rule for never matches one
        (
            {
                "libtype": "flydsl",
                "kernelName": "hgemm_t16x16x128_ks1_w1x1x2",
                "splitK": 0,
            },
            ("unknown", "hgemm_t16x16x128_ks1_w1x1x2"),
        ),
        # nor does a name it cannot read
        ({"libtype": "flydsl", "kernelName": "hgemm_new"}, ("unknown", "hgemm_new")),
    ],
)
def test_an_aiter_pick_reads_as_its_partial_sum_rounding(pick, rule):
    assert layout.aiter_bf16_rule(pick) == rule


@pytest.mark.parametrize("weights", ["fp8", "mxfp4"])
def test_every_bf16_gemm_the_kernel_follows_is_asked_of_aiter(weights):
    """Every BF16 GEMM on the decode path the kernel reproduces, at its TP 4 shard shape: a rule of
    ``BF16_GEMM_RULES`` or one fp32 sum."""
    got = {
        name: (n, k, rule)
        for name, n, k, rule in layout.bf16_gemms(Q35Config(weights=weights))
    }
    want = {"router": (513, 4096, "router"), "in_proj_ba": (32, 4096, None)}
    if weights == "mxfp4":
        want |= {
            "in_proj_qkvz": (5120, 4096, None),
            "qkv_proj": (4608, 4096, None),
            "out_proj": (4096, 2048, "out"),
            "o_proj": (4096, 2048, "out"),
            "shared gate_up": (512, 4096, "shared_ug"),
            "shared down": (4096, 256, None),
        }
    assert got == want


@pytest.mark.parametrize("weights", ["fp8", "mxfp4"])
def test_bf16_gemm_rules_match_aiters_tuned_picks(weights):
    """A re-tune of aiter's BF16 GEMM configs can change which partial sums baseline ATOM rounds: the
    runner refuses then (``bf16_gemm_refusal``), and the installed aiter must agree with the table
    (needs its gfx950 tuned configs, so a GPU)."""
    pytest.importorskip("aiter.tuned_gemm")
    if not torch.cuda.is_available():
        pytest.skip("aiter's tuned config lookup reads the GPU's CU count")
    from atom.models.qwen3_5.mono.runner import bf16_gemm_refusal

    assert bf16_gemm_refusal(Q35Config(weights=weights)) is None


def test_every_launch_width_has_its_bf16_gemm_rule():
    """A launch's rows are the M baseline ATOM's GEMMs run at (ATOM pads the step before the model sees
    it): every width the kernel launches has its rounding rule."""
    for name, rules in layout.BF16_GEMM_RULES.items():
        assert set(TOKEN_COUNTS) <= set(rules), name


def test_the_mono_package_imports_without_the_model():
    """``from atom.models.qwen3_5 import mono`` loads the subpackage alone: not the model (and AITER)."""
    import subprocess
    import sys

    code = (
        "import sys; from atom.models.qwen3_5 import mono; "
        "assert 'atom.models.qwen3_5.model' not in sys.modules, 'model imported'"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_rope_tables_hold_the_drafted_positions_past_the_last_kept_token():
    """``max_pos + spare`` rows: ATOM's own rows as they are, past its caches the last one repeated (a
    token there is dropped); a cache short of ``max_pos`` positions is refused."""
    from types import SimpleNamespace

    from atom.models.qwen3_5.mono.weights import rope_tables

    cfg = Q35Config()
    half = cfg.rope_dim // 2
    cos = torch.arange(10 * half, dtype=torch.bfloat16).view(10, 1, half)
    rot = SimpleNamespace(cos_cache=cos, sin_cache=-cos)
    c, s_ = rope_tables(rot, cfg, 8, spare=3)
    assert c.shape == (11, half) and c.dtype == torch.float32 and c.is_contiguous()
    assert torch.equal(c[:10], cos.view(10, half).float())
    assert torch.equal(c[10], c[9]) and torch.equal(s_, -c)
    assert rope_tables(rot, cfg, 11) is None


@pytest.mark.parametrize(
    "qlen, seqs, bt_rows, refused",
    [
        (4, 2, 2, False),  # the kernel's 2 requests of 4 tokens
        (
            4,
            4,
            4,
            False,
        ),  # a graph's padded sequence axis: the original path still takes 2
        (6, 2, 2, True),  # the metadata's query length is another verify's
        (4, 1, 2, True),  # fewer sequences than requests: the original path runs gluon
        (4, 2, 1, True),  # fewer block-table rows than requests: gluon too
    ],
)
def test_flydsl_splits_follow_the_original_sequence_count(qlen, seqs, bt_rows, refused):
    """Under ATOM_PA_FLYDSL the original path plans the first rows // max_seqlen_q sequences, and only while
    they fit its metadata (``_flydsl_pa_decode_num_seqs``); else it runs gluon. A step otherwise keeps the
    original path."""
    pytest.importorskip("aiter")
    from types import SimpleNamespace

    from atom.models.qwen3_5.mono.op import StepInputError
    from atom.models.qwen3_5.mono.runner import check_splits

    md = SimpleNamespace(
        context_lens=torch.zeros(seqs, dtype=torch.int32),
        block_tables=torch.zeros(bt_rows, 4, dtype=torch.int32),
        max_seqlen_q=qlen,
    )
    if refused:
        with pytest.raises(StepInputError):
            check_splits(md, 2, 4, flydsl=True)
    else:
        check_splits(md, 2, 4, flydsl=True)
