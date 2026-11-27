# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""The Qwen3.8-Flash-Next wrappers that launch through aiter's `fast_launch`:
each wrapper on the GPU against the same launches through plain
`kernel[grid]`. The key itself is tested against Triton's in aiter
(`op_tests/triton_tests/utils/test_fast_launch.py`)."""

from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("triton")
fast_launch = pytest.importorskip(
    "aiter.ops.triton.utils.fast_launch", exc_type=ImportError
)


def _bitwise_equal(a, b):
    return len(a) == len(b) and all(
        torch.equal(x.view(torch.uint8), y.view(torch.uint8)) for x, y in zip(a, b)
    )


def _plain_vs_cached(run, *launches):
    """`run()` through plain `kernel[grid]`, then twice through the wrappers of
    `launches` ((wrapper, kernel) pairs): the first run compiles into each
    wrapper, the second hits for every launch."""
    for wrapper, _ in launches:
        wrapper.clear()
    with fast_launch.bypassed():
        ref = [t.clone() for t in run()]
    with fast_launch.recording() as compiled:
        first = [t.clone() for t in run()]
    with fast_launch.recording() as missed:
        second = run()
    torch.cuda.synchronize()
    assert {kernel.__name__ for _, kernel in launches} <= {n for n, _ in compiled}
    assert missed == []
    assert _bitwise_equal(first, ref)
    assert _bitwise_equal(second, ref)


gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")


def _randn(*shape, dtype=torch.bfloat16, offset=0, scale=1.0, seed=0, device="cuda"):
    gen = torch.Generator(device=device).manual_seed(seed)
    numel = 1
    for size in shape:
        numel *= size
    flat = (torch.randn(numel + offset, generator=gen, device=device) * scale).to(dtype)
    # Slice after allocating operations so offset cases stay misaligned.
    out = flat[offset:].view(shape)
    assert out.storage_offset() == offset
    assert out.data_ptr() % 16 == (offset * out.element_size()) % 16
    return out


# --- GDN: q/k L2 norm, causal conv1d, gating -------------------------------

# Qwen3.8-Flash-Next linear attention at TP2: 8 key heads, 24 value heads.
_K_HEADS, _V_HEADS, _HEAD_DIM, _CONV_WIDTH = 8, 24, 128, 4
_K_DIM, _V_DIM = _K_HEADS * _HEAD_DIM, _V_HEADS * _HEAD_DIM
_CONV_DIM = 2 * _K_DIM + _V_DIM


@pytest.fixture
def gdn():
    pytest.importorskip("aiter.dist.parallel_state", exc_type=ImportError)
    from atom.model_ops import attention_gdn
    from atom.model_ops.fla_ops import l2norm
    from atom.model_ops.mamba_ops import causal_conv1d

    return SimpleNamespace(l2norm=l2norm, conv=causal_conv1d, gating=attention_gdn)


def _conv_inputs(lens, offset=0, native_layout=False, initial=True, device="cuda"):
    tokens = sum(lens)
    mixed = _randn(tokens, _CONV_DIM, offset=offset, seed=tokens, device=device)
    weight = _randn(_CONV_DIM, _CONV_WIDTH, scale=0.3, seed=1, device=device)
    slots = 8
    if native_layout:  # ModelRunner's [slot, state_len, conv_dim], transposed
        states = _randn(slots, _CONV_WIDTH - 1, _CONV_DIM, device=device)
        states = states.transpose(-1, -2)
    else:  # SGLang's [slot, conv_dim, state_len]
        states = _randn(slots, _CONV_DIM, _CONV_WIDTH - 1, device=device)
    starts = [0]
    for n in lens:
        starts.append(starts[-1] + n)
    return SimpleNamespace(
        x=mixed.transpose(0, 1),
        weight=weight,
        states=states,
        query_start_loc=torch.tensor(starts, dtype=torch.int32, device=device),
        cache_indices=torch.arange(1, len(lens) + 1, dtype=torch.int32, device=device),
        has_initial_state=torch.full(
            (len(lens),), initial, dtype=torch.bool, device=device
        ),
        tokens=tokens,
    )


def _conv1d(gdn, t, states, metadata):
    return gdn.conv.causal_conv1d_fn(
        t.x,
        t.weight,
        None,
        states,
        t.query_start_loc,
        _K_DIM,
        _V_DIM,
        cache_indices=t.cache_indices,
        has_initial_state=t.has_initial_state,
        activation="silu",
        metadata=metadata,
    )


def _conv_metadata(gdn, t):
    nums, batch, offsets = gdn.conv.compute_causal_conv1d_metadata(
        t.query_start_loc, t.tokens
    )
    return SimpleNamespace(
        nums_dict=nums, batch_ptr=batch, token_chunk_offset_ptr=offsets
    )


def _gating_inputs(tokens, offset=0, device="cuda"):
    ba = _randn(tokens, 2 * _V_HEADS, offset=offset, seed=tokens, device=device)
    params = [
        torch.nn.Parameter(_randn(_V_HEADS, seed=s, device=device), False)
        for s in (2, 3)
    ]
    # `in_proj_ba` hands over strided views of one projection.
    return params[0], ba[:, _V_HEADS:], ba[:, :_V_HEADS], params[1]


@gpu
@pytest.mark.parametrize("offset", [0, 1], ids=["aligned", "misaligned"])
@pytest.mark.parametrize("tokens", [2112, 2096, 16, 2111, 3, 1])
@torch.inference_mode()
def test_l2norm_and_gating_match_plain_triton_bitwise(gdn, tokens, offset):
    x = _randn(1, tokens, _K_HEADS, _HEAD_DIM, offset=offset, seed=tokens)
    _plain_vs_cached(
        lambda: [gdn.l2norm.l2norm_fwd(x)],
        (gdn.l2norm._l2norm_fwd2, gdn.l2norm.l2norm_fwd_kernel2),
    )
    gating = _gating_inputs(tokens, offset)
    _plain_vs_cached(
        lambda: list(gdn.gating.fused_gdn_gating(*gating)),
        (gdn.gating._fused_gdn_gating, gdn.gating.fused_gdn_gating_kernel),
    )


@gpu
@pytest.mark.parametrize("initial", [True, False], ids=["resume", "fresh"])
@pytest.mark.parametrize("native_layout", [False, True], ids=["sglang", "native"])
@pytest.mark.parametrize("offset", [0, 1], ids=["aligned", "misaligned"])
@pytest.mark.parametrize("lens", [(2112,), (16,), (2111,), (7, 1000, 1, 64)])
@torch.inference_mode()
def test_causal_conv1d_matches_plain_triton_bitwise(
    gdn, lens, offset, native_layout, initial
):
    """The conv state is updated in place, so each run starts from a copy."""
    t = _conv_inputs(lens, offset, native_layout, initial)
    metadata = _conv_metadata(gdn, t)

    def run():
        if native_layout:
            storage = t.states.transpose(-1, -2).clone()
            states = storage.transpose(-1, -2)
        else:
            storage = states = t.states.clone()
        return [*_conv1d(gdn, t, states, metadata), storage]

    _plain_vs_cached(
        run, (gdn.conv._causal_conv1d_fwd, gdn.conv._causal_conv1d_fwd_kernel)
    )
    _plain_vs_cached(
        lambda: list(_conv_metadata(gdn, t).__dict__.values())[1:],
        (gdn.conv._causal_conv1d_metadata, gdn.conv._causal_conv1d_metadata_kernel),
    )


@gpu
@torch.inference_mode()
def test_gating_and_l2norm_replay_in_a_cuda_graph(gdn):
    """`fused_gdn_gating` also runs in the decode graph."""
    gating = _gating_inputs(16)
    x = _randn(1, 16, _K_HEADS, _HEAD_DIM)
    ref = (*gdn.gating.fused_gdn_gating(*gating), gdn.l2norm.l2norm_fwd(x))
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = (*gdn.gating.fused_gdn_gating(*gating), gdn.l2norm.l2norm_fwd(x))
    graph.replay()
    torch.cuda.synchronize()
    assert _bitwise_equal(out, ref)


# --- output gates: sigmoid RMSNorm + FP8 quant (GDN), sigmoid gate (QSA, MoE)


@pytest.fixture
def gated():
    pytest.importorskip("aiter.ops.enum", exc_type=ImportError)
    from atom.model_ops.qwen4_exp.ops import gated

    return gated


# GDN output per rank at TP2 (24 value heads x 128), QSA output (12 x 256).
_GDN_OUT, _QSA_OUT = (24, 128), 12 * 256
_HIDDEN, _ROUTED_EXPERTS = 2560, 512
_FP8 = torch.float8_e4m3fnuz


def _rmsnorm_quant_inputs(tokens, offset=0, device="cuda"):
    heads, dim = _GDN_OUT
    x = _randn(tokens, heads, dim, offset=offset, seed=tokens, device=device)
    # The gate is a slice of the in_proj_qkvz output: strided tokens.
    z = _randn(tokens, 2 * heads * dim, seed=1, device=device)[:, heads * dim :]
    weight = torch.nn.Parameter(_randn(dim, seed=2, device=device), False)
    return x, z.view(tokens, heads, dim), weight, 1e-6, _FP8


def _mul_quant_inputs(tokens, offset=0, device="cuda"):
    x = _randn(tokens, _QSA_OUT, offset=offset, seed=tokens, device=device)
    # The gate is a slice of the fused QKVG projection.
    gate = _randn(tokens, _QSA_OUT + 1024, seed=3, device=device)[:, :_QSA_OUT]
    return x, gate, _FP8


@gpu
@pytest.mark.parametrize("offset", [0, 1], ids=["aligned", "misaligned"])
@pytest.mark.parametrize("tokens", [2112, 1056, 1024, 1025, 16, 3, 1])
@torch.inference_mode()
def test_gated_quant_match_plain_triton_bitwise(gated, tokens, offset):
    """Across the 1024-row switch from 4 to 2 warps in `sigmoid_mul_quant`."""
    inputs = _rmsnorm_quant_inputs(tokens, offset)
    _plain_vs_cached(
        lambda: list(gated.sigmoid_rmsnorm_quant(*inputs)),
        (gated._sigmoid_rmsnorm_quant, gated._sigmoid_rmsnorm_quant_kernel),
    )
    inputs = _mul_quant_inputs(tokens, offset)
    _plain_vs_cached(
        lambda: list(gated.sigmoid_mul_quant(*inputs)),
        (gated._sigmoid_mul_quant, gated._sigmoid_mul_quant_kernel),
    )


def _mul_inputs(tokens, per_row, offset=0, device="cuda"):
    """The BF16 output gates: QSA attention's per-element gate, or the MoE
    shared expert's per-row gate (the last router logit) plus the routed
    output."""
    if not per_row:
        x, gate, _ = _mul_quant_inputs(tokens, offset, device)
        return x, gate, None
    x = _randn(tokens, _HIDDEN, offset=offset, seed=tokens, device=device)
    logits = _randn(tokens, _ROUTED_EXPERTS + 1, seed=4, device=device)
    routed = _randn(tokens, _HIDDEN, seed=5, device=device)
    return x, logits[:, _ROUTED_EXPERTS:], routed


@gpu
@pytest.mark.parametrize("per_row", [False, True], ids=["qsa", "shared_expert"])
@pytest.mark.parametrize("offset", [0, 1], ids=["aligned", "misaligned"])
@pytest.mark.parametrize("tokens", [2112, 2096, 16, 3, 1])
@torch.inference_mode()
def test_sigmoid_mul_matches_plain_triton_bitwise(gated, tokens, offset, per_row):
    inputs = _mul_inputs(tokens, per_row, offset)
    _plain_vs_cached(
        lambda: [gated.sigmoid_mul(*inputs)],
        (gated._gated_pointwise, gated._gated_pointwise_kernel),
    )


# --- QSA: mRoPE, cache rows, group pooling, selection expansion -------------


@pytest.fixture
def qsa():
    pytest.importorskip("aiter.ops.topk", exc_type=ImportError)
    from atom.model_ops.qwen4_exp.ops import qsa

    return qsa


def _rope(device="cuda", rotary_dim=64, max_position=16384):
    """The checkpoint's mRoPE: 64 rotated dims, interleaved [11, 11, 10]."""
    return SimpleNamespace(
        rotary_dim=rotary_dim,
        is_neox_style=True,
        mrope_section=[11, 11, 10],
        mrope_interleaved=True,
        cos_cache=_randn(
            max_position, rotary_dim // 2, dtype=torch.float32, seed=5, device=device
        ),
        sin_cache=_randn(
            max_position, rotary_dim // 2, dtype=torch.float32, seed=6, device=device
        ),
    )


def _qsa_batch(tokens, context=8192, page=64, pool_pages=400, device="cuda"):
    """One request whose last `tokens` of `context` are in this forward, on
    randomly placed pages."""
    gen = torch.Generator(device=device).manual_seed(tokens)
    order = torch.randperm(pool_pages, generator=gen, device=device).to(torch.int32)
    used = -(-context // page)
    table = torch.full((1, 256), -1, dtype=torch.int32, device=device)
    table[0, :used] = order[:used]
    positions = torch.arange(context - tokens, context, device=device)
    token_to_req = torch.zeros(tokens, dtype=torch.int32, device=device)
    slots = table[0, (positions // page).long()].long() * page + positions % page
    return SimpleNamespace(
        table=table,
        positions=positions,
        token_to_req=token_to_req,
        slots=slots,
        compressed=torch.where((positions + 1) % 4 == 0, slots // 4, -1),
        seq_lens=torch.full((1,), context, dtype=torch.int32, device=device),
        raw_cache=_randn(pool_pages, page, 1, 128, seed=7, device=device),
        compressed_cache=_randn(pool_pages, page // 4, 1, 128, seed=8, device=device),
    )


_KERNELS = {
    "_qsa_mrope": "_qsa_mrope_kernel",
    "_qsa_store_rows": "_qsa_store_rows_kernel",
    "_qsa_compress_groups": "_qsa_compress_groups_kernel",
    "_expand_selected": "_expand_selected_groups",
    "_qsa_compressed_slots": "_qsa_compressed_slots_kernel",
}


def _qsa_helper_runs(qsa, tokens, offset=0, device="cuda"):
    """name -> (launcher attribute, run) over one QSA layer's helper calls,
    with the shapes and strides the layer passes them at TP2. A run returns
    every tensor it writes; the caches are written into copies."""
    b = _qsa_batch(tokens, device=device)
    rope = _rope(device)
    qkvg = _randn(tokens, 2 * 3072 + 512, offset=offset, seed=9, device=device)
    q, k = qkvg[:, 3072:6144].view(tokens, 12, 256), qkvg[:, 6144:6400].view(
        tokens, 1, 256
    )
    index_q = _randn(tokens, 4, 128, offset=offset, seed=10, device=device)
    positions = b.positions.unsqueeze(0).expand(3, -1).contiguous()
    # The indexer's raw key is a slice of its fused q/k projection.
    raw_key = _randn(tokens, 5 * 128, seed=11, device=device)[:, 512:].view(
        tokens, 1, 128
    )
    pooled = _randn(tokens, 1, 128, offset=offset, seed=12, device=device)
    first = b.positions.unsqueeze(1).expand(-1, 3).transpose(0, 1)
    gen = torch.Generator(device=device).manual_seed(13)
    groups = torch.randint(
        -1, 2048, (tokens, 512), dtype=torch.int32, device=device, generator=gen
    )

    def store(cache, slots, values):
        cache = cache.clone()
        qsa.qsa_store_rows(cache, slots, values)
        return [cache]

    def compressed_slots():
        out = torch.empty(tokens, dtype=torch.int64, device=device)
        qsa.qsa_compressed_slots(b.slots, b.positions, 4, out)
        return [out]

    return {
        "mrope": (
            "_qsa_mrope",
            lambda: list(qsa.qsa_apply_mrope(rope, positions, q, k)),
        ),
        "mrope_index_q": (
            "_qsa_mrope",
            lambda: qsa.qsa_apply_mrope(rope, positions, index_q)[:1],
        ),
        "mrope_keys": (
            "_qsa_mrope",
            lambda: qsa.qsa_apply_mrope(rope, first, pooled)[:1],
        ),
        "store_raw": ("_qsa_store_rows", lambda: store(b.raw_cache, b.slots, raw_key)),
        "store_compressed": (
            "_qsa_store_rows",
            lambda: store(b.compressed_cache, b.compressed, pooled),
        ),
        "compress": (
            "_qsa_compress_groups",
            lambda: list(
                qsa.qsa_compress_groups(
                    b.raw_cache, b.table, b.token_to_req, b.positions, b.compressed, 4
                )
            ),
        ),
        "expand": (
            "_expand_selected",
            lambda: [
                qsa.qsa_expand_block_indices(
                    groups,
                    b.positions,
                    b.seq_lens,
                    b.token_to_req,
                    4,
                    2048,
                    output_width=2051,
                )
            ],
        ),
        "compressed_slots": ("_qsa_compressed_slots", compressed_slots),
    }


@gpu
@pytest.mark.parametrize("offset", [0, 1], ids=["aligned", "misaligned"])
@pytest.mark.parametrize("tokens", [2112, 1056, 2096, 16, 2111, 3, 1])
@torch.inference_mode()
def test_qsa_helpers_match_plain_triton_bitwise(qsa, tokens, offset):
    for attr, run in _qsa_helper_runs(qsa, tokens, offset).values():
        _plain_vs_cached(run, (getattr(qsa, attr), getattr(qsa, _KERNELS[attr])))


@gpu
@torch.inference_mode()
def test_qsa_helpers_replay_in_a_cuda_graph(qsa):
    """mRoPE and the cache writes also run in the decode graph."""
    runs = _qsa_helper_runs(qsa, 16)
    names = ("mrope", "store_raw", "store_compressed")
    ref = [t for name in names for t in runs[name][1]()]
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = [t for name in names for t in runs[name][1]()]
    graph.replay()
    torch.cuda.synchronize()
    assert _bitwise_equal(out, ref)


# --- QSA scoring and sparse attention (Native: KV block 16, Triton kernels) --


def _native_qsa_batch(tokens, context=8192, page=16, pool_pages=2400, device="cuda"):
    gen = torch.Generator(device=device).manual_seed(tokens)
    order = torch.randperm(pool_pages, generator=gen, device=device).to(torch.int32)
    used = -(-context // page)
    table = torch.full((1, 1024), -1, dtype=torch.int32, device=device)
    table[0, :used] = order[:used]
    positions = torch.arange(context - tokens, context, device=device)
    picks = torch.randint(0, context, (tokens, 2051), generator=gen, device=device)
    return SimpleNamespace(
        table=table,
        positions=positions,
        token_to_req=torch.zeros(tokens, dtype=torch.int32, device=device),
        seq_lens=torch.full((1,), context, dtype=torch.int32, device=device),
        # Causal token picks, -1 padded, as `qsa_select_paged_tokens` hands over.
        selected=torch.where(picks <= positions[:, None], picks, -1).to(torch.int32),
        keys=_randn(pool_pages, page // 4, 1, 128, seed=14, device=device),
        k_cache=_randn(pool_pages, page, 1, 256, seed=15, device=device),
        v_cache=_randn(pool_pages, page, 1, 256, seed=16, device=device),
    )


def _score(qsa, b, q):
    """Logits only up to each row's visible count: column tiles past every
    row's horizon are never written, and top-k never reads them."""
    row_starts = torch.empty(q.shape[0], dtype=torch.int32, device=q.device)
    logits, visible = qsa.qsa_paged_mqa_logits(
        q,
        b.keys,
        b.table,
        b.token_to_req,
        b.positions,
        b.seq_lens,
        4,
        max_columns=2048,
        row_starts=row_starts,
    )
    columns = torch.arange(logits.shape[1], device=q.device)
    seen = torch.where(columns[None, :] < visible[:, None], logits, 0)
    return [seen, visible, row_starts]


def _attend(qsa, b, q):
    out = qsa.qsa_sparse_paged_gqa(
        q,
        b.k_cache,
        b.v_cache,
        b.selected,
        b.table,
        b.token_to_req,
        softmax_scale=256**-0.5,
    )
    return [out]


@gpu
@pytest.mark.parametrize("offset", [0, 1], ids=["aligned", "misaligned"])
@pytest.mark.parametrize("tokens", [2096, 1056, 2111, 300, 256, 16, 3, 1])
@torch.inference_mode()
def test_qsa_scoring_and_attention_match_plain_triton_bitwise(qsa, tokens, offset):
    """Above 256 rows the MFMA scorer, up to 256 the stacked one; one
    attention split for prefill-sized batches, split-K for small ones."""
    b = _native_qsa_batch(tokens)
    q = _randn(tokens, 4, 128, offset=offset, seed=17)
    scorer = (
        (qsa._scorer_mfma, qsa._qsa_paged_mqa_logits_mfma_kernel)
        if tokens > 256
        else (qsa._scorer_stacked, qsa._qsa_paged_mqa_logits_stacked_kernel)
    )
    _plain_vs_cached(lambda: _score(qsa, b, q), scorer)
    q = _randn(tokens, 12, 256, offset=offset, seed=18)
    _plain_vs_cached(
        lambda: _attend(qsa, b, q),
        (qsa._sparse_paged_gqa, qsa._qsa_sparse_paged_gqa_kernel),
    )


# --- fused Q/K RMSNorm (QSA attention's q_norm / k_norm) --------------------


@pytest.fixture
def layernorm():
    pytest.importorskip("aiter.ops.triton.fused_add_rmsnorm_pad", exc_type=ImportError)
    from atom.model_ops import layernorm

    return layernorm


def _qk_norm_inputs(tokens, offset=0, device="cuda"):
    qkvg = _randn(tokens, 2 * 3072 + 512, offset=offset, seed=tokens, device=device)
    weights = [
        torch.nn.Parameter(_randn(256, seed=s, device=device), False) for s in (19, 20)
    ]
    q = qkvg[:, 3072:6144].view(tokens, 12, 256)
    k = qkvg[:, 6144:6400].view(tokens, 1, 256)
    return q, k, *weights, 1e-6


@gpu
@pytest.mark.parametrize("weightless_q", [False, True])
@pytest.mark.parametrize("offset", [0, 1], ids=["aligned", "misaligned"])
@pytest.mark.parametrize("tokens", [2112, 2096, 683, 682, 16, 3, 1])
@torch.inference_mode()
def test_fused_qk_norm_matches_plain_triton_bitwise(
    layernorm, tokens, offset, weightless_q
):
    """683 tokens are the first past the 8192-row switch to two rows a program."""
    q, k, q_weight, k_weight, eps = _qk_norm_inputs(tokens, offset)
    q_weight = None if weightless_q else q_weight
    _plain_vs_cached(
        lambda: list(
            layernorm.fused_qk_norm(
                q, k, q_weight, k_weight, eps, add_unit_offset=not weightless_q
            )
        ),
        (layernorm._fused_qk_norm_single, layernorm._fused_qk_norm_single_kernel),
    )


# --- HC: combine + norm, scaled SiLU, gated mean (+ FP8 quant) -------------


@pytest.fixture
def hc():
    pytest.importorskip("aiter.ops.enum", exc_type=ImportError)
    from atom.model_ops.qwen4_exp.ops import gated, hc_fused

    return SimpleNamespace(gated=gated, hc_fused=hc_fused)


def _hc_inputs(m, offset=0, seed=0, hidden=2560, streams=4, rank=320):
    gen = torch.Generator(device="cuda").manual_seed(seed)

    def randn(*shape, scale=1.0):
        x = torch.randn(*shape, generator=gen, device="cuda", dtype=torch.bfloat16)
        return x * scale

    h = randn(m * streams * hidden + offset)[offset:].view(m, streams * hidden)
    d = randn(m, rank + streams)
    return SimpleNamespace(
        h=h,
        norm_w=randn(streams * hidden, scale=0.1),
        block=randn(m, hidden),
        raw=d[:, rank:],
        x=d[:, :rank],
        up=randn(m, streams * hidden),
        hidden=hidden,
        streams=streams,
        rank=rank,
    )


def _hc_launches(hc, t, combine, quant_dtype):
    block, raw = (t.block, t.raw) if combine else (None, None)
    h, normed = hc.hc_fused.hc_combine_norm(t.h, t.norm_w, 1e-6, t.streams, block, raw)
    gate = hc.gated.scaled_silu(t.x, t.streams)
    if quant_dtype is None:
        tail = (gate, hc.hc_fused.hc_gated_mean(normed, t.up, t.streams))
    else:
        tail = (
            gate,
            *hc.hc_fused.hc_gated_mean_quant(normed, t.up, t.streams, quant_dtype),
        )
    return (h, normed), tail


@gpu
@pytest.mark.parametrize("offset", [0, 1], ids=["aligned", "misaligned"])
@pytest.mark.parametrize("combine", [True, False])
@pytest.mark.parametrize("m", [2112, 1056, 1050, 256, 16, 3])
@torch.inference_mode()
def test_hc_launches_match_plain_triton_bitwise(hc, m, combine, offset):
    """Prefill (whole and SP-shard), mid and small batches, combining or not,
    streams 16-byte aligned or not."""
    t = _hc_inputs(m, offset=offset, seed=m + offset)
    quant = None if m > 1024 else torch.float8_e4m3fnuz
    kernels = hc.hc_fused
    combine_norm = (
        (kernels._hc_combine_norm_rows, kernels._hc_combine_norm_rows_kernel)
        if m >= kernels.COMBINE_NORM_ROWS_MIN_M
        else (kernels._hc_combine_norm, kernels._hc_combine_norm_kernel)
    )
    gated_mean = (
        (kernels._hc_gated_mean_tiled, kernels._hc_gated_mean_tiled_kernel)
        if quant is None
        else (kernels._hc_gated_mean_quant, kernels._hc_gated_mean_quant_kernel)
    )
    _plain_vs_cached(
        lambda: [x for part in _hc_launches(hc, t, combine, quant) for x in part],
        combine_norm,
        (hc.gated._gated_pointwise, hc.gated._gated_pointwise_kernel),
        gated_mean,
    )


@gpu
@torch.inference_mode()
def test_hc_launches_replay_in_a_cuda_graph(hc):
    t = _hc_inputs(1056, seed=11)
    ref, _ = _hc_launches(hc, t, True, None)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        head, _ = _hc_launches(hc, t, True, None)
    graph.replay()
    torch.cuda.synchronize()
    assert _bitwise_equal(head, ref)
