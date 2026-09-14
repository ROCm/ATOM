"""The FP4 sparse indexer: the predicate, the ABI shapes, the KV pool, and one
component cross-check against the bytes the production writer emits.

`indexer_qk_rope_quant_and_cache` in FP4 mode is the only writer of the packed
E2M1 Q/K and their e8m0 planes, and `flydsl_pa_mqa_logits_fp4[_prefill]` the
only readers, so what is worth checking is that the two agree on a real DSA
indexer's shapes (H=32, D=128, kv_block=64, block_k=256) -- decode at both
next_n GLM-5.2 runs, prefill down both schedule paths. The reference dequantizes
exactly what the writer produced, so a disagreement is a layout bug, not
rounding. The FP8 default has to come out of all of it untouched.
"""

from types import SimpleNamespace

import pytest
import torch

from atom.model_ops import sparse_indexer_fp4
from atom.model_ops.attentions.mla_kv_pool import MlaKvPool
from atom.model_ops.sparse_indexer_fp4 import (
    FP4_KV_BLOCK_SIZE,
    FP4_MQA_BLOCK_K,
    FP4_QUANT_BLOCK_SIZE,
    assert_fp4_indexer_supported,
    fp4_decode_parallel_units,
    fp4_decode_schedule,
    fp4_prefill_schedule,
    fp4_q_scale_shape,
    sparse_indexer_fp4_enabled,
)

# The DSA indexer geometry GLM-5.2 and DeepSeek-V3.2 share.
DSA = SimpleNamespace(index_topk=2048, index_n_heads=32, index_head_dim=128)

HEADS, HEAD_DIM, _BLOCK = 32, 128, FP4_KV_BLOCK_SIZE
WEIGHTS_SCALE = HEAD_DIM**-0.5 * HEADS**-0.5
_E2M1_MAG = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
_E2M1 = torch.cat([_E2M1_MAG, -_E2M1_MAG])


def _pool(**overrides):
    args = {
        "layers": 2,
        "block_size": 64,
        "entry_dim": 576,
        "kv_dtype": torch.bfloat16,
        "index_layers": 3,
        "index_rows_per_block": 64,
        "index_dim": 144,
        "index_dtype": torch.uint8,
        "index_head_dim": 128,
    }
    args.update(overrides)
    return MlaKvPool(**args)


def test_predicate_is_structural_and_never_probes_the_chip_for_fp8(monkeypatch):
    monkeypatch.setattr(sparse_indexer_fp4, "_gfx", lambda: "gfx950")
    assert sparse_indexer_fp4_enabled("fp4", DSA)
    # An MTP draft: `_MTP_TYPE_MAP` rewrote its model_type but not its indexer,
    # and it shares the target's cache, so it must reach the target's verdict.
    assert sparse_indexer_fp4_enabled(
        "fp4", SimpleNamespace(model_type="deepseek_mtp", **vars(DSA))
    )

    monkeypatch.delattr(sparse_indexer_fp4, "_gfx")
    assert not sparse_indexer_fp4_enabled("fp8", DSA)
    assert not sparse_indexer_fp4_enabled(None, DSA)


@pytest.mark.parametrize(
    ("override", "gfx", "why"),
    [
        ({"index_topk": 0}, "gfx950", "no sparse indexer"),
        ({"index_head_dim": 64}, "gfx950", "index_head_dim is 64"),
        ({"index_n_heads": 24}, "gfx950", "index_n_heads is 24"),
        ({}, "gfx942", "gfx942"),
    ],
)
def test_predicate_falls_back_and_names_what_blocked_it(
    monkeypatch, caplog, override, gfx, why
):
    monkeypatch.setattr(sparse_indexer_fp4, "_gfx", lambda: gfx)
    config = SimpleNamespace(**{**vars(DSA), **override})
    assert not sparse_indexer_fp4_enabled("fp4", config)
    with caplog.at_level("WARNING", logger="atom"):
        assert not sparse_indexer_fp4_enabled("fp4", config, warn=True)
    assert why in caplog.text


def test_unsupported_fp4_requests_name_the_knob_that_blocked_them():
    assert_fp4_indexer_supported(fused_writer=True, prefill_context_parallel=False)
    with pytest.raises(ValueError, match="fused QK/RoPE/cache"):
        assert_fp4_indexer_supported(fused_writer=False, prefill_context_parallel=False)
    # PCP's candidate exchange is the only reader of the indexer op's return, so
    # the FP4 path may leave that tensor unwritten only while this refusal holds.
    with pytest.raises(ValueError, match="does not support PCP"):
        assert_fp4_indexer_supported(fused_writer=True, prefill_context_parallel=True)


def test_decode_parallel_units_are_a_multiple_of_next_n_covering_the_batch():
    for next_n in (1, 2, 3, 4, 8):
        for max_bs in (1, 16, 512, 8192):
            units = fp4_decode_parallel_units(max_bs, next_n)
            assert units % next_n == 0
            assert units // next_n >= max_bs
            assert units >= sparse_indexer_fp4.FP4_MQA_VARCTX_PARALLEL_UNIT_NUM


def test_q_scale_shape_pads_the_m_tile_axis_to_one_dword():
    # H=32 is two M-tiles, still loaded as one dword of four scale bytes.
    assert fp4_q_scale_shape(7, 32, 128) == (7, 1, 4, 16, 4)
    assert fp4_q_scale_shape(7, 64, 128) == (7, 1, 4, 16, 4)
    assert fp4_q_scale_shape(7, 128, 128) == (7, 1, 4, 16, 8)


def test_index_field_narrows_under_fp4_without_adding_an_arena():
    fp8, fp4 = _pool(), _pool(index_fp4=True)
    assert len(fp8.field_groups) == len(fp4.field_groups) == 2
    assert [f.name for f in fp8.index_fields] == ["index"]
    assert [f.name for f in fp4.index_fields] == ["index", "index_scale"]
    assert fp8.entry_bytes == 2 * 64 * 576 * 2 + 3 * 64 * 144
    assert fp4.entry_bytes == 2 * 64 * 576 * 2 + 3 * (4 * 64 * 16) + 3 * (4 * 64)
    assert fp4.entry_bytes < fp8.entry_bytes

    fp8.allocate(3, "cpu")
    fp4.allocate(3, "cpu")
    assert fp8.layer("index", 0).shape == (3, 64, 144)
    data, scale = fp4.layer("index", 0), fp4.layer("index_scale", 0)
    assert data.shape == (3, 1, 4, 64, 16) and data.dtype is torch.uint8
    assert scale.shape == (3, 1, 4, 64) and scale.dtype is torch.uint8
    spans = [
        (t.data_ptr(), t.data_ptr() + t.numel() * t.element_size())
        for t in (fp4.layer("kv", 0), data, scale)
    ]
    for i, lhs in enumerate(spans):
        for rhs in spans[i + 1 :]:
            assert lhs[1] <= rhs[0] or rhs[1] <= lhs[0]

    # The indexer rows per block are constrained, the KV block size is not.
    _pool(index_fp4=True, block_size=32, index_rows_per_block=64)
    with pytest.raises(ValueError, match="--block-size 64"):
        _pool(index_fp4=True, index_rows_per_block=32)


@pytest.fixture
def on_gfx950(monkeypatch):
    """Gate the cross-check on a chip that has the kernels, single-rank."""
    if not torch.cuda.is_available():
        pytest.skip("requires a ROCm GPU")
    from aiter.jit.utils.chip_info import get_gfx

    if get_gfx() != "gfx950":
        pytest.skip("the FP4 paged-MQA-logits kernels are gfx950-only")
    # ATOM's shim reads the DCP world size off the global config, which a unit
    # test has no reason to build.
    from atom.model_ops import attention_mla

    monkeypatch.setattr(attention_mla, "get_dcp_world_size", lambda: 1)


def _dequant(packed: torch.Tensor, e8m0: torch.Tensor) -> torch.Tensor:
    """`[..., D // 2]` E2M1 pairs plus `[..., D // 32]` e8m0 -> fp32 `[..., D]`."""
    nibbles = torch.empty(
        *packed.shape[:-1], packed.shape[-1] * 2, dtype=torch.long, device=packed.device
    )
    nibbles[..., 0::2] = packed & 0xF
    nibbles[..., 1::2] = packed >> 4
    scale = torch.exp2(e8m0.float() - 127.0).repeat_interleave(
        FP4_QUANT_BLOCK_SIZE, dim=-1
    )
    return _E2M1.to(packed.device)[nibbles] * scale


def _paged_layout(batch: int, ctx_len: int):
    """A shuffled block table plus the slot of every KV token."""
    blocks_per_seq = ctx_len // _BLOCK
    num_blocks = batch * blocks_per_seq
    table = torch.randperm(num_blocks, device="cuda").to(torch.int32)
    table = table.reshape(batch, blocks_per_seq)
    token = torch.arange(ctx_len, device="cuda").repeat(batch)
    seq = torch.arange(batch, device="cuda").repeat_interleave(ctx_len)
    slots = table[seq, token // _BLOCK].long() * _BLOCK + token % _BLOCK
    return table, num_blocks, token, seq, slots


def _fused_fp4(slots, positions, num_blocks, weight_gain=1.0):
    """The production writer, in FP4 mode. Returns everything it emits.

    Through ATOM's own shim rather than `aiter.` directly: the shim is what
    decides `compute_all_q_rope` and forwards the two scale buffers, so it is
    the seam worth covering.
    """
    from atom.model_ops import attention_mla

    rows = slots.shape[0]
    u8 = {"dtype": torch.uint8, "device": "cuda"}
    bf16 = {"dtype": torch.bfloat16, "device": "cuda"}
    angles = torch.randn(4096, 32, device="cuda")
    norm = torch.randn(HEAD_DIM, dtype=torch.float32, device="cuda")
    weights = (torch.randn(rows, HEADS, device="cuda") * weight_gain).bfloat16()
    q_fp4 = torch.zeros(rows, HEADS, HEAD_DIM // 2, **u8)
    q_scale = torch.zeros(fp4_q_scale_shape(rows, HEADS, HEAD_DIM), **u8)
    weights_out = torch.zeros_like(weights)
    kv_cache = torch.zeros(num_blocks, 1, 4, _BLOCK, 16, **u8)
    kv_scale = torch.zeros(num_blocks, 1, 4, _BLOCK, **u8)
    attention_mla.indexer_qk_rope_quant_and_cache(
        torch.randn(rows, HEADS, HEAD_DIM, **bf16),
        q_fp4,
        weights,
        weights_out,
        torch.randn(rows, HEAD_DIM, **bf16),
        kv_cache,
        slots,
        norm,
        norm,
        positions,
        angles.cos().bfloat16(),
        angles.sin().bfloat16(),
        1e-6,
        FP4_QUANT_BLOCK_SIZE,
        "ue8m0",
        WEIGHTS_SCALE,
        is_neox=True,
        q_scale_out=q_scale,
        kv_cache_scale=kv_scale,
    )
    return q_fp4, q_scale, weights_out, kv_cache, kv_scale


def _oracle(q_fp4, q_scale, kv_cache, kv_scale, table, ctx_len, weights, rows_of):
    """The scorer's math in fp32 over the cache as written: per-head ReLU(q.k),
    weighted and summed. `rows_of` maps the per-sequence keys onto query rows."""
    batch = table.shape[0]
    token = torch.arange(ctx_len, device=kv_cache.device)
    phys = table[:, token // _BLOCK].long().unsqueeze(-1)
    pos = (token % _BLOCK).expand(batch, ctx_len).unsqueeze(-1)
    group = torch.arange(4, device=kv_cache.device)
    packed = kv_cache[phys, 0, group, pos].reshape(batch, ctx_len, HEAD_DIM // 2)
    keys = _dequant(packed, kv_scale[phys, 0, group, (pos % 16) * 4 + pos // 16])
    # `[T, k_tiles, 4, 16, qs_pad]` -> the dense `[T, H, D // 32]` a reader sees.
    dense = (
        q_scale[..., : HEADS // 16]
        .permute(0, 4, 3, 1, 2)
        .reshape(q_scale.shape[0], HEADS, HEAD_DIM // FP4_QUANT_BLOCK_SIZE)
    )
    scores = torch.einsum(
        "rhd,rtd->rht", _dequant(q_fp4, dense.contiguous()), rows_of(keys)
    )
    return (torch.relu(scores) * weights.float().unsqueeze(-1)).sum(1) * WEIGHTS_SCALE


def _assert_agrees(got, want, visible, topk):
    """Cosine over the visible window and the worst row's top-k overlap: the two
    numbers that say whether the selection this feeds would differ."""
    mask = torch.arange(want.shape[1], device=want.device)[None, :] < visible[:, None]
    a, b = got[mask].double(), want[mask].double()
    cosine = (a @ b / (a.norm() * b.norm())).item()
    lens = [min(topk, int(n)) for n in visible]
    overlap = min(
        len(
            set(got[r, : int(visible[r])].topk(k).indices.tolist())
            & set(want[r, : int(visible[r])].topk(k).indices.tolist())
        )
        / k
        for r, k in enumerate(lens)
        if k
    )
    assert cosine > 0.9999, cosine
    assert overlap > 0.99, overlap


@pytest.mark.parametrize(("batch", "next_n", "ctx_len"), [(3, 1, 1024), (2, 4, 768)])
def test_decode_scores_the_cache_the_fused_writer_wrote(
    on_gfx950, batch, next_n, ctx_len
):
    from aiter.ops.flydsl import flydsl_pa_mqa_logits_fp4
    from aiter.ops.flydsl.kernels.mqa_logits.pa_mqa_logits_fp4 import (
        compute_varctx_schedule,
    )

    torch.manual_seed(0)
    table, num_blocks, token, _, slots = _paged_layout(batch, ctx_len)
    *_, kv_cache, kv_scale = _fused_fp4(slots, token, num_blocks)

    # The query rows get their own throwaway cache, so their slots are real --
    # which is what a decode step passes, and what makes the writer compute Q.
    rows = batch * next_n
    q_fp4, q_scale, weights_out, *_ = _fused_fp4(
        torch.arange(rows, dtype=torch.int64, device="cuda"),
        torch.full((rows,), ctx_len - 1, dtype=torch.int64, device="cuda"),
        num_blocks,
        weight_gain=0.1,
    )

    ctx_lens = torch.full((batch,), ctx_len, dtype=torch.int32, device="cuda")
    _, cta_info, n_ctas = compute_varctx_schedule(
        ctx_lens, FP4_MQA_BLOCK_K, None, ctx_len, next_n=next_n
    )
    logits = torch.empty(rows, ctx_len, dtype=torch.float32, device="cuda")
    flydsl_pa_mqa_logits_fp4(
        q_fp4.reshape(batch, next_n, HEADS, HEAD_DIM // 2),
        q_scale.reshape(batch, next_n, *q_scale.shape[1:]),
        kv_cache,
        kv_scale,
        table,
        weights_out,
        ctx_lens,
        ctx_len,
        weight_scale=WEIGHTS_SCALE,
        next_n=next_n,
        block_k=FP4_MQA_BLOCK_K,
        kv_block_size=_BLOCK,
        out=logits,
        cta_info=cta_info,
        total_ctas=n_ctas,
    )

    want = _oracle(
        q_fp4,
        q_scale,
        kv_cache,
        kv_scale,
        table,
        ctx_len,
        weights_out,
        lambda keys: keys.repeat_interleave(next_n, dim=0),
    )
    # Each of a request's next_n rows sees one token less than the one after it.
    row = torch.arange(rows, device="cuda")
    visible = ctx_lens.repeat_interleave(next_n) - (next_n - 1 - row % next_n)
    _assert_agrees(logits, want, visible, topk=512)


def test_dcp_decode_scores_each_query_token_over_its_own_local_window(on_gfx950):
    """The geometry `dcp_decode_candidate_exchange_fused` hands the scorer: one
    row per query token over this rank's shard, so the windows are ragged and
    the schedule is built at next_n=1 whatever the speculation width."""
    from aiter.ops.flydsl import flydsl_pa_mqa_logits_fp4

    torch.manual_seed(2)
    batch, next_n, width = 3, 4, 1024
    rows = batch * next_n
    table, num_blocks, token, _, slots = _paged_layout(batch, width)
    *_, kv_cache, kv_scale = _fused_fp4(slots, token, num_blocks)
    q_fp4, q_scale, weights_out, *_ = _fused_fp4(
        torch.arange(rows, dtype=torch.int64, device="cuda"),
        torch.full((rows,), width - 1, dtype=torch.int64, device="cuda"),
        num_blocks,
        weight_gain=0.1,
    )

    # Ragged on purpose: a draft position's extra token lands on ONE rank, so
    # the local lengths of a request's next_n rows do not all advance together.
    local_ctx = torch.tensor(
        [width - (r % 7) * 37 for r in range(rows)], dtype=torch.int32, device="cuda"
    )
    units = fp4_decode_parallel_units(batch, next_n)
    cta_info = torch.zeros(units, 4, dtype=torch.int32, device="cuda")
    fp4_decode_schedule(local_ctx, FP4_MQA_BLOCK_K, units, width, 1, cta_info)

    logits = torch.empty(rows, width, dtype=torch.float32, device="cuda")
    flydsl_pa_mqa_logits_fp4(
        q_fp4.reshape(rows, 1, HEADS, HEAD_DIM // 2),
        q_scale.reshape(rows, 1, *q_scale.shape[1:]),
        kv_cache,
        kv_scale,
        table.repeat_interleave(next_n, dim=0),
        weights_out,
        local_ctx,
        width,
        weight_scale=WEIGHTS_SCALE,
        next_n=1,
        block_k=FP4_MQA_BLOCK_K,
        kv_block_size=_BLOCK,
        out=logits,
        cta_info=cta_info,
        total_ctas=units,
    )

    want = _oracle(
        q_fp4,
        q_scale,
        kv_cache,
        kv_scale,
        table,
        width,
        weights_out,
        lambda keys: keys.repeat_interleave(next_n, dim=0),
    )
    _assert_agrees(logits, want, local_ctx, topk=512)


def test_staged_page_table_is_bounded_by_the_block_table_width():
    """`pages` counts the whole co-scheduled prefill batch while the staged
    table is one sequence wide, so the two can cross -- and a short table would
    silently address page 0 for the tail columns rather than fault."""
    import numpy as np

    aiter_mla = pytest.importorskip(
        "atom.model_ops.attentions.aiter_mla",
        reason="the MLA builder imports triton at module scope",
    )

    build = aiter_mla.AiterMLAMetadataBuilder._build_dcp_indexer_fp4_prefill_meta
    block, bs, cols = 64, 2, 6
    builder = SimpleNamespace(
        model_runner=SimpleNamespace(block_size=block), device=torch.device("cpu")
    )
    lpad = np.full(bs, block, dtype=np.int64)
    cu_pad = np.concatenate([[0], np.cumsum(lpad)]).astype(np.int64)
    var = {"block_tables": SimpleNamespace(np=np.zeros((bs, 8), dtype=np.int32))}
    meta = SimpleNamespace(block_tables=torch.zeros(bs, cols, dtype=torch.int32))

    # Short of the bound: identity over `pages`, and the tail the scorer never
    # reads stays zero rather than aliasing a real page.
    build(builder, meta, bs, lpad, cu_pad, 4 * block, var)
    staged = meta.dcp_indexer_fp4_block_tables
    assert staged.shape == (bs, cols)
    assert torch.equal(staged[:, :4], torch.arange(4, dtype=torch.int32).expand(bs, 4))
    assert not staged[:, 4:].any()

    # Exactly at it: the width is the last addressable page, not one past.
    build(builder, meta, bs, lpad, cu_pad, cols * block, var)
    assert torch.equal(
        meta.dcp_indexer_fp4_block_tables,
        torch.arange(cols, dtype=torch.int32).expand(bs, cols),
    )

    with pytest.raises(ValueError, match="only 6 columns wide"):
        build(builder, meta, bs, lpad, cu_pad, cols * block + 1, var)


@pytest.mark.parametrize("whole_batch", [True, False])
def test_prefill_scores_the_same_cache_seq_locally(on_gfx950, whole_batch):
    from atom.models.deepseek_v2 import _prefill_mqa_logits_fp4

    torch.manual_seed(1)
    batch, ctx_len = 2, 512
    table, num_blocks, token, seq, slots = _paged_layout(batch, ctx_len)
    q_fp4, q_scale, weights_out, kv_cache, kv_scale = _fused_fp4(
        slots, token, num_blocks, weight_gain=0.1
    )

    # One row per query token, each seeing `[0, its own position]` of its own
    # sequence -- the seq-local windows the metadata builder publishes.
    rows = batch * ctx_len
    local_ends = (token + 1).to(torch.int32)
    row_to_batch = seq.to(torch.int32)
    cta_info, n_ctas, local_starts = fp4_prefill_schedule(
        row_to_batch, local_ends, FP4_MQA_BLOCK_K, rows, ctx_len
    )
    # With `whole_batch=False` the model rebuilds the schedule per chunk instead
    # of reusing the one the builder left on the metadata.
    logits = _prefill_mqa_logits_fp4(
        SimpleNamespace(
            batch_id_per_q_token=row_to_batch,
            block_tables=table,
            indexer_fp4_local_starts=local_starts,
            indexer_fp4_local_ends=local_ends,
            indexer_fp4_max_seq_len=ctx_len,
            indexer_fp4_cta_info=cta_info,
            indexer_fp4_n_ctas=n_ctas,
        ),
        slice(0, rows),
        whole_batch,
        q_fp4,
        q_scale,
        weights_out,
        kv_cache,
        kv_scale,
        WEIGHTS_SCALE,
        _BLOCK,
        table,
    )

    want = _oracle(
        q_fp4,
        q_scale,
        kv_cache,
        kv_scale,
        table,
        ctx_len,
        weights_out,
        lambda keys: keys[row_to_batch.long()],
    )
    _assert_agrees(logits, want, local_ends, topk=256)
