"""The FP4 sparse indexer: the predicate, the ABI shapes, the KV pool, and one
component cross-check against the bytes the production writer emits.

`indexer_qk_rope_quant_and_cache` in FP4 mode is the only writer of the packed
E2M1 Q/K and their e8m0 planes, and `flydsl_pa_mqa_logits_fp4[_prefill]` the
only readers, so what is worth checking is that the two agree on a real DSA
indexer's shapes (H=32, D=128, kv_block=64, block_k=256) -- decode at a
speculation width and at the DCP path's one row per query token, prefill down
both schedule paths. The reference dequantizes
exactly what the writer produced, so a disagreement is a layout bug, not
rounding. The FP8 default has to come out of all of it untouched.
"""

import importlib
from types import SimpleNamespace

import numpy as np
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
    fp4_index_scale_rows,
    fp4_prefill_schedule,
    fp4_q_scale_shape,
    sparse_indexer_fp4_enabled,
)


def _expect_scale_row(rows, block):
    """The e8m0 row swizzle, written out so tests don't ask the code under test."""
    return (rows % 16) * (block // 16) + rows // 16


# The DSA indexer geometry GLM-5.2 and DeepSeek-V3.2 share.
DSA = SimpleNamespace(index_topk=2048, index_n_heads=32, index_head_dim=128)

HEADS, HEAD_DIM, _BLOCK = 32, 128, FP4_KV_BLOCK_SIZE
WEIGHTS_SCALE = HEAD_DIM**-0.5 * HEADS**-0.5
_E2M1_MAG = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
_E2M1 = torch.cat([_E2M1_MAG, -_E2M1_MAG])


def _import_or_skip(name: str, reason: str | None = None):
    """Import `name`, skipping the test where this box cannot.

    Not `pytest.importorskip`: it warns on a plain `ImportError` rather than a
    missing module -- which CI escalates, and pytest 9.1 will too -- and on a
    CPU runner that is exactly how these fail, `aiter` importing but carrying no
    `QuantType`.
    """
    try:
        return importlib.import_module(name)
    except ImportError as exc:
        pytest.skip(reason or f"{name} is unavailable here: {exc}")


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
        ({"index_kpool": 2}, "gfx950", "index_kpool is 2"),
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
    def check(**overrides):
        assert_fp4_indexer_supported(
            **{
                "fused_writer": True,
                "prefill_context_parallel": False,
                "prefill_ubatching": False,
                **overrides,
            }
        )

    check()
    with pytest.raises(ValueError, match="fused QK/RoPE/cache"):
        check(fused_writer=False)
    # PCP's candidate exchange is the only reader of the indexer op's return, so
    # the FP4 path may leave that tensor unwritten only while this refusal holds.
    with pytest.raises(ValueError, match="does not support PCP"):
        check(prefill_context_parallel=True)
    # Decode micro-batching is supported; only the prefill split is not, because
    # it rebuilds metadata from declared fields and the schedule rides on
    # undeclared ones. A blanket TBO refusal would take the decode path with it.
    with pytest.raises(ValueError, match="prefill micro-batching"):
        check(prefill_ubatching=True)


def test_every_backend_answers_the_draft_s_fp4_schedule_publish():
    """`EagleProposer` refreshes this on whatever builder the target uses, and
    only the MLA one has an FP4 indexer to refresh. Every other backend has to
    answer it anyway: EAGLE3 on Llama-3, MTP on Qwen3-Next and on DeepSeek-V4
    (whose builder is a `CommonAttentionBuilder` sibling, not an MLA subclass)
    all reach that line with FP4 nowhere in the picture."""
    backends = _import_or_skip("atom.model_ops.attentions.backends")
    base = backends.CommonAttentionBuilder._publish_indexer_fp4_decode_schedule

    mla = _import_or_skip("atom.model_ops.attentions.aiter_mla")
    assert mla.AiterMLAMetadataBuilder._publish_indexer_fp4_decode_schedule is not base

    for module, name in (
        ("atom.model_ops.attentions.aiter_attention", "AiterAttentionMetadataBuilder"),
        (
            "atom.model_ops.attentions.deepseek_v4_attn",
            "DeepseekV4AttentionMetadataBuilder",
        ),
        ("atom.model_ops.attentions.gdn_attn", "GDNAttentionMetadataBuilder"),
        ("atom.model_ops.attentions.triton_mha", "TritonMHAMetadataBuilder"),
    ):
        builder = getattr(_import_or_skip(module), name)
        assert builder._publish_indexer_fp4_decode_schedule is base, name

    # Inert, not merely present: the draft reuses the target's metadata object,
    # so anything written here would reach the verify step.
    metadata = SimpleNamespace()
    base(object(), metadata, 4, 1)
    assert not vars(metadata)


def test_the_builder_compares_the_indexer_s_fp4_verdict_instead_of_setting_it():
    """The builder and `Indexer.__init__` answer the same predicate from the
    same two inputs, and the Indexer has already built `k_cache` from its answer
    by the time the builder reaches it. Assigning over it cannot fix a
    divergence -- the object is already built -- it only hides one, to surface
    later as a graph/eager dtype mismatch. Read off the source because reaching
    that line needs an allocated pool and a loaded model."""
    import ast
    import pathlib

    root = pathlib.Path(sparse_indexer_fp4.__file__).parent
    tree = ast.parse((root / "attentions" / "aiter_mla.py").read_text(encoding="utf-8"))
    overwrites = [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Attribute)
        and target.attr == "_indexer_fp4"
        and isinstance(target.value, ast.Attribute)
        and target.value.attr == "indexer"
    ]
    assert not overwrites, f"builder overwrites the Indexer's verdict at {overwrites}"


def test_decode_parallel_units_cover_the_batch_at_every_speculation_width():
    """Everything the one captured buffer rests on: a slot per sequence per
    step, the varctx floor, and `f(n) >= f(1)` -- the buffer is sized at
    `max_seqlen_qo` and the draft then asks at 1. That last one is the weakest
    true statement, not the obvious one: `f` is NOT monotonic in `next_n`."""
    for max_bs in (1, 7, 16, 64, 128, 300, 512, 8192):
        floor = fp4_decode_parallel_units(max_bs, 1)
        for next_n in range(1, 17):
            units = fp4_decode_parallel_units(max_bs, next_n)
            assert units % next_n == 0
            assert units // next_n >= max_bs
            assert units >= sparse_indexer_fp4.FP4_MQA_VARCTX_PARALLEL_UNIT_NUM
            assert units >= floor

    # The counterexample the docstring names, so it cannot rot back.
    assert fp4_decode_parallel_units(1, 3) > fp4_decode_parallel_units(1, 4)


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


def _written_cache_and_queries(batch, next_n, ctx_len, seed):
    """A paged cache plus `batch * next_n` query rows, both from the writer.

    The query rows get real slots of their own, which is what a decode step
    passes and what makes the writer compute Q at all.
    """
    torch.manual_seed(seed)
    rows = batch * next_n
    table, num_blocks, token, _, slots = _paged_layout(batch, ctx_len)
    *_, kv_cache, kv_scale = _fused_fp4(slots, token, num_blocks)
    q_fp4, q_scale, weights_out, *_ = _fused_fp4(
        torch.arange(rows, dtype=torch.int64, device="cuda"),
        torch.full((rows,), ctx_len - 1, dtype=torch.int64, device="cuda"),
        num_blocks,
        weight_gain=0.1,
    )
    return table, kv_cache, kv_scale, q_fp4, q_scale, weights_out, rows


def _oracle(q_fp4, q_scale, kv_cache, kv_scale, table, ctx_len, weights, rows_of):
    """The scorer's math in fp32 over the cache as written: per-head ReLU(q.k),
    weighted and summed. `rows_of` maps the per-sequence keys onto query rows."""
    batch = table.shape[0]
    token = torch.arange(ctx_len, device=kv_cache.device)
    phys = table[:, token // _BLOCK].long().unsqueeze(-1)
    pos = (token % _BLOCK).expand(batch, ctx_len).unsqueeze(-1)
    group = torch.arange(4, device=kv_cache.device)
    packed = kv_cache[phys, 0, group, pos].reshape(batch, ctx_len, HEAD_DIM // 2)
    keys = _dequant(packed, kv_scale[phys, 0, group, _expect_scale_row(pos, _BLOCK)])
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


def test_scale_row_swizzle_matches_its_oracle():
    """The e8m0 row swizzle on CPU: a wrong one is silent, every index in bounds."""
    rows = torch.arange(_BLOCK)
    want = _expect_scale_row(rows, _BLOCK)
    assert torch.equal(fp4_index_scale_rows(rows, _BLOCK), want)
    assert sorted(want.tolist()) == list(range(_BLOCK))

    with pytest.raises(ValueError, match="64-row blocks"):
        fp4_index_scale_rows(rows, 16)


def test_decode_scores_the_cache_the_fused_writer_wrote(on_gfx950):
    """The rectangular kernel at a speculation width: one row per (seq, step),
    each seeing one token less than the step after it. `next_n=1` is the DCP
    test's shape, so what this one holds down is the `next_n > 1` reshape."""
    from aiter.ops.flydsl import flydsl_pa_mqa_logits_fp4
    from aiter.ops.flydsl.kernels.mqa_logits.pa_mqa_logits_fp4 import (
        compute_varctx_schedule,
    )

    batch, next_n, ctx_len = 2, 4, 768
    table, kv_cache, kv_scale, q_fp4, q_scale, weights_out, rows = (
        _written_cache_and_queries(batch, next_n, ctx_len, seed=0)
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

    batch, next_n, width = 3, 4, 1024
    table, kv_cache, kv_scale, q_fp4, q_scale, weights_out, rows = (
        _written_cache_and_queries(batch, next_n, width, seed=2)
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


def test_dcp_prefill_staging_keeps_every_key_with_its_own_exponent(monkeypatch):
    """Staging moves two planes whose row axes disagree -- the packed one flat,
    the e8m0 one transposed -- so it cannot address them with a single index.

    Every index stays in bounds either way, so the failure is silent: keys come
    back wearing another row's exponent. Real exponents are nearly uniform
    inside a block because `k_norm` precedes the quantizer, which is why an
    end-to-end accuracy run can pass with this broken; the random planes here
    remove that cover.
    """
    dsv2 = _import_or_skip("atom.models.deepseek_v2")

    torch.manual_seed(0)
    block, world, src_pages = _BLOCK, 2, 4
    local = 64
    total_kv = world * local
    shape = {"dtype": torch.uint8, "device": "cpu"}
    data_src = torch.randint(0, 256, (src_pages, 1, 4, block, 16), **shape)
    scale_src = torch.randint(0, 256, (src_pages, 1, 4, block), **shape)

    # No source row may land on its own row, or a missing swizzle goes unseen.
    # Built by rotation, not randperm, so it holds on any box and RNG stream.
    g = torch.Generator().manual_seed(0)
    src_row = (torch.arange(local) + 1) % block
    src_page = torch.randperm(src_pages, generator=g).repeat(-(-local // src_pages))[
        :local
    ]
    slots = (src_page * block + src_row).to(torch.int32)

    tok = torch.arange(total_kv)
    gather_index = ((tok + 1) % local + local * (tok // local)).to(torch.int32)

    _src = slots[gather_index.long() % local].long()
    assert not torch.any(_src % block == tok % block), (
        "some source row lands on its own row; those positions cannot tell a "
        "correct swizzle from a missing one"
    )
    monkeypatch.setattr(
        dsv2,
        "get_dcp_group",
        lambda: SimpleNamespace(
            all_gather=lambda t, dim: t.repeat(world, *([1] * (t.dim() - 1)))
        ),
    )

    staged, staged_scale = dsv2._dcp_stage_indexer_fp4_prefill(
        data_src,
        scale_src,
        SimpleNamespace(
            dcp_indexer_fp4_local_slots=slots,
            dcp_indexer_gather_index=gather_index,
            # What the builder publishes, written out.
            dcp_indexer_fp4_read_page=slots.long() // block,
            dcp_indexer_fp4_read_row=slots.long() % block,
            dcp_indexer_fp4_read_scale_row=_expect_scale_row(
                slots.long() % block, block
            ),
            dcp_indexer_fp4_stage_page=tok // block,
            dcp_indexer_fp4_stage_row=tok % block,
            dcp_indexer_fp4_stage_scale_row=_expect_scale_row(tok % block, block),
        ),
        total_kv,
        block,
    )

    src = slots[gather_index.long() % local].long()
    got_rows = torch.arange(total_kv)
    q_dst = _expect_scale_row(got_rows % block, block)
    q_src = _expect_scale_row(src % block, block)
    assert torch.equal(
        staged[got_rows // block, 0, :, got_rows % block, :],
        data_src[src // block, 0, :, src % block, :],
    )
    assert torch.equal(
        staged_scale[got_rows // block, 0, :, q_dst],
        scale_src[src // block, 0, :, q_src],
    )


def test_staged_page_table_spans_a_whole_batch_not_one_sequence():
    """`pages` counts the summed co-scheduled prefill context, and prefix
    caching lets that run past any one sequence's block allowance --
    `max_num_batched_tokens` bounds only the uncached tokens. Sized at that
    allowance the table would turn a legal schedule into a mid-serving raise,
    so it spans a full batch; the tail the scorer never reads stays zero rather
    than aliasing a real page."""
    aiter_mla = _import_or_skip(
        "atom.model_ops.attentions.aiter_mla",
        reason="the MLA builder imports triton at module scope",
    )

    build = aiter_mla.AiterMLAMetadataBuilder._build_dcp_indexer_fp4_prefill_meta
    block, bs, per_seq = 64, 2, 6
    builder = SimpleNamespace(
        model_runner=SimpleNamespace(block_size=block),
        device=torch.device("cpu"),
        max_bs=4,
        block_table_cols=per_seq,
    )
    cols = builder.max_bs * per_seq
    lpad = np.full(bs, block, dtype=np.int64)
    cu_pad = np.concatenate([[0], np.cumsum(lpad)]).astype(np.int64)
    var = {"block_tables": SimpleNamespace(np=np.zeros((bs, 8), dtype=np.int32))}
    meta = SimpleNamespace()

    def staged_for(total_kv):
        build(builder, meta, bs, lpad, cu_pad, total_kv, var)
        return meta.dcp_indexer_fp4_block_tables

    for pages in (4, per_seq + 1, cols):
        staged = staged_for(pages * block)
        # Past `per_seq` is the case a one-sequence width used to raise on.
        assert staged.shape == (bs, cols), pages
        want = torch.arange(pages, dtype=torch.int32).expand(bs, pages)
        assert torch.equal(staged[:, :pages], want), pages
        assert not staged[:, pages:].any(), pages


@pytest.mark.parametrize(
    "n_slots, n_iota",
    [(0, 0), (1, 1), (7, 5), (300, 100), (1023, 1024), (1025, 4000), (60_000, 232_003)],
)
def test_decompose_slots_matches_torch(n_slots, n_iota):
    """decompose_slots_triton vs torch: tile tails, either input longer, negative slots."""
    if not torch.cuda.is_available():
        pytest.skip("requires a ROCm GPU")
    block_convert = _import_or_skip("atom.utils.block_convert")

    g = torch.Generator().manual_seed(0)
    slots = torch.randint(-5 * _BLOCK, 40_000 * _BLOCK, (n_slots,), generator=g)
    slots = slots.to(torch.int32).cuda()
    lut = fp4_index_scale_rows(torch.arange(_BLOCK, dtype=torch.int32), _BLOCK).cuda()
    got = block_convert.decompose_slots_triton(slots, n_iota, _BLOCK, lut)

    token = torch.arange(n_iota, dtype=torch.int32, device="cuda")
    want = []
    for src in (slots, token):
        page, row = src // _BLOCK, src % _BLOCK
        want += [page, row, fp4_index_scale_rows(row, _BLOCK)]
    for i, (a, b) in enumerate(zip(got, want)):
        assert a.dtype == torch.int32 and torch.equal(a, b), i


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_builder_publishes_the_staging_indices_it_derives(device):
    """The builder's six staging index tensors, on the kernel and torch paths.

    Non-trivial block tables and mid-block sequence ends, so page, row and
    swizzled row all differ; on GPU this also pins outputs to their names.
    """
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("requires a ROCm GPU")
    aiter_mla = _import_or_skip(
        "atom.model_ops.attentions.aiter_mla",
        reason="the MLA builder imports triton at module scope",
    )

    build = aiter_mla.AiterMLAMetadataBuilder._build_dcp_indexer_fp4_prefill_meta
    block, bs, per_seq = 64, 2, 6
    builder = SimpleNamespace(
        model_runner=SimpleNamespace(block_size=block),
        device=torch.device(device),
        max_bs=4,
        block_table_cols=per_seq,
    )
    if device == "cuda":
        rows = torch.arange(block, dtype=torch.int32, device=device)
        builder._fp4_scale_row_lut = fp4_index_scale_rows(rows, block)
    lpad = np.array([block + 5, 2 * block + 3], dtype=np.int64)
    cu_pad = np.concatenate([[0], np.cumsum(lpad)]).astype(np.int64)
    table = np.array([[7, 3, 0, 0], [11, 2, 9, 0]], dtype=np.int32)
    var = {"block_tables": SimpleNamespace(np=table)}
    total_kv = 3 * block + 7
    meta = SimpleNamespace()
    build(builder, meta, bs, lpad, cu_pad, total_kv, var)

    want_slots = torch.tensor(
        [7 * block + j for j in range(block)]
        + [3 * block + j for j in range(5)]
        + [11 * block + j for j in range(block)]
        + [2 * block + j for j in range(block)]
        + [9 * block + j for j in range(3)]
    )
    assert torch.equal(meta.dcp_indexer_fp4_local_slots.long().cpu(), want_slots)

    tok = torch.arange(total_kv)
    for side, src in (("read", want_slots), ("stage", tok)):
        page = getattr(meta, f"dcp_indexer_fp4_{side}_page").cpu()
        row = getattr(meta, f"dcp_indexer_fp4_{side}_row").cpu()
        scale_row = getattr(meta, f"dcp_indexer_fp4_{side}_scale_row").cpu()
        assert page.dtype == row.dtype == scale_row.dtype == torch.int32, side
        assert torch.equal(page.long(), src // block), side
        assert torch.equal(row.long(), src % block), side
        assert torch.equal(
            scale_row.long(), _expect_scale_row(src % block, block)
        ), side


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
