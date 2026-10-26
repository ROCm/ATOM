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
from atom.model_ops.sparse_indexer_chunk import dcp_qsplit_rows
from atom.model_ops.sparse_indexer_fp4 import (
    FP4_KV_BLOCK_SIZE,
    FP4_QUANT_BLOCK_SIZE,
    assert_fp4_indexer_supported,
    fp4_index_scale_rows,
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
    # PCP's candidate exchange needs fp32 weights that the FP4 writer does not
    # produce, independently of the native custom op's side-effect-only ABI.
    with pytest.raises(ValueError, match="does not support PCP"):
        check(prefill_context_parallel=True)
    # Decode micro-batching is supported; only the prefill split is not, because
    # it rebuilds metadata from declared fields and the schedule rides on
    # undeclared ones. A blanket TBO refusal would take the decode path with it.
    with pytest.raises(ValueError, match="prefill micro-batching"):
        check(prefill_ubatching=True)


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


def _fused_fp4(slots, positions, num_blocks, weight_gain=1.0, weights_fill=0.0):
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
    weights_out = torch.full_like(weights, weights_fill)
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


@pytest.mark.parametrize("next_n", [5, 6])
def test_fp4_writer_covers_valid_rows_and_skips_padding(on_gfx950, monkeypatch, next_n):
    from atom.model_ops import attention_mla

    batch, short = 2, 2
    rows = batch * next_n
    ends = torch.tensor(
        list(range(1, next_n + 1))
        + [max(0, short - next_n + row + 1) for row in range(next_n)],
        dtype=torch.int32,
        device="cuda",
    )
    valid = ends > 0
    slots = torch.full((rows,), -1, dtype=torch.int64, device="cuda")
    slots[:next_n] = torch.arange(next_n, device="cuda")
    slots[next_n:][valid[next_n:]] = 8 * _BLOCK + torch.arange(short, device="cuda")
    positions = (ends - 1).clamp_min(0).long()

    def write(fill):
        torch.manual_seed(7)
        return _fused_fp4(slots, positions, 16, weight_gain=0.1, weights_fill=fill)

    zero, poison = write(0.0), write(float("nan"))
    assert torch.equal(zero[2][valid], poison[2][valid])
    assert torch.isnan(poison[2][~valid]).all()
    assert torch.equal(zero[0][valid], poison[0][valid])
    assert torch.equal(zero[1][valid], poison[1][valid])
    assert torch.equal(zero[3], poison[3]) and torch.equal(zero[4], poison[4])

    # DCP computes every scheduled row instead of skipping slot<0 rows.
    monkeypatch.setattr(attention_mla, "get_dcp_world_size", lambda: 2)
    dcp_zero, dcp_poison = write(0.0), write(float("nan"))
    assert torch.equal(dcp_zero[2], dcp_poison[2])
    assert not torch.isnan(dcp_poison[2]).any()


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
    """A speculation width's rows, a request's one sequence
    (`Fp4MqaRaggedMetadata`, as the decode scorer passes them): one row per
    (seq, step), each seeing one token less than the step after it. `next_n=1`
    is the DCP test's shape, so what this one holds down is a sequence's rows
    sharing its block table."""
    from aiter.ops.flydsl import flydsl_pa_mqa_logits_fp4

    from atom.model_ops.fp4_mqa_ragged_metadata import Fp4MqaRaggedMetadata

    batch, next_n, ctx_len = 2, 4, 768
    table, kv_cache, kv_scale, q_fp4, q_scale, weights_out, rows = (
        _written_cache_and_queries(batch, next_n, ctx_len, seed=0)
    )
    ctx_lens = torch.full((batch,), ctx_len, dtype=torch.int32, device="cuda")
    # Each of a request's next_n rows sees one token less than the one after it.
    row = torch.arange(rows, device="cuda")
    visible = (ctx_lens.repeat_interleave(next_n) - (next_n - 1 - row % next_n)).int()
    starts = torch.arange(0, (batch + 1) * next_n, next_n, device="cuda").int()
    logits = torch.empty(rows, ctx_len, dtype=torch.float32, device="cuda")
    flydsl_pa_mqa_logits_fp4(
        q_fp4.reshape(batch, next_n, HEADS, HEAD_DIM // 2),
        q_scale,
        kv_cache,
        kv_scale,
        weights=weights_out,
        max_seq_len=ctx_len,
        weight_scale=WEIGHTS_SCALE,
        kv_block_size=_BLOCK,
        out=logits,
        **Fp4MqaRaggedMetadata(starts, next_n, table).kernel_args(
            visible, heads=HEADS, page_size=_BLOCK, max_seq_len=ctx_len
        ),
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
    _assert_agrees(logits, want, visible, topk=512)


def _fp4_query(q_fp4, q_scale):
    """The shared scorer's FP4 query: values and scales a row each, `[rows,
    1, ...]`, as `quantize_query` returns them."""
    return q_fp4.unsqueeze(1), q_scale.unsqueeze(1)


def test_v4_decode_scorer_takes_ragged_requests_and_padding(on_gfx950):
    """The shared scorer at a ragged MTP step, on rows as the V4 builder
    publishes them (`fp4_rows`, a request's rows together): requests of 4, 1
    and 3 rows (each row of a c-row request one token behind the next), an
    empty padding request and two padding rows. Every real row picks the
    cache's top-k within its own bound; a padding row picks nothing."""
    from atom.model_ops.fp4_mqa_ragged_metadata import Fp4MqaRaggedMetadata
    from atom.model_ops.indexer import IndexCache, fp4_rows, score_topk

    batch, next_n, ctx_len, topk = 3, 4, 768, 512
    table, kv_cache, kv_scale, q_fp4, q_scale, weights, _ = _written_cache_and_queries(
        batch, next_n, ctx_len, seed=1
    )
    counts = [4, 1, 3]
    seq = [b for b, c in enumerate(counts) for _ in range(c)]
    ends = [ctx_len - c + n + 1 for c in counts for n in range(c)]
    picks = [b * next_n + n for b, c in enumerate(counts) for n in range(c)]
    pad = 2
    rows = torch.tensor(picks + picks[:pad], device="cuda")
    starts = [0, *np.cumsum(counts).tolist(), sum(counts)]
    index_rows = fp4_rows(
        torch.tensor(ends + [0] * pad, dtype=torch.int32, device="cuda"),
        ctx_len,
        None,
        # The padding request's row too.
        Fp4MqaRaggedMetadata(
            torch.tensor(starts, dtype=torch.int32, device="cuda"),
            next_n,
            torch.cat([table, table[:1]]),
        ),
        heads=weights.shape[1],
        page_size=_BLOCK,
    )
    got = score_topk(
        _fp4_query(q_fp4[rows], q_scale[rows]),
        weights[rows].bfloat16(),
        IndexCache(kv_cache, kv_scale, _BLOCK),
        index_rows,
        topk=topk,
        weight_scale=WEIGHTS_SCALE,
    )

    want = _oracle(
        q_fp4[rows[: len(picks)]], q_scale[rows[: len(picks)]], kv_cache, kv_scale,
        table, ctx_len, weights[rows[: len(picks)]], lambda keys: keys[seq],
    )  # fmt: skip
    for r, end in enumerate(ends):
        chosen = set(got[r][got[r] >= 0].tolist())
        assert len(chosen) == min(topk, end) and max(chosen) < end, r
        best = set(want[r, :end].topk(min(topk, end)).indices.tolist())
        assert len(chosen & best) / len(best) > 0.99, r
    assert (got[len(picks) :] == -1).all()


def test_dcp_decode_scores_each_query_token_over_its_own_local_window(on_gfx950):
    """The geometry `dcp_decode_candidate_exchange_fused` hands the scorer: one
    row per query token over this rank's shard, each its own sequence, so the
    windows are ragged whatever the speculation width."""
    from aiter.ops.flydsl import flydsl_pa_mqa_logits_fp4

    from atom.model_ops.fp4_mqa_ragged_metadata import Fp4MqaRaggedMetadata

    batch, next_n, width = 3, 4, 1024
    table, kv_cache, kv_scale, q_fp4, q_scale, weights_out, rows = (
        _written_cache_and_queries(batch, next_n, width, seed=2)
    )

    # Ragged on purpose: a draft position's extra token lands on ONE rank, so
    # the local lengths of a request's next_n rows do not all advance together.
    local_ctx = torch.tensor(
        [width - (r % 7) * 37 for r in range(rows)], dtype=torch.int32, device="cuda"
    )

    logits = torch.empty(rows, width, dtype=torch.float32, device="cuda")
    flydsl_pa_mqa_logits_fp4(
        q_fp4.reshape(rows, 1, HEADS, HEAD_DIM // 2),
        q_scale.reshape(rows, 1, *q_scale.shape[1:]),
        kv_cache,
        kv_scale,
        weights=weights_out,
        max_seq_len=width,
        weight_scale=WEIGHTS_SCALE,
        kv_block_size=_BLOCK,
        out=logits,
        **Fp4MqaRaggedMetadata(
            torch.arange(rows + 1, dtype=torch.int32, device="cuda"),
            1,
            table.repeat_interleave(next_n, dim=0),
        ).kernel_args(local_ctx, heads=HEADS, page_size=_BLOCK, max_seq_len=width),
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
            dcp_indexer_gather_index=gather_index,
            # What the builder publishes, written out: one request, staged
            # from page 0.
            dcp_indexer_fp4_read_page=slots.long() // block,
            dcp_indexer_fp4_read_row=slots.long() % block,
            dcp_indexer_fp4_read_scale_row=_expect_scale_row(
                slots.long() % block, block
            ),
            dcp_indexer_fp4_stage_page=tok // block,
            dcp_indexer_fp4_stage_row=tok % block,
            dcp_indexer_fp4_stage_scale_row=_expect_scale_row(tok % block, block),
            dcp_indexer_fp4_stage_pages=-(-total_kv // block),
        ),
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
    """The builder's six staging index tensors, on the kernel and torch paths,
    and the staged pages' tables.

    Non-trivial block tables and mid-block sequence ends, so page, row and
    swizzled row all differ; on GPU this also pins outputs to their names.
    Each request's keys are staged from a page of their own, so its row of
    the staged tables pages them as its own block table does its cache --
    the scorer's columns are the request's own -- at the block tables' width,
    the tail zero rather than aliasing a real page.
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
    g_lens = np.array([100, 150], dtype=np.int64)
    g_cu = np.concatenate([[0], np.cumsum(g_lens)]).astype(np.int64)
    table = np.array([[7, 3, 0, 0], [11, 2, 9, 0]], dtype=np.int32)
    var = {"block_tables": SimpleNamespace(np=table)}
    meta = SimpleNamespace()
    build(builder, meta, bs, lpad, cu_pad, g_cu, g_lens, var)

    read = torch.tensor(
        [7 * block + j for j in range(block)]
        + [3 * block + j for j in range(5)]
        + [11 * block + j for j in range(block)]
        + [2 * block + j for j in range(block)]
        + [9 * block + j for j in range(3)]
    )
    # Request 0 over pages 0..1, request 1 from page 2.
    stage = torch.cat([torch.arange(100), 2 * block + torch.arange(150)])
    for side, src in (("read", read), ("stage", stage)):
        page = getattr(meta, f"dcp_indexer_fp4_{side}_page").cpu()
        row = getattr(meta, f"dcp_indexer_fp4_{side}_row").cpu()
        scale_row = getattr(meta, f"dcp_indexer_fp4_{side}_scale_row").cpu()
        assert page.dtype == row.dtype == scale_row.dtype == torch.int32, side
        assert torch.equal(page.long(), src // block), side
        assert torch.equal(row.long(), src % block), side
        assert torch.equal(
            scale_row.long(), _expect_scale_row(src % block, block)
        ), side
    assert meta.dcp_indexer_fp4_stage_pages == 5
    tables = meta.dcp_indexer_fp4_block_tables.cpu()
    assert tables.tolist() == [[0, 1, 0, 0, 0, 0], [2, 3, 4, 0, 0, 0]]


def _assert_picks(got, want, ends, topk):
    """Each row's selection: as many keys as it may keep, none past its
    bound, and the oracle's top-k but for near-ties."""
    for r, end in enumerate(ends.tolist()):
        k = min(topk, end)
        chosen = set(got[r][got[r] >= 0].tolist())
        assert len(chosen) == k and (not chosen or max(chosen) < end), r
        if k:
            best = set(want[r, :end].topk(k).indices.tolist())
            assert len(chosen & best) / k > 0.99, r


def _prefill_rows(batch, ctx_len, bands=None):
    """A ragged FP4 prefill over the writer's cache: one row per query token,
    each seeing `[0, its own position]` of its own sequence, a sequence's
    rows together -- the rows the MLA builder publishes."""
    from atom.model_ops.fp4_mqa_ragged_metadata import Fp4MqaRaggedMetadata
    from atom.model_ops.indexer import IndexCache, fp4_rows

    torch.manual_seed(1)
    table, num_blocks, token, seq, slots = _paged_layout(batch, ctx_len)
    q_fp4, q_scale, weights, kv_cache, kv_scale = _fused_fp4(
        slots, token, num_blocks, weight_gain=0.1
    )
    ends = (token + 1).to(torch.int32)
    starts = torch.arange(0, (batch + 1) * ctx_len, ctx_len, device="cuda")
    rows = fp4_rows(
        ends,
        ctx_len,
        None,
        Fp4MqaRaggedMetadata(starts.to(torch.int32), ctx_len, table),
        heads=HEADS,
        page_size=_BLOCK,
        bands=bands,
    )
    want = _oracle(
        q_fp4, q_scale, kv_cache, kv_scale, table, ctx_len, weights,
        lambda keys: keys[seq],
    )  # fmt: skip
    return (
        _fp4_query(q_fp4, q_scale),
        weights,
        IndexCache(kv_cache, kv_scale, _BLOCK),
        rows,
        ends,
        want,
    )


def test_prefill_scores_the_same_cache_seq_locally(on_gfx950):
    """A prefill scored ragged off the paged cache selects each row's top-k
    in its own sequence's columns."""
    from atom.model_ops.indexer import score_topk

    query, weights, cache, rows, ends, want = _prefill_rows(2, 512)
    got = score_topk(query, weights, cache, rows, topk=256, weight_scale=WEIGHTS_SCALE)
    _assert_picks(got, want, ends, topk=256)


@pytest.mark.parametrize("first", [256, 512, 768])
def test_prefill_band_selects_as_the_whole_batch(on_gfx950, first):
    """A band of rows -- a DCP rank's share, a sequence cut through -- selects
    exactly as those rows do in the whole batch: its launch counts sequences
    from its own first row, so a band planned wrong scores rows against
    another row's window. Same kernel on the same bytes, so any difference is
    an offset bug, not arithmetic."""
    from atom.model_ops.indexer import score_topk

    stride = 256
    query, weights, cache, whole, _, _ = _prefill_rows(2, 512)
    *_, band, _, _ = _prefill_rows(2, 512, bands=((first, first + stride),))
    kwargs = {"topk": 256, "weight_scale": WEIGHTS_SCALE, "stable": True}
    full = score_topk(query, weights, cache, whole, **kwargs)
    part = score_topk(query, weights, cache, band, **kwargs)
    span = slice(first, first + stride)
    assert torch.equal(part[span], full[span])


def _fp4_decode_scorer(monkeypatch, rows, width, seed, max_bs=None):
    """The gfx950 decode scorer over `rows` query rows, each its own KV
    sequence, on step rows (`IndexRows`) the test republishes over
    fixed-address buffers, as the V4 builder does. Batch buffers hold
    `max_bs` sequences, block-table rows past the real ones zero (as
    `prepare_decode` publishes them out to `running_bs`)."""
    from atom.model_ops.fp4_mqa_ragged_metadata import Fp4MqaRaggedMetadata
    from atom.model_ops.indexer import IndexCache, fp4_rows, score_topk
    from atom.model_ops.sparse_indexer_chunk import LogitsWorkspace
    from atom.utils import forward_context

    table, kv, ks, q, qs, weights, _ = _written_cache_and_queries(
        rows, 1, width, seed=seed
    )
    weights = weights.bfloat16()
    query = _fp4_query(q, qs)
    cache = IndexCache(kv, ks, _BLOCK)
    workspace = LogitsWorkspace.for_rows(rows, width, "cuda")
    max_bs = max(rows, max_bs or 0)
    padded_table = torch.zeros(
        max_bs, table.shape[1], dtype=table.dtype, device=table.device
    )
    padded_table[:rows] = table
    cu_buf = torch.zeros(max_bs + 1, dtype=torch.int32, device="cuda")
    ends = torch.zeros(rows, dtype=torch.int32, device="cuda")
    metadata = SimpleNamespace(index_rows=None)

    class StepContext:
        """The step a region reads at call time: its rows as last published,
        on the stream the caller issues on, outside the capture loop."""

        in_hipgraph = False

        @property
        def attn_metadata(self):
            return metadata

        @property
        def main_stream(self):
            return torch.cuda.current_stream()

    monkeypatch.setattr(forward_context, "get_forward_context", StepContext)

    def publish(counts, running_bs, max_q):
        """`counts` real sequences padded with empty ones to `running_bs`, as
        `prepare_cu_seqlens_q` publishes them; per-row bounds zero on the pad
        rows. A new metadata object each step, as the builder makes one."""
        nonlocal metadata
        cu = np.full(running_bs + 1, sum(counts), dtype=np.int32)
        cu[0] = 0
        cu[1 : len(counts) + 1] = np.cumsum(counts)
        cu_buf[: running_bs + 1].copy_(torch.from_numpy(cu))
        bounds = [width - c + i + 1 for c in counts for i in range(c)]
        bounds += [0] * (rows - len(bounds))
        ends.copy_(torch.tensor(bounds, dtype=torch.int32))
        metadata = SimpleNamespace(
            index_rows=fp4_rows(
                ends,
                width,
                workspace,
                Fp4MqaRaggedMetadata(
                    cu_buf[: running_bs + 1], max_q, padded_table[:running_bs]
                ),
                heads=weights.shape[1],
                page_size=_BLOCK,
            )
        )
        return bounds

    def run():
        return score_topk(
            query,
            weights,
            cache,
            metadata.index_rows,
            topk=512,
            weight_scale=WEIGHTS_SCALE,
        )

    def check(got, counts, bounds):
        n_real = sum(counts)
        seq = [b for b, count in enumerate(counts) for _ in range(count)]
        if n_real:
            want = _oracle(
                q[:n_real],
                qs[:n_real],
                kv,
                ks,
                table[: len(counts)],
                width,
                weights[:n_real],
                lambda keys, seq=seq: keys[seq],
            )
            for row, end in enumerate(bounds[:n_real]):
                chosen = set(got[row][got[row] >= 0].tolist())
                best = set(want[row, :end].topk(512).indices.tolist())
                assert len(chosen) == 512 and max(chosen) < end
                assert len(chosen & best) / 512 > 0.99
        assert bool((got[n_real:] == -1).all())

    return publish, run, check


@pytest.mark.parametrize("running_bs,max_q", [(8, 1), (16, 4), (8, 8)])
def test_v4_fp4_decode_graph_replays_every_layout_of_its_batch(
    on_gfx950, monkeypatch, running_bs, max_q
):
    """The row-group launch is planned from the sequence count and the query
    span, so a captured scorer is valid for every step sharing both -- the key
    a request bucket (`ReqGraph`) uses: rectangular, ragged, DP/CG-padded with
    empty sequences, and all-empty."""
    from atom.model_ops.indexer import score as shared

    rows, width = running_bs * max_q, 768
    publish, run, check = _fp4_decode_scorer(monkeypatch, rows, width, seed=17)
    ragged = [max_q, 1] * (running_bs // 2)
    shapes = [
        [max_q] * running_bs,
        [max(1, max_q - i % max_q) for i in range(running_bs)],
        ragged,
        [max_q] * (running_bs // 2),
        [1],
        [],
    ]
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        publish(shapes[0], running_bs, max_q)
        for _ in range(3):
            run()
        stream.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            got = run()
        stream.synchronize()

        def no_relaunch(*args, **kwargs):
            raise AssertionError("scorer was relaunched instead of replayed")

        monkeypatch.setattr(shared, "flydsl_pa_mqa_logits_fp4", no_relaunch)
        for counts in shapes[::-1] + shapes:
            bounds = publish(counts, running_bs, max_q)
            graph.replay()
            stream.synchronize()
            check(got, counts, bounds)


def test_v4_fp4_decode_scores_a_dp_padded_batch_wider_than_its_rows(
    on_gfx950, monkeypatch
):
    """Under DP a prefilling peer keeps `running_bs` at the group's batch while
    this rank runs only its own rows, so `cu_seqlens_q` carries more sequences
    (empty ones) than the step has rows: an idle rank's one-row dummy against a
    16-request peer. The scorer takes them as they are."""
    publish, run, check = _fp4_decode_scorer(monkeypatch, 4, 768, seed=23, max_bs=16)
    for counts in ([1], [1, 1, 1]):
        bounds = publish(counts, 16, 1)
        torch.cuda.synchronize()
        check(run(), counts, bounds)


def test_v4_fp4_decode_token_graph_serves_every_batch_of_its_rows(
    on_gfx950, monkeypatch
):
    """A token graph records the scorer once for its rows and breaks at the
    logits (`_fp4_band_logits`), whose launch is planned from the live
    step's sequences. So the one recording serves every batch those rows can
    hold -- any sequence count, any query span -- while the top-k replays
    from the graph."""
    from atom.model_ops.indexer import score as shared
    from atom.utils.region_graph import _record_segments

    rows = 32
    publish, run, check = _fp4_decode_scorer(
        monkeypatch, rows, 768, seed=29, max_bs=rows
    )
    layouts = [
        ([4] * 8, 8, 4),
        ([1] * 32, 32, 1),
        ([2, 1] * 8, 16, 2),
        ([3, 1] * 4, 8, 3),
        ([], 8, 4),
    ]
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        publish(*layouts[0])
        for _ in range(3):
            run()
        stream.synchronize()
        segments, got = _record_segments(run)
        stream.synchronize()
        assert segments.num_segments == 2

        def no_relaunch(*args, **kwargs):
            raise AssertionError("top-k was relaunched instead of replayed")

        monkeypatch.setattr(shared, "top_k_per_row_decode", no_relaunch)
        for counts, running_bs, max_q in layouts[::-1] + layouts:
            bounds = publish(counts, running_bs, max_q)
            segments.replay()
            stream.synchronize()
            check(got, counts, bounds)


def _qsplit_roundtrip(full_topk, world_size, sentinel=-7):
    """Replay what the DCP prefill path does to a selection, on one process.

    Modelled the way the path actually runs it: ONE buffer of
    ``world_size`` strides, each rank writing the rows it scored into its own
    slot, which is what the in-place all-gather makes the ranks agree on. The
    first ``num_rows`` rows are then the result. Everything else is filled with
    a sentinel, so a failure tells us whether an unwritten row leaked into the
    result rather than just that something moved.
    """
    num_rows, topk = full_topk.shape
    stride = dcp_qsplit_rows(num_rows, 0, world_size)[0]
    gathered = torch.full((stride * world_size, topk), sentinel, dtype=full_topk.dtype)
    for rank in range(world_size):
        rows_per_rank, slot_lo, row_lo, row_hi = dcp_qsplit_rows(
            num_rows, rank, world_size
        )
        assert rows_per_rank == stride
        gathered[slot_lo : slot_lo + (row_hi - row_lo)] = full_topk[row_lo:row_hi]
    return gathered[:num_rows]


@pytest.mark.parametrize("world_size", [4, 8])
@pytest.mark.parametrize(
    "num_rows",
    [
        4096,  # divides evenly
        4097,  # one row over -- the last rank carries a single row
        4095,  # one row under -- the last rank's tail is padding
        12,  # small but still even
        7,  # fewer rows than ranks at W=8, uneven at W=4
        3,  # fewer rows than ranks at both widths
        1,  # one row, every other rank is empty
    ],
)
def test_qsplit_reassembles_the_unsplit_selection_exactly(num_rows, world_size):
    """Reassembly must return the selection the unsplit path would.

    Splitting the QUERY rows leaves each rank scoring its rows against the whole
    key set, so its top-k is the global one; what remains between "scored" and
    "used" is the partition and the reassembly, which is what this pins.
    Compared element-wise against the unsplit tensor, not against a tolerance --
    there is no arithmetic here to be approximate about.

    The scorer under a restricted row range is the other half, and it needs a
    GPU: see test_prefill_scores_a_restricted_row_range_the_same_as_the_whole_batch.
    """
    torch.manual_seed(num_rows * 100 + world_size)
    full = torch.randint(0, 1 << 20, (num_rows, 2048), dtype=torch.int32)
    out = _qsplit_roundtrip(full, world_size)
    assert out.shape == full.shape
    assert torch.equal(out, full)


@pytest.mark.parametrize("world_size", [1, 2, 4, 8])
@pytest.mark.parametrize("num_rows", [1, 3, 7, 12, 4095, 4096, 4097])
def test_qsplit_rows_partition_covers_every_row_once(num_rows, world_size):
    """Every row belongs to exactly one rank, and the padding is bounded.

    The reassembly above would also pass if two ranks claimed the same row and
    a third claimed none, as long as the concatenation happened to line up, so
    pin the partition itself: the slices tile [0, num_rows) in order, the
    stride is uniform, and the gathered payload exceeds num_rows by less than
    world_size rows (this is the padding the all-gather pays for).
    """
    seen = []
    strides = set()
    slots = []
    for rank in range(world_size):
        rows_per_rank, slot_lo, row_lo, row_hi = dcp_qsplit_rows(
            num_rows, rank, world_size
        )
        strides.add(rows_per_rank)
        assert row_lo <= row_hi <= num_rows
        assert row_hi - row_lo <= rows_per_rank
        # The slot is where the in-place all-gather reads this rank's input
        # from, so it must be the rank's OWN stride -- including for a rank
        # with no rows, whose clamped row range sits in a lower rank's slot.
        assert slot_lo == rank * rows_per_rank
        # A rank with rows scores at the head of its own slot; one without
        # rows has a clamped, empty range that must NOT be read as a slot.
        assert row_lo == row_hi or (
            row_lo == slot_lo and row_hi <= slot_lo + rows_per_rank
        )
        slots.append((slot_lo, slot_lo + rows_per_rank))
        seen.extend(range(row_lo, row_hi))
    assert seen == list(range(num_rows))
    assert len(strides) == 1
    stride = strides.pop()
    gathered_rows = stride * world_size
    assert num_rows <= gathered_rows < num_rows + world_size
    # The slots tile the gathered buffer exactly, in rank order: no two ranks
    # read from the same rows, and the last slot ends at the buffer's end.
    assert slots == [(r * stride, (r + 1) * stride) for r in range(world_size)]


def test_qsplit_degenerates_to_the_whole_range_without_dcp():
    """At dcp=1 the rank's slice is every row, i.e. the pre-DCP behaviour."""
    assert dcp_qsplit_rows(4096, 0, 1) == (4096, 0, 0, 4096)
