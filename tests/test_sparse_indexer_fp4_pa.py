"""The FP4 sparse indexer over the bytes the production path actually writes.

`indexer_qk_rope_quant_and_cache` in FP4 mode is the only writer of the packed
E2M1 Q/K and their e8m0 planes, and `flydsl_pa_mqa_logits_fp4[_prefill]` the
only readers, so what is worth checking is that the two agree on a real DSA
indexer's shapes (H=32, D=128, kv_block=64, block_k=256) -- decode at both
next_n GLM-5.2 runs, and ragged prefill. The reference dequantizes exactly what
the writer produced, so a disagreement is a layout bug, not rounding.
"""

from types import SimpleNamespace

import pytest
import torch

from atom.model_ops.sparse_indexer_fp4 import (
    FP4_KV_BLOCK_SIZE,
    FP4_MQA_BLOCK_K,
    FP4_QUANT_BLOCK_SIZE,
    fp4_prefill_schedule,
    fp4_q_scale_shape,
)

HEADS = 32
HEAD_DIM = 128
ROPE_DIM = 64
MAX_POSITION = 4096
EPSILON = 1e-6
WEIGHTS_SCALE = HEAD_DIM**-0.5 * HEADS**-0.5
GROUPS = HEAD_DIM // FP4_QUANT_BLOCK_SIZE
_BLOCK = FP4_KV_BLOCK_SIZE

_E2M1_MAG = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
_E2M1 = torch.cat([_E2M1_MAG, -_E2M1_MAG])


@pytest.fixture(autouse=True)
def _gfx950_single_rank(monkeypatch):
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


def _unshuffle_q_scale(q_scale: torch.Tensor) -> torch.Tensor:
    """`[T, k_tiles, 4, 16, qs_pad]` -> dense `[T, H, D // 32]`."""
    m_tiles = HEADS // 16
    return (
        q_scale[..., :m_tiles]
        .permute(0, 4, 3, 1, 2)
        .reshape(q_scale.shape[0], HEADS, GROUPS)
        .contiguous()
    )


def _rope_caches():
    angles = torch.randn(MAX_POSITION, ROPE_DIM // 2, device="cuda")
    return angles.cos().bfloat16(), angles.sin().bfloat16()


def _paged_layout(batch: int, ctx_len: int):
    """A shuffled block table plus the slot of every KV token."""
    blocks_per_seq = ctx_len // _BLOCK
    num_blocks = batch * blocks_per_seq
    block_tables = torch.randperm(num_blocks, device="cuda").to(torch.int32)
    block_tables = block_tables.reshape(batch, blocks_per_seq)
    token = torch.arange(ctx_len, device="cuda").repeat(batch)
    seq = torch.arange(batch, device="cuda").repeat_interleave(ctx_len)
    slots = block_tables[seq, token // _BLOCK].long() * _BLOCK + token % _BLOCK
    return block_tables, num_blocks, token, seq, slots


def _fused_fp4(slots, positions, cos, sin, num_blocks, weight_gain=1.0):
    """The production writer, in FP4 mode. Returns everything it emits.

    Through ATOM's own shim rather than `aiter.` directly: the shim is what
    decides `compute_all_q_rope` and forwards the two scale buffers, so it is
    the seam worth covering.
    """
    from atom.model_ops import attention_mla

    rows = slots.shape[0]
    u8 = {"dtype": torch.uint8, "device": "cuda"}
    bf16 = {"dtype": torch.bfloat16, "device": "cuda"}
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
        torch.randn(HEAD_DIM, dtype=torch.float32, device="cuda"),
        torch.randn(HEAD_DIM, dtype=torch.float32, device="cuda"),
        positions,
        cos,
        sin,
        EPSILON,
        FP4_QUANT_BLOCK_SIZE,
        "ue8m0",
        WEIGHTS_SCALE,
        is_neox=True,
        q_scale_out=q_scale,
        kv_cache_scale=kv_scale,
    )
    return q_fp4, q_scale, weights_out, kv_cache, kv_scale


def _gathered_keys(kv_cache, kv_scale, block_tables, ctx_len):
    """The FP4 cache read back per sequence as fp32 `[B, ctx_len, D]`."""
    batch = block_tables.shape[0]
    token = torch.arange(ctx_len, device=kv_cache.device)
    phys = block_tables[:, token // _BLOCK].long().unsqueeze(-1)  # [B, T, 1]
    pos = (token % _BLOCK).expand(batch, ctx_len).unsqueeze(-1)  # [B, T, 1]
    group = torch.arange(4, device=kv_cache.device)  # -> [B, T, 4]
    packed = kv_cache[phys, 0, group, pos].reshape(batch, ctx_len, HEAD_DIM // 2)
    e8m0 = kv_scale[phys, 0, group, (pos % 16) * 4 + pos // 16]
    return _dequant(packed, e8m0)


def _oracle(q_fp4, q_scale, keys, weights, width):
    """The scorer's math in fp32: ReLU(q.k) weighted per head, then summed."""
    q_dense = _dequant(q_fp4, _unshuffle_q_scale(q_scale))
    scores = torch.einsum("rhd,rtd->rht", q_dense, keys)
    out = (torch.relu(scores) * weights.float().unsqueeze(-1)).sum(1) * WEIGHTS_SCALE
    return out[:, :width]


def _agreement(got, want, visible, topk):
    """Cosine over the visible window and the worst row's top-k overlap -- the
    two numbers that say whether the selection this feeds would differ."""
    mask = torch.arange(want.shape[1], device=want.device)[None, :] < visible[:, None]
    a, b = got[mask].double(), want[mask].double()
    cosine = (a @ b / (a.norm() * b.norm())).item()
    overlaps = []
    for row in range(want.shape[0]):
        n = int(visible[row])
        k = min(topk, n)
        if k:
            got_k = set(got[row, :n].topk(k).indices.tolist())
            want_k = set(want[row, :n].topk(k).indices.tolist())
            overlaps.append(len(got_k & want_k) / k)
    return cosine, min(overlaps)


@pytest.mark.parametrize(("batch", "next_n", "ctx_len"), [(3, 1, 1024), (2, 4, 768)])
def test_decode_scores_the_cache_the_fused_writer_wrote(batch, next_n, ctx_len):
    from aiter.ops.flydsl import flydsl_pa_mqa_logits_fp4
    from aiter.ops.flydsl.kernels.mqa_logits.pa_mqa_logits_fp4 import (
        compute_varctx_schedule,
    )

    torch.manual_seed(0)
    cos, sin = _rope_caches()
    block_tables, num_blocks, token, _, slots = _paged_layout(batch, ctx_len)
    *_, kv_cache, kv_scale = _fused_fp4(slots, token, cos, sin, num_blocks)

    # The query rows get their own throwaway cache, so their slots are real --
    # which is what a decode step passes, and what makes the writer compute Q.
    rows = batch * next_n
    q_fp4, q_scale, weights_out, *_ = _fused_fp4(
        torch.arange(rows, dtype=torch.int64, device="cuda"),
        torch.full((rows,), ctx_len - 1, dtype=torch.int64, device="cuda"),
        cos,
        sin,
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
        block_tables,
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

    keys = _gathered_keys(kv_cache, kv_scale, block_tables, ctx_len)
    want = _oracle(
        q_fp4, q_scale, keys.repeat_interleave(next_n, dim=0), weights_out, ctx_len
    )
    # Each of a request's next_n rows sees one token less than the one after it.
    row = torch.arange(rows, device="cuda")
    visible = ctx_lens.repeat_interleave(next_n) - (next_n - 1 - row % next_n)

    cosine, overlap = _agreement(logits, want, visible, topk=512)
    assert cosine > 0.9999, cosine
    assert overlap > 0.99, overlap


@pytest.mark.parametrize("whole_batch", [True, False])
def test_prefill_scores_the_same_cache_seq_locally(whole_batch):
    from atom.models.deepseek_v2 import _prefill_mqa_logits_fp4

    torch.manual_seed(1)
    batch, ctx_len = 2, 512
    cos, sin = _rope_caches()
    block_tables, num_blocks, token, seq, slots = _paged_layout(batch, ctx_len)
    q_fp4, q_scale, weights_out, kv_cache, kv_scale = _fused_fp4(
        slots, token, cos, sin, num_blocks, weight_gain=0.1
    )

    # One row per query token, each seeing `[0, its own position]` of its own
    # sequence -- the seq-local windows the metadata builder publishes.
    rows = batch * ctx_len
    local_ends = (token + 1).to(torch.int32)
    row_to_batch = seq.to(torch.int32)
    cta_info, n_ctas, local_starts = fp4_prefill_schedule(
        row_to_batch, local_ends, FP4_MQA_BLOCK_K, rows, ctx_len
    )
    # With `whole_batch=False` the model rebuilds the schedule instead of
    # reusing the one the builder left here.
    logits = _prefill_mqa_logits_fp4(
        SimpleNamespace(
            batch_id_per_q_token=row_to_batch,
            block_tables=block_tables,
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
    )

    keys = _gathered_keys(kv_cache, kv_scale, block_tables, ctx_len)
    want = _oracle(q_fp4, q_scale, keys[row_to_batch.long()], weights_out, ctx_len)

    cosine, overlap = _agreement(logits, want, local_ends, topk=256)
    assert cosine > 0.9999, cosine
    assert overlap > 0.99, overlap
