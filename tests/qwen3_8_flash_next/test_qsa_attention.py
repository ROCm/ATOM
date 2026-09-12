"""QSA attention must equal dense causal attention while the budget covers it.

Below `indexer_budget` tokens of context, the indexer's top-k selects every
causally visible group, so the sparse kernel has no approximation left to make:
its output must match plain causal GQA over the same q/k/v. That makes dense
attention an exact oracle for the whole QSA path -- projections, the three
paged caches, group pooling, scoring, selection, and the sparse GQA kernel --
without needing the reference engine.

The reference for q/k/v themselves is the module's own projections, so this
test does NOT check the checkpoint layout (`test_indexer_parity` and the
weight-loading path cover that); it checks that everything downstream of the
projections computes causal attention.
"""

import math
from types import SimpleNamespace

import pytest
import torch

from tests.qwen3_8_flash_next.parity_harness import init_single_rank

HIDDEN = 2560
NUM_HEADS = 24
NUM_KV_HEADS = 2
HEAD_DIM = 256
BLOCK_SIZE = 64
# The scoring kernel needs at least `budget / compress_ratio` = 512
# compressed columns to select from, i.e. >= 32 blocks of 16 rows.
NUM_BLOCKS = 40
DTYPE = torch.bfloat16


def _config():
    return SimpleNamespace(
        hidden_size=HIDDEN,
        num_attention_heads=NUM_HEADS,
        num_key_value_heads=NUM_KV_HEADS,
        head_dim=HEAD_DIM,
        rms_norm_eps=1e-6,
        max_position_embeddings=8192,
        # The checkpoint's real rope: mRoPE, which with three equal position
        # rows must still reduce exactly to 1D RoPE.
        rope_parameters={
            "rope_theta": 10_000_000,
            "partial_rotary_factor": 0.25,
            "rope_type": "default",
            "mrope_section": [11, 11, 10],
            "mrope_interleaved": True,
        },
        indexer_n_heads=4,
        indexer_kv_heads=1,
        indexer_head_dim=128,
        indexer_budget=2048,
        indexer_compress_ratio=4,
    )


def _metadata(num_tokens: int, device):
    """One request occupying logical positions 0..num_tokens-1 of block 0..N."""
    from atom.model_ops.attentions.qwen3_8_flash_next_attn import Qwen3_8FlashNextQSAMetadata

    positions = torch.arange(num_tokens, device=device, dtype=torch.int64)
    block_tables = torch.arange(
        NUM_BLOCKS, device=device, dtype=torch.int32
    ).unsqueeze(0)
    # block_ratio == 1, so a slot is just the flat token index.
    slot_mapping = positions.clone()
    compressed_positions = positions // 4
    storage_block = BLOCK_SIZE // 4
    compressed_slots = torch.where(
        (positions + 1) % 4 == 0,
        (compressed_positions // storage_block) * storage_block
        + compressed_positions % storage_block,
        torch.full_like(positions, -1),
    )
    return Qwen3_8FlashNextQSAMetadata(
        block_tables=block_tables,
        slot_mapping=slot_mapping,
        compressed_slot_mapping=compressed_slots,
        token_to_req=torch.zeros(num_tokens, device=device, dtype=torch.int32),
        logical_positions=positions,
        seq_lens=torch.tensor([num_tokens], device=device, dtype=torch.int32),
        max_seq_len=num_tokens,
    )


@pytest.fixture(scope="module")
def layer():
    init_single_rank()
    from atom.model_ops.qwen3_8_flash_next.qsa_attention import Qwen3_8FlashNextAttention

    atom_config = SimpleNamespace(max_num_batched_tokens=1024)
    module = (
        Qwen3_8FlashNextAttention(_config(), atom_config, prefix="self_attn").cuda().to(DTYPE)
    )
    torch.manual_seed(0)
    with torch.no_grad():
        for weight in (
            module.qkv_proj.weight,
            module.o_proj.weight,
            module.indexer.index_qk_proj.weight,
        ):
            weight.copy_(torch.randn_like(weight.float()).to(DTYPE) * 0.02)
        for norm in (
            module.q_norm,
            module.k_norm,
            module.indexer.q_layernorm,
            module.indexer.k_layernorm,
        ):
            norm.weight.copy_(torch.randn_like(norm.weight.float()).to(DTYPE) * 0.1)

    module.bind_caches(
        torch.zeros(
            NUM_BLOCKS, BLOCK_SIZE, NUM_KV_HEADS, HEAD_DIM, dtype=DTYPE, device="cuda"
        ),
        torch.zeros(
            NUM_BLOCKS, BLOCK_SIZE, NUM_KV_HEADS, HEAD_DIM, dtype=DTYPE, device="cuda"
        ),
        torch.zeros(NUM_BLOCKS, BLOCK_SIZE, 1, 128, dtype=DTYPE, device="cuda"),
        torch.zeros(NUM_BLOCKS, BLOCK_SIZE // 4, 1, 128, dtype=DTYPE, device="cuda"),
    )
    return module


def _dense_reference(module, hidden_states, positions):
    """Causal GQA over the module's own q/k/v, in float32."""
    num_tokens = hidden_states.shape[0]
    q_size = NUM_HEADS * HEAD_DIM
    kv_size = NUM_KV_HEADS * HEAD_DIM
    qkv = module.qkv_proj(hidden_states)
    gate, q, k, v = torch.split(qkv, [q_size, q_size, kv_size, kv_size], dim=-1)
    q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
    q = module.q_norm(q.view(num_tokens, NUM_HEADS, HEAD_DIM)).view(-1, q_size)
    k = module.k_norm(k.view(num_tokens, NUM_KV_HEADS, HEAD_DIM)).view(-1, kv_size)
    q, k = module.rotary_emb(positions, q, k)

    query = q.view(num_tokens, NUM_HEADS, HEAD_DIM).float()
    key = k.view(num_tokens, NUM_KV_HEADS, HEAD_DIM).float()
    value = v.view(num_tokens, NUM_KV_HEADS, HEAD_DIM).float()
    group = NUM_HEADS // NUM_KV_HEADS
    key = key.repeat_interleave(group, dim=1)
    value = value.repeat_interleave(group, dim=1)

    scores = torch.einsum("qhd,khd->hqk", query, key) / math.sqrt(HEAD_DIM)
    mask = torch.ones(num_tokens, num_tokens, device=scores.device, dtype=torch.bool)
    scores = scores.masked_fill(~torch.tril(mask), float("-inf"))
    out = torch.einsum("hqk,khd->qhd", scores.softmax(dim=-1), value)
    gated = out.reshape(num_tokens, -1).to(DTYPE) * torch.sigmoid(gate)
    return module.o_proj(gated)


@pytest.mark.parametrize("num_tokens", [1, 7, 64, 130])
def test_matches_dense_attention_within_budget(layer, num_tokens):
    torch.manual_seed(num_tokens)
    hidden = (torch.randn(num_tokens, HIDDEN) * 0.5).cuda().to(DTYPE)
    positions = torch.arange(num_tokens, device="cuda")

    from atom.model_ops.qwen3_8_flash_next import qsa_attention

    metadata = SimpleNamespace(qsa_metadata=_metadata(num_tokens, hidden.device))
    context = SimpleNamespace(attn_metadata=metadata)
    original = qsa_attention.get_forward_context
    qsa_attention.get_forward_context = lambda: context
    try:
        with torch.no_grad():
            expected = _dense_reference(layer, hidden, positions)
            got = layer(positions, hidden)
    finally:
        qsa_attention.get_forward_context = original

    torch.testing.assert_close(got.float(), expected.float(), rtol=2e-2, atol=2e-2)
