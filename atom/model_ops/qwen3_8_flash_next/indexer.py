"""Qwen3.8-Flash-Next QSA indexer: project index Q/K and score compressed key groups.

Port of `qwen3_8_flash_next/nvidia/indexer_qsa.py:QSAIndexer`.

Only 12 of the 48 layers use full attention, and each of those replaces dense
attention with QSA: an auxiliary 4-head MQA "indexer" scores mean-pooled groups
of `compress_ratio` past keys, keeps the best `budget / compress_ratio` groups,
and expands them back to at most `budget + compress_ratio - 1` logical token
positions for the real attention to read.

Two orderings here are easy to get backwards and produce plausible-but-wrong
numbers rather than errors:

  * Q is normalized (Gemma RMSNorm, `x * (1 + w)`) and rotated at the token's
    own position.
  * K is kept RAW through projection. It is mean-pooled over each complete
    group FIRST, and only the pooled key is normalized and rotated -- at the
    position of the group's FIRST token, not the last.

The projection is replicated rather than TP-sharded: there are only 4 index
heads, and every rank has to reach the same selection anyway.
"""

import torch
from torch import nn

from atom.model_ops.layernorm import GemmaRMSNorm
from atom.model_ops.linear import ReplicatedLinear


def apply_qsa_rope(
    rotary_emb: nn.Module,
    positions: torch.Tensor,
    tensor: torch.Tensor,
) -> torch.Tensor:
    """Rotate `[tokens, heads, head_dim]` at `positions`, tail passed through.

    `rotary_emb` must be an `MRotaryEmbedding` built for the INDEXER's head
    size, not the attention one: it reshapes by `head_size`, and the two differ
    here (128 vs 256). The cos/sin cache depends only on `rotary_dim`, so the
    two instances stay in exact agreement.

    `positions` is `[tokens]` for text or `[3, tokens]` for mRoPE; the same
    call covers both, because with three equal rows mRoPE and 1D RoPE compute
    the same rotation whichever section a frequency pair belongs to.
    """
    shape = tensor.shape
    flat = tensor.reshape(shape[0], -1)
    # MRotaryEmbedding rotates a query and a key together; the indexer has only
    # one tensor per call, so a one-head scratch stands in for the other.
    scratch = flat.new_zeros((shape[0], rotary_emb.head_size))
    rotated, _ = rotary_emb(positions, flat, scratch)
    return rotated.reshape(shape)


class Qwen3_8FlashNextIndexer(nn.Module):
    """Replicated index Q/K projection plus the QSA selection contract."""

    def __init__(
        self,
        config,
        rotary_emb: nn.Module | None = None,
        quant_config=None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.index_n_heads = int(config.indexer_n_heads)
        self.index_kv_heads = int(config.indexer_kv_heads)
        self.index_head_dim = int(config.indexer_head_dim)
        self.token_topk = int(config.indexer_budget)
        self.compress_ratio = int(config.indexer_compress_ratio)
        if self.index_kv_heads != 1:
            raise ValueError("the QSA MQA operators require indexer_kv_heads=1")
        if self.token_topk % self.compress_ratio:
            raise ValueError("indexer_budget must divide by indexer_compress_ratio")
        self.rotary_emb = rotary_emb
        self.prefix = prefix

        eps = float(getattr(config, "rms_norm_eps", 1e-6))
        self.index_qk_proj = ReplicatedLinear(
            int(config.hidden_size),
            (self.index_n_heads + self.index_kv_heads) * self.index_head_dim,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.index_qk_proj" if prefix else "index_qk_proj",
        )
        self.q_layernorm = GemmaRMSNorm(self.index_head_dim, eps=eps)
        self.k_layernorm = GemmaRMSNorm(self.index_head_dim, eps=eps)

    @property
    def block_topk(self) -> int:
        """Compressed groups kept per query."""
        return self.token_topk // self.compress_ratio

    @property
    def output_width(self) -> int:
        """Logical token ids emitted: full groups plus the causal tail."""
        return self.token_topk + self.compress_ratio - 1

    def project_qk(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Normalized+rotated index Q, and RAW index K for later pooling."""
        otype = hidden_states.dtype
        qk = self.index_qk_proj(hidden_states, otype=otype)
        q_raw, token_k = qk.split(
            (
                self.index_n_heads * self.index_head_dim,
                self.index_kv_heads * self.index_head_dim,
            ),
            dim=-1,
        )
        q = q_raw.reshape(-1, self.index_n_heads, self.index_head_dim)
        q = self.q_layernorm(q.reshape(-1, self.index_head_dim)).reshape_as(q)
        q = apply_qsa_rope(self.rotary_emb, positions, q)
        return q, token_k.reshape(-1, 1, self.index_head_dim)

    def normalize_compressed_keys(
        self,
        compressed_keys: torch.Tensor,
        first_rope_positions: torch.Tensor,
    ) -> torch.Tensor:
        """Normalize pooled K and rotate it at its group's first-token position."""
        keys = compressed_keys.reshape(-1, self.index_head_dim)
        keys = self.k_layernorm(keys).reshape(-1, 1, self.index_head_dim)
        if getattr(self.rotary_emb, "mrope_section", None):
            # `[groups, 3]` -> the `[3, groups]` row layout mRoPE expects.
            positions = first_rope_positions.transpose(0, 1).contiguous()
        else:
            positions = first_rope_positions[:, 0]
        return apply_qsa_rope(self.rotary_emb, positions, keys)

    @staticmethod
    def pool_key_groups(raw_keys: torch.Tensor, compress_ratio: int) -> torch.Tensor:
        """Mean-pool complete groups of `compress_ratio` raw keys.

        Only whole groups get a compressed entry; the ragged tail stays as
        individual tokens and is appended during index expansion.
        """
        groups = raw_keys.shape[0] // compress_ratio
        if groups == 0:
            return raw_keys.new_zeros((0, *raw_keys.shape[1:]))
        head = raw_keys[: groups * compress_ratio]
        return head.unflatten(0, (groups, compress_ratio)).mean(dim=1)
