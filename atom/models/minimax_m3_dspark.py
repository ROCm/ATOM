# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""MiniMax-M3.1 DSpark semi-autoregressive block drafter for ATOM.

The same algorithm as the Kimi-K3 draft (``kimi_k3_dspark.py``) -- the target's
aux hidden states are projected once into a context, every draft layer writes
its own K/V rows for that context into a paged cache, and one non-causal block
pass over ``[anchor, MASK, ...]`` drafts the whole block before a Markov head
samples it left to right -- on a MiniMax-M3 backbone instead of MLA. The
reference is NeMo AutoModel ``speculative/dspark/draft_minimax_m3.py``:

  * Attention is plain GQA (64 q / 4 kv heads, head_dim 128) with per-head
    Gemma RMSNorm on Q and K, partial neox RoPE on the first 64 lanes, and no
    norm or RoPE on V. The K/V of the context and of the block are projected
    separately and attended together, non-causally: every block position sees
    the whole context and the whole block.
  * Every norm is Gemma style, ``x * (1 + w)``; the MLP is dense SwiGLU-OAI.
    There is no indexer and no MoE -- both belong to the target.
  * The context is ``hidden_norm(fc(concat(aux)))`` and goes straight into each
    layer's K/V projection, never through ``input_layernorm``.

KV lives in a sibling pool of its own (an MHA draft cannot share the target's
latent pool the way K3's MLA draft does): ``draft_kv_builder`` resolves this
config to the MHA backend, which owns a pool laid out like the target's and
binds it onto each layer's ``Attention``. Both writes -- context rows and the
block's own rows -- go through ``triton_fused_norm_rope_cache`` with an explicit
slot mapping, the kernel the target's dense layers use for the same norm, RoPE
and per-token fp8 cache format.

The block attention, with one KV head per rank (TP >= 4), is ``B * T``
independent q_len=1 paged decode rows -- non-causal by construction -- over each
request's ``[context | block]``, through the per-row ASM / Gluon decode the
target's sparse layers run and a page-16 table ``DSparkProposer`` builds outside
the graph; the block's own K/V are written into the cache first, except at
positions past the request's allocated pages. No host sync and fixed shapes, so
the block pass is CUDA-graph captured like the other drafts. Other TP sizes fall
back to an eager path: gather the context, attend it together with the block's
K/V straight from the pass with one non-causal varlen attention.

Checkpoint layout (``<draft>/dspark``, MXFP8 [1, 32] linears, BF16 rest):
  language_model.model.dspark.fc.weight                       [6144, 5 * 6144]
  language_model.model.dspark.{hidden_norm,final_norm}.weight [6144]
  language_model.model.dspark.layers.{0..4}.decoder_layer.
      {input_layernorm,post_attention_layernorm}.weight
      self_attn.{q,k,v,o}_proj.weight(_scale_inv)             -> qkv_proj
      self_attn.{q,k}_norm.weight                             [128]
      mlp.{gate,up,down}_proj.weight(_scale_inv)              -> gate_up_proj
  language_model.model.dspark.markov_head.markov_w{1,2}.weight [200064, 256]
  language_model.model.dspark.confidence_head.proj.{weight,bias} [1, 6400]
The draft ships no embedding or LM head; both are the target's.
"""

from typing import TYPE_CHECKING, ClassVar

import torch
from aiter import QuantType, dtypes
from aiter.dist.parallel_state import get_tensor_model_parallel_world_size
from aiter.rotary_embedding import get_rope
from torch import nn

from atom.config import QuantizationConfig
from atom.model_ops.base_attention import Attention
from atom.model_ops.layernorm import GemmaRMSNorm
from atom.model_ops.linear import QKVParallelLinear, ReplicatedLinear, RowParallelLinear
from atom.model_ops.triton_fused_qkv_norm_rope_cache import (
    triton_fused_norm_rope_cache,
)
from atom.models.dspark_draft import DSparkDraftModel
from atom.models.kimi_k3_dspark import DSparkMarkovHead
from atom.models.minimax_m3.model import MiniMaxM3MLP
from atom.quant_spec import LayerQuantConfig
from atom.utils.forward_context import get_forward_context

if TYPE_CHECKING:
    from atom.config import Config


def _linear_out(output):
    """ATOM quantized linears may return (tensor, scale); take the tensor."""
    return output[0] if isinstance(output, tuple) else output


def _draft_rope_theta(config) -> float:
    # transformers 5 moves rope_theta into rope_parameters on the M3 VL text
    # config class this draft loads as; the plain-dict target keeps it flat.
    params = getattr(config, "rope_parameters", None) or {}
    theta = params.get("rope_theta") or getattr(config, "rope_theta", None)
    if not theta:
        raise ValueError("MiniMax-M3 DSpark draft config names no rope_theta.")
    return float(theta)


def _draft_quant_config(config) -> QuantizationConfig:
    """The draft's own quantization, independent of the target's.

    The draft checkpoint is MXFP8 (``quant_method: mxfp8``, ``[1, 32]`` blocks)
    whatever the target is -- the M3.1 targets come as MXFP8 + NVFP4 experts,
    Quark MXFP4, or BF16 -- so the target's config, which ``_build_draft_model``
    copies onto the draft, cannot be trusted to describe these weights. The
    checkpoint's own block parses here; a draft config without one is BF16.
    """
    qc = getattr(config, "quantization_config", None)
    if not qc:
        return QuantizationConfig()
    method = str(qc.get("quant_method", "")).lower()
    block = list(qc.get("weight_block_size") or [])
    if method != "mxfp8" or block != [1, 32]:
        raise ValueError(
            "MiniMax-M3 DSpark draft supports MXFP8 [1, 32] or BF16 weights, "
            f"got quant_method={method!r} weight_block_size={block}."
        )
    draft_qc = QuantizationConfig()
    draft_qc.quant_method = "mxfp8"
    # Only remap_layer_name sets this, and nothing here needs a remap: one
    # global spec covers every quantized draft linear.
    draft_qc.packed_modules_mapping = {}
    draft_qc.global_spec = LayerQuantConfig(
        quant_type=QuantType.per_1x32,
        quant_dtype=dtypes.fp8,
        is_dynamic=True,
        quant_method="mxfp8",
        weight_block_size=(1, 32),
    )
    return draft_qc


class M3DSparkAttention(nn.Module):
    """GQA attention for one draft layer, with a second entry point for the
    target-derived context rows."""

    def __init__(
        self,
        atom_config: "Config",
        config,
        quant_config: QuantizationConfig,
        layer_num: int,
        prefix: str = "",
    ) -> None:
        super().__init__()
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = config.num_attention_heads
        self.total_num_kv_heads = config.num_key_value_heads
        if self.total_num_heads % tp_size:
            raise ValueError(
                f"DSpark draft has {self.total_num_heads} attention heads, which "
                f"is not divisible by TP size {tp_size}."
            )
        self.num_heads = self.total_num_heads // tp_size
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = config.head_dim
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.kv_cache_dtype = atom_config.kv_cache_dtype
        # One KV head per rank (TP >= 4): the block runs as capturable per-row
        # paged decode; otherwise the eager gather + varlen path below.
        self.paged_block = self.num_kv_heads == 1 and self.total_num_kv_heads <= tp_size
        # 0 or absent: full attention over the context.
        self.sliding_window = int(getattr(config, "sliding_window", 0) or 0)

        self.qkv_proj = QKVParallelLinear(
            config.hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            config.hidden_size,
            bias=False,
            reduce_results=True,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )
        self.q_norm = GemmaRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = GemmaRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        rotary_dim = int(
            getattr(config, "rotary_dim", 0)
            or self.head_dim * getattr(config, "partial_rotary_factor", 1.0)
        )
        # neox (half-split) rotation of the first `rotary_dim` lanes -- the
        # gpt_oss rotary the reference uses, and what the fused kernel applies.
        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=rotary_dim,
            max_position=config.max_position_embeddings,
            base=_draft_rope_theta(config),
            rope_scaling=None,
        )
        # Never called: it is here to own this layer's KV binding. The MHA
        # backend's draft pool walks the draft for `Attention` modules and binds
        # k_cache / v_cache (5-D SHUFFLE) and the fp8 per-token scales onto it.
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            self.num_kv_heads,
            kv_cache_dtype=self.kv_cache_dtype,
            layer_num=layer_num,
            use_mla=False,
            rotary_emb=self.rotary_emb,
            q_norm=self.q_norm,
            k_norm=self.k_norm,
            per_layer_sliding_window=self.sliding_window or None,
            prefix=f"{prefix}.attn",
        )

    def _bound_cache(self):
        """This layer's KV cache in the layout every access of it uses.

        The paged block path reads through the page-16 SHUFFLE views the per-row
        decode kernels take, so it must also WRITE through them: a page-128
        SHUFFLE view puts the 128 in-page rows inside the head-dim tiles, and
        reading that memory as eight 16-row pages scrambles it. Slot numbers are
        the same in both (page * 128 + off == (page * 8 + off // 16) * 16 +
        off % 16), only the view differs. The eager path keeps page-128 views,
        which its gather reads.
        """
        attn = self.attn
        if attn.k_cache.numel() == 0:
            return None
        cache = (attn.k_cache, attn.v_cache, attn.k_scale, attn.v_scale)
        if self.paged_block:
            from atom.model_ops.attention_mha import SparseMHAPagedAttentionImpl

            cache = SparseMHAPagedAttentionImpl._to_page16_shuffle(*cache)
        return cache

    def _norm_rope_write(self, q, k, v, num_heads, positions, slot_mapping, cache):
        k_cache, v_cache, k_scale, v_scale = cache
        return triton_fused_norm_rope_cache(
            q,
            k,
            v,
            positions,
            q_norm=self.q_norm,
            k_norm=self.k_norm,
            rotary_emb=self.rotary_emb,
            num_heads=num_heads,
            num_kv_heads=self.num_kv_heads,
            head_dim=self.head_dim,
            k_cache=k_cache,
            v_cache=v_cache,
            k_scale=k_scale,
            v_scale=v_scale,
            slot_mapping=slot_mapping,
            kv_cache_dtype=self.kv_cache_dtype,
        )

    def write_context_kv(
        self,
        ctx_hidden: torch.Tensor,  # [N, hidden] projected + normed target hidden
        positions: torch.Tensor,  # [N]
        slot_mapping: torch.Tensor,  # [N] the target forward's slots
    ) -> None:
        """K/V rows for the target context: k_norm + RoPE on K, V as is.

        The q third of the fused projection is computed and dropped; the kernel
        writes K/V only when told it has zero query heads.
        """
        cache = self._bound_cache()
        if cache is None:
            return
        qkv = _linear_out(self.qkv_proj(ctx_hidden))
        _, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        self._norm_rope_write(k, k, v, 0, positions, slot_mapping, cache)

    def forward(self, positions: torch.Tensor, hidden_states: torch.Tensor):
        """The block pass: non-causal attention over ``[context | block]``.

        Paged path (``paged_block``): write the block's K/V at the block slots
        the proposer masked to the request's allocated pages -- an unallocated
        column reads as page 0, a page another request owns (under prefix
        caching, a shared prefix) -- then decode per row. Eager fallback: keep
        the block's K/V out of the cache and attend them from this pass.
        """
        qkv = _linear_out(self.qkv_proj(hidden_states))
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        cache = self._bound_cache()
        if cache is None:
            # warmup_model() runs before the pool exists; the pass only sizes
            # activations then, so shapes are all that must be right.
            out = q.new_zeros(q.shape[0], self.q_size)
            return _linear_out(self.o_proj(out))

        md = get_forward_context().attn_metadata
        if self.paged_block and getattr(md, "dspark_bt16_rows", None) is not None:
            # Capturable path: the block's K/V go into the cache at the block's
            # slots (positions past the allocated pages are -1, dropped), then
            # every block position decodes as its own q_len=1 row.
            q, _ = self._norm_rope_write(
                q, k, v, self.num_heads, positions, md.slot_mapping, cache
            )
            out = self._block_attention_paged(q, md, cache)
            return _linear_out(self.o_proj(out))
        no_write = torch.full_like(positions, -1)  # slot -1: the kernel drops it
        q, k = self._norm_rope_write(
            q, k, v, self.num_heads, positions, no_write, cache
        )
        out = self._block_attention(q, k, v, md, cache)
        return _linear_out(self.o_proj(out))

    def _block_attention_paged(self, q: torch.Tensor, md, cache) -> torch.Tensor:
        """``[context | block]`` attention as ``B * T`` q_len=1 paged decode rows.

        A length-1 query sees its whole context, so per-row decode is the
        non-causal block attention by construction. The rows read the request's
        page-16 table and capped length that ``DSparkProposer`` built
        (``dspark_bt16_rows`` / ``dspark_ctx_rows``, sliding window applied
        there), through the same per-row ASM / Gluon decode the target's sparse
        layers run -- no host sync, fixed shapes, so the block can be captured.
        The context is read once per block position; only one KV head per rank
        is supported (TP >= 4).
        """
        from atom.model_ops.minimax_m3.sparse_attn import _sparse_pa_per_row

        k16, v16, ks16, vs16 = cache  # page-16 views, see _bound_cache
        rows = q.shape[0]
        q = q.view(rows, self.num_heads, self.head_dim)
        out = torch.empty_like(q)
        _sparse_pa_per_row(
            q,
            k16,
            v16,
            md.dspark_bt16_rows,
            md.dspark_ctx_rows,
            num_kv_heads=self.num_kv_heads,
            sm_scale=self.scaling,
            output=out,
            k_scale=ks16,
            v_scale=vs16,
        )
        return out.view(rows, -1)


class M3DSparkDecoderLayer(nn.Module):
    """One draft layer: pre-norm attention + pre-norm dense SwiGLU-OAI MLP."""

    def __init__(
        self,
        atom_config: "Config",
        config,
        quant_config: QuantizationConfig,
        layer_num: int,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.self_attn = M3DSparkAttention(
            atom_config, config, quant_config, layer_num, prefix=f"{prefix}.self_attn"
        )
        self.mlp = MiniMaxM3MLP(
            config=config,
            intermediate_size=int(config.dspark_ffn_hidden_size),
            quant_config=quant_config,
            reduce_results=True,
            prefix=f"{prefix}.mlp",
        )
        self.input_layernorm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = GemmaRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def write_context_kv(self, ctx_hidden, positions) -> None:
        """Populate this layer's context rows at the TARGET forward's slots,
        which are still the installed metadata when the proposer calls this."""
        slot_mapping = get_forward_context().attn_metadata.slot_mapping[
            : ctx_hidden.shape[0]
        ]
        self.self_attn.write_context_kv(ctx_hidden, positions, slot_mapping)

    def forward(self, positions, hidden_states, residual=None):
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(positions, hidden_states)
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        return self.mlp(hidden_states), residual


class DSparkConfidenceHead(nn.Module):
    """``sigmoid(proj([h_k ; W1[x_{k-1}]]))``: the draft's per-position estimate
    that position k survives verification, fed to the confidence scheduler.
    ``h_k`` is the post-final-norm hidden the LM head reads (the reference's)."""

    def __init__(self, hidden_size: int, markov_rank: int):
        super().__init__()
        self.proj = nn.Linear(hidden_size + markov_rank, 1, bias=True)

    def forward(self, hidden: torch.Tensor, markov_embed: torch.Tensor):
        x = torch.cat([hidden.float(), markov_embed.float()], dim=-1)
        return torch.sigmoid(
            torch.nn.functional.linear(
                x, self.proj.weight.float(), self.proj.bias.float()
            )
        ).squeeze(-1)


class MiniMaxM3DSpark(DSparkDraftModel):
    """Top-level standalone DSpark draft for MiniMax-M3.1 (GQA backbone +
    Markov head + confidence head)."""

    # Overridden per instance: only the per-row paged block path is capturable.
    supports_block_graph = False
    # The proposer plans MLA persistent decode metadata only for MLA drafts.
    block_attention_is_mla = False

    weights_mapping: ClassVar[dict[str, str]] = {
        "language_model.model.dspark.": "",
        ".decoder_layer.": ".",
    }
    packed_modules_mapping: ClassVar[dict[str, tuple[str, str | int]]] = {
        ".q_proj": (".qkv_proj", "q"),
        ".k_proj": (".qkv_proj", "k"),
        ".v_proj": (".qkv_proj", "v"),
        ".gate_proj": (".gate_up_proj", 0),
        ".up_proj": (".gate_up_proj", 1),
    }

    def __init__(self, atom_config: "Config", layer_offset: int = 0) -> None:
        super().__init__()
        config = atom_config.hf_config
        self.hf_config = config
        quant_config = _draft_quant_config(config)
        # The checkpoint says `swigluoai`, and the reference draft MLP is
        # SwiGLU-OAI whatever the field says; transformers' MiniMaxM3VLTextConfig,
        # which this config loads as, rewrites the field to its `silu` default.
        config.hidden_act = "swigluoai"

        self.hidden_size = config.hidden_size
        self.noise_token_id = int(config.dspark_noise_token_id)
        self.target_layer_ids = tuple(config.dspark_target_layer_ids)
        self.markov_rank = int(config.dspark_markov_rank)
        self.vocab_size = int(config.vocab_size)

        # Replicated: the aux hidden states arrive replicated, and the
        # checkpoint keeps this projection BF16 (an ignored layer).
        self.fc = ReplicatedLinear(
            self.hidden_size * len(self.target_layer_ids),
            self.hidden_size,
            bias=False,
            quant_config=None,
            prefix="fc",
        )
        self.hidden_norm = GemmaRMSNorm(self.hidden_size, eps=config.rms_norm_eps)
        self.layers = nn.ModuleList(
            [
                M3DSparkDecoderLayer(
                    atom_config,
                    config,
                    quant_config,
                    layer_offset + i,
                    prefix=f"layers.{i}",
                )
                for i in range(config.num_hidden_layers)
            ]
        )
        self.final_norm = GemmaRMSNorm(self.hidden_size, eps=config.rms_norm_eps)
        self.markov_head = DSparkMarkovHead(self.vocab_size, self.markov_rank)
        # Read by DSparkProposer: capture the block pass when every layer runs
        # the per-row paged path, and the window its per-row tables apply.
        self.supports_block_graph = all(
            layer.self_attn.paged_block for layer in self.layers
        )
        self.sliding_window = int(getattr(config, "sliding_window", 0) or 0)
        self.confidence_head = (
            DSparkConfidenceHead(self.hidden_size, self.markov_rank)
            if getattr(config, "enable_confidence_head", False)
            else None
        )

        # Bound by share_with_target(); the checkpoint ships neither.
        self.embed_tokens = None
        self.lm_head = None

    # ---- weight-loading hooks ---------------------------------------------

    def share_with_target(self, target_base: nn.Module, loaded: set) -> None:
        target_vocab = target_base.model.embed_tokens.num_embeddings
        if target_vocab != self.vocab_size:
            raise ValueError(
                f"DSpark draft vocab {self.vocab_size} != target vocab "
                f"{target_vocab}. The draft shares the target's embedding and LM "
                "head, so the two must agree."
            )
        self.embed_tokens = target_base.model.embed_tokens
        self.lm_head = target_base.lm_head

    # ---- drafting entry points (called by the proposer) --------------------

    def project_context(self, aux_concat: torch.Tensor) -> torch.Tensor:
        """``hidden_norm(fc(concat(aux)))``, once per step for every layer.

        A target layer id is that layer's output, the residual stream before
        any final norm, which is what this checkpoint was trained on: feeding
        the last one through the target's final norm instead (the NeMo recipe's
        post-norm feature) dropped GSM8K acceptance from 51% to 39%.
        """
        return self.hidden_norm(_linear_out(self.fc(aux_concat)))

    @property
    def context_layers(self):
        return self.layers

    def forward_spec(self, input_ids, positions, num_draft):
        return self.head_and_sample(
            self.block_backbone(input_ids, positions, num_draft),
            input_ids,
            num_draft,
        )

    def block_backbone(
        self,
        input_ids: torch.Tensor,  # [B]   verified anchor token per request
        positions: torch.Tensor,  # [B*T] block absolute positions
        num_draft: int,
    ) -> torch.Tensor:
        """Embed ``[anchor, MASK, ...]``, run the layers, final norm.

        Returns the post-final-norm hidden states ``[B*T, hidden]``. No
        embedding scale: the reference embeds with the target's table as is.
        """
        bs = input_ids.shape[0]
        draft_ids = input_ids.new_full((bs, num_draft), self.noise_token_id)
        draft_ids[:, 0] = input_ids
        hidden = self.embed_tokens(draft_ids.view(-1))
        residual = None
        for layer in self.layers:
            hidden, residual = layer(positions, hidden, residual)
        hidden, _ = self.final_norm(hidden, residual)
        return hidden

    def head_and_sample(self, hidden, anchor_ids, num_draft):
        bs = anchor_ids.shape[0]
        base_logits = self.lm_head(hidden).view(bs, num_draft, -1)
        out_ids = anchor_ids.new_empty(bs, num_draft + 1)
        out_ids[:, 0] = anchor_ids
        markov_embeds = []
        for k in range(num_draft):
            # Greedy: sampling temperature is the target verify's business.
            markov_embeds.append(
                self.markov_head.sample_next(
                    out_ids[:, k], base_logits[:, k], out_ids[:, k + 1]
                )
            )
        confidence = None
        if self.confidence_head is not None:
            confidence = self.confidence_head(
                hidden.view(bs, num_draft, -1), torch.stack(markov_embeds, dim=1)
            )
        return out_ids[:, 1:], confidence
