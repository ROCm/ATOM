# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Kimi-K3 DSpark semi-autoregressive block drafter for ATOM.

DSpark is a parallel *block* drafter: one backbone pass produces the base logits
for a whole draft block, then a lightweight sequential Markov head samples the
block left-to-right. ATOM already ships a DSpark drafter for DeepSeek-V4
(``deepseek_v4_dspark.py``); this is the SAME algorithm on a DIFFERENT backbone,
and the two share the proposer, the verify scheduler and the block sampler.

What differs from the V4 flavor (all of it load-bearing):

  * Checkpoint. V4-Pro-DSpark lives inside the target checkpoint under ``mtp.*``.
    Kimi-K3-DSpark is a standalone checkpoint whose parameter names map 1:1 onto
    this module tree. It ships an ``embed_tokens`` and a ``confidence_head`` that
    are both deliberately NOT loaded -- see ``skip_weight_prefixes``.

  * Backbone. V4 reuses V4 decoder layers (MLA + mHC + MoE). This is a plain
    pre-norm stack: 5 x (MLA + dense SwiGLU MLP). No mHC, no MoE, no attention
    sink, no fp8 QAT.

  * How the target context reaches the draft. V4 keeps a private rolling
    128-slot target-KV window. Here the draft projects the target's aux hidden
    states into its OWN latent rows and writes them into the paged cache at the
    verified tokens' positions (``write_context_kv``). Because the draft's MLA
    latent is ``kv_lora_rank + qk_rope_head_dim`` = 576 wide -- identical to the
    Kimi-K3 target's full-attention layers, and independent of head count -- the
    draft binds into the TARGET's pool as extra layers rather than needing a
    sibling pool the way the Eagle3 MHA draft does.

Ported from vLLM PR #49999 ``vllm/models/kimi_k3/nvidia/dspark_mla.py`` (saved
locally at ``/home/lirzhang/ref_vllm_k3dspark/``). Trained with TorchSpec on
hidden states streamed from a live vLLM Kimi-K3 target.

Checkpoint layout (Inferact/Kimi-K3-DSpark, 68 tensors, single-file BF16):
  context_proj.weight                          [7168, 35840]  5 aux x 7168
  context_norm.weight / final_norm.weight      [7168]
  layers.{0..4}.input_layernorm.weight
  layers.{i}.post_attention_layernorm.weight
  layers.{i}.self_attn.q_a_proj.weight          [1536, 7168]  -> fused_qkv_a_proj
  layers.{i}.self_attn.kv_a_proj_with_mqa.weight [576, 7168]  -> fused_qkv_a_proj
  layers.{i}.self_attn.q_a_layernorm.weight     [1536]
  layers.{i}.self_attn.q_b_proj.weight          [12288, 1536]
  layers.{i}.self_attn.kv_a_layernorm.weight    [512]
  layers.{i}.self_attn.kv_b_proj.weight         [16384, 512]
  layers.{i}.self_attn.o_proj.weight            [7168, 8192]
  layers.{i}.mlp.{gate,up}_proj.weight          [14336, 7168] -> gate_up_proj
  layers.{i}.mlp.down_proj.weight               [7168, 14336]
  markov_head.markov_w{1,2}.weight              [163840, 256]
  confidence_head.proj.{weight,bias}            SKIPPED (training-only)
  embed_tokens.weight                           SKIPPED (target's is shared)
"""

from typing import TYPE_CHECKING, ClassVar

import torch
from aiter import QuantType, dtypes
from aiter.rotary_embedding import get_rope
from torch import nn

from atom.model_ops.activation import SiluAndMul
from atom.model_ops.attention_mla import MLAModules, mla_min_query_heads
from atom.model_ops.base_attention import Attention
from atom.model_ops.dspark_markov_sample import dspark_markov_argmax
from atom.model_ops.layernorm import RMSNorm
from atom.model_ops.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    MergedReplicatedLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from atom.models.deepseek_v2 import _fuse_rmsnorm_quant, yarn_get_mscale
from atom.models.dspark_draft import DSparkDraftModel
from atom.models.kimi_k3 import _RMS_FUSABLE_QUANT_TYPES, _effective_layer_quant
from atom.utils import envs

if TYPE_CHECKING:
    from atom.config import Config


def _linear_out(output):
    """ATOM quantized linears may return (tensor, scale); take the tensor."""
    return output[0] if isinstance(output, tuple) else output


class DSparkMarkovHead(nn.Module):
    """Low-rank first-order Markov transition bias over the draft logits.

    The full ``V x V`` transition matrix is factorized as ``B = W1 @ W2^T`` with
    ``W1, W2 in R^{V x r}``. Given the previously sampled token ``x_{k-1}``, the
    bias added to position ``k``'s base logits is ``W1[x_{k-1}] @ W2^T in R^V``.

    Because the bias is added INSIDE a per-position softmax (a local correction,
    not a global renormalization), per-token probabilities stay exact, which is
    what lets speculative verification remain lossless.

    Both tables stay replicated across TP ranks: the bias is added to the
    full-vocab logits the shared LM head produces, so a sharded ``W2`` would have
    to be gathered anyway. Same choice ATOM's V4 DSpark makes.
    """

    def __init__(self, vocab_size: int, rank: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.rank = rank
        self.markov_w1 = nn.Embedding(vocab_size, rank)
        self.markov_w2 = nn.Embedding(vocab_size, rank)
        # Read once here rather than per sampled position: envs re-reads the
        # environment on every attribute access.
        self.fused_sample = envs.ATOM_DSPARK_FUSED_MARKOV_SAMPLE

    def forward(self, token_ids: torch.Tensor):
        """Args:
            token_ids: [*]  ids of the previously sampled token x_{k-1}.
        Returns:
            logits_bias:  [*, V]  bias to add to the next position's logits.
            markov_embed: [*, r]  W1[x_{k-1}].
        """
        markov_embed = self.markov_w1(token_ids)
        # fp32 matmul: the bias enters the softmax that gates acceptance.
        logits_bias = torch.matmul(
            markov_embed.float(), self.markov_w2.weight.float().t()
        )
        return logits_bias, markov_embed

    def sample_next(self, token_ids: torch.Tensor, base_logits: torch.Tensor):
        """One greedy block position: the argmax of the biased logits, and W1[x].

        The bias itself is never returned, which is what lets the fused path
        skip materializing it: ``dspark_markov_argmax`` keeps ``W2`` bf16 and
        reduces straight to ids, with the same fp32 accumulation the softmax
        guarantee above asks for (see that module for the numerics argument).
        ``markov_embed`` is still returned because V4's confidence head
        consumes it.

        Args:
            token_ids:   [B]     ids of the previously sampled token x_{k-1}.
            base_logits: [B, V]  this position's base logits.
        Returns:
            next_ids:     [B]     argmax over the biased logits.
            markov_embed: [B, r]  W1[x_{k-1}].
        """
        if self.fused_sample:
            markov_embed = self.markov_w1(token_ids)
            next_ids = dspark_markov_argmax(
                base_logits, markov_embed, self.markov_w2.weight
            )
            return next_ids, markov_embed
        bias, markov_embed = self(token_ids)
        # bf16 + fp32 promotes the slice to fp32 before the add, so an explicit
        # .float() would only materialize it twice for the same sum.
        return (base_logits + bias).argmax(dim=-1), markov_embed


def _dspark_block_width(draft_config, atom_config) -> int:
    """The draft block width T, resolved as `DSparkProposer._resolve_mtp_k` does.

    Returns 0 when neither source names a width, which only happens outside a
    speculative run; callers treat that as "no block pass to size for".
    """
    spec_config = getattr(atom_config, "speculative_config", None)
    num_spec = getattr(spec_config, "num_speculative_tokens", None)
    return int(num_spec or getattr(draft_config, "dspark_block_size", 0) or 0)


class K3DSparkMLAAttention(nn.Module):
    """MLA attention for one draft layer, with a second entry point for the
    target-derived context rows.

    Standard DeepSeek-style MLA: ``fused_qkv_a_proj`` -> (q_lora, kv_lora), a
    ``q_b_proj`` up-projection for queries, and a ``kv_lora`` that splits into
    the ``kv_lora_rank``-wide compressed latent plus a ``qk_rope_head_dim``-wide
    positional lane. The cache stores the 576-wide latent, never expanded K/V.

    Two callers:

    * :meth:`write_context_kv` -- run once per drafting step over the tokens the
      target just verified. It computes ONLY the KV half of the projection from
      the projected target hidden states and scatters the resulting latent rows
      into this layer's slice of the paged cache. These are the rows the draft
      block will attend over.

    * :meth:`forward` -- the block pass. Ordinary MLA self-attention over the
      draft block's own hidden states.

    NOTE on causality: the draft block must attend NON-causally (every block
    position sees the whole block, not just its prefix); the block is decoded in
    one parallel pass, so intra-block order is carried by RoPE, not by a mask.
    That is settled -- sglang types these layers ``ENCODER_ONLY``, vLLM passes
    ``non_causal_multi_token_decode=True``, and vLLM ships a
    ``test_dspark_noncausal_sparse_mla``. ATOM expresses it in
    :class:`DSparkProposer`, which clears ``attn_metadata.causal`` and plans the
    decode's work descriptors non-causally; the head padding chosen in
    :meth:`__init__` is what keeps a non-causal kernel dispatchable.
    """

    def __init__(
        self,
        atom_config: "Config",
        config,
        layer_num: int,
        prefix: str = "",
    ) -> None:
        super().__init__()
        quant_config = atom_config.quant_config

        from aiter.dist.parallel_state import get_tensor_model_parallel_world_size

        self.hidden_size = config.hidden_size
        # TOTAL head count: the parallel linears below are declared at full
        # width and shard themselves. `num_local_heads` is this rank's share and
        # is what the attention op must be told — it validates kv_b_proj against
        # the weight it actually holds, which is already sharded.
        self.num_heads = config.num_attention_heads
        tp_size = get_tensor_model_parallel_world_size()
        if self.num_heads % tp_size != 0:
            raise ValueError(
                f"DSpark draft has {self.num_heads} attention heads, which is "
                f"not divisible by TP size {tp_size}."
            )
        self.num_local_heads = self.num_heads // tp_size
        self.q_lora_rank = config.q_lora_rank
        self.kv_lora_rank = config.kv_lora_rank
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.qk_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        self.v_head_dim = config.v_head_dim
        self.scaling = self.qk_head_dim**-0.5

        # q_a_proj and kv_a_proj_with_mqa share an input, so the checkpoint's two
        # weights load into one merged projection (see packed_modules_mapping).
        self.fused_qkv_a_proj = MergedReplicatedLinear(
            self.hidden_size,
            [self.q_lora_rank, self.kv_lora_rank + self.qk_rope_head_dim],
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.fused_qkv_a_proj",
        )
        # Both A-norms stay plain modules: on the block pass they are applied by
        # one `_fuse_rmsnorm_quant` launch (q norm + q activation quant + kv
        # norm), which reads their `.weight` / `.eps` directly. The module form
        # is still what `write_context_kv` calls, where only kv_a_layernorm runs
        # and its output must stay bf16 for the cache write.
        self.q_a_layernorm = RMSNorm(self.q_lora_rank, eps=config.rms_norm_eps)
        self.q_b_proj = ColumnParallelLinear(
            self.q_lora_rank,
            self.num_heads * self.qk_head_dim,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.q_b_proj",
        )
        self.kv_a_layernorm = RMSNorm(self.kv_lora_rank, eps=config.rms_norm_eps)
        self.kv_b_proj = ColumnParallelLinear(
            self.kv_lora_rank,
            self.num_heads * (self.qk_nope_head_dim + self.v_head_dim),
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.kv_b_proj",
        )
        self.o_proj = RowParallelLinear(
            self.num_heads * self.v_head_dim,
            self.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )

        # YaRN. Unlike the Kimi-K3 target (rope_theta 50000 with its own scaling)
        # these are the DRAFT's own rope parameters and must be read from the
        # draft config -- the draft was trained with them and the context rows it
        # writes are rotated with them.
        rope_params = dict(getattr(config, "rope_parameters", None) or {})
        rope_theta = rope_params.get("rope_theta") or getattr(
            config, "rope_theta", 10000.0
        )
        use_yarn = rope_params.get("rope_type") in ("yarn", "deepseek_yarn") or (
            rope_params.get("factor", 1.0) not in (1.0, None)
        )
        rope_scaling = None
        if use_yarn:
            rope_scaling = dict(rope_params)
            rope_scaling["rope_type"] = "deepseek_yarn"
            rope_scaling.setdefault(
                "original_max_position_embeddings",
                config.max_position_embeddings,
            )
        self.rotary_emb = get_rope(
            self.qk_rope_head_dim,
            rotary_dim=self.qk_rope_head_dim,
            max_position=config.max_position_embeddings,
            base=rope_theta,
            rope_scaling=rope_scaling,
            # DeepSeek-style MLA rope is interleaved unless told otherwise.
            is_neox_style=bool(getattr(config, "rope_interleave", False)),
        )
        if rope_scaling:
            mscale = yarn_get_mscale(
                float(rope_scaling["factor"]),
                float(rope_scaling.get("mscale_all_dim", False)),
            )
            self.scaling = self.scaling * mscale * mscale

        mla_modules = MLAModules(
            q_lora_rank=self.q_lora_rank,
            kv_lora_rank=self.kv_lora_rank,
            qk_nope_head_dim=self.qk_nope_head_dim,
            qk_rope_head_dim=self.qk_rope_head_dim,
            qk_head_dim=self.qk_head_dim,
            v_head_dim=self.v_head_dim,
            rotary_emb=self.rotary_emb,
            q_proj=self.q_b_proj,
            kv_b_proj=self.kv_b_proj,
            o_proj=self.o_proj,
            indexer=None,
            is_sparse=False,
            topk_tokens=None,
        )
        self.mla_attn = Attention(
            num_heads=self.num_local_heads,
            head_dim=self.kv_lora_rank + self.qk_rope_head_dim,
            scale=self.scaling,
            num_kv_heads=1,
            # The draft binds into the engine's own MLA pool, so it caches in
            # whatever dtype that pool was allocated with. This string has to
            # agree with the bound tensor: it selects the fp8 decode overload
            # and is passed straight to concat_and_cache_mla on the
            # context-write path, and "fp8" over a bf16 tensor (or the reverse)
            # aborts inside cache_kernels.
            kv_cache_dtype=atom_config.kv_cache_dtype,
            # The block pass runs this decode non-causally, which narrows the
            # aiter kernels available for it; the draft's own block width picks
            # the head padding that keeps one dispatchable.
            min_query_heads=mla_min_query_heads(
                atom_config.kv_cache_dtype, _dspark_block_width(config, atom_config)
            ),
            layer_num=layer_num,
            use_mla=True,
            mla_modules=mla_modules,
            config=atom_config,
            prefix=f"{prefix}.mla_attn",
        )

        # Which scheme -- if any -- the two fusable activation quants run in.
        # Resolved from the CONSUMER linear, the way kimi_k3 does it, so an
        # excluded or differently-quantized layer silently falls back to plain
        # norms instead of feeding a GEMM the wrong layout.
        #
        # q side: q_b_proj consumes the normed query, inside the MLA impl.
        qknorm_type, qknorm_dtype = _effective_layer_quant(
            quant_config, f"{prefix}.q_b_proj"
        )
        self.fuse_qknorm_quant = qknorm_dtype in (dtypes.fp8, dtypes.fp4x2)
        self.qknorm_dtype = qknorm_dtype if self.fuse_qknorm_quant else torch.bfloat16
        self.qknorm_quant_type_value = (
            qknorm_type.value if self.fuse_qknorm_quant else QuantType.No.value
        )
        # Attention-input side: fused_qkv_a_proj is the only consumer of the
        # decoder's input_layernorm here (the draft has no g_proj gate), so one
        # scheme decides it. The layer reads these two back off the attention.
        self.fuse_input_norm_quant = (
            _effective_layer_quant(quant_config, f"{prefix}.fused_qkv_a_proj")[0]
            in _RMS_FUSABLE_QUANT_TYPES
        )
        self.input_quant_prefix = f"{prefix}.fused_qkv_a_proj"

    # ---- context rows (target-derived) -------------------------------------

    def write_context_kv(
        self,
        ctx_hidden: torch.Tensor,  # [N, hidden] projected + normed target hidden
        positions: torch.Tensor,  # [N]         absolute positions
        slot_mapping: torch.Tensor,  # [N]      flat slots in the paged pool
    ) -> None:
        """Project the target context into this layer's latent cache rows.

        The checkpoint keeps q_a and kv_a as one fused weight and so does this
        module (see packed_modules_mapping); the block pass in :meth:`forward`
        wants both halves, this path only the kv one -- so 1536 of the 2112
        output columns are computed and dropped. Narrowing the GEMM to the kv
        shard is not available where it would pay: the served configuration
        quantizes this projection per output channel and preshuffles its rows,
        after which a row slice of the merged weight is not that shard's weight.
        (vLLM's reference computes the full projection here too, and offers a
        cross-layer fused fast path on top; that optimization is not ported.)
        """
        kv_lora = _linear_out(self.fused_qkv_a_proj(ctx_hidden))[
            ..., self.q_lora_rank :
        ]
        # norm + rope + concat + store live behind one call so every cache
        # layout stays in the attention impl: the normal store hardcodes
        # attn_metadata.slot_mapping, which is the draft block's, not these
        # context rows'. It fuses the four ops into one kernel when the layout
        # allows and otherwise runs exactly the chain this used to run inline.
        self.mla_attn.impl.write_context_kv_latent(
            self.mla_attn.kv_cache,
            kv_lora,
            positions,
            slot_mapping,
            self.kv_a_layernorm,
        )

    # ---- block pass --------------------------------------------------------

    def forward(
        self, positions: torch.Tensor, hidden_states: torch.Tensor
    ) -> torch.Tensor:
        # `hidden_states` is a (fp8, scale) tuple when the decoder's
        # input_layernorm fused its activation quant; fused_qkv_a_proj is its
        # only consumer and takes the scale directly.
        hidden_states_scale = None
        if isinstance(hidden_states, tuple):
            hidden_states, hidden_states_scale = hidden_states
        qkv_lora = _linear_out(
            self.fused_qkv_a_proj(hidden_states, hidden_states_scale)
        )
        q_lora, kv_c, k_pe = qkv_lora.split(
            [self.q_lora_rank, self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
        )
        # Stop at q_a_layernorm and hand the LORA-rank query to the attention.
        # `q_b_proj`, the RoPE on k_pe, the cache write, and `o_proj` all live
        # INSIDE the MLA impl -- they were handed to it via MLAModules, and it
        # returns a hidden_size-wide tensor, already output-projected. Doing any
        # of them out here applies them twice. (Same call shape as
        # deepseek_v2: `mla_attn(q_a_layernorm(q_c), kv_a_layernorm(kv_c),
        # k_pe, positions)`.)
        #
        # One kernel for both A-norms plus the query's activation quant, as in
        # kimi_k3's own MLA: q_b_proj takes the (fp8, scale) pair via `q_scale=`,
        # while kv stays bf16 because the cache is written from it. When the q
        # scheme is not fusable, qknorm_dtype is bf16 / QuantType.No and this
        # degrades to a plain fused norm pair with q_scale None.
        q_shuffle = False
        q_scale_shuffle_padding = False
        if self.qknorm_dtype == dtypes.fp4x2:
            from atom.model_ops.linear import use_triton_gemm
            from atom.models.deepseek_v2 import _mxfp4_activation_quant_layout

            if not use_triton_gemm():
                q_shuffle, q_scale_shuffle_padding = _mxfp4_activation_quant_layout(
                    q_lora.shape[0]
                )
        (q_c, q_scale), _, kv_c, _ = _fuse_rmsnorm_quant(
            q_lora,
            self.q_a_layernorm.weight,
            self.q_a_layernorm.eps,
            kv_c,
            self.kv_a_layernorm.weight,
            self.kv_a_layernorm.eps,
            None,
            dtype_quant=self.qknorm_dtype,
            shuffle=q_shuffle,
            scale_shuffle_padding=q_scale_shuffle_padding,
            group_size=128,
            quant_type=self.qknorm_quant_type_value,
            output_unquantized_inp1=False,
            transpose_scale=True,
        )
        return self.mla_attn(q_c, kv_c, k_pe, positions, q_scale=q_scale)


class K3DSparkMLP(nn.Module):
    """Dense SwiGLU MLP (gate_proj / up_proj fused, down_proj)."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        quant_config=None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_up_proj",
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.down_proj",
        )
        self.act_fn = SiluAndMul()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # `x` is a (fp8, scale) tuple when post_attention_layernorm fused its
        # activation quant; gate_up_proj is its only consumer.
        x_scale = None
        if isinstance(x, tuple):
            x, x_scale = x
        return _linear_out(
            self.down_proj(self.act_fn(_linear_out(self.gate_up_proj(x, x_scale))))
        )


class K3DSparkDecoderLayer(nn.Module):
    """One draft layer: pre-norm MLA + pre-norm dense MLP, both residual."""

    def __init__(
        self,
        atom_config: "Config",
        config,
        layer_num: int,
        prefix: str = "",
    ) -> None:
        super().__init__()
        quant_config = atom_config.quant_config
        self.self_attn = K3DSparkMLAAttention(
            atom_config, config, layer_num, prefix=f"{prefix}.self_attn"
        )
        self.mlp = K3DSparkMLP(
            config.hidden_size,
            config.intermediate_size,
            quant_config=quant_config,
            prefix=f"{prefix}.mlp",
        )
        # Both pre-norms fuse their activation quant into the GEMM that consumes
        # them (fused_qkv_a_proj / gate_up_proj), each the sole consumer of its
        # normed output -- the residual stream is branched off BEFORE the norm,
        # so nothing else needs the bf16 form. The quant op then disappears from
        # the drafting step instead of running as its own pass over [T, 7168].
        self.input_layernorm = RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
            fused_quant=self.self_attn.fuse_input_norm_quant,
            quant_config=(
                quant_config if self.self_attn.fuse_input_norm_quant else None
            ),
            prefix=self.self_attn.input_quant_prefix,
        )
        self.fuse_ffn_norm_quant = (
            _effective_layer_quant(quant_config, f"{prefix}.mlp.gate_up_proj")[0]
            in _RMS_FUSABLE_QUANT_TYPES
        )
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
            fused_quant=self.fuse_ffn_norm_quant,
            quant_config=quant_config if self.fuse_ffn_norm_quant else None,
            prefix=f"{prefix}.mlp.gate_up_proj",
        )

    def write_context_kv(self, ctx_hidden, positions) -> None:
        """Populate this layer's context rows.

        The context source does NOT go through ``input_layernorm``: the reference
        feeds every layer the same ``context_norm(context_proj(aux))`` tensor
        straight into the KV projection, while ``input_layernorm`` applies only
        to the residual stream carrying the draft block.
        """
        from atom.utils.forward_context import get_forward_context

        slot_mapping = get_forward_context().attn_metadata.slot_mapping[
            : ctx_hidden.shape[0]
        ]
        self.self_attn.write_context_kv(ctx_hidden, positions, slot_mapping)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(positions, hidden_states)
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        return self.mlp(hidden_states), residual


class KimiK3DSpark(DSparkDraftModel):
    """Top-level standalone DSpark draft (MLA backbone + Markov head).

    Parameter names match the checkpoint 1:1, so no ``WeightsMapper`` is needed;
    only the fused q/kv-A and gate/up projections are remapped, via
    ``packed_modules_mapping``.
    """

    packed_modules_mapping: ClassVar[dict[str, tuple[str, int]]] = {
        "q_a_proj": ("fused_qkv_a_proj", 0),
        "kv_a_proj_with_mqa": ("fused_qkv_a_proj", 1),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    # Both are present in the checkpoint and both are deliberately not loaded.
    #   confidence_head: training-only. vLLM skips it by the same name and its
    #     model card lists confidence-based scheduling as future work, so the
    #     head is not calibrated for inference. ATOM's confidence-scheduled
    #     ragged verify therefore has no trustworthy input with THIS checkpoint
    #     -- run it with a fixed verify length.
    #   embed_tokens: a 2.35GB copy of the target's table. The target's is
    #     shared instead (share_with_target), so loading it would just burn
    #     memory and risk drifting from the target's.
    skip_weight_prefixes: ClassVar[list[str]] = [
        "confidence_head.",
        "embed_tokens.",
        "lm_head.",
    ]

    def __init__(self, atom_config: "Config", layer_offset: int = 0) -> None:
        super().__init__()
        self.atom_config = atom_config
        config = atom_config.hf_config
        self.hf_config = config

        self.hidden_size = config.hidden_size
        self.noise_token_id = int(config.dspark_noise_token_id)
        self.target_layer_ids = tuple(config.dspark_target_layer_ids)
        self.markov_rank = int(config.dspark_markov_rank)

        # Aux fusion: [N, target_hidden * num_target_layers] -> [N, hidden].
        # Replicated, matching the reference and ATOM's other drafts: the aux
        # hidden states arrive replicated (they are the target's residual
        # stream), so a row-parallel split would trade memory for an all-reduce
        # on every drafting step.
        self.context_proj = ReplicatedLinear(
            config.target_hidden_size * config.num_target_layers,
            self.hidden_size,
            bias=False,
            prefix="context_proj",
        )
        self.context_norm = RMSNorm(self.hidden_size, eps=config.rms_norm_eps)

        # Draft layer_num continues past the target's layers so each binds to its
        # own kv_cache_data entry in the shared pool.
        self.layers = nn.ModuleList(
            [
                K3DSparkDecoderLayer(
                    atom_config, config, layer_offset + i, prefix=f"layers.{i}"
                )
                for i in range(config.num_hidden_layers)
            ]
        )
        self.final_norm = RMSNorm(self.hidden_size, eps=config.rms_norm_eps)
        self.markov_head = DSparkMarkovHead(config.vocab_size, self.markov_rank)
        # Markov-head vocab, exposed at the top level so the proposer can clamp
        # the anchor uniformly across DSpark flavors (V4 exposes it too).
        self.vocab_size = int(config.vocab_size)

        # Bound by share_with_target(); both are skipped at load.
        self.embed_tokens = None
        self.lm_head = None

    # ---- weight-loading hooks ---------------------------------------------

    def share_with_target(self, target_base: nn.Module, loaded: set) -> None:
        """Bind embed/LM head to the target's already-loaded instances.

        ``target_base`` is the Kimi-K3 ``KimiLinearForCausalLM`` (ModelRunner
        unwraps the ``language_model`` multimodal wrapper before calling this).
        The vocabularies must agree or the shared LM head would silently score
        the wrong rows.
        """
        target_vocab = target_base.model.embed_tokens.num_embeddings
        if target_vocab != self.hf_config.vocab_size:
            raise ValueError(
                f"DSpark draft vocab {self.hf_config.vocab_size} != target vocab "
                f"{target_vocab}. The draft shares the target's embedding and LM "
                "head, so the two must agree."
            )
        self.embed_tokens = target_base.model.embed_tokens
        self.lm_head = target_base.lm_head

    # ---- drafting entry points (called by the proposer) --------------------

    def project_context(self, aux_concat: torch.Tensor) -> torch.Tensor:
        """``context_norm(context_proj(concat(aux)))`` -- the shared context.

        Computed ONCE per step and reused by every layer: the reference hands
        the same projected tensor to all five, which each apply their own KV
        projection to it.
        """
        return self.context_norm(_linear_out(self.context_proj(aux_concat)))

    @property
    def context_layers(self):
        """Every draft layer holds context rows; `layers` is already in order.

        The dummy-run guard and the project-once-share-everywhere loop live in
        :meth:`DSparkDraftModel.write_context_kv`. The specific abort this draft
        would hit without the guard is `kv_cache.size(2) == kv_lora_rank +
        qk_rope_head_dim` in cache_kernels, on the still-empty init tensor.
        """
        return self.layers

    def forward_spec(
        self,
        input_ids: torch.Tensor,  # [B]   verified anchor token per request
        positions: torch.Tensor,  # [B*T] block absolute positions
        num_draft: int,
    ):
        """One DSpark block: parallel backbone pass + sequential Markov sampling.

        Returns ``(draft_token_ids [B, T], confidence)``. ``confidence`` is
        always ``None`` here -- this checkpoint's confidence head is
        training-only (see ``skip_weight_prefixes``). The proposer already
        handles ``None`` by leaving the verify length fixed.
        """
        bs = input_ids.shape[0]
        T = num_draft

        # Block input ids: [anchor, MASK, MASK, ...]. T positions in, T draft
        # tokens out -- the same convention ATOM's V4 DSpark and vLLM/sglang all
        # use. (dflash.py's standalone `spec_generate` demo slices
        # `[:, -block_size+1:]` instead; that is NOT what any production
        # implementation does. Do not "fix" the sampler to match it.)
        draft_ids = input_ids.new_full((bs, T), self.noise_token_id)
        draft_ids[:, 0] = input_ids
        hidden = self.embed_tokens(draft_ids.view(-1))

        residual = None
        for layer in self.layers:
            hidden, residual = layer(positions, hidden, residual)
        hidden, _ = self.final_norm(hidden, residual)

        base_logits = self.lm_head(hidden).view(bs, T, -1)
        return self._sample_block(base_logits, input_ids), None

    def _sample_block(
        self,
        base_logits: torch.Tensor,  # [B, T, V]
        anchor_ids: torch.Tensor,  # [B]
    ) -> torch.Tensor:
        """Markov-biased left-to-right sampling over the already-computed block.

            logits_k <- U_k + B(x_{k-1}, .);  x_k <- argmax(logits_k)

        Only this loop is sequential, and it is T tiny rank-256 matmuls rather
        than T backbone passes -- which is the entire point of DSpark.
        """
        bs, T, _ = base_logits.shape
        out_ids = anchor_ids.new_empty(bs, T + 1)
        out_ids[:, 0] = anchor_ids
        for k in range(T):
            # Greedy: temperature/sampling is applied by the target's verify.
            out_ids[:, k + 1], _ = self.markov_head.sample_next(
                out_ids[:, k], base_logits[:, k]
            )
        return out_ids[:, 1:]
