# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Inference-only GLM-5.3-Flash (``glm5_next``) text model.

320B total / 18B active. 45 text layers in a hybrid pattern -- 34 KDA linear
attention layers and 11 MLA layers -- with 4-stream manifold-constrained
hyper-connections (mHC) at every attention and FFN site, and a 288-expert MoE.
The checkpoint is multimodal; ATOM serves the text path, so the vision tower
under ``model.visual.*`` is skipped at load.

Almost every piece here is an existing ATOM component:

* KDA            -> ``kimi_k3.KimiKDAAttention`` + aiter ``kimi_delta_attn``
* mHC            -> ``model_ops.sparse_attn_v4.hc_split_sinkhorn`` + aiter ``mhc_*``
* MLA            -> ``model_ops.attention_mla`` via ``MLAModules``
* MoE            -> ``model_ops.moe.FusedMoE`` (sigmoid / noaux_tc)
* clamped SwiGLU -> ``model_ops.swiglu_oai.swiglu_oai_split``

Three deliberate v1 decisions, each exact rather than approximate:

1. **DSA layers run dense.** ``index_topk=2048`` with ``index_kpool=4`` selects
   ``2048/4 = 512`` pools covering 2048 tokens, so for any sequence at or below
   2048 tokens the indexer selects *everything* and dense causal MLA is
   numerically identical. ATOM already gates sparsity the same way
   (``deepseek_v2._pcp_sparse_active``: ``max_seqlen_k > index_topk``). Beyond
   2048 tokens this model is not yet correct -- see ``recipes/GLM-5.3-Flash.md``.
   The validated dense reference for the sparse path is
   ``model_ops.kpool_indexer``.

2. **NoPE runs on a zero-width rope slice.** The text model is entirely NoPE
   (``qk_rope_head_dim == 0``) while ATOM's MLA splits q/k into nope+rope parts,
   so the rope half is simply empty and the rotary is the identity
   (``_NoPositionalRotaryEmbedding``). Zero-padding those lanes instead would
   also have been exact, but is impossible: ``qk_nope_head_dim`` is already 256
   and the CK prefill kernel caps head dimensions at 256.

3. **The KDA output gate is folded at load.** GLM's gate is low-rank
   (``g_b_proj @ g_a_proj``) where Kimi-K3's is a single ``g_proj``. Both are
   linear with nothing in between, so the product is materialised once after
   load and written into the fused ``in_proj``'s ``g`` shard. That makes
   ``KimiKDAAttention`` -- and all of its state-cache, TP and CUDA-graph
   integration -- reusable unchanged.

Not yet wired: the MTP draft layer (checkpoint layer 45) and the vision tower.
"""

import os
from itertools import islice
from typing import Any, ClassVar

import aiter
import torch
import torch.nn.functional as F
from aiter.dist.communication_op import tensor_model_parallel_all_reduce
from aiter.dist.parallel_state import (
    get_ep_group,
    get_pp_group,
    get_tensor_model_parallel_world_size,
    get_tp_group,
)
from torch import nn

from atom.config import Config, QuantizationConfig
from atom.model_ops.attention_mla import MLAModules
from atom.model_ops.base_attention import Attention
from atom.model_ops.embed_head import ParallelLMHead, VocabParallelEmbedding
from atom.model_ops.layernorm import RMSNorm
from atom.model_ops.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from atom.model_ops.moe import FusedMoE
from atom.model_ops.sparse_attn_v4 import hc_split_sinkhorn
from atom.model_ops.swiglu_oai import swiglu_oai_split
from atom.model_ops.topK import (
    is_rocm_aiter_fuse_routed_scaling_factor,
    is_rocm_aiter_fusion_shared_expert_enabled,
)
from atom.model_ops.utils import atom_parameter
from atom.models.kimi_k3 import (
    KimiKDAAttention,
    _NoPositionalRotaryEmbedding,
    _text_config,
)
from atom.models.utils import (
    IntermediateTensors,
    PPMissingLayer,
    make_layers,
    maybe_prefix,
)

# NoPE is carried by a zero-width rope slice plus an identity rotary
# (`_NoPositionalRotaryEmbedding`), not by padding.
#
# Padding the rope lanes with zeros looked attractive -- a zero q_pe dotted with
# a zero k_pe is exactly NoPE, and it would have kept MLA on its well-trodden
# rope path -- but it is not available here: `qk_nope_head_dim` is already 256
# and the CK prefill kernel refuses head dimensions above that ("CK only
# supports head dimension at most 256"), so any padding at all overflows it.


def _normalize_glm5_next_config(config) -> None:
    """Fill the aliases the shared ATOM MoE / KDA / MLA infrastructure expects.

    The text config arrives as a bare ``PretrainedConfig`` (this image's
    transformers has no ``Glm5NextTextConfig``), so everything is a plain attr.
    """
    lin = getattr(config, "linear_attn_config", {}) or {}

    # --- KDA aliases (names KimiKDAAttention reads) ---
    config.linear_num_key_heads = lin.get("num_heads", config.num_attention_heads)
    config.linear_num_value_heads = config.linear_num_key_heads
    config.linear_key_head_dim = lin.get("head_dim", 128)
    config.linear_value_head_dim = config.linear_key_head_dim
    config.linear_conv_kernel_dim = lin.get("short_conv_kernel_size", 4)

    # GLM lists these 0-based (layer_types[3] == "deepseek_sparse_attention" and
    # full_attn_layers starts at 3). Kimi-K3 lists them 1-based and subtracts one
    # -- do NOT do that here.
    config.glm5_kda_layers = [int(i) for i in lin.get("kda_layers", [])]
    config.glm5_full_attn_layers = [int(i) for i in lin.get("full_attn_layers", [])]
    if not config.glm5_kda_layers:
        types = getattr(config, "layer_types", []) or []
        config.glm5_kda_layers = [
            i for i, t in enumerate(types) if t == "linear_attention"
        ]
        config.glm5_full_attn_layers = [
            i for i, t in enumerate(types) if t != "linear_attention"
        ]
    config.num_gdn_attn_state = len(config.glm5_kda_layers)
    config.num_full_attn = len(config.glm5_full_attn_layers)
    # KimiKDAAttention keys its layer membership off this name.
    config.kimi_kda_layers = config.glm5_kda_layers
    config.kimi_full_attn_layers = config.glm5_full_attn_layers

    # --- MoE aliases ---
    config.num_experts = getattr(config, "n_routed_experts", None)
    config.moe_layer_freq = getattr(config, "moe_layer_freq", 1)

    # --- MLA aliases (the checkpoint is NoPE: qk_rope_head_dim == 0) ---
    config.glm5_is_nope = int(getattr(config, "qk_rope_head_dim", 0)) == 0
    config.head_dim = config.qk_nope_head_dim + config.qk_rope_head_dim

    if getattr(config, "rope_parameters", None) is None:
        config.rope_parameters = {
            "rope_theta": getattr(config, "rope_theta", 10000.0),
            "rope_type": "default",
        }


# Checkpoint KDA projections -> the fused `in_proj`.
#
# Kimi-K3 has to enumerate this per KDA layer, because its full-attention layers
# also own a `g_proj` that must not be folded. GLM-5.3 needs no such care: its
# MLA layers use `q_a_proj` / `q_b_proj` / `kv_a_proj_with_mqa` / `kv_b_proj`, and
# none of those contain `.q_proj`, `.k_proj` or `.v_proj` as a substring (the
# match is anchored on a leading "."). So a single layer-agnostic mapping is
# unambiguous, and -- unlike a per-layer one -- it can live on the class, where
# `model_runner` reads it *before* the model is constructed. Getting that
# ordering right matters: remapping the quant config a second time from
# `__init__` corrupts its layer pattern specs and silently marks every
# attention projection quantized.
#
# Only q/k/v come from the checkpoint. Shard 3 (`g`) is filled after load by
# folding the low-rank `g_b_proj @ g_a_proj`; see
# `Glm5NextKDAAttention.process_weights_after_loading`.
_KDA_PACKED_MODULES_MAPPING: dict[str, tuple[str, int]] = {
    ".q_proj": (".in_proj", 0),
    ".k_proj": (".in_proj", 1),
    ".v_proj": (".in_proj", 2),
}


class Glm5NextHyperConnection(nn.Module):
    """One mHC site (attention or FFN) over a 4-wide residual stream.

    Owns the learned ``(fn, base, scale)`` that turn the ``hc_mult`` incoming
    streams into collapse / expand weights, exactly as DeepSeek-V4's ``Block``
    does -- the checkpoint even uses the same tensor names (``hc_attn_fn`` /
    ``hc_attn_base`` / ``hc_attn_scale``), and ``hc_attn_fn`` is
    ``[(2 + hc)*hc, hc*dim]``, precisely ``hc_split_sinkhorn``'s ``mixes`` layout.

    ``pre`` collapses ``[T, hc, dim]`` to ``[T, dim]`` for the sub-layer; ``post``
    and ``comb`` expand its output back. GLM applies the sub-layer's own RMSNorm
    to the collapsed vector, which the fused aiter kernel folds in.
    """

    HC_POST_MULT = 2.0  # GLM's post gate is 2 * sigmoid(.), same as V4

    def __init__(self, config) -> None:
        super().__init__()
        self.hc_mult = int(config.hc_mult)
        self.hc_sinkhorn_iters = int(config.hc_sinkhorn_iters)
        self.hc_eps = float(getattr(config, "hc_eps", 1e-6))
        self.norm_eps = float(config.rms_norm_eps)

        # aiter's mhc kernels trap unless hidden % 512 == 0 or % 256 == 0.
        # GLM53_DISABLE_FUSED_MHC=1 forces the torch reference path, for
        # bisecting numerical differences against transformers.
        dim_ok = config.hidden_size % 512 == 0 or config.hidden_size % 256 == 0
        if os.environ.get("GLM53_DISABLE_FUSED_MHC") == "1":
            dim_ok = False
        self._mhc_pre = getattr(aiter, "mhc_pre", None) if dim_ok else None
        self._mhc_post = getattr(aiter, "mhc_post", None) if dim_ok else None

    def pre(
        self,
        residual: torch.Tensor,
        fn: torch.Tensor,
        scale: torch.Tensor,
        base: torch.Tensor,
        norm_weight: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """``[T, hc, dim]`` -> sub-layer input ``[T, dim]`` plus (post, comb)."""
        if self._mhc_pre is not None:
            post, comb, y = self._mhc_pre(
                residual,
                fn,
                scale,
                base,
                float(self.norm_eps),
                float(self.hc_eps),
                float(self.hc_eps),
                self.HC_POST_MULT,
                int(self.hc_sinkhorn_iters),
                norm_weight,
                self.norm_eps,
            )
            return y, post.squeeze(-1), comb

        dtype = residual.dtype
        flat = residual.flatten(-2).float()
        # Unweighted RMSNorm over the flattened streams, then the hc-fn linear.
        normed = flat * torch.rsqrt(
            flat.square().mean(-1, keepdim=True) + self.norm_eps
        )
        mixes = F.linear(normed, fn)
        pre, post, comb = hc_split_sinkhorn(
            mixes,
            scale,
            base,
            self.hc_mult,
            self.hc_sinkhorn_iters,
            self.hc_eps,
        )
        y = torch.sum(pre.unsqueeze(-1) * residual, dim=-2)
        if norm_weight is not None:
            y = F.rms_norm(
                y.float(), (y.shape[-1],), norm_weight.float(), self.norm_eps
            )
        return y.to(dtype), post, comb

    def post_expand(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
        post: torch.Tensor,
        comb: torch.Tensor,
    ) -> torch.Tensor:
        """Sub-layer output ``[T, dim]`` -> new residual ``[T, hc, dim]``."""
        if self._mhc_post is not None:
            out = torch.empty_like(residual)
            self._mhc_post(out, x, residual, post.unsqueeze(-1), comb)
            return out
        y = post.unsqueeze(-1) * x.unsqueeze(-2) + torch.sum(
            comb.unsqueeze(-1) * residual.unsqueeze(-2), dim=-3
        )
        return y.type_as(x)


class Glm5NextMLP(nn.Module):
    """Dense FFN with GLM's clamped SwiGLU.

    ``swiglu_oai_split(alpha=1, beta=0, limit=swiglu_limit)`` computes
    ``gate*sigmoid(gate) * (up + 0)`` with ``gate`` clamped above and ``up``
    clamped both ways -- identical to the reference's
    ``silu(clamp(gate)) * clamp(up)``.
    """

    def __init__(
        self,
        config,
        intermediate_size: int,
        quant_config: QuantizationConfig | None = None,
        reduce_results: bool = True,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            config.hidden_size,
            [intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_up_proj",
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            config.hidden_size,
            bias=False,
            quant_config=quant_config,
            reduce_results=reduce_results,
            prefix=f"{prefix}.down_proj",
        )
        self.swiglu_limit = float(getattr(config, "swiglu_limit", 10.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up = self.gate_up_proj(x)
        x = swiglu_oai_split(gate_up, alpha=1.0, beta=0.0, limit=self.swiglu_limit)
        return self.down_proj(x)


class Glm5NextMoE(nn.Module):
    """288 routed experts (8/token) + 1 shared, sigmoid routing with noaux_tc."""

    def __init__(
        self,
        config,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.tp_size = get_tp_group().world_size
        self.routed_scaling_factor = config.routed_scaling_factor
        self.n_routed_experts = int(config.n_routed_experts)
        self.n_shared_experts = int(config.n_shared_experts or 0)
        self.swiglu_limit = float(getattr(config, "swiglu_limit", 10.0))

        ep_group = get_ep_group().device_group
        self.ep_size = ep_group.size()

        # Matches the transformers reference: the router is a plain fp32 matmul,
        # not a ReplicatedLinear.
        self.gate = nn.Linear(
            config.hidden_size, self.n_routed_experts, bias=False, dtype=torch.float32
        )
        self.gate.e_score_correction_bias = atom_parameter(
            torch.empty(self.n_routed_experts, dtype=torch.float32)
        )

        self.is_fusion_shared_expert = is_rocm_aiter_fusion_shared_expert_enabled(
            shared_expert_prefix=f"{prefix}.shared_experts",
            routed_expert_prefix=f"{prefix}.experts",
        )
        self.n_logical_experts = self.n_routed_experts
        self.n_redundant_experts = 0
        self.n_physical_experts = self.n_logical_experts
        self.n_local_physical_experts = self.n_physical_experts // self.ep_size

        if self.n_shared_experts and not self.is_fusion_shared_expert:
            self.shared_experts = Glm5NextMLP(
                config,
                intermediate_size=config.moe_intermediate_size * self.n_shared_experts,
                quant_config=quant_config,
                reduce_results=False,
                prefix=f"{prefix}.shared_experts",
            )
        else:
            self.shared_experts = None

        self.experts = FusedMoE(
            num_experts=self.n_routed_experts,
            top_k=int(config.num_experts_per_tok),
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            reduce_results=False,
            renormalize=bool(config.norm_topk_prob),
            quant_config=quant_config,
            use_grouped_topk=True,
            num_expert_group=int(getattr(config, "n_group", 1)),
            topk_group=int(getattr(config, "topk_group", 1)),
            prefix=f"{prefix}.experts",
            scoring_func="sigmoid",
            e_score_correction_bias=self.gate.e_score_correction_bias,
            config=config,
            shared_expert_prefix=f"{prefix}.shared_experts",
        )
        # GLM clamps gate/up inside the expert SwiGLU; aiter bakes the limit into
        # the GEMM1 kernel when the layer carries this attribute.
        self.experts.swiglu_limit = self.swiglu_limit

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        num_tokens, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)

        router_logits = self.gate(hidden_states.to(dtype=torch.float32))
        shared_output = None
        if self.shared_experts is not None:
            shared_output = self.shared_experts(hidden_states)

        out = self.experts(hidden_states=hidden_states, router_logits=router_logits)
        if not is_rocm_aiter_fuse_routed_scaling_factor():
            out = out * self.routed_scaling_factor
        if shared_output is not None:
            out = out + shared_output
        if self.tp_size > 1:
            out = tensor_model_parallel_all_reduce(out)
        return out.view(num_tokens, hidden_dim)


class Glm5NextKDAAttention(KimiKDAAttention):
    """Kimi-K3 KDA with GLM's low-rank output gate folded into ``in_proj``.

    Structurally the two are the same layer -- same separate ``q/k/v_conv1d``,
    per-head ``A_log``, per-channel ``dt_bias``, low-rank forget gate, and the
    same ``gate_lower_bound`` sigmoid. The only difference is the *output* gate:
    Kimi projects it in one step (``g_proj``), GLM factorises it
    (``g_b_proj @ g_a_proj``, rank 128).

    Both are linear with no nonlinearity between, so materialising the product
    once after load is exact, and lets the parent's fused single-GEMM forward,
    recurrent state cache, TP sharding and CUDA-graph handling all apply
    unchanged.
    """

    def __init__(self, atom_config: Config, quant_config, prefix: str = "") -> None:
        super().__init__(atom_config, quant_config, prefix=prefix)
        config = _text_config(atom_config.hf_config)
        # Replaces the parent's `g_proj`, which this checkpoint does not have.
        self.g_a_proj = ReplicatedLinear(
            self.hidden_size,
            self.head_dim,
            bias=False,
            quant_config=None,
            prefix=f"{prefix}.g_a_proj",
        )
        self.g_b_proj = ColumnParallelLinear(
            self.head_dim,
            self.proj_size,
            bias=False,
            quant_config=None,
            prefix=f"{prefix}.g_b_proj",
        )
        self.o_norm_eps = config.rms_norm_eps

    def process_weights_after_loading(self) -> None:
        if getattr(self, "_in_proj_fused", False):
            return
        # in_proj is [q | k | v | g]; the checkpoint only supplied q/k/v, so
        # write the folded gate into the g shard before the parent appends its
        # b_proj / f_a_proj tails.
        g = (self.g_b_proj.weight.data.float() @ self.g_a_proj.weight.data.float()).to(
            self.in_proj.weight.dtype
        )
        lp = self.local_proj_size
        assert g.shape == (
            lp,
            self.hidden_size,
        ), f"folded KDA gate has shape {tuple(g.shape)}, expected {(lp, self.hidden_size)}"
        self.in_proj.weight.data[3 * lp : 4 * lp].copy_(g)
        # Release the factors; they are never used again.
        for m in (self.g_a_proj, self.g_b_proj):
            m.weight.data = m.weight.data.new_empty(0)
        super().process_weights_after_loading()


class Glm5NextIndexer(nn.Module):
    """Weights of the k-pool DSA indexer.

    Declared but not yet used: v1 runs these layers as dense MLA, which is exact
    at or below ``index_topk`` tokens. Holding the parameters keeps the load
    report clean (they are real checkpoint data, and the loader is right to
    complain when it has to drop them) and is what the sparse path will bind to.
    The selection maths itself is already implemented and validated in
    ``atom.model_ops.kpool_indexer``.
    """

    def __init__(self, config) -> None:
        super().__init__()
        n_heads = int(config.index_n_heads)
        head_dim = int(config.index_head_dim)
        kpool = int(config.index_kpool)
        self.wq_b = ReplicatedLinear(
            config.q_lora_rank, n_heads * head_dim, bias=False, quant_config=None
        )
        self.wk = ReplicatedLinear(
            config.hidden_size, head_dim, bias=False, quant_config=None
        )
        self.k_norm = nn.LayerNorm(head_dim, eps=1e-6)
        self.weights_proj = ReplicatedLinear(
            config.hidden_size, n_heads, bias=False, quant_config=None
        )
        self.index_kpool_compress_ape = atom_parameter(torch.empty(kpool, head_dim))
        self.index_kpool_compress_gate = atom_parameter(
            torch.empty(head_dim, config.hidden_size)
        )


class Glm5NextMLAAttention(nn.Module):
    """NoPE MLA. Dense in v1 -- see the module docstring for why that is exact
    at or below ``index_topk`` tokens.

    The checkpoint has ``qk_rope_head_dim == 0``. ATOM's MLA still splits q/k
    into nope+rope, so the rope half is an empty slice and the rotary is the
    identity (``_NoPositionalRotaryEmbedding``).
    """

    def __init__(
        self,
        atom_config: Config,
        layer_num: int,
        prefix: str = "",
    ) -> None:
        super().__init__()
        config = _text_config(atom_config.hf_config)
        quant_config = atom_config.quant_config
        self.config = config
        self.layer_num = layer_num

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        tp_size = get_tensor_model_parallel_world_size()
        assert self.num_heads % tp_size == 0
        self.num_local_heads = self.num_heads // tp_size

        self.q_lora_rank = config.q_lora_rank
        self.kv_lora_rank = config.kv_lora_rank
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.qk_rope_head_dim = config.qk_rope_head_dim  # padded; zero-filled
        self.qk_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        self.v_head_dim = config.v_head_dim
        self.is_nope = bool(getattr(config, "glm5_is_nope", False))
        # The checkpoint's projections are sized for the *unpadded* widths.
        self.ckpt_qk_head_dim = self.qk_nope_head_dim + (
            0 if self.is_nope else self.qk_rope_head_dim
        )
        # Scores are scaled by the real (unpadded) head width.
        self.scaling = self.ckpt_qk_head_dim**-0.5

        self.q_a_proj = ReplicatedLinear(
            self.hidden_size,
            self.q_lora_rank,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.q_a_proj",
        )
        self.q_a_layernorm = RMSNorm(self.q_lora_rank, eps=config.rms_norm_eps)
        # MLA applies q_proj internally and splits its output into nope|rope, so
        # the NoPE zero-padding has to live in this weight rather than in the
        # activation: allocate the padded width and load the checkpoint's 256-wide
        # rows into the leading lanes of each head, leaving the rope lanes zero.
        self.q_b_proj = ColumnParallelLinear(
            self.q_lora_rank,
            self.num_heads * self.qk_head_dim,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.q_b_proj",
        )
        self.kv_a_proj_with_mqa = ReplicatedLinear(
            self.hidden_size,
            self.kv_lora_rank,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.kv_a_proj_with_mqa",
        )
        self.kv_a_layernorm = RMSNorm(self.kv_lora_rank, eps=config.rms_norm_eps)
        self.kv_b_proj = ColumnParallelLinear(
            self.kv_lora_rank,
            self.num_heads * (self.qk_nope_head_dim + self.v_head_dim),
            bias=False,
            quant_config=None,  # BF16 in the checkpoint; absorbed by MLA
            prefix=f"{prefix}.kv_b_proj",
        )
        self.o_proj = RowParallelLinear(
            self.num_heads * self.v_head_dim,
            self.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )

        self.indexer = Glm5NextIndexer(config)

        self.rotary_emb = _NoPositionalRotaryEmbedding(
            head_size=self.qk_rope_head_dim,
            rotary_dim=self.qk_rope_head_dim,
            max_position_embeddings=getattr(config, "max_position_embeddings", 4096),
            base=10000.0,
            is_neox_style=True,
            dtype=torch.bfloat16,
        )

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
            kv_cache_dtype=atom_config.kv_cache_dtype,
            layer_num=layer_num,
            use_mla=True,
            mla_modules=mla_modules,
            prefix=prefix,
        )
        self.fuse_input_norm_quant = False
        self.input_quant_prefix = f"{prefix}.q_a_proj"

    def forward(
        self, hidden_states: torch.Tensor, positions: torch.Tensor
    ) -> torch.Tensor:
        # MLA owns q_proj / kv_b_proj: hand it the q-LoRA residual and the
        # normalised compressed KV, not projected tensors.
        q_c = self.q_a_layernorm(self.q_a_proj(hidden_states))
        kv_c = self.kv_a_layernorm(self.kv_a_proj_with_mqa(hidden_states))
        # NoPE: MLA wants a rope tensor, but its width is zero here, so this
        # is an empty block that contributes nothing to the scores.
        k_pe = torch.zeros(
            kv_c.shape[0],
            self.qk_rope_head_dim,
            dtype=kv_c.dtype,
            device=kv_c.device,
        )
        return self.mla_attn(q_c, kv_c, k_pe, positions)


class Glm5NextDecoderLayer(nn.Module):
    """One hybrid layer: (mHC -> KDA or MLA) then (mHC -> MoE or dense FFN)."""

    def __init__(
        self,
        atom_config: Config,
        prefix: str,
        layer_num: int = 0,
    ) -> None:
        super().__init__()
        config = _text_config(atom_config.hf_config)
        quant_config = atom_config.quant_config
        self.config = config
        self.layer_num = layer_num

        self.is_linear_attn = layer_num in config.glm5_kda_layers
        if self.is_linear_attn:
            self.self_attn = Glm5NextKDAAttention(
                atom_config, quant_config, prefix=f"{prefix}.self_attn"
            )
        else:
            self.self_attn = Glm5NextMLAAttention(
                atom_config, layer_num, prefix=f"{prefix}.self_attn"
            )

        if layer_num >= config.first_k_dense_replace:
            self.mlp = Glm5NextMoE(config, quant_config, prefix=f"{prefix}.mlp")
        else:
            self.mlp = Glm5NextMLP(
                config,
                intermediate_size=config.intermediate_size,
                quant_config=quant_config,
                prefix=f"{prefix}.mlp",
            )

        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

        # The mHC parameters are flat on the layer in the checkpoint
        # (`layers.N.hc_attn_fn`, ...), exactly as DeepSeek-V4 declares them, so
        # own them here rather than inside the helper -- a parameter's name comes
        # from its attribute path, and burying them in a submodule would rename
        # them to `attn_hc.fn` and silently leave them at their init values.
        # All three are fp32 in the checkpoint.
        self.hc = Glm5NextHyperConnection(config)
        mix = (2 + self.hc.hc_mult) * self.hc.hc_mult
        hc_dim = self.hc.hc_mult * config.hidden_size
        for site in ("attn", "ffn"):
            setattr(
                self,
                f"hc_{site}_fn",
                atom_parameter(torch.empty(mix, hc_dim, dtype=torch.float32)),
            )
            setattr(
                self,
                f"hc_{site}_base",
                atom_parameter(torch.empty(mix, dtype=torch.float32)),
            )
            setattr(
                self,
                f"hc_{site}_scale",
                atom_parameter(torch.empty(3, dtype=torch.float32)),
            )

    def forward(self, residual: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        # --- attention site ---
        x, post, comb = self.hc.pre(
            residual,
            self.hc_attn_fn,
            self.hc_attn_scale,
            self.hc_attn_base,
            self.input_layernorm.weight,
        )
        if self.is_linear_attn:
            x = self.self_attn(x)
        else:
            x = self.self_attn(x, positions)
        residual = self.hc.post_expand(x, residual, post, comb)

        # --- FFN site ---
        x, post, comb = self.hc.pre(
            residual,
            self.hc_ffn_fn,
            self.hc_ffn_scale,
            self.hc_ffn_base,
            self.post_attention_layernorm.weight,
        )
        x = self.mlp(x)
        return self.hc.post_expand(x, residual, post, comb)


class Glm5NextModel(nn.Module):
    def __init__(self, *, atom_config: Config, prefix: str = "") -> None:
        super().__init__()
        config = _text_config(atom_config.hf_config)
        self.config = config
        self.hc_mult = int(config.hc_mult)
        self.vocab_size = config.vocab_size

        if get_pp_group().is_first_rank:
            self.embed_tokens = VocabParallelEmbedding(
                config.vocab_size, config.hidden_size
            )
        else:
            self.embed_tokens = PPMissingLayer()

        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix, layer_num=None: Glm5NextDecoderLayer(
                atom_config=atom_config, prefix=prefix, layer_num=layer_num
            ),
            prefix=f"{prefix}.layers",
            layer_num_offset=0,
        )

        if get_pp_group().is_last_rank:
            self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.norm = PPMissingLayer()

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **model_kwargs: dict[str, Any],
    ) -> torch.Tensor | IntermediateTensors:
        if get_pp_group().is_first_rank:
            hidden = (
                inputs_embeds
                if inputs_embeds is not None
                else self.embed_input_ids(input_ids)
            )
            # Widen into the mHC residual: every stream starts as the embedding.
            residual = hidden.unsqueeze(-2).expand(-1, self.hc_mult, -1).contiguous()
        else:
            assert intermediate_tensors is not None
            residual = intermediate_tensors["residual"]

        for layer in islice(self.layers, self.start_layer, self.end_layer):
            residual = layer(residual, positions)

        if not get_pp_group().is_last_rank:
            return IntermediateTensors({"residual": residual})

        # GLM collapses the streams with an unweighted mean (DeepSeek-V4 uses a
        # learned reduction here; this one has no parameters).
        return self.norm(residual.mean(dim=-2))

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        return FusedMoE.make_expert_params_mapping(
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=self.config.n_routed_experts
            + (self.config.n_shared_experts or 0),
        )


class Glm5NextForConditionalGeneration(nn.Module):
    """GLM-5.3-Flash.

    The checkpoint nests the language model under ``model.language_model.*`` and
    the vision tower under ``model.visual.*``. Both are built here; the tower is
    skipped only when there is no `vision_config` to build it from, or on a
    pipeline rank that does not produce token embeddings.
    """

    packed_modules_mapping: ClassVar[dict] = {
        ".gate_proj": (".gate_up_proj", 0),
        ".up_proj": (".gate_up_proj", 1),
        **_KDA_PACKED_MODULES_MAPPING,
    }
    # Strip the multimodal nesting. The language model's parameter names are
    # rooted at `model.*` and the tower's at `visual.*`, which also lines the
    # language names up with the checkpoint's quantization_config (it already
    # writes exclusions as `model.layers.N....`), so no quant name mapping is
    # needed. Order matters only in that neither key is a substring of the other.
    weights_mapping: ClassVar[dict[str, str]] = {
        "model.visual.": "visual.",
        "model.language_model.": "model.",
    }
    # Prefixes dropped when the tower is not built; set per-instance in __init__.
    # The MTP draft layer needs no entry: it is checkpoint layer 45 and the
    # loader drops any layer index at or beyond `num_hidden_layers`.
    vision_weight_prefixes: ClassVar[tuple[str, ...]] = ("model.visual.",)
    skip_weight_prefixes: ClassVar[list[str]] = []

    fall_back_to_pt_during_load = False

    def __init__(self, atom_config: Config, prefix: str = "") -> None:
        super().__init__()
        config = _text_config(atom_config.hf_config)
        _normalize_glm5_next_config(config)

        self.atom_config = atom_config
        self.config = config
        self.quant_config = atom_config.quant_config

        self.model = Glm5NextModel(
            atom_config=atom_config, prefix=maybe_prefix(prefix, "model")
        )
        if get_pp_group().is_last_rank:
            self.lm_head = ParallelLMHead(
                config.vocab_size,
                config.hidden_size,
                quant_config=None,  # BF16 in the checkpoint
                prefix=maybe_prefix(prefix, "lm_head"),
            )
        else:
            self.lm_head = PPMissingLayer()

        self.moe_mlp_layers = [
            layer.mlp
            for layer in self.model.layers
            if isinstance(layer, Glm5NextDecoderLayer)
            and isinstance(layer.mlp, Glm5NextMoE)
        ]
        self.moe_layers = [m.experts for m in self.moe_mlp_layers]
        self.expert_weights = []

        self._init_vision_tower(atom_config)

    def _init_vision_tower(self, atom_config: Config) -> None:
        """Build the vision tower, or arrange for its weights to be skipped.

        The tower only runs where token embeddings are produced, so under
        pipeline parallelism it lives on the first rank alone. It is also skipped
        when the config carries no `vision_config` -- the text path stays fully
        usable in that case rather than failing to start.
        """
        multimodal = getattr(atom_config, "multimodal_config", None)
        vision_config = getattr(multimodal, "vision_config", None)

        if vision_config is None or not get_pp_group().is_first_rank:
            self.visual = None
            self.image_token_id = None
            self.video_token_id = None
            self.skip_weight_prefixes = list(self.vision_weight_prefixes)
            return

        from atom.models.glm5_next_vl import build_vision_tower

        self.visual = build_vision_tower(vision_config)
        self.image_token_id = getattr(multimodal, "image_token_id", None)
        self.video_token_id = getattr(multimodal, "video_token_id", None)

    @property
    def has_vision_tower(self) -> bool:
        return getattr(self, "visual", None) is not None

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def get_vision_embeddings(
        self, pixel_values: torch.Tensor, grid_thw: torch.Tensor
    ) -> torch.Tensor:
        """Patches -> one 4096-wide embedding per merged 2x2 block."""
        if not self.has_vision_tower:
            raise RuntimeError(
                "GLM-5.3-Flash vision embeddings were requested but no tower is "
                "built. Either the config carried no `vision_config`, or this is "
                "a pipeline rank other than the first."
            )
        return self.visual(pixel_values, grid_thw)

    def merge_multimodal_embeddings(
        self,
        input_ids: torch.Tensor,
        inputs_embeds: torch.Tensor,
        vision_embeds: torch.Tensor,
    ) -> torch.Tensor:
        """Scatter vision embeddings onto the image/video placeholder tokens."""
        mask = torch.zeros_like(input_ids, dtype=torch.bool)
        for token_id in (self.image_token_id, self.video_token_id):
            if token_id is not None:
                mask |= input_ids == token_id
        n_slots = int(mask.sum())
        if n_slots != vision_embeds.shape[0]:
            raise ValueError(
                f"GLM-5.3-Flash got {vision_embeds.shape[0]} vision embeddings for "
                f"{n_slots} placeholder tokens. Each image contributes "
                "(t*h*w)/spatial_merge_size**2 tokens, and multimodal prefills "
                "must not be chunked."
            )
        inputs_embeds[mask] = vision_embeds.to(inputs_embeds.dtype)
        return inputs_embeds

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **model_kwargs: dict[str, Any],
    ) -> torch.Tensor | IntermediateTensors:
        return self.model(
            input_ids, positions, intermediate_tensors, inputs_embeds, **model_kwargs
        )

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        return self.lm_head(hidden_states)

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        return self.model.get_expert_mapping()

    @staticmethod
    def get_spec_layer_idx_from_weight_name(config, weight_name: str) -> int | None:
        """Checkpoint layer 45 is the MTP draft layer, which is not served yet."""
        n_predict = int(getattr(config, "num_nextn_predict_layers", 0) or 0)
        if n_predict > 0:
            base = config.num_hidden_layers
            for i in range(n_predict):
                if f"layers.{base + i}." in weight_name:
                    return base + i
        return None
