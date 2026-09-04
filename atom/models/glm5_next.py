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

Three decisions worth knowing about, each exact rather than approximate:

1. **DSA layers select at POOL granularity.** ``index_topk=2048`` with
   ``index_kpool=4`` ranks ``2048/4 = 512`` pooled candidates covering 2048
   tokens, and each selected pool expands back to its ``index_kpool`` token
   positions. At or below 2048 tokens top-k selects every pool, so the expansion
   yields every position and dense causal MLA is numerically identical -- which
   is what ATOM's own gate relies on
   (``deepseek_v2._pcp_sparse_active``: ``max_seqlen_k > index_topk``). Past that
   threshold the pooled path decides what is attended to; dispatch lives in
   ``model_ops.glm5_next.indexer`` and its kernels in
   ``model_ops.glm5_next.kpool``.

2. **NoPE runs on a 64-wide block of zeros.** The text model is entirely NoPE
   (``qk_rope_head_dim == 0``), but the ROCm MLA kernels assume the DeepSeek
   576-wide KV entry, so the rope block is materialized at ``_ROPE_PAD`` lanes
   and held at zero and the rotary is the identity
   (``NoPositionalRotaryEmbedding``). See ``_ROPE_PAD`` below for why a
   zero-WIDTH slice does not work.

3. **The KDA output gate is folded at load.** GLM's gate is low-rank
   (``g_b_proj @ g_a_proj``) where Kimi-K3's is a single ``g_proj``. Both are
   linear with nothing in between, so the product is materialised once after
   load and written into the fused ``in_proj``'s ``g`` shard. That makes
   ``KimiKDAAttention`` -- and all of its state-cache, TP and CUDA-graph
   integration -- reusable unchanged.

Not yet wired: the MTP draft layer (checkpoint layer 45) or multimodal input.
The checkpoint's unreachable vision tower is skipped on the text-only path.
See ``recipes/GLM-5.3-Flash.md``.
"""

from itertools import islice
from typing import Any, ClassVar

import aiter
import torch
import torch.nn.functional as F
from aiter import dtypes
from aiter.dist.communication_op import tensor_model_parallel_all_reduce
from aiter.dist.parallel_state import (
    get_ep_group,
    get_pp_group,
    get_tensor_model_parallel_world_size,
    get_tp_group,
)
from torch import nn

from atom.config import Config, QuantizationConfig, get_current_atom_config
from atom.model_ops.attention_mla import MLAModules
from atom.model_ops.base_attention import Attention
from atom.model_ops.embed_head import ParallelLMHead, VocabParallelEmbedding
from atom.model_ops.glm5_next import geometry as kpool_geometry
from atom.model_ops.glm5_next import indexer as _glm5_indexer_ops  # noqa: F401
from atom.model_ops.glm5_next import kpool as kpool_ops
from atom.model_ops.layernorm import RMSNorm
from atom.model_ops.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from atom.model_ops.moe import FusedMoE
from atom.model_ops.rotary_embedding import NoPositionalRotaryEmbedding
from atom.model_ops.sparse_attn_v4 import hc_split_sinkhorn
from atom.model_ops.swiglu_oai import swiglu_oai_split
from atom.model_ops.topK import (
    is_rocm_aiter_fuse_routed_scaling_factor,
    is_rocm_aiter_fusion_shared_expert_enabled,
)
from atom.model_ops.utils import atom_parameter
from atom.models.deepseek_v2 import Indexer
from atom.models.kimi_k3 import (
    KimiKDAAttention,
    _text_config,
)
from atom.models.utils import (
    IntermediateTensors,
    PPMissingLayer,
    make_layers,
    maybe_prefix,
)
from atom.utils import envs
from atom.utils.decorators import support_torch_compile

# GLM-5.3-Flash's MLA is NoPE (`qk_rope_head_dim == 0`), but the ROCm stack
# assumes DeepSeek's geometry throughout: ATOM allocates the paged MLA entry at
# a hard-coded 576 (`aiter_mla.py`), and aiter's asm decode kernel is built for
# a 576-wide query -- it only ASSERTS that on the gfx1250 path, and its dispatch
# table never keys on head_size, so on gfx950 a 512-wide query against a
# 576-wide entry is silently mis-computed rather than rejected. Triton is no
# better off: a zero `KV_PeDim` turns every `tl.arange(0, KV_PeDim)` into
# `arange(0, 0)`, which Triton refuses at compile time, and upstream aiter's
# `gather_kv_b_proj` kernels carry no guard for it -- so a zero-width rope also
# crashes outright under chunked prefill.
#
# So the rope block is materialized at 64 lanes and held at **zero**. That is
# bit-for-bit NoPE -- a zero block contributes `sum(0*0) == 0` to every QK dot
# product -- and it makes the latent/cache side exactly the 576 those kernels
# expect.
#
# The catch, and why this needs care rather than a blanket pad: the padding is
# only valid on the LATENT side. `qk_nope_head_dim` is already 256, so a padded
# per-head query would be 320 and CK's flash-attention caps head_dim at 256.
# Kimi-K3 never hits this because its `qk_nope_head_dim` is 128 (128+64=192).
# The two constraints apply to DIFFERENT tensors, so both are satisfiable:
# `MLAModules.rope_is_zero_pad` makes the MLA drop the zero lanes at every
# flash-attention site, which is exact, and each side sees the width it needs:
#     latent / KV cache / decode : kv_lora_rank + 64 = 576
#     per-head qk / prefill      : qk_nope_head_dim  = 256
_ROPE_PAD = 64


def _normalize_glm5_next_config(config) -> None:
    """Fill the aliases the shared ATOM MoE / KDA / MLA infrastructure expects.

    The text config arrives as a bare ``PretrainedConfig`` (this image's
    transformers has no ``Glm5NextTextConfig``), so everything is a plain attr.
    """
    lin = getattr(config, "linear_attn_config", {}) or {}
    if not isinstance(lin, dict):
        lin = vars(lin)

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
    layer_types = list(getattr(config, "layer_types", []) or [])
    all_layers = set(range(int(config.num_hidden_layers)))
    if not config.glm5_kda_layers:
        config.glm5_kda_layers = (
            [i for i, kind in enumerate(layer_types) if kind == "linear_attention"]
            if layer_types
            else sorted(all_layers - set(config.glm5_full_attn_layers))
        )
    if not config.glm5_full_attn_layers:
        config.glm5_full_attn_layers = (
            [i for i, kind in enumerate(layer_types) if kind != "linear_attention"]
            if layer_types
            else sorted(all_layers - set(config.glm5_kda_layers))
        )

    kda_layers = set(config.glm5_kda_layers)
    full_layers = set(config.glm5_full_attn_layers)
    if (
        len(kda_layers) != len(config.glm5_kda_layers)
        or len(full_layers) != len(config.glm5_full_attn_layers)
        or kda_layers & full_layers
        or kda_layers | full_layers != all_layers
    ):
        raise ValueError(
            "GLM-5.3 attention layers must form a disjoint, complete partition"
        )
    if layer_types:
        typed_kda = {
            i for i, kind in enumerate(layer_types) if kind == "linear_attention"
        }
        if len(layer_types) != config.num_hidden_layers or typed_kda != kda_layers:
            raise ValueError("GLM-5.3 linear_attn_config disagrees with layer_types")

    config.num_gdn_attn_state = len(config.glm5_kda_layers)
    config.num_full_attn = len(config.glm5_full_attn_layers)
    # KimiKDAAttention keys its layer membership off this name.
    config.kimi_kda_layers = config.glm5_kda_layers
    config.kimi_full_attn_layers = config.glm5_full_attn_layers

    # --- MoE aliases ---
    config.num_experts = getattr(config, "n_routed_experts", None)
    config.moe_layer_freq = getattr(config, "moe_layer_freq", 1)

    # --- MLA aliases (this implementation is intentionally NoPE-only) ---
    if int(getattr(config, "qk_rope_head_dim", 0)) != 0:
        raise ValueError("GLM-5.3 requires qk_rope_head_dim == 0")
    if int(getattr(config, "index_head_dim", 0)) != kpool_ops.INDEX_HEAD_DIM:
        raise ValueError(
            f"GLM-5.3 requires index_head_dim == {kpool_ops.INDEX_HEAD_DIM}"
        )
    index_kpool = int(getattr(config, "index_kpool", 1) or 1)
    index_topk = int(config.index_topk)
    if index_kpool < 1 or index_topk % index_kpool:
        raise ValueError("GLM-5.3 requires index_topk divisible by index_kpool")
    if index_kpool > 1 and not bool(
        getattr(config, "index_kpool_always_select_tail", True)
    ):
        raise ValueError("GLM-5.3 requires index_kpool_always_select_tail=true")
    # Leave `qk_rope_head_dim` at its true 0 so the indexer stays NoPE; the MLA
    # attention pads to `_ROPE_PAD` locally (see the module docstring).
    config.head_dim = config.qk_nope_head_dim + config.qk_rope_head_dim
    # The KV cache entry must be sized for the PADDED rope block, not the
    # checkpoint's 0. The backends size the paged pool off this
    # (`aiter_mla.mla_kv_entry_dim`); deriving it from the raw config instead
    # allocates 512-wide rows under a 576-wide write.
    config.mla_kv_entry_dim = config.kv_lora_rank + _ROPE_PAD

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
        # ATOM_GLM5_DISABLE_FUSED_MHC=1 forces the torch reference path, for
        # bisecting numerical differences against transformers.
        dim_ok = config.hidden_size % 512 == 0 or config.hidden_size % 256 == 0
        if envs.ATOM_GLM5_DISABLE_FUSED_MHC:
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
        # Coverage is tracked per packed shard: the loader's parameter-level
        # report cannot distinguish "q loaded" from "q/k/v all loaded" because
        # all three checkpoint tensors target the same in_proj parameter.
        self._loaded_input_shards: set[int] = set()
        base_loader = self.in_proj.weight.weight_loader

        def record_input_shard(param, loaded_weight, shard_id=None):
            if shard_id is not None:
                self._loaded_input_shards.add(int(shard_id))
            return base_loader(param, loaded_weight, shard_id)

        self.in_proj.weight.weight_loader = record_input_shard
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

    def process_weights_after_loading(self) -> None:
        if getattr(self, "_in_proj_fused", False):
            return
        # `--load_dummy` never runs a weight_loader, so the shard set is empty
        # by design and this coverage check would refuse to start on all 34 KDA
        # layers. The check exists to catch a checkpoint that silently supplied
        # only part of q/k/v; with no checkpoint there is nothing to catch. The
        # gate fold below still runs, so the layer keeps the same shape and
        # dtype it would have had.
        if not getattr(get_current_atom_config(), "load_dummy", None):
            missing = {0, 1, 2} - self._loaded_input_shards
            if missing:
                raise RuntimeError(
                    "Incomplete GLM-5.3 KDA input projection: missing "
                    f"checkpoint shards {sorted(missing)} from q/k/v"
                )
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


class Glm5NextIndexer(Indexer):
    """Sparse-attention indexer for GLM-5.3-Flash.

    Two differences from the DeepSeek-V3.2 / GLM-5.2 indexer this extends:

    **NoPE.**  ``qk_rope_head_dim == 0``, so there is no rope component to split
    off and rotate.  ``forward_impl`` is overridden to skip the split/rotate
    entirely -- splitting to a 0-wide ``q_pe`` and calling the rope kernel on it
    is at best a no-op and at worst an out-of-bounds read.

    **kpool.**  The indexer K cache stores one *pooled* entry per
    ``index_kpool`` tokens: a per-dimension softmax over the pool's slots,
    weighted by ``index_kpool_compress_gate`` plus a per-slot bias
    ``index_kpool_compress_ape``, then Hadamard-rotated and FP8-quantized.
    Top-k then runs at pool granularity (``index_topk // index_kpool`` pools)
    and each selected pool expands back to its ``index_kpool`` token positions,
    with the trailing incomplete pool always selected.
    """

    def __init__(
        self,
        atom_config,
        config,
        hidden_size: int,
        q_lora_rank: int,
        quant_config,
        cache_config,
        use_wk_weights_proj_fusion: bool = False,
        prefix: str = "",
    ) -> None:
        super().__init__(
            atom_config,
            config,
            hidden_size,
            q_lora_rank,
            quant_config,
            cache_config,
            use_wk_weights_proj_fusion,
            prefix,
        )
        self.index_kpool = int(getattr(config, "index_kpool", 1) or 1)
        always_select_tail = bool(
            getattr(config, "index_kpool_always_select_tail", True)
        )
        if self.index_kpool > 1 and not always_select_tail:
            raise NotImplementedError(
                "GLM-5.3 kpool currently always appends the uncompressed tail; "
                "index_kpool_always_select_tail=false would require scoring it "
                "instead and cannot be silently ignored."
            )
        if self.index_kpool > 1:
            # Pool-compression parameters. Held so the checkpoint loads
            # completely and so the pooled path can be switched on without a
            # second weight-loading pass.
            self.index_kpool_compress_ape = atom_parameter(
                torch.zeros(self.index_kpool, self.head_dim, dtype=torch.float32)
            )
            self.index_kpool_compress_gate = ReplicatedLinear(
                hidden_size,
                self.head_dim,
                bias=False,
                quant_config=None,
                prefix=f"{prefix}.index_kpool_compress_gate",
            )
        # Bound by the metadata builder alongside the index K cache.
        self.kpool_tail_cache: torch.Tensor | None = None
        # One helper owns the producer/metadata width contract, including the
        # ATOM_GLM5_KPOOL off-switch.
        self.topk_out_width = kpool_geometry.topk_output_width(
            self.topk_tokens, self.index_kpool
        )

    def use_kpool(self) -> bool:
        """Whether to run the pooled indexer.

        ``ATOM_GLM5_KPOOL`` switches the pooled write, the pooled scoring and
        the pooled selection together, so the two settings are genuinely
        independent implementations of the same selection -- which is what
        makes the short-context A/B a check and not a tautology.
        """
        return kpool_geometry.pooled_path_enabled(self.index_kpool)

    def forward_impl(
        self,
        hidden_states: torch.Tensor,
        qr: torch.Tensor,
        qr_scale: torch.Tensor | None,
        positions,
        rotary_emb=None,
    ) -> torch.Tensor:
        from atom.utils.forward_context import get_forward_context

        forward_context = get_forward_context()
        attn_metadata = forward_context.attn_metadata

        if rotary_emb is None:
            rotary_emb = self.rotary_emb
        q = self.wq_b(qr, qr_scale)
        q = q.view(-1, self.n_head, self.head_dim)
        k = self.wk(hidden_states)
        weights = self.weights_proj(hidden_states)

        # NoPE: no rope split, no rotation -- q and k are entirely "nope".
        k = self.k_norm(k)
        q = q.view(-1, self.head_dim)
        kpool = self.use_kpool()
        if kpool:
            # The pooled keys are cached Hadamard-rotated, so the query has to
            # be rotated by the same orthonormal transform or the dot products
            # mean nothing. Fused with the FP8 quant to avoid a round trip.
            q_fp8, q_scale = kpool_ops.fwht128_quant_fp8(q)
        else:
            q_fp8, q_scale = self.quant_func(q, quant_dtype=dtypes.fp8)
        q_fp8 = q_fp8.view(-1, self.n_head, self.head_dim)
        q_scale = q_scale.view(-1, self.n_head, 1)
        weights = (weights.unsqueeze(-1) * q_scale * self._weights_scale).squeeze(-1)

        if kpool:
            # The memory-profiling forward runs BEFORE the KV cache (and the
            # tail buffer with it) exists, so "unbound" is expected there and
            # only there -- the op discards dummy runs before touching either.
            # Anywhere else it means this model is on a builder that never
            # allocated the buffer, which must not be papered over.
            tail_cache = self.kpool_tail_cache
            if tail_cache is None:
                if not forward_context.context.is_dummy_run:
                    raise RuntimeError("kpool tail cache is unbound on a real forward")
                tail_cache = torch.zeros(
                    1,
                    2,
                    self.index_kpool,
                    self.head_dim,
                    dtype=torch.bfloat16,
                    device=hidden_states.device,
                )
            gdn = getattr(attn_metadata, "gdn_metadata", None)
            state_slot_idx = None if gdn is None else gdn.non_spec_state_indices_tensor
            state_slot_idx_in = (
                None if gdn is None else gdn.non_spec_state_indices_in_tensor
            )
            if state_slot_idx is None:
                # The profile/warmup forward carries no block tables, so the
                # builder leaves gdn_metadata unset. That forward is a dummy
                # run and the op discards it before touching any state; the
                # placeholder only has to exist. CUDAGraph capture DOES get
                # real slots (`_build_gdn_capture_metadata`), and it must --
                # capture bakes this pointer in, so a zeros stand-in there
                # would send every request's tail to slot 0 on replay.
                if not forward_context.context.is_dummy_run:
                    raise RuntimeError(
                        "kpool state-slot metadata is missing on a real forward"
                    )
                state_slot_idx = torch.zeros(
                    hidden_states.shape[0],
                    dtype=torch.int32,
                    device=hidden_states.device,
                )
            if state_slot_idx_in is None:
                state_slot_idx_in = state_slot_idx
            out = torch.ops.aiter.sparse_attn_indexer_kpool(
                hidden_states,
                self.k_cache.kv_cache[0],
                q_fp8,
                k,
                # The gate that drives the pooling softmax comes from the same
                # hidden states that produced k, so it stays token-aligned.
                self.index_kpool_compress_gate(hidden_states),
                weights,
                self.index_kpool_compress_ape,
                tail_cache,
                state_slot_idx_in,
                state_slot_idx,
                positions,
                self.sparse_kv_indices_buffer,
                self.topk_tokens,
                self.index_kpool,
                self.head_dim,
                self.max_model_len,
                self.topk_out_width,
                self.scale_fmt,
                self.stable_topk,
            )
            return out

        out = self.sparse_attn_indexer_impl(
            hidden_states,
            self.k_cache.prefix,
            self.k_cache.kv_cache[0],
            q_fp8,
            k,
            weights,
            self.quant_block_size,
            self.scale_fmt,
            self.topk_tokens,
            self.head_dim,
            self.max_model_len,
            self.max_total_seq_len,
            self.sparse_kv_indices_buffer,
            self.dcp_sparse_kv_indptr_buffer,
            self.dcp_owned_counts_buffer,
            self.k_norm.weight,
            self.k_norm.bias,
            self.k_norm.eps,
            positions,
            rotary_emb.cos_cache.squeeze(-2).squeeze(-2),
            rotary_emb.sin_cache.squeeze(-2).squeeze(-2),
            self._weights_scale,
            rotary_emb.is_neox_style,
            # NoPE always takes the unfused path; the cos/sin caches above are
            # passed only to satisfy the op signature and are never read.
            False,
            self.stable_topk,
        )
        return out


class _ZeroRopePad:
    """Appends ``pad`` zero lanes per head to a projection's output.

    Deliberately NOT an ``nn.Module``: wrapping the Linear in one would insert a
    level into the parameter path (``q_b_proj.inner.weight``), which no longer
    matches the checkpoint and leaves the weights at their init values -- a
    silent, hard-to-spot failure. This holds only a reference, so the wrapped
    Linear stays registered under its own name on the attention module.
    """

    def __init__(self, inner, num_heads: int, nope_dim: int, pad: int) -> None:
        self.inner = inner
        self.num_heads = num_heads
        self.nope_dim = nope_dim
        self.pad = pad

    def __call__(self, x, x_scale=None):
        y = self.inner(x, x_scale) if x_scale is not None else self.inner(x)
        if isinstance(y, tuple):
            y = y[0]
        if self.pad == 0:
            return y
        tokens = y.shape[0]
        out = y.new_zeros(tokens, self.num_heads, self.nope_dim + self.pad)
        out[..., : self.nope_dim] = y.view(tokens, self.num_heads, self.nope_dim)
        return out.view(tokens, self.num_heads * (self.nope_dim + self.pad))


class Glm5NextMLAAttention(nn.Module):
    """NoPE MLA with the k-pool sparse indexer. Attention runs dense at or below
    ``index_topk`` tokens -- see the module docstring for why that is exact --
    and on the pooled selection past it.

    The checkpoint has ``qk_rope_head_dim == 0``. ATOM's MLA still splits q/k
    into nope+rope, so the rope half is materialized at ``_ROPE_PAD`` lanes and
    held at zero, and the rotary is the identity
    (``NoPositionalRotaryEmbedding``). See ``_ROPE_PAD``.
    """

    def __init__(
        self,
        atom_config: Config,
        layer_num: int,
        rotary_emb: nn.Module,
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
        # The WIDENED rope block: 0 + `_ROPE_PAD` = 64, not the checkpoint's true
        # rope width of 0. Everything the MLA kernels size off this -- the KV
        # entry (512 + 64 = 576) and the per-head q/k -- sees the padded width;
        # the pad lanes are identically zero. See `_ROPE_PAD` above.
        self.qk_rope_head_dim = config.qk_rope_head_dim + _ROPE_PAD
        self.qk_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        self.v_head_dim = config.v_head_dim
        # The checkpoint's projections are sized for the *unpadded* widths.
        self.ckpt_qk_head_dim = self.qk_nope_head_dim + config.qk_rope_head_dim
        # Scores are scaled by the real (unpadded) head width -- the zero pad
        # contributes nothing to a QK dot product and must not change the norm.
        self.scaling = self.ckpt_qk_head_dim**-0.5

        self.q_a_proj = ReplicatedLinear(
            self.hidden_size,
            self.q_lora_rank,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.q_a_proj",
        )
        self.q_a_layernorm = RMSNorm(self.q_lora_rank, eps=config.rms_norm_eps)
        # Allocated at the CHECKPOINT width (256 per head), so the parameter
        # matches the checkpoint tensor exactly and loads normally. The rope pad
        # is appended at call time by `_ZeroRopePad` below rather than baked into
        # this weight, which keeps the parameter path unchanged -- see that
        # class for why a wrapper `nn.Module` would silently break loading.
        self.q_b_proj = ColumnParallelLinear(
            self.q_lora_rank,
            self.num_heads * self.ckpt_qk_head_dim,
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

        self.indexer = Glm5NextIndexer(
            atom_config,
            config,
            self.hidden_size,
            self.q_lora_rank,
            None,  # the whole indexer is BF16 in modules_to_not_convert
            atom_config.kv_cache_dtype,
            False,  # GLM-5.3 ships wk / weights_proj unfused
            f"{prefix}.indexer",
        )

        self.rotary_emb = rotary_emb
        self.indexer.rotary_emb = self.rotary_emb

        # ATOM_GLM5_FORCE_DENSE_MLA=1 runs MLA with sparsity off. Below
        # `index_topk` candidates that is not an approximation -- top-k would
        # select every token anyway -- so any output difference isolates a bug
        # in the indexer / sparse top-k path rather than in MLA itself.
        force_dense = envs.ATOM_GLM5_FORCE_DENSE_MLA
        # The indexer is not called by MLA -- it has to be driven from this
        # forward. It writes the pooled index keys and the selected KV slots
        # that `is_sparse=True` then makes MLA read, so leaving it uncalled
        # gives the sparse path an empty selection buffer rather than an error.
        self.run_indexer = not force_dense

        mla_modules = MLAModules(
            q_lora_rank=self.q_lora_rank,
            kv_lora_rank=self.kv_lora_rank,
            qk_nope_head_dim=self.qk_nope_head_dim,
            qk_rope_head_dim=self.qk_rope_head_dim,
            qk_head_dim=self.qk_head_dim,
            v_head_dim=self.v_head_dim,
            rotary_emb=self.rotary_emb,
            q_proj=_ZeroRopePad(
                self.q_b_proj, self.num_local_heads, self.qk_nope_head_dim, _ROPE_PAD
            ),
            rope_is_zero_pad=_ROPE_PAD > 0,
            kv_b_proj=self.kv_b_proj,
            o_proj=self.o_proj,
            indexer=None if force_dense else self.indexer,
            is_sparse=not force_dense,
            topk_tokens=config.index_topk,
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
        # NoPE: the key's rope block is `_ROPE_PAD` lanes of zeros. It is stored
        # in the KV entry so the cache is the 576 the kernels expect, and it
        # contributes `sum(0*0) == 0` to every score.
        k_pe = torch.zeros(
            kv_c.shape[0],
            self.qk_rope_head_dim,
            dtype=kv_c.dtype,
            device=kv_c.device,
        )
        # Drive the indexer before MLA: it writes this layer's pooled index
        # keys and the selected KV slots that the sparse MLA path then reads.
        if self.run_indexer:
            self.indexer(hidden_states, q_c, None, positions, self.rotary_emb)
        return self.mla_attn(q_c, kv_c, k_pe, positions)


class Glm5NextDecoderLayer(nn.Module):
    """One hybrid layer: (mHC -> KDA or MLA) then (mHC -> MoE or dense FFN)."""

    def __init__(
        self,
        atom_config: Config,
        prefix: str,
        layer_num: int = 0,
        rotary_emb: nn.Module | None = None,
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
            if rotary_emb is None:
                raise ValueError("MLA layers require the shared NoPE rotary module")
            self.self_attn = Glm5NextMLAAttention(
                atom_config,
                layer_num,
                rotary_emb,
                prefix=f"{prefix}.self_attn",
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


@support_torch_compile
class Glm5NextModel(nn.Module):
    def __init__(self, *, atom_config: Config, prefix: str = "") -> None:
        super().__init__()
        config = _text_config(atom_config.hf_config)
        self.config = config
        self.hc_mult = int(config.hc_mult)
        self.vocab_size = config.vocab_size
        self.rotary_emb = NoPositionalRotaryEmbedding(
            head_size=_ROPE_PAD,
            rotary_dim=_ROPE_PAD,
            max_position_embeddings=int(config.max_position_embeddings),
            base=10000.0,
            is_neox_style=True,
            dtype=torch.bfloat16,
        )

        if get_pp_group().is_first_rank:
            self.embed_tokens = VocabParallelEmbedding(
                config.vocab_size, config.hidden_size
            )
        else:
            self.embed_tokens = PPMissingLayer()

        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix, layer_num=None: Glm5NextDecoderLayer(
                atom_config=atom_config,
                prefix=prefix,
                layer_num=layer_num,
                rotary_emb=self.rotary_emb,
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

    def make_empty_intermediate_tensors(
        self,
        batch_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> IntermediateTensors:
        return IntermediateTensors(
            {
                "residual": torch.zeros(
                    batch_size,
                    self.hc_mult,
                    self.config.hidden_size,
                    dtype=dtype,
                    device=device,
                )
            }
        )

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
    """Text-only GLM-5.3-Flash runtime."""

    packed_modules_mapping: ClassVar[dict] = {
        ".gate_proj": (".gate_up_proj", 0),
        ".up_proj": (".gate_up_proj", 1),
        **_KDA_PACKED_MODULES_MAPPING,
    }
    # Strip the multimodal language nesting. The checkpoint's vision tower has
    # no processor/input path in ATOM yet, so it is skipped rather than
    # consuming VRAM with unreachable randomly-risky code.
    weights_mapping: ClassVar[dict[str, str]] = {
        "model.language_model.": "model.",
        # The checkpoint stores this projection as a bare matrix parameter.
        "index_kpool_compress_gate": "index_kpool_compress_gate.weight",
    }
    # The MTP draft layer needs no entry: it is checkpoint layer 45 and the
    # loader drops any layer index at or beyond `num_hidden_layers`.
    skip_weight_prefixes: ClassVar[list[str]] = ["model.visual."]

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
        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors
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

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

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


class Glm5NextMultimodalModel(nn.Module):
    """GLM-5.3-Flash language model plus its native image/video encoder."""

    packed_modules_mapping = Glm5NextForConditionalGeneration.packed_modules_mapping
    hf_to_atom_mapper: ClassVar[dict[str, str]] = {
        "model.visual.": "visual.",
        "model.language_model.": "language_model.model.",
        "lm_head.": "language_model.lm_head.",
    }
    quant_exclude_name_mapping = {
        "model.visual.": "visual.",
        "model.": "language_model.model.",
        "lm_head.": "language_model.lm_head.",
    }
    weights_mapping: ClassVar[dict[str, str]] = {
        "index_kpool_compress_gate": "index_kpool_compress_gate.weight",
    }
    fall_back_to_pt_during_load = False

    def __init__(self, atom_config: Config, prefix: str = "") -> None:
        super().__init__()
        from atom.models.glm5_next_vl import Glm5NextVisionTransformer

        mm_config = atom_config.multimodal_config
        if mm_config is None:
            raise ValueError("GLM-5.3 multimodal serving requires multimodal_config")
        self.visual = Glm5NextVisionTransformer(mm_config.vision_config)
        self.language_model = Glm5NextForConditionalGeneration(
            atom_config, prefix=maybe_prefix(prefix, "language_model")
        )
        self.image_token_id = int(getattr(mm_config, "image_token_id", 154854))
        self.video_token_id = int(getattr(mm_config, "video_token_id", 154855))
        self.make_empty_intermediate_tensors = (
            self.language_model.make_empty_intermediate_tensors
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.language_model.embed_input_ids(input_ids)

    def get_vision_embeddings(self, pixel_values, grid_thw):
        return self.visual(pixel_values, grid_thw)

    def merge_multimodal_embeddings(self, input_ids, inputs_embeds, vision_embeds):
        mask = (input_ids == self.image_token_id) | (input_ids == self.video_token_id)
        if int(mask.sum()) != vision_embeds.shape[0]:
            raise ValueError(
                f"GLM vision produced {vision_embeds.shape[0]} embeddings for "
                f"{int(mask.sum())} image/video placeholder tokens"
            )
        inputs_embeds[mask] = vision_embeds.to(inputs_embeds.dtype)
        return inputs_embeds

    def forward(
        self,
        input_ids,
        positions,
        intermediate_tensors=None,
        inputs_embeds=None,
        **model_kwargs,
    ):
        if inputs_embeds is None:
            inputs_embeds = self.embed_input_ids(input_ids)
        return self.language_model(
            input_ids, positions, intermediate_tensors, inputs_embeds, **model_kwargs
        )

    def compute_logits(self, hidden_states):
        return self.language_model.compute_logits(hidden_states)

    def get_expert_mapping(self):
        return self.language_model.get_expert_mapping()

    @staticmethod
    def get_spec_layer_idx_from_weight_name(config, weight_name):
        return Glm5NextForConditionalGeneration.get_spec_layer_idx_from_weight_name(
            config, weight_name
        )
