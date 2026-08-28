# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Inference-only GLM-5.3-Flash (``Glm5NextForConditionalGeneration``).

Ported from vLLM PR #53906.  The checkpoint is multimodal; ATOM serves the text
path here, so the vision tower under ``model.visual.*`` is skipped (same policy
as ``kimi_k3.py``).

Architecture, read off the shipped ``config.json`` / weight index rather than
the PR prose -- 45 layers plus one MTP layer that ATOM does not load:

  * 34 KDA (linear-attention) layers and 11 sparse-MLA layers, in a 3:1 pattern
    (``layer_types``).  The KDA layers are Kimi Linear, so they reuse
    ``kimi_k3.KimiKDAAttention`` -- with its low-rank output gate, since
    GLM-5.3 ships ``g_a_proj``/``g_b_proj`` rather than Kimi's full-rank
    ``g_proj``.
  * MLA is **NoPE**: ``qk_rope_head_dim == 0``, ``qk_nope_head_dim == 256``,
    ``v_head_dim == 256``, ``kv_lora_rank == 512``.  See `_ROPE_PAD` below.
  * The MLA layers carry a sparse indexer whose K cache is *pooled*
    ``index_kpool``-to-1 (``atom/model_ops/glm5_next/kpool.py``).
  * The residual stream is a Manifold-Constrained Hyper-Connections stack of
    ``hc_mult`` streams (``atom/model_ops/mhc.py``), exactly the mechanism
    DeepSeek-V4 uses, down to the ``hc_attn_*`` / ``hc_ffn_*`` tensor names.
  * MoE: 288 routed + 1 shared expert, sigmoid routing with ``noaux_tc`` bias
    and a clamped SwiGLU (``swiglu_limit``).

Quantization follows the checkpoint exactly: only the MLA ``q_a``/``q_b`` /
``kv_a_proj_with_mqa`` / ``o_proj`` and every MoE / dense-MLP projection carry
``weight_scale_inv`` (FP8 block 128x128).  Everything else -- all KDA
projections, the whole indexer, ``kv_b_proj``, ``mlp.gate``, the norms and all
mHC parameters -- is BF16 and is listed in ``modules_to_not_convert``.  The MLA
projections are additionally forced to BF16 here (see `Glm5NextMLAAttention`).
"""

from __future__ import annotations

import os
from typing import ClassVar

import torch
from aiter import dtypes
from aiter import silu_and_mul as aiter_silu_and_mul
from aiter.dist.parallel_state import (
    get_pp_group,
    get_tensor_model_parallel_world_size,
)
from aiter.ops.cache import (
    cp_gather_indexer_k_quant_cache,
    indexer_k_quant_and_cache,
)
from aiter.ops.topk import top_k_per_row_decode, top_k_per_row_prefill
from aiter.ops.triton.attention.fp8_mqa_logits import fp8_mqa_logits
from aiter.ops.triton.attention.pa_mqa_logits import deepgemm_fp8_paged_mqa_logits
from torch import nn

from atom.config import Config, QuantizationConfig
from atom.model_ops.attention_mla import (
    MLAModules,
    triton_convert_req_index_to_global_index,
    triton_convert_req_index_to_global_index_dsa_prefill,
)
from atom.model_ops.base_attention import Attention
from atom.model_ops.embed_head import ParallelLMHead, VocabParallelEmbedding
from atom.model_ops.glm5_next import kpool as kpool_ops
from atom.model_ops.layernorm import RMSNorm
from atom.model_ops.linear import (
    ColumnParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from atom.model_ops.mhc import HyperConnection, MHCOps, hc_contract, hc_expand
from atom.model_ops.utils import atom_parameter
from atom.models.deepseek_v2 import (
    SPARSE_INDEXER_LOGITS_BUDGET_MB,
    DeepseekV2MLP,
    DeepseekV2MoE,
    Indexer,
)
from atom.models.kimi_k3 import KimiKDAAttention, _NoPositionalRotaryEmbedding
from atom.models.utils import (
    IntermediateTensors,
    PPMissingLayer,
    make_empty_intermediate_tensors_factory,
    make_layers,
    maybe_prefix,
)
from atom.utils.custom_register import direct_register_custom_op

# GLM-5.3-Flash's MLA is NoPE (`qk_rope_head_dim == 0`), but the ROCm stack
# assumes a rope block exists in more places than is practical to special-case:
# aiter's asm MLA decode kernel hard-codes a 576-wide head dim and does not even
# dispatch on head_size (csrc/py_itfs_cu/asm_mla.cu), and several Triton kernels
# do `tl.arange(0, KV_PeDim)`, which is a compile error at 0.
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
# `MLAModules.rope_is_zero_pad` tells the MLA to drop the zero lanes in prefill,
# which is exact, so each side sees the width it needs:
#     latent / KV cache / decode : kv_lora_rank + 64 = 576
#     per-head qk / prefill      : qk_nope_head_dim  = 256
_ROPE_PAD = 64


def _glm5_text_config(config):
    """The language sub-config -- GLM-5.3 nests it under ``text_config``."""
    return getattr(config, "text_config", config)


def _normalize_glm5_config(config) -> None:
    """Fill the aliases ATOM's shared MoE / KDA infrastructure expects.

    Mutates the *text* config in place and is idempotent.
    """
    # ---- layer layout -------------------------------------------------
    # `layer_types` is the authority: entry i is "linear_attention" (KDA) or
    # "deepseek_sparse_attention" (MLA).  `linear_attn_config.kda_layers` says
    # the same thing but Kimi-K3's copy of that field is 1-BASED while
    # GLM-5.3's is 0-based, so deriving from `layer_types` avoids the trap.
    layer_types = list(getattr(config, "layer_types", []) or [])
    if not layer_types:
        layer_types = ["deepseek_sparse_attention"] * config.num_hidden_layers
    config.layer_types = layer_types
    config.glm5_kda_layers = [
        i for i, t in enumerate(layer_types) if t == "linear_attention"
    ]
    config.glm5_mla_layers = [
        i for i, t in enumerate(layer_types) if t != "linear_attention"
    ]
    # Shared hybrid accounting reads these two names.
    config.num_gdn_attn_state = len(config.glm5_kda_layers)
    config.num_full_attn = len(config.glm5_mla_layers)

    # Per-layer MLP type; `first_k_dense_replace` is the fallback spelling.
    mlp_layer_types = list(getattr(config, "mlp_layer_types", []) or [])
    if not mlp_layer_types:
        k = getattr(config, "first_k_dense_replace", 0) or 0
        mlp_layer_types = ["dense"] * k + ["sparse"] * (config.num_hidden_layers - k)
    config.mlp_layer_types = mlp_layer_types

    # ---- KDA head config ----------------------------------------------
    # KimiKDAAttention reads the flattened `linear_*` names.
    lin = getattr(config, "linear_attn_config", {}) or {}
    config.linear_num_key_heads = lin.get("num_heads", config.num_attention_heads)
    config.linear_num_value_heads = config.linear_num_key_heads
    config.linear_key_head_dim = lin.get("head_dim", 128)
    config.linear_value_head_dim = config.linear_key_head_dim
    config.linear_conv_kernel_dim = lin.get("short_conv_kernel_size", 4)

    # ---- MoE aliases ----------------------------------------------------
    config.num_experts_per_tok = getattr(
        config, "num_experts_per_tok", getattr(config, "num_experts_per_token", 8)
    )
    config.norm_topk_prob = getattr(
        config, "norm_topk_prob", getattr(config, "moe_renormalize", True)
    )
    config.n_group = getattr(config, "n_group", 1)
    config.topk_group = getattr(config, "topk_group", 1)
    config.topk_method = getattr(config, "topk_method", "noaux_tc")
    config.scoring_func = getattr(config, "scoring_func", "sigmoid")
    config.swiglu_limit = getattr(config, "swiglu_limit", None)

    # ---- mHC ------------------------------------------------------------
    # Checkpoint spells these `hc_mult` / `hc_sinkhorn_iters`; the PR's config
    # class also accepts the `mhc_*` spellings.
    config.hc_mult = getattr(
        config, "hc_mult", getattr(config, "mhc_num_residual_streams", 4)
    )
    config.hc_sinkhorn_iters = getattr(
        config, "hc_sinkhorn_iters", getattr(config, "mhc_sinkhorn_iterations", 20)
    )
    config.hc_eps = getattr(config, "hc_eps", 1e-6)
    config.mhc = bool(getattr(config, "mhc", True))

    # ---- MLA ------------------------------------------------------------
    # Leave `qk_rope_head_dim` at its true 0 so the indexer stays NoPE; the MLA
    # attention pads to `_ROPE_PAD` locally (see the module docstring).
    config.head_dim = config.qk_nope_head_dim + config.qk_rope_head_dim
    # The MLA KV cache is sized from this, not from `kv_lora_rank +
    # qk_rope_head_dim`: the rope block is materialized at `_ROPE_PAD` and held
    # at zero (see the module docstring), so the cache rows must be that wide
    # even though the checkpoint's rope width is 0.
    config.mla_kv_entry_dim = config.kv_lora_rank + _ROPE_PAD
    rope_params = dict(getattr(config, "rope_parameters", None) or {})
    if not rope_params.get("rope_theta"):
        rope_params["rope_theta"] = getattr(config, "rope_theta", None) or 10000.0
    rope_params.setdefault("rope_type", "default")
    config.rope_parameters = rope_params


def _glm5_packed_modules_mapping(kda_layers: list[int]) -> dict[str, tuple[str, int]]:
    """Checkpoint projections -> this model's fused parameters.

    The KDA entries are per-layer-indexed so that only linear-attention layers
    fold q/k/v into ``in_proj``.

    Deliberately absent: the DeepSeek-style ``q_a_proj`` /
    ``kv_a_proj_with_mqa`` -> ``fused_qkv_a_proj`` entries.
    ``Glm5NextMLAAttention`` keeps those two as SEPARATE modules, so mapping
    them into a fused parameter this model never builds silently drops both
    weights and leaves the MLA projections at their init values -- every MLA
    layer then outputs zero and the model degrades to KDA-only.
    """
    mapping: dict[str, tuple[str, int]] = {
        ".gate_proj": (".gate_up_proj", 0),
        ".up_proj": (".gate_up_proj", 1),
    }
    for layer_idx in kda_layers:
        prefix = f".layers.{layer_idx}.self_attn."
        for shard_id, name in enumerate(("q_proj", "k_proj", "v_proj")):
            mapping[f"{prefix}{name}"] = (f"{prefix}in_proj", shard_id)
    return mapping


class Glm5NextMLP(DeepseekV2MLP):
    """Dense MLP / shared expert with GLM-5.3's clamped SwiGLU.

    ``DeepseekV2MLP`` activates through the plain ``SiluAndMul`` module, which
    has no clamp, so the activation is replaced here with the aiter kernel that
    takes a limit -- the same call DeepSeek-V4's ``Expert`` uses.  ``limit > 0``
    enables the in-kernel clamp (gate <= limit, up in [-limit, limit]) via ROCm
    ``v_med3_f32``; ``limit == 0`` disables it, so an unset ``swiglu_limit``
    leaves behaviour identical to the base class.
    """

    def __init__(self, *args, swiglu_limit: float | None = None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.swiglu_limit = float(swiglu_limit or 0.0)

    def forward(self, x):
        gate_up = self.gate_up_proj(x)
        if self.swiglu_limit <= 0.0:
            return self.down_proj(self.act_fn(gate_up))
        out = torch.empty(
            (gate_up.shape[0], gate_up.shape[-1] // 2),
            dtype=gate_up.dtype,
            device=gate_up.device,
        )
        aiter_silu_and_mul(out, gate_up, self.swiglu_limit)
        return self.down_proj(out)


class Glm5NextMoE(DeepseekV2MoE):
    """GLM-5.3 MoE: DeepSeek-shaped routing plus a clamped SwiGLU.

    Routing (sigmoid + ``noaux_tc`` correction bias, grouped top-k, shared
    expert) is identical to DeepSeek-V3, so only the activation clamp differs.
    """

    def __init__(
        self,
        config,
        quant_config: QuantizationConfig | None = None,
        reduce_results: bool = True,
        prefix: str = "",
        alt_stream: torch.cuda.Stream | None = None,
    ) -> None:
        super().__init__(
            config,
            quant_config=quant_config,
            reduce_results=reduce_results,
            prefix=prefix,
            alt_stream=alt_stream,
        )
        limit = float(getattr(config, "swiglu_limit", None) or 0.0)
        # Routed experts read the limit off the FusedMoE module (see
        # atom/model_ops/moe.py's `getattr(layer, "swiglu_limit", ...)`).
        self.experts.swiglu_limit = limit
        # When the shared expert is NOT fused into the routed tensor, the base
        # class builds it as a plain DeepseekV2MLP whose SiluAndMul cannot
        # clamp, so rebuild it as a Glm5NextMLP. Under aiter's fused-shared-
        # expert path there is no `shared_experts` attribute at all -- the
        # shared expert lives in routed slot `n_routed_experts` and picks up
        # the limit from `self.experts.swiglu_limit` set above.
        shared = getattr(self, "shared_experts", None)
        if shared is not None and limit > 0.0:
            self.shared_experts = Glm5NextMLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.moe_intermediate_size
                * config.n_shared_experts,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                reduce_results=False,
                prefix=f"{prefix}.shared_experts",
                swiglu_limit=limit,
            )


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

    def __getattr__(self, name):
        # Forward `.weight`, `.quant_type`, ... to the wrapped Linear.
        return getattr(self.__dict__["inner"], name)


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

    The pooled scoring path is not implemented yet (see ``_assert_kpool_regime``).
    Below ``index_topk`` candidates it is not needed: top-k then selects *every*
    pool, so the expansion yields every token position regardless of the pooled
    K values, and the token-granular selection this class inherits is exactly
    equal to it.  Past that threshold the two genuinely differ, so it refuses
    rather than returning a quietly wrong answer.
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
        # The base Indexer takes hidden_size but does not keep it.
        self.hidden_size = hidden_size
        self.index_kpool = int(getattr(config, "index_kpool", 1) or 1)
        self.kpool_always_select_tail = bool(
            getattr(config, "index_kpool_always_select_tail", True)
        )
        if self.index_kpool > 1:
            # Pool-compression parameters. Held so the checkpoint loads
            # completely and so the pooled path can be switched on without a
            # second weight-loading pass.
            self.index_kpool_compress_ape = atom_parameter(
                torch.zeros(self.index_kpool, self.head_dim, dtype=torch.float32)
            )
            # Kept without a `.weight` suffix to match the checkpoint name;
            # F.linear consumes its [head_dim, hidden_size] shape directly.
            self.index_kpool_compress_gate = atom_parameter(
                torch.empty(self.head_dim, self.hidden_size, dtype=torch.bfloat16)
            )
        self._kpool_warned = False
        # Bound by the metadata builder alongside the index K cache.
        self.kpool_tail_cache: torch.Tensor | None = None
        # Selection width: `index_topk` history tokens PLUS the unscored tail,
        # rounded up to the conversion kernels' BLOCK_N. Kept in sync with
        # `AiterMLAMetadataBuilder.index_topk_out`, which sizes the buffer this
        # writes into; the extra columns are never read because
        # `sparse_kv_indptr` caps every row at its true count.
        if self.index_kpool > 1:
            width = self.topk_tokens + self.index_kpool - 1
            self.topk_out_width = ((width + 127) // 128) * 128
        else:
            self.topk_out_width = self.topk_tokens

    def use_kpool(self) -> bool:
        """Whether to run the pooled indexer.

        ``ATOM_GLM5_KPOOL`` switches the pooled write, the pooled scoring and
        the pooled selection together, so the two settings are genuinely
        independent implementations of the same selection -- which is what
        makes the short-context A/B a check and not a tautology.
        """
        return kpool_ops.pooled_path_enabled(self.index_kpool)

    def _assert_kpool_regime(self, max_seqlen_k: int) -> None:
        """Refuse the regime where pooled and token-granular top-k diverge."""
        if self.index_kpool <= 1 or max_seqlen_k <= self.topk_tokens:
            return
        if self.use_kpool():
            return
        raise NotImplementedError(
            "GLM-5.3-Flash: this batch has max_seqlen_k="
            f"{max_seqlen_k} > index_topk={self.topk_tokens}, where pooled "
            "selection genuinely differs from the token-granular fallback, and "
            "ATOM_GLM5_KPOOL=0 disabled the pooled path. Returning the fallback "
            "here would be silently wrong, not merely slower."
        )

    def _maybe_dump_selection(self, attn_metadata) -> None:
        """Save this layer's selected KV slots when ATOM_GLM5_KPOOL_DUMP is set.

        Both the pooled and the token-granular path write the SAME buffer, so
        one hook here captures either -- which is what makes the A/B a
        comparison of two independent implementations rather than of one path
        against its own past output.

        Compare as SETS, not element-wise: the dense path emits in score order,
        the pooled path in pool-expanded order. Attention is permutation
        invariant over keys, so the set is what has to match.
        """
        path = os.environ.get("ATOM_GLM5_KPOOL_DUMP")
        if not path or self.prefix != os.environ.get(
            "ATOM_GLM5_KPOOL_DUMP_LAYER", self.prefix
        ):
            return
        # This reads a device value (`.item()`), which CUDAGraph capture
        # forbids outright -- and the profile/capture batches would be
        # meaningless to compare anyway. Same guard every other probe in this
        # file goes through, minus its dependence on ATOM_GLM5_DEBUG_STATS.
        if _dbg_capturing():
            return
        from atom.utils.forward_context import get_forward_context

        try:
            if get_forward_context().context.is_dummy_run:
                return
        except (RuntimeError, AttributeError, AssertionError):
            return
        indptr = getattr(attn_metadata, "sparse_kv_indptr", None)
        if indptr is None:
            return
        # An empty selection is not a passing comparison, it is a broken probe:
        # two empty sets match trivially. Refuse to write a dump that would let
        # the A/B report success without having compared anything.
        total = int(indptr[int(indptr.shape[0]) - 1].item())
        if total <= 0:
            return
        import torch as _torch

        rows = int(indptr.shape[0]) - 1
        _torch.save(
            {
                "prefix": self.prefix,
                "indptr": indptr[: rows + 1].detach().cpu(),
                "indices": self.sparse_kv_indices_buffer[: int(indptr[rows].item())]
                .detach()
                .cpu(),
            },
            f"{path}.r{_torch.cuda.current_device()}.{self.prefix}."
            f"{'prefill' if attn_metadata.max_seqlen_q > 1 else 'decode'}.pt",
        )

    def forward_impl(
        self,
        hidden_states: torch.Tensor,
        qr: torch.Tensor,
        qr_scale: torch.Tensor | None,
        positions,
        rotary_emb=None,
    ) -> torch.Tensor:
        from atom.utils.forward_context import get_forward_context

        attn_metadata = get_forward_context().attn_metadata
        if attn_metadata is not None:
            self._assert_kpool_regime(int(getattr(attn_metadata, "max_seqlen_k", 0)))

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
                assert get_forward_context().context.is_dummy_run, (
                    "kpool needs its per-request tail buffer, which the MLA+GDN "
                    "metadata builder binds next to the index K cache. Unbound "
                    "on a real forward means this model is running on a builder "
                    "that does not allocate it."
                )
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
            if state_slot_idx is None:
                # The profile/warmup forward carries no block tables, so the
                # builder leaves gdn_metadata unset. That forward is a dummy
                # run and the op discards it before touching any state; the
                # placeholder only has to exist. CUDAGraph capture DOES get
                # real slots (`_build_gdn_capture_metadata`), and it must --
                # capture bakes this pointer in, so a zeros stand-in there
                # would send every request's tail to slot 0 on replay.
                state_slot_idx = torch.zeros(
                    hidden_states.shape[0],
                    dtype=torch.int32,
                    device=hidden_states.device,
                )
            out = torch.ops.aiter.sparse_attn_indexer_kpool(
                hidden_states,
                self.k_cache.kv_cache[0],
                q_fp8,
                k,
                # The gate that drives the pooling softmax comes from the same
                # hidden states that produced k, so it stays token-aligned.
                torch.nn.functional.linear(
                    hidden_states, self.index_kpool_compress_gate
                ),
                weights,
                self.index_kpool_compress_ape,
                tail_cache,
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
            self._maybe_dump_selection(attn_metadata)
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
        self._maybe_dump_selection(attn_metadata)
        return out


def _mla_ref_check(layer, q_c, kv_c_normed, out) -> None:
    """Numerical oracle for the NoPE + zero-pad MLA path.

    Every structural check on this port passes (weights all load, routing is
    healthy, token diversity is fine) yet quality is uniformly degraded, which
    is what a subtly wrong kernel looks like. So recompute the same layer as
    plain causal SDPA from the same weights and compare. Valid only on a whole-
    prompt prefill, where the reference sees exactly the keys the kernel does.

    Off unless ATOM_GLM5_MLA_REF=1 -- it materializes the full [T, H, 256] K/V
    that MLA exists to avoid.
    """
    if os.environ.get("ATOM_GLM5_MLA_REF", "0") != "1" or not _dbg_real_forward():
        return
    from atom.utils.forward_context import get_forward_context

    ctx = get_forward_context()
    # Dispatch on prefill/decode BEFORE the once-per-layer guard: the guard key
    # is set by the prefill call, so checking it first would swallow every
    # decode call for that layer and print nothing at all.
    is_prefill = bool(getattr(getattr(ctx, "context", None), "is_prefill", False))
    if not is_prefill:
        _mla_ref_check_decode(layer, q_c, out, ctx)
        return
    key = ("mla_ref", layer.layer_num)
    if key in _DBG_SEEN:
        return
    _DBG_SEEN.add(key)

    with torch.no_grad():
        t = q_c.shape[0]
        h = layer.num_local_heads
        q = layer.q_b_proj(q_c)
        if isinstance(q, tuple):
            q = q[0]
        q = q.reshape(t, h, -1)[..., : layer.qk_nope_head_dim]
        kv = layer.kv_b_proj(kv_c_normed)
        if isinstance(kv, tuple):
            kv = kv[0]
        kv = kv.reshape(t, h, layer.qk_nope_head_dim + layer.v_head_dim)
        k = kv[..., : layer.qk_nope_head_dim]
        v = kv[..., layer.qk_nope_head_dim :]
        ref = torch.nn.functional.scaled_dot_product_attention(
            q.transpose(0, 1).unsqueeze(0).float(),
            k.transpose(0, 1).unsqueeze(0).float(),
            v.transpose(0, 1).unsqueeze(0).float(),
            is_causal=True,
            scale=layer.scaling,
        )
        ref = ref.squeeze(0).transpose(0, 1).reshape(t, h * layer.v_head_dim)
        ref = layer.o_proj(ref.to(q_c.dtype))
        if isinstance(ref, tuple):
            ref = ref[0]
        d = (out.float() - ref.float()).abs()
        denom = ref.float().abs().mean().clamp_min(1e-6)
        cos = torch.nn.functional.cosine_similarity(
            out.float().flatten(), ref.float().flatten(), dim=0
        )
        print(
            f"[glm5-mla-ref] layer={layer.layer_num:02d} T={t} "
            f"rel_err={float(d.mean() / denom):.4f} max_abs={float(d.max()):.4f} "
            f"cos={float(cos):+.6f} out_absmax={float(out.float().abs().max()):.4f} "
            f"ref_absmax={float(ref.float().abs().max()):.4f}",
            flush=True,
        )


def _mla_ref_check_decode(layer, q_c, out, ctx) -> None:
    """Same oracle, decode step.

    Prefill and decode run DIFFERENT math: prefill attends q_nope [H, 256]
    against materialized k_nope, while decode uses the absorbed form, dotting a
    latent q [H, 512+64] straight against the cached KV entry. Verifying prefill
    therefore says nothing about decode -- and the 64 zero-pad lanes this port
    adds live in the decode kernel's hard-coded 576-wide entry. So rebuild the
    attention from the KV cache and compare there too.

    Needs a bf16 KV cache (an fp8 cache would have to be de-quantized here to
    mean anything); it just reports and skips otherwise.
    """
    md = getattr(ctx, "attn_metadata", None)
    if md is None:
        return
    if isinstance(md, dict):
        md = next(iter(md.values()), None)
    cache = getattr(layer.mla_attn, "kv_cache", None)
    if md is None or cache is None or cache.numel() == 0:
        return
    n_done = sum(1 for k in _DBG_SEEN if isinstance(k, tuple) and k[0] == "mla_ref_d")
    if n_done >= 3:
        return
    bt = getattr(md, "block_tables", None)
    clen = getattr(md, "context_lens", None)
    if bt is None or clen is None or q_c.shape[0] != 1:
        return
    _DBG_SEEN.add(("mla_ref_d", layer.layer_num, n_done))

    with torch.no_grad():
        entry = cache.reshape(-1, cache.shape[-1])
        if entry.element_size() == 1:
            print(
                f"[glm5-mla-decref] layer={layer.layer_num:02d} skipped: fp8 cache",
                flush=True,
            )
            return
        bsz = int(clen[0])
        # Page size comes from the engine config, NOT from a cache dimension.
        # The natural MLA cache layout is [..., page_size, 1, 576] (see the
        # `kv_buffer.view(-1, page_size, 1, ...)` in attention_mla); reading
        # shape[-2] yields 1 and silently gathers one token per block, which
        # makes the reference garbage and the kernels look broken.
        from atom.config import get_current_atom_config

        page = int(get_current_atom_config().kv_cache_block_size)
        rows = bt[0][: (bsz + page - 1) // page].tolist()
        idx = torch.cat(
            [torch.arange(r * page, r * page + page, device=entry.device) for r in rows]
        )[:bsz]
        kv_entry = entry.index_select(0, idx)  # [L, 576]
        kv_c = kv_entry[:, : layer.kv_lora_rank]
        pad = kv_entry[:, layer.kv_lora_rank :]

        h = layer.num_local_heads
        q = layer.q_b_proj(q_c)
        if isinstance(q, tuple):
            q = q[0]
        q = q.reshape(1, h, -1)[..., : layer.qk_nope_head_dim]
        kv = layer.kv_b_proj(kv_c)
        if isinstance(kv, tuple):
            kv = kv[0]
        kv = kv.reshape(bsz, h, layer.qk_nope_head_dim + layer.v_head_dim)
        k = kv[..., : layer.qk_nope_head_dim]
        v = kv[..., layer.qk_nope_head_dim :]
        scores = torch.einsum("qhd,khd->hqk", q.float(), k.float()) * layer.scaling
        ref = torch.einsum("hqk,khd->qhd", scores.softmax(-1), v.float())
        ref = layer.o_proj(ref.reshape(1, h * layer.v_head_dim).to(q_c.dtype))
        if isinstance(ref, tuple):
            ref = ref[0]
        cos = torch.nn.functional.cosine_similarity(
            out.float().flatten(), ref.float().flatten(), dim=0
        )
        denom = ref.float().abs().mean().clamp_min(1e-6)
        print(
            f"[glm5-mla-decref] layer={layer.layer_num:02d} ctx_len={bsz} "
            f"page={page} cache={tuple(cache.shape)} "
            f"rel_err={float((out.float() - ref.float()).abs().mean() / denom):.4f} "
            f"cos={float(cos):+.6f} ropepad_absmax={float(pad.float().abs().max()):.5f} "
            f"out_absmax={float(out.float().abs().max()):.4f} "
            f"ref_absmax={float(ref.float().abs().max()):.4f}",
            flush=True,
        )


class Glm5NextMLAAttention(nn.Module):
    """NoPE sparse MLA.

    Deliberately narrower than ``DeepseekV2MLAAttention``: GLM-5.3-Flash needs
    none of that class's FP4 / MXFP4 / DCP / PCP branching, and reproducing it
    would mean carrying branches no GLM config can reach.  What remains is the
    plain V3.2 shape -- ``q_a -> q_a_layernorm -> q_b``, ``kv_a ->
    kv_a_layernorm``, sparse indexer, absorbed MLA -- with the rope block held
    at zero (``_ROPE_PAD``).

    ``q_a_proj`` and ``kv_a_proj_with_mqa`` are kept as separate GEMMs rather
    than fused into ``fused_qkv_a_proj``: the fusion would have to interleave
    the zero rope lanes into a shared FP8 block-scale layout, and the saving is
    one small launch on 11 of 46 layers.
    """

    def __init__(
        self,
        atom_config: Config,
        config,
        quant_config: QuantizationConfig | None,
        layer_num: int,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.qk_nope_head_dim = config.qk_nope_head_dim
        # `_ROPE_PAD` is 0 today: the MLA runs at the checkpoint's true width.
        self.qk_rope_head_dim = config.qk_rope_head_dim + _ROPE_PAD
        self.qk_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        self.v_head_dim = config.v_head_dim
        self.q_lora_rank = config.q_lora_rank
        self.kv_lora_rank = config.kv_lora_rank
        self.num_heads = config.num_attention_heads
        tp_size = get_tensor_model_parallel_world_size()
        assert self.num_heads % tp_size == 0
        self.num_local_heads = self.num_heads // tp_size
        self.layer_num = layer_num
        # Scale is the TRUE (unpadded) query width: the padded rope lanes are
        # zero and contribute nothing to the dot product, so including them
        # here would wrongly shrink the logits.
        self.scaling = (self.qk_nope_head_dim + config.qk_rope_head_dim) ** -0.5

        self.q_a_proj = ReplicatedLinear(
            self.hidden_size,
            self.q_lora_rank,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.q_a_proj",
        )
        self.q_a_layernorm = RMSNorm(self.q_lora_rank, eps=config.rms_norm_eps)
        # Checkpoint width (no rope); the zero lanes are appended per head at
        # call time by `_ZeroRopePad` so the parameter path stays exactly
        # `...self_attn.q_b_proj.weight`.
        self.q_b_proj = ColumnParallelLinear(
            self.q_lora_rank,
            self.num_heads * config.qk_nope_head_dim,
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

        # An IDENTITY rope (cos=1, sin=0), the same device Kimi-K3 uses for its
        # own un-rotated 64-wide block. A real rope would also be "correct" in
        # principle -- rotating the zero block yields zero -- but only if the
        # rotation lands exactly on those 64 lanes. Several MLA paths read
        # `cos_cache`/`sin_cache` inside FUSED kernels that rope and write the
        # KV cache in one pass; if any of them slices or interleaves differently
        # than assumed, a real rope's non-trivial cos/sin corrupts the latent.
        # With cos=1/sin=0 the transform is the identity no matter which lanes a
        # kernel touches, so the whole class of mismatch cannot arise.
        self.rotary_emb = _NoPositionalRotaryEmbedding(
            head_size=self.qk_rope_head_dim,
            rotary_dim=self.qk_rope_head_dim,
            max_position_embeddings=config.max_position_embeddings,
            base=(config.rope_parameters or {}).get("rope_theta") or 10000.0,
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
        self.indexer.rotary_emb = self.rotary_emb

        # ATOM_GLM5_FORCE_DENSE_MLA=1 runs MLA with sparsity off. Below
        # `index_topk` candidates that is not an approximation -- top-k would
        # select every token anyway -- so any output difference isolates a bug
        # in the indexer / sparse top-k path rather than in MLA itself.
        force_dense = os.environ.get("ATOM_GLM5_FORCE_DENSE_MLA", "0") == "1"
        # Detaching the indexer from MLAModules also stops `build_kv_cache_tensor`
        # from binding its K cache (that binding is gated on the attention module
        # owning an indexer), so the indexer must not be RUN either -- it would
        # reshape an unallocated cache and abort.
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

    def forward(
        self, hidden_states: torch.Tensor, positions: torch.Tensor
    ) -> torch.Tensor:
        q_c = self.q_a_proj(hidden_states)
        if isinstance(q_c, tuple):
            q_c = q_c[0]
        q_c = self.q_a_layernorm(q_c)

        kv_c = self.kv_a_proj_with_mqa(hidden_states)
        if isinstance(kv_c, tuple):
            kv_c = kv_c[0]
        # The checkpoint has no rope half, so the positional block is supplied
        # here as zeros. `concat_and_cache_mla` then writes a 576-wide entry --
        # the width the MLA decode kernels hard-code -- while the values stay
        # bit-for-bit NoPE.
        k_pe = kv_c.new_zeros(kv_c.shape[0], self.qk_rope_head_dim)
        kv_c_normed = self.kv_a_layernorm(kv_c)

        if self.run_indexer:
            self.indexer(hidden_states, q_c, None, positions, self.rotary_emb)
        out = self.mla_attn(q_c, kv_c_normed, k_pe, positions, None)
        _mla_ref_check(self, q_c, kv_c_normed, out)
        if _DBG_STATS and not _dbg_capturing() and self.layer_num == 3:
            from atom.utils.forward_context import get_forward_context

            _ctx = get_forward_context()
            _md = getattr(_ctx, "attn_metadata", None) if _ctx else None
            _is_prefill = bool(
                getattr(getattr(_ctx, "context", None), "is_prefill", False)
            )
            # Log the first few calls of ONE layer, tagged prefill/decode: the
            # question is whether the KV cache the decode step reads was
            # actually written during prefill.
            _n = sum(1 for k in _DBG_SEEN if isinstance(k, tuple) and k[0] == "mla_seq")
            key = ("mla_seq", _n)
            if _n < 6 and _md is not None:
                _DBG_SEEN.add(key)
                cache = getattr(self.mla_attn, "kv_cache", None)
                # Verify the padding actually landed where it is supposed to:
                # each 576-wide entry must be [ latent(512) | zero rope(64) ].
                # Non-zero rope lanes mean something is writing into the pad;
                # all-zero latent lanes mean the write went to the wrong slot.
                seg = "n/a"
                if cache is not None and cache.numel() > 0:
                    c = cache.reshape(-1, cache.shape[-1])
                    # slot_mapping rows actually written this step
                    rows = c[: min(64, c.shape[0])]
                    if rows.element_size() == 1:
                        rows = rows.view(torch.uint8).float()
                    else:
                        rows = rows.float().abs()
                    lat = float(rows[:, : self.kv_lora_rank].abs().mean())
                    pad = float(rows[:, self.kv_lora_rank :].abs().mean())
                    seg = f"latent={lat:.4f} ropepad={pad:.4f}"

                print(
                    f"[glm5-mla] call={_n} prefill={int(_is_prefill)}"
                    f" q_len={hidden_states.shape[0]}"
                    f" hs={_dbg_absmax(hidden_states):8.4f}"
                    f" q_c={_dbg_absmax(q_c):8.4f}"
                    f" kv_c={_dbg_absmax(kv_c_normed):8.4f}"
                    f" k_pe_numel={k_pe.numel()}"
                    f" out={_dbg_absmax(out):8.4f}"
                    f" cache={_dbg_absmax(cache) if cache is not None else -1.0:8.4f}"
                    f" cache_shape={tuple(cache.shape) if cache is not None else None}"
                    f" | {seg}",
                    flush=True,
                )
        return out


_DBG_STATS = os.environ.get("ATOM_GLM5_DEBUG_STATS", "0") == "1"
_DBG_SEEN: set = set()


def _dbg_capturing() -> bool:
    """True while this stream is being captured into a CUDA graph.

    Reading a device value then is illegal ("operation not permitted when
    stream is capturing") and aborts the run, and `--level 0` does NOT turn
    capture off, so the guard has to be on the capture state itself rather than
    on a launch flag.
    """
    try:
        return torch.cuda.is_current_stream_capturing()
    except (RuntimeError, AttributeError):
        return False


# Reducing more than this many elements is not worth a debug print, and the
# KV cache is orders of magnitude bigger -- sample its head instead.
_DBG_MAX_ELEMS = 1 << 22


def _dbg_absmax(x: torch.Tensor) -> float:
    """Largest magnitude in `x`, or -1 when stats are off / unsafe to read.

    Deliberately does NOT cast to fp32 first: `.float()` materializes a full
    copy, and handing this the whole KV cache then tries to allocate tens of
    GiB and OOMs the run. Reduce in the native dtype and cast the scalar.
    """
    if not _DBG_STATS or _dbg_capturing():
        return -1.0
    # NoPE makes several tensors legitimately 0-element (k_pe), and an
    # unallocated KV cache is empty too; max() over 0 elements raises.
    if x is None or x.numel() == 0:
        return -1.0
    t = x.detach()
    if t.numel() > _DBG_MAX_ELEMS:
        t = t.reshape(-1)[:_DBG_MAX_ELEMS]
    # abs() has no fp8 kernel, and for a cache the question is only "was
    # anything written", so read the raw bytes: a max of 0 means untouched.
    if t.element_size() == 1:
        return float(t.view(torch.uint8).max())
    return float(t.abs().max())


def _dbg_layer_stats(layer, stage: str, in_absmax: float, out: torch.Tensor) -> None:
    """One line per (layer, stage) on the first forward that reaches it.

    Localizes a numerical fault to a layer KIND: with 34 KDA layers and 11 MLA
    layers interleaved, a magnitude that explodes or collapses at one kind is
    the fastest way to tell which sub-layer is wrong.
    """
    if not _DBG_STATS or _dbg_capturing():
        return
    # The profile/warmup forward runs with no attention metadata, so every
    # attention sub-layer legitimately returns zeros. Logging that would say
    # "attention outputs 0" for a perfectly healthy model, so skip it and
    # report the first REAL forward instead.
    from atom.utils.forward_context import get_forward_context

    try:
        ctx = get_forward_context()
        if ctx is None or ctx.attn_metadata is None or ctx.context.is_dummy_run:
            return
    except (RuntimeError, AttributeError, AssertionError):
        return
    # CUDAGraph capture also carries real metadata but feeds ZERO-filled
    # inputs, so it looks like a healthy forward that outputs nothing. Wait for
    # a batch that actually carries signal.
    if in_absmax <= 0.0:
        return
    key = (layer.layer_idx, stage)
    if key in _DBG_SEEN:
        return
    _DBG_SEEN.add(key)
    o = out.detach().float()
    kind = "kda" if layer.is_kda else "mla"
    # Mean pairwise cosine between TOKENS. Magnitude stats (absmax/rms) are blind
    # to direction collapse: every token can carry a healthy norm while pointing
    # the same way, which is exactly what makes the MoE route all tokens to one
    # set of experts. ~0 is healthy diversity, ~1 means the tokens are parallel.
    cos = -1.0
    if o.dim() == 2 and o.shape[0] > 1:
        v = torch.nn.functional.normalize(o[: min(256, o.shape[0])], dim=-1)
        g = v @ v.T
        n = g.shape[0]
        cos = float((g.sum() - n) / (n * (n - 1)))
    print(
        f"[glm5-stats] layer={layer.layer_idx:02d} kind={kind} stage={stage} "
        f"in_absmax={in_absmax:9.3f} out_absmax={float(o.abs().max()):9.3f} "
        f"out_rms={float(o.pow(2).mean().sqrt()):9.4f} "
        f"tok_cos={cos:+.4f} "
        f"nan={bool(o.isnan().any())} inf={bool(o.isinf().any())}",
        flush=True,
    )


class Glm5NextDecoderLayer(nn.Module):
    """One layer: attention and FFN, each wrapped in a hyper-connection pair.

    The residual is an ``hc_mult``-wide stack, so each sub-layer is bracketed by
    ``hc_pre`` (stack -> one input) and ``hc_post`` (output -> stack).  Because
    a layer's FFN ``hc_post`` is immediately followed by the next layer's
    attention ``hc_pre``, every ``hc_post`` except the model's last is
    **deferred** and fused into the following ``hc_pre`` by one aiter kernel.
    ``forward`` therefore threads ``(residual, post, comb)`` between layers and
    a layer returns its FFN output *unmixed*.
    """

    def __init__(
        self,
        atom_config: Config,
        config,
        layer_idx: int,
        mhc_ops: MHCOps,
        prefix: str = "",
        alt_stream: torch.cuda.Stream | None = None,
    ) -> None:
        super().__init__()
        quant_config = atom_config.quant_config
        self.layer_idx = layer_idx
        self.num_hidden_layers = config.num_hidden_layers
        self.is_kda = layer_idx in config.glm5_kda_layers
        self.mhc_ops = mhc_ops
        self.hc_mult = config.hc_mult
        self.top_k = config.num_experts_per_tok

        if self.is_kda:
            self.self_attn = KimiKDAAttention(
                atom_config,
                quant_config,
                prefix=f"{prefix}.self_attn",
                # GLM-5.3 ships g_a_proj / g_b_proj, not Kimi's full-rank g_proj.
                lowrank_out_gate=True,
            )
        else:
            self.self_attn = Glm5NextMLAAttention(
                atom_config,
                config,
                quant_config,
                layer_num=layer_idx,
                prefix=f"{prefix}.self_attn",
            )

        is_sparse = (
            layer_idx < len(config.mlp_layer_types)
            and config.mlp_layer_types[layer_idx] == "sparse"
        )
        if is_sparse and config.n_routed_experts:
            self.mlp = Glm5NextMoE(
                config,
                quant_config=quant_config,
                prefix=f"{prefix}.mlp",
                alt_stream=alt_stream,
            )
        else:
            self.mlp = Glm5NextMLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                prefix=f"{prefix}.mlp",
                swiglu_limit=config.swiglu_limit,
            )

        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.hc_attn = HyperConnection(config.hidden_size, config.hc_mult)
        self.hc_ffn = HyperConnection(config.hidden_size, config.hc_mult)

    def _attn(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        if self.is_kda:
            return self.self_attn(x)
        return self.self_attn(x, positions)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
        post: torch.Tensor | None,
        comb: torch.Tensor | None,
    ):
        """Returns ``(x, residual, post, comb)``.

        On the last layer the deferred ``hc_post`` has nothing to fuse with, so
        it is materialized here and the stack is contracted to ``[T, dim]``;
        that call returns ``(x, None, None, None)``.
        """
        x = hidden_states
        if post is None:
            # Model entry: no deferred hc_post to fuse, so hc_pre stands alone.
            if self.layer_idx == 0:
                x = hc_expand(x, self.hc_mult)
            residual = x
            x, post, comb = self.mhc_ops.pre(
                residual,
                self.hc_attn,
                norm_weight=self.input_layernorm.weight.data,
                norm_eps=self.input_layernorm.eps,
            )
        else:
            residual, post, comb, x = self.mhc_ops.fused_post_pre(
                x,
                residual,
                post,
                comb,
                self.hc_attn,
                norm_weight=self.input_layernorm.weight.data,
                norm_eps=self.input_layernorm.eps,
            )

        attn_in_absmax = _dbg_absmax(x)
        _dbg_ckpt(self, "attn_in", x)
        _dbg_trace(self, "attn_in", x)
        x = self._attn(x, positions)
        _dbg_trace(self, "attn_out", x)
        _dbg_ckpt(self, "attn_out", x)
        _dbg_layer_stats(self, "attn", attn_in_absmax, x)

        residual, post, comb, x = self.mhc_ops.fused_post_pre(
            x,
            residual,
            post,
            comb,
            self.hc_ffn,
            norm_weight=self.post_attention_layernorm.weight.data,
            norm_eps=self.post_attention_layernorm.eps,
        )

        _dbg_trace(self, "mlp_in", x)
        mlp_in_absmax = _dbg_absmax(x)
        _dbg_router(self, x)
        x = self.mlp(x)
        _dbg_layer_stats(self, "mlp", mlp_in_absmax, x)

        if self.layer_idx == self.num_hidden_layers - 1:
            x = self.mhc_ops.post(x, residual, post, comb)
            return hc_contract(x), None, None, None
        return x, residual, post, comb


_CKPT_REQ = [0]


def _dbg_ckpt(layer, tag: str, x: torch.Tensor) -> None:
    """Per-layer checksum, to find WHERE nondeterminism enters.

    Identical requests give different first tokens even in eager at level 0,
    and the control model (GLM-5.2) does not, so something in this model is
    unstable. Rather than double-calling sub-layers -- unsafe for KDA, which
    mutates conv/recurrent state -- just fingerprint the activations of every
    layer on each of several identical requests. The first layer whose
    fingerprint differs between two runs is where instability enters.
    """
    if os.environ.get("ATOM_GLM5_CKPT", "0") != "1" or not _dbg_real_forward():
        return
    if layer.layer_idx == 0 and tag == "attn_in":
        _CKPT_REQ[0] += 1
    print(
        f"[ckpt] req={_CKPT_REQ[0]:02d} layer={layer.layer_idx:02d} {tag:9s}"
        f" sum={float(x.float().sum()):.6f} absmax={float(x.float().abs().max()):.6f}",
        flush=True,
    )


def _dbg_real_forward() -> bool:
    """True only on a forward carrying REAL tokens.

    The profile/warmup forward feeds a dummy batch whose input_ids are all the
    same id, so every embedding is identical and every downstream diversity
    metric reads as total collapse. CUDAGraph capture likewise feeds zeros.
    Every probe must pass through here -- measuring either one produces
    confident, completely fictitious findings.
    """
    if not _DBG_STATS or _dbg_capturing():
        return False
    from atom.utils.forward_context import get_forward_context

    try:
        ctx = get_forward_context()
        if ctx is None or ctx.attn_metadata is None:
            return False
        return not ctx.context.is_dummy_run
    except (RuntimeError, AttributeError, AssertionError):
        return False


def _dbg_cos(x: torch.Tensor) -> float:
    """Mean pairwise cosine between tokens; +1.0 means they are all parallel."""
    if not _dbg_real_forward() or x is None or x.dim() != 2 or x.shape[0] < 2:
        return -9.0
    v = torch.nn.functional.normalize(x[: min(192, x.shape[0])].float(), dim=-1)
    g = v @ v.T
    n = g.shape[0]
    return float((g.sum() - n) / (n * (n - 1)))


def _dbg_trace(layer, tag: str, x: torch.Tensor) -> None:
    """Where along the stack does token diversity die?

    The MoE input already measures +1.0000 (all tokens parallel), but that is
    layer 3+. Printing the same number at each point of each layer localizes
    the exact step that collapses it, instead of guessing which sub-layer is
    responsible.
    """
    if not _dbg_real_forward():
        return
    key = ("trace", layer.layer_idx, tag)
    if key in _DBG_SEEN:
        return
    _DBG_SEEN.add(key)
    print(
        f"[glm5-trace] layer={layer.layer_idx:02d} {tag:12s} tok_cos={_dbg_cos(x):+.4f}",
        flush=True,
    )


def _dbg_router(layer, x: torch.Tensor) -> None:
    """Inspect MoE routing read-only, without perturbing the model.

    A mis-wired router (wrong scoring function, unapplied correction bias,
    collapsed top-k) yields fluent-but-wrong text -- the same symptom as every
    other candidate -- so it has to be looked at rather than reasoned about.
    Recomputes the gate on the side; it is a tiny [T, 4096] x [4096, 288] GEMM.
    """
    if not _dbg_real_forward():
        return
    if not isinstance(layer.mlp, Glm5NextMoE):
        return
    key = ("router", layer.layer_idx)
    if key in _DBG_SEEN or len(_DBG_SEEN) > 400:
        return
    _DBG_SEEN.add(key)
    with torch.no_grad():
        # Direction diversity of the GATE'S OWN INPUT. `tok_cos` elsewhere is
        # measured on layer OUTPUTS; this is the tensor the router actually
        # sees. If all tokens point the same way here, every token ranks the
        # experts identically and top-k collapses onto one set -- which is
        # exactly the `experts_used=8/288` symptom.
        xv = torch.nn.functional.normalize(x[: min(256, x.shape[0])].float(), dim=-1)
        gram = xv @ xv.T
        nn_ = gram.shape[0]
        x_cos = float((gram.sum() - nn_) / (nn_ * (nn_ - 1)))
        # Rank agreement between two different tokens, independent of scale.
        logits = layer.mlp.gate(x)
        if isinstance(logits, tuple):
            logits = logits[0]
        scores = logits.float().sigmoid()
        bias = getattr(layer.mlp.gate, "e_score_correction_bias", None)
        biased = scores + bias.float() if bias is not None else scores
        topk = min(int(layer.top_k), biased.shape[-1])
        _, idx = torch.topk(biased, topk, dim=-1)
        # Renormalized weights are gathered from the UNBIASED scores; the bias
        # only steers selection (noaux_tc).
        gw = scores.gather(-1, idx)
        gw = gw / gw.sum(-1, keepdim=True).clamp_min(1e-9)
        uniq = int(idx.reshape(-1).unique().numel())
        # The decisive comparison: how much does sigmoid vary ACROSS experts for
        # a single token, versus how much the bias varies across experts? topk
        # ranks on their sum, so whichever spread is larger decides selection.
        sig_spread = float((scores.max(-1).values - scores.min(-1).values).mean())
        bias_spread = float(bias.max() - bias.min()) if bias is not None else -1.0
        # If selection is bias-dominated it will simply be the bias's own top-k.
        bias_top = (
            set(torch.topk(bias.float(), topk).indices.tolist())
            if bias is not None
            else set()
        )
        sel_top = set(idx[0].tolist())
        print(
            f"[glm5-router] layer={layer.layer_idx:02d} tokens={x.shape[0]}"
            f" experts_used={uniq}/{biased.shape[-1]}"
            f" bias={'yes' if bias is not None else 'NO'}"
            f" score.mean={float(scores.mean()):.4f}"
            f" topw.mean={float(gw.mean()):.4f}"
            f" topw.max={float(gw.max()):.4f}"
            f" | sigmoid_spread={sig_spread:.4f} bias_spread={bias_spread:.4f}"
            f" sel==bias_top8:{sel_top == bias_top}"
            f" | x_tok_cos={x_cos:+.4f}"
            f" logit_spread={float((logits.float().max(-1).values - logits.float().min(-1).values).mean()):.4f}",
            flush=True,
        )


class Glm5NextModel(nn.Module):
    def __init__(self, atom_config: Config, prefix: str = "") -> None:
        super().__init__()
        config = _glm5_text_config(atom_config.hf_config)
        _normalize_glm5_config(config)
        self.config = config
        self.vocab_size = config.vocab_size

        if get_pp_group().is_first_rank:
            self.embed_tokens = VocabParallelEmbedding(
                config.vocab_size, config.hidden_size
            )
        else:
            self.embed_tokens = PPMissingLayer()

        self.alt_stream = (
            torch.cuda.Stream() if getattr(config, "n_shared_experts", 0) else None
        )
        _alt_stream = self.alt_stream
        # One MHCOps for the whole model: it holds only the resolved kernel
        # entry points and scalars, so every layer can share it.
        self.mhc_ops = MHCOps(
            dim=config.hidden_size,
            hc_mult=config.hc_mult,
            norm_eps=config.rms_norm_eps,
            hc_eps=config.hc_eps,
            sinkhorn_iters=config.hc_sinkhorn_iters,
        )
        _mhc_ops = self.mhc_ops

        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix, layer_num=None: Glm5NextDecoderLayer(
                atom_config,
                config,
                layer_idx=layer_num or 0,
                mhc_ops=_mhc_ops,
                prefix=prefix,
                alt_stream=_alt_stream,
            ),
            prefix=f"{prefix}.layers",
            layer_num_offset=0,
        )

        if get_pp_group().is_last_rank:
            self.norm = RMSNorm(
                config.hidden_size, eps=config.rms_norm_eps, prefix=f"{prefix}.norm"
            )
        else:
            self.norm = PPMissingLayer()
        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states", "residual"], config.hidden_size
        )

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ):
        if get_pp_group().is_first_rank:
            hidden_states = (
                inputs_embeds
                if inputs_embeds is not None
                else self.get_input_embeddings(input_ids)
            )
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]

        if _dbg_real_forward() and ("trace", -1, "embed") not in _DBG_SEEN:
            _DBG_SEEN.add(("trace", -1, "embed"))
            print(
                f"[glm5-trace] layer=-1 embed        tok_cos={_dbg_cos(hidden_states):+.4f}",
                flush=True,
            )
        # mHC state threaded across layers; see Glm5NextDecoderLayer.forward.
        residual = post = comb = None
        for i in range(self.start_layer, self.end_layer):
            hidden_states, residual, post, comb = self.layers[i](
                positions, hidden_states, residual, post, comb
            )

        if not get_pp_group().is_last_rank:
            return IntermediateTensors({"hidden_states": hidden_states})
        return self.norm(hidden_states)

    def get_expert_mapping(self):
        from atom.model_ops.moe import FusedMoE

        # `+ n_shared_experts` is NOT optional. When aiter fuses the shared
        # expert into the routed tensor, `w13_weight` is allocated with
        # `n_routed_experts + n_shared_experts` slots (289 here) and the loader
        # rewrites `mlp.shared_experts.*` to `mlp.experts.288.*`
        # (model_loader/weight_names.py:_maybe_fuse_shared_expert). Without a
        # mapping entry for that id the renamed tensors match nothing and slot
        # 288 keeps its `torch.empty` contents -- uninitialized memory that the
        # shared expert then contributes to EVERY token of EVERY MoE layer.
        # Nothing reports it: `w13_weight` is one parameter, so writing any
        # slot marks the whole thing loaded.
        return FusedMoE.make_expert_params_mapping(
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=self.config.n_routed_experts
            + (self.config.n_shared_experts or 0),
        )


class Glm5NextForCausalLM(nn.Module):
    """Text stack of GLM-5.3-Flash."""

    # The checkpoint nests the language model under `model.language_model.` and
    # the vision tower under `model.visual.`; flatten the former and skip the
    # latter (text-only serving).
    weights_mapping: ClassVar[dict[str, str]] = {
        "model.language_model.": "model.",
        # The checkpoint stores the mHC tensors flat on the layer
        # (`hc_attn_fn`); this model groups each sub-layer's three into a
        # `HyperConnection` submodule (`hc_attn.fn`).
        "hc_attn_fn": "hc_attn.fn",
        "hc_attn_base": "hc_attn.base",
        "hc_attn_scale": "hc_attn.scale",
        "hc_ffn_fn": "hc_ffn.fn",
        "hc_ffn_base": "hc_ffn.base",
        "hc_ffn_scale": "hc_ffn.scale",
    }
    skip_weight_prefixes: ClassVar[list[str]] = ["model.visual."]
    # `modules_to_not_convert` names the KDA input projection by the upstream
    # fused spellings (`qkv_proj` / `fused_qkvbfg_a_proj`); ATOM fuses q|k|v
    # into `in_proj`, so without this translation the exclusion misses and the
    # BF16 KDA weights get treated as FP8. Every other KDA projection
    # (b/f_a/f_b/g_a/g_b/o/conv1d) is listed under a name ATOM already uses.
    quant_exclude_name_mapping: ClassVar[dict[str, str]] = {
        "qkv_proj": "in_proj",
    }
    # Consulted at class level by ModelRunner before construction; the
    # per-layer KDA entries are added on the instance in __init__ once the
    # layer layout is known.
    packed_modules_mapping: ClassVar[dict[str, tuple[str, int]]] = (
        _glm5_packed_modules_mapping([])
    )

    def __init__(self, atom_config: Config, prefix: str = "") -> None:
        super().__init__()
        config = _glm5_text_config(atom_config.hf_config)
        _normalize_glm5_config(config)
        self.config = config
        self.quant_config = atom_config.quant_config
        self.packed_modules_mapping = _glm5_packed_modules_mapping(
            config.glm5_kda_layers
        )
        self.model = Glm5NextModel(atom_config, prefix=maybe_prefix(prefix, "model"))
        if get_pp_group().is_last_rank:
            self.lm_head = ParallelLMHead(
                config.vocab_size,
                config.hidden_size,
                org_num_embeddings=config.vocab_size,
                prefix=maybe_prefix(prefix, "lm_head"),
            )
        else:
            self.lm_head = PPMissingLayer()
        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors
        )

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.get_input_embeddings(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ):
        return self.model(input_ids, positions, intermediate_tensors, inputs_embeds)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        return self.lm_head(hidden_states)

    def get_expert_mapping(self):
        return self.model.get_expert_mapping()


class Glm5NextForConditionalGeneration(Glm5NextForCausalLM):
    """Registered entry point.

    The checkpoint's architecture string is the multimodal one; ATOM serves the
    text path, so this is the text stack with the vision tower skipped. The MTP
    layer (index ``num_hidden_layers``) is likewise not built -- speculative
    decoding is out of scope -- so its tensors are skipped at load.
    """

    @staticmethod
    def is_mtp_weight(name: str, num_hidden_layers: int) -> bool:
        mtp_marker = f"layers.{num_hidden_layers}."
        return mtp_marker in name

    def __init__(self, atom_config: Config, prefix: str = "") -> None:
        super().__init__(atom_config, prefix=prefix)
        n = self.config.num_hidden_layers
        # Skip the (unbuilt) MTP layer's tensors and the vision tower.
        self.skip_weight_prefixes = list(self.skip_weight_prefixes) + [
            f"model.language_model.layers.{n}.",
            f"model.layers.{n}.",
        ]


# ==========================================================================
# kpool: the pooled indexer path
# ==========================================================================
#
# The token-granular indexer this file inherits from DeepSeek-V3.2 scores every
# cached token. GLM-5.3 instead caches ONE key per `index_kpool` tokens, runs
# top-k over pools, expands each selected pool back to its tokens, and always
# appends the trailing incomplete pool.
#
# Pool p lives at `block_table[p // pool_rows]`, row `p % pool_rows`, where
# `pool_rows = kv_cache_block_size // kpool` is the index cache's rows per
# block -- one index block per KV block, so the request's own block table
# addresses the pooled cache with no remapping and the KV allocator is
# untouched.
#
# `pool_rows` must stay a multiple of 16 or the paged MQA-logits kernel loses
# its preshuffled layout, the only one it computes correctly. `Config` raises
# `kv_cache_block_size` to `kpool * 16` for this model so that holds with no
# rows to spare: the index cache is exactly `1/kpool` of one row per token,
# which is all the pooled path ever writes.
#
# Below `index_topk` the pooled and token-granular selections are the SAME SET:
# top-k picks every pool, so the expansion yields every token position. That is
# the equality the A/B gate checks, and it exercises the pooled write, the
# pooled scoring, the pooled top-k and the expansion all at once.


def _kpool_request_index(cu_seqlens_q: torch.Tensor, n_tokens: int) -> torch.Tensor:
    """Per-token request id for a flat prefill batch."""
    counts = (cu_seqlens_q[1:] - cu_seqlens_q[:-1]).to(torch.int64)
    return torch.repeat_interleave(
        torch.arange(counts.shape[0], device=cu_seqlens_q.device, dtype=torch.int64),
        counts,
        output_size=n_tokens,
    )


def _kpool_write_completed_pools(
    kv_cache: torch.Tensor,
    k: torch.Tensor,
    gate_score: torch.Tensor,
    positions: torch.Tensor,
    pool_bt: torch.Tensor,
    req_idx: torch.Tensor,
    compress_ape: torch.Tensor,
    index_kpool: int,
    head_dim: int,
    scale_fmt: str,
    pool_rows: int,
    chunk_start: torch.Tensor | None = None,
    tail_cache: torch.Tensor | None = None,
    state_slot_idx: torch.Tensor | None = None,
) -> None:
    """Compress and cache every pool that closes inside this batch.

    Every token is treated as a pool-completion candidate and non-completions
    are masked off with a ``-1`` slot, which ``indexer_k_quant_and_cache``
    skips. Compacting the valid rows first would cost two device syncs on the
    eager prefill path and buys nothing numerically.

    ``chunk_start`` (per request, absolute) enables cross-chunk carry-over for
    pools split by chunked prefill; pass None when every request starts at 0.
    """
    kpool = index_kpool
    n = k.shape[0]
    if n < kpool:
        return
    row = torch.arange(n, device=k.device)
    offs = torch.arange(kpool, device=k.device)
    # Token i closes the pool spanning tokens i-(kpool-1) .. i.
    idx = (row - (kpool - 1)).clamp_min(0)[:, None] + offs[None, :]
    pool_k, pool_gate = k[idx], gate_score[idx]
    if chunk_start is not None:
        # Chunked prefill can split a request mid-pool. That pool's earlier
        # tokens are not in this batch -- but they ARE in the tail buffer, which
        # the previous chunk seeded with exactly them. Substitute those rows
        # instead of refusing the batch (or, worse, compressing a pool from the
        # wrong tokens: the pool would then be silently wrong forever, since
        # nothing ever revisits it).
        #
        # Pool starts are multiples of kpool, so a slot's absolute position has
        # `abs % kpool == s` and the tail row index is just the slot index.
        abs_slot = positions.to(torch.int64)[:, None] - (kpool - 1) + offs[None, :]
        from_tail = abs_slot < chunk_start[req_idx][:, None]
        stash = tail_cache[state_slot_idx[req_idx]]  # [n, 2, kpool, head_dim]
        pool_k = torch.where(from_tail[..., None], stash[:, 0], pool_k)
        pool_gate = torch.where(from_tail[..., None], stash[:, 1], pool_gate)
    pooled = kpool_ops.pool_and_rotate(pool_k, pool_gate, compress_ape)
    abs_pos = positions.to(torch.int64)
    closes = (abs_pos % kpool == kpool - 1) & (row >= kpool - 1)
    slots = kpool_ops.pool_slot_mapping(
        pool_bt,
        torch.where(closes, abs_pos // kpool, torch.full_like(abs_pos, -1)),
        req_idx,
        pool_rows,
    )
    indexer_k_quant_and_cache(
        pooled, kv_cache, slots, head_dim, scale_fmt, preshuffle=True
    )


def _kpool_pool_counts(seq_lens_k: torch.Tensor, kpool: int) -> torch.Tensor:
    """Complete pools per request. The tail is never cached, only appended."""
    return seq_lens_k.to(torch.int64) // kpool


_KPOOL_REF_SEEN: set = set()


def _kpool_verify_cache(
    kv_cache, k, gate_score, positions, pool_bt, cu_q, ape, kpool, head_dim
) -> None:
    """ATOM_GLM5_KPOOL_REF=1: read the pooled keys back and check them.

    The synthetic round-trip test proves the ADDRESSING; this proves the whole
    write actually landed under the engine's real metadata -- that the k and
    gate reaching the kernel are the right rows, that `positions` line up with
    the slots, and that the gather reads back what was written. Those are
    exactly the couplings a synthetic test cannot see.

    Off by default: it gathers and syncs, so it is a debugging tool, not a
    runtime check.
    """
    if os.environ.get("ATOM_GLM5_KPOOL_REF", "0") != "1" or _dbg_capturing():
        return
    from atom.utils.forward_context import get_forward_context

    try:
        if get_forward_context().context.is_dummy_run:
            return
    except (RuntimeError, AttributeError, AssertionError):
        return
    n_req = int(cu_q.shape[0]) - 1
    if n_req < 1:
        return
    q0, q1 = int(cu_q[0].item()), int(cu_q[1].item())
    seq_len = int(positions[q1 - 1].item()) + 1
    n_pools = seq_len // kpool
    # Only the pools whose four tokens are all inside this chunk can be checked
    # against `k` here; earlier ones came from a previous chunk's tensors.
    first = ((seq_len - (q1 - q0)) + kpool - 1) // kpool
    if n_pools <= first:
        return
    key = (id(kv_cache), seq_len)
    if key in _KPOOL_REF_SEEN:
        return
    _KPOOL_REF_SEEN.add(key)

    from aiter.ops.cache import cp_gather_indexer_k_quant_cache

    dst_k = torch.empty(n_pools, head_dim, dtype=dtypes.fp8, device=k.device)
    dst_s = torch.empty(n_pools, 1, dtype=torch.float32, device=k.device)
    cu = torch.tensor([0, n_pools], dtype=torch.int32, device=k.device)
    cp_gather_indexer_k_quant_cache(
        kv_cache, dst_k, dst_s.view(dtypes.fp8), pool_bt[:1], cu, preshuffle=True
    )
    got = (dst_k.float() * dst_s)[first:n_pools]

    base = (first * kpool) - (seq_len - (q1 - q0)) + q0
    m = (n_pools - first) * kpool
    want = kpool_ops.pool_and_rotate(
        k[base : base + m].view(-1, kpool, head_dim),
        gate_score[base : base + m].view(-1, kpool, head_dim),
        ape,
    ).float()
    num = (got * want).sum(-1)
    den = got.norm(dim=-1) * want.norm(dim=-1) + 1e-9
    cos = (num / den).min().item()
    rel = ((got - want).abs().max() / want.abs().max().clamp_min(1e-9)).item()
    print(
        f"[kpool-ref] seq_len={seq_len} pools[{first}:{n_pools}] "
        f"min_cos={cos:.6f} rel_err={rel:.4f}",
        flush=True,
    )


def _sparse_attn_indexer_kpool(
    hidden_states: torch.Tensor,
    kv_cache: torch.Tensor,
    q_fp8: torch.Tensor,
    k: torch.Tensor,
    gate_score: torch.Tensor,
    weights: torch.Tensor,
    compress_ape: torch.Tensor,
    tail_cache: torch.Tensor,
    state_slot_idx: torch.Tensor,
    positions: torch.Tensor,
    sparse_kv_indices_buffer: torch.Tensor,
    topk_tokens: int,
    index_kpool: int,
    head_dim: int,
    max_model_len: int,
    topk_out_width: int,
    scale_fmt: str,
    stable_topk: bool,
) -> torch.Tensor:
    """Pooled sparse-indexer top-k. Writes ``sparse_kv_indices_buffer``.

    Mirrors `deepseek_v2.sparse_attn_indexer` step for step, with pools where
    it has tokens: the cache holds one compressed key per `index_kpool` tokens,
    top-k selects `topk_tokens // index_kpool` POOLS, and the selection is
    expanded back to token positions with the unscored tail appended.
    """
    from atom.config import get_current_atom_config
    from atom.utils.forward_context import get_forward_context

    forward_context = get_forward_context()
    attn_metadata = forward_context.attn_metadata
    context = forward_context.context
    if context.is_dummy_run:
        return torch.zeros_like(weights, dtype=torch.float32)

    # Axes this landing does not cover. Each one would mis-index rather than
    # fail, so refuse explicitly instead of returning a quietly wrong selection.
    from atom.distributed.dcp_utils import get_dcp_world_size
    from atom.distributed.pcp_utils import pcp_is_enabled

    if get_dcp_world_size() > 1 or pcp_is_enabled():
        raise NotImplementedError(
            "GLM-5.3 kpool + DCP/PCP: both shard tokens round-robin across "
            "ranks, which does not commute with pooling four CONSECUTIVE "
            "tokens into one key. Run kpool at dcp=pcp=1."
        )
    if not context.is_prefill and attn_metadata.max_seqlen_q > 1:
        raise NotImplementedError(
            "GLM-5.3 kpool + speculative decode: the decode path assumes one "
            f"token per request, got max_seqlen_q={attn_metadata.max_seqlen_q}. "
            "GLM-5.3's MTP layer is not loaded, so this is unreachable today."
        )

    device = hidden_states.device
    # Two different granularities, equal only by coincidence before the block
    # size was raised: `block_size` is TOKENS per block, used to turn a token id
    # into a slot through the token block table, while `pool_rows` is the index
    # cache's ROWS per block. Confusing them writes pools to the wrong slots
    # without erroring.
    block_size = get_current_atom_config().kv_cache_block_size
    pool_rows = block_size // index_kpool
    kv_cache = kv_cache.view(-1, pool_rows, kv_cache.shape[-1])
    # One index block per KV block, so the request's own block table addresses
    # the pooled cache unchanged.
    pool_bt = attn_metadata.block_tables
    select_k = topk_tokens // index_kpool
    n_tokens = hidden_states.shape[0]
    n_head = q_fp8.shape[1]

    topk_indices = torch.full(
        (n_tokens, topk_out_width), -1, dtype=torch.int32, device=device
    )

    if context.is_prefill:
        cu_q = attn_metadata.cu_seqlens_q
        req_idx = _kpool_request_index(cu_q, n_tokens)
        chunk_start = None
        if attn_metadata.has_cached:
            cu_k0 = attn_metadata.cu_seqlens_k
            nreq = cu_q.shape[0] - 1
            chunk_start = (cu_k0[1 : nreq + 1] - cu_k0[:nreq]).to(torch.int64) - (
                cu_q[1:] - cu_q[:-1]
            ).to(torch.int64)
        _kpool_write_completed_pools(
            kv_cache,
            k,
            gate_score,
            positions,
            pool_bt,
            req_idx,
            compress_ape,
            index_kpool,
            head_dim,
            scale_fmt,
            pool_rows,
            chunk_start=chunk_start,
            tail_cache=tail_cache,
            state_slot_idx=state_slot_idx,
        )
        # The trailing incomplete pool has to outlive this forward; decode
        # finishes it one token at a time.
        kpool_ops.kpool_seed_tail(
            tail_cache, k, gate_score, positions, cu_q, state_slot_idx, index_kpool
        )
        _kpool_verify_cache(
            kv_cache,
            k,
            gate_score,
            positions,
            pool_bt,
            cu_q,
            compress_ape,
            index_kpool,
            head_dim,
        )
        if attn_metadata.max_seqlen_k <= topk_tokens:
            # Every pool would be selected; the caller runs dense, exactly as
            # the token-granular path does below its own threshold.
            return weights

        bs = cu_q.shape[0] - 1
        if attn_metadata.has_cached:
            cu_k = attn_metadata.cu_seqlens_k
            seq_lens_k = (cu_k[1 : bs + 1] - cu_k[:bs]).to(torch.int64)
        else:
            seq_lens_k = (cu_q[1:] - cu_q[:-1]).to(torch.int64)
        pool_counts = _kpool_pool_counts(seq_lens_k, index_kpool)
        pool_cu = torch.zeros(bs + 1, dtype=torch.int32, device=device)
        pool_cu[1:] = torch.cumsum(pool_counts, 0).to(torch.int32)

        # EXACTLY pool_cu[bs] rows, never more. `cp_gather_indexer_k_quant_cache`
        # resolves each destination row to a request by scanning cu_seq_lens; a
        # row past the last sequence matches nothing, leaves `batch_idx` as
        # uninitialized shared memory, and indexes the block table with garbage
        # -- an illegal memory access, surfacing later at whatever kernel next
        # synchronizes. The token-granular path is safe only because it sizes
        # this buffer to `total_kv` exactly, so do the same.
        #
        # This costs one D2H sync per prefill. Prefill is eager (never captured),
        # and the builder already does host-side work per batch, so it is free
        # in practice -- and much cheaper than being wrong.
        max_pools = int(pool_cu[bs].item())
        if max_pools <= 0:
            return weights
        k_fp8 = torch.empty([max_pools, head_dim], device=device, dtype=dtypes.fp8)
        k_scale = torch.empty([max_pools, 1], device=device, dtype=torch.float32)
        cp_gather_indexer_k_quant_cache(
            kv_cache, k_fp8, k_scale.view(dtypes.fp8), pool_bt, pool_cu, preshuffle=True
        )

        # Per-query causal window, in POOLS: a query at absolute position p sees
        # every pool that is COMPLETE at or before p, i.e. (p + 1) // kpool.
        pool_ks = pool_cu.to(torch.int64)[req_idx]
        pool_ke = pool_ks + (positions.to(torch.int64) + 1) // index_kpool
        pool_ks = pool_ks.to(torch.int32)
        pool_ke = pool_ke.to(torch.int32)

        pool_topk = torch.empty((n_tokens, select_k), dtype=torch.int32, device=device)
        # The logits buffer is [rows, max_pools] fp32 and max_pools is the sum
        # over co-scheduled requests, which `max_num_batched_tokens` does not
        # bound -- a burst of long-context requests can otherwise push one
        # allocation into the GiB range. Same query-row chunking the
        # token-granular path uses (deepseek_v2.py), and pooling has already
        # divided the column count by index_kpool. Each chunk still scores the
        # FULL pool set, so every row's top-k is exact with no cross-chunk merge.
        budget_bytes = SPARSE_INDEXER_LOGITS_BUDGET_MB * 1024 * 1024
        if (
            budget_bytes > 0
            and max_pools > 0
            and budget_bytes // (max_pools * 4) < n_tokens
        ):
            budget_rows = budget_bytes // (max_pools * 4)
            chunk_rows = (
                (budget_rows // 128) * 128
                if budget_rows >= 128
                else 1 << (max(1, budget_rows).bit_length() - 1)
            )
        else:
            chunk_rows = n_tokens
        for c0 in range(0, n_tokens, chunk_rows):
            c1 = min(c0 + chunk_rows, n_tokens)
            row_starts = pool_ks[c0:c1]
            row_ends = pool_ke[c0:c1]
            logits = fp8_mqa_logits(
                Q=q_fp8[c0:c1],
                KV=k_fp8,
                kv_scales=k_scale.squeeze(-1).contiguous(),
                weights=weights[c0:c1],
                cu_starts=row_starts,
                cu_ends=row_ends,
            )
            top_k_per_row_prefill(
                logits=logits,
                rowStarts=row_starts,
                rowEnds=row_ends,
                indices=pool_topk[c0:c1],
                values=None,
                numRows=c1 - c0,
                stride0=logits.stride(0),
                stride1=logits.stride(1),
                k=select_k,
                stable=stable_topk,
            )
        # `pool_topk` indexes the batch-wide gathered pool buffer, while the
        # converter below subtracts cu_seqlens_k[req] from each entry. Rebase
        # both ends inside the expansion kernel: pool ids become request-local,
        # emitted token ids get the request's key offset back.
        kpool_ops.expand_pools_and_append_tail(
            pool_topk,
            (positions.to(torch.int32) + 1),
            index_kpool,
            out=topk_indices,
            pool_base=pool_ks,
            tok_base=attn_metadata.cu_seqlens_k.to(torch.int32)[req_idx],
        )
        triton_convert_req_index_to_global_index_dsa_prefill(
            attn_metadata.sparse_cu_seqlens_q,
            attn_metadata.sparse_kv_indptr,
            attn_metadata.token_to_seq_idxs,
            topk_indices,
            attn_metadata.block_tables,
            attn_metadata.cu_seqlens_k,
            PAGE_SIZE=block_size,
            NUM_TOPK_TOKENS=topk_out_width,
            BLOCK_N=128,
            out=sparse_kv_indices_buffer,
        )
        return weights

    # ---- decode ----------------------------------------------------------
    bs = context.scheduled_bs
    pos = positions[:bs].to(torch.int64)
    pooled = kpool_ops.kpool_decode_stash_and_pool(
        tail_cache,
        k[:bs],
        gate_score[:bs],
        positions[:bs],
        state_slot_idx,
        compress_ape,
        index_kpool,
    )
    # Only a pool that closed on this token gets written; the rest carry a -1
    # slot, which the cache write skips.
    closes = (pos % index_kpool) == (index_kpool - 1)
    pool_ids = torch.where(closes, pos // index_kpool, torch.full_like(pos, -1))
    slots = kpool_ops.pool_slot_mapping(
        pool_bt,
        pool_ids,
        torch.arange(bs, device=device, dtype=torch.int64),
        pool_rows,
    )
    indexer_k_quant_and_cache(
        pooled, kv_cache, slots, head_dim, scale_fmt, preshuffle=True
    )

    seq_lens = attn_metadata.context_lens[:bs]
    pool_ctx = (seq_lens.to(torch.int32) // index_kpool).contiguous()
    pool_max_len = -(-max_model_len // index_kpool)
    logits = torch.empty([bs, pool_max_len], dtype=torch.float32, device=device)
    deepgemm_fp8_paged_mqa_logits(
        q_fp8[:bs].view(bs, 1, n_head, head_dim),
        kv_cache.unsqueeze(-2),
        weights[:bs],
        logits,
        pool_ctx,
        pool_bt,
        pool_max_len,
        KVBlockSize=pool_rows,
        Preshuffle=True,
    )
    pool_topk = torch.empty((bs, select_k), dtype=torch.int32, device=device)
    top_k_per_row_decode(
        logits,
        1,
        pool_ctx,
        pool_topk,
        bs,
        logits.stride(0),
        logits.stride(1),
        k=select_k,
        stable=stable_topk,
    )
    kpool_ops.expand_pools_and_append_tail(
        pool_topk,
        seq_lens.to(torch.int32),
        index_kpool,
        out=topk_indices[:bs],
    )
    triton_convert_req_index_to_global_index(
        attn_metadata.cu_seqlens_q,
        attn_metadata.kv_indptr,
        attn_metadata.sparse_kv_indptr,
        attn_metadata.kv_indices,
        topk_indices,
        NUM_TOPK_TOKENS=topk_out_width,
        out=sparse_kv_indices_buffer,
    )
    return weights


def _sparse_attn_indexer_kpool_fake(
    hidden_states: torch.Tensor,
    kv_cache: torch.Tensor,
    q_fp8: torch.Tensor,
    k: torch.Tensor,
    gate_score: torch.Tensor,
    weights: torch.Tensor,
    compress_ape: torch.Tensor,
    tail_cache: torch.Tensor,
    state_slot_idx: torch.Tensor,
    positions: torch.Tensor,
    sparse_kv_indices_buffer: torch.Tensor,
    topk_tokens: int,
    index_kpool: int,
    head_dim: int,
    max_model_len: int,
    topk_out_width: int,
    scale_fmt: str,
    stable_topk: bool,
) -> torch.Tensor:
    return torch.empty_like(weights, dtype=torch.float32)


direct_register_custom_op(
    op_name="sparse_attn_indexer_kpool",
    op_func=_sparse_attn_indexer_kpool,
    # The pooled cache write and the per-request tail stash are both in-place,
    # and the MLA reads the indices right after: without declaring them,
    # inductor is free to hoist that read above these writes.
    mutates_args=["sparse_kv_indices_buffer", "tail_cache", "kv_cache"],
    fake_impl=_sparse_attn_indexer_kpool_fake,
)
