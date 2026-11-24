# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""Inference-only MiniMax-M3 model support for ATOM."""

from typing import ClassVar

import functools
import os

import torch
from aiter import ActivationType, QuantType, dtypes
from aiter.dist.communication_op import tensor_model_parallel_all_reduce
from aiter.dist.parallel_state import (
    get_pp_group,
    get_tensor_model_parallel_world_size,
)
from aiter.ops.triton.gluon.pa_decode_gluon import pa_decode_gluon_supports_arch
from aiter.rotary_embedding import get_rope
from torch import nn
from transformers import PretrainedConfig

from atom.config import Config, QuantizationConfig
from atom.distributed.indexer_cp import indexer_cp_enabled
from atom.distributed.ulysses_sp import local_num_kv_heads
from atom.model_ops import module_dispatch_ops as _module_dispatch_ops  # noqa: F401
from atom.model_ops.attention_mha import SparseMHAPagedAttentionImpl
from atom.model_ops.base_attention import (
    Attention,
    _flydsl_arch_supported,
    _flydsl_plan_max_batch,
)
from atom.model_ops.embed_head import ParallelLMHead, VocabParallelEmbedding
from atom.model_ops.layernorm import (
    GemmaRMSNorm,
    fused_allreduce_gemma_rms_norm,
    fused_allreduce_gemma_rms_norm_qkv_codes,
    fused_allreduce_gemma_rms_norm_quant,
)
from atom.model_ops.linear import (
    MergedColumnParallelLinear,
    MinimaxM3QKVParallelLinearWithIndexer,
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
    mxfp4_k_pad_align,
)
from atom.model_ops.minimax_m3.sparse_attn import (
    SPARSE_BLOCK_SIZE,
)
from atom.model_ops.moe import FusedMoE
from atom.model_ops.mx8_fused import swiglu_oai_quant_fp8
from atom.model_ops.swiglu_oai import swiglu_oai_split
from atom.model_ops.utils import atom_parameter
from atom.models.minimax_m3 import ak_moe
from atom.models.minimax_m3 import sp_prefill
from atom.models.minimax_m3.mono.dispatch import MonoDecode
from atom.models.utils import (
    IntermediateTensors,
    PPMissingLayer,
    make_empty_intermediate_tensors_factory,
    make_layers,
    maybe_prefix,
    pp_holds_embedding,
)
from atom.quant_spec import effective_layer_quant
from atom.utils import envs
from atom.utils.decorators import support_torch_compile
from atom.utils.forward_context import get_forward_context
from atom.utils.region_graph import token_graph
from atom.utils.custom_register import direct_register_custom_op

# ---- ATOM_P1_DSPARK_AUX_NEXT (p1/decode-g9-16; integration-only, default 0 = off; value = minimum rows) ----
# A DSpark-tapped layer's aux is its output residual stream, ``all_reduce(hidden) + residual`` (``aux_hidden_state``):
# an AITER all-reduce + an add per tap per target forward (5 taps). The next fused all-reduce + add + Gemma norm (the
# next layer's input tail, or the final norm after the last layer) all-reduces the same partial into the same
# residual and returns ``bf16(fp32(x) + fp32(residual))`` with x = AITER's two-stage bits (amd_kernels
# gemma_add_rmsnorm_h6144/reference.py: "x of the all-reduce paths and the residual out: bit for bit"). On
# pure-decode forwards of at least this many rows (rows % 4 == 0) the tap skips its all-reduce + add and the next
# tail copies its residual out into the tap's capture buffer instead (``set_aux_hidden_state_rows``).
_P1_AUX_NEXT_MIN_ROWS = int(os.environ.get("ATOM_P1_DSPARK_AUX_NEXT", "0") or 0)


def _p1_aux_offset(rows: int) -> int:
    """The tap's first token when this forward defers its aux to the next tail, else -1 (same decision in the tap
    and in the tail: rows and the forward context only)."""
    if _P1_AUX_NEXT_MIN_ROWS <= 0 or rows < _P1_AUX_NEXT_MIN_ROWS or rows % 4:
        return -1
    try:
        ctx = get_forward_context().context
    except Exception:  # noqa: BLE001
        return -1
    if ctx is None or ctx.is_prefill or getattr(ctx, "is_draft", False) or ctx.ubatch_token_order is not None:
        return -1
    return int(ctx.ubatch_token_offset or 0)


_P1_AUX_BUFS: dict = {}  # tapped layer id -> the drafter's capture buffer (set_aux_hidden_state_rows)


def _p1_aux_rows(ctx, n: int):
    """The capture rows of this forward's n tokens (``drafter._aux_hook``'s rule)."""
    order = getattr(ctx, "ubatch_token_order", None) if ctx is not None else None
    if order is not None:
        return order
    off = int(getattr(ctx, "ubatch_token_offset", 0) or 0) if ctx is not None else 0
    return slice(off, off + n)


def _p1_aux_tap(hidden_states: torch.Tensor, residual: torch.Tensor, layer: int) -> None:
    """The tap of ``layer`` (writes its capture buffer itself, the hook records nothing): on deferred forwards
    nothing (the next tail writes it), else ``all_reduce(hidden) + residual`` as before. ``hidden_states`` is
    declared mutated only to keep the call ordered and alive in the compiled graph; it is not written."""
    if _p1_aux_offset(hidden_states.shape[0]) >= 0:
        return
    try:
        ctx = get_forward_context().context
    except Exception:  # noqa: BLE001
        ctx = None
    if ctx is not None and getattr(ctx, "is_draft", False):
        return
    buf = _P1_AUX_BUFS[layer]
    buf[_p1_aux_rows(ctx, hidden_states.shape[0])] = tensor_model_parallel_all_reduce(hidden_states) + residual


def _p1_aux_tap_fake(hidden_states: torch.Tensor, residual: torch.Tensor, layer: int) -> None:
    return None


def _p1_aux_capture(residual: torch.Tensor, layer: int) -> None:
    """The fused tail after ``layer``: on deferred forwards copy its residual out (== the tap's aux) into the
    tap's capture buffer. ``residual`` is declared mutated only to keep the call ordered; it is not written."""
    off = _p1_aux_offset(residual.shape[0])
    if off >= 0:
        _P1_AUX_BUFS[layer][off: off + residual.shape[0]].copy_(residual)


def _p1_aux_capture_fake(residual: torch.Tensor, layer: int) -> None:
    return None


if _P1_AUX_NEXT_MIN_ROWS > 0:
    # the capture buffers are NOT op arguments: as a graph input mutated twice (the hook and the tail) the compiled
    # graph cloned and copied back each whole buffer every step (+1.6 ms at 96 rows, L2 run)
    p1_aux_tap_op = direct_register_custom_op("p1_minimax_aux_tap", _p1_aux_tap, ["hidden_states"],
                                              _p1_aux_tap_fake)
    p1_aux_capture_op = direct_register_custom_op("p1_minimax_aux_capture", _p1_aux_capture, ["residual"],
                                                  _p1_aux_capture_fake)


def _get_text_config(config: PretrainedConfig) -> PretrainedConfig:
    return config.text_config if hasattr(config, "text_config") else config


def _sparse_attention_layer_ids(config: PretrainedConfig) -> set[int]:
    cfg = getattr(config, "sparse_attention_config", None)
    if not cfg:
        return set()
    freq = cfg.get("sparse_attention_freq")
    if freq is None:
        return set()
    return {i for i, enabled in enumerate(freq) if enabled != 0}


def _sparse_attention_layer_ordinals(config: PretrainedConfig) -> dict[int, int]:
    return {
        layer_id: ordinal
        for ordinal, layer_id in enumerate(sorted(_sparse_attention_layer_ids(config)))
    }


def _validate_sparse_flydsl_config(
    atom_config: Config, config: PretrainedConfig
) -> None:
    """Reject gfx1250 M3 sparse PA configurations that cannot be served."""
    if not _sparse_attention_layer_ids(config) or pa_decode_gluon_supports_arch():
        return
    if not envs.ATOM_PA_FLYDSL:
        raise ValueError(
            "MiniMax-M3 sparse paged attention on this GPU requires "
            "ATOM_PA_FLYDSL=1 because there is no gluon fallback"
        )
    if not _flydsl_arch_supported():
        raise ValueError(
            "MiniMax-M3 sparse paged attention requires a GPU supported by "
            "aiter's FlyDSL paged decode because there is no gluon fallback"
        )
    local_kv_heads = local_num_kv_heads(int(config.num_key_value_heads))
    max_tokens = int(atom_config.max_num_batched_tokens)
    max_rows = max_tokens * local_kv_heads
    plan_limit = _flydsl_plan_max_batch()
    if max_rows > plan_limit:
        raise ValueError(
            "MiniMax-M3 sparse paged attention can schedule at most "
            f"{plan_limit} FlyDSL rows, but max_num_batched_tokens "
            f"({max_tokens}) * local num_kv_heads ({local_kv_heads}) "
            f"= {max_rows}; lower --max-num-batched-tokens to at most "
            f"{plan_limit // local_kv_heads}"
        )


def _should_skip_minimax_m3_index_topk(
    config: PretrainedConfig, layer_id: int
) -> tuple[bool, int]:
    sparse_ordinals = _sparse_attention_layer_ordinals(config)
    sparse_ordinal = sparse_ordinals.get(layer_id, -1)
    if sparse_ordinal < 0:
        return False, sparse_ordinal
    if not getattr(config, "use_index_cache", False):
        return False, sparse_ordinal

    index_topk_freq = int(getattr(config, "index_topk_freq", 1) or 1)
    index_topk_pattern = getattr(config, "index_topk_pattern", None)
    if index_topk_pattern is not None:
        if 0 <= sparse_ordinal < len(index_topk_pattern):
            return index_topk_pattern[sparse_ordinal] == "S", sparse_ordinal
        return False, sparse_ordinal

    if index_topk_freq <= 0:
        raise ValueError("index_topk_freq must be a positive integer")
    if index_topk_freq == 1:
        return False, sparse_ordinal

    # MiniMax-M3 schedules sharing by sparse-layer ordinal, not absolute layer id.
    offset = int(getattr(config, "index_skip_topk_offset", 0))
    return max(sparse_ordinal - offset, 0) % index_topk_freq != 0, sparse_ordinal


def _is_moe_layer(config: PretrainedConfig, layer_id: int) -> bool:
    moe_layer_freq = getattr(config, "moe_layer_freq", None)
    if moe_layer_freq is None:
        return True
    return moe_layer_freq[layer_id] != 0


def _rope_theta(config: PretrainedConfig) -> float:
    return getattr(config, "rope_theta", 1000000.0)


def _linear_consumes_per_token_fp8(linear: nn.Module) -> bool:
    quant_type = getattr(linear, "quant_type", None)
    quant_type_value = getattr(quant_type, "value", quant_type)
    return (
        quant_type_value == QuantType.per_Token.value
        and getattr(linear, "params_dtype", None) == dtypes.fp8
    )


def _minimax_m3_cos_sin_cache(
    rotary_emb: nn.Module,
    query: torch.Tensor,
) -> torch.Tensor:
    cache_name = "_minimax_m3_cos_sin_cache"
    cos_cache = rotary_emb.cos_cache.squeeze(-2).squeeze(-2)
    cached = getattr(rotary_emb, cache_name, None)
    expected_shape = (*cos_cache.shape[:-1], cos_cache.shape[-1] * 2)
    if (
        cached is not None
        and cached.dtype == query.dtype
        and cached.device == query.device
        and tuple(cached.shape) == expected_shape
    ):
        return cached

    sin_cache = rotary_emb.sin_cache.squeeze(-2).squeeze(-2)
    if cos_cache.dtype != query.dtype or cos_cache.device != query.device:
        cos_cache = cos_cache.to(device=query.device, dtype=query.dtype)
        sin_cache = sin_cache.to(device=query.device, dtype=query.dtype)
    cos_sin_cache = torch.cat([cos_cache, sin_cache], dim=-1).contiguous()

    if torch.compiler.is_compiling():
        return cos_sin_cache

    if cache_name in rotary_emb._buffers:
        rotary_emb._buffers[cache_name] = cos_sin_cache
    else:
        rotary_emb.register_buffer(cache_name, cos_sin_cache, persistent=False)
    return cos_sin_cache


def make_minimax_m3_expert_params_mapping(
    num_experts: int,
) -> list[tuple[str, str, int, str]]:
    """Return loader mapping for MiniMax-M3 split expert checkpoint weights."""
    mapping: list[tuple[str, str, int, str]] = []
    for expert_id in range(num_experts):
        for shard_id, weight_names in (
            ("w1", ("w1", "gate_proj")),
            ("w2", ("w2", "down_proj")),
            ("w3", ("w3", "up_proj")),
        ):
            # weight_scale_2 can be easily replaced using the mapping provided here.
            if shard_id in ("w1", "w3"):
                param_prefix = "experts.w13_"
                scale_param = "experts.w13_weight_scale"
            else:
                param_prefix = "experts.w2_"
                scale_param = "experts.w2_weight_scale"
            for weight_name in weight_names:
                for scale_name in ("scale", "weight_scale"):
                    mapping.append(
                        (
                            scale_param,
                            f"experts.{expert_id}.{weight_name}.{scale_name}",
                            expert_id,
                            shard_id,
                        )
                    )
                mapping.append(
                    (
                        param_prefix,
                        f"experts.{expert_id}.{weight_name}.",
                        expert_id,
                        shard_id,
                    )
                )
    return mapping


# p1/dense-gemm ``ATOM_DG_DENSE_AK=1`` (default off): dense layers 0-2 at the measured-win rows only: ATOM's gate_up,
# then amd-kernels' SwiGLU-OAI writing tiled MXFP8 codes (bit-identical codes+scales to swiglu_oai_quant_fp8 at
# every measured M) and its prequant down GEMM. Measured (p1/dense-gemm/NOTES.md): M32 -0.10 us, M64 -0.68 us
# (bit-identical), M128 -0.92 us, M384 -2.95 us (fp32 summation order in the down GEMM, <= 1 bf16 ulp).
# Slower at M4/8/16/256: those rows keep the ATOM chain.
_DG_DENSE_AK = os.environ.get("ATOM_DG_DENSE_AK", "0") == "1"
_DG_DENSE_AK_ROWS = frozenset((32, 64, 128, 384))
_DG_DENSE_AK_WARM = [False]


def _dg_dense_ak_ready(device) -> bool:
    """Compile the SwiGLU and prequant down kernels on the first eager call; never inside a graph capture."""
    if _DG_DENSE_AK_WARM[0]:
        return True
    if torch.cuda.is_current_stream_capturing():
        return False
    from amd_kernels.activation.swiglu_oai_mxfp8_i3072 import warmup_swiglu_oai_mxfp8

    warmup_swiglu_oai_mxfp8(device, forms=("fp32",))
    _DG_DENSE_AK_WARM[0] = True
    return True


class MiniMaxM3MLP(nn.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        intermediate_size: int,
        quant_config: QuantizationConfig | None = None,
        reduce_results: bool = True,
        prefix: str = "",
        hidden_act: str | None = None,
    ) -> None:
        super().__init__()
        # An MXFP4 down_proj cannot take an unaligned K (its per-rank
        # intermediate); the pair pads it together. See mxfp4_k_pad_align.
        intermediate_pad = mxfp4_k_pad_align(quant_config, f"{prefix}.down_proj")
        self.gate_up_proj = MergedColumnParallelLinear(
            config.hidden_size,
            [intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_up_proj",
            output_pad_align=intermediate_pad,
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            config.hidden_size,
            bias=False,
            quant_config=quant_config,
            reduce_results=reduce_results,
            prefix=f"{prefix}.down_proj",
            input_pad_align=intermediate_pad,
        )
        hidden_act = hidden_act or config.hidden_act
        if hidden_act != "swigluoai":
            raise ValueError(
                f"Unsupported MiniMax-M3 activation {hidden_act!r}; "
                "expected 'swigluoai'."
            )
        self.swiglu_alpha = getattr(config, "swiglu_alpha", 1.702)
        self.swiglu_beta = getattr(config, "swiglu_beta", 1.0)
        self.swiglu_limit = getattr(config, "swiglu_limit", 7.0)
        # A native MXFP8 1x32 down_proj quantizes its input to E4M3 / E8M0 group 32
        # itself; the activation kernel emits those bytes directly instead.
        self.act_quant_fp8 = (
            getattr(self.down_proj, "native_a8_group_rows", None) == 1
            and self.down_proj.params_dtype == torch.float8_e4m3fn
        )

    def forward(
        self, x: torch.Tensor, x_scale: torch.Tensor | None = None
    ) -> torch.Tensor:
        gate_up = self.gate_up_proj(x, x_scale=x_scale)
        dg_ws = getattr(self.down_proj, "_ak_weight_scale", None) if _DG_DENSE_AK else None
        if (dg_ws is not None and self.act_quant_fp8 and gate_up.shape[-1] == 6144 and gate_up.dim() == 2
                and gate_up.shape[0] in _DG_DENSE_AK_ROWS and _dg_dense_ak_ready(gate_up.device)):
            # p1/dense-gemm ATOM_DG_DENSE_AK=1: same fp32 SwiGLU-OAI + bf16 rounding + MXFP8 rule as
            # swiglu_oai_quant_fp8, codes written where amd-kernels' down GEMM reads them (no quantizer launch)
            from amd_kernels.activation.swiglu_oai_mxfp8_i3072 import swiglu_oai_mxfp8
            from amd_kernels.gemm.mxfp8_common import gemm as _akg

            m = gate_up.shape[0]
            tile = _akg.prequant_code_tile_rows(m, 6144, 3072, torch.bfloat16)
            qd, sd = swiglu_oai_mxfp8(gate_up, self.swiglu_alpha, self.swiglu_beta, self.swiglu_limit, tile,
                                      form="fp32")
            return _akg.mxfp8_gemm_prequant(qd, sd, self.down_proj.weight, dg_ws, torch.bfloat16, m)
        if self.act_quant_fp8:
            xq, xs = swiglu_oai_quant_fp8(
                gate_up,
                alpha=self.swiglu_alpha,
                beta=self.swiglu_beta,
                limit=self.swiglu_limit,
            )
            return self.down_proj(xq, x_scale=xs)
        x = swiglu_oai_split(
            gate_up,
            alpha=self.swiglu_alpha,
            beta=self.swiglu_beta,
            limit=self.swiglu_limit,
        )
        return self.down_proj(x)


class MiniMaxM3MoE(nn.Module):
    """MiniMax-M3 routed MoE for MXFP4 checkpoints."""

    def __init__(
        self,
        config: PretrainedConfig,
        layer_id: int,
        quant_config: QuantizationConfig | None = None,
        params_dtype: torch.dtype | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        del layer_id
        tp_size = get_tensor_model_parallel_world_size()
        if tp_size > config.num_local_experts:
            raise ValueError(
                f"Tensor parallel size {tp_size} is greater than "
                f"the number of experts {config.num_local_experts}."
            )

        if getattr(config, "use_routing_bias", False):
            self.e_score_correction_bias = atom_parameter(
                torch.empty(config.num_local_experts, dtype=torch.float32)
            )
        else:
            self.register_parameter("e_score_correction_bias", None)

        self.gate = ReplicatedLinear(
            config.hidden_size,
            config.num_local_experts,
            bias=False,
            quant_config=None,
            prefix=f"{prefix}.gate",
        )
        # The checkpoint stores router weights as fp32, but routing tolerates bf16
        # logits. Let the weight loader cast once instead of casting every forward.

        self.routed_scaling_factor = getattr(config, "routed_scaling_factor", 1.0)
        self.experts = FusedMoE(
            num_experts=config.num_local_experts,
            top_k=config.num_experts_per_tok,
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            params_dtype=params_dtype,
            reduce_results=False,
            renormalize=True,
            activation=ActivationType.Swiglu,
            scoring_func=getattr(config, "scoring_func", "sigmoid"),
            e_score_correction_bias=self.e_score_correction_bias,
            quant_config=quant_config,
            prefix=f"{prefix}.experts",
            config=config,
            shared_expert_prefix=f"{prefix}.shared_experts",
        )
        if hasattr(self.experts.quant_method, "intermediate_pad"):
            # MiniMax-M3 pads expert weights at load time; computing the full
            # padded intermediate avoids backend pad-skip precision issues.
            self.experts.quant_method.intermediate_pad = 0
        self.experts.swiglu_limit = getattr(config, "swiglu_limit", 7.0)
        self.fuse_shared_experts = (
            getattr(self.experts, "num_fused_shared_experts", 0) > 0
        )

        self.shared_experts: MiniMaxM3MLP | None = None
        if getattr(config, "n_shared_experts", 0) and not self.fuse_shared_experts:
            self.shared_experts = MiniMaxM3MLP(
                config=config,
                intermediate_size=config.intermediate_size * config.n_shared_experts,
                quant_config=quant_config,
                reduce_results=False,
                prefix=f"{prefix}.shared_experts",
            )
        # last child: its post-load hook runs after the experts' and shared expert's
        self.ak_moe = (
            ak_moe.AkMoe(self, prefix)
            if ak_moe.enabled() and self.shared_experts is not None
            else None
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.ak_moe is None:
            return self.aiter_forward(hidden_states)
        orig_shape = hidden_states.shape
        out = ak_moe.ak_moe_op(
            hidden_states.view(-1, orig_shape[-1]), self.ak_moe.prefix
        )
        return out.view(orig_shape)

    def aiter_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        orig_shape = hidden_states.shape
        hidden_states = hidden_states.view(-1, orig_shape[-1])
        router_logits = self.gate(hidden_states)

        routed_output = self.experts(
            hidden_states=hidden_states,
            router_logits=router_logits,
        )
        if not self.fuse_shared_experts and self.routed_scaling_factor != 1.0:
            routed_output = routed_output * self.routed_scaling_factor

        if self.shared_experts is not None:
            routed_output = routed_output + self.shared_experts(hidden_states)

        return routed_output.view(orig_shape)


class MiniMaxM3Attention(nn.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        layer_id: int,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        cache_config: str = "bf16",
        index_cache_config: str = "auto",
    ) -> None:
        super().__init__()
        self.layer_num = layer_id
        self.hidden_size = config.hidden_size
        self.tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = config.num_attention_heads
        self.num_heads = self.total_num_heads // self.tp_size
        self.total_num_kv_heads = config.num_key_value_heads
        self.num_kv_heads = max(1, self.total_num_kv_heads // self.tp_size)
        self.head_dim = config.head_dim
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5

        self.qkv_proj = QKVParallelLinear(
            self.hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            self.hidden_size,
            bias=False,
            reduce_results=False,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )
        self.q_norm = GemmaRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = GemmaRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        rotary_dim = int(self.head_dim * getattr(config, "partial_rotary_factor", 1.0))
        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=rotary_dim,
            max_position=config.max_position_embeddings,
            base=_rope_theta(config),
            rope_scaling=getattr(config, "rope_scaling", None),
        )
        _minimax_m3_cos_sin_cache(self.rotary_emb, self.q_norm.weight)
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            self.num_kv_heads,
            kv_cache_dtype=cache_config,
            layer_num=layer_id,
            use_mla=False,
            rotary_emb=self.rotary_emb,
            q_norm=self.q_norm,
            k_norm=self.k_norm,
            prefix=f"{prefix}.attn",
            per_token_fp8_out=effective_layer_quant(quant_config, f"{prefix}.o_proj")
            == (QuantType.per_Token, dtypes.fp8),
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        hidden_states_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        qkv = self.qkv_proj(hidden_states, x_scale=hidden_states_scale)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        attn_output = self.attn(q, k, v, positions=positions, qkv=qkv)
        if isinstance(attn_output, tuple):
            return self.o_proj(*attn_output)
        return self.o_proj(attn_output)


class MiniMaxM3SparseAttention(nn.Module):
    """Native ATOM MiniMax-M3 lightning-indexer sparse attention."""

    def __init__(
        self,
        config: PretrainedConfig,
        layer_id: int,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        cache_config: str = "bf16",
        index_cache_config: str = "auto",
    ) -> None:
        super().__init__()
        self.is_indexed_sparse_attention = True
        self.hidden_size = config.hidden_size
        self.layer_num = layer_id
        self.layer_name = f"{prefix}.attn"
        self.tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = config.num_attention_heads
        if self.total_num_heads % self.tp_size != 0:
            raise ValueError("num_attention_heads must be divisible by TP size.")
        self.num_heads = self.total_num_heads // self.tp_size
        self.total_num_kv_heads = config.num_key_value_heads
        if self.total_num_kv_heads >= self.tp_size:
            if self.total_num_kv_heads % self.tp_size != 0:
                raise ValueError("num_key_value_heads must divide TP size.")
        elif self.tp_size % self.total_num_kv_heads != 0:
            raise ValueError("TP size must divide num_key_value_heads replication.")
        self.num_kv_heads = max(1, self.total_num_kv_heads // self.tp_size)
        self.head_dim = config.head_dim
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.kv_cache_dtype = cache_config

        sparse_cfg = config.sparse_attention_config
        sparse_block_size = sparse_cfg["sparse_block_size"]
        if sparse_block_size != SPARSE_BLOCK_SIZE:
            raise ValueError(
                "MiniMax-M3 native sparse attention requires sparse_block_size "
                f"{SPARSE_BLOCK_SIZE}, got {sparse_block_size}."
            )
        self.total_idx_heads = sparse_cfg["sparse_num_index_heads"]
        # Indexer-only CP projects EVERY index head on every rank and shards the
        # context instead, so the width is the full head count rather than this
        # rank's kv-head share. `index_q_size` and the fused-qkv split in
        # `forward` are both derived from this, so they follow with no edit --
        # and they must: `aiter.fused_qknorm_idxrqknorm` finds index_q by offset
        # inside `qkv`, so the wide tensor has to reach it intact.
        #
        # This is an __init__-time value, not a traced branch: `forward` closes
        # over the resulting int exactly as it already does for TP2 vs TP4.
        self.num_idx_heads = (
            self.total_idx_heads if indexer_cp_enabled() else self.num_kv_heads
        )
        self.idx_head_dim = sparse_cfg["sparse_index_dim"]
        self.index_q_size = self.num_idx_heads * self.idx_head_dim
        self.topk_blocks = sparse_cfg["sparse_topk_blocks"]
        self.init_blocks = sparse_cfg.get("sparse_init_block", 0)
        self.local_blocks = sparse_cfg.get("sparse_local_block", 0)
        self.skip_index_topk, self.sparse_layer_ordinal = (
            _should_skip_minimax_m3_index_topk(config, layer_id)
        )
        score_type = sparse_cfg.get("sparse_score_type", "max")
        if score_type != "max":
            raise ValueError(
                "MiniMax-M3 native sparse attention only supports "
                f"sparse_score_type='max', got {score_type!r}."
            )

        self.qkv_proj = MinimaxM3QKVParallelLinearWithIndexer(
            self.hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            self.total_idx_heads,
            self.idx_head_dim,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            self.hidden_size,
            bias=False,
            reduce_results=False,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )
        # SP prefill's replicated o_proj (sp_prefill.py), None unless enabled.
        self.o_proj_full = sp_prefill.make_o_proj_full(
            self.o_proj,
            self.total_num_heads * self.head_dim,
            self.hidden_size,
            quant_config,
            f"{prefix}.o_proj",
        )

        self.q_norm = GemmaRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = GemmaRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        rotary_dim = int(self.head_dim * getattr(config, "partial_rotary_factor", 1.0))
        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=rotary_dim,
            max_position=config.max_position_embeddings,
            base=_rope_theta(config),
            rope_scaling=getattr(config, "rope_scaling", None),
        )
        _minimax_m3_cos_sin_cache(self.rotary_emb, self.q_norm.weight)
        self.index_q_norm = GemmaRMSNorm(self.idx_head_dim, eps=config.rms_norm_eps)
        self.index_k_norm = GemmaRMSNorm(self.idx_head_dim, eps=config.rms_norm_eps)
        self.index_rotary_emb = self.rotary_emb

        # First-class atom attention: plug in the MiniMax-M3 sparse impl, which
        # owns all sparse/fp8/gluon behavior (fused qk/index norm+rope+SHUFFLE KV
        # insert in rope_cache; index top-k -> page-16 sparse block table -> gluon
        # PA in dispatch_backend). The standard AiterAttentionMetadataBuilder binds
        # the page-16 SHUFFLE KV cache + scales (KVCacheTensor) and the page-128
        # index cache (onto the impl). All indexer state lives on the impl.
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            self.num_kv_heads,
            kv_cache_dtype=cache_config,
            layer_num=layer_id,
            use_mla=False,
            rotary_emb=self.rotary_emb,
            q_norm=self.q_norm,
            k_norm=self.k_norm,
            prefix=f"{prefix}.attn",
            impl_cls=SparseMHAPagedAttentionImpl,
            # --- MiniMax-M3 sparse-attention indexer kwargs (impl-local) ---
            index_q_norm=self.index_q_norm,
            index_k_norm=self.index_k_norm,
            index_rotary_emb=self.index_rotary_emb,
            index_head_dim=self.idx_head_dim,
            topk=self.topk_blocks,
            init_blocks=self.init_blocks,
            local_blocks=self.local_blocks,
            skip_index_topk=self.skip_index_topk,
            sparse_layer_ordinal=self.sparse_layer_ordinal,
            index_cache_dtype=index_cache_config,
            per_token_fp8_out=effective_layer_quant(quant_config, f"{prefix}.o_proj")
            == (QuantType.per_Token, dtypes.fp8),
        )
        # p1/glue-front (default off): decode rows the chain takes run this qkv linear as MXFP8 quant only and the
        # attention op's insert as the fused QKV GEMM + norm/RoPE/NVFP4 insert (decided at run time, p1_front.take).
        from atom.model_ops.minimax_m3 import p1_front

        impl = self.attn.impl
        self.qkv_proj._p1_front = p1_front.layer_ok(
            self.num_heads, self.num_kv_heads, self.num_idx_heads, rotary_dim, cache_config,
            getattr(impl, "index_cache_dtype", None), self.skip_index_topk, getattr(impl, "per_token_fp8_out", False),
        )

    def attend(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        hidden_states_scale: torch.Tensor | None = None,
        fuse_oproj: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """qkv + attention: this rank's heads of every token (o_proj's input)."""
        # Keep index Q/K packed with main QKV. Layers that reuse cached top-k skip
        # the indexer norm/rope/top-k path, but still compute the packed GEMM.
        qkv = self.qkv_proj(hidden_states, x_scale=hidden_states_scale)
        q, k, v, _, _ = qkv.split(
            [
                self.q_size,
                self.kv_size,
                self.kv_size,
                self.index_q_size,
                self.idx_head_dim,
            ],
            dim=-1,
        )
        if fuse_oproj:
            # ATOM_M3_AK_DECODE_QOUT: attention + o_proj in one op (decode only; the caller checked _ak_oproj)
            return self.attn.forward_ak_oproj(q, k, v, positions, qkv, self.o_proj)
        return self.attn(q, k, v, positions, qkv=qkv)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        hidden_states_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self._ak_oproj():
            return self.attend(positions, hidden_states, hidden_states_scale, fuse_oproj=True)
        attn_output = self.attend(positions, hidden_states, hidden_states_scale)
        if isinstance(attn_output, tuple):
            return self.o_proj(*attn_output)
        return self.o_proj(attn_output)

    def _ak_oproj(self) -> bool:
        """ATOM_M3_AK_DECODE_QOUT (p1/attn-decode, default off): attention + o_proj in one op so
        the amd-kernels decode attention can write o_proj's MXFP8 input. Needs the o_proj on
        amd-kernels' MXFP8 GEMM (ATOM_AK_MXFP8_GEMM=1) and the bf16 attention output."""
        from atom.model_ops.minimax_m3 import ak_kernels

        return (
            ak_kernels.QOUT
            and ak_kernels.DECODE
            and getattr(self.o_proj, "_ak_weight_scale", None) is not None
            and getattr(self.o_proj, "bias", None) is None
            and not getattr(self.attn, "per_token_fp8_out", False)
            and getattr(self.attn, "sp_columns", None) is None
        )


class MiniMaxM3DecoderLayer(nn.Module):
    _p1_aux_prev = None  # ATOM_P1_DSPARK_AUX_NEXT: id of the tapped layer before this one
    def __init__(
        self,
        config: PretrainedConfig,
        prefix: str,
        cache_config: str = "bf16",
        index_cache_config: str = "auto",
        quant_config: QuantizationConfig | None = None,
        params_dtype: torch.dtype | None = None,
        layer_num: int = 0,
    ) -> None:
        super().__init__()
        attn_cls = (
            MiniMaxM3SparseAttention
            if layer_num in _sparse_attention_layer_ids(config)
            else MiniMaxM3Attention
        )
        self.self_attn = attn_cls(
            config=config,
            layer_id=layer_num,
            quant_config=quant_config,
            prefix=f"{prefix}.self_attn",
            cache_config=cache_config,
            index_cache_config=index_cache_config,
        )

        self.is_moe_layer = _is_moe_layer(config, layer_num)
        if self.is_moe_layer:
            self.block_sparse_moe = MiniMaxM3MoE(
                config=config,
                layer_id=layer_num,
                quant_config=quant_config,
                params_dtype=params_dtype,
                prefix=f"{prefix}.block_sparse_moe",
            )
        else:
            self.mlp = MiniMaxM3MLP(
                config=config,
                intermediate_size=config.dense_intermediate_size,
                quant_config=quant_config,
                reduce_results=False,
                prefix=f"{prefix}.mlp",
            )

        self.input_layernorm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = GemmaRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
        capture_aux: bool = False,
        sp: "sp_prefill.SeqParallelPrefill | None" = None,
    ) -> (
        tuple[torch.Tensor, torch.Tensor]
        | tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ):
        if sp is not None:
            return sp.layer_forward(self, positions, hidden_states, residual)
        hidden_states_scale = None
        fuse_input_ar_rmsnorm_quant = _linear_consumes_per_token_fp8(
            self.self_attn.qkv_proj
        )
        fuse_post_attention_ar_rmsnorm_quant = (
            not self.is_moe_layer
            and _linear_consumes_per_token_fp8(self.mlp.gate_up_proj)
        )
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        elif fuse_input_ar_rmsnorm_quant:
            hidden_states, hidden_states_scale, residual = (
                fused_allreduce_gemma_rms_norm_quant(
                    hidden_states, residual, self.input_layernorm
                )
            )
        elif self._p1_ar_codes():
            # p1/ar-epilogue chain Y: the tail emits the qkv entry's MXFP8 codes (hidden_states not computed on
            # the chain's rows; only this layer's qkv linear reads it)
            hidden_states, residual = fused_allreduce_gemma_rms_norm_qkv_codes(
                hidden_states, residual, self.input_layernorm
            )
        else:
            hidden_states, residual = fused_allreduce_gemma_rms_norm(
                hidden_states, residual, self.input_layernorm
            )

        # Eagle3 aux hidden state = the all-reduced residual stream entering this
        # layer (post input-norm). Captured here, not as `hidden_states + residual`
        # in the model loop, because M3's fused all-reduce RMSNorm leaves that sum
        # TP-partial / NaN-prone under CUDAGraph.
        if self._p1_aux_prev is not None:  # ATOM_P1_DSPARK_AUX_NEXT: the previous layer's deferred tap
            p1_aux_capture_op(residual, self._p1_aux_prev)
        aux_hidden_state = residual.clone() if capture_aux else None

        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            hidden_states_scale=hidden_states_scale,
        )
        ffn = self.block_sparse_moe if self.is_moe_layer else self.mlp
        if fuse_post_attention_ar_rmsnorm_quant:
            hidden_states, hidden_states_scale, residual = (
                fused_allreduce_gemma_rms_norm_quant(
                    hidden_states, residual, self.post_attention_layernorm
                )
            )
            hidden_states = ffn(hidden_states, x_scale=hidden_states_scale)
        else:
            hidden_states, residual = fused_allreduce_gemma_rms_norm(
                hidden_states, residual, self.post_attention_layernorm
            )
            hidden_states = ffn(hidden_states)
        if aux_hidden_state is not None:
            return hidden_states, residual, aux_hidden_state
        return hidden_states, residual

    def _p1_ar_codes(self) -> bool:
        """ATOM_P1_AR_QKV_CODES (p1/ar-epilogue, default off): this layer's qkv linear runs p1_front's chain on
        amd-kernels' MXFP8 GEMM, so its pre-attention tail may emit the codes (decided per call at run time)."""
        from atom.model_ops.minimax_m3 import p1_front

        qkv = self.self_attn.qkv_proj
        return (
            p1_front.AR_CODES
            and getattr(qkv, "_p1_front", False)
            and getattr(qkv, "_ak_weight_scale", None) is not None
            and getattr(qkv, "bias", None) is None
        )

    @staticmethod
    def aux_hidden_state(output: tuple) -> torch.Tensor:
        """This layer's output as a DSpark draft reads it: the residual stream
        after the layer, ``hidden + residual``.

        ``hidden`` is TP-partial -- o_proj and the FFN skip their all-reduce,
        which the next layer's fused all-reduce RMSNorm performs -- so it is
        reduced here first. That costs one all-reduce per tapped layer per
        target forward, only when a DSpark drafter taps this layer.
        """
        if isinstance(output, sp_prefill.ShardedOutput):
            return output.sp.aux_hidden_state(output)
        hidden_states, residual = output[0], output[1]
        if get_tensor_model_parallel_world_size() > 1:
            hidden_states = tensor_model_parallel_all_reduce(hidden_states)
        return hidden_states + residual


def _p1_aux_hidden_state(output, layer: int):
    """``MiniMaxM3DecoderLayer.aux_hidden_state`` with ATOM_P1_DSPARK_AUX_NEXT: ``_p1_aux_tap`` writes the capture
    buffer (or leaves it to the next tail), so the hook records nothing (None)."""
    if isinstance(output, sp_prefill.ShardedOutput):
        return output.sp.aux_hidden_state(output)
    hidden_states, residual = output[0], output[1]
    p1_aux_tap_op(hidden_states, residual, layer)
    return None


@token_graph
@support_torch_compile
class MiniMaxM3Model(nn.Module):
    def __init__(
        self,
        atom_config: Config,
        prefix: str = "",
        layer_type: type[nn.Module] = MiniMaxM3DecoderLayer,
    ) -> None:
        super().__init__()
        config = _get_text_config(atom_config.hf_config)
        _validate_sparse_flydsl_config(atom_config, config)
        self.config = config
        cache_config = atom_config.kv_cache_dtype
        index_cache_config = atom_config.index_cache_dtype
        quant_config = atom_config.quant_config

        if pp_holds_embedding(atom_config):
            self.embed_tokens = VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
            )
        else:
            self.embed_tokens = PPMissingLayer()

        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix, layer_num=None: layer_type(
                config,
                prefix,
                cache_config=cache_config,
                index_cache_config=index_cache_config,
                quant_config=quant_config,
                layer_num=layer_num,
                params_dtype=atom_config.torch_dtype,
            ),
            prefix=f"{prefix}.layers",
            layer_num_offset=0,
        )

        if get_pp_group().is_last_rank:
            self.norm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.norm = PPMissingLayer()

        # Eagle3 aux hidden-state capture layer ids. Empty unless an Eagle3 drafter
        # registers them via MiniMaxM3SparseForCausalLM.set_aux_hidden_state_layers.
        self.aux_hidden_state_layers: tuple[int, ...] = ()

        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states", "residual"], config.hidden_size
        )
        self._p1_aux_last = None

    def set_aux_hidden_state_rows(self, layer_ids, buffers) -> None:
        """ATOM_P1_DSPARK_AUX_NEXT: register each DSpark tap's capture buffer, bind the tap to the fused tail after
        the tapped layer (the next layer's input tail; the final norm after the last layer) and route the tap
        through ``_p1_aux_tap``. Called once by the drafter after it installed its hooks; no-op with the flag off."""
        if _P1_AUX_NEXT_MIN_ROWS <= 0 or get_tensor_model_parallel_world_size() <= 1:
            return
        for lid, buf in zip(layer_ids, buffers):
            if lid < self.start_layer or lid >= self.end_layer or buf.numel() == 0:
                continue
            if lid + 1 < self.end_layer:
                self.layers[lid + 1]._p1_aux_prev = lid
            else:
                # the last layer keeps its own tap: the final norm's residual out has no later reader, so a capture
                # op "mutating" it is dead code to the compiler and was dropped (S2: layer 59's aux went stale)
                continue
            _P1_AUX_BUFS[lid] = buf
            # instance attribute over the staticmethod: the hook gets None (records nothing), the op writes
            self.layers[lid].aux_hidden_state = functools.partial(_p1_aux_hidden_state, layer=lid)

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors | tuple[torch.Tensor, list[torch.Tensor]]:
        if get_pp_group().is_first_rank:
            hidden_states = (
                inputs_embeds
                if inputs_embeds is not None
                else self.get_input_embeddings(input_ids)
            )
            residual = None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]

        aux_hidden_states: list[torch.Tensor] = []
        for idx in range(self.start_layer, self.end_layer):
            if idx in self.aux_hidden_state_layers:
                hidden_states, residual, aux_hidden_state = self.layers[idx](
                    positions, hidden_states, residual, capture_aux=True
                )
                aux_hidden_states.append(aux_hidden_state)
            else:
                hidden_states, residual = self.layers[idx](
                    positions, hidden_states, residual
                )

        if not get_pp_group().is_last_rank:
            return IntermediateTensors(
                {"hidden_states": hidden_states, "residual": residual}
            )

        hidden_states, residual = fused_allreduce_gemma_rms_norm(
            hidden_states, residual, self.norm
        )
        if self._p1_aux_last is not None:  # ATOM_P1_DSPARK_AUX_NEXT: the last layer's deferred tap
            p1_aux_capture_op(residual, self._p1_aux_last)
        if aux_hidden_states:
            return hidden_states, aux_hidden_states
        return hidden_states

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        num_fused_shared = getattr(self.config, "n_shared_experts", 0) or 0
        return make_minimax_m3_expert_params_mapping(
            self.config.num_local_experts + num_fused_shared
        )


class MiniMaxM3SparseForCausalLM(nn.Module):
    # The verify head (`compute_head`) draws from the temperature alone.
    head_ids_top_p = False
    packed_modules_mapping: ClassVar[dict[str, tuple[str, str | int]]] = {
        ".index_q_proj": (".qkv_proj", "index_q"),
        ".index_k_proj": (".qkv_proj", "index_k"),
        ".q_proj": (".qkv_proj", "q"),
        ".k_proj": (".qkv_proj", "k"),
        ".v_proj": (".qkv_proj", "v"),
        ".gate_proj": (".gate_up_proj", 0),
        ".up_proj": (".gate_up_proj", 1),
    }

    def __init__(
        self,
        atom_config: Config,
        prefix: str = "",
        layer_type: type[nn.Module] = MiniMaxM3DecoderLayer,
    ) -> None:
        super().__init__()
        config = _get_text_config(atom_config.hf_config)
        self.config = config
        self.model = MiniMaxM3Model(
            atom_config=atom_config,
            prefix=maybe_prefix(prefix, "model"),
            layer_type=layer_type,
        )

        if get_pp_group().is_last_rank:
            self.lm_head = ParallelLMHead(
                config.vocab_size,
                config.hidden_size,
                org_num_embeddings=config.vocab_size,
                prefix=maybe_prefix(prefix, "lm_head"),
            )
        else:
            self.lm_head = PPMissingLayer()

        if getattr(config, "tie_word_embeddings", False):
            self.lm_head.weight = self.model.embed_tokens.weight

        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors
        )
        self._mono = MonoDecode(self, atom_config)
        # The in-graph DSpark verify head (`verify_head.M3VerifyHead`), bound by
        # `install_mono_decode`; it reads the step's input ids `forward` saw.
        self._verify_head = None
        self._step_input_ids = None
        # (positions, slot_mapping) -> None: the DSpark draft's context write of
        # a decode step, bound by `install_mono_decode`.
        self._draft_context_writer = None
        self._sp = sp_prefill.SeqParallelPrefill.create(
            self.model,
            _linear_consumes_per_token_fp8,
            atom_config.max_num_batched_tokens,
        )
        if envs.ATOM_M3_SP_DUMP:
            sp_prefill.install_dump_hooks(
                self, self.model, envs.ATOM_M3_SP_DUMP_LAYERS
            )
        # Set by a VL model with a vision tower: its compiled MiniMaxM3Model is
        # captured once, so it must always see inputs_embeds -- a text-only
        # capture would replay without them and drop the image embeddings.
        self.embed_before_model = False

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.get_input_embeddings(input_ids)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.get_input_embeddings(input_ids)

    def set_aux_hidden_state_layers(self, layers: tuple[int, ...]) -> None:
        self.model.aux_hidden_state_layers = layers

    def bind_mono_drafter(self, drafter) -> None:
        self._mono.bind_drafter(drafter)

    def get_eagle3_aux_hidden_state_layers(self) -> tuple[int, ...]:
        """Default Eagle3 aux hidden-state layer ids: early / middle / late of
        the target model (early=2, mid=n//2, late=n-3), matching vLLM's default.
        """
        num_layers = len(self.model.layers)
        return (2, num_layers // 2, num_layers - 3)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **_: object,
    ) -> torch.Tensor | IntermediateTensors:
        self._step_input_ids = input_ids
        # Outside the compiled MiniMaxM3Model on purpose: a supported decode step of
        # up to MAX_TOKENS tokens runs the mono kernels, everything else the
        # original model.
        if self._mono.supports(
            input_ids, positions, intermediate_tensors, inputs_embeds
        ):
            out = self._mono.forward(input_ids, positions)
        elif self._sp is not None and self._sp.supports(
            input_ids, positions, intermediate_tensors, inputs_embeds
        ):
            out = self._sp.forward(input_ids, positions)
        else:
            if self.embed_before_model and inputs_embeds is None:
                inputs_embeds = self.embed_input_ids(input_ids)
            out = self.model(input_ids, positions, intermediate_tensors, inputs_embeds)
        if self._draft_context_writer is not None:
            self._write_draft_context(positions)
        return out

    def _write_draft_context(self, positions: torch.Tensor) -> None:
        """A decode step's DSpark context K/V, inside the forward (so inside its
        graph); the proposer skips its own write on the same steps
        (`MiniMaxM3DSpark.context_in_target`)."""
        fwd = get_forward_context()
        ctx = fwd.context
        md = fwd.attn_metadata
        if (
            ctx is None
            or ctx.is_prefill
            or ctx.is_draft
            or md is None
            or md.slot_mapping is None
        ):
            return
        slots = md.slot_mapping[: positions.shape[0]]
        head = self._verify_head
        stream = None if head is None else head.ctx_stream
        if stream is not None:
            head.join_ctx()  # a write no head joined (none should be left)
            # below this many rows the graph runs the forked write before the LM
            # head, not beside it (measured), so the fork only adds its join
            if positions.shape[0] < envs.ATOM_M3_DRAFT_CTX_OVERLAP_MIN_ROWS:
                stream = None
        held = self._draft_context_writer(positions, slots, stream)
        if held is not None:
            head.ctx_pending, head.ctx_held = True, held

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        return self.lm_head(hidden_states)

    def compute_head(self, hidden_states: torch.Tensor):
        """A DSpark verify step's output ids and verdict from the in-graph
        verify head, or None (the runner computes the logits)."""
        if self._verify_head is None:
            return None
        return self._verify_head(hidden_states)

    def make_empty_intermediate_tensors(
        self, batch_size: int, dtype: torch.dtype, device: torch.device
    ) -> IntermediateTensors:
        return IntermediateTensors(
            {
                "hidden_states": torch.zeros(
                    (batch_size, self.config.hidden_size), dtype=dtype, device=device
                ),
                "residual": torch.zeros(
                    (batch_size, self.config.hidden_size), dtype=dtype, device=device
                ),
            }
        )

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        return self.model.get_expert_mapping()


class MiniMaxM3SparseForConditionalGenerationTextOnly(nn.Module):
    """Native ATOM text-only view of a MiniMax-M3 VL checkpoint."""

    # The verify head (`compute_head`) draws from the temperature alone, and
    # every TP rank picks the same ids from the same all-gathered pairs.
    head_ids_top_p = False
    head_ids_rank_uniform = True

    packed_modules_mapping = MiniMaxM3SparseForCausalLM.packed_modules_mapping
    quant_exclude_name_mapping: ClassVar[dict[str, str]] = {
        "language_model.model.": "model.",
        "language_model.lm_head": "lm_head",
    }
    weights_mapping: ClassVar[dict[str, str]] = {
        "model.language_model.": "language_model.",
    }
    skip_weight_prefixes: ClassVar[list[str]] = [
        "vision_tower.",
        "multi_modal_projector.",
        "patch_merge_mlp.",
    ]

    def __init__(self, atom_config: Config, prefix: str = "") -> None:
        super().__init__()
        self.config = atom_config.hf_config
        self.language_model = MiniMaxM3SparseForCausalLM(
            atom_config=atom_config,
            prefix=prefix,
        )
        self.make_empty_intermediate_tensors = (
            self.language_model.make_empty_intermediate_tensors
        )

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.language_model.get_input_embeddings(input_ids)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.language_model.embed_input_ids(input_ids)

    def set_aux_hidden_state_layers(self, layers: tuple[int, ...]) -> None:
        self.language_model.set_aux_hidden_state_layers(layers)

    def bind_mono_drafter(self, drafter) -> None:
        self.language_model.bind_mono_drafter(drafter)

    def get_eagle3_aux_hidden_state_layers(self) -> tuple[int, ...]:
        return self.language_model.get_eagle3_aux_hidden_state_layers()

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor | IntermediateTensors:
        return self.language_model(
            input_ids,
            positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        return self.language_model.compute_logits(hidden_states)

    def compute_head(self, hidden_states: torch.Tensor):
        return self.language_model.compute_head(hidden_states)

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        return self.language_model.get_expert_mapping()


# The plugin entry points (vLLM/SGLang) bring their own vision stacks and keep
# the text-only view; native ATOM serving selects the VL class directly.
MiniMaxM3SparseForConditionalGeneration = (
    MiniMaxM3SparseForConditionalGenerationTextOnly
)
