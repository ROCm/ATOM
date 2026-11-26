# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""Inference-only MiniMax-M3 model support for ATOM."""

import contextlib
import logging
import os
import weakref
from typing import ClassVar

import torch
from aiter import ActivationType, QuantType, dtypes
from aiter.dist.communication_op import tensor_model_parallel_all_reduce
from aiter.dist.parallel_state import (
    get_pp_group,
    get_tensor_model_parallel_world_size,
    get_tp_group,
)
from aiter.ops.triton.gluon.pa_decode_gluon import pa_decode_gluon_supports_arch
from aiter.rotary_embedding import get_rope
from torch import nn
from transformers import PretrainedConfig

from atom.config import Config, QuantizationConfig, get_current_atom_config
from atom.distributed.indexer_cp import indexer_cp_enabled
from atom.model_ops import module_dispatch_ops as _module_dispatch_ops
from atom.model_ops.attention_mha import SparseMHAPagedAttentionImpl
from atom.model_ops.base_attention import (
    Attention,
    _flydsl_arch_supported,
    _flydsl_explicit_plan_api,
    _flydsl_output_quant_api,
)
from atom.model_ops.embed_head import ParallelLMHead, VocabParallelEmbedding
from atom.model_ops.layernorm import (
    GemmaRMSNorm,
    fused_allreduce_gemma_rms_norm,
    fused_allreduce_gemma_rms_norm_mxfp8_quant,
    fused_allreduce_gemma_rms_norm_quant,
    sp_allreduce_add_residual,
    sp_allreduce_supported,
    sp_fused_allreduce_gemma_rms_norm,
    sp_fused_allreduce_gemma_rms_norm_mxfp8_quant,
    tp1_norm_mxfp8_one_launch,
)
from atom.model_ops.linear import (
    MergedColumnParallelLinear,
    MinimaxM3QKVParallelLinearWithIndexer,
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
    mxfp4_k_pad_align,
    splitk_consumer_reduce_supported,
)
from atom.model_ops.minimax_m3.sparse_attn import (
    SPARSE_BLOCK_SIZE,
)
from atom.model_ops.minimax_m3.sparse_attn_nvfp4 import (
    reads_page_table as nvfp4_reads_page_table,
)
from atom.model_ops.moe import FusedMoE, Mxfp4MoEMethod, moe_after_router_hook

try:  # aiter with the grouped MoE output epilogue (routed scale + shared add)
    from aiter.ops.flydsl.grouped_moe_gfx1250 import (
        moe_output_epilogue as _moe_output_epilogue,
    )
except ImportError:  # pragma: no cover - older aiter: the flag is a no-op
    _moe_output_epilogue = None
try:  # aiter with the routed MoE tail deferred into the next all-reduce
    from aiter.ops.flydsl.moe_deferred_tail import moe_defer_tail as _moe_defer_tail
except ImportError:  # pragma: no cover - older aiter: the flag is a no-op
    _moe_defer_tail = None
try:  # aiter with the shared-expert fold into the routed MoE GEMMs
    from aiter.ops.flydsl.grouped_moe_gfx1250 import (
        moe_shared_expert_fold as _moe_shared_expert_fold,
        prepare_shared_fold_weights as _prepare_shared_fold_weights,
    )
except ImportError:  # pragma: no cover - older aiter: the flag is a no-op
    _moe_shared_expert_fold = _prepare_shared_fold_weights = None
from atom.model_ops.swiglu_oai import (
    swiglu_oai_mxfp8_arch_supported,
    swiglu_oai_mxfp8_supported,
    swiglu_oai_quant_supported,
    swiglu_oai_split,
    swiglu_oai_split_mxfp8,
    swiglu_oai_split_quant,
    swiglu_oai_split_reducing,
    swiglu_oai_split_reducing_quant,
)
from atom.model_ops.utils import atom_parameter
from atom.models.minimax_m3.index_schedule import (
    should_skip_index_topk as _should_skip_minimax_m3_index_topk,
)
from atom.models.minimax_m3.index_schedule import (
    sparse_attention_layer_ids as _sparse_attention_layer_ids,
)
from atom.models.minimax_m3 import launch_a
from atom.models.minimax_m3.mono.dispatch import MonoDecode
from atom.models.utils import (
    IntermediateTensors,
    PPMissingLayer,
    make_empty_intermediate_tensors_factory,
    make_layers,
    maybe_prefix,
)
from atom.quant_spec import effective_layer_quant
from atom.utils import envs
from atom.utils.custom_register import direct_register_custom_op
from atom.utils.decorators import support_torch_compile
from atom.utils.region_graph import token_graph


# ATOM_MINIMAX_M3_MOE_TAIL_IN_AR: the largest MoE call whose tail is deferred into the
# next all-reduce (the k5 one-shot kernel's default row cap; larger calls would only
# run the deferred launch in front of another AR path).
_TAIL_IN_AR_MAX_ROWS = 256
_logged_once: set[str] = set()
# Diagnostic: join the shared-expert stream before the routed MoE (see dual_stream_moe_forward).
_EARLY_JOIN = os.environ.get("ATOM_MINIMAX_M3_MOE_EARLY_JOIN", "0") == "1"


def _log_once(key: str, msg: str, warn: bool = False) -> None:
    if key not in _logged_once:
        _logged_once.add(key)
        log = logging.getLogger("atom")
        (log.warning if warn else log.info)(msg)


def _get_text_config(config: PretrainedConfig) -> PretrainedConfig:
    return config.text_config if hasattr(config, "text_config") else config


def _validate_sparse_flydsl_config(
    atom_config: Config, config: PretrainedConfig
) -> None:
    """Reject gfx1250 M3 sparse PA configurations that cannot be served.

    Any ``max_num_batched_tokens`` is: the sparse attention runs rows past one
    FlyDSL plan (tokens x local KV heads, e.g. PP x TP1) in chunks
    (``flydsl_row_chunks``)."""
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
    if not _flydsl_explicit_plan_api():
        raise ValueError(
            "MiniMax-M3 sparse paged attention requires aiter's FlyDSL "
            "explicit-plan API (#5809) because there is no gluon fallback"
        )


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


def _sparse_pa_mxfp8_conditions(attn: nn.Module, o_proj: nn.Module) -> dict:
    """What the sparse attention needs to write o_proj's MXFP8 input
    (ATOM_MINIMAX_M3_PA_MXFP8_OUT): NVFP4 K/V, which gfx1250 attends through
    aiter's paged decode, and an o_proj that is a native group-32 A8 GEMM, whose
    own input quant (``quantize_fp8``) the attention's epilogue replaces bit for
    bit.

    All of it is known at construction, and has to be: torch.compile traces the
    forward once, in the warmup run, before the KV cache is bound, and the
    compiled graph keeps the branch it saw. The K/V format is the one the layer
    declares (``SparseMHAPagedAttentionImpl.kv_format``, which sizes and binds
    its pool), not ``impl.kv_nvfp4``: that is only set when the cache is bound,
    so a forward that asked it traced the BF16 output and o_proj's own quant
    into every phase."""
    return {
        "ATOM_MINIMAX_M3_PA_MXFP8_OUT": bool(envs.ATOM_MINIMAX_M3_PA_MXFP8_OUT),
        "gfx1250 paged decode": nvfp4_reads_page_table(),
        "NVFP4 K/V (declared)": getattr(attn.impl, "kv_format", "") == "nvfp4",
        "native group-32 o_proj": getattr(o_proj, "native_a8_group_rows", None)
        is not None,
    }


_logged_once: set[str] = set()


def _log_conditions_once(what: str, conditions: dict, prefix: str) -> None:
    """Log each condition of a fused path once per process: the first layer's,
    and any later layer whose verdict differs from it."""
    verdict = all(conditions.values())
    key = f"{what}:{verdict}"
    if key in _logged_once:
        return
    first = not any(k.startswith(f"{what}:") for k in _logged_once)
    _logged_once.add(key)
    text = ", ".join(f"{name}={value}" for name, value in conditions.items())
    logging.getLogger("atom").log(
        logging.INFO if first else logging.WARNING,
        f"MiniMax-M3 {what} {'ON' if verdict else 'OFF'} ({prefix}): {text}",
    )


def _rope_cache_claims_splitk_partials(attn: nn.Module) -> bool:
    """Whether ``attn``'s impl claims a deferred QKV's split-K partials in its
    rope_cache (`PagedAttentionImpl.claims_splitk_partials`)."""
    return getattr(type(getattr(attn, "impl", None)), "claims_splitk_partials", False)


def _defer_splitk_reduce(linear: nn.Module, consumer_claims: bool) -> None:
    """Let ``linear``'s per-token FP8 GEMM leave its split-K reduction to the one
    kernel that reads its output, which sums the partials as it reads them (the
    result is bit-identical), when that consumer claims them. The GEMM decides per
    call whether it splits K (aiter.gemm_a8w8_bpreshuffle_splitk_deferred)."""
    if consumer_claims and splitk_consumer_reduce_supported():
        linear.splitk_consumer_max_m = envs.ATOM_SPLITK_CONSUMER_REDUCE_MAX_M


def _fused_qkv_chain_supported(attn: nn.Module) -> bool:
    """Whether ``attn`` (a MiniMaxM3SparseAttention) may route its MXFP8 QKV input
    into rope_cache's fused chain (ATOM_M3_FUSED_QKV_CHAIN): the M3 TP4 head
    shape on gfx1250, a layer that runs the indexer, no sequence parallelism and
    no plugin bridge. The per-call conditions are rope_cache's."""
    if not envs.ATOM_M3_FUSED_QKV_CHAIN or attn.skip_index_topk:
        return False
    try:
        from aiter.ops.flydsl.mxfp8_qkv_rope_nvfp4_gfx1250 import (
            mxfp8_qkv_chain_supported,
        )
    except ImportError:
        return False
    from atom.distributed.ulysses_sp import get_sp_world_size
    from atom.plugin.prepare import is_plugin_mode

    return (
        not is_plugin_mode()
        and get_sp_world_size() == 1
        and attn.idx_head_dim == attn.head_dim
        and mxfp8_qkv_chain_supported(
            1,
            attn.num_heads,
            attn.num_kv_heads,
            attn.head_dim,
            attn.rotary_emb.rotary_dim,
            attn.hidden_size,
        )
        and isinstance(getattr(attn.attn, "impl", None), SparseMHAPagedAttentionImpl)
        # sp_attention_fp8 (per-token FP8 o_proj input) has no q_scale to carry
        # the activation scales.
        and not getattr(attn.attn, "per_token_fp8_out", False)
    )


def _fast_oproj_conditions(o_proj: nn.Module) -> dict:
    """What aiter's gfx1250 decode GEMM (ATOM_M3_FAST_OPROJ) needs of o_proj: a
    bias-free E4M3 weight with native 1x32 scales, preshuffled, its scales in
    ``shuffle_mxfp8_bpreshuffle_scale``'s ``[N, K / 32]`` layout, an (N, K) in
    the kernel's table, and no all-reduce of its own. The weight's layout is
    settled by its ``process_weights_after_loading`` (after the attention's own,
    parents run first), so this is asked at the first call, not at
    construction."""
    try:
        from aiter.ops.flydsl.mxfp8_decode_gemm_gfx1250 import (
            mxfp8_decode_gemm_supported,
        )
    except ImportError:
        mxfp8_decode_gemm_supported = None
    weight = getattr(o_proj, "weight", None)
    scale = getattr(o_proj, "weight_scale", None)
    n, k = tuple(weight.shape) if weight is not None and weight.dim() == 2 else (0, 0)
    return {
        "aiter mxfp8_decode_gemm": mxfp8_decode_gemm_supported is not None,
        "gfx1250 and (N, K) served": mxfp8_decode_gemm_supported is not None
        and n > 0
        and mxfp8_decode_gemm_supported(1, n, k),
        "no bias": getattr(o_proj, "bias", None) is None,
        "native 1x32 scales": getattr(o_proj, "native_a8_group_rows", None) == 1,
        "E4M3 weight": weight is not None and weight.dtype == torch.float8_e4m3fn,
        "preshuffled weight": bool(getattr(weight, "is_shuffled", False)),
        "contiguous weight": weight is not None and weight.is_contiguous(),
        "scale [N, K/32]": scale is not None
        and tuple(scale.shape) == (n, k // 32)
        and scale.is_contiguous(),
        "no own all-reduce": not getattr(o_proj, "reduce_results", False),
    }


def _fast_oproj_resolved(o_proj: nn.Module) -> bool:
    """``_fast_oproj_conditions``, evaluated (and logged) once per o_proj."""
    fast = o_proj.__dict__.get("_fast_oproj")
    if fast is None:
        conditions = _fast_oproj_conditions(o_proj)
        fast = all(conditions.values())
        o_proj.__dict__["_fast_oproj"] = fast
        _log_conditions_once(
            "fast o_proj (ATOM_M3_FAST_OPROJ)",
            conditions,
            getattr(o_proj, "prefix", ""),
        )
    return fast


def _o_proj_mxfp8(o_proj: nn.Module, x: torch.Tensor, x_scale: torch.Tensor):
    """o_proj on the PA epilogue's MXFP8 output: aiter's decode GEMM where it serves
    the token count (ATOM_M3_FAST_OPROJ), the linear otherwise. Same bytes up to
    the GEMM's FP32 K order. Runs per call, inside ``minimax_m3_o_proj_mxfp8``."""
    if (
        _fast_oproj_resolved(o_proj)
        and x.dtype == torch.float8_e4m3fn
        and x_scale is not None
    ):
        from aiter.ops.flydsl.mxfp8_decode_gemm_gfx1250 import (
            mxfp8_decode_gemm,
            mxfp8_decode_gemm_supported,
        )

        k = x.shape[-1]
        x2 = x.reshape(-1, k)
        s2 = x_scale.reshape(x2.shape[0], -1)
        n = o_proj.weight.shape[0]
        if (
            mxfp8_decode_gemm_supported(x2.shape[0], n, k)
            and x2.stride(1) == 1
            and x2.stride(0) % 16 == 0
            and s2.shape[1] == k // 32
            and s2.is_contiguous()
        ):
            y = mxfp8_decode_gemm(
                x2, s2.view(torch.float8_e8m0fnu), o_proj.weight, o_proj.weight_scale
            )
            return y.view(*x.shape[:-1], n)
    return o_proj(x, x_scale=x_scale)


def minimax_m3_o_proj_mxfp8(
    x: torch.Tensor, x_scale: torch.Tensor, layer_name: str
) -> torch.Tensor:
    """The o_proj registered as ``layer_name`` on the sparse attention's MXFP8
    output (``_o_proj_mxfp8``). A custom op, opaque to torch.compile: the decode
    GEMM is a FlyDSL launch Dynamo cannot trace, its weight conditions hold only
    after the weights are processed, and which kernel serves a call depends on
    its token count, which the compiled graph keeps symbolic. Inside the op both
    are decided per call (a CUDA graph keeps the capture's choice)."""
    o_proj = get_current_atom_config().compilation_config.static_forward_context[
        layer_name
    ]
    return _o_proj_mxfp8(o_proj, x, x_scale)


def _minimax_m3_o_proj_mxfp8_fake(
    x: torch.Tensor, x_scale: torch.Tensor, layer_name: str
) -> torch.Tensor:
    o_proj = get_current_atom_config().compilation_config.static_forward_context[
        layer_name
    ]
    return x.new_empty((*x.shape[:-1], o_proj.weight.shape[0]), dtype=torch.bfloat16)


direct_register_custom_op(
    op_name="minimax_m3_o_proj_mxfp8",
    op_func=minimax_m3_o_proj_mxfp8,
    mutates_args=(),
    fake_impl=_minimax_m3_o_proj_mxfp8_fake,
    # The decode GEMM reads row-major E4M3 rows and contiguous E8M0 scales.
    tags=(torch.Tag.needs_fixed_stride_order,),
)


def _shared_mlp(
    shared_experts: nn.Module,
    hidden_states: torch.Tensor,
    shared_in: tuple[torch.Tensor, torch.Tensor] | None,
) -> torch.Tensor:
    """The shared MLP of `hidden_states`, from its MXFP8 rows when given."""
    if shared_in is None:
        return shared_experts(hidden_states)
    return shared_experts(shared_in[0], x_scale=shared_in[1])


def _linear_consumes_mxfp8(linear: nn.Module) -> bool:
    """Whether a linear consumes native E4M3 plus row-major group-32 E8M0."""
    quant_type = getattr(linear, "quant_type", None)
    quant_type_value = getattr(quant_type, "value", quant_type)
    params_dtype = getattr(linear, "params_dtype", None)
    return (
        quant_type_value == QuantType.per_1x32.value
        and params_dtype in (torch.float8_e4m3fn, torch.float8_e4m3fnuz)
        and getattr(linear, "native_a8_group_rows", None) is not None
        and not getattr(linear, "runs_bf16", False)
    )


def _linear_consumes_native_mxfp8(linear: nn.Module) -> bool:
    """Return whether ``linear`` accepts row-major native MXFP8 activations."""
    quant_type = getattr(linear, "quant_type", None)
    quant_type_value = getattr(quant_type, "value", quant_type)
    return (
        quant_type_value == QuantType.per_1x32.value
        and getattr(linear, "params_dtype", None) == dtypes.fp8
        # The saved downstream validation covers MiniMax-M3's native 1x32
        # shared down projection. Do not broaden this to 32x32 implicitly.
        and getattr(linear, "native_a8_group_rows", None) == 1
        and not getattr(linear, "runs_bf16", False)
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


class MiniMaxM3MLP(nn.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        intermediate_size: int,
        quant_config: QuantizationConfig | None = None,
        reduce_results: bool = True,
        prefix: str = "",
        hidden_act: str | None = None,
        fuse_swiglu_mxfp8: bool = False,
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
        # gate_up's only reader is the activation (its op takes a set limit).
        _defer_splitk_reduce(self.gate_up_proj, self.swiglu_limit is not None)
        # The activation kernel can also quantize per token for a per-token FP8
        # down_proj (bitwise its own quant). Whether down_proj takes per-token FP8
        # is only known after online quantization re-quantizes it at load time,
        # so forward checks that part.
        self.swiglu_quant_supported = swiglu_oai_quant_supported(
            intermediate_size // get_tensor_model_parallel_world_size(), dtypes.fp8
        )
        self.fuse_swiglu_mxfp8 = fuse_swiglu_mxfp8 and swiglu_oai_mxfp8_arch_supported()

    def forward(
        self, x: torch.Tensor, x_scale: torch.Tensor | None = None
    ) -> torch.Tensor:
        gate_up = self.gate_up_proj(x, x_scale=x_scale)
        reduces_partials = self.gate_up_proj.splitk_consumer_max_m > 0 and (
            _linear_consumes_per_token_fp8(self.gate_up_proj)
        )
        if (
            self.fuse_swiglu_mxfp8
            and not reduces_partials
            and _linear_consumes_native_mxfp8(self.down_proj)
            and swiglu_oai_mxfp8_supported(gate_up)
        ):
            activated, activated_scale = swiglu_oai_split_mxfp8(
                gate_up,
                alpha=self.swiglu_alpha,
                beta=self.swiglu_beta,
                limit=self.swiglu_limit,
            )
            return self.down_proj(activated, x_scale=activated_scale)
        if self.swiglu_quant_supported and _linear_consumes_per_token_fp8(
            self.down_proj
        ):
            swiglu_quant = (
                swiglu_oai_split_reducing_quant
                if reduces_partials
                else swiglu_oai_split_quant
            )
            act, act_scale = swiglu_quant(
                gate_up, self.swiglu_alpha, self.swiglu_beta, self.swiglu_limit
            )
            return self.down_proj(act, x_scale=act_scale)
        if reduces_partials:
            x = swiglu_oai_split_reducing(
                gate_up, self.swiglu_alpha, self.swiglu_beta, self.swiglu_limit
            )
        else:
            x = swiglu_oai_split(
                gate_up,
                alpha=self.swiglu_alpha,
                beta=self.swiglu_beta,
                limit=self.swiglu_limit,
            )
        return self.down_proj(x)


def _unshuffled(shuffle, shuffled: torch.Tensor) -> torch.Tensor:
    """Undo ``shuffle``, a pure byte permutation of a 1-byte tensor: shuffle the
    byte planes of an index tensor of the same shape and dtype to get its map."""
    n = shuffled.numel()
    idx = torch.arange(n, device=shuffled.device, dtype=torch.int64).view(
        shuffled.shape
    )
    src = torch.zeros(n, device=shuffled.device, dtype=torch.int64)
    for b in range(4):
        plane = ((idx >> (8 * b)) & 0xFF).to(torch.uint8).view(shuffled.dtype)
        src |= shuffle(plane).view(torch.uint8).reshape(-1).to(torch.int64) << (8 * b)
    raw = torch.empty_like(shuffled)
    raw.view(torch.uint8).view(-1)[src] = shuffled.view(torch.uint8).reshape(-1)
    return raw


def _shared_fold_weights_of(mlp: "MiniMaxM3MLP") -> dict | None:
    """The shared expert's weights in aiter's fold layouts, or None when it is
    not a pair of native MXFP8 1x32 linears (or not yet loaded)."""
    from aiter.ops.shuffle import shuffle_mxfp8_bpreshuffle_scale, shuffle_weight

    parts = []
    for lin in (mlp.gate_up_proj, mlp.down_proj):
        w, s = lin.weight.data, getattr(lin, "weight_scale", None)
        if (
            getattr(lin, "native_a8_group_rows", None) != 1
            or s is None
            or w.dtype != torch.float8_e4m3fn
            or w.dim() != 2
            or getattr(lin, "bias", None) is not None
        ):
            return None
        s = s.data
        if getattr(lin.weight, "is_shuffled", False) or getattr(w, "is_shuffled", False):
            w = _unshuffled(lambda t: shuffle_weight(t, layout=(16, 16)), w)
            s = _unshuffled(shuffle_mxfp8_bpreshuffle_scale, s)
        if s.shape != (w.shape[0], w.shape[1] // 32):
            return None
        parts += [w, s]
    w13, s13, w2, s2 = parts
    if w13.shape[0] != 2 * w2.shape[1] or w13.shape[1] != w2.shape[0]:
        return None
    return _prepare_shared_fold_weights(w13, s13, w2, s2)


class MiniMaxM3MoE(nn.Module):
    """MiniMax-M3 routed MoE for MXFP4 checkpoints."""

    def __init__(
        self,
        config: PretrainedConfig,
        layer_id: int,
        quant_config: QuantizationConfig | None = None,
        params_dtype: torch.dtype | None = None,
        prefix: str = "",
        alt_stream: torch.cuda.Stream | None = None,
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
                fuse_swiglu_mxfp8=envs.ATOM_MINIMAX_M3_SWIGLU_MXFP8,
            )
        # The shared gate_up's input is `quantize_fp8(hidden_states)` when it
        # is a native group-32 FP8/W4A8 linear; the MXFP4 experts' A1 quant can
        # write those bytes in its own pass (ATOM_MINIMAX_M3_MOE_QUANT_SHARED).
        self.shared_input_from_moe_quant = (
            envs.ATOM_MINIMAX_M3_MOE_QUANT_SHARED
            and self.shared_experts is not None
            and self.shared_experts.gate_up_proj.native_a8_group_rows is not None
            and isinstance(self.experts.quant_method, Mxfp4MoEMethod)
        )

        # Router GEMM + biased top-k in one aiter launch (bitwise the same
        # routing); FusedMoE says whether its routing and experts take it.
        self._front_fusion = bool(
            envs.ATOM_MINIMAX_M3_MOE_FRONT_FUSION
            and hasattr(self.experts, "supports_fused_router")
            and self.experts.supports_fused_router(self.gate.weight)
        )

        self.alt_stream = alt_stream
        self.prefix = prefix
        self._use_dual_stream = False
        # The shared expert folded into the routed GEMM launches at decode
        # token counts (from the dual-stream op, eager); its weights in the
        # fold's layouts are built once loaded (`process_weights_after_loading`).
        self._shared_fold = bool(
            envs.ATOM_MINIMAX_M3_SHARED_FOLD
            and _moe_shared_expert_fold is not None
            and self.shared_experts is not None
            and not self.shared_input_from_moe_quant
            and isinstance(self.experts.quant_method, Mxfp4MoEMethod)
        )
        self._shared_fold_weights = None
        # Token counts aiter did not fold (e.g. routed tiles it has no shared
        # tile for): their first call ran the shared MLP after the routed one;
        # later calls take the dual-stream path. Decided in warm-up, before
        # graph capture.
        self._shared_fold_declined: set[int] = set()
        # The routed scale and the shared add folded into the routed MoE's
        # gather-reduce. Python-side state, so only the paths that run eagerly
        # inside the dual-stream custom op take it (`_use_dual_stream`).
        self._glue_epilogue = (
            _moe_output_epilogue
            if (
                envs.ATOM_MINIMAX_M3_MOE_GLUE_FUSION
                and _moe_output_epilogue is not None
                and self.shared_experts is not None
                and not self.shared_input_from_moe_quant
                and isinstance(self.experts.quant_method, Mxfp4MoEMethod)
            )
            else None
        )
        # ATOM_MINIMAX_M3_MOE_TAIL_IN_AR: with the glue epilogue or the shared fold,
        # the gather-reduce (+ routed scale + shared add) is left to the all-reduce that reads this
        # layer's output (aiter `moe_defer_tail`); `_owner_layer` (set by the
        # decoder layer) tells whether anything else reads it (forward hooks).
        self._tail_in_ar = bool(
            envs.ATOM_MINIMAX_M3_MOE_TAIL_IN_AR
            and (self._glue_epilogue is not None or self._shared_fold)
            and _moe_defer_tail is not None
            and tp_size > 1
            and get_pp_group().world_size == 1
        )
        self._owner_layer = None
        # ATOM_MINIMAX_M3_MOE_ROUTED_FIRST: the router launches before the shared
        # MLP's kernels (dual stream + glue epilogue + fused router only).
        self._routed_first = bool(
            envs.ATOM_MINIMAX_M3_MOE_ROUTED_FIRST
            and self._glue_epilogue is not None
            and self._front_fusion
        )
        if self.shared_experts is not None and (
            self.alt_stream is not None or self._shared_fold
        ):
            atom_config = get_current_atom_config()
            if (
                envs.ATOM_DUAL_STREAM_MOE_TOKEN_THRESHOLD > 0
                and not atom_config.enable_tbo
            ):
                self._use_dual_stream = True
                atom_config.compilation_config.static_forward_context[prefix] = self
        # ATOM_M3_SP_MOE_FRONT (1 or 2): the MoE front on the rows this rank owns after the
        # post-attention reduce-scatter, see sp_front_forward. The layer takes it on its
        # sequence-parallel path only (_forward_sp_ar); whether a call is in the
        # sequence-parallel range is decided per call, from shapes.
        experts = self.experts
        self.sp_front = 0
        if (
            envs.ATOM_M3_SP_MOE_FRONT in (1, 2)
            and self.shared_experts is not None
            and not self.fuse_shared_experts
            and isinstance(experts.quant_method, Mxfp4MoEMethod)
            and experts.scoring_func == "sigmoid"
            and not experts.use_grouped_topk
            and experts.custom_routing_function is None
            and experts.num_fused_shared_experts == 0
            and experts.balance_router_logits is None
            and experts.dp_size == 1
        ):
            self.sp_front = int(envs.ATOM_M3_SP_MOE_FRONT)
            get_current_atom_config().compilation_config.static_forward_context[
                prefix
            ] = self
        self._sp_front_quant = None  # mode 2's consumers, checked once loaded

    def process_weights_after_loading(self) -> None:
        # Parent-first: the shared linears are normally still unshuffled here.
        if self._shared_fold:
            self._shared_fold_weights = _shared_fold_weights_of(self.shared_experts)

    def _shared_fold_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Routed + shared in the routed MoE's launches; the separate shared
        MLP after the routed one when aiter did not take the fold (and the
        token count is then remembered as declined)."""
        defer = getattr(self, "_tail_in_ar", False) and self._defer_tail_now(
            hidden_states
        )
        with _moe_shared_expert_fold(
            self._shared_fold_weights, self._routed_scale()
        ) as fold:
            # ATOM_MINIMAX_M3_MOE_TAIL_IN_AR: the fold's gather-reduce (+ routed
            # scale + its shared rows) may go to the next all-reduce too.
            with _moe_defer_tail() if defer else contextlib.nullcontext():
                routed_output = self._route_and_experts(hidden_states)
        if fold.consumed:
            return routed_output
        self._shared_fold_declined.add(hidden_states.shape[0])
        return self._scale_routed(routed_output) + self.shared_experts(hidden_states)

    def _routed_scale(self) -> float:
        if not self.fuse_shared_experts and self.routed_scaling_factor != 1.0:
            return float(self.routed_scaling_factor)
        return 1.0

    def _scale_routed(self, routed_output: torch.Tensor) -> torch.Tensor:
        scale = self._routed_scale()
        if scale != 1.0:
            routed_output = routed_output * scale
        return routed_output

    def _route_and_experts(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """The routed experts' TP-partial output, before the routed scale."""
        if self._front_fusion:
            return self.experts.forward_fused_router(hidden_states, self.gate.weight)
        router_logits = self.gate(hidden_states)
        return self.experts(hidden_states=hidden_states, router_logits=router_logits)

    def _defer_tail_now(self, hidden_states: torch.Tensor) -> bool:
        """Whether this call's routed tail may go to the next all-reduce: the flag,
        a decode-sized call, and nothing but the next layer's input AR (or the
        final norm's) reading this layer's output: no forward hook on this layer,
        no forward pre-hook on the next one (DSpark / Eagle3 taps), no global
        module hook."""
        if not self._tail_in_ar or hidden_states.shape[0] > _TAIL_IN_AR_MAX_ROWS:
            return False
        from torch.nn.modules import module as _nn_module

        if _nn_module._global_forward_hooks or _nn_module._global_forward_pre_hooks:
            return False
        owner = self._owner_layer() if self._owner_layer is not None else None
        if owner is None or owner._forward_hooks:
            return False
        nxt = getattr(owner, "_next_layer", None)
        nxt = nxt() if nxt is not None else None
        return nxt is None or not nxt._forward_pre_hooks

    def _routed_plus_shared(
        self,
        hidden_states: torch.Tensor,
        shared_output: torch.Tensor | None,
        before_add=None,
        launch_shared=None,
    ) -> torch.Tensor:
        """`_scale_routed(routed) + shared_output` with the scale and the add in
        the routed MoE's gather-reduce when aiter takes the epilogue; `before_add`
        (the wait for the stream producing `shared_output`) runs right before
        that launch, or before the separate add when it did not.

        `launch_shared` (ATOM_MINIMAX_M3_MOE_ROUTED_FIRST, `shared_output` None):
        launches the shared MLP and returns its output; it runs right after the
        router launch (or after the routed MoE when no router hook ran).
        ATOM_MINIMAX_M3_MOE_TAIL_IN_AR: the gather-reduce may be deferred into
        the all-reduce that reads the returned (then unwritten) tensor."""
        defer = getattr(self, "_tail_in_ar", False) and self._defer_tail_now(
            hidden_states
        )
        with self._glue_epilogue(
            shared_output, self._routed_scale(), before=before_add
        ) as epilogue:
            with contextlib.ExitStack() as stack:
                if launch_shared is not None:

                    def _after_router():
                        epilogue.addend = launch_shared()
                        _log_once(
                            "routed_first",
                            "MiniMax-M3 MoE: the shared MLP is launched after the router "
                            "(ATOM_MINIMAX_M3_MOE_ROUTED_FIRST)",
                        )

                    stack.enter_context(moe_after_router_hook(_after_router))
                scope = stack.enter_context(_moe_defer_tail()) if defer else None
                routed_output = self._route_and_experts(hidden_states)
            if scope is not None and scope.deferred is None:
                _log_once(
                    f"nodefer_{hidden_states.shape[0]}",
                    f"MiniMax-M3 MoE: tail not deferred at {hidden_states.shape[0]} rows "
                    f"(epilogue consumed {epilogue.consumed}, addend "
                    f"{None if epilogue.addend is None else tuple(epilogue.addend.shape)})",
                    warn=True,
                )
            if scope is not None and scope.deferred is not None:
                _log_once(
                    "tail_in_ar",
                    "MiniMax-M3 MoE: the gather-reduce (+ routed scale + shared add) is "
                    "deferred into the next all-reduce (ATOM_MINIMAX_M3_MOE_TAIL_IN_AR, "
                    f"first at {hidden_states.shape[0]} rows)",
                )
            if launch_shared is not None and epilogue.addend is None:
                # No router launch ran the hook: the shared MLP after the routed one.
                epilogue.addend = launch_shared()
        if epilogue.consumed:
            return routed_output
        if before_add is not None:
            before_add()
        if launch_shared is not None:
            shared_output = epilogue.addend
        return self._scale_routed(routed_output) + shared_output

    def routed_expert_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Run the routed E128/top-4 branch and keep its TP-partial output."""
        return self._scale_routed(self._route_and_experts(hidden_states))

    def takes_shared_mxfp8_input(self) -> bool:
        """Whether `forward` can take the shared expert's MXFP8 input from the
        caller (`shared_x`, `shared_x_scale`): a separate shared MLP whose gate_up
        is a native MXFP8 linear, not fed by the routed quant or folded into the
        routed launches."""
        return (
            self.shared_experts is not None
            and not self.shared_input_from_moe_quant
            and not self._shared_fold
            and _linear_consumes_mxfp8(self.shared_experts.gate_up_proj)
        )

    def single_stream_moe_forward(
        self,
        hidden_states: torch.Tensor,
        shared_in: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        orig_shape = hidden_states.shape
        hidden_states = hidden_states.view(-1, orig_shape[-1])
        if self.shared_input_from_moe_quant:
            # Serial only: the shared expert reads the routed quant's output
            # here, so the two branches are not independent. The dual-stream
            # path below keeps the shared expert's own quant.
            if self._front_fusion:
                routed_output, shared_x, shared_x_scale = (
                    self.experts.forward_fused_router_with_mxfp8_copy(
                        hidden_states, self.gate.weight
                    )
                )
            else:
                router_logits = self.gate(hidden_states)
                routed_output, shared_x, shared_x_scale = (
                    self.experts.forward_with_mxfp8_copy(hidden_states, router_logits)
                )
            routed_output = self._scale_routed(routed_output) + self.shared_experts(
                shared_x, x_scale=shared_x_scale
            )
            return routed_output.view(orig_shape)
        if getattr(self, "_glue_epilogue", None) is not None and self._use_dual_stream:
            # Reached only from the dual-stream op (eager), see `_glue_epilogue`.
            shared_output = _shared_mlp(self.shared_experts, hidden_states, shared_in)
            return self._routed_plus_shared(hidden_states, shared_output).view(
                orig_shape
            )
        routed_output = self.routed_expert_forward(hidden_states)
        if self.shared_experts is not None:
            routed_output = routed_output + _shared_mlp(
                self.shared_experts, hidden_states, shared_in
            )
        return routed_output.view(orig_shape)

    def dual_stream_moe_forward(
        self,
        hidden_states: torch.Tensor,
        shared_in: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """Overlap the independent shared MLP with the routed MoE branch (or,
        with the shared fold, run it inside the routed MoE's launches)."""
        orig_shape = hidden_states.shape
        hidden_states = hidden_states.view(-1, orig_shape[-1])
        if (
            getattr(self, "_shared_fold_weights", None) is not None
            and hidden_states.shape[0] not in self._shared_fold_declined
        ):
            return self._shared_fold_forward(hidden_states).view(orig_shape)
        if self.alt_stream is None:
            return self.single_stream_moe_forward(hidden_states, shared_in).view(
                orig_shape
            )
        current_stream = torch.cuda.current_stream()
        alt_stream = self.alt_stream

        if (
            getattr(self, "_routed_first", False)
            and getattr(self, "_glue_epilogue", None) is not None
        ):
            # The shared MLP forks here as before but is launched after the router,
            # so the router is the fork's first dependent launch (a captured graph
            # keeps it on this stream).
            fork = torch.cuda.Event()
            fork.record(current_stream)
            launched = []

            def launch_shared() -> torch.Tensor:
                alt_stream.wait_event(fork)
                with torch.cuda.stream(alt_stream):
                    launched.append(
                        _shared_mlp(self.shared_experts, hidden_states, shared_in)
                    )
                return launched[0]

            output = self._routed_plus_shared(
                hidden_states,
                None,
                before_add=lambda: current_stream.wait_stream(alt_stream),
                launch_shared=launch_shared,
            )
            launched[0].record_stream(current_stream)
            return output.view(orig_shape)

        alt_stream.wait_stream(current_stream)
        with torch.cuda.stream(alt_stream):
            shared_output = _shared_mlp(self.shared_experts, hidden_states, shared_in)

        if getattr(self, "_glue_epilogue", None) is not None:
            # The join moves to right before the gather-reduce that adds it.
            # ATOM_MINIMAX_M3_MOE_EARLY_JOIN=1 (diagnostic): join right here instead,
            # before any routed launch (the routed path then waits for the shared MLP).
            if _EARLY_JOIN:
                current_stream.wait_stream(alt_stream)
            output = self._routed_plus_shared(
                hidden_states,
                shared_output,
                before_add=None
                if _EARLY_JOIN
                else (lambda: current_stream.wait_stream(alt_stream)),
            )
            shared_output.record_stream(current_stream)
            return output.view(orig_shape)

        routed_output = self.routed_expert_forward(hidden_states)
        current_stream.wait_stream(alt_stream)
        shared_output.record_stream(current_stream)
        return (routed_output + shared_output).view(orig_shape)

    def forward(
        self,
        hidden_states: torch.Tensor,
        shared_x: torch.Tensor | None = None,
        shared_x_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """`shared_x`, `shared_x_scale`: the shared expert's MXFP8 input, the
        quant of `hidden_states` its gate_up would make (see
        `takes_shared_mxfp8_input`)."""
        if self._use_dual_stream:
            if shared_x is not None:
                return torch.ops.aiter.minimax_m3_moe_shared_in(
                    hidden_states, shared_x, shared_x_scale, self.prefix
                )
            return torch.ops.aiter.maybe_dual_stream_forward(hidden_states, self.prefix)
        shared_in = None if shared_x is None else (shared_x, shared_x_scale)
        return self.single_stream_moe_forward(hidden_states, shared_in)

    # ---- ATOM_M3_SP_MOE_FRONT: the MoE front on the sequence-parallel owner rows ----
    def _sp_front_quant_ok(self) -> bool:
        """Mode 2's consumers take the AR's quantized rows: the shared gate_up a native
        MXFP8 (row-major E8M0) linear, the routed experts aiter's grouped fused_moe."""
        if self._sp_front_quant is None:
            qm = self.experts.quant_method
            self._sp_front_quant = bool(
                _linear_consumes_mxfp8(self.shared_experts.gate_up_proj)
                and getattr(qm, "fused_experts", None) is None
                and not getattr(qm, "use_triton", False)
                and not getattr(qm, "use_triton_ep", False)
            )
        return self._sp_front_quant

    def _sp_route(
        self, own_rows: list[torch.Tensor], rows: int, hidden: int, ca
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """`select_experts(x, gate(x))` of every row, (topk_weights fp32, topk_ids int32)
        [rows, top_k] on every rank, from `own_rows[i]`: this rank's normalized rows of ar2s
        launch i (`ar2s_sp_own_rows`). Each rank routes its own rows -- the router GEMM with
        the kernel the tuned table picks for the whole `rows` (aiter
        `gemm_a16w16_rows_of`), so the logits are the replicated GEMM's -- and one all-gather
        a launch hands every rank the routes."""
        from aiter import topk_gating
        from aiter.tuned_gemm import gemm_a16w16_rows_of
        from atom.utils.forward_context import pass_scheduled_tokens_gpu

        experts = self.experts
        top_k = experts.top_k
        world = ca.world_size
        launches = ca.ar2s_launch_rows(rows, hidden)
        # rows past the pass's real count route to -1 (as select_experts' topk_gating does)
        num_valid = pass_scheduled_tokens_gpu(rows)
        fused = envs.ATOM_M3_SP_MOE_ROUTER == "fused"
        parts = []
        for (s, e), x in zip(launches, own_rows):
            chunk = -(-(e - s) // world)
            first = s + ca.rank * chunk
            m = x.shape[0]
            # ids then weights a row: one gather moves both
            send = torch.empty((chunk, 2 * top_k), dtype=torch.int32, device=x.device)
            if m:
                nv = (
                    None
                    if num_valid is None
                    else (num_valid.reshape(-1)[:1] - first).clamp_(0, m).to(torch.int32)
                )
                if fused:
                    from aiter.ops.flydsl.moe_front_gfx1250 import (
                        flydsl_moe_router_topk as _flydsl_moe_router_topk,
                    )

                    w, ids = _flydsl_moe_router_topk(
                        x,
                        self.gate.weight,
                        experts.e_score_correction_bias,
                        top_k,
                        1,
                        need_renorm=experts.renormalize,
                        routed_scaling_factor=1.0,
                        num_valid_rows=nv,
                    )
                    send[:m, :top_k].copy_(ids)
                    send[:m, top_k:].copy_(w.view(torch.int32))
                else:
                    logits = gemm_a16w16_rows_of(
                        x, self.gate.weight, rows, otype=dtypes.bf16
                    )
                    topk_gating(
                        send[:m, top_k:].view(torch.float32),
                        send[:m, :top_k],
                        logits,
                        experts.e_score_correction_bias,
                        experts.renormalize,
                        1.0,
                        score_func=experts.scoring_func,
                        num_valid_rows=nv,
                    )
            parts.append(send)
        gathered = ca.ar2s_sp_gather_rows(parts, rows, hidden)
        both = gathered.view(rows, 2, top_k).transpose(0, 1).contiguous()
        return both[1].view(torch.float32), both[0]

    def sp_front_forward(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        norm_weight: torch.Tensor,
        eps: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """The layer's post-attention AR + Gemma RMSNorm and this MoE, on a sequence-parallel
        residual: (MoE output, residual_out) from the TP-partial attention output, the bytes
        of `sp_fused_allreduce_gemma_rms_norm` followed by `forward`.

        In the sequence-parallel range each rank routes only the rows it owns after the
        reduce-scatter (`_sp_route`), and with ATOM_M3_SP_MOE_FRONT=2 the AR's owners also
        quantize their rows to the shared expert's MXFP8 and the routed experts' MXFP4 A1
        (aiter `sp_fused_allreduce_rmsnorm_moe_front`), which the all-gather carries in place
        of the bf16 rows; the experts then skip both quants. Below it, the replicated path."""
        comm = get_tp_group().device_communicator
        rows, hidden = hidden_states.shape
        mode = self.sp_front
        if mode == 2 and not self._sp_front_quant_ok():
            mode = 1
        if not comm.sp_moe_front_eligible(hidden_states, norm_weight):
            normed, residual = comm.sp_fused_allreduce_rmsnorm(
                hidden_states, residual, norm_weight, eps, True
            )
            return self.forward(normed), residual
        ca = comm.ca_comm
        if mode == 2:
            x_own, residual, q8, s8, q4, s4 = comm.sp_fused_allreduce_rmsnorm_moe_front(
                hidden_states, residual, norm_weight, eps, True
            )
            own_rows, off = [], 0
            for s, e in ca.ar2s_launch_rows(rows, hidden):
                chunk = -(-(e - s) // ca.world_size)
                first = s + ca.rank * chunk
                m = max(0, min(e, first + chunk) - first)
                own_rows.append(x_own[off : off + m])
                off += m
        else:
            normed, residual = comm.sp_fused_allreduce_rmsnorm(
                hidden_states, residual, norm_weight, eps, True
            )
            own_rows = []
            for s, e in ca.ar2s_launch_rows(rows, hidden):
                chunk = -(-(e - s) // ca.world_size)
                first = s + ca.rank * chunk
                own_rows.append(normed[first : max(first, min(e, first + chunk))])
        routing = self._sp_route(own_rows, rows, hidden, ca)
        if mode == 2:
            shared_output = self.shared_experts(q8, x_scale=s8)
            x, x_scale = q4.view(dtypes.fp4x2), s4
        else:
            shared_output = self.shared_experts(normed)
            x, x_scale = normed, None
        if self._glue_epilogue is not None:
            with self._glue_epilogue(shared_output, self._routed_scale()) as epilogue:
                routed_output = self.experts.forward_impl(
                    x, None, routing=routing, x_scale=x_scale
                )
            if epilogue.consumed:
                return routed_output, residual
        else:
            routed_output = self.experts.forward_impl(
                x, None, routing=routing, x_scale=x_scale
            )
        return self._scale_routed(routed_output) + shared_output, residual


def minimax_m3_sp_moe_front(
    hidden_states: torch.Tensor,
    residual: torch.Tensor,
    norm_weight: torch.Tensor,
    eps: float,
    layer_name: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """`MiniMaxM3MoE.sp_front_forward` of the MoE `layer_name`, eager (ATOM_M3_SP_MOE_FRONT)."""
    moe = get_current_atom_config().compilation_config.static_forward_context[layer_name]
    return moe.sp_front_forward(hidden_states, residual, norm_weight, eps)


def _minimax_m3_sp_moe_front_fake(
    hidden_states: torch.Tensor,
    residual: torch.Tensor,
    norm_weight: torch.Tensor,
    eps: float,
    layer_name: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.empty_like(hidden_states), torch.empty_like(residual)


direct_register_custom_op(
    op_name="minimax_m3_sp_moe_front",
    op_func=minimax_m3_sp_moe_front,
    mutates_args=(),
    fake_impl=_minimax_m3_sp_moe_front_fake,
)


def minimax_m3_moe_shared_in(
    hidden_states: torch.Tensor,
    shared_x: torch.Tensor,
    shared_x_scale: torch.Tensor,
    layer_name: str,
) -> torch.Tensor:
    """`maybe_dual_stream_forward` of the MoE `layer_name` with the shared
    expert's MXFP8 input given (`MiniMaxM3MoE.forward`'s `shared_x`)."""
    moe = get_current_atom_config().compilation_config.static_forward_context[
        layer_name
    ]
    shared_in = (shared_x, shared_x_scale)
    if _module_dispatch_ops._dual_stream_is_active(moe, hidden_states.shape[0]):
        return moe.dual_stream_moe_forward(hidden_states, shared_in)
    return moe.single_stream_moe_forward(hidden_states, shared_in)


def _minimax_m3_moe_shared_in_fake(
    hidden_states: torch.Tensor,
    shared_x: torch.Tensor,
    shared_x_scale: torch.Tensor,
    layer_name: str,
) -> torch.Tensor:
    return torch.empty_like(hidden_states)


direct_register_custom_op(
    op_name="minimax_m3_moe_shared_in",
    op_func=minimax_m3_moe_shared_in,
    mutates_args=(),
    fake_impl=_minimax_m3_moe_shared_in_fake,
    tags=(torch.Tag.needs_fixed_stride_order,),
)


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
        # qkv's only reader is attn's rope_cache (q/k/v below are views for shapes).
        _defer_splitk_reduce(
            self.qkv_proj,
            _rope_cache_claims_splitk_partials(self.attn),
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
        # A layer reusing cached top-k never reads the index columns, so its
        # rope_cache would reduce the partials with the standalone kernel anyway.
        _defer_splitk_reduce(
            self.qkv_proj,
            not self.skip_index_topk and _rope_cache_claims_splitk_partials(self.attn),
        )
        # Shape and kernel conditions are fixed at construction. Whether o_proj
        # takes per-token FP8 is only known after online quantization re-quantizes
        # it at load time, so forward checks that part.
        self.sparse_pa_output_quant_supported = (
            envs.ATOM_PA_FLYDSL
            and self.num_kv_heads == 1
            and self.num_heads == 16
            and self.head_dim == 128
            and callable(getattr(self.attn, "forward_quantized", None))
            and _flydsl_output_quant_api()
        )
        # Opt-in: the NVFP4 sparse attention writes o_proj's MXFP8 input. Only
        # gfx1250's NVFP4 attention (aiter's paged decode) has that output.
        # Decided here, never in forward (see _sparse_pa_mxfp8_conditions).
        mxfp8_conditions = _sparse_pa_mxfp8_conditions(self.attn, self.o_proj)
        self.sparse_pa_mxfp8_out = all(mxfp8_conditions.values())
        if envs.ATOM_MINIMAX_M3_PA_MXFP8_OUT:
            _log_conditions_once(
                "sparse PA -> o_proj MXFP8 input (ATOM_MINIMAX_M3_PA_MXFP8_OUT)",
                mxfp8_conditions,
                prefix,
            )
        # Opt-in: the QKV GEMM moves into rope_cache, which fuses it with the
        # norm / RoPE / NVFP4 inserts where aiter's chain kernel takes the call
        # and runs the GEMM itself otherwise (SparseMHAPagedAttentionImpl).
        self.fused_qkv_chain = _fused_qkv_chain_supported(self)
        # Opt-in: o_proj's MXFP8 GEMM through aiter's gfx1250 decode GEMM, by
        # the custom op that looks the linear up by name; whether its weight
        # qualifies is asked at the op's first call (_fast_oproj_resolved).
        self.fast_oproj = bool(envs.ATOM_M3_FAST_OPROJ) and self.sparse_pa_mxfp8_out
        self.o_proj_name = f"{prefix}.o_proj"
        if self.fast_oproj:
            compilation_config = get_current_atom_config().compilation_config
            if self.o_proj_name in compilation_config.static_forward_context:
                raise ValueError(f"Duplicate layer name: {self.o_proj_name}")
            compilation_config.static_forward_context[self.o_proj_name] = self.o_proj
        if self.fused_qkv_chain:
            # A tuple: the impl reads the linear, it does not own it.
            self.attn.impl.fused_qkv_linear = (self.qkv_proj,)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        hidden_states_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.sparse_pa_mxfp8_out:
            attn_output, attn_scale = self.attend_mxfp8(
                positions, hidden_states, hidden_states_scale
            )
            if self.fast_oproj:
                return torch.ops.aiter.minimax_m3_o_proj_mxfp8(
                    attn_output, attn_scale, self.o_proj_name
                )
            return self.o_proj(attn_output, x_scale=attn_scale)
        q, k, v, qkv, x_scale = self._qkv(hidden_states, hidden_states_scale)
        if self.sparse_pa_output_quant_supported and _linear_consumes_per_token_fp8(
            self.o_proj
        ):
            attn_output, attn_scale = self.attn.forward_quantized(
                q, k, v, positions, q_scale=x_scale, qkv=qkv
            )
            return self.o_proj(attn_output, x_scale=attn_scale)
        attn_output = self.attn(q, k, v, positions, q_scale=x_scale, qkv=qkv)
        if isinstance(attn_output, tuple):
            return self.o_proj(*attn_output)
        return self.o_proj(attn_output)

    def attend_mxfp8(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        hidden_states_scale: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """The attention up to o_proj's MXFP8 input (``sparse_pa_mxfp8_out``):
        (E4M3 rows, row-major E8M0 scales). ``forward`` runs o_proj on them; a
        decoder layer with a launch-A seam hands them to its post-attention op."""
        q, k, v, qkv, x_scale = self._qkv(hidden_states, hidden_states_scale)
        return self.attn.forward_quantized(
            q, k, v, positions, q_scale=x_scale, qkv=qkv, mxfp8=True
        )

    def _qkv(self, hidden_states, hidden_states_scale):
        """(q, k, v, qkv, q_scale) for the attention op: the packed QKV GEMM's
        output, or the MXFP8 activation itself where rope_cache runs the GEMM."""
        x_scale = None
        if (
            self.fused_qkv_chain
            and hidden_states_scale is not None
            and hidden_states.dtype == torch.float8_e4m3fn
            and _linear_consumes_mxfp8(self.qkv_proj)
        ):
            # The MXFP8 activation rides to rope_cache as `qkv` with its E8M0
            # scales as `q_scale`; q/k/v only size the attention's outputs.
            qkv, x_scale = hidden_states, hidden_states_scale
            rows = hidden_states.new_empty(
                (hidden_states.shape[0], self.q_size + 2 * self.kv_size),
                dtype=torch.bfloat16,
            )
            q, k, v = rows.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        else:
            # Keep index Q/K packed with main QKV. Layers that reuse cached top-k
            # skip the indexer norm/rope/top-k path, but still compute the packed
            # GEMM.
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
        return q, k, v, qkv, x_scale


class MiniMaxM3DecoderLayer(nn.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        prefix: str,
        cache_config: str = "bf16",
        index_cache_config: str = "auto",
        quant_config: QuantizationConfig | None = None,
        params_dtype: torch.dtype | None = None,
        layer_num: int = 0,
        alt_stream: torch.cuda.Stream | None = None,
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
                alt_stream=alt_stream,
            )
            # ATOM_MINIMAX_M3_MOE_TAIL_IN_AR asks this layer whether anything
            # (forward hooks) reads its output besides the next all-reduce.
            if hasattr(self.block_sparse_moe, "_owner_layer"):
                self.block_sparse_moe._owner_layer = weakref.ref(self)
        else:
            self.mlp = MiniMaxM3MLP(
                config=config,
                intermediate_size=config.dense_intermediate_size,
                quant_config=quant_config,
                reduce_results=False,
                prefix=f"{prefix}.mlp",
                fuse_swiglu_mxfp8=envs.ATOM_MINIMAX_M3_SWIGLU_MXFP8,
            )

        self.input_layernorm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = GemmaRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        # Launch A (C4 decode, gfx1250 TP4): the name of this layer's post-attention
        # op (`launch_a.minimax_m3_attn_tail`), or None. Fixed at construction, so
        # the traced forward keeps one branch; the op picks the fused launch or the
        # production calls per step.
        self.launch_a_seam = launch_a.bind(self, prefix)

    def _attn_tail(self, positions, hidden_states, hidden_states_scale, residual, mode):
        """The launch-A seam: the attention up to o_proj's MXFP8 input, then o_proj
        + the post-attention all-reduce + residual + RMSNorm (+ the MoE in
        ``launch_a.TAIL_SP_MOE_FRONT``) as one op opaque to torch.compile."""
        attn_output, attn_scale = self.self_attn.attend_mxfp8(
            positions, hidden_states, hidden_states_scale
        )
        return torch.ops.aiter.minimax_m3_attn_tail(
            attn_output,
            attn_scale,
            residual,
            self.post_attention_layernorm.weight,
            self.post_attention_layernorm.variance_epsilon,
            self.launch_a_seam,
            mode,
        )

    def sp_ar_capable(self) -> bool:
        """Whether both of this layer's all-reduce + norms take the sequence-parallel residual ops
        (ATOM_M3_PREFILL_SP_AR): bf16 or native-MXFP8 outputs, not the per-token FP8 one."""
        return not (
            _linear_consumes_per_token_fp8(self.self_attn.qkv_proj)
            or (
                not self.is_moe_layer
                and _linear_consumes_per_token_fp8(self.mlp.gate_up_proj)
            )
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
        capture_aux: bool = False,
        sp_ar: bool = False,
    ) -> (
        tuple[torch.Tensor, torch.Tensor]
        | tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ):
        """``sp_ar``: the residual stream is sequence-parallel at prefill sizes (see
        ATOM_M3_PREFILL_SP_AR); the model sets it for every layer of a forward or for none."""
        if sp_ar:
            return self._forward_sp_ar(positions, hidden_states, residual)
        hidden_states_scale = None
        fuse_input_ar_rmsnorm_quant = _linear_consumes_per_token_fp8(
            self.self_attn.qkv_proj
        )
        fuse_post_attention_ar_rmsnorm_quant = (
            not self.is_moe_layer
            and _linear_consumes_per_token_fp8(self.mlp.gate_up_proj)
        )
        enable_mxfp8_fused_ar = envs.ATOM_MINIMAX_M3_GLUON_AR_MXFP8
        fuse_input_ar_rmsnorm_mxfp8 = enable_mxfp8_fused_ar and _linear_consumes_mxfp8(
            self.self_attn.qkv_proj
        )
        fuse_post_attention_ar_rmsnorm_mxfp8 = (
            enable_mxfp8_fused_ar
            and not self.is_moe_layer
            and _linear_consumes_mxfp8(self.mlp.gate_up_proj)
        )
        # TP 1: the MoE's post-attention norm also writes the shared expert's
        # MXFP8 input (its gate_up's own quant of the same row, bit for bit).
        post_attention_norm_shared_mxfp8 = (
            enable_mxfp8_fused_ar
            and tp1_norm_mxfp8_one_launch(
                hidden_states.numel() // hidden_states.shape[-1]
            )
            and self.is_moe_layer
            and get_tensor_model_parallel_world_size() == 1
            and self.block_sparse_moe.takes_shared_mxfp8_input()
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
        elif fuse_input_ar_rmsnorm_mxfp8:
            hidden_states, hidden_states_scale, residual = (
                fused_allreduce_gemma_rms_norm_mxfp8_quant(
                    hidden_states,
                    residual,
                    self.input_layernorm,
                )
            )
        else:
            hidden_states, residual = fused_allreduce_gemma_rms_norm(
                hidden_states, residual, self.input_layernorm
            )

        # Eagle3 aux hidden state = the all-reduced residual stream entering this
        # layer (post input-norm). Captured here, not as `hidden_states + residual`
        # in the model loop, because M3's fused all-reduce RMSNorm leaves that sum
        # TP-partial / NaN-prone under CUDAGraph.
        aux_hidden_state = residual.clone() if capture_aux else None

        if self.launch_a_seam is not None:
            # a MoE layer: the plain AR + norm below, here inside the seam's op
            hidden_states, residual = self._attn_tail(
                positions, hidden_states, hidden_states_scale, residual,
                launch_a.TAIL_AR_NORM,
            )  # fmt: skip
            hidden_states = self.block_sparse_moe(hidden_states)
            if aux_hidden_state is not None:
                return hidden_states, residual, aux_hidden_state
            return hidden_states, residual
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
        elif fuse_post_attention_ar_rmsnorm_mxfp8:
            hidden_states, hidden_states_scale, residual = (
                fused_allreduce_gemma_rms_norm_mxfp8_quant(
                    hidden_states,
                    residual,
                    self.post_attention_layernorm,
                )
            )
            hidden_states = ffn(hidden_states, x_scale=hidden_states_scale)
        elif post_attention_norm_shared_mxfp8:
            shared_x, shared_x_scale, hidden_states, residual = (
                fused_allreduce_gemma_rms_norm_mxfp8_quant(
                    hidden_states,
                    residual,
                    self.post_attention_layernorm,
                    emit_bf16=True,
                )
            )
            hidden_states = ffn(
                hidden_states, shared_x=shared_x, shared_x_scale=shared_x_scale
            )
        else:
            hidden_states, residual = fused_allreduce_gemma_rms_norm(
                hidden_states, residual, self.post_attention_layernorm
            )
            hidden_states = ffn(hidden_states)
        if aux_hidden_state is not None:
            return hidden_states, residual, aux_hidden_state
        return hidden_states, residual

    def _forward_sp_ar(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """forward with the sequence-parallel residual ops (bit-identical normed outputs)."""
        hidden_states_scale = None
        mxfp8 = envs.ATOM_MINIMAX_M3_GLUON_AR_MXFP8
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        elif mxfp8 and _linear_consumes_mxfp8(self.self_attn.qkv_proj):
            hidden_states, hidden_states_scale, residual = (
                sp_fused_allreduce_gemma_rms_norm_mxfp8_quant(
                    hidden_states, residual, self.input_layernorm
                )
            )
        else:
            hidden_states, residual = sp_fused_allreduce_gemma_rms_norm(
                hidden_states, residual, self.input_layernorm
            )
        if self.launch_a_seam is not None:
            if self.block_sparse_moe.sp_front:
                return self._attn_tail(
                    positions, hidden_states, hidden_states_scale, residual,
                    launch_a.TAIL_SP_MOE_FRONT,
                )  # fmt: skip
            hidden_states, residual = self._attn_tail(
                positions, hidden_states, hidden_states_scale, residual,
                launch_a.TAIL_SP_AR_NORM,
            )  # fmt: skip
            return self.block_sparse_moe(hidden_states), residual
        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            hidden_states_scale=hidden_states_scale,
        )
        if (
            mxfp8
            and not self.is_moe_layer
            and _linear_consumes_mxfp8(self.mlp.gate_up_proj)
        ):
            hidden_states, hidden_states_scale, residual = (
                sp_fused_allreduce_gemma_rms_norm_mxfp8_quant(
                    hidden_states, residual, self.post_attention_layernorm
                )
            )
            return self.mlp(hidden_states, x_scale=hidden_states_scale), residual
        if self.is_moe_layer and self.block_sparse_moe.sp_front:
            # ATOM_M3_SP_MOE_FRONT: the AR + norm and the MoE in one op, whose front runs on
            # the rows this rank owns (MiniMaxM3MoE.sp_front_forward)
            return torch.ops.aiter.minimax_m3_sp_moe_front(
                hidden_states.contiguous(),
                residual,
                self.post_attention_layernorm.weight,
                self.post_attention_layernorm.variance_epsilon,
                self.block_sparse_moe.prefix,
            )
        hidden_states, residual = sp_fused_allreduce_gemma_rms_norm(
            hidden_states, residual, self.post_attention_layernorm
        )
        ffn = self.block_sparse_moe if self.is_moe_layer else self.mlp
        return ffn(hidden_states), residual

    @staticmethod
    def aux_hidden_state(output: tuple) -> torch.Tensor:
        """This layer's output as a DSpark draft reads it: the residual stream
        after the layer, ``hidden + residual``.

        ``hidden`` is TP-partial -- o_proj and the FFN skip their all-reduce,
        which the next layer's fused all-reduce RMSNorm performs -- so it is
        reduced here first. That costs one all-reduce per tapped layer per
        target forward, only when a DSpark drafter taps this layer.
        """
        hidden_states, residual = output[0], output[1]
        if get_tensor_model_parallel_world_size() > 1:
            if MiniMaxM3Model.sp_ar_enabled:
                # The residual may be sequence-parallel (prefill sizes): the op gathers it.
                return sp_allreduce_add_residual(hidden_states, residual)
            hidden_states = tensor_model_parallel_all_reduce(hidden_states)
        return hidden_states + residual


@token_graph(per_phase=True)
@support_torch_compile
class MiniMaxM3Model(nn.Module):
    # ATOM_M3_PREFILL_SP_AR in effect (decided at construction; read by the DSpark taps, whose
    # extract is a static method of the layer).
    sp_ar_enabled: ClassVar[bool] = False

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

        if get_pp_group().is_first_rank:
            self.embed_tokens = VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
            )
        else:
            self.embed_tokens = PPMissingLayer()

        self.alt_stream = (
            torch.cuda.Stream()
            if torch.cuda.is_available() and getattr(config, "n_shared_experts", 0)
            else None
        )
        alt_stream = self.alt_stream
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
                alt_stream=alt_stream,
            ),
            prefix=f"{prefix}.layers",
            layer_num_offset=0,
        )
        # ATOM_MINIMAX_M3_MOE_TAIL_IN_AR: a layer's output is the next layer's input
        # (its forward pre-hooks read it too).
        for idx in range(self.start_layer, self.end_layer - 1):
            if isinstance(self.layers[idx], MiniMaxM3DecoderLayer):
                self.layers[idx]._next_layer = weakref.ref(self.layers[idx + 1])

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
        # Sequence-parallel residual at prefill sizes (ATOM_M3_PREFILL_SP_AR): TP only, one PP stage
        # (the residual crosses no stage boundary), every layer on the bf16 / MXFP8 AR + norm ops,
        # and no TBO (its micro-batches would interleave the owner-row split).
        self.sp_ar = bool(
            envs.ATOM_M3_PREFILL_SP_AR
            and sp_allreduce_supported()
            and get_tensor_model_parallel_world_size() > 1
            and get_pp_group().world_size == 1
            and not getattr(atom_config, "enable_tbo", False)
            and all(
                isinstance(layer, MiniMaxM3DecoderLayer) and layer.sp_ar_capable()
                for layer in self.layers
            )
        )
        MiniMaxM3Model.sp_ar_enabled = self.sp_ar

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
        # Eagle3's capture clones the residual stream, which the sequence-parallel one is not.
        sp_ar = self.sp_ar and not self.aux_hidden_state_layers
        for idx in range(self.start_layer, self.end_layer):
            if idx in self.aux_hidden_state_layers:
                hidden_states, residual, aux_hidden_state = self.layers[idx](
                    positions, hidden_states, residual, capture_aux=True
                )
                aux_hidden_states.append(aux_hidden_state)
            elif sp_ar:
                hidden_states, residual = self.layers[idx](
                    positions, hidden_states, residual, sp_ar=True
                )
            else:
                hidden_states, residual = self.layers[idx](
                    positions, hidden_states, residual
                )

        if not get_pp_group().is_last_rank:
            return IntermediateTensors(
                {"hidden_states": hidden_states, "residual": residual}
            )

        if sp_ar:
            hidden_states, _ = sp_fused_allreduce_gemma_rms_norm(
                hidden_states, residual, self.norm
            )
        else:
            hidden_states, _ = fused_allreduce_gemma_rms_norm(
                hidden_states, residual, self.norm
            )
        if aux_hidden_states:
            return hidden_states, aux_hidden_states
        return hidden_states

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        num_fused_shared = getattr(self.config, "n_shared_experts", 0) or 0
        return make_minimax_m3_expert_params_mapping(
            self.config.num_local_experts + num_fused_shared
        )


class MiniMaxM3SparseForCausalLM(nn.Module):
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
        # Outside the compiled MiniMaxM3Model on purpose: a supported decode step of
        # up to MAX_TOKENS tokens runs the mono kernels, everything else the
        # original model.
        if self._mono.supports(
            input_ids, positions, intermediate_tensors, inputs_embeds
        ):
            return self._mono.forward(input_ids, positions)
        if self.embed_before_model and inputs_embeds is None:
            inputs_embeds = self.embed_input_ids(input_ids)
        return self.model(input_ids, positions, intermediate_tensors, inputs_embeds)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        return self.lm_head(hidden_states)

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

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        return self.language_model.get_expert_mapping()


# The plugin entry points (vLLM/SGLang) bring their own vision stacks and keep
# the text-only view; native ATOM serving selects the VL class directly.
MiniMaxM3SparseForConditionalGeneration = (
    MiniMaxM3SparseForConditionalGenerationTextOnly
)
