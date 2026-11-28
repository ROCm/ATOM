# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""Inference-only MiniMax-M3 model support for ATOM."""

import contextlib
import logging
import weakref
from typing import ClassVar

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

from atom.config import Config, QuantizationConfig, get_current_atom_config
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
from atom.model_ops.moe import FusedMoE, Mxfp4MoEMethod, moe_after_router_hook
from atom.model_ops.swiglu_oai import swiglu_oai_split

try:  # aiter with the grouped MoE output epilogue (routed scale + shared add)
    from aiter.ops.flydsl.grouped_moe_gfx1250 import (
        moe_output_epilogue as _moe_output_epilogue,
    )
except ImportError:  # pragma: no cover - older aiter keeps the separate ops
    _moe_output_epilogue = None
try:  # aiter with the routed MoE tail deferred into the next all-reduce
    from aiter.ops.flydsl.moe_deferred_tail import moe_defer_tail as _moe_defer_tail
except ImportError:  # pragma: no cover - older aiter: the flag is a no-op
    _moe_defer_tail = None
try:  # aiter with the shared-expert fold into the routed MoE GEMMs
    from aiter.ops.flydsl.grouped_moe_gfx1250 import (
        moe_shared_expert_fold as _moe_shared_expert_fold,
    )
    from aiter.ops.flydsl.grouped_moe_gfx1250 import (
        prepare_shared_fold_weights as _prepare_shared_fold_weights,
    )
except ImportError:  # pragma: no cover - older aiter: the flag is a no-op
    _moe_shared_expert_fold = _prepare_shared_fold_weights = None
from atom.model_ops.utils import atom_parameter
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
from atom.utils.region_graph import token_graph

# ATOM_MINIMAX_M3_SHARED_FOLD: batches of at most this many tokens keep the
# separate shared MLP. aiter's T<=64 fold launch (routed GEMM2 16x128 with the
# 64x64 shared tiles) was measured nondeterministic under hot-expert routing
# (16% of launches wrote garbage 16-B pieces into routed rows, MI455X); larger
# buckets measured 0/300. Remove once that launch is fixed.
_SHARED_FOLD_MAX_DECLINED_TOKENS = 64
_shared_fold_floor_logged = False

# The largest MoE call whose tail is deferred into the next all-reduce (the k5
# one-shot kernel's row cap; larger calls would only run the deferred launch in
# front of another AR path).
_TAIL_IN_AR_MAX_ROWS = 256


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

    def forward(
        self, x: torch.Tensor, x_scale: torch.Tensor | None = None
    ) -> torch.Tensor:
        gate_up = self.gate_up_proj(x, x_scale=x_scale)
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
        if getattr(lin.weight, "is_shuffled", False) or getattr(
            w, "is_shuffled", False
        ):
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
            hasattr(self.experts, "supports_fused_router")
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
        global _shared_fold_floor_logged
        if self._shared_fold and not _shared_fold_floor_logged:
            _shared_fold_floor_logged = True
            logging.getLogger("atom").info(
                "MiniMax-M3 shared fold: batches of <= %d tokens keep the separate "
                "shared MLP (the T<=64 fold launch is nondeterministic)",
                _SHARED_FOLD_MAX_DECLINED_TOKENS,
            )
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
                _moe_output_epilogue is not None
                and self.shared_experts is not None
                and not self.shared_input_from_moe_quant
                and isinstance(self.experts.quant_method, Mxfp4MoEMethod)
            )
            else None
        )
        # With the glue epilogue or the shared fold, the gather-reduce (+ routed scale + shared add) is left to the all-reduce that reads this
        # layer's output (aiter `moe_defer_tail`); `_owner_layer` (set by the
        # decoder layer) tells whether anything else reads it (forward hooks).
        self._tail_in_ar = bool(
            (self._glue_epilogue is not None or self._shared_fold)
            and _moe_defer_tail is not None
            and tp_size > 1
            and get_pp_group().world_size == 1
        )
        self._owner_layer = None
        # The router launches before the shared MLP's kernels (dual stream + glue
        # epilogue + fused router only).
        self._routed_first = bool(
            self._glue_epilogue is not None and self._front_fusion
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
        # The fold's gather-reduce (+ routed
        # scale + its shared rows) may go to the next all-reduce too.
        with (
            _moe_shared_expert_fold(
                self._shared_fold_weights, self._routed_scale()
            ) as fold,
            _moe_defer_tail() if defer else contextlib.nullcontext(),
        ):
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

        `launch_shared` (routed first, `shared_output` None): launches the shared
        MLP and returns its output; it runs right after the router launch (or
        after the routed MoE when no router hook ran). The gather-reduce may be
        deferred into the all-reduce that reads the returned (then unwritten)
        tensor."""
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

                    stack.enter_context(moe_after_router_hook(_after_router))
                if defer:
                    stack.enter_context(_moe_defer_tail())
                routed_output = self._route_and_experts(hidden_states)
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

    def single_stream_moe_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
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
            shared_output = self.shared_experts(hidden_states)
            return self._routed_plus_shared(hidden_states, shared_output).view(
                orig_shape
            )
        routed_output = self.routed_expert_forward(hidden_states)
        if self.shared_experts is not None:
            routed_output = routed_output + self.shared_experts(hidden_states)
        return routed_output.view(orig_shape)

    def dual_stream_moe_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Overlap the independent shared MLP with the routed MoE branch (or,
        with the shared fold, run it inside the routed MoE's launches)."""
        orig_shape = hidden_states.shape
        hidden_states = hidden_states.view(-1, orig_shape[-1])
        if (
            getattr(self, "_shared_fold_weights", None) is not None
            and hidden_states.shape[0] > _SHARED_FOLD_MAX_DECLINED_TOKENS
            and hidden_states.shape[0] not in self._shared_fold_declined
        ):
            return self._shared_fold_forward(hidden_states).view(orig_shape)
        if self.alt_stream is None:
            return self.single_stream_moe_forward(hidden_states).view(orig_shape)
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
                    launched.append(self.shared_experts(hidden_states))
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
            shared_output = self.shared_experts(hidden_states)

        if getattr(self, "_glue_epilogue", None) is not None:
            # The join moves to right before the gather-reduce that adds it.
            output = self._routed_plus_shared(
                hidden_states,
                shared_output,
                before_add=lambda: current_stream.wait_stream(alt_stream),
            )
            shared_output.record_stream(current_stream)
            return output.view(orig_shape)

        routed_output = self.routed_expert_forward(hidden_states)
        current_stream.wait_stream(alt_stream)
        shared_output.record_stream(current_stream)
        return (routed_output + shared_output).view(orig_shape)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self._use_dual_stream:
            return torch.ops.aiter.maybe_dual_stream_forward(hidden_states, self.prefix)
        return self.single_stream_moe_forward(hidden_states)


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

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        hidden_states_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
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
        attn_output = self.attn(q, k, v, positions, qkv=qkv)
        if isinstance(attn_output, tuple):
            return self.o_proj(*attn_output)
        return self.o_proj(attn_output)


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
            # The deferred MoE tail asks this layer whether anything
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
    ) -> (
        tuple[torch.Tensor, torch.Tensor]
        | tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ):
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
        else:
            hidden_states, residual = fused_allreduce_gemma_rms_norm(
                hidden_states, residual, self.input_layernorm
            )

        # Eagle3 aux hidden state = the all-reduced residual stream entering this
        # layer (post input-norm). Captured here, not as `hidden_states + residual`
        # in the model loop, because M3's fused all-reduce RMSNorm leaves that sum
        # TP-partial / NaN-prone under CUDAGraph.
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
            hidden_states = tensor_model_parallel_all_reduce(hidden_states)
        return hidden_states + residual


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
        # Deferred MoE tail: a layer's output is the next layer's input
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
