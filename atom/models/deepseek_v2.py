# SPDX-License-Identifier: Apache-2.0

# Adapted from
# https://github.com/huggingface/transformers/blob/v4.28.0/src/transformers/models/llama/modeling_llama.py
# Copyright 2023 The vLLM team.
# Copyright 2023 DeepSeek-AI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Inference-only DeepseekV2/DeepseekV3 model."""

import logging
from typing import Optional, Tuple, Union

import torch
from aiter import (
    QuantType,
    cp_gather_indexer_k_quant_cache,
    dtypes,
    fused_qk_rmsnorm,
    gemm_a8w8_blockscale_bpreshuffle,
    get_hip_quant,
    indexer_k_quant_and_cache,
    indexer_qk_rope_quant_and_cache,
    top_k_per_row_decode,
    top_k_per_row_prefill,
)
from aiter.dist.communication_op import tensor_model_parallel_all_reduce
from aiter.dist.parallel_state import get_pp_group, get_tensor_model_parallel_world_size
from aiter.jit.utils.torch_guard import torch_compile_guard
from aiter.ops.triton.fp8_mqa_logits import fp8_mqa_logits
from aiter.ops.triton.fused_fp8_quant import fused_reduce_rms_fp8_group_quant
from aiter.ops.triton.fused_mxfp4_quant import (
    fused_reduce_rms_mxfp4_quant,
    fused_rms_mxfp4_quant,
)
from aiter.ops.triton.pa_mqa_logits import deepgemm_fp8_paged_mqa_logits
from aiter.rotary_embedding import get_rope
from atom.config import Config, QuantizationConfig, get_current_atom_config
from atom.model_ops.activation import SiluAndMul
from atom.model_ops.attention_mla import (
    MLAModules,
    is_rocm_aiter_fp4bmm_enabled,
    triton_convert_req_index_to_global_index,
    triton_convert_req_index_to_global_index_dsa_prefill,
    triton_gather_kv_indices_sparse,
)
from atom.model_ops.base_attention import Attention
from atom.model_ops.embed_head import (
    ParallelLMHead,
    ReplicatedEmbedding,
    VocabParallelEmbedding,
)
from atom.model_ops.layernorm import LayerNorm, RMSNorm
from atom.model_ops.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    MergedReplicatedLinear,
    ReplicatedLinear,
    RowParallelLinear,
    use_fp4_non_shuffle_triton_gemm,
    use_triton_gemm,
)
from atom.model_ops.moe import FusedMoE
from atom.model_ops.topK import is_rocm_aiter_fusion_shared_expert_enabled
from atom.model_ops.utils import MXFP4_QUANT_BLOCK_SIZE, atom_parameter
from atom.models.utils import (
    IntermediateTensors,
    PPMissingLayer,
    make_empty_intermediate_tensors_factory,
    make_layers,
    maybe_prefix,
)
from atom.quant_spec import should_skip_online_quant
from atom.utils import envs
from atom.utils.custom_register import direct_register_custom_op

# Side-effect import: registers `torch.ops.aiter.maybe_dual_stream_forward`,
# shared with deepseek_v4. DeepseekV2MoE.forward dispatches via this op when
# `_use_dual_stream` is True so torch.compile/Dynamo treats stream code as opaque.
from atom.model_ops import module_dispatch_ops as _module_dispatch_ops  # noqa: F401
from atom.distributed.pcp_utils import (
    get_pcp_world_size,
    pcp_allgather_rerange,
    pcp_is_enabled,
    pcp_pad_dense,
    pcp_pad_len,
    pcp_round_robin_split,
)
from atom.utils.decorators import mark_trace, support_torch_compile
from atom.utils.forward_context import get_forward_context
from atom.plugin.vllm.attention.layer_sparse_mla import (
    IndexerDecoratorForPluginMode,
    DeepseekV32IndexerCacheDecoratorForPluginMode,
)
from torch import nn
from transformers import PretrainedConfig

# from vllm.model_executor.layers.quantization.utils.fp8_utils import per_token_group_quant_fp8


logger = logging.getLogger("atom")
if use_triton_gemm():
    try:
        from aiter.ops.triton.gemm_a8w8_blockscale import (
            gemm_a8w8_blockscale_preshuffle,
        )
        from aiter.ops.triton.gemm_a16w8_blockscale import (
            gemm_a16w8_blockscale_preshuffle,
        )
        from aiter.ops.triton.gemm_a16wfp4 import gemm_a16wfp4_preshuffle
        from aiter.ops.triton.gemm_afp4wfp4 import gemm_afp4wfp4_preshuffle
    except ImportError as e:
        logger.warning(
            f"Triton GEMM kernels not available: {e}. Ensure AITER is up-to-date."
        )
        gemm_afp4wfp4_preshuffle = None
        gemm_a16wfp4_preshuffle = None
        gemm_a8w8_blockscale_preshuffle = None
        gemm_a16w8_blockscale_preshuffle = None

ENABLE_DS_QKNORM_QUANT_FUSION = envs.ATOM_ENABLE_DS_QKNORM_QUANT_FUSION
ENABLE_DS_QKNORM_FUSION = envs.ATOM_ENABLE_DS_QKNORM_FUSION
ENABLE_ALLREDUCE_RMSNORM_FUSION = envs.ATOM_ENABLE_ALLREDUCE_RMSNORM_FUSION
ENABLE_DS_INPUT_RMSNORM_QUANT_FUSION = envs.ATOM_ENABLE_DS_INPUT_RMSNORM_QUANT_FUSION
ENABLE_DS_INDEXER_QK_ROPE_CACHE_FUSION = (
    envs.ATOM_ENABLE_DS_INDEXER_QK_ROPE_CACHE_FUSION
)
SPARSE_INDEXER_LOGITS_BUDGET_MB = envs.ATOM_SPARSE_INDEXER_LOGITS_BUDGET_MB
ENABLE_GLM_FUSED_INDEXER = envs.ATOM_ENABLE_GLM_FUSED_INDEXER
_FP8_DTYPES = tuple(
    dtype
    for dtype in (
        getattr(torch, "float8_e4m3fn", None),
        getattr(torch, "float8_e4m3fnuz", None),
    )
    if dtype is not None
)


def _pcp_active() -> bool:
    """True when Prefill Context Parallel must reshape the current forward.

    PCP only reshapes the *sparse-MLA prefill* path (round-robin query split +
    full-KV all-gather). It returns False and the code stays byte-for-byte
    identical to the non-PCP path when any of these hold:
      * pcp_size == 1;
      * decode (PCP keeps decode on the full, already-cached KV);
      * dummy/warmup runs (graphs are captured on the full token layout — this
        matches the metadata builder, which gates its reindex on
        `not batch.is_dummy_run`, so model split and metadata reindex stay in
        lock-step);
      * short batches that fall back to dense MHA prefill (max_seqlen_k <=
        index_topk): only the sparse indexer / sparse-attn path is PCP-wired.

    Every PCP call site (model split/gather, indexer, MLA write, metadata
    reindex) keys off the SAME batch-global condition so they agree within a
    forward.
    """
    if not pcp_is_enabled():
        return False
    ctx = get_forward_context()
    context = getattr(ctx, "context", None)
    attn_metadata = getattr(ctx, "attn_metadata", None)
    if context is None or attn_metadata is None:
        return False
    if not bool(context.is_prefill) or bool(getattr(context, "is_dummy_run", False)):
        return False
    index_topk = getattr(get_current_atom_config().hf_config, "index_topk", None)
    if index_topk is None:
        return False
    return int(getattr(attn_metadata, "max_seqlen_k", 0)) > int(index_topk)


def _install_increment_version_pcp_shim() -> None:
    """Make ``torch.autograd.graph.increment_version`` tolerate a torch bug that
    only surfaces under PCP + torch.compile.

    Under Prefill Context Parallel the sparse indexer must run through a
    Dynamo-opaque custom op (``indexer_with_output``) so its runtime
    ``_pcp_active()`` branch is not baked to the warmup value. Inserting that op
    reshapes the pre-attention piecewise submodule, and torch's *inference*
    runtime wrapper (``keep_input_mutations=True``, hard-coded in
    ``torch/_inductor/compile_fx.py``) then resolves that submodule's
    ``mutated_graph_handled_indices_seen_by_autograd`` against a runtime arg list
    that interleaves SymInt shape symbols. The mutated-input index lands on a
    SymInt, so ``increment_version`` is handed a non-tensor:

        RuntimeError: increment_version expects each element ... to be a tensor
        IndexError:   list index out of range   (when the index is past the end)

    The baseline (indexer inlined, no extra op) only dodges this by arg-layout
    luck. Crucially the version counter is *autograd-only* metadata: in inference
    (no backward) it is never read, and the real in-place mutation is applied by
    the compiled kernel regardless of this bump. So it is safe to filter the
    iterable to real tensors and swallow the out-of-range walk. This is a strict
    no-op for every well-formed call (all-tensor iterables materialize to the
    same list), so baseline / non-PCP compilation is byte-for-byte unaffected.
    """
    import torch.autograd.graph as _agraph

    if getattr(_agraph.increment_version, "_pcp_shim", False):
        return
    _orig_increment_version = _agraph.increment_version

    def increment_version(tensor):
        if not isinstance(tensor, torch.Tensor):
            safe = []
            try:
                for t in tensor:
                    if isinstance(t, torch.Tensor):
                        safe.append(t)
            except IndexError:
                # torch handed us `(args[i] for i in <bad indices>)`; stop at the
                # first out-of-range index (autograd metadata only — see docstring).
                pass
            tensor = safe
        return _orig_increment_version(tensor)

    increment_version._pcp_shim = True
    # runtime_wrappers.py calls this via attribute access on the module, so
    # rebinding the module attribute is enough for it to pick up the shim.
    _agraph.increment_version = increment_version


_install_increment_version_pcp_shim()


def _enable_non_triton_global_mxfp4_input_norm_quant(
    config: PretrainedConfig,
    quant_config: Optional[QuantizationConfig],
    quant_dtype: Optional[torch.dtype],
    is_mtp_block: bool,
) -> bool:
    if (
        is_mtp_block
        or quant_dtype != dtypes.fp4x2
        or quant_config is None
        or quant_config.quant_method != "quark"
        or quant_config.quant_dtype != dtypes.fp4x2
        or quant_config.layer_pattern_specs
    ):
        return False
    architectures = set(getattr(config, "architectures", None) or [])
    return bool(
        architectures & {"DeepseekV2ForCausalLM", "DeepseekV3ForCausalLM"}
    ) or str(getattr(config, "model_type", "")).lower() in {
        "deepseek_v2",
        "deepseek_v3",
        "deepseek_v32",
        "deepseek_v4",
    }


def _supports_fused_indexer_kernel_config(config: PretrainedConfig) -> bool:
    if not hasattr(config, "index_topk"):
        return False
    # GLM-5.2 (glm_moe_dsa) shares DeepSeek-V3.2's sparse-MLA indexer: same dims
    # (index_head_dim=128, qk_rope_head_dim=64), same per_1x128 fp8 quant, and the
    # indexer rope is always neox for both. The fused kernel path is therefore
    # math-equivalent to the per-op path, so allow it here (gated by an env flag for
    # easy rollback). This also enables the wk+weights_proj GEMM-merge for GLM — its
    # checkpoint uses the standard indexer.wk / indexer.weights_proj tensor names
    # (the "indexers_proj" alias only lives in the HF quant config), so the merge
    # loads correctly; see _can_fuse_indexer_wk_weights_proj.
    if getattr(config, "model_type", None) == "glm_moe_dsa":
        if not ENABLE_GLM_FUSED_INDEXER:
            return False
    return (
        getattr(config, "index_head_dim", None) == 128
        and getattr(config, "qk_rope_head_dim", None) == 64
    )


def _is_neox_rope_style(
    config: PretrainedConfig, interleave_attr: str, *, default_is_neox: bool
) -> bool:
    """Resolve ``is_neox_style`` from a model's ``*_interleave`` rope config flag.

    neox and interleaved (GPT-J) are the two mutually-exclusive rope layouts, so
    an interleave flag of True means ``is_neox_style=False``. A missing or null
    flag falls back to ``default_is_neox`` — the layout of DeepSeek checkpoints
    that predate the flag, which differs per rope instance (see call sites):
    DeepSeek's main MLA rope is interleaved, but its V3.2 indexer rope is neox.
    """
    interleave = getattr(config, interleave_attr, None)
    if interleave is None:
        return default_is_neox
    return not bool(interleave)


def _can_fuse_indexer_wk_weights_proj(
    config: PretrainedConfig,
    quant_config: Optional[QuantizationConfig],
    indexer_prefixes: list[str],
) -> bool:
    if not ENABLE_DS_INDEXER_QK_ROPE_CACHE_FUSION:
        return False
    if not _supports_fused_indexer_kernel_config(config):
        return False
    # GLM-5.2 (glm_moe_dsa) reuses the same indexer weight layout as DeepSeek-V3.2:
    # separate indexer.wk (fp8 block-scale) + indexer.weights_proj (bf16). The
    # "indexers_proj" name only appears in the HF quant config's modules_to_not_convert
    # list (remapped via quant_exclude_name_mapping); the actual checkpoint tensors use
    # the standard indexer.wk / indexer.weights_proj paths, which is exactly what the
    # packed_modules_mapping merge and IndexerWkWeightsProjLinear's fp8-wk load expect.
    # So GLM takes the same wk+weights_proj GEMM-merge path as V3.2 below.
    if quant_config is None:
        return True

    for indexer_prefix in indexer_prefixes:
        wk_quant_config = quant_config.get_layer_quant_config(f"{indexer_prefix}.wk")
        if (
            wk_quant_config.quant_type != QuantType.No
            and wk_quant_config.quant_dtype != dtypes.fp8
        ):
            return False
    return True


def _extract_layer_index_from_prefix(prefix: str) -> int:
    for part in reversed(prefix.split(".")):
        if part.isdigit():
            return int(part)
    return 0


def _should_skip_index_topk(config: PretrainedConfig, prefix: str) -> bool:
    if not getattr(config, "use_index_cache", False):
        # IndexShare (e.g. GLM-5.2): index_topk_freq > 1 shares the indexer across
        # layers, so enable the cache even if the config omits the flag; otherwise
        # there is nothing to skip.
        if int(getattr(config, "index_topk_freq", 1)) > 1:
            config.use_index_cache = True
        else:
            return False

    layer_id = _extract_layer_index_from_prefix(prefix)

    # GLM-5.2 MTP layer (index >= num_hidden_layers): the MTP block ships its
    # OWN indexer weights and computes its own top-k for the drafted position,
    # so do not skip it. `index_share_for_mtp_iteration` only concerns sharing
    # across MULTIPLE MTP draft steps (num_speculative_tokens>1); it does NOT
    # mean the MTP reuses the target model's index. Matches vLLM upstream and
    # the ATOM sglang plugin, which both run the MTP indexer independently.
    num_hidden_layers = getattr(config, "num_hidden_layers", None)
    if num_hidden_layers is not None and layer_id >= num_hidden_layers:
        return False

    # GLM-5.2 IndexShare: per-layer schedule, "shared" reuses the prior "full"
    # layer's topk. Authoritative when present; else fall back to pattern/freq.
    indexer_types = getattr(config, "indexer_types", None)
    if indexer_types is not None:
        return (
            0 <= layer_id < len(indexer_types) and indexer_types[layer_id] == "shared"
        )

    index_topk_pattern = getattr(config, "index_topk_pattern", None)
    if index_topk_pattern is not None:
        return (
            0 <= layer_id < len(index_topk_pattern)
            and index_topk_pattern[layer_id] == "S"
        )

    index_topk_freq = int(getattr(config, "index_topk_freq", 1))
    if index_topk_freq <= 0:
        raise ValueError("index_topk_freq must be a positive integer")
    # offset defaults to 1 = prior `layer_id - 1` behavior for DeepSeek configs.
    offset = int(getattr(config, "index_skip_topk_offset", 1))
    return max(layer_id - offset, 0) % index_topk_freq != 0


def _indexer_weights_shared(config: PretrainedConfig, prefix: str) -> bool:
    """GLM-5.2 IndexShare: "shared" layers carry no indexer weights (they reuse
    the prior "full" layer), so don't build params for them. DeepSeek: per-layer."""
    indexer_types = getattr(config, "indexer_types", None)
    if indexer_types is None:
        return False
    layer_id = _extract_layer_index_from_prefix(prefix)
    return 0 <= layer_id < len(indexer_types) and indexer_types[layer_id] == "shared"


def _fuse_rmsnorm_fp4_quant_fake(
    x1: torch.Tensor,
    x1_weight: torch.Tensor,
    x1_epsilon: float,
    x2: Optional[torch.Tensor] = None,
    x2_weight: Optional[torch.Tensor] = None,
    x2_epsilon: Optional[float] = None,
    res1: Optional[torch.Tensor] = None,
    shuffle: bool = True,
    scale_shuffle_padding: bool = True,
    output_unquantized_inp1: bool = False,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    m, n1 = x1.shape
    n2 = x2.shape[1] if x2 is not None else 0

    out1_quantized = torch.empty((m, n1 // 2), dtype=torch.uint8, device=x1.device)

    scale_n_valid = (n1 + MXFP4_QUANT_BLOCK_SIZE - 1) // MXFP4_QUANT_BLOCK_SIZE

    if scale_shuffle_padding:
        scale_m = ((m + 255) // 256) * 256
        scale_n = ((scale_n_valid + 7) // 8) * 8
    else:
        scale_m = m
        scale_n = scale_n_valid

    out1_bs = torch.empty((scale_m, scale_n), dtype=torch.uint8, device=x1.device)

    out2 = None
    if x2 is not None:
        out2 = torch.empty((m, n2), dtype=x1.dtype, device=x1.device)

    out_res1 = None
    if res1 is not None:
        out_res1 = torch.empty((m, n1), dtype=x1.dtype, device=x1.device)

    out1_unquantized = None
    return out1_quantized, out1_bs, out1_unquantized, out2, out_res1


def _mxfp4_activation_quant_layout(num_tokens: int) -> Tuple[bool, bool]:
    if use_fp4_non_shuffle_triton_gemm():
        return False, False
    if use_triton_gemm():
        should_shuffle = num_tokens >= MXFP4_QUANT_BLOCK_SIZE
        return should_shuffle, should_shuffle
    return True, True


def _fused_rms_fp8_quant_fake(
    x1: torch.Tensor,
    x1_weight: torch.Tensor,
    x1_epsilon: float,
    x2: Optional[torch.Tensor] = None,
    x2_weight: Optional[torch.Tensor] = None,
    x2_epsilon: Optional[float] = None,
    res1: Optional[torch.Tensor] = None,
    dtype_quant: torch.dtype = dtypes.fp8,
    group_size: int = 128,
    quant_type: Optional[int] = None,
    output_unquantized_inp1: bool = False,
    transpose_scale: bool = False,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    m, n1 = x1.shape
    no_quant = quant_type is None or quant_type == QuantType.No.value
    if not no_quant:
        out1_quantized = torch.empty((m, n1), dtype=dtype_quant, device=x1.device)
    else:
        out1_quantized = torch.empty_like(x1)
    if no_quant:
        out1_bs = None
    elif quant_type == QuantType.per_Token.value:
        out1_bs = torch.empty((m, 1), dtype=torch.float32, device=x1.device)
    else:
        num_bs_cols = (n1 + group_size - 1) // group_size
        out1_bs = torch.empty((m, num_bs_cols), dtype=torch.float32, device=x1.device)
    out1_unquantized = torch.empty_like(x1) if output_unquantized_inp1 else None
    out2 = None
    if x2 is not None:
        _, n2 = x2.shape
        out2 = torch.empty((m, n2), dtype=x1.dtype, device=x1.device)
    out_res1 = None
    if res1 is not None:
        out_res1 = torch.empty((m, n1), dtype=x1.dtype, device=x1.device)
    return out1_quantized, out1_bs, out1_unquantized, out2, out_res1


@torch_compile_guard(gen_fake=_fuse_rmsnorm_fp4_quant_fake)
def _fuse_rmsnorm_fp4_quant(
    x1: torch.Tensor,
    x1_weight: torch.Tensor,
    x1_epsilon: float,
    x2: Optional[torch.Tensor] = None,
    x2_weight: Optional[torch.Tensor] = None,
    x2_epsilon: Optional[float] = None,
    res1: Optional[torch.Tensor] = None,
    shuffle: bool = True,
    scale_shuffle_padding: bool = True,
    output_unquantized_inp1: bool = False,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    (out1_quantized, out1_bs), _out1_unquantized, out2, out_res1 = (
        fused_rms_mxfp4_quant(
            x1=x1,
            x1_weight=x1_weight,
            x1_epsilon=x1_epsilon,
            x2=x2,
            x2_weight=x2_weight,
            x2_epsilon=0.0 if x2_epsilon is None else x2_epsilon,
            res1=res1,
            shuffle=shuffle,
            scale_shuffle_padding=scale_shuffle_padding,
            output_unquantized_inp1=output_unquantized_inp1,
        )
    )

    out1_unquantized = None
    return out1_quantized, out1_bs, out1_unquantized, out2, out_res1


@torch_compile_guard(gen_fake=_fused_rms_fp8_quant_fake)
def _fused_rms_fp8_quant(
    x1: torch.Tensor,
    x1_weight: torch.Tensor,
    x1_epsilon: float,
    x2: Optional[torch.Tensor] = None,
    x2_weight: Optional[torch.Tensor] = None,
    x2_epsilon: Optional[float] = None,
    res1: Optional[torch.Tensor] = None,
    dtype_quant: torch.dtype = dtypes.fp8,
    group_size: int = 128,
    quant_type: Optional[int] = None,
    output_unquantized_inp1: bool = False,
    transpose_scale: bool = False,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    out1_quantized, out1_bs, out1_unquantized, out2, out_res1 = (
        _fused_rms_fp8_quant_fake(
            x1,
            x1_weight,
            x1_epsilon,
            x2,
            x2_weight,
            x2_epsilon,
            res1,
            dtype_quant,
            group_size,
            quant_type,
            output_unquantized_inp1,
            transpose_scale,
        )
    )

    if quant_type is None:
        quant_type = QuantType.No
    else:
        quant_type = QuantType(quant_type)

    fused_qk_rmsnorm(
        q_out_quantized=out1_quantized,
        q_out_scale=out1_bs,
        q=x1,
        q_weight=x1_weight,
        q_epsilon=x1_epsilon,
        q_out_unquantized=out1_unquantized,
        k_out=out2,
        q_res_out=out_res1,
        k=x2,
        k_weight=x2_weight,
        k_epsilon=x2_epsilon,
        q_residual=res1,
        quant_type=quant_type,
        group_size=group_size,
        transpose_scale=transpose_scale,
    )
    return out1_quantized, out1_bs, out1_unquantized, out2, out_res1


@mark_trace(prefix="rmsnorm_quant", torch_compile=True)
def _fuse_rmsnorm_quant(
    x1: torch.Tensor,
    x1_weight: torch.Tensor,
    x1_epsilon: float,
    x2: Optional[torch.Tensor] = None,
    x2_weight: Optional[torch.Tensor] = None,
    x2_epsilon: Optional[float] = None,
    res1: Optional[torch.Tensor] = None,
    dtype_quant: torch.dtype = dtypes.fp8,
    shuffle: bool = True,
    scale_shuffle_padding: bool = False,
    group_size: int = 128,
    quant_type: Optional[int] = None,
    output_unquantized_inp1: bool = False,
    transpose_scale: bool = False,
):
    if dtype_quant == dtypes.fp4x2:
        out1_quantized, out1_bs, out1_unquantized, out2, out_res1 = (
            _fuse_rmsnorm_fp4_quant(
                x1,
                x1_weight,
                x1_epsilon,
                x2,
                x2_weight,
                x2_epsilon,
                res1,
                shuffle,
                scale_shuffle_padding,
                output_unquantized_inp1,
            )
        )
    elif dtype_quant == dtypes.fp8 or dtype_quant == torch.bfloat16:
        out1_quantized, out1_bs, out1_unquantized, out2, out_res1 = (
            _fused_rms_fp8_quant(
                x1,
                x1_weight,
                x1_epsilon,
                x2,
                x2_weight,
                x2_epsilon,
                res1,
                dtype_quant=dtype_quant,
                group_size=group_size,
                quant_type=quant_type,
                output_unquantized_inp1=output_unquantized_inp1,
                transpose_scale=transpose_scale,
            )
        )
    else:
        raise ValueError(
            f"No fused rmsnorm quant kernel availble for quant dtype: {dtype_quant}."
        )
    return (out1_quantized, out1_bs), out1_unquantized, out2, out_res1


def _fuse_qkv_a_proj_reduce_rmsnorm_quant_fp4_fake(
    hidden_states_quant: torch.Tensor,
    weight_qkv_a_proj: torch.Tensor,
    weight_scale_qkv_a_proj: torch.Tensor,
    q_a_layernorm_weight: torch.Tensor,
    q_a_layernorm_variance_epsilon: float,
    kv_a_layernorm_weight: torch.Tensor,
    kv_a_layernorm_variance_epsilon: float,
    q_lora_rank: int,
    kv_lora_rank: int,
    qk_rope_head_dim: int,
    hidden_states_quant_scale: Optional[torch.Tensor] = None,
    shuffle: Optional[bool] = True,
    scale_shuffle_padding: Optional[bool] = True,
    output_unquantized_inp1: Optional[bool] = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    M = hidden_states_quant.shape[0]
    device = hidden_states_quant.device
    q_c = torch.empty((M, q_lora_rank // 2), dtype=torch.uint8, device=device)
    scale_n_valid = (q_lora_rank + MXFP4_QUANT_BLOCK_SIZE - 1) // MXFP4_QUANT_BLOCK_SIZE
    if scale_shuffle_padding:
        scale_m = ((M + 255) // 256) * 256
        scale_n = ((scale_n_valid + 7) // 8) * 8
    else:
        scale_m = M
        scale_n = scale_n_valid
    q_c_scale = torch.empty((scale_m, scale_n), dtype=torch.uint8, device=device)
    kv_c_normed = torch.empty((M, kv_lora_rank), dtype=torch.bfloat16, device=device)
    k_pe = torch.empty(
        (M, q_lora_rank + kv_lora_rank + qk_rope_head_dim),
        dtype=torch.bfloat16,
        device=device,
    )[..., :qk_rope_head_dim]
    return q_c, q_c_scale, kv_c_normed, k_pe


def _fuse_qkv_a_proj_reduce_rmsnorm_quant_fp8_fake(
    hidden_states_quant: torch.Tensor,
    weight_qkv_a_proj: torch.Tensor,
    weight_scale_qkv_a_proj: torch.Tensor,
    q_a_layernorm_weight: torch.Tensor,
    q_a_layernorm_variance_epsilon: float,
    kv_a_layernorm_weight: torch.Tensor,
    kv_a_layernorm_variance_epsilon: float,
    q_lora_rank: int,
    kv_lora_rank: int,
    qk_rope_head_dim: int,
    hidden_states_quant_scale: Optional[torch.Tensor] = None,
    output_unquantized_inp1: Optional[bool] = False,
    transpose_scale: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    M = hidden_states_quant.shape[0]
    FP8_QUANT_BLOCK_SIZE = 128
    device = hidden_states_quant.device
    q_c = torch.empty((M, q_lora_rank), dtype=dtypes.fp8, device=device)
    scale_n = (q_lora_rank + FP8_QUANT_BLOCK_SIZE - 1) // FP8_QUANT_BLOCK_SIZE
    q_c_scale = torch.empty((M, scale_n), dtype=dtypes.fp8, device=device)
    kv_c_normed = torch.empty((M, kv_lora_rank), dtype=torch.bfloat16, device=device)
    k_pe = torch.empty(
        (M, q_lora_rank + kv_lora_rank + qk_rope_head_dim),
        dtype=torch.bfloat16,
        device=device,
    )[..., :qk_rope_head_dim]
    return q_c, q_c_scale, kv_c_normed, k_pe


@torch_compile_guard(
    gen_fake=_fuse_qkv_a_proj_reduce_rmsnorm_quant_fp4_fake, mutates_args=[]
)
def _fuse_qkv_a_proj_reduce_rmsnorm_quant_fp4(
    hidden_states_quant: torch.Tensor,
    weight_qkv_a_proj: torch.Tensor,
    weight_scale_qkv_a_proj: torch.Tensor,
    q_a_layernorm_weight: torch.Tensor,
    q_a_layernorm_variance_epsilon: float,
    kv_a_layernorm_weight: torch.Tensor,
    kv_a_layernorm_variance_epsilon: float,
    q_lora_rank: int,
    kv_lora_rank: int,
    qk_rope_head_dim: int,
    hidden_states_quant_scale: Optional[torch.Tensor] = None,
    shuffle: Optional[bool] = True,
    scale_shuffle_padding: Optional[bool] = True,
    output_unquantized_inp1: Optional[bool] = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    M = hidden_states_quant.shape[0]

    if hidden_states_quant_scale is None:
        if M <= MXFP4_QUANT_BLOCK_SIZE:
            qkv_lora = gemm_a16wfp4_preshuffle(
                hidden_states_quant,
                weight_qkv_a_proj.view(torch.uint8).view(
                    weight_qkv_a_proj.shape[0] // 16, -1
                ),
                weight_scale_qkv_a_proj.view(torch.uint8).view(
                    weight_scale_qkv_a_proj.shape[0] // MXFP4_QUANT_BLOCK_SIZE, -1
                ),
                prequant=True,
                skip_reduce=True,
            )
        else:
            quant_func = get_hip_quant(QuantType.per_1x32)
            x, x_scale = quant_func(
                hidden_states_quant,
                quant_dtype=dtypes.fp4x2,
                shuffle=(M >= MXFP4_QUANT_BLOCK_SIZE),
            )

            if M >= MXFP4_QUANT_BLOCK_SIZE:
                x_scale = x_scale.view(torch.uint8).view(
                    x_scale.shape[0] // MXFP4_QUANT_BLOCK_SIZE, -1
                )
            else:
                x_scale = x_scale[:M, ...].view(torch.uint8)

            qkv_lora = gemm_afp4wfp4_preshuffle(
                x.view(torch.uint8),
                weight_qkv_a_proj.view(torch.uint8).view(
                    weight_qkv_a_proj.shape[0] // 16, -1
                ),
                x_scale,
                weight_scale_qkv_a_proj.view(torch.uint8).view(
                    weight_scale_qkv_a_proj.shape[0] // MXFP4_QUANT_BLOCK_SIZE, -1
                ),
                skip_reduce=True,
            )
    else:
        if M >= MXFP4_QUANT_BLOCK_SIZE:
            hidden_states_quant_scale = hidden_states_quant_scale.view(
                torch.uint8
            ).view(hidden_states_quant_scale.shape[0] // MXFP4_QUANT_BLOCK_SIZE, -1)
        else:
            hidden_states_quant_scale = hidden_states_quant_scale[:M, ...].view(
                torch.uint8
            )

        qkv_lora = gemm_afp4wfp4_preshuffle(
            hidden_states_quant.view(torch.uint8),
            weight_qkv_a_proj.view(torch.uint8).view(
                weight_qkv_a_proj.shape[0] // 16, -1
            ),
            hidden_states_quant_scale,
            weight_scale_qkv_a_proj.view(torch.uint8).view(
                weight_scale_qkv_a_proj.shape[0] // MXFP4_QUANT_BLOCK_SIZE, -1
            ),
            skip_reduce=True,
        )

    q_c, kv_c, k_pe = torch.split(
        qkv_lora,
        [q_lora_rank, kv_lora_rank, qk_rope_head_dim],
        dim=-1,
    )

    shuffle_bool = shuffle and (M >= MXFP4_QUANT_BLOCK_SIZE)

    k_pe_reduced = None
    k_pe_reduced_out = None
    if k_pe.dim() == 3:
        device = hidden_states_quant.device
        k_pe_reduced = k_pe
        k_pe_reduced_out = torch.empty(
            (M, q_lora_rank + kv_lora_rank + qk_rope_head_dim),
            dtype=torch.bfloat16,
            device=device,
        )[..., :qk_rope_head_dim]
    (q_c, q_c_scale), _, kv_c_normed, _, k_pe_reduced_out = (
        fused_reduce_rms_mxfp4_quant(
            q_c,
            q_a_layernorm_weight,
            q_a_layernorm_variance_epsilon,
            kv_c,
            kv_a_layernorm_weight,
            kv_a_layernorm_variance_epsilon,
            k_pe_reduced,
            res1=None,
            shuffle=shuffle_bool,
            scale_shuffle_padding=scale_shuffle_padding,
            output_unquantized_inp1=output_unquantized_inp1,
            dtype=torch.bfloat16,
            out3=k_pe_reduced_out,
        )
    )

    if k_pe_reduced_out is not None:
        k_pe = k_pe_reduced_out
    return q_c, q_c_scale, kv_c_normed, k_pe


@mark_trace(prefix="qkv_a_proj_reduce_rmsnorm", torch_compile=True)
@torch_compile_guard(
    gen_fake=_fuse_qkv_a_proj_reduce_rmsnorm_quant_fp8_fake, mutates_args=[]
)
def _fuse_qkv_a_proj_reduce_rmsnorm_quant_fp8(
    hidden_states_quant: torch.Tensor,
    weight_qkv_a_proj: torch.Tensor,
    weight_scale_qkv_a_proj: torch.Tensor,
    q_a_layernorm_weight: torch.Tensor,
    q_a_layernorm_variance_epsilon: float,
    kv_a_layernorm_weight: torch.Tensor,
    kv_a_layernorm_variance_epsilon: float,
    q_lora_rank: int,
    kv_lora_rank: int,
    qk_rope_head_dim: int,
    hidden_states_quant_scale: Optional[torch.Tensor] = None,
    output_unquantized_inp1: Optional[bool] = False,
    transpose_scale: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    M = hidden_states_quant.shape[0]

    # NOTE: this fused path always calls aiter's *preshuffle* blockscale GEMMs,
    # which require a 16x16-shuffled weight. fused_qkv_a_proj is flagged with
    # needs_preshuffled_weight=True so the loader shuffles it once even under the
    # non-preshuffle path (ATOM_FP8_BLOCKSCALE_WEIGHT_PRESHUFFLE=0) -- see
    # LinearBase.process_weights_after_loading.

    if hidden_states_quant_scale is None:
        if M <= 32:
            qkv_lora = gemm_a16w8_blockscale_preshuffle(
                hidden_states_quant,
                weight_qkv_a_proj.view(weight_qkv_a_proj.shape[0] // 16, -1),
                weight_scale_qkv_a_proj,
                prequant=False,
                skip_reduce=True,
            )
        else:
            quant_func = get_hip_quant(QuantType.per_1x128)
            x, x_scale = quant_func(
                hidden_states_quant,
                quant_dtype=dtypes.fp8,
                transpose_scale=transpose_scale,
            )
            if M <= 128:
                qkv_lora = gemm_a8w8_blockscale_preshuffle(
                    x,
                    weight_qkv_a_proj.view(weight_qkv_a_proj.shape[0] // 16, -1),
                    x_scale,
                    weight_scale_qkv_a_proj,
                    skip_reduce=True,
                )
            else:
                qkv_lora = gemm_a8w8_blockscale_bpreshuffle(
                    x,
                    weight_qkv_a_proj,
                    x_scale,
                    weight_scale_qkv_a_proj,
                    torch.bfloat16,
                )
    else:
        if M <= 128:
            qkv_lora = gemm_a8w8_blockscale_preshuffle(
                hidden_states_quant,
                weight_qkv_a_proj.view(weight_qkv_a_proj.shape[0] // 16, -1),
                hidden_states_quant_scale,
                weight_scale_qkv_a_proj,
                skip_reduce=True,
            )
        else:
            qkv_lora = gemm_a8w8_blockscale_bpreshuffle(
                hidden_states_quant,
                weight_qkv_a_proj,
                hidden_states_quant_scale,
                weight_scale_qkv_a_proj,
                torch.bfloat16,
            )

    q_c, kv_c, k_pe = torch.split(
        qkv_lora,
        [q_lora_rank, kv_lora_rank, qk_rope_head_dim],
        dim=-1,
    )

    k_pe_reduced = None
    k_pe_reduced_out = None
    if k_pe.dim() == 3:
        device = hidden_states_quant.device
        k_pe_reduced = k_pe
        k_pe_reduced_out = torch.empty(
            (M, q_lora_rank + kv_lora_rank + qk_rope_head_dim),
            dtype=torch.bfloat16,
            device=device,
        )[..., :qk_rope_head_dim]
    (q_c, q_c_scale), _, kv_c_normed, _, k_pe_reduced_out = (
        fused_reduce_rms_fp8_group_quant(
            q_c,
            q_a_layernorm_weight,
            q_a_layernorm_variance_epsilon,
            kv_c,
            kv_a_layernorm_weight,
            kv_a_layernorm_variance_epsilon,
            k_pe_reduced,
            res1=None,
            output_unquantized_inp1=output_unquantized_inp1,
            dtype=torch.bfloat16,
            out3=k_pe_reduced_out,
            transpose_scale=transpose_scale,
        )
    )

    if k_pe_reduced_out is not None:
        k_pe = k_pe_reduced_out

    return q_c, q_c_scale, kv_c_normed, k_pe


def _fuse_qkv_a_proj_reduce_rmsnorm_quant(
    hidden_states_quant: torch.Tensor,
    weight_qkv_a_proj: torch.Tensor,
    weight_scale_qkv_a_proj: torch.Tensor,
    q_a_layernorm_weight: torch.Tensor,
    q_a_layernorm_variance_epsilon: float,
    kv_a_layernorm_weight: torch.Tensor,
    kv_a_layernorm_variance_epsilon: float,
    q_lora_rank: int,
    kv_lora_rank: int,
    qk_rope_head_dim: int,
    dtype_quant=dtypes.fp8,
    hidden_states_quant_scale: Optional[torch.Tensor] = None,
    shuffle: Optional[bool] = False,
    scale_shuffle_padding: Optional[bool] = False,
    group_size: Optional[int] = 128,
    output_unquantized_inp1: Optional[bool] = False,
    transpose_scale: Optional[bool] = False,
):
    if dtype_quant == dtypes.fp4x2:
        q_c, q_c_scale, kv_c_normed, k_pe = _fuse_qkv_a_proj_reduce_rmsnorm_quant_fp4(
            hidden_states_quant,
            weight_qkv_a_proj,
            weight_scale_qkv_a_proj,
            q_a_layernorm_weight,
            q_a_layernorm_variance_epsilon,
            kv_a_layernorm_weight,
            kv_a_layernorm_variance_epsilon,
            q_lora_rank,
            kv_lora_rank,
            qk_rope_head_dim,
            hidden_states_quant_scale,
            shuffle,
            scale_shuffle_padding,
            output_unquantized_inp1,
        )
    elif dtype_quant == dtypes.fp8:
        q_c, q_c_scale, kv_c_normed, k_pe = _fuse_qkv_a_proj_reduce_rmsnorm_quant_fp8(
            hidden_states_quant,
            weight_qkv_a_proj,
            weight_scale_qkv_a_proj,
            q_a_layernorm_weight,
            q_a_layernorm_variance_epsilon,
            kv_a_layernorm_weight,
            kv_a_layernorm_variance_epsilon,
            q_lora_rank,
            kv_lora_rank,
            qk_rope_head_dim,
            hidden_states_quant_scale,
            output_unquantized_inp1,
            transpose_scale,
        )
    else:
        raise ValueError(
            f"No fused rmsnorm quant kernel availble for quant dtype: {dtype_quant}."
        )

    # logger.info(f"{q_c.shape=}, {q_c_scale.shape=}, {kv_c_normed.shape=}, {k_pe.shape=}, {q_c.stride()=}, {q_c_scale.stride()=}, {kv_c_normed.stride()=}, {k_pe.stride()=}")
    return q_c, q_c_scale, kv_c_normed, k_pe


class DeepseekV2MLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        quant_config: Optional[QuantizationConfig] = None,
        reduce_results: bool = True,
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
            reduce_results=reduce_results,
            prefix=f"{prefix}.down_proj",
        )
        if hidden_act != "silu":
            raise ValueError(
                f"Unsupported activation: {hidden_act}. Only silu is supported for now."
            )
        self.act_fn = SiluAndMul()

    def forward(self, x):
        gate_up = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x = self.down_proj(x)
        return x


class DeepseekV2MoE(nn.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        reduce_results: bool = True,
        prefix: str = "",
        alt_stream: Optional[torch.cuda.Stream] = None,
    ):
        super().__init__()
        self.tp_size = get_tensor_model_parallel_world_size()
        self.routed_scaling_factor = config.routed_scaling_factor
        self.n_shared_experts = config.n_shared_experts
        self.reduce_results = reduce_results

        if config.hidden_act != "silu":
            raise ValueError(
                f"Unsupported activation: {config.hidden_act}. "
                "Only silu is supported for now."
            )

        self.gate = ReplicatedLinear(
            config.hidden_size,
            config.n_routed_experts,
            bias=False,
            # MoE gate normally remains unquantized, but may not declare as ignore layers in quantization_config
            quant_config=None,
            prefix=f"{prefix}.gate",
        )
        if config.topk_method == "noaux_tc":
            self.gate.e_score_correction_bias = atom_parameter(
                torch.empty(config.n_routed_experts)
            )
        else:
            self.gate.e_score_correction_bias = None

        self.experts = FusedMoE(
            num_experts=config.n_routed_experts,
            top_k=config.num_experts_per_tok,
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            reduce_results=False,
            renormalize=config.norm_topk_prob,
            quant_config=quant_config,
            use_grouped_topk=True,
            num_expert_group=config.n_group,
            topk_group=config.topk_group,
            prefix=f"{prefix}.experts",
            scoring_func=config.scoring_func,
            e_score_correction_bias=self.gate.e_score_correction_bias,
            config=config,
        )

        # Dual-stream support: parallelize shared expert and routed expert
        # computation using a separate CUDA stream. Registered as a custom op
        # (dual_stream_moe_forward) so it is opaque to torch.compile/Dynamo.
        self._use_dual_stream = False
        self.alt_stream = alt_stream
        self.prefix = prefix
        self.is_rocm_aiter_fusion_shared_expert_enabled = (
            is_rocm_aiter_fusion_shared_expert_enabled(
                shared_expert_prefix=f"{prefix}.shared_experts",
                routed_expert_prefix=f"{prefix}.experts",
            )
        )

        if config.n_shared_experts is not None:
            if not self.is_rocm_aiter_fusion_shared_expert_enabled:
                tbo_active = get_current_atom_config().enable_tbo
                if envs.ATOM_DUAL_STREAM_MOE_TOKEN_THRESHOLD > 0 and not tbo_active:
                    self._use_dual_stream = True
                    compilation_config = get_current_atom_config().compilation_config
                    compilation_config.static_forward_context[prefix] = self
                intermediate_size = (
                    config.moe_intermediate_size * config.n_shared_experts
                )
                self.shared_experts = DeepseekV2MLP(
                    hidden_size=config.hidden_size,
                    intermediate_size=intermediate_size,
                    hidden_act=config.hidden_act,
                    quant_config=quant_config,
                    reduce_results=False,
                    prefix=f"{prefix}.shared_experts",
                )

    def routed_expert_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        router_logits = self.gate(hidden_states)
        final_hidden_states = self.experts(
            hidden_states=hidden_states, router_logits=router_logits
        )
        return final_hidden_states

    def combine_outputs(
        self,
        final_hidden_states: torch.Tensor,
        shared_output: Optional[torch.Tensor],
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        if shared_output is not None:
            if hidden_states.dtype != torch.float16:
                final_hidden_states = final_hidden_states + shared_output
            else:
                final_hidden_states = final_hidden_states + shared_output * (
                    1.0 / self.routed_scaling_factor
                )
        if self.tp_size > 1 and self.reduce_results:
            final_hidden_states = tensor_model_parallel_all_reduce(final_hidden_states)
        return final_hidden_states

    def dual_stream_moe_forward(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        num_tokens, hidden_dim = hidden_states.shape
        current_stream = torch.cuda.current_stream()
        alt_stream = self.alt_stream

        alt_stream.wait_stream(current_stream)

        with torch.cuda.stream(alt_stream):
            # final_hidden_states = self.routed_expert_forward(hidden_states)
            shared_output = self.shared_experts(hidden_states)

        final_hidden_states = self.routed_expert_forward(hidden_states)
        # shared_output = self.shared_experts(hidden_states)

        current_stream.wait_stream(alt_stream)

        final_hidden_states = self.combine_outputs(
            final_hidden_states, shared_output, hidden_states
        )
        return final_hidden_states.view(num_tokens, hidden_dim)

    def single_stream_moe_forward(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        shared_output = None
        if (
            self.n_shared_experts is not None
            and not self.is_rocm_aiter_fusion_shared_expert_enabled
        ):
            shared_output = self.shared_experts(hidden_states)

        final_hidden_states = self.routed_expert_forward(hidden_states)
        final_hidden_states = self.combine_outputs(
            final_hidden_states, shared_output, hidden_states
        )
        return final_hidden_states

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        assert (
            hidden_states.dim() == 2
        ), f"Expected hidden_states to be 2D (seq_len, hidden_dim), but got {hidden_states.dim()}D, with shape {hidden_states.shape}"
        assert (
            hidden_states.shape[1] == self.experts.hidden_size
        ), f"Hidden states dimension {hidden_states.shape[1]} does not match expected {self.experts.hidden_size}"

        if self._use_dual_stream:
            return torch.ops.aiter.maybe_dual_stream_forward(hidden_states, self.prefix)

        # Non-dual-stream path: shared experts + routed experts sequentially
        return self.single_stream_moe_forward(hidden_states)


def yarn_get_mscale(scale: float = 1, mscale: float = 1) -> float:
    import math

    if scale <= 1:
        return 1.0
    return 0.1 * mscale * math.log(scale) + 1.0


@DeepseekV32IndexerCacheDecoratorForPluginMode
class DeepseekV32IndexerCache(nn.Module):
    def __init__(
        self, head_dim: int, dtype: torch.dtype, prefix: str, cache_config: str
    ):
        super().__init__()
        self.kv_cache = [torch.tensor([])]
        self.head_dim = head_dim
        self.prefix = prefix
        self.cache_config = cache_config
        self.dtype = dtype


def sparse_attn_indexer(
    hidden_states: torch.Tensor,
    k_cache_prefix: str,
    kv_cache: torch.Tensor,
    q_input: torch.Tensor,
    k: torch.Tensor,
    weights: torch.Tensor,
    quant_block_size: int,
    scale_fmt: Optional[str],
    topk_tokens: int,
    head_dim: int,
    max_model_len: int,
    total_seq_lens: int,
    sparse_kv_indices_buffer: torch.Tensor,
    k_norm_weight: torch.Tensor,
    k_norm_bias: torch.Tensor,
    k_norm_eps: float,
    positions: torch.Tensor,
    cos_cache: torch.Tensor,
    sin_cache: torch.Tensor,
    weights_scale: float,
    is_neox_style: bool,
    use_qk_rope_cache_fusion: bool,
) -> torch.Tensor:
    topk_indices = torch.empty(
        hidden_states.shape[0],
        topk_tokens,
        dtype=torch.int32,
        device=hidden_states.device,
    )
    # careful! this will be None in dummy run
    forward_context = get_forward_context()
    attn_metadata = forward_context.attn_metadata
    context = forward_context.context
    slot_mapping = attn_metadata.slot_mapping
    # Skip for dummy runs to avoid corrupting KV cache
    if forward_context.context.is_dummy_run:
        # dummy runner
        return torch.zeros_like(weights, dtype=torch.float32)
    # For MTP verify decode, max_seqlen_q > 1 so total decode tokens = batch_size * max_seqlen_q
    num_decode_tokens = (
        context.batch_size * attn_metadata.max_seqlen_q if not context.is_prefill else 0
    )
    runner_block_size = get_current_atom_config().kv_cache_block_size
    kv_cache = kv_cache.view(-1, runner_block_size, kv_cache.shape[-1])
    # PCP prefill: `k` (and `positions`) arrive as the full PADDED key set
    # [S_pad] produced by an all-gather of the round-robin shards. The KV-cache
    # write (driven by slot_mapping) and the gathered-KV sizing (total_kv =
    # k.shape[0]) both use the real token count S_real == slot_mapping length,
    # so trim the round-robin padding here. Gated on the same condition as the
    # caller's PCP path (sparse prefill with max_seqlen_k > topk); no-op
    # otherwise.
    if (
        pcp_is_enabled()
        and context.is_prefill
        and attn_metadata.max_seqlen_k > topk_tokens
    ):
        n_real = slot_mapping.shape[0]
        k = k[:n_real]
        if positions is not None:
            positions = positions[:n_real]
    if use_qk_rope_cache_fusion:
        q_bf16 = q_input
        q_fp8 = torch.empty_like(q_bf16, dtype=dtypes.fp8)
        weights_out = torch.empty(
            weights.shape, device=weights.device, dtype=torch.float32
        )
        indexer_qk_rope_quant_and_cache(
            q_bf16,
            q_fp8,
            weights,
            weights_out,
            k,
            kv_cache,
            slot_mapping,
            k_norm_weight,
            k_norm_bias,
            positions,
            cos_cache,
            sin_cache,
            k_norm_eps,
            quant_block_size,
            scale_fmt,
            weights_scale,
            preshuffle=True,
            is_neox=is_neox_style,
        )
        weights = weights_out
    else:
        q_fp8 = q_input
        indexer_k_quant_and_cache(
            k,
            kv_cache,
            slot_mapping,
            quant_block_size,
            scale_fmt,
            preshuffle=True,
        )
    if context.is_prefill:
        if attn_metadata.max_seqlen_k <= topk_tokens:
            return weights
        prefill_metadata = attn_metadata
        num_prefills = context.batch_size
        # Size the gathered-KV buffer off the KEY length, not the hidden/query
        # length. Under PCP the query side (hidden_states) is 1/pcp while `k` is
        # the full all-gathered key set, so `k.shape[0]` is the correct full
        # token count; without PCP the two are equal so behaviour is unchanged.
        # When has_cached, gather full KV (cached + new) for indexer top-k.
        total_kv = (
            prefill_metadata.total_kv if prefill_metadata.has_cached else k.shape[0]
        )
        k_fp8 = torch.empty([total_kv, head_dim], device=k.device, dtype=dtypes.fp8)
        k_scale = torch.empty([total_kv, 1], device=k.device, dtype=torch.float32)
        if prefill_metadata.block_tables.shape[0] < num_prefills:
            new_shape = (num_prefills, prefill_metadata.block_tables.shape[1])
            prefill_metadata.block_tables = torch.full(
                new_shape,
                -1,
                dtype=torch.long,
                device=prefill_metadata.block_tables.device,
            )
        cp_gather_indexer_k_quant_cache(
            kv_cache,
            k_fp8,
            k_scale.view(dtypes.fp8),
            prefill_metadata.block_tables,
            (
                prefill_metadata.cu_seqlens_k
                if prefill_metadata.has_cached
                else prefill_metadata.cu_seqlens_q
            ),
            preshuffle=True,
        )
        cu_seqlen_ks = prefill_metadata.cu_seqlen_ks
        cu_seqlen_ke = prefill_metadata.cu_seqlen_ke
        num_tokens = hidden_states.shape[0]
        q_prefill = q_fp8[num_decode_tokens:num_tokens]
        weights_prefill = weights[num_decode_tokens:num_tokens]
        num_rows = q_prefill.shape[0]
        assert topk_tokens == 2048, "top_k_per_row assumes size 2048"
        topk_indices_prefill = topk_indices[num_decode_tokens:num_tokens, :topk_tokens]
        # The dense logits buffer is [num_rows, total_kv] fp32. total_kv is the
        # sum of all co-scheduled prefill contexts and is unbounded by
        # max_num_batched_tokens, so a burst of long-context requests can push a
        # single allocation to tens of GiB (#1376). Under chunked prefill
        # num_rows is already capped by max_num_batched_tokens, so the OOM is
        # driven by total_kv (the column dim). Chunk along the Q (query-row)
        # dimension with q_chunk sized so the buffer [q_chunk, total_kv] fp32
        # stays within the memory budget — q_chunk shrinks as total_kv grows.
        # Each chunk still scores the FULL KV, so every row's top-k is computed
        # completely in one shot: the result is exact with no cross-chunk merge,
        # the kernel's column indices are already global (no remapping), and each
        # chunk writes straight into its output row slice (no copy). When the
        # budget is disabled (0) or a single chunk fits, the loop runs exactly
        # once and matches the original single-shot behavior.
        budget_bytes = SPARSE_INDEXER_LOGITS_BUDGET_MB * 1024 * 1024
        if (
            budget_bytes > 0
            and total_kv > 0
            and budget_bytes // (total_kv * 4) < num_rows
        ):
            # 4 bytes per fp32 logit; total_kv * 4 is one query row's footprint.
            # Round the budget-derived row count DOWN to keep the buffer within
            # budget: a multiple of 128 (aligned to the kernel's row tiling) in
            # the normal regime, avoiding the coarse power-of-2 doubling. When
            # the budget affords < 128 rows (extreme total_kv), fall back to a
            # power-of-2 floor so it degrades to 64/32/.../1 instead of
            # collapsing straight to 1.
            budget_rows = budget_bytes // (total_kv * 4)
            if budget_rows >= 128:
                chunk_tokens = (budget_rows // 128) * 128
            else:
                chunk_tokens = 1 << (max(1, budget_rows).bit_length() - 1)
        else:
            # Budget disabled, or a single chunk already fits all rows.
            chunk_tokens = num_rows
        for chunk_start in range(0, num_rows, chunk_tokens):
            chunk_end = min(chunk_start + chunk_tokens, num_rows)
            # Per-row window bounds slice 1:1 with this chunk's rows.
            row_starts = cu_seqlen_ks[chunk_start:chunk_end]
            row_ends = cu_seqlen_ke[chunk_start:chunk_end]
            logits = fp8_mqa_logits(
                Q=q_prefill[chunk_start:chunk_end],
                KV=k_fp8,
                kv_scales=k_scale,
                weights=weights_prefill[chunk_start:chunk_end],
                cu_starts=row_starts,
                cu_ends=row_ends,
            )
            top_k_per_row_prefill(
                logits=logits,
                rowStarts=row_starts,
                rowEnds=row_ends,
                indices=topk_indices_prefill[chunk_start:chunk_end],
                values=None,
                numRows=chunk_end - chunk_start,
                stride0=logits.stride(0),
                stride1=logits.stride(1),
            )
        triton_convert_req_index_to_global_index_dsa_prefill(
            attn_metadata.sparse_cu_seqlens_q,
            attn_metadata.sparse_kv_indptr,
            attn_metadata.token_to_seq_idxs,
            topk_indices,
            attn_metadata.block_tables,
            attn_metadata.cu_seqlens_k,
            NUM_TOPK_TOKENS=topk_tokens,
            PAGE_SIZE=runner_block_size,
            out=sparse_kv_indices_buffer,
        )
    else:
        decode_metadata = attn_metadata
        # kv_cache size requirement [num_block, block_size, n_head, head_dim],
        # we only have [num_block, block_size, head_dim],
        kv_cache = kv_cache.unsqueeze(-2)
        padded_q_fp8_decode_tokens = q_fp8[:num_decode_tokens].reshape(
            context.batch_size, -1, *q_fp8.shape[1:]
        )
        # TODO: move and optimize below logic with triton kernels
        batch_size = padded_q_fp8_decode_tokens.shape[0]
        next_n = padded_q_fp8_decode_tokens.shape[1]
        assert batch_size == context.batch_size
        num_padded_tokens = batch_size * next_n
        batch_size, next_n, heads, _ = padded_q_fp8_decode_tokens.shape
        logits = torch.empty(
            [batch_size * next_n, max_model_len], dtype=torch.float32, device="cuda"
        )
        deepgemm_fp8_paged_mqa_logits(
            padded_q_fp8_decode_tokens,
            kv_cache,
            weights[:num_padded_tokens],
            logits,
            decode_metadata.context_lens,
            attn_metadata.block_tables,
            max_model_len,
            KVBlockSize=runner_block_size,
            Preshuffle=True,
        )
        num_rows = logits.shape[0]
        assert topk_tokens == 2048, "top_k_per_row assumes size 2048"
        topk_indices_decode = topk_indices[:num_decode_tokens, :topk_tokens]
        top_k_per_row_decode(
            logits,
            next_n,
            decode_metadata.context_lens,
            topk_indices_decode,
            num_rows,
            logits.stride(0),
            logits.stride(1),
        )
        if attn_metadata.max_seqlen_q > 1:
            triton_gather_kv_indices_sparse(
                attn_metadata.sparse_kv_indptr,
                attn_metadata.token_to_seq_idxs,
                topk_indices,
                attn_metadata.kv_indices,
                attn_metadata.kv_indptr,
                NUM_TOPK_TOKENS=topk_tokens,
                out=sparse_kv_indices_buffer,
            )
        else:
            triton_convert_req_index_to_global_index(
                attn_metadata.cu_seqlens_q,
                attn_metadata.kv_indptr,
                attn_metadata.sparse_kv_indptr,
                attn_metadata.kv_indices,
                topk_indices,
                NUM_TOPK_TOKENS=topk_tokens,
                out=sparse_kv_indices_buffer,
            )
    return weights


def sparse_attn_indexer_fake(
    hidden_states: torch.Tensor,
    k_cache_prefix: str,
    kv_cache: torch.Tensor,
    q_input: torch.Tensor,
    k: torch.Tensor,
    weights: torch.Tensor,
    quant_block_size: int,
    scale_fmt: Optional[str],
    topk_tokens: int,
    head_dim: int,
    max_model_len: int,
    total_seq_lens: int,
    sparse_kv_indices_buffer: torch.Tensor,
    k_norm_weight: torch.Tensor,
    k_norm_bias: torch.Tensor,
    k_norm_eps: float,
    positions: torch.Tensor,
    cos_cache: torch.Tensor,
    sin_cache: torch.Tensor,
    weights_scale: float,
    is_neox_style: bool,
    use_qk_rope_cache_fusion: bool,
) -> torch.Tensor:
    # profile run
    # NOTE(Chen): create the max possible flattened_kv. So that
    # profile_run can get correct memory usage.
    _flattened_kv = torch.empty(
        [total_seq_lens, head_dim + 4], device=k.device, dtype=torch.uint8
    )
    _k_fp8 = _flattened_kv[..., :head_dim].view(torch.float8_e4m3fn).contiguous()
    _k_scale = _flattened_kv[..., head_dim:].view(torch.float32).contiguous()
    return torch.empty(weights.shape, device=weights.device, dtype=torch.float32)


direct_register_custom_op(
    op_name="sparse_attn_indexer",
    op_func=sparse_attn_indexer,
    mutates_args=["sparse_kv_indices_buffer"],
    fake_impl=sparse_attn_indexer_fake,
)


def _dequant_fp8_block_to_bf16(
    weight_fp8: torch.Tensor,
    scale: torch.Tensor,
    block_size: int = 128,
) -> torch.Tensor:
    """Dequantize FP8 wk weights to BF16 for BF16-only fused GEMMs.

    DeepSeek-V3.2 stores indexer.wk with block scales, while some PTPC
    quantized checkpoints store a per-output-channel scale vector.
    """
    out_dim, in_dim = weight_fp8.shape
    scale = scale.float()
    if scale.dim() == 1:
        if scale.numel() != out_dim:
            raise ValueError(
                "FP8 per-channel dequant expects one scale per output row, "
                f"got scale {tuple(scale.shape)} for weight {tuple(weight_fp8.shape)}"
            )
        return (weight_fp8.float() * scale[:, None]).bfloat16()
    if scale.dim() == 2 and tuple(scale.shape) == (out_dim, 1):
        return (weight_fp8.float() * scale).bfloat16()

    if out_dim % block_size != 0 or in_dim % block_size != 0:
        raise ValueError(
            "FP8 block dequant expects dimensions divisible by "
            f"{block_size}, got {tuple(weight_fp8.shape)}"
        )
    expected_scale_shape = (out_dim // block_size, in_dim // block_size)
    if tuple(scale.shape) != expected_scale_shape:
        raise ValueError(
            "FP8 block dequant scale shape mismatch: expected "
            f"{expected_scale_shape}, got {tuple(scale.shape)} for weight "
            f"{tuple(weight_fp8.shape)}"
        )
    weight = (
        weight_fp8.unflatten(0, (-1, block_size))
        .unflatten(-1, (-1, block_size))
        .float()
    )
    return (weight * scale[:, None, :, None]).flatten(2, 3).flatten(0, 1).bfloat16()


class IndexerWkWeightsProjLinear(MergedReplicatedLinear):
    """Fused Indexer wk + weights projection with FP8 wk load support."""

    def __init__(
        self,
        hidden_size: int,
        head_dim: int,
        n_head: int,
        prefix: str = "",
    ):
        self._wk_pending_weight: Optional[torch.Tensor] = None
        self._wk_pending_scale: Optional[torch.Tensor] = None
        self._wk_loaded = False
        super().__init__(
            hidden_size,
            [head_dim, n_head],
            bias=False,
            quant_config=None,
            prefix=prefix,
        )
        # Checkpoints may store indexer.wk as FP8 plus block or per-channel
        # scales. The fused GEMM runs in BF16, so this parameter only helps
        # collect the scale during loading and is not consumed in forward.
        self.weight_scale = atom_parameter(
            torch.empty(
                ((head_dim + 127) // 128, (hidden_size + 127) // 128),
                dtype=torch.float32,
            )
        )
        self.weight_scale.weight_loader_process = self.weight_loader_process
        self.weight_scale.weight_loader = self.weight_loader

    def _maybe_load_pending_wk(self) -> None:
        if self._wk_pending_weight is None or self._wk_pending_scale is None:
            return
        wk_weight_fp8 = self._wk_pending_weight
        if wk_weight_fp8.device != self._wk_pending_scale.device:
            wk_weight_fp8 = wk_weight_fp8.to(self._wk_pending_scale.device)
        wk_weight = _dequant_fp8_block_to_bf16(
            wk_weight_fp8,
            self._wk_pending_scale,
        )
        super().weight_loader(self.weight, wk_weight, 0)
        self._wk_pending_weight = None
        self._wk_pending_scale = None
        self._wk_loaded = True

    def weight_loader(
        self,
        param: nn.Parameter,
        loaded_weight: torch.Tensor,
        loaded_shard_id: Optional[int] = None,
    ):
        if param is self.weight_scale:
            if loaded_shard_id == 0:
                if param.data.shape == loaded_weight.shape:
                    param.weight_loader_process(param.data, loaded_weight)
                self._wk_pending_scale = loaded_weight.detach().clone()
                self._maybe_load_pending_wk()
            return

        if (
            param is self.weight
            and loaded_shard_id == 0
            and loaded_weight.dtype in _FP8_DTYPES
        ):
            self._wk_pending_weight = loaded_weight.detach().clone()
            self._maybe_load_pending_wk()
            return

        if param is self.weight and loaded_shard_id == 0:
            self._wk_pending_weight = None
            self._wk_pending_scale = None
            self._wk_loaded = True

        super().weight_loader(param, loaded_weight, loaded_shard_id)

    def process_weights_after_loading(self):
        if self._wk_pending_weight is not None or (
            self._wk_pending_scale is not None and not self._wk_loaded
        ):
            raise RuntimeError(
                "Incomplete FP8 indexer.wk load: both weight and weight_scale "
                "are required before building wk_weights_proj."
            )
        if not self._wk_loaded:
            raise RuntimeError(
                "Missing indexer.wk load before building wk_weights_proj."
            )
        super().process_weights_after_loading()


@IndexerDecoratorForPluginMode
def _indexer_with_output_fake(
    hidden_states: torch.Tensor,
    qr: torch.Tensor,
    qr_scale: Optional[torch.Tensor],
    positions: torch.Tensor,
    layer_name: str,
    sparse_kv_indices_buffer: torch.Tensor,
) -> torch.Tensor:
    # Identity-passthrough contract: the op returns a fresh tensor shaped like
    # `qr` (see `indexer_with_output`). The caller consumes it as the query into
    # `mla_attn`, which keeps the op alive and ordered even independent of the
    # declared buffer mutation.
    return torch.empty_like(qr)


def indexer_with_output(
    hidden_states: torch.Tensor,
    qr: torch.Tensor,
    qr_scale: Optional[torch.Tensor],
    positions: torch.Tensor,
    layer_name: str,
    sparse_kv_indices_buffer: torch.Tensor,
) -> torch.Tensor:
    """Dynamo-opaque wrapper around ``Indexer.forward_impl``.

    Registered as a REGULAR custom op (like ``sparse_attn_indexer``), NOT a
    splitting op. Opacity — not a graph *split* — is what defeats the bake:
    Dynamo treats a custom op as a leaf and never traces the body, so the
    runtime ``_pcp_active()`` branch inside the indexer (round-robin k all-gather
    + separate q/k rope) is evaluated LIVE every forward instead of being frozen
    to its dummy-warmup value (``False``). Prefill is not cudagraph-captured, so
    the eager all-gather inside runs safely; decode never takes the PCP branch.

    Why regular and not ``@mark_spliting_op``: adding a second split point
    between ``qkv_a_proj`` and ``mla_attn`` shrinks the pre-attention submodule
    to a handful of ops, and AOT-autograd then mis-resolves that tiny piece's
    mutated-input index (``increment_version`` IndexError). Staying a plain
    opaque op keeps the pre-attention submodule identical in shape to the
    baseline's (which compiles cleanly).

    Buffer mutation MUST be declared. ``forward_impl`` writes the indexer's
    top-k result into ``sparse_kv_indices_buffer`` (via the nested eager
    ``sparse_attn_indexer`` op) and ``mla_attn`` reads that same buffer to know
    which KV to attend to. In the non-PCP path ``sparse_attn_indexer`` runs
    *directly* in the traced graph and declares ``mutates_args=
    ["sparse_kv_indices_buffer"]``, so inductor keeps the write ordered before
    the MLA read. Nesting it inside this opaque op HIDES that write; with
    ``mutates_args=[]`` inductor thinks the buffer is unchanged and the MLA reads
    a stale / mis-ordered copy — visible as token-stutter corruption ("errerr",
    doubled fragments) even on PCP-noop prompts. So the buffer is threaded
    through as an explicit arg and re-declared mutated here, restoring the exact
    write→read edge the baseline relies on. (Declaring the mutation re-triggers a
    latent AOT-autograd ``increment_version`` bug on graphs with SymInt args,
    which ``_install_increment_version_pcp_shim`` neutralizes.)

    The identity ``qr`` return, fed by the caller as the ``mla_attn`` query, is
    kept as belt-and-suspenders ordering + DCE protection.
    """
    self = get_current_atom_config().compilation_config.static_forward_context[
        layer_name
    ]
    # Side effect: writes sparse_kv_indices_buffer (via nested eager
    # sparse_attn_indexer). `self.sparse_kv_indices_buffer` is the same tensor
    # object passed in, so the declared mutation matches the real one.
    self.forward_impl(hidden_states, qr, qr_scale, positions)
    # Fresh tensor equal to qr; consumed by the caller as the mla_attn query.
    # Clone (not a bare return of qr) so the runtime output matches the fake's
    # fresh-tensor contract and never aliases an input.
    return qr.clone()


direct_register_custom_op(
    op_name="indexer_with_output",
    op_func=indexer_with_output,
    mutates_args=["sparse_kv_indices_buffer"],
    fake_impl=_indexer_with_output_fake,
)


class Indexer(nn.Module):
    def __init__(
        self,
        atom_config: Config,
        config: PretrainedConfig,
        hidden_size: int,
        q_lora_rank: int,
        quant_config: Optional[QuantizationConfig],
        cache_config: str,
        use_wk_weights_proj_fusion: bool = True,
        prefix: str = "",
    ):
        super().__init__()
        self.atom_config = atom_config
        self.config = config
        # self.indexer_cfg = config.attn_module_list_cfg[0]["attn_index"]
        self.topk_tokens = config.index_topk
        self.n_head = config.index_n_heads  # 64
        self.head_dim = config.index_head_dim  # 128
        self.rope_dim = config.qk_rope_head_dim  # 64
        self.q_lora_rank = q_lora_rank  # 1536
        # no tensor parallel, just replicated
        self.wq_b = ReplicatedLinear(
            self.q_lora_rank,
            self.head_dim * self.n_head,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.wq_b",
        )
        self.scale_fmt = "ue8m0"
        self.quant_func = get_hip_quant(QuantType.per_1x128)
        self.quant_block_size = 128  # TODO: get from config
        self.use_qk_rope_cache_fusion = (
            ENABLE_DS_INDEXER_QK_ROPE_CACHE_FUSION
            and _supports_fused_indexer_kernel_config(config)
            and self.head_dim == self.quant_block_size
            and self.rope_dim == self.head_dim // 2
        )
        self.use_wk_weights_proj_fusion = (
            use_wk_weights_proj_fusion and self.use_qk_rope_cache_fusion
        )
        if self.use_wk_weights_proj_fusion:
            self.wk_weights_proj = IndexerWkWeightsProjLinear(
                hidden_size,
                self.head_dim,
                self.n_head,
                prefix=f"{prefix}.wk_weights_proj",
            )
        else:
            self.wk = ReplicatedLinear(
                hidden_size,
                self.head_dim,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.wk",
            )
            self.weights_proj = ReplicatedLinear(
                hidden_size,
                self.n_head,
                quant_config=None,
                prefix=f"{prefix}.weights_proj",
            )
        self.k_norm = LayerNorm(self.head_dim, eps=1e-6)
        self.softmax_scale = self.head_dim**-0.5
        self._weights_scale = self.softmax_scale * self.n_head**-0.5

        # TODO (zyongye) change dim to fp8 later to (self.head_dim + 4)
        self.k_cache = DeepseekV32IndexerCache(
            head_dim=self.head_dim + 4,
            dtype=torch.uint8,
            prefix=f"{prefix}.k_cache",
            cache_config=cache_config,
        )
        self.max_model_len = atom_config.max_model_len
        self.prefix = prefix
        self.max_total_seq_len = atom_config.max_num_seqs * self.max_model_len
        # register_metadata_builder("indexer_attn_metadata", self.k_cache.get_attn_backend().get_builder_cls())

        self.sparse_kv_indices_buffer = torch.empty(0, dtype=torch.int32, device="cuda")
        atom_config.compilation_config.static_forward_context[prefix] = self

        # Rope module used by `forward_impl` (and the `indexer_with_output`
        # splitting op, which can't take a module arg). Bound by the owning
        # DeepseekV2MLAAttention right after construction; mirrors V4's
        # `self.indexer.rotary_emb = self.rotary_emb`.
        self.rotary_emb = None

        self.sparse_attn_indexer_impl = torch.ops.aiter.sparse_attn_indexer

    def forward(
        self,
        hidden_states: torch.Tensor,
        qr: torch.Tensor,
        qr_scale: Optional[torch.Tensor],
        positions,
        rotary_emb=None,
    ) -> torch.Tensor:
        # Under PCP, route the whole indexer through the Dynamo-opaque
        # `indexer_with_output` splitting op so the runtime `_pcp_active()` branch
        # (round-robin k all-gather + separate q/k rope) evaluates live instead of
        # being baked to its warmup value by torch.compile. `pcp_is_enabled()` is
        # a run-level constant (pcp world size is fixed for the process), so this
        # guard is compile-safe and a no-op — a direct `forward_impl` call, graph
        # unchanged — for non-PCP and plugin (SGLang / vLLM / RTP) backends.
        if pcp_is_enabled():
            # Returns `qr` (identity); the caller feeds it to mla_attn so the
            # opaque op stays live and ordered. The top-k result travels the
            # side-buffer (declared mutated, so its write is ordered before the
            # sparse-MLA read), not this return value.
            return torch.ops.aiter.indexer_with_output(
                hidden_states,
                qr,
                qr_scale,
                positions,
                self.prefix,
                self.sparse_kv_indices_buffer,
            )
        return self.forward_impl(hidden_states, qr, qr_scale, positions, rotary_emb)

    def forward_impl(
        self,
        hidden_states: torch.Tensor,
        qr: torch.Tensor,
        qr_scale: Optional[torch.Tensor],
        positions,
        rotary_emb=None,
    ) -> torch.Tensor:
        # The opaque `indexer_with_output` op can't pass a module, so it relies on
        # the bound `self.rotary_emb`; direct callers (non-PCP / plugins) may still
        # pass their own rope explicitly, which takes precedence.
        if rotary_emb is None:
            rotary_emb = self.rotary_emb
        q = self.wq_b(qr, qr_scale)
        q = q.view(-1, self.n_head, self.head_dim)

        if self.use_wk_weights_proj_fusion:
            k, weights = torch.split(
                self.wk_weights_proj(hidden_states),
                [self.head_dim, self.n_head],
                dim=-1,
            )
        else:
            k = self.wk(hidden_states)
            weights = self.weights_proj(hidden_states)

        # Under PCP prefill the fused qk-rope-cache kernel cannot be used: it
        # ropes q and writes k in one pass keyed on a single token count, but
        # here q/weights are this rank's 1/pcp queries while k must become the
        # FULL key set (every rank keeps full KV). So force the unfused path,
        # all-gather k (and the key positions) to the full padded sequence, and
        # rope q (1/pcp) and k (full) separately. The op then scores 1/pcp
        # queries against the gathered full KV and writes the full k-cache.
        pcp = _pcp_active()
        positions_op = positions
        if (not self.use_qk_rope_cache_fusion) or pcp:
            q_pe, _ = torch.split(
                q, [self.rope_dim, self.head_dim - self.rope_dim], dim=-1
            )
            if pcp:
                pcp_ws = get_pcp_world_size()
                # k is 1/pcp (from 1/pcp hidden); gather to full padded [S_pad].
                k = pcp_allgather_rerange(k, pcp_ws)
                positions_op = pcp_allgather_rerange(positions, pcp_ws)
            k = self.k_norm(k)
            k_pe, _ = torch.split(
                k, [self.rope_dim, self.head_dim - self.rope_dim], dim=-1
            )
            if pcp:
                # Rope q (1/pcp) and k (full) separately: they have different
                # token counts under PCP so they can't share one rope call. The
                # rope kernel is 2-component (ropes query AND key in place) and
                # requires a non-None partner, so pass a throwaway of the
                # matching length for the side we don't need. rope is in-place
                # on the rotary_dim views (q_pe/k_pe alias q/k).
                rotary_emb(positions, q_pe, torch.empty_like(q_pe))
                rotary_emb(positions_op, torch.empty_like(k_pe), k_pe)
            else:
                q_pe, k_pe = rotary_emb(positions, q_pe, k_pe)

            q = q.view(-1, self.head_dim)
            q_fp8, q_scale = self.quant_func(q, quant_dtype=dtypes.fp8)
            q_fp8 = q_fp8.view(-1, self.n_head, self.head_dim)
            q_scale = q_scale.view(-1, self.n_head, 1)
            weights = (weights.unsqueeze(-1) * q_scale * self._weights_scale).squeeze(
                -1
            )
            q_input = q_fp8
        else:
            q_input = q

        return self.sparse_attn_indexer_impl(
            hidden_states,
            self.k_cache.prefix,
            self.k_cache.kv_cache[0],
            q_input,
            k,
            weights,
            self.quant_block_size,
            self.scale_fmt,
            self.topk_tokens,
            self.head_dim,
            self.max_model_len,
            self.max_total_seq_len,
            self.sparse_kv_indices_buffer,
            self.k_norm.weight,
            self.k_norm.bias,
            self.k_norm.eps,
            positions_op,
            rotary_emb.cos_cache.squeeze(-2).squeeze(-2),
            rotary_emb.sin_cache.squeeze(-2).squeeze(-2),
            self._weights_scale,
            rotary_emb.is_neox_style,
            self.use_qk_rope_cache_fusion and not pcp,
        )


class DeepseekV2MLAAttention(nn.Module):
    """
    Main reference: DeepseekV2 paper, and FlashInfer Implementation
    (https://arxiv.org/abs/2405.04434 and https://github.com/flashinfer-ai/flashinfer/pull/551).

    For more info see MLACommonImpl in: vllm/attention/backends/mla/utils.py
    """

    def __init__(
        self,
        config: PretrainedConfig,
        hidden_size: int,
        num_heads: int,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        v_head_dim: int,
        q_lora_rank: Optional[int],
        kv_lora_rank: int,
        max_position_embeddings: int = 8192,
        cache_config: str = "bf16",
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        layer_num: int = 0,
        use_indexer_wk_weights_proj_fusion: Optional[bool] = None,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        self.v_head_dim = v_head_dim

        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        model_quant_config = quant_config

        self.num_heads = num_heads
        tp_size = get_tensor_model_parallel_world_size()
        assert num_heads % tp_size == 0
        self.num_local_heads = num_heads // tp_size

        self.scaling = self.qk_head_dim**-0.5
        self.max_position_embeddings = max_position_embeddings
        self.layer_num = layer_num

        # For FP4 and use_triton_gemm(), fused_qkv_a_proj and q_b_proj are AITER-Triton FP4 GEMMs but o_proj remains AITER BF16 GEMMs,
        # For FP8 and use_triton_gemm(), fused_qkv_a_proj is AITER-Triton FP8 GEMMs while others remain AITER FP8 GEMMs
        q_a_proj_name = (
            "fused_qkv_a_proj" if self.q_lora_rank is not None else "q_a_proj"
        )
        layer_quant_dtype = quant_config.get_layer_quant_config(
            f"{prefix}.{q_a_proj_name}"
        ).quant_dtype
        layer_quant_type = quant_config.get_layer_quant_config(
            f"{prefix}.{q_a_proj_name}"
        ).quant_type
        if layer_quant_dtype == dtypes.fp4x2:
            if not use_triton_gemm():
                source_quant_dtype = None
                # Full-MXFP4 V2 checkpoints store attention weights/scales on disk.
                # Keep their quant_config only for this narrow static Quark path.
                q_a_proj_quant_config = quant_config.get_layer_quant_config(
                    f"{prefix}.{q_a_proj_name}"
                )
                is_quark_static_mxfp4 = (
                    q_a_proj_quant_config.quant_method == "quark"
                    and layer_quant_type == QuantType.per_1x32
                )
                if is_quark_static_mxfp4:
                    base_quant_config = quant_config
                else:
                    quant_config = None
                    base_quant_config = None
            else:
                source_quant_dtype = torch.bfloat16
                base_quant_config = None
        else:
            source_quant_dtype = None
            # Check exclude patterns (e.g. W4A8 checkpoints exclude attention)
            if quant_config is not None and quant_config._is_excluded(prefix):
                quant_config = None
                base_quant_config = None
            else:
                base_quant_config = quant_config

        if self.q_lora_rank is not None:
            # self.q_a_proj = ReplicatedLinear(self.hidden_size,
            #                                  self.q_lora_rank,
            #                                  bias=False,
            #                                  quant_config=quant_config,
            #                                  prefix=f"{prefix}.q_a_proj")
            self.fused_qkv_a_proj = MergedReplicatedLinear(
                self.hidden_size,
                [self.q_lora_rank, self.kv_lora_rank + self.qk_rope_head_dim],
                bias=False,
                quant_config=quant_config,
                source_quant_dtype=source_quant_dtype,
                prefix=f"{prefix}.fused_qkv_a_proj",
            )
            # The fused qkv_a_proj forward calls *preshuffle* blockscale GEMMs, so
            # its weight must be 16x16-shuffled even when the global non-preshuffle
            # path (ATOM_FP8_BLOCKSCALE_WEIGHT_PRESHUFFLE=0) is selected. The loader
            # honors this flag in LinearBase.process_weights_after_loading.
            self.fused_qkv_a_proj.needs_preshuffled_weight = True
            self.q_a_layernorm = RMSNorm(self.q_lora_rank, eps=config.rms_norm_eps)
            self.q_b_proj = ColumnParallelLinear(
                q_lora_rank,
                self.num_heads * self.qk_head_dim,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.q_b_proj",
                source_quant_dtype=source_quant_dtype,
            )
        else:
            self.q_proj = ColumnParallelLinear(
                self.hidden_size,
                self.num_heads * self.qk_head_dim,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.q_proj",
                source_quant_dtype=source_quant_dtype,
            )

            self.kv_a_proj_with_mqa = ReplicatedLinear(
                self.hidden_size,
                self.kv_lora_rank + self.qk_rope_head_dim,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.kv_a_proj_with_mqa",
                source_quant_dtype=source_quant_dtype,
            )
        self.kv_a_layernorm = RMSNorm(self.kv_lora_rank, eps=config.rms_norm_eps)
        self.kv_b_proj = ColumnParallelLinear(
            self.kv_lora_rank,
            self.num_heads * (self.qk_nope_head_dim + self.v_head_dim),
            bias=False,
            quant_config=(
                quant_config if is_rocm_aiter_fp4bmm_enabled() else base_quant_config
            ),
            prefix=f"{prefix}.kv_b_proj",
            source_quant_dtype=(
                source_quant_dtype if is_rocm_aiter_fp4bmm_enabled() else None
            ),
        )
        self.o_proj = RowParallelLinear(
            self.num_heads * self.v_head_dim,
            self.hidden_size,
            bias=False,
            quant_config=base_quant_config,
            reduce_results=not ENABLE_ALLREDUCE_RMSNORM_FUSION,
            prefix=f"{prefix}.o_proj",
            source_quant_dtype=None,
        )

        rope_params = config.rope_parameters
        rope_theta = rope_params.get("rope_theta") or 10000
        # Only use YaRN scaling when config has it (e.g. DeepSeek with factor/type "yarn").
        # GLM-5 has no rope_scaling in config -> use default RoPE (no scaling).
        use_yarn = (
            rope_params.get("factor", 1.0) not in (1.0, None)
            or rope_params.get("type") in ("yarn", "deepseek_yarn")
            or rope_params.get("rope_type") in ("yarn", "deepseek_yarn")
        )
        if use_yarn:
            rope_scaling = dict(rope_params)
            rope_scaling["rope_type"] = "deepseek_yarn"
            if "original_max_position_embeddings" not in rope_scaling:
                factor = float(rope_scaling.get("factor", 1.0))
                rope_scaling["original_max_position_embeddings"] = (
                    int(max_position_embeddings / factor)
                    if factor > 0
                    else max_position_embeddings
                )
        else:
            rope_scaling = None
        self.rotary_emb = get_rope(
            qk_rope_head_dim,
            rotary_dim=qk_rope_head_dim,
            max_position=max_position_embeddings,
            base=rope_theta,
            rope_scaling=rope_scaling,
            # DeepSeek's main MLA rope is interleaved (is_neox_style=False) when
            # unspecified; GLM-5.x sets rope_interleave=true, i.e. also interleaved.
            is_neox_style=_is_neox_rope_style(
                config, "rope_interleave", default_is_neox=False
            ),
        )
        if rope_scaling:
            mscale_all_dim = rope_scaling.get("mscale_all_dim", False)
            scaling_factor = rope_scaling["factor"]
            mscale = yarn_get_mscale(scaling_factor, float(mscale_all_dim))
            self.scaling = self.scaling * mscale * mscale

        self.is_v32 = hasattr(config, "index_topk")
        self.skip_topk = False

        if self.is_v32:
            self.skip_topk = _should_skip_index_topk(config, prefix)
            self.indexer_rope_emb = get_rope(
                qk_rope_head_dim,
                rotary_dim=qk_rope_head_dim,
                max_position=max_position_embeddings,
                base=rope_theta,
                rope_scaling=rope_scaling,
                # DeepSeek-V3.2's indexer rope is neox (is_neox_style=True) when
                # unspecified; GLM-5.x sets indexer_rope_interleave=true to override
                # it to interleaved.
                is_neox_style=_is_neox_rope_style(
                    config, "indexer_rope_interleave", default_is_neox=True
                ),
            )
            if _indexer_weights_shared(config, prefix):
                # GLM-5.2 IndexShare: reuses prior "full" layer's indexer; the
                # forward and index-cache binding guard on `indexer is not None`.
                self.indexer = None
            else:
                self.indexer = Indexer(
                    get_current_atom_config(),
                    config,
                    hidden_size,
                    q_lora_rank,
                    base_quant_config,
                    cache_config,
                    (
                        _can_fuse_indexer_wk_weights_proj(
                            config,
                            model_quant_config,
                            [f"{prefix}.indexer"],
                        )
                        if use_indexer_wk_weights_proj_fusion is None
                        else use_indexer_wk_weights_proj_fusion
                    ),
                    f"{prefix}.indexer",
                )
                # Bind the indexer's rope so forward_impl (and the opaque
                # indexer_with_output splitting op) can rope without receiving a
                # module argument. Mirrors deepseek_v4.Attention.__init__.
                self.indexer.rotary_emb = self.indexer_rope_emb
        else:
            self.indexer_rope_emb = None
            self.indexer = None
        # In the MLA backend, kv_cache includes both k_c and
        # pe (i.e. decoupled position embeddings). In particular,
        # the concat_and_cache_mla op requires
        #     k_c.size(1) + k_pe.size(1) == kv_cache.size(2)
        # i.e.
        #     kv_lora_rank + qk_rope_head_dim == head_size

        mla_modules = MLAModules(
            q_lora_rank=self.q_lora_rank,
            kv_lora_rank=self.kv_lora_rank,
            qk_nope_head_dim=self.qk_nope_head_dim,
            qk_rope_head_dim=self.qk_rope_head_dim,
            qk_head_dim=self.qk_head_dim,
            v_head_dim=self.v_head_dim,
            rotary_emb=self.rotary_emb,
            q_proj=self.q_proj if self.q_lora_rank is None else self.q_b_proj,
            kv_b_proj=self.kv_b_proj,
            o_proj=self.o_proj,
            indexer=self.indexer,
            # v3.2 / GLM-5.2 runs sparse MLA on every layer. For GLM-5.2 IndexShare
            # "shared" layers self.indexer is None, but they must still run sparse
            # attention and reuse the prior full layer's top-k, so flag sparsity at
            # the model level rather than per-layer.
            is_sparse=self.is_v32,
            topk_tokens=(config.index_topk if self.is_v32 else None),
        )

        self.mla_attn = Attention(
            num_heads=self.num_local_heads,
            head_dim=self.kv_lora_rank + self.qk_rope_head_dim,
            scale=self.scaling,
            num_kv_heads=1,
            kv_cache_dtype=cache_config,
            layer_num=layer_num,
            use_mla=True,
            mla_modules=mla_modules,
            prefix=prefix,
        )

        # Enable q/k RMSNorm + q quant fusion for FP8 and FP4. The larger
        # qkv_a_proj + reduce + RMSNorm + quant fusion remains gated by
        # use_triton_gemm() in forward(), because that path depends on Triton GEMM.
        self.prefix = prefix
        # Online-aware scheme for the fused q/kv norm+quant feeding q_b_proj:
        # use the online target (applied after __init__), else the static config.
        # layer_quant_dtype/type stay static for the fp4x2 weight-loading branch.
        eff_dtype, eff_type = layer_quant_dtype, layer_quant_type
        if quant_config is not None and quant_config.online_quant:
            online_cfg = quant_config.get_layer_quant_config(
                f"{prefix}.{q_a_proj_name}", use_online_quant=True
            )
            if not should_skip_online_quant(eff_type, eff_dtype, online_cfg):
                eff_dtype, eff_type = online_cfg.quant_dtype, online_cfg.quant_type
        self.quant_dtype = eff_dtype
        self.qknorm_quant_type = None if eff_type is None else eff_type.value
        self.fuse_qknorm_quant = False
        # always fuse qknorm
        self.fuse_qknorm = ENABLE_DS_QKNORM_FUSION
        if quant_config is not None and ENABLE_DS_QKNORM_QUANT_FUSION:
            if eff_dtype in (dtypes.fp8, dtypes.fp4x2):
                self.fuse_qknorm_quant = True

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states_scale = None
        # When input_layernorm fused AR+RMSNorm+quant, hidden_states is a tuple.
        # A 3-tuple (fp8, scale, bf16) additionally carries the unquantized bf16
        # normed activation for the v32 indexer (see RMSNorm.fused_quant_emit_bf16);
        # a 2-tuple (fp8, scale) is the plain fused-quant output.
        indexer_hidden = None
        if isinstance(hidden_states, tuple):
            if len(hidden_states) == 3:
                hidden_states, hidden_states_scale, indexer_hidden = hidden_states
            else:
                hidden_states, hidden_states_scale = hidden_states

        if self.q_lora_rank is not None:
            if self.fuse_qknorm_quant and use_triton_gemm():
                q_c, q_c_scale, kv_c_normed, k_pe = (
                    _fuse_qkv_a_proj_reduce_rmsnorm_quant(
                        hidden_states,
                        self.fused_qkv_a_proj.weight,
                        self.fused_qkv_a_proj.weight_scale,
                        self.q_a_layernorm.weight,
                        self.q_a_layernorm.eps,
                        self.kv_a_layernorm.weight,
                        self.kv_a_layernorm.eps,
                        self.q_lora_rank,
                        self.kv_lora_rank,
                        self.qk_rope_head_dim,
                        dtype_quant=self.quant_dtype,
                        hidden_states_quant_scale=hidden_states_scale,
                        shuffle=True,
                        scale_shuffle_padding=True,
                        group_size=128,
                        output_unquantized_inp1=False,
                        transpose_scale=True,
                    )
                )
                hidden_states_or_q_c = q_c
                hidden_states_or_q_c_scale = q_c_scale
            else:
                qkv_lora = self.fused_qkv_a_proj(hidden_states, hidden_states_scale)
                # ckq = self.q_a_proj(hidden_states)
                q_c, kv_c, k_pe = torch.split(
                    qkv_lora,
                    [self.q_lora_rank, self.kv_lora_rank, self.qk_rope_head_dim],
                    dim=-1,
                )
                # fuse q_c norm + kv_c norm + quant of hidden_states_or_q_c
                if self.fuse_qknorm_quant or self.fuse_qknorm:
                    q_shuffle = False
                    q_scale_shuffle_padding = False
                    if self.quant_dtype == dtypes.fp4x2 and not use_triton_gemm():
                        q_shuffle, q_scale_shuffle_padding = (
                            _mxfp4_activation_quant_layout(q_c.shape[0])
                        )
                    (
                        (hidden_states_or_q_c, hidden_states_or_q_c_scale),
                        _,
                        kv_c_normed,
                        _,
                    ) = _fuse_rmsnorm_quant(
                        q_c,
                        self.q_a_layernorm.weight,
                        self.q_a_layernorm.eps,
                        kv_c,
                        self.kv_a_layernorm.weight,
                        self.kv_a_layernorm.eps,
                        None,
                        dtype_quant=self.quant_dtype,
                        shuffle=q_shuffle,
                        scale_shuffle_padding=q_scale_shuffle_padding,
                        group_size=128,
                        quant_type=self.qknorm_quant_type,
                        output_unquantized_inp1=False,
                        transpose_scale=True,
                    )
                else:
                    hidden_states_or_q_c = self.q_a_layernorm(q_c)
        else:
            hidden_states_or_q_c = hidden_states
            kv_c, k_pe = torch.split(
                self.kv_a_proj_with_mqa(hidden_states, hidden_states_scale),
                [self.kv_lora_rank, self.qk_rope_head_dim],
                dim=-1,
            )
        if not self.fuse_qknorm_quant and not self.fuse_qknorm:
            kv_c_normed = self.kv_a_layernorm(kv_c)
            hidden_states_or_q_c_scale = None
        if self.is_v32 and self.indexer is not None and not self.skip_topk:
            # The indexer's wk/weights_proj GEMMs run in BF16. When input_layernorm
            # fused the quant it emits a bf16 mirror (indexer_hidden); otherwise
            # hidden_states is already the bf16 normed activation.
            idx_ret = self.indexer(
                indexer_hidden if indexer_hidden is not None else hidden_states,
                hidden_states_or_q_c,
                hidden_states_or_q_c_scale,
                positions,
                self.indexer_rope_emb,
            )
            if pcp_is_enabled():
                # Under PCP the indexer runs through the opaque `indexer_with_output`
                # split op, which returns `hidden_states_or_q_c` unchanged
                # (identity). Feeding it forward as the mla_attn query is what keeps
                # the op live under torch.compile — its real result (top-k) is a
                # hidden write to the sparse buffer that mla_attn reads via `self` —
                # and orders the write before that read. `pcp_is_enabled()` is a
                # run-level constant, so baking this branch at trace time is correct
                # (non-PCP / plugins keep discarding the return, graph unchanged).
                hidden_states_or_q_c = idx_ret

        return self.mla_attn(
            hidden_states_or_q_c,
            kv_c_normed,
            k_pe,
            positions,
            hidden_states_or_q_c_scale,
        )


class DeepseekV2DecoderLayer(nn.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        prefix: str,
        cache_config: str = "bf16",
        quant_config: Optional[QuantizationConfig] = None,
        layer_num: int = 0,
        is_mtp_block: bool = False,
        alt_stream: Optional[torch.cuda.Stream] = None,
        use_indexer_wk_weights_proj_fusion: Optional[bool] = None,
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        max_position_embeddings = getattr(config, "max_position_embeddings", 8192)
        # DecoderLayers are created with `make_layers` which passes the prefix
        # with the layer's index.
        layer_idx = int(prefix.split(sep=".")[-1])
        self.layer_idx = layer_idx
        self.quant_dtype = None
        self.input_norm_quant_type = None

        self.self_attn = DeepseekV2MLAAttention(
            config=config,
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            qk_nope_head_dim=config.qk_nope_head_dim,
            qk_rope_head_dim=config.qk_rope_head_dim,
            v_head_dim=config.v_head_dim,
            q_lora_rank=config.q_lora_rank if hasattr(config, "q_lora_rank") else None,
            kv_lora_rank=config.kv_lora_rank,
            max_position_embeddings=max_position_embeddings,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.self_attn",
            layer_num=layer_num,
            use_indexer_wk_weights_proj_fusion=use_indexer_wk_weights_proj_fusion,
        )

        # Keep input RMSNorm quant fusion narrow: non-Triton FP8 activation quant is only supported for per-token layouts.
        # Block/group FP8 would hit aiter's dynamic_per_group_scaled_quant FP8 path, which is not implemented.
        # The non-Triton FP4 path is only enabled for the pure global MXFP4 DeepSeek v2 checkpoint layout.
        # Because AR_RMS and RMS_Quant cannot co-exist for input_layernorm, this block of codes ensures 3 things when ATOM_ENABLE_DS_INPUT_RMSNORM_QUANT_FUSION is turned on:
        #   1. RMS_Quant fusion is only used for input_layernorm
        #   2. The reduce_results variable is re-enabled for feed forward layers (MOE and MLP), because AR_RMS is now disabled in the beginning of the next layer
        #   3. AR_RMS is turned off for input_layernorm but still enabled for post_attention_layernorm if ENABLE_ALLREDUCE_RMSNORM_FUSION is turned on
        attn_input_proj_name = (
            "fused_qkv_a_proj"
            if getattr(config, "q_lora_rank", None) is not None
            else "q_proj"
        )
        if quant_config is not None:
            attn_input_layer_name = f"{prefix}.self_attn.{attn_input_proj_name}"
            attn_input_quant_config = quant_config.get_layer_quant_config(
                attn_input_layer_name
            )

            def uses_quantized_attn_input(layer_quant_config):
                return (
                    layer_quant_config.quant_type != QuantType.No
                    and layer_quant_config.quant_dtype in (dtypes.fp8, dtypes.fp4x2)
                )

            # Consult the online override whenever it actually applies (same rule
            # as should_skip_online_quant, which governs the real weight quant in
            # the input_norm_fused_quant block below) — not only when the base is
            # unquantized. A block-scale FP8 base that ptpc-online overrides to
            # per_Token must report per_Token here; otherwise quant_dtype /
            # input_norm_quant_type disagree with the actual runtime quant.
            if quant_config.online_quant:
                online_cfg = quant_config.get_layer_quant_config(
                    attn_input_layer_name,
                    use_online_quant=True,
                )
                if not should_skip_online_quant(
                    attn_input_quant_config.quant_type,
                    attn_input_quant_config.quant_dtype,
                    online_cfg,
                ):
                    attn_input_quant_config = online_cfg

            if uses_quantized_attn_input(attn_input_quant_config):
                self.quant_dtype = attn_input_quant_config.quant_dtype
                self.input_norm_quant_type = attn_input_quant_config.quant_type.value
        self.fuse_input_norm_quant = False
        self.fuse_ar_input_norm = ENABLE_ALLREDUCE_RMSNORM_FUSION
        # DSA models (e.g., GLM-5/DeepSeek-V3.2): the indexer's wk/weights_proj GEMMs
        # run in BF16 and consume the same normed activation, so the RMSNorm(+quant)
        # must also emit the pre-quant bf16 mirror. Gate the mirror on this layer
        # actually owning an active indexer (shared / skip_topk layers don't).
        is_v32 = getattr(self.self_attn, "is_v32", False)
        emit_bf16_for_indexer = (
            is_v32
            and getattr(self.self_attn, "indexer", None) is not None
            and not getattr(self.self_attn, "skip_topk", False)
        )
        self.emit_bf16_for_indexer = emit_bf16_for_indexer
        if quant_config is not None and ENABLE_DS_INPUT_RMSNORM_QUANT_FUSION:
            enable_fp8_input_norm_quant = self.quant_dtype == dtypes.fp8 and (
                use_triton_gemm()
                or self.input_norm_quant_type == QuantType.per_Token.value
            )
            enable_fp4_input_norm_quant = self.quant_dtype == dtypes.fp4x2 and (
                use_triton_gemm()
                or _enable_non_triton_global_mxfp4_input_norm_quant(
                    config,
                    quant_config,
                    self.quant_dtype,
                    is_mtp_block,
                )
            )
            self.fuse_input_norm_quant = (
                enable_fp8_input_norm_quant or enable_fp4_input_norm_quant
            )
        # When both AR fusion and quant fusion are on they can't co-exist on one
        # input_layernorm.
        if self.fuse_input_norm_quant and self.fuse_ar_input_norm:
            if self.layer_idx == 0:
                logger.info(
                    "Warning: Both ENABLE_ALLREDUCE_RMSNORM_FUSION and ENABLE_DS_INPUT_RMSNORM_QUANT_FUSION are enabled, INPUT_RMSNORM_QUANT_FUSION is applied on layer 0, ALLREDUCE_RMSNORM_FUSION (with possible quant fusion) is applied on other layers."
                )
            else:
                self.fuse_input_norm_quant = False

        if (
            config.n_routed_experts is not None
            and layer_idx >= config.first_k_dense_replace
            and layer_idx % config.moe_layer_freq == 0
        ):
            self.mlp = DeepseekV2MoE(
                config=config,
                quant_config=quant_config,
                reduce_results=not self.fuse_ar_input_norm,
                prefix=f"{prefix}.mlp",
                alt_stream=alt_stream,
            )
        else:
            self.mlp = DeepseekV2MLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                reduce_results=not self.fuse_ar_input_norm,
                prefix=f"{prefix}.mlp",
            )
        # Fuse activation quant into the AR+RMSNorm when the attention input
        # projection is per-1x128/per-token FP8, so the GEMM consumes the
        # (fp8, scale) directly. self.quant_dtype / self.input_norm_quant_type
        # were already resolved above (including the online-quant override).
        # Restricted to the fused_qkv_a_proj MLA path: on the q_proj path the
        # normed output also feeds kv_a_proj_with_mqa, whose quant isn't checked
        # here, so fusing there could emit the wrong dtype for that GEMM.
        input_norm_fused_quant = (
            attn_input_proj_name == "fused_qkv_a_proj"
            and self.quant_dtype == dtypes.fp8
            and self.input_norm_quant_type
            in (QuantType.per_1x128.value, QuantType.per_Token.value)
        )
        fused_allreduce = (
            self.fuse_ar_input_norm and self.layer_idx > 0 and not is_mtp_block
        )
        self.input_layernorm = RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
            fused_allreduce=fused_allreduce,
            fused_quant=fused_allreduce and input_norm_fused_quant,
            fused_quant_emit_bf16=(
                fused_allreduce and input_norm_fused_quant and emit_bf16_for_indexer
            ),
            quant_config=quant_config,
            prefix=f"{prefix}.self_attn.{attn_input_proj_name}",
        )
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
            fused_allreduce=ENABLE_ALLREDUCE_RMSNORM_FUSION,
        )
        self.routed_scaling_factor = config.routed_scaling_factor

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: Optional[torch.Tensor],
    ) -> torch.Tensor:
        # Self Attention
        if self.fuse_input_norm_quant:
            assert self.quant_dtype is not None
            weight = self.input_layernorm.weight
            eps = self.input_layernorm.eps
            if self.quant_dtype == dtypes.fp4x2:
                shuffle_input_norm_quant, scale_shuffle_padding = (
                    _mxfp4_activation_quant_layout(hidden_states.shape[0])
                )
            else:
                shuffle_input_norm_quant = True
                scale_shuffle_padding = True
            if residual is None:
                residual = hidden_states
                (
                    (hidden_states_quant, hidden_states_quant_scale),
                    hidden_states_bf16,
                    _,
                    _,
                ) = _fuse_rmsnorm_quant(
                    hidden_states,
                    weight,
                    eps,
                    None,
                    None,
                    None,
                    None,
                    dtype_quant=self.quant_dtype,
                    shuffle=shuffle_input_norm_quant,
                    scale_shuffle_padding=scale_shuffle_padding,
                    group_size=128,
                    quant_type=self.input_norm_quant_type,
                    output_unquantized_inp1=self.emit_bf16_for_indexer,
                    transpose_scale=True,
                )
            else:
                (
                    (hidden_states_quant, hidden_states_quant_scale),
                    hidden_states_bf16,
                    _,
                    residual,
                ) = _fuse_rmsnorm_quant(
                    hidden_states,
                    weight,
                    eps,
                    None,
                    None,
                    None,
                    residual,
                    dtype_quant=self.quant_dtype,
                    shuffle=shuffle_input_norm_quant,
                    scale_shuffle_padding=scale_shuffle_padding,
                    group_size=128,
                    quant_type=self.input_norm_quant_type,
                    output_unquantized_inp1=self.emit_bf16_for_indexer,
                    transpose_scale=True,
                )

            # v32 indexer layers: pass the bf16 mirror as the 3rd tuple slot so the
            # indexer's BF16 wk/weights_proj GEMMs get bf16, while qkv proj gets fp8.
            if self.emit_bf16_for_indexer:
                hidden_states = (
                    hidden_states_quant,
                    hidden_states_quant_scale,
                    hidden_states_bf16,
                )
            else:
                hidden_states = (hidden_states_quant, hidden_states_quant_scale)

        else:
            if residual is None:
                residual = hidden_states
                hidden_states = self.input_layernorm(hidden_states)
            else:
                hidden_states, residual = self.input_layernorm(hidden_states, residual)

        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
        )

        if hidden_states.dtype == torch.float16:
            # Fix FP16 overflow
            # We scale both hidden_states and residual before
            # rmsnorm, and rmsnorm result would not affect by scale.
            hidden_states *= 1.0 / self.routed_scaling_factor
            if self.layer_idx == 0:
                # The residual is shared by all layers, we only scale it on
                # first layer.
                residual *= 1.0 / self.routed_scaling_factor

        # Fully Connected
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)

        if isinstance(self.mlp, DeepseekV2MLP) and hidden_states.dtype == torch.float16:
            # Fix FP16 overflow
            # Scaling the DeepseekV2MLP output, it is the input of
            # input_layernorm of next decoder layer.
            # The scaling of DeepseekV2MOE output would be done in the forward
            # of DeepseekV2MOE
            hidden_states *= 1.0 / self.routed_scaling_factor

        return hidden_states, residual


def use_replicated_vocab_embed(config: PretrainedConfig) -> bool:
    """Whether to hold the full vocab embedding on every TP rank (local lookup,
    no post-embedding all-reduce) instead of a ``VocabParallelEmbedding`` shard.

    Enabled by default (gated by ``ATOM_REPLICATE_VOCAB_EMBED``) for GLM-5.2
    (``glm_moe_dsa``) — both the main model and its MTP draft — whose embedding is
    independent of the still TP-sharded ``lm_head`` (``tie_word_embeddings=False``),
    so the lookup is bit-identical to the sharded masked-embedding + all-reduce
    path.

    Enabled under the **vLLM** plugin as well: its MTP proposer unconditionally
    shares the *target* model's ``embed_tokens`` into the draft
    (``llm_base_proposer._maybe_share_embeddings``), so replicating the main
    model's table also removes the per-step all-reduce from the draft rollout —
    and the plugin loader honours each param's ``weight_loader`` so every rank
    loads the full (un-sharded) table. Left on the sharded path for the SGLang/RTP
    plugins (their embedding lifecycle is not verified here) and whenever the
    embedding is tied to the sharded head.

    The GLM-5.2 main model keeps ``model_type == "glm_moe_dsa"``; its MTP draft
    config has ``model_type`` rewritten to ``"deepseek_mtp"`` (see
    ``SpeculativeConfig.hf_config_override``) but still carries the GLM-only
    ``index_share_for_mtp_iteration`` flag, so we detect either.
    """
    if not envs.ATOM_REPLICATE_VOCAB_EMBED:
        return False
    if getattr(config, "tie_word_embeddings", False):
        return False
    return getattr(config, "model_type", None) == "glm_moe_dsa" or bool(
        getattr(config, "index_share_for_mtp_iteration", False)
    )


@support_torch_compile
class DeepseekV2Model(nn.Module):
    def __init__(
        self,
        atom_config: Config,
        prefix: str = "",
        layer_type: type[nn.Module] = DeepseekV2DecoderLayer,
        use_indexer_wk_weights_proj_fusion: Optional[bool] = None,
    ):
        super().__init__()

        config = atom_config.hf_config
        cache_config = atom_config.kv_cache_dtype
        quant_config = atom_config.quant_config
        self.config = config

        self.vocab_size = config.vocab_size
        self.is_v32 = hasattr(config, "index_topk")

        if get_pp_group().is_first_rank:
            if use_replicated_vocab_embed(config):
                # GLM-5.2: full table per rank, no post-embedding all-reduce.
                self.embed_tokens = ReplicatedEmbedding(
                    config.vocab_size,
                    config.hidden_size,
                )
                logger.info(
                    "vocab embedding: REPLICATED (full %d-row table per rank, "
                    "no post-embed all-reduce)",
                    config.vocab_size,
                )
            else:
                self.embed_tokens = VocabParallelEmbedding(
                    config.vocab_size,
                    config.hidden_size,
                )
        else:
            self.embed_tokens = PPMissingLayer()

        self.alt_stream: Optional[torch.cuda.Stream] = None
        if getattr(config, "n_shared_experts", None) is not None:
            self.alt_stream = torch.cuda.Stream()

        _alt_stream = self.alt_stream
        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix, layer_num=None: DeepseekV2DecoderLayer(
                config,
                prefix,
                cache_config=cache_config,
                quant_config=quant_config,
                layer_num=layer_num,
                alt_stream=_alt_stream,
                use_indexer_wk_weights_proj_fusion=use_indexer_wk_weights_proj_fusion,
            ),
            prefix=f"{prefix}.layers",
            layer_num_offset=0,
        )

        # fused_allreduce will have to be turned off here if the fuse_ar_input_norm variable is False in the last layer
        if get_pp_group().is_last_rank:
            self.norm = RMSNorm(
                config.hidden_size,
                eps=config.rms_norm_eps,
                fused_allreduce=self.layers[self.end_layer - 1].fuse_ar_input_norm,
            )
        else:
            self.norm = PPMissingLayer()
        self.aux_hidden_state_layers: tuple[int, ...] = tuple()

        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states", "residual"], config.hidden_size
        )

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors],
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> Union[
        torch.Tensor, IntermediateTensors, Tuple[torch.Tensor, list[torch.Tensor]]
    ]:
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.get_input_embeddings(input_ids)
            residual = None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]

        aux_hidden_states = []
        for idx in range(self.start_layer, self.end_layer):
            layer = self.layers[idx]
            if idx in self.aux_hidden_state_layers:
                aux_hidden_states.append(
                    hidden_states if residual is None else hidden_states + residual
                )
            hidden_states, residual = layer(positions, hidden_states, residual)

        if not get_pp_group().is_last_rank:
            return IntermediateTensors(
                {"hidden_states": hidden_states, "residual": residual}
            )

        hidden_states, _ = self.norm(hidden_states, residual)

        if aux_hidden_states:
            return hidden_states, aux_hidden_states
        return hidden_states

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        # Params for weights, fp8 weight scales, fp8 activation scales
        # (param_name, weight_name, expert_id, shard_id)
        return FusedMoE.make_expert_params_mapping(
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=self.config.n_routed_experts
            + (self.config.n_shared_experts or 0),
        )


class DeepseekV2ForCausalLM(nn.Module):
    def __init__(
        self,
        atom_config: Config,
        prefix: str = "",
        layer_type: type[nn.Module] = DeepseekV2DecoderLayer,
    ):
        super().__init__()
        config = atom_config.hf_config
        quant_config = atom_config.quant_config
        self.config = config
        self.quant_config = quant_config

        model_prefix = maybe_prefix(prefix, "model")
        attn_module_list_cfg = getattr(config, "attn_module_list_cfg", None)
        indexer_prefixes = []
        if isinstance(attn_module_list_cfg, (list, tuple)):
            indexer_prefixes = [
                f"{model_prefix}.layers.{layer_idx}.self_attn.indexer"
                for layer_idx, layer_cfg in enumerate(attn_module_list_cfg)
                if isinstance(layer_cfg, dict)
                and layer_cfg.get("attn_index") is not None
            ]
        if not indexer_prefixes:
            indexer_prefixes = [f"{model_prefix}.layers.0.self_attn.indexer"]
        use_indexer_wk_weights_proj_fusion = _can_fuse_indexer_wk_weights_proj(
            config,
            quant_config,
            indexer_prefixes,
        )
        if hasattr(config, "q_lora_rank") and config.q_lora_rank is not None:
            self.packed_modules_mapping = {
                "q_a_proj": ("fused_qkv_a_proj", 0),
                "kv_a_proj_with_mqa": ("fused_qkv_a_proj", 1),
                "gate_proj": ("gate_up_proj", 0),
                "up_proj": ("gate_up_proj", 1),
            }
        else:
            self.packed_modules_mapping = {
                "gate_proj": ("gate_up_proj", 0),
                "up_proj": ("gate_up_proj", 1),
            }
        if use_indexer_wk_weights_proj_fusion:
            self.packed_modules_mapping.update(
                {
                    "indexer.wk": ("indexer.wk_weights_proj", 0),
                    "indexer.weights_proj": ("indexer.wk_weights_proj", 1),
                }
            )

        self.model = DeepseekV2Model(
            atom_config=atom_config,
            prefix=model_prefix,
            layer_type=layer_type,
            use_indexer_wk_weights_proj_fusion=use_indexer_wk_weights_proj_fusion,
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

        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors
        )

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.get_input_embeddings(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        # ---- Prefill Context Parallel (PCP) query split ------------------
        # During prefill with pcp_size > 1 the token sequence is round-robin
        # split so each PCP rank runs the whole model (embed / norm / q-proj /
        # MoE) on only 1/pcp of the tokens. The attention modules re-materialise
        # the full KV internally (see DeepseekV2MLAAttention / Indexer /
        # MLAAttention), so decode and the cache layout are untouched. When PCP
        # is inactive (`pcp=1` or decode) this whole block is skipped and the
        # forward is identical to the original path.
        pcp = _pcp_active()
        n_global = positions.shape[0]
        if pcp:
            pcp_ws = get_pcp_world_size()
            n_pad = pcp_pad_len(n_global, pcp_ws) - n_global
            positions = pcp_round_robin_split(pcp_pad_dense(positions, n_pad), pcp_ws)
            if input_ids is not None:
                input_ids = pcp_round_robin_split(
                    pcp_pad_dense(input_ids, n_pad), pcp_ws
                )
            if inputs_embeds is not None:
                inputs_embeds = pcp_round_robin_split(
                    pcp_pad_dense(inputs_embeds, n_pad), pcp_ws
                )

        hidden_states = self.model(
            input_ids, positions, intermediate_tensors, inputs_embeds
        )

        # ---- PCP gather: 1/pcp rows -> full token order, then drop pad ----
        # Only the last PP rank produces real hidden states; earlier stages
        # forward IntermediateTensors (already 1/pcp) unchanged.
        if pcp and get_pp_group().is_last_rank:
            if isinstance(hidden_states, tuple):
                hs, aux = hidden_states
                hs = pcp_allgather_rerange(hs, pcp_ws)[:n_global]
                aux = [pcp_allgather_rerange(a, pcp_ws)[:n_global] for a in aux]
                hidden_states = (hs, aux)
            elif isinstance(hidden_states, torch.Tensor):
                hidden_states = pcp_allgather_rerange(hidden_states, pcp_ws)[:n_global]
        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        logits = self.lm_head(hidden_states)
        return logits

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

    def set_aux_hidden_state_layers(self, layers: tuple[int, ...]) -> None:
        self.model.aux_hidden_state_layers = layers

    def get_eagle3_aux_hidden_state_layers(self) -> tuple[int, ...]:
        """Default Eagle3 aux hidden-state layer ids: early / middle / late
        of the target model. Aligned with vLLM's default (see
        vllm/model_executor/models/deepseek_v2.py).
        """
        num_layers = len(self.model.layers)
        return (2, num_layers // 2, num_layers - 3)

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        return self.model.get_expert_mapping()


class DeepseekV3ForCausalLM(DeepseekV2ForCausalLM):
    # DeepSeek-V3.2's indexer weights projection is BF16.  Keep the original
    # checkpoint path and the fused ATOM path excluded from default quantization.
    quant_default_exclude_layers: list[str] = [
        "*.indexer.weights_proj",
        "*.indexer.wk_weights_proj",
    ]


class GlmMoeDsaForCausalLM(DeepseekV2ForCausalLM):
    """GLM 5.0 MoE (structurally similar to DeepSeek v3.2). Reuses DeepseekV2 implementation."""

    # GLM-5's HF quant config uses `indexers_proj` in modules_to_not_convert, but
    # the unfused ATOM module path is `indexer.weights_proj`.  Keep that path
    # excluded so FP4/MXFP4 fallback does not quantize the BF16 projection.
    quant_exclude_name_mapping: dict[str, str] = {
        # HF quant config uses "indexers_proj" but the ATOM module path is
        # "indexer.weights_proj".  str.replace translates each exclude entry.
        "indexers_proj": "indexer.weights_proj",
    }
