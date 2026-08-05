# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import logging
import os
from dataclasses import dataclass
from functools import partial as functools_partial
from typing import Optional, Protocol

import torch
import triton
import triton.language as tl
from aiter import (
    QuantType,
    concat_and_cache_mla,
    dtypes,
    flash_attn_varlen_func,
    fused_qk_rope_concat_and_cache_mla,
    get_hip_quant,
)

# The segmented (page_size>1) MLA cache kernels only exist in newer aiter
# builds. Import them lazily so that the default page_size=1 path keeps working
# on aiter versions that do not ship the seg variants.
try:
    from aiter import (
        concat_and_cache_mla_seg,
        fused_qk_rope_concat_and_cache_mla_seg,
    )
except ImportError:
    concat_and_cache_mla_seg = None
    fused_qk_rope_concat_and_cache_mla_seg = None
from aiter.dist.parallel_state import get_dp_group
from aiter.mla import mla_decode_fwd, mla_prefill_fwd
from aiter.ops.triton.attention.mla import (
    mla_decode_fwd as triton_shuffle_mla_decode_fwd,
)
from aiter.ops.triton.fusions.fused_kv_cache import (
    fused_qk_rope_cat_and_cache_mla as triton_fused_qk_rope_cat_and_cache_mla,
)
from aiter.ops.triton.gather_kv_b_proj import gather_kv_b_proj
from aiter.ops.triton.kv_cache import cat_and_cache_mla as triton_cat_and_cache_mla
from torch import nn

from atom.config import get_current_atom_config
from atom.distributed.dcp_utils import (
    dcp_persistent_supported,
    get_dcp_group,
    get_dcp_rank,
    get_dcp_world_size,
)
from atom.distributed.pcp_utils import (
    get_pcp_world_size,
    pcp_allgather_rerange,
    pcp_is_enabled,
)
from atom.model_ops.linear import use_triton_gemm
from atom.model_ops.utils import get_and_maybe_dequant_weights
from atom.utils import envs
from atom.utils.decorators import mark_trace
from atom.utils.forward_context import (
    AttentionMetaData,
    ForwardContext,
    get_forward_context,
)

from aiter.ops.triton.batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant import (  # isort: skip
    batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant as _aiter_triton_fp8_bmm,
)

concat_and_cache_mla = mark_trace(
    concat_and_cache_mla, prefix="kv_cache", torch_compile=False
)
if concat_and_cache_mla_seg is not None:
    concat_and_cache_mla_seg = mark_trace(
        concat_and_cache_mla_seg, prefix="kv_cache_seg", torch_compile=False
    )
fused_qk_rope_concat_and_cache_mla = mark_trace(
    fused_qk_rope_concat_and_cache_mla, prefix="rope_and_kv_cache", torch_compile=False
)
if fused_qk_rope_concat_and_cache_mla_seg is not None:
    fused_qk_rope_concat_and_cache_mla_seg = mark_trace(
        fused_qk_rope_concat_and_cache_mla_seg,
        prefix="rope_and_kv_cache",
        torch_compile=False,
    )
mla_prefill_fwd = mark_trace(mla_prefill_fwd, prefix="mla_prefill", torch_compile=False)
mla_decode_fwd = mark_trace(mla_decode_fwd, prefix="mla_decode", torch_compile=False)

# Shuffled-KV (block_size=64) Triton/Gluon MLA kernels, gated by
# ATOM_USE_TRITON_MLA and ATOM_USE_TRITON_MLA_SHUFFLE_KV:. Write kernels mirror the aiter
# concat_and_cache / fused_qk_rope_concat_and_cache_mla but store the cache in
# the shuffled layout the shuffled decode kernel reads back.
triton_shuffle_mla_decode_fwd = mark_trace(
    triton_shuffle_mla_decode_fwd, prefix="mla_decode_shuffle", torch_compile=False
)
triton_cat_and_cache_mla = mark_trace(
    triton_cat_and_cache_mla, prefix="kv_cache_shuffle", torch_compile=False
)
triton_fused_qk_rope_cat_and_cache_mla = mark_trace(
    triton_fused_qk_rope_cat_and_cache_mla,
    prefix="rope_and_kv_cache_shuffle",
    torch_compile=False,
)

# torch.set_printoptions(threshold=10_000)

logger = logging.getLogger("atom")

_MLA_MIN_HEADS = 16  # AITER MLA kernels require at least 16 attention heads

# The fused seg MLA kernels (fused_qk_rope_concat_and_cache_mla_seg +
# concat_and_cache_mla_seg + the gfx1250 mla_decode_fwd asm) share a single
# segmented KV cache layout (all tokens' nope packed first, then all tokens'
# pe) and a fixed page size hard-coded in the kernels.
_MLA_SEG_PAGE_SIZE = 64
# The gfx1250 decode asm consumes an fp8 Q whose per-head row stride is padded
# to 768 bytes (poc_kl pack_q_page1_padded layout). q_out is allocated with this
# padded last dim and sliced to the logical kv_lora_rank + qk_rope_head_dim
# columns; the padding tail is never read by the decode kernel.
_MLA_Q_OUT_PADDED_DIM = 768
# Dims the fused seg kernels are compiled against (KV_LORA / PE_DIM constexprs).
_MLA_SEG_KV_LORA_RANK = 512
_MLA_SEG_PE_DIM = 64

# DIAGNOSTIC (DCP/DCP_Sparse_MLA.md §5.5): set ATOM_DSA_DCP_PREFILL_MERGE=0 to run
# the DCP sparse prefill WITHOUT the cross-rank merge (P3) while keeping the
# indexer gather (P1) and the compact filter (P2). The output is wrong by
# construction -- each rank keeps only its own partial softmax -- but it isolates
# a fault in the compacted indices from a fault in the q all-gather / LSE merge.
_DSA_DCP_PREFILL_MERGE = os.environ.get("ATOM_DSA_DCP_PREFILL_MERGE", "1") == "1"

# DIAGNOSTIC: ATOM_DSA_ATTN_DUMP=<dir>[:<l0,l1,...>] -- dump the prefill attention
# output (post o_proj) of the listed layers, for the first two prefill calls of
# the process. One run therefore yields BOTH a depth profile and a same-input
# repeat, which together separate "the DCP sparse-prefill math is wrong" from
# "something is nondeterministic" and show where either starts. Layers default
# to just 0.
_DSA_ATTN_DUMP_SPEC = os.environ.get("ATOM_DSA_ATTN_DUMP", "")
_DSA_ATTN_DUMP = _DSA_ATTN_DUMP_SPEC.split(":")[0] if _DSA_ATTN_DUMP_SPEC else ""
_DSA_ATTN_DUMP_LAYERS = frozenset(
    int(x) for x in _DSA_ATTN_DUMP_SPEC.split(":")[1].split(",")
) if ":" in _DSA_ATTN_DUMP_SPEC else frozenset({0})
# one counter per layer: how many prefill calls of that layer we have dumped
_dsa_attn_dump_count = {}

# DIAGNOSTIC: ATOM_DSA_DCP_MERGE_FP32=1 -- run the DCP prefill merge in fp32.
_DSA_DCP_MERGE_FP32 = os.environ.get("ATOM_DSA_DCP_MERGE_FP32", "0") == "1"

# DIAGNOSTIC: ATOM_DSA_FORCE_DECODE_KERNEL=1 -- use mla_decode_fwd for sparse
# prefill even when no LSE is needed. dcp=1 otherwise runs mla_prefill_fwd while
# dcp>1 runs mla_decode_fwd, so a dcp=1 vs dcp=N comparison confounds "DCP is
# wrong" with "the two kernels differ numerically". Setting this on the dcp=1
# reference removes that confound.
_DSA_FORCE_DECODE_KERNEL = (
    os.environ.get("ATOM_DSA_FORCE_DECODE_KERNEL", "0") == "1"
)

if False:
    try:
        from aiter.ops.triton.fused_gemm_a8w8_blockscale_split_cat import (
            fused_gemm_a8w8_blockscale_preshuffle_split_cat,
        )
        from aiter.ops.triton.fused_gemm_afp4wfp4_split_cat import (
            fused_gemm_afp4wfp4_preshuffle_split_cat,
        )
    except ImportError as e:
        logger.warning(f"Triton fused GEMM split_cat not available: {e}")
        fused_gemm_afp4wfp4_preshuffle_split_cat = None
        fused_gemm_a8w8_blockscale_preshuffle_split_cat = None
fused_gemm_afp4wfp4_preshuffle_split_cat = None
fused_gemm_a8w8_blockscale_preshuffle_split_cat = None


def is_rocm_aiter_fp4bmm_enabled() -> bool:
    return envs.ATOM_USE_TRITON_MXFP4_BMM


def _maybe_view_mxfp4_weight_for_gather(
    kv_b_proj: nn.Module, weight: torch.Tensor
) -> torch.Tensor:
    fp4_dtype = getattr(torch, "float4_e2m1fn_x2", None)
    if fp4_dtype is None or weight.dtype != torch.uint8:
        return weight

    layer_quant_config = getattr(kv_b_proj, "layer_quant_config", None)
    is_mxfp4 = getattr(kv_b_proj, "params_dtype", None) == dtypes.fp4x2 or (
        layer_quant_config is not None
        and getattr(layer_quant_config, "quant_dtype", None) == dtypes.fp4x2
    )
    if is_mxfp4:
        return weight.view(fp4_dtype)
    return weight


if is_rocm_aiter_fp4bmm_enabled():
    # from aiter.ops.triton.batched_gemm_afp4wfp4_pre_quant import  batched_gemm_afp4wfp4_pre_quant
    from aiter.ops.triton.batched_gemm_a16wfp4 import batched_gemm_a16wfp4

    from atom.model_ops.utils import quark_post_load_weights


# MLA Specific Arguments
@dataclass
class MLAModules:
    """Modules used in MLA."""

    q_lora_rank: Optional[int]
    kv_lora_rank: int
    qk_nope_head_dim: int
    qk_rope_head_dim: int
    qk_head_dim: int
    v_head_dim: int
    rotary_emb: torch.nn.Module
    q_proj: Optional[torch.nn.Module]
    kv_b_proj: torch.nn.Module
    o_proj: torch.nn.Module
    indexer: Optional[torch.nn.Module]
    # Model-level sparse flag. A v3.2 / GLM-5.2 model runs sparse MLA on ALL its
    # layers. GLM-5.2 IndexShare "shared" layers carry no indexer module yet must
    # still run sparse attention (reusing the prior "full" layer's top-k), so
    # sparsity must be derived from the model, not from whether this layer owns
    # an indexer. Defaults keep non-sparse models unchanged.
    is_sparse: bool = False
    topk_tokens: int | None = None


class _MLAOutputShape(Protocol):
    o_proj: nn.Module
    num_heads: int
    v_head_dim: int


def _mla_output_width(impl: _MLAOutputShape, hidden_size: int) -> int:
    if isinstance(getattr(impl, "o_proj", None), nn.Identity):
        return impl.num_heads * impl.v_head_dim
    return hidden_size


def dynamic_per_batched_tensor_quant(
    x: torch.Tensor, dtype: torch.dtype = torch.float8_e4m3fn
):
    DTYPE_MAX = torch.finfo(dtype).max
    min_val, max_val = x.aminmax()
    amax = torch.maximum(min_val.abs(), max_val.abs()).clamp(min=1e-10)
    scale = DTYPE_MAX / amax
    x_scl_sat = (x * scale).clamp(min=-DTYPE_MAX, max=DTYPE_MAX)
    return x_scl_sat.to(dtype).contiguous(), scale.float().reciprocal()


class MLAAttention(nn.Module):
    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        scale: float,
        num_kv_heads: int,
        kv_cache_dtype: str,
        layer_num: int = 0,
        mla_modules: MLAModules = None,
        dtype: torch.dtype = torch.bfloat16,
        **kwargs,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = float(scale)
        self.num_kv_heads = num_kv_heads
        self.kv_cache_dtype = "fp8" if kv_cache_dtype.startswith("fp8") else "auto"
        self.dtype = dtype

        self.padded_num_heads = max(num_heads, _MLA_MIN_HEADS)
        self.head_repeat_factor = 1
        self.head_pad = 0
        if self.padded_num_heads != num_heads:
            if self.padded_num_heads % num_heads == 0:
                self.head_repeat_factor = self.padded_num_heads // num_heads
                if not getattr(MLAAttention, "_head_repeat_logged", False):
                    MLAAttention._head_repeat_logged = True
                    logger.info(
                        f"MLA head repeat enabled: {num_heads} -> {self.padded_num_heads} "
                        f"(repeat factor {self.head_repeat_factor})"
                    )
            else:
                self.head_pad = self.padded_num_heads - num_heads

        self.q_lora_rank = mla_modules.q_lora_rank
        self.kv_lora_rank = mla_modules.kv_lora_rank
        self.qk_nope_head_dim = mla_modules.qk_nope_head_dim
        self.qk_rope_head_dim = mla_modules.qk_rope_head_dim
        self.qk_head_dim = mla_modules.qk_head_dim
        self.v_head_dim = mla_modules.v_head_dim
        self.rotary_emb = mla_modules.rotary_emb
        self.q_proj = mla_modules.q_proj
        self.o_proj = mla_modules.o_proj
        self.kv_b_proj = mla_modules.kv_b_proj
        self.kv_cache = torch.tensor([])
        self.one_scale = torch.tensor(1.0, dtype=torch.float32)
        self._k_scale = self.one_scale
        self._q_scale = self.one_scale
        # Derive sparsity from the model-level flag, not from whether THIS layer
        # owns an indexer: GLM-5.2 IndexShare "shared" layers have indexer=None
        # but must still run sparse MLA, reusing the prior "full" layer's top-k.
        # (`mla_modules.is_sparse` defaults False, so non-sparse models and the
        # `indexer is not None` fallback keep their previous behavior.)
        self.is_sparse_mla = mla_modules.is_sparse or (mla_modules.indexer is not None)
        self.topk_tokens = (
            mla_modules.indexer.topk_tokens
            if mla_modules.indexer is not None
            else mla_modules.topk_tokens
        )
        # Shared layers have no indexer buffer at construction; the metadata
        # builder rebinds it to the shared `_sparse_kv_indices_gpu` at runtime,
        # so the layer reads the prior full layer's selected indices.
        self.sparse_kv_indices_buffer = (
            mla_modules.indexer.sparse_kv_indices_buffer
            if mla_modules.indexer is not None
            else None
        )
        self.layer_num = layer_num
        # When the triton MLA backend is selected we keep the original
        # interleaved KV cache layout (concat_and_cache_mla /
        # fused_qk_rope_concat_and_cache_mla) and an unpadded 576-wide q_out;
        # only the gfx1250 asm decode path needs the segmented layout + 768 pad.
        self.use_triton_mla = bool(envs.ATOM_USE_TRITON_MLA)
        # On the non-triton (aiter) path, ATOM_MLA_PAGE_SIZE selects the KV cache
        # layout: >1 uses the segmented (paged) seg kernels + padded q_out, while
        # ==1 falls back to the original interleaved per-token (page_size=1)
        # kernels with an unpadded 576-wide q_out. The triton path never uses seg.
        self.use_seg_mla = (not self.use_triton_mla) and envs.ATOM_MLA_PAGE_SIZE > 1
        if self.use_seg_mla:
            if envs.ATOM_MLA_PAGE_SIZE != _MLA_SEG_PAGE_SIZE:
                raise RuntimeError(
                    f"Segmented MLA requires ATOM_MLA_PAGE_SIZE={_MLA_SEG_PAGE_SIZE} "
                    f"(got {envs.ATOM_MLA_PAGE_SIZE})."
                )
            if get_current_atom_config().kv_cache_block_size != _MLA_SEG_PAGE_SIZE:
                raise RuntimeError(
                    f"Segmented MLA requires kv_cache_block_size={_MLA_SEG_PAGE_SIZE} "
                    f"(got {get_current_atom_config().kv_cache_block_size})."
                )
            if (
                concat_and_cache_mla_seg is None
                or fused_qk_rope_concat_and_cache_mla_seg is None
            ):
                raise RuntimeError(
                    "ATOM_MLA_PAGE_SIZE > 1 requires the segmented MLA kernels "
                    "(concat_and_cache_mla_seg / fused_qk_rope_concat_and_cache_mla_seg), "
                    "which are not available in the installed aiter build. Upgrade "
                    "aiter or set ATOM_MLA_PAGE_SIZE=1."
                )

        # Decode context parallel (DCP): KV cache is sharded across TP ranks, so
        # decode attention runs locally then combines via all-gather LSE +
        # reduce-scatter. Disabled (world_size==1) unless -dcp is set.
        self.dcp_world_size = get_dcp_world_size()
        if self.dcp_world_size > 1:
            from atom.model_ops.dcp_ops import CPTritonContext

            self.dcp_group = get_dcp_group()
            self.dcp_rank = get_dcp_rank()
            self._cp_triton_ctx = CPTritonContext()
        else:
            self.dcp_group = None
            self.dcp_rank = 0
            self._cp_triton_ctx = None

        # Whether DCP decode can run in persistent mode on this GPU (gfx950 has
        # the lse persistent kernel, gfx942 does not — see dcp_utils). Cached
        # once here to avoid a per-forward get_gfx() (graph-break).
        self.dcp_persistent_supported = dcp_persistent_supported()

        # Compacted per-layer sparse offsets for DCP decode; rebound by the
        # metadata builder to the shared buffer (see aiter_mla.py).
        self.dcp_sparse_kv_indptr_buffer = None
        self.dcp_owned_counts_buffer = None

        # DIAGNOSTIC (DCP/DCP_Sparse_MLA.md §6.4): force sparse decode onto the
        # non-persistent kernel even at dcp=1, to measure what persistent
        # actually buys for sparse decode. Read once to keep it out of forward.
        self._dsa_force_nonpersist = (
            os.environ.get("ATOM_DSA_FORCE_NONPERSIST", "0") == "1"
        )
        # DIAGNOSTIC: keep prefill dense under DCP at any prompt length.
        self._dsa_dcp_dense_prefill = (
            os.environ.get("ATOM_DSA_DCP_DENSE_PREFILL", "0") == "1"
        )

    def _pad_query_heads(self, q: torch.Tensor) -> torch.Tensor:
        if self.head_repeat_factor > 1:
            return q.repeat_interleave(self.head_repeat_factor, dim=1)
        if self.head_pad > 0:
            return torch.nn.functional.pad(q, (0, 0, 0, self.head_pad))
        return q

    def _restore_query_heads(
        self, output: torch.Tensor, num_heads: int | None = None
    ) -> torch.Tensor:
        if self.head_repeat_factor > 1:
            return output[:, :: self.head_repeat_factor, ...].contiguous()
        if self.head_pad > 0:
            return output[:, : (num_heads or self.num_heads), ...].contiguous()
        return output

    def _seg_kv_cache_view(self, kv_cache: torch.Tensor) -> torch.Tensor:
        """Reshape the KV cache buffer into the page-level flat seg layout
        ``[num_blocks, page_size*(kv_lora_rank + qk_rope_head_dim)]`` that the
        seg write kernels expect (they derive page_size from ``stride(0)``).

        The cache is allocated token-major as ``[num_blocks*page_size, ..., entry]``
        (so ``kv_cache.shape[0]`` is the total slot count, not the block count).
        A plain view groups every ``page_size`` consecutive token slots into one
        block, i.e. slot = block*page_size + offset, which matches slot_mapping
        and the page-level view used on the decode side
        (``kv_buffer.view(-1, page_size, 1, entry)``). Using
        ``kv_cache.view(kv_cache.shape[0], -1)`` here is WRONG: it keeps the
        token-level stride (entry), so the kernel derives page_size=1 and writes
        an interleaved layout that the page_size=64 decode then misreads."""
        page_size = get_current_atom_config().kv_cache_block_size
        entry = self.kv_lora_rank + self.qk_rope_head_dim
        return kv_cache.view(-1, page_size * entry)

    def process_weights_after_loading(self):
        if is_rocm_aiter_fp4bmm_enabled():
            kv_b_proj_weight = get_and_maybe_dequant_weights(self.kv_b_proj)
            self.W_K, self.W_K_scale, W_V, self.W_V_scale = quark_post_load_weights(
                self, kv_b_proj_weight, "mxfp4"
            )
            self.W_V = W_V.contiguous().transpose(1, 2)

            self.W_K = self.W_K.transpose(-2, -1).contiguous()
            self.W_K_scale = self.W_K_scale.transpose(-2, -1).contiguous()
            self.W_V = self.W_V.transpose(-2, -1).contiguous()
            self.W_V_scale = self.W_V_scale.transpose(-2, -1).contiguous()
        else:  # is_rocm_aiter_fp8bmm_enabled()
            kv_b_proj_weight = get_and_maybe_dequant_weights(self.kv_b_proj).T
            assert kv_b_proj_weight.shape == (
                self.kv_lora_rank,
                self.num_heads * (self.qk_nope_head_dim + self.v_head_dim),
            ), (
                f"{kv_b_proj_weight.shape=}, "
                f"{self.kv_lora_rank=}, "
                f"{self.num_heads=}, "
                f"{self.qk_nope_head_dim=}, "
                f"{self.v_head_dim=}"
            )
            kv_b_proj_weight = kv_b_proj_weight.view(
                self.kv_lora_rank,
                self.num_heads,
                self.qk_nope_head_dim + self.v_head_dim,
            )
            W_UK, W_UV = kv_b_proj_weight.split(
                [self.qk_nope_head_dim, self.v_head_dim], dim=-1
            )
            W_K = W_UK.transpose(0, 1)  # 16 512 128
            W_V = W_UV.permute(1, 2, 0)  # 16 128 512
            self.W_K, self.W_K_scale = dynamic_per_batched_tensor_quant(
                W_K, dtype=dtypes.fp8
            )
            self.W_V, self.W_V_scale = dynamic_per_batched_tensor_quant(
                W_V, dtype=dtypes.fp8
            )

    @mark_trace(prefix="v_up_proj_and_o_proj", torch_compile=False)
    def _v_up_proj_and_o_proj(self, x):
        # Convert from (B, N, L) to (N, B, L)
        x = x.view(-1, self.num_heads, self.kv_lora_rank).transpose(0, 1)
        # Multiply (N, B, L) x (N, L, V) -> (N, B, V), Convert from (N, B, V) to (B, N, V)
        # x = torch.bmm(x, self.W_UV).transpose(0, 1)
        # Convert from (B, N, L) to (N, B, L)
        if is_rocm_aiter_fp4bmm_enabled():
            output = torch.empty(
                x.shape[1],
                x.shape[0],
                self.W_V.shape[1],
                device=x.device,
                dtype=torch.bfloat16,
            )
            output = batched_gemm_a16wfp4(
                x,
                self.W_V,
                self.W_V_scale,
                y=output,
                transpose_bm=True,
                prequant=True,
                y_scale=None,
            )
            # x = x.transpose(0, 1).flatten(1, 2)
            output = output.view(-1, self.num_heads * self.v_head_dim)
            x = output
        else:
            x = _aiter_triton_fp8_bmm(
                x, self.W_V, self.W_V_scale, group_size=128, transpose_bm=True
            )
            # Convert from (B, N, V) to (B, N * V)
            x = x.reshape(-1, self.num_heads * self.v_head_dim)
        return self.o_proj(x)

    @mark_trace(prefix="q_proj_and_k_up_proj", torch_compile=False)
    def _q_proj_and_k_up_proj(self, x, x_scale=None):
        q_nope, q_pe = (
            self.q_proj(x, x_scale)
            .view(-1, self.num_heads, self.qk_head_dim)
            .split([self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
        )

        # Convert from (B, N, P) to (N, B, P)
        q_nope = q_nope.transpose(0, 1)

        if is_rocm_aiter_fp4bmm_enabled():
            # FP4 BMM: (N, B, P) x (N, P, L) -> (N, B, L)
            ql_nope = batched_gemm_a16wfp4(
                q_nope,
                self.W_K,
                self.W_K_scale,
                y=None,
                transpose_bm=True,
                prequant=True,
                y_scale=None,
            )
        else:
            # Multiply (N, B, P) x (N, P, L) -> (N, B, L), Convert from (N, B, L) to (B, N, L)
            # ql_nope = torch.bmm(q_nope, self.W_UK_T).transpose(0, 1)
            ql_nope = _aiter_triton_fp8_bmm(
                q_nope, self.W_K, self.W_K_scale, group_size=128, transpose_bm=True
            )
        return ql_nope, q_pe

    def fused_kv_bmm(
        self, x, x_scale, k_nope, k_rope, positions, kv_cache, attn_metadata
    ):
        q_nope, q_pe = (
            self.q_proj(x, x_scale)
            .view(-1, self.num_heads, self.qk_head_dim)
            .split([self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
        )

        q_nope = q_nope.transpose(0, 1)

        if is_rocm_aiter_fp4bmm_enabled():
            from aiter.ops.triton.fusions.fused_bmm_rope_kv_cache import (
                fused_fp4_bmm_rope_cat_and_cache_mla,
            )

            result, _, _, _ = fused_fp4_bmm_rope_cat_and_cache_mla(
                q_nope,
                self.W_K,
                self.W_K_scale,
                q_pe,
                k_nope.view(-1, self.num_kv_heads, self.kv_lora_rank),
                k_rope.view(-1, self.num_kv_heads, self.qk_rope_head_dim),
                kv_cache,
                attn_metadata.slot_mapping,
                positions,
                self.rotary_emb.cos_cache,
                self.rotary_emb.sin_cache,
                y=None,
                transpose_bm=True,
                prequant=True,
                y_scale=None,
                k_scale=self._k_scale,
                is_neox=self.rotary_emb.is_neox_style,
                q_out_dtype=kv_cache.dtype,
                num_decode_toks_for_zeros=0,
            )
        else:
            from aiter.ops.triton.fusions.fused_bmm_rope_kv_cache import (
                fused_fp8_bmm_rope_cat_and_cache_mla,
            )

            result, _, _, _ = fused_fp8_bmm_rope_cat_and_cache_mla(
                q_nope,
                self.W_K,
                self.W_K_scale,
                q_pe,
                k_nope.view(-1, self.num_kv_heads, self.kv_lora_rank),
                k_rope.view(-1, self.num_kv_heads, self.qk_rope_head_dim),
                kv_cache,
                attn_metadata.slot_mapping,
                positions,
                self.rotary_emb.cos_cache,
                self.rotary_emb.sin_cache,
                group_size=128,
                transpose_bm=True,
                k_scale=self._k_scale,
                is_neox=self.rotary_emb.is_neox_style,
                q_out_dtype=kv_cache.dtype,
                num_decode_toks_for_zeros=0,
            )

        return result

    def _forward_prefill_cached_single_pass(
        self,
        prefill_q: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: AttentionMetaData,
    ) -> torch.Tensor:
        """Legacy single-pass path: gather the full cached+new context into
        k_full / v_full and run one flash_attn. OOMs on long contexts (peak
        ≈ total_kv × heads × (qk_dim + v_dim) × dtype)."""
        k_full = torch.empty(
            (
                attn_metadata.total_kv,
                self.num_heads,
                self.qk_nope_head_dim + self.qk_rope_head_dim,
            ),
            device=prefill_q.device,
            dtype=self.dtype,
        )
        v_full = torch.empty(
            (attn_metadata.total_kv, self.num_heads, self.v_head_dim),
            device=prefill_q.device,
            dtype=self.dtype,
        )
        self._gather_cached_kv_b_proj(
            kv_cache,
            attn_metadata.kv_indptr,
            attn_metadata.kv_indices,
            attn_metadata.cu_seqlens_k,
            k_full,
            v_full,
            getattr(attn_metadata, "shuffle_kv_block_indptr", None),
            getattr(attn_metadata, "shuffle_kv_block_indices", None),
        )
        output = flash_attn_varlen_func(
            q=prefill_q,
            k=k_full,
            v=v_full,
            cu_seqlens_q=attn_metadata.cu_seqlens_q,
            cu_seqlens_k=attn_metadata.cu_seqlens_k,
            max_seqlen_q=attn_metadata.max_seqlen_q,
            max_seqlen_k=attn_metadata.max_seqlen_k,
            min_seqlen_q=attn_metadata.min_seqlen_q,
            dropout_p=attn_metadata.dropout_p,
            softmax_scale=self.scale,
            causal=True,
        )
        return self.o_proj(output.flatten(start_dim=-2))

    def _gather_cached_kv_b_proj(
        self,
        kv_cache: torch.Tensor,
        kv_indptr: torch.Tensor,
        kv_indices: torch.Tensor,
        cu_seqlens_k: torch.Tensor,
        k_out: torch.Tensor,
        v_out: torch.Tensor,
        shuffle_kv_block_indptr: Optional[torch.Tensor] = None,
        shuffle_kv_block_indices: Optional[torch.Tensor] = None,
    ) -> None:
        weight = self.kv_b_proj.weight
        if envs.ATOM_USE_TRITON_MLA and envs.ATOM_USE_TRITON_MLA_SHUFFLE_KV:
            # Shuffled KV: read the block_size-shuffled cache with block-granular
            # CSR indices built by the metadata builder. cu_seqlens_k stays the
            # token-granular context cumsum (output token positions).
            kv_buffer = self._shuffled_kv_view(kv_cache)
            gather_kv_b_proj(
                kv_buffer.squeeze(1),  # [num_blocks, block_size, kv_lora+rope]
                self._k_scale,
                shuffle_kv_block_indptr,
                shuffle_kv_block_indices,
                cu_seqlens_k,
                _maybe_view_mxfp4_weight_for_gather(self.kv_b_proj, weight),
                getattr(self.kv_b_proj, "weight_scale", None),
                k_out,
                v_out,
                weight_preshuffle=getattr(self.kv_b_proj.weight, "is_shuffled", False),
                shuffled_kv_cache=True,
            )
        else:
            gather_kv_b_proj(
                kv_cache,
                self._k_scale,
                kv_indptr,
                kv_indices,
                cu_seqlens_k,
                _maybe_view_mxfp4_weight_for_gather(self.kv_b_proj, weight),
                getattr(self.kv_b_proj, "weight_scale", None),
                k_out,
                v_out,
                weight_preshuffle=getattr(weight, "is_shuffled", False),
            )

    def _forward_prefill_cached_chunked(
        self,
        prefill_q: torch.Tensor,
        kv_c_normed_new: torch.Tensor,
        k_rope_new: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: AttentionMetaData,
        chunk_meta,
    ) -> torch.Tensor:
        """Chunked prefill for the has_cached branch.

        Pattern (mirrors atom/plugin/attention_mha.py:extend_forward): the
        cached prefix and the new tokens are attended separately and merged
        via softmax-LSE recombination. This bounds peak memory to
        ``CHUNK_TOKENS × heads × (qk_dim + v_dim)``, independent of context
        length.

        Step 1 — new-tokens self-attention (causal). New k/v come from
        kv_b_proj on the input latent kv_c_normed; cu_seqlens_k = cu_seqlens_q.
        Step 2 — per chunk c of the cached prefix: gather expanded K/V into
        the shared workspace, flash_attn(causal=False, return_lse), merge
        into a running (chunked_out, chunked_lse).
        Step 3 — final merge of (chunked_out, chunked_lse) with (new_out,
        new_lse). The cached prefix is the "prefix" side (smaller token
        positions), new tokens are the "suffix".
        """
        from atom.model_ops.attentions.triton_merge_attn_states import merge_attn_states

        # Trigger counter: log first hit + every 500th to confirm the chunked
        # path is actually exercised (not silently bypassed when
        # has_cached=True but cached prefix < CHUNK_TOKENS for every seq).
        # Counter is class-level so all layers/instances share a single count.
        n = MLAAttention._chunked_prefill_calls = (
            getattr(MLAAttention, "_chunked_prefill_calls", 0) + 1
        )
        if n == 1 or n % 500 == 0:
            logger.info(
                "MLA chunked-prefill #%d: layer=%d num_chunks=%d "
                "total_kv=%s cu_seqlens_q[-1]=%d",
                n,
                self.layer_num,
                chunk_meta.num_chunks,
                attn_metadata.total_kv,
                int(attn_metadata.cu_seqlens_q[-1].item()),
            )

        # Step 1: new-tokens self-attn via kv_b_proj on the latent.
        if k_rope_new.dim() == 2:
            k_rope_new = k_rope_new.unsqueeze(1)
        kv_nope_new = self.kv_b_proj(kv_c_normed_new).view(
            -1, self.num_heads, self.qk_nope_head_dim + self.v_head_dim
        )
        k_nope_new, v_new = kv_nope_new.split(
            [self.qk_nope_head_dim, self.v_head_dim], dim=-1
        )
        k_new = torch.cat(
            (k_nope_new, k_rope_new.expand((*k_nope_new.shape[:-1], -1))), dim=-1
        )
        new_out, new_lse = flash_attn_varlen_func(
            q=prefill_q,
            k=k_new,
            v=v_new,
            cu_seqlens_q=attn_metadata.cu_seqlens_q,
            cu_seqlens_k=attn_metadata.cu_seqlens_q,
            max_seqlen_q=attn_metadata.max_seqlen_q,
            max_seqlen_k=attn_metadata.max_seqlen_q,
            min_seqlen_q=attn_metadata.min_seqlen_q,
            dropout_p=attn_metadata.dropout_p,
            softmax_scale=self.scale,
            causal=True,
            return_lse=True,
        )

        # Step 2: chunked cached-prefix attention.
        chunked_out: torch.Tensor | None = None
        chunked_lse: torch.Tensor | None = None
        if getattr(chunk_meta, "is_dcp", False):
            # DCP: the cached context KV is sharded (interleaved) across ranks,
            # so the per-chunk gather becomes gather-local -> AllGather ->
            # reorg -> kv_b_proj. The rest of the merge logic below
            # is identical to the non-DCP path.
            chunked_out, chunked_lse = self._dcp_compute_prefill_context(
                prefill_q, kv_cache, attn_metadata, chunk_meta
            )
        else:
            k_workspace = chunk_meta.k_workspace
            v_workspace = chunk_meta.v_workspace
            for c in range(chunk_meta.num_chunks):
                n_tok = chunk_meta.total_tokens[c]
                if n_tok == 0:
                    continue
                k_chunk = k_workspace[:n_tok]
                v_chunk = v_workspace[:n_tok]
                self._gather_cached_kv_b_proj(
                    kv_cache,
                    chunk_meta.kv_indptr[c],
                    chunk_meta.kv_indices[c],
                    chunk_meta.cu_seqlens_k[c],
                    k_chunk,
                    v_chunk,
                    shuffle_kv_block_indptr=(
                        chunk_meta.shuffle_kv_block_indptr[c]
                        if chunk_meta.shuffle_kv_block_indptr is not None
                        else None
                    ),
                    shuffle_kv_block_indices=(
                        chunk_meta.shuffle_kv_block_indices[c]
                        if chunk_meta.shuffle_kv_block_indices is not None
                        else None
                    ),
                )
                suf_out, suf_lse = flash_attn_varlen_func(
                    q=prefill_q,
                    k=k_chunk,
                    v=v_chunk,
                    cu_seqlens_q=attn_metadata.cu_seqlens_q,
                    cu_seqlens_k=chunk_meta.cu_seqlens_k[c],
                    max_seqlen_q=attn_metadata.max_seqlen_q,
                    max_seqlen_k=chunk_meta.max_seqlen_k[c],
                    min_seqlen_q=attn_metadata.min_seqlen_q,
                    dropout_p=attn_metadata.dropout_p,
                    softmax_scale=self.scale,
                    causal=False,
                    return_lse=True,
                )
                if chunked_out is None:
                    chunked_out = suf_out
                    chunked_lse = suf_lse
                else:
                    tmp_out = torch.empty_like(new_out)
                    tmp_lse = torch.empty_like(new_lse)
                    merge_attn_states(
                        output=tmp_out,
                        output_lse=tmp_lse,
                        prefix_output=chunked_out,
                        prefix_lse=chunked_lse,
                        suffix_output=suf_out,
                        suffix_lse=suf_lse,
                    )
                    chunked_out = tmp_out
                    chunked_lse = tmp_lse

        # Step 3: merge cached prefix (prefix) with new tokens (suffix).
        # If every seq happened to have zero cached tokens this iter, fall
        # back to the new-only output (should not happen since has_cached
        # implies ≥1 seq has cached_len > 0).
        if chunked_out is None:
            output = new_out
        else:
            output = torch.empty_like(new_out)
            merge_attn_states(
                output=output,
                prefix_output=chunked_out,
                prefix_lse=chunked_lse,
                suffix_output=new_out,
                suffix_lse=new_lse,
            )
        return self.o_proj(output.flatten(start_dim=-2))

    def _dcp_compute_prefill_context(
        self,
        prefill_q: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: AttentionMetaData,
        chunk_meta,
    ):
        """DCP chunked cached-prefix context attention.

        Per chunk: index_select this rank's local compressed KV, AllGather it
        across the DCP group, ``reorg_kvcache`` back to per-sequence contiguous
        layout, ``kv_b_proj``-decompress, run flash_attn(causal=False), and
        LSE-merge across chunks. The context attention is unmasked, so the
        rank-major token order produced by reorg does not affect the result.
        """
        from atom.model_ops.attentions.triton_merge_attn_states import merge_attn_states
        from atom.model_ops.dcp_ops import dcp_gather_compressed_kv, reorg_kvcache

        is_fp8_kv = self.kv_cache_dtype.startswith("fp8")

        chunked_out: torch.Tensor | None = None
        chunked_lse: torch.Tensor | None = None
        for c in range(chunk_meta.num_chunks):
            toks = chunk_meta.seq_tot[c]
            if toks == 0:
                continue
            # 1. gather this rank's local compressed KV for the chunk (keeps the
            #    cache dtype, so fp8 stays fp8 here).
            local_kv = dcp_gather_compressed_kv(kv_cache, chunk_meta.local_slot_ids[c])
            # 2. AllGather across DCP ranks -> [toks * dcp_world_size, d]. This is
            #    a copy-only collective, so an fp8 payload is safe (no fp8
            #    arithmetic, unlike an all-reduce).
            ag_kv = self.dcp_group.all_gather(local_kv, dim=0)
            if is_fp8_kv:
                # Dequant the fp8 compressed KV -> model dtype before reorg /
                # kv_b_proj (which expect a bf16 latent). dequant = stored *
                # scale mirrors the write-side quant (value / scale); _k_scale
                # is 1.0 for MLA today. AllGather-then-dequant keeps the wire
                # payload at fp8 (half the bf16 traffic). Tensor arithmetic (not
                # float()/.item()): _k_scale may be a GPU tensor and a host sync
                # is illegal under cudagraph capture; a 0-dim scale broadcasts on
                # device with no sync.
                ag_kv = ag_kv.to(self.dtype) * self._k_scale

            sum_seq_len = chunk_meta.total_tokens[c]
            if sum_seq_len == 0:
                # All tokens in this chunk are padding for every seq (tail
                # chunk); collective already ran, just skip the compute.
                continue

            # 3. reorg interleaved AllGather blocks -> per-seq contiguous.
            ag_kv_c, ag_k_pe = ag_kv.unsqueeze(1).split(
                [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
            )
            kv_c_normed, k_pe = reorg_kvcache(
                ag_kv_c,
                ag_k_pe,
                padded_local_chunk_seq_lens_lst=chunk_meta.padded_local_chunk_seq_lens[
                    c
                ],
                local_context_lens_allranks=chunk_meta.local_context_lens_allranks,
                sum_seq_len=sum_seq_len,
                max_seq_len=chunk_meta.max_seqlen_k[c],
                chunk_size=chunk_meta.chunk_size,
                chunk_idx=c,
                toks=toks,
            )

            # 4. kv_b_proj decompress -> k_nope, v; concat k_pe -> k.
            kv_c_normed = kv_c_normed.squeeze(1)
            kv_nope = self.kv_b_proj(kv_c_normed).view(
                -1, self.num_heads, self.qk_nope_head_dim + self.v_head_dim
            )
            k_nope, v = kv_nope.split([self.qk_nope_head_dim, self.v_head_dim], dim=-1)
            k = torch.cat((k_nope, k_pe.expand((*k_nope.shape[:-1], -1))), dim=-1)

            # 5. flash attention over the (unmasked) context chunk.
            ctx_out, ctx_lse = flash_attn_varlen_func(
                q=prefill_q,
                k=k,
                v=v,
                cu_seqlens_q=attn_metadata.cu_seqlens_q,
                cu_seqlens_k=chunk_meta.cu_seqlens_k[c],
                max_seqlen_q=attn_metadata.max_seqlen_q,
                max_seqlen_k=chunk_meta.max_seqlen_k[c],
                min_seqlen_q=attn_metadata.min_seqlen_q,
                dropout_p=attn_metadata.dropout_p,
                softmax_scale=self.scale,
                causal=False,
                return_lse=True,
            )

            # 6. LSE-merge across chunks.
            if chunked_out is None:
                chunked_out = ctx_out
                chunked_lse = ctx_lse
            else:
                tmp_out = torch.empty_like(ctx_out)
                tmp_lse = torch.empty_like(ctx_lse)
                merge_attn_states(
                    output=tmp_out,
                    output_lse=tmp_lse,
                    prefix_output=chunked_out,
                    prefix_lse=chunked_lse,
                    suffix_output=ctx_out,
                    suffix_lse=ctx_lse,
                )
                chunked_out = tmp_out
                chunked_lse = tmp_lse

        return chunked_out, chunked_lse

    def _forward_prefill_mha(
        self,
        q: torch.Tensor,
        kv_c_normed: torch.Tensor,
        k_rope: torch.Tensor,
        kv_c_and_k_pe_cache: torch.Tensor,
        attn_metadata: AttentionMetaData,
    ) -> torch.Tensor:
        assert attn_metadata is not None

        if k_rope.dim() == 2:
            k_rope = k_rope.unsqueeze(1)

        if use_triton_gemm():
            weight = self.kv_b_proj.weight
            weight_scale = self.kv_b_proj.weight_scale
            if (
                fused_gemm_afp4wfp4_preshuffle_split_cat is not None
                and weight.dtype == dtypes.fp4x2
            ):  # FP4 GEMM + split + cat
                m = kv_c_normed.shape[0]
                # from aiter.ops.triton.quant import dynamic_mxfp4_quant
                # input = kv_c_normed
                # input_2d = input.view(-1, input.shape[-1])
                output_dtype = kv_c_normed.dtype

                # q_input, x_scale = dynamic_mxfp4_quant(input_2d)
                quant_func = get_hip_quant(QuantType.per_1x32)
                q_input, x_scale = quant_func(
                    kv_c_normed,
                    quant_dtype=dtypes.fp4x2,
                    shuffle=(m >= 32),
                )

                if m >= 32:
                    x_scale = x_scale.view(torch.uint8).view(x_scale.shape[0] // 32, -1)
                else:
                    x_scale = x_scale[:m, ...].view(torch.uint8)

                k, v = fused_gemm_afp4wfp4_preshuffle_split_cat(
                    q_input.view(torch.uint8),
                    weight.view(torch.uint8).view(weight.shape[0] // 16, -1),
                    k_rope.expand((-1, self.num_heads, -1)),
                    x_scale,
                    weight_scale.view(torch.uint8).view(
                        weight_scale.shape[0] // 32, -1
                    ),
                    self.qk_nope_head_dim,
                    self.v_head_dim,
                    output_dtype,
                )
            elif (
                fused_gemm_a8w8_blockscale_preshuffle_split_cat is not None
                and weight.dtype == dtypes.fp8
            ):  # FP8 GEMM + split + cat
                weight_shuffled = weight.reshape(
                    weight.shape[0] // 16, weight.shape[1] * 16
                )

                output_dtype = kv_c_normed.dtype

                quant_func = functools_partial(
                    get_hip_quant(QuantType.per_1x128), transpose_scale=True
                )
                q_input, x_scale = quant_func(
                    kv_c_normed,
                    quant_dtype=dtypes.fp8,
                    scale=getattr(self.kv_b_proj, "input_scale", None),
                )

                k, v = fused_gemm_a8w8_blockscale_preshuffle_split_cat(
                    q_input,
                    weight_shuffled,
                    k_rope.expand((-1, self.num_heads, -1)),
                    x_scale,
                    weight_scale,
                    self.qk_nope_head_dim,
                    self.v_head_dim,
                    output_dtype,
                )
            else:
                kv_nope = self.kv_b_proj(kv_c_normed).view(
                    -1, self.num_heads, self.qk_nope_head_dim + self.v_head_dim
                )
                k_nope, v = kv_nope.split(
                    [self.qk_nope_head_dim, self.v_head_dim], dim=-1
                )

                k = torch.cat((k_nope, k_rope.expand((*k_nope.shape[:-1], -1))), dim=-1)
        else:
            kv_nope = self.kv_b_proj(kv_c_normed).view(
                -1, self.num_heads, self.qk_nope_head_dim + self.v_head_dim
            )
            k_nope, v = kv_nope.split([self.qk_nope_head_dim, self.v_head_dim], dim=-1)

            k = torch.cat((k_nope, k_rope.expand((*k_nope.shape[:-1], -1))), dim=-1)

        output = flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=attn_metadata.cu_seqlens_q,
            cu_seqlens_k=attn_metadata.cu_seqlens_k,
            max_seqlen_q=attn_metadata.max_seqlen_q,
            max_seqlen_k=attn_metadata.max_seqlen_k,
            min_seqlen_q=attn_metadata.min_seqlen_q,
            dropout_p=attn_metadata.dropout_p,
            softmax_scale=self.scale,
            causal=True,
        )

        return self.o_proj(output.flatten(start_dim=-2))

    def _forward_prefill_mla(
        self,
        q: torch.Tensor,
        kv_c_and_k_pe_cache: torch.Tensor,
        attn_metadata: AttentionMetaData,
        return_lse: bool = False,
    ) -> torch.Tensor:
        assert attn_metadata is not None
        B = q.shape[0]
        num_heads_q = q.shape[1]

        q = self._pad_query_heads(q)

        # In the seg path q arrives with a padded per-head row stride
        # (_MLA_Q_OUT_PADDED_DIM); slice back to the logical
        # kv_lora_rank + qk_rope_head_dim columns. The slice keeps the padded row
        # stride, which the asm kernel expects. The triton and non-seg
        # (page_size=1) paths use an unpadded 576-wide q_out, so no slicing.
        if self.use_seg_mla:
            q = q[..., : self.kv_lora_rank + self.qk_rope_head_dim]

        o = torch.empty(
            B,
            q.shape[1],
            self.kv_lora_rank,
            dtype=self.dtype,
            device=q.device,
        )

        paged_cu_seqlens_q = attn_metadata.cu_seqlens_q
        paged_kv_indptr = attn_metadata.kv_indptr
        paged_kv_indices = attn_metadata.kv_indices
        kv_last_page_lens = attn_metadata.kv_last_page_lens
        max_q_len = attn_metadata.max_seqlen_q
        if self.is_sparse_mla:
            paged_cu_seqlens_q = attn_metadata.sparse_cu_seqlens_q
            paged_kv_indptr = attn_metadata.sparse_kv_indptr
            paged_kv_indices = self.sparse_kv_indices_buffer
            # Sparse attention needs one last-page len per query token; the dense
            # kv_last_page_lens (per-seq) would over-read -> illegal access.
            kv_last_page_lens = attn_metadata.sparse_kv_last_page_lens
            if self.dcp_world_size > 1:
                # The indexer compacted this rank's owned candidates to the front
                # of each query token's region, so the region lengths are the
                # per-rank (and per-layer) ones, not the global sparse_kv_indptr.
                # Same substitution the decode path makes. See §5.5.5 (P2/P3).
                #
                # MUST be sliced, unlike the decode path: mla_prefill_asm_fwd
                # derives its grid from `kv_indptr->size(0) - 1`
                # (asm_mla.cu:1208) whereas both decode entries use
                # `qo_indptr->size(0) - 1`. Handing it the whole
                # max_num_batched_tokens-sized buffer launches ~32k rows over
                # stale tail entries and faults instantly.
                paged_kv_indptr = self.dcp_sparse_kv_indptr_buffer[
                    : paged_cu_seqlens_q.shape[0]
                ]
            max_q_len = 1

        final_lse = None
        if kv_c_and_k_pe_cache.numel() > 0:
            if envs.ATOM_MLA_PAGE_SIZE is not None:
                page_size = envs.ATOM_MLA_PAGE_SIZE
            else:
                page_size = 1
            # `mla_prefill_asm_fwd` NEVER writes its LSE output: aiter allocates
            # attn_lse with torch.empty and hands the uninitialized buffer straight
            # back (verified with a sentinel probe, DCP_Sparse_MLA.md §5.5.9 --
            # 80/80 entries untouched). So whenever an LSE is required, route
            # through the decode kernel instead. That is sound for sparse prefill
            # because it runs max_q_len=1: every query token is already its own row
            # with its own candidate list, i.e. structurally a decode. The fp8 path
            # has always used this kernel; `return_lse` extends it to bf16.
            use_decode_kernel = (
                self.kv_cache_dtype.startswith("fp8")
                or return_lse
                or _DSA_FORCE_DECODE_KERNEL
            )
            if use_decode_kernel:
                is_fp8 = self.kv_cache_dtype.startswith("fp8")
                # DCP compacts each rank's candidates per layer, so the once-per-step
                # persistent work metadata (built from the GLOBAL sparse_kv_indptr)
                # does not describe this rank's regions -- run non-persistent.
                use_work_meta = is_fp8 and self.dcp_world_size <= 1
                _, final_lse = mla_decode_fwd(
                    q,
                    kv_c_and_k_pe_cache.view(-1, page_size, 1, q.shape[-1]),
                    o,
                    paged_cu_seqlens_q,
                    paged_kv_indptr,
                    paged_kv_indices,
                    kv_last_page_lens,
                    max_q_len,
                    page_size=page_size,
                    # Mirror the decode path's explicit choice. Leaving this None
                    # lets aiter pick, and several of its dispatch branches key on
                    # `num_kv_splits == 1`, which sends stage2 down the FINAL_OUT
                    # copy-through path that NEVER stores Final_lse -- handing back
                    # an untouched torch.empty as the LSE (the same trap as §4.4
                    # and as the asm-prefill LSE bug). Keep it > 1 so stage2 stays
                    # on the reduce path that writes the LSE.
                    num_kv_splits=max(2, 16 // max(1, self.dcp_world_size)),
                    sm_scale=self.scale,
                    q_scale=self._q_scale if is_fp8 else None,
                    kv_scale=self._k_scale if is_fp8 else None,
                    work_meta_data=(
                        getattr(attn_metadata, "sparse_prefill_work_meta_data", None)
                        if use_work_meta
                        else None
                    ),
                    work_indptr=(
                        getattr(attn_metadata, "sparse_prefill_work_indptr", None)
                        if use_work_meta
                        else None
                    ),
                    work_info_set=(
                        getattr(attn_metadata, "sparse_prefill_work_info_set", None)
                        if use_work_meta
                        else None
                    ),
                    reduce_indptr=(
                        getattr(attn_metadata, "sparse_prefill_reduce_indptr", None)
                        if use_work_meta
                        else None
                    ),
                    reduce_final_map=(
                        getattr(attn_metadata, "sparse_prefill_reduce_final_map", None)
                        if use_work_meta
                        else None
                    ),
                    reduce_partial_map=(
                        getattr(
                            attn_metadata, "sparse_prefill_reduce_partial_map", None
                        )
                        if use_work_meta
                        else None
                    ),
                    return_lse=return_lse,
                )
            else:
                mla_prefill_fwd(
                    q,
                    kv_c_and_k_pe_cache.view(-1, page_size, 1, q.shape[-1]),
                    o,
                    paged_cu_seqlens_q,
                    paged_kv_indptr,
                    paged_kv_indices,
                    kv_last_page_lens,
                    max_q_len,
                    self.scale,
                    0.0,
                    None,
                )

        o = self._restore_query_heads(o, num_heads_q)
        if final_lse is not None:
            final_lse = self._restore_query_heads(final_lse, num_heads_q)

        if return_lse:
            assert final_lse is not None, (
                "return_lse requested but the attention kernel produced no LSE "
                "(empty KV cache?)"
            )
            if self.is_sparse_mla and self.dcp_world_size > 1:
                # Rows this rank owns no candidate of (unavoidable at prefill:
                # query token 0 has a single candidate, so W-1 ranks are empty for
                # it -- §5.5.8) get a zero-length KV region. mla_decode_fwd handles
                # that correctly for the LSE (-inf, i.e. zero weight downstream)
                # but leaves `o` as NaN from the 0/0 in acc/e_sum. Zero those rows:
                # cp_lse_ag_out_rs computes `o * exp(lse - global_lse)`, and
                # NaN * 0 would poison the whole merged row. Derived from the
                # kernel's own LSE, so no side-channel buffer and no fake KV.
                o = torch.where(
                    torch.isfinite(final_lse).unsqueeze(-1), o, torch.zeros_like(o)
                )
            return o, final_lse

        return self._v_up_proj_and_o_proj(o)

    def _shuffled_kv_view(self, kv_cache: torch.Tensor):
        """View the flat ``[num_token_slots, 1, d]`` MLA cache as the
        ``[num_blocks, num_kv_heads=1, block_size, d]`` shuffled layout the
        block_size=64 Triton/Gluon MLA kernels read and write.

        This is a pure view: ``num_token_slots == num_blocks * block_size`` by
        construction (block_ratio == kv_cache_block_size), and the per-block
        ``block_size * d`` region is contiguous, which is all the shuffled
        kernels require (they compute their own within-block byte offsets).
        """
        if not hasattr(self, "_shuffle_block_size_cached"):
            self._shuffle_block_size_cached = int(
                get_current_atom_config().kv_cache_block_size
            )
        block_size = self._shuffle_block_size_cached
        d = self.kv_lora_rank + self.qk_rope_head_dim
        num_token_slots = kv_cache.shape[0]
        num_blocks = num_token_slots // block_size
        # [num_token_slots, 1, d] -> [num_blocks, block_size, d] -> [.., 1, ..]
        return kv_cache.view(num_blocks, block_size, d).unsqueeze(1)

    def _forward_decode(
        self,
        q: torch.Tensor,
        kv_c_and_k_pe_cache: torch.Tensor,
        attn_metadata: AttentionMetaData,
        return_lse: bool = False,
    ) -> torch.Tensor:
        assert kv_c_and_k_pe_cache.numel() > 0
        assert attn_metadata is not None
        B = q.shape[0]
        num_heads_q = q.shape[1]

        q = self._pad_query_heads(q)

        # In the seg path q arrives with a padded per-head row stride
        # (_MLA_Q_OUT_PADDED_DIM); slice back to the logical
        # kv_lora_rank + qk_rope_head_dim columns. The slice keeps the padded row
        # stride, which the asm kernel expects. The triton and non-seg
        # (page_size=1) paths use an unpadded 576-wide q_out, so no slicing.
        if self.use_seg_mla:
            q = q[..., : self.kv_lora_rank + self.qk_rope_head_dim]

        o = torch.empty(
            B,
            q.shape[1],
            self.kv_lora_rank,
            dtype=self.dtype,
            device=q.device,
        )

        final_lse = None

        if envs.ATOM_USE_TRITON_MLA and envs.ATOM_USE_TRITON_MLA_SHUFFLE_KV:
            # Shuffled block_size=64 Triton/Gluon MLA decode kernel.
            kv_buffer = self._shuffled_kv_view(kv_c_and_k_pe_cache)
            triton_shuffle_mla_decode_fwd(
                q,  # [num_tokens, num_query_heads, kv_lora_rank + qk_rope_head_dim]
                kv_buffer,  # [num_blocks, 1, block_size, kv_lora_rank + qk_rope_head_dim]
                o,
                attn_metadata.cu_seqlens_q,
                attn_metadata.context_lens,  # seqused_k
                int(attn_metadata.max_seqlen_k),  # max_seqlen_kv
                attn_metadata.block_tables,  # [bs, max_num_blocks_per_seq] (logical)
                self.scale,
                self.kv_lora_rank,
                self.qk_rope_head_dim,
                True,  # causal
                # q is bf16 (the shuffled fused write does not quantize q), so
                # no q de-scale; kv carries its own per-tensor scale.
                None,  # q_descale
                self._k_scale,  # kv_descale
                shuffled_kv_cache=True,
            )
        elif hasattr(attn_metadata, "triton_block_table"):
            from aiter.ops.triton.attention.mla_decode import decode_attention_fwd

            k_buffer = kv_c_and_k_pe_cache.unsqueeze(2)
            v_buffer = k_buffer[..., : self.kv_lora_rank]
            page_size = k_buffer.shape[1]

            q_for_triton = (
                q.to(torch.bfloat16)
                if q.dtype.is_floating_point and q.element_size() == 1
                else q
            )

            # Use pre-built dense block_table from prepare_decode()
            decode_attention_fwd(
                q_for_triton,
                k_buffer,
                v_buffer,
                o,
                attn_metadata.triton_lse,
                attn_metadata.triton_block_table,
                attn_metadata.context_lens,
                attn_metadata.triton_attn_logits,
                4,  # num_kv_splits
                self.scale,
                page_size,
                k_scale=self._k_scale,
                v_scale=self._k_scale,
            )
        else:
            kv_buffer = kv_c_and_k_pe_cache.unsqueeze(2)
            paged_cu_seqlens_q = attn_metadata.cu_seqlens_q
            paged_kv_indptr = attn_metadata.kv_indptr
            paged_kv_indices = attn_metadata.kv_indices
            paged_kv_last_page_lens = attn_metadata.kv_last_page_lens
            max_q_len = attn_metadata.max_seqlen_q
            if self.is_sparse_mla:
                if attn_metadata.max_seqlen_q > 1:
                    # MTP verify: per-token layout with max_q_len=1.
                    # Persistent metadata is per-token (from _set_mla_persistent_worker_buffers_sparse_mtp).
                    paged_cu_seqlens_q = attn_metadata.sparse_cu_seqlens_q
                    paged_kv_indptr = attn_metadata.sparse_kv_indptr
                    paged_kv_last_page_lens = attn_metadata.sparse_kv_last_page_lens
                    paged_kv_indices = self.sparse_kv_indices_buffer
                    max_q_len = 1
                else:
                    # Non-MTP sparse decode: KV is packed per token at
                    # page_size=1, so last_page_len is 1 for every seq. Use the
                    # all-1s sparse buffer, NOT the dense per-block
                    # kv_last_page_lens (which makes the asm kernel over-read
                    # past the written sparse-index region -> illegal access).
                    paged_kv_indptr = attn_metadata.sparse_kv_indptr
                    paged_kv_indices = self.sparse_kv_indices_buffer
                    paged_kv_last_page_lens = attn_metadata.sparse_kv_last_page_lens
                    if self.dcp_world_size > 1:
                        # DCP compacts each rank's owned top-k to the front, so
                        # the region length is shorter than the global clip and
                        # varies per layer. The indexer just rewrote these
                        # offsets for this layer (§5.3).
                        paged_kv_indptr = self.dcp_sparse_kv_indptr_buffer

            dp_size = get_dp_group().world_size
            use_persistent_mode = not (dp_size > 1)
            if envs.ATOM_MLA_PAGE_SIZE > 1:
                use_persistent_mode = False
            if self.dcp_world_size > 1 and not self.dcp_persistent_supported:
                use_persistent_mode = False
            # sparse + DCP compacts the per-rank top-k (§5.1), which makes the
            # sparse region length depend on the *per-layer* selection. The
            # persistent work/reduce metadata is built once per step from
            # sparse_kv_indptr (aiter_mla.set_mla_persistent_worker_buffers), so
            # it cannot describe a length that changes layer to layer -- the
            # timing simply does not line up. Run non-persistent until either the
            # metadata build moves per-layer or aiter grows a per-request valid
            # length (§5.5).
            if self.is_sparse_mla and (
                self.dcp_world_size > 1 or self._dsa_force_nonpersist
            ):
                use_persistent_mode = False

            # Sparse layers in MTP verify use separate persistent metadata
            # (per-token, max_seqlen_qo=1) while dense layers use normal metadata
            # (max_seqlen_qo=2).
            is_sparse_mtp = self.is_sparse_mla and attn_metadata.max_seqlen_q > 1

            if not use_persistent_mode:
                work_meta_data = None
                work_indptr = None
                work_info_set = None
                reduce_indptr = None
                reduce_final_map = None
                reduce_partial_map = None
            elif is_sparse_mtp:
                work_meta_data = attn_metadata.sparse_mtp_work_meta_data
                work_indptr = attn_metadata.sparse_mtp_work_indptr
                work_info_set = attn_metadata.sparse_mtp_work_info_set
                reduce_indptr = attn_metadata.sparse_mtp_reduce_indptr
                reduce_final_map = attn_metadata.sparse_mtp_reduce_final_map
                reduce_partial_map = attn_metadata.sparse_mtp_reduce_partial_map
            else:
                work_meta_data = attn_metadata.work_meta_data
                work_indptr = attn_metadata.work_indptr
                work_info_set = attn_metadata.work_info_set
                reduce_indptr = attn_metadata.reduce_indptr
                reduce_final_map = attn_metadata.reduce_final_map
                reduce_partial_map = attn_metadata.reduce_partial_map

            # persistent lets metadata (reduce_partial_map) drive the split, so
            # 16 is inert; the non-persistent fallback (gfx942 DCP) still needs
            # the DCP-scaled split to keep the fp32 `logits` small (CUDA-graph OOM).
            num_kv_splits = (
                16 if use_persistent_mode else max(1, 16 // self.dcp_world_size)
            )
            # NOTE (§5.2): sparse + DCP used to force num_kv_splits=1 here, because
            # a -1-padded fixed-length region made aiter's stage2 believe every
            # split held tokens and one of them could be all sentinels -> NaN. That
            # forcing then hit a second problem: num_kv_splits==1 sends aiter's
            # stage2 down its copy-through fast path (FINAL_OUT, aiter/mla.py:74),
            # which never stores Final_lse, so return_lse handed back an untouched
            # torch.empty buffer. Compaction removes both: cur_kv_seq_len is now the
            # true (small) owned count, so stage2's own
            # min(splits, cdiv(seq_len, mgc)) clamp skips the empty splits, and
            # num_kv_splits>1 keeps it on the reduce path that writes the lse.

            # TODO refactor this
            if envs.ATOM_MLA_PAGE_SIZE is not None:
                page_size = envs.ATOM_MLA_PAGE_SIZE
            else:
                page_size = 1

            # DCP + MTP (max_q_len>1): KV is round-robin sharded across ranks, so
            # the intra-block causal mask must be applied on GLOBAL positions
            # g(j)=j*W+r. Pass the cprr params (g_kv_indptr + cp world/rank) so the
            # kernel selects the cprr variant and masks correctly. qlen=1 keeps the
            # plain path (single query sees all local KV -> no mask needed).
            cp_world_size = 1
            cp_rank = 0
            g_kv_indptr = None
            if self.dcp_world_size > 1 and max_q_len > 1:
                # cprr kernel is bf16-only (no fp8 cprr kernel on gfx950).
                if self.kv_cache_dtype == "fp8":
                    raise NotImplementedError(
                        "MTP + DCP (max_q_len>1) requires bf16 KV cache: the "
                        "round-robin CP (cprr) MLA kernel has no fp8 variant. "
                        "Use --kv_cache_dtype bf16, or disable DCP/MTP for fp8."
                    )
                cp_world_size = self.dcp_world_size
                cp_rank = self.dcp_rank
                g_kv_indptr = getattr(attn_metadata, "g_kv_indptr", None)
                assert g_kv_indptr is not None, (
                    "MTP+DCP decode requires attn_metadata.g_kv_indptr; the "
                    "metadata builder / cudagraph-capture path must set it."
                )

            seg_kv_buffer_4d = kv_buffer.view(-1, page_size, 1, q.shape[-1])
            _, final_lse = mla_decode_fwd(
                q,
                seg_kv_buffer_4d,
                o,
                paged_cu_seqlens_q,
                paged_kv_indptr,
                paged_kv_indices,
                paged_kv_last_page_lens,
                max_q_len,
                page_size=page_size,
                # The seg/asm decode path runs with a single kv split; the
                # page_size=1 persistent path keeps 16 splits (metadata-derived).
                num_kv_splits=None if self.use_seg_mla else num_kv_splits,
                sm_scale=self.scale,
                work_meta_data=work_meta_data,
                work_indptr=work_indptr,
                work_info_set=work_info_set,
                reduce_indptr=reduce_indptr,
                reduce_final_map=reduce_final_map,
                reduce_partial_map=reduce_partial_map,
                q_scale=self._q_scale,
                kv_scale=self._k_scale,
                return_lse=return_lse,
                g_kv_indptr=g_kv_indptr,
                cp_world_size=cp_world_size,
                cp_rank=cp_rank,
            )

        o = self._restore_query_heads(o, num_heads_q)
        if final_lse is not None:
            final_lse = self._restore_query_heads(final_lse, num_heads_q)

        if return_lse:
            return o, final_lse

        return self._v_up_proj_and_o_proj(o)

    def _pcp_write_full_kv(self, kv_cache, k_nope, k_rope, slot_mapping):
        """Write an already-roped full k (kv_lora + rope) into the k-cache.

        Used by the PCP prefill path to materialise the full sequence's KV after
        the fused MLA kernel produced q_out on 1/pcp queries. Mirrors the
        non-fused k-writes used by the dense (`not use_prefill_mla`) prefill
        branch so the physical cache layout matches exactly. `k_rope` must
        already be rotary-embedded.
        """
        if envs.ATOM_USE_TRITON_MLA and envs.ATOM_USE_TRITON_MLA_SHUFFLE_KV:
            shuffled_cache = self._shuffled_kv_view(kv_cache)
            triton_cat_and_cache_mla(
                k_nope.view(-1, self.num_kv_heads, self.kv_lora_rank),
                k_rope.view(-1, self.num_kv_heads, self.qk_rope_head_dim),
                shuffled_cache,
                slot_mapping.flatten(),
                self._k_scale,
                apply_scale=True,
                shuffled_kv_cache=True,
            )
        elif self.use_seg_mla:
            kv_cache_seg = self._seg_kv_cache_view(kv_cache)
            concat_and_cache_mla_seg(
                k_nope,
                k_rope.squeeze(1),
                kv_cache_seg,
                slot_mapping.flatten(),
                kv_cache_dtype=self.kv_cache_dtype,
                scale=self._k_scale,
            )
        else:
            concat_and_cache_mla(
                k_nope,
                k_rope.squeeze(1),
                kv_cache,
                slot_mapping.flatten(),
                kv_cache_dtype=self.kv_cache_dtype,
                scale=self._k_scale,
            )

    def forward_impl(
        self,
        q: torch.Tensor,
        k_nope: torch.Tensor,
        k_rope: torch.Tensor,
        positions: torch.Tensor = None,
        q_scale: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # kv_cache = self.kv_cache
        forward_context: ForwardContext = get_forward_context()
        attn_metadata = forward_context.attn_metadata
        context = forward_context.context
        use_prefill_mla = (
            self.is_sparse_mla and attn_metadata.max_seqlen_k > self.topk_tokens
        )
        # DIAGNOSTIC (see ATOM_DSA_DCP_DENSE_PREFILL in deepseek_v2.py): keep
        # prefill on the dense path under DCP regardless of prompt length, to
        # separate a sparse-prefill fault from a sparse-decode one. The indexer
        # skips writing top-k indices under the same env, so this must flip with it.
        if self._dsa_dcp_dense_prefill:
            use_prefill_mla = False
        if forward_context.context.is_dummy_run:
            output_shape = list(q.shape)
            atom_config = get_current_atom_config()
            output_shape[-1] = _mla_output_width(
                self, atom_config.hf_config.hidden_size
            )
            output_dtype = atom_config.torch_dtype
            output = torch.empty(output_shape, dtype=output_dtype, device=q.device)
            return output
        kv_cache_data = forward_context.kv_cache_data
        kv_cache = kv_cache_data[f"layer_{self.layer_num}"].k_cache

        if context.is_prefill and not use_prefill_mla:
            prefill_q = self.q_proj(q, x_scale=q_scale).view(
                -1, self.num_heads, self.qk_head_dim
            )
            prefill_q_pe = prefill_q[..., self.qk_nope_head_dim :]
            self.rotary_emb(positions, prefill_q_pe, k_rope)

            if kv_cache.numel() > 0:
                if envs.ATOM_USE_TRITON_MLA and envs.ATOM_USE_TRITON_MLA_SHUFFLE_KV:
                    shuffled_cache = self._shuffled_kv_view(kv_cache)
                    triton_cat_and_cache_mla(
                        k_nope.view(-1, self.num_kv_heads, self.kv_lora_rank),
                        k_rope.view(-1, self.num_kv_heads, self.qk_rope_head_dim),
                        shuffled_cache,
                        attn_metadata.slot_mapping.flatten(),
                        self._k_scale,
                        apply_scale=True,
                        shuffled_kv_cache=True,
                    )
                elif self.use_seg_mla:
                    # Write the KV cache in the segmented layout so the
                    # decode-phase mla_decode_fwd (which reads seg layout) sees a
                    # consistent cache for tokens written during prefill.
                    # kv_cache is flattened to
                    # [num_blocks, page_size*(kv_lora_rank + qk_rope_head_dim)] so
                    # the kernel derives page_size from stride(0).
                    kv_cache_seg = self._seg_kv_cache_view(kv_cache)
                    concat_and_cache_mla_seg(
                        k_nope,
                        k_rope.squeeze(1),
                        kv_cache_seg,
                        attn_metadata.slot_mapping.flatten(),
                        kv_cache_dtype=self.kv_cache_dtype,
                        scale=self._k_scale,
                    )
                else:
                    concat_and_cache_mla(
                        k_nope,
                        k_rope.squeeze(1),
                        kv_cache,
                        attn_metadata.slot_mapping.flatten(),
                        kv_cache_dtype=self.kv_cache_dtype,
                        scale=self._k_scale,
                    )

            if attn_metadata.has_cached:
                # Shuffled KV: the builder nulls mla_chunk_meta, so cached-prefix
                # prefill always takes the single-pass gather (which is shuffle
                # aware). The chunked path stays on the plain layout.
                chunk_meta = getattr(attn_metadata, "mla_chunk_meta", None)
                if chunk_meta is not None:
                    output = self._forward_prefill_cached_chunked(
                        prefill_q, k_nope, k_rope, kv_cache, attn_metadata, chunk_meta
                    )
                else:
                    output = self._forward_prefill_cached_single_pass(
                        prefill_q, kv_cache, attn_metadata
                    )
            else:
                output = self._forward_prefill_mha(
                    prefill_q, k_nope, k_rope, kv_cache, attn_metadata
                )
        else:
            q_nope, q_rope = self._q_proj_and_k_up_proj(q, x_scale=q_scale)

            # ---- Prefill Context Parallel --------------------------------
            # q is this rank's 1/pcp queries, so q_out is naturally 1/pcp. But
            # the k-cache must hold the FULL sequence (every rank keeps full KV).
            # The fused MLA kernel below couples q_out with the k-write on one
            # token count, so under PCP it runs on the owned slots (q_out is
            # correct; its k-write is throwaway) and the full k-cache is written
            # afterwards from the all-gathered k. Gather the raw (un-roped) k and
            # key positions BEFORE the fused kernel ropes k in place.
            pcp = (
                pcp_is_enabled()
                and context.is_prefill
                and not context.is_dummy_run
                and use_prefill_mla
            )
            if pcp:
                pcp_ws = get_pcp_world_size()
                n_real = attn_metadata.slot_mapping.shape[0]
                k_nope_full = pcp_allgather_rerange(k_nope, pcp_ws)[:n_real]
                k_rope_full = pcp_allgather_rerange(k_rope, pcp_ws)[:n_real]
                positions_full = pcp_allgather_rerange(positions, pcp_ws)[:n_real]
                write_slot_mapping = attn_metadata.slot_mapping_owned
            else:
                write_slot_mapping = attn_metadata.slot_mapping

            if self.use_seg_mla:
                # Seg path: allocate q_out with a padded last dim so each head row
                # has a 768-byte stride (required by the gfx1250 decode asm). The
                # kernel only writes the first kv_lora_rank + qk_rope_head_dim
                # columns; the padding tail is left untouched and never read.
                q_out = torch.empty(
                    (
                        q_nope.shape[0],
                        self.num_heads,
                        _MLA_Q_OUT_PADDED_DIM,
                    ),
                    dtype=attn_metadata.dtype_q,
                    device=q_nope.device,
                )
            else:
                q_out = torch.empty(
                    (
                        q_nope.shape[0],
                        self.num_heads,
                        self.kv_lora_rank + self.qk_rope_head_dim,
                    ),
                    dtype=attn_metadata.dtype_q,
                    device=q_nope.device,
                )
            if kv_cache.numel() > 0:
                if (
                    envs.ATOM_USE_TRITON_MLA
                    and envs.ATOM_USE_TRITON_MLA_SHUFFLE_KV
                    and self.dcp_world_size <= 1
                ):
                    shuffled_cache = self._shuffled_kv_view(kv_cache)
                    triton_fused_qk_rope_cat_and_cache_mla(
                        q_nope,
                        q_rope,
                        k_nope.view(-1, self.num_kv_heads, self.kv_lora_rank),
                        k_rope.view(-1, self.num_kv_heads, self.qk_rope_head_dim),
                        shuffled_cache,
                        write_slot_mapping,
                        positions,
                        self.rotary_emb.cos_cache,
                        self.rotary_emb.sin_cache,
                        self._k_scale,
                        self.rotary_emb.is_neox_style,
                        num_decode_toks_for_zeros=0,
                        apply_scale=True,
                        q_out=q_out,
                        shuffled_kv_cache=True,
                    )
                elif self.use_seg_mla and self.dcp_world_size <= 1:
                    kv_cache_seg = self._seg_kv_cache_view(kv_cache)
                    fused_qk_rope_concat_and_cache_mla_seg(
                        q_nope,
                        q_rope,
                        k_nope,
                        k_rope,
                        # Flat seg layout: [num_blocks, page_size*(kv_lora + pe)].
                        kv_cache_seg,
                        q_out,
                        write_slot_mapping,
                        self._k_scale,
                        self._q_scale,
                        positions,
                        self.rotary_emb.cos_cache,
                        self.rotary_emb.sin_cache,
                        is_neox=self.rotary_emb.is_neox_style,
                    )
                else:
                    # DCP: q_out is head all-gathered, so every rank must compute
                    # Q RoPE for all tokens (incl. slot=-1 non-owned). Non-DCP
                    # keeps the default (early-return on padded tokens).
                    fused_qk_rope_concat_and_cache_mla(
                        q_nope,
                        q_rope,
                        k_nope,
                        k_rope,
                        kv_cache.view(
                            kv_cache.shape[0],
                            -1,
                            self.kv_lora_rank + self.qk_rope_head_dim,
                        ),
                        q_out,
                        write_slot_mapping,
                        self._k_scale,
                        self._q_scale,
                        positions,
                        self.rotary_emb.cos_cache,
                        self.rotary_emb.sin_cache,
                        is_neox=self.rotary_emb.is_neox_style,
                        is_nope_first=True,
                        compute_all_q_rope=self.dcp_world_size > 1,
                    )
                # q_out = self.fused_kv_bmm(q, q_scale, k_nope, k_rope, positions, kv_cache, attn_metadata)

                if pcp:
                    # Complete the full k-cache: rope the gathered full k (in
                    # place) then write every real slot, overwriting the fused
                    # kernel's throwaway owned-slot write. The rope kernel is
                    # 2-component and needs a non-None partner, so pass a
                    # throwaway query of matching length.
                    self.rotary_emb(
                        positions_full, k_rope_full, torch.empty_like(k_rope_full)
                    )
                    self._pcp_write_full_kv(
                        kv_cache,
                        k_nope_full,
                        k_rope_full,
                        attn_metadata.slot_mapping,
                    )

            if context.is_prefill:
                if (
                    self.is_sparse_mla
                    and self.dcp_world_size > 1
                    and _DSA_DCP_PREFILL_MERGE
                ):
                    # DCP sparse prefill: the indexer left each rank with a
                    # DISJOINT slice of the global top-k (§5.5.5 P2), so this is
                    # a partial softmax that must be merged exactly like decode.
                    # Legal because sparse prefill runs max_q_len=1 -- every query
                    # token is its own row with its own candidate list, and the
                    # causal window was already applied when the candidates were
                    # chosen, so no global-position masking is needed (§5.5.6 ③).
                    from atom.model_ops.dcp_ops import cp_lse_ag_out_rs

                    q_out = self.dcp_group.all_gather(q_out, dim=1)
                    o, lse = self._forward_prefill_mla(
                        q_out, kv_cache, attn_metadata, return_lse=True
                    )
                    # DIAGNOSTIC (§5.5): cp_lse_ag_out_rs runs entirely in the
                    # attention dtype -- correct_attn_out rescales in place and the
                    # ReduceScatter sums W partials. In bf16 that is one extra
                    # rounding per rank per layer, on top of dcp=1's single fp32
                    # accumulation inside the kernel. Prefill chains 78 such layers,
                    # so promote to fp32 to measure how much of the observed
                    # divergence is merge precision.
                    if _DSA_DCP_MERGE_FP32:
                        o = cp_lse_ag_out_rs(
                            o.float(), lse, self.dcp_group, ctx=None
                        ).to(o.dtype)
                    else:
                        o = cp_lse_ag_out_rs(
                            o, lse, self.dcp_group, ctx=self._cp_triton_ctx
                        )
                    output = self._v_up_proj_and_o_proj(o)
                else:
                    output = self._forward_prefill_mla(q_out, kv_cache, attn_metadata)
            elif self.dcp_world_size > 1:
                # DCP decode: AllGather Q on the head dim, decode locally with LSE,
                # then combine partial outputs across ranks (AG LSE + correct + RS).
                from atom.model_ops.dcp_ops import cp_lse_ag_out_rs

                q_out = self.dcp_group.all_gather(q_out, dim=1)
                o, lse = self._forward_decode(
                    q_out, kv_cache, attn_metadata, return_lse=True
                )
                o = cp_lse_ag_out_rs(o, lse, self.dcp_group, ctx=self._cp_triton_ctx)
                output = self._v_up_proj_and_o_proj(o)
            else:
                output = self._forward_decode(q_out, kv_cache, attn_metadata)

        if (
            _DSA_ATTN_DUMP
            and context.is_prefill
            and self.layer_num in _DSA_ATTN_DUMP_LAYERS
            and _dsa_attn_dump_count.get(self.layer_num, 0) < 2
            and torch.distributed.get_rank() == 0
        ):
            # Prefill output of one layer, post o_proj: identical shape and meaning at
            # dcp=1 and dcp=N (TP sharding is already all-reduced away), so it is
            # the one tensor that directly answers "is the DCP sparse-prefill
            # attention + merge numerically right?" (DCP_Sparse_MLA.md §5.5).
            _n = _dsa_attn_dump_count.get(self.layer_num, 0)
            _dsa_attn_dump_count[self.layer_num] = _n + 1
            import numpy as _np

            _np.savez(
                f"{_DSA_ATTN_DUMP}/attn_L{self.layer_num}"
                f"_w{self.dcp_world_size}_r{_n}.npz",
                out_tail=output[-4:].float().cpu().numpy(),
                # Strided subsample across ALL rows. The tail-4 max|diff| used
                # earlier has almost no statistical power: at |o|~6e-3 the bf16
                # ULP is 1.5e-5, so a systematic error below ~0.25% relative is
                # invisible in a max over 4 rows. What discriminates is the
                # FRACTION of differing elements over many rows (§5.5).
                out_sub=output[::7, ::13].float().cpu().numpy(),
                shape=_np.asarray(list(output.shape)),
            )
            print(
                f"[DSA-ATTN-DUMP] L={self.layer_num} "
                f"w={self.dcp_world_size} run={_n} shape={tuple(output.shape)}",
                flush=True,
            )
        return output

    def forward(
        self,
        query: torch.Tensor,  # query in unified attn
        k_nope: torch.Tensor,
        k_rope: torch.Tensor,
        kv_cache: torch.Tensor = None,
        attn_metadata=None,
        positions: torch.Tensor = None,
        q_scale: Optional[torch.Tensor] = None,
        output: torch.Tensor = None,
        **kwargs,
    ) -> torch.Tensor:
        return self.forward_impl(
            q=query,
            k_nope=k_nope,
            k_rope=k_rope,
            positions=positions,
            q_scale=q_scale,
        )


@triton.jit
def _convert_req_index_to_global_index_kernel(
    qo_indptr,  # int32 [num_requests]
    kv_indptr,  # int32 [num_requests+1]
    page_kv_indptr,  # int32 [num_requests+1]
    kv_indices,  # int32 [num_requests * max_num_blocks_per_req]
    token_indices_ptr,  # int32 [num_tokens, NUM_TOPK_TOKENS]
    out_kv_indices,  # int32
    # shapes (compile-time where possible)
    NUM_TOPK_TOKENS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_N: tl.constexpr,  # tile width along columns
    # strides (in elements)
    ti_stride0,
    ti_stride1,
):
    # program_id(0) -> batch_id (row)
    # program_id(1) -> tile index along columns
    batch_id = tl.program_id(0)
    tile_id = tl.program_id(1)

    # Each program covers BLOCK_N consecutive columns
    indice_id = tile_id * BLOCK_N + tl.arange(0, BLOCK_N)

    # Load request id for this token (no mask: grid is exact)
    kv_start = tl.load(kv_indptr + batch_id)
    kv_end = tl.load(kv_indptr + batch_id + 1)
    out_kv_start = tl.load(page_kv_indptr + batch_id)
    kv_len = kv_end - kv_start
    qo_start = tl.load(qo_indptr + batch_id)
    qo_end = tl.load(qo_indptr + batch_id + 1)

    for token_id in range(qo_start, qo_end):
        # Load token indices for this tile
        ti_ptr = token_indices_ptr + token_id * ti_stride0 + indice_id * ti_stride1
        tok = tl.load(ti_ptr)  # int32

        # Split masks: store_mask = column-valid, load_mask adds tok-bound
        # guard to prevent OOB GPU fault (masked load yields 0 = valid page).
        store_mask = (indice_id < kv_len) & (indice_id < NUM_TOPK_TOKENS)
        load_mask = store_mask & (tok >= 0) & (tok < kv_len)
        out_val = tl.load(
            kv_indices + kv_start + tok,
            mask=load_mask,
            other=0,
        )

        # Store results
        out_ptr_ij = out_kv_indices + out_kv_start + indice_id
        tl.store(
            out_ptr_ij,
            out_val,
            mask=store_mask,
        )


def triton_convert_req_index_to_global_index(
    qo_indptr: torch.Tensor,  # int32 [num_tokens + 1]
    kv_indptr: torch.Tensor,  # int32 [num_tokens + 1]
    page_kv_indptr: torch.Tensor,  # int32 [num_tokens + 1]
    kv_indices: torch.Tensor,  # int32 [total_kv_seqlen]
    token_indices: torch.Tensor,  # int32 [num_tokens, NUM_TOPK_TOKENS]
    BLOCK_SIZE: int = 1,  # page_block_size = 1 for now
    NUM_TOPK_TOKENS: int = 2048,
    BLOCK_N: int = 128,  # tile width along columns
    out: Optional[torch.Tensor] = None,
):
    """
    out[token_id, indice_id] =
        block_table[req_id[token_id],
            token_indices[token_id, indice_id] // BLOCK_SIZE] * BLOCK_SIZE
        + token_indices[token_id, indice_id] % BLOCK_SIZE

    Only when token_indices[token_id, indice_id] == -1 do we output -1.
    For safety, we also output -1 if the derived block_id would be
        out-of-bounds.
    """
    assert kv_indices.dtype == torch.int32
    assert token_indices.dtype == torch.int32
    assert token_indices.shape[1] == NUM_TOPK_TOKENS
    assert NUM_TOPK_TOKENS % BLOCK_N == 0, (
        f"NUM_TOPK_TOKENS ({NUM_TOPK_TOKENS}) must be divisible by"
        f"BLOCK_N ({BLOCK_N})"
    )

    num_batch = kv_indptr.shape[0] - 1
    tiles_per_row = NUM_TOPK_TOKENS // BLOCK_N

    # Ensure contiguous tensors on the same device
    qo_indptr_c = qo_indptr.contiguous()
    kv_indptr_c = kv_indptr.contiguous()
    kv_indices_c = kv_indices.contiguous()
    token_indices_c = token_indices.contiguous()
    page_kv_indptr_c = page_kv_indptr.contiguous()
    # NOTE: MTP (max_seqlen_q > 1) uses triton_convert_req_index_to_global_index_dsa_prefill instead
    if out is not None:
        new_kv_indices = out[: kv_indices.shape[0]]
    else:
        new_kv_indices = torch.empty_like(kv_indices)

    # Strides in elements
    ti_stride0, ti_stride1 = token_indices_c.stride()

    # Exact 2D grid: tokens x column tiles
    grid = (num_batch, tiles_per_row)

    _convert_req_index_to_global_index_kernel[grid](
        qo_indptr_c,
        kv_indptr_c,
        page_kv_indptr_c,
        kv_indices_c,
        token_indices_c,
        new_kv_indices,
        # shapes / constexprs
        NUM_TOPK_TOKENS,
        BLOCK_SIZE,
        BLOCK_N,
        # strides
        ti_stride0,
        ti_stride1,
    )
    return new_kv_indices


@triton.jit
def _count_owned_dcp_kernel(
    qo_indptr,  # int32 [num_requests + 1]
    global_kv_indptr,  # int32 [num_requests + 1] -- GLOBAL context (column range)
    token_indices_ptr,  # int32 [num_tokens, NUM_TOPK_TOKENS] -- GLOBAL top-k positions
    out_counts,  # int32 [num_requests] -- owned top-k count per request
    DCP_RANK: tl.constexpr,
    DCP_WORLD: tl.constexpr,
    NUM_TOPK_TOKENS: tl.constexpr,
    BLOCK_N: tl.constexpr,
    ti_stride0,
    ti_stride1,
):
    """Pass 1 of the compacting DCP filter: how many of the global top-k
    positions does this rank own, per request? Its exclusive cumsum gives the
    compacted output offsets used by ``_compact_filter_dcp_kernel``.

    qlen==1 only (DCP + sparse + MTP is rejected upstream), so each request has
    exactly one query token.
    """
    batch_id = tl.program_id(0)
    g_kv_start = tl.load(global_kv_indptr + batch_id)
    g_kv_len = tl.load(global_kv_indptr + batch_id + 1) - g_kv_start
    token_id = tl.load(qo_indptr + batch_id)

    count = 0
    for tile_start in range(0, NUM_TOPK_TOKENS, BLOCK_N):
        indice_id = tile_start + tl.arange(0, BLOCK_N)
        col_valid = (indice_id < g_kv_len) & (indice_id < NUM_TOPK_TOKENS)
        ti_ptr = token_indices_ptr + token_id * ti_stride0 + indice_id * ti_stride1
        tok = tl.load(ti_ptr, mask=col_valid, other=-1)
        owned = col_valid & (tok >= 0) & ((tok % DCP_WORLD) == DCP_RANK)
        count += tl.sum(owned.to(tl.int32))

    tl.store(out_counts + batch_id, count)


@triton.jit
def _compact_filter_dcp_kernel(
    qo_indptr,  # int32 [num_requests + 1]
    global_kv_indptr,  # int32 [num_requests + 1] -- GLOBAL context (column range)
    out_kv_indptr,  # int32 [num_requests + 1] -- COMPACTED offsets (cumsum of pass 1)
    block_table,  # int32 [num_req, max_num_blocks_per_req] -- logical(global) blocks
    token_indices_ptr,  # int32 [num_tokens, NUM_TOPK_TOKENS] -- GLOBAL top-k positions
    out_kv_indices,  # int32 [>= out_kv_indptr[-1]]
    DCP_RANK: tl.constexpr,
    DCP_WORLD: tl.constexpr,
    PAGE_SIZE: tl.constexpr,  # runner (physical) block size
    NUM_TOPK_TOKENS: tl.constexpr,
    BLOCK_N: tl.constexpr,
    ti_stride0,
    ti_stride1,
    bt_stride0: tl.int64,
    bt_stride1: tl.constexpr,
):
    # DCP round-robin (interleave=1): a GLOBAL position g is owned by rank
    # ``g % W``; on the owner rank its physical slot follows the round-robin
    # (virtual-block) layout used by _dcp_round_robin_slot / ATOM PR #847:
    #     vbs  = PAGE_SIZE * W
    #     slot = block_table[req, g // vbs] * PAGE_SIZE + (g % vbs) // W
    # token_indices holds GLOBAL positions (the indexer scored the full sequence
    # via all-gathered logits). This rank keeps ONLY the positions it owns and
    # writes them COMPACTED to the front of its region -- no -1 holes. Holes are
    # exactly what breaks aiter's lse path (immediate fault on the persistent
    # kernel, silently unwritten lse on the split-KV one); see
    # DCP/DCP_Sparse_MLA.md §4.11/§4.13.
    #
    # Compaction is order-preserving (tl.cumsum within a tile plus a running
    # offset across tiles) rather than atomic-allocated like vLLM, so the KV
    # order -- and hence the floating-point accumulation order -- is
    # deterministic run to run, which the dcp=1 vs dcp=N comparison relies on.
    #
    # NOTE: the slot is computed from block_table directly (like vLLM) rather
    # than gathered from a precomputed kv_indices -- the DCP round-robin
    # per-token slot array does not exist on the sparse path (dense reads go
    # through block_tables in-kernel).
    batch_id = tl.program_id(0)

    g_kv_start = tl.load(global_kv_indptr + batch_id)
    g_kv_len = tl.load(global_kv_indptr + batch_id + 1) - g_kv_start
    out_kv_start = tl.load(out_kv_indptr + batch_id)
    token_id = tl.load(qo_indptr + batch_id)

    vbs = PAGE_SIZE * DCP_WORLD
    written = 0
    for tile_start in range(0, NUM_TOPK_TOKENS, BLOCK_N):
        indice_id = tile_start + tl.arange(0, BLOCK_N)
        # Columns within [0, min(global_ctx, topk)) carry a real top-k position.
        col_valid = (indice_id < g_kv_len) & (indice_id < NUM_TOPK_TOKENS)

        ti_ptr = token_indices_ptr + token_id * ti_stride0 + indice_id * ti_stride1
        tok = tl.load(ti_ptr, mask=col_valid, other=-1)  # GLOBAL position

        idx_valid = col_valid & (tok >= 0) & ((tok % DCP_WORLD) == DCP_RANK)

        block_id = tok // vbs
        inblock_offset = (tok % vbs) // DCP_WORLD
        physical_block = tl.load(
            block_table + batch_id * bt_stride0 + block_id * bt_stride1,
            mask=idx_valid,
            other=0,
        )
        slot = physical_block * PAGE_SIZE + inblock_offset

        # Exclusive prefix sum of the owned mask -> destination inside this tile.
        owned_i32 = idx_valid.to(tl.int32)
        dst = written + tl.cumsum(owned_i32, axis=0) - owned_i32
        tl.store(out_kv_indices + out_kv_start + dst, slot, mask=idx_valid)
        written += tl.sum(owned_i32)


def triton_filter_and_convert_dcp_index(
    qo_indptr: torch.Tensor,  # int32 [num_requests + 1]
    global_kv_indptr: torch.Tensor,  # int32 [num_requests + 1]
    block_table: torch.Tensor,  # int32 [num_req, max_num_blocks_per_req] logical
    token_indices: torch.Tensor,  # int32 [num_tokens, NUM_TOPK_TOKENS] GLOBAL pos
    dcp_rank: int,
    dcp_world_size: int,
    block_size: int,  # runner (physical) block size == PAGE_SIZE
    out_kv_indptr: torch.Tensor,  # int32 [num_requests + 1] COMPACTED, written here
    owned_counts: torch.Tensor,  # int32 [>= num_requests] scratch for pass 1
    NUM_TOPK_TOKENS: int = 2048,
    BLOCK_N: int = 128,
    out: Optional[torch.Tensor] = None,
):
    """DCP (interleave=1) filter + round-robin localize of global top-k positions,
    **compacting** each rank's owned slots to the front of its region.

    ``token_indices[token_id, indice_id]`` is a GLOBAL token position selected by
    the indexer (scored over the full sequence via all-gathered logits). This
    rank keeps a position ``g`` only if ``g % W == dcp_rank`` and maps it to its
    physical slot via the round-robin (virtual-block) layout, computed directly
    from ``block_table`` (like vLLM):
        vbs  = block_size * W
        slot = block_table[req, g // vbs] * block_size + (g % vbs) // W

    Non-owned positions are **dropped**, not marked: the kept slots are packed
    contiguously (original top-k order preserved) and ``out_kv_indptr`` is
    rewritten to the resulting per-request lengths. This replaces the earlier
    "fixed length + -1 sentinel" layout, whose holes broke aiter's lse output
    (see DCP/DCP_Sparse_MLA.md ch.4-5). Because the kept count depends on the
    per-layer top-k selection, ``out_kv_indptr`` is layer-dependent and must be
    recomputed on every call -- it cannot feed the once-per-step persistent
    metadata, which is why sparse+DCP runs non-persistent for now (§5.1).

    The 8 ranks' kept sets are disjoint and their union is exactly the global
    top-k, which is what makes the downstream ``cp_lse_ag_out_rs`` merge valid.
    """
    assert token_indices.dtype == torch.int32
    assert token_indices.shape[1] == NUM_TOPK_TOKENS
    assert NUM_TOPK_TOKENS % BLOCK_N == 0, (
        f"NUM_TOPK_TOKENS ({NUM_TOPK_TOKENS}) must be divisible by"
        f"BLOCK_N ({BLOCK_N})"
    )
    assert 0 <= dcp_rank < dcp_world_size
    assert out is not None, "sparse_kv_indices_buffer (out) is required"

    num_batch = global_kv_indptr.shape[0] - 1

    qo_indptr_c = qo_indptr.contiguous()
    global_kv_indptr_c = global_kv_indptr.contiguous()
    block_table_c = block_table.contiguous()
    token_indices_c = token_indices.contiguous()

    ti_stride0, ti_stride1 = token_indices_c.stride()
    bt_stride0, bt_stride1 = block_table_c.stride()
    grid = (num_batch,)

    # Pass 1: per-request count of owned top-k positions.
    counts = owned_counts[:num_batch]
    _count_owned_dcp_kernel[grid](
        qo_indptr_c,
        global_kv_indptr_c,
        token_indices_c,
        counts,
        dcp_rank,
        dcp_world_size,
        NUM_TOPK_TOKENS,
        BLOCK_N,
        ti_stride0,
        ti_stride1,
    )

    # Exclusive cumsum -> compacted offsets. Written in place so the caller's
    # tensor (and anything already holding a view of it) sees the update.
    # dtype=int32 keeps the accumulation in int32 (torch would promote integral
    # cumsum to int64 by default, which the kernels' int32 pointers reject).
    # zero_() rather than `out_kv_indptr[0] = 0`: assigning a Python scalar goes
    # through a host->device copy, which HIP rejects while a graph is capturing
    # (hipErrorStreamCaptureUnsupported). Everything here must stay device-side.
    out_kv_indptr[:1].zero_()
    torch.cumsum(
        counts, dim=0, dtype=torch.int32, out=out_kv_indptr[1 : num_batch + 1]
    )

    # Pass 2: write the owned slots packed to the front of each region.
    _compact_filter_dcp_kernel[grid](
        qo_indptr_c,
        global_kv_indptr_c,
        out_kv_indptr,
        block_table_c,
        token_indices_c,
        out,
        dcp_rank,
        dcp_world_size,
        block_size,
        NUM_TOPK_TOKENS,
        BLOCK_N,
        ti_stride0,
        ti_stride1,
        bt_stride0,
        bt_stride1,
    )
    return out


@triton.jit
def _convert_req_index_to_global_index_dsa_prefill_kernel(
    dsa_qo_indptr,  # int32 [num_tokens + 1]
    dsa_kv_indptr,  # int32 [num_tokens + 1]
    token_to_seq_idxs,  # int32 [num_tokens]
    topk_indices,  # int32 [num_tokens, NUM_TOPK_TOKENS]
    block_table,  # int32 [num_req, max_num_blocks_per_req]
    cu_seqlens_q,  # int32 [num_tokens + 1]
    out_kv_indices,  # int32
    # shapes (compile-time where possible)
    NUM_TOPK_TOKENS: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    BLOCK_N: tl.constexpr,  # tile width along columns
    # strides (in elements)
    ti_stride0: tl.int64,  # topk_indices stride 0
    ti_stride1: tl.constexpr,  # topk_indices stride 1
    bt_stride0: tl.int64,  # block_table stride 0
    bt_stride1: tl.constexpr,  # block_table stride 1
):
    token_id = tl.program_id(0)
    tile_id = tl.program_id(1)

    col_id = tile_id * BLOCK_N + tl.arange(0, BLOCK_N)

    req_id = tl.load(token_to_seq_idxs + token_id)  # int32

    kv_start = tl.load(dsa_kv_indptr + token_id)
    kv_end = tl.load(dsa_kv_indptr + token_id + 1)
    kv_len = kv_end - kv_start

    # Load token indices for this tile
    indice = tl.load(
        topk_indices + token_id * ti_stride0 + col_id * ti_stride1
    )  # int32
    pre_seqlens_q = tl.load(cu_seqlens_q + req_id)

    seq_token_idx = indice - pre_seqlens_q
    block_id = seq_token_idx // PAGE_SIZE
    inblock_offset = seq_token_idx % PAGE_SIZE

    # Guard block_table access
    store_mask = (col_id < kv_len) & (col_id < NUM_TOPK_TOKENS)
    valid_mask = store_mask & (indice >= 0)
    physical_block = tl.load(
        block_table + req_id * bt_stride0 + block_id * bt_stride1,
        mask=valid_mask,
        other=-1,
    )
    out_val = tl.where(valid_mask, physical_block * PAGE_SIZE + inblock_offset, -1)

    # Store results
    out_ptr_ij = out_kv_indices + kv_start + col_id
    tl.store(
        out_ptr_ij,
        out_val,
        mask=store_mask,
    )


def triton_convert_req_index_to_global_index_dsa_prefill(
    dsa_qo_indptr: torch.Tensor,  # int32 [num_tokens + 1]
    dsa_kv_indptr: torch.Tensor,  # int32 [num_tokens + 1]
    token_to_seq_idxs: torch.Tensor,  # int32 [num_tokens]
    topk_indices: torch.Tensor,  # int32 [num_tokens, NUM_TOPK_TOKENS]
    block_table: torch.Tensor,  # int32 [num_req, max_num_blocks_per_req]
    cu_seqlens_q: torch.Tensor,  # int32 [num_tokens + 1]
    # dsa_kv_indices: torch.Tensor,  # int32 [total_kv_seqlen]           -->>>     output for this kernel
    PAGE_SIZE: int = 1,
    NUM_TOPK_TOKENS: int = 2048,
    BLOCK_N: int = 1024,  # tile width along columns
    out: Optional[torch.Tensor] = None,
):

    assert topk_indices.shape[1] == NUM_TOPK_TOKENS
    assert NUM_TOPK_TOKENS % BLOCK_N == 0, (
        f"NUM_TOPK_TOKENS ({NUM_TOPK_TOKENS}) must be divisible by"
        f"BLOCK_N ({BLOCK_N})"
    )

    num_tokens = dsa_qo_indptr.shape[0] - 1
    tiles_per_row = NUM_TOPK_TOKENS // BLOCK_N

    total_out = num_tokens * NUM_TOPK_TOKENS
    if out is not None:
        new_kv_indices = out[:total_out]
    else:
        new_kv_indices = torch.empty(
            total_out, dtype=torch.int32, device=topk_indices.device
        )

    # Strides in elements
    ti_stride0, ti_stride1 = topk_indices.stride()
    bt_stride0, bt_stride1 = block_table.stride()

    grid = (num_tokens, tiles_per_row)

    _convert_req_index_to_global_index_dsa_prefill_kernel[grid](
        dsa_qo_indptr,
        dsa_kv_indptr,
        token_to_seq_idxs,
        topk_indices,
        block_table,
        cu_seqlens_q,
        new_kv_indices,
        # shapes / constexprs
        NUM_TOPK_TOKENS,
        PAGE_SIZE,
        BLOCK_N,
        # strides
        ti_stride0,
        ti_stride1,
        bt_stride0,
        bt_stride1,
    )
    return new_kv_indices


@triton.jit
def _count_owned_dcp_prefill_kernel(
    dsa_kv_indptr,  # int32 [num_tokens + 1] -- GLOBAL per-token candidate counts
    token_to_seq_idxs,  # int32 [num_tokens]
    topk_indices,  # int32 [num_tokens, NUM_TOPK_TOKENS] -- FLAT KV indices
    cu_seqlens_k,  # int32 [num_req + 1] -- per-seq base of the flat KV axis
    out_counts,  # int32 [num_tokens]
    DCP_RANK: tl.constexpr,
    DCP_WORLD: tl.constexpr,
    NUM_TOPK_TOKENS: tl.constexpr,
    BLOCK_N: tl.constexpr,
    ti_stride0: tl.int64,
    ti_stride1: tl.constexpr,
):
    """Pass 1 of the compacting DCP filter for sparse PREFILL: how many of the
    global top-k candidates does this rank own, per QUERY TOKEN?

    Differs from the decode twin (`_count_owned_dcp_kernel`) in two ways:
    the row unit is a query token rather than a request (prefill has many query
    tokens per request), and `topk_indices` holds a FLAT index into the
    concatenated KV of all sequences, so the within-sequence position -- which
    is what the round-robin owner is derived from -- is `indice - cu_seqlens_k[req]`.
    """
    token_id = tl.program_id(0)
    req_id = tl.load(token_to_seq_idxs + token_id)
    kv_start = tl.load(dsa_kv_indptr + token_id)
    kv_len = tl.load(dsa_kv_indptr + token_id + 1) - kv_start
    base = tl.load(cu_seqlens_k + req_id)

    count = 0
    for tile_start in range(0, NUM_TOPK_TOKENS, BLOCK_N):
        col_id = tile_start + tl.arange(0, BLOCK_N)
        col_valid = (col_id < kv_len) & (col_id < NUM_TOPK_TOKENS)
        indice = tl.load(
            topk_indices + token_id * ti_stride0 + col_id * ti_stride1,
            mask=col_valid,
            other=-1,
        )
        pos = indice - base  # position within the sequence
        owned = col_valid & (indice >= 0) & ((pos % DCP_WORLD) == DCP_RANK)
        count += tl.sum(owned.to(tl.int32))

    tl.store(out_counts + token_id, count)


@triton.jit
def _compact_filter_dcp_prefill_kernel(
    dsa_kv_indptr,  # int32 [num_tokens + 1] -- GLOBAL per-token candidate counts
    out_kv_indptr,  # int32 [num_tokens + 1] -- COMPACTED offsets (cumsum of pass 1)
    token_to_seq_idxs,  # int32 [num_tokens]
    topk_indices,  # int32 [num_tokens, NUM_TOPK_TOKENS] -- FLAT KV indices
    cu_seqlens_k,  # int32 [num_req + 1]
    block_table,  # int32 [num_req, max_num_blocks_per_req] -- logical(global) blocks
    out_kv_indices,  # int32 [>= out_kv_indptr[-1]]
    DCP_RANK: tl.constexpr,
    DCP_WORLD: tl.constexpr,
    PAGE_SIZE: tl.constexpr,  # runner (physical) block size
    NUM_TOPK_TOKENS: tl.constexpr,
    BLOCK_N: tl.constexpr,
    ti_stride0: tl.int64,
    ti_stride1: tl.constexpr,
    bt_stride0: tl.int64,
    bt_stride1: tl.constexpr,
):
    """Pass 2 for sparse PREFILL: keep only this rank's owned candidates, map
    them through the round-robin (virtual-block) layout, and pack them to the
    front of the token's region.

        vbs  = PAGE_SIZE * W
        slot = block_table[req, pos // vbs] * PAGE_SIZE + (pos % vbs) // W

    Same rationale as the decode twin: no `-1` holes (they break aiter's lse
    path, §4.3/§4.4) and compaction is order-preserving so the fp accumulation
    order is deterministic. See DCP/DCP_Sparse_MLA.md §5.5.5 (P2).
    """
    token_id = tl.program_id(0)
    req_id = tl.load(token_to_seq_idxs + token_id)
    kv_start = tl.load(dsa_kv_indptr + token_id)
    kv_len = tl.load(dsa_kv_indptr + token_id + 1) - kv_start
    base = tl.load(cu_seqlens_k + req_id)
    out_kv_start = tl.load(out_kv_indptr + token_id)

    vbs = PAGE_SIZE * DCP_WORLD
    written = 0
    for tile_start in range(0, NUM_TOPK_TOKENS, BLOCK_N):
        col_id = tile_start + tl.arange(0, BLOCK_N)
        col_valid = (col_id < kv_len) & (col_id < NUM_TOPK_TOKENS)
        indice = tl.load(
            topk_indices + token_id * ti_stride0 + col_id * ti_stride1,
            mask=col_valid,
            other=-1,
        )
        pos = indice - base
        idx_valid = col_valid & (indice >= 0) & ((pos % DCP_WORLD) == DCP_RANK)

        block_id = pos // vbs
        inblock_offset = (pos % vbs) // DCP_WORLD
        physical_block = tl.load(
            block_table + req_id * bt_stride0 + block_id * bt_stride1,
            mask=idx_valid,
            other=0,
        )
        slot = physical_block * PAGE_SIZE + inblock_offset

        owned_i32 = idx_valid.to(tl.int32)
        dst = written + tl.cumsum(owned_i32, axis=0) - owned_i32
        tl.store(out_kv_indices + out_kv_start + dst, slot, mask=idx_valid)
        written += tl.sum(owned_i32)

    # A row this rank owns nothing of is left EMPTY (zero-length region). That is
    # both legal and correct for mla_decode_fwd: it writes lse = -inf, which
    # cp_lse_ag_out_rs turns into a zero weight. Its `o` comes out NaN (0/0), so
    # the caller zeroes those rows -- see _forward_prefill_mla. No dummy candidate
    # is injected; the attention never sees fabricated KV. (§5.5.8 / §5.7)


def triton_filter_and_convert_dcp_index_prefill(
    dsa_kv_indptr: torch.Tensor,  # int32 [num_tokens + 1] GLOBAL counts
    token_to_seq_idxs: torch.Tensor,  # int32 [num_tokens]
    topk_indices: torch.Tensor,  # int32 [num_tokens, NUM_TOPK_TOKENS] FLAT indices
    cu_seqlens_k: torch.Tensor,  # int32 [num_req + 1]
    block_table: torch.Tensor,  # int32 [num_req, max_num_blocks_per_req] logical
    dcp_rank: int,
    dcp_world_size: int,
    block_size: int,  # runner (physical) block size == PAGE_SIZE
    out_kv_indptr: torch.Tensor,  # int32 [num_tokens + 1] COMPACTED, written here
    owned_counts: torch.Tensor,  # int32 [>= num_tokens] scratch for pass 1
    NUM_TOPK_TOKENS: int = 2048,
    BLOCK_N: int = 128,
    out: Optional[torch.Tensor] = None,
):
    """Sparse-PREFILL twin of ``triton_filter_and_convert_dcp_index``.

    The decode version keys on requests (qlen==1); prefill has one row per query
    token and its ``topk_indices`` are flat KV indices rather than
    within-sequence positions. Everything else -- two passes, in-place int32
    cumsum, order-preserving compaction -- is identical, and the same
    layer-scoped buffers are reused (they are sized ``max_num_batched_tokens``,
    which bounds the prefill token count too).
    """
    assert topk_indices.dtype == torch.int32
    assert topk_indices.shape[1] == NUM_TOPK_TOKENS
    assert NUM_TOPK_TOKENS % BLOCK_N == 0, (
        f"NUM_TOPK_TOKENS ({NUM_TOPK_TOKENS}) must be divisible by"
        f"BLOCK_N ({BLOCK_N})"
    )
    assert 0 <= dcp_rank < dcp_world_size
    assert out is not None, "sparse_kv_indices_buffer (out) is required"

    num_tokens = dsa_kv_indptr.shape[0] - 1

    dsa_kv_indptr_c = dsa_kv_indptr.contiguous()
    token_to_seq_idxs_c = token_to_seq_idxs.contiguous()
    topk_indices_c = topk_indices.contiguous()
    cu_seqlens_k_c = cu_seqlens_k.contiguous()
    block_table_c = block_table.contiguous()

    ti_stride0, ti_stride1 = topk_indices_c.stride()
    bt_stride0, bt_stride1 = block_table_c.stride()
    grid = (num_tokens,)

    counts = owned_counts[:num_tokens]
    _count_owned_dcp_prefill_kernel[grid](
        dsa_kv_indptr_c,
        token_to_seq_idxs_c,
        topk_indices_c,
        cu_seqlens_k_c,
        counts,
        dcp_rank,
        dcp_world_size,
        NUM_TOPK_TOKENS,
        BLOCK_N,
        ti_stride0,
        ti_stride1,
    )

    # Same device-side-only cumsum as the decode path (see its comment for why
    # zero_() and dtype=torch.int32 are required).
    out_kv_indptr[:1].zero_()
    torch.cumsum(
        counts, dim=0, dtype=torch.int32, out=out_kv_indptr[1 : num_tokens + 1]
    )

    _compact_filter_dcp_prefill_kernel[grid](
        dsa_kv_indptr_c,
        out_kv_indptr,
        token_to_seq_idxs_c,
        topk_indices_c,
        cu_seqlens_k_c,
        block_table_c,
        out,
        dcp_rank,
        dcp_world_size,
        block_size,
        NUM_TOPK_TOKENS,
        BLOCK_N,
        ti_stride0,
        ti_stride1,
        bt_stride0,
        bt_stride1,
    )
    return out


@triton.jit
def _gather_kv_indices_sparse_kernel(
    sparse_kv_indptr,
    token_to_seq_idxs,
    topk_indices,
    kv_indices,
    kv_indptr,
    out_kv_indices,
    NUM_TOPK_TOKENS: tl.constexpr,
    BLOCK_N: tl.constexpr,
    ti_stride0: tl.int64,
    ti_stride1: tl.constexpr,
):
    token_id = tl.program_id(0)
    tile_id = tl.program_id(1)
    col_id = tile_id * BLOCK_N + tl.arange(0, BLOCK_N)

    req_id = tl.load(token_to_seq_idxs + token_id)

    out_start = tl.load(sparse_kv_indptr + token_id)
    out_end = tl.load(sparse_kv_indptr + token_id + 1)
    kv_len = out_end - out_start

    pos = tl.load(topk_indices + token_id * ti_stride0 + col_id * ti_stride1)

    kv_base = tl.load(kv_indptr + req_id)
    kv_end = tl.load(kv_indptr + req_id + 1)
    req_kv_len = kv_end - kv_base

    store_mask = (col_id < kv_len) & (col_id < NUM_TOPK_TOKENS)
    valid_mask = store_mask & (pos >= 0) & (pos < req_kv_len)

    out_val = tl.load(
        kv_indices + kv_base + pos,
        mask=valid_mask,
        other=0,
    )

    tl.store(
        out_kv_indices + out_start + col_id,
        out_val,
        mask=store_mask,
    )


def triton_gather_kv_indices_sparse(
    sparse_kv_indptr: torch.Tensor,
    token_to_seq_idxs: torch.Tensor,
    topk_indices: torch.Tensor,
    kv_indices: torch.Tensor,
    kv_indptr: torch.Tensor,
    NUM_TOPK_TOKENS: int = 2048,
    BLOCK_N: int = 1024,
    out: Optional[torch.Tensor] = None,
):
    assert topk_indices.shape[1] == NUM_TOPK_TOKENS
    assert NUM_TOPK_TOKENS % BLOCK_N == 0

    # MTP decode can carry metadata tensors padded to a larger query layout
    # than the number of rows produced by the current indexer call. Keep all
    # per-token inputs aligned to the actual valid intersection before launch;
    # otherwise the kernel may read past topk_indices.
    num_tokens = min(
        token_to_seq_idxs.shape[0],
        topk_indices.shape[0],
        sparse_kv_indptr.shape[0] - 1,
    )
    sparse_kv_indptr = sparse_kv_indptr[: num_tokens + 1]
    token_to_seq_idxs = token_to_seq_idxs[:num_tokens]
    topk_indices = topk_indices[:num_tokens]
    tiles_per_row = NUM_TOPK_TOKENS // BLOCK_N

    total_out = num_tokens * NUM_TOPK_TOKENS
    if out is not None:
        out_buf = out[:total_out]
    else:
        out_buf = torch.empty(total_out, dtype=torch.int32, device=topk_indices.device)

    ti_stride0, ti_stride1 = topk_indices.stride()
    grid = (num_tokens, tiles_per_row)

    _gather_kv_indices_sparse_kernel[grid](
        sparse_kv_indptr,
        token_to_seq_idxs,
        topk_indices,
        kv_indices,
        kv_indptr,
        out_buf,
        NUM_TOPK_TOKENS,
        BLOCK_N,
        ti_stride0,
        ti_stride1,
    )
    return out_buf
