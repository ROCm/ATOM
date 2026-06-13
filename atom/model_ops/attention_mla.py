# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import logging
from dataclasses import dataclass
from functools import partial as functools_partial
from typing import Optional

import torch
import triton
import triton.language as tl
from aiter import (
    QuantType,
    concat_and_cache_mla,
    concat_and_cache_mla_seg,
    dtypes,
    flash_attn_varlen_func,
    fused_qk_rope_concat_and_cache_mla,
    fused_qk_rope_concat_and_cache_mla_seg,
    get_hip_quant,
)
from aiter.jit.utils.chip_info import get_gfx
from aiter.dist.parallel_state import get_dp_group
from aiter.mla import mla_decode_fwd, mla_prefill_fwd
from aiter.ops.triton.gather_kv_b_proj import gather_kv_b_proj
from atom.config import get_current_atom_config
from atom.model_ops.linear import use_triton_gemm
from atom.model_ops.utils import get_and_maybe_dequant_weights
from atom.utils import envs
from atom.utils.decorators import mark_trace
from atom.utils.forward_context import (
    AttentionMetaData,
    ForwardContext,
    get_forward_context,
)
from torch import nn

from aiter.ops.triton.batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant import (  # noqa: E501 # isort: skip
    batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant as _aiter_triton_fp8_bmm,
)

concat_and_cache_mla = mark_trace(
    concat_and_cache_mla, prefix="kv_cache", torch_compile=False
)
concat_and_cache_mla_seg = mark_trace(
    concat_and_cache_mla_seg, prefix="kv_cache_seg", torch_compile=False
)
fused_qk_rope_concat_and_cache_mla = mark_trace(
    fused_qk_rope_concat_and_cache_mla, prefix="rope_and_kv_cache", torch_compile=False
)
fused_qk_rope_concat_and_cache_mla_seg = mark_trace(
    fused_qk_rope_concat_and_cache_mla_seg, prefix="rope_and_kv_cache", torch_compile=False
)
mla_prefill_fwd = mark_trace(mla_prefill_fwd, prefix="mla_prefill", torch_compile=False)
mla_decode_fwd = mark_trace(mla_decode_fwd, prefix="mla_decode", torch_compile=False)

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

if use_triton_gemm():
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
        self.head_repeat_factor = self.padded_num_heads // num_heads
        if self.head_repeat_factor > 1:
            assert self.padded_num_heads % num_heads == 0, (
                f"Padded head count ({self.padded_num_heads}) must be divisible "
                f"by num_heads ({num_heads}) for head repeat"
            )
            if not getattr(MLAAttention, "_head_repeat_logged", False):
                MLAAttention._head_repeat_logged = True
                logger.info(
                    f"MLA head repeat enabled: {num_heads} -> {self.padded_num_heads} "
                    f"(repeat factor {self.head_repeat_factor})"
                )

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
        self.is_sparse_mla = mla_modules.indexer is not None
        self.topk_tokens = (
            mla_modules.indexer.topk_tokens if mla_modules.indexer is not None else None
        )
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
        # One-shot guard so the seg-layout constraint checks run once per layer
        # instead of on every forward step.
        self._seg_layout_validated = False

    def _validate_seg_layout(self, attn_metadata, positions) -> None:
        """Validate the static constraints required by the segmented MLA
        kernels (fused_qk_rope_concat_and_cache_mla_seg / concat_and_cache_mla_seg
        / gfx1250 mla_decode_fwd). Runs once per layer."""
        if self._seg_layout_validated:
            return
        page_size = get_current_atom_config().kv_cache_block_size
        assert page_size == _MLA_SEG_PAGE_SIZE, (
            "seg MLA kernels require kv_cache_block_size == "
            f"{_MLA_SEG_PAGE_SIZE}, got {page_size}"
        )
        assert self.kv_cache_dtype.startswith("fp8"), (
            "fused seg MLA write/decode path requires an fp8 kv cache "
            f"(kv_cache_dtype={self.kv_cache_dtype})"
        )
        assert attn_metadata.dtype_q == dtypes.fp8, (
            f"seg q_out must be fp8, got dtype_q={attn_metadata.dtype_q}"
        )
        assert (
            self.kv_lora_rank == _MLA_SEG_KV_LORA_RANK
            and self.qk_rope_head_dim == _MLA_SEG_PE_DIM
        ), (
            "fused seg kernel is fixed to kv_lora_rank="
            f"{_MLA_SEG_KV_LORA_RANK}, qk_rope_head_dim={_MLA_SEG_PE_DIM}; got "
            f"{self.kv_lora_rank}, {self.qk_rope_head_dim}"
        )
        cos_dim = self.rotary_emb.cos_cache.shape[-1]
        assert cos_dim == self.qk_rope_head_dim // 2, (
            f"cos/sin cache last dim must be {self.qk_rope_head_dim // 2} "
            f"(reuse_freqs_front_part), got {cos_dim}"
        )
        assert positions.dtype == torch.int64, (
            f"positions must be int64 for the seg kernel, got {positions.dtype}"
        )
        assert attn_metadata.slot_mapping.dtype == torch.int64, (
            "slot_mapping must be int64 for the seg kernel, got "
            f"{attn_metadata.slot_mapping.dtype}"
        )
        self._seg_layout_validated = True

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

    def _assert_seg_write(
        self, kv_cache_seg: torch.Tensor, slot_mapping: torch.Tensor
    ) -> None:
        """Per-step runtime asserts for the seg KV-cache write path (operates on
        the page-level seg view from _seg_kv_cache_view). Gated by
        ATOM_DEBUG_MLA_SEG. Catches layout/stride/slot_mapping mistakes that
        would otherwise silently scatter the cache to wrong offsets."""
        if not envs.ATOM_DEBUG_MLA_SEG:
            return
        entry = self.kv_lora_rank + self.qk_rope_head_dim
        num_blocks = kv_cache_seg.shape[0]
        assert kv_cache_seg.is_contiguous(), "seg kv_cache view must be contiguous"
        assert (
            kv_cache_seg.dim() == 2
            and kv_cache_seg.shape[1] == entry * _MLA_SEG_PAGE_SIZE
        ), (
            f"seg kv_cache view shape {tuple(kv_cache_seg.shape)} != "
            f"[num_blocks, {entry * _MLA_SEG_PAGE_SIZE}]"
        )
        assert kv_cache_seg.stride(0) == entry * _MLA_SEG_PAGE_SIZE, (
            f"seg kv_cache block stride {kv_cache_seg.stride(0)} != "
            f"page_size*entry {entry * _MLA_SEG_PAGE_SIZE}"
        )
        slot = slot_mapping.flatten()
        valid = slot[slot >= 0]
        if valid.numel() > 0:
            max_slot = int(valid.max().item())
            assert max_slot < num_blocks * _MLA_SEG_PAGE_SIZE, (
                f"seg slot_mapping max {max_slot} >= "
                f"num_blocks*page_size {num_blocks * _MLA_SEG_PAGE_SIZE}"
            )

    def _assert_seg_decode(
        self,
        kv_buffer_4d: torch.Tensor,
        q: torch.Tensor,
        kv_indptr: torch.Tensor,
        kv_indices: torch.Tensor,
    ) -> None:
        """Per-step runtime asserts for the seg decode inputs. Gated by
        ATOM_DEBUG_MLA_SEG."""
        if not envs.ATOM_DEBUG_MLA_SEG:
            return
        entry = self.kv_lora_rank + self.qk_rope_head_dim
        num_blocks = kv_buffer_4d.shape[0]
        assert kv_buffer_4d.shape[1] == _MLA_SEG_PAGE_SIZE, (
            f"seg decode kv_buffer page_size {kv_buffer_4d.shape[1]} != "
            f"{_MLA_SEG_PAGE_SIZE}"
        )
        assert kv_buffer_4d.shape[-1] == entry, (
            f"seg decode kv_buffer last dim {kv_buffer_4d.shape[-1]} != {entry}"
        )
        assert q.shape[-1] == entry, (
            f"seg decode q last dim {q.shape[-1]} != {entry}"
        )
        # q must keep the padded per-head row stride (768) the asm kernel reads.
        assert q.stride(-2) == _MLA_Q_OUT_PADDED_DIM, (
            f"seg decode q per-head stride {q.stride(-2)} != "
            f"{_MLA_Q_OUT_PADDED_DIM} (padded row stride not preserved)"
        )
        # kv_indptr is page-level: total referenced pages == kv_indices length,
        # and every page id must be a valid physical block.
        total_pages = int(kv_indptr[-1].item())
        assert total_pages <= kv_indices.shape[0], (
            f"seg decode kv_indptr[-1] {total_pages} > kv_indices len "
            f"{kv_indices.shape[0]}"
        )
        if total_pages > 0:
            max_blk = int(kv_indices[:total_pages].max().item())
            assert max_blk < num_blocks, (
                f"seg decode kv_indices max block {max_blk} >= num_blocks "
                f"{num_blocks}"
            )
        # One-shot per-layer diagnostic of the actual runtime config so we can
        # compare against the asm-supported variants (nhead/decode_qlen) and the
        # scale story (triton casts fp8 q to bf16 without applying q_scale, while
        # mla_decode_fwd applies q_scale/kv_scale here).
        if not getattr(self, "_seg_decode_logged", False):
            self._seg_decode_logged = True
            logger.info(
                "SEG-DECODE layer=%d nhead=%d padded_nh=%d hrf=%d B=%d "
                "q.shape=%s q.stride=%s q.dtype=%s kv.shape=%s kv.dtype=%s "
                "kv_indptr[:4]=%s kv_indices[:4]=%s q_scale=%s kv_scale=%s",
                self.layer_num,
                self.num_heads,
                self.padded_num_heads,
                self.head_repeat_factor,
                q.shape[0],
                tuple(q.shape),
                tuple(q.stride()),
                str(q.dtype),
                tuple(kv_buffer_4d.shape),
                str(kv_buffer_4d.dtype),
                kv_indptr[:4].tolist(),
                kv_indices[: min(4, kv_indices.shape[0])].tolist(),
                None if self._q_scale is None else self._q_scale.flatten()[:1].tolist(),
                None if self._k_scale is None else self._k_scale.flatten()[:1].tolist(),
            )

    @staticmethod
    def _assert_seg_decode_output(o: torch.Tensor) -> None:
        """Assert the seg decode kernel produced a fully-written, finite output.
        Gated by ATOM_DEBUG_MLA_SEG. A NaN/inf here points at the asm kernel not
        writing all of `o` (e.g. unsupported nhead/decode_qlen variant)."""
        if not envs.ATOM_DEBUG_MLA_SEG:
            return
        assert torch.isfinite(o).all(), (
            "seg decode output `o` contains NaN/inf (asm likely did not write "
            "the whole buffer for this nhead/decode_qlen variant)"
        )
        # Flag fully-zero head rows: with zero-init `o`, an all-zero row means
        # the asm did not write that (head, token) -> partial write / variant
        # mismatch. Logged once.
        if not getattr(MLAAttention, "_seg_zero_row_logged", False):
            zero_rows = int((o.reshape(o.shape[0], o.shape[1], -1).abs().sum(-1) == 0).sum().item())
            if zero_rows > 0:
                MLAAttention._seg_zero_row_logged = True
                logger.warning(
                    "SEG-DECODE output has %d all-zero (head,token) rows out of "
                    "%d -> asm did not write them (partial write / variant "
                    "mismatch)",
                    zero_rows,
                    o.shape[0] * o.shape[1],
                )

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
    ) -> None:
        weight = self.kv_b_proj.weight
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
        k_workspace = chunk_meta.k_workspace
        v_workspace = chunk_meta.v_workspace
        chunked_out: Optional[torch.Tensor] = None
        chunked_lse: Optional[torch.Tensor] = None
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
                and get_gfx() != "gfx1250"
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
    ) -> torch.Tensor:
        assert attn_metadata is not None
        B = q.shape[0]

        if self.head_repeat_factor > 1:
            q = q.repeat_interleave(self.head_repeat_factor, dim=1)

        # q arrives with a padded per-head row stride (_MLA_Q_OUT_PADDED_DIM);
        # slice back to the logical kv_lora_rank + qk_rope_head_dim columns. The
        # slice keeps the padded row stride, which the asm kernel expects. The
        # triton path uses an unpadded 576-wide q_out, so no slicing is needed.
        if not self.use_triton_mla:
            q = q[..., : self.kv_lora_rank + self.qk_rope_head_dim]

        # DEBUG(seg): zero-init instead of empty so any region the decode asm
        # does not write shows up as 0 rather than garbage (isolates
        # uninitialized-read bugs in the seg pass).
        o = torch.zeros(
            B,
            self.padded_num_heads,
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
            max_q_len = 1

        if kv_c_and_k_pe_cache.numel() > 0:
            page_size = get_current_atom_config().kv_cache_block_size
            if self.kv_cache_dtype.startswith("fp8"):
                mla_decode_fwd(
                    q,
                    kv_c_and_k_pe_cache.view(-1, page_size, 1, q.shape[-1]),
                    o,
                    paged_cu_seqlens_q,
                    paged_kv_indptr,
                    paged_kv_indices,
                    kv_last_page_lens,
                    max_q_len,
                    page_size=page_size,
                    sm_scale=self.scale,
                    q_scale=self._q_scale,
                    kv_scale=self._k_scale,
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

        if self.head_repeat_factor > 1:
            o = o[:, :: self.head_repeat_factor, :].contiguous()

        return self._v_up_proj_and_o_proj(o)

    def _forward_decode(
        self,
        q: torch.Tensor,
        kv_c_and_k_pe_cache: torch.Tensor,
        attn_metadata: AttentionMetaData,
    ) -> torch.Tensor:
        assert kv_c_and_k_pe_cache.numel() > 0
        assert attn_metadata is not None
        B = q.shape[0]

        if self.head_repeat_factor > 1:
            q = q.repeat_interleave(self.head_repeat_factor, dim=1)

        # q arrives with a padded per-head row stride (_MLA_Q_OUT_PADDED_DIM);
        # slice back to the logical kv_lora_rank + qk_rope_head_dim columns. The
        # slice keeps the padded row stride, which the asm kernel expects. The
        # triton path uses an unpadded 576-wide q_out, so no slicing is needed.
        if not self.use_triton_mla:
            q = q[..., : self.kv_lora_rank + self.qk_rope_head_dim]

        # DEBUG(seg): zero-init instead of empty so any region the decode asm
        # does not write shows up as 0 rather than garbage (isolates
        # uninitialized-read bugs in the seg pass).
        o = torch.zeros(
            B,
            self.padded_num_heads,
            self.kv_lora_rank,
            dtype=self.dtype,
            device=q.device,
        )

        if hasattr(attn_metadata, "triton_block_table"):
            from aiter.ops.triton.attention.mla_decode import decode_attention_fwd

            k_buffer = kv_c_and_k_pe_cache.unsqueeze(2)
            v_buffer = k_buffer[..., : self.kv_lora_rank]
            # The KV cache is bound flattened to a per-token layout
            # ([num_blocks*block_size, 1, 1, 576]), so k_buffer.shape[1] is 1,
            # not the real paged block size. The dense block_table stores page
            # ids at block granularity, so PAGE_SIZE must be the real KV cache
            # block size for the kernel's page// and intra-page% addressing.
            page_size = get_current_atom_config().kv_cache_block_size
            logger.info("triton_mla decode: page_size=%d", page_size)

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
                # Must match triton_attn_logits.shape[2] allocated in
                # TritonMLAMetadataBuilder (num_kv_splits=4); the kernel asserts
                # num_kv_splits == attn_logits.shape[2].
                attn_metadata.triton_attn_logits.shape[2],  # num_kv_splits
                self.scale,
                page_size,
                k_scale=self._k_scale,
                v_scale=self._k_scale,
            )

            if envs.ATOM_DUMP_MLA_DECODE and not self.is_sparse_mla:
                # Dump the token-major (interleaved) KV cache + this layer's
                # decode inputs so op_tests/test_mla_decode_replay.py can replay
                # them through AITER mla_decode_fwd vs the Triton reference.
                from atom.utils.mla_decode_dump import dump_decode_mla

                dump_decode_mla(
                    layer_num=self.layer_num,
                    q=q,
                    kv_buffer_view=kv_c_and_k_pe_cache.view(
                        -1, page_size, 1, q.shape[-1]
                    ),
                    o=o,
                    qo_indptr=attn_metadata.cu_seqlens_q,
                    kv_indptr=attn_metadata.kv_indptr,
                    kv_indices=attn_metadata.kv_indices,
                    kv_last_page_lens=attn_metadata.kv_last_page_lens,
                    max_q_len=attn_metadata.max_seqlen_q,
                    page_size=page_size,
                    sm_scale=self.scale,
                    q_scale=self._q_scale,
                    kv_scale=self._k_scale,
                    num_kv_splits=1,
                    context_lens=getattr(attn_metadata, "context_lens", None),
                    num_heads=self.num_heads,
                    padded_num_heads=self.padded_num_heads,
                    kv_lora_rank=self.kv_lora_rank,
                    qk_rope_head_dim=self.qk_rope_head_dim,
                    v_head_dim=self.v_head_dim,
                    head_repeat_factor=self.head_repeat_factor,
                    is_sparse_mla=self.is_sparse_mla,
                    kv_layout="interleaved",
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
                    paged_kv_indptr = attn_metadata.sparse_kv_indptr
                    paged_kv_indices = self.sparse_kv_indices_buffer

            dp_size = get_dp_group().world_size
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

            page_size = get_current_atom_config().kv_cache_block_size
            seg_kv_buffer_4d = kv_buffer.view(-1, page_size, 1, q.shape[-1])
            self._assert_seg_decode(
                seg_kv_buffer_4d, q, paged_kv_indptr, paged_kv_indices
            )
            mla_decode_fwd(
                q,
                seg_kv_buffer_4d,
                o,
                paged_cu_seqlens_q,
                paged_kv_indptr,
                paged_kv_indices,
                paged_kv_last_page_lens,
                max_q_len,
                page_size=page_size,
                num_kv_splits=1,
                sm_scale=self.scale,
                work_meta_data=work_meta_data,
                work_indptr=work_indptr,
                work_info_set=work_info_set,
                reduce_indptr=reduce_indptr,
                reduce_final_map=reduce_final_map,
                reduce_partial_map=reduce_partial_map,
                q_scale=self._q_scale,
                kv_scale=self._k_scale,
            )

            self._assert_seg_decode_output(o)

            if envs.ATOM_DUMP_MLA_DECODE and not self.is_sparse_mla:
                from atom.utils.mla_decode_dump import dump_decode_mla

                dump_decode_mla(
                    layer_num=self.layer_num,
                    q=q,
                    kv_buffer_view=kv_buffer.view(-1, page_size, 1, q.shape[-1]),
                    o=o,
                    qo_indptr=paged_cu_seqlens_q,
                    kv_indptr=paged_kv_indptr,
                    kv_indices=paged_kv_indices,
                    kv_last_page_lens=paged_kv_last_page_lens,
                    max_q_len=max_q_len,
                    page_size=page_size,
                    sm_scale=self.scale,
                    q_scale=self._q_scale,
                    kv_scale=self._k_scale,
                    num_kv_splits=1,
                    context_lens=getattr(attn_metadata, "context_lens", None),
                    num_heads=self.num_heads,
                    padded_num_heads=self.padded_num_heads,
                    kv_lora_rank=self.kv_lora_rank,
                    qk_rope_head_dim=self.qk_rope_head_dim,
                    v_head_dim=self.v_head_dim,
                    head_repeat_factor=self.head_repeat_factor,
                    is_sparse_mla=self.is_sparse_mla,
                    kv_layout="seg",
                )

        if self.head_repeat_factor > 1:
            o = o[:, :: self.head_repeat_factor, :].contiguous()

        return self._v_up_proj_and_o_proj(o)

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
        if forward_context.context.is_dummy_run:
            output_shape = list(q.shape)
            atom_config = get_current_atom_config()
            output_shape[-1] = atom_config.hf_config.hidden_size
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
                if self.use_triton_mla:
                    # Triton MLA path: original interleaved per-token layout.
                    concat_and_cache_mla(
                        k_nope,
                        k_rope.squeeze(1),
                        kv_cache,
                        attn_metadata.slot_mapping.flatten(),
                        kv_cache_dtype=self.kv_cache_dtype,
                        scale=self._k_scale,
                    )
                else:
                    self._validate_seg_layout(attn_metadata, positions)
                    # Write the KV cache in the segmented layout so the
                    # decode-phase mla_decode_fwd (which reads seg layout) sees a
                    # consistent cache for tokens written during prefill.
                    # kv_cache is flattened to
                    # [num_blocks, page_size*(kv_lora_rank + qk_rope_head_dim)] so
                    # the kernel derives page_size from stride(0).
                    kv_cache_seg = self._seg_kv_cache_view(kv_cache)
                    self._assert_seg_write(kv_cache_seg, attn_metadata.slot_mapping)
                    concat_and_cache_mla_seg(
                        k_nope,
                        k_rope.squeeze(1),
                        kv_cache_seg,
                        attn_metadata.slot_mapping.flatten(),
                        kv_cache_dtype=self.kv_cache_dtype,
                        scale=self._k_scale,
                    )

            if attn_metadata.has_cached:
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

            if self.use_triton_mla:
                # Triton MLA path: unpadded 576-wide q_out, interleaved cache.
                q_out = torch.empty(
                    (
                        q_nope.shape[0],
                        self.num_heads,
                        self.kv_lora_rank + self.qk_rope_head_dim,
                    ),
                    dtype=attn_metadata.dtype_q,
                    device=q_nope.device,
                )
            else:
                # Allocate q_out with a padded last dim so each head row has a
                # 768-byte stride (required by the gfx1250 decode asm). The kernel
                # only writes the first kv_lora_rank + qk_rope_head_dim columns;
                # the padding tail is left untouched and never read.
                # DEBUG(seg): zero-init the padded q_out so the unwritten
                # [576:768] tail is 0 rather than garbage (the decode asm reads
                # rows at a 768 stride). Helps isolate uninitialized-read bugs.
                q_out = torch.zeros(
                    (
                        q_nope.shape[0],
                        self.num_heads,
                        _MLA_Q_OUT_PADDED_DIM,
                    ),
                    dtype=attn_metadata.dtype_q,
                    device=q_nope.device,
                )
            if kv_cache.numel() > 0:
                if self.use_triton_mla:
                    fused_qk_rope_concat_and_cache_mla(
                        q_nope,
                        q_rope,
                        k_nope,
                        k_rope,
                        # Interleaved layout: [num_blocks, block_size, kv_lora+pe].
                        kv_cache.view(
                            kv_cache.shape[0],
                            -1,
                            self.kv_lora_rank + self.qk_rope_head_dim,
                        ),
                        q_out,
                        attn_metadata.slot_mapping,
                        self._k_scale,
                        self._q_scale,
                        positions,
                        self.rotary_emb.cos_cache,
                        self.rotary_emb.sin_cache,
                        is_neox=self.rotary_emb.is_neox_style,
                        is_nope_first=True,
                    )
                else:
                    self._validate_seg_layout(attn_metadata, positions)
                    kv_cache_seg = self._seg_kv_cache_view(kv_cache)
                    self._assert_seg_write(kv_cache_seg, attn_metadata.slot_mapping)
                    fused_qk_rope_concat_and_cache_mla_seg(
                        q_nope,
                        q_rope,
                        k_nope,
                        k_rope,
                        # Flat seg layout: [num_blocks, page_size*(kv_lora + pe)].
                        kv_cache_seg,
                        q_out,
                        attn_metadata.slot_mapping,
                        self._k_scale,
                        self._q_scale,
                        positions,
                        self.rotary_emb.cos_cache,
                        self.rotary_emb.sin_cache,
                        is_neox=self.rotary_emb.is_neox_style,
                    )
                # q_out = self.fused_kv_bmm(q, q_scale, k_nope, k_rope, positions, kv_cache, attn_metadata)

            if context.is_prefill:
                output = self._forward_prefill_mla(q_out, kv_cache, attn_metadata)
            else:
                output = self._forward_decode(q_out, kv_cache, attn_metadata)

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

        # Guard block_table access
        valid_mask = (indice_id < kv_len) & (indice_id < NUM_TOPK_TOKENS)
        out_val = tl.load(
            kv_indices + kv_start + tok,
            mask=valid_mask,
            other=0,
        )

        # Store results
        out_ptr_ij = out_kv_indices + out_kv_start + indice_id
        tl.store(
            out_ptr_ij,
            out_val,
            mask=valid_mask,
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

    # Exact 2D grid: tokens × column tiles
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
