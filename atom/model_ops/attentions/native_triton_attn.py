# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Triton-only attention backend for ATOM on gfx1201 (RDNA4 / RX 9070 XT).

Why this exists
---------------
The AITER package shipped in rocm/atom-dev:latest has prebuilt HIP .so files
only for gfx94x/95x. On gfx1201 the AITER paged-attention HIP modules fail
to load with "No compatible code objects found for: gfx1201" and SIGSEGV
the ModelRunner. This backend replaces them with JIT-compiled triton kernels
(aiter's triton paged-attention + an in-tree triton kv-cache write) that
build for gfx1201 at first call.

There is NO torch fallback in this build: the path raises a clear
RuntimeError if any required triton kernel is unavailable, instead of
silently falling back to a slow path that would also reintroduce
GPU->CPU syncs that break CUDAGraph capture.

Selection
---------
atom/utils/selector.py:get_attn_backend_cls routes here when
torch.cuda.get_device_properties(0).gcnArchName starts with 'gfx1201',
or when ATOM_NATIVE_TRITON_ATTN=1 is set explicitly.

KV cache layout (matches aiter's pa_decode triton kernel expectations)
----------------------------------------------------------------------
    runner.kv_cache : [2, num_layers, num_blocks, num_kv_heads, block_size, head_dim]
                     |--K-and-V--||--per-layer--||---paged storage in aiter format---|

Forward
-------
* Prefill: in-tree triton kv-cache write, then aiter triton
  context_attention_fwd (handles GQA internally).
* Decode: same triton kv-cache write, then a thin v1/v2 dispatcher
  around aiter's paged_attn_decode_v1 / paged_attn_decode_v2 that
  takes Python-float scales (the higher-level paged_attention_decode
  wrapper does .item() on every call -- a GPU->CPU sync that breaks
  CUDAGraph capture).
"""

from __future__ import annotations

import logging
import os
from typing import Optional, Type

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from atom.config import KVCacheTensor
from atom.model_engine.scheduler import ScheduledBatch
from atom.model_ops.attentions.backends import (
    AttentionBackend,
    AttentionImpl,
    CommonAttentionBuilder,
)
from atom.utils.forward_context import (
    AttentionMetaData,
    Context,
    get_forward_context,
    set_forward_context,
)

logger = logging.getLogger("atom")


def _is_gfx1201() -> bool:
    if not torch.cuda.is_available():
        return False
    name = torch.cuda.get_device_properties(0).gcnArchName or ""
    return name.startswith("gfx1201")


def use_native_triton_attn() -> bool:
    val = os.environ.get("ATOM_NATIVE_TRITON_ATTN", "").lower()
    if val in ("1", "true"):
        return True
    if val in ("0", "false"):
        return False
    return _is_gfx1201()


# ---------------------------------------------------------------------------
# Cached triton paged-attention decode kernel
# ---------------------------------------------------------------------------
_TRITON_PA_DECODE = None
_TRITON_TL_BF16 = None
_TRITON_PREFILL = None


def _get_triton_prefill():
    global _TRITON_PREFILL
    if _TRITON_PREFILL is None:
        try:
            from aiter.ops.triton.attention.prefill_attention import context_attention_fwd
            _TRITON_PREFILL = context_attention_fwd
        except Exception as e:
            logger.warning("triton context_attention_fwd unavailable: %s", e)
            _TRITON_PREFILL = False
    return _TRITON_PREFILL if _TRITON_PREFILL is not False else None


_PA_SEQ_PARTITION_SIZE = 1024  # mirrors aiter's wrapper constant


def _get_triton_pa_decode():
    """Return (pa_decode_dispatch, tl.bfloat16) or (None, None).

    pa_decode_dispatch mirrors aiter's ``paged_attention_decode`` v1/v2
    selection but takes Python float scales instead of 0-dim tensors --
    avoids the ``k_scale.item()`` / ``v_scale.item()`` sync that breaks
    CUDAGraph capture. BF16 KV path only (k_scale=v_scale=1.0).
    """
    global _TRITON_PA_DECODE, _TRITON_TL_BF16
    if _TRITON_PA_DECODE is None:
        try:
            from aiter.ops.triton.attention.pa_decode import (
                paged_attn_decode_v1,
                paged_attn_decode_v2,
            )
            import triton.language as tl

            def _dispatch(
                out, q, k_cache, v_cache,
                block_tables, seq_lens,
                max_seq_len, compute_type, num_kv_heads, scale,
            ):
                num_seqs = q.shape[0]
                num_q_heads = q.shape[1]
                max_num_partitions = (
                    max_seq_len + _PA_SEQ_PARTITION_SIZE - 1
                ) // _PA_SEQ_PARTITION_SIZE
                use_v1 = max_seq_len <= 8192 and (
                    max_num_partitions == 1 or num_seqs * num_q_heads > 512
                )
                if use_v1:
                    paged_attn_decode_v1(
                        out, q, k_cache, v_cache,
                        block_tables, seq_lens,
                        max_seq_len, compute_type, num_kv_heads,
                        scale, None, 1.0, 1.0,
                    )
                else:
                    paged_attn_decode_v2(
                        out, q, k_cache, v_cache,
                        block_tables, seq_lens,
                        max_seq_len, compute_type, num_kv_heads,
                        scale, None, 1.0, 1.0, max_num_partitions,
                    )

            _TRITON_PA_DECODE = _dispatch
            _TRITON_TL_BF16 = tl.bfloat16
        except Exception as e:
            logger.warning("triton paged_attn_decode unavailable: %s", e)
            _TRITON_PA_DECODE = False
    return (
        (_TRITON_PA_DECODE, _TRITON_TL_BF16)
        if _TRITON_PA_DECODE is not False
        else (None, None)
    )


# ---------------------------------------------------------------------------
# Backend
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# Triton KV-cache write kernel (skips -1 sentinels in-kernel; no Python sync)
# ---------------------------------------------------------------------------
import triton
import triton.language as tl


@triton.jit
def _kv_cache_write_kernel(
    K_NEW_PTR, V_NEW_PTR,                # [N, H, D] BF16 (or compatible)
    SLOT_PTR,                            # [N] int64
    K_CACHE_PTR, V_CACHE_PTR,            # [B, H, S, D] BF16
    new_stride_token, new_stride_head,
    cache_stride_block, cache_stride_head, cache_stride_within,
    N: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    S: tl.constexpr,
):
    """One program per token; copies the token's full (H, D) K/V slab into
    cache[block_id, :, within, :]. Slot < 0 sentinels are skipped."""
    token_idx = tl.program_id(0)
    if token_idx >= N:
        return
    slot = tl.load(SLOT_PTR + token_idx)
    if slot < 0:
        return
    block_id = slot // S
    within = slot % S

    head_offs = tl.arange(0, H)
    d_offs = tl.arange(0, D)

    new_off = (
        token_idx * new_stride_token
        + head_offs[:, None] * new_stride_head
        + d_offs[None, :]
    )
    cache_off = (
        block_id * cache_stride_block
        + head_offs[:, None] * cache_stride_head
        + within * cache_stride_within
        + d_offs[None, :]
    )

    k_vals = tl.load(K_NEW_PTR + new_off)
    v_vals = tl.load(V_NEW_PTR + new_off)
    tl.store(K_CACHE_PTR + cache_off, k_vals)
    tl.store(V_CACHE_PTR + cache_off, v_vals)


def _kv_cache_write_triton(
    k_cache: torch.Tensor,   # [B, H, S, D]
    v_cache: torch.Tensor,   # [B, H, S, D]
    slot_mapping: torch.Tensor,  # [N]
    k_new: torch.Tensor,     # [N, H, D]
    v_new: torch.Tensor,     # [N, H, D]
):
    N = slot_mapping.shape[0]
    if N == 0:
        return
    B, H, S, D = k_cache.shape
    # Triton requires power-of-two block sizes; H, D should be already.
    # k_new strides assume contiguous [N, H, D].
    k_new_c = k_new.contiguous() if not k_new.is_contiguous() else k_new
    v_new_c = v_new.contiguous() if not v_new.is_contiguous() else v_new
    slot_i64 = slot_mapping.to(torch.int64) if slot_mapping.dtype != torch.int64 else slot_mapping

    new_stride = k_new_c.stride()
    cache_stride = k_cache.stride()
    grid = (N,)
    _kv_cache_write_kernel[grid](
        k_new_c, v_new_c,
        slot_i64,
        k_cache, v_cache,
        new_stride[0], new_stride[1],
        cache_stride[0], cache_stride[1], cache_stride[2],
        N=N, H=H, D=D, S=S,
    )

class NativeTritonBackend(AttentionBackend):
    """AITER-free attention backend (torch + selectively triton)."""

    @staticmethod
    def get_name() -> str:
        return "NATIVE_TRITON_ATTENTION"

    @staticmethod
    def get_builder_cls() -> Type["NativeTritonMetadataBuilder"]:
        return NativeTritonMetadataBuilder

    @staticmethod
    def get_impl_cls() -> Type["NativeTritonAttentionImpl"]:
        return NativeTritonAttentionImpl


# ---------------------------------------------------------------------------
# Metadata builder
# ---------------------------------------------------------------------------


class NativeTritonMetadataBuilder(CommonAttentionBuilder):
    """Inherits prepare_prefill from CommonAttentionBuilder; provides decode
    metadata + KV cache allocation in aiter's [blocks, heads, block_size, d]
    layout."""

    def __init__(
        self,
        kv_cache_spec=None,
        layer_names=None,
        config=None,
        device=None,
        model_runner=None,
    ):
        self.block_size = 16 if model_runner.block_size != 1024 else 1024
        CommonAttentionBuilder.__init__(self, model_runner)
        # ModelRunner.capture_cudagraph() unconditionally calls
        # forward_vars["kv_indptr"].gpu.zero_() — that buffer is allocated by
        # AiterAttentionMetadataBuilder. Add a tiny stub here so cudagraph
        # capture does not KeyError on our backend (we don't actually use it
        # because pa_decode is paged-block-table-based).
        from atom.utils import CpuGpuBuffer
        if "kv_indptr" not in self.model_runner.forward_vars:
            self.model_runner.forward_vars["kv_indptr"] = CpuGpuBuffer(
                self.max_bs + 1, dtype=torch.int32, device=self.device
            )
        logger.info(
            "NativeTritonMetadataBuilder: initialized (no aiter HIP allocations)"
        )

    # ------------------------------------------------------------------ #
    # KV pool sizing                                                     #
    # ------------------------------------------------------------------ #

    def _kv_layout_dims(self):
        runner = self.model_runner
        hf = runner.config.hf_config
        head_dim = getattr(hf, "head_dim", None) or (
            hf.hidden_size // hf.num_attention_heads
        )
        num_kv_heads = max(1, runner._get_num_kv_heads())
        n_layers = runner._get_total_num_layers()
        return n_layers, num_kv_heads, head_dim

    def _kv_dtype(self):
        return torch.bfloat16

    def compute_block_bytes(self) -> int:
        n_layers, num_kv_heads, head_dim = self._kv_layout_dims()
        elem = self._kv_dtype().itemsize
        return 2 * n_layers * self.block_size * num_kv_heads * head_dim * elem

    def allocate_kv_cache_tensors(
        self, num_kv_heads: int, num_draft_layers: int
    ) -> dict:
        runner = self.model_runner
        n_layers, _, head_dim = self._kv_layout_dims()
        # aiter pa_decode expects [num_blocks, num_kv_heads, block_size, head_dim].
        return {
            "kv_cache": torch.zeros(
                2,
                n_layers,
                runner.num_physical_kvcache_blocks,
                num_kv_heads,
                runner.physical_block_size,
                head_dim,
                dtype=self._kv_dtype(),
                device="cuda",
            ),
        }

    def build_kv_cache_tensor(self, layer_id: int, module):
        if not (
            hasattr(module, "base_attention")
            and hasattr(module, "use_mla")
            and not module.use_mla
        ):
            return None

        runner = self.model_runner
        # [num_blocks, num_kv_heads, block_size, head_dim]
        k_cache = runner.kv_cache[0, layer_id]
        v_cache = runner.kv_cache[1, layer_id]

        module.max_model_len = runner.config.max_model_len
        module.k_cache = k_cache
        module.v_cache = v_cache
        if not hasattr(module, "k_scale"):
            module.k_scale = None
            module.v_scale = None

        if hasattr(module, "impl") and module.impl is not None:
            module.impl.k_cache = k_cache
            module.impl.v_cache = v_cache

        return KVCacheTensor(
            layer_num=layer_id,
            k_cache=k_cache,
            v_cache=v_cache,
            k_scale=module.k_scale,
            v_scale=module.v_scale,
        )

    # ------------------------------------------------------------------ #
    # Decode metadata                                                    #
    # ------------------------------------------------------------------ #

    def prepare_decode(self, batch: ScheduledBatch, bs: int):
        scheduled_bs = batch.total_seqs_num_decode
        max_seqlen_q = 1
        block_size = self.model_runner.block_size

        context_lens = np.asarray(batch.context_lens, dtype=np.int32)
        block_tables = batch.block_tables

        slot_mapping = [
            block_table[-1] * block_size + last_block_num - 1
            for block_table, last_block_num in zip(
                block_tables, batch.last_block_num_tokens
            )
        ]
        positions = np.array(
            [cl - 1 for cl in context_lens[:scheduled_bs]], dtype=np.int32
        )
        max_seqlen_k = int(context_lens[:scheduled_bs].max()) if scheduled_bs > 0 else 0

        self.prepare_block_tables(batch)

        var = self.model_runner.forward_vars
        sum_scheduled_tokens = batch.total_tokens_num_decode
        # CUDAGRAPH PADDING (scheduled_bs < bs, e.g. when the engine pads
        # a 3-seq batch up to a captured bs=4 graph): the padded slots
        # must not trigger NaN-producing paths in pa_decode.
        #
        # With context_lens=0, aiter's pa_decode_v1/v2 kernels run zero
        # loop iterations and end with `acc /= exp_sum` where exp_sum
        # stayed 0 -> 0/0 = NaN. That NaN at slot[i>=scheduled_bs]
        # propagates through the per-tensor FP8 quant of attn_out
        # (`amax(... NaN ...) = NaN` -> the entire batch's x_scale
        # becomes NaN -> every downstream gemm_a8w8 output is NaN),
        # corrupting ALL real slots. Symptom: wrong logits at the first
        # decode step, model emits a stop token, request finishes after
        # one token. Reproduces in lm_eval (variable scheduled_bs) but
        # NOT in `concurrent==captured_bs` curl tests where padding
        # never kicks in.
        #
        # Fix: pad context_lens to 1 (a single garbage KV read,
        # producing a FINITE attn_out for the padded row) and leave
        # block_tables[padded_slot, 0] = 0 (the prepare_block_tables
        # default points at block 0, which holds real but unrelated KV
        # — fine for this purpose, the row's output is discarded
        # downstream by the engine which only reads outputs[:scheduled_bs]).
        # Keep slot_mapping = -1 for padded slots so our kv-write kernel's
        # `if slot < 0: return` sentinel skips the write — otherwise we'd
        # overwrite slot 0's real KV data.
        var["slot_mapping"].np[: bs * max_seqlen_q] = -1
        if not batch.is_dummy_run:
            var["slot_mapping"].np[:sum_scheduled_tokens] = slot_mapping[
                :sum_scheduled_tokens
            ]
        var["positions"].np[:sum_scheduled_tokens] = positions[:sum_scheduled_tokens]
        var["context_lens"].np[:scheduled_bs] = context_lens[:scheduled_bs]
        var["context_lens"].np[scheduled_bs:bs] = 1  # was 0 -> 0/0 NaN in pa_decode

        vars_used = [
            ("slot_mapping", bs * max_seqlen_q),
            ("context_lens", bs),
            ("cu_seqlens_q", bs + 1),
            ("block_tables", bs),
        ]
        ctx = {el: var[el].copy_to_gpu(num) for el, num in vars_used}

        attn_metadata = AttentionMetaData(
            max_seqlen_q=max_seqlen_q,
            min_seqlen_q=0,
            max_seqlen_k=max_seqlen_k,
            dropout_p=0.0,
            **ctx,
        )
        positions_gpu = var["positions"].copy_to_gpu(sum_scheduled_tokens)
        return attn_metadata, positions_gpu

    def build_for_cudagraph_capture(self, bs: int):
        """Return a (AttentionMetaData, Context) for cudagraph capture at a
        fixed decode batch size `bs`. Slices the pre-allocated forward_vars
        buffers so the captured graph re-uses the same GPU memory across
        replays. is_prefill=False -> graphs only the decode path.
        """
        var = self.model_runner.forward_vars
        attn_metadata = AttentionMetaData(
            slot_mapping=var["slot_mapping"].gpu[:bs],
            context_lens=var["context_lens"].gpu[:bs],
            block_tables=var["block_tables"].gpu[:bs],
            cu_seqlens_q=var["cu_seqlens_q"].gpu[: bs + 1],
            max_seqlen_q=1,
            min_seqlen_q=0,
            max_seqlen_k=self.model_runner.config.max_model_len,
            dropout_p=0.0,
        )
        positions = var["positions"].gpu[:bs]
        context = Context(
            positions=positions, is_prefill=False, batch_size=bs, graph_bs=bs
        )

        # Comprehensive pre-warm: triggers JIT compile of every triton kernel
        # in the decode forward path at this bs, on a fresh non-capturing
        # stream. Belt-and-suspenders against hipModuleLoad-during-capture
        # failures even though the engine's profile_run usually JITs first.
        self._prewarm_full_decode_for_bs(bs, attn_metadata, context)

        return attn_metadata, context

    # ------------------------------------------------------------------ #
    # Pre-warm helpers                                                   #
    # ------------------------------------------------------------------ #
    _prewarm_done_bs: set = None

    def _prewarm_full_decode_for_bs(
        self, bs: int, attn_metadata: AttentionMetaData, context: Context
    ) -> None:
        """JIT-compile every triton kernel used in the decode forward at this
        bs by running a full model.forward call on a non-capturing stream.

        Why: ATOM's capture_cudagraph runs its per-bs warmup inside
        `with graph_capture()`, which puts the stream in HIP capture mode
        (via ca_comm.capture()). Triton kernels first-call JIT via
        hipModuleLoad — not allowed in capture mode. A full forward on a
        FRESH stream pre-compiles every kernel (FP8 GEMM, kv-write,
        RMSNorm, SiLU+Mul, paged_attention_decode, and lm_head GEMM)
        at the exact (shape, dtype, stride) combo the engine will use,
        so the engine's subsequent warmup just replays cached kernels.
        """
        if NativeTritonMetadataBuilder._prewarm_done_bs is None:
            NativeTritonMetadataBuilder._prewarm_done_bs = set()
        if bs in NativeTritonMetadataBuilder._prewarm_done_bs:
            return

        runner = self.model_runner

        # Bind a safe decode metadata: 1-token context per request, all reading
        # block 0. Garbage data is fine — we only care about kernel compilation.
        var = runner.forward_vars
        var["context_lens"].np[:bs] = 1
        var["context_lens"].copy_to_gpu(bs)
        var["slot_mapping"].np[:bs] = np.arange(bs, dtype=np.int32)
        var["slot_mapping"].copy_to_gpu(bs)
        var["block_tables"].np[:bs] = 0
        var["block_tables"].copy_to_gpu(bs)
        var["positions"].np[:bs] = 0
        var["positions"].copy_to_gpu(bs)

        # Set forward context so the model knows we're in decode mode.
        set_forward_context(
            attn_metadata=attn_metadata,
            atom_config=runner.config,
            context=context,
            num_tokens=bs,
            num_tokens_across_dp=None,
            ubatch_slices=None,
        )

        input_ids = var["input_ids"].gpu[:bs]
        positions = var["positions"].gpu[:bs]
        # Zero input_ids (token 0) for stable warmup.
        input_ids.zero_()

        # PER SGLANG / PYTORCH PATTERN:
        # The warmup must run on the SAME stream that capture will use, NOT
        # a freshly-allocated side stream. `with graph_capture()` (entered
        # by ModelRunner.capture_cudagraph before calling us) has already
        # `torch.cuda.stream(gc.stream)`-d into gc.stream — so the current
        # stream IS gc.stream, and is NOT yet in capture mode (capture is
        # entered later by `torch.cuda.graph(stream=gc.stream)`).
        #
        # Run the warmup forward TWICE on the current stream:
        #   1st pass: triggers all triton JIT (hipModuleLoad) and any first-
        #             time autotune sync. Does this BEFORE capture begins.
        #   2nd pass: stabilizes allocator state in the graph mempool — by
        #             the second call, every torch.empty/torch.empty_like
        #             address is reused from the same pool slot the captured
        #             graph will then reuse at replay.
        # Skipping the second pass is the documented pitfall on AMD: HIP
        # capture errors are silent; the captured graph appears to capture
        # cleanly but reads/writes mismatched addresses at replay.
        for _ in range(2):
            try:
                outputs = runner.model(input_ids, positions)
                if hasattr(runner.model, "compute_logits"):
                    runner.model.compute_logits(outputs)
            except Exception as e:
                logger.warning(
                    "Full decode pre-warm bs=%d raised %s; cudagraph may still fail.",
                    bs, e,
                )
                break
        torch.cuda.current_stream().synchronize()
        NativeTritonMetadataBuilder._prewarm_done_bs.add(bs)
        logger.info("Full decode pre-warm complete for cudagraph bs=%d", bs)


# ---------------------------------------------------------------------------
# Attention impl
# ---------------------------------------------------------------------------


class NativeTritonAttentionImpl(AttentionImpl):
    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        scale: float,
        num_kv_heads: Optional[int] = None,
        alibi_slopes=None,
        sliding_window: Optional[int] = None,
        kv_cache_dtype: str = "bf16",
        logits_soft_cap=None,
        attn_type=None,
        kv_sharing_target_layer_name=None,
        layer_num: int = 0,
        mla_modules=None,
        sinks=None,
        rotary_emb=None,
        q_norm=None,
        k_norm=None,
        **kwargs,
    ):
        nn.Module.__init__(self)
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.head_size = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        self.sliding_window = sliding_window if sliding_window is not None else -1
        self.kv_cache_dtype = kv_cache_dtype
        self.layer_num = layer_num
        self.rotary_emb = rotary_emb
        self.q_norm = q_norm
        self.k_norm = k_norm
        self.q_size = num_heads * head_dim
        self.kv_size = self.num_kv_heads * head_dim
        # Set by build_kv_cache_tensor after engine_core.allocate_kv_cache.
        self.k_cache = torch.tensor([])
        self.v_cache = torch.tensor([])
        # Reusable scale tensors for the triton paged-attention kernel
        # (BF16 KV path -> identity scales). Pre-created here so that
        # CUDAGraph capture does not see a torch.tensor() allocation on the
        # first decode call.
        self._pa_k_scale = torch.tensor(1.0, dtype=torch.float32, device="cuda")
        self._pa_v_scale = torch.tensor(1.0, dtype=torch.float32, device="cuda")
        if kv_cache_dtype != "bf16":
            logger.warning(
                f"NativeTritonAttentionImpl: kv_cache_dtype={kv_cache_dtype} "
                "is a TODO; force --kv_cache_dtype bf16."
            )

    # ------------------------------------------------------------------ #
    # KV cache helpers                                                   #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _write_kv_cache(
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
        k_new: torch.Tensor,
        v_new: torch.Tensor,
    ) -> None:
        """Triton-launched scatter into the paged KV pool. Slot == -1 entries
        are skipped inside the kernel, so this path has no Python-side
        conditional and is CUDAGraph-capturable."""
        if slot_mapping.numel() == 0:
            return
        # Cast K/V to cache dtype if needed (cheap pointwise; otherwise no-op).
        if k_new.dtype != k_cache.dtype:
            k_new = k_new.to(k_cache.dtype)
        if v_new.dtype != v_cache.dtype:
            v_new = v_new.to(v_cache.dtype)
        _kv_cache_write_triton(k_cache, v_cache, slot_mapping, k_new, v_new)

    # ------------------------------------------------------------------ #
    # Forward                                                            #
    # ------------------------------------------------------------------ #

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        positions: Optional[torch.Tensor] = None,
        kv_cache: Optional[torch.Tensor] = None,
        layer_name: Optional[str] = None,
        use_mla: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        if use_mla:
            raise NotImplementedError(
                "NativeTritonAttentionImpl: MLA path is not implemented."
            )

        ctx = get_forward_context()
        attn_md: Optional[AttentionMetaData] = ctx.attn_metadata
        fc = ctx.context
        is_prefill = bool(getattr(fc, "is_prefill", True)) if fc is not None else True
        if attn_md is None:
            raise RuntimeError(
                "NativeTritonAttentionImpl: forward called without AttentionMetaData."
            )

        total_tokens = query.shape[0]
        q = query.view(total_tokens, self.num_heads, self.head_dim)
        k = key.view(total_tokens, self.num_kv_heads, self.head_dim)
        v = value.view(total_tokens, self.num_kv_heads, self.head_dim)

        if self.rotary_emb is not None and positions is not None:
            q_flat = q.reshape(total_tokens, self.num_heads * self.head_dim)
            k_flat = k.reshape(total_tokens, self.num_kv_heads * self.head_dim)
            q_flat, k_flat = self.rotary_emb(positions, q_flat, k_flat)
            q = q_flat.view(total_tokens, self.num_heads, self.head_dim)
            k = k_flat.view(total_tokens, self.num_kv_heads, self.head_dim)

        slot_mapping = attn_md.slot_mapping
        if (
            slot_mapping is not None
            and getattr(self, "k_cache", torch.empty(0)).numel() > 0
            and getattr(self, "v_cache", torch.empty(0)).numel() > 0
        ):
            self._write_kv_cache(
                self.k_cache, self.v_cache, slot_mapping[:total_tokens], k, v
            )

        if is_prefill:
            return self._forward_prefill(q, k, v, attn_md, total_tokens)
        return self._forward_decode(q, attn_md)

    # ---------------- prefill ---------------- #

    def _forward_prefill(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_md: AttentionMetaData,
        total_tokens: int,
    ) -> torch.Tensor:
        # Prefer triton context_attention_fwd (handles GQA internally; ~2x
        # faster than the torch SDPA loop on gfx1201 at gsm8k context lengths).
        # Triton-only — no torch SDPA fallback.
        if self.sliding_window is not None and self.sliding_window > 0:
            raise RuntimeError(
                "NativeTritonAttentionImpl: sliding_window prefill is not "
                "supported (triton context_attention_fwd has no sliding window)."
            )
        prefill = _get_triton_prefill()
        if prefill is None:
            raise RuntimeError(
                "aiter triton context_attention_fwd unavailable — required "
                "for prefill on gfx1201 (no torch fallback in this build)."
            )
        out = torch.empty_like(q)
        cu_q_gpu = attn_md.cu_seqlens_q.to(torch.int32)
        b_start_loc = cu_q_gpu[:-1].contiguous()
        b_seq_len = (cu_q_gpu[1:] - cu_q_gpu[:-1]).contiguous()
        prefill(
            q.contiguous(),
            k.contiguous(),
            v.contiguous(),
            out,
            b_start_loc,
            b_seq_len,
            int(attn_md.max_seqlen_q),
            is_causal=True,
        )
        return out.reshape(total_tokens, self.num_heads * self.head_dim)

    # ---------------- decode ---------------- #

    def _forward_decode(
        self,
        q: torch.Tensor,  # [bs, num_q_heads, head_dim]
        attn_md: AttentionMetaData,
    ) -> torch.Tensor:
        bs = q.shape[0]
        # Triton-only — no torch decode fallback.
        if self.sliding_window is not None and self.sliding_window > 0:
            raise RuntimeError(
                "NativeTritonAttentionImpl: sliding_window decode is not "
                "supported (aiter pa_decode has no sliding window)."
            )
        pa_decode, tl_bf16 = _get_triton_pa_decode()
        if pa_decode is None:
            raise RuntimeError(
                "aiter triton paged_attn_decode unavailable — required for "
                "decode on gfx1201 (no torch fallback in this build)."
            )
        if self.k_cache.numel() == 0:
            raise RuntimeError(
                "NativeTritonAttentionImpl: KV cache is empty at decode time "
                "(build_kv_cache_tensor was not called?)."
            )
        out = torch.empty_like(q)
        block_tables = attn_md.block_tables[:bs]
        seq_lens = attn_md.context_lens[:bs]
        pa_decode(
            out, q,
            self.k_cache, self.v_cache,
            block_tables, seq_lens,
            int(attn_md.max_seqlen_k),
            tl_bf16,
            self.num_kv_heads,
            float(self.scale),
        )
        return out.reshape(bs, self.num_heads * self.head_dim)
