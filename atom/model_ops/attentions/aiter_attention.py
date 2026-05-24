# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Aiter-triton attention backend for ATOM.

The whole backend is orchestration on top of upstream aiter triton kernels
(aiter.ops.triton.kv_cache.reshape_and_cache, .attention.prefill_attention,
.attention.pa_decode.paged_attention_decode, .rope.rope.rope_cached_thd_positions_2c_fwd).
These kernels are JIT-compiled by triton for the target arch, so the path
is correct on gfx94x/95x as well as gfx1201 (RDNA4); there is no arch
detection in this file.

There is NO torch fallback in this build: the path raises a clear
RuntimeError if any required triton kernel is unavailable, instead of
silently falling back to a slow path that would also reintroduce
GPU->CPU syncs that break CUDAGraph capture.

The previous HIP-fast-path implementation lives at aiter_attention_hip.py
for a future capability-based opt-in (HIP first, this triton path as the
fallback). For now selector.py routes here unconditionally.

KV cache layout (matches aiter's pa_decode triton kernel expectations)
----------------------------------------------------------------------
    runner.kv_cache : [2, num_layers, num_blocks, num_kv_heads, block_size, head_dim]
                     |--K-and-V--||--per-layer--||---paged storage in aiter format---|

Forward
-------
* Prefill: aiter reshape_and_cache (KV write), then aiter triton
  context_attention_fwd (handles GQA internally).
* Decode: aiter reshape_and_cache, then aiter paged_attention_decode
  (called with Python-float scales so the wrapper skips .item() and
  the path is safe inside a CUDAGraph capture).

The NativeTriton* class names are kept inside this module for now;
the public Backend class is named AiterBackend to match the selector
contract.
"""

from __future__ import annotations

import logging
from typing import Optional, Type

import numpy as np
import torch
import triton.language as tl
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


# ---------------------------------------------------------------------------
# Cached lazy-imports of aiter triton kernels
# ---------------------------------------------------------------------------
_TRITON_PA_DECODE = None
_TRITON_TL_BF16 = None
_TRITON_PREFILL = None
_TRITON_RESHAPE_AND_CACHE = None
_TRITON_ROPE = None
_TRITON_ROPE_NEOX_STYLE = None


def _get_triton_prefill():
    global _TRITON_PREFILL
    if _TRITON_PREFILL is None:
        try:
            from aiter.ops.triton.attention.prefill_attention import (
                context_attention_fwd,
            )

            _TRITON_PREFILL = context_attention_fwd
        except Exception as e:
            logger.warning("triton context_attention_fwd unavailable: %s", e)
            _TRITON_PREFILL = False
    return _TRITON_PREFILL if _TRITON_PREFILL is not False else None


def _get_triton_pa_decode():
    """Return (paged_attention_decode, tl.bfloat16) or (None, None).

    Uses aiter's public wrapper directly; pass Python-float scales so the
    wrapper's Tensor->.item() branch is skipped (CUDAGraph-safe path that
    landed in aiter via the Union[float, Tensor] signature relaxation).
    """
    global _TRITON_PA_DECODE, _TRITON_TL_BF16
    if _TRITON_PA_DECODE is None:
        try:
            from aiter.ops.triton.attention.pa_decode import paged_attention_decode

            _TRITON_PA_DECODE = paged_attention_decode
            _TRITON_TL_BF16 = tl.bfloat16
        except Exception as e:
            logger.warning("triton paged_attention_decode unavailable: %s", e)
            _TRITON_PA_DECODE = False
    return (
        (_TRITON_PA_DECODE, _TRITON_TL_BF16)
        if _TRITON_PA_DECODE is not False
        else (None, None)
    )


def _get_triton_reshape_and_cache():
    global _TRITON_RESHAPE_AND_CACHE
    if _TRITON_RESHAPE_AND_CACHE is None:
        try:
            from aiter.ops.triton.kv_cache import reshape_and_cache

            _TRITON_RESHAPE_AND_CACHE = reshape_and_cache
        except Exception as e:
            logger.warning("triton reshape_and_cache unavailable: %s", e)
            _TRITON_RESHAPE_AND_CACHE = False
    return _TRITON_RESHAPE_AND_CACHE if _TRITON_RESHAPE_AND_CACHE is not False else None


def _get_triton_rope():
    """Return (rope_cached_thd_positions_2c_fwd, RotateStyle.NEOX) or (None, None)."""
    global _TRITON_ROPE, _TRITON_ROPE_NEOX_STYLE
    if _TRITON_ROPE is None:
        try:
            from aiter.ops.triton.rope.rope import (
                RotateStyle,
                rope_cached_thd_positions_2c_fwd,
            )

            _TRITON_ROPE = rope_cached_thd_positions_2c_fwd
            _TRITON_ROPE_NEOX_STYLE = int(RotateStyle.NEOX)
        except Exception as e:
            logger.warning("triton rope_cached_thd_positions_2c_fwd unavailable: %s", e)
            _TRITON_ROPE = False
    return (
        (_TRITON_ROPE, _TRITON_ROPE_NEOX_STYLE)
        if _TRITON_ROPE is not False
        else (None, None)
    )


class AiterBackend(AttentionBackend):
    """Aiter-triton paged attention backend (arch-portable; default path)."""

    @staticmethod
    def get_name() -> str:
        # Match the historical name so plugin-mode routing and any
        # downstream string checks behave the same as the prior
        # implementation (see aiter_attention_hip.py for the previous
        # HIP-backed version).
        from atom.utils import is_plugin_mode

        return "ROCM_AITER_ATTENTION" if not is_plugin_mode() else "CUSTOM"

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
                    bs,
                    e,
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
        """Triton-launched scatter into the paged KV pool via aiter's
        reshape_and_cache. Slot < 0 entries are skipped inside the kernel,
        so this path has no Python-side conditional and is CUDAGraph-safe."""
        if slot_mapping.numel() == 0:
            return
        if k_new.dtype != k_cache.dtype:
            k_new = k_new.to(k_cache.dtype)
        if v_new.dtype != v_cache.dtype:
            v_new = v_new.to(v_cache.dtype)
        reshape_and_cache = _get_triton_reshape_and_cache()
        if reshape_and_cache is None:
            raise RuntimeError(
                "aiter triton reshape_and_cache unavailable -- required for "
                "KV-cache write on gfx1201 (no torch fallback in this build)."
            )
        reshape_and_cache(k_new, v_new, k_cache, v_cache, slot_mapping)

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
            rope, neox_style = _get_triton_rope()
            if rope is None:
                raise RuntimeError(
                    "aiter triton rope_cached_thd_positions_2c_fwd unavailable "
                    "-- required for RoPE on gfx1201 (no torch fallback in this build)."
                )
            if not getattr(self.rotary_emb, "is_neox_style", True):
                raise RuntimeError(
                    "NativeTritonAttentionImpl: aiter rope path currently used "
                    "with Neox style only."
                )
            # cos_cache/sin_cache hold only the front half of the rotary dim
            # (reuse_freqs_front_part=True) -- the same shape the in-tree
            # kernel previously inferred via shape[-1]*2 == rotary_dim.
            # ATOM rotary caches are [max_pos, 1, 1, rotary_dim/2] (4D)
            # because rotary_embedding.forward unsqueezes for broadcast over
            # head dims; aiter rope_cached_thd_positions_2c_fwd expects
            # 2D cos/sin tables [max_pos, rotary_dim/2 or rotary_dim].
            cos2d = self.rotary_emb.cos_cache.squeeze(-2).squeeze(-2)
            sin2d = self.rotary_emb.sin_cache.squeeze(-2).squeeze(-2)
            q, k = rope(
                q.contiguous(),
                k.contiguous(),
                cos2d,
                sin2d,
                positions,
                rotate_style=neox_style,
                reuse_freqs_front_part=True,
                nope_first=False,
            )

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
        # aiter paged_attention_decode (positional + kw mix to match wrapper).
        # k_scale=v_scale=1.0 (Python float) -> wrapper skips .item() and the
        # call is CUDAGraph-capture-safe (no GPU->CPU sync at the boundary).
        pa_decode(
            out,
            q,
            self.k_cache,
            self.v_cache,
            seq_lens,
            block_tables,
            float(self.scale),
            int(attn_md.max_seqlen_k),
            tl_bf16,
            k_scale=1.0,
            v_scale=1.0,
        )
        return out.reshape(bs, self.num_heads * self.head_dim)
