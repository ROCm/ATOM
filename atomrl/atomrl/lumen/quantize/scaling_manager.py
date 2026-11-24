# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

import logging
import os
from collections import defaultdict, deque

import torch
from torch import nn

from atomrl.lumen.quantize.config import (
    MXFP4_BLOCK_SIZE,
    AmaxAlgo,
    QuantConfig,
    QuantFormat,
    ScalingType,
    _get_float8_e4m3,
)
from atomrl.lumen.quantize.descriptor import FP8Descriptor

logger = logging.getLogger(__name__)

_fused_compute_scale = None
_fused_amax_abs = None


def _get_fused_compute_scale():
    global _fused_compute_scale
    if _fused_compute_scale is None:
        from atomrl.lumen.kernels.compute_scale import compute_scale

        _fused_compute_scale = compute_scale
    return _fused_compute_scale


def _get_fused_amax_abs():
    global _fused_amax_abs
    if _fused_amax_abs is None:
        from atomrl.lumen.ops.quantize.quant_amax_fused import fused_amax_abs

        _fused_amax_abs = fused_amax_abs
    return _fused_amax_abs


_TRACE_AMAX_TID = os.environ.get("LUMEN_TRACE_AMAX", "0") == "1"
_trace_amax_tids: dict = {}


def dump_amax_tids(label: str = "") -> None:
    """Print tensor_ids hitting len(history)==0. Call from training loop."""
    if not _TRACE_AMAX_TID or not _trace_amax_tids:
        return
    total = sum(_trace_amax_tids.values())
    unique = len(_trace_amax_tids)
    print(
        f"\n[AMAX TID] {label} — {total} total, {unique} unique tensor_ids hitting len(history)==0",
        flush=True,
    )
    # Show first 10 unique ids
    for i, (tid, cnt) in enumerate(
        sorted(_trace_amax_tids.items(), key=lambda x: -x[1])
    ):
        if i >= 10:
            print(f"  ... and {unique - 10} more", flush=True)
            break
        print(f"  {cnt:>4}x  {tid}", flush=True)
    _trace_amax_tids.clear()


_FUSED_QUANT_SCALE = os.environ.get("LUMEN_FUSED_QUANT_SCALE", "0") == "1"
_FUSED_CAST_TRANSPOSE = os.environ.get("LUMEN_FUSED_CAST_TRANSPOSE", "0") == "1"
_FUSED_QUANT_AMAX = os.environ.get("LUMEN_FUSED_QUANT_AMAX", "0") == "1"
_FUSED_QUANT_TRANSPOSE_CPP = (
    os.environ.get("LUMEN_FUSED_QUANT_TRANSPOSE_CPP", "0") == "1"
)
# fused_quant_transpose.cu only implements the gfx942 fnuz formats.
_HIP_FNUZ_DTYPES = frozenset(
    dt
    for name in ("float8_e4m3fnuz", "float8_e5m2fnuz")
    if (dt := getattr(torch, name, None)) is not None
)
_PREFER_HIPBLASLT = os.environ.get("LUMEN_PREFER_HIPBLASLT", "0") == "1"
_REUSE_QUANT_BUFFER = os.environ.get("LUMEN_REUSE_QUANT_BUFFER", "0") == "1"
_CAST_AMAX_AVAILABLE: bool | None = None
_CAST_TRANSPOSE_AVAILABLE: bool | None = None
_FUSED_QUANT_AMAX_AVAILABLE: bool | None = None
_HIP_CAST_TRANSPOSE_AVAILABLE: bool | None = None


def _probe_cast_transpose() -> bool:
    """Return True if Triton fused cast+transpose can run on this machine (cached)."""
    global _CAST_TRANSPOSE_AVAILABLE
    if _CAST_TRANSPOSE_AVAILABLE is not None:
        return _CAST_TRANSPOSE_AVAILABLE
    try:
        import triton  # noqa: F401

        if not torch.cuda.is_available():
            _CAST_TRANSPOSE_AVAILABLE = False
            return False
        from atomrl.lumen.ops.quantize.cast_transpose import _TORCH_TO_TL_FP8

        _CAST_TRANSPOSE_AVAILABLE = bool(_TORCH_TO_TL_FP8)
    except (ImportError, OSError, ValueError):
        _CAST_TRANSPOSE_AVAILABLE = False
    return _CAST_TRANSPOSE_AVAILABLE


def _probe_fused_quant_amax() -> bool:
    """Return True if the fused quant+amax Triton kernel is functional (cached)."""
    global _FUSED_QUANT_AMAX_AVAILABLE
    if _FUSED_QUANT_AMAX_AVAILABLE is not None:
        return _FUSED_QUANT_AMAX_AVAILABLE
    try:
        from atomrl.lumen.ops.quantize.quant_amax_fused import (
            _probe_fused_quant_amax as _probe,
        )

        _FUSED_QUANT_AMAX_AVAILABLE = _probe()
    except (ImportError, OSError):
        _FUSED_QUANT_AMAX_AVAILABLE = False
    return _FUSED_QUANT_AMAX_AVAILABLE


def _probe_hip_cast_transpose() -> bool:
    """Return True if the HIP C++ fused quant+transpose kernel is available (cached)."""
    global _HIP_CAST_TRANSPOSE_AVAILABLE
    if _HIP_CAST_TRANSPOSE_AVAILABLE is not None:
        return _HIP_CAST_TRANSPOSE_AVAILABLE
    try:
        from atomrl.lumen.ops.quantize.cast_transpose_hip import (
            _probe_hip_cast_transpose as _probe,
        )

        _HIP_CAST_TRANSPOSE_AVAILABLE = _probe()
    except (ImportError, OSError):
        _HIP_CAST_TRANSPOSE_AVAILABLE = False
    return _HIP_CAST_TRANSPOSE_AVAILABLE


def _get_quant_ops():
    """Lazy import to avoid circular dependency with atomrl.lumen.ops."""
    from atomrl.lumen.ops.quantize import (
        convert_from_mxfp8,
        convert_to_mxfp8,
        quant_fp8_blockwise_impl,
    )

    return convert_to_mxfp8, convert_from_mxfp8, quant_fp8_blockwise_impl


# ---------------------------------------------------------------------------
# Gradient quantization helpers
# ---------------------------------------------------------------------------

GRAD_QUANT_TYPES = (None, "fp8", "mxfp8", "mxfp4", "fp4")


def _round_to_fp8(tensor: torch.Tensor, fp8_dtype: torch.dtype) -> torch.Tensor:
    """Per-tensor FP8 quant-dequant round-trip."""
    orig_dtype = tensor.dtype
    amax = tensor.abs().amax().clamp(min=1e-12)
    fp8_max = torch.finfo(fp8_dtype).max
    scale = fp8_max / amax
    tensor_fp8 = (tensor.float() * scale).clamp(-fp8_max, fp8_max).to(fp8_dtype)
    return tensor_fp8.to(orig_dtype) / scale


def _round_to_mxfp8(tensor: torch.Tensor, block_size: int = 32) -> torch.Tensor:
    """Microscaling FP8 quant-dequant round-trip."""
    from atomrl.lumen.ops.quantize.padding import pad_to_block

    orig_dtype = tensor.dtype
    orig_shape = tensor.shape

    flat = tensor.reshape(-1, orig_shape[-1]).contiguous()
    flat, orig_m = pad_to_block(flat, block_size, dim=0)
    flat, orig_n = pad_to_block(flat, block_size, dim=-1)

    data_bf16 = flat.to(torch.bfloat16)
    convert_to_mxfp8, convert_from_mxfp8, _ = _get_quant_ops()
    data_lp, scales = convert_to_mxfp8(data_bf16, block_size=block_size, axis=-1)
    data_hp = convert_from_mxfp8(
        data_lp,
        scales,
        output_dtype=torch.bfloat16,
        block_size=block_size,
        axis=-1,
    )

    data_hp = data_hp[:orig_m, :orig_n]

    return data_hp.reshape(orig_shape).to(orig_dtype)


def _round_to_mxfp4(tensor: torch.Tensor, block_size: int = 32) -> torch.Tensor:
    """Microscaling FP4 quant-dequant round-trip."""
    from atomrl.lumen.ops.quantize.ops import convert_from_mxfp4, convert_to_mxfp4
    from atomrl.lumen.ops.quantize.padding import pad_to_block

    orig_dtype = tensor.dtype
    orig_shape = tensor.shape

    flat = tensor.reshape(-1, orig_shape[-1]).contiguous()
    flat, orig_m = pad_to_block(flat, block_size, dim=0)
    flat, orig_n = pad_to_block(flat, block_size, dim=-1)

    data_bf16 = flat.to(torch.bfloat16)
    data_lp, scales = convert_to_mxfp4(
        data_bf16, block_size=block_size, axis=-1, use_sr=True
    )
    data_hp = convert_from_mxfp4(
        data_lp, scales, output_dtype=torch.bfloat16, block_size=block_size, axis=-1
    )

    data_hp = data_hp[:orig_m, :orig_n]

    return data_hp.reshape(orig_shape).to(orig_dtype)


class ScalingManager:
    """Tracks per-tensor scaling factors across training iterations.

    Accepts either a :class:`QuantConfig` or legacy keyword arguments.

    Examples::

        # With QuantConfig
        mgr = ScalingManager(QuantConfig(format=QuantFormat.MXFP8,
                                         scaling=ScalingType.BLOCKWISE))

        # MLPerf-style: most_recent amax, history=4
        mgr = ScalingManager(QuantConfig(amax_algo=AmaxAlgo.MOST_RECENT,
                                         history_len=4))

        # Legacy style (still supported)
        mgr = ScalingManager(recipe="delayed")
    """

    def __init__(
        self,
        config: QuantConfig | None = None,
        *,
        recipe: str | None = None,
        history_len: int = 16,
        block_size: int = 32,
        fp8_dtype: torch.dtype | None = None,
    ):
        if config is not None:
            self.config = config
        elif recipe is not None:
            self.config = QuantConfig.from_str(
                format="mxfp8" if recipe == "mxfp8" else "fp8_e4m3",
                scaling=recipe if recipe not in ("mxfp8", "block") else "blockwise",
                block_size=block_size,
                history_len=history_len,
            )
        else:
            self.config = QuantConfig(block_size=block_size, history_len=history_len)

        if fp8_dtype is None:
            from atomrl.lumen.quantize.config import _get_float8_e4m3

            fp8_dtype = _get_float8_e4m3()

        self.fp8_dtype = self.config.torch_dtype or fp8_dtype
        self.fp8_dtype_bwd = self.config.torch_dtype_bwd or self.fp8_dtype
        self._fp8_max = self.config.fp8_max
        self._fp8_max_bwd = self.config.fp8_max_bwd
        self._margin = self.config.margin
        self.amax_history = defaultdict(lambda: deque(maxlen=self.config.history_len))
        self.scale_cache = {}
        self._dp_group = None
        self._use_sdma = config.use_sdma if config else False

        # FP8 param lifecycle state
        self._fp8_param_ids: set[str] = set()
        self._fp8_params: dict[str, nn.Parameter] = {}
        self._fp8_param_cache: dict[str, FP8Descriptor | None] = {}
        self._fp8_param_stale: set[str] = set()
        self._fp8_step_counter: int = -1
        self._sdma_allgather = None

        # Batch scale precomputation (delayed + MOST_RECENT only).
        # Replaces ~1,089 single-element Triton kernel launches per step with
        # 2 vectorized PyTorch ops.  Zero per-update overhead — all work is
        # deferred to a single lazy gather in _precompute_batch_scales().
        self._batch_enabled = bool(
            int(os.environ.get("LUMEN_BATCH_PRECOMPUTE_SCALES", "0"))
        )
        # Step-frozen batch scales (delayed + MAX-over-history). Computes ALL
        # per-tensor scales ONCE per optimizer step from a frozen history
        # snapshot (TE's amax_and_scale_update model), replacing ~N per-tensor
        # ``stack().amax().compute_scale()`` launches with a single flat
        # ``stack`` + ``amax(dim=1)``.  Invalidated only at the step boundary
        # (optimizer post-step hook), NOT on every update_amax, so a step's
        # reads all reuse the same frozen scales.
        self._step_batch_enabled = bool(
            int(os.environ.get("LUMEN_STEP_BATCH_SCALES", "0"))
        )
        # Contiguous per-tensor amax ring buffers for the step-frozen path.
        # Each tid owns a preallocated ``[history_len]`` row written in-place on
        # ``update_amax`` (O(1) scalar copy), so the per-step MAX reduction is a
        # single ``stack(rows).amax(dim=1)`` instead of ``N`` scattered
        # ``stack(list(deque)).amax()`` launches (critical at history_len=1024).
        self._ring_rows: dict[str, torch.Tensor] = {}
        self._ring_cursor: dict[str, int] = {}
        self._batch_scale_fwd: torch.Tensor | None = None
        self._batch_scale_bwd: torch.Tensor | None = None
        self._batch_amax_buf: torch.Tensor | None = None
        self._batch_tid_to_idx: dict[str, int] = {}
        self._batch_ordered_deques: list = []
        self._batch_scales_valid = False

        # Per-shape FP8 scratch buffers for backward-only quantizations.
        # Eliminates ~1,089 torch.empty_like allocations per step when enabled.
        self._fp8_scratch: dict[tuple, torch.Tensor] = {}

        # C++ FP8 quant dispatch (replaces Python get_scale + quant + update_amax)
        self._cpp_dispatcher = None
        if (
            _PREFER_HIPBLASLT
            and _FUSED_QUANT_AMAX
            and self.recipe == "delayed"
            and self.config.amax_algo == AmaxAlgo.MOST_RECENT
            and os.environ.get("LUMEN_CPP_QUANT_DISPATCH", "0") == "1"
        ):
            try:
                from atomrl.lumen.ops.quantize.quant_dispatch_cpp import (
                    get_cpp_dispatcher,
                )

                self._cpp_dispatcher = get_cpp_dispatcher(
                    self._fp8_max,
                    self._fp8_max_bwd,
                    self._margin,
                    self.config.history_len,
                )
                if self._cpp_dispatcher is not None:
                    logger.info("C++ FP8 quant dispatch enabled")
            except RuntimeError as e:
                logger.warning("C++ FP8 quant dispatch init failed: %s", e)

    @property
    def recipe(self) -> str:
        return self.config.recipe

    def set_dp_group(self, group) -> None:
        """Set the data-parallel process group for ``reduce_amax``."""
        self._dp_group = group

    def _compute_scale(self, amax: torch.Tensor, fp8_max: float) -> torch.Tensor:
        """Compute the quantization scale from *amax*, accounting for margin.

        Formula: ``sf = (fp8_max / amax) / (2 ** margin)``,
        returned as ``amax / (fp8_max / (2 ** margin))`` so that the caller
        can divide by the scale to quantize.
        Always returns fp32 to satisfy hipb_mm's scale dtype requirement.

        Uses a single Triton kernel launch on CUDA instead of 4 separate ops
        (div + gt + ones_like + where).
        """
        if amax.is_cuda:
            return _get_fused_compute_scale()(amax, fp8_max, margin=self._margin)
        amax = amax.float().reshape(1)
        effective_max = fp8_max / (2**self._margin)
        scale = amax / effective_max
        scale = torch.where(scale > 1.0e-30, scale, torch.ones_like(scale))
        return scale

    def get_bwd_scale_for_epilogue(self, tensor_id: str):
        """Return the delayed-scaling bwd scale for GEMM FP8 epilogue, or None.

        Returns a pre-computed per-tensor scale when delayed scaling has history
        for *tensor_id*.  Returns ``None`` when history is empty (first step) or
        the recipe is not delayed — callers should fall back to the standard
        BF16 GEMM + quantize path.
        """
        if self.recipe != "delayed":
            return None
        history = self.amax_history.get(tensor_id)
        if history is None or len(history) == 0:
            return None
        if self.config.amax_algo == AmaxAlgo.MOST_RECENT:
            amax = torch.tensor(history[-1], dtype=torch.float32, device="cuda")
        else:
            if self._step_batch_enabled and not (
                self.config.reduce_amax and self._dp_group is not None
            ):
                self._precompute_batch_scales()
                if self._batch_scales_valid and tensor_id in self._batch_tid_to_idx:
                    idx = self._batch_tid_to_idx[tensor_id]
                    return self._batch_scale_bwd[idx : idx + 1]
            amax = torch.tensor(max(history), dtype=torch.float32, device="cuda")
        return self._compute_scale(amax, self._fp8_max_bwd)

    def _precompute_batch_scales(self):
        """Batch-compute all delayed+MOST_RECENT scales via lazy gather.

        Gathers latest amaxes from ``amax_history`` into a pre-allocated
        contiguous buffer, then computes fwd/bwd scales in 2 vectorized ops.
        Called once per step on the first ``get_scale()`` after invalidation.
        """
        if self._batch_scales_valid:
            return
        # Rebuild mapping + ordered deque list when tensor_ids change
        if len(self._batch_tid_to_idx) != len(self.amax_history):
            self._batch_tid_to_idx = {tid: i for i, tid in enumerate(self.amax_history)}
            self._batch_ordered_deques = [
                self.amax_history[tid] for tid in self._batch_tid_to_idx
            ]
            self._batch_amax_buf = None  # force re-alloc on new size
        N = len(self._batch_tid_to_idx)
        if N == 0:
            return
        deques = self._batch_ordered_deques
        if self.config.amax_algo == AmaxAlgo.MOST_RECENT:
            # Pre-allocate buffer once, reuse across steps
            if self._batch_amax_buf is None:
                device = deques[0][-1].device
                self._batch_amax_buf = torch.empty(
                    N,
                    dtype=torch.float32,
                    device=device,
                )
            # Scatter-copy latest amaxes into contiguous buffer (tight loop,
            # no per-element dict lookup, no .float()/.reshape() when not needed)
            buf = self._batch_amax_buf
            for i, dq in enumerate(deques):
                buf[i] = dq[-1]
        else:
            # MAX-over-history: reduce every tensor's contiguous ring-buffer row
            # in a single vectorized op.  Replaces ~N per-tensor
            # ``stack(list(deque)).amax()`` launches (one Cat over history_len
            # scalars + one reduce each) with a single ``stack`` of N rows +
            # ``amax(dim=1)``.  Unfilled ring slots are 0, which is the identity
            # for MAX over non-negative amaxes, so no masking is needed.
            rows = self._ring_rows
            if any(tid not in rows for tid in self._batch_tid_to_idx):
                # A tensor has not been written to its ring yet (warmup): leave
                # scales invalid so callers use the per-tensor slow path.
                return
            ordered = [rows[tid] for tid in self._batch_tid_to_idx]
            buf = torch.stack(ordered).amax(dim=1)
        fwd_recip = 1.0 / (self._fp8_max / (2**self._margin))
        bwd_recip = 1.0 / (self._fp8_max_bwd / (2**self._margin))
        self._batch_scale_fwd = torch.where(
            buf > 0,
            buf * fwd_recip,
            torch.ones_like(buf),
        )
        self._batch_scale_bwd = torch.where(
            buf > 0,
            buf * bwd_recip,
            torch.ones_like(buf),
        )
        self._batch_scales_valid = True

    def get_scale(
        self,
        tensor_id: str,
        tensor: torch.Tensor,
        *,
        backward: bool = False,
        return_amax: bool = False,
        skip_amax_precompute: bool = False,
    ):
        """Return the scale factor for this tensor (None for block/mxfp8/per_token/none).

        When *return_amax* is True, returns ``(scale, current_amax)`` so the
        caller can pass the amax to :meth:`update_amax_value` instead of
        recomputing ``tensor.abs().amax()``.  The returned ``current_amax``
        is the amax of **the current tensor** (for history update), which is
        only computed for the delayed-first-call and dynamic paths.  For
        delayed with existing history, ``current_amax`` is ``None`` (the
        tensor was not scanned).

        When *skip_amax_precompute* is True AND history is empty, uses
        ``fp8_max`` as a conservative first-call scale (scale=1.0) instead
        of scanning the tensor.  The caller is expected to use a fused
        kernel that computes its own amax and appends it to history.
        This avoids a redundant ``_amax_abs_kernel`` launch when activation
        checkpointing causes many first-call hits per step.
        """
        recipe = self.recipe
        fp8_max = self._fp8_max_bwd if backward else self._fp8_max

        if recipe == "none":
            return (None, None) if return_amax else None

        if recipe == "delayed":
            history = self.amax_history[tensor_id]
            current_amax = None
            if len(history) == 0:
                if skip_amax_precompute:
                    # Caller will use a fused kernel (e.g. static_quant_with_amax)
                    # that computes its own amax. Use fp8_max as placeholder →
                    # scale = fp8_max / fp8_max = 1.0.
                    amax = torch.tensor(
                        float(fp8_max), device=tensor.device, dtype=torch.float32
                    )
                    current_amax = None
                else:
                    if _TRACE_AMAX_TID:
                        _trace_amax_tids[tensor_id] = (
                            _trace_amax_tids.get(tensor_id, 0) + 1
                        )
                    current_amax = (
                        _get_fused_amax_abs()(tensor)
                        if tensor.is_cuda and tensor.dim() >= 2
                        else tensor.abs().amax()
                    )
                    amax = current_amax
            elif self.config.amax_algo == AmaxAlgo.MOST_RECENT:
                # Batch fast path: return precomputed scale from contiguous buffer
                if (
                    self._batch_enabled
                    and tensor_id in self._batch_tid_to_idx
                    and not (self.config.reduce_amax and self._dp_group is not None)
                ):
                    self._precompute_batch_scales()
                    if self._batch_scales_valid:
                        idx = self._batch_tid_to_idx[tensor_id]
                        buf = (
                            self._batch_scale_bwd if backward else self._batch_scale_fwd
                        )
                        scale = buf[idx : idx + 1]
                        return (scale, None) if return_amax else scale
                amax = torch.tensor(
                    history[-1], dtype=torch.float32, device=tensor.device
                )
            else:
                # Step-frozen batch fast path: one precompute per step covers
                # all tensors' MAX-over-history scales; subsequent reads index
                # the frozen buffer.  ``_precompute_batch_scales`` is called
                # unconditionally (not guarded by map membership) so the
                # tid→idx map bootstraps on the first read of the step.
                if self._step_batch_enabled and not (
                    self.config.reduce_amax and self._dp_group is not None
                ):
                    self._precompute_batch_scales()
                    if self._batch_scales_valid and tensor_id in self._batch_tid_to_idx:
                        idx = self._batch_tid_to_idx[tensor_id]
                        buf = (
                            self._batch_scale_bwd if backward else self._batch_scale_fwd
                        )
                        scale = buf[idx : idx + 1]
                        return (scale, None) if return_amax else scale
                # history now contains Python floats — max() is pure Python, zero GPU ops.
                amax = torch.tensor(
                    max(history), dtype=torch.float32, device=tensor.device
                )

            if self.config.reduce_amax and self._dp_group is not None:
                if self._use_sdma:
                    amax = self._reduce_single_amax_sdma(amax)
                else:
                    torch.distributed.all_reduce(
                        amax,
                        op=torch.distributed.ReduceOp.MAX,
                        group=self._dp_group,
                    )

            scale = self._compute_scale(amax, fp8_max)
            return (scale, current_amax) if return_amax else scale
        elif recipe == "dynamic":
            current_amax = (
                _get_fused_amax_abs()(tensor)
                if tensor.is_cuda and tensor.dim() >= 2
                else tensor.abs().amax()
            )
            if self.config.reduce_amax and self._dp_group is not None:
                if self._use_sdma:
                    current_amax = self._reduce_single_amax_sdma(current_amax)
                else:
                    torch.distributed.all_reduce(
                        current_amax,
                        op=torch.distributed.ReduceOp.MAX,
                        group=self._dp_group,
                    )
            scale = self._compute_scale(current_amax, fp8_max)
            return (scale, current_amax) if return_amax else scale
        elif recipe in ("blockwise", "blockwise2d", "mxfp8", "per_token"):
            return (None, None) if return_amax else None

    def update_amax(self, tensor_id: str, tensor: torch.Tensor):
        """Record amax for delayed scaling (tensor-based).

        Uses a fused Triton kernel on CUDA to replace abs()+amax() (2 launches)
        with a single launch.  The resulting amax scalar is stored as a Python
        float in the deque so that get_scale's MAX-over-history path can use
        ``max(history)`` (pure Python, zero GPU kernels) instead of
        ``torch.stack(list(history)).amax()`` (CatArrayBatchedCopy kernel).
        """
        t = tensor.detach()
        if t.is_cuda and t.dim() >= 2:
            amax_t = _get_fused_amax_abs()(t)
        else:
            amax_t = t.abs().amax()
        self.amax_history[tensor_id].append(amax_t.item())
        if self._step_batch_enabled:
            self._ring_append(tensor_id, amax_t)
        else:
            self._batch_scales_valid = False

    def update_amax_value(self, tensor_id: str, amax: torch.Tensor):
        """Record a pre-computed amax value, skipping the abs().amax() pass."""
        amax_val = float(amax.detach().squeeze())
        self.amax_history[tensor_id].append(amax_val)
        if self._step_batch_enabled:
            self._ring_append(tensor_id, amax.detach().squeeze())
        else:
            self._batch_scales_valid = False

    def _ring_append(self, tensor_id: str, amax: torch.Tensor) -> None:
        """Write *amax* into this tensor's contiguous ring buffer (in-place)."""
        row = self._ring_rows.get(tensor_id)
        if row is None:
            row = torch.zeros(
                self.config.history_len, dtype=torch.float32, device=amax.device
            )
            self._ring_rows[tensor_id] = row
            self._ring_cursor[tensor_id] = 0
        c = self._ring_cursor[tensor_id]
        row[c] = amax
        self._ring_cursor[tensor_id] = (c + 1) % self.config.history_len

    def quantize(self, tensor_id: str, tensor: torch.Tensor, *, backward: bool = False):
        """Quantize tensor. Returns :class:`~atomrl.lumen.quantize.descriptor.FP8Descriptor` or ``None``.

        When FP8 param mode is active for *tensor_id*, this method
        returns the cached (and possibly lazily re-quantized) FP8 weight
        instead of quantizing on-the-fly.

        When *backward* is True and the format is HYBRID, E5M2 dtype and its
        corresponding FP8_MAX are used instead of the forward-pass values.
        """
        if not backward:
            cached = self.get_fp8_param_cached(tensor_id, tensor)
            if cached is not None:
                return cached

        return self._quantize_core(tensor_id, tensor, backward=backward)

    # ------------------------------------------------------------------
    # FP8 parameter lifecycle (replaces standalone FP8ParamManager)
    # ------------------------------------------------------------------

    @property
    def num_fp8_params(self) -> int:
        return len(self._fp8_params)

    def register_fp8_param(self, tensor_id: str, param: nn.Parameter) -> None:
        """Register a parameter for FP8 lifecycle management."""
        self._fp8_param_ids.add(tensor_id)
        self._fp8_params[tensor_id] = param

    def mark_fp8_params_stale(self) -> None:
        """Mark all cached FP8 params as stale.

        Call after ``optimizer.step()`` so that the next forward pass
        re-quantizes from the updated master weights.
        """
        self._fp8_param_stale.update(self._fp8_param_ids)

    def check_and_mark_fp8_stale(self, current_step: int) -> None:
        """Mark stale only when the training step counter advances.

        Safe to call on every micro-batch; actual staleness is flagged
        only once per optimizer step.
        """
        if current_step > self._fp8_step_counter:
            self._fp8_step_counter = current_step
            self.mark_fp8_params_stale()

    def quantize_fp8_params(self) -> None:
        """Eagerly re-quantize all registered master params to FP8.

        Populates the internal FP8 param cache.  Equivalent to
        Megatron's ``cast_master_weights_to_fp8``.

        For FSDP workloads prefer :meth:`mark_fp8_params_stale` (lazy)
        because the full unsharded param is only available during
        FSDP's forward pass.
        """
        if not self._fp8_params:
            return

        with torch.no_grad():
            for tid, param in self._fp8_params.items():
                self._fp8_param_stale.discard(tid)
                desc = self._quantize_core(tid, param.data)
                if desc is not None:
                    desc.data = desc.data.detach()
                    desc.scale = desc.scale.detach()
                self._fp8_param_cache[tid] = desc

        if self._dp_group is not None:
            if self._use_sdma:
                self._reduce_fp8_amax_sdma()
            else:
                self._reduce_fp8_amax_dist()

    def get_fp8_param_cached(self, tensor_id: str, tensor: torch.Tensor):
        """Return cached FP8 weight, re-quantizing lazily if stale.

        Called from the forward path instead of :meth:`quantize` when
        FP8 param mode is active for *tensor_id*.
        """
        if tensor_id not in self._fp8_param_ids:
            return None

        cached = self._fp8_param_cache.get(tensor_id)
        if cached is not None and tensor_id not in self._fp8_param_stale:
            return cached

        self._fp8_param_stale.discard(tensor_id)
        desc = self._quantize_core(tensor_id, tensor)
        if desc is not None:
            desc.data = desc.data.detach()
            desc.scale = desc.scale.detach()
        self._fp8_param_cache[tensor_id] = desc
        return desc

    def _collect_amaxes(self):
        """Gather the latest amax values from all registered FP8 params."""
        amaxes = []
        tensor_ids = []
        for tid in self._fp8_params:
            history = self.amax_history.get(tid)
            if history and len(history) > 0:
                # history stores Python floats; wrap as CUDA tensor for distributed reduce.
                val = history[-1]
                t = (
                    val
                    if isinstance(val, torch.Tensor)
                    else torch.tensor(val, dtype=torch.float32, device="cuda")
                )
                amaxes.append(t)
                tensor_ids.append(tid)
        return amaxes, tensor_ids

    def _scatter_amaxes(self, max_amaxes, tensor_ids) -> None:
        """Write reduced amax values back into the history deques."""
        for i, tid in enumerate(tensor_ids):
            history = self.amax_history[tid]
            if len(history) > 0:
                # Store as Python float regardless of whether max_amaxes[i] is a tensor.
                val = max_amaxes[i]
                history[-1] = float(val) if isinstance(val, torch.Tensor) else val

    def _reduce_fp8_amax_sdma(self) -> None:
        """All-reduce (MAX) per-param amax across DP ranks via mori SDMA."""
        from atomrl.lumen.ops.sdma import SdmaAllgather, sdma_allgather_max

        amaxes, tensor_ids = self._collect_amaxes()
        if not amaxes:
            return

        packed = torch.stack(amaxes)

        if self._sdma_allgather is None:
            self._sdma_allgather = SdmaAllgather()
            logger.info(
                "ScalingManager: created SdmaAllgather handle (%d amax elements)",
                packed.numel(),
            )

        max_amaxes = sdma_allgather_max(packed, self._sdma_allgather)
        self._scatter_amaxes(max_amaxes, tensor_ids)

    def _reduce_fp8_amax_dist(self) -> None:
        """All-reduce (MAX) per-param amax across DP ranks via torch.distributed."""
        amaxes, tensor_ids = self._collect_amaxes()
        if not amaxes:
            return

        packed = torch.stack(amaxes)
        torch.distributed.all_reduce(
            packed, op=torch.distributed.ReduceOp.MAX, group=self._dp_group
        )
        self._scatter_amaxes(packed, tensor_ids)

    def _reduce_single_amax_sdma(self, amax: torch.Tensor) -> torch.Tensor:
        """Reduce a single amax scalar via SDMA (allgather + max)."""
        from atomrl.lumen.ops.sdma import sdma_allgather_max

        packed = amax.float().unsqueeze(0)
        if self._sdma_allgather is None:
            from atomrl.lumen.ops.sdma import SdmaAllgather

            self._sdma_allgather = SdmaAllgather()
        result = sdma_allgather_max(packed, self._sdma_allgather)
        return result[0]

    def register_fp8_optimizer_hook(
        self,
        optimizer: torch.optim.Optimizer,
    ) -> "ScalingManager":
        """Register a post-step hook that marks FP8 params stale.

        The next forward pass will lazily re-quantize from the updated
        master weights — no explicit call needed in the training loop.
        """

        def _on_post_step():
            self.mark_fp8_params_stale()
            # Step boundary: drop frozen batch scales so the next step's first
            # get_scale recomputes them from the updated history snapshot.
            if self._step_batch_enabled:
                self._batch_scales_valid = False

        # torch.optim.Optimizer exposes register_step_post_hook, but Megatron's
        # ChainedOptimizer / DistributedOptimizer do not — wrap ``step()`` in
        # that case so the hook still fires exactly once per optimizer step.
        if hasattr(optimizer, "register_step_post_hook"):
            optimizer.register_step_post_hook(lambda _opt, _a, _k: _on_post_step())
            logger.info("ScalingManager: registered FP8 optimizer post-step hook")
        else:
            _orig_step = optimizer.step

            def _wrapped_step(*args, **kwargs):
                result = _orig_step(*args, **kwargs)
                _on_post_step()
                return result

            optimizer.step = _wrapped_step
            logger.info(
                "ScalingManager: wrapped optimizer.step() for FP8 staleness (no register_step_post_hook)"
            )
        return self

    def enable_fp8_params(self, model: nn.Module) -> "ScalingManager":
        """Scan *model* for quantized layers and register their weights.

        Looks for modules with ``_quant_tensor_id`` (set during
        ``quant.enable()`` patching) and registers their ``.weight``
        parameters for FP8 lifecycle management.
        """
        # This cache holds per-tensor / blockwise FP8 descriptors, and every
        # reader of it is on an FP8 path: MXFP4's forward quantizes the weight
        # in quantize_input and its backward works off the saved tensors, so
        # nothing here would ever be read back. Enabling it under MXFP4 was not
        # inert though -- it spent a full quantization pass per weight per
        # optimizer step, and stored the result with row-wise scales where an
        # MXFP4 consumer expects 2D tiles, so the first reader to appear would
        # have gotten a silently wrong layout rather than an error.
        if self.config.format == QuantFormat.MXFP4:
            raise ValueError(
                "The FP8 param cache (--lumen-fp8-param-gather, or "
                "LUMEN_WEIGHT_QUANT_ONCE=1) does not apply to MXFP4: its "
                "descriptors are never read back, and the layout would not match. "
                "Nothing is lost by turning it off: MXFP4 caches its own FP4 "
                "weight per optimizer step unconditionally "
                "(atomrl.lumen.quantize._mxfp4_cached_weight)."
            )
        count = 0
        for _name, module in model.named_modules():
            tensor_id = getattr(module, "_quant_tensor_id", None)
            if tensor_id is not None and hasattr(module, "weight"):
                self.register_fp8_param(tensor_id, module.weight)
                count += 1
        if count > 0:
            self.quantize_fp8_params()
            logger.info(
                "ScalingManager: FP8 param mode enabled (%d params)",
                count,
            )
        return self

    def fp8_allgather_weight(
        self,
        weight_shard: torch.Tensor,
        group=None,
        use_sdma: bool | None = None,
        num_chunks: int = 1,
    ) -> torch.Tensor:
        """All-gather a weight shard via FP8, dequantize per-shard.

        Delegates to :func:`atomrl.lumen.quantize.fp8_params.fp8_allgather_weight`
        using this manager's FP8 dtype and SDMA preference.  Per-rank
        scales are gathered automatically so each shard is dequantized
        with its originating rank's scale.

        Args:
            weight_shard: Local weight shard ``[rows_local, cols]``.
            group: Process group. Defaults to ``self._dp_group`` when
                set, otherwise WORLD (resolved inside
                ``fp8_allgather_weight``).
            use_sdma: Override SDMA preference. Defaults to
                ``self._use_sdma``.

        Returns:
            Full weight ``[rows_full, cols]`` in BF16.
        """
        from atomrl.lumen.quantize.fp8_params import fp8_allgather_weight

        if use_sdma is None:
            use_sdma = self._use_sdma
        if group is None:
            group = self._dp_group

        return fp8_allgather_weight(
            weight_shard,
            group=group,
            fp8_dtype=self.fp8_dtype,
            target_dtype=torch.bfloat16,
            use_sdma=use_sdma,
            num_chunks=num_chunks,
        )

    def fp8_allgather_weights_pipelined(
        self,
        weight_shards: list,
        group=None,
        use_sdma: bool | None = None,
        num_chunks: int = 1,
    ) -> list:
        """Pipelined FP8 all-gather for multiple weight shards.

        Overlaps allgather(i+1) with dequant(i) on separate streams.

        Args:
            weight_shards: List of local weight shards.
            group: Process group (defaults to ``self._dp_group`` or WORLD).
            use_sdma: Override SDMA preference.

        Returns:
            List of full (dequantized) weight tensors in BF16.
        """
        from atomrl.lumen.quantize.fp8_params import fp8_allgather_weight_pipelined

        if use_sdma is None:
            use_sdma = self._use_sdma
        if group is None:
            group = self._dp_group

        return fp8_allgather_weight_pipelined(
            weight_shards,
            group=group,
            fp8_dtype=self.fp8_dtype,
            target_dtype=torch.bfloat16,
            use_sdma=use_sdma,
            num_chunks=num_chunks,
        )

    # ------------------------------------------------------------------
    # Quantize core (shared by both on-the-fly and FP8 param paths)
    # ------------------------------------------------------------------

    def _get_fp8_scratch(self, device, shape, dtype):
        """Return a reusable FP8 output buffer for backward-only quantizations."""
        dev_idx = device.index if hasattr(device, "index") else device
        key = (dev_idx, shape, dtype)
        buf = self._fp8_scratch.get(key)
        if buf is None or buf.shape != shape or buf.dtype != dtype:
            buf = torch.empty(shape, dtype=dtype, device=device)
            self._fp8_scratch[key] = buf
        return buf

    def _aiter_static_quant(
        self,
        tensor: torch.Tensor,
        scale: torch.Tensor,
        fp8_dtype: torch.dtype,
        *,
        compute_amax: bool = False,
        out: torch.Tensor = None,
    ) -> "torch.Tensor | tuple[torch.Tensor, torch.Tensor]":
        """Fused per-tensor static quant via AITER Triton (x / scale -> fp8).

        When *compute_amax* is True, returns ``(fp8_tensor, amax)``. AITER owns
        quantization; PyTorch computes the accompanying statistic.
        """
        if tensor.dim() == 2:
            tensor_2d = tensor.contiguous()
        else:
            tensor_2d = tensor.reshape(-1, tensor.shape[-1]).contiguous()
        if out is not None and out.shape == tensor_2d.shape and out.dtype == fp8_dtype:
            qx = out
        else:
            qx = torch.empty_like(tensor_2d, dtype=fp8_dtype)
        if (
            scale.dtype == torch.float32
            and scale.ndim == 1
            and scale.numel() == 1
            and scale.is_contiguous()
        ):
            scale_in = scale
        else:
            scale_in = scale.float().reshape(1).contiguous()
        if scale_in.device != tensor.device:
            scale_in = scale_in.to(device=tensor.device)

        if compute_amax:
            from aiter.ops.triton.quant import static_per_tensor_quant_fp8_i8

            static_per_tensor_quant_fp8_i8(qx, tensor_2d, scale_in)
            return qx.view(tensor.shape), tensor_2d.abs().amax().float()
        else:
            from aiter.ops.triton.quant import static_per_tensor_quant_fp8_i8

            static_per_tensor_quant_fp8_i8(qx, tensor_2d, scale_in)
            return qx.view(tensor.shape)

    @staticmethod
    def _eager_transpose(desc: "FP8Descriptor") -> "FP8Descriptor":
        """Eagerly populate ``_transpose`` on a 2D FP8Descriptor if missing.

        Avoids a lazy recomputation on first ``transpose_cached`` access.
        Uses fast Triton transpose when available.
        """
        if desc._transpose is not None or desc.data.dim() != 2 or not desc.data.is_cuda:
            return desc
        try:
            from atomrl.lumen.ops.quantize.fast_transpose import fast_transpose_fp8

            desc._transpose = fast_transpose_fp8(desc.data)
        except (ImportError, OSError, RuntimeError):
            desc._transpose = desc.data.t().contiguous()
        return desc

    def _quantize_core(
        self, tensor_id: str, tensor: torch.Tensor, *, backward: bool = False
    ):
        """Core quantization logic (delegates to format-specific paths)."""
        if self.config.scaling == ScalingType.NONE:
            return None

        # C++ fast path: fused scale+quant+amax in a single C++ call
        if (
            self._cpp_dispatcher is not None
            and tensor.is_cuda
            and not (self.config.reduce_amax and self._dp_group is not None)
        ):
            fp8_data, scale = self._cpp_dispatcher.quantize(tensor_id, tensor, backward)
            dtype = self.fp8_dtype_bwd if backward else self.fp8_dtype
            return FP8Descriptor(data=fp8_data, scale=scale, fp8_dtype=dtype)

        # When Branch 2 (hipBLASLt fused quant+amax) will fire, skip the
        # standalone amax precompute in get_scale — the fused kernel computes
        # its own amax, so the precomputed one is wasted.  This eliminates
        # ~557 _amax_abs_kernel launches/step from activation checkpointing.
        _will_use_fused_quant_amax = (
            _PREFER_HIPBLASLT
            and _FUSED_QUANT_AMAX
            and _probe_fused_quant_amax()
            and tensor.is_cuda
        )
        scale, _precomputed_amax = self.get_scale(
            tensor_id,
            tensor,
            backward=backward,
            return_amax=True,
            skip_amax_precompute=_will_use_fused_quant_amax,
        )
        fp8_max = self._fp8_max_bwd if backward else self._fp8_max
        dtype = self.fp8_dtype_bwd if backward else self.fp8_dtype

        if scale is None and self.config.format == QuantFormat.MXFP4:
            from atomrl.lumen.ops.quantize.ops import convert_to_mxfp4

            # Row-wise scales along the reduction axis: the activation
            # convention, matching quantize_input(is_weight=False). Weights use
            # 2D tiles and are quantized by quantize_input, never here --
            # enable_fp8_params refuses MXFP4 so the weight paths cannot reach
            # this branch and pick up the wrong one of the two layouts.
            # RTN for weights (SR only for gradients per NVFP4 paper §4.4)
            fp4_tensor, mx_scale = convert_to_mxfp4(
                tensor,
                block_size=self.config.block_size,
                axis=-1,
                use_sr=False,
            )
            return FP8Descriptor(data=fp4_tensor, scale=mx_scale, fp8_dtype=None)

        if scale is None and self.config.format == QuantFormat.MXFP8:
            convert_to_mxfp8, _, _ = _get_quant_ops()
            fp8_tensor, mx_scale = convert_to_mxfp8(
                tensor,
                block_size=self.config.block_size,
                axis=-1,
                float8_dtype_pt=dtype,
            )
            return FP8Descriptor(data=fp8_tensor, scale=mx_scale, fp8_dtype=dtype)

        if scale is None and self.config.scaling in (
            ScalingType.BLOCKWISE,
            ScalingType.BLOCKWISE2D,
        ):
            _, _, quant_fp8_blockwise_impl = _get_quant_ops()
            orig_shape = tensor.shape
            flat = tensor.reshape(-1, orig_shape[-1])
            fp8_tensor, fp8_scales = quant_fp8_blockwise_impl(
                flat.contiguous(),
                dtype=dtype,
                axis=1,
                block_size=self.config.block_size,
            )
            return FP8Descriptor(
                data=fp8_tensor.view(orig_shape), scale=fp8_scales, fp8_dtype=dtype
            )

        if scale is None and self.config.scaling == ScalingType.PER_TOKEN:
            orig_shape = tensor.shape
            flat = tensor.reshape(-1, orig_shape[-1])
            row_max = flat.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12)
            row_scale = row_max / fp8_max
            fp8_tensor = (flat / row_scale).clamp(-fp8_max, fp8_max).to(dtype)
            return FP8Descriptor(
                data=fp8_tensor.view(orig_shape), scale=row_scale, fp8_dtype=dtype
            )

        if (
            _FUSED_QUANT_TRANSPOSE_CPP
            and backward
            and dtype in _HIP_FNUZ_DTYPES
            and _probe_hip_cast_transpose()
            and tensor.is_cuda
            and tensor.dim() == 2
        ):
            # Only fuse cast+transpose on the backward grad quantization: there
            # BOTH layouts are consumed (row-major for dgrad, transposed for
            # wgrad), so producing both in one pass is a net win. Forward
            # activation/weight quantization would waste the transpose (not used
            # that step), so it falls through to the single-output fast path.
            from atomrl.lumen.ops.quantize.cast_transpose_hip import (
                cast_transpose_amax_fp8_hip,
            )

            tensor_2d = tensor.reshape(-1, tensor.shape[-1]).contiguous()
            fp8_data, fp8_data_t, amax = cast_transpose_amax_fp8_hip(
                tensor_2d,
                scale,
                dtype,
            )
            self.amax_history[tensor_id].append(float(amax.squeeze()))
            return FP8Descriptor(
                data=fp8_data.view(tensor.shape),
                scale=scale,
                fp8_dtype=dtype,
                _transpose=fp8_data_t,
            )

        if (
            _PREFER_HIPBLASLT
            and _FUSED_QUANT_AMAX
            and _probe_fused_quant_amax()
            and tensor.is_cuda
        ):
            # Row-based kernel: fewer atomics and better coalescing than
            # the 2D-tiled cast_amax_fp8 (1 atomic/row vs 1 per 64x64 tile).
            from atomrl.lumen.ops.quantize.quant_amax_fused import (
                static_quant_with_amax,
            )

            tensor_2d = tensor.reshape(-1, tensor.shape[-1]).contiguous()
            _out_buf = None
            if backward and _REUSE_QUANT_BUFFER:
                _out_buf = self._get_fp8_scratch(tensor.device, tensor_2d.shape, dtype)
            fp8_data, amax = static_quant_with_amax(
                tensor_2d, scale, dtype, out=_out_buf
            )
            self.amax_history[tensor_id].append(float(amax.squeeze()))
            return FP8Descriptor(
                data=fp8_data.view(tensor.shape),
                scale=scale,
                fp8_dtype=dtype,
            )

        if (
            _FUSED_CAST_TRANSPOSE
            and _FUSED_QUANT_AMAX
            and _probe_cast_transpose()
            and _probe_fused_quant_amax()
            and tensor.is_cuda
            and tensor.dim() == 2
        ):
            from atomrl.lumen.ops.quantize.cast_transpose import (
                _TORCH_TO_TL_FP8,
                cast_transpose_amax_fp8,
            )

            if dtype in _TORCH_TO_TL_FP8:
                tensor_2d = tensor.reshape(-1, tensor.shape[-1]).contiguous()
                fp8_data, fp8_data_t, amax = cast_transpose_amax_fp8(
                    tensor_2d,
                    scale,
                    dtype,
                    clamp_max=float(fp8_max),
                )
                self.amax_history[tensor_id].append(float(amax.squeeze()))
                return FP8Descriptor(
                    data=fp8_data.view(tensor.shape),
                    scale=scale,
                    fp8_dtype=dtype,
                    _transpose=fp8_data_t,
                )

        if (
            _FUSED_CAST_TRANSPOSE
            and _probe_cast_transpose()
            and tensor.is_cuda
            and tensor.dim() == 2
        ):
            from atomrl.lumen.ops.quantize.cast_transpose import (
                _TORCH_TO_TL_FP8,
                cast_transpose_fp8,
            )

            if dtype in _TORCH_TO_TL_FP8:
                tensor_2d = tensor.reshape(-1, tensor.shape[-1]).contiguous()
                fp8_data, fp8_data_t = cast_transpose_fp8(
                    tensor_2d,
                    scale,
                    dtype,
                    clamp_max=float(fp8_max),
                )
                fp8_tensor = fp8_data.view(tensor.shape)
                if _precomputed_amax is not None:
                    self.update_amax_value(tensor_id, _precomputed_amax)
                else:
                    self.update_amax(tensor_id, tensor)
                return FP8Descriptor(
                    data=fp8_tensor,
                    scale=scale,
                    fp8_dtype=dtype,
                    _transpose=fp8_data_t,
                )

        if _FUSED_QUANT_AMAX and _probe_fused_quant_amax() and tensor.is_cuda:
            from atomrl.lumen.ops.quantize.quant_amax_fused import (
                static_quant_with_amax,
            )

            tensor_2d = tensor.reshape(-1, tensor.shape[-1]).contiguous()
            _out_buf = None
            if backward and _REUSE_QUANT_BUFFER:
                _out_buf = self._get_fp8_scratch(tensor.device, tensor_2d.shape, dtype)
            fp8_2d, amax = static_quant_with_amax(tensor_2d, scale, dtype, out=_out_buf)
            self.amax_history[tensor_id].append(float(amax.squeeze()))
            desc = FP8Descriptor(
                data=fp8_2d.view(tensor.shape), scale=scale, fp8_dtype=dtype
            )
            return self._eager_transpose(desc) if not backward else desc

        if tensor.is_cuda:
            needs_amax = _precomputed_amax is None
            _out_buf = None
            if backward and _REUSE_QUANT_BUFFER:
                _t2d = tensor.reshape(-1, tensor.shape[-1])
                _out_buf = self._get_fp8_scratch(tensor.device, _t2d.shape, dtype)
            result = self._aiter_static_quant(
                tensor, scale, dtype, compute_amax=needs_amax, out=_out_buf
            )
            if needs_amax:
                fp8_tensor, kernel_amax = result
                self.update_amax_value(tensor_id, kernel_amax)
            else:
                fp8_tensor = result
                self.update_amax_value(tensor_id, _precomputed_amax)
        else:
            fp8_tensor = (tensor * (1.0 / scale)).clamp(-fp8_max, fp8_max).to(dtype)
            if _precomputed_amax is not None:
                self.update_amax_value(tensor_id, _precomputed_amax)
            else:
                self.update_amax(tensor_id, tensor)
        desc = FP8Descriptor(data=fp8_tensor, scale=scale, fp8_dtype=dtype)
        return self._eager_transpose(desc) if not backward else desc

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self):
        """Clear all tracked state (e.g. after warmup)."""
        self.amax_history.clear()
        self.scale_cache.clear()
        self._fp8_param_cache.clear()
        self._fp8_param_stale.clear()
        self._fp8_step_counter = -1
        self._batch_tid_to_idx.clear()
        self._batch_ordered_deques.clear()
        self._batch_amax_buf = None
        self._batch_scale_fwd = None
        self._batch_scale_bwd = None
        self._batch_scales_valid = False
        if self._cpp_dispatcher is not None:
            self._cpp_dispatcher.reset()

    # ------------------------------------------------------------------
    # Backward delayed scaling (blockwise2d cross-iteration reuse)
    # ------------------------------------------------------------------

    def quantize_bwd_delayed(
        self,
        tensor_id: str,
        tensor: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Quantize a backward tensor using delayed per-tensor scaling.

        On the first call for *tensor_id* (no history), computes scale from
        the current tensor's amax (same as dynamic).  On subsequent calls,
        reuses the amax history to derive the scale without waiting for the
        current tensor's reduction — mirroring
        :meth:`Blockwise2DScaleManager.cache_do_scale` from the attention path.

        The current tensor's amax is always appended to history for the next
        iteration.

        Returns ``(fp8_tensor, per_tensor_scale)``.
        """
        # C++ fast path
        if self._cpp_dispatcher is not None and tensor.is_cuda:
            fp8_data, scale = self._cpp_dispatcher.quantize(tensor_id, tensor, True)
            return fp8_data, scale

        fp8_max = self._fp8_max_bwd
        dtype = self.fp8_dtype_bwd

        history = self.amax_history[tensor_id]
        precomputed_amax = None
        use_batch_scale = False
        _will_use_fused = (
            _FUSED_QUANT_AMAX and _probe_fused_quant_amax() and tensor.is_cuda
        )
        if len(history) == 0:
            if _will_use_fused:
                # Fused kernel (static_quant_with_amax) computes its own amax.
                # Use fp8_max → scale=1.0 as a safe first-call placeholder.
                amax = torch.tensor(
                    float(fp8_max), device=tensor.device, dtype=torch.float32
                )
            else:
                precomputed_amax = (
                    _get_fused_amax_abs()(tensor)
                    if tensor.is_cuda and tensor.dim() >= 2
                    else tensor.abs().amax()
                )
                amax = precomputed_amax
        elif self.config.amax_algo == AmaxAlgo.MOST_RECENT:
            # Batch fast path: use precomputed bwd scale from contiguous buffer
            if self._batch_enabled and tensor_id in self._batch_tid_to_idx:
                self._precompute_batch_scales()
                if self._batch_scales_valid:
                    use_batch_scale = True
            if not use_batch_scale:
                amax = torch.tensor(
                    history[-1], dtype=torch.float32, device=tensor.device
                )
        else:
            # Step-frozen batch fast path (MAX-over-history), bootstraps map.
            if self._step_batch_enabled and not (
                self.config.reduce_amax and self._dp_group is not None
            ):
                self._precompute_batch_scales()
                if self._batch_scales_valid and tensor_id in self._batch_tid_to_idx:
                    use_batch_scale = True
            if not use_batch_scale:
                amax = torch.tensor(
                    max(history), dtype=torch.float32, device=tensor.device
                )

        if use_batch_scale:
            idx = self._batch_tid_to_idx[tensor_id]
            scale = self._batch_scale_bwd[idx : idx + 1]
        else:
            scale = self._compute_scale(amax, fp8_max)

        if _FUSED_QUANT_AMAX and _probe_fused_quant_amax() and tensor.is_cuda:
            from atomrl.lumen.ops.quantize.quant_amax_fused import (
                static_quant_with_amax,
            )

            tensor_2d = tensor.reshape(-1, tensor.shape[-1]).contiguous()
            fp8_2d, new_amax = static_quant_with_amax(tensor_2d, scale, dtype)
            self.update_amax_value(tensor_id, new_amax)
            return fp8_2d.view(tensor.shape), scale

        if tensor.is_cuda:
            needs_amax = precomputed_amax is None
            result = self._aiter_static_quant(
                tensor, scale, dtype, compute_amax=needs_amax
            )
            if needs_amax:
                fp8_tensor, kernel_amax = result
                self.update_amax_value(tensor_id, kernel_amax)
            else:
                fp8_tensor = result
                self.update_amax_value(tensor_id, precomputed_amax)
        else:
            fp8_tensor = (tensor / scale).clamp(-fp8_max, fp8_max).to(dtype)
            if precomputed_amax is not None:
                self.update_amax_value(tensor_id, precomputed_amax)
            else:
                self.update_amax(tensor_id, tensor)
        return fp8_tensor, scale

    # ------------------------------------------------------------------
    # Gradient quantization
    # ------------------------------------------------------------------

    def quantize_grad(self, tensor: torch.Tensor) -> torch.Tensor:
        """Apply gradient quantization based on ``self.config.quantize_grad``.

        Performs a quant-dequant round-trip that reduces the gradient tensor
        to the representable precision of the configured low-precision format,
        then returns a tensor with the original shape and dtype.

        The format is read from ``self.config.quantize_grad`` (one of
        ``None``, ``"fp8"``, ``"mxfp8"``, ``"fp4"``).  When ``None``, the
        tensor is returned unchanged.
        """
        return self.quantize_grad_tensor(
            tensor,
            self.config.quantize_grad,
            fp8_dtype=self.fp8_dtype,
            block_size=self.config.block_size,
        )

    @staticmethod
    def quantize_grad_tensor(
        tensor: torch.Tensor,
        grad_quant_type: str | None,
        fp8_dtype: torch.dtype | None = None,
        block_size: int = 32,
    ) -> torch.Tensor:
        """Stateless gradient quant-dequant round-trip.

        Can be called without a :class:`ScalingManager` instance.

        Args:
            tensor: The gradient tensor.
            grad_quant_type: ``"fp8"``, ``"mxfp8"``, ``"mxfp4"``, ``"fp4"``, or ``None``.
            fp8_dtype: Explicit FP8 dtype for the ``"fp8"`` path.  Auto-detects
                when ``None``.
            block_size: Block size for ``"mxfp8"`` / ``"mxfp4"`` quantization.
        """
        if grad_quant_type is None:
            return tensor

        if grad_quant_type == "fp8":
            if fp8_dtype is None:
                fp8_dtype = _get_float8_e4m3()
            return _round_to_fp8(tensor, fp8_dtype)

        if grad_quant_type == "mxfp8":
            return _round_to_mxfp8(tensor, block_size=block_size)

        if grad_quant_type == "mxfp4":
            # MXFP4's E8M0 scale belongs to exactly 32 elements. ``block_size``
            # may come from an otherwise-FP8 linear recipe (normally 128), so
            # gradient format selection must not inherit that unrelated knob.
            return _round_to_mxfp4(tensor, block_size=MXFP4_BLOCK_SIZE)

        if grad_quant_type == "fp4":
            raise NotImplementedError(
                "Unscaled FP4 gradient quantization is not implemented. Use "
                "'mxfp4' for block-scaled FP4, or 'fp8'/'mxfp8'."
            )

        raise ValueError(
            f"Unknown grad_quant_type={grad_quant_type!r}. "
            f"Valid options: {GRAD_QUANT_TYPES}"
        )

    # ------------------------------------------------------------------
    # Attention quantization primitives
    # ------------------------------------------------------------------

    @staticmethod
    def quantize_per_tensor_fp8(
        tensor: torch.Tensor,
        float8_dtype: torch.dtype | None = None,
    ):
        """Per-tensor FP8 quantization.

        Returns ``(tensor_fp8, descale)`` where *descale* has shape ``(1,)``
        matching the aiter per-tensor convention.
        """
        if float8_dtype is None:
            float8_dtype = _get_float8_e4m3()
        dtype_max = torch.finfo(float8_dtype).max
        amax = tensor.abs().amax()
        amax = torch.where(
            amax == 0,
            torch.tensor(dtype_max, device=tensor.device, dtype=tensor.dtype),
            amax,
        )
        scale = dtype_max / amax
        tensor_fp8 = (tensor * scale).clamp(-dtype_max, dtype_max).to(float8_dtype)
        descale = (1.0 / scale).to(torch.float32).reshape(1)
        return tensor_fp8, descale

    @staticmethod
    def quantize_block_fp8(
        tensor: torch.Tensor,
        block_m: int,
        float8_dtype: torch.dtype | None = None,
    ):
        """Per-block FP8 quantization for attention Q/K tensors.

        Expects input in ``[B, S, H, D]`` (BSHD) layout.  Internally permutes
        to ``[B, H, S, D]``, reshapes into blocks of *block_m* rows, computes
        per-block max for scaling, quantizes, and permutes back to BSHD.

        Returns ``(tensor_fp8, scale_inv)`` where *scale_inv* is the inverse
        scale suitable for dequantization.
        """
        if float8_dtype is None:
            float8_dtype = _get_float8_e4m3()
        tensor = tensor.permute(0, 2, 1, 3)  # [B, H, S, D]
        B, H, L, D = tensor.shape
        MAX_FP8 = torch.finfo(float8_dtype).max
        tensor = tensor.reshape(B, H, L // block_m, block_m, D).reshape(
            B,
            H,
            L // block_m,
            block_m * D,
        )
        tensor_max = tensor.abs().max(dim=-1)[0]
        tensor_max = torch.where(tensor_max == 0, MAX_FP8, tensor_max)
        scale = MAX_FP8 / tensor_max
        tensor = tensor * scale.reshape(scale.shape + (1,))
        tensor = tensor.clamp(-MAX_FP8, MAX_FP8).to(float8_dtype)
        tensor = tensor.reshape(B, H, L, D).permute(0, 2, 1, 3).contiguous()
        return tensor, 1.0 / scale.to(torch.float32).contiguous()

    @staticmethod
    def quantize_v_fp8(
        v: torch.Tensor,
        float8_dtype: torch.dtype | None = None,
    ):
        """Per-tensor FP8 quantization for attention V tensor.

        Returns ``(v_fp8, v_scale, p_scale)`` where *p_scale* is the FP8 max
        value used for softmax scaling in the attention kernel.
        """
        if float8_dtype is None:
            float8_dtype = _get_float8_e4m3()
        range_v = torch.max(torch.abs(v))
        dtype_max = torch.finfo(float8_dtype).max
        v_scale = dtype_max / range_v
        p_scale = dtype_max
        finfo = torch.finfo(float8_dtype)
        v_fp8 = (v * v_scale).clamp(min=finfo.min, max=finfo.max).to(float8_dtype)
        return v_fp8, v_scale, p_scale

    @staticmethod
    def quantize_block_mxfp8(
        tensor: torch.Tensor,
        block_size: int,
        layout: str = "bshd",
        *,
        is_2d_block: bool = True,
        float8_dtype: torch.dtype | None = None,
        cu_seqlens=None,
        max_seqlens=None,
    ):
        """MXFP8 block quantization with attention layout support.

        Supports ``"bhsd"``, ``"bshd"``, and ``"thd"`` layouts.

        Returns ``(tensor_mxfp8, scale)``.
        """
        if float8_dtype is None:
            float8_dtype = _get_float8_e4m3()

        if layout == "bhsd":
            tensor_bhsd = tensor
            B, H, _S, D = tensor_bhsd.shape
        elif layout == "bshd":
            tensor_bhsd = tensor.permute(0, 2, 1, 3).contiguous()
            B, H, _S, D = tensor_bhsd.shape
        elif layout == "thd":
            assert cu_seqlens is not None, "thd layout requires cu_seqlens"
            assert (
                tensor.dim() == 3
            ), f"expected thd tensor shape [T,H,D], got {tensor.shape}"
            _T, H, D = tensor.shape
            B = int(cu_seqlens.numel() - 1)
            assert max_seqlens is not None, "thd layout requires max_seqlens"
            max_seqlen = (
                int(max_seqlens)
                if isinstance(max_seqlens, int)
                else int(max(max_seqlens))
            )
            if is_2d_block:
                padded = ((max_seqlen + block_size - 1) // block_size) * block_size
            else:
                padded = max_seqlen
            tensor_bhsd = torch.zeros(
                (B, H, padded, D),
                device=tensor.device,
                dtype=tensor.dtype,
            )
            for b in range(B):
                s = int(cu_seqlens[b].item())
                e = int(cu_seqlens[b + 1].item())
                tensor_bhsd[b, :, : e - s, :] = tensor[s:e].transpose(0, 1)
        else:
            raise ValueError(f"Unsupported layout: {layout}")

        convert_to_mxfp8, _, _ = _get_quant_ops()
        quanted_bhsd, scale_bhsd = convert_to_mxfp8(
            tensor_bhsd,
            block_size=block_size,
            axis=-1,
            is_2d_block=is_2d_block,
            float8_dtype_pt=float8_dtype,
        )

        if layout == "bshd":
            return (
                quanted_bhsd.permute(0, 2, 1, 3).contiguous(),
                scale_bhsd.permute(0, 2, 1, 3).contiguous(),
            )

        if layout == "thd":
            qs, ss = [], []
            for b in range(B):
                s0 = int(cu_seqlens[b].item())
                e0 = int(cu_seqlens[b + 1].item())
                L = e0 - s0
                qs.append(quanted_bhsd[b, :, :L, :].transpose(0, 1).contiguous())
                if is_2d_block:
                    m_blocks = (L + block_size - 1) // block_size
                    ss.append(
                        scale_bhsd[b, :, :m_blocks, :].transpose(0, 1).contiguous()
                    )
                else:
                    ss.append(scale_bhsd[b, :, :L, :].transpose(0, 1).contiguous())
            return torch.cat(qs, dim=0), torch.cat(ss, dim=0)

        return quanted_bhsd, scale_bhsd

    @staticmethod
    def compute_p_scale_mxfp8(
        float8_dtype: torch.dtype | None = None,
    ) -> int:
        """Compute softmax P scale constant for MXFP8 attention kernels.

        Returns an integer scale derived from the FP8 format's exponent
        encoding, used by Triton MXFP8 attention kernels for correct
        softmax rescaling.
        """
        if float8_dtype is None:
            float8_dtype = _get_float8_e4m3()
        p_scale_f = torch.finfo(float8_dtype).max
        if float8_dtype == torch.float8_e4m3fn:
            mask_s, mbits, s_bias = 0b1111, 3, 7
        else:
            mask_s, mbits, s_bias = 0b11111, 2, 15
        hp_ebias = 127
        raw = (
            torch.bitwise_right_shift(
                torch.tensor(p_scale_f).to(float8_dtype).view(torch.uint8),
                mbits,
            )
            & mask_s
        )
        return (raw - s_bias + hp_ebias).to(torch.uint32).item()

    @staticmethod
    def quantize_block_2d_fp8(
        tensor: torch.Tensor,
        block_m: int,
        block_n: int,
        float8_dtype: torch.dtype | None = None,
    ):
        """2D block FP8 quantization for attention Q/K/V.

        Expects input in ``[B, S, H, D]`` (BSHD) layout.  Internally permutes
        to ``[B, H, S, D]``, partitions into ``(block_m x block_n)`` 2D tiles,
        computes one scale per tile, quantizes, and permutes back to BSHD.

        Scale shape: ``[B, H, S//block_m, D//block_n]`` (float32).

        Returns ``(tensor_fp8, scale_inv)`` where *scale_inv* is the inverse
        scale suitable for dequantization.
        """
        if float8_dtype is None:
            float8_dtype = _get_float8_e4m3()
        tensor = tensor.permute(0, 2, 1, 3).float()  # [B, H, S, D]
        B, H, S, D = tensor.shape
        MAX_FP8 = torch.finfo(float8_dtype).max

        tensor = tensor.reshape(B, H, S // block_m, block_m, D // block_n, block_n)
        tensor_max = tensor.abs().amax(dim=(3, 5))  # [B, H, S//bm, D//bn]
        tensor_max = torch.where(tensor_max == 0, MAX_FP8, tensor_max)
        scale = MAX_FP8 / tensor_max  # [B, H, S//bm, D//bn]

        scale_expanded = scale[:, :, :, None, :, None].expand_as(tensor)
        tensor = (tensor * scale_expanded).clamp(-MAX_FP8, MAX_FP8).to(float8_dtype)
        tensor = tensor.reshape(B, H, S, D).permute(0, 2, 1, 3).contiguous()

        return tensor, (1.0 / scale).to(torch.float32).contiguous()


class Blockwise2DScaleManager:
    """Manages 2D block FP8 scales across forward/backward for a single attention layer.

    Lifecycle:
        1. Forward: call :meth:`quantize_and_cache` — quantizes Q/K/V with 2D
           blocks and caches the FP8 tensors + scales.
        2. Backward: call :meth:`get_cached` — returns the cached FP8 tensors
           and scales so that backward can skip re-quantization.
        3. Backward (dO): optionally call :meth:`get_do_scale` to reuse the dO
           scale from the previous iteration, and :meth:`cache_do_scale` to
           persist the current iteration's dO scale.
    """

    def __init__(self, block_m: int = 64, block_n: int = 64):
        self.block_m = block_m
        self.block_n = block_n
        self._q_fp8 = None
        self._k_fp8 = None
        self._v_fp8 = None
        self._q_scale = None
        self._k_scale = None
        self._v_scale = None
        self._do_scale = None

    def quantize_and_cache(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        float8_dtype: torch.dtype | None = None,
    ):
        """Forward: quantize Q/K/V with 2D blocks, cache results.

        Args:
            q, k, v: Tensors in ``[B, S, H, D]`` layout.
            float8_dtype: Target FP8 dtype (default: auto-detected E4M3).

        Returns:
            ``(q_fp8, k_fp8, v_fp8, q_scale, k_scale, v_scale)``
        """
        q_fp8, q_scale = ScalingManager.quantize_block_2d_fp8(
            q, self.block_m, self.block_n, float8_dtype
        )
        k_fp8, k_scale = ScalingManager.quantize_block_2d_fp8(
            k, self.block_m, self.block_n, float8_dtype
        )
        v_fp8, v_scale = ScalingManager.quantize_block_2d_fp8(
            v, self.block_m, self.block_n, float8_dtype
        )

        self._q_fp8 = q_fp8
        self._k_fp8 = k_fp8
        self._v_fp8 = v_fp8
        self._q_scale = q_scale
        self._k_scale = k_scale
        self._v_scale = v_scale

        return q_fp8, k_fp8, v_fp8, q_scale, k_scale, v_scale

    def get_cached(self):
        """Backward: return cached FP8 tensors and scales.

        Returns:
            ``(q_fp8, k_fp8, v_fp8, q_scale, k_scale, v_scale)``

        Raises:
            RuntimeError: If :meth:`quantize_and_cache` was not called first.
        """
        if self._q_fp8 is None:
            raise RuntimeError(
                "Blockwise2DScaleManager.get_cached() called before "
                "quantize_and_cache(). Ensure forward runs first."
            )
        return (
            self._q_fp8,
            self._k_fp8,
            self._v_fp8,
            self._q_scale,
            self._k_scale,
            self._v_scale,
        )

    def get_do_scale(self) -> torch.Tensor | None:
        """Return cached dO scale from previous backward, or None."""
        return self._do_scale

    def cache_do_scale(self, do_scale: torch.Tensor) -> None:
        """Backward: save dO scale for next iteration's backward."""
        self._do_scale = do_scale.detach()

    def clear(self) -> None:
        """Release all cached tensors (useful between training steps)."""
        self._q_fp8 = self._k_fp8 = self._v_fp8 = None
        self._q_scale = self._k_scale = self._v_scale = None
        self._do_scale = None
