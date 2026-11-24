# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""QKV MXFP8 GEMM + norm / RoPE / NVFP4 inserts, bitwise the unfused chain: the split-K GEMM
(``mxfp8_gemm_shuffled(defer_reduce=True)``), then ``splitk_reduce_norm_rope_nvfp4_insert``.

One launch at 1-64 and 129-192 rows; the common split-K GEMM plus this entry's reduce-insert at 65-128 and
193-288; the common bf16-out GEMM plus ``norm_rope_nvfp4_insert`` at 289-1536. Counter storage is separated by
stream and graph capture.
"""

import ctypes
import functools
import struct
import threading

import torch

from amd_kernels.attention.norm_rope_nvfp4_insert_d128_rot64_p128 import (
    norm_rope_nvfp4_insert,
)
from amd_kernels.common.tensor_shim import _run_compiled
from amd_kernels.gemm.mxfp8_common import gemm as _gemm
from amd_kernels.warm_up import entry

from . import kernel as _kernel
from . import kernel_cooperative as _cooperative
from . import kernel_k_waves as _k_waves
from . import reduce_insert as _reduce_insert

N, K = _kernel.N, _kernel.K
# One launch through 64 rows: the last arriver at 1-16, cooperative eight-row owners at 17-64.
FUSED_MAX_ROWS = 64
MAX_ROWS = 1536  # the shuffled-code producer's range (``prequant_code_tile_rows``)


@functools.cache
def _fused_launcher(pairs: int):
    return _kernel.compile_qkv_insert(pairs)


@functools.cache
def _cooperative_launcher(rows: int):
    return _cooperative.compile_qkv_cooperative(rows)


@functools.cache
def _k_wave_launcher(rows: int):
    return _k_waves.compile_qkv_k_waves(rows)


_COUNTER_SETS = 512  # eager streams and retained graph captures per device; each graph shares its set serially
_COUNTER_WORDS = 3 * _kernel.HEADS * _kernel.CNT_STRIDE  # up to three row blocks at 129-192
_COUNTER_LOCK = threading.Lock()  # pool initialization and slice reservation may release the GIL
_POOL: dict[torch.device, torch.Tensor] = {}
_COUNTERS: dict[tuple[torch.device, tuple], torch.Tensor] = {}


@functools.cache
def _capture_info_api():
    """The HIP capture ID distinguishes graphs built on the same stream (stdlib ctypes, no device launch)."""
    lib = ctypes.CDLL("libamdhip64.so")
    fn = lib.hipStreamGetCaptureInfo
    fn.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_ulonglong)]
    fn.restype = ctypes.c_int
    return lib, fn


def _counter_key(device):
    stream = torch.cuda.current_stream(device).cuda_stream
    if not torch.cuda.is_current_stream_capturing():
        return device, ("stream", stream)
    _, fn = _capture_info_api()
    status, capture_id = ctypes.c_int(), ctypes.c_ulonglong()
    rc = fn(ctypes.c_void_p(stream), ctypes.byref(status), ctypes.byref(capture_id))
    if rc or status.value != 1:
        raise RuntimeError(f"hipStreamGetCaptureInfo failed: rc={rc}, status={status.value}")
    return device, ("capture", capture_id.value, stream)


def _counters(device: torch.device) -> torch.Tensor:
    """One counter set per eager stream or (capture ID, stream); epoch counters (each call adds 16 to every counter
    it uses, so a set stays a multiple of 16 between calls and needs no re-arm).

    Separate graphs can be captured on the same stream and replayed concurrently on different streams. Their
    capture IDs must select different storage. Calls on separate streams within one capture also stay disjoint.
    Pool initialization and slice reservation are serialized across host threads. The pool is zeroed and
    acknowledged outside capture; no captured call allocates or zero-fills device state.
    """
    key = _counter_key(device)
    cnt = _COUNTERS.get(key)
    if cnt is not None:
        return cnt
    # Tensor allocation, slicing and event waits can release the GIL. Serialize first-use publication and
    # reservation, or two host threads can select the same ``used`` index for different streams/captures.
    # Established keys keep the lock-free fast path above (including CUDA graph replay setup).
    with _COUNTER_LOCK:
        cnt = _COUNTERS.get(key)
        if cnt is not None:
            return cnt
        pool = _POOL.get(device)
        if pool is None:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("mxfp8_qkv_norm_rope_nvfp4_insert: first call inside CUDA graph capture; call it "
                                   "once eagerly on the device first (the entry's warm-up does)")
            pool = torch.zeros(_COUNTER_SETS * _COUNTER_WORDS, dtype=torch.int32, device=device)
            # A future stream must not race this initialization. Wait only for this stream's zero fill, once.
            ready = torch.cuda.Event()
            ready.record(torch.cuda.current_stream(device))
            ready.synchronize()
            _POOL[device] = pool  # publish only initialized storage to another Python thread
        used = sum(1 for d, _ in _COUNTERS if d == device)
        if used >= _COUNTER_SETS:
            raise RuntimeError(f"mxfp8_qkv_norm_rope_nvfp4_insert: more than {_COUNTER_SETS} stream/capture counter sets on {device}")
        cnt = _COUNTERS[key] = pool[used * _COUNTER_WORDS:(used + 1) * _COUNTER_WORDS]
        return cnt


def _fused(activation, activation_scale, weight, weight_scale, m, q_norm_weight, k_norm_weight,
           index_q_norm_weight, index_k_norm_weight, cos_sin_cache, positions, slot_mapping, index_slot_mapping,
           kv_cache, index_cache, k_scale, v_scale, q_out, index_q_out, eps):
    slab = torch.empty((2 if m >= 129 else _kernel.KS, m, N), dtype=torch.float32, device=activation.device)
    _run_compiled(
        (_fused_launcher(4 if m <= 8 else 8) if m <= 16 else
         _cooperative_launcher((m + 15) // 16 * 16) if m <= 64 else
         _k_wave_launcher(144 if m <= 144 else 192)),
        activation.data_ptr(),
        weight.data_ptr(),
        activation_scale.data_ptr(),
        weight_scale.data_ptr(),
        slab.data_ptr(),
        _counters(activation.device).data_ptr(),
        q_out.data_ptr(),
        index_q_out.data_ptr(),
        q_norm_weight.data_ptr(),
        k_norm_weight.data_ptr(),
        index_q_norm_weight.data_ptr(),
        index_k_norm_weight.data_ptr(),
        cos_sin_cache.data_ptr(),
        positions.data_ptr(),
        slot_mapping.data_ptr(),
        index_slot_mapping.data_ptr(),
        kv_cache.data_ptr(),
        index_cache.data_ptr(),
        k_scale.data_ptr(),
        v_scale.data_ptr(),
        m,
        activation_scale.numel() * activation_scale.element_size(),
        index_q_out.stride(0),
        kv_cache.stride(0),
        index_cache.stride(0),
        cos_sin_cache.shape[0],
        struct.unpack("<i", struct.pack("<f", eps))[0],
        torch.cuda.current_stream(),
    )



@functools.cache
def _reduce_launcher(tp: int, sk: int):
    return _reduce_insert.compile_reduce_insert(tp, sk)


def _reduce(qkv, ins, eps):
    slab, sk = qkv.tmp, qkv.split_k
    (qw, kw, iqw, ikw, cs, pos, slot, islot, kv, idx, ks, vs, q, iq) = ins
    m = slab.shape[1]
    # One pair per norm wave complements the split-3 V halo/CTA join (65-128 rows).
    # Split-2 retains one pair above 176, two at 129-176.
    tp = 1 if sk == 3 or m > 176 else 2
    _run_compiled(
        _reduce_launcher(tp, sk),
        slab.data_ptr(), q.data_ptr(), iq.data_ptr(), qw.data_ptr(), kw.data_ptr(), iqw.data_ptr(), ikw.data_ptr(),
        cs.data_ptr(), pos.data_ptr(), slot.data_ptr(), islot.data_ptr(), kv.data_ptr(), idx.data_ptr(), ks.data_ptr(),
        vs.data_ptr(), m, iq.stride(0), kv.stride(0), idx.stride(0), cs.shape[0],
        struct.unpack("<i", struct.pack("<f", eps))[0], torch.cuda.current_stream(),
    )

def mxfp8_qkv_norm_rope_nvfp4_insert(
    activation: torch.Tensor,  # e4m3 codes in the (16, 16) layout, prequant_code_tile_rows(m, 2560, 6144) rows
    activation_scale: torch.Tensor,  # shuffled E8M0 scales [ceil(rows / 32) * 32, 192]
    weight: torch.Tensor,  # preshuffle_mxfp8_weight codes [2560, 6144]
    weight_scale: torch.Tensor,  # its shuffled scales
    m: int,
    q_norm_weight: torch.Tensor,
    k_norm_weight: torch.Tensor,
    index_q_norm_weight: torch.Tensor,
    index_k_norm_weight: torch.Tensor,
    cos_sin_cache: torch.Tensor,  # [max_pos, 64] bf16
    positions: torch.Tensor,  # [m] int64
    slot_mapping: torch.Tensor,  # [m] int64, -1 skips the token
    index_slot_mapping: torch.Tensor,  # [m] int64, -1 skips the token
    kv_cache: torch.Tensor,  # [blocks, 2, 128, 72] uint8
    index_cache: torch.Tensor,  # [blocks, 128, 72] uint8
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    q_out: torch.Tensor,  # [m, 2048] bf16
    index_q_out: torch.Tensor,  # [m, 128] e4m3, row stride free
    *,
    num_heads: int,
    num_kv_heads: int,
    num_index_heads: int,
    eps: float,
) -> None:
    """``mxfp8_gemm_shuffled(defer_reduce=True)`` + the reduce-insert (or the bf16 GEMM + ``norm_rope_nvfp4_insert``
    where the table does not defer), bit for bit; one launch at 1-64 and 129-192 rows."""
    if m == 0:
        return
    assert (num_heads, num_kv_heads, num_index_heads) == (_kernel.NQ, _kernel.NKV, _kernel.NIQ)
    assert tuple(weight.shape) == (N, K) and activation.shape[1] == K
    assert cos_sin_cache.shape[1] == _kernel.ROT
    if not 1 <= m <= MAX_ROWS or not _gemm.prequant_code_tile_rows(m, N, K, torch.bfloat16):
        raise ValueError(f"no shuffled-code QKV row at M {m}")
    ins = (q_norm_weight, k_norm_weight, index_q_norm_weight, index_k_norm_weight, cos_sin_cache, positions,
           slot_mapping, index_slot_mapping, kv_cache, index_cache, k_scale, v_scale, q_out, index_q_out)
    kw = dict(num_heads=num_heads, num_kv_heads=num_kv_heads, num_index_heads=num_index_heads, eps=eps)
    if m <= FUSED_MAX_ROWS or 129 <= m <= 192:
        assert eps >= 2.0**-126  # the kernel's rsqrt skips ocml's denormal scaling
        assert activation.shape[0] >= 16 and activation.is_contiguous() and weight.is_contiguous()
        assert q_out.dtype == torch.bfloat16 and q_out.is_contiguous()
        assert index_q_out.dtype == torch.float8_e4m3fn and index_q_out.stride(1) == 1
        assert index_q_out.stride(0) % 4 == 0
        assert kv_cache.shape[1:] == (2, 128, 72) and kv_cache.stride(1) == 128 * 72 and kv_cache.stride(3) == 1
        assert index_cache.shape[1:] == (128, 72) and index_cache.stride(1) == 72 and index_cache.stride(2) == 1
        _fused(activation, activation_scale, weight, weight_scale, m, *ins, eps)
        return
    qkv = _gemm.mxfp8_gemm_shuffled(activation, activation_scale, weight, weight_scale, torch.bfloat16, m,
                                    defer_reduce=True)
    if isinstance(qkv, _gemm.SkPartials):
        _reduce(qkv, ins, eps)
    else:
        norm_rope_nvfp4_insert(qkv, *ins, **kw)


def _mode(*args, **kwargs):
    """One mode: the variants depend on the token count only (``amd_kernels.warm_up``)."""
    return ()


def _sizes(mode):
    """Every token count the entry serves: each picks its GEMM row, split and insert pairs from tables keyed by the
    count, so the warm-up walks them all (compile-only calls)."""
    return range(1, MAX_ROWS + 1)


def _inputs(mode, n, activation, activation_scale, weight, weight_scale, m, q_norm_weight, k_norm_weight,
            index_q_norm_weight, index_k_norm_weight, cos_sin_cache, positions, slot_mapping, index_slot_mapping,
            kv_cache, index_cache, k_scale, v_scale, q_out, index_q_out, **kw):
    """One warm-up call at ``n`` tokens: zero codes in the producer's tile, every slot -1 (no page is written)."""
    dev = weight.device
    rows = _gemm.prequant_code_tile_rows(n, N, K, torch.bfloat16)
    rows = (n + rows - 1) // rows * rows
    codes = torch.zeros((rows, K), dtype=activation.dtype, device=dev)
    scales = torch.full(((rows + 31) // 32 * 32, K // 32), 127, dtype=activation_scale.dtype, device=dev)
    zeros = torch.zeros(n, dtype=torch.int64, device=dev)
    skip = torch.full((n,), -1, dtype=torch.int64, device=dev)
    q = torch.empty((n, q_out.shape[1]), dtype=q_out.dtype, device=dev)
    iq = torch.empty((n, index_q_out.shape[1]), dtype=index_q_out.dtype, device=dev)
    return ((codes, scales, weight, weight_scale, n, q_norm_weight, k_norm_weight, index_q_norm_weight,
             index_k_norm_weight, cos_sin_cache, zeros, skip, skip, kv_cache, index_cache, k_scale, v_scale, q, iq),
            kw)


mxfp8_qkv_norm_rope_nvfp4_insert = entry("attention/mxfp8_qkv_norm_rope_nvfp4_insert_n2560_k6144_rot64_p128",
                                         mode=_mode, sizes=_sizes, inputs=_inputs)(mxfp8_qkv_norm_rope_nvfp4_insert)
