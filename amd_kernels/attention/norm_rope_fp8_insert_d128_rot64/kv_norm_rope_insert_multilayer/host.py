# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Compile cache + launch wrapper of the FlyDSL DSpark context-KV tail."""

import functools

import torch

from amd_kernels.warm_up import entry

from . import kernel as _kernel

# Selected by tests/y18_ctx_kv_test.py against the C++ rotary kernel's fp
# contraction (0: Inductor order, 1: the other). The tested value is the default.
_ROPE_FORM = 0


def _run_compiled(exe, *args):
    """First call compiles and runs (``flyc.compile``); later calls dispatch the
    cached CompiledFunction."""
    import flydsl.compiler as flyc

    cf = getattr(exe, "_cf", None)
    if cf is None:
        exe._cf = flyc.compile(exe, *args)
    else:
        cf(*args)


@functools.cache
def _get_kernel(nl: int, nkv: int, block_size: int, rope_form: int, split_k: int = 0):
    return _kernel.compile_kv_norm_rope_insert_multilayer(nl, nkv, block_size, rope_form, split_k)


@functools.cache
def _pointer_table(ptrs: tuple[int, ...], device: torch.device) -> torch.Tensor:
    """Cache only persistent KV and scale pointers, never batch slot views."""
    return torch.tensor(ptrs, dtype=torch.int64, device=device)


def _kv_norm_rope_insert_multilayer(
    kv,  # [n, nl * 2 * nkv * 128] bf16 context-KV GEMM output, or its gemm.mxfp8_common.SkPartials
    positions: torch.Tensor,  # [n] int64
    k_norm_weight: torch.Tensor,  # [nl, 128] bf16
    cos_sin_cache: torch.Tensor,  # [rows, 64] bf16
    kv_caches: list[torch.Tensor],  # nl x [pages, nkv, BS, 256] e4m3
    slot_mappings: list[torch.Tensor],  # nl x [>= n] int64 (< 0 skips)
    k_scales: list[torch.Tensor],  # nl x fp32 scalar
    v_scales: list[torch.Tensor],
    eps: float,
    n_insert: int | None = None,
) -> None:
    """Norm + RoPE + fp8 insert of the context rows into every layer's cache.
    ``n_insert``: only the first n_insert rows have valid slots (None: all).
    ``kv`` may be the GEMM's deferred split-K partials (``mxfp8_gemm_shuffled(...,
    defer_reduce=True)``): the kernel then sums the fp32 slabs in the split-K
    reduce's order, so the caches get the same bytes as reduce-then-insert."""
    import struct

    split_k = 0
    if hasattr(kv, "split_k") and hasattr(kv, "tmp"):
        assert kv.out_dtype == torch.bfloat16
        split_k = kv.split_k
        tmp = kv.tmp
        assert tmp.dtype == torch.float32 and tmp.is_contiguous()
        assert tmp.shape == (split_k, kv.m, kv.n)
        kv = tmp[0]  # [n, width] view: the shape, stride and base pointer of slab 0
    n = kv.shape[0]
    nl = len(kv_caches)
    if n == 0 or nl == 0:
        return
    nkv = kv_caches[0].shape[1]
    block_size = kv_caches[0].shape[2]
    assert kv.dtype == (torch.float32 if split_k else torch.bfloat16) and kv.is_contiguous()
    assert kv.shape[1] == nl * 2 * nkv * _kernel.D
    assert positions.dtype == torch.int64 and positions.is_contiguous()
    assert k_norm_weight.shape == (nl, _kernel.D) and k_norm_weight.is_contiguous()
    assert cos_sin_cache.dtype == torch.bfloat16
    assert cos_sin_cache.shape[1] == _kernel.ROT
    if n_insert is None:
        n_insert = n
    for c in kv_caches:
        assert c.stride(0) == kv_caches[0].stride(0)
        assert c.stride(1) == kv_caches[0].stride(1)
        assert c.stride(2) == kv_caches[0].stride(2)
        assert c.stride(3) == 1 and c.shape[3] == _kernel.TOK_B
    ptrs = tuple(
        [c.data_ptr() for c in kv_caches]
        + [s.data_ptr() for s in k_scales]
        + [s.data_ptr() for s in v_scales]
    )
    table = _pointer_table(ptrs, kv.device)
    slots = slot_mappings[0]
    slot_stride = 0
    if any(s.data_ptr() != slots.data_ptr() for s in slot_mappings):
        # Distinct layer mappings are packed on-device, without host copies.
        slots = torch.stack([s[:n_insert] for s in slot_mappings])
        slot_stride = slots.stride(0)
    _run_compiled(
        _get_kernel(nl, nkv, block_size, _ROPE_FORM, split_k),
        kv.data_ptr(),
        positions.data_ptr(),
        k_norm_weight.data_ptr(),
        cos_sin_cache.data_ptr(),
        table.data_ptr(),
        slots.data_ptr(),
        slot_stride,
        n,
        cos_sin_cache.shape[0],
        kv_caches[0].stride(0),
        kv_caches[0].stride(1),
        kv_caches[0].stride(2),
        kv.stride(0),
        struct.unpack("<I", struct.pack("<f", eps))[0],
        n_insert,
        torch.cuda.current_stream(kv.device),
    )


def supports_multilayer(kv_caches: list[torch.Tensor], k_scales: list[torch.Tensor],
                        v_scales: list[torch.Tensor]) -> bool:
    """Whether ``kv_norm_rope_insert_multilayer`` takes these caches: e4m3 (or its uint8 view) ``[pages, kv_heads,
    page, 256]``, unit last stride, every layer the same shape and strides, fp32 scales."""
    if not kv_caches or not len(kv_caches) == len(k_scales) == len(v_scales):
        return False
    ref = (tuple(kv_caches[0].shape[1:]), kv_caches[0].stride())
    return all(
        c.numel() > 0 and c.dtype in (torch.uint8, torch.float8_e4m3fn) and c.dim() == 4
        and c.shape[3] == _kernel.TOK_B and c.stride(3) == 1 and (tuple(c.shape[1:]), c.stride()) == ref
        for c in kv_caches
    ) and all(s.dtype == torch.float32 for s in (*k_scales, *v_scales))


def _ml_mode(kv, positions, k_norm_weight, cos_sin_cache, kv_caches, slot_mappings, k_scales, v_scales, eps,
             n_insert=None):
    """What a call fixes about its kernel (``amd_kernels.warm_up``; the row count does not)."""
    split_k = kv.split_k if hasattr(kv, "split_k") and hasattr(kv, "tmp") else 0
    return len(kv_caches), kv_caches[0].shape[1], kv_caches[0].shape[2], split_k


def _ml_inputs(mode, n, kv, positions, k_norm_weight, cos_sin_cache, kv_caches, slot_mappings, k_scales, v_scales,
               eps, n_insert=None):
    """``n`` rows of zeros with slot -1 (writes nothing; the caller's caches, nothing launches)."""
    from types import SimpleNamespace

    nl, nkv, _, split_k = mode
    dev = positions.device
    width = nl * 2 * nkv * _kernel.D
    if split_k:
        x = SimpleNamespace(split_k=split_k, tmp=torch.zeros((split_k, n, width), dtype=torch.float32, device=dev),
                            m=n, n=width, out_dtype=torch.bfloat16)
    else:
        x = torch.zeros((n, width), dtype=torch.bfloat16, device=dev)
    pos = torch.zeros(n, dtype=torch.int64, device=dev)
    slots = torch.full((n,), -1, dtype=torch.int64, device=dev)
    return (x, pos, k_norm_weight, cos_sin_cache, kv_caches, [slots] * nl, k_scales, v_scales, eps), {}


kv_norm_rope_insert_multilayer = entry("attention/norm_rope_fp8_insert_d128_rot64:multilayer", mode=_ml_mode,
                                       sizes=lambda mode: (8,), inputs=_ml_inputs)(_kv_norm_rope_insert_multilayer)
