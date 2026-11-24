# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Shared validation, compilation and dispatch for NVFP4 sparse prefill."""

import functools
import importlib

import numpy as np
import torch
from amd_kernels.warm_up import entry

from . import switches

D, HG, BLK, TOPK = 128, 16, 128, 16
LOG2E = 1.4426950409
LOG2E_F32 = np.float32(LOG2E)  # immutable score-scale constant, rounded once


def _run_compiled(exe, *args):
    """The first call compiles and runs (``flyc.compile``); later calls run the cached CompiledFunction."""
    import flydsl.compiler as flyc

    cf = getattr(exe, "_cf", None)
    if cf is None:
        exe._cf = flyc.compile(exe, *args)
    else:
        cf(*args)


@functools.cache
def _get_kernel(
    compute_dtype: str,
    pack: bool,
    rounds: int,
    quant: bool,
    q_f16: bool = False,
    sched: bool = True,
    q_bf16: bool = False,
    single: int = 0,
    lazy: bool = False,
):
    # two-round grids: the dimension-split kernel, bf16 P or fp16 P (fp16 compute), bitwise equal to fp16/kernel.py's
    # two-round kernel with or without its scheduling ports and 1.7-4.2% faster (Hkv 1 / 4, 4096-32768 tokens)
    if rounds == 2 and compute_dtype in ("bf16", "fp16"):
        from .f16mfma_gqa16_d128_p128_top16_tq4_dsplit import (
            compile_nvfp4_f16mfma_gqa16_d128_p128_top16_tq4_dsplit,
        )

        p_f16 = compute_dtype == "fp16"
        return compile_nvfp4_f16mfma_gqa16_d128_p128_top16_tq4_dsplit(pack, quant, p_f16, q_f16 and p_f16, lazy)
    module = importlib.import_module(f".{compute_dtype}.kernel", __package__)
    compile_kernel = getattr(module, f"compile_nvfp4_prefill_{compute_dtype}")
    if compute_dtype == "fp8":
        if single:
            if single == 3:
                from .fp8mfma_gqa16_d128_p128_top16_tq3_dsplit import (
                    compile_nvfp4_fp8mfma_gqa16_d128_p128_top16_tq3_dsplit,
                )
                return compile_nvfp4_fp8mfma_gqa16_d128_p128_top16_tq3_dsplit(pack, 2, quant, q_bf16)
            if single == 2:
                from .fp8mfma_gqa16_d128_p128_top16_tq1_qlds import (
                    compile_nvfp4_fp8mfma_gqa16_d128_p128_top16_tq1_qlds,
                )
                return compile_nvfp4_fp8mfma_gqa16_d128_p128_top16_tq1_qlds(pack, quant)
            from .fp8.single_walk import compile_nvfp4_sparse_gqa16_d128_p128_top16_tq1_walk

            return compile_nvfp4_sparse_gqa16_d128_p128_top16_tq1_walk(pack, quant, q_bf16)
        if rounds == 2:
            from .fp8.union_walk import compile_nvfp4_sparse_gqa16_d128_p128_top16_tq4_walk

            return compile_nvfp4_sparse_gqa16_d128_p128_top16_tq4_walk(pack, 2, quant, q_bf16)
        if q_bf16:
            from .fp8mfma_gqa16_d128_p128_top16_tq2_qlds import (
                compile_nvfp4_fp8mfma_gqa16_d128_p128_top16_tq2_qlds,
            )
            return compile_nvfp4_fp8mfma_gqa16_d128_p128_top16_tq2_qlds(pack, quant)
        return compile_kernel(pack, 1, quant, q_bf16)
    if compute_dtype == "fp16":
        return compile_kernel(pack, rounds, quant, q_f16, sched)
    return compile_kernel(pack, rounds, quant)


@functools.cache
def _get_bf16_local(tq: int, pack: bool, quant: bool, p_f16: bool = False, q_f16: bool = False, lazy: bool = False,
                    q_shared: bool = False):
    """One- / two- / three-query tiles with bf16 or fp16 P, optional shared Q staging and opt-in lazy rescale."""
    if tq == 3:
        from .f16mfma_gqa16_d128_p128_top16_tq3_dsplit import (
            compile_nvfp4_f16mfma_gqa16_d128_p128_top16_tq3_dsplit,
        )

        return compile_nvfp4_f16mfma_gqa16_d128_p128_top16_tq3_dsplit(pack, quant, p_f16, q_f16, lazy)
    if q_shared:
        assert tq == 2
        from .f16mfma_gqa16_d128_p128_top16_tq2_qlds import (
            compile_nvfp4_f16mfma_gqa16_d128_p128_top16_tq2_qlds,
        )

        return compile_nvfp4_f16mfma_gqa16_d128_p128_top16_tq2_qlds(pack, quant, p_f16, q_f16, lazy)
    if tq == 1 and p_f16:
        if quant:
            from .f16mfma_gqa16_d128_p128_top16_tq1_qpre import (
                compile_nvfp4_f16mfma_gqa16_d128_p128_top16_tq1_qpre,
            )

            return compile_nvfp4_f16mfma_gqa16_d128_p128_top16_tq1_qpre(q_f16)
        from .f16mfma_gqa16_d128_p128_top16_tq1_walk import (
            compile_nvfp4_f16mfma_gqa16_d128_p128_top16_tq1_walk,
        )

        return compile_nvfp4_f16mfma_gqa16_d128_p128_top16_tq1_walk(pack, quant, q_f16)
    module = importlib.import_module(f".bf16mfma_gqa16_d128_p128_top16_tq{tq}", __package__)
    compile_kernel = getattr(
        module, f"compile_nvfp4_bf16mfma_gqa16_d128_p128_top16_tq{tq}"
    )
    if lazy:
        assert tq == 2 and p_f16
        return compile_kernel(pack, 1, quant, p_f16, q_f16, lazy)
    return compile_kernel(pack, 1, quant, p_f16, q_f16)


@functools.cache
def _num_cus(device: torch.device) -> int:
    return torch.cuda.get_device_properties(device).multi_processor_count


@functools.cache
def _on_gfx950() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_properties(
        torch.cuda.current_device()
    ).gcnArchName.startswith("gfx950")


def supports_nvfp4_sparse_prefill(
    num_heads: int, num_kv_heads: int, head_dim: int, topk_blocks: int, block_size: int
) -> bool:
    """Whether the kernel serves this layer: gfx950, 128-dim heads, 16 query heads per KV head, 16 selected 128-token
    blocks per token."""
    return (
        _on_gfx950()
        and head_dim == D
        and num_heads == num_kv_heads * HG
        and topk_blocks == TOPK
        and block_size == BLK
    )


@functools.cache
def _unit_scale(device: torch.device) -> torch.Tensor:
    return torch.ones((), dtype=torch.float32, device=device)


@functools.cache
def _fp8_lut(device: torch.device) -> torch.Tensor:
    from .query import nvfp4_e4m3_lut

    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("Warm up FP8 prefill before graph capture")
    return nvfp4_e4m3_lut(device)


def nvfp4_sparse_prefill(
    q: torch.Tensor,  # [T, H, 128] bf16 (fp16 when the producer rounded it directly)
    kv_cache: torch.Tensor,  # [pages, 2 * Hkv, 128, 72] uint8 (reference.py)
    topk: torch.Tensor,  # [Hkv, T, 16] int32 logical blocks, -1 = none
    block_table: torch.Tensor,  # [reqs, max_blocks] int32
    query_req_id: torch.Tensor,  # [T] int32, row of block_table per token
    query_abs_pos: torch.Tensor,  # [T] int32
    num_kv_heads: int,
    sm_scale: float,
    out: torch.Tensor,  # [T, H, 128] bf16, written
    k_scale: torch.Tensor | None = None,  # fp32 global scales (None = 1)
    v_scale: torch.Tensor | None = None,
    pack_out: torch.Tensor | None = None,  # [>= T, H * 132] uint8, see below
    quant_out: tuple[torch.Tensor, torch.Tensor] | None = None,
    *,
    compute_dtype: str = "fp16",
    sched: bool = False,
    lazy_rescale: bool = False,
) -> None:
    """NVFP4 sparse prefill attention with a bf16 output (or a fused MXFP8 one) and a selectable compute precision.

    fp8 also takes a prepacked uint8 Q (``query.pack_query``: each eight E4M3 bytes ordered [0, 2, 4, 6, 1, 3, 5, 7]);
    a bf16 Q is quantized here. A packed or quantized output leaves ``out`` and the padding rows untouched. ``sched``
    runs the fp16 kernel's scheduling ports (bitwise equal) while ``switches.PREFILL_MSA_SCHED`` is on.
    ``lazy_rescale`` (fp16 only, opt-in: another numeric rule) computes fp16 P against a running reference that is
    re-based only when a row max passes it by 2^8, on the two- / three-query tiles and the two-round dimension-split kernel
    (the one-query tile and the one-round kernels keep the eager rescale).
    """
    if compute_dtype not in ("fp16", "bf16", "fp8"):
        raise ValueError(f"Unsupported prefill compute precision: {compute_dtype}")
    if lazy_rescale and compute_dtype != "fp16":
        raise ValueError("lazy_rescale needs compute_dtype='fp16' (fp16 P)")
    n_tok, n_heads, head_dim = q.shape
    if n_tok == 0:
        return
    pack = pack_out is not None
    quant = quant_out is not None
    assert not (pack and quant)
    if quant:
        codes, scales = quant_out  # type: ignore[misc]
        assert codes.dtype == scales.dtype == torch.uint8
        assert codes.is_contiguous() and scales.is_contiguous()
        assert codes.shape[0] >= n_tok and codes.shape[1] == n_heads * head_dim
        assert scales.shape[0] >= (n_tok + 31) // 32 * 32
        assert scales.shape[1] == n_heads * head_dim // 32
    if pack_out is not None:
        assert pack_out.dtype == torch.uint8 and pack_out.is_contiguous()
        assert pack_out.shape[0] >= n_tok and pack_out.shape[1] == n_heads * 132
    assert head_dim == D and n_heads == num_kv_heads * HG
    assert (
        q.dtype == torch.bfloat16
        or (compute_dtype == "fp8" and q.dtype == torch.uint8)
        or (compute_dtype == "fp16" and q.dtype == torch.float16)
    )
    assert out.dtype == torch.bfloat16
    assert out.shape == q.shape
    assert q.is_contiguous() and out.is_contiguous()
    assert kv_cache.dtype == torch.uint8 and tuple(kv_cache.shape[1:]) == (
        2 * num_kv_heads,
        BLK,
        72,
    )
    assert kv_cache.stride(1) == BLK * 72 and kv_cache.stride(3) == 1
    assert topk.dtype == torch.int32 and tuple(topk.shape) == (
        num_kv_heads,
        n_tok,
        TOPK,
    )
    assert topk.stride(2) == 1 and topk.stride(1) == TOPK
    for t in (block_table, query_req_id, query_abs_pos):
        assert t.dtype == torch.int32 and t.stride(-1) == 1
    assert kv_cache.stride(0) < 2**31
    result = out
    fused_pack, fused_quant = pack, quant
    extra_ptr = 0
    if fused_quant:
        extra_ptr = scales.data_ptr()
    elif fused_pack:
        assert pack_out is not None
        extra_ptr = pack_out.data_ptr()
    unit = _unit_scale(q.device)
    ks = unit if k_scale is None else k_scale
    vs = unit if v_scale is None else v_scale
    # c = fp32(sm_scale) * fp32(log2 e), one rounding
    c = float(np.float32(sm_scale) * LOG2E_F32)
    # two rounds (two workgroups per CU) once the grid exceeds the CUs; up to one per CU, one round is ~1 us faster
    n_wg = (n_tok + 3) // 4 * num_kv_heads
    rounds = 2 if n_wg > _num_cus(q.device) else 1
    fp8_args = (_fp8_lut(q.device).data_ptr(),) if compute_dtype == "fp8" else ()
    # Hkv 1 small tiles by the measured crossovers: bf16 / fp16 (not with the fp16 scheduling ports) one query to
    # 256 tokens, two to 1025, three to 1536; fp8 one to 512
    use_sched = sched and switches.PREFILL_MSA_SCHED and compute_dtype != "fp8"
    local = (compute_dtype == "bf16" or (compute_dtype == "fp16" and not use_sched)) and num_kv_heads == 1 \
        and n_tok <= 1536
    q_bf16 = compute_dtype == "fp8" and q.dtype == torch.bfloat16
    # the lazy rescale runs on the two- / three-query tiles and the two-round kernel; the default call keeps its arguments
    lazy_local = {"lazy": True} if lazy_rescale and n_tok > 256 else {}
    lazy_grid = {"lazy": True} if lazy_rescale and rounds == 2 else {}
    # Shared Q conversion wins above one two-query workgroup per CU (513-1025 tokens; see the README).
    shared_local = {"q_shared": True} if 512 < n_tok <= 1025 else {}
    exe = (_get_bf16_local(1 if n_tok <= 256 else 2 if n_tok <= 1025 else 3, fused_pack, fused_quant, compute_dtype == "fp16",
                           q.dtype == torch.float16, **lazy_local, **shared_local)
           if local else _get_kernel(
            compute_dtype,
            fused_pack,
            rounds,
            fused_quant,
            q_f16=q.dtype == torch.float16,
            sched=use_sched,
            q_bf16=q_bf16,
            # 0: multi-query; 1: register Q; 2: LDS Q; 3: three-query tile
            single=(
                3 if compute_dtype == "fp8" and num_kv_heads == 1 and 1087 < n_tok <= 1536
                else (2 if q_bf16 and n_tok > 256 else 1)
                if compute_dtype == "fp8" and num_kv_heads == 1 and n_tok <= 512 else 0
            ),
            **lazy_grid,
        ))
    _run_compiled(
        exe,
        q.data_ptr(),
        kv_cache.data_ptr(),
        topk.data_ptr(),
        block_table.data_ptr(),
        query_req_id.data_ptr(),
        query_abs_pos.data_ptr(),
        codes.data_ptr() if fused_quant else result.data_ptr(),
        ks.data_ptr(),
        vs.data_ptr(),
        *fp8_args,
        extra_ptr,
        n_tok,
        n_heads,
        num_kv_heads,
        topk.stride(0),
        block_table.stride(0),
        kv_cache.stride(0),
        c,
        torch.cuda.current_stream(q.device),
    )


# ---- warm-up (root README rule 4)

# token counts that reach every bf16 / fp16 variant at one and four KV heads: both combines, the one- / two- / three-query
# tiles; fp8 at one KV head also has the 513-1024 and 1088-1536 tiles
WARM_TOKENS = (8, 512, 1024, 1536, 2048)
WARM_TOKENS_FP8 = (8, 512, 1024, 1536, 2048)


def _warm_tokens(compute_dtype: str) -> tuple[int, ...]:
    return WARM_TOKENS_FP8 if compute_dtype == "fp8" else WARM_TOKENS


def _prefill_mode(q, kv_cache, topk, block_table, query_req_id, query_abs_pos, num_kv_heads, sm_scale, out,
                  k_scale=None, v_scale=None, pack_out=None, quant_out=None, *, compute_dtype="fp16", sched=False,
                  lazy_rescale=False):
    """What fixes the variants besides the token count (``amd_kernels.warm_up``)."""
    return (q.shape[1], num_kv_heads, compute_dtype, q.dtype, pack_out is not None, quant_out is not None, sched,
            q.device.index, lazy_rescale)


def _prefill_sizes(mode):
    return _warm_tokens(mode[2])


def _prefill_inputs(mode, n, q, *args, **kwargs):
    """One call at ``n`` tokens: zero pages, one live block per token."""
    heads, hkv, compute, qdtype, pack, quant, sched, _, lazy = mode
    dev = q.device
    q0 = torch.zeros((n, heads, D), dtype=qdtype, device=dev)
    kv = torch.zeros((2, 2 * hkv, BLK, 72), dtype=torch.uint8, device=dev)
    idx = torch.full((hkv, n, TOPK), -1, dtype=torch.int32, device=dev)
    idx[..., 0] = 0
    zeros = torch.zeros((n,), dtype=torch.int32, device=dev)
    one = torch.ones((), dtype=torch.float32, device=dev)
    width = heads * D
    pack_out = torch.empty((n, heads * 132), dtype=torch.uint8, device=dev) if pack else None
    quant_out = ((torch.empty((n, width), dtype=torch.uint8, device=dev),
                  torch.empty(((n + 31) // 32 * 32, width // 32), dtype=torch.uint8, device=dev))
                 if quant else None)
    out = torch.empty((n, heads, D), dtype=torch.bfloat16, device=dev)
    return ((q0, kv, idx, torch.zeros((1, 1), dtype=torch.int32, device=dev), zeros, zeros, hkv, D**-0.5, out, one,
             one, pack_out, quant_out), dict(compute_dtype=compute, sched=sched, lazy_rescale=lazy))


nvfp4_sparse_prefill = entry(
    "attention/nvfp4_sparse_prefill_gqa16_d128_p128_top16",
    mode=_prefill_mode, sizes=_prefill_sizes, inputs=_prefill_inputs,
)(nvfp4_sparse_prefill)


def warm_up_prefill(device: torch.device, *, num_heads: int, num_kv_heads: int, compute_dtype: str = "fp16",
                    lazy_rescale: bool = False) -> None:
    """Compile every prefill variant of ``compute_dtype`` with the bf16 output: one call per warm-up token count."""
    dev = torch.device(device)
    with torch.inference_mode():
        kv = torch.zeros((2, 2 * num_kv_heads, BLK, 72), dtype=torch.uint8, device=dev)
        one = torch.ones((), dtype=torch.float32, device=dev)
        for n in _warm_tokens(compute_dtype):
            q = torch.zeros((n, num_heads, D), dtype=torch.bfloat16, device=dev)
            idx = torch.full((num_kv_heads, n, TOPK), -1, dtype=torch.int32, device=dev)
            idx[..., 0] = 0
            zeros = torch.zeros((n,), dtype=torch.int32, device=dev)
            nvfp4_sparse_prefill(q, kv, idx, torch.zeros((1, 1), dtype=torch.int32, device=dev), zeros, zeros,
                                 num_kv_heads, D**-0.5, torch.empty_like(q), one, one, compute_dtype=compute_dtype,
                                 lazy_rescale=lazy_rescale)


# per-precision entry points
nvfp4_sparse_prefill_bf16 = functools.partial(nvfp4_sparse_prefill, compute_dtype="bf16")
nvfp4_sparse_prefill_fp16 = functools.partial(nvfp4_sparse_prefill, compute_dtype="fp16")
nvfp4_sparse_prefill_fp8 = functools.partial(nvfp4_sparse_prefill, compute_dtype="fp8")
