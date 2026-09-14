# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Persistent scratch for Flash-Next decode CUDA-graph capture/replay.

QSA already has ``qsa_graph_workspace``. This module covers the *other*
ephemeral ``torch.empty`` / ``empty_like`` / ``contiguous()`` sites that
still run inside the captured forward and become dangling aliases after a
long eager prefill reclaims the caching-allocator slab:

  * GDN ``core_attn_out = torch.empty_like(z)``
  * HyperConnection mix / combine outs
  * Grouped Gemma RMSNorm outs
  * QSA expand-index outs when no caller buffer is provided
  * Q/K/V ``.contiguous()`` after ``torch.split`` of the fused QKVG projection
  * Indexer RoPE scratch / mRoPE position transpose
  * PLE ``short_conv_decode`` history / conv / index_select temps
  * Triton MoE ``empty_like`` output + ``intermediate_cache``

Layers run sequentially, so one shared buffer per kind is enough.
"""

from __future__ import annotations

import threading

import torch

_lock = threading.Lock()
_enabled = False
_device: torch.device | None = None
_max_tokens = 0
_hidden = 0
_hc_width = 0
_v_heads = 0
_head_v_dim = 0
_expand_width = 0
_q_dim = 0
_kv_dim = 0
_rope_dim = 0
_head_norm_rows = 0
_head_norm_dim = 0
_k_heads = 0
_head_k_dim = 0
_dtype: torch.dtype = torch.bfloat16

_gdn_out: torch.Tensor | None = None
_hc_mix_out: torch.Tensor | None = None
_hc_combine_out: torch.Tensor | None = None
_rmsnorm_out: torch.Tensor | None = None
_expand_out: torch.Tensor | None = None
_q_buf: torch.Tensor | None = None
_k_buf: torch.Tensor | None = None
_v_buf: torch.Tensor | None = None
_rope_scratch: torch.Tensor | None = None
_pos_scratch: torch.Tensor | None = None
# Generic 2-D staging for HC / RMSNorm input ``.contiguous()`` copies.
_contig_a: torch.Tensor | None = None
_contig_b: torch.Tensor | None = None
_contig_c: torch.Tensor | None = None
_contig_width = 0
_head_norm_out: torch.Tensor | None = None
_head_norm_out_b: torch.Tensor | None = None
_head_norm_slot: int = 0
_gate_buf: torch.Tensor | None = None
_gdn_q: torch.Tensor | None = None
_gdn_k: torch.Tensor | None = None
_gdn_v: torch.Tensor | None = None
_gdn_g: torch.Tensor | None = None
_gdn_beta: torch.Tensor | None = None
_ple_state_len = 0
_ple_hist: torch.Tensor | None = None
_ple_prev: torch.Tensor | None = None
_ple_out: torch.Tensor | None = None
_ple_idx_in: torch.Tensor | None = None
_ple_idx_out: torch.Tensor | None = None
_ple_valid: torch.Tensor | None = None
_ple_keep: torch.Tensor | None = None
_ple_mask: torch.Tensor | None = None
_ple_i64: torch.Tensor | None = None
_ple_emb: torch.Tensor | None = None
_ple_bool: torch.Tensor | None = None
_ple_arange: torch.Tensor | None = None
_ple_key: torch.Tensor | None = None
_ple_value: torch.Tensor | None = None
_ple_gated: torch.Tensor | None = None
_moe_topk = 0
_moe_half_n = 0
_moe_out: torch.Tensor | None = None
_moe_inter: torch.Tensor | None = None


def enable() -> None:
    global _enabled
    _enabled = True


def disable() -> None:
    global _enabled
    _enabled = False


def is_enabled() -> bool:
    return _enabled


def fits(
    rows: int,
    *,
    hidden: int | None = None,
    hc_width: int | None = None,
    v_heads: int | None = None,
    head_v_dim: int | None = None,
    expand_width: int | None = None,
    q_dim: int | None = None,
    kv_dim: int | None = None,
    rope_dim: int | None = None,
    contig_width: int | None = None,
    dtype: torch.dtype | None = None,
) -> bool:
    if not _enabled or _gdn_out is None:
        return False
    if int(rows) > _max_tokens:
        return False
    if hidden is not None and int(hidden) > _hidden:
        return False
    if hc_width is not None and int(hc_width) > _hc_width:
        return False
    if v_heads is not None and int(v_heads) > _v_heads:
        return False
    if head_v_dim is not None and int(head_v_dim) > _head_v_dim:
        return False
    if expand_width is not None and int(expand_width) > _expand_width:
        return False
    if q_dim is not None and int(q_dim) > _q_dim:
        return False
    if kv_dim is not None and int(kv_dim) > _kv_dim:
        return False
    if rope_dim is not None and int(rope_dim) > _rope_dim:
        return False
    if contig_width is not None and int(contig_width) > _contig_width:
        return False
    if dtype is not None and dtype != _dtype:
        return False
    return True


def fits_head_norm(
    rows: int, dim: int, dtype: torch.dtype | None = None
) -> bool:
    """GemmaRMSNorm flattens ``[tokens, heads, dim]`` to ``tokens*heads`` rows."""
    if not _enabled or _head_norm_out is None or _head_norm_out_b is None:
        return False
    if int(rows) > _head_norm_rows or int(dim) > _head_norm_dim:
        return False
    if dtype is not None and dtype != _dtype:
        return False
    return True


def reserve(
    *,
    max_tokens: int,
    hidden: int,
    hc_count: int,
    v_heads: int,
    head_v_dim: int,
    expand_width: int,
    device: torch.device,
    dtype: torch.dtype = torch.bfloat16,
    q_dim: int = 0,
    kv_dim: int = 0,
    rope_dim: int = 0,
    head_norm_rows: int = 0,
    head_norm_dim: int = 0,
    k_heads: int = 0,
    head_k_dim: int = 0,
    ple_state_len: int = 0,
    moe_topk: int = 0,
    moe_intermediate: int = 0,
) -> None:
    """Reserve (or grow once) decode scratch before CUDA-graph capture."""
    global _device, _max_tokens, _hidden, _hc_width, _v_heads, _head_v_dim
    global _expand_width, _dtype, _q_dim, _kv_dim, _rope_dim, _contig_width
    global _head_norm_rows, _head_norm_dim, _k_heads, _head_k_dim
    global _gdn_out, _hc_mix_out, _hc_combine_out, _rmsnorm_out, _expand_out
    global _q_buf, _k_buf, _v_buf, _rope_scratch, _pos_scratch
    global _contig_a, _contig_b, _contig_c
    global _head_norm_out, _head_norm_out_b, _head_norm_slot, _gate_buf
    global _gdn_q, _gdn_k, _gdn_v, _gdn_g, _gdn_beta
    global _ple_state_len, _ple_hist, _ple_prev, _ple_out
    global _ple_idx_in, _ple_idx_out, _ple_valid, _ple_keep, _ple_mask
    global _ple_i64, _ple_emb, _ple_bool, _ple_arange
    global _ple_key, _ple_value, _ple_gated
    global _moe_topk, _moe_half_n, _moe_out, _moe_inter

    max_tokens = max(int(max_tokens), 1)
    hidden = max(int(hidden), 1)
    hc_width = max(int(hc_count), 1) * hidden
    v_heads = max(int(v_heads), 1)
    head_v_dim = max(int(head_v_dim), 1)
    expand_width = max(int(expand_width), 1)
    q_dim = max(int(q_dim), hidden)  # fallback: at least hidden
    kv_dim = max(int(kv_dim), 1)
    rope_dim = max(int(rope_dim), head_v_dim)
    contig_width = max(hc_width, hidden, q_dim, kv_dim)
    k_heads = max(int(k_heads), 1)
    head_k_dim = max(int(head_k_dim), head_v_dim)
    head_norm_rows = max(int(head_norm_rows), max_tokens * max(q_dim // max(rope_dim, 1), 1))
    head_norm_dim = max(int(head_norm_dim), rope_dim)
    ple_state_len = max(int(ple_state_len), 0)
    moe_topk = max(int(moe_topk), 0)
    moe_half_n = max(int(moe_intermediate) // 2, 0)

    with _lock:
        need = (
            _gdn_out is None
            or _head_norm_out is None
            or _head_norm_out_b is None
            or _gate_buf is None
            or _gdn_g is None
            or _gdn_beta is None
            or _device != device
            or _dtype != dtype
            or _max_tokens < max_tokens
            or _hidden < hidden
            or _hc_width < hc_width
            or _v_heads < v_heads
            or _head_v_dim < head_v_dim
            or _expand_width < expand_width
            or _q_dim < q_dim
            or _kv_dim < kv_dim
            or _rope_dim < rope_dim
            or _contig_width < contig_width
            or _head_norm_rows < head_norm_rows
            or _head_norm_dim < head_norm_dim
            or _k_heads < k_heads
            or _head_k_dim < head_k_dim
            or (
                ple_state_len > 0
                and (
                    _ple_hist is None
                    or _ple_i64 is None
                    or _ple_emb is None
                    or _ple_key is None
                    or _ple_state_len < ple_state_len
                )
            )
            or (moe_topk > 0 and moe_half_n > 0 and (
                _moe_out is None or _moe_inter is None
                or _moe_topk < moe_topk or _moe_half_n < moe_half_n
            ))
        )
        if not need:
            enable()
            return
        if _gdn_out is not None:
            import logging

            logging.getLogger(__name__).warning(
                "Flash decode graph workspace grew after init "
                "(tokens=%s hidden=%s); existing CUDA graphs may be stale",
                max(_max_tokens, max_tokens),
                max(_hidden, hidden),
            )
        _max_tokens = max(_max_tokens, max_tokens)
        _hidden = max(_hidden, hidden)
        _hc_width = max(_hc_width, hc_width)
        _v_heads = max(_v_heads, v_heads)
        _head_v_dim = max(_head_v_dim, head_v_dim)
        _expand_width = max(_expand_width, expand_width)
        _q_dim = max(_q_dim, q_dim)
        _kv_dim = max(_kv_dim, kv_dim)
        _rope_dim = max(_rope_dim, rope_dim)
        _contig_width = max(_contig_width, contig_width)
        _head_norm_rows = max(_head_norm_rows, head_norm_rows)
        _head_norm_dim = max(_head_norm_dim, head_norm_dim)
        _k_heads = max(_k_heads, k_heads)
        _head_k_dim = max(_head_k_dim, head_k_dim)
        _ple_state_len = max(_ple_state_len, ple_state_len)
        _moe_topk = max(_moe_topk, moe_topk)
        _moe_half_n = max(_moe_half_n, moe_half_n)
        _device = device
        _dtype = dtype
        _gdn_out = torch.empty(
            (_max_tokens, _v_heads, _head_v_dim), dtype=dtype, device=device
        )
        _hc_mix_out = torch.empty(
            (_max_tokens, _hidden), dtype=dtype, device=device
        )
        _hc_combine_out = torch.empty(
            (_max_tokens, _hc_width), dtype=dtype, device=device
        )
        _rmsnorm_out = torch.empty(
            (_max_tokens, _hc_width), dtype=dtype, device=device
        )
        _expand_out = torch.empty(
            (_max_tokens, _expand_width), dtype=torch.int32, device=device
        )
        _q_buf = torch.empty(
            (_max_tokens, _q_dim), dtype=dtype, device=device
        )
        _k_buf = torch.empty(
            (_max_tokens, _kv_dim), dtype=dtype, device=device
        )
        _v_buf = torch.empty(
            (_max_tokens, _kv_dim), dtype=dtype, device=device
        )
        _rope_scratch = torch.empty(
            (_max_tokens, _rope_dim), dtype=dtype, device=device
        )
        _pos_scratch = torch.empty(
            (3, _max_tokens), dtype=torch.int64, device=device
        )
        _contig_a = torch.empty(
            (_max_tokens, _contig_width), dtype=dtype, device=device
        )
        _contig_b = torch.empty(
            (_max_tokens, _contig_width), dtype=dtype, device=device
        )
        _contig_c = torch.empty(
            (_max_tokens, _contig_width), dtype=dtype, device=device
        )
        _head_norm_out = torch.empty(
            (_head_norm_rows, _head_norm_dim), dtype=dtype, device=device
        )
        _head_norm_out_b = torch.empty(
            (_head_norm_rows, _head_norm_dim), dtype=dtype, device=device
        )
        _head_norm_slot = 0
        _gate_buf = torch.empty(
            (_max_tokens, _q_dim), dtype=dtype, device=device
        )
        _gdn_q = torch.empty(
            (_max_tokens, _k_heads, _head_k_dim), dtype=dtype, device=device
        )
        _gdn_k = torch.empty(
            (_max_tokens, _k_heads, _head_k_dim), dtype=dtype, device=device
        )
        _gdn_v = torch.empty(
            (_max_tokens, _v_heads, _head_v_dim), dtype=dtype, device=device
        )
        _gdn_g = torch.empty(
            (_max_tokens, _v_heads), dtype=torch.float32, device=device
        )
        _gdn_beta = torch.empty(
            (_max_tokens, _v_heads), dtype=dtype, device=device
        )
        if _ple_state_len > 0:
            _ple_hist = torch.empty(
                (_max_tokens, _hc_width, _ple_state_len + 1),
                dtype=dtype,
                device=device,
            )
            _ple_prev = torch.empty(
                (_max_tokens, _hc_width, _ple_state_len),
                dtype=dtype,
                device=device,
            )
            _ple_out = torch.empty(
                (_max_tokens, _hc_width), dtype=dtype, device=device
            )
            _ple_idx_in = torch.empty(
                (_max_tokens,), dtype=torch.int64, device=device
            )
            _ple_idx_out = torch.empty(
                (_max_tokens,), dtype=torch.int64, device=device
            )
            _ple_valid = torch.empty(
                (_max_tokens,), dtype=torch.bool, device=device
            )
            _ple_keep = torch.empty(
                (_max_tokens,), dtype=torch.bool, device=device
            )
            _ple_mask = torch.empty(
                (_max_tokens,), dtype=dtype, device=device
            )
            _ple_i64 = torch.empty(
                (16, _max_tokens, 32), dtype=torch.int64, device=device
            )
            _ple_emb = torch.empty(
                (_max_tokens * 32, 256), dtype=dtype, device=device
            )
            _ple_bool = torch.empty(
                (4, _max_tokens, 32), dtype=torch.bool, device=device
            )
            _ple_arange = torch.arange(32, dtype=torch.int64, device=device)
            _ple_key = torch.empty(
                (_max_tokens, _hc_width), dtype=dtype, device=device
            )
            _ple_value = torch.empty(
                (_max_tokens, _hidden), dtype=dtype, device=device
            )
            _ple_gated = torch.empty(
                (_max_tokens, _hc_width), dtype=dtype, device=device
            )

        if _moe_topk > 0 and _moe_half_n > 0:
            _moe_out = torch.empty(
                (_max_tokens, _hidden), dtype=dtype, device=device
            )
            _moe_inter = torch.empty(
                (_max_tokens * _moe_topk, _moe_half_n), dtype=dtype, device=device
            )
        enable()


def gdn_out(
    rows: int, v_heads: int, head_v_dim: int, dtype: torch.dtype
) -> torch.Tensor:
    assert _gdn_out is not None
    if _gdn_out.dtype != dtype:
        return torch.empty(
            (rows, v_heads, head_v_dim), dtype=dtype, device=_gdn_out.device
        )
    return _gdn_out[:rows, :v_heads, :head_v_dim]


def hc_mix_out(rows: int, hidden: int, dtype: torch.dtype) -> torch.Tensor:
    assert _hc_mix_out is not None
    if _hc_mix_out.dtype != dtype:
        return torch.empty((rows, hidden), dtype=dtype, device=_hc_mix_out.device)
    return _hc_mix_out[:rows, :hidden]


def hc_combine_out(rows: int, hc_width: int, dtype: torch.dtype) -> torch.Tensor:
    assert _hc_combine_out is not None
    if _hc_combine_out.dtype != dtype:
        return torch.empty(
            (rows, hc_width), dtype=dtype, device=_hc_combine_out.device
        )
    return _hc_combine_out[:rows, :hc_width]


def rmsnorm_out(rows: int, width: int, dtype: torch.dtype) -> torch.Tensor:
    assert _rmsnorm_out is not None
    if _rmsnorm_out.dtype != dtype:
        return torch.empty((rows, width), dtype=dtype, device=_rmsnorm_out.device)
    return _rmsnorm_out[:rows, :width]


def expand_out(rows: int, width: int) -> torch.Tensor:
    assert _expand_out is not None
    return _expand_out[:rows, :width]


def as_contiguous(
    src: torch.Tensor,
    *,
    kind: str = "q",
) -> torch.Tensor:
    """Return a contiguous view of ``src``, copying into a pinned buffer if needed.

    When the graph workspace is active, tensors that feed capture-sensitive
    kernels (fused-QKV / gate split views, HC/RMSNorm flats) are always copied
    into a grow-once buffer so Triton/HIP pointer operands stay valid after a
    long eager prefill reclaims the caching-allocator slab. Already-contiguous
    tensors are returned as-is only when the workspace cannot cover them.
    """
    rows = int(src.shape[0])
    width = int(src.shape[-1])
    dtype_arg = src.dtype if src.dtype == _dtype else None
    if kind == "q" or kind == "gate":
        use_ws = fits(rows, q_dim=width, dtype=dtype_arg)
    elif kind in ("k", "v"):
        use_ws = fits(rows, kv_dim=width, dtype=dtype_arg)
    elif kind.startswith("contig"):
        use_ws = fits(rows, contig_width=width, dtype=dtype_arg)
    else:
        use_ws = False
    if not use_ws:
        return src if src.is_contiguous() else src.contiguous()
    if kind == "q":
        assert _q_buf is not None
        dst = _q_buf[:rows, :width]
    elif kind == "k":
        assert _k_buf is not None
        dst = _k_buf[:rows, :width]
    elif kind == "v":
        assert _v_buf is not None
        dst = _v_buf[:rows, :width]
    elif kind == "gate":
        assert _gate_buf is not None
        dst = _gate_buf[:rows, :width]
    elif kind == "contig_a":
        assert _contig_a is not None
        dst = _contig_a[:rows, :width]
    elif kind == "contig_b":
        assert _contig_b is not None
        dst = _contig_b[:rows, :width]
    elif kind == "contig_c":
        assert _contig_c is not None
        dst = _contig_c[:rows, :width]
    else:
        return src if src.is_contiguous() else src.contiguous()
    if dst.dtype != src.dtype:
        return src if src.is_contiguous() else src.contiguous()
    # Skip copy only when src already aliases the pinned buffer.
    if src.data_ptr() != dst.data_ptr() or src.stride() != dst.stride():
        dst.copy_(src)
    return dst


def rope_scratch(rows: int, head_size: int, dtype: torch.dtype) -> torch.Tensor:
    """Pinned RoPE partner scratch (indexer / attention dual-arg rotary)."""
    if fits(rows, rope_dim=head_size, dtype=dtype):
        assert _rope_scratch is not None
        buf = _rope_scratch[:rows, :head_size]
        buf.zero_()
        return buf
    return torch.zeros((rows, head_size), dtype=dtype, device=_device or "cuda")


def pos_scratch(rows: int) -> torch.Tensor:
    """Pinned ``[3, rows]`` int64 buffer for mRoPE position transpose."""
    if fits(rows):
        assert _pos_scratch is not None
        return _pos_scratch[:, :rows]
    assert _device is not None
    return torch.empty((3, rows), dtype=torch.int64, device=_device)


def head_norm_out(rows: int, dim: int, dtype: torch.dtype) -> torch.Tensor:
    """Pinned 2-D GemmaRMSNorm output for flattened ``[tokens*heads, dim]``.

    Alternates between two buffers so consecutive q_norm / k_norm (attention or
    indexer) do not alias the same storage into in-place RoPE.
    """
    global _head_norm_slot
    if not fits_head_norm(rows, dim, dtype):
        device = _device if _device is not None else torch.device("cuda")
        return torch.empty((rows, dim), dtype=dtype, device=device)
    assert _head_norm_out is not None and _head_norm_out_b is not None
    buf = _head_norm_out if (_head_norm_slot & 1) == 0 else _head_norm_out_b
    _head_norm_slot ^= 1
    return buf[:rows, :dim]


def fits_ple(
    rows: int,
    channels: int,
    state_len: int,
    dtype: torch.dtype | None = None,
) -> bool:
    """True when PLE decode temps fit the pinned ``[tokens, hc_width, state_len]`` slabs."""
    if not _enabled or _ple_hist is None or _ple_prev is None or _ple_out is None:
        return False
    if int(rows) > _max_tokens:
        return False
    # Exact channel / window match keeps ``[:rows]`` contiguous for ``out=``.
    if int(channels) != _hc_width or int(state_len) != _ple_state_len:
        return False
    if dtype is not None and dtype != _dtype:
        return False
    return True


def ple_decode_bufs(rows: int) -> tuple[torch.Tensor, ...]:
    """Pinned PLE decode scratch: hist, prev, out, idx_in, idx_out, valid, keep, mask."""
    assert _ple_hist is not None and _ple_prev is not None and _ple_out is not None
    assert _ple_idx_in is not None and _ple_idx_out is not None
    assert _ple_valid is not None and _ple_keep is not None and _ple_mask is not None
    return (
        _ple_hist[:rows],
        _ple_prev[:rows],
        _ple_out[:rows],
        _ple_idx_in[:rows],
        _ple_idx_out[:rows],
        _ple_valid[:rows],
        _ple_keep[:rows],
        _ple_mask[:rows],
    )


def pin_gdn_qkv(
    tensor: torch.Tensor, *, kind: str
) -> torch.Tensor:
    """Copy rearrange'd ``[1, L, H, D]`` Q/K/V into a graph-stable buffer."""
    if tensor is None:
        return tensor
    if tensor.ndim != 4 or tensor.shape[0] != 1:
        return tensor.contiguous()
    _seq, heads, dim = int(tensor.shape[1]), int(tensor.shape[2]), int(tensor.shape[3])
    if not _enabled or _gdn_q is None or _seq > _max_tokens:
        return tensor.contiguous()
    if kind in ("q", "k"):
        buf = _gdn_q if kind == "q" else _gdn_k
        if buf is None or heads > buf.shape[1] or dim > buf.shape[2] or buf.dtype != tensor.dtype:
            return tensor.contiguous()
        dst = buf[:_seq, :heads, :dim]
    elif kind == "v":
        if (
            _gdn_v is None
            or heads > _gdn_v.shape[1]
            or dim > _gdn_v.shape[2]
            or _gdn_v.dtype != tensor.dtype
        ):
            return tensor.contiguous()
        dst = _gdn_v[:_seq, :heads, :dim]
    else:
        return tensor.contiguous()
    dst.copy_(tensor.reshape(_seq, heads, dim))
    return dst.view(1, _seq, heads, dim)


def gdn_conv_qkv(
    num_tokens: int, k_dim: int, v_dim: int, dtype: torch.dtype
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
    """Pinned ``[T, dim, 1]`` outputs for ``causal_conv1d_update``."""
    if not _enabled or _gdn_q is None or _gdn_k is None or _gdn_v is None:
        return None
    if dtype != _dtype or int(num_tokens) > _max_tokens:
        return None
    q_flat = _k_heads * _head_k_dim
    v_flat = _v_heads * _head_v_dim
    if int(k_dim) > q_flat or int(v_dim) > v_flat:
        return None
    q = _gdn_q.reshape(_max_tokens, q_flat)[:num_tokens, :k_dim]
    k = _gdn_k.reshape(_max_tokens, q_flat)[:num_tokens, :k_dim]
    v = _gdn_v.reshape(_max_tokens, v_flat)[:num_tokens, :v_dim]
    return (
        q.view(num_tokens, k_dim, 1),
        k.view(num_tokens, k_dim, 1),
        v.view(num_tokens, v_dim, 1),
    )


def gdn_gate_out(
    batch: int, num_heads: int, dtype: torch.dtype
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Pinned fused-GDN gating ``g`` (fp32) and ``beta``."""
    if not _enabled or _gdn_g is None or _gdn_beta is None:
        return None
    if int(batch) > _max_tokens or int(num_heads) > _v_heads:
        return None
    if dtype != _dtype:
        return None
    g = _gdn_g[:batch, :num_heads].view(1, batch, num_heads)
    beta = _gdn_beta[:batch, :num_heads].view(1, batch, num_heads)
    return g, beta


def ple_i64(slot: int, rows: int, cols: int) -> torch.Tensor | None:
    if not _enabled or _ple_i64 is None:
        return None
    if slot < 0 or slot >= _ple_i64.shape[0]:
        return None
    if rows > _ple_i64.shape[1] or cols > _ple_i64.shape[2]:
        return None
    return _ple_i64[slot, :rows, :cols]


def ple_emb(rows: int, dim: int, dtype: torch.dtype) -> torch.Tensor | None:
    if not _enabled or _ple_emb is None:
        return None
    if dtype != _dtype or rows > _ple_emb.shape[0] or dim > _ple_emb.shape[1]:
        return None
    return _ple_emb[:rows, :dim]


def ple_bool(slot: int, rows: int, cols: int) -> torch.Tensor | None:
    if not _enabled or _ple_bool is None:
        return None
    if slot < 0 or slot >= _ple_bool.shape[0]:
        return None
    if rows > _ple_bool.shape[1] or cols > _ple_bool.shape[2]:
        return None
    return _ple_bool[slot, :rows, :cols]


def ple_arange(n: int) -> torch.Tensor | None:
    if not _enabled or _ple_arange is None or n > _ple_arange.numel():
        return None
    return _ple_arange[:n]


def ple_proj_bufs(
    rows: int, hc_width: int, hidden: int, dtype: torch.dtype
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
    if not _enabled or _ple_key is None or _ple_value is None or _ple_gated is None:
        return None
    if dtype != _dtype or rows > _max_tokens:
        return None
    if hc_width > _hc_width or hidden > _hidden:
        return None
    return (
        _ple_key[:rows, :hc_width],
        _ple_value[:rows, :hidden],
        _ple_gated[:rows, :hc_width],
    )

def fits_moe(rows: int, topk: int, half_n: int, dtype: torch.dtype | None = None) -> bool:
    if not _enabled or _moe_out is None or _moe_inter is None:
        return False
    if int(rows) > _max_tokens:
        return False
    if int(topk) > _moe_topk or int(half_n) > _moe_half_n:
        return False
    if dtype is not None and dtype != _dtype:
        return False
    return True


def moe_out(rows: int, hidden: int, dtype: torch.dtype) -> torch.Tensor:
    if not fits_moe(rows, 1, 1, dtype) or hidden > _hidden:
        device = _device if _device is not None else torch.device("cuda")
        return torch.empty((rows, hidden), dtype=dtype, device=device)
    assert _moe_out is not None
    return _moe_out[:rows, :hidden]


def moe_inter(rows: int, topk: int, half_n: int, dtype: torch.dtype) -> torch.Tensor:
    if not fits_moe(rows, topk, half_n, dtype):
        device = _device if _device is not None else torch.device("cuda")
        return torch.empty((rows * topk, half_n), dtype=dtype, device=device)
    assert _moe_inter is not None
    return _moe_inter[: rows * topk, :half_n]

