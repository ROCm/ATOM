# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Persistent HCA fp8 decode (``ATOM_V4_HCA_PERSIST``, on by default).

HCA (compress_ratio 128) decode calls run aiter's persistent V4-NM kernel
(``aiter.mla.mla_decode_fwd_v4_nm_ps``): one launch that plans the KV split
from ``kv_indptr`` on the GPU, runs the attention and merges the partials, so
no host-side split plan and no stage-2 merge. Same tensors and layouts as the
decode ASM path (``_sparse_attn_v4_paged_decode_asm``); rows with K = 0 (CUDA
graph padding) are left unwritten, as the ASM does. Limits: 128 local heads,
``N <= 32768`` rows, gfx950, row-dense KV pools. Everything else, and an aiter
without the kernel, stays on the ASM + split plan.

Workspace: one per device, allocated by :func:`prepare` outside graph capture.
Every HCA attention layer asks for it at construction
(:func:`prepare_if_usable`, so model load allocates it in native ATOM and in
the vLLM / SGLang plugins alike, before KV sizing and warmup), and the native
V4 metadata builder asks again at init (idempotent). A call that finds no
workspace allocates it when eager; under CUDA-graph capture it stays on the
ASM path instead (:func:`workspace_ready`). It
holds the split partials and the merge counters, which return to zero at the
end of every call, so calls sharing it must be ordered on the GPU. In ATOM
every HCA decode call runs on the forward's compute stream (layers are
sequential; TBO micro-batches share one compute stream; eager forwards and
graph replays do not overlap), so one workspace per device is enough.
"""

from __future__ import annotations

import logging

import torch

from atom.utils import envs

logger = logging.getLogger("atom")

try:
    from aiter.mla import get_mla_v4_nm_ps_workspace, mla_decode_fwd_v4_nm_ps
except ImportError:  # aiter without the persistent V4-NM kernel
    get_mla_v4_nm_ps_workspace = mla_decode_fwd_v4_nm_ps = None

HEADS = 128
MAX_ROWS = 32768  # byte-offset wrap inherited from the ASM
HCA_RATIO = 128
PARTITIONS = 128
_DIM = 512
_ROPE = 64

# Host-side call counts: eager calls count per call, a CUDA graph once at
# capture (replays do not run Python).
stats = {"persist": 0, "persist_captured": 0}
_logged: set = set()
_workspaces: dict[int, object] = {}


def _log_once(key, msg: str, *args) -> None:
    if key not in _logged:
        _logged.add(key)
        logger.info(msg, *args)


def available() -> bool:
    if mla_decode_fwd_v4_nm_ps is None:
        _log_once(
            "no_api",
            "V4 HCA persistent decode unavailable: aiter lacks "
            "mla_decode_fwd_v4_nm_ps; HCA decode stays on the ASM path",
        )
        return False
    return True


def _dev_index(device) -> int:
    device = torch.device(device)
    return device.index if device.index is not None else torch.cuda.current_device()


def prepare(device) -> None:
    """Allocate this device's workspace (idempotent). Must run outside CUDA
    graph capture; a no-op when aiter lacks the kernel."""
    idx = _dev_index(device)
    if idx in _workspaces or not available():
        return
    _workspaces[idx] = get_mla_v4_nm_ps_workspace(
        torch.device("cuda", idx), num_partitions=PARTITIONS
    )
    logger.info(
        "V4 HCA persistent decode: device %d ready (P=%d, min rows %d)",
        idx,
        PARTITIONS,
        envs.ATOM_V4_HCA_PERSIST_MIN_ROWS,
    )


def unusable_reason(*, kv_fp8: bool, heads: int, gfx: str) -> str | None:
    """Why the persistent kernel cannot serve this model/rank, or None."""
    if not kv_fp8:
        return "kv cache is not fp8"
    if heads != HEADS:
        return f"{heads} local heads (needs {HEADS})"
    if gfx != "gfx950":
        return f"arch {gfx} (needs gfx950)"
    return None


def prepare_if_usable(*, kv_fp8: bool, heads: int, gfx: str, device=None) -> bool:
    """Allocate the workspace when this rank can use the kernel (switch on,
    fp8 KV, 128 local heads, gfx950). Called at model construction, outside
    graph capture; ``device`` defaults to the current CUDA device."""
    if not envs.ATOM_V4_HCA_PERSIST:
        return False
    why = unusable_reason(kv_fp8=kv_fp8, heads=heads, gfx=gfx)
    if why is not None:
        _log_once(("unused", why), "V4 HCA persistent decode not used: %s", why)
        return False
    device = torch.device("cuda") if device is None else device
    prepare(device)
    return _dev_index(device) in _workspaces


def workspace_ready(device) -> bool:
    """True when this device has a workspace, allocating it if the stream is
    not capturing. Under capture with no workspace (nothing prepared it before
    capture) the call stays on the ASM path: aiter refuses to allocate there."""
    idx = _dev_index(device)
    if idx in _workspaces:
        return True
    if torch.cuda.is_current_stream_capturing():
        _log_once(
            ("no_ws_capture", idx),
            "V4 HCA persistent decode: no workspace on device %d at CUDA-graph "
            "capture (prepare() was not called before capture); this graph "
            "keeps HCA decode on the ASM path",
            idx,
        )
        return False
    prepare(device)
    return idx in _workspaces


def wanted(*, compress_ratio: int | None, heads: int, rows: int, gfx: str) -> bool:
    """Shape/config gate; the layout gate is :func:`layout_ok`."""
    return (
        envs.ATOM_V4_HCA_PERSIST
        and compress_ratio == HCA_RATIO
        and heads == HEADS
        and gfx == "gfx950"
        and envs.ATOM_V4_HCA_PERSIST_MIN_ROWS <= rows <= MAX_ROWS
        and available()
    )


def layout_ok(unified_kv: torch.Tensor, unified_kv_rope: torch.Tensor) -> bool:
    """The kernel addresses pool row r at base + r * row_bytes."""
    return (
        unified_kv.dim() == 2
        and unified_kv.shape[-1] == _DIM
        and unified_kv.element_size() == 1
        and unified_kv.stride() == (_DIM, 1)
        and unified_kv_rope.dim() == 2
        and unified_kv_rope.shape[-1] == _ROPE
        and unified_kv_rope.dtype == torch.bfloat16
        and unified_kv_rope.stride() == (_ROPE, 1)
    )


def hca_persist_decode(
    unified_kv: torch.Tensor,
    kv_indices: torch.Tensor,
    kv_indptr: torch.Tensor,
    attn_sink: torch.Tensor,
    unified_kv_rope: torch.Tensor,
    q_packed_in: torch.Tensor,
    q_rope_in: torch.Tensor,
) -> torch.Tensor:
    """HCA decode of the ``N = q_packed_in.shape[0]`` rows the ASM path would
    run (``kv_indptr[:N + 1]``; an eager forward may have N < T_pad)."""
    n, h, _ = q_packed_in.shape
    if h != HEADS or n > MAX_ROWS or kv_indptr.numel() < n + 1:
        raise ValueError(
            f"hca_persist: needs {HEADS} heads, <= {MAX_ROWS} rows and "
            f">= N+1 kv_indptr entries; got heads={h}, rows={n}, "
            f"kv_indptr={kv_indptr.numel()}"
        )
    idx = _dev_index(q_packed_in.device)
    if not workspace_ready(q_packed_in.device):
        raise RuntimeError(
            "hca_persist: no workspace on this device and the stream is "
            "capturing; call prepare() before CUDA-graph capture"
        )
    out = torch.empty((n, HEADS, _DIM), dtype=torch.bfloat16, device=q_packed_in.device)
    if n == 0:
        return out
    mla_decode_fwd_v4_nm_ps(
        q_packed_in.contiguous(),
        q_rope_in.contiguous(),
        unified_kv,
        unified_kv_rope,
        kv_indptr[: n + 1],
        kv_indices.contiguous(),
        attn_sink.contiguous(),
        _workspaces[idx],
        out=out,
    )
    capturing = torch.cuda.is_current_stream_capturing()
    stats["persist"] += 1
    stats["persist_captured"] += int(capturing)
    _log_once(
        ("hit", capturing),
        "V4 HCA persistent decode: first %s call on device %d (rows=%d)",
        "CUDA-graph captured" if capturing else "eager",
        idx,
        n,
    )
    return out
