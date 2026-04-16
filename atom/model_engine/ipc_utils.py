# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""
CUDA / ROCm IPC helpers for sharing GPU tensors across processes.

Uses tensor._share_cuda_() / UntypedStorage._new_shared_cuda() for the
low-level IPC handle path (hipIpcGetMemHandle / hipIpcOpenMemHandle on ROCm).
Both processes must be on the same physical GPU device.

Phase 1 (KV cache sharing):
  - export_kv_cache_handle  — called by PrefillEngineCore after allocate_kv_cache()
  - import_kv_cache         — called by DecodeEngineCore at startup

Phase 2 (weight sharing):
  - export_model_weight_handles  — called by PrefillEngineCore after load_model()
  - import_model_weights         — called by DecodeEngineCore at startup (frees own copy)
"""

import torch
import torch.nn as nn


def _export_tensor(t: torch.Tensor) -> dict:
    """Serialize a CUDA tensor to a dict that can be pickled and sent cross-process.

    Uses tensor._share_cuda_() which calls hipIpcGetMemHandle on ROCm.
    Returns metadata needed to reconstruct the tensor on the other side.
    """
    t = t.contiguous()
    share_cuda_args = t.untyped_storage()._share_cuda_()
    return {
        "share_cuda_args": share_cuda_args,
        "dtype": t.dtype,
        "shape": t.shape,
        "stride": t.stride(),
        "storage_offset": t.storage_offset(),
    }


def _import_tensor(meta: dict) -> torch.Tensor:
    """Reconstruct a CUDA tensor from the dict produced by _export_tensor.

    Calls UntypedStorage._new_shared_cuda() which calls hipIpcOpenMemHandle.
    """
    storage = torch.UntypedStorage._new_shared_cuda(*meta["share_cuda_args"])
    t = torch.empty(0, dtype=meta["dtype"], device="cuda")
    t.set_(storage, meta["storage_offset"], meta["shape"], meta["stride"])
    return t


# ---------------------------------------------------------------------------
# KV cache (Phase 1)
# ---------------------------------------------------------------------------


def export_kv_cache_handle(
    kv_cache: torch.Tensor, kv_scale: torch.Tensor | None = None
) -> dict:
    """Export kv_cache (and optionally kv_scale for fp8) as CUDA IPC handles.

    Must be called from the process that allocated the tensor (prefill).
    Returns a dict that can be pickled and sent over ZMQ to the decode process.
    """
    result = {"kv_cache": _export_tensor(kv_cache)}
    if kv_scale is not None:
        result["kv_scale"] = _export_tensor(kv_scale)
    return result


def import_kv_cache(meta: dict) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Reconstruct kv_cache (and kv_scale if present) from CUDA IPC handles.

    Must be called from the consumer process (decode).
    Returns (kv_cache, kv_scale) — kv_scale is None when not fp8.
    The returned tensors share GPU memory with prefill's allocation — no copy.
    """
    kv_cache = _import_tensor(meta["kv_cache"])
    kv_scale = _import_tensor(meta["kv_scale"]) if "kv_scale" in meta else None
    return kv_cache, kv_scale


# ---------------------------------------------------------------------------
# Model weights (Phase 2)
# ---------------------------------------------------------------------------


def export_model_weight_handles(model: nn.Module) -> dict:
    """Export all model parameter tensors as CUDA IPC handles.

    Must be called from the process that allocated the weights (prefill),
    after load_model() completes.  Returns a dict {param_name: meta_dict}
    where each meta_dict can be passed to _import_tensor().
    """
    handles = {}
    for name, param in model.named_parameters():
        handles[name] = _export_tensor(param.data)
    return handles


def import_model_weights(model: nn.Module, handles: dict) -> None:
    """Replace model parameters with views into another process's GPU allocation.

    Must be called from the consumer process (decode) after receiving the
    handles dict from the producer (prefill).  After this call the decode
    model's parameters point into prefill's GPU memory — zero additional bytes
    are allocated.  The decode process's original weight tensors are freed when
    their reference counts drop to zero.
    """
    params = dict(model.named_parameters())
    for name, meta in handles.items():
        if name not in params:
            continue
        shared_tensor = _import_tensor(meta)
        params[name].data = shared_tensor
