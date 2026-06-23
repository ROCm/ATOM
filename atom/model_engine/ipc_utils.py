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

import logging

import torch
import torch.nn as nn

logger = logging.getLogger("atom")


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

    Also exports MLA weight-absorbed tensors (W_K/W_K_scale/W_V/W_V_scale)
    which are plain tensor attributes set by process_weights_after_loading(),
    not nn.Parameters, so named_parameters() misses them.

    Must be called from the process that allocated the weights (prefill),
    after load_model() completes.  Returns a dict {key: meta_dict}.
    """
    handles = {}
    # Parameters. remove_duplicate=False so a Parameter registered under multiple
    # names (e.g. e_score_correction_bias, shared by gate + experts) is exported
    # under EVERY name — otherwise the consumer only materializes one of the
    # aliased registrations and the other stays on meta.
    for name, param in model.named_parameters(remove_duplicate=False):
        handles[f"__param__{name}"] = _export_tensor(param.data)
    # Registered buffers (non-persistent included).
    for name, buf in model.named_buffers():
        if isinstance(buf, torch.Tensor) and buf.is_cuda and buf.numel() > 0:
            handles[f"__buf__{name}"] = _export_tensor(buf)
    # Plain tensor attributes set by process_weights_after_loading() — e.g. the
    # MLA absorbed W_K/W_V — which are neither Parameters nor registered buffers.
    for mod_name, mod in model.named_modules():
        for attr, val in list(mod.__dict__.items()):
            if (
                isinstance(val, torch.Tensor)
                and not isinstance(val, nn.Parameter)
                and val.is_cuda
                and val.numel() > 0
            ):
                key = f"{mod_name}.{attr}" if mod_name else attr
                handles[f"__attr__{key}"] = _export_tensor(val)

    return handles


def import_model_weights(model: nn.Module, handles: dict) -> None:
    """Replace model parameters with views into another process's GPU allocation.

    Also restores MLA absorbed tensors exported by export_model_weight_handles.

    Must be called from the consumer process (decode) after receiving the
    handles dict from the producer (prefill).  After this call the decode
    model's parameters point into prefill's GPU memory — zero additional bytes
    are allocated.  The decode process's original weight tensors are freed when
    their reference counts drop to zero.
    """
    modules = dict(model.named_modules())
    # remove_duplicate=False to match the export and to materialize every
    # registration of a shared Parameter (see export note).
    params = dict(model.named_parameters(remove_duplicate=False))
    buffers = dict(model.named_buffers())

    for key, meta in handles.items():
        t = _import_tensor(meta)
        if key.startswith("__param__"):
            # Rebuild the Parameter around the imported CUDA view (set_data fails
            # for meta->cuda). Create the slot if the consumer's meta model lacks
            # it (process_weights_after_loading may add params on the producer).
            name = key[len("__param__") :]
            parent, _, attr = name.rpartition(".")
            mod = modules.get(parent, model)
            rg = params[name].requires_grad if name in params else False
            mod._parameters[attr] = nn.Parameter(t, requires_grad=rg)
        elif key.startswith("__buf__"):
            # Keep decode's locally-built real buffers (e.g. RoPE caches built
            # during construction); only fill buffers it is missing or left on
            # meta (those created inside process_weights_after_loading).
            name = key[len("__buf__") :]
            existing = buffers.get(name)
            if existing is None or existing.is_meta:
                parent, _, attr = name.rpartition(".")
                mod = modules.get(parent, model)
                if mod is not None:
                    mod._buffers[attr] = t
        elif key.startswith("__attr__"):
            # Plain tensor attribute (e.g. MLA W_K/W_V from process_weights).
            name = key[len("__attr__") :]
            parent, _, attr = name.rpartition(".")
            mod = modules.get(parent, model)
            if mod is not None:
                setattr(mod, attr, t)

    leftover = [n for n, p in model.named_parameters() if p.is_meta] + [
        n
        for n, b in model.named_buffers()
        if isinstance(b, torch.Tensor) and b.is_meta
    ]
    if leftover:
        logger.warning(
            f"[WT-IMPORT] {len(leftover)} tensors still on meta after import "
            f"(not exported by producer): {leftover[:12]}"
        )
