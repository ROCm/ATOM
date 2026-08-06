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


# Reserved key in the handles dict holding non-tensor Python metadata.
_META_KEY = "__meta__"

# Simple picklable value types carried by the sidecar (below).
_META_VALUE_TYPES = (str, bool, int, float)


def _tensor_meta_attrs(t: torch.Tensor) -> dict:
    """Custom Python attributes stamped onto a tensor object.

    process_weights_after_loading marks tensors with plain attributes rather
    than wrapping them — e.g. ``weight.is_shuffled = True`` (model_ops/moe.py,
    model_ops/utils.py).  A CUDA IPC handle carries storage + dtype/shape/stride
    and nothing else, and the consumer rebuilds a fresh tensor object, so these
    are silently dropped unless carried explicitly.  aiter reads ``is_shuffled``
    via getattr(..., False) as a KERNEL-SELECTION key for QuantType.per_1x32,
    so losing it makes the consumer run MXFP4 kernels against a layout the
    bytes do not have — wrong results, no error.
    """
    return {
        k: v
        for k, v in getattr(t, "__dict__", {}).items()
        if not k.startswith("_") and isinstance(v, _META_VALUE_TYPES)
    }


def _module_meta_attrs(mod: nn.Module) -> dict:
    """Non-tensor str/bool attributes stashed on a module by post-load hooks.

    Same class of bug as _tensor_meta_attrs, one level up: e.g. the swizzle
    layout labels ``w13_swizzle_layout`` / ``w2_swizzle_layout`` (moe.py:1109),
    which are plain strings initialised to None at construction and only filled
    by process_weights_after_loading.

    Restricted to str/bool deliberately.  Layout labels and format flags live in
    those types; ints/floats on a module are construction-time config, identical
    in both processes, so carrying them would only bloat the payload.
    """
    return {
        k: v
        for k, v in mod.__dict__.items()
        if not k.startswith("_")
        and k != "training"
        and isinstance(v, (str, bool))
    }


def export_model_weight_handles(model: nn.Module) -> dict:
    """Export all model parameter tensors as CUDA IPC handles.

    Also exports MLA weight-absorbed tensors (W_K/W_K_scale/W_V/W_V_scale)
    which are plain tensor attributes set by process_weights_after_loading(),
    not nn.Parameters, so named_parameters() misses them.

    Alongside the handles, a ``__meta__`` sidecar carries the non-tensor Python
    metadata that post-load hooks attach to tensors and modules; see
    _tensor_meta_attrs / _module_meta_attrs.  The consumer never runs
    process_weights_after_loading (it builds on meta and imports), so anything
    that hook produced must travel in the payload or it does not exist there.

    Must be called from the process that allocated the weights (prefill),
    after load_model() completes.  Returns a dict {key: meta_dict}.
    """
    handles = {}
    tensor_attrs: dict[str, dict] = {}
    module_attrs: dict[str, dict] = {}
    # Parameters. remove_duplicate=False so a Parameter registered under multiple
    # names (e.g. e_score_correction_bias, shared by gate + experts) is exported
    # under EVERY name — otherwise the consumer only materializes one of the
    # aliased registrations and the other stays on meta.
    for name, param in model.named_parameters(remove_duplicate=False):
        key = f"__param__{name}"
        handles[key] = _export_tensor(param.data)
        # Read attrs off the Parameter object, not param.data — process_weights
        # stamps the Parameter (e.g. `layer.w13_weight.is_shuffled = True`).
        attrs = _tensor_meta_attrs(param)
        if attrs:
            tensor_attrs[key] = attrs
    # Registered buffers (non-persistent included).
    for name, buf in model.named_buffers():
        if isinstance(buf, torch.Tensor) and buf.is_cuda and buf.numel() > 0:
            key = f"__buf__{name}"
            handles[key] = _export_tensor(buf)
            attrs = _tensor_meta_attrs(buf)
            if attrs:
                tensor_attrs[key] = attrs
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
                attrs = _tensor_meta_attrs(val)
                if attrs:
                    tensor_attrs[f"__attr__{key}"] = attrs
        mattrs = _module_meta_attrs(mod)
        if mattrs:
            module_attrs[mod_name] = mattrs

    handles[_META_KEY] = {"tensor_attrs": tensor_attrs, "module_attrs": module_attrs}
    logger.info(
        f"[WT-EXPORT] {len(handles) - 1} tensors, "
        f"{len(tensor_attrs)} with attrs, {len(module_attrs)} modules with attrs"
    )
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

    sidecar = handles.get(_META_KEY, {})
    tensor_attrs = sidecar.get("tensor_attrs", {})
    module_attrs = sidecar.get("module_attrs", {})

    def _restore_attrs(obj, key):
        """Re-stamp the producer's non-tensor attributes onto the rebuilt object."""
        for k, v in tensor_attrs.get(key, {}).items():
            setattr(obj, k, v)

    for key, meta in handles.items():
        if key == _META_KEY:
            continue
        t = _import_tensor(meta)
        if key.startswith("__param__"):
            # Rebuild the Parameter around the imported CUDA view (set_data fails
            # for meta->cuda). Create the slot if the consumer's meta model lacks
            # it (process_weights_after_loading may add params on the producer).
            name = key[len("__param__") :]
            parent, _, attr = name.rpartition(".")
            mod = modules.get(parent, model)
            rg = params[name].requires_grad if name in params else False
            p = nn.Parameter(t, requires_grad=rg)
            _restore_attrs(p, key)
            mod._parameters[attr] = p
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
                    _restore_attrs(t, key)
                    mod._buffers[attr] = t
        elif key.startswith("__attr__"):
            # Plain tensor attribute (e.g. MLA W_K/W_V from process_weights).
            name = key[len("__attr__") :]
            parent, _, attr = name.rpartition(".")
            mod = modules.get(parent, model)
            if mod is not None:
                _restore_attrs(t, key)
                setattr(mod, attr, t)

    # Module-level non-tensor metadata (e.g. swizzle layout labels). Fill only
    # what the consumer lacks — mirrors the buffer policy above. Never clobber a
    # value the consumer computed itself during construction.
    restored_mod_attrs = 0
    for mod_name, attrs in module_attrs.items():
        mod = modules.get(mod_name)
        if mod is None:
            continue
        for k, v in attrs.items():
            if getattr(mod, k, None) is None:
                setattr(mod, k, v)
                restored_mod_attrs += 1
    logger.info(
        f"[WT-IMPORT] restored {sum(len(a) for a in tensor_attrs.values())} tensor "
        f"attrs and {restored_mod_attrs} module attrs from the producer"
    )

    leftover = [n for n, p in model.named_parameters() if p.is_meta] + [
        n for n, b in model.named_buffers() if isinstance(b, torch.Tensor) and b.is_meta
    ]
    if leftover:
        logger.warning(
            f"[WT-IMPORT] {len(leftover)} tensors still on meta after import "
            f"(not exported by producer): {leftover[:12]}"
        )
