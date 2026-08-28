# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""
CUDA / ROCm IPC helpers for sharing GPU tensors across processes.

Uses tensor._share_cuda_() / UntypedStorage._new_shared_cuda() for the
low-level IPC handle path (hipIpcGetMemHandle / hipIpcOpenMemHandle on ROCm).
Both processes must be on the same physical GPU device.

Phase 1 (KV cache sharing):
  - export_kv_cache_handles — called by PrefillEngineCore after allocate_kv_cache()
  - import_kv_cache_handles — called by DecodeEngineCore at startup

Phase 2 (weight sharing):
  - export_model_weight_handles  — called by PrefillEngineCore after load_model()
  - import_model_weights         — called by DecodeEngineCore at startup (frees own copy)
"""

import logging

import torch
import torch.nn as nn

logger = logging.getLogger("atom")


def _export_tensor(t: torch.Tensor, retain: list | None = None) -> dict:
    """Serialize a CUDA tensor to a dict that can be pickled and sent cross-process.

    Uses tensor._share_cuda_() which calls hipIpcGetMemHandle on ROCm.
    Returns metadata needed to reconstruct the tensor on the other side.

    `retain` collects any tensor this call MATERIALISED. `.contiguous()` on a
    non-contiguous input returns a new tensor, and the handle is taken from that
    copy's storage — if it is left to go out of scope here, the consumer's IPC
    view points at freed memory. The producer must therefore hold a reference for
    as long as the consumer might read it, i.e. the life of the process.
    """
    src = t
    t = t.contiguous()
    if retain is not None and t is not src:
        retain.append(t)
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


def export_kv_cache_handles(named_values: dict) -> dict:
    """Export a name→value map of KV state as CUDA IPC handles.

    Name-keyed rather than hardcoded, because every attention backend produces
    a different set: MLA yields only `kv_cache`, MHA adds `kv_scale` and
    `_kv_layer_cache_store`, GDN adds `mamba_*`, and DeepSeek-V4 yields
    `v4_csa_idx_kv` (no `kv_cache` at all) plus a per-layer LIST in
    `v4_unified_kv`. Hardcoding two names silently broke every backend that
    does not happen to use them.

    Handles three value shapes: a CUDA tensor, a list/tuple of CUDA tensors
    (V4's per-layer pools), and a plain picklable scalar (e.g.
    `aligned_index_dim`). Anything else is skipped and reported by the caller.

    Must be called from the process that allocated the tensors (prefill).
    """
    out: dict = {}
    for name, value in named_values.items():
        if isinstance(value, torch.Tensor):
            if value.is_cuda and value.numel() > 0:
                out[name] = ("tensor", _export_tensor(value))
        elif (
            isinstance(value, (list, tuple))
            and value
            and all(
                isinstance(v, torch.Tensor) and v.is_cuda and v.numel() > 0
                for v in value
            )
        ):
            out[name] = (
                "tensor_list",
                [_export_tensor(v) for v in value],
            )
        elif value is None or isinstance(value, (str, bool, int, float)):
            out[name] = ("value", value)
    return out


def import_kv_cache_handles(meta: dict) -> dict:
    """Reconstruct the name→value map produced by export_kv_cache_handles.

    Must be called from the consumer process (decode). Returned tensors share
    GPU memory with the producer's allocation — no copy.
    """
    out: dict = {}
    for name, (kind, payload) in meta.items():
        if kind == "tensor":
            out[name] = _import_tensor(payload)
        elif kind == "tensor_list":
            out[name] = [_import_tensor(p) for p in payload]
        else:
            out[name] = payload
    return out


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


# Attributes produced by process_weights_after_loading that the str/bool sweep
# below would miss. Named explicitly rather than widening the type filter:
# blanket int copying would also carry `tp_size` / `tp_rank`, which legitimately
# DIFFER between a TP=8 producer and a TP=1 consumer and are read by
# `LinearBase.forward` (linear.py:988) — forcing the producer's values there
# would turn a padding bug into a parallelism bug.
_PRODUCER_ONLY_ATTRS = (
    # Set only when a8w8 preshuffle pads the output (linear.py:827) and read
    # alongside is_output_padded to slice the pad off (linear.py:985-987).
    "_output_size_before_padding",
)


def _module_meta_attrs(mod: nn.Module) -> dict:
    """Non-tensor attributes stashed on a module by post-load hooks.

    Same class of bug as _tensor_meta_attrs, one level up: e.g. the swizzle
    layout labels ``w13_swizzle_layout`` / ``w2_swizzle_layout`` (moe.py:1109),
    which are plain strings initialised to None at construction and only filled
    by process_weights_after_loading.

    str/bool covers layout labels and format flags. Anything else the hook
    produces has to be named in _PRODUCER_ONLY_ATTRS — see the note there for
    why the type filter is not simply widened.
    """
    out = {
        k: v
        for k, v in mod.__dict__.items()
        if not k.startswith("_")
        and k != "training"
        and isinstance(v, (str, bool))
    }
    for k in _PRODUCER_ONLY_ATTRS:
        if k in mod.__dict__:
            out[k] = mod.__dict__[k]
    return out


def _moe_placement(mod: nn.Module) -> tuple | None:
    """(ep_size, ep_rank, local_num_experts, tp_size, tp_rank) for a MoE module.

    One definition shared by producer and consumer on purpose: the two sides
    compare these tuples, so a field added to one and not the other would make
    the guard silently stop matching.
    """
    if not (hasattr(mod, "ep_size") and hasattr(mod, "ep_rank")):
        return None
    return (
        int(mod.ep_size),
        int(mod.ep_rank),
        int(getattr(mod, "local_num_experts", -1)),
        int(getattr(mod, "tp_size", -1)),
        int(getattr(mod, "tp_rank", -1)),
    )


def _expert_placement(model: nn.Module) -> dict:
    """Per-MoE-module sharding identity, for cross-checking across the alias.

    Under asymmetric rapidserve the two processes reach the same MoE sharding by
    different routes — prefill via plain TP (`tp_size=8, tp_rank=rank`), decode
    via the DP-attention flatten (`flatten_tp_across_dp`, moe.py:133-139, giving
    `tp_size = dp*tp, tp_rank = dp_rank`). Both land on the same split with the
    same rank-to-GPU binding, so rank k holds the same slice in both.

    WHICH split that is depends on `--enable-expert-parallel` (off by default,
    config.py:1328). With it OFF the experts are TENSOR-sharded on the
    intermediate dim (moe.py:2558) and ep_size is 1 on both sides; with it ON
    the weights are sharded by whole expert and tp_size is 1 (moe.py:2727). Both
    pairs are captured because checking only the expert fields would be vacuous
    in the default configuration — (1, 0, N) on both sides no matter what.

    That agreement is load-bearing and NOT implied by tensor shape: every rank
    holds the same-shaped slice either way, so a divergent mapping (a different
    flatten, a different DP-rank-to-GPU binding, EPLB rearranging one process's
    placement but not the other's) produces identically-shaped tensors holding
    the WRONG data. Aliasing those is silent numerical corruption, so the
    consumer verifies rather than assumes.
    """
    out: dict[str, tuple] = {}
    for name, mod in model.named_modules():
        placement = _moe_placement(mod)
        if placement is not None:
            out[name] = placement
    return out


def export_model_weight_handles(
    model: nn.Module, twins=None, retain: list | None = None
) -> dict:
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
        handles[key] = _export_tensor(param.data, retain)
        # Read attrs off the Parameter object, not param.data — process_weights
        # stamps the Parameter (e.g. `layer.w13_weight.is_shuffled = True`).
        attrs = _tensor_meta_attrs(param)
        if attrs:
            tensor_attrs[key] = attrs
    # Registered buffers (non-persistent included).
    for name, buf in model.named_buffers():
        if isinstance(buf, torch.Tensor) and buf.is_cuda and buf.numel() > 0:
            key = f"__buf__{name}"
            handles[key] = _export_tensor(buf, retain)
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
                handles[f"__attr__{key}"] = _export_tensor(val, retain)
                attrs = _tensor_meta_attrs(val)
                if attrs:
                    tensor_attrs[f"__attr__{key}"] = attrs
        mattrs = _module_meta_attrs(mod)
        if mattrs:
            module_attrs[mod_name] = mattrs

    # Asymmetric rapidserve: overlay the TP=1 twins. Same names, so the consumer
    # needs no special case — it just receives a full matrix where the producer's
    # own parameter is a shard. Done last so a twin always wins.
    n_overlaid = 0
    if twins is not None:
        for name, tensor in twins.overrides().items():
            key = f"__param__{name}"
            handles[key] = _export_tensor(tensor, retain)
            attrs = _tensor_meta_attrs(tensor)
            if attrs:
                tensor_attrs[key] = attrs
            n_overlaid += 1

    handles[_META_KEY] = {
        "tensor_attrs": tensor_attrs,
        "module_attrs": module_attrs,
        # Authoritative, unlike `module_attrs`: these come from the TP=1 twins,
        # so they must OVERRIDE whatever the consumer set at construction rather
        # than only filling gaps. See DecodeTwins.module_attr_overrides.
        "twin_module_attrs": (twins.module_attr_overrides() if twins else {}),
        "expert_placement": _expert_placement(model),
    }
    logger.info(
        f"[WT-EXPORT] {len(handles) - 1} tensors, "
        f"{len(tensor_attrs)} with attrs, {len(module_attrs)} modules with attrs, "
        f"{n_overlaid} overlaid from TP=1 twins"
    )
    return handles


class ExpertPlacementMismatch(RuntimeError):
    """Producer and consumer disagree about which experts each rank owns."""


def _assert_expert_placement_matches(model: nn.Module, producer: dict) -> None:
    """Fail loudly if the two processes shard experts differently.

    Silent-wrong-answer guard, not a shape check: mismatched placements are
    shape-compatible, so aliasing them would run the right kernels over the
    wrong experts and simply degrade output quality.
    """
    if not producer:
        return  # producer has no MoE modules, or predates the sidecar field
    bad = []
    for name, mod in model.named_modules():
        mine = _moe_placement(mod)
        if mine is None:
            continue
        theirs = producer.get(name)
        if theirs is None:
            continue
        if tuple(theirs) != mine:
            bad.append(f"{name}: producer={tuple(theirs)} consumer={mine}")
    if bad:
        raise ExpertPlacementMismatch(
            f"{len(bad)} MoE module(s) shard experts differently between the "
            f"prefill and decode processes, so their weights cannot be aliased "
            f"(the tensors are the same SHAPE but hold different experts). "
            f"Expected (ep_size, ep_rank, local_num_experts, tp_size, tp_rank) "
            f"to agree:\n  "
            + "\n  ".join(bad[:10])
        )


def import_model_weights(model: nn.Module, handles: dict) -> None:
    """Replace model parameters with views into another process's GPU allocation.

    Also restores MLA absorbed tensors exported by export_model_weight_handles.

    Must be called from the consumer process (decode) after receiving the
    handles dict from the producer (prefill).  After this call the decode
    model's parameters point into prefill's GPU memory — zero additional bytes
    are allocated.  The decode process's original weight tensors are freed when
    their reference counts drop to zero.

    Under asymmetric rapidserve the producer exports TP=1 twins for the tensors
    whose shape differs (decode_twins.py), so every entry aliases and the
    consumer allocates nothing.
    """
    _import_model_weights_impl(model, handles)


def _import_model_weights_impl(model: nn.Module, handles: dict) -> None:
    modules = dict(model.named_modules())
    # remove_duplicate=False to match the export and to materialize every
    # registration of a shared Parameter (see export note).
    params = dict(model.named_parameters(remove_duplicate=False))
    buffers = dict(model.named_buffers())

    sidecar = handles.get(_META_KEY, {})
    tensor_attrs = sidecar.get("tensor_attrs", {})
    module_attrs = sidecar.get("module_attrs", {})

    # MoE aliasing rests on both processes deriving the SAME expert placement
    # (see _expert_placement). Shape equality cannot detect a divergence — every
    # rank holds global//ep_size experts either way — so check it explicitly
    # before trusting a single alias. No-op when the producer has no MoE.
    _assert_expert_placement_matches(model, sidecar.get("expert_placement", {}))

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
                # The producer's post-load hook may have turned a registered
                # Parameter into a plain attribute — moe.py:1103-1108 does
                # exactly that for w13/w2_weight_scale. The consumer built on
                # meta and never ran that hook, so it still holds a meta
                # Parameter under this name. Drop that stale slot: leaving it
                # makes named_parameters() report the tensor as un-materialized,
                # so the consumer re-loads it from the checkpoint, which drags
                # its whole module into post-load processing — and that
                # re-shuffles the module's ALIASED weights, allocating a second
                # copy and writing through the alias into the producer's memory.
                # Measured cost of getting this wrong on V4-Pro: +90GB and
                # corrupted prefill weights.
                mod._parameters.pop(attr, None)
                _restore_attrs(t, key)
                setattr(mod, attr, t)

    # Module-level non-tensor metadata (e.g. swizzle layout labels). Fill only
    # what the consumer lacks — mirrors the buffer policy above. Never clobber a
    # value the consumer computed itself during construction.
    # Authoritative, NOT fill-the-gaps. The consumer never runs
    # process_weights_after_loading for any module, so it has no computed value
    # to protect — only construction-time defaults. The old `is None` guard
    # therefore restored nothing at all for the common case: is_output_padded
    # starts as False (linear.py:508), and `False is not None`, so 1343 modules'
    # worth of post-load state was sent and silently dropped every run.
    restored_mod_attrs = 0
    for mod_name, attrs in module_attrs.items():
        mod = modules.get(mod_name)
        if mod is None:
            continue
        for k, v in attrs.items():
            setattr(mod, k, v)
            restored_mod_attrs += 1
    # Twin attributes are authoritative: the consumer runs TP=1 and never ran
    # the post-load hook, so its own values (e.g. is_output_padded=False from
    # __init__) are stale, not a computed result to preserve.
    twin_attrs = sidecar.get("twin_module_attrs", {})
    forced = 0
    for mod_name, attrs in twin_attrs.items():
        mod = modules.get(mod_name)
        if mod is None:
            continue
        for k, v in attrs.items():
            setattr(mod, k, v)
            forced += 1
    logger.info(
        f"[WT-IMPORT] restored {sum(len(a) for a in tensor_attrs.values())} tensor "
        f"attrs and {restored_mod_attrs} module attrs from the producer; "
        f"forced {forced} attr(s) from {len(twin_attrs)} TP=1 twin(s)"
    )
    # A post-load hook may DELETE a parameter on the producer — V4's attention
    # dequants wo_a to BF16 and then `delattr(self.wo_a, "weight_scale")`
    # (deepseek_v4.py:2453). The consumer built on meta and never ran that hook,
    # so it still declares the parameter and nothing arrives to fill it. Drop the
    # stale slot: leaving a meta tensor in named_parameters() trips every
    # subsequent sweep over the model.
    dropped = 0
    for mod_name, mod in list(model.named_modules()):
        for attr in [n for n, p in mod.named_parameters(recurse=False) if p.is_meta]:
            full = f"{mod_name}.{attr}" if mod_name else attr
            if f"__param__{full}" not in handles:
                del mod._parameters[attr]
                dropped += 1
    if dropped:
        logger.info(
            f"[WT-IMPORT] dropped {dropped} meta parameter(s) the producer "
            f"deleted during post-load processing"
        )

    leftover = [n for n, p in model.named_parameters() if p.is_meta] + [
        n for n, b in model.named_buffers() if isinstance(b, torch.Tensor) and b.is_meta
    ]
    if leftover:
        logger.warning(
            f"[WT-IMPORT] {len(leftover)} tensors still on meta after import "
            f"(not exported by producer): {leftover[:12]}"
        )
