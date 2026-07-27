# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""Checkpoint -> parameter loading, with no GPU or AITER dependency.

`load_model` in `loader.py` is a thin wrapper that binds the host-specific
callables below and then runs post-processing; everything that decides *which*
checkpoint tensor lands in *which* parameter lives here.  The split exists so
this logic is unit-testable on a plain CPU runner — the unit-test gate has no
AITER build, and `loader.py` imports AITER at module level.
"""

import concurrent.futures
import logging
import re
import threading
from collections.abc import Callable, Iterable

import torch
from torch import nn
from transformers import AutoConfig

from atom.model_loader.weight_names import (
    WeightsMapper,
    extract_expert_target_and_id,
    have_shared_expert,
    shared_expert_prefixes,
)
from atom.utils import envs

logger = logging.getLogger("atom")


def load_weights_into_model(
    model: nn.Module,
    model_name_or_path: str,
    hf_config: AutoConfig,
    load_dummy: str | None = None,
    spec_decode: bool = False,
    prefix: str = "",
    weights_mapper: WeightsMapper | None = None,
    load_fused_expert_weights_fn=None,
    *,
    default_weight_loader: Callable,
    fuse_shared_expert: Callable[[str, str], bool],
    is_rank0: Callable[[], bool],
    weights_iterator: Callable[[str, bool], Iterable[tuple[str, torch.Tensor]]],
) -> set[str]:
    """Copy every checkpoint tensor into the model parameter it belongs to.

    The four keyword-only callables are the host environment this module
    refuses to import itself:

    - ``default_weight_loader``  fallback copy for params without their own
    - ``fuse_shared_expert``     ``(shared_prefix, routed_prefix) -> fuse?``
    - ``is_rank0``               suppress duplicate diagnostics off rank 0
    - ``weights_iterator``       ``(path, disable_mmap) -> (name, tensor)``
    """

    def should_fuse_shared_expert_weight(name: str, matching_name: str) -> bool:
        return fuse_shared_expert(*shared_expert_prefixes(name, matching_name))

    # need to record the loaded weight name for vllm load check
    # it is only used in plugin mode for vllm
    loaded_weights_record: set[str] = set()

    # Auto-detect weight mapper from model if not provided explicitly
    if weights_mapper is None:
        model_mapper = getattr(model, "hf_to_atom_mapper", None)
        if isinstance(model_mapper, dict):
            weights_mapper = WeightsMapper(orig_to_new_prefix=model_mapper)
        elif isinstance(model_mapper, WeightsMapper):
            weights_mapper = model_mapper

    packed_modules_mapping = getattr(model, "packed_modules_mapping", {})
    weights_mapping = getattr(model, "weights_mapping", {})
    skip_weight_prefixes = getattr(model, "skip_weight_prefixes", [])
    mtp_remap = getattr(model, "remap_mtp_weight_name", None)
    # Models can also expose a `weights_mapper` (WeightsMapper instance) for
    # precise prefix/suffix-anchored renames that the dumb substring-substitution
    # `weights_mapping` dict cannot express safely. If both are set they are
    # composed: weights_mapper applies first, then the legacy substring map.
    if weights_mapper is None:
        weights_mapper = getattr(model, "weights_mapper", None)
    params_dict = dict(model.named_parameters())
    # Pre-index expert_mapping by weight_name_part for O(1) lookup.
    # Original code does O(N) scan of expert_mapping (768 entries) per tensor,
    # causing ~19s of CPU time for 90k expert tensors. This reduces it to O(1).
    has_expert_mapping = hasattr(model, "get_expert_mapping")
    expert_index = {}  # {weight_name_part: (param_name_part, expert_id, shard_id)}
    expert_weight_prefixes = []  # sorted longest-first for prefix matching
    if has_expert_mapping:
        for (
            param_name_part,
            weight_name_part,
            expert_id,
            shard_id,
        ) in model.get_expert_mapping():
            expert_index[weight_name_part] = (param_name_part, expert_id, shard_id)
        # Sort by length descending so longer (more specific) prefixes match first
        expert_weight_prefixes = sorted(expert_index.keys(), key=len, reverse=True)

    # Get fused expert mapping from model if it provides one
    is_fused_expert = False
    fused_expert_params_mapping = []
    detect_fused_expert_fn = getattr(model, "detect_fused_expert_format", None)
    get_fused_expert_mapping_fn = getattr(model, "get_fused_expert_mapping", None)

    # Track ckpt names that were silently dropped at `get_parameter`
    # AttributeError sites — these indicate weights_mapping bugs where the
    # rewritten name doesn't correspond to any model param. (orig, mapped) pairs.
    dropped_ckpt_keys: list[tuple[str, str]] = []

    staging_map: dict = {}  # id(param) -> entry, one per in-flight fused param
    fallback_pids: set = set()  # params that opted out of batching
    staging_lock = threading.Lock()

    moe_module_cache: dict = {}
    param_batchable: dict = {}

    def _lookup_moe_module(full_param_name: str):
        module_path = full_param_name.rsplit(".", 1)[0]
        if module_path not in moe_module_cache:
            moe_module_cache[module_path] = (
                model.get_submodule(module_path) if "." in full_param_name else None
            )
        return moe_module_cache[module_path]

    def _param_is_batchable(param, full_param_name: str) -> bool:
        pid = id(param)
        if pid not in param_batchable:
            moe = _lookup_moe_module(full_param_name)
            expected = (
                moe.expected_batched_arrivals(param)
                if moe is not None and hasattr(moe, "stage_expert_weight")
                else None
            )
            param_batchable[pid] = bool(expected)
        return param_batchable[pid]

    def _do_flush(param, staging):
        if staging.dtype != param.data.dtype:
            param.data.view(torch.uint8).copy_(staging)
        else:
            param.data.copy_(staging)

    def _make_staging(param):
        pin = torch.cuda.is_available()

        def _alloc(pinned):
            try:
                t = torch.empty(
                    param.data.shape,
                    dtype=param.data.dtype,
                    device="cpu",
                    pin_memory=pinned,
                )
                t.zero_()
            except NotImplementedError:
                t = torch.empty(
                    param.data.shape,
                    dtype=torch.uint8,
                    device="cpu",
                    pin_memory=pinned,
                )
                t.zero_()
            return t

        try:
            return _alloc(pin)
        except RuntimeError as e:
            logger.warning("Pinned staging alloc failed (%s); using unpinned.", e)
            return _alloc(False)

    def _fallback(param, full_param_name, shard_id, global_expert_id, loaded_weight):
        param.weight_loader(
            param, loaded_weight, full_param_name, shard_id, global_expert_id
        )

    def _stage_task(param, full_param_name, shard_id, global_expert_id, loaded_weight):
        pid = id(param)
        with staging_lock:
            opted_out = pid in fallback_pids
            entry = None if opted_out else staging_map.get(pid)
        if opted_out:
            _fallback(param, full_param_name, shard_id, global_expert_id, loaded_weight)
            return

        # Map to this rank's local expert id BEFORE touching staging_map. Under
        # expert parallelism every rank iterates all global experts, but a
        # non-local expert contributes nothing to this rank's staging. If such a
        # straggler ran after the param already reached `expected` and flushed
        # (which deletes its staging entry), creating an entry here would leave a
        # fresh, never-filled entry that is miscounted as "under-filled" at the
        # end of loading. Return early so non-local shards never create entries.
        moe = _lookup_moe_module(full_param_name)
        local_eid = moe._map_global_expert_id_to_local_expert_id(global_expert_id)
        if local_eid == -1:
            return

        if entry is None:
            new_entry = {
                "staging": _make_staging(param),
                "arrived": 0,
                "expected": moe.expected_batched_arrivals(param),
                "moe": moe,
                "param": param,
                "lock": threading.Lock(),
            }
            with staging_lock:
                opted_out = pid in fallback_pids
                if not opted_out:
                    entry = staging_map.get(pid)
                    if entry is None:
                        entry = staging_map[pid] = new_entry
            if opted_out:
                _fallback(
                    param, full_param_name, shard_id, global_expert_id, loaded_weight
                )
                return

        ok = moe.stage_expert_weight(
            param=param,
            staging=entry["staging"],
            loaded_weight=loaded_weight,
            local_expert_id=local_eid,
            shard_id=shard_id,
            weight_name=full_param_name,
        )
        if not ok:
            with staging_lock:
                fallback_pids.add(pid)
                staging_map.pop(pid, None)
            _fallback(param, full_param_name, shard_id, global_expert_id, loaded_weight)
            return

        with entry["lock"]:
            entry["arrived"] += 1
            flush_now = entry["arrived"] >= entry["expected"]
        if flush_now:
            _do_flush(param, entry["staging"])
            with staging_lock:
                if staging_map.get(pid) is entry:
                    del staging_map[pid]

    num_threads = envs.ATOM_LOADER_NUM_THREADS
    if num_threads > 1:
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=num_threads)
    else:
        executor = None
    futures = []

    def _submit(fn, *args):
        if executor is not None:
            futures.append(executor.submit(fn, *args))
        else:
            fn(*args)

    try:
        disable_mmap = envs.ATOM_DISABLE_MMAP
        for name, weight_tensor in weights_iterator(model_name_or_path, disable_mmap):
            _orig_ckpt_name = name  # preserve for ckpt-side coverage report
            if weights_mapper is not None:
                mapped_name = weights_mapper._map_name(name)
                if mapped_name is None:
                    continue
                name = mapped_name
            if load_dummy:
                continue
            # Draft models may remap ckpt-side `mtp.*` entries into params
            # whose names do not themselves contain `mtp` (e.g. Qwen3.5 MTP
            # rewrites `mtp.*` -> `model.*`). Gate only on `spec_decode`,
            # otherwise we can drop the entire drafter checkpoint before the
            # model-specific remap logic has a chance to run.
            if "mtp" in name and not spec_decode:
                continue
            if name.endswith("kv_scale") or "inv_freq" in name:
                continue
            # Skip weights matching model-defined prefixes (e.g. vision encoder
            # weights in multimodal checkpoints that are not needed for text-only
            # inference).
            if skip_weight_prefixes and any(
                name.startswith(p) for p in skip_weight_prefixes
            ):
                continue
            if spec_decode and mtp_remap is not None:
                remapped = mtp_remap(name)
                if remapped is None:
                    continue
                name = remapped
            for mapping_part in weights_mapping:
                if mapping_part in name:
                    name = name.replace(mapping_part, weights_mapping[mapping_part])
            if "weight_scale_inv" in name:
                name = name.replace("weight_scale_inv", "weight_scale")

            layerId_ = re.search(r"model\.layers\.(\d+)\.", name)
            layerId = int(layerId_.group(1)) if layerId_ else 0
            if (
                hf_config.num_hidden_layers
                and layerId >= hf_config.num_hidden_layers
                and not spec_decode
            ):
                continue
            maybe_matching_name = have_shared_expert(name)
            if (
                maybe_matching_name is not None
                # When the model keeps shared experts unfused (e.g. V4-Pro with
                # FP4 routed vs FP8 shared, or DP + mori all2all), do NOT rewrite
                # the shared weights into the fused slot — they must load into the
                # standalone Expert module. Stays True for models without this
                # attr (GLM4 etc.) so their fused-shared path is unchanged.
                and not getattr(model, "disable_fused_shared_loading", False)
                and should_fuse_shared_expert_weight(name, maybe_matching_name)
            ):
                # Preserve the module-naming prefix (mlp. / ffn.) so the rewritten
                # name matches this model's routed-expert param naming.
                module_prefix = maybe_matching_name.split("shared_expert", 1)[0]
                n_routed_experts = (
                    getattr(hf_config, "n_routed_experts", None)
                    or getattr(hf_config, "num_local_experts", None)
                    or getattr(hf_config, "num_experts", None)
                )
                if n_routed_experts is None:
                    raise AttributeError(
                        "Cannot remap shared expert weights without "
                        "n_routed_experts, num_local_experts, or num_experts "
                        "on the model config."
                    )
                name = name.replace(
                    maybe_matching_name,
                    f"{module_prefix}experts.{n_routed_experts}.",
                )
            for k in packed_modules_mapping:
                # We handle the experts below in expert_params_mapping
                if (
                    "mlp.experts." in name
                    or "ffn.experts." in name
                    or "block_sparse_moe.experts." in name
                ) and name not in params_dict:
                    continue
                if k in name:
                    packed_value = packed_modules_mapping[k]
                    # Handle both tuple (fuse parameter) and list (shard parameter)
                    if isinstance(packed_value, list):
                        # Checkpoint has fused weight, split into separate params
                        for shard_idx, target_name in enumerate(packed_value):
                            param_name = name.replace(k, target_name)
                            if "output_scale" not in param_name:
                                try:
                                    param = model.get_parameter(param_name)
                                except AttributeError:
                                    dropped_ckpt_keys.append(
                                        (_orig_ckpt_name, param_name)
                                    )
                                    continue
                                weight_loader = param.weight_loader
                                _submit(weight_loader, param, weight_tensor, shard_idx)
                                loaded_weights_record.add(prefix + param_name)
                    else:
                        # Checkpoint has separate weights, load into fused param
                        v, shard_id = packed_value
                        param_name = name.replace(k, v)
                        # FIXME output_scale has a value, so accuracy is incorrect. this should be loaded and used in llfp4.
                        if "output_scale" not in param_name:
                            try:
                                param = model.get_parameter(param_name)
                            except AttributeError:
                                dropped_ckpt_keys.append((_orig_ckpt_name, param_name))
                                break
                            weight_loader = param.weight_loader
                            _submit(weight_loader, param, weight_tensor, shard_id)
                            loaded_weights_record.add(prefix + param_name)
                    break
            else:
                # Detect fused expert format if model provides detection function
                if detect_fused_expert_fn is not None and not is_fused_expert:
                    is_fused_expert = detect_fused_expert_fn(name)
                    if is_fused_expert and get_fused_expert_mapping_fn is not None:
                        fused_expert_params_mapping = get_fused_expert_mapping_fn()

                # Check if model has expert mapping before processing
                if has_expert_mapping:
                    # Handle fused expert format
                    # Model-specific detection and handling via callback functions
                    if (
                        is_fused_expert
                        and load_fused_expert_weights_fn is not None
                        and fused_expert_params_mapping
                    ):
                        matched = False
                        for mapping_entry in fused_expert_params_mapping:
                            param_name, weight_name, shard_id = mapping_entry[:3]
                            if weight_name not in name:
                                continue
                            name_mapped = name.replace(weight_name, param_name)
                            if name_mapped not in params_dict:
                                continue

                            # Generic call - model provides implementation details
                            num_experts = getattr(
                                hf_config, "n_routed_experts", 0
                            ) or getattr(hf_config, "num_experts", 0)
                            matched = load_fused_expert_weights_fn(
                                name,  # Original checkpoint name
                                name_mapped,  # Mapped parameter name
                                params_dict,
                                weight_tensor,
                                shard_id,
                                num_experts,
                            )

                            if matched:
                                loaded_weights_record.add(prefix + name)
                                break

                        if matched:
                            continue

                    matched = False
                    for wm_name in expert_weight_prefixes:
                        if wm_name not in name:
                            continue
                        pm_name, expert_id, shard_id = expert_index[wm_name]
                        name = name.replace(wm_name, pm_name)
                        if (
                            name.endswith((".bias", "_bias"))
                            and name not in params_dict
                        ):
                            matched = True
                            break
                        if "mtp" in name and not spec_decode:
                            matched = True
                            break
                        param = params_dict.get(name)
                        if param is None:
                            # Parameter absent from model (e.g. weight scales for
                            # an unquantized drafter MTP block); skip silently.
                            matched = True
                            break
                        if executor is not None and _param_is_batchable(param, name):
                            _submit(
                                _stage_task,
                                param,
                                name,
                                shard_id,
                                expert_id,
                                weight_tensor,
                            )
                            loaded_weights_record.add(prefix + name)
                            matched = True
                            break
                        weight_loader = param.weight_loader
                        _submit(
                            weight_loader,
                            param,
                            weight_tensor,
                            name,
                            shard_id,
                            expert_id,
                        )
                        loaded_weights_record.add(prefix + name)
                        matched = True
                        break
                    if not matched:
                        if "mtp" in name and not spec_decode:
                            continue
                        if merged_target := extract_expert_target_and_id(name):
                            fused_name, expert_id = merged_target
                            try:
                                param = model.get_parameter(fused_name)
                            except AttributeError:
                                dropped_ckpt_keys.append((_orig_ckpt_name, fused_name))
                                continue
                            weight_loader = getattr(
                                param, "weight_loader", default_weight_loader
                            )
                            _submit(
                                weight_loader,
                                param,
                                weight_tensor,
                                "",  # use merged moe loader
                                "",
                                expert_id,
                            )
                            loaded_weights_record.add(prefix + fused_name)
                            continue
                        try:
                            param = model.get_parameter(name)
                        except AttributeError:
                            dropped_ckpt_keys.append((_orig_ckpt_name, name))
                            continue
                        weight_loader = getattr(
                            param, "weight_loader", default_weight_loader
                        )
                        _submit(weight_loader, param, weight_tensor)
                        loaded_weights_record.add(prefix + name)
                else:
                    # Model doesn't have expert mapping, use generic loading
                    try:
                        param = model.get_parameter(name)
                    except AttributeError:
                        dropped_ckpt_keys.append((_orig_ckpt_name, name))
                        continue
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    _submit(weight_loader, param, weight_tensor)
                    loaded_weights_record.add(prefix + name)

        if executor is not None:
            # Drain all tasks (surfacing errors) before the safety flush.
            for future in concurrent.futures.as_completed(futures):
                future.result()

        with staging_lock:
            pending = list(staging_map.values())
            staging_map.clear()
        if pending:
            raise RuntimeError(
                f"Batched loader: {len(pending)} MoE param group(s) under-filled "
                f"Set ATOM_LOADER_NUM_THREADS=1 to use the per-expert loader."
            )
    finally:
        if executor is not None:
            executor.shutdown(wait=True)

    # Verify every model parameter actually got loaded from the checkpoint.
    # Without this check, weights_mapping bugs (e.g. a substring rule
    # accidentally rewriting `attn_norm.weight` → `attn_model.norm.weight`)
    # silently leave the destination parameter at its init value (all-ones for
    # RMSNorm, all-zeros for newly-allocated buffers), corrupting forward
    # outputs in ways that are extremely hard to diagnose. WARN loudly here
    # so the failure surfaces at load time instead of at generation time.
    loaded_param_names = {
        n.removeprefix(prefix) if prefix else n for n in loaded_weights_record
    }
    expected_param_names = set(params_dict.keys())
    unloaded = sorted(expected_param_names - loaded_param_names)
    # Filter known-OK skips: post-load-derived params (e.g. FusedMoE shuffle
    # output buffers, weight_scale params merged from multiple checkpoint scales).
    # Heuristic: anything ending in `_shuffled`, `_packed`, etc. Conservative
    # default = report everything else.
    suppressed_suffixes = ("_shuffled", "_packed", "_meta_for_quant", "weight_scale_2")
    truly_unloaded = [
        n for n in unloaded if not any(n.endswith(s) for s in suppressed_suffixes)
    ]
    # Only report from rank 0 (other ranks have the same view).
    if truly_unloaded and is_rank0():
        sample = truly_unloaded[:20]
        logger.warning(
            "load_model: %d/%d model parameters were NOT loaded from "
            "checkpoint and remain at their init values. This is almost "
            "always a bug (typically a `weights_mapping` substring rule "
            "that accidentally renames a param to something the model "
            "doesn't have). Fix the mapping or the on-disk → param name "
            "translation. First %d unloaded names: %s",
            len(truly_unloaded),
            len(expected_param_names),
            len(sample),
            sample,
        )

    # Reverse direction: ckpt names that were silently dropped by
    # `get_parameter` AttributeError. These are the actionable bug class —
    # the mapping rewrote the ckpt name to something the model has no slot for,
    # so legitimate ckpt data was thrown away. Filter known-benign families
    # (output_scale, kv_scale, etc.) so the warning is signal, not noise.
    if dropped_ckpt_keys:
        benign_substrings = (
            "output_scale",
            "kv_scale",
            "inv_freq",
            "weight_scale_2",
        )
        actionable_drops = [
            (orig, mapped)
            for orig, mapped in dropped_ckpt_keys
            if not any(s in orig or s in mapped for s in benign_substrings)
        ]
        if actionable_drops and is_rank0():
            sample = actionable_drops[:20]
            logger.warning(
                "load_model: %d checkpoint tensors were silently dropped "
                "because the rewritten name has no matching model parameter. "
                "This is a `weights_mapping` / `WeightsMapper` bug — real "
                "ckpt data is being thrown away. Fix the rewrite rule. "
                "First %d (orig_ckpt_name → rewritten_name): %s",
                len(actionable_drops),
                len(sample),
                sample,
            )

    # Avoid holding stale Parameter refs that prevent storage release.
    del params_dict

    return loaded_weights_record
