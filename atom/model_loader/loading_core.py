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
from collections.abc import Callable, Iterable

import torch
from torch import nn
from transformers import AutoConfig

from atom.model_loader.expert_staging import ExpertStagingPool
from atom.model_loader.weight_dispatch import WeightDispatcher
from atom.model_loader.weight_names import (
    CheckpointNameRewriter,
    WeightsMapper,
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
    weights_iterator: Callable[..., Iterable[tuple[str, torch.Tensor]]],
) -> set[str]:
    """Copy every checkpoint tensor into the model parameter it belongs to.

    The four keyword-only callables are the host environment this module
    refuses to import itself:

    - ``default_weight_loader``  fallback copy for params without their own
    - ``fuse_shared_expert``     ``(shared_prefix, routed_prefix) -> fuse?``
    - ``is_rank0``               suppress duplicate diagnostics off rank 0
    - ``weights_iterator``       ``(path, disable_mmap, wants) -> (name, tensor)``
    """

    def _n_routed_experts() -> int | None:
        return (
            getattr(hf_config, "n_routed_experts", None)
            or getattr(hf_config, "num_local_experts", None)
            or getattr(hf_config, "num_experts", None)
        )

    # need to record the loaded weight name for vllm load check
    # it is only used in plugin mode for vllm

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
    rewriter = CheckpointNameRewriter(
        weights_mapper=weights_mapper,
        weights_mapping=weights_mapping,
        skip_weight_prefixes=skip_weight_prefixes,
        mtp_remap=mtp_remap,
        spec_decode=spec_decode,
        num_hidden_layers=hf_config.num_hidden_layers,
        n_routed_experts=_n_routed_experts(),
        fuse_shared_expert=fuse_shared_expert,
        # Stays False for models without the attribute (GLM4 etc.), so their
        # fused-shared path is unchanged.
        disable_fused_shared_loading=getattr(
            model, "disable_fused_shared_loading", False
        ),
    )
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
    moe_module_cache: dict = {}

    def _lookup_moe_module(full_param_name: str):
        module_path = full_param_name.rsplit(".", 1)[0]
        if module_path not in moe_module_cache:
            moe_module_cache[module_path] = (
                model.get_submodule(module_path) if "." in full_param_name else None
            )
        return moe_module_cache[module_path]

    staging_pool = ExpertStagingPool(_lookup_moe_module)

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

    dispatcher = WeightDispatcher(
        model=model,
        params_dict=params_dict,
        hf_config=hf_config,
        prefix=prefix,
        spec_decode=spec_decode,
        submit=_submit,
        staging_pool=staging_pool,
        batching_enabled=executor is not None,
        default_weight_loader=default_weight_loader,
        packed_modules_mapping=packed_modules_mapping,
        expert_index=expert_index,
        expert_weight_prefixes=expert_weight_prefixes,
        has_expert_mapping=has_expert_mapping,
        detect_fused_expert_fn=getattr(model, "detect_fused_expert_format", None),
        get_fused_expert_mapping_fn=getattr(model, "get_fused_expert_mapping", None),
        load_fused_expert_weights_fn=load_fused_expert_weights_fn,
    )

    # Rewriting a name is the same question as "is this tensor wanted", and the
    # iterator asks it before materializing anything -- so answer it once and
    # keep the answer. On a target-model load nearly every tensor is wanted, and
    # rewriting is a dozen substring scans plus a regex per tensor.
    rewritten: dict[str, str | None] = {}

    def _wanted(ckpt_name: str) -> bool:
        if ckpt_name not in rewritten:
            rewritten[ckpt_name] = rewriter.rewrite(ckpt_name)
        return rewritten[ckpt_name] is not None

    try:
        disable_mmap = envs.ATOM_DISABLE_MMAP
        # Reject by name before the tensor is materialized. A drafter load reads
        # the whole target checkpoint to pick out the MTP block, so most shards
        # can be skipped without being read at all. Under `--load_dummy` nothing
        # is loaded, so nothing is wanted -- and the rewriter, which is allowed
        # to raise on a checkpoint it cannot map, is never consulted.
        for name, weight_tensor in weights_iterator(
            model_name_or_path,
            disable_mmap,
            (lambda _: False) if load_dummy else _wanted,
        ):
            if load_dummy:
                continue
            _orig_ckpt_name = name  # preserve for ckpt-side coverage report
            # Normally a cache hit: the iterator just asked. Recomputed only if
            # a caller supplied an iterator that ignores the predicate.
            name = rewritten[name] if name in rewritten else rewriter.rewrite(name)
            if name is None:
                continue
            dispatcher.dispatch(_orig_ckpt_name, name, weight_tensor)

        if executor is not None:
            # Drain all tasks (surfacing errors) before the safety flush.
            for future in concurrent.futures.as_completed(futures):
                future.result()

        loaded_weights_record = dispatcher.loaded_weights_record
        dropped_ckpt_keys = dispatcher.dropped_ckpt_keys

        # Whatever the pool still holds is written back here; anything short of
        # its expected region count means the checkpoint never delivered some
        # routed base experts. The per-parameter check further down is too
        # coarse to see this -- it only knows whether a parameter was touched
        # at all -- so report it while the (slot, shard) detail is still around.
        staging_report = staging_pool.flush_pending()
        if staging_report.incomplete:
            detail = "\n  ".join(staging_report.incomplete)
            message = (
                f"Batched loader: {len(staging_report.incomplete)} MoE "
                f"parameter(s) did not receive every routed expert from the "
                f"checkpoint:\n  {detail}"
            )
            if envs.ATOM_LOADER_STRICT_COVERAGE:
                raise RuntimeError(
                    f"{message}\nSet ATOM_LOADER_STRICT_COVERAGE=false to load "
                    "anyway, leaving those expert slots at their init values."
                )
            logger.warning("%s\nLoading anyway (strict coverage disabled).", message)
    finally:
        if executor is not None:
            executor.shutdown(wait=True)

    _report_coverage(
        loaded_weights_record=loaded_weights_record,
        params_dict=params_dict,
        dropped_ckpt_keys=dropped_ckpt_keys,
        prefix=prefix,
        is_rank0=is_rank0,
    )

    # Avoid holding stale Parameter refs that prevent storage release.
    del params_dict

    return loaded_weights_record


def _report_coverage(
    *,
    loaded_weights_record: set[str],
    params_dict: dict,
    dropped_ckpt_keys: list[tuple[str, str]],
    prefix: str,
    is_rank0: Callable[[], bool],
) -> None:
    """Warn about parameters nothing wrote, and checkpoint tensors nothing took.

    Both directions point at the same bug class -- a name-rewrite rule that
    produces a name the model does not have -- and both are silent otherwise:
    the destination keeps its init value (all-ones for RMSNorm, whatever
    `torch.empty` left elsewhere) and the forward pass is quietly wrong.
    """
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
