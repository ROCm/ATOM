# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import logging
from typing import Optional

import torch

logger = logging.getLogger("atom")

# The fused buffers a routed-expert weight can land in, and the shard ids that
# together make up one expert's slice of each. Re-establishing the layout works
# on a whole slice, so it can only run once every part of that slice has been
# rewritten -- shuffling a half-rewritten one mixes two layouts.
_EXPERT_BUFFER_SHARDS = {
    "w13_weight": frozenset({"w1", "w3"}),
    "w2_weight": frozenset({"w2"}),
}

# A trainer whose transformers keeps MoE experts fused sends one 3D tensor per
# layer instead of three per expert: (E, 2I, H) gate_up_proj and (E, H, I)
# down_proj. Same buffers, same dim order, w13's first half along the
# intermediate dim being the gate projection -- only the leaf name differs.
# Accepting both means a caller does not have to know which convention ATOM
# happens to use, nor pre-apply the kernel layout on ATOM's behalf.
_FUSED_EXPERT_LEAVES = {
    "gate_up_proj": ("w13_weight", ("w1", "w3")),
    "down_proj": ("w2_weight", ("w2",)),
}

_EXPERTS_PREFIX_SUFFIX = ".experts"


class WeightUpdaterMixin:
    """Mixin providing weight update capabilities for ModelRunner.

    Host class must provide:
      - self.model (nn.Module)
      - self.device (torch.device)
      - self.rank (int) — TP rank
      - self.world_size (int) — TP size
      - self.label (str)
      - self.clear_kv_cache() — method
    """

    def _get_param_to_module_mapping(self) -> dict[str, tuple]:
        """
        Get or build the parameter name to module mapping.

        This mapping is cached after the first call to avoid expensive
        rebuilding on every weight update.

        Returns:
            Dict mapping parameter full name to (module, param_name, param) tuple
        """
        if not hasattr(self, "_param_to_module") or self._param_to_module is None:
            self._param_to_module = {}
            for module_name, module in self.model.named_modules():
                for param_name, param in module.named_parameters(recurse=False):
                    full_name = (
                        f"{module_name}.{param_name}" if module_name else param_name
                    )
                    self._param_to_module[full_name] = (module, param_name, param)
            logger.debug(
                f"{self.label}: Built param_to_module mapping with "
                f"{len(self._param_to_module)} parameters"
            )
        return self._param_to_module

    def _get_packed_modules_mapping(self) -> dict:
        if not hasattr(self, "_cached_packed_mapping"):
            self._cached_packed_mapping = (
                getattr(self.model, "packed_modules_mapping", None) or {}
            )
        return self._cached_packed_mapping

    def _get_packed_shard_order(self) -> dict[str, list]:
        """Build {target_suffix: [shard_id_0, shard_id_1, ...]} preserving declaration order."""
        if not hasattr(self, "_cached_packed_shard_order"):
            order: dict[str, list] = {}
            for _, (tgt, shard_id) in self._get_packed_modules_mapping().items():
                order.setdefault(tgt, []).append(shard_id)
            self._cached_packed_shard_order = order
        return self._cached_packed_shard_order

    def _resolve_packed_name(
        self, name: str, param_to_module: dict
    ) -> tuple[str, object, str] | None:
        """Try to resolve an HF name to an ATOM packed parameter.

        Returns (atom_full_name, shard_id, target_suffix) or None.
        """
        for src_suffix, (
            tgt_suffix,
            shard_id,
        ) in self._get_packed_modules_mapping().items():
            if src_suffix in name:
                atom_name = name.replace(src_suffix, tgt_suffix)
                if atom_name in param_to_module:
                    return atom_name, shard_id, tgt_suffix
        return None

    def _apply_packed_weight(
        self,
        name: str,
        tensor: torch.Tensor,
        param_to_module: dict,
    ) -> str:
        """Handle a single incoming weight that belongs to a packed (fused) module.

        For FP8 params, shards are accumulated in a float32 buffer using the
        module's weight_loader (which handles GQA-aware TP sharding for QKV).
        Once all shards arrive, the buffer is requantized to FP8 in one shot.

        Returns:
            'updated'     – fused param fully updated (all shards received)
            'accumulated' – shard stored, waiting for remaining shards
            'skipped'     – not a packed param or lookup failed
        """
        resolved = self._resolve_packed_name(name, param_to_module)
        if resolved is None:
            return "skipped"

        atom_name, shard_id, tgt_suffix = resolved
        module, param_name, param = param_to_module[atom_name]
        weight_loader = getattr(module, "weight_loader", None)
        if weight_loader is None:
            return "skipped"

        if self._is_fp8_param(module, param) and tensor.dtype != param.dtype:
            if not hasattr(self, "_packed_weight_accum"):
                self._packed_weight_accum = {}

            if atom_name not in self._packed_weight_accum:
                self._packed_weight_accum[atom_name] = {"shards": {}}

            self._packed_weight_accum[atom_name]["shards"][shard_id] = tensor.clone()

            expected = self._get_packed_shard_order().get(tgt_suffix, [])
            if set(self._packed_weight_accum[atom_name]["shards"].keys()) >= set(
                expected
            ):
                buf = torch.nn.Parameter(
                    torch.zeros(param.shape, dtype=torch.float32, device=self.device),
                    requires_grad=False,
                )
                # The accumulation buffer is a fresh Parameter, so it carries
                # none of the target's attributes, and weight_loader() reads
                # weight_loader_process off the parameter it is handed.
                wlp = getattr(param, "weight_loader_process", None)
                if wlp is not None:
                    buf.weight_loader_process = wlp

                for sid in expected:
                    shard_t = self._packed_weight_accum[atom_name]["shards"][sid]
                    shard_gpu = shard_t.to(device=self.device, dtype=torch.float32)
                    weight_loader(buf, shard_gpu, sid)

                self._requantize_fp8_weight(module, param_name, param, buf.data)
                del self._packed_weight_accum[atom_name]
                logger.debug(
                    f"{self.label}: FP8 packed weight updated: {atom_name} "
                    f"(composed from {len(expected)} shards)"
                )
                return "updated"
            return "accumulated"

        tensor_gpu = tensor.to(device=self.device)
        weight_loader(param, tensor_gpu, shard_id)
        return "updated"

    def _apply_unmatched_weight(
        self,
        name: str,
        tensor: torch.Tensor,
        param_to_module: dict,
    ) -> str:
        """A name that is not a parameter of the model as ATOM built it.

        Either one expert of a fused MoE, or one shard of a packed module.
        """
        result = self._apply_expert_weight(name, tensor, param_to_module)
        if result == "skipped":
            result = self._apply_packed_weight(name, tensor, param_to_module)
        return result

    def _get_expert_params_mapping(self) -> list[tuple[str, str, int, str]]:
        """[(ckpt weight fragment, ATOM param fragment, expert_id, shard_id)].

        The same mapping the model loader consults, from the same
        ``model.get_expert_mapping()``, ordered longest fragment first so a
        more specific one wins. Built once; models with no MoE layer leave it
        empty and every lookup then short-circuits.
        """
        if not hasattr(self, "_cached_expert_mapping"):
            get_expert_mapping = getattr(self.model, "get_expert_mapping", None)
            entries = [
                (weight_name_part, param_name_part, expert_id, shard_id)
                for param_name_part, weight_name_part, expert_id, shard_id in (
                    get_expert_mapping() if callable(get_expert_mapping) else ()
                )
            ]
            entries.sort(key=lambda entry: len(entry[0]), reverse=True)
            self._cached_expert_mapping = entries
        return self._cached_expert_mapping

    @property
    def _pending_expert_relayout(self) -> dict:
        """``{(module, param_name): {expert_id: {shard_id, ...}}}`` for this sync.

        Accumulates across buckets and is consumed by
        ``_finalize_expert_weight_sync`` when the last one lands, so an
        expert whose w1 and w3 arrive in different buckets is still relaid out
        exactly once.
        """
        if not hasattr(self, "_expert_relayout_pending"):
            self._expert_relayout_pending = {}
        return self._expert_relayout_pending

    def _apply_expert_weight(
        self,
        name: str,
        tensor: torch.Tensor,
        param_to_module: dict,
    ) -> str:
        """Route one routed-expert weight into its FusedMoE buffer.

        A model's experts arrive one tensor per expert and land in the fused
        w13_weight / w2_weight of the layer's FusedMoE, which is neither the
        incoming name nor anything packed_modules_mapping describes. Without
        this the tensor matches nothing and is counted as skipped, at debug
        level -- the rollout then serves whatever the experts held at load
        time and nothing says so. On Qwen3-30B-A3B that is 96 tensors per
        replica per sync, 48 layers x 2.

        Returns 'updated' or 'skipped'. Never 'updated' for a combination
        this path does not implement: see _check_expert_sync_supported.
        """
        fused = self._apply_fused_expert_weight(name, tensor, param_to_module)
        if fused != "skipped":
            return fused

        for (
            weight_name_part,
            param_name_part,
            expert_id,
            shard_id,
        ) in self._get_expert_params_mapping():
            if weight_name_part not in name:
                continue
            atom_name = name.replace(weight_name_part, param_name_part)
            if atom_name not in param_to_module:
                continue
            module, param_name, param = param_to_module[atom_name]
            weight_loader = getattr(module, "weight_loader", None)
            if not callable(weight_loader):
                continue

            self._check_expert_sync_supported(
                name, atom_name, param_name, module, param, tensor
            )
            weight_loader(
                param,
                tensor.to(device=self.device),
                weight_name=name,
                shard_id=shard_id,
                expert_id=expert_id,
            )
            # The layout these buffers must end up in is re-established once
            # per sync -- see _finalize_expert_weight_sync.
            arrived = self._pending_expert_relayout.setdefault((module, param_name), {})
            arrived.setdefault(expert_id, set()).add(shard_id)
            return "updated"
        return "skipped"

    def _apply_fused_expert_weight(
        self,
        name: str,
        tensor: torch.Tensor,
        param_to_module: dict,
    ) -> str:
        """Route a trainer's fused 3D expert tensor into the same buffers.

        One (E, 2I, H) ``...experts.gate_up_proj`` covers every expert and both
        halves of w13, so it is driven through ``weight_loader`` once per half:
        a 3D ``loaded_weight`` puts the loader on its full-load path, where the
        expert dimension is written whole and the intermediate dimension is
        still narrowed by TP rank.

        Returns 'updated' or 'skipped'.
        """
        prefix, _, leaf = name.rpartition(".")
        entry = _FUSED_EXPERT_LEAVES.get(leaf)
        if entry is None or not prefix.endswith(_EXPERTS_PREFIX_SUFFIX):
            return "skipped"
        atom_leaf, shard_ids = entry
        atom_name = f"{prefix}.{atom_leaf}"
        if atom_name not in param_to_module:
            return "skipped"
        module, param_name, param = param_to_module[atom_name]
        weight_loader = getattr(module, "weight_loader", None)
        if not callable(weight_loader):
            return "skipped"

        self._check_expert_sync_supported(
            name, atom_name, param_name, module, param, tensor
        )
        if tensor.dim() != 3:
            raise NotImplementedError(
                f"{self.label}: {name} resolves to the fused expert buffer "
                f"{atom_name}, which needs a 3D (experts, out, in) tensor; got "
                f"{tuple(tensor.shape)}."
            )

        gpu = tensor.to(device=self.device)
        # Split w13's gate and up halves along the intermediate dim, the way
        # the buffer stacks them. w2 arrives whole. Views, not copies: the
        # loader's copy handles a strided source, and materialising these
        # would double the largest tensor in the sync.
        for shard_id, chunk in zip(shard_ids, gpu.chunk(len(shard_ids), dim=1)):
            weight_loader(
                param,
                chunk,
                # _copy_expert_shard dispatches on the name containing
                # "weight"; the fused leaf names do not, so hand it the
                # resolved ATOM name.
                weight_name=atom_name,
                shard_id=shard_id,
                expert_id=0,
            )
        # One tensor covers every expert, so every slice of the buffer is new.
        arrived = self._pending_expert_relayout.setdefault((module, param_name), {})
        for expert_id in range(param.shape[0]):
            arrived.setdefault(expert_id, set()).update(shard_ids)
        return "updated"

    def _check_expert_sync_supported(
        self,
        name: str,
        atom_name: str,
        param_name: str,
        module: torch.nn.Module,
        param: torch.nn.Parameter,
        tensor: torch.Tensor,
    ) -> None:
        """Refuse the combinations this path does not actually implement.

        Loud, and before the write. ``FusedMoE.weight_loader`` copies into
        whatever buffer it is handed: on a quantized layer it byte-copies or
        numerically casts, either of which leaves the expert computing
        something else while the sync reports updated=1. A silent wrong answer
        in a rollout is worse than a stopped job.
        """
        if param_name not in _EXPERT_BUFFER_SHARDS:
            raise NotImplementedError(
                f"{self.label}: routed-expert weight sync writes "
                f"{sorted(_EXPERT_BUFFER_SHARDS)}, not {param_name!r} "
                f"(resolved from {name!r}). Scales and packed metadata are not "
                f"synced; send an unquantized MoE checkpoint."
            )
        if not (param.dtype.is_floating_point and param.element_size() >= 2):
            raise NotImplementedError(
                f"{self.label}: {atom_name} holds experts as {param.dtype}, a "
                f"quantized storage format. Writing {tensor.dtype} into it needs "
                f"the weight and its scale recomputed together, which this path "
                f"does not do -- FusedMoE.weight_loader would byte-copy or "
                f"numerically cast, leaving the existing scale describing the old "
                f"weight. Run the rollout with an unquantized MoE, or extend this "
                f"path with a requantizing loader for the format."
            )
        if getattr(module, "expert_map", None) is not None or getattr(
            module, "num_redundant_experts", 0
        ):
            raise NotImplementedError(
                f"{self.label}: {atom_name} is expert-parallel or carries redundant "
                f"expert replicas. Incoming ids then address a rank's local slots "
                f"through expert_map, only some arrive on this rank, and the "
                f"replicas are filled after loading rather than sent -- none of "
                f"which this path tracks. Run the rollout MoE with EP off."
            )

    def _finalize_expert_weight_sync(self) -> None:
        """Re-establish the expert layout for the slices this sync rewrote.

        FusedMoE holds w13_weight / w2_weight in the permutation its aiter
        kernel reads. ``weight_loader`` writes plain row-major bytes over it,
        so a sync has to re-establish that layout exactly as the initial load
        does -- once, after the last shard.

        Not by re-running ``process_weights_after_loading``. Those hooks are
        initialisation, not a repeatable transform: they fold scales, hand the
        module new Parameter objects through ``atom_parameter()`` while a
        captured CUDA graph and ``_param_to_module`` still point at the old
        ones, and several are not idempotent at all. ``Fp8MoEMethod``'s
        per-tensor path collapses ``w13_weight_scale`` from [E, 2] to [E] on
        its first call, so a second raises IndexError on ``max(dim=1)``; the
        channel and block paths re-shuffle weights that are already shuffled,
        which does not undo the first shuffle but produces a third layout.

        What a sync needs is the layout step alone, on only the slices that
        were rewritten, in place.
        """
        pending = self._pending_expert_relayout
        if not pending:
            return

        from atom.model_ops.utils import shuffle_expert_slices

        experts = 0
        for (module, param_name), arrived in pending.items():
            required = _EXPERT_BUFFER_SHARDS[param_name]
            for expert_id, shards in sorted(arrived.items()):
                missing = sorted(required - shards)
                if missing:
                    raise RuntimeError(
                        f"{self.label}: expert {expert_id} of {param_name} was "
                        f"rewritten without {missing}. Its slice is half new and "
                        f"half old, and re-establishing the layout over that "
                        f"would mix two layouts. Send every shard of an expert "
                        f"in the same weight update."
                    )
            shuffle_expert_slices(getattr(module, param_name), sorted(arrived))
            experts += len(arrived)
        logger.info(
            f"{self.label}: expert layout re-established for {experts} expert "
            f"slices across {len(pending)} fused buffers"
        )
        pending.clear()

    def _try_shard_weight(
        self,
        param: torch.nn.Parameter,
        tensor: torch.Tensor,
        tp_rank: int,
        tp_size: int,
    ) -> bool:

        param_shape = param.shape
        tensor_shape = tensor.shape

        if len(param_shape) != len(tensor_shape):
            return False

        # Find which dimension needs sharding
        shard_dim = None
        for dim in range(len(param_shape)):
            if tensor_shape[dim] == param_shape[dim] * tp_size:
                shard_dim = dim
                break
            elif tensor_shape[dim] != param_shape[dim]:
                # Dimension mismatch but not by tp_size factor
                return False

        if shard_dim is None:
            # No dimension needs sharding but shapes don't match
            return False

        # Shard the tensor along the identified dimension
        shard_size = param_shape[shard_dim]
        start_idx = tp_rank * shard_size

        tensor = tensor.to(device=self.device, dtype=param.dtype)
        sharded_tensor = tensor.narrow(shard_dim, start_idx, shard_size)
        param.data.copy_(sharded_tensor)

        return True

    @staticmethod
    def _is_fp8_param(module: torch.nn.Module, param: torch.nn.Parameter) -> bool:
        return (
            param.dtype.is_floating_point
            and param.element_size() < 2
            and getattr(module, "weight_scale", None) is not None
        )

    def _requantize_fp8_weight(
        self,
        module: torch.nn.Module,
        param_name: str,
        param: torch.nn.Parameter,
        tensor: torch.Tensor,
    ) -> None:
        """Requantize a full-precision weight to FP8 with updated weight_scale.

        Called when FSDP sends float32/bfloat16 trained weights to an FP8 model.
        Computes new per-block (or per-tensor/per-token) scale factors and writes
        both the FP8 weight and scale into the module in place.
        """
        weight_scale = module.weight_scale
        fp8_dtype = param.dtype
        fp8_max = torch.finfo(fp8_dtype).max

        tensor_gpu = tensor.to(device=self.device, dtype=torch.float32)

        tp_size = self.world_size
        if tp_size > 1 and tensor_gpu.shape != param.shape:
            for dim in range(len(param.shape)):
                if tensor_gpu.shape[dim] == param.shape[dim] * tp_size:
                    shard_size = param.shape[dim]
                    tensor_gpu = tensor_gpu.narrow(
                        dim, self.rank * shard_size, shard_size
                    )
                    break

        if tensor_gpu.shape != param.shape:
            logger.warning(
                f"{self.label}: Shape mismatch in FP8 requantize for {param_name}: "
                f"param={param.shape}, tensor={tensor_gpu.shape}"
            )
            return

        from aiter import QuantType as _QT

        quant_type = getattr(module, "quant_type", None)

        if quant_type is not None and quant_type.value == _QT.per_1x128.value:
            # Must match the load-time online_quantize_weight layout: a true
            # 128x128 block scale of shape (N//128, K//128). The previous code
            # produced a 1x128-along-K scale (N, K//128) and sliced it into the
            # (N//128, K//128) buffer, which is inconsistent with the blockscale
            # GEMM and collapses generation after the first weight update.
            from atom.quantization.quark.utils import (
                quantize_weight_to_fp8_128x128_blockscale,
            )

            q_weight, scale = quantize_weight_to_fp8_128x128_blockscale(
                tensor_gpu, fp8_dtype
            )
            param.data.copy_(q_weight)
            weight_scale.data.copy_(scale.to(weight_scale.dtype))

        elif quant_type is not None and quant_type.value == _QT.per_Tensor.value:
            amax = tensor_gpu.abs().max()
            scale = (amax / fp8_max).clamp(min=1e-12)
            param.data.copy_((tensor_gpu / scale).to(fp8_dtype))
            weight_scale.data.fill_(scale.item())

        elif quant_type is not None and quant_type.value == _QT.per_Token.value:
            row_amax = tensor_gpu.abs().amax(dim=-1, keepdim=True)
            scale = (row_amax / fp8_max).clamp(min=1e-12)
            param.data.copy_((tensor_gpu / scale).to(fp8_dtype))
            weight_scale.data.copy_(scale.to(weight_scale.dtype))

        else:
            logger.warning(
                f"{self.label}: Unknown quant_type {quant_type} for FP8 requantize"
            )
            return

        self._post_process_fp8_weight(module, param)
        logger.debug(
            f"{self.label}: FP8 requantized {param_name} on {type(module).__name__}, "
            f"quant_type={quant_type}, scale_shape={weight_scale.shape}"
        )

    def _post_process_fp8_weight(
        self,
        module: torch.nn.Module,
        param: torch.nn.Parameter,
    ) -> None:
        """Post-process an FP8 weight after update: normalization and shuffle.

        Must be called after any FP8 weight write (both requantize and direct copy)
        to ensure the weight layout matches what ATOM's GEMM kernels expect.
        """
        weight_scale = getattr(module, "weight_scale", None)

        if (
            getattr(module, "need_normalize_e4m3fn_to_e4m3fnuz", False)
            and weight_scale is not None
        ):
            from atom.model_ops.utils import normalize_e4m3fn_to_e4m3fnuz

            param.data, weight_scale.data, _ = normalize_e4m3fn_to_e4m3fnuz(
                param.data, weight_scale.data
            )

        quant_type = getattr(module, "quant_type", None)
        if quant_type is None:
            return

        from atom.model_ops.linear import weight_is_stored_preshuffled
        from atom.model_ops.utils import shuffle_weights

        # The same decision the initial load makes, from the same function.
        needs_shuffle = weight_is_stored_preshuffled(
            quant_type,
            getattr(module, "params_dtype", param.dtype),
            needs_preshuffled_weight=getattr(module, "needs_preshuffled_weight", False),
        )

        # And the same rank check. 3D is Qwen3-Next's GDN conv1d, which the
        # loader deliberately leaves row-major; shuffling it here would be the
        # divergence rather than the fix.
        if needs_shuffle and param.dim() == 2:
            shuffle_weights(param)

    def update_weights(
        self, named_tensors: list[tuple[str, torch.Tensor]], clear_kv_cache: bool = True
    ) -> int:
        """
        Update model weights from named tensors.

        Called by RLHF frameworks after each training step to
        synchronize weights from training engine to inference engine.

        Supports both direct parameter names and HuggingFace-style names that
        map to ATOM's fused parameters (qkv_proj, gate_up_proj) via the model's
        packed_modules_mapping.

        Args:
            named_tensors: List of (parameter_name, tensor) tuples.
                           Tensors should be full (unsharded) weights.
            clear_kv_cache: Whether to clear KV cache after update

        Returns:
            Number of parameters successfully updated
        """
        param_to_module = self._get_param_to_module_mapping()

        updated = 0
        skipped = 0
        ignored_scales = 0

        for name, tensor in named_tensors:
            if name not in param_to_module:
                result = self._apply_unmatched_weight(name, tensor, param_to_module)
                if result == "updated":
                    updated += 1
                elif result == "accumulated":
                    pass
                elif "weight_scale" in name or "input_scale" in name:
                    ignored_scales += 1
                else:
                    logger.debug(f"{self.label}: Unmatched parameter: {name}")
                    skipped += 1
                continue

            module, param_name, param = param_to_module[name]
            weight_loader = getattr(module, "weight_loader", None)

            if self._is_fp8_param(module, param) and tensor.dtype != param.dtype:
                self._requantize_fp8_weight(module, param_name, param, tensor)
                updated += 1
            elif self._is_fp8_param(module, param) and tensor.dtype == param.dtype:
                tensor = tensor.to(device=self.device)
                param.data.copy_(tensor)
                self._post_process_fp8_weight(module, param)
                updated += 1
            elif tensor.shape == param.shape:
                tensor = tensor.to(device=self.device, dtype=param.dtype)
                param.data.copy_(tensor)
                updated += 1
            elif weight_loader is not None and callable(weight_loader):
                try:
                    tensor = tensor.to(device=self.device)
                    weight_loader(param, tensor)
                    updated += 1
                except Exception as e:
                    logger.warning(
                        f"{self.label}: weight_loader failed for {name}: {e}"
                    )
                    skipped += 1
            else:
                tp_size = self.world_size
                tp_rank = self.rank
                if tp_size > 1 and self._try_shard_weight(
                    param, tensor, tp_rank, tp_size
                ):
                    updated += 1
                else:
                    logger.warning(
                        f"{self.label}: Shape mismatch for {name}: "
                        f"expected {param.shape}, got {tensor.shape}"
                    )
                    skipped += 1

        self._finalize_expert_weight_sync()

        if clear_kv_cache:
            self.clear_kv_cache()

        if hasattr(self, "_packed_weight_accum"):
            self._packed_weight_accum.clear()

        logger.info(
            f"{self.label}: Weight update complete - "
            f"updated={updated}, skipped={skipped}, "
            f"ignored_scales={ignored_scales}"
        )
        return updated

    def update_weights_from_shm(
        self,
        shm_name: str,
        bucket_meta: dict,
        is_last: bool = True,
    ) -> int:
        """
        Update model weights by reading tensor data from POSIX shared memory.

        Only lightweight metadata (shm_name, bucket_meta) is transmitted through
        the control path (EngineCore -> MessageQueue).  The heavy tensor payload
        resides in ``/dev/shm/<shm_name>`` and each ModelRunner maps it directly.

        Args:
            shm_name: Name of the POSIX shared-memory segment created by the
                       caller (LLMEngine).
            bucket_meta: ``{param_name: {"shape": tuple, "dtype": str,
                       "offset": int, "nbytes": int}}``.
            is_last: If ``True``, clear the KV cache after applying the weights
                     (last bucket in a multi-bucket transfer).

        Returns:
            Number of parameters successfully updated in this bucket.
        """
        from multiprocessing import shared_memory as _shm_mod
        from unittest.mock import patch

        # Open the existing shared-memory segment (do NOT unlink – caller owns it)
        with patch(
            "multiprocessing.resource_tracker.register",
            lambda *args, **kwargs: None,
        ):
            shm = _shm_mod.SharedMemory(name=shm_name)

        try:
            buffer = torch.frombuffer(shm.buf, dtype=torch.uint8)
            param_to_module = self._get_param_to_module_mapping()

            updated = 0
            skipped = 0
            ignored_scales = 0

            for name, meta in bucket_meta.items():
                # Reconstruct a CPU tensor view from shared memory
                dtype_str = meta["dtype"].replace("torch.", "")
                dtype = getattr(torch, dtype_str)
                offset = meta["offset"]
                nbytes = meta["nbytes"]
                tensor = (
                    buffer[offset : offset + nbytes]
                    .view(dtype=dtype)
                    .view(meta["shape"])
                )

                if name not in param_to_module:
                    result = self._apply_unmatched_weight(name, tensor, param_to_module)
                    if result == "updated":
                        updated += 1
                    elif result == "accumulated":
                        pass
                    elif "weight_scale" in name or "input_scale" in name:
                        ignored_scales += 1
                    else:
                        logger.debug(f"{self.label}: Unmatched parameter: {name}")
                        skipped += 1
                    continue

                module, param_name, param = param_to_module[name]
                weight_loader = getattr(module, "weight_loader", None)

                if self._is_fp8_param(module, param) and tensor.dtype != param.dtype:
                    self._requantize_fp8_weight(module, param_name, param, tensor)
                    updated += 1
                elif self._is_fp8_param(module, param) and tensor.dtype == param.dtype:
                    tensor = tensor.to(device=self.device)
                    param.data.copy_(tensor)
                    self._post_process_fp8_weight(module, param)
                    updated += 1
                elif tensor.shape == param.shape:
                    tensor = tensor.to(device=self.device, dtype=param.dtype)
                    param.data.copy_(tensor)
                    updated += 1
                elif weight_loader is not None and callable(weight_loader):
                    try:
                        tensor = tensor.to(device=self.device)
                        weight_loader(param, tensor)
                        updated += 1
                    except Exception as e:
                        logger.warning(
                            f"{self.label}: weight_loader failed for {name}: {e}"
                        )
                        skipped += 1
                else:
                    tp_size = self.world_size
                    tp_rank = self.rank
                    if tp_size > 1 and self._try_shard_weight(
                        param, tensor, tp_rank, tp_size
                    ):
                        updated += 1
                    else:
                        logger.warning(
                            f"{self.label}: Shape mismatch for {name}: "
                            f"expected {param.shape}, got {tensor.shape}"
                        )
                        skipped += 1

            if is_last:
                self._finalize_expert_weight_sync()
                self.clear_kv_cache()
                if hasattr(self, "_packed_weight_accum"):
                    if self._packed_weight_accum:
                        logger.warning(
                            f"{self.label}: Incomplete packed weight accumulators: "
                            f"{list(self._packed_weight_accum.keys())}"
                        )
                    self._packed_weight_accum.clear()
            logger.info(
                f"{self.label}: SHM weight update bucket done - "
                f"updated={updated}, skipped={skipped}, "
                f"ignored_scales={ignored_scales}, is_last={is_last}"
            )
            return updated
        finally:
            shm.close()

    def update_weights_from_ipc(
        self,
        ipc_handle,
        bucket_meta: dict,
        is_last: bool = True,
        ipc_handles: Optional[dict] = None,
    ) -> int:
        """Update model weights by reading tensor data from a CUDA IPC shared buffer.

        The sender (typically the RLHF training process) has allocated a GPU
        buffer, copied weight data into it, and obtained a CUDA IPC handle via
        ``reduce_tensor()``.

        When ``ipc_handles`` (per-GPU) is provided, each ModelRunner opens
        ONLY its own GPU's handle — always same-GPU IPC, no cross-GPU
        ``hipIpcOpenMemHandle``.  This avoids the ROCm/MI300X crash where
        opening an IPC handle from a different physical GPU causes a
        "Memory access fault".

        When ``ipc_handles`` is ``None``, falls back to the original
        ``ipc_handle`` (single handle) behavior.

        Args:
            ipc_handle: CUDA IPC handle from ``reduce_tensor(buffer)`` in
                the sender process.  Used as fallback when ``ipc_handles``
                is not provided.
            bucket_meta: ``{param_name: {"shape": tuple, "dtype": str,
                       "offset": int, "nbytes": int}}``.
            is_last: If ``True``, clear the KV cache after applying the weights
                     (last bucket in a multi-bucket transfer).
            ipc_handles: Per-GPU IPC handles dict ``{device_index: handle}``.
                When provided, each ModelRunner opens the handle for its own
                GPU (same-GPU IPC, safe on ROCm).

        Returns:
            Number of parameters successfully updated in this bucket.
        """
        # Cache the IPC buffer mapping: only open once per weight-update cycle.
        if not hasattr(self, "_ipc_buffer") or self._ipc_buffer is None:
            from atom.rollout.weight_sync import rebuild_ipc_handle

            dp_rank_local = self.config.parallel_config.data_parallel_rank_local or 0
            global_device_idx = dp_rank_local * self.world_size + self.rank
            local_device_idx = self.device.index
            if ipc_handles is not None and global_device_idx in ipc_handles:
                self._ipc_buffer = rebuild_ipc_handle(
                    ipc_handles[global_device_idx], device_id=local_device_idx
                )
                logger.info(
                    f"{self.label}: opened per-GPU IPC buffer mapping "
                    f"(size={self._ipc_buffer.numel()} bytes, "
                    f"global_device_idx={global_device_idx}, local_device_idx={local_device_idx}, "
                    f"buffer_device={self._ipc_buffer.device}, "
                    f"runner_device={self.device})"
                )
            else:
                self._ipc_buffer = rebuild_ipc_handle(ipc_handle)
                logger.info(
                    f"{self.label}: opened IPC buffer mapping "
                    f"(size={self._ipc_buffer.numel()} bytes, "
                    f"buffer_device={self._ipc_buffer.device}, "
                    f"runner_device={self.device})"
                )
        buffer = self._ipc_buffer

        param_to_module = self._get_param_to_module_mapping()

        updated = 0
        skipped = 0
        ignored_scales = 0

        for name, meta in bucket_meta.items():
            dtype_str = meta["dtype"].replace("torch.", "")
            dtype = getattr(torch, dtype_str)
            offset = meta["offset"]
            nbytes = meta["nbytes"]

            # View into the IPC buffer (on sender's GPU), then copy to
            # this runner's device.  .to() always returns a new tensor
            # when the device differs; for same-device case we need an
            # explicit copy so the sender can safely overwrite the buffer.
            src = buffer[offset : offset + nbytes].view(dtype=dtype).view(meta["shape"])
            if src.device == self.device:
                tensor = src.clone()
            else:
                tensor = src.to(device=self.device)

            if name not in param_to_module:
                result = self._apply_unmatched_weight(name, tensor, param_to_module)
                if result == "updated":
                    updated += 1
                elif result == "accumulated":
                    pass
                elif "weight_scale" in name or "input_scale" in name:
                    ignored_scales += 1
                else:
                    logger.debug(f"{self.label}: Unmatched parameter: {name}")
                    skipped += 1
                continue

            module, param_name, param = param_to_module[name]
            weight_loader = getattr(module, "weight_loader", None)

            if self._is_fp8_param(module, param) and tensor.dtype != param.dtype:
                self._requantize_fp8_weight(module, param_name, param, tensor)
                updated += 1
            elif self._is_fp8_param(module, param) and tensor.dtype == param.dtype:
                param.data.copy_(tensor)
                self._post_process_fp8_weight(module, param)
                updated += 1
            elif tensor.shape == param.shape:
                if tensor.dtype != param.dtype:
                    tensor = tensor.to(dtype=param.dtype)
                param.data.copy_(tensor)
                updated += 1
            elif weight_loader is not None and callable(weight_loader):
                try:
                    weight_loader(param, tensor)
                    updated += 1
                except Exception as e:
                    logger.warning(
                        f"{self.label}: weight_loader failed for {name}: {e}"
                    )
                    skipped += 1
            else:
                tp_size = self.world_size
                tp_rank = self.rank
                if tp_size > 1 and self._try_shard_weight(
                    param, tensor, tp_rank, tp_size
                ):
                    updated += 1
                else:
                    logger.warning(
                        f"{self.label}: Shape mismatch for {name}: "
                        f"expected {param.shape}, got {tensor.shape}"
                    )
                    skipped += 1

        # Only release the IPC buffer mapping on the last bucket
        if is_last:
            self._finalize_expert_weight_sync()
            self._ipc_buffer = None
            try:
                torch.cuda.ipc_collect()
            except Exception:
                pass  # ipc_collect may not be available on all platforms

            self.clear_kv_cache()
            if hasattr(self, "_packed_weight_accum"):
                if self._packed_weight_accum:
                    logger.warning(
                        f"{self.label}: Incomplete packed weight accumulators: "
                        f"{list(self._packed_weight_accum.keys())}"
                    )
                self._packed_weight_accum.clear()
        logger.info(
            f"{self.label}: IPC weight update bucket done - "
            f"updated={updated}, skipped={skipped}, "
            f"ignored_scales={ignored_scales}, is_last={is_last}"
        )
        return updated
