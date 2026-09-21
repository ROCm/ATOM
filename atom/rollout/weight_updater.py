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


def _unwrap_once(module) -> torch.nn.Module | None:
    """The model inside one wrapper `ModelRunner` may have rebound onto itself,
    or None when *module* is not one. See `_sync_target_model`.

    `_orig_mod` is `torch.compile`'s own name for what it wrapped, private
    enough that a model is not going to have one of its own. The import is
    deferred to keep this module importable without the TBO package, as the
    tests that drive the updater on a stand-in rely on.
    """
    compiled_inner = getattr(module, "_orig_mod", None)
    if isinstance(compiled_inner, torch.nn.Module):
        return compiled_inner
    from atom.utils.tbo.ubatch_wrapper import UBatchWrapper

    if isinstance(module, UBatchWrapper):
        return module.model
    return None


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

    def _sync_target_model(self) -> torch.nn.Module:
        """The model the trainer's parameter names are relative to.

        Not always ``self.model``. ``ModelRunner`` rebinds that twice after the
        model is built: to a ``UBatchWrapper`` under TBO, and to
        ``torch.compile(...)`` at compilation level 1. Both are ``nn.Module``s
        holding the real model as a CHILD, so ``named_modules()`` on either
        prefixes every parameter with the wrapper's own attribute name --
        ``model.`` or ``_orig_mod.``. Not one name the trainer sends then
        matches anything, every weight is counted `skipped` at debug level, and
        the whole update is the silent no-op this path exists to remove --
        under two supported configurations.

        Both wrappers do forward plain attribute lookups to what they wrap, so
        ``get_expert_mapping`` and ``packed_modules_mapping`` were reachable
        through them. They are asked of this model anyway, so that all four
        lookups describe one module rather than relying on each wrapper to
        keep forwarding.

        Peeled by TYPE, not by attribute name: nearly every HF-derived model has
        a submodule literally called ``model``, and peeling that would drop a
        prefix the trainer does send.
        """
        model = self.model
        while True:
            inner = _unwrap_once(model)
            if inner is None:
                return model
            model = inner

    def _warn_if_nothing_matched(self, updated: int, skipped: int) -> None:
        """A sync that matched nothing is the failure this whole path is about.

        It stays reachable however many name conventions are covered -- a wrapper
        rebound onto `self.model` was one, a trainer whose names come from
        somewhere new is the next -- and `updated=0, skipped=N` on an info line
        reads exactly like a bucket that legitimately held nothing. Said once, at
        the level someone reads, and phrased as what it costs: the rollout goes
        on serving the weights it already had.
        """
        if updated == 0 and skipped > 0:
            logger.warning(
                f"{self.label}: weight update matched NOTHING -- {skipped} "
                f"parameter(s) resolved to no module, so nothing was written and "
                f"the rollout is still serving the weights it had. Compare the "
                f"sender's names against the model's own "
                f"(`model.named_parameters()`)."
            )

    def _get_param_to_module_mapping(self) -> dict[str, tuple]:
        """
        Get or build the parameter name to module mapping.

        This mapping is cached after the first call to avoid expensive
        rebuilding on every weight update.

        Returns:
            Dict mapping parameter full name to (module, param_name, param) tuple
        """
        model = self._sync_target_model()
        # Keyed on the model it was built from, so a rebind of `self.model`
        # rebuilds instead of serving names from the module that is no longer
        # there. A `hasattr` cache cannot be invalidated by anything, and there
        # is no hook here to invalidate it from.
        if (
            getattr(self, "_param_to_module", None) is None
            or getattr(self, "_param_to_module_of", None) is not model
        ):
            self._param_to_module = {}
            for module_name, module in model.named_modules():
                for param_name, param in module.named_parameters(recurse=False):
                    full_name = (
                        f"{module_name}.{param_name}" if module_name else param_name
                    )
                    self._param_to_module[full_name] = (module, param_name, param)
            self._param_to_module_of = model
            logger.debug(
                f"{self.label}: Built param_to_module mapping with "
                f"{len(self._param_to_module)} parameters"
            )
        return self._param_to_module

    def _get_packed_modules_mapping(self) -> dict:
        model = self._sync_target_model()
        if getattr(self, "_cached_packed_mapping_of", None) is not model:
            self._cached_packed_mapping = (
                getattr(model, "packed_modules_mapping", None) or {}
            )
            self._cached_packed_mapping_of = model
        return self._cached_packed_mapping

    def _get_packed_shard_order(self) -> dict[str, list]:
        """Build {target_suffix: [shard_id_0, shard_id_1, ...]} preserving declaration order."""
        packed = self._get_packed_modules_mapping()
        # Derived from that mapping, so it is stale exactly when that is: keyed
        # on the same object rather than on its own first call.
        if getattr(self, "_cached_packed_shard_order_of", None) is not packed:
            order: dict[str, list] = {}
            for tgt, shard_id in packed.values():
                order.setdefault(tgt, []).append(shard_id)
            self._cached_packed_shard_order = order
            self._cached_packed_shard_order_of = packed
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
        self._load_into_param(param, weight_loader, tensor_gpu, shard_id)
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
        more specific one wins. Built once per model; models with no MoE layer
        leave it empty and every lookup then short-circuits.

        Asked of the unwrapped model, like every other lookup here: an answer of
        "no mapping" is indistinguishable from a model with no MoE, so it is not
        a question to leave depending on a wrapper's attribute forwarding.
        """
        model = self._sync_target_model()
        if getattr(self, "_cached_expert_mapping_of", None) is not model:
            get_expert_mapping = getattr(model, "get_expert_mapping", None)
            entries = [
                (weight_name_part, param_name_part, expert_id, shard_id)
                for param_name_part, weight_name_part, expert_id, shard_id in (
                    get_expert_mapping() if callable(get_expert_mapping) else ()
                )
            ]
            entries.sort(key=lambda entry: len(entry[0]), reverse=True)
            self._cached_expert_mapping = entries
            self._cached_expert_mapping_of = model
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
            self._load_into_param(
                param,
                weight_loader,
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
        expert dimension is written whole.

        Rank-local halves only. ``FusedMoE._load_w13`` and ``_load_w2`` narrow
        the *destination* under ``load_full``; they skip the branch that slices
        the source by ``tp_rank``, so under TP the incoming halves have to be
        this rank's shard already, not the global intermediate dimension.

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
            self._load_into_param(
                param,
                weight_loader,
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

    def _apply_named_expert_buffer(
        self,
        name: str,
        param_name: str,
        module: torch.nn.Module,
        param: torch.nn.Parameter,
        tensor: torch.Tensor,
    ) -> None:
        """Write a whole fused expert buffer sent under ATOM's own name.

        Checks support, writes through the fence, and registers every expert
        slice for the relayout -- the same three obligations as the per-expert
        route, reached by a different name.

        ``w13_weight`` and ``w2_weight`` are real parameters of the FusedMoE,
        so a trainer that mirrors ATOM's state dict rather than the
        checkpoint's resolves in ``_get_param_to_module_mapping`` and never
        reaches ``_apply_expert_weight``. Without this function such a tensor
        took the plain dispatch: a row-major ``copy_`` into a buffer the kernel
        reads through aiter's 16x16 expert permutation, with
        ``_pending_expert_relayout`` left empty so
        ``_finalize_expert_weight_sync`` returned at ``if not pending``,
        ``updated`` counting it as a success, and ``_check_expert_sync_supported``
        never running to refuse a quantized or expert-parallel MoE.

        Rank-local buffers only. A TP rollout's parameter is its own shard, so
        a full global buffer is refused below rather than written at the wrong
        width; ``FusedMoE``'s full-load path does not slice the source by
        ``tp_rank`` either.
        """
        self._check_expert_sync_supported(name, name, param_name, module, param, tensor)
        if param.dim() != 3:
            raise NotImplementedError(
                f"{self.label}: {name} carries an expert-buffer name but is "
                f"{param.dim()}D, not the (experts, out, in) buffer the expert "
                f"relayout works on."
            )
        if tensor.shape != param.shape:
            raise NotImplementedError(
                f"{self.label}: {name} resolves to the fused expert buffer "
                f"{tuple(param.shape)} but arrived as {tuple(tensor.shape)}. "
                f"Re-establishing the layout works on whole expert slices, so "
                f"a partial write cannot be relaid out. Send this rank's "
                f"buffer whole -- under TP{self.world_size} that is its own "
                f"shard, not the global tensor -- or one tensor per expert."
            )
        self._copy_into_param(param, tensor.to(device=self.device, dtype=param.dtype))
        # One tensor covers every expert, so every slice of the buffer is new.
        arrived = self._pending_expert_relayout.setdefault((module, param_name), {})
        for expert_id in range(param.shape[0]):
            arrived.setdefault(expert_id, set()).update(
                _EXPERT_BUFFER_SHARDS[param_name]
            )

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
        buffers = 0
        settled = []
        half_delivered = []
        try:
            for (module, param_name), arrived in pending.items():
                required = _EXPERT_BUFFER_SHARDS[param_name]
                # An expert whose shards all arrived can have its layout
                # re-established; one missing a shard cannot, because the
                # permutation is per whole slice and half of this one is still
                # in the old layout. Per expert rather than per buffer, so one
                # bad slice does not cost the buffer's good ones their layout.
                complete = []
                for expert_id, shards in sorted(arrived.items()):
                    if required <= shards:
                        complete.append(expert_id)
                    else:
                        half_delivered.append(
                            f"{param_name}[{expert_id}] arrived without "
                            f"{sorted(required - shards)}"
                        )
                if complete:
                    buffer = getattr(module, param_name)
                    # The relayout is the second in-place write this sync makes
                    # to these slices, and the one the graph is most likely to
                    # catch mid-flight: a half-permuted expert reads as
                    # plausible garbage.
                    self._await_readers_of(buffer)
                    shuffle_expert_slices(buffer, complete)
                    experts += len(complete)
                    buffers += 1
                settled.append((module, param_name))
        finally:
            # Drop what this call settled, and only that. Settled covers both
            # answers: a slice that was relaid out must not be shuffled again
            # -- per this function's own docstring that produces a third layout
            # rather than undoing the first -- and a half-delivered one must not
            # be kept either. Keeping it was worse than useless: nothing later
            # completes an entry (a sync sends every shard of an expert or the
            # write is refused), so it would fail this function again on every
            # sync for the life of the process, taking every other buffer's
            # relayout down with it. The bytes are recoverable without the
            # entry -- the next update that sends the whole expert overwrites
            # both halves row-major, and that one relays out normally.
            #
            # An entry the loop never reached is the opposite case and has to
            # survive: that buffer is row-major right now, and a later sync
            # re-establishing its layout is the only thing that fixes it.
            for key in settled:
                del pending[key]
            if pending:
                logger.error(
                    f"{self.label}: expert layout NOT re-established for "
                    f"{len(pending)} fused buffer(s) "
                    f"{sorted(param_name for _, param_name in pending)}; they "
                    f"hold row-major bytes the kernel reads through the expert "
                    f"permutation until a later sync finishes the job"
                )
        logger.info(
            f"{self.label}: expert layout re-established for {experts} expert "
            f"slices across {buffers} fused buffers"
        )
        if half_delivered:
            # After the cleanup, not instead of it, and once. Still loud: the
            # slices named here are half new and half old, in two layouts, and
            # the kernel reads them as plausible garbage rather than failing.
            raise RuntimeError(
                f"{self.label}: {len(half_delivered)} expert slice(s) were "
                f"rewritten shard by shard and cannot have their layout "
                f"re-established: {half_delivered}. Each is half new and half "
                f"old, in two layouts. Send every shard of an expert in the "
                f"same weight update; the next update that sends a whole expert "
                f"repairs it."
            )

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
        self._copy_into_param(param, sharded_tensor)

        return True

    def _await_readers_of(self, param: torch.nn.Parameter) -> None:
        """Let work still reading this weight finish before it is overwritten.

        A weight update rewrites the parameter buffer in place, and the FP8
        path does it twice: once for the quantized bytes, again for the
        kernel's shuffled layout. In place is deliberate -- rebinding
        ``tensor.data`` moves the address out from under a captured decode
        graph, which then replays against the old buffer and returns nothing
        but punctuation. Keeping the address is what costs us: the write lands
        in a buffer that is still live.

        An update follows generation immediately, so the last decode replays of
        the step that just ended can still be in flight. Overlap one with the
        shuffle and the graph reads a half-permuted weight; generation carries
        on and every sequence past that point is token soup, with nothing
        raised anywhere.

        Waiting once per update at the entry points is not enough -- measured
        over seven weight syncs it still lost five of them. The wait has to sit
        with the write, which is why every writer goes through
        ``_copy_into_param`` or ``_load_into_param`` rather than calling this.
        """
        device = getattr(param, "device", None)
        if getattr(device, "type", None) == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize(device)

    def _copy_into_param(self, param: torch.nn.Parameter, tensor: torch.Tensor) -> None:
        """Overwrite a live parameter's buffer, after its readers are done.

        The wait and the write belong together. Kept apart, the FP8 path ended
        up fencing the second of its two writes and not the first, and the
        bf16, TP-sharded, loader-fallback and expert paths fenced none of
        theirs -- two of six writes covered, which is indistinguishable from
        uncovered given the failure is probabilistic per sync.
        """
        self._await_readers_of(param)
        param.data.copy_(tensor)

    def _load_into_param(
        self,
        param: torch.nn.Parameter,
        weight_loader,
        tensor: torch.Tensor,
        *args,
        **kwargs,
    ) -> None:
        """Same contract for the writes a module's own ``weight_loader`` makes.

        A loader narrows and copies into whatever buffer it is handed, so its
        write is as in-place as ours. This is the only fence the MoE expert
        path can have: ``_check_expert_sync_supported`` requires an unquantized
        MoE, so the experts never reach ``_post_process_fp8_weight``.
        """
        self._await_readers_of(param)
        weight_loader(param, tensor, *args, **kwargs)

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
            # The scale writes ride on the weight's fence: `_await_readers_of`
            # synchronizes the device, so the reader that held the weight held
            # its scale too and both are done by the time this returns.
            self._copy_into_param(param, q_weight)
            weight_scale.data.copy_(scale.to(weight_scale.dtype))

        elif quant_type is not None and quant_type.value == _QT.per_Tensor.value:
            amax = tensor_gpu.abs().max()
            scale = (amax / fp8_max).clamp(min=1e-12)
            self._copy_into_param(param, (tensor_gpu / scale).to(fp8_dtype))
            weight_scale.data.fill_(scale.item())

        elif quant_type is not None and quant_type.value == _QT.per_Token.value:
            row_amax = tensor_gpu.abs().amax(dim=-1, keepdim=True)
            scale = (row_amax / fp8_max).clamp(min=1e-12)
            self._copy_into_param(param, (tensor_gpu / scale).to(fp8_dtype))
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

        # `need_normalize_e4m3fn_to_e4m3fnuz` is a static property of the
        # layer -- `params_dtype == torch.float8_e4m3fnuz`, set once in
        # `create_weights` -- not a to-do list, and nothing clears it after the
        # load-time conversion. Re-running that conversion on an
        # already-converted parameter is not idempotent in either buffer:
        #
        #   * `weight_scale` is rebuilt as `scale * 2.0`, so it doubles again
        #     on every sync (3.0 -> 6.0 -> 12.0, measured) and the dequantized
        #     weight comes out 2**N too large after N syncs. It is also a fresh
        #     allocation, which moves the scale's address out from under a
        #     captured decode graph -- the same hazard the `shuffle_weights`
        #     fix removed, on the buffer nobody checked.
        #   * the weight needs no conversion at all: `_requantize_fp8_weight`
        #     quantizes into `param.dtype` against `finfo(e4m3fnuz).max`, and
        #     the direct-copy path is handed bytes already in `param.dtype`, so
        #     both arrive in the target convention.
        #
        # Gate on the dtype, so this stays right for a load path that does
        # leave an e4m3fn parameter behind rather than just never firing.
        if (
            getattr(module, "need_normalize_e4m3fn_to_e4m3fnuz", False)
            and weight_scale is not None
            and param.dtype == torch.float8_e4m3fn
        ):
            from atom.model_ops.utils import normalize_e4m3fn_to_e4m3fnuz

            # Writes the NaN-byte fixup straight into the weight's storage.
            self._await_readers_of(param)
            normalized, normalized_scale, _ = normalize_e4m3fn_to_e4m3fnuz(
                param.data, weight_scale.data
            )
            # The weight's bytes are fixed through an int8 view of the same
            # storage, so `normalized` is the original buffer at the original
            # address with a reinterpreted dtype -- the rebind a captured graph
            # cannot see. The scale is a new tensor, so it goes back in place.
            param.data = normalized
            weight_scale.data.copy_(normalized_scale.to(weight_scale.dtype))

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
            # `shuffle_weights` permutes through the existing storage, so this
            # is the second in-place write to a weight the caller has just
            # overwritten -- and the one the PR measured a decode graph
            # catching half-done, five syncs out of seven.
            self._await_readers_of(param)
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

            if param_name in _EXPERT_BUFFER_SHARDS:
                self._apply_named_expert_buffer(name, param_name, module, param, tensor)
                updated += 1
            elif self._is_fp8_param(module, param) and tensor.dtype != param.dtype:
                self._requantize_fp8_weight(module, param_name, param, tensor)
                updated += 1
            elif self._is_fp8_param(module, param) and tensor.dtype == param.dtype:
                tensor = tensor.to(device=self.device)
                self._copy_into_param(param, tensor)
                self._post_process_fp8_weight(module, param)
                updated += 1
            elif tensor.shape == param.shape:
                tensor = tensor.to(device=self.device, dtype=param.dtype)
                self._copy_into_param(param, tensor)
                updated += 1
            elif weight_loader is not None and callable(weight_loader):
                try:
                    tensor = tensor.to(device=self.device)
                    self._load_into_param(param, weight_loader, tensor)
                    updated += 1
                except Exception as e:  # noqa: BLE001 - a loader raises anything
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
        self._warn_if_nothing_matched(updated, skipped)
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

                if param_name in _EXPERT_BUFFER_SHARDS:
                    self._apply_named_expert_buffer(
                        name, param_name, module, param, tensor
                    )
                    updated += 1
                elif self._is_fp8_param(module, param) and tensor.dtype != param.dtype:
                    self._requantize_fp8_weight(module, param_name, param, tensor)
                    updated += 1
                elif self._is_fp8_param(module, param) and tensor.dtype == param.dtype:
                    tensor = tensor.to(device=self.device)
                    self._copy_into_param(param, tensor)
                    self._post_process_fp8_weight(module, param)
                    updated += 1
                elif tensor.shape == param.shape:
                    tensor = tensor.to(device=self.device, dtype=param.dtype)
                    self._copy_into_param(param, tensor)
                    updated += 1
                elif weight_loader is not None and callable(weight_loader):
                    try:
                        tensor = tensor.to(device=self.device)
                        self._load_into_param(param, weight_loader, tensor)
                        updated += 1
                    except Exception as e:  # noqa: BLE001 - a loader raises anything
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
            self._warn_if_nothing_matched(updated, skipped)
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

            if param_name in _EXPERT_BUFFER_SHARDS:
                self._apply_named_expert_buffer(name, param_name, module, param, tensor)
                updated += 1
            elif self._is_fp8_param(module, param) and tensor.dtype != param.dtype:
                self._requantize_fp8_weight(module, param_name, param, tensor)
                updated += 1
            elif self._is_fp8_param(module, param) and tensor.dtype == param.dtype:
                self._copy_into_param(param, tensor)
                self._post_process_fp8_weight(module, param)
                updated += 1
            elif tensor.shape == param.shape:
                if tensor.dtype != param.dtype:
                    tensor = tensor.to(dtype=param.dtype)
                self._copy_into_param(param, tensor)
                updated += 1
            elif weight_loader is not None and callable(weight_loader):
                try:
                    self._load_into_param(param, weight_loader, tensor)
                    updated += 1
                except Exception as e:  # noqa: BLE001 - a loader raises anything
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
        self._warn_if_nothing_matched(updated, skipped)
        return updated
