# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""Fused-MoE weight sync between a transformers-5.x trainer and vLLM.

transformers 5.x stores Qwen3-MoE experts as fused 3D tensors, so the training
side's ``state_dict()`` -- which is exactly what the IPC weight sync sends --
emits::

    model.layers.N.mlp.experts.gate_up_proj    (E, 2*I, H)
    model.layers.N.mlp.experts.down_proj       (E, H, I)

vLLM calls the same buffers ``experts.w13_weight`` / ``experts.w2_weight`` and
its ``expert_params_mapping`` only recognises the *per-expert* checkpoint names
``experts.{id}.{gate,up,down}_proj.weight``. A fused name therefore matches no
mapping and falls through ``Qwen3MoeForCausalLM.load_weights``'s silent
``if is_expert_weight: continue`` branch: no exception, no loaded param, and
93% of the policy's parameters never reach the rollout engine.

vLLM >=0.22 adds a wrinkle: the expert buffers moved into a nested
``RoutedExperts`` submodule, so their path gains a ``routed_experts`` segment
while the trainer keeps sending the shorter HF name. See ``_CONTAINER_SEGMENTS``.

This module closes that gap on the receiving side. ``FusedMoEWeightRouter``
intercepts the fused tensors and feeds them to vLLM's own
``FusedMoE.weight_loader`` through its 3D "full load" path (one call per
shard instead of 128), so TP sharding, hidden-dim padding and any load-time
kernel-format handling stay vLLM's business. The layouts are element-wise
identical -- verified against Qwen3-30B-A3B-Base by loading the same checkpoint
through HF and through vLLM and comparing ``gate_up_proj`` with ``w13_weight``
bit for bit on layers 0/1/23/47.

``assert_weight_sync_coverage`` is the second half: the original bug survived 54
training steps only because ``load_weights()``'s return value was discarded, so
every sync now has to account for every parameter in the rollout model.
"""

from __future__ import annotations

import logging
import os
import re
from collections.abc import Iterable, Sequence

import torch

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("ATOMRL_LOGGING_LEVEL", "WARN"))

# transformers-5.x fused expert parameter names -> vLLM shard ids. gate_up_proj
# carries two logical shards (w1 = gate, w3 = up) concatenated along dim 1, in
# that order, matching HF's `linear(x, gate_up_proj[e]).chunk(2, dim=-1)`.
_FUSED_GATE_UP = "gate_up_proj"
_FUSED_DOWN = "down_proj"
_FUSED_LEAVES = (_FUSED_GATE_UP, _FUSED_DOWN)

# Expert stacks from the Megatron-native trainer: experts start, start+step, ...
# of one layer as ``[n, out, in]``, named ``<experts>.{start}::{step}.<leaf>``
# (``bridges.core.stacked_expert_rules``). Same per-expert layout as the fused
# names above, but a strided subset of the experts.
_STACKED = re.compile(r"^(.*\.experts)\.(\d+)::(\d+)\.(gate_up_proj|down_proj)$")


def _stacked_expert_shards(
    tensor: torch.Tensor,
    leaf: str,
    act_and_mul: bool = True,
) -> list[tuple[str, torch.Tensor]]:
    """Split one expert of a stack into vLLM shard ids (w1 = gate, w3 = up, w2 = down)."""
    if leaf == _FUSED_DOWN:
        return [("w2", tensor)]
    if not act_and_mul:
        return [("w1", tensor)]
    half = tensor.shape[0] // 2
    return [("w1", tensor[:half]), ("w3", tensor[half:])]


def unstack_expert_weights(
    weights: Iterable[tuple[str, torch.Tensor]],
) -> list[tuple[str, torch.Tensor]]:
    """Expand expert stacks into per-expert checkpoint names (views, no copies).

    For receive paths that hand tensors to ``model.load_weights`` without the
    router, e.g. online FP8, whose layerwise reload tracks experts one by one.
    """
    out: list[tuple[str, torch.Tensor]] = []
    for name, tensor in weights:
        m = _STACKED.match(name)
        if m is None:
            out.append((name, tensor))
            continue
        prefix, start, step, leaf = (
            m.group(1),
            int(m.group(2)),
            int(m.group(3)),
            m.group(4),
        )
        names = {"w1": "gate_proj", "w3": "up_proj", "w2": "down_proj"}
        for k in range(tensor.shape[0]):
            p = f"{prefix}.{start + k * step}."
            for shard_id, shard in _stacked_expert_shards(tensor[k], leaf):
                out.append((f"{p}{names[shard_id]}.weight", shard))
    return out


# vLLM submodules that hold the expert buffers *below* the layer everyone else
# addresses: vLLM >=0.22 splits FusedMoE so the weights live in a nested
# RoutedExperts registered as "routed_experts", making the parameter path
# ``...mlp.experts.routed_experts.w13_weight`` while both the trainer and vLLM's
# own loader talk about ``...mlp.experts.<leaf>``.
#
# That one extra segment breaks two different things, so it is consulted twice:
#   _lookup_keys      routing -- a discovered module must also be reachable
#                     under the shortened name the trainer addresses it by
#   _coverage_aliases accounting -- a parameter reported loaded under the
#                     shortened name must not be counted as untouched
# Skipping either one produces the same message, "left 96/435 rollout parameters
# untouched" (Qwen3-30B-A3B, 48 layers x 2), which is why fixing only the first
# did not make it go away.
_CONTAINER_SEGMENTS = ("routed_experts",)

# Parameters that a BF16 trainer legitimately never sends: quantization
# artifacts that vLLM derives locally in process_weights_after_loading.
_COVERAGE_IGNORE_SUFFIXES = (
    "_scale",
    "_scale_inv",
    "_scale_2",
    "_offset",
    "_zero_point",
    "_shape",
    "g_idx",
)


def _split_name(name: str) -> tuple[str, str]:
    prefix, _, leaf = name.rpartition(".")
    return prefix, leaf


def _coverage_aliases(name: str) -> tuple[str, ...]:
    """Every name a loader might report back for this parameter.

    The same nesting asymmetry ``_lookup_keys`` handles on the routing side, seen
    from the other end. ``RoutedExperts.load_weights`` yields names relative to
    the *FusedMoE* layer (``w13_weight``, because it strips ``layer_name``), and
    ``AutoWeightsLoader`` prefixes them with the module it walked to -- the
    FusedMoE. So a loaded expert buffer is reported as
    ``...mlp.experts.w13_weight`` while ``named_parameters()`` calls it
    ``...mlp.experts.routed_experts.w13_weight``.

    Differencing the two sets literally therefore flags every fused expert
    parameter as untouched. Observed as "left 96/435 rollout parameters
    untouched" on Qwen3-30B-A3B (48 layers x 2) while the weights were
    demonstrably live -- kl stayed at 1.6e-3 and accuracy doubled across the
    sync (dsv4_agent.md 3.9).
    """
    aliases = [name]
    for segment in _CONTAINER_SEGMENTS:
        needle = f".{segment}."
        if needle in name:
            aliases.append(name.replace(needle, ".", 1))
    return tuple(aliases)


class FusedMoEWeightRouter:
    """Load transformers-style fused expert tensors into vLLM FusedMoE params.

    Instantiate once per weight-sync round (discovery walks ``named_modules``)
    and call :meth:`route` on every bucket. When the rollout model has no
    FusedMoE layers -- the dense 8B path -- :attr:`active` is False and
    :meth:`route` degenerates to returning its input.
    """

    def __init__(self, model: torch.nn.Module):
        # experts module prefix -> (module, w13 param name, w2 param name)
        self._experts: dict[str, tuple[torch.nn.Module, str, str]] = {}
        self._params = dict(model.named_parameters())
        self._verify = os.environ.get("ATOMRL_WEIGHT_SYNC_VERIFY", "0") == "1"

        for mod_name, module in model.named_modules():
            if not callable(getattr(module, "weight_loader", None)):
                continue
            owned = {n for n, _ in module.named_parameters()}
            w13 = next((n for n in owned if n.endswith("w13_weight")), None)
            w2 = next((n for n in owned if n.endswith("w2_weight")), None)
            if w13 is None or w2 is None:
                continue
            entry = (module, f"{mod_name}.{w13}", f"{mod_name}.{w2}")
            for key in self._lookup_keys(mod_name):
                # First writer wins: the exact module name is registered before any
                # shortened alias, so a real FusedMoE can never be shadowed by one.
                self._experts.setdefault(key, entry)

        if self._experts:
            logger.info(
                "fused-MoE weight router: %d expert modules (e.g. %s)",
                len(self._experts),
                next(iter(self._experts)),
            )

    @staticmethod
    def _lookup_keys(mod_name: str) -> list[str]:
        """The module's own name plus the prefix the trainer addresses it by."""
        keys = [mod_name]
        prefix, _, leaf = mod_name.rpartition(".")
        if prefix and leaf in _CONTAINER_SEGMENTS:
            keys.append(prefix)
        return keys

    @property
    def active(self) -> bool:
        return bool(self._experts)

    def route(
        self, weights: Sequence[tuple[str, torch.Tensor]]
    ) -> tuple[list[tuple[str, torch.Tensor]], set[str]]:
        """Split a bucket into (weights for ``model.load_weights``, names loaded here)."""
        if not self._experts:
            return list(weights), set()

        passthrough: list[tuple[str, torch.Tensor]] = []
        loaded: set[str] = set()
        for name, tensor in weights:
            stacked = _STACKED.match(name)
            if stacked is not None:
                loaded.add(self._load_stacked(name, stacked, tensor))
                continue
            prefix, leaf = _split_name(name)
            entry = self._experts.get(prefix)
            if entry is None or leaf not in _FUSED_LEAVES:
                passthrough.append((name, tensor))
                continue
            loaded.add(self._load_fused(entry, name, leaf, tensor))
        return passthrough, loaded

    def _load_fused(
        self,
        entry: tuple[torch.nn.Module, str, str],
        name: str,
        leaf: str,
        tensor: torch.Tensor,
    ) -> str:
        module, w13_name, w2_name = entry
        if tensor.ndim != 3:
            raise RuntimeError(
                f"fused MoE weight {name} must be 3D (experts, out, in), got "
                f"{tuple(tensor.shape)}. The training side is not emitting the "
                "transformers-5.x fused layout this router expects."
            )

        param_name = w13_name if leaf == _FUSED_GATE_UP else w2_name
        param = self._params[param_name]
        if leaf == _FUSED_DOWN:
            shards = [("w2", tensor)]
        elif self._is_act_and_mul(module):
            half = tensor.shape[1] // 2
            shards = [("w1", tensor[:, :half]), ("w3", tensor[:, half:])]
        else:
            shards = [("w1", tensor)]

        for shard_id, shard in shards:
            self._dispatch(module, param, param_name, shard_id, shard)
        if self._verify:
            self._verify_written(module, param, param_name, shards)
        return param_name

    def _load_stacked(self, name: str, match: re.Match, tensor: torch.Tensor) -> str:
        """Load a strided expert stack: one copy when the layer is unsharded.

        Unsharded and unpadded, the stack is exactly ``param[start::step]``, so
        it bypasses vLLM's per-expert loader. Otherwise every expert goes
        through ``weight_loader`` with its global id, which handles TP slicing,
        the EP expert map and hidden-dim padding.
        """
        prefix, start, step, leaf = (
            match.group(1),
            int(match.group(2)),
            int(match.group(3)),
            match.group(4),
        )
        entry = self._experts.get(prefix)
        if entry is None:
            raise RuntimeError(
                f"expert stack {name} names no FusedMoE layer of the rollout model"
            )
        if tensor.ndim != 3:
            raise RuntimeError(
                f"expert stack {name} must be 3D (experts, out, in), got "
                f"{tuple(tensor.shape)}"
            )
        module, w13_name, w2_name = entry
        param_name = w13_name if leaf == _FUSED_GATE_UP else w2_name
        param = self._params[param_name]

        # Basic slicing, so a strided view of param: copy_ writes through it in
        # place. Advanced indexing (param.data[idx]) would copy and drop the write.
        dest = param.data[start::step]
        if (
            self._ep_size(module) == 1
            and self._tp_size(module) == 1
            and dest.shape == tensor.shape
            and dest.dtype == tensor.dtype
        ):
            dest.copy_(tensor)
            if self._verify and not torch.equal(dest, tensor):
                raise RuntimeError(
                    f"weight sync verify failed for {param_name} stack {start}::{step}"
                )
            return param_name

        act_and_mul = self._is_act_and_mul(module)
        for k in range(tensor.shape[0]):
            for shard_id, shard in _stacked_expert_shards(tensor[k], leaf, act_and_mul):
                module.weight_loader(
                    param,
                    shard,
                    param_name,
                    shard_id=shard_id,
                    expert_id=start + k * step,
                    return_success=True,
                )
        return param_name

    def _verify_written(
        self,
        module: torch.nn.Module,
        param: torch.nn.Parameter,
        param_name: str,
        shards: list[tuple[str, torch.Tensor]],
    ) -> None:
        """Read back what vLLM stored and require it to equal what was sent.

        Under tensor parallelism the destination holds one slice of the
        intermediate dim, so compare against the same slice the loader takes:
        ``per_rank * tp_rank`` along the sharded dim (w1/w3 shard dim 0, w2 dim 1,
        both shifted by one for the leading expert dim). Expert parallelism is
        still skipped -- there the parameter holds a subset of experts and the
        global-to-local expert map lives inside vLLM.

        Any shape disagreement downgrades to a skip rather than a failure, so a
        mistake here can only cost coverage, never invent a false alarm.
        """
        if self._ep_size(module) != 1:
            logger.warning(
                "weight sync verify skipped for %s: expert-parallel layer", param_name
            )
            return
        tp_size = self._tp_size(module)
        tp_rank = self._tp_rank(module)
        offset = 0
        for shard_id, shard in shards:
            if tp_size > 1:
                dim = 2 if shard_id == "w2" else 1
                per_rank = shard.shape[dim] // tp_size
                if per_rank == 0 or per_rank * tp_size != shard.shape[dim]:
                    logger.warning(
                        "weight sync verify skipped for %s shard %s: dim %d of %s "
                        "does not divide by tp_size=%d",
                        param_name,
                        shard_id,
                        dim,
                        tuple(shard.shape),
                        tp_size,
                    )
                    return
                shard = shard.narrow(dim, per_rank * tp_rank, per_rank)
            dest = param.data.narrow(1, offset, shard.shape[1])
            offset += shard.shape[1]
            if dest.shape != shard.shape:
                logger.warning(
                    "weight sync verify skipped for %s shard %s: padded destination %s",
                    param_name,
                    shard_id,
                    tuple(dest.shape),
                )
                continue
            if not torch.equal(dest, shard):
                raise RuntimeError(
                    f"weight sync verify failed for {param_name} shard {shard_id}: "
                    "vLLM's buffer does not match the tensor the trainer sent"
                )

    def _dispatch(
        self,
        module: torch.nn.Module,
        param: torch.nn.Parameter,
        param_name: str,
        shard_id: str,
        shard: torch.Tensor,
    ) -> None:
        """Hand one logical shard to vLLM's loader, whole or expert by expert.

        A 3D ``loaded_weight`` puts ``FusedMoE.weight_loader`` on its ``full_load``
        branch, which writes all experts in one ``copy_``. Two things make that
        branch wrong unless the layer is entirely unsharded:

        * under expert parallelism the parameter only holds this rank's experts,
          so global expert ids have to be mapped to local ones;
        * under tensor parallelism the parameter only holds this rank's slice of
          the intermediate dim, and ``_load_w13`` / ``_load_w2`` guard their
          ``tp_rank`` narrowing with ``if not load_full`` -- a full 3D tensor is
          taken to be pre-sharded and is copied as is.

        So for either kind of sharding, send 2D per-expert tensors and let the
        loader do the narrowing it already knows how to do.
        """
        if self._ep_size(module) == 1 and self._tp_size(module) == 1:
            ok = module.weight_loader(
                param,
                shard,
                param_name,
                shard_id=shard_id,
                expert_id=0,
                return_success=True,
            )
            if ok is False:
                raise RuntimeError(
                    f"vLLM refused the fused {shard_id} shard of {param_name}"
                )
            return

        loaded_any = False
        for expert_id in range(shard.shape[0]):
            loaded_any |= bool(
                module.weight_loader(
                    param,
                    shard[expert_id],
                    param_name,
                    shard_id=shard_id,
                    expert_id=expert_id,
                    return_success=True,
                )
            )
        if not loaded_any:
            raise RuntimeError(
                f"no expert of the {shard_id} shard of {param_name} was accepted "
                "by this rank; the parallel map and the sent tensor disagree"
            )

    @staticmethod
    def _is_act_and_mul(module: torch.nn.Module) -> bool:
        moe_config = getattr(module, "moe_config", None)
        return bool(getattr(moe_config, "is_act_and_mul", True))

    @staticmethod
    def _ep_size(module: torch.nn.Module) -> int:
        return FusedMoEWeightRouter._parallel_size(module, "ep_size")

    @staticmethod
    def _tp_size(module: torch.nn.Module) -> int:
        return FusedMoEWeightRouter._parallel_size(module, "tp_size")

    @staticmethod
    def _tp_rank(module: torch.nn.Module) -> int:
        """This rank's index inside the layer's TP group.

        vLLM reads it as ``moe_config.tp_rank`` while the sizes live on
        ``moe_config.moe_parallel_config``, so check both holders.
        """
        moe_config = getattr(module, "moe_config", None)
        parallel = getattr(moe_config, "moe_parallel_config", None)
        for holder in (parallel, moe_config):
            value = getattr(holder, "tp_rank", None)
            if value is not None:
                return int(value)
        return 0

    @staticmethod
    def _parallel_size(module: torch.nn.Module, attr: str) -> int:
        moe_config = getattr(module, "moe_config", None)
        parallel = getattr(moe_config, "moe_parallel_config", None)
        return int(getattr(parallel, attr, 1) or 1)


def assert_weight_sync_coverage(
    model: torch.nn.Module,
    loaded: Iterable[str],
    *,
    context: str = "ipc",
    default_mode: str = "error",
) -> None:
    """Fail loudly when a weight sync left parameters at their previous values.

    vLLM's ``load_weights`` skips names it does not recognise without raising,
    so an unnoticed naming change silently freezes part of the rollout policy
    while training continues -- the failure mode is a slowly growing
    train/rollout divergence, not a crash. Comparing the loaded set against
    ``named_parameters`` turns that into an immediate error.

    ``ATOMRL_WEIGHT_SYNC_CHECK`` selects ``error``, ``warn`` or ``off``, and
    overrides ``default_mode``. Callers whose loaded-name set is not known to be
    directly comparable to ``named_parameters`` pass ``default_mode="warn"``.
    """
    mode = os.environ.get("ATOMRL_WEIGHT_SYNC_CHECK", default_mode).lower()
    if mode == "off":
        return

    loaded = set(loaded)
    expected = {
        name
        for name in dict(model.named_parameters())
        if not name.endswith(_COVERAGE_IGNORE_SUFFIXES)
    }
    missing = sorted(
        name for name in expected if loaded.isdisjoint(_coverage_aliases(name))
    )
    if not missing:
        logger.info(
            "weight sync coverage (%s): %d/%d params",
            context,
            len(expected),
            len(expected),
        )
        return

    sample = ", ".join(missing[:8])
    message = (
        f"weight sync ({context}) left {len(missing)}/{len(expected)} rollout "
        f"parameters untouched: {sample}"
        f"{' ...' if len(missing) > 8 else ''}. The rollout engine is now serving "
        "a mix of current and stale weights. Set ATOMRL_WEIGHT_SYNC_CHECK=warn "
        "to downgrade this to a log line."
    )
    if mode == "warn":
        logger.warning(message)
        return
    raise RuntimeError(message)
