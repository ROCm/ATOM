# SPDX-License-Identifier: Apache-2.0
"""MoonEP expert balancing in front of MegaMoEGfx1250 (the ATOM_MORI_V2 fused
transport on gfx1250).

The gfx1250 counterpart of ``moonep_mega_experts``.  Each rank keeps its
``EPR`` resident experts plus ``B`` prefetch slots in one P2P-readable
``[EPR + B]`` window per weight, carved from MoonEP's own cco communicator.
A step any DP rank prefills is planned: the group exchanges its logical route
histogram, places hot experts' routes on other ranks, rewrites each route to
the virtual id of its replica and pulls the placed experts' weights into the
slots.  The shared MegaMoEGfx1250 is built over ``R * (EPR + B)`` virtual
experts and runs it unchanged.  A unified decode step keeps every route on
its owner's resident copy.
"""

import logging

import torch

from atom.config import get_current_atom_config
from atom.utils import envs
from atom.utils.forward_context import get_forward_context

logger = logging.getLogger("atom")

_WEIGHTS = ("w13_weight", "w13_weight_scale", "w2_weight", "w2_weight_scale")
# One communicator and one planner per EP layout serve every layer.
_COMMUNICATORS: dict = {}
_PLANNERS: dict = {}
# Balanced layer forwards so far; logged at powers of two to show the path runs.
_BALANCED = [0]


def moonep_virtual_experts(num_experts: int, ep_size: int) -> int:
    """Experts the shared MegaMoE is built over: ``R * (EPR + B)``."""
    return ep_size * (num_experts // ep_size + envs.MOONEP_PREFETCH_SLOTS)


def check_moonep_geometry(*, ep_rank: int, ep_size: int, num_experts: int) -> None:
    config = get_current_atom_config()
    if getattr(config, "eplb_enable", False) or getattr(config, "fake_eplb", False):
        raise ValueError("ATOM_ENABLE_MOONEP=1 cannot be combined with (fake) EPLB")
    if ep_size not in (4, 8):
        raise ValueError(f"MoonEP supports EP4 and EP8, got EP{ep_size}")
    if num_experts % ep_size:
        raise ValueError("MoonEP requires num_experts divisible by the EP size")
    slots = envs.MOONEP_PREFETCH_SLOTS
    if not 0 < slots <= 64:
        raise ValueError(f"MOONEP_PREFETCH_SLOTS must be in [1, 64], got {slots}")
    if torch.cuda.current_device() != ep_rank:
        raise ValueError(
            f"MoonEP needs EP rank {ep_rank} on cuda:{ep_rank}, "
            f"running on cuda:{torch.cuda.current_device()}"
        )


def moonep_balance_step() -> bool:
    """Balance a step that prefills and is big enough to pay for its prefetch.

    Every EP rank must take the same branch (planning is collective), so both
    tests are DP-agreed.  Prefill is the test ``combine_quant_for_step`` uses:
    ``is_prefill`` only speaks for dp_size == 1, ``running_tokens_are_unified``
    for the group.  Size is the group's token sum: a slot fill costs the same
    whatever the step, so a step where one rank prefills a little next to
    decoding peers is left on the home copies.  No context (warmup, profile
    runs) balances, like a prefill.
    """
    context = get_forward_context().context
    if context is None:
        return True
    if context.is_prefill or not context.running_tokens_are_unified:
        across_dp = context.running_tokens_across_dp
        tokens = sum(across_dp) if across_dp else context.running_tokens
        return tokens >= envs.MOONEP_MIN_BALANCE_TOKENS
    return False


def _expert_rows(t: torch.Tensor, epr: int) -> tuple:
    """Per-expert shape of ``t``; scales may be stored flat as ``[epr * rows, ...]``."""
    if t.shape[0] == epr:
        return tuple(t.shape[1:])
    if t.shape[0] % epr:
        raise ValueError(
            f"cannot index {tuple(t.shape)} by expert: leading dim is neither "
            f"{epr} nor a multiple of it"
        )
    return (t.shape[0] // epr, *t.shape[1:])


def _communicator(*, ep_rank, ep_size, ep_src_global_rank, per_layer_bytes):
    """MoonEP's own cco communicator, sized for one pool per model layer."""
    from atom.model_ops.fused_moe.mori_v2_prepare_finalize import _init_cco_comm

    key = (ep_rank, ep_size)
    comm = _COMMUNICATORS.get(key)
    if comm is None:
        hf = get_current_atom_config().hf_config
        layers = int(getattr(hf, "num_hidden_layers", 0)) + int(
            getattr(hf, "num_nextn_predict_layers", 0) or 0
        )
        # Every allocation rounds up to the VMM granularity; 64 MiB per pool
        # part covers it, plus 1 GiB for the planner and slack.
        per_rank_vmm = layers * (per_layer_bytes + len(_WEIGHTS) * (64 << 20)) + (
            1 << 30
        )
        comm = _COMMUNICATORS[key] = _init_cco_comm(
            ep_size, ep_rank, ep_src_global_rank, per_rank_vmm
        )
    return comm


class MoonEPLayer:
    """One MoE layer's ``[home | prefetch]`` windows and slot state."""

    def __init__(
        self,
        layer: torch.nn.Module,
        *,
        ep_rank: int,
        ep_size: int,
        ep_src_global_rank: int,
        num_experts: int,
        topk: int,
        max_tokens_per_rank: int,
    ) -> None:
        """Move ``layer``'s expert weights into a symmetric pool.

        Collective: every EP rank builds its layers in the same order.  The
        layer's parameters become views of the pool's home rows, so the
        weights are not held twice.
        """
        from aiter.dist.parallel_state import get_ep_group
        from aiter.ops.flydsl.kernels.moonep_plan import MoonEPPlanner
        from aiter.ops.flydsl.kernels.moonep_weights import MoonEPWeightPool

        check_moonep_geometry(ep_rank=ep_rank, ep_size=ep_size, num_experts=num_experts)
        if getattr(layer, "w13_bias", None) is not None or (
            getattr(layer, "w2_bias", None) is not None
        ):
            raise NotImplementedError("MoonEP does not move expert biases")
        epr = num_experts // ep_size
        slots = envs.MOONEP_PREFETCH_SLOTS
        group = get_ep_group().cpu_group
        self._rank = ep_rank

        params = [getattr(layer, name) for name in _WEIGHTS]
        resident = [p.data for p in params]
        parts = [(_expert_rows(t, epr), t.dtype) for t in resident]
        per_layer_bytes = sum(
            (epr + slots) * (t.numel() // epr) * t.element_size() for t in resident
        )
        comm = _communicator(
            ep_rank=ep_rank,
            ep_size=ep_size,
            ep_src_global_rank=ep_src_global_rank,
            per_layer_bytes=per_layer_bytes,
        )
        self._pool = MoonEPWeightPool(
            rank=ep_rank,
            world_size=ep_size,
            experts_per_rank=epr,
            prefetch_slots=slots,
            parts=parts,
            communicator=comm,
            group=group,
        )
        self._pool.stage_home(
            [t.view(epr, *rows) for t, (rows, _) in zip(resident, parts)]
        )
        self.windows = []
        for param, t, home, window in zip(
            params, resident, self._pool.home, self._pool.local
        ):
            param.data = home.reshape(t.shape)
            if window.dim() != t.dim():
                window = window.reshape(-1, *t.shape[1:])
            if getattr(param, "is_shuffled", False):
                window.is_shuffled = True
            self.windows.append(window)
        del resident

        key = (ep_rank, ep_size, num_experts, slots, max_tokens_per_rank * topk)
        planner = _PLANNERS.get(key)
        if planner is None:
            planner = _PLANNERS[key] = MoonEPPlanner(
                rank=ep_rank,
                world_size=ep_size,
                experts=num_experts,
                slots=slots,
                max_routes=max_tokens_per_rank * topk,
                communicator=comm,
                group=group,
            )
        self._planner = planner
        self._state = planner.new_slot_state()

    def route(self, ids: torch.Tensor, *, balance: bool) -> torch.Tensor:
        """Virtual ids for ``ids`` (int32, -1 = no route).

        Balancing plans the whole group's routes and fills this rank's slots
        first; otherwise every route stays on its owner's resident copy.
        """
        if not balance:
            return self._planner.home_ids(ids)
        _BALANCED[0] += 1
        count = _BALANCED[0]
        if self._rank == 0 and count & (count - 1) == 0:
            logger.info("MoonEP balanced %d layer forwards", count)
        virtual = self._planner.plan(ids, self._state)
        # Slots already holding their expert are skipped.
        self._pool.prefetch(
            self._state["placed"][self._rank], self._state["prev"][self._rank]
        )
        return virtual


__all__ = [
    "MoonEPLayer",
    "check_moonep_geometry",
    "moonep_balance_step",
    "moonep_virtual_experts",
]
