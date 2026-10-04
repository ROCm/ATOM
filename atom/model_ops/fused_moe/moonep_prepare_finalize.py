# SPDX-License-Identifier: Apache-2.0
"""MoonEP planning in front of the production MoRI transport.

Both phases dispatch virtual ids through one MoRI handle configured with
``EPR + B`` local experts, using the EPLB contract::

    physical_id = destination_rank * (EPR + B) + destination_slot

with slots ``[0, EPR)`` resident and ``[EPR, EPR + B)`` prefetched.  Prefill
plans which remote experts each rank prefetches; decode never plans and sends
every expert to its owner's resident slot.  Each rank then runs one fused_moe
over its ``[EPR + B]`` weight window.
"""

import logging
from typing import Any

import torch
import triton
import triton.language as tl
from aiter import QuantType

import atom.model_ops.fused_moe.modular_kernel as mk
from atom.model_ops.fused_moe.config import FusedMoEQuantConfig
from atom.model_ops.fused_moe.mori_prepare_finalize import MoriPrepareAndFinalize
from atom.utils.forward_context import get_forward_context

logger = logging.getLogger("atom")

# Prefill planners and the histogram exchange are shared by every MoE layer:
# layers run one after another on one stream, so a plan is consumed before the
# next layer's planner overwrites it.
_PREFILL_POLICIES: dict[tuple, Any] = {}
_HISTOGRAM_EXCHANGES: dict[tuple, Any] = {}


def _make_virtual_expert_mask(
    *,
    rank: int,
    world_size: int,
    experts_per_rank: int,
    prefetch_slots: int,
    device: torch.device,
) -> torch.Tensor:
    """Return the mask mapping this rank's virtual ids onto its ``[EPR+B]`` window."""

    physical_per_rank = experts_per_rank + prefetch_slots
    mask = torch.zeros(
        world_size * physical_per_rank, dtype=torch.int32, device=device
    )
    begin = rank * physical_per_rank
    mask[begin : begin + physical_per_rank] = 1
    return mask


@triton.jit
def _owner_virtual_ids_kernel(
    ids_ptr,
    out_ptr,
    numel,
    EXPERTS_PER_RANK: tl.constexpr,
    PREFETCH_SLOTS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < numel
    ids = tl.load(ids_ptr + offs, mask=mask, other=-1).to(tl.int32)
    virtual = ids + (ids // EXPERTS_PER_RANK) * PREFETCH_SLOTS
    tl.store(out_ptr + offs, tl.where(ids >= 0, virtual, ids), mask=mask)


def _owner_virtual_ids(
    topk_ids: torch.Tensor, experts_per_rank: int, prefetch_slots: int
) -> torch.Tensor:
    """Map expert ``e`` to its owner's resident slot ``owner * (EPR + B) + e % EPR``.

    Negative (invalid) ids pass through unchanged.
    """

    if not topk_ids.is_cuda:
        ids = topk_ids.to(torch.int32)
        virtual = ids + (ids // experts_per_rank) * prefetch_slots
        return torch.where(ids >= 0, virtual, ids)
    ids = topk_ids.contiguous()
    out = torch.empty(ids.shape, dtype=torch.int32, device=ids.device)
    numel = ids.numel()
    block = 1024
    _owner_virtual_ids_kernel[(triton.cdiv(numel, block),)](
        ids,
        out,
        numel,
        EXPERTS_PER_RANK=experts_per_rank,
        PREFETCH_SLOTS=prefetch_slots,
        BLOCK=block,
    )
    return out


class MoonEPPrepareAndFinalize(mk.FusedMoEPrepareAndFinalize):
    """Route MoRI dispatch/combine through MoonEP's virtual expert ids.

    ``transport`` is an ordinary ``MoriPrepareAndFinalize`` configured with
    ``EPR + B`` local experts; this class hands it the virtual ids for
    dispatch and again for combine.
    """

    ADOPTED = (
        "w13_weight",
        "w2_weight",
        "w13_weight_scale",
        "w2_weight_scale",
        "w13_bias",
        "w2_bias",
    )

    def __init__(
        self,
        *,
        transport: MoriPrepareAndFinalize,
        rank: int,
        world_size: int,
        num_experts: int,
        prefetch_slots: int,
    ) -> None:
        super().__init__()
        if num_experts % world_size:
            raise ValueError("MoonEP requires num_experts divisible by world_size")
        if prefetch_slots <= 0:
            raise ValueError("MoonEP prefetch_slots must be positive")

        self._transport = transport
        self._rank = rank
        self._world_size = world_size
        self._num_experts = num_experts
        self._experts_per_rank = num_experts // world_size
        self._prefetch_slots = prefetch_slots
        self._pools = None
        self._virtual_masks: dict[str, torch.Tensor] = {}
        self._active_ids = None
        self._active_plan = None

    def output_is_reduced(self) -> bool:
        return True

    def num_dispatchers(self) -> int:
        return self._transport.num_dispatchers()

    def max_num_tokens_per_rank(self) -> int | None:
        return self._transport.max_num_tokens_per_rank()

    def topk_indices_dtype(self) -> torch.dtype | None:
        return torch.int32

    def _is_prefill(self) -> bool:
        """Decode only when every EP rank decodes, else plan as prefill.

        ``running_tokens_are_unified`` is agreed across the DP group, so every
        rank picks the same MoRI geometry and the prefill histogram exchange is
        entered by all of them or by none.
        """

        context = get_forward_context().context
        if context is None:
            return True
        return context.is_prefill or not context.running_tokens_are_unified

    def _prefill_policy(self, topk_ids: torch.Tensor):
        from aiter.ops.flydsl.moonep import (
            MoonEPPlanConfig,
            MoonEPPrefillPolicy,
            MoonEPSymmetricHistogramExchange,
        )

        # Planners compile per capacity, so bucket the row count; every layer
        # shares the bucket's planner and the one histogram exchange.
        bucket = max(64, 1 << max(0, topk_ids.shape[0] - 1).bit_length())
        geometry = (
            self._rank,
            self._world_size,
            self._num_experts,
            topk_ids.shape[1],
            self._prefetch_slots,
            str(topk_ids.device),
        )
        key = (*geometry, bucket)
        policy = _PREFILL_POLICIES.get(key)
        if policy is not None:
            return policy

        exchange = _HISTOGRAM_EXCHANGES.get(geometry)
        if exchange is None:
            exchange = MoonEPSymmetricHistogramExchange(
                rank=self._rank,
                world_size=self._world_size,
                num_experts=self._num_experts,
                device=topk_ids.device,
            )
            _HISTOGRAM_EXCHANGES[geometry] = exchange
        config = MoonEPPlanConfig(
            rank=self._rank,
            world_size=self._world_size,
            num_tokens=bucket,
            top_k=topk_ids.shape[1],
            num_experts=self._num_experts,
            prefetch_slots=self._prefetch_slots,
        )
        policy = MoonEPPrefillPolicy(
            config,
            topk_ids.device,
            histogram_exchange=exchange,
        )
        _PREFILL_POLICIES[key] = policy
        return policy

    def prepare(
        self,
        a1: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        num_experts: int,
        expert_map: torch.Tensor | None,
        apply_router_weight_on_input: bool,
        quant_config: FusedMoEQuantConfig,
        quant_type: QuantType = QuantType.No,
    ) -> mk.PrepareResultType:
        if self._active_ids is not None:
            raise RuntimeError("prepare() called before the previous finalize()")
        if num_experts != self._num_experts:
            raise ValueError(
                f"routing width changed from {self._num_experts} to {num_experts}"
            )

        if self._is_prefill():
            self._active_plan = self._prefill_policy(topk_ids).plan(topk_ids)
            self._active_ids = self._active_plan.planned_topk_ids
        else:
            # Every expert on its owner's resident slot, with no histogram or
            # planning; one launch per layer on the decode critical path.
            self._active_ids = _owner_virtual_ids(
                topk_ids, self._experts_per_rank, self._prefetch_slots
            )
        return self._transport.prepare(
            a1,
            topk_weights,
            self._active_ids,
            num_experts,
            expert_map,
            apply_router_weight_on_input,
            quant_config,
            quant_type,
        )

    def finalize(
        self,
        output: torch.Tensor,
        fused_expert_output: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        apply_router_weight_on_input: bool,
    ) -> torch.Tensor:
        if self._active_ids is None:
            raise RuntimeError("finalize() called before prepare()")
        planned_ids = self._active_ids
        self._active_ids = self._active_plan = None
        # MoRI v1 rebuilds the combine routing from the ids it dispatched.
        return self._transport.finalize(
            output,
            fused_expert_output,
            topk_weights,
            planned_ids,
            apply_router_weight_on_input,
        )

    def adopt_weights(self, layer: torch.nn.Module) -> None:
        """Move resident weights to P2P-readable storage and add B cache rows."""

        if self._pools is not None:
            return
        pools = []
        for name in self.ADOPTED:
            param = getattr(layer, name, None)
            entry = (None, False, False)
            if param is not None and param.data is not None:
                pool, flat = self._pool_for(param.data)
                shuffled = bool(getattr(param, "is_shuffled", False))
                param.data = pool.home.reshape(param.data.shape)
                entry = (pool, flat, shuffled)
            pools.append(entry)
        self._pools = tuple(pools)

    def _pool_for(self, tensor: torch.Tensor):
        from aiter.ops.flydsl.kernels.moonep_weights import MoonEPWeightPool

        epn = self._experts_per_rank
        flat = tensor.shape[0] != epn
        if flat:
            if tensor.shape[0] % epn:
                raise ValueError(
                    f"cannot index {tuple(tensor.shape)} by expert: leading dim "
                    f"is neither {epn} nor a multiple of it"
                )
            view = tensor.reshape(epn, tensor.shape[0] // epn, *tensor.shape[1:])
        else:
            view = tensor
        pool = MoonEPWeightPool(
            rank=self._rank,
            world_size=self._world_size,
            experts_per_rank=epn,
            prefetch_slots=self._prefetch_slots,
            weight_shape=tuple(view.shape[1:]),
            dtype=tensor.dtype,
        )
        pool.stage_home(view.contiguous())
        logger.info(
            "MoonEP adopted %s%s: %d resident + %d prefetch slots",
            tuple(tensor.shape),
            tensor.dtype,
            epn,
            self._prefetch_slots,
        )
        return pool, flat

    @staticmethod
    def _pool_view(entry):
        pool, flat, shuffled = entry
        if pool is None:
            return None
        tensor = pool.local
        if flat:
            tensor = tensor.reshape(-1, *tensor.shape[2:])
        if shuffled:
            tensor.is_shuffled = True
        return tensor

    def _mask(self, device: torch.device) -> torch.Tensor:
        key = str(device)
        mask = self._virtual_masks.get(key)
        if mask is None:
            mask = _make_virtual_expert_mask(
                rank=self._rank,
                world_size=self._world_size,
                experts_per_rank=self._experts_per_rank,
                prefetch_slots=self._prefetch_slots,
                device=device,
            )
            self._virtual_masks[key] = mask
        return mask

    def run_dispatched_experts(
        self,
        rows: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        *,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        expert_mask: torch.Tensor | None,
        num_local_tokens: torch.Tensor | None,
        activation=None,
        quant_type=None,
        w1_scale: torch.Tensor | None = None,
        w2_scale: torch.Tensor | None = None,
        a1_scale: torch.Tensor | None = None,
        a2_scale: torch.Tensor | None = None,
        hidden_pad: int = 0,
        intermediate_pad: int = 0,
        bias1: torch.Tensor | None = None,
        bias2: torch.Tensor | None = None,
        dtype=None,
        extra_kwargs: dict | None = None,
    ) -> torch.Tensor:
        from aiter import ActivationType
        from aiter.fused_moe import fused_moe

        if self._pools is None:
            raise RuntimeError("adopt_weights() must run before the first MoE call")
        if w1.data_ptr() != self._pools[0][0].home.data_ptr():
            raise RuntimeError("the layer weights no longer alias the MoonEP pool")
        if self._active_ids is None:
            raise RuntimeError("experts called before prepare()")

        # Decode ids never name a prefetch slot, so only prefill refreshes them.
        if self._active_plan is not None:
            selected = self._active_plan.experts_to_copy[self._rank].contiguous()
            for pool, _flat, _shuffled in self._pools:
                if pool is not None:
                    pool.prefetch(selected)

        return fused_moe(
            rows,
            self._pool_view(self._pools[0]),
            self._pool_view(self._pools[1]),
            topk_weights,
            topk_ids,
            self._mask(rows.device),
            activation=activation if activation is not None else ActivationType.Silu,
            quant_type=quant_type if quant_type is not None else QuantType.No,
            num_local_tokens=num_local_tokens,
            w1_scale=self._pool_view(self._pools[2]),
            w2_scale=self._pool_view(self._pools[3]),
            a1_scale=a1_scale,
            a2_scale=a2_scale,
            hidden_pad=hidden_pad,
            intermediate_pad=intermediate_pad,
            bias1=(
                self._pool_view(self._pools[4])
                if self._pools[4][0] is not None
                else bias1
            ),
            bias2=(
                self._pool_view(self._pools[5])
                if self._pools[5][0] is not None
                else bias2
            ),
            dtype=dtype if dtype is not None else rows.dtype,
            **(extra_kwargs or {}),
        )
