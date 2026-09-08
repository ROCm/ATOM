# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Native MoE transport implemented with RCCL.

Uniform decode uses a graph-safe pre-routed all-gather/reduce-scatter path over
ATOM's PyNccl communicator. Prefill and mixed prefill/decode batches use the
same pre-routed payload with variable-size all-gather/reduce-scatter whenever
the EP and DP groups have the same width. Other layouts retain the
variable-split routed ``all_to_all_single`` correctness path.
"""

from dataclasses import dataclass
from typing import Any

import torch
from aiter import QuantType

import atom.model_ops.fused_moe.modular_kernel as mk
from atom.model_ops.fused_moe.config import FusedMoEQuantConfig
from atom.model_ops.fused_moe.routed_all2all import (
    build_routed_dispatch_plan,
    combine_routed_rows,
    pack_routed_payload,
    unpack_routed_payload,
)
from atom.utils.forward_context import get_forward_context


@dataclass
class _PendingCombine:
    token_indices: torch.Tensor
    send_splits: list[int]
    recv_splits: list[int]
    num_tokens: int
    recv_rows: int


@dataclass
class _PendingGatherCombine:
    num_tokens: int
    sizes: list[int] | None = None


class RcclPrepareAndFinalize(mk.FusedMoEPrepareAndFinalize):
    """Synchronous RCCL dispatch/combine for expert-parallel MoE.

    On ROCm, PyTorch's NCCL process group is backed by RCCL.  The class name
    describes that production target while keeping the implementation usable
    with any backend that implements ``all_to_all_single`` for tests.

    Uniform decode avoids dynamic split sizes and remains graph-capturable.
    Matching DP/EP groups reuse scheduler-owned split sizes for variable gather
    and scatter. Other layouts use routed all-to-all and require a device-to-host
    synchronization. TBO is unsupported. The backend is opt-in and never
    replaces MoRI or the existing gather/scatter path implicitly.
    """

    def __init__(
        self,
        ep_group: Any,
        *,
        num_local_experts: int,
        max_tokens_per_rank: int,
        num_replicated_shared_experts: int = 0,
        num_routed_experts_per_rank: int | None = None,
    ) -> None:
        super().__init__()
        self._group = ep_group.device_group
        self._device_communicator = ep_group.device_communicator
        self._rank = int(ep_group.rank_in_group)
        self._world_size = int(ep_group.world_size)
        self._num_local_experts = int(num_local_experts)
        self._max_tokens_per_rank = int(max_tokens_per_rank)
        self._num_replicated_shared_experts = int(num_replicated_shared_experts)
        self._num_routed_experts_per_rank = int(
            num_routed_experts_per_rank
            if num_routed_experts_per_rank is not None
            else num_local_experts - num_replicated_shared_experts
        )
        self._pending: _PendingCombine | _PendingGatherCombine | None = None
        device_communicator = getattr(ep_group, "device_communicator", None)
        pynccl = getattr(device_communicator, "pynccl_comm", None)
        self._pynccl = (
            pynccl
            if pynccl is not None and not getattr(pynccl, "disabled", True)
            else None
        )

        if self._world_size <= 1:
            raise ValueError("RCCL routed MoE requires an EP group larger than one")
        if self._num_local_experts <= 0:
            raise ValueError("num_local_experts must be positive")
        if not (0 <= self._num_replicated_shared_experts <= self._num_local_experts):
            raise ValueError(
                "num_replicated_shared_experts must be between zero and "
                "num_local_experts"
            )
        if (
            self._num_routed_experts_per_rank + self._num_replicated_shared_experts
            != self._num_local_experts
        ):
            raise ValueError(
                "local routed and replicated shared expert counts must add up "
                "to num_local_experts"
            )

    @property
    def activation_format(self) -> mk.FusedMoEActivationFormat:
        return mk.FusedMoEActivationFormat.Standard

    def output_is_reduced(self) -> bool:
        return True

    def needs_dispatch_output_trim(self) -> bool:
        # All RCCL paths allocate their result to the exact receive size. The
        # zero-receive routed path uses one explicit dummy row that finalize
        # already drops; neither case has a MoRI-style inactive arena tail.
        return False

    def num_dispatchers(self) -> int:
        return self._world_size

    def max_num_tokens_per_rank(self) -> int | None:
        return self._max_tokens_per_rank

    def topk_indices_dtype(self) -> torch.dtype | None:
        return torch.int32

    @staticmethod
    def _reject_graph_capture(tensor: torch.Tensor) -> None:
        is_capturing = getattr(torch.cuda, "is_current_stream_capturing", None)
        if tensor.is_cuda and is_capturing is not None and is_capturing():
            raise RuntimeError(
                "the ATOM RCCL routed MoE backend uses dynamic all-to-all split "
                "sizes and cannot run inside a CUDA/HIP graph; pass --enforce-eager"
            )

    def _exchange_counts(self, send_counts: torch.Tensor) -> torch.Tensor:
        recv_counts = torch.empty_like(send_counts)
        torch.distributed.all_to_all_single(
            recv_counts,
            send_counts.contiguous(),
            group=self._group,
        )
        return recv_counts

    def _exchange_rows(
        self,
        tensor: torch.Tensor,
        *,
        send_splits: list[int],
        recv_splits: list[int],
    ) -> torch.Tensor:
        output = tensor.new_empty((sum(recv_splits), *tensor.shape[1:]))
        torch.distributed.all_to_all_single(
            output,
            tensor.contiguous(),
            output_split_sizes=recv_splits,
            input_split_sizes=send_splits,
            group=self._group,
        )
        return output

    def _use_static_decode_path(self) -> bool:
        if self._pynccl is None:
            return False
        context = get_forward_context().context
        return bool(
            context is not None
            and not context.is_prefill
            and context.running_tokens_are_unified
        )

    def _variable_gather_sizes(self, a1: torch.Tensor) -> list[int] | None:
        """Return DP token counts when EP can reuse the DP gather/scatter plan.

        DPA8+EP8 uses the same ordered ranks for DP token sharding and EP expert
        sharding. The scheduler has already exchanged every rank's token count,
        so reusing that metadata avoids both the routed-plan GPU work and the
        device-to-host count synchronization required by dynamic all-to-all.

        A flattened DP*TP EP group can be wider than the DP metadata. Keep the
        routed all-to-all fallback for that topology rather than guessing how
        token shards should be replicated across TP peers.
        """
        forward_context = get_forward_context()
        context = forward_context.context
        dp_metadata = getattr(forward_context, "dp_metadata", None)
        if context is None or context.running_tokens_are_unified or dp_metadata is None:
            return None

        sizes = [int(size) for size in dp_metadata.get_sizes_across_dp()]
        if len(sizes) != self._world_size:
            return None
        if sizes[self._rank] != a1.shape[0]:
            raise ValueError(
                "RCCL variable gather token count disagrees with the local "
                f"input: sizes[{self._rank}]={sizes[self._rank]}, "
                f"rows={a1.shape[0]}"
            )
        return sizes

    def _balance_replicated_shared_experts(self, dispatch_ids: torch.Tensor) -> None:
        """Round-robin gathered rows over the replicated shared experts.

        ``LOCAL_REPLICA`` routing initially pins a token's shared expert to its
        source rank, which is ideal for routed transports but creates a severe
        hotspot when one DPA rank owns a much longer Agentic request. After an
        all-gather every rank has the same row order, so the row index provides
        a deterministic, communication-free owner that balances shared-expert
        GEMMs while still computing each shared contribution exactly once.
        """
        num_shared = self._num_replicated_shared_experts
        if num_shared == 0 or dispatch_ids.shape[0] == 0:
            return
        if dispatch_ids.shape[1] < num_shared:
            raise ValueError(
                "top-k width is smaller than the replicated shared expert count"
            )

        rows = torch.arange(
            dispatch_ids.shape[0],
            dtype=dispatch_ids.dtype,
            device=dispatch_ids.device,
        )
        owner_bases = (rows % self._world_size) * self._num_local_experts
        owner_bases += self._num_routed_experts_per_rank
        shared_offsets = torch.arange(
            num_shared,
            dtype=dispatch_ids.dtype,
            device=dispatch_ids.device,
        )
        dispatch_ids[:, -num_shared:].copy_(
            owner_bases[:, None] + shared_offsets[None, :]
        )

    def _prepare_static_decode(
        self,
        a1: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> mk.PrepareResultType:
        """Gather pre-routed rows without host-visible split metadata.

        Uniform decode pads every DP/EP rank to the same row count. Gathering
        the compact top-k result instead of router logits avoids both the
        dynamic count exchange and redundant top-k calculation on peer ranks.
        """
        local_payload, payload_layout = pack_routed_payload(
            a1,
            topk_ids,
            topk_weights,
        )
        dispatch_payload = local_payload.new_empty(
            (self._world_size * a1.shape[0], payload_layout.row_bytes)
        )
        # ncclGroupStart/End only batches launch submission; it does not turn
        # collectives over three native dtypes into one collective. In the full
        # 61-layer decode graph, three grouped AllGathers caused severe channel
        # scheduling/launch amplification despite looking faster in an isolated
        # microbenchmark. One packed byte collective is consistently faster at
        # AgentX c96 and also keeps collective ordering simpler during replay.
        self._pynccl.all_gather(dispatch_payload, local_payload)
        dispatch_a1, dispatch_ids, dispatch_weights = unpack_routed_payload(
            dispatch_payload,
            payload_layout,
        )
        self._balance_replicated_shared_experts(dispatch_ids)
        self._pending = _PendingGatherCombine(num_tokens=a1.shape[0])
        return (
            dispatch_a1,
            None,
            mk.ExpertTokensMetadata(
                expert_num_tokens=None,
                expert_num_tokens_cpu=None,
            ),
            dispatch_ids,
            dispatch_weights,
        )

    def _prepare_variable_gather(
        self,
        a1: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        sizes: list[int],
    ) -> mk.PrepareResultType:
        """Gather variable token shards after routing, without a route plan.

        Routing before the gather is important for two reasons: it avoids
        gathering the full router-logit matrix and preserves source-rank IDs
        for locally replicated shared experts. Each rank then evaluates only
        the experts selected by its ``expert_mask`` and reduce-scatter sums the
        partial results back to the source token shard.
        """
        local_payload, payload_layout = pack_routed_payload(
            a1,
            topk_ids,
            topk_weights,
        )
        dispatch_payload = self._device_communicator.all_gatherv(
            local_payload,
            dim=0,
            sizes=sizes,
        )
        dispatch_a1, dispatch_ids, dispatch_weights = unpack_routed_payload(
            dispatch_payload,
            payload_layout,
        )
        self._balance_replicated_shared_experts(dispatch_ids)
        self._pending = _PendingGatherCombine(
            num_tokens=a1.shape[0],
            sizes=sizes,
        )
        return (
            dispatch_a1,
            None,
            mk.ExpertTokensMetadata(
                expert_num_tokens=None,
                expert_num_tokens_cpu=None,
            ),
            dispatch_ids,
            dispatch_weights,
        )

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
        del expert_map, quant_config, quant_type
        if apply_router_weight_on_input:
            raise NotImplementedError(
                "RCCL routed MoE does not support apply_router_weight_on_input"
            )
        if self._pending is not None:
            raise RuntimeError("prepare called twice without a matching finalize")
        if num_experts != self._world_size * self._num_local_experts:
            raise ValueError(
                "RCCL routed MoE requires an evenly sharded dispatch space: "
                f"num_experts={num_experts}, world_size={self._world_size}, "
                f"num_local_experts={self._num_local_experts}"
            )

        if self._use_static_decode_path():
            return self._prepare_static_decode(a1, topk_weights, topk_ids)

        variable_sizes = self._variable_gather_sizes(a1)
        if variable_sizes is not None:
            return self._prepare_variable_gather(
                a1,
                topk_weights,
                topk_ids.to(torch.int32),
                variable_sizes,
            )

        self._reject_graph_capture(a1)
        topk_ids = topk_ids.to(torch.int32)
        plan = build_routed_dispatch_plan(
            a1,
            topk_ids,
            topk_weights,
            num_local_experts=self._num_local_experts,
            world_size=self._world_size,
        )

        recv_counts = self._exchange_counts(plan.send_counts)
        # PyTorch's variable-split API accepts host lists.  This synchronization
        # is the main known performance gap of the correctness backend.
        send_splits = [int(v) for v in plan.send_counts.cpu().tolist()]
        recv_splits = [int(v) for v in recv_counts.cpu().tolist()]
        recv_rows = sum(recv_splits)

        send_payload, payload_layout = pack_routed_payload(
            plan.hidden_states,
            plan.topk_ids,
            plan.topk_weights,
        )
        recv_payload = self._exchange_rows(
            send_payload,
            send_splits=send_splits,
            recv_splits=recv_splits,
        )
        dispatch_a1, dispatch_ids, dispatch_weights = unpack_routed_payload(
            recv_payload,
            payload_layout,
        )

        self._pending = _PendingCombine(
            token_indices=plan.token_indices,
            send_splits=send_splits,
            recv_splits=recv_splits,
            num_tokens=a1.shape[0],
            recv_rows=recv_rows,
        )

        if recv_rows == 0:
            # Keep the downstream fused-MoE launch shape valid. AITER's EP
            # sorter indexes expert_mask before filtering, so use an in-range
            # local expert ID instead of a -1 sentinel. Zero weights make the
            # dummy row's contribution zero, and finalize drops it.
            dispatch_a1 = a1.new_zeros((1, a1.shape[1]))
            dispatch_ids = topk_ids.new_full(
                (1, topk_ids.shape[1]),
                self._rank * self._num_local_experts,
            )
            dispatch_weights = topk_weights.new_zeros((1, topk_weights.shape[1]))

        # Unlike MoRI v1, RCCL returns an exactly-sized receive tensor rather
        # than a fixed-capacity arena with a live prefix. Passing recv_rows as
        # num_local_tokens makes some AITER A8W4/EP kernels take the arena path
        # and can corrupt memory when M is already the exact row count. Let
        # fused_moe derive M directly from dispatch_ids instead.
        return (
            dispatch_a1,
            None,
            mk.ExpertTokensMetadata(
                expert_num_tokens=None,
                expert_num_tokens_cpu=None,
            ),
            dispatch_ids,
            dispatch_weights,
        )

    def finalize(
        self,
        output: torch.Tensor | None,
        fused_expert_output: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        apply_router_weight_on_input: bool,
    ) -> torch.Tensor:
        del topk_weights, topk_ids
        if apply_router_weight_on_input:
            raise NotImplementedError(
                "RCCL routed MoE does not support apply_router_weight_on_input"
            )
        pending = self._pending
        if pending is None:
            raise RuntimeError("finalize called without a matching prepare")
        self._pending = None

        if isinstance(pending, _PendingGatherCombine):
            expected_rows = (
                sum(pending.sizes)
                if pending.sizes is not None
                else pending.num_tokens * self._world_size
            )
            if fused_expert_output.shape[0] != expected_rows:
                raise ValueError(
                    "gathered RCCL output has an unexpected row count: "
                    f"got {fused_expert_output.shape[0]}, expected {expected_rows}"
                )
            if pending.sizes is not None:
                result = self._device_communicator.reduce_scatterv(
                    fused_expert_output,
                    dim=0,
                    sizes=pending.sizes,
                )
                if output is not None:
                    output.copy_(result)
                    return output
                return result
            result = (
                output
                if output is not None
                else fused_expert_output.new_empty(
                    (pending.num_tokens, *fused_expert_output.shape[1:])
                )
            )
            self._pynccl.reduce_scatter(result, fused_expert_output.contiguous())
            return result

        returned_rows = self._exchange_rows(
            fused_expert_output[: pending.recv_rows],
            send_splits=pending.recv_splits,
            recv_splits=pending.send_splits,
        )
        return combine_routed_rows(
            returned_rows,
            pending.token_indices,
            pending.num_tokens,
            output,
        )
