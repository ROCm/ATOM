# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the ATOM project
"""
Mori EP Dispatcher implementation for DBO (Dynamic Batching Optimization).

This module provides a Mori-based expert parallel dispatcher that properly
handles the interleaved execution pattern in DBO mode using:
1. Separate dispatch_a/dispatch_b and combine_a/combine_b phases
2. Independent comm_stream for all-to-all communication
3. CUDA events for proper cross-stream synchronization
4. State machine to ensure correct operation ordering

Reference: SGLang's MoriEPDispatcher implementation
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum, auto
from functools import lru_cache
from typing import Any, NamedTuple, Optional

import torch

import atom.model_ops.fused_moe.modular_kernel as mk
from atom.model_ops.fused_moe.config import FusedMoEQuantConfig
from aiter import dtypes, QuantType
from atom.utils.dbo.ubatching import (
    dbo_current_ubatch_id,
    dbo_enabled,
    dbo_yield,
)

# Lazy import mori
try:
    import mori
    MORI_AVAILABLE = True
except ImportError:
    mori = None  # type: ignore
    MORI_AVAILABLE = False

logger = logging.getLogger(__name__)


class CommStreamPool:
    """
    Pool of communication streams, one per device/group combination.
    This ensures that all-to-all operations use a dedicated stream
    separate from the compute stream.
    """
    _streams: dict[tuple, torch.cuda.Stream] = {}

    @classmethod
    def _make_key(cls, group_id: int) -> tuple:
        return (torch.cuda.current_device(), group_id)

    @classmethod
    def get_stream(cls, group_id: int = 0) -> torch.cuda.Stream:
        key = cls._make_key(group_id)
        stream = cls._streams.get(key)
        if stream is None:
            # Create high-priority stream for communication
            stream = torch.cuda.Stream(priority=0)
            cls._streams[key] = stream
        return stream

    @classmethod
    def clear(cls, group_id: int = 0):
        key = cls._make_key(group_id)
        cls._streams.pop(key, None)


class _Stage(Enum):
    """State machine stages for dispatch/combine operations."""
    INITIAL = auto()
    AFTER_DISPATCH_A = auto()
    AFTER_DISPATCH_B = auto()
    AFTER_COMBINE_A = auto()


@dataclass
class DispatchIntermediateState:
    """Intermediate state between dispatch_a and dispatch_b."""
    hidden_states: torch.Tensor
    topk_weights: torch.Tensor
    topk_ids: torch.Tensor
    scale: torch.Tensor | None
    previous_event: torch.cuda.Event | None


@dataclass
class CombineIntermediateState:
    """Intermediate state between combine_a and combine_b."""
    fused_expert_output: torch.Tensor
    topk_ids: torch.Tensor
    output: torch.Tensor
    num_token: int
    previous_event: torch.cuda.Event | None


class DispatchResult(NamedTuple):
    """Result from dispatch operation."""
    hidden_states: torch.Tensor
    hidden_states_scale: torch.Tensor | None
    topk_ids: torch.Tensor
    topk_weights: torch.Tensor
    num_recv_tokens_per_expert: list[int]


class MoriEPPrepareAndFinalize(mk.FusedMoEPrepareAndFinalize):
    """
    Mori EP Prepare/Finalize implementation with proper DBO support.

    Unlike the basic MoriPrepareAndFinalize, this implementation:
    1. Splits dispatch/combine into a/b phases for proper interleaving
    2. Uses a dedicated comm_stream for all-to-all communication
    3. Uses CUDA events for cross-stream synchronization
    4. Maintains a state machine to ensure correct operation ordering

    This design ensures that even with DBO's interleaved execution pattern
    (dispatch0 -> dispatch1 -> combine0 -> combine1), each dispatch/combine
    pair is properly matched through event synchronization.
    """

    def __init__(
        self,
        mori_op: Any,  # mori.ops.EpDispatchCombineOp
        max_tokens_per_rank: int,
        num_dispatchers: int,
        use_fp8_dispatch: bool = False,
        quant_type=None,
        quant_dtype: torch.dtype = None,
        async_finish: bool = True,
    ):
        if not MORI_AVAILABLE:
            raise ImportError(
                "mori is required for MoriEPPrepareAndFinalize but not installed."
            )
        super().__init__()
        self.mori_op = mori_op
        self.num_dispatchers_ = num_dispatchers
        self.max_tokens_per_rank = max_tokens_per_rank
        self.use_fp8_dispatch = use_fp8_dispatch
        self.quant_type = quant_type
        self.quant_dtype = quant_dtype
        self.async_finish = async_finish

        # Communication stream for all-to-all operations
        self._comm_stream = CommStreamPool.get_stream()

        # Per-ubatch state tracking
        self._dispatch_states: list[DispatchIntermediateState | None] = [None, None]
        self._combine_states: list[CombineIntermediateState | None] = [None, None]
        self._stages: list[_Stage] = [_Stage.INITIAL, _Stage.INITIAL]

    @property
    def activation_format(self) -> mk.FusedMoEActivationFormat:
        return mk.FusedMoEActivationFormat.Standard

    def output_is_reduced(self) -> bool:
        return True

    def num_dispatchers(self):
        return self.num_dispatchers_

    def max_num_tokens_per_rank(self) -> int | None:
        return self.max_tokens_per_rank

    def topk_indices_dtype(self) -> torch.dtype | None:
        return torch.int32

    def supports_async(self) -> bool:
        return True

    def _capture_event_if_async(self) -> torch.cuda.Event | None:
        """Capture an event on current stream if async mode is enabled."""
        if not self.async_finish:
            return None
        event = torch.cuda.Event(blocking=False, interprocess=False)
        event.record(torch.cuda.current_stream())
        return event

    def _update_stage(self, ubatch_id: int, expected: _Stage, new: _Stage):
        """Update stage with validation."""
        if self._stages[ubatch_id] != expected:
            raise RuntimeError(
                f"Ubatch {ubatch_id}: expected stage {expected}, "
                f"got {self._stages[ubatch_id]}"
            )
        self._stages[ubatch_id] = new

    # ======================== Dispatch Phase A ========================
    def _dispatch_a(
        self,
        a1: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        scale: torch.Tensor | None,
    ) -> None:
        """
        Phase A of dispatch: capture event and store intermediate state.
        This phase runs on compute stream and prepares for the actual dispatch.
        """
        ubatch_id = dbo_current_ubatch_id()
        print(f"dispatch_a called for ubatch {ubatch_id}")
        self._update_stage(ubatch_id, _Stage.INITIAL, _Stage.AFTER_DISPATCH_A)

        # Capture event on compute stream to synchronize with comm stream later
        previous_event = self._capture_event_if_async()
        print(f"after dispatch_a ubatch {ubatch_id}")
        # Store intermediate state for dispatch_b
        self._dispatch_states[ubatch_id] = DispatchIntermediateState(
            hidden_states=a1,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            scale=scale,
            previous_event=previous_event,
        )

    # ======================== Dispatch Phase B ========================
    def _dispatch_b(self) -> DispatchResult:
        """
        Phase B of dispatch: execute actual mori dispatch on comm stream.
        Returns dispatch results after proper synchronization.
        """
        ubatch_id = dbo_current_ubatch_id()
        print(f"dispatch_b called for ubatch {ubatch_id}")
        self._update_stage(ubatch_id, _Stage.AFTER_DISPATCH_A, _Stage.AFTER_DISPATCH_B)

        state = self._dispatch_states[ubatch_id]
        if state is None:
            raise RuntimeError(f"dispatch_b called without dispatch_a for ubatch {ubatch_id}")
        self._dispatch_states[ubatch_id] = None

        compute_stream = torch.cuda.current_stream()
        comm_stream = self._comm_stream

        # Mark tensors as used by comm_stream
        for t in (state.hidden_states, state.topk_weights, state.topk_ids):
            t.record_stream(comm_stream)
        if state.scale is not None:
            state.scale.record_stream(comm_stream)

        with torch.cuda.stream(comm_stream):
            # Synchronize: comm_stream waits for compute_stream
            if state.previous_event is not None:
                comm_stream.wait_event(state.previous_event)
            else:
                comm_stream.wait_stream(compute_stream)

            # Execute mori dispatch
            (
                dispatch_hidden,
                dispatch_weights,
                dispatch_scale,
                dispatch_ids,
                dispatch_recv_count,
            ) = self.mori_op.dispatch(
                state.hidden_states,
                state.topk_weights,
                state.scale,
                state.topk_ids,
            )

            # Record done event or sync
            done_event: torch.cuda.Event | None = None
            if self.async_finish:
                done_event = torch.cuda.Event(blocking=False, interprocess=False)
                done_event.record(comm_stream)
            else:
                compute_stream.wait_stream(comm_stream)

        # Mark output tensors
        for t in (dispatch_hidden, dispatch_weights, dispatch_scale, dispatch_ids):
            if t is not None:
                t.record_stream(comm_stream)

        # # Wait for dispatch to complete on compute stream
        # if self.async_finish and done_event is not None:
        #     torch.cuda.current_stream().wait_event(done_event)
        print(f"after dispatch_b ubatch {ubatch_id}")
        print("dispatch_recv_count", dispatch_recv_count.shape)
        print("dispatch_recv_count", dispatch_recv_count)
        return DispatchResult(
            hidden_states=dispatch_hidden,
            hidden_states_scale=dispatch_scale,
            topk_ids=dispatch_ids,
            topk_weights=dispatch_weights,
            num_recv_tokens_per_expert=dispatch_recv_count,
        )

    # ======================== Combine Phase A ========================
    def _combine_a(
        self,
        output: torch.Tensor,
        fused_expert_output: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> None:
        """
        Phase A of combine: capture event and store intermediate state.
        This phase runs on compute stream after expert computation.
        """
        ubatch_id = dbo_current_ubatch_id()
        print(f"_combine_a called ubatch {ubatch_id}")
        self._update_stage(ubatch_id, _Stage.AFTER_DISPATCH_B, _Stage.AFTER_COMBINE_A)

        # Capture event on compute stream
        previous_event = self._capture_event_if_async()

        # Store intermediate state for combine_b
        self._combine_states[ubatch_id] = CombineIntermediateState(
            fused_expert_output=fused_expert_output,
            topk_ids=topk_ids,
            output=output,
            num_token=output.shape[0],
            previous_event=previous_event,
        )

    # ======================== Combine Phase B ========================
    def _combine_b(self) -> torch.Tensor:
        """
        Phase B of combine: execute actual mori combine on comm stream.
        Returns combined output after proper synchronization.
        """
        ubatch_id = dbo_current_ubatch_id()
        print(f"_combine_b called ubatch {ubatch_id}")
        self._update_stage(ubatch_id, _Stage.AFTER_COMBINE_A, _Stage.INITIAL)

        state = self._combine_states[ubatch_id]
        if state is None:
            raise RuntimeError(f"combine_b called without combine_a for ubatch {ubatch_id}")
        self._combine_states[ubatch_id] = None

        compute_stream = torch.cuda.current_stream()
        comm_stream = self._comm_stream

        # Mark tensors as used by comm_stream
        for t in (state.fused_expert_output, state.topk_ids):
            t.record_stream(comm_stream)

        with torch.cuda.stream(comm_stream):
            # Synchronize: comm_stream waits for compute_stream
            if state.previous_event is not None:
                comm_stream.wait_event(state.previous_event)
            else:
                comm_stream.wait_stream(compute_stream)

            # Execute mori combine
            combined = self.mori_op.combine(
                state.fused_expert_output,
                None,
                state.topk_ids,
            )[0]

            # Record done event or sync
            done_event: torch.cuda.Event | None = None
            if self.async_finish:
                done_event = torch.cuda.Event(blocking=False, interprocess=False)
                done_event.record(comm_stream)
            else:
                compute_stream.wait_stream(comm_stream)

        combined.record_stream(comm_stream)

        # Wait for combine to complete on compute stream
        if self.async_finish and done_event is not None:
            torch.cuda.current_stream().wait_event(done_event)
        print(f"after combine_b ubatch {ubatch_id}")
        # Copy to output
        state.output.copy_(combined[:state.num_token], non_blocking=True)
        return state.output

    # ======================== Async API (for modular_kernel) ========================
    def prepare_async(
        self,
        a1: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        num_experts: int,
        expert_map: torch.Tensor | None,
        apply_router_weight_on_input: bool,
        quant_config: FusedMoEQuantConfig = None,
        quant_type: QuantType = QuantType.No,
    ) -> mk.PrepareResultType:
        """
        Async prepare that directly returns the result (like sglang's dispatch pattern).
        
        Flow:
        1. dispatch_a: capture event and store state
        2. dbo_yield(): yield to other ubatch
        3. dispatch_b: execute mori dispatch and return results
        
        This ensures all ranks call mori operations in the same order.
        """
        assert not apply_router_weight_on_input, (
            "mori does not support apply_router_weight_on_input=True"
        )

        scale = None
        if self.use_fp8_dispatch:
            from aiter import get_hip_quant
            quant_func = get_hip_quant(quant_type)
            a1, scale = quant_func(a1, quant_dtype=dtypes.fp8)

        # Phase A: capture event and store state
        self._dispatch_a(a1, topk_weights, topk_ids, scale)

        # Yield to other ubatch (like sglang's pattern)
        # This ensures both ubatches complete dispatch_a before any dispatch_b
        dbo_yield()

        # Phase B: execute dispatch and return results directly
        result = self._dispatch_b()
        print("this is result expert_tokens_meta.expert_num_tokens", result.num_recv_tokens_per_expert)

        expert_tokens_meta = mk.ExpertTokensMetadata(
            expert_num_tokens=result.num_recv_tokens_per_expert,
            expert_num_tokens_cpu=None,
        )

        return (
            result.hidden_states,
            result.hidden_states_scale,
            expert_tokens_meta,
            result.topk_ids,
            result.topk_weights,
        )

    def finalize_async(
        self,
        output: torch.Tensor,
        fused_expert_output: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        apply_router_weight_on_input: bool,
    ) -> None:
        """
        Async finalize that directly executes combine (like sglang's pattern).
        
        Flow:
        1. combine_a: capture event and store state
        2. dbo_yield(): yield to other ubatch
        3. combine_b: execute mori combine
        """
        # Phase A: capture event and store state
        self._combine_a(output, fused_expert_output, topk_ids)

        # Yield to other ubatch
        dbo_yield()

        # Phase B: execute combine directly
        self._combine_b()

        # Final yield to allow other ubatch to complete its combine_b
        # Without this, the first ubatch to finish will continue to next layer
        # while the second ubatch is stuck waiting to be woken up
        dbo_yield()

    # ======================== Sync API (for non-DBO mode) ========================
    def prepare(
        self,
        a1: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        num_experts: int,
        expert_map: torch.Tensor | None,
        apply_router_weight_on_input: bool,
        quant_config: FusedMoEQuantConfig = None,
        quant_type: QuantType = QuantType.No,
    ) -> mk.PrepareResultType:
        """Synchronous prepare for non-DBO mode."""
        assert not apply_router_weight_on_input, (
            "mori does not support apply_router_weight_on_input=True"
        )

        scale = None
        if self.use_fp8_dispatch:
            from aiter import get_hip_quant
            quant_func = get_hip_quant(quant_type)
            a1, scale = quant_func(a1, quant_dtype=dtypes.fp8)

        # Direct dispatch without DBO interleaving
        (
            dispatch_hidden,
            dispatch_weights,
            dispatch_scale,
            dispatch_ids,
            dispatch_recv_count,
        ) = self.mori_op.dispatch(a1, topk_weights, scale, topk_ids)

        expert_tokens_meta = mk.ExpertTokensMetadata(
            expert_num_tokens=dispatch_recv_count,
            expert_num_tokens_cpu=None,
        )

        return (
            dispatch_hidden,
            dispatch_scale,
            expert_tokens_meta,
            dispatch_ids,
            dispatch_weights,
        )

    def finalize(
        self,
        output: torch.Tensor,
        fused_expert_output: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        apply_router_weight_on_input: bool,
    ) -> None:
        """Synchronous finalize for non-DBO mode."""
        num_token = output.shape[0]
        result = self.mori_op.combine(
            fused_expert_output,
            None,
            topk_ids,
        )[0]
        output.copy_(result[:num_token])
