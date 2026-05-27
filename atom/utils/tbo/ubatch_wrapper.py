# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import logging
import threading
import traceback
from dataclasses import dataclass
from typing import Any, Optional

import torch
import torch.nn as nn

from atom.utils.forward_context import (
    Context,
    ForwardContext,
    get_forward_context,
    _forward_context_local,
)

from .ubatch_splitting import UBatchSlice
from .ubatching import make_tbo_contexts

logger = logging.getLogger("atom")


@dataclass
class TBOGraphData:
    """Stores a captured CUDAGraph alongside objects that must stay alive."""

    graph: torch.cuda.CUDAGraph
    tbo_ctxs: list  # keep torch.Event objects alive for replay
    output: Any = None  # output tensor reference from capture


class UBatchWrapper(nn.Module):
    """Wraps a model to split decode batches into micro-batches."""

    def __init__(
        self,
        model: nn.Module,
        attn_metadata_builder=None,
        dp_gather_scatter: bool = False,
    ):
        super().__init__()
        self.model = model
        self.attn_metadata_builder = attn_metadata_builder
        self.dp_gather_scatter = dp_gather_scatter
        self.comm_stream: Optional[torch.cuda.Stream] = None
        # Barrier: ubatch threads + main thread
        self.ready_barrier = threading.Barrier(3)  # 2 ubatch threads + 1 main
        # TBO CUDAGraph storage: keyed by (graph_bs, max_q_len)
        self.tbo_graphs: dict[tuple, TBOGraphData] = {}

    def _ensure_comm_stream(self):
        if self.comm_stream is None:
            self.comm_stream = torch.cuda.Stream()

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        ctx = get_forward_context()
        # Hit-rate counters: split by prefill / decode to diagnose whether
        # the collective TBO decision is firing where we expect.
        is_prefill = ctx.context.is_prefill
        if ctx.ubatch_slices is None:
            if is_prefill:
                self._n_skip_prefill = getattr(self, "_n_skip_prefill", 0) + 1
            else:
                self._n_skip_decode = getattr(self, "_n_skip_decode", 0) + 1
        else:
            if is_prefill:
                self._n_used_prefill = getattr(self, "_n_used_prefill", 0) + 1
            else:
                self._n_used_decode = getattr(self, "_n_used_decode", 0) + 1
        total = (
            getattr(self, "_n_skip_prefill", 0)
            + getattr(self, "_n_skip_decode", 0)
            + getattr(self, "_n_used_prefill", 0)
            + getattr(self, "_n_used_decode", 0)
        )
        # Log on (a) the very first TBO hit so we know it actually fired,
        # (b) every 50 fwds thereafter. Prefill-only TBO can total way less
        # than 200 fwds across a whole benchmark, so the old `% 200 == 0`
        # gate could silently never fire.
        first_hit = ctx.ubatch_slices is not None and (
            getattr(self, "_n_used_prefill", 0) + getattr(self, "_n_used_decode", 0)
        ) == 1
        if first_hit or total % 50 == 0:
            logger.info(
                "[TBO] used_p=%d skip_p=%d used_d=%d skip_d=%d (hit_p=%.1f%% hit_d=%.1f%%)",
                getattr(self, "_n_used_prefill", 0),
                getattr(self, "_n_skip_prefill", 0),
                getattr(self, "_n_used_decode", 0),
                getattr(self, "_n_skip_decode", 0),
                100.0
                * getattr(self, "_n_used_prefill", 0)
                / max(
                    1,
                    getattr(self, "_n_used_prefill", 0)
                    + getattr(self, "_n_skip_prefill", 0),
                ),
                100.0
                * getattr(self, "_n_used_decode", 0)
                / max(
                    1,
                    getattr(self, "_n_used_decode", 0)
                    + getattr(self, "_n_skip_decode", 0),
                ),
            )
        if ctx.ubatch_slices is None:
            return self.model(input_ids, positions)
        return self._run_ubatches(input_ids, positions, ctx)

    def _run_ubatches(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        ctx: ForwardContext,
    ) -> torch.Tensor:
        """Launch threads that each call self.model() inside a TBOContext."""
        self._ensure_comm_stream()
        original_ctx = ctx
        N = len(ctx.ubatch_slices)
        compute_stream = torch.cuda.current_stream()

        full_graph_bs = ctx.context.graph_bs
        forward_contexts = []
        ub_inputs = []

        # When using DP gather/scatter (no EP/all2all), compute
        # DP-synchronized graph_bs per ubatch so MoE's pad_for_all_gather
        # can just read context.graph_bs.  MORI path doesn't need this.
        dp_size = self._get_dp_size() if self.dp_gather_scatter else 1
        ub_graph_bs_list = self._compute_ub_graph_bs(
            ctx,
            N,
            full_graph_bs,
            dp_size,
            input_ids.device,
        )

        for i, ub_slice in enumerate(ctx.ubatch_slices):
            ub_num_reqs = ub_slice.request_slice.stop - ub_slice.request_slice.start
            if ctx.context.is_prefill:
                padded_bs = ub_num_reqs
            else:
                if i < N - 1:
                    padded_bs = full_graph_bs // N
                else:
                    padded_bs = full_graph_bs - (full_graph_bs // N) * (N - 1)
            ub_ctx = self._make_ubatch_context(
                original_ctx,
                ub_slice,
                padded_bs,
                i,
                ub_num_reqs,
                ub_graph_bs=ub_graph_bs_list[i],
            )
            forward_contexts.append(ub_ctx)
            ub_token_slice = (
                input_ids[ub_slice.token_slice],
                positions[ub_slice.token_slice],
            )
            # Empty ubatch would return immediately, leaving the partner
            # wedged at its next yield (see TBOContext.__exit__ comment).
            # The new partner.done short-circuit covers this case too, but
            # an empty ubatch is always a metadata bug upstream — assert
            # loudly rather than silently waste a TBO split.
            assert ub_token_slice[0].numel() > 0, (
                f"ubatch {i} produced an empty token slice "
                f"(ts={ub_slice.token_slice}, rs={ub_slice.request_slice}); "
                "check _local_tbo_precompute / maybe_create_ubatch_slices."
            )
            ub_inputs.append(ub_token_slice)

        tbo_ctxs = make_tbo_contexts(
            num_micro_batches=N,
            compute_stream=compute_stream,
            comm_stream=self.comm_stream,
            forward_contexts=forward_contexts,
            ready_barrier=self.ready_barrier,
        )

        results: list[tuple[int, torch.Tensor]] = []
        errors: list[Optional[Exception]] = [None] * N

        device = input_ids.device

        @torch.inference_mode()
        def _ubatch_thread(idx):
            try:
                # Ensure the child thread has a HIP/CUDA context on the
                # correct device — without this, the first HIP API call
                # (e.g. RCCL collective, kernel launch) may block or fail
                # because the runtime lazily initialises per-thread state.
                torch.cuda.set_device(device)

                ub_input_ids, ub_positions = ub_inputs[idx]
                with tbo_ctxs[idx]:
                    model_output = self.model(ub_input_ids, ub_positions)
                results.append((idx, model_output))
            except Exception as e:
                # `logger.exception` captures the full traceback to the atom
                # logger (which goes to stderr + log file); previously the
                # bare `traceback.print_exc()` could be drowned by other
                # stderr output across 8 worker procs. The partner thread
                # is now unblocked via TBOContext.partner.done (set in
                # __exit__) so the main thread's `t.join()` will return
                # promptly and re-raise this `errors[idx]`.
                logger.exception("[TBO] ubatch %d crashed: %s", idx, e)
                errors[idx] = e

        # Clear thread-local forward context so threads don't inherit it
        saved_ctx = getattr(_forward_context_local, "ctx", None)
        _forward_context_local.ctx = None

        try:
            threads = []
            for i in range(N):
                t = threading.Thread(target=_ubatch_thread, args=(i,))
                threads.append(t)
                t.start()

            self.ready_barrier.wait()
            tbo_ctxs[0].cpu_wait_event.set()

            for t in threads:
                t.join()
        finally:
            # Restore original forward context
            _forward_context_local.ctx = saved_ctx

        # Check for errors
        for e in errors:
            if e is not None:
                raise e

        sorted_results = [value for _, value in sorted(results)]
        return torch.cat(sorted_results, dim=0)

    def capture_tbo_graph(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        graph_pool,
        capture_stream: torch.cuda.Stream,
        output_buffer: Optional[torch.Tensor] = None,
    ) -> tuple[torch.cuda.CUDAGraph, torch.Tensor]:
        """Capture a CUDAGraph for TBO ubatch execution.

        Threads are started and cuBLAS is initialized BEFORE graph capture
        begins (following vLLM's _capture_ubatches pattern). Only the model
        forward execution happens during capture.

        If output_buffer is provided, the concatenated output is copied into
        it inside the graph capture so replay writes to the same buffer.

        Returns (graph, output_tensor).
        """
        self._ensure_comm_stream()
        ctx = get_forward_context()
        N = len(ctx.ubatch_slices)
        compute_stream = capture_stream

        # Build per-ubatch ForwardContexts from pre-allocated forward_vars.
        full_graph_bs = ctx.context.graph_bs
        # only padding for all_gather/reduce_scatter pass
        all_gahter_dp_size = self._get_dp_size() if self.dp_gather_scatter else 1
        forward_contexts = []
        ub_inputs = []
        for i, ub_slice in enumerate(ctx.ubatch_slices):
            if i < N - 1:
                padded_bs = full_graph_bs // N
            else:
                padded_bs = full_graph_bs - (full_graph_bs // N) * (N - 1)
            ub_ctx = self._make_ubatch_context(
                ctx,
                ub_slice,
                padded_bs,
                i,
                ub_graph_bs=padded_bs * all_gahter_dp_size,
            )
            forward_contexts.append(ub_ctx)
            ub_inputs.append(
                (
                    input_ids[ub_slice.token_slice],
                    positions[ub_slice.token_slice],
                )
            )

        tbo_ctxs = make_tbo_contexts(
            num_micro_batches=N,
            compute_stream=compute_stream,
            comm_stream=self.comm_stream,
            forward_contexts=forward_contexts,
            ready_barrier=self.ready_barrier,
        )

        results: list[tuple[int, torch.Tensor]] = []
        errors: list[Optional[Exception]] = [None] * N
        device = input_ids.device

        @torch.inference_mode()
        def _capture_thread(idx):
            try:
                torch.cuda.set_device(device)
                # Initialize cuBLAS on both streams BEFORE barrier.
                # This prevents workspace allocation during graph capture.
                with torch.cuda.stream(tbo_ctxs[idx].compute_stream):
                    _ = torch.cuda.current_blas_handle()
                with torch.cuda.stream(tbo_ctxs[idx].comm_stream):
                    _ = torch.cuda.current_blas_handle()

                ub_input_ids, ub_positions = ub_inputs[idx]
                with tbo_ctxs[idx]:
                    model_output = self.model(ub_input_ids, ub_positions)
                results.append((idx, model_output))
            except Exception as e:
                traceback.print_exc()
                errors[idx] = e

        saved_ctx = getattr(_forward_context_local, "ctx", None)
        _forward_context_local.ctx = None

        try:
            # Start threads — cuBLAS init happens before barrier
            threads = []
            for i in range(N):
                t = threading.Thread(target=_capture_thread, args=(i,))
                threads.append(t)
                t.start()

            # Wait for all threads to be ready (past cuBLAS init, at barrier)
            self.ready_barrier.wait()

            # Capture the CUDAGraph
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, pool=graph_pool, stream=capture_stream):
                # Wake thread 0 — all GPU work from threads is captured
                tbo_ctxs[0].cpu_wait_event.set()
                for t in threads:
                    t.join()
                # Concatenate results (this op is captured too)
                sorted_results = [v for _, v in sorted(results)]
                output = torch.cat(sorted_results, dim=0)
                # Copy into caller's buffer so replay writes to the right place
                if output_buffer is not None:
                    output_buffer.copy_(output)
        finally:
            _forward_context_local.ctx = saved_ctx

        for e in errors:
            if e is not None:
                raise e

        # Store TBOContext objects to keep torch.Event alive during replay
        graph_key = (ctx.context.graph_bs, ctx.attn_metadata.max_seqlen_q)
        self.tbo_graphs[graph_key] = TBOGraphData(
            graph=graph,
            tbo_ctxs=tbo_ctxs,
            output=output,
        )
        logger.info(f"[TBO] Captured CUDAGraph for {graph_key}")

        return graph, output

    @staticmethod
    def _get_dp_size() -> int:
        """Return DP world size (1 if DP is not active)."""
        try:
            from aiter.dist.parallel_state import get_dp_group

            return get_dp_group().world_size
        except Exception:
            return 1

    @staticmethod
    def _compute_ub_graph_bs(
        ctx: ForwardContext,
        N: int,
        full_graph_bs: int,
        dp_size: int,
        device: torch.device,
    ) -> list[int]:
        """
        For prefill (eager only): use cross-DP per-ubatch token MAX that
            ``ModelRunner._preprocess`` already packed into the single DP
            all_reduce (``ctx.ub_max_tokens_across_dp``). Falls back to
            local sizes when DP is off / value not precomputed.
        For decode: padded_bs * dp_size.
        """
        if ctx.context.is_prefill:
            if (
                dp_size > 1
                and ctx.ub_max_tokens_across_dp is not None
                and len(ctx.ub_max_tokens_across_dp) == N
            ):
                # Precomputed via the packed DP reduce — no extra all_reduce.
                return list(ctx.ub_max_tokens_across_dp)
            ub_sizes = []
            for ub_slice in ctx.ubatch_slices:
                ub_num_tokens = ub_slice.token_slice.stop - ub_slice.token_slice.start
                ub_sizes.append(ub_num_tokens)
            return ub_sizes
        else:
            result = []
            for i in range(N):
                if i < N - 1:
                    padded_bs = full_graph_bs // N
                else:
                    padded_bs = full_graph_bs - (full_graph_bs // N) * (N - 1)
                result.append(padded_bs * dp_size)
            return result

    def _make_ubatch_context(
        self,
        ctx: ForwardContext,
        ub_slice: UBatchSlice,
        padded_bs: int,
        ubatch_idx: int = 0,
        actual_num_reqs: int | None = None,
        ub_graph_bs: int | None = None,
    ) -> ForwardContext:
        """Build a ForwardContext for a single micro-batch."""
        ub_num_reqs = ub_slice.request_slice.stop - ub_slice.request_slice.start

        if ctx.context.is_prefill:
            ub_attn = self.attn_metadata_builder.build_ubatch_prefill_metadata(
                ctx.attn_metadata,
                ub_slice,
                padded_bs,
                ubatch_idx=ubatch_idx,
            )
        else:
            attn_bs = actual_num_reqs if actual_num_reqs is not None else padded_bs
            ub_attn = self.attn_metadata_builder.build_ubatch_metadata(
                ubatch_idx,
                attn_bs,
            )

        # Split Context
        ub_num_tokens = ub_slice.token_slice.stop - ub_slice.token_slice.start
        if ub_graph_bs is not None:
            graph_bs = ub_graph_bs
        elif ctx.context.is_prefill:
            graph_bs = ub_num_tokens
        else:
            graph_bs = padded_bs
        ub_context = Context(
            positions=ctx.context.positions[ub_slice.token_slice],
            is_prefill=ctx.context.is_prefill,
            is_dummy_run=ctx.context.is_dummy_run,
            batch_size=ub_num_reqs,
            graph_bs=graph_bs,
            is_draft=ctx.context.is_draft,
        )

        return ForwardContext(
            attn_metadata=ub_attn,
            no_compile_layers=ctx.no_compile_layers,
            kv_cache_data=ctx.kv_cache_data,
            context=ub_context,
            dp_metadata=ctx.dp_metadata,  # shared across ubatches
            spec_decode_metadata=None,  # not supported with TBO
            ubatch_slices=None,  # prevent recursion
            # Propagate parent's cached current_stream so downstream code
            # (V4 dual-stream MoE / attention compressor / etc.) that reads
            # `get_forward_context().main_stream` sees a real Stream rather
            # than None. Without this the per-ubatch ForwardContext picks up
            # the dataclass default (None) and the first `wait_stream(None)`
            # crashes with `AttributeError: 'NoneType' has no record_event`.
            # The TBO worker thread switches stream via TBOContext, so we
            # use the parent's stream here (= the original compute stream),
            # which is also the stream the worker resumes on after each
            # comm→compute switch.
            main_stream=ctx.main_stream,
            in_hipgraph=ctx.in_hipgraph,
        )

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Delegate to the wrapped model's compute_logits."""
        return self.model.compute_logits(hidden_states)

    def __getattr__(self, name: str):
        """Forward attribute access to the wrapped model for non-overridden attrs."""
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)
