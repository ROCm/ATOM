# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import logging
import threading
import traceback
from dataclasses import dataclass
from typing import Optional, TypeAlias

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

UBatchModelOutput: TypeAlias = torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]


@dataclass
class TBOGraphData:
    """Stores a captured CUDAGraph alongside objects that must stay alive."""

    graph: torch.cuda.CUDAGraph
    tbo_ctxs: list  # keep torch.Event objects alive for replay
    output: Optional[UBatchModelOutput] = None  # output reference from capture


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

    def forward(
        self, input_ids: torch.Tensor, positions: torch.Tensor
    ) -> UBatchModelOutput:
        ctx = get_forward_context()
        if ctx.ubatch_slices is None:
            return self.model(input_ids, positions)
        return self._run_ubatches(input_ids, positions, ctx)

    def _run_ubatches(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        ctx: ForwardContext,
    ) -> UBatchModelOutput:
        """Launch threads that each call self.model() inside a TBOContext."""
        self._ensure_comm_stream()
        original_ctx = ctx
        N = len(ctx.ubatch_slices)
        compute_stream = torch.cuda.current_stream()

        ub_dp_metadata = self._make_ubatch_dp_metadata(ctx, N)

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
                padded_bs = self._decode_ub_padded_bs(ctx, i, N, full_graph_bs)
            ub_ctx = self._make_ubatch_context(
                original_ctx,
                ub_slice,
                padded_bs,
                i,
                ub_num_reqs,
                ub_graph_bs=ub_graph_bs_list[i],
                dp_metadata=ub_dp_metadata[i] if ub_dp_metadata is not None else None,
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
                "check local_tbo_precompute / maybe_create_ubatch_slices."
            )
            ub_inputs.append(ub_token_slice)

        tbo_ctxs = make_tbo_contexts(
            num_micro_batches=N,
            compute_stream=compute_stream,
            comm_stream=self.comm_stream,
            forward_contexts=forward_contexts,
            ready_barrier=self.ready_barrier,
        )

        results: list[tuple[int, UBatchModelOutput]] = []
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
                results.append((idx, self._validate_ubatch_output(model_output)))
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
        return self._concat_ubatch_outputs(sorted_results)

    def capture_tbo_graph(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        graph_pool,
        capture_stream: torch.cuda.Stream,
        output_buffer: Optional[torch.Tensor] = None,
    ) -> tuple[torch.cuda.CUDAGraph, UBatchModelOutput]:
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
        ub_dp_metadata = self._make_ubatch_dp_metadata(ctx, N)
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
                ub_graph_bs=padded_bs,
                dp_metadata=ub_dp_metadata[i] if ub_dp_metadata is not None else None,
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

        results: list[tuple[int, UBatchModelOutput]] = []
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
                results.append((idx, self._validate_ubatch_output(model_output)))
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
                output = self._concat_ubatch_outputs(sorted_results)
                # Copy into caller's buffer so replay writes to the right place
                if output_buffer is not None:
                    output_buffer.copy_(self._primary_output(output))
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
    def _concat_ubatch_outputs(
        outputs: list[UBatchModelOutput],
    ) -> UBatchModelOutput:
        """Concatenate Tensor or Eagle3 aux outputs from per-ubatch forwards."""
        first = outputs[0]
        if isinstance(first, torch.Tensor):
            if not all(isinstance(output, torch.Tensor) for output in outputs):
                raise TypeError("TBO ubatch outputs must have matching structures")
            return torch.cat(outputs, dim=0)

        if not all(isinstance(output, tuple) for output in outputs):
            raise TypeError("TBO ubatch outputs must have matching structures")

        hidden_states = torch.cat([output[0] for output in outputs], dim=0)
        num_aux = len(first[1])
        if not all(len(output[1]) == num_aux for output in outputs):
            raise ValueError("TBO ubatch aux output counts must match")
        aux_hidden_states = [
            torch.cat([output[1][idx] for output in outputs], dim=0)
            for idx in range(num_aux)
        ]
        return hidden_states, aux_hidden_states

    @staticmethod
    def _primary_output(output: UBatchModelOutput) -> torch.Tensor:
        """Return the tensor backed by ModelRunner's preallocated output buffer."""
        if isinstance(output, tuple):
            return output[0]
        return output

    @staticmethod
    def _validate_ubatch_output(output: object) -> UBatchModelOutput:
        """Accept only the TBO output contracts: Tensor or Eagle3 aux tuple."""
        if isinstance(output, torch.Tensor):
            return output
        if (
            isinstance(output, tuple)
            and len(output) == 2
            and isinstance(output[0], torch.Tensor)
            and isinstance(output[1], list)
            and all(isinstance(aux, torch.Tensor) for aux in output[1])
        ):
            return output
        raise TypeError(
            "TBO ubatch output must be a Tensor or "
            "(Tensor, list[Tensor]), got "
            f"{type(output).__name__}"
        )

    @staticmethod
    def _get_dp_size() -> int:
        """Return DP world size (1 if DP is not active)."""
        try:
            from aiter.dist.parallel_state import get_dp_group

            return get_dp_group().world_size
        except Exception:
            return 1

    def _make_ubatch_dp_metadata(self, ctx: ForwardContext, N: int):
        """Build per-ubatch :class:`DPMetadata` so the MoE DP collective uses
        each ubatch's own per-rank token counts.

        Returns ``None`` when DP is disabled / no dp_metadata on the parent
        context (the shared metadata is then reused, which is correct for the
        single-rank case). Otherwise returns a list of length ``N``.

        Each ubatch's per-rank token count is obtained with the same CPU
        all_reduce that :meth:`DPMetadata.num_tokens_across_dp` uses, one per
        ubatch. This is a CPU collective (cheap) and keeps every rank's
        all_gatherv / reduce_scatterv consistently sized.
        """
        if ctx.dp_metadata is None:
            return None
        from atom.config import get_current_atom_config
        from atom.utils.forward_context import DPMetadata

        parallel_config = get_current_atom_config().parallel_config
        metas = []
        for ub_slice in ctx.ubatch_slices:
            ub_tokens = ub_slice.token_slice.stop - ub_slice.token_slice.start
            metas.append(DPMetadata.make(parallel_config, int(ub_tokens), None))
        return metas

    @staticmethod
    def _decode_ub_padded_bs(
        ctx: ForwardContext, i: int, N: int, full_graph_bs: int
    ) -> int:
        """Per-ubatch padded request count for a decode micro-batch.

        Must be IDENTICAL across DP ranks: the MoE all_gather/reduce_scatter
        pads each ubatch to this size, so a per-rank-local split (which differs
        when ranks carry different decode batch sizes, e.g. during drain)
        desyncs the collective and faults. Derive it from the DP-unified
        ``ub_max_tokens_across_dp`` (MAX-reduced in ModelRunner._preprocess),
        converting the per-ubatch token max back to a request count via
        ``max_seqlen_q``. Falls back to the local split only when DP is off or
        the precomputed value is unavailable.
        """
        ub_max = ctx.ub_max_tokens_across_dp
        if ub_max is not None and len(ub_max) == N:
            max_q = getattr(ctx.attn_metadata, "max_seqlen_q", 1) or 1
            return max(1, ub_max[i] // max_q)
        # Fallback: local split (single-rank / value not precomputed).
        if i < N - 1:
            return full_graph_bs // N
        return full_graph_bs - (full_graph_bs // N) * (N - 1)

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
        For decode: per-rank padded_bs (the cross-DP all_gather in MoE's
            pad_for_all_gather multiplies by dp_size itself, so do NOT
            pre-multiply here).
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
                padded_bs = UBatchWrapper._decode_ub_padded_bs(ctx, i, N, full_graph_bs)
                result.append(padded_bs)
            return result

    def _make_ubatch_context(
        self,
        ctx: ForwardContext,
        ub_slice: UBatchSlice,
        padded_bs: int,
        ubatch_idx: int = 0,
        actual_num_reqs: int | None = None,
        ub_graph_bs: int | None = None,
        dp_metadata=None,
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
            dp_metadata=dp_metadata if dp_metadata is not None else ctx.dp_metadata,
            spec_decode_metadata=None,  # not supported with TBO
            ubatch_slices=None,  # prevent recursion
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
