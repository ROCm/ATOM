# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import logging
import threading
from contextlib import nullcontext
from typing import TypeAlias

import numpy as np
import torch
from torch import nn

from atom.utils.forward_context import (
    Context,
    DPMetadata,
    ForwardContext,
    _forward_context_local,
    get_forward_context,
)

from .ubatch_splitting import UBatchSlice
from .ubatching import make_tbo_contexts

logger = logging.getLogger("atom")

UBatchModelOutput: TypeAlias = torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]


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
        self.comm_stream: torch.cuda.Stream | None = None
        # Barrier: ubatch threads + main thread
        self.ready_barrier = threading.Barrier(3)  # 2 ubatch threads + 1 main

        # Persistent ubatch worker pool. Previously every forward spawned + joined
        # 2 threads.
        self._num_workers = 2
        self._workers: list[threading.Thread] = []
        self._worker_jobs: list[callable | None] = [None] * self._num_workers
        self._worker_job_ready = [threading.Event() for _ in range(self._num_workers)]
        self._worker_job_done = [threading.Event() for _ in range(self._num_workers)]
        self._workers_device: torch.device | None = None

    def _worker_loop(self, idx: int):
        # Bind this long-lived thread to the device ONCE — the HIP per-thread
        # context (and its getprops storm) is paid here, not every forward.
        if self._workers_device is not None:
            torch.cuda.set_device(self._workers_device)
        while True:
            self._worker_job_ready[idx].wait()
            self._worker_job_ready[idx].clear()
            job = self._worker_jobs[idx]
            self._worker_jobs[idx] = None
            try:
                if job is not None:
                    job()
            finally:
                self._worker_job_done[idx].set()

    def _ensure_workers(self, device: torch.device):
        if self._workers:
            return
        self._workers_device = device
        for i in range(self._num_workers):
            t = threading.Thread(
                target=self._worker_loop, args=(i,), daemon=True, name=f"tbo-ub-{i}"
            )
            self._workers.append(t)
            t.start()

    def _ensure_comm_stream(self):
        if self.comm_stream is None:
            priority = getattr(
                self.attn_metadata_builder, "tbo_comm_stream_priority", 0
            )
            self.comm_stream = torch.cuda.Stream(priority=priority)

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

        full_running_bs = ctx.context.running_bs
        forward_contexts = []
        ub_inputs = []

        # When using DP gather/scatter (no EP/all2all), compute the
        # DP-synchronized per-ubatch row count so MoE's pad_for_all_gather can
        # just read context.running_tokens.  MORI path doesn't need this.
        dp_size = (
            len(ctx.context.running_tokens_across_dp) if self.dp_gather_scatter else 1
        )
        ub_running_tokens_list = self._compute_ub_running_tokens(
            ctx,
            N,
            full_running_bs,
            dp_size,
            input_ids.device,
        )

        for i, ub_slice in enumerate(ctx.ubatch_slices):
            ub_num_reqs = ub_slice.request_slice.stop - ub_slice.request_slice.start
            if ctx.context.is_prefill:
                ub_running_bs = ub_num_reqs
            else:
                ub_running_bs = self._decode_ub_running_bs(ctx, i, N, full_running_bs)
            ub_ctx = self._make_ubatch_context(
                original_ctx,
                ub_slice,
                ub_running_bs,
                i,
                ub_num_reqs,
                ub_running_tokens=ub_running_tokens_list[i],
                dp_metadata=ub_dp_metadata[i] if ub_dp_metadata is not None else None,
                running_tokens_across_dp=self._ub_tokens_across_dp(ctx, N, i),
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
        errors: list[Exception | None] = [None] * N

        device = input_ids.device
        assert (
            N <= self._num_workers
        ), f"TBO needs {N} ubatch workers but pool has {self._num_workers}"
        self._ensure_workers(device)

        def _make_job(idx):
            @torch.inference_mode()
            def _job():
                try:
                    ub_input_ids, ub_positions = ub_inputs[idx]
                    with tbo_ctxs[idx]:
                        model_output = self.model(ub_input_ids, ub_positions)
                    results.append((idx, self._validate_ubatch_output(model_output)))
                except Exception as e:
                    # logger.exception captures the full traceback. The partner
                    # thread is unblocked via TBOContext.partner.done (set in
                    # __exit__) so the main thread's done-wait returns promptly
                    # and re-raises errors[idx].
                    logger.exception("[TBO] ubatch %d crashed", idx)
                    errors[idx] = e
                finally:
                    # The worker thread outlives the forward; do not let its
                    # thread-local context keep this microbatch's step alive.
                    _forward_context_local.ctx = None

            return _job

        # Clear thread-local forward context so worker threads don't inherit it
        saved_ctx = getattr(_forward_context_local, "ctx", None)
        _forward_context_local.ctx = None

        try:
            # The backend owns resources shared by both microbatches.
            parent_forward = (
                self.attn_metadata_builder.ubatch_forward(ctx.attn_metadata)
                if self.attn_metadata_builder is not None
                else nullcontext()
            )
            with parent_forward:
                # Hand each ubatch job to its persistent worker and wake it.
                for i in range(N):
                    self._worker_job_done[i].clear()
                    self._worker_jobs[i] = _make_job(i)
                    self._worker_job_ready[i].set()

                # Same handshake as before: all reach the barrier, then wake thread 0.
                self.ready_barrier.wait()
                tbo_ctxs[0].cpu_wait_event.set()

                # Wait for this step's jobs to finish (replaces Thread.join()).
                for i in range(N):
                    self._worker_job_done[i].wait()
        finally:
            # The contexts point at each other (`partner`), a cycle that
            # outlives the forward until Python's cycle collector runs, and
            # each holds its microbatch's ForwardContext -- the attention
            # metadata and every GPU tensor the step memoized. The collector
            # runs on object counts and never sees that memory, so those
            # steps piled up until the GPU ran out. Break the cycle here, with
            # every worker done.
            for tbo_ctx in tbo_ctxs:
                tbo_ctx.partner = None
                tbo_ctx.forward_context = None
            # Order everything the workers left on the comm stream before any
            # later work on the compute stream. A comm section normally rejoins
            # (`switch_to_compute_sync`), but one that raised did not, and the
            # next forward republishes the microbatch buffers on this stream.
            compute_stream.wait_stream(self.comm_stream)
            # Restore original forward context
            _forward_context_local.ctx = saved_ctx

        # Check for errors
        for e in errors:
            if e is not None:
                raise e

        sorted_results = [value for _, value in sorted(results)]
        return self._concat_ubatch_outputs(sorted_results)

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

    def _make_ubatch_dp_metadata(self, ctx: ForwardContext, N: int):
        """Build per-ubatch :class:`DPMetadata` so the MoE DP collective uses
        each ubatch's own per-rank token counts.

        Returns ``None`` when DP is disabled / no dp_metadata on the parent
        context (the shared metadata is then reused, which is correct for the
        single-rank case). Otherwise returns a list of length ``N``.

        Each ubatch's per-rank tokens are the step's sync's own
        (`ub_tokens_across_dp`): `local_tbo_precompute` sizes them exactly as
        the slices are cut, so no collective of its own is needed.
        """
        if ctx.dp_metadata is None:
            return None
        return [
            DPMetadata(np.asarray(ctx.ub_tokens_across_dp[i], dtype=np.int32))
            for i in range(N)
        ]

    @staticmethod
    def _ub_tokens_across_dp(ctx: ForwardContext, N: int, i: int):
        """Ubatch ``i``'s per-rank token counts, or None if none were reduced.

        Passed down because inside a ubatch only this rank's split is in reach.
        """
        rows = ctx.ub_tokens_across_dp
        return None if rows is None or len(rows) != N else rows[i]

    @staticmethod
    def _decode_ub_running_bs(
        ctx: ForwardContext, i: int, N: int, full_running_bs: int
    ) -> int:
        """Per-ubatch padded request count for a decode micro-batch.

        Must be IDENTICAL across DP ranks: the MoE all_gather/reduce_scatter
        pads each ubatch to this size, so a per-rank-local split (which differs
        when ranks carry different decode batch sizes, e.g. during drain)
        desyncs the collective and faults. Derive it from the DP-unified
        ``ub_max_tokens_across_dp`` (MAX-reduced in ``ForwardMode.decide``),
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
            return full_running_bs // N
        return full_running_bs - (full_running_bs // N) * (N - 1)

    @staticmethod
    def _compute_ub_running_tokens(
        ctx: ForwardContext,
        N: int,
        full_running_bs: int,
        dp_size: int,
        device: torch.device,
    ) -> list[int]:
        """Per-ubatch ``running_tokens`` -- what each micro-batch's MoE pads to.

        In TOKENS on both branches, which is why neither multiplies by
        ``dp_size``: pad_for_all_gather gathers across ranks itself.

        For prefill (eager only): the cross-DP per-ubatch token MAX that
            ``ForwardMode.decide`` already packed into the single DP
            all_reduce (``ctx.ub_max_tokens_across_dp``). Falls back to
            local sizes when DP is off / value not precomputed.
        For decode: the per-ubatch padded request count times ``max_seqlen_q``.
        """
        if ctx.context.is_prefill:
            if (
                dp_size > 1
                and ctx.ub_max_tokens_across_dp is not None
                and len(ctx.ub_max_tokens_across_dp) == N
            ):
                # Precomputed via the packed DP reduce — no extra all_reduce.
                return list(ctx.ub_max_tokens_across_dp)
            # A prefill ubatch runs the rows it scheduled.
            return [
                ub_slice.token_slice.stop - ub_slice.token_slice.start
                for ub_slice in ctx.ubatch_slices
            ]
        max_q = int(getattr(ctx.attn_metadata, "max_seqlen_q", 1) or 1)
        return [
            UBatchWrapper._decode_ub_running_bs(ctx, i, N, full_running_bs) * max_q
            for i in range(N)
        ]

    def _make_ubatch_context(
        self,
        ctx: ForwardContext,
        ub_slice: UBatchSlice,
        ub_running_bs: int,
        ubatch_idx: int = 0,
        actual_num_reqs: int | None = None,
        ub_running_tokens: int | None = None,
        dp_metadata=None,
        running_tokens_across_dp: tuple[int, ...] | None = None,
    ) -> ForwardContext:
        """Build a ForwardContext for a single micro-batch."""
        ub_num_reqs = ub_slice.request_slice.stop - ub_slice.request_slice.start

        if ctx.context.is_prefill:
            ub_attn = self.attn_metadata_builder.build_ubatch_prefill_metadata(
                ctx.attn_metadata,
                ub_slice,
                ub_running_bs,
                ubatch_idx=ubatch_idx,
            )
        else:
            attn_bs = actual_num_reqs if actual_num_reqs is not None else ub_running_bs
            ub_attn = self.attn_metadata_builder.build_ubatch_metadata(
                ubatch_idx,
                attn_bs,
            )

        # Split Context
        ub_scheduled_tokens = ub_slice.token_slice.stop - ub_slice.token_slice.start
        if ub_running_tokens is not None:
            running_tokens = ub_running_tokens
        elif ctx.context.is_prefill:
            running_tokens = ub_scheduled_tokens
        else:
            running_tokens = ub_running_bs * int(
                getattr(ctx.attn_metadata, "max_seqlen_q", 1) or 1
            )
        ub_context = Context(
            positions=ctx.context.positions[ub_slice.token_slice],
            is_prefill=ctx.context.is_prefill,
            is_dummy_run=ctx.context.is_dummy_run,
            scheduled_bs=ub_num_reqs,
            scheduled_tokens=ub_scheduled_tokens,
            running_bs=ub_running_bs,
            running_tokens=running_tokens,
            running_tokens_across_dp=running_tokens_across_dp,
            is_draft=ctx.context.is_draft,
            # Carry over per-ubatch slice of input_ids for hash MoE (PCP+TBO mode).
            # run_model stores local (1/pcp) ids; each ubatch takes its token_slice.
            # ForCausalLM.forward then allgathers the slice to get ids matching the
            # MoE's per-ubatch allgathered hidden states (padded_total//2 tokens).
            input_ids=(
                ctx.context.input_ids[ub_slice.token_slice]
                if ctx.context.input_ids is not None
                else None
            ),
            # Where this ubatch's rows start on the full forward's token axis;
            # drafter aux capture writes at this offset.
            ubatch_token_offset=ub_slice.token_slice.start,
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
