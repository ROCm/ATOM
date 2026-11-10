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

from .ubatch_splitting import NUM_TBO_UBATCHES, UBatchSlice
from .ubatching import make_tbo_contexts

logger = logging.getLogger("atom")

UBatchModelOutput: TypeAlias = torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]


def _packed_tokens(
    tensors: list[torch.Tensor], token_order: torch.Tensor | None
) -> torch.Tensor:
    """The micro-batches' tokens end to end, each put back at its packed token
    (``token_order``, a permutation) when one is given."""
    if token_order is None:
        return torch.cat(tensors)
    first = tensors[0]
    out = first.new_empty((sum(t.shape[0] for t in tensors), *first.shape[1:]))
    start = 0
    for tensor in tensors:
        out.index_copy_(0, token_order[start : start + tensor.shape[0]], tensor)
        start += tensor.shape[0]
    return out


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
        self.ready_barrier = threading.Barrier(NUM_TBO_UBATCHES + 1)

        # Persistent ubatch worker pool, one thread a micro-batch.
        self._num_workers = NUM_TBO_UBATCHES
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
                # Drop the closure's tensors before the worker waits for its
                # next job, including when subsequent steps do not use TBO.
                job = None
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

        forward_contexts = []
        ub_inputs = []

        # A split decode's micro-batches run their tokens in this order; a
        # prefill's run them in place.
        token_order = ctx.attn_metadata.ubatch_token_order
        if token_order is not None:
            input_ids = input_ids[token_order]
            positions = positions[..., token_order]

        for i, ub_slice in enumerate(ctx.ubatch_slices):
            ub_ctx = self._make_ubatch_context(
                original_ctx,
                ub_slice,
                i,
                positions,
                token_order,
                dp_metadata=ub_dp_metadata[i] if ub_dp_metadata is not None else None,
                running_tokens_across_dp=self._ub_tokens_across_dp(ctx, N, i),
            )
            forward_contexts.append(ub_ctx)
            ub_token_slice = (
                input_ids[ub_slice.token_slice],
                positions[..., ub_slice.token_slice],
            )
            # Empty ubatch would return immediately, leaving the partner
            # wedged at its next yield (see TBOContext.__exit__ comment).
            # The new partner.done short-circuit covers this case too, but
            # an empty ubatch is always a metadata bug upstream — assert
            # loudly rather than silently waste a TBO split.
            assert ub_token_slice[0].numel() > 0, (
                f"ubatch {i} produced an empty token slice "
                f"(ts={ub_slice.token_slice}, rs={ub_slice.request_slice}); "
                "check prefill_tbo_vote / cut_prefill_ubatches / "
                "cut_decode_ubatches."
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
        return self._merge_ubatch_outputs(sorted_results, token_order)

    @staticmethod
    def _merge_ubatch_outputs(
        outputs: list[UBatchModelOutput], token_order: torch.Tensor | None = None
    ) -> UBatchModelOutput:
        """The micro-batches' Tensor or Eagle3 aux outputs, end to end, put
        back in the step's packed token order (``token_order``) when they ran
        in another."""
        first = outputs[0]
        if isinstance(first, torch.Tensor):
            if not all(isinstance(output, torch.Tensor) for output in outputs):
                raise TypeError("TBO ubatch outputs must have matching structures")
            return _packed_tokens(outputs, token_order)

        if not all(isinstance(output, tuple) for output in outputs):
            raise TypeError("TBO ubatch outputs must have matching structures")

        hidden_states = _packed_tokens([output[0] for output in outputs], token_order)
        num_aux = len(first[1])
        if not all(len(output[1]) == num_aux for output in outputs):
            raise ValueError("TBO ubatch aux output counts must match")
        aux_hidden_states = [
            _packed_tokens([output[1][idx] for output in outputs], token_order)
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

        Each ubatch's per-rank tokens are known without a collective of its
        own (`_ub_tokens_across_dp`).
        """
        if ctx.dp_metadata is None:
            return None
        return [
            DPMetadata(np.asarray(self._ub_tokens_across_dp(ctx, N, i), np.int32))
            for i in range(N)
        ]

    @staticmethod
    def _ub_tokens_across_dp(ctx: ForwardContext, N: int, i: int):
        """Ubatch ``i``'s per-rank token counts, or None if none were reduced.
        A decode's every rank cuts the same bucket, so each runs the ubatch's
        `running_tokens`, as does a group of one; a prefill's peers' are the
        step's sync's (`ub_tokens_across_dp`, sized by `prefill_tbo_vote` as
        the slices are cut).

        Passed down because inside a ubatch only this rank's split is in reach.
        """
        running_tokens = ctx.ubatch_slices[i].running_tokens
        if not ctx.context.is_prefill:
            step_rows = ctx.context.running_tokens_across_dp
            if step_rows is None:
                return None
            return (running_tokens,) * len(step_rows)
        rows = ctx.ub_tokens_across_dp
        if rows is None or len(rows) != N:
            return None
        return (running_tokens,) if len(rows[i]) == 1 else rows[i]

    @staticmethod
    def _ubatch_scheduled_tokens_gpu(
        step_scheduled_tokens_gpu: torch.Tensor | None,
        token_start: int,
        token_stop: int,
    ) -> torch.Tensor | None:
        """A micro-batch's scheduled token count on the device: the step's
        within its own tokens, ``[token_start, token_stop)``. A step's tokens
        past its scheduled ones trail it, so that is the micro-batch's own.
        Clamped here, not by the consumer: aiter's sorter reads the count as
        the row count."""
        if step_scheduled_tokens_gpu is None:
            return None
        return (step_scheduled_tokens_gpu - token_start).clamp_(
            0, token_stop - token_start
        )

    def _make_ubatch_context(
        self,
        ctx: ForwardContext,
        ub_slice: UBatchSlice,
        ubatch_idx: int,
        positions: torch.Tensor,
        token_order: torch.Tensor | None,
        dp_metadata=None,
        running_tokens_across_dp: tuple[int, ...] | None = None,
    ) -> ForwardContext:
        """Build a ForwardContext for a single micro-batch, over the metadata
        the step built for it (`AttentionMetaData.ubatches`). ``positions`` are
        the step's in its micro-batches' order, ``token_order`` that order."""
        ub_attn = ctx.attn_metadata.ubatches[ubatch_idx]

        # Split Context. Its tokens past its own requests' are padding.
        ts = ub_slice.token_slice
        ub_scheduled_bs = ub_slice.scheduled_bs(ctx.context.scheduled_bs)
        ub_token_order = None if token_order is None else token_order[ts]
        scheduled_tokens_gpu = ub_attn.scheduled_tokens_gpu
        if scheduled_tokens_gpu is None:
            scheduled_tokens_gpu = self._ubatch_scheduled_tokens_gpu(
                ctx.context.scheduled_tokens_gpu,
                ts.start,
                ub_slice.scheduled_token_slice.stop,
            )
        ub_context = Context(
            positions=positions[..., ts],
            is_prefill=ctx.context.is_prefill,
            is_dummy_run=ctx.context.is_dummy_run,
            scheduled_bs=ub_scheduled_bs,
            scheduled_tokens=ub_slice.scheduled_tokens,
            running_bs=ub_slice.running_bs,
            running_tokens=ub_slice.running_tokens,
            # Each rank runs its own micro-batch tokens exactly when it runs
            # the step's own: a prefill's are gathered by every rank's count.
            running_tokens_are_unified=ctx.context.running_tokens_are_unified,
            running_tokens_across_dp=running_tokens_across_dp,
            is_draft=ctx.context.is_draft,
            # Carry over per-ubatch slice of input_ids for hash MoE (PCP+TBO mode).
            # run_model stores local (1/pcp) ids; each ubatch takes its token_slice,
            # and the MoE hash-routes its local rows by them before any gather.
            input_ids=(
                None
                if ctx.context.input_ids is None
                else ctx.context.input_ids[
                    ts if ub_token_order is None else ub_token_order
                ]
            ),
            ubatch_token_offset=ts.start,
            ubatch_token_order=ub_token_order,
            scheduled_tokens_gpu=scheduled_tokens_gpu,
            pcp_split=ctx.context.pcp_split,
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
