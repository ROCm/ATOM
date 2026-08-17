import logging

import torch

from atom.plugin.sglang.runtime import SGLangForwardBatchMetadata
from atom.plugin.sglang.tbo.adapter import prepare_sglang_ubatch
from atom.plugin.sglang.tbo.sglang_tbo_compat_patches import (
    bind_sglang_tbo_child_attn_backend,
)
from atom.utils.forward_context import (
    Context,
    ForwardContext,
    _forward_context_local,
    get_forward_context,
)
from atom.utils.tbo.ubatch_wrapper import UBatchModelOutput, UBatchWrapper
from atom.utils.tbo.ubatching import make_tbo_contexts

logger = logging.getLogger("atom.plugin.sglang.tbo")


class SGLangPluginUBatchWrapper(UBatchWrapper):
    """Run ATOM TBO while binding each SGLang child batch in its worker."""

    def _make_sglang_ubatch_context(
        self,
        ctx: ForwardContext,
        ub_slice,
        positions: torch.Tensor,
        input_ids: torch.Tensor,
        padded_bs: int,
        *,
        ub_graph_bs: int | None,
        dp_metadata,
    ) -> ForwardContext:
        ub_num_reqs = ub_slice.request_slice.stop - ub_slice.request_slice.start
        ub_num_tokens = ub_slice.token_slice.stop - ub_slice.token_slice.start
        graph_bs = (
            ub_graph_bs
            if ub_graph_bs is not None
            else ub_num_tokens if ctx.context.is_prefill else padded_bs
        )
        ub_context = Context(
            positions=positions,
            is_prefill=ctx.context.is_prefill,
            is_dummy_run=ctx.context.is_dummy_run,
            batch_size=ub_num_reqs,
            graph_bs=graph_bs,
            is_draft=ctx.context.is_draft,
            dp_uniform_decode=ctx.context.dp_uniform_decode,
            forward_mode=ctx.context.forward_mode,
            input_ids=input_ids,
            ubatch_token_offset=ub_slice.token_slice.start,
        )
        return ForwardContext(
            attn_metadata=ctx.attn_metadata,
            no_compile_layers=ctx.no_compile_layers,
            kv_cache_data=ctx.kv_cache_data,
            context=ub_context,
            dp_metadata=dp_metadata if dp_metadata is not None else ctx.dp_metadata,
            spec_decode_metadata=None,
            ubatch_slices=None,
            ub_max_tokens_across_dp=None,
            main_stream=ctx.main_stream,
            in_hipgraph=ctx.in_hipgraph,
            cudagraph_runtime_mode=ctx.cudagraph_runtime_mode,
            batch_descriptor=ctx.batch_descriptor,
        )

    @staticmethod
    def _trim_ubatch_output(
        output: UBatchModelOutput, num_tokens: int
    ) -> UBatchModelOutput:
        if torch.is_tensor(output):
            return output[:num_tokens]
        hidden_states, aux_hidden_states = output
        return (
            hidden_states[:num_tokens],
            [aux[:num_tokens] for aux in aux_hidden_states],
        )

    def forward_with_sglang_children(
        self,
        *,
        child_forward_batches: list,
        save_kv_cache: bool = True,
    ) -> UBatchModelOutput:
        """Execute SGLang's padded children and merge only their real outputs.

        This follows the current native ``UBatchWrapper._run_ubatches`` worker
        and synchronization lifecycle. The plugin-specific differences are the
        child tensors, child metadata binding, and logical output trimming.
        """

        ctx = get_forward_context()
        if ctx.ubatch_slices is None:
            raise RuntimeError("SGLang TBO execution requires ubatch_slices")
        self._ensure_comm_stream()
        num_ubatches = len(ctx.ubatch_slices)
        if len(child_forward_batches) != num_ubatches:
            raise RuntimeError(
                "SGLang TBO child count does not match ubatch slice count: "
                f"children={len(child_forward_batches)}, slices={num_ubatches}"
            )

        from sglang.srt.layers.attention.tbo_backend import TboAttnBackend
        from sglang.srt.model_executor.forward_context import get_attn_backend

        active_attn_backend = get_attn_backend()
        if not isinstance(active_attn_backend, TboAttnBackend):
            raise TypeError(
                "SGLang ATOM TBO requires an active TboAttnBackend, got "
                f"{type(active_attn_backend).__name__}"
            )
        child_attn_backends = active_attn_backend.children
        if len(child_attn_backends) != num_ubatches:
            raise RuntimeError(
                "SGLang TBO attention child count does not match ubatches: "
                f"backends={len(child_attn_backends)}, slices={num_ubatches}"
            )

        compute_stream = torch.cuda.current_stream()
        ub_dp_metadata = self._make_ubatch_dp_metadata(ctx, num_ubatches)
        forward_contexts = []
        ub_inputs = []
        ub_output_lengths = []

        for idx, ub_slice in enumerate(ctx.ubatch_slices):
            adapted = prepare_sglang_ubatch(
                ub_slice,
                child_forward_batches[idx],
                is_prefill=ctx.context.is_prefill,
                full_graph_bs=ctx.context.graph_bs,
                ubatch_idx=idx,
                num_ubatches=num_ubatches,
                ub_max_tokens_across_dp=ctx.ub_max_tokens_across_dp,
            )
            ub_context = self._make_sglang_ubatch_context(
                ctx,
                ub_slice,
                adapted.positions,
                adapted.input_ids,
                adapted.padded_bs,
                ub_graph_bs=adapted.graph_bs,
                dp_metadata=(
                    ub_dp_metadata[idx] if ub_dp_metadata is not None else None
                ),
            )
            forward_contexts.append(ub_context)
            ub_inputs.append((adapted.input_ids, adapted.positions))
            ub_output_lengths.append(adapted.num_tokens)

        tbo_ctxs = make_tbo_contexts(
            num_micro_batches=num_ubatches,
            compute_stream=compute_stream,
            comm_stream=self.comm_stream,
            forward_contexts=forward_contexts,
            ready_barrier=self.ready_barrier,
        )

        results: list[tuple[int, UBatchModelOutput]] = []
        errors: list[Exception | None] = [None] * num_ubatches
        device = ub_inputs[0][0].device
        if num_ubatches > self._num_workers:
            raise RuntimeError(
                f"TBO needs {num_ubatches} workers but pool has {self._num_workers}"
            )
        self._ensure_workers(device)

        def _make_job(idx):
            @torch.inference_mode()
            def _job():
                try:
                    child_metadata = SGLangForwardBatchMetadata.build(
                        child_forward_batches[idx],
                        save_kv_cache=save_kv_cache,
                    )
                    ub_input_ids, ub_positions = ub_inputs[idx]
                    with bind_sglang_tbo_child_attn_backend(  # noqa: SIM117
                        child_attn_backends[idx]
                    ):
                        with SGLangForwardBatchMetadata.bind(child_metadata):
                            with tbo_ctxs[idx]:
                                model_output = self.model(ub_input_ids, ub_positions)
                    model_output = self._validate_ubatch_output(model_output)
                    results.append(
                        (
                            idx,
                            self._trim_ubatch_output(
                                model_output, ub_output_lengths[idx]
                            ),
                        )
                    )
                except Exception as exc:
                    logger.exception("[SGLang ATOM TBO] ubatch %d crashed", idx)
                    errors[idx] = exc

            return _job

        saved_ctx = getattr(_forward_context_local, "ctx", None)
        _forward_context_local.ctx = None
        try:
            for idx in range(num_ubatches):
                self._worker_job_done[idx].clear()
                self._worker_jobs[idx] = _make_job(idx)
                self._worker_job_ready[idx].set()

            self.ready_barrier.wait()
            tbo_ctxs[0].cpu_wait_event.set()
            for idx in range(num_ubatches):
                self._worker_job_done[idx].wait()
        finally:
            _forward_context_local.ctx = saved_ctx

        for error in errors:
            if error is not None:
                raise error

        sorted_results = [value for _, value in sorted(results)]
        return self._concat_ubatch_outputs(sorted_results)
