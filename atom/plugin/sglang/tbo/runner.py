from __future__ import annotations

import logging
import os
import threading
import traceback
from dataclasses import replace
from typing import Any, Optional

import torch
from torch import nn

from atom.utils.forward_context import AttentionMetaData, Context, ForwardContext
from atom.utils.tbo.ubatching import make_tbo_contexts

logger = logging.getLogger("atom.plugin.sglang.tbo")


def _debug_enabled() -> bool:
    return os.environ.get("ATOM_SGLANG_TBO_DEBUG", "0").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _disabled_by_env() -> bool:
    return os.environ.get("ATOM_SGLANG_TBO_DISABLE", "0").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


class SGLangAtomTBORunner:
    """Run ATOM's model on SGLang TBO child batches using ATOM TBO streams.

    SGLang still owns scheduling, child ForwardBatch construction, and attention
    metadata.  This runner only supplies the dual-thread execution context that
    lets ATOM MoE/MORI kernels use ``atom.utils.tbo`` yield points.
    """

    def __init__(self, model: nn.Module):
        self.model = model
        self.comm_stream: Optional[torch.cuda.Stream] = None
        self.ready_barrier = threading.Barrier(3)

    def can_run(
        self,
        *,
        atom_config: Any,
        forward_batch: Any,
        model_inputs: dict[str, Any],
        uses_context_only_forward: bool,
        pp_proxy_tensors: Any,
    ) -> bool:
        if _disabled_by_env():
            return False
        if not getattr(atom_config, "enable_tbo", False):
            return False
        if getattr(atom_config, "enable_dp_attention", False) and not getattr(
            atom_config, "enable_expert_parallel", False
        ):
            return False
        if not uses_context_only_forward:
            return False
        if pp_proxy_tensors is not None:
            return False
        if model_inputs.get("inputs_embeds") is not None:
            return False
        if model_inputs.get("intermediate_tensors") is not None:
            return False

        forward_mode = getattr(forward_batch, "forward_mode", None)
        if forward_mode is None:
            return False
        is_decode = hasattr(forward_mode, "is_decode") and forward_mode.is_decode()
        if is_decode and not getattr(atom_config, "enable_tbo_decode", False):
            return False
        if is_decode:
            if getattr(forward_batch, "next_token_logits_buffer", None) is not None:
                return False
            is_capturing = (
                torch.cuda.is_available()
                and hasattr(torch.cuda, "is_current_stream_capturing")
                and torch.cuda.is_current_stream_capturing()
            )
            if is_capturing:
                return False
        elif hasattr(forward_mode, "is_extend_without_speculative"):
            if not forward_mode.is_extend_without_speculative():
                return False
        elif not forward_mode.is_prefill():
            return False

        children = getattr(forward_batch, "tbo_children", None)
        if children is None or len(children) != 2:
            return False
        if any(child is None or int(child.batch_size) <= 0 for child in children):
            return False

        sgl_ctx = self._get_sglang_forward_context()
        if sgl_ctx is None:
            return False
        attn_backend = getattr(sgl_ctx, "attn_backend", None)
        return len(getattr(attn_backend, "children", []) or []) == 2

    def run(
        self,
        *,
        atom_config: Any,
        model_inputs: dict[str, Any],
        forward_batch: Any,
        metadata: Any,
        set_atom_forward_context,
        reset_atom_forward_context,
    ):
        self._ensure_comm_stream()

        positions = model_inputs["positions"]
        parent_sgl_ctx = self._get_sglang_forward_context()
        if parent_sgl_ctx is None:
            raise RuntimeError("[SGL+ATOM TBO] missing active SGLang ForwardContext")

        try:
            set_atom_forward_context(atom_config, forward_batch, positions)
            from atom.utils.forward_context import get_forward_context

            parent_atom_ctx = get_forward_context()

            attn_backend = parent_sgl_ctx.attn_backend
            sgl_child_contexts = [
                replace(parent_sgl_ctx, attn_backend=child_backend)
                for child_backend in attn_backend.children
            ]

            children = list(forward_batch.tbo_children)
            atom_child_contexts = [
                self._make_atom_child_context(parent_atom_ctx, child)
                for child in children
            ]
            restore_callbacks = [
                self._make_sglang_restore_callback(child_ctx)
                for child_ctx in sgl_child_contexts
            ]

            output = self._run_children(
                model_inputs=model_inputs,
                children=children,
                metadata=metadata,
                atom_child_contexts=atom_child_contexts,
                restore_callbacks=restore_callbacks,
            )
            return self._merge_outputs(output, children, int(positions.shape[0]))
        finally:
            self._restore_sglang_forward_context(parent_sgl_ctx)
            reset_atom_forward_context()

    def _ensure_comm_stream(self) -> None:
        if self.comm_stream is None:
            self.comm_stream = torch.cuda.Stream()

    @staticmethod
    def _get_sglang_forward_context():
        from sglang.srt.model_executor import forward_context as sgl_forward_context

        if not sgl_forward_context.has_forward_context():
            return None
        return sgl_forward_context.get_forward_context()

    @staticmethod
    def _make_sglang_restore_callback(child_ctx):
        from sglang.srt.model_executor import forward_context as sgl_forward_context

        def _restore():
            sgl_forward_context.set_forward_context(child_ctx)

        return _restore

    @staticmethod
    def _restore_sglang_forward_context(parent_ctx) -> None:
        from sglang.srt.model_executor import forward_context as sgl_forward_context

        # Avoid leaving the last child backend visible before SGLang's outer
        # ModelRunner context manager restores its own value.
        sgl_forward_context.set_forward_context(parent_ctx)

    @staticmethod
    def _make_atom_child_context(parent_ctx: ForwardContext, child_batch: Any):
        positions = child_batch.positions
        forward_mode = child_batch.forward_mode
        is_prefill = forward_mode.is_prefill()
        max_seqlen_q = 1 if forward_mode.is_decode_or_idle() else 0
        attn_metadata = AttentionMetaData(max_seqlen_q=max_seqlen_q)
        num_tokens = int(positions.shape[0])
        batch_size = int(child_batch.batch_size)
        graph_bs = num_tokens if is_prefill else batch_size

        return ForwardContext(
            attn_metadata=attn_metadata,
            no_compile_layers=parent_ctx.no_compile_layers,
            kv_cache_data=parent_ctx.kv_cache_data,
            context=Context(
                positions=positions,
                is_prefill=is_prefill,
                is_dummy_run=False,
                batch_size=batch_size,
                graph_bs=graph_bs,
            ),
            dp_metadata=parent_ctx.dp_metadata,
            spec_decode_metadata=None,
            ubatch_slices=None,
        )

    def _run_children(
        self,
        *,
        model_inputs: dict[str, Any],
        children: list[Any],
        metadata: Any,
        atom_child_contexts: list[ForwardContext],
        restore_callbacks: list[Any],
    ):
        compute_stream = torch.cuda.current_stream()
        tbo_ctxs = make_tbo_contexts(
            num_micro_batches=2,
            compute_stream=compute_stream,
            comm_stream=self.comm_stream,
            forward_contexts=atom_child_contexts,
            ready_barrier=self.ready_barrier,
            restore_callbacks=restore_callbacks,
        )

        input_ids = model_inputs["input_ids"]
        device = input_ids.device
        results: list[tuple[int, Any]] = []
        errors: list[Optional[BaseException]] = [None, None]
        metadata_cls = type(metadata)
        save_kv_cache = getattr(metadata, "save_kv_cache", True)

        @torch.inference_mode()
        def _thread_main(idx: int) -> None:
            try:
                torch.cuda.set_device(device)
                child = children[idx]
                child_metadata = metadata_cls.build(
                    child,
                    pp_proxy_tensors=None,
                    save_kv_cache=save_kv_cache,
                )
                child_inputs = dict(model_inputs)
                child_inputs["input_ids"] = child.input_ids
                child_inputs["positions"] = child.positions
                child_inputs["intermediate_tensors"] = None
                child_inputs["inputs_embeds"] = None

                with metadata_cls.bind(child_metadata):
                    with tbo_ctxs[idx]:
                        result = self.model(**child_inputs)
                results.append((idx, result))
            except BaseException as exc:
                traceback.print_exc()
                errors[idx] = exc

        threads = [
            threading.Thread(target=_thread_main, args=(idx,), daemon=False)
            for idx in range(2)
        ]
        for thread in threads:
            thread.start()

        self.ready_barrier.wait()
        tbo_ctxs[0].cpu_wait_event.set()

        for thread in threads:
            thread.join()

        for exc in errors:
            if exc is not None:
                raise exc

        if len(results) != 2:
            raise RuntimeError(
                f"[SGL+ATOM TBO] expected 2 child outputs, got {len(results)}"
            )
        if _debug_enabled():
            token_ranges = [
                getattr(child, "tbo_parent_token_range", None) for child in children
            ]
            forward_mode = getattr(children[0], "forward_mode", None)
            logger.info(
                "[SGL+ATOM TBO] completed %s child forwards: %s",
                forward_mode,
                token_ranges,
            )
        return [value for _, value in sorted(results)]

    def _merge_outputs(self, outputs: list[Any], children: list[Any], original_len: int):
        first = outputs[0]
        if first is None:
            return None
        if torch.is_tensor(first):
            return self._merge_tensors(outputs, children, original_len)
        if isinstance(first, tuple):
            return tuple(
                self._merge_outputs([output[i] for output in outputs], children, original_len)
                for i in range(len(first))
            )
        if isinstance(first, list):
            return [
                self._merge_outputs([output[i] for output in outputs], children, original_len)
                for i in range(len(first))
            ]
        raise TypeError(f"[SGL+ATOM TBO] unsupported model output type: {type(first)}")

    @staticmethod
    def _merge_tensors(
        outputs: list[torch.Tensor],
        children: list[Any],
        original_len: int,
    ) -> torch.Tensor:
        merged = outputs[0].new_empty((original_len, *outputs[0].shape[1:]))
        for output, child in zip(outputs, children, strict=True):
            start, end = child.tbo_parent_token_range
            real_len = end - start
            merged[start:end] = output[:real_len]
        return merged
