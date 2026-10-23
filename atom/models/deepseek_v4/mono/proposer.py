# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Reserved joint V4 DSpark route; native DSpark stays in spec_decode.

Installed before warmup/capture, this proposer keeps the native loaded model,
aux buffers and anchor staging. Only joint publication and execution differ.
"""

import torch
from torch.profiler import record_function

from atom.model_ops.attentions.deepseek_v4_attn import (
    DeepseekV4AttentionMetadataBuilder,
)
from atom.mono.runtime.consensus import MonoUnsupported, bind_agreed
from atom.spec_decode.dspark_proposer import DSparkProposer
from atom.utils.forward_context import get_forward_context, set_forward_context


class MonoMetadataBuilder(DeepseekV4AttentionMetadataBuilder):
    """Only joint DSpark retires draft readers before native state copies."""

    def __init__(self, native, session):
        self.__dict__.update(vars(native))
        self._mono_session = session

    def build(self, batch, running_bs, running_tokens, max_seqlen_q):
        operations = batch.state_maintenance_ops

        def maintain():
            if operations.relocations:
                self.relocate_state_slots(operations.relocations)
            if operations.checkpoint_stores or operations.checkpoint_restores:
                self.execute_paged_state_copies(
                    operations.checkpoint_stores, operations.checkpoint_restores
                )

        self._mono_session.run_state_maintenance(maintain, empty=operations.empty)
        if batch.total_tokens_num_prefill > 0:
            return self.prepare_prefill(batch, running_bs)
        return self.prepare_decode(batch, running_bs, running_tokens, max_seqlen_q)


class MonoDSparkProposer(DSparkProposer):
    def __init__(self, native, session):
        if native._with_draft or native.block._cuda_graphs:
            raise MonoUnsupported(
                "joint V4 draft installation requires an uncaptured inline draft"
            )
        # Aux hooks close over the fixed tensors, not the proposer. Keep those
        # tensors and rebind every DraftGraph callback to the new owner.
        self.__dict__.update(vars(native))
        self._mono_session = session
        self.block.forward = self._block_backbone
        self.block.epilogue = self._block_head
        self.block.warmup_inputs = self._block_warmup_inputs

    @torch.inference_mode()
    def warmup_draft_graphs(self, build_context, stream, end_trace=None) -> None:
        mono = self._mono_session
        capture = self.block.will_capture
        mono.retire_draft_captures_for_startup()
        self.block.release_graphs()
        ladder = sorted(self.runner.capture_sizes)
        # The target pads capacity; the joint draft only writes real requests.
        # Prepare intermediate real widths too, including for eager deployments.
        capacity = len(mono.draft_runner._request_destinations)
        sizes = range(1, min(capacity, ladder[-1]) + 1) if ladder else ()
        for bs in sizes:
            target_bs = next(size for size in ladder if size >= bs)
            attn_metadata, context = build_context(bs=target_bs)
            context.is_draft = True
            context.is_prefill = False
            context.scheduled_bs = bs
            context.running_tokens_are_unified = True
            context.scheduled_tokens = context.running_tokens = (
                bs * self.draft_tokens_per_seq
            )
            set_forward_context(
                attn_metadata=attn_metadata,
                atom_config=self.config,
                context=context,
                num_tokens=context.running_tokens,
            )
            mono.warmup_draft_block(self, bs, get_forward_context(), capture=capture)
            if end_trace is not None:
                step = self.block.name.rsplit(".", 1)[-1].lstrip("_")
                end_trace(f"draft_{step}_bs_{bs}")

    def _block_head(self, out, running_bs, *, anchor_ids, **_):
        """Validate real rows before the native LM head and Markov sampler."""
        mono = self._mono_session

        def validate():
            if mono.disabled:
                raise MonoUnsupported("joint draft head requires a live session owner")
            batch = get_forward_context().context.scheduled_bs
            width = self.draft_tokens_per_seq
            if (
                not isinstance(out, (tuple, list))
                or len(out) != 2
                or any(not isinstance(tensor, torch.Tensor) for tensor in out)
            ):
                raise MonoUnsupported("joint draft head requires backbone outputs")
            normed, residual = out
            if (
                batch <= 0
                or residual.ndim != 3
                or residual.shape[:2] != (batch, width)
                or normed.shape != (batch * width, residual.shape[-1])
                or normed.dtype != residual.dtype
                or normed.device != residual.device
                or anchor_ids.ndim != 1
                or anchor_ids.numel() < batch
                or anchor_ids.dtype not in (torch.int32, torch.int64)
                or anchor_ids.device != residual.device
            ):
                raise MonoUnsupported(
                    "joint draft head layout differs from real requests"
                )

        try:
            bind_agreed(validate, mono.group)
            # A padded target does not enlarge the joint draft's real outputs.
            return self.model.head_and_sample(
                out, anchor_ids[: out[1].shape[0]], self.draft_tokens_per_seq
            )
        except Exception:
            # Never reuse state after a possibly submitted head/collective.
            mono.disabled = True
            raise

    def _block_backbone(self, running_bs, *, anchor_ids, anchor_positions):
        real_bs = get_forward_context().context.scheduled_bs
        return self._mono_session.forward_draft_block(
            self.model.model, anchor_ids[:real_bs], anchor_positions[:real_bs]
        )

    def compute_draft_kv(self, positions, hidden_states, next_token_ids) -> None:
        """Publish native target-context writes before switching draft metadata."""
        fc = get_forward_context()
        bs = fc.context.scheduled_bs
        slots = fc.attn_metadata.state_slot_out_cpu

        def produce():
            aux = self.aux_for(hidden_states)
            if aux is None:
                raise RuntimeError(
                    "DSpark mono context requires target auxiliary states"
                )
            hidden = torch.cat(aux, dim=-1)
            with record_function(f"draft_kv[bs={bs} tok={hidden.shape[0]}]"):
                self.model.write_context_kv(hidden, positions)

        self._mono_session.write_draft_context(
            produce,
            None if slots is None else slots[:bs],
            expected_rows=bs,
            positions=positions,
            cu_seqlens=fc.attn_metadata.cu_seqlens_q[: bs + 1],
            write_per_batch=min(
                stage.write_per_batch for stage in self.model.model.mtp
            ),
        )

    def propose(
        self,
        target_token_ids,
        target_positions,
        target_hidden_states,
        num_reject_tokens,
        next_token_ids,
        last_token_indices,
    ):
        """Keep native anchor staging/sampling; replace only the joint backbone."""
        fc = get_forward_context()
        context, metadata = fc.context, fc.attn_metadata
        mono = self._mono_session
        # The sole pre-KV dummy pass initializes native draft kernels. Snapshot
        # this before replacing the target's phase and shape below.
        native_initialization = mono.uses_native_draft_initialization(context)
        context.is_draft = True

        def require_aux():
            if self.aux_for(target_hidden_states) is None:
                raise RuntimeError("DSpark requires captured target auxiliary states")

        if native_initialization:
            require_aux()
        else:

            def validate_aux():
                if mono.disabled:
                    raise MonoUnsupported("joint draft requires a live session owner")
                require_aux()

            try:
                bind_agreed(validate_aux, mono.group)
            except Exception:
                mono.disabled = True
                raise

        bs, running_bs = next_token_ids.shape[0], context.running_bs
        if not native_initialization:
            slots = metadata.state_slot_out_cpu
            mono.consume_draft_context(
                None if slots is None else slots[:bs],
                expected_rows=bs,
                anchor_rows=last_token_indices,
                target_positions=target_positions,
            )
        staged = self._stage_anchors(
            running_bs, next_token_ids, target_positions, last_token_indices
        )
        width = self.draft_tokens_per_seq
        self.model.prepare_block(
            self.runner.attn_metadata_builder, width, bs, running_bs
        )
        context.is_prefill = False
        self._publish_draft_shape(
            fc,
            scheduled_tokens=bs * width,
            running_tokens=running_bs * width,
            running_tokens_are_unified=True,
        )
        if not native_initialization:

            def refuse_stale_graph():
                if self.block.is_captured(running_bs):
                    raise MonoUnsupported("joint draft cannot replay a native graph")

            bind_agreed(refuse_stale_graph, mono.group)

        label = self.block.label(bs, running_bs)
        with record_function(f"propose_dspark[{label} T={width}]"):
            if native_initialization:
                out = self.model.block_backbone(
                    staged["anchor_ids"],
                    self._block_positions(running_bs, staged["anchor_positions"]),
                    width,
                )
                tokens, confidence = self._block_head(out, running_bs, **staged)
            else:
                # The session graph already contains the native head/sampler;
                # DraftGraph's epilogue must not run a second time.
                tokens, confidence = mono.forward_draft_block(
                    self.model.model,
                    staged["anchor_ids"][:bs],
                    staged["anchor_positions"][:bs],
                    sampler=self.model.head_and_sample,
                )
        tokens = tokens[:bs, : self.mtp_k]
        if confidence is not None:
            confidence = confidence[:bs]
        if self.verify_scheduler is not None:
            with record_function(f"dspark_sched[bs={bs}]"):
                self.verify_scheduler.set_last_ell(
                    self.verify_scheduler.compute_ell(confidence[:, : self.mtp_k])
                    if confidence is not None
                    else None
                )
        return tokens
