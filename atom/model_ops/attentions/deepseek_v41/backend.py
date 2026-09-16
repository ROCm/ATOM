# SPDX-License-Identifier: MIT
"""ATOM scheduling adapter for the eager CSA2 paged runtime."""

from types import SimpleNamespace

import numpy as np
import torch

from atom.model_engine.engram_runtime import EngramInputPreparer
from atom.model_engine.kv_block import STATE_SLOT_CLASS
from atom.model_engine.state_runtime import StateTransfer
from atom.model_ops.attentions.backends import AttentionBackend, CommonAttentionBuilder
from atom.model_ops.attentions.deepseek_v4_attn import (
    DeepseekV4AttentionMetadataBuilder,
)
from atom.model_ops.attentions.pool_layout.sub_pool_spec import page_pool, state_pool
from atom.model_ops.attentions.pool_layout.v41_pool_geometry import V41PoolGeometry
from atom.models.deepseek_v41.config import AttentionMode, build_attention_topology
from atom.utils.forward_context import AttentionMetaData, AttnState, Context
from tests.models.deepseek_v41.cache_visibility_snapshot import (  # DIAGNOSTIC179
    CacheVisibilitySnapshot,
)

from .cache import PagedAttentionCache
from .checkpoints import StateCopies
from .metadata import RequestSpan


class DeepseekV41Backend(AttentionBackend):
    use_custom_all_reduce = False

    @staticmethod
    def get_name():
        return "CSA2"

    @staticmethod
    def get_builder_cls():
        return DeepseekV41MetadataBuilder


class DeepseekV41MetadataBuilder(CommonAttentionBuilder):
    # Reuse V4's publisher and staging contract, including fixed addresses and
    # running_bs padding. Only pool-slot -> physical-row geometry differs.
    _stage = DeepseekV4AttentionMetadataBuilder._stage
    _populate_state_slot_mappings = (
        DeepseekV4AttentionMetadataBuilder._populate_state_slot_mappings
    )

    @staticmethod
    def _physical_slots(pool_slots):
        # V4's unified plane reverses pool slots. V4.1's EntryMajorArena uses
        # the scheduler's slot index directly.
        return pool_slots

    def __init__(self, model_runner):
        self.block_size = model_runner.block_size
        super().__init__(model_runner)
        model_runner.forward_vars.update(
            DeepseekV4AttentionMetadataBuilder._state_slot_buffers(
                self.max_bs, self.device, read_side=False
            )
        )
        self.config = model_runner.config.hf_config
        topology = build_attention_topology(self.config)[
            : self.config.num_hidden_layers
        ]
        speculative = model_runner.config.speculative_config
        num_drafts = 0 if speculative is None else speculative.num_speculative_tokens
        self.geometry = V41PoolGeometry(
            len(topology) + (self.config.num_nextn_predict_layers if num_drafts else 0),
            tuple(
                (spec.layer_id, spec.ratio)
                for spec in topology
                if spec.mode == AttentionMode.FULL
            ),
            self.block_size,
            self.config.sliding_window,
            self.config.head_dim,
            self.config.index_head_dim,
            self.config.engram_max_ngram_size - 1,
            packed=model_runner.config.kv_cache_dtype == "fp4",
            speculative_tokens=num_drafts,
        )
        self.cache = self.copies = self.engram = None
        # DIAGNOSTIC179: see the snapshot module. Returns None unless
        # ATOM_DSPARK_CACHE_SNAPSHOT is set, so production builds nothing.
        self._cache_snapshot = CacheVisibilitySnapshot.from_env(
            torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        )
        self.dummy_weights = bool(model_runner.config.load_dummy)
        if not self.dummy_weights and self.config.engram_layer_ids:
            self.engram = EngramInputPreparer.from_checkpoint(
                model_runner.config.model,
                self.config,
                self.max_num_batched_tokens,
                self.device,
            )

    def sub_pool_specs(self):
        return [
            page_pool(self.geometry.page_bytes),
            state_pool(STATE_SLOT_CLASS, self.geometry.state_bytes, entries_per_req=1),
        ]

    def state_transfer(self):
        # Includes the tie policy: exact prefix images must belong to the same
        # computation even when their byte geometry happens to agree.
        return StateTransfer.copy(
            f"{self.geometry.layout_id}:ties={self.config.index_topk_tie_break}"
        )

    def checkpoint_image_bytes(self):
        return self.geometry.state_bytes

    def allocate_kv_cache_tensors(self, *, blocks, buf):
        self.num_blocks = blocks
        return {}

    def allocate_per_req_cache(self, entries):
        self.cache = PagedAttentionCache(
            self.geometry, self.num_blocks, entries[STATE_SLOT_CLASS], self.device
        )
        self.copies = StateCopies(
            self.cache, self.model_runner.state_runtime.checkpoint_spec, self.max_bs
        )
        return {}

    def state_entry_views(self, slot):
        return [self.copies.entry(slot)]

    def relocate_state_slots(self, pairs):
        self.copies.relocate(pairs)

    def execute_paged_state_copies(self, stores, restores):
        self.copies.execute(stores, restores)

    def warmup_per_req_cache(self):
        self.copies.warmup()

    def release_kv_pools(self):
        self.cache = self.copies = None

    def close(self):
        if self.engram is not None:
            self.engram.close()
            self.engram = None
        self.release_kv_pools()

    def _prepare(
        self,
        batch,
        running_bs,
        running_tokens,
        *,
        tentative=False,
        start_positions=None,
    ):
        spans, offset, next_page = [], 0, 0
        slots = batch.state_slots_committed
        if not batch.is_dummy_run and len(slots) != batch.total_seqs_num:
            raise ValueError("CSA2 requires a STATE slot for every scheduled request")
        for i, (request_id, length, end) in enumerate(
            zip(batch.req_ids, batch.num_scheduled_tokens, batch.context_lens)
        ):
            length, end = int(length), int(end)
            if length == 0:
                continue
            if batch.is_dummy_run:
                # Warmup uses private scratch. A dummy rank may never mutate
                # a live slot or PAGE, even when its fabricated block ID is 0.
                position = 0
                count = -(-length // self.block_size)
                blocks = tuple(range(next_page, next_page + count))
                next_page += count
                slot = len(spans)
            else:
                position = (
                    end - length if start_positions is None else int(start_positions[i])
                )
                blocks = tuple(batch.block_tables[i])
                slot = slots[i]
            spans.append(
                RequestSpan(request_id, position, offset, length, slot, blocks)
            )
            offset += length
        if offset != batch.total_tokens_num or running_tokens < offset:
            raise ValueError("CSA2 batch token spans disagree with the runner")
        cache = (
            PagedAttentionCache(
                self.geometry, max(next_page, 1), max(len(spans), 1), self.device
            )
            if batch.is_dummy_run
            else self.cache
        )
        if cache is None:
            raise RuntimeError("CSA2 cache must be allocated before serving")
        # Zero-token scheduler rows are excluded from spans. Publish in this
        # same request order, including the private dummy slots used at startup.
        state_slot_out = self._populate_state_slot_mappings(
            SimpleNamespace(state_slots_committed=[span.slot for span in spans]),
            len(spans),
            running_bs,
        )
        step = cache.begin_step(
            spans,
            tentative=tentative and not batch.is_dummy_run and bool(spans),
            buffers=self.model_runner.forward_vars,
            running_bs=running_bs,
            running_tokens=running_tokens,
            state_slot_out=state_slot_out,
        )
        positions = self.model_runner.forward_vars["positions"]
        cu = self.model_runner.forward_vars["cu_seqlens_q"].gpu[: running_bs + 1]
        metadata = AttentionMetaData(
            cu_seqlens_q=cu,
            max_seqlen_q=step.max_length,
            max_seqlen_k=max((span.end for span in spans), default=0),
            state=AttnState.DECODE if step.decode else AttnState.PREFILL_PREFIX,
        )
        metadata.cache, metadata.step = cache, step
        metadata.dummy = batch.is_dummy_run
        token_mask = np.ones(offset, dtype=np.bool_)
        for span in spans:
            data = getattr(batch, "multimodal_data", {}).get(span.request_id)
            if data is not None:
                for start, count in data.get("embedding_spans", ()):
                    first, end = max(start, span.position), min(start + count, span.end)
                    if first < end:
                        token_mask[
                            span.offset + first - span.position : span.offset
                            + end
                            - span.position
                        ] = False
        metadata.token_mask = token_mask
        metadata.image_mask = (
            torch.from_numpy(~token_mask).to(self.device).unsqueeze(0)
            if not token_mask.all()
            else None
        )
        return metadata, positions.gpu[:running_tokens]

    def prepare_prefill(self, batch, running_bs):
        return self._prepare(batch, running_bs, batch.total_tokens_num)

    def prepare_decode(self, batch, running_bs, running_tokens, max_seqlen_q):
        starts = None
        if self.geometry.speculative_tokens and not batch.is_dummy_run:
            # The scheduler reserves a full draft span, including placeholders
            # from the previous step. Ragged verification takes its head.
            starts = np.asarray(batch.context_lens) - (batch.num_spec_step + 1)
            rejected = self.model_runner.tokenID_processor.num_rejected
            if rejected is not None:
                starts = starts - rejected
        return self._prepare(
            batch,
            running_bs,
            running_tokens,
            tentative=bool(self.geometry.speculative_tokens),
            start_positions=starts,
        )

    def prepare_model_inputs(self, input_ids, metadata):
        step, cache = metadata.step, metadata.cache
        # DIAGNOSTIC179: the cache still holds only committed state here, so this
        # is the one point a DSpark run and a baseline run can be compared.
        # Off unless ATOM_DSPARK_CACHE_SNAPSHOT is set. Revert before any
        # acceptance run: grep -rn DIAGNOSTIC179 atom/
        if self._cache_snapshot is not None and not metadata.dummy:
            self._cache_snapshot.capture(cache, step, step.block_tables)
        histories = cache.prepare_state(step)
        tokens = input_ids[: step.length]
        if self.engram is not None:
            prepared = self.engram.prepare(
                step.requests,
                tokens,
                histories,
                dummy=metadata.dummy,
                token_mask=metadata.token_mask,
            )
            embeddings, histories = prepared.embeddings, prepared.histories
            if cache.pending is not None:
                for span, compressed in zip(step.requests, prepared.compressed_rows):
                    cache.pending.stage_history(span, compressed)
        else:
            width = (
                (self.config.engram_max_ngram_size - 1)
                * self.config.engram_n_heads
                * self.config.engram_head_dim
            )
            embeddings = {
                layer: torch.zeros(
                    1, step.length, width, dtype=torch.bfloat16, device=self.device
                )
                for layer in self.config.engram_layer_ids
            }
            if cache.pending is not None:
                for span in step.requests:
                    cache.pending.stage_history(span, [-1] * span.length)
        metadata.engram_embeddings = embeddings
        metadata.next_histories = histories

    def commit_speculative_state(self, metadata, last_token_indices):
        if metadata.cache.pending is not None:
            counts = last_token_indices - metadata.step.cu_seqlens_q[:-1] + 1
            metadata.cache.commit_tentative(counts)

    def build_for_cudagraph_capture(self, bs, max_q_len=1):
        # Only pure dense stages are captured. All attention warmup uses a
        # private PAGE/STATE allocation and can never alter live requests.
        if bs < 1 or max_q_len < 1 or bs * max_q_len > self.max_num_batched_tokens:
            raise ValueError("CSA2 capture shape exceeds the token buffer")
        tokens = bs * max_q_len
        batch = SimpleNamespace(
            is_dummy_run=True,
            req_ids=tuple(range(bs)),
            num_scheduled_tokens=(max_q_len,) * bs,
            context_lens=(max_q_len,) * bs,
            state_slots_committed=(),
            total_seqs_num=bs,
            total_tokens_num=tokens,
        )
        metadata, positions = self._prepare(batch, bs, tokens)
        self.prepare_model_inputs(
            self.model_runner.forward_vars["input_ids"].gpu[:tokens], metadata
        )
        return metadata, Context(
            positions=positions,
            is_prefill=False,
            is_dummy_run=True,
            scheduled_bs=bs,
            scheduled_tokens=tokens,
            running_bs=bs,
            running_tokens=tokens,
        )
