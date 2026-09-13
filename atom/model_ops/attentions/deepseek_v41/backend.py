# SPDX-License-Identifier: MIT
"""ATOM scheduling adapter for the eager CSA2 paged runtime."""

from types import SimpleNamespace

import torch

from atom.model_engine.engram_runtime import EngramInputPreparer
from atom.model_engine.kv_block import STATE_SLOT_CLASS
from atom.model_engine.state_runtime import StateTransfer
from atom.model_ops.attentions.backends import AttentionBackend, CommonAttentionBuilder
from atom.model_ops.attentions.pool_layout.sub_pool_spec import page_pool, state_pool
from atom.model_ops.attentions.pool_layout.v41_pool_geometry import V41PoolGeometry
from atom.models.deepseek_v41.config import AttentionMode, build_attention_topology
from atom.utils.forward_context import AttentionMetaData, AttnState, Context

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
    def __init__(self, model_runner):
        self.block_size = model_runner.block_size
        super().__init__(model_runner)
        self.config = model_runner.config.hf_config
        topology = build_attention_topology(self.config)[
            : self.config.num_hidden_layers
        ]
        self.geometry = V41PoolGeometry(
            len(topology),
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
        )
        self.cache = self.copies = self.engram = None
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

    def _prepare(self, batch, running_bs, running_tokens):
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
                position = end - length
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
        step = cache.begin_step(spans)
        positions = self.model_runner.forward_vars["positions"]
        positions.gpu[:offset].copy_(step.positions)
        positions.gpu[offset:running_tokens].zero_()
        cu = self.model_runner.forward_vars["cu_seqlens_q"].gpu[: running_bs + 1]
        metadata = AttentionMetaData(
            cu_seqlens_q=cu,
            max_seqlen_q=step.max_length,
            max_seqlen_k=max((span.end for span in spans), default=0),
            state=AttnState.DECODE if step.decode else AttnState.PREFILL_PREFIX,
        )
        metadata.cache, metadata.step = cache, step
        metadata.dummy = batch.is_dummy_run
        return metadata, positions.gpu[:running_tokens]

    def prepare_prefill(self, batch, running_bs):
        return self._prepare(batch, running_bs, batch.total_tokens_num)

    def prepare_decode(self, batch, running_bs, running_tokens, max_seqlen_q):
        return self._prepare(batch, running_bs, running_tokens)

    def prepare_model_inputs(self, input_ids, metadata):
        step, cache = metadata.step, metadata.cache
        histories = cache.prepare_state(step)
        tokens = input_ids[: step.length]
        if self.engram is not None:
            embeddings, histories = self.engram.prepare(
                step.requests, tokens, histories, dummy=metadata.dummy
            )
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
        metadata.engram_embeddings = embeddings
        metadata.next_histories = histories

    def build_for_cudagraph_capture(self, bs):
        # Only pure dense stages are captured. All attention warmup uses a
        # private PAGE/STATE allocation and can never alter live requests.
        batch = SimpleNamespace(
            is_dummy_run=True,
            req_ids=tuple(range(bs)),
            num_scheduled_tokens=(1,) * bs,
            context_lens=(1,) * bs,
            state_slots_committed=(),
            total_seqs_num=bs,
            total_tokens_num=bs,
        )
        metadata, positions = self._prepare(batch, bs, bs)
        self.prepare_model_inputs(
            self.model_runner.forward_vars["input_ids"].gpu[:bs], metadata
        )
        return metadata, Context(
            positions=positions,
            is_prefill=False,
            is_dummy_run=True,
            scheduled_bs=bs,
            scheduled_tokens=bs,
            running_bs=bs,
            running_tokens=bs,
        )
