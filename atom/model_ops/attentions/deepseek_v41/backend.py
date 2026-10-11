# SPDX-License-Identifier: MIT
"""ATOM scheduling adapter for the eager CSA2 paged runtime."""

import logging
from types import SimpleNamespace

import numpy as np
import torch

from atom.model_engine.kv_block import STATE_SLOT_CLASS
from atom.model_engine.state_runtime import StateTransfer
from atom.model_loader.weight_utils import local_model_dir
from atom.model_ops.attentions.backends import AttentionBackend, CommonAttentionBuilder
from atom.model_ops.attentions.deepseek_v4_attn import (
    DeepseekV4AttentionMetadataBuilder,
)
from atom.model_ops.attentions.pool_layout.sub_pool_spec import page_pool, state_pool
from atom.model_ops.attentions.pool_layout.v41_pool_geometry import V41PoolGeometry
from atom.model_ops.deepseek_v41.score_workspace import ScoreWorkspace
from atom.model_ops.engram.device.hashing import (
    EngramBatch,
    engram_compress,
    engram_cursor_rows,
)
from atom.model_ops.engram.device.runtime import EngramInputPreparer
from atom.model_ops.engram.device.staging import EngramStep
from atom.models.deepseek_v41.config import AttentionMode, build_attention_topology
from atom.utils import CpuGpuBuffer
from atom.utils.forward_context import AttentionMetaData, AttnState, Context

from .cache import PagedAttentionCache
from .checkpoints import StateCopies
from .metadata import RequestSpan, visible_buffer_name
from .score_planner import ScorePlanner

logger = logging.getLogger("atom")

# the forward_vars buffer of Engram's live count and dead flags (``EngramStep``)
ENGRAM_ROWS = "v41_engram_rows"


class DeepseekV41Backend(AttentionBackend):
    @staticmethod
    def get_name():
        return "CSA2"

    @staticmethod
    def get_builder_cls():
        return DeepseekV41MetadataBuilder


class DeepseekV41MetadataBuilder(CommonAttentionBuilder):
    capture_owns_cu_seqlens_q = True

    # Reuse V4's publisher and staging contract, including fixed addresses and
    # running_bs padding. Only pool-slot -> physical-row geometry differs.
    _stage = DeepseekV4AttentionMetadataBuilder._stage
    _populate_state_slot_mappings = (
        DeepseekV4AttentionMetadataBuilder._populate_state_slot_mappings
    )
    # Borrowed the same way, and for the same reason: it reads
    # `_unique_compress_ratios_overlap` and the `v4_*_plan_{ratio}` buffers,
    # which the property and `__init__` below supply under V4's names.
    _build_compress_plans = DeepseekV4AttentionMetadataBuilder._build_compress_plans
    _compress_publication_group = (
        DeepseekV4AttentionMetadataBuilder._compress_publication_group
    )
    # An index key is rotated at its compression group's first token, not at
    # its own, so the plan has to publish those positions.
    _publishes_key_rope = True

    @property
    def h2d_group_members(self):
        # Plans, state slots and step rows have no GPU consumer until
        # begin_step builds its indptrs. Derive members from producer groups.
        return {
            "v41_metadata": tuple(
                name
                for name, buffer in self.model_runner.forward_vars.items()
                if isinstance(buffer, CpuGpuBuffer)
                and (
                    buffer.publication_group in ("v4_plans", "v4_state", "v41_step")
                    or name == "block_tables"
                )
            )
        }

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
        # The V4.1 step producer fills these together with per-ratio visibility.
        for name in ("positions", "batch_id_per_q_token"):
            model_runner.forward_vars[name].publication_group = "v41_step"
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
            # Only the ratios the built layers run: a configuration with no
            # window-only layer gets no buffer for one.
            layer_ratios=tuple(sorted({spec.ratio for spec in topology})),
            index_topk=self.config.index_topk,
            # Paged at the length candidates are picked in, which is what lets
            # a candidate list be a block table. A GPU that cannot page that
            # short refuses when asked, so there is nothing to pre-empt here.
            index_block_rows=self.config.candidate_block_size,
            index_fp4=model_runner.config.index_cache_dtype == "fp4",
        )
        model_runner.forward_vars.update(
            self._compress_plan_buffers(
                self.geometry, self.max_num_batched_tokens, self.max_bs, self.device
            )
            | self._visible_buffers(
                self.geometry, self.max_num_batched_tokens, self.device
            )
            | self._engram_rows_buffer(self.max_num_batched_tokens, self.device)
        )
        # Before the memory profile, so the budget counts it; and so before
        # the pool is sized, which its packed logits follow: the weights are
        # in, so what is left of the budget bounds the PAGEs from above.
        self.score_workspace = ScoreWorkspace(
            self.geometry,
            self.max_num_batched_tokens,
            self.block_table_cols,
            self.device,
            pages=self._page_bound(model_runner.config.gpu_memory_utilization),
        )
        logger.info(
            "V4.1 score workspace: %s logits, %.2f GiB",
            "packed" if self.score_workspace.packed else "plane",
            self.score_workspace.capacity * 4 / (1 << 30),
        )
        if self.score_workspace.packed:
            self.add_step_planner(
                ScorePlanner(
                    self.score_workspace,
                    (ratio for ratio, _ in self.geometry.compress_ratios),
                    self.max_num_batched_tokens,
                )
            )
        self.cache = self.copies = self.engram = None
        self.dummy_weights = bool(model_runner.config.load_dummy)
        if not self.dummy_weights and self.config.engram_layer_ids:
            self.engram = EngramInputPreparer.from_checkpoint(
                local_model_dir(model_runner.config.model),
                self.config,
                self.max_num_batched_tokens,
                self.device,
            )

    step_planners = ()

    def add_step_planner(self, planner):
        """Have `planner` lay its per-step tables out of each step's staged rows
        (`prepare_batch_step`), in buffers of its own published with the step,
        and hand the step what it works out on the host (`StepPlan`). Before
        the forward buffers are bound to their publication."""
        self.model_runner.forward_vars.update(planner.buffers(self.device, "v41_step"))
        self.step_planners = (*self.step_planners, planner)

    # V4's plan builder asks for the ratio set under this name.
    _unique_compress_ratios_overlap = property(
        lambda self: self.geometry.compress_ratios
    )

    @staticmethod
    def _compress_plan_buffers(geometry, max_num_batched_tokens, max_bs, device):
        """Fixed-address plan buffers under V4's names, sized for prefill.

        A forward writes into these and slices the grid down; the pointers
        never move, which is what lets a captured graph replay another step's
        plan. Static so a hand-assembled `forward_vars` declares them from here
        rather than from a second copy of the sizing.
        """
        retained = max(geometry.speculative_tokens + 1, 1)
        buffers = {}
        for ratio, _ in geometry.compress_ratios:
            # Whichever regime is larger: a prefill's tight grid over its own
            # tokens, or the fixed `running_bs * per-seq bound` a CUDAGraph
            # decode cuts, which does not shrink with the batch. Sizing off
            # the tokens alone is how the write plan came out four rows short
            # of the six a six-token verify step declares.
            sizes = {
                # One boundary per `ratio` tokens, plus the partial group each
                # request can open; at most `ceil(q / ratio)` per request.
                f"v4_compress_plan_{ratio}": max(
                    max_num_batched_tokens // ratio + max_bs,
                    max_bs * -(-retained // ratio),
                ),
                # A bound, not a token count: the plan keeps a request's last
                # `max(K_pool, 1 + speculative_tokens)` positions.
                f"v4_write_plan_{ratio}": max(
                    min(max_num_batched_tokens, max_bs * max(ratio, retained)),
                    max_bs * retained,
                ),
            }
            for name, rows in sizes.items():
                buffer = CpuGpuBuffer(
                    rows,
                    4,
                    dtype=torch.int32,
                    device=device,
                    pin_memory=device != "cpu",
                    publication_group="v4_plans",
                )
                # Sentinel, so a capture before the first real forward reads
                # rows the kernels skip rather than zeros -- which would name
                # request 0 at position 0.
                buffer.cpu.fill_(-1)
                buffer.copy_to_gpu()
                buffers[name] = buffer
            # Beside the plan, never a fifth column in it: the fused kernel's
            # row is a 16-byte 4xi32 struct it loads once. int64 so the RoPE
            # ABI's own cast to int64 is a no-op.
            key_rope = CpuGpuBuffer(
                sizes[f"v4_compress_plan_{ratio}"],
                dtype=torch.int64,
                device=device,
                pin_memory=device != "cpu",
                publication_group="v4_plans",
            )
            # What a sentinel row works out to, so a pre-forward capture reads
            # the value every forward writes.
            key_rope.cpu.fill_(-ratio)
            key_rope.copy_to_gpu()
            buffers[f"v41_key_rope_positions_{ratio}"] = key_rope
        return buffers

    @staticmethod
    def _engram_rows_buffer(max_num_batched_tokens, device):
        """Engram's per-forward inputs the forward's graph cannot take as
        arguments: word 0 the live tokens (0: the kernels touch nothing), then
        each token's no-own-id flag (``EngramStep``). Published with the step."""
        return {
            ENGRAM_ROWS: CpuGpuBuffer(
                max_num_batched_tokens + 1,
                dtype=torch.int32,
                device=device,
                pin_memory=device != "cpu",
                publication_group="v41_step",
            )
        }

    @staticmethod
    def _visible_buffers(geometry, max_num_batched_tokens, device):
        """Fixed-address per-ratio visibility, one row per query token.

        Every indexer layer at a ratio reads the same rows, so it is the
        forward's metadata and not any layer's working set.
        """
        return {
            visible_buffer_name(ratio): CpuGpuBuffer(
                max_num_batched_tokens,
                dtype=torch.int32,
                device=device,
                pin_memory=device != "cpu",
                publication_group="v41_step",
            )
            for ratio, _ in geometry.compress_ratios
        }

    def sub_pool_specs(self):
        return [
            page_pool(self.geometry.paged_bytes),
            state_pool(STATE_SLOT_CLASS, self.geometry.state_bytes, entries_per_req=1),
        ]

    def _page_bound(self, utilization):
        """PAGEs the pool can reach at most: the budget the weights leave.

        The pool is sized later from what the memory profile leaves, which
        starts from at least what is allocated now.
        """
        total = torch.cuda.mem_get_info(self.device)[1]
        left = int(total * utilization) - torch.cuda.memory_allocated(self.device)
        return max(1, left // self.geometry.paged_bytes)

    def step_needs_eager(self, batch):
        """True when the step's rows cannot all be packed at once: a captured
        graph scores one band, so such a step runs eagerly, in bands.

        A row sees at most its request's context; shared prefixes, counted
        once in the pool, are why the pool-sized buffer can fall short.
        """
        return not self.score_workspace.fits(
            batch.num_scheduled_tokens, batch.context_lens
        )

    def state_transfer(self):
        return StateTransfer.copy(self.geometry.layout_id)

    def checkpoint_image_bytes(self):
        return self.geometry.state_bytes

    def allocate_kv_cache_tensors(self, *, blocks, buf):
        self.num_blocks = blocks
        return {}

    def allocate_per_req_cache(self, entries):
        self.cache = PagedAttentionCache(
            self.geometry,
            self.num_blocks,
            entries[STATE_SLOT_CLASS],
            self.device,
            max_tokens=self.max_num_batched_tokens,
            workspace=self.score_workspace,
        )
        self.copies = StateCopies(
            self.cache, self.model_runner.state_runtime.checkpoint_spec, self.max_bs
        )
        return {}

    def state_entry_views(self, slot):
        return [self.copies.entry(slot)]

    def relocate_state_slots(self, pairs):
        self.copies.relocate(pairs)

    def execute_paged_state_copies(self, stores, restores, descriptor_slot=0):
        self.copies.execute(stores, restores, descriptor_slot=descriptor_slot)

    def reserve_checkpoint_descriptors(self, descriptor_slots):
        self.copies.staging.reserve(descriptor_slots)

    def get_kv_transfer_tensors(self):
        """PAGE units and the native checkpoint contract, for `lmcache_mp`.

        A PAGE unit is its main page plus that page's rows in each index plane
        (`PagedAttentionCache.unit_regions`; an FP4 owner has two, its values
        and its scales). Each plane is published whole as one region, in that
        order -- the order `StateCopies` cuts a checkpoint
        image into, which `build_native_state_mp_layout` aliases unit by unit.
        Nothing else is published: no SLOT and no P/D staging, since the only
        transport admitted (`validate_runtime_config`) is the native-state MP
        path, which moves STATE through `execute_paged_state_copies`.
        """
        runner = self.model_runner
        if not getattr(runner.config, "kv_transfer_config", None):
            return None
        from atom.kv_transfer.disaggregation.page_region import page_region
        from atom.kv_transfer.disaggregation.types import KVTransferTensors

        if self.cache is None:
            raise RuntimeError(
                "DeepSeek-V4.1 publishes transfer regions after allocation"
            )
        names = [name for name, _ in self.cache.geometry.index_planes]
        planes = [("dsv41.page", self.cache.page_bytes)] + [
            (f"dsv41.{name}_plane.{owner}", plane)
            for owner, owned in self.cache.index_planes.items()
            for name, plane in zip(names, owned)
        ]
        pages = [page_region(plane, semantic_role=role) for role, plane in planes]
        spec = runner.state_runtime.checkpoint_spec
        published = [page.region.unit_bytes for page in pages]
        if (
            published != [size for _, size in self.cache.unit_regions()]
            or sum(published) != spec.page_unit_bytes
        ):
            raise RuntimeError(
                "DeepSeek-V4.1 transfer regions do not match the checkpoint's "
                f"PAGE unit: regions={published}, unit={spec.page_unit_bytes}"
            )
        # No cache dimension is split across TP: every rank holds the same PAGE
        # and STATE bytes, as DeepSeek-V4's compressed cache does.
        tp_size = int(getattr(runner.config, "tensor_parallel_size", 1) or 1)
        return KVTransferTensors(
            pages=pages,
            tp_replication_factor=tp_size,
            native_state_tp_replication_factor=tp_size,
            paged_state_checkpoint_spec=spec,
            execute_paged_state_copies=self.execute_paged_state_copies,
            paged_state_region_count=len(pages),
        )

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
        max_q_len=None,
        tentative=False,
        start_positions=None,
        query_prefix_ready=False,
        engram_live=True,
    ):
        """``engram_live`` False: Engram's forward kernels run on no token (a
        capture: they record on serving's buffers, touching none)."""
        spans, rows, offset, next_page = [], [], 0, 0
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
                blocks = np.arange(next_page, next_page + count, dtype=np.int32)
                next_page += count
                slot = len(spans)
            else:
                position = (
                    end - length if start_positions is None else int(start_positions[i])
                )
                blocks = batch.block_tables[i]
                slot = slots[i]
            spans.append(RequestSpan(request_id, position, offset, length, slot))
            rows.append(blocks)
            offset += length
        if offset != batch.total_tokens_num or running_tokens < offset:
            raise ValueError("CSA2 batch token spans disagree with the runner")
        cache = (
            PagedAttentionCache(
                self.geometry,
                max(next_page, 1),
                max(len(spans), 1),
                self.device,
                max_tokens=running_tokens,
                workspace=self.score_workspace,
            )
            if batch.is_dummy_run
            else self.cache
        )
        if cache is None:
            raise RuntimeError("CSA2 cache must be allocated before serving")
        groups = getattr(self.model_runner, "h2d_groups", None)
        combined = None if groups is None else groups.get("v41_metadata")
        if combined is not None and combined.transport != "packed":
            combined = None
        if combined is not None:
            for i in range(len(combined.counts)):
                combined.counts[i] = None
        # Zero-token scheduler rows are excluded from spans. Publish in this
        # same request order, including the private dummy slots used at startup.
        state_slot_out = self._populate_state_slot_mappings(
            SimpleNamespace(state_slots_committed=[span.slot for span in spans]),
            len(spans),
            running_bs,
            publication_group=combined,
        )
        verifying = tentative and not batch.is_dummy_run and bool(spans)
        # One plan per ratio for the whole batch, into the fixed-address
        # buffers. `running_bs` / `max_q_len` cut both plans to a capacity that
        # depends on neither the batch nor its content -- the shape a capture
        # records and every replay has to dispatch -- and sentinel the tail.
        # A prefill passes neither and gets the tight grid.
        # `extra_write`: CSA2's K_pool is 1 or 2, narrower than a verify step,
        # so without the slack the plan drops what a rejection re-exposes.
        plans = self._build_compress_plans(
            np.asarray([span.length for span in spans], dtype=np.int32),
            np.asarray([span.end for span in spans], dtype=np.int32),
            running_bs=None if max_q_len is None else running_bs,
            max_q_len=max_q_len,
            extra_write=self.geometry.speculative_tokens if verifying else 0,
            defer_to=combined,
        )
        token_mask = self._token_mask(batch, spans, offset)
        step_group = (
            combined
            if combined is not None
            else None if groups is None else groups["v41_step"]
        )
        live = offset if engram_live and not batch.is_dummy_run else 0
        self._stage(
            ENGRAM_ROWS,
            np.concatenate(([live], ~token_mask)).astype(np.int32),
            publication_group=step_group,
        )
        step = cache.begin_step(
            spans,
            block_tables=rows,
            tentative=verifying,
            buffers=self.model_runner.forward_vars,
            running_bs=running_bs,
            running_tokens=running_tokens,
            max_q_len=max_q_len,
            state_slot_out=state_slot_out,
            plans=plans,
            planners=self.step_planners,
            publication_group=step_group,
            query_prefix_ready=query_prefix_ready and len(spans) == len(batch.req_ids),
            query_prefix_republish_reason=(
                "compact zero-token scheduler rows for CSA2 after input assembly"
                if query_prefix_ready and len(spans) != len(batch.req_ids)
                else None
            ),
        )
        positions = self.model_runner.forward_vars["positions"]
        cu = self.model_runner.forward_vars["cu_seqlens_q"].gpu[: running_bs + 1]
        metadata = AttentionMetaData(
            cu_seqlens_q=cu,
            # The bucket the runner settled on, which is what `run_model` keys
            # the graph by. A ragged verify step whose longest request came in
            # shorter still replays the bucket's graph.
            max_seqlen_q=step.max_q_len,
            max_seqlen_k=max((span.end for span in spans), default=0),
            state=AttnState.DECODE if step.decode else AttnState.PREFILL_PREFIX,
        )
        metadata.cache, metadata.step = cache, step
        metadata.state_slot_out = state_slot_out
        metadata.dummy = batch.is_dummy_run
        metadata.token_mask = token_mask
        metadata.image_mask = (
            torch.from_numpy(~token_mask).to(self.device).unsqueeze(0)
            if not token_mask.all()
            else None
        )
        return metadata, positions.gpu[:running_tokens]

    @staticmethod
    def _token_mask(batch, spans, tokens):
        """False where a token carries no id of its own (an image row)."""
        token_mask = np.ones(tokens, dtype=np.bool_)
        for span in spans:
            data = getattr(batch, "multimodal_data", {}).get(span.request_id)
            if data is not None:
                for start, count in data.get("embedding_spans", ()):
                    first, end = max(start, span.position), min(start + count, span.end)
                    if first < end:
                        at = span.offset + first - span.position
                        token_mask[at : at + end - first] = False
        return token_mask

    def prepare_prefill(self, batch, running_bs):
        return self._prepare(
            batch, running_bs, batch.total_tokens_num, query_prefix_ready=True
        )

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
            max_q_len=max_seqlen_q,
            tentative=bool(self.geometry.speculative_tokens),
            start_positions=starts,
            query_prefix_ready=True,
        )

    def _engram_batch(self, step, cache, metadata, tokens):
        """This forward on the device, for the Engram kernels.

        `None` whenever the hashing has to stay on the host: a synthetic batch,
        whose cursor belongs to whoever owns those slots, or a build without
        the UVA lookup, where the gather reads the tables on the host and so
        wants the rows there too.

        `image_mask` goes in as it stands -- true where a token carries no id
        of its own, which is the DEAD sense `engram_compress` takes.
        """
        if metadata.dummy or self.engram is None or not self.engram.host.uva:
            return None
        tables = self.engram.host.hash_tables
        dead = metadata.image_mask
        return EngramBatch(
            compressed=engram_compress(
                tables, tokens, None if dead is None else dead[0, : step.scheduled]
            ),
            batch_ids=step.batch_ids[: step.scheduled],
            cu_seqlens=step.cu_seqlens_q,
            history=cache.cursor[:, 1:],
            history_index=step.slots[: step.scheduled_bs],
        )

    def _engram_step(self, step, cache, input_ids):
        """This forward's Engram inputs at fixed addresses, for the forward to
        run on (overlap only; a synthetic batch's live count is 0)."""
        if self.engram is None or self.engram.host.overlap is None:
            return None
        rows = self.model_runner.forward_vars[ENGRAM_ROWS].gpu
        return EngramStep(
            input_ids=input_ids,
            live=rows[:1],
            dead=rows[1:],
            batch_ids=step.batch_ids,
            cu_seqlens=step.cu_seqlens_q,
            positions=step.positions,
            cursor=cache.cursor,
            history_index=step.slots,
            candidates=cache.tentative_staging.gpu if step.tentative else None,
        )

    def _write_engram_cursor(self, step, cache, batch):
        """Advance the cursor, or stage every prefix a verify step may accept
        (its cursor is the sampler's: `commit_tentative` picks one)."""
        tentative = step.tentative
        engram_cursor_rows(
            self.engram.host.hash_tables,
            batch.compressed,
            batch.cu_seqlens,
            step.positions,
            cache.cursor[:, 1:],
            batch.history_index,
            (
                cache.tentative_staging.gpu[: step.scheduled_bs]
                if tentative
                else cache.cursor
            ),
            candidates=step.max_q_len if tentative else 0,
        )

    def prepare_model_inputs(self, input_ids, metadata):
        step, cache = metadata.step, metadata.cache
        # The rows the requests own, not the rows the forward runs: the padding
        # tail is zeroed inside `run_model`, after this, so what stands there
        # now is the previous step's ids. The width goes separately.
        tokens = input_ids[: step.scheduled]
        # Before `prepare_state`: a device path defers its history readback.
        # A synthetic batch's slots are someone else's: it never moves them.
        staged = self._engram_step(step, cache, input_ids)
        batch = (
            None
            if staged is not None
            else self._engram_batch(step, cache, metadata, tokens)
        )
        on_device = not metadata.dummy and (staged is not None or batch is not None)
        histories = (
            np.full((step.scheduled_bs, self.geometry.history_size), -1, np.int64)
            if metadata.dummy
            else cache.prepare_state(step, histories=not on_device)
        )
        if self.engram is not None:
            prepared = self.engram.prepare(
                step.requests,
                tokens,
                histories,
                dummy=metadata.dummy,
                token_mask=metadata.token_mask,
                padded_rows=step.width,
                batch=batch,
                staged=staged,
            )
            embeddings, histories = prepared.embeddings, prepared.histories
            if not on_device and cache.pending is not None:
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
                    1, step.width, width, dtype=torch.bfloat16, device=self.device
                )
                for layer in self.config.engram_layer_ids
            }
            if cache.pending is not None:
                for span in step.requests:
                    cache.pending.stage_history(span, [-1] * span.length)
        metadata.engram_embeddings = embeddings
        # Last: every reader of the cursor this overwrites has run (`staged`:
        # the forward advances it, after its snapshot).
        if batch is not None:
            self._write_engram_cursor(step, cache, batch)
        elif staged is None and not metadata.dummy and not step.tentative:
            cache.advance_cursor(step, histories)
        if on_device and step.tentative:
            cache.pending.staged_on_device = True

    def commit_speculative_state(self, metadata, last_token_indices):
        if metadata.cache.pending is not None:
            metadata.cache.commit_tentative(metadata.step, last_token_indices)

    def build_for_cudagraph_capture(self, bs, max_q_len=1):
        # Binds the serving allocation, as V4 does: a scratch cache would bake
        # the wrong window address into the shared draft graph. Runtime dummies
        # still get the private cache `_prepare` picks for them.
        if self.cache is None:
            raise RuntimeError("Allocate the serving cache before graph capture")
        if bs < 1 or max_q_len < 1 or bs * max_q_len > self.max_num_batched_tokens:
            raise ValueError("CSA2 capture shape exceeds the token buffer")
        tokens = bs * max_q_len
        # A full window in, not position 0: there the compressor reads no
        # history and captures a cold branch replay never takes. `tentative` is
        # what makes a multi-token bucket a decode step -- otherwise the
        # capture records the prefill FFN while replay runs `decode_ffn` eager.
        # The bound cache's geometry, since that is the pool being captured.
        geometry = self.cache.geometry
        start = geometry.window_size
        pages = -(-(start + max_q_len) // self.block_size)
        batch = SimpleNamespace(
            is_dummy_run=False,
            req_ids=tuple(range(bs)),
            num_scheduled_tokens=(max_q_len,) * bs,
            context_lens=(start + max_q_len,) * bs,
            state_slots_committed=tuple(range(bs)),
            # Block 0 for every entry of every request, which is what V4's
            # capture builds: a placeholder whose values capture reads and
            # throws away. One page rather than a run of them is the point --
            # naming `pages` distinct pages makes capture write that many, and
            # those are the ones the block pool hands out first.
            block_tables=((0,) * pages,) * bs,
            total_seqs_num=bs,
            total_tokens_num=tokens,
        )
        metadata, positions = self._prepare(
            batch,
            bs,
            tokens,
            max_q_len=max_q_len,
            tentative=bool(geometry.speculative_tokens),
            engram_live=False,
        )
        metadata.dummy = True  # No host Engram lookup for synthetic tokens.
        self.prepare_model_inputs(
            self.model_runner.forward_vars["input_ids"].gpu[:tokens], metadata
        )
        return metadata, Context(
            positions=positions,
            is_prefill=False,
            is_dummy_run=False,
            scheduled_bs=bs,
            scheduled_tokens=tokens,
            running_bs=bs,
            running_tokens=tokens,
        )
