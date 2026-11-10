# SPDX-License-Identifier: MIT
"""ATOM scheduling adapter for the eager CSA2 paged runtime."""

import logging
from contextlib import contextmanager
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import torch

from atom.model_engine.kv_block import STATE_SLOT_CLASS
from atom.model_engine.state_runtime import StateTransfer
from atom.model_loader.weight_utils import local_model_dir
from atom.model_ops.attentions.backends import (
    AttentionBackend,
    AttentionMetadataBuilder,
    CommonAttentionBuilder,
)
from atom.model_ops.attentions.deepseek_v4_attn import (
    DeepseekV4AttentionMetadataBuilder,
)
from atom.model_ops.attentions.pool_layout.sub_pool_spec import page_pool, state_pool
from atom.model_ops.deepseek_v41.score_workspace import ScoreWorkspace
from atom.model_ops.engram.device.hashing import (
    EngramBatch,
    engram_compress,
    engram_cursor_rows,
)
from atom.model_ops.engram.device.runtime import EngramInputPreparer
from atom.model_ops.engram.device.staging import (
    EngramRowsView,
    EngramStep,
    engram_staging,
)
from atom.model_ops.v4_kernels import make_compress_plans
from atom.models.deepseek_v41.bounded_replay import (
    ReplayPlan,
    decoder_replay_unsupported,
    lay_out_tail,
    tail_spans,
    tail_width,
)
from atom.models.deepseek_v41.pool_geometry import build_v41_pool_geometry
from atom.utils import CpuGpuBuffer, envs
from atom.utils.forward_context import AttentionMetaData, AttnState, Context
from atom.utils.tbo import NUM_TBO_UBATCHES

from .cache import PagedAttentionCache
from .checkpoints import StateCopies
from .metadata import RequestSpan, prepare_batch_step, visible_buffer_name
from .score_planner import ScorePlanner

logger = logging.getLogger("atom")

# the forward_vars buffer of Engram's live count and dead flags (``EngramStep``)
ENGRAM_ROWS = "v41_engram_rows"
# the prefix of the step buffers a bounded replay's decoder rows publish into
DECODER_ROWS = "tail_"


def _compress_plan_buffer_names(ratio):
    return {
        "compress": f"v4_compress_plan_{ratio}",
        "write": f"v4_write_plan_{ratio}",
        "key_rope": f"v41_key_rope_positions_{ratio}",
    }


class DeepseekV41Backend(AttentionBackend):
    @staticmethod
    def get_name():
        return "CSA2"

    @staticmethod
    def get_builder_cls():
        return DeepseekV41MetadataBuilder


class DeepseekV41MetadataBuilder(CommonAttentionBuilder):
    tbo_comm_stream_priority = -1

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
    # An index key is rotated at its compression group's first token, not at
    # its own, so the plan has to publish those positions.
    _publishes_key_rope = True

    def h2d_group_members(self) -> dict[str, tuple[str, ...]]:
        # Plans, state slots and step rows: one group for a prefill's and a
        # decode's step alike; a prefill TBO microbatch's step and plans its
        # own. Derived from their producer groups.
        def members(groups, extra=()):
            return tuple(
                name
                for name, buffer in self.model_runner.forward_vars.items()
                if isinstance(buffer, CpuGpuBuffer)
                and (buffer.publication_group in groups or name in extra)
            )

        groups = super().h2d_group_members()
        groups["step_metadata"] = members(
            ("v4_plans", "v4_state", "v41_step"), ("block_tables",)
        )
        if self.model_runner.config.enable_tbo:
            for i in range(NUM_TBO_UBATCHES):
                groups[f"ub{i}_prefill_metadata"] = members(
                    (f"ub{i}_v4_plans", f"ub{i}_v41_step")
                )
        return groups

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
        # The V4.1 step producer fills it together with per-ratio visibility.
        model_runner.forward_vars["positions"].publication_group = "v41_step"
        self.config = model_runner.config.hf_config
        speculative = model_runner.config.speculative_config
        num_drafts = 0 if speculative is None else speculative.num_speculative_tokens
        # Whether a serving prefill gets a decoder tail (`_publish_decoder`):
        # the runtime model replays exactly when this holds
        # (`DeepseekV41RuntimeModel.__init__`).
        self._decoder_replays = (
            getattr(model_runner.config, "enable_decoder_swa_bounded_replay", False)
            and decoder_replay_unsupported(self.config) is None
        )
        self.geometry = build_v41_pool_geometry(
            self.config,
            self.block_size,
            kv_dtype=model_runner.config.kv_cache_dtype,
            speculative_tokens=num_drafts,
            index_fp4=model_runner.config.index_cache_dtype == "fp4",
            index_layout=model_runner.config.index_cache_layout,
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
        # The planner's fixed rows and their child copies count against the
        # PAGE budget too. Bind the scorer after its storage has been sized.
        score_planner = None
        if self.geometry.index_fp4:
            score_planner = ScorePlanner(
                None,
                (ratio for ratio, _ in self.geometry.compress_ratios),
                self.max_num_batched_tokens,
            )
            self.add_step_planner(score_planner)
        # Prefill TBO microbatches publish their steps into `ub{i}_` copies of
        # the step buffers, owned by the runner like every other builder's
        # (V4, MLA, MHA): one set per microbatch. Reuse is fenced twice. The
        # pinned host sources wait for the H2D publication owner's completion
        # event. The device copies, and each microbatch's indptr fill (by its
        # first reader), are issued on the compute stream, after every reader
        # of the last forward's: the workers' side streams
        # rejoin it inside attention, and `UBatchWrapper._run_ubatches` orders
        # the comm stream before it however the microbatches exit. These fixed
        # allocations precede the memory profile: get_num_blocks accounts
        # for them in peak/current PyTorch bytes, outside the KV pool. They
        # remain resident across KV release, like the parent forward buffers.
        if model_runner.config.enable_tbo:
            model_runner.forward_vars.update(self._ubatch_step_buffers())
            self._ubatch_indptrs = [
                {
                    ratio: tuple(
                        torch.empty(
                            self.max_num_batched_tokens + 1,
                            dtype=torch.int32,
                            device=self.device,
                        )
                        for _ in range(2)
                    )
                    for ratio in self.geometry.layer_ratios
                }
                for _ in range(NUM_TBO_UBATCHES)
            ]
        # A replaying prefill's decoder runs on rows of its own, the tail,
        # published as a step at addresses of its own (`_publish_decoder`), so
        # a recording of the decoder reads what every replay of it publishes.
        # Fenced like the `ub{i}_` copies.
        if self._decoder_replays:
            model_runner.forward_vars.update(
                self._step_buffer_copies(DECODER_ROWS, self._step_buffer_names())
                | self._decoder_layout_buffers()
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
            # A candidate-consuming layer's plane is as wide as the candidate
            # list, which the checkpoint fixes and `max_model_len` does not
            # bound. Short contexts make it the widest plane in the model.
            candidate_blocks=int(getattr(self.config, "candidate_topk_blocks", 0) or 0),
        )
        logger.info(
            "V4.1 score workspace: %s logits, %.2f GiB",
            "packed" if self.score_workspace.packed else "plane",
            self.score_workspace.capacity * 4 / (1 << 30),
        )
        if score_planner is not None:
            score_planner.workspace = self.score_workspace
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
    _decoder_replays = False

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
            names = _compress_plan_buffer_names(ratio)
            # Whichever regime is larger: a prefill's tight grid over its own
            # tokens, or the fixed `running_bs * per-seq bound` a CUDAGraph
            # decode cuts, which does not shrink with the batch. Sizing off
            # the tokens alone is how the write plan came out four rows short
            # of the six a six-token verify step declares.
            sizes = {
                # One boundary per `ratio` tokens, plus the partial group each
                # request can open; at most `ceil(q / ratio)` per request.
                names["compress"]: max(
                    max_num_batched_tokens // ratio + max_bs,
                    max_bs * -(-retained // ratio),
                ),
                # A bound, not a token count: the plan keeps a request's last
                # `max(K_pool, 1 + speculative_tokens)` positions.
                names["write"]: max(
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
                sizes[names["compress"]],
                dtype=torch.int64,
                device=device,
                pin_memory=device != "cpu",
                publication_group="v4_plans",
            )
            # What a sentinel row works out to, so a pre-forward capture reads
            # the value every forward writes.
            key_rope.cpu.fill_(-ratio)
            key_rope.copy_to_gpu()
            buffers[names["key_rope"]] = key_rope
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
        graph scores one band, so such a step runs eagerly, in bands. Also for
        a prefill carrying images, which no token graph runs: image rows route
        on their mask.

        A row sees at most its request's context; shared prefixes, counted
        once in the pool, are why the pool-sized buffer can fall short.
        """
        if batch.total_tokens_num_prefill > 0 and batch.multimodal_data:
            return True
        return not self.score_workspace.fits(
            batch.num_scheduled_tokens, batch.context_lens
        )

    def state_transfer(self):
        # Encoder SWA bounded replay: a hit rebuilds the state from the PAGE
        # below it instead of restoring a checkpoint (`BlockManager.
        # replay_start`).
        if self.model_runner.config.enable_encoder_swa_bounded_replay:
            return StateTransfer.replay()
        return StateTransfer.copy(self.geometry.layout_id)

    def checkpoint_image_bytes(self):
        return self.geometry.checkpoint_image_bytes

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

        For P/D (`slot_regions`): the request's whole STATE slot as one
        slot-indexed region. A slot is contiguous and holds every
        layer's window ring -- the decoder's bounded-replay tail and the DSpark
        context layers included -- plus the compressor rings and the cursor, so
        it is all a consumer needs to decode and draft from the prompt's end.
        """
        runner = self.model_runner
        if not getattr(runner.config, "kv_transfer_config", None):
            return None
        from atom.kv_transfer.disaggregation.page_region import page_region
        from atom.kv_transfer.disaggregation.types import (
            KVTransferRegion,
            KVTransferTensors,
        )

        if self.cache is None:
            raise RuntimeError(
                "DeepSeek-V4.1 publishes transfer regions after allocation"
            )
        names = [name for name, _ in self.cache.geometry.index_planes]
        if self.cache.geometry.index_fp4:
            # Equal byte widths do not imply equal physical layout. Mooncake
            # checks these roles before writing any PAGE.
            names = [f"{name}_fp4_{self.cache.geometry.index_layout}" for name in names]
        planes = [("dsv41.page", self.cache.page_bytes)] + [
            (f"dsv41.{name}_plane.{owner}", plane)
            for owner, owned in self.cache.index_planes.items()
            for name, plane in zip(names, owned)
        ]
        pages = [page_region(plane, semantic_role=role) for role, plane in planes]
        # None under encoder SWA bounded replay, which keeps no checkpoint
        # (`state_transfer`); P/D needs none -- it moves the live STATE slot.
        spec = runner.state_runtime.checkpoint_spec
        published = [page.region.unit_bytes for page in pages]
        if published != [size for _, size in self.cache.unit_regions()] or (
            spec is not None and sum(published) != spec.page_unit_bytes
        ):
            raise RuntimeError(
                "DeepSeek-V4.1 transfer regions do not match the PAGE unit: "
                f"regions={published}, cache={self.cache.unit_regions()}, "
                f"checkpoint={None if spec is None else spec.page_unit_bytes}"
            )
        # No cache dimension is split across TP: every rank holds the same PAGE
        # and STATE bytes, as DeepSeek-V4's compressed cache does.
        tp_size = int(getattr(runner.config, "tensor_parallel_size", 1) or 1)
        # The STATE slot travels as one slot-indexed region the connector
        # writes straight across (`src_slot * unit` to `dst_slot * unit`). The
        # consumer posts the write only once its device ran the in-flight step
        # of the slot's previous tenant (`MooncakeConnector._post_write_requests`).
        from atom.kv_transfer.disaggregation.factory import KVConnectorFactory

        state = self.cache.state_bytes  # [slots, state_bytes] uint8 view
        slot_unit = int(state.shape[1])
        p_d = (
            KVConnectorFactory.connector_name(runner.config.kv_transfer_config)
            == "mooncake"
        )
        return KVTransferTensors(
            pages=pages,
            # `lmcache_mp` moves STATE through `execute_paged_state_copies`.
            slot_regions=(
                [
                    KVTransferRegion(
                        state.data_ptr(),
                        int(state.shape[0]) * slot_unit,
                        slot_unit,
                        semantic_role="dsv41.state_slot",
                    )
                ]
                if p_d
                else []
            ),
            num_slots=int(state.shape[0]) if p_d else 0,
            tp_replication_factor=tp_size,
            native_state_tp_replication_factor=tp_size,
            paged_state_checkpoint_spec=spec,
            # Under encoder SWA bounded replay there is no checkpoint to copy:
            # `lmcache_mp` moves PAGE alone and every hit replays.
            execute_paged_state_copies=(
                None if spec is None else self.execute_paged_state_copies
            ),
            paged_state_region_count=len(pages),
        )

    def warmup_per_req_cache(self):
        self.copies.warmup()

    def release_kv_pools(self):
        # Release PAGE/STATE views only. Runner-owned forward buffers and
        # ubatch indptrs are persistent, already counted in the KV budget.
        # H2D publication retains their fixed addresses across sleep/wake.
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
        is_prefill=False,
        start_positions=None,
        engram_live=True,
        ubatch_slices=None,
    ):
        """``engram_live`` False: Engram's forward kernels run on no token (a
        capture: they record on serving's buffers, touching none).
        ``ubatch_slices``: a TBO prefill's micro-batches, each a step of its
        own staged and published with this one."""
        spans, rows, offset, next_page = [], [], 0, 0
        slots = batch.state_slots_committed
        if not batch.is_dummy_run and len(slots) != batch.total_seqs_num:
            raise ValueError("CSA2 requires a STATE slot for every scheduled request")
        for i, (request_id, length, end) in enumerate(
            zip(batch.req_ids, batch.num_scheduled_tokens, batch.context_lens)
        ):
            length, end = int(length), int(end)
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
        resumes, floors, cached_until, histories = (
            (None,) * 4
            if batch.is_dummy_run or not batch.replaying
            else self._replay_layout(batch.replaying, spans)
        )
        # Fabricated requests get private scratch.
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
        return self._prepare_step(
            batch,
            spans,
            rows,
            cache,
            running_bs,
            running_tokens,
            max_q_len=max_q_len,
            tentative=tentative,
            is_prefill=is_prefill,
            engram_live=engram_live,
            replay_layout=(resumes, floors, cached_until, histories),
            ubatch_slices=ubatch_slices,
        )

    def _step_group(self, name):
        """The group a step is staged into; none for an
        isolated caller with no runner registry, whose step uploads each
        buffer as it stages it (`prepare_batch_step`)."""
        if getattr(self.model_runner, "h2d_groups", None) is None:
            return None
        return self.model_runner.h2d_groups[name]

    def _prepare_step(
        self,
        batch,
        spans,
        rows,
        cache,
        running_bs,
        running_tokens,
        *,
        max_q_len,
        tentative,
        is_prefill,
        engram_live=True,
        replay_layout=(None, None, None, None),
        ubatch_slices=None,
    ):
        """``replay_layout``: ``_replay_layout``'s (resumes, floors,
        cached_until, histories) for a resumed prefill, else all None."""
        resumes, floors, cached_until, histories = replay_layout
        offset = sum(span.length for span in spans)
        # The step's one upload, after everything below is staged into it.
        group = self._step_group("step_metadata")
        # Published in span order, including the private dummy slots used at
        # startup.
        state_slot_out = self._populate_state_slot_mappings(
            SimpleNamespace(state_slots_committed=[span.slot for span in spans]),
            len(spans),
            running_bs,
            publication_group=group,
        )
        verifying = tentative and bool(spans)
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
            write_from=resumes,
            running_bs=None if max_q_len is None else running_bs,
            max_q_len=max_q_len,
            # Idle ranks replay the same write grid as verifying peers.
            extra_write=self.geometry.speculative_tokens if tentative else 0,
            group=group,
        )
        token_mask = self._token_mask(batch, spans, offset)
        live = offset if engram_live and not batch.is_dummy_run else 0
        self._stage(
            ENGRAM_ROWS,
            np.concatenate(([live], ~token_mask)).astype(np.int32),
            publication_group=group,
        )
        step = cache.begin_step(
            spans,
            block_tables=rows,
            tentative=verifying,
            is_prefill=is_prefill,
            buffers=self.model_runner.forward_vars,
            running_bs=running_bs,
            running_tokens=running_tokens,
            max_q_len=max_q_len,
            state_slot_out=state_slot_out,
            plans=plans,
            planners=self.step_planners,
            floors=floors,
            cached_until=cached_until,
            finals=(
                None
                if batch.is_final_chunk is None or batch.is_dummy_run
                else np.asarray(batch.is_final_chunk)
            ),
            start_histories=histories,
            publication_group=group,
            query_prefix_ready=True,
        )
        step.request_resumes = resumes
        ubatch_slices = ubatch_slices or ()
        ubatch_groups = tuple(
            f"ub{i}_prefill_metadata" for i in range(len(ubatch_slices))
        )
        self._check_metadata_writable(*ubatch_groups)
        ubatch_steps = [
            self._stage_ubatch_step(step, rows, cache, ub_idx, ub)
            for ub_idx, ub in enumerate(ubatch_slices)
        ]
        for staged in (group, *map(self._step_group, ubatch_groups)):
            if staged is not None:
                staged.publish()
        # A serving prefill, an idle rank's included; not a warmup's, which
        # computes every row.
        if self._decoder_replays and is_prefill and cache is self.cache:
            step.decoder = self._publish_decoder(step, rows, cache, running_bs)
        metadata = self._assemble_metadata(
            cache,
            step,
            rows,
            dummy=batch.is_dummy_run,
            token_mask=token_mask,
        )
        if ubatch_slices:
            metadata.ubatches = [
                self._finish_ubatch_step(metadata, ub_idx, ub, ub_step)
                for ub_idx, (ub, ub_step) in enumerate(zip(ubatch_slices, ubatch_steps))
            ]
        return metadata, step.positions

    def _publish_decoder(self, step, rows, cache, running_bs):
        """The `ReplayPlan` of the rows a replaying prefill's decoder runs on:
        each finishing request's last window rows (`tail_spans`), its window
        starting there, laid out for ``running_bs`` requests (`lay_out_tail`)
        as a step of their own in the `DECODER_ROWS` buffers -- one set of
        addresses per request bucket, so its recording (`DecoderGraph`) reads
        what every replay publishes. No compress plan: the forward's
        compression ran over every row."""
        window = self.geometry.window_size
        picks = tail_spans(step, window)
        width = tail_width(running_bs, window, self.max_num_batched_tokens)
        layout = lay_out_tail(picks, running_bs, width)
        lengths = np.asarray([n for _, n, _ in picks], dtype=np.int32)
        spans = tuple(
            RequestSpan(span.request_id, span.end - n, int(offset), n, span.slot)
            for (span, n, _), offset in zip(picks, np.cumsum(lengths) - lengths)
        )
        group = self._step_group(f"{DECODER_ROWS}v41_step")

        def stage(name, values, pad_to=None):
            return self._stage(
                f"{DECODER_ROWS}{name}",
                np.asarray(values, dtype=np.int32),
                pad_to,
                publication_group=group,
            )

        slots = stage("state_slot_out", [span.slot for span in spans], running_bs)
        floors = stage("replay_start", [span.position for span in spans], running_bs)
        plan = {
            name: stage(name, getattr(layout, name))
            for name in ("rows", "query_ids", "last", "query_ptrs", "last_rows")
        }
        # The decoder passes' real row counts -- the tail's rows, then layer
        # 39's one per request -- read in place like the step's
        # `scheduled_tokens_gpu`: a recording replays every step of its bucket.
        filled_gpu = stage("filled", [layout.filled])
        finishing_gpu = stage("finishing", [len(picks)])
        tail = prepare_batch_step(
            spans,
            self.device,
            block_tables=[rows[i] for _, _, i in picks],
            is_prefill=True,
            buffers=self._prefixed_buffers(DECODER_ROWS),
            buffer_prefix=DECODER_ROWS,
            running_bs=running_bs,
            running_tokens=width,
            max_q_len=window,
            state_slot_out=slots,
            ratios=tuple(ratio for ratio, _ in self.geometry.compress_ratios),
            publication_group=group,
        )
        if group is not None:
            group.publish()
        tail.request_floors = tail.request_positions.astype(np.int32)
        tail.request_cached_until = np.zeros(len(spans), np.int32)
        cache.publish_replay_layout(tail, running_bs, floors=floors)
        return ReplayPlan(
            tail=tail,
            tail_rows=plan["rows"],
            query_ids=plan["query_ids"],
            last=plan["last"],
            query_ptrs=plan["query_ptrs"],
            last_rows=plan["last_rows"],
            filled=layout.filled,
            filled_gpu=filled_gpu,
            finishing_gpu=finishing_gpu,
        )

    @staticmethod
    def _assemble_metadata(
        cache,
        step,
        rows,
        *,
        dummy,
        token_mask,
        image_mask=None,
        engram_embeddings=None,
    ):
        metadata = AttentionMetaData(
            cu_seqlens_q=step.cu_seqlens_q,
            max_seqlen_q=step.max_q_len,
            max_seqlen_k=max((span.end for span in step.requests), default=0),
            state=AttnState.DECODE if step.decode else AttnState.PREFILL_PREFIX,
        )
        metadata.cache, metadata.step = cache, step
        metadata.block_table_rows = rows
        # A bounded replay's `QueryRows`, while a layer runs on some rows only
        metadata.queries = None
        metadata.state_slot_out = step.slots
        metadata.dummy = dummy
        metadata.token_mask = token_mask
        metadata.image_mask = image_mask
        if image_mask is None and not token_mask.all():
            metadata.image_mask = (
                torch.from_numpy(~token_mask).to(step.positions.device).unsqueeze(0)
            )
        metadata.engram_embeddings = (
            {} if engram_embeddings is None else engram_embeddings
        )
        return metadata

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

    def _replay_layout(self, replaying, spans):
        """Per span, for the forwards starting below where they resumed
        (`ScheduledBatch.replaying`), 0 for every other: that point, below
        which it writes no global KV; its window floor, the forward's start;
        that point again where its state came along, its ring exact below it;
        and, with Engram, the history its cursor starts at, as a full prefill
        leaves it there."""
        resumes, floors, cached = np.zeros((3, len(spans)), np.int32)
        histories = (
            None
            if self.engram is None
            else np.full((len(spans), self.geometry.history_size), -1, np.int64)
        )
        for k, replay in replaying.items():
            resumes[k], floors[k] = replay.until, spans[k].position
            cached[k] = 0 if replay.rebuilds else replay.until
            if histories is not None:
                mapping = self.engram.mapping
                ids = np.asarray(replay.history_ids, np.int64)
                # An image position, which carries the image id, is DEAD.
                image = getattr(self.config, "image_token_id", None)
                compressed = mapping.compress_tokens(
                    ids, None if image is None else (ids != image)[None]
                )
                histories[k] = mapping.advance_history(None, compressed)[0]
        return resumes, floors, cached, histories

    def query_lengths(self, batch, is_prefill, running_tokens):
        # A step lays out its scheduled spans alone (`_prepare`), a prefill
        # with none included: no padding request.
        return AttentionMetadataBuilder.query_lengths(
            self, batch, is_prefill, running_tokens
        )

    def prepare_prefill(
        self,
        batch,
        running_bs,
        running_tokens,
        ubatch_slices=None,
        pcp_groups=None,
    ):
        return self._prepare(
            batch,
            running_bs,
            running_tokens,
            is_prefill=True,
            ubatch_slices=ubatch_slices,
        )

    # Every launch outside the model's eager regions is sized by the token
    # count alone, so a prefill may pad to a recorded token-graph bucket.
    supports_token_graph = True

    def _ubatch_step_buffers(self):
        """`ub{i}_` copies of every buffer a V4.1 step and its compress plans
        are staged into, with the parent's shapes and dtypes."""
        var = self.model_runner.forward_vars
        names = self._step_buffer_names()
        for ratio, _ in self.geometry.compress_ratios:
            names += _compress_plan_buffer_names(ratio).values()
        # Step planners (including packed FP4 score offsets) need independent
        # published rows for each child, just like positions and visibility.
        names += [
            name
            for name, buffer in var.items()
            if isinstance(buffer, CpuGpuBuffer)
            and buffer.publication_group == "v41_step"
            and name not in names
            and name != ENGRAM_ROWS
        ]
        buffers = {}
        for i in range(NUM_TBO_UBATCHES):
            buffers |= self._step_buffer_copies(f"ub{i}_", names)
        return buffers

    def _step_buffer_names(self):
        """The buffers `prepare_batch_step` stages a step into."""
        names = ["positions", "cu_seqlens_q", "batch_id_per_q_token", "block_tables"]
        return names + [
            visible_buffer_name(r) for r, _ in self.geometry.compress_ratios
        ]

    def _step_buffer_copies(self, prefix, names):
        """A row domain's own copies of ``names``, as ``{prefix}name`` with the
        parent's shapes and dtypes: a step of other rows than the forward's,
        published at addresses of its own (`prepare_batch_step`)."""
        var = self.model_runner.forward_vars
        buffers = {}
        for name in names:
            src = var[name]
            group = "v4_plans" if src.publication_group == "v4_plans" else "v41_step"
            buffers[f"{prefix}{name}"] = CpuGpuBuffer(
                *src.cpu.shape,
                dtype=src.cpu.dtype,
                device=self.device,
                pin_memory=self.device != "cpu",
                publication_group=f"{prefix}{group}",
            )
        return buffers

    def _decoder_layout_buffers(self):
        """What a tail publishes beside its step (`_publish_decoder`): its
        slots and window starts, one per request, and its `TailLayout`."""
        bs, tokens = self.max_bs, self.max_num_batched_tokens
        sizes = {
            "state_slot_out": bs,
            "replay_start": bs,
            "rows": tokens,
            "query_ids": tokens,
            "last": bs,
            "query_ptrs": bs + 1,
            "last_rows": bs,
            "filled": 1,
            "finishing": 1,
        }
        return {
            f"{DECODER_ROWS}{name}": CpuGpuBuffer(
                size,
                dtype=torch.int32,
                device=self.device,
                pin_memory=self.device != "cpu",
                publication_group=f"{DECODER_ROWS}v41_step",
            )
            for name, size in sizes.items()
        }

    def _prefixed_buffers(self, prefix):
        """A row domain's step buffers under their parent names."""
        var = self.model_runner.forward_vars
        return {
            name[len(prefix) :]: buffer
            for name, buffer in var.items()
            if name.startswith(prefix)
        }

    def _ubatch_buffers(self, ubatch_idx):
        """This microbatch's step buffers under their parent names."""
        return self._prefixed_buffers(f"ub{ubatch_idx}_")

    @contextmanager
    def ubatch_forward(self, metadata):
        with engram_staging(metadata.engram_embeddings, tbo=True):
            yield

    def _stage_ubatch_step(self, parent, rows, cache, ubatch_idx, ub_slice):
        """Micro-batch ``ub_slice``'s step of ``parent``'s requests, and its
        compression plans, staged into its ``ub{i}_prefill_metadata`` group."""
        if parent.tentative:
            raise ValueError("V4.1 TBO supports prefill only")
        ts, rs = ub_slice.token_slice, ub_slice.request_slice
        if not 0 <= rs.start <= rs.stop <= len(parent.requests):
            raise ValueError("V4.1 microbatch request slice is outside its parent")
        if not 0 <= ts.start < ts.stop <= parent.running_tokens:
            raise ValueError("V4.1 microbatch token slice is outside its parent")
        spans = []
        for span in parent.requests[rs]:
            first, end = (
                max(span.offset, ts.start),
                min(span.offset + span.length, ts.stop),
            )
            if first < end:
                spans.append(
                    replace(
                        span,
                        position=span.position + first - span.offset,
                        offset=first - ts.start,
                        length=end - first,
                    )
                )
        width = ub_slice.running_tokens
        if sum(span.length for span in spans) != ub_slice.scheduled_tokens:
            raise ValueError("V4.1 microbatch request and token slices disagree")
        # A resumed prefill keeps its replay layout in each microbatch: the
        # window floor, the ring rows it already holds exact (never rewritten),
        # and the compressed rows below where it resumed (never recompressed).
        # Per request, in absolute positions, so a request split across both
        # microbatches carries the same values into each.
        resumes = None if parent.request_resumes is None else parent.request_resumes[rs]
        buffers = self._ubatch_buffers(ubatch_idx)
        prefix = f"ub{ubatch_idx}_"
        # The microbatch's step and plans go up with the parent's.
        group = self._step_group(f"{prefix}prefill_metadata")
        step = prepare_batch_step(
            tuple(spans),
            self.device,
            block_tables=rows[rs],
            is_prefill=True,
            buffers=buffers,
            buffer_prefix=prefix,
            planners=self.step_planners,
            running_bs=ub_slice.running_bs,
            running_tokens=width,
            state_slot_out=parent.slots[rs],
            ratios=tuple(ratio for ratio, _ in self.geometry.compress_ratios),
            publication_group=group,
        )
        if cache.workspace is not None:
            step.tile_workspace = cache.workspace.tile_slice(ts)
        step.plans = make_compress_plans(
            np.asarray([span.length for span in spans], dtype=np.int32),
            np.asarray([span.end for span in spans], dtype=np.int32),
            self.geometry.compress_ratios,
            plan_buffers={
                ratio: {
                    role: buffers[name]
                    for role, name in _compress_plan_buffer_names(ratio).items()
                }
                for ratio, _ in self.geometry.compress_ratios
            },
            group=group,
            extra_write=0,
            write_from=resumes,
        )
        step.request_floors = parent.request_floors[rs].copy()
        step.request_cached_until = parent.request_cached_until[rs].copy()
        step.request_resumes = None if resumes is None else resumes.copy()
        return step

    def _finish_ubatch_step(self, metadata, ubatch_idx, ub_slice, step):
        """Micro-batch ``ub_slice``'s metadata over its uploaded ``step``, its
        rows of the parent ``metadata``'s."""
        ts, rs = ub_slice.token_slice, ub_slice.request_slice
        if step.positions.is_cuda:
            # Indexed by this microbatch's rows, not the parent's. Before the
            # indptrs, which count each row's window from its floor.
            metadata.cache.publish_replay_layout(step, ub_slice.running_bs)
            step.indptr_buffers = self._ubatch_indptrs[ubatch_idx]
        child = self._assemble_metadata(
            metadata.cache,
            step,
            metadata.block_table_rows[rs],
            dummy=metadata.dummy,
            token_mask=metadata.token_mask[ts],
            image_mask=(
                None if metadata.image_mask is None else metadata.image_mask[:, ts]
            ),
        )
        # Its Engram rows are the parent's, once `prepare_model_inputs` has
        # them.
        child.parent_tokens = ts
        return child

    def prepare_decode(
        self, batch, running_bs, running_tokens, max_seqlen_q, decode_ubatch_slices=None
    ):
        # The config refuses decode TBO (`deepseek_v41.config`): never sliced.
        starts = None
        if self.geometry.speculative_tokens and batch.total_seqs_num:
            # The scheduler reserves a full draft span, including placeholders
            # from the previous step. Ragged verification takes its head.
            starts = self.model_runner.tokenID_processor.verify_ctx_lens(
                np.asarray(batch.context_lens) - (batch.num_spec_step + 1)
            )
        return self._prepare(
            batch,
            running_bs,
            running_tokens,
            max_q_len=max_seqlen_q,
            tentative=bool(self.geometry.speculative_tokens),
            start_positions=starts,
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
                tables,
                tokens,
                None if dead is None else dead[0, : step.scheduled_tokens],
            ),
            batch_ids=step.batch_ids[: step.scheduled_tokens],
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
        tokens = input_ids[: step.scheduled_tokens]
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
                padded_rows=step.running_tokens,
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
                    1,
                    step.running_tokens,
                    width,
                    dtype=torch.bfloat16,
                    device=self.device,
                )
                for layer in self.config.engram_layer_ids
            }
            if cache.pending is not None:
                for span in step.requests:
                    cache.pending.stage_history(span, [-1] * span.length)
        metadata.engram_embeddings = embeddings
        for child in metadata.ubatches or ():
            child.engram_embeddings = EngramRowsView(embeddings, child.parent_tokens)
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

    def build_for_cudagraph_capture(
        self, bs, max_q_len=1, *, decode_ubatch_slices=None
    ):
        # Binds the serving allocation, as V4 does: a scratch cache would bake
        # the wrong window address into the shared draft graph. Runtime idle
        # ranks also use this pool; startup dummies use private scratch.
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
            is_final_chunk=None,
            replaying={},
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

    @property
    def prefill_req_graph_sizes(self):
        """The decoder's request buckets (`DecoderGraph`,
        `ATOM_DECODER_GRAPH_SIZES`) a step can hold: no more requests than a
        batch, one window each within the token buffer
        (`build_for_req_graph_capture`); a larger step's tail runs eagerly.
        Not on an FP8 index plane, whose tile tables follow each request's
        context (`PagedAttentionCache.unit_tiles`), which a recording would
        fix."""
        if not (self._decoder_replays and self.geometry.index_fp4):
            return ()
        window = self.geometry.window_size
        return tuple(
            n
            for n in envs.ATOM_DECODER_GRAPH_SIZES
            if n <= self.max_bs and n * window <= self.max_num_batched_tokens
        )

    def build_for_token_graph_capture(self, tokens):
        """A prefill of ``tokens`` rows for recording that token-graph bucket:
        final requests from position 0 of one window each (the last takes the
        remainder). On the serving pool for the reason
        `build_for_cudagraph_capture` binds it, every row on placeholder PAGE
        0. The regions run for real on it; the segments only need its shape."""
        if not 0 < tokens <= self.max_num_batched_tokens:
            raise ValueError("CSA2 token-graph shape exceeds the token buffer")
        window = self.geometry.window_size
        lengths = (window,) * (tokens // window) + (
            (tokens % window,) * bool(tokens % window)
        )
        if len(lengths) > self.max_bs:
            lengths = (tokens,)
        return self._build_for_prefill_capture(lengths, finishing=True)

    def build_for_req_graph_capture(self, running_bs, finishing):
        """A prefill of ``running_bs`` requests of one window each, for
        recording that request bucket of the decoder (`DecoderGraph`): with
        every request ``finishing`` its prompt, or none, the tail a DP rank
        runs to join its peers' decoder."""
        if running_bs * self.geometry.window_size > self.max_num_batched_tokens:
            raise ValueError("CSA2 request-graph shape exceeds the token buffer")
        return self._build_for_prefill_capture(
            (self.geometry.window_size,) * running_bs, finishing=finishing
        )

    def _build_for_prefill_capture(self, lengths, finishing):
        """A prefill of requests of ``lengths`` from position 0, each
        ``finishing`` its prompt or not, on the serving pool."""
        if self.cache is None:
            raise RuntimeError("Allocate the serving cache before graph capture")
        count, tokens = len(lengths), sum(lengths)
        pages = -(-max(lengths) // self.block_size)
        batch = SimpleNamespace(
            is_dummy_run=False,
            req_ids=tuple(range(count)),
            num_scheduled_tokens=lengths,
            context_lens=lengths,
            state_slots_committed=tuple(range(count)),
            block_tables=((0,) * pages,) * count,
            total_seqs_num=count,
            total_tokens_num=tokens,
            total_tokens_num_prefill=tokens,
            is_final_chunk=None if finishing else (False,) * count,
            replaying={},
        )
        self.publish_query_metadata(np.asarray(lengths, np.int32), count, tokens)
        metadata, positions = self._prepare(
            batch, count, tokens, is_prefill=True, engram_live=False
        )
        metadata.dummy = True
        self.prepare_model_inputs(
            self.model_runner.forward_vars["input_ids"].gpu[:tokens], metadata
        )
        context = Context(
            positions=positions,
            is_prefill=True,
            is_dummy_run=False,
            scheduled_bs=count,
            scheduled_tokens=tokens,
            running_bs=count,
            running_tokens=tokens,
        )
        return metadata, context
