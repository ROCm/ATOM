"""Opt-in persistent owner execution at the real ModelRunner target seam.

Native prefill, vocabulary, rejection sampling, T5, deferred status and scheduler
output remain their original implementations. No diagnostic snapshots or repairs.
"""

import collections
import copy
import json
import os
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path

from .owner_graph import GraphCoverage, OwnerGraphInputs, graph_key
from .owner_graph_agreement import FixedGraphAgreement
from .owner_reconciliation_plan import reconciliation_requests
from .owner_serving_plan import (
    RequestOwners,
    ServingRows,
    prefix_lengths,
    rebuild_prefix_batch,
    written_page_units,
)

HERE = Path(__file__).resolve().parent


def published_input_view(input_ids, width):
    """Expose the already allocated native input capacity to graph capture.

    Owner packet indices address executed rows only (padding addresses row 0).
    The original runner publishes into this same allocation each step; an
    active-row slice must not become a separate graph for each departing request.
    """
    if input_ids.ndim != 1 or input_ids.stride() != (1,) or width < input_ids.numel():
        raise ValueError("Owner target needs the native contiguous published input")
    extent = (input_ids.storage_offset() + width) * input_ids.element_size()
    if extent > input_ids.untyped_storage().nbytes():
        raise ValueError("Native input allocation does not hold its published capacity")
    return input_ids.as_strided((width,), (1,))


def bind_published_engram(embeddings, input_ids):
    """Bind native Engram hashing to the same published graph input capacity."""
    from atom.model_ops.engram.device.staging import EngramStagedRows

    if not isinstance(embeddings, EngramStagedRows):
        return
    staging = embeddings.staging
    if (
        embeddings.width != input_ids.numel()
        or staging.step.input_ids.data_ptr() != input_ids.data_ptr()
    ):
        raise ValueError("Engram and owner target must read the same native input")
    staging.step = replace(staging.step, input_ids=input_ids)


@contextmanager
def owner_context(metadata, stream=None):
    """A scoped context works for both ordinary and thread-local frontends."""
    import torch

    from atom.utils import forward_context as module

    old = module.get_forward_context()
    context = copy.copy(old)
    context.context = copy.copy(old.context)
    # Owner packets contain per-rank padding holes, not a global valid prefix.
    # Their metadata masks the inactive slots. The outer scheduled-row scalar
    # describes a different layout and must never trim this packed pass.
    context.context.scheduled_tokens_gpu = None
    context.attn_metadata = metadata
    context.in_hipgraph = True
    context.main_stream = torch.cuda.current_stream() if stream is None else stream
    context.ubatch_slices = None
    previous = getattr(module._forward_context_local, "ctx", None)
    module._forward_context_local.ctx = context
    try:
        yield context
    finally:
        module._forward_context_local.ctx = previous


class OwnerEngram:
    def __init__(self, parent, rows):
        self.parent, self.rows = parent, rows

    def get(self, layer, default=None):
        value = self.parent.get(layer, default)
        return value.index_select(1, self.rows) if value is not None else None


class ContextPacket:
    # Use the destination writer's policy before projection/quantization.
    packed = False

    def __init__(self, step, *, window_fp8=False):
        self.step, self.keys = step, {}
        self.window_fp8 = window_fp8

    def write_window(self, layer, keys, step):
        if step is not self.step:
            raise ValueError("Draft context writer received another owner step")
        self.keys[layer] = keys


class OwnerServing:
    def __init__(self, runner):
        import torch
        from aiter.dist.parallel_state import get_tp_group

        self.runner, self.group = runner, get_tp_group()
        if self.group.world_size != 4:
            raise ValueError("Owner serving currently requires TP4")
        self.rank = self.group.rank_in_group
        self.registry = RequestOwners()
        self.policy = json.loads((HERE / "FFN_COST_FIXED_POLICY.json").read_text())
        self.counters = collections.Counter()
        self.dirty = {}
        self.active = None
        self.graphs = {}
        from atom.utils.graph_holders import register_graph_holder

        register_graph_holder(self)
        self.graph_pool = None
        self.stable_graphs = os.environ.get("DSV41_OWNER_STABLE_GRAPHS") == "1"
        self.graph_inputs = None
        self.graph_coverage = GraphCoverage()
        self.graph_agreement = FixedGraphAgreement(self.group.cpu_group)
        control = os.environ.get("DSV41_OWNER_GRAPH_PHASE_FILE")
        self.graph_phase_file = Path(control) if control else None
        self.stream = torch.cuda.Stream()
        self.done = torch.cuda.Event()
        self.done_recorded = False
        self.metadata_uploaded = torch.cuda.Event()
        self.metadata_upload_recorded = False
        self.packet_publisher = None
        self.packet_publication = (
            os.environ.get("DSV41_OWNER_PACKET_PUBLICATION") == "1"
        )
        self.confidence = []
        self.confidence_ring = torch.empty(
            (4, 256, 5), dtype=torch.float32, device="cpu", pin_memory=True
        )
        self.confidence_events = [None] * 4
        self.confidence_index = 0
        self.heads = None
        self.cache = None
        self.mode = "full"
        self.model = runner.model
        while not hasattr(self.model, "layers") and hasattr(self.model, "model"):
            self.model = self.model.model
        if not hasattr(self.model, "layers") or len(self.model.layers) != 40:
            raise ValueError("Bind the actual 40-layer V4.1 runtime")
        self.model._owner_serving = self
        self.control = Path(os.environ["DSV41_OWNER_SERVING_CONTROL"])
        self.evidence = (
            Path(os.environ["DSV41_OWNER_SERVING_OUTPUT"])
            / f"route_rank{self.rank}.json"
        )

    def publish(self):
        result = {
            "mode": self.mode,
            "counts": dict(self.counters),
            "request_epochs": len(self.registry.slots),
            "dirty_epochs": len(self.dirty),
            "captured_owner_graphs": len(self.graphs),
            "stable_owner_graph_inputs": self.stable_graphs,
            "owner_graph_coverage": self.graph_coverage.report(),
            "native_kernels": True,
            "real_model_runner_target": True,
            "original_scheduler_and_client_output": True,
            "selective_native_reconciliation": os.environ.get(
                "DSV41_OWNER_SELECTIVE_RECONCILIATION"
            )
            == "1",
            "retained_prefill_contract": bool(
                getattr(self.model, "owner_service_prefill", False)
            ),
        }
        content = (json.dumps(result, indent=2) + "\n").encode()
        if len(content) > 128_000:
            raise RuntimeError("Owner route evidence exceeds its bound")
        temporary = self.evidence.with_suffix(".tmp")
        temporary.write_bytes(content)
        temporary.replace(self.evidence)

    def ready_confidence(self):
        if len(self.confidence) < 2:
            return {}
        event, data, ids = self.confidence[-2]
        # Fixed two-generation publication, never latest-ready per rank.
        event.synchronize()
        import numpy as np

        invalid = ~np.isfinite(data.numpy()) | (data.numpy() < 0) | (data.numpy() > 1)
        self.counters["invalid_old_confidence_values"] += int(invalid.sum())
        return dict(zip(ids, data.numpy().tolist(), strict=True))

    def record_confidence(self, confidence, request_ids):
        import torch

        n = len(request_ids)
        if confidence is None or n > 256 or confidence.shape[0] < n:
            self.confidence.clear()
            return
        index = self.confidence_index
        busy = self.confidence_events[index]
        if busy is not None:
            busy.synchronize()
        data = self.confidence_ring[index, :n]
        stream = self.runner.tokenID_processor.async_copy_stream
        producer = torch.cuda.current_stream()
        event = torch.cuda.Event()
        with torch.cuda.stream(stream):
            stream.wait_stream(producer)
            # The native proposer may drop its owned confidence output as soon
            # as this hook returns. Keep its allocator storage unavailable for
            # reuse until the asynchronous D2H reader has finished.
            confidence.record_stream(stream)
            data.copy_(confidence[:n, :5].detach(), non_blocking=True)
            event.record(stream)
        self.confidence_events[index] = event
        self.confidence_index = (index + 1) % 4
        self.confidence.append((event, data, tuple(request_ids)))
        self.confidence = self.confidence[-4:]

    def before_prepare(self, batch):
        self.active = None
        self.candidate_batch = None
        self.current_batch = batch
        self.mode = self.control.read_text().strip()
        if self.mode not in ("native", "full", "packed"):
            raise ValueError("Unknown owner serving capacity")
        self.counters["runner_batches"] += 1
        if batch.is_dummy_run:
            return
        requests = tuple(
            zip(
                map(int, batch.req_ids),
                map(int, batch.state_slots_committed),
                strict=True,
            )
        )
        owners = self.registry.bind(requests)
        eligible = (
            self.mode != "native"
            and batch.total_seqs_num_prefill == 0
            and 4 <= batch.total_seqs_num <= 256
            and all(1 <= int(n) <= 6 for n in batch.num_scheduled_tokens)
            and not any(batch.return_logprobs)
            and len(set(owners.values())) == 4
        )
        if not eligible:
            self.reconcile(
                request_ids=tuple(map(int, batch.req_ids)),
                slots=(
                    *map(int, batch.state_slots_committed),
                    *map(int, batch.state_fork_srcs),
                ),
                pages=(page for table in batch.block_tables for page in table),
            )
            reason = "native_mode" if self.mode == "native" else "prefill_or_boundary"
            self.counters["fallback_" + reason] += 1
            return
        self._owners = owners
        if self.mode == "packed":
            lengths = prefix_lengths(
                tuple(map(int, batch.req_ids)),
                tuple(map(int, batch.num_scheduled_tokens)),
                owners,
                self.ready_confidence(),
                self.policy,
            )
            before = batch.total_tokens_num
            if rebuild_prefix_batch(batch, lengths):
                self.counters["prefix_cut_batches"] += 1
                self.counters["removed_query_rows"] += before - batch.total_tokens_num
        self.candidate_batch = batch

    def reconcile(self, *, request_ids=None, slots=(), pages=()):
        """Native writers/checkpoint consumers get complete current PAGE/STATE.

        Owner target regions alias native storage. Only non-owner replicas need
        repair; compact draft context is already replicated after every target.
        """
        import torch
        import torch.distributed as dist

        if not self.dirty:
            return
        selected = reconciliation_requests(
            self.dirty,
            request_ids=() if request_ids is None else request_ids,
            slots=slots,
            pages=pages,
            all_dirty=(
                request_ids is None
                or os.environ.get("DSV41_OWNER_SELECTIVE_RECONCILIATION") != "1"
            ),
        )
        self.counters["unrelated_dirty_epochs_retained"] += len(self.dirty) - len(
            selected
        )
        if not selected:
            return
        cache = self.runner.attn_metadata_builder.cache
        if self.done_recorded:
            torch.cuda.current_stream().wait_event(self.done)
        if self.packet_publication:
            from .owner_state_publication import StatePacketPublisher

            if self.packet_publisher is None:
                self.packet_publisher = StatePacketPublisher(self.group)
            if self.packet_publisher.publish(
                cache, self.dirty, selected, self.counters
            ):
                for request_id in selected:
                    del self.dirty[request_id]
                self.counters["native_reconciliations"] += 1
                return
        # Reused PAGE units may occur in prefix sharing. Serialize the last
        # owner of a unit; exact duplicate old-prefix units are read-only.
        for request_id in selected:
            owner, slot, pages = self.dirty[request_id]
            state = cache.state_bytes[slot]
            dist.broadcast(
                state, src=self.group.ranks[owner], group=self.group.device_group
            )
            self.counters["reconciliation_bytes"] += state.numel()
            for page in pages:
                for region in cache.unit_views(page):
                    dist.broadcast(
                        region,
                        src=self.group.ranks[owner],
                        group=self.group.device_group,
                    )
                    self.counters["reconciliation_bytes"] += region.numel()
        for request_id in selected:
            del self.dirty[request_id]
        self.counters["native_reconciliations"] += 1

    def allocate(self, source):
        import torch

        from atom.model_ops.attentions.deepseek_v41.metadata import visible_buffer_name
        from atom.model_ops.attentions.deepseek_v41.score_planner import ScorePlanner
        from atom.utils import CpuGpuBuffer

        from .owner_cost_attention_chain import FullOwnerHeads

        if self.heads is None:
            print(
                f"OWNER_SERVING rank={self.rank} gather original full head weights",
                flush=True,
            )
            self.heads = tuple(
                FullOwnerHeads.gather_native(layer.attn, self.group)
                for layer in self.model.layers
            )
        if self.cache is not None:
            if self.cache.backing.data_ptr() != source.backing.data_ptr():
                raise ValueError("Persistent serving cache allocation changed")
            return
        self.cache = copy.copy(source)
        self.cache.pending = None
        self.cache.indptr_buffers = {}
        self.cache.max_tokens = 0
        self.cache._reserve_indptrs(384)
        device = source.backing.device
        columns = self.runner.forward_vars["block_tables"].np.shape[1]
        ratios = tuple(ratio for ratio, _ in source.geometry.compress_ratios)
        shapes = {
            "positions": (384,),
            "cu_seqlens_q": (65,),
            "batch_id_per_q_token": (384,),
            "block_tables": (64, columns),
            **{visible_buffer_name(ratio): (384,) for ratio in ratios},
        }
        self.buffers = {
            name: CpuGpuBuffer(
                *shape,
                dtype=torch.int64 if name == "positions" else torch.int32,
                device=device,
                pin_memory=True,
            )
            for name, shape in shapes.items()
        }
        self.planner = ScorePlanner(source.workspace, ratios, 384)
        self.buffers.update(self.planner.buffers(device, publication_group=None))
        self.slots = CpuGpuBuffer(64, dtype=torch.int32, device=device, pin_memory=True)
        self.plan_buffers = {
            ratio: {
                "compress": CpuGpuBuffer(
                    448, 4, dtype=torch.int32, device=device, pin_memory=True
                ),
                "write": CpuGpuBuffer(
                    448, 4, dtype=torch.int32, device=device, pin_memory=True
                ),
                "key_rope": CpuGpuBuffer(
                    448, dtype=torch.int64, device=device, pin_memory=True
                ),
            }
            for ratio, _ in source.geometry.compress_ratios
        }
        self.packet_rows = CpuGpuBuffer(
            1536, dtype=torch.int64, device=device, pin_memory=True
        )
        self.local_rows = CpuGpuBuffer(
            384, dtype=torch.int64, device=device, pin_memory=True
        )
        self.inverse_rows = CpuGpuBuffer(
            1536, dtype=torch.int64, device=device, pin_memory=True
        )

    def bind(self, metadata):
        import numpy as np

        from atom.model_ops.attentions.deepseek_v41.metadata import RequestSpan
        from atom.model_ops.v4_kernels import make_compress_plans

        source_step = metadata.step
        rows = ServingRows.bind(source_step.requests, self._owners)
        self.allocate(metadata.cache)
        # Our pinned buffers are separate from runner's publication ring.
        # Protect their reuse, including lazy-capture warmup reads.
        if self.packet_publication:
            if self.metadata_upload_recorded:
                self.metadata_uploaded.synchronize()
        elif self.done_recorded:
            self.done.synchronize()
        indices = rows.rank_indices[self.rank]
        spans, offset = [], 0
        for i in indices:
            span = source_step.requests[i]
            spans.append(
                RequestSpan(
                    span.request_id, span.position, offset, span.length, span.slot
                )
            )
            offset += span.length
        n = len(spans)
        self.slots.np[:n] = [span.slot for span in spans]
        self.slots.np[n:64] = 0
        slots = self.slots.copy_to_gpu(64)
        lengths = np.asarray([span.length for span in spans], dtype=np.int32)
        plans = make_compress_plans(
            lengths,
            np.asarray([span.end for span in spans], dtype=np.int32),
            self.cache.geometry.compress_ratios,
            plan_buffers=self.plan_buffers,
            running_bs=64,
            max_q_len=source_step.max_q_len,
            extra_write=(
                self.cache.geometry.speculative_tokens if source_step.tentative else 0
            ),
        )
        tables = tuple(metadata.block_table_rows[i] for i in indices)

        def select(values):
            return None if values is None else np.asarray(values)[list(indices)].copy()

        step = self.cache.begin_step(
            spans,
            block_tables=tables,
            tentative=source_step.tentative,
            is_prefill=False,
            buffers=self.buffers,
            running_bs=64,
            running_tokens=rows.capacity,
            max_q_len=source_step.max_q_len,
            state_slot_out=slots,
            plans=plans,
            planners=(self.planner,),
            floors=select(source_step.request_floors),
            cached_until=select(source_step.request_cached_until),
            finals=select(source_step.request_finals),
            start_histories=select(source_step.start_histories),
        )
        width = rows.capacity * 4
        self.packet_rows.np[:width] = rows.packet_rows
        local = rows.packet_rows[
            self.rank * rows.capacity : (self.rank + 1) * rows.capacity
        ]
        self.local_rows.np[: rows.capacity] = local
        self.inverse_rows.np[: source_step.scheduled] = rows.global_from_packet
        packet = self.packet_rows.copy_to_gpu(width)
        local = self.local_rows.copy_to_gpu(rows.capacity)
        inverse = self.inverse_rows.copy_to_gpu(source_step.scheduled)
        if self.packet_publication:
            self.metadata_uploaded.record()
            self.metadata_upload_recorded = True
        owned = copy.copy(metadata)
        owned.step, owned.cache = step, self.cache
        owned.engram_embeddings = OwnerEngram(metadata.engram_embeddings, local)
        owned.image_mask = None
        return rows, owned, packet, inverse

    def target(self, input_ids, metadata, packet_rows):
        import torch

        from atom.model_ops.deepseek_v41.mhc import SinglePassHCState
        from atom.model_ops.deepseek_v41.row_sharded_mhc import v41_mhc_reduce_scatter
        from atom.model_ops.engram.device.staging import EngramStagedRows
        from atom.model_ops.module_dispatch_ops import _dual_stream_is_active
        from atom.models.deepseek_v41.model import Block

        from .owner_cost_attention_chain import forward_owner
        from .owner_serving_kernels import sum_original_tp

        step, model = metadata.step, self.model
        capacity = step.width
        step.begin_forward()
        global_engram = (
            None if self.stable_graphs else metadata.engram_embeddings.parent
        )
        if isinstance(global_engram, EngramStagedRows):
            global_engram.stage()
        tokens = (
            input_ids if self.stable_graphs else input_ids.index_select(0, packet_rows)
        )
        # Identical packet order before the sharded vocabulary embedding AR.
        embedded = model.embed(tokens)
        state = SinglePassHCState.from_embeddings(
            embedded[self.rank * capacity : (self.rank + 1) * capacity].unsqueeze(0),
            model.config.hc_mult,
        )
        taps = []
        for i, layer in enumerate(model.layers):
            if i in (37, 38, 39):
                taps.append(
                    state.settle().residual.mean(dim=-2).reshape(capacity, 5120)
                )
            embeddings = (
                metadata.engram_embeddings.get(layer.engram.layer_id)
                if layer.engram is not None
                else None
            )
            normed, residual, pre, post, comb = Block.prepare_attention(
                layer, state, embeddings, None
            )
            output, _ = forward_owner(
                layer.attn,
                normed,
                self.cache,
                step,
                model._rope(i),
                self.heads[i],
                self.rank,
                sum_original_tp,
            )
            normed, residual, pre, post, comb = Block.prepare_ffn(
                layer, output, residual, pre, post, comb
            )
            packed = self.group.all_gather(
                normed.reshape(capacity, 5120), use_custom=True, dim=0
            )
            moe = layer.ffn
            if _dual_stream_is_active(moe, packed.shape[0]):
                main = torch.cuda.current_stream()
                moe.alt_stream.wait_stream(main)
                with torch.cuda.stream(moe.alt_stream):
                    shared = moe.shared_experts(packed)
                routed = moe.experts(packed, moe.router_logits(packed))
                main.wait_stream(moe.alt_stream)
            else:
                shared = moe.shared_experts(packed)
                routed = moe.experts(packed, moe.router_logits(packed))
            output = v41_mhc_reduce_scatter(routed + shared).view(1, capacity, 5120)
            state = Block.finish_ffn(layer, output, residual, pre, post, comb)
        if isinstance(global_engram, EngramStagedRows):
            global_engram.join()
        return state.collapse().reshape(capacity, 5120), torch.cat(taps, dim=-1)

    def run(self, input_ids, batch):
        import torch
        from aiter.dist.parallel_state import GraphCaptureContext

        from atom.utils.forward_context import get_forward_context

        original = get_forward_context().attn_metadata
        input_ids = published_input_view(input_ids, original.step.width)
        bind_published_engram(original.engram_embeddings, input_ids)
        rows, metadata, packet, inverse = self.bind(original)
        source_input = input_ids
        if self.stable_graphs:
            if self.graph_inputs is None:
                self.graph_inputs = OwnerGraphInputs(
                    input_ids, original.engram_embeddings
                )
            input_ids, embeddings = self.graph_inputs.views(rows.capacity)
            key = graph_key(
                rows.capacity,
                input_ids,
                metadata.step.max_q_len,
                metadata.step.planned_host,
                embeddings,
            )
        else:
            bands = tuple(sorted(metadata.step.planned_host.items()))
            key = (
                rows.capacity,
                input_ids.shape[0],
                input_ids.data_ptr(),
                metadata.step.max_q_len,
                bands,
            )
        was_sealed = self.graph_coverage.sealed
        phase = (
            self.graph_phase_file.read_text().strip()
            if self.graph_phase_file is not None
            else "warmup"
        )
        # All ranks capture together even if only one rank's band layout changed.
        missing = self.graph_agreement.exchange(
            self.graph_coverage.request(key not in self.graphs, phase)
        )
        self.counters["owner_fixed_graph_agreements"] += 1
        try:
            capture = self.graph_coverage.require_capture(missing)
        except RuntimeError:
            self.counters["owner_graph_misses_after_seal"] += 1
            self.publish_graph_coverage(missing_key=key)
            self.publish()
            raise
        if self.graph_coverage.sealed and not was_sealed:
            self.publish_graph_coverage()
            self.publish()
        if self.stable_graphs:
            from atom.model_ops.engram.device.staging import EngramStagedRows

            global_engram = original.engram_embeddings
            if isinstance(global_engram, EngramStagedRows):
                global_engram.stage()
            input_ids, embeddings = self.graph_inputs.stage(
                source_input,
                packet,
                metadata.engram_embeddings.rows,
                global_engram,
                rows.capacity,
            )
            if isinstance(global_engram, EngramStagedRows):
                global_engram.join()
            metadata.engram_embeddings = embeddings
            self.counters["stable_owner_input_publications"] += 1
        if capture:
            self.counters["owner_graph_captures"] += 1
            print(
                f"OWNER_SERVING rank={self.rank} capture M={rows.capacity * 4} real={original.step.scheduled}",
                flush=True,
            )
            torch.cuda.synchronize()
            with owner_context(metadata, self.stream), torch.cuda.stream(self.stream):
                self.target(input_ids, metadata, packet)
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with (
                owner_context(metadata, self.stream),
                self.group.graph_capture(GraphCaptureContext(self.stream)),
                torch.cuda.graph(graph, pool=self.graph_pool, stream=self.stream),
            ):
                hidden, auxiliary = self.target(input_ids, metadata, packet)
            self.graph_pool = graph.pool()
            self.graphs[key] = graph, hidden, auxiliary
            self.graph_coverage.captured()
            torch.cuda.synchronize()
            self.publish_graph_coverage()
        graph, local_hidden, auxiliary = self.graphs[key]
        graph.replay()
        packet_hidden = self.group.all_gather(local_hidden, use_custom=True, dim=0)
        hidden = packet_hidden.index_select(0, inverse)
        self.active = (metadata, original, auxiliary, inverse)
        logits = self.model.compute_logits(hidden)
        self.counters["owner_target_steps"] += 1
        self.counters[f"FFN_M{rows.capacity * 4}_steps"] += 1
        self.counters["executed_owner_query_rows"] += original.step.scheduled
        writes = written_page_units(
            original.step,
            original.block_table_rows,
            self.cache.geometry,
            compress_buffers={
                ratio: self.runner.forward_vars[f"v4_compress_plan_{ratio}"]
                for ratio in original.step.plans
            },
        )
        for span in original.step.requests:
            pages = set(writes[span.request_id])
            previous = self.dirty.get(span.request_id)
            if previous is not None:
                if previous[:2] != (self._owners[span.request_id], span.slot):
                    raise ValueError(
                        "Owner epoch moved before native state reconciliation"
                    )
                pages.update(previous[2])
            self.dirty[span.request_id] = (
                self._owners[span.request_id],
                span.slot,
                tuple(sorted(pages)),
            )
        self.done.record()
        self.done_recorded = True
        return logits, hidden

    def release_graphs(self):
        """Retire this owner's captures before their communicators are destroyed.

        The caller must stop replay and synchronize all ranks first. Keep the
        sealed coverage contract; releasing captures does not enable new ones.
        """
        count = len(self.graphs)
        for graph, _, _ in self.graphs.values():
            graph.reset()
        self.graphs.clear()
        self.graph_pool = None
        return count

    def publish_graph_coverage(self, *, missing_key=None):
        """Exact local bindings, written only on capture/seal/refusal.

        These keys are an inventory of observed executions, not a claim that
        every future scheduler step is covered. The sealed miss guard checks
        that claim on each real forward, collectively across the four ranks.
        """
        if self.graph_phase_file is None:
            return
        path = self.graph_phase_file.parent / f"owner_graph_rank{self.rank}.json"
        payload = (
            json.dumps(
                {
                    "rank": self.rank,
                    "stable_inputs": self.stable_graphs,
                    "coverage": self.graph_coverage.report(),
                    "keys": list(self.graphs),
                    "missing_key": missing_key,
                }
            )
            + "\n"
        )
        if len(payload.encode()) > 1_000_000:
            raise RuntimeError("Owner graph coverage inventory exceeds 1 MB")
        temporary = path.with_suffix(".tmp")
        temporary.write_text(payload)
        temporary.replace(path)

    def draft_context(self, drafter):
        import torch
        import torch.distributed as dist

        metadata, original, auxiliary, inverse = self.active
        recorder = ContextPacket(metadata.step, window_fp8=original.cache.window_fp8)
        projection = copy.copy(metadata)
        projection.cache = recorder
        with owner_context(projection):
            drafter.model.write_context_kv(auxiliary, metadata.step.positions)
        layers = tuple(layer.spec.layer_id for layer in drafter.model.context_layers)
        if tuple(recorder.keys) != layers:
            raise ValueError("Incomplete owner-to-global draft context packet")
        if recorder.window_fp8:
            # One lossless byte collective carries both planes of each row.
            local = torch.stack(
                [
                    torch.cat(
                        tuple(t[0].view(torch.uint8) for t in recorder.keys[layer]),
                        dim=-1,
                    )
                    for layer in layers
                ]
            )
        else:
            local = torch.stack([recorder.keys[layer][0] for layer in layers])
        packets = [torch.empty_like(local) for _ in range(4)]
        dist.all_gather(packets, local, group=self.group.device_group)
        stage_major = (
            torch.stack(packets)
            .permute(1, 0, 2, 3)
            .reshape(len(layers), -1, local.shape[-1])
        )
        keys = stage_major.index_select(1, inverse)
        # Native writer uses global target slots, absolute positions and ring133.
        for i, layer in enumerate(layers):
            payload = keys[i : i + 1]
            if recorder.window_fp8:
                payload = (
                    payload[..., :512].view(torch.float8_e4m3fn),
                    payload[..., 512:].view(torch.float8_e8m0fnu),
                )
            original.cache.write_window(layer, payload, original.step)
        self.counters["compact_draft_context_updates"] += 1


def install():
    from functools import wraps

    import torch

    from atom.model_engine.model_runner import ModelRunner
    from atom.model_ops.attentions.deepseek_v41.backend import (
        DeepseekV41MetadataBuilder,
    )
    from atom.models.deepseek_v41.runtime import DeepseekV41RuntimeModel
    from atom.spec_decode.draft_graph import DraftGraph
    from atom.spec_decode.dspark_proposer import DSparkProposer
    from atom.utils import backends

    backends.VLLM_CACHE_ROOT = os.environ["DSV41_OWNER_SERVING_COMPILE_CACHE"]
    original_forward = ModelRunner.forward
    original_run = ModelRunner.run_model
    original_q = ModelRunner._dspark_apply_q_bucket
    original_kv = DSparkProposer.compute_draft_kv
    original_aux = DSparkProposer.aux_for
    original_block = DraftGraph.run
    original_state_copies = DeepseekV41MetadataBuilder.execute_paged_state_copies
    original_relocate = DeepseekV41MetadataBuilder.relocate_state_slots
    original_replayed = DeepseekV41RuntimeModel._replayed

    @wraps(original_replayed)
    def replayed(self, state, num_tokens, plan):
        owned = state.row_sharded
        result = original_replayed(self, state, num_tokens, plan)
        serving = getattr(self, "_owner_serving", None)
        if owned and serving is not None:
            serving.counters["accepted_retained_prefill_forwards"] += 1
            serving.counters["accepted_retained_prefill_rows"] += num_tokens
        return result

    @wraps(original_forward)
    @torch.inference_mode()
    def forward(self, batch):
        if not hasattr(self, "_owner_serving") and not batch.is_dummy_run:
            self._owner_serving = OwnerServing(self)
        serving = getattr(self, "_owner_serving", None)
        try:
            if serving is not None:
                serving.before_prepare(batch)
            output = original_forward(self, batch)
        except BaseException as error:
            if serving is not None:
                failure = {
                    "rank": serving.rank,
                    "error": repr(error)[:2000],
                    "mode": serving.mode,
                    "requests": int(batch.total_seqs_num),
                    "owner_resident_counts": [
                        sum(
                            owner == rank
                            for _, owner in serving.registry.slots.values()
                        )
                        for rank in range(4)
                    ],
                }
                serving.evidence.with_name(
                    f"owner_failure_rank{serving.rank}.json"
                ).write_text(json.dumps(failure) + "\n")
            raise
        if serving is not None and not batch.is_dummy_run:
            import torch

            serving.done.record(torch.cuda.current_stream())
            serving.done_recorded = True
            serving.counters["client_output_batches"] += int(bool(output.req_ids))
            serving.counters["client_output_tokens"] += sum(
                sum(int(token) >= 0 for token in row) for row in output.token_ids
            )
            if serving.counters["runner_batches"] % 16 == 0:
                serving.publish()
        return output

    @wraps(original_run)
    def run(self, input_ids, batch=None):
        serving = getattr(self, "_owner_serving", None)
        if (
            serving is not None
            and batch is getattr(serving, "candidate_batch", None)
            and serving.mode != "native"
        ):
            return serving.run(input_ids, batch)
        return original_run(self, input_ids, batch)

    @wraps(original_q)
    def q_bucket(self, batch):
        serving = getattr(self, "_owner_serving", None)
        if serving is not None:
            # The candidate owns flat prefix admission. Original native FULL
            # ragged guards remain intact for every other runner.
            return None
        return original_q(self, batch)

    @wraps(original_kv)
    def draft_kv(self, positions, hidden_states, next_token_ids):
        serving = getattr(self.runner, "_owner_serving", None)
        if serving is not None and serving.active is not None:
            serving.draft_context(self)
            return
        return original_kv(self, positions, hidden_states, next_token_ids)

    @wraps(original_aux)
    def aux(self, hidden_states):
        serving = getattr(self.runner, "_owner_serving", None)
        if serving is not None and serving.active is not None:
            # propose validates presence only; context KV was already published.
            return list(serving.active[2].split(5120, dim=-1))
        return original_aux(self, hidden_states)

    @wraps(original_block)
    def block(self, running_bs, **kwargs):
        output = original_block(self, running_bs, **kwargs)
        proposer = getattr(getattr(self, "forward", None), "__self__", None)
        serving = getattr(getattr(proposer, "runner", None), "_owner_serving", None)
        if serving is not None and self is getattr(proposer, "block", None):
            batch = getattr(serving, "current_batch", None)
            if batch is not None:
                serving.record_confidence(output[1], batch.req_ids)
        return output

    ModelRunner.forward, ModelRunner.run_model = forward, run
    ModelRunner._dspark_apply_q_bucket = q_bucket
    DSparkProposer.compute_draft_kv, DSparkProposer.aux_for = draft_kv, aux
    DraftGraph.run = block

    def state_copies(self, *args, **kwargs):
        serving = getattr(self.model_runner, "_owner_serving", None)
        if serving is not None:
            stores = args[0] if args else kwargs["stores"]
            restores = args[1] if len(args) > 1 else kwargs["restores"]
            serving.reconcile(
                request_ids=(),
                slots=(
                    *[op.src_slot for op in stores],
                    *[op.dst_slot for op in restores],
                ),
                pages=(unit for op in (*stores, *restores) for unit in op.unit_ids),
            )
        return original_state_copies(self, *args, **kwargs)

    def relocate(self, *args, **kwargs):
        serving = getattr(self.model_runner, "_owner_serving", None)
        if serving is not None:
            pairs = args[0] if args else kwargs["pairs"]
            serving.reconcile(
                request_ids=(), slots=(slot for pair in pairs for slot in pair)
            )
        return original_relocate(self, *args, **kwargs)

    DeepseekV41MetadataBuilder.execute_paged_state_copies = state_copies
    DeepseekV41MetadataBuilder.relocate_state_slots = relocate
    DeepseekV41RuntimeModel._replayed = replayed
    os.environ["DSV41_OWNER_SERVING_INSTALLED"] = "1"
