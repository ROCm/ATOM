# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Private V4 serving integration; the engine does not own Mono policy.

The existing metadata builder identifies the engine owner. Keep publication,
graph identity and storage invalidation together here,
without adding configuration fields or forwarding flags to ModelRunner.
"""

from __future__ import annotations

import json
import logging
from contextlib import contextmanager, nullcontext

import torch

from atom.mono.runtime.consensus import MonoUnsupported, bind_agreed, tp_agree

logger = logging.getLogger("atom")
# Shutdown cannot prove dead peers have stopped reading exported IPC storage.
# Hold the owner until process termination; live rebind still closes collectively.
_EXIT_OWNERS = []


class V4MonoEngine:
    def __init__(self, session):
        self.session = session
        self.owner = session.builder.model_runner
        self.selected = False
        self._lane_graph = False
        self._capture_selected = None
        self._exit_retained = False
        self.native_graph_replays = 0
        self._slot_owners = {}

    def prepare_step(self, input_ids, positions, context, batch):
        """Authorize the existing batch and choose its actual execution graph."""
        self.abort_step()
        mono, runner = self.session, self.owner
        fc = context.context
        mode = fc.forward_mode
        allow_auto = bool(
            batch is not None
            and (batch.temperatures == 0).all()
            and not any(batch.return_logprobs)
        )
        # Publication and native eager execution share the canonical padded
        # views. Exact-live tails select their own graph, never the padded one.
        if mode.use_cudagraph and not fc.is_prefill and mode.running_tokens_are_unified:
            input_ids, positions = runner._padded_decode_inputs(mode)
        if batch is not None:
            self._track_slot_owners(batch)
            mono.state_maintained(batch.state_maintenance_ops)
        self.selected = mono.publish(
            input_ids, positions, context, allow_auto=allow_auto
        )
        self._lane_graph = mono.lane_graph_selected
        if not fc.is_dummy_run:
            mono.live_started = True
        graph_key = (
            mono.execution_graph_key
            if self.selected
            else (fc.running_bs, mode.max_seqlen_q)
        )
        graph_is_mono = mono.graph_is_current(graph_key)
        if self.selected and graph_is_mono:
            graph_key = (*graph_key, "mono")
        use_graph = bool(
            mode.use_cudagraph
            and graph_key in runner.graphs
            and (not self.selected or graph_is_mono)
            and not (
                self.selected
                and (mono.lane_graph_selected or mono.eager_layout_selected)
            )
        )
        return input_ids, positions, graph_key, use_graph

    def _track_slot_owners(self, batch):
        """Detect slot reuse from the existing request/slot snapshot."""
        adapter = self.session.adapter
        if adapter is None:
            return
        changed = []
        for row, slots in zip(batch.state_rows, batch.state_slots, strict=True):
            request = batch.req_ids[row]
            for slot in slots:
                if self._slot_owners.get(slot) != request:
                    self._slot_owners[slot] = request
                    changed.append(adapter.geometry.physical_slot(slot))
        if changed:
            self.session.slots_reset(changed)

    def model_forwarded(self):
        """Count live model-boundary execution, not startup capture warmups."""
        if self._capture_selected is not None or not self.selected:
            return
        if self._lane_graph:
            self.session.lane_graph_forwards += 1
        else:
            self.session.eager_forwards += 1

    def replay(self, graph):
        # The recorded graph already contains dispatch. No eager authorization
        # may survive a replay, including a replay which raises.
        self.session.armed = False
        graph.replay()
        if self.selected:
            self.session.graph_replays += 1
        else:
            self.native_graph_replays += 1

    def finish_step(self):
        self.session.armed = False
        self.selected = False
        self._lane_graph = False

    def abort_step(self):
        self.finish_step()

    def before_capture(self):
        mono = self.session
        if not tp_agree(not mono.live_started, mono.group):
            raise MonoUnsupported(
                "retire/reset live requests before recapturing V4 Mono graphs"
            )
        self.abort_step()

    def capture_ladder(self, declared_q, *, piecewise):
        mono = self.session
        queries = list(mono.profile.capture_query_lengths(declared_q))
        if queries != list(declared_q):
            logger.info(
                "V4 Mono startup capture query superset: %s; "
                "DSpark scheduling buckets unchanged: %s",
                queries,
                declared_q,
            )
        widths = (
            self.owner.capture_sizes
            if piecewise
            else mono.profile.capture_request_widths(self.owner.capture_sizes)
        )
        return queries, widths

    @contextmanager
    def capture(self, input_ids, positions, context):
        mono = self.session
        key = (context.context.running_bs, context.attn_metadata.max_seqlen_q)
        try:
            selected = mono.prepare_engine_capture(input_ids, positions, context)
            self._capture_selected = selected
            preserve = (
                mono.preserve_tail_capture()
                if selected and key[0] not in self.owner.capture_sizes
                else nullcontext()
            )
            with preserve:
                yield
                if key in self.owner.graphs:
                    if selected:
                        self.owner.graphs[(*key, "mono")] = self.owner.graphs[key]
                        self._capture_selected = False
                        self._capture_native(input_ids, positions, key)
                    mono.remember_graph(key, selected)
        finally:
            self._capture_selected = None
            mono.finish_capture_publication()

    def _capture_native(self, input_ids, positions, key):
        """Keep the existing native FULL key as the rollback route.

        This runs in the engine's startup graph-capture scope, using its fixed
        inputs, metadata, output and pool. No live step captures a graph.
        """
        runner = self.owner
        outputs = runner.forward_vars["outputs"][: input_ids.numel()]
        snapshots = []
        bind_agreed(
            lambda: self.session._snapshot_capture_views(
                self.session.adapter.capture_write_views(), snapshots
            ),
            self.session.group,
        )
        try:
            self.session.finish_capture_publication()
            outputs.copy_(self.session.model(input_ids, positions))
            if runner.logits_in_graph:
                self.session.model.compute_logits(outputs)
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            # Match the outer engine capture stream. Native multi-stream ops
            # join work to it; PyTorch's implicit side stream is not that owner.
            with torch.cuda.graph(
                graph, runner.graph_pool, stream=torch.cuda.current_stream()
            ):
                outputs.copy_(self.session.model(input_ids, positions))
                if runner.logits_in_graph:
                    runner.graph_logits[key] = self.session.model.compute_logits(
                        outputs
                    )
            runner.graphs[key] = graph
            torch.cuda.synchronize()
        except BaseException:
            # A failed capture may invalidate the HIP context. Do not submit
            # restore kernels or replace its root error; startup must abort.
            self.session.disabled = True
            raise
        else:
            try:
                self.session._restore_capture_views(snapshots)
            except BaseException:
                self.session.disabled = True
                raise

    def prepare_model_forward(self):
        if self._capture_selected is not None:
            # Both warmup and actual FULL capture consume one authorization.
            self.session.armed = self._capture_selected

    def _retire_graphs(self):
        self.abort_step()
        runner = self.owner
        torch.cuda.synchronize()
        for name in ("graphs", "graph_logits"):
            graphs = getattr(runner, name, None)
            if graphs is not None:
                graphs.clear()

    def shutdown(self):
        if self._exit_retained:
            return
        self._exit_retained = True
        _EXIT_OWNERS.append(self.session)
        logger.info(
            "ATOM_V4_MONO_SERVING_STATS %s",
            json.dumps(self.session.serving_status(), sort_keys=True),
        )
        self.abort_step()
        # A dead TP peer cannot acknowledge retirement. Keep IPC mappings and
        # exported storage alive until process exit rather than free memory a
        # survivor may still read. Process teardown reclaims them locally.
        self.session.disabled = True
        self.session.finish_capture_publication()
        for name in ("graphs", "graph_logits"):
            graphs = getattr(self.owner, name, None)
            if graphs is not None:
                graphs.clear()

    def cache_rebinding(self):
        if self.session.adapter is not None:
            self._retire_graphs()
            self.owner.graph_pool = None
            self.session.close()

    def cache_bound(self):
        self._slot_owners.clear()
        self.session.prepare(cache_rebound=True)

    def prepare_weights(self):
        return self.session.prepare_weights()
