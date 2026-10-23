# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Model-owned publication and graphs, using the native region graph protocol."""

from __future__ import annotations

import json
import logging

import torch

from atom.mono.runtime.consensus import MonoUnsupported, bind_agreed, tp_agree
from atom.utils.forward_context import get_forward_context, request_bucket
from atom.utils.region_graph import ReqGraph

logger = logging.getLogger("atom")
# Dead peers cannot acknowledge retirement of exported IPC storage.
_EXIT_OWNERS = []


class V4MonoReqGraph(ReqGraph):
    """Native FULL buckets with independently retained Mono alternatives."""

    def __init__(self, engine):
        super().__init__()
        self.engine = engine
        self.native = ReqGraph()
        self.cache_generation = engine.cache_generation

    def has(self, key, phase=None):
        return self.native.has(key, phase) or super().has(key, phase)

    def keys(self):
        return list(dict.fromkeys((*self.native.keys(), *super().keys())))

    def release_graphs(self):
        return self.native.release_graphs() + super().release_graphs()

    def record(self, forward, args=(), kwargs=None, key=None, phase=None):
        engine, mono = self.engine, self.engine.session
        selected = engine._capture_selected
        shape = engine._capture_key
        snapshots = []
        try:
            if selected:
                super().record(forward, args, kwargs, key, phase)
                torch.cuda.synchronize()
                bind_agreed(
                    lambda: mono._snapshot_capture_views(
                        mono.adapter.capture_write_views(), snapshots
                    ),
                    mono.group,
                )
            engine._capture_selected = False
            mono.finish_capture_publication()
            # The fallback must be warmed as Native before its recording.
            forward(*args, **(kwargs or {}))
            torch.cuda.synchronize()
            output = self.native.record(forward, args, kwargs, key, phase)
            torch.cuda.synchronize()
            if selected:
                mono._restore_capture_views(snapshots)
            mono.remember_graph(shape, bool(selected))
            return output
        except BaseException:
            mono.disabled = True
            raise
        finally:
            engine._capture_key = engine._capture_selected = None
            mono.finish_capture_publication()

    def replay(self, args=(), kwargs=None, key=None, phase=None):
        engine, runner = self.engine, self.engine.owner
        try:
            if (
                self.cache_generation != engine.cache_generation
                or runner.req_graph is not self
            ):
                raise MonoUnsupported("V4 graph belongs to a retired cache/capture")
            context = get_forward_context()
            ids, positions = runner._padded_inputs(context.context.forward_mode)
            ids, positions, chosen, use_graph = engine.prepare_step(
                ids, positions, context, engine.session.builder._mono_batch
            )
            if not engine.selected:
                engine.native_graph_replays += 1
                return self.native.replay(args, kwargs, key, phase)
            if use_graph:
                output = super().replay(args, kwargs, request_bucket(*chosen), phase)
                engine.session.graph_replays += 1
                return output
            output = engine.session.forward(ids, positions)
            engine.model_forwarded()
            runner.forward_vars["outputs"][: output.shape[0]].copy_(output)
            return None  # TP8 computes its logits outside the native FULL graph.
        finally:
            engine.abort_step()


class V4MonoEngine:
    def __init__(self, session):
        self.session = session
        self.owner = session.builder.model_runner
        self.selected = False
        self._lane_graph = False
        self._capture_selected = None
        self._capture_key = None
        self._target_graph = None
        self._exit_retained = False
        self.cache_generation = 0
        self.native_graph_replays = 0
        self._slot_owners = {}

    def prepare_step(self, input_ids, positions, context, batch):
        """Authorize live metadata before either graph can write state."""
        self.abort_step()
        mono, runner = self.session, self.owner
        fc, mode = context.context, context.context.forward_mode
        allow_auto = bool(
            batch is not None
            and (batch.temperatures == 0).all()
            and not any(batch.return_logprobs)
        )
        if mode.use_cudagraph and not fc.is_prefill and mode.running_tokens_are_unified:
            input_ids, positions = runner._padded_inputs(mode)
        if batch is not None:
            self._track_slot_owners(batch)
            mono.state_maintained(batch.state_maintenance_ops)
        self.selected = mono.publish(
            input_ids, positions, context, allow_auto=allow_auto
        )
        self._lane_graph = mono.lane_graph_selected
        if not fc.is_dummy_run:
            mono.live_started = True
        key = (
            mono.execution_graph_key
            if self.selected
            else (fc.running_bs, mode.max_seqlen_q)
        )
        use_graph = bool(
            mode.use_cudagraph
            and self._target_graph is not None
            and self._target_graph.has(request_bucket(*key))
            and mono.graph_is_current(key)
            and not (mono.lane_graph_selected or mono.eager_layout_selected)
        )
        return input_ids, positions, key, use_graph

    def _track_slot_owners(self, batch):
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
        if self._capture_selected is not None or not self.selected:
            return
        if self._lane_graph:
            self.session.lane_graph_forwards += 1
        else:
            self.session.eager_forwards += 1

    def finish_step(self):
        self.session.armed = False
        self.selected = False
        self._lane_graph = False

    def abort_step(self):
        self.finish_step()

    def bind_graphs(self):
        mono = self.session
        if not tp_agree(not mono.live_started, mono.group):
            raise MonoUnsupported(
                "retire/reset live requests before recapturing V4 Mono"
            )
        self.abort_step()
        self._capture_key = self._capture_selected = None
        mono.finish_capture_publication()
        self._target_graph = self.owner.req_graph = V4MonoReqGraph(self)

    def prepare_model_forward(self, input_ids, positions):
        context, mono = get_forward_context(), self.session
        if context.context.is_prefill:
            self._capture_key = self._capture_selected = None
            mono.finish_capture_publication()
        if context.in_hipgraph and not context.context.is_prefill:
            key = (context.context.running_bs, context.attn_metadata.max_seqlen_q)
            if (
                self._capture_key != key
                and not torch.cuda.is_current_stream_capturing()
            ):
                self._capture_key = key
                self._capture_selected = mono.prepare_engine_capture(
                    input_ids, positions, context
                )
            mono.armed = bool(self._capture_selected)
        elif context.context.forward_mode is not None:
            self.prepare_step(input_ids, positions, context, mono.builder._mono_batch)

    def _retire_graphs(self):
        from atom.utils.graph_holders import release_registered_graphs

        self.abort_step()
        torch.cuda.synchronize()
        release_registered_graphs()
        self._target_graph = None

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
        self.session.disabled = True
        self.session.finish_capture_publication()

    def cache_rebinding(self):
        self.cache_generation += 1
        if self.session.adapter is not None:
            self._retire_graphs()
            self.session.close()

    def cache_bound(self):
        self._slot_owners.clear()
        self.session.prepare(cache_rebound=True)
