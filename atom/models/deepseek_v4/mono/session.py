# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
# Modifications: namespace/dependency extraction for ATOM, 2026-10-13.

"""DeepSeek-V4 serving lifecycle, graph admission and publication."""

from __future__ import annotations

import logging
import threading
import weakref
from contextlib import contextmanager
from dataclasses import dataclass
from itertools import pairwise

import torch

from atom.models.deepseek_v4.mono.binding import input_identity
from atom.models.deepseek_v4.mono.cache import V4CacheAdapter
from atom.models.deepseek_v4.mono.contract import (
    BatchContinuation,
    ContinuationRow,
    continues_batch,
)
from atom.models.deepseek_v4.mono.engine import V4MonoEngine
from atom.models.deepseek_v4.mono.profile import ORDINARY_PROFILE
from atom.models.deepseek_v4.mono.runner import V4MonoDecodeRunner
from atom.mono.runtime.consensus import MonoUnsupported, bind_agreed, tp_agree

logger = logging.getLogger("atom")

_session_lease_lock = threading.Lock()
_session_lease_owner: weakref.ReferenceType | None = None
_session_generation = 0


def _live_session():
    return None if _session_lease_owner is None else _session_lease_owner()


__all__ = ["V4MonoSession"]


@dataclass(frozen=True)
class _DraftContextPublication:
    owner_generation: int
    slot_generations: tuple[tuple[int, int], ...]
    completion: torch.cuda.Event
    positions: torch.Tensor | None = None
    segments: tuple[int, ...] = ()
    write_per_batch: int = 0
    device_segments: torch.Tensor | None = None


@dataclass(frozen=True)
class _DraftGraphRegistration:
    """Session-owned startup capture, not a live replay credential."""

    owner_generation: int
    graph_generation: int
    runner: object
    capture: object
    publication: object = None


def _agree_draft_graph(selected, group):
    """Agree both graph/eager alternatives in one pre-launch collective."""
    if group is None:
        return selected
    import torch.distributed as dist

    # MIN(flag, -flag) returns min/max, including the mixed-rank case.
    flag = int(selected)
    extrema = torch.tensor((flag, -flag), dtype=torch.int32)
    dist.all_reduce(extrema, op=dist.ReduceOp.MIN, group=group)
    if int(extrema[0]) != -int(extrema[1]):
        raise MonoUnsupported("draft graph selection differs across TP ranks")
    return bool(extrema[0])


class V4MonoSession:
    """Own publication and graph authorization for the loaded native model.

    Publication happens before graph replay, because ATOM rotates H2D buffers.
    A graph captured as Mono is never replayed for a refused/forked step.
    This bring-up gate votes outside capture; its cost remains in step timing.
    """

    profile = ORDINARY_PROFILE

    def __init__(self, model, config, builder, profile=ORDINARY_PROFILE):
        from aiter.dist.parallel_state import (
            get_tensor_model_parallel_rank,
            get_tp_group,
        )

        self.model, self.config, self.builder = model, config, builder
        self.profile = profile
        self.rank, self.group = (
            get_tensor_model_parallel_rank(),
            get_tp_group().cpu_group,
        )
        self.runner = self.adapter = None
        self.armed = False
        self.disabled = False
        self.refusal = None
        self.graph_keys = set()
        self.graph_generations: dict[tuple[int, int], int] = {}
        self.graph_layouts = {}
        self.pending_capture_layout = None
        self.execution_graph_key = None
        self.selected_steps = 0
        self.graph_replays = 0
        self.eager_forwards = 0
        self.draft_backbone_forwards = 0
        self.draft_graph_replays = 0
        self.lane_graph_forwards = 0
        self.lane_graph_selected = False
        self.eager_layout_selected = False
        self.decode_attempts = 0
        self.fallback_steps = 0
        self.fallback_reasons: dict[str, int] = {}
        self.microbucket_stacks = 0
        self.index_refresh_executions = 0
        self.index_reuse_executions = 0
        self.index_suppressed_rows = 0
        self.live_started = False
        self.entry_sources = ()
        self.output = None
        self.owner_generation = 0
        self.graph_generation = 0
        self.publication_generation = 0
        self.publication_slot_generations: tuple[tuple[int, int], ...] = ()
        self.capture_publication = False
        self.slot_generations: dict[int, int] = {}
        self.batch_continuation: BatchContinuation | None = None
        self.full_publications = 0
        self.fast_publications_by_width: dict[int, int] = {}
        self._draft_context: _DraftContextPublication | None = None
        self.draft_runner = None
        self._draft_frontier = None
        self._draft_metadata = {}
        self._draft_metadata_completion = None
        self._draft_graphs = {}
        self.engine = V4MonoEngine(self)

    def _acquire_lease(self):
        """Own the one live Mono arena/cache binding in this rank process."""
        global _session_lease_owner, _session_generation
        with _session_lease_lock:
            holder = _live_session()
            if holder is self:
                raise MonoUnsupported("this V4 Mono session is already prepared")
            if holder is not None:
                raise MonoUnsupported(
                    "another V4 Mono session still owns the live arena/cache binding"
                )
            _session_generation += 1
            self.owner_generation = _session_generation
            _session_lease_owner = weakref.ref(self)

    def _release_lease(self):
        global _session_lease_owner
        with _session_lease_lock:
            if _live_session() is self:
                _session_lease_owner = None
        self.owner_generation = 0

    def owns_live_session(self) -> bool:
        with _session_lease_lock:
            return self.owner_generation != 0 and _live_session() is self

    def uses_native_draft_initialization(self, context) -> bool:
        """Only the engine's pre-KV dummy warmup remains native by design.

        The installer precedes warmup_model, but get_num_blocks prepares mono
        weights later. This is not a live fallback or a graph-capture bypass.
        Once any mono resource is bound, warmup must use its own state-safe
        startup protocol rather than accidentally warming native kernels.
        """
        return bool(
            context.is_dummy_run
            and context.is_prefill
            and not self.live_started
            and self.owner_generation == 0
            and self.runner is None
            and self.adapter is None
            and getattr(self, "draft_runner", None) is None
        )

    def slots_reset(self, slots) -> None:
        """Invalidate any publication that predates request-state slot reuse."""
        changed = {int(slot) for slot in slots}
        self._draft_frontier = None
        for slot in changed:
            self.slot_generations[slot] = self.slot_generations.get(slot, 0) + 1
        draft_context = getattr(self, "_draft_context", None)
        if draft_context is not None and any(
            slot in changed for slot, _ in draft_context.slot_generations
        ):
            self._draft_context = None
        if self.armed and any(
            slot in changed for slot, _ in self.publication_slot_generations
        ):
            self.armed = False
            self.publication_generation = 0
            self.publication_slot_generations = ()
            if self.adapter is not None:
                self.adapter.clear_publication()
            self.refusal = (
                "request-state slot reset invalidated the published Mono step"
            )
        if self.batch_continuation is not None and any(
            row.slot in changed for row in self.batch_continuation.rows
        ):
            self.batch_continuation = None

    def _has_draft_resources(self) -> bool:
        return bool(
            getattr(self, "draft_runner", None) is not None
            or getattr(self, "_draft_graphs", {})
            or getattr(self, "_draft_metadata_completion", None) is not None
        )

    def _wait_draft_reader(self) -> None:
        """Order current-stream writes after the previous draft read."""
        previous = getattr(self, "_draft_metadata_completion", None)
        if previous is not None:
            torch.cuda.current_stream().wait_event(previous)

    def run_state_maintenance(self, write, *, empty=False) -> None:
        """Retire draft readers before native prefix/COW pool copies.

        Metadata build precedes target/context publication, so the context
        producer's reader wait is too late to protect these earlier writes.
        The callback retains the original builder's copy order and stream.
        """
        # Every TP rank calls this, including an empty local batch. A local
        # early return would strand peers in the reader/producer consensus.
        if tp_agree(empty, self.group):
            return
        if not tp_agree(not empty, self.group):
            self.disabled = True
            self._draft_context = self._draft_frontier = None
            raise MonoUnsupported("state maintenance presence differs across TP ranks")
        if tp_agree(not self._has_draft_resources(), self.group):
            # Pre-KV startup/native-only maintenance has no joint readers.
            write()
            return

        def validate():
            if self.disabled or not self.owns_live_session():
                raise MonoUnsupported("state maintenance requires the joint owner")
            if torch.cuda.is_current_stream_capturing():
                raise MonoUnsupported("state maintenance cannot run inside capture")

        try:
            bind_agreed(validate, self.group)
            # Agree on successful reader ordering before any rank submits a
            # state copy. A peer's failed wait cannot permit our early writes.
            bind_agreed(self._wait_draft_reader, self.group)
            bind_agreed(write, self.group)
        except Exception:
            # Some ranks may already have copied state; do not authorize a
            # retry or a native fallback on the uncertain shared history.
            self.disabled = True
            self._draft_context = self._draft_frontier = None
            raise

    def state_maintained(self, operations) -> None:
        """Invalidate physical frontiers after native pool-slot maintenance.

        Native metadata preparation enqueues moves/restores on the compute
        stream before publication. Their pool IDs are not the plane positions
        recorded by Mono; unchanged tensor addresses do not mean unchanged
        history. Stores only read a slot and need no invalidation.
        """
        if operations.empty or self.adapter is None:
            return
        changed = {dst for _, dst in operations.relocations}
        changed.update(op.dst_slot for op in operations.checkpoint_restores)
        if changed:
            geometry = self.adapter.geometry
            self.slots_reset(geometry.physical_slot(slot) for slot in changed)

    def close(self):
        integration = getattr(self, "engine", None)
        if integration is not None:
            integration.abort_step()
        # A graph generation is a capture lifetime, not merely a session-owner
        # lifetime.  Advancing it here makes any external/stale continuation
        # token unusable even if this Python session is prepared again.
        self.graph_generation = getattr(self, "graph_generation", 0) + 1
        self.armed = False
        self.execution_graph_key = None
        self.lane_graph_selected = False
        self.eager_layout_selected = False
        self.graph_keys.clear()
        self.graph_generations.clear()
        self.graph_layouts.clear()
        self.pending_capture_layout = None
        self.publication_generation = 0
        self.publication_slot_generations = ()
        self.capture_publication = False
        self.batch_continuation = None
        self._draft_context = None
        self._draft_frontier = None
        # Captures own graph-pool outputs as well as kernel arguments. Retire
        # their readers while those owners are still retained. A failed live
        # submission may not have produced an event, so it needs a full fence.
        completion = getattr(self, "_draft_metadata_completion", None)
        draft_runner = getattr(self, "draft_runner", None)
        # Registration can fail on just one rank. All ranks must choose the
        # same retirement branch, including eager-only draft owners/readers.
        if not tp_agree(not self._has_draft_resources(), getattr(self, "group", None)):

            def retire_draft_graphs():
                if self.disabled or completion is None:
                    torch.cuda.synchronize()
                    return
                completion.synchronize()

            try:
                bind_agreed(retire_draft_graphs, self.group)
            except Exception:
                self.disabled = True
                raise
        # Unpublish before either runner releases captured peer/kernel storage.
        getattr(self, "_draft_graphs", {}).clear()
        if self.adapter is not None:
            self.adapter.clear_publication()
        if self.runner is not None:
            self.runner.close()
        if draft_runner is not None:
            draft_runner.close()
        self.draft_runner = None
        self._draft_metadata = {}
        self._draft_metadata_completion = None
        self.runner = self.adapter = None
        self._frontier_segments = {}
        self._frontier_workspace = {}
        self.entry_sources = ()
        self.output = None
        self._release_lease()

    def write_draft_context(
        self,
        write,
        slots,
        *,
        expected_rows=None,
        positions=None,
        cu_seqlens=None,
        write_per_batch=None,
    ) -> None:
        """Publish completion of the actual native context producer.

        Called outside graph capture with the target's host physical slots,
        before draft metadata replaces them. The event orders device writes;
        owner/slot generations authorize their use. A failed write leaves no
        publication and must not be retried via a native fallback. This is not
        recovery from a device collective failure: retire the serving owner.
        """
        self._draft_context = None
        self._draft_frontier = None
        resolved = []
        frontier = []

        def validate():
            if not self.profile.name.startswith("dspark_"):
                raise MonoUnsupported("draft context requires a DSpark session")
            if self.disabled or not self.owns_live_session():
                raise MonoUnsupported("draft context requires the live session owner")
            if torch.cuda.is_current_stream_capturing():
                raise MonoUnsupported("draft context publication cannot be captured")
            if slots is None:
                raise MonoUnsupported("draft context requires host physical slots")
            physical_slots = tuple(int(slot) for slot in slots)
            if (
                not physical_slots
                or (expected_rows is not None and len(physical_slots) != expected_rows)
                or any(slot < 0 for slot in physical_slots)
                or len(set(physical_slots)) != len(physical_slots)
            ):
                raise MonoUnsupported(
                    "draft context requires unique live physical slots"
                )
            resolved.append(
                tuple(
                    (slot, self.slot_generations.get(slot, 0))
                    for slot in physical_slots
                )
            )
            if any(
                value is not None for value in (positions, cu_seqlens, write_per_batch)
            ):
                if (
                    not isinstance(positions, torch.Tensor)
                    or positions.ndim != 1
                    or positions.dtype not in (torch.int32, torch.int64)
                    or not isinstance(cu_seqlens, torch.Tensor)
                    or cu_seqlens.ndim != 1
                    or cu_seqlens.dtype not in (torch.int32, torch.int64)
                    or cu_seqlens.device != positions.device
                    or cu_seqlens.numel() != len(physical_slots) + 1
                    or type(write_per_batch) is not int
                    or write_per_batch <= 0
                ):
                    raise MonoUnsupported("draft context frontier layout is invalid")
                # The sampler verdict is device-resident. Bring-up explicitly
                # pays this synchronization rather than guessing a host anchor.
                segments = tuple(cu_seqlens.tolist())
                if (
                    segments[0] != 0
                    or segments[-1] > positions.numel()
                    or any(a >= b for a, b in pairwise(segments))
                ):
                    raise MonoUnsupported(
                        "draft context requires nonempty request segments"
                    )
                snapshot = positions[: segments[-1]].clone()
                for a, b in pairwise(segments):
                    span = snapshot[a:b]
                    if bool(torch.any(span < 0).item()) or not bool(
                        torch.all(span[1:] == span[:-1] + 1).item()
                    ):
                        raise MonoUnsupported(
                            "draft context positions must be contiguous"
                        )
                device_segments = None
                if positions.is_cuda:
                    buffers = getattr(self, "_frontier_segments", {})
                    key = (positions.device, len(segments))
                    cached = buffers.get(key)
                    if cached is None:
                        device_segments = torch.tensor(
                            segments, dtype=torch.int64, device=positions.device
                        )
                    else:
                        previous, device_segments = cached
                        if previous != segments:
                            device_segments.copy_(torch.tensor(segments))
                    buffers[key] = (segments, device_segments)
                    self._frontier_segments = buffers
                frontier.append((snapshot, segments, write_per_batch, device_segments))

        bind_agreed(validate, self.group)
        completed = []

        def produce():
            # The preceding draft reads the same native context planes this
            # producer updates. Producer->consumer ordering alone does not
            # retire that reader when the next target pass uses another stream.
            self._wait_draft_reader()
            write()
            completion = torch.cuda.Event()
            completion.record(torch.cuda.current_stream())
            completed.append(completion)

        try:
            bind_agreed(produce, self.group)
        except Exception:
            # Either rank may already have submitted context writes before
            # local event failure or the post-producer TP vote fails. There
            # is no rollback contract for that uncertain shared KV state.
            self.disabled = True
            raise
        self._draft_context = _DraftContextPublication(
            self.owner_generation,
            resolved[0],
            completed[0],
            *(frontier[0] if frontier else (None, (), 0, None)),
        )

    def consume_draft_context(
        self,
        slots,
        *,
        expected_rows=None,
        anchor_rows=None,
        target_positions=None,
    ) -> None:
        """Authorize one draft pass and wait for its context producer.

        Live rows may be reordered or reduced by the sampler, never padded or
        drawn from another target pass. Retire the publication even on refusal
        so exceptions cannot authorize the next request's draft. Graph/static
        metadata and the draft runner's ownership still need separate checks.
        """
        publication = getattr(self, "_draft_context", None)
        self._draft_context = None
        self._draft_frontier = None
        frontier = []
        resolved = []

        def validate():
            if (
                publication is None
                or self.disabled
                or not self.owns_live_session()
                or publication.owner_generation != self.owner_generation
            ):
                raise MonoUnsupported("draft context has no current owner publication")
            if slots is None:
                raise MonoUnsupported("draft context requires host physical slots")
            physical_slots = tuple(int(slot) for slot in slots)
            published = dict(publication.slot_generations)
            if (
                not physical_slots
                or (expected_rows is not None and len(physical_slots) != expected_rows)
                or len(set(physical_slots)) != len(physical_slots)
                or any(
                    slot not in published
                    or published[slot] != self.slot_generations.get(slot, 0)
                    for slot in physical_slots
                )
            ):
                raise MonoUnsupported(
                    "draft context belongs to stale or unpublished slots"
                )
            if torch.cuda.is_current_stream_capturing():
                raise MonoUnsupported("draft context consumption cannot be captured")
            if publication.positions is not None:
                if (
                    not isinstance(anchor_rows, torch.Tensor)
                    or anchor_rows.ndim != 1
                    or anchor_rows.numel() != len(physical_slots)
                    or anchor_rows.dtype not in (torch.int32, torch.int64)
                    or anchor_rows.device != publication.positions.device
                    or not isinstance(target_positions, torch.Tensor)
                    or target_positions.ndim != 1
                    or target_positions.dtype != publication.positions.dtype
                    or target_positions.device != publication.positions.device
                ):
                    raise MonoUnsupported(
                        "draft anchor requires the actual sampler rows"
                    )
            elif anchor_rows is not None or target_positions is not None:
                raise MonoUnsupported("draft context has no target frontier snapshot")
            resolved.append((physical_slots, published))

        def validate_frontier():
            physical_slots, published = resolved[0]
            if publication.positions is not None:
                published_slots = tuple(
                    slot for slot, _ in publication.slot_generations
                )
                if (
                    publication.device_segments is not None
                    and anchor_rows.is_contiguous()
                    and target_positions.is_contiguous()
                ):
                    from atom.models.deepseek_v4.mono.step_meta import (
                        read_draft_frontier,
                    )

                    requests = tuple(
                        published_slots.index(slot) for slot in physical_slots
                    )
                    buffers = getattr(self, "_frontier_workspace", {})
                    key = (anchor_rows.device, anchor_rows.numel())
                    cached = buffers.get(key)
                    if cached is None:
                        request_ids = torch.tensor(
                            requests, dtype=torch.int32, device=anchor_rows.device
                        )
                        result = torch.empty(
                            (anchor_rows.numel(), 2),
                            dtype=torch.int64,
                            device=anchor_rows.device,
                        )
                    else:
                        previous, request_ids, result = cached
                        if previous != requests:
                            request_ids.copy_(torch.tensor(requests, dtype=torch.int32))
                    buffers[key] = (requests, request_ids, result)
                    self._frontier_workspace = buffers
                    values = read_draft_frontier(
                        anchor_rows,
                        target_positions,
                        publication.positions,
                        publication.device_segments,
                        requests,
                        publication.write_per_batch,
                        workspace=(request_ids, result),
                    )
                    if any(status == 1 for status, _ in values):
                        raise MonoUnsupported(
                            "draft anchor is outside written request rows"
                        )
                    if any(status != 0 for status, _ in values):
                        raise MonoUnsupported(
                            "draft anchor positions changed after publication"
                        )
                    frontier.append(
                        (
                            self.owner_generation,
                            tuple((slot, published[slot]) for slot in physical_slots),
                            tuple(position for _, position in values),
                        )
                    )
                    return
                rows = tuple(anchor_rows.tolist())
                for slot, row in zip(physical_slots, rows):
                    i = published_slots.index(slot)
                    start, end = publication.segments[i : i + 2]
                    # swa_write retains only this tail. Accepted anchor rows
                    # outside it cannot be authorized by a completed event.
                    if not max(start, end - publication.write_per_batch) <= row < end:
                        raise MonoUnsupported(
                            "draft anchor is outside written request rows"
                        )
                    if row >= target_positions.numel():
                        raise MonoUnsupported(
                            "draft anchor positions changed after publication"
                        )
                # Read both position vectors in one transfer rather than a
                # comparison and snapshot scalar per request. The sampler rows
                # have already passed the request/tail bounds above; retain
                # their device indices and the producer fence for this read.
                indices = anchor_rows.to(dtype=torch.int64)
                current, snapshot = torch.stack(
                    (
                        target_positions.index_select(0, indices),
                        publication.positions.index_select(0, indices),
                    )
                ).tolist()
                if current != snapshot:
                    raise MonoUnsupported(
                        "draft anchor positions changed after publication"
                    )
                frontier.append(
                    (
                        self.owner_generation,
                        tuple((slot, published[slot]) for slot in physical_slots),
                        tuple(snapshot),
                    )
                )

        bind_agreed(validate, self.group)
        try:

            def wait_and_validate_frontier():
                # Neither operation enters a TP device collective. A local
                # wait failure skips the read and joins the same final verdict
                # as a peer's read failure, before any backbone can execute.
                torch.cuda.current_stream().wait_event(publication.completion)
                validate_frontier()

            # This stage is unconditional, including publications without a
            # snapshot: rank-local fields must not select collective sequences.
            bind_agreed(wait_and_validate_frontier, self.group)
        except Exception:
            # Agree reader readiness before any peer enters the backbone.
            # Device snapshot reads must also be inside this failure boundary:
            # a failed wait/read/TP vote has no safe retry contract. Refuse
            # device frontier mismatches conservatively by retiring the owner.
            self.disabled = True
            raise
        self._draft_frontier = frontier[0] if frontier else None

    def forward_draft_block(
        self, draft_model, anchor_ids, anchor_positions, *, sampler=None
    ):
        """Run the draft under this target owner's one-use frontier.

        No independently leased draft session and no native retry. Only real
        unpadded requests are supported here. Registered exact-shape captures
        revalidate live metadata before replay; padding and default startup
        capture remain separate unfinished integration.
        """
        from atom.models.deepseek_v4.mono.draft_cache import native_draft_steps

        frontier = getattr(self, "_draft_frontier", None)
        self._draft_frontier = None
        resolved = []

        def resolve():
            runner = getattr(self, "draft_runner", None)
            if (
                self.disabled
                or not self.owns_live_session()
                or frontier is None
                or frontier[0] != self.owner_generation
                or runner is None
                or runner.draft_model is not draft_model
                or runner._closed
            ):
                raise MonoUnsupported("draft backbone requires the joint live owner")
            if torch.cuda.is_current_stream_capturing():
                raise MonoUnsupported(
                    "joint draft graph publication is not yet qualified"
                )
            slots = tuple(slot for slot, _ in frontier[1])
            if any(
                self.slot_generations.get(slot, 0) != generation
                for slot, generation in frontier[1]
            ):
                raise MonoUnsupported("draft backbone frontier has stale slots")
            if (
                anchor_ids.ndim != 1
                or anchor_positions.shape != anchor_ids.shape
                or anchor_positions.dtype not in (torch.int32, torch.int64)
                or anchor_ids.numel() != len(slots)
                or tuple(anchor_positions.tolist()) != frontier[2]
            ):
                raise MonoUnsupported(
                    "draft backbone anchors differ from the verified frontier"
                )
            # Fixed metadata must not overwrite an in-flight backbone reader.
            self._wait_draft_reader()
            metadata = getattr(self, "_draft_metadata", {})
            registration = getattr(self, "_draft_graphs", {}).get(
                (anchor_ids.numel(), anchor_ids.dtype)
            )
            if registration is not None and sampler is not None:
                epilogue = getattr(registration.capture, "epilogue", None)
                if epilogue != sampler:
                    raise MonoUnsupported(
                        "draft graph sampler changed; retire and recapture"
                    )
            publication = getattr(registration, "publication", None)
            if publication is not None:
                steps = publication.publish(frontier[2], slots)
            else:
                steps = native_draft_steps(
                    draft_model,
                    (weight.cfg for weight in runner.weights),
                    frontier[2],
                    slots,
                    out=metadata.get(len(slots)),
                )
            metadata[len(slots)] = steps
            self._draft_metadata = metadata
            resolved.append((runner, steps))

        try:
            bind_agreed(resolve, self.group)
            runner, steps = resolved[0]
            key = (anchor_ids.numel(), anchor_ids.dtype)
            selected = key in getattr(self, "_draft_graphs", {})
            all_graph = _agree_draft_graph(selected, self.group)
            if all_graph:
                result = (
                    self._replay_draft_capture(
                        key, anchor_ids, steps, sampled=sampler is not None
                    )
                    if sampler is not None
                    else self._replay_draft_capture(key, anchor_ids, steps)
                )
            else:
                result = runner.forward_block(
                    anchor_ids,
                    anchor_positions,
                    steps,
                    num_draft=runner.weights[0].cfg.draft_block_width,
                )
                if sampler is not None:
                    sampled = []
                    bind_agreed(
                        lambda: sampled.append(
                            sampler(
                                result,
                                anchor_ids,
                                runner.weights[0].cfg.draft_block_width,
                            )
                        ),
                        self.group,
                    )
                    result = sampled[0]
            completions = []

            def retire():
                completion = torch.cuda.Event()
                completion.record(torch.cuda.current_stream())
                completions.append(completion)

            bind_agreed(retire, self.group)
            self._draft_metadata_completion = completions[0]
            if all_graph:
                runner._request_completion = completions[0]
                runner._request_output_completion = completions[0]
            # Successful live backbone submission plus TP-agreed reader event.
            # Warmup uses runner.forward_block directly. This is not evidence
            # of head/acceptance correctness or device completion before drain.
            self.draft_backbone_forwards += 1
            if all_graph:
                self.draft_graph_replays += 1
            return result
        except Exception:
            # A failed copy/launch/event may have submitted partial writes.
            # Keep resident storage alive but never reuse an uncertain owner.
            self.disabled = True
            raise

    def retire_draft_captures_for_startup(self):
        """Retire old registrations before an original startup capture sweep.

        Never drop graph owners on a failed fence, or permit recapture over
        live requests. All ranks fence even if local registration sets differ.
        """

        def validate():
            if self.disabled or self.live_started or not self.owns_live_session():
                raise MonoUnsupported("draft recapture requires the startup owner")

        bind_agreed(validate, self.group)
        try:
            bind_agreed(torch.cuda.synchronize, self.group)
        except Exception:
            self.disabled = True
            raise
        getattr(self, "_draft_graphs", {}).clear()
        self._draft_context = self._draft_frontier = None
        self._draft_metadata_completion = None
        # Keep fixed metadata addresses: the new sweep may reuse them, but
        # these tensors alone do not authorize old captures or frontiers.

    def warmup_draft_block(self, drafter, running_bs, context, *, capture=False):
        """Startup eager warmup on actual bindings, restoring every write.

        By default this warms backbone AND original terminal. Optional capture
        registers an owned graph only after the startup write set restores.
        Synthetic metadata comes from the engine's capture builder, never a
        serving request. Original graph support remains a separate gate.
        """
        from atom.models.deepseek_v4.mono.draft_cache import (
            draft_write_views,
            native_draft_steps,
        )

        snapshots, resolved, captures = [], [], []
        self._draft_context = self._draft_frontier = None

        def prepare():
            runner = getattr(self, "draft_runner", None)
            if (
                self.live_started
                or self.disabled
                or not self.owns_live_session()
                or runner is None
                or runner.draft_model is not drafter.model.model
                or torch.cuda.is_current_stream_capturing()
            ):
                raise MonoUnsupported(
                    "draft warmup requires the prepared startup owner"
                )
            slots_cpu = context.attn_metadata.state_slot_out_cpu
            if slots_cpu is None or len(slots_cpu) < running_bs:
                raise MonoUnsupported(
                    "draft warmup requires engine physical slot metadata"
                )
            slots = tuple(int(slot) for slot in slots_cpu[:running_bs])
            width = runner.weights[0].cfg.draft_block_width
            configs = tuple(weight.cfg for weight in runner.weights)
            anchors = (int(drafter.model.window_size),) * running_bs
            steps = native_draft_steps(runner.draft_model, configs, anchors, slots)
            ids = drafter.block.buffer("anchor_ids", running_bs)
            positions = drafter.block.buffer("anchor_positions", running_bs)
            views = [*draft_write_views(steps, configs), ids, positions]
            output = getattr(runner, "_request_output", None)
            if output is not None:
                views.append(output)
            tokens = getattr(runner, "_request_tokens", None)
            hidden = getattr(runner, "_request_hidden", None)
            if hidden is not None:
                views.append(hidden)
            if tokens is not None:
                views.extend(tokens.values())
            fixed = getattr(runner, "_request_destinations", None)
            if fixed is not None:
                for destinations in fixed[:running_bs]:
                    views.extend(destinations)
            # Preflight and snapshot every arena in the composed request stack.
            for request in range(running_bs):
                build = runner._build_for((width, width), request)
                start = request * width
                published = runner._request_steps(steps, start, start + width)
                runner.preflight(
                    build.hidden[0],
                    published,
                    tokens=published[0].positions,
                    tokens_per_seq=width,
                    lane=request,
                )
                views.extend(build.hidden)
                # Timeline storage belongs to each layer, not its variant.
                # Captured launches may write it even though peer/epoch
                # storage is already covered by the shared arena snapshot.
                for layer in getattr(build, "layers", ()):
                    timeline = getattr(layer, "timeline", None)
                    if timeline is not None:
                        views.append(timeline)
                for variant in build.variants:
                    views.extend(
                        (
                            variant.scratch,
                            variant.step,
                            variant.hang,
                            variant.peer_buffer.storage,
                        )
                    )
            self._snapshot_capture_views(views, snapshots)
            resolved.append((runner, steps, ids, positions, width, anchors[0]))

        # Retire target capture work on every stream before reading the shared
        # pool and peer arenas. A builder's host mirrors alone are not a fence.
        bind_agreed(torch.cuda.synchronize, self.group)
        bind_agreed(prepare, self.group)

        try:
            runner, steps, ids, positions, width, anchor = resolved[0]
            # Agree before mutating even startup inputs. The forward repeats
            # this check before embedding, but that is too late to authorize
            # the warmup's earlier staging writes on different TP layouts.
            runner._agree_request_layout(running_bs * width, width, 0)
            bind_agreed(lambda: (ids.zero_(), positions.fill_(anchor)), self.group)
            out = runner.forward_block(ids, positions, steps, num_draft=width)
            bind_agreed(
                lambda: drafter.model.head_and_sample(out, ids, width), self.group
            )
            if capture:
                prepared = []

                def capture_inputs():
                    # The proposer stages its buffer before entering this
                    # session, potentially on another stream. Keep graph
                    # anchors private so that staging cannot overwrite an
                    # in-flight replay before our reader fence.
                    graph_anchors = ids.clone()
                    tokens = runner._request_tokens[ids.dtype][:running_bs]
                    hidden = runner._request_hidden[: running_bs * width]
                    residual = runner._request_output[: running_bs * width]
                    # Same original embedding values, now on fixed storage.
                    runner._embed_draft_block(
                        ids, width, ids_out=tokens, hidden_out=hidden
                    )
                    prepared.append((graph_anchors, tokens, hidden, residual))

                bind_agreed(capture_inputs, self.group)
                inputs = prepared[0]
                lanes, _ = runner._resolve_draft_requests(
                    inputs[2],
                    steps,
                    tokens=inputs[1].reshape(-1),
                    width=width,
                    lane=0,
                    out=inputs[3],
                )
                captures.append(
                    runner._capture_resolved_draft_block(
                        inputs,
                        lanes,
                        source_steps=steps,
                        epilogue=drafter.model.head_and_sample,
                    )
                )
        except Exception:
            if capture:
                # Capture may have submitted device collectives or advanced
                # peer epochs. Restoring tensor bytes is not proof that a
                # partially captured owner is safe to reuse.
                self.disabled = True
            raise
        finally:
            try:
                self._restore_capture_views(snapshots)
            except Exception:
                self.disabled = True
                raise
            finally:
                self._draft_context = self._draft_frontier = None
        if captures:
            try:
                self._register_draft_capture(runner, captures[0], steps=steps)
            except Exception:
                self.disabled = True
                raise
            return captures[0]
        return None

    def _register_draft_capture(self, runner, capture, *, steps=None):
        """Publish a restored startup graph collectively, without enabling replay."""
        registrations = []

        def validate():
            if (
                self.disabled
                or self.live_started
                or not self.owns_live_session()
                or runner is not self.draft_runner
            ):
                raise MonoUnsupported(
                    "draft capture registration requires startup owner"
                )
            runner._require_draft_capture_storage(capture, capture.resolved)
            ids = capture.inputs[0]
            key = (ids.numel(), ids.dtype)
            graphs = getattr(self, "_draft_graphs", {})
            if key in graphs:
                raise MonoUnsupported("draft capture shape is already registered")
            publication = None
            if steps is not None and getattr(capture, "source_steps", ()):
                from atom.models.deepseek_v4.mono.draft_cache import (
                    PreparedDraftPublication,
                )

                publication = PreparedDraftPublication(
                    runner.draft_model,
                    (weight.cfg for weight in runner.weights),
                    capture.source_steps,
                )
            registrations.append(
                (
                    key,
                    _DraftGraphRegistration(
                        self.owner_generation,
                        self.graph_generation,
                        runner,
                        capture,
                        publication,
                    ),
                )
            )

        bind_agreed(validate, self.group)
        key, registration = registrations[0]
        graphs = getattr(self, "_draft_graphs", {})
        graphs[key] = registration
        self._draft_graphs = graphs
        if steps is not None:
            metadata = getattr(self, "_draft_metadata", {})
            metadata[key[0]] = steps
            self._draft_metadata = metadata

    def _replay_draft_capture(self, key, anchor_ids, steps, *, sampled=False):
        """Replay after the caller consumes its verified frontier and reader fence.

        Native publication validates live addresses/frontiers. Fixed-source
        captures reuse startup lane contracts after checking retained storage;
        destination publication is recorded in the graph. Return invocation-
        owned output copies: the original head may
        retain them after the graph's fixed output storage is reused.
        """
        captures = []

        def validate():
            capture = self._require_draft_graph_registration(key)
            if bool(getattr(capture, "sampled", False)) != sampled:
                raise MonoUnsupported("draft graph output contract differs from caller")
            if (
                anchor_ids.shape != capture.inputs[0].shape
                or anchor_ids.dtype != capture.inputs[0].dtype
                or anchor_ids.device != capture.inputs[0].device
                or not anchor_ids.is_contiguous()
            ):
                raise MonoUnsupported("draft graph anchor layout changed")
            if getattr(capture, "source_steps", ()):
                self.draft_runner._require_draft_capture_publication(capture, steps)
            captures.append(capture)

        bind_agreed(validate, self.group)
        capture = captures[0]
        runner = self.draft_runner
        anchors, tokens, hidden, residual = capture.inputs
        width = tokens.shape[1]
        if not getattr(capture, "source_steps", ()):
            # Standalone captures without a fixed native publisher retain the
            # original call-local qualification route. Startup serving captures
            # above always bind their source stack and use graph publication.
            runner._agree_request_layout(hidden.shape[0], width, 0)
            lanes, _ = runner._resolve_draft_requests(
                hidden,
                steps,
                tokens=tokens.reshape(-1),
                width=width,
                lane=0,
                out=residual,
            )
            bind_agreed(
                lambda: runner._require_draft_capture_storage(capture, lanes),
                self.group,
            )
        results = []

        def replay():
            anchors.copy_(anchor_ids)
            capture.graph.replay()
            results.append(
                tuple(
                    output.clone() if output is not None else None
                    for output in capture.outputs
                )
            )

        bind_agreed(replay, self.group)
        return results[0]

    def _require_draft_graph_registration(self, key):
        """Require this session's capture lifetime; frontier checks remain separate."""
        registration = getattr(self, "_draft_graphs", {}).get(key)
        if (
            registration is None
            or self.disabled
            or not self.owns_live_session()
            or registration.owner_generation != self.owner_generation
            or registration.graph_generation != self.graph_generation
            or registration.runner is not self.draft_runner
        ):
            raise MonoUnsupported("draft graph registration is stale")
        registration.runner._require_draft_capture_storage(
            registration.capture, registration.capture.resolved
        )
        return registration.capture

    def graph_is_current(self, key: tuple[int, int]) -> bool:
        """Whether ``key`` was captured by the live owner generation."""
        return (
            key in self.graph_keys
            and self.owner_generation != 0
            and self.graph_generations.get(key) is not None
            and self.owns_live_session()
        )

    def remember_graph(self, key: tuple[int, int], selected: bool) -> None:
        """Bind or remove one graph under a new capture generation."""
        self.graph_generation += 1
        if not hasattr(self, "graph_layouts"):
            self.graph_layouts = {}
        if (
            self.batch_continuation is not None
            and self.batch_continuation.graph_key == key
        ):
            self.batch_continuation = None
        if selected:
            self.graph_keys.add(key)
            self.graph_generations[key] = self.graph_generation
            self.graph_layouts[key] = getattr(self, "pending_capture_layout", None)
        else:
            self.graph_keys.discard(key)
            self.graph_generations.pop(key, None)
            self.graph_layouts.pop(key, None)
        self.pending_capture_layout = None

    def require_graph_layout(self, key, layout) -> None:
        """Refuse stale captured operation sequences before any kernel write.

        Preserve graph identity on refusal: the engine must recognize that its
        stored graph is Mono and run native eagerly, not replay it as native.
        """
        if key in self.graph_keys and self.graph_layouts.get(key) != layout:
            raise MonoUnsupported("verify layout differs from the captured Mono graph")

    def select_graph_execution(self, key, layout) -> None:
        """Select prepared lane composition without authorizing an old FULL graph.

        Whole-forward identity remains unchanged. DSpark may compose prepared
        lane graphs; ordinary padded tails may execute exact live lanes eagerly.
        Neither route replays a mismatched FULL graph. Publication preflight
        validates storage and TP consensus agrees the route before any launch.
        """
        self.lane_graph_selected = False
        self.eager_layout_selected = False
        if key not in self.graph_keys or self.graph_layouts.get(key) == layout:
            return
        buckets = tuple(self.adapter.published_buckets)
        graphs = getattr(self.runner, "_lane_graphs", {})
        if (
            self.profile.name == ORDINARY_PROFILE.name
            and key[1] == 1
            and self.graph_is_current(key)
            and 0 < layout[0] < key[0]
            and buckets
            and all(
                bucket.tokens_per_seq == 1
                and bucket.width in self.adapter.prepared_widths
                for _, bucket in buckets
            )
            and sum(bucket.width for _, bucket in buckets) == layout[0]
        ):
            # The ladder rounds live request widths up to a captured capacity.
            # Its old operation sequence includes padded lanes. Keep the strict
            # graph identity guard, but avoid native eager for this exact-row
            # publication: forward_buckets revalidates and executes live lanes.
            self.eager_layout_selected = True
            return
        if (
            self.profile.name.startswith("dspark_")
            and self.graph_is_current(key)
            and buckets
            and all(
                (bucket.width, bucket.tokens_per_seq, bucket.ordinal) in graphs
                for _, bucket in buckets
            )
        ):
            self.lane_graph_selected = True
            return
        self.require_graph_layout(key, layout)

    def supports_ragged_graph_layout(self, lengths, max_seqlen_q: int) -> bool:
        """Agree prepared graph coverage before the scheduler rewrites tokens.

        This is a shape feasibility gate, not a live publication authorization.
        Slots, metadata addresses, epochs and execution order must still pass
        the normal publication consensus after input preparation. Keep the
        original FULL capacity key; never replay it for a different layout.
        """
        from atom.models.deepseek_v4.mono.verify_plan import plan_verify_launches

        supported = False
        try:
            if (
                not self.disabled
                and self.owns_live_session()
                and self.runner is not None
                and self.adapter is not None
                and self.profile.name.startswith("dspark_")
                and self.profile.ragged
                and 0 < len(lengths) <= self.profile.max_requests
                and all(0 < length <= max_seqlen_q for length in lengths)
            ):
                self.adapter.assert_current()
                ladder = self.builder.model_runner.capture_sizes
                capacity = next((size for size in ladder if size >= len(lengths)), None)
                key = (capacity, max_seqlen_q)
                launches = plan_verify_launches(lengths, self.adapter.prepared_shapes)
                supported = self.graph_is_current(key) and all(
                    (launch.rows, launch.tokens_per_seq, launch.lane)
                    in self.runner._lane_graphs
                    and launch.lane
                    < len(
                        self.adapter.shape_buckets(launch.rows, launch.tokens_per_seq)
                    )
                    for launch in launches
                )
        except (AttributeError, TypeError, ValueError, MonoUnsupported):
            supported = False
        # A rank-local missing graph cannot let other TP ranks rewrite the
        # request layout and subsequently enter a different launch sequence.
        return tp_agree(bool(supported), self.group)

    def prepare_weights(self) -> bool:
        """Pack and retain Mono weights before the KV pool is sized.

        The production weight layout needs a one-time, lossless repack for the
        imported kernels.  Keeping that allocation behind ``allocate_kv_cache``
        made it invisible to ``get_num_blocks``: the ordinary cache consumed
        the complete memory budget first and Mono then sat on top of it.  This
        phase has no cache dependency, so materialize it early and let the
        existing pool planner observe the actual resident bytes.

        Cache adapters, scratch arenas and output buffers remain in
        ``prepare`` because they require the final pool geometry.  Repeated
        calls are idempotent; a completed cache binding also owns a valid
        weight binding.
        """
        if self.disabled:
            return False
        if self.runner is not None:
            return True
        try:
            bind_agreed(self._acquire_lease, self.group)
            self.runner = V4MonoDecodeRunner.from_model(
                self.model.model,
                self.config.max_model_len,
                rank=self.rank,
                npes=8,
                group=self.group,
                compressor_ring_extra=self.builder.max_spec_steps,
            )

            def bind_aux():
                drafter = getattr(self.builder.model_runner, "drafter", None)
                if getattr(drafter, "_captures_aux", False):
                    from atom.mono.runtime.aux_taps import bind_aux_taps
                    from atom.spec_decode.dspark_proposer import DSparkProposer

                    spec = drafter._aux_capture_spec(self.model)
                    if (
                        spec is None
                        or spec.capture != "output"
                        or spec.extract is not DSparkProposer._extract_layer_hidden
                    ):
                        raise MonoUnsupported(
                            "V4 Mono requires the settled DSpark post-layer aux contract"
                        )
                    taps = bind_aux_taps(drafter, spec.hidden_size)
                    if taps.layer_ids != spec.layer_ids:
                        raise MonoUnsupported("DSpark aux capture declaration changed")
                    self.runner.bind_aux_outputs(taps.layer_ids, taps.buffers)

            bind_agreed(bind_aux, self.group)
            if self.profile.name.startswith("dspark_"):
                from atom.models.deepseek_v4.mono.draft_runner import V4MonoDraftRunner

                draft_models = []

                def resolve_draft():
                    drafter = self.builder.model_runner.drafter
                    if drafter._with_draft:
                        raise MonoUnsupported(
                            "joint standalone draft ownership is not yet qualified"
                        )
                    inner = drafter.model.model
                    if (
                        inner.embed is not self.model.model.embed
                        or inner.head is not self.model.model.head
                    ):
                        raise MonoUnsupported(
                            "joint draft must share the loaded target embed/head"
                        )
                    draft_models.append((inner, drafter.draft_tokens_per_seq))

                bind_agreed(resolve_draft, self.group)
                # A native recording bypasses _block_backbone entirely and
                # retains old cache pointers. It cannot survive joint binding.
                self.builder.model_runner.drafter.block.release_graphs()
                self.draft_runner = V4MonoDraftRunner.from_model(
                    draft_models[0][0],
                    self.config.max_model_len,
                    num_draft=draft_models[0][1],
                    rank=self.rank,
                    npes=8,
                    group=self.group,
                )
            logger.info(
                "V4 Mono weights prepared before KV sizing on TP rank %d",
                self.rank,
            )
            return True
        except Exception as error:
            self.refusal = str(error)
            logger.exception("V4 Mono weight prepare refused before KV sizing")
            self.close()
            self.disabled = True
            return False

    def prepare(self, *, cache_rebound=False):
        if self.disabled:
            return False
        # A complete old cache binding must retire before rebinding.  A runner
        # with no adapter is the weight-only phase installed by
        # ``prepare_weights`` and must survive the initial KV allocation.
        if self.adapter is not None:
            self.engine._retire_graphs()
            self.builder.model_runner.graph_pool = None
            self.close()
        if not self.prepare_weights():
            return False
        try:
            adapters = []
            declared_capture_sizes = self.config.capture_sizes
            max_declared_requests = max(declared_capture_sizes, default=1)
            policy_request_limit = max(self.profile.auto_request_widths)
            max_requests = (
                # The declared ladder is not yet filtered by ModelRunner.
                # Never allocate lanes for requests the scheduler cannot admit.
                max(
                    1,
                    min(
                        int(max_declared_requests),
                        policy_request_limit,
                        self.config.max_num_seqs,
                    ),
                )
                if self.config.compilation_config.cudagraph_mode.name == "FULL"
                else policy_request_limit
            )
            max_rows = max_requests * self.profile.tokens_per_seq
            bind_agreed(
                lambda: adapters.append(
                    V4CacheAdapter(
                        self.model.model,
                        self.builder,
                        [w.cfg for w in self.runner.weights],
                        max_rows=max_rows,
                        tokens_per_seq=self.profile.tokens_per_seq,
                        max_requests=max_requests,
                        # Both packed ragged and uniform confidence-scheduled
                        # q need q-specific lanes and the native live offsets.
                        ragged=(self.profile.ragged or self.profile.variable_q),
                        launch_rows=self.profile.launch_rows,
                        fused_metadata=self.profile.name.startswith("dspark_"),
                        request_widths=self.profile.capture_request_widths(
                            tuple(
                                width for width in (1, 2, 4, 8) if width <= max_requests
                            )
                        ),
                    )
                ),
                self.group,
            )
            self.adapter = adapters[0]
            if self.profile.name.startswith(
                "dspark_"
            ) and not self.draft_runner.prepare_requests(max_requests):
                raise MonoUnsupported("joint draft binary admission failed")
            from atom.models.deepseek_v4.mono.weights import _signature

            inner = self.model.model

            def bind_entry():
                self.entry_sources = tuple(
                    (
                        owner,
                        name,
                        getattr(owner, name),
                        _signature(getattr(owner, name)),
                    )
                    for owner, name in (
                        (inner.embed, "weight"),
                        (inner.norm, "weight"),
                        (inner.head, "weight"),
                        (inner, "hc_head_fn"),
                        (inner, "hc_head_scale"),
                        (inner, "hc_head_base"),
                    )
                )

            bind_agreed(bind_entry, self.group)
            for rows, q in self.adapter.prepared_shapes:
                buckets = self.adapter.shape_buckets(rows, q)
                if not self.runner.prepare(
                    rows,
                    tokens_per_seq=q,
                    lanes=len(buckets),
                ):
                    raise MonoUnsupported(f"S{rows} binary admission failed")
                for ordinal, bucket in enumerate(buckets):
                    build = self.runner._build_for((rows, q), ordinal)
                    bind_agreed(
                        lambda b=build, x=bucket, lane=ordinal, q=q: (
                            self.runner.preflight(
                                b.hidden[0],
                                x.layers,
                                tokens=x.tokens,
                                tokens_per_seq=q,
                                lane=lane,
                            )
                        ),
                        self.group,
                    )
            cfg = self.runner.weights[0].cfg
            shape = (
                (self.adapter.max_rows, cfg.hidden)
                if cfg.hc_mult == 1
                else (self.adapter.max_rows, cfg.hc_mult, cfg.hidden)
            )

            def allocate_output():
                self.output = torch.empty(shape, dtype=torch.bfloat16, device="cuda")

            bind_agreed(allocate_output, self.group)
            logger.info(
                "V4 Mono profile %s exact-row widths %s real cache prepared on TP rank %d",
                self.profile.name,
                self.adapter.prepared_widths,
                self.rank,
            )
            self.refusal = None
            if cache_rebound:
                self.live_started = False
            return True
        except Exception as error:
            self.refusal = str(error)
            logger.exception("V4 Mono exact-row prepare refused before launch")
            self.close()
            self.disabled = True
            return False

    def publish(
        self,
        input_ids,
        positions,
        context,
        *,
        capture=False,
        allow_auto=True,
    ):
        # A new target pass supersedes its predecessor even if this pass uses
        # native prefill/fallback or never reaches the context producer.
        self._draft_context = None
        self._draft_frontier = None
        self.finish_capture_publication()
        if self.runner is None:
            self.batch_continuation = None
            return False
        # Mono serves decode only.  A real prefill or an eager dummy warmup is
        # already a TP-uniform scheduler mode and cannot launch the Mono graph;
        # walking 61 weight owners and entering Gloo consensus before refusing
        # it only adds TTFT.  Capture-time dummy decode is intentionally kept:
        # it is how the selected Mono graph is constructed.
        if not capture and (
            context.context.is_prefill
            or getattr(context.context, "is_dummy_run", False)
        ):
            self.batch_continuation = None
            return False
        is_live_decode = bool(
            not capture
            and not context.context.is_prefill
            and not getattr(context.context, "is_dummy_run", False)
        )
        if is_live_decode:
            self.decode_attempts += 1
        if is_live_decode and not allow_auto:
            self.refusal = (
                "sampling/logprob request is outside the V4 Mono auto profile"
            )
            return self._refuse_publication(self.refusal, live=True)
        graph_key = (
            int(context.context.running_bs),
            int(context.attn_metadata.max_seqlen_q),
        )
        exact_tail = (
            self.profile.name == ORDINARY_PROFILE.name
            and graph_key == (3, 1)
            and 4 in self.builder.model_runner.capture_sizes
            and (capture or self.graph_is_current(graph_key))
        )
        # Native ladders may already contain C3. Serve it through the same
        # prepared exact graph as a padded C4 tail, never an uncaptured width.
        if not exact_tail and not self.profile.accepts_graph(*graph_key):
            self.refusal = (
                f"graph shape {graph_key} is not enabled for V4 Mono "
                f"{self.profile.name} profile"
            )
            return self._refuse_publication(self.refusal, live=is_live_decode)
        # The scheduler still rounds live3 up to C4. Only a current exact-live
        # recording may replace that execution key; never replay the padded
        # C4 topology for three state-writing requests.
        if (
            not capture
            and self.profile.name == ORDINARY_PROFILE.name
            and graph_key == (4, 1)
            and self.graph_is_current((3, 1))
            and int(context.context.scheduled_bs) == 3
        ):
            graph_key = (3, 1)
        self.execution_graph_key = graph_key
        batch_contract = None
        if (
            self.profile.continuation
            and is_live_decode
            and self.batch_continuation is not None
            and self.batch_continuation.owner_generation == self.owner_generation
            and self.graph_is_current(self.batch_continuation.graph_key)
            and self.graph_generations.get(self.batch_continuation.graph_key)
            == self.batch_continuation.graph_generation
            and all(
                self.slot_generations.get(row.slot, 0) == row.slot_generation
                for row in self.batch_continuation.rows
            )
        ):
            batch_contract = self.adapter.continuation_contract(context)
        if continues_batch(self.batch_continuation, graph_key, batch_contract):
            # The anchor step already agreed owner/cache/layer ABI and the
            # scheduler broadcasts the same position/slot/block-table frontier
            # to every TP worker. Ordinary output and MTP target/draft
            # sampling already obey the engine's TP-uniform device contract, so
            # the next rows need no host object collective or D2H validation.
            # Do not turn a rank-local enqueue failure into a slow-path
            # collective: peers have made the same fast decision and would
            # otherwise wait in a different phase.
            self.adapter.write(
                input_ids,
                positions,
                context,
                validate_static=False,
                validate_device_values=False,
                supported=True,
            )
            self._remember_continuation(graph_key, batch_contract)
            width = graph_key[0] * graph_key[1]
            self.fast_publications_by_width[width] = (
                self.fast_publications_by_width.get(width, 0) + 1
            )
            return self._arm_publication(capture=False)
        # Host consensus is outside the captured graph. In particular, a
        # rank-local owner failure cannot leave its peers launching alone.
        try:
            supported = (
                self.owns_live_session()
                and self.adapter.supported(context)
                and input_ids.ndim == 1
                and positions.ndim == 1
                and input_ids.numel() == positions.numel()
                and input_ids.numel() <= self.adapter.max_rows
            )
            if (
                supported
                and getattr(context.context, "is_dummy_run", False)
                and not capture
            ):
                supported = False
            self.adapter.assert_current()
            for binding in self.runner.bindings:
                binding.assert_current()
            from atom.models.deepseek_v4.mono.weights import _signature

            for owner, name, value, signature in self.entry_sources:
                current = getattr(owner, name, None)
                if current is not value:
                    raise MonoUnsupported(
                        f"{type(owner).__name__}.{name} owner changed; "
                        "retire graphs and reprepare Mono"
                    )
                actual = _signature(value)
                if actual != signature:
                    changes = ", ".join(
                        field
                        for field, before, after in zip(
                            (
                                "pointer",
                                "version",
                                "shape",
                                "stride",
                                "dtype",
                                "device",
                                "shuffled",
                            ),
                            signature,
                            actual,
                        )
                        if before != after
                    )
                    raise MonoUnsupported(
                        f"{type(owner).__name__}.{name} changed ({changes}); "
                        "retire graphs and reprepare Mono"
                    )
        # Owner and adapter checks raise backend-specific errors; convert every
        # one into the same TP-agreed native fallback decision.
        except Exception as error:  # noqa: BLE001
            self.refusal = str(error)
            supported = False
        if not tp_agree(supported, self.group):
            reason = self.refusal or (
                "tp_peer_refused" if supported else "unsupported_contract"
            )
            return self._refuse_publication(reason, live=is_live_decode)
        try:
            encoded_contracts = []

            def publish_and_preflight():
                self.adapter.write(
                    input_ids,
                    positions,
                    context,
                    # DSpark admission immediately above checked these same
                    # owners and bounds in this synchronous transaction. Keep
                    # device mirrors/layout checks; do not retain a credential
                    # across calls or change ordinary TP/MTP publication.
                    **(
                        {"validate_static": False, "supported": True}
                        if self.profile.name.startswith("dspark_")
                        else {}
                    ),
                    # Startup still generates valid metadata for warmup/lane
                    # capture. Live DSpark defers only integer address generation
                    # to the captured forward; all admission checks remain here.
                    publish_metadata=(
                        capture or not getattr(self.adapter, "fused_metadata", False)
                    ),
                )
                if not capture:
                    self.select_graph_execution(graph_key, self.adapter.graph_layout())
                if self.profile.name.startswith("dspark_"):
                    # The admission phase immediately above already checked
                    # every runner binding in this same synchronous transaction.
                    # Metadata publication does not modify weights. Retain all
                    # layer/cache/graph/alias resolution, without scanning those
                    # identical weight owners a second time on every step.
                    self.runner._preflight_publication_layers(
                        self.adapter.published_buckets, x_out=self.output
                    )
                else:
                    self.runner.preflight_publication(
                        self.adapter.published_buckets, x_out=self.output
                    )
                # Encoding validates the packed layout as well as its bounds.
                # Any local error must be agreed before peers enter all_gather.
                encoded_contracts.append(
                    torch.cat(
                        (
                            self.adapter.publication_consensus_tensor(),
                            self.runner.lane_execution_consensus_tensor(
                                self.adapter.published_buckets,
                                capacity=self.adapter.max_rows,
                            ),
                            torch.tensor(
                                [
                                    int(self.lane_graph_selected),
                                    int(self.eager_layout_selected),
                                    *graph_key,
                                ],
                                dtype=torch.int64,
                            ),
                        )
                    )
                )

            bind_agreed(publish_and_preflight, self.group)
            import torch.distributed as dist

            local_contract = encoded_contracts[0]
            contracts = [
                torch.empty_like(local_contract)
                for _ in range(dist.get_world_size(self.group))
            ]
            dist.all_gather(contracts, local_contract, group=self.group)
            if any(
                not torch.equal(contract, contracts[0]) for contract in contracts[1:]
            ):
                raise MonoUnsupported(
                    "TP ranks published different token/position/slot/block tables "
                    "or verify layouts/arena lanes/eager-graph execution"
                )
        # Publication may fail in torch.distributed, cache adapters, or the
        # compiled preflight; all must refuse before any rank launches Mono.
        except Exception as error:  # noqa: BLE001
            self.refusal = str(error)
            self.lane_graph_selected = False
            self.eager_layout_selected = False
            self.adapter.clear_publication()
            logger.warning(
                "V4 Mono metadata publication refused before launch: %s", error
            )
            return self._refuse_publication(
                self.refusal or "metadata_publication_refused", live=is_live_decode
            )
        if is_live_decode:
            self.full_publications += 1
        published_rows = len(self.adapter.published_contract)
        continuation_contract = (
            self.adapter.continuation_contract(context)
            if self.profile.continuation and is_live_decode
            else None
        )
        if (
            self.profile.continuation
            and is_live_decode
            and (published_rows in self.adapter.prepared_widths or graph_key == (3, 1))
            and continuation_contract is not None
            and self.graph_is_current(graph_key)
        ):
            self._remember_continuation(graph_key, continuation_contract)
        else:
            self.batch_continuation = None
        return self._arm_publication(capture=capture)

    def _refuse_publication(self, reason, *, live):
        """Forget continuation on refusal; count only attempted live decode."""
        self.batch_continuation = None
        if live:
            self.fallback_steps += 1
            self.fallback_reasons[reason] = self.fallback_reasons.get(reason, 0) + 1
        return False

    def _remember_continuation(self, graph_key, contract):
        self.batch_continuation = BatchContinuation(
            self.owner_generation,
            graph_key,
            self.graph_generations[graph_key],
            tuple(
                ContinuationRow(
                    slot, self.slot_generations.get(slot, 0), position, blocks
                )
                for position, slot, blocks in contract
            ),
        )

    def _arm_publication(self, *, capture: bool) -> bool:
        slots = tuple(
            sorted({contract[2] for contract in self.adapter.published_contract})
        )
        self.publication_generation = self.owner_generation
        self.publication_slot_generations = tuple(
            (slot, self.slot_generations.get(slot, 0)) for slot in slots
        )
        self.armed = True
        self.capture_publication = capture
        if capture:
            self.pending_capture_layout = self.adapter.graph_layout()
        self.refusal = None
        if not capture:
            self.selected_steps += (
                1  # successful publications, not a device-launch counter
            )
            policy = self.runner.index_policy()
            buckets = tuple(self.adapter.published_buckets)
            self.microbucket_stacks += len(buckets)
            self.index_refresh_executions += len(buckets) * policy["refresh_csa_layers"]
            self.index_reuse_executions += len(buckets) * policy["reuse_csa_layers"]
            self.index_suppressed_rows += (
                sum(bucket.width for _, bucket in buckets) * policy["reuse_csa_layers"]
            )
        return True

    def serving_status(self) -> dict:
        """Host counters for a completed serving run; no device reads or syncs."""
        selected = self.selected_steps
        return {
            "policy": "auto",
            "profile": self.profile.name,
            "decode_attempts": self.decode_attempts,
            "selected_steps": selected,
            "graph_replays": self.graph_replays,
            "eager_forwards": self.eager_forwards,
            "draft_backbone_forwards": self.draft_backbone_forwards,
            "draft_graph_replays": getattr(self, "draft_graph_replays", 0),
            "lane_graph_forwards": getattr(self, "lane_graph_forwards", 0),
            "fallback_steps": self.fallback_steps,
            "fallback_reasons": dict(sorted(self.fallback_reasons.items())),
            "mono_hit_rate": (
                selected / self.decode_attempts if self.decode_attempts else 0.0
            ),
            "graph_replay_rate": self.graph_replays / selected if selected else 0.0,
            "last_refusal": self.refusal,
            "publication": {
                "full_consensus": self.full_publications,
                "batch_fast": sum(self.fast_publications_by_width.values()),
                "c1_fast": self.fast_publications_by_width.get(1, 0),
                "fast_by_width": {
                    str(width): count
                    for width, count in sorted(self.fast_publications_by_width.items())
                },
            },
            "owner_generation": self.owner_generation,
            "graph_generations": sorted(
                [key[0], key[1], generation]
                for key, generation in self.graph_generations.items()
            ),
            "index_reuse": self.index_reuse_status(),
        }

    def index_reuse_status(self) -> dict:
        """Static policy plus serving-generation refresh/reuse hit counters."""
        policy = (
            self.runner.index_policy()
            if self.runner is not None
            else {
                "refresh_csa_layers": 0,
                "reuse_csa_layers": 0,
                "refresh_layer_ids": [],
                "reuse_layer_ids": [],
                "suppressed_stages_per_reuse_hit": [],
            }
        )
        return {
            **policy,
            "microbucket_stacks": self.microbucket_stacks,
            "refresh_layer_executions": self.index_refresh_executions,
            "reuse_layer_executions": self.index_reuse_executions,
            "index_task_suppression_hits": self.index_reuse_executions,
            "index_suppressed_live_rows": self.index_suppressed_rows,
        }

    def finish_capture_publication(self) -> None:
        """Clear current target authorization, without retiring graphs or readers."""
        self.armed = False
        self.execution_graph_key = None
        self.lane_graph_selected = False
        self.eager_layout_selected = False
        self.capture_publication = False
        self.publication_generation = 0
        self.publication_slot_generations = ()
        if self.adapter is not None:
            self.adapter.clear_publication()

    def prepare_engine_capture(self, input_ids, positions, context) -> bool:
        """Publish engine dummy state and prepare DSpark lane graphs at startup.

        Lane capture consumes and retires its publication while restoring state.
        Republish before the engine's warmup/outer FULL capture; never retain
        stale authorization or create graphs on a live decode step. Ordinary
        and MTP retain their existing whole-forward capture path.
        """
        selected = self.publish(input_ids, positions, context, capture=True)
        if not selected or not self.profile.name.startswith("dspark_"):
            return selected
        self.capture_published_lanes()
        return self.publish(input_ids, positions, context, capture=True)

    @contextmanager
    def preserve_tail_capture(self):
        """Restore the added graph's write set even if startup capture fails."""
        snapshots = []

        def prepare():
            self._snapshot_tail_capture(snapshots)

        bind_agreed(prepare, self.group)
        try:
            yield
        finally:
            try:
                self._restore_capture_views(snapshots)
            except Exception:
                # Contaminated state cannot safely fall back to native serving.
                self.disabled = True
                raise
            finally:
                self.finish_capture_publication()

    def _snapshot_tail_capture(self, snapshots):
        if (
            self.live_started
            or not self.capture_publication
            or not self.armed
            or not self.owns_live_session()
            or self.execution_graph_key != (3, 1)
            or self.profile.name != ORDINARY_PROFILE.name
            or torch.cuda.is_current_stream_capturing()
        ):
            raise MonoUnsupported("tail snapshot requires its startup publication")
        views = list(self.adapter.capture_write_views())
        views.append(self.output)
        for _, bucket in self.adapter.published_buckets:
            build = self.runner._build_for(
                (bucket.width, bucket.tokens_per_seq), bucket.ordinal
            )
            views.extend(build.hidden)
            for variant in build.variants:
                views.extend(
                    (
                        variant.scratch,
                        variant.step,
                        variant.hang,
                        variant.peer_buffer.storage,
                    )
                )
        self._snapshot_capture_views(views, snapshots)

    def capture_published_lanes(self) -> None:
        """Startup-only backbone capture, restoring the published write set.

        Caller supplies a TP-agreed capture publication on valid dummy state.
        No live request, engine graph, embedding/terminal or draft capture is
        performed here. Graph routing remains a separate engine contract.
        """
        snapshots = []
        buckets = ()
        original_graphs = set()
        candidates = ()

        def prepare():
            nonlocal buckets, original_graphs, candidates
            if (
                self.live_started
                or self.disabled
                or not self.capture_publication
                or not self.armed
                or not self.owns_live_session()
                or self.publication_generation != self.owner_generation
                or any(
                    self.slot_generations.get(slot, 0) != generation
                    for slot, generation in self.publication_slot_generations
                )
                or torch.cuda.is_current_stream_capturing()
            ):
                raise MonoUnsupported(
                    "lane preparation requires a startup capture publication"
                )
            buckets = tuple(self.adapter.published_buckets)
            if not buckets:
                raise MonoUnsupported("lane preparation requires published buckets")
            self.runner.preflight_publication(buckets, x_out=self.output)
            candidates = self.adapter.capture_lane_candidates()
            views = list(self.adapter.capture_write_views())
            original_graphs.update(self.runner._lane_graphs)
            for _, bucket in candidates:
                build = self.runner._build_for(
                    (bucket.width, bucket.tokens_per_seq), bucket.ordinal
                )
                views.extend(build.hidden)
                for variant in build.variants:
                    views.extend(
                        (
                            variant.scratch,
                            variant.step,
                            variant.hang,
                            variant.peer_buffer.storage,
                        )
                    )
            views.extend(self.runner._aux_outputs.values())
            self._snapshot_capture_views(views, snapshots)

        try:
            bind_agreed(prepare, self.group)
        except Exception:
            self.finish_capture_publication()
            raise

        try:
            for source, bucket in candidates:
                key = (bucket.width, bucket.tokens_per_seq, bucket.ordinal)
                present = key in original_graphs
                all_present = tp_agree(present, self.group)
                all_absent = tp_agree(not present, self.group)
                if not (all_present or all_absent):
                    raise MonoUnsupported("TP ranks disagree on prepared lane graphs")
                if all_present:
                    continue
                build = self.runner._build_for(key[:2], key[2])
                embedded = []

                def stage(source=source, bucket=bucket, build=build, embedded=embedded):
                    if source is not bucket:
                        self.adapter.stage_capture_lane(source, bucket)
                    from atom.models.deepseek_v4.mono.embedding import mhc_input

                    embedded.append(mhc_input(self.runner.model, bucket.tokens))
                    build.hidden[0].copy_(embedded[0])
                    self.runner.preflight(
                        build.hidden[0],
                        bucket.layers,
                        tokens=bucket.tokens,
                        tokens_per_seq=bucket.tokens_per_seq,
                        lane=bucket.ordinal,
                    )

                bind_agreed(stage, self.group)
                bind_agreed(
                    lambda build=build, bucket=bucket: self.runner.forward(
                        build.hidden[0],
                        bucket.layers,
                        tokens=bucket.tokens,
                        tokens_per_seq=bucket.tokens_per_seq,
                        lane=bucket.ordinal,
                    ),
                    self.group,
                )
                self._restore_capture_views(snapshots)
                bind_agreed(
                    lambda build=build, embedded=embedded: build.hidden[0].copy_(
                        embedded[0]
                    ),
                    self.group,
                )
                self.runner.capture_lane(
                    bucket.layers,
                    tokens=bucket.tokens,
                    rows=bucket.width,
                    tokens_per_seq=bucket.tokens_per_seq,
                    lane=bucket.ordinal,
                )
                self._restore_capture_views(snapshots)
        except Exception:
            for key in set(self.runner._lane_graphs) - original_graphs:
                self.runner._lane_graphs.pop(key)
            raise
        finally:
            try:
                self._restore_capture_views(snapshots)
            except Exception:
                # A failed restore is not a safe native fallback. Prevent
                # subsequent serving on potentially contaminated request state.
                self.disabled = True
                self.runner._lane_graphs.clear()
                raise
            finally:
                self.finish_capture_publication()

    @staticmethod
    def _snapshot_capture_views(views, snapshots):
        """Save exact views once; callers retain their distinct target/draft write sets."""
        seen = set()
        for view in views:
            key = input_identity(view)
            if key not in seen:
                seen.add(key)
                snapshots.append((view, view.clone()))

    def _restore_capture_views(self, snapshots):
        """Retire every rank's readers before restoring bytes and fence the copies."""
        bind_agreed(torch.cuda.synchronize, self.group)
        bind_agreed(
            lambda: [view.copy_(saved) for view, saved in snapshots], self.group
        )
        bind_agreed(torch.cuda.synchronize, self.group)

    def forward(self, input_ids, positions):
        if not self.armed:
            raise MonoUnsupported("Mono input metadata has not been published")
        try:
            if (
                not self.owns_live_session()
                or self.publication_generation != self.owner_generation
                or any(
                    self.slot_generations.get(slot, 0) != generation
                    for slot, generation in self.publication_slot_generations
                )
            ):
                raise MonoUnsupported(
                    "published Mono metadata belongs to a stale owner/slot generation"
                )
            rows = self.adapter.published_rows
            if (
                not rows
                or input_ids.ndim != 1
                or positions.ndim != 1
                or input_ids.numel() != positions.numel()
                or not rows <= input_ids.numel() <= self.adapter.max_rows
                or self.output is None
            ):
                raise MonoUnsupported(
                    "published exact-row metadata does not match model input"
                )
            out = self.output[: input_ids.numel()]
            # Resolve the whole live step against its actual input/output views
            # before the first cache write. Never reuse publication preflight's
            # geometry-only contracts across the intervening consensus gate.
            buckets = self.adapter.published_buckets
            if getattr(self, "lane_graph_selected", False) and not all(
                (bucket.width, bucket.tokens_per_seq, bucket.ordinal)
                in self.runner._lane_graphs
                for _, bucket in buckets
            ):
                raise MonoUnsupported("selected lane graph retired before execution")
            # Read stable published token buffers, not the engine's rotating
            # H2D input pointer: a captured graph must consume the next step.
            tokens = (
                buckets[0][1].tokens
                if len(buckets) == 1
                else torch.cat([bucket.tokens for _, bucket in buckets])
            )
            from atom.models.deepseek_v4.mono.embedding import mhc_input

            h = mhc_input(self.runner.model, tokens)
            if getattr(self.adapter, "fused_metadata", False):
                self.adapter.publish_device_metadata()
            self.runner.forward_buckets(h, buckets, x_out=out[:rows])
            if input_ids.numel() > rows:
                out[rows:].zero_()
            return out
        finally:
            self.armed = False  # authorization is one forward, never a stale model hook
            self.lane_graph_selected = False
            self.eager_layout_selected = False
            if not self.capture_publication:
                self.publication_generation = 0
                self.publication_slot_generations = ()
                if self.adapter is not None:
                    self.adapter.clear_publication()
