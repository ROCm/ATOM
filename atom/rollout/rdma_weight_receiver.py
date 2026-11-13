# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Receive trained weights straight into resident GPU memory over RCCL.

Every worker joins a process group alongside the trainer, so the trainer
broadcasts BF16 buckets directly into this engine's parameters. No safetensors
round-trip, no shared folder, no host bounce.

The wire format is frozen by the sending side
(``lumenrl.engine.inference.rdma_weight_transfer``) and must match byte for byte:

    header   4x int64, broadcast from rank 0:
             [command, metadata_bytes, payload_bytes, version]
             command 0 = end of stream, 1 = a bucket follows
    metadata ``metadata_bytes`` of uint8, a JSON list of
             {"name", "shape", "dtype", "offset", "nbytes"}
             where dtype is a torch name with the "torch." prefix stripped
    payload  ``payload_bytes`` of uint8; each entry is the slice
             [offset, offset+nbytes) viewed as its dtype and shape

A stream is one weight sync, driven by the engine's ``receive_weights_rdma``
command, which fences serving around it as it does any other sync.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import time
import traceback
from collections.abc import Callable
from dataclasses import dataclass
from datetime import timedelta

import torch
import torch.distributed as dist

from atom.distributed.group_agreement import group_min_each
from atom.model_engine.collective_rpc import WeightReloadRefused
from atom.rollout.weight_bucket import decode_bucket
from atom.rollout.weight_updater import _one_update

logger = logging.getLogger("atom")

_CMD_END = 0
_CMD_BUCKET = 1
_HEADER_WORDS = 4  # command, metadata bytes, payload bytes, version

# What a header is held to before anything is allocated for it. Metadata is a
# JSON entry per tensor, so 64 MiB is hundreds of thousands of them, and no
# bucket carries a terabyte: past either, the header is corrupt or another
# protocol's.
_MAX_METADATA_BYTES = 64 << 20
_MAX_PAYLOAD_BYTES = 1 << 40

# A rank's ballot at each vote on a stream; a vote's outcome is the least
# ballot any rank cast. See receive_weight_stream.
_LEFT = -1  # broke off between votes and left the group: no rank may keep it
_BROKEN_OFF = 0  # cannot receive the frame just announced
_ENDED = 1  # at the end marker, still in step with the group
_RECEIVING = 2  # can receive the frame just announced

# A worker's ballot before it joins a weight group, the least deciding again;
# see init_rdma_weight_group.
_CANNOT_JOIN = 0
_JOINED_ALREADY = 1
_WILL_JOIN = 2


class RDMAStreamOutOfStep(RuntimeError):
    """This rank can no longer follow the stream.

    A broadcast is received whole, into a buffer of the size it was sent with,
    so a frame one receiver cannot allocate for is one none of them can take
    part in -- and the sender, already broadcasting it, cannot be told. A
    broadcast that failed, typically because a peer left, puts the group in the
    same place, and so does a vote that failed. Nothing more can ride the
    group: this rank's end is torn down, and the sender and any peer still in a
    broadcast are released when their side of it fails, or by its timeout.
    """


@contextlib.contextmanager
def _aborted_not_ended_on_timeout():
    """Build a group under TORCH_NCCL_ASYNC_ERROR_HANDLING=2, which a
    ProcessGroupNCCL reads once, when built: a collective of it that times
    out -- a trainer gone mid-stream -- aborts its communicator, and its wait
    raises. Under torch's default the watchdog ended the runner, and the
    engine with it. The engine's other groups keep the process's setting."""
    name = "TORCH_NCCL_ASYNC_ERROR_HANDLING"
    saved = os.environ.get(name)
    os.environ[name] = "2"
    try:
        yield
    finally:
        if saved is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = saved


def _receive(tensor: torch.Tensor, group, timeout: timedelta) -> None:
    """Take *tensor* from the trainer's broadcast, raising if it has not
    arrived within *timeout*.

    Waited on here: a broadcast on a communicator aborted for a timeout
    returns with *tensor* never written -- read as an END marker of version
    0, or as weights.
    """
    dist.broadcast(tensor, src=0, group=group, async_op=True).wait(timeout=timeout)


@dataclass
class _WeightGroup:
    group: object
    join: dict
    # Each broadcast's bound: the group's collective timeout.
    timeout: timedelta
    # Versions rise within a group; a new group is a new sender, e.g. a
    # restarted trainer counting from 1 again.
    last_version: int | None = None


def _frames_no_bucket(
    command: int, metadata_bytes: int, payload_bytes: int
) -> str | None:
    """Why a header that is not the end marker frames no bucket, if it does not.

    The sender sends buckets, each followed by its metadata and payload
    broadcasts, and the end marker, which nothing follows. Any other header
    says nothing of what the sender does next: received as if it framed a
    bucket, its broadcasts are ones the sender may never post, and every later
    frame would be read out of step.
    """
    if command != _CMD_BUCKET:
        return f"command {command} is neither a bucket nor the end marker"
    if metadata_bytes == 0 or payload_bytes == 0:
        return "a bucket without metadata or without a payload"
    return None


def _unreceivable(
    device: torch.device, metadata_bytes: int, payload_bytes: int
) -> str | None:
    """Why a frame of these sizes cannot be allocated for here, if it cannot."""
    if metadata_bytes < 0 or payload_bytes < 0:
        return "a negative size"
    if metadata_bytes > _MAX_METADATA_BYTES:
        return f"metadata over the {_MAX_METADATA_BYTES}-byte ceiling"
    if payload_bytes > _MAX_PAYLOAD_BYTES:
        return f"a payload over the {_MAX_PAYLOAD_BYTES}-byte ceiling"
    if device.type == "cuda":
        free, _ = torch.cuda.mem_get_info(device)
        # Blocks the caching allocator holds unused are allocatable too.
        cached = torch.cuda.memory_reserved(device) - torch.cuda.memory_allocated(
            device
        )
        needed = metadata_bytes + payload_bytes
        if needed > free + cached:
            return f"{needed} bytes needed, {free + cached} free on {device}"
    return None


def _recv_header(
    group, *, device: torch.device, timeout: timedelta
) -> tuple[int, int, int, int]:
    header = torch.empty(_HEADER_WORDS, dtype=torch.int64, device=device)
    _receive(header, group, timeout)
    command, metadata_bytes, payload_bytes, version = (
        int(v) for v in header.cpu().tolist()
    )
    return command, metadata_bytes, payload_bytes, version


def _decode_frame(
    metadata_tensor: torch.Tensor, payload: torch.Tensor
) -> list[tuple[str, torch.Tensor]]:
    # Through a byte buffer: tolist() builds a list slot per byte, which at the
    # metadata ceiling is half a gigabyte of host memory before JSON sees it.
    metadata = json.loads(metadata_tensor.cpu().numpy().tobytes().decode("utf-8"))
    if not isinstance(metadata, list) or not metadata:
        raise ValueError("invalid RDMA weight metadata: expected a non-empty list")
    return decode_bucket(metadata, payload)


class _Votes:
    """How the ranks taking a stream decide on it, and what this rank owes
    them if it breaks off between two votes."""

    def __init__(self, agree: Callable[[int], int]):
        self._agree = agree
        # Set by the vote that is every rank's last, and by no other.
        self.decided = False
        self.failed = False

    def _cast(self, ballot: int) -> int:
        try:
            return self._agree(ballot)
        except Exception:
            self.failed = True
            raise

    def on_frame(self, receivable: bool) -> bool:
        """Whether every rank receives the frame just announced; if not, that
        was every rank's last vote."""
        if self._cast(_RECEIVING if receivable else _BROKEN_OFF) == _RECEIVING:
            return True
        self.decided = True
        return False

    def at_end(self) -> bool:
        """Every rank's last vote, at the end marker: whether a rank left."""
        left = self._cast(_ENDED) == _LEFT
        self.decided = True
        return left

    def break_off(self, leave: Callable[[], object], version: int) -> None:
        """Broken off between votes, in a broadcast a peer may still be in:
        *leave* this rank's end of the group, which releases that peer, then
        cast the ballot it waits for."""
        if self.decided:
            return
        try:
            leave()
        except Exception:  # the stream's own failure is raised
            logger.exception(
                "RDMA weight stream v%d: leaving its group failed", version
            )
        if not self.failed:
            try:
                self._agree(_LEFT)
            except Exception:  # the stream's own failure is raised
                logger.exception(
                    "RDMA weight stream v%d: its last vote failed", version
                )


@dataclass
class _Received:
    """What this rank has made of a stream so far."""

    # Raised at the end marker; the rest of the stream is still received.
    failure: Exception | None
    opened: bool = False
    applied: bool = False
    ended: bool = False
    updated: int = 0
    buckets: int = 0
    bytes: int = 0

    def stats(self, seconds: float) -> dict[str, float]:
        return {
            "updated": self.updated,
            "buckets": self.buckets,
            "bytes": self.bytes,
            "seconds": seconds,
            "gbps": (self.bytes * 8 / 1e9 / seconds) if seconds > 0 else 0.0,
        }


def _open_sync_for_stream(runner, got: _Received, version: int) -> None:
    if got.failure is None:
        try:
            if runner._sync is not None:
                raise RuntimeError("another weight sync is open on this rank")
            runner._open_sync()
            got.opened = True
        except Exception as exc:  # noqa: BLE001 - raised at the end marker
            got.failure = exc
    if got.failure is not None:
        logger.error(
            "RDMA weight stream v%d refused on this rank, receiving it before "
            "reporting that: %s",
            version,
            got.failure,
        )


def _take_frames(
    group,
    runner,
    got: _Received,
    votes: _Votes,
    *,
    device: torch.device,
    version: int,
    timeout: timedelta,
) -> None:
    """Receive frames up to the end marker, writing each bucket unless this
    rank has failed already; see ``receive_weight_stream``."""
    while True:
        # The last bucket goes before anything else is allocated, or two are
        # resident at the peak.
        metadata_tensor = payload = weights = None
        command, metadata_bytes, payload_bytes, frame_version = _recv_header(
            group, device=device, timeout=timeout
        )
        # Checked on every header: a version change mid-stream means two
        # senders are interleaving on one group.
        if got.failure is None and frame_version != version:
            got.failure = RuntimeError(
                f"RDMA weight version mismatch: expected {version}, "
                f"got {frame_version}"
            )
        if command == _CMD_END:
            # The contract is [END, 0, 0, version].
            if got.failure is None and (metadata_bytes != 0 or payload_bytes != 0):
                got.failure = RuntimeError(
                    f"invalid RDMA end marker: metadata_bytes={metadata_bytes} "
                    f"payload_bytes={payload_bytes}, both must be 0"
                )
            got.ended = True
            return
        frame = (
            f"command={command} metadata_bytes={metadata_bytes} "
            f"payload_bytes={payload_bytes}"
        )
        # Free memory differs between ranks; a header that frames no bucket is
        # the same on every rank, so all leave at it together.
        reason = _frames_no_bucket(
            command, metadata_bytes, payload_bytes
        ) or _unreceivable(device, metadata_bytes, payload_bytes)
        if reason is None:
            try:
                metadata_tensor = torch.empty(
                    metadata_bytes, dtype=torch.uint8, device=device
                )
                payload = torch.empty(payload_bytes, dtype=torch.uint8, device=device)
            except RuntimeError as exc:  # out of memory, within the bounds
                metadata_tensor = payload = None
                reason = str(exc)
        if not votes.on_frame(reason is None):
            raise RDMAStreamOutOfStep(
                f"RDMA weight frame cannot be received: {frame} "
                f"({reason or 'not by every rank taking the stream'})"
            )

        _receive(metadata_tensor, group, timeout)
        _receive(payload, group, timeout)
        if got.failure is not None:
            continue
        try:
            weights = _decode_frame(metadata_tensor, payload)
            got.applied = True
            counts = runner._apply_named_tensors(weights)
            runner._refuse_unwritten(counts)
        except Exception as exc:  # noqa: BLE001 - raised at the end marker
            # Kept until the end marker; its frames must not keep this bucket
            # alive while the rest of the stream is drained.
            traceback.clear_frames(exc.__traceback__)
            got.failure = exc
            logger.error(
                "RDMA weight bucket %d failed on this rank, receiving the rest "
                "of the stream before reporting it: %s",
                got.buckets + 1,
                exc,
            )
            continue
        got.updated += counts["updated"]
        got.bytes += payload_bytes
        got.buckets += 1


@torch.no_grad()
def receive_weight_stream(
    group,
    runner,
    *,
    device: torch.device,
    version: int,
    require_full: bool,
    timeout: timedelta,
    agree: Callable[[int], int],
    leave: Callable[[], object],
    refusal: Exception | None = None,
) -> dict[str, float]:
    """Consume one weight stream and write it as one sync on *runner*.

    *agree* is how the ranks taking the stream decide on it. They vote on
    every frame's header before any of them enters the frame's broadcasts, so
    a frame one rank cannot receive is broken off by all of them there, rather
    than leaving the others in a broadcast it never joined. And they vote at
    the end marker, so a rank does not keep a group a peer has just left. The
    first vote with any ballot below ``_RECEIVING`` is every rank's last, so
    each casts as many as its peers whatever happened to it.

    A failure on this rank does not end its part in the stream: the trainer
    and every other rank are still in the broadcasts, and leaving early hangs
    all of them in the next one. So the rest is received and discarded, the
    header's sizes being all that is needed to stay in step, and the failure
    is raised at the end marker. What breaks the stream off before then -- a
    header that frames no bucket, a frame some rank cannot receive, a
    broadcast that fails or does not arrive within *timeout*, as when the
    trainer is gone -- raises ``RDMAStreamOutOfStep``. *leave* tears
    down this rank's end of *group* before its last vote: a peer still in the
    failed broadcast is released only once that end is gone.

    *refusal*, if given, refuses the stream: it is received in full and
    discarded, nothing is written, and the refusal is raised at the end.

    A failure raised at the end marker before this rank wrote any of the
    stream is a ``WeightReloadRefused``, so the engine can tell that nothing
    was written here.
    """
    started = time.perf_counter()
    got = _Received(failure=refusal)
    votes = _Votes(agree)
    try:
        _open_sync_for_stream(runner, got, version)
        _take_frames(
            group,
            runner,
            got,
            votes,
            device=device,
            version=version,
            timeout=timeout,
        )
        if got.failure is None:
            try:
                runner._complete_sync(require_full)
            except Exception as exc:  # noqa: BLE001 - raised after the vote
                got.failure = exc
        if votes.at_end():
            # Counts as written, so the engine stays fenced, as it should: the
            # stream broke off, and what the other ranks wrote is unknown even
            # where this one wrote nothing.
            raise RDMAStreamOutOfStep(
                f"RDMA weight stream v{version}: a rank taking it left its group"
            ) from got.failure
        if got.failure is not None:
            if got.applied or isinstance(got.failure, WeightReloadRefused):
                raise got.failure
            raise WeightReloadRefused(str(got.failure)) from got.failure
    except Exception as exc:
        votes.break_off(leave, version)
        if got.opened:
            runner._discard_failed_weight_sync()
        if not isinstance(exc, RDMAStreamOutOfStep) and (not got.ended or votes.failed):
            # Counts as written and keeps the engine fenced, as above.
            raise RDMAStreamOutOfStep(
                f"RDMA weight stream v{version} broke off: {exc}"
            ) from exc
        raise
    return got.stats(time.perf_counter() - started)


class RDMAWeightReceiverMixin:
    """The worker side of the trainer's RDMA weight group.

    ``init_rdma_weight_group`` and ``destroy_rdma_weight_group`` are reached
    over ``collective_rpc``; ``receive_weights_rdma`` writes weights, so only
    the engine's command of that name drives it. The host calls
    ``_init_rdma_receiver(config)`` from its ``__init__``.
    """

    def _init_rdma_receiver(self, config) -> None:
        self._rdma_groups: dict[str, _WeightGroup] = {}
        # Read before the runner isolates DP, after which every DP engine's
        # workers see data_parallel_size=1; one group would span them all.
        self._deployment_data_parallel_size = config.parallel_config.data_parallel_size

    def _rdma_receive_refusal(self) -> str | None:
        """Why this engine cannot take weights over RDMA, if it cannot."""
        dp = int(self._deployment_data_parallel_size or 1)
        if dp > 1:
            return (
                f"data_parallel_size={dp}: one group would span {dp} "
                f"EngineCores, each deciding the stream for its own workers "
                f"alone, so DP replicas could end on different versions; run "
                f"each DP replica as its own server"
            )
        pp = int(self.config.pipeline_parallel_size or 1)
        if pp > 1:
            return (
                f"pipeline_parallel_size={pp}: each stage's workers would "
                f"decide the stream as if they held the whole model"
            )
        return self._reload_refusal()

    def _vote_with_engine_peers(self, *ballots: int) -> tuple[int, ...]:
        """A decision across this engine's workers, every one of which takes
        the stream.

        Over aiter's world group, not its TP group: a runner's world is its
        engine's workers alone, DP being isolated per engine, and with DP and
        PP refused those workers are every receiver in the weight group.
        """
        # The engine's workers as ModelRunner places devices: one per TP
        # shard that has a process, per prefill-context rank.
        workers = int(self.config.tp_world_size) * int(
            self.config.prefill_context_parallel_size
        )
        if workers <= 1:
            return ballots
        from aiter.dist.parallel_state import get_world_group

        world = get_world_group()
        if world.world_size != workers:
            raise RuntimeError(
                f"cannot decide on the RDMA stream: the process world has "
                f"{world.world_size} rank(s), the engine {workers} worker(s)"
            )
        return group_min_each(ballots, world.cpu_group)

    def init_rdma_weight_group(
        self,
        master_addr: str,
        master_port: int,
        base_rank: int,
        world_size: int,
        group_name: str,
        timeout_s: int = 600,
    ) -> bool:
        """Join the trainer's weight-broadcast group as this rank.

        Rank 0 is the trainer; the caller assigns one ``base_rank`` per
        replica, whose workers follow it in order. Decided by every worker of
        the engine before any joins: one that cannot -- refused, its rank out
        of range, its earlier join a different one -- or that has joined where
        the others have not would leave them in the rendezvous waiting for it.
        Idempotent for the same join, as the orchestrator may retry.
        """
        from atom.utils.distributed.utils import (
            stateless_init_torch_distributed_process_group,
        )

        rank = int(base_rank) + int(self.rank)
        join = {
            "master": f"{master_addr}:{int(master_port)}",
            "world_size": int(world_size),
            "rank": rank,
        }
        joined = self._rdma_groups.get(group_name)
        problem: Exception | None = None
        refusal = self._rdma_receive_refusal()
        if refusal:
            problem = RuntimeError(f"cannot receive weights over RDMA: {refusal}")
        elif not 0 < rank < int(world_size):
            problem = ValueError(
                f"invalid RDMA rank {rank} for world_size={world_size} "
                f"(base_rank={base_rank}, worker={self.rank}); rank 0 is the "
                f"trainer"
            )
        elif joined is not None and joined.join != join:
            problem = RuntimeError(
                f"RDMA weight group {group_name!r} is already joined as "
                f"{joined.join}, not {join}; destroy it on every rank first"
            )

        if problem is not None:
            ballot = _CANNOT_JOIN
        else:
            ballot = _JOINED_ALREADY if joined is not None else _WILL_JOIN
        try:
            outcome, none_will_join = self._vote_with_engine_peers(
                ballot, int(ballot != _WILL_JOIN)
            )
        except Exception as exc:
            if problem is None:
                raise
            raise problem from exc
        if outcome == _CANNOT_JOIN:
            raise problem or RuntimeError(
                f"RDMA weight group {group_name!r} not joined: another worker "
                f"of this engine cannot join it"
            )
        if outcome == _JOINED_ALREADY and not none_will_join:
            raise RuntimeError(
                f"RDMA weight group {group_name!r} is joined on only some "
                f"workers of this engine; destroy it on every rank first"
            )
        if joined is not None:
            logger.warning(
                "%s: RDMA weight group %s already joined, reusing",
                self.label,
                group_name,
            )
            return True

        timeout = timedelta(seconds=int(timeout_s))
        with _aborted_not_ended_on_timeout():
            group = stateless_init_torch_distributed_process_group(
                master_addr,
                int(master_port),
                rank,
                int(world_size),
                "nccl",  # RCCL on ROCm
                group_name=group_name,
                timeout=timeout,
            )
        self._rdma_groups[group_name] = _WeightGroup(group, join, timeout)
        logger.info(
            "%s: joined RDMA weight group %s as rank %d/%d (worker %d)",
            self.label,
            group_name,
            rank,
            world_size,
            self.rank,
        )
        return True

    @_one_update
    def receive_weights_rdma(
        self,
        group_name: str,
        version: int,
        require_full: bool = False,
        refusal: str | None = None,
    ) -> int:
        """Receive one version of the weights into resident memory; the
        number of tensors written.

        *refusal* is the engine's reason to refuse the stream, which is
        received and discarded all the same: refused before it began, it
        would leave the trainer in its first broadcast. *require_full*: see
        ``_complete_sync``.
        """
        version = int(version)
        weight_group = self._rdma_groups.get(group_name)
        # Agreed before the first header: a worker without the group, raising
        # alone, would leave the others waiting for it in the stream's first
        # vote.
        (outcome,) = self._vote_with_engine_peers(
            _RECEIVING if weight_group is not None else _BROKEN_OFF
        )
        if outcome != _RECEIVING:
            raise WeightReloadRefused(
                f"RDMA weight group {group_name!r} is not joined on every "
                f"worker of this engine; destroy it on every rank and join it "
                f"again"
            )
        last = weight_group.last_version
        if refusal is None and last is not None and version <= last:
            refusal = (
                f"weight versions must rise within a group: last={last}, "
                f"got={version}"
            )
        try:
            stats = receive_weight_stream(
                weight_group.group,
                self,
                device=self.device,
                version=version,
                require_full=require_full,
                timeout=weight_group.timeout,
                agree=lambda ballot: self._vote_with_engine_peers(ballot)[0],
                leave=lambda: self.destroy_rdma_weight_group(group_name),
                refusal=None if refusal is None else WeightReloadRefused(refusal),
            )
        except RDMAStreamOutOfStep as exc:
            logger.error(
                "%s: left RDMA weight group %s, which has to be joined again: %s",
                self.label,
                group_name,
                exc,
            )
            self.destroy_rdma_weight_group(group_name)
            raise
        # Only a stream written and completed uses up its version: a retry of
        # one that failed is the same version again.
        weight_group.last_version = version
        logger.info(
            "%s: RDMA weight stream v%d written: %s", self.label, version, stats
        )
        return stats["updated"]

    def destroy_rdma_weight_group(self, group_name: str) -> bool:
        from atom.utils.distributed.utils import forget_process_group

        weight_group = self._rdma_groups.pop(group_name, None)
        if weight_group is None:
            return True
        try:
            dist.destroy_process_group(weight_group.group)
        except Exception as exc:  # noqa: BLE001 - teardown must not mask the real error
            # Or the group's name stays taken, and it cannot be joined again.
            forget_process_group(weight_group.group)
            logger.warning(
                "%s: tearing down RDMA weight group %s failed: %s",
                self.label,
                group_name,
                exc,
            )
        return True
