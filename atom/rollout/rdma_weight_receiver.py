# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Receive trained weights straight into resident GPU memory over RCCL.

Every TP worker joins an independent process group alongside the trainer, so the
trainer broadcasts BF16 buckets directly into this engine's parameters. No
safetensors round-trip, no shared folder, no host bounce.

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

Two things make this more than a memcpy. ATOM's parameters are *fused* --
``qkv_proj`` and ``gate_up_proj`` combine several checkpoint tensors -- and the
shards of one fused parameter can land in different buckets, so state has to
survive across buckets. And applying weights in place means a stream that fails
halfway leaves the model as a mix of old and new, which is worse than not having
started: inference would keep serving, quietly wrong. Hence the transaction
below, and the fence that refuses to serve after a partial write.
"""

from __future__ import annotations

import json
import logging
import sys
import time
import traceback
from collections import Counter
from collections.abc import Callable
from datetime import timedelta
from itertools import pairwise

import torch
import torch.distributed as dist

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

# The version of LumenRL's RDMA capability contract, whose wire format is the
# one decoded here; see get_rdma_capabilities.
_RDMA_CAPABILITY_VERSION = 2

# A rank's ballot at each vote on a stream; a vote's outcome is the least
# ballot any rank cast. See receive_weight_stream.
_BROKEN_OFF = 0  # cannot follow the stream any further, or cannot commit it
_PREPARED = 1  # at the end marker, with this rank's commit prepared
_RECEIVING = 2  # can receive the frame just announced


class RDMAStreamOutOfStep(RuntimeError):
    """This rank can no longer follow the stream.

    A broadcast is received whole, into a buffer of the size it was sent with,
    so a frame one receiver cannot allocate for is one none of them can take
    part in -- and the sender, already broadcasting it, cannot be told. A
    broadcast that failed, typically because a peer left, puts the group in the
    same place, and so does a vote that failed: the group still lists the peer
    it could not reach. Nothing more can ride it: this rank's end is torn down,
    and the sender and any peer still in a broadcast are released when their
    side of it fails, or by the group's timeout.
    """


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


def _recv_header(group, *, device: torch.device) -> tuple[int, int, int, int]:
    header = torch.empty(_HEADER_WORDS, dtype=torch.int64, device=device)
    dist.broadcast(header, src=0, group=group)
    command, metadata_bytes, payload_bytes, version = (
        int(v) for v in header.cpu().tolist()
    )
    return command, metadata_bytes, payload_bytes, version


def _is_count(value) -> bool:
    """Whether *value* is a size the sender could have written: a JSON integer,
    not negative, and not a bool, which Python counts as an int."""
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _decode_bucket(
    metadata_tensor: torch.Tensor, payload: torch.Tensor
) -> list[tuple[str, torch.Tensor]]:
    """Turn one received bucket into named tensor views over the payload.

    Views, not copies: the payload stays alive for the duration of the bucket's
    application, and copying here would double the peak footprint of a transfer
    already sized in tens of GB. The one exception is a tensor whose offset is
    not a multiple of its own element size, which view() cannot reinterpret in
    place; that tensor alone is copied.
    """
    # Through a byte buffer: tolist() builds a list slot per byte, which at the
    # metadata ceiling is half a gigabyte of host memory before JSON sees it.
    metadata = json.loads(metadata_tensor.cpu().numpy().tobytes().decode("utf-8"))
    if not isinstance(metadata, list) or not metadata:
        raise RuntimeError("invalid RDMA weight metadata: expected a non-empty list")

    weights: list[tuple[str, torch.Tensor]] = []
    spans: list[tuple[int, int, str]] = []
    for entry in metadata:
        # Checked as sent, not coerced: int() made a shape of [2.5] into [2],
        # another tensor that still fits its bytes, and let -1 through, which
        # view() takes as "infer this dimension".
        fields = entry if isinstance(entry, dict) else {}
        name, shape = fields.get("name"), fields.get("shape")
        start, nbytes = fields.get("offset"), fields.get("nbytes")
        dtype_name = fields.get("dtype")
        dtype = (
            getattr(torch, dtype_name, None) if isinstance(dtype_name, str) else None
        )
        if not (
            isinstance(name, str)
            and name
            and isinstance(shape, list)
            and all(_is_count(dim) for dim in shape)
            and isinstance(dtype, torch.dtype)
            and _is_count(start)
            and _is_count(nbytes)
        ):
            raise RuntimeError(f"invalid RDMA weight metadata entry: {entry}")

        end = start + nbytes
        # Bounds-check before slicing: a corrupt offset would otherwise either
        # silently truncate or read a neighbouring tensor's bytes.
        if nbytes == 0 or end > payload.numel():
            raise RuntimeError(
                f"RDMA payload range out of bounds for {name}: "
                f"offset={start} nbytes={nbytes} payload={payload.numel()}"
            )
        if torch.Size(shape).numel() * dtype.itemsize != nbytes:
            raise RuntimeError(
                f"RDMA metadata size mismatch for {name}: "
                f"shape={shape} dtype={dtype} nbytes={nbytes}"
            )
        raw = payload[start:end]
        if start % dtype.itemsize:
            # The sender packs tensors back to back, so one that follows an
            # odd-sized tensor of a narrower dtype starts off its alignment.
            raw = raw.clone()
        value = raw.view(dtype).view(shape)
        weights.append((name, value))
        spans.append((start, end, name))

    # Each entry can be in bounds and still be wrong as a set: a repeated name
    # writes one parameter twice, and overlapping ranges hand the same bytes to
    # two parameters. Either way coverage still passes. The sender packs one
    # tensor per name, back to back, so neither is ever legitimate.
    names = [name for _, _, name in spans]
    if len(set(names)) != len(names):
        repeated = sorted(n for n, seen in Counter(names).items() if seen > 1)
        raise RuntimeError(f"RDMA bucket repeats weight names: {repeated[:20]}")
    spans.sort()
    for (_, prev_end, prev_name), (start, _, name) in pairwise(spans):
        if start < prev_end:
            raise RuntimeError(
                f"RDMA payload ranges overlap: {prev_name} ends at byte "
                f"{prev_end}, {name} starts at {start}"
            )
    return weights


def _workers_per_engine(config) -> int:
    """The engine's own stride, which ModelRunner places devices by: one
    worker per TP shard that has a process, per prefill-context rank. The
    logical TP width over-counts under simulated TP and under-counts under
    PCP."""
    return int(getattr(config, "tp_world_size", 1) or 1) * int(
        getattr(config, "prefill_context_parallel_size", 1) or 1
    )


def _has_mla_attention(model) -> bool:
    """Whether *model* attends through MLA, whose decode reads W_K and W_V
    derived from ``kv_b_proj`` at load.

    By what an MLA layer has, not by one class: each model defines its own
    attention module around a ``kv_b_proj`` -- DeepseekV2MLAAttention,
    Glm5NextMLAAttention and others -- and reaches MLAAttention only as the
    impl ``Attention(use_mla=True)`` builds, which a model may replace.
    MLAAttention is looked up, not imported: importing it pulls in aiter, and
    no instance exists unless it was loaded.
    """
    modules = getattr(model, "modules", None)
    if not callable(modules):
        return False
    mla = getattr(sys.modules.get("atom.model_ops.attention_mla"), "MLAAttention", None)
    return any(
        hasattr(module, "kv_b_proj")
        or getattr(module, "use_mla", False) is True
        or (mla is not None and isinstance(module, mla))
        for module in modules()
    )


def _alone(ballot: int, began: bool) -> tuple[int, bool]:
    """The outcome of a vote this rank took by itself."""
    return ballot, began


def vote_on_stream(group, ballot: int, began: bool) -> tuple[int, bool]:
    """One vote on a stream, by every rank of *group*.

    Each rank casts ``_RECEIVING``, ``_PREPARED`` or ``_BROKEN_OFF``, and says
    whether it began the stream at all; the outcome is the least ballot cast,
    and whether any rank began. *group* is a CPU one: a vote is two integers.
    """
    votes = torch.tensor([int(ballot), -int(began)], dtype=torch.int64)
    dist.all_reduce(votes, op=dist.ReduceOp.MIN, group=group)
    return int(votes[0].item()), votes[1].item() < 0


@torch.no_grad()
def receive_weight_stream(
    group,
    runner,
    *,
    device: torch.device,
    expected_version: int,
    agree=_alone,
    leave: Callable[[], object] | None = None,
    refusal: Exception | None = None,
) -> dict[str, float]:
    """Consume one weight stream and apply it as a single transaction.

    Returns throughput and coverage statistics. Once this stream's reload has
    begun, any failure fences the runner rather than leaving it half-updated --
    see ``abort_weight_update``. A stream refused at begin wrote nothing, and
    leaves the runner as it was, a reload another caller has open included --
    unless a peer took it.

    *agree* is how the ranks taking one stream decide on it; see
    ``vote_on_stream``. They vote on every frame's header before any of them
    enters the frame's broadcasts, so a frame one rank cannot receive is
    broken off by all of them there, rather than leaving the others in a
    broadcast it never joined. And they vote at the end marker: one rank's
    version alone would serve the engine mixed, so each prepares its commit
    and commits only if every rank can. If that fails and any rank began the
    stream, every rank ends fenced. The first vote with any ballot below
    ``_RECEIVING`` is every rank's last, so each casts as many as its peers
    whatever happened to it; a peer already in a vote would otherwise wait for
    it until its group timed out.

    A failure on this rank does not end its part in the stream. The trainer and
    every other rank are still in the broadcasts, and one rank leaving early
    hangs all of them in the next one -- the orchestrator then sees a timeout
    naming nobody instead of this rank's error. So the rest of the stream is
    received and discarded, the header's sizes being all that is needed to stay
    in step, and the failure is raised once the end marker arrives. What breaks
    the stream off before then -- a header that frames no bucket, a frame some
    rank cannot receive, a broadcast that fails -- raises
    ``RDMAStreamOutOfStep``.

    *leave* tears down this rank's end of *group*. A rank whose broadcast
    failed calls it before its last vote: a peer still in that broadcast is
    released only once this rank's end is gone, and the vote would wait for
    that peer until then.

    *refusal*, if given, refuses the stream as a refused begin does: it is
    received in full and discarded, nothing is written, and the refusal is
    raised at the end marker.
    """
    total_bytes = 0
    total_weights = 0
    total_buckets = 0
    started = time.perf_counter()
    failure: Exception | None = None
    drained = 0
    began = False
    any_began = False
    ended = False
    # Set by the vote that is every rank's last -- at a frame some rank cannot
    # take, or at the end marker -- and by no other. A frame vote all ranks
    # pass leaves it unset, so a failure in the broadcasts after it still
    # leaves the group and votes.
    decided = False
    vote_failed = False

    def decide(ballot: int) -> tuple[int, bool]:
        nonlocal vote_failed
        try:
            return agree(ballot, began)
        except Exception:
            vote_failed = True
            raise

    try:
        try:
            if refusal is not None:
                raise refusal
            runner.begin_weight_update(expected_version)
            began = True
        except Exception as exc:  # noqa: BLE001 - raised at the end marker
            # Refused before anything was written -- a replayed or older
            # version, or a reload already open. Received all the same, or the
            # trainer and every other rank wait in the next broadcast. Nothing
            # is aborted on the way out: a reload already open is its caller's
            # to finish.
            failure = exc
            logger.error(
                "RDMA weight stream v%d refused on this rank, receiving it "
                "before reporting that: %s",
                expected_version,
                exc,
            )
        while True:
            # The last bucket goes before anything else is allocated. Held
            # until its names were reassigned, it was still resident while the
            # next one was allocated -- two buckets at the peak -- and the
            # final one stayed resident through commit.
            metadata_tensor = payload = weights = None
            command, metadata_bytes, payload_bytes, version = _recv_header(
                group, device=device
            )
            # Checked on every header, not just the first: a version change
            # mid-stream means two senders are interleaving on one group, and
            # the halves would silently mix.
            if failure is None and version != expected_version:
                failure = RuntimeError(
                    f"RDMA weight version mismatch: expected {expected_version}, "
                    f"got {version}"
                )
            if command == _CMD_END:
                # The contract is [END, 0, 0, version]. Anything else is a
                # corrupt stream or a mismatched sender, not a clean finish.
                if failure is None and (metadata_bytes != 0 or payload_bytes != 0):
                    failure = RuntimeError(
                        f"invalid RDMA end marker: metadata_bytes={metadata_bytes} "
                        f"payload_bytes={payload_bytes}, both must be 0"
                    )
                ended = True
                break
            frame = (
                f"command={command} metadata_bytes={metadata_bytes} "
                f"payload_bytes={payload_bytes}"
            )
            # Decided by every rank before any enters the frame's broadcasts.
            # Free memory differs between them, and one that could not
            # allocate, leaving alone, left its peers in a broadcast it never
            # joined while it waited for them in the vote. A header that frames
            # no bucket is the same header on every rank, so all of them leave
            # at it together.
            reason = _frames_no_bucket(
                command, metadata_bytes, payload_bytes
            ) or _unreceivable(device, metadata_bytes, payload_bytes)
            if reason is None:
                try:
                    metadata_tensor = torch.empty(
                        metadata_bytes, dtype=torch.uint8, device=device
                    )
                    payload = torch.empty(
                        payload_bytes, dtype=torch.uint8, device=device
                    )
                except RuntimeError as exc:  # out of memory, within the bounds
                    metadata_tensor = payload = None
                    reason = str(exc)
            outcome, any_began = decide(_RECEIVING if reason is None else _BROKEN_OFF)
            if outcome != _RECEIVING:
                decided = True
                raise RDMAStreamOutOfStep(
                    f"RDMA weight frame cannot be received: {frame} "
                    f"({reason or 'not by every rank taking the stream'})"
                )

            # A bucket is followed by these two broadcasts, so they are received
            # before the bucket is judged: one this rank cannot use still has
            # to keep it in step.
            dist.broadcast(metadata_tensor, src=0, group=group)
            dist.broadcast(payload, src=0, group=group)
            if failure is not None:
                drained += 1
                continue

            try:
                weights = _decode_bucket(metadata_tensor, payload)
                runner.apply_weight_bucket(weights, payload_bytes=payload_bytes)
            except Exception as exc:  # noqa: BLE001 - raised at the end marker
                # Kept until the end marker, so the frames it was raised
                # through must not keep this bucket alive while the rest of
                # the stream is drained.
                traceback.clear_frames(exc.__traceback__)
                failure = exc
                logger.error(
                    "RDMA weight bucket %d failed on this rank, receiving the "
                    "rest of the stream before reporting it: %s",
                    total_buckets + 1,
                    exc,
                )
                continue

            total_bytes += payload_bytes
            total_weights += len(weights)
            total_buckets += 1

        if failure is None:
            try:
                runner.prepare_weight_commit(expected_version)
                # Before the vote: the writes are asynchronous, so a device
                # fault in them, or in the relayout, surfaces only here.
                torch.cuda.synchronize(device)
            except Exception as exc:  # noqa: BLE001 - voted on, then raised
                failure = exc
        outcome, any_began = decide(_PREPARED if failure is None else _BROKEN_OFF)
        decided = True
        if failure is None and outcome != _PREPARED:
            failure = RuntimeError(
                f"RDMA weight stream v{expected_version} not committed: another "
                f"rank taking it could not"
            )
        if failure is not None:
            if drained:
                logger.error(
                    "RDMA weight stream v%d: discarded %d bucket(s) after the "
                    "failure",
                    expected_version,
                    drained,
                )
            raise failure

        manifest = runner.finish_weight_commit(expected_version)
    except Exception as exc:
        if not decided:
            # Broken off between votes, in a broadcast a peer may still be in.
            # That peer is released only once this rank's end is gone, so it
            # goes first: voting first would wait for the peer until the group
            # timed out, with nothing left to tear the group down.
            if leave is not None:
                try:
                    leave()
                except Exception:  # the stream's own failure is raised
                    logger.exception(
                        "RDMA weight stream v%d: leaving its group failed",
                        expected_version,
                    )
            if vote_failed:
                # Not voted on again over a group that just failed to vote.
                any_began = True  # unknown, so take it that a peer began
            else:
                try:
                    _, any_began = agree(_BROKEN_OFF, began)
                except Exception:  # noqa: BLE001 - the stream's own failure is raised
                    any_began = True  # unknown, so take it that a peer began
        if began:
            # Fence before re-raising: the parameters are now a mix of
            # versions, so serving must stop until a later full reload succeeds.
            runner.abort_weight_update(expected_version, exc)
        elif any_began:
            # A peer took what this rank refused, so the engine's ranks no
            # longer hold one version -- and a rank still serving would wait in
            # its next forward's collectives for peers that refuse to.
            runner.fence_weight_update(exc)
        if not isinstance(exc, RDMAStreamOutOfStep):
            if not ended:
                raise RDMAStreamOutOfStep(
                    f"RDMA weight stream broke off before its end marker: {exc}"
                ) from exc
            if vote_failed:
                raise RDMAStreamOutOfStep(
                    f"RDMA weight stream v{expected_version}: the ranks taking it "
                    f"could not vote on it: {exc}"
                ) from exc
        raise

    elapsed = time.perf_counter() - started
    return {
        "version": float(expected_version),
        "buckets": float(total_buckets),
        "weights": float(total_weights),
        "bytes": float(total_bytes),
        "seconds": elapsed,
        "gbps": (total_bytes * 8 / 1e9 / elapsed) if elapsed > 0 else 0.0,
        "loaded_internal": float(manifest["loaded_internal"]),
    }


class RDMAWeightReceiverMixin:
    """The worker methods a trainer-side orchestrator drives over collective_rpc.

    Named to match the vLLM worker extension, so the same caller drives either
    backend.
    """

    def get_rdma_capabilities(self) -> dict[str, object]:
        """This worker's RDMA reload capabilities, in the contract LumenRL's
        orchestrator reads from the vLLM worker
        (``lumenrl.engine.inference.rdma_protocol``).

        Answered as this receiver is, not as that check wants: it takes
        full-precision streams and requantises them to the resident format,
        on an engine it does not refuse RDMA reloads on, and no prequantised
        stream at all. ``refusal`` says why an engine takes none.
        """
        refusal = self._rdma_reload_refusal()
        return {
            "protocol_version": _RDMA_CAPABILITY_VERSION,
            "module_path": RDMAWeightReceiverMixin.__module__,
            "online_quant_reload": refusal is None,
            "prequantized_stream": False,
            "refusal": refusal,
        }

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

        Rank 0 is the trainer, so every worker sits at ``base_rank`` or above.
        The caller assigns one ``base_rank`` per replica, whose workers follow
        it in order. Refused before joining on an engine where a reload over
        RDMA would be unsafe, so the trainer's rendezvous fails rather than a
        later stream.
        """
        # Vendored, so a container with ATOM but no RL framework still works.
        from atom.utils.independent_process_group import (
            init_independent_process_group,
        )

        refusal = self._rdma_reload_refusal()
        if refusal:
            raise RuntimeError(f"cannot receive weights over RDMA: {refusal}")

        groups = getattr(self, "_rdma_weight_groups", None)
        if groups is None:
            groups = {}
            self._rdma_weight_groups = groups
        joins = getattr(self, "_rdma_weight_group_joins", None)
        if joins is None:
            joins = {}
            self._rdma_weight_group_joins = joins

        rank = int(base_rank) + int(self.rank)
        if rank <= 0 or rank >= int(world_size):
            workers = _workers_per_engine(getattr(self, "config", None))
            raise ValueError(
                f"invalid RDMA rank {rank} for world_size={world_size} "
                f"(base_rank={base_rank}, worker={self.rank}, "
                f"workers_per_engine={workers}); rank 0 is the trainer"
            )

        join = {
            "master": f"{master_addr}:{int(master_port)}",
            "world_size": int(world_size),
            "rank": rank,
        }
        if group_name in groups:
            # Idempotent only for the same join: the orchestrator may retry
            # init after a partial failure elsewhere, and rejoining would
            # deadlock the rendezvous. A changed one -- a restarted trainer, a
            # new rank layout -- would leave this worker in the old group while
            # its peers joined the new one.
            if joins.get(group_name, join) != join:
                raise RuntimeError(
                    f"RDMA weight group {group_name!r} is already joined as "
                    f"{joins[group_name]}, not {join}; call "
                    f"destroy_rdma_weight_group({group_name!r}) on every rank "
                    f"before joining it again"
                )
            logger.warning(
                "%s: RDMA weight group %s already joined, reusing",
                getattr(self, "label", "runner"),
                group_name,
            )
            return True

        groups[group_name] = init_independent_process_group(
            backend="nccl",  # RCCL on ROCm
            init_method=f"tcp://{master_addr}:{master_port}",
            timeout=timedelta(seconds=int(timeout_s)),
            world_size=int(world_size),
            rank=rank,
            group_name=group_name,
        )
        joins[group_name] = join
        logger.info(
            "%s: joined RDMA weight group %s as rank %d/%d (worker %d)",
            getattr(self, "label", "runner"),
            group_name,
            rank,
            world_size,
            self.rank,
        )
        return True

    def _rdma_reload_refusal(self) -> str | None:
        """Why a reload over RDMA would be unsafe on this engine, if it would."""
        config = getattr(self, "config", None)
        # Not the runner's own config alone: once a DP engine's workers have a
        # world of their own, it says data_parallel_size=1 on every one of them.
        dp = getattr(self, "_deployment_data_parallel_size", None)
        if dp is None:
            parallel = getattr(config, "parallel_config", None)
            dp = getattr(parallel, "data_parallel_size", 1)
        dp = int(dp or 1)
        if dp > 1:
            return (
                f"data_parallel_size={dp}. One group would span {dp} "
                f"EngineCores, which enter its broadcast independently -- one "
                f"can block in it while another waits in the per-step DP "
                f"all-reduce -- and each decides the stream for its own "
                f"workers alone, so DP replicas could commit different "
                f"versions. Run each DP replica as its own server."
            )
        pp = int(getattr(config, "pipeline_parallel_size", 1) or 1)
        if pp > 1:
            return (
                f"pipeline_parallel_size={pp}. Each stage holds its own layers "
                f"and its workers alone are counted for the stream: it would "
                f"take, vote on and commit the stream as if it held the whole "
                f"model, so a stage that failed would leave the others serving "
                f"a newer version than its own."
            )
        pcp = int(getattr(config, "prefill_context_parallel_size", 1) or 1)
        if pcp > 1:
            return (
                f"prefill_context_parallel_size={pcp}. A worker's rank counts "
                f"every prefill-context rank of its stage, but the weight "
                f"updater takes it for the worker's TP rank when it slices a "
                f"full tensor, so some workers would load another rank's shard, "
                f"or one that does not exist."
            )
        if _has_mla_attention(getattr(self, "model", None)):
            return (
                "the model attends through MLA, whose decode reads W_K and W_V "
                "derived from kv_b_proj once, at load. A reload rewrites "
                "kv_b_proj and leaves them as they were, so the engine would "
                "serve old and new weights together."
            )
        return None

    def _vote_with_engine_peers(self, ballot: int, began: bool) -> tuple[int, bool]:
        """The stream's decision across this engine's workers, every one of
        which takes it: one worker's yes is not enough to serve it.

        Over aiter's world group, not its TP group. A runner's world is its
        engine's workers alone, DP being isolated per engine, and holds every
        prefill-context rank the stream reaches, where a TP group holds one.
        With DP > 1 refused at init, those workers are every receiver in the
        weight group, so no receiver commits a stream another rejected.
        """
        workers = _workers_per_engine(getattr(self, "config", None))
        if workers <= 1:
            return ballot, began
        from aiter.dist.parallel_state import get_world_group

        world = get_world_group()
        if world.world_size != workers:
            # A wider world would hang the vote; a narrower one would leave
            # some of the stream's receivers out of it.
            raise RuntimeError(
                f"cannot decide the RDMA stream: the process world has "
                f"{world.world_size} rank(s), the engine {workers} worker(s)"
            )
        return vote_on_stream(world.cpu_group, ballot, began)

    def receive_weights_rdma(
        self,
        group_name: str,
        version: int,
        verify_full_load: bool = True,
        prequantized_fp8: bool = False,
    ) -> dict[str, float]:
        """Receive one version of the weights into resident memory.

        Commits only a stream that rewrote every parameter. ``verify_full_load``
        is accepted because the caller drives the vLLM worker with the same
        arguments, but it cannot turn that off: a commit of part of the model
        would serve the rest from another version.

        So is ``prequantized_fp8``, which a stream is refused for. This
        receiver takes full-precision weights and requantises them to its
        resident FP8 format itself; a stream already quantised carries FP8
        bytes, of a format the resident one may not be, with scales named
        ``weight_scale_inv`` that nothing here maps onto the resident
        ``weight_scale``. It is received and discarded, so the trainer is not
        left in its broadcast, and nothing is written.
        """
        label = getattr(self, "label", "runner")
        groups = getattr(self, "_rdma_weight_groups", {})
        if group_name not in groups:
            raise RuntimeError(
                f"RDMA weight group is not initialized: {group_name!r}; "
                "call init_rdma_weight_group first"
            )
        if not verify_full_load:
            logger.warning(
                "%s: verify_full_load=False ignored; an RDMA reload commits only "
                "once it has rewritten every parameter",
                label,
            )
        try:
            stats = receive_weight_stream(
                groups[group_name],
                self,
                device=self.device,
                expected_version=int(version),
                agree=self._vote_with_engine_peers,
                leave=lambda: self.destroy_rdma_weight_group(group_name),
                refusal=(
                    NotImplementedError(
                        "a prequantized FP8 stream is not supported: send the "
                        "weights unquantised, and this receiver requantises "
                        "them to its resident format"
                    )
                    if prequantized_fp8
                    else None
                ),
            )
        except RDMAStreamOutOfStep as exc:
            logger.error(
                "%s: left RDMA weight group %s, which has to be initialized "
                "again: %s",
                label,
                group_name,
                exc,
            )
            self.destroy_rdma_weight_group(group_name)
            raise
        logger.info(
            "%s: RDMA weight reload v%d committed: %s", label, int(version), stats
        )
        return stats

    def destroy_rdma_weight_group(self, group_name: str) -> bool:
        getattr(self, "_rdma_weight_group_joins", {}).pop(group_name, None)
        groups = getattr(self, "_rdma_weight_groups", {})
        group = groups.pop(group_name, None)
        if group is None:
            return True
        try:
            dist.destroy_process_group(group)
        except Exception as exc:  # noqa: BLE001 - teardown must not mask the real error
            # A failure here is not worth propagating: the caller is usually
            # already unwinding from something else, and the group is gone from
            # our map either way.
            logger.warning(
                "%s: tearing down RDMA weight group %s failed: %s",
                getattr(self, "label", "runner"),
                group_name,
                exc,
            )
        return True
