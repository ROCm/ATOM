# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""The RDMA weight receiver: the wire format, and keeping every rank in step.

The wire format is frozen by the sender
(``lumenrl.engine.inference.rdma_weight_transfer``) and is reproduced here from
its own constants rather than copied by eye, so a drift on either side shows up
as a test failure instead of a 61 GB transfer that decodes to garbage.

What a stream writes, and the fence around it, are the engine's weight sync:
``test_weight_sync_coverage.py`` and ``test_collective_rpc_dispatch.py``.
"""

import json
import os
import sys
from datetime import timedelta
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from aiter_stub import stubbed_aiter

with stubbed_aiter():
    import atom.rollout.rdma_weight_receiver as receiver
    from atom.model_engine.collective_rpc import (
        COLLECTIVE_RPC_METHODS,
        UTILITY_MANAGED_RUNNER_METHODS,
        WeightReloadRefused,
    )
    from atom.rollout.capabilities import CapabilityProviderMixin
    from atom.rollout.rdma_weight_receiver import (
        _CMD_BUCKET,
        _CMD_END,
        _HEADER_WORDS,
        _decode_frame,
    )
    from atom.rollout.weight_updater import WeightUpdaterMixin


def test_only_the_group_lifecycle_is_on_the_public_rpc_path():
    """The stream writes weights, so it is the engine's command, inside the
    weight sync that owns the fence; joining and leaving the group are not."""
    assert {"init_rdma_weight_group", "destroy_rdma_weight_group"} <= (
        COLLECTIVE_RPC_METHODS
    )
    assert "receive_weights_rdma" not in COLLECTIVE_RPC_METHODS
    assert "receive_weights_rdma" in UTILITY_MANAGED_RUNNER_METHODS


def test_the_caller_waits_for_a_stream_as_long_as_the_engines_do():
    """At five minutes the caller gave up on a stream the engines were still
    taking, so a version they then wrote was reported unknown, and they stayed
    fenced until a sync rewrote every parameter."""
    with stubbed_aiter():
        from atom.model_engine.engine_utility import WEIGHT_STREAM_TIMEOUT_S
        from atom.rollout.async_engine import AsyncLLMEngine

    sent = []
    engine = AsyncLLMEngine.__new__(AsyncLLMEngine)
    engine.core_mgr = SimpleNamespace(
        broadcast_utility_command_sync=lambda cmd, **kw: sent.append(kw)
        or [{"result": 1}]
    )

    assert engine.receive_weights_rdma("g", 1) == [1]
    assert sent[0]["timeout"] == WEIGHT_STREAM_TIMEOUT_S


# ── the frozen wire format ─────────────────────────────────────────────────


def test_the_command_and_header_constants_match_the_sender():
    """These three numbers are the contract. LumenRL's sender hardcodes the
    same values; a change on either side silently corrupts every transfer."""
    assert _CMD_END == 0
    assert _CMD_BUCKET == 1
    assert _HEADER_WORDS == 4  # command, metadata_bytes, payload_bytes, version


def _encode(entries_and_bytes):
    """Build a (metadata, payload) pair exactly as the sender does."""
    import torch

    entries = []
    blobs = []
    offset = 0
    for name, tensor in entries_and_bytes:
        raw = tensor.contiguous().view(torch.uint8).reshape(-1)
        entries.append(
            {
                "name": name,
                "shape": list(tensor.shape),
                # The sender strips the "torch." prefix.
                "dtype": str(tensor.dtype).removeprefix("torch."),
                "offset": offset,
                "nbytes": raw.numel(),
            }
        )
        blobs.append(raw)
        offset += raw.numel()

    payload = torch.cat(blobs) if blobs else torch.empty(0, dtype=torch.uint8)
    return _metadata(entries), payload


def _metadata(entries):
    import torch

    return torch.tensor(
        list(json.dumps(entries, separators=(",", ":")).encode("utf-8")),
        dtype=torch.uint8,
    )


def test_a_bucket_round_trips_through_the_real_encoding():
    import torch

    a = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.bfloat16)
    b = torch.tensor([5.0, 6.0, 7.0], dtype=torch.float32)

    decoded = _decode_frame(*_encode([("layer.a", a), ("layer.b", b)]))

    assert [name for name, _ in decoded] == ["layer.a", "layer.b"]
    assert torch.equal(decoded[0][1], a) and torch.equal(decoded[1][1], b)


def test_metadata_is_decoded_without_a_list_entry_per_byte():
    """tolist() built a list slot per metadata byte, eight bytes each, before
    JSON saw any of it: half a gigabyte of host memory at the 64 MiB ceiling."""
    import tracemalloc

    import torch

    meta, payload = _encode([("w" * (1 << 20), torch.zeros(1))])
    _decode_frame(meta, payload)  # any lazy import, outside the measurement
    tracemalloc.start()
    try:
        _decode_frame(meta, payload)
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    assert peak < 6 * meta.numel()


def test_the_decoded_tensors_are_views_not_copies():
    """A 61 GB transfer cannot afford to double its peak footprint."""
    import torch

    meta, payload = _encode([("w", torch.tensor([1.0, 2.0], dtype=torch.bfloat16))])
    ((_, view),) = _decode_frame(meta, payload)
    assert view.data_ptr() == payload.data_ptr()


def test_a_tensor_packed_off_its_alignment_is_decoded_too():
    """The sender packs tensors back to back, so an FP32 tensor after three
    BF16 values starts six bytes in, which view() cannot reinterpret. That
    tensor is copied; the rest are still views."""
    import torch

    a = torch.tensor([1.0, 2.0, 3.0], dtype=torch.bfloat16)
    b = torch.tensor([4.5, 5.5], dtype=torch.float32)
    meta, payload = _encode([("a", a), ("b", b)])

    (_, got_a), (_, got_b) = _decode_frame(meta, payload)

    assert torch.equal(got_a, a) and torch.equal(got_b, b)
    assert got_a.data_ptr() == payload.data_ptr()


def test_an_out_of_bounds_offset_is_rejected_not_truncated():
    """Silently clamping would hand a neighbouring tensor's bytes to the model."""
    import torch

    meta, payload = _encode([("w", torch.tensor([1.0], dtype=torch.float32))])
    entries = json.loads(bytes(meta.tolist()).decode())
    entries[0]["nbytes"] = 4096
    with pytest.raises(ValueError, match="out of bounds"):
        _decode_frame(_metadata(entries), payload)


@pytest.mark.parametrize(
    ("mutate", "expect"),
    [
        (lambda e: e.update({"shape": [99]}), "size mismatch"),
        (lambda e: e.pop("name"), "invalid weight bucket entry"),
        (lambda e: e.update({"dtype": "not_a_dtype"}), "invalid weight bucket entry"),
        (lambda e: e.update({"nbytes": 0}), "size mismatch"),
        (lambda e: e.update({"offset": -1}), "invalid weight bucket entry"),
        # Each of these int() coerced into a different layout that still fit
        # the bytes, or into one view() reshapes to fit.
        (lambda e: e.update({"shape": [2.5]}), "invalid weight bucket entry"),
        (lambda e: e.update({"shape": [-1]}), "invalid weight bucket entry"),
        (lambda e: e.update({"shape": [True]}), "invalid weight bucket entry"),
        (lambda e: e.update({"shape": "2"}), "invalid weight bucket entry"),
        (lambda e: e.update({"offset": 0.5}), "invalid weight bucket entry"),
        (lambda e: e.update({"nbytes": 8.0}), "invalid weight bucket entry"),
        (lambda e: e.update({"name": 7}), "invalid weight bucket entry"),
        (lambda e: e.update({"dtype": "tensor"}), "invalid weight bucket entry"),
    ],
)
def test_corrupt_metadata_is_rejected(mutate, expect):
    import torch

    meta, payload = _encode([("w", torch.tensor([1.0, 2.0], dtype=torch.float32))])
    entries = json.loads(bytes(meta.tolist()).decode())
    mutate(entries[0])
    with pytest.raises(ValueError, match=expect):
        _decode_frame(_metadata(entries), payload)


def test_a_zero_element_tensor_is_decoded_wherever_it_sits():
    """SHM and IPC took one before they shared this decoder, which then
    refused it as out of bounds. Its range holds no bytes, so none of a
    neighbour's either."""
    import torch

    w = torch.tensor([1.0, 2.0])
    meta, payload = _encode([("w", w), ("empty", torch.empty(0, 4))])
    entries = json.loads(bytes(meta.tolist()).decode())
    entries[1]["offset"] = 4

    (_, got_w), (_, empty) = _decode_frame(_metadata(entries), payload)

    assert torch.equal(got_w, w) and empty.shape == (0, 4)


def test_an_empty_metadata_list_is_rejected():
    import torch

    with pytest.raises(ValueError, match="non-empty list"):
        _decode_frame(_metadata([]), torch.empty(0, dtype=torch.uint8))


def test_a_repeated_name_is_rejected():
    """Each entry is in bounds, so only the set of them shows it: one parameter
    written twice, the second copy winning silently."""
    import torch

    meta, payload = _encode(
        [
            ("w", torch.tensor([1.0], dtype=torch.float32)),
            ("w", torch.tensor([2.0], dtype=torch.float32)),
        ]
    )
    with pytest.raises(ValueError, match="repeats names"):
        _decode_frame(meta, payload)


def test_overlapping_ranges_are_rejected():
    """The same bytes handed to two parameters."""
    import torch

    meta, payload = _encode(
        [
            ("a", torch.tensor([1.0, 2.0], dtype=torch.float32)),
            ("b", torch.tensor([3.0], dtype=torch.float32)),
        ]
    )
    entries = json.loads(bytes(meta.tolist()).decode())
    entries[1]["offset"] = 4  # inside a's [0, 8)
    with pytest.raises(ValueError, match="overlap"):
        _decode_frame(_metadata(entries), payload)


def test_a_gap_between_ranges_is_not_an_overlap():
    import torch

    payload = torch.zeros(12, dtype=torch.uint8)
    payload[0:4] = torch.tensor([1.0], dtype=torch.float32).view(torch.uint8)
    payload[8:12] = torch.tensor([2.0], dtype=torch.float32).view(torch.uint8)
    entries = [
        {"name": n, "shape": [1], "dtype": "float32", "offset": o, "nbytes": 4}
        for n, o in (("a", 0), ("b", 8))
    ]
    decoded = dict(_decode_frame(_metadata(entries), payload))
    assert decoded["a"].item() == 1.0 and decoded["b"].item() == 2.0


def test_the_receiver_reads_the_header_in_the_documented_order(monkeypatch):
    import torch

    trainer = _Trainer([torch.tensor([1, 10, 20, 7], dtype=torch.int64)])
    monkeypatch.setattr(receiver, "dist", SimpleNamespace(broadcast=trainer.broadcast))
    header = receiver._recv_header(
        None, device=torch.device("cpu"), timeout=timedelta(seconds=60)
    )
    assert header == (1, 10, 20, 7)


# ── the stream: one rank's failure must not strand the others ──────────────


class _Work:
    """A broadcast's handle; *waits* gets each wait's timeout."""

    def __init__(self, waits):
        self.waits = waits

    def wait(self, timeout=None):
        self.waits.append(timeout)
        return True


class _Trainer:
    """Plays the sending side: what each broadcast delivers, in order."""

    def __init__(self, frames):
        self.frames = list(frames)
        self.waits = []

    def broadcast(self, tensor, src=0, group=None, async_op=False):
        assert src == 0 and async_op, "a broadcast left to the watchdog"
        frame = self.frames.pop(0)
        # A receiver out of step posts a broadcast of the wrong size, which
        # RCCL does not check; here it fails loudly instead.
        assert tensor.numel() == frame.numel(), "receiver fell out of step"
        tensor.copy_(frame)
        return _Work(self.waits)


def _frames(buckets, version=1):
    import torch

    frames = []
    for meta, payload in buckets:
        header = [_CMD_BUCKET, meta.numel(), payload.numel(), version]
        frames += [torch.tensor(header, dtype=torch.int64), meta, payload]
    frames.append(torch.tensor([_CMD_END, 0, 0, version], dtype=torch.int64))
    return frames


class _Runner:
    """The updater surface ``receive_weight_stream`` drives."""

    def __init__(self, fail_on_bucket=None, refuse=False, fail_complete=False):
        self._sync = None
        self.events = []
        self.fail_on_bucket = fail_on_bucket
        self.refuse = refuse
        self.fail_complete = fail_complete
        self.applied = 0

    def _open_sync(self):
        if self.refuse:
            raise RuntimeError("weights cannot be reloaded: MLA")
        self._sync = object()
        self.events.append("open")

    def _apply_named_tensors(self, weights):
        self.applied += 1
        self.events.append(f"apply{self.applied}")
        if self.applied == self.fail_on_bucket:
            raise RuntimeError(f"bucket {self.applied} rejected")
        return {"updated": len(weights), "unwritten": []}

    def _refuse_unwritten(self, counts):
        pass

    def _complete_sync(self, require_full):
        self.events.append(f"complete(require_full={require_full})")
        self._sync = None
        if self.fail_complete:
            raise RuntimeError("left 3 parameter(s) as they were")

    def _discard_failed_weight_sync(self):
        self.events.append("discard")
        self._sync = None


def _three_buckets():
    import torch

    return [
        _encode([(f"w{i}", torch.tensor([float(i)], dtype=torch.float32))])
        for i in range(3)
    ]


def _alone(ballot):
    return ballot


def _receive(
    monkeypatch,
    trainer,
    runner,
    *,
    version=1,
    require_full=False,
    agree=_alone,
    leave=lambda: None,
    refusal=None,
):
    import torch

    monkeypatch.setattr(receiver, "dist", SimpleNamespace(broadcast=trainer.broadcast))
    return receiver.receive_weight_stream(
        None,
        runner,
        device=torch.device("cpu"),
        version=version,
        require_full=require_full,
        timeout=timedelta(seconds=60),
        agree=agree,
        leave=leave,
        refusal=refusal,
    )


def _peers(*, take_frames=None, at_end=receiver._ENDED):
    """Peers that take every frame, or the first *take_frames* of them, and
    cast *at_end* at the end marker; records the ballots this rank cast."""

    def agree(ballot):
        agree.ballots.append(ballot)
        frames = agree.ballots.count(receiver._RECEIVING)
        if ballot == receiver._RECEIVING:
            if take_frames is None or frames <= take_frames:
                return ballot
            return receiver._BROKEN_OFF
        if ballot == receiver._ENDED:
            return min(ballot, at_end)
        return ballot

    agree.ballots = []
    return agree


def test_a_clean_stream_is_written_as_one_sync(monkeypatch):
    trainer = _Trainer(_frames(_three_buckets()))
    runner = _Runner()
    agree = _peers()

    stats = _receive(monkeypatch, trainer, runner, require_full=True, agree=agree)

    assert runner.events == [
        "open",
        "apply1",
        "apply2",
        "apply3",
        "complete(require_full=True)",
    ]
    assert stats["updated"] == 3 and stats["buckets"] == 3
    assert agree.ballots == [receiver._RECEIVING] * 3 + [receiver._ENDED]
    assert trainer.frames == [], "the end marker was never read"


@pytest.mark.parametrize("fail_on_bucket", [None, 1], ids=["clean", "draining"])
def test_a_bucket_is_released_before_the_next_is_allocated(monkeypatch, fail_on_bucket):
    """Held until its names were reassigned, the last bucket was still
    resident while the next was allocated: two buckets at the peak. Draining
    after a failure held the failed one the same way, through the traceback
    of the error kept for the end marker."""
    import torch
    from torch.multiprocessing.reductions import StorageWeakRef

    trainer = _Trainer(_frames(_three_buckets()))
    runner = _Runner(fail_on_bucket=fail_on_bucket)
    buffers = []
    empty = torch.empty

    def tracking_empty(*args, **kwargs):
        if kwargs.get("dtype") is torch.uint8:
            # A payload is allocated just after its own metadata, which may
            # still be alive; nothing older may be.
            older = buffers[: len(buffers) - len(buffers) % 2]
            alive = sum(not ref.expired() for ref in older)
            assert not alive, f"{alive} buffer(s) of an earlier bucket still alive"
        tensor = empty(*args, **kwargs)
        if kwargs.get("dtype") is torch.uint8:
            buffers.append(StorageWeakRef(tensor.untyped_storage()))
        return tensor

    monkeypatch.setattr(torch, "empty", tracking_empty)
    if fail_on_bucket is None:
        _receive(monkeypatch, trainer, runner)
    else:
        with pytest.raises(RuntimeError, match="bucket 1 rejected"):
            _receive(monkeypatch, trainer, runner)
    assert len(buffers) == 6, "every bucket's two buffers were allocated"


def test_a_failed_bucket_is_raised_only_after_the_stream_ends(monkeypatch):
    """Leaving at the failure hung the trainer and every other rank in the next
    broadcast, so the caller saw a timeout naming nobody."""
    trainer = _Trainer(_frames(_three_buckets()))
    runner = _Runner(fail_on_bucket=1)
    agree = _peers()

    with pytest.raises(RuntimeError, match="bucket 1 rejected"):
        _receive(monkeypatch, trainer, runner, agree=agree)

    assert trainer.frames == [], "every remaining broadcast must still be joined"
    assert runner.events == ["open", "apply1", "discard"]
    assert agree.ballots[-1] == receiver._ENDED, "still in step at the end"


def test_a_sync_that_cannot_complete_is_raised_once_the_ranks_are_done(monkeypatch):
    """Left as they were by a stream short of the whole model, where the
    weights were not whole: still in step, so the group is kept."""
    trainer = _Trainer(_frames(_three_buckets()))
    runner = _Runner(fail_complete=True)

    with pytest.raises(RuntimeError, match="as they were") as raised:
        _receive(monkeypatch, trainer, runner)

    assert not isinstance(raised.value, receiver.RDMAStreamOutOfStep)
    assert runner.events[-1] == "discard"


def test_a_version_mismatch_drains_before_raising(monkeypatch):
    trainer = _Trainer(_frames(_three_buckets(), version=2))
    runner = _Runner()

    with pytest.raises(RuntimeError, match="expected 1, got 2"):
        _receive(monkeypatch, trainer, runner)

    assert trainer.frames == []
    assert runner.events == ["open", "discard"]


@pytest.mark.parametrize(
    ("metadata_bytes", "payload_bytes", "reason"),
    [
        (-1, 8, "negative size"),
        (1, 2**63 - 1, "payload over"),
        ((64 << 20) + 1, 8, "metadata over"),
    ],
    ids=["negative", "payload-ceiling", "metadata-ceiling"],
)
def test_a_frame_that_cannot_be_allocated_for_fails_at_its_header(
    monkeypatch, metadata_bytes, payload_bytes, reason
):
    """A broadcast is received whole, so a frame no buffer can be posted for
    leaves no step to keep."""
    import torch

    bad = torch.tensor([_CMD_BUCKET, metadata_bytes, payload_bytes, 1])
    trainer = _Trainer([bad, *_frames(_three_buckets())])
    runner = _Runner()
    allocated = []
    empty = torch.empty

    def recording_empty(*args, **kwargs):
        if kwargs.get("dtype") is torch.uint8:
            allocated.append(args)
        return empty(*args, **kwargs)

    monkeypatch.setattr(torch, "empty", recording_empty)
    with pytest.raises(receiver.RDMAStreamOutOfStep, match=reason):
        _receive(monkeypatch, trainer, runner)
    assert allocated == [], "nothing may be allocated for the frame"
    assert runner.events == ["open", "discard"]


def test_an_allocation_that_fails_anyway_is_out_of_step_too(monkeypatch):
    """Within the bounds, memory can still run out."""
    import torch

    trainer = _Trainer(_frames(_three_buckets()))
    empty = torch.empty

    def exhausted(*args, **kwargs):
        if kwargs.get("dtype") is torch.uint8:
            raise torch.OutOfMemoryError("out of memory")
        return empty(*args, **kwargs)

    monkeypatch.setattr(torch, "empty", exhausted)
    with pytest.raises(receiver.RDMAStreamOutOfStep, match="out of memory"):
        _receive(monkeypatch, trainer, _Runner())


@pytest.mark.parametrize(
    "header",
    [[9, 3, 5, 1], [9, 0, 0, 1], [_CMD_BUCKET, 0, 5, 1], [_CMD_BUCKET, 3, 0, 1]],
    ids=["unknown", "unknown-without-sizes", "no-metadata", "no-payload"],
)
def test_a_header_that_frames_no_bucket_breaks_the_stream_off(monkeypatch, header):
    """Received as if it framed a bucket, its broadcasts were ones the sender
    may never post, and every later frame was read out of step."""
    import torch

    trainer = _Trainer(
        [torch.tensor(header, dtype=torch.int64), *_frames(_three_buckets())]
    )
    runner = _Runner()
    after_the_header = len(trainer.frames) - 1

    with pytest.raises(receiver.RDMAStreamOutOfStep, match=f"command={header[0]}"):
        _receive(monkeypatch, trainer, runner)

    assert len(trainer.frames) == after_the_header, "a broadcast was posted for it"
    assert runner.events == ["open", "discard"]


def test_an_end_marker_carrying_sizes_is_not_a_clean_finish(monkeypatch):
    import torch

    frames = _frames(_three_buckets())
    frames[-1] = torch.tensor([_CMD_END, 5, 7, 1], dtype=torch.int64)
    runner = _Runner()

    with pytest.raises(RuntimeError, match="end marker"):
        _receive(monkeypatch, _Trainer(frames), runner)

    assert runner.events[-1] == "discard"
    assert not any(e.startswith("complete") for e in runner.events)


@pytest.mark.parametrize(
    "runner",
    [_Runner(refuse=True), None],
    ids=["model-refused", "engine-refused"],
)
def test_a_refused_stream_is_received_and_writes_nothing(monkeypatch, runner):
    """Refused before the loop, this rank left while the trainer and every
    other rank sat in the next broadcast."""
    trainer = _Trainer(_frames(_three_buckets()))
    refusal = None
    if runner is None:
        runner, refusal = _Runner(), RuntimeError("1 batch(es) in the pipeline")

    with pytest.raises(RuntimeError, match="MLA|in the pipeline"):
        _receive(monkeypatch, trainer, runner, refusal=refusal)

    assert trainer.frames == [], "every remaining broadcast must still be joined"
    assert runner.events == [], "nothing applied, and nothing discarded"


def test_a_stream_leaves_a_sync_another_caller_has_open(monkeypatch):
    """Discarding on the way out ended the reload that refused this stream,
    which was not this stream's."""
    trainer = _Trainer(_frames(_three_buckets()))
    runner = _Runner()
    runner._sync = theirs = object()

    with pytest.raises(RuntimeError, match="another weight sync is open"):
        _receive(monkeypatch, trainer, runner)

    assert runner._sync is theirs and runner.events == []


# ── the ranks taking one stream stay in step ───────────────────────────────


def test_a_rank_that_breaks_off_still_votes(monkeypatch):
    """A peer waits in the vote on the frame; without this rank's ballot it
    waited until its group timed out."""
    import torch

    agree = _peers()
    with pytest.raises(receiver.RDMAStreamOutOfStep):
        _receive(
            monkeypatch,
            _Trainer([torch.tensor([_CMD_BUCKET, -1, 8, 1])]),
            _Runner(),
            agree=agree,
        )
    assert agree.ballots == [receiver._BROKEN_OFF]


def test_a_frame_a_peer_cannot_receive_is_broken_off_here_too(monkeypatch):
    """One rank that could not allocate broke off alone, and its peers went on
    into the frame's broadcasts, held there for it."""
    trainer = _Trainer(_frames(_three_buckets()))
    runner = _Runner()
    agree = _peers(take_frames=1)
    left = []

    with pytest.raises(receiver.RDMAStreamOutOfStep, match="not by every rank"):
        _receive(
            monkeypatch, trainer, runner, agree=agree, leave=lambda: left.append(1)
        )

    assert agree.ballots == [receiver._RECEIVING] * 2
    assert len(trainer.frames) == 6, "the frame's broadcasts were entered"
    assert runner.events == ["open", "apply1", "discard"]
    # Every rank left at that vote, so none is in a broadcast to release.
    assert left == []


def test_a_rank_whose_broadcast_failed_leaves_the_group_before_it_votes(
    monkeypatch,
):
    """A peer may still be in the broadcast that failed here, held there until
    this rank's end of the group is gone."""
    trainer = _Trainer(_frames(_three_buckets()))
    calls, order = [], []

    def failing(tensor, src=0, group=None, async_op=False):
        calls.append(1)
        if len(calls) == 5:  # the second frame's metadata
            raise RuntimeError("broadcast timed out")
        return trainer.broadcast(tensor, src=src, group=group, async_op=async_op)

    peers = _peers()

    def agree(ballot):
        order.append("vote")
        return peers(ballot)

    with pytest.raises(receiver.RDMAStreamOutOfStep, match="broadcast timed out"):
        _receive(
            monkeypatch,
            SimpleNamespace(broadcast=failing),
            _Runner(),
            agree=agree,
            leave=lambda: order.append("leave"),
        )

    assert order == ["vote", "vote", "leave", "vote"]
    assert peers.ballots[-1] == receiver._LEFT


def test_a_rank_that_left_the_group_takes_its_peers_out_of_it_too(monkeypatch):
    """A peer that had reached the end marker kept a group the other was gone
    from, and the next stream waited for it."""
    runner = _Runner()
    with pytest.raises(receiver.RDMAStreamOutOfStep, match="left its group"):
        _receive(
            monkeypatch,
            _Trainer(_frames(_three_buckets())),
            runner,
            agree=_peers(at_end=receiver._LEFT),
        )
    assert runner.events[-1] == "discard"


def test_a_vote_that_failed_puts_the_rank_out_of_step(monkeypatch):
    """A peer gone during the vote: the group still lists the dead rank, and
    is not voted on again."""
    ballots = []

    def agree(ballot):
        ballots.append(ballot)
        if ballot == receiver._ENDED:
            raise RuntimeError("peer gone mid-vote")
        return ballot

    with pytest.raises(receiver.RDMAStreamOutOfStep, match="peer gone"):
        _receive(
            monkeypatch, _Trainer(_frames(_three_buckets())), _Runner(), agree=agree
        )
    assert ballots == [receiver._RECEIVING] * 3 + [receiver._ENDED]


# ── the worker: the group, and the stream as the engine drives it ──────────


def _config(*, tp_world=1, pcp=1, pp=1, dp=1):
    return SimpleNamespace(
        tp_world_size=tp_world,
        prefill_context_parallel_size=pcp,
        pipeline_parallel_size=pp,
        parallel_config=SimpleNamespace(
            data_parallel_size=dp, data_parallel_rank_local=0
        ),
    )


class _Worker(
    WeightUpdaterMixin, receiver.RDMAWeightReceiverMixin, CapabilityProviderMixin
):
    """One engine worker over a model of one weight."""

    label = "test"
    world_size = 1

    def __init__(self, rank=0, **config):
        import torch
        from torch import nn

        self.rank = rank
        self.device = torch.device("cpu")
        self.config = _config(**config)
        self._init_weight_updater()
        self._init_rdma_receiver(self.config)
        self.model = nn.Module()
        self.model.w = nn.Parameter(torch.zeros(1), requires_grad=False)

    def clear_kv_cache(self):
        pass

    def _sync_target_model(self):
        return self.model


def _joins(monkeypatch):
    """The groups init_rdma_weight_group builds, with no rendezvous behind."""
    import atom.utils.distributed.utils as dist_utils

    joined = []

    def build(host, port, rank, world_size, backend, **kw):
        joined.append({"master": f"{host}:{port}", "rank": rank, **kw})
        return object()

    monkeypatch.setattr(
        dist_utils, "stateless_init_torch_distributed_process_group", build
    )
    return joined


def _joined_worker(monkeypatch, trainer, **config):
    import torch

    _joins(monkeypatch)
    destroyed = []
    monkeypatch.setattr(
        receiver,
        "dist",
        SimpleNamespace(
            broadcast=trainer.broadcast, destroy_process_group=destroyed.append
        ),
    )
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *a, **k: None)
    worker = _Worker(**config)
    worker.init_rdma_weight_group(
        "127.0.0.1", 29500, base_rank=1, world_size=2, group_name="g"
    )
    worker.destroyed = destroyed
    return worker


def _stream_of_w(value, version=1):
    import torch

    return _frames([_encode([("w", torch.tensor([value]))])], version=version)


def test_the_group_is_built_with_the_callers_name_and_one_timeout(monkeypatch):
    """A second process-group builder had its own rendezvous, store prefix and
    torch-version handling, and two default timeouts."""
    joined = _joins(monkeypatch)
    _Worker(rank=1).init_rdma_weight_group(
        "10.0.0.1", 29500, base_rank=1, world_size=3, group_name="g", timeout_s=42
    )
    assert joined == [
        {
            "master": "10.0.0.1:29500",
            "rank": 2,
            "group_name": "g",
            "timeout": timedelta(seconds=42),
        }
    ]


def test_a_stream_is_written_into_the_model(monkeypatch):
    trainer = _Trainer(_stream_of_w(5.0))
    worker = _joined_worker(monkeypatch, trainer)

    assert worker.receive_weights_rdma("g", 1) == 1

    assert worker.model.w.item() == 5.0
    assert trainer.frames == [] and worker._sync is None


def test_a_stream_the_engine_refuses_is_received_and_writes_nothing(monkeypatch):
    trainer = _Trainer(_stream_of_w(5.0))
    worker = _joined_worker(monkeypatch, trainer)

    with pytest.raises(RuntimeError, match="in the pipeline"):
        worker.receive_weights_rdma("g", 1, refusal="1 batch(es) in the pipeline")

    assert trainer.frames == [], "the stream was not received"
    assert worker.model.w.item() == 0.0
    assert "g" in worker._rdma_groups, "a group still in step was dropped"


def test_a_stream_reaches_a_weight_its_module_writes(monkeypatch):
    """A module that writes its own weight (``sync_weight``) waits for the
    readers once per update; a stream is one."""
    import torch
    from torch import nn

    class _WritesItsWeight(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.zeros(1), requires_grad=False)

        def sync_weight(self, param_name, tensor, shard_id=None):
            self.weight.data.copy_(tensor)
            return True

    trainer = _Trainer(_frames([_encode([("table.weight", torch.tensor([5.0]))])]))
    worker = _joined_worker(monkeypatch, trainer)
    worker.model = nn.Module()
    worker.model.table = _WritesItsWeight()

    assert worker.receive_weights_rdma("g", 1, require_full=True) == 1
    assert worker.model.table.weight.item() == 5.0


def test_versions_rise_within_a_group(monkeypatch):
    """A replayed or reordered stream is received and refused, not written."""
    trainer = _Trainer([*_stream_of_w(5.0, version=3), *_stream_of_w(9.0, version=3)])
    worker = _joined_worker(monkeypatch, trainer)
    worker.receive_weights_rdma("g", 3)

    with pytest.raises(RuntimeError, match="must rise.*last=3, got=3"):
        worker.receive_weights_rdma("g", 3)

    assert trainer.frames == [] and worker.model.w.item() == 5.0


@pytest.mark.parametrize(
    ("case", "reason"),
    [
        ("no group", "not joined on every worker"),
        ("another sync open", "another weight sync is open"),
        ("a version again", "must rise"),
        ("a header of another version", "version mismatch"),
    ],
)
def test_a_stream_refused_before_any_write_says_so(monkeypatch, case, reason):
    """Each fails before the stream writes anything, yet raised a plain
    RuntimeError, so the engine counted the stream as written and fenced."""
    frames = {
        "a version again": [
            *_stream_of_w(5.0, version=3),
            *_stream_of_w(9.0, version=3),
        ],
        "a header of another version": _stream_of_w(9.0, version=4),
    }.get(case, _stream_of_w(9.0, version=3))
    trainer = _Trainer(frames)
    worker = _joined_worker(monkeypatch, trainer)
    if case == "no group":
        worker._vote_with_engine_peers = lambda *ballots: (receiver._BROKEN_OFF,)
    elif case == "another sync open":
        worker._sync = object()
    elif case == "a version again":
        worker.receive_weights_rdma("g", 3)

    with pytest.raises(WeightReloadRefused, match=reason):
        worker.receive_weights_rdma("g", 3)

    assert worker.model.w.item() in (0.0, 5.0)


def test_a_stream_that_failed_after_writing_is_no_refusal(monkeypatch):
    """Its first bucket landed, so the weights hold part of it."""
    import torch

    buckets = [
        _encode([("w", torch.tensor([5.0]))]),
        _encode([("w", torch.tensor([1.0, 2.0]))]),
    ]
    trainer = _Trainer(_frames(buckets))
    worker = _joined_worker(monkeypatch, trainer)

    with pytest.raises(RuntimeError) as raised:
        worker.receive_weights_rdma("g", 1)

    assert not isinstance(raised.value, WeightReloadRefused)


def test_a_stream_that_failed_does_not_use_up_its_version(monkeypatch):
    """Recorded before the stream was received, the version of one that then
    failed refused its own retry as a version that must rise."""
    import torch

    unwritable = _frames([_encode([("w", torch.tensor([1.0, 2.0]))])])
    trainer = _Trainer([*unwritable, *_stream_of_w(5.0)])
    worker = _joined_worker(monkeypatch, trainer)
    with pytest.raises(RuntimeError, match="w"):
        worker.receive_weights_rdma("g", 1)

    assert worker.receive_weights_rdma("g", 1) == 1
    assert worker.model.w.item() == 5.0


def test_a_version_sent_again_fences_no_engine(monkeypatch):
    """A trainer retried the version the engine already held: every rank took
    the stream and refused it before writing, and the engine, whole and
    serving, was fenced until a sync rewrote every parameter."""
    from tests.test_weight_sync_coverage import _engine_with

    trainer = _Trainer([*_stream_of_w(5.0, version=5), *_stream_of_w(9.0, version=5)])
    worker = _joined_worker(monkeypatch, trainer)
    engine, send = _engine_with(worker)
    args = {"group_name": "g", "version": 5}

    assert "error" not in send(
        "receive_weights_rdma", sync_id="s1", request_id="a", **args
    )
    send("finish_weight_sync", sync_id="s1", completed_request_id="a")
    reply = send("receive_weights_rdma", sync_id="s2", request_id="b", **args)
    send("discard_failed_weight_sync", failed_cmd="receive_weights_rdma", error="")

    assert "must rise" in reply["error"]
    assert not (engine._weight_sync_fenced or engine._rl_weights_inconsistent)
    assert trainer.frames == [] and worker.model.w.item() == 5.0


def test_a_new_group_starts_its_own_version_count(monkeypatch):
    """A restarted trainer counts from 1 again; every stream it sent was
    refused for good."""
    trainer = _Trainer([*_stream_of_w(5.0, version=7), *_stream_of_w(9.0)])
    worker = _joined_worker(monkeypatch, trainer)
    worker.receive_weights_rdma("g", 7)
    worker.destroy_rdma_weight_group("g")
    worker.init_rdma_weight_group(
        "127.0.0.1", 29501, base_rank=1, world_size=2, group_name="g"
    )

    worker.receive_weights_rdma("g", 1)

    assert worker.model.w.item() == 9.0


def test_a_failed_stream_discards_what_it_opened(monkeypatch):
    import torch

    trainer = _Trainer(_frames([_encode([("w", torch.ones(3))])]))
    worker = _joined_worker(monkeypatch, trainer)

    with pytest.raises(RuntimeError, match="could not be written"):
        worker.receive_weights_rdma("g", 1)

    assert worker._sync is None and "g" in worker._rdma_groups


def test_a_partial_stream_cannot_make_the_weights_whole_again(monkeypatch):
    """Only a stream that rewrites every parameter, as any sync."""
    trainer = _Trainer(_frames([_encode([("other", __import__("torch").ones(1))])]))
    worker = _joined_worker(monkeypatch, trainer)

    with pytest.raises(RuntimeError, match=r"not whole.*\['w'\]"):
        worker.receive_weights_rdma("g", 1, require_full=True)


def test_a_worker_without_the_group_votes_before_it_refuses():
    """Raising alone, it left the workers that had the group waiting for it in
    the stream's first vote."""
    worker = _Worker()
    cast = []
    worker._vote_with_engine_peers = lambda *ballots: cast.extend(ballots) or ballots

    with pytest.raises(RuntimeError, match="not joined on every worker"):
        worker.receive_weights_rdma("g", 1)
    assert cast == [receiver._BROKEN_OFF]


def test_a_stream_is_refused_on_every_worker_if_one_lacks_the_group(monkeypatch):
    trainer = _Trainer(_stream_of_w(5.0))
    worker = _joined_worker(monkeypatch, trainer)
    worker._vote_with_engine_peers = lambda *ballots: (receiver._BROKEN_OFF,)
    frames = len(trainer.frames)

    with pytest.raises(RuntimeError, match="not joined on every worker"):
        worker.receive_weights_rdma("g", 1)
    assert len(trainer.frames) == frames, "a header was read"


def test_a_group_whose_teardown_failed_can_be_joined_again(monkeypatch):
    """Its shutdown raised on the broken communicator, so torch kept the
    group's name while this rank forgot the group: joining it again failed
    with "already been created"."""
    import atom.utils.distributed.utils as dist_utils

    worker = _joined_worker(monkeypatch, _Trainer([]))
    (group,) = [g.group for g in worker._rdma_groups.values()]

    def broken(pg):
        raise RuntimeError("communicator aborted")

    forgotten = []
    monkeypatch.setattr(receiver.dist, "destroy_process_group", broken)
    monkeypatch.setattr(dist_utils, "forget_process_group", forgotten.append)

    assert worker.destroy_rdma_weight_group("g")
    assert forgotten == [group]
    assert worker.init_rdma_weight_group(
        "127.0.0.1", 29500, base_rank=1, world_size=2, group_name="g"
    )


def test_a_rank_out_of_step_leaves_the_group(monkeypatch):
    """The sender is mid-broadcast in a frame this rank never posted, so a later
    stream on the same group would be read out of step from its first header."""
    import torch

    trainer = _Trainer([torch.tensor([_CMD_BUCKET, -1, 8, 1])])
    worker = _joined_worker(monkeypatch, trainer)
    (group,) = [g.group for g in worker._rdma_groups.values()]

    with pytest.raises(receiver.RDMAStreamOutOfStep):
        worker.receive_weights_rdma("g", 1)

    assert worker.destroyed == [group]
    with pytest.raises(RuntimeError, match="not joined"):
        worker.receive_weights_rdma("g", 2)


def test_every_broadcast_is_waited_on_for_the_groups_timeout(monkeypatch):
    """Left to the watchdog, a broadcast the trainer never finished ended the
    runner at the group's timeout, and the engine with it."""
    trainer = _Trainer(_stream_of_w(5.0))
    worker = _joined_worker(monkeypatch, trainer)
    frames = len(trainer.frames)

    worker.receive_weights_rdma("g", 1)

    assert trainer.waits == [timedelta(seconds=600)] * frames


def test_the_weight_group_aborts_on_a_timeout_rather_than_end_the_runner(
    monkeypatch,
):
    """A ProcessGroupNCCL reads its error handling once, when it is built;
    under torch's default its watchdog takes the whole process down."""
    import atom.utils.distributed.utils as dist_utils

    seen = []

    def build(*args, **kwargs):
        seen.append(os.environ.get("TORCH_NCCL_ASYNC_ERROR_HANDLING"))
        return object()

    monkeypatch.setattr(
        dist_utils, "stateless_init_torch_distributed_process_group", build
    )
    monkeypatch.setenv("TORCH_NCCL_ASYNC_ERROR_HANDLING", "3")

    _Worker().init_rdma_weight_group(
        "127.0.0.1", 29500, base_rank=1, world_size=2, group_name="g"
    )

    assert seen == ["2"]
    assert (
        os.environ["TORCH_NCCL_ASYNC_ERROR_HANDLING"] == "3"
    ), "the engine's other groups keep the process's setting"


def test_a_trainer_gone_mid_stream_breaks_it_off_at_the_timeout(monkeypatch):
    """Its first bucket arrived and nothing after: the wait on the next
    header raises at the group's timeout, and the stream fails as written."""

    class _Gone(_Trainer):
        def broadcast(self, tensor, src=0, group=None, async_op=False):
            if self.frames:
                return super().broadcast(tensor, src, group, async_op)

            def wait(timeout=None):
                raise RuntimeError("WorkNCCL timed out in blocking wait")

            return SimpleNamespace(wait=wait)

    worker = _joined_worker(monkeypatch, _Gone(_stream_of_w(5.0)[:-1]))
    (group,) = [g.group for g in worker._rdma_groups.values()]

    with pytest.raises(receiver.RDMAStreamOutOfStep, match="timed out"):
        worker.receive_weights_rdma("g", 1)

    assert worker.destroyed == [group] and not worker._rdma_groups


def test_a_peer_whose_broadcast_failed_leaves_the_group_too(monkeypatch):
    def peer_left(tensor, src=0, group=None, async_op=False):
        raise RuntimeError("peer left the broadcast")

    worker = _joined_worker(monkeypatch, SimpleNamespace(broadcast=peer_left))

    with pytest.raises(receiver.RDMAStreamOutOfStep, match="peer left"):
        worker.receive_weights_rdma("g", 1)
    assert len(worker.destroyed) == 1 and not worker._rdma_groups


# ── joining the group ──────────────────────────────────────────────────────


def _unsafe(monkeypatch, which):
    """A worker of an engine that cannot take weights over RDMA."""
    import torch
    from torch import nn

    worker = _Worker(
        dp=2 if which == "dp" else 1,
        pp=2 if which == "pp" else 1,
        pcp=2 if which == "pcp" else 1,
    )
    if which == "mla":
        from atom.utils.selector import Family

        worker.attn_family = Family.MLA
    if which == "int8":
        worker.model.q = nn.Parameter(
            torch.zeros(2, dtype=torch.int8), requires_grad=False
        )
    return worker


_UNSAFE = {
    "dp": "data_parallel_size=2",
    "pp": "pipeline_parallel_size=2",
    "pcp": "prefill_context_parallel_size=2",
    "mla": "through MLA",
    "int8": "full-precision write cannot produce",
}


@pytest.mark.parametrize("which", list(_UNSAFE))
def test_an_engine_that_cannot_take_a_stream_refuses_before_joining(monkeypatch, which):
    """Refused at its first stream instead, every rank joined the group only
    for the stream to fail -- or, under DP, two engines to commit different
    versions."""
    joined = _joins(monkeypatch)
    with pytest.raises(RuntimeError, match=_UNSAFE[which]):
        _unsafe(monkeypatch, which).init_rdma_weight_group(
            "127.0.0.1", 29500, base_rank=1, world_size=2, group_name="g"
        )
    assert joined == []


def test_the_deployments_dp_size_is_read_before_the_runner_hides_it():
    """Once a DP engine's workers have a world of their own, their config says
    data_parallel_size=1. Read, not run: importing the runner needs aiter."""
    import ast

    src = (
        Path(__file__).resolve().parent.parent / "atom/rollout/model_runner_ext.py"
    ).read_text()
    runner = next(
        n
        for n in ast.parse(src).body
        if isinstance(n, ast.ClassDef) and n.name == "RLHFModelRunner"
    )
    init = next(
        n
        for n in runner.body
        if isinstance(n, ast.FunctionDef) and n.name == "__init__"
    )
    calls = [ast.unparse(n.value) for n in init.body if isinstance(n, ast.Expr)]
    assert calls.index("self._init_rdma_receiver(config)") < calls.index(
        "super().__init__(rank, config)"
    )


def test_a_repeated_init_reuses_its_group_only_for_the_same_join(monkeypatch):
    """After a trainer restart on a new endpoint, a worker kept the old group
    while its peers joined the new one, and the next broadcast hung."""
    joined = _joins(monkeypatch)
    monkeypatch.setattr(receiver.dist, "destroy_process_group", lambda group: None)
    worker = _Worker()
    join = {"base_rank": 1, "world_size": 2, "group_name": "g"}

    assert worker.init_rdma_weight_group("10.0.0.1", 29500, **join)
    assert worker.init_rdma_weight_group("10.0.0.1", 29500, **join)  # a retry
    with pytest.raises(RuntimeError, match="already joined"):
        worker.init_rdma_weight_group("10.0.0.2", 29500, **join)
    worker.destroy_rdma_weight_group("g")
    assert worker.init_rdma_weight_group("10.0.0.2", 29500, **join)
    assert [j["master"] for j in joined] == ["10.0.0.1:29500", "10.0.0.2:29500"]


def _voting_worker(outcome):
    """A worker whose engine's vote on joining comes out as *outcome* gives it;
    records the ballots it cast."""
    worker = _Worker()
    worker.cast = []

    def vote(ballot, none_will_join):
        worker.cast.append(ballot)
        return outcome(ballot, none_will_join)

    worker._vote_with_engine_peers = vote
    return worker


def test_no_worker_joins_while_another_of_its_engine_cannot(monkeypatch):
    joined = _joins(monkeypatch)
    worker = _voting_worker(lambda ballot, n: (receiver._CANNOT_JOIN, 0))

    with pytest.raises(RuntimeError, match="another worker of this engine cannot"):
        worker.init_rdma_weight_group(
            "127.0.0.1", 29500, base_rank=1, world_size=2, group_name="g"
        )
    assert worker.cast == [receiver._WILL_JOIN] and joined == []


def test_a_worker_that_cannot_join_says_so_before_any_joins(monkeypatch):
    joined = _joins(monkeypatch)
    worker = _voting_worker(lambda ballot, n: (ballot, n))

    with pytest.raises(ValueError, match="invalid RDMA rank"):
        worker.init_rdma_weight_group(
            "127.0.0.1", 29500, base_rank=1, world_size=1, group_name="g"
        )
    assert worker.cast == [receiver._CANNOT_JOIN] and joined == []


def test_a_worker_that_refuses_casts_its_refusal_as_a_ballot(monkeypatch):
    """Raised before the vote, a refusal one worker found in its own model
    left the peers that found none in the vote's all-reduce."""
    joined = _joins(monkeypatch)
    worker = _voting_worker(lambda ballot, n: (ballot, n))
    worker._rdma_receive_refusal = lambda: "1 parameter(s) held as MXFP4"

    with pytest.raises(RuntimeError, match="cannot receive.*MXFP4"):
        worker.init_rdma_weight_group(
            "127.0.0.1", 29500, base_rank=1, world_size=2, group_name="g"
        )
    assert worker.cast == [receiver._CANNOT_JOIN] and joined == []


def test_a_refusal_is_still_raised_where_the_vote_cannot_be_taken(monkeypatch):
    """Under PP the world spans stages the engine does not count, so the vote
    fails on every worker alike; each raises the refusal."""
    joined = _joins(monkeypatch)
    worker = _Worker(pp=2)

    def untakeable(*ballots):
        raise RuntimeError("the process world has 4 rank(s), the engine 2")

    worker._vote_with_engine_peers = untakeable
    with pytest.raises(RuntimeError, match="pipeline_parallel_size=2"):
        worker.init_rdma_weight_group(
            "127.0.0.1", 29500, base_rank=1, world_size=2, group_name="g"
        )
    assert joined == []


def test_a_group_joined_on_only_some_workers_is_not_reused(monkeypatch):
    """Retried after a partial failure, a worker that had joined reused its
    group while one that had not joined it anew, alone in a rendezvous."""
    joined = _joins(monkeypatch)
    worker = _voting_worker(lambda ballot, n: (ballot, n))
    join = {"base_rank": 1, "world_size": 2, "group_name": "g"}
    worker.init_rdma_weight_group("127.0.0.1", 29500, **join)

    worker._vote_with_engine_peers = lambda ballot, n: (
        min(ballot, receiver._JOINED_ALREADY),
        0,
    )
    with pytest.raises(RuntimeError, match="joined on only some workers"):
        worker.init_rdma_weight_group("127.0.0.1", 29500, **join)
    assert len(joined) == 1


def _world(monkeypatch, size):
    """aiter's world group as the vote finds it, and the groups voted over."""
    world = SimpleNamespace(world_size=size, cpu_group=object())
    parallel_state = ModuleType("aiter.dist.parallel_state")
    parallel_state.get_world_group = lambda: world
    monkeypatch.setitem(sys.modules, "aiter.dist.parallel_state", parallel_state)
    voted = []

    def vote(ballots, group):
        voted.append(group)
        return tuple(ballots)

    monkeypatch.setattr(receiver, "group_min_each", vote)
    return world.cpu_group, voted


@pytest.mark.parametrize(
    ("tp_world", "pcp"), [(1, 2), (2, 2), (2, 1)], ids=["pcp", "tp-and-pcp", "tp"]
)
def test_every_worker_of_the_engine_decides_the_stream(monkeypatch, tp_world, pcp):
    """Over the TP group, one prefill-context rank's receivers decided alone."""
    world, voted = _world(monkeypatch, tp_world * pcp)
    worker = _Worker(tp_world=tp_world, pcp=pcp)

    assert worker._vote_with_engine_peers(2, 1) == (2, 1)
    assert voted == [world]


def test_a_single_worker_engine_decides_alone(monkeypatch):
    _, voted = _world(monkeypatch, 1)
    assert _Worker()._vote_with_engine_peers(0) == (0,)
    assert voted == []


def test_a_world_that_is_not_the_engine_is_not_voted_in(monkeypatch):
    """With ranks beyond the stream's the vote would hang; without some of its
    receivers it would let the rest decide alone."""
    _, voted = _world(monkeypatch, 2)
    with pytest.raises(RuntimeError, match=r"world has 2 rank.*engine 4 worker"):
        _Worker(tp_world=2, pcp=2)._vote_with_engine_peers(2)
    assert voted == []


# ── discovery ──────────────────────────────────────────────────────────────


def test_the_feature_is_advertised_where_the_engine_takes_a_stream():
    report = _Worker().get_worker_capabilities()
    assert "rdma_weight_receive" in report["features"]
    assert {"init_rdma_weight_group", "destroy_rdma_weight_group"} <= set(
        report["methods"]
    )


@pytest.mark.parametrize("which", list(_UNSAFE))
def test_the_feature_is_withheld_where_init_would_refuse(monkeypatch, which):
    """So an orchestrator can choose another path before a rendezvous that
    cannot complete."""
    report = _unsafe(monkeypatch, which).get_worker_capabilities()
    assert "rdma_weight_receive" not in report["features"]
