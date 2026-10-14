# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""The RDMA weight receiver: wire format and transaction semantics.

The wire format is frozen by the sender
(``lumenrl.engine.inference.rdma_weight_transfer``) and is reproduced here from
its own constants rather than copied by eye, so a drift on either side shows up
as a test failure instead of a 61 GB transfer that decodes to garbage.

The transaction exists because weights are applied *in place*. A stream that
fails halfway leaves the model a mix of two versions, and inference would keep
serving -- quietly wrong. So a failure must fence serving, not just log.
"""

import json
import os
import re
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from aiter_stub import stubbed_aiter

with stubbed_aiter():
    import atom.rollout.rdma_weight_receiver as receiver
    from atom.model_engine.collective_rpc import COLLECTIVE_RPC_METHODS
    from atom.rollout.rdma_weight_receiver import (
        _CMD_BUCKET,
        _CMD_END,
        _HEADER_WORDS,
        _decode_bucket,
    )


def test_an_orchestrator_can_reach_every_receiver_rpc():
    """collective_rpc runs only COLLECTIVE_RPC_METHODS, so a receiver RPC
    missing from it is refused before any worker sees the call."""
    public = {
        name
        for name in vars(receiver.RDMAWeightReceiverMixin)
        if not name.startswith("_")
    }
    assert public == {
        "init_rdma_weight_group",
        "receive_weights_rdma",
        "destroy_rdma_weight_group",
    }
    assert public <= COLLECTIVE_RPC_METHODS


def test_the_reload_transaction_steps_stay_off_the_public_path():
    """receive_weights_rdma runs them in order inside one call. Driven one at
    a time from outside, a commit could follow no apply at all."""
    steps = {
        "begin_weight_update",
        "apply_weight_bucket",
        "prepare_weight_commit",
        "finish_weight_commit",
        "abort_weight_update",
        "fence_weight_update",
    }
    assert not steps & COLLECTIVE_RPC_METHODS


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
                # The sender strips the "torch." prefix; getattr(torch, ...) on
                # the receiving side is what has to match.
                "dtype": str(tensor.dtype).removeprefix("torch."),
                "offset": offset,
                "nbytes": raw.numel(),
            }
        )
        blobs.append(raw)
        offset += raw.numel()

    payload = torch.cat(blobs) if blobs else torch.empty(0, dtype=torch.uint8)
    meta = torch.tensor(
        list(json.dumps(entries, separators=(",", ":")).encode("utf-8")),
        dtype=torch.uint8,
    )
    return meta, payload


def test_a_bucket_round_trips_through_the_real_encoding():
    import torch

    a = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.bfloat16)
    b = torch.tensor([5.0, 6.0, 7.0], dtype=torch.float32)
    meta, payload = _encode([("layer.a", a), ("layer.b", b)])

    decoded = _decode_bucket(meta, payload)

    assert [name for name, _ in decoded] == ["layer.a", "layer.b"]
    got_a, got_b = decoded[0][1], decoded[1][1]
    assert got_a.dtype == torch.bfloat16 and tuple(got_a.shape) == (2, 2)
    assert got_b.dtype == torch.float32 and tuple(got_b.shape) == (3,)
    assert torch.equal(got_a, a)
    assert torch.equal(got_b, b)


def test_metadata_is_decoded_without_a_list_entry_per_byte():
    """tolist() built a list slot per metadata byte, eight bytes each, before
    JSON saw any of it: half a gigabyte of host memory at the 64 MiB ceiling."""
    import tracemalloc

    import torch

    meta, payload = _encode([("w" * (1 << 20), torch.zeros(1))])
    _decode_bucket(meta, payload)  # any lazy import, outside the measurement
    tracemalloc.start()
    try:
        _decode_bucket(meta, payload)
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    # The bytes, their text and the name decoded from it: a few copies of the
    # frame, where the list alone was eight.
    assert peak < 6 * meta.numel()


def test_the_decoded_tensors_are_views_not_copies():
    """A 61 GB transfer cannot afford to double its peak footprint."""
    import torch

    a = torch.tensor([1.0, 2.0], dtype=torch.bfloat16)
    meta, payload = _encode([("w", a)])
    ((_, view),) = _decode_bucket(meta, payload)
    assert view.data_ptr() == payload.data_ptr()


def test_a_tensor_packed_off_its_alignment_is_decoded_too():
    """The sender packs tensors back to back, so an FP32 tensor after three
    BF16 values starts six bytes in, which view() cannot reinterpret: a valid
    bucket failed. That tensor is copied; the rest are still views."""
    import torch

    a = torch.tensor([1.0, 2.0, 3.0], dtype=torch.bfloat16)
    b = torch.tensor([4.5, 5.5], dtype=torch.float32)
    meta, payload = _encode([("a", a), ("b", b)])

    (_, got_a), (_, got_b) = _decode_bucket(meta, payload)

    assert torch.equal(got_a, a) and torch.equal(got_b, b)
    assert got_a.data_ptr() == payload.data_ptr()


def test_an_out_of_bounds_offset_is_rejected_not_truncated():
    """Silently clamping would hand a neighbouring tensor's bytes to the model."""
    import torch

    meta, payload = _encode([("w", torch.tensor([1.0], dtype=torch.float32))])
    entries = json.loads(bytes(meta.tolist()).decode())
    entries[0]["nbytes"] = 4096  # past the end
    bad = torch.tensor(list(json.dumps(entries).encode()), dtype=torch.uint8)
    with pytest.raises(RuntimeError, match="out of bounds"):
        _decode_bucket(bad, payload)


@pytest.mark.parametrize(
    ("mutate", "expect"),
    [
        (lambda e: e.update({"shape": [99]}), "size mismatch"),
        (lambda e: e.pop("name"), "invalid RDMA weight metadata entry"),
        (
            lambda e: e.update({"dtype": "not_a_dtype"}),
            "invalid RDMA weight metadata entry",
        ),
        (lambda e: e.update({"nbytes": 0}), "out of bounds"),
        (lambda e: e.update({"offset": -1}), "invalid RDMA weight metadata entry"),
        # Each of these int() coerced into a different layout that still fit
        # the bytes, or into one view() reshapes to fit.
        (lambda e: e.update({"shape": [2.5]}), "invalid RDMA weight metadata entry"),
        (lambda e: e.update({"shape": [-1]}), "invalid RDMA weight metadata entry"),
        (lambda e: e.update({"shape": [True]}), "invalid RDMA weight metadata entry"),
        (lambda e: e.update({"shape": "2"}), "invalid RDMA weight metadata entry"),
        (lambda e: e.update({"offset": 0.5}), "invalid RDMA weight metadata entry"),
        (lambda e: e.update({"nbytes": 8.0}), "invalid RDMA weight metadata entry"),
        (lambda e: e.update({"name": 7}), "invalid RDMA weight metadata entry"),
        (lambda e: e.update({"dtype": "tensor"}), "invalid RDMA weight metadata entry"),
    ],
)
def test_corrupt_metadata_is_rejected(mutate, expect):
    import torch

    meta, payload = _encode([("w", torch.tensor([1.0, 2.0], dtype=torch.float32))])
    entries = json.loads(bytes(meta.tolist()).decode())
    mutate(entries[0])
    bad = torch.tensor(list(json.dumps(entries).encode()), dtype=torch.uint8)
    with pytest.raises(RuntimeError, match=expect):
        _decode_bucket(bad, payload)


def test_an_empty_metadata_list_is_rejected():
    import torch

    bad = torch.tensor(list(b"[]"), dtype=torch.uint8)
    with pytest.raises(RuntimeError, match="non-empty list"):
        _decode_bucket(bad, torch.empty(0, dtype=torch.uint8))


def _metadata(entries):
    import torch

    return torch.tensor(list(json.dumps(entries).encode()), dtype=torch.uint8)


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
    with pytest.raises(RuntimeError, match="repeats weight names"):
        _decode_bucket(meta, payload)


def test_overlapping_ranges_are_rejected():
    """The same bytes handed to two parameters, with coverage none the wiser."""
    import torch

    meta, payload = _encode(
        [
            ("a", torch.tensor([1.0, 2.0], dtype=torch.float32)),
            ("b", torch.tensor([3.0], dtype=torch.float32)),
        ]
    )
    entries = json.loads(bytes(meta.tolist()).decode())
    entries[1]["offset"] = 4  # inside a's [0, 8)
    with pytest.raises(RuntimeError, match="overlap"):
        _decode_bucket(_metadata(entries), payload)


def test_a_gap_between_ranges_is_not_an_overlap():
    """Only overlap is refused; padding between entries is not corruption."""
    import torch

    payload = torch.zeros(12, dtype=torch.uint8)
    payload[0:4] = torch.tensor([1.0], dtype=torch.float32).view(torch.uint8)
    payload[8:12] = torch.tensor([2.0], dtype=torch.float32).view(torch.uint8)
    entries = [
        {"name": n, "shape": [1], "dtype": "float32", "offset": o, "nbytes": 4}
        for n, o in (("a", 0), ("b", 8))
    ]
    decoded = dict(_decode_bucket(_metadata(entries), payload))
    assert decoded["a"].item() == 1.0 and decoded["b"].item() == 2.0


# ── the stream: one rank's failure must not strand the others ──────────────


class _Trainer:
    """Plays the sending side: what each broadcast delivers, in order."""

    def __init__(self, frames):
        self.frames = list(frames)

    def broadcast(self, tensor, src=0, group=None):
        assert src == 0
        frame = self.frames.pop(0)
        # A receiver out of step posts a broadcast of the wrong size, which
        # RCCL does not check; here it fails loudly instead.
        assert tensor.numel() == frame.numel(), "receiver fell out of step"
        tensor.copy_(frame)


def _frames(buckets, version=1):
    import torch

    frames = []
    for meta, payload in buckets:
        header = [_CMD_BUCKET, meta.numel(), payload.numel(), version]
        frames += [torch.tensor(header, dtype=torch.int64), meta, payload]
    frames.append(torch.tensor([_CMD_END, 0, 0, version], dtype=torch.int64))
    return frames


class _Runner:
    """The transaction surface ``receive_weight_stream`` drives."""

    def __init__(self, fail_on_bucket=None, refuse_begin=False):
        self.events = []
        self.fail_on_bucket = fail_on_bucket
        self.refuse_begin = refuse_begin
        self.applied = 0

    def begin_weight_update(self, version):
        self.events.append("begin")
        if self.refuse_begin:
            raise RuntimeError(f"weight update version must increase, got {version}")

    def apply_weight_bucket(self, weights, payload_bytes=0):
        self.applied += 1
        self.events.append(f"apply{self.applied}")
        if self.applied == self.fail_on_bucket:
            raise RuntimeError(f"bucket {self.applied} rejected")

    def prepare_weight_commit(self, version):
        self.events.append("prepare")

    def finish_weight_commit(self, version):
        self.events.append("commit")
        return {"loaded_internal": 1}

    def abort_weight_update(self, version, error):
        self.events.append(f"abort: {error}")

    def fence_weight_update(self, error):
        self.events.append(f"fence: {error}")


def _three_buckets():
    import torch

    return [
        _encode([(f"w{i}", torch.tensor([float(i)], dtype=torch.float32))])
        for i in range(3)
    ]


def _receive(
    monkeypatch,
    trainer,
    runner,
    synchronize=lambda *a, **k: None,
    version=1,
    agree=receiver._alone,
    leave=None,
):
    import torch

    monkeypatch.setattr(receiver, "dist", SimpleNamespace(broadcast=trainer.broadcast))
    monkeypatch.setattr(torch.cuda, "synchronize", synchronize)
    return receiver.receive_weight_stream(
        None,
        runner,
        device=torch.device("cpu"),
        expected_version=version,
        agree=agree,
        leave=leave,
    )


def _peers_decide(all_ready, any_began, *, take_frames=None):
    """Peers that take every frame, or the first *take_frames* of them, and
    then decide as given; records the ballots this rank cast."""

    def agree(ballot, began):
        agree.ballots.append((ballot, began))
        frames = sum(cast == receiver._RECEIVING for cast, _ in agree.ballots)
        if ballot == receiver._RECEIVING and (
            take_frames is None or frames <= take_frames
        ):
            return receiver._RECEIVING, any_began
        if ballot == receiver._PREPARED and all_ready:
            return receiver._PREPARED, any_began
        return receiver._BROKEN_OFF, any_began

    agree.ballots = []
    return agree


def test_a_clean_stream_commits(monkeypatch):
    trainer = _Trainer(_frames(_three_buckets()))
    runner = _Runner()
    stats = _receive(monkeypatch, trainer, runner)
    assert runner.events == ["begin", "apply1", "apply2", "apply3", "prepare", "commit"]
    assert stats["buckets"] == 3.0
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
            # A bucket's payload is allocated just after its own metadata,
            # which may still be alive; nothing older may be. By storage, not
            # by tensor: the decoded views keep the bytes alive without
            # keeping the tensor they were sliced from.
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

    with pytest.raises(RuntimeError, match="bucket 1 rejected"):
        _receive(monkeypatch, trainer, runner)

    assert trainer.frames == [], "every remaining broadcast must still be joined"
    assert runner.events == ["begin", "apply1", "abort: bucket 1 rejected"]


def test_a_version_mismatch_drains_before_raising(monkeypatch):
    trainer = _Trainer(_frames(_three_buckets(), version=2))
    runner = _Runner()

    with pytest.raises(RuntimeError, match="expected 1, got 2"):
        _receive(monkeypatch, trainer, runner)

    assert trainer.frames == []
    assert runner.events[:1] == ["begin"] and "apply1" not in runner.events


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
    leaves no step to keep. Tried anyway, the allocation raised past the
    stream's own handling, and the bad header was never named."""
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
    assert runner.events[-1].startswith("abort")


def test_an_allocation_that_fails_anyway_is_out_of_step_too(monkeypatch):
    """Within the bounds, memory can still run out -- fragmentation, another
    allocation in between. That is the same position: the frame cannot be
    received, so the stream cannot be followed."""
    import torch

    trainer = _Trainer(_frames(_three_buckets()))
    runner = _Runner()
    empty = torch.empty

    def exhausted(*args, **kwargs):
        if kwargs.get("dtype") is torch.uint8:
            raise torch.OutOfMemoryError("out of memory")
        return empty(*args, **kwargs)

    monkeypatch.setattr(torch, "empty", exhausted)
    with pytest.raises(receiver.RDMAStreamOutOfStep, match="out of memory"):
        _receive(monkeypatch, trainer, runner)
    assert runner.events[-1].startswith("abort")


def test_an_unknown_command_is_received_before_it_is_refused(monkeypatch):
    """Like any frame but the end marker, it carries the sizes of the two
    broadcasts that follow it. Refused at the header, this rank left the
    trainer and every other rank waiting in those."""
    import torch

    unknown = [
        torch.tensor([9, 3, 5, 1], dtype=torch.int64),
        torch.zeros(3, dtype=torch.uint8),
        torch.zeros(5, dtype=torch.uint8),
    ]
    trainer = _Trainer([*unknown, *_frames(_three_buckets())])
    runner = _Runner()

    with pytest.raises(RuntimeError, match="command=9"):
        _receive(monkeypatch, trainer, runner)

    assert trainer.frames == [], "every remaining broadcast must still be joined"
    assert not [e for e in runner.events if e.startswith("apply")]
    assert runner.events[-1].startswith("abort")


def test_a_refused_begin_still_receives_the_stream(monkeypatch):
    """begin refused a replayed version before the loop was entered, so this
    rank left while the trainer and every other rank sat in the next
    broadcast."""
    trainer = _Trainer(_frames(_three_buckets()))
    runner = _Runner(refuse_begin=True)

    with pytest.raises(RuntimeError, match="must increase"):
        _receive(monkeypatch, trainer, runner)

    assert trainer.frames == [], "every remaining broadcast must still be joined"
    assert runner.events == ["begin"], "nothing applied, and nothing aborted"


def _transacting_runner():
    """The real transaction and RDMA entry points, over a model of one weight."""
    import torch
    from torch import nn

    from atom.rollout.weight_updater import WeightUpdaterMixin

    class _Real(WeightUpdaterMixin, receiver.RDMAWeightReceiverMixin):
        device = torch.device("cpu")
        label = "test"
        rank = 0
        world_size = 1

        def __init__(self):
            self.model = nn.Module()
            self.model.w = nn.Parameter(torch.zeros(1), requires_grad=False)

        def clear_kv_cache(self):
            pass

        def _sync_target_model(self):
            return self.model

    return _Real()


def test_a_refused_stream_leaves_a_reload_another_caller_has_open(monkeypatch):
    """The abort on the way out ends whatever reload is open, and the one that
    refused this stream was not this stream's."""
    import torch

    runner = _transacting_runner()
    runner.begin_weight_update(1)
    trainer = _Trainer(_frames(_three_buckets(), version=2))

    with pytest.raises(RuntimeError, match="already in progress"):
        _receive(monkeypatch, trainer, runner, version=2)

    assert trainer.frames == []
    assert runner.get_weight_update_status()["in_progress"] == 1
    runner.apply_weight_bucket([("w", torch.ones(1))])
    runner.commit_weight_update(1)
    runner.assert_weight_update_ready()


def test_a_replayed_stream_leaves_the_committed_weights_serving(monkeypatch):
    """It wrote nothing, so there is nothing to fence; fenced anyway, serving
    stopped until a reload the weights did not need."""
    import torch

    runner = _transacting_runner()
    runner.begin_weight_update(1)
    runner.apply_weight_bucket([("w", torch.ones(1))])
    runner.commit_weight_update(1)
    trainer = _Trainer(_frames(_three_buckets(), version=1))

    with pytest.raises(RuntimeError, match="must increase"):
        _receive(monkeypatch, trainer, runner, version=1)

    runner.assert_weight_update_ready()
    assert runner.get_weight_update_status()["last_committed"] == 1


def test_a_rank_out_of_step_leaves_the_group(monkeypatch):
    """The sender is mid-broadcast in a frame this rank never posted, so a later
    stream on the same group would be read out of step from its first header."""
    import torch

    group = object()
    destroyed = []
    trainer = _Trainer([torch.tensor([_CMD_BUCKET, -1, 8, 1])])
    monkeypatch.setattr(
        receiver,
        "dist",
        SimpleNamespace(
            broadcast=trainer.broadcast, destroy_process_group=destroyed.append
        ),
    )
    runner = _transacting_runner()
    runner._rdma_weight_groups = {"g": group}

    with pytest.raises(receiver.RDMAStreamOutOfStep):
        runner.receive_weights_rdma("g", 1)

    assert destroyed == [group]
    with pytest.raises(RuntimeError, match="not initialized"):
        runner.receive_weights_rdma("g", 2)


def _rdma_runner(monkeypatch, trainer):
    """The real entry points over one weight, its group joined, fed by *trainer*."""
    import torch

    monkeypatch.setattr(
        receiver,
        "dist",
        SimpleNamespace(
            broadcast=trainer.broadcast, destroy_process_group=lambda group: None
        ),
    )
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *a, **k: None)
    runner = _transacting_runner()
    runner._rdma_weight_groups = {"g": object()}
    return runner


def test_the_vllm_callers_arguments_are_taken(monkeypatch):
    """LumenRL drives this RPC with the vLLM worker's arguments, and
    prequantized_fp8 was not among this one's: the call raised TypeError on
    every rank before it read a header."""
    import torch

    trainer = _Trainer(_frames([_encode([("w", torch.tensor([5.0]))])]))
    runner = _rdma_runner(monkeypatch, trainer)

    stats = runner.receive_weights_rdma(
        group_name="g", version=1, verify_full_load=True, prequantized_fp8=False
    )

    assert stats["buckets"] == 1.0
    assert runner.model.w.item() == 5.0


def test_a_prequantized_stream_is_received_then_refused(monkeypatch):
    """Nothing here maps its weight_scale_inv onto the resident scales, so it
    is refused; received first, or the trainer would wait in its broadcast,
    and with nothing written, so the runner serves on."""
    import torch

    trainer = _Trainer(_frames([_encode([("w", torch.tensor([5.0]))])]))
    runner = _rdma_runner(monkeypatch, trainer)

    with pytest.raises(NotImplementedError, match="prequantized FP8"):
        runner.receive_weights_rdma(
            group_name="g", version=1, verify_full_load=True, prequantized_fp8=True
        )

    assert trainer.frames == [], "the stream was not received"
    assert runner.model.w.item() == 0.0
    runner.assert_weight_update_ready()
    assert "g" in runner._rdma_weight_groups, "a group still in step was dropped"


def test_a_peer_whose_broadcast_failed_leaves_the_group_too(monkeypatch):
    """Only the rank whose own frame could not be received tore its end down.
    Its peers failed in the broadcast it never joined, kept the group, and the
    next stream rode it out of step."""
    group = object()
    destroyed = []

    def peer_left(tensor, src=0, group=None):
        raise RuntimeError("peer left the broadcast")

    monkeypatch.setattr(
        receiver,
        "dist",
        SimpleNamespace(broadcast=peer_left, destroy_process_group=destroyed.append),
    )
    runner = _transacting_runner()
    runner._rdma_weight_groups = {"g": group}

    with pytest.raises(receiver.RDMAStreamOutOfStep, match="peer left"):
        runner.receive_weights_rdma("g", 1)

    assert destroyed == [group]


def test_verify_full_load_false_does_not_waive_the_check(monkeypatch, caplog):
    """Accepted because the caller drives the vLLM worker with the same
    arguments. Honoured, it committed part of the model and served the rest
    from another version."""
    import logging

    import torch

    trainer = _Trainer(_frames(_three_buckets()))  # nothing this model has
    monkeypatch.setattr(receiver, "dist", SimpleNamespace(broadcast=trainer.broadcast))
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *a, **k: None)
    runner = _transacting_runner()
    runner._rdma_weight_groups = {"g": object()}

    with (
        caplog.at_level(logging.WARNING, logger="atom"),
        pytest.raises(RuntimeError, match="incomplete weight reload"),
    ):
        runner.receive_weights_rdma("g", 1, verify_full_load=False)

    assert "verify_full_load=False ignored" in caplog.text
    with pytest.raises(RuntimeError, match="fenced"):
        runner.assert_weight_update_ready()


def test_an_end_marker_carrying_sizes_is_not_a_clean_finish(monkeypatch):
    """The contract is [END, 0, 0, version]; one with sizes was taken as a
    successful end and committed."""
    import torch

    frames = _frames(_three_buckets())
    frames[-1] = torch.tensor([_CMD_END, 5, 7, 1], dtype=torch.int64)
    trainer = _Trainer(frames)
    runner = _Runner()

    with pytest.raises(RuntimeError, match="end marker"):
        _receive(monkeypatch, trainer, runner)

    assert "commit" not in runner.events
    assert runner.events[-1].startswith("abort")


def test_a_device_fault_keeps_the_stream_from_being_committed(monkeypatch):
    """The copies are asynchronous, so a fault in them surfaces at the
    synchronize -- which came after commit, leaving a version declared good
    over writes that never landed. It now comes before the vote."""

    def fault(*args, **kwargs):
        raise RuntimeError("device fault")

    trainer = _Trainer(_frames(_three_buckets()))
    runner = _Runner()
    agree = _peers_decide(False, True)

    with pytest.raises(RuntimeError, match="device fault"):
        _receive(monkeypatch, trainer, runner, synchronize=fault, agree=agree)

    assert runner.events[-2:] == ["prepare", "abort: device fault"]
    assert agree.ballots[-1] == (receiver._BROKEN_OFF, True)


# ── the ranks taking one stream decide on it together ──────────────────────


def test_a_rank_that_could_commit_does_not_when_a_peer_cannot(monkeypatch):
    """One rank's version alone serves the engine mixed: its peers' shards of
    the same layers would still be the old version."""
    trainer = _Trainer(_frames(_three_buckets()))
    runner = _Runner()
    agree = _peers_decide(False, True)

    with pytest.raises(RuntimeError, match="another rank taking it could not"):
        _receive(monkeypatch, trainer, runner, agree=agree)

    assert agree.ballots == [(receiver._RECEIVING, True)] * 3 + [
        (receiver._PREPARED, True)
    ]
    assert "commit" not in runner.events
    assert runner.events[-1].startswith("abort")


def test_a_rank_that_refused_what_its_peers_took_is_fenced(monkeypatch):
    """Refused, it kept its old weights and went on serving them, beside peers
    that had taken the new version -- and a rank still serving would also wait
    in its next forward's collectives for peers that refuse to."""
    trainer = _Trainer(_frames(_three_buckets()))
    runner = _Runner(refuse_begin=True)
    agree = _peers_decide(False, True)

    with pytest.raises(RuntimeError, match="must increase"):
        _receive(monkeypatch, trainer, runner, agree=agree)

    assert agree.ballots[-1] == (receiver._BROKEN_OFF, False)
    assert runner.events[0] == "begin" and runner.events[-1].startswith("fence")


def test_a_stream_every_rank_refused_fences_none_of_them(monkeypatch):
    """Nothing was written anywhere, so every rank still holds one version."""
    trainer = _Trainer(_frames(_three_buckets()))
    runner = _Runner(refuse_begin=True)

    with pytest.raises(RuntimeError, match="must increase"):
        _receive(monkeypatch, trainer, runner, agree=_peers_decide(False, False))

    assert runner.events == ["begin"]


def test_a_rank_that_breaks_off_still_votes(monkeypatch):
    """A peer waits in the vote on the frame; without this rank's ballot it
    waited until its group timed out."""
    import torch

    trainer = _Trainer([torch.tensor([_CMD_BUCKET, -1, 8, 1])])
    runner = _Runner()
    agree = _peers_decide(False, True)

    with pytest.raises(receiver.RDMAStreamOutOfStep):
        _receive(monkeypatch, trainer, runner, agree=agree)

    assert agree.ballots == [(receiver._BROKEN_OFF, True)]


def test_a_frame_a_peer_cannot_receive_is_broken_off_here_too(monkeypatch):
    """Free memory differs between ranks. One that could not allocate for a
    frame broke off alone, and its peers went on into the frame's broadcasts,
    held there for it while it waited for them in the vote."""
    trainer = _Trainer(_frames(_three_buckets()))
    runner = _Runner()
    agree = _peers_decide(False, True, take_frames=1)
    left = []

    with pytest.raises(receiver.RDMAStreamOutOfStep, match="not by every rank"):
        _receive(
            monkeypatch, trainer, runner, agree=agree, leave=lambda: left.append(1)
        )

    assert agree.ballots == [(receiver._RECEIVING, True)] * 2
    assert len(trainer.frames) == 6, "the frame's broadcasts were entered"
    assert runner.events[:2] == ["begin", "apply1"]
    assert runner.events[-1].startswith("abort")
    # Every rank left at that vote, so none is in a broadcast to release.
    assert left == []


def test_a_rank_whose_broadcast_failed_leaves_the_group_before_it_votes(
    monkeypatch,
):
    """A peer may still be in the broadcast that failed here, held there until
    this rank's end of the group is gone. Voting first, this rank waited for
    that peer until the group timed out, with nothing left to tear it down."""
    trainer = _Trainer(_frames(_three_buckets()))
    calls, order = [], []

    def failing(tensor, src=0, group=None):
        calls.append(1)
        if len(calls) == 5:  # the second frame's metadata
            raise RuntimeError("broadcast timed out")
        trainer.broadcast(tensor, src=src, group=group)

    peers = _peers_decide(False, True)

    def agree(ballot, began):
        order.append("vote")
        return peers(ballot, began)

    with pytest.raises(receiver.RDMAStreamOutOfStep, match="broadcast timed out"):
        _receive(
            monkeypatch,
            SimpleNamespace(broadcast=failing),
            _Runner(),
            agree=agree,
            leave=lambda: order.append("leave"),
        )

    assert order == ["vote", "vote", "leave", "vote"]
    assert peers.ballots[-1] == (receiver._BROKEN_OFF, True)


def test_a_vote_that_failed_leaves_the_group_to_be_joined_again(monkeypatch):
    """A peer gone during the vote at the end marker. Raised as it was, the
    error kept receive_weights_rdma from tearing the group down, and the next
    stream rode a group that still listed the dead rank."""
    import torch

    group = object()
    destroyed = []
    trainer = _Trainer(_frames([_encode([("w", torch.tensor([5.0]))])]))
    monkeypatch.setattr(
        receiver,
        "dist",
        SimpleNamespace(
            broadcast=trainer.broadcast, destroy_process_group=destroyed.append
        ),
    )
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *a, **k: None)
    runner = _transacting_runner()
    runner._rdma_weight_groups = {"g": group}
    ballots = []

    def vote(ballot, began):
        ballots.append(ballot)
        if ballot != receiver._RECEIVING:
            raise RuntimeError("peer gone mid-vote")
        return ballot, began

    runner._vote_with_engine_peers = vote

    with pytest.raises(receiver.RDMAStreamOutOfStep, match="could not vote"):
        runner.receive_weights_rdma("g", 1)

    # Not voted on again over a group that had just failed to vote.
    assert ballots == [receiver._RECEIVING, receiver._PREPARED]
    assert destroyed == [group]
    with pytest.raises(RuntimeError, match="not initialized"):
        runner.receive_weights_rdma("g", 2)
    with pytest.raises(RuntimeError, match="fenced"):
        runner.assert_weight_update_ready()


def test_a_broadcast_that_fails_puts_the_rank_out_of_step(monkeypatch):
    """Typically a peer that left: this rank's own frames are fine, but the
    group is no longer in step, so it is out of step all the same."""
    trainer = _Trainer(_frames(_three_buckets()))
    calls = []

    def failing(tensor, src=0, group=None):
        calls.append(1)
        if len(calls) == 5:
            raise RuntimeError("broadcast timed out")
        trainer.broadcast(tensor, src=src, group=group)

    trainer_failing = SimpleNamespace(broadcast=failing)
    runner = _Runner()

    with pytest.raises(receiver.RDMAStreamOutOfStep, match="broadcast timed out"):
        _receive(monkeypatch, trainer_failing, runner)
    assert runner.events[-1].startswith("abort")


class _Worker(receiver.RDMAWeightReceiverMixin):
    """One engine worker, as init_rdma_weight_group sees it."""

    def __init__(self, rank, dp_local, *, tp, tp_world, pcp, dp=1):
        self.rank = rank
        self.world_size = tp  # the logical TP width, as on ModelRunner
        self.label = f"worker{rank}"
        self.config = SimpleNamespace(
            tensor_parallel_size=tp,
            tp_world_size=tp_world,
            prefill_context_parallel_size=pcp,
            parallel_config=SimpleNamespace(
                data_parallel_rank_local=dp_local, data_parallel_size=dp
            ),
        )


def _joins(monkeypatch):
    """The ranks init_rdma_weight_group joins as, with no rendezvous behind."""
    import atom.utils.independent_process_group as ipg

    joined = []
    monkeypatch.setattr(
        ipg,
        "init_independent_process_group",
        lambda **kw: joined.append(kw["rank"]) or object(),
    )
    return joined


def _mla_attention(monkeypatch):
    """MLAAttention as the receiver finds it, with no aiter to import it from."""
    from torch import nn

    mla = ModuleType("atom.model_ops.attention_mla")
    mla.MLAAttention = type("MLAAttention", (nn.Module,), {})
    monkeypatch.setitem(sys.modules, mla.__name__, mla)
    return mla.MLAAttention


def test_a_dp_engine_refuses_the_weight_group_before_joining(monkeypatch):
    """One group spanned every DP engine, and each entered its broadcast and
    decided the stream on its own: one could block in it while another waited
    in the DP all-reduce, or commit a version another had rejected."""
    joined = _joins(monkeypatch)
    for dp_local in range(2):
        worker = _Worker(0, dp_local, tp=1, tp_world=1, pcp=1, dp=2)
        with pytest.raises(RuntimeError, match="data_parallel_size=2"):
            worker.init_rdma_weight_group(
                "127.0.0.1", 29500, base_rank=1, world_size=3, group_name="g"
            )
    assert joined == []


def test_a_dp_engine_is_refused_though_its_world_was_isolated_to_one(monkeypatch):
    """Once a DP engine's workers have a world of their own, their config says
    data_parallel_size=1, and the refusal, reading that, let every DP engine
    of the deployment join one weight group."""
    joined = _joins(monkeypatch)
    worker = _Worker(0, 1, tp=1, tp_world=1, pcp=1, dp=1)
    worker._deployment_data_parallel_size = 2

    with pytest.raises(RuntimeError, match="data_parallel_size=2"):
        worker.init_rdma_weight_group(
            "127.0.0.1", 29500, base_rank=1, world_size=2, group_name="g"
        )
    assert joined == []


def test_the_rlhf_runner_keeps_the_deployments_dp_size_before_hiding_it():
    """The runner that builds those worlds is what rewrites the config, so it
    has to keep the deployment's size first. Read, not run: importing the
    runner needs aiter on a ROCm device."""
    import ast

    src = (
        Path(__file__).resolve().parent.parent
        / "atom"
        / "rollout"
        / "model_runner_ext.py"
    ).read_text()
    runner = next(
        n
        for n in ast.parse(src).body
        if isinstance(n, ast.ClassDef) and n.name == "RLHFModelRunner"
    )
    setup = next(
        n
        for n in runner.body
        if isinstance(n, ast.FunctionDef) and n.name == "_setup_device_and_distributed"
    )
    writes = sorted(
        (node.lineno, ast.unparse(target), ast.unparse(node.value))
        for node in ast.walk(setup)
        if isinstance(node, ast.Assign)
        for target in node.targets
        if ast.unparse(target)
        in (
            "self._deployment_data_parallel_size",
            "config.parallel_config.data_parallel_size",
        )
    )

    assert [(target, value) for _, target, value in writes] == [
        (
            "self._deployment_data_parallel_size",
            "config.parallel_config.data_parallel_size",
        ),
        ("config.parallel_config.data_parallel_size", "1"),
    ]


def test_every_serving_forward_goes_through_the_fence():
    """The fence was checked in RLHFModelRunner.forward alone, and a
    disaggregated EngineCore serves through prefill_forward instead. It is
    checked in run_model now, which every serving forward runs the model
    through and graph capture does not. Read, not run, as above."""
    import ast

    root = Path(__file__).resolve().parent.parent / "atom"

    def method(path, cls_name, name):
        tree = ast.parse((root / path).read_text())
        cls = next(
            n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls_name
        )
        return next(
            n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == name
        )

    def calls(function):
        return {
            ast.unparse(n.func) for n in ast.walk(function) if isinstance(n, ast.Call)
        }

    gate = method("rollout/model_runner_ext.py", "RLHFModelRunner", "run_model")
    assert ast.unparse(gate.body[0]) == "self.assert_weight_update_ready()"
    runner = "model_engine/model_runner.py"
    for cls_name, name in [
        ("ModelRunner", "forward"),
        ("RapidServeModelRunner", "forward"),
        ("RapidServeModelRunner", "prefill_forward"),
    ]:
        assert "self.run_model" in calls(method(runner, cls_name, name)), name
    assert "self.run_model" not in calls(
        method(runner, "ModelRunner", "capture_cudagraph")
    )


def test_an_mla_model_refuses_the_weight_group_before_joining(monkeypatch):
    """MLA decodes through W_K and W_V, derived from kv_b_proj once at load.
    A reload rewrote kv_b_proj and left them, and the engine served old and
    new weights together."""
    from torch import nn

    joined = _joins(monkeypatch)
    worker = _Worker(0, 0, tp=1, tp_world=1, pcp=1)
    worker.model = nn.Sequential(nn.Linear(2, 2), _mla_attention(monkeypatch)())

    with pytest.raises(RuntimeError, match="through MLA"):
        worker.init_rdma_weight_group(
            "127.0.0.1", 29500, base_rank=1, world_size=2, group_name="g"
        )
    assert joined == []

    worker.model = nn.Sequential(nn.Linear(2, 2))
    assert worker.init_rdma_weight_group(
        "127.0.0.1", 29500, base_rank=1, world_size=2, group_name="g"
    )
    assert joined == [1]


def _attention(*, use_mla=None, kv_b_proj=False):
    """An attention module as a model defines one, with no MLAAttention in it."""
    from torch import nn

    module = type("DeepseekV2MLAAttention", (nn.Module,), {})()
    if use_mla is not None:
        module.use_mla = use_mla
    if kv_b_proj:
        module.kv_b_proj = nn.Linear(2, 2)
    return module


@pytest.mark.parametrize(
    "layer",
    [{"kv_b_proj": True}, {"use_mla": True}],
    ids=["its-own-kv_b_proj", "attention-built-with-use_mla"],
)
def test_an_mla_model_is_known_by_its_layers_not_one_class(layer):
    """Each model defines its own MLA attention module, and an
    Attention(use_mla=True) may be given an impl other than MLAAttention: the
    check looked for MLAAttention alone, and would have let such a model take
    a reload its derived W_K and W_V do not follow."""
    from torch import nn

    worker = _Worker(0, 0, tp=1, tp_world=1, pcp=1)
    worker.model = nn.Sequential(_attention(**layer))
    assert "through MLA" in worker._rdma_reload_refusal()

    worker.model = nn.Sequential(_attention(use_mla=False))
    assert worker._rdma_reload_refusal() is None


def test_a_repeated_init_reuses_its_group_only_for_the_same_join(monkeypatch):
    """Retried under the same name, init returned success whatever it was
    asked: after a trainer restart on a new endpoint, a worker kept the old
    group while its peers joined the new one, and the next broadcast hung."""
    import atom.utils.independent_process_group as ipg

    joined = []
    monkeypatch.setattr(
        ipg,
        "init_independent_process_group",
        lambda **kw: joined.append(kw["init_method"]) or object(),
    )
    monkeypatch.setattr(receiver.dist, "destroy_process_group", lambda group: None)
    worker = _Worker(0, 0, tp=1, tp_world=1, pcp=1)
    join = {"base_rank": 1, "world_size": 2, "group_name": "g"}

    assert worker.init_rdma_weight_group("10.0.0.1", 29500, **join)
    assert worker.init_rdma_weight_group("10.0.0.1", 29500, **join)  # a retry
    with pytest.raises(RuntimeError, match="already joined"):
        worker.init_rdma_weight_group("10.0.0.2", 29500, **join)
    assert joined == ["tcp://10.0.0.1:29500"]

    worker.destroy_rdma_weight_group("g")
    assert worker.init_rdma_weight_group("10.0.0.2", 29500, **join)
    assert joined == ["tcp://10.0.0.1:29500", "tcp://10.0.0.2:29500"]


def _world(monkeypatch, size):
    """aiter's world group as the vote finds it, and the groups voted over."""
    world = SimpleNamespace(world_size=size, cpu_group=object())
    parallel_state = ModuleType("aiter.dist.parallel_state")
    parallel_state.get_world_group = lambda: world
    monkeypatch.setitem(sys.modules, "aiter.dist.parallel_state", parallel_state)
    voted = []

    def vote(group, ready, began):
        voted.append(group)
        return ready, began

    monkeypatch.setattr(receiver, "vote_on_stream", vote)
    return world.cpu_group, voted


@pytest.mark.parametrize(
    ("tp_world", "pcp"), [(1, 2), (2, 2), (2, 1)], ids=["pcp", "tp-and-pcp", "tp"]
)
def test_every_worker_of_the_engine_decides_the_stream(monkeypatch, tp_world, pcp):
    """The vote was over the TP group, which holds one prefill-context rank of
    the receivers: under PCP one TP group could commit while another failed,
    and the engine served two versions. At TP=1 nobody voted at all."""
    world, voted = _world(monkeypatch, tp_world * pcp)
    worker = _Worker(0, 0, tp=tp_world, tp_world=tp_world, pcp=pcp)

    assert worker._vote_with_engine_peers(True, True) == (True, True)
    assert voted == [world]


def test_a_single_worker_engine_decides_alone(monkeypatch):
    _, voted = _world(monkeypatch, 1)
    worker = _Worker(0, 0, tp=1, tp_world=1, pcp=1)

    assert worker._vote_with_engine_peers(False, True) == (False, True)
    assert voted == []


def test_a_world_that_is_not_the_engine_is_not_voted_in(monkeypatch):
    """With ranks beyond the stream's the vote would hang; without some of its
    receivers it would let the rest commit alone."""
    _, voted = _world(monkeypatch, 2)
    worker = _Worker(0, 0, tp=2, tp_world=2, pcp=2)

    with pytest.raises(RuntimeError, match=r"world has 2 rank.*engine 4 worker"):
        worker._vote_with_engine_peers(True, True)
    assert voted == []


def test_the_receiver_reads_the_header_in_the_documented_order(monkeypatch):
    import torch

    trainer = _Trainer([torch.tensor([1, 10, 20, 7], dtype=torch.int64)])
    monkeypatch.setattr(receiver, "dist", SimpleNamespace(broadcast=trainer.broadcast))
    assert receiver._recv_header(None, device=torch.device("cpu")) == (1, 10, 20, 7)


# ── discovery ──────────────────────────────────────────────────────────────


def test_the_lifecycle_the_real_mixins_define_is_advertised():
    """Methods are listed only where the runner has them, so a name the list
    spells differently from the mixin would drop out of the report silently."""
    from atom.rollout.capabilities import CapabilityProviderMixin
    from atom.rollout.weight_updater import WeightUpdaterMixin

    class _Composed(
        WeightUpdaterMixin, receiver.RDMAWeightReceiverMixin, CapabilityProviderMixin
    ):
        rank = 0

    report = _Composed().get_worker_capabilities()
    assert {
        "init_rdma_weight_group",
        "receive_weights_rdma",
        "destroy_rdma_weight_group",
        "get_weight_update_status",
    } <= set(report["methods"])
    assert "rdma_weight_receive" in report["features"]


@pytest.mark.parametrize("unsafe", ["dp", "isolated-dp", "mla"])
def test_the_feature_is_withheld_where_init_would_refuse(monkeypatch, unsafe):
    """The methods are still there; the feature is what says they work here,
    so an orchestrator can choose another path before a rendezvous that
    cannot complete."""
    from torch import nn

    from atom.rollout.capabilities import CapabilityProviderMixin

    class _Composed(_Worker, CapabilityProviderMixin):
        pass

    worker = _Composed(0, 0, tp=1, tp_world=1, pcp=1, dp=2 if unsafe == "dp" else 1)
    if unsafe == "isolated-dp":
        worker._deployment_data_parallel_size = 2
    if unsafe == "mla":
        worker.model = nn.Sequential(_mla_attention(monkeypatch)())
    report = worker.get_worker_capabilities()

    assert "receive_weights_rdma" in report["methods"]
    assert "rdma_weight_receive" not in report["features"]


# ── against the sender's own source ────────────────────────────────────────
#
# Read, not imported: LumenRL is a separate repo, and an import would couple
# the two. CI reads a snapshot -- everything above receive_weight_stream in
# lumenrl/engine/inference/rdma_weight_transfer.py, copied verbatim -- and with
# LUMENRL_ROOT naming a checkout, the snapshot is held to the sender itself.

_SENDER_SNAPSHOT = Path(__file__).parent / "fixtures" / "lumenrl_rdma_sender.txt"
_HEADER_ORDER = r"\[\s*command,\s*metadata_bytes,\s*payload_bytes,\s*version\s*,?\s*\]"


def _protocol(src: str) -> dict:
    """What the receiver depends on, as the sender's source states it."""
    facts = {}
    for const in ("_CMD_END", "_CMD_BUCKET", "_HEADER_WORDS"):
        m = re.search(rf"^{const}\s*=\s*(\d+)", src, re.MULTILINE)
        facts[const] = int(m[1]) if m else None
    facts["header order"] = re.search(_HEADER_ORDER, src) is not None
    entry = re.search(r"entries\.append\(\s*\{(.*?)\}\s*\)", src, re.DOTALL)
    facts["entry keys"] = sorted(re.findall(r'"(\w+)":', entry[1])) if entry else None
    return facts


def test_the_sender_and_receiver_agree_on_the_constants():
    facts = _protocol(_SENDER_SNAPSHOT.read_text())
    for const, value in (
        ("_CMD_END", _CMD_END),
        ("_CMD_BUCKET", _CMD_BUCKET),
        ("_HEADER_WORDS", _HEADER_WORDS),
    ):
        assert (
            facts[const] == value
        ), f"{const}: sender says {facts[const]}, receiver says {value}"


def test_the_sender_packs_the_header_in_the_order_it_is_read():
    """The part of the contract no bounds check can see: swapped sizes make the
    two sides post broadcasts of different lengths, which RCCL does not check."""
    assert _protocol(_SENDER_SNAPSHOT.read_text())["header order"], (
        "the sender no longer builds its header as "
        "[command, metadata_bytes, payload_bytes, version]"
    )


def test_the_sender_describes_each_tensor_by_what_the_decoder_reads():
    keys = _protocol(_SENDER_SNAPSHOT.read_text())["entry keys"]
    assert keys == ["dtype", "name", "nbytes", "offset", "shape"]


def test_the_snapshot_still_states_the_senders_protocol():
    """The snapshot stands in for the sender only while it says what the
    sender says. Opt-in, since ATOM's CI has no LumenRL checkout; named but
    missing fails rather than skips, as the check was asked for."""
    root = os.environ.get("LUMENRL_ROOT")
    if not root:
        pytest.skip("set LUMENRL_ROOT to a LumenRL checkout to check the snapshot")
    sender = Path(root, "lumenrl", "engine", "inference", "rdma_weight_transfer.py")
    assert sender.is_file(), f"LUMENRL_ROOT={root} has no {sender}"
    assert _protocol(sender.read_text()) == _protocol(_SENDER_SNAPSHOT.read_text()), (
        f"the sender's protocol changed: refresh {_SENDER_SNAPSHOT.name} from "
        f"{sender} and check the receiver against it"
    )
