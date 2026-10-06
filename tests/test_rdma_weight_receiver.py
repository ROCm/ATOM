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
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from aiter_stub import stubbed_aiter

with stubbed_aiter():
    import atom.rollout.rdma_weight_receiver as receiver
    from atom.rollout.rdma_weight_receiver import (
        _CMD_BUCKET,
        _CMD_END,
        _HEADER_WORDS,
        _decode_bucket,
    )


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


def test_the_decoded_tensors_are_views_not_copies():
    """A 61 GB transfer cannot afford to double its peak footprint."""
    import torch

    a = torch.tensor([1.0, 2.0], dtype=torch.bfloat16)
    meta, payload = _encode([("w", a)])
    ((_, view),) = _decode_bucket(meta, payload)
    assert view.data_ptr() == payload.data_ptr()


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
        # torch's own view() rejects this before the explicit size check does,
        # which is fine -- what matters is that it is refused, not silently
        # reshaped into whatever fits.
        (lambda e: e.update({"shape": [99]}), "invalid for input of size"),
        (lambda e: e.pop("name"), "invalid RDMA weight metadata entry"),
        (
            lambda e: e.update({"dtype": "not_a_dtype"}),
            "invalid RDMA weight metadata entry",
        ),
        (lambda e: e.update({"nbytes": 0}), "out of bounds"),
        (lambda e: e.update({"offset": -1}), "out of bounds"),
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
):
    import torch

    monkeypatch.setattr(receiver, "dist", SimpleNamespace(broadcast=trainer.broadcast))
    monkeypatch.setattr(torch.cuda, "synchronize", synchronize)
    return receiver.receive_weight_stream(
        None, runner, device=torch.device("cpu"), expected_version=version, agree=agree
    )


def _peers_decide(all_ready, any_began):
    """Peers whose vote comes out as given; records the ballots this rank cast."""

    def agree(ready, began):
        agree.ballots.append((ready, began))
        return all_ready, any_began

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
    assert agree.ballots == [(False, True)]


# ── the ranks taking one stream decide on it together ──────────────────────


def test_a_rank_that_could_commit_does_not_when_a_peer_cannot(monkeypatch):
    """One rank's version alone serves the engine mixed: its peers' shards of
    the same layers would still be the old version."""
    trainer = _Trainer(_frames(_three_buckets()))
    runner = _Runner()
    agree = _peers_decide(False, True)

    with pytest.raises(RuntimeError, match="another rank taking it could not"):
        _receive(monkeypatch, trainer, runner, agree=agree)

    assert agree.ballots == [(True, True)]
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

    assert agree.ballots == [(False, False)]
    assert runner.events[0] == "begin" and runner.events[-1].startswith("fence")


def test_a_stream_every_rank_refused_fences_none_of_them(monkeypatch):
    """Nothing was written anywhere, so every rank still holds one version."""
    trainer = _Trainer(_frames(_three_buckets()))
    runner = _Runner(refuse_begin=True)

    with pytest.raises(RuntimeError, match="must increase"):
        _receive(monkeypatch, trainer, runner, agree=_peers_decide(False, False))

    assert runner.events == ["begin"]


def test_a_rank_that_breaks_off_still_votes(monkeypatch):
    """A peer that reached the end marker waits in the vote; without this
    rank's ballot it waited until its group timed out."""
    import torch

    trainer = _Trainer([torch.tensor([_CMD_BUCKET, -1, 8, 1])])
    runner = _Runner()
    agree = _peers_decide(False, True)

    with pytest.raises(receiver.RDMAStreamOutOfStep):
        _receive(monkeypatch, trainer, runner, agree=agree)

    assert agree.ballots == [(False, True)]


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

    def __init__(self, rank, dp_local, *, tp, tp_world, pcp):
        self.rank = rank
        self.world_size = tp  # the logical TP width, as on ModelRunner
        self.label = f"worker{rank}"
        self.config = SimpleNamespace(
            tensor_parallel_size=tp,
            tp_world_size=tp_world,
            prefill_context_parallel_size=pcp,
            parallel_config=SimpleNamespace(data_parallel_rank_local=dp_local),
        )


@pytest.mark.parametrize(
    ("tp", "tp_world", "pcp"),
    [(2, 2, 2), (4, 2, 1)],
    ids=["prefill-context-parallel", "simulated-tp"],
)
def test_every_worker_of_every_dp_engine_gets_its_own_rank(
    monkeypatch, tp, tp_world, pcp
):
    """The stride was the logical TP width, but an engine runs tp_world x pcp
    workers: under PCP one DP engine's workers took the next one's ranks, and
    under simulated TP ranks went unclaimed."""
    import atom.utils.independent_process_group as ipg

    joined = []
    monkeypatch.setattr(
        ipg,
        "init_independent_process_group",
        lambda **kw: joined.append(kw["rank"]) or object(),
    )
    dp, workers = 2, tp_world * pcp
    for dp_local in range(dp):
        for rank in range(workers):
            _Worker(
                rank, dp_local, tp=tp, tp_world=tp_world, pcp=pcp
            ).init_rdma_weight_group(
                "127.0.0.1",
                29500,
                base_rank=1,
                world_size=1 + dp * workers,
                group_name="g",
            )

    assert sorted(joined) == list(range(1, 1 + dp * workers))


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
