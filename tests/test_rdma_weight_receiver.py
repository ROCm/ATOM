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

    def commit_weight_update(self, version, verify_full_load=True):
        self.events.append("commit")
        return {"loaded_internal": 1}

    def abort_weight_update(self, version, error):
        self.events.append(f"abort: {error}")


def _three_buckets():
    import torch

    return [
        _encode([(f"w{i}", torch.tensor([float(i)], dtype=torch.float32))])
        for i in range(3)
    ]


def _receive(monkeypatch, trainer, runner, synchronize=lambda *a, **k: None, version=1):
    import torch

    monkeypatch.setattr(receiver, "dist", SimpleNamespace(broadcast=trainer.broadcast))
    monkeypatch.setattr(torch.cuda, "synchronize", synchronize)
    return receiver.receive_weight_stream(
        None, runner, device=torch.device("cpu"), expected_version=version
    )


def test_a_clean_stream_commits(monkeypatch):
    trainer = _Trainer(_frames(_three_buckets()))
    runner = _Runner()
    stats = _receive(monkeypatch, trainer, runner)
    assert runner.events == ["begin", "apply1", "apply2", "apply3", "commit"]
    assert stats["buckets"] == 3.0
    assert trainer.frames == [], "the end marker was never read"


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


def test_a_header_whose_sizes_cannot_be_received_fails_at_once(monkeypatch):
    """No buffer of a negative size can be posted, so there is no step left
    to keep; failing at once is the only option."""
    import torch

    bad = torch.tensor([_CMD_BUCKET, -1, 8, 1], dtype=torch.int64)
    trainer = _Trainer([bad, *_frames(_three_buckets())])
    runner = _Runner()

    with pytest.raises(RuntimeError, match="invalid RDMA weight header"):
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
    """The real transaction, over a model of one weight."""
    import torch
    from torch import nn

    from atom.rollout.weight_updater import WeightUpdaterMixin

    class _Real(WeightUpdaterMixin):
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


def test_a_device_fault_found_after_commit_still_fences(monkeypatch):
    """The copies are asynchronous, so a fault in them surfaces at the
    synchronize -- which ran after the fenced region, leaving a committed
    version serving over writes that never landed."""

    def fault(*args, **kwargs):
        raise RuntimeError("device fault")

    trainer = _Trainer(_frames(_three_buckets()))
    runner = _Runner()

    with pytest.raises(RuntimeError, match="device fault"):
        _receive(monkeypatch, trainer, runner, synchronize=fault)

    assert runner.events[-2:] == ["commit", "abort: device fault"]


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


def _lumenrl_sender() -> str:
    """LumenRL's sender, read from the checkout ``LUMENRL_ROOT`` names.

    Opt-in: ATOM's CI has no LumenRL checkout, and a check that skips wherever
    one path is missing reads as coverage it never gives. The frozen constants
    are pinned above either way. Named but missing fails rather than skips --
    the check was asked for.
    """
    root = os.environ.get("LUMENRL_ROOT")
    if not root:
        pytest.skip("set LUMENRL_ROOT to a LumenRL checkout to check its sender")
    sender = Path(root, "lumenrl", "engine", "inference", "rdma_weight_transfer.py")
    assert sender.is_file(), f"LUMENRL_ROOT={root} has no {sender}"
    return sender.read_text()


def test_the_sender_and_receiver_agree_on_the_constants():
    """Read LumenRL's sender from source. It is a separate repo, so an import
    would couple the two -- but the numbers still have to match."""
    src = _lumenrl_sender()
    for const, value in (
        ("_CMD_END", _CMD_END),
        ("_CMD_BUCKET", _CMD_BUCKET),
        ("_HEADER_WORDS", _HEADER_WORDS),
    ):
        m = re.search(rf"^{const}\s*=\s*(\d+)", src, re.MULTILINE)
        assert m, f"{const} not found in the sender"
        assert (
            int(m.group(1)) == value
        ), f"{const}: sender says {m.group(1)}, receiver says {value}"


def test_the_sender_packs_the_header_in_the_order_it_is_read():
    """The part of the contract no bounds check can see: swapped sizes make the
    two sides post broadcasts of different lengths, which RCCL does not check."""
    order = r"\[\s*command,\s*metadata_bytes,\s*payload_bytes,\s*version\s*,?\s*\]"
    assert re.search(order, _lumenrl_sender()), (
        "the sender no longer builds its header as "
        "[command, metadata_bytes, payload_bytes, version]"
    )
