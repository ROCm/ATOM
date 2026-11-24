# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""Tests for the production bucketed weight-transfer transport."""

from __future__ import annotations

import asyncio
import io
import threading
import time

import pytest
import torch
import zmq
from atomrl.engine.inference import bucketed_weight_transfer as bwt
from atomrl.engine.inference.bucketed_weight_transfer import (
    BucketedWeightReceiver,
    BucketedWeightSender,
)


@pytest.fixture
def no_cuda(monkeypatch):
    monkeypatch.setenv("LUMEN_WEIGHT_TRANSFER_ZMQ_TIMEOUT_MS", "5000")
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(torch.cuda, "ipc_collect", lambda: None)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)


def _weights(seed: int) -> list[tuple[str, torch.Tensor]]:
    g = torch.Generator().manual_seed(seed)
    out = [
        ("model.large", torch.randn(200_000, generator=g)),
        ("model.small", torch.randn(200, 500, generator=g)),
        ("model.ids", torch.randint(0, 1000, (257,), generator=g)),
    ]
    out += [(f"model.layer{i}", torch.randn(60_000, generator=g)) for i in range(12)]
    return out


def _roundtrip(sender, receiver, weights, on_bucket=None) -> dict[str, torch.Tensor]:
    errors: list[BaseException] = []

    def _send() -> None:
        try:
            asyncio.run(sender.async_send_weights(weights))
        except (OSError, RuntimeError, ValueError) as exc:
            errors.append(exc)

    thread = threading.Thread(target=_send, daemon=True)
    thread.start()
    received: dict[str, torch.Tensor] = {}

    def _capture(bucket: list[tuple[str, torch.Tensor]]) -> None:
        # Receiver tensors view a staging buffer that the next bucket overwrites.
        received.update({name: tensor.clone() for name, tensor in bucket})
        if on_bucket is not None:
            on_bucket(bucket)

    receiver.receive_weights(on_bucket_received=_capture)
    thread.join(timeout=10)
    assert not thread.is_alive(), "weight sender did not finish"
    assert not errors, errors
    return received


def _assert_same(received, weights) -> None:
    assert set(received) == {name for name, _ in weights}
    for name, expected in weights:
        actual = received[name]
        assert actual.shape == expected.shape
        assert actual.dtype == expected.dtype
        assert torch.equal(actual, expected)


def test_shared_memory_roundtrip_across_multiple_buckets(tmp_path, no_cuda) -> None:
    """Preserve tensor names, values, shapes, and dtypes across bucket reuse."""
    handle = f"ipc://{tmp_path / 'weight-transfer.sock'}"
    weights = _weights(0)
    sender = BucketedWeightSender(handle, bucket_size_mb=2, use_shm=True)
    receiver = BucketedWeightReceiver(handle, device=torch.device("cpu"), use_shm=True)
    _assert_same(_roundtrip(sender, receiver, weights), weights)
    assert sender.buffer is None and sender.shm is None
    assert sender.stats["buckets"] > 2


@pytest.mark.parametrize("gc_collect", [True, False])
def test_gc_collect_is_optional_and_buffer_is_still_freed(
    tmp_path, no_cuda, monkeypatch, gc_collect
) -> None:
    calls: list[str] = []
    monkeypatch.setattr(bwt.gc, "collect", lambda: calls.append("gc"))
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: calls.append("empty_cache"))
    handle = f"ipc://{tmp_path / 'weight-transfer.sock'}"
    weights = _weights(1)
    sender = BucketedWeightSender(
        handle, bucket_size_mb=2, use_shm=True, gc_collect=gc_collect
    )
    receiver = BucketedWeightReceiver(handle, device=torch.device("cpu"), use_shm=True)
    received = _roundtrip(sender, receiver, weights)
    _assert_same(received, weights)
    assert sender.buffer is None and sender.shm is None
    assert ("cleanup_gc_s" in sender.stats) is gc_collect
    assert "cleanup_empty_cache_s" in sender.stats


def test_double_buffer_never_overwrites_the_slot_being_loaded(
    tmp_path, no_cuda
) -> None:
    """The sender races ahead into the other slot while the receiver is slow."""
    handle = f"ipc://{tmp_path / 'weight-transfer.sock'}"
    weights = _weights(3)
    expected = dict(weights)
    sender = BucketedWeightSender(
        handle, bucket_size_mb=2, use_shm=True, double_buffer=True
    )
    assert sender.bucket_size == 1 << 20, "two slots split the configured bucket"
    receiver = BucketedWeightReceiver(handle, device=torch.device("cpu"), use_shm=True)
    slots: list[int] = []
    offsets: dict[str, int] = {}

    def _slow_load(bucket) -> None:
        # Give the sender time to fill the other slot, then check that this
        # bucket's bytes in the shared buffer are still intact.
        time.sleep(0.05)
        for name, _ in bucket:
            ref = expected[name]
            start = offsets[name]
            view = (
                receiver.buffer[start : start + ref.nbytes]
                .view(ref.dtype)
                .view(ref.shape)
            )
            assert torch.equal(view, ref), f"{name} overwritten mid-load"

    orig_control = sender._control

    def _control(meta, is_last, ack_early=False):
        for name, m in meta.items():
            offsets[name] = m["offset"]
        if meta:
            slots.append(next(iter(meta.values()))["offset"] // sender.bucket_size)
        return orig_control(meta, is_last, ack_early)

    sender._control = _control
    _assert_same(_roundtrip(sender, receiver, weights, on_bucket=_slow_load), weights)
    assert slots[:4] == [0, 1, 0, 1]
    assert sender.stats["buckets"] > 3


def test_double_buffer_acks_last_bucket_after_loading(tmp_path, no_cuda) -> None:
    handle = f"ipc://{tmp_path / 'weight-transfer.sock'}"
    sender = BucketedWeightSender(
        handle, bucket_size_mb=2, use_shm=True, double_buffer=True
    )
    assert sender._control({}, is_last=False, ack_early=True)["ack_early"] is True
    messages: list[dict] = []
    orig = sender._control

    def _control(meta, is_last, ack_early=False):
        msg = orig(meta, is_last, ack_early)
        messages.append(msg)
        return msg

    sender._control = _control
    weights = _weights(4)
    receiver = BucketedWeightReceiver(handle, device=torch.device("cpu"), use_shm=True)
    _assert_same(_roundtrip(sender, receiver, weights), weights)
    assert all(m.get("ack_early") for m in messages[:-1])
    assert messages[-1]["is_last"] and "ack_early" not in messages[-1]


def test_legacy_sender_speaks_the_verl_protocol(tmp_path, no_cuda) -> None:
    """ATOM's receive loop reads the handshake and control messages raw."""
    handle = f"ipc://{tmp_path / 'weight-transfer.sock'}"
    sender = BucketedWeightSender(
        handle,
        bucket_size_mb=1,
        use_shm=True,
        gc_collect=True,
        double_buffer=False,
    )
    assert sender.bucket_size == 1 << 20
    assert sender._control({}, is_last=False) == {"bucket_meta": {}, "is_last": False}

    ctx = zmq.Context.instance()
    seen: list = []

    def _raw_receiver() -> None:
        sock = ctx.socket(zmq.REP)
        sock.setsockopt(zmq.RCVTIMEO, 5000)
        sock.connect(handle)
        try:
            while True:
                msg = sock.recv_pyobj()
                seen.append(msg)
                sock.send(b"")
                if isinstance(msg, dict) and msg.get("is_last"):
                    break
        finally:
            sock.close()

    thread = threading.Thread(target=_raw_receiver, daemon=True)
    thread.start()
    asyncio.run(sender.async_send_weights(_weights(5)))
    thread.join(timeout=10)
    assert set(seen[0]) == {"name", "size"}
    assert all(set(m) == {"bucket_meta", "is_last"} for m in seen[1:])


# --- CUDA IPC across processes -------------------------------------------------


def _gpu_receiver(handle: str, syncs: int, out) -> None:
    torch.cuda.set_device(0)
    try:
        for _ in range(syncs):
            got: dict[str, torch.Tensor] = {}

            def _load(bucket, got=got):
                got.update({n: t.clone().cpu() for n, t in bucket})

            rcv = BucketedWeightReceiver(handle, device=torch.device("cuda:0"))
            rcv.receive_weights(on_bucket_received=_load)
            # Plain bytes: torch's queue pickler would share tensors through fds
            # that vanish when this process exits.
            buf = io.BytesIO()
            torch.save(got, buf)
            out.put(("ok", buf.getvalue()))
    except (OSError, RuntimeError, ValueError) as exc:
        out.put(("err", repr(exc)))


def _gpu_weights(seed: int) -> list[tuple[str, torch.Tensor]]:
    g = torch.Generator().manual_seed(seed)
    return [
        ("a", torch.randn(100_000, generator=g)),
        ("b", torch.randn(150_000, generator=g)),
        ("big", torch.randn(700_000, generator=g)),  # > 2 MB bucket: direct send
        ("c", torch.randn(120_000, generator=g)),
        ("d", torch.randn(90_000, generator=g).to(torch.bfloat16)),
        ("e", torch.randn(200_000, generator=g)),
    ]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
@pytest.mark.parametrize(
    "options",
    [
        {"gc_collect": True, "double_buffer": False},
        {},
    ],
)
def test_cuda_ipc_two_syncs_with_direct_send(tmp_path, options) -> None:
    ctx = torch.multiprocessing.get_context("spawn")
    handle = f"ipc://{tmp_path / 'weight-transfer.sock'}"
    out = ctx.Queue()
    proc = ctx.Process(target=_gpu_receiver, args=(handle, 2, out), daemon=True)
    proc.start()
    torch.cuda.set_device(0)
    sender = BucketedWeightSender(handle, bucket_size_mb=2, **options)
    try:
        for seed in (10, 11):
            weights = [(n, t.cuda()) for n, t in _gpu_weights(seed)]
            asyncio.run(sender.async_send_weights(weights))
            status, payload = out.get(timeout=60)
            assert status == "ok", payload
            got = torch.load(io.BytesIO(payload))
            assert sender.stats["direct"] == 1
            assert sender.buffer is None
            for name, ref in weights:
                assert torch.equal(got[name], ref.cpu()), name
    finally:
        proc.join(timeout=30)
    assert proc.exitcode == 0
