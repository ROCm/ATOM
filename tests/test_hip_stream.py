# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

from types import SimpleNamespace

import pytest

from atom.utils import hip_stream


class _FakeCreateStream:
    def __init__(self, *, result: int = 0, handle: int = 0x1234) -> None:
        self.result = result
        self.handle = handle
        self.restype = None
        self.argtypes = None
        self.words: list[int] | None = None

    def __call__(self, raw_stream, word_count, mask) -> int:
        count = int(word_count)
        self.words = [int(mask[i]) for i in range(count)]
        raw_stream._obj.value = self.handle
        return self.result


def _mock_rocm(monkeypatch: pytest.MonkeyPatch, *, cu_count: int = 65):
    create_stream = _FakeCreateStream()
    hip = SimpleNamespace(hipExtStreamCreateWithCUMask=create_stream)
    external_stream_calls: list[int] = []

    monkeypatch.setattr(hip_stream.torch.version, "hip", "7.2")
    monkeypatch.setattr(hip_stream.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(hip_stream.torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(
        hip_stream.torch.cuda,
        "get_device_properties",
        lambda _device: SimpleNamespace(multi_processor_count=cu_count),
    )
    monkeypatch.setattr(hip_stream.ctypes, "CDLL", lambda _name: hip)
    monkeypatch.setattr(
        hip_stream.torch.cuda,
        "ExternalStream",
        lambda handle: external_stream_calls.append(handle) or ("stream", handle),
    )
    return create_stream, external_stream_calls


def test_full_device_stream_uses_exact_cu_mask(monkeypatch: pytest.MonkeyPatch) -> None:
    create_stream, external_stream_calls = _mock_rocm(monkeypatch, cu_count=65)

    stream = hip_stream.create_full_device_hip_stream()

    assert stream == ("stream", 0x1234)
    assert create_stream.words == [0xFFFFFFFF, 0xFFFFFFFF, 0x1]
    assert external_stream_calls == [0x1234]


def test_cu_mask_stream_rejects_non_rocm_build(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(hip_stream.torch.version, "hip", None)

    with pytest.raises(RuntimeError, match="ROCm PyTorch build"):
        hip_stream.create_hip_stream_with_cu_mask([0xFFFFFFFF])


def test_cu_mask_stream_surfaces_hip_error(monkeypatch: pytest.MonkeyPatch) -> None:
    create_stream, _ = _mock_rocm(monkeypatch)
    create_stream.result = 17

    with pytest.raises(RuntimeError, match="HIP error 17"):
        hip_stream.create_hip_stream_with_cu_mask([0xFFFFFFFF])
