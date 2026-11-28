# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""atom.utils.gpu_guard on a fake KFD sysfs tree (no GPU needed)."""

import time

import pytest

from atom.utils import gpu_guard

GPU_IDS = [111, 222, 333, 444]


def _proc(root, pid, vram):
    d = root / "proc" / str(pid)
    d.mkdir(parents=True)
    for gid in GPU_IDS:
        (d / f"vram_{gid}").write_text(str(vram.get(gid, 0)))


@pytest.fixture
def kfd(tmp_path, monkeypatch):
    nodes = tmp_path / "topology" / "nodes"
    for i, gid in enumerate([0] + GPU_IDS):  # node 0: the CPU
        (nodes / str(i)).mkdir(parents=True)
        (nodes / str(i) / "gpu_id").write_text(str(gid))
    (tmp_path / "proc").mkdir()
    monkeypatch.setattr(gpu_guard, "KFD", str(tmp_path))
    monkeypatch.setenv("HIP_VISIBLE_DEVICES", "2,3")
    return tmp_path


def test_scan_maps_visible_gpus(kfd):
    _proc(kfd, 10, {333: 200 << 30})
    _proc(kfd, 11, {111: 5 << 30})  # another server's GPU
    _proc(kfd, 12, {})  # opened KFD, none of these GPUs
    _proc(kfd, 13, {444: 2 << 20})  # a fresh context on GPU 3
    assert gpu_guard.kfd_gpu_ids() == GPU_IDS
    gpus = {g: GPU_IDS[g] for g in gpu_guard.visible_gpus()}
    assert gpus == {2: 333, 3: 444}
    assert gpu_guard.scan(gpus) == {10: {2: 200 << 30}, 13: {3: 2 << 20}}


def _wait(cond, timeout=5.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if cond():
            return True
        time.sleep(0.01)
    return cond()


def _rm(kfd, pid):
    d = kfd / "proc" / str(pid)
    for f in d.iterdir():
        f.unlink()
    d.rmdir()


def test_guard_reports_newcomer_until_it_leaves(kfd):
    _proc(kfd, 10, {333: 200 << 30})
    _proc(kfd, 11, {444: 200 << 30})
    guard = gpu_guard.GpuGuard("test", interval_s=0.01, hold_off_s=0.0)
    try:
        assert not guard.present()
        _proc(kfd, 99, {444: 3 << 30})
        assert _wait(guard.present) and set(guard.foreign) == {99}
        _rm(kfd, 99)
        assert _wait(lambda: not guard.present())
    finally:
        guard.stop()


def test_guard_holds_off_after_the_newcomer_leaves(kfd):
    _proc(kfd, 10, {333: 200 << 30})
    guard = gpu_guard.GpuGuard("test", interval_s=0.01, hold_off_s=0.5)
    try:
        _proc(kfd, 99, {333: 2 << 20})
        assert _wait(guard.present)
        _rm(kfd, 99)
        time.sleep(0.2)
        assert guard.present()  # still held off
        assert _wait(lambda: not guard.present())
    finally:
        guard.stop()


def test_other_servers_gpus_are_not_watched(kfd):
    """DP ranks / a second server on the other GPUs of the node are not foreign."""
    _proc(kfd, 10, {333: 200 << 30})
    guard = gpu_guard.GpuGuard("test", interval_s=0.01)
    try:
        _proc(kfd, 50, {111: 200 << 30, 222: 200 << 30})
        time.sleep(0.1)
        assert not guard.present()
    finally:
        guard.stop()


def test_visible_gpus_rocr_and_uuid(kfd, monkeypatch, caplog):
    monkeypatch.delenv("HIP_VISIBLE_DEVICES")
    assert gpu_guard.visible_gpus() == [0, 1, 2, 3]
    # HIP indices count within the ROCR set
    monkeypatch.setenv("ROCR_VISIBLE_DEVICES", "1,3")
    monkeypatch.setenv("HIP_VISIBLE_DEVICES", "1")
    assert gpu_guard.visible_gpus() == [3]
    monkeypatch.setenv("ROCR_VISIBLE_DEVICES", "")
    monkeypatch.setenv("HIP_VISIBLE_DEVICES", "0,GPU-abc,9")
    with caplog.at_level("ERROR", logger="atom"):
        assert gpu_guard.visible_gpus() == [0]
    assert "not GPU indices" in caplog.text
