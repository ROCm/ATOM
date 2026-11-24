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


def test_visible_gpus_follow_rocr_uuids(kfd, monkeypatch):
    # unique ids per GPU node (node 0 is the CPU): ROCR_VISIBLE_DEVICES=GPU-<hex>
    uids = [0xA1, 0xB2, 0xC3, 0xD4]
    for i, uid in enumerate(uids, start=1):
        (kfd / "topology" / "nodes" / str(i) / "properties").write_text(
            f"cpu_cores_count 0\nunique_id {uid}\n"
        )
    monkeypatch.delenv("HIP_VISIBLE_DEVICES")
    monkeypatch.setenv("ROCR_VISIBLE_DEVICES", "GPU-d4,GPU-00000000000000C3")
    assert gpu_guard.visible_gpus() == [3, 2]
    # HIP indices count within what ROCr leaves visible
    monkeypatch.setenv("HIP_VISIBLE_DEVICES", "1")
    assert gpu_guard.visible_gpus() == [2]
    monkeypatch.setenv("ROCR_VISIBLE_DEVICES", "0,1")
    monkeypatch.delenv("HIP_VISIBLE_DEVICES")
    assert gpu_guard.visible_gpus() == [0, 1]


def test_guard_reports_newcomer_until_it_leaves(kfd):
    _proc(kfd, 10, {333: 200 << 30})
    _proc(kfd, 11, {444: 200 << 30})
    guard = gpu_guard.GpuGuard("test", interval_s=0.01)
    try:
        time.sleep(0.05)
        assert not guard.present()
        _proc(kfd, 99, {444: 3 << 30})
        time.sleep(0.05)
        assert guard.present() and set(guard.foreign) == {99}
        for f in (kfd / "proc" / "99").iterdir():
            f.unlink()
        (kfd / "proc" / "99").rmdir()
        time.sleep(0.05)
        assert not guard.present()
    finally:
        guard.stop()


def test_own_kickers_are_not_foreign(kfd):
    _proc(kfd, 10, {333: 200 << 30})
    guard = gpu_guard.GpuGuard("test", interval_s=0.01)
    try:
        guard.transient_until = time.monotonic() + 60  # as guard.kick() sets it
        _proc(kfd, 77, {333: 2 << 20})
        time.sleep(0.05)
        assert not guard.present()
    finally:
        guard.stop()
