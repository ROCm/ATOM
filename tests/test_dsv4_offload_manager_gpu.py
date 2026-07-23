# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""End-to-end GPU test for the DSV4 offload manager (save -> store -> load)."""

from __future__ import annotations

import pytest
import torch

from atom.kv_transfer.disaggregation.types import KVTransferRegion, KVTransferTensors
from atom.kv_transfer.offload.dsv4.gpu_connector import (
    DSV4OffloadUnitGPUConnector,
)
from atom.kv_transfer.offload.dsv4.admission import (
    DSV4CheckpointAdmission,
    DSV4SwaIoPins,
)
from atom.kv_transfer.offload.dsv4.manager import (
    DSV4OffloadManager,
    InMemoryUnitStore,
)
from atom.kv_transfer.offload.dsv4.sources import DSV4OffloadSources
from atom.kv_transfer.offload.dsv4.unit import (
    LAYOUT_VERSION,
    DSV4OffloadUnitGeometry,
)
from atom.kv_transfer.offload.dsv4.unit_codec import DSV4OffloadUnitCodec

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="manager data path requires GPU"
)

BLOCK_SIZE = 256
COMP_BPB = 512
SWA_BPB = 384
SLOT_BYTES = 1024
N_STATE_SLOTS = 8


def _geometry() -> DSV4OffloadUnitGeometry:
    return DSV4OffloadUnitGeometry(
        model_name="deepseek-v4",
        layout_version=LAYOUT_VERSION,
        num_layers=1,
        head_dim=128,
        window_size=128,
        block_size=BLOCK_SIZE,
        swa_block_size=BLOCK_SIZE,
        k1_csa=64,
        k2_hca=2,
        kv_dtype="bf16",
        compress_ratios=(4,),
        tp_size=1,
        tp_rank=0,
    )


class _Harness:
    """Fake KV backing + gather/scatter closures over a compute-state pool."""

    def __init__(self, pool_size: int = 2):
        self.comp = torch.randint(0, 256, (4 * COMP_BPB,), dtype=torch.uint8, device="cuda")
        self.swa = torch.randint(0, 256, (4 * SWA_BPB,), dtype=torch.uint8, device="cuda")
        self.pool = torch.zeros(pool_size * SLOT_BYTES, dtype=torch.uint8, device="cuda")
        self.compute_state = torch.randint(
            0, 256, (N_STATE_SLOTS * SLOT_BYTES,), dtype=torch.uint8, device="cuda"
        )
        self.pool_size = pool_size

    def gather_slot(self, compute_slot: int, pool_idx: int) -> None:
        cs, pi = int(compute_slot), int(pool_idx)
        self.pool[pi * SLOT_BYTES : (pi + 1) * SLOT_BYTES] = self.compute_state[
            cs * SLOT_BYTES : (cs + 1) * SLOT_BYTES
        ]

    def scatter_slot(self, compute_slot: int, pool_idx: int) -> None:
        cs, pi = int(compute_slot), int(pool_idx)
        self.compute_state[cs * SLOT_BYTES : (cs + 1) * SLOT_BYTES] = self.pool[
            pi * SLOT_BYTES : (pi + 1) * SLOT_BYTES
        ]

    def transfer_tensors(self) -> KVTransferTensors:
        return KVTransferTensors(
            block_regions=[KVTransferRegion(self.comp.data_ptr(), 4 * COMP_BPB, COMP_BPB)],
            slot_regions=[],
            num_blocks=4,
            swa_block_regions=[
                KVTransferRegion(self.swa.data_ptr(), 4 * SWA_BPB, SWA_BPB)
            ],
            staging_region=KVTransferRegion(
                self.pool.data_ptr(), self.pool_size * SLOT_BYTES, SLOT_BYTES
            ),
            staging_pool_size=self.pool_size,
            gather_slot=self.gather_slot,
            scatter_slot=self.scatter_slot,
        )


def _manager(h: _Harness, *, pool_size=2, inflight=2):
    codec = DSV4OffloadUnitCodec(_geometry())
    conn = DSV4OffloadUnitGPUConnector(codec, device="cuda")
    sources = DSV4OffloadSources(h.transfer_tensors(), block_size=BLOCK_SIZE, window_size=128)
    return DSV4OffloadManager(
        connector=conn,
        sources=sources,
        store=InMemoryUnitStore(),
        admission=DSV4CheckpointAdmission(
            state_pool_size=pool_size, max_inflight_saves=inflight
        ),
        swa_pins=DSV4SwaIoPins(),
    )


def test_save_then_load_full_round_trip():
    h = _Harness()
    mgr = _manager(h)
    block_table = [0, 1]  # B=512 -> 2 compressed blocks
    swa_bt = [-1, -1, 2, 3]  # 2 live SWA pages
    compute_slot = 5
    B = 512

    comp0 = h.comp.clone()
    swa0 = h.swa.clone()
    state0 = h.compute_state[compute_slot * SLOT_BYTES : (compute_slot + 1) * SLOT_BYTES].clone()

    assert mgr.try_save_checkpoint(
        key="k", block_table=block_table, swa_block_table=swa_bt,
        compute_slot=compute_slot, B=B,
    )
    torch.cuda.synchronize()
    assert mgr.swa_pins.pinned_pages() == set()  # unpinned after save
    assert mgr.admission.free_slots == 2  # slot released

    # Wipe the KV + compute state, then restore from the stored unit.
    h.comp.zero_()
    h.swa.zero_()
    h.compute_state[compute_slot * SLOT_BYTES : (compute_slot + 1) * SLOT_BYTES] = 0

    assert mgr.load_checkpoint(
        key="k", block_table=block_table, swa_block_table=swa_bt,
        compute_slot=compute_slot, state_pool_idx=1, B=B,
    )
    torch.cuda.synchronize()

    # Compressed blocks 0,1 restored.
    assert torch.equal(h.comp[: 2 * COMP_BPB], comp0[: 2 * COMP_BPB])
    # SWA live pages 2,3 restored.
    assert torch.equal(h.swa[2 * SWA_BPB : 4 * SWA_BPB], swa0[2 * SWA_BPB : 4 * SWA_BPB])
    # CSA state round-tripped pool->compute.
    restored = h.compute_state[compute_slot * SLOT_BYTES : (compute_slot + 1) * SLOT_BYTES]
    assert torch.equal(restored, state0)


def test_save_skips_when_admission_exhausted():
    h = _Harness()
    mgr = _manager(h, pool_size=0, inflight=0)  # no resources
    ok = mgr.try_save_checkpoint(
        key="k", block_table=[0, 1], swa_block_table=[-1, -1, 2, 3],
        compute_slot=5, B=512,
    )
    assert ok is False
    assert not mgr.store.contains("k")
    assert mgr.swa_pins.pinned_pages() == set()  # no pin leak


def test_save_skips_unaligned_boundary():
    h = _Harness()
    mgr = _manager(h)
    assert not mgr.try_save_checkpoint(
        key="k", block_table=[0, 1, 2], swa_block_table=[-1, 2],
        compute_slot=5, B=520,  # not 128-aligned
    )
    assert not mgr.store.contains("k")


def test_load_miss_returns_false():
    h = _Harness()
    mgr = _manager(h)
    assert not mgr.load_checkpoint(
        key="absent", block_table=[0, 1], swa_block_table=[-1, -1, 2, 3],
        compute_slot=5, state_pool_idx=0, B=512,
    )


def test_load_corrupt_fails_closed_without_touching_kv():
    h = _Harness()
    mgr = _manager(h)
    block_table = [0, 1]
    swa_bt = [-1, -1, 2, 3]
    assert mgr.try_save_checkpoint(
        key="k", block_table=block_table, swa_block_table=swa_bt,
        compute_slot=5, B=512,
    )
    # Corrupt the stored unit's payload.
    mgr.store._d["k"][-1] ^= 0xFF  # type: ignore[attr-defined]

    h.comp.zero_()
    comp_after = h.comp.clone()
    assert not mgr.load_checkpoint(
        key="k", block_table=block_table, swa_block_table=swa_bt,
        compute_slot=5, state_pool_idx=1, B=512,
    )
    torch.cuda.synchronize()
    # Fail-closed before any scatter: KV untouched.
    assert torch.equal(h.comp, comp_after)
