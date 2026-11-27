# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""umbp_pool on a pipeline-parallel engine (MiniMax-M3.1's PP4 prefill).

The stages of a PP engine hold different layers of the same blocks under the
same block ids, so their buffers are the parts of one UMBP pool, as the TP
shards of a rank are (`pool.pool_part`). The head's scheduler owns the pool.
Each stage commits the head's stamp of the batch it ran
(`ScheduledBatch.kv_pool_stamp`), not its own clock: the head runs batches
ahead of the later stages, so a stage's clock can be past a flush whose
batch it has not run.
"""

import ctypes
import os
import random
import sys
import time
import types

import pytest
import test_umbp_l1_pool as umbp_tests
from conftest import MockConfig, _MockHFConfig

from atom.kv_transfer.pool.umbp import pool as umbp_pool
from atom.kv_transfer.pool.umbp.connector import UmbpPoolConnectorScheduler
from atom.kv_transfer.pool.umbp.settings import UmbpPoolSettings

ADDRESS = "unix:///tmp/umbp-l1/server.sock"
M3 = "MiniMaxM3SparseForConditionalGeneration"


def pp_config(pp=4, tp=1, pp_rank=0, dp_rank=0, layers=60):
    """The fields the pool reads, for one stage of a PP x TP engine."""
    return types.SimpleNamespace(
        tensor_parallel_size=tp,
        pipeline_parallel_size=pp,
        parallel_config=types.SimpleNamespace(
            data_parallel_rank=dp_rank, pipeline_parallel_rank=pp_rank
        ),
        model="MiniMax-M3.1",
        kv_cache_dtype="fp8",
        index_cache_dtype="fp8",
        kv_cache_block_size=128,
        quant_config=types.SimpleNamespace(online_quant_config_raw=None),
        speculative_config=None,
        hf_config=types.SimpleNamespace(num_hidden_layers=layers),
    )


class TestParts:
    def test_a_stage_is_a_part_and_its_tp_ranks_parts_within_it(self):
        assert umbp_pool.pool_parts(pp_config(pp=4)) == 4
        assert [
            umbp_pool.pool_part(pp_config(pp=4, pp_rank=r), 0) for r in range(4)
        ] == [0, 1, 2, 3]
        assert umbp_pool.pool_parts(pp_config(pp=2, tp=2)) == 4
        assert umbp_pool.pool_part(pp_config(pp=2, tp=2, pp_rank=1), 1) == 3

    def test_a_runner_outside_its_stage_has_no_part(self):
        with pytest.raises(ValueError, match="no part"):
            umbp_pool.pool_part(pp_config(pp=4), 1)

    def test_every_stage_names_the_same_pool(self):
        settings = UmbpPoolSettings(address=ADDRESS)
        ids = {umbp_pool.pool_id_for(pp_config(pp_rank=r), settings) for r in range(4)}
        assert len(ids) == 1

    def test_the_runner_side_brings_its_stages_part(self, monkeypatch):
        monkeypatch.setattr(umbp_pool, "_connect", lambda settings: (None, object()))
        settings = UmbpPoolSettings(address=ADDRESS)
        memory = umbp_pool.make_umbp_pool_memory(pp_config(pp=4, pp_rank=2), settings)
        assert (memory.part, memory.num_parts, memory.batch_stamps) == (2, 4, True)
        flat = umbp_pool.make_umbp_pool_memory(pp_config(pp=1), settings)
        assert (flat.part, flat.num_parts, flat.batch_stamps) == (0, 1, False)


class TestShareGroup:
    def test_a_pp_pool_shares_only_with_pools_of_the_same_split(self):
        settings = UmbpPoolSettings(address=ADDRESS)
        pp4 = umbp_pool.share_group_for(pp_config(pp=4), settings)
        # Every stage and every engine of the same layout derive the same id.
        assert umbp_pool.share_group_for(pp_config(pp=4, pp_rank=3), settings) == pp4
        assert umbp_pool.share_group_for(pp_config(pp=4, dp_rank=1), settings) == pp4
        # A TP4 pool has four parts too, holding other bytes.
        assert umbp_pool.share_group_for(pp_config(pp=1, tp=4), settings) != pp4
        assert umbp_pool.share_group_for(pp_config(pp=2, tp=2), settings) != pp4

    def test_without_pp_the_group_is_as_before(self):
        settings = UmbpPoolSettings(address=ADDRESS)
        config = pp_config(pp=1)
        before = types.SimpleNamespace(
            **{k: v for k, v in vars(config).items() if k != "pipeline_parallel_size"}
        )
        assert umbp_pool.share_group_for(config, settings) == (
            umbp_pool.share_group_for(before, settings)
        )


def m3_config(pp=4, pp_rank=0):
    config = MockConfig(
        kv_transfer_config={
            "kv_connector": "umbp_pool",
            "kv_connector_extra_config": {"address": ADDRESS},
        },
        hf_config=_MockHFConfig([M3]),
    )
    config.pipeline_parallel_size = pp
    config.parallel_config = types.SimpleNamespace(
        data_parallel_rank=0, pipeline_parallel_rank=pp_rank
    )
    return config


class TestTheHeadOwnsThePool:
    def test_pp_is_accepted(self):
        UmbpPoolConnectorScheduler(m3_config())

    def test_a_downstream_stage_builds_no_pool(self, monkeypatch):
        opened = []
        monkeypatch.setattr(
            umbp_pool, "make_umbp_block_pool", lambda *a, **k: opened.append(a)
        )
        for stage in (1, 2, 3):
            conn = UmbpPoolConnectorScheduler(m3_config(pp_rank=stage))
            assert conn.make_block_pool(None, 8) is None
            assert conn.pool is None
        assert opened == []

    def test_the_head_builds_it(self, monkeypatch):
        monkeypatch.setattr(
            umbp_pool, "make_umbp_block_pool", lambda *a, **k: "the pool"
        )
        conn = UmbpPoolConnectorScheduler(m3_config(pp_rank=0))
        assert conn.make_block_pool(None, 8) == "the pool"


class _DoneEvent:
    """A device event whose forward has already finished."""

    def record(self):
        pass

    def query(self):
        return True


@pytest.fixture
def done_events(monkeypatch):
    import torch

    monkeypatch.setattr(torch.cuda, "Event", _DoneEvent)


class _Commits:
    """Forwards to `client` (if any) and keeps every commit, by part."""

    def __init__(self, client=None):
        self._client = client
        self.stamps: list[tuple[int | None, int]] = []

    def __getattr__(self, name):
        return getattr(self._client, name)

    def commit(self, pool_id, stamp, **kwargs):
        if self._client is not None:
            self._client.commit(pool_id, stamp, **kwargs)
        self.stamps.append((kwargs.get("part"), stamp))

    def detach(self, *args, **kwargs):
        if self._client is not None:
            self._client.detach(*args, **kwargs)


def _committed(recorder, stamp, timeout=5.0):
    """Wait until the progress thread has committed `stamp`."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if recorder.stamps and recorder.stamps[-1][1] == stamp:
            return recorder.stamps
        time.sleep(0.001)
    raise AssertionError(f"{stamp} was not committed: {recorder.stamps}")


@pytest.mark.usefixtures("done_events")
class TestStageStamps:
    def memory(self, client, batch_stamps=True, part=3, num_parts=4):
        memory = umbp_pool.UmbpPoolMemory(
            client,
            "atom:dp0",
            "g",
            1.0,
            part=part,
            num_parts=num_parts,
            batch_stamps=batch_stamps,
        )
        memory._mapped_base = 0x1000  # as `bind_layout` leaves it
        return memory

    def test_a_stage_commits_the_stamp_of_the_batch_it_ran(self):
        recorder = _Commits()
        memory = self.memory(recorder)
        try:
            memory.writes_issued(100)
            assert _committed(recorder, 100) == [(3, 100)]
            # This process's clock is far past 100, but only the batch the
            # forward carried counts.
            memory.writes_issued(250)
            _committed(recorder, 250)
            assert max(stamp for _, stamp in recorder.stamps) == 250
        finally:
            memory.close()

    def test_a_forward_no_batch_decided_proves_nothing_new(self):
        recorder = _Commits()
        memory = self.memory(recorder)
        try:
            memory.writes_issued(None)  # before any batch: nothing to report
            time.sleep(0.05)
            assert recorder.stamps == []
            memory.writes_issued(100)
            memory.writes_issued(None)  # a dummy forward after it
            _committed(recorder, 100)
            assert {stamp for _, stamp in recorder.stamps} == {100}
        finally:
            memory.close()

    def test_a_stamp_never_goes_back(self):
        recorder = _Commits()
        memory = self.memory(recorder)
        try:
            memory.writes_issued(300)
            memory.writes_issued(200)
            _committed(recorder, 300)
            assert all(stamp == 300 for _, stamp in recorder.stamps)
        finally:
            memory.close()

    def test_without_pp_a_runner_commits_its_own_clock(self):
        recorder = _Commits()
        memory = self.memory(recorder, batch_stamps=False, part=0, num_parts=1)
        try:
            before = time.monotonic_ns()
            memory.writes_issued(5)  # a batch stamp means nothing here
            deadline = time.monotonic() + 5
            while not recorder.stamps and time.monotonic() < deadline:
                time.sleep(0.001)
            assert recorder.stamps and recorder.stamps[0][1] >= before
            assert recorder.stamps[0][0] is None  # one part: no part argument
        finally:
            memory.close()


@pytest.mark.usefixtures("done_events")
class TestTwoPipelineEngines:
    """Two PP4 engines on one UMBP server: a block one computed moves to the
    other part by part, and only once every stage has run the batch."""

    BLOCKS = 4
    # A block's bytes on each stage: M3.1's split gives every stage a layout
    # of its own (17/17/17/9 layers, the DSpark draft's KV on the last).
    UNITS = (24, 24, 24, 16)

    @pytest.fixture
    def service(self):
        mori = umbp_tests.mori_umbp_l1
        if mori is None or "num_parts" not in (mori.L1Client.attach.__doc__ or ""):
            pytest.skip("mori_umbp_l1 is not built, or has no pool parts")
        path = f"/tmp/atom_l1_pp_{os.getpid()}_{random.randrange(1 << 30)}.sock"
        server = mori.HostTestServer(f"unix://{path}")
        assert server.ok()
        yield mori.L1Client(f"unix://{path}")
        del server
        if os.path.exists(path):
            os.unlink(path)

    @pytest.fixture(autouse=True)
    def host_handles(self, monkeypatch):
        """The host test server adopts memory by address, not by HIP handle."""
        from atom.mono import runtime

        stub = types.SimpleNamespace(
            get_allocation_base=lambda address: address,
            get_ipc_handle=lambda address: address.to_bytes(8, "little"),
        )
        monkeypatch.setattr(runtime, "hip_ipc", stub, raising=False)
        monkeypatch.setitem(sys.modules, "atom.mono.runtime.hip_ipc", stub)

    def stage(self, client, dp_rank, pp_rank):
        """What one stage's `ModelRunner` does with its paged pool."""
        config = pp_config(pp=4, pp_rank=pp_rank, dp_rank=dp_rank)
        settings = UmbpPoolSettings(address=ADDRESS)
        recorder = _Commits(client)
        memory = umbp_pool.UmbpPoolMemory(
            recorder,
            umbp_pool.pool_id_for(config, settings),
            "g",
            1.0,
            part=umbp_pool.pool_part(config, 0),
            num_parts=umbp_pool.pool_parts(config),
            batch_stamps=True,
        )
        unit = self.UNITS[pp_rank]
        raw = (ctypes.c_uint8 * (self.BLOCKS * unit))()
        buf = types.SimpleNamespace(
            data_ptr=lambda: ctypes.addressof(raw),
            numel=lambda: len(raw),
            device=types.SimpleNamespace(index=4 * dp_rank + pp_rank),
        )
        region = types.SimpleNamespace(
            reverse_indexed=False, base_addr=ctypes.addressof(raw), unit_bytes=unit
        )
        memory.bind_layout(buf, [region], self.BLOCKS)
        return memory, raw, recorder

    def head(self, client, dp_rank):
        """The head stage's scheduler: the engine's one pool."""
        config = pp_config(pp=4, dp_rank=dp_rank)
        return umbp_pool.UmbpL1Pool(
            client,
            umbp_tests.OPS,
            umbp_pool.pool_id_for(config, UmbpPoolSettings(address=ADDRESS)),
            self.BLOCKS,
            # One block a grant: a larger grant would hold the whole pool.
            grant_batch=1,
            parts=umbp_pool.pool_parts(config),
        )

    def test_a_block_moves_between_engines_only_once_every_stage_ran_it(self, service):
        stages = {
            (dp, pp): self.stage(service, dp, pp) for dp in (0, 1) for pp in range(4)
        }
        a, b = self.head(service, 0), self.head(service, 1)
        b.allocate(b.pop())  # so the block lands at another id on engine 1
        block = a.pop()
        a.allocate(block)
        for pp in range(4):
            unit = self.UNITS[pp]
            stages[0, pp][1][block * unit : (block + 1) * unit] = (
                bytes([0x40 + pp]) * unit
            )
        a.publish(block, 5, umbp_tests.toks(5))
        a.free(block)
        a.flush()
        # The head's stamp of the batch after this flush, and of one before it.
        batch = time.monotonic_ns()
        earlier = batch - 1_000_000
        # Stages 0 to 2 ran the batch. The last stage, behind the head, is
        # still on an earlier one: its own clock is past the flush already,
        # which is what it must not commit.
        for pp in range(3):
            stages[0, pp][0].writes_issued(batch)
            _committed(stages[0, pp][2], batch)
        stages[0, 3][0].writes_issued(earlier)
        _committed(stages[0, 3][2], earlier)
        assert time.monotonic_ns() > batch
        assert b.fetch([5], lambda i: umbp_tests.toks(5)) == 0
        # Once the last stage has run it too, the block is the other engine's.
        stages[0, 3][0].writes_issued(batch)
        _committed(stages[0, 3][2], batch)
        assert b.fetch([5], lambda i: umbp_tests.toks(5)) == 1
        landed = b.lookup(5)
        assert landed not in (-1, block)
        for pp in range(4):
            unit = self.UNITS[pp]
            assert (
                bytes(stages[1, pp][1][landed * unit : (landed + 1) * unit])
                == bytes([0x40 + pp]) * unit
            )
        stats = {s["pool_id"]: s for s in service.stats()}
        assert stats[a.pool_id]["parts"] == 4
        assert stats[b.pool_id]["fetched_in_bytes"] == sum(self.UNITS)
