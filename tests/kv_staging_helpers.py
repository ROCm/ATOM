# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""CPU fakes shared by the Mooncake KV staging and landing tests.

The producer connector is built without ``__init__`` and a memmove stands in
for the NIC, so the per-token, staged and landed paths all write real bytes
that a test compares.
"""

import ctypes
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

from atom.kv_transfer.disaggregation.mooncake.landing import LandingCredits
from atom.kv_transfer.disaggregation.types import MLA_KV_ROLE

BLOCK_SIZE = 16
TOKEN_BYTES = 576
PAGE_BYTES = BLOCK_SIZE * TOKEN_BYTES


def mooncake(monkeypatch=None):
    """The connector module; with ``monkeypatch``, CUDA streams are no-ops."""
    mc = pytest.importorskip(
        "atom.kv_transfer.disaggregation.mooncake.mooncake_connector"
    )
    if monkeypatch is not None:
        monkeypatch.setattr(mc.torch.cuda, "stream", lambda _s: nullcontext())
    return mc


class Nic:
    """RDMA stand-in: memmove every descriptor and record each write."""

    def __init__(self):
        self.labels = []
        self.writes = []  # descriptors per write
        self.dst_spans = []

    @property
    def descriptors(self):
        return sum(self.writes)

    def write(
        self, _target, src, dst, sizes, _req, label, *, engine=None, deadline=None
    ):
        assert len(src) == len(dst) == len(sizes)
        for s, d, n in zip(src, dst, sizes):
            ctypes.memmove(d, s, n)
            self.dst_spans.append((d, n))
        self.labels.append(label)
        self.writes.append(len(src))
        return True


class Stream:
    def __init__(self):
        self.waited = []
        self.synchronized = 0

    def wait_event(self, event):
        self.waited.append(event)

    def synchronize(self):
        self.synchronized += 1


def source_regions(num_regions, num_blocks):
    """MLA regions whose every token is distinct: (region, block, token) tag."""
    gen = torch.Generator().manual_seed(num_regions * 1000 + num_blocks)
    regions = []
    for r in range(num_regions):
        t = torch.randint(0, 256, (num_blocks, PAGE_BYTES), generator=gen).to(
            torch.uint8
        )
        tag = t.view(num_blocks * BLOCK_SIZE, TOKEN_BYTES)[:, :4].view(torch.int32)
        tag[:, 0] = torch.arange(num_blocks * BLOCK_SIZE, dtype=torch.int32) + (r << 20)
        regions.append(t)
    return regions


def make_producer(mc, regions, *, slots=0, slot_bytes=0):
    """A producer over ``regions``; ``slots`` gives it a KV staging pool."""
    conn = object.__new__(mc.MooncakeConnector)
    conn.tp_size = conn.dcp_size = 1
    conn.pp_rank = conn.tp_rank = 0
    conn.block_size = BLOCK_SIZE
    conn.kv_caches_base_addr = [t.data_ptr() for t in regions]
    conn._per_block_bytes_list = [PAGE_BYTES] * len(regions)
    conn._block_region_roles = [MLA_KV_ROLE] * len(regions)
    conn._block_region_consumer_indices = None
    conn._block_unit_specs = None
    conn._block_head_major_runs = ()
    conn._unit_layout = None
    conn._consumer_region_map = lambda *_a, **_k: list(range(len(regions)))
    conn._prepare_sharded_index = None
    conn._gather_sharded_index = None
    conn._index_staging_chunk_pages = 0
    conn._landing_credits = LandingCredits()
    conn._nic = Nic()
    conn._rdma_write = conn._nic.write
    if slots:
        conn._kv_staging = torch.empty((slots, slot_bytes), dtype=torch.uint8)
        conn._kv_staging_slots = mc.SlotPool(slots)
        conn._stream = Stream()
        conn._send_worker_stream = lambda: conn._stream
        conn._kv_source_views = {
            i: t.view(-1, TOKEN_BYTES) for i, t in enumerate(regions)
        }
    return conn


def write_request(num_regions, dcp_size, dcp_rank, dst_bufs, *, interleave=1):
    """A DCP consumer's write request for its MLA regions ``dst_bufs``."""
    return {
        "request_id": "req",
        "write_nonce": 5,
        "notify_host": "decode",
        "notify_port": 1,
        "consumer_base_addrs": [t.data_ptr() for t in dst_bufs],
        "consumer_num_layers": num_regions,
        "consumer_region_roles": [MLA_KV_ROLE] * num_regions,
        "consumer_block_bpb": [PAGE_BYTES] * num_regions,
        "consumer_dcp_size": dcp_size,
        "consumer_dcp_rank": dcp_rank,
        "consumer_dcp_interleave": interleave,
    }


def run_transfer(mc, conn, request, src_ids, dst_ids, event):
    return mc.MooncakeConnector._execute_block_transfer(
        conn, request, "consumer:1", list(src_ids), list(dst_ids), "req", event
    )


def mla_producer_config(workers, **overrides):
    """A plain MLA engine config with one Mooncake producer connector."""
    config = {
        "kv_transfer_config": {
            "kv_connector": "mooncake",
            "kv_role": "kv_producer",
            "num_worker_threads": workers,
        },
        "hf_config": SimpleNamespace(kv_lora_rank=512),
        "decode_context_parallel_size": 1,
    }
    config.update(overrides)
    return SimpleNamespace(**config)
