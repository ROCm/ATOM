# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""ATOMesh's KV event decoder is tested on the batches in
atom/mesh/tests/fixtures/kv_events/samples.jsonl. This test publishes them again
with ATOM's block manager and publisher and checks that the file holds what ATOM
puts on the wire; with ATOM_UPDATE_MESH_SAMPLES=1 it rewrites the file instead.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import msgspec
import numpy as np
import zmq

from atom.distributed import kv_events
from atom.model_engine.block_manager import BlockManager
from atom.model_engine.prefix_hash import block_hash
from atom.model_engine.sequence import Sequence
from atom.sampling_params import SamplingParams

FIXTURE = Path(__file__).parents[1] / "atom/mesh/tests/fixtures/kv_events/samples.jsonl"

# Every sample in file order with its note; {seed} is the multimodal seq's cache_seed.
NOTES = {
    "startup_clear": "Seq 0 of every ATOM stream: make_publisher publishes a lone"
    " AllBlocksCleared before the engine's first batch.",
    "stored_root": "Root store from the block manager: parent_block_hash null, two"
    " blocks of 4 with xxh64 chain hashes (u64, one >= 2**63), medium GPU, lora_id"
    " and lora_name null, and ATOM's own token_offset 0, which the router ignores.",
    "stored_child": "A longer prompt hits stored_root's two blocks and stores one"
    " more: parent_block_hash is stored_root's last hash, token_offset 8.",
    "removed": "Eviction of the three cached blocks for a new prompt: one"
    " BlockRemoved per hash, medium GPU, no group_idx.",
    "remote_kv_gpu": "KV received from a PD prefill producer (decode side), medium"
    " GPU like any block in this engine's HBM. Two blocks after a locally cached"
    " one: parent_block_hash set, token_offset 4.",
    "multimodal_extra_keys": "Multimodal prompt: every block carries extra_keys"
    " [[seed, 0]], seed being the seq's cache_seed in 16 hex digits ({seed}). The"
    " seed also roots the hash chain, so this root event's parent_block_hash is"
    " the cache_seed itself, not a block hash.",
    "dp_rank_2": "Batch from data-parallel rank 2 (seq 1, after that publisher's"
    " startup clear); ATOM fills data_parallel_rank on every batch, 0 in the other"
    " samples.",
}


def _replay(pub, endpoint: str, rank: int, count: int) -> list[bytes]:
    """Every payload `pub` keeps for replay, in seq order."""
    deadline = time.monotonic() + 30
    while pub.stats["sent"] < count and time.monotonic() < deadline:
        time.sleep(0.01)
    dealer = zmq.Context.instance().socket(zmq.DEALER)
    try:
        dealer.connect(kv_events.offset_endpoint(f"{endpoint}-replay", rank))
        dealer.send_multipart([b"", (0).to_bytes(8, "big")])
        payloads = []
        while dealer.poll(timeout=5000):
            _, _, seq, payload = dealer.recv_multipart()
            if seq == kv_events.REPLAY_DONE:
                assert len(payloads) == count
                return payloads
            assert int.from_bytes(seq, "big") == len(payloads)
            payloads.append(payload)
        raise TimeoutError(f"no replay end marker from {endpoint}")
    finally:
        dealer.close(linger=0)


def _published() -> tuple[dict[str, bytes], str]:
    """Each sample's payload, and the multimodal seq's cache_seed."""
    config = SimpleNamespace(
        kv_cache_block_size=4,
        num_kvcache_blocks=4,
        enable_prefix_caching=True,
        decode_context_parallel_size=1,
        kv_events_config=SimpleNamespace(enable=True),
    )

    def seq(tokens, **kwargs):
        return Sequence(tokens, 4, sampling_params=SamplingParams(), **kwargs)

    def admit(bm, s, pub):
        n = bm.can_allocate(s)
        bm.allocate(s, n)
        bm.hash_blocks(s, (s.num_blocks - n) * bm.block_size)
        pub.publish(bm.take_events())

    def publisher(endpoint, rank):
        return kv_events.make_publisher(
            True,
            "zmq",
            endpoint,
            replay_endpoint=f"{endpoint}-replay",
            data_parallel_rank=rank,
        )

    stamps = iter(1763000400.0 + i / 4 for i in range(1000))
    with mock.patch.object(
        kv_events, "time", SimpleNamespace(time=lambda: next(stamps))
    ):
        pub = publisher("inproc://mesh-samples", 0)
        bm = BlockManager(config)
        s1, s2 = seq(list(range(100, 108))), seq(list(range(100, 112)))
        admit(bm, s1, pub)
        admit(bm, s2, pub)
        bm.deallocate(s1)
        bm.deallocate(s2)
        s3 = seq([7] * 16)
        bm.allocate(s3, bm.can_allocate(s3))
        pub.publish(bm.take_events())

        tokens = list(range(500, 512))
        h0 = block_hash(tokens[0:4])
        h1 = block_hash(tokens[4:8], h0)
        h2 = block_hash(tokens[8:12], h1)
        bm = BlockManager(config)
        bm.record_remote_store(
            [h1, h2], tokens[4:12], parent_block_hash=h0, token_offset=4
        )
        pub.publish(bm.take_events())

        pixels = np.zeros((4, 8), dtype=np.float32)
        media = {"pixel_values": pixels, "image_grid_thw": np.array([[1, 2, 2]])}
        mm = seq(list(range(600, 608)), multimodal_data=media)
        admit(BlockManager(config), mm, pub)

        pub_dp = publisher("inproc://mesh-samples-dp", 2)
        admit(BlockManager(config), seq(list(range(100, 108))), pub_dp)

    try:
        published = _replay(pub, "inproc://mesh-samples", 0, 6)
        published_dp = _replay(pub_dp, "inproc://mesh-samples-dp", 2, 2)
    finally:
        pub.shutdown()
        pub_dp.shutdown()
    payloads = dict(zip(list(NOTES)[:6], published))
    payloads["dp_rank_2"] = published_dp[1]
    return payloads, format(mm.cache_seed & (2**64 - 1), "016x")


def _line(name: str, note: str, payload: bytes) -> str:
    ts, events, rank = msgspec.msgpack.decode(payload)
    sample = {
        "engine": "atom",
        "name": name,
        "note": note,
        "batch": {"ts": ts, "data_parallel_rank": rank},
        "events": events,
        "payload": payload.hex(),
    }
    return json.dumps(sample, separators=(",", ":")) + "\n"


def test_mesh_samples_are_what_atom_publishes():
    payloads, seed = _published()
    _, [root], _ = msgspec.msgpack.decode(payloads["stored_root"])
    assert max(root["block_hashes"]) >= 2**63
    text = "".join(
        _line(name, note.format(seed=seed), payloads[name])
        for name, note in NOTES.items()
    )
    if os.environ.get("ATOM_UPDATE_MESH_SAMPLES") == "1":
        FIXTURE.parent.mkdir(parents=True, exist_ok=True)
        FIXTURE.write_text(text)
    assert FIXTURE.read_text() == text
