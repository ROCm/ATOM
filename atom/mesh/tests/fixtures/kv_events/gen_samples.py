"""Generate samples.jsonl, the KV event batches ATOMesh's decoder tests replay.

    python gen_samples.py [--atom-root DIR] [--out FILE] [--check]

Each line is one batch as its publisher puts it on the wire: the msgpack bytes
as hex (`payload`), the batch and events decoded from them (byte hashes as
`{"bytes_hex"}`), the producer code the sample follows and a note on it.

vLLM and SGLang batches are encoded as their publishers encode them, with the
engines' structs vendored below, trimmed to the fields the samples need. ATOM
batches come from the block manager and publisher of the ATOM checkout at
`--atom-root` (default: this one), read back from the publisher's replay socket;
atom/main_* re-encode two of them with the kv_events.py of ATOM main before
#3275, read from the checkout's git history.
`--out` defaults to samples.jsonl next to this script. `--check` also decodes
every vLLM and ATOM payload with vLLM's `KVEventBatch` and every SGLang payload
with SGLang's; only sglang/eagle_bigram and the atom/main_* arrays are rejected.
"""

import argparse
import hashlib
import importlib.util
import json
import re
import subprocess
import sys
import tempfile
import time
import types
from pathlib import Path
from typing import Any, TypeAlias
from unittest import mock

import msgspec

# SGLang's structs are pinned at 0c44fbff, the code emitting its events at 72f72cca.
PINS = {
    "{vllm}": "https://github.com/vllm-project/vllm/blob/3a104103ac70740824599453c7036edcca1e8548/",
    "{sglang}": "https://github.com/sgl-project/sglang/blob/0c44fbff44da7d1a11f0e8b4299e4b8a702ac414/",
    "{sglang_emit}": "https://github.com/sgl-project/sglang/blob/72f72ccacb3a73fcd99d401e25ca986bda6144c8/",
    "{atom}": "https://github.com/ROCm/ATOM/blob/2b2fb2e6f83daa9a78202adf2c168ede5b600036/",
}
# The ROCm/ATOM#3275 commit whose serializer published the ATOM payloads in samples.jsonl.
ATOM_PUBLISHED = "Published by ATOM at edd7abdd4744."

# Every sample in samples.jsonl order: its engine/name and source permalink, then
# its note on indented lines. ATOM notes get ATOM_PUBLISHED appended, and {seed}
# is the multimodal sequence's cache_seed.
SAMPLES = """
vllm/stored_root {vllm}vllm/v1/core/block_pool.py#L364-L376
    Root store: parent_block_hash null, two blocks, u64 hashes (both >= 2**63, so msgpack uint64), medium GPU. As vLLM publishes plain text: extra_keys holds one null per block, and group_idx 0 with kv_cache_spec_kind full_attention are filled in by the KV cache manager ({vllm}vllm/v1/core/kv_cache_manager.py#L721-L746).
vllm/stored_child {vllm}vllm/v1/core/block_pool.py#L364-L376
    Continuation of stored_root: parent_block_hash is stored_root's last block hash.
vllm/stored_bytes_hashes {vllm}vllm/v1/core/kv_cache_utils.py#L86-L89
    VLLM_KV_EVENTS_USE_INT_BLOCK_HASHES=0: hashes are the raw 32-byte sha256 digests (msgpack bin). A root of two blocks, then a child whose parent_block_hash is bytes too. The u64 hashes of stored_root are the low 8 bytes of these.
vllm/removed {vllm}vllm/v1/core/block_pool.py#L626-L639
    Eviction of stored_root's blocks: one BlockRemoved per hash, each with medium GPU and its group_idx.
vllm/all_cleared {vllm}vllm/v1/core/block_pool.py#L898-L899
    Prefix-cache reset: a lone AllBlocksCleared, a map holding only type.
vllm/dp_rank_1 {vllm}vllm/distributed/kv_events.py#L392-L396
    Batch from data-parallel rank 1: the publisher fills data_parallel_rank (it is 0 in every other vLLM sample, the publisher default).
vllm/salt_first_block {vllm}vllm/v1/core/kv_cache_utils.py#L655-L659
    cache_salt 'salt-abc': only the block starting at token 0 carries the salt, as a bare string in its extra_keys entry. The second block of the root and the continuation event that follows have no salt (null entries).
vllm/lora_salt {vllm}vllm/v1/core/kv_cache_utils.py#L674-L700
    LoRA adapter 'my-lora' (adapter id 1) with cache_salt 'salt-abc': the LoRA name leads every block's extra_keys entry and the salt follows it on the first block only, so the entries are ('my-lora', 'salt-abc') and ('my-lora',).
vllm/multimodal {vllm}vllm/v1/core/kv_cache_utils.py#L518-L583
    Two images after a text-only block: A at token offset 5 (length 5), B at offset 11 (length 2), placeholder token 151655. Each block lists an (mm_identifier, offset) pair per image it overlaps, offset relative to the block start and so negative past an image's first block: null for the text-only block, then ((A, 1),), ((A, -3), (B, 3)), ((B, -1),). Identifiers are 64-hex-char stand-ins for vLLM's blake3 media hash.
vllm/sliding_window_group {vllm}vllm/v1/core/kv_cache_manager.py#L721-L746
    Hybrid model: the same blocks published once per KV cache group with the same hashes, group 0 full_attention and group 1 sliding_window with kv_cache_spec_sliding_window 128.
vllm/lookahead_token_ids {vllm}vllm/v1/core/block_pool.py#L296-L315
    Lookahead hashing with 2 lookahead tokens: lookahead_token_ids has one list per block, the 2 tokens after it that its hash also covers.
vllm/mooncake_cpu {vllm}vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/worker.py#L1180-L1199
    Mooncake store connector: medium lowercase 'cpu', one 256-token chunk per event (block_size 256), group_idx set, no extra_keys, no locality or ownership.
vllm/umbp_cpu_local {vllm}vllm/distributed/kv_transfer/kv_connector/v1/umbp/scheduler.py#L1340-L1385
    UMBP connector: medium 'CPU' with locality 'LOCAL', a store and a removal of another block, each one hash with group_idx; no extra_keys or kv_cache_spec_kind.
vllm/ownership_kvcr {vllm}vllm/distributed/kv_transfer/kv_connector/v1/offloading/events.py#L561-L588
    Offloading connector, KVCR secondary tier: ownership 'kvcr' on the store and the removal, medium CPU, no locality ({vllm}vllm/v1/kv_offload/tiering/kvcr/manager.py#L693-L708).
vllm/locality_remote {vllm}vllm/distributed/kv_transfer/kv_connector/v1/offloading/events.py#L561-L588
    Offloading connector, object-store tier configured REMOTE: locality 'REMOTE' on the store and the removal, medium STORAGE ({vllm}vllm/v1/kv_offload/tiering/obj/manager.py#L422-L431).
vllm/offloading_placeholder {vllm}vllm/distributed/kv_transfer/kv_connector/v1/offloading/events.py#L490-L559
    Offloading connector placeholder (no captured request metadata): one hash, parent_block_hash null, token_ids empty, block_size 0, medium CPU, group_idx only.
vllm/medium_nvme_unknown {vllm}vllm/distributed/kv_events.py#L56-L68
    Synthetic: medium 'NVME', which no vLLM producer emits; medium is a free string in vLLM's schema, so the event decodes.
vllm/medium_storage {vllm}vllm/distributed/kv_transfer/kv_connector/v1/offloading/events.py#L68-L71
    Offloading connector, file-system tier with no locality configured: medium STORAGE.
vllm/extra_unknown_field {vllm}vllm/distributed/kv_events.py#L56-L98
    Synthetic: stored_root with one key no vLLM version defines, x_unknown_field = {nested: [1, 'two', null]}, patched into the end of the encoded event map (its map16 count raised by one). msgspec decoders skip unknown keys.
sglang/stored_root {sglang_emit}python/sglang/srt/mem_cache/events.py#L213-L271
    Root store: parent_block_hash null, two pages coalesced into one event, signed i64 hashes (both negative here), medium GPU. lora_id is always null; cache_salt, session_id and lora_name are left out when unset.
sglang/stored_child {sglang_emit}python/sglang/srt/mem_cache/events.py#L180-L197
    Continuation of stored_root: parent_block_hash is stored_root's last page hash (negative), the new page's hash is positive.
sglang/startup_all_cleared {sglang_emit}python/sglang/srt/mem_cache/radix_cache.py#L348-L367
    The batch an SGLang publisher sends first (seq 0): the radix cache resets while it is constructed, which queues a lone AllBlocksCleared ({sglang_emit}python/sglang/srt/mem_cache/radix_cache.py#L322).
sglang/eagle_bigram {sglang_emit}python/sglang/srt/mem_cache/events.py#L245-L249
    EAGLE speculative decoding (bigram radix keys): each token_ids entry is a pair [t_j, t_j+1], 8 pairs for two pages of 4. SGLang's own typed decoder rejects this (its token_ids is list[int]).
sglang/cache_salt_every_page {sglang_emit}python/sglang/srt/mem_cache/events.py#L256-L268
    cache_salt 'salt-abc' on every stored event of a salted request, continuations included: a root of two pages, then a child of its last page. Their session_id differs, which keeps the recorder from coalescing them ({sglang_emit}python/sglang/srt/mem_cache/events.py#L140-L170).
sglang/hicache_cpu_pinned {sglang_emit}python/sglang/srt/mem_cache/unified_cache/unified_tree_core.py#L3186-L3202
    HiCache: blocks stored on GPU, written through to host memory (medium CPU_PINNED, the value of StorageMedium.CPU), then evicted from GPU. The host copy remains.
sglang/declared_disk_external {sglang}python/sglang/srt/disaggregation/kv_events.py#L261-L267
    Declared, not emitted: StorageMedium defines DISK and EXTERNAL, but no code at 72f72cca stores or removes with them (the Rust radix tree's medium has only Gpu and Cpu, {sglang_emit}rust/sglang-radix-tree/src/unified_tree_core.rs#L6389-L6402). The events are what record_store would queue with those media.
sglang/dp_rank_1 {sglang}python/sglang/srt/disaggregation/kv_events.py#L463-L468
    Batch from attention-DP rank 1. SGLang names the third batch element attn_dp_rank; the publisher fills it (0 in every other SGLang sample).
atom/startup_clear {atom}atom/distributed/kv_events.py#L523-L549
    Seq 0 of every ATOM stream: make_publisher publishes a lone AllBlocksCleared before the engine's first batch (new in #3275), kept for replay.
atom/stored_root {atom}atom/model_engine/block_manager.py#L2465-L2474
    Root store from the block manager: parent_block_hash null, two blocks of 4 with xxh64 chain hashes (u64, one >= 2**63), medium GPU, lora_id and lora_name null, and ATOM's own token_offset 0, which vLLM's decoder skips.
atom/stored_child {atom}atom/model_engine/block_manager.py#L2465-L2474
    A longer prompt hits stored_root's two blocks and stores one more: parent_block_hash is stored_root's last hash, token_offset 8.
atom/removed {atom}atom/model_engine/block_manager.py#L615-L627
    Eviction of the three cached blocks for a new prompt: one BlockRemoved per hash, medium GPU, no group_idx.
atom/remote_kv_gpu {atom}atom/model_engine/block_manager.py#L3726-L3750
    KV received from a PD prefill producer (decode side): before #3275 medium REMOTE, now GPU like any block in this engine's HBM. Two blocks after a locally cached one: parent_block_hash set, token_offset 4.
atom/multimodal_extra_keys {atom}atom/model_engine/block_manager.py#L54-L86
    Multimodal prompt: every block carries the media digest in extra_keys as ((mm_identifier, 0),) with mm_identifier '{seed}', the seq's cache_seed in 16 hex digits (new in #3275). The seed also roots the hash chain, so this root event's parent_block_hash is the cache_seed itself, not a block hash.
atom/dp_rank_2 {atom}atom/distributed/kv_events.py#L312-L334
    Batch from data-parallel rank 2 (seq 1, after that publisher's startup clear); ATOM fills data_parallel_rank on every batch, 0 in the other ATOM samples.
atom/main_stored_root {atom}atom/distributed/kv_events.py#L39-L67
    stored_root as ATOM main encodes it before #3275: the same batch re-encoded with that commit's kv_events.py, whose events are array_like. The event is an array: the tag, then all twelve fields in declaration order, defaults included (omit_defaults does not apply to array_like structs); vLLM's decoder rejects it. events lists the fields by name, as that kv_events.py decodes them.
atom/main_removed {atom}atom/distributed/kv_events.py#L70-L75
    removed as ATOM main encodes it before #3275, re-encoded like main_stored_root: each event is [tag, block_hashes, medium, group_idx].
"""


def vllm_structs() -> types.SimpleNamespace:
    # https://github.com/vllm-project/vllm/blob/3a104103ac70740824599453c7036edcca1e8548/vllm/v1/core/kv_cache_utils.py#L58-L61
    ExternalBlockHash: TypeAlias = bytes | int

    # https://github.com/vllm-project/vllm/blob/3a104103ac70740824599453c7036edcca1e8548/vllm/distributed/kv_events.py#L31-L39
    class EventBatch(msgspec.Struct, array_like=True, omit_defaults=True, gc=False):
        ts: float
        events: list[Any]
        data_parallel_rank: int | None = None

    # https://github.com/vllm-project/vllm/blob/3a104103ac70740824599453c7036edcca1e8548/vllm/distributed/kv_events.py#L42-L48
    class KVCacheEvent(msgspec.Struct, omit_defaults=True, gc=False, tag=True): ...

    # https://github.com/vllm-project/vllm/blob/3a104103ac70740824599453c7036edcca1e8548/vllm/distributed/kv_events.py#L56-L120
    class BlockStored(KVCacheEvent):
        block_hashes: list[ExternalBlockHash]
        parent_block_hash: ExternalBlockHash | None
        token_ids: list[int]
        block_size: int
        lora_id: int | None
        medium: str | None
        lora_name: str | None
        extra_keys: list[tuple[Any, ...] | None] | None = None
        group_idx: int | None = None
        kv_cache_spec_kind: str | None = None
        kv_cache_spec_sliding_window: int | None = None
        locality: str | None = None
        ownership: str | None = None
        # No sample sets it, but with 16 possible keys msgspec writes a map16.
        session_id: str | None = None
        lookahead_token_ids: list[list[int]] | None = None

    # https://github.com/vllm-project/vllm/blob/3a104103ac70740824599453c7036edcca1e8548/vllm/distributed/kv_events.py#L123-L141
    class BlockRemoved(KVCacheEvent):
        block_hashes: list[ExternalBlockHash]
        medium: str | None
        group_idx: int | None = None
        locality: str | None = None
        ownership: str | None = None

    # https://github.com/vllm-project/vllm/blob/3a104103ac70740824599453c7036edcca1e8548/vllm/distributed/kv_events.py#L144-L145
    class AllBlocksCleared(KVCacheEvent): ...

    # https://github.com/vllm-project/vllm/blob/3a104103ac70740824599453c7036edcca1e8548/vllm/distributed/kv_events.py#L148-L149
    class KVEventBatch(EventBatch):
        events: list[BlockStored | BlockRemoved | AllBlocksCleared]

    return types.SimpleNamespace(**locals())


def sglang_structs() -> types.SimpleNamespace:
    # https://github.com/sgl-project/sglang/blob/0c44fbff44da7d1a11f0e8b4299e4b8a702ac414/python/sglang/srt/disaggregation/kv_events.py#L233-L240
    class EventBatch(msgspec.Struct, array_like=True, gc=False):
        ts: float
        events: list[Any]
        attn_dp_rank: int | None = None

    # https://github.com/sgl-project/sglang/blob/0c44fbff44da7d1a11f0e8b4299e4b8a702ac414/python/sglang/srt/disaggregation/kv_events.py#L243-L258
    class KVCacheEvent(msgspec.Struct, omit_defaults=True, gc=False, tag=True): ...

    # https://github.com/sgl-project/sglang/blob/0c44fbff44da7d1a11f0e8b4299e4b8a702ac414/python/sglang/srt/disaggregation/kv_events.py#L279-L295
    class BlockStored(KVCacheEvent):
        block_hashes: list[int]
        parent_block_hash: int | None
        token_ids: list[int]
        block_size: int
        lora_id: int | None
        medium: str | None = None
        cache_salt: str | None = None
        session_id: str | None = None

    # https://github.com/sgl-project/sglang/blob/0c44fbff44da7d1a11f0e8b4299e4b8a702ac414/python/sglang/srt/disaggregation/kv_events.py#L298-L300
    class BlockRemoved(KVCacheEvent):
        block_hashes: list[int]
        medium: str | None = None

    # https://github.com/sgl-project/sglang/blob/0c44fbff44da7d1a11f0e8b4299e4b8a702ac414/python/sglang/srt/disaggregation/kv_events.py#L303-L304
    class AllBlocksCleared(KVCacheEvent): ...

    # https://github.com/sgl-project/sglang/blob/0c44fbff44da7d1a11f0e8b4299e4b8a702ac414/python/sglang/srt/disaggregation/kv_events.py#L343-L344
    class KVEventBatch(EventBatch):
        events: list[BlockStored | BlockRemoved | AllBlocksCleared]

    return types.SimpleNamespace(**locals())


def vllm_hashes(*labels: str, as_bytes: bool = False) -> list:
    """Stand-ins for vLLM's block hashes, the sha256 digests of `vllm/<label>`;
    unless `as_bytes`, each as `maybe_convert_block_hash` publishes it with the
    default VLLM_KV_EVENTS_USE_INT_BLOCK_HASHES=1 (kv_cache_utils.py#L86-L89)."""
    digests = [hashlib.sha256(f"vllm/{label}".encode()).digest() for label in labels]
    return digests if as_bytes else [int.from_bytes(d[-8:], "big") for d in digests]


def sglang_hashes(*labels: str) -> list[int]:
    """Stand-ins for SGLang's page hashes: `hash_str_to_int64` of a sha256 hex
    digest, a signed 64-bit integer."""
    digests = [hashlib.sha256(f"sglang/{label}".encode()).digest() for label in labels]
    return [int.from_bytes(d[:8], "big", signed=True) for d in digests]


def vllm_payloads(v: types.SimpleNamespace) -> dict[str, bytes]:
    encoder = msgspec.msgpack.Encoder()
    stamps = iter(1763000000.0 + i / 4 for i in range(1000))
    payloads = {}

    def add(name, *events, rank=0):
        batch = v.KVEventBatch(next(stamps), list(events), rank)
        payloads[f"vllm/{name}"] = encoder.encode(batch)

    def gpu(hashes, tokens, parent=None, **fields):
        """A store as the KV cache manager publishes it: an extra_keys entry per
        block, and every event names its group and the group's spec kind."""
        fields = {
            "lora_id": None,
            "medium": "GPU",
            "lora_name": None,
            "extra_keys": [None] * len(hashes),
            "group_idx": 0,
            "kv_cache_spec_kind": "full_attention",
            **fields,
        }
        return v.BlockStored(hashes, parent, tokens, 4, **fields)

    def connector(hashes, tokens, medium, block_size=4, **fields):
        """A store from a KV connector: a root with a group, no extra_keys or kind."""
        fields |= {"lora_id": None, "medium": medium, "lora_name": None}
        return v.BlockStored(hashes, None, tokens, block_size, group_idx=0, **fields)

    def removed(hashes, medium, **fields):
        return v.BlockRemoved(hashes, medium, group_idx=0, **fields)

    root = vllm_hashes("root/0", "root/1")
    tokens = list(range(101, 109))
    child_tokens = list(range(109, 113))
    block = list(range(201, 205))
    add("stored_root", gpu(root, tokens))
    add("stored_child", gpu(vllm_hashes("child/0"), child_tokens, root[-1]))
    raw = vllm_hashes("root/0", "root/1", as_bytes=True)
    raw_child = vllm_hashes("child/0", as_bytes=True)
    add("stored_bytes_hashes", gpu(raw, tokens), gpu(raw_child, child_tokens, raw[-1]))
    add("removed", removed([root[1]], "GPU"), removed([root[0]], "GPU"))
    add("all_cleared", v.AllBlocksCleared())
    add("dp_rank_1", gpu(root, tokens), rank=1)
    salted = vllm_hashes("salt/0", "salt/1")
    add(
        "salt_first_block",
        gpu(salted, tokens, extra_keys=[("salt-abc",), None]),
        gpu(vllm_hashes("salt/2"), child_tokens, salted[-1]),
    )
    lora = {"lora_id": 1, "lora_name": "my-lora"}
    lora_keys = [("my-lora", "salt-abc"), ("my-lora",)]
    lora_hashes = vllm_hashes("lora/0", "lora/1")
    add("lora_salt", gpu(lora_hashes, tokens, extra_keys=lora_keys, **lora))
    a, b = (hashlib.sha256(image).hexdigest() for image in (b"image-a", b"image-b"))
    pad = 151655
    add(
        "multimodal",
        gpu(
            vllm_hashes("mm/0", "mm/1", "mm/2", "mm/3"),
            [*range(100, 105), *[pad] * 5, 105, pad, pad, *range(106, 109)],
            extra_keys=[None, ((a, 1),), ((a, -3), (b, 3)), ((b, -1),)],
        ),
    )
    sliding = gpu(root, tokens, group_idx=1, kv_cache_spec_kind="sliding_window")
    sliding.kv_cache_spec_sliding_window = 128
    add("sliding_window_group", gpu(root, tokens), sliding)
    lookahead = gpu(vllm_hashes("lookahead/0", "lookahead/1"), tokens)
    lookahead.lookahead_token_ids = [[105, 106], [109, 110]]
    add("lookahead_token_ids", lookahead)
    chunk = list(range(1000, 1256))
    add("mooncake_cpu", connector(vllm_hashes("mooncake/0"), chunk, "cpu", 256))
    add(
        "umbp_cpu_local",
        connector(vllm_hashes("umbp/0"), block, "CPU", locality="LOCAL"),
        removed(vllm_hashes("umbp/1"), "CPU", locality="LOCAL"),
    )
    kvcr = vllm_hashes("kvcr/0", "kvcr/1")
    add(
        "ownership_kvcr",
        gpu(kvcr, tokens, medium="CPU", ownership="kvcr"),
        removed(kvcr, "CPU", ownership="kvcr"),
    )
    remote = vllm_hashes("remote/0")
    add(
        "locality_remote",
        gpu(remote, block, medium="STORAGE", locality="REMOTE"),
        removed(remote, "STORAGE", locality="REMOTE"),
    )
    add("offloading_placeholder", connector(vllm_hashes("placeholder/0"), [], "CPU", 0))
    add("medium_nvme_unknown", gpu(vllm_hashes("nvme/0"), block, medium="NVME"))
    storage = vllm_hashes("storage/0", "storage/1")
    add("medium_storage", gpu(storage, tokens, medium="STORAGE"))
    add("extra_unknown_field", gpu(root, tokens))
    plain = payloads["vllm/extra_unknown_field"]
    # The event map (map16, see BlockStored) follows the batch's array header,
    # the float64 ts and the events' array header. The patch bumps its count and
    # appends one entry, which ends the map just before the trailing one-byte rank.
    head = 1 + 9 + 1
    assert plain[head] == 0xDE
    count = (int.from_bytes(plain[head + 1 : head + 3], "big") + 1).to_bytes(2, "big")
    unknown = [
        encoder.encode(x) for x in ("x_unknown_field", {"nested": [1, "two", None]})
    ]
    patched = [plain[: head + 1], count, plain[head + 3 : -1], *unknown, plain[-1:]]
    payloads["vllm/extra_unknown_field"] = b"".join(patched)
    return payloads


def sglang_payloads(s: types.SimpleNamespace) -> dict[str, bytes]:
    encoder = msgspec.msgpack.Encoder()
    stamps = iter(1763000200.0 + i / 4 for i in range(1000))
    payloads = {}

    def add(name, *events, rank=0):
        batch = s.KVEventBatch(next(stamps), list(events), rank)
        payloads[f"sglang/{name}"] = encoder.encode(batch)

    def stored(hashes, tokens, parent=None, medium="GPU", **fields):
        """What `record_store` queues for a radix node: one event per page, pages
        of one chain coalesced, block_size the page size."""
        return s.BlockStored(hashes, parent, tokens, 4, None, medium, **fields)

    root = sglang_hashes("root/0", "root/1")
    tokens = list(range(201, 209))
    child_tokens = list(range(209, 213))
    block = list(range(401, 405))
    add("stored_root", stored(root, tokens))
    add("stored_child", stored(sglang_hashes("child/1"), child_tokens, root[-1]))
    add("startup_all_cleared", s.AllBlocksCleared())
    bigrams = [(t, t + 1) for t in range(301, 309)]
    add("eagle_bigram", stored(sglang_hashes("eagle/0", "eagle/1"), bigrams))
    salted = sglang_hashes("salt/0", "salt/1")
    salted_child = sglang_hashes("salt/2")
    salt = {"cache_salt": "salt-abc"}
    add(
        "cache_salt_every_page",
        stored(salted, tokens, session_id="sess-a", **salt),
        stored(salted_child, child_tokens, salted[-1], session_id="sess-b", **salt),
    )
    add(
        "hicache_cpu_pinned",
        stored(root, tokens),
        stored(root, tokens, medium="CPU_PINNED"),
        s.BlockRemoved(root, "GPU"),
    )
    add(
        "declared_disk_external",
        stored(sglang_hashes("disk/0"), block, medium="DISK"),
        stored(sglang_hashes("external/0"), block, medium="EXTERNAL"),
    )
    add("dp_rank_1", stored(root, tokens), rank=1)
    return payloads


def atom_payloads(atom_root: Path) -> tuple[dict[str, bytes], str]:
    """ATOM's payloads, and the cache_seed of the multimodal sequence."""
    sys.path.insert(0, str(atom_root))
    import numpy as np
    import zmq

    from atom.distributed import kv_events
    from atom.model_engine.block_manager import BlockManager
    from atom.model_engine.prefix_hash import block_hash
    from atom.model_engine.sequence import Sequence
    from atom.sampling_params import SamplingParams

    config = types.SimpleNamespace(
        kv_cache_block_size=4,
        num_kvcache_blocks=4,
        enable_prefix_caching=True,
        decode_context_parallel_size=1,
        kv_events_config=types.SimpleNamespace(enable=True),
    )

    def seq(tokens, **kwargs):
        return Sequence(tokens, 4, sampling_params=SamplingParams(), **kwargs)

    def admit(bm, s, pub):
        n = bm.can_allocate(s)
        bm.allocate(s, n)
        bm.hash_blocks(s, (s.num_blocks - n) * bm.block_size)
        pub.publish(bm.take_events())

    def publisher(endpoint, rank):
        replay_at = f"{endpoint}-replay"
        return kv_events.make_publisher(
            True, "zmq", endpoint, replay_endpoint=replay_at, data_parallel_rank=rank
        )

    def replay(pub, endpoint, rank, count):
        """Every retained payload by seq, requested the way vLLM's subscriber asks."""
        deadline = time.monotonic() + 30
        while pub.stats["sent"] < count and time.monotonic() < deadline:
            time.sleep(0.01)
        dealer = zmq.Context.instance().socket(zmq.DEALER)
        try:
            dealer.connect(kv_events.offset_endpoint(f"{endpoint}-replay", rank))
            dealer.send_multipart([b"", (0).to_bytes(8, "big")])
            payloads = {}
            while dealer.poll(timeout=5000):
                _, _, seq_no, payload = dealer.recv_multipart()
                if seq_no == kv_events.REPLAY_DONE:
                    assert len(payloads) == count, payloads.keys()
                    return payloads
                payloads[int.from_bytes(seq_no, "big")] = payload
            raise TimeoutError(f"no replay end marker from {endpoint}")
        finally:
            dealer.close(linger=0)

    stamps = iter(1763000400.0 + i / 4 for i in range(1000))
    clock = types.SimpleNamespace(time=lambda: next(stamps))
    with mock.patch.object(kv_events, "time", clock):
        pub = publisher("inproc://atom-samples", 0)
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

        pub_dp = publisher("inproc://atom-samples-dp", 2)
        admit(BlockManager(config), seq(list(range(100, 108))), pub_dp)

    try:
        published = replay(pub, "inproc://atom-samples", 0, 6)
        published_dp = replay(pub_dp, "inproc://atom-samples-dp", 2, 2)
    finally:
        pub.shutdown()
        pub_dp.shutdown()
    names = "startup_clear stored_root stored_child removed remote_kv_gpu"
    payloads = {f"atom/{name}": published[i] for i, name in enumerate(names.split())}
    payloads["atom/multimodal_extra_keys"] = published[5]
    payloads["atom/dp_rank_2"] = published_dp[1]
    return payloads, format(mm.cache_seed & (2**64 - 1), "016x")


def atom_main_samples(atom_root: Path, atom: dict[str, bytes]) -> dict[str, tuple]:
    """stored_root and removed re-encoded by the kv_events.py of {atom}, ATOM main
    before #3275, each with its events by field name as that module decodes them."""
    rev = PINS["{atom}"].split("/")[-2]
    show = ["git", "-C", str(atom_root), "show", f"{rev}:atom/distributed/kv_events.py"]
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp, "kv_events.py")
        path.write_bytes(subprocess.check_output(show))
        spec = importlib.util.spec_from_file_location("atom_main_kv_events", path)
        main = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = main
        spec.loader.exec_module(main)
    samples = {}
    for name in ("stored_root", "removed"):
        ts, events, rank = msgspec.msgpack.decode(atom[f"atom/{name}"])
        events = [getattr(main, e.pop("type"))(**e) for e in events]
        payload = msgspec.msgpack.Encoder().encode(main.EventBatch(ts, events, rank))
        decoded = msgspec.msgpack.Decoder(main.EventBatch).decode(payload).events
        by_name = [
            {"type": type(e).__name__, **msgspec.structs.asdict(e)} for e in decoded
        ]
        samples[f"atom/main_{name}"] = (payload, by_name)
    return samples


def sample_table(seed: str) -> dict[str, tuple[str, str]]:
    """SAMPLES as {engine/name: (permalink, note)}."""
    text = SAMPLES.replace("{seed}", seed)
    for placeholder, url in PINS.items():
        text = text.replace(placeholder, url)
    table = {}
    for entry in re.split(r"\n(?=\S)", text.strip()):
        head, *lines = entry.split("\n")
        key, permalink = head.split()
        note = " ".join(line.strip() for line in lines)
        if key.startswith("atom/") and not key.startswith("atom/main_"):
            note = f"{note} {ATOM_PUBLISHED}"
        table[key] = (permalink, note)
    return table


def as_json(value: Any) -> Any:
    if isinstance(value, bytes):
        return {"bytes_hex": value.hex()}
    if isinstance(value, list):
        return [as_json(item) for item in value]
    if isinstance(value, dict):
        return {key: as_json(item) for key, item in value.items()}
    return value


def record(key: str, permalink: str, note: str, payload: bytes, by_name=None) -> str:
    engine, name = key.split("/")
    ts, events, rank = msgspec.msgpack.decode(payload)
    events = events if by_name is None else by_name
    line = {
        "engine": engine,
        "name": name,
        "profile": engine,
        "source_permalink": permalink,
        "note": note,
        "batch": {"ts": ts, "data_parallel_rank": rank},
        "events": as_json(events),
        "payload": payload.hex(),
    }
    return json.dumps(line, separators=(",", ":")) + "\n"


def rejected(payloads: dict[str, bytes], v, s) -> list[str]:
    """Samples their engine's `KVEventBatch` fails to decode (ATOM's as vLLM's)."""
    vllm, sglang = (msgspec.msgpack.Decoder(ns.KVEventBatch) for ns in (v, s))
    failed = []
    for key, payload in payloads.items():
        try:
            (sglang if key.startswith("sglang/") else vllm).decode(payload)
        except msgspec.DecodeError as e:
            print(f"{key}: {e}")
            failed.append(key)
    return failed


def main() -> None:
    script = Path(__file__).resolve()
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--atom-root", type=Path, default=script.parents[5])
    parser.add_argument("--out", type=Path, default=script.with_name("samples.jsonl"))
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    v, s = vllm_structs(), sglang_structs()
    atom, seed = atom_payloads(args.atom_root)
    main_samples = atom_main_samples(args.atom_root, atom)
    atom |= {key: payload for key, (payload, _) in main_samples.items()}
    payloads = {**vllm_payloads(v), **sglang_payloads(s), **atom}
    table = sample_table(seed)
    assert table.keys() == payloads.keys(), table.keys() ^ payloads.keys()
    named = {key: by_name for key, (_, by_name) in main_samples.items()}
    lines = [record(key, *table[key], payloads[key], named.get(key)) for key in table]
    args.out.write_bytes("".join(lines).encode())
    print(f"{len(lines)} samples written to {args.out}")
    rejects = ["sglang/eagle_bigram", *main_samples]
    if args.check and rejected(payloads, v, s) != rejects:
        sys.exit(f"--check: only {' and '.join(rejects)} should fail to decode")


if __name__ == "__main__":
    main()
