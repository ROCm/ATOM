# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""MiniMax-M3.1's prefill node with a UMBP tier, end to end on CPU.

The prefill node of the M3.1 1P1D (``recipes/m31-pd-umbp``) is PP4 x TP1 and
runs ``multi[P/D producer, umbp_store save_policy=lazy]``. One prompt goes all
the way round:

1. it is prefilled and sent to the decode node;
2. HBM, about to evict its blocks, demotes them to the UMBP server. Each stage
   stores its own layers under its own keys, and the demotion holds its blocks
   until the last stage has reported;
3. a filler evicts them;
4. the same prompt comes back. Its lookup finds them past the HBM hit on all
   four stages, each stage loads its layers into new blocks, and the request
   runs on what they hold.

Real:
- the `Scheduler` and `BlockManager`;
- the lazy UMBP scheduler under `multi`;
- the PP head's poll and `PPKVAggregator`, which complete a save or a load
  only once all four stages reported it;
- four `UMBPStoreOffloadConnector` workers. Each registers (codec, transfer
  pool, startup probe) on the real M3.1 PP4 stage publication: the M3
  builder's NVFP4 K/V and FP4 index of the stage's 17/17/17/9 layers, with the
  DSpark draft merged on the last;
- the UMBP client and its calls.

Stood in for:
- the UMBP server (``test_umbp_store_offload.FakeServer``: bytes move by
  address between registered host buffers and a dict);
- the P/D producer: it holds a sent request's blocks until the decode node
  read them, as MoRIIO does;
- the GPU copies. The block GPU connector's staged pack and unpack and the
  PAGE codec's in-place restore are CPU copies that address each region's
  units as the Triton kernels do;
- the step's CUDA fence;
- the forward, which writes every block it completes with bytes its stage and
  prompt prefix determine.

A block the tier gave back holds those bytes only if the right bytes went out
under the right key and came back into the right block on the right stage.

The last stage reports a step after the others, as a downstream stage does. A
prefetched lookup answers within the step that sent it: a UMBP lookup takes
well under a millisecond, a prefill step tens.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from types import SimpleNamespace
from unittest import mock

import pytest
import test_moriio_pp_fabric
import torch
from conftest import MockConfig
from kv_store_test_support import engine_config, use_one_rank
from test_moriio_pp_fabric import PARTITION, rank_pools, stage_layers
from test_offload_lazy_save import _cpu_fences, _multi, _pp_head, _Producer
from test_umbp_store_offload import ADDRESS, FakeServer

from atom.kv_transfer.disaggregation.types import KVConnectorOutput, SaveSourceGroupId
from atom.kv_transfer.offload import atom_lmcache_staging
from atom.kv_transfer.offload.hybrid.dsv4.codec import DSV4PageSlotCodec
from atom.kv_transfer.offload.kv_store import keys
from atom.kv_transfer.offload.kv_store import pool as pool_mod
from atom.kv_transfer.offload.kv_store.keys import DIGEST_BYTES
from atom.kv_transfer.offload.kv_store.scheduler import prompt_chunk_hashes
from atom.kv_transfer.offload.metadata import LMCacheOffloadMetadata
from atom.kv_transfer.offload.umbp_store import client as umbp_client
from atom.kv_transfer.offload.umbp_store import taint
from atom.kv_transfer.offload.umbp_store.scheduler import (
    UMBPLazyStoreScheduler,
    make_umbp_store_scheduler,
)
from atom.kv_transfer.offload.umbp_store.worker import UMBPStoreOffloadConnector
from atom.model_engine.scheduler import ScheduledBatchOutput, Scheduler
from atom.model_engine.sequence import Sequence, SequenceStatus
from atom.sampling_params import SamplingParams

PP = len(PARTITION)  # 4 stages
BLOCK = 128  # M3.1's KV block, and the store chunk under lazy
STEP = 4 * BLOCK  # max_num_batched_tokens
KV_BLOCKS = 10  # HBM, in blocks
PROMPT = 6  # blocks per prompt
DEMOTION = "umbp-lazy-demotion-"

# The M3.1 PP4 stage stack, a module-scoped fixture of the fabric tests.
stack = test_moriio_pp_fabric.stack


@pytest.fixture(autouse=True)
def _process_state(monkeypatch):
    monkeypatch.setattr(taint, "_TAINTED", {})
    monkeypatch.setattr(umbp_client, "_RETAINED_CLIENTS", [])
    monkeypatch.setattr(pool_mod, "_UNSETTLED_ALLOCATIONS", [])
    # PAGE offload needs the fused staging kernels; `_CpuCopies` stands in.
    monkeypatch.setattr(
        DSV4PageSlotCodec, "has_fused_chunk_major_staging", property(lambda _: True)
    )
    monkeypatch.setattr(
        atom_lmcache_staging, "fence_submitted_work", lambda stream: stream.sync()
    )
    # A load of one block is worth it here; the default floor is 8192 tokens.
    monkeypatch.setenv("OFFLOAD_MIN_LOAD_TOKENS", str(BLOCK))
    monkeypatch.setenv("ATOM_OFFLOAD_MISS_ACCOUNTING", "1")
    use_one_rank(monkeypatch)
    _cpu_fences(monkeypatch)
    with mock.patch("torch.cuda.current_device", lambda: 0):
        yield


@pytest.fixture
def server(monkeypatch):
    server = FakeServer()
    monkeypatch.setattr(umbp_client, "_new_umbp_client", server.connect)
    return server


@pytest.fixture
def prefill_node(stack, server, monkeypatch):
    nodes = []

    def build(**kwargs):
        node = _PrefillNode(stack, monkeypatch, **kwargs)
        nodes.append(node)
        return node

    yield build
    for node in nodes:
        node.close()


def _extra(lookup_threads):
    """P's ``umbp_store`` settings, as the recipe's ``P_KV_UMBP``."""
    return {
        "umbp_store.address": ADDRESS,
        "umbp_store.chunk_tokens": BLOCK,
        "umbp_store.save_policy": "lazy",
        "umbp_store.lookup_threads": lookup_threads,
        # Eight padded slots each way: two chunks a window.
        "umbp_store.save_pool_mib": 12,
        "umbp_store.load_pool_mib": 12,
    }


def _config(stage, extra):
    """Stage `stage`'s engine config; the head is stage 0."""
    config = engine_config(
        "umbp_store",
        dict(extra),
        pp=PP,
        layers=60,
        kv_cache_block_size=BLOCK,
        index_cache_dtype="fp4",
        max_num_batched_tokens=STEP,
        parallel_config=SimpleNamespace(pipeline_parallel_rank=stage),
    )
    config.hf_config.model_type = "minimax_m3"
    return config


class _Stream:
    """The forward's stream on the CPU, where a copy is done once issued."""

    device = torch.device("cpu")

    def wait_event(self, event):
        event.synchronize()

    def sync(self):
        pass


class _CpuCopies:
    """A stage's GPU copies on the CPU: the block GPU connector's staged pack
    and unpack, and the PAGE codec's in-place calls.

    Each copy addresses the region units as `triton_page_slot`'s kernels do.
    A block's bytes are its regions' units in registration order, each read
    at the region's ``unit_addr``; the blocks of a chunk, and the chunks of a
    buffer, follow one another. A strided restore starts chunk c at
    ``c * chunk_stride_bytes``. A pack waits for the step's fence and then
    reports its chunks source-safe, as the connector does once a copy has
    fenced.
    """

    has_fused_chunk_major_staging = True
    supports_strided_restore = True

    def __init__(self, stage, grid, codec):
        self._stage = stage
        self._grid = grid  # the real connector: token ranges to block ids
        self.device = torch.device("cpu")
        self.bytes_per_block = codec.bytes_per_block
        self.gpu_staging_chunk_bytes = grid.gpu_staging_chunk_bytes
        self._source_safe = grid._source_safe_callback
        self._operation = None
        self.restored: list[int] = []
        self.strided_restores = 0

    def _ranges_to_block_ids(self, starts, ends, **kwargs):
        return self._grid._ranges_to_block_ids(starts, ends, **kwargs)

    # -- the block GPU connector --------------------------------------------
    @contextmanager
    def track_save_source(self, operation):
        previous, self._operation = self._operation, operation
        try:
            yield
        finally:
            self._operation = previous

    def batched_from_gpu(self, slots, starts, ends, *, block_ids, producer_event):
        producer_event.synchronize()
        groups = self._ranges_to_block_ids(starts, ends, block_ids=block_ids)
        # Tail to head, as the connector packs a save.
        for slot, start, end, blocks in sorted(
            zip(slots, starts, ends, groups, strict=True), key=lambda item: -item[1]
        ):
            self._pack(slot.tensor, blocks)
            if self._operation is not None:
                self._source_safe(SaveSourceGroupId(self._operation, ((start, end),)))

    def batched_to_gpu(self, slots, starts, ends, *, block_ids):
        groups = self._ranges_to_block_ids(starts, ends, block_ids=block_ids)
        for slot, blocks in zip(slots, groups, strict=True):
            self._unpack(slot.tensor, blocks)

    def close(self):
        self._grid.close()

    # -- the PAGE codec's in-place calls -------------------------------------
    def prepare_block_id_groups(self, groups, *, device, stream):
        assert stream is not None
        return SimpleNamespace(groups=groups)

    def gpu_to_chunk_major_device_buffer_prepared(self, buf, owner, index, *, stream):
        self._pack(buf, [block for chunk in owner.groups[index] for block in chunk])

    def chunk_major_device_buffer_to_gpu_prepared(self, buf, owner, index, *, stream):
        self._unpack(buf, [block for chunk in owner.groups[index] for block in chunk])

    def chunk_major_strided_buffer_to_gpu_prepared(
        self, buf, owner, index, *, stream, items_per_chunk, chunk_stride_bytes
    ):
        self.strided_restores += 1
        width = items_per_chunk * self.bytes_per_block
        for c, blocks in enumerate(owner.groups[index]):
            assert len(blocks) == items_per_chunk
            start = c * chunk_stride_bytes
            self._unpack(buf[start : start + width], blocks)

    # -- bytes ---------------------------------------------------------------
    def _pack(self, buf, blocks):
        width = self.bytes_per_block
        assert buf.numel() >= len(blocks) * width
        for i, block in enumerate(blocks):
            buf[i * width : (i + 1) * width].copy_(self._stage.read(block))

    def _unpack(self, buf, blocks):
        width = self.bytes_per_block
        for i, block in enumerate(blocks):
            self._stage.write(block, buf[i * width : (i + 1) * width])
            self.restored.append(block)


class _Stage:
    """One PP stage: its M3.1 KV publication and the umbp_store worker
    registered on it, with `_CpuCopies` for its GPU copies."""

    def __init__(self, stack, stage, extra):
        self.index = stage
        side = rank_pools(
            stack, stage_layers(stage), 1, stage == PP - 1, KV_BLOCKS, "kv_producer"
        )
        self.kv = side.pool  # every region of the stage lies in it
        self.worker = UMBPStoreOffloadConnector(_config(stage, extra))
        self.worker.register_kv_caches({}, side.tt, num_blocks=KV_BLOCKS)
        codec = self.worker._codec
        base = self.kv.data_ptr()
        self.units = [(r.unit_addr(0) - base, r.unit_bytes) for r in codec.page_regions]
        self.bytes_per_block = codec.bytes_per_block
        self.copies = _CpuCopies(self, self.worker._gpu_connector, codec)
        self.worker._gpu_connector = self.worker._codec = self.copies
        self.worker.set_forward_stream(_Stream())
        self._images: dict[tuple, torch.Tensor] = {}

    def image(self, prefix: tuple) -> torch.Tensor:
        """What this stage's forward leaves in the block that ends `prefix`."""
        image = self._images.get(prefix)
        if image is None:
            generator = torch.Generator().manual_seed(
                hash((self.index, prefix)) & 0x7FFF_FFFF
            )
            image = torch.randint(
                0, 256, (self.bytes_per_block,), dtype=torch.uint8, generator=generator
            )
            self._images[prefix] = image
        return image

    def read(self, block: int) -> torch.Tensor:
        return torch.cat(
            [self.kv[off + block * n : off + (block + 1) * n] for off, n in self.units]
        )

    def write(self, block: int, data: torch.Tensor) -> None:
        at = 0
        for off, n in self.units:
            self.kv[off + block * n : off + (block + 1) * n] = data[at : at + n]
            at += n


def _prefix(seq, chunk: int) -> tuple:
    return tuple(seq.token_ids[: (chunk + 1) * BLOCK])


class _PrefillNode:
    """The PP4 prefill node: the head's scheduler and poll, and four stages."""

    def __init__(self, stack, monkeypatch, *, lookup_threads):
        extra = _extra(lookup_threads)
        self.stages = [_Stage(stack, stage, extra) for stage in range(PP)]
        self.sched = make_umbp_store_scheduler(_config(0, extra))
        assert type(self.sched) is UMBPLazyStoreScheduler
        self.producer = _Producer()
        self.connector = _multi(self.producer, self.sched)
        self.engine = Scheduler(
            MockConfig(
                enable_prefix_caching=True,
                num_kvcache_blocks=KV_BLOCKS,
                kv_cache_block_size=BLOCK,
                max_num_batched_tokens=STEP,
                max_model_len=10**6,
            )
        )
        self.engine.kv_connector = self.connector
        self.connector.bind_block_manager(self.engine.block_manager)
        self.head, self.reports = _pp_head(
            monkeypatch, self.connector, scheduler=self.engine
        )
        self.namespace = self.stages[0].worker._namespace
        self.loads, self.saves = [], []
        # The last stage's report of the step before.
        self._late = KVConnectorOutput()

    def close(self):
        for stage in self.stages:
            stage.worker.close()
        self.sched._lookup_client.close()

    def key(self, stage: int, digest: bytes) -> str:
        [key] = keys.chunk_keys(self.namespace, stage, PP, digest, 0, 1)
        return key

    # -- one engine step ---------------------------------------------------
    def step(self) -> None:
        result = self.engine.schedule()
        if result is None or result[0] is None:
            idle = self.connector.has_pending_work()
            meta = self.connector.build_connector_meta() if idle else None
        else:
            batch, seqs = result
            self._forward(batch, seqs)
            self.engine.postprocess(
                list(seqs.values()),
                ScheduledBatchOutput(
                    req_ids=list(seqs),
                    token_ids=[(7,)] * len(seqs),
                    num_rejected=None,
                    num_bonus=None,
                    draft_token_ids=None,
                ),
                batch=batch,
            )
            meta = batch.connector_meta_output
        for sub in getattr(meta, "metas", None) or ([meta] if meta else []):
            if isinstance(sub, LMCacheOffloadMetadata):
                self._run(sub)
        self._report()
        self._answer_lookups()

    def _forward(self, batch, seqs):
        """Write every block the step completes, on every stage."""
        for seq, n in zip(seqs.values(), batch.num_scheduled_tokens, strict=True):
            lo = seq.num_cached_tokens
            for j in range(lo // BLOCK, (lo + n) // BLOCK):
                for stage in self.stages:
                    stage.write(seq.block_table[j], stage.image(_prefix(seq, j)))

    def _run(self, meta):
        """Every stage takes the step's metadata and runs its copies."""
        for req in meta.requests:
            if req.load_spec is not None:
                self.loads.append(req)
                self._garble(req)
            if req.save_spec is not None:
                self.saves.append(req)
        for stage in self.stages:
            stage.worker.start_load_kv(meta)
            stage.worker.wait_for_requests([req.req_id for req in meta.requests])

    def _garble(self, load):
        """A load's blocks hold another prompt's bytes until it fills them."""
        spec = load.load_spec
        first, end = (
            spec.hbm_cached_tokens // BLOCK,
            spec.lmcache_cached_tokens // BLOCK,
        )
        for block in load.block_ids[first:end]:
            for stage in self.stages:
                stage.write(block, torch.full_like(stage.image(()), 0xA5))

    def _report(self):
        """Hand the head each stage's report; the last stage's comes a step late."""
        outputs = [stage.worker.get_finished() for stage in self.stages]
        outputs[-1], self._late = self._late, outputs[-1]
        for stage, output in zip(self.stages, outputs, strict=True):
            self.reports.post(stage.index, output)
        self.head._poll_kv_transfer_progress()

    def _answer_lookups(self):
        """Wait for the lookup thread to answer what this step prefetched."""
        lookup = self.sched._lookup_client
        deadline = time.monotonic() + 10.0
        while any(lookup.is_pending(sid) for sid in lookup.pending_ids()):
            assert time.monotonic() < deadline, "a prefetched lookup never answered"
            time.sleep(0.001)

    # -- requests --------------------------------------------------------------
    def serve(self, seq):
        self.engine.add(seq)
        for _ in range(100):
            if seq.status == SequenceStatus.FINISHED:
                return
            self.step()
        raise AssertionError("the request never finished")

    def settle(self):
        """Step until no demotion, load or late report is left."""
        for _ in range(50):
            if not self.connector.has_pending_work() and self._late.is_empty():
                return
            self.step()
        raise AssertionError("demotions did not settle")

    def prefill(self, seq):
        """Admit `seq` and step until its whole prompt is computed."""
        self.engine.add(seq)
        for _ in range(100):
            if seq.status == SequenceStatus.RUNNING and (
                seq.num_cached_tokens >= seq.num_prompt_tokens
            ):
                return
            self.step()
        raise AssertionError("the prompt was never computed")

    def holds(self, seq) -> bool:
        """Whether each prompt block of `seq` holds, on every stage, what the
        forward of its prefix left there."""
        return all(
            torch.equal(stage.read(seq.block_table[j]), stage.image(_prefix(seq, j)))
            for j in range(seq.num_prompt_tokens // BLOCK)
            for stage in self.stages
        )


def _seq(n_blocks, base=0, max_tokens=1, **kwargs):
    return Sequence(
        [base + i for i in range(n_blocks * BLOCK)],
        BLOCK,
        sampling_params=SamplingParams(max_tokens=max_tokens),
        **kwargs,
    )


def _digests(seq) -> list[bytes]:
    hashes = prompt_chunk_hashes(seq, BLOCK)
    return [hashes[i : i + DIGEST_BYTES] for i in range(0, len(hashes), DIGEST_BYTES)]


def _sent_and_evicted(node):
    """`first`, prefilled and sent, its blocks demoted, then evicted by a
    filler. Returns `first`."""
    first = _seq(PROMPT, kv_transfer_params={"do_remote_decode": True})
    node.serve(first)
    node.engine._update_from_kv_xfer_finished(
        KVConnectorOutput(finished_sending={first.id})
    )
    node.settle()
    node.serve(_seq(PROMPT, base=100_000))
    node.settle()
    return first


# --------------------------------------------------------------------- tests
@pytest.mark.parametrize("lookup_threads", [1, 0], ids=["prefetched", "sync"])
def test_a_prompt_hbm_evicted_comes_back_from_umbp_on_every_stage(
    prefill_node, server, lookup_threads
):
    node = prefill_node(lookup_threads=lookup_threads)
    # The head and the stages key a chunk alike.
    assert node.sched._lookup_client._namespace == node.namespace
    assert {s.worker._namespace for s in node.stages} == {node.namespace}
    probes = set(server.at())
    assert len(probes) == PP  # each stage's startup probe

    # 1. Prefilled and sent: the producer holds the blocks, nothing is stored.
    first = _seq(PROMPT, kv_transfer_params={"do_remote_decode": True})
    node.serve(first)
    assert first.id in node.engine.deferred_free_blocks
    assert node.saves == [] and set(server.at()) == probes
    node.engine._update_from_kv_xfer_finished(
        KVConnectorOutput(finished_sending={first.id})
    )

    # 2. HBM demotes the coldest blocks. Every stage stores them in the step,
    # but a demotion holds its blocks until the last stage's report is in.
    node.step()
    demotions = list(node.saves)
    assert demotions and all(d.req_id.startswith(DEMOTION) for d in demotions)
    pool = node.engine.block_manager.kv
    for demotion in demotions:
        assert demotion.save_operation in node.sched._demotions
        assert all(pool.is_used(block) for block in demotion.block_ids)
    node.step()
    assert not any(d.save_operation in node.sched._demotions for d in demotions)
    node.settle()
    assert all(save.req_id.startswith(DEMOTION) for save in node.saves)
    # The prompt's four coldest blocks, its last freed: what HBM's eviction
    # window holds past its vacant blocks. Its head stays in HBM. Each
    # demoted chunk is one object per stage, under that stage's key, holding
    # that stage's layers of the chunk; the last stage's carries the DSpark
    # draft too.
    digests = _digests(first)
    demoted = {
        digests.index(save.chunk_hashes[i : i + DIGEST_BYTES])
        for save in node.saves
        for i in range(0, len(save.chunk_hashes), DIGEST_BYTES)
    }
    assert demoted == {2, 3, 4, 5}
    sizes = {stage.index: stage.worker._chunk_bytes for stage in node.stages}
    assert sizes[PP - 1] > sizes[0]
    for chunk in demoted:
        for stage in node.stages:
            stored = server.at()[node.key(stage.index, digests[chunk])]
            assert len(stored) == sizes[stage.index]
            assert stored == stage.image(_prefix(first, chunk)).numpy().tobytes()

    # 3. A filler evicts them from HBM.
    node.serve(_seq(PROMPT, base=100_000))
    node.settle()

    # 4. The prompt again: HBM serves its head, UMBP the rest past it, and its
    # last block is computed again.
    again = _seq(PROMPT, max_tokens=2)
    node.prefill(again)
    (load,) = node.loads
    hbm = load.load_spec.hbm_cached_tokens
    end = load.load_spec.lmcache_cached_tokens
    assert 0 < hbm < end == (PROMPT - 1) * BLOCK
    loaded = load.block_ids[hbm // BLOCK : end // BLOCK]
    for stage in node.stages:
        # Into the load's blocks, in place over the padded slots.
        assert stage.copies.restored == loaded
        assert stage.copies.strided_restores >= 1
    assert node.holds(again)

    stats = node.sched.get_statistics()
    assert stats["load_failures"] == 0
    assert stats["loaded_tokens"] == end - hbm
    # The prompt: computed before, but for its last block.
    assert stats["miss_hbm_tokens"] == hbm
    assert stats["miss_tier_tokens"] == end - hbm
    assert stats["miss_tier_not_loaded_tokens"] == stats["miss_evicted_tokens"] == 0
    assert stats["miss_new_tokens"] == 2 * PROMPT * BLOCK + BLOCK
    assert stats["lazy_evicted_unsaved"] == stats["lazy_demotions_failed"] == 0
    if lookup_threads:
        assert stats["umbp_lookups_prefetched"] == stats["umbp_prefetch_answers_used"]
        assert stats["umbp_prefetch_answers_used"] >= 3  # one per prompt


def test_a_load_one_stage_cannot_read_is_computed_again(prefill_node, server):
    node = prefill_node(lookup_threads=1)
    first = _sent_and_evicted(node)
    # Stage 2's copy of the prompt's fifth chunk cannot be read: the load
    # fails as a whole once every stage reported, and the request computes
    # what it was to load.
    server.fail_gets.add(node.key(2, _digests(first)[4]))
    again = _seq(PROMPT, max_tokens=2)
    node.prefill(again)
    (load,) = node.loads
    hbm = load.load_spec.hbm_cached_tokens
    end = load.load_spec.lmcache_cached_tokens
    assert hbm < 5 * BLOCK <= end
    assert node.holds(again)
    stats = node.sched.get_statistics()
    assert stats["load_failures"] == 1 and stats["loads_pending"] == 0
    assert stats["miss_tier_tokens"] == 0
    assert stats["miss_tier_not_loaded_tokens"] == end - hbm


def test_a_chunk_one_stage_lost_is_loaded_from_no_stage(prefill_node, server):
    node = prefill_node(lookup_threads=1)
    first = _sent_and_evicted(node)
    # The server lost stage 3's copy of the prompt's fourth chunk (a put that
    # failed there, or an eviction of that object alone): the hit ends before
    # it on every stage.
    del server.at()[node.key(3, _digests(first)[3])]
    again = _seq(PROMPT, max_tokens=2)
    node.prefill(again)
    (load,) = node.loads
    assert load.load_spec.lmcache_cached_tokens == 3 * BLOCK
    assert node.holds(again)
    stats = node.sched.get_statistics()
    assert stats["load_failures"] == 0 and stats["umbp_uneven_lookups"] >= 1
    assert stats["loaded_tokens"] == 3 * BLOCK - load.load_spec.hbm_cached_tokens
