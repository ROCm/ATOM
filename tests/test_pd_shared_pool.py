# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Mooncake P/D over a pool the ranks share: `multi` of a `mooncake` leg and
the `umbp_pool` leg, as DeepSeek-V4.1 runs it.

The prefill engine is the plain case: a producer admits a request like any
engine, so a prefix another pool holds is copied in, hit, and its state rebuilt
by replay. The decode engine is not. A request with per-request state is
received whole -- every block of its table is written by the producer -- so in
a pool other pools read it must take no prefix hit: a claimed hit is published
blocks rewritten under whoever is copying them out.

Each test runs against the Python model of the UMBP service and again against
the real one when `mori_umbp_l1` is importable (`test_umbp_l1_pool.l1`).
"""

import logging
import threading
import time
from types import SimpleNamespace

import pytest
import test_umbp_l1_pool as umbp_tests
from aiter_stub import stubbed_aiter
from conftest import MockConfig
from test_umbp_l1_pool import count_fetches, land, make_pool, settle

from atom.kv_transfer.disaggregation.base import KVConnectorSchedulerBase
from atom.kv_transfer.disaggregation.multi.multi_connector import (
    MultiConnectorScheduler,
    validate_multi_config,
)
from atom.kv_transfer.disaggregation.types import KVConnectorOutput
from atom.kv_transfer.pool.umbp import connector as umbp_connector
from atom.kv_transfer.pool.umbp import pool as umbp_pool
from atom.kv_transfer.pool.umbp.pool import UmbpBlockPool
from atom.model_engine.block_manager import BlockManager
from atom.model_engine.block_pool import BlockPool
from atom.model_engine.kv_block import STATE_SLOT_CLASS
from atom.model_engine.prefill_delayer import PrefillDecision
from atom.model_engine.prefix_hash import chained_block_hashes
from atom.model_engine.scheduler import ScheduledBatchOutput, Scheduler
from atom.model_engine.sequence import SequenceStatus
from atom.model_engine.state_runtime import StateRuntime, StateTransfer
from atom.sampling_params import SamplingParams

l1 = umbp_tests.l1

BLOCK = 4
# A prompt of 7 blocks, 6 of them a prefix a walk can hit.
TOKENS = list(range(26))
ADDRESS = "unix:///tmp/umbp-l1/server.sock"
# DeepSeek-V4.1 under encoder SWA bounded replay: a STATE slot a request, and
# no checkpoint of it.
REPLAY = StateRuntime(transfer=StateTransfer.replay())


def pool_leg(namespace, **extra):
    return {
        "kv_connector": "umbp_pool",
        "kv_connector_extra_config": {
            "address": ADDRESS,
            "namespace": namespace,
            **extra,
        },
    }


def pd_leg(role):
    return {"kv_connector": "mooncake", "kv_role": role}


def multi(role, namespace="atom", **extra):
    return {
        "kv_connector": "multi",
        "connectors": [pd_leg(role), pool_leg(namespace, **extra)],
    }


@pytest.fixture
def mooncake(monkeypatch):
    with stubbed_aiter():
        from atom.kv_transfer.disaggregation.mooncake import mooncake_connector
    monkeypatch.setattr(mooncake_connector, "get_open_port", lambda: 30000)
    monkeypatch.setattr(mooncake_connector, "get_ip", lambda: "127.0.0.1")
    return mooncake_connector


@pytest.fixture
def engine(monkeypatch, mooncake, l1):
    """Schedulers of engines attached to one UMBP service, by transfer config."""
    # The stand-in config is no model the pool lays out; that refusal has its
    # own test (`test_umbp_pool_connector`).
    monkeypatch.setattr(
        umbp_connector,
        "_SUPPORTED_ARCHITECTURES",
        umbp_connector._SUPPORTED_ARCHITECTURES + ("LlamaForCausalLM",),
    )

    def make_l1_pool(config, settings, num_blocks, *, on_evict=None, on_remap=None):
        pool = make_pool(
            l1,
            umbp_pool.pool_id_for(config, settings),
            num_blocks,
            on_evict=on_evict,
            on_remap=on_remap,
        )
        pool.shares_content = settings.fetch
        return pool

    monkeypatch.setattr(umbp_pool, "make_umbp_l1_pool", make_l1_pool)

    def make(kv_transfer_config, **overrides):
        config = MockConfig(
            num_kvcache_blocks=32,
            kv_cache_block_size=BLOCK,
            enable_prefix_caching=True,
            max_num_seqs=8,
            tensor_parallel_size=1,
            parallel_config=SimpleNamespace(data_parallel_size=1, data_parallel_rank=0),
            pool_entries={STATE_SLOT_CLASS: 8},
            pool_entries_per_req={STATE_SLOT_CLASS: 1},
            kv_transfer_config=kv_transfer_config,
        )
        for name, value in overrides.items():
            setattr(config, name, value)
        monkeypatch.setattr(
            "atom.utils.forward_context.get_kvconnector",
            lambda role, cfg: (
                MultiConnectorScheduler(cfg)
                if cfg.kv_transfer_config.get("kv_connector") == "multi"
                else mooncake.MooncakeConnectorScheduler(cfg)
            ),
        )
        return Scheduler(config, state_runtime=REPLAY)

    return make


@pytest.fixture
def received(seq_factory):
    """A request the router sends a decode engine after its prefill."""

    def make(tokens, **params):
        seq = seq_factory(list(tokens), block_size=BLOCK, has_per_req_cache=True)
        seq.kv_transfer_params = {
            "do_remote_prefill": True,
            "transfer_id": f"p-{seq.id}",
            "remote_block_ids": list(range(-(-len(tokens) // BLOCK))),
            "block_size": BLOCK,
            "dcp_size": 1,
            "remote_host": "127.0.0.1",
            "remote_port": 30000,
            "remote_handshake_port": 30000,
            "tp_size": 1,
            "first_token_id": 42,
            **params,
        }
        return seq

    return make


class IdleGroupDelayer:
    """The delayer of a DP group whose other ranks have nothing to run: its
    reduce carries this rank's own report and nothing from the peers."""

    is_local = False
    max_queue_ms = None
    kv_high_watermark = 0.9
    rank_prefill_costs: tuple = ()

    def protects_decode(self, running_decode_batch):
        return False

    def should_allow_prefill(self, *, prefillable, running_decode_batch, **_):
        # Nothing to admit anywhere is a vacuous FIRE; with no decode to hide
        # a wait behind, a prefillable rank fires too.
        return PrefillDecision(
            allow=True,
            prefillable_ranks=int(prefillable),
            running_decodes=running_decode_batch,
        )

    def notify_prefill_executed(self, *args): ...


class BusyGroupDelayer(IdleGroupDelayer):
    """The delayer of a DP group under load: peers prefill and decode, so a
    pass ends in their step whatever this rank does."""

    def should_allow_prefill(self, *, prefillable, running_decode_batch, **_):
        return PrefillDecision(
            allow=True,
            prefillable_ranks=int(prefillable) + 2,
            running_decodes=running_decode_batch + 5,
        )


class HeldCopies:
    """A pool's client whose copies into held slots take as long as the test
    says, or fail. Everything else goes straight through."""

    def __init__(self, client, fail=False):
        self._client = client
        self.fail = fail
        self.land = threading.Event()
        self.started = threading.Event()

    def __getattr__(self, name):
        return getattr(self._client, name)

    def fetch(self, *args, **kwargs):
        if kwargs.get("dst_slots"):
            self.started.set()
            if self.fail:
                raise RuntimeError("copy failed")
            assert self.land.wait(10), "the test never let the copy land"
        return self._client.fetch(*args, **kwargs)


def umbp_pool_hashes(bm: BlockManager, seq) -> list[int]:
    """The hashes of `seq`'s prefix a walk can hit: every block but the last."""
    n = (len(seq) + BLOCK - 1) // BLOCK - 1
    return chained_block_hashes(seq.token_ids, BLOCK, 0, n, seq.cache_seed)


def fetch_lines(caplog) -> list[str]:
    return [
        record.getMessage()
        for record in caplog.records
        if "from peer pools in the background" in record.getMessage()
    ]


def compute(bm: BlockManager, seq, release=True) -> list[int]:
    """`seq` prefilled on `bm`'s rank: its full blocks published and, once the
    device has got there, readable by other pools."""
    assert bm.allocate(seq, 0)
    bm.hash_blocks(seq, len(seq))
    blocks = list(seq.block_table)
    if release:
        bm.deallocate(seq)
    if isinstance(bm.kv, UmbpBlockPool):
        settle(bm.kv.umbp)
    return blocks


def recv_meta(batch, seq):
    return batch.connector_meta_output.metas[0].reqs_to_recv[seq.id]


class TestAdmittedAtStartup:
    @pytest.mark.parametrize("role", ["kv_producer", "kv_consumer"])
    def test_the_composite_passes_multis_own_refusals(self, role):
        # `umbp_pool` is no offload tier: the rule about a P/D consumer beside
        # one (a received prefill resumed as a tier load) does not apply.
        assert len(validate_multi_config(multi(role))) == 2

    def test_a_consumer_beside_an_offload_tier_is_still_refused(self):
        config = multi("kv_consumer")
        config["connectors"].append({"kv_connector": "mooncake_store"})
        with pytest.raises(ValueError, match="P/D consumer with an offload tier"):
            validate_multi_config(config)

    @pytest.mark.parametrize(
        "role, producer", [("kv_producer", True), ("kv_consumer", False)]
    )
    def test_the_pool_is_umbps_and_the_role_is_mooncakes(self, engine, role, producer):
        sched = engine(multi(role))
        assert isinstance(sched.block_manager.kv, UmbpBlockPool)
        assert sched.block_manager.kv.read_by_other_pools
        assert sched.kv_connector.is_producer is producer
        # Not an offload tier: a completed receive goes to its first decode.
        assert not sched.kv_connector.is_offload
        # A producer hands off after its first token and never speculates.
        assert sched.pd_admission is None

    def test_a_pool_that_copies_nothing_in_is_still_read(self, engine, received):
        # `fetch: false` stops this engine asking; the others still copy out
        # of its pool, so a receive is kept off published blocks all the same.
        sched = engine(multi("kv_consumer", "d", fetch=False))
        kv = sched.block_manager.kv
        assert not kv.shares_content and kv.read_by_other_pools
        assert sched.block_manager._lands_whole_in_shared_pool(received(range(20)))


class TestWhichReceivesRewriteAPrefix:
    def connector(self, mooncake, role):
        return mooncake.MooncakeConnectorScheduler(
            MockConfig(
                kv_cache_block_size=BLOCK,
                tensor_parallel_size=1,
                parallel_config=SimpleNamespace(
                    data_parallel_size=1, data_parallel_rank=0
                ),
                kv_transfer_config=pd_leg(role),
            )
        )

    def test_a_request_with_state_is_received_whole(self, mooncake, received):
        consumer = self.connector(mooncake, "kv_consumer")
        seq = received(range(20))
        assert consumer.receive_rewrites_prefix(seq)
        # The same answer the worker is handed after allocation.
        seq.block_table = [0, 1, 2, 3, 4]
        seq.num_cached_tokens = 12
        consumer.update_state_after_alloc(seq)
        assert seq.kv_transfer_params["num_computed_blocks"] == 0
        # Told of the allocation, the request is no longer about to receive.
        assert not consumer.receive_rewrites_prefix(seq)

    def test_paged_kv_alone_skips_the_prefix_it_holds(self, mooncake, received):
        consumer = self.connector(mooncake, "kv_consumer")
        seq = received(range(20))
        seq.has_per_req_cache = False
        assert not consumer.receive_rewrites_prefix(seq)
        # ... unless the two sides' pages do not line up.
        assert consumer.receive_rewrites_prefix(received(range(20), block_size=8))

    def test_only_a_request_about_to_receive(self, mooncake, received, seq_factory):
        consumer = self.connector(mooncake, "kv_consumer")
        local = seq_factory(list(range(20)), block_size=BLOCK, has_per_req_cache=True)
        assert not consumer.receive_rewrites_prefix(local)
        producer = self.connector(mooncake, "kv_producer")
        assert not producer.receive_rewrites_prefix(received(range(20)))

    def test_other_connectors_receive_nothing(self):
        assert not KVConnectorSchedulerBase.receive_rewrites_prefix(None, object())

    @pytest.mark.parametrize("role", ["kv_producer", "kv_consumer"])
    def test_multi_asks_its_pd_leg(self, engine, received, role):
        sched = engine(multi(role))
        asked = sched.kv_connector.receive_rewrites_prefix(received(range(20)))
        assert asked is (role == "kv_consumer")


class TestDecodeEngine:
    """`multi[mooncake kv_consumer, umbp_pool]`."""

    @pytest.fixture
    def decode(self, engine):
        return engine(multi("kv_consumer", "d"))

    def test_a_receive_takes_no_prefix_and_fresh_blocks(
        self, decode, engine, received, seq_factory
    ):
        bm = decode.block_manager
        # The previous turn left its blocks cached here, and a prefill rank
        # holds the whole prompt.
        mine = compute(bm, seq_factory(TOKENS[:16], block_size=BLOCK))
        prefill = engine(multi("kv_producer", "p")).block_manager
        compute(prefill, seq_factory(TOKENS, block_size=BLOCK))
        asked = []
        count_fetches(bm, asked)

        seq = received(TOKENS)
        decode.add(seq)
        batch, _ = decode.schedule()

        assert seq.status == SequenceStatus.WAITING_FOR_REMOTE_KVS
        # Nothing copied in from the prefill rank: it would be overwritten.
        assert asked == [] and not bm.kv.fetch_pending(seq.id)
        # No hit on the four blocks cached here, which stay as published and
        # are not this request's.
        assert seq.num_cached_tokens == 0
        assert len(seq.block_table) == 7
        assert not set(seq.block_table) & set(mine)
        assert not any(bm.kv.is_used(block_id) for block_id in mine)
        # The producer is asked for every block, into those fresh ones.
        meta = recv_meta(batch, seq)
        assert meta.num_computed_blocks == 0
        assert list(meta.local_block_ids) == list(seq.block_table)

    def test_what_arrived_is_published_once_and_decodes(
        self, decode, received, seq_factory
    ):
        bm = decode.block_manager
        mine = compute(bm, seq_factory(TOKENS[:16], block_size=BLOCK))
        seq = received(TOKENS)
        decode.add(seq)
        decode.schedule()
        decode._update_from_kv_xfer_finished(
            KVConnectorOutput(finished_recving={seq.id})
        )
        batch, _ = decode.schedule()

        # Its first decode, on the STATE slot the producer filled: no prefill
        # forward ran here, so nothing replays over it.
        assert seq.id in batch.req_ids and batch.total_tokens_num_prefill == 0
        assert seq.replay is None and not batch.replaying
        assert seq.state_slot >= 0
        assert seq.token_ids[len(TOKENS)] == 42
        # The two blocks nobody held here are now this pool's content; the
        # four already cached keep their one indexed copy.
        chain = [bm.kv.block(block_id).hash for block_id in seq.block_table[:6]]
        assert -1 not in chain
        assert [bm.kv.lookup(h) for h in chain[:4]] == mine
        assert [bm.kv.lookup(h) for h in chain[4:]] == list(seq.block_table[4:6])

    def test_parked_for_its_receive_it_still_copies_nothing(
        self, decode, engine, received, seq_factory
    ):
        bm = decode.block_manager
        # The prefill rank that ran this request holds its whole prompt.
        prefill = engine(multi("kv_producer", "p")).block_manager
        compute(prefill, seq_factory(TOKENS, block_size=BLOCK))
        asked = []
        count_fetches(bm, asked)
        seq = received(TOKENS)
        decode.add(seq)
        decode.schedule()
        # Told of the allocation, the request no longer reads as a receive to
        # come, and admission asks about a prefix copy before it looks at the
        # receive: the passes it waits through must not start one.
        assert not seq.kv_transfer_params["do_remote_prefill"]
        for _ in range(3):
            decode.schedule()
        assert seq.status == SequenceStatus.WAITING_FOR_REMOTE_KVS
        assert asked == [] and not bm.shared_prefix_pending(seq)

        decode._update_from_kv_xfer_finished(
            KVConnectorOutput(finished_recving={seq.id})
        )
        batch, _ = decode.schedule()
        # Decoding on the pass its receive resolved, not held back for a copy.
        assert seq.id in batch.req_ids
        assert asked == [] and bm.kv.umbp.blocks_fetched == 0
        # What it received is this pool's one copy of the prompt.
        chain = [bm.kv.block(block_id).hash for block_id in seq.block_table[:6]]
        assert [bm.kv.lookup(h) for h in chain] == list(seq.block_table[:6])

    def test_another_rank_reads_what_the_receive_published(
        self, decode, engine, received, seq_factory
    ):
        seq = received(TOKENS)
        decode.add(seq)
        decode.schedule()
        decode._update_from_kv_xfer_finished(
            KVConnectorOutput(finished_recving={seq.id})
        )
        decode.schedule()
        # The device has run the forward issued after the publish.
        settle(decode.block_manager.kv.umbp)

        # The next turn's prefill, on a prefill rank that never saw the prompt.
        prefill = engine(multi("kv_producer", "p")).block_manager
        nxt = seq_factory(TOKENS + [100, 101], block_size=BLOCK)
        assert prefill.shared_prefix_pending(nxt)
        land(prefill, nxt)
        assert prefill.can_allocate(nxt) == 6
        assert prefill.kv.umbp.blocks_fetched == 6

    def test_preempted_after_its_receive_it_prefills_from_the_pool(
        self, decode, received
    ):
        bm = decode.block_manager
        seq = received(TOKENS)
        decode.add(seq)
        decode.schedule()
        decode._update_from_kv_xfer_finished(
            KVConnectorOutput(finished_recving={seq.id})
        )
        decode.schedule()
        assert decode.preempt(seq)
        # No longer a receive: the blocks it published are a hit like any.
        assert not bm._lands_whole_in_shared_pool(seq)
        assert bm.can_allocate(seq) == 6

    def test_a_request_sent_here_directly_still_copies_its_prefix(
        self, decode, engine, seq_factory
    ):
        bm = decode.block_manager
        prefill = engine(multi("kv_producer", "p")).block_manager
        compute(prefill, seq_factory(TOKENS, block_size=BLOCK))
        seq = seq_factory(TOKENS, block_size=BLOCK, has_per_req_cache=True)
        assert bm.shared_prefix_pending(seq)
        land(bm, seq)
        assert bm.can_allocate(seq) == 6

    def test_requests_that_ask_for_nothing_are_not_counted_as_copies(
        self, decode, received
    ):
        bm = decode.block_manager
        seq = received(TOKENS)
        # `_early_copy_shared_prefixes` stops at a number of copies a tick.
        assert not bm.start_shared_prefix_copy(seq)
        assert not bm.start_shared_prefix_copy(seq)

    def test_the_engines_own_pool_keeps_its_hit(
        self, engine, received, seq_factory, monkeypatch
    ):
        # Mooncake alone, ATOM's pool: only this process reads its blocks, as
        # before, and the receive lands on the prefix the request claimed.
        sched = engine(pd_leg("kv_consumer"))
        bm = sched.block_manager
        assert isinstance(bm.kv, BlockPool) and not bm.kv.read_by_other_pools
        mine = compute(bm, seq_factory(TOKENS[:16], block_size=BLOCK))
        seq = received(TOKENS)
        sched.add(seq)
        sched.schedule()
        assert seq.status == SequenceStatus.WAITING_FOR_REMOTE_KVS
        assert seq.num_cached_tokens == 16
        assert list(seq.block_table[:4]) == mine


class TestPrefillEngine:
    """`multi[mooncake kv_producer, umbp_pool]`."""

    @pytest.fixture
    def prefill(self, engine):
        return engine(multi("kv_producer", "p"))

    def request(self, seq_factory, tokens):
        # As the router sends it: one token, then the decode engine's.
        seq = seq_factory(
            list(tokens),
            block_size=BLOCK,
            has_per_req_cache=True,
            sampling_params=SamplingParams(max_tokens=1),
        )
        seq.kv_transfer_params = {"do_remote_decode": True}
        return seq

    def test_a_prefix_a_decode_rank_holds_is_copied_in_and_replayed(
        self, engine, seq_factory, monkeypatch
    ):
        # The copy indexed at the top of the next pass (no wait within one).
        monkeypatch.setenv("ATOM_POOL_COPY_INPASS_WAIT_MS", "0")
        prefill = engine(multi("kv_producer", "p"))
        bm = prefill.block_manager
        # A decode rank holds the previous turn: its prompt and what it
        # generated.
        decode = engine(multi("kv_consumer", "d")).block_manager
        compute(decode, seq_factory(TOKENS[:24], block_size=BLOCK))

        seq = self.request(seq_factory, TOKENS)
        prefill.add(seq)
        # Held back while the copy is in flight, then admitted on the hit.
        _, admitted = prefill.schedule()
        assert not admitted
        assert seq.status == SequenceStatus.WAITING and not seq.block_table
        land(bm, seq)
        batch, seqs = prefill.schedule()

        assert seq.id in seqs and batch.total_tokens_num_prefill > 0
        assert bm.kv.umbp.blocks_fetched == 6
        assert seq.prefix_cache_hit_tokens == 24
        # The STATE the consumer will be sent is rebuilt on this producer: the
        # forward starts one block below the hit and resumes at it.
        assert seq.replay is not None
        assert (seq.replay.start, seq.replay.until) == (20, 24)
        assert seq.replay.rebuilds
        assert seq.num_cached_tokens == 20

    def test_in_an_idle_group_the_copy_and_the_prefill_are_one_pass(
        self, engine, seq_factory, monkeypatch
    ):
        # Under DP attention a pass that schedules nothing ends in a dummy step
        # on every rank, and on a producer that step is slow. A rank whose only
        # work is mid-copy, in a group with nothing else to run, waits for the
        # copy instead (`Scheduler._await_arriving_copies`). Without the wait
        # every pass has, which would land the copy first.
        monkeypatch.setenv("ATOM_POOL_COPY_INPASS_WAIT_MS", "0")
        prefill = engine(multi("kv_producer", "p"))
        prefill.set_prefill_delayer(IdleGroupDelayer())
        bm = prefill.block_manager
        decode = engine(multi("kv_consumer", "d")).block_manager
        compute(decode, seq_factory(TOKENS[:24], block_size=BLOCK))

        seq = self.request(seq_factory, TOKENS)
        prefill.add(seq)
        batch, seqs = prefill.schedule()

        # The first pass: copy started, landed, indexed, hit and replayed.
        assert seq.id in seqs and batch.total_tokens_num_prefill > 0
        assert bm.kv.umbp.blocks_fetched == 6
        assert seq.prefix_cache_hit_tokens == 24
        assert (seq.replay.start, seq.replay.until) == (20, 24)

    # --- the copy's hand-off: indexed in the pass that asked for it ---------

    def loaded_producer(self, engine, monkeypatch, wait_ms, delayer=True):
        """A prefill rank of a group under load, with or without the prefill
        delayer (a P/D recipe may run without it: ATOM_ENABLE_PREFILL_DELAYER=0)."""
        monkeypatch.setenv("ATOM_POOL_COPY_INPASS_WAIT_MS", str(wait_ms))
        prefill = engine(multi("kv_producer", "p"))
        if delayer:
            prefill.set_prefill_delayer(BusyGroupDelayer())
        return prefill

    def previous_turn(self, engine, seq_factory):
        decode = engine(multi("kv_consumer", "d")).block_manager
        compute(decode, seq_factory(TOKENS[:24], block_size=BLOCK))

    @pytest.mark.parametrize("delayer", [True, False])
    def test_a_copy_is_indexed_and_its_request_scheduled_in_the_pass_that_asked(
        self, engine, seq_factory, monkeypatch, caplog, delayer
    ):
        prefill = self.loaded_producer(engine, monkeypatch, 20, delayer=delayer)
        bm = prefill.block_manager
        self.previous_turn(engine, seq_factory)
        seq = self.request(seq_factory, TOKENS)
        prefill.add(seq)

        with caplog.at_level(logging.INFO, logger="atom"):
            batch, seqs = prefill.schedule()

        # One pass: copy started, waited for, indexed, hit and replayed.
        assert seq.id in seqs and batch.total_tokens_num_prefill > 0
        assert seq.prefix_cache_hit_tokens == 24
        assert (seq.replay.start, seq.replay.until) == (20, 24)
        # Every block of the hit is one the copy filled and the pool indexed.
        chain = [bm.kv.block(block_id).hash for block_id in seq.block_table[:6]]
        assert -1 not in chain
        assert [bm.kv.lookup(h) for h in chain] == list(seq.block_table[:6])
        assert bm.kv.umbp.blocks_fetched == 6
        assert (bm.kv.umbp.fetches_indexed, bm.kv.umbp.fetches_indexed_same_pass) == (
            1,
            1,
        )
        (line,) = fetch_lines(caplog)
        assert "ms until indexed), same pass, " in line
        assert line.endswith(" ms to ask")

    def test_switched_off_the_copy_is_indexed_by_the_next_pass(
        self, engine, seq_factory, monkeypatch
    ):
        prefill = self.loaded_producer(engine, monkeypatch, wait_ms=0)
        bm = prefill.block_manager
        self.previous_turn(engine, seq_factory)
        seq = self.request(seq_factory, TOKENS)
        prefill.add(seq)

        _, seqs = prefill.schedule()
        # Held back for the whole pass, with nothing of its own.
        assert not seqs and not seq.block_table
        assert seq.status == SequenceStatus.WAITING
        land(bm, seq)  # the test's own polls stand in for the step
        _, seqs = prefill.schedule()
        assert seq.id in seqs and seq.prefix_cache_hit_tokens == 24
        assert bm.kv.umbp.fetches_indexed_same_pass == 0

    def test_the_log_says_how_many_passes_the_hand_off_took(
        self, engine, seq_factory, monkeypatch, caplog
    ):
        prefill = self.loaded_producer(engine, monkeypatch, wait_ms=0)
        bm = prefill.block_manager
        self.previous_turn(engine, seq_factory)
        seq = self.request(seq_factory, TOKENS)
        prefill.add(seq)
        prefill.schedule()
        deadline = time.monotonic() + 10
        while not bm.kv.wait_fetch(seq.id, 0.01):
            assert time.monotonic() < deadline

        with caplog.at_level(logging.INFO, logger="atom"):
            prefill.schedule()

        (line,) = fetch_lines(caplog)
        assert "fetched 6 of 6 blocks from peer pools in the background" in line
        assert "ms until indexed), 1 pass later" in line

    def test_a_copy_that_outlasts_the_wait_leaves_its_request_where_it_was(
        self, engine, seq_factory, monkeypatch, caplog
    ):
        prefill = self.loaded_producer(engine, monkeypatch, wait_ms=20)
        bm = prefill.block_manager
        self.previous_turn(engine, seq_factory)
        copies = HeldCopies(bm.kv.umbp._client)
        monkeypatch.setattr(bm.kv.umbp, "_client", copies)
        seq = self.request(seq_factory, TOKENS)
        prefill.add(seq)
        wanted = umbp_pool_hashes(bm, seq)

        waits = []
        wait_fetch = bm.kv.wait_fetch
        monkeypatch.setattr(
            bm.kv,
            "wait_fetch",
            lambda owner, timeout_s: (
                waits.append(timeout_s),
                wait_fetch(owner, timeout_s),
            )[1],
        )
        _, seqs = prefill.schedule()

        # The pass gave the copy its 20 ms and went on without the request:
        # not scheduled, no blocks of its own, nothing of its prefix indexed.
        assert copies.started.is_set()
        assert waits[0] == pytest.approx(0.02, abs=0.01)
        assert not seqs and not seq.block_table
        assert seq.status == SequenceStatus.WAITING
        assert bm.kv.fetch_pending(seq.id)
        assert all(bm.kv.lookup(h) == -1 for h in wanted)

        # Still in flight a pass later: looked at, not waited for again.
        del waits[:]
        _, seqs = prefill.schedule()
        assert waits and all(timeout_s <= 0 for timeout_s in waits)
        assert not seqs and not seq.block_table

        # It lands during a step; the next pass's top indexes it, as without
        # the wait, and only then is the request scheduled on it.
        copies.land.set()
        deadline = time.monotonic() + 10
        while not wait_fetch(seq.id, 0.01):
            assert time.monotonic() < deadline
        assert all(bm.kv.lookup(h) == -1 for h in wanted)  # landed, not indexed
        with caplog.at_level(logging.INFO, logger="atom"):
            _, seqs = prefill.schedule()
        assert seq.id in seqs and seq.prefix_cache_hit_tokens == 24
        assert [bm.kv.lookup(h) for h in wanted] == list(seq.block_table[:6])
        (line,) = fetch_lines(caplog)
        assert "ms until indexed), 2 passes later" in line

    @pytest.mark.parametrize("wait_ms", [0, 20])
    def test_a_copy_that_fails_is_a_miss_with_or_without_the_wait(
        self, engine, seq_factory, monkeypatch, wait_ms
    ):
        prefill = self.loaded_producer(engine, monkeypatch, wait_ms=wait_ms)
        bm = prefill.block_manager
        self.previous_turn(engine, seq_factory)
        copies = HeldCopies(bm.kv.umbp._client, fail=True)
        monkeypatch.setattr(bm.kv.umbp, "_client", copies)
        seq = self.request(seq_factory, TOKENS)
        prefill.add(seq)
        wanted = umbp_pool_hashes(bm, seq)

        _, seqs = prefill.schedule()
        if not wait_ms:
            # As before: held back for the pass, the failure seen by the next.
            assert not seqs
            deadline = time.monotonic() + 10
            while not bm.kv.wait_fetch(seq.id, 0.01):
                assert time.monotonic() < deadline
            _, seqs = prefill.schedule()

        # Scheduled on nothing it did not compute: no hit, no replay, and none
        # of the copy's blocks indexed or left held.
        assert seq.id in seqs
        assert seq.num_cached_tokens == 0 and seq.prefix_cache_hit_tokens == 0
        assert seq.replay is None
        assert all(bm.kv.lookup(h) == -1 for h in wanted)
        assert bm.kv.num_used == len(seq.block_table)
        assert bm.kv.umbp.blocks_fetched == 0

    def test_its_blocks_are_published_and_handed_off(
        self, prefill, engine, seq_factory
    ):
        bm = prefill.block_manager
        seq = self.request(seq_factory, TOKENS)
        prefill.add(seq)
        batch, seqs = prefill.schedule()
        assert list(seqs) == [seq.id]
        prefill.postprocess(
            list(seqs.values()),
            ScheduledBatchOutput([seq.id], [(42,)], None, None, None),
            batch=batch,
        )

        # Handed off with its blocks still held for the send.
        out = seq.kv_transfer_params_output
        assert out["do_remote_prefill"] and out["first_token_id"] == 42
        assert out["remote_block_ids"] == list(seq.block_table[:7])
        assert prefill.kv_connector.should_defer_free(seq)
        assert all(bm.kv.is_used(block_id) for block_id in seq.block_table)
        # Published as the forward's postprocess ran; another rank reads them
        # once this rank's device has passed that point.
        decode = engine(multi("kv_consumer", "d")).block_manager
        nxt = seq_factory(TOKENS + [100, 101], block_size=BLOCK)
        assert not decode.shared_prefix_pending(nxt)
        settle(bm.kv.umbp)
        later = seq_factory(TOKENS + [100, 102], block_size=BLOCK)
        assert decode.shared_prefix_pending(later)
        land(decode, later)
        assert decode.can_allocate(later) == 6

    def test_a_producer_does_not_speculate(self, engine):
        spec = SimpleNamespace(
            num_speculative_tokens=5,
            method="dspark",
            model=None,
            use_dspark=lambda: True,
        )
        producer = engine(multi("kv_producer", "p"), speculative_config=spec)
        assert not producer.use_spec and producer.mtp_k == 0


class TestShareGroup:
    """Both engines have to land in one group, each rank under its own name."""

    def config(self, namespace, dp_rank, **overrides):
        fields = {
            "model": "/data/models/DeepSeek-V4.1-Flash",
            "kv_cache_dtype": "bf16",
            "index_cache_dtype": "fp4",
            "kv_cache_block_size": 128,
            "quant_config": SimpleNamespace(online_quant_config_raw=None),
            "speculative_config": SimpleNamespace(method="dspark", model=None),
            "parallel_config": SimpleNamespace(
                data_parallel_size=4, data_parallel_rank=dp_rank
            ),
            "tensor_parallel_size": 1,
        }
        fields.update(overrides)
        return SimpleNamespace(**fields)

    def settings(self, role, namespace, **extra):
        leaf = multi(role, namespace, **extra)["connectors"][1]
        return umbp_pool.UmbpPoolSettings.from_transfer_config(leaf)

    def test_prefill_and_decode_ranks_share_one_group(self):
        p = self.settings("kv_producer", "p")
        d = self.settings("kv_consumer", "d", device_offset=4)
        groups = {
            umbp_pool.share_group_for(self.config(ns, rank), s)
            for ns, s in (("p", p), ("d", d))
            for rank in range(4)
        }
        assert len(groups) == 1
        names = [
            umbp_pool.pool_id_for(self.config(ns, rank), s)
            for ns, s in (("p", p), ("d", d))
            for rank in range(4)
        ]
        assert names == [f"{ns}:dp{rank}:tp0" for ns in "pd" for rank in range(4)]
        assert (p.device_offset, d.device_offset) == (0, 4)

    @pytest.mark.parametrize(
        "field, value",
        [
            # The same checkpoint under another path is another group.
            ("model", "/data/models/DeepSeek-V4.1-Flash/"),
            ("kv_cache_dtype", "fp8"),
            ("index_cache_dtype", "fp8"),
            ("kv_cache_block_size", 64),
            # A prefill engine started without the draft.
            ("speculative_config", None),
        ],
    )
    def test_a_flag_one_engine_leaves_out_splits_them(self, field, value):
        p = self.settings("kv_producer", "p")
        d = self.settings("kv_consumer", "d", device_offset=4)
        prefill = umbp_pool.share_group_for(self.config("p", 0), p)
        decode = umbp_pool.share_group_for(self.config("d", 0, **{field: value}), d)
        assert prefill != decode

    def test_a_named_group_overrides_the_derivation(self):
        p = self.settings("kv_producer", "p", group="v41")
        d = self.settings("kv_consumer", "d", group="v41")
        other = self.config("d", 0, kv_cache_dtype="fp8")
        assert umbp_pool.share_group_for(self.config("p", 0), p) == "v41"
        assert umbp_pool.share_group_for(other, d) == "v41"
