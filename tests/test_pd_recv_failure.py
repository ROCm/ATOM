# SPDX-License-Identifier: MIT
"""Minimal CPU reproduction of a failed P/D receive followed by local prefill."""

from types import SimpleNamespace

import numpy as np
import pytest
from conftest import MockConfig

from atom.kv_transfer.disaggregation.aggregator import KVOutputAggregator
from atom.kv_transfer.disaggregation.types import KVConnectorOutput
from atom.model_engine.scheduler import ScheduledBatchOutput, Scheduler
from atom.model_engine.sequence import SequenceStatus
from atom.sampling_params import SamplingParams


class ConsumerConnector:
    """Only replace transport; use the real scheduler and block allocator."""

    is_producer = False
    is_offload = False

    def __init__(self):
        self.receives = []

    def get_num_new_matched_tokens(self, seq):
        remote = bool(seq.kv_transfer_params.get("do_remote_prefill"))
        return (seq.num_prompt_tokens, True) if remote else (0, False)

    def update_state_after_alloc(self, seq):
        if seq.kv_transfer_params.get("do_remote_prefill"):
            self.receives.append(seq.id)
            seq.kv_transfer_params["do_remote_prefill"] = False

    def build_connector_meta(self):
        return None

    def request_finished(self, seq):
        pass


@pytest.mark.parametrize("pool_blocks", [8, 2])
@pytest.mark.parametrize("mtp_k", [0, 4])
def test_failed_receive_recomputes_and_releases_blocks(seq_factory, pool_blocks, mtp_k):
    spec = (
        SimpleNamespace(num_speculative_tokens=mtp_k, use_dspark=lambda: False)
        if mtp_k
        else None
    )
    sched = Scheduler(
        MockConfig(num_kvcache_blocks=pool_blocks, speculative_config=spec)
    )
    connector = sched.kv_connector = ConsumerConnector()
    seq = seq_factory(
        list(range(8)),
        sampling_params=SamplingParams(max_tokens=1),
        kv_transfer_params={
            "do_remote_prefill": True,
            "first_token_id": 123,
            "draft_token_ids": [124, 125, 126, 127],
        },
    )
    sched.add(seq)
    sched.schedule()
    assert seq.status == SequenceStatus.WAITING_FOR_REMOTE_KVS
    assert len(seq.block_table) == 2
    assert sched.block_manager.kv.num_free == pool_blocks - 2

    # A completed, failed receive is the only injected event: no model or data set.
    sched._update_from_kv_xfer_finished(KVConnectorOutput(failed_recving={seq.id}))
    batch, seqs = sched.schedule()
    assert batch.total_tokens_num_prefill == 8
    assert list(seqs) == [seq.id]
    assert not seq.is_first_decode
    assert not seq.output_tokens  # Do not inject an upstream token after failure.
    assert connector.receives == [seq.id]
    assert sched._num_parked_remote_kv == 0
    assert sched.block_manager.kv.num_free == pool_blocks - 2

    # Native serving defers the prefill's sampled token by one output batch.
    sched.postprocess(
        list(seqs.values()),
        ScheduledBatchOutput(
            req_ids=[],
            token_ids=[],
            num_rejected=None,
            num_bonus=None,
            draft_token_ids=None,
            is_deferred_out=True,
        ),
        batch=batch,
    )
    finished = sched.postprocess(
        [],
        ScheduledBatchOutput(
            req_ids=[seq.id],
            token_ids=[(42,)],
            num_rejected=np.asarray([0]),
            num_bonus=np.asarray([0]),
            draft_token_ids=None,
            is_deferred_out=True,
        ),
    )
    assert finished == [seq]
    assert list(seq.completion_token_ids) == [42]
    assert not seq.block_table
    assert sched.block_manager.kv.num_free == pool_blocks


def test_failed_receive_waits_for_all_tp_workers(seq_factory):
    sched = Scheduler(MockConfig(num_kvcache_blocks=8))
    sched.kv_connector = ConsumerConnector()
    seq = seq_factory(list(range(8)), kv_transfer_params={"do_remote_prefill": True})
    sched.add(seq)
    sched.schedule()
    blocks = list(seq.block_table)
    aggregator = KVOutputAggregator(world_size=4)

    # One rank failed; three others may still be writing the reservation.
    pending = aggregator.aggregate(
        [KVConnectorOutput(failed_recving={seq.id})]
        + [KVConnectorOutput() for _ in range(3)]
    )
    sched._update_from_kv_xfer_finished(pending)
    sched.schedule()
    assert seq.status == SequenceStatus.WAITING_FOR_REMOTE_KVS
    assert list(seq.block_table) == blocks
    assert sched.block_manager.kv.num_free == 6

    terminal = aggregator.aggregate(
        [KVConnectorOutput()]
        + [KVConnectorOutput(finished_recving={seq.id}) for _ in range(3)]
    )
    assert terminal.failed_recving == {seq.id}
    sched._update_from_kv_xfer_finished(terminal)
    batch, _ = sched.schedule()
    assert batch.total_tokens_num_prefill == 8
    assert seq.status == SequenceStatus.RUNNING
    assert sched.block_manager.kv.num_free == 6


def test_failed_receive_reuses_only_the_valid_local_prefix(seq_factory):
    sched = Scheduler(MockConfig(num_kvcache_blocks=6, enable_prefix_caching=True))
    connector = sched.kv_connector = ConsumerConnector()
    bm = sched.block_manager
    seed = seq_factory(list(range(9)))
    assert bm.allocate(seed)
    bm.hash_blocks(seed, 8)
    shared = list(seed.block_table[:2])

    seq = seq_factory(
        list(range(8)) + [20, 21, 22, 23],
        sampling_params=SamplingParams(max_tokens=1),
        kv_transfer_params={"do_remote_prefill": True},
    )
    sched.add(seq)
    sched.schedule()
    assert seq.num_cached_tokens == 8
    assert list(seq.block_table[:2]) == shared

    sched._update_from_kv_xfer_finished(KVConnectorOutput(failed_recving={seq.id}))
    batch, seqs = sched.schedule()
    assert batch.total_tokens_num_prefill == 4
    assert seq.num_cached_tokens == 8
    assert list(seq.block_table[:2]) == shared
    assert all(bm.kv.block(block).ref_count == 2 for block in shared)
    assert connector.receives == [seq.id]

    sched.postprocess(
        list(seqs.values()),
        ScheduledBatchOutput(
            req_ids=[seq.id],
            token_ids=[(42,)],
            num_rejected=None,
            num_bonus=None,
            draft_token_ids=None,
        ),
        batch=batch,
    )
    assert all(bm.kv.block(block).ref_count == 1 for block in shared)
    bm.deallocate(seed)
    assert bm.kv.num_free == 6
