# SPDX-License-Identifier: MIT

"""KV RPC timeouts must preserve destructive worker completion reports."""

import queue
from unittest.mock import Mock, call

import pytest
from aiter_stub import stubbed_aiter

with stubbed_aiter():
    from atom.kv_transfer.disaggregation import (
        ConnectorCompletion,
        KVConnectorOutput,
        SaveOperationId,
    )
    from atom.model_engine.async_proc import AsyncIOProcManager


_POLL = "async_proc_aggregation"
_STORE = "dense.page.store"
_RETIRED = "dense.page.retired"


def _manager(world_size=2):
    # Exercise the actual manager method and aggregator without worker
    # processes, ZMQ threads, shared memory, or GPU initialization.
    manager = AsyncIOProcManager.__new__(AsyncIOProcManager)
    manager.label = "test-kv-manager"
    manager.proc_num = world_size
    manager.rpc_broadcast_mq = Mock()
    manager.kv_output_aggregator = None
    manager.kv_outputs_queues = [queue.Queue() for _ in range(world_size)]
    manager._pending_kv_aggregation = None
    return manager


def _report(operation, *channels, succeeded=True):
    return KVConnectorOutput(
        finished_saving={operation} if _RETIRED in channels else set(),
        connector_completions={
            ConnectorCompletion(
                channel,
                operation,
                succeeded if channel == _STORE else True,
            )
            for channel in channels
        },
    )


def _poll(manager):
    # Non-blocking collection; timeout=0 also suppresses HOL logging.
    return manager.call_func_with_aggregation(_POLL, timeout=0)


@pytest.mark.parametrize("world_size", [2, 8])
@pytest.mark.parametrize("req_id", [0, "0"])
@pytest.mark.parametrize("generation", [0, 1])
def test_timeout_keeps_consumed_reports_until_all_workers_reply(
    world_size, req_id, generation
):
    manager = _manager(world_size)
    operation = SaveOperationId(req_id, generation)
    manager.kv_outputs_queues[0].put(_report(operation, _STORE, _RETIRED))

    assert _poll(manager) is None
    assert manager.kv_outputs_queues[0].empty()
    for _ in range(2):
        assert _poll(manager) is None
    manager.rpc_broadcast_mq.enqueue.assert_called_once_with((_POLL,))

    for output_queue in manager.kv_outputs_queues[1:]:
        output_queue.put(_report(operation, _STORE, _RETIRED))
    result = _poll(manager)

    assert result is not None
    assert result.finished_saving == {operation}
    assert (
        result.connector_completions
        == _report(operation, _STORE, _RETIRED).connector_completions
    )
    assert manager.kv_output_aggregator.pending_count == (0, 0)
    assert manager._pending_kv_aggregation is not None
    assert manager.rpc_broadcast_mq.enqueue.call_count == 2

    # The next aggregation was already armed before the completed result was
    # returned. Already-consumed completions must not appear in that batch.
    for output_queue in manager.kv_outputs_queues:
        output_queue.put(KVConnectorOutput())
    assert _poll(manager).is_empty()
    assert manager._pending_kv_aggregation is not None
    assert manager.rpc_broadcast_mq.enqueue.call_count == 3
    assert all(output_queue.empty() for output_queue in manager.kv_outputs_queues)


def test_timeout_before_first_reply_does_not_repeat_broadcast():
    manager = _manager()
    operation = SaveOperationId(0, 0)
    manager.kv_outputs_queues[1].put(_report(operation, _STORE, _RETIRED))

    assert _poll(manager) is None
    # Rank 1 can be consumed while rank 0 is still missing.
    assert manager.kv_outputs_queues[1].empty()
    assert manager._pending_kv_aggregation is not None
    assert manager._pending_kv_aggregation.worker_outputs[1] is not None
    assert manager._pending_kv_aggregation.missing_worker_ranks() == [0]
    assert _poll(manager) is None
    manager.rpc_broadcast_mq.enqueue.assert_called_once_with((_POLL,))

    manager.kv_outputs_queues[0].put(_report(operation, _STORE, _RETIRED))
    result = _poll(manager)

    assert result is not None
    assert result.finished_saving == {operation}
    assert len(result.connector_completions) == 2
    assert manager._pending_kv_aggregation is not None
    assert manager.rpc_broadcast_mq.enqueue.call_count == 2


@pytest.mark.parametrize("first_channel", [_STORE, _RETIRED])
@pytest.mark.parametrize("store_succeeded", [True, False])
def test_timeout_preserves_separate_store_and_retired_quorums(
    first_channel, store_succeeded
):
    manager = _manager()
    operation = SaveOperationId(0, 0)
    second_channel = _RETIRED if first_channel == _STORE else _STORE
    results = []

    for poll_count, channel in enumerate((first_channel, second_channel), start=1):
        # A failure from the delayed rank must still dominate the store
        # result. Retirement has its own independent success quorum.
        manager.kv_outputs_queues[0].put(_report(operation, channel))
        assert _poll(manager) is None
        manager.kv_outputs_queues[1].put(
            _report(operation, channel, succeeded=store_succeeded)
        )
        result = _poll(manager)

        assert result is not None
        expected = _report(operation, channel, succeeded=store_succeeded)
        assert result.connector_completions == expected.connector_completions
        assert result.finished_saving == expected.finished_saving
        assert manager.rpc_broadcast_mq.enqueue.call_count == poll_count + 1
        results.append(result)

    assert set().union(*(result.connector_completions for result in results)) == (
        _report(
            operation, _STORE, _RETIRED, succeeded=store_succeeded
        ).connector_completions
    )
    assert manager.kv_output_aggregator.pending_count == (0, 0)


def test_resumed_batches_keep_worker_identity_and_reply_order():
    manager = _manager()
    operations = (SaveOperationId(0, 0), SaveOperationId(0, 1))

    # Each rank reports a different generation in this batch. Neither has a
    # quorum; consuming a second reply from rank 0 after a timeout must not
    # count that reply as rank 1 or skip rank 1's older reply.
    manager.kv_outputs_queues[0].put(_report(operations[0], _STORE, _RETIRED))
    assert _poll(manager) is None
    manager.kv_outputs_queues[1].put(_report(operations[1], _STORE, _RETIRED))
    first = _poll(manager)
    assert first is not None and first.is_empty()
    assert manager.rpc_broadcast_mq.enqueue.call_count == 2

    # The opposite ranks now report the missing generations. This second
    # batch must consume exactly one reply from each original rank.
    manager.kv_outputs_queues[0].put(_report(operations[1], _STORE, _RETIRED))
    assert _poll(manager) is None
    manager.kv_outputs_queues[1].put(_report(operations[0], _STORE, _RETIRED))
    second = _poll(manager)

    assert second is not None
    assert second.finished_saving == set(operations)
    assert second.connector_completions == set().union(
        *(
            _report(operation, _STORE, _RETIRED).connector_completions
            for operation in operations
        )
    )
    assert manager.rpc_broadcast_mq.enqueue.call_count == 3
    assert manager.kv_output_aggregator.pending_count == (0, 0)
    assert manager._pending_kv_aggregation is not None


def test_later_ranks_are_consumed_while_an_earlier_rank_is_missing():
    manager = _manager(world_size=8)
    operation = SaveOperationId(0, 0)
    for rank in (3, 7, 1):
        manager.kv_outputs_queues[rank].put(_report(operation, _STORE, _RETIRED))

    assert _poll(manager) is None
    pending = manager._pending_kv_aggregation
    assert pending is not None
    assert pending.missing_worker_ranks() == [0, 2, 4, 5, 6]
    for rank in (1, 3, 7):
        assert pending.worker_outputs[rank] is not None
        assert manager.kv_outputs_queues[rank].empty()
    manager.rpc_broadcast_mq.enqueue.assert_called_once_with((_POLL,))

    for rank in pending.missing_worker_ranks():
        manager.kv_outputs_queues[rank].put(_report(operation, _STORE, _RETIRED))
    result = _poll(manager)
    assert result is not None
    assert result.finished_saving == {operation}
    assert manager._pending_kv_aggregation is not None
    assert manager.rpc_broadcast_mq.enqueue.call_count == 2


def test_completed_aggregation_arms_the_next_batch_before_returning():
    manager = _manager(world_size=2)
    operation = SaveOperationId(0, 0)

    assert _poll(manager) is None
    manager.rpc_broadcast_mq.enqueue.assert_called_once_with((_POLL,))
    first_pending = manager._pending_kv_aggregation

    for output_queue in manager.kv_outputs_queues:
        output_queue.put(_report(operation, _STORE, _RETIRED))
    result = _poll(manager)

    assert result is not None
    assert result.finished_saving == {operation}
    manager.rpc_broadcast_mq.enqueue.assert_has_calls([call((_POLL,)), call((_POLL,))])
    assert manager.rpc_broadcast_mq.enqueue.call_count == 2
    assert manager._pending_kv_aggregation is not None
    assert manager._pending_kv_aggregation is not first_pending
    assert manager._pending_kv_aggregation.worker_outputs == [None, None]
