# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

from atom.kv_transfer.disaggregation.aggregator import KVOutputAggregator
from atom.kv_transfer.disaggregation.types import (
    ConnectorCompletion,
    KVConnectorOutput,
    SaveOperationId,
    SaveSourceGroupId,
)


def _report(channel, operation, succeeded=True):
    return ConnectorCompletion(channel, operation, succeeded)


def _output(*completions):
    return KVConnectorOutput(connector_completions=set(completions))


def _source(operation, start=0):
    return _report(
        "dense.page.source_safe", SaveSourceGroupId(operation, ((start, start + 8),))
    )


def _retired(operation, succeeded=True):
    return _report("dense.page.retired", operation, succeeded)


def test_mixed_never_started_rank_drops_groups_only_after_full_retirement():
    aggregator = KVOutputAggregator(world_size=2)
    operation = SaveOperationId("request", 1)
    first, second = _source(operation), _source(operation, 8)
    never_started = _report("dense.page.store", operation, False)

    result = aggregator.aggregate(
        [_output(first, second), _output(never_started, _retired(operation))]
    )
    assert result.connector_completions == set()
    assert aggregator.pending_count == (0, 4)

    result = aggregator.aggregate([_output(_retired(operation)), _output()])
    assert result.connector_completions == {_retired(operation)}
    assert aggregator.pending_count == (0, 1)  # unfinished store outcome survives

    result = aggregator.aggregate(
        [_output(_report("dense.page.store", operation)), _output()]
    )
    assert result.connector_completions == {never_started}
    assert aggregator.pending_count == (0, 0)
    assert aggregator.aggregate([_output(first, second), _output()]).is_empty()
    assert aggregator.pending_count == (0, 0)


def test_older_running_generation_is_not_suppressed_by_newer_retirement():
    aggregator = KVOutputAggregator(world_size=2)
    older, newer = SaveOperationId("old", 1), SaveOperationId("new", 2)
    older_group, newer_group = _source(older), _source(newer)
    aggregator.aggregate([_output(older_group, newer_group), _output()])
    aggregator.aggregate([_output(_retired(newer)), _output(_retired(newer))])
    assert aggregator.pending_count == (0, 1)

    result = aggregator.aggregate([_output(), _output(older_group)])
    assert result.connector_completions == {older_group}
    result = aggregator.aggregate([_output(_retired(older)), _output(_retired(older))])
    assert result.connector_completions == {_retired(older)}
    assert aggregator.pending_count == (0, 0)
    assert aggregator._retired_dense_generations._intervals == [(1, 2)]


def test_retirement_tombstones_survive_bounded_group_tombstone_eviction():
    aggregator = KVOutputAggregator(world_size=2, terminal_tombstone_limit=2)
    oldest = SaveOperationId("1", 1)
    for generation in range(1, 101):
        operation = SaveOperationId(str(generation), generation)
        aggregator.aggregate([_output(_source(operation)), _output()])
        result = aggregator.aggregate(
            [_output(_retired(operation)), _output(_retired(operation))]
        )
        assert result.connector_completions == {_retired(operation)}
        assert aggregator.pending_count == (0, 0)
    assert aggregator._retired_dense_generations._intervals == [(1, 100)]

    late = _output(_source(oldest), _retired(oldest))
    assert aggregator.aggregate([late, _output()]).is_empty()
    assert aggregator.aggregate([_output(), late]).is_empty()
    assert aggregator.pending_count == (0, 0)


def test_failed_retirement_does_not_authorize_group_cleanup():
    aggregator = KVOutputAggregator(world_size=2)
    operation = SaveOperationId("request", 1)
    group = _source(operation)
    result = aggregator.aggregate(
        [_output(group, _retired(operation)), _output(_retired(operation, False))]
    )
    assert result.connector_completions == {_retired(operation, False)}
    assert aggregator.pending_count == (0, 1)
    result = aggregator.aggregate([_output(), _output(group)])
    assert result.connector_completions == {group}
    assert aggregator.pending_count == (0, 0)


def test_retirement_does_not_discard_other_channels_or_store_results():
    aggregator = KVOutputAggregator(world_size=2)
    operation = SaveOperationId("request", 1)
    unrelated = _report("other.page.source_safe", _source(operation).operation_id)
    store = _report("dense.page.store", operation)
    aggregator.aggregate([_output(unrelated, store), _output()])
    aggregator.aggregate([_output(_retired(operation)), _output(_retired(operation))])
    assert aggregator.pending_count == (0, 2)
    result = aggregator.aggregate([_output(), _output(unrelated, store)])
    assert result.connector_completions == {unrelated, store}
    assert aggregator.pending_count == (0, 0)


def test_reset_clears_retired_generation_fences_for_new_scheduler_lifetime():
    aggregator = KVOutputAggregator(world_size=2)
    operation = SaveOperationId("request", 1)
    aggregator.aggregate([_output(_retired(operation)), _output(_retired(operation))])
    aggregator.reset()
    group = _source(operation)
    result = aggregator.aggregate([_output(group), _output(group)])
    assert result.connector_completions == {group}
