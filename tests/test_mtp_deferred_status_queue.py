from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest

# On the module, not on "aiter" -- see tests/test_dspark.py for why a name-only
# guard passes on the non-GPU runner and then aborts collection.
pytest.importorskip(
    "atom.model_engine.model_runner",
    reason="requires AITER to import model_runner",
    exc_type=ImportError,
)

from atom.model_engine.model_runner import (
    _kv_config_has_producer,
    tokenIDProcessor,
)
from atom.model_engine.scheduler import ScheduledBatch


def _prefill_batch(is_final_chunk: list[bool]) -> ScheduledBatch:
    batch = object.__new__(ScheduledBatch)
    batch.scheduled_tokens = np.arange(4, dtype=np.int32)
    batch.total_tokens_num = 4
    batch.total_tokens_num_prefill = 4
    batch.total_tokens_num_decode = 0
    batch.total_seqs_num_prefill = len(is_final_chunk)
    batch.total_seqs_num_decode = 0
    batch.is_final_chunk = is_final_chunk
    return batch


def _batch_with_ids(req_ids: list[int], *, is_dummy_run: bool) -> ScheduledBatch:
    """The two fields `get_token_locations` reads, and nothing else."""
    batch = object.__new__(ScheduledBatch)
    batch.req_ids = req_ids
    batch.is_dummy_run = is_dummy_run
    return batch


def _locations(prev: ScheduledBatch, cur: ScheduledBatch):
    processor = object.__new__(tokenIDProcessor)
    processor.prev_batch = prev
    return tokenIDProcessor.get_token_locations(processor, cur)


def test_a_dummy_carries_nothing_over_to_the_next_dummy():
    """Back-to-back DP-sync dummies must not look like one carried-over request.

    `dummy_execution` fabricates its sequence with a fixed id, so a second
    dummy matches the first by id and is taken for a request that ran last
    step. It then reads that "request's" anchor and drafts out of
    `prev_token_ids` / `draft_token_ids`, which by then hold whatever the draft
    pass's graph-pool storage was overwritten with -- out-of-vocab ids that
    fault the target's embedding gather. A rank idling behind someone else's
    long chunked prefill runs dummies back to back, so this is the ordinary
    case at high concurrency, not a corner.
    """
    dummy_id = -1  # what dummy_execution builds; the collision is with ITSELF
    first = _batch_with_ids([dummy_id], is_dummy_run=True)
    second = _batch_with_ids([dummy_id], is_dummy_run=True)

    locs = _locations(first, second)

    assert locs.deferred_curr.tolist() == []
    assert locs.deferred_prev.tolist() == []
    assert locs.new_curr.tolist() == [0]


def test_a_real_batch_after_a_dummy_takes_the_host_path():
    """...and so does a real request that wakes up after one.

    Not a consolation prize for the line above: the dummy's own `postprocess`
    reports the PREVIOUS batch's tokens, so by the time this request is
    scheduled its anchor is already on the host. Reading `prev_token_ids`
    instead would be reading the dummy's forward.
    """
    dummy = _batch_with_ids([-1], is_dummy_run=True)
    real = _batch_with_ids([7, 9], is_dummy_run=False)

    locs = _locations(dummy, real)

    assert locs.deferred_curr.tolist() == []
    assert locs.new_curr.tolist() == [0, 1]


def test_a_real_batch_still_carries_over_from_a_real_batch():
    """The dummy rule must not cost an ordinary decode step its deferred rows."""
    prev = _batch_with_ids([7, 9], is_dummy_run=False)
    cur = _batch_with_ids([9, 11], is_dummy_run=False)

    locs = _locations(prev, cur)

    assert locs.deferred_curr.tolist() == [0]  # req 9 is position 0 now...
    assert locs.deferred_prev.tolist() == [1]  # ...and was position 1 before
    assert locs.new_curr.tolist() == [1]


def _processor() -> tokenIDProcessor:
    processor = object.__new__(tokenIDProcessor)
    processor.input_ids = SimpleNamespace(
        np=np.zeros(8, dtype=np.int32),
        gpu=np.zeros(8, dtype=np.int32),
        copy_to_gpu=mock.Mock(),
    )
    processor.recv_mtp_status_async = mock.Mock(
        return_value=(
            np.array([2], dtype=np.int32),
            np.array([1], dtype=np.int32),
        )
    )
    processor.prev_rejected_num = np.array([7], dtype=np.int32)
    processor.prev_bonus_num = np.array([8], dtype=np.int32)
    return processor


@pytest.mark.parametrize(
    ("kv_config", "expected"),
    [
        ({}, False),
        ({"kv_connector": "mooncake", "kv_role": "kv_consumer"}, False),
        ({"kv_connector": "mooncake", "kv_role": "kv_producer"}, True),
        (
            {
                "kv_connector": "multi",
                "connectors": [
                    {"kv_connector": "lmcache_offload", "kv_role": "offload"},
                    {"kv_connector": "mooncake", "kv_role": "kv_producer"},
                ],
            },
            True,
        ),
    ],
)
def test_detects_remote_prefill_producer(kv_config, expected):
    assert _kv_config_has_producer(kv_config) is expected


@pytest.mark.parametrize(
    ("kv_config", "expected"),
    [
        ({"kv_role": "kv_consumer"}, True),
        ({"kv_role": "kv_producer"}, False),
        (
            {
                "kv_connector": "multi",
                "connectors": [
                    {"kv_role": "offload"},
                    {"kv_role": "kv_producer"},
                ],
            },
            False,
        ),
    ],
)
def test_remote_prefill_producer_disables_deferred_output(kv_config, expected):
    runner = SimpleNamespace(
        config=SimpleNamespace(
            pipeline_parallel_size=1,
            kv_transfer_config=kv_config,
        ),
        device="cuda",
    )
    with (
        mock.patch("atom.model_engine.model_runner.CpuGpuBuffer"),
        mock.patch("atom.model_engine.model_runner.torch.cuda.Stream"),
        mock.patch("atom.model_engine.model_runner.torch.zeros"),
    ):
        processor = tokenIDProcessor(runner, max_num_batched_tokens=8)

    assert processor.is_deferred_out is expected


def test_middle_prefills_preserve_status_until_mixed_final_batch():
    processor = _processor()

    # Pure middle chunks skip postprocess, so neither the deferred-token queue
    # nor its matching MTP-status queue may advance.
    tokenIDProcessor.prepare_input_ids(processor, _prefill_batch([False]), 1)
    tokenIDProcessor.prepare_input_ids(processor, _prefill_batch([False, False]), 1)

    processor.recv_mtp_status_async.assert_not_called()
    np.testing.assert_array_equal(processor.prev_rejected_num, [7])
    np.testing.assert_array_equal(processor.prev_bonus_num, [8])

    # If any request reaches its final chunk, the batch runs postprocess. Its
    # status dequeue must therefore happen exactly once, even though another
    # request in the same batch is still a middle chunk.
    tokenIDProcessor.prepare_input_ids(processor, _prefill_batch([False, True]), 1)

    processor.recv_mtp_status_async.assert_called_once_with()
    np.testing.assert_array_equal(processor.prev_rejected_num, [2])
    np.testing.assert_array_equal(processor.prev_bonus_num, [1])


def test_deferred_decode_returns_the_flat_width_it_staged():
    processor = object.__new__(tokenIDProcessor)
    processor.input_ids = SimpleNamespace(
        np=np.zeros(16, dtype=np.int32),
        gpu=np.zeros(16, dtype=np.int32),
        copy_to_gpu=mock.Mock(),
    )
    processor.decode_cu = SimpleNamespace(
        np=np.zeros(3, dtype=np.int32),
        copy_to_gpu=mock.Mock(return_value=np.zeros(3, dtype=np.int32)),
    )
    processor.decode_src = SimpleNamespace(
        np=np.zeros(2, dtype=np.int32),
        copy_to_gpu=mock.Mock(return_value=np.zeros(2, dtype=np.int32)),
    )
    processor.is_deferred_out = True
    processor.prev_batch = object()
    processor.use_spec = True
    processor.prev_rejected_num = None
    processor.prev_bonus_num = None
    processor.pre_num_decode_token_per_seq = 1
    processor.prev_token_ids = np.zeros(2, dtype=np.int32)
    processor.draft_token_ids = None
    processor.runner = SimpleNamespace(enforce_eager=True, capture_sizes=[])
    processor.get_token_locations = mock.Mock(
        return_value=SimpleNamespace(
            deferred_curr=np.array([], dtype=np.int32),
            deferred_prev=np.array([], dtype=np.int32),
            new_curr=np.array([0, 1], dtype=np.int32),
        )
    )

    batch = SimpleNamespace(
        scheduled_tokens=np.arange(8, dtype=np.int32),
        # Simulate a worker-side shape expansion after the scheduler total was
        # captured: the deferred path stages two q=4 rows.
        total_tokens_num=2,
        total_tokens_num_prefill=0,
        total_tokens_num_decode=2,
        total_seqs_num_prefill=0,
        req_ids=[10, 11],
        dynamic_spec_query_tokens_per_req=None,
        scheduled_spec_decode_tokens=np.arange(6, dtype=np.int32).reshape(2, 3),
        num_rejected=np.zeros(2, dtype=np.int32),
        num_bonus=np.zeros(2, dtype=np.int32),
        produces_output=lambda: False,
    )

    with mock.patch(
        "atom.model_engine.model_runner.fill_deferred_decode_ids"
    ) as fill_ids:
        result = tokenIDProcessor.prepare_input_ids(processor, batch, 4)

    assert result.shape == (8,)
    # The prefill stage at the top of the method runs unconditionally and is a
    # no-op on a decode batch; the staging that matters is the flat decode one.
    assert processor.input_ids.copy_to_gpu.call_args_list == [
        mock.call(0),
        mock.call(8),
    ]
    fill_ids.assert_called_once()
