from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest

pytest.importorskip("aiter", reason="requires AITER to import model_runner")

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
    tokenIDProcessor.prepare_input_ids(processor, _prefill_batch([False]))
    tokenIDProcessor.prepare_input_ids(processor, _prefill_batch([False, False]))

    processor.recv_mtp_status_async.assert_not_called()
    np.testing.assert_array_equal(processor.prev_rejected_num, [7])
    np.testing.assert_array_equal(processor.prev_bonus_num, [8])

    # If any request reaches its final chunk, the batch runs postprocess. Its
    # status dequeue must therefore happen exactly once, even though another
    # request in the same batch is still a middle chunk.
    tokenIDProcessor.prepare_input_ids(processor, _prefill_batch([False, True]))

    processor.recv_mtp_status_async.assert_called_once_with()
    np.testing.assert_array_equal(processor.prev_rejected_num, [2])
    np.testing.assert_array_equal(processor.prev_bonus_num, [1])
