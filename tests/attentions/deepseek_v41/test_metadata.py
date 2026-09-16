# SPDX-License-Identifier: MIT
"""Shared V4 metadata staging survives ragged request changes and graph replay."""

from types import SimpleNamespace

import pytest
import torch

from atom.model_ops.attentions.deepseek_v41.backend import DeepseekV41MetadataBuilder
from atom.model_ops.attentions.deepseek_v41.cache import PagedAttentionCache
from atom.model_ops.attentions.deepseek_v41.metadata import RequestSpan
from atom.model_ops.attentions.pool_layout.v41_pool_geometry import V41PoolGeometry
from tests.attentions.deepseek_v41.helpers import metadata_buffers


def staged_step(cache, requests, *, buffers, running_bs, running_tokens):
    builder = DeepseekV41MetadataBuilder.__new__(DeepseekV41MetadataBuilder)
    builder.model_runner = SimpleNamespace(forward_vars=buffers)
    state_slot_out = builder._populate_state_slot_mappings(
        SimpleNamespace(state_slots_committed=[span.slot for span in requests]),
        len(requests),
        running_bs,
    )
    return cache.begin_step(
        requests,
        buffers=buffers,
        running_bs=running_bs,
        running_tokens=running_tokens,
        state_slot_out=state_slot_out,
    )


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_staged_metadata_refreshes_reordered_ragged_and_empty_batches(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("ROCm GPU required")
    cache = PagedAttentionCache(
        V41PoolGeometry(1, ((0, 2),), 4, 4, 512, 32), 8, 4, device
    )
    buffers = metadata_buffers(4, 8, 4, device)
    pointers = {name: value.gpu.data_ptr() for name, value in buffers.items()}
    first = RequestSpan(17, 1, 0, 3, 3, (5, 1))
    second = RequestSpan(24, 4, 3, 1, 1, (2, 7))
    for requests in (
        (first, second),
        (RequestSpan(24, 5, 0, 2, 1, (2, 7)), RequestSpan(17, 4, 2, 1, 3, (5, 1))),
        (RequestSpan(9, 0, 0, 1, 0, (6,)),),
        (),
    ):
        step = staged_step(
            cache, requests, buffers=buffers, running_bs=4, running_tokens=8
        )
        expected_positions = [
            p for span in requests for p in range(span.position, span.end)
        ]
        expected_batches = [
            i for i, span in enumerate(requests) for _ in range(span.length)
        ]
        assert step.positions.tolist() == expected_positions
        assert step.batch_ids.tolist() == expected_batches
        assert step.slots.tolist() == [span.slot for span in requests]
        assert step.cu_seqlens_q.tolist() == [span.offset for span in requests] + [
            step.length
        ]
        assert step.block_tables.stride(0) == 4
        for i, span in enumerate(requests):
            assert step.block_tables[i].tolist() == list(span.block_ids) + [0] * (
                4 - len(span.block_ids)
            )
            assert step.request_steps[i].cu_seqlens_q.tolist() == [0, span.length]
        assert buffers["batch_id_per_q_token"].gpu[step.length :].tolist() == [-1] * (
            8 - step.length
        )
        assert buffers["positions"].gpu[step.length :].count_nonzero() == 0
        assert buffers["cu_seqlens_q"].gpu[len(requests) + 1 :].tolist() == [
            step.length
        ] * (4 - len(requests))
        assert buffers["block_tables"].gpu[len(requests) :].count_nonzero() == 0
        assert {
            name: value.gpu.data_ptr() for name, value in buffers.items()
        } == pointers
        for name, tensor in (
            ("positions", step.positions),
            ("block_tables", step.block_tables),
            ("v4_meta_state_slot_out", step.slots),
        ):
            assert (
                tensor.untyped_storage().data_ptr()
                == buffers[name].gpu.untyped_storage().data_ptr()
            )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="ROCm GPU required")
def test_v4_window_write_graph_reads_updated_requests_without_recapture():
    geo = V41PoolGeometry(1, ((0, 2),), 4, 4, 512, 32)
    cache = PagedAttentionCache(geo, 8, 4, "cuda")
    buffers = metadata_buffers(2, 2, 2, "cuda")
    first = (RequestSpan(17, 0, 0, 1, 1, (0, 1)), RequestSpan(24, 2, 1, 1, 3, (2, 3)))
    step = staged_step(cache, first, buffers=buffers, running_bs=2, running_tokens=2)
    values = torch.randn(1, 2, 512, device="cuda", dtype=torch.bfloat16)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        cache.write_window(0, values, step)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            cache.write_window(0, values, step)
    torch.cuda.current_stream().wait_stream(stream)
    window = cache.state.view("window")[0]
    for spans in (
        (RequestSpan(24, 3, 0, 1, 3, (2, 3)), RequestSpan(17, 1, 1, 1, 1, (0, 1))),
        (RequestSpan(17, 4, 0, 1, 0, (4, 5)), RequestSpan(24, 6, 1, 1, 2, (6, 7))),
    ):
        staged_step(cache, spans, buffers=buffers, running_bs=2, running_tokens=2)
        window.zero_()
        values.normal_()
        graph.replay()
        expected = torch.zeros_like(window)
        for i, span in enumerate(spans):
            expected[span.slot, span.position % geo.window_size] = values[0, i]
        torch.testing.assert_close(window, expected, rtol=0, atol=0)


@pytest.mark.parametrize("model", ["v4", "v41"])
def test_shared_slot_publisher_preserves_pool_geometry_and_empty_padding(model):
    from atom.model_ops.attentions.deepseek_v4_attn import (
        DeepseekV4AttentionMetadataBuilder,
    )

    cls = (
        DeepseekV4AttentionMetadataBuilder
        if model == "v4"
        else DeepseekV41MetadataBuilder
    )
    builder = cls.__new__(cls)
    builder.pool_geometry = SimpleNamespace(slot_positions=8)
    builder.model_runner = SimpleNamespace(forward_vars=metadata_buffers(4, 8, 2))
    pointer = builder.model_runner.forward_vars["v4_meta_state_slot_out"].gpu.data_ptr()
    for scheduled_slots in ([3, 1], [6], []):
        result, cpu = builder._populate_state_slot_mappings(
            SimpleNamespace(state_slots_committed=scheduled_slots),
            len(scheduled_slots),
            4,
            return_cpu=True,
        )
        expected = [7 - slot if model == "v4" else slot for slot in scheduled_slots]
        assert result.tolist() == expected + [0] * (4 - len(expected))
        assert cpu.tolist() == expected
        assert result.data_ptr() == pointer
