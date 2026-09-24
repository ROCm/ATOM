# SPDX-License-Identifier: MIT
"""Regressions for asynchronous TBO ownership and multimodal forwarding."""

import weakref
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("aiter", reason="V4.1 metadata requires the AITER runtime")

from atom.model_ops.engram.device.staging import EngramStagedRows, EngramStaging
from atom.utils import CpuGpuBuffer
from atom.utils.forward_context import Context, ForwardContext, _forward_context_local
from atom.utils.tbo.ubatch_splitting import UBatchSlice
from atom.utils.tbo.ubatch_wrapper import UBatchWrapper
from tests.attentions.deepseek_v41.test_tbo import make_parent


def _metadata_tensors(child):
    step = child.step
    tensors = {
        "positions": step.positions,
        "cu_seqlens_q": step.cu_seqlens_q,
        "batch_ids": step.batch_ids,
    }
    for ratio, plan in step.plans.items():
        tensors[f"compress_{ratio}"] = plan.compress_plan_gpu[: plan.num_compress]
    for ratio, pair in step.indptrs.items():
        for i, value in enumerate(pair[:2]):
            tensors[f"indptr_{ratio}_{i}"] = value[: step.width + 1]
    return tensors


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
@pytest.mark.parametrize("failed", [False, True])
def test_child_metadata_reuse_waits_for_gpu_readers(failed):
    builder, first = make_parent("cuda", starts=(3, 8))
    reference_builder, second = make_parent("cuda", starts=(17, 21))
    part = UBatchSlice(slice(0, 2), slice(0, 14))
    expected = []
    for i, parent in enumerate((first, second)):
        child = reference_builder.build_ubatch_prefill_metadata(parent, part, 2, i)
        expected.append(
            {name: value.cpu() for name, value in _metadata_tensors(child).items()}
        )

    # Allocate before delaying the stream so allocation/first-use work cannot
    # accidentally close the race under test.
    builder.build_ubatch_prefill_metadata(first, part, 2)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    reads = []
    with torch.cuda.stream(stream):
        for i in range(6):
            parent = (first, second)[i % 2]
            torch.cuda._sleep(100_000_000)
            child = builder.build_ubatch_prefill_metadata(parent, part, 2)
            try:
                with builder.ubatch_forward(parent):
                    reads.append(
                        {
                            name: value.clone()
                            for name, value in _metadata_tensors(child).items()
                        }
                    )
                    if failed:
                        raise RuntimeError("child failed after enqueue")
            except RuntimeError as error:
                assert failed and str(error) == "child failed after enqueue"
            # No synchronize between forwards: reuse must fence the pinned
            # writes and device readers itself, including the error path.
    stream.synchronize()
    for i, tensors in enumerate(reads):
        for name, value in tensors.items():
            torch.testing.assert_close(value.cpu(), expected[i % 2][name])


def test_release_kv_pools_drops_child_storage_and_rebuilds():
    builder, parent = make_parent("cpu")
    buffers, indptrs = builder._prefill_ubatch_storage(0)
    buffer_ref = weakref.ref(buffers["positions"].gpu)
    indptr_ref = weakref.ref(next(iter(indptrs.values()))[0])
    del buffers, indptrs
    builder.release_kv_pools()
    assert buffer_ref() is None
    assert indptr_ref() is None
    assert builder.cache is builder.copies is None
    part = UBatchSlice(slice(0, 1), slice(0, 7))
    child = builder.build_ubatch_prefill_metadata(parent, part, 1)
    torch.testing.assert_close(child.step.positions, parent.step.positions[:7])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
@pytest.mark.parametrize("split", [False, True])
def test_wrapper_preserves_image_embeddings_and_masks(monkeypatch, split):
    builder, parent = make_parent("cuda")
    ids = torch.arange(14, device="cuda", dtype=torch.int32)
    embeddings = (
        torch.arange(56, device="cuda", dtype=torch.float32).reshape(14, 4) + 100
    )
    parent.image_mask = ((ids >= 5) & (ids < 10)).unsqueeze(0)
    parts = [
        UBatchSlice(slice(0, 1), slice(0, 7)),
        UBatchSlice(slice(0, 2), slice(7, 14)),
    ]
    context = ForwardContext(
        attn_metadata=parent,
        no_compile_layers={},
        kv_cache_data={},
        context=Context(
            positions=parent.step.positions,
            is_prefill=True,
            scheduled_bs=2,
            running_bs=2,
            scheduled_tokens=14,
            running_tokens=14,
        ),
        ubatch_slices=parts if split else None,
    )
    monkeypatch.setattr(_forward_context_local, "ctx", context, raising=False)

    class Model(torch.nn.Module):
        def forward(self, input_ids, positions, inputs_embeds=None):
            from atom.utils.forward_context import get_forward_context

            mask = get_forward_context().attn_metadata.image_mask[0]
            return inputs_embeds + positions[:, None] + mask[:, None] * 1000

    actual = UBatchWrapper(Model(), builder)(
        ids, parent.step.positions, inputs_embeds=embeddings
    )
    expected = (
        embeddings
        + parent.step.positions[:, None]
        + parent.image_mask[0, :, None] * 1000
    )
    torch.testing.assert_close(actual, expected)
    if split:
        # Persistent workers must not retain the last child through thread
        # locals or job closures after the builder releases its storage.
        refs = [
            weakref.ref(buffers["positions"].gpu)
            for buffers, _, _ in builder._tbo_storage.values()
        ]
        builder.release_kv_pools()
        assert all(ref() is None for ref in refs)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
def test_parent_fallback_gathers_once_per_layer_and_stage():
    device = torch.device("cuda")
    layers = (1, 3)
    gathered = []
    generation = [0]

    class Group:
        def all_gather(self, value, **kwargs):
            gathered.append(value.shape)
            return torch.cat((value, value + 10), dim=1)

    staging = EngramStaging.__new__(EngramStaging)
    staging.host = SimpleNamespace(
        device=device,
        _tp_group=Group(),
        embed_width=4,
        buffers={
            layer: CpuGpuBuffer(8, 4, dtype=torch.float32, device=device)
            for layer in layers
        },
    )
    staging.collective = None
    staging.flat = {layer: torch.empty(8, 2, device=device) for layer in layers}
    staging.stream = torch.cuda.Stream()
    staging.done = {layer: torch.cuda.Event() for layer in layers}

    def start(width):
        staging.stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(staging.stream):
            torch.cuda._sleep(20_000_000)
            for layer in layers:
                staging.flat[layer].fill_(generation[0] * 100 + layer)
                staging.done[layer].record()

    staging.start = start
    parent = EngramStagedRows(staging, 8)
    views = (parent.slice(slice(0, 3)), parent.slice(slice(3, 8)))
    consumer = torch.cuda.Stream()
    results = []
    for cycle in range(2):
        generation[0] = cycle
        parent.stage()
        assert len(gathered) == len(layers) * (cycle + 1)
        with torch.cuda.stream(consumer):
            results.append(
                [view.get(layer).clone() for layer in layers for view in views]
            )
        assert len(gathered) == len(layers) * (cycle + 1)
        torch.cuda.current_stream().wait_stream(consumer)
        parent.join()
    consumer.synchronize()
    for cycle, values in enumerate(results):
        for i, layer in enumerate(layers):
            for value in values[2 * i : 2 * i + 2]:
                expected = torch.tensor(
                    [cycle * 100 + layer] * 2 + [cycle * 100 + layer + 10] * 2,
                    device=device,
                    dtype=value.dtype,
                )
                torch.testing.assert_close(value, expected.expand_as(value))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
@pytest.mark.parametrize(
    "lengths,request_index,token", [((2,), 0, 0), ((1, 1), 1, 1), ((10, 4), 1, 13)]
)
def test_single_token_prefill_child_keeps_prefill_semantics(
    lengths, request_index, token
):
    from atom.model_ops.attentions.pool_layout.v4_pool_fields import (
        MQA_LOGITS_PRESHUFFLE_ROWS,
    )
    from atom.utils.forward_context import AttnState

    builder, parent = make_parent("cuda", lengths=lengths)
    part = UBatchSlice(slice(request_index, request_index + 1), slice(token, token + 1))
    child = builder.build_ubatch_prefill_metadata(parent, part, 1)
    assert child.step.width == 1
    assert child.step.is_prefill and not child.step.decode
    assert child.state == AttnState.PREFILL_PREFIX
    for prefix, extend, _ in child.step.indptrs.values():
        assert prefix.data_ptr() != extend.data_ptr()
    assert child.step.indptrs[0][1].tolist() == [0, 1]
    # Keep the prefill tile bound even though this microbatch has one token.
    tiles = child.cache.unit_tiles(child.step, 1)
    end = child.step.requests[0].end
    columns = (end + builder.geometry.block_size - 1) // builder.geometry.block_size
    assert tiles.shape[-1] == columns * (
        builder.geometry.rows_per_page(1) // MQA_LOGITS_PRESHUFFLE_ROWS
    )
