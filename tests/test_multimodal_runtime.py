# SPDX-License-Identifier: MIT
"""Independent scatter oracle plus scheduler/worker lease lifecycle tests."""

from types import SimpleNamespace

import pytest
import torch
from conftest import MockConfig

from atom.model_engine.engine_core import EngineCore
from atom.model_engine.multimodal_runtime import (
    VisionEmbeddingCache,
    embed_multimodal_batch,
)
from atom.model_engine.scheduler import ScheduledBatchOutput, Scheduler
from atom.model_engine.sequence import Sequence, SequenceStatus
from atom.sampling_params import SamplingParams


def payload(seed=9):
    return {
        "cache_seed": seed,
        "embedding_spans": ((3, 5), (12, 4)),
        "pixel_values": torch.arange(18).view(9, 2).float(),
        "image_grid_thw": torch.tensor([[1, 2, 2], [1, 1, 2]]),
    }


class VisionModel:
    def embed_input_ids(self, ids):
        return ids[:, None].expand(-1, 2).float().clone()

    def get_vision_embeddings(self, values, grids):
        return values + 100


@pytest.mark.parametrize("width", [1, 2, 3, 4, 7, 13])
def test_every_image_boundary_scatter_and_reorder(width):
    cache = VisionEmbeddingCache()
    model = VisionModel()
    data = payload()
    full_ids = torch.full((19,), 77)
    expected = model.embed_input_ids(full_ids)
    expected[3:8] = data["pixel_values"][:5] + 100
    expected[12:16] = data["pixel_values"][5:] + 100
    # Both requests share one encoding while their rows are reordered every
    # step. Real and stray placeholder IDs are deliberately identical.
    for start in range(0, len(full_ids), width):
        length = min(width, len(full_ids) - start)
        order = [10, 11] if start % 2 else [11, 10]
        descriptor = (
            data
            if start == 0
            else {
                k: v
                for k, v in data.items()
                if k not in ("pixel_values", "image_grid_thw")
            }
        )
        batch = SimpleNamespace(
            req_ids=order,
            num_scheduled_tokens=[length] * 2,
            context_lens=[start + length] * 2,
            multimodal_data={i: descriptor for i in order},
        )
        actual = embed_multimodal_batch(
            model,
            cache,
            full_ids[start : start + length].repeat(2),
            batch,
            "cpu",
            torch.float32,
        )
        torch.testing.assert_close(
            actual, expected[start : start + length].repeat(2, 1), rtol=0, atol=0
        )
    assert cache.encodes == 2 and len(cache.entries) == 1
    cache.release([10])
    assert len(cache.entries) == 1
    cache.release([11])
    assert not cache.entries and not cache.leases


def test_missing_lease_or_changed_live_identity_fails():
    cache = VisionEmbeddingCache()
    data = payload()
    descriptor = {k: v for k, v in data.items() if k != "pixel_values"}
    with pytest.raises(RuntimeError, match="lease missing"):
        cache.acquire(3, descriptor, lambda _: None)
    cache.acquire(3, data, lambda d: d["pixel_values"])
    with pytest.raises(ValueError, match="change its image identity"):
        cache.acquire(3, payload(10), lambda d: d["pixel_values"])
    cache.release([3, 3, 99])
    assert not cache.entries


def test_scheduler_ack_retry_preempt_abort_and_no_decode_payload():
    config = MockConfig(
        max_num_batched_tokens=6,
        max_model_len=128,
        num_kvcache_blocks=100,
        enable_chunked_prefill=True,
    )
    scheduler = Scheduler(config)
    seq = Sequence(
        [77] * 19,
        4,
        SamplingParams(temperature=0, max_tokens=2),
        multimodal_data=payload(),
    )
    scheduler.add(seq)
    batch, _ = scheduler.schedule()
    assert batch.total_tokens_num == 6
    assert "pixel_values" in batch.multimodal_data[seq.id]
    assert seq.multimodal_data is not None and not seq.multimodal_cache_ready
    # An output-free middle chunk still acknowledges successful encoding.
    scheduler.postprocess(
        list(scheduler.running),
        ScheduledBatchOutput([], [], None, None, None),
        batch=batch,
    )
    assert seq.multimodal_cache_ready
    scheduler.running.remove(seq)
    assert scheduler.preempt(seq)
    retry, _ = scheduler.schedule()
    assert retry.num_cached_tokens[0] == 0
    assert "pixel_values" not in retry.multimodal_data[seq.id]
    assert seq.multimodal_data["pixel_values"] is not None
    scheduler.postprocess(
        list(scheduler.running),
        ScheduledBatchOutput([], [], None, None, None),
        batch=retry,
    )
    seq.status = SequenceStatus.ABORTED
    finished = scheduler.postprocess(
        list(scheduler.running), ScheduledBatchOutput([], [], None, None, None)
    )
    assert finished == [seq]
    assert seq.multimodal_data is None
    calls = []
    core = SimpleNamespace(
        runner_mgr=SimpleNamespace(call_func=lambda *args: calls.append(args))
    )
    EngineCore._release_multimodal_requests(core, finished)
    assert calls == [("release_multimodal_requests", [seq.id])]


def test_different_images_do_not_share_published_prefix():
    from atom.model_engine.block_manager import BlockManager

    manager = BlockManager(
        MockConfig(enable_prefix_caching=True, num_kvcache_blocks=100)
    )
    first = Sequence([77] * 20, 4, multimodal_data=payload(10))
    assert manager.allocate(first, 0)
    manager.hash_blocks(first, 20)
    same = Sequence([77] * 20, 4, multimodal_data=payload(10))
    other = Sequence([77] * 20, 4, multimodal_data=payload(11))
    assert manager.can_allocate(same) > 0
    assert manager.can_allocate(other) == 0


def test_prefill_ack_omits_media_and_decode_has_no_payload():
    config = MockConfig(
        max_num_batched_tokens=6,
        max_model_len=128,
        num_kvcache_blocks=100,
        enable_chunked_prefill=True,
    )
    scheduler = Scheduler(config)
    data = payload()
    data["token_types"] = torch.zeros(19, dtype=torch.int32)
    seq = Sequence(
        [77] * 19,
        4,
        SamplingParams(temperature=0, max_tokens=2, ignore_eos=True),
        multimodal_data=data,
    )
    scheduler.add(seq)
    prefills = 0
    decoded = False
    for _ in range(10):
        if scheduler.is_finished():
            break
        batch, active = scheduler.schedule()
        if batch.total_seqs_num_prefill:
            transmitted = batch.multimodal_data[seq.id]
            assert "token_types" not in transmitted
            assert ("pixel_values" in transmitted) == (prefills == 0)
            assert ("image_grid_thw" in transmitted) == (prefills == 0)
            # The retry source survives all worker acknowledgements.
            assert seq.multimodal_data is data
            prefills += 1
        else:
            assert batch.total_seqs_num_decode == 1
            assert not batch.multimodal_data
            decoded = True
        scheduler.postprocess(
            list(active.values()),
            ScheduledBatchOutput([seq.id], [(90,)], None, None, None),
            batch=batch,
        )
    assert scheduler.is_finished() and decoded and prefills == 4
    assert seq.multimodal_data is None
