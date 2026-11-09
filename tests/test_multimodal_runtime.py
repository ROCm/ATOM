# SPDX-License-Identifier: MIT
"""Independent scatter oracle plus scheduler/worker lease lifecycle tests."""

from types import SimpleNamespace

import pytest
import torch
from aiter_stub import stubbed_aiter
from conftest import MockConfig

# `engine_core` reaches `async_proc`, which needs a real AITER build at import
# time; a CPU runner has none.
with stubbed_aiter():
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
    EngineCore._release_request_resources(core, finished)
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


def test_kimi_k3_image_prompt_longer_than_the_token_budget_is_chunked():
    # Kimi-K3 used to hand the scheduler span-less media, so an image prompt
    # longer than `max_num_batched_tokens` was rejected outright and the client
    # got an empty response (OCRBench document images at the 8192 default).
    from atom.models.kimi_k3.image_processing import build_kimi_k3_inputs

    def processor(**kwargs):
        return {
            "input_ids": torch.tensor([[9, 42, 7]]),
            "pixel_values": torch.ones(64, 3, 14, 14),
            "grid_thws": torch.tensor([[1, 4, 16]]),
        }

    atom_config = SimpleNamespace(
        multimodal_config=SimpleNamespace(
            media_placeholder_token_id=42,
            vision_config=SimpleNamespace(merge_kernel_size=[2, 2]),
        )
    )
    token_ids, data = build_kimi_k3_inputs(
        atom_config, processor, [{"role": "user", "content": []}], [object()], {}
    )
    assert token_ids == [9] + [42] * 16 + [7]
    assert data["embedding_spans"] == ((1, 16),)

    scheduler = Scheduler(
        MockConfig(
            max_num_batched_tokens=6,
            max_model_len=128,
            num_kvcache_blocks=100,
            enable_chunked_prefill=True,
        )
    )
    seq = Sequence(
        token_ids, 4, SamplingParams(temperature=0, max_tokens=2), multimodal_data=data
    )
    assert scheduler._unschedulable_reason(seq) is None
    scheduler.add(seq)
    chunks = []
    while seq.num_cached_tokens < len(token_ids):
        batch, _ = scheduler.schedule()
        assert batch.multimodal_data[seq.id]["embedding_spans"] == ((1, 16),)
        chunks.append(batch.total_tokens_num_prefill)
        scheduler.postprocess(
            list(scheduler.running),
            ScheduledBatchOutput([], [], None, None, None),
            batch=batch,
        )
    assert chunks == [6, 6, 6]


def qwen3_5_payload(token_ids, image_grid_thw, seed=21):
    from atom.models.qwen3_5_vl import qwen3_5_image_spans

    spans = qwen3_5_image_spans(token_ids, image_grid_thw, 42, 2)
    rows = sum(count for _, count in spans)
    return {
        "cache_seed": seed,
        "embedding_spans": spans,
        "pixel_values": torch.arange(2 * rows).view(rows, 2).float(),
        "image_grid_thw": torch.tensor(image_grid_thw),
    }


# 3 text tokens, one 1x4x8 image (8 tokens after the 2x2 merge), 5 text tokens.
QWEN3_5_IDS = [7, 8, 9] + [42] * 8 + [5, 6, 7, 8, 9]


@pytest.mark.parametrize("cached", [0, 3, 5, 10, 11, 13])
def test_qwen3_5_prefix_hit_scatters_only_the_image_rows_left_in_the_forward(cached):
    # ROCm/ATOM#2865: the whole-image merge scattered all 8 image rows into the
    # forward even when a prefix hit had already covered some (cached=5, 10) or
    # all (cached=11, 13) of them, and crashed on the shape mismatch.
    model = VisionModel()
    data = qwen3_5_payload(QWEN3_5_IDS, [[1, 4, 8]])
    full_ids = torch.tensor(QWEN3_5_IDS)
    expected = model.embed_input_ids(full_ids)
    expected[3:11] = data["pixel_values"] + 100

    length = len(QWEN3_5_IDS) - cached
    batch = SimpleNamespace(
        req_ids=[4],
        num_scheduled_tokens=[length],
        context_lens=[len(QWEN3_5_IDS)],
        multimodal_data={4: data},
    )
    actual = embed_multimodal_batch(
        model, VisionEmbeddingCache(), full_ids[cached:], batch, "cpu", torch.float32
    )
    torch.testing.assert_close(actual, expected[cached:], rtol=0, atol=0)


def test_qwen3_5_image_prompt_is_chunked_and_resumes_after_a_prefix_hit():
    from atom.models.qwen3_5_vl import build_qwen_vl_inputs

    def processor(**kwargs):
        return {
            "input_ids": torch.tensor([QWEN3_5_IDS]),
            "pixel_values": torch.ones(32, 3),
            "image_grid_thw": torch.tensor([[1, 4, 8]]),
        }

    atom_config = SimpleNamespace(
        multimodal_config=SimpleNamespace(
            model_type="qwen3_5",
            image_token_id=42,
            vision_config=SimpleNamespace(spatial_merge_size=2),
        )
    )
    token_ids, data = build_qwen_vl_inputs(
        atom_config, processor, "<|image_pad|> prompt", [object()], {}
    )
    assert data["embedding_spans"] == ((3, 8),)

    scheduler = Scheduler(
        MockConfig(
            max_num_batched_tokens=6,
            max_model_len=128,
            num_kvcache_blocks=100,
            enable_chunked_prefill=True,
            enable_prefix_caching=True,
        )
    )

    def prefill(seq):
        """Schedule `seq` to the end of its prompt; return each chunk's size."""
        assert scheduler._unschedulable_reason(seq) is None
        assert not scheduler._requires_atomic_prefill(seq)
        scheduler.add(seq)
        chunks = []
        while seq.num_cached_tokens < len(token_ids):
            batch, _ = scheduler.schedule()
            assert batch.multimodal_data[seq.id]["embedding_spans"] == ((3, 8),)
            chunks.append(batch.total_tokens_num_prefill)
            scheduler.postprocess(
                list(scheduler.running),
                ScheduledBatchOutput([], [], None, None, None),
                batch=batch,
            )
        return chunks

    params = SamplingParams(temperature=0, max_tokens=2)
    # The 16-token prompt no longer has to fit one forward.
    assert prefill(Sequence(token_ids, 4, params, multimodal_data=data)) == [6, 6, 4]
    # The same image again hits the 12 published tokens, past the image's end
    # at 11: the forward holds 4 text tokens and none of the image (the
    # `[0, 2048]` crash in ROCm/ATOM#2865).
    again = Sequence(token_ids, 4, params, multimodal_data=dict(data))
    assert prefill(again) == [4]


def _cached_qwen3_5_processor():
    """The real Qwen3.5 processor, when its files are in the local HF cache."""
    transformers = pytest.importorskip("transformers")
    pytest.importorskip("PIL")
    pytest.importorskip("torchvision")  # the Qwen3-VL processor needs it
    try:
        return transformers.AutoProcessor.from_pretrained(
            "Qwen/Qwen3.5-0.8B", local_files_only=True
        )
    except OSError:
        pytest.skip("Qwen/Qwen3.5-0.8B is not in the local Hugging Face cache")


class MergingVisionModel(VisionModel):
    """One row per merged 2x2 patch group, as the real vision tower returns."""

    def get_vision_embeddings(self, values, grids):
        rows = int(grids.prod(-1).sum()) // 4
        return values.reshape(rows, -1)[:, :2] + 100


def test_qwen3_5_real_processor_spans_survive_every_prefix_hit():
    # ROCm/ATOM#2865 with the real HF processor instead of a stub: two images
    # of different sizes back to back, then text. For a prefix hit ending
    # before, inside, between, at the end of and past the images, the forward
    # must get exactly the image rows it still holds.
    from PIL import Image

    from atom.models.qwen3_5_vl import build_qwen_vl_inputs

    processor = _cached_qwen3_5_processor()
    atom_config = SimpleNamespace(
        multimodal_config=SimpleNamespace(
            model_type="qwen3_5",
            image_token_id=248056,
            vision_config=SimpleNamespace(spatial_merge_size=2),
        )
    )
    images = [Image.new("RGB", (256, 256), "red"), Image.new("RGB", (320, 256))]
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "image"},
                {"type": "text", "text": "What differs between these?"},
            ],
        }
    ]
    token_ids, data = build_qwen_vl_inputs(
        atom_config, processor, conversation, images, {}
    )
    spans = data["embedding_spans"]
    image_positions = [i for i, t in enumerate(token_ids) if t == 248056]
    assert [p for start, count in spans for p in range(start, start + count)] == (
        image_positions
    )
    merged = [int(n) // 4 for n in data["image_grid_thw"].prod(-1).tolist()]
    assert [count for _, count in spans] == merged

    model = MergingVisionModel()
    full_ids = torch.tensor(token_ids)
    rows = model.get_vision_embeddings(
        data["pixel_values"].float(), data["image_grid_thw"]
    )
    expected = model.embed_input_ids(full_ids)
    expected[image_positions] = rows
    (first, first_count), (second, second_count) = spans
    for cached in sorted(
        {
            first - 1,
            first + first_count // 2,
            second,
            second + second_count - 1,
            second + second_count,
            len(token_ids) - 1,
        }
    ):
        batch = SimpleNamespace(
            req_ids=[4],
            num_scheduled_tokens=[len(token_ids) - cached],
            context_lens=[len(token_ids)],
            multimodal_data={4: data},
        )
        actual = embed_multimodal_batch(
            model,
            VisionEmbeddingCache(),
            full_ids[cached:],
            batch,
            "cpu",
            torch.float32,
        )
        torch.testing.assert_close(actual, expected[cached:], rtol=0, atol=0)
