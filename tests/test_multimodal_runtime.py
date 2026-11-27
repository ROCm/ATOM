# SPDX-License-Identifier: MIT
"""Independent scatter oracle plus scheduler/worker lease lifecycle tests."""

from types import SimpleNamespace

import pytest
import torch
from aiter_stub import atom_without_aiter, stubbed_aiter
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


class CountingTower(VisionModel):
    def __init__(self):
        self.calls = 0

    def get_vision_embeddings(self, values, grids):
        self.calls += 1
        return super().get_vision_embeddings(values, grids)


EC_PARAMS = {
    "transport": "mooncake",
    "key": 9,
    "rows": 9,
    "dtype": "float32",
    "mooncake": {"session": "192.0.2.7:17000", "address": 64, "length": 36},
}


def run_chunks(model, cache, data, read, width=4):
    """Prefill one 19-token request in chunks; the scattered rows."""
    ids = torch.full((19,), 77)
    rows = []
    for start in range(0, len(ids), width):
        length = min(width, len(ids) - start)
        descriptor = (
            data
            if start == 0
            else {k: v for k, v in data.items() if k != "pixel_values"}
        )
        batch = SimpleNamespace(
            req_ids=[5],
            num_scheduled_tokens=[length],
            context_lens=[start + length],
            multimodal_data={5: descriptor},
        )
        rows.append(
            embed_multimodal_batch(
                model,
                cache,
                ids[start : start + length],
                batch,
                "cpu",
                torch.float32,
                read,
            )
        )
    return torch.cat(rows)


def test_encoder_rows_replace_the_tower():
    stored = -torch.arange(18).view(9, 2).float()
    reads = []

    def read(params, cache_seed, dtype):
        reads.append((params, cache_seed, dtype))
        return stored.clone()

    model = CountingTower()
    data = {**payload(), "ec_transfer_params": EC_PARAMS}
    actual = run_chunks(model, VisionEmbeddingCache(), data, read)

    assert model.calls == 0
    assert reads == [(EC_PARAMS, 9, torch.float32)]
    torch.testing.assert_close(actual[3:8], stored[:5], rtol=0, atol=0)
    torch.testing.assert_close(actual[12:16], stored[5:], rtol=0, atol=0)
    assert torch.equal(actual[:3, 0], torch.full((3,), 77.0))


def test_failed_read_runs_the_tower_once():
    reads = []
    model = CountingTower()
    data = {**payload(), "ec_transfer_params": EC_PARAMS}
    actual = run_chunks(
        model, VisionEmbeddingCache(), data, lambda *args: reads.append(args)
    )

    assert model.calls == 1 and len(reads) == 1
    torch.testing.assert_close(actual[3:8], data["pixel_values"][:5] + 100)
    torch.testing.assert_close(actual[12:16], data["pixel_values"][5:] + 100)


def test_request_without_ec_params_never_reads():
    def read(*_args):
        raise AssertionError("read without ec_transfer_params")

    model = CountingTower()
    data = payload()
    actual = run_chunks(model, VisionEmbeddingCache(), data, read)

    assert model.calls == 1
    torch.testing.assert_close(actual[3:8], data["pixel_values"][:5] + 100)


def test_read_rows_get_the_span_row_check():
    model = CountingTower()
    data = {**payload(), "ec_transfer_params": EC_PARAMS}
    with pytest.raises(ValueError, match="disagree with explicit image spans"):
        run_chunks(
            model, VisionEmbeddingCache(), data, lambda *_args: torch.zeros(8, 2)
        )
    assert model.calls == 0


def _runner(max_model_len=1000, hidden_size=6, kv_transfer_config=None):
    return SimpleNamespace(
        ec_readers={},
        device="cuda:3",
        config=SimpleNamespace(
            max_model_len=max_model_len,
            hf_config=SimpleNamespace(hidden_size=hidden_size),
            kv_transfer_config=kv_transfer_config,
        ),
    )


def _read_with(monkeypatch, create_reader, runner, *handles):
    with atom_without_aiter() as load:
        ec_transfer = load("atom.ec_transfer")
        monkeypatch.setattr(ec_transfer, "create_reader", create_reader)
        model_runner = load("atom.model_engine.model_runner")
        model_runner_cls = model_runner.ModelRunner
        runner._ec_reader = model_runner_cls._ec_reader.__get__(runner)
        return [
            model_runner_cls.read_image_embeddings(runner, h, 9, torch.bfloat16)
            for h in handles
        ]


def test_runner_builds_one_reader_per_transport_for_its_first_rows(monkeypatch):
    built = []

    class FakeReader:
        def __init__(self):
            self.answers = [torch.ones(2, 3), None]

        def get(self, handle, key, dtype):
            assert key == 9 and dtype == torch.bfloat16
            return self.answers.pop(0)

    def create_reader(transport, size_bytes, device, **options):
        built.append((transport, size_bytes, device, options))
        return FakeReader()

    kv = {"ib_device": "rocep139s0"}
    first, second = _read_with(
        monkeypatch,
        create_reader,
        _runner(kv_transfer_config=kv),
        {**EC_PARAMS, "rows": 3},
        {**EC_PARAMS, "rows": 900},
    )

    assert torch.equal(first, torch.ones(2, 3)) and second is None
    # The first request's rows of `hidden_size` bf16 values, growing to every
    # token of the longest prompt; the KV connector's HCA options go along.
    assert built == [
        (
            "mooncake",
            3 * 6 * 2,
            "cuda:3",
            {"max_bytes": 1000 * 6 * 2, "kv_transfer_config": kv},
        )
    ]


def test_runner_reader_never_starts_past_the_longest_prompt(monkeypatch):
    built = []

    class FakeReader:
        def get(self, *_args):
            return None

    def create_reader(transport, size_bytes, device, **options):
        built.append((size_bytes, options["max_bytes"]))
        return FakeReader()

    _read_with(
        monkeypatch,
        create_reader,
        _runner(max_model_len=10),
        {**EC_PARAMS, "rows": 1 << 30},
    )
    assert built == [(120, 120)]


def test_reader_that_cannot_start_falls_back_to_the_tower_once(monkeypatch, caplog):
    attempts = []

    def create_reader(transport, *_args, **_kwargs):
        attempts.append(transport)
        raise RuntimeError("Mooncake is not installed")

    runner = _runner()
    with caplog.at_level("WARNING", logger="atom"):
        got = _read_with(monkeypatch, create_reader, runner, EC_PARAMS, EC_PARAMS)

    assert got == [None, None]
    # Built once: a transport that cannot start is not retried per request.
    assert attempts == ["mooncake"] and runner.ec_readers == {"mooncake": None}
    assert "Mooncake is not installed" in caplog.text


@pytest.mark.parametrize("handle", [None, "x", {}, {"transport": 3}])
def test_handle_without_a_transport_runs_the_tower(monkeypatch, handle):
    def create_reader(*_args, **_kwargs):
        raise AssertionError("no reader for a handle without a transport")

    assert _read_with(monkeypatch, create_reader, _runner(), handle) == [None]


@pytest.mark.parametrize(
    "local, others, expect_rows",
    [(True, True, True), (True, False, False), (False, True, False)],
)
def test_ranks_take_e_rows_only_when_every_read_succeeded(
    monkeypatch, local, others, expect_rows
):
    rows = torch.ones(2, 3)
    agreed = []

    class FakeReader:
        def get(self, *_args):
            return rows if local else None

    runner = SimpleNamespace(ec_readers={"mooncake": FakeReader()})
    runner._ec_reader = lambda handle, _dtype: runner.ec_readers["mooncake"]
    with atom_without_aiter() as load:
        model_runner = load("atom.model_engine.model_runner")
        monkeypatch.setattr(model_runner, "_forward_cpu_groups", lambda: ("tp", "pcp"))

        def group_agree(ok, group):
            agreed.append((ok, group))
            return ok and others

        monkeypatch.setattr(model_runner, "group_agree", group_agree)
        got = model_runner.ModelRunner.read_image_embeddings(
            runner, EC_PARAMS, 9, torch.bfloat16
        )

    assert (got is rows) == expect_rows and (got is None) != expect_rows
    # One verdict per group, the rank's own read first.
    assert [group for _, group in agreed] == ["tp", "pcp"]
    assert agreed[0][0] == local


@pytest.mark.parametrize(
    "ec_params, accept, reaches_engine",
    [
        (None, True, False),
        (EC_PARAMS, True, True),
        # A client may reach P directly: its handle is dropped unless the
        # server was started to take them.
        (EC_PARAMS, False, False),
    ],
)
def test_chat_request_puts_ec_params_into_multimodal_data(
    monkeypatch, ec_params, accept, reaches_engine
):
    import asyncio

    import httpx

    from atom.entrypoints.openai import api_server

    seen = []

    async def generate(ids, data, _sampling, _request_id, **_kwargs):
        seen.append(data)
        yield {
            "text": "red",
            "finish_reason": "stop",
            "num_tokens_input": 3,
            "num_tokens_output": 1,
        }

    for name, value in {
        "model_name": "m",
        "_request_logger": None,
        "default_chat_template_kwargs": {},
        "reasoning_toggle": None,
        "custom_message_encoder": None,
        "tool_call_parser_cls": None,
        "tokenizer": SimpleNamespace(decode=lambda _ids, **_kw: ""),
        "_get_multimodal_processor": lambda: object(),
        "_prepare_multimodal_inputs": lambda *_args: ([1, 2, 3], payload()),
        "generate_async_multimodal": generate,
        "accept_ec_transfer_params": accept,
    }.items():
        monkeypatch.setattr(api_server, name, value)
    image = {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA=="}}
    body = {
        "model": "m",
        "messages": [{"role": "user", "content": [image]}],
        "ec_transfer_params": ec_params,
    }

    async def post():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=api_server.app), base_url="http://test"
        ) as client:
            return await client.post("/v1/chat/completions", json=body)

    assert asyncio.run(post()).status_code == 200
    assert seen[0].get("ec_transfer_params") == (ec_params if reaches_engine else None)
