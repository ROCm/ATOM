# SPDX-License-Identifier: MIT
"""Publication contracts exercised with real tensors, including delayed GPUs."""

import gc
import os
import sys
import weakref
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest
import torch

from atom.utils import CpuGpuBuffer
from atom.utils.h2d import PublicationError, PublicationOwner, PublicationRegistry


def buffer(*shape, dtype=torch.int32, device="cpu"):
    return CpuGpuBuffer(
        *shape,
        dtype=dtype,
        device=device,
        pin_memory=device != "cpu",
        with_numpy=dtype != torch.bfloat16,
    )


def setup(device="cpu", shape=(16,), dtype=torch.int32, unit="rows"):
    owner = PublicationOwner(device, torch.cuda.Event() if device != "cpu" else None)
    a, b = buffer(*shape, dtype=dtype, device=device), buffer(
        *shape, dtype=dtype, device=device
    )
    x, y = owner.bind(a, "a", unit=unit), owner.bind(b, "b", unit=unit)
    group = owner.group("pair", (x, y))
    return owner, a, b, x, y, group


@pytest.mark.parametrize("bad", [-1, 17, 1.5, True])
def test_group_validation_is_atomic_and_retryable(bad):
    owner, a, b, _, _, group = setup()
    owner.begin()
    a.cpu.fill_(11)
    b.cpu.fill_(22)
    group.set_count(a, 8)
    group.set_count(b, bad)
    with pytest.raises((TypeError, ValueError)):
        group.publish(group.counts)
    assert torch.all(a.gpu == 0) and torch.all(b.gpu == 0)
    # Correct one producer without losing the other producer's staged prefix.
    group.set_count(b, 8)
    group.publish(group.counts)
    assert torch.all(a.gpu[:8] == 11) and torch.all(a.gpu[8:] == 0)
    assert torch.all(b.gpu[:8] == 22)


def test_direct_and_groups_share_epoch_including_later_phases():
    owner, a, b, x, _, group = setup()
    other = owner.group("other", (x,))
    owner.begin()
    a.copy_to_gpu(8)
    with pytest.raises(PublicationError, match="republish_reason"):
        group.publish((8, 8))
    b.copy_to_gpu(8)  # Failed group did not consume b's publication.
    owner.finish()
    owner.resume()
    with pytest.raises(PublicationError, match="republish_reason"):
        other.publish((8,))
    with pytest.raises(PublicationError, match="cannot begin"):
        owner.begin()


def test_another_thread_cannot_publish_or_resume_the_owner():
    owner, a, _, _, _, _ = setup()
    owner.begin()
    with ThreadPoolExecutor(max_workers=1) as worker:
        with pytest.raises(PublicationError, match="owner thread"):
            worker.submit(a.copy_to_gpu).result()
        a.copy_to_gpu()
        owner.finish()
        with pytest.raises(PublicationError, match="owner thread"):
            worker.submit(owner.resume).result()
    owner.resume()
    with pytest.raises(PublicationError, match="republish_reason"):
        a.copy_to_gpu()


def test_omitted_members_differ_from_explicit_empty_publication():
    owner, a, b, _, _, group = setup()
    owner.begin()
    group.publish((None, None))
    group.publish((0, None))
    with pytest.raises(PublicationError, match="republish_reason"):
        a.copy_to_gpu(0)
    b.cpu.fill_(22)
    b.copy_to_gpu(8)
    b.gpu.fill_(-88)
    group.publish((None, None))
    assert torch.all(b.gpu == -88)


def test_reason_does_not_release_source_and_reacquire_does_not_reset_ledger():
    owner, a, _, x, _, _ = setup()
    owner.begin()
    a.cpu.fill_(11)
    a.copy_to_gpu()
    first = a.gpu.clone()
    with pytest.raises(PublicationError, match="acquire_write"):
        a.copy_to_gpu(republish_reason="publish postprocess correction")
    with pytest.raises(PublicationError, match="republish_reason"):
        x.acquire_write()
    x.acquire_write(republish_reason="publish postprocess correction").fill_(22)
    with pytest.raises(PublicationError, match="republish_reason"):
        a.copy_to_gpu()
    a.copy_to_gpu(republish_reason="publish postprocess correction")
    assert torch.all(first == 11) and torch.all(a.gpu == 22)


def test_aliases_rejected_across_owners_but_disjoint_regions_allowed():
    registry = PublicationRegistry()
    owner = PublicationOwner("cpu", registry=registry)
    other = PublicationOwner("cpu", registry=registry)
    a, b = buffer(16), buffer(8)
    a.cpu, a.gpu = a.cpu[:8], a.gpu[:8]
    owner.bind(a, "a")
    b.gpu = a.gpu[2:6]
    b.cpu = b.cpu[:4]
    with pytest.raises(ValueError, match="destination overlaps"):
        other.bind(b, "alias")
    c, d = buffer(16), buffer(16)
    d.cpu, d.gpu = c.cpu[8:], c.gpu[8:]
    c.cpu, c.gpu = c.cpu[:8], c.gpu[:8]
    owner.bind(c, "left")
    other.bind(d, "right")


def test_shared_host_arena_cannot_be_registered_with_independent_reuse_state():
    registry = PublicationRegistry()
    one, two = PublicationOwner("cpu", registry=registry), PublicationOwner(
        "cpu", registry=registry
    )
    a, b = buffer(16), buffer(16)
    b.cpu = a.cpu
    one.bind(a, "a")
    with pytest.raises(ValueError, match="source overlaps"):
        two.bind(b, "b")


def test_duplicate_destination_in_group_rejected():
    owner, _, _, x, _, _ = setup()
    with pytest.raises(ValueError, match="destination twice"):
        owner.group("duplicate", (x, x))


@pytest.mark.parametrize(
    "dtype", [torch.int32, torch.int64, torch.float32, torch.bfloat16, torch.bool]
)
def test_mixed_representation_is_bitwise_and_tail_preserved(dtype):
    owner, a, _, _, _, group = setup(shape=(7,), dtype=dtype, unit="bytes")
    src = a.cpu.view(torch.uint8)
    dst = a.gpu.view(torch.uint8)
    src.copy_(torch.arange(src.numel(), dtype=torch.uint8) * 31)
    dst.fill_(197)
    owner.begin()
    group.publish((src.numel() - 1, None))
    assert torch.equal(src[:-1], dst[:-1])
    assert dst[-1] == 197


def test_rows_and_flat_prefix_are_distinct_and_legacy_copy_keeps_rows():
    owner, a, b, _, _, group = setup(shape=(3, 16), unit="elements")
    a.cpu.fill_(11)
    b.cpu.fill_(22)
    owner.begin()
    group.publish((3 * 5, None))
    b.copy_to_gpu(1)
    assert torch.all(a.gpu.flatten()[:15] == 11)
    assert torch.all(a.gpu.flatten()[15:] == 0)
    assert torch.all(b.gpu[0] == 22) and torch.all(b.gpu[1:] == 0)


def test_noncontiguous_direct_does_not_copy_unselected_columns():
    owner = PublicationOwner("cpu")
    a = buffer(3, 8)
    backing = a.gpu
    a.cpu, a.gpu = a.cpu[:, ::2], a.gpu[:, ::2]
    a.cpu.fill_(11)
    owner.bind(a, "strided")
    owner.begin()
    a.copy_to_gpu(2)
    assert torch.all(backing[:2, ::2] == 11)
    assert torch.all(backing[:, 1::2] == 0) and torch.all(backing[2] == 0)


def test_partial_enqueue_poison_prevents_next_forward(monkeypatch):
    owner, a, b, _, y, group = setup()
    owner.begin()
    a.cpu.fill_(11)
    b.cpu.fill_(22)

    def fail(count):
        raise RuntimeError("injected enqueue failure")

    monkeypatch.setattr(y, "_copy", fail)
    with pytest.raises(RuntimeError, match="injected"):
        group.publish((8, 8))
    assert torch.all(a.gpu[:8] == 11) and torch.all(b.gpu == 0)
    owner.drain()
    with pytest.raises(PublicationError, match="failed"):
        owner.begin()
    with pytest.raises(PublicationError, match="failed"):
        group.publish((None, None))


@pytest.mark.parametrize("failure_phase", ["begin", "finish", "acquire"])
def test_completion_failure_poison_prevents_source_reuse(failure_phase):
    class Completion:
        fail = False

        def synchronize(self):
            if self.fail:
                raise RuntimeError("event failed")

        def record(self, stream):
            if self.fail:
                raise RuntimeError("event failed")

    owner, a, _, binding, _, group = setup()
    event = owner.completion = Completion()
    if failure_phase != "begin":
        owner.begin()
        a.copy_to_gpu()
    event.fail = True
    with pytest.raises(RuntimeError, match="event failed"):
        if failure_phase == "begin":
            owner.begin()
        elif failure_phase == "finish":
            owner.finish()
        else:
            binding.acquire_write(republish_reason="update after first consumer")
    owner.drain()
    with pytest.raises(PublicationError, match="failed"):
        owner.begin()
    with pytest.raises(PublicationError, match="failed"):
        group.publish((None, None))


def test_strong_reference_and_binding_replacement():
    owner, a, _, _, _, group = setup()
    ref = weakref.ref(a.cpu)
    owner.begin()
    with pytest.raises(PublicationError, match="bound metadata storage is fixed"):
        a.cpu = torch.ones_like(a.cpu)
    gc.collect()
    assert ref() is not None
    group.publish((8, None))


@pytest.mark.parametrize("transport", ["packed"])
def test_transport_selection_preserves_fallback_and_owner_lifecycle(
    monkeypatch, transport
):
    owner, a, _, _, _, group = setup()
    # CPU-only users must not import the Triton backend, even when requested.
    monkeypatch.setitem(sys.modules, "atom.utils.packed_h2d", None)
    assert group.use_transport(transport) == "direct"
    reason = group.fallback_reason
    assert reason
    with pytest.raises(ValueError, match="H2D transport"):
        group.use_transport("unknown")
    assert group.fallback_reason == reason
    assert group.use_transport("direct") == "direct"
    assert group.fallback_reason is None and group._backend is None
    owner.begin()
    a.cpu.fill_(31)
    group.publish((8, None))
    with pytest.raises(PublicationError, match="during initialization"):
        group.use_transport(transport)
    with pytest.raises(PublicationError, match="republish_reason"):
        group.publish((8, None))
    assert torch.all(a.gpu[:8] == 31) and torch.all(a.gpu[8:] == 0)


@pytest.mark.parametrize("compact", [False, True])
def test_v41_reuses_early_query_prefix_or_explicitly_compacts(compact):
    from atom.model_ops.attentions.deepseek_v41.metadata import (
        RequestSpan,
        prepare_batch_step,
    )

    owner = PublicationOwner("cpu")
    buffers = {
        "positions": buffer(8),
        "cu_seqlens_q": buffer(4),
        "batch_id_per_q_token": buffer(8),
        "block_tables": buffer(3, 2),
    }
    cu = buffers["cu_seqlens_q"]
    owner.bind(cu, "cu_seqlens_q")
    owner.begin()
    original = [0, 2, 2, 5] if compact else [0, 2, 5, 5]
    cu.cpu.copy_(torch.tensor(original, dtype=torch.int32))
    cu.copy_to_gpu()
    first_consumer = cu.gpu.clone()
    step = prepare_batch_step(
        [RequestSpan(1, 0, 0, 2, 0), RequestSpan(2, 0, 2, 3, 1)],
        "cpu",
        block_tables=[(0,), (1,)],
        buffers=buffers,
        running_bs=3,
        running_tokens=8,
        query_prefix_ready=not compact,
        query_prefix_republish_reason=(
            "compact zero-token scheduler rows" if compact else None
        ),
    )
    assert first_consumer.tolist() == original
    assert step.cu_seqlens_q.tolist() == [0, 2, 5, 5]
    assert buffers["batch_id_per_q_token"].gpu.tolist() == [0, 0, 1, 1, 1, -1, -1, -1]
    with pytest.raises(PublicationError, match="republish_reason"):
        cu.copy_to_gpu()


gpu = pytest.mark.skipif(
    os.environ.get("RUN_H2D_GPU_TESTS") != "1" or not torch.cuda.is_available(),
    reason="set RUN_H2D_GPU_TESTS=1 with a GPU",
)


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=gpu)])
@pytest.mark.parametrize("unit", ["rows", "elements", "bytes"])
def test_repeated_and_changing_prefixes_preserve_contents_and_tail(device, unit):
    owner, a, _, binding, _, _ = setup(device, shape=(7, 3), unit=unit)
    expected = torch.zeros_like(a.cpu)
    width = {"rows": 1, "elements": 3, "bytes": 12}[unit]
    for step, rows in enumerate((1, 2, 3, 4, 5, 6, 2, 2, 0, 7), 1):
        owner.begin()
        a.cpu.fill_(step)
        binding._single.publish((rows * width,))
        owner.finish()
        if device != "cpu":
            owner.completion.synchronize()
        expected[:rows] = step
        assert torch.equal(a.gpu.cpu(), expected)


def test_single_member_failure_poison_and_invalid_counts_are_retryable(monkeypatch):
    owner, a, _, binding, _, _ = setup()
    owner.begin()
    for counts, reasons in (((), None), ((8,), ()), ((True,), None)):
        with pytest.raises((ValueError, TypeError)):
            binding._single.publish(counts, republish_reasons=reasons)
    assert binding._epoch != owner.epoch and torch.all(a.gpu == 0)

    def fail(count):
        raise RuntimeError("injected single enqueue failure")

    monkeypatch.setattr(binding, "_copy", fail)
    with pytest.raises(RuntimeError, match="single enqueue"):
        a.copy_to_gpu(8)
    with pytest.raises(PublicationError, match="failed"):
        binding._single.publish((None,))


@gpu
def test_delayed_gpu_different_buffers_and_explicit_republication():
    owner, a, b, x, _, group = setup("cuda")
    torch.cuda.synchronize()
    owner.begin()
    torch.cuda._sleep(20_000_000)
    a.cpu.fill_(11)
    group.publish((16, None))
    first = a.gpu.clone()
    b.cpu.fill_(22)
    group.publish((None, 16))
    second = b.gpu.clone()
    x.acquire_write(republish_reason="postprocess correction").fill_(33)
    a.copy_to_gpu(republish_reason="postprocess correction")
    owner.finish()
    owner.completion.synchronize()
    assert torch.all(first.cpu() == 11)
    assert torch.all(second.cpu() == 22)
    assert torch.all(a.gpu.cpu() == 33)


@gpu
def test_slot_rotation_waits_before_payload_changes():
    slots = [setup("cuda") for _ in range(2)]
    torch.cuda.synchronize()
    observations = []
    for epoch in range(8):
        owner, a, _, _, _, group = slots[epoch % 2]
        owner.begin()
        a.cpu.fill_(epoch)
        torch.cuda._sleep(2_000_000)
        group.publish((16, None))
        observations.append(a.gpu.clone())
        owner.finish()
    torch.cuda.synchronize()
    for epoch, result in enumerate(observations):
        assert torch.all(result.cpu() == epoch)


@gpu
def test_publish_before_graph_replay_and_reject_actual_capture():
    owner, a, _, _, _, group = setup("cuda")
    owner.begin()
    group.publish((16, None))
    output = torch.empty_like(a.gpu)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output.copy_(a.gpu)
        with pytest.raises(PublicationError, match="actual graph capture"):
            group.publish((None, None))
    owner.finish()
    pointer = a.gpu.data_ptr()
    for value, count in [(11, 16), (22, 8), (33, 0)]:
        owner.begin()
        a.cpu.fill_(value)
        group.publish((count, None))
        owner.finish()
        graph.replay()
        torch.cuda.synchronize()
        assert a.gpu.data_ptr() == pointer
        assert torch.all(output[:count].cpu() == value)
    assert torch.all(output[:8].cpu() == 22)
    assert torch.all(output[8:].cpu() == 11)


@gpu
def test_wrong_stream_is_rejected_before_any_submission():
    owner, _, _, _, _, group = setup("cuda")
    owner.begin()
    with (
        torch.cuda.stream(torch.cuda.Stream()),
        pytest.raises(PublicationError, match="compute stream"),
    ):
        group.publish((16, 16))
    group.publish((16, 16))
    owner.finish()


@gpu
@pytest.mark.parametrize("pp_size", [1, 2])
def test_real_runner_and_prefill_builder_share_reuse_and_publication(pp_size):
    from atom.model_engine.model_runner import ModelRunner
    from atom.model_ops.attentions.backends import CommonAttentionBuilder

    runner = ModelRunner.__new__(ModelRunner)
    runner.device = torch.device("cuda", 0)
    runner.config = SimpleNamespace(pipeline_parallel_size=pp_size)
    runner.enforce_eager = True
    runner.forward_vars = {
        "input_ids": buffer(16, device="cuda"),
        "decode_src": buffer(16, device="cuda"),
    }
    for name, shape in (
        ("slot_mapping", (16,)),
        ("context_lens", (4,)),
        ("block_tables", (4, 8)),
        ("cu_seqlens_k", (5,)),
        ("num_cached_tokens", (4,)),
        ("seq_starts", (4,)),
    ):
        value = buffer(*shape, device="cuda")
        value.publication_group = "prefill"
        runner.forward_vars[name] = value
    cu = buffer(5, device="cuda")
    cu.publication_group = "early"
    runner.forward_vars["cu_seqlens_q"] = cu
    runner.tokenID_processor = SimpleNamespace(
        input_ids=runner.forward_vars["input_ids"]
    )
    runner._init_forward_vars_ring()
    runner._init_h2d_publication()

    # Avoid constructing a model, but exercise actual production methods and
    # tensors rather than checking their spelling or number of copy calls.
    class Builder(CommonAttentionBuilder):
        __abstractmethods__ = frozenset()

    Builder.__abstractmethods__ = frozenset()
    builder = Builder.__new__(Builder)
    builder.model_runner = runner
    results = []
    for iteration in range(6):
        runner._advance_forward_vars()
        runner._gate_staging_reuse()
        var = runner.forward_vars
        for value in var.values():
            value.cpu.fill_(iteration + 1)
        var["cu_seqlens_q"].copy_to_gpu(3)
        torch.cuda._sleep(2_000_000)
        ctx = builder._upload_prefill_mirrors(
            2, 4, 8, iteration % 2 == 0, var["context_lens"].np[:2]
        )
        assert ("block_tables" in ctx) == (iteration % 2 == 0)
        assert ctx["context_lens"].shape == (4,)
        results.append(ctx["context_lens"].clone())
        with pytest.raises(PublicationError, match="republish_reason"):
            var["context_lens"].copy_to_gpu(4)
        runner._mark_staging_h2d_enqueued()
        runner._record_forward_vars_event()
    torch.cuda.synchronize()
    for iteration, result in enumerate(results):
        assert torch.all(result.cpu() == iteration + 1)


@gpu
@pytest.mark.parametrize("prefill", [False, True])
def test_rapidserve_entries_gate_sources_on_the_selected_stream(prefill):
    from atom.model_engine.model_runner import RapidServeModelRunner

    runner = RapidServeModelRunner.__new__(RapidServeModelRunner)
    runner.device = torch.device("cuda", 0)
    runner.config = SimpleNamespace(pipeline_parallel_size=2)
    runner.enforce_eager = True
    data = buffer(16, device="cuda")
    data.publication_group = "early"
    runner.forward_vars = {"input_ids": data, "decode_src": buffer(16, device="cuda")}
    runner.tokenID_processor = SimpleNamespace(input_ids=data)
    runner._init_forward_vars_ring()
    runner._init_h2d_publication()
    streams = {i: torch.cuda.Stream() for i in range(2)}
    runner._decode_streams = runner._prefill_streams = streams
    runner._done_event = torch.cuda.Event()
    runner._model_fwd_event = torch.cuda.Event()

    def prepare(batch):
        buf = runner.forward_vars["input_ids"]
        buf.cpu.fill_(batch.value)
        torch.cuda._sleep(2_000_000)
        return buf.copy_to_gpu(), None, None, None, True, False

    runner.prepare_model = prepare
    runner.run_model = lambda inputs, batch: (inputs.clone(), None)
    runner.postprocess = lambda batch, logits, *args, **kwargs: logits.clone()
    runner.sampler = lambda logits, *args: logits
    runner._record_kv_cache_ready = lambda batch: None
    results = []
    for value in range(6):
        batch = SimpleNamespace(cu_stream_fraction=value % 2, value=value + 1)
        output = runner.prefill_forward(batch) if prefill else runner.forward(batch)
        results.append(output)
    torch.cuda.synchronize()
    for i, output in enumerate(results):
        assert (output if prefill else output.cpu().tolist()) == [i + 1] * 16


def stream_setup(fast, backend):
    owner = PublicationOwner("cuda", torch.cuda.Event())
    if fast:
        if owner._get_current_stream is None:
            pytest.skip("current-stream tuple API unavailable")
    else:
        owner._get_current_stream = None
    buffers = [CpuGpuBuffer(8, dtype=torch.int32, device="cuda") for _ in range(2)]
    members = [owner.bind(buf, f"value_{i}") for i, buf in enumerate(buffers)]
    group = owner.group("pair", members)
    if backend == "packed":
        assert group.use_transport("packed") == "packed"
    return owner, buffers, group


@gpu
@pytest.mark.parametrize("fast", [False, True], ids=["public", "tuple"])
@pytest.mark.parametrize("backend", ["direct", "packed"])
def test_stream_alias_and_new_epoch_preserve_values_and_ledger(fast, backend):
    owner, buffers, group = stream_setup(fast, backend)
    first, second = torch.cuda.Stream(), torch.cuda.Stream()
    # Another Python wrapper with the same PyTorch stream ID is valid.
    alias = torch.cuda.Stream(
        stream_id=first.stream_id,
        device_index=first.device_index,
        device_type=first.device_type,
    )
    external = torch.cuda.ExternalStream(first.cuda_stream, device=first.device)
    torch.cuda.synchronize()  # Complete constructor writes before crossing streams.
    with torch.cuda.stream(first):
        owner.begin()
        group.publish((8, 8))  # Compile packing before the delayed queue.
        owner.finish()
        owner.begin()
        for i, buf in enumerate(buffers):
            buf.cpu.fill_(11 + i)
        with (
            torch.cuda.stream(second),
            pytest.raises(PublicationError, match="compute stream"),
        ):
            group.publish((8, 8))
        # ExternalStream may have a distinct logical ID despite the same HIP handle.
        if external != first:
            with (
                torch.cuda.stream(external),
                pytest.raises(PublicationError, match="compute stream"),
            ):
                group.publish((8, 8))
        else:
            with torch.cuda.stream(external):
                group.publish((None, None))
        torch.cuda._sleep(2_000_000)
        with torch.cuda.stream(alias):
            group.publish((8, 8))
        observed_first = [buf.gpu.clone() for buf in buffers]
        with pytest.raises(PublicationError, match="republish_reason"):
            buffers[0].copy_to_gpu()
        owner.finish()
    # begin must wait for old source reads and refresh identity on a new stream.
    with torch.cuda.stream(external):
        owner.begin()
        group.publish((8, 8))
        owner.finish()
    with torch.cuda.stream(second):
        owner.begin()
        for i, buf in enumerate(buffers):
            buf.cpu.fill_(21 + i)
        with (
            torch.cuda.stream(first),
            pytest.raises(PublicationError, match="compute stream"),
        ):
            group.publish((8, 8))
        group.publish((8, 8))
        observed_second = [buf.gpu.clone() for buf in buffers]
        owner.finish()
    owner.completion.synchronize()
    assert [buf.cpu().tolist() for buf in observed_first] == [[11] * 8, [12] * 8]
    assert [buf.cpu().tolist() for buf in observed_second] == [[21] * 8, [22] * 8]


@gpu
@pytest.mark.parametrize("fast", [False, True], ids=["public", "tuple"])
@pytest.mark.parametrize("backend", ["direct", "packed"])
def test_capture_and_other_thread_reject_without_consuming_publication(fast, backend):
    owner, buffers, group = stream_setup(fast, backend)
    owner.begin()
    group.publish((8, 8))
    owner.finish()
    owner.begin()
    for i, buf in enumerate(buffers):
        buf.cpu.fill_(31 + i)
    with (
        ThreadPoolExecutor(max_workers=1) as worker,
        pytest.raises(PublicationError, match="owner thread"),
    ):
        worker.submit(group.publish, (8, 8)).result()
    output = torch.empty_like(buffers[0].gpu)
    graph = torch.cuda.CUDAGraph()
    torch.cuda.synchronize()
    with torch.cuda.graph(graph):
        output.copy_(buffers[0].gpu)
        with pytest.raises(PublicationError, match="actual graph capture"):
            group.publish((8, 8))
    group.publish((8, 8))
    owner.finish()
    graph.replay()
    torch.cuda.synchronize()
    assert output.cpu().tolist() == [31] * 8


@gpu
@pytest.mark.parametrize("fast", [False, True], ids=["public", "tuple"])
def test_other_device_rejects_before_publication(fast):
    if torch.cuda.device_count() < 2:
        pytest.skip("requires two GPU devices")
    with torch.cuda.device(0):
        owner, buffers, group = stream_setup(fast, "direct")
        owner.begin()
        buffers[0].cpu.fill_(41)
        buffers[1].cpu.fill_(42)
        with (
            torch.cuda.device(1),
            pytest.raises(PublicationError, match="wrong device"),
        ):
            group.publish((8, 8))
        group.publish((8, 8))
        owner.finish()
        owner.completion.synchronize()
        assert [buf.gpu.cpu().tolist() for buf in buffers] == [[41] * 8, [42] * 8]
