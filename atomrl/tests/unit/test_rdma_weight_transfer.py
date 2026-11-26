# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

import json
import typing

import atomrl.engine.inference.rdma_weight_transfer as rdma
import pytest
import torch
from atomrl.engine.inference.vllm_fp8_utils import (
    ReloadFingerprintTracker,
    TensorNameClass,
    classify_tensor_name,
    fingerprint_tensor,
    snapshot_model_fingerprints,
)

from atom.rollout import rdma_protocol


def _bucket(name, tensor):
    payload = tensor.contiguous().view(torch.uint8).reshape(-1)
    metadata = json.dumps(
        [
            {
                "name": name,
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype).removeprefix("torch."),
                "offset": 0,
                "nbytes": payload.numel(),
            }
        ],
        separators=(",", ":"),
    ).encode("utf-8")
    return metadata, payload


def test_receive_stream_calls_model_loader_once_across_all_buckets(monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("requires a GPU to verify receiver CPU staging")
    first = _bucket("first.weight", torch.tensor([1.0, 2.0]))
    second = _bucket("second.weight", torch.tensor([3.0]))
    headers = iter(
        [
            torch.tensor(
                [rdma_protocol.CMD_BUCKET, len(first[0]), first[1].numel(), 7]
            ),
            torch.tensor(
                [rdma_protocol.CMD_BUCKET, len(second[0]), second[1].numel(), 7]
            ),
            torch.tensor([rdma_protocol.CMD_END, 0, 0, 7]),
        ]
    )
    broadcasts = iter(
        [
            torch.tensor(list(first[0]), dtype=torch.uint8),
            first[1],
            torch.tensor(list(second[0]), dtype=torch.uint8),
            second[1],
        ]
    )
    monkeypatch.setattr(
        rdma, "_broadcast_header", lambda *args, **kwargs: next(headers)
    )
    monkeypatch.setattr(
        rdma.dist,
        "broadcast",
        lambda output, src, group: output.copy_(next(broadcasts)),
    )
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device: None)

    class Model:
        calls = 0
        received: typing.ClassVar = []

        def named_parameters(self):
            return iter(())

        def named_buffers(self):
            return iter(())

        def load_weights(self, weights):
            self.calls += 1
            for name, tensor in weights:
                assert tensor.device.type == "cpu"
                self.received.append((name, tensor.clone()))
            return {"first", "second"}

    model = Model()
    stats = rdma.receive_weight_stream(
        object(),
        model,
        device=torch.device("cuda", 0),
        expected_version=7,
        verify_full_load=False,
    )

    assert model.calls == 1
    assert [name for name, _ in model.received] == ["first.weight", "second.weight"]
    torch.testing.assert_close(model.received[0][1], torch.tensor([1.0, 2.0]))
    torch.testing.assert_close(model.received[1][1], torch.tensor([3.0]))
    assert stats["buckets"] == 2
    assert stats["weights"] == 2


def _receive_one_bucket(monkeypatch, staging):
    metadata, payload = _bucket("w.weight", torch.tensor([1.0, 2.0, 3.0]))
    headers = iter(
        [
            torch.tensor([rdma_protocol.CMD_BUCKET, len(metadata), payload.numel(), 5]),
            torch.tensor([rdma_protocol.CMD_END, 0, 0, 5]),
        ]
    )
    broadcasts = iter([torch.tensor(list(metadata), dtype=torch.uint8), payload])
    received_payloads = []

    def broadcast(output, src, group):
        output.copy_(next(broadcasts))
        received_payloads.append(output)

    monkeypatch.setattr(
        rdma, "_broadcast_header", lambda *args, **kwargs: next(headers)
    )
    monkeypatch.setattr(rdma.dist, "broadcast", broadcast)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device: None)

    class Model:
        received: typing.ClassVar = []

        def named_parameters(self):
            return iter(())

        def named_buffers(self):
            return iter(())

        def load_weights(self, weights):
            self.received.extend(weights)
            return {"w"}

    model = Model()
    rdma.receive_weight_stream(
        object(),
        model,
        device=torch.device("cpu"),
        expected_version=5,
        verify_full_load=False,
        staging=staging,
    )
    ((name, tensor),) = model.received
    return name, tensor, received_payloads[-1]


def test_receive_stream_without_staging_yields_views_into_the_payload(monkeypatch):
    name, tensor, payload = _receive_one_bucket(monkeypatch, "none")

    assert name == "w.weight"
    assert tensor.untyped_storage().data_ptr() == payload.untyped_storage().data_ptr()
    torch.testing.assert_close(tensor, torch.tensor([1.0, 2.0, 3.0]))


def test_receive_stream_gpu_staging_yields_copies(monkeypatch):
    _, tensor, payload = _receive_one_bucket(monkeypatch, "gpu")

    assert tensor.untyped_storage().data_ptr() != payload.untyped_storage().data_ptr()
    torch.testing.assert_close(tensor, torch.tensor([1.0, 2.0, 3.0]))


def test_receive_stream_rejects_unknown_staging_mode():
    with pytest.raises(ValueError, match="invalid RDMA staging mode"):
        rdma.receive_weight_stream(
            object(),
            object(),
            device=torch.device("cpu"),
            expected_version=1,
            staging="pinned",
        )


def test_receive_stream_reports_last_source_tensor_on_loader_failure(monkeypatch):
    bucket = _bucket("layers.0.attn.wq_b.weight_scale_inv", torch.ones(2, 3))
    headers = iter(
        [
            torch.tensor(
                [rdma_protocol.CMD_BUCKET, len(bucket[0]), bucket[1].numel(), 7]
            ),
            torch.tensor([rdma_protocol.CMD_END, 0, 0, 7]),
        ]
    )
    broadcasts = iter(
        [
            torch.tensor(list(bucket[0]), dtype=torch.uint8),
            bucket[1],
        ]
    )
    original_empty = torch.empty

    def cpu_empty(*args, **kwargs):
        kwargs["device"] = "cpu"
        return original_empty(*args, **kwargs)

    monkeypatch.setattr(
        rdma, "_broadcast_header", lambda *args, **kwargs: next(headers)
    )
    monkeypatch.setattr(
        rdma.dist,
        "broadcast",
        lambda output, src, group: output.copy_(next(broadcasts)),
    )
    monkeypatch.setattr(rdma.torch, "empty", cpu_empty)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device: None)

    class Model:
        def named_parameters(self):
            return iter(())

        def named_buffers(self):
            return iter(())

        def load_weights(self, weights):
            list(weights)
            raise AssertionError("shape mismatch")

    with pytest.raises(
        RuntimeError,
        match=(
            r"source=layers\.0\.attn\.wq_b\.weight_scale_inv; "
            r"shape=\(2, 3\); dtype=torch\.float32"
        ),
    ):
        rdma.receive_weight_stream(
            object(),
            Model(),
            device=torch.device("cuda", 0),
            expected_version=7,
            verify_full_load=False,
            streamed_scales=True,
        )


def _serve_buckets(monkeypatch, weights, version=7):
    """Feed one single-tensor bucket per weight to receive_weight_stream, on CPU."""
    buckets = [_bucket(name, tensor) for name, tensor in weights]
    headers = iter(
        [
            torch.tensor(
                [
                    rdma_protocol.CMD_BUCKET,
                    len(metadata),
                    payload.numel(),
                    version,
                ]
            )
            for metadata, payload in buckets
        ]
        + [torch.tensor([rdma_protocol.CMD_END, 0, 0, version])]
    )
    broadcasts = iter(
        [
            part
            for metadata, payload in buckets
            for part in (torch.tensor(list(metadata), dtype=torch.uint8), payload)
        ]
    )
    original_empty = torch.empty

    def cpu_empty(*args, **kwargs):
        kwargs["device"] = "cpu"
        return original_empty(*args, **kwargs)

    monkeypatch.setattr(
        rdma, "_broadcast_header", lambda *args, **kwargs: next(headers)
    )
    monkeypatch.setattr(
        rdma.dist,
        "broadcast",
        lambda output, src, group: output.copy_(next(broadcasts)),
    )
    monkeypatch.setattr(rdma.torch, "empty", cpu_empty)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device: None)


def test_receive_stream_loads_fsdp2_fused_experts_through_the_router(monkeypatch):
    from atomrl.engine.inference.vllm_moe_weight_sync import FusedMoEWeightRouter

    from tests.atomrl.test_moe_weight_sync import FakeModel, _fused_payload

    torch.manual_seed(0)
    model = FakeModel()
    weights = _fused_payload(model)
    _serve_buckets(monkeypatch, weights)

    stats = rdma.receive_weight_stream(
        object(),
        model,
        device=torch.device("cpu"),
        expected_version=7,
        verify_full_load=True,
        moe_router=FusedMoEWeightRouter(model),
    )

    sent = dict(weights)
    for idx, layer in enumerate(model.model.layers):
        prefix = f"model.layers.{idx}.mlp.experts"
        torch.testing.assert_close(
            layer.mlp.experts.w13_weight.data, sent[f"{prefix}.gate_up_proj"]
        )
        torch.testing.assert_close(
            layer.mlp.experts.w2_weight.data, sent[f"{prefix}.down_proj"]
        )
    assert stats["weights"] == len(weights)
    for key in (
        "wait_s",
        "recv_s",
        "host_copy_s",
        "observe_s",
        "route_s",
        "load_weights_s",
    ):
        assert stats[key] >= 0.0


def test_receive_stream_without_a_router_rejects_dropped_fused_experts(monkeypatch):
    from tests.atomrl.test_moe_weight_sync import FakeModel, _fused_payload

    model = FakeModel()
    _serve_buckets(monkeypatch, _fused_payload(model))

    with pytest.raises(
        RuntimeError, match=r"Incomplete vLLM RDMA weight reload.*experts\.w13_weight"
    ):
        rdma.receive_weight_stream(
            object(),
            model,
            device=torch.device("cpu"),
            expected_version=7,
            verify_full_load=True,
        )


def test_reload_verification_ignores_generated_scales_and_static_router_table():
    expected = {
        "model.layers.0.attn.wq_b.weight",
        "model.layers.0.attn.wq_b.weight_scale_inv",
        "model.layers.0.ffn.gate.tid2eid",
        "model.layers.0.ffn.gate.e_score_correction_bias",
        "model.layers.0.ffn.shared_experts.weight",
    }
    loaded = {
        "model.layers.0.attn.wq_b.weight",
        "model.layers.0.ffn.shared_experts.weight",
    }

    assert rdma._missing_reloadable_names(expected, loaded) == []
    assert rdma._missing_reloadable_names(
        expected | {"model.layers.0.ffn.shared_experts.bias"},
        loaded,
    ) == ["model.layers.0.ffn.shared_experts.bias"]


def test_reload_verification_counts_experts_reported_above_routed_experts():
    # vLLM 0.31 keeps FusedMoE's buffers in a nested RoutedExperts, and its
    # loader reports them under the FusedMoE's own name.
    nested = "model.layers.0.mlp.experts.routed_experts.w13_weight"

    assert (
        rdma._missing_reloadable_names(
            {nested}, {"model.layers.0.mlp.experts.w13_weight"}
        )
        == []
    )
    assert rdma._missing_reloadable_names({nested}, set()) == [nested]


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("model.layers.0.ffn.gate.weight", TensorNameClass.RELOADABLE),
        ("model.layers.0.attn.wq_b.weight", TensorNameClass.RELOADABLE),
        ("model.layers.0.ffn.experts.0.w1.weight", TensorNameClass.RELOADABLE),
        ("model.lm_head.weight", TensorNameClass.RELOADABLE),
        ("model.layers.0.ffn.shared_experts.bias", TensorNameClass.RELOADABLE),
        ("model.layers.0.ffn.gate.tid2eid", TensorNameClass.CHECKPOINT_STATIC),
        (
            "model.layers.0.ffn.gate.e_score_correction_bias",
            TensorNameClass.CHECKPOINT_STATIC,
        ),
        ("model.layers.0.attn.wq_b.weight_scale_inv", TensorNameClass.GENERATED),
        (
            "model.layers.0.ffn.experts.routed_experts.w13_weight_scale_inv",
            TensorNameClass.GENERATED,
        ),
    ],
)
def test_classify_tensor_name(name, expected):
    assert classify_tensor_name(name) is expected


def test_gate_weight_is_never_skipped_by_reload_coverage():
    gate = "model.layers.0.ffn.gate.weight"
    assert rdma._missing_reloadable_names({gate}, set()) == [gate]


def test_mlp_gate_representative_is_not_displaced_by_fallback():
    fallback = "model.aaa.weight"
    gate = "model.layers.0.mlp.gate.weight"

    snapshot = snapshot_model_fingerprints(
        type(
            "Model",
            (),
            {
                "named_parameters": lambda self: iter(
                    [(fallback, torch.ones(2)), (gate, torch.ones(2))]
                ),
                "named_buffers": lambda self: iter(()),
            },
        )()
    )

    assert set(snapshot) == {fallback, gate}


def test_streamed_fp8_scales_become_reloadable_and_require_manifest_coverage():
    scale = "model.layers.0.attn.q.weight_scale_inv"

    assert classify_tensor_name(scale) is TensorNameClass.GENERATED
    assert (
        classify_tensor_name(scale, streamed_scales=True) is TensorNameClass.RELOADABLE
    )
    assert rdma._missing_reloadable_names(
        {scale},
        set(),
        streamed_scales=True,
    ) == [scale]


def test_tracker_only_requires_scales_for_explicit_prequantized_stream():
    model = _FingerprintModel()
    tracker = ReloadFingerprintTracker(model)

    assert tracker.classify_name("weight_scale_inv") is TensorNameClass.GENERATED
    tracker.observe_source([("weight_scale_inv", torch.tensor([2.0]))])

    assert tracker.streamed_scales is False
    assert tracker.classify_name("weight_scale_inv") is TensorNameClass.GENERATED

    tracker.observe_source([("weight", torch.ones(1, dtype=torch.float8_e4m3fn))])

    assert tracker.streamed_scales is False
    prequantized = ReloadFingerprintTracker(model, streamed_scales=True)
    prequantized.observe_source([("weight", torch.ones(1, dtype=torch.float8_e4m3fn))])
    assert prequantized.classify_name("weight_scale_inv") is TensorNameClass.RELOADABLE


def test_rdma_prequantized_scale_requires_loaded_manifest_entry(monkeypatch):
    weight = "model.layers.0.attn.q.weight"
    scale = "model.layers.0.attn.q.weight_scale_inv"
    weight_bucket = _bucket(weight, torch.ones(1, dtype=torch.float8_e4m3fn))
    scale_bucket = _bucket(scale, torch.tensor([2.0]))
    headers = iter(
        [
            torch.tensor(
                [
                    rdma_protocol.CMD_BUCKET,
                    len(weight_bucket[0]),
                    weight_bucket[1].numel(),
                    3,
                ]
            ),
            torch.tensor(
                [
                    rdma_protocol.CMD_BUCKET,
                    len(scale_bucket[0]),
                    scale_bucket[1].numel(),
                    3,
                ]
            ),
            torch.tensor([rdma_protocol.CMD_END, 0, 0, 3]),
        ]
    )
    broadcasts = iter(
        [
            torch.tensor(list(weight_bucket[0]), dtype=torch.uint8),
            weight_bucket[1],
            torch.tensor(list(scale_bucket[0]), dtype=torch.uint8),
            scale_bucket[1],
        ]
    )
    monkeypatch.setattr(
        rdma,
        "_broadcast_header",
        lambda *args, **kwargs: next(headers),
    )
    monkeypatch.setattr(
        rdma.dist,
        "broadcast",
        lambda output, src, group: output.copy_(next(broadcasts)),
    )

    class Model:
        def named_parameters(self):
            return iter([(weight, torch.ones(1)), (scale, torch.ones(1))])

        def named_buffers(self):
            return iter(())

        def load_weights(self, weights):
            list(weights)
            return {weight}

    with pytest.raises(RuntimeError, match="missing=.*weight_scale_inv"):
        rdma.receive_weight_stream(
            object(),
            Model(),
            device=torch.device("cpu"),
            expected_version=3,
            streamed_scales=True,
        )


def test_rdma_stream_mode_remains_authoritative_if_tracker_state_changes(monkeypatch):
    weight = "model.layers.0.attn.q.weight"
    scale = "model.layers.0.attn.q.weight_scale_inv"
    bucket = _bucket(weight, torch.ones(1))
    headers = iter(
        [
            torch.tensor(
                [rdma_protocol.CMD_BUCKET, len(bucket[0]), bucket[1].numel(), 3]
            ),
            torch.tensor([rdma_protocol.CMD_END, 0, 0, 3]),
        ]
    )
    broadcasts = iter(
        [
            torch.tensor(list(bucket[0]), dtype=torch.uint8),
            bucket[1],
        ]
    )
    monkeypatch.setattr(
        rdma,
        "_broadcast_header",
        lambda *args, **kwargs: next(headers),
    )
    monkeypatch.setattr(
        rdma.dist,
        "broadcast",
        lambda output, src, group: output.copy_(next(broadcasts)),
    )
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device: None)

    class Tracker:
        streamed_scales = False

        def observe_source(self, weights):
            list(weights)
            self.streamed_scales = True

    class Model:
        def named_parameters(self):
            return iter([(weight, torch.ones(1)), (scale, torch.ones(1))])

        def named_buffers(self):
            return iter(())

        def load_weights(self, weights):
            list(weights)
            return {weight}

    stats = rdma.receive_weight_stream(
        object(),
        Model(),
        device=torch.device("cpu"),
        expected_version=3,
        streamed_scales=False,
        fingerprint_tracker=Tracker(),
        finalize_fingerprints=False,
    )

    assert stats["loaded_internal"] == 1.0


def test_fingerprint_samples_a_bounded_number_of_elements():
    tensor = torch.arange(100_000, dtype=torch.float32).reshape(100, 1000)
    fingerprint = fingerprint_tensor(tensor, max_samples=17)

    assert fingerprint.shape == (100, 1000)
    assert fingerprint.dtype == "torch.float32"
    assert fingerprint.sample_count == 17


def test_model_snapshot_includes_representative_parameters_and_buffers():
    names = [
        "model.layers.0.attn.wq.weight",
        "model.layers.0.ffn.experts.0.w1.weight",
        "model.layers.0.ffn.gate.weight",
        "model.lm_head.weight",
        "model.layers.0.ffn.shared_experts.bias",
    ]
    buffers = [
        "model.layers.0.ffn.gate.tid2eid",
        "model.layers.0.attn.wq.weight_scale_inv",
    ]

    class Model:
        def named_parameters(self):
            return iter((name, torch.ones(2)) for name in names)

        def named_buffers(self):
            return iter((name, torch.ones(2)) for name in buffers)

    snapshot = snapshot_model_fingerprints(Model())

    assert set(snapshot) == set(names[:4] + buffers[:1])
    assert all(value.sample_count == 2 for value in snapshot.values())


def test_model_snapshot_bounds_reloadables_but_keeps_all_static_tensors():
    static_names = [f"model.layers.{index}.ffn.gate.tid2eid" for index in range(20)] + [
        f"model.layers.{index}.ffn.gate.e_score_correction_bias" for index in range(20)
    ]
    reloadable_names = [
        *(f"model.layers.{index}.attn.q.weight" for index in range(20)),
        *(f"model.layers.{index}.ffn.experts.0.w1.weight" for index in range(20)),
        *(f"model.layers.{index}.ffn.gate.weight" for index in range(20)),
        *(f"model.layers.{index}.norm.weight" for index in range(20)),
        *(f"model.extra_heads.{index}.weight" for index in range(20)),
        "model.lm_head.weight",
    ]
    generated_names = [
        f"model.layers.{index}.attn.q.weight_scale_inv" for index in range(20)
    ]

    class Model:
        def named_parameters(self):
            return iter(
                (name, torch.ones(2)) for name in reloadable_names + static_names
            )

        def named_buffers(self):
            return iter((name, torch.ones(2)) for name in generated_names)

    snapshot = snapshot_model_fingerprints(Model())

    assert set(static_names) <= set(snapshot)
    assert not set(generated_names) & set(snapshot)
    assert len(snapshot) <= len(static_names) + 5


def test_checkpoint_static_fingerprints_are_exact_not_sampled():
    name = "model.layers.0.ffn.gate.tid2eid"

    class Model:
        def named_parameters(self):
            return iter(())

        def named_buffers(self):
            return iter([(name, torch.arange(100))])

    snapshot = snapshot_model_fingerprints(Model(), max_samples=7)

    assert snapshot[name].sample_count == 100


def test_source_fingerprints_have_a_bounded_tracked_count():
    class Model:
        def named_parameters(self):
            return iter(
                (
                    f"model.layers.{index}.attn.q.weight",
                    torch.ones(2),
                )
                for index in range(100)
            )

        def named_buffers(self):
            return iter(())

    tracker = ReloadFingerprintTracker(Model())
    tracker.observe_source(
        (
            f"source.layers.{index}.attn.q.weight",
            torch.ones(2),
        )
        for index in range(100)
    )

    assert len(tracker.before) <= 5
    assert len(tracker.source) <= 5


def test_source_role_history_is_stable_when_stream_order_changes():
    model = _FingerprintModel()
    source = {
        "source.layers.0.attn.q.weight": torch.tensor([3.0]),
        "source.layers.1.attn.q.weight": torch.tensor([4.0]),
    }
    first = ReloadFingerprintTracker(model)
    first.observe_source(list(source.items())[::-1])
    first.finalize()

    second = ReloadFingerprintTracker(model)
    second.observe_source(list(source.items()))
    unchanged = second.finalize()

    assert unchanged["source_unchanged"] == 1
    assert unchanged["source_changed"] == 0

    source["source.layers.0.attn.q.weight"] = torch.tensor([9.0])
    third = ReloadFingerprintTracker(model)
    third.observe_source(list(source.items())[::-1])
    changed = third.finalize()

    assert changed["source_changed"] == 1


def test_disabling_internal_fingerprint_finalize_requires_external_tracker():
    with pytest.raises(ValueError, match="external fingerprint_tracker"):
        rdma.receive_weight_stream(
            object(),
            _FingerprintModel(),
            device=torch.device("cpu"),
            expected_version=1,
            finalize_fingerprints=False,
        )


class _FingerprintModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
        self.register_buffer("tid2eid", torch.tensor([1, 2], dtype=torch.int32))
        self.register_buffer("weight_scale_inv", torch.tensor([3.0]))


def test_static_tensor_change_fails_reload_verification():
    model = _FingerprintModel()
    tracker = ReloadFingerprintTracker(model)
    model.tid2eid[0] = 9

    assert tracker.checkpoint_static_changed_names() == ["tid2eid"]
    with pytest.raises(RuntimeError, match="checkpoint_static_changed.*tid2eid"):
        tracker.finalize()


def test_checkpoint_static_meta_tensor_is_reported_without_reading_data():
    model = _FingerprintModel()
    tracker = ReloadFingerprintTracker(model)
    model.tid2eid = torch.empty(2, dtype=torch.int32, device="meta")

    assert tracker.checkpoint_static_changed_names() == ["tid2eid"]


def test_tracker_restores_checkpoint_static_tensor_after_reload_lifecycle():
    model = _FingerprintModel()
    tracker = ReloadFingerprintTracker(model)
    expected = model.tid2eid.clone()
    model.tid2eid[0] = 9

    tracker.restore_checkpoint_static_tensors()

    torch.testing.assert_close(model.tid2eid, expected)
    assert tracker.finalize()["checkpoint_static_changed"] == []


def test_tracker_does_not_restore_nonpreserved_static_tensor():
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("tid2eid", torch.tensor([1], dtype=torch.int32))
            self.e_score_correction_bias = torch.nn.Parameter(torch.tensor([2.0]))

    model = Model()
    tracker = ReloadFingerprintTracker(model)
    model.tid2eid[0] = 9
    model.e_score_correction_bias.data[0] = 7

    tracker.restore_checkpoint_static_tensors()

    torch.testing.assert_close(model.tid2eid, torch.tensor([1], dtype=torch.int32))
    torch.testing.assert_close(
        model.e_score_correction_bias,
        torch.tensor([7.0]),
    )
    with pytest.raises(
        RuntimeError,
        match="checkpoint_static_changed.*e_score_correction_bias",
    ):
        tracker.finalize()


def test_generated_tensor_change_is_excluded_from_verification():
    model = _FingerprintModel()
    tracker = ReloadFingerprintTracker(model)
    model.weight_scale_inv.add_(1)

    summary = tracker.finalize()

    assert summary["generated_excluded"] == 1
    assert summary["classified"]["generated"] == 1
    assert summary["classified"]["checkpoint_static"] == 1
    assert summary["classified"]["reloadable"] == 1


def test_unchanged_source_allows_unchanged_reload_target():
    model = _FingerprintModel()
    source = [("attn", model.attn.detach().clone())]
    first = ReloadFingerprintTracker(model)
    first.observe_source(source)
    initial = first.finalize()

    assert initial["first_source_snapshot"] is True
    assert initial["exact_name_mappings"] == 1
    assert initial["exact_name_change_checks"] == 0

    second = ReloadFingerprintTracker(model)
    second.observe_source(source)
    summary = second.finalize()

    assert summary["source_unchanged"] == 1
    assert summary["reload_target_unchanged_failures"] == 0


def test_changed_same_name_source_requires_changed_reload_target():
    model = _FingerprintModel()
    first = ReloadFingerprintTracker(model)
    first.observe_source([("attn", torch.tensor([1.0, 2.0]))])
    first.finalize()

    second = ReloadFingerprintTracker(model)
    second.observe_source([("attn", torch.tensor([7.0, 8.0]))])

    with pytest.raises(RuntimeError, match="reload_target_unchanged.*attn"):
        second.finalize()


def test_redhat_ffn_gate_maps_to_resident_mlp_gate_by_canonical_shape():
    source_name = "model.layers.0.ffn.gate.weight"
    target_name = "model.layers.0.mlp.gate.weight"

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_parameter(
                "gate",
                torch.nn.Parameter(torch.tensor([[1.0, 2.0]])),
            )

        def named_parameters(self, *args, **kwargs):
            return iter([(target_name, self.gate)])

        def named_buffers(self, *args, **kwargs):
            return iter(())

    model = Model()
    tracker = ReloadFingerprintTracker(model)
    tracker.observe_source([(source_name, torch.tensor([[1.0, 2.0]]))])
    summary = tracker.finalize()

    assert summary["canonical_shape_mappings"] == 1
    assert summary["mapping_verification_scope"] == (
        "canonical-name+exact-shape change correspondence; "
        "no source/target value-tolerance parity"
    )


def test_changed_redhat_ffn_gate_requires_changed_resident_mlp_gate():
    source_name = "model.layers.0.ffn.gate.weight"
    target_name = "model.layers.0.mlp.gate.weight"

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_parameter(
                "gate",
                torch.nn.Parameter(torch.tensor([[1.0, 2.0]])),
            )

        def named_parameters(self, *args, **kwargs):
            return iter([(target_name, self.gate)])

        def named_buffers(self, *args, **kwargs):
            return iter(())

    model = Model()
    first = ReloadFingerprintTracker(model)
    first.observe_source([(source_name, torch.tensor([[1.0, 2.0]]))])
    first.finalize()

    second = ReloadFingerprintTracker(model)
    second.observe_source([(source_name, torch.tensor([[7.0, 8.0]]))])

    with pytest.raises(
        RuntimeError,
        match=r"reload_target_unchanged.*model\.layers\.0\.mlp\.gate\.weight",
    ):
        second.finalize()


def test_redhat_attn_maps_to_resident_self_attn_by_canonical_shape():
    source_name = "model.layers.0.attn.q_proj.weight"
    target_name = "model.layers.0.self_attn.q_proj.weight"

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_parameter(
                "q_proj",
                torch.nn.Parameter(torch.tensor([[1.0, 2.0]])),
            )

        def named_parameters(self, *args, **kwargs):
            return iter([(target_name, self.q_proj)])

        def named_buffers(self, *args, **kwargs):
            return iter(())

    tracker = ReloadFingerprintTracker(Model())
    tracker.observe_source([(source_name, torch.tensor([[1.0, 2.0]]))])

    summary = tracker.finalize()

    assert summary["canonical_shape_mappings"] == 1
    assert summary["mapping_verification_scope"] == (
        "canonical-name+exact-shape change correspondence; "
        "no source/target value-tolerance parity"
    )


def test_canonical_alias_requires_identical_shape():
    source_name = "model.layers.0.attn.q_proj.weight"
    target_name = "model.layers.0.self_attn.q_proj.weight"

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_parameter(
                "q_proj",
                torch.nn.Parameter(torch.ones(2, 2)),
            )

        def named_parameters(self, *args, **kwargs):
            return iter([(target_name, self.q_proj)])

        def named_buffers(self, *args, **kwargs):
            return iter(())

    tracker = ReloadFingerprintTracker(Model())
    tracker.observe_source([(source_name, torch.ones(4))])

    assert tracker.finalize()["canonical_shape_mappings"] == 0


def test_lm_head_uses_exact_name_and_shape_mapping():
    name = "model.lm_head.weight"

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_parameter(
                "lm_head",
                torch.nn.Parameter(torch.tensor([[1.0, 2.0]])),
            )

        def named_parameters(self, *args, **kwargs):
            return iter([(name, self.lm_head)])

        def named_buffers(self, *args, **kwargs):
            return iter(())

    tracker = ReloadFingerprintTracker(Model())
    tracker.observe_source([(name, torch.tensor([[1.0, 2.0]]))])

    summary = tracker.finalize()

    assert summary["exact_name_mappings"] == 1
    assert summary["canonical_shape_mappings"] == 0
    assert (
        "no source/target value-tolerance parity"
        in summary["mapping_verification_scope"]
    )


def test_lm_head_exact_mapping_requires_identical_shape():
    name = "model.lm_head.weight"

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_parameter(
                "lm_head",
                torch.nn.Parameter(torch.ones(2, 2)),
            )

        def named_parameters(self, *args, **kwargs):
            return iter([(name, self.lm_head)])

        def named_buffers(self, *args, **kwargs):
            return iter(())

    tracker = ReloadFingerprintTracker(Model())
    tracker.observe_source([(name, torch.ones(4))])

    assert tracker.finalize()["exact_name_mappings"] == 0
