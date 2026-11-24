# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""Unit tests for the shared Megatron-Native engine helpers."""

from __future__ import annotations

import inspect
import os
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

try:
    import resource  # noqa: F401
except ModuleNotFoundError:
    sys.modules["resource"] = SimpleNamespace(
        RUSAGE_SELF=0,
        getrusage=lambda _: SimpleNamespace(ru_maxrss=0),
    )

import atomrl.engine.training  # noqa: F401 - populate EngineRegistry
from atomrl.algorithms.loss_functions import asymmetric_clip_loss
from atomrl.core.config import OptimizerConfig
from atomrl.core.protocol import DataProto
from atomrl.engine.training import megatron_engine, megatron_lumen_dsv4_engine
from atomrl.engine.training.base_engine import EngineRegistry
from atomrl.engine.training.megatron_base_engine import (
    MegatronBaseEngine,
    _FusedTokenLogProb,
    moe_dispatcher_kwargs,
)
from atomrl.engine.training.megatron_native_engine import MegatronNativeEngine
from atomrl.workers.actor_worker import AtomActorWorker


def test_megatron_base_grpo_applies_rollout_importance_weights() -> None:
    source = inspect.getsource(MegatronBaseEngine._row_policy_loss)
    grpo_branch = source.split("elif algo_name == AlgorithmName.GRPO.value:", 1)[
        1
    ].split("\n        else:", 1)[0]

    assert "rollout_is_weights=ris" in grpo_branch


def test_megatron_backends_are_registered_for_existing_configs() -> None:
    language_backends = EngineRegistry._engines["language_model"]
    value_backends = EngineRegistry._engines["value_model"]

    assert "cuda" in language_backends["megatron"]
    assert "cuda" in value_backends["megatron"]
    assert "megatron_native" in language_backends
    assert "megatron_native" in value_backends


def test_native_engine_uses_shared_base() -> None:
    assert issubclass(MegatronNativeEngine, MegatronBaseEngine)


def test_megatron_engine_checkpoint_uses_hdo_compatible_optimizer_state() -> None:
    model_state = {"model": object()}
    optimizer_state = {"optimizer": object()}

    class ModuleStub:
        def sharded_state_dict(self):
            return model_state

    class OptimizerStub:
        def sharded_state_dict(self, state, *, is_loading, metadata):
            assert state is model_state
            assert is_loading is False
            assert metadata == {"distrib_optim_sharding_type": "dp_zero_gather_scatter"}
            return optimizer_state

    engine = megatron_engine.MegatronEngine.__new__(megatron_engine.MegatronEngine)
    engine.module = ModuleStub()
    engine.optimizer = OptimizerStub()

    assert engine._dist_sharded_state_dict(False) == {
        "model": model_state,
        "optimizer": optimizer_state,
    }


def test_checkpoint_fingerprints_detect_tensor_data_mismatch() -> None:
    class ShardedTensorStub:
        def __init__(self, data):
            self.data = data

    saved = {
        "model": {"weight": ShardedTensorStub(torch.tensor([1.0, 2.0]))},
        "optimizer": {"exp_avg": torch.tensor([3.0, 4.0])},
    }
    restored = {
        "model": {"weight": ShardedTensorStub(torch.tensor([1.0, 2.0]))},
        "optimizer": {"exp_avg": torch.tensor([3.0, 5.0])},
    }

    expected = megatron_engine._checkpoint_tensor_fingerprints(saved)
    actual = megatron_engine._checkpoint_tensor_fingerprints(restored)

    with pytest.raises(RuntimeError, match=r"optimizer\.exp_avg"):
        megatron_engine._verify_checkpoint_tensor_fingerprints(
            expected,
            actual,
            stage="post-load",
        )


def test_checkpoint_fingerprints_hash_noncontiguous_tensors_in_bounded_chunks() -> None:
    contiguous = torch.arange(48, dtype=torch.float32).reshape(6, 8)
    noncontiguous = contiguous.t()

    expected = megatron_engine._checkpoint_tensor_fingerprints(
        {"weight": noncontiguous.contiguous()},
        chunk_bytes=16,
    )
    actual = megatron_engine._checkpoint_tensor_fingerprints(
        {"weight": noncontiguous},
        chunk_bytes=16,
    )

    assert not noncontiguous.is_contiguous()
    assert actual == expected


def test_checkpoint_fingerprint_manifest_round_trip(tmp_path) -> None:
    saved = {
        "model": {"weight": torch.tensor([1.0, 2.0])},
        "optimizer": {"exp_avg": torch.tensor([3.0, 4.0])},
    }
    restored = {
        "model": {"weight": torch.tensor([1.0, 2.0])},
        "optimizer": {"exp_avg": torch.tensor([3.0, 4.0])},
    }

    manifest = megatron_engine._save_checkpoint_fingerprint_manifest(
        tmp_path,
        saved,
        rank=7,
    )

    assert manifest.name == "tensor_fingerprints_rank_00007.json"
    megatron_engine._verify_checkpoint_fingerprint_manifest(
        tmp_path,
        restored,
        rank=7,
        stage="post-load",
    )


def test_dist_checkpoint_save_writes_verification_manifest(
    monkeypatch,
    tmp_path,
) -> None:
    state = {"model": {"weight": torch.tensor([1.0])}, "optimizer": {}}
    fake_megatron = ModuleType("megatron")
    fake_megatron.__path__ = []
    fake_core = ModuleType("megatron.core")
    fake_core.__path__ = []
    fake_dc = ModuleType("megatron.core.dist_checkpointing")
    fake_dc.save = lambda saved, path: None
    fake_core.dist_checkpointing = fake_dc
    fake_megatron.core = fake_core
    monkeypatch.setitem(sys.modules, "megatron", fake_megatron)
    monkeypatch.setitem(sys.modules, "megatron.core", fake_core)
    monkeypatch.setitem(
        sys.modules,
        "megatron.core.dist_checkpointing",
        fake_dc,
    )
    monkeypatch.setenv("ATOMRL_VERIFY_CHECKPOINT_ROUNDTRIP", "1")
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: False)

    engine = megatron_engine.MegatronEngine.__new__(megatron_engine.MegatronEngine)
    engine.lr_scheduler = None
    engine._dist_sharded_state_dict = lambda is_loading: state

    assert engine.save_dist_checkpoint(str(tmp_path), global_step=1)
    assert (tmp_path / "tensor_fingerprints_rank_00000.json").is_file()


def test_r3_validation_accepts_uint8_ids_for_256_experts() -> None:
    engine = object.__new__(megatron_engine.MegatronEngine)
    engine._dims = SimpleNamespace(num_experts=256)

    engine._r3_validate_expert_ids(torch.tensor([0, 1, 254, 255], dtype=torch.uint8))


def test_row_policy_loss_aligns_pre_shifted_response_mask_after_left_padding() -> None:
    engine = object.__new__(MegatronBaseEngine)
    token_log_probs = torch.zeros(1, 3, requires_grad=True)
    tensors = {
        # Global shifted-token columns.  The real sequence starts at column 2;
        # its three transitions are prompt, response, response.
        "old_log_probs": torch.zeros(1, 5),
        "response_mask": torch.tensor([[0, 0, 0, 1, 1]]),
        "advantages": torch.ones(1),
    }

    loss, _ = engine._row_policy_loss(
        tensors,
        r=0,
        start=2,
        token_lp=token_log_probs,
        algo_name="grpo",
        cfg_fn=lambda name, default: default,
        bnt=None,
        dp=1,
        loss_agg_mode="token-mean",
        global_batch_size=1,
    )
    assert loss is not None
    loss.backward()

    assert token_log_probs.grad is not None
    assert token_log_probs.grad[0, 0].item() == 0.0
    assert token_log_probs.grad[0, 1].item() != 0.0
    assert token_log_probs.grad[0, 2].item() != 0.0


def test_row_policy_loss_aligns_token_indexed_response_mask_after_left_padding() -> (
    None
):
    engine = object.__new__(MegatronBaseEngine)
    token_log_probs = torch.zeros(1, 3, requires_grad=True)
    tensors = {
        "input_ids": torch.tensor([[0, 0, 11, 12, 21, 22]]),
        # Token-indexed width S: response tokens occupy input positions 4 and 5.
        "response_mask": torch.tensor([[0, 0, 0, 0, 1, 1]]),
        "old_log_probs": torch.zeros(1, 5),
        "advantages": torch.ones(1),
    }

    loss, _ = engine._row_policy_loss(
        tensors,
        r=0,
        start=2,
        token_lp=token_log_probs,
        algo_name="grpo",
        cfg_fn=lambda name, default: default,
        bnt=None,
        dp=1,
        loss_agg_mode="token-mean",
        global_batch_size=1,
    )
    assert loss is not None
    loss.backward()

    assert token_log_probs.grad is not None
    assert token_log_probs.grad[0, 0].item() == 0.0
    assert token_log_probs.grad[0, 1].item() != 0.0
    assert token_log_probs.grad[0, 2].item() != 0.0


def test_native_pp_grpo_passes_sequence_normalization_under_dp_cp(
    monkeypatch,
) -> None:
    log_probs = torch.zeros(2, 2, requires_grad=True)
    scheduled_losses = []

    def get_forward_backward_func():
        def forward_backward(
            *,
            forward_step_func,
            data_iterator,
            model,
            num_microbatches,
            **_,
        ):
            output, loss_func = forward_step_func(data_iterator, model)
            scaled_loss, metrics = loss_func(output)
            scheduled_loss = scaled_loss * 2 / num_microbatches
            scheduled_losses.append(float(scheduled_loss.detach()))
            scheduled_loss.backward()
            return [metrics]

        return forward_backward

    megatron = ModuleType("megatron")
    megatron_core = ModuleType("megatron.core")
    pipeline_parallel = ModuleType("megatron.core.pipeline_parallel")
    pipeline_parallel.get_forward_backward_func = get_forward_backward_func
    megatron.core = megatron_core
    megatron_core.pipeline_parallel = pipeline_parallel
    monkeypatch.setitem(sys.modules, "megatron", megatron)
    monkeypatch.setitem(sys.modules, "megatron.core", megatron_core)
    monkeypatch.setitem(
        sys.modules, "megatron.core.pipeline_parallel", pipeline_parallel
    )

    class FakeModule:
        @staticmethod
        def train():
            return None

    class FakeDDP:
        @staticmethod
        def zero_grad_buffer():
            return None

    class FakeOptimizer:
        @staticmethod
        def zero_grad():
            return None

        @staticmethod
        def step():
            return True, torch.tensor(1.0), None

    engine = object.__new__(MegatronNativeEngine)
    engine._pp = 2
    engine._cp = 2
    engine._ep = 1
    engine._is_moe = False
    engine._is_last_stage = True
    engine._r3_enabled = False
    engine._r3_store = None
    engine.module = FakeModule()
    engine._ddp = FakeDDP()
    engine.optimizer = FakeOptimizer()
    engine._pp_setup_config = lambda: None
    engine.get_data_parallel_size = lambda: 2
    engine._collect_rows = lambda seqs, am: [(0, 0, 3), (1, 0, 3)]
    engine._build_microbatches = lambda seqs, rows: [
        {
            "rows": rows,
            "ids_list": [seqs[0], seqs[1]],
        }
    ]
    engine._pad_mbs_for_ep = lambda mbs: mbs
    engine._pp_forward_model = lambda model, ids: (
        torch.zeros(1, 1, requires_grad=True),
        [0, 1],
    )

    def cp_log_probs(logits, ids, row, **_):
        # Preserve the forward value while emulating CP all-reduce's backward SUM.
        return log_probs[row] * engine._cp, None

    engine._cp_row_logprob_entropy = cp_log_probs
    engine._sched_step = lambda: 1.0e-6
    engine._cur_lr = lambda: 1.0e-6
    batch = DataProto(
        tensors={
            "input_ids": torch.ones(2, 3, dtype=torch.long),
            "attention_mask": torch.ones(2, 3, dtype=torch.long),
            "old_log_probs": torch.zeros(2, 2),
            "advantages": torch.ones(2),
            "response_mask": torch.tensor([[0, 1, 1], [0, 1, 1]]),
        },
        meta={
            "algorithm": "grpo",
            "dp_size": 2,
            "algo_config": {
                "loss_agg_mode": "seq-mean-token-mean",
                "clip_ratio_high": 0.28,
                "grpo": {"clip_ratio": 0.2},
            },
        },
    )

    engine._pp_update_policy(batch)

    # Fallback global B is local_B * DP = 4. Two local sequence means each
    # contribute -1 * DP/global_B = -0.5; PP scheduling preserves their sum.
    assert scheduled_losses == pytest.approx([-1.0])
    torch.testing.assert_close(log_probs.grad, torch.full_like(log_probs, -0.5))


def test_actor_optimizer_config_propagates_optimizer_type() -> None:
    actor = AtomActorWorker(rank=0, world_size=1, config={})

    config = actor._build_optimizer_config(
        {"optimizer_type": "sgd", "sgd_momentum": 0.0}
    )

    assert config["optimizer_type"] == "sgd"
    assert config["sgd_momentum"] == 0.0

    # The same dict is what the FSDP2 engine builds its OptimizerConfig from, so
    # every key in it has to be a field on that dataclass.
    assert OptimizerConfig(**config).optimizer_type == "sgd"


def test_rdma_weight_sync_releases_cuda_cache_before_gather(monkeypatch) -> None:
    events = []

    class FakeEngine:
        @staticmethod
        def get_per_tensor_param():
            assert events == ["synchronize", "empty_cache"]
            return iter(()), None

    actor = AtomActorWorker(rank=1, world_size=2, config={})
    actor._engine = FakeEngine()
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: events.append("synchronize"))
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: events.append("empty_cache"))

    result = actor.send_weights_rdma(version=1, bucket_size_mb=1)

    assert result == {"writer": False}


def test_safetensors_export_can_select_diagnostic_weights(
    monkeypatch, tmp_path
) -> None:
    class FakeEngine:
        @staticmethod
        def get_per_tensor_param():
            return (
                iter(
                    [
                        ("embed.weight", torch.ones(2, 3)),
                        ("layers.0.ffn.gate.weight", torch.full((2, 3), 2.0)),
                        ("layers.0.ffn.experts.0.w1.weight", torch.full((2, 3), 3.0)),
                    ]
                ),
                None,
            )

    actor = AtomActorWorker(rank=0, world_size=1, config={})
    actor._engine = FakeEngine()
    monkeypatch.setattr(actor, "_ensure_weight_http_server", lambda _path: "")

    result = actor.export_state_dict_safetensors(
        str(tmp_path),
        include_names=["embed.weight", "layers.0.ffn.gate.weight"],
    )

    import json

    index = json.loads((tmp_path / "model.safetensors.index.json").read_text())
    assert set(index["weight_map"]) == {
        "embed.weight",
        "layers.0.ffn.gate.weight",
    }
    assert result["num_params"] == 2


def test_fused_token_log_prob_matches_reference_value_and_gradient() -> None:
    torch.manual_seed(7)
    target = torch.tensor([0, 3, 1, 4, 2], dtype=torch.long)
    weight = torch.randn(target.numel())

    fused_logits = torch.randn(target.numel(), 7, requires_grad=True)
    reference_logits = fused_logits.detach().clone().requires_grad_(True)

    fused = _FusedTokenLogProb.apply(fused_logits, target)
    reference = (
        torch.log_softmax(reference_logits, dim=-1)
        .gather(-1, target.unsqueeze(-1))
        .squeeze(-1)
    )

    torch.testing.assert_close(fused, reference)

    (fused * weight).sum().backward()
    (reference * weight).sum().backward()
    torch.testing.assert_close(fused_logits.grad, reference_logits.grad)


def test_pipeline_tokens_are_padded_to_tensor_parallel_alignment() -> None:
    token_ids = torch.tensor([11, 12, 13, 14, 15, 16])

    assert hasattr(megatron_engine, "_pad_token_ids_for_sequence_parallel")
    padded = megatron_engine._pad_token_ids_for_sequence_parallel(
        token_ids, tensor_parallel_size=4
    )

    assert padded.tolist() == [11, 12, 13, 14, 15, 16, 0, 0]
    assert padded.numel() % 4 == 0


def test_pipeline_tokens_are_padded_to_fixed_schedule_length() -> None:
    token_ids = torch.tensor([11, 12, 13, 14])

    assert hasattr(megatron_engine, "_pad_token_ids_to_pipeline_length")
    padded = megatron_engine._pad_token_ids_to_pipeline_length(
        token_ids, pipeline_sequence_length=8
    )

    assert padded.tolist() == [11, 12, 13, 14, 0, 0, 0, 0]


def test_fixed_pipeline_shapes_restore_dynamic_shape_config_on_error() -> None:
    config = SimpleNamespace(variable_seq_lengths=True)

    with (
        pytest.raises(RuntimeError, match="schedule failed"),
        megatron_engine._fixed_pipeline_shapes(config),
    ):
        assert config.variable_seq_lengths is False
        raise RuntimeError("schedule failed")

    assert config.variable_seq_lengths is True


def test_dsv4_pipeline_shape_adjuster_preserves_hyper_connection_streams() -> None:
    config = SimpleNamespace(dsv4_mode=True, dsv4_hc_mult=4)

    adjust = megatron_engine._pipeline_shape_adjuster(config)
    recv_shapes, send_shapes = adjust(
        [(1152, 1, 4096)],
        [(1152, 1, 4096)],
    )

    assert recv_shapes == [(1152, 1, 4, 4096)]
    assert send_shapes == [(1152, 1, 4, 4096)]


def test_pp_logprob_and_update_wire_dsv4_shape_adjuster() -> None:
    for method in (
        megatron_engine.MegatronEngine._engine_compute_log_probs_pp,
        megatron_engine.MegatronEngine._engine_update_policy_pp,
    ):
        source = inspect.getsource(method)
        assert "adjust_tensor_shapes_fn=_pipeline_shape_adjuster(config)" in source


def test_pipeline_logits_discard_sequence_parallel_padding() -> None:
    logits = torch.arange(8 * 3).reshape(8, 1, 3)

    assert hasattr(megatron_engine, "_flatten_pipeline_logits")
    flattened = megatron_engine._flatten_pipeline_logits(logits, unpadded_length=6)

    assert flattened.shape == (6, 3)
    torch.testing.assert_close(flattened, logits[:6, 0])


def test_grpo_rowwise_packed_and_pipeline_scaling_match() -> None:
    advantages = torch.tensor([[1.0, 3.0, 100.0], [2.0, 4.0, 6.0]])
    mask = torch.tensor([[1.0, 1.0, 0.0], [1.0, 1.0, 1.0]])

    packed_logp = torch.zeros(2, 3, requires_grad=True)
    packed_loss = asymmetric_clip_loss(
        packed_logp,
        torch.zeros_like(packed_logp),
        advantages,
        0.2,
        0.28,
        mask=mask,
        loss_agg_mode="seq-mean-token-mean",
        global_batch_size=2,
    )
    packed_loss.backward()

    rowwise_logp = torch.zeros(2, 3, requires_grad=True)
    row_losses = [
        asymmetric_clip_loss(
            rowwise_logp[row : row + 1],
            torch.zeros_like(rowwise_logp[row : row + 1]),
            advantages[row : row + 1],
            0.2,
            0.28,
            mask=mask[row : row + 1],
            loss_agg_mode="seq-mean-token-mean",
            global_batch_size=2,
        )
        for row in range(2)
    ]
    rowwise_loss = sum(row_losses)
    rowwise_loss.backward()

    pipeline_logp = torch.zeros(2, 3, requires_grad=True)
    pipeline_rows = [
        asymmetric_clip_loss(
            pipeline_logp[row : row + 1],
            torch.zeros_like(pipeline_logp[row : row + 1]),
            advantages[row : row + 1],
            0.2,
            0.28,
            mask=mask[row : row + 1],
            loss_agg_mode="seq-mean-token-mean",
            global_batch_size=2,
        )
        for row in range(2)
    ]
    # Megatron divides each returned microbatch loss by num_microbatches.
    pipeline_loss = (
        sum(megatron_engine._pipeline_schedule_loss(loss, 2) for loss in pipeline_rows)
        / 2
    )
    pipeline_loss.backward()

    torch.testing.assert_close(rowwise_loss, packed_loss)
    torch.testing.assert_close(pipeline_loss, packed_loss)
    torch.testing.assert_close(rowwise_logp.grad, packed_logp.grad)
    torch.testing.assert_close(pipeline_logp.grad, packed_logp.grad)


def test_dsv4_indexer_tuning_preserves_aiter_default(monkeypatch) -> None:
    monkeypatch.delenv("V4_INDEXER_BLOCK_N", raising=False)
    monkeypatch.delenv("V4_INDEXER_NUM_STAGES", raising=False)
    monkeypatch.delenv("V4_INDEXER_IMPL", raising=False)

    assert hasattr(megatron_lumen_dsv4_engine, "_configure_dsv4_indexer_environment")
    megatron_lumen_dsv4_engine._configure_dsv4_indexer_environment(
        {"v4_indexer_block_n": 64, "v4_indexer_num_stages": 1}
    )

    assert os.environ["V4_INDEXER_BLOCK_N"] == "64"
    assert os.environ["V4_INDEXER_NUM_STAGES"] == "1"
    assert "V4_INDEXER_IMPL" not in os.environ


def test_dsv4_sequence_alignment_includes_compressor_ratios() -> None:
    assert hasattr(megatron_lumen_dsv4_engine, "_dsv4_sequence_alignment")
    assert (
        megatron_lumen_dsv4_engine._dsv4_sequence_alignment(
            tensor_parallel_size=4, compress_ratios=[0, 4, 128]
        )
        == 128
    )


def test_shared_packing_helpers() -> None:
    engine = MegatronBaseEngine({}, {}, {}, "")

    bins = engine._build_bins([7, 5, 4, 2], budget=9)
    assert sorted(index for group in bins for index in group) == [0, 1, 2, 3]
    assert all(sum([7, 5, 4, 2][index] for index in group) <= 9 for group in bins)

    start, length = engine._real_block(torch.tensor([0, 0, 1, 1, 1, 0]))
    assert (start, length) == (2, 3)
    assert engine._real_block(torch.zeros(4, dtype=torch.long)) == (0, 0)


def test_moe_dispatcher_kwargs_alltoall_default() -> None:
    assert moe_dispatcher_kwargs({}, tp=1, cp=1, sp=False) == {
        "moe_token_dispatcher_type": "alltoall",
    }


def test_moe_dispatcher_kwargs_mori_auto_derives_heap() -> None:
    kwargs = moe_dispatcher_kwargs(
        {
            "moe_token_dispatcher_type": "flex",
            "moe_flex_dispatcher_backend": "mori",
        },
        tp=1,
        cp=1,
        sp=False,
        max_tokens_per_gpu=2048,
    )
    assert kwargs["moe_token_dispatcher_type"] == "flex"
    assert kwargs["moe_flex_dispatcher_backend"] == "mori"
    assert kwargs["moe_mori_max_tokens_per_rank"] == 2048


def test_moe_dispatcher_kwargs_mori_scales_for_cp_and_sp() -> None:
    kwargs = moe_dispatcher_kwargs(
        {
            "moe_token_dispatcher_type": "flex",
            "moe_flex_dispatcher_backend": "mori",
        },
        tp=2,
        cp=2,
        sp=True,
        max_tokens_per_gpu=2048,
    )
    # ceil(2048/2)=1024 for CP, then ceil(1024/2)=512 for SP.
    assert kwargs["moe_mori_max_tokens_per_rank"] == 512


def test_moe_dispatcher_kwargs_mori_explicit_heap_and_kernel() -> None:
    kwargs = moe_dispatcher_kwargs(
        {
            "moe_token_dispatcher_type": "flex",
            "moe_flex_dispatcher_backend": "mori",
            "moe_mori_max_tokens_per_rank": 4096,
            "moe_mori_kernel_type": "intranode",
        },
        tp=2,
        cp=2,
        sp=True,
        max_tokens_per_gpu=2048,
    )
    assert kwargs["moe_mori_max_tokens_per_rank"] == 4096
    assert kwargs["moe_mori_kernel_type"] == "intranode"
