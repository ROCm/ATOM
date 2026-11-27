# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

import inspect
import sys
from collections import UserDict
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf

try:
    import resource  # noqa: F401
except ModuleNotFoundError:
    sys.modules["resource"] = SimpleNamespace(
        RUSAGE_SELF=0,
        getrusage=lambda _: SimpleNamespace(ru_maxrss=0),
    )

from atomrl.core.protocol import DataProto
from atomrl.workers.actor_worker import AtomActorWorker


def _mock_checkpoint_control_group(monkeypatch) -> object:
    control_group = object()
    monkeypatch.setattr(
        torch.distributed,
        "new_group",
        lambda *, ranks, backend: control_group,
    )
    monkeypatch.setattr(
        torch.distributed,
        "destroy_process_group",
        lambda group: None,
    )
    return control_group


@pytest.mark.parametrize("worker_type", [AtomActorWorker])
def test_rank_local_checkpoint_format_bypasses_dist_checkpoint(
    worker_type,
    monkeypatch,
    tmp_path,
) -> None:
    module = torch.nn.Linear(2, 2)
    dist_save_calls = []
    engine = SimpleNamespace(
        module=module,
        optimizer=None,
        lr_scheduler=None,
        save_dist_checkpoint=lambda *args, **kwargs: dist_save_calls.append(
            (args, kwargs)
        ),
    )
    worker = object.__new__(worker_type)
    worker._engine = engine
    worker.rank = 0
    worker.world_size = 1
    monkeypatch.setenv("ATOMRL_CHECKPOINT_FORMAT", "rank_local")
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    assert worker.save_checkpoint(str(tmp_path), global_step=3)

    assert dist_save_calls == []
    assert (tmp_path / "model_world_size_1_rank_0.pt").is_file()
    extra = torch.load(
        tmp_path / "extra_state_world_size_1_rank_0.pt",
        map_location="cpu",
        weights_only=False,
    )
    assert extra["global_step"] == 3


@pytest.mark.parametrize("worker_type", [AtomActorWorker])
def test_rank_local_checkpoint_format_bypasses_dist_checkpoint_load(
    worker_type,
    monkeypatch,
    tmp_path,
) -> None:
    module = torch.nn.Linear(2, 2)
    expected_weight = module.weight.detach().clone()
    torch.save(module.state_dict(), tmp_path / "model_world_size_1_rank_0.pt")
    torch.save(
        {"global_step": 3, "lr_scheduler": None, "rng": {}},
        tmp_path / "extra_state_world_size_1_rank_0.pt",
    )
    dist_load_calls = []
    engine = SimpleNamespace(
        module=module,
        optimizer=None,
        lr_scheduler=None,
        load_dist_checkpoint=lambda *args, **kwargs: dist_load_calls.append(
            (args, kwargs)
        ),
    )
    worker = object.__new__(worker_type)
    worker._engine = engine
    worker.rank = 0
    worker.world_size = 1
    module.weight.data.zero_()
    monkeypatch.setenv("ATOMRL_CHECKPOINT_FORMAT", "rank_local")
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: False)

    assert worker.load_checkpoint(str(tmp_path)) == 3

    assert dist_load_calls == []
    assert torch.equal(module.weight, expected_weight)


@pytest.mark.parametrize("worker_type", [AtomActorWorker])
def test_rank_local_checkpoint_serializes_ranks_with_barriers(
    worker_type,
    monkeypatch,
    tmp_path,
) -> None:
    worker = object.__new__(worker_type)
    worker._engine = SimpleNamespace(
        module=torch.nn.Linear(2, 2),
        optimizer=None,
        lr_scheduler=None,
    )
    worker.rank = 1
    worker.world_size = 2
    barriers = []
    monkeypatch.setenv("ATOMRL_CHECKPOINT_FORMAT", "rank_local")
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(
        torch.distributed,
        "all_gather_object",
        lambda output, _, *, group=None: output.__setitem__(
            slice(None),
            ["node-a", "node-a"],
        ),
    )
    monkeypatch.setattr(torch.distributed, "barrier", lambda: barriers.append(1))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    assert worker.save_checkpoint(str(tmp_path), global_step=3)

    assert len(barriers) == 2
    assert (tmp_path / "model_world_size_2_rank_1.pt").is_file()


@pytest.mark.parametrize("worker_type", [AtomActorWorker])
def test_rank_local_checkpoint_saves_one_rank_per_node_in_parallel(
    worker_type,
    monkeypatch,
    tmp_path,
) -> None:
    worker = object.__new__(worker_type)
    worker._engine = SimpleNamespace(
        module=torch.nn.Linear(2, 2),
        optimizer=None,
        lr_scheduler=None,
    )
    worker.rank = 1
    worker.world_size = 4
    barriers = []
    monkeypatch.setenv("ATOMRL_CHECKPOINT_FORMAT", "rank_local")
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(
        torch.distributed,
        "all_gather_object",
        lambda output, _, *, group=None: output.__setitem__(
            slice(None),
            ["node-a", "node-a", "node-b", "node-b"],
        ),
    )
    monkeypatch.setattr(torch.distributed, "barrier", lambda: barriers.append(1))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    assert worker.save_checkpoint(str(tmp_path), global_step=3)

    assert len(barriers) == 2
    assert (tmp_path / "model_world_size_4_rank_1.pt").is_file()


@pytest.mark.parametrize("worker_type", [AtomActorWorker])
def test_rank_local_checkpoint_load_preserves_restored_master_params(
    worker_type,
    monkeypatch,
    tmp_path,
) -> None:
    class OptimizerStub:
        def __init__(self):
            self.reload_calls = 0

        def state_dict(self):
            return {"step": 1}

        def load_state_dict(self, state):
            assert state == {"step": 1}

        def load_parameter_state(self, path):
            assert path.endswith("optim_parameter_state_world_size_1_rank_0.pt")

        def reload_model_params(self):
            self.reload_calls += 1

    module = torch.nn.Linear(2, 2)
    optimizer = OptimizerStub()
    torch.save(module.state_dict(), tmp_path / "model_world_size_1_rank_0.pt")
    torch.save(optimizer.state_dict(), tmp_path / "optim_world_size_1_rank_0.pt")
    (tmp_path / "optim_parameter_state_world_size_1_rank_0.pt").touch()
    torch.save(
        {"global_step": 3, "lr_scheduler": None, "rng": {}},
        tmp_path / "extra_state_world_size_1_rank_0.pt",
    )
    worker = object.__new__(worker_type)
    worker._engine = SimpleNamespace(
        module=module,
        optimizer=optimizer,
        lr_scheduler=None,
    )
    worker.rank = 0
    worker.world_size = 1
    monkeypatch.setenv("ATOMRL_CHECKPOINT_FORMAT", "rank_local")
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: False)

    assert worker.load_checkpoint(str(tmp_path)) == 3

    assert optimizer.reload_calls == 0


@pytest.mark.parametrize("worker_type", [AtomActorWorker])
def test_rank_local_checkpoint_patches_hybrid_optimizer_before_load(
    worker_type,
    monkeypatch,
    tmp_path,
) -> None:
    calls = []

    class OptimizerStub:
        def load_state_dict(self, state):
            assert state == {"step": 1}
            calls.append("load_optimizer")

        def load_parameter_state(self, path):
            assert path.endswith("optim_parameter_state_world_size_1_rank_0.pt")

    module = torch.nn.Linear(2, 2)
    optimizer = OptimizerStub()
    torch.save(module.state_dict(), tmp_path / "model_world_size_1_rank_0.pt")
    torch.save({"step": 1}, tmp_path / "optim_world_size_1_rank_0.pt")
    (tmp_path / "optim_parameter_state_world_size_1_rank_0.pt").touch()
    torch.save(
        {"global_step": 3, "lr_scheduler": None, "rng": {}},
        tmp_path / "extra_state_world_size_1_rank_0.pt",
    )
    worker = object.__new__(worker_type)
    worker._engine = SimpleNamespace(
        module=module,
        optimizer=optimizer,
        lr_scheduler=None,
        _patch_hybrid_optimizer_checkpoint_load=lambda: calls.append(
            "patch_hybrid_optimizer"
        ),
    )
    worker.rank = 0
    worker.world_size = 1
    monkeypatch.setenv("ATOMRL_CHECKPOINT_FORMAT", "rank_local")
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: False)

    assert worker.load_checkpoint(str(tmp_path)) == 3
    assert calls == ["patch_hybrid_optimizer", "load_optimizer"]


@pytest.mark.parametrize("worker_type", [AtomActorWorker])
def test_rank_local_checkpoint_propagates_peer_load_failure(
    worker_type,
    monkeypatch,
    tmp_path,
) -> None:
    module = torch.nn.Linear(2, 2)
    torch.save(module.state_dict(), tmp_path / "model_world_size_2_rank_1.pt")
    torch.save(
        {"global_step": 3, "lr_scheduler": None, "rng": {}},
        tmp_path / "extra_state_world_size_2_rank_1.pt",
    )
    worker = object.__new__(worker_type)
    worker._engine = SimpleNamespace(
        module=module,
        optimizer=None,
        lr_scheduler=None,
    )
    worker.rank = 1
    worker.world_size = 2
    barriers = []

    def fake_all_gather_object(output, value, *, group=None):
        if isinstance(value, str):
            output[:] = ["node-a", "node-a"]
        else:
            output[:] = [
                {"rank": 0, "type": "KeyError", "message": "missing master param"},
                None,
            ]

    monkeypatch.setenv("ATOMRL_CHECKPOINT_FORMAT", "rank_local")
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    _mock_checkpoint_control_group(monkeypatch)
    monkeypatch.setattr(
        torch.distributed,
        "all_gather_object",
        fake_all_gather_object,
    )
    monkeypatch.setattr(torch.distributed, "barrier", lambda: barriers.append(1))

    with pytest.raises(
        RuntimeError,
        match="checkpoint phase failed.*rank 0.*KeyError.*missing master param",
    ):
        worker.load_checkpoint(str(tmp_path))

    assert barriers == []


@pytest.mark.parametrize("worker_type", [AtomActorWorker])
def test_rank_local_checkpoint_load_serializes_ranks(
    worker_type,
    monkeypatch,
    tmp_path,
) -> None:
    module = torch.nn.Linear(2, 2)
    torch.save(module.state_dict(), tmp_path / "model_world_size_2_rank_1.pt")
    torch.save(
        {"global_step": 3, "lr_scheduler": None, "rng": {}},
        tmp_path / "extra_state_world_size_2_rank_1.pt",
    )
    worker = object.__new__(worker_type)
    worker._engine = SimpleNamespace(
        module=module,
        optimizer=None,
        lr_scheduler=None,
    )
    worker.rank = 1
    worker.world_size = 2
    barriers = []
    phase_syncs = []

    def fake_all_gather_object(output, value, *, group=None):
        if isinstance(value, str):
            output[:] = ["node-a", "node-a"]
        else:
            phase_syncs.append(value)
            output[:] = [None, None]

    monkeypatch.setenv("ATOMRL_CHECKPOINT_FORMAT", "rank_local")
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    _mock_checkpoint_control_group(monkeypatch)
    monkeypatch.setattr(
        torch.distributed,
        "all_gather_object",
        fake_all_gather_object,
    )
    monkeypatch.setattr(torch.distributed, "barrier", lambda: barriers.append(1))

    assert worker.load_checkpoint(str(tmp_path)) == 3

    assert phase_syncs == [None, None]
    assert barriers == []


def test_dsv4_indexer_launch_tuning_reaches_engine_config() -> None:
    worker = object.__new__(AtomActorWorker)
    worker.config = {}
    training_cfg = {
        "megatron_cfg": {
            "v4_indexer_block_n": 64,
            "v4_indexer_num_stages": 1,
        }
    }

    engine_config = worker._build_engine_config("megatron_lumen_dsv4", training_cfg, {})

    assert engine_config["v4_indexer_block_n"] == 64
    assert engine_config["v4_indexer_num_stages"] == 1


@pytest.mark.parametrize("worker_type", [AtomActorWorker])
def test_precision_aware_optimizer_reaches_dsv4_engine_config(worker_type) -> None:
    worker = object.__new__(worker_type)
    worker.config = {}

    engine_config = worker._build_engine_config(
        "megatron_lumen_dsv4",
        {"megatron_cfg": {"use_precision_aware_optimizer": True}},
        {},
    )

    assert engine_config["use_precision_aware_optimizer"] is True


@pytest.mark.parametrize("worker_type", [AtomActorWorker])
def test_streamed_optimizer_settings_reach_dsv4_engine_config(worker_type) -> None:
    worker = object.__new__(worker_type)
    worker.config = {}

    engine_config = worker._build_engine_config(
        "megatron_lumen_dsv4",
        {
            "megatron_cfg": {
                "streamed_optimizer_mode": "adam",
                "streamed_optimizer_chunk_size_mib": 64,
                "streamed_optimizer_moment_dtype": "bf16",
            }
        },
        {},
    )

    assert engine_config["streamed_optimizer_mode"] == "adam"
    assert engine_config["streamed_optimizer_chunk_size_mib"] == 64
    assert engine_config["streamed_optimizer_moment_dtype"] == "bf16"


@pytest.mark.parametrize("worker_type", [AtomActorWorker])
@pytest.mark.parametrize(
    ("megatron_cfg", "expected_mode", "expected_chunk_size_mib"),
    [
        ({}, "off", 256),
        (
            {
                "streamed_optimizer_mode": "ADAM",
                "streamed_optimizer_chunk_size_mib": "64",
            },
            "adam",
            64,
        ),
    ],
)
def test_streamed_optimizer_settings_are_normalized_with_defaults(
    worker_type,
    megatron_cfg,
    expected_mode,
    expected_chunk_size_mib,
) -> None:
    worker = object.__new__(worker_type)
    worker.config = {}

    engine_config = worker._build_engine_config(
        "megatron_lumen_dsv4",
        {"megatron_cfg": megatron_cfg},
        {},
    )

    assert engine_config["streamed_optimizer_mode"] == expected_mode
    assert engine_config["streamed_optimizer_chunk_size_mib"] == expected_chunk_size_mib


@pytest.mark.parametrize("worker_type", [AtomActorWorker])
@pytest.mark.parametrize(
    "config_factory",
    [
        pytest.param(OmegaConf.create, id="dict-config"),
        pytest.param(UserDict, id="user-dict"),
    ],
)
def test_mapping_megatron_config_reaches_dsv4_engine_config(
    worker_type,
    config_factory,
) -> None:
    worker = object.__new__(worker_type)
    worker.config = {}
    megatron_cfg = config_factory(
        {
            "streamed_optimizer_mode": "ADAM",
            "streamed_optimizer_chunk_size_mib": "64",
            "tensor_model_parallel_size": 8,
        }
    )

    engine_config = worker._build_engine_config(
        "megatron_lumen_dsv4",
        {"megatron_cfg": megatron_cfg},
        {},
    )

    assert engine_config["streamed_optimizer_mode"] == "adam"
    assert engine_config["streamed_optimizer_chunk_size_mib"] == 64
    assert engine_config["tensor_model_parallel_size"] == 8


@pytest.mark.parametrize("worker_type", [AtomActorWorker])
@pytest.mark.parametrize(
    "config_factory",
    [
        pytest.param(dict, id="dict"),
        pytest.param(OmegaConf.create, id="dict-config"),
    ],
)
@pytest.mark.parametrize("invalid_chunk_size_mib", [True, 1.5])
def test_streamed_optimizer_rejects_lossy_chunk_size_coercion(
    worker_type,
    config_factory,
    invalid_chunk_size_mib,
) -> None:
    worker = object.__new__(worker_type)
    worker.config = {}
    megatron_cfg = config_factory(
        {"streamed_optimizer_chunk_size_mib": invalid_chunk_size_mib}
    )

    with pytest.raises(
        (TypeError, ValueError),
        match="streamed_optimizer_chunk_size_mib",
    ):
        worker._build_engine_config(
            "megatron_lumen_dsv4",
            {"megatron_cfg": megatron_cfg},
            {},
        )


@pytest.mark.parametrize("worker_type", [AtomActorWorker])
@pytest.mark.parametrize(
    "config_factory",
    [
        pytest.param(dict, id="dict"),
        pytest.param(OmegaConf.create, id="dict-config"),
    ],
)
@pytest.mark.parametrize("chunk_size_mib", ["64", 64])
def test_streamed_optimizer_normalizes_integer_chunk_size(
    worker_type,
    config_factory,
    chunk_size_mib,
) -> None:
    worker = object.__new__(worker_type)
    worker.config = {}
    megatron_cfg = config_factory({"streamed_optimizer_chunk_size_mib": chunk_size_mib})

    engine_config = worker._build_engine_config(
        "megatron_lumen_dsv4",
        {"megatron_cfg": megatron_cfg},
        {},
    )

    assert engine_config["streamed_optimizer_chunk_size_mib"] == 64
    assert isinstance(engine_config["streamed_optimizer_chunk_size_mib"], int)


@pytest.mark.parametrize("worker_type", [AtomActorWorker])
def test_actor_grpo_loss_uses_miles_asymmetric_sequence_mean(worker_type) -> None:
    worker = object.__new__(worker_type)
    worker._engine = None
    worker._pol_meta = {
        "algorithm": "grpo",
        "algo_config": {
            "loss_agg_mode": "seq-mean-token-mean",
            "clip_ratio_high": 0.28,
            "grpo": {"clip_ratio": 0.2},
        },
        "global_batch_size": 4,
        "dp_size": 2,
    }
    ratios = torch.tensor([[2.0, 1.0], [0.5, 0.5]])
    log_probs = ratios.log().requires_grad_(True)
    data = {
        "old_log_probs": torch.zeros_like(log_probs),
        "advantages": torch.tensor([1.0, -1.0]),
        "response_mask": torch.tensor([[1, 0], [1, 1]]),
    }

    loss, metrics = worker._policy_loss_fn({"log_probs": log_probs}, data)

    torch.testing.assert_close(loss, torch.tensor((-1.28 + 0.8) / 2))
    assert metrics["loss"] == pytest.approx(float(loss.detach()))


@pytest.mark.parametrize("worker_type", [AtomActorWorker])
def test_actor_grpo_loss_applies_rollout_importance_weights(worker_type) -> None:
    worker = object.__new__(worker_type)
    worker._engine = None
    worker._pol_meta = {
        "algorithm": "grpo",
        "algo_config": {
            "loss_agg_mode": "seq-mean-token-mean",
            "clip_ratio_high": 0.28,
            "grpo": {"clip_ratio": 0.2},
        },
        "global_batch_size": 2,
        "dp_size": 1,
    }
    log_probs = torch.tensor([[2.0, 1.0], [0.5, 0.5]]).log().requires_grad_(True)
    data = {
        "old_log_probs": torch.zeros_like(log_probs),
        "advantages": torch.tensor([1.0, -1.0]),
        "response_mask": torch.tensor([[1, 0], [1, 1]]),
        "rollout_is_weights": torch.tensor([[0.5, 0.0], [2.0, 2.0]]),
    }

    loss, metrics = worker._policy_loss_fn({"log_probs": log_probs}, data)

    torch.testing.assert_close(loss, torch.tensor((-1.28 * 0.5 + 0.8 * 2.0) / 2))
    assert metrics["loss"] == pytest.approx(float(loss.detach()))


@pytest.mark.parametrize("worker_type", [AtomActorWorker])
def test_actor_legacy_grpo_loss_applies_rollout_importance_weights(
    worker_type,
) -> None:
    source = inspect.getsource(worker_type._policy_loss_fn)
    grpo_branch = source.split("elif algo_name == AlgorithmName.GRPO.value:", 1)[
        1
    ].split("\n        else:", 1)[0]

    assert "rollout_is_weights=ris" in grpo_branch


@pytest.mark.parametrize("worker_type", [AtomActorWorker])
def test_actor_grpo_reports_sum_of_global_microbatch_contributions(
    worker_type,
) -> None:
    class FakeEngine:
        @staticmethod
        def train_mode():
            return nullcontext()

        @staticmethod
        def train_batch(data, loss_fn):
            return {
                "loss": [0.2, 0.3],
                "metrics": {"loss": [0.2, 0.3]},
            }

        @staticmethod
        def lr_scheduler_step():
            return 1.0e-6

    worker = object.__new__(worker_type)
    worker._engine = FakeEngine()
    worker._logged_norm = True
    worker.config = {}
    batch = DataProto(
        tensors={"input_ids": torch.ones(2, 2, dtype=torch.long)},
        meta={"algorithm": "grpo"},
    )

    metrics = worker.update_policy(batch)

    assert metrics["loss"] == pytest.approx(0.5)
