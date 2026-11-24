# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

from __future__ import annotations

from dataclasses import fields
from pathlib import Path
from typing import get_type_hints

import pytest
from atomrl.core.config import (
    AtomRLConfig,
    WeightSyncConfig,
)
from atomrl.core.config import MegatronConfig as CoreMegatronConfig
from atomrl.core.config import (
    TeacherConfig as CoreTeacherConfig,
)
from atomrl.core.config import VLLMConfig as CoreVLLMConfig
from atomrl.core.types import AlgorithmName, GenerationBackend, TrainingBackend
from atomrl.engine.training.config import MegatronConfig as TrainingMegatronConfig
from atomrl.engine.training.config import TeacherConfig as TrainingTeacherConfig
from atomrl.engine.training.config import VLLMConfig as TrainingVLLMConfig
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[2]
GRPO_YAML = REPO_ROOT / "configs" / "grpo_dense_bf16.yaml"
DSV4_SMOKE_YAML = (
    REPO_ROOT / "examples" / "GRPO" / "configs" / "grpo_dsv4_flash_vllm_smoke.yaml"
)
DSV4_LONGRUN_YAML = (
    REPO_ROOT / "examples" / "GRPO" / "configs" / "grpo_dsv4_flash_vllm_longrun.yaml"
)
ACTOR_NODE_IPS = "192.0.2.10,192.0.2.11"
ROLLOUT_NODE_IPS = "192.0.2.12"


@pytest.fixture(autouse=True)
def _example_config_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    # The example configs resolve these through ${oc.env:...}.
    monkeypatch.setenv("DATA_ROOT", str(tmp_path))
    monkeypatch.setenv("ACTOR_NODE_IPS", ACTOR_NODE_IPS)
    monkeypatch.setenv("ROLLOUT_NODE_IPS", ROLLOUT_NODE_IPS)


def test_default_config() -> None:
    cfg = AtomRLConfig()
    assert cfg.num_training_steps == 1000
    assert cfg.seed == 42
    assert cfg.algorithm.name == AlgorithmName.GRPO.value
    assert cfg.policy.training_backend == TrainingBackend.FSDP2.value
    assert cfg.policy.generation_backend == GenerationBackend.ATOM.value
    assert cfg.policy.train_global_batch_size == 64
    assert cfg.policy.train_micro_batch_size == 8
    assert cfg.checkpointing.save_steps == 50
    assert cfg.logger.wandb_enabled is False
    assert cfg.assembly.training_backend == "fsdp2"
    assert cfg.assembly.inference_backend == "atom"


@pytest.mark.parametrize("config_type", [CoreTeacherConfig, TrainingTeacherConfig])
def test_teacher_config_rejects_removed_sglang_backend(config_type: type) -> None:
    with pytest.raises(ValueError, match="Unsupported teacher inference backend"):
        config_type(inference_backend="sglang")


@pytest.mark.parametrize("config_type", [CoreMegatronConfig, TrainingMegatronConfig])
def test_megatron_streamed_optimizer_defaults(config_type: type) -> None:
    cfg = config_type()

    assert cfg.streamed_optimizer_mode == "off"
    assert cfg.streamed_optimizer_chunk_size_mib == 256


def test_megatron_streamed_optimizer_schema_parity() -> None:
    core_fields = {field.name: field for field in fields(CoreMegatronConfig)}
    training_fields = {field.name: field for field in fields(TrainingMegatronConfig)}
    core_hints = get_type_hints(CoreMegatronConfig)
    training_hints = get_type_hints(TrainingMegatronConfig)

    for name, annotation, default in (
        ("streamed_optimizer_mode", str, "off"),
        ("streamed_optimizer_chunk_size_mib", int, 256),
        ("streamed_optimizer_moment_dtype", str, "fp32"),
    ):
        assert core_fields[name].name == training_fields[name].name == name
        assert core_hints[name] == training_hints[name] == annotation
        assert core_fields[name].default == training_fields[name].default == default


def test_vllm_prefix_cache_schema_parity() -> None:
    for config_type in (CoreVLLMConfig, TrainingVLLMConfig):
        assert config_type().enable_prefix_caching is False


def test_vllm_backend_defaults_are_the_leave_it_to_vllm_sentinel() -> None:
    # A second `moe_backend: str = ""` further down the same dataclass silently
    # replaced this default, and the trainer only skips "auto" -- so every MoE
    # config that does not name a backend asked vLLM for one called "", which an
    # unquantized MoE model rejects. One declaration, one default.
    for config_type in (CoreVLLMConfig, TrainingVLLMConfig):
        names = [field.name for field in fields(config_type)]
        assert len(names) == len(set(names)), (
            f"{config_type.__module__}.{config_type.__name__} declares a field twice: "
            f"{sorted({n for n in names if names.count(n) > 1})}"
        )
    assert CoreVLLMConfig().moe_backend == "auto"
    assert CoreVLLMConfig().linear_backend == "auto"


def test_from_yaml() -> None:
    assert GRPO_YAML.is_file(), f"Missing fixture config: {GRPO_YAML}"
    cfg = AtomRLConfig.from_yaml(GRPO_YAML)
    assert cfg.algorithm.name == "grpo"
    assert cfg.policy.model_name == "Qwen/Qwen3-0.6B"
    assert cfg.policy.max_total_sequence_length == 2048
    assert cfg.policy.train_global_batch_size == 32
    assert cfg.num_training_steps == 200
    assert cfg.reward.dataset == "nvidia/OpenMathInstruct-2"


def test_cli_overrides() -> None:
    cfg = AtomRLConfig.from_yaml(
        GRPO_YAML,
        overrides=[
            "seed=999",
            "policy.train_micro_batch_size=4",
            "logger.log_interval=10",
        ],
    )
    assert cfg.seed == 999
    assert cfg.policy.train_micro_batch_size == 4
    assert cfg.logger.log_interval == 10


def test_fp8_config_values() -> None:
    cfg = AtomRLConfig.from_yaml(GRPO_YAML)
    assert cfg.quantization.rollout.precision == "bf16"
    assert cfg.quantization.training.fp8 is None
    assert cfg.quantization.training.fp8_recipe == "blockwise"
    assert cfg.quantization.training.fp8_weight_cache is False

    merged = AtomRLConfig.from_yaml(
        GRPO_YAML,
        overrides=[
            "quantization.training.fp8=e4m3",
            "quantization.training.fp8_recipe=tensorwise",
            "quantization.training.fp8_weight_cache=true",
        ],
    )
    assert merged.quantization.training.fp8 == "e4m3"
    assert merged.quantization.training.fp8_recipe == "tensorwise"
    assert merged.quantization.training.fp8_weight_cache is True


def test_weight_sync_fp8_quantization_location() -> None:
    assert (
        WeightSyncConfig(
            fp8_quantization_location="trainer",
        ).resolve_fp8_quantize()
        is True
    )
    assert (
        WeightSyncConfig(
            fp8_quantization_location="inference",
        ).resolve_fp8_quantize()
        is False
    )
    assert WeightSyncConfig(fp8_quantize=True).resolve_fp8_quantize() is True
    assert WeightSyncConfig(fp8_quantize=False).resolve_fp8_quantize() is False


def test_weight_sync_fp8_quantization_location_rejects_invalid_or_conflicting() -> None:
    for config in (
        WeightSyncConfig(fp8_quantization_location="invalid"),
        WeightSyncConfig(
            fp8_quantization_location="inference",
            fp8_quantize=True,
        ),
    ):
        with pytest.raises(ValueError, match="fp8_quantization_location"):
            config.resolve_fp8_quantize()


def test_weight_sync_fp8_location_requires_a_quantizing_backend_and_fp8_rollout() -> (
    None
):
    for config, rollout_quantization in (
        (
            WeightSyncConfig(
                backend="shared_folder",
                fp8_quantization_location="trainer",
            ),
            "fp8_per_block",
        ),
        (
            WeightSyncConfig(
                backend="rdma",
                fp8_quantization_location="trainer",
            ),
            None,
        ),
        (
            WeightSyncConfig(
                backend="rdma",
                fp8_quantization_location="inference",
            ),
            "fp8",
        ),
    ):
        with pytest.raises(ValueError, match="fp8_quantization_location"):
            config.validate_fp8_quantization_location(rollout_quantization)


def test_weight_sync_fp8_location_accepts_valid_and_legacy_configs() -> None:
    WeightSyncConfig(
        backend="rdma",
        fp8_quantization_location="trainer",
    ).validate_fp8_quantization_location("fp8_per_block")
    # "auto" is the colocated CUDA-IPC path, which also quantizes on the trainer.
    WeightSyncConfig(
        backend="auto",
        fp8_quantization_location="trainer",
    ).validate_fp8_quantization_location("fp8_per_block")
    WeightSyncConfig(
        backend="auto",
        fp8_quantize=True,
    ).validate_fp8_quantization_location("fp8_per_block")
    WeightSyncConfig(
        backend="rdma",
        fp8_quantization_location="inference",
    ).validate_fp8_quantization_location("fp8_per_block")
    WeightSyncConfig().validate_fp8_quantization_location(None)


def test_moe_config_values() -> None:
    cfg = AtomRLConfig.from_yaml(GRPO_YAML)
    assert cfg.moe.r3.enabled is False
    assert cfg.moe.r3.record_router_logits is False
    assert cfg.moe.r3.replay_mode == "hard_assignment"
    assert cfg.moe.r3.print_train_rollout_mismatch is False

    schema = OmegaConf.structured(AtomRLConfig)
    assert OmegaConf.select(schema, "moe.r3.enabled") is not None


def test_ray_controller_config_defaults() -> None:
    cfg = AtomRLConfig()
    assert cfg.controller.ray.enabled is False
    assert cfg.controller.ray.actor.dispatch_mode == "dp_compute_proto"
    assert cfg.controller.ray.actor.detached is False
    assert cfg.controller.ray.actor.num_workers == 0
    assert cfg.controller.ray.actor.mesh_mapping is None
    assert cfg.controller.ray.fuse_actor_ref is False


def test_ray_controller_config_overrides() -> None:
    cfg = AtomRLConfig()
    cfg.controller.ray.enabled = True
    cfg.controller.ray.actor.dispatch_mode = "dp_compute_proto_with_func"
    cfg.controller.ray.actor.detached = True
    cfg.controller.ray.actor.num_workers = 4
    cfg.controller.ray.actor.mesh_mapping = [0, 0, 1, 1]
    cfg.controller.ray.actor.lazy_dispatch_key = "actor_mesh"
    cfg.controller.ray.ref.dispatch_mode = "rank_zero"
    cfg.controller.ray.fuse_actor_ref = True
    cfg.controller.ray.topology_map["actor"] = "actor"

    assert cfg.controller.ray.enabled is True
    assert cfg.controller.ray.actor.dispatch_mode == "dp_compute_proto_with_func"
    assert cfg.controller.ray.actor.detached is True
    assert cfg.controller.ray.actor.num_workers == 4
    assert cfg.controller.ray.actor.mesh_mapping == [0, 0, 1, 1]
    assert cfg.controller.ray.actor.lazy_dispatch_key == "actor_mesh"
    assert cfg.controller.ray.ref.dispatch_mode == "rank_zero"
    assert cfg.controller.ray.fuse_actor_ref is True
    assert cfg.controller.ray.topology_map["actor"] == "actor"


def test_dsv4_smoke_disables_vllm_custom_all_reduce() -> None:
    cfg = AtomRLConfig.from_yaml(DSV4_SMOKE_YAML)

    assert cfg.policy.generation.vllm_cfg.disable_custom_all_reduce is True


def test_dsv4_smoke_selects_fp8_quantization_location() -> None:
    inference_cfg = AtomRLConfig.from_yaml(DSV4_SMOKE_YAML)
    assert inference_cfg.weight_sync.fp8_quantization_location == "inference"
    assert inference_cfg.weight_sync.resolve_fp8_quantize() is False

    trainer_cfg = AtomRLConfig.from_yaml(
        DSV4_SMOKE_YAML,
        overrides=["weight_sync.fp8_quantization_location=trainer"],
    )
    assert trainer_cfg.weight_sync.resolve_fp8_quantize() is True


def test_dsv4_smoke_preserves_sgd_and_uses_miles_grpo_semantics() -> None:
    cfg = AtomRLConfig.from_yaml(DSV4_SMOKE_YAML)

    assert cfg.policy.optimizer_type == "sgd"
    assert cfg.policy.lr_warmup_steps == 0
    assert cfg.policy.lr_decay_style == "constant"
    assert cfg.policy.training.megatron_cfg.use_precision_aware_optimizer is False
    assert cfg.algorithm.loss_agg_mode == "seq-mean-token-mean"
    assert cfg.algorithm.grpo.clip_ratio == 0.2
    assert cfg.algorithm.clip_ratio_high == 0.28
    assert cfg.quantization.rollout_correction.rollout_is is None


def test_dsv4_longrun_uses_miles_adam_and_grpo_semantics() -> None:
    cfg = AtomRLConfig.from_yaml(DSV4_LONGRUN_YAML)

    assert cfg.policy.generation.vllm_cfg.moe_backend == "auto"
    assert cfg.policy.generation.vllm_cfg.linear_backend == "auto"
    assert cfg.policy.generation.vllm_cfg.enable_prefix_caching is False
    assert cfg.policy.generation.vllm_cfg.quantization == "fp8_per_block"
    # The DSV4 fp8_ds_mla attention layout requires an FP8 KV cache. MILES does
    # not override its inference rollout to BF16 KV, so keep the backend contract.
    assert cfg.policy.generation.vllm_cfg.kv_cache_dtype == "fp8_e4m3"
    assert cfg.policy.generation.vllm_cfg.calculate_log_probs is True
    assert cfg.weight_sync.fp8_quantization_location == "inference"
    assert cfg.policy.max_response_length == 4096
    assert cfg.policy.max_total_sequence_length == 16384
    assert cfg.policy.max_token_len_per_gpu == 8192
    assert cfg.policy.train_global_batch_size == 256
    assert cfg.policy.gen_batch_size == 32
    assert cfg.val_steps == 5
    assert cfg.eval.num_generations == 8
    assert cfg.eval.temperature == 0.8
    assert cfg.eval.top_p == 0.7
    assert cfg.eval.top_k == -1
    assert cfg.policy.optimizer_type == "adam"
    assert cfg.quantization.rollout.precision == "fp8"
    assert cfg.quantization.training.fp8 is None
    assert cfg.policy.training.megatron_cfg.tensor_model_parallel_size == 4
    assert cfg.policy.training.megatron_cfg.pipeline_model_parallel_size == 4
    assert cfg.policy.training.megatron_cfg.expert_model_parallel_size == 4
    assert cfg.policy.training.megatron_cfg.grad_reduce_in_fp32 is True
    assert cfg.policy.training.megatron_cfg.optimizer_cpu_offload is True
    assert cfg.policy.training.megatron_cfg.optimizer_offload_fraction == 1.0
    assert cfg.policy.training.megatron_cfg.use_precision_aware_optimizer is True
    assert cfg.policy.training.megatron_cfg.streamed_optimizer_mode == "adam"
    assert cfg.policy.training.megatron_cfg.streamed_optimizer_chunk_size_mib == 256
    assert cfg.policy.training.megatron_cfg.streamed_optimizer_moment_dtype == "bf16"
    assert cfg.policy.learning_rate == 1.0e-6
    assert cfg.policy.weight_decay == 0.1
    assert cfg.policy.adam_beta1 == 0.9
    assert cfg.policy.adam_beta2 == 0.98
    assert cfg.policy.lr_warmup_steps == 0
    assert cfg.policy.lr_decay_style == "constant"
    assert cfg.algorithm.loss_agg_mode == "seq-mean-token-mean"
    assert cfg.algorithm.grpo.clip_ratio == 0.2
    assert cfg.algorithm.clip_ratio_high == 0.28
    assert cfg.quantization.rollout_correction.rollout_is == "token"
    assert cfg.quantization.rollout_correction.rollout_is_batch_normalize is False
    assert cfg.moe.r3.enabled is True
    assert cfg.moe.r3.record_router_logits is False
    assert cfg.checkpointing.save_steps == 25
    assert cfg.checkpointing.save_total_limit == 2
    assert cfg.cluster.num_nodes == 3
    assert cfg.controller.ray.actor.process_on_nodes == [8, 8]
    assert cfg.controller.ray.actor.topology_tags["node_ips"] == ACTOR_NODE_IPS
    assert cfg.controller.ray.rollout.num_workers == 8
    assert cfg.controller.ray.rollout.process_on_nodes == [8]
    assert cfg.controller.ray.rollout.topology_tags["node_ips"] == ROLLOUT_NODE_IPS
    assert cfg.num_training_steps == 200
