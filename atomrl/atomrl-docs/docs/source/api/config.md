# Configuration API

The `atomrl.core.config` module defines structured dataclasses for YAML-driven training. The top-level object is `AtomRLConfig`, loaded through OmegaConf so nested dictionaries merge cleanly into defaults.

## `AtomRLConfig`

| Field | Type | Description |
| --- | --- | --- |
| `cluster` | `ClusterConfig` | Node / GPU layout and optional Ray address |
| `policy` | `PolicyConfig` | Model id, backends, batching, sequence cap |
| `algorithm` | `AlgorithmConfig` | Algorithm name and nested hyperparameters |
| `reward` | `RewardConfig` | Reward channel selection (`function`, dataset, optional judge model) |
| `quantization` | `QuantizationConfig` | Rollout, training, and correction numerics |
| `moe` | `MoEConfig` | MoE R3 knobs |
| `checkpointing` | `CheckpointConfig` | Output directory, cadence, rotation |
| `logger` | `LoggerConfig` | W&B toggles and log density |
| `num_training_steps` | `int` | Total optimizer-bearing steps for the driver loop |
| `seed` | `int` | RNG seed surfaced to workers |

### Loading helpers

```python
from pathlib import Path
from atomrl.core.config import AtomRLConfig, load_config

cfg = AtomRLConfig.from_yaml("configs/grpo_dense_bf16.yaml")
cfg_over = AtomRLConfig.from_yaml(
    Path("configs/grpo_dense_fp8.yaml"),
    overrides=["policy.train_global_batch_size=16"],
)

same = load_config("configs/grpo_dense_bf16.yaml", overrides=["seed=123"])
```

`from_cli()` parses `--config path.yaml` and forwards unknown argv entries as OmegaConf dotlist overrides, matching `examples/run_grpo.py` style CLIs.

## Nested dataclasses

### `ClusterConfig`

| Field | Default | Description |
| --- | --- | --- |
| `num_nodes` | `1` | Logical node count for scheduling |
| `gpus_per_node` | `1` | GPUs reserved per node |
| `ray_address` | `None` | Optional `ray://` or `auto` head address |

### `PolicyConfig`

| Field | Default | Description |
| --- | --- | --- |
| `model_name` | `""` | Hugging Face hub id or local path |
| `training_backend` | `"fsdp2"` | `TrainingBackend` string (`fsdp2`, `megatron_native`) |
| `generation_backend` | `"atom"` | `GenerationBackend` string (`atom`, `vllm`) |
| `training` | `TrainingConfig` | Nested Megatron / FSDP knobs |
| `generation` | `GenerationConfig` | Nested ATOM generation knobs |
| `max_total_sequence_length` | `4096` | Token cap for prompts + responses |
| `train_global_batch_size` | `64` | Global batch for actor updates |
| `train_micro_batch_size` | `8` | Per-step microbatch for gradient accumulation |

`TrainingConfig` holds `megatron_cfg: MegatronConfig` and `fsdp_cfg: Optional[dict]` for backend-specific blobs.

`GenerationConfig` holds `atom_cfg: AtomConfig` (TP width, KV dtype, optional `max_model_len`).

### `MegatronConfig`

| Field | Default | Description |
| --- | --- | --- |
| `tensor_model_parallel_size` | `1` | TP degree |
| `pipeline_model_parallel_size` | `1` | PP degree |
| `context_parallel_size` | `1` | CP degree |
| `expert_model_parallel_size` | `1` | EP degree |
| `sequence_parallel` | `False` | Sequence parallelism toggle |
| `num_experts` | `None` | Expert count hint for MoE recipes |
| `moe_grouped_gemm` | `False` | Prefer grouped GEMM kernels |
| `use_distributed_optimizer` | `True` | Shard FP32 master weights and optimizer state |
| `log_probs_chunk_size` | `0` | Chunk size for memory-efficient token log-prob |
| `enable_dynamic_batch` | `False` | Pack variable-length sequences into TE forwards |

### `AtomConfig`

| Field | Default | Description |
| --- | --- | --- |
| `tensor_parallel_size` | `1` | Inference tensor parallel width |
| `kv_cache_dtype` | `"auto"` | KV storage dtype string consumed by ATOM |
| `max_model_len` | `None` | Optional cap forwarded to inference engine |

### `AlgorithmConfig`

| Field | Default | Description |
| --- | --- | --- |
| `name` | `"grpo"` | `AlgorithmName` value (`grpo`, `dapo`, `ppo`) |
| `grpo` | `GRPOConfig` | Group size, KL, clip, PPO-style epoch settings |
| `dapo` | `DAPOConfig` | Asymmetric clip + DAPO toggles |
| `ppo` | `PPOConfig` | GAE, discount, PPO epoch settings |

`GRPOConfig` fields: `num_generations`, `kl_coeff`, `clip_ratio`, `num_ppo_epochs`, `num_mini_batches`.

`DAPOConfig` fields: `num_generations`, `kl_coeff`, `clip_ratio_low`, `clip_ratio_high`, `dynamic_sampling`, `token_level_pg`, `overlong_reward_shaping`.

`PPOConfig` fields: `kl_coeff`, `clip_ratio`, `num_ppo_epochs`, `num_mini_batches`, `gae_lambda`, `discount`.

### `QuantizationConfig`

| Field | Type | Description |
| --- | --- | --- |
| `rollout` | `RolloutQuantConfig` | ATOM rollout dtype + layer boundaries |
| `training` | `TrainingQuantConfig` | Lumen FP8 training recipe |
| `rollout_correction` | `RolloutCorrectionConfig` | TIS/MIS advantage correction |

`RolloutQuantConfig` includes `precision`, `use_deep_gemm`, `num_first_layers_in_bf16`, `num_last_layers_in_bf16`.

`TrainingQuantConfig` includes `fp8`, `fp8_recipe`, `fp8_weight_cache`.

`RolloutCorrectionConfig` includes `enabled`, `method` (`tis` / `mis`), `clip`.

### `MoEConfig` / `R3Config`

| `R3Config` field | Default | Description |
| --- | --- | --- |
| `enabled` | `False` | Master R3 toggle |
| `record_router_logits` | `True` | Capture logits during rollout |
| `replay_mode` | `"distribution"` | `distribution` or `hard_assignment` |

### `RewardConfig`, `CheckpointConfig`, `LoggerConfig`, `WandbConfig`

- `RewardConfig`: `type`, `function`, `dataset`, optional `model_name` for learned judges.
- `CheckpointConfig`: `checkpoint_dir`, `save_steps`, `save_total_limit`.
- `LoggerConfig`: `wandb_enabled`, nested `wandb` (`project`, `name`, `entity`), `log_interval`, `num_val_samples_to_print`.

### Runtime assembly

With `assembly.strict_policy: true`, the trainer validates the backends at
startup: the training backend must be `fsdp`, `fsdp2` or `megatron`, and the
inference backend `atom`. `assembly.use_new_assembler: true` builds the trainer
through the runtime assembler.

## YAML loading semantics

OmegaConf merges the structured schema with the YAML file, so missing keys keep dataclass defaults and type validation stays strict. Lists and dictionaries inherit OmegaConf’s dot-access rules inside overrides.

## CLI overrides

Any trailing `key=value` arguments passed to `AtomRLConfig.from_yaml(..., overrides=[...])` use OmegaConf dotlist syntax:

```bash
python examples/run_grpo.py --config configs/grpo_dense_bf16.yaml \
  cluster.gpus_per_node=8 \
  algorithm.name=dapo
```

```{note}
Dataclasses are the source of truth; if a YAML key is misspelled, OmegaConf raises during merge rather than silently falling back.
```

See also: {doc}`/api/protocol`, {doc}`/api/quantization`, {doc}`/api/moe`.
