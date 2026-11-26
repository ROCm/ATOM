# Quick start

This guide runs GRPO in BF16 and FP8 regimes, then shows SDDD draft
distillation and multi-node SLURM launch.

## Run GRPO (BF16 baseline)

Use the dense BF16 reference config as a correctness baseline:

```bash
cd ATOM/atomrl
python examples/run_grpo.py --config configs/grpo_dense_bf16.yaml
```

The recipe trains `Qwen/Qwen3-0.6B` with `algorithm.name=grpo`, ATOM rollouts, and FSDP2 policy updates. Checkpointing writes under `checkpointing.checkpoint_dir` from the YAML (defaults to `results/grpo_dense_bf16`).

## Run GRPO (FP8 end-to-end)

Enable FP8 rollout, hybrid FP8 training, and TIS rollout correction:

```bash
python examples/run_grpo.py --config configs/grpo_dense_fp8.yaml
```

See {doc}`/examples/fp8_training` and {doc}`/advance/fp8_quantization` for how the YAML maps to quantizers and correction math.

## YAML configuration sketch

Below is a trimmed skeleton showing the main sections you will edit across recipes:

```yaml
cluster:
  num_nodes: 1
  gpus_per_node: 8

policy:
  model_name: Qwen/Qwen3-0.6B
  training_backend: fsdp2
  generation_backend: atom
  max_total_sequence_length: 2048
  train_global_batch_size: 32
  train_micro_batch_size: 8

algorithm:
  name: grpo
  grpo:
    num_generations: 8
    clip_ratio: 0.2

quantization:
  rollout:
    precision: bf16
  training:
    fp8: null
  rollout_correction:
    enabled: false

num_training_steps: 200
seed: 42
```

Full recipes live under `configs/` in the repository; multi-node launches
override `cluster.num_nodes` and `cluster.gpus_per_node` as needed.

## Run SDDD (Eagle3 Draft Distillation)

Train an Eagle3 draft model for speculative decoding using teacher hidden-state distillation:

```bash
# Kimi K2.5 SDDD smoke test
CUDA_VISIBLE_DEVICES=0 python -m atomrl.trainer.main \
  --config examples/SDDD/Kimi_K25_SDDD_MI350_ATOM/configs/smoke_test_hf.yaml
```

See {doc}`/examples/sddd_training` and {doc}`/advance/sddd` for architecture details and configuration.

## Multi-node SLURM launch

The README pattern parameterizes node count and forwards OmegaConf overrides after the script entrypoint:

```bash
bash scripts/launch_slurm.sh 2 configs/grpo_dense_fp8.yaml \
    policy.model_name=Qwen/Qwen3-8B \
    logger.wandb_enabled=true
```

Adapt `scripts/ray.sub` to your site’s partition, account, and module loads. Ray head/worker startup should match `cluster.ray_address` when using an external cluster; see {doc}`/advance/distributed`.

## Where to go next

- RL walkthroughs: {doc}`/examples/grpo_training`, {doc}`/examples/fp8_training`
- SDDD walkthroughs: {doc}`/examples/sddd_training`, {doc}`/advance/sddd`
- Systems topics: {doc}`/advance/distributed`, {doc}`/advance/algorithms`, {doc}`/architecture`
- Python APIs: {doc}`/api/config`, {doc}`/api/protocol`, {doc}`/api/algorithms`
