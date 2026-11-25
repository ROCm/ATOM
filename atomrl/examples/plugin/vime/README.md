# vime + AtomRL plugin: Qwen3-8B on Megatron

Qwen3-8B RL training on vime with AtomRL's Megatron bridge and optional Lumen FP8
training, injected through vime's `--custom-*-path` hooks without modifying vime.

## Hooks

`atomrl.plugin.vime.hooks` provides:

| vime argument | Hook | Effect |
|---|---|---|
| `--custom-megatron-init-path` | `custom_megatron_init` | creates the Lumen `FP8TrainingManager` when `TRAIN_FP8=1` |
| `--custom-model-provider-path` | `custom_model_provider` | builds the Megatron GPTModel with AtomRL's transformer config and applies FP8 |
| `--custom-generate-function-path` | `generate_rollout` | optional: ATOM rollout instead of vime's vLLM |
| `--custom-megatron-before-train-step-hook-path` | `custom_megatron_before_train_step_hook` | resets Lumen FP8 state |
| `--custom-megatron-before-log-prob-hook-path` | `custom_megatron_before_log_prob_hook` | resets Lumen FP8 state |
| `--custom-update-weight-post-write-path` | `custom_update_weight_post_write` | rank-zero ATOM checkpoint reload |

vime still decides when to write weights. The post-write hook reloads the
resulting HF checkpoint through the public ATOM manager.

## Requirements

- 8× AMD MI300X / MI325X / MI355X
- Megatron-LM, ROCm fork (`rocm_dev` branch)
- `$DATA_ROOT/models/Qwen3-8B-Base` and `$DATA_ROOT/data_cached/dapo-math-17k.filtered.parquet`

## Run

```bash
# Build from the ATOM repository root
docker build -f atomrl/examples/plugin/vime/Dockerfile \
  --build-arg VIME_REPO=<vime repository URL> --build-arg VIME_REF=<revision> \
  -t atomrl-vime-plugin .

# 3-step smoke; add -e TRAIN_FP8=1 for FP8 training, -e STEPS=<n> for longer runs
docker run --rm -it --device /dev/kfd --device /dev/dri --group-add video --group-add render \
  --shm-size 64g -v /data:/data -e DATA_ROOT=/data -e VIME_ROOT=/workspace/vime \
  atomrl-vime-plugin
```

Without Docker:

```bash
cd /path/to/Megatron-LM && pip install -e .
cd /path/to/vime && pip install -e .
pip install "ray[default]>=2.9" datasets "accelerate>=0.28" \
  "omegaconf>=2.3" safetensors peft "megatron-core>=0.9" "megatron-bridge>=0.1"
cd /path/to/ATOM && LUMEN_BUILD_EXTENSIONS=1 pip install --no-build-isolation --no-deps -e .
DATA_ROOT=/data VIME_ROOT=/path/to/vime bash examples/plugin/vime/run.sh
```

A successful smoke prints `AtomRL vime hooks: importable and signature-verified` and
Megatron training logs with finite losses.
