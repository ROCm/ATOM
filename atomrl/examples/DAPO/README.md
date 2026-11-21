# DAPO: math RL on Qwen3-8B and Qwen3-30B-A3B

DAPO on DAPO-Math-17k, validated on AIME-2024, across two models, two training backends,
two rollout engines, BF16 and FP8, and synchronous and fully async scheduling. The
synchronous examples run 8 training actors and 8 colocated rollout replicas (TP=1) on
one node; all share one entrypoint (`atomrl.trainer.main`) and one launcher
(`run_dapo.sh`) and differ only in configuration.

Algorithm: clip-higher, dual-clip and token-mean policy loss; GRPO advantages with
per-uid group normalization; dynamic sampling (`filter_groups`); overlong reward
buffer; TIS rollout correction.

## Examples

| Alias | `run.sh` id | Model | Training | Rollout |
|---|---|---|---|---|
| 1 | `dapo-qwen3-8b-vllm-bf16` | Qwen3-8B-Base | FSDP2 BF16 | vLLM BF16 |
| 2 | `dapo-qwen3-8b-vllm-fp8-rollout` | Qwen3-8B-Base | FSDP2 BF16 | vLLM FP8 (per block) |
| 3 | `dapo-qwen3-8b-vllm-fp8-e2e` | Qwen3-8B-Base | FSDP2 FP8 (blockwise 2D) | vLLM FP8 (per block) |
| 4 | `dapo-qwen3-8b-atom-fp8` | Qwen3-8B-Base | FSDP2 FP8 (blockwise 2D) | ATOM FP8 (per block) |
| 5 | `dapo-qwen3-8b-atom-bf16` | Qwen3-8B-Base | FSDP2 BF16 | ATOM BF16 |
| 6 | `dapo-qwen3moe-vllm-bf16` | Qwen3-30B-A3B-Base | FSDP2 BF16 | vLLM BF16 |
| 7 | `dapo-qwen3moe-megatron-ep8` | Qwen3-30B-A3B-Base | Megatron EP=8, DP=8 | vLLM BF16 |
| 9 | `dapo-qwen3moe-atom-bf16` | Qwen3-30B-A3B-Base | FSDP2 BF16 | ATOM BF16 |
| | `dapo-qwen3moe-vllm-fully-async` | Qwen3-30B-A3B-Base | FSDP2 BF16, 4 GPUs | vLLM BF16, 4 GPUs, fully async |
| | `dapo-qwen3moe-fully-async-vllm-2node` | Qwen3-30B-A3B-Base | FSDP2 BF16, node 1 | vLLM BF16, node 2, RDMA sync |
| | `dapo-qwen3moe-fully-async-atom-2node` | Qwen3-30B-A3B-Base | FSDP2 BF16, node 1 | ATOM BF16, node 2, RDMA sync |

Examples 2 and 3 differ only in training precision; 5 is the BF16 control for 4; 7 is 6
on Megatron; 9 is 6 with ATOM rollout, including MoE expert weight sync.

## Results

### Reproducibility

Train/rollout alignment (`rollout_corr/k3_kl`) at step 1 of each smoke, 8× MI355X,
seed 10086. The reference is the mean of 3–5 runs; spread is the worst deviation of
any of those runs from the mean.

| # | Reference | Runs | Spread | Tolerance | 2026-11-17, release image | Δ |
|---|---|---|---|---|---|---|
| 1 | 0.00110 | 3 | 2.6% | ±30% | 0.000950 | −13.7% |
| 2 | 0.00481 | 3 | 2.0% | ±30% | 0.00469 | −2.5% |
| 3 | 0.00410 | 3 | 17.2% | ±30% | 0.00351 | −14.3% |
| 4 | 0.00399 | 5 | 32.1% | ±50% | 0.00253 | −36.5% |
| 5 | 0.000954 | 3 | 11.8% | ±50% | 0.00102 | +7.3% |
| 6 | 0.00144 | 3 | 9.2% | ±50% | 0.00147 | +2.3% |
| 7 | 0.00161 | 3 | 3.2% | ±50% | 0.00162 | +0.8% |
| 9 | 0.00153 | 3 | 4.7% | ±50% | 0.00169 | +10.3% |

All eight pass, with no runtime errors and every weight-sync bucket complete.

### Rollout engine and training backend

Qwen3-30B-A3B, same training config, 4k responses; `k3_kl` is the reference above:

| | Example | Time per step | `k3_kl` |
|---|---|---|---|
| FSDP2 + vLLM | 6 | 106.7 s | 0.00144 |
| Megatron + vLLM | 7 | 103.0 s | 0.00161 |
| FSDP2 + ATOM | 9 | **56.7 s** (1.9×) | 0.00153 |

### Fully async

Long runs on Qwen3-30B-A3B-Base, 20k-token responses, identical hyperparameters
([full settings](../../atomrl-docs/docs/source/advance/async_training.md#results)):

| | Synchronous, 8 GPUs | Fully async, 4 + 4 GPUs |
|---|---|---|
| Time to step 60 | 555 min | **204 min (2.7×)** |
| Train reward, steps 1–60 | 0.426 | 0.421 |
| AIME24, steps 10–60 | 0.225 | 0.211 |

| | Fully async, 4 + 4, one node | Fully async, 8 + 8, two nodes (vLLM) | Fully async, 8 + 8, two nodes (ATOM) |
|---|---|---|---|
| Time to step 30 | 108 min | 47 min (2.3×) | **41 min (2.6×)** |
| Train reward, steps 21–30 | 0.474 | 0.467 | 0.470 |
| AIME24 at step 30 | 0.228 | 0.225 | 0.212 |

## Run

```bash
export DATA_ROOT=/path/to/data
bash examples/run.sh prepare 1 6        # models, datasets, prompt filtering (once)
bash examples/run.sh 4                  # one example, judged against its reference
bash examples/run.sh suite --tier t1    # the whole table
bash examples/run.sh 4 --full           # the long run behind the smoke
```

Requirements: 8 AMD MI350X / MI355X GPUs, Docker, and about 134 GB of disk for the
image, models and data.

## Without the launcher

`run_dapo.sh` runs in any environment with AtomRL, Lumen, AITER, vLLM and (for the ATOM
modes) ATOM; [§8.4.5](../docs/08-release.md#845-the-manual-command-without-the-launcher)
gives the full container commands, and `bash examples/run.sh show <N>` prints them.

```bash
export RL_ROOT=/path/to/rl_root DATA_ROOT=/path/to/data
CONFIG_OVERRIDE=examples/DAPO/configs/dapo_qwen3_8b_ray_vllm_smoke.yaml \
MODE=bf16 STEPS=3 LOG="$DATA_ROOT/logs/dapo-smoke.log" \
bash "$RL_ROOT/ATOM/atomrl/examples/DAPO/run_dapo.sh"
```

| Variable | Meaning |
|---|---|
| `MODE` | `bf16`, `fp8` (vLLM FP8 rollout), `atomfp8`, `atombf16` |
| `TRAIN_FP8` | `1` trains in FP8 (with `fp8` / `atomfp8`) |
| `CONFIG_OVERRIDE` | config relative to `$RL_ROOT/ATOM/atomrl`; without it `MODE` picks the long-run config |
| `STEPS`, `MODEL_PATH`, `TRAIN_FILE`, `VAL_FILE`, `LOG` | run length, model, data, trainer log |
| `EXTRA_OVERRIDE` | extra config overrides, space separated |
| `ENABLE_MORI` | `true` routes Megatron MoE dispatch through MORI |
| `RAY_ROLE`, `NODE_IP`, `HEAD_IP` | multi-node: `head` / `worker` start a Ray node, `driver` runs the trainer |

## Configs

| Config (`configs/`) | Used by |
|---|---|
| `dapo_qwen3_8b_ray_vllm_{smoke,longrun}.yaml` | 1 |
| `dapo_qwen3_8b_ray_vllm_fp8_{smoke,4k_smoke,longrun}.yaml` | 2, 3 |
| `dapo_qwen3_8b_ray_atom_fp8_{4k_smoke,longrun,debug}.yaml` | 4 |
| `dapo_qwen3_8b_ray_atom_bf16_{4k_smoke,longrun}.yaml` | 5 |
| `dapo_qwen3_8b_ray_megatron_{smoke,longrun}.yaml` | Qwen3-8B on Megatron |
| `dapo_qwen3moe_a3b_ray_vllm_verlref_{4k_smoke,longrun}.yaml` | 6 |
| `dapo_qwen3moe_a3b_ray_megatron_verlref_{4k_smoke,4k_longrun,longrun}.yaml` | 7 |
| `dapo_qwen3moe_a3b_ray_megatron_smoke.yaml` | Qwen3-30B-A3B on Megatron with MORI dispatch |
| `dapo_qwen3moe_a3b_ray_atom_bf16_{4k_smoke,longrun}.yaml` | 9, two-node fully async (ATOM) |
| `dapo_qwen3moe_a3b_ray_vllm_verlref_fully_async_longrun.yaml` | fully async (one and two nodes, vLLM) |
