# GRPO: training and rollout on separate nodes

GRPO with the trainer and the rollout engine on different machines. After each step the
trainer broadcasts its weights to every rollout replica over RCCL with RoCE GPU Direct
RDMA. The recipes follow LMSYS MILES RL (GRPO 32×8, global batch 256, no KL).

## Examples

| `run.sh` id (alias) | Model | Training | Rollout | Hosts | Status |
|---|---|---|---|---|---|
| `grpo-qwen3moe-vllm-2node` (8) | Qwen3-30B-A3B | Megatron TP=4, EP=8, host 1 | vLLM BF16, TP=2 × 4, host 2 | 2×8 | validated |
| `grpo-qwen3moe-atom-2node` | Qwen3-30B-A3B | Megatron TP=4, EP=8, host 1 | ATOM BF16, TP=2 × 4, host 2 | 2×8 | validated |
| `grpo-dsv4-flash-3node` | DeepSeek-V4-Flash | Megatron TP=4, PP=4, EP=4, hosts 1–2 | vLLM FP8 (per block), TP=8, host 3 | 3×8 | long run |

Each smoke runs 5 full-size steps of its long-run config, with validation and
checkpointing off. The two Qwen3-30B-A3B configs are identical except for the rollout
engine and R3 routing replay, which ATOM does not support.

## Results

Qwen3-30B-A3B on 2×8 MI355X, RoCE between the nodes, 5 steps, release image
`…-261119` (2026-11-19):

| Metric | vLLM rollout (example 8) | ATOM rollout |
|---|---|---|
| Time per step | 196–239 s | 199–234 s |
| Generation per step | 99–104 s | 103–107 s |
| Weight sync, trainer → 4 rollout replicas | 3.9–4.4 s | 4.5–5.0 s |
| `rollout_corr/k3_kl`, steps 1–5 | 0.00074–0.00083 | 0.00153–0.00171 |
| Train accuracy, steps 1–5 | 0.35–0.48 | 0.32–0.49 |
| `k3_kl` at step 1, all runs | 0.000818 / 0.000820 / 0.000815 / 0.000819 / 0.000821 | 0.00168 / 0.00171 |

ATOM's higher `k3_kl` reflects the missing R3 routing replay. Its generation, 145–150 s per
step on the previous image, now matches vLLM's.

## Run

Multi-host runs need a site file naming the hosts, their IPs and the RDMA NIC
([`../site.example.yaml`](../site.example.yaml)); the data root and this checkout must
be visible at the same path on every host.

```bash
bash examples/run.sh prepare 8 --site site.yaml   # Qwen3-30B-A3B and the DAPO data
bash examples/run.sh 8 --site site.yaml
bash examples/run.sh show 8 --site site.yaml      # the commands, per host
```

[docs/07-disaggregated-rdma.md](../docs/07-disaggregated-rdma.md) deploys example 8 by
hand, step by step: images, RDMA checks, Ray and the launch.

## Launcher variables

`run_grpo.sh` (Qwen3-30B-A3B) and `run_grpo_dsv4.sh` (DeepSeek-V4-Flash) take:

| Variable | Meaning |
|---|---|
| `MODE`, `BACKEND` | `smoke` / `longrun` and `vllm` / `atom`; pick the config when `CONFIG_OVERRIDE` is unset |
| `CONFIG_OVERRIDE` | config relative to the AtomRL root |
| `STEPS`, `MODEL_PATH`, `TRAIN_FILE`, `VAL_FILE`, `CKPT_DIR`, `LOG` | run length, model, data, outputs |
| `WEIGHT_SYNC_BACKEND` | `rdma` or `shared_folder` |
| `RAY_ROLE`, `NODE_IP`, `HEAD_IP`, `GPUS_PER_NODE` | `head` / `worker` start a Ray node, `driver` runs the trainer |
| `NCCL_SOCKET_IFNAME`, `NCCL_IB_HCA`, `NCCL_IB_GID_INDEX` | the RDMA NIC; `run.sh` sets them from the site file |

The DeepSeek-V4-Flash long-run config reads the node IPs from `ACTOR_NODE_IPS` and
`ROLLOUT_NODE_IPS`.

## Images

The smokes run in the release image. The manual deployment and the DeepSeek-V4 example
build their own, from the ATOM repository root:

```bash
docker build -f atomrl/examples/GRPO/docker/Dockerfile --target trainer -t qwen3-30b-a3b:trainer .
docker build -f atomrl/examples/GRPO/docker/Dockerfile --target rollout -t qwen3-30b-a3b:rollout .
docker build -f atomrl/examples/GRPO/docker/Dockerfile.dsv4 --target trainer -t dsv4-flash:trainer .
```
