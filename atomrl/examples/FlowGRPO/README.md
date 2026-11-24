# Qwen-Image FlowGRPO full-parameter RL on 8× MI350X

> 中文版：[README_cn.md](README_cn.md)

Reproduces verl-omni's official recipe
[`run_qwen_image_ocr_veomni.sh`](https://github.com/verl-project/verl-omni/blob/9ab544d81f93cbf49754aabacfa67f6fcf01b505/examples/flowgrpo_trainer/qwen_image/run_qwen_image_ocr_veomni.sh)
on one node with 8× AMD Instinct MI350X: full-parameter FlowGRPO on Qwen-Image (20B DiT),
VeOmni training engine, vLLM-Omni rollout, Qwen3-VL-8B as the OCR reward, 300 steps.

- **Baseline**: the official recipe, with only paths, GPU count and ROCm attention backends changed (§4).
- **Lumen**: the baseline plus two [Lumen](../../Lumen) patches on the actor (§5).

| | Official (4× H200) | Baseline (8× MI350X) | Lumen (8× MI350X) |
|---|---:|---:|---:|
| Best validation reward | 0.975 | 0.960 | **0.978** |
| Throughput (images/GPU/s) | 0.517 | 0.496 | **0.523** |
| Time per step | 247.7 s | 129.1 s | **122.4 s** |

## 1. Comparison with the official results

Official reference: verl-omni's public W&B run
[`andyzhou/VeRL-Omni-demo/8p8y9olb`](https://wandb.ai/andyzhou/VeRL-Omni-demo/runs/8p8y9olb) (same algorithm and
hyper-parameters, 4× H200, FSDP engine, ends at step 277) and
[`docs/algo/performance.md`](https://github.com/verl-project/verl-omni/blob/9ab544d81f93cbf49754aabacfa67f6fcf01b505/docs/algo/performance.md).
Ours: 8× MI350X, VeOmni engine, same seed for both versions.

### Correctness

Validation reward (OCR test split, 1018 prompts):

| step | official | baseline | Lumen |
|---:|---:|---:|---:|
| 0 | | 0.402 | 0.403 |
| 30 | 0.689 | 0.644 | 0.618 |
| 60 | 0.851 | 0.815 | 0.822 |
| 90 | 0.932 | 0.901 | 0.919 |
| 120 | 0.964 | 0.928 | 0.937 |
| 180 | 0.975 | 0.934 | 0.964 |
| 240 | 0.950 | 0.939 | 0.968 |
| 270 | 0.924 | 0.960 | 0.975 |
| 300 | | 0.949 | **0.978** |

Train reward (mean over the window):

| steps | official | baseline | Lumen |
|---|---:|---:|---:|
| 1–60 | 0.407 | 0.359 | 0.354 |
| 61–120 | 0.850 | 0.806 | 0.814 |
| 121–180 | 0.958 | 0.919 | 0.922 |
| 181–240 | 0.950 | 0.883 | 0.952 |
| 241–277 | 0.924 | 0.926 | 0.954 |

- All three reach the same plateau (validation 0.95–0.98). The official run rises faster over the first 120 steps
  and then declines; both of ours keep rising to step 300, and the Lumen version ends highest.
- Training health matches the official run (30-step means): clip fraction 0.11–0.31 (official 0.09–0.24), grad
  norm 0.005–0.018 (official 0.006–0.019), no non-finite gradients in any of the three.
- Separate runs of one configuration vary noticeably; compare on the same machine with the same seed.

### Performance

| run | GPUs | engine | time per step | throughput (images/GPU/s) |
|---|---|---|---:|---:|
| official, W&B run | 4× H200 | FSDP | 247.7 s | 0.517 |
| official, docs | 64× H100 | VeOmni | 100 s | 0.079 |
| baseline | 8× MI350X | VeOmni | 129.1 s | 0.496 |
| **Lumen** | 8× MI350X | VeOmni | **122.4 s** | **0.523** |

Medians over all training steps, validation and checkpoint steps excluded. Every step trains on 32 prompts × 16
images, so images/GPU/s is the comparable number across GPU counts.

| Lumen vs baseline | baseline | Lumen |
|---|---:|---:|
| actor (`old_log_prob` + `update_actor`) | 31.5 s | 29.1 s (−8%) |
| whole step | 129.1 s | 122.4 s (−5%) |
| 300 steps end to end (startup, validation and checkpoints included) | about 12.5 h | about 12.0 h |

Lumen accelerates only the actor, about 24% of a step: 2.4 s of the 6.7 s saved per step. The rest comes from
weight sync and generation, which the patches do not touch, and is run-to-run variation.

## 2. Requirements

- 8× MI350X (gfx950, 288 GB); peak memory about 208 GB per GPU.
- Host memory ≥ 1.5 TB: the 8 sleeping rollout replicas keep their pipelines in host memory (~900 GB).
- ≥ 1 TB free under `$WORK`: models 71 GB; each checkpoint 228 GB, the last 2 are kept.
- Docker; a host ROCm driver compatible with ROCm 7.2; access to GitHub, Hugging Face, Docker Hub and PyPI.

## 3. Reproduce

`bash examples/run.sh flowgrpo-qwen-image-baseline` (or `flowgrpo-qwen-image-lumen`) runs the
steps below end to end with 90 steps. By hand:

```bash
export WORK=/path/to/work                       # code, models, data and run outputs
git clone -b main https://github.com/ROCm/ATOM.git $WORK/src/ATOM
EX=$WORK/src/ATOM/atomrl/examples/FlowGRPO

docker pull vllm/vllm-openai-rocm:v0.27.0
WORK=$WORK bash $EX/launch_container.sh         # container qwen-image-rl
docker exec qwen-image-rl bash $EX/setup_env.sh # ~30 min; ends with 19653 / 1018 rows and SETUP OK

# Training: each run uses all 8 GPUs; log in $WORK/runs/<RUN_NAME>/train.log
docker exec -d qwen-image-rl bash -c "RUN_NAME=base bash $EX/run_flowgrpo.sh"           # baseline
docker exec -d qwen-image-rl bash -c "RUN_NAME=lumen LUMEN=1 bash $EX/run_flowgrpo.sh"  # Lumen
docker exec qwen-image-rl python3 $EX/compare.py $WORK/runs/base $WORK/runs/lumen      # vs official and each other
```

- Quick comparison: append `trainer.total_training_steps=90` (validation at steps 30/60/90), about 3.5 h per run.
- W&B: add `-e WANDB_API_KEY=<key> -e WANDB_PROJECT=<project>` to `docker exec`; without a key the run logs offline.

## 4. Differences from the official recipe

No algorithm or hyper-parameter changes; only:

| item | official | here | why |
|---|---|---|---|
| GPUs | script default 4 | 8 (`NUM_GPUS`) | one 8-GPU node |
| actor param / optimizer offload | on | off | fits in 288 GB; offload exhausts host memory and is slow |
| attention backends | default | actor `native` (torch SDPA); rollout `TORCH_SDPA`, `enforce_eager=True` | ROCm |
| validation before training | off | on | a step-0 reference point |
| seeds | default | data and rollout fixed to 42 | same-seed comparisons |

External dependencies are pinned in `setup_env.sh`: the image, the commits of verl, verl-omni, VeOmni, vllm-omni
and flow_grpo, and the revisions of both models. Lumen comes from this checkout's in-tree `atomrl/lumen`.

## 5. Lumen patches (`LUMEN=1`)

Installed only in the actor processes that build the VeOmni training engine; the rollout and the reward model are
untouched. The patch code is in Lumen's [`examples/perf-veomni/veomni_patches`](../../Lumen/examples/perf-veomni/veomni_patches).

| `LUMEN_PATCH` | what | numerics |
|---|---|---|
| `rmsnorm_fuse` | diffusers RMSNorm forward fused by `torch.compile` | 55.6 dB vs FP32 (original 52.6 dB) |
| `local_adamw` | fused AdamW on the FSDP2 local shards | bit-identical |

A third patch, `mask_all_true_to_none`, saves about 3% more actor time but slows the train reward over steps
30–150, so it is off by default. The other perf-veomni optimizations do not apply to this RL setup (no VAE in the
actor, memory needed by the rollout, or non-finite gradients from the fused attention backward).
