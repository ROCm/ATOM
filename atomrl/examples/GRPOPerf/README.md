# GRPOPerf: NeMo-RL v0.8's GRPO performance cases on MI355X

The eight cases of [NeMo-RL v0.8's performance summary](https://docs.nvidia.com/nemo/rl/latest/about/performance-summary.html)
run with AtomRL (Megatron native training, ATOM rollout) on 2 x 8 MI355X, with the
metrics recomputed by NeMo-RL's formula. [中文](README_cn.md)

## Examples

| Alias | `run.sh` id | Model | Policy | Training | Rollout | Hosts |
|---|---|---|---|---|---|---|
| B1 | `grpoperf-qwen3moe-onpolicy-2node` | Qwen3-30B-A3B | on-policy | Megatron EP8 x DP2 | ATOM BF16 TP1 x 16, colocated | 2x8 |
| B2 | `grpoperf-qwen3moe-1stepoff-2node` | Qwen3-30B-A3B | 1-step off | Megatron EP8, host 1 | ATOM BF16 TP1 x 8, host 2 | 2x8 |
| B3 | `grpoperf-qwen3-8b-1stepoff` | Qwen3-8B | 1-step off | Megatron TP1 x DP4, 4 GPUs | ATOM BF16 TP1 x 4, 4 GPUs | 1x8 |
| B4 | `grpoperf-qwen3-8b-onpolicy` | Qwen3-8B | on-policy | Megatron TP1 x DP8 | ATOM BF16 TP1 x 8, colocated | 1x8 |
| F1 | `grpoperf-qwen3moe-onpolicy-fp8gen-2node` | Qwen3-30B-A3B | on-policy | as B1 | ATOM `ptpc_fp8` | 2x8 |
| F2 | `grpoperf-qwen3moe-1stepoff-fp8gen-2node` | Qwen3-30B-A3B | 1-step off | as B2 | ATOM `ptpc_fp8` | 2x8 |
| F3 | `grpoperf-qwen3-8b-1stepoff-fp8gen` | Qwen3-8B | 1-step off | as B3 | ATOM `per_block_fp8` | 1x8 |
| F4 | `grpoperf-qwen3-8b-onpolicy-fp8gen` | Qwen3-8B | on-policy | as B4 | ATOM `per_block_fp8` | 1x8 |

Every case runs NeMo-RL's workload and GRPO for 10 steps: 64 prompts x 32 generations =
2048 samples per step, prompt + response <= 4096, OpenMathInstruct-2, one optimizer step
per batch, Adam lr 3e-7 with 50 warmup steps from 3e-8, KL 0.01 (k3) to the initial
model, leave-one-out advantages and the Math-Verify reward; for the MoE the router stays
frozen, in fp64, and the permutes are fused (NeMo-RL's defaults). Training is BF16; FP8 rows quantize generation only (attention, router
and `lm_head` stay BF16). F1 and F2 need ATOM's FP8 expert requantization in a weight
sync. The check holds `perf/throughput` at step 8 within 10% of the case's default
configuration (`examples.yaml`).

Each case follows the NeMo-RL recipe in the table on two settings that change a step's
work: whether the old log-probs come from the training forward
(`force_on_policy_ratio`) and whether the loss carries importance-sampling correction
against the rollout. NeMo-RL has no Qwen3-8B rows; B3, B4, F3 and F4 follow its Qwen3-32B
recipes.

| # | NeMo-RL recipe | Old log-probs | IS correction | Forwards besides training |
|---|---|---|---|---|
| B1 | `grpo-qwen3-30ba3b-4n4g` | training forward | - | reference |
| B2 | `grpo-qwen3-30ba3b-4n4g-async-1off` | training forward | token | reference |
| F1 | `grpo-qwen3-30ba3b-4n4g-mxfp8-rollout` | own forward | token | old, reference |
| F2 | `grpo-qwen3-30ba3b-4n4g-async-1off-mxfp8-rollout` | own forward | token | old, reference |
| B3 | `grpo-qwen3-32b-8n4g-async-1off` | training forward | token | reference |
| B4 | `grpo-qwen3-32b-4n4g` | training forward | - | reference |
| F3 | `grpo-qwen3-32b-8n4g-async-1off-mxfp8-rollout` | training forward | token | reference |
| F4 | `grpo-qwen3-32b-4n4g-mxfp8-rollout` | training forward | token | reference |

The reference forward runs the actors' own model with the initial weights swapped in
(`controller.ray.ref_in_actor`), so it costs what a policy forward costs.

## Results

Steps 3-8 on `rocm/atom-dev:atomrl-dapo-gfx950-rocm7.2.4-261125`, each case in its fastest
parallel configuration (searched with the `atomrl-strategy-search` skill in
`atomrl/.claude/skills/`), one counted run each; every counted run passes NeMo-RL's
correctness gate. Runs are in [measured/runs.csv](measured/runs.csv). Each row does NeMo-RL's
work per step: generation, the forwards in the table above, and training.

| # | GPUs | Configuration | Gen [TP,PP] | Train [TP,CP,EP,PP] | G-Avg seq len | Tokens/s/GPU | Step (s) | NeMo-RL GB200 | Ratio |
|---|---|---|---|---|---|---|---|---|---|
| B1 | 16 | colocated | [1,1] | [1,1,8,1] | 3,164 | 2,543 | 165.0 | 2,656, 159 s | 0.96 |
| B2 | 12+4 | `B2-12P4`: EP4 x DP3, four ATOM replicas | [1,1] | [1,1,4,1] | 3,169 | 2,800 | 150.1 | 2,031, 208 s | 1.38 |
| B3 | 4+4 | as in Examples | [1,1] | [1,1,1,1] | 3,251 | 6,189 | 139.2 | - | |
| B4 | 8 | as in Examples | [1,1] | [1,1,1,1] | 3,251 | 5,412 | 159.3 | - | |
| F1 | 16 | colocated | [1,1] | [1,1,8,1] | 3,163 | 2,335 | 179.6 | - | |
| F2 | 12+4 | `F2-12P4`: EP4 x DP3, four ATOM replicas | [1,1] | [1,1,4,1] | 3,165 | 2,296 | 182.8 | 1,819, 232 s (MXFP8) | 1.26 |
| F3 | 4+4 | as in Examples | [1,1] | [1,1,1,1] | 3,248 | 6,155 | 139.9 | - | |
| F4 | 8 | as in Examples | [1,1] | [1,1,1,1] | 3,247 | 5,452 | 157.9 | - | |

- B1's step is 71.1 s of generation, 22.9 s of reference forward, 64.9 s of training and
  5.8 s of weight sync; F1's 61.0 s of generation, 23.3 s of old-log-prob and 23.0 s of
  reference forward, 65.6 s of training and 6.5 s of weight sync. F2's trainers are the
  critical side: 32.7 s of old-log-prob and 32.2 s of reference forward, 110.8 s of
  training and 6.9 s of weight sync, while generation (114.1 s) overlaps them. B2's
  trainers are the critical side too: 32.4 s of reference forward, 111.2 s of training and
  6.3 s of weight sync, with generation (128.4 s) overlapping them.
- B3's two sides are about even: 133.9 s of generation against 31.4 s of reference forward,
  101.2 s of training and 2.4 s of weight sync. Its counted run is steps 1-10 of a 200-step
  run of the same configuration (the learning rate is constant after warmup, and nothing
  validates or checkpoints), so those steps are the 10-step run. F3, B3 with FP8
  generation, has the same balance: 134.6 s of generation against 31.5 s of reference
  forward, 100.8 s of training and 2.5 s of weight sync, so FP8 generation does not
  shorten its step.
- F4's step is 86.1 s of generation, 17.3 s of reference forward, 51.0 s of training and
  3.3 s of weight sync. FP8 generation saves Qwen3-8B about a second against B4 (87.4 s).
- NeMo-RL has no Qwen3-8B rows (its rows are Qwen3-32B).
- NeMo-RL's fp64 router costs Qwen3-30B-A3B about 11% of its training time here: B1's
  layout on one node (`B1-1N`, steps 2-4) trains in 106.4 s a step with it and 95.2 s with
  an fp32 router, and its reference forward takes 40 s against 36 s. The KL term adds about
  2% (104 s without it); freezing the router changes nothing measurable. Both stay as
  NeMo-RL has them.

## Long runs

B4 and B3 for 200 steps on this code (`B4-200`, `B3-200`; B3's table row is steps 1-10 of
its run). Windows of 20 steps are in [measured/longrun.csv](measured/longrun.csv).

| 200 steps | B4 (on-policy, 9.1 h) | B3 (1-step off, 7.9 h) |
|---|---|---|
| `token_mult_prob_error`, median of every 20 steps | 1.013 | 1.013 |
| Train-generation KL (`rollout_corr/kl`) | 7.7e-4 throughout | 7.6e-4 throughout |
| `ref_kl`, steps 1-20 / 181-200 | 7.3e-4 / 1.21e-3 | 7.3e-4 / 1.26e-3 |
| Entropy | 0.25 | 0.25 |
| Response length, steps 1-20 / 181-200 (none truncated) | 3,311 / 3,207 | 3,312 / 3,203 |
| Peak reserved memory | 222 GB, flat | 237 GB, flat |
| Steps above 1.1 (single tokens) / last step | 8 of 200 / 1.013 | 9 of 200 / 1.013 |

- The train-generation KL never moves, so every one of the 200 weight syncs lands.
- `ref_kl` sits at about 7.5e-4 until step 80 and then rises smoothly. That floor is
  numerical: with weights a few ulps apart the reference and training forwards stop
  agreeing bit for bit, and BF16 rounding differences compound to about 7.5e-4 (a run with
  the learning rate near 0 stays below 1e-4). Only `ref_kl` above it is the policy moving away
  from the initial model.
- At NeMo-RL's learning rate (3e-7) accuracy does not move yet in 200 steps (0.48-0.51);
  B3's importance-sampling weights average 1.0000. The runs show that nothing diverges,
  drifts or breaks in the weight sync.

## Run

From `atomrl/`, with a site file ([site.example.yaml](../site.example.yaml); the 2x8
cases need both hosts and the `rdma` section):

```bash
bash examples/run.sh prepare B4 --site site.yaml   # once: Qwen3-8B, Qwen3-30B-A3B, the data
bash examples/run.sh B4 --site site.yaml --atom-src ..
python3 examples/GRPOPerf/metrics.py <trainer log> --gpus 8   # 16 for B1, B2, F1, F2

# all eight in a row, about 5 hours
bash examples/run.sh suite --tier long --only B1,B2,B3,B4,F1,F2,F3,F4 --site site.yaml --atom-src ..
```

B2's and F2's rows run as `B2-12P4` and `F2-12P4`, 12 trainers (8 + 4) and four ATOM
replicas on the second node's other GPUs; the other rows are the cases as they are.
`B1-1N` runs B1's colocated layout on one node, for training-side experiments.

```bash
bash examples/run.sh B2-12P4 --site site.yaml --atom-src .. --no-check
```

`prepare` downloads both models at the revisions DAPO's examples pin (the ones measured)
and builds NeMo-RL's data with [prepare_data.py](prepare_data.py): OpenMathInstruct-2
`train_1M` at a pinned revision, its 95/5 split (seed 42) and `cot.txt` prompt, 950,000 /
50,000 rows. A case takes 30-55 minutes. Each run is judged like every other example (exit code,
final step, weight sync, the throughput check) and lands in
`atomrl/output/regression/<timestamp>-<label>/` with `summary.md` and the logs; `metrics.py`
computes the table's columns and NeMo-RL's gate from a trainer log. For NeMo-RL's layouts add,
for example, `--override policy.training.megatron_cfg.expert_model_parallel_size=16` to B1.

The numbers were measured with `rocm/atom-dev:atomrl-dapo-gfx950-rocm7.2.4-261125`
(`4ede265ad0f1`) and this checkout mounted with `--atom-src ..`: F1 and F2 need its FP8
expert requantization, which the image's ATOM predates. The colocated cases (B1, F1, B4, F4)
set `PYTORCH_CUDA_ALLOC_CONF` empty: on this image's torch `expandable_segments` takes
effect, and a sleeping colocated ATOM then keeps its memory.

## FP8 training

The four FP8 cases with training in FP8 as well: TransformerEngine tensorwise current
scaling, E4M3, first and last layer in BF16, the rollout unchanged, each in its
configuration from Examples; one run each on 261125
([measured/runs.csv](measured/runs.csv)). NeMo-RL's FP8 rows train in BF16, so its gate,
written for a BF16 trainer, does not apply; the mismatch is reported instead. These runs
predate the NeMo-RL GRPO above (KL 0, group mean/std advantages, string-match reward)
and have not been repeated on it.

| # | Tokens/s/GPU | Step (s) | Train (s) | vs BF16 training | `k3_kl` | `token_mult_prob_error` median / last |
|---|---|---|---|---|---|---|
| F1 | 2,370 | 176.8 | 78.8 | -12.9% | 0.0069 | 1.20 / 2.74 |
| F2 | 2,076 | 202.0 | 138.8 | -22.8% | 0.0069 | 1.70 / 1.06 |
| F3 | 6,315 | 136.1 | 75.7 | +0.6% | 0.0074 | 1.95 / 1.96 |
| F4 | 5,968 | 144.3 | 39.2 | +6.0% | 0.0074 | 3.15 / 30.2 |

- "vs BF16 training" is the same configuration with BF16 training (2,719, 2,688, 6,278 and
  5,631 tokens/s/GPU). F2 is 1.14x NeMo-RL's F2 (1,819, FP8 generation with BF16 training).
- Qwen3-8B trains 18-19% faster; Qwen3-30B-A3B's grouped expert GEMMs are slower in FP8 than
  in BF16 (hipBLASLt reports no tuned kernel for them). F3 is bound by generation.
- `k3_kl` is 0.0028-0.0034 with BF16 training: the FP8 training quantizes per tensor, the
  rollout per block (8B) or per token and channel (30B).
- TE refuses blockwise (`Float8BlockScaling`) and MXFP8 on this image. DAPO examples 3 and
  4 train in FP8 through Lumen's blockwise kernels on FSDP2, which Megatron native does not use.

```bash
bash examples/run.sh F4 --site site.yaml --atom-src .. --no-check \
  --override quantization.training.fp8=e4m3 \
  --override quantization.training.fp8_recipe=tensorwise \
  --override quantization.training.num_first_layers_in_bf16=1 \
  --override quantization.training.num_last_layers_in_bf16=1
# F1 also: --override policy.generation.atom_cfg.gpu_memory_utilization=0.52
# (FP8 training holds more memory, and ATOM's wake runs out at 0.60)
```

## Metrics

- Tokens/s/GPU = tokens of a step (prompt + response, all 2048 samples) / step time /
  every GPU of the run, rollout GPUs included; AtomRL logs it as `perf/throughput`.
  G-Avg seq len = mean response tokens.
- Gate: `token_mult_prob_error` (mean exp|rollout logprob - train logprob|) has a
  median below 1.1 over the 10 steps and is below 1.1 at the last step.

## Differences from NeMo-RL

The same algorithm and the same work per step; what differs is where things run.

- The reference weights stay on the GPU and are exchanged in place for the reference
  forward; NeMo-RL keeps them on the host and copies them in and out every step. The
  optimizer is not offloaded either: MI355X has the memory for both.
- 1-step off waits for the in-flight batch before the weight update; NeMo-RL updates the
  weights in flight (`in_flight_weight_updates`), so some of its trajectories mix two
  policies.
- FP8 generation uses ATOM's `ptpc_fp8` (30B) and `per_block_fp8` (8B) where NeMo-RL uses
  MXFP8, with the same layers kept in BF16; ATOM's weight sync cannot requantize MXFP8
  experts.
- Gradients are reduced in FP32 (NeMo-RL: BF16).
- Each row uses its fastest parallel configuration (in the table), not NeMo-RL's layout.
- NeMo-RL validates every 10 steps; these runs do not, which does not change training.
- The prompts come from the same data, split and seed, in a different shuffle order.
