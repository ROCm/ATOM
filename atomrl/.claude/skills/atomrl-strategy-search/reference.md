# AtomRL Strategy Search Reference

AtomRL's Ray path with Megatron native training and ATOM rollout, as measured on
MI355X with `rocm/atom-dev:atomrl-dapo-gfx950-rocm7.2.4-261125`. Re-check a knob
against the code before relying on it after an upgrade.

## Knobs

| Key | Default | Notes |
|---|---|---|
| `algorithm.force_on_policy_ratio` | false | Old log-probs from the training forward, one forward less. Megatron native only; works with rollout IS (`token` / `sequence`, weights formed in the loss), refused with `rollout_is_batch_normalize`, rollout RS and `bypass_mode`. Use it only where the reference does |
| `algorithm.grpo.kl_coeff`, `algorithm.kl_penalty` | 0, `kl` | KL to the initial model; `k3` is NeMo-RL's. Adds a reference forward per step |
| `controller.ray.ref_in_actor` | false | Reference forward through the actors' own model with the initial weights swapped in, on the GPU; no reference workers |
| `algorithm.adv_estimator` | from `algorithm.name` | `grpo_leave_one_out` is NeMo-RL's GRPO advantage |
| `policy.training.megatron_cfg.moe_permute_fusion` | false | Fused MoE permute / sort / unpermute (NeMo-RL's default); MoE training about 9% faster |
| `megatron_cfg.{tensor,pipeline,context,expert}_model_parallel_size` | 1 | No VPP. A MoE with TP > 1 turns sequence parallel on. EP ranks are consecutive, so EP stays inside a node |
| `megatron_cfg.enable_dynamic_batch`, `max_tokens_per_gpu` | false, 0 (21504) | Token budget of a training / log-prob micro batch. GRPOPerf uses true / 16384; Qwen3-30B-A3B on one node already peaks near 238 GB |
| `megatron_cfg.freeze_moe_router`, `moe_router_dtype`, `moe_router_load_balancing_type` | false, fp32, aux_loss | NeMo-RL's GRPO freezes the router, in fp64, with no load balancing |
| `controller.ray.actor.{num_workers,process_on_nodes,topology_tags.node_ips}` | - | Trainer placement, e.g. `[8,4]` with `node_ips=A,B`; do not set `node_ip` and `node_ips` together |
| `controller.ray.rollout.{num_workers,process_on_nodes,topology_tags.node_ip}` | 0 | > 0 separates rollout GPUs; required for 1-step off |
| `controller.ray.one_step_off_policy` | false | Overlaps generation with training; needs `vllm_cfg.enable_sleep_mode: false` |
| `policy.generation.atom_cfg.{tensor_parallel_size,gpu_memory_utilization,online_quant_config}` | 1, 0.6, - | `transport: ray_http`. FP8 formats: `ptpc_fp8` (MoE), `per_block_fp8` (dense) |
| `policy.generation.vllm_cfg.{max_num_seqs,max_num_batched_tokens}` | 64, 8192 | Read by ATOM too; `max_num_seqs` at least the rows per replica |
| `policy.generation.vllm_cfg.update_weights_bucket_megabytes` | 512 | Bucket size of the colocated IPC weight sync |

No effect on this path: `num_ppo_epochs`, `num_mini_batches`, `balance_batch`,
`train_micro_batch_size`, and `policy.max_token_len_per_gpu` for Megatron. A
misspelled key under `engine_kwargs` is dropped silently. Not settable from YAML:
`grad_reduce_in_fp32` (always true) and `overlap_grad_reduce`.

## Reading A Step

Each step logs one `step=N key=value ...` line.

- Phases: `timing_s/{gen,old_log_prob,ref,update_actor,update_weights}`;
  `timing_s/step` is their sum and is the step time NeMo-RL's Tokens/s/GPU uses.
  `timing/step_s` is measured differently and can be a few seconds shorter.
- Throughput: `perf/throughput` (NeMo-RL's Tokens/s/GPU) and `perf/total_num_tokens`.
- Correctness: `rollout_corr/token_mult_prob_error`, `entropy`, `ref_kl`,
  `rollout_corr/rollout_is_mean`, `actor/grad_norm`.
- ATOM preemptions appear only in the replica logs (`ATOM[i] kv: ... preemptions=`).

`examples/GRPOPerf/metrics.py <trainer.log> --gpus N` gives the steps 3-8
throughput, the phase means and NeMo-RL's gate.

## Harness

```bash
cd atomrl
bash examples/run.sh <id or alias> --site <site.yaml> --atom-src .. --no-check \
  [--steps N] [--override key=value ...] [--env K=V ...] [--dry-run]
```

- Aliases: GRPOPerf's B1-B4, F1-F4, and its other layouts `B2-12P4`, `F2-12P4`,
  `B1-1N` (B1's colocated layout on one node, for training-side experiments).
- Overrides are OmegaConf dotlist, no spaces in values; `{hosts[0].ip}` and
  `{rdma.interface}` expand from the site file ([site.example.yaml](../../../examples/site.example.yaml)).
- Results land in `<results_dir>/<time>-<label>/<id>/trainer.log`.
- `--no-check` when the configuration differs from the entry's: its throughput check
  holds the entry's own configuration.

## Pitfalls

- `run_grpo.sh` defaults `NCCL_SOCKET_IFNAME=ens11np0`; one-node entries set it to
  `{rdma.interface}`. A trainer group across nodes needs `GLOO_SOCKET_IFNAME`, or
  Gloo advertises 127.0.0.1.
- On the 261125 image's torch, colocated runs set `PYTORCH_ALLOC_CONF` and
  `PYTORCH_CUDA_ALLOC_CONF` empty: with `expandable_segments` a sleeping ATOM keeps
  its memory, and CUDA IPC of a VMM buffer fails.
- The ATOM compile cache lives under `<root>/<Ray job id>/replica_N`, and a fresh
  Ray cluster's first job id is always the same: a cache from another precision, TP
  size or ATOM checkout is loaded as is and fails (or misbehaves) in warmup. Give
  each its own `ATOM_TORCH_COMPILE_CACHE_ROOT`. A fresh cache costs about 2 minutes
  per replica on the first run.
- Colocated ATOM with TP > 1: the custom all-reduce cannot IPC the sleep pool's
  memory, and without it ATOM's custom all-gather asserts; run with
  `--env ATOM_USE_CUSTOM_ALL_GATHER=0`.
- 2048 samples not divisible by the trainer count skip row balancing silently.

## Candidate Priorities (GRPOPerf)

In order of measured payoff, each with why it computes the same training and why
it is faster. "Pre-alignment" numbers were taken before GRPOPerf ran NeMo-RL's GRPO
(KL 0, one old-log-prob forward); the mechanisms, and so the relative effects, carry
over.

1. **Fused MoE permutations** (`moe_permute_fusion=true`, level A).
   - Equivalent: the same tokens are copied to the same experts, sorted by the same
     chunk order and combined with the same router weights; only the kernels change.
     Copies and sorts are exact; the weighted unpermute can sum a token's top-k
     outputs in another order, a bf16 rounding difference that the step-1 comparison
     and the gate cover. It is NeMo-RL's default.
   - Faster: unfused, each MoE layer runs three separate permutation passes over the
     token tensor (top-k copy before the all-to-all, `sort_chunks_by_idxs` either
     side of it, the weighted unpermute after combine), and `sort_chunks` needs the
     per-chunk token counts on the host: one device-to-host copy and stream sync per
     layer and micro batch, during which the GPU idles. TE's fused kernels do each
     pass once on the device, with no sync. Measured pre-alignment: old-log-prob
     forward -9.8%, training -9.3%. The gain grows with local experts, top-k and
     micro batches; none for dense models.
2. **Rebalance trainer and rollout GPUs in 1-step off** (level A).
   - Equivalent: the same 2048 samples per step, the same one-step staleness and one
     optimizer step; only which GPUs train and which generate, and the expert
     parallelism of training (EP4 x DP3 instead of EP8), which partitions the same
     experts differently. The data-parallel gradient reduction is an fp32 sum in
     another order, a rounding-level difference.
   - Faster: a 1-step-off step is max(generation, forwards + training) + sync. At
     8 + 8 the trainers are the slower side and the rollout GPUs wait. Moving four GPUs
     to training shortens the critical side; generation, now with twice the rows per
     replica, stays shorter because MoE decode is bound by reading the experts'
     weights, which more rows per replica share. B2 / F2 at 12 + 4: +13% pre-alignment.
     Decode time per step fits `a + c x rows per replica` from two measured points
     (predicted 126 s, measured 126 s); training does not scale with trainer count
     (smaller EP, gradients across nodes, an uneven sample split: 138 s against 104 s
     predicted linearly).
3. **Find the critical path first.** Not a change, but it decides which ones can
   help: in 1-step off, when generation is slower than the forwards plus training,
   shortening training gains nothing (B3 / F3: generation 132-134 s, the other side
   126-127 s).
4. **FP8 rollout where the task allows it** (level B: rollout log-probs change).
   - Equivalent within a tolerance: the policy is trained in BF16 and only the
     sampler's weights are quantized (attention, router and `lm_head` stay BF16);
     the importance-sampling correction weighs each token by the train / rollout
     probability ratio, and `token_mult_prob_error` measures the mismatch every step.
     Allowed only where the reference also runs FP8 generation.
   - Faster only when decode reads weights: Qwen3-30B-A3B decode reads nearly every
     expert per step, so FP8 halves those bytes (13% faster decode). Dense Qwen3-8B
     with many rows per replica spends 77-91% of decode bytes on the KV cache, so FP8
     weights do not help (F4).
5. **Memory and sync settings** (`gpu_memory_utilization`, sync buckets, sleep level;
   level A, no value changes) only when the timings show room: B4's weight sync is
   2.9 s, so bucket sizes are not worth a run.

Not optimizations: skipping a forward the reference runs (`force_on_policy_ratio`
where the reference recipe does not set it), dropping or changing the KL or any loss
term, truncating prompts or responses, changing the batch or the staleness,
`bypass_mode`. Each makes a step cheaper by computing a different training.

## Layout And Network (2 x 8 MI355X, one usable 200 Gb/s NIC per node)

- Keep per-layer communication (EP, TP) inside a node; across nodes only what
  happens once a step (weight sync, data-parallel gradients). EP16 across the two
  nodes: 6.7x slower than EP8.
- Dense Qwen3-8B fits one GPU: TP2 + SP trains 2.4x slower than TP1 x DP (223 s
  against 93 s), TP2 x PP4 slower still (NeMo-RL's B3 / B4 layouts: about 2,700 and
  2,300 tokens/s/GPU against about 6,300, pre-alignment).
- Cross-node fp32 gradient reduction does not overlap compute (about 3.8 s, 2% of
  B1's step).

## Correctness Notes

- ATOM occasionally returns a single token with a rollout log-prob near 0 where
  training has about -20, pushing `token_mult_prob_error` to tens or 1e9 for that
  step. An HF reference confirmed the rollout side is off. Repeat a run whose last
  step hits one.
- On a MoE the separate old-log-prob forward and the training forward already differ
  (`ppo_kl` 1e-5 to 5e-5, dense about 1e-6), and taking the old log-probs from the
  training forward lowered B1's gradient norm by about 28% (dense B4 unchanged).
  NeMo-RL's recipes do the same; compare gradients offline before calling the two
  equivalent elsewhere.
- NeMo-RL's fp64 router is the main cost of its GRPO settings on the MoE: B1-1N trains
  in 106.4 s a step with it, 95.2 s with fp32 (reference forward 40 s against 36 s). The
  router GEMM itself is small, so a slow fp64 GEMM for that shape is the likely cause (not
  yet profiled); a faster fp64 kernel would be the equivalent fix, an fp32 router is not
  (near-tied top-k choices differ). The KL term adds about 2%; freezing the router changes
  nothing measurable.
- `ref_kl` at step 1, when the policy is still the reference, is the difference
  between two forward paths: about 8e-5 on Qwen3-8B, 1.7e-3 on Qwen3-30B-A3B
  (routing flips).
- `bypass_mode` changes the training semantics (level C); it is not a way to skip
  a forward.

## Measured Results (GRPOPerf, NeMo-RL's GRPO)

| Case | Configuration | Tokens/s/GPU | Step (s) | Phases (s) |
|---|---|---|---|---|
| B1 | colocated EP8 x DP2, permute fusion | 2,543 | 165.0 | generation 71.1, reference 22.9, training 64.9, sync 5.8 |
| F1 | colocated EP8 x DP2, permute fusion, FP8 generation | 2,335 | 179.6 | generation 61.0, old 23.3, reference 23.0, training 65.6, sync 6.5 |
| F2 | `F2-12P4`, FP8 generation | 2,296 | 182.8 | old 32.7, reference 32.2, training 110.8, sync 6.9; generation 114.1 overlapped |
| B4 | colocated TP1 x DP8 | 5,412 | 159.3 | generation 87.4, reference 17.3, training 51.3, sync 3.1 |

The full table and run commands: `examples/GRPOPerf/README.md`.

## Dead Ends (AtomRL on the 261125 image)

| Attempt | Result |
|---|---|
| MoE `flex` dispatcher (MORI) | the image's megatron-core has no MORI fields; fails at start |
| ATOM rollout with EP | the weight reload refuses expert-parallel MoE layers |
| ATOM `data_parallel_size > 1` | exclusive with the RDMA weight sync |
| `mxfp8` rollout | Qwen3-30B-A3B reload refused, Qwen3-8B lacks a kernel |
| Qwen3-30B-A3B one-node colocated, `gpu_memory_utilization=0.75` | ATOM runs out of memory waking its KV cache after step 3; 0.65 works, with little margin once the actors also hold the reference weights (one run with a trainable router ran out at step 4) |
| Megatron native FP8 training (TE `tensorwise`) | 30B grouped expert GEMMs slower than BF16 (no tuned hipBLASLt kernels); TE has no `blockwise` or `mxfp8` on ROCm |
