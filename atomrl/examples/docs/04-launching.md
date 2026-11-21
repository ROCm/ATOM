> [Examples README](../README.md) > Launching

# 4. Launching

## 4.1 Configs and scale

All under `examples/DAPO/configs/`:

```text
# 1  8B BF16
dapo_qwen3_8b_ray_vllm_smoke.yaml                     resp=512
dapo_qwen3_8b_ray_vllm_longrun.yaml

# 2, 3  8B vLLM FP8 (shared config; training precision selected by TRAIN_FP8)
dapo_qwen3_8b_ray_vllm_fp8_smoke.yaml                 resp=512
dapo_qwen3_8b_ray_vllm_fp8_4k_smoke.yaml              resp=4096
dapo_qwen3_8b_ray_vllm_fp8_longrun.yaml

# 4  8B ATOM FP8
dapo_qwen3_8b_ray_atom_fp8_4k_smoke.yaml              resp=4096
dapo_qwen3_8b_ray_atom_fp8_longrun.yaml

# 5  8B ATOM BF16 (field-for-field identical to example 4, quantization off)
dapo_qwen3_8b_ray_atom_bf16_4k_smoke.yaml             resp=4096
dapo_qwen3_8b_ray_atom_bf16_longrun.yaml

# 6  MoE FSDP2
dapo_qwen3moe_a3b_ray_vllm_verlref_4k_smoke.yaml
dapo_qwen3moe_a3b_ray_vllm_verlref_longrun.yaml

# 7  MoE Megatron EP=8
dapo_qwen3moe_a3b_ray_megatron_verlref_4k_smoke.yaml
dapo_qwen3moe_a3b_ray_megatron_verlref_longrun.yaml
dapo_qwen3moe_a3b_ray_megatron_verlref_4k_longrun.yaml   # compressed, conclusive in hours
```

**8B long-run scale** (identical for examples 1-5): 1000 steps,
`train_global_batch_size=512` (32 prompts x 16), `gen_batch_size=96`,
`max_response_length=20480`, `max_total_sequence_length=21504`, lr 1e-6 / warmup 10 /
wd 0.1 / clip_grad 1.0, clip 0.2/0.28/10 + token-mean, `overlong_buffer` 512/1.0,
`filter_groups` on acc with at most 10 rounds, `rollout_is=token` with threshold 2.0,
`val_steps=10` / `save_steps=50` / seed 10086.
The BF16 and FP8 configs **differ by exactly one line, `vllm_cfg.quantization`**.

**MoE long-run scale** (identical for examples 6 and 7): prompt=2048, resp=20480,
**128 prompts x 16 = 2048 sequences**, `gen_batch_size=384`, **lr warmup = 0**,
1000 steps. The two configs are **field-for-field identical** apart from
`policy.training_backend` and `megatron_cfg`, so any difference between the two lines can
only come from the training backend.

> **Mind the units**: `train_global_batch_size` counts **sequences** (2048) while
> `gen_batch_size` counts **prompts** (384). The framework derives the prompt count as
> `train_prompts = train_global_batch_size // num_generations`.

**Example 7's `megatron_cfg` (long run)**:

```yaml
use_distributed_optimizer: true
tensor_model_parallel_size: 1
pipeline_model_parallel_size: 1
context_parallel_size: 1
expert_model_parallel_size: 8       # 128 experts over 8 cards, 16 each
sequence_parallel: false
moe_grouped_gemm: true
moe_permute_fusion: true
moe_aux_loss_coeff: 0.0
moe_router_dtype: fp32              # pairs with ATOMRL_FP32_MOE_ROUTER=1
recompute_granularity: full         # required at resp=20480
recompute_method: uniform
recompute_num_layers: 1
log_probs_chunk_size: 1024
enable_dynamic_batch: true
max_tokens_per_gpu: 8192            # not 22528, see troubleshooting
```

**Why EP=8 and not something else**: `DP = 8 / (TP x PP x CP) = 8`, matching FSDP2's DP8,
so each rank still sees 2048/8 = 256 sequences. Anything that shrinks DP doubles the
distributed optimizer state per card (DP 8->4 costs about 8.5 GB more) and gives back what
was saved on activations — **CP=2 OOMs immediately**, dying earlier than CP=1.

---

## 4.2 Environment variables

Every `run_dapo.sh` switch is an environment variable; **the script itself never needs
editing**:

- `MODE` (default `bf16`): `bf16` / `fp8` / `atomfp8` / `atombf16`, selecting the config
  plus the rollout engine and precision.
- `TRAIN_FP8` (default `0`): `1` enables Lumen FP8 blockwise2d on the training side and
  sets `FP8_PARAM_MANAGER=0` automatically.
- `STEPS` (default `1000`): overrides `num_training_steps`.
- `CONFIG_OVERRIDE` (default: derived from `MODE`): names a config directly.
  **Required for smoke runs.**
- `EXTRA_OVERRIDE` (default empty): appends arbitrary Hydra overrides, space separated.
- `MODEL_PATH` / `TRAIN_FILE` / `VAL_FILE`: swap model or data; defaults follow the
  standard `$DATA_ROOT` layout.
- `LOG`: log path, default `$DATA_ROOT/logs/$RUN_ID.log`, also written to
  `/tmp/run_dapo_log.txt`.
- `ATOMRL_FP32_MOE_ROUTER` (default `1`): **must be passed explicitly for examples 6
  and 7**, see below.
- `PYTORCH_CUDA_ALLOC_CONF`: **set it to empty**; the ROCm/HIP allocator does not support
  `expandable_segments`.

> **The script is the single source of truth. If it gets modified by accident, restore
> it**: `git -C "$RL_ROOT/ATOM/atomrl" checkout -- examples/DAPO/run_dapo.sh`.

All commands use this prefix. `export VAR=` means "set to empty", which the script reads
as a signal to `unset` it:

```bash
S=$RL_ROOT/ATOM/atomrl/examples/DAPO/run_dapo.sh
ENVX="export RL_ROOT='$RL_ROOT' DATA_ROOT='$DATA_ROOT' PYTORCH_CUDA_ALLOC_CONF=;"
```

> `run_dapo.sh` starts with `: "${RL_ROOT:?}"`, so an empty `RL_ROOT` inside the
> container exits immediately. `ENVX` exists to avoid exactly that — a detached exec should
> not rely on the `-e` injection from the container setup.

---

## 4.3 Examples 1-5: Qwen3-8B-Base

Run the smoke first, in the foreground. **A smoke run must point `CONFIG_OVERRIDE` at
a `*_smoke.yaml`**; setting only `STEPS=1` still uses the long-run config
(resp=20480, batch=512), which is not a smoke run:

```bash
# Example 1
sudo docker exec "$CONTAINER" bash -lc "$ENVX \
  CONFIG_OVERRIDE=examples/DAPO/configs/dapo_qwen3_8b_ray_vllm_smoke.yaml \
  STEPS=1 MODE=bf16 LOG=$DATA_ROOT/logs/smoke-bf16.log bash '$S'; \
  tail -80 \"\$(cat /tmp/run_dapo_log.txt)\""

# Example 2 (TRAIN_FP8=0, verifying only the fp8_per_block rollout)
sudo docker exec "$CONTAINER" bash -lc "$ENVX \
  CONFIG_OVERRIDE=examples/DAPO/configs/dapo_qwen3_8b_ray_vllm_fp8_smoke.yaml \
  STEPS=1 MODE=fp8 LOG=$DATA_ROOT/logs/smoke-fp8.log bash '$S'; \
  tail -80 \"\$(cat /tmp/run_dapo_log.txt)\""

# Example 4 (4k config; finish the precompile first)
sudo docker exec "$CONTAINER" bash -lc "$ENVX \
  CONFIG_OVERRIDE=examples/DAPO/configs/dapo_qwen3_8b_ray_atom_fp8_4k_smoke.yaml \
  STEPS=1 MODE=atomfp8 LOG=$DATA_ROOT/logs/smoke-atomfp8.log bash '$S'; \
  tail -80 \"\$(cat /tmp/run_dapo_log.txt)\""

# Example 5 (also needs the precompile; does not need the RMSNorm patch)
sudo docker exec "$CONTAINER" bash -lc "$ENVX \
  CONFIG_OVERRIDE=examples/DAPO/configs/dapo_qwen3_8b_ray_atom_bf16_4k_smoke.yaml \
  STEPS=1 MODE=atombf16 LOG=$DATA_ROOT/logs/smoke-atombf16.log bash '$S'; \
  tail -80 \"\$(cat /tmp/run_dapo_log.txt)\""
```

Then start the long run, detached so it survives disconnects:

```bash
sudo docker exec -d "$CONTAINER" bash -lc "$ENVX STEPS=1000 MODE=bf16                 bash '$S'"  # 1
sudo docker exec -d "$CONTAINER" bash -lc "$ENVX STEPS=1000 MODE=fp8                  bash '$S'"  # 2
sudo docker exec -d "$CONTAINER" bash -lc "$ENVX STEPS=1000 MODE=fp8      TRAIN_FP8=1 bash '$S'"  # 3
sudo docker exec -d "$CONTAINER" bash -lc "$ENVX STEPS=1000 MODE=atomfp8  TRAIN_FP8=1 bash '$S'"  # 4
sudo docker exec -d "$CONTAINER" bash -lc "$ENVX STEPS=1000 MODE=atombf16             bash '$S'"  # 5
```

> Do not pass `TRAIN_FP8=1` to example 5: `MODE=atombf16` unsets `LUMEN_FP8`,
> `FP8_PARAM_MANAGER`, `LUMEN_FP8_SCALING` and friends, so training is unconditionally
> BF16. It also **does not import the Lumen/AITER norm patch** (HF Qwen3's RMSNorm is
> already model-sensitive), which is what makes it strictly comparable to example 1.

> Consider starting with `STEPS=30` to confirm memory and metrics look healthy before
> committing to 1000 steps.
> W&B is optional: put `WANDB_API_KEY=xxxx` in `$RL_ROOT/wandb.key` and the script picks
> it up.
> To change checkpoint frequency use
> `EXTRA_OVERRIDE='checkpointing.save_steps=10 checkpointing.save_total_limit=2'`; one 8B
> FSDP2 checkpoint is about 90 GB, so check `df -h` first.

Confirm it is actually running:

```bash
sudo docker exec "$CONTAINER" bash -lc 'L=$(cat /tmp/run_dapo_log.txt); sleep 200
  grep -aE "setup .ray-controller. complete|filter_groups round|View run" "$L" | tail -3
  grep -aiE "Traceback|OutOfMemory|CUDA error" "$L" | tail'
```

---

## 4.4 Example 6: MoE + FSDP2

```bash
ENVX_MOE="export RL_ROOT='$RL_ROOT' DATA_ROOT='$DATA_ROOT' SCRATCH_ROOT='$DATA_ROOT' \
ATOMRL_FP32_MOE_ROUTER=0 PYTORCH_CUDA_ALLOC_CONF=;"

# smoke (4k, 3 steps, ~10 min, of which ~5 min is 8 actors each loading the 57GB model)
sudo docker exec "$CONTAINER" bash -lc "$ENVX_MOE \
  CONFIG_OVERRIDE=examples/DAPO/configs/dapo_qwen3moe_a3b_ray_vllm_verlref_4k_smoke.yaml \
  MODEL_PATH=\$DATA_ROOT/models/Qwen3-30B-A3B-Base STEPS=3 MODE=bf16 \
  LOG=\$DATA_ROOT/logs/moe-4k-smoke.log bash '$S'; \
  tail -40 \"\$(cat /tmp/run_dapo_log.txt)\""

# long run (check the disk first: at least 400G free)
df -h "$DATA_ROOT"
sudo docker exec -d "$CONTAINER" bash -lc "$ENVX_MOE \
  CONFIG_OVERRIDE=examples/DAPO/configs/dapo_qwen3moe_a3b_ray_vllm_verlref_longrun.yaml \
  MODEL_PATH=\$DATA_ROOT/models/Qwen3-30B-A3B-Base STEPS=1000 MODE=bf16 \
  LOG=\$DATA_ROOT/logs/longrun-moe.log bash '$S'"
```

> **`MODEL_PATH` must be given explicitly** — `run_dapo.sh` defaults to the 8B model.

> **`ATOMRL_FP32_MOE_ROUTER=0` is mandatory here.** The framework defaults to fp32, but
> this line wants the router in BF16: FSDP2 and vLLM run **the same PyTorch router op with
> the same layout**, so BF16 rounding lands both sides on the same top-8 experts, and
> agreeing with each other matters more than raising precision on one side. The log should
> show `[atomrl] MoE router patched on 48 gates (fp32=False)`; `True` means the variable
> was forgotten.

> **`SCRATCH_ROOT` must be exported**: the config resolves `model_name` and
> `checkpoint_dir` through `${oc.env:SCRATCH_ROOT}`, and omegaconf exits outright if it
> cannot resolve. **This is required even when checkpointing is disabled.**

**On a new machine, verify weight sync end to end before the first real MoE run.** If
transformers 5.x's fused expert tensors (~57 GB, **93% of the parameters**) fail to match
vLLM's `expert_params_mapping`, vLLM takes a silent `continue` branch: no error, no load,
and the rollout engine's experts stay at their on-disk values forever. The coverage
assertion (`ATOMRL_WEIGHT_SYNC_CHECK=error`) is on by default; a bit-exact comparison on
top of it is safer still:

```bash
sudo docker exec "$CONTAINER" bash -lc "$ENVX_MOE ATOMRL_WEIGHT_SYNC_VERIFY=1 \
  CONFIG_OVERRIDE=examples/DAPO/configs/dapo_qwen3moe_a3b_ray_vllm_verlref_4k_smoke.yaml \
  MODEL_PATH=\$DATA_ROOT/models/Qwen3-30B-A3B-Base STEPS=3 MODE=bf16 \
  LOG=\$DATA_ROOT/logs/moe-verify.log bash '$S'"
```

> Passing means **no exception**: exit 0 says all 96 fused tensors x 8 replicas x 3 syncs
> matched bit for bit. Failure raises either
> `weight sync verify failed for ... shard w1/w3/w2` or
> `weight sync (colocate-ipc) left N/M rollout parameters untouched: ...`.

Also run the CPU-only unit tests to confirm the code is complete:

```bash
sudo docker exec "$CONTAINER" bash -lc 'cd "$RL_ROOT/ATOM/atomrl" &&
  python3 -m pytest atomrl/tests/atomrl/test_moe_weight_sync.py &&      # 11 checks, fused expert sync
  python3 -m atomrl.tests.test_rollout_routing &&      #  9 checks
  python3 -m atomrl.tests.test_dataproto_ragged &&     # 10 checks
  python3 -m atomrl.tests.test_mismatch_metrics'       #  4 checks
```

---

## 4.5 Example 7: MoE + Megatron EP=8

```bash
# smoke: the config's moe_router_dtype is null, hence =0 here
sudo docker exec "$CONTAINER" bash -lc "$ENVX_MOE \
  CONFIG_OVERRIDE=examples/DAPO/configs/dapo_qwen3moe_a3b_ray_megatron_verlref_4k_smoke.yaml \
  MODEL_PATH=\$DATA_ROOT/models/Qwen3-30B-A3B-Base STEPS=3 MODE=bf16 \
  LOG=\$DATA_ROOT/logs/moe-4k-smoke-megatron.log bash '$S'; \
  tail -40 \"\$(cat /tmp/run_dapo_log.txt)\""

# long run: the config's moe_router_dtype is fp32, so this must flip to =1
ENVX_MEGA="export RL_ROOT='$RL_ROOT' DATA_ROOT='$DATA_ROOT' SCRATCH_ROOT='$DATA_ROOT' \
ATOMRL_FP32_MOE_ROUTER=1 PYTORCH_CUDA_ALLOC_CONF=;"

df -h "$DATA_ROOT"     # a Megatron dist-checkpoint is about 400GB
sudo docker exec -d "$CONTAINER" bash -lc "$ENVX_MEGA \
  CONFIG_OVERRIDE=examples/DAPO/configs/dapo_qwen3moe_a3b_ray_megatron_verlref_longrun.yaml \
  MODEL_PATH=\$DATA_ROOT/models/Qwen3-30B-A3B-Base STEPS=1000 MODE=bf16 \
  LOG=\$DATA_ROOT/logs/longrun-moe-megatron.log bash '$S'"
```

> **`ATOMRL_FP32_MOE_ROUTER` only affects the vLLM worker**; the Megatron training side
> reads `megatron_cfg.moe_router_dtype`. **Both must be flipped together.**

**Why this line uses fp32 while example 6 uses BF16**: Megatron runs its own `TopKRouter`
feeding a grouped-GEMM, a different implementation from vLLM's. In BF16 the two pick
different experts on near-tied tokens, and flipping one expert moves that token's
log-prob a lot. Measured: with `moe_router_dtype: null`, `rollout_corr/kl` sat flat at
6.5e-4 through step 77 and then climbed about 16% per step to 2.4e-2 by step 110; with
fp32 it was still 7e-4 at step 185.

The long-run config's `save_steps: 5` is aggressive (9.3 min/step -> a 400GB write roughly
every 46 minutes). Raise it to match your fault-tolerance needs:
`EXTRA_OVERRIDE='checkpointing.save_steps=20'`.

After launching, confirm three things before walking away:
`MoE+EP spec ... EP=8 ... router_dtype=fp32`, no `Traceback` / `HSA_STATUS`, and
`callbacks: step=1` within roughly 14 minutes.

### Checkpoint verification (example 7 / Megatron)

A complete Megatron distributed checkpoint requires exactly:
- 8 model shards (`model_world_size_8_rank_*.pt`)
- 8 optimizer metadata shards (`optim_world_size_8_rank_*.pt`)
- 8 extra-state shards (`extra_state_world_size_8_rank_*.pt`)
- 8 optimizer parameter-state shards (`optim_parameter_state_world_size_8_rank_*.pt`, each ~41-45 GiB)

Verify after the first checkpoint save:

```bash
# Fill in the RUN_ID from the launch log
RUN_ID=<your-run-id>
P="$DATA_ROOT/ckpts/$RUN_ID/global_step_5/actor"
ls -lh "$P"/model_world_size_8_rank_*.pt
ls -lh "$P"/optim_world_size_8_rank_*.pt
ls -lh "$P"/optim_parameter_state_world_size_8_rank_*.pt
ls -lh "$P"/extra_state_world_size_8_rank_*.pt
```

Automated count check:

```bash
sudo docker exec -e RUN_ID="$RUN_ID" "$CONTAINER" bash -lc '
python3 - <<PY
import os
from pathlib import Path

p = Path(os.environ["DATA_ROOT"]) / "ckpts" / os.environ["RUN_ID"] / "global_step_5" / "actor"
for pattern in (
    "model_world_size_8_rank_*.pt",
    "optim_world_size_8_rank_*.pt",
    "optim_parameter_state_world_size_8_rank_*.pt",
    "extra_state_world_size_8_rank_*.pt",
):
    files = list(p.glob(pattern))
    print(pattern, len(files), sum(x.stat().st_size for x in files))
    assert len(files) == 8
print("checkpoint verification passed")
PY
'
```

> **Never resume from an incomplete checkpoint.** A checkpoint missing the large
> `optim_parameter_state_*` shards loads without error but produces NaN on the next
> optimizer step — the FP32 master weights and Adam moments are missing.
> See [Multi-node RDMA](05-multinode-rdma.md#44-checkpoint-corruption-history) for a
> detailed incident record.

---

## 4.6 Disabling checkpoints (when disk is short)

```bash
EXTRA_OVERRIDE='checkpointing.save_steps=1000000 checkpointing.resume=false'
```

> **Do not write `checkpointing.checkpoint_dir=`.** Hydra parses the empty value as
> `None` and omegaconf immediately fails with
> `Incompatible value 'None' for field of type 'str'`. A `save_steps` large enough never to
> be reached is the clean way. Think it through first — a crash then means starting over.

---

## 4.7 Health criteria

**Hard criteria for a passing smoke run**: exit 0, no `Traceback`, no `HSA_STATUS`, and
`RLTrainer.setup (ray-controller) complete: ... actor_workers=8` in the log.

Per-example `rollout_corr/kl` / memory / step time (at `resp=20480`) / checkpoint size:

- **Example 1**: kl ~0.001, `mem/actor_allocated_gb` 11.6 GB, 4-5 min/step, ckpt ~90 GB.
  `grad_norm` ~0.85, `ppo_kl` ~0.
- **Examples 2, 3**: kl **~0.003-0.004** (the FP8 gap, expected; only worry as it
  approaches the TIS threshold of 2.0). Memory and step time as example 1, ckpt ~90 GB.
- **Example 4**: kl ~0.004 (slightly above vLLM FP8), memory as above. no-eager level=3
  mainly speeds up rollout, but sleep/wake plus weight sync add fixed overhead.
  ckpt ~90 GB.
- **Example 5**: kl should land at example 1's magnitude (~0.001), not example 4's
  ~0.004 — quantization is off, so all that remains is the implementation difference
  between ATOM and the training side. **This is the check for whether ATOM is aligned
  correctly**: if ATOM BF16 also sits at 0.004, the gap is not from FP8, so go look at
  the ATOM RMSNorm alignment in [troubleshooting](06-troubleshooting.md). Memory, step time and ckpt match example 4.
- **Example 6**: kl ~1.5e-3, `mem/actor_max_reserved_gb` 75-115 GB, ~11 min/step,
  ckpt **~342 GB**. The `lr` at step 1 is already `9.99998e-07` (full value), confirming
  warmup is really 0; seeing `2e-07` means the wrong config is in use.
- **Example 7**: kl ~1.5e-3 (healthy band 6e-4 to 1.8e-3), allocated 72 GB (4k) /
  130 GB (20k), `max_reserved` 128-140 GB, ~9.3 min/step (first step ~14 min including
  vLLM load), ckpt **~400 GB**. The log should carry
  `MoE+EP spec: num_experts=128 topk=8 moe_ffn=768 | tp=1 pp=1 cp=1 EP=8 etp=1
  -> local_experts/rank=16 | grouped_gemm=True router_dtype=fp32 pre_softmax=False`.

Across all of them: `timing/weight_sync_s` stays at 1.1-1.7 s and **does not grow with
step count**; `mem/actor_allocated_gb` stays constant (`max_reserved` fluctuating with
each step's batch is normal — **live memory moving is what indicates a leak**).

**The single most important criterion: `rollout_corr/kl` must not climb monotonically
with step count.** Going down is normal (the policy converges and becomes more
deterministic, so divergence in log space shrinks). Going up has three causes, in order
of likelihood: MoE router precision mismatched between the two sides, weight sync
missing parameters (recheck with `ATOMRL_WEIGHT_SYNC_VERIFY=1`), or a new alignment bug.

**Watch `seq/max_len` for length collapse.** Fluctuating near the budget ceiling is
healthy — it means some sequence hits the cap every step. Monotonically shrinking means
it has collapsed.

### Measured reference curves

**Example 6** (101 steps / 21.6 hours): `reward/accuracy` 0.136 -> 0.494 (step 50) ->
**0.581**. On AIME-2024 online validation (every 10 steps, greedy),
`val-core/acc/mean@1` rose from 0.041 at step 10 to **0.361** at step 90, and
`val/response_length_mean` from 2407 to 10389 — the model learned to think longer, which
is the evidence this line works.

**Example 7** (the `verlref_4k_longrun` compressed recipe, 91 steps):
`reward/accuracy` 0.168 -> 0.42, `seq/mean_response_len` 773 -> 925, `rollout_corr/kl`
0.00136 -> 0.00060 (falling, which is correct), AIME `mean@1` 0.086 -> 0.199.

> **The known entropy collapse is not a bug**: examples 6 and 7 are configured with
> `entropy_coeff=0`, so monotonically falling entropy (0.844 -> 0.094 over 101 steps) is a
> direct consequence of that setting. Only worry when entropy drops below 0.05 **and**
> length starts shrinking at the same time. Fixing it means adding `entropy_coeff` first.

---

## 4.8 Monitoring / stopping / resuming

```bash
# monitor
sudo docker exec "$CONTAINER" bash -lc 'L=$(cat /tmp/run_dapo_log.txt)
  grep -aE "callbacks: step=" "$L" | tail -5
  grep -aiE "Traceback|OutOfMemory|CUDA error|HSA_STATUS" "$L" | tail'

# stop (clearing the Ray actors too)
sudo docker exec "$CONTAINER" bash -lc '
  ray stop --force 2>/dev/null
  pkill -9 -f "[l]umenrl.trainer.main"; pkill -9 -f "[V]LLMRayServer"; pkill -9 -f "[E]ngineCore"
  sleep 10; rocm-smi --showmeminfo vram | grep -i used | head -1'   # expect ~298MB/card
```

**Resuming**: the configs set `resume: true`, so rerunning the same long-run command
picks up from the most recent checkpoint. On a new machine with an empty directory that
simply means starting at step 0.
