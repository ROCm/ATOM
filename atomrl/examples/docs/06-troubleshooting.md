> [Examples README](../README.md) > Troubleshooting

# 6. Troubleshooting

**FP8 training diverges** (entropy ~0.04 / `grad_norm` 1e4+ / `rollout_corr/kl` 1e4+):
essentially only two causes — `FP8_PARAM_MANAGER` was not set to 0 (it conflicts with
native FSDP2's fp32 master weights), or the [vLLM RMSNorm patch](02-dependencies.md#27-vllm-aiter-rmsnorm-patch-required-for-examples-2-3-4) was never applied
(a new container needs it again).

**Falling back on memory (OOM)**:
- lower `policy.max_response_length=8192` + `max_total_sequence_length=9216` +
  `max_token_len_per_gpu=9216`;
- or lower `train_global_batch_size` / `gen_batch_size`;
- **do not** enable `fsdp_cfg.param_offload` / `optimizer_offload` on the Ray path; it
  fails with `parameters should be materialized on CPU`.

**Example 7's OOM is counterintuitive**: at `max_tokens_per_gpu: 22528` it dies in the
actor backward around step 14, but **the crash is not about peak allocated memory**
(~130 GB either way) — it is **fragmentation**. ROCm has no `expandable_segments`, and
roughly 7 bins per step each filled to 22.5k tokens repeatedly allocate and free huge
blocks, leaving reserved 42 GB above allocated. Capping bins at 8192 collapses the
fragmentation gap to 4-11 GB and drops peak reserved from 177 GB to 134 GB. **So that
8192 must not be casually raised back.**

**`weight sync (colocate-ipc) left N/M rollout parameters untouched`**: the sync missed
parameters. The exception lists the first 8 names; if they look like
`...experts.w13_weight` / `w2_weight`, the fused MoE routing did not take effect — either
the code is not up to date, or a vLLM/transformers upgrade invalidated the layout
assumption.

**Example 7 dies in `init_model` with `TypeError: 'NoneType' object is not callable`**
(the frame above it is `get_gpt_layer_with_transformer_engine_spec`): TransformerEngine
is not installed. `megatron.core` imports fine without it, so the import check in
[Dependencies §2.5](02-dependencies.md) passes and only the layer-spec check catches
it. Build TE per [§2.3](02-dependencies.md); `pip install megatron-core` alone is not
enough.

**Example 5 (`MODE=atombf16`) can die in ATOM engine init** with
`RuntimeError: Engine Core Mgr: Received unexpected SHUTDOWN signal from DP rank 0
during initialization` on the driver. That message is only the symptom — always read
the replica-side exception above it, because two different causes share it:

- `ValueError: too many values to unpack (expected 4)` under
  `rope_cache` / `fused_qk_rope_reshape_and_cache`: this is `ATOM_FORCE_ATTN_TRITON=1`.
  ATOM allocates a 5D SHUFFLE V-cache while aiter's triton kernel wants 4D in the
  non-flash layout, and the two are not `view()`-convertible. Unset the variable;
  it cannot be worked around from the AtomRL side.
- `IndexError: list index out of range` inside
  `torch/_functorch/_aot_autograd/runtime_wrappers.py`, reached from ATOM's
  `warmup_model`: **a compile cache left behind by example 4.** Both examples drive
  ATOM through `enforce_eager=false` + `compilation_config.level=3`, but example 4
  compiles the FP8 model and example 5 the BF16 one, and torch's inductor cache
  (`/tmp/torchinductor_root`) is not scoped per run the way
  `ATOM_ISOLATE_TORCH_COMPILE_CACHE` scopes ATOM's own. Example 5 then loads a graph
  whose input signature does not match and dies in AOTAutograd.

  Measured, same container and stack each time:

  | Sequence | Example 5 |
  |---|---|
  | caches cleared, example 5 alone | passes (twice, cold and warm) |
  | caches cleared, example 4, then example 5 | fails in 71 s |

  So clear the caches between any two ATOM runs of different precision:

  ```bash
  sudo docker exec "$CONTAINER" bash -lc \
    'rm -rf /tmp/aiter_configs /tmp/atom_torch_compile_cache /tmp/torchinductor_root'
  ```

  `/tmp/aiter_configs` matters on its own account too: aiter otherwise silently
  reuses a merged tuning config from the previous run.

**Example 5 runs but its `rollout_corr/kl` is clearly too high.** Distinct from the
crash above, and it does not trip the pass criteria — exit 0, generation healthy
(`ppo_kl` ~0, no `kept 0/` rounds, no `finished with reason max`). Only the
train-vs-rollout log-prob gap is off.

Since example 5 exists precisely to answer "is the gap FP8 or ATOM alignment", a
value **above** example 4's says the answer is alignment, not quantization. Do not
re-check `use_model_sensitive_rmsnorm` for this: the ATOM this release pins already
passes it at both call sites in `atom/model_ops/layernorm.py`. The misalignment is
elsewhere and worth reporting upstream.

**ATOM rollout degradation** (with `MODE=atomfp8` / `atombf16`: `filter_groups: kept 0/96`
plus `Rollout reward: accuracy=0.0000` plus many `finished with reason max` and no `eos`
in the log): generation has broken down. Check first whether the plain RMSNorm in ATOM's
`atom/model_ops/layernorm.py` passes `use_model_sensitive_rmsnorm=1`; a misalignment
shows up first as an elevated `rollout_corr/kl` (~0.007 instead of ~0.004).
**Example 5 localizes this faster**: with quantization off, a still-elevated kl means the
problem is in the ATOM/training alignment rather than in FP8.

**Do not set `TORCHDYNAMO_DISABLE` by hand.** The script keeps it globally at `=1`
(dynamo off for training actors). The dynamo that examples 4 and 5 need for no-eager
level=3 rollout is injected by `ATOMReplicaManager` through `runtime_env` when it creates
the ATOM Ray actors, so `TORCHDYNAMO_DISABLE=0` applies **to the rollout processes only**.
A top-level `export TORCHDYNAMO_DISABLE=0` makes the training actors inherit it too,
which is pure side effect on the training side.

**Two more prerequisites for examples 4 and 5** (`MODE=atomfp8` / `atombf16` set both
automatically; do not drop them when overriding by hand):
`ATOM_ISOLATE_TORCH_COMPILE_CACHE=1` (otherwise 8 single-card replicas write the same
torch compile cache concurrently and trigger a `FileNotFoundError` in Inductor's
`write_atomic -> rename`), and `enable_sleep_mode=true` with `sleep_level=2` (releasing
KV cache / weights / CUDA graph after rollout, without which the training backward
easily hits `HSA_STATUS_ERROR_OUT_OF_RESOURCES`).

**Memory not released after a run finishes or is interrupted** (`rocm-smi` still shows
~90 GB per card while `ps` shows no trainer): `run_dapo.sh` cleans up processes **before**
starting, not afterwards, so ATOM EngineCore's `spawn_main` children (and their inductor
compile workers) become orphans still holding memory. When cleaning up manually, write
the patterns as `spawn[_]main` and so on, or `pkill -f` matches its own command line and
kills itself:

The Ray actor workers orphan the same way, and killing only the launcher leaves
them holding the model. Cover all of it:

```bash
sudo docker exec "$CONTAINER" bash -lc '
  ray stop --force >/dev/null 2>&1
  for p in "[a]tomrl.trainer.main" "[r]ay::AtomActorWorker" "[r]ay::VLLMRayServer" \
           "[r]ay::ATOMRayServer" "[V]LLMRayServer" "[E]ngineCore" "[r]aylet" \
           "compile_[w]orker" "spawn[_]main" "resource[_]tracker"; do
    pkill -9 -f "$p" || true
  done
  sleep 10; rocm-smi --showmeminfo vram | grep -i used | head -3'
```

> ⚠️ **Match on process name, never on "holds a lot of VRAM".** These nodes are
> often shared, and a `pkill` keyed on memory footprint takes other people's jobs
> with it. If memory is still held after the patterns above, the remainder is not
> yours — identify it before touching it:
>
> ```bash
> # read-only: what is actually on the cards, and under whose container
> rocm-smi --showpids
> docker ps --format '{{.Names}}  {{.Status}}'
> docker top <container>            # per container, to attribute a PID
> ```
>
> A process whose command line has nothing to do with AtomRL (this repo never
> belongs to a co-tenant. Leave unrelated processes alone and see the next
> entry.

**A co-tenant took the card, and the OOM message hides it.** The scheduler can hand
you 8 GPUs on a node someone else is already computing on. The tell is an OOM whose
two numbers do not add up:

```text
torch.OutOfMemoryError: HIP out of memory. Tried to allocate 1.50 GiB.
GPU 0 has a total capacity of 287.98 GiB of which 0 bytes is free.
Of the allocated memory 47.66 GiB is allocated by PyTorch, ...
```

`0 bytes free` while PyTorch only holds 47 GiB means the other ~240 GiB is not ours.
Check the baseline **before** launching rather than diagnosing it afterwards:

```bash
sudo docker exec "$CONTAINER" bash -lc \
  'rocm-smi --showmeminfo vram | grep -i used'
```

Every card should read ~298 MB. Budget from what is left, not from the card size:
the rollout engine reserves `gpu_memory_utilization x 288 GiB` — **86 GiB at the
0.30 the example configs use** — as a fraction of the *whole* card, not of what is
free, and training needs another 44 GiB (8B) to 115 GiB (MoE) on top. A co-tenant
holding 230 GiB leaves 58 GiB and nothing fits, so wait for an exclusive node
instead of lowering `gpu_memory_utilization` to squeeze in: the metrics stop being
comparable to the baselines in [Launching](04-launching.md#47-health-criteria).

**Changing `DATA_ROOT` requires an unconditional assignment.** The
`docker run -e DATA_ROOT=...` in the container setup baked the value into the container
environment, so `export DATA_ROOT="${DATA_ROOT:-/new/disk}"` in a wrapper **has no effect**
(the variable already exists) and checkpoints silently go back to the old disk. Write
`export DATA_ROOT=/new/disk` directly.

**`logger.info` inside the vLLM worker does not reach the driver log.** That is why the
`weight sync coverage` line is invisible on example 7. This is **not** evidence that the
assertion did not run — to know whether it fired, check whether it raised.

For multi-node RDMA troubleshooting, see [Multi-node RDMA](05-multinode-rdma.md).
