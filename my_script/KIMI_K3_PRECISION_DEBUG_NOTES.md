# Kimi-K3 Precision Debug Notes

## Current Goal

Fix the Kimi-K3 accuracy issue where a single OpenAI-compatible request can
produce garbled output on the ATOM + aiter gfx1250 stack.

## Docker Environment

- Container: `zxb_kimi`
- Image: `rocm/fw-bringup:gfx1250-atom-dev-20260715-tp4_pro_flash`
- Host script used to recreate the container:
  `/home/carhuang/xiaobing/run_docker.sh`
- Local source trees mounted through `/home`:
  - `/home/carhuang/xiaobing/ATOM-K3`
  - `/home/carhuang/xiaobing/aiter-k3`
- Model path used by `my_script/run_models.sh`:
  `/data/models/Kimi-K3`

## Container Reinstall Steps

Inside `zxb_kimi`, remove image-bundled `atom`/`aiter` and reinstall local
sources:

```bash
python -m pip uninstall -y atom amd-aiter aiter || true
AITER_USE_SYSTEM_TRITON=1 python -m pip install --force-reinstall --no-deps -e /home/carhuang/xiaobing/aiter-k3
python -m pip install --force-reinstall --no-deps "git+https://github.com/fla-org/flash-linear-attention.git"
python -m pip install --force-reinstall --no-deps -e /home/carhuang/xiaobing/ATOM-K3
python -m pip install --no-deps fastsafetensors
```

Important: keep `AITER_USE_SYSTEM_TRITON=1` and `--no-deps` for the local
`aiter-k3` install. The container image already provides the gfx1250-compatible
Triton build at `/app/triton-mi450`; reinstalling Triton can break this stack.

## Import Path Issue Found

`docker exec zxb_kimi ...` defaults to working directory `/app`. From that
directory, Python imports image-bundled modules before the editable installs:

- Bad: `/app/aiter`
- Bad: `/app/triton`
- Good: `/home/carhuang/xiaobing/aiter-k3/aiter`
- Good: `/app/triton-mi450/python/triton`
- Good: `/home/carhuang/xiaobing/ATOM-K3/atom`

Observed failure from the bad working directory:

```text
ImportError: cannot import name 'QuantType' from 'aiter' (unknown location)
ImportError: cannot import name '__version__' from 'triton' (unknown location)
```

Fix applied: `my_script/run_models.sh` now changes directory to the `ATOM-K3`
repo root before starting `python -m atom.entrypoints.openai_server`.

## Verified Good Imports

From `/home/carhuang/xiaobing/ATOM-K3` inside the container:

```text
triton -> /app/triton-mi450/python/triton/__init__.py
fastsafetensors -> /opt/venv/lib/python3.12/site-packages/fastsafetensors/__init__.py
fla -> /opt/venv/lib/python3.12/site-packages/fla/__init__.py
aiter -> /home/carhuang/xiaobing/aiter-k3/aiter/__init__.py
atom -> /home/carhuang/xiaobing/ATOM-K3/atom/__init__.py
torch.cuda.is_available() -> True
torch.cuda.device_count() -> 4
```

## Loader Settings

Real-weight loading was slow with the default safetensors path. Keep the
fastsafetensors path enabled while debugging accuracy:

```bash
export ATOM_USE_FASTSAFETENSORS=1
export ATOM_FASTSAFETENSORS_NOGDS=1
export ATOM_LOADER_USE_THREADPOOL=0
export ATOM_LOADER_THREADPOOL_WORKERS=1
```

If `fastsafetensors` is missing, ATOM logs a warning and falls back to
`safe_open`, which makes validation much slower.

Loading investigation:

- `server.log` shows all four ModelRunner ranks enter the safetensors loader.
- The original iterator called `f.get_tensor(name)` before `load_model()` had a
  chance to skip unmapped or irrelevant checkpoint names. This means skipped
  tensors still paid the read/decode cost.
- Loader optimization applied: `safetensors_weights_iterator()` now yields a
  lazy tensor getter, and `load_model()` calls it only immediately before a
  matching `weight_loader` consumes the tensor.
- This does not solve the deeper TP=4 issue where each rank can still need to
  scan/read checkpoint shards, but it removes avoidable reads for skipped names
  and is the first low-risk speed fix.
- First validation after restart:
  - Before lazy tensor getter: about 16 seconds per 15.8 GiB shard, projected
    roughly 25 minutes for 96 shards.
  - After lazy tensor getter with `ATOM_FASTSAFETENSORS_NOGDS=1`: about
    6-7 seconds per shard through shards 1-38, projected roughly 10-12 minutes.
  - The speed stayed improved past shard 30, so this is not only page-cache
    reuse from the previous partial load.
- Further speed fix:
  - Added `ATOM_FASTSAFETENSORS_DEVICE`. `cpu` keeps the original conservative
    behavior; `cuda` opens fastsafetensors on `cuda:<current_device>` after each
    ModelRunner has called `torch.cuda.set_device()`.
  - Test run with `ATOM_FASTSAFETENSORS_DEVICE=cuda` reduced early shard time to
    about 2 seconds per shard, and stayed much faster than CPU loading past the
    previous slow region around shard 49.
  - `ATOM_FASTSAFETENSORS_NOGDS=0` emitted `libcufile.so does not exist`, so this
    container is not actually using CUDA/ROCm GDS. The practical win came from
    reading tensors directly onto each rank's GPU and avoiding the old CPU
    staging path.
  - `run_models.sh` now defaults to `ATOM_FASTSAFETENSORS_DEVICE=cuda` and
    `ATOM_FASTSAFETENSORS_NOGDS=1` to keep the fast GPU-target path without the
    missing-libcufile warning.

Warmup/JIT status:

- The latest warmup failed in Kimi MoE routing:
  `ModuleNotFoundError: No module named 'aiter.jit.module_moe_asm'`.
- The build log shows `module_moe_asm` failed while compiling
  `csrc/kernels/moe_fused_gate.cu` for `gfx1250`, again inside CK headers
  (`config.hpp` static assert and missing `CK_TILE_BUFFER_RESOURCE_3RD_DWORD`).
- So aiter is not fully compiled for this container/ROCm/CK/gfx1250 combination.
  Only `module_aiter_core.so` is currently present; several JIT modules are
  absent and have been bypassed one by one.
- Added `AITER_USE_TORCH_TOPK=1` fallback in `atom/model_ops/topK.py`.
  This routes grouped MoE topK through `aiter.ops.topk.*_torch` instead of
  triggering `module_moe_asm`. A small CUDA tensor smoke test passed.
- This fallback is for correctness/debug unblocking. It may be slower, but it
  lets the server reach request handling so we can finally test the garbled
  output/precision issue.

## Accuracy-Related Runtime Settings

Current `my_script/run_models.sh` keeps the previous Kimi-K3 accuracy/debug
settings:

```bash
export ATOM_USE_UNIFIED_ATTN=1
export ATOM_FORCE_ATTN_TRITON=1
export ATOM_SYNC_AFTER_LOAD=1
export AITER_DISABLE_CUSTOM_ALL_REDUCE=1
export AITER_USE_TRITON_QUANT=1
export AITER_USE_OPUS_RMSNORM=1
export AITER_GROUPED_CONTIGUOUS_TOKEN_THRESHOLD=512
```

Known context:

- Single-request output can become garbled.
- Earlier debugging indicated the first full-attention/MLA layer is the first
  place where hidden-state stats can become non-finite.
- The gfx1250 route-indexed preshuffle path can fault on small Kimi prefill
  batches, for example around 96 tokens. The current threshold routes small
  batches through the regular fused grouped path while keeping large warmup
  batches on contiguous-M.
- With `composable_kernel` re-pinned to the previous available commit
  `af7118e342580ecd3f71edce7b1d0ba465012ecf`, JIT build of
  `module_custom_all_reduce` fails on gfx1250 with CK `get_compiler_target()`
  errors. This caused `CustomAllreduce` to be disabled but still used for
  signal-buffer registration, failing with:

  ```text
  AttributeError: 'CustomAllreduce' object has no attribute '_pool'
  ```

  Workaround applied for accuracy validation: set
  `AITER_DISABLE_CUSTOM_ALL_REDUCE=1`, skip custom all-reduce JIT, and fall
  back to PyNccl/torch all-reduce.
- After fastsafetensors finished loading all Kimi-K3 shards, model
  post-processing failed while bf16 weights were being online-quantized to
  MXFP4:

  ```text
  ModuleNotFoundError: No module named 'aiter.jit.module_quant'
  RuntimeError: Error building extension 'module_quant'
  ```

  The underlying compile errors are again from CK target detection for
  `gfx1250`, matching the `module_custom_all_reduce` failure. Workaround
  applied: set `AITER_USE_TRITON_QUANT=1` so `get_hip_quant()` dispatches to
  Triton quant implementations for this validation run. A small in-container
  smoke test confirmed `get_hip_quant(QuantType.per_1x32)` can quantize a
  bf16 CUDA tensor to `torch.float4_e2m1fn_x2` without building
  `module_quant`.
- With Triton quant enabled, the run advanced past weight post-processing and
  reached warmup, then failed in the first Kimi-K3 layer RMSNorm:

  ```text
  ModuleNotFoundError: No module named 'aiter.jit.module_rmsnorm_quant'
  ```

  This is another CK-backed JIT module on the common bf16/fp16 hidden<=8192
  RMSNorm path. Workaround applied: set `AITER_USE_OPUS_RMSNORM=1` so the
  public RMSNorm entrypoints route to the existing opus backend instead of
  `module_rmsnorm_quant`. A small bf16 CUDA smoke test confirmed
  `rmsnorm2d_fwd` returns finite output through opus.

## 2026-07-19 Dummy Request Smoke Result

Dummy-weight startup now skips safetensors iteration and reaches health on port
8000. The current debugging defaults include:

```bash
export AITER_USE_TORCH_TOPK=1
export ATOM_USE_TORCH_SAMPLER=1
export ATOM_USE_TORCH_CACHE=1
export ATOM_USE_CUSTOM_ALL_GATHER=0
```

The first deterministic completion request previously crashed in
`aiter.reshape_and_cache()` because `module_cache` failed to JIT on gfx1250 with
the same CK target-detection errors seen in the other CK-backed modules. Added
`ATOM_USE_TORCH_CACHE=1` to route bf16 KV-cache insertion through a PyTorch
fallback in `atom/model_ops/attention_mha.py`.

After restart with log truncation, the dummy request:

```bash
curl -fsS --max-time 180 http://127.0.0.1:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"/data/models/Kimi-K3","prompt":"Hello","max_tokens":8,"temperature":0,"top_p":1,"stream":false}'
```

returned HTTP 200:

```text
text="!!!!!!!!"
prompt_tokens=1 completion_tokens=8 total_tokens=9
ttft_s=3.7039 tpot_s=0.3365 latency_s=6.0613
```

No Traceback, JIT failure, or runner death was found in the request log after
the response. Since this uses dummy weights, output content is only a request
pipeline smoke test, not an accuracy signal.

Current startup/request time is still dominated by warmup and first-shape
kernel selection:

- Warmup logs repeatedly show missing grouped GEMM CSV configs and default
  config selection.
- First real request still emits many per-shape grouped GEMM config misses for
  token=1.
- This is separate from real-weight load speed. For real weights, the remaining
  slow path to verify is shard I/O plus per-rank GPU tensor materialization with
  `fastsafetensors` and GDS toggled by `ATOM_FASTSAFETENSORS_NOGDS`.

### 2026-07-19 real-weight load findings

The loader now supports TP-distributed `fastsafetensors` shard reads via
`ATOM_FASTSAFETENSORS_DIST_LOAD=1`, enabled by `my_script/run_models.sh`.
Instead of every TP rank opening and copying every shard, the iterator opens
one batch of `tp_world_size` shards at a time and assigns one shard to each TP
rank through `fastsafetensors`' rank-file map.

Observed behavior on Kimi-K3 real weights:

- Full safetensors phase completed successfully for all 96 shards.
- With `MAX_NUM_BATCHED_TOKENS=128 MAX_NUM_SEQS=1 MAX_MODEL_LEN=1024`, model load
  completed at `05:01:37`, warmup completed at `05:01:39`, and `/health`
  returned OK.
- A deterministic real-weight request succeeded:
  `prompt="1+1=", max_tokens=8, temperature=0` returned text `!!!!!!!!`
  with `ttft_s=3.478`, `tpot_s=0.3346`, `latency_s=5.8221`.
- There were no new runtime tracebacks or illegal-memory logs during that
  request.

Important caveats:

- Large default warmup (`MAX_NUM_BATCHED_TOKENS=7168`, effective grouped MoE
  token log around 8192) still crashes in
  `aiter/ops/flydsl/grouped_moe_gfx1250.py::_maybe_grouped_gfx1250_a8w4_moe`
  with `CUDA error: an illegal memory access was encountered`.
- `AITER_DISABLE_GROUPED_A8W4=1` is not usable as a simple fallback on gfx1250:
  it falls through to `module_moe_sorting_opus`, which fails to JIT-build.
- `AITER_USE_FLYDSL_MOE_SORTING=1` avoids the Opus JIT path but fails for
  Kimi-K3's `E=896` with `ValueError: LDS too small for E=896`.
- The remaining load-time bottleneck is not shard file I/O alone. Later MoE
  shards take roughly 35-55s per 4-shard batch because full tensors are still
  materialized/broadcast before ATOM's MoE/linear loaders slice TP shards. A
  deeper optimization would need to thread `fastsafetensors.get_sharded(dim)`
  into the linear and MoE weight-loader paths.
- `ATOM_FASTSAFETENSORS_DIST_LOAD=1` is not stable enough for precision
  debugging. A later restart hung in `fastsafetensors` distributed
  `broadcast` during load:
  `fastsafetensors/frameworks/_torch.py:147 -> tensor_factory.py:143`.
  Ranks timed out after 600s with different broadcast sequence numbers. Keep
  distributed shard loading disabled by default and use the slower per-rank
  fastsafetensors path for deterministic debug runs.
- A separate env propagation bug was found: the server process had debug/fallback
  variables such as `AITER_DISABLE_CUSTOM_ALL_REDUCE=1`,
  `ATOM_USE_TORCH_CACHE=1`, `ATOM_DEBUG_TOPK`, and `ATOM_FWD_DUMP_DIR`, but the
  spawned EngineCore/TP runner processes did not inherit them. This caused the
  real workers to take custom all-reduce/debug-disabled paths. The fix is to
  pass an explicit `os.environ` snapshot through `EngineCore.run_engine()` and
  `AsyncIOProc`.

## Next Validation Plan

1. Start the server from the script:

   ```bash
   docker exec -d zxb_kimi bash -lc '
   ATOM_USE_FASTSAFETENSORS=1 \
   ATOM_FASTSAFETENSORS_DIST_LOAD=0 \
   ATOM_FASTSAFETENSORS_NOGDS=1 \
   ATOM_LOADER_USE_THREADPOOL=0 \
   bash /home/carhuang/xiaobing/ATOM-K3/my_script/run_models.sh \
     > /home/carhuang/xiaobing/ATOM-K3/my_script/server.log 2>&1
   '
   ```

2. Confirm health:

   ```bash
   curl -fsS http://127.0.0.1:8000/health
   ```

3. Run deterministic single-request smoke tests:

   ```bash
   curl -sS http://127.0.0.1:8000/v1/completions \
     -H "Content-Type: application/json" \
     -d '{"model":"/data/models/Kimi-K3","prompt":"Hello, my name is","max_tokens":32,"temperature":0}'
   ```

4. If output is still garbled, compare hidden-state stats around the first
   full-attention/MLA layer with:

   - `ATOM_USE_UNIFIED_ATTN=1`
   - `ATOM_FORCE_ATTN_TRITON=1`
   - grouped contiguous threshold toggled between `0`, `512`, and a large value
   - dummy weights vs real weights

## 2026-07-19 08:4x — First real-weight signal (decode-step NaN)

Environment finally loaded real weights on `zxb_kimi` (original loader path:
`ATOM_USE_FASTSAFETENSORS=1` → falls back to `safetensors.safe_open` since
fastsafetensors is not installed; `ATOM_LOADER_USE_THREADPOOL=0`). Load takes
~25 min because this host has only 1TB RAM vs the 1.5TB model, so the NFS page
cache thrashes (cannot fully cache the model). This is a host limitation, not a
loader-code regression — the committed loader change only *adds* fastsafetensors
as an option.

Deterministic request `prompt="The capital of France is", max_tokens=16, temp=0`:

- `topk.log`:
  - token 1: `0:0.048 1698:0.011 98277:0.007 ...` (finite but nearly flat, token 0
    wins with only 0.048 → already wrong / garbage-ish logits)
  - token 2: `0:nan 1:nan ... 9:nan` (ALL NaN)
- Then `ModelRunner3/4 proc died unexpectedly (exitcode=-6)` (SIGABRT) → workers
  down, server must be reloaded.

Per-layer forward dumps (`ATOM_FWD_DUMP_STATS_ONLY=1`, `ONE_SHOT=1`, layers
`0,1,2,3,4,8,16,32,48,64,80,92`, all 4 ranks) captured the FIRST forward
(warmup prefill, shape 128×7168). **All layers `finite_ratio=1.0`** with healthy,
growing stats (std 0.010 → 0.71 → ... → 1.21 at layer 92, min/max within ±18).

Interpretation:
- The 93-layer decoder trunk is FINITE in the prefill/warmup forward.
- NaN appears only on the **decode step** (2nd token), and/or the logits are
  already wrong in prefill (flat, token 0). So the fault is in either:
  (a) final_norm / lm_head (logits), or
  (b) the decode-path attention (KDA/MLA with KV cache) — prefill was clean but
      decode NaNs.
- `finite` != `correct`: prefill logits are flat, so hidden states may be finite
  but numerically wrong vs a reference.

### Decisive next run config (ONE reload, capture decode NaN)
- `ATOM_FWD_DUMP_ONE_SHOT=0` (dump every forward: warmup=call000, prefill=call001,
  decode=call002+), `ATOM_FWD_DUMP_LAYERS=` empty → ALL 93 layers, STATS_ONLY=1.
- Send one deterministic request, then find the first `call*` + layer where
  `finite_ratio < 1.0`. If layer92 decode is finite but sampler logits NaN →
  fault is post-decoder (final_norm/lm_head). If a decoder layer NaNs in decode →
  fault is in that layer's attention/KV path.

## 2026-07-19 09:xx — Applied MI355 reference fixes; isolated remaining bug to KDA

Merged the two reference commits that give correct single-request output on MI355:
- ATOM `e68fa86` (ff-merge onto xiaobing/k3): KimiFullAttention prefill now routes
  through `self.attn` (writes paged KV cache); KimiSparseMoeBlock all-reduces the
  routed expert output BEFORE `routed_expert_norm` in the latent-MoE path.
- aiter `69d82b5a` (cherry-pick --no-commit; kept local `68da9acf` ck re-pin):
  sigmoid `score_mode` in flat MoE topk + `return_layout` in shuffle_scale_moe.
Both repos are editable installs, so a plain server restart picks up the changes.

### Result on gfx1250 (this host, NOT MI355)
The MoE/logits fix clearly WORKED. With a long prompt (221 tokens) the prefill
token is now finite AND sharp: `topk row0: 2021:1.000 ...` (pre-fix it was a flat
`0:0.048`). So core model + long-seq KDA prefill + MoE + lm_head are correct.

Two REMAINING gfx1250-specific NaNs, both in the KDA (gated delta net / FLA) path
(`atom/models/kimi_k3.py` KimiKDAAttention -> `fla.ops.kda` chunk_kda /
fused_recurrent_kda; layers 0,1,2,4,5,6,... are KDA, layer 3,7,11,... are full-attn):

1. **Short-sequence prefill NaN.** 5-token prompt → layer 0 output 100% NaN
   (`finite_ratio=0.0` at EVERY layer, real request shape (5,7168)); 221-token
   prompt → prefill finite. Warmup (128 tokens) is also finite. So `chunk_kda`
   NaNs when the prompt is shorter than the chunk size (~64) on gfx1250.
2. **Decode-step NaN (the real blocker).** Even when prefill is finite (long
   prompt, token1 = 2021:1.000), the FIRST decode token (token2) and all
   subsequent tokens are NaN. Decode uses `causal_conv1d_update` +
   `fused_recurrent_kda` reading conv_state/ssm_state written during prefill.
   So either the KDA state cache write/read or `fused_recurrent_kda` NaNs on
   gfx1250.

### Suggested next steps (pick one; each server reload on this host ≈ 25 min
### because 1TB RAM < 1.5TB model, so keep the server ALIVE and iterate by request)
- Confirm sub-op: reload once with `ATOM_FWD_DUMP_ONE_SHOT=0` +
  `ATOM_FWD_DUMP_BLOCK_CLASS=KimiDecoderLayer,KimiKDAAttention` + layers `0,1,2`
  to see KDA-vs-MLP and prefill-call vs decode-call finite_ratio.
- Try a KDA kernel workaround for gfx1250: force `fused_recurrent_kda` for
  prefill too (avoids chunk_kda short-seq path) and/or a torch reference for the
  gated-delta recurrence; compare against fla decode.
- Check for an additional MI355/gfx1250 reference commit specifically for the
  KDA/FLA gated-delta path.

## 2026-07-19 10:xx — RESOLVED: coherent correct output on gfx1250

Three upstream reference fixes + one local gfx1250 KDA workaround make Kimi-K3
produce correct output on this gfx1250 host:

1. ATOM `e68fa86` (merged): KimiFullAttention prefill writes paged KV cache;
   latent-MoE all-reduce before routed_expert_norm.
2. aiter `69d82b5a` (cherry-pick --no-commit): sigmoid score_mode in flat MoE topk.
3. ATOM `91968c5` (merged): situ (not SiLU) activation in triton MoE experts
   (`fused_moe_triton.py` + `moe.py`). Fixes sporadic wrong tokens / garbage
   insertions after the first token.
4. LOCAL gfx1250 workaround (uncommitted in `atom/models/kimi_k3.py`): env
   `ATOM_KDA_FORCE_RECURRENT=1` forces the KDA (gated delta net) prefill through
   `fused_recurrent_kda` instead of `chunk_kda`. On gfx1250 `chunk_kda` NaNs for
   short prompts (seq < chunk size) AND its `transpose_state_layout` output
   mismatches the decode-time `fused_recurrent_kda` reader (NaN on first decode).
   Forcing the recurrent path keeps prefill/decode state layout consistent.

### Verified outputs (temperature=0, greedy)
- "The capital of France is" -> " Paris."
- "Question: What is 12 + 7? Answer:" -> " 19"
- "Artificial intelligence is" -> " a branch of computer science that focuses on
  creating intelligent machines. ... how to make computers do things that require
  intelligence when done by humans. ..."

Remaining greedy-decode drift/repetition is normal base-model behavior, not NaN.

### Working run recipe (this host)
PYTHONPATH must prepend `/app/triton-mi450/python`, local `aiter-k3`, `ATOM-K3`,
and KEEP `/opt/venv/lib/python3.12/site-packages/_rocm_sdk_core/share/amd_smi`
(dropping it makes torch.cuda.device_count()==0). Do NOT set HIP/ROCR/CUDA
VISIBLE_DEVICES. Loader: `ATOM_USE_FASTSAFETENSORS=1` (falls back to safe_open),
`ATOM_LOADER_USE_THREADPOOL=0`. Plus `ATOM_KDA_FORCE_RECURRENT=1`. HTTP server on
port 8000. Load takes ~25 min (1TB RAM < 1.5TB model); keep the server ALIVE and
iterate by request instead of restarting.

### TODO / follow-up
- The KDA workaround is env-gated and local. Proper fix would be in
  `fla.ops.kda` chunk_kda for gfx1250 (short-seq + state layout), then the fast
  chunk prefill can be re-enabled. Consider reporting upstream.

## 2026-07-19 11:xx — GSM8K run blockers found after increasing max_model_len

Goal: run 30-case GSM8K with `max_model_len=8192` so 3-shot chat prompts do not
hit the old 1024-token limit.

### Issue A: ABORTED + deferred-output scheduler crash

First GSM8K attempt used the old debug server config:

- `MAX_MODEL_LEN=1024`
- `MAX_NUM_SEQS=1`
- `MAX_NUM_BATCHED_TOKENS=128`

One GSM8K 3-shot request had ~853 input tokens and default 256 output tokens, so
the API returned 400 (`853 + 256 > 1024`). `lm_eval` then disconnected the other
concurrent requests. EngineCore later crashed in:

- `atom/model_engine/scheduler.py`
- `seq.output_tokens[-num_placeholder - offset + i] = el`
- `IndexError: list assignment index out of range`

Root cause: in deferred-output mode, the scheduler appends an EOS placeholder at
the end of a RUNNING seq, then overwrites it with the delayed real token on the
next step. A client-disconnected seq is marked `ABORTED`; an aborted partial
prefill may have no `output_tokens` placeholder, but the delayed-output path still
tried to overwrite `output_tokens[-1]`.

Local fix applied:

- In `Scheduler.postprocess`, if a running seq is already `ABORTED`, mark it
  `FINISHED` with `leave_reason="aborted"` and skip delayed token backfill.
- Added regression test:
  `tests/test_scheduler.py::TestPostprocess::test_aborted_deferred_output_finishes_without_placeholder`
- Verified in container:
  `/opt/venv/bin/python -m pytest tests/test_scheduler.py -q` -> `53 passed`.

TODO: keep this fix; clean it up if needed and commit/upstream it.

### Issue B: large warmup crashes on gfx1250 when max_num_batched_tokens=7168

Next restart used:

- `MAX_MODEL_LEN=8192`
- `MAX_NUM_SEQS=16`
- `MAX_NUM_BATCHED_TOKENS=7168`

This fixed the static context limit, but warmup used one dummy prefill sequence of
7168 tokens:

`Model Runner*/4: warmup_max_tokens=7168 < max_model_len=8192. Using 1 seq with length 7168 for warmup.`

After all 96 shards loaded, ModelRunner crashed during warmup:

- `ModelRunner1/4 proc died unexpectedly (exitcode=-6)`
- `CUDA error: unspecified launch failure`
- stack included `torch.multinomial`

Likely trigger: very large warmup/dummy prefill on gfx1250 exercises large
GEMM/MoE/sampler path that is not stable. This is separate from the earlier
precision/KDA fixes.

Current workaround for continuing GSM8K:

- Keep `MAX_MODEL_LEN=8192`
- Lower `MAX_NUM_BATCHED_TOKENS` aggressively. `1024` still crashed in warmup;
  the old known-good debug run used `128`, so the current retry uses
  `MAX_NUM_BATCHED_TOKENS=128`.
- Keep chunked prefill enabled so long prompts are processed in chunks
- Keep `ATOM_KDA_FORCE_RECURRENT=1`

Parallel fixes added without disturbing the current loading service:

- `ATOM_WARMUP_MAX_TOKENS`: caps only dummy warmup token count while preserving
  the runtime scheduler's `max_num_batched_tokens`. This should let a future
  restart use `MAX_NUM_BATCHED_TOKENS=1024` or higher with
  `ATOM_WARMUP_MAX_TOKENS=128`.
- `AITER_USE_TORCH_RMSNORM`: routes non-quant RMSNorm and add+RMSNorm through
  a PyTorch fallback before either opus or `module_rmsnorm_quant`. This is for
  isolating the `aiter::rmsnorm_opus_kernel` memory fault observed during
  1024-token warmup.

Validation:

- `pytest -q tests/test_envs.py` passed.
- `python -m py_compile` passed for the modified Python files.
- CPU smoke test with `AITER_USE_TORCH_RMSNORM=1` passed for RMSNorm and
  add+RMSNorm, and no longer tried to JIT-build `module_rmsnorm_quant`.
- Current `MAX_MODEL_LEN=8192`, `MAX_NUM_BATCHED_TOKENS=128` service reached
  ready and completed 128-token warmup.
- `/v1/completions` GSM8K limit=30, 3-shot, `NUM_CONCURRENT=4` completed
  without crashing the service:
  - flexible-extract exact_match: `0.3333 +/- 0.0875`
  - strict-match exact_match: `0.2667 +/- 0.0821`

Chat endpoint note: current running service was started before the Kimi token
postprocessor fix, so `/v1/chat/completions` still returns raw
`<|close|>think<|sep|><|open|>response<|sep|>` markers. A parser fix was added
for the next restart; until then, prefer `/v1/completions` for lm_eval.

Follow-up observation: `MAX_NUM_BATCHED_TOKENS=1024` also fails during warmup.
The first device fault in the log is not `torch.multinomial`; it is:

`Memory Fault Error ... kernel: aiter::rmsnorm_opus_kernel<...>`

The Python stack then reports illegal address while launching later MXFP4
quant/GEMM work (`dynamic_mxfp4_quant` / `gemm_a4w4_quant`) because HIP reports
kernel failures asynchronously. So the large-warmup bug likely starts in the
Opus RMSNorm path or earlier memory corruption, not necessarily in sampling.

TODO: properly fix the large warmup path. Options:

- Cap `warmup_model()` dummy sequence length independently of
  `max_num_batched_tokens` on gfx1250.
- Avoid sampling during dummy warmup, or force greedy/no-sampler in dummy runs.
- Bisect whether crash starts in Opus RMSNorm, MXFP4 quant/GEMM, or another
  earlier kernel whose error only surfaces later.

