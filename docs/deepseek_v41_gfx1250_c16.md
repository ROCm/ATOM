# DeepSeek-V4.1-Flash: gfx1250 TP4 C16 recipe

This is the C16 agentic recipe used to compare PR #3176 with native ATOM.
The reference is the E13 recipe in `/home/yifehuan/ws/e13-02-c16-dsv41/`,
recorded as `1110T0844_c16_pr3109`. This document records deployment and
reproduction requirements; the new standard performance comparison is still
running. It makes no throughput or quality claim for the current PR.

## Configuration identity

| Item | Standard C16 setting |
|---|---|
| Checkpoint | `DeepSeek-V4.1-Flash-5254448103b6` |
| `config.json` SHA256 | `8be45ce0476004a3f529fd896115a4a2e800a129ad2d3ec05b16050f52e21879` |
| Parallelism | TP4, without expert parallelism or DP attention |
| Client concurrency / server capacity | 16 / `max-num-seqs=64` |
| Speculation | DSpark, 5 speculative tokens; probabilistic draft and block verify |
| KV / index cache | FP8 / FP4 |
| Prefill budget / chunk | 16384 / 16384 tokens |
| Cache block size / compilation level | 16 / 0 |
| Decoder request graph sizes | `1,2,3,4,5,6,7,8,12,16` |
| Capture sizes | `1,2,3,4,5,6,7,8,12,16,24,32,40,48,56,64` |
| Sampling | temperature 1, top-p 0.95, top-k 0, maximum output 8192 |
| Chat template | `thinking=true`, `reasoning_effort=max` |
| Load | 600 seconds issuing, 1226 trajectories, salt and tool delays enabled |
| Dataset SHA256 | `1beb24c882122df96571cf11b390acbea388944038bc55c78b891475459014ae` |
| Inline accuracy threshold | 52.36% |

C16 is the client concurrency, not the tensor-parallel size. A full C16
DSpark verify has 96 target rows and 80 draft rows. Do not substitute a
256-output-token microbenchmark or a 1024-token prefill budget for this
standard request measurement.

## Image and dependencies

The original E13 image is
`rocm/fw-bringup-mlperf:gfx1250-atom-20261109-dsv41-clean`, with image ID
`sha256:672b2c0a53ee3c0e3acbf07a21df2768de4fbc94900246789fb0ba0be734ca64`.
It was unavailable on the selected GPU host, and its registry manifest could
not be retrieved with the available credentials.

The comparison uses the local private image `codex-c16-e13-server-20261112`:

```text
sha256:1a83bb3271f354154e5eb7633b6cd0411ee885954ad5ec73cb345040a5ac5d34
```

It was built from the ordinary `gfx1250-atom-20261109-dsv41` image, restoring
the E13 AITER tracked source through a verified Git bundle and binary patch.
It is a reconstruction of the dependencies, not a binary copy of the clean
image. The tag is local to the test host; it has not been published to a
registry. Export/import it with `docker save` / `docker load` when another
host needs this exact image.

| Dependency | Recorded comparison state |
|---|---|
| ATOM implementation | PR #3176 `d830a37db9717ecc193689bb3b5c7790f484ed26`; checkout mounted read-only |
| AITER | `b7b8d2a37738a8f1c7f796f63808c241ede9011d`, clean tracked tree, branch `hyi1109-clean` |
| MORI | `c1f2f646cc82a9b98cb83abf72c55418e7c33ce3`, original E13 modifications retained |
| MLCommons endpoints | `d3c57f2eba127c819e5a8fd83101d9e5f4e15fce`, original four client modifications retained |
| Torch | `2.11.0+rocm10.1.0a20260811.bkc.20260820` |
| Transformers in this image | `5.16.0.dev0` |
| New server API dependencies | `jsonschema==4.25.1`, `jsonschema-specifications==2025.9.1`, `referencing==0.36.2`, `rpds-py==0.27.1` |

The new PR imports JSON Schema support when loading the serving tool parsers;
the older E13 image does not contain it. The additional image layer is:

```dockerfile
FROM codex-c16-e13-20261112
RUN python3 -m pip install \
    --target /usr/local/lib/python3.12/dist-packages --no-deps \
    jsonschema==4.25.1 jsonschema-specifications==2025.9.1 \
    referencing==0.36.2 rpds-py==0.27.1
```

Here `codex-c16-e13-20261112` is the reconstructed E13 dependency image.
The original clean image, if available, can instead be used as the base for
this API dependency layer, with a separately recorded resulting image ID.
The layer preserves Torch, ROCm, AITER, MORI and the endpoints client.
MORI's two modified files are `python/mori/ops/dispatch_combine.py` and
`src/ops/dispatch_combine/intranode_1250x.hpp`; its commit alone does not
identify those changes. The endpoints modifications are
`token_metrics.py`, `execute.py`, `watchdog.py` and
`agentic_inference_strategy.py`. Preserve the image's versions rather than
replacing them with clean upstream checkouts.

## Host and container requirements

- Four currently available gfx1250 GPUs with a compatible BKC driver/runtime;
  identify them by UUID and their actual render nodes before launch.
- Model and dataset mounted read-only at `/data/DeepSeek-V4.1-Flash` and
  `/dataset`, respectively. Keep the checkpoint's tokenizer and chat template.
- Private `/dev/shm` of at least 64 GiB and unlimited container memlock.
  Engram page-locking uses about 45.8 GiB of host memory per TP rank on this
  checkpoint; provide host-memory headroom for all four ranks and loading.
- The tested host already has
  `/sys/module/amdgpu/parameters/no_system_mem_limit=Y`. This is a prerequisite
  recorded by the launcher, not an instruction to change a shared host.
- External writable cache and result mounts; keep the implementation checkout
  read-only. Preserve the image's default HIP library for performance, including
  its `HSA_ENABLE_SCRATCH_ASYNC_RECLAIM=1` TP4 setting.

The current harness uses a private IPC namespace and `--network=none`, with
server and client in the same container communicating over loopback. A
container template, after selecting four idle UUIDs and their render nodes:

```bash
docker run --name c16-bench --network none --ipc private --shm-size 64g \
  --ulimit core=0 --ulimit memlock=-1 --device /dev/kfd \
  "${C16_RENDER_ARGS[@]}" \
  -e ROCR_VISIBLE_DEVICES="${C16_GPU_UUIDS:?four selected physical UUIDs}" \
  -e HIP_VISIBLE_DEVICES=0,1,2,3 -e CUDA_VISIBLE_DEVICES=0,1,2,3 \
  -v "${C16_ATOM:?checkout}:/app/ATOM:ro" \
  -v "${C16_MODEL:?checkpoint}:/data/DeepSeek-V4.1-Flash:ro" \
  -v "${C16_DATASET_ROOT:?dataset root}:/dataset:ro" \
  -v "${C16_RESULTS:?external results}:/results" \
  -v "${C16_CACHE:?external cache}:/image-cache" \
  --entrypoint bash codex-c16-e13-server-20261112 -lc 'sleep infinity'
```

`C16_RENDER_ARGS` is an array of `--device /dev/dri/renderD...` pairs for
the selected GPUs, determined on that host. Enter the container with
`docker exec`; this template only keeps the container available. It does not
start the service or client by itself.

## Server recipe

Inside the container, retain the E13 environment:

```bash
export AITER_CUSTOM_AR_MIN_SIZE=131072 AITER_CUSTOM_AR_MAX_SIZE=8388608
export AITER_RUNTIME_GPU_ARCH=gfx1250 AITER_USE_SYSTEM_TRITON=1
export FLYDSL_GPU_ARCH=gfx1250 MORI_GPU_ARCHS=gfx1250
export HSA_OVERRIDE_GFX_VERSION=12.5.0 HSA_XNACK=1
export HSA_DISABLE_COREDUMP_ON_EXCEPTION=1 HSA_ENABLE_SCRATCH_ASYNC_RECLAIM=1
export HIP_PATH=/usr/local/lib/python3.12/dist-packages/_rocm_sdk_core
export HIP_DEVICE_LIB_PATH="$HIP_PATH/lib/llvm/amdgcn/bitcode"
export ATOM_DSPARK_PROB_DRAFT=1 ATOM_DSPARK_BLOCK_VERIFY=1
export ATOM_ENGRAM_MLOCK=1 ATOM_GC_THRESHOLD=20000,50,50
export ATOM_MEGA_MASK_PAD_ROWS=1 ATOM_MLA_QH128_ASM_PAGE_SIZE=64
export ATOM_DECODER_GRAPH_SIZES=1,2,3,4,5,6,7,8,12,16
export ATOM_MONO_ENABLE=1  # PR candidate; set 0 for the same-head native arm

python3 -m atom.entrypoints.openai_server \
  --model /data/DeepSeek-V4.1-Flash --trust-remote-code \
  --host 0.0.0.0 --server-port 8000 -tp 4 \
  --kv_cache_dtype fp8 --index-cache-dtype fp4 \
  --max-num-seqs 64 --max-num-batched-tokens 16384 \
  --attn-prefill-chunk-size 16384 --block-size 16 --level 0 \
  --method dspark --num-speculative-tokens 5 \
  --cudagraph-capture-sizes '[1,2,3,4,5,6,7,8,12,16,24,32,40,48,56,64]' \
  --disable-uvicorn-access-log
```

When using the E13 `run-c16.sh` wrapper, use `MODE=native` or `MODE=mono`.
Its `serve_tp4.sh` sets C16's default mono switch to 0 before applying
`EXTRA_ENV`. Set the final override through that hook; merely exporting
`ATOM_MONO_ENABLE=1` before the shell script is insufficient:

```bash
export EXTRA_ENV='ATOM_MONO_ENABLE=1 HSA_ENABLE_SCRATCH_ASYNC_RECLAIM=1'
# Equivalent E13 wrapper selection:
MODE=mono bash /path/to/e13-02-c16-dsv41/run-c16.sh /results/NEW_C16
```

The comparison's external `server_entry.py` installs a preparation-time
occupancy check before invoking the same server module. Before timed client
work it requires actual target/draft ELF capacities of at least 256 CTAs on
each rank, including the C16 build. It does not instrument steady-state decode.

## Client configuration and reporting

Use the E13 `client/c16.yaml`. Its rendered configuration must remain identical
except for `report_dir`:

```yaml
name: deepseek-v4.1-flash-agentic-benchmark
version: '1.0'
type: online
submission_ref:
  model: deepseek-ai/DeepSeek-V4.1-Flash
  ruleset: mlperf-endpoints-v1.0-2026-10-C1-A
model_params:
  name: /data/DeepSeek-V4.1-Flash
  tokenizer_name: /data/DeepSeek-V4.1-Flash
  temperature: 1.0
  top_k: 0
  top_p: 0.95
  max_new_tokens: 8192
  streaming: 'on'
  chat_template_kwargs:
    thinking: true
    reasoning_effort: max
datasets:
- name: agentic_combined
  type: performance
  path: /dataset/mlperf_agentic_inference_dataset/mlperf_agentic_inference_dataset.jsonl
  accuracy_config:
    eval_method: agentic_inference_inline
    num_repeats: 1
  agentic_inference:
    turn_timeout_s: 86400.0
    enable_salt: true
    inject_tool_delay: true
    num_trajectories_to_issue: 1226
    stop_issuing_on_first_user_complete: true
settings:
  load_pattern:
    type: agentic_inference
    target_concurrency: 16
  client:
    warmup_connections: 0
    max_idle_time: 0.5
  runtime:
    max_issue_duration_ms: 600000
  timeouts:
    run_timeout_s: 14400
endpoint_config:
  endpoints: [http://127.0.0.1:8000]
  api_type: openai
report_dir: /results/c16/report
```

The preserved E13 runner starts a fresh server, waits for readiness, sends its
smoke/warmup request, resets the prefix cache, and invokes
`/app/mlcommons-endpoints/venv/bin/inference-endpoint benchmark from-config
--config CLIENT_YAML --mode perf`. After 600 seconds it waits for the client
to finalize rather than signaling it. Report full-client TPS, TPS/4, E2E
interactivity, TPOT and TTFT p50/p90, completed/failed/dropped requests and
inline accuracy. Require `state=complete`, `complete=true`, client exit 0,
and zero failed/dropped requests. Keep first batches, warmup logs and tails.

Quality is accepted through the matched full-network inline metric. Local
floating-point, quantization, MID or generated-token differences are diagnostic;
addresses, integer metadata, epochs, guards, synchronization and finite outputs
remain strict. Do not require generated tokens to match between the arms.

## Exact comparison with E13

The actual server argument list matches E13, with an external preparation
hook as its entry point. Every environment item recorded by E13's
`server_env.txt` matches, except the deliberate candidate mono override.
The actual client YAML was also checked against E13, excluding `report_dir`.
This is recipe equivalence; the full image and execution environment differ:

| Additional comparison setting | Purpose / scope |
|---|---|
| `ATOM_DISABLE_MMAP=true` | Checkpoint loading policy; common to the three arms |
| `ATOM_MONO_TIMELINE=''`, `ATOM_MONO_SINGLE_TIMING=0` | Disable mono diagnostic timing |
| `ATOM_MONO_C16_EY_ROLE=0`, `ATOM_MONO_C16_DE_ROLE=0` | Disable old experimental front helpers |
| `ATOM_MONO_C16_MOE_ROLE=0` for PR/native, `1` only for old CTA-ready | Select the old benchmark branch's helper; not a PR dynamic-writer transplant |
| `ATOM_MONO_SINGLE_SERVE=1` | Retained legacy harness setting; the current PR routes through `ATOM_MONO_ENABLE` |
| `ROCR_VISIBLE_DEVICES`, logical HIP/CUDA device lists | Restrict the task to the four selected physical GPUs |
| `HF_HUB_OFFLINE=1`, `PYTHONDONTWRITEBYTECODE=1` | Offline model/client operation and a read-only source checkout |
| AITER/FlyDSL/Triton/Torch/XDG cache directories | External, task-owned build and runtime caches |
| `OMP_NUM_THREADS=4`, `OPENBLAS_NUM_THREADS=4` | Common CPU threading settings; not captured in E13's filtered environment record |
| `MAX_JOBS=16` in the server, `AITER_COMPILE_THREADS=4`, `AITER_LOG_LEVEL=WARNING` | Common compilation and logging controls |
| Private IPC, 64-GiB shared memory, loopback-only networking | Task isolation with a colocated client |

The old CTA-ready branch uses a fixed producer-owner map. Current PR MoE
uses a dynamic UG queue, so the old dependency map cannot be copied into it.
The complete implementations also differ outside MoE; their standard results
cannot isolate the cost of dynamic scheduling alone. All three standard arms
use the same reconstructed image, host and client configuration. Historical
E13 throughput and old local-MoE timings are not substituted for this comparison.
