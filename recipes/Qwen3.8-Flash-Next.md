# Qwen3.8-Flash-Next Usage Guide

[Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) is a
multimodal MoE model combining Gated DeltaNet (GDN), sparse attention (QSA),
and n-gram memory. ATOM supports text and image inference with BF16, FP8, or MXFP4
weights, BF16 activations, and a BF16 or FP8 KV cache.

## Preparing the environment

Follow the [ATOM installation instructions](../README.md). Use the
dependencies pinned by ATOM, including `transformers==5.16.1`, which provides
the model configuration and multimodal processor.

## Serving FP8 on one GPU

```bash
python -m atom.entrypoints.openai_server \
  --model Qwen/Qwen3.8-Flash-Next-FP8 \
  --trust-remote-code \
  -tp 1
```

The API listens on port 8000 by default. Use
`Qwen/Qwen3.8-Flash-Next-FP8` as the `model` in API requests.

The KV cache is BF16 by default. The default block size already satisfies
`indexer_compress_ratio` (4); no block-size override is needed.

## FP8 KV cache

```bash
python -m atom.entrypoints.openai_server \
  --model Qwen/Qwen3.8-Flash-Next-FP8 \
  --trust-remote-code \
  -tp 1 \
  --kv_cache_dtype fp8 --index-cache-dtype bf16 --block-size 64
```

`--kv_cache_dtype fp8` stores the QSA K/V as E4M3 with one FP32 scale per
token and KV head, which fits about 1.65x the tokens of a BF16 cache. The
compressed index keys follow the KV dtype unless `--index-cache-dtype` is
given: `bf16`, `fp8` (one FP32 scale a row) or `fp4` (MXFP4). The raw index
keys stay BF16. With `--block-size 64` and a BF16 index cache, prefill index
scoring runs on the FlyDSL kernel, which takes BF16 keys only; this is the
recommended combination. FP4 index keys work but score slower than FP8 or
BF16 on MI355X. The MTP draft layer keeps BF16 caches unless
`ATOM_QSA_DRAFT_KV_BF16=0`.

Choose a GPU with enough memory for the weights, recurrent state, and KV
cache. Adjust `--max-num-seqs`, `--max-model-len`, and
`--max-num-batched-tokens` for the available memory and workload.
In particular, `--max-num-seqs` defaults to 512; lowering it reduces the
preallocated recurrent-state memory. These are resource controls, not
model-specific requirements.

## Native MTP

Use the draft weights included in the same checkpoint:

```bash
python -m atom.entrypoints.openai_server \
  --model Qwen/Qwen3.8-Flash-Next-FP8 \
  --trust-remote-code \
  -tp 1 \
  --method mtp --num-speculative-tokens 2 \
  --no-enable_prefix_caching
```

Draft depths 1, 2, and 3 are supported. The checkpoint contains one draft
layer, reused at each step; no separate draft model is needed. The target
and draft share the embedding and output head.

## Agentic serving on one MI350X (prefix caching + MTP)

For agent workloads (e.g. Claude Code through `/v1/messages`: long shared prefixes, many short turns, several concurrent sub-agents),
**prefix caching is what makes turns cheap** and can be combined with native MTP. Measured on 1× MI350X (VF), FP8, TP1, 1M context,
agent turn = 100k cached tokens + 4k new tokens, 256 output tokens:

| | prefix caching off | prefix caching on |
|---|---|---|
| 1 agent, time to first token | 6.2 s | **0.53–0.58 s** |
| 6 agents, time to first token (median) | 31 s | **~1.9 s** |

Numerical check with MTP + prefix caching: 12/12 identical answers (5 markers spread over 58k tokens; cold, exact repeat and
new-question turns).

```bash
python -m atom.entrypoints.openai_server \
  --model Qwen/Qwen3.8-Flash-Next-FP8 --trust-remote-code -tp 1 \
  --method mtp --num-speculative-tokens 3 \
  --state-checkpoint-interval-tokens 16384 \
  --cudagraph-capture-sizes "[1,2,3,4,5,6,8,16]" \
  --max-num-seqs 16 --kv_cache_dtype bf16
```

- **`--state-checkpoint-interval-tokens`**: a GDN state checkpoint must exist at the resume point. `-1` (prompt-end anchors only) serves
  conversations that only grow at the end, but sibling sub-agents that share a long prefix and diverge in the middle then recompute
  everything. With `16384`, a real Claude Code session (auto mode + sub-agents) reached **94% prefix-cache hits**; the default `8192`
  costs ~5–8% prefill throughput for the extra rungs.
- **`--cudagraph-capture-sizes`**: capture every batch size you expect (the default list skips 3, 5, 6).
- **Persist the Triton/comgr caches** (`~/.triton`, `~/.cache`) across container restarts and send a warm-up request after start:
  first-time JIT compilation of a new shape stalls a request for 5–25 s.
- MTP depth: 3 was best for 1–3 concurrent agents, 2 for 4–5 (measured 240 / 545 / 745 tok/s aggregate decode at 1 / 3 / 6 agents).
- Clients that send `role: system` messages in the middle of `messages` (Claude Code 2.1.x) get HTTP 500 ("System message must be at the
  beginning"); converting them in place (e.g. into a `<system-reminder>` block of the preceding user turn) keeps the prefix stable —
  moving them to the top-level `system` field invalidates the cache on every new reminder.
- Concurrent requests with an identical prefix do not share it until the first one finishes prefill; serializing them at the proxy
  (release on first token) raised hits from 77% to 88% in the same session.

## Usage notes and limitations

- For image requests, use `--no-enable_prefix_caching` and sufficient cache
  capacity to avoid preemption until the image caching/recompute limitations
  are resolved. Image prompts must fit within `--max-num-batched-tokens`.
- Preserve the checkpoint's quantization exclusions and `ple_embedding_dtype`
  when present. GDN inputs support unquantized weights or per-channel FP8 QKV/Z
  with unquantized B/A. GDN input online quantization supports only matching
  `ptpc_fp8` targets on QKV/Z; B/A retain their source precision.
- PTPC (`compressed-tensors`) checkpoint loading is supported, but end-to-end
  execution on gfx950 is blocked by the default small-batch MoE dispatch:
  its second-stage kernel requires a 256-aligned intermediate size, whereas
  this model uses 640. The FP8 example above uses block-wise quantization.
- Chunked text prefill and full decode CUDA graphs are supported. Piecewise
  compilation/CUDA graphs are not supported.
- Only native MTP speculation is supported. Pipeline parallelism, context
  parallelism, DP attention, and TBO are not supported.
- LMCache offload (`lmcache_offload`, `lmcache_mp`) stores the QSA pages plus
  a checkpoint of the GDN and PLE state.
  With MTP, a decode-step checkpoint is stored as the image of its verify
  step.
  P/D connectors and `mooncake_store` are rejected: they do not carry the
  required state image.
- This example covers FP8 with TP1. Pure TP2 is not currently validated;
  it can fail MoE warmup on gfx950 due to an MoE dispatch limitation.
- Video inference and long-context accuracy have not been validated.

## GSM8K evaluation

Install the evaluation client:

```bash
python -m pip install 'lm-eval[api]'
```

With the server running, evaluate all 1,319 test questions using 5-shot chat:

```bash
lm_eval --model local-chat-completions \
  --model_args 'model=Qwen/Qwen3.8-Flash-Next-FP8,base_url=http://localhost:8000/v1/chat/completions,num_concurrent=64,max_retries=3,timeout=900,tokenized_requests=False,trust_remote_code=True' \
  --tasks gsm8k --num_fewshot 5 \
  --apply_chat_template --fewshot_as_multiturn \
  --gen_kwargs 'max_gen_toks=4096,until=<|im_end|>' \
  --log_samples --output_path ./results/gsm8k
```

The model emits reasoning before its final answer, so allow enough output
tokens to avoid truncation. The explicit stop string replaces GSM8K's default
`Question:` stop, which can prematurely end a reasoning response.
The harness scores the final `content`, not `reasoning_content`.
The timeout accommodates long requests under load; it does not change the
generation limit or scoring. Set evaluation concurrency to suit the server's
capacity.

## AgentX c1-c64 performance replay on MI355X

This text-only TP1 recipe uses the
[AMD MXFP4 checkpoint](https://huggingface.co/amd/Qwen3.8-Flash-Next-Quark-MXFP4),
FP8 KV, BF16 index keys, prefix caching, and three-token MTP. It covers the
historical curve's eight concurrency points: 1, 2, 4, 8, 16, 32, 48, and 64.
The Mono split below follows the November 20, 2026 full-curve preparation
recipe. Only c8 and c64 were measured with this setup on ATOM
`40a4d6d57bb16600c84ac758fff00e3379f133fa` (a pre-merge #2636 branch commit,
not on main); the other six points are prepared configurations. The commands
below instead require a checkout containing
[#2636](https://github.com/ROCm/ATOM/pull/2636); no point has been performance
tested on current main here.
`--spec-decode-acceptance-length 2.32` forces synthetic acceptance for
performance testing. Remove it for real generation or accuracy evaluation.

### Environment and assets

Use one MI355X and run the selected points sequentially, starting a fresh
server for each. For c32, c48, or c64, reserve a 208.6 GiB host cache pool
plus memory for weights/loading, the server, and the client. Its 104.3 GiB state allowance is
carved from that pool, not added to it. The 32 GiB container shared-memory
limit is separate from the host cache budget.

The public base image below supplies ROCm 7.2.4, PyTorch 2.10.0,
Transformers 5.16.1 for the server, and LMCache
`0.5.6.dev139+gf22dec28.rocm7.2.4.torch2.10.git3d3aa833.cxx11abi1`.
Use `rocm-smi --showbus --showuniqueid` to select the GPU. Set `GPU_UUID` to
`GPU-` followed by its Unique ID without `0x`; select the render node whose
`/sys/class/drm/renderD*/device` resolves to the same PCI bus address. Reuse an existing checkout by skipping `git clone` and
checking the same ancestor. Record the actual checkout SHA with the results;
use an unused output directory for each replay.

```bash
set -euo pipefail
export ATOM_DIR=/path/to/ATOM
export ASSET_DIR=/path/to/agentx-assets
export RESULT_DIR=/path/to/agentx-results
export GPU_UUID=GPU-REPLACE_ME
export RENDER_DEVICE=/dev/dri/renderD128  # Match GPU_UUID on this host.
export IMAGE=rocm/atom-dev@sha256:32c81e56e4680065b6ea3e0ca8363a00cd3ccb932d8990e83d799b8bbb58a4b3

git clone https://github.com/ROCm/ATOM.git "$ATOM_DIR"
git -C "$ATOM_DIR" merge-base --is-ancestor \
  811b02bb41774591b24d44c0bdff2f0147fc602f HEAD
mkdir -p "$ASSET_DIR" "$RESULT_DIR"
git -C "$ATOM_DIR" rev-parse HEAD | tee "$RESULT_DIR/atom-commit.txt"
docker pull "$IMAGE"
docker run --rm -d --name qwen38-agentx \
  --network host --shm-size 32g --ulimit memlock=-1:-1 --ulimit core=0 \
  --cap-add SYS_NICE --device /dev/kfd --device "$RENDER_DEVICE" \
  --group-add video --group-add render \
  -e ROCR_VISIBLE_DEVICES="$GPU_UUID" -e HIP_VISIBLE_DEVICES=0 \
  -e PYTHONPATH=/workspace/ATOM -e PYTHONUNBUFFERED=1 \
  -e ATOM_ONLINE_QUANT_INFO_DIR=/results \
  -e NO_PROXY=localhost,127.0.0.1 -e no_proxy=localhost,127.0.0.1 \
  -v "$ATOM_DIR:/workspace/ATOM:ro" -v "$ASSET_DIR:/assets" \
  -v "$RESULT_DIR:/results" -w /workspace/ATOM \
  --entrypoint /bin/sleep "$IMAGE" infinity
docker exec -it qwen38-agentx /bin/bash
```

In the container, install the measured AITER and FlyDSL versions and the
additional server dependencies. The base leaves an empty FlyDSL 0.4 metadata
directory after upgrading; archive only that known `REQUESTED`-only remnant
before checking the installed version. Keep the client in its own environment:
the `aiperf` executable used here belongs to
[`agentx-harness`](https://github.com/SemiAnalysisAI/agentx-harness).

```bash
set -euo pipefail
/opt/venv/bin/python -m pip install --no-deps \
  'https://rocm.frameworks-nightlies.amd.com/whl-staging/gfx942-gfx950/amd_aiter-0.1.1.dev1%2Bgd17ee8dfe-cp312-cp312-linux_x86_64.whl' \
  'flydsl==0.5.0'
/opt/venv/bin/python - <<'PY_METADATA'
from pathlib import Path
import shutil
import sysconfig
import tempfile
orphan = Path(sysconfig.get_path('purelib')) / 'flydsl-0.4.0.dist-info'
if orphan.exists():
    assert [p.name for p in orphan.iterdir()] == ['REQUESTED'], str(orphan)
    archive = Path(tempfile.mkdtemp(prefix='flydsl-metadata-'))
    shutil.move(str(orphan), str(archive / orphan.name))
PY_METADATA
/opt/venv/bin/python -m pip install \
  'jsonschema==4.26.0' 'llguidance==1.9.1' 'google-re2==1.1.20251105'
/opt/venv/bin/python -m venv /opt/agentx
/opt/agentx/bin/python -m pip install \
  'agentx-harness==1.0.9' 'datasets==5.1.0' 'transformers==5.19.0'
/opt/venv/bin/python - <<'PY'
from importlib.metadata import version
from pathlib import Path
import atom
import flydsl
import torch
assert torch.cuda.device_count() == 1
assert torch.cuda.get_device_properties(0).gcnArchName.split(':')[0] == 'gfx950'
assert Path(atom.__file__).resolve().is_relative_to('/workspace/ATOM')
assert version('flydsl') == flydsl.__version__ == '0.5.0'
assert version('transformers') == '5.16.1'
print('ATOM:', atom.__file__, 'FlyDSL:', version('flydsl'))
PY
```

Use the same unmodified model directory for server weights and the client
tokenizer. Prepare the model and dataset cache before starting the server;
the timed replay runs with `HF_DATASETS_OFFLINE=1`. Replaying archived
assistant responses is the pinned client's fixed behavior. The model revision is
`b2091c3809eb51902f46a163973362d8176f9e58`.

Use a dedicated dataset cache so the client's unversioned public-dataset
loader resolves to this revision when offline. Existing complete model
weights may be mounted at `MODEL_PATH` instead of downloaded again.

```bash
export MODEL_PATH=/assets/model HF_HOME=/assets/hf-cache
unset HF_HUB_OFFLINE HF_DATASETS_OFFLINE
/opt/agentx/bin/python - <<'PY_ASSETS'
import os
from datasets import load_dataset
from huggingface_hub import snapshot_download
snapshot_download(
    'amd/Qwen3.8-Flash-Next-Quark-MXFP4',
    revision='b2091c3809eb51902f46a163973362d8176f9e58',
    local_dir=os.environ['MODEL_PATH'],
)
ds = load_dataset(
    'semianalysisai/cc-traces-weka-062126-256k',
    revision='8fecd2fc56694469f758f0afbbb6335ad3043740', split='train',
)
assert len(ds) == 393 and ds._fingerprint == '0d8fdac271289f80'
PY_ASSETS
HF_DATASETS_OFFLINE=1 /opt/agentx/bin/python - <<'PY_CACHE'
from datasets import load_dataset
ds = load_dataset('semianalysisai/cc-traces-weka-062126-256k', split='train')
assert len(ds) == 393 and ds._fingerprint == '0d8fdac271289f80'
assert all('8fecd2fc56694469f758f0afbbb6335ad3043740' in f['filename']
           for f in ds.cache_files)
PY_CACHE
```

### Server configuration

All points use the same server command, with these differences:

| Concurrency (`CONC`) | `ATOM_MONO_ENABLE` | `--max-num-seqs` | CPU offload | CPU pool (GiB) | State cap within pool (GiB) |
|---|---|---|---|---|---|
| 1, 2, 4, 8, 16 | 1 | 160 | None | — | — |
| 32, 48, 64 | 0 | 64 | `lmcache_offload` | 208.6 | 104.3 |

The c1-c16 configuration uses the same `--max-num-seqs 160` and online
quantization settings as the [Mono deployment guide](Qwen3.8-Flash-Next-Mono-Program.md).
See that guide for Mono's log lines and deployment refusal rules.
`ATOM_MONO_ENABLE=1` below explicitly restates the default for this deployment.

Set the desired concurrency in the container, for example `export CONC=16`,
then run the block below; it defaults to 8 when `CONC` is unset or empty. After
the client finishes, stop that server with Ctrl-C before selecting the next
point and starting a fresh server. If the shell exits, re-enter with
`docker exec` above. The three ports may be changed together if occupied.

```bash
set -euo pipefail
export CONC="${CONC:-8}"
export MODEL_PATH=/assets/model
export MODEL_ID=amd/Qwen3.8-Flash-Next-Quark-MXFP4
export HTTP_PORT=8000 ENGINE_PORT=29500 DP_PORT=29501
export AITER_LOG_LEVEL=WARNING
unset LMCACHE_LOCAL_CPU LMCACHE_MAX_LOCAL_CPU_SIZE OFFLOAD_STATE_CPU_SIZE
unset LMCACHE_CHUNK_SIZE OFFLOAD_MIN_SAVE_TOKENS OFFLOAD_MIN_LOAD_TOKENS
extra_args=()
case "$CONC" in
  1|2|4|8|16)
    export ATOM_MONO_ENABLE=1
    max_num_seqs=160
    ;;
  32|48|64)
    export ATOM_MONO_ENABLE=0
    max_num_seqs=64
    export LMCACHE_LOCAL_CPU=True LMCACHE_MAX_LOCAL_CPU_SIZE=208.6
    export OFFLOAD_STATE_CPU_SIZE=104.3 LMCACHE_CHUNK_SIZE=256
    export OFFLOAD_MIN_SAVE_TOKENS=256 OFFLOAD_MIN_LOAD_TOKENS=256
    extra_args=(--kv-transfer-config '{"kv_connector":"lmcache_offload","kv_role":"offload"}')
    ;;
  *) echo 'CONC must be 1, 2, 4, 8, 16, 32, 48, or 64' >&2; exit 1 ;;
esac
/opt/venv/bin/python -u -m atom.entrypoints.openai_server \
  --model "$MODEL_PATH" --served-model-name "$MODEL_ID" \
  --host 127.0.0.1 --server-port "$HTTP_PORT" \
  --port "$ENGINE_PORT" --data-parallel-master-port "$DP_PORT" \
  --trust-remote-code -tp 1 --gpu-memory-utilization 0.95 \
  --max-num-seqs "$max_num_seqs" --max-num-batched-tokens 16384 \
  --max-model-len 262144 --block-size 64 \
  --kv_cache_dtype fp8 --index-cache-dtype bf16 \
  --cudagraph-capture-sizes '[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,18,20,22,24,26,28,30,32,40,48,56,64]' \
  --enable-prefix-caching --state-checkpoint-interval-tokens 8192 \
  --method mtp --num-speculative-tokens 3 \
  --online_quant_config '{"global_quant_config":"ptpc_fp8","exclude_layer":["lm_head","*mlp.gate","*shared_expert_gate","*mlp.experts*","*mlp.shared_expert*","*hyper_connection*","*indexer*","*.ple.*","model.visual.*","mtp.*"]}' \
  --spec-decode-acceptance-length 2.32 --timeout-keep-alive 60 \
  "${extra_args[@]}" 2>&1 | tee "/results/server-c${CONC}-$(date -u +%Y%m%dT%H%M%SZ).log"
```

### Fixed client measurement

Before measuring, check the server log: c1-c16 should report both `Qwen mono:`
and `Qwen mono draft:` activation; c32/c48/c64 should report `Qwen mono off:` and
`state tier up` with `sharing the paged-KV CPU pool`. Confirm GPU memory is
allocated and model loading and warmup complete, then open a second host terminal with
`docker exec -it qwen38-agentx /bin/bash`. Set `CONC`, `HTTP_PORT`, and
`MODEL_PATH` to match the server. Each point has a 3,600-second request
issuance window; allow additional wall time for dataset configuration,
priming ten requests per lane, and draining requests. For a full curve,
collect two complete rounds per point and retain each round's artifacts.

```bash
set -euo pipefail
export CONC="${CONC:-8}" HTTP_PORT=8000 MODEL_PATH=/assets/model
export MODEL_ID=amd/Qwen3.8-Flash-Next-Quark-MXFP4
export HF_HOME=/assets/hf-cache HF_HUB_ENABLE_HF_TRANSFER=0
unset HF_HUB_OFFLINE
export HF_DATASETS_OFFLINE=1
export AIPERF_DATASET_CONFIGURATION_TIMEOUT=1800
export AIPERF_SERVICE_PROFILE_CONFIGURE_TIMEOUT=1800
export AIPERF_UI_REALTIME_METRICS_ENABLED=true
curl --fail "http://127.0.0.1:${HTTP_PORT}/v1/models" | \
  /opt/agentx/bin/python -c 'import json, os, sys; assert os.environ["MODEL_ID"] in [m["id"] for m in json.load(sys.stdin)["data"]]'
output_dir="/results/c${CONC}-$(date -u +%Y%m%dT%H%M%SZ)"
/opt/agentx/bin/aiperf profile \
  --scenario inferencex-agentx-mvp \
  --url "http://127.0.0.1:${HTTP_PORT}" \
  --endpoint /v1/chat/completions --endpoint-type chat --streaming \
  --model "$MODEL_ID" --tokenizer "$MODEL_PATH" \
  --concurrency "$CONC" --benchmark-duration 3600 \
  --stats-interval 30 --random-seed 42 --failed-request-threshold 0.10 \
  --trajectory-start-min-ratio 0.25 --trajectory-start-max-ratio 0.75 \
  --warmup-requests-per-lane 10 --trace-idle-gap-cap-seconds 300 \
  --agentic-warmup-grace-period 1800 --use-server-token-count --no-gpu-telemetry \
  --tokenizer-trust-remote-code --num-dataset-entries 393 --slice-duration 1.0 \
  --server-metrics "http://127.0.0.1:${HTTP_PORT}/metrics" \
  --output-artifact-dir "$output_dir" \
  --public-dataset semianalysis_cc_traces_weka_062126_256k \
  2>&1 | tee "${output_dir}.log"
test -s "$output_dir/profile_export_aiperf.json"
```

`--agentic-warmup-grace-period 1800` sets the agentic warmup deadline.
The original runs used `--warmup-grace-period 1800`; the pinned 1.0.9 client
routes that flag to the same agentic field under this scenario. The exported
configuration should show
`agentic_warmup_grace_period=1800`, profiling `grace_period=30`,
`system_idle_gap_cap_seconds=10`, and `trace_idle_gap_cap_seconds=300`.

Keep the client log, `profile_export_aiperf.json`, request JSONL, lifecycle
counts, and server metrics together. A successful client exit does not mean
all requests completed: `--failed-request-threshold 0.10` is only an abort
threshold. Report errors and cancellations separately, including requests
cancelled at the 30-second drain limit. Request latency percentiles
cover completed requests; cancelled requests have no latency samples, and
one-token outputs have no inter-token latency.

For historical comparisons, report **input + output tokens/GPU/s** from
successful profiling rows in `profile_export.jsonl`. Sum
`metrics.input_sequence_length.value + metrics.output_sequence_length.value`
across those rows, including usage-counted cached prompt tokens. Divide by
`(max(metadata.request_end_ns) - min(metadata.request_start_ns)) / 1e9`
and by the GPU count (one here). This actual request span includes drain
time; 3,600 seconds is only the issuance window. Exclude warmup rows and
report cancelled requests separately. This replay with forced acceptance and
archived responses does not measure model accuracy.

After the selected points finish, stop the recipe container with `docker stop qwen38-agentx`.
