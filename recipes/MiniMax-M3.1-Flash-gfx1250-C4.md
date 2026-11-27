# MiniMax-M3.1-Flash (Quark MXFP4) on 4 x MI455X: C4-optimized ATOM + aiter

This recipe installs the C4-decode-optimized ATOM and aiter branches into the official ATOM base image for
gfx1250, starts MiniMax-M3.1-Flash with TP4 + DSpark (K = 7), and reproduces our accuracy and InferenceX AgentX
numbers (C4 and C64, 1 hour). A prebuilt image with everything installed is in section 2.

| | |
|---|---|
| Model | `amd/MiniMax-M3.1-Flash-Quark-MXFP4` (MXFP4 MoE, MXFP8 dense), with its DSpark draft in `dspark/` |
| Hardware | 4 x AMD Instinct MI455X (gfx1250), TP4 |
| Base image | `rocm/atom-dev:ub24-dcgpu-rk-10.2.0-pyt2.14.0-rocmtri3.9-24d5394c-atomsrc-gfx1250-b1123` (torch 2.14.0 + ROCm 10.2, triton 3.9.0, ATOM / aiter in `/root/venv/atom-venv-gfx1250`) |
| ATOM | branch `xiaobingsuper/m31-c4-gfx1250-opt` of ROCm/ATOM; measured code = commit `b1142d8af8d61aa21e2dd1fcf65bd8a8148963d8` (later commits on the branch only touch `recipes/` and `tests/`) |
| aiter | branch `xiaobingsuper/m31-c4-gfx1250-opt` of ROCm/aiter, commit `e871889f933d410c32147815ab454710580f944a` (composable_kernel submodule `96afae1d`) |
| LMCache (C64 only) | LMCache/LMCache `f22dec28` (0.5.6rc3.dev20), built in the container |
| Benchmark clients | aiperf (SemiAnalysisAI/aiperf `754356e9`, scenario `inferencex-agentx-mvp`), lm_eval 0.4.12 |

## What the branches contain

Both branches are the code we measured, unchanged:

- **ATOM** = the MiniMax-M3.1 gfx1250 stack `61c8db8` (ROCm/ATOM main + the M3.1 PRs + the combo16 fusions) plus:
  - the first round of C4 decode work (November 25): async verdict and early output, one-launch draft metadata /
    sampler / verify, draft KV inside the decode graph, Markov-chain resident kernel, stream-capture and H2D host
    trims, the LM head in its own graph, o_proj + post-attention all-reduce + residual + RMSNorm in one launch
    ("Stage O"), DSpark draft layers as two launches each;
  - the second round (November 27), one commit per piece. The gfx1250 decode launches live in
    `atom/models/minimax_m3/mono/gfx1250/`:
    - MoE front launch (o_proj + all-reduce + residual + RMSNorm + router + shared expert at 8 rows), the routed MoE on
      one stream, and its GEMM2 + tail + the next all-reduce + norm + MXFP8 quant as one launch, which also runs the
      shared expert's down projection;
    - AQ: QKV norm / RoPE / NVFP4 insert, this rank's indexer-CP scores and the cross-rank index top-k exchange in one
      launch, including layer 0's QKV chain;
    - DMF: a dense layer's o_proj + all-reduce + norm + MLP in one launch;
    - the LM head at <= 8 rows as one TDM-streamed launch (bit-identical to the GEMM it replaces);
    - DSpark: the draft tap written by the next all-reduce, four draft-graph nodes removed;
    - host trims of the decode step: logits all-gather in the decode graph, deferred D2H, next ids from the verify
      launch, no per-step attention-plan refresh;
  - work from another workstream on the same branch: the prefill / decode (PD) disaggregation stack, and the FlyDSL
    FP4 indexer paths for gfx1250, which are on by default (decode scorer + top-k, fused prefill scorer + top-16).
- **aiter** = the matching aiter stack `9c3f0ef` plus:
  - the C4 kernels (Stage O kernels, 8-row router, QKV / o_proj tiles, GEMM1 wave-level route for <= 16 tokens,
    split-K GEMM1 plan for 8 tokens);
  - the routed-MoE tuning rows in `aiter/configs/model_configs/minimax_m3_gfx1250_tuned_grouped_fmoe.csv`, so aiter's
    default config lookup picks them (no `AITER_CONFIG_GROUPED_FMOE`, no file copy);
  - a model-free hook that lets a model register a deferred MoE tail launch with the all-reduce, plus residual taps;
  - the other workstream's MXFP8 GEMM kernels and tables (ROCm/aiter#8273), FlyDSL indexer kernels and the PD
    prefill MoE rows.

The C4 pieces are on by default and gate themselves (gfx1250, TP4, a decode step of one request x 8 rows, the
indexer CP, ...); other shapes take the stack's regular path. They add no environment variables. The stack's other
fusions are opt-in flags and are set in the launch command below.

## 1. Host

- amdgpu loaded with `sudo modprobe amdgpu gpu_recovery=0 halt_if_hws_hang=1` (how our nodes load it).
- `sudo sysctl -w kernel.numa_balancing=0` for the benchmark runs.
- About 230 GB of disk for the model, 60 GB for the image, ~25 GB for the container's build outputs.
- C64 with LMCache (384 GiB per rank) needs about **1,690 GiB of available host memory** (4 x 384 x 1.1 GiB; a ~2 TB
  node). On a ~1 TB node run C64 **without LMCache** (section 4).

## 2. Container

Prebuilt image with section 3 already done: `rocm/fw-bringup:gfx1250-minimax-m3-20261127`. Set `IMAGE` to it below
and skip section 3. It holds:
- the two branches at the measured commits, installed editable in `/root/venv/atom-venv-gfx1250`
  (`/opt/ATOM`, `/opt/aiter`), with the CK fix of section 3 applied;
- LMCache `f22dec28` built for this torch / ROCm;
- the benchmark clients in `/opt/bench` (aiperf, lm_eval);
- prebuilt aiter JIT modules and warm FlyDSL / Triton / TorchInductor caches for the C4 and C64 configurations, so
  the first server start and the first requests do not wait for kernel compilation;
- a 1P1D prefill / decode disaggregation recipe for two nodes in `/app/m31_pd_recipe/` (README, `kit/`, `ctl/`).

The numbers in section 7 dated November 27 were measured with the code in this image.

```bash
IMAGE=rocm/atom-dev:ub24-dcgpu-rk-10.2.0-pyt2.14.0-rocmtri3.9-24d5394c-atomsrc-gfx1250-b1123
MODEL_DIR=/path/to/MiniMax-M3.1-Flash-Quark-MXFP4      # with dspark/ inside
docker pull $IMAGE
docker run -d --name m31 --network host --ipc host --group-add video --cap-add SYS_PTRACE \
  --security-opt seccomp=unconfined --security-opt label=disable \
  --device /dev/kfd --device /dev/dri --shm-size 64g --ulimit memlock=-1 \
  -v $MODEL_DIR:/models/MiniMax-M3.1-Flash-Quark-MXFP4:ro \
  --entrypoint sleep $IMAGE infinity
docker exec -it m31 bash
# inside the container: the image's default PATH has /root/venv/rocm-venv; use the ATOM venv
source /root/venv/atom-venv-gfx1250/bin/activate
export GPU_ARCHS=gfx1250 PYTORCH_ROCM_ARCH=gfx1250
```

## 3. Install the branches (replaces the image's ATOM and aiter)

```bash
BRANCH=xiaobingsuper/m31-c4-gfx1250-opt
# ATOM (pure Python)
git clone --depth 1 -b $BRANCH https://github.com/ROCm/ATOM.git /opt/ATOM
# later commits only touch recipes/ and tests/; this prints nothing:
git -C /opt/ATOM fetch -q --depth 1 origin b1142d8af8d61aa21e2dd1fcf65bd8a8148963d8 && \
  git -C /opt/ATOM diff --stat FETCH_HEAD HEAD -- . ':!recipes' ':!tests'
pip install --no-deps --no-build-isolation -e /opt/ATOM
# aiter (Python + JIT-built HIP modules; nothing is compiled here)
git clone --depth 1 -b $BRANCH https://github.com/ROCm/aiter.git /opt/aiter
git -C /opt/aiter log -1 --format=%H         # e871889f933d410c32147815ab454710580f944a
git -C /opt/aiter submodule update --init --depth 1 3rdparty/composable_kernel
# ROCm 10.2's clang defines __gfx125__ for gfx1250, which turns on CK's TDM block, where a misplaced brace in
# amd_tdm_store breaks the module_sample / module_activation JIT builds (C64 needs them). One-line fix, the same as in
# CK develop; no kernel code changes.
CKP=/opt/ATOM/recipes/minimax-m31-gfx1250/ck_tdm_store_brace.patch
git -C /opt/aiter/3rdparty/composable_kernel apply --check $CKP && git -C /opt/aiter/3rdparty/composable_kernel apply $CKP
pip uninstall -y amd-aiter
cd /opt/aiter && PREBUILD_KERNELS=0 pip install --no-deps --no-build-isolation -e . && cd -
python3 -c 'import atom, aiter, flydsl; print(atom.__file__, aiter.__file__, flydsl.__version__)'
# -> /opt/ATOM/atom/__init__.py /opt/aiter/aiter/__init__.py 0.5.0
```

Notes:
- The aiter branch pins `flydsl==0.5.0` (the FlyDSL the numbers were measured with); its editable install puts 0.5.0
  into the ATOM venv, in front of the image's 0.5.3 in `/root/venv/rocm-venv`. Do not upgrade it.
- aiter's HIP modules (custom all-reduce, the k5 / two-shot AR + RMSNorm, the fused router, ...) are JIT-built into
  `/opt/aiter/aiter/jit/build` on the server's first start (a few minutes; the first start took ~8 minutes to healthy in
  our runs, later starts ~5); FlyDSL kernels compile on first use. Keep the container to keep the builds.

### LMCache (only for the C64 LMCache-on configuration)

The base image has no LMCache. Build the LMCache commit the numbers were measured with for this torch / ROCm. torch
2.14's headers need C++20, which this LMCache commit does not set yet:

```bash
git init -q /opt/LMCache && cd /opt/LMCache
git remote add origin https://github.com/LMCache/LMCache.git
git fetch -q --depth 1 origin f22dec2832124ea4ffeb464553b819a6369d4a30 && git checkout -q FETCH_HEAD
sed -i 's/"-std=c++17"/"-std=c++20"/' setup_extensions/build_profiles/rocm.py setup_extensions/common_cpp.py
BUILD_WITH_HIP=1 CXX=hipcc SETUPTOOLS_SCM_PRETEND_VERSION=0.5.6rc3.dev20 \
  pip wheel --no-build-isolation --no-deps -w /opt/wheels .
# its runtime dependencies, without moving any package the image already has (protobuf moves into LMCache's range)
pip freeze | grep -vE '^-e |@ file:|^#|^(protobuf|importlib_metadata)==' > /tmp/cons.txt
pip install -c /tmp/cons.txt prometheus_client aiofile aiofiles caio blake3 redis sortedcontainers pyzmq cachetools \
  cryptography numba openai opentelemetry-api==1.40.0 opentelemetry-sdk==1.40.0 opentelemetry-exporter-otlp==1.40.0 \
  opentelemetry-exporter-prometheus==0.61b0 'grpcio>=1.78.0' 'protobuf>=6.31.1,<7' py-cpuinfo
pip install --no-deps /opt/wheels/lmcache-0.5.6rc3.dev20-*.whl
cd -    # leave the source tree before the check, or `import lmcache` finds /opt/LMCache/lmcache (no extensions)
python3 -c 'import lmcache, lmcache.cuda_ops; from lmcache.v1.cache_engine import LMCacheEngineBuilder; print(lmcache.__version__)'
```

(pip then reports that grpcio-tools 1.85.3 wants protobuf >= 7.35; it is a build tool and the server does not use
it.)

### Benchmark clients (separate venv, so the server's packages do not move)

```bash
/root/venv/rocm-venv/bin/python3 -m venv /opt/bench
/opt/bench/bin/pip install "aiperf @ git+https://github.com/SemiAnalysisAI/aiperf.git@754356e9a39acc6cc6afb242d123bb57c3fb6f75" \
  'lm_eval[api]==0.4.12'
```

## 4. Launch the server

aiter.plugin's config snapshot must be built when it is imported, outside torch.compile's trace (otherwise Dynamo
breaks the graph on its lock and the start fails with "VllmBackend can only be called once"); the hook
`recipes/minimax-m31-gfx1250/pyhooks/sitecustomize.py` does that and is on `PYTHONPATH` below. It must be a
sitecustomize: inside the venv Python never imports a usercustomize.

```bash
cd /opt/ATOM
M=/models/MiniMax-M3.1-Flash-Quark-MXFP4
CONC=4                     # 4 for C4; 64 for C64
MAX_SEQS=$((2 * CONC))     # 8 for C4, 128 for C64
CG_SIZES=$(python3 -c "print([s for s in [1,2,3,4,5,6,7,8,10,12,14,16,20,24,28,32,40,48,56,64,80,96,112,128] if s <= $MAX_SEQS])")
TG_SIZES=256$(seq -s '' -f ',%g' 512 512 11264)
env NCCL_IB_DISABLE=1 RCCL_IB_DISABLE=1 ATOM_FORCE_ATTN_TRITON=1 AITER_LOG_LEVEL=WARNING \
  ATOM_GC_THRESHOLD=20000,50,50 ATOM_PA_FLYDSL=1 ATOM_MONO_ENABLE=0 \
  PYTORCH_ROCM_ARCH=gfx1250 GPU_ARCHS=gfx1250 HSA_OVERRIDE_GFX_VERSION=12.5.0 \
  ATOM_MOE_GU_ITLV=1 AITER_ROPE_TRITON_BACKEND=1 AITER_USE_SYSTEM_TRITON=1 AITER_QUICK_REDUCE_CAST_BF16_TO_FP16=1 \
  RCCL_DDA_THRESHOLD=67108864 \
  ATOM_MINIMAX_M3_GLUON_AR_MXFP8=1 ATOM_MINIMAX_M3_SWIGLU_MXFP8=1 ATOM_MINIMAX_M3_PA_MXFP8_OUT=1 \
  LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libgomp.so.1 \
  PYTHONPATH=/opt/ATOM/recipes/minimax-m31-gfx1250/pyhooks \
  ATOM_PREFILL_DECODE_INTERVAL=20000 \
  ATOM_M3_INDEXER_CP=1 ATOM_M3_SPARSE_KV_NVFP4=1 \
  ATOM_MINIMAX_M3_PREFILL_TOKEN_GRAPH=1 ATOM_TOKEN_GRAPH_SIZES=$TG_SIZES \
  ATOM_MINIMAX_M3_MOE_FRONT_FUSION=1 ATOM_MINIMAX_M3_MOE_GLUE_FUSION=1 ATOM_MINIMAX_M3_MOE_ROUTED_FIRST=1 \
  ATOM_MINIMAX_M3_MOE_TAIL_IN_AR=1 ATOM_MINIMAX_M3_MOE_ROUTER_QUANT=1 AITER_K5_AR=1 AITER_K5_AR_FUSED_MXFP8=1 \
  ATOM_M3_FUSED_QKV_CHAIN=1 ATOM_M3_FAST_OPROJ=1 ATOM_M3_PA_FUSED_REDUCE=1 ATOM_M3_CP_SCORE_FLYDSL=1 \
  ATOM_MINIMAX_M3_SHARED_FOLD=0 ATOM_MINIMAX_M3_MOE_QUANT_SHARED=0 \
  ATOM_M3_PREFILL_SP_AR=1 AITER_AR2S_RMS_MAX_SIZE=201326592 ATOM_M3_SP_MOE_FRONT=2 ATOM_M3_INDEX_SCORE_FLYDSL_PREFILL=1 \
  ATOM_M3_SPARSE_PREFILL_REUSE=1 ATOM_M3_SPARSE_PREFILL_REUSE_IMPL=v2 ATOM_M3_SPARSE_PREFILL_REUSE_TQ=2 \
  ATOM_SAMPLER_SPLIT_V=1 ATOM_DSPARK_FUSED_AR=1 ATOM_DSPARK_SWIGLU_MXFP8=1 AITER_MOE_ROUTER_QUANT_ROWS_MIN_TOKENS=8 \
  ATOM_ENGINE_IDLE_YIELD_S=0.0002 ATOM_OFFLOAD_PREHASHED_SAVE=1 ATOM_LOADER_PREFETCH=false \
  python3 -u -m atom.entrypoints.openai_server \
    --model $M --served-model-name $M \
    --host 0.0.0.0 --port 8796 --server-port 8790 \
    --tensor-parallel-size 4 --trust-remote-code \
    --kv_cache_dtype fp8 --index-cache-dtype fp4 --block-size 128 \
    --max-model-len 1048576 --max-num-seqs $MAX_SEQS --max-num-batched-tokens 32768 \
    --gpu-memory-utilization 0.9 --enable-prefix-caching \
    --cudagraph-capture-sizes "$CG_SIZES" \
    --method dspark --draft-model $M/dspark --num-speculative-tokens 7 \
    --spec-decode-acceptance-length 4.04 \
    > /tmp/server.log 2>&1 &
until curl -sf -m 5 http://127.0.0.1:8790/health > /dev/null; do sleep 30; done
```

- **Accuracy runs** (section 5): drop `--spec-decode-acceptance-length 4.04` (real DSpark acceptance), `CONC=4`.
- **AgentX C4**: as above (`CONC=4`, delayer 20000, forced acceptance 4.04, LMCache off).
- **AgentX C64**: `CONC=64` (max-num-seqs 128), `ATOM_PREFILL_DECODE_INTERVAL=8` instead of 20000, forced acceptance
  4.04.
  - **On a node with ~2 TB of RAM**, add the LMCache CPU tier: add to the env
    `PYTHONHASHSEED=0 LMCACHE_LOCAL_CPU=True LMCACHE_MAX_LOCAL_CPU_SIZE=384 LMCACHE_CHUNK_SIZE=256
    ATOM_PREFIX_CACHE_POLICY=slru ATOM_PREFIX_CACHE_PROTECTED_RATIO=0.5 LMCACHE_CACHE_POLICY=ATOM_SLRU
    LMCACHE_LOOKUP_SERVER_WORKER_IDS=0,1,2,3`
    and to the arguments `--kv-transfer-config '{"kv_connector":"lmcache_offload","kv_role":"offload"}'`.
    `LMCACHE_MAX_LOCAL_CPU_SIZE` is GiB **per TP rank**; 384 needs 4 x 384 x 1.1 = ~1,690 GiB of `MemAvailable` before
    the start (`grep MemAvailable /proc/meminfo`). Once the tier is full, LMCache logs "Failed to allocate memory block
    ... no memory is available" for saves it cannot place; this is expected.
  - **On a node with ~1 TB of RAM, run C64 without LMCache**: leave out the LMCache env and `--kv-transfer-config`. A
    smaller tier does not help there: at 160 GiB/rank it stayed full from the warm-up on and loaded 0 tokens in the hour,
    and C64 P90 was 244.3 against 266.7 with LMCache off on the same node.
- The forced acceptance length 4.04 is InferenceX's golden DSpark acceptance for M3.1 at K = 7; the generated text is
  synthetic under it, so never measure accuracy with it.
- Stop the server with `pkill -INT -f atom.entrypoints.openai_server` and wait until no `ATOM::` process is left.

## 5. Accuracy

On a fresh server without `--spec-decode-acceptance-length` (`CONC=4`):

```bash
M=/models/MiniMax-M3.1-Flash-Quark-MXFP4
# sanity gate: 17 * 23 = 391 and a parsed tool call
python3 /opt/ATOM/recipes/minimax-m31-gfx1250/gate.py http://127.0.0.1:8790 $M          # last line: GATE PASS
# GSM8K 5-shot, greedy, one request at a time (first 250), then all 1319 at 16 concurrent requests
/opt/bench/bin/lm_eval --model local-chat-completions --apply_chat_template --fewshot_as_multiturn \
  --model_args "model=$M,base_url=http://localhost:8790/v1/chat/completions,num_concurrent=1,max_retries=3,max_gen_toks=16384,tokenized_requests=False,trust_remote_code=True" \
  --tasks gsm8k --num_fewshot 5 --limit 250 --log_samples --output_path /tmp/gsm8k_c1
/opt/bench/bin/lm_eval --model local-chat-completions --apply_chat_template --fewshot_as_multiturn \
  --model_args "model=$M,base_url=http://localhost:8790/v1/chat/completions,num_concurrent=16,max_retries=3,max_gen_toks=16384,tokenized_requests=False,trust_remote_code=True" \
  --tasks gsm8k --num_fewshot 5 --log_samples --output_path /tmp/gsm8k_c16
```

Accepted length: the server log prints cumulative `[MTP Stats ...] Accepted/Total Draft tokens: A/T`; over a phase,
accepted length = 1 + dA / (dT / 7) (take the line before and after the phase).

## 6. AgentX (InferenceX, 1 hour)

With the server up in the C4 or C64 configuration, run the AgentX replay (it downloads the dataset
`semianalysisai/cc-traces-weka-062126` from Hugging Face on first use):

```bash
CONC=4   # or 64
M=/models/MiniMax-M3.1-Flash-Quark-MXFP4
env AIPERF_HTTP_TCP_USER_TIMEOUT=900000 AIPERF_DATASET_CONFIGURATION_TIMEOUT=1800 \
  AIPERF_SERVICE_PROFILE_CONFIGURE_TIMEOUT=1800 AIPERF_TIMING_CANCEL_DRAIN_TIMEOUT=30 \
  AIPERF_DATASET_WEKA_LIVE_ASSISTANT_RESPONSES=0 AIPERF_UI_REALTIME_METRICS_ENABLED=true \
  AIPERF_FAILED_REQUEST_THRESHOLD=0.10 AIPERF_LIVE_FAILED_REQUEST_THRESHOLD=0.10 \
  AIPERF_WARMUP_REQUESTS_PER_LANE=10 AIPERF_BENCHMARK_GRACE_PERIOD=30 \
  AIPERF_SERVER_METRICS_URLS=http://localhost:8790/metrics AIPERF_REQUIRED_SERVER_METRIC_PREFIX=atom: \
  /opt/bench/bin/aiperf profile --scenario inferencex-agentx-mvp \
    --url http://localhost:8790 --endpoint /v1/chat/completions --endpoint-type chat --streaming \
    --model $M --tokenizer $M --tokenizer-trust-remote-code --apply-chat-template \
    --concurrency $CONC --benchmark-duration 3600 --stats-interval 30 \
    --random-seed 42 --failed-request-threshold 0.10 \
    --trajectory-start-min-ratio 0.25 --trajectory-start-max-ratio 0.75 \
    --warmup-requests-per-lane 10 --trace-idle-gap-cap-seconds 300 --agentic-warmup-grace-period 1800 \
    --use-server-token-count --no-gpu-telemetry \
    --num-dataset-entries 393 --slice-duration 1.0 \
    --server-metrics http://localhost:8790/metrics \
    --public-dataset semianalysis_cc_traces_weka_062126 \
    --output-artifact-dir /tmp/agentx_c$CONC
TP=4 python3 /opt/ATOM/recipes/minimax-m31-gfx1250/summarize.py /tmp/agentx_c$CONC
```

`summarize.py` prints one line with InferenceX's per-request formulas: **tok/s/chip** = (sum of input + output tokens
of the successful profiling requests) / (first start .. last end) / 4, and **interactivity P90** = 1 / (p90 of the
per-request inter-token latency) in tok/s/user (higher is better; the InferenceX "P90" number), plus the median,
ITL / TTFT percentiles, the prefix-cache hit and the request counts.

bs1 decode step (optional): `python3 /opt/ATOM/recipes/minimax-m31-gfx1250/a1_fixed.py --url http://localhost:8790
--model $M --summary /tmp/step.json --phases '[{"name":"b1_110k","bs":1,"ctx":110038,"settle_s":3,"dec_tokens":10000,"ref_s":6,"win_s":0,"pf_reps":0}]'`
on the C4 server (forced 4.04); `ref step` is the client-side time of one decode step at 110K context.

## 7. Expected numbers

November 27 rows: the prebuilt image of section 2 (the branches at the commits in the table at the top), on a
4 x MI455X node with 1 TB of host RAM (accuracy, C4). November 25 rows: the earlier branch commits (ATOM `51930f36`,
aiter `c3907928`) installed into the base image with this recipe, on 4 x MI455X nodes with 1 TB of host RAM (accuracy,
C4, C64 without LMCache) and 2 TB (C64 with LMCache at 384 GiB/rank).

**Accuracy** (fresh server, real DSpark acceptance, `CONC=4`):

| Check | November 27 | November 25 |
|---|---|---|
| gate.py (17 x 23, tool call) | PASS | PASS |
| GSM8K 5-shot c1, first 250 (flexible / strict) | 0.960 / 0.956 (responses byte-identical to November 25) | 0.960 / 0.956 |
| Accepted length, c1 phase | 4.637 | 4.637 |
| GSM8K 5-shot c16, all 1319 (flexible / strict) | 0.9538 / 0.9522 | 0.9522 / 0.9500 |
| Accepted length, c16 phase | 4.722 | 4.740 |

**bs1 decode step** (C4 server, forced acceptance 4.04, `a1_fixed.py`, 110K / 500K context): 4.099 / 4.353 ms with the
November 25 commits; not re-measured for November 27 (the C4 AgentX row below reflects the decode step).

**AgentX, 1 hour** (InferenceX AgentX replay, forced acceptance 4.04):

| Run | tok/s/chip | Interactivity P90 (tok/s/user) | Median | ITL p50 / p90 | TTFT p50 / p90 | Prefix hit | Requests / failed |
|---|---:|---:|---:|---|---|---|---|
| **C4, Nov 27** (LMCache off, 1 TB node) | **11,049** | **967.9** | 1004.2 | 1.0 / 1.0 ms | 0.05 / 0.54 s | 97.8% | 804 / 0 |
| C4, Nov 25 (LMCache off, 1 TB node) | 10,460 | 874.4 | 923.7 | 1.1 / 1.1 ms | 0.24 / 0.98 s | 97.7% | 764 / 0 |
| C64, Nov 25 (LMCache on, 384 GiB/rank, 2 TB node) | 128,726 | 286.1 | 393.9 | 2.5 / 3.5 ms | 0.33 / 0.83 s | 97.3% (LMCache 1.3%) | 12,236 / 3 |
| C64, Nov 25 (LMCache off, 1 TB node) | 128,233 | 266.7 | 403.5 | 2.5 / 3.7 ms | 0.31 / 1.01 s | 95.8% | 12,209 / 1 |
| C4, vLLM on MI355X (InferenceX) | 10,322 | 581.4 | | | | | |
| C64, vLLM on MI355X (InferenceX) | 112,879 | 114.7 | | | | | |

November 27, C4: interactivity P90 **1.66x** vLLM on MI355X (967.9 vs 581.4), tok/s/chip 1.07x.
November 25, C64 with LMCache at 384 GiB/rank (2 TB node): interactivity P90 **2.49x** vLLM on MI355X (286.1 vs 114.7),
tok/s/chip 1.14x (128,726 vs 112,879).
The vLLM MI355X numbers are the InferenceX values these comparisons have used since November 23; InferenceX's later
single-node TP4 run (November 24) reports P90 561.8 at C4 and 116.0 at C64.
On a ~1 TB node, C64 without LMCache: November 25 gave P90 266.7 (2.33x). Not recommended, for reference: LMCache at
160 GiB/rank on such a node gave P90 244.3 and 127,074 tok/s/chip (the tier stayed full and loaded nothing).
