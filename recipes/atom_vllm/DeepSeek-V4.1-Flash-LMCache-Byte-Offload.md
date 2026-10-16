# DeepSeek-V4.1-Flash: LMCache byte offload

## Startup

Run from the repository root in a prepared vLLM 0.31 / ATOM environment.
Use only your allocated GPUs and unused ports. Keep the source unchanged until
both the model and MP server have stopped.

Start the MP PAGE server in a separate terminal:

```bash
HIP_VISIBLE_DEVICES=0,1,2,3 LMCACHE_DISABLE_BANNER=1 lmcache server \
  --host localhost --port 5561 --chunk-size 256 \
  --l1-size-gb 64 --eviction-policy LRU \
  --supported-transfer-mode auto --shm-name dsv41mp_5561 \
  --no-l1-use-lazy --max-gpu-workers 4 \
  --http-host 127.0.0.1 --http-port 8086 --prometheus-port 9006
```

Start the model (ON):

```bash
export MODEL=/data/amd_int/models/DeepSeek-V4.1-Flash
export PYTHONPATH="$PWD"
export HIP_VISIBLE_DEVICES=0,1,2,3
export AITER_LOG_LEVEL=WARNING VLLM_WORKER_MULTIPROC_METHOD=spawn
export PYTHONHASHSEED=0
export ATOM_V41_EXPERIMENTAL_CUDAGRAPH=1 VLLM_USE_BREAKABLE_CUDAGRAPH=1
export ATOM_V41_ALLOW_SPECULATIVE_OFFLOAD=1
export ATOM_V41_STATE_CHECKPOINTS=0 ATOM_V41_LOCAL_PREFIX_HITS=0
export ATOM_V41_CHECKPOINT_LOCAL_HITS=0
export LMCACHE_LOCAL_CPU=True LMCACHE_MAX_LOCAL_CPU_SIZE=16
export LMCACHE_CHUNK_SIZE=256 LMCACHE_CACHE_POLICY=ATOM_SLRU
export OFFLOAD_MIN_LOAD_TOKENS=256

vllm serve "$MODEL" \
  --served-model-name DeepSeek-V4.1-Flash \
  --host 127.0.0.1 --port 8021 \
  --dtype auto --kv-cache-dtype auto \
  --tensor-parallel-size 4 --distributed-executor-backend mp \
  --trust-remote-code --tokenizer-mode deepseek_v4 \
  --gpu-memory-utilization 0.78 \
  --max-model-len 32768 --max-num-seqs 64 --max-num-batched-tokens 8192 \
  --enable-prefix-caching \
  --speculative-config '{"method":"dspark","num_speculative_tokens":5,"kv_cache_dtype":"fp8_ds_mla"}' \
  --kv-transfer-config '{"kv_connector":"AtomLMCacheOffloadConnector",
    "kv_connector_module_path":"atom.plugin.vllm.kv_transfer.connector",
    "kv_role":"kv_both","kv_load_failure_policy":"recompute",
    "kv_connector_extra_config":{"atom.offload.backend":"mp",
      "lmcache.mp.host":"tcp://localhost","lmcache.mp.port":5561,
      "lmcache.mp.state_transport":"own-pool","lmcache.mp.state_cpu_size_gb":16,
      "lmcache.mp.tp_rank_collapse":true,"atom.offload.v41.state_interval":4096}}'
```

This allocates **64 GiB shared PAGE + 16 GiB STATE per rank** (128 GiB nominal
CPU pools at TP4, plus process/staging overhead). The target uses the ATOM BF16
pool; DSpark requires its separate explicit `fp8_ds_mla` pool.

For OFF, stop ON and its MP server, then start the same model command without
`--kv-transfer-config`, replacing `--enable-prefix-caching` with
`--no-enable-prefix-caching`. Keep graphs, DSpark and all other model parameters
unchanged. This compares CPU PAGE+STATE reuse against **no prefix reuse**, not
CPU offload against existing local-HBM prefix reuse. The three checkpoint/local-hit
flags above must remain zero for this tested configuration.

## Test method

Check model identity, graph capture, GPU/process ownership and readiness before
sending requests. Use an otherwise idle endpoint; check client exit status,
complete results, server faults and actual token-source counters, not HTTP 200
or launcher exit alone. Stop only the verified processes belonging to your run.

### Fixed-manifest GSM8K

Render the official test1319/5-shot prompts **once** and reuse the same
manifest for all four passes (ON twice, then OFF twice) -- GSM8K resamples its
few-shot exemplars per question, so a re-render changes the prompts and the
two arms stop being comparable. Score with the official strict/flexible
filters over raw completions, at concurrency 16, temperature and seed 0, and a
4096-token budget.

All 1319 responses must complete and be scored; any length truncation fails
the gate. The second ON pass must show the tier actually supplying at least
10% of response prompt tokens, and the three source counters (local HBM, tier,
recomputed) must sum to that same token count -- a measurement that passes
without the tier supplying anything says nothing about the tier. A passing
measurement alone does not establish accuracy equivalence.

### Fixed-request performance

Generate each request set once per `(concurrency, round)` and reuse it on both
arms. Prime a dozen distinct documents serially (one output token each), then
measure streaming requests with unique tails and a fixed output length
(`ignore_eos=true`). Measure from first dispatch to last complete response,
including ramp and drain but excluding the prime and any metric scrapes.

Require every request to complete with `[DONE]`, usage and valid source
accounting; on ON the tier's share must exceed 50%. Never report throughput
from a partially successful run, and only compare two passing summaries taken
from identical request sets. Priming is not a PAGE/STATE save barrier, so this
is not a proven steady-state warm-cache window. Write evidence to fast local
storage: synchronous writes can stall the client event loop and inflate the
latency you measure.

## Test results

2026-10-14, server source **`fdb24d5d2`**, MI355 TP4; vLLM
`0.31.1.dev0+gdb9527a46.d20261008.rocm724`, LMCache
`0.5.5rc3+rocm7.2.4.torch2.10.git3d3aa833.cxx11abi1`. Both boots captured target
PIECEWISE 53 / FULL 41 and DSpark FULL 38 graphs. CPU pools were 64 GiB shared
PAGE + 16 GiB STATE/rank. The four GSM8K passes and six successful ON performance
windows had zero local-HBM supply; the failed OFF performance window has no
complete source measurement.

| GSM8K pass | Flexible exact match | Strict exact match | Actual CPU / prompt tokens | Length truncations |
|---|---:|---:|---:|---:|
| ON 1 | 1221/1319 (92.57%) | 1222/1319 (92.65%) | 0 / 1,154,587 | 0 |
| ON 2 | 1216/1319 (92.19%) | 1217/1319 (92.27%) | 975,104 / 1,154,587 (84.45%) | 1 |
| OFF 1 | 1225/1319 (92.87%) | 1226/1319 (92.95%) | 0 / 1,154,587 | 1 |
| OFF 2 | 1218/1319 (92.34%) | 1219/1319 (92.42%) | 0 / 1,154,587 | 2 |

All four passes completed/scored 1319 responses without HTTP errors; their
three source counters matched response usage. **Accuracy acceptance is not
passed:** ON 2 and both OFF passes failed the predeclared no-truncation gate.
Repeated enumeration occurred with and without offload. These two repeats per
arm do not establish a noise floor, equivalence or a causal attribution.

ON performance: each row is a complete 240-request window. Total tokens/s
includes reused input tokens, not just tokens computed on the GPU. TPOT is the
client-observed `(last_content-first_content)/(output_tokens-1)`, not per-token
ITL. Latency columns are request percentiles in milliseconds.

| Concurrency / round | Total tokens/s | TTFT p50 / p99 | TPOT p50 / p99 | Actual CPU supply |
|---|---:|---:|---:|---:|
| 16 / 1 | 103,430 | 504 / 859 | 10.75 / 18.66 | 97.275% |
| 16 / 2 | 100,184 | 475 / 768 | 11.38 / 20.49 | 97.287% |
| 16 / 3 | 102,137 | 448 / 770 | 11.91 / 20.04 | 97.287% |
| 24 / 1 | 116,619 | 608 / 1077 | 15.37 / 26.77 | 97.262% |
| 24 / 2 | 125,360 | 718 / 1048 | 11.80 / 22.23 | 97.236% |
| 24 / 3 | 119,290 | 620 / 1024 | 14.95 / 25.14 | 97.287% |

Actual mean input length was 7,631–7,635 tokens/request. These are three rounds
per concurrency in one ON boot, not independent boot replications.
**No ON/OFF speedup is available:** OFF 16/1 crashed after 160/240 measured
requests with HIP illegal memory access; its throughput is invalid and the
remaining five OFF rounds were not run. The error surfaced asynchronously at
`deepseek_v41_bridge.py:1029` during `seq_lens.cpu()`; the offending kernel is
not identified. ON ran before OFF on a shared host, not randomized/interleaved.

**Shutdown is still failing.** ON logged four CA and eight IPC destructor
exceptions and four-worker SIGKILL escalation after explicit TP close hooks.
OFF exited after its inference fault. Launcher exit 0 did not override either
failure. The model/MP KFD resources were released; OFF left four non-holding
zombies (no KFD entry), not a clean process teardown.

The server and experimental clients were frozen throughout testing. The shipped
tools preserve request/score construction but add post-measurement deadline,
UTF-8, source/lookup, response-limit and SSE-termination fixes; they are not the
client hashes used above. Legacy performance summaries are not version-2 passes:
the old client stopped at `[DONE]`, so any later HTTP bytes were not observed.
Scoped CPU tests: **78 passed, 1 skipped, 189 subtests passed**; the official
offline integration separately passed and reproduced the frozen 1319-question
manifest exactly. All four tool/test files pass Black and Ruff. The experimental
client's missing-source gate was supplemented by independent checks on every
pass. Historical smoke/CPU results and detailed evidence remain in PR #2357.
Local-HBM + CPU-tail joint restoration, STATE byte equality, cancellation/load
failure recomputation and recoverable STATE capacity remain unvalidated or
unfinished. These results are **not complete feature acceptance**.
