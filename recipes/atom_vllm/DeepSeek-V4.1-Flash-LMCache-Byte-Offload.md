# DeepSeek-V4.1-Flash: LMCache byte offload

## Startup

Run inside the prepared vLLM 0.31 / ATOM environment. This tested configuration
uses TP4, FULL decode graphs (FULL_AND_PIECEWISE with breakable capture),
DSpark 5, and in-process PAGE + STATE offload.

```bash
export PYTHONPATH=/workspace/ATOM-dsv41-lmc031
export HIP_VISIBLE_DEVICES=0,1,2,3
export AITER_LOG_LEVEL=WARNING
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export PYTHONHASHSEED=0
export ATOM_V41_EXPERIMENTAL_CUDAGRAPH=1
export VLLM_USE_BREAKABLE_CUDAGRAPH=1
export ATOM_V41_ALLOW_SPECULATIVE_OFFLOAD=1
export ATOM_V41_STATE_IMAGES=0
export ATOM_V41_LOCAL_PREFIX_HITS=0
export ATOM_V41_IMAGE_LOCAL_HITS=0
export LMCACHE_LOCAL_CPU=True
export LMCACHE_MAX_LOCAL_CPU_SIZE=16
export LMCACHE_CHUNK_SIZE=256
export LMCACHE_CACHE_POLICY=ATOM_SLRU
export OFFLOAD_MIN_LOAD_TOKENS=256

vllm serve /data/amd_int/models/DeepSeek-V4.1-Flash \
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
    "kv_connector_extra_config":{"atom.offload.backend":"inproc",
      "atom.offload.v41.state_interval":4096}}'
```

For MP PAGE offload, start the shared server in a separate terminal with the
same GPU visibility. Use a unique shared-memory name and unused ports:

```bash
HIP_VISIBLE_DEVICES=0,1,2,3 LMCACHE_DISABLE_BANNER=1 lmcache server \
  --host localhost --port 5561 --chunk-size 256 \
  --l1-size-gb 32 --eviction-policy LRU \
  --supported-transfer-mode auto --shm-name dsv41mp_5561 \
  --no-l1-use-lazy --max-gpu-workers 4 \
  --http-host 127.0.0.1 --http-port 8086 --prometheus-port 9006
```

In the model command above, replace only `--kv-transfer-config` with:

```bash
--kv-transfer-config '{"kv_connector":"AtomLMCacheOffloadConnector",
  "kv_connector_module_path":"atom.plugin.vllm.kv_transfer.connector",
  "kv_role":"kv_both","kv_load_failure_policy":"recompute",
  "kv_connector_extra_config":{"atom.offload.backend":"mp",
    "lmcache.mp.host":"tcp://localhost","lmcache.mp.port":5561,
    "lmcache.mp.state_transport":"own-pool","lmcache.mp.state_cpu_size_gb":4,
    "lmcache.mp.tp_rank_collapse":true,"atom.offload.v41.state_interval":4096}}'
```

This MP configuration has one shared 32 GiB PAGE pool and a separate 4 GiB
STATE pool per rank, not a 16 GiB PAGE pool per rank.

Keep the explicit draft `kv_cache_dtype`: the target uses the ATOM BF16 pool,
while DSpark uses its own `fp8_ds_mla` pool. Local HBM prefix reuse is **not yet
supported by this tested configuration**; keep all three image/local-hit flags
above at zero. Stop the service before changing its editable source tree.

## Test method

Against an otherwise idle server, run long concurrency-4 probes:

```bash
python3 tools/kv_offload_accuracy_arms.py 8021 16 900 640 DeepSeek-V4.1-Flash \
  --concurrency 4 --seed 2026101301 --out accuracy-seed1.json
python3 tools/kv_offload_accuracy_arms.py 8021 16 900 640 DeepSeek-V4.1-Flash \
  --concurrency 4 --seed 2026101302 --out accuracy-seed2.json
python3 tools/kv_offload_accuracy_arms.py 8021 16 900 640 DeepSeek-V4.1-Flash \
  --concurrency 4 --seed 2026101303 --out accuracy-seed3.json
```

The different seeds produce different warm documents. For a short smoke test,
replace `900 640` with `64 96` and use another output filename.

Use fresh output filenames. Each invocation primes one synthetic document, then
runs blind (no document), cold (unique nonce at the document head), and warm
(shared document) arms. The tool checks both correctness and actual cache
supply; the default warm-tier minimum is 50% of **all prompt tokens in the arm**.
HTTP 200 or a completed drain alone is not a passing result: check the client
exit code and JSON status. For concurrency testing, add `--concurrency N`.

Full accuracy requires a separate two-pass evaluation that replays the same
prompts and verifies nonzero tier supply. A single GSM8K pass with independently
resampled few-shot examples does not establish that offload recovery occurred.
Performance must compare the same source, prompts, concurrency, and measurement
window with graphs on in both arms. Without the connector, V4.1 requires
`--no-enable-prefix-caching`, so that OFF comparison is reuse versus no reuse,
not CPU tier versus local HBM reuse.

## Test results

2026-10-13: vLLM `0.31.1.dev0+gdb9527a46.d20261008`, LMCache
`0.5.5rc3+rocm7.2.4.torch2.10.git3d3aa833.cxx11abi1`; TP4, FULL decode
and PIECEWISE graphs + DSpark 5, MP PAGE 32 GiB shared + STATE 4 GiB/rank.
Supply below is warm-tier tokens / all actual warm-arm prompt tokens.

| Source / long probe, concurrency 4 | Blind | Cold | Warm | Tier supply | Verdict |
|---|---:|---:|---:|---:|---|
| `f29878cab`, seed 2026101301 | 0/16 | 16/16 | 16/16 | 278,528 / 278,880 (99.874%) | Passed |
| `f29878cab`, seed 2026101302 | 0/16 | 16/16 | 16/16 | 278,528 / 279,488 (99.657%) | Passed |
| `10228cb99`, seed 2026101303 | 0/16 | 16/16 | 16/16 | 278,528 / 279,264 (99.736%) | Passed |

The first two distinct warm documents shared one boot. The third ran in a new
boot with the TP lifecycle fix (`3a97b1657`); its tested production/test bytes
match `10228cb99`, committed after teardown. All client/drain exits were zero;
cold-tier and local-HBM supply were zero. The latest run completed target and
DSpark FULL graph capture on four ranks, served 49 requests, and verified every
question's correctness and prompt-token sums. All 1,051 tracked ATOM files and
five validation files stayed unchanged through its model/MP teardown. These
are service smoke tests, not proof of every CPU corner case or full accuracy.

Latest shutdown **failed graceful-exit acceptance**: four explicit reused-AITER
TP close markers, four `LocalCPUBackend closed successfully` lines and zero
`Error closing` lines, followed by 12 AITER `free_meta_buffer` destructor
`UnicodeDecodeError`s and vLLM SIGKILL escalation for all four workers. The
worker timeout remained the default 5 seconds plus 4 seconds after SIGTERM.
Explicit close markers do not establish that every native resource was freed.
Model launcher exit was 0; MP exit was 143 from intentional SIGTERM. All seven
recorded processes and host KFD entries disappeared and four ports closed.

CPU-focused receive/STATE/namespace/scheduling/codec/probe/wait regression:
**825 passed + 69 subtests**, with 90 deprecation warnings. Includes ten TP
ownership/cleanup cases, 29 real sync/async admission cases (DSpark input-budget
edges use the real drafting-slot property), and real LMCache token-accounting,
flat-byte, `clear`/`close`, and full-pool cases. Black and Ruff passed.

Earlier in-process short/long and MP short/long probes passed. The earlier MP
namespace run with zero tier supply was invalid; the pre-holding-queue-fix
concurrency-4 run failed with HTTP 500. Historical results remain in PR #2357.
Full two-pass GSM8K, STATE byte equality, cancellation/failure-recompute,
high-concurrency capacity safety, and local-HBM restoration remain pending.
**No current performance result is claimed.**
