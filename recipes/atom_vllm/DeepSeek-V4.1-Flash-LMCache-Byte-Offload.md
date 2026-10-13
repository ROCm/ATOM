# DeepSeek-V4.1-Flash: LMCache byte offload

## Startup

Run inside the prepared vLLM 0.31 / ATOM environment. This tested configuration
uses TP4, FULL graphs, DSpark 5, and in-process PAGE + STATE offload.

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

Against an otherwise idle server, run the short and long three-arm probes:

```bash
python3 tools/kv_offload_accuracy_arms.py 8021 16 64 96 DeepSeek-V4.1-Flash \
  --out accuracy-short.json
python3 tools/kv_offload_accuracy_arms.py 8021 16 900 640 DeepSeek-V4.1-Flash \
  --out accuracy-long.json
```

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
`0.5.5rc3+rocm7.2.4.torch2.10.git3d3aa833.cxx11abi1`; configuration above,
synthetic recall, concurrency 1 unless specified. Supply is warm-tier tokens /
warm prompt tokens.

| Backend / run | Blind | Cold | Warm | Tier supply | Verdict |
|---|---:|---:|---:|---:|---|
| In-process, short 1 | 0/16 | 16/16 | 16/16 | 20,480 / 21,520 (95.2%) | Passed |
| In-process, short 2 | 0/16 | 16/16 | 16/16 | 20,480 / 21,520 (95.2%) | Passed |
| In-process, long (~17.5K prompt) | 0/16 | 16/16 | 16/16 | 278,528 / 279,344 (99.7%) | Passed |
| MP, short before namespace fix | 0/16 | 16/16 | 16/16 | 0 / 21,520 (0%) | Invalid: no recovery |
| MP, short after namespace fix | 0/16 | 16/16 | 16/16 | 20,480 / 21,520 (95.2%) | Passed |
| MP, long after namespace fix | 0/16 | 16/16 | 16/16 | 278,528 / 279,344 (99.7%) | Passed |
| MP, long, concurrency 4, before holding-queue fix | 0/16 | 16/16 | Incomplete | Not scored | Failed: mixed verify/prefill step |
| MP, long, after holding-queue fix | 0/16 | 16/16 | 16/16 | 278,528 / 279,344 (99.7%) | Passed |
| MP, long, concurrency 4, after holding-queue fix, twice | 0/16 | 16/16 | 16/16 | 278,528 / 279,344 (99.7%) each | Passed both runs |

Completed arms had zero cold-tier and local-HBM supply. MP used a 32 GiB shared
PAGE pool and a 4 GiB STATE pool per rank. The namespace fix aligns scheduler
and worker PAGE blocks at 256 tokens while preserving draft blocks at 32.
Before the holding-queue fix, concurrency 4 returned HTTP 500 on a mixed
verify / restored tail-prefill batch. After the fix, one long single-client run
and two concurrency-4 runs passed in the **same boot** at source `1e3ecb590`;
all client/drain exits were zero. Only cold-arm nonces changed between runs;
the warm document was the same fixed-seed prefix, not three independently
primed documents. The 1,051 tracked ATOM
source files were unchanged throughout those runs. This is not independent-boot
or high-concurrency validation.

Shutdown subsequently logged LMCache `LocalCPUBackend` close errors and AITER
`free_meta_buffer` destructor errors. They did not occur during the probes;
vLLM escalated its four workers to SIGKILL on shutdown. All seven recorded
model/MP processes and their host KFD entries were confirmed absent afterward.

CPU-focused receive/STATE/namespace/scheduling/probe/wait-script regression:
**520 passed**, plus 69 subtests. This includes a subsequent CPU-only fix for
running output-history replay beyond the original prompt; graph validation of
that change is pending. These checks and the smoke runs do not establish full
GSM8K accuracy, STATE byte equality, high-concurrency capacity safety, or local
HBM restoration. Those validations remain pending. **No current performance
result is claimed.**
