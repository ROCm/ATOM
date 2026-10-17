# DeepSeek-V4.1-Flash: LMCache byte offload (vLLM plugin, MP server)

DeepSeek-V4.1 offloads through the LMCache **MP server only**. Its per-request
CSA2 state travels with the PAGE bytes: state checkpoints are written into
single-layer checkpoint groups of the PAGE pool, and the MP worker registers
those groups as recurrent engine groups beside the PAGE planes, so one tier
object carries a prefix's KV and the checkpoint that continues it. There is no
in-process LMCache engine and no separate state pool; the connector refuses
`atom.offload.backend=inproc` for V4.1 and refuses to start without the
checkpoint groups.

## Startup

Run from the repository root in a prepared vLLM 0.31 / ATOM environment.
Use only your allocated GPUs and unused ports. Keep the source unchanged until
both the model and the MP server have stopped.

Start the MP server first:

```bash
HIP_VISIBLE_DEVICES=0,1,2,3 LMCACHE_DISABLE_BANNER=1 lmcache server \
  --host localhost --port 5561 --chunk-size 256 \
  --l1-size-gb 96 --eviction-policy LRU \
  --supported-transfer-mode auto --shm-name dsv41mp_5561 \
  --no-l1-use-lazy --separate-object-groups --max-gpu-workers 4 \
  --http-host 127.0.0.1 --http-port 8086 --prometheus-port 9006
```

- `--separate-object-groups` is required: the checkpoint groups are restored
  as a one-chunk window, which needs object groups of their own.
- `--no-l1-use-lazy` is required, or `--shm-name` silently falls back to
  pickled transfers.
- `--chunk-size` must be a multiple of the 256-token PAGE block and equal to
  `ATOM_STATE_CHECKPOINT_INTERVAL_TOKENS`; a save carries one checkpoint, at
  the chunk boundary it ends on. 256 is the measured default (see results).

Start the model:

```bash
export MODEL=/data/amd_int/models/DeepSeek-V4.1-Flash
export HIP_VISIBLE_DEVICES=0,1,2,3
export AITER_LOG_LEVEL=WARNING VLLM_WORKER_MULTIPROC_METHOD=spawn PYTHONHASHSEED=0
export ATOM_V41_EXPERIMENTAL_CUDAGRAPH=1 VLLM_USE_BREAKABLE_CUDAGRAPH=1
export ATOM_V41_STATE_CHECKPOINTS=1 ATOM_STATE_CHECKPOINT_INTERVAL_TOKENS=256
export OFFLOAD_MIN_LOAD_TOKENS=256

vllm serve "$MODEL" \
  --served-model-name DeepSeek-V4.1-Flash --host 127.0.0.1 --port 8021 \
  --dtype auto --kv-cache-dtype auto \
  --tensor-parallel-size 4 --distributed-executor-backend mp \
  --trust-remote-code --tokenizer-mode deepseek_v4 \
  --gpu-memory-utilization 0.78 --block-size 256 \
  --max-model-len 8192 --max-num-seqs 32 --max-num-batched-tokens 8192 \
  --enable-prefix-caching --mamba-cache-mode align --no-async-scheduling \
  --speculative-config '{"method":"dspark","num_speculative_tokens":5,"kv_cache_dtype":"fp8_ds_mla"}' \
  --kv-transfer-config '{"kv_connector":"AtomLMCacheOffloadConnector",
    "kv_connector_module_path":"atom.plugin.vllm.kv_transfer.connector",
    "kv_role":"kv_both","kv_load_failure_policy":"recompute",
    "kv_connector_extra_config":{"atom.offload.backend":"mp",
      "lmcache.mp.host":"tcp://localhost","lmcache.mp.port":5561,
      "lmcache.mp.tp_rank_collapse":false}}'
```

- Drop `--speculative-config` for the non-speculative configuration.
- `ATOM_V41_STATE_CHECKPOINTS=1`, `--enable-prefix-caching` and
  `--mamba-cache-mode align` together register the checkpoint groups.
- `ATOM_V41_CHECKPOINT_LOCAL_HITS=1` additionally lets vLLM serve prefix hits
  from HBM (restored from a cached checkpoint) before the tier is asked. It is
  off by default; see the results for when it pays.
- The OFF arm of every comparison below is the same command without
  `--kv-transfer-config`, with `--no-enable-prefix-caching` and
  `ATOM_V41_STATE_CHECKPOINTS=0`: no prefix reuse at all.

## Test method

### Accuracy: two passes over one server

GSM8K test (1319 questions), 5-shot, `lm_eval --model local-completions`,
`num_concurrent=32`, default generation budget, against a fresh MP server.
Pass 1 fills the tier; pass 2 repeats the identical prompts, which are then
served from the tier (and, with local hits on, from HBM). Pass 2 is the
measurement, pass 1 the control on the same engine. Record
`vllm:external_prefix_cache_hits_total` and `vllm:prefix_cache_hits_total`
before and after each pass: a pass 2 whose tier and HBM hits are both zero
measured nothing. `--max-model-len 4096`, `--max-num-seqs 8`, HBM pool
`--num-gpu-blocks-override 950` (so pass 1's prefixes are evicted from HBM
before pass 2) unless stated.

### Performance: aiperf, ON vs OFF

```bash
aiperf profile --model DeepSeek-V4.1-Flash --tokenizer "$MODEL" \
  --url http://127.0.0.1:8021 --endpoint-type completions --streaming \
  --concurrency 16 --benchmark-duration 180 --benchmark-grace-period 60 \
  --random-seed 5201 \
  --synthetic-input-tokens-mean 3000 --synthetic-input-tokens-stddev 0 \
  --output-tokens-mean 64 --output-tokens-stddev 0 --extra-inputs ignore_eos:true \
  --prompt-prefix-length 4096 --prompt-prefix-pool-size 4 \
  --use-server-token-count
```

Mean input 7,097 tokens per request: one of four 4,096-token shared prefixes
plus a 3,000-token tail. aiperf's synthetic set is 100 prompts, replayed for
the whole window, so the tails repeat too and the reusable share approaches
the whole prompt. Each arm gets a fresh MP server and a fresh model process,
`--num-gpu-blocks-override 4000`, `--max-num-seqs 32`,
`--max-num-batched-tokens 8192`; the source share (HBM / tier / recomputed)
is taken from the metric deltas over the measurement window.

## Test results

Measured on MI355X, TP4, vLLM `0.31.1.dev0+gdb9527a46.d20261008`, LMCache
`0.5.5rc3+rocm7.2.4.torch2.10.git3d3aa833.cxx11abi1`, V4.1 cudagraph on
(`ATOM_V41_EXPERIMENTAL_CUDAGRAPH=1`). The commit each row ran on is given;
rows at an earlier commit exercise code the later commits do not change for
that configuration.

### Accuracy (GSM8K test 1319, 5-shot, strict match)

Pass-2 source columns are the share of pass-2 prompt tokens served from the
tier and from HBM local hits. Every run: zero engine faults, zero
`STORE block ID underflow` lines in the tier log.

| Configuration | Commit | Pass 1 | Pass 2 | Pass 2 from tier | Pass 2 from HBM |
|---|---|---:|---:|---:|---:|
| DSpark, offload OFF, no prefix reuse | `404de3216` | 0.9257 | 0.9204 | – | – |
| DSpark | `cd24c85b5` | 0.9257 | 0.9249 | 84.3% | 0.0% |
| DSpark | `444d764cf` | 0.9303 | 0.9287 | 84.3% | 0.0% |
| DSpark, local hits on | `444d764cf` | 0.9333 | 0.9356 | 84.3% | 0.0% |
| DSpark, local hits on, pool 20000 | `444d764cf` | 0.9303 | 0.9303 | 76.9% | 7.8% |
| DSpark, local hits on, pool 3000 | `444d764cf` | 0.9303 | 0.9204 | 84.3% | 0.0% |
| No spec, local hits on | `444d764cf` | 0.9227 | 0.9303 | 84.8% | 0.0% |
| No spec, local hits on, pool 20000 | `444d764cf` | 0.9242 | 0.9310 | 0.0% | 85.1% |
| No spec, chunk 1024 | `444d764cf` | 0.9242 | 0.9272 | 14.9% | 0.0% |

All passes fall within 0.920–0.936. The three restore paths are each
exercised: tier only, HBM local hit only (pool 20000, no spec), and a local
hit with the tier supplying the rest (pool 20000, DSpark: 122 requests). At
chunk 1024 only prompts longer than 1024 tokens can hit, hence 14.9%.

### Performance, HBM pool 4000 blocks (the working set fits)

| Configuration | Commit | req/s | TTFT avg / p90 (ms) | ITL avg / p90 (ms) | HBM / tier / recomputed |
|---|---|---:|---:|---:|---:|
| No spec, OFF | `444d764cf` | 4.72 | 703 / 771 | 42.5 / 47.2 | – |
| No spec, ON | `cd24c85b5` | 5.98 | 834 / 1235 | 29.1 / 33.3 | 0 / 92.1 / 7.9% |
| No spec, ON + local hits | `d27413237` | **13.68** | 289 / 298 | 13.9 / 14.8 | 95.6 / 0 / 4.4% |
| DSpark, OFF | `404de3216` | 4.63 | 369 / 579 | 48.8 / 57.5 | – |
| DSpark, ON | `d27413237` | 6.65 | 563 / 753 | 29.1 / 31.4 | 0 / 93.3 / 6.7% |
| DSpark, ON + local hits | `d27413237` | 6.79 | 534 / 703 | 28.8 / 31.3 | 13.0 / 80.8 / 6.2% |

Against no reuse at all, the tier gives +27% throughput without speculation
and +44% with DSpark, mostly through ITL (−32% / −40%): the prefill it
replaces no longer shares steps with decode. TTFT does not improve, because a
tier hit parks the request for its load (about 90 ms server-side per load,
nearly independent of its size), and steps taking about 80 ms occur at the
rate loads complete, shortly after each (per-step timing against the tier's
retrieve log; the cause is not yet isolated).

With local hits on and the working set in HBM, prefix reuse comes from HBM and
the tier is idle (no spec, 13.68 req/s). Under DSpark, HBM local hits stay at
13%: a local hit needs every KV cache group to hold the prefix, including the
draft's.

### Performance, HBM pool 1500 blocks (the working set does not fit), local hits on, no spec

| Configuration | Commit | req/s | TTFT avg (ms) | ITL avg (ms) | HBM / tier / recomputed |
|---|---|---:|---:|---:|---:|
| HBM prefix caching only (no `--kv-transfer-config`) | `444d764cf` | **7.88** | 779 | 19.7 | 57.4 / – / 42.6% |
| HBM prefix caching + LMCache | `cd24c85b5` | 6.14 | 740 | 29.5 | 8.2 / 85.3 / 6.5% |

**Known limitation.** When HBM can hold the shared prefixes but not the
per-request tails, enabling the tier is currently slower than HBM prefix
caching alone. Two measured causes: a load restores the checkpoint at its own
end only, so once a shared prefix's checkpoint is evicted, requests load past
it and HBM local hits collapse (57% → 8%); and the ~80 ms steps that come
with each load raise ITL from 19.7 to 29.5 ms. Prefer HBM prefix
caching without the tier for such workloads until these are addressed.

### Unit tests

`tests/plugin` plus the offload, LMCache MP and V4.1 suites, on CPU in the
serving container: 1740 passed, 16 failed. The 16 fail identically on the
previous head `404de3216` and lie outside this change
(`test_vllm_deepseek_v4_proxy_state_arena_layout.py`,
`test_vllm_kimi_k3.py::test_atom_patch_preserves_rocm_dcp_full_decode_cuda_graph_mode`,
`test_vllm_kv_cache_layout.py::test_non_block_major_scales_are_rejected`,
`test_vllm_mha_backend.py::test_duck_typed_backends_keep_vllm_028_kv_spec`).

