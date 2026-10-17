# DeepSeek-V4.1-Flash: LMCache offload (vLLM plugin, MP server)

DeepSeek-V4.1-Flash offloads prefix KV to an LMCache **MP server**. Its CSA2
per-request state (window ring, compressor rings and Engram cursor) is
checkpointed at every 256-token PAGE boundary. The MP worker stores each
checkpoint with the PAGE KV that ends at the same boundary, so a tier hit
restores both. Supported with FULL_AND_PIECEWISE cudagraph and DSpark. There
is no in-process LMCache path for V4.1.

## Launch

MP server (start it first):

```bash
HIP_VISIBLE_DEVICES=0,1,2,3 lmcache server --host localhost --port 5561 \
  --chunk-size 256 --l1-size-gb 96 --eviction-policy LRU \
  --supported-transfer-mode auto --shm-name dsv41mp_5561 \
  --no-l1-use-lazy --separate-object-groups --max-gpu-workers 4
```

Model:

```bash
export MODEL=/data/amd_int/models/DeepSeek-V4.1-Flash
export HIP_VISIBLE_DEVICES=0,1,2,3 AITER_LOG_LEVEL=WARNING
export ATOM_V41_EXPERIMENTAL_CUDAGRAPH=1 VLLM_USE_BREAKABLE_CUDAGRAPH=1
export ATOM_V41_STATE_CHECKPOINTS=1 ATOM_STATE_CHECKPOINT_INTERVAL_TOKENS=256
export OFFLOAD_MIN_LOAD_TOKENS=256 VLLM_WORKER_MULTIPROC_METHOD=spawn PYTHONHASHSEED=0

vllm serve $MODEL --served-model-name DeepSeek-V4.1-Flash --port 8021 \
  --tensor-parallel-size 4 --trust-remote-code --tokenizer-mode deepseek_v4 \
  --gpu-memory-utilization 0.78 --block-size 256 \
  --max-model-len 8192 --max-num-seqs 32 --max-num-batched-tokens 8192 \
  --enable-prefix-caching --mamba-cache-mode align --no-async-scheduling \
  --compilation-config '{"max_cudagraph_capture_size":512}' \
  --speculative-config '{"method":"dspark","num_speculative_tokens":5,"kv_cache_dtype":"fp8_ds_mla"}' \
  --kv-transfer-config '{"kv_connector":"AtomLMCacheOffloadConnector",
    "kv_connector_module_path":"atom.plugin.vllm.kv_transfer.connector",
    "kv_role":"kv_both","kv_load_failure_policy":"recompute",
    "kv_connector_extra_config":{"atom.offload.backend":"mp",
      "lmcache.mp.host":"tcp://localhost","lmcache.mp.port":5561,
      "lmcache.mp.tp_rank_collapse":false}}'
```

- `--chunk-size` must equal `ATOM_STATE_CHECKPOINT_INTERVAL_TOKENS` (a
  multiple of 256). `--separate-object-groups` and `--no-l1-use-lazy` are
  required. `OFFLOAD_MIN_LOAD_TOKENS` (default 8192) is the shortest tier
  load worth making.
- `ATOM_V41_STATE_CHECKPOINTS=1` with `--enable-prefix-caching` and
  `--mamba-cache-mode align` registers the checkpoint groups. Without them the
  connector refuses to start.
- `max_cudagraph_capture_size` 512: the vLLM default is 2 × `max-num-seqs`
  (64), so the step after a tier load (a prefill tail of a few hundred tokens
  plus the decode rows) would run eager.
- Drop `--speculative-config` to run without speculative decoding.
  `ATOM_V41_CHECKPOINT_LOCAL_HITS=1` also serves prefix hits from HBM
  checkpoints. It is off by default.
- HBM-only prefix caching: the same command without `--kv-transfer-config`.
- OFF baseline: the same command without `--kv-transfer-config`, with
  `--no-enable-prefix-caching`, without `--mamba-cache-mode align`, and with
  `ATOM_V41_STATE_CHECKPOINTS=0`.

## Accuracy

Method: run GSM8K twice against one server and one fresh MP server. Pass 1
fills the tier. Pass 2 repeats the same prompts, which then load from the
tier. `--max-model-len 4096 --max-num-seqs 8 --num-gpu-blocks-override 950`,
so pass 1's prefixes are evicted from HBM before pass 2.

```bash
lm_eval --model local-completions --tasks gsm8k --num_fewshot 5 \
  --model_args model=DeepSeek-V4.1-Flash,base_url=http://127.0.0.1:8021/v1/completions,num_concurrent=32,tokenized_requests=False,tokenizer_backend=none
```

| Configuration | Pass 1 | Pass 2 | Pass-2 prompt tokens from tier |
|---|---:|---:|---:|
| DSpark | 0.9257 | 0.9340 | 84.3% |
| No speculative decoding | 0.9265 | 0.9325 | 84.8% |

Zero engine faults and zero `STORE block ID underflow` lines in the tier log.

## Performance

Method: aiperf at concurrency 16, 180 s. Each prompt is one of four
4096-token shared prefixes plus a 3000-token tail (7097 tokens on average)
with 64 output tokens. aiperf replays its 100 synthetic prompts, so the tails
repeat as well. Each arm gets a fresh model and a fresh MP server, with
`--num-gpu-blocks-override 4000`. The prefix-hit shares are vLLM's
`prefix_cache_hits` (HBM) and `external_prefix_cache_hits` (tier) counters
over the measurement window.

```bash
aiperf profile --model DeepSeek-V4.1-Flash --tokenizer $MODEL \
  --url http://127.0.0.1:8021 --endpoint-type completions --streaming \
  --concurrency 16 --benchmark-duration 180 --benchmark-grace-period 60 \
  --random-seed 5201 --synthetic-input-tokens-mean 3000 \
  --synthetic-input-tokens-stddev 0 --output-tokens-mean 64 \
  --output-tokens-stddev 0 --extra-inputs ignore_eos:true \
  --prompt-prefix-length 4096 --prompt-prefix-pool-size 4 --use-server-token-count
```

Results (MI355X, TP4, FULL_AND_PIECEWISE; HBM / tier / recomputed are
shares of prompt tokens):

| Configuration | req/s | TTFT avg (ms) | ITL avg (ms) | HBM / tier / recomputed |
|---|---:|---:|---:|---|
| No spec, OFF | 4.82 | 699 | 41.4 | 0 / 0 / 100% |
| No spec, LMCache | **7.49** (+55%) | 524 | 25.5 | 0 / 92.5 / 7.5% |
| No spec, LMCache + `ATOM_V41_CHECKPOINT_LOCAL_HITS=1` | 14.39 | 144 | 15.3 | 95.7 / 0 / 4.3% |
| No spec, HBM prefix caching only (checkpoints, no `--kv-transfer-config`) | 14.82 | 136 | 14.9 | 95.8 / 0 / 4.2% |
| DSpark, OFF | 4.66 | 360 | 48.6 | 0 / 0 / 100% |
| DSpark, LMCache | **6.56** (+41%) | 595 | 29.1 | 0 / 92.8 / 7.2% |

Theoretical vs measured (no spec):

| Quantity | Theoretical | Measured |
|---|---:|---:|
| Prefix hit rate | 97.4% in steady state (6912 of 7097 tokens on the 256 grid); ≈94% in this window, where each of the 100 tails misses once | 92.5% tier |
| LMCache throughput | 13.8 req/s at zero transfer cost[^model] | 7.49 req/s (54%) |
| Per-load transfer (23.4 MiB per rank: PAGE + one 5.0 MiB checkpoint) | 0.49 ms at 50 GB/s H2D | 84 ms p50 server-side retrieve |

[^model]: Per-request engine time `t(c) = 20.6 us × c + 61 ms`, where `c` is
the recomputed prompt tokens, calibrated on the OFF and HBM-only rows (which
move no bytes), evaluated at the LMCache row's 530 recomputed tokens.

The hit rate is close to its ceiling. The throughput gap is per-request
latency, not bandwidth. A load parks the request for a near-constant ~85 ms
whatever its size, and the TTFT is +388 ms against HBM-only. The rest of the
gap is ITL (+10.6 ms).

## Known limitations

- **When the working set fits in HBM, HBM prefix caching alone is faster.**
  Use the launch command without `--kv-transfer-config` (14.82 req/s
  above). The checkpoints alone make V4.1 prefix caching correct.
- **HBM pool too small for the per-request tails** (`--num-gpu-blocks-override
  1500`, no spec, local hits on). Compared with HBM-only, LMCache cuts
  recompute from 42.6% to 5.7% and TTFT from 769 to 462 ms, but throughput is
  8.19 → 7.76 req/s (−5%) because ITL rises from 18.7 to 25.3 ms. Separately,
  a tier load restores the checkpoint at its own end only. Once a shared
  prefix's checkpoint is evicted, requests load past it, and HBM local hits
  fall from 57.4% to 6.4%.
- **DSpark.** V4.1 cannot run a prefill in the same step as a speculative
  verify, so a request resumed from the tier prefills its tail in a step of
  its own. An HBM local hit needs every KV cache group to hold the prefix,
  including the draft's, so local hits stay low.
