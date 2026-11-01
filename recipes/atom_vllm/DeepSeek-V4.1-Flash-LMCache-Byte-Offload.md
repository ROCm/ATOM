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
  multiple of 256) and the client's LMCache chunk size (`LMCACHE_CHUNK_SIZE`,
  default 256). `--separate-object-groups` and `--no-l1-use-lazy` are
  required. `OFFLOAD_MIN_LOAD_TOKENS` (default 8192) is the shortest tier
  load worth making. Give the MP server the model's `HIP_VISIBLE_DEVICES`:
  its GPU workers open the model's buffers by local device index.
- `ATOM_V41_STATE_CHECKPOINTS=1` with `--enable-prefix-caching` and
  `--mamba-cache-mode align` registers the checkpoint groups. Without them the
  connector refuses to start.
- `max_cudagraph_capture_size` 512: the vLLM default is 2 × `max-num-seqs`
  (64), so the step after a tier load (a prefill tail of a few hundred tokens
  plus the decode rows) would run eager.
- Drop `--speculative-config` to run without speculative decoding. With it,
  keep `--no-async-scheduling`; the checkpoints refuse async speculative
  scheduling.
- Prefix hits are served from HBM first, at a boundary whose checkpoint is
  cached, and the tier is asked only for the rest.
- HBM-only prefix caching: the same command without `--kv-transfer-config`.
- No prefix reuse (as on main): the same command without
  `--kv-transfer-config`, with `--no-enable-prefix-caching`, without
  `--mamba-cache-mode align`, and with `ATOM_V41_STATE_CHECKPOINTS=0`.

## Accuracy

Method: run GSM8K twice against one server and one fresh MP server. Pass 1
fills the tier, and pass 2 repeats the same prompts. Settings:
`--max-num-seqs 8 --num-gpu-blocks-override 950`, so pass 1's tails are
evicted from HBM before pass 2. Two prompt sets:

- **Tier only**: plain 5-shot prompts with `--max-model-len 4096`. Every
  question has its own exemplars, so pass 2 reloads each prompt from the tier.
- **HBM + tier**: the same prompts behind a shared 3842-token preamble
  (GSM8K train examples 200–223, in the task's `Question: … Answer: …`
  format), passed as `--system_instruction`, with `--max-model-len 8192`. The
  preamble stays hot in HBM and the rest of each prompt loads from the tier.

```bash
lm_eval --model local-completions --tasks gsm8k --num_fewshot 5 \
  --model_args model=DeepSeek-V4.1-Flash,base_url=http://127.0.0.1:8021/v1/completions,num_concurrent=32,tokenized_requests=False,tokenizer_backend=none
# HBM + tier: add  --system_instruction "$(cat preamble.txt)"
```

| Configuration | Pass 1 | Pass 2 | Pass-2 prompt tokens: HBM / tier |
|---|---:|---:|---:|
| DSpark, tier only | 0.9272 | 0.9303 | 0 / 84.3% |
| No spec, tier only | 0.9295 | 0.9265 | 0 / 84.8% |
| DSpark, HBM + tier | 0.9151 | 0.9158 | 21.8 / 74.7% |
| No spec, HBM + tier | 0.9181 | 0.9128 | 19.5 / 77.0% |
| DSpark, preamble, LMCache off (control) | 0.9227 | 0.9204 | – |

Zero engine faults and zero `STORE block ID underflow` lines. Pass-to-pass
differences range from −0.5 to +0.3 pp; one standard error at n = 1319 is
about 0.7 pp, and the control without any reuse moved −0.2 pp. The preamble
lowers both passes by about 1 pp, with or without LMCache.

## Performance

Method: aiperf at concurrency 16 for 180 s. Each prompt is one of four
4096-token shared prefixes plus a 3000-token tail (7097 tokens on average),
with 64 output tokens. aiperf replays its 100 synthetic prompts, so the tails
repeat too. Each run uses `--num-gpu-blocks-override 4000` and a fresh model
and MP server. The HBM, tier and recomputed shares come from vLLM's
`prefix_cache_hits` and `external_prefix_cache_hits` counters over the
measurement window.

```bash
aiperf profile --model DeepSeek-V4.1-Flash --tokenizer $MODEL \
  --url http://127.0.0.1:8021 --endpoint-type completions --streaming \
  --concurrency 16 --benchmark-duration 180 --benchmark-grace-period 60 \
  --random-seed 5201 --synthetic-input-tokens-mean 3000 \
  --synthetic-input-tokens-stddev 0 --output-tokens-mean 64 \
  --output-tokens-stddev 0 --extra-inputs ignore_eos:true \
  --prompt-prefix-length 4096 --prompt-prefix-pool-size 4 --use-server-token-count
```

Results (MI355X, TP4, FULL_AND_PIECEWISE, DSpark):

| Configuration | req/s | TTFT avg (ms) | ITL avg (ms) | HBM / tier / recomputed |
|---|---:|---:|---:|---|
| No prefix reuse | 4.62 | 361 | 49.1 | 0 / 0 / 100% |
| HBM prefix caching, LMCache off | 6.90 | 286 | 32.2 | 57.3 / 0 / 42.7% |
| HBM prefix caching + LMCache | **7.80** (+13%) | 387 | 26.3 | 11.4 / 82.9 / 5.7% |

Under DSpark, HBM alone keeps only the shared prefixes (57.3%). The tier
supplies the tails, which cuts recompute from 42.7% to 5.7%.

Theoretical vs measured (DSpark):

| Quantity | Theoretical | Measured |
|---|---:|---:|
| Prefix hit rate | 97.4% (6912 of 7097 tokens on the 256 grid) | 94.3% |
| LMCache throughput | 10.11 req/s at zero transfer cost[^model] | 7.80 req/s (77%) |
| One load (21.3 MiB per rank: PAGE + a 5.0 MiB checkpoint) | 0.45 ms at 50 GB/s H2D | 40 ms p50 server-side retrieve |

[^model]: Per-request engine time is `t(c) = 17.6 us × c + 91.7 ms`, where `c`
is the recomputed prompt tokens. It is calibrated on the two rows above that
move no bytes and evaluated at 408 recomputed tokens.

The gap is per-load latency, not bandwidth: TTFT is 101 ms above HBM-only,
while ITL is 6 ms lower.

## Known limitations

- **The working set fits in HBM (no spec, pool 4000).** The tier is idle
  (HBM 95.7%), and attaching LMCache costs about 4% (14.00 vs 14.55 req/s).
- **Tier loads restore only the checkpoint at their own end.** Once a shared
  prefix's checkpoint is evicted from HBM, requests load past it: 1125 of the
  1311 loads above started at token 0.
- **DSpark.** V4.1 cannot run a prefill in the same step as a speculative
  verify, so a request resumed from the tier prefills its tail in a step of
  its own.
