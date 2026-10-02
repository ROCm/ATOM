# DeepSeek-V4.1-Flash — LMCache byte offload on the vLLM plugin

V4.1 buys its cache in two currencies. PAGE bytes scale with history; STATE
bytes do not -- every in-flight request owns a fixed 5.03 MiB entry holding a
window ring per layer, the compressor's incomplete group, and the Engram
cursor. `PagedAttentionCache` refuses any request whose cursor is not exactly
the frontier the scheduler claims, so a PAGE prefix restored without its STATE
is not a degraded answer, it is a dead engine. The two therefore travel
together or not at all, and a hit is capped to the last boundary whose STATE
image exists.

## Serving

```bash
vllm serve /data/amd_int/models/DeepSeek-V4.1-Flash \
  --served-model-name DeepSeek-V4.1-Flash \
  --tensor-parallel-size 4 --distributed-executor-backend mp \
  --trust-remote-code --tokenizer-mode deepseek_v4 \
  --gpu-memory-utilization 0.9 \
  --max-model-len 32768 --max-num-seqs 64 --max-num-batched-tokens 4096 \
  --no-enable-prefix-caching \
  --kv-transfer-config '{"kv_connector":"AtomLMCacheOffloadConnector",
    "kv_connector_module_path":"atom.plugin.vllm.kv_transfer.connector",
    "kv_role":"kv_both","kv_load_failure_policy":"recompute",
    "kv_connector_extra_config":{"atom.offload.backend":"inproc",
      "atom.offload.v41.state_interval":4096}}'
```

Environment, on **both** arms of any comparison:

```bash
export PYTHONHASHSEED=0
export LMCACHE_LOCAL_CPU=True LMCACHE_MAX_LOCAL_CPU_SIZE=16   # GiB per rank
export LMCACHE_CHUNK_SIZE=256 LMCACHE_CACHE_POLICY=ATOM_SLRU
export OFFLOAD_MIN_LOAD_TOKENS=256
```

### Three settings that silently disable the tier

| setting | what goes wrong |
|---|---|
| `state_interval >= max_model_len` | `cap_hit`'s boundary is `floor(hit/interval)*interval`, so every boundary is 0 and every hit is declined. The tier is dead **by construction** and reads exactly like a wiring fault. One line of arithmetic decides it; do not spend GPU on it. |
| `OFFLOAD_MIN_LOAD_TOKENS` (default **8192**) | A prompt shorter than this has every load declined by the admission gate *before* the tier is asked. Queries climb, hits stay at zero, and nothing in the log says why. |
| `max-num-batched-tokens != state_interval` | A prefill chunk larger than the interval steps over boundaries instead of landing on them. `boundary_passed` counts it. |

`--no-enable-prefix-caching` is deliberate, not a limitation. vLLM builds the
block hasher when `enable_prefix_caching` **or** a KV connector is configured,
and calls `get_num_new_matched_tokens` on the same condition, so the connector
keeps its full key space while vLLM's own pool serves nothing locally -- which
is what makes every hit pass through `cap_hit`.

## Verifying it is actually on

```bash
grep -c "V4.1 state tier up" server.log      # one per worker (4 at TP4)
grep -c "V4.1 state leg on"  server.log      # one, from EngineCore
grep -o "PAGE unit = .*"     server.log | head -1
```

For the BF16 pool the registration arithmetic must close exactly:

```
PAGE unit = 739,840 B/block over 5 planes [655360, 16896, 16896, 16896, 33792]
          = 2,890.0 B/token ;  STATE entry = 5,276,672 B
```

`Σ planes == B/block == geometry.paged_bytes`, and `num_blocks` is the pool's
page count -- **not** the proxy tensor's leading dimension, which is larger by
the withheld STATE tail (measured: 250,258 vs 249,801, i.e. 457 blocks =
337.7 MB = 64 slots x 5,276,672 B).

A hit's size is the discriminator: it is always a multiple of
`state_interval`, because that is where a STATE image exists.

## Measured

TP4, gfx950, BF16 pool, `--max-model-len 32768`, tier 16 GiB/rank,
`state_interval 4096`. Prompts: 16,384-token shared prefix (pool of 4) +
4,000-token suffix, 8 output tokens, concurrency 1, 120 s windows, aiperf.
The two arms differ **only** in whether `--kv-transfer-config` is passed.

| | OFF | ON | delta |
|---|---|---|---|
| req/s | 0.995 | 1.431 | **+43.7 %** |
| TTFT p50 | 596.2 ms | 202.8 ms | **−66.0 %** |
| request latency p50 | 1002.6 ms | 664.7 ms | −33.7 % |
| ITL p50 | 58.0 ms | 66.0 ms | +13.8 % |
| requests in window | 120 | 172 | |
| tier supplied | 0 | **77.6 %** of prompt tokens | |

Run-to-run variance, from a second window per arm: OFF 0.5 %, ON 1.5 % -- the
effect is ~30x that. The ON window above starts on a cold tier, so it is the
conservative one (a warm repeat read 1.452 req/s, +45.8 %).

**The output length is load-bearing and is why it is stated first.** Offload
removes prefill work and no decode work, so the ceiling is Amdahl's. The same
tier on the same prompts with 128 output tokens is capped near **+6 %**,
because decode is then 92 % of request latency -- measured: OFF 0.712 req/s,
TTFT p50 742 ms, ITL p50 80.1 ms. Quote this result only with its output
length and prefix share attached.

ITL is **worse** with the tier on, consistently ~14 %: the load threads
contend with decode for the GIL. It is a real cost, not noise.

## Accuracy

Two-pass marker recall (`tools/kv_offload_twopass_check.py`, 8 prompts of
~15,000 tokens, same seed on both arms):

| | ON | OFF |
|---|---|---|
| marker recall, pass 2 | **8/8** | **8/8** |
| tier supplied | 82 % of each prompt | 0 (`external_queries` = 0) |

The ON arm scores full marks *while* 82 % of every prompt is restored from the
tier, so the comparison has a positive control: it is not measuring
recomputation. `identical_text` is not a judgement -- the engine is not
bit-reproducible under concurrency.

## Known issue: concurrency

**The engine dies at concurrency 8** with
`ValueError: Request ... needs state at N, found N+1; replay from a
recoverable boundary`. Concurrency 1 is clean over full 120 s windows with the
tier actively storing and retrieving.

Bisected: setting `state_interval` past the context length makes the STATE leg
inert by construction (`cap_hit` declines everything, the sweep never fires,
`Retrieved` is 0) **and the crash still happens**. So the fault is in the
connector/PAGE path with a connector attached, not in the STATE leg. The OFF
arm runs the identical aiperf configuration at concurrency 8 for 120 s without
incident.

Two defects *were* found and fixed on the way, both of which put two requests
on one state slot (signature: two requests straddling one cursor by a token
each): `reserve` counted every key in the slot table as live, so a full table
sent `_acquire` to its slot-0 fallback; and `assign` evicted reserved slots,
which are absent from its batch by definition. A reservation can now also be
refused -- `batch + reservations` may exceed the pool even though vLLM caps
concurrency at its size -- and a refused restore recomputes instead.
