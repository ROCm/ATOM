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

## Prefix caching

Admitted only alongside this connector, and even then it buys nothing. Enable
it and vLLM's block pool will offer local hits that the connector never sees;
those are **refused**, not shortened, and all reuse continues through the
tier.

Shortening them is the tempting mistake. A local hit capped to a boundary the
index holds still has nobody to restore that boundary's STATE --
`resolve_load` runs only for tokens the connector supplied -- so the request
arrives with its pages in HBM and a slot nobody wrote. Measured as
`needs state at 8192, found 0`: the cap picked 8192 and no one filled it.
PAGE reuse and STATE restore are one operation here, and only the connector
performs both.

Measured cost of enabling it, same window and workload as the table above:

| | req/s | TTFT p50 |
|---|---|---|
| prefix caching off | 2.206 | 2913 ms |
| prefix caching on | 2.139 | 3036 ms |

**−3.1 %.** vLLM does the hashing and bookkeeping; the hits it produces are
declined. `prefix_cache_hits_total` reads 0 and
`external_prefix_cache_hits_total` carries the whole workload. Leave it off
unless something else in the deployment needs it.

Making a local hit useful means giving it a STATE restore of its own -- the
request would have to park for it, the way a connector load does. That is not
implemented.

The cap is installed as a patch on `Scheduler._get_local_prefix_cache_hit`
from `register_model`, not by selecting a scheduler subclass from the
platform hook: measured on a V4.1 serve, `ATOMPlatform.check_and_update_config`
ran **zero** times while the model wrapper's own hook ran four. A scheduler
chosen there is a scheduler never chosen, and the symptom is prefix caching
coming up uncapped -- the dead engine the cap exists to prevent, reintroduced
by where it was installed.

## Why caching is off by default here

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

### The same tier at three working points

| concurrency | output tokens | prefix share | req/s OFF -> ON | delta |
|---|---|---|---|---|
| 1 | 8 | 16,384 / 20,384 | 0.995 -> 1.431 | **+43.7 %** |
| 8 | 8 | 16,384 / 20,384 | 1.880 -> 2.207 | **+17.1 %** |
| 8 | 128 | 8,192 / 20,192 | 0.712 -> 0.689 | **−3.3 %** |

One implementation, three answers. The third is Amdahl: with 128 output
tokens decode is 92 % of request latency, the tier supplied 10.6 % of prompt
tokens, and the ~3 % ITL cost is larger than what is left to win.

The second is a scheduling effect worth naming, because it is the lever for
raising it. A boundary only exists where the frontier *lands*, and vLLM
spends one shared token budget across the step: with
`max-num-batched-tokens 4096` and six decodes in the batch, the prefill
request is scheduled 4,090 tokens, not 4,096, so the frontier lands at an
arbitrary offset and steps over the 4,096-token grid. Measured over that
window: `boundary_passed` 1,315 against `sweep_offered` 106, and
`cap_declined` 248 against `cap_kept` 102 -- seven hits in ten are refused
for want of a claimed boundary. The hits that do happen come from decode,
where the frontier advances one token at a time and eventually lands exactly.

Raising concurrency therefore does not reduce what the tier *can* do, it
reduces how often a boundary is reachable. Aligning a prefill chunk to the
interval is a scheduler-side change and is the next lever; `boundary_passed`
is the number to watch.

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

Characterised by two arms that differ in one variable, with the STATE leg
structurally inert in **both**:

| `state_interval` | `max-num-batched-tokens` | chunked prefill | result |
|---|---|---|---|
| 2^30 (leg inert) | 4096 | on | **crash** |
| 4096 | 32768 (= `max-model-len`) | off | 124.8 s clean |

So the necessary condition is **chunked prefill x KV connector x
concurrency**, and the tier's own activity is irrelevant -- it was doing
nothing in either arm. The OFF arm runs the identical aiperf configuration at
concurrency 8 for 120 s without incident, and concurrency 1 is clean with the
tier actively storing and retrieving.

The second row is **not a workaround**: with one-shot prefill the frontier
jumps straight from 0 to the prompt length, which is not a multiple of the
interval, so no boundary is ever landed on and the tier supplies nothing
(measured: `external_prefix_cache_hits_total` 0.0 against 1.45 M queries, and
0.577 req/s -- *slower* than OFF). The STATE tier needs chunked prefill to
reach a boundary during prefill, which is exactly what triggers the fault.

Two hypotheses were tested and refuted, both cheaply:

* The missing non-immediate-block-reuse patch (V4.1's layer name does not
  match V4's substring markers). Real defect, fixed, **not** this one -- the
  crash survives the fix.
* Preemption advancing the cursor optimistically. The crash dump reads
  `preempted_requests=0`.

The failing step is a mixed batch -- one request on a 4,090-token prefill
chunk, six on 1-token decode -- and the dumped `num_computed_tokens` contains
the cursor's value but not the position the scheduler asked for. Fixing it
means going into `_v41_scheduled_batch` / `_prepare` in the bridge, not the
offload layer.

Two defects *were* found and fixed on the way, both of which put two requests
on one state slot (signature: two requests straddling one cursor by a token
each): `reserve` counted every key in the slot table as live, so a full table
sent `_acquire` to its slot-0 fallback; and `assign` evicted reserved slots,
which are absent from its batch by definition. A reservation can now also be
refused -- `batch + reservations` may exceed the pool even though vLLM caps
concurrency at its size -- and a refused restore recomputes instead.
