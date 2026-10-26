# GLM-5.3 — LMCache KV offload on the vLLM plugin (byte codec)

GLM-5.3 (`GlmMoeDsaForCausalLM`) runs on `AtomLMCacheOffloadConnector`
unchanged. **No GLM-5.3-specific code exists, and none is wanted** — the
mapping added for GLM-5.2 in #2231 folds `<p>.indexer.k_cache` onto `<p>.attn`
by registered layer *name* and is never told which model it has, and GLM-5.3
registers the identical entries: 78 MLA layers at 47,700 B per token per rank,
with the 21 DSA indexers folded in rather than counted separately.

This recipe is the operational one: the server line, the client line, and what
they measured. For the mechanism, the tuning sweep and the full gotcha list,
read [GLM-5.2 LMCache Byte Offload](GLM-5.2-LMCache-Byte-Offload.md) — all of
it applies here verbatim.

Both checkpoints serve: `amd/GLM-5.3-MXFP4` and `amd/GLM-5.3-FP8`. Quantization
is a weight-side choice and does not reach the KV cache, which both declare as
fp8; the FP8 arm differs only in its `online_quant_config`, which is
GLM-5.2-FP8's verbatim (see [GLM-5.md](GLM-5.md#glm-52-fp8)). Everything below
was measured on MXFP4.

## Server

TP=4 on gfx950 GPUs 0-3, vLLM 0.28 plugin backend, LMCache 0.4.5 from
`rocm/atom-dev:vllm-0.28.0`. This is the exact configuration the numbers in
*Measured* came from.

```bash
export PYTHONHASHSEED=0                 # mandatory -- see below
export LMCACHE_LOCAL_CPU=True
export LMCACHE_MAX_LOCAL_CPU_SIZE=256   # GiB **per TP rank**; see Sizing the tier
export LMCACHE_CHUNK_SIZE=256           # must be a multiple of --block-size
export LMCACHE_CACHE_POLICY=ATOM_SLRU
export AITER_QUICK_REDUCE_QUANTIZATION=INT4
export AITER_USE_FLYDSL_MOE_SORTING=1
export AITER_LOG_LEVEL=WARNING

vllm serve /data/amd_int/models/GLM-5.3-MXFP4 \
  --served-model-name amd/GLM-5.3-MXFP4 --trust-remote-code \
  --load-format fastsafetensors --tensor-parallel-size 4 \
  --gpu-memory-utilization 0.95 --block-size 64 --kv-cache-dtype fp8 \
  --max-num-batched-tokens 16384 --max-model-len 1048576 \
  --compilation-config '{"cudagraph_mode": "FULL_AND_PIECEWISE"}' \
  --additional-config '{"online_quant_config": {"global_quant_config": "ptpc_fp8", "exclude_layer": ["lm_head", "model.embed_tokens", "*.mlp.gate", "*expert*"]}}' \
  --enable-prefix-caching --enable-prompt-tokens-details \
  --kv-transfer-config '{"kv_connector":"AtomLMCacheOffloadConnector","kv_connector_module_path":"atom.plugin.vllm.kv_transfer.connector","kv_role":"kv_both","kv_load_failure_policy":"recompute"}'
```

Drop the final `--kv-transfer-config` line and nothing else to get the OFF arm.

**`PYTHONHASHSEED=0` is not optional**, and it is needed on the client too.
Without it each TP worker hashes the same prompt to a different key and the hit
rate is 0.

**`kv_load_failure_policy` defaults to `fail`**, which turns a chunk evicted
between lookup and load into a 500 for the user. `recompute` re-prefills the
invalid blocks instead, which is what you want in every deployment.

**No `--num-gpu-blocks-override`.** vLLM sizes the pool itself; at
`--gpu-memory-utilization 0.95` it reported `Available KV cache memory: 152.98
GiB` per rank and `GPU KV cache size: 3,440,832 tokens` (53,763 blocks),
byte-identical on both arms. If you *do* pin the pool, you must also pass
`--max-model-len`: GLM-5.3's config declares a 1,048,576-token context and vLLM
sizes its one-request floor from that, so an 8,192-block override aborts
startup with *"46.58 GiB KV cache is needed, which is larger than the available
KV cache memory"*. That is arithmetic about the override, not a GLM-5.3 defect.
(The `_OpNamespace 'aiter' object has no attribute 'free_meta_buffer'` that
follows on all four workers is a teardown artifact of that abort, not a second
bug.)

**`ATOM_PREFIX_CACHE_POLICY` and `ATOM_PREFIX_CACHE_PROTECTED_RATIO` are inert
here** and are deliberately absent above. Their only readers are
`atom/model_engine/block_manager.py:137-138`, the ATOM *native* engine's HBM
prefix cache. On the plugin path the HBM cache is vLLM's, and nothing under
`atom/plugin/` instantiates that block manager. Copying them across from a
native recipe changes nothing.

### Sizing the tier

`LMCACHE_MAX_LOCAL_CPU_SIZE` is **per rank**, so TP=4 pins four times it. Read
[GLM-5.2's Sizing section](GLM-5.2-LMCache-Byte-Offload.md#sizing-do-this-before-benchmarking)
for the full treatment; the arithmetic is identical at 47,700 B/rank/token.

The one number worth carrying across runs is the per-prefix unit, and the
connector logs it rather than making you derive it:

    Retrieved 28672 out of 28672 required tokens ... size: 1.2737 gb

(28,672 x 47,700 B = 1.2736 GiB, so that `gb` is GiB.) The **reusable** working
set is `pool_size x 1.2737 GiB/rank` — 163.0 GiB at the 128-prefix pool used
below, against which 256 is 1.57x.

Size for the reusable set, **not** for the run's whole byte traffic. The
per-request cache-bust tails are stored too, are never reused, and at 0.18
GiB/rank each they exceed any sane tier within one benchmark window. They fit
because the tier evicts them, which is exactly what it should do with them.
Then check the outcome, because each failure mode is visible in one line:

* under-sized: `Failed to allocate memory block ... no memory is available`
  appears at all (27,695 times in one GLM-5.2 run at 40 GiB/rank);
* over-sized: `RssAnon` carries the tier while the external-hit count does not
  move.

47,700 B/rank/token is also the exchange rate between tier capacity and reuse
distance: `tier_tokens = tier_bytes_per_rank / 47,700`. A reuse distance beyond
that is asking the tier to hold something it will evict first.

> **Trap.** `Staging buffers: 300 allocated (90.0 GiB, 7.25s pinning)` is **not
> the KV tier.** It is ATOM's MoE expert staging, it is tier-independent, and an
> arm with LMCache entirely absent prints it on all four workers. Measure what
> the tier actually costs with `RssAnon`, and measure the fixed residency on an
> arm with the tier switched off rather than deriving it as `R - TP x tier` —
> that construction assumes the declared tier in order to produce a number that
> cannot then check it.

If the host is NUMA-split, do **not** reach for `numactl --membind=<node>`
first: when `TP x tier` exceeds one node's free memory it is not a policy choice
but a physical impossibility, and the allocation spills or fails rather than
honouring the binding. Gate on `F + TP x tier` against **measured per-node**
free memory. (If you do bind, verify it took with `bind:0` in
`/proc/<worker>/numa_maps`, not `Mems_allowed_list` — `--cpuset-mems` on
rootless podman is silently ineffective.)

## Client

```bash
PYTHONHASHSEED=0 aiperf profile \
  --model amd/GLM-5.3-MXFP4 \
  --url "http://127.0.0.1:8330" --endpoint-type chat --streaming \
  --tokenizer /data/amd_int/models/GLM-5.3-MXFP4 --tokenizer-trust-remote-code \
  --isl 4096 --isl-stddev 0 --osl 512 --osl-stddev 0 \
  --prompt-prefix-length 28672 --prompt-prefix-pool-size 128 \
  --num-dataset-entries 256 --concurrency 8 \
  --benchmark-duration 1800 --benchmark-grace-period 60 \
  --extra-inputs ignore_eos:true --use-server-token-count \
  --cache-bust first-turn-suffix --random-seed 530419 \
  --request-timeout-seconds 3600 --no-gpu-telemetry --artifact-dir "$OUT"
```

Identical on both arms, and identical across the concurrency sweep except
for `--concurrency`, which takes 8, 16 and 32. Four of those flags are
load-bearing and none is cosmetic:

* `--random-seed` fixes the dataset, so both arms replay the same prompts. Pass
  it explicitly; a default would let the arms diverge silently.
* `PYTHONHASHSEED=0` — aiperf builds the prompts locally, so an unseeded hash
  makes the "shared" prefix differ per process, on the client side this time.
* `--use-server-token-count` makes every hit rate below a ratio of two
  server-side counters instead of of a client-side estimate, so the denominator
  cannot drift between arms.
* `--osl-stddev 0` with `ignore_eos:true` fixes every response at exactly 512
  tokens; otherwise the length distribution contaminates ITL and throughput.

The check that it worked: both arms measured ISL 32,768.88 and OSL 512.

**Place the reuse distance before you run.** A chunk is only fetchable from the
CPU tier once it has fallen *out* of HBM, so the tier can only hit on reuse
distances inside

    [ hbm_pool_tokens , tier_tokens )  =  [ 3,440,832 , 5,762,639 )

and this workload's distance is `(distinct_prefixes - 1) x ISL` at
ISL = 32,768. Pool 64 gives 2,064,384 — **below** the HBM pool, so the tier's
hit rate is zero by construction and a larger tier cannot help, because that
only moves the top of the band. This arithmetic is the reason the pool is 128,
and it has to be done before the run rather than discovered after it.

**`--prompt-prefix-pool-size` is a sampling pool, not a count.** aiperf draws
each dataset entry's prefix from the pool *with replacement*, so at
`--prompt-prefix-pool-size 128 --num-dataset-entries 256` only **109** distinct
prefixes actually occur, spread over the 256 entries as 34 x 1, 31 x 2, 28 x 3,
8 x 4, 5 x 5, 2 x 6, 1 x 7. Two instruments agree on the 109: counting distinct
prefixes in the generated `inputs.json`, and counting requests that cached
nothing, which is **exactly 109 in each of the three ON arms** at concurrency 8,
16 and 32. Plan with 109, not 128: the distance is 108 x 32,768 = 3,538,944,
which clears the 3,440,832-token HBM floor by 2.9% rather than the 20.9% the
nominal 128 suggests, and sits 38.6% below the tier ceiling. The margin above
the floor is thin, and the prefix weights above mean it is a distribution rather
than a point — which is why the band arithmetic selects the pool and the
measured external-hit rate confirms it.

## Measured

Three matched ON/OFF pairs — concurrency 8, 16 and 32 — back to back, same tree
(`b6e22c4792ab5202063089b28b4d4197750c3186`, `atom_dirty_lines=0` verified at
the start of each arm), same model, 1800 s per arm, seed 530419. The arms
differ in exactly one thing: whether the connector is loaded. All four ranks'
`Creating LMCacheEngine with config:` dumps were parsed before any traffic was
sent and all four read `'chunk_size': 256, 'max_local_cpu_size': 256.0,
'cache_policy': 'ATOM_SLRU', 'lookup_server_worker_ids': [0]`, and all four
logged `registered 78 layers, num_blocks=53763` — the server block above is a
checked fact, not a list of exported variables.

**Full window, 1800 s.** `prefix hit` is the combined rate `(HBM + tier) / Q`;
`HBM` and `tier` split it, and their denominators differ — see below.

| | Conc | tput/GPU (tok/s/GPU) | TTFT p50 / p90 | ITL p50 | ITL p90 | prefix hit | HBM | tier, of HBM-miss tail | ceiling |
|---|---|---|---|---|---|---|---|---|---|
| OFF | 8 | 86.03 | 2279 / 4301 ms | 18.11 ms | 21.80 ms | 61.13% | 61.13% | — | 89.52% |
| ON | 8 | **90.28** | 1314 / 2927 ms | 16.11 ms | 18.74 ms | **82.55%** | 60.14% | 56.21% | 89.98% |
| | | **+4.94%** | −42.4 / −32.0% | −11.0% | −14.0% | +21.42 pp | | | |
| OFF | 16 | 121.88 | 3223 / 6199 ms | 25.60 ms | 31.37 ms | 62.11% | 62.11% | — | 92.63% |
| ON | 16 | **156.81** | 2448 / 3164 ms | 19.74 ms | 23.41 ms | **86.28%** | 60.76% | 65.04% | 94.23% |
| | | **+28.66%** | −24.0 / −49.0% | −22.9% | −25.4% | +24.17 pp | | | |
| OFF | 32 | 158.89 | 3263 / 6464 ms | 42.04 ms | 50.61 ms | 62.64% | 62.64% | — | 94.31% |
| ON | 32 | **223.62** | 2035 / 3480 ms | 29.44 ms | 33.26 ms | **87.61%** | 61.05% | 68.20% | 95.97% |
| | | **+40.74%** | −37.6 / −46.2% | −30.0% | −34.3% | +24.98 pp | | | |

**Steady state, requests started at t >= 600 s.** The cold term is zero on every
ON arm here, so the ceiling formula degenerates and is left out rather than
printed as 100%.

| | Conc | tput/GPU (tok/s/GPU) | TTFT p50 / p90 | ITL p50 | ITL p90 | prefix hit |
|---|---|---|---|---|---|---|
| OFF | 8 | 88.08 | 2239 / 3649 ms | 18.10 ms | 21.73 ms | 64.37% |
| ON | 8 | **108.07** | 1257 / 1696 ms | 15.96 ms | 17.67 ms | **90.73%** |
| | | **+22.70%** | −43.9 / −53.5% | −11.8% | −18.7% | +26.36 pp |
| OFF | 16 | 124.75 | 3111 / 6076 ms | 25.18 ms | 29.70 ms | 64.85% |
| ON | 16 | **166.92** | 2419 / 3019 ms | 19.04 ms | 21.85 ms | **91.00%** |
| | | **+33.80%** | −22.2 / −50.3% | −24.4% | −26.4% | +26.15 pp |
| OFF | 32 | 161.63 | 3238 / 6314 ms | 41.78 ms | 49.22 ms | 64.23% |
| ON | 32 | **239.69** | 2076 / 3420 ms | 28.93 ms | 31.68 ms | **90.94%** |
| | | **+48.30%** | −35.9 / −45.8% | −30.8% | −35.6% | +26.71 pp |

**The gain grows monotonically with concurrency and has not turned over by 32**
(+4.94% / +28.66% / +40.74% on the full window, +22.70% / +33.80% / +48.30% at
steady state). The mechanism is visible in the same rows: the HBM hit rate is
pinned near 61% on all six arms, so what rises is the share of the HBM-miss tail
the tier answers — 56.21% / 65.04% / 68.20%. Higher concurrency lengthens the
reuse distance, pushing more of the working set out of HBM and into the band the
tier covers. **Do not extrapolate past 32**: three points rising is not a curve,
and the GLM-5.2 measurements on this connector turn over at 16.

The steady-state rows are computed as `requests started at t >= cut`, divided by
`(aiperf Benchmark Duration - cut)`. All six arms share this one construction.

`tput/GPU` is `output_token_throughput / 4`. Output length is fixed at 512, so
it is `request_throughput x 128` by construction and is not a second result.

**The headline number is a function of the window, and the second row is not
cherry-picking.** The tier starts empty and fills at ~0.287 GiB/s/rank, while
this workload's rotating working set is the 109 distinct prefixes plus the 256
distinct tails hanging off them, `109 x 1.2737 + 256 x 0.182 = 185.4 GiB/rank`,
so **~650 s pass before steady-state reuse is possible at all**. A 600 s run at this pool size
would measure the fill, not the cache — which is why each arm runs 1800 s. The
same start-cut is applied to **both** arms, and the OFF arm is the control that
licenses it: it is flat across every cut (0.6721 / 0.6889 / 0.6888 / 0.6900
req/s at t >= 0 / 200 / 300 / 600 s), so cutting the window does not itself
manufacture throughput.

**There is no warmup phase.** Neither `--warmup-request-count` nor
`--warmup-duration` is set, and aiperf's rule is that absent both, no warmup
runs — all 1272 / 1216 exported records are `benchmark_phase: profiling`.
`--benchmark-grace-period 60` drains in-flight requests at the *end* and is not
a warmup. Adding one would not remove the ramp anyway: it would have to run
~650 s itself to cover the fill.

Per 120 s window at concurrency 8, `n` requests started / `cold` = requests
that cached nothing:

| t (s) | ON n / cold / hit% | OFF n / cold / hit% |
|---|---|---|
| 0 | 29 / 26 / 9.05% | 64 / 51 / 17.77% |
| 120 | 59 / 34 / 37.08% | 80 / 30 / 54.69% |
| 240 | 80 / 27 / 57.97% | 80 / 20 / 65.08% |
| 360 | 48 / 11 / 67.45% | 72 / 26 / 55.90% |
| 480 | 40 / 11 / 62.36% | 88 / 19 / 68.61% |
| 600 | 104 / **0** / 90.47% | 88 / 20 / 66.88% |
| 720-1680 | 96-104 / **0** / 89.7-91.9% | 72-88 / 16-28 / 54.3-70.0% |

Two different things are visible here and they must not be conflated. The ON
arm's ramp is transient: cold requests stop entirely at t = 579 s and the hit
rate locks at 90-92%. The OFF arm's scatter is **not** a ramp — it is still
producing 16-28 cold requests per window at t = 1680 s, because the 186.3
GiB/rank working set does not fit the 152.98 GiB/rank HBM pool and evicted
prefixes have nothing underneath them. It is not slow to warm up; it never
warms up.

Cold requests are also spread across the first 600 s rather than bunched at
t = 0, and that is coupon collection, not a defect: each request draws one of
256 dataset entries, which carry 109 distinct prefixes with unequal weights
(1 to 7 entries each), so the last distinct prefix first appears several hundred
requests in. The uniform estimate `n x ln(n)` is only an order-of-magnitude
guide here, because the singleton prefixes — 34 of the 109 — are the ones that
arrive last.

**Where the prompt tokens went**, at concurrency 8.
`vllm:prompt_tokens_by_source_total`, whose three components sum to
`vllm:prompt_tokens_total` exactly on both arms:

| source | OFF | ON |
|---|---|---|
| `local_compute` | 15,488,497 (38.87%) | **7,274,722 (17.45%)** |
| `local_cache_hit` | 24,358,464 (61.13%) | 25,067,392 (60.14%) |
| `external_kv_transfer` | 0 | **9,339,904 (22.41%)** |
| total | 39,846,961 | 41,682,018 |

Prefill recompute fell from 8,561 to 4,034 tok/s (−52.9%) while the server
delivered 4.9% more requests over the full window. The tier only feeds prefill,
so TTFT moves most (−42.4% at p50) and ITL follows second-hand (−11.0%) as
prefill stops competing with decode.

Note that the HBM hit rate is **not** what improved — 61.13% OFF against 60.14%
ON, i.e. very slightly *lower* with the tier on. On the plugin path vLLM asks
the connector only about what the HBM pool missed (`queries = num_tokens -
local_computed`), so the tier's 22.41% is carved out of the miss tail
(16,614,626 tokens, 39.86% of prompt tokens) and the two percentages have
different denominators. **They must not be summed as if they were shares of the
same thing.** Within the tail the tier answered **56.21%**.

That number has an independent witness. LMCache's own `Retrieved X out of Y
required tokens` lines across all four ranks sum to 37,359,616; divided by TP=4
that is 9,339,904, which equals `vllm:external_prefix_cache_hits` bit for bit.
The two instruments share no code path, so this is corroboration rather than a
restatement.

**The ceiling column.** aiperf does **not** export a theoretical hit rate; its
`overall_usage_prompt_cache_read_pct` is the *measured* combined rate, equal by
construction to `(hbm_hits + tier_hits) / Q`. The ceiling is computed here from
the generated dataset, as the share of prompt tokens an infinite cache could
have supplied:

    uncacheable = 109 x ISL          (first sight of each distinct prefix)
                + 147 x (ISL - P)    (first sight of each remaining entry:
                                      prefix hits, its 4,096-token tail cannot)
                + (R - 256) x 1      (the final token is never a cache hit)
    ceiling     = 1 - uncacheable / Q      with P = 28,672, ISL ~ 32,769

**This supersedes the form published in #2291**, `(Q - 128 x 28,672) / Q`, which
gave 90.79% / 91.20% at concurrency 8. That form was loose in two ways: it used
the nominal pool size 128 where only 109 prefixes occur, and it charged nothing
for the tails of the 147 first-seen entries. The corrected bound is tighter
(89.52% / 89.98% at concurrency 8) and still above every measured rate, so no
conclusion moves — but a bound that is not tight is not a bound anyone can use.

The stricter form that would follow from "the 4,096-token tail is cache-busted
and therefore never reusable" — `(R - 109) x 28,672 / Q` — remains **wrong**, and
the measurement is what shows it, because ON measured 82.55% against that form's
80.00%. `--cache-bust first-turn-suffix` does not make every tail unique. The
per-request histogram at concurrency 8 says so directly:

| cached tokens in a request | ON requests |
|---|---|
| 32,768 (the whole prompt) | 252 |
| 28,672 (prefix only) | 878 |
| 0 (cold prefix) | 109 |
| other (partial) | 33 |

So part of the measured ON gain comes from whole-prompt replay rather than pure
prefix reuse. Anyone reproducing a *pure* prefix-reuse result has to make the
tail genuinely unique first; this recipe's numbers are for the workload as
written, and the ceiling above is the one that matches it.

## Related

- [GLM-5.2 LMCache Byte Offload](GLM-5.2-LMCache-Byte-Offload.md) — the full
  treatment of this connector: mechanism, sizing, tuning, gotchas
- [MiniMax-M3 LMCache Byte Offload](MiniMax-M3-LMCache-Byte-Offload.md) — the
  same connector on M3's three-layout registration
- [LMCache KV Cache Offload](LMCache-KV-Cache-Offload.md) — generic plugin path
  (`LMCacheConnectorV1`); does **not** support GLM-5.3's registration
- [GLM-5.3-Flash](../GLM-5.3-Flash.md) — a *different* architecture
  (`glm5_next`, MLA + KDA + DSA), not covered by this recipe
