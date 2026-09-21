# GLM-5.3 — LMCache KV offload on the vLLM plugin (byte codec)

GLM-5.3 (`GlmMoeDsaForCausalLM`) uses `AtomLMCacheOffloadConnector` unchanged.
**No GLM-5.3-specific code exists, and none is wanted** — the mapping added for
GLM-5.2 in #2231 is keyed on the registered layer *names*, never on the model,
and GLM-5.3 registers the identical entries. This recipe records the launch
line and what was measured to establish that; for the mechanism, the sizing
arithmetic, the tuning sweep and the full gotcha list, read
[GLM-5.2 LMCache Byte Offload](GLM-5.2-LMCache-Byte-Offload.md) — all of it
applies here verbatim.

## Why nothing had to change

GLM-5.3 and GLM-5.2 are the same architecture. Every KV-relevant config field
is identical across `GLM-5.2-MXFP4`, `GLM-5.3-MXFP4` and `GLM-5.3-FP8`:

| field | value |
|---|---|
| `architectures` / `model_type` | `GlmMoeDsaForCausalLM` / `glm_moe_dsa` |
| `num_hidden_layers` | 78 |
| `kv_lora_rank` + `qk_rope_head_dim` | 512 + 64 = 576 B/token/layer |
| `index_head_dim` (+4 B packed scale) | 128 → 132 B/token/indexer |
| `index_topk` / `index_n_heads` / `index_topk_freq` | 2048 / 32 / 4 |
| `first_k_dense_replace` | 3 |
| `num_nextn_predict_layers` | 1 |

The **only** field GLM-5.3 adds is `moe_router_dtype: float32`. It selects MoE
routing precision; nothing under `atom/plugin/vllm/kv_transfer/` or
`atom/kv_transfer/offload/` reads it, and it does not touch the KV layout.

So GLM-5.3 registers the same 99 entries in the same two layouts, and the fold
collapses them to the same 78 layers at the same 47,700 B/rank/token.
`tests/plugin/test_vllm_glm_dsa_kv_cache_layout.py` pins those config fields per
checkpoint and derives the byte geometry from them, so a future GLM that moves
`kv_lora_rank` or `index_head_dim` fails there rather than offloading at the
wrong stride.

## Launch

Identical to the GLM-5.2 line except for the model path. Measured at TP=4 on
GPUs 0-3 (gfx950):

```bash
export PYTHONHASHSEED=0               # mandatory
export LMCACHE_LOCAL_CPU=True
export LMCACHE_MAX_LOCAL_CPU_SIZE=90  # GiB **per TP rank** -- size it, see Sizing below
export LMCACHE_CHUNK_SIZE=64          # must equal --block-size
export AITER_QUICK_REDUCE_QUANTIZATION=INT4
export AITER_USE_FLYDSL_MOE_SORTING=1

vllm serve /data/amd_int/models/GLM-5.3-MXFP4 \
  --served-model-name amd/GLM-5.3-MXFP4 --trust-remote-code \
  --load-format fastsafetensors --tensor-parallel-size 4 \
  --gpu-memory-utilization 0.9 --block-size 64 --kv-cache-dtype fp8 \
  --max-num-batched-tokens 16384 --max-model-len 131072 \
  --num-gpu-blocks-override 8192 \
  --compilation-config '{"cudagraph_mode": "FULL_AND_PIECEWISE"}' \
  --additional-config '{"online_quant_config": {"global_quant_config": "ptpc_fp8", "exclude_layer": ["lm_head", "model.embed_tokens", "*.mlp.gate", "*expert*"]}}' \
  --enable-prefix-caching --enable-prompt-tokens-details \
  --kv-transfer-config '{"kv_connector":"AtomLMCacheOffloadConnector","kv_connector_module_path":"atom.plugin.vllm.kv_transfer.connector","kv_role":"kv_both","kv_load_failure_policy":"recompute"}'
```

**Pass `--max-model-len` whenever you pass `--num-gpu-blocks-override`.**
GLM-5.3's config declares a 1,048,576-token context, and vLLM sizes its
one-request floor from that, so an overridden pool of 8,192 blocks (524,288
tokens) aborts startup with *"46.58 GiB KV cache is needed, which is larger than
the available KV cache memory"*. That is arithmetic about the override, not a
GLM-5.3 defect. (The `_OpNamespace 'aiter' object has no attribute
'free_meta_buffer'` that follows on all four workers is a teardown artifact of
that abort, not a second bug.)

## Verify it is actually on

Same four checks as GLM-5.2, plus a fifth that is worth running on any TP>1 arm:

```bash
# 1. vLLM's factory -- expect TP+1 lines (4 workers + EngineCore)
grep -c "Creating v1 connector with name: AtomLMCacheOffloadConnector" server.log

# 2. the fold fired: 78, not 99
grep "ATOM LMCache offload: registered" server.log

# 3. every rank built a live LMCache client, not just rank 0
grep -o "worker_id: [0-9]*, worker_ids: \[[0-9, ]*\]" server.log | sort -u

# 4. the tier is queried AND returns data
curl -s localhost:8330/metrics | grep -E 'external_prefix_cache_(hits|queries)'

# 5. every rank actually READ, not just built a client
grep "Retrieved" server.log | grep -o "Worker_TP[0-9]*" | sort | uniq -c
```

Checks 3 and 5 are the ones that catch a tier that looks healthy and is not. Before
`lmcache_replica_world_size()` learned to read `parallel_config` (#2282), the
plugin path reported `world=1` for a TP4 replica, ranks 1-3 raised inside
`build_lmcache_metadata` and the exception was swallowed by a bare `except` in
`dense/connector.py`. The arm then stored on all four ranks and read back on
one, with no error anywhere. Measured here, all four ranks report distinct ids:

```
worker_id: 0, worker_ids: [0, 1, 2, 3]   (and 1, 2, 3)
```

Check 5 is the same fault seen one step later, and it is the stronger of the
two: check 3 proves each rank *built* a client, check 5 proves each rank *read*
through it. Healthy TP4 spreads the `Retrieved` lines evenly -- measured here,
`8 Worker_TP0 / 8 Worker_TP1 / 8 Worker_TP2 / 8 Worker_TP3` for 8 requests.
The `world=1` failure shows up as lines from `Worker_TP0` alone.

**A zero histogram is underdetermined, not a diagnosis.** No `Retrieved` lines
at all has at least three causes that look identical:

1. the ranks never built a client (`world=1`, check 3);
2. the tier is healthy but the connector never reaches it -- normal during early
   warmup, and the *steady* state whenever the HBM pool is large enough that only
   a short miss tail is ever offered to the connector. Measured on M3 TP4 with
   `world` correctly 4 and preflight green; measured again on GLM-5.2 at
   `conc=32`, where the same tree gives `Retrieved=8` at
   `--num-gpu-blocks-override 8192` and `Retrieved=0` at `16896` -- **only the
   pool size changed**;
3. the engine died on the first real tier load, before any line could be
   written.

Check 3 separates (1); `grep -c EngineDeadError` separates (3). Zero is never
positive evidence; this probe is strong only when it is nonzero.

Because of (2), **`Retrieved > 0` is a property of (tree, HBM pool), not of the
tree** -- always report `--num-gpu-blocks-override` next to it, or the next
reader will run a larger pool, see 0, and conclude the tree is broken.

**Judging liveness: `Retrieved > 0` and `external_prefix_cache_hits > 0`, and
nothing else.** `Stored` lines appear on a tier that is never read. And on
LMCache 0.4.5 a healthy readback logs **zero** `Double unpin` lines — that
counter is a failure probe whose polarity is the opposite of what it looks like,
so do not read 0 as "no retrieval".

## Measured

**Both GLM-5.3 checkpoints**, `amd/GLM-5.3-MXFP4` and `amd/GLM-5.3-FP8`, each
TP=4 on gfx950 GPUs 0-3, block size 64, fp8 KV, `--num-gpu-blocks-override
8192`, `--max-model-len 131072`, `LMCACHE_CHUNK_SIZE=64`, `PYTHONHASHSEED=0`,
LMCache 0.4.5 from `rocm/atom-dev:vllm-0.28.0`. CPU tier **32 GiB/rank**, not
the 90 in the launch block above -- these arms only have to prove restore, and
host memory was shared at the time (see Sizing). The FP8 arm differs from the
MXFP4 arm only in its `online_quant_config`, which is GLM-5.2-FP8's verbatim (see
[GLM-5.md](GLM-5.md#glm-52-fp8)):

```
--additional-config '{"online_quant_config": {"global_quant_config": "ptpc_fp8", "layer_quant_config": {"model.layers.*.mlp.experts": "per_block_fp8"}, "exclude_layer": ["lm_head", "model.embed_tokens", "*.mlp.gate"]}}'
```

Quantization is a weight-side choice; it does not reach the KV cache, which both
arms declare as fp8. The two arms are reported together below because every
figure that follows came out **identical** on them unless a row says otherwise.

**That identity is the expected result, so it cannot double as evidence the two
arms were really different.** The layer count is fixed by the architecture, the
token counts by the (seeded) prompt set, and the `Retrieved` histogram by the
hit structure -- none of them *can* move when only the weight dtype changes, so
they would read exactly the same if the second arm had never been restarted.
What separates the arms is weight-side, and it is in the same two logs:

| | MXFP4 | FP8 |
|---|---|---|
| `quantization=` in the engine config line | `quark` | `fp8` |
| steady `consumed memory (weights + non-torch)`, per rank | 117.84 / 117.93 / 117.93 / 117.96 GiB | 196.43 / 196.53 / 196.53 / 196.56 GiB |
| `Available KV cache memory`, per rank | 138.49 / 138.39 / 138.39 / 138.36 GiB | 60.47 / 60.38 / 60.38 / 60.35 GiB |
| `num_gpu_blocks` vLLM sized *before* the override | 48,709 / 48,676 / 48,676 / 48,665 | 21,269 / 21,236 / 21,236 / 21,225 |
| *peak* HBM during weight load + online quant, per rank | 112.46 GB | 187.96 GB |
| fused quant custom ops | none | `+quant_fp8`, `fuse_norm_quant`, `fuse_act_quant` |
| `/v1/models` on the port under test (check 0) | `amd/GLM-5.3-MXFP4` | `amd/GLM-5.3-FP8` |

The peak row and the steady rows are different quantities and must not be
chained: the peak is what the loader touched (and here it reads *lower* than the
steady figure, because `consumed memory` is measured afterwards and includes
non-torch overhead the loader's own counter never saw). What sets the block
count is the steady row, and it sets it by **subtraction, not by ratio** --
FP8's weights consume 78.6 GiB more, so 78.0 GiB less is left for KV, which is
the whole of the 2.29x difference in blocks. Quoting the 1.67x weight ratio as
if it produced the 2.29x block ratio would be wrong arithmetic that happens to
have the right sign.

Note also what this says about `--num-gpu-blocks-override 8192`. Pinning the
pool is what makes the KV comparison apples-to-apples -- but it is also what
erases the obvious witness, because both arms then report the same
524,288-token pool *by construction*. The witness survives one line earlier, in
the sizing vLLM computed before the override.

### Registration

```
5  Creating v1 connector with name: AtomLMCacheOffloadConnector
4  ATOM LMCache offload: registered 78 layers, num_blocks=8192
   ATOM LMCache offload:   78 x kv tail_shape=(64, 576) dtype=torch.uint8
   ATOM LMCache offload:   21 x kv tail_shape=(64, 132) dtype=torch.uint8
```

78, not 99 — the fold fired. Byte-for-byte the same on MXFP4 and FP8, and
byte-for-byte the GLM-5.2 tails.

### Correctness: two-pass restore

`tools/kv_offload_twopass_check.py --n 8 --flood 24`:

| | MXFP4 | FP8 |
|---|---|---|
| pass-2 prompt tokens | 160,200 | 160,200 |
| pass-2 served from cache | **159,936 (99.84%)** | **159,936 (99.84%)** |
| marker recall, pass 1 / pass 2 | 8/8 / 8/8 | 8/8 / 8/8 |
| `Retrieved` lines | 32, all full (`20160 out of 20160`) | 32, all full |
| `Retrieved` per rank (check 5) | 8 / 8 / 8 / 8 | 8 / 8 / 8 / 8 |
| alloc failures / KV load failures / chunk-misalignment | 0 / 0 / 0 | 0 / 0 / 0 |

The prompt set is seeded, so the two arms were handed the *same* 160,200 tokens
— the columns agreeing is a real comparison, not two independent runs that
happened to land nearby.

**This arm is its own control.** The GLM-5.2 recipe needed a separate
tier-off arm to prove the flood really evicted HBM; here the same-arm
decomposition says it directly:

```
vllm:prefix_cache_hits_total                                  0
vllm:external_prefix_cache_hits_total                   159,936
vllm:prompt_tokens_by_source_total{source="local_cache_hit"}        0
vllm:prompt_tokens_by_source_total{source="external_kv_transfer"}  159,936
vllm:prompt_tokens_by_source_total{source="local_compute"}    642,277
```

(Figures shown are the FP8 arm's; the MXFP4 arm's are the same to the token.)
HBM's prefix cache hit **zero** times over the whole run, so every one of those
159,936 restored tokens came through LMCache and nowhere else. (It is zero
because the tool puts each prompt's random marker at the very *start*, so no two
prompts share a prefix at all, and the 24-prompt flood is larger than the
524,288-token pool.)

Both identities close exactly on the drained server, on both arms:

```
prefix_cache_queries - prefix_cache_hits == external_prefix_cache_queries
        802,213      -        0          ==        802,213                 OK
prefix_cache_hits + external_prefix_cache_hits == prompt_tokens_cached
        0         +      159,936           ==      159,936                 OK
```

Do **not** read the tool's `identical_text_pass2_vs_pass1` (1/8 on MXFP4, 4/8 on
FP8) as a regression, and do not read the gap between the two arms as a quality
difference. The reason is structural rather than statistical: a prefix-cache hit
is not bit-reproducible against a cold run, so text that differs is expected
whether or not anything is wrong, and no number of samples turns a low count --
or a gap between two arms -- into a regression signal. The field is not
symmetric and the claim here is only the one half the structure licenses: an
*identical* pass would still be weak evidence that nothing broke; what the
structure rules out is reading non-identical text as evidence that something
did. What the field does under known-correct behaviour has been seen once —
GLM-5.2's tier-*off* arm, where nothing was restored at all, scores 0/8 — but
that is a different model, path and arm, so read it as a demonstration that 0/8
is compatible with correct output, not as a dispersion this run may be compared
against. **No repeat of this field was measured on GLM-5.3**, so the 1/8-vs-4/8
gap is unexplained, not shown to be noise. Marker recall is the criterion, and
it is 8/8 everywhere — measured here, on both arms.

### Throughput at a pinned HBM pool (override 8192), tier 90 GiB/rank

A matched ON/OFF pair, back to back in the same slot, same tree
(`b6e22c4792ab5202063089b28b4d4197750c3186`, `atom_dirty_lines=0`), same model
(`amd/GLM-5.3-MXFP4`), TP=4 on GPUs 0-3, `--num-gpu-blocks-override 8192`,
`LMCACHE_MAX_LOCAL_CPU_SIZE=90`, aiperf 600 s at concurrency 8 with a 64-prefix
pool of 28,672 tokens and a 4,096-token unique tail, seed 530419. Both arms
measured ISL 32,768.87 / 32,768.86 and OSL 512 / 512, so the workload matched.

| | OFF | ON | |
|---|---|---|---|
| **request_throughput** | 0.5176 req/s | **0.6225 req/s** | **+20.3%** |
| output_token_throughput | 265.01 tok/s | 318.73 tok/s | +20.3% |
| request_count | 312 | 376 | +20.5% |
| request_latency avg / p50 / p90 | 15446 / 15567 / 17415 ms | 12841 / 12284 / 15215 ms | −16.9 / −21.1 / −12.6% |
| **TTFT** avg / p50 / p90 | 4377 / 4296 / 6758 ms | 3111 / 3062 / 5067 ms | **−28.9 / −28.7 / −25.0%** |
| ITL avg / p50 / p90 | 21.66 / 21.32 / 27.90 ms | 19.04 / 18.61 / 22.90 ms | −12.1 / −12.7 / −17.9% |
| benchmark_duration | 602.78 s | 604.00 s | +0.2% |

`request_throughput` and `output_token_throughput` are **one** number, not two:
OSL is fixed at 512, so the second is the first times 512 by construction.

**Why it moved.** Server-side `vllm:prompt_tokens_by_source_total`, whose three
components sum to `vllm:prompt_tokens_total` exactly on both arms:

| source | OFF | ON |
|---|---|---|
| `local_compute` | 8,102,174 (79.25%) | 4,807,184 (39.02%) |
| `local_cache_hit` | 2,121,728 (20.75%) | 2,465,792 (20.01%) |
| `external_kv_transfer` | 0 | **5,048,128 (40.97%)** |
| total | 10,223,902 | 12,321,104 |

Every percentage above is against that server-side total. aiperf's own
`total_usage_prompt_tokens` reads 13 lower on **both** arms (10,223,889 /
12,321,091) -- the startup health-check request, which the server counts and the
client never issued. It is 0.0001% and changes nothing, but recompute against
the same total the table used, or the sums stop closing.

Prefill recompute fell from 13,441 to 7,959 tok/s (−40.8%) while the server
delivered 20.5% more requests. The tier only feeds prefill, so TTFT should move
most and most uniformly — it does (−28.9 / −28.7 / −25.0% across avg/p50/p90),
while ITL's −12.1% is the second-order effect of prefill no longer competing
with decode.

**How much was reachable.** On the plugin path vLLM asks the connector only for
what the HBM pool missed (`queries = num_tokens - local_computed`), so the
connector's ceiling is the reusable prefix minus what HBM already served. Write
the construction, not the total — `pool_size = 64` distinct prefixes means the
**first occurrence of each is not reusable by anything**:

    reusable  = (376 - 64) x 28,672 =  8,945,664   72.60% of ON prompt tokens
    - HBM     =              2,465,792             20.01%
    = headroom=              6,479,872             52.59%
    supplied  =              5,048,128             40.97%  -> 77.9% of headroom

For contrast, the same connector on GLM-5.2 at concurrency 40 supplied 0.43% of
prompt tokens, because that arm's HBM pool was large enough to leave almost no
miss tail. **The tier's value is set by the HBM pool, not by the tier size.**

**Limits on the above.** The OFF side has three repeats (0.5148 / 0.5150 /
0.5176 req/s, range 0.54%); the ON side is n=1, so no ON-side dispersion is
quoted and the OFF range must not be borrowed for it. Throughput is quantised at
1/376 = 0.27% (ON) and 1/312 = 0.32% (OFF). Two known asymmetries both favour
the OFF arm — it ran second, on a warmer page cache, and started with 624.6 GiB
free against the ON arm's 154.7 — so they cannot manufacture the gain, but they
are not quantified either.

### Throughput at the default HBM pool, tier 256 GiB/rank

The pair above pins the pool with `--num-gpu-blocks-override 8192`. That is
what makes its KV comparison apples-to-apples, but it is not the pool a user
gets, and it is a 524,288-token pool — small enough that almost any reuse
distance falls outside it. This second pair drops the override and lets vLLM
size the pool instead. At the Launch block's `--gpu-memory-utilization 0.9`
that is the 48,709 blocks (3,117,376 tokens) already in the table above; this
pair also raises utilisation to 0.95 and measured **3,440,832 tokens**, 6.6x
the pinned pool. Every other knob that moved with it is listed next, because a
pool change on its own would not be worth a second table.

Same tree (`b6e22c4792ab5202063089b28b4d4197750c3186`, `atom_dirty_lines=0`),
same model (`amd/GLM-5.3-MXFP4`), TP=4 on GPUs 0-3, block size 64, fp8 KV, both
arms back to back in the same slot. Differences from the pair above:

| | pair above | this pair |
|---|---|---|
| HBM pool | `--num-gpu-blocks-override 8192` = 524,288 tok | **no override**; vLLM sized **3,440,832 tok** (152.98 GiB/rank), byte-identical on both arms |
| `--gpu-memory-utilization` | 0.90 | **0.95** |
| `--max-model-len` | 131072 | **1048576** (the checkpoint's own `max_position_embeddings`) |
| `LMCACHE_MAX_LOCAL_CPU_SIZE` | 90 | **256** GiB/rank |
| `LMCACHE_CHUNK_SIZE` | 64 | **256** (= 4 KV blocks; prefix 28,672 = 112 chunks and tail 4,096 = 16, so no chunk straddles the seam) |
| `LMCACHE_CACHE_POLICY` | LMCache default (LRU) | **ATOM_SLRU** |
| prefix pool / dataset entries | 64 / 128 | **128 / 256** |
| duration | 600 s | **1800 s** |

`LMCACHE_LOOKUP_SERVER_WORKER_IDS` is left unset, so ATOM's own default applies
(`config.py:376-377` forces `[0]`). All four ranks' `Creating LMCacheEngine with
config:` dumps were parsed before any traffic was sent, and all four read
`chunk_size=256, max_local_cpu_size=256.0, cache_policy='ATOM_SLRU',
lookup_server_worker_ids=[0]` — every row in the table above is a checked fact,
not an exported variable.

**The load.** Both arms, identical except for the connector:

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

Three of those flags are load-bearing for comparability and none of them is
optional. `--random-seed` fixes the dataset, so both arms replay the same
prompts. `PYTHONHASHSEED=0` is needed on the **client** as well as the server:
aiperf builds the prompts locally, and an unseeded hash makes the "shared"
prefix differ per process. `--use-server-token-count` makes every hit rate
below a ratio of two server-side counters rather than of a client-side estimate,
so the denominator cannot drift between arms. Measured ISL came out
32,768.88 on both arms and OSL 512 on both, which is the check that it worked.

**Why pool size 128.** A chunk is only fetchable from the CPU tier once it has
fallen *out* of HBM, so the tier can only hit on reuse distances inside

    [ hbm_pool_tokens , tier_tokens )  =  [ 3,440,832 , 5,762,639 )

and this workload's reuse distance is `(pool_size - 1) x ISL` with ISL = 32,768.
Pool 64 gives 2,064,384 — **below** the HBM pool, so the tier's hit rate is zero
by construction and raising the tier cannot help, because that only moves the
top of the band. Pool 128 gives 4,161,536: 20.9% above the floor and 27.8%
below the ceiling. This is the arithmetic that has to be done *before* the run,
not the result of one.

| | Conc | tput/GPU | TTFT p50 / p90 | ITL p50 | ITL p90 | prefix hit | ceiling |
|---|---|---|---|---|---|---|---|
| **full window, 1800 s** | | | | | | | |
| OFF | 8 | 86.03 tok/s/GPU | 2279 / 4301 ms | 18.11 ms | 21.80 ms | 61.13% | 90.79% |
| ON | 8 | **90.28 tok/s/GPU** | 1314 / 2927 ms | 16.11 ms | 18.74 ms | **82.55%** | 91.20% |
| | | **+4.94%** | −42.3 / −32.0% | −11.0% | −14.0% | +21.4 pp | |
| **steady state, t >= 600 s** | | | | | | | |
| OFF | 8 | 88.32 tok/s/GPU | 2239 / 3649 ms | 18.10 ms | 21.73 ms | 64.37% | n/a |
| ON | 8 | **108.23 tok/s/GPU** | 1257 / 1696 ms | 15.96 ms | 17.67 ms | **90.73%** | n/a |
| | | **+22.54%** | −43.9 / −53.5% | −11.8% | −18.7% | +26.4 pp | |

`tput/GPU` is `output_token_throughput / 4`. OSL is fixed at 512, so it is
`request_throughput x 128` by construction and carries no information the
request rate does not.

**The headline number is a function of the window, and the second row is not
cherry-picking.** The tier starts empty and is filled at ~0.287 GiB/s/rank,
while this workload's rotating working set is 128 prefixes plus the 128 tails
in flight with them, `128 x (1.2737 + 0.182) = 186.3 GiB/rank`, so **~650 s
pass before steady-state reuse is possible at all**. A 600 s run at this pool
size would measure the fill, not the cache — which is why this pair runs 1800
s. The same start-cut is applied to **both** arms, and the OFF arm is the
control that licenses it: it is flat across every cut (0.6721 / 0.6889 /
0.6888 / 0.6900 req/s at t >= 0 / 200 / 300 / 600 s), so cutting the window
does not itself manufacture throughput.

**There is no warmup phase.** Neither `--warmup-request-count` nor
`--warmup-duration` is set, and aiperf's own rule is that absent both, no warmup
runs (`profile_export.jsonl` contains 1272 / 1216 records and all are
`benchmark_phase: profiling`). `--benchmark-grace-period 60` drains in-flight
requests at the *end* and is not a warmup. Adding one would not remove the ramp
anyway: it would have to run ~650 s itself to cover the fill.

Per 120 s window, `n` requests started / `cold` = requests that cached nothing:

| t (s) | ON n / cold / hit% | OFF n / cold / hit% |
|---|---|---|
| 0 | 29 / 26 / 9.05% | 64 / 51 / 17.77% |
| 120 | 59 / 34 / 37.08% | 80 / 30 / 54.69% |
| 240 | 80 / 27 / 57.97% | 80 / 20 / 65.08% |
| 360 | 48 / 11 / 67.45% | 72 / 26 / 55.90% |
| 480 | 40 / 11 / 62.36% | 88 / 19 / 68.61% |
| 600 | 104 / **0** / 90.47% | 88 / 20 / 66.88% |
| 720-1680 | 96-104 / **0** / 89.7-91.9% | 72-88 / 16-28 / 54.3-70.0% |

Two different things are visible here and they should not be conflated. The ON
arm's ramp is transient: cold requests stop entirely at t = 579 s and the hit
rate locks at 90-92%. The OFF arm's scatter is **not** a ramp — it is still
producing 16-28 cold requests per window at t = 1680 s, because the 186.3
GiB/rank working set does not fit the 152.98 GiB/rank HBM pool and evicted
prefixes have nothing underneath them. It is not slow to warm up; it never
warms up.

Cold requests are also spread across the first 600 s rather than bunched at
t = 0, and that is coupon collection, not a defect: each request draws a prefix
uniformly from 128, so the last distinct prefix first appears around request
`128 x ln(128) ~ 621`.

**Where the prompt tokens went.** `vllm:prompt_tokens_by_source_total`, whose
three components sum to `vllm:prompt_tokens_total` exactly on both arms:

| source | OFF | ON |
|---|---|---|
| `local_compute` | 15,488,497 (38.87%) | **7,274,722 (17.45%)** |
| `local_cache_hit` | 24,358,464 (61.13%) | 25,067,392 (60.14%) |
| `external_kv_transfer` | 0 | **9,339,904 (22.41%)** |
| total | 39,846,961 | 41,682,018 |

Prefill recompute fell from 8,561 to 4,034 tok/s (−52.9%) while the server
delivered 4.9% more requests over the full window. As in the pair above, the
tier only feeds prefill, so TTFT moves most (−42.3% at p50) and ITL follows
second-hand (−11.0%) as prefill stops competing with decode.

Note that the HBM hit rate is **not** what improved — it is 61.13% OFF against
60.14% ON, i.e. very slightly *lower* with the tier on. On the plugin path vLLM
asks the connector only about what the HBM pool missed (`queries = num_tokens -
local_computed`), so the tier's 22.41% is carved out of the miss tail
(16,614,626 tokens, 39.86% of prompt tokens) and the two percentages have
different denominators. Within the tail the tier answered **56.21%**.

That number has an independent witness. LMCache's own `Retrieved X out of Y
required tokens` lines across all four ranks sum to 37,359,616; divided by TP=4
that is 9,339,904, which equals `vllm:external_prefix_cache_hits` bit for bit.
The two instruments share no code path, so this is corroboration rather than a
restatement.

**The ceiling column, and a correction.** aiperf does **not** export a
theoretical hit rate; its `overall_usage_prompt_cache_read_pct` (82.547% /
61.130%) is the *measured* combined rate, equal by construction to
`(hbm_hits + tier_hits) / prompt_tokens`. The ceiling above is computed here as

    ceiling = (Q - pool_size x 28,672) / Q

on the reasoning that only the first load of each of the 128 distinct prefixes
is unavoidable. The tempting stricter form — "the 4,096-token tail is
cache-busted and therefore never reusable", giving `(R - 128) x 28,672 / Q =
78.69%` — **is wrong, and the measurement is what shows it**, because ON
measured 82.55%, above that supposed ceiling. The per-request histogram says
why: `--cache-bust first-turn-suffix` does not make every tail unique.

| cached tokens in a request | ON requests |
|---|---|
| 32,768 (the whole prompt) | 252 |
| 28,672 (prefix only) | 878 |
| 0 (cold prefix) | 109 |
| other (partial) | 33 |

So 91.20% is a loose but genuine upper bound, and the steady-state row is
deliberately left `n/a`: with zero cold requests after t = 600 s the cold term
vanishes and the formula degenerates to 100%, while the real limit there is how
often tails repeat — which this workload does not pin down. What can be said is
that ON's steady 90.73% sits within half a point of the full-window bound.

**Limits.** Each arm is **n=1 in runs**. No dispersion is quoted, and the OFF
arm's flatness across window cuts bounds within-run drift, not run-to-run
variance — it is what licenses the start-cut, not an error bar on +22.54%.
Throughput is quantised at 1/1272 = 0.08% (ON) and 1/1216 = 0.08% (OFF), which
is far below the effect but says nothing about the effect's repeatability. The
+4.94% full-window figure in particular is a mixture of a 600 s transient and a
1200 s steady state, and should be quoted as such or not at all.

### Host memory and tier residency

Per-worker `RssAnon`, sampled every 10 s across both arms by one sampler into
one file:

| arm | per-worker | TP4 total | plateau |
|---|---|---|---|
| ON (tier 90) | 100.4 GiB | 401.7 GiB | n=51, spread 0.30 GiB |
| OFF (no tier) | 9.7 GiB | 38.8 GiB | n=42, spread 0.00 GiB |

`n` is how long each plateau lasted in 10 s samples, not how many independent
chances the number had to be wrong: the OFF arm's spread of 0.00 says its 42
readings are one value read 42 times. So the spread column bounds sampler and
allocator stability within a run, and says nothing about run-to-run variance —
each arm is still n=1 in runs, like the throughput above.

The OFF arm sets the tier to zero, so **its `RssAnon` measures the fixed
residency directly**: F = 9.7 GiB/worker. The difference then bounds tier
residency at `(401.7 - 38.8) / 4 = 90.7 GiB/rank` against a declared 90.0
(+0.8%) — an **upper** bound, because the subtraction charges the tier with
everything that differs between the arms, including LMCache's own non-tier anon.
Do not compute F as `R - TP x tier`: that assumes the declared tier in order to
produce F, and cannot then check it.

> **Trap.** `Staging buffers: 300 allocated (90.0 GiB, 7.25s pinning)` is **not
> the KV tier.** It is ATOM's MoE expert staging. The OFF arm above prints it on
> all four workers with `grep -cE "AtomLMCacheOffloadConnector|max_local_cpu_size"`
> returning **0**, and a separate tier=64 arm prints the same `90.0 GiB`. Its
> value is tier-independent, and on a tier=90 run it collides byte-for-byte with
> the tier setting. Use `RssAnon` to see what the tier actually costs.

## Sizing

Read [GLM-5.2's Sizing section](GLM-5.2-LMCache-Byte-Offload.md#sizing-do-this-before-benchmarking);
the arithmetic is identical at 47,700 B/rank/token.

The **32 GiB/rank used in the correctness arms is deliberately small** — host
memory was shared with another job. It was enough for a 40-request run (802,213
prompt tokens, no allocation failures logged) and is not enough for a benchmark.

The benchmark above ran at **`LMCACHE_MAX_LOCAL_CPU_SIZE=90`**, not the 180 that
GLM-5.2's Sizing section prescribes, and logged **zero** `Failed to allocate
memory block ... no memory is available` over 600 s and 376 requests.

The one number worth carrying across runs is the per-prefix unit, and it is
measured rather than derived — the connector logs it:

    Retrieved 28672 out of 28672 required tokens ... size: 1.2737 gb

(28,672 x 47,700 B = 1.2736 GiB, so that `gb` is GiB.) The **reusable** working
set is then `64 distinct prefixes x 1.2737 GiB = 81.5 GiB/rank`, and 90 is
1.10x that.

81.5 is a lower bound on what the tier must hold, not the requirement: the
per-request cache-bust tails are stored too, are never reused, and at 0.18
GiB/rank each they exceed the tier several times over within one 600 s window.
They fit because the tier evicts them, which is what the tier is supposed to do
with them. So do **not** try to size for the whole run's byte traffic — size for
the reusable set and then check the outcome, because the two failure modes are
each visible in one line:

* under-sized: `no memory is available` appears at all (27,695 times in one
  GLM-5.2 run at 40 GiB/rank);
* over-sized: `RssAnon` carries the tier (see *Host memory and tier residency*)
  while the external-hit count does not move.

Both were checked here: zero allocation failures, and 40.97% of prompt tokens
served externally. 180 was not tried; at TP=4 it is 720 GiB of pinned anon,
which on this host is more than one NUMA node has free.

47,700 B/rank/token is also the exchange rate between tier capacity and reuse
distance: `tier_tokens = tier_bytes_per_rank / 47,700`. If your reuse distance
exceeds that, the tier is being asked to hold something it will evict first.

If the host is NUMA-split, **do not reach for `numactl --membind=<node>` first**:
when `TP x tier` exceeds one node's free memory it is not a policy choice but a
physical impossibility, and the allocation will spill or fail rather than
honour the binding. At TP=4 and 90 GiB/rank that is 360 GiB of pinned anon, more
than a single node had. The rank skew that follows is structural, not noise.
(If you do bind, verify it took with `bind:0` in `/proc/<worker>/numa_maps`, not
`Mems_allowed_list` — `--cpuset-mems` on rootless podman is silently ineffective.)

Gate on `F + TP x tier` against **measured per-node free memory**, not on
`MemFree >= 1.05 x tier`: the latter omits both the fixed residency and the
other ranks. Measure F on an arm with the tier switched off (see *Host memory
and tier residency* above) rather than deriving it from a declared tier.

"Pinned memory does not reclaim page cache" is **too strong**. On the arm above,
the tier pinned 360 GiB starting from 154.7 GiB free, and `Cached` fell 470.3
GiB while `Dirty` stayed at 0.0 — reclaim carried the whole pin. The cheap
retrospective test is `MemFree_after_stop - MemFree_before > 0` (here +469.9
GiB, closing to 0.1% against the independently constructed `ΔCached`). But
reclaim is **regional, not unconditional**: a separate arm asking for 850 GiB
died with page cache that never moved. Neither case decides the other — measure
your own.

## Related

- [GLM-5.2 LMCache Byte Offload](GLM-5.2-LMCache-Byte-Offload.md) — the full
  treatment of this connector: mechanism, sizing, tuning, gotchas
- [MiniMax-M3 LMCache Byte Offload](MiniMax-M3-LMCache-Byte-Offload.md) — the
  same connector on M3's three-layout registration
- [LMCache KV Cache Offload](LMCache-KV-Cache-Offload.md) — generic plugin path
  (`LMCacheConnectorV1`); does **not** support GLM-5.3's registration
- [GLM-5.3-Flash](../GLM-5.3-Flash.md) — a *different* architecture
  (`glm5_next`, MLA + KDA + DSA), not covered by this recipe
