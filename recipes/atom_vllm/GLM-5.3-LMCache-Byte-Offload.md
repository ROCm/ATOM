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
export LMCACHE_MAX_LOCAL_CPU_SIZE=180 # GiB **per TP rank** -- see Sizing below
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
at all looks identical whether the ranks never built a client or the tier is
simply empty -- which is the normal state during early warmup, and also the
steady state on an arm whose HBM hit rate is high enough that the connector
only ever sees a short miss tail (measured on M3 TP4 by a parallel arm: `world`
correctly 4, preflight green, and still zero rows). Separate the two by reading
check 3 alongside it. Zero is never positive evidence; this probe is strong
only when it is nonzero.

**Judging liveness: `Retrieved > 0` and `external_prefix_cache_hits > 0`, and
nothing else.** `Stored` lines appear on a tier that is never read. And on
LMCache 0.4.5 a healthy readback logs **zero** `Double unpin` lines — that
counter is a failure probe whose polarity is the opposite of what it looks like,
so do not read 0 as "no retrieval".

## Measured

**Both GLM-5.3 checkpoints**, `amd/GLM-5.3-MXFP4` and `amd/GLM-5.3-FP8`, each
TP=4 on gfx950 GPUs 0-3, block size 64, fp8 KV, `--num-gpu-blocks-override
8192`, `--max-model-len 131072`, `LMCACHE_CHUNK_SIZE=64`, `PYTHONHASHSEED=0`,
LMCache 0.4.5 from `rocm/atom-dev:vllm-0.28.0`. CPU tier 32 GiB/rank (see the
caveat under Sizing). The FP8 arm differs from the launch block above only in
its `online_quant_config`, which is GLM-5.2-FP8's verbatim (see
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
| peak HBM during weight load + online quant, per rank | 112.46 GB | 187.96 GB |
| `num_gpu_blocks` vLLM sized *before* the override | 48,709 / 48,676 / 48,676 / 48,665 | 21,269 / 21,236 / 21,236 / 21,225 |
| fused quant custom ops | none | `+quant_fp8`, `fuse_norm_quant`, `fuse_act_quant` |
| `/v1/models` on the port under test (check 0) | `amd/GLM-5.3-MXFP4` | `amd/GLM-5.3-FP8` |

Note what that says about `--num-gpu-blocks-override 8192`. Pinning the pool is
what makes the KV comparison apples-to-apples -- but it is also what erases the
obvious witness, because both arms then report the same 524,288-token pool
*by construction*. The witness has to be read one line earlier, from the sizing
vLLM computed before the override, which still carries the 1.7x difference in
weight footprint as a 2.3x difference in blocks.

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
difference. A prefix-cache hit is not bit-reproducible against a cold run, so
this field is sampling noise in both directions — GLM-5.2's tier-*off* arm,
where nothing was restored at all, scores 0/8 on it. Marker recall is the
criterion, and it is 8/8 everywhere.

### Not measured here

**No throughput number for GLM-5.3.** Both arms ran correctness only, on a
machine whose other socket was under a concurrent 256 GiB-tier job — exactly
the host DRAM, CPU and PCIe the offload path spends, so any timing taken here
would be measuring the neighbour. GLM-5.3 is the same architecture, the same KV
volume per token and the same code path as GLM-5.2, so the GLM-5.2 numbers
(+50.4% at 64 prefixes) are the best available estimate — but they are GLM-5.2's
numbers, not GLM-5.3's. Do not quote them as measured on GLM-5.3.

## Sizing

Read [GLM-5.2's Sizing section](GLM-5.2-LMCache-Byte-Offload.md#sizing-do-this-before-benchmarking);
the arithmetic is identical at 47,700 B/rank/token.

The **32 GiB/rank used above is deliberately below what that section
recommends** — host memory was shared with another job. It was enough for a
40-request correctness run (802,213 prompt tokens ≈ 36 GiB/rank of traffic, with
no allocation failures logged), and it is *not* enough for a benchmark: use
`LMCACHE_MAX_LOCAL_CPU_SIZE=180` there, and size the tier for the whole run
rather than for the prefix pool.

If the host is NUMA-split, bind the server to the socket its GPUs are on
(`numactl --membind=<node>`) and gate on `MemFree >= 1.05 x tier`, on that node,
not on `MemAvailable` — pinned memory does not reclaim page cache. Verify the
binding took: `bind:0` in `/proc/<worker>/numa_maps`, not
`Mems_allowed_list` (`--cpuset-mems` on rootless podman is silently ineffective).

## Related

- [GLM-5.2 LMCache Byte Offload](GLM-5.2-LMCache-Byte-Offload.md) — the full
  treatment of this connector: mechanism, sizing, tuning, gotchas
- [MiniMax-M3 LMCache Byte Offload](MiniMax-M3-LMCache-Byte-Offload.md) — the
  same connector on M3's three-layout registration
- [LMCache KV Cache Offload](LMCache-KV-Cache-Offload.md) — generic plugin path
  (`LMCacheConnectorV1`); does **not** support GLM-5.3's registration
- [GLM-5.3-Flash](../GLM-5.3-Flash.md) — a *different* architecture
  (`glm5_next`, MLA + KDA + DSA), not covered by this recipe
