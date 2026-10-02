# Kimi-K3 — LMCache KV offload on the vLLM plugin (byte codec)

Kimi-K3 (`KimiK3ForCausalLM`) offloads KV through `AtomLMCacheOffloadConnector`,
the same connector MiniMax-M3 and GLM-5.2 use — but unlike those, and unlike
GLM-5.3, it is **not** zero-code. K3 is hybrid: MLA full-attention layers plus
KDA recurrent layers, so vLLM builds several KV cache groups and the connector
runs a second leg for the recurrent boundary state. That leg is what the rest
of this recipe is about.

For the LMCache build steps, the mechanism and the generic tier-sizing
arithmetic, read
[MiniMax-M3 — LMCache KV offload](MiniMax-M3-LMCache-Byte-Offload.md) and
[GLM-5.2 LMCache Byte Offload](GLM-5.2-LMCache-Byte-Offload.md); all of it
applies here unchanged. What follows is only what K3 adds, plus the pair that
was measured.

The server and client lines this recipe extends live in
[Kimi-K3.md](Kimi-K3.md) — this file adds the `--kv-transfer-config` leg to the
DSpark launch there rather than restating it.

K3 is hybrid, so vLLM splits the cache into two *kinds* of group — MLA full
attention and KDA recurrent state. It does not build one group per kind. vLLM
builds **equal-sized** groups, sized by the largest layer family, so K3's 29 MLA
layers and 69 KDA layers come out as **four** groups: one attention group of 29
and **three** mamba groups of 23 (measured: `groups=23,23,23`). All three mamba
groups hold one logical state — one token boundary, one block id each — and none
of them is restorable alone.

A restored MLA prefix is correct only if the KDA state at the **same token
boundary** is restored with it — half a restore is not a crash and not a log
line, it is wrong output.

The two groups therefore move by two different mechanisms:

| group | moved by | addressed by |
|---|---|---|
| MLA | `DenseKVByteCodec`, whole blocks per chunk | the request's block table, positionally |
| KDA | `StateByteCodec`, one opaque page per boundary | vLLM's explicit boundary hand-off |

The KDA group cannot be read positionally at all. In `--mamba-cache-mode align`
a mamba block table is not append-only — a superseded state block is freed and
nulled, and speculative blocks relocate in place — so indexing it by
`token // block_size` can land on a null, freed, or live speculative block and
persist those bytes under a valid prefix hash. The only safe source is vLLM's
explicit hand-off — `SchedulerOutput.partial_tail_offloads` on the pinned 0.28
(`KVCacheManager.take_partial_tail_offloads`), the same
`{req_id: [(group_id, block_id, boundary_tokens)]}` payload under
`kv_connector_block_state.boundary_state_offloads` on 0.29 — which names the
exact block holding a committed boundary state. The connector reads whichever
spelling the running vLLM provides.

Correctness of the pair is enforced on **lookup**, not on save: the reported
external hit is capped at the largest chunk boundary whose KDA state the index
still claims. A KDA state that was never stored, or that LMCache evicted,
shortens the prefix instead of corrupting it.

## Server

TP=8 on gfx950 GPUs 0-7, vLLM 0.28 plugin backend with ATOM as an out-of-tree
plugin, LMCache 0.5.5rc3. This is the complete launch — nothing is elided.

One value here is **not** the one the first pair in *Measured* ran at: the tier
is 90 GiB/rank, and that pair ran at 64. 90 is what the second working point
(*A controlled-prefix client*) used and what is recommended; 64 is recorded with
its own numbers in that section so the older pair stays reproducible.

```bash
export AITER_LOG_LEVEL=WARNING
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export PYTHONHASHSEED=0                 # mandatory -- see below
export LMCACHE_LOCAL_CPU=True
export LMCACHE_MAX_LOCAL_CPU_SIZE=90    # GiB **per TP rank**; see Sizing the tier
export LMCACHE_CHUNK_SIZE=1536          # multiple of the *effective* block size
export LMCACHE_CACHE_POLICY=ATOM_SLRU
export OFFLOAD_MIN_LOAD_TOKENS=256      # default 8192 disables the tier for chat-sized prompts

MODEL=/models/moonshotai/Kimi-K3
DRAFT=/models/Inferact/Kimi-K3-DSpark

vllm serve "${MODEL}" \
    --host 127.0.0.1 --port 8713 \
    --tensor-parallel-size 8 \
    --trust-remote-code \
    --enable-prefix-caching \
    --enable-prompt-tokens-details \
    --mamba-cache-mode align \
    --kv-cache-dtype fp8 \
    --max-model-len 65536 \
    --max-num-seqs 64 \
    --max-num-batched-tokens 16384 \
    --gpu-memory-utilization 0.85 \
    --block-size 128 \
    --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE"}' \
    --speculative-config '{"method":"dspark","model":"'"${DRAFT}"'","num_speculative_tokens":2}' \
    --additional-config '{"online_quant_config":{"global_quant_config":"ptpc_fp8","exclude_layer":["lm_head","model.embed_tokens","*self_attn.[qkv]_conv1d*","*block_sparse_moe.experts*","*block_sparse_moe.routed_expert_*","*vision_tower*","*mm_projector*"]}}' \
    --kv-transfer-config '{"kv_connector":"AtomLMCacheOffloadConnector","kv_connector_module_path":"atom.plugin.vllm.kv_transfer.connector","kv_role":"kv_both","kv_load_failure_policy":"recompute"}'
```

Drop the final `--kv-transfer-config` line and the `LMCACHE_*` exports, and
nothing else, to get the OFF arm.

**No `--num-gpu-blocks-override`.** vLLM sizes the pool itself; at
`--gpu-memory-utilization 0.85` it reports `GPU KV cache size: 1,887,436
tokens` (1584 blocks), identical on both arms. If you do pin the pool you must
also pass `--max-model-len`, which is already above.

**`PYTHONHASHSEED=0` is not optional**, and it is needed on the client too.
Without it each TP worker hashes the same prompt to a different key and the hit
rate is 0.

**`LMCACHE_CACHE_POLICY=ATOM_SLRU`.** `ATOM_SLRU` is not one of LMCache's own
policies — LMCache ships fifo/lfu/lru/mru. ATOM registers it into LMCache's
`POLICY_MAPPING` from `atom/kv_transfer/offload/config.py:build_lmcache_config`,
which the plugin path reaches through `dense/connector.py`, so the name resolves
on this path too. Leave the variable unset and you silently get plain `LRU`.
Read the effective value back out of the rank-0 `Creating LMCacheEngine with
config:` dump: the `Initializing LRUCachePolicy` line **cannot** tell the two
apart, because SLRU subclasses LRU and that line is printed by the base
constructor.

Three further settings are K3-specific and each one is a hard failure if wrong:

- **`--mamba-cache-mode align` is mandatory**, not just useful for DSpark. Any
  other mode keeps the recurrent state where this connector has no hand-off for
  it, so a boundary block id would be a guess. The connector refuses to start
  rather than guess.
- **`"kv_load_failure_policy":"recompute"`** must be set. vLLM's default is
  `fail`, which turns an offload-tier miss into a user-visible 500. The KDA leg
  reports a load error deliberately whenever a recurrent state does not come
  back — that is the mechanism that keeps a half-restored prefix from being
  served — so under the default policy an ordinary eviction fails the request.
- **`LMCACHE_CHUNK_SIZE` must be a multiple of the mamba block size, which is
  not the `--block-size` you passed.** vLLM raises the attention block size so
  the attention page is at least as large as the mamba page, logging `Setting
  attention block size to 1536 tokens to ensure that attention page size is >=
  mamba page size` (`vllm/platforms/interface.py`). `--block-size 128` does not
  prevent this: it only fixes the *requested* size, and the hybrid alignment
  step overrides it afterwards. On K3 the effective size is **1536**, so read it
  out of the log rather than assuming the flag won. Only chunk-aligned
  boundaries are stored and only chunk-aligned boundaries are probed on lookup,
  so a chunk that ends between two boundaries can never produce a usable pair.
  The connector validates this at construction and names both numbers if it does
  not hold.

Hybrid models also require vLLM's hybrid memory allocator, which vLLM
auto-disables for a connector that does not declare `SupportsHMA` — so without
that declaration K3 plus `--kv-transfer-config` does not mis-save, it does not
boot. The connector declares it; the check below confirms HMA stayed on.

`ATOM_PREFIX_CACHE_POLICY` and `ATOM_PREFIX_CACHE_PROTECTED_RATIO` are **inert
here** and are deliberately absent. Their only readers are
`atom/model_engine/block_manager.py`, the ATOM *native* engine's HBM prefix
cache; on the plugin path the HBM cache is vLLM's, and nothing under
`atom/plugin/` instantiates that block manager.

### Sizing the tier

Per rank K3 costs **56,448 B/token** — `29 x 576 = 16,704` for attention plus
`58.22 MiB / 1536 = 39,744` for the recurrent boundary state, so the recurrent
leg is 2.4x the attention leg and dominates the tier.

The tier can only do work for a turn whose reuse distance — the KV volume other
requests write between the turn that produced a prefix and the turn that wants
it back — lands in the half-open band `[reusable HBM prefix, tier)`. Shorter
than the HBM prefix and HBM already had it; longer than the tier and both
missed. Count the population in that band before booting anything: it is one
pass over a previous run's `profile_export.jsonl` and it costs no GPU.

**The lower edge of the band is not `GPU KV cache size`.** That log line is
`max_concurrency x max_model_len`, not a capacity. What caches a reusable prefix
is the blocks left over once the live requests have taken theirs:

```
blocks per request = ceil(max_model_len / 1536) + 3 x (2 + num_speculative_blocks)
                   = 43 + 12 = 55                        # at 65,536, spec_blocks=2
reusable prefix     = (num_blocks - concurrency x 55) / 4 x 1536 tokens
```

The mamba term does not depend on `max_model_len` — in align mode `MambaSpec`
asks for `page_size x (2 + num_speculative_blocks)` regardless
(`vllm/v1/kv_cache_interface.py`) — so shortening the window only shrinks the
attention term. The `/ 4` is one attention block plus one retained boundary
block in each of the three mamba groups. Check the whole formula against the
boot: `num_blocks / Maximum concurrency` printed `1584 / 28.80 = 55`.

At the launch above that is `(1584 - 16 x 55) / 4 x 1536 = 270,336` tokens. So
the band is `[270,336, tier)`, and the only free parameter left is the tier.
Measured against the reuse distances of this very workload
(`inferencex-agentx-mvp` on `semianalysis_cc_traces_weka_062126`, offline from a
previous run's `profile_export.jsonl` at the same 16 lanes, n=466):

| `LMCACHE_MAX_LOCAL_CPU_SIZE` | tier tokens | pinned across TP8 | in band |
|---|---|---|---|
| 20 GiB/rank | 380,435 | 160 GiB | 4.3% |
| 40 GiB/rank | 760,871 | 320 GiB | 14.4% |
| **64 GiB/rank** | **1,217,394** | **512 GiB** | **18.7%** |
| 80 GiB/rank | 1,521,742 | 640 GiB | 19.7% |
| 160 GiB/rank | 3,043,485 | 1280 GiB | 22.5% |

**64 GiB/rank is the knee.** The reuse-distance p90 is 1,100,136 tokens =
57.8 GiB/rank, so 64 clears it with margin and captures 83% of the population
the tier can ever reach; 80 buys one more point for another 128 GiB pinned, and
everything past 160 buys nothing. Size to the *working set*, not to the pool.

**64 is this workload's knee; the launch above uses 90.** 90 GiB/rank is the
per-rank ceiling this host allows at TP8 (see below) and is what the second
working point ran at. Re-derive the knee for your own traces rather than taking
either number as a constant.

GLM-5.3's rule (`GLM-5.3-LMCache-Byte-Offload.md`, *Sizing the tier*) — tier
~1.5x the **reusable** working set — lands in the same place from a different
direction, and is the cheaper check of the two because it needs no percentiles:

```
reusable set = SUM over distinct conversations of (that conversation's largest ISL) x 56,448 B
             = 580,727 tok x 56,448 B = 30.5 GiB/rank      # 20 conversations, 485 requests
1.57x        = 47.9 GiB/rank                               # GLM-5.3's own multiplier
```

So 64 is 2.10x the reusable set — above the rule, not below it. Note the
denominator: this agentic replay has only **20 distinct conversations** behind
485 requests, so the reusable set is far smaller than the 1,030 GiB/rank of
prompt bytes the run actually issues. Size to the former.

**GLM-5.3's `LMCACHE_MAX_LOCAL_CPU_SIZE=256` is that workload's number, not a
constant, and it cannot be copied here.** It is 1.57x *its* 163.0 GiB/rank
reusable set, at TP4 (1024 GiB pinned). K3 is TP8, so 256/rank would pin
2048 GiB against ~850 GiB of host `MemFree` minus a ~128 GiB engine floor —
a per-rank ceiling of roughly 90 GiB on this machine. Carry the rule across;
the number does not survive the change in TP.

**The host, not the cards, is what runs out.** 512 GiB pinned plus the engine's
own ~128 GiB resident floor at TP8 needs ~640 GiB of `MemFree`, against 838 GiB
here. Check it before launching, and check it **per NUMA node** as well as in
total: on this host node1 had 7.9 GiB free while node0 had 761 GiB, so a tier
that fits the machine can still fail to bind locally. The arm script gates on
`TP x LMC_CPU_GIB x 1.05` and prints both nodes.

Every percentage above is an **upper bound**: landing in the band is necessary
for the tier to help, not sufficient, because the turn still has to be looked up
and returned. Treat it as a filter that rules configurations out, not as a
prediction. After the pair runs, recompute the band from its own
`profile_export.jsonl` rather than carrying these numbers forward.

The gap between the bound and reality was measured, and it is large: the 64
GiB/rank row promises **18.7%** and the tier delivered **0.30%** of prompt
tokens (*Measured*), a factor of 62. The band model asks whether the tier could
hold the prefix; it does not model the fact that on the plugin path vLLM
queries the connector only for what the HBM pool already missed. When the pool
is unpinned and hits 85%, the tier is bidding for the leftover 15% no matter
how it is sized.

**That factor of 62 is a property of that client, not of an unpinned pool.** At
the same unpinned pool, a client whose reuse distance sits above the reusable
prefix leaves HBM at ~11% and the tier delivers ~82% (*A controlled-prefix
client*). So the sentence above should be read as: the bound is loose exactly
when the pool already holds the working set. Check which regime a workload is
in with the band arithmetic before concluding the tier is oversized.

Set `LMC_CPU_GIB`, `MAX_MODEL_LEN`, `CONC` and any pool override identically on
both arms: the band is defined by them, and an arm whose band is empty measures
its own configuration, not the connector.

**Earlier pairs pinned the pool instead, and that is a different regime.** Three
900s/1800s pairs were run with `--num-gpu-blocks-override`, which moves the
band's *lower* edge rather than its upper one. There the sign of the tier's
benefit followed `tier / GPU KV cache size`: at 792 blocks (ratio 0.81) the tier
cost **−3.59%**; at 320 blocks (ratio 2.0, `GPU KV cache size: 381,300 tokens`,
40 GiB/rank) it gained **+10.81%**; and shrinking the pool from 792 to 320 cost
−12.35% without the tier but only −1.47% with it. Those runs also all ran plain
`LRU`, because `LMCACHE_CACHE_POLICY` was unset. They are kept here as the
record of how the ratio behaves, not as a configuration to copy.

### Verify it booted right

On top of the four checks in the M3 recipe:

```bash
# the recurrent leg found its groups. These are TWO different lines from two
# different processes, and each one alone leaves the other side unchecked:
# the EngineCore prints the scheduler-side line, every worker prints its own.
grep "ATOM LMCache offload: recurrent state leg on group" server.log  # EngineCore
grep "ATOM LMCache offload: recurrent state tier up"      server.log  # 1 per worker

# the dense leg strides by blocks, not tokens. `leading dim` is 1536x num_blocks
# on K3 because the MLA backend asks for a kernel block size of 1; the two being
# equal here would mean the codec is striding per token.
grep "ATOM LMCache offload: registered" server.log  # 1 per worker

# HMA must NOT have been turned off -- this line means the connector was not
# recognised as SupportsHMA and the recurrent leg is not running
grep "Turning off hybrid kv cache manager" server.log   # expect no match
```

**Boot: validated** on 8xMI355 TP8 (2026-09-18). Three mamba groups on 0,1,2 and
the attention group on 3, the hybrid memory allocator left on, and the
deterministic smoke test from [Kimi-K3.md](Kimi-K3.md#smoke-test) answering
42. Measured geometry, for comparison against a future boot:

```text
Setting attention block size to 1536 tokens ...   # effective block size, not 128
Available KV cache memory: 37.85 GiB              # per rank
GPU KV cache size: 1,887,436 tokens, Maximum concurrency ... 28.80x
registered 29 layers, num_blocks=1584 (leading dim 2433024, block_size=1536)
recurrent state leg on group(s) 0,1,2 (mamba_block=1536, hash_block=1536, chunk=1536)
recurrent state tier up, 69 layers, entry=58.22 MiB, ... groups=23,23,23
bytes_per_block=25657344 chunk=1536
```

Two numbers there are worth checking rather than skimming, because both fail
silently if they are wrong:

- `bytes_per_block` must be `block_size x bytes_per_token`, here
  `1536 x 16,704 = 25,657,344`. It is what the codec charges for one block-table
  entry. If it comes back as the per-token figure instead, every entry is being
  read as one token and the restored bytes come from the wrong rows — no error,
  no log, just wrong output.
- `leading dim` is `1536 x num_blocks`, not `num_blocks`. The MLA backend asks
  vLLM for a kernel block size of 1, so the cache is allocated one row per token
  while the block ids a connector receives stay manager ids. The two being equal
  would mean this model is not on the kernel-block path and the check above is
  the one that matters.

## Multiprocess tier (alternative to the in-process connector)

Everything above pins the CPU tier inside each TP worker through
`AtomLMCacheOffloadConnector`. LMCache can instead hold the tier in a separate
process that every worker reaches over ZMQ. The model, the quantisation and the
client are unchanged; three things move.

**1. Start the tier before the server.**

```bash
LMCACHE_DISABLE_BANNER=1 lmcache server \
  --host localhost --port 5555 --chunk-size 1536 \
  --l1-size-gb 720 --l1-init-size-gb 600 --l1-align-bytes 16384 \
  --eviction-policy LRU \
  --max-gpu-workers 8 --max-cpu-workers 8 \
  --http-host 127.0.0.1 --http-port 8080 --prometheus-port 9000
# wait for http://127.0.0.1:8080/healthcheck to answer before starting vLLM
```

Four of those flags are sized rather than copied:

* `--max-gpu-workers` / `--max-cpu-workers` must be **TP**, not the default 1.
  One worker thread per pool serialises every STORE/RETRIEVE (GPU pool) and
  every LOOKUP (CPU pool) across all eight ranks.
* `--l1-init-size-gb` pre-reserves the pinned pool at startup instead of growing
  it during the run. Size it to the tier's steady-state footprint, which the
  `/status` endpoint reports as `memory_used_bytes` once the run has plateaued
  (234 GB on the arm below); the 600 here is what was measured and is larger
  than needed.
* `--l1-align-bytes 16384` aligns tier objects to the DMA granularity.

`--chunk-size` must equal the effective KV block (1536 on K3), exactly as
`LMCACHE_CHUNK_SIZE` did. `--l1-size-gb` is the **whole** tier, not a per-rank
share: an in-process `LMCACHE_MAX_LOCAL_CPU_SIZE=N` at TP8 is the same capacity
as `--l1-size-gb 8N`, so the 720 above is the 90 GiB/rank the throughput arms
use.

**2. Swap the connector.** Replace the `--kv-transfer-config` from the Server
section with:

```
--kv-transfer-config '{"kv_connector":"LMCacheMPConnector","kv_connector_module_path":"lmcache.integration.vllm.lmcache_mp_connector","kv_role":"kv_both","kv_load_failure_policy":"recompute","kv_connector_extra_config":{"lmcache.mp.host":"tcp://localhost","lmcache.mp.port":5555,"lmcache.mp.eager_prefetch":true,"lmcache.mp.lazy_offload":true}}'
```

Both `lmcache.mp.*` booleans default to **false** and both are load-bearing:
`eager_prefetch` submits the tier lookup when the request arrives instead of
waiting for the scheduler to poll for it, and `lazy_offload` moves the store
submission out of the forward pass. Neither prints an `lmcache.mp.X = ...` line
at startup — confirm them in the server log's
`kv_connector_extra_config={...}` echo instead.

**3. Drop the environment that no longer applies.** `LMCACHE_*` are not read in
MP mode and `OFFLOAD_*` belong to `AtomLMCacheOffloadConnector`. Unset them
rather than leaving them set, where they read as load-bearing but are not.

### Verifying an MP run

The `Stored`/`Retrieved` counters in the worker log are **structurally zero**
here -- the worker never touches the tier -- so the checks from
[Verify it booted right](#verify-it-booted-right) do not transfer. Use instead:

```bash
curl -s http://127.0.0.1:8080/status | python3 -m json.tool
```

* `storage_manager.l1_manager.total_object_count` and `memory_used_bytes` rising
  across a run is the tier taking writes.
* `memory_total_bytes` is **not** the capacity -- the pinned pool is grown on
  demand, so it climbs toward the ceiling during warmup. The ceiling is
  `memory_configured_bytes`.
* `storage_manager.l1_eviction_controller.trigger_watermark` (0.8) is the
  fraction of `memory_configured_bytes` at which LRU starts evicting.

Server-side hit accounting still comes from vLLM:

```bash
curl -s http://127.0.0.1:8331/metrics | grep -E '^vllm:external_prefix_cache_(hits|queries)_total'
```

Note that this arm does **not** exercise the staging fence or the lookup memo,
both of which live in `AtomLMCacheOffloadConnector`.

### ATOM MP backend (`atom.offload.backend: mp`)

`LMCacheMPConnector` above is LMCache's own connector: it replaces
`AtomLMCacheOffloadConnector` outright, and with it the staging fence, the
lookup memo and K3's recurrent-state leg. The alternative is to keep
`AtomLMCacheOffloadConnector` and move only its *KV tier* into the
`lmcache server` process, by setting `atom.offload.backend` to `mp`. K3's
recurrent state stays on a per-rank in-process pool, which is what
`lmcache.mp.state_transport: own-pool` selects.

**1. Start the tier.** Same binary as above, different flags — this backend
needs the shared-memory pool, and the SHM pool is incompatible with LMCache's
default lazy L1 allocator:

```bash
LMCACHE_DISABLE_BANNER=1 lmcache server \
  --host localhost --port 5555 --chunk-size 1536 \
  --l1-size-gb 192 --eviction-policy LRU \
  --supported-transfer-mode auto --shm-name k3mp_5555 --no-l1-use-lazy \
  --max-gpu-workers 8 \
  --http-host 127.0.0.1 --http-port 8080 --prometheus-port 9000
# wait for http://127.0.0.1:8080/healthcheck to answer before starting vLLM
```

* `--no-l1-use-lazy` is not optional. With the default lazy allocator
  `_compute_shm_pool_info` returns an empty pool, an empty pool silently
  selects `PickleTransferStrategy`, and every chunk then travels over ZMQ —
  the run measures the fallback rather than the path. `--shm-name` is
  likewise inert under the default, with no warning.
* The server drops SHM silently if `/dev/shm` cannot hold `--l1-size-gb`.
  Grep `mpserver.log` for that warning rather than assuming.
* `--l1-size-gb` is the whole tier, not a per-rank share.

**2. Keep the ATOM connector, add the backend key.**

```
--kv-transfer-config '{"kv_connector":"AtomLMCacheOffloadConnector","kv_connector_module_path":"atom.plugin.vllm.kv_transfer.connector","kv_role":"kv_both","kv_load_failure_policy":"recompute","kv_connector_extra_config":{"atom.offload.backend":"mp","lmcache.mp.host":"tcp://localhost","lmcache.mp.port":5555,"lmcache.mp.state_transport":"own-pool","lmcache.mp.state_cpu_size_gb":24,"lmcache.mp.tp_rank_collapse":true}}'
```

* `lmcache.mp.tp_rank_collapse` makes the eight ranks publish one replicated
  KV object instead of eight per-rank ones. Without it the tier holds eight
  copies of the same bytes and the effective capacity is `--l1-size-gb / TP`.
* `lmcache.mp.state_cpu_size_gb` is **per rank** (the state tier is still
  in-process), unlike `--l1-size-gb`.
* `LMCACHE_*` are not exported in this mode — the tier process parses its own
  storage options from its command line. `OFFLOAD_*` still apply, because the
  connector is still ATOM's.

**3. Transfer mode.** `lmcache.mp.mp_transfer_mode` picks which process owns
the gather/scatter kernels: `lmcache_driven` (the default) runs them in the
`lmcache server` process, `engine_driven` runs them in the TP worker. Both are
correct. **Use `engine_driven`** — it is what closes the gap to the in-process
tier (see *Measured*), and at the same settings it also carries a higher tier
share (71.9% vs 55.9% of prompt tokens at conc 16).

```
"lmcache.mp.mp_transfer_mode": "engine_driven"
```

The two modes use the registered tensor shape for different things, which is
why ATOM publishes a different view for each: `engine_driven` sizes objects
from it, `lmcache_driven` addresses pages with it.

### Verifying an ATOM MP run

The worker's `Stored`/`Retrieved` lines are zero here, same as for
`LMCacheMPConnector`, so read the tier from its own endpoint and the hit
accounting from vLLM:

```bash
curl -s http://127.0.0.1:8080/status | python3 -m json.tool   # objects, bytes
curl -s http://127.0.0.1:8331/metrics | \
  grep -E '^vllm:external_prefix_cache_(hits|queries)_total'
```

* `storage_manager.l1_manager.total_object_count` rising across the run is the
  tier taking writes; `write_locked_count` must stay **0**. A non-zero
  `write_locked_count` means a store failed and orphaned its object, and the
  tier fills with objects that can never be read or evicted.
* `grep -c 'ATOM LMCache offload: publishing PAGE layout' server.log` must be
  TP (8), and the recurrent-state leg must report `state tier up` once per rank.

## Client

A controlled-prefix synthetic pool, run once per concurrency rung of the sweep
under *Measured* with `--concurrency` set to the row (16, 20, 24, 32) and
everything else held byte-for-byte. The reuse distance has to be a *chosen*
number for the tier to be measurable at all — a trace replay puts its reuse
inside HBM and never asks the tier. Identical on both arms:

```bash
aiperf profile --url "http://127.0.0.1:8713" \
  --endpoint /v1/chat/completions --endpoint-type chat --streaming --model "${MODEL}" \
  --tokenizer "${MODEL}" --tokenizer-trust-remote-code \
  --isl 4608 --isl-stddev 0 --osl 512 --osl-stddev 0 \
  --prompt-prefix-length 27648 --prompt-prefix-pool-size 16 \
  --num-dataset-entries 256 --concurrency 16 \
  --benchmark-duration 1800 --benchmark-grace-period 60 --stats-interval 30 \
  --extra-inputs ignore_eos:true --use-server-token-count \
  --cache-bust first-turn-suffix --random-seed 530419 \
  --request-timeout-seconds 3600 --no-gpu-telemetry \
  --server-metrics "http://127.0.0.1:8713/metrics"
```

The geometry is chosen, not inherited. `--prompt-prefix-pool-size 16` with
`27648 + 4608` tokens per request puts the reuse distance at
`d = (16 - 1) x 32,256 = 483,840` tokens, which is above the reusable prefix
(270,336 at conc 16) and below the tier (1,711,961 at 90 GiB/rank) — i.e. inside
the band *Sizing the tier* defines. `--cache-bust first-turn-suffix` keeps each
request's tail unique so the prefix is the only thing that can be reused;
`--prompt-prefix-length` must be a multiple of the 1536-token chunk or the chunk
ends do not land on prefix boundaries.

`--max-context-length` is deliberately **not** passed: aiperf implements it only
for trace datasets and rejects it on a synthetic one.

Four of these flags are load-bearing and none is cosmetic:

* `--random-seed` fixes the dataset, so both arms replay the same prompts. Pass
  it explicitly; a default would let the arms diverge silently.
* `--use-server-token-count` makes every hit rate below a ratio of two
  server-side counters rather than of a client-side estimate, so the
  denominator cannot drift between arms.
* `--osl-stddev 0` with `ignore_eos:true` pins every response at exactly 512
  tokens; otherwise the length distribution contaminates ITL and throughput.
* `--num-dataset-entries 256` is sized to the window rather than copied from a
  shorter run: a pool that the duration exhausts replays prompts and inflates
  the late window on both arms.

`--enable-prompt-tokens-details` is in the launch above. Without it
`usage.prompt_tokens_details.cached_tokens` is absent from every exported
record, aiperf's own prompt-cache columns come back empty, and it says so at the
foot of its summary. The arms in *Measured* ran without it, which is why every
hit rate there is read from the server's `/metrics` instead. Whichever way you
go, set it on both arms — never on one.

## Measured

All numbers below were taken on the PR #2369 tree, vLLM 0.28
plugin path, image `rocm/atom-dev:vllm-0.28.0`, TP8 on eight gfx950 GPUs.

### Throughput and latency

Matched ON/OFF pairs, 1800 s per arm, one arm at a time in the same slot, arm
order alternated within each pair. Client: the controlled-prefix synthetic pool
from *Client* above — `PREFIX_LEN=27648`, `GEN_ISL=4608`, `GEN_OSL=512`,
`PREFIX_POOL=16`, `ENTRIES=256`, `SEED=530419`, `--max-model-len 65536`. Server:
the launch in *Server*, tier 90 GiB/rank, `LMCACHE_CHUNK_SIZE=1536`,
`LMCACHE_CACHE_POLICY=ATOM_SLRU`. The two arms differ in exactly one knob —
whether the connector is loaded — and every other knob is diffed out of the two
arms' `run.env` rather than assumed equal.

| conc | arm | tok/s/GPU | req/s | TTFT p50/p90 (ms) | ITL p50/p90 (ms) | HBM hit | tier share | n |
|---|---|---|---|---|---|---|---|---|
| 16 | OFF | 65.24 | 1.0194 | 629/1182 | 28.66/30.84 | 79.82% | 0.00% | 1843 |
| 16 | ON | 85.05 | 1.3289 | 670/879 | 20.96/25.46 | 10.65% | 82.19% | 2403 |
| 16 | delta | **+30.36%** | +30.36% | +6.44%/-25.67% | -26.87%/-17.44% | | | |
| 20 | OFF | 66.91 | 1.0455 | 642/1307 | 35.19/37.70 | 79.89% | 0.00% | 1890 |
| 20 | ON | 88.28 | 1.3794 | 586/873 | 25.76/29.39 | 9.36% | 83.60% | 2493 |
| 20 | delta | **+31.93%** | +31.93% | -8.69%/-33.15% | -26.82%/-22.02% | | | |
| 24 | OFF | 75.25 | 1.1758 | 777/1443 | 37.47/40.54 | 79.93% | 0.00% | 2127 |
| 24 | ON | 103.27 | 1.6136 | 682/948 | 26.32/29.40 | 8.21% | 85.08% | 2918 |
| 24 | delta | **+37.23%** | +37.23% | -12.25%/-34.32% | -29.74%/-27.49% | | | |
| 32 | OFF | 80.78 | 1.2622 | 792/1457 | 46.90/50.30 | 80.02% | 0.00% | 2283 |
| 32 | ON | 115.69 | 1.8076 | 709/1043 | 31.47/34.46 | 8.37% | 85.02% | 3270 |
| 32 | delta | **+43.21%** | +43.21% | -10.53%/-28.40% | -32.89%/-31.49% | | | |

`HBM hit` and `tier share` are both fractions of prompt tokens, computed from
the server counters as end-minus-start deltas over the measured window.

#### Multiprocess tier

The three arms below were taken together in the same slot on image
`rocm/atom-dev:vllm-v0.28.0-nightly_20260928-lmcache-v0.10`, so they are
comparable to each other but not to the table above, which is a different
image. Client as in *Client*, `--concurrency 16`, 1800 s per arm, seed 530419.
The MP arm uses the `lmcache server` flags and the two `lmcache.mp.*` booleans
from *Multiprocess tier* above.

| arm | tok/s/GPU | req/s | TTFT p50 (ms) | ITL p50 (ms) | HBM hit | tier share | n |
|---|---|---|---|---|---|---|---|
| OFF | 66.44 | 1.038 | 626 | 28.16 | 78.4% | 0.00% | 1877 |
| ON in-process | **87.01** | 1.359 | 651 | 20.54 | 18.6% | 81.5% | 2457 |
| ON multiprocess | 64.60 | 1.010 | 1127 | 28.47 | 38.3% | 56.3% | 1824 |

The multiprocess tier reaches the in-process tier's total cache coverage but not
its throughput: its residual cost is TTFT, which stays ~500 ms above OFF because
a tier lookup is a cross-process round trip. Use the in-process connector unless
the tier has to be shared across engines.

#### ATOM MP backend

Arms taken one at a time on image
`rocm/atom-dev:vllm-v0.28.0-nightly_20260928-lmcache-v0.10`
(vLLM `0.28.1.dev0+g2cf0a6915`, LMCache `0.5.5rc3+rocm7.2.4.torch2.10`), TP8 on
eight gfx950 GPUs, **900 s** per arm, `SEED=1234`, `--max-model-len 65536`,
`--gpu-memory-utilization 0.85`, `--block-size 128`,
`cudagraph_mode=FULL_AND_PIECEWISE`. Client: the controlled-prefix synthetic
pool from *Client* — `PREFIX_LEN=27648`, `GEN_ISL=4608`, `GEN_OSL=512`,
`PREFIX_POOL=16`, `ENTRIES=256`; measured ISL p50 32334, OSL 512. Tier: 90
GiB/rank in-process (`LMCACHE_MAX_LOCAL_CPU_SIZE=90`), 192 GiB whole-tier for
MP (`--l1-size-gb 192`) plus 24 GiB/rank for the recurrent state.

`--concurrency 16`:

| arm | tok/s/GPU | req/s | per-user tok/s p50 | TTFT p50 (ms) | ITL p50 (ms) | HBM hit | tier share | n |
|---|---|---|---|---|---|---|---|---|
| OFF (no connector) | 65.02 | 1.0159 | 35.46 | 627 | 28.20 | 78.88% | 0.00% | 923 |
| ON in-process | 81.57 | 1.2745 | **48.12** | 670 | 20.78 | 23.04% | 67.43% | 1158 |
| ON MP, `lmcache_driven` | 60.80 | 0.9500 | 42.49 | 795 | 23.53 | 33.99% | 55.89% | 912 |
| ON MP, `engine_driven` | 82.43 | 1.2880 | **48.97** | 763 | 20.42 | 18.84% | 71.86% | 1170 |

`engine_driven` was measured twice at this concurrency; the second arm read
57.87 tok/s/GPU and 47.52 per-user tok/s p50. The p50 figures differ by 3%
between the two, the aggregate by 42%: the lower arm took a burst of 79-103 s
TTFT outliers (p99 78,953 ms against 8,753 ms, zero errors) which moves the
aggregate without moving the p50. Read the per-user p50 as the stable number
here and treat a single aggregate reading as provisional.

Across concurrency — same client, same knobs, only `--concurrency` varies, and
the in-process and MP arms alternate within one batch rather than running as
two blocks:

| conc | in-process per-user p50 | MP `engine_driven` per-user p50 | in-process tok/s/GPU | MP tok/s/GPU |
|---|---|---|---|---|
| 8 | 67.36 | 67.35 | 57.00 | 56.53 |
| 16 | 48.12 | 47.52 - 48.97 | 81.57 | 57.87 - 82.43 |
| 32 | 31.62 | 32.70 | 109.26 | 110.59 |

MP with `engine_driven` matches the in-process tier at all three points: the
deviations are -0.015%, bracketing, and +3.4%, i.e. they do not share a sign,
which is what a difference below the arm-to-arm noise looks like. The same
sweep with `lmcache_driven` read -9.6% / -11.1% / -7.5%.

`HBM hit` and `tier share` are fractions of prompt tokens, computed from the
server counters as end-minus-start deltas over the measured window;
`tok/s/GPU` is `req/s x OSL / TP`. The conc-16 OFF and in-process rows come
from an earlier batch than the two MP rows; the across-concurrency table is
single-batch throughout, and its conc-32 in-process arm reads 31.62 against
31.55 for the earlier batch, which is how the two batches were checked to be
comparable.

### Accuracy

gsm8k, all 1319 questions, 12-shot, greedy:

```bash
lm_eval run --model local-chat-completions \
  --model_args "model=amd/Kimi-K3,base_url=http://127.0.0.1:8331/v1/chat/completions,num_concurrent=64,max_retries=3,max_gen_toks=16384,timeout=1800,tokenized_requests=False" \
  --tasks gsm8k --num_fewshot 12 --apply_chat_template --fewshot_as_multiturn \
  --gen_kwargs temperature=0,top_p=1 --seed 0,1234,1234,1234 --limit 1319 \
  --log_samples --output_path <run>/<label>
```

Server for the accuracy points: `--num-gpu-blocks-override 64`,
`--max-model-len 32768`, tier 24 GiB/rank. The small HBM pool is the point: it
cannot hold the corpus, so pass 2 can only come back from the CPU tier.

| point | connector | pass | gsm8k exact_match (flexible) |
|---|---|---|---|
| m1_base | off | single | 0.9644 ± 0.0051 |
| m2_store | on | 1 (fills the tier) | 0.9606 ± 0.0054 |
| m3_load | on | 2 (reads the tier) | 0.9689 ± 0.0048 |

`m2_store`/`m3_load` ran in 7 chunks of 189 questions, one server boot per
chunk, because a 64-block pool plus a 24 GiB tier cannot hold 1319 questions at
once; the figures are the aggregate over all 1319.

#### ATOM MP, `engine_driven`

The table above was taken with the in-process tier. `engine_driven` is
validated with the same two-pass idea on a smaller corpus, because the limit
is the tier's capacity rather than the eval's:

```bash
ARM=atommp CLIENT=gsm8k CONC=32 ACC_CONC=32 ACC_SHOTS=64 ACC_LIMIT=120 \
  NUM_GPU_BLOCKS_OVERRIDE=500 MAX_MODEL_LEN=16384 \
  TP_RANK_COLLAPSE=true STATE_TRANSPORT=own-pool \
  MP_L1_GIB=192 STATE_GIB=24 LMC_CPU_GIB=90 \
  MP_TRANSFER_MODE=engine_driven SEED=530419
```

64-shot (~10,506 prompt tokens) so prompts clear both the 1536-token block and
`OFFLOAD_MIN_LOAD_TOKENS=8192`; `ACC_LIMIT=120` so one pass (1,260,720 tok)
fits the tier and pass 1 is not evicted before pass 2 reads it.

| arm | pass 1 | pass 2 | delta |
|---|---|---|---|
| OFF (noise floor) | 0.9833 | 0.9917 | +1 question |
| ON MP, `engine_driven` | 0.9917 | 0.9833 | -1 question |

Pass 2 of the ON arm served 966,144 prompt tokens out of the tier against
1,249,858 queried (77.3%), with `vllm:prefix_cache_hits_total` at **exactly
0** — the 500-block HBM pool holds nothing, so the tier is the only place
those tokens can have come from. The ON arm's -1 question is the same size as
the OFF arm's own +1: at concurrency the engine is not bit-deterministic, and
that is the noise floor. At 120 questions one question is 0.83%, so this
rules out a structural byte error but not a sub-question bias.

#### How this run measures LMCache

An accuracy run only tests the tier if the bytes it read actually came from the
tier. Three things make that true, and all three have to hold:

1. **Two passes over the same corpus, one server boot each.** Pass 1
   (`m2_store`) starts with an empty tier, so it can only compute -- its job is
   to fill the tier. Pass 2 (`m3_load`) replays the identical questions with
   the identical `--seed` and `--num_fewshot`, so every prompt already has an
   entry.
2. **An HBM pool too small to hold the corpus** -- that is what
   `--num-gpu-blocks-override 64` is for. Without it pass 2's hits land in
   vLLM's own prefix cache and the tier is never asked; with it the prefix is
   evicted long before it is reused, so the CPU tier is the only place it can
   come back from. The same constraint is why the corpus is split into chunks:
   the pool plus the tier has to hold one chunk, not the whole 1319.
3. **Confirm on the server side, not from the score.** The run counts as a tier
   measurement only if the pass-2 window actually shows tier traffic:

```bash
curl -s http://127.0.0.1:8331/metrics | \
  grep -E 'vllm:external_prefix_cache_(hits|queries)_total'   # hits > 0 on pass 2
grep -c 'Retrieved' server.log                                # 0 on pass 1, > 0 on pass 2
```

Only then is the score comparison meaningful: `m1_base` is the no-connector
reference and `m3_load` is the same questions answered from restored bytes. A
real KV corruption moves gsm8k by much more than one standard error, so
agreement inside one standard error is the pass criterion.

## Related

- [Kimi-K3](Kimi-K3.md) — the base recipe: prerequisites, launch, accuracy,
  DSpark speculative decoding
- [MiniMax-M3 LMCache Byte Offload](MiniMax-M3-LMCache-Byte-Offload.md) — the
  LMCache build steps and the four generic "is it on" checks this recipe
  extends
- [GLM-5.2 LMCache Byte Offload](GLM-5.2-LMCache-Byte-Offload.md) — the full
  treatment of this connector: mechanism, sizing, tuning, gotchas
- `GLM-5.3-LMCache-Byte-Offload.md` — the same connector with **no**
  model-specific code, and the reporting protocol this recipe's *Measured*
  section follows. Not linked because it is not on `main` yet; it lands with
  GLM-5.3's own PR.
- [LMCache KV Cache Offload](LMCache-KV-Cache-Offload.md) — generic plugin path
  (`LMCacheConnectorV1`); does **not** support K3's multi-group registration
