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
plugin, LMCache 0.5.5rc3. This is the complete launch the numbers in *Measured*
came from — nothing is elided.

```bash
export AITER_LOG_LEVEL=WARNING
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export PYTHONHASHSEED=0                 # mandatory -- see below
export LMCACHE_LOCAL_CPU=True
export LMCACHE_MAX_LOCAL_CPU_SIZE=64    # GiB **per TP rank**; see Sizing the tier
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

## Client

The workload is an **agentic trace replay**, not a synthetic prefix pool — there
is no `--prompt-prefix-pool-size` to reason about, and reuse comes from the
trace's own multi-turn structure. Identical on both arms:

```bash
aiperf profile --scenario inferencex-agentx-mvp \
  --public-dataset semianalysis_cc_traces_weka_062126 \
  --url "http://127.0.0.1:8713" --endpoint /v1/chat/completions \
  --endpoint-type chat --streaming --model "${MODEL}" \
  --concurrency 16 --benchmark-duration 1800 --random-seed 1234 \
  --num-dataset-entries 800 --max-context-length 65536 \
  --warmup-requests-per-lane 10 --warmup-grace-period 1800 \
  --trajectory-start-min-ratio 0.25 --trajectory-start-max-ratio 0.75 \
  --trace-idle-gap-cap-seconds 300 --use-server-token-count \
  --tokenizer "${MODEL}" --tokenizer-trust-remote-code \
  --server-metrics "http://127.0.0.1:8713/metrics" --no-gpu-telemetry
```


`--num-dataset-entries 800` is sized to the window, not copied from a shorter
run: 900 s drew 416 records against 400 entries, already at the replay edge, so
doubling the duration without doubling the pool would have replayed prompts and
inflated the late window on both arms.

Four of these are load-bearing and none is cosmetic:

* `--random-seed` fixes the replay, so both arms draw the same trajectories.
  Pass it explicitly; a default would let the arms diverge silently.
* `--use-server-token-count` makes every hit rate below a ratio of two
  server-side counters rather than of a client-side estimate, so the
  denominator cannot drift between arms.
* `--max-context-length 65536` must equal the server's `--max-model-len`. This
  trace's input sequences run past it (median ISL ~74k against a 65,536
  window), so the two numbers disagreeing truncates the workload asymmetrically
  rather than the pool.
* `--warmup-requests-per-lane 10` is a **count** budget, not a time budget, and
  `--warmup-grace-period` does not cap it. At this ISL warmup ran for minutes;
  it is excluded from the exported records (`benchmark_phase`) and must not be
  extrapolated into the measured window.

`--enable-prompt-tokens-details` is in the launch above. Without it
`usage.prompt_tokens_details.cached_tokens` is absent from every exported
record, aiperf's own prompt-cache columns come back empty, and it says so at the
foot of its summary. The pair in *Measured* ran without it, which is why every
hit rate there is read from the server's `/metrics` instead. Whichever way you
go, set it on both arms — never on one.

## Measured

A matched ON/OFF pair, back to back in the same slot, TP8 on MI355, 1800 s per
arm, conc 16, seed 1234, 800 client entries. This pair **is** the launch in
*Server*: no `--num-gpu-blocks-override`, `LMCACHE_MAX_LOCAL_CPU_SIZE=64`,
`LMCACHE_CACHE_POLICY=ATOM_SLRU` (read back from the rank-0 config dump, not
from the launch env), `--enable-prompt-tokens-details` on. The arms differ in
exactly one thing: whether the connector is loaded. That is checked rather than
asserted — the pair check reads the effective tier out of each arm's **log**
and diffs every other knob.

Both arms ran `atom_head=cd4c5153c`. One uncontrolled difference is recorded
rather than hidden: `atom_dirty` was 0 on ON and 1 on OFF, because this recipe
file itself was edited between the arms. The dirty path is
`recipes/atom_vllm/Kimi-K3-LMCache-Byte-Offload.md` — documentation, no code.

### The result: no measurable difference

| full window, 1800 s | req/s | out tok/s | TTFT p50 / p90 | ITL p50 / p90 | preempt | n |
|---|---|---|---|---|---|---|
| OFF | 0.2645 | 116.26 | 519 / 1156 ms | 16.46 / 28.52 ms | 0 | 484 |
| ON | 0.2650 | 119.84 | 543 / 1225 ms | 16.68 / 29.23 ms | 0 | 485 |
| | +0.21% | +3.08% | +4.4% / +6.0% | +1.3% / +2.5% | | |

**Do not read +3.08% as a gain.** Two rulers over the same two arms straddle
zero: aiperf's own `req/s` gives **+0.21%**, and recomputing the rate over each
arm's first-to-last record span gives **−3.73%** (the arms' record spans are
not equal). With `n=1` per arm and no noise floor, the *sign* is undetermined.
The honest reading is that the tier changed nothing measurable here.

The validity control also fails, so no steady-state row is quoted either: the
start-cut sweep is licensed only when the OFF arm is flat, and OFF drifts
−6.58% across cuts (ON −9.11%). The ON/OFF ratio stays negative at every cut
(−1.7% to −6.3%), which is at least consistent — it does not turn positive if
you pick a later window.

### Why the tier had nothing to do

`vllm:prompt_tokens_by_source_total`, whose three components sum to
`vllm:prompt_tokens_total` exactly on both arms:

| source | OFF | ON |
|---|---|---|
| `local_compute` | 4,074,976 (14.89%) | 3,969,906 (14.65%) |
| `local_cache_hit` | 23,288,832 (85.11%) | 23,049,216 (85.05%) |
| `external_kv_transfer` | 0 | **81,408 (0.30%)** |
| total | 27,363,808 | 27,100,530 |

**The unpinned HBM pool already answers 85.05% of prompt tokens by itself.** On
the plugin path vLLM asks the connector only about what the pool missed, so the
tier's entire addressable market is the remaining 14.9% — and within that miss
tail it answered **2.01%**. Prefill recompute moved 14.89% → 14.65%, i.e. by
0.24 pp.

This is the structural point of this configuration, and it is not a tier-sizing
problem. Compare the pinned-pool pair at the end of *Sizing the tier*: with the
pool cut to 320 blocks the same tier supplied **44.12%** of prompt tokens and
bought +10.81% req/s. Same model, same client, same connector — the only
difference is how much of the working set HBM was allowed to keep. **The tier's
power is set by the HBM pool, not by `LMCACHE_MAX_LOCAL_CPU_SIZE`.**

### The instruments agreed, for once

| ruler | ON | meaning |
|---|---|---|
| `vllm:external_prefix_cache_hits` | 81,408 | what scheduler-side lookup **promised** |
| LMCache `Retrieved` sum / TP=8 | 81,408 | what the ranks **delivered** |
| `prompt_tokens_by_source` external | 81,408 | what the engine **counted** |

Gap 0 on both hops: 392 retrieve lines, all full, no partials, and
`Sum(required − retrieved)` is 0 — no chunk was evicted between lookup and
load, so `kv_load_failure_policy: recompute` was never exercised. On the OFF
arm every external counter is exactly 0, which is the control this pair needs.

`Failed to allocate memory block ... no memory is available` appears 1,992
times on the ON arm. Read it against *Sizing the tier* rather than as an
under-size verdict: the per-request cache-bust tails are stored, never reused,
and evicted, which is what the tier should do with them. The reusable set here
is 30.5 GiB/rank and the tier was 64.

**Cost of this configuration.** Pre-allocating 8 x 64 GiB of pinned host memory
took about 30 minutes, with host `MemFree` falling 838 → 191 GiB. That is the
real gate at TP8, and it is why GLM-5.3's 256 GiB/rank cannot be copied across
(see *Sizing the tier*).

**Throughput is all that was measured; accuracy was not.** When it is run, use
the two-pass method: a single SAVE-only pass measures nothing, so salt the
prefixes to defeat the GPU prefix cache, size the HBM pool below the working
set to force read-back, and take the noise floor from the OFF arm's own
two-pass delta. The baseline to beat is the one in
[Kimi-K3.md](Kimi-K3.md#accuracy-validation).

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
