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

Add to the DSpark launch in
[Kimi-K3.md](Kimi-K3.md#speculative-decoding-with-dspark) (prefix caching and
`--mamba-cache-mode align` are already there and are both **mandatory** for
offload):

```bash
export PYTHONHASHSEED=0              # mandatory, see Gotchas in the M3 recipe
export LMCACHE_LOCAL_CPU=True
export LMCACHE_MAX_LOCAL_CPU_SIZE=40 # GiB **per TP rank** -- TP8 x 40 = 320 GiB pinned
export LMCACHE_CHUNK_SIZE=1536       # multiple of the *effective* block size,
                                     # which vLLM raises to 1536 -- see below
export OFFLOAD_MIN_LOAD_TOKENS=256   # default 8192 disables the tier for chat-sized prompts

vllm serve "${MODEL}" \
    ... the DSpark flags from Kimi-K3.md ... \
    --kv-transfer-config '{"kv_connector":"AtomLMCacheOffloadConnector","kv_connector_module_path":"atom.plugin.vllm.kv_transfer.connector","kv_role":"kv_both","kv_load_failure_policy":"recompute"}'
```

Three settings are K3-specific and each one is a hard failure if wrong:

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
  out of the log rather than assuming the flag won. Only
  chunk-aligned boundaries are stored and only chunk-aligned boundaries are
  probed on lookup, so a chunk that ends between two boundaries can never
  produce a usable pair. The connector validates this at construction and names
  both numbers if it does not hold.

Hybrid models also require vLLM's hybrid memory allocator, which vLLM
auto-disables for a connector that does not declare `SupportsHMA` — so without
that declaration K3 plus `--kv-transfer-config` does not mis-save, it does not
boot. The connector declares it; the check below confirms HMA stayed on.

### Sizing the tier

Per rank K3 costs **56,448 B/token** — `29 x 576 = 16,704` for attention plus
`58.22 MiB / 1536 = 39,744` for the recurrent boundary state, so the recurrent
leg is 2.4x the attention leg and dominates the tier.

The tier can only do work for a turn whose reuse distance — the KV volume other
requests write between the turn that produced a prefix and the turn that wants
it back — lands in the half-open band `[HBM pool, tier)`. Shorter than the pool
and HBM already had it; longer than the tier and both missed. So size the pool
first, and count the population in the band before booting anything.

**The pool is not `GPU KV cache size`.** That log line is
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

At the defaults that leaves `(1584 - 16 x 55) / 4 x 1536 = 270,336` tokens of
reusable prefix against a `20 GiB / 56,448 = 380,435`-token tier: a band barely
one part wide. Measured against the reuse distances of this very workload
(`inferencex-agentx-mvp` on `semianalysis_cc_traces_weka_062126`, two
independent K3 runs, offline from each run's `profile_export.jsonl`), **2-3% of turns fall in it** —
the arm can only return a null result, and would do so no matter how the
connector behaved.

| `--num-gpu-blocks-override` | concurrency | reusable prefix | in band @4 lanes | @16 lanes |
|---|---|---|---|---|
| 1584 (default) | 28.8 | 270,336 tok | 2-3% | 8-16% |
| 1300 | 23.6 | 161,280 tok | 16-22% | 18-35% |
| **1100** | **20.0** | **84,480 tok** | **44-45%** | **36-46%** |
| 1000 | 18.2 | 46,080 tok | 71-72% | 38-65% |

The 4-lane column is measured directly on two independent runs. The 16-lane
column extrapolates one doubling from a third, measured at 8 lanes in the same
sweep, over the range that brackets its own 4-to-8 ratio.

**Shrink the pool; do not grow the tier.** Making the tier merely exceed the
*default* pool would want ~107 GiB/rank, i.e. ~853 GiB pinned across TP8 — the
arithmetic runs into the host long before it runs into the cards. At 1100 blocks
the default `20 GiB/rank` tier is already the larger end of the band, so the
measurement needs no extra host memory at all.

**Do not shorten `--max-model-len` to shrink the pool.** This trace's input
sequences run past it already (median ISL ~74k tokens against a 65,536 window),
so a shorter window truncates the workload rather than the pool.

Every percentage above is an **upper bound**: landing in the band is necessary
for the tier to help, not sufficient, because the turn still has to be looked up
and hit. Only a zero is a hard result.

Reuse distance grows close to **linearly** in lanes. Measured within one K3
sweep, 4 to 8 lanes moved it x1.90 at p75 and x1.97 at p50, against prompt-length
distributions 7% apart. Do not measure this across models: a cross-model pair
varies block size, bytes per token and the prompt-length distribution at the same
time as the lane count, and attributes the product to lanes alone — doing exactly
that produced a "sub-linear x2.3" here that the within-model axis then refuted.
The 1-lane point in the same sweep is no good for it either, for the same reason
in miniature: its median input sequence is 1.8x the other two.

The choice of 1100 survives the whole range anyway — it holds 36-46% at 16 lanes
where the default holds 8-16%. After the OFF arm runs, recompute the band from
its own `profile_export.jsonl` at the concurrency actually used, rather than
carrying any of these numbers forward.

Set `NUM_GPU_BLOCKS_OVERRIDE`, `MAX_MODEL_LEN`, `CONC` and `LMC_CPU_GIB`
identically on both arms: the band is defined by the first and last of them, and
an arm whose band is empty measures its own configuration, not the connector.

**What was actually run was 320 blocks, not 1100.** The table above is an upper
bound on the in-band population, and 1100 is where it first gets large; the pair
in *Measured* went further down, to 320 blocks (`GPU KV cache size: 381,300
tokens`) at 40 GiB/rank, which puts the tier at 2.0x the pool instead of roughly
level with it. That was chosen from one line of arithmetic before booting
anything, and it is the reason the result is a clear +10.81% rather than the
null this measurement returns at the default pool. Two earlier pairs at this
same window bracket it: at 792 blocks (tier/pool = 0.81) the tier cost **−3.59%**,
and shrinking the pool from 792 to 320 costs −12.35% without the tier but only
−1.47% with it. **The sign of the tier's benefit follows `tier / pool`**, and
that ratio is the number to compute first.

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

One thing the base launch does **not** pass is `--enable-prompt-tokens-details`,
so aiperf's own prompt-cache columns come back empty and it says so at the foot
of its summary. That is why every hit rate in *Measured* is read from the
server's `/metrics` instead. If you want aiperf to compute them too, add the
flag on both arms — never on one.

## Measured

A matched ON/OFF pair, back to back in the same slot, same tree
(`0ad111463`, `atom_dirty=0` recorded at the start of each arm), same model,
TP8 on MI355, 1800 s per arm, seed 1234. The arms differ in exactly one thing:
whether the connector is loaded. That is checked rather than asserted — the
pair check reads the effective tier out of each arm's **log** (`recurrent state
tier up` / its absence) instead of off the launch environment, and diffs every
other knob.

The server is the block in *Server* with `LMCACHE_MAX_LOCAL_CPU_SIZE=40` and
`--num-gpu-blocks-override 320`; both arms booted `GPU KV cache size: 381,300
tokens`. The client is the block in *Client*, identical on both arms.

| full window, 1800 s | req/s | out tok/s | TTFT p50 / p90 | ITL p50 / p90 | preempt |
|---|---|---|---|---|---|
| OFF | 0.2397 | 106.28 | 1367 / 5266 ms | 22.28 / 52.79 ms | 35 |
| ON | **0.2656** | **120.35** | 559 / 1997 ms | 19.79 / 30.48 ms | 2 |
| | **+10.81%** | **+13.24%** | −59.1% / −62.1% | −11.2% / −42.3% | |

**Where the prompt tokens went.** `vllm:prompt_tokens_by_source_total`, whose
three components sum to `vllm:prompt_tokens_total` exactly on both arms:

| source | OFF | ON |
|---|---|---|
| `local_compute` | 16,134,428 (64.01%) | **4,428,242 (16.27%)** |
| `local_cache_hit` | 9,071,616 (35.99%) | 10,781,184 (39.61%) |
| `external_kv_transfer` | 0 | **12,006,912 (44.12%)** |
| total | 25,206,044 | 27,216,338 |

Prefill recompute fell by 47.7 pp of prompt tokens, which is where the whole
result comes from: TTFT moves most, ITL follows second-hand as prefill stops
competing with decode, and preemptions fall 35 → 2 because a preempted request
resumes off the tier instead of re-prefilling.

Unlike GLM-5.3, **the HBM hit rate also rose** (35.99% → 39.61% of prompt
tokens) rather than falling slightly. The two percentages still have different
denominators — on the plugin path vLLM asks the connector only about what the
HBM pool missed — so they must not be summed as shares of one thing. Within the
miss tail the tier answered **73.06%**.

**Three instruments, two gaps.** They do not measure the same thing and the
difference is informative rather than noise:

| ruler | ON | meaning |
|---|---|---|
| `vllm:external_prefix_cache_hits` | 12,498,432 | what scheduler-side lookup **promised** |
| LMCache `Retrieved` sum / TP=8 | 12,189,312 | what the ranks **delivered** |
| `prompt_tokens_by_source` external | 12,006,912 | what the engine **counted** |

The promised→delivered gap of 309,120 closes exactly against
`Sum(required − retrieved)` over every rank line ÷ 8 — these are 10 KV load
failures, i.e. chunks evicted between lookup and load, absorbed by
`kv_load_failure_policy: recompute`. Compute that sum over **all** rank lines;
the per-event shortcut `affected × failed_ranks / TP` silently misses partial
retrievals (28 of the 2,568 retrieves here came back partial) and overstated the
gap by 39% on this arm. The delivered→counted gap of 182,400 is **not**
explained and is left open rather than papered over.

**The validity control fails, so no steady-state row is quoted.** The start-cut
sweep GLM-5.3 uses is only licensed when the OFF arm is flat across cuts. Here
it is not: OFF drifts −17.1% from t ≥ 0 to t ≥ 1200 (ON −10.5%), because the
replay's per-request work grows over the window on both arms. What can be said
is that the **ratio** is stable and non-monotone in the cut — +9.86%, +11.47%,
+12.86%, +11.44%, +13.56%, +14.68%, +10.54%, +18.57% at cuts 0/100/200/300/450/
600/900/1200 s — so the drift is common-mode and the full-window +10.81% is not
an artefact of where the window starts. It is **not** grounds for quoting the
+14.68% at t ≥ 600 as a steady-state gain.

**Limits.** Each arm is **n=1 in runs**; no dispersion is quoted and the cut
sweep bounds within-run drift, not run-to-run variance. Two quantities were not
held fixed and are stated rather than hidden: ISL differs by +1.24% and OSL by
+2.20% between arms (the replay draws different trajectories once throughput
differs), and host `MemFree` at launch was 804 GiB (OFF) against 1262 GiB (ON)
because neighbours released memory in between — neither arm was near the
ceiling, but the pair is not a controlled test of host pressure. The tier is
**under-sized on purpose** at 40 GiB/rank: `Failed to allocate memory block`
appears 8,847 times on the ON arm, so this is a working tier with eviction, not
a tier that holds the working set.

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
