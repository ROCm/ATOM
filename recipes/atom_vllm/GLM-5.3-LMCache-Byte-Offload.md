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

Identical on both arms. Four of those flags are load-bearing and none is
cosmetic:

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

and this workload's distance is `(pool_size - 1) x ISL` at ISL = 32,768. Pool
64 gives 2,064,384 — **below** the HBM pool, so the tier's hit rate is zero by
construction and a larger tier cannot help, because that only moves the top of
the band. Pool 128 gives 4,161,536: 20.9% above the floor, 27.8% below the
ceiling. This arithmetic is the reason the pool is 128, and it has to be done
before the run rather than discovered after it.

## Measured

All numbers below come from the `atom` branch of PR #2369, vLLM 0.28 plugin
backend in `rocm/atom-dev:vllm-0.28.0`, TP=4 on gfx950 GPUs 0-3.

### Throughput and latency

Matched ON/OFF pairs: one arm per concurrency per setting, 1800 s of traffic
each, seed 530419, the *Client* line above verbatim with `--concurrency` set to
the row. The two arms of a pair differ in exactly one thing -- whether
`--kv-transfer-config` is on the server line. `tput/GPU` is
`output_token_throughput / 4`; output length is pinned at 512, so `req/s` is
the same measurement and is not a second result.

| conc | arm | tput/GPU (tok/s) | req/s | TTFT p50 / p90 (ms) | ITL p50 / p90 (ms) | HBM prefix hit | CPU tier share |
|---|---|---|---|---|---|---|---|
| 8 | OFF | 89.00 | 0.6953 | 2216 / 4259 | 17.37 / 20.99 | 60.54% | 0.00% |
| 8 | ON | **103.47** | 0.8083 | 1286 / 2442 | 15.68 / 17.99 | 59.65% | 24.02% |
| 8 | delta | **+16.26%** | +16.26% | -41.95% / -42.66% | -9.74% / -14.29% | | |
| 16 | OFF | 124.56 | 0.9732 | 3607 / 6130 | 24.63 / 30.70 | 61.61% | 0.00% |
| 16 | ON | **159.23** | 1.2440 | 2375 / 3175 | 19.47 / 23.15 | 59.99% | 26.32% |
| 16 | delta | **+27.83%** | +27.83% | -34.16% / -48.21% | -20.95% / -24.59% | | |
| 32 | OFF | 158.84 | 1.2409 | 2799 / 6133 | 43.08 / 52.11 | 62.06% | 0.00% |
| 32 | ON | **229.71** | 1.7946 | 3422 / 5162 | 25.96 / 31.90 | 60.52% | 27.12% |
| 32 | delta | **+44.61%** | +44.61% | +22.26% / -15.83% | -39.74% / -38.78% | | |

`HBM prefix hit` and `CPU tier share` are read from the server's own counters
as end-minus-start deltas over the measured window, not from the client.

### Agentic multi-turn workload (cache-validation shape)

The shape published by
[inference-benchmarking/aiperf-cache-validation](https://github.com/DO-FDE/inference-benchmarking/blob/main/aiperf-cache-validation/run_cache_validation.sh):
open-loop, 64 users at 1.6 req/s, 256 conversations of 8 turns, an 8 K shared
system prompt plus 112 K per-user context, ISL 12.7 K +/- 4 K and OSL 917 +/-
300, natural stopping (no `ignore_eos`). One arm, connector ON, 1800 s of
traffic, seed 530419.

Server: the *Server* block above, unmodified. Client:

```bash
aiperf profile \
  --model amd/GLM-5.3-MXFP4 \
  --tokenizer /data/amd_int/models/GLM-5.3-MXFP4 --tokenizer-trust-remote-code \
  --url http://127.0.0.1:8332 --endpoint-type chat --streaming \
  --use-server-token-count \
  --user-centric-rate 1.6 --num-users 64 \
  --conversation-num 256 --conversation-turn-mean 8 --conversation-turn-stddev 0 \
  --conversation-turn-delay-mean 30000 --conversation-turn-delay-stddev 25000 \
  --shared-system-prompt-length 8000 --user-context-prompt-length 112000 \
  --num-dataset-entries 256 \
  --isl 12700 --isl-stddev 4000 --osl 917 --osl-stddev 300 \
  --benchmark-duration 1800 --random-seed 530419 \
  --server-metrics --server-metrics-formats json parquet \
  --export-level raw --request-timeout-seconds 3600
```

| metric | value |
|---|---|
| input token throughput | 19,283.82 tok/s |
| output token throughput | 100.76 tok/s |
| request throughput | 0.14 req/s |
| requests completed in window | 259 |
| ISL avg | 136,253 tokens |
| OSL avg | 711.94 tokens |
| TTFT p50 / p90 | 553,937 / 926,782 ms |
| ITL p50 / p90 | 219.71 / 260.96 ms |
| request latency p50 / p90 | 717,516 / 1,082,921 ms |
| HBM prefix hit | 5.84% (2,264,000 / 38,768,750) |
| CPU tier share | 0.00% |
| illegal-memory faults | 0 |

The offered rate is above what this configuration serves at these sequence
lengths, so the run is queue-bound: latency percentiles reflect the standing
queue, and throughput is the served rate. The CPU tier stored 145,716,224
tokens (summed over the four ranks) and served none of them back -- at this
turn cadence the reused prefix is still resident in HBM when the next turn
arrives, so nothing falls into the tier's fetchable band.

### Accuracy

gsm8k, all 1319 questions, 3-shot, greedy:

```bash
lm_eval run --model local-chat-completions \
  --model_args "model=amd/GLM-5.3-MXFP4,base_url=http://127.0.0.1:8330/v1/chat/completions,num_concurrent=64,max_retries=3,max_gen_toks=16384,timeout=1800,tokenized_requests=False" \
  --tasks gsm8k --num_fewshot 3 --apply_chat_template --fewshot_as_multiturn \
  --gen_kwargs temperature=0,top_p=1 --seed 0,1234,1234,1234 \
  --limit 1319 --log_samples
```

Three points, one server boot each. The server is the *Server* block above with
`--num-gpu-blocks-override 2048 --max-model-len 32768`: the small HBM pool is
what makes the third point mean something, because a repeat of the same
question can then only come back from the CPU tier.

| point | connector | pass | gsm8k exact_match (flexible) |
|---|---|---|---|
| `m1_base` | absent | -- | 0.9583 +/- 0.0055 |
| `m2_store` | present | 1st (tier filling) | 0.9629 +/- 0.0052 |
| `m3_load` | present | 2nd (same questions) | 0.9568 +/- 0.0056 |

All three agree inside one standard error, so restoring KV bytes from the CPU
tier does not change the answers.

## Related

- [GLM-5.2 LMCache Byte Offload](GLM-5.2-LMCache-Byte-Offload.md) — the full
  treatment of this connector: mechanism, sizing, tuning, gotchas
- [MiniMax-M3 LMCache Byte Offload](MiniMax-M3-LMCache-Byte-Offload.md) — the
  same connector on M3's three-layout registration
- [LMCache KV Cache Offload](LMCache-KV-Cache-Offload.md) — generic plugin path
  (`LMCacheConnectorV1`); does **not** support GLM-5.3's registration
- [GLM-5.3-Flash](../GLM-5.3-Flash.md) — a *different* architecture
  (`glm5_next`, MLA + KDA + DSA), not covered by this recipe
