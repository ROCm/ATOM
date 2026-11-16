# DeepSeek-V4.1-Flash with ATOM SGLang Plugin (DSpark + radix cache)

This recipe serves DeepSeek-V4.1-Flash (FP4 PTPC checkpoint) through the ATOM SGLang plugin. It uses DSpark
speculative decoding (block 5) and SGLang's radix cache, with decode, verify and draft CUDA graphs and the overlap
scheduler. The validated configuration is TP4 on MI308X with SGLang 0.5.22.

On MI308X (gfx942) the routed experts are converted to per-channel FP8 at load and run through aiter's `fused_moe`,
which needs an aiter with ROCm/aiter#7820: without it the per-token FP8 MoE takes aiter's one-stage kernel, which drops
the experts' SwiGLU clamp. Native ATOM has the same requirement. ATOM's V4.1 code also imports aiter's
`get_mla_sparse_prefill_v41_backend` (ROCm/aiter#7407, on aiter main), so #7820 has to sit on a recent aiter main: the
validated aiter is the #7820 head 7bec3c672 (aiter main 13b6e87ce plus #7820).

## Launching Server

```bash
MODEL_PATH=${MODEL_PATH:-/data/models/DeepSeek-V4.1-Flash-FP4PTPC}
PORT=${PORT:-30000}

export SGLANG_PLUGINS=atom_sglang
export SGLANG_EXTERNAL_MODEL_PACKAGE=atom.plugin.sglang.models
export SGLANG_EXTERNAL_MM_PROCESSOR_PACKAGE=atom.plugin.sglang.models
export OMP_NUM_THREADS=4 AITER_REUSE_IDENTICAL_COMM_GROUPS=1
export AITER_LOG_LEVEL=WARNING
export SGLANG_RAGGED_VERIFY_MODE=static
rm -rf /root/.cache/atom/*

python3 -m sglang.launch_server \
    --model-path "${MODEL_PATH}" --trust-remote-code \
    --tp-size 4 --host 0.0.0.0 --port "${PORT}" \
    --language-model-only --kv-cache-dtype bfloat16 --page-size 256 \
    --cuda-graph-backend-decode full --cuda-graph-backend-prefill disabled \
    --cuda-graph-max-bs-decode 64 --cuda-graph-bs-decode 1 2 4 8 16 32 48 64 \
    --speculative-algorithm DSPARK --speculative-dspark-block-size 5 \
    --speculative-num-draft-tokens 6 \
    --context-length 65536 \
    --chunked-prefill-size 8192 --max-prefill-tokens 16384 \
    --mem-fraction-static 0.85 --watchdog-timeout 1800 --dist-timeout 3600 \
    --max-running-requests 64 \
    --skip-server-warmup
```

The decode graphs (the target's verify and the draft's block) cover every running batch up to
`--max-running-requests 64`, padded to the next bucket (native ATOM's list up to 64). Against buckets that stop at 32,
this adds about 0.6 GB of graph memory per GPU (the KV pool is unchanged) and 1-2 s of capture. To go back to graphs
up to 32 only, launch with `--cuda-graph-max-bs-decode 32 --cuda-graph-bs-decode 1 2 4 8 16 32`: then any round
with more than 32 running requests runs the verify and the draft eagerly, at 1.5-1.9x the TPOT (see Measured). A
finer list (`1 2 4 8 16 24 32 40 48 56 64`) takes about 0.03 GB more graph memory per GPU and 1 s more capture, and
measured 3% / 1% less wall time at 40 / 56 requests, within run-to-run noise. The plugin refuses DSpark decode graphs
larger than `--max-running-requests`.

Radix cache is on by default. Under DSpark, a hit's replay rows also rebuild the draft context. To turn the radix
cache off for DSpark only, set `ATOM_V41_RADIX_DSPARK=0`.

The radix cache's environment variables (read by the plugin, `atom/plugin/sglang/deepseek_v41_envs.py`):

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| **ATOM_V41_RADIX** | bool | 1 (true) | SGLang plugin, DeepSeek-V4.1: serve SGLang's radix (prefix) cache with the replay below (`deepseek_v41_radix`). On whenever SGLang's radix cache is on, i.e. without `--disable-radix-cache`. `0` turns SGLang's radix cache off for the launch (declared through SGLang's model-override registry at server-args resolution, so the tokenizer manager and the schedulers agree; one warning). DSpark speculative decoding runs with it (see `ATOM_V41_RADIX_DSPARK`); any other speculative algorithm turns it off the same way (and is refused by the bridge). |
| **ATOM_V41_RADIX_DSPARK** | bool | 1 (true) | SGLang plugin, DeepSeek-V4.1 with DSpark speculative decoding and radix cache on: serve both together. A prefix hit's replay rows (`ATOM_V41_RADIX_REPLAY_ROWS`) rebuild the draft's three context windows in the request's STATE slot along with the target's, as native ATOM's DSpark does under its prefix cache; with decoder SWA bounded replay on (the default, `ATOM_SGLANG_V41_DSPARK_DECODER_REPLAY`), a fused hit's widened span replays and the draft windows take its last 128 rows, the first draft block's context. Verify and draft-block graphs are unaffected. `0` is the kill switch: a DSpark launch gets SGLang's radix cache turned off (one warning, same mechanism as `ATOM_V41_RADIX=0`); launches without speculation keep radix. |
| **ATOM_V41_RADIX_REPLAY_ROWS** | int | 128 | SGLang plugin, DeepSeek-V4.1 with radix cache on: how many rows below a prefix hit are recomputed into the request's freshly reset STATE slot (window ring, compressor rings, Engram history) before its extend. Clamped to the hit length; must be even and at least the sliding window (128), and `--chunked-prefill-size` must not be below it. Larger values shrink the bounded-window approximation at the cost of more recomputed rows. |
| **ATOM_V41_RADIX_REPLAY_MODE** | str | fused | SGLang plugin, DeepSeek-V4.1 with radix cache on: where those rows run. `fused` prepends them to the hit request's own span in the step's forward, as native ATOM does (SGLang's `ForwardBatch` is unchanged; the plugin widens ATOM's inputs and gathers SGLang's rows out of the output; the builder holds `R` more tokens). `side` runs them in one separate forward before the step's (the first implementation, ~0.2 s per step with a hit; kept for A/B). A fused step that would not fit the builder falls back to `side` and logs an error (unreachable as shipped: SGLang's admission charges each hit's `R` to the chunk and the builder holds chunk + `R`). |
| **ATOM_V41_RADIX_DEBUG** | bool | 0 (false) | SGLang plugin, DeepSeek-V4.1 with radix cache on, diagnostic: around every replay (a side forward, or a fused step's forward), checksum every shared PAGE (main rows and index planes) below each hit before and after it and log `V4.1 radix debug: ... pages_changed=N`, and all-reduce the step's replay row count over the TP group (`tp_agree=`). Costs a device sync and a pass over the hit pages per replay. |
| **ATOM_V41_RADIX_TEST_SIDE_STEPS** | int | 0 | SGLang plugin, DeepSeek-V4.1 with radix cache on, test hook: the first N steps that would fuse their replay (`ATOM_V41_RADIX_REPLAY_MODE=fused`) take the side-forward fallback instead, the path a step widened past the builder's tokens takes. Logged per forced step. Leave at 0 outside tests. |

## Prefill

ATOM records prefill token graphs for the target's layers 0-19 after SGLang's graph capture (buckets 256, 512, 1024,
2048 and 3072 tokens, about 5.5 s at startup). A prefill step pads to the smallest bucket that holds it and replays the
graph; a step past 3072 rows runs eagerly. A radix hit step pads on the rows ATOM actually runs (its new rows plus 128
replay rows per hit), so hit steps replay too. An arriving prefill also takes its page rows from a copy made when SGLang
allocated it, instead of waiting for the step in flight. Both are on by default. Kill switches:
`ATOM_SGLANG_V41_PREFILL_TOKEN_GRAPHS=0` (all prefills eager), `ATOM_SGLANG_V41_PREFILL_TOKEN_GRAPHS_RADIX=0` (steps
carrying a radix replay eager), `ATOM_SGLANG_V41_EXTEND_ROWS_AT_ALLOC=0` (read the rows back at staging).

A long prompt is prefilled in chunks of `--chunked-prefill-size`. A chunk that leaves at least 128 rows of its prompt
for later chunks runs layers 0-19 and layer 20's compression only. The decoder layers (20-39) and the DSpark draft
context write run on the chunk that ends the prompt, as in native ATOM. This saves about 23 ms of GPU time per 8k-token
chunk. If the step that continues such a prompt computes every row instead (another request in it asks input logprobs,
or SGLang cuts the next chunk below 128 rows), that prompt's decoder window starts at the chunk boundary instead of
reading rows the skipped chunk never wrote. A request that asks input logprobs of its own prompt runs the decoder on every
chunk's last 128 rows. Kill switch: `ATOM_SGLANG_V41_CHUNK_FINALS=0` (every chunk runs the decoder on its last 128 rows).

## Prefill admission (`--min-free-slots-delay`, optional)

On the two GSM8K workloads at concurrency 32 below, the default launch is within 4% of native ATOM's default scheduling
on wall time (0.96-1.04x). On long documents at concurrency 4 it is 1-5% behind native on wall time (numbers below).
For throughput first,
coalesce arrivals into fewer prefill steps: `--max-running-requests <concurrency> --min-free-slots-delay N` holds new
admissions until N running slots have freed. It only engages when the running batch is full, so set
`--max-running-requests` to the serving concurrency. Below 64, also cap the decode graphs at the same size
(`--cuda-graph-max-bs-decode <concurrency> --cuda-graph-bs-decode` with the buckets up to it, e.g. `1 2 4 8` for 8):
the plugin refuses DSpark decode graphs larger than `--max-running-requests` at launch. `N=4` gives -8% to -15% wall
for about +125-145 ms TTFT median; `N=8` gives -12% to -17% for about +320-360 ms. At low concurrency (e.g. 4 requests
on 32 slots) it never engages. On a closed-loop benchmark at fixed concurrency (random prompts, every request 2048
output tokens) it costs throughput instead: with `--max-running-requests` = concurrency and `N=4`, output tok/s was
5-53% below the default launch at concurrency 2-32, the most at 2-4, where each wave waits for the whole previous one.

## Measured

The tables below were measured on an earlier PR 3102 head (fc0acaf6b). Re-checked on 3102 at 111930487 (aiter #7820
head 7bec3c672), launch above, same GPUs and client: 3-shot GSM8K (1319 prompts, cold) wall 161.4 s at concurrency 32
and 113.4 s at 64 (native ATOM DSpark with prefix caching, a fresh server for each: 162.1 s and 123.5 s); acceptance
49.37%, accept length 3.47 (native 49.56%, 3.48) on the same prompts (z = -0.58); lm_eval GSM8K 3-shot 0.920 (native
0.927).

TP4 MI308X, SGLang 0.5.22, DSpark block 5 + radix cache
+ overlap + decode / verify / draft CUDA graphs + prefill token graphs, real DSpark acceptance (49-50% on GSM8K, accept
length 3.46-3.50 in every arm), greedy, token-id prompts, streaming client, radix cache flushed before each run.
Native ATOM DSpark (k=5) with prefix caching on, same GPUs, same day. Wall time in seconds; TTFT median excludes each
run's first wave (the first 32 requests); with `--skip-server-warmup`.

| launch | shared 8-shot c32 (85% radix hits): wall / TTFT p50 / TPOT | 3-shot c32 (cold): wall / TTFT p50 / TPOT |
|---|---|---|
| default (`--max-running-requests 64`, no delay) | 147.4-148.8 s / 236-238 ms / 34 ms | 154.2-162.0 s / 260-277 ms / 38-39 ms |
| `--max-running-requests 32 --min-free-slots-delay 4` | 134.5-135.2 s / 369-378 ms / 29 ms | 138.3-138.7 s / 403-406 ms / 31 ms |
| `--max-running-requests 32 --min-free-slots-delay 8` | 129.4-130.5 s / 561 ms / 25.5 ms | 133.5-133.8 s / 610-622 ms / 27 ms |
| native ATOM DSpark, default | 153.4 s / 277 ms / 35.7 ms | 156.5 s / 285 ms / 38 ms |
| native ATOM DSpark, `--prefill-admit-policy adaptive` | 122.6 s / 565 ms / 24 ms | 127.4 s / 580 ms / 26.5 ms |

These rows were measured with decode graphs up to 32. Rounds of up to 32 requests replay the same graphs with the
buckets up to 64 (3-shot c32 cold: 160.3 s, TTFT p50 270 ms, TPOT 38.8 ms, same day as 157.5 s with graphs up to 32).

Above 32 running requests: 3-shot GSM8K (1319 prompts, cold), real acceptance (49.3-49.8%), same client and setup as
above, TTFT median over all requests, measured with the launch above. Native ATOM DSpark with
`--max-num-seqs 64 --cudagraph-capture-sizes [1,2,4,8,16,32,48,64]`, prefix caching on, a fresh server for every
concurrency above 32 (its prefix cache cannot be flushed between runs).

| concurrency | graphs up to 32: wall / TTFT p50 / TPOT | graphs up to 64 (launch above) | native ATOM DSpark |
|---|---|---|---|
| 48 | 204.1 s / 411 ms / 75.6 ms | 135.7-137.0 s / 299-312 ms / 49.1-49.6 ms | 134.6 s / 348 ms / 49.1 ms |
| 64 | 166.4 s / 682 ms / 77.9 ms | 112.7-112.9 s / 514-530 ms / 51.5 ms | 121.7 s / 387 ms / 59.4 ms |

Random 4096-token prompts, 2048 output tokens, forced accept length 3.47 (`atom.benchmarks.benchmark_serving`, 2x
concurrency prompts): at concurrency 48, 989 tok/s with graphs up to 32, 1753 tok/s with graphs up to 64 (native 1736);
at 64, 1154 and 1649 tok/s (native 1822; TPOT 30.1 vs 32.6 ms). The forced accept length is drawn once a round for the
whole batch, so all 64 requests finish together and the next 64 arrive at once (14 s of back-to-back prefill); native
draws it per request. At real acceptance on these prompts (repetition), 1786 / 2045 tok/s at 48 / 64 with plugin accept
length 5.1 / 5.2 (native 2125 / 2158 at accept length 5.4 / 5.2); the accept lengths differ, so this is not a parity
measure.

Long documents, concurrency 4, on the launch above with decode graphs up to 32
(at concurrency 4 only the buckets up to 4 run, the same graphs as the launch above), native ATOM DSpark (k=5) with
prefix caching on the same GPUs within the same hours, each run after the same long-prompt warmup:

| workload | plugin | native |
|---|---|---|
| 12 documents of 11.5k-30k tokens x 5 questions (79% radix hits): wall | 18.5-19.1 s (4 runs) | 18.2-18.4 s (3 runs) |
| same: end-to-end time of the 12 first questions (uncached documents) | 4.90-4.93 s | 4.84-4.87 s |
| same: end-to-end time of the 48 later questions (cache hits) | 0.31-0.36 s | 0.31-0.32 s |
| 40 uncached documents of 11.5k-30k tokens, up to 512 output tokens: wall | 61.6-63.4 s (4 runs) | 62.5-63.0 s (2 runs) |

The plugin streams the first token of an uncached document about 0.4 s earlier than native, so its mean TPOT is higher
(36 vs 28.5-29 ms) for the same end-to-end time. Hit questions are short prefill steps that are bound by host CPU time
on both stacks, so their latency rises when other jobs load the host (0.31 s on a quiet host, 0.34-0.36 s under load).
The 40-document runs differ in output length (6.5k-7.5k tokens). Scaled to the same number of verify steps, the plugin
is 1.5-2.0% behind native (0.6-1.5% with `SGLANG_DSPARK_USE_GREEDY_DRAFT=1`, see below). Before the chunk change above,
it was 4.4-4.8% behind on the 40 documents and 3-7% on the shared documents.

`SGLANG_DSPARK_USE_GREEDY_DRAFT=1` makes the draft always take the argmax, as native's draft does, and runs ATOM's
sharded fused argmax in the draft graph. In a decode round of 4 long requests it cuts the draft step from 1.85 to
1.46 ms (the round is about 21 ms). It changes the acceptance rule only for sampled requests (temperature > 0);
greedy requests draft the same tokens up to numerics.

A 598-token request arriving into 31 decoding requests makes the decode round that holds its prefill 146 ms long
(native: 154 ms; a round without a prefill is 69 ms); a 4096-token one 323 ms (native: 320 ms). A 598-token request that
hits the radix cache (512 tokens cached) makes it 121 ms.

With `--skip-server-warmup`, the first requests after a server start wait longer for their first token (first-wave TTFT
1.8-2.9 s at concurrency 32, native 0.9-1.4 s). A server warmup that covers prefill shapes is expected to remove that,
but this has not been measured.
