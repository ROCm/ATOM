# DeepSeek-V4.1-Flash with ATOM SGLang Plugin (DSpark + radix cache)

This recipe serves DeepSeek-V4.1-Flash (FP4 PTPC checkpoint) through the ATOM SGLang plugin. It uses DSpark
speculative decoding (block 5) and SGLang's radix cache, with decode, verify and draft CUDA graphs and the overlap
scheduler. The validated configuration is TP4 on MI308X with SGLang 0.5.22.

On MI308X (gfx942) the routed experts are converted to per-channel FP8 at load and run through aiter's `fused_moe`,
which needs an aiter with ROCm/aiter#7820: without it the per-token FP8 MoE takes aiter's one-stage kernel, which drops
the experts' SwiGLU clamp. Native ATOM has the same requirement. ATOM's V4.1 code also imports aiter's
`get_mla_sparse_prefill_v41_backend` (ROCm/aiter#7407) and, to build the DSpark draft's block-32 FP8 linears,
`gemm_a8w8_blockscale_group32_supported` (ROCm/aiter#8086); without #8086 the draft fails to build with an
ImportError. All three are merged, but aiter main does not run this model yet (checked at ca7cfa1c3, 2026-11-18):
ROCm/aiter#7361, merged before #7820, makes `fused_moe` raise `NotImplementedError: fused_moe cannot apply
swiglu_limit=10.0 for QuantType.per_Token SiLU at inter_dim=576; the runtime-limit CK stage1 needs a multiple of 128`
(TP4 gives the routed experts inter_dim 576) before it reaches #7820's gfx942 FlyDSL path, so the plugin and native
ATOM both fail at the first MoE call (CUDA graph capture or warm-up). Until aiter main raises that error only when the
FlyDSL path is not taken, use the validated aiter: the #7820 head 7bec3c672 (aiter main 13b6e87ce plus #7820) with
#8086's `gemm_a8w8_blockscale_group32_supported` added to `aiter/ops/gemm_op_a8w8.py` (on gfx942 it returns False and
those linears take the BF16 fallback). aiter main ca7cfa1c3 with only that check deferred selects the same MoE kernels
and gives the same acceptance, accuracy and speed (see Measured). ROCm/aiter#8178, merged on 2026-11-18 as a53213dc4
(the AITER PR 3102's recipe pins at 85999b8b4), moves that check after the gfx942 FlyDSL path; the plugin has not been
run on it yet.

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
export SGLANG_DSPARK_USE_GREEDY_DRAFT=1  # temperature-0 workloads; see "Greedy draft" below
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

After recording them, ATOM runs one eager prefill at the chunk budget (`min(--chunked-prefill-size, builder budget)`
rows, 8192 with the launch above, split over two sequences under DSpark), as native ATOM's start-up warm-up does. It
runs on the graph-capture STATE slots and SGLang's padding page, so no request state is touched, and adds about 0.75 s
to start-up (logged as `eager prefill warm-up of 8192 rows (4096+4096)`). Without it the first chunk-sized prefill after
a start pays the eager path's first-use costs (kernel loads, allocator growth), about 0.4 s on the first wave's TTFT.
Kill switch: `ATOM_SGLANG_V41_PREFILL_WARMUP=0`.

A long prompt is prefilled in chunks of `--chunked-prefill-size`. A chunk that leaves at least 128 rows of its prompt
for later chunks runs layers 0-19 and layer 20's compression only. The decoder layers (20-39) and the DSpark draft
context write run on the chunk that ends the prompt, as in native ATOM. This saves about 23 ms of GPU time per 8k-token
chunk. If the step that continues such a prompt computes every row instead (another request in it asks input logprobs,
or SGLang cuts the next chunk below 128 rows), that prompt's decoder window starts at the chunk boundary instead of
reading rows the skipped chunk never wrote. A request that asks input logprobs of its own prompt runs the decoder on every
chunk's last 128 rows. Kill switch: `ATOM_SGLANG_V41_CHUNK_FINALS=0` (every chunk runs the decoder on its last 128 rows).

## Scheduler main thread

Every scheduler loop iteration, TP rank 0 broadcasts the requests it pulled to the other ranks over the TP group's
gloo CPU group. SGLang's broadcast sleeps in the collective's wait (a futex), so an idle or lightly loaded scheduler
main thread sleeps and wakes a few thousand times a second. On hosts whose cores clock down on such a load
(intel_pstate/HWP with C1E, e.g. the 4xMI308X Xeon 8480C hosts), its core then sits near its minimum clock (0.8 GHz
measured) and the first steps of a request run their host Python up to ~3x slower: a radix-hit prefill step's host
time was 119-125 ms against native ATOM's 81 ms, whose model runner busy-polls and keeps its core at full clock. The
plugin therefore makes this broadcast poll the collective (`Work.is_completed()`) instead of sleeping, for
DeepSeek-V4.1 only: the same bytes and protocol as SGLang's, and each scheduler main thread uses a full core while the
server runs, as native's runners do (plus about 0.45 core per rank of gloo helper wakeups at idle). Between polls it
calls `os.sched_yield()`, which releases the GIL without idling the core, so other Python threads of the scheduler
process (e.g. HiCache's cache controller) are not held off by the wait. Kill switch:
`ATOM_SGLANG_V41_SPIN_BROADCAST=0`.

Measured on MI308X TP4, DSpark K=5 + radix, a fresh server per session (4 sessions each way): radix-hit TTFT (a
4105-token prompt with 3840 tokens cached) 150 ms -> 112 ms on average (native ATOM 114 ms); decode rounds and GSM8K
c32 wall time unchanged. Without speculation and radix: GSM8K 1319 c32 wall 189-193 s -> 174-184 s (native 173-180 s).

### Optional: pin the scheduler main threads (radix-hit-latency-sensitive deployments)

With the busy-polling broadcast, the remaining radix-hit TTFT spread comes from the kernel migrating the busy scheduler
main threads between cores: a freshly picked core can stay at base clock (2.0 GHz on the Xeon 8480C hosts) for seconds,
and a TP step waits for the slowest rank (hit TTFT ~135-140 ms instead of ~98 ms; native ATOM 105-112 ms). Where that
latency matters, pin each rank's scheduler main thread to its own physical core on the GPUs' NUMA node after the
server is up and warmed up. This is optional and deployment-only (no code change, the plugin does not pin threads): it
needs `taskset` and CAP_SYS_NICE (e.g. root) in the container, and `NODE` / `MAIN_CORES` must be adapted to the machine
(one physical core per TP rank, not shared with other heavy work). Threads created later inherit the mask of their
creator, so run it after the warm-up requests:

```bash
SERVER_PID=$!                  # PID of "python3 -m sglang.launch_server ..." started in the background
NODE=0                         # NUMA node of the GPUs: cat /sys/class/drm/card*/device/numa_node
MAIN_CORES=(48 49 50 51)       # one physical core per TP rank (TP0..TP3) on that node; adapt per machine
# ... wait for /health_generate and send the warm-up requests ...
expand() { local p; for p in ${1//,/ }; do seq "${p%-*}" "${p#*-}"; done; }
NODE_CPUS=$(expand "$(cat /sys/devices/system/node/node$NODE/cpulist)")
EXCL=$(for c in "${MAIN_CORES[@]}"; do expand "$(cat /sys/devices/system/cpu/cpu$c/topology/thread_siblings_list)"; done)
REST=$(comm -23 <(sort <<< "$NODE_CPUS") <(sort <<< "$EXCL") | sort -n | paste -sd,)
# every thread of the server and its child processes -> the node minus the main cores and their SMT siblings
for pid in $SERVER_PID $(pgrep -P "$SERVER_PID"); do taskset -a -p -c "$REST" "$pid" > /dev/null; done
# each TP rank's scheduler main thread (TID == PID of sglang::scheduler_TP<i>) alone on its core
for pid in $(pgrep -P "$SERVER_PID"); do
  i=$(tr '\0' ' ' < /proc/$pid/cmdline | sed -n 's/^sglang::scheduler_TP\([0-9]*\).*/\1/p')
  if [ -n "$i" ]; then taskset -p -c "${MAIN_CORES[$i]}" "$pid" > /dev/null; fi
done
```

Measured (TP4 MI308X, DSpark block 5 + radix, recipe launch, 2026-11-18): radix-hit TTFT (3840 cached tokens) median
98.0 ms over 6 sessions (144 hits, none above 104 ms) vs 104.3 ms unpinned (7 sessions, 77 of 168 hits above 120 ms) and
109.7 ms native ATOM; GSM8K 1319 prompts at concurrency 32 wall time 155.1 s pinned vs 153.9 s unpinned (-0.6%
throughput) and 160.2 s native.

## Decode step tail: greedy token pick and step-metadata copies

Between two decode graphs the plugin now does two things the way native ATOM does, for DeepSeek-V4.1 only:

- An all-greedy batch (every request at temperature 0, no logprobs, no sampling mask, `SGLANG_USE_AITER` unset) gets
  its tokens from aiter `topk_select(k=1, tie="low")`, native's greedy sampler, instead of SGLang's `torch.argmax` over
  the full fp32 vocabulary (on MI308X ~47 / 86 / 198 us for 1 / 2 / 4 rows against ~13 us). Same lowest-id tie rule,
  ids returned as int64; SGLang's NaN check, logit preprocessing and TP token sync run around it, and every other
  sampler call is SGLang's own. It is installed only on SGLang 0.5.22 with the expected `Sampler.forward` signature
  (otherwise one log line and SGLang's path). DSpark verify and draft pick their tokens elsewhere, so with DSpark it
  only picks each request's first token after prefill. Kill switch: `ATOM_SGLANG_V41_FAST_GREEDY=0`.
- Step metadata goes to the GPU on native's packed transport (one pinned arena, one copy and one scatter per phase
  group) instead of one copy per member: the 13 small copies of a decode step without speculation become 2. Every
  destination receives the same bytes. Kill switch: `ATOM_SGLANG_V41_PACKED_H2D=0`.

Measured on MI308X TP4 on 3102 at 483d8feae (aiter main 5385051a5 with #7361's check deferred and gfx942 tuned
configs; 3 fresh sessions per arm in alternating order, paired 95% CI, 2026-11-19), not re-measured on 85999b8b4.
Without speculation and radix (`--context-length 67584`), median ITL plugin minus native ATOM on 4096-token prompts
goes from +57 us to -1 us [-17, +15] at concurrency 1 and from +144 us to +14 us [-5, +34] at 2; at 63488 tokens
+106 / +124 us remain, from aiter's indexer top-k on gfx942, not the plugin. With the launch above (DSpark K=5 + radix,
forced accept length 3.47) the round time is unchanged within session noise (4096x1 -32 us [-485, +422], 4096x32
-219 us [-827, +388]). Gates on the launch above: 3-shot GSM8K (1319 prompts, c32) acceptance 49.69% vs 49.45% before
(native 49.43%), lm_eval 0.9212 vs 0.9265 before (McNemar p = 0.41), verify and draft graphs in every round.

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

The tables below were measured on an earlier PR 3102 head (fc0acaf6b). Re-checked on 3102 at 483d8feae with the
validated aiter above, the launch above, 4 fresh plugin servers (2 with and 2 without
`SGLANG_DSPARK_USE_GREEDY_DRAFT=1`) and 2 fresh native ATOM DSpark servers with prefix caching, same GPUs and client:
3-shot GSM8K (1319 prompts, cold) wall 154.7-157.0 s at concurrency 32 and 109.5-111.1 s at 64 (native 158.2 s and
122.2 s); acceptance 49.43-49.58% at 32 and 49.19-49.63% at 64, accept length 3.46-3.48 (native 49.57% and 49.48%)
on the same prompts (|z| <= 0.43 at 32 and <= 0.94 at 64); lm_eval GSM8K 3-shot 0.922-0.923 on every plugin server; radix-hit TTFT
(4105-token prompt, 3840 tokens cached, 12 pairs per prompt set, 2 sets per server) 97-128 ms per set, 108 ms mean
(native 105-107 ms); first-wave TTFT at 64 clients on a fresh server 1.34-1.39 s (native 1.45 s). PR 3102 has since
moved to 85999b8b4 (FP4PTPC review fixes: V4.1 now refuses an FP4PTPC layer whose shared expert did not fuse into the
converted routed experts, and a draft whose routed experts are not converted; the plugin's gfx942 default converts
them, for an FP4PTPC checkpoint only). On 85999b8b4 the plugin is checked by its unit tests only (CPU); the numbers
in this recipe have not been re-measured on GPU there yet.

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

At 64 the plugin trades TTFT for throughput because the client concurrency equals `--max-running-requests`. SGLang's
scheduler decides whether a waiting request fits before it drops the requests that finished in the step it has just
processed (`get_new_batch_prefill` runs before `update_running_batch` filters them), so with every slot taken a request
that replaces a finished one is admitted one decode step later. Arrivals then coalesce into about half as many prefill
steps as native ATOM takes (2.2 vs 4.2 a second), which lowers TPOT and wall time and raises TTFT. To schedule like
native ATOM, leave a few slots above the client concurrency, e.g. `--max-running-requests 68` for 64 clients. The decode
graphs can stay at 64 while there are at most 64 clients: a decode round never holds more than the clients' requests.
With more clients than graph buckets, rounds above 64 run eagerly, so raise `--cuda-graph-max-bs-decode` too. Same
3-shot GSM8K workload at 64, same day, plugin rows 1-3 runs (a fresh server or a server that has served before),
native a fresh server for each run. The client sends one request before the burst; the earlier client built its HTTP
opener in all 64 threads at once, which added about 2.5 s to the first wave's TTFT of every plugin run.

| launch | wall | TTFT mean / p50 / p99 | TPOT | output tok/s |
|---|---|---|---|---|
| default (`--max-running-requests 64`), 3 runs | 109.4-111.0 s | 567-590 / 498-512 / 2028-2493 ms | 51.5-51.9 ms | 1116-1117 |
| `--max-running-requests 68`, 3 runs | 122.7-123.9 s | 410-436 / 357-369 / 2028-2509 ms | 59.9-60.9 ms | 989-1003 |
| `--max-running-requests 80`, 1 run | 123.1 s | 408 / 363 / 2037 ms | 60.2 ms | 1006 |
| native ATOM DSpark, default, 2 runs | 123.6-124.1 s | 435-444 / 394-400 / 2090-2120 ms | 60.1-60.3 ms | 1000-1002 |

The p99 is set by the first wave (64 prompts at once), whose mean TTFT is 1.31-1.34 s on a server that has served
before and 1.77-1.79 s on the first run after start-up (native 1.46 s, fresh server), whatever
`--max-running-requests`; these runs predate the start-up eager prefill warm-up (see Prefill), which brings the first
run after start-up to 1.37-1.39 s. At 32 clients the setting
changes nothing (wall 158.0 s with 80, 160.1 s with 64, native 158.0 s); acceptance (49.6% with 80, native 49.3%,
z = 0.90) and lm_eval GSM8K 3-shot (0.922 with 80, 0.920 with 64, native 0.923) are unchanged.

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

### Greedy draft (`SGLANG_DSPARK_USE_GREEDY_DRAFT=1`, temperature-0 workloads)

The launch above sets `SGLANG_DSPARK_USE_GREEDY_DRAFT=1` for temperature-0 (greedy) serving: the draft always takes
the argmax, as native's draft does, and the draft graph runs ATOM's sharded fused argmax (each rank's LM-head shard,
the ranks exchanging only each row's `(max, id)`) instead of SGLang's Markov chain over all-gathered full-vocab logits.
In a decode round of 4 long requests it cuts the draft step from 1.85 to 1.46 ms (the round is about 21 ms). Greedy
requests draft the same tokens up to numerics. Measured on MI308X TP4 (DSpark K=5 + radix, temperature 0, a fresh
server per session, 4 sessions each way): on 4k-token prompts with 512 output tokens a decode round is 2.7% shorter at
c1, 1.6% at c8 and 1.0% at c32, which leaves the plugin's round 0.7% / 1.6% / 0.1% above native ATOM's instead of
3.4% / 3.3% / 1.1%. On GSM8K (1319 prompts, real acceptance, 6 fresh sessions each way, measured on 3102 at
483d8feae) the greedy draft shortens wall time by 1.05% (95% CI 0.08-2.03%) at concurrency 32, 152.7 s against 154.3 s,
and by 1.28% (0.18-2.38%) at 64, 109.5 s against 110.9 s; acceptance (49.47% with it, 49.43% without, native 49.43%)
and lm_eval GSM8K 3-shot (0.9200 and 0.9218, difference -0.19%, CI -0.93% to +0.54%) are unchanged.

Sampled requests (temperature > 0) are not covered by these numbers: with the flag, a sampled request's draft is still
the argmax and is accepted with the target's probability of that token (native's rule) instead of SGLang's rejection
sampling against a sampled draft, so its acceptance and outputs differ from SGLang's default and were not measured.
For workloads with temperature > 0, unset it to keep SGLang's sampled draft. Native ATOM also drafts the argmax by
default and verifies a sampled request by the target's own draw. Its lossless sampled draft
(`ATOM_SPEC_LOSSLESS_SAMPLING=1`: the draft samples its block from q and the verify accepts by the probability ratio,
as one block for DeepSeek-V4.1 since ATOM #2724) is opt-in and has no counterpart in the plugin; with the flag unset, a
sampled request takes SGLang's own sampled draft and verify instead.

A 598-token request arriving into 31 decoding requests makes the decode round that holds its prefill 146 ms long
(native: 154 ms; a round without a prefill is 69 ms); a 4096-token one 323 ms (native: 320 ms). A 598-token request that
hits the radix cache (512 tokens cached) makes it 121 ms.

With `--skip-server-warmup` and the eager prefill warm-up above, the first wave after a start (3-shot GSM8K, 64
clients, after 104 short chat requests) has a mean TTFT of 1.37-1.39 s: 1.78-1.84 s without the warm-up, 1.33-1.34 s
on a server that has served before, 1.46 s for native ATOM on a fresh server. Earlier first-wave numbers for this
recipe (1.8-2.9 s at 32 clients) also included 1.3-2.5 s spent in the benchmark client's own HTTP set-up.
