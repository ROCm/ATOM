---
name: run-atom-workload
description: Run any ATOM workload — accuracy eval (GSM8K via lm_eval), performance benchmark, concurrency sweep, offline simple_inference, or fault repro under rocm-debug-agent. Use when the user asks to "test accuracy", "测精度", "跑 GSM8K", "跑 benchmark", "test performance", "run sweep", "repro the fault", "测一下 MTP1 精度", "跑 simple_inference" — anything that drives an ATOM workload. Encodes the canonical flow (stop → start → workload-in-shell-bg → wait_infer_drain → stop) and the model-family env vars. Same pattern works for both server-based workloads (lm_eval / benchmark client) and offline simple_inference. Do NOT use for profiling traces (use capture-trace).
version: 1.2.0
scope: ATOM on AMD ROCm; `scripts/` orchestration scripts under repo root
last_updated: 2026-05-18
---

## Path convention

All script paths in this skill are **project-relative** (`scripts/foo.sh`),
not absolute. The skill lives inside the ATOM repo at `.claude/skills/`, so
it travels with the repo wherever it's cloned. CWD when invoking commands
should be the repo root (Claude Code's default).

If the user is somewhere else, prefix with `bash $(git rev-parse --show-toplevel)/scripts/foo.sh`
or `cd` to the repo first.

## Why this skill exists

Every blocking ATOM workload follows the same 4-step shape: **stop → start → workload-in-bg → wait_infer_drain → stop**. The scripts in `scripts/` are orchestration-grade — chain them via **separate Bash tool calls**, not wrappers, not `&&`.

Past failure modes this skill prevents (collected from many sessions):

1. **Writing wrapper scripts in `/app/logs_claude/`** — `start_atom_server.sh` etc. ARE the orchestration layer. Wrapping them is pure noise. The dozens of `run_*.sh` / `start_*_safe.sh` files in `/app/logs_claude/` are session debris — **do not mimic them**.
2. **Chaining all steps with `&&` into one long command** — the user has explicitly forbidden this. Each step gets its own Bash tool call so logs are separate, errors abort cleanly, and the user can interrupt at any boundary.
3. **Double-backgrounding the server** — `start_atom_server.sh` already forks python in background internally and polls ready. Adding `&` after `start_atom_server.sh` is wrong; it blocks until ready or fail.
4. **Skipping `wait_infer_drain.sh`** — without it, GPU faults take the whole timeout to surface, and hangs go undetected. `wait_infer_drain.sh` exits in ~10s on fault and ~1min on hang, with tail-log attached.
5. **Using `curl /health` for liveness** — under heavy load it can false-negative. The flow uses `/v1/models` (start script) and `pgrep` + Engine Core marker (drain script).
6. **Forgetting model-family env vars** — V4-Pro silently regresses on accuracy without `AITER_BF16_FP8_MOE_BOUND=0 ATOM_MOE_GU_ITLV=1`. Pinned in the table below.
7. **Skipping drain for offline simple_inference** — `wait_infer_drain.sh` supports offline mode (process-exit detection + fault scan). Without it you lose early fault visibility.
8. **Passing the wrong LOG_FILE to drain** — historically callers had to know which log carries the "Engine Core: output send" marker. v1.2 of drain auto-discovers the server log via `/proc/<pid>/fd/1`, so the user LOG_FILE is now only a **secondary** signal (fault scan + mtime progress for clients with tqdm output). Pass any log you want extra coverage on, or pass nothing — drain still works.

## Backgrounding mechanism — shell `&`, NOT claude task

In step 3 the workload must run concurrent with step 4's drain monitor. Use **shell-level `&`** (append to the bash command), NOT the Bash tool's `run_in_background: true`:

- Shell `&`: bash starts workload, returns immediately, workload runs as orphan; drain finds it via `pgrep` — no claude task tracking dependency
- `run_in_background: true`: workload becomes a claude task accessed via TaskOutput — adds complexity, doesn't help drain since drain uses pgrep anyway

Pattern (literal):
```
# Step 3 — single Bash tool call, command ends with `&`
bash scripts/run_gsm8k_eval.sh /data/MODEL 30000 3 &

# Step 4 — single Bash tool call, blocks
bash scripts/wait_infer_drain.sh 30000 30 10
```

The Bash invocation for step 3 returns the instant `&` is processed (`bash -c 'cmd &'` exits as soon as cmd is backgrounded). Step 4 then runs as the next Bash call and blocks on drain.

## Canonical 4-step flow

Run each step as a **separate Bash tool call**. Never chain with `&&`.

### Step 1 — clean GPU (always)

```bash
bash scripts/stop_atom_server.sh
```

Idempotent. SIGTERM → SIGKILL → force-kill GPU PIDs, waits ≤60s for VRAM=0. Always run first even if you believe no server is up — clears orphaned multiprocessing children that hold GPU memory.

### Step 2 — start workload host (blocks until ready / completion)

**Server-based workloads** (GSM8K / benchmark / sweep / fault repro):

```bash
<MODEL_ENV_VARS> bash scripts/start_atom_server.sh <MODEL_PATH> <TP> <PORT> <EXTRA_ARGS...>
```

- Self-contained: forks python in background internally, polls `/v1/models` + GPU VRAM, returns when ready or 1 on fail
- **Do NOT** add `&` — the script self-backgrounds python and waits for ready
- **Do NOT** chain `wait_server_ready.sh` after — already inlined inside start script
- Log: hard-coded `/app/logs_claude/atom_server.log` (`LOG_FILE` env is NOT respected by this script). Drain auto-discovers it via `/proc/<pid>/fd/1` regardless of path

**Offline workload** (simple_inference): step 2 is fused with step 3, see below.

Model-family env vars (set as `VAR=val VAR=val bash ...` prefix):

| Model | Required env vars | Required CLI args |
|---|---|---|
| DeepSeek-V4-Pro | `AITER_BF16_FP8_MOE_BOUND=0 ATOM_MOE_GU_ITLV=1 AITER_LOG_LEVEL=WARNING` | `--kv_cache_dtype fp8 --level 0` |
| DeepSeek-R1-0528 (default) | `AITER_LOG_LEVEL=WARNING` | `--kv_cache_dtype fp8` |
| Kimi-K2.5-MXFP4 | `HSA_NO_SCRATCH_RECLAIM=1 AITER_LOG_LEVEL=WARNING` | `--kv_cache_dtype fp8 --trust-remote-code` (tp=4) |

MTP add-on (any supporting model): append `--method mtp --num-speculative-tokens N` to EXTRA_ARGS. V4-Pro: keep `--level 0`.

### Step 3 — launch workload in shell background (`&`)

The workload script must end with shell `&` so the Bash tool returns immediately and step 4 can start monitoring in parallel.

**Server-based workloads** (PORT is needed):

| Workload | Command (note trailing `&`) | Optional client log for drain |
|---|---|---|
| GSM8K accuracy | `bash scripts/run_gsm8k_eval.sh MODEL PORT NUM_FEWSHOT &` | `/app/logs_claude/gsm8k_eval.log` (lm_eval is silent during requests; drain's auto-discovered server log carries the engine markers — passing this log only helps fault grep coverage) |
| Single benchmark | `bash scripts/run_benchmark.sh MODEL PORT ISL OSL CONC [PROMPT_MULT] [PROFILE] &` | `/app/logs_claude/benchmark.log` (has tqdm progress, useful mtime signal) |
| Concurrency sweep | `bash scripts/run_benchmark_sweep.sh MODEL PORT ISL OSL "CONC1 CONC2 ..." &` | `/app/logs_claude/benchmark.log` (overwritten per step) |

**Offline simple_inference** (no PORT; step 2 is skipped since this script IS the workload host):

```bash
<MODEL_ENV_VARS> bash scripts/start_simple_inference.sh MODEL TP <EXTRA_ARGS...> &
```

Optional client log for drain: `/app/logs_claude/simple_inference.log` (drain auto-discovers via /proc anyway; this only helps fault grep redundancy).

Common workload knobs:
- GSM8K shots: 3 for fast/CI parity, 5 for thorough. Set `LIMIT=50` env for first-50-sample sanity.
- Benchmark `PROMPT_MULTIPLIER` default 10. Profiling: use 2 (CONC × 2 requests).
- MTP benchmark MUST add `--use-chat-template` via EXTRA_ARGS (tokenizer mismatch otherwise).
- Benchmark throughput metric: report **Total Token throughput (tok/s)**, NOT Output throughput. Total = input+output, which users care about.
- Never add `--mark-trace` or `ENABLE_TORCH_PROFILER=1` (handled by capture-trace skill).

### Step 4 — wait_infer_drain (blocks, with early fault/hang detection)

```bash
bash scripts/wait_infer_drain.sh PORT MAX_MIN POLL_SEC [LOG_FILE] [STUCK_POLLS]
```

Defaults: PORT=8000, MAX_MIN=30, POLL_SEC=10, LOG_FILE=empty (server log auto-discovered via `/proc/<pid>/fd/1`), STUCK_POLLS=6.

`LOG_FILE` is **optional** in v1.2+. The drain script discovers the server log itself from the running `atom.entrypoints` process. Pass an additional client/workload log only if you want:
- Extra fault grep coverage (drain scans both)
- Mtime-based progress detection for client tools that write tqdm to a file (benchmark, simple_inference)

PORT is unused in offline mode but kept positional for API symmetry.

How drain decides (auto-detects server vs offline by `SERVER_PATTERN` pgrep):
- **Server mode**: client gone (lm_eval / curl / benchmark process exited) + no new "Engine Core: output send" since last poll → exit 0
- **Offline mode**: simple_inference process exited cleanly (no fault grep) → exit 0
- **Either mode**: fault grep on auto-discovered server log + optional caller LOG_FILE → exit 2 in ≤10s
- **Server only**: no progress (engine output count flat + caller LOG_FILE mtime flat) AND client still running for STUCK_POLLS × POLL_SEC ≈ 1min → exit 1 (hang)
- **Either mode**: MAX_MIN elapsed without resolution → exit 4

If exit ≠ 0: read the printed tail, then run step 5 regardless.

Typical wait windows:
- GSM8K (1319 samples): MAX_MIN=30 plenty for V4-Pro
- Single benchmark: MAX_MIN=30
- Sweep (8 conc points): MAX_MIN=60
- Simple_inference (default ~10 prompts): MAX_MIN=15 plenty
- Fault repro: MAX_MIN=10 (fault should land within first request)

### Step 5 — teardown (always)

```bash
bash scripts/stop_atom_server.sh
```

Same script as step 1. ALWAYS run, even on fault or for offline workloads — releases GPU for next attempt and kills any lingering multiprocessing children.

## Reading results

After step 4 returns 0:

```bash
# GSM8K
grep -E "flexible-extract|strict-match" /app/logs_claude/gsm8k_eval.log | head -2
# Benchmark
grep -E "Total Token throughput|Mean TPOT|Mean TTFT" /app/logs_claude/benchmark.log
# Simple_inference
grep -E "^Generated|^Output|tokens/s" /app/logs_claude/simple_inference.log
```

GSM8K format: `|gsm8k|3|flexible-extract|3|exact_match|↑|0.XXXX|±|0.00XX|` — flexible-extract is the headline number, ±value is the noise band. Anything within 1σ of baseline = no regression.

## V4-Pro accuracy baselines (current, 3-shot, no `--limit`)

- Non-MTP: ~0.9545 ± 0.0057
- MTP-1: ~0.9492 ± 0.006
- MTP-3: see `feedback_v4_cg_mtp_status_deadlock.md` — currently hangs on high-concurrency GSM8K

## Hard rules (do not violate)

1. **One Bash tool call per script.** No `&&` chains. User has explicitly forbidden chaining.
2. **No wrapper scripts in `/app/logs_claude/`.** Call `scripts/*` directly.
3. **Shell `&`, not `run_in_background: true`** for step 3 (drain finds workload via pgrep, no task tracking needed).
4. **Drain auto-discovers server log via `/proc/<pid>/fd/1`** — you no longer need to know or pass the canonical server log path. Pass a client log only as supplementary signal.
5. **Always step 5 (`stop_atom_server.sh`)**, even after a fault, even for offline workloads.
6. **Never use `curl /health`** to verify ready — only `/v1/models` (already inlined in start script).
7. **No extra `&`, `wait_server_ready.sh`, `LOG_FILE=` env on `start_atom_server.sh`** — start script self-backgrounds, self-polls, hard-codes log path.

## Reference: each script in one line

| Script | What it does | Step | Blocks? |
|---|---|---|---|
| `stop_atom_server.sh` | Kill all atom + multiproc children, wait for VRAM=0 | 1, 5 | Yes ≤60s |
| `start_atom_server.sh MODEL TP PORT [ARGS...]` | Clean GPU, fork python in bg, poll ready | 2 (server) | Yes, until ready or fail |
| `start_simple_inference.sh MODEL TP [ARGS...]` | Offline inference (no server, runs prompts) — wrap with `&` for drain | 3 (offline) | Blocks unless `&` |
| `run_gsm8k_eval.sh MODEL PORT FEWSHOT` | lm_eval local-completions GSM8K — wrap with `&` for drain | 3 (server) | Blocks unless `&` |
| `run_benchmark.sh MODEL PORT ISL OSL CONC [PMULT] [PROF]` | Single perf point — wrap with `&` for drain | 3 (server) | Blocks unless `&` |
| `run_benchmark_sweep.sh MODEL PORT ISL OSL "CONCs"` | Loop run_benchmark — wrap with `&` for drain | 3 (server) | Blocks unless `&` |
| `wait_infer_drain.sh PORT MAX_MIN POLL [LOG] [STUCK]` | Monitor workload for drain / hang / fault (auto-discovers server log) | 4 | Yes, until exit code |
| `wait_server_ready.sh PORT MAX_MIN POLL LOG` | Standalone ready poller (rarely needed; start script self-polls) | (debug) | Yes |
| `run_debug_agent.sh [--simple] MODEL TP [PORT] [ARGS...]` | Server (or simple_inference) under rocm-debug-agent — fault repro | 2 (replaces start) | Yes, until ready or fault |

## Worked example: V4-Pro MTP1 GSM8K accuracy

```
# Step 1
bash scripts/stop_atom_server.sh

# Step 2
AITER_BF16_FP8_MOE_BOUND=0 ATOM_MOE_GU_ITLV=1 AITER_LOG_LEVEL=WARNING \
  bash scripts/start_atom_server.sh /data/DeepSeek-V4-Pro 8 30000 \
  --kv_cache_dtype fp8 --method mtp --num-speculative-tokens 1 --level 0

# Step 3 — note trailing `&`
bash scripts/run_gsm8k_eval.sh /data/DeepSeek-V4-Pro 30000 3 &

# Step 4 — drain auto-discovers server log; no LOG_FILE needed
bash scripts/wait_infer_drain.sh 30000 30 10

# Step 5
bash scripts/stop_atom_server.sh

# Read result
grep -E "flexible-extract|strict-match" /app/logs_claude/gsm8k_eval.log | head -2
```

## Worked example: V4-Pro offline simple_inference

```
# Step 1
bash scripts/stop_atom_server.sh

# Step 2+3 fused (simple_inference IS the workload host) — note trailing `&`
AITER_BF16_FP8_MOE_BOUND=0 ATOM_MOE_GU_ITLV=1 AITER_LOG_LEVEL=WARNING \
  bash scripts/start_simple_inference.sh /data/DeepSeek-V4-Pro 8 \
  --kv_cache_dtype fp8 --level 0 &

# Step 4 — drain auto-discovers via /proc; PORT unused
bash scripts/wait_infer_drain.sh 0 15 10

# Step 5
bash scripts/stop_atom_server.sh
```
