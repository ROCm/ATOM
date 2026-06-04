---
name: run-atom-workload
description: Run any ATOM workload — accuracy eval (GSM8K via lm_eval), performance benchmark, concurrency sweep, offline simple_inference, or fault repro under rocm-debug-agent. Use when the user asks to "test accuracy", "测精度", "跑 GSM8K", "跑 benchmark", "test performance", "run sweep", "repro the fault", "测一下 MTP1 精度", "跑 simple_inference" — anything that drives an ATOM workload. Encodes the canonical flow (stop → start → workload-in-shell-bg → wait_infer_drain → stop) and the model-family env vars. Same pattern works for both server-based workloads (lm_eval / benchmark client) and offline simple_inference. Do NOT use for profiling traces (use capture-trace).
version: 1.6.0
scope: ATOM on AMD ROCm; `scripts/` orchestration scripts under repo root
last_updated: 2026-05-21
---

## Path convention

All script paths in this skill are **project-relative** (`scripts/foo.sh`),
not absolute. The skill lives inside the ATOM repo at `.claude/skills/`, so
it travels with the repo wherever it's cloned. CWD when invoking commands
should be the repo root (Claude Code's default).

If the user is somewhere else, prefix with `bash $(git rev-parse --show-toplevel)/scripts/foo.sh`
or `cd` to the repo first.

## Why this skill exists

Every blocking ATOM workload follows the same 5-step shape: **stop → start → verify-ready → workload-in-bg → wait_infer_drain → stop**. The scripts in `scripts/` are orchestration-grade — chain them via **separate Bash tool calls**, not wrappers, not `&&`.

Past failure modes this skill prevents (collected from many sessions):

1. **Writing wrapper scripts in `/app/logs_claude/`** — `start_atom_server.sh` etc. ARE the orchestration layer. Wrapping them is pure noise. The dozens of `run_*.sh` / `start_*_safe.sh` files in `/app/logs_claude/` are session debris — **do not mimic them**.
2. **Chaining all steps with `&&` into one long command** — the user has explicitly forbidden this. Each step gets its own Bash tool call so logs are separate, errors abort cleanly, and the user can interrupt at any boundary.
3. **Foregrounding `start_atom_server.sh`** — its inline ready-poll caps at **120 iterations × `sleep 1` = 120s** with no failure exit, so any model that takes >2min to cold-start (V4-Pro takes 5–10min) lets the script return success while the server is still loading. Step 3 then launches against a not-yet-ready server. Fix: background it with `&`, use step 2.5 as the real foreground ready gate.
4. **Skipping `wait_infer_drain.sh`** — without it, GPU faults take the whole timeout to surface, and hangs go undetected. `wait_infer_drain.sh` exits in ~10s on fault and ~1min on hang, with tail-log attached.
5. **Using `curl /health` for liveness** — under heavy load it can false-negative. The flow uses `/v1/models` (start script) and `pgrep` + Engine Core marker (drain script).
6. **Forgetting model-family env vars** — V4-Pro silently regresses on accuracy without `AITER_BF16_FP8_MOE_BOUND=0 ATOM_MOE_GU_ITLV=1`. Pinned in the table below.
7. **Skipping drain for offline simple_inference** — `wait_infer_drain.sh` supports offline mode (process-exit detection + fault scan). Without it you lose early fault visibility.
8. **Passing the wrong LOG_FILE to drain** — `wait_infer_drain.sh` auto-discovers the server log via `/proc/<pid>/fd/1`; the user-supplied LOG_FILE is a **secondary** signal (fault scan + mtime progress for clients with tqdm output). Pass any log you want extra coverage on, or pass nothing — drain still works.
9. **Trusting `start_atom_server.sh` exit alone** — same root cause as failure 3. Background step 2 with `&` and use **step 2.5 `wait_server_ready.sh` as the mandatory foreground ready gate** with a real (15min) timeout. If 2.5 fails, abort to step 5 — do **not** launch workload.
10. **Writing a custom monitor loop to "catch the hang at the right moment"** for rocgdb / py-spy attach — `wait_infer_drain.sh` exit=1 IS that moment: workers are still alive in livelock state, GPU queues still loaded, the next call is your attach. Self-written `for i in ...; sleep 10; grep -c "output send" ...` loops always misjudge the heuristic (drain's STUCK_POLLS check is tuned across hundreds of runs; yours isn't), waste a turn re-deriving it, and leak past the user-forbidden "no ad-hoc orchestration" rule. Use drain → on exit=1 attach rocgdb (see step 4.5) → step 5.

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

## Canonical 5-step flow

Run each step as a **separate Bash tool call**. Never chain with `&&`.

Step order: **1 stop → 2 start → 2.5 verify-ready → 3 workload `&` → 4 drain → 4.5 (optional) hang inspection → 5 stop**.

### Step 1 — clean GPU (always)

```bash
bash scripts/stop_atom_server.sh
```

Idempotent. SIGTERM → SIGKILL → force-kill GPU PIDs, waits ≤60s for VRAM=0. Always run first even if you believe no server is up — clears orphaned multiprocessing children that hold GPU memory.

### Step 2 — start workload host (blocks until ready / completion)

**Server-based workloads** (GSM8K / benchmark / sweep / fault repro): **background with shell `&`**, then use step 2.5 as the real ready gate.

```bash
<MODEL_ENV_VARS> bash scripts/start_atom_server.sh <MODEL_PATH> <TP> <PORT> <EXTRA_ARGS...> &
```

- **MUST end with `&`** so the Bash tool returns immediately and step 2.5 can do the real foreground wait
- The script forks python in background and runs a best-effort inline poll, but that poll caps at 120s and falls through to exit-0 without raising — for any model that takes >2min to load, the inline poll's outcome is meaningless
- Step 2.5 is the source of truth for ready/fail
- Log: hard-coded `/app/logs_claude/atom_server.log` (`LOG_FILE` env is NOT respected by this script). Drain auto-discovers it via `/proc/<pid>/fd/1` regardless of path

**Offline workload** (simple_inference): steps 2 and 2.5 are skipped — workload runs offline; jump to step 3.

Model-family env vars (set as `VAR=val VAR=val bash ...` prefix):

| Model | Required env vars | Required CLI args |
|---|---|---|
| DeepSeek-V4-Pro | `AITER_BF16_FP8_MOE_BOUND=0 ATOM_MOE_GU_ITLV=1 AITER_LOG_LEVEL=WARNING` | `--kv_cache_dtype fp8 --level 0` |
| DeepSeek-R1-0528 (default) | `AITER_LOG_LEVEL=WARNING` | `--kv_cache_dtype fp8` |
| Kimi-K2.5-MXFP4 | `HSA_NO_SCRATCH_RECLAIM=1 AITER_LOG_LEVEL=WARNING` | `--kv_cache_dtype fp8 --trust-remote-code` (tp=4) |

MTP add-on (any supporting model): append `--method mtp --num-speculative-tokens N` to EXTRA_ARGS. V4-Pro: keep `--level 0`.

### Step 2.5 — verify server ready (MANDATORY for server-based workloads)

```bash
bash scripts/wait_server_ready.sh <PORT> <MAX_MIN> <POLL_SEC> /app/logs_claude/atom_server.log
```

Typical: `bash scripts/wait_server_ready.sh 30000 15 5 /app/logs_claude/atom_server.log`.

- Polls `/v1/models` and grep-watches the server log for startup errors (`cluster_dims`, `InductorError`, `SHUTDOWN`, `proc died`, `AssertionError`)
- Exit 0 → server ready, proceed to step 3
- Exit non-zero → **abort to step 5** (`stop_atom_server.sh`). Do NOT launch step 3
- `MAX_MIN`: V4-Pro cold start ~5–10min; use 15 to be safe. Smaller models: 5–8
- Set Bash tool timeout to ≥ `MAX_MIN × 60 × 1000` ms (e.g. `900000` for 15min) so the tool doesn't kill the poll prematurely

Why mandatory: step 2 was backgrounded with `&`, so its exit code is meaningless to us — we deliberately ignored it. Step 2.5 is the **one and only** clean gate — its exit code, blocking behavior, and tail-on-fail are all visible to the operator. Without it, step 3 may launch against a dead or not-yet-ready server (lm_eval just sits in its own retry loop and hides the failure).

Skip step 2.5 for: offline simple_inference (no server), debug-agent fault repro (the fault IS the goal).

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

`LOG_FILE` is **optional**. The drain script discovers the server log itself from the running `atom.entrypoints` process. Pass an additional client/workload log only if you want:
- Extra fault grep coverage (drain scans both)
- Mtime-based progress detection for client tools that write tqdm to a file (benchmark, simple_inference)

PORT is unused in offline mode but kept positional for API symmetry.

How drain decides (auto-detects server vs offline by `SERVER_PATTERN` pgrep):
- **Server mode**: client gone (lm_eval / curl / benchmark process exited) + no new "Engine Core: output send" since last poll → exit 0
- **Offline mode**: simple_inference process exited cleanly (no fault grep) → exit 0
- **Either mode**: fault grep on auto-discovered server log + optional caller LOG_FILE → exit 2 in ≤10s
- **Server only**: no progress (engine output count flat + caller LOG_FILE mtime flat) AND client still running for STUCK_POLLS × POLL_SEC ≈ 1min → exit 1 (hang)
- **Either mode**: MAX_MIN elapsed without resolution → exit 4

If exit ≠ 0: read the printed tail. If exit=1 (hang) and you want to inspect the stuck GPU state, do step 4.5 BEFORE step 5 (step 5 kills the workers and destroys the evidence). Otherwise run step 5 regardless.

Typical wait windows:
- GSM8K (1319 samples): MAX_MIN=30 plenty for V4-Pro
- Single benchmark: MAX_MIN=30
- Sweep (8 conc points): MAX_MIN=60
- Simple_inference (default ~10 prompts): MAX_MIN=15 plenty
- Fault repro: MAX_MIN=10 (fault should land within first request)

### Step 4.5 — hang inspection (optional, only when drain exit=1)

Trigger: step 4 returned exit=1 (HANG detected) AND you want to know which kernel / Python frame is stuck.

`wait_infer_drain.sh exit=1` means the engine stopped emitting "output send" for STUCK_POLLS×POLL_SEC seconds while the benchmark client is still alive. At that instant:

- Server process is alive (`pgrep atom.entrypoints` returns)
- Worker spawn_main processes are alive (`ps -ef | grep spawn_main`)
- GPU queues still hold the stuck dispatches
- HSA debugger hook is FREE (you launched with plain `start_atom_server.sh`, not `run_debug_agent.sh`)

This is the unique window to attach **rocgdb** or **py-spy** before step 5 destroys the evidence. After step 5 all workers are SIGKILL'd and `info dispatches` returns "No dispatches" — too late.

Use the **[[debug-agent-locate-kernel]]** skill for the rocgdb workflow:
- Step R2 in that skill: pick a worker PID (`ps -ef | grep spawn_main`, PPID = dispatcher, NOT the openai_server)
- Step R3: `rocgdb -p $WORKER_PID -x cmdfile -batch` with `detach` before `quit`
- Step R4-R6: read `info dispatches` / `info queues` from the dump

Quick py-spy companion (for the Python-side stack on the same worker):
```bash
py-spy dump --pid $WORKER_PID 2>&1 | tee /app/logs_claude/pyspy_${WORKER_PID}.txt
```

**Hard rule**: do NOT replace step 4 with a self-written `for/while` polling loop just to "time the attach better". Drain's exit=1 IS the attach moment. See failure mode 10.

After inspection completes, proceed to step 5.

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

## Reading baselines

Per-model accuracy baselines and thresholds live in `.github/benchmark/models_accuracy.json` (CI is the source of truth). For one-off comparison, run the same `(model, fewshot)` pair locally and diff against the CI threshold in that file.

## Hard rules (do not violate)

1. **One Bash tool call per script.** No `&&` chains. User has explicitly forbidden chaining.
2. **No wrapper scripts in `/app/logs_claude/`.** Call `scripts/*` directly.
3. **Shell `&`, not `run_in_background: true`** for step 3 (drain finds workload via pgrep, no task tracking needed).
4. **Drain auto-discovers server log via `/proc/<pid>/fd/1`** — you no longer need to know or pass the canonical server log path. Pass a client log only as supplementary signal.
5. **Always step 5 (`stop_atom_server.sh`)**, even after a fault, even for offline workloads.
6. **Never use `curl /health`** to verify ready — only `/v1/models` (already inlined in start script).
7. **No `LOG_FILE=` env on `start_atom_server.sh`** — log path is hard-coded.
8. **Step 2 (`start_atom_server.sh`) MUST end with shell `&`.** Foregrounding it is a trap: its inline ready-poll caps at 120s and falls through to exit-0 without raising — for V4-Pro and other slow-starting models the script silently returns success while the server is still loading. See failure mode 3.
9. **Step 2.5 (`wait_server_ready.sh`) is MANDATORY for server-based workloads.** Run it as a separate foreground Bash tool call after step 2. If it exits non-zero, abort to step 5; do not launch step 3.
10. **Step 4 (`wait_infer_drain.sh`) is the ONLY hang detector.** Never replace it with a custom `for/sleep/grep` monitor loop, even if your goal is to attach rocgdb / py-spy at hang time — drain's exit=1 IS the attach moment (workers still alive, GPU queues still loaded). Step 4.5 covers the inspection workflow. See failure mode 10.

## Reference: each script in one line

| Script | What it does | Step | Blocks? |
|---|---|---|---|
| `stop_atom_server.sh` | Kill all atom + multiproc children, wait for VRAM=0 | 1, 5 | Yes ≤60s |
| `start_atom_server.sh MODEL TP PORT [ARGS...]` | Clean GPU, fork python in bg, best-effort 120s ready poll (must wrap with `&` — step 2.5 is the real gate) | 2 (server) | Self-blocks ≤120s but unreliable as gate |
| `start_simple_inference.sh MODEL TP [ARGS...]` | Offline inference (no server, runs prompts) — wrap with `&` for drain | 3 (offline) | Blocks unless `&` |
| `run_gsm8k_eval.sh MODEL PORT FEWSHOT` | lm_eval local-completions GSM8K — wrap with `&` for drain | 3 (server) | Blocks unless `&` |
| `run_benchmark.sh MODEL PORT ISL OSL CONC [PMULT] [PROF]` | Single perf point — wrap with `&` for drain | 3 (server) | Blocks unless `&` |
| `run_benchmark_sweep.sh MODEL PORT ISL OSL "CONCs"` | Loop run_benchmark — wrap with `&` for drain | 3 (server) | Blocks unless `&` |
| `wait_infer_drain.sh PORT MAX_MIN POLL [LOG] [STUCK]` | Monitor workload for drain / hang / fault (auto-discovers server log) | 4 | Yes, until exit code |
| `wait_server_ready.sh PORT MAX_MIN POLL LOG` | Mandatory ready-gate after start; polls `/v1/models` + greps log for startup errors | 2.5 (server) | Yes, until ready or fail |
| `run_debug_agent.sh [--simple] MODEL TP [PORT] [ARGS...]` | Server (or simple_inference) under rocm-debug-agent — fault repro | 2 (replaces start) | Yes, until ready or fault |

## Worked example: V4-Pro MTP3 GSM8K accuracy

```
# Step 1
bash scripts/stop_atom_server.sh

# Step 2 — note trailing `&` (REQUIRED — see Hard rule 8)
AITER_BF16_FP8_MOE_BOUND=0 ATOM_MOE_GU_ITLV=1 AITER_LOG_LEVEL=WARNING \
  bash scripts/start_atom_server.sh /data/DeepSeek-V4-Pro 8 30000 \
  --kv_cache_dtype fp8 --method mtp --num-speculative-tokens 3 --level 0 &

# Step 2.5 — MANDATORY foreground ready gate. Bash tool timeout ≥ 900000ms.
# Abort to step 5 on non-zero exit.
bash scripts/wait_server_ready.sh 30000 15 5 /app/logs_claude/atom_server.log

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
