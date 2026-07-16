---
name: capture-trace
description: Capture a PyTorch profiler / kineto trace from a running ATOM server for a short benchmark window. Use when the user asks for "a trace", "profiler trace", "GPU trace", or "抓 trace" for performance investigation — what kernels ran, what's on the critical path, what's slow. Do NOT use for crashes (use debug-agent-locate-kernel) or numerical bugs (use dump-bisect-debug).
version: 1.2.0
scope: ATOM on AMD ROCm (PyTorch kineto profiler, per-rank `*.pt.trace.json.gz`)
last_updated: 2026-05-20
---

## When to use

- User asks for a trace, profiler dump, or kineto dump
- Performance analysis: "what's eating the time", "is decode fused", "did kernel X get called", "is this kernel on the critical path"
- Verifying that a code path was actually exercised at runtime (search by kernel name)

Do NOT use this skill for:

- Crashes / `Memory access fault` → [[debug-agent-locate-kernel]]
- Wrong outputs / accuracy regression → [[dump-bisect-debug]]
- Suspicion of a hang (no progress) → `scripts/wait_infer_drain.sh` first

## Critical pre-flight

1. **Stop the existing server cleanly** — the profiler argument has to be on the launch command line. `start_atom_server.sh` auto-kills the prior atom workers, so just relaunching with the new args is enough.
2. **Pick a SHORT workload** — a trace from a long run is unreadable and OOMs the profiler exporter. Default to `CONC * 1` requests (one prompt per concurrent slot). Never use the production `PROMPT_MULTIPLIER=10` default for a profiling run.
3. **`ATOM_PROFILER_MORE` belongs on the server, not the benchmark client.** The profiler runs inside the model-runner worker processes; an env on the bench client does nothing.
4. **Trace dir must be empty** for a clean per-rank layout. `start_atom_server.sh` does NOT clear it — `rm -rf $TRACE_DIR` before relaunch if you're iterating.

## Required tools

```bash
ls /app/ATOM/scripts/start_atom_server.sh         # launcher
ls /app/ATOM/scripts/run_benchmark.sh             # bench driver (passes --profile when PROFILE=1)
ls /app/ATOM/scripts/wait_server_ready.sh         # ready-poll
python3 -c "import torch.profiler"                # kineto present
```

## Parameters

Pull these out of the user's request; everything except `MODEL` has a sensible default.

| Param | Meaning | Typical |
|---|---|---|
| `MODEL` | Model path under `/data/` | `/data/DeepSeek-V4-Pro` |
| `TP` | Tensor-parallel size | `8` (4 for Kimi, 1 for gpt-oss-120b) |
| `ISL` / `OSL` | Random input / output length | `1024 / 1024` |
| `CONC` | Concurrency the bench keeps in flight | `64` or `128` |
| `PROMPT_MULTIPLIER` | Total prompts = `CONC * this` | **`1` for trace runs** (override the script default of 10) |
| `ATOM_PROFILER_MORE` | `1` = shapes + stack + memory (large traces, OOM risk); `0` = kernel-name only | **`0`** unless asked |
| `TRACE_DIR` | Where the kineto `.pt.trace.json.gz` lands | `/app/logs_claude/traces/<run-name>` |
| `EXTRA_ARGS` | Forwarded to the openai server (MTP, kv-cache, etc.) | See [[atom-patterns]] |

## Workflow

### Step 1: Launch the server with the profiler bound

```bash
TRACE_DIR=/app/logs_claude/traces/<run-name>
mkdir -p "$TRACE_DIR"

# ATOM_PROFILER_MORE on the server env — not the client.
ATOM_PROFILER_MORE=0 \
  bash /app/ATOM/scripts/start_atom_server.sh \
    "$MODEL" "$TP" 8000 \
    --torch-profiler-dir "$TRACE_DIR" \
    $EXTRA_ARGS
```

`start_atom_server.sh` blocks until either `Server is ready!` or `Server process died`. Check the tail line; if it died, no point profiling.

### Step 2: Drive a SHORT bench with `PROFILE=1`

```bash
bash /app/ATOM/scripts/run_benchmark.sh \
  "$MODEL" 8000 "$ISL" "$OSL" "$CONC" \
  1 \      # PROMPT_MULTIPLIER — keep this at 1 for traces
  1 \      # PROFILE=1 flag
  $BENCH_EXTRA_ARGS
```

Position 6 is `PROMPT_MULTIPLIER`; position 7 is `PROFILE`. The bench sends a `start` HTTP call before the run and a `stop` call after, which is what trips the kineto exporter on the server.

### Step 3: Wait for the exporter to finish (asynchronous on the server)

Kineto exports lazily on the worker side — `stop_profiler` returns immediately to the bench, but the per-rank `.json` write + gzip can take 10-60 seconds. Poll the output dir:

```bash
for i in $(seq 1 60); do
  GZ=$(find "$TRACE_DIR" -name "*.pt.trace.json.gz" | wc -l)
  JSON=$(find "$TRACE_DIR" -name "*.pt.trace.json" -not -name "*.gz" | wc -l)
  echo "[t=${i}0s] gz=$GZ json=$JSON"
  # Done = expected gz count AND no orphan .json (the .json is deleted after gzip)
  [ "$GZ" -ge "$TP" ] && [ "$JSON" -eq 0 ] && break
  sleep 10
done
```

The completion signal is **`.gz` exists AND the same-name `.json` is gone**. File size of the `.gz` alone is unreliable (per `feedback_trace_gz_truncated.md`) — the exporter writes the raw `.json`, then gzip + unlink, so an orphan `.json` means it crashed mid-export.

### Step 4: Verify the layout

```bash
find "$TRACE_DIR" -type f | xargs ls -la
```

Expected:

- `<TRACE_DIR>/rank_0/`, `rank_1/`, …, `rank_<TP-1>/` — one dir per rank
- Each dir has exactly one `*.pt.trace.json.gz`
- `ATOM_PROFILER_MORE=0`: ~50-80 MB per rank
- `ATOM_PROFILER_MORE=1`: ~200-300 MB per rank

### Step 5: Inspect

For a quick "did kernel X run" check:

```bash
zcat "$TRACE_DIR"/rank_0/*.gz | python3 -c "
import json, sys
events = json.load(sys.stdin)['traceEvents']
names = {e.get('name','') for e in events}
for kw in ['<kernel-substring>', ...]:
    hits = sorted(n for n in names if kw in n)
    print(f'{kw}: {len(hits)} matches')
    for h in hits[:5]: print(f'  {h}')
"
```

For counts and aggregate time per kernel:

```bash
zcat "$TRACE_DIR"/rank_0/*.gz | python3 -c "
import json, sys
events = json.load(sys.stdin)['traceEvents']
def stat(kw):
    m = [e for e in events if kw in e.get('name','') and 'dur' in e]
    if m: print(f'{kw}: count={len(m)} total_us={sum(e[\"dur\"] for e in m)}')
stat('aiter::topk_softplus')
stat('aiter::moe_forward')
# ...
"
```

For a UI view, drop the `.gz` (decompressed `.json`) into <https://ui.perfetto.dev> or `chrome://tracing`.

## `record_function` tag format

ATOM annotates the critical path with `torch.profiler.record_function`. The **kind** is the label prefix (groups in Perfetto, greppable); sub-attributes are `key=value` fields. Taxonomy lives in `atom/model_engine/run_labels.py`:

| Prefix | Meaning |
|---|---|
| `prefill[bs= tok= ctx=]` | real prefill (eager) |
| `decode[bs= tok= p= d= spec=]` | real decode via CUDAGraph |
| `eager_decode[bs= tok= ctx=]` | real decode forced eager |
| `dummy_decode[...]` | **DP-sync dummy** (idle rank keeps the MoE collective aligned) |
| `dummy_eager_decode[...]` | DP-sync dummy, forced eager |
| `dummy_prefill[...]` | warmup dummy prefill |
| `draft[i/k bs=]` | eagle/MTP draft mid-step |

Fields: `bs` effective batch, `tok` total tokens, `ctx` per-seq context lens (truncated if many), `p`/`d` prefill/decode seq counts, `spec` speculative steps, and **`tbo=1`** appended when the step ran Two-Batch-Overlap ubatches.

Distinguishing dummy vs real matters: e.g. the leading `dummy_decode[bs=1 tok=1]` runs are DP-sync idle steps, NOT real decode. `parse_trace.py` matches only the exact `prefill[` / `decode[` prefixes, so dummy/eager variants are auto-excluded from its stats.

> **Comparability note:** pre-taxonomy traces labeled DP-sync dummies as plain `decode[...]`, so `parse_trace.py` counted them as real decodes. New traces exclude dummies — decode/prefill counts and averages on a new trace will differ from an old trace of the same workload. Don't attribute that delta to a perf change; re-baseline with a fresh trace.

Searching by these tags is far more reliable than searching by kernel name (which varies across PyTorch/Triton/AITER versions).

## `ATOM_PROFILER_MORE` cost

`ATOM_PROFILER_MORE=1` enables `record_shapes + with_stack + profile_memory`. This multiplies trace size ~3-4x and, more importantly, the **C++ kineto aggregation at `stop_profiler` time scales with the recorded event count × TP**. On 8-rank V4-Pro, a 60-second profile with `PROFILER_MORE=1` has been observed to OOM the worker processes during export. Rules of thumb:

- Default `=0` (kernel names + durations only — enough for 95% of investigations)
- `=1` only when you specifically need shapes or Python stacks AND you've kept the window short (≤ 5 seconds of bench traffic, ≤ `CONC * 1` prompts)

## Looking up model configs

- **Server launch args + env vars**: `.github/benchmark/models.json` (CI source of truth).
- **Does this model need `--use-chat-template` on the bench?** Inspect the model's `tokenizer_config.json` — if it has a non-null `chat_template` field, pass `--use-chat-template`; otherwise the bench will tokenize the raw prompt directly.

## Anti-patterns

- **Setting `ATOM_PROFILER_MORE=1` on the bench client.** The profiler runs in the model-runner workers; client env is ignored. Set it on the server launch.
- **Using `PROMPT_MULTIPLIER=10` (the bench default) for a trace run.** That sends 10× `CONC` requests and produces a multi-GB trace that takes forever to export and can OOM the workers.
- **Trusting the `.gz` size to decide if the export finished.** Use `.gz exists AND no orphan .json`. The exporter writes `.json` first, then gzips + unlinks; a leftover `.json` means it crashed mid-export.
- **Forgetting `--use-chat-template` on MTP DeepSeek-R1 benchmarks.** Tokenizer mismatch silently degrades accuracy — the trace will look fine but the workload is wrong.
- **Adding `--mark-trace` or `ENABLE_TORCH_PROFILER=1`.** Neither is needed — `--torch-profiler-dir` on the server + `PROFILE=1` on the bench is the complete handshake. The extras either no-op or interfere.
- **Profiling V4-Pro under default `--level 3`.** V4-Pro Inductor + autotune hits a `cluster_dims` bug on AMD — pass `--level 0` until that bug is fixed.

## Cross-references

- [[debug-agent-locate-kernel]] — when the server crashes or hangs, this skill is the wrong tool
- [[dump-bisect-debug]] — when the trace shows correct kernels but outputs are wrong
- [[atom-patterns]] — V4 attention buffer/stream conventions referenced from trace kernel names
