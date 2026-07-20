---
name: debug-agent-locate-kernel
description: Identify which GPU kernel is faulting/hanging in ATOM via rocm-debug-agent (for faults/asserts) or rocgdb (for silent livelocks). debug-agent dumps wave registers + faulting PC + (with --save-code-objects) disassembled code object on memory faults / ASSERT_TRAP. rocgdb attaches to a live process and lists in-flight `info dispatches` + HSA `info queues` — works when the kernel isn't faulting but just stuck (e.g. atomic-counter deadlock). Use when: server crashes with "Memory access fault by GPU node-N", server hangs with GPU at 100% but no token output, kernel asserting `s_trap`, or `HIP_LAUNCH_BLOCKING=1` makes a hang vanish. Do NOT use for: numerical bugs (use dump-bisect-debug), compile errors, OOM.
version: 1.2.0
scope: ATOM on AMD ROCm (debug-agent at /opt/rocm/lib/librocm-debug-agent.so.2; rocgdb at /opt/rocm/bin/rocgdb)
last_updated: 2026-05-20
---

## Tool selection: debug-agent vs rocgdb

| Symptom | Use first |
|---------|-----------|
| `Memory access fault by GPU node-N` / `MEMORY_VIOLATION` / `ASSERT_TRAP` in log | **debug-agent** — only it dumps wave regs + faulting PC + code-object disassembly |
| Silent livelock (GPU 100%, no fault, no log output) | **rocgdb** — debug-agent never fires (no trap event); rocgdb's `info dispatches` lists in-flight kernels directly |
| `HIP_LAUNCH_BLOCKING=1` makes hang disappear (async race) | **rocgdb first** to name the stuck kernel(s); debug-agent only if you need wave-level detail later |
| Need disassembly / per-lane register values | **debug-agent** (rocgdb doesn't go to that depth on AMD) |

**The two tools cannot be combined.** debug-agent loads `HSA_TOOLS_LIB=librocm-debug-agent.so.2` which occupies the HSA debugger hook — rocgdb attached to the same process reports "No queues / No dispatches are currently active" because the agent has the slot. To use rocgdb, launch the server with **plain** `start_atom_server.sh` (no `run_debug_agent.sh` wrapper).

## When to use

Symptoms that point to this skill:

- `Memory access fault by GPU node-N (Agent handle: 0x...) on address 0x...` in `atom_server.log`
- Server alive (`curl /v1/models` returns) but `rocm-smi --showuse` shows 100% GPU and no `Engine Core: output send` for >30s — silent kernel hang / livelock
- Workers stuck at `torch.cuda.synchronize()` per `py-spy dump --pid <rank-pid>` — prior kernel never completes
- `HIP_LAUNCH_BLOCKING=1` makes the bug disappear → you have an async race; agent will tell you which kernel
- Reproduces only at certain batch shapes (e.g. MTP-3 + long prefill)

Do NOT use this skill for: precision bugs (use [[dump-bisect-debug]]), build/compile errors (use `/build-fix`), OOM.

## Required tools (verify before starting)

```bash
ls /opt/rocm/lib/librocm-debug-agent.so.2     # debug-agent (fault path)
ls /opt/rocm/bin/rocgdb                       # rocgdb (livelock path)
ls /opt/rocm/llvm/bin/llvm-objdump            # for disassembling code objects
which py-spy && py-spy --version              # Python stacks of stuck workers
```

If any missing: install `rocm-debug-agent`, `rocgdb`, `llvm`, `pip install py-spy`. Stop here if not available.

## Critical pre-flight

1. **`ulimit -c 0`** — disables gpucore dumps. ROCm fault dumps gpucore files of 30-50 GB each per rank; on 8-GPU TP this fills disk in seconds. The launcher script sets this for you.
2. **`--enforce-eager` / `--level 0` — optional fallbacks, not required.** Try the default launch first; the debug agent runs fine under hipgraph in most cases. Only reach for these flags when symptoms point at graph mode:
   - **`--enforce-eager`** disables CUDAGraph capture. Try this when the agent reports faults that don't reproduce in eager mode, or when capture/replay itself crashes under the agent's no-caching-allocator behavior.
   - **`--level 0`** disables Inductor. Try this on AMD when you hit the `cluster_dims` autotune bug or other Inductor-side crashes during warmup.
   - They are independent — apply only the one(s) the symptom points at. The launcher script does NOT inject either; pass via `EXTRA_ARGS` when you want them.
3. **Clean GPU state** — kill any prior `spawn_main`/`openai_server` processes. Stale `KFD process` entries (`rocm-smi --showpids` showing UNKNOWN PIDs holding VRAM) cause the next launch to OOM at NCCL barrier. `scripts/start_atom_server.sh` does the standard cleanup.
4. **Model-specific env** — pass on the command line or export before calling. Examples:
   - V4-Pro requires `ATOM_USE_TRITON_MOE=1`
   - Kimi-K2.5-MXFP4 requires `--trust-remote-code` + `HSA_NO_SCRATCH_RECLAIM=1`

## Launcher scripts (in repo)

| Script | Purpose |
|--------|---------|
| `scripts/start_atom_server.sh [MODEL] [TP] [PORT] [EXTRA_ARGS...]` | Standard launcher: clears GPU, clears compile cache, backgrounds server, redirects to `atom_server.log`. |
| `scripts/stop_atom_server.sh` | SIGTERM atom.entrypoints, force-kill spawn workers, wait for VRAM release. |
| `scripts/run_debug_agent.sh [MODEL] [TP] [PORT] [EXTRA_ARGS...]` | Wraps `start_atom_server.sh` with `HSA_TOOLS_LIB=librocm-debug-agent.so.2 + --save-code-objects`. Server output goes to `atom_server.log`; code objects land in `/app/logs_claude/debug_run/`. |
| `scripts/run_debug_agent.sh --simple [MODEL] [TP] [EXTRA_ARGS...]` | Same wrapper but invokes `start_simple_inference.sh` (offline, no port). Default log: `/app/logs_claude/simple_inference_debug_agent.log` (override via `LOG_FILE=`). Use for offline batch repros (e.g. V4 MTP-3 prefill hang). |
| `scripts/wait_server_ready.sh [PORT] [MAX_MIN] [POLL] [LOG_FILE]` | Poll `/v1/models` until ready or startup error detected. Allow MAX_MIN ≥ 5 under the agent (3-5× slower than normal). |

## Workflow

### Step 1: Reproduce under the agent

```bash
bash scripts/stop_atom_server.sh                         # ensure clean
ATOM_USE_TRITON_MOE=1 \
  bash scripts/run_debug_agent.sh \
  /data/DeepSeek-V4-Pro 8 8000 \
  --method mtp --num-speculative-tokens 3 &
# If launch fails / faults look graph-mode-specific, retry with
# `--enforce-eager` (and `--level 0` on AMD for the Inductor cluster_dims bug)
# appended to EXTRA_ARGS.

bash scripts/wait_server_ready.sh 8000 5 30              # 2-4 min under agent
cd /app/logs_claude && python <repro_script>.py          # smallest hang trigger
```

Server load is **3-5× slower** under the debug agent. Expect ready at 2-4 min, repro at 30-90s after first big batch.

### Step 2: Find the fault wave dump

```bash
grep -E "stopped, reason|Memory access fault|MEMORY_VIOLATION|Disassembly" \
  /app/logs_claude/atom_server.log | head -20
```

Each fault produces a block like:

```
wave_27876: pc=0x7f20f5e534c4 (kernel_code_entry=0x7f20f5e52900 <FQN OF KERNEL>) (stopped, reason: <REASON>)

scalar registers: ...
vector registers: ...   ← v0..v? show per-lane values; v6 often holds index values being processed
trap registers: ...
general registers: pc=...

Disassembly for function <FQN>:
    code object: memory://<pid>#offset=<hex>&size=<bytes>
    loaded at: [<base>-<top>]
 => <pc>: <faulting instruction>
```

The `<FQN>` is the demangled kernel name. **That's the suspect kernel.** Common cases:

| Kernel name fragment | What it actually is |
|----------------------|---------------------|
| `at::native::index_copy_kernel_impl<OpaqueType<N>>` | `Tensor.index_copy_(dim, idx, src)` for dtype with N-byte size (4=int32/float32, 8=int64/float64) |
| `at::native::scatter_kernel` | `Tensor.scatter_(dim, idx, src)` |
| `at::native::index_kernel_impl` | Advanced indexing READ `tensor[idx]` |
| `_swa_write_kernel` / `_update_compressor_states_kernel` | ATOM Triton kernel — name in `state_writes.py` |

### Step 3: Read the trap reason

| reason | what it means |
|--------|---------------|
| `ASSERT_TRAP` | Kernel hit `s_trap 2` — almost always a `CUDA_KERNEL_ASSERT(...)` failed device-side. For PyTorch `index_copy_`/`scatter_` this is the bound check `0 <= idx < self.size(dim)`. Recompile PyTorch with `TORCH_USE_HIP_DSA=1` for the assert text — usually unavailable, infer from kernel name. |
| `MEMORY_VIOLATION` | Real OOB load/store. The `pc` instruction is the access; back-trace the address from `s_*`/`v_*` registers. |
| `INVALID_OPCODE` | Corrupted code object — usually an allocator stomp on the kernel binary (very rare). |

### Step 4: Disassemble the code object

The trap dump points to `code object: memory://<pid>#offset=<hex>&size=<bytes>`. The agent saved it under `/app/logs_claude/debug_run/`. Find it:

```bash
ls /app/logs_claude/debug_run/ | grep "<pid>" | grep "size_<bytes>"
# Returns e.g.: 7_memory___2188702_offset_0x546c3060_size_4026672
```

Disassemble:

```bash
/opt/rocm/llvm/bin/llvm-objdump --disassemble-all \
  /app/logs_claude/debug_run/<file> > /app/logs_claude/fault.s
grep -nE "<faulting-pc-low-bits>|s_trap|s_endpgm" /app/logs_claude/fault.s | head -20
```

The PC's surrounding instructions tell you what the kernel was doing. For `s_trap 2` followed by `s_endpgm` you've confirmed an assert (PyTorch `CUDA_KERNEL_ASSERT` lowering). For random other instructions it's a true memory violation — read the address from registers (e.g. `v[0:1]` typically holds the destination address being stored).

### Step 5: Verify it's actually that kernel (PC can lie)

Wave debugger PC reports can be **off** when the wave is mid-flight or when the trap fires from a sibling wave. Especially common with Triton — a swa_write trap might be a downstream kernel's fault attributed back. Cross-check:

- Does the trap reproduce **only when this code path runs**? Disable the call (comment out in Python), retest.
- Does **`HIP_LAUNCH_BLOCKING=1`** make it disappear? Then it's an async race, not a static OOB; the PC kernel is the **victim**, not necessarily the root cause. Bisect for the racer (next step).
- Does inserting `torch.cuda.synchronize()` **right before** this kernel call eliminate the trap? Then root cause is upstream of this point on the same stream.

### Step 6: Bisect the racer (when PC is racer-victim)

1. **Comment out one suspect call at a time.** The one whose absence fixes it is the racer (or one of the racing parties).
2. **If neither alone but both together fail**: the race is between them sharing storage / launch slot. Add `torch.cuda.synchronize()` between them as a workaround, but THIS IS NOT A SHIPPABLE FIX — see Step 7.
3. **`py-spy dump --pid <rank-pid>`** on stuck ranks: shows the Python frame waiting on the GPU. If it's at your inserted `synchronize()`, the racer is upstream of that line.

### Step 7: Real fix vs workaround

Per [[atom-patterns]] / DeepSeek V4 guidance, do not ship `cuda.synchronize()` workarounds without root-causing the race — they mask one workload and surface a worse hang on a larger one. Common real fixes:

| Symptom | Real fix |
|---------|----------|
| Race involves freshly-allocated transient tensors (e.g. from `torch.where`, `arange`, `.reshape`, `.to(int64)`) | Pre-allocate them in `_alloc_v4_metadata_buffers` (ATOM) or as module-level scratch. Eliminates allocator churn entirely. |
| Multiple `index_copy_` / `scatter_` in sequence | Replace with a single Triton kernel that writes all destinations once. |
| Per-fwd kernel reads stale forward_vars from prior fwd | Switch H2D path off `prep_stream` to default stream (matches ATOM `prepare_mtp_decode` v2 pattern). |
| Cross-rank inconsistency causes one rank to OOB | Ensure all ranks see identical batch shapes before launching kernel; check `cu_seqlens_q` / `state_slot_mapping` parity. |

## rocgdb workflow (for silent livelocks — when debug-agent gives no wave dump)

debug-agent only fires on `MEMORY_VIOLATION` / `ASSERT_TRAP` etc. — for a **silent livelock** (GPU stuck at 100% with no kernel making progress, no fault, no log), it sits idle and gives you nothing. rocgdb fills that gap: attached to a live worker, it can enumerate in-flight HSA dispatches and queue head/tail pointers, naming the stuck kernel directly.

### Pre-flight (rocgdb only)

```bash
which rocgdb                                          # /opt/rocm/bin/rocgdb
rocgdb --version | head -3                            # confirm GNU gdb 16.x rocm-rel
```

The "Symbol PySlice_Type has different size" warnings on attach are benign — Python symbol size mismatch between rocgdb's bundled Python and the venv. Wave-debug commands still work.

### Step R1: Launch WITHOUT debug-agent

Use plain `start_atom_server.sh` — **NOT** `run_debug_agent.sh`. debug-agent's `HSA_TOOLS_LIB` occupies the HSA debugger hook and rocgdb will report "No agents / No dispatches / No queues are currently active" because the agent has the slot. Run only ONE of the two tools at a time on the same process.

```bash
bash scripts/stop_atom_server.sh
<MODEL_ENV> bash scripts/start_atom_server.sh <MODEL> <TP> <PORT> <EXTRA_ARGS...> &
bash scripts/wait_server_ready.sh <PORT> 10 5 /app/logs_claude/atom_server.log
<run workload that triggers the hang>
```

### Step R2: Pick the right worker PID (NOT the dispatcher)

ATOM at TP=N has 1 `openai_server` + 1 spawn dispatcher + N spawn workers. Only the **workers** hold GPU queues — attaching to the dispatcher returns "No dispatches" (it has no GPU work).

```bash
# Process tree: dispatcher has PPID = openai_server; workers have PPID = dispatcher
ps -ef | grep spawn_main | grep -v grep
# Workers' PPID equals the dispatcher PID and they sit ~99% CPU during forward;
# the dispatcher itself shows lower CPU. Pick any worker.
WORKER_PID=<one of the worker PIDs>
```

### Step R3: Dump GPU state non-interactively

```bash
cat > /tmp/rocgdb_cmds.txt <<'EOF'
set pagination off
set confirm off
set logging file /app/logs_claude/rocgdb_dump.txt
set logging overwrite on
set logging on
echo === info agents ===\n
info agents
echo \n=== info dispatches ===\n
info dispatches
echo \n=== info queues ===\n
info queues
echo \n=== main thread bt ===\n
bt 30
detach
quit
EOF
timeout 90 rocgdb -p $WORKER_PID -x /tmp/rocgdb_cmds.txt -batch
```

`detach` (not just `quit`) is required or the worker stays SIGSTOP'd after rocgdb exits — kills your repro and leaves zombies.

### Step R3.5: Anchor on the stuck kernel name + PC — BEFORE any theory

**This is the single most important step, and the easiest to skip.** `info dispatches` already prints the demangled kernel name of the in-flight dispatch, and the AMDGPU **wave backtrace prints the exact stuck PC inside that kernel**:

```bash
# AMDGPU waves show up as rocgdb "threads"; each prints its kernel + PC:
timeout 90 rocgdb -p $WORKER_PID -batch \
  -ex "set pagination off" -ex "set confirm off" -ex "info threads" \
  | grep -iE "AMDGPU Wave|in void|ncclDev|aiter::" | head
# -> e.g.  #0 0x...b3c in void aiter::allgather_vec<bf16,8>(...)  <-- kernel
#          all waves at the same PC + threadIdx 0..ngpus-1 = spinning in a barrier
```

Then map that PC to a **source location inside the kernel** (which loop/barrier), and READ that source, before forming any hypothesis:

```bash
# pick one wave thread id from `info threads`, then:
rocgdb -p $WORKER_PID -batch -ex "thread <id>" -ex "info line *\$pc" -ex "bt"
# no line info? disassemble the kernel and locate the PC offset (kernel+NNNN):
/opt/rocm/llvm/bin/llvm-objdump -d <module>.so | less   # find the spin loop (s_cbranch back to s_load/atomic)
```

**Let the kernel name + PC drive the investigation — not a narrative.** The name tells you the exact source file; the PC tells you the exact line. A collective kernel (`ncclDevKernel_*runRing`, `aiter::allgather_vec`, `reduce_scatter_*`) stuck with all sync-lane waves (threadIdx `< ngpus`) at one PC = a **cross-rank barrier spin** (`start_sync`/`end_sync` `while(flag < ...)`): some rank never wrote the expected flag. Open that kernel's `start_sync`/`end_sync` and reason about *why a peer's flag is missing* (grid size differs across ranks, flag counter desynced from unequal call counts, e.g. TBO uneven ubatch splits) — do NOT guess at unrelated fixes (a missing `end_sync` cannot be the cause when the wave is spinning in `start_sync`, which runs first).

### Step R4: Read the dump

`/app/logs_claude/rocgdb_dump.txt` contains four sections:

| Section | What to read |
|---------|--------------|
| `info agents` | One row per GPU (8 for TP=8). Confirms rocgdb sees the HSA runtime. If empty → debug-agent is still loaded, restart without it. |
| `info dispatches` | **The smoking gun.** Each in-flight kernel: dispatch ID, grid, workgroup, fence, demangled kernel name. Two-or-more dispatches active = concurrent streams. |
| `info queues` | HSA queue table with `Read` and `Write` pointers per queue. `Write > Read` = packets pending; queue is stalled if the head dispatch never completes. Type DMA queues handle memcpy, HSA queues handle kernel launches. |
| `bt` | Python main thread's C stack. Look for `hipMemcpyAsync → memcpy_and_sync → _local_scalar_dense_cuda` to confirm `.item()` is blocked waiting for GPU. |

### Step R5: Cross-reference kernel name → source

The demangled name in `info dispatches` is the AITER (or PyTorch) kernel symbol. For AITER ASM-precompiled kernels, grep the kernel name across `aiter/aiter/ops/` to find the Python wrapper, then check whatever singleton workspace / semaphore the wrapper allocates — that is the most common shared resource a cross-stream race fights over.

### Step R6: Tell-tale of the shared-workspace deadlock class

**Two or more `_clean`-suffixed dispatches on different queues, each with `Fence: B|Aa|Ra` (full memory fence)** = the classic shared-workspace race. Split-K GEMMs use a reduction phase that atomic-increments a counter in a per-process workspace; if that workspace is a singleton (e.g. `@functools.lru_cache(maxsize=1)` over device only) and two splitk kernels run concurrently on different streams, their counters interleave → neither hits its expected count → both deadlock.

Fix shape (general): make the workspace cache stream-keyed (e.g. `lru_cache` over `(device, stream_id)`) so each stream gets its own counter. Workaround shape: serialize the streams (`current_stream.wait_stream(other_stream)`) — masks the bug for one workload but resurfaces on a larger one; not shippable per [[atom-patterns]].

### rocgdb anti-patterns

- **Do Step R3.5 before any hypothesis.** Anchor on the kernel name (`info dispatches`) + wave PC, map PC → source line, read it. Don't theorize or chase gpucore/debug-agent/other files before the PC is located.
- **Don't fix before locating the PC.** Spinning in `start_sync` ⇒ `end_sync` (and everything after it) is off the deadlock path; changing it is a wasted edit→rebuild→retest.
- **Different stuck collectives across runs/ranks = one race, not "just waiting for a dead rank".** Open the named kernel anyway.
- **Don't attach to the dispatcher** (PPID = openai_server). It has no GPU queues; you'll get "No dispatches" and waste 90s on the timeout.
- **Don't combine debug-agent + rocgdb on the same process**. The debug-agent's HSA tool hook shadows rocgdb's queue/dispatch visibility — you'll see agents but no dispatches.
- **Don't run rocgdb interactively** when the worker is in HSA wait — it can take 30+ seconds to attach, and an accidental `^C` SIGSTOPs the worker permanently. Use `-x scriptfile -batch` with `detach` before `quit`.
- **Don't trust `info threads`' Python frame names** — rocgdb's Python integration doesn't speak the venv ABI. Use `py-spy dump --pid $WORKER_PID` in parallel for the Python-side stack.

## Recovery checklist (after agent run)

1. `bash scripts/stop_atom_server.sh` — agent leaves zombie KFD entries; if you skip, next launch OOMs at NCCL barrier.
2. `pkill -9 -f spawn_main` — sometimes `stop_atom_server.sh` misses workers stuck in fault state.
3. Wait 10s, then `rocm-smi --showmemuse` — all GPUs must show 0% before relaunching. If not, `rocm-smi --showpids` to find lingering UNKNOWN PIDs (killed but KFD hasn't cleaned yet — wait or escalate).
4. `rm /app/logs_claude/debug_run/memory_*` — code objects are 4 MB each, accumulate fast across runs.
5. Drop `--save-code-objects` from production launches — disk pressure (~500 MB per run).

## Anti-patterns

- **Don't assume `--enforce-eager --level 0` is mandatory.** Default launch is fine for most agent runs; reach for these flags only when symptoms point at hipgraph or Inductor (see pre-flight item 2). Adding them blindly hides graph-mode-only bugs.
- **Don't grep `atom_server.log` for "error" or "Traceback"** — agent's wave dump has neither; grep `"stopped, reason"` instead.
- **Don't trust PC literally** — see Step 5. Especially Triton kernels are notorious for cross-wave PC misattribution. Bisect-confirm.
- **Don't leave `--save-code-objects` on for routine runs** — each run dumps ~500 MB. Only enable for the bisect run.
- **Don't add `torch.cuda.synchronize()` "fixes" and ship** — they mask the race for one workload and surface a worse hang (livelock) on a larger one. Find the allocator/stream root cause.

## Sample wave dump (what to expect in atom_server.log)

Trimmed example. Key fields are the mangled function name in `kernel_code_entry=...`,
the `stopped, reason: ...` tag, the `code object: memory://<pid>#offset=<hex>&size=<bytes>`
line that points at the saved file, and the `=> <pc>: <instruction>` arrow showing
the faulting PC. Vector registers (only v0/v1/v6 shown — full dump prints v0..v15
and beyond) often reveal address / index values that pin down the operand.

```
[atom 15:31:09] Scheduled prefill batch: 19 reqs, 9573 tokens, req_ids: (1, 2, ..., 19)
... (some [aiter] type-hints chatter, then the agent's wave dump arrives) ...
--------------------------------------------------------
wave_27876: pc=0x7f20f5e534c4 (kernel_code_entry=0x7f20f5e52900 <void at::native::index_elementwise_kernel<128, 4, at::native::index_copy_kernel_impl<at::native::OpaqueType<4> >(at::TensorIterator&, long, long, long)::{lambda(int)#1}>(long, at::native::index_copy_kernel_impl<at::native::OpaqueType<4> >(at::TensorIterator&, long, long, long)::{lambda(int)#1})>) (stopped, reason: ASSERT_TRAP)

scalar registers:
            s0: ffffffff            s1: ffffffff            s2: 00000000            s3: f8000000
            ...
           s32: 0ec00000           s33: 00000002           ...

system registers:
          mode: 000003f0       trapsts: 80000000     status: 80010041

trap registers:
         ttmp4: 00006ce4         ttmp5: 00000000         ...

vector registers:
            v0: [0] 95f02814 [1] 95f02818 [2] 95f0281c [3] 95f02820 ... [58] 95f028fc [59] 00000a40 ...
            v1: [0] 00007f20 [1] 00007f20 ...                              ← v0:v1 = per-lane dst address
            v6: [0] 000080a6 [1] 000080a7 ... [58] 000080e0 [59] 000027ec ← per-lane src VALUES being stored

general registers:
            m0: 000103c0
            pc: 00007f20f5e534c4          exec: f800000000000000
           vcc: ffffffffffffffff

Disassembly for function void at::native::index_elementwise_kernel<128, 4, at::native::index_copy_kernel_impl<at::native::OpaqueType<4> >(...)>:
    code object: memory://2188702#offset=0x546c3060&size=4026672
    loaded at: [0x7f20f5e00000-0x7f20f615ff09]
 => 0x7f20f5e534c4 <+3012>:    s_endpgm
    0x7f20f5e534c8 <+3016>:    v_cndmask_b32_e32 v0, s0, v0, vcc
```

How to read this:

- **Kernel** = `at::native::index_copy_kernel_impl<OpaqueType<4>>` → PyTorch
  `Tensor.index_copy_(dim, idx, src)` for 4-byte dtype (int32 / float32).
- **Reason** = `ASSERT_TRAP` → some lane's `index_value` failed
  `0 <= idx < self.size(dim)`. Look at `v6` per-lane values to see what was
  being processed (if v6 holds the stored value here; the relevant register
  varies by kernel).
- **PC** lands on `s_endpgm` because the assert lowering is
  `s_trap 2; s_endpgm` — the actual condition test was earlier (look ~10
  instructions back in the disassembly for `s_cbranch_*` + `s_trap`).
- **Code object** at `memory://2188702#offset=0x546c3060&size=4026672` →
  saved as `7_memory___2188702_offset_0x546c3060_size_4026672` in
  `/app/logs_claude/debug_run/`. Use `llvm-objdump --disassemble-all` on it.

## Cross-references

- [[dump-bisect-debug]] — for numerical bugs (wrong output, not crashes)
- [[capture-trace]] — for performance investigation (which kernels eat time)
- [[atom-patterns]] — V4 attention buffer / stream conventions
