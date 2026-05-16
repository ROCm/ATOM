---
name: debug-agent-locate-kernel
description: Use rocm-debug-agent to identify which GPU kernel is faulting/hanging when ATOM server hits a HIP memory access fault, MEMORY_VIOLATION, or silent infinite loop. The agent dumps wave registers, faulting PC, and (with --save-code-objects) the disassembled code object so you can name the exact kernel and trace it back to the Python call site. Use when: server crashes with "Memory access fault by GPU node-N", server hangs with GPU at 100% but no token output, or you need to identify a kernel asserting `s_trap`. Do NOT use for: numerical bugs (use dump-bisect-debug), compile errors, OOM.
version: 1.0.0
scope: ATOM on AMD ROCm (debug-agent at /opt/rocm/lib/librocm-debug-agent.so.2)
last_updated: 2026-05-14
---

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
ls /opt/rocm/lib/librocm-debug-agent.so.2     # the agent itself
ls /opt/rocm/llvm/bin/llvm-objdump            # for disassembling code objects
which py-spy && py-spy --version              # for stack traces of stuck workers
```

If any missing: install `rocm-debug-agent`, `llvm`, `pip install py-spy`. Stop here if not available.

## Critical pre-flight

1. **`ulimit -c 0`** — disables gpucore dumps. ROCm fault dumps gpucore files of 30-50 GB each per rank; on 8-GPU TP this fills disk in seconds. The launcher script sets this for you.
2. **`--enforce-eager --level 0`** — graph mode is incompatible with the agent's no-caching-allocator. The two flags are NOT redundant: `--enforce-eager` disables CUDAGraph capture; `--level 0` disables Inductor (avoids the AMD `cluster_dims` autotune bug). **Always pass both** via `EXTRA_ARGS`. The launcher script does NOT inject them — that's intentional, you control the launch.
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
  --method mtp --num-speculative-tokens 3 --enforce-eager --level 0 &

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

## Recovery checklist (after agent run)

1. `bash scripts/stop_atom_server.sh` — agent leaves zombie KFD entries; if you skip, next launch OOMs at NCCL barrier.
2. `pkill -9 -f spawn_main` — sometimes `stop_atom_server.sh` misses workers stuck in fault state.
3. Wait 10s, then `rocm-smi --showmemuse` — all GPUs must show 0% before relaunching. If not, `rocm-smi --showpids` to find lingering UNKNOWN PIDs (killed but KFD hasn't cleaned yet — wait or escalate).
4. `rm /app/logs_claude/debug_run/memory_*` — code objects are 4 MB each, accumulate fast across runs.
5. Drop `--save-code-objects` from production launches — disk pressure (~500 MB per run).

## Anti-patterns

- **Don't run debug agent under `--level 3` (default CUDAGraph)** — graph mode incompatible, server crashes at warmup with cryptic errors. Always `--enforce-eager --level 0`.
- **Don't grep `atom_server.log` for "error" or "Traceback"** — agent's wave dump has neither; grep `"stopped, reason"` instead.
- **Don't trust PC literally** — see Step 5. Especially Triton kernels are notorious for cross-wave PC misattribution. Bisect-confirm.
- **Don't leave `--save-code-objects` on for routine runs** — each run dumps ~500 MB. Only enable for the bisect run.
- **Don't add `torch.cuda.synchronize()` "fixes" and ship** — they mask the race for one workload and surface a worse hang (livelock) on a larger one. Find the allocator/stream root cause.

## Real example (May 14 2026 — V4-Pro MTP-3 prefill hang)

**Symptom**: `Memory access fault by GPU node-N` on all 8 ranks within 1s of "Scheduled prefill batch: 19 reqs, 9573 tokens". `HIP_LAUNCH_BLOCKING=1` and `ATOM_DUMP_SWA_WRITE=1` (which inserts `.cpu()` sync) both eliminated it.

**Workflow**:
1. Step 1-2: ran `run_debug_agent.sh`, grep `"stopped, reason"` → got `ASSERT_TRAP` in `at::native::index_copy_kernel_impl<OpaqueType<4>>`.
2. Step 4: disassembled code object → confirmed `s_trap 2` followed by `s_endpgm` → PyTorch `CUDA_KERNEL_ASSERT` failed (bound check on index_copy).
3. Step 5: grep'd ATOM source for `index_copy_` of int32 dtype → only csa/hca calls in `deepseek_v4_attn.py:1505-1506`.
4. Step 6: bisected — neither alone trapped, both together did. Added `torch.cuda.synchronize()` between → small workload passed, GSM8K (1319 reqs) eventually livelocked at 1281, py-spy showed all ranks stuck in the synchronize call → confirmed root cause is allocator churn from transient tensors (`csa_win_pos`/`hca_win_pos`/`swa_paged_flat` built via `torch.where`/`arange`/`.reshape`/`.to(int64)`).
5. Real fix (in progress): replace 3 sequential `index_copy_` + transient tensor construction with a single Triton kernel reading only persistent forward_vars buffers.

Wave dumps and code objects from this run: `/app/logs_claude/atom_server.log` (line 419+), `/app/logs_claude/debug_run/`, `/app/logs_claude/fault.s`.

## Sample wave dump (what to expect in atom_server.log)

Trimmed real example from the May 14 2026 V4-Pro MTP-3 run. Key fields are the
mangled function name in `kernel_code_entry=...`, the `stopped, reason: ...`
tag, the `code object: memory://<pid>#offset=<hex>&size=<bytes>` line that
points at the saved file, and the `=> <pc>: <instruction>` arrow showing the
faulting PC. Vector registers (only v0/v1/v6 shown — full dump prints v0..v15
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
