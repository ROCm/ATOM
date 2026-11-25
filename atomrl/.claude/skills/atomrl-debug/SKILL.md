---
name: atomrl-debug
description: >-
  Debugging upstream library bugs in AtomRL. Use when a bug originates in
  Lumen, ATOM, or AITER code rather than AtomRL itself. Guides where to
  apply fixes in the current checkout or the official upstream repository.
---

# AtomRL Upstream Debug Guide

## Core Principle

AtomRL depends on three upstream libraries. When a bug is found during RL
training or inference, first identify which library owns the buggy code, then
fix it in the correct location:

| Bug origin | Fix location | How to propagate |
|------------|-------------|------------------|
| **Lumen** (training ops, FP8 quantize, models) | `Lumen/` | Edit and commit in the current ATOM checkout |
| **ATOM** (inference engine, rollout) | repository root (`../` from `atomrl/`) | Edit and commit in the current ATOM checkout |
| **AITER** (GPU kernels: attention, GEMM, MoE, norm) | official ROCm/aiter checkout | Edit in the AITER repo, commit, push to `main` |
| **MORI** (communication, collectives, expert dispatch) | official ROCm/mori checkout | Edit in the MORI repo, commit, push to `main` |
| **AtomRL itself** (controller, workers, algorithms) | `atomrl/` | Normal commit in the AtomRL repo |

**Never** patch upstream bugs directly in `atomrl/` with workarounds. Always fix at the source.

## Identifying Bug Origin

Use this decision tree:

1. **Stack trace contains `atomrl.lumen.quantize`, `atomrl.lumen.models`, `atomrl.lumen.ops`** -> Lumen bug -> fix in `Lumen/`
2. **Stack trace contains `atom.`, `atom.model_engine`, `atom.model_ops`** -> ATOM bug -> fix in the repository root/current checkout
3. **Stack trace contains `aiter.`, `aiter_fused_`, or ASM/CK kernel names** -> AITER bug -> fix in an official ROCm/aiter checkout
4. **Stack trace contains `mori.`, `mori.ccl`, `mori.ep`** -> MORI bug -> fix in an official ROCm/mori checkout
5. **Stack trace only contains `atomrl/`** -> AtomRL bug -> fix in `atomrl/`

For ambiguous cases (e.g., wrong config passed from AtomRL to Lumen), fix the caller in `atomrl/`.

## Workflow

### Step 1: Reproduce and log

```bash
# Document in .claude/tmp-rl-bugs.md
echo "## [$(date)] Bug: <title>" >> .claude/tmp-rl-bugs.md
echo "- Symptom: ..." >> .claude/tmp-rl-bugs.md
echo "- Stack trace: ..." >> .claude/tmp-rl-bugs.md
echo "- Origin: Lumen | ATOM | AITER | MORI | AtomRL" >> .claude/tmp-rl-bugs.md
```

### Step 2: Fix in the correct location

**For Lumen bugs:**
```bash
cd Lumen
# make the fix
git add -A && git commit -m "fix: <description>"
git push origin <branch>
```

**For ATOM bugs:**
```bash
cd ..  # from atomrl/ to the current ATOM checkout
# make the fix
git add -A && git commit -m "fix: <description>"
git push origin <branch>
```

**For AITER bugs:**
```bash
# AITER lives at: https://github.com/ROCm/aiter.git (main branch)
cd /path/to/ROCm/aiter
# make the fix
git add -A && git commit -m "fix: <description>"
git push origin main
```

**For MORI bugs:**
```bash
# MORI lives at: https://github.com/ROCm/mori.git (main branch)
cd /path/to/ROCm/mori
# make the fix
git add -A && git commit -m "fix: <description>"
git push origin main
```

### Step 3: Verify

After fixing, re-run the failing test or training step to confirm the fix.
Update `.claude/tmp-rl-bugs.md` with the resolution.

## Hard Rules

- Never monkey-patch upstream code from `atomrl/` (e.g., `atomrl.lumen.quantize.some_fn = patched_fn`)
- Never copy upstream files into `atomrl/` to work around a bug
- Commit Lumen and ATOM fixes in the current checkout; commit AITER and MORI fixes in their official upstream repositories
- If a fix requires coordinated changes across multiple repos (e.g., Lumen API change + AtomRL caller update), commit the upstream change first, then the AtomRL adaptation
- Document every upstream bug and fix in `.claude/tmp-rl-bugs.md`

## Common Upstream Bug Patterns

| Pattern | Likely origin |
|---------|--------------|
| FP8 quantization produces NaN/Inf | Lumen (`lumen/quantize/`) |
| KV-cache scale mismatch after weight sync | ATOM (`atom/model_engine/`) |
| Triton kernel launch failure or wrong result | AITER |
| NCCL/RCCL hang or collective mismatch | MORI |
| ROCm HIP API error (`hipMem*`, `HSA_STATUS_*`) | AITER or ROCm driver (file upstream issue) |
| Model output diverges between train and inference | Check both Lumen and ATOM; likely weight format mismatch |

## Pairing

Pair with `atomrl-coding` for code changes in `atomrl/`.
Pair with `atomrl-training` for RL training stability issues.
