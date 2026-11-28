---
name: atomrl-strategy-search
description: >-
  Searches for the fastest correct system configuration of an AtomRL RL job
  (GRPO / PPO with Megatron native training and ATOM rollout) for a given
  training goal and machine count: decomposes the step time, ranks candidate
  layouts and kernels, screens them in short runs, validates the winner under
  correctness gates, and reports the gain with reproduction commands. Use when
  asked to raise RL training throughput or cut step time, choose TP / EP / PP /
  DP or the trainer / rollout GPU split, pick colocated vs 1-step-off layouts,
  or compare AtomRL against another framework's published numbers.
---

# AtomRL Strategy Search

Given a training goal and machines, find the configuration with the highest
throughput that does the same training, and hand over a report someone else can
reproduce. The worked example is GRPOPerf (`examples/GRPOPerf/`, NeMo-RL v0.8's
GRPO table on 2 x 8 MI355X); its numbers and lessons are in
[reference.md](reference.md).

## Core Principle

A faster configuration must compute the same training. An optimization is accepted
only when it is equivalent to what it replaces, and its speedup is explained.
Throughput bought by doing less work per step, or different work, is not a result.

- **Equivalent computation.** The loss, the gradients and the update must be the
  ones the baseline computes: mathematically identical (a different order of the
  same operations, a fused kernel, another placement), or, only where the task
  allows it, numerically different within a stated and measured tolerance.
  Anything that changes what is computed is refused, however fast it is.
- **Same algorithm.** KL term, advantage estimator, reward function, prompt and
  response limits, optimizer and schedule, clipping, importance-sampling
  correction: all as the reference (the task's baseline, or the framework being
  compared against).
- **Same work per step.** Every forward the reference runs (old log-probs,
  reference model) is run here too. Skipping one is allowed only where the
  reference itself skips it (for example NeMo-RL's `force_on_policy_ratio`), and
  then it is part of matching the reference, not an optimization.
- **Hardware differences are fair.** Not offloading what the reference offloads
  for lack of memory, a different parallel layout, fused kernels, another FP8
  format the hardware runs: all fine, and all stated.
- **Long-run correctness, not ten steps.** A change that only holds for a short
  benchmark (dropping the KL, truncating data, changing the batch or the
  staleness, skipping a forward the reference runs) is refused.

## Every Optimization States Two Reasons

Before a candidate is run, and again in the report, write down:

1. **Why it is equivalent.** Which computation it touches and why the result is
   unchanged: the same arithmetic in another kernel or order, another placement of
   the same tensors, a memory setting that changes no value. Where values can
   differ (reduction order, a fused kernel's accumulation, FP8), say by how much,
   how it was measured (unit test, step-1 comparison against the baseline, gate
   metrics) and why it cannot drift over a long run. A code change needs a unit
   test that holds loss and gradient to the baseline and fails on a broken version.
2. **Why it is faster.** The mechanism, tied to the decomposition: which phase it
   shortens, which resource bounds that phase (FLOPs, HBM bytes, host-device syncs,
   kernel launches, link bandwidth, idle GPUs on the critical path), and how the
   change removes or relieves it. A prediction with its basis, then the measured
   per-phase effect. A gain without a mechanism is not reported as an optimization
   until one is found; a gain that the mechanism does not account for is
   investigated first, because it often means less work is being done.

## Use When

- A throughput or step-time target for an RL job, or a comparison with another
  framework's table
- Choosing the parallel layout, the trainer / rollout split, colocated vs 1-step off
- Deciding whether FP8 rollout, fused kernels or memory settings are worth it

## Inputs And Outputs

- **Task card**: model, algorithm and its settings, samples per step, maximum
  lengths, allowed staleness, allowed precisions, machines, budget, gates. If only
  a sentence is given, draft the card and confirm it first.
- **Outputs**: a ledger (one row per run: config, steps, per-phase timings,
  throughput, gate metrics), a strategy note per case (candidates, prediction,
  measurement, decision), and the final report (format below). Add new lessons to
  [reference.md](reference.md).

## Workflow

```
- [ ] 1 Task card: lock what may not change; list allowed change levels
- [ ] 2 Baseline: same image, code and seed as the candidates; run it if missing
- [ ] 3 Decompose: phase timings -> critical path -> utilization per phase
- [ ] 4 Candidates: rank by the priorities in reference.md, predict each
- [ ] 5 Screen: 4 steps per candidate, compare steps 2-4 with the baseline's
- [ ] 6 Validate: the best per case, 10 steps, steps 3-8, every gate passes
- [ ] 7 Report, and add what was learned to reference.md
```

1. **Task card.** Everything in the goal is fixed. When unsure which level a change
   is, treat it as the higher one.
2. **Baseline.** An image upgrade alone moved GRPOPerf by 10-16%, so baseline and
   candidates share the image, the code and the seed. A full run of the same
   configuration on the same image can stand in for the baseline.
3. **Decompose.** Colocated: generation + old log-prob + reference + training +
   weight sync, in series. 1-step off: max(generation, forwards + training) +
   weight sync. Scoring and sleep count in generation, wake in weight sync. Compute
   each phase's utilization (MFU for training and forwards, HBM bandwidth for decode
   and the KV share of its bytes) and the communication volume against link peaks.
   Shortening a phase off the critical path gains nothing.
4. **Candidates.** Go down the priority list in [reference.md](reference.md). Each
   candidate carries its two reasons (above) and a prediction whose basis is stated:
   measured, extrapolated or assumed. Training time does not scale linearly with
   trainer count; measure it. A candidate whose equivalence cannot be argued is
   dropped before it is run.
5. **Screen.** Run one single-node candidate per node in parallel. Dry-run a
   configuration with many overrides and read the expanded command. Several
   changes can share one run when the phase timings attribute them.
6. **Validate.** Only candidates more than 3% faster than the baseline. Check that
   the measured per-phase change matches the mechanism (the phase it should shorten
   is the one that got shorter) and that the step does the baseline's work (same
   forwards, same tokens per step). A run that misses the gate on a single-token
   rollout outlier at the last step is repeated, not counted; budget for it
   (Qwen3-30B-A3B: 0-2 outliers per 10 steps).
7. **Report.** See below.

## Change Levels

| Level | Meaning | Allowed |
|---|---|---|
| A | Mathematically unchanged: parallel layout, trainer / rollout GPUs, token budgets, fused kernels with the same arithmetic, memory and sync settings | yes, with both reasons |
| B | Same objective, values change within a tolerance: FP8 rollout or KV cache, a kernel that accumulates differently | only where the task's precision allows it, with the tolerance measured and the gate passed; code changes need unit tests |
| C | Different computation: on-policy to async, `bypass_mode`, another batch, length or staleness, fewer forwards than the reference, a dropped or changed loss term | never as an optimization; that is a different task |

## Gates

Every counted run passes all of them:

- Samples per step unchanged; the weight sync completes every step.
- NeMo-RL's `token_mult_prob_error`: median over 10 steps < 1.1, last step < 1.1
  (`examples/GRPOPerf/metrics.py` computes it from a trainer log).
- Loss, gradient norm and `ref_kl` finite.
- Step 1 against the baseline with the same seed: accuracy within 2 points,
  entropy within 5%.
- A code change comes with a unit test that shows loss and gradient unchanged and
  that fails on the unchanged code.

## Hard Rules

- Check every node before a run (`docker ps`, GPU memory); never start on a node
  someone else is using. Never pass `--force` to `examples/run.sh`: it removes other
  users' containers of the same image.
- Code changes go in your own worktree and branch.
- Large files (models, data, compile caches, logs) on node-local disk; one ATOM
  compile cache root per precision and per ATOM code version
  (`--env ATOM_TORCH_COMPILE_CACHE_ROOT=...`), since the cache is keyed on neither.
- Long runs in the background; check every 20-30 minutes.

## Report

```markdown
# <framework> <algorithm> strategy search (<GPU>, <nodes> x <GPUs>)

## 1. Result
| Case | Configuration | Optimized | Baseline (same image and code) | Gain | Reference (if any) |
- Metric definition and window; baseline and candidates on the same image, code, seed.
- Same work per step as the reference: which forwards, which algorithm settings.

## 2. Methods
| Method | Level | Change (config key or commit) | Why it is equivalent | Why it is faster | Cases | Predicted / measured effect (phase) |

## 3. Validation runs
| Case | Run directory | Throughput | Step | Phases | Gate |

## 4. Reproduce
Environment (hardware, network, image with ID, code with commit, data, site file),
one command per configuration and its baseline, how to compute the metric and the
gate, expected numbers and tolerance, known pitfalls.

## 5. Tried and rejected
| Attempt | Result | Why |

## 6. Predicted vs measured
| Candidate | Predicted | Measured | Why they differ |
```

Before handing it over: every percentage traces to ledger rows, every method has
both reasons and a per-phase measurement that matches its mechanism, every command
expands under `--dry-run`, and unpushed changes name their branch.
