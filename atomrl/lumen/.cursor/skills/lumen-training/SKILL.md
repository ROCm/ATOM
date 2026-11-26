---
name: lumen-training
description: Use when bringing up, comparing, debugging, or reviewing Lumen training runs, especially when behavior diverges from a trusted reference, training correctness is uncertain, AITER is suspected, or someone proposes changing already-aligned parameters to get a green run.
---

# Lumen Training Guide

## Core Principle

When a Lumen training run disagrees with a trusted reference, start each new debug session by reading the repo-local bug note file, then compare against the current reference and reduce the problem to the smallest reproducible mismatch. Use the bug note file as working memory, not as proof. If no trusted reference exists yet, define one before tuning. Do not solve correctness problems by retuning parameters that already match the reference.

## Use When

- Training diverges, produces NaN or Inf, or shows an unexpected loss or grad curve
- A new Lumen training path is being brought up against a known-good baseline
- A run differs from a reference implementation, recipe, dataset, or checkpoint
- AITER is suspected in a training mismatch
- Someone proposes "just tweak LR, clip, warmup, or scaling" even though those values already match the reference

Treat any fresh return to the same debugging problem as a new debug session:

- a new chat or agent session
- a new day or work block
- returning after unrelated work
- starting a new round of debug after prior tests finished

Treat a meaningful test or experiment as a step that changes confidence in a hypothesis, such as a new repro, written diff, backend toggle, layerwise compare, kernel test, or targeted integration check. Do not log every identical rerun.

## Default Workflow

1. At the start of every new debug session, read the whole `.cursor/tmp-training-bugs.md` file relative to the `Lumen` repo root before proposing new hypotheses or next steps.
2. Stop the run once behavior is clearly wrong. Save logs, checkpoints, and the first failing signal. Do not keep training "for more evidence" when the run is already untrustworthy.
3. Before training or debugging, compare the current run against the reference across effective parameters, config, environment, and dataset. If no trusted reference exists yet, define one first. Use [reference.md](reference.md).
4. If a value is already aligned with the reference, do not change it to chase the symptom.
5. If the relevant parameters, config, and dataset are aligned and the mismatch remains, compare forward and backward layer by layer on the same inputs unless AITER evidence already isolates the failing path earlier.
6. If AITER is implicated by op or backend evidence, validate the kernel path first, then run a targeted integration check, then do end-to-end confirmation. Never reverse this order.

## Required Order

1. Freeze one trusted reference:
   - read `.cursor/tmp-training-bugs.md` relative to the `Lumen` repo root first and note any still-open suspicions
   - choose the reference branch, checkpoint, and recipe
   - if no trusted reference exists, define one before tuning
   - record the exact first failing signal
2. Compare before acting:
   - effective training settings
   - dataset and data pipeline
   - resume state and backend flags
   - write down the compared values in one diff artifact
3. Fix real diffs first:
   - if a field differs from the reference, fix that diff before deeper debugging
4. Refuse shortcut tuning:
   - if a field already matches the reference, do not change it as a "quick fix"
5. Reduce to a minimal repro:
   - same code path, same batch, same seed, same checkpoint when possible
6. Escalate debugging:
   - config and dataset diff
   - minimal repro
   - layerwise forward comparison to localize the first divergence
   - layerwise backward comparison if forward localization is not enough
   - if a backend toggle or logs already isolate an AITER-backed failing path before layerwise, go directly to AITER kernel validation
   - targeted integration
   - short end-to-end confirmation

## Hard Rules

- Stop training when the run is clearly wrong. Debug first, resume only after the cause is understood or isolated.
- Start every new debug session by reading the whole `.cursor/tmp-training-bugs.md` file relative to the `Lumen` repo root.
- After each meaningful test or experiment, append or update any new possible bug, evidence, and status in `.cursor/tmp-training-bugs.md` relative to the `Lumen` repo root, including negative results that rule a suspicion out.
- When in doubt, stop and diff against the reference instead of continuing to burn GPU.
- Compare parameters, config, data pipeline, and dataset before starting training and before debugging.
- Do not change any already-aligned hyperparameter or training knob to make a bad run look better.
- If parameters and dataset are aligned, the next step is layerwise forward and backward comparison, not hyperparameter sweeping, unless AITER evidence already isolates the failing path earlier.
- If AITER is implicated by the failing op, backend toggle, or first localized divergence, start with kernel-level unit validation against a trusted path. Do not jump directly to end-to-end validation.

## What Counts As AITER Evidence

Treat AITER as implicated only when at least one of these is true:

- the first localized divergence lands on an AITER-backed op
- an AITER enable or disable toggle changes the mismatch
- logs or backend selection show the failing path routes through AITER

## Rationalizations To Reject

| Excuse | Reality |
|--------|---------|
| "We already spent a lot of GPU time, keep it running" | Sunk cost is not evidence. Stop the bad run and debug from a stable snapshot. |
| "We do not have a perfect reference, so let's tune first" | Freeze a reference artifact and success signal before tuning anything. |
| "Just tweak LR or grad clip to get green" | If those values already match the reference, changing them hides the real bug. |
| "The configs are basically the same" | "Basically" is not enough. Compare effective values, dataset, resume state, and backend flags. |
| "Layerwise compare is too slow" | It is faster than blind sweeps once parity is established. |
| "End-to-end AITER validation is faster than kernel tests" | End-to-end results do not isolate kernel correctness. Start with the kernel path. |

## Red Flags

- Starting a fresh debug session without reading `.cursor/tmp-training-bugs.md` relative to the `Lumen` repo root
- Continuing a clearly bad run for "more signal"
- Changing aligned LR, warmup, clip, scaling, batch semantics, or precision settings
- Comparing config files but not effective runtime values
- Claiming alignment from memory instead of a written diff artifact
- Ignoring dataset version, tokenizer, sharding, sampler order, or resume state
- Running layerwise comparisons on different checkpoints or different microbatches
- Going straight to end-to-end validation when AITER is suspected

## References

- `.cursor/tmp-training-bugs.md` relative to the `Lumen` repo root - repo-local temporary file for open bug candidates, evidence, and outcomes
- [reference.md](reference.md) - pre-training diff checklist, debug entry checklist, layerwise compare checklist, and AITER validation ladder

## Pairing

When code or tests are needed, pair this skill with `lumen-coding` and `lumen-test`.
