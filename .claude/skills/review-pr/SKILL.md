---
name: review-pr
description: Review an ATOM PR to the maintainer's bar and draft the review for posting. Combines the team rule catalog (silent bypass, coverage gaps, hardcoded arch/dtype, uninitialized state, cross-repo aiter/plugin sync, backbone risk, perf evidence, AI-generated code checks) with the maintainer's own lens distilled from their ROCm/ATOM review comments and corrections (where code belongs, reuse what exists, root cause vs patch, one mechanism per concept, no lost perf, semantics over coincidental numbers, defaults over env flags, tests that can fail). Runs the affected unit tests against the merge base, verifies every finding by reading or a probe, and writes the English prose review (Must change / Worth addressing / Smaller + inline comments) that is posted only for the findings the user picks. Use for "review PR #N", "/review-pr N", "/code-review <ATOM PR URL>", "看一下这个 PR", "re-review", even when the user only pastes an ATOM PR link. Do NOT use for aiter / mori PRs, for the local uncommitted diff before a commit (use /code-review on the working tree), or for writing the fix.
argument-hint: <PR number or URL>
---

# ATOM PR Review

ATOM consumes aiter kernels (attention, MoE, GEMM, norm, quant), plugs into
vLLM / SGLang / RTP-LLM, and runs on MI300X / MI355X / gfx1250. A change here can
break inference for every model on the affected path.

Two sources drive this review:
- **The maintainer's lens** (section 3): what the maintainer actually asks on ATOM
  PRs, each item with the PR where they raised it (`#N`). Open that PR when a case
  is borderline. Where the lens and the catalog differ, the lens wins.
- **The rule catalog** (`references/rule-catalog.md`): the team checklist — PR
  type → rules, backbone file risk, rules A–G, perf evidence, housekeeping and
  AI-generated-code checks. Section 2 tells you which parts to open.

## 1. Set up

1. The first tool call reads what the user gave (PR URL, issue, CI run). Then
   fetch the PR, its diff and every comment:
   ```bash
   bash /app/logs_claude/get_gh_token.sh gh pr view N --repo ROCm/ATOM --json title,body,number,labels,files,author,headRefOid,reviews,comments
   bash /app/logs_claude/get_gh_token.sh gh api --paginate repos/ROCm/ATOM/pulls/N/comments
   ```
   Read the body, every review, every inline thread and the author's replies. A
   point raised and answered is not re-raised; a point raised and ignored is, and
   says so. Note what Copilot and other reviewers already raised.
2. Check the PR out in a worktree **on the PR branch** — never `main`, never a
   branch switch in `/app/ATOM`:
   `git -C /app/ATOM fetch origin pull/N/head:prN --force`, then
   `git -C /app/ATOM worktree add /tmp/prN_review prN`. For the next PR, switch
   that clean worktree to the next branch instead of adding another. Read and grep
   the worktree, not `git show`.
3. Artefacts (fetched JSON, probes, the draft) go to `/app/logs_claude/prN_review/`.
4. Record the head SHA and the merge base. If the author pushes while you review,
   re-diff before posting and say which head each point is about.
5. Start the tests now, in the background, and read the code while they run:
   `python /app/ATOM/.claude/skills/review-pr/scripts/affected_tests.py /tmp/prN_review <merge-base> /app/logs_claude/prN_review/tests`.
   It picks the diff's test files plus every existing test that imports a changed
   module — a test the new code breaks is usually not in the diff — runs them the
   way CI does (no aiter / triton / FlyDSL), then runs the files CI skips (GPU and
   aiter-backed tests) in the real environment on HIP 4–7, and reruns every
   head failure at the merge base. Read `tests/summary.txt`: a `REGRESSION` or
   `NEW-TEST` line is a finding to trace; a failure that also fails at the
   merge base predates the PR and needs at most one sentence, not an
   investigation. Do not rerun the suite yourself.
6. Classify every red CI check before blaming the PR: read the failed step; a
   readthedocs build failing on a code-only or `.claude/`-only change is
   unrelated infra (ATOM#1549); a job main also fails in the same window is baseline; logs
   that return `410 Gone` are stale — ask for a rebase and a fresh run.

Work in parallel throughout: independent reads and greps go out in the same turn.

## 2. Understand before judging

Answer these first, in your notes:
1. What changed computationally — which kernel path, data flow, formula?
2. Which archs (gfx942 / gfx950 / gfx1250) and model families, which TP / DP / EP?
3. Which aiter ops are added, changed or removed, and with which new kwargs?
4. If it claims performance, by what mechanism (fewer launches, fused round trips,
   better tiling)?
5. Does the description explain why, or only what?

Then classify the PR with `references/rule-catalog.md` section 1 and open the
rules it points to; if a Tier 1 or Tier 2 file changed, also its section 2.

For every new field, flag, env var, class, RPC, guard, buffer, file or code path
the diff adds:

- **Should it exist?** Which existing component could not express the need — and
  is that limit real, or an abstraction this same PR introduced? The usual answer
  is an existing mechanism generalized by one parameter. Name it, and name what
  goes away ("this flag, both call sites and the test stub disappear").
- **Is it a second mechanism for one concept?** Prefill and decode, draft and
  target, V4 and V4.1, DP and TP, native and plugin must not grow parallel
  answers to the same question (#2116 "this file is for common path!!", #2686
  align the mono drafter with other models). Two mechanisms is the defect even
  when each half is correct.
- **Is it a second implementation of logic that already exists?** An estimator
  that predicts what Phase 2 admits instead of asking it (#2238), a Triton
  re-implementation of an algorithm `dcp_ops.py` already runs on aiter kernels
  (#2229). List where the two disagree.
- **Does it patch an edge of a feature the PR itself added?** Then the question is
  about that feature, not a licence for another guard.

Never invent a constraint to defend or attack a design ("would be overwritten",
"needs a kernel change", "h2d never takes that long") — check it first.

## 3. The maintainer's lens

Work through every subsection. Each item is something the maintainer has
written on an ATOM PR or corrected in development.

### 3.1 Where the code belongs (most frequent)
- **Kernel- and arch-level logic belongs in aiter**, not ATOM: capability checks,
  arch gates (`can_swizzle = get_arch() in ...` #1793), backend choice between
  implementations ("keep one api, switch impl in aiter" #2510; multi-backend
  GEMM #1903), quant helpers, missing kernel features (#1665 "fix in aiter
  instead of here", #2710). ATOM calls one API.
- **Do not duplicate aiter or ATOM utilities** (#57, #119, #1401, #1202 "keep the
  only truth"). Grep aiter and `atom/` for the helper before accepting a new one.
- **No model-specific code in shared files**: `model_runner.py`, `scheduler.py`,
  `backends.py`, `base_attention.py`, entrypoints (#1712 "no model specific stuff
  should be here", #2431 "mla_xxx shouldn't be in model_runner", #127 no
  `if/else` per backend in the common builder — set it in the subclass). It goes
  in the model, its backend, or a subclass (#860).
- **No hacks inside model files**; a concern every model shares is solved once at
  the top level (loader, base class, config) instead of in each model (#56, #126,
  #262, #401 "we have to do this for every new model?", #1119 one
  `quant_config` path).
- **Plugin code lives under `atom/plugin/`**, and the plugin should reuse ATOM's
  native implementation rather than fork it (#265, #399, #582, #1006, #1910).
- **Shared concepts get one shared layer**: an AR layer instead of `if/else`
  around every all-reduce (#1681), a state-pool setting every model can use
  (#1874), a common MoE helper in `moe.py`, not in one backend (#1202).
- **Each concern in its own module/owner**: a new feature's logic in its own file
  (#2007 "move all changes into dynamic_chunking.py"), runner-owned facts in the
  runner, not the attention backend (#2365), a mapping table that is not a model
  implementation does not sit in `atom/models/` (#2517).

### 3.2 Reuse what is already there
- The repo usually has the field, util, op or precedent: "we already have it
  here" (#1334), `get_tp_group().all_reduce` already dispatches to quick-AR
  (#45), `impl.use_flash_layout` already exists (#2106), the new util function
  (#1423), `self.fc_norm` / `self.act_fn` (#1605, #2089), `scheduled_bs` from the
  forward context (#2090, #2557). Point at it with a file:line.
- One place for one fact: env names (#2347 "use unified env"), versions
  (#1587 one place for `VLLM_COMMIT`), argument names across backends (#265
  "please unify the args").
- Fold duplicated branches and duplicated guards into one (#1027, #1600 "merge
  these 2 path into 1").

### 3.3 Root cause, not patch
- Fix it, not a workaround for the special case (#741, #2756 "should be fixed by
  moe side, instead of here").
- A fix that turns off the signal that misfired (disable stall release on a path,
  skip the check, raise the threshold) instead of fixing what the signal watches
  → ask for the root-cause fix and name it (#2727).
- A guard with no measurement and no end-to-end mechanism behind it ("only when
  dp == 1", "skip for MTP to be safe") is itself an untested change and not
  neutral — it turns a feature off where it is most used. Ask for the
  measurement or the mechanism; with neither, it goes.
- Fallbacks: if a new path is unsupported, fall back to exactly what ran before
  the PR (#1600), and fall back on runtime error where the file already does
  (#1003, #2199) — but not by `try/except` around code that is in the repo
  (#838 "why we need this weird try... except").
- A fast path guarded by a conjunction over the batch (`if all(row is full)`)
  fires less as the batch grows, opposite to its payoff — check how often it
  fails per step at bs=256.
- When one defect keeps recurring across parallel implementations (connectors,
  formats, backends), the fix is a declared axis (an abstract method, a
  declare-and-check table over the registry), not the next point patch (#2053).
- A defect in a shared path is fixed there, not refused or guarded per model
  (#2641: Triton TP MoE ignoring `custom_routing_function`).
- Before accepting a workaround for an aiter bug, check whether current aiter
  already fixed it (#2477: the ring the PR works around was removed upstream).
  Name the aiter PR / commit either way.

### 3.4 Performance: no lost perf, cost is cost
"I don't like loss any perf" (#265). Every one of these is a finding, without
weighing whether it is "worth it":
- D2H syncs: `.cpu()`, `.item()`, `.tolist()`, Python reads of GPU tensors in the
  step (#286, #1142 "be careful with every .item()").
- Per-step allocation or construction: `torch.zeros` / `fill(-1)` / `np.array` /
  `np.full` per forward (#835, #2495 "why we need zeros?"), where an owner-held
  preallocated buffer filled in place would do.
- Copies: `.contiguous()`, `.clone()`, `.to()`, `copy_` that are not needed — aiter
  quant and GEMM accept strided input (#265, #873, #1965 "aiter quant should
  support stride already", #2496 "this copy needed?", #1600 "useless .to()").
  A fusion that forces a `contiguous()` may gain nothing (#873).
- Doubled memory: a new weight variant kept alongside the original (#1469), a
  per-rank copy of a large store (#2582: 10 GiB/GPU out of the KV budget), a
  buffer reserved outside the KV budget (#2464).
- Host traffic: pass a scalar, not a per-request tensor, scheduler → runner → GPU
  (#227); compute `time.time()` only when it is used (#169).
- Many small torch ops on a hot path → one Triton kernel (#2549, #2775). Small
  kernels sit at a ~5–7 µs dispatch floor: count launches, not bytes.
- Per-shape JIT or per-shape caches that grow in serving (#2710: a planner
  compile plus a never-freed buffer set per new prefill length).
- Work on a side stream does not shorten E2E unless it was on the critical path
  (#2807) — ask what it overlaps with.
- A perf claim must come from measuring that exact change: kernel timing through
  `aiter.test_common.run_perftest`, e2e through the repo benchmark scripts, both
  arms on the same tree. A copied bench keeps its own `cuda.Event` timer. A
  number in the PR that you cannot trace to output is `[unverified]`; a faster
  shape you suggest is measured or labelled unmeasured.

### 3.5 Hardcoded dtype, arch and constants
- `torch.float8_e4m3fn` → `aiter.dtypes.fp8` so gfx942 (fnuz) works (#960, #1576);
  448 is not the fp8 max on gfx942 (#1044); "please also consider gfx942" (#1044).
- "not always bf16" (#1423), "force bf16?" (#127), "hard code to fp8_e8m0?" (#1458):
  derive dtypes from the tensor or config (#287 "get datatype from config").
- `use_gfx1250_xxx` style switches → a capability, not an arch name (#2466, #1547).
- Hardcoded model numbers (`576`, `7168`, `16`) → derive from config (#265
  "num_heads * v_head_dim is not 7168 for deepseek").
- Hardcoded worker counts, Python versions, docker names, paths in scripts (#1465
  `max_workers=16`, #1124 `py3.10`).

### 3.6 Semantics, not coincidental numbers
- Runner block size (KV management, prefix cache) and attention block size
  (kernel view) are different things and need not be equal (#1305); assert the
  relation instead of assuming equality (#566). Tokens vs rows vs requests,
  per-rank vs DP-uniform, `scheduled_bs` vs `running_bs` vs padded rows: different
  units even when equal today. An argument that holds only because a ratio is 1
  is not an argument — ask for the conversion and a test where it is not 1.
- MTP invariants: `token_id == batch_id` is false under MTP (#1391); a target step
  always emits ≥ 1 token and ≤ `mtp_k + 1` (#228); `mtp_k + 1` vs `mtp_k` (#147);
  `2 * drafts + 1` needs a derivation (#2728).
- Names that mean the same thing must be one name (`max_q_len` vs `max_seqlen_q`
  #127); names that differ must say how (`spec_enabled` vs `use_spec` #2365;
  `--enable-disagg` collides with multi-node disagg #860; `kv_scale` tensor vs a
  Python-float `one_scale` #346). A new name must not reuse a head word that
  already means something else in `atom/` (`block` = KV page / Triton tile,
  `region` = memory layout, `seed` = RNG): `grep -rniow <word> atom/`. A name whose
  behaviour changed in the diff is renamed in the same PR.
- Readability of imports and aliases (#119).

### 3.7 ATOM correctness traps (when the diff touches the area)
- **Collectives**: a decision that changes which collective a rank issues (graph
  replay vs eager, padding, bucket, gather path) must read DP-uniform state, never
  `is_dummy_run`, a local `scheduled_bs` or local token counts. Keep collectives on
  the group's ProcessGroup so they reach the custom collective path (#2722).
- **Cross-process state**: `ModelRunner` runs in worker processes; a `config`
  attribute set there never reaches `Scheduler` / `EngineCore`. Same for a
  scheduler-class attribute written by the worker-class connector (#2053).
- **CUDA graphs**: new branching, `.item()`, allocation in a captured region; a
  mask keyed to `cu_seqlens_q[-1]` sees the capture-time shape; FlyDSL launches
  must pass `stream=fx.Stream(torch.cuda.current_stream())`.
- **Draft steps**: every metadata field the target sets must be re-sliced for the
  draft (`slot_mapping`, `index_slot_mapping`, `token_to_seq_idxs` #2008); a dummy
  run must not replay the last real step's slots (#2151).
- **Padding rows**: pad slots with `-1` (kernels skip negative slots), never `0`
  (block 0 is allocatable, #2008); zero-filled pad tails can fault.
- **Implicit fences**: middle prefill chunks return before `postprocess`; whatever
  relied on it as a sync point has no fence there.
- **Masking**: `0 * NaN = NaN` — out-of-range lanes are selected away, indices
  clamped.
- **Determinism**: results may differ from the reference in reduction order but
  must be batch-deterministic and identical across ranks; arrival-order merges
  and atomics into sums are findings.
- **Addressing**: int64 for index products at long context; `buffer_load` reaches
  base + 4 GiB only.
- **Lifetimes**: claims, pins and leases that can be taken but not always retired;
  sticky failure flags never cleared; executors never shut down; a static config
  mismatch detected per request instead of at handshake (#2666).
- **Runtime vs build arch**: `get_gfx_runtime()` for dispatch, `get_gfx()` only
  for codegen (#2199).
- **Plugin signatures**: a new positional argument on a shared op, model
  `forward` or registered custom op breaks every vLLM / SGLang / RTP-LLM plugin
  copy bound to it (#2149, #2494, #2607). Grep `atom/plugin/` for each changed
  signature; `tests/plugin` is not run by CI, so CI will not catch it.
- **Threads**: non-daemon threads never joined, executors never shut down
  (#2787, #2666).
- **Model ports**: compare against the reference implementation (vLLM, SGLang,
  the HF modelling code) point by point — scales applied once, RoPE convention,
  norm eps and gain centre, window W vs W-1, routing scale — and list what
  matches (#2641).

### 3.8 Defaults, env vars, config
- A feature that works should be on by default, not behind an env var (#2808,
  #682); a feature default-off needs its limitation stated (#2506 "any limitation
  here? why default off"). An env var added only "to be safe" is a finding.
- Every `ATOM_*` read goes through `atom/utils/envs.py` and has a row in
  `docs/environment_variables.md` (#2605, #2678, #1601); a changed default changes
  what the doc says happens by default.
- An option that over-designs a simple field (#2188).
- CI and recipe configs agree with each other and with CI (#872 `num_concurrent`
  64, #2033 why `--gpu-memory-utilization 0.75`, #1733 drop extra args from an
  accuracy entry). Accuracy entries that need `--enforce-eager` pass it (#2051).
- Docker: no blanket `requirements` that override pinned packages (#269), no
  unexplained torch downgrades (#583), every workaround says when it can go
  (#1585).

### 3.9 Evidence the PR must carry
- Perf PRs: e2e numbers (#76, #1049, #2657), on more than one model when the path
  is shared (#1465 dsr1 / dsv4 / glm5), and a trace view when the claim is about
  overlap (#915).
- Accuracy: GSM8K for anything on the numeric path (#1893), MTP acceptance rate
  for anything touching spec decode (#560, #1655). An accuracy drop → split the PR
  so the drop is attributable (#17), or do not merge (#75).
- Check the experiment exercised the new path: a benchmark on a model that never
  takes it (#2045), an accuracy run whose knob switched the feature off (#2007),
  numbers only at c128 when the regression is at c ≤ dp_size (#2727), bf16 KV
  when the defect needs fp8 KV (#2636), a fixed synthetic acceptance length that
  hides an acceptance drop (#2535), a CI cell published under a feature's name
  that cannot arm the feature (#2787).
- The description must describe the diff (#2267, #2348: a promised cache that does
  not exist). Doubt an implausible claim and ask for the source (#1465, #1300).
- Run-to-run noise: V4.1 GSM8K-100 swings ~9 pp, full GSM8K ~2 pp. A sub-noise
  "drop" or "gain" is not evidence either way.

### 3.10 Tests that can fail
- A test that asserts the defect as expected behaviour will fail the correct fix
  (#2727) — name the test and line.
- A test that greps / AST-scans the code it tests; a golden test that copies
  strings out of `.github/benchmark/*.json`; a reference that is a twin of the
  kernel; a self-comparison — none can catch a regression.
- CI's unit runner has CPU torch and no aiter / triton / FlyDSL. A module-scope
  import that reaches them aborts the whole CI run at collection (#2051); a file
  behind `importorskip("aiter")` never runs in CI and rots, and a fixture import
  from such a file skips the importer too (#2451, #2644). Logic that does not need
  aiter belongs in a file that runs. The test script's `ci_sim.log` shows both.
- A new property test must go red with the fix reverted — ask, or check it.
- A test named "every format / every backend" must enumerate the registry.
- Throwaway tests, benches and scripts do not belong in the PR (#1766, #2069).
- Adding CI coverage that might fail is not a risk to raise.

### 3.11 Cleanliness
- Dead code: never read (#2678), never used (#146, #1681), "no user?" (#2236),
  a file with no real change (#2535), debug logs / prints (#860), `print` instead of
  `logger` (#239).
- Over-built code: "too much `return RpcResult(...)`" (#2297), "too much `if`"
  (#320), "over design?" (#2188), ugly special-casing (#613), a 2000-line new file
  that holds a custom op, a model and a backend (#2051).
- Simplify to the one-line form when there is one (#2028).
- Comments: no narration-only comments ("KNOWN ISSUE: …" → an issue); a comment
  longer than the code it explains means the structure is wrong; every number
  and factual claim in a comment or doc matches the code or the PR's own
  measurement (#2666); comments that explain shapes where shapes are not obvious
  (#752); duplicated paragraphs; docs tables that render (#2535).
- No function-local imports (#2678); `ruff check` and `black --check` clean on
  every changed file — CI's reviewdog uses `diff_context`, so pre-existing
  violations in a touched hunk fail the PR.
- Process: title carries the area tag (#1046 `[atom-vllm]`), one feature per PR
  (#17), out-of-scope work taken out (#2582 "Please take out of this PR").

## 4. Pre-existing defects

Two columns for every finding: **is it a real defect** and **did this PR
introduce it**. The second decides only how it is filed. "Not a regression" is
always followed by its disposition in the same sentence ("…so not a regression,
but worth stating that a failed post is fatal to the consumer"). Calling
something pre-existing needs the same evidence as any other claim — check the
merge base. A failure that "already fails on main" is said once, with the run.

## 5. Verify every finding

- Each finding is either **read** (traced end to end: callers outside the diff,
  every sibling on the same axis, the aiter / vLLM / upstream source at the pinned
  version) or **probed** (a script under `/app/logs_claude/prN_review/`, run on
  the PR head and the merge base). The review says which. The diff shows changed
  lines, not the whole story: synchronization, the other half of a scatter, or
  the missing `else` often lives in a header or elsewhere in the function, so a
  "no sync" or "missing branch" finding needs the full symbol family read.
- Before calling anything a blocker, name the concrete input that triggers it
  (rule-catalog "blocker gate"); without one it is worth checking, not a blocker.
- A probe that shows no failure needs a positive control that breaks the
  precondition and shows the failure.
- Quantify impact with the PR's parameters (ticks, µs, bytes per GPU, requests),
  not adjectives.
- Probes are for must-change candidates whose impact the code alone cannot settle.
  A side observation, a nit or something the PR got right does not need one.
- Unit tests may use the GPUs (HIP 4–7 only). Servers, GSM8K and benchmarks run
  only when the user asks; say which of these were and were not run.
- Keep a "checked and cleared" list for candidates you refuted; mention the
  load-bearing ones in one sentence so nobody re-chases them.
- Unverifiable → drop it, or ask it as a question. Never state an inferred root
  cause as fact.
- Re-read each finding against the final worktree before writing: wrong line
  numbers, a guard in a helper you missed, a reply that already settled it.
- Group findings by root: several defects that are one change applied to some
  but not all siblings, or one wrong owner, are written as one root with its
  instances.
- Last pass (rule-catalog section 7): is there a correctness risk, resource hazard
  or edge case nothing above caught?

## 6. Write the review

English, plain declarative prose, one engineer to another. First review:

```
Thanks, <what is right about the change, specifically>. I reviewed <sha>
[against <merge-base>] by <reading the code / running the unit tests / a probe>;
<what was not run, e.g. "nothing end to end">. <the overall call, e.g. "Two
things I'd fix before merge, then a few smaller ones." / "Nothing I'd call a
correctness blocker.">

## Must change
**1. <the claim in one sentence>** (`file.py:LINE`). <mechanism, traced>.
<impact with numbers>. <the concrete change — preferably the root-cause fix,
naming the existing mechanism it reuses>.

## Worth addressing
**N. …**

Smaller: <one line each>
```

Re-review (author pushed fixes):

```
Thanks for the round of fixes. I re-read at <sha>. <Confirm each fixed item in one
line — checked against the code, not the commit message.> <Correct your own
earlier argument if it was wrong, plainly.> <Only new points or ones still open,
"not in the thread yet"; do not re-argue settled ones.>
```

When the author asked the maintainer questions, answer them first under
`## Decisions`, one decision each (do it here / follow-up is fine / take it out).
When asking to align with an external implementation (vLLM, SGLang, DeepInfer),
link files pinned to a commit and mark which points to align.

Rules:
- Omit empty sections; if nothing blocks, say so.
- Do not repeat what Copilot or another reviewer already raised; say you left it
  out (#2112).
- A claim about a runtime (HIP queues, allocator, ZMQ) is backed by its source at
  the pinned version, cited by file:line (#2604).
- Points on diff lines also go inline, one comment per point; points outside the
  diff stay in the body.
- Number findings continuously; refer back by number.
- No emoji, no severity icons, no rule codes, no AI attribution or footer.
- Credit what is right; be exact about what is wrong.

## 7. Hand off and post

1. Reply to the user **in Chinese**: the PR in one sentence, the numbered findings
   (one line each: claim, verified how, section), the worktree path and the draft
   path. Ask which numbers to post.
2. Build `/app/logs_claude/prN_review/review.json` from the picked findings only,
   renumbered:
   `{"commit_id": <head sha>, "event": "COMMENT", "body": ..., "comments": [{"path", "line", "side": "RIGHT", "body"}]}`.
3. Post only after the user picks:
   `bash /app/logs_claude/get_gh_token.sh gh api repos/ROCm/ATOM/pulls/N/reviews --input /app/logs_claude/prN_review/review.json`.
   `event` stays `COMMENT` unless the user says approve / request changes.
4. Report on the current PR only; never remind the user about earlier reviews that
   were not posted.

## Keeping this skill current

When the maintainer raises a point this skill missed, or rejects a finding it
produced: a point about their judgement goes into the matching subsection of
section 3 with the PR number; a mechanical rule (a pattern any reviewer would
check) goes into `references/rule-catalog.md` with the PR as evidence. Commit as
`review-pr: add <rule> from ATOM#<N> — <one line>`. Refresh the maintainer's
comments from GitHub with `/app/logs_claude/atom_review_skill/fetch_valarlip_reviews.sh`.
