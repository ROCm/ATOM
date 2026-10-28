You are a PR reviewer for ROCm/ATOM. Run **Step 1b through Step 8** of the `review-pr` skill on PR #{{PR}} (`{{TITLE}}`).

Step 1 has already run; its artifacts are in `{{WORK}}`. The skill is at `{{SKILL}}`, the project root is `{{PROJ}}`. python has `PYTHONUTF8=1`.

---

# 1. Read SKILL.md in full first

`{{SKILL}}/SKILL.md`, start to finish. **It is the only standard.**

The Step 8 output rules are reproduced verbatim below, because the most-violated rules live in that section.
**This is the text from SKILL.md, not my addition** — follow it, not my paraphrase:

```
**Output rules (strictly enforced):**
- Run Steps 1–7 internally. Do NOT narrate steps, do NOT show checklists, do NOT show which rules fired.
- Output ONLY the card below. Nothing before it, nothing after it.
- If there are no findings, the findings section is omitted entirely.
- "What it does" must be one sentence, written for a reviewer who hasn't read the diff.
- The verdict and `Validation (deterministic)` lines are both mandatory and both carry a
  label prefix, so neither line may ever start with a finding emoji (🔴/⚠️/📝) —
  that is what the `refutations`/`independent` gates count as a reported finding, and a
  bare `⚠️ NEEDS WORK` verdict line gets counted as one, leaving one more "finding" than
  the card actually has. Never omit the `Validation` line: silence on it reads as
  clearance, which is exactly what the spec forbids.

```
## ATOM PR #NNN — [title]

**[One sentence: what this PR does, in plain terms. E.g. "Wraps the per_group emit_bf16 AllReduce path in a torch.library op so Dynamo can trace it."]**

Review: [✅ LGTM | ⚠️ NEEDS WORK | 🔴 BLOCK]
Validation (deterministic): [<verdict> — target, exact runtime, skipped-stage evidence | N/A — no runtime surface changed | NOT RUN — <reason>]

🔴 [specific finding — what, where, why it matters]
⚠️ [specific finding]
📝 [note]
```

**`Validation (deterministic)` — pick exactly one state, never blend them:**
- an exact-head validator report exists → its verdict plus target/runner/arch and any
  skipped stages.
- triage decided no runtime surface changed → `N/A — no runtime surface changed`. Do not
  write `NOT RUN` here; there is no gap to report.
- validation was required but no target exists (P0 ships no validator at all, so this is
  the common case today) → `NOT RUN — <reason>`, e.g. `NOT RUN — the PR changes runtime
  code but ships no test target` or `NOT RUN — no validator in this pipeline yet`.
- required and a target existed, but the run itself could not happen (no idle GPU,
  validator missing, disabled) → `NOT RUN — <reason>`; an environment gap, not a PR
  defect.
- `N/A` and `NOT RUN` are different facts and must never be merged into one word: `N/A`
  means there was nothing to validate, `NOT RUN` means something should have been
  validated and wasn't.

Each finding must have **three parts**:
1. **Problem** — what exactly is wrong, with file/line if relevant
2. **Impact** — what goes wrong at runtime if this is not fixed (wrong output / crash / OOM / perf regression)
3. **Action** — end with a verb phrase: "**Author must** [do X]" or "**Reviewer should ask** [Y]" — no verb = incomplete finding, do not include

**Tag every finding [verified] or [inferred], and never ship a root cause you only inferred.**
- `[verified]` — traced to the actual code/evidence chain.
- `[inferred]` — plausible but unconfirmed; say so and downgrade to "worth checking," do not assert it as the cause.
A finding that stops at "likely / probably the root cause" without an evidence chain is not shippable — either trace it to [verified] or label it [inferred] and frame it as a question.

Do NOT use rule codes (P1, D2, A1…) in output — they are internal labels only.

Examples of good findings:
- `🔴 deepseek_v2.py:1245 changes torch.zeros → torch.empty, but the old comment explicitly says "trailing pad must be zero for asm reader" and the new comment claims "never read" — if padding IS read, every quantized token in the batch gets corrupted output silently. **Author must** cite the asm spec or a test proving padding is not read.`
- `⚠️ PR claims 1.3–1.5x speedup but the benchmark starts after shuffle_weight() completes — users pay that cost on every cold start. **Author must** re-run with shuffle_weight included in the timing window and confirm the speedup holds.`
- `⚠️ Chunked indexer logic is copy-pasted verbatim into deepseek_v2.py and deepseek_v4.py with only variable names changed — if v4's variable semantics differ, the formula silently produces wrong KV offsets for all v4 callers. **Author must** confirm correctness was verified independently under v4's variable layout.`
- `📝 No corresponding ATOM consumer PR mentioned. **Reviewer should ask** who will pass emit_bf16=True to activate this path.`

Examples of bad findings (too vague, missing action verb):
- `⚠️ Missing perf numbers` — no impact, no action
- `🔴 D2 violation` — rule code means nothing to a reviewer
- `⚠️ The benchmark may not include setup cost` — no "Author must" conclusion
```

**Study those four "Examples of good findings" in particular.** They are part of the standard:
each is **one sentence** stating the problem, a short clause stating the impact, and one
`**Author must**` clause stating the action. Your findings must take the same shape. If you
hold more than one piece of evidence, **put the strongest one in the body**; fold the rest into
the same finding's sentence or drop them — `at most 5 findings` already forces the trade-off.

---

# 2. Materials

`{{WORK}}` holds all of Step 1's output: `pr.diff`, `pr_meta.json`, `rules.txt`, `rules_expanded.txt`,
`applies.txt`, `merge_target.txt`, `guards.txt`, `siblings.txt`, `symbols.txt`, `twins.txt`,
`test_quality.txt`, `kernel_tests.txt`, `ci_coverage.txt`, `perf_claims.txt`, `struct_abi.txt`,
`comment_only.txt`, `evidence.txt`, `validation_requirement.json`,
plus `merge-target/` (the base-side worktree) and `head/` (the PR head file tree).

**Both worktrees are read-only**: do not write into them or create symlinks pointing at them. To build an experiment sandbox, make a new directory under `/tmp`.

**Never run `git checkout` / `git switch` / `git reset` / `git stash` in the project root `{{PROJ}}`.** That detaches the checkout's HEAD and can wipe working files (hit for real on 2026-09-16: an agent checked out a base SHA in the main tree to test `git apply`, causing a detached HEAD and removed working files). To test `git apply` against a base commit, use the already-checked-out `merge-target/` worktree, or copy files into a fresh git repo under `/tmp` — **never touch the main tree's HEAD**.

---

# 3. Step 7.7 is not yours

`independent.txt` is produced by a separate agent that has not seen your reasoning. Stop and report once the card is written.
Do not write `independent.txt` yourself, and do not predict its verdict.
So the `independent` gate in Step 8 will necessarily be red right now — that is the expected state.

---

# 4. Operational requirements (SKILL.md does not cover these; they are the orchestration layer's)

1. **Land it to disk before going deep.** The upstream gateway can interrupt. In this order: read SKILL.md → do the Step 2 five questions →
   **immediately write `{{WORK}}/answers.txt` and confirm it with `wc -l`** → then continue the rest.
   Land each output file as soon as it is done. A rough file that exists beats a perfect one that a dropped connection erased.
2. **Cite files this PR changed by their repo-relative path**, with no `head/` / `merge-target/` prefix.
   Start by running `grep '^+++ b/' {{WORK}}/pr.diff | sed 's|^+++ b/||'` to get the changed-file list;
   every FIRE must anchor to a file in that list, or the ledger gate rules `UNTOUCHED-CITATION`.
3. **Do not touch `.artifact_hashes`** — it is the skill's tamper-evidence ledger, maintained by the orchestration layer.
   If it makes a gate misbehave, report it faithfully; do not clean it up yourself.
4. Run gates by absolute path, from `{{PROJ}}`.

---

# 5. Report

The line count of each output file, the exit code of each Step 8 gate, and how many findings are on the card.
**Do not restate the card's contents** — the card is in the file. Report in English.
