# Where this skill's judgement comes from

`SKILL.md` is the **bot pipeline**: what each step fetches, what each gate requires, what
the worker and the refuter are asked. It does not invent the standard it applies, and
ROCm/ATOM PR #2762 supersedes neither document that holds it. Both arrived with #2871 and
both are cited rather than replaced.

- **`maintainer-lens.md`** — the maintainer's lens, verbatim as the ATOM maintainers wrote
  it, moved here when the bot pipeline took the `SKILL.md` path. Its section 3 is what a
  human applies by hand, and the `N1`-`N6` rules in `rules.md` are distilled from it: a
  reviewer reading the lens and a bot applying `N1`-`N6` are applying the same standard.
  **Read this file to review a PR yourself.**
- **`rule-catalog.md`** — the team rule catalog. `rules.md` shares its A-G taxonomy and
  rule ids by descent, with two deliberate divergences recorded at the top of `rules.md`:
  22 rules this skill carries that the catalog does not, and the `P1`-`P5` performance
  family, removed here on evidence and re-addable only one at a time with an ATOM citation.
- **`../scripts/affected_tests.py`** — which tests a diff reaches. Used by validation
  triage to find a target when the PR ships no test of its own.

This lives here rather than in `SKILL.md` because `SKILL.md` is injected into every review
prompt and is held to a 500-line budget; provenance a reviewer reads once does not need to
cost attention on every run.
