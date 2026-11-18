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

## Asking for a run

SKILL.md: "when the action is `run something`, ask for ONE run." The record behind it,
from ATOM reviews labelled by whether the author acted. Seven of them are asks to RUN
something -- three carried out, four not. (Two more judged asks in the same family were
code changes, not runs, and are left out rather than padding the count.)

| | ask | outcome |
|---|---|---|
| ✅ | ATOM#3165 "rebase onto current main and re-run the CPU suite and the `ci:atom` heavy jobs" | done |
| ✅ | ATOM#3275 "one GPU A/B ... once a GPU host is available" (two arms, one node, one sitting) | posted 17 h later |
| ✅ | ATOM#3323 "re-run at least one 512c arm on the final head" | came back with the table |
| ❌ | ATOM#3165 "rebase, re-home `running=self.running` in the refactored `_schedule`, and re-run" | three things, none done |
| ❌ | ATOM#3165 "admission off vs W only vs W + 512 ... post req/s, errors and preemptions" | merged with the box still empty, after 12 commits |
| ❌ | ATOM#3306 "GSM8K plus staged/landed throughput on this exact head" | merged unmet, after 14 commits |
| ❌ | ATOM#3306 "attach **one** P/D throughput run on this head, **or name the existing run**" | merged unmet |

**The last row is the counterexample, and it is the useful one.** It names a single run and
offers the cheaper out, which is exactly what this rule asks for, and it still died. The
difference is not the shape of the ask: ATOM#3306's body says its control run's "node failed
the GPU-idle preflight, and was not rerun", while ATOM#3323 ran "both arms at the same time on
one node" and ATOM#3275 had "one MI355X". The authors who complied had a working setup; the
one who did not, did not.

So minimising the ask is a lever, not a guarantee. A matrix is reliably out of reach. A single
run is out of reach too when the machine is. What the rule buys is that when an author *can*
act, the ask is small enough to be worth acting on -- and the cleanest evidence is still the
ATOM#3165 pair: one author, one PR, one week, cheap ask done and three-arm ask not.

An ask the author cannot fit into their day -- or cannot run at all -- looks, in the record,
exactly like one they disagreed with.
