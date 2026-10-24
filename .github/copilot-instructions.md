# Copilot review instructions — ATOM

Do not comment on anything `ruff` or `black` already enforces (formatting,
import order, line length, naming case). CI runs both in
`.github/workflows/pre-checks.yaml`. Report only what a linter cannot see.

Rank findings by severity and report at most 10. Prefer one precise finding
with a `file:line` and a named alternative over three vague ones. If a section
below yields nothing on a diff, say nothing for it.

## Correctness

- When a change fixes a bug, grep for the same pattern across the repo and name
  every other occurrence left unfixed in the same diff.
- Report comments and docstrings that no longer match the code, including a
  docstring that states a check the code does not perform.
- Report `assert` used as a runtime guard on a serving path. This repo's
  convention is `raise` for anything that must survive `python -O`; several
  modules say so in comments where they chose `raise` deliberately.
- Report a value snapshotted at one point and used at another without
  revalidation, especially an index, slot, or timestamp.
- Report a raised exception that a caller will silently swallow — an
  `AttributeError` reaching a `getattr(obj, name, default)`, or anything caught
  by a bare `except` that then continues.

## Cleanliness

- Report code that reimplements a helper already in the repo. Name the existing
  symbol and the file it lives in.
- Report a hardcoded literal whose shared named constant is imported by the very
  same file.
- Report dead code: functions with no caller, parameters never read in the body,
  return values discarded and then recomputed.
- Report a new setting that bypasses the mechanism its siblings use — a raw
  `os.environ` read beside registered entries in `atom/utils/envs.py`.

## Elegance

- Report an encoder and its inverse living in different files, or any format
  whose two ends are not adjacent.
- Report the same fact derived independently in more than one place. Say which
  site should own it and which should read from it.
- Report sibling branches in one function handled inconsistently — several that
  degrade gracefully and one that raises — with no stated principle separating
  them.
- Report code that temporarily mutates shared or thread-local state and restores
  it, where passing the value explicitly would do.

## Performance

- Report per-call work added to a hot path: per-request allocation, invariants
  revalidated on every call, string formatting for a probe that is off by
  default.
- Report an ungated `record_function` or debug block that costs time even when
  the feature is disabled.
- Report a property or helper that rebuilds a dict, list, or set on every read,
  especially one called inside a loop.
- Report a loop or buffer bounded by a configured maximum (`max_model_len`, a
  capture width) instead of the live length.
- Report a runtime quantity promoted to a compile-time constant (`tl.constexpr`,
  a captured shape), which forces a recompile per distinct value.
- State the cost in the units that matter here: per decode step, per layer, per
  request.

## Organization and boundaries

- Report a file whose name does not describe its contents, and a module holding
  two unrelated responsibilities — config parsing inside a runtime module, a
  transport inside a file named for selection.
- Report a new feature spread across several existing large files where a new
  module would isolate it. Name the files and the natural seam.
- Report a function this change pushes past 50 lines, and a file it pushes past
  800 lines or grows substantially when already past, and say what would split
  out. Do not report size the diff did not move: roughly a tenth of this repo
  is already over those thresholds, including most of the files that change
  most often.
- Report one policy decided at several scattered call sites instead of in one
  table. List the sites.
- Report a name that stopped describing its behaviour after the change, and
  propose the corrected name.
