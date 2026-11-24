"""Run the unit tests a PR can affect, the way CI runs them, and report only
what the PR changed.

Scope: the test files the diff touches, plus every test that imports a changed
`atom/` module (as `atom.a.b.c` or `from atom.a.b import c`, at any depth in
the file, local imports included). Modules imported by more than --hub-limit
test files (envs.py, config.py) are hubs: they would pull in the whole suite,
so they are listed but not expanded.

Two arms, each logged under <out_dir>:
  ci_sim  CPU torch with aiter / triton / FlyDSL absent, as on the CI runner.
          A file that cannot import is reported and does not stop the rest.
  local   the real environment on the given GPUs, for the files ci_sim skipped
          or could not import (GPU and aiter-backed tests).

Every test that fails on the head is rerun at the merge base (exported with
`git archive` into <out_dir>/base_tree; no worktree is created). The summary
separates regressions (pass at base, fail at head), failing tests the PR adds,
and failures that predate the PR.

Usage (start it in the background, read code meanwhile):
  python affected_tests.py <tree> <merge-base> <out_dir> [--gpus 4,5,6,7]
"""

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

CI_SIM = "/app/logs_claude/ci_sim"
PYTEST_IGNORE = [
    "--ignore=tests/plugin",
    "--ignore=tests/entrypoints/test_openai_server.py",
]
# FAILED / ERROR lines of pytest's short summary; ERROR may name a whole file.
_PROBLEM = re.compile(r"^(?:FAILED|ERROR) (tests/\S+?)(?: - |$)")
_SKIPPED_FILE = re.compile(r"^SKIPPED \[\d+\] (tests/\S+\.py)(?::\d+)?: ")


def changed_files(tree: Path, base: str) -> list[str]:
    out = subprocess.check_output(
        ["git", "-C", str(tree), "diff", "--name-only", f"{base}...HEAD"], text=True
    )
    return [f for f in out.split() if (tree / f).exists()]


def module_pattern(path: str) -> re.Pattern:
    """`atom/a/b/c.py` -> matches `atom.a.b.c` or `from atom.a.b import ..., c`."""
    mod = path[: -len(".py")].replace("/", ".").removesuffix(".__init__")
    parent, _, leaf = mod.rpartition(".")
    return re.compile(
        rf"\b{re.escape(mod)}\b"
        rf"|from\s+{re.escape(parent)}\s+import\s+[^\n]*\b{re.escape(leaf)}\b"
    )


def affected_tests(tree: Path, changed: list[str], hub_limit: int):
    tests = sorted(
        p for p in (tree / "tests").rglob("test_*.py") if "plugin" not in p.parts
    )
    texts = {p: p.read_text(errors="ignore") for p in tests}
    picked = {tree / f for f in changed if f.startswith("tests/") and f.endswith(".py")}
    hubs = []
    for f in changed:
        if not (f.startswith("atom/") and f.endswith(".py")):
            continue
        pat = module_pattern(f)
        users = [p for p, t in texts.items() if pat.search(t)]
        if len(users) > hub_limit:
            hubs.append((f, len(users)))
            continue
        picked.update(users)
    rel = sorted(str(p.relative_to(tree)) for p in picked if p.name.startswith("test_"))
    return rel, hubs


def pytest(
    cwd: Path, targets: list[str], env: dict, log: Path
) -> tuple[str, list[str]]:
    """Run pytest; return its last line and the failing / erroring ids."""
    cmd = [
        sys.executable, "-m", "pytest", "-q", "-rfEs", "-p", "no:cacheprovider",
        "--continue-on-collection-errors", *PYTEST_IGNORE, *targets,
    ]  # fmt: skip
    with open(log, "w") as fh:
        subprocess.run(
            cmd, cwd=cwd, env=env, stdout=fh, stderr=subprocess.STDOUT, check=False
        )
    lines = log.read_text().strip().splitlines()
    problems = [m.group(1) for line in lines if (m := _PROBLEM.match(line))]
    return (lines[-1] if lines else "(no output)"), sorted(set(problems))


def export_base(tree: Path, base: str, dest: Path) -> Path:
    if not dest.exists():
        dest.mkdir(parents=True)
        archive = subprocess.Popen(
            ["git", "-C", str(tree), "archive", base], stdout=subprocess.PIPE
        )
        subprocess.run(["tar", "-x", "-C", str(dest)], stdin=archive.stdout, check=True)
        archive.wait()
    return dest


def compare_at_base(name, head_fail, base_tree, env_for, out) -> list[str]:
    """Rerun the head's failures at the merge base; return the summary lines."""
    at_base = [i for i in head_fail if (base_tree / i.split("::")[0]).exists()]
    new_tests = [i for i in head_fail if i not in at_base]
    base_fail = set()
    if at_base:
        last, base_fail = pytest(
            base_tree, at_base, env_for(base_tree), out / f"{name}_base.log"
        )
        base_fail = set(base_fail)
        # A base run that reports no outcome compared nothing: an empty failure
        # list there must not turn every head failure into a regression.
        if not re.search(r"\d+ (passed|failed|errors?)\b", last):
            return [
                f"{name}: base run reported no results ({last!r}); not compared"
            ] + [f"  UNCOMPARED {i}" for i in head_fail]
    # A file-level error at base covers every id in that file.
    regressions = [
        i for i in at_base if i not in base_fail and i.split("::")[0] not in base_fail
    ]
    lines = [
        (
            f"{name}: {len(head_fail)} failing on head; "
            f"{len(regressions)} regressions, {len(new_tests)} in tests the PR adds, "
            f"{len(at_base) - len(regressions)} also fail at the merge base"
        )
    ]
    lines += [f"  REGRESSION {i}" for i in regressions]
    lines += [f"  NEW-TEST   {i}" for i in new_tests]
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tree", type=Path)
    ap.add_argument("base")
    ap.add_argument("out", type=Path)
    ap.add_argument("--gpus", default="4,5,6,7")
    ap.add_argument("--hub-limit", type=int, default=30)
    a = ap.parse_args()
    tree, out = a.tree.resolve(), a.out.resolve()
    out.mkdir(parents=True, exist_ok=True)

    files, hubs = affected_tests(tree, changed_files(tree, a.base), a.hub_limit)
    (out / "affected_tests.txt").write_text(
        "\n".join(files)
        + "".join(f"\n# hub, not expanded: {h} ({n} test files)" for h, n in hubs)
        + "\n"
    )
    print(f"{len(files)} test files; hubs not expanded: {hubs}")
    if not files:
        return

    env = {**os.environ, "PYTHONSAFEPATH": "1", "HIP_VISIBLE_DEVICES": a.gpus}
    arms = {
        "ci_sim": lambda t: {**env, "PYTHONPATH": f"{CI_SIM}:{t}"},
        "local": lambda t: {**env, "PYTHONPATH": str(t)},
    }

    last, ci_fail = pytest(tree, files, arms["ci_sim"](tree), out / "ci_sim.log")
    print("ci_sim:", last)
    ci_log = (out / "ci_sim.log").read_text().splitlines()
    skipped = {m.group(1) for line in ci_log if (m := _SKIPPED_FILE.match(line))}
    not_imported = {i for i in ci_fail if "::" not in i}
    local_files = sorted(f for f in files if f in skipped | not_imported)
    local_fail = []
    if local_files:
        last, local_fail = pytest(
            tree, local_files, arms["local"](tree), out / "local.log"
        )
        print(
            f"local ({len(local_files)} files ci_sim skipped or could not import):",
            last,
        )

    summary = []
    for name, fails in (("ci_sim", ci_fail), ("local", local_fail)):
        if fails:
            base_tree = export_base(tree, a.base, out / "base_tree")
            summary += compare_at_base(name, fails, base_tree, arms[name], out)
        else:
            summary.append(f"{name}: nothing failing on head")
    (out / "summary.txt").write_text("\n".join(summary) + "\n")
    print("\n".join(summary))


if __name__ == "__main__":
    main()
