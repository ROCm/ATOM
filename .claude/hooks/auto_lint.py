#!/usr/bin/env python3
"""PostToolUse hook: black + ruff on the file just edited, and say what is left.

Fixing silently is only half of it. `ruff --fix` cannot fix every rule -- RUF012
is one it only advises on -- and a violation it leaves behind used to vanish
here, because this hook captured ruff's output and never read it. CI does not
vanish: reviewdog filters to the pull request's diff context, so any finding on
a line this edit moved becomes a red check, whatever the repository's total was
before.

So whatever survives `--fix` goes back to Claude on stderr with exit 2, which
for a PostToolUse hook means "show this, the tool already ran". Pre-existing
findings in the file are reported too, on purpose: they are exactly the ones CI
picks up once an edit lands beside them.
"""

import json
import shutil
import subprocess
import sys

data = sys.stdin.read()
info = json.loads(data)
file_path = info.get("tool_input", {}).get("file_path", "")

remaining = ""
if file_path.endswith(".py"):
    black = shutil.which("black")
    ruff = shutil.which("ruff")
    # `check=False` throughout: a formatter that finds nothing to do and a
    # linter that finds something both exit non-zero, and neither is this
    # hook's failure -- the second one is its whole output.
    if black:
        subprocess.run([black, "--quiet", file_path], capture_output=True, check=False)
    if ruff:
        subprocess.run(
            [ruff, "check", "--fix", "--quiet", file_path],
            capture_output=True,
            check=False,
        )
        left = subprocess.run(
            [ruff, "check", "--output-format=concise", file_path],
            capture_output=True,
            text=True,
            check=False,
        )
        if left.returncode:
            remaining = left.stdout.strip()

# Output original data to stdout per hook protocol
print(data)

if remaining:
    print(
        f"ruff still reports these in {file_path} after --fix. CI reviewdog "
        "flags any of them that land in the diff context, so fix them now "
        "rather than after the push:\n" + remaining,
        file=sys.stderr,
    )
    sys.exit(2)
