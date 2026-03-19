#!/usr/bin/env python3
"""PostToolUse hook: auto-format Python files with black + ruff after Edit/Write."""

import json
import shutil
import subprocess
import sys

data = sys.stdin.read()
info = json.loads(data)
file_path = info.get("tool_input", {}).get("file_path", "")

if file_path.endswith(".py"):
    black = shutil.which("black")
    ruff = shutil.which("ruff")
    if black:
        subprocess.run([black, "--quiet", file_path], capture_output=True)
    if ruff:
        subprocess.run(
            [ruff, "check", "--fix", "--quiet", file_path], capture_output=True
        )

# Output original data to stdout per hook protocol
print(data)
