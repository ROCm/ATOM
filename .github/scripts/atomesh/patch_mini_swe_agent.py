"""Apply ATOMesh's pinned mini-swe-agent 2.4.5 runtime fixes."""

from __future__ import annotations

import importlib
import importlib.metadata
import sys
from pathlib import Path

EXPECTED_VERSION = "2.4.5"
MARKER = "ATOMesh local-Docker submission fallback"


def replace_once(source: str, old: str, new: str) -> str:
    if new in source:
        return source
    count = source.count(old)
    if count != 1:
        raise RuntimeError(
            f"mini-swe-agent patch anchor {old.splitlines()[0]!r} "
            f"occurred {count} times"
        )
    return source.replace(old, new)


def main() -> int:
    installed = importlib.metadata.version("mini-swe-agent")
    if installed != EXPECTED_VERSION:
        raise RuntimeError(
            f"expected mini-swe-agent {EXPECTED_VERSION}, found {installed}"
        )

    swebench = importlib.import_module("minisweagent.run.benchmarks.swebench")
    path = Path(swebench.__file__)
    source = path.read_text(encoding="utf-8")
    if MARKER in source:
        print(f"[swebench] mini-swe-agent patch already applied: {path}")
        return 0

    source = replace_once(
        source,
        "    agent = None\n" "    exit_status = None",
        "    agent = None\n" "    env = None\n" "    exit_status = None",
    )
    source = replace_once(
        source,
        "        info = agent.run(task)\n"
        '        exit_status = info.get("exit_status")\n'
        '        result = info.get("submission")',
        "        info = agent.run(task)\n"
        '        exit_status = info.get("exit_status")\n'
        '        result = info.get("submission")\n'
        f"        # {MARKER}\n"
        "        if not result and env is not None:\n"
        "            try:\n"
        '                fallback = env.execute({"command": "git diff"})\n'
        '                diff = (fallback.get("output") or "").strip()\n'
        '                if fallback.get("returncode") == 0 and '
        'diff.startswith("diff --git"):\n'
        '                    result = diff + "\\n"\n'
        '                    extra_info["submission_source"] = '
        'f"fallback_after_{exit_status}"\n'
        "            except Exception:\n"
        "                pass",
    )
    source = replace_once(
        source,
        '        exit_status, result = type(e).__name__, ""\n'
        '        extra_info = {"traceback": traceback.format_exc(), '
        '"exception_str": str(e)}',
        '        exit_status, result = type(e).__name__, ""\n'
        '        extra_info = {"traceback": traceback.format_exc(), '
        '"exception_str": str(e)}\n'
        "        if env is not None:\n"
        "            try:\n"
        '                fallback = env.execute({"command": "git diff"})\n'
        '                diff = (fallback.get("output") or "").strip()\n'
        '                if fallback.get("returncode") == 0 and '
        'diff.startswith("diff --git"):\n'
        '                    result = diff + "\\n"\n'
        '                    extra_info["submission_source"] = '
        'f"fallback_after_{exit_status}"\n'
        "            except Exception:\n"
        "                pass",
    )
    source = replace_once(
        source,
        "    finally:\n" "        if agent is not None:",
        "    finally:\n"
        '        if env is not None and callable(getattr(env, "cleanup", None)):\n'
        "            try:\n"
        "                env.cleanup()\n"
        "            except Exception:\n"
        "                pass\n"
        "        if agent is not None:",
    )

    path.write_text(source, encoding="utf-8")
    print(f"[swebench] patched mini-swe-agent for local Docker: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
