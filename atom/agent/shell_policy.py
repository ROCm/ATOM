# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Which shell commands may run inside a deployment without asking.

Skills are full of inline shell: poll loops, ``find | xargs ls``, ``zcat |
python3 -c 'import json ...'``, ``sed -n 1,45p``. Treating every compound
command as "arbitrary" forces a human to approve each read-only step (and in
an unattended run, declines it). So a command is split into its simple
commands -- across ``;``, ``&&``, ``||``, ``|``, ``$(...)``, loop and
conditional keywords -- with ``VAR=...`` assignments resolved, and judged
part by part:

* ``forbidden`` -- starts/stops servers or resets GPUs (never runs);
* ``workload``  -- every part is a read-only tool, a known workload entry
  point (the repos' benchmark/eval/drain scripts), or a write confined to the
  scratch areas the scripts use (``/app/logs_claude``, ``/tmp``);
* ``exec``      -- anything else; the user decides.

This is a guard for cooperative agents, not a sandbox: anything it cannot
read precisely falls through to ``exec`` (ask), never to ``workload``.
"""

from __future__ import annotations

import os
import re
import shlex

# Executing these is forbidden; mentioning them (ls, cat, grep) is not.
LIFECYCLE_SCRIPT = re.compile(
    r"(^|/)(start_atom_server|stop_atom_server|wait_server_ready|run_debug_agent)\.sh$"
)
KILLERS = {"pkill", "killall", "reboot", "shutdown", "poweroff", "halt"}
SHELLS = {"bash", "sh", "source", ".", "exec", "nohup", "setsid", "nice", "timeout"}
# Fallback when a command cannot be parsed: any mention counts.
FORBIDDEN = re.compile(
    r"\b(start_atom_server|stop_atom_server|wait_server_ready|run_debug_agent)\.sh\b"
    r"|\b(pkill|killall|reboot|shutdown|poweroff)\b|rocm-smi\s+--gpureset|amd-smi\s+reset"
)

READ_ONLY = {
    "ls", "cat", "zcat", "head", "tail", "grep", "egrep", "zgrep", "wc", "sort",
    "uniq", "cut", "echo", "printf", "test", "[", "[[", "seq", "sleep", "stat",
    "du", "df", "true", "false", "date", "basename", "dirname", "tr", "nproc",
    "rocm-smi", "amd-smi", "pgrep", "ps", "free", "uptime", "which", "pwd",
    "jq", "md5sum", "sha256sum", "file", "readlink", "realpath", "diff", "cmp",
    "column", "tac", "nl", "expr", "bc", "lm_eval", "find", "sed", "awk",
    "xargs", "gzip", "env", "printenv", "hostname", "id", "whoami",
    # builtins that run nothing themselves
    "break", "continue", "exit", "return", ":", "wait", "set", "cd",
}  # fmt: skip
SCRATCH_WRITERS = {"mkdir", "rm", "touch", "cp", "mv", "tar", "gunzip", "tee"}
SCRATCH = ("/app/logs_claude", "/tmp")
KEYWORDS = {"do", "done", "then", "else", "elif", "fi", "esac", "!", "{", "}", "time"}
WORKLOAD_SCRIPTS = re.compile(
    r"^(/app/ATOM/)?scripts/"
    r"(run_benchmark|run_benchmark_sweep|run_gsm8k_eval|wait_infer_drain)\.sh$"
)
WORKLOAD_TOOLS = re.compile(
    r"^(/app/ATOM/)?tools/(parse_trace|analyze_trace_summary)\.py$"
)
SAFE_PY_IMPORTS = {
    "json", "sys", "re", "math", "collections", "statistics", "gzip",
    "itertools", "functools", "operator", "glob", "time", "datetime",
}  # fmt: skip
_PY_DANGER = re.compile(
    r"__import__|\bexec\(|\beval\(|\bcompile\(|\bopen\([^)]*['\"][wax+]|"
    r"\bos\.|\bsubprocess\b|\bshutil\b|\bsocket\b|\bctypes\b|\bPath\("
)
_ASSIGN = re.compile(r"\b([A-Za-z_]\w*)=(\"[^\"]*\"|'[^']*'|[^\s;&|)]+)")
_REDIRECT = re.compile(r"(?<![0-9&<>])(?:[0-9]?)>{1,2}\s*([^\s;&|)]+)")
_SPLIT = re.compile(
    r"\$\(|`|\|\||&&|;|\||\n|\(|\)|\b(?:do|then|else|elif)\b|\bdone\b|\bfi\b"
)


def _expand(command: str) -> str:
    """Resolve VAR=... assignments made in the command itself."""
    values = {}
    for name, value in _ASSIGN.findall(command):
        values[name] = value.strip("\"'")

    def sub(m: re.Match) -> str:
        return values.get(m.group(1) or m.group(2), m.group(0))

    return re.sub(r"\$\{(\w+)\}|\$(\w+)", sub, command)


def _in_scratch(path: str, scratch: tuple[str, ...] = SCRATCH) -> bool:
    path = path.strip("\"'")
    if path in ("/dev/null", "-"):
        return True
    norm = os.path.normpath(path) if path.startswith("/") else ""
    return any(norm == s or norm.startswith(s + "/") for s in scratch)


def _python_ok(code: str) -> bool:
    for line in code.splitlines():
        m = re.match(r"\s*(?:import\s+([\w.,\s]+)|from\s+([\w.]+)\s+import)", line)
        if m:
            mods = [
                x.strip().split(".")[0] for x in (m.group(1) or m.group(2)).split(",")
            ]
            if any(mod not in SAFE_PY_IMPORTS for mod in mods if mod):
                return False
    return not _PY_DANGER.search(code)


def _strip_prefix(words: list[str]) -> list[str]:
    while words and (
        re.fullmatch(r"[A-Za-z_]\w*=.*", words[0]) or words[0] in KEYWORDS
    ):
        words = words[1:]
    return words


def _part_forbidden(words: list[str]) -> bool:
    """Does this simple command start/stop a server, kill by name, or reset a GPU?"""
    words = _strip_prefix(words)
    if not words:
        return False
    head = os.path.basename(words[0])
    if LIFECYCLE_SCRIPT.search(words[0]) or head in KILLERS:
        return True
    if head in SHELLS:
        # bash scripts/start_atom_server.sh, timeout 60 bash ..., nohup ...
        return any(
            LIFECYCLE_SCRIPT.search(w) or os.path.basename(w) in KILLERS
            for w in words[1:]
        )
    if head == "rocm-smi" and any("gpureset" in w for w in words):
        return True
    return head == "amd-smi" and "reset" in words[1:]


def _part_ok(words: list[str], scratch: tuple[str, ...] = SCRATCH) -> bool:
    words = _strip_prefix(words)
    if not words:
        return True
    if words[0] in ("for", "while", "until", "if", "case", "in"):
        return True  # loop/condition header; its commands are other parts
    head = os.path.basename(words[0])
    args = words[1:]
    if head in ("bash", "sh"):
        return bool(args) and bool(WORKLOAD_SCRIPTS.match(args[0]))
    if head in ("python", "python3"):
        if args[:2] in (["-m", "atom.benchmarks.benchmark_serving"],) or (
            args[:1] == ["-m"]
            and len(args) > 1
            and args[1].startswith("atom.benchmarks.")
        ):
            return True
        if args and WORKLOAD_TOOLS.match(args[0]):
            return True
        if args[:1] == ["-c"] and len(args) > 1:
            return _python_ok(args[1])
        return False
    if head == "curl":
        urls = [a for a in args if "://" in a]
        return bool(urls) and all(
            re.match(r"https?://(localhost|127\.0\.0\.1)(:\d+)?/", u) for u in urls
        )
    if head == "sed":
        return not any(a.startswith(("-i", "--in-place")) for a in args)
    if head == "awk":
        prog = " ".join(args)
        return "system(" not in prog and not re.search(r"print[^;]*>", prog)
    if head == "find":
        return not any(
            a in ("-delete", "-exec", "-execdir", "-ok", "-okdir") for a in args
        )
    if head == "xargs":
        rest = [a for a in args if not a.startswith("-")]
        return bool(rest) and os.path.basename(rest[0]) in READ_ONLY - {"xargs"}
    if head == "gzip":
        return any(a in ("-t", "-l", "--test", "--list") for a in args)
    if head in SCRATCH_WRITERS:
        paths = [a for a in args if not a.startswith("-")]
        if head in ("cp", "mv") and paths:
            paths = paths[-1:]  # only the destination is written
        return bool(paths) and all(_in_scratch(p, scratch) for p in paths)
    return head in READ_ONLY


def classify(command: str, scratch: tuple[str, ...] = SCRATCH) -> str:
    """forbidden | workload | exec -- see the module docstring.

    ``scratch`` is where writes are allowed: the container's script output
    areas by default; ``()`` (reads only) for commands on the host.
    """
    expanded = _expand(command)
    redirect_targets = _REDIRECT.findall(expanded)
    # Redirects are judged; drop them so the lexer sees only commands.
    expanded = re.sub(r"&>\s*/dev/null|[0-9]?>&[0-9]", " ", expanded)
    expanded = re.sub(r"[0-9]?>{1,2}\s*[^\s;&|)]+", " ", expanded)
    if "`" in expanded:
        return "forbidden" if FORBIDDEN.search(expanded) else "exec"
    # find's escaped grouping parens are arguments, not subshells.
    expanded = expanded.replace("\\(", " __lp__ ").replace("\\)", " __rp__ ")
    # Quotes stay intact (python -c code, grep patterns); separators split.
    try:
        lexed = shlex.shlex(
            expanded.replace("$(", " ( "), posix=True, punctuation_chars=";&|()"
        )
        lexed.whitespace_split = True
        tokens = list(lexed)
    except ValueError:
        # Unparseable: be conservative about anything that looks dangerous.
        return "forbidden" if FORBIDDEN.search(expanded) else "exec"
    separators = {";", "&&", "||", "|", "(", ")", "&", ";;", "&;", "|&"}
    breaks = {"do", "then", "else", "elif", "done", "fi"}
    cleaned: list[list[str]] = [[]]
    for tok in tokens:
        if tok in separators or tok in breaks:
            cleaned.append([])
        else:
            cleaned[-1].append(tok)
    if any(_part_forbidden(p) for p in cleaned):
        return "forbidden"
    for target in redirect_targets:
        if not target.startswith("&") and not _in_scratch(target, scratch):
            return "exec"
    return "workload" if all(_part_ok(p, scratch) for p in cleaned) else "exec"
