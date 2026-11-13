# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Who may approve what, and a record of everything that happened.

Permission modes (``--mode``):

========== ===================================================================
read-only  Nothing that changes the machine runs.
ask        Every machine-changing action asks the user (default).
auto-safe  Actions on resources the agent owns or has leased run directly
           (launch on free leased GPUs, stop its own deployments, switch its
           brain); risky ones still ask (forcing past the sysfs check, shell
           commands outside the allowlist).
auto       Everything runs (``--yes``).
========== ===================================================================

The decision is made here, in code, from the action's risk class; the
prompt never decides what is allowed. Every tool call and every decision is
appended to ``<state_dir>/audit.jsonl``.
"""

from __future__ import annotations

import fcntl
import fnmatch
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

from atom.agent.leases import current_user

MODES = ("read-only", "ask", "auto-safe", "auto")

# Risk classes a tool states when it asks for permission.
SAFE_RISKS = {"launch", "stop_own", "switch_brain", "export"}
RISKY = {"launch_forced", "command"}


@dataclass
class Decision:
    action: str  # allow | ask | deny
    reason: str


class Policy:
    def __init__(self, mode: str = "ask", allow_commands: tuple[str, ...] = ()):
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, not {mode!r}")
        self.mode = mode
        self.allow_commands = tuple(allow_commands)

    def decide(self, risk: str, detail: str = "") -> Decision:
        if self.mode == "read-only":
            return Decision("deny", "read-only mode: nothing that changes the machine")
        if self.mode == "auto":
            return Decision("allow", "auto mode")
        if self.mode == "auto-safe":
            if risk in SAFE_RISKS:
                return Decision("allow", f"auto-safe: {risk} on agent-owned resources")
            if risk == "command" and any(
                fnmatch.fnmatch(detail, pat) for pat in self.allow_commands
            ):
                return Decision("allow", "auto-safe: command matches the allowlist")
        return Decision("ask", f"{self.mode}: {risk} needs the user")


class AuditLog:
    """Append-only JSONL; one line per event, safe across processes."""

    def __init__(self, state_dir: Path, session: str = ""):
        self.path = Path(state_dir) / "audit.jsonl"
        self.session = session

    def write(self, event: str, **fields) -> None:
        record = {
            "ts": round(time.time(), 3),
            "event": event,
            "user": current_user(),
            "pid": os.getpid(),
            "session": self.session or None,
            **fields,
        }
        line = json.dumps(record, default=str, ensure_ascii=False) + "\n"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            f.write(line)
            fcntl.flock(f, fcntl.LOCK_UN)

    def read(self, session: str | None = None, since: float = 0) -> list[dict]:
        if not self.path.exists():
            return []
        out = []
        for line in self.path.read_text().splitlines():
            try:
                rec = json.loads(line)
            except ValueError:
                continue
            if rec.get("ts", 0) >= since and (
                session is None or rec.get("session") == session
            ):
                out.append(rec)
        return out
