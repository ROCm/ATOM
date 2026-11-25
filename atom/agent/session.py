# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""The agent loop: brain proposes, tools act, results go back to the brain.

Every message is appended to ``<state_dir>/sessions/<id>.jsonl`` as it
happens, so a crashed or closed chat can be resumed (``--resume``) with its
history intact. A tool call cut off by a crash is answered on load with an
explicit "interrupted" result, so the history stays valid for the API and the
model knows to re-check state instead of assuming the call finished.
"""

from __future__ import annotations

import json
import os
import secrets
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from atom.agent.brain import Brain, BrainError
from atom.agent.tools import ToolBox, render_result

MAX_STEPS = 30
# Past this, older tool results are shortened, then whole old turns dropped.
HISTORY_CHAR_BUDGET = 120_000
KEEP_RECENT_MESSAGES = 12
INTERRUPTED = (
    '{"error": "interrupted: the agent stopped before this tool finished. Its '
    'effect is unknown -- check with deployment_status before acting on it."}'
)


@dataclass
class Hooks:
    on_content: Callable[[str], None] = lambda s: None
    on_reasoning: Callable[[str], None] = lambda s: None
    on_tool_start: Callable[[str, dict], None] = lambda n, a: None
    on_tool_end: Callable[[str, dict], None] = lambda n, r: None
    on_turn_end: Callable[[], None] = lambda: None
    on_notice: Callable[[str], None] = lambda s: None


class SessionLog:
    """Append-only transcript of one conversation."""

    def __init__(self, state_dir: Path, session_id: str | None = None):
        self.dir = Path(state_dir) / "sessions"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.id = session_id or time.strftime("s%Y%m%d-%H%M%S-") + secrets.token_hex(2)
        self.path = self.dir / f"{self.id}.jsonl"

    def _write(self, record: dict) -> None:
        with open(self.path, "a") as f:
            f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
            f.flush()
            os.fsync(f.fileno())

    def meta(self, **fields) -> None:
        self._write({"type": "meta", "ts": time.time(), **fields})

    def message(self, msg: dict) -> None:
        self._write({"type": "message", "message": msg})

    def pop(self) -> None:
        self._write({"type": "pop"})

    @classmethod
    def resolve(cls, state_dir: Path, ref: str) -> str | None:
        """'last' or an id prefix -> session id."""
        sessions = cls.list(state_dir, limit=10_000)
        if ref == "last":
            return sessions[0]["id"] if sessions else None
        matches = [s["id"] for s in sessions if s["id"].startswith(ref)]
        return matches[0] if len(matches) == 1 else None

    @classmethod
    def load(cls, state_dir: Path, session_id: str) -> tuple[dict, list[dict]]:
        path = Path(state_dir) / "sessions" / f"{session_id}.jsonl"
        meta: dict = {}
        messages: list[dict] = []
        for line in path.read_text().splitlines():
            try:
                rec = json.loads(line)
            except ValueError:
                continue  # a line torn by a crash
            if rec.get("type") == "meta":
                meta.update({k: v for k, v in rec.items() if k != "type"})
            elif rec.get("type") == "message":
                messages.append(rec["message"])
            elif rec.get("type") == "pop" and messages:
                messages.pop()
        return meta, repair(messages)

    @classmethod
    def list(cls, state_dir: Path, limit: int = 10) -> list[dict]:
        d = Path(state_dir) / "sessions"
        if not d.is_dir():
            return []
        out = []
        for f in sorted(
            d.glob("s*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True
        )[:limit]:
            first, turns, brain = None, 0, None
            for line in f.read_text().splitlines():
                try:
                    rec = json.loads(line)
                except ValueError:
                    continue
                if rec.get("type") == "meta":
                    brain = rec.get("brain", brain)
                msg = rec.get("message") or {}
                if msg.get("role") == "user":
                    turns += 1
                    first = first or msg.get("content", "")[:80]
            out.append(
                {
                    "id": f.stem,
                    "updated": time.strftime(
                        "%Y-%m-%d %H:%M", time.localtime(f.stat().st_mtime)
                    ),
                    "turns": turns,
                    "brain": brain,
                    "first_message": first,
                }
            )
        return out


def repair(messages: list[dict]) -> list[dict]:
    """Answer tool calls that never got a result (the agent died mid-call)."""
    out: list[dict] = []
    pending: list[str] = []
    for msg in messages:
        if pending and not (
            msg.get("role") == "tool" and msg.get("tool_call_id") in pending
        ):
            out += [
                {"role": "tool", "tool_call_id": i, "content": INTERRUPTED}
                for i in pending
            ]
            pending = []
        out.append(msg)
        if msg.get("role") == "assistant" and msg.get("tool_calls"):
            pending = [c["id"] for c in msg["tool_calls"]]
        elif msg.get("role") == "tool" and msg.get("tool_call_id") in pending:
            pending.remove(msg["tool_call_id"])
    out += [
        {"role": "tool", "tool_call_id": i, "content": INTERRUPTED} for i in pending
    ]
    return out


class ChatSession:
    def __init__(
        self,
        brain: Brain,
        toolbox: ToolBox,
        system_prompt: str,
        hooks: Hooks | None = None,
        log: SessionLog | None = None,
        history: list[dict] | None = None,
    ):
        self.brain = brain
        self.toolbox = toolbox
        self.hooks = hooks or Hooks()
        self.log = log
        self.messages: list[dict] = [{"role": "system", "content": system_prompt}]
        self._note: str | None = None
        if history:
            # A resumed chat: same history, today's system prompt (the
            # machine may have changed since).
            self.messages += [m for m in history if m.get("role") != "system"]
            self._note = (
                "atom resumed this conversation after a restart. Deployments "
                "may have changed since; re-check with deployment_status before "
                "relying on earlier results."
            )

    def _append(self, msg: dict) -> None:
        self.messages.append(msg)
        if self.log:
            self.log.message(msg)

    def reset(self) -> None:
        self.messages = self.messages[:1]
        self._note = None
        if self.log:
            self.log = SessionLog(self.log.dir.parent)
            self.toolbox.ctx.session = self.log.id
            self.toolbox.ctx.audit.session = self.log.id

    def switch_brain(self, brain: Brain) -> None:
        """Swap the LLM from outside the conversation (a /brain command).

        The model only knows what the transcript says, so the next user turn
        carries a note; otherwise it keeps believing it runs on the old one.
        """
        self.brain = brain
        self._note = f"atom switched this conversation to {brain.describe()}."
        if self.log:
            self.log.meta(brain=brain.describe())

    def _compact(self) -> None:
        def size() -> int:
            return sum(len(json.dumps(m, default=str)) for m in self.messages)

        if size() <= HISTORY_CHAR_BUDGET:
            return
        # 1) Shorten old tool results.
        for m in self.messages[1:-KEEP_RECENT_MESSAGES]:
            if m["role"] == "tool" and len(m["content"]) > 400:
                m["content"] = m["content"][:400] + '..." [older result elided]'
        # 2) Drop whole old turns (a user message up to the next one), so a
        #    tool call is never separated from its result.
        while size() > HISTORY_CHAR_BUDGET:
            starts = [i for i, m in enumerate(self.messages) if m["role"] == "user"]
            if len(starts) < 3:
                break
            first, second = starts[0], starts[1]
            dropped = self.messages[first:second]
            del self.messages[first:second]
            self.messages.insert(
                first,
                {
                    "role": "user",
                    "content": f"[{len(dropped)} earlier messages elided to fit the "
                    "context; ask the user or re-run tools if you need them]",
                },
            )
            # Merge the marker with the next user message (no two in a row).
            if self.messages[first + 1]["role"] == "user":
                marker = self.messages.pop(first)
                self.messages[first]["content"] = (
                    marker["content"] + "\n\n" + self.messages[first]["content"]
                )

    def _chat(self):
        """One brain call, falling back to the next brain if this one is down."""
        while True:
            try:
                return self.brain.chat(
                    self.messages,
                    tools=self.toolbox.schemas(),
                    on_content=self.hooks.on_content,
                    on_reasoning=self.hooks.on_reasoning,
                )
            except BrainError as e:
                if not self.brain.fallbacks:
                    raise
                failed, nxt = self.brain, self.brain.fallbacks[0]
                nxt.fallbacks = failed.fallbacks[1:]
                self.hooks.on_notice(
                    f"brain {failed.describe()} failed ({e}); continuing on {nxt.describe()}"
                )
                self.brain = nxt
                self._note = f"atom moved this conversation to {nxt.describe()}."
                if self.log:
                    self.log.meta(brain=nxt.describe(), fallback_from=failed.describe())

    def send(self, user_text: str) -> str:
        if self._note:
            user_text = f"[{self._note}]\n\n{user_text}"
            self._note = None
        self._append({"role": "user", "content": user_text})
        final = ""
        for step in range(MAX_STEPS):
            self._compact()
            try:
                reply = self._chat()
            except BrainError as e:
                if step == 0:
                    # Nothing happened: drop the turn so a retry is clean.
                    self.messages.pop()
                    if self.log:
                        self.log.pop()
                else:
                    # Tools already ran; close the turn so history stays valid.
                    self._append(
                        {"role": "assistant", "content": f"(interrupted: {e})"}
                    )
                final = f"[brain error] {e}"
                break
            msg: dict = {"role": "assistant", "content": reply.content}
            if reply.reasoning:
                # Servers that stream reasoning want it back on tool-call turns
                # (DeepSeek V3.2+/V4 thinking with tools).
                msg["reasoning_content"] = reply.reasoning
            if reply.tool_calls:
                msg["tool_calls"] = [
                    {
                        "id": c.id,
                        "type": "function",
                        "function": {"name": c.name, "arguments": c.arguments or "{}"},
                    }
                    for c in reply.tool_calls
                ]
            self._append(msg)
            self.hooks.on_turn_end()
            if not reply.tool_calls:
                final = reply.content
                break
            for call in reply.tool_calls:
                try:
                    args = call.parsed_arguments()
                except ValueError as e:
                    result = {"error": str(e), "raw_arguments": call.arguments[:500]}
                    args = {}
                else:
                    self.hooks.on_tool_start(call.name, args)
                    result = self.toolbox.call(call.name, args)
                self.hooks.on_tool_end(call.name, result)
                self._append(
                    {
                        "role": "tool",
                        "tool_call_id": call.id,
                        "content": render_result(result),
                    }
                )
            pending = self.toolbox.ctx.pending_brain
            if pending:
                self.toolbox.ctx.pending_brain = None
                self.brain = Brain(
                    pending["base_url"], pending["model"], label=pending["label"]
                )
                if self.log:
                    self.log.meta(brain=self.brain.describe())
        else:
            final = f"(stopped after {MAX_STEPS} tool steps)"
        return final
