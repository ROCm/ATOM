# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""ATOM's and AITER's skills as the agent's procedures.

The repos already carry their engineers' playbooks as Claude Code skills
(``.claude/skills/<name>/SKILL.md``) and commands (``.claude/commands/*.md``).
The agent uses those files as they are -- one source of truth for both
front-ends -- read from the image that will run (see ``image.py``) so a skill
always matches the code it drives.

Progressive disclosure: the system prompt lists name + one-line description;
``load_skill`` returns the body; ``read_skill_file`` returns references and
scripts. Names are namespaced by repo (``atom:review-pr`` vs
``aiter:review-pr``).

Profiles decide what is visible. ``customer`` gets the deployment-side
procedures (run workloads, benchmark, trace, debug guide); ``developer``
gets everything, including kernel authoring and PR review, which assume a
code-editing agent.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

# Visible to customers: running, measuring and diagnosing deployments.
CUSTOMER_SKILLS = {
    "atom:run-atom-workload",
    "atom:capture-trace",
    "atom:benchmark-guide",
    "atom:debug-guide",
}
PROFILES = ("customer", "developer")
MAX_BODY_CHARS = 24_000


@dataclass
class Skill:
    name: str  # namespaced: "atom:capture-trace"
    origin: str  # atom | aiter | user
    kind: str  # skill | command
    description: str
    path: Path  # SKILL.md or the command's .md
    meta: dict = field(default_factory=dict)

    @property
    def root(self) -> Path:
        return self.path.parent if self.kind == "skill" else self.path.parent

    def files(self) -> list[str]:
        if self.kind != "skill":
            return []
        return sorted(
            str(p.relative_to(self.root))
            for p in self.root.rglob("*")
            if p.is_file() and p.name != "SKILL.md"
        )[:200]


def parse_frontmatter(text: str) -> tuple[dict, str]:
    """The YAML subset skills use: ``key: value`` and folded/literal blocks."""
    if not text.startswith("---"):
        return {}, text
    end = text.find("\n---", 3)
    if end == -1:
        return {}, text
    head, body = text[3:end], text[end + 4 :].lstrip("\n")
    meta: dict = {}
    key = None
    block: list[str] = []

    def flush():
        if key is not None and block:
            meta[key] = " ".join(x.strip() for x in block if x.strip())

    for line in head.splitlines():
        m = re.match(r"^([A-Za-z_][\w-]*):\s*(.*)$", line)
        if m and not line.startswith((" ", "\t")):
            flush()
            key, value = m.group(1), m.group(2).strip()
            block = []
            if value in (">", "|", ">-", "|-", ""):
                continue  # block scalar follows on indented lines
            meta[key] = value.strip("'\"")
            key = None
        elif key is not None:
            block.append(line)
    flush()
    return meta, body


def _first_paragraph(body: str) -> str:
    for para in re.split(r"\n\s*\n", body):
        text = " ".join(
            ln.strip()
            for ln in para.splitlines()
            if ln.strip() and not ln.startswith("#")
        )
        if text and not text.startswith("```"):
            return text[:300]
    return ""


def load_catalog(sources: list[tuple[str, Path]]) -> dict[str, Skill]:
    """sources: (origin, .claude dir) pairs; later sources do not override."""
    skills: dict[str, Skill] = {}
    for origin, claude_dir in sources:
        claude_dir = Path(claude_dir)
        for skill_md in sorted(claude_dir.glob("skills/*/SKILL.md")):
            meta, body = parse_frontmatter(skill_md.read_text(errors="replace"))
            short = meta.get("name") or skill_md.parent.name
            name = f"{origin}:{short}"
            skills.setdefault(
                name,
                Skill(
                    name,
                    origin,
                    "skill",
                    meta.get("description") or _first_paragraph(body),
                    skill_md,
                    meta,
                ),
            )
        for cmd in sorted(claude_dir.glob("commands/*.md")):
            meta, body = parse_frontmatter(cmd.read_text(errors="replace"))
            name = f"{origin}:{cmd.stem}"
            title = next(
                (
                    ln.lstrip("# ").strip()
                    for ln in body.splitlines()
                    if ln.startswith("#")
                ),
                cmd.stem,
            )
            skills.setdefault(
                name,
                Skill(
                    name,
                    origin,
                    "command",
                    meta.get("description") or f"{title}. {_first_paragraph(body)}",
                    cmd,
                    meta,
                ),
            )
    return skills


def visible(skills: dict[str, Skill], profile: str) -> dict[str, Skill]:
    if profile == "developer":
        return skills
    return {
        k: v for k, v in skills.items() if k in CUSTOMER_SKILLS or v.origin == "user"
    }


def resolve(skills: dict[str, Skill], name: str) -> Skill | None:
    """'atom:capture-trace', or a bare 'capture-trace' when unambiguous."""
    if name in skills:
        return skills[name]
    matches = [s for k, s in skills.items() if k.split(":", 1)[1] == name]
    return matches[0] if len(matches) == 1 else None


def catalog_text(skills: dict[str, Skill], limit: int = 170) -> str:
    lines = []
    for s in sorted(skills.values(), key=lambda s: s.name):
        desc = re.sub(r"\s+", " ", s.description).strip()
        lines.append(f"- {s.name}: {desc[:limit]}{'...' if len(desc) > limit else ''}")
    return "\n".join(lines)


ADAPTER_NOTES = """\
Skills are ATOM/AITER engineers' procedures written for Claude Code in a shell \
inside the ATOM container. When you follow one, translate:
- Never run scripts/start_atom_server.sh, stop_atom_server.sh, \
wait_server_ready.sh or run_debug_agent.sh -- they kill and start servers \
behind the lease system. Use plan_deployment / launch_server (extra server \
flags via add_args, env such as ATOM_PROFILER_MORE via set_env), \
wait_until_ready and stop_server instead.
- "Bash" = exec_in_runtime against the deployment (cwd is the repo root, \
e.g. /app/ATOM; scripts write to /app/logs_claude). A trailing "&", \
"run_in_background" or anything over a minute = background=true, then \
wait_job; "Monitor"/poll loops = wait_job / job_log.
- A deployment started by launch_server is a fresh container: its \
directories are already empty. Skip a skill's "clean before relaunch" steps \
(e.g. `rm -rf $TRACE_DIR`) -- doing them after the launch deletes what the \
server has just set up.
- Files the user should get (traces, result JSON) come back with \
copy_from_runtime.
- Model paths and ports are the same inside the container as on the host."""
