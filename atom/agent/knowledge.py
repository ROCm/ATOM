# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""ATOM's own documentation as the agent's knowledge base.

Recipes (recipes/**.md), guides (docs/), and the repo's Claude commands and
skills (.claude/) are what ATOM engineers already wrote down about deploying
and debugging it. They are chunked by heading and searched with BM25, and
recipe code blocks are parsed into launch commands the planner can reuse.
"""

from __future__ import annotations

import math
import re
import shlex
from collections import Counter
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

SOURCES = (
    "recipes/**/*.md",
    "docs/*.md",
    "docs/*.rst",
    ".claude/commands/*.md",
    ".claude/skills/*/SKILL.md",
    "README.md",
)
SERVER_MODULE = "atom.entrypoints.openai_server"
_TOKEN = re.compile(r"[a-z0-9]+(?:\.[0-9]+)*")
_RST_UNDERLINE = re.compile(r"^([=\-~^\"'`#*+])\1{2,}\s*$")


def tokenize(text: str) -> list[str]:
    out = []
    for tok in _TOKEN.findall(text.lower()):
        out.append(tok)
        if "." in tok:  # "v4.1" also matches a query for "v4"
            out.append(tok.split(".", 1)[0])
    return out


@dataclass
class Chunk:
    path: str
    heading: str
    text: str
    tokens: Counter = field(default_factory=Counter)
    length: int = 0


def _split_sections(path: str, text: str) -> list[tuple[str, str]]:
    """(heading trail, body) per markdown / reStructuredText section."""
    lines = text.splitlines()
    trail: list[str] = []
    sections: list[tuple[str, list[str]]] = [("", [])]
    in_fence = False
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
        heading = None
        if not in_fence:
            m = re.match(r"^(#{1,4})\s+(.*)", line)
            if m and path.endswith(".md"):
                heading = (len(m.group(1)), m.group(2).strip())
            elif (
                path.endswith(".rst")
                and i + 1 < len(lines)
                and line.strip()
                and _RST_UNDERLINE.match(lines[i + 1])
            ):
                level = "=-~^".find(lines[i + 1][0]) + 1 or 4
                heading = (level, line.strip())
                i += 1
        if heading:
            level, title = heading
            trail = trail[: level - 1] + [title]
            sections.append((" > ".join(trail), [line]))
        else:
            sections[-1][1].append(line)
        i += 1
    return [(h, "\n".join(body).strip()) for h, body in sections if "".join(body)]


class KnowledgeBase:
    def __init__(self, root: Path | None):
        self.root = root
        self.chunks: list[Chunk] = []
        self.files: dict[str, str] = {}
        if root is None:
            return
        for pattern in SOURCES:
            for f in sorted(root.glob(pattern)):
                rel = str(f.relative_to(root))
                if rel in self.files:
                    continue
                try:
                    text = f.read_text(errors="replace")
                except OSError:
                    continue
                self.files[rel] = text
                name_tokens = tokenize(f.stem)
                for heading, body in _split_sections(rel, text):
                    toks = Counter(tokenize(body))
                    # Titles say what a section is about; weight them up.
                    for t in tokenize(heading) + name_tokens:
                        toks[t] += 3
                    self.chunks.append(
                        Chunk(rel, heading, body, toks, sum(toks.values()))
                    )
        n = len(self.chunks) or 1
        self.avg_len = sum(c.length for c in self.chunks) / n
        df: Counter = Counter()
        for c in self.chunks:
            df.update(c.tokens.keys())
        self.idf = {t: math.log(1 + (n - d + 0.5) / (d + 0.5)) for t, d in df.items()}

    def search(self, query: str, k: int = 6, path_prefix: str | None = None) -> list:
        q = tokenize(query)
        k1, b = 1.4, 0.75
        scored = []
        for c in self.chunks:
            if path_prefix and not c.path.startswith(path_prefix):
                continue
            s = 0.0
            for t in q:
                tf = c.tokens.get(t)
                if tf:
                    norm = tf + k1 * (1 - b + b * c.length / self.avg_len)
                    s += self.idf.get(t, 0) * tf * (k1 + 1) / norm
            if s > 0:
                scored.append((s, c))
        scored.sort(key=lambda x: -x[0])
        return [
            {
                "path": c.path,
                "section": c.heading,
                "score": round(s, 2),
                "snippet": c.text[:700] + (" ..." if len(c.text) > 700 else ""),
            }
            for s, c in scored[:k]
        ]

    def read(self, path: str, section: str | None = None, max_chars: int = 12000):
        text = self.files.get(path)
        if text is None:
            matches = [p for p in self.files if p.endswith(path) or path in p]
            if len(matches) != 1:
                return {
                    "error": f"no unique doc for {path!r}",
                    "candidates": matches[:10],
                }
            path, text = matches[0], self.files[matches[0]]
        if section:
            parts = [
                body
                for heading, body in _split_sections(path, text)
                if section.lower() in heading.lower()
            ]
            if not parts:
                headings = [h for h, _ in _split_sections(path, text) if h]
                return {
                    "error": f"no section {section!r} in {path}",
                    "sections": headings,
                }
            text = "\n\n".join(parts)
        return {
            "path": path,
            "section": section,
            "truncated": len(text) > max_chars,
            "text": text[:max_chars],
        }

    def list_recipes(self) -> list[dict]:
        out = []
        for path, text in self.files.items():
            if not path.startswith("recipes/"):
                continue
            title = next(
                (
                    ln.lstrip("# ").strip()
                    for ln in text.splitlines()
                    if ln.startswith("#")
                ),
                Path(path).stem,
            )
            out.append(
                {
                    "path": path,
                    "title": title,
                    "launch_commands": len(extract_launch_commands(text)),
                }
            )
        return sorted(out, key=lambda r: r["path"])


@lru_cache(maxsize=4)
def load_knowledge(root: Path | None) -> KnowledgeBase:
    return KnowledgeBase(root)


# --------------------------------------------------------------------------
# Recipe launch commands
# --------------------------------------------------------------------------

_ASSIGN = re.compile(r"^(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)=(.*)$")
# $VAR, ${VAR}, and ${VAR:-default} / ${VAR-default}.
_VAR_REF = re.compile(
    r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::?-([^}]*))?\}|\$([A-Za-z_][A-Za-z0-9_]*)"
)


def _unquote(value: str) -> str:
    try:
        parts = shlex.split(value, comments=True)
    except ValueError:
        return value.strip()
    return parts[0] if len(parts) == 1 else value.strip()


def _expand(token: str, variables: dict[str, str]) -> str:
    def sub(m: re.Match) -> str:
        name = m.group(1) or m.group(3)
        if name in variables:
            return variables[name]
        if m.group(2) is not None:  # the recipe's default
            return m.group(2)
        return m.group(0)

    return _VAR_REF.sub(sub, token)


def parse_launch_block(code: str) -> dict | None:
    """One shell block -> {env, args} for the ATOM server it starts."""
    joined = re.sub(r"\\\n\s*", " ", code)
    variables: dict[str, str] = {}
    exported: dict[str, str] = {}
    depth = 0
    for raw in joined.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if re.match(r"^(if|for|while|case)\b", line):
            depth += 1
            continue
        if re.match(r"^(fi|done|esac)\b", line):
            depth = max(depth - 1, 0)
            continue
        if SERVER_MODULE in line:
            try:
                tokens = shlex.split(line, comments=True)
            except ValueError:
                return None
            if SERVER_MODULE not in tokens:
                return None
            at = tokens.index(SERVER_MODULE)
            env = dict(exported)
            for tok in tokens[:at]:
                m = _ASSIGN.match(tok)
                if m:
                    env[m.group(1)] = m.group(2)
            args = [_expand(t, variables) for t in tokens[at + 1 :]]
            # Pipes / redirections end the command.
            for stop in ("|", ">", ">>", "2>&1", "&", "&&", ";"):
                if stop in args:
                    args = args[: args.index(stop)]
            env = {k: _expand(v, variables) for k, v in env.items()}
            return {"env": env, "args": args, "uses_docker": "docker" in tokens[:at]}
        if depth:
            continue  # a value set inside a branch is not the default
        m = _ASSIGN.match(line)
        if m:
            value = _expand(_unquote(m.group(2)), variables)
            variables[m.group(1)] = value
            if line.startswith("export "):
                exported[m.group(1)] = value
    return None


def extract_launch_commands(text: str, path: str = "") -> list[dict]:
    out = []
    heading = ""
    trail: list[str] = []
    pattern = r"^(#{1,4})\s+([^\n]*)$|^```[^\n]*\n(.*?)^```"
    for m in re.finditer(pattern, text, re.MULTILINE | re.DOTALL):
        if m.group(1):
            trail = trail[: len(m.group(1)) - 1] + [m.group(2).strip()]
            heading = " > ".join(trail)
            continue
        code = m.group(3) or ""
        if SERVER_MODULE not in code:
            continue
        parsed = parse_launch_block(code)
        if parsed:
            out.append({"recipe": path, "section": heading, **parsed})
    return out
