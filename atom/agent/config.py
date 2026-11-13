# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Agent settings shared by the tools, the chat loop and the MCP server."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# Not under ~/.cache/atom: scripts/start_atom_server.sh clears that directory
# on every launch, which would forget every deployment the agent started.
DEFAULT_STATE_DIR = Path(
    os.environ.get("ATOM_AGENT_HOME")
    or Path(os.environ.get("XDG_STATE_HOME", Path.home() / ".local" / "state"))
    / "atom-agent"
)

# Where checkpoints usually live on ROCm boxes. Hugging Face caches are added
# at runtime from HF_HOME / HF_HUB_CACHE.
DEFAULT_MODEL_ROOTS = ("/models", "/data", "/raid", "/mnt", str(Path.home()))

DEFAULT_IMAGE = "rocm/atom:latest"


def repo_root() -> Path | None:
    """The ATOM checkout this package runs from, if it ships the recipes."""
    env = os.environ.get("ATOM_AGENT_DOCS")
    if env:
        return Path(env)
    root = Path(__file__).resolve().parents[2]
    return root if (root / "recipes").is_dir() else None


@dataclass
class AgentConfig:
    state_dir: Path = DEFAULT_STATE_DIR
    docs_root: Path | None = field(default_factory=repo_root)
    model_roots: tuple[str, ...] = DEFAULT_MODEL_ROOTS
    # "auto" picks local when this Python can run ATOM, docker otherwise.
    runtime: str = "auto"
    image: str | None = None
    # Machine-wide GPU/port leases shared by every user's agent.
    lease_dir: Path | None = None
    # Extra skill directories (each laid out like a repo's .claude/).
    skill_dirs: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        self.state_dir = Path(self.state_dir)
        if self.lease_dir is None:
            from atom.agent.leases import lease_dir

            self.lease_dir = lease_dir()
        self.state_dir.mkdir(parents=True, exist_ok=True)
        (self.state_dir / "logs").mkdir(exist_ok=True)

    @property
    def logs_dir(self) -> Path:
        return self.state_dir / "logs"
