# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""What is inside an ATOM image: versions, flags, env names, docs and skills.

The recipes in the checkout the agent runs from can be newer (or older) than
the ATOM in the image that will serve, and flags, environment variables and
skills drift with the code. So everything version-sensitive is read from the
image itself -- with ``cat``/``tar``/``grep``, never by importing torch -- and
cached per image id under ``<state_dir>/images/<id>/``:

* ``info.json``  -- ATOM/AITER commits, ROCm and torch versions, image id
* ``atom/``      -- the image's recipes, docs, .claude (skills + commands),
                    README and the two argparse sources
* ``aiter/``     -- the image's AITER .claude (skills)
* ``env_names``  -- every ATOM_* / AITER_* name the image's Python mentions
"""

from __future__ import annotations

import io
import json
import re
import shutil
import subprocess
import tarfile
from dataclasses import dataclass, field
from pathlib import Path

from atom.agent import planner

_INFO_PROBE = r"""
spec() { python3 -c "import importlib.util as u; s=u.find_spec('$1'); print(s.submodule_search_locations[0] if s and s.submodule_search_locations else '')" 2>/dev/null; }
A=$(spec atom); B=$(spec aiter)
AR=$(dirname "$A" 2>/dev/null); BR=$(dirname "$B" 2>/dev/null)
echo "atom_root=$AR"
echo "aiter_root=$BR"
echo "atom_commit=$( [ -n "$AR" ] && cd "$AR" && git log -1 --format='%h %cs' 2>/dev/null)"
echo "aiter_commit=$( [ -n "$BR" ] && cd "$BR" && git log -1 --format='%h %cs' 2>/dev/null)"
echo "rocm=$(cat /opt/rocm/.info/version 2>/dev/null)"
echo "torch=$(python3 -c 'import importlib.metadata as m; print(m.version("torch"))' 2>/dev/null)"
echo "=====ENV-NAMES====="
grep -rhoE '\b(ATOM|AITER)_[A-Z0-9_]{2,}\b' ${A:+"$A"} ${B:+"$B"} --include=*.py 2>/dev/null | sort -u
"""

# Bumped when info.json gains fields; older caches are re-probed.
CACHE_FORMAT = 2

# Paths copied out of the image, relative to the ATOM / AITER repo roots.
ATOM_PATHS = (
    "recipes",
    "docs",
    ".claude",
    "README.md",
    "atom/model_engine/arg_utils.py",
    "atom/entrypoints/openai/api_server.py",
)
AITER_PATHS = (".claude",)


@dataclass
class ImageInfo:
    image: str
    image_id: str
    cache: Path
    atom_commit: str = ""
    aiter_commit: str = ""
    rocm: str = ""
    torch: str = ""
    env_names: set[str] = field(default_factory=set)
    # Repo roots inside the image: where skills' "bash scripts/..." run.
    atom_root: str = "/app/ATOM"
    aiter_root: str = ""

    @property
    def atom_dir(self) -> Path:
        return self.cache / "atom"

    @property
    def aiter_dir(self) -> Path:
        return self.cache / "aiter"

    def provenance(self) -> dict:
        return {
            "image": self.image,
            "image_id": self.image_id,
            "atom": self.atom_commit or None,
            "aiter": self.aiter_commit or None,
            "rocm": self.rocm or None,
            "torch": self.torch or None,
        }

    def flag_table(self) -> dict | None:
        texts = []
        for rel in ATOM_PATHS[4:]:
            f = self.atom_dir / rel
            if f.is_file():
                texts.append(f.read_text(errors="replace"))
        return planner.parse_flag_table(texts) if texts else None

    def describe(self) -> str:
        parts = [f"ATOM {self.atom_commit or '?'}"]
        if self.aiter_commit:
            parts.append(f"AITER {self.aiter_commit}")
        return ", ".join(parts) + f" in {self.image}"


def _run(cmd: list[str], timeout: float, binary: bool = False):
    try:
        return subprocess.run(
            cmd, capture_output=True, timeout=timeout, check=False, text=not binary
        )
    except (OSError, subprocess.SubprocessError):
        return None


def image_id(image: str) -> str | None:
    res = _run(["docker", "image", "inspect", "-f", "{{.Id}}", image], 20)
    if res is None or res.returncode != 0:
        return None
    return res.stdout.strip()


def _extract(image: str, root: str, paths: tuple[str, ...], dest: Path) -> None:
    """Copy ``root/<paths>`` out of the image into ``dest`` (missing paths skipped)."""
    script = (
        f'cd "{root}" 2>/dev/null || exit 0; '
        f"tar -cf - $(for p in {' '.join(paths)}; do [ -e \"$p\" ] && echo \"$p\"; done) "
        "2>/dev/null"
    )
    res = _run(
        ["docker", "run", "--rm", "--entrypoint", "sh", image, "-c", script],
        180,
        binary=True,
    )
    if res is None or res.returncode != 0 or not res.stdout:
        return
    dest.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(res.stdout)) as tar:
        safe = [
            m
            for m in tar.getmembers()
            if (m.isfile() or m.isdir())
            and not m.name.startswith(("/", ".."))
            and ".." not in Path(m.name).parts
        ]
        tar.extractall(dest, members=safe)


def probe(image: str, state_dir: Path) -> ImageInfo | None:
    """Read (or reuse the cached reading of) a local image. None if absent."""
    full_id = image_id(image)
    if not full_id:
        return None
    short = full_id.split(":")[-1][:16]
    cache = Path(state_dir) / "images" / short
    info_file = cache / "info.json"
    data = json.loads(info_file.read_text()) if info_file.is_file() else {}
    if data.get("format") == CACHE_FORMAT:
        return ImageInfo(
            image=image,
            image_id=full_id,
            cache=cache,
            atom_commit=data.get("atom_commit", ""),
            aiter_commit=data.get("aiter_commit", ""),
            rocm=data.get("rocm", ""),
            torch=data.get("torch", ""),
            env_names=set(data.get("env_names", [])),
            atom_root=data.get("atom_root") or "/app/ATOM",
            aiter_root=data.get("aiter_root", ""),
        )
    res = _run(
        ["docker", "run", "--rm", "--entrypoint", "sh", image, "-c", _INFO_PROBE], 120
    )
    if res is None or res.returncode != 0:
        return None
    head, _, env_block = res.stdout.partition("=====ENV-NAMES=====")
    kv = dict(
        line.split("=", 1) for line in head.splitlines() if "=" in line and line.strip()
    )
    tmp = cache.with_name(cache.name + ".partial")
    shutil.rmtree(tmp, ignore_errors=True)
    if kv.get("atom_root"):
        _extract(image, kv["atom_root"], ATOM_PATHS, tmp / "atom")
    if kv.get("aiter_root"):
        _extract(image, kv["aiter_root"], AITER_PATHS, tmp / "aiter")
    data = {
        "format": CACHE_FORMAT,
        "image": image,
        "image_id": full_id,
        "atom_root": kv.get("atom_root", "").strip(),
        "aiter_root": kv.get("aiter_root", "").strip(),
        "atom_commit": kv.get("atom_commit", "").strip(),
        "aiter_commit": kv.get("aiter_commit", "").strip(),
        "rocm": kv.get("rocm", "").strip(),
        "torch": kv.get("torch", "").strip(),
        "env_names": sorted({x for x in env_block.split() if x}),
    }
    tmp.mkdir(parents=True, exist_ok=True)
    (tmp / "info.json").write_text(json.dumps(data, indent=1))
    shutil.rmtree(cache, ignore_errors=True)
    tmp.rename(cache)  # complete or absent, never half-written
    return probe(image, state_dir)


def unknown_env(env: dict[str, str], known: set[str]) -> list[str]:
    """ATOM_*/AITER_* names in a plan that the image's code never mentions."""
    if not known:
        return []
    return sorted(
        k for k in env if k.startswith(("ATOM_", "AITER_")) and k not in known
    )


def _git_head(path: Path) -> str:
    try:
        res = subprocess.run(
            ["git", "-C", str(path), "log", "-1", "--format=%h %cs"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    return res.stdout.strip() if res.returncode == 0 else ""


def _package_dir(name: str) -> Path | None:
    import importlib.util

    spec = importlib.util.find_spec(name)
    if spec is None or not spec.submodule_search_locations:
        return None
    return Path(next(iter(spec.submodule_search_locations)))


def local_provenance() -> dict:
    """Versions of the ATOM/AITER this Python would run (runtime=local)."""
    atom_dir, aiter_dir = _package_dir("atom"), _package_dir("aiter")
    rocm = Path("/opt/rocm/.info/version")
    return {
        "image": None,
        "atom": _git_head(atom_dir.parent) if atom_dir else None,
        "aiter": _git_head(aiter_dir.parent) if aiter_dir else None,
        "rocm": rocm.read_text().strip() if rocm.is_file() else None,
    }


_ENV_RE = re.compile(r"\b(?:ATOM|AITER)_[A-Z0-9_]{2,}\b")
_local_env: set[str] | None = None


def local_env_names() -> set[str]:
    """ATOM_*/AITER_* names mentioned by this Python's ATOM and AITER."""
    global _local_env
    if _local_env is None:
        names: set[str] = set()
        for pkg in ("atom", "aiter"):
            root = _package_dir(pkg)
            if root is None:
                continue
            for f in root.rglob("*.py"):
                try:
                    names.update(_ENV_RE.findall(f.read_text(errors="replace")))
                except OSError:
                    continue
        _local_env = names
    return _local_env
