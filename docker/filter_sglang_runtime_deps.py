"""Write SGLang runtime_common deps, expanding extras and dropping transformers.

SGLang 0.5.20 lists transformers==5.12.1 on runtime_base. runtime_common only
pulls that extra in as sglang[runtime_base], so a prefix filter on the
runtime_common lines never sees transformers. Expand the extra, then skip it.
"""

from __future__ import annotations

import tomli
from pathlib import Path

BLOCKED_PREFIXES = (
    "compressed-tensors",
    "outlines==",
    "timm==",
    "torchao==",
    "xgrammar==",
    "transformers",
)


def _pkg_name(dep: str) -> str:
    name = dep.strip()
    for sep in ("[", "=", "<", ">", "!"):
        name = name.split(sep, 1)[0]
    return name.strip()


def _expand(extras: dict[str, list[str]], names: list[str], seen: set[str]) -> list[str]:
    out: list[str] = []
    for dep in names:
        key = dep.strip()
        if not key or key in seen:
            continue
        seen.add(key)
        if key.startswith("sglang[") and key.endswith("]"):
            extra = key[len("sglang[") : -1]
            out.extend(_expand(extras, extras[extra], seen))
            continue
        out.append(key)
    return out


def main() -> None:
    data = tomli.loads(Path("pyproject_other.toml").read_text())
    extras = data["project"]["optional-dependencies"]
    deps = _expand(extras, extras["runtime_common"], set())
    kept = []
    for dep in deps:
        name = _pkg_name(dep)
        if name == "numpy":
            continue
        if any(dep.startswith(prefix) or name == prefix.split("=", 1)[0] for prefix in BLOCKED_PREFIXES):
            continue
        kept.append(dep)
    Path("/tmp/sglang-runtime-common.txt").write_text("".join(f"{dep}\n" for dep in kept))


if __name__ == "__main__":
    main()
