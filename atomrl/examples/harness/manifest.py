# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""Load and validate the ``examples.yaml`` descriptions.

A description file has three top-level keys, all optional:

``images``
    Container images by key: the reference to pull, where the AtomRL checkout
    is bind-mounted inside, and the environment the image expects.
``defaults``
    Fields merged into every example of the file (examples override them).
``examples``
    The examples themselves. Each has a ``smoke`` variant (what regression
    runs use) and optionally a ``full`` variant (the long run it is cut from).

Unknown keys are errors, so a typo cannot silently drop a check.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

import yaml

EXAMPLES_DIR = Path(__file__).resolve().parents[1]
ATOMRL_DIR = EXAMPLES_DIR.parent

TIERS = ("t0", "t1", "t2", "long")
DEFAULT_FORBID = ("Traceback", "OutOfMemory", "CUDA error", "HSA_STATUS")
STATUSES = ("validated", "unvalidated")
LOG_MODES = ("file", "stdout")


class ManifestError(ValueError):
    pass


@dataclasses.dataclass(frozen=True)
class MetricCheck:
    """One metric read from the log line of ``Checks.step``.

    Exactly one judgement applies: ``rtol`` (relative deviation from ``ref``),
    ``band`` (|value| within a factor of ``ref``), ``min``/``max`` bounds, or
    ``info`` (recorded, never judged).
    """

    name: str
    ref: float | None = None
    rtol: float | None = None
    band: float | None = None
    min: float | None = None
    max: float | None = None
    info: bool = False


@dataclasses.dataclass(frozen=True)
class Checks:
    step: int = 1
    # Several lines can start with "step=N "; the metrics dump is the one
    # containing this marker.
    marker: str = "rollout_corr/kl="
    metrics: tuple[MetricCheck, ...] = ()
    forbid: tuple[str, ...] = DEFAULT_FORBID
    # The run must log the metrics line of its final step.
    last_step: bool = True
    # Every weight-sync bucket must report skipped=0: a bucket that skips
    # tensors leaves the run green while the rollout serves stale weights.
    weight_sync_buckets: bool = True
    # Judge a pytest session (summary line, no failures) instead of metrics.
    pytest: bool = False


@dataclasses.dataclass(frozen=True)
class Variant:
    config: str | None  # relative to atomrl/
    steps: int | None
    env: dict[str, str]
    overrides: tuple[str, ...]
    timeout_min: int
    stall_min: int
    checks: Checks


@dataclasses.dataclass(frozen=True)
class Image:
    key: str
    ref: str
    atomrl_mount: str
    atom_mount: str | None = None
    env: dict[str, str] = dataclasses.field(default_factory=dict)
    build: str | None = None
    # Extra `docker run` arguments, e.g. to clear an entrypoint.
    run_args: tuple[str, ...] = ()


@dataclasses.dataclass(frozen=True)
class Example:
    id: str
    title: str
    family: str
    tier: str
    aliases: tuple[str, ...]
    nodes: int
    gpus_per_node: int
    image: str
    command: str
    log_mode: str
    env: dict[str, str]
    requires: tuple[str, ...]
    needs: tuple[str, ...]
    prepare: str | None
    compile_cache_key: str | None
    status: str
    notes: str
    # A known defect: failures are reported as XFAIL, a pass as XPASS.
    known_issue: str | None
    smoke: Variant
    full: Variant | None
    source: Path

    def variant(self, full: bool) -> Variant:
        if full:
            if self.full is None:
                raise ManifestError(f"{self.id} has no full variant")
            return self.full
        return self.smoke


@dataclasses.dataclass(frozen=True)
class Catalog:
    examples: tuple[Example, ...]
    images: dict[str, Image]

    def get(self, name: str) -> Example:
        for ex in self.examples:
            if name == ex.id or name in ex.aliases:
                return ex
        raise ManifestError(f"no example with id or alias {name!r}; see `run.sh list`")


_EXAMPLE_KEYS = {
    "id", "title", "tier", "aliases", "nodes", "gpus_per_node", "image", "command",
    "log_mode", "env", "requires", "needs", "prepare", "compile_cache_key", "status",
    "notes", "known_issue", "smoke", "full",
}  # fmt: skip
_VARIANT_KEYS = {
    "config",
    "steps",
    "env",
    "overrides",
    "timeout_min",
    "stall_min",
    "checks",
}
_CHECK_KEYS = {
    "step", "marker", "metrics", "forbid", "last_step", "weight_sync_buckets", "pytest"
}  # fmt: skip
_METRIC_KEYS = {f.name for f in dataclasses.fields(MetricCheck)}
_IMAGE_KEYS = {"ref", "atomrl_mount", "atom_mount", "env", "build", "run_args"}
NEEDS = {"atom_rdma", "rdma"}


def _unknown(where: str, data: dict, allowed: set[str]) -> None:
    extra = set(data) - allowed
    if extra:
        raise ManifestError(f"{where}: unknown key(s) {sorted(extra)}")


def _str_env(where: str, env: Any) -> dict[str, str]:
    if env is None:
        return {}
    if not isinstance(env, dict):
        raise ManifestError(f"{where}: env must be a mapping")
    return {str(k): "" if v is None else str(v) for k, v in env.items()}


def _metric(where: str, data: dict) -> MetricCheck:
    _unknown(where, data, _METRIC_KEYS)
    if "name" not in data:
        raise ManifestError(f"{where}: metric without a name")
    m = MetricCheck(**data)
    kinds = [
        m.rtol is not None,
        m.band is not None,
        m.min is not None or m.max is not None,
        m.info,
    ]
    if sum(kinds) != 1:
        raise ManifestError(
            f"{where}: metric {m.name!r} needs exactly one of rtol, band, min/max, info"
        )
    if (m.rtol is not None or m.band is not None) and m.ref is None:
        raise ManifestError(f"{where}: metric {m.name!r} needs a ref")
    if m.band is not None and m.band <= 1:
        raise ManifestError(f"{where}: metric {m.name!r} band must be > 1")
    return m


def _checks(where: str, data: dict | None) -> Checks:
    data = dict(data or {})
    _unknown(where, data, _CHECK_KEYS)
    metrics = tuple(
        _metric(f"{where}.metrics[{i}]", m)
        for i, m in enumerate(data.pop("metrics", []))
    )
    forbid = tuple(data.pop("forbid", DEFAULT_FORBID))
    return Checks(metrics=metrics, forbid=forbid, **data)


def _variant(
    where: str, data: dict, base_dir: Path, base_env: dict[str, str], atomrl_dir: Path
) -> Variant:
    _unknown(where, data, _VARIANT_KEYS)
    config = data.get("config")
    if config is not None:
        path = (base_dir / config).resolve()
        if not path.is_file():
            raise ManifestError(f"{where}: config {config} does not exist ({path})")
        config = path.relative_to(atomrl_dir).as_posix()
    steps = data.get("steps")
    if steps is not None and (not isinstance(steps, int) or steps < 1):
        raise ManifestError(f"{where}: steps must be a positive integer")
    overrides = data.get("overrides") or []
    if not isinstance(overrides, list):
        raise ManifestError(f"{where}: overrides must be a list")
    return Variant(
        config=config,
        steps=steps,
        env={**base_env, **_str_env(where, data.get("env"))},
        overrides=tuple(str(o) for o in overrides),
        timeout_min=int(data.get("timeout_min", 60)),
        stall_min=int(data.get("stall_min", 40)),
        checks=_checks(f"{where}.checks", data.get("checks")),
    )


def _example(where: str, data: dict, defaults: dict, path: Path, root: Path) -> Example:
    merged = {**defaults, **data}
    merged["env"] = {**(defaults.get("env") or {}), **(data.get("env") or {})}
    for key in ("smoke", "full"):
        if key in defaults and key in data:
            merged[key] = {**defaults[key], **data[key]}
    _unknown(where, merged, _EXAMPLE_KEYS)
    for key in ("id", "title", "tier", "command", "smoke"):
        if key not in merged:
            raise ManifestError(f"{where}: missing {key!r}")
    where = f"{path.relative_to(root)}:{merged['id']}"
    if merged["tier"] not in TIERS:
        raise ManifestError(f"{where}: tier must be one of {TIERS}")
    status = merged.get("status", "unvalidated")
    if status not in STATUSES:
        raise ManifestError(f"{where}: status must be one of {STATUSES}")
    log_mode = merged.get("log_mode", "file")
    if log_mode not in LOG_MODES:
        raise ManifestError(f"{where}: log_mode must be one of {LOG_MODES}")
    needs = tuple(merged.get("needs") or ())
    if set(needs) - NEEDS:
        raise ManifestError(f"{where}: unknown needs {sorted(set(needs) - NEEDS)}")
    env = _str_env(where, merged["env"])
    base_dir = path.parent
    smoke = _variant(f"{where}.smoke", merged["smoke"], base_dir, env, root.parent)
    full = merged.get("full")
    full = _variant(f"{where}.full", full, base_dir, env, root.parent) if full else None
    family = path.parent.relative_to(root).as_posix()
    return Example(
        id=str(merged["id"]),
        title=str(merged["title"]),
        family="." if family == "." else family,
        tier=merged["tier"],
        aliases=tuple(str(a) for a in merged.get("aliases") or ()),
        nodes=int(merged.get("nodes", 1)),
        gpus_per_node=int(merged.get("gpus_per_node", 8)),
        image=str(merged.get("image", "release")),
        command=str(merged["command"]).strip(),
        log_mode=log_mode,
        env=env,
        requires=tuple(merged.get("requires") or ()),
        needs=needs,
        prepare=(str(merged["prepare"]).strip() if merged.get("prepare") else None),
        compile_cache_key=merged.get("compile_cache_key"),
        status=status,
        notes=str(merged.get("notes", "")).strip(),
        known_issue=(
            str(merged["known_issue"]).strip() if merged.get("known_issue") else None
        ),
        smoke=smoke,
        full=full,
        source=path,
    )


def _image(where: str, key: str, data: dict) -> Image:
    _unknown(where, data, _IMAGE_KEYS)
    for k in ("ref", "atomrl_mount"):
        if k not in data:
            raise ManifestError(f"{where}: image {key!r} needs {k!r}")
    return Image(
        key=key,
        ref=str(data["ref"]),
        atomrl_mount=str(data["atomrl_mount"]),
        atom_mount=data.get("atom_mount"),
        env=_str_env(where, data.get("env")),
        build=data.get("build"),
        run_args=tuple(str(a) for a in data.get("run_args") or ()),
    )


def manifest_paths(root: Path = EXAMPLES_DIR) -> list[Path]:
    return sorted(p for p in root.rglob("examples.yaml") if "harness" not in p.parts)


def load(root: Path = EXAMPLES_DIR) -> Catalog:
    examples: list[Example] = []
    images: dict[str, Image] = {}
    for path in manifest_paths(root):
        rel = path.relative_to(root)
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        _unknown(str(rel), data, {"images", "defaults", "examples"})
        for key, spec in (data.get("images") or {}).items():
            if key in images:
                raise ManifestError(f"{rel}: image {key!r} is defined twice")
            images[key] = _image(str(rel), key, spec)
        defaults = data.get("defaults") or {}
        for i, item in enumerate(data.get("examples") or []):
            examples.append(_example(f"{rel}[{i}]", item, defaults, path, root))
    seen: dict[str, str] = {}
    for ex in examples:
        for name in (ex.id, *ex.aliases):
            if name in seen:
                raise ManifestError(
                    f"{name!r} is used by both {seen[name]} and {ex.id}"
                )
            seen[name] = ex.id
        if ex.image not in images:
            raise ManifestError(f"{ex.id}: unknown image {ex.image!r}")
    return Catalog(examples=tuple(examples), images=images)
