# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

###############################################################################
#
# See LICENSE for license information.
###############################################################################

"""Central registry for Lumen patches."""

from __future__ import annotations

import inspect
import logging
import typing
from collections import deque
from collections.abc import Callable, Iterable
from typing import Any

from atomrl.lumen.patches.types import PatchPhase, PatchResult, PatchSpec

logger = logging.getLogger(__name__)


class PatchRegistry:
    """Global store of :class:`PatchSpec` entries."""

    _patches: typing.ClassVar[dict[str, PatchSpec]] = {}

    @classmethod
    def register(cls, spec: PatchSpec) -> PatchSpec:
        if spec.name in cls._patches:
            raise ValueError(f"Patch {spec.name!r} is already registered")
        cls._patches[spec.name] = spec
        return spec

    @classmethod
    def get(cls, name: str) -> PatchSpec:
        try:
            return cls._patches[name]
        except KeyError as exc:
            raise KeyError(f"Unknown patch {name!r}") from exc

    @classmethod
    def all(cls) -> tuple[PatchSpec, ...]:
        return tuple(cls._patches.values())

    @classmethod
    def clear(cls) -> None:
        """Remove all registrations (testing only)."""
        cls._patches.clear()


def register_patch(
    name: str,
    phase: PatchPhase,
    *,
    description: str = "",
    enabled: Callable[[], bool] | None = None,
    depends_on: tuple[str, ...] = (),
    tags: frozenset[str] = frozenset(),
    config_fields: tuple[str, ...] = (),
    default: bool = True,
):
    """Decorator that registers a patch function."""

    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        PatchRegistry.register(
            PatchSpec(
                name=name,
                phase=phase,
                fn=fn,
                description=description,
                enabled=enabled or (lambda: True),
                depends_on=depends_on,
                tags=tags,
                config_fields=config_fields,
                default=default,
            )
        )
        return fn

    return decorator


def _tags_match(
    patch_tags: frozenset[str],
    filter_tags: set[str],
    tag_mode: str,
) -> bool:
    """Return whether *patch_tags* matches *filter_tags* under *tag_mode*."""
    if tag_mode == "all":
        return filter_tags.issubset(patch_tags)
    if tag_mode == "any":
        return bool(filter_tags & patch_tags)
    raise ValueError(f"Unknown tag_mode {tag_mode!r} (expected 'all' or 'any')")


def _filter_specs(
    phase: PatchPhase,
    *,
    tags: set[str] | None = None,
    tag_mode: str = "all",
    names: set[str] | None = None,
    default_only: bool = False,
) -> list[PatchSpec]:
    specs: list[PatchSpec] = []
    for spec in PatchRegistry.all():
        if spec.phase is not phase:
            continue
        if names is not None and spec.name not in names:
            continue
        if default_only and not spec.default:
            continue
        if tags is not None and not _tags_match(spec.tags, tags, tag_mode):
            continue
        specs.append(spec)
    return specs


def _topological_sort(specs: Iterable[PatchSpec]) -> list[PatchSpec]:
    by_name = {spec.name: spec for spec in specs}
    in_degree = {name: 0 for name in by_name}
    dependents: dict[str, list[str]] = {name: [] for name in by_name}

    for spec in specs:
        for dep in spec.depends_on:
            if dep not in by_name:
                raise ValueError(
                    f"Patch {spec.name!r} depends on unknown patch {dep!r}"
                )
            in_degree[spec.name] += 1
            dependents[dep].append(spec.name)

    queue = deque(name for name, degree in in_degree.items() if degree == 0)
    ordered: list[PatchSpec] = []

    while queue:
        name = queue.popleft()
        ordered.append(by_name[name])
        for dependent in dependents[name]:
            in_degree[dependent] -= 1
            if in_degree[dependent] == 0:
                queue.append(dependent)

    if len(ordered) != len(by_name):
        remaining = [name for name, degree in in_degree.items() if degree > 0]
        raise ValueError(f"Patch dependency cycle involving: {remaining}")

    return ordered


def list_patches(
    phase: PatchPhase | None = None,
    *,
    tags: set[str] | None = None,
    tag_mode: str = "all",
) -> list[PatchSpec]:
    """Return registered patches, optionally filtered."""
    specs = list(PatchRegistry.all())
    if phase is not None:
        specs = [spec for spec in specs if spec.phase is phase]
    if tags is not None:
        specs = [spec for spec in specs if _tags_match(spec.tags, tags, tag_mode)]
    return sorted(specs, key=lambda spec: (spec.phase.value, spec.name))


def apply_patches(
    phase: PatchPhase,
    *,
    tags: set[str] | None = None,
    tag_mode: str = "all",
    names: set[str] | None = None,
    default_only: bool = False,
    include_disabled: bool = False,
    dry_run: bool = False,
    **kwargs: Any,
) -> dict[str, PatchResult]:
    """Apply registered patches for *phase* in dependency order."""
    specs = _filter_specs(
        phase,
        tags=tags,
        tag_mode=tag_mode,
        names=names,
        default_only=default_only,
    )
    results: dict[str, PatchResult] = {}

    for spec in _topological_sort(specs):
        if not include_disabled and not spec.enabled():
            results[spec.name] = PatchResult(
                name=spec.name,
                skipped=True,
                reason="disabled",
            )
            continue

        if dry_run:
            results[spec.name] = PatchResult(
                name=spec.name,
                skipped=True,
                reason="dry-run",
            )
            continue

        try:
            sig = inspect.signature(spec.fn)
            bound_kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters}
            has_var_keyword = any(
                p.kind is inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
            )
            return_value = spec.fn(**(kwargs if has_var_keyword else bound_kwargs))
        except Exception:
            logger.exception("Patch %s failed", spec.name)
            raise

        results[spec.name] = PatchResult(
            name=spec.name,
            applied=True,
            return_value=return_value,
        )
        logger.debug("Applied patch %s (phase=%s)", spec.name, phase.name)

    return results
