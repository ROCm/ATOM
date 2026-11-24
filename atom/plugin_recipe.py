# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Plugin recipes: declarative module/seam routing and engine/model options.

Supports both ``atom.plugin_recipe/1`` (seam tables) and ``atom.plugin_recipe/2``
(hierarchical ``nn.Module`` subtree takeover + declarative ``engine``,
``model_options``, and ``dspark_options`` blocks so deployments need no
proliferation of ``ATOM_*`` environment variables).

Graph modes: ``any`` (default) serves every pass; ``eager`` serves uncaptured
passes; ``capture`` with ``buckets`` serves captured passes whose padded rows
match one of the buckets. ``python -m atom.plugin_recipe PATH`` validates and
dumps a recipe or recipe directory on CPU.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path
from types import MappingProxyType
from typing import Any

from atom import plugin_host

RECIPE_SCHEMA = "atom.plugin_recipe/1"
RECIPE_SCHEMA_V2 = "atom.plugin_recipe/2"
SCHEMAS = (RECIPE_SCHEMA, RECIPE_SCHEMA_V2)
MODES = ("any", "eager", "capture")
ON_DECLINE = ("native", "error")
SHADOW = ("error", "shadow", "nested")
MONO_MODES = ("auto", "off")
_SLOT_RE = re.compile(r"^[A-Za-z0-9_*]+(\.[A-Za-z0-9_*]+)*(:[A-Za-z_][A-Za-z0-9_]*)?$")


class RecipeRefused(ValueError):
    """A recipe that can never be valid here; ``reason`` is its short code."""

    def __init__(self, reason: str, detail: str):
        super().__init__(f"{reason}: {detail}")
        self.reason = reason


def _strict_bool(value: Any, where: str) -> bool:
    if not isinstance(value, bool):
        raise RecipeRefused("recipe-schema", f"{where} must be a bool")
    return value


def _strict_int(value: Any, where: str, *, min_value: int | None = None) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise RecipeRefused("recipe-schema", f"{where} must be an int")
    if min_value is not None and value < min_value:
        raise RecipeRefused("recipe-schema", f"{where} must be >= {min_value}")
    return value


def _opt_str(value: Any, where: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise RecipeRefused("recipe-schema", f"{where} must be a non-empty string")
    return value.strip()


def _freeze_placement(
    value: Any, where: str
) -> str | tuple[tuple[str, tuple[int, ...]], ...] | None:
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        return text or None
    if isinstance(value, tuple) and all(
        isinstance(item, tuple)
        and len(item) == 2
        and isinstance(item[0], str)
        and isinstance(item[1], tuple)
        for item in value
    ):
        return value
    if not isinstance(value, Mapping):
        raise RecipeRefused(
            "recipe-schema",
            f"{where} must be a JSON path string, a layer->perm mapping, or null",
        )
    items: list[tuple[str, tuple[int, ...]]] = []
    for layer_key, perm in sorted(value.items(), key=lambda kv: str(kv[0])):
        if not isinstance(perm, (list, tuple)) or not all(
            isinstance(x, int) and not isinstance(x, bool) for x in perm
        ):
            raise RecipeRefused(
                "recipe-schema", f"{where}[{layer_key!r}] must be an int permutation"
            )
        items.append((str(layer_key), tuple(int(x) for x in perm)))
    return tuple(items)


@dataclass(frozen=True)
class EngineOptions:
    """Scheduler and DP-routing options configured from a recipe (PR #3644)."""

    prefill_short_first_rows: int = 0
    prefill_short_first_long_rows: int = 4096
    prefill_short_first_max_skips: int = 32
    prefill_cost_ceil: bool = False
    prefill_decode_interval_busy: int = 0
    prefill_decode_interval_busy_seqs: int = 48
    prefill_long_cap_busy: int = 0
    prefill_long_cap_busy_seqs: int = 48
    dp_session_prefix_owner_tokens: int = 0
    dp_session_prefix_owner_max_extra_reqs: int = 8

    @classmethod
    def from_mapping(cls, raw: Any) -> EngineOptions:
        if raw is None:
            return cls()
        if not isinstance(raw, Mapping):
            raise RecipeRefused("recipe-schema", "engine must be an object")
        known = {f.name for f in fields(cls)}
        unknown = sorted(set(raw) - known)
        if unknown:
            raise RecipeRefused("recipe-schema", f"unknown engine option(s): {unknown}")
        defaults = cls()
        return cls(
            prefill_short_first_rows=_strict_int(
                raw.get("prefill_short_first_rows", defaults.prefill_short_first_rows),
                "engine.prefill_short_first_rows",
                min_value=0,
            ),
            prefill_short_first_long_rows=_strict_int(
                raw.get(
                    "prefill_short_first_long_rows",
                    defaults.prefill_short_first_long_rows,
                ),
                "engine.prefill_short_first_long_rows",
                min_value=1,
            ),
            prefill_short_first_max_skips=_strict_int(
                raw.get(
                    "prefill_short_first_max_skips",
                    defaults.prefill_short_first_max_skips,
                ),
                "engine.prefill_short_first_max_skips",
                min_value=0,
            ),
            prefill_cost_ceil=_strict_bool(
                raw.get("prefill_cost_ceil", defaults.prefill_cost_ceil),
                "engine.prefill_cost_ceil",
            ),
            prefill_decode_interval_busy=_strict_int(
                raw.get(
                    "prefill_decode_interval_busy",
                    defaults.prefill_decode_interval_busy,
                ),
                "engine.prefill_decode_interval_busy",
                min_value=0,
            ),
            prefill_decode_interval_busy_seqs=_strict_int(
                raw.get(
                    "prefill_decode_interval_busy_seqs",
                    defaults.prefill_decode_interval_busy_seqs,
                ),
                "engine.prefill_decode_interval_busy_seqs",
                min_value=1,
            ),
            prefill_long_cap_busy=_strict_int(
                raw.get("prefill_long_cap_busy", defaults.prefill_long_cap_busy),
                "engine.prefill_long_cap_busy",
                min_value=0,
            ),
            prefill_long_cap_busy_seqs=_strict_int(
                raw.get(
                    "prefill_long_cap_busy_seqs", defaults.prefill_long_cap_busy_seqs
                ),
                "engine.prefill_long_cap_busy_seqs",
                min_value=1,
            ),
            dp_session_prefix_owner_tokens=_strict_int(
                raw.get(
                    "dp_session_prefix_owner_tokens",
                    defaults.dp_session_prefix_owner_tokens,
                ),
                "engine.dp_session_prefix_owner_tokens",
                min_value=0,
            ),
            dp_session_prefix_owner_max_extra_reqs=_strict_int(
                raw.get(
                    "dp_session_prefix_owner_max_extra_reqs",
                    defaults.dp_session_prefix_owner_max_extra_reqs,
                ),
                "engine.dp_session_prefix_owner_max_extra_reqs",
                min_value=0,
            ),
        )

    def dump(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ModelOptions:
    """Target-model options configured from a recipe (PRs #3529, #3645)."""

    target_model_cls: str | None = None
    draft_model_cls: str | None = None
    ffn_mhc_fuse_quant: bool = False
    shared_act_quant: bool = False
    fused_attn: bool = False
    fused_attn_min_rows: int = 0
    maxima_grid: int = 0
    expert_placement: str | tuple[tuple[str, tuple[int, ...]], ...] | None = None
    block_transition: bool = False
    mono_mode: str = "auto"

    @classmethod
    def from_mapping(cls, raw: Any) -> ModelOptions:
        if raw is None:
            return cls()
        if not isinstance(raw, Mapping):
            raise RecipeRefused("recipe-schema", "model_options must be an object")
        known = {f.name for f in fields(cls)}
        unknown = sorted(set(raw) - known)
        if unknown:
            raise RecipeRefused(
                "recipe-schema", f"unknown model_options field(s): {unknown}"
            )
        defaults = cls()
        mono_mode = raw.get("mono_mode", defaults.mono_mode)
        if mono_mode not in MONO_MODES:
            raise RecipeRefused(
                "recipe-schema",
                f"model_options.mono_mode must be one of {MONO_MODES}",
            )
        return cls(
            target_model_cls=_opt_str(
                raw.get("target_model_cls", defaults.target_model_cls),
                "model_options.target_model_cls",
            ),
            draft_model_cls=_opt_str(
                raw.get("draft_model_cls", defaults.draft_model_cls),
                "model_options.draft_model_cls",
            ),
            ffn_mhc_fuse_quant=_strict_bool(
                raw.get("ffn_mhc_fuse_quant", defaults.ffn_mhc_fuse_quant),
                "model_options.ffn_mhc_fuse_quant",
            ),
            shared_act_quant=_strict_bool(
                raw.get("shared_act_quant", defaults.shared_act_quant),
                "model_options.shared_act_quant",
            ),
            fused_attn=_strict_bool(
                raw.get("fused_attn", defaults.fused_attn),
                "model_options.fused_attn",
            ),
            fused_attn_min_rows=_strict_int(
                raw.get("fused_attn_min_rows", defaults.fused_attn_min_rows),
                "model_options.fused_attn_min_rows",
                min_value=0,
            ),
            maxima_grid=_strict_int(
                raw.get("maxima_grid", defaults.maxima_grid),
                "model_options.maxima_grid",
                min_value=0,
            ),
            expert_placement=_freeze_placement(
                raw.get("expert_placement", defaults.expert_placement),
                "model_options.expert_placement",
            ),
            block_transition=_strict_bool(
                raw.get("block_transition", defaults.block_transition),
                "model_options.block_transition",
            ),
            mono_mode=mono_mode,
        )

    def placement_spec(self) -> str | dict[str, list[int]] | None:
        if self.expert_placement is None or isinstance(self.expert_placement, str):
            return self.expert_placement
        return {k: list(v) for k, v in self.expert_placement}

    def dump(self) -> dict[str, Any]:
        data = asdict(self)
        data["expert_placement"] = self.placement_spec()
        return data


@dataclass(frozen=True)
class DSparkOptions:
    """DSpark speculative-decode options configured from a recipe (PRs #3639, #3645)."""

    fused_prologue: bool = False
    two_buffer_attn: bool = False
    lean_head: bool = False
    verify_argmax_once: bool = False

    @classmethod
    def from_mapping(cls, raw: Any) -> DSparkOptions:
        if raw is None:
            return cls()
        if not isinstance(raw, Mapping):
            raise RecipeRefused("recipe-schema", "dspark_options must be an object")
        known = {f.name for f in fields(cls)}
        unknown = sorted(set(raw) - known)
        if unknown:
            raise RecipeRefused(
                "recipe-schema", f"unknown dspark_options field(s): {unknown}"
            )
        defaults = cls()
        return cls(
            fused_prologue=_strict_bool(
                raw.get("fused_prologue", defaults.fused_prologue),
                "dspark_options.fused_prologue",
            ),
            two_buffer_attn=_strict_bool(
                raw.get("two_buffer_attn", defaults.two_buffer_attn),
                "dspark_options.two_buffer_attn",
            ),
            lean_head=_strict_bool(
                raw.get("lean_head", defaults.lean_head),
                "dspark_options.lean_head",
            ),
            verify_argmax_once=_strict_bool(
                raw.get("verify_argmax_once", defaults.verify_argmax_once),
                "dspark_options.verify_argmax_once",
            ),
        )

    def dump(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Plan:
    """What the host runs on: module/seam tables and options, frozen before build.

    ``tables`` keys are ``"any"``, ``"eager"`` or a capture bucket (int) for
    seam routing; ``modules`` uses the same keys for ``nn.Module`` slot paths.
    A table of ``None`` (the ``--plugin-enable`` shorthand) means "whatever the
    enabled plugins register, in ``providers`` order".
    """

    plan_id: str
    strategy: str
    model_arch: str | None
    tables: Mapping[Any, Mapping[str, tuple[str, ...]] | None]
    providers: tuple[str, ...]
    aiter: tuple[str, ...]
    arms: Mapping[str, tuple[str, ...]]
    shadowed: tuple[tuple[str, str, str], ...] = ()
    # How a nested / covered owner shadowed by an outer one is treated, checked
    # against the built model (``plugin_host.check_plan``): nested (record),
    # error (refuse).
    shadow_policy: str = "nested"
    # Seams the recipe plans with no owner ([]): native by decision, which
    # counts as planned for a taken-over module's inner seams.
    native_seams: tuple[str, ...] = ()
    strict_overlap: bool = False
    on_decline: str = "native"
    expect: Mapping[str, Any] = MappingProxyType({})
    requires: Mapping[str, Any] = MappingProxyType({})
    recipe_digests: tuple[str, ...] = ()
    modules: Mapping[Any, Mapping[str, tuple[str, ...]] | None] = MappingProxyType({})
    engine: EngineOptions = EngineOptions()
    model_options: ModelOptions = ModelOptions()
    dspark_options: DSparkOptions = DSparkOptions()
    schema_version: int = 1
    # Process environment a v2 recipe sets for the server it configures
    # (`apply_env`): native ATOM / AITER switches that are not recipe options.
    env: Mapping[str, str] = MappingProxyType({})
    # Options for plugin packages, by name (a provider or a model package):
    # ATOM passes them through unread (``plugin_host.plugin_options``).
    plugin_options: Mapping[str, Any] = MappingProxyType({})

    @property
    def has_options(self) -> bool:
        return (
            self.engine != EngineOptions()
            or self.model_options != ModelOptions()
            or self.dspark_options != DSparkOptions()
            or bool(self.env)
            or bool(self.plugin_options)
        )

    @property
    def empty(self) -> bool:
        return not self.providers and not self.aiter and not self.has_options

    def __reduce__(self):
        state = {f.name: getattr(self, f.name) for f in fields(self)}
        state["tables"] = {
            k: None if t is None else dict(t) for k, t in self.tables.items()
        }
        state["modules"] = {
            k: None if t is None else dict(t) for k, t in self.modules.items()
        }
        for name in ("arms", "expect", "requires", "env", "plugin_options"):
            state[name] = dict(state[name])
        return _rebuild_plan, (state,)

    def dump(self) -> dict[str, Any]:
        out = {
            "plan_id": self.plan_id,
            "schema_version": self.schema_version,
            "strategy": self.strategy,
            "model_arch": self.model_arch,
            "tables": {
                str(key): (
                    None if table is None else {k: list(v) for k, v in table.items()}
                )
                for key, table in self.tables.items()
            },
            "modules": {
                str(key): (
                    None if table is None else {k: list(v) for k, v in table.items()}
                )
                for key, table in self.modules.items()
            },
            "providers": list(self.providers),
            "aiter": list(self.aiter),
            "arms": {k: list(v) for k, v in self.arms.items()},
            "shadowed": [list(x) for x in self.shadowed],
            "strict_overlap": self.strict_overlap,
            "on_decline": self.on_decline,
            "expect": dict(self.expect),
            "requires": dict(self.requires),
            "recipe_digests": list(self.recipe_digests),
            "engine": self.engine.dump(),
            "model_options": self.model_options.dump(),
            "dspark_options": self.dspark_options.dump(),
            "env": dict(self.env),
            "plugin_options": json.loads(_canonical(dict(self.plugin_options))),
        }
        return out


def _rebuild_plan(state: dict[str, Any]) -> Plan:
    state["tables"] = MappingProxyType(
        {
            k: None if t is None else MappingProxyType(t)
            for k, t in state["tables"].items()
        }
    )
    state["modules"] = MappingProxyType(
        {
            k: None if t is None else MappingProxyType(t)
            for k, t in state.get("modules", {}).items()
        }
    )
    for name in ("arms", "expect", "requires", "env", "plugin_options"):
        state[name] = MappingProxyType(state.get(name, {}))
    return Plan(**state)


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _digest(value: Any) -> str:
    return "sha256:" + hashlib.sha256(_canonical(value).encode()).hexdigest()


def _names(value: Any, where: str) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)) or not all(
        isinstance(v, str) and v for v in value
    ):
        raise RecipeRefused("recipe-schema", f"{where} must be a list of names")
    return tuple(value)


def _api_ok(requires: str, api: int | None = None) -> bool:
    return plugin_host.api_satisfies(requires, api)


def shorthand(enable: Iterable[str] | str | None) -> Plan | None:
    """``--plugin-enable a,b``: every name to both halves, owners by this order."""
    if isinstance(enable, str):
        enable = enable.split(",")
    names = tuple(dict.fromkeys(n.strip() for n in enable or () if n.strip()))
    if not names:
        return None
    return Plan(
        plan_id=_digest(["enable", names, plugin_host.HOOK_API]),
        strategy="",
        model_arch=None,
        tables=MappingProxyType({"any": None}),
        modules=MappingProxyType({"any": None}),
        providers=names,
        aiter=names,
        arms=MappingProxyType({"native": (), "plugin": names, "identity": names}),
    )


def _scope_inside(scope: str, slot: str) -> bool:
    """Whether a seam's call-site module path lies in (or at) module ``slot``."""
    import fnmatch

    have, want = scope.split("."), slot.split(".")
    return len(have) >= len(want) and all(
        h == w or fnmatch.fnmatchcase(h, w) or fnmatch.fnmatchcase(w, h)
        for h, w in zip(have, want)
    )


_SEAM_RE = re.compile(r"^[a-z0-9_]+(\.[a-z0-9_]+)+$")


def _check_slot_pattern(slot: str) -> None:
    if not isinstance(slot, str) or not slot or not _SLOT_RE.match(slot):
        raise RecipeRefused("recipe-schema", f"invalid module slot pattern {slot!r}")


ENV_PREFIXES = (
    "ATOM_",
    "AITER_",
    "FLYDSL_",
    "PYTORCH_",
    "HIP_",
    "HSA_",
    "GPU_",
    "NCCL_",
    "RCCL_",
    "TORCH_",
)
_ENV_RE = re.compile(r"^[A-Z][A-Z0-9_]*$")


def _parse_env(raw: Any, is_v2: bool) -> dict[str, str]:
    if raw is None:
        return {}
    if not is_v2:
        raise RecipeRefused(
            "recipe-schema", "env requires schema 'atom.plugin_recipe/2'"
        )
    if not isinstance(raw, Mapping):
        raise RecipeRefused("recipe-schema", "env must map NAME -> value")
    out = {}
    for name, value in raw.items():
        if not (
            isinstance(name, str)
            and _ENV_RE.match(name)
            and name.startswith(ENV_PREFIXES)
        ):
            raise RecipeRefused(
                "recipe-schema", f"env name {name!r}: one of {ENV_PREFIXES}, upper case"
            )
        if name.startswith("ATOM_PLUGIN"):
            raise RecipeRefused("recipe-schema", f"env {name}: the host's own switch")
        if isinstance(value, bool):
            value = "1" if value else "0"
        if not isinstance(value, (str, int, float)):
            raise RecipeRefused("recipe-schema", f"env {name}: a string or a number")
        out[name] = str(value)
    return out


_DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")


def _aiter_names(raw: Any, is_v2: bool) -> tuple[str, ...]:
    """The AITER providers, in priority order. A v2 recipe may group them by
    the AITER op they serve (``{op: [providers]}``): the order is op by op,
    each op's list in order, a provider named twice kept at its first place
    (AITER's enable list is one global order)."""
    if isinstance(raw, Mapping):
        if not is_v2:
            raise RecipeRefused("recipe-schema", "aiter by op requires schema 2")
        ordered: dict[str, None] = {}
        for op, names in raw.items():
            if not isinstance(op, str) or not op:
                raise RecipeRefused("recipe-schema", "aiter ops must be names")
            for name in _names(names, f"aiter.{op}"):
                ordered.setdefault(name, None)
        return tuple(ordered)
    return _names(raw, "aiter") if raw else ()


def _parse(document: Mapping[str, Any], *, model_arch: str | None) -> dict[str, Any]:
    """One recipe document (v1 or v2). A seam's contract is the model's (it is
    written at the model's call site): seams are checked against the built
    model (``plugin_host.check_plan``), not here."""
    if not isinstance(document, Mapping):
        raise RecipeRefused("recipe-schema", "recipe must be a JSON object")
    schema = document.get("schema")
    if schema not in SCHEMAS:
        raise RecipeRefused(
            "recipe-schema", f"expected schema in {SCHEMAS!r}, got {schema!r}"
        )
    is_v2 = schema == RECIPE_SCHEMA_V2
    arch = (
        document.get("arch")
        if is_v2 and "arch" in document
        else document.get("model_arch")
    )
    if not isinstance(arch, str) or not arch:
        raise RecipeRefused("recipe-schema", "model_arch (or arch) is required")
    if model_arch is not None and arch != model_arch:
        raise RecipeRefused("recipe-model-mismatch", f"{arch} != {model_arch}")
    strategy = document.get("strategy", "")
    if not isinstance(strategy, str):
        raise RecipeRefused("recipe-schema", "strategy must be a string")
    graph = document.get("graph", {"mode": "any"})
    mode = graph.get("mode", "any") if isinstance(graph, Mapping) else None
    if mode not in MODES:
        raise RecipeRefused("recipe-schema", f"graph.mode must be one of {MODES}")
    buckets: tuple[int, ...] = ()
    if mode == "capture":
        raw = graph.get("buckets")
        if (
            not isinstance(raw, (list, tuple))
            or not raw
            or not all(
                isinstance(b, int) and not isinstance(b, bool) and b > 0 for b in raw
            )
        ):
            raise RecipeRefused("recipe-schema", "capture recipes need int buckets")
        buckets = tuple(sorted(set(raw)))

    requires = document.get("requires", {})
    if not isinstance(requires, Mapping):
        raise RecipeRefused("recipe-schema", "requires must be an object")
    api = str(requires.get("atom_hook_api", f">={plugin_host.HOOK_API}"))
    if not _api_ok(api):
        raise RecipeRefused(
            "recipe-pin-mismatch", f"atom_hook_api {api} vs {plugin_host.HOOK_API}"
        )
    pins = requires.get("plugins", {})
    if not isinstance(pins, Mapping) or not all(
        isinstance(k, str) and isinstance(v, str) and _DIGEST_RE.match(v)
        for k, v in pins.items()
    ):
        raise RecipeRefused(
            "recipe-schema", "requires.plugins maps provider -> 'sha256:<64 hex>'"
        )
    required = requires.get("seams", {})
    if not isinstance(required, Mapping) or not all(
        _SEAM_RE.match(str(k)) and isinstance(v, int) and v >= 1
        for k, v in required.items()
    ):
        raise RecipeRefused("recipe-schema", "requires.seams maps seam -> contract")

    engine_opts = EngineOptions.from_mapping(document.get("engine"))
    model_opts = ModelOptions.from_mapping(document.get("model_options"))
    dspark_opts = DSparkOptions.from_mapping(document.get("dspark_options"))
    env = _parse_env(document.get("env"), is_v2)
    plugin_opts = document.get("plugin_options", {})
    if plugin_opts and not is_v2:
        raise RecipeRefused("recipe-schema", "plugin_options requires schema 2")
    if not isinstance(plugin_opts, Mapping) or not all(
        isinstance(k, str) and k and isinstance(v, Mapping)
        for k, v in plugin_opts.items()
    ):
        raise RecipeRefused(
            "recipe-schema", "plugin_options maps a plugin name -> an object"
        )
    try:
        plugin_opts = json.loads(json.dumps(plugin_opts))
    except (TypeError, ValueError) as error:
        raise RecipeRefused("recipe-schema", f"plugin_options: {error}") from error

    raw_plugins = document.get("plugins")
    has_explicit_plugins = raw_plugins is not None
    if raw_plugins is None:
        raw_plugins = []
    if not isinstance(raw_plugins, (list, tuple)):
        raise RecipeRefused("recipe-schema", "plugins must be a list")
    plugins: dict[str, Mapping[str, Any]] = {}
    for raw in raw_plugins:
        name = raw.get("name") if isinstance(raw, Mapping) else None
        if not isinstance(name, str) or not name or name in plugins:
            raise RecipeRefused("recipe-schema", "plugins need unique names")
        kind = raw.get("kind", "layer")
        if kind not in ("layer", "aiter"):
            raise RecipeRefused("recipe-schema", f"{name}: kind {kind!r}")
        if "serves" in raw or not is_v2:
            _names(raw.get("serves", []), f"{name}.serves")
        plugins[name] = raw

    extra_aiter = _aiter_names(document.get("aiter", ()), is_v2)
    for name in extra_aiter:
        plugins.setdefault(name, {"name": name, "kind": "aiter", "serves": []})

    plan = document.get("plan", {})
    if not isinstance(plan, Mapping):
        raise RecipeRefused("recipe-schema", "plan must be an object")
    seams = plan.get("seams", document.get("seams", {}))
    if not isinstance(seams, Mapping):
        raise RecipeRefused("recipe-schema", "plan.seams must map seam -> owners")
    raw_modules = document.get("modules", plan.get("modules", {}))
    if not isinstance(raw_modules, Mapping):
        raise RecipeRefused("recipe-schema", "modules must map slot -> owners")
    if not is_v2 and raw_modules:
        raise RecipeRefused(
            "recipe-schema", "modules requires schema 'atom.plugin_recipe/2'"
        )

    defaults = document.get("defaults", {})
    if not isinstance(defaults, Mapping):
        raise RecipeRefused("recipe-schema", "defaults must be an object")
    allow_partition = bool(defaults.get("allow_partition", is_v2))
    shadow_policy = defaults.get("shadow_policy", "nested" if is_v2 else "error")
    on_decline = defaults.get("on_decline", "native")
    if shadow_policy not in SHADOW or on_decline not in ON_DECLINE:
        raise RecipeRefused("recipe-schema", "shadow_policy / on_decline")

    table: dict[str, tuple[str, ...]] = {}
    module_table: dict[str, tuple[str, ...]] = {}

    native_seams: list[str] = []
    for seam, owners in seams.items():
        if not isinstance(seam, str) or not _SEAM_RE.match(seam):
            raise RecipeRefused("recipe-unknown-seam", f"{seam!r} is not a seam name")
        owners = _names(owners, f"plan.seams[{seam}]")
        if not owners:
            native_seams.append(seam)
        if len(set(owners)) != len(owners):
            raise RecipeRefused("recipe-schema", f"{seam}: an owner twice")
        for owner in owners:
            ref = plugins.get(owner)
            if ref is None and is_v2 and not has_explicit_plugins:
                ref = plugins[owner] = {
                    "name": owner,
                    "kind": "layer",
                    "serves": [seam],
                }
            if ref is None:
                raise RecipeRefused("recipe-unknown-provider", f"{seam}: {owner}")
            if ref.get("kind", "layer") != "layer":
                raise RecipeRefused("recipe-conflict", f"{owner} is an AITER provider")
            if "serves" in ref and seam not in ref.get("serves", []):
                raise RecipeRefused(
                    "recipe-plugin-drift", f"{owner} does not declare {seam}"
                )
        if len(owners) > 1 and not allow_partition:
            raise RecipeRefused(
                "recipe-conflict",
                f"{seam} has {list(owners)}: set defaults.allow_partition "
                "(disjoint cells) or keep one",
            )
        if owners:
            table[seam] = owners

    for slot, owners in raw_modules.items():
        _check_slot_pattern(slot)
        owners = _names(owners, f"modules[{slot}]")
        if len(set(owners)) != len(owners):
            raise RecipeRefused("recipe-schema", f"modules[{slot}]: an owner twice")
        if len(owners) > 1 and not allow_partition:
            raise RecipeRefused(
                "recipe-conflict",
                f"modules[{slot}] has {list(owners)}: set defaults.allow_partition "
                "or keep one",
            )
        for owner in owners:
            ref = plugins.get(owner)
            if ref is None and not has_explicit_plugins:
                ref = plugins[owner] = {"name": owner, "kind": "layer"}
            if ref is None:
                raise RecipeRefused(
                    "recipe-unknown-provider", f"modules[{slot}]: {owner}"
                )
            if ref.get("kind", "layer") != "layer":
                raise RecipeRefused("recipe-conflict", f"{owner} is an AITER provider")
            serves = ref.get("serves")
            if serves is not None and slot not in serves:
                raise RecipeRefused(
                    "recipe-plugin-drift", f"{owner} does not declare {slot}"
                )
        if owners:
            module_table[slot] = owners

    arms = document.get("arms", {})
    if not isinstance(arms, Mapping):
        raise RecipeRefused("recipe-schema", "arms must map name -> providers")
    for arm, members in arms.items():
        unknown = [m for m in _names(members, f"arms[{arm}]") if m not in plugins]
        if unknown:
            raise RecipeRefused("recipe-unknown-provider", f"arms[{arm}]: {unknown}")
    runtime = document.get("runtime", {})
    expect = runtime.get("expect", {}) if isinstance(runtime, Mapping) else None
    if not isinstance(expect, Mapping):
        raise RecipeRefused("recipe-schema", "runtime.expect must be an object")
    return {
        "schema_version": 2 if is_v2 else 1,
        "digest": _digest(document),
        "model_arch": arch,
        "strategy": strategy,
        "mode": mode,
        "buckets": buckets,
        "plugins": plugins,
        "table": table,
        "module_table": module_table,
        "shadow_policy": shadow_policy,
        "native_seams": tuple(native_seams),
        "arms": {k: tuple(v) for k, v in arms.items()},
        "strict_overlap": bool(defaults.get("strict_overlap", False)),
        "on_decline": on_decline,
        "expect": dict(expect),
        "requires": dict(requires),
        "plugin_options": plugin_opts,
        "engine": engine_opts,
        "model_options": model_opts,
        "dspark_options": dspark_opts,
        "env": env,
    }


def resolve(
    documents: Iterable[Mapping[str, Any]], *, model_arch: str | None = None
) -> Plan:
    """One strategy's recipes (one per graph mode / bucket set) -> Plan."""
    parsed = [_parse(d, model_arch=model_arch) for d in documents]
    if not parsed:
        raise RecipeRefused("recipe-schema", "no recipe")
    first = parsed[0]
    for other in parsed[1:]:
        for field_name in (
            "model_arch",
            "strategy",
            "on_decline",
            "strict_overlap",
            "engine",
            "model_options",
            "dspark_options",
            "env",
        ):
            if other[field_name] != first[field_name]:
                raise RecipeRefused("recipe-set-conflict", f"{field_name} differs")
        if other["expect"] != first["expect"]:
            raise RecipeRefused("recipe-set-conflict", "runtime.expect differs")
    tables: dict[Any, Mapping[str, tuple[str, ...]]] = {}
    module_tables: dict[Any, Mapping[str, tuple[str, ...]]] = {}
    for recipe in parsed:
        keys = recipe["buckets"] or (recipe["mode"],)
        for key in keys:
            clash = key in tables or (
                key in ("any", "eager") and {"any", "eager"} & set(tables)
            )
            if clash:
                raise RecipeRefused("recipe-set-conflict", f"two recipes serve {key!r}")
            tables[key] = MappingProxyType(dict(recipe["table"]))
            module_tables[key] = MappingProxyType(dict(recipe["module_table"]))

    providers: dict[str, None] = {}
    aiter: dict[str, None] = {}
    arms: dict[str, tuple[str, ...]] = {}
    for recipe in parsed:
        for name, ref in recipe["plugins"].items():
            if ref.get("kind", "layer") == "aiter":
                aiter[name] = None
        for table_owners in recipe["table"].values():
            providers.update(dict.fromkeys(table_owners))
        for mod_owners in recipe["module_table"].values():
            providers.update(dict.fromkeys(mod_owners))
        for arm, members in recipe["arms"].items():
            if arm in arms and arms[arm] != members:
                raise RecipeRefused("recipe-set-conflict", f"arm {arm} differs")
            arms[arm] = members
    everyone = (*providers, *aiter)
    arms.setdefault("native", ())
    arms.setdefault("plugin", everyone)
    arms.setdefault("identity", everyone)
    digests = tuple(sorted(r["digest"] for r in parsed))
    # The contracts the recipes require (the model checks them once built).
    contracts = {
        seam: contract
        for r in parsed
        for seam, contract in dict(r["requires"].get("seams", {})).items()
    }
    native_seams = tuple(sorted({s for r in parsed for s in r["native_seams"]}))
    policies = {r["shadow_policy"] for r in parsed}
    shadow_policy = "error" if "error" in policies else first["shadow_policy"]
    schema_version = max(r["schema_version"] for r in parsed)
    id_payload: list[Any] = [
        digests,
        plugin_host.HOOK_API,
        contracts,
        sorted(arms),
        list(providers),
    ]
    if schema_version >= 2:
        id_payload.append(
            {
                "modules": {
                    str(k): {s: list(o) for s, o in v.items()}
                    for k, v in sorted(module_tables.items(), key=lambda kv: str(kv[0]))
                },
                "engine": first["engine"].dump(),
                "model_options": first["model_options"].dump(),
                "dspark_options": first["dspark_options"].dump(),
                "env": dict(sorted(first["env"].items())),
                "plugin_options": first["plugin_options"],
            }
        )
    plan_id = _digest(id_payload)
    return Plan(
        plan_id=plan_id,
        strategy=first["strategy"],
        model_arch=first["model_arch"],
        tables=MappingProxyType(tables),
        providers=tuple(providers),
        aiter=tuple(aiter),
        arms=MappingProxyType(arms),
        shadow_policy=shadow_policy,
        native_seams=native_seams,
        strict_overlap=first["strict_overlap"],
        on_decline=first["on_decline"],
        expect=MappingProxyType(first["expect"]),
        requires=MappingProxyType(first["requires"]),
        recipe_digests=digests,
        modules=MappingProxyType(module_tables),
        engine=first["engine"],
        model_options=first["model_options"],
        dspark_options=first["dspark_options"],
        env=MappingProxyType(dict(first["env"])),
        plugin_options=MappingProxyType(dict(first["plugin_options"])),
        schema_version=schema_version,
    )


def _read(path: str | Path) -> tuple[str, ...]:
    """A recipe file, or a directory holding one strategy's recipes, as
    canonical JSON text."""
    path = Path(path)
    files = sorted(path.glob("*.json")) if path.is_dir() else [path]
    if not files:
        raise RecipeRefused("recipe-schema", f"no *.json under {path}")
    documents = []
    for file in files:
        try:
            documents.append(_canonical(json.loads(file.read_text(encoding="utf-8"))))
        except (OSError, ValueError) as error:
            raise RecipeRefused("recipe-schema", f"{file}: {error}") from error
    return tuple(documents)


def load(path: str | Path, *, model_arch: str | None = None) -> Plan:
    """A recipe file, or a directory holding one strategy's recipes."""
    return resolve([json.loads(d) for d in _read(path)], model_arch=model_arch)


def _plan_of(documents: tuple[str, ...], enable: tuple[str, ...]) -> Plan | None:
    """Recipes plus ``--plugin-enable``, which may only narrow them."""
    names = shorthand(enable)
    if not documents:
        return names
    plan = resolve([json.loads(d) for d in documents])
    if names is None:
        return plan
    extra = [n for n in names.providers if n not in plan.arms["plugin"]]
    if extra:
        raise RecipeRefused("recipe-unknown-provider", f"--plugin-enable adds {extra}")
    arms = dict(plan.arms)
    arms["plugin"] = arms["identity"] = names.providers
    return replace(
        plan,
        arms=MappingProxyType(arms),
        plan_id=_digest([plan.plan_id, "enable", names.providers]),
    )


@dataclass(frozen=True)
class RecipeSource:
    """What ``Config`` carries: the recipes as canonical JSON text and the
    ``--plugin-enable`` names, plain data a rank's config copies and pickles;
    every process resolves it to the same ``Plan`` (``plan_id`` checks it)."""

    documents: tuple[str, ...] = ()
    enable: tuple[str, ...] = ()
    plan_id: str = ""
    empty: bool = True

    def resolve(self) -> Plan | None:
        plan = _plan_of(self.documents, self.enable)
        if plan is not None and plan.plan_id != self.plan_id:
            raise RecipeRefused(
                "recipe-set-conflict", "the recipes resolved to another plan here"
            )
        return plan


def from_cli(
    recipe: str | None, enable: str | None, arm: str = "plugin"
) -> RecipeSource | None:
    """``--plugin-recipe`` / ``--plugin-enable`` / ``--plugin-arm``, resolved
    once to refuse a bad recipe before any worker starts; None: no plugin."""
    documents = _read(recipe) if recipe else ()
    names = shorthand(enable)
    names = () if names is None else names.providers
    plan = _plan_of(documents, names)
    if plan is None:
        return None
    if arm not in plan.arms:
        raise RecipeRefused("recipe-unknown-arm", f"{arm!r} not in {sorted(plan.arms)}")
    apply_env(plan)
    return RecipeSource(documents, names, plan.plan_id, plan.empty)


def apply_env(plan: Plan | None, environ=None) -> dict[str, str]:
    """Set a recipe's ``env`` in this process before any worker starts (the
    workers inherit it). A variable already set to another value refuses the
    recipe: the deployment has one source of truth. Returns what was set."""
    environ = os.environ if environ is None else environ
    if plan is None or not plan.env:
        return {}
    clash = {
        k: (environ[k], v)
        for k, v in plan.env.items()
        if k in environ and environ[k] != v
    }
    if clash:
        raise RecipeRefused(
            "recipe-env-conflict",
            f"set in the environment and differently in the recipe: {clash}",
        )
    applied = {k: v for k, v in plan.env.items() if k not in environ}
    environ.update(applied)
    return applied


def _document_text(document: Any, where: str) -> str:
    """One recipe document (JSON text, bytes or a decoded object) as JSON text."""
    if isinstance(document, Mapping):
        return _canonical(dict(document))
    if isinstance(document, (bytes, bytearray)):
        document = document.decode("utf-8")
    if isinstance(document, str):
        try:
            decoded = json.loads(document)
        except ValueError as error:
            raise RecipeRefused("recipe-schema", f"{where}: {error}") from error
        if not isinstance(decoded, Mapping):
            raise RecipeRefused("recipe-schema", f"{where}: a recipe is an object")
        return document
    raise RecipeRefused("recipe-schema", f"{where}: {type(document).__name__}")


def _enable_names(enable: Any) -> tuple[str, ...]:
    if enable is None:
        return ()
    if isinstance(enable, str):
        enable = enable.split(",")
    return tuple(
        dict.fromkeys(n.strip() for n in _names(enable, "enable") if n.strip())
    )


def from_payload(payload: Any) -> RecipeSource | None:
    """The ``RecipeSource`` a hot-plug payload describes."""
    if payload is None:
        return None
    if isinstance(payload, RecipeSource):
        return payload
    plan_id = None
    if isinstance(payload, Mapping):
        if "schema" in payload:
            documents, enable = (payload,), ()
        else:
            documents = payload.get("documents", payload.get("recipe", ()))
            enable = payload.get("enable", ())
            plan_id = payload.get("plan_id")
    elif isinstance(payload, (str, bytes)):
        documents, enable = (payload,), ()
    else:
        try:
            documents = tuple(payload)
        except TypeError as error:
            raise RecipeRefused(
                "recipe-schema", f"payload documents: {type(payload).__name__}"
            ) from error
    if isinstance(documents, (str, bytes, Mapping)):
        documents = (documents,)
    texts = tuple(
        _document_text(document, f"documents[{index}]")
        for index, document in enumerate(documents)
    )
    names = _enable_names(enable)
    plan = _plan_of(texts, names)
    if plan is None:
        return RecipeSource(texts, names, "", True)
    if plan_id is not None and plan_id != plan.plan_id:
        raise RecipeRefused(
            "recipe-set-conflict",
            f"payload plan_id {plan_id!r} does not resolve to the same plan here",
        )
    return RecipeSource(texts, names, plan.plan_id, plan.empty)


def engine_options_for(config: Any) -> EngineOptions:
    """Engine-level scheduling/DP options resolved from ``config``'s recipe."""
    explicit = getattr(config, "plugin_engine_options", None)
    if isinstance(explicit, EngineOptions):
        return explicit
    if isinstance(explicit, Mapping):
        return EngineOptions.from_mapping(explicit)
    from atom.utils import envs

    # As the model options: any arm (native included) runs one engine; only
    # the kill switch drops the recipe.
    if not envs.ATOM_PLUGIN_HOST:
        return EngineOptions()
    source = getattr(config, "plugin_source", None)
    if isinstance(source, Plan):
        return source.engine
    if not isinstance(source, RecipeSource):
        return EngineOptions()
    plan = source.resolve()
    return EngineOptions() if plan is None else plan.engine


def check_runtime(plan: Plan, config: Any) -> None:
    """``runtime.expect``: the server must be the one the recipe was made for."""

    def is_window(seam: str) -> bool:
        spec = plugin_host.SEAMS.get(seam)
        return spec is not None and spec.form == "window"

    has_transition = any(
        isinstance(key, int) and table and any(is_window(s) for s in table)
        for key, table in plan.tables.items()
    )
    if has_transition:
        compile_cfg = getattr(config, "compilation_config", None)
        mode = str(getattr(compile_cfg, "cudagraph_mode", "")).rsplit(".", 1)[-1]
        transition_opted_in = (
            plan.model_options.block_transition
            or os.getenv("ATOM_V41_TRANSITION_AB") == "1"
        )
        if (
            not transition_opted_in
            or mode != "FULL"
            or getattr(compile_cfg, "level", None) != 0
            or getattr(config, "tensor_parallel_size", None) != 4
            or getattr(
                getattr(config, "parallel_config", None), "data_parallel_size", 1
            )
            != 1
            or any(
                getattr(config, n, 1) != 1
                for n in (
                    "prefill_context_parallel_size",
                    "decode_context_parallel_size",
                    "pipeline_parallel_size",
                )
            )
            or any(
                getattr(config, n, False)
                for n in (
                    "enable_dp_attention",
                    "enable_expert_parallel",
                    "enable_tbo",
                    "enable_decoder_swa_bounded_replay",
                    "enforce_eager",
                )
            )
        ):
            raise RecipeRefused(
                "recipe-window-topology",
                "capture transition needs opted-in FULL level-0 TP4 without DPA/EP/TBO/replay",
            )
    actual = {
        "level": lambda c: getattr(
            getattr(c, "compilation_config", None), "level", None
        ),
        "cudagraph_mode": lambda c: str(
            getattr(getattr(c, "compilation_config", None), "cudagraph_mode", None)
        ).rsplit(".", 1)[-1],
        "tensor_parallel_size": lambda c: getattr(c, "tensor_parallel_size", None),
        "model_arch": lambda c: (
            getattr(getattr(c, "hf_config", None), "architectures", None) or [None]
        )[0],
        "mono": lambda c: _mono_enabled(plan),
    }
    for name, expected in plan.expect.items():
        read = actual.get(name)
        if read is None:
            raise RecipeRefused("recipe-schema", f"runtime.expect has no {name!r}")
        value = read(config)
        if value != expected:
            raise RecipeRefused(
                "recipe-runtime-mismatch",
                f"{name}: expected {expected!r}, got {value!r}",
            )


def _mono_enabled(plan: Plan | None = None) -> bool:
    if plan is not None and plan.model_options.mono_mode == "off":
        return False
    from atom.utils import envs

    return bool(envs.ATOM_MONO_ENABLE)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m atom.plugin_recipe")
    parser.add_argument("path", help="recipe file or a strategy's recipe directory")
    parser.add_argument(
        "--installed",
        action="append",
        default=[],
        help="name=seam_or_slot,...: what an installed plugin registers (drift check)",
    )
    args = parser.parse_args(argv)
    try:
        plan = load(args.path)
        for item in args.installed:
            name, _, targets = item.partition("=")
            served = set(filter(None, targets.split(",")))
            for table in plan.tables.values():
                for seam, owners in (table or {}).items():
                    if name in owners and seam not in served:
                        raise RecipeRefused(
                            "recipe-plugin-drift", f"installed {name} lacks {seam}"
                        )
            for table in plan.modules.values():
                for slot, owners in (table or {}).items():
                    if name in owners and slot not in served:
                        raise RecipeRefused(
                            "recipe-plugin-drift", f"installed {name} lacks {slot}"
                        )
    except RecipeRefused as refused:
        print(f"REFUSED {refused}")
        return 2
    print(json.dumps(plan.dump(), indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
