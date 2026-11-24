# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""Plugin recipes (``atom.plugin_recipe/2``): which plugin serves which seam or
module, per capture bucket set, and the model, environment and plugin options
a deployment needs, in one document.

Graph modes: ``any`` (default) serves every pass; ``eager`` serves uncaptured
passes; ``capture`` with ``buckets`` serves captured passes whose padded rows
match one of the buckets. A recipe is fixed for a process: to switch recipes,
restart (``ATOM_PRELOAD=1`` keeps the weights on the GPUs across the restart).
``python -m atom.plugin_recipe PATH`` validates and dumps a recipe or a
strategy's recipe directory on CPU.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from types import MappingProxyType
from typing import Any

from atom import plugin_host

RECIPE_SCHEMA = "atom.plugin_recipe/2"
MODES = ("any", "eager", "capture")
ON_DECLINE = ("native", "error")
SHADOW = ("error", "shadow", "nested")
# A module slot: a dotted path (``*`` one segment) or ``@<qualified.Class>``
# (every instance of the class, wherever it sits), with an optional
# ``:method`` (that method alone is taken over).
_SLOT_RE = re.compile(
    r"^(@[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)+"
    r"|[A-Za-z0-9_*]+(\.[A-Za-z0-9_*]+)*)"
    r"(:[A-Za-z_][A-Za-z0-9_]*)?$"
)
_SEAM_RE = re.compile(r"^[a-z0-9_]+(\.[a-z0-9_]+)+$")
_DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_TOP_LEVEL = frozenset(
    {
        "schema",
        "arch",
        "model_arch",
        "strategy",
        "note",
        "graph",
        "defaults",
        "plugins",
        "aiter",
        "seams",
        "plan",
        "modules",
        "arms",
        "requires",
        "runtime",
        "model_options",
        "plugin_options",
        "env",
    }
)


class RecipeRefused(ValueError):
    """A recipe that can never be valid here; ``reason`` is its short code."""

    def __init__(self, reason: str, detail: str):
        super().__init__(f"{reason}: {detail}")
        self.reason = reason


def _opt_str(value: Any, where: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise RecipeRefused("recipe-schema", f"{where} must be a non-empty string")
    return value.strip()


@dataclass(frozen=True)
class ModelOptions:
    """The model a plan builds: a plugin package's model class (with its seams
    in it) for the target and the drafter; ATOM's own where None."""

    target_model_cls: str | None = None
    draft_model_cls: str | None = None

    @classmethod
    def from_mapping(cls, raw: Any) -> ModelOptions:
        if raw is None:
            return cls()
        if not isinstance(raw, Mapping):
            raise RecipeRefused("recipe-schema", "model_options must be an object")
        unknown = sorted(set(raw) - {f.name for f in fields(cls)})
        if unknown:
            raise RecipeRefused(
                "recipe-schema",
                f"unknown model_options field(s): {unknown} (a model package's "
                "own options go to plugin_options)",
            )
        return cls(
            target_model_cls=_opt_str(
                raw.get("target_model_cls"), "model_options.target_model_cls"
            ),
            draft_model_cls=_opt_str(
                raw.get("draft_model_cls"), "model_options.draft_model_cls"
            ),
        )

    def dump(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Plan:
    """What the host runs on: seam and module tables and options, frozen before
    the build. ``tables`` / ``modules`` keys are ``"any"``, ``"eager"`` or a
    capture bucket (int)."""

    plan_id: str
    strategy: str
    model_arch: str | None
    tables: Mapping[Any, Mapping[str, tuple[str, ...]]]
    providers: tuple[str, ...]
    aiter: tuple[str, ...]
    arms: Mapping[str, tuple[str, ...]]
    # How a nested / covered owner shadowed by an outer one is treated, checked
    # against the built model (``plugin_host.check_plan``): nested or shadow
    # (record), error (refuse).
    shadow_policy: str = "nested"
    # Seams the recipe plans with no owner ([]): native by decision, which
    # counts as planned for a taken-over module's inner seams.
    native_seams: tuple[str, ...] = ()
    strict_overlap: bool = False
    on_decline: str = "native"
    expect: Mapping[str, Any] = MappingProxyType({})
    requires: Mapping[str, Any] = MappingProxyType({})
    recipe_digests: tuple[str, ...] = ()
    modules: Mapping[Any, Mapping[str, tuple[str, ...]]] = MappingProxyType({})
    model_options: ModelOptions = ModelOptions()
    # Process environment the recipe sets for the server it configures
    # (`apply_env`): native ATOM / AITER switches that are not recipe options.
    env: Mapping[str, str] = MappingProxyType({})
    # Options for plugin packages, by name (a provider or a model package):
    # ATOM passes them through unread (``plugin_host.plugin_options``).
    plugin_options: Mapping[str, Any] = MappingProxyType({})

    @property
    def has_options(self) -> bool:
        return (
            self.model_options != ModelOptions()
            or bool(self.env)
            or bool(self.plugin_options)
        )

    @property
    def empty(self) -> bool:
        return not self.providers and not self.aiter and not self.has_options

    def __reduce__(self):
        state = {f.name: getattr(self, f.name) for f in fields(self)}
        state["tables"] = {k: dict(t) for k, t in self.tables.items()}
        state["modules"] = {k: dict(t) for k, t in self.modules.items()}
        for name in ("arms", "expect", "requires", "env", "plugin_options"):
            state[name] = dict(state[name])
        return _rebuild_plan, (state,)

    def dump(self) -> dict[str, Any]:
        def table_dump(tables):
            return {
                str(key): {k: list(v) for k, v in table.items()}
                for key, table in tables.items()
            }

        return {
            "plan_id": self.plan_id,
            "strategy": self.strategy,
            "model_arch": self.model_arch,
            "tables": table_dump(self.tables),
            "modules": table_dump(self.modules),
            "providers": list(self.providers),
            "aiter": list(self.aiter),
            "arms": {k: list(v) for k, v in self.arms.items()},
            "shadow_policy": self.shadow_policy,
            "native_seams": list(self.native_seams),
            "strict_overlap": self.strict_overlap,
            "on_decline": self.on_decline,
            "expect": dict(self.expect),
            "requires": dict(self.requires),
            "recipe_digests": list(self.recipe_digests),
            "model_options": self.model_options.dump(),
            "env": dict(self.env),
            "plugin_options": json.loads(_canonical(dict(self.plugin_options))),
        }


def _rebuild_plan(state: dict[str, Any]) -> Plan:
    for name in ("tables", "modules"):
        state[name] = MappingProxyType(
            {k: MappingProxyType(t) for k, t in state.get(name, {}).items()}
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


def _parse_env(raw: Any) -> dict[str, str]:
    if raw is None:
        return {}
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


def _aiter_names(raw: Any) -> tuple[str, ...]:
    """The AITER providers, in priority order. A recipe may group them by the
    AITER op they serve (``{op: [providers]}``): the order is op by op, each
    op's list in order, a provider named twice kept at its first place
    (AITER's enable list is one global order)."""
    if isinstance(raw, Mapping):
        ordered: dict[str, None] = {}
        for op, names in raw.items():
            if not isinstance(op, str) or not op:
                raise RecipeRefused("recipe-schema", "aiter ops must be names")
            for name in _names(names, f"aiter.{op}"):
                ordered.setdefault(name, None)
        return tuple(ordered)
    return _names(raw, "aiter") if raw else ()


def _parse(document: Mapping[str, Any], *, model_arch: str | None) -> dict[str, Any]:
    """One recipe document. A seam's contract is the model's (it is written at
    the model's call site): seams are checked against the built model
    (``plugin_host.check_plan``), not here."""
    if not isinstance(document, Mapping):
        raise RecipeRefused("recipe-schema", "recipe must be a JSON object")
    schema = document.get("schema")
    if schema != RECIPE_SCHEMA:
        raise RecipeRefused(
            "recipe-schema", f"expected schema {RECIPE_SCHEMA!r}, got {schema!r}"
        )
    # Other top-level fields are the document's own metadata (a tool's plan_id,
    # purpose, baseline, notes...): ATOM reads none of them. Sections this ATOM
    # no longer has are refused rather than silently dropped.
    removed = sorted(set(document) & {"engine", "dspark_options"})
    if removed:
        raise RecipeRefused(
            "recipe-schema",
            f"{removed}: not a recipe section (model options go in plugin_options)",
        )
    arch = document.get("arch", document.get("model_arch"))
    if not isinstance(arch, str) or not arch:
        raise RecipeRefused("recipe-schema", "arch is required")
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
    if not plugin_host.api_satisfies(api):
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

    model_opts = ModelOptions.from_mapping(document.get("model_options"))
    env = _parse_env(document.get("env"))
    plugin_opts = document.get("plugin_options", {})
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
        if "serves" in raw:
            _names(raw.get("serves", []), f"{name}.serves")
        plugins[name] = raw

    for name in _aiter_names(document.get("aiter", ())):
        plugins.setdefault(name, {"name": name, "kind": "aiter", "serves": []})

    plan = document.get("plan", {})
    if not isinstance(plan, Mapping):
        raise RecipeRefused("recipe-schema", "plan must be an object")
    seams = document.get("seams", plan.get("seams", {}))
    if not isinstance(seams, Mapping):
        raise RecipeRefused("recipe-schema", "seams must map seam -> owners")
    raw_modules = document.get("modules", plan.get("modules", {}))
    if not isinstance(raw_modules, Mapping):
        raise RecipeRefused("recipe-schema", "modules must map slot -> owners")

    defaults = document.get("defaults", {})
    if not isinstance(defaults, Mapping):
        raise RecipeRefused("recipe-schema", "defaults must be an object")
    allow_partition = bool(defaults.get("allow_partition", True))
    shadow_policy = defaults.get("shadow_policy", "nested")
    on_decline = defaults.get("on_decline", "native")
    if shadow_policy not in SHADOW or on_decline not in ON_DECLINE:
        raise RecipeRefused("recipe-schema", "shadow_policy / on_decline")

    def owner_ref(owner: str, target: str, where: str) -> Mapping[str, Any]:
        ref = plugins.get(owner)
        if ref is None and not has_explicit_plugins:
            # An inferred plugin serves what the recipe plans for it: one owner
            # may serve several seams or slots (its pattern's bindings).
            ref = plugins[owner] = {"name": owner, "kind": "layer", "serves": []}
        if ref is None:
            raise RecipeRefused("recipe-unknown-provider", f"{where}: {owner}")
        if ref.get("kind", "layer") != "layer":
            raise RecipeRefused("recipe-conflict", f"{owner} is an AITER provider")
        if not has_explicit_plugins and target not in ref["serves"]:
            ref["serves"].append(target)
        if "serves" in ref and target not in ref.get("serves", []):
            raise RecipeRefused(
                "recipe-plugin-drift", f"{owner} does not declare {target}"
            )
        return ref

    table: dict[str, tuple[str, ...]] = {}
    native_seams: list[str] = []
    for seam, owners in seams.items():
        if not isinstance(seam, str) or not _SEAM_RE.match(seam):
            raise RecipeRefused("recipe-unknown-seam", f"{seam!r} is not a seam name")
        owners = _names(owners, f"seams[{seam}]")
        if len(set(owners)) != len(owners):
            raise RecipeRefused("recipe-schema", f"{seam}: an owner twice")
        if len(owners) > 1 and not allow_partition:
            raise RecipeRefused(
                "recipe-conflict",
                f"{seam} has {list(owners)}: set defaults.allow_partition "
                "(disjoint cells) or keep one",
            )
        for owner in owners:
            owner_ref(owner, seam, seam)
        if owners:
            table[seam] = owners
        else:
            native_seams.append(seam)

    module_table: dict[str, tuple[str, ...]] = {}
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
            owner_ref(owner, slot, f"modules[{slot}]")
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
        "model_options": model_opts,
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
            "model_options",
            "env",
            "plugin_options",
        ):
            if other[field_name] != first[field_name]:
                raise RecipeRefused("recipe-set-conflict", f"{field_name} differs")
        if other["expect"] != first["expect"]:
            raise RecipeRefused("recipe-set-conflict", "runtime.expect differs")
    tables: dict[Any, Mapping[str, tuple[str, ...]]] = {}
    module_tables: dict[Any, Mapping[str, tuple[str, ...]]] = {}
    for recipe in parsed:
        for key in recipe["buckets"] or (recipe["mode"],):
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
        for owners in (*recipe["table"].values(), *recipe["module_table"].values()):
            providers.update(dict.fromkeys(owners))
        for arm, members in recipe["arms"].items():
            if arm in arms and arms[arm] != members:
                raise RecipeRefused("recipe-set-conflict", f"arm {arm} differs")
            arms[arm] = members
    everyone = (*providers, *aiter)
    arms.setdefault("native", ())
    arms.setdefault("plugin", everyone)
    arms.setdefault("identity", everyone)
    digests = tuple(sorted(r["digest"] for r in parsed))
    native_seams = tuple(sorted({s for r in parsed for s in r["native_seams"]}))
    policies = {r["shadow_policy"] for r in parsed}
    shadow_policy = "error" if "error" in policies else first["shadow_policy"]
    plan_id = _digest(
        [
            digests,
            plugin_host.HOOK_API,
            sorted(arms),
            list(providers),
            {
                str(k): {s: list(o) for s, o in v.items()}
                for k, v in sorted(module_tables.items(), key=lambda kv: str(kv[0]))
            },
        ]
    )
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
        model_options=first["model_options"],
        env=MappingProxyType(dict(first["env"])),
        plugin_options=MappingProxyType(dict(first["plugin_options"])),
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


@dataclass(frozen=True)
class RecipeSource:
    """What ``Config`` carries: the recipes as canonical JSON text, plain data a
    rank's config copies and pickles; every process resolves it to the same
    ``Plan`` (``plan_id`` checks it)."""

    documents: tuple[str, ...] = ()
    plan_id: str = ""
    empty: bool = True

    def resolve(self) -> Plan | None:
        if not self.documents:
            return None
        plan = resolve([json.loads(d) for d in self.documents])
        if plan.plan_id != self.plan_id:
            raise RecipeRefused(
                "recipe-set-conflict", "the recipes resolved to another plan here"
            )
        return plan


def from_cli(recipe: str | None, arm: str = "plugin") -> RecipeSource | None:
    """``--plugin-recipe`` / ``--plugin-arm``, resolved once to refuse a bad
    recipe before any worker starts; None: no plugin."""
    if not recipe:
        return None
    documents = _read(recipe)
    plan = resolve([json.loads(d) for d in documents])
    if arm not in plan.arms:
        raise RecipeRefused("recipe-unknown-arm", f"{arm!r} not in {sorted(plan.arms)}")
    apply_env(plan)
    return RecipeSource(documents, plan.plan_id, plan.empty)


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


def check_runtime(plan: Plan, config: Any) -> None:
    """``runtime.expect``: the server must be the one the recipe was made for."""

    def mono(_config) -> bool:
        from atom.utils import envs

        return bool(envs.ATOM_MONO_ENABLE)

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
        "mono": mono,
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
            for tables in (plan.tables, plan.modules):
                for table in tables.values():
                    for target, owners in table.items():
                        if name in owners and target not in served:
                            raise RecipeRefused(
                                "recipe-plugin-drift",
                                f"installed {name} lacks {target}",
                            )
    except RecipeRefused as refused:
        print(f"REFUSED {refused}")
        return 2
    print(json.dumps(plan.dump(), indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
