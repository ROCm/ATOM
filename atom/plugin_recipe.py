# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Plugin recipes: which installed plugins serve which seams, as data.

A recipe (schema ``atom.plugin_recipe/1``) names already-installed plugins and
pins them to seams of ``atom.plugin_host.SEAMS``; it carries no code. ATOM
owns the schema and every refusal; ``load`` turns one recipe, or a strategy's
directory of recipes (one per graph-capture bucket set), into an immutable
``Plan`` before the model is built. ``python -m atom.plugin_recipe PATH``
checks a recipe on a CPU.

Graph modes: ``any`` (default) serves every pass; ``eager`` the passes that
are not captured; ``capture`` with ``buckets`` the captured passes whose
padded rows are one of them. A captured bucket no recipe names falls back to
the ``any`` recipe, else native.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, fields, replace
from pathlib import Path
from types import MappingProxyType
from typing import Any

from atom import plugin_host

RECIPE_SCHEMA = "atom.plugin_recipe/1"
MODES = ("any", "eager", "capture")
ON_DECLINE = ("native", "error")
SHADOW = ("error", "shadow")


class RecipeRefused(ValueError):
    """A recipe that can never be valid here; ``reason`` is its short code."""

    def __init__(self, reason: str, detail: str):
        super().__init__(f"{reason}: {detail}")
        self.reason = reason


@dataclass(frozen=True)
class Plan:
    """What the host runs on: owners per seam per table, frozen before build.

    ``tables`` keys are ``"any"``, ``"eager"`` or a capture bucket (int); a
    table of ``None`` (the ``--plugin-enable`` shorthand) means "whatever the
    enabled plugins register, in ``providers`` order".
    """

    plan_id: str
    strategy: str
    model_arch: str | None
    tables: Mapping[Any, Mapping[str, tuple[str, ...]] | None]
    providers: tuple[str, ...]  # layer plugins, in load / preference order
    aiter: tuple[str, ...]  # AITER providers, for aiter.plugin.configure
    arms: Mapping[str, tuple[str, ...]]
    shadowed: tuple[tuple[str, str, str], ...] = ()
    strict_overlap: bool = False
    on_decline: str = "native"
    expect: Mapping[str, Any] = MappingProxyType({})
    requires: Mapping[str, Any] = MappingProxyType({})
    recipe_digests: tuple[str, ...] = ()

    @property
    def empty(self) -> bool:
        return not self.providers and not self.aiter

    def __reduce__(self):
        # Copies and pickles go through plain dicts (a mappingproxy has neither).
        state = {f.name: getattr(self, f.name) for f in fields(self)}
        state["tables"] = {
            k: None if t is None else dict(t) for k, t in self.tables.items()
        }
        for name in ("arms", "expect", "requires"):
            state[name] = dict(state[name])
        return _rebuild_plan, (state,)

    def dump(self) -> dict[str, Any]:
        return {
            "plan_id": self.plan_id,
            "strategy": self.strategy,
            "model_arch": self.model_arch,
            "tables": {
                str(key): (
                    None if table is None else {k: list(v) for k, v in table.items()}
                )
                for key, table in self.tables.items()
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
        }


def _rebuild_plan(state: dict[str, Any]) -> Plan:
    state["tables"] = MappingProxyType(
        {
            k: None if t is None else MappingProxyType(t)
            for k, t in state["tables"].items()
        }
    )
    for name in ("arms", "expect", "requires"):
        state[name] = MappingProxyType(state[name])
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


def _api_ok(requires: str, api: int = plugin_host.HOOK_API) -> bool:
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
        providers=names,
        aiter=names,
        arms=MappingProxyType({"native": (), "plugin": names, "identity": names}),
    )


def _parse(document: Mapping[str, Any], *, model_arch: str | None) -> dict[str, Any]:
    """One recipe document, checked against this ATOM's seams."""
    if not isinstance(document, Mapping) or document.get("schema") != RECIPE_SCHEMA:
        raise RecipeRefused("recipe-schema", f"expected schema {RECIPE_SCHEMA!r}")
    arch = document.get("model_arch")
    if not isinstance(arch, str) or not arch:
        raise RecipeRefused("recipe-schema", "model_arch is required")
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
    for seam, contract in dict(requires.get("seams", {})).items():
        spec = plugin_host.SEAMS.get(seam)
        if spec is None:
            raise RecipeRefused("recipe-unknown-seam", f"requires.seams: {seam}")
        if contract != spec.contract:
            raise RecipeRefused(
                "recipe-contract-mismatch",
                f"{seam}: recipe {contract} != ATOM {spec.contract}",
            )

    raw_plugins = document.get("plugins", [])
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
        _names(raw.get("serves", []), f"{name}.serves")
        plugins[name] = raw

    plan = document.get("plan", {})
    seams = plan.get("seams", {}) if isinstance(plan, Mapping) else None
    if not isinstance(seams, Mapping):
        raise RecipeRefused("recipe-schema", "plan.seams must map seam -> owners")
    defaults = document.get("defaults", {})
    if not isinstance(defaults, Mapping):
        raise RecipeRefused("recipe-schema", "defaults must be an object")
    allow_partition = bool(defaults.get("allow_partition", False))
    shadow_policy = defaults.get("shadow_policy", "error")
    on_decline = defaults.get("on_decline", "native")
    if shadow_policy not in SHADOW or on_decline not in ON_DECLINE:
        raise RecipeRefused("recipe-schema", "shadow_policy / on_decline")

    table: dict[str, tuple[str, ...]] = {}
    for seam, owners in seams.items():
        if seam not in plugin_host.SEAMS:
            raise RecipeRefused("recipe-unknown-seam", seam)
        owners = _names(owners, f"plan.seams[{seam}]")
        if len(set(owners)) != len(owners):
            raise RecipeRefused("recipe-schema", f"{seam}: an owner twice")
        for owner in owners:
            ref = plugins.get(owner)
            if ref is None:
                raise RecipeRefused("recipe-unknown-provider", f"{seam}: {owner}")
            if ref.get("kind", "layer") != "layer":
                raise RecipeRefused("recipe-conflict", f"{owner} is an AITER provider")
            if seam not in ref.get("serves", []):
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

    shadowed: list[tuple[str, str, str]] = []
    for child, owners in table.items():
        parent = plugin_host.SEAMS[child].parent
        if parent in table:
            for owner in owners:
                if owner in table[parent]:
                    continue
                if shadow_policy == "error":
                    raise RecipeRefused(
                        "recipe-shadow",
                        f"{parent} owner {table[parent][0]} shadows {child} owner {owner}",
                    )
                shadowed.append((child, owner, table[parent][0]))

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
        "shadowed": shadowed,
        "arms": {k: tuple(v) for k, v in arms.items()},
        "strict_overlap": bool(defaults.get("strict_overlap", False)),
        "on_decline": on_decline,
        "expect": dict(expect),
        "requires": dict(requires),
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
        for field in ("model_arch", "strategy", "on_decline", "strict_overlap"):
            if other[field] != first[field]:
                raise RecipeRefused("recipe-set-conflict", f"{field} differs")
        if other["expect"] != first["expect"]:
            raise RecipeRefused("recipe-set-conflict", "runtime.expect differs")
    tables: dict[Any, Mapping[str, tuple[str, ...]]] = {}
    for recipe in parsed:
        keys = recipe["buckets"] or (recipe["mode"],)
        for key in keys:
            clash = key in tables or (
                key in ("any", "eager") and {"any", "eager"} & set(tables)
            )
            if clash:
                raise RecipeRefused("recipe-set-conflict", f"two recipes serve {key!r}")
            tables[key] = MappingProxyType(dict(recipe["table"]))

    providers: dict[str, None] = {}
    aiter: dict[str, None] = {}
    arms: dict[str, tuple[str, ...]] = {}
    for recipe in parsed:
        for name, ref in recipe["plugins"].items():
            if ref.get("kind", "layer") == "aiter":
                aiter[name] = None
        for table_owners in recipe["table"].values():
            providers.update(dict.fromkeys(table_owners))
        for arm, members in recipe["arms"].items():
            if arm in arms and arms[arm] != members:
                raise RecipeRefused("recipe-set-conflict", f"arm {arm} differs")
            arms[arm] = members
    everyone = (*providers, *aiter)
    arms.setdefault("native", ())
    arms.setdefault("plugin", everyone)
    arms.setdefault("identity", everyone)
    digests = tuple(sorted(r["digest"] for r in parsed))
    contracts = {
        seam: plugin_host.SEAMS[seam].contract
        for table in tables.values()
        for seam in table
    }
    shadowed = tuple(s for r in parsed for s in r["shadowed"])
    plan_id = _digest(
        [digests, plugin_host.HOOK_API, contracts, sorted(arms), list(providers)]
    )
    return Plan(
        plan_id=plan_id,
        strategy=first["strategy"],
        model_arch=first["model_arch"],
        tables=MappingProxyType(tables),
        providers=tuple(providers),
        aiter=tuple(aiter),
        arms=MappingProxyType(arms),
        shadowed=shadowed,
        strict_overlap=first["strict_overlap"],
        on_decline=first["on_decline"],
        expect=MappingProxyType(first["expect"]),
        requires=MappingProxyType(first["requires"]),
        recipe_digests=digests,
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
    return RecipeSource(documents, names, plan.plan_id, plan.empty)


def check_runtime(plan: Plan, config: Any) -> None:
    """``runtime.expect``: the server must be the one the recipe was made for."""
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
        "mono": lambda c: _mono_enabled(),
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


def _mono_enabled() -> bool:
    from atom.utils import envs

    return bool(envs.ATOM_MONO_ENABLE)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m atom.plugin_recipe")
    parser.add_argument("path", help="recipe file or a strategy's recipe directory")
    parser.add_argument(
        "--installed",
        action="append",
        default=[],
        help="name=seam,seam: what an installed plugin registers (drift check)",
    )
    args = parser.parse_args(argv)
    try:
        plan = load(args.path)
        for item in args.installed:
            name, _, seams = item.partition("=")
            served = set(filter(None, seams.split(",")))
            for table in plan.tables.values():
                for seam, owners in (table or {}).items():
                    if name in owners and seam not in served:
                        raise RecipeRefused(
                            "recipe-plugin-drift", f"installed {name} lacks {seam}"
                        )
    except RecipeRefused as refused:
        print(f"REFUSED {refused}")
        return 2
    print(json.dumps(plan.dump(), indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
