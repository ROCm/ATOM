# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Plugin host: named model seams that enabled plugins may serve (HOOK_API 2).

A seam is an op ATOM calls directly (``v41.attn.qk_norm``) or a region of
native code whose consumer stays ATOM's (``v41.attn.out``: wo_a, whose output
wo_b reads). ATOM declares them in ``SEAMS`` and calls them either through a
``bind`` made at model build (hot paths) or through ``call`` (module-level
functions). Ops AITER routes itself stay on ``aiter.plugin``; one plan drives
both halves.

Rules:
- Default deny. Installing a plugin makes it available; only a plan
  (``--plugin-recipe`` / ``--plugin-enable``) enables it. ``ATOM_PLUGIN_HOST=0``
  only disables; ``ATOM_PLUGIN_HOST_STRICT=1`` only adds checks.
- Native is untouched: a seam no plan owner registered binds to the native
  object itself, and a ``call`` costs one dict lookup.
- A seam may have ordered owners for disjoint cells. Their ``select`` is a
  pure metadata predicate, asked once per route-cache key (the seam's key
  inputs, the call's keyword arguments and the graph state); the first that
  accepts serves. Strict mode asks them all and refuses two accepts. Once
  ``impl`` runs its errors propagate; native is never retried.
- What ATOM knows about a call reaches a plugin only as a ``Capability``; every
  impl also gets ``valid_rows``, the real row count on the device (pad-M), or
  None when all rows are real. A decision made inside CUDA-graph capture is the
  capture's: replays reuse it for any real row count up to the bucket.
- Traced Python (``torch.compiler.is_compiling()``) is native; a seam inside a
  compiled region with an owner runs as one opaque op (``atom_plugin::span``)
  that decides on every execution.
- Decisions, not launches, are counted: a CUDA-graph replay runs no Python.
"""

from __future__ import annotations

import functools
import logging
import sys
import threading
import time
from collections import Counter
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from importlib import metadata
from types import MappingProxyType
from typing import Any

import torch

logger = logging.getLogger("atom")

HOOK_API = 2
ENTRY_POINT_GROUP = "atom.layer_plugins"
AITER_SOURCE = "atom plugin plan"
GRAPHS = ("eager", "capture_safe")  # what a plugin may claim
MX = "mxfp8_e8m0_g32"  # e4m3 values, row-major group-32 e8m0 scales


@dataclass(frozen=True)
class SeamSpec:
    """One seam: its contract and how ATOM reaches it.

    ``inputs``/``kwargs`` are what the native and the impl both receive;
    ``returns`` names the result (one value for one name, nothing checked for
    none: in place); ``key`` the inputs whose layout (shape, dtype, strides,
    16-byte alignment; a scalar's value) keys the route cache; ``graph``
    where it runs (``eager``, ``capture_safe`` or ``opaque_op``: inside a
    compiled region, as one opaque op); ``parent`` the region it is nested in.
    A select may tell calls apart only by key inputs, keyword arguments and
    the capability; an unkeyed input is a per-call-site constant (weights) or
    a per-step index that must not decide the route.
    """

    name: str
    contract: int
    kind: str  # "op" | "region"
    form: str  # "bind" | "call"
    inputs: tuple[str, ...]
    kwargs: tuple[str, ...]
    returns: tuple[str, ...]
    key: tuple[str, ...]
    graph: str
    target: str
    parent: str | None = None
    boundary: str = "tensor"
    doc: str = ""


def _seam(
    name, contract, kind, form, inputs, kwargs, returns, key, graph, target, **kw
):
    return name, SeamSpec(
        name, contract, kind, form, inputs, kwargs, returns, key, graph, target, **kw
    )


_V41 = "atom/models/deepseek_v41/"
SEAMS = MappingProxyType(
    dict(
        [
            _seam(
                "v41.block.ffn",
                1,
                "region",
                "bind",
                ("x",),
                (),
                ("y",),
                ("x",),
                "opaque_op",
                _V41 + "model.py",
                doc="the FFN sublayer, Block.ffn(normed): bf16 [T,H] -> bf16 [T,H]",
            ),
            _seam(
                "v41.attn.out",
                1,
                "region",
                "bind",
                ("x_fp8", "x_scale", "wo_a_w", "wo_a_ws"),
                ("dtype",),
                ("y", "y_scale"),
                ("x_fp8", "x_scale"),
                "capture_safe",
                _V41 + "attention.py",
                doc="wo_a: wo_b's input pair (e4m3 + e8m0 g32) or (bf16, None)",
            ),
            _seam(
                "v41.attn.qk_norm",
                1,
                "op",
                "bind",
                ("q", "k", "q_weight", "k_weight", "eps_q"),
                ("eps_k", "amax_floor"),
                ("q_fp8", "q_scale", "k_out"),
                ("q", "k", "eps_q"),
                "capture_safe",
                "atom/model_ops/layernorm.py",
                doc="dual RMSNorm, q to MXFP8: the kernel DualRMSNormMXFP8 calls",
            ),
            _seam(
                "v41.moe.shared_act",
                1,
                "region",
                "bind",
                ("combined", "weights"),
                (),
                ("act", "act_scale"),
                ("combined", "weights"),
                "opaque_op",
                "atom/models/deepseek_v4/__init__.py",
                parent="v41.block.ffn",
                doc="shared expert clamp-act(+quant): w2's input pair or (bf16, None)",
            ),
            _seam(
                "v41.moe.topk",
                1,
                "op",
                "call",
                (
                    "topk_weights",
                    "topk_ids",
                    "router_logits",
                    "e_score_correction_bias",
                    "renormalize",
                    "route_scale",
                ),
                ("score_func", "num_valid_rows"),
                (),
                (
                    "router_logits",
                    "topk_weights",
                    "topk_ids",
                    "e_score_correction_bias",
                    "renormalize",
                    "route_scale",
                ),
                "capture_safe",
                "atom/model_ops/moe.py",
                parent="v41.block.ffn",
                doc="topk_gating in FusedMoE.select_experts, in place",
            ),
            _seam(
                "v41.op.quant_fp8_g32",
                1,
                "op",
                "call",
                ("x",),
                ("dequantize",),
                ("x_fp8", "x_scale"),
                ("x",),
                "capture_safe",
                "atom/model_ops/blockscale.py",
                doc="quantize_fp8: e4m3 + e8m0 [..., K/32]; dequantize=True is QAT",
            ),
            _seam(
                "v41.attn.index_topk",
                1,
                "op",
                "call",
                ("scores", "group", "row_ends", "out", "count", "stride0", "stride1"),
                ("k", "stable", "row_starts", "plane_width"),
                (),
                ("scores", "group", "row_ends", "out", "count", "stride0", "stride1"),
                "capture_safe",
                "atom/model_ops/indexer/score.py",
                doc="packed top_k_per_row_decode, in place into out",
            ),
            _seam(
                "v41.attn.attend",
                1,
                "op",
                "call",
                (
                    "q",
                    "unified_kv",
                    "kv_indices_prefix",
                    "kv_indptr_prefix",
                    "kv",
                    "kv_indices_extend",
                    "kv_indptr_extend",
                    "attn_sink",
                    "softmax_scale",
                ),
                (
                    "out",
                    "unified_kv_rope",
                    "q_packed",
                    "q_rope",
                    "k_packed",
                    "k_rope",
                    "prefix",
                ),
                ("out",),
                ("q", "unified_kv", "kv", "attn_sink", "softmax_scale"),
                "capture_safe",
                "atom/model_ops/v4_kernels/paged_2src.py",
                doc="sparse_attn_v4_paged_2src; out= is written and returned",
            ),
            _seam(
                "v41.draft.ctx",
                1,
                "region",
                "bind",
                ("taps", "rows", "running_tokens"),
                (),
                ("aux", "aux_scale"),
                ("taps", "rows", "running_tokens"),
                "eager",
                _V41 + "dspark.py",
                doc="DSpark context taps -> main_proj's input pair or (bf16, None)",
            ),
        ]
    )
)


@dataclass(frozen=True)
class Capability:
    """What ATOM tells a plugin about a call; it replaces private flags."""

    seam: str
    quant: str | None = None
    rows: int = 0  # padded leading dim of the first key input (M)
    captured: bool = False  # decided inside CUDA-graph capture
    bucket: int | None = None  # the captured pass's padded rows
    flags: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))


@dataclass(frozen=True)
class Registration:
    seam: str
    owner: str
    accepted: bool
    reason: str


@dataclass(frozen=True)
class Decision:
    seam: str
    route: str  # "plugin" | "native"
    reason: str  # selected, declined, select-error, graph-unsafe, identity, ...
    owner: str | None = None
    declined: int = 0  # owners asked before it (all of them, when native)
    captured: bool = False
    bucket: int | None = None


class RegistrationRefused(ValueError):
    """A registration that can never be valid; ``reason`` is its short code."""

    def __init__(self, reason: str, detail: str):
        super().__init__(f"{reason}: {detail}")
        self.reason = reason


class PluginConflict(RuntimeError):
    """Strict mode: two owners accepted the same call."""


class PluginContractError(RuntimeError):
    """An owner's result does not have its seam's declared shape."""


class PluginDeclined(RuntimeError):
    """``on_decline: error``: every owner of a planned seam declined a call."""


@dataclass(frozen=True)
class _Route:
    owner: str
    impl: Callable[..., Any]
    select: Callable[..., Any] | None
    graph: str


_lock = threading.RLock()
_plan = None  # atom.plugin_recipe.Plan
_requested: str | None = None  # the plan id configure was asked for
_kill_switch = False
_strict = False
_on_decline = "native"
_arm = "plugin"
_loading: str | None = None  # the entry point whose register() is running
_registered: dict[str, dict[str, _Route]] = {}  # seam -> owner -> route
_tables: dict[Any, dict[str, tuple[_Route, ...]]] = {}  # table -> seam -> routes
_active: dict[Any, dict[str, tuple[_Route, ...]]] = {}  # the same, for the arm
_owned: frozenset[str] = frozenset()  # seams with an owner in some table
_bound: set[str] = set()
_cache: dict[tuple, tuple[Decision, _Route | None, Capability]] = {}
_CACHE_LIMIT = 4096
_counts: Counter = Counter()
_refusals: list[Registration] = []
_last: Decision | None = None


def api_satisfies(requires: str, api: int = HOOK_API) -> bool:
    """``">=2,<3"``-style clauses (``>=``, ``<``, ``==``) against ``api``."""
    for clause in (c.strip() for c in str(requires).split(",") if c.strip()):
        for op, test in ((">=", api.__ge__), ("==", api.__eq__), ("<", api.__lt__)):
            if clause.startswith(op):
                try:
                    if not test(int(clause[len(op) :])):
                        return False
                except ValueError:
                    return False
                break
        else:
            return False
    return True


def _refuse(seam, owner, reason, detail):
    _refusals.append(Registration(seam, owner, False, reason))
    logger.warning("plugin_host: %s refused for %s: %s", owner, seam, detail)
    raise RegistrationRefused(reason, detail)


def _planned(seam: str, owner: str) -> bool:
    if _plan is None:
        return False
    for table in _plan.tables.values():
        if table is None or owner in table.get(seam, ()):
            return True
    return False


def register(
    *,
    seam: str,
    contract: int,
    impl: Callable[..., Any],
    owner: str,
    select: Callable[..., Any] | None = None,
    graph: str = "capture_safe",
    requires_api: str = f">={HOOK_API},<{HOOK_API + 1}",
) -> Registration:
    """Offer ``owner`` for ``seam``. Refusals that are bugs raise; an owner the
    plan does not enable here gets ``accepted=False``."""
    with _lock:
        spec = SEAMS.get(seam)
        if spec is None:
            _refuse(seam, owner, "unknown-seam", f"hook API {HOOK_API} has no {seam}")
        if contract != spec.contract:
            _refuse(seam, owner, "contract-mismatch", f"{contract} != {spec.contract}")
        if not api_satisfies(requires_api):
            _refuse(seam, owner, "api-mismatch", f"{requires_api} vs {HOOK_API}")
        if graph not in GRAPHS:
            _refuse(seam, owner, "graph-unsupported", f"{graph!r} not in {GRAPHS}")
        if not callable(impl) or (select is not None and not callable(select)):
            _refuse(seam, owner, "not-callable", "impl and select must be callable")
        if _loading is not None and owner != _loading:
            _refuse(seam, owner, "owner-mismatch", f"loading entry point {_loading}")
        if owner in _registered.get(seam, {}):
            _refuse(seam, owner, "duplicate", f"{owner} already serves {seam}")
        if _kill_switch or _plan is None or owner not in _plan.providers:
            reason = "kill-switch" if _kill_switch else "not-enabled"
            _refusals.append(Registration(seam, owner, False, reason))
            return _refusals[-1]
        if not _planned(seam, owner):
            _refusals.append(Registration(seam, owner, False, "not-planned"))
            return _refusals[-1]
        _registered.setdefault(seam, {})[owner] = _Route(owner, impl, select, graph)
        return Registration(seam, owner, True, "registered")


def _entry_points() -> dict[str, metadata.EntryPoint]:
    return {ep.name: ep for ep in metadata.entry_points(group=ENTRY_POINT_GROUP)}


def configure(
    plan=None,
    *,
    arm: str = "plugin",
    config: Any = None,
    kill_switch: bool | None = None,
    strict: bool | None = None,
) -> None:
    """Load and route ``plan`` (a ``plugin_recipe.Plan``, the
    ``RecipeSource`` a Config carries, or names for the ``--plugin-enable``
    shorthand) before the model is built.

    A recipe plan is all or nothing: a plugin that fails to load, or a planned
    owner that did not register, refuses the start. Under the shorthand a
    failing plugin is logged and skipped. The same plan's AITER providers go to
    ``aiter.plugin.configure`` when AITER has it.
    """
    from atom import plugin_recipe
    from atom.utils import envs

    global _plan, _kill_switch, _strict, _on_decline, _loading, _owned, _requested
    if isinstance(plan, plugin_recipe.RecipeSource):
        plan = plan.resolve()  # Config carries the recipes; each process resolves
    elif not isinstance(plan, (plugin_recipe.Plan, type(None))):
        plan = plugin_recipe.shorthand(plan)
    requested = None if plan is None else plan.plan_id
    if _bound:
        # A second runner in this process may restate the plan the built binds
        # were made with; any other plan would leave them on the old one.
        if requested != _requested:
            raise RuntimeError("plugin_host.configure after model build; use set_arm")
        set_arm(arm if _plan is not None else "plugin")
        return
    _requested = requested
    if plan is not None and config is not None:
        plugin_recipe.check_runtime(plan, config)
    kill = not envs.ATOM_PLUGIN_HOST if kill_switch is None else kill_switch
    with _lock:
        _plan = None if kill else plan
        _kill_switch = kill
        _strict = (
            envs.ATOM_PLUGIN_HOST_STRICT or bool(plan and plan.strict_overlap)
            if strict is None
            else strict
        )
        _on_decline = plan.on_decline if plan is not None else "native"
        _registered.clear()
        _tables.clear()
        _refusals.clear()
        _spans.clear()
        shorthand = plan is not None and None in plan.tables.values()
        if _plan is not None:
            available = _entry_points()
            for name in _plan.providers:
                if name not in available:
                    if not shorthand:
                        raise plugin_recipe.RecipeRefused(
                            "recipe-plugin-drift", f"{name} is not installed"
                        )
                    continue  # an AITER-only provider, or not installed
                _loading = name
                try:
                    available[name].load()(sys.modules[__name__])
                except Exception as error:
                    _refusals.append(Registration("*", name, False, "load-error"))
                    logger.warning("plugin_host: %s failed to load: %s", name, error)
                    if not shorthand:
                        raise plugin_recipe.RecipeRefused(
                            "recipe-plugin-drift", f"{name} failed to load: {error}"
                        ) from error
                finally:
                    _loading = None
            _tables.update(_route_tables(_plan, shorthand))
        _owned = frozenset(s for table in _tables.values() for s in table)
        set_arm(arm if _plan is not None else "plugin")
    _configure_aiter(() if kill or plan is None else plan.aiter)
    if plan is not None:
        logger.info("plugin_host: %s", status())


def _route_tables(plan, shorthand: bool) -> dict[Any, dict[str, tuple[_Route, ...]]]:
    from atom import plugin_recipe

    if shorthand:
        order = {name: i for i, name in enumerate(plan.providers)}
        return {
            "any": {
                seam: tuple(sorted(routes.values(), key=lambda r: order[r.owner]))
                for seam, routes in _registered.items()
            }
        }
    tables = {}
    for key, table in plan.tables.items():
        tables[key] = {}
        for seam, owners in table.items():
            missing = [o for o in owners if o not in _registered.get(seam, {})]
            if missing:
                raise plugin_recipe.RecipeRefused(
                    "recipe-coverage-miss", f"{seam}: {missing} did not register"
                )
            tables[key][seam] = tuple(_registered[seam][o] for o in owners)
    return tables


def set_arm(arm: str) -> None:
    """Which planned providers serve: ``plugin`` (all), ``native`` (none, the
    plugin topology kept), ``identity`` (decide as plugin, run native) or a
    recipe arm. Only the route table changes; binds and opaque ops stay."""
    global _arm, _active
    with _lock:
        arms = _plan.arms if _plan is not None else {}
        if arm not in ("plugin", "native", "identity") and arm not in arms:
            raise ValueError(f"unknown arm {arm!r}")
        allowed = None if arm in ("plugin", "identity") else set(arms.get(arm, ()))
        _active = {
            key: {
                seam: tuple(r for r in routes if allowed is None or r.owner in allowed)
                for seam, routes in table.items()
            }
            for key, table in _tables.items()
        }
        _arm = arm
        _cache.clear()


# -- dispatch -----------------------------------------------------------------


def _graph_state() -> tuple[bool, int | None]:
    """``(captured, bucket)``: inside a CUDA-graph capture, the pass's padded
    rows (``running_tokens``). A target decode graph of batch ``bs`` runs
    ``bs * q_len`` rows (a DSpark verify: ``bs * (1 + k)``); a draft pass,
    recorded by its own graph outside ``in_hipgraph``, ``bs`` times its
    tokens per sequence."""
    from atom.utils.forward_context import get_forward_context

    try:
        forward = get_forward_context()
    except Exception:  # noqa: BLE001 -- outside an ATOM forward
        return False, None
    if not (getattr(forward, "in_hipgraph", False) or _stream_capturing()):
        return False, None
    return True, getattr(getattr(forward, "context", None), "running_tokens", None)


def _stream_capturing() -> bool:
    try:
        return torch.cuda.is_initialized() and torch.cuda.is_current_stream_capturing()
    except Exception:  # noqa: BLE001 -- a build without CUDA
        return False


def _valid_rows(rows: int) -> torch.Tensor | None:
    from atom.utils.forward_context import pass_scheduled_tokens_gpu

    try:
        return pass_scheduled_tokens_gpu(rows)
    except Exception:  # noqa: BLE001 -- outside an ATOM forward: all rows real
        return None


def _routes(seam: str, captured: bool, bucket: int | None) -> tuple[_Route, ...]:
    if captured:
        table = _active.get(bucket, _active.get("any"))
    else:
        table = _active.get("eager", _active.get("any"))
    return () if table is None else table.get(seam, ())


_warned_misses: set[tuple[str, int | None]] = set()


def _no_table(seam: str, captured: bool, bucket: int | None) -> None:
    """A seam with owners elsewhere has none for this pass: never silent."""
    _record(Decision(seam, "native", "no-table", captured=captured, bucket=bucket))
    if _strict and captured and (seam, bucket) not in _warned_misses:
        declared = _declared_rows()
        if declared:
            _warned_misses.add((seam, bucket))
            logger.warning(
                "plugin_host: %s has no table for captured rows %s (the plan "
                "declares rows %s); it runs native in that graph",
                seam,
                bucket,
                declared,
            )


def _declared_rows() -> list[int]:
    """The capture-bucket keys (padded rows) the plan's tables declare."""
    return sorted(k for k in _tables if isinstance(k, int))


def _key_of(value: Any) -> Any:
    """A key input's part of the route-cache key: what a select may read."""
    if isinstance(value, torch.Tensor):
        # Layout, not just shape: a sliced view has the same shape and other
        # strides, and a kernel's select may check strides and alignment.
        return (
            tuple(value.shape),
            value.dtype,
            value.stride(),
            value.data_ptr() % 16 if value.device.type != "meta" else None,
        )
    shape = getattr(value, "shape", None)
    if shape is not None:
        return tuple(shape), getattr(value, "dtype", None)
    if isinstance(value, (list, tuple)):
        return tuple(_key_of(v) for v in value)
    if value is None or isinstance(value, (bool, int, float, str, torch.dtype)):
        return value
    return type(value).__name__


def _rows_of(value: Any) -> int:
    if isinstance(value, (list, tuple)):
        value = value[0] if value else None
    shape = getattr(value, "shape", None)
    return int(shape[0]) if shape else 0


def _decide(seam, routes, cap, inputs, kwargs, strict) -> tuple[Decision, Any]:
    chosen, declined, errors, unsafe = None, 0, 0, 0
    for route in routes:
        if cap.captured and route.graph != "capture_safe":
            unsafe += 1
            declined += chosen is None
            continue
        answer = True
        if route.select is not None:
            try:
                answer = bool(route.select(cap, *inputs, **kwargs))
            except Exception as error:  # noqa: BLE001 -- a broken select declines
                logger.warning("plugin_host: %s select failed: %s", route.owner, error)
                answer, errors = False, errors + 1
        if not answer:
            declined += chosen is None
            continue
        if chosen is not None:
            raise PluginConflict(
                f"{seam}: {chosen.owner} and {route.owner} both accept this call"
            )
        chosen = route
        if not strict:
            break
    where = {"captured": cap.captured, "bucket": cap.bucket}
    if chosen is None:
        reason = "select-error" if errors else "declined"
        if unsafe and unsafe == declined and not errors:
            reason = "graph-unsafe"
        return Decision(seam, "native", reason, None, declined, **where), None
    return Decision(seam, "plugin", "selected", chosen.owner, declined, **where), chosen


def _record(decision: Decision) -> None:
    global _last
    _last = decision
    _counts[(decision.seam, decision.route, decision.reason, decision.owner)] += 1


def _record_traced(ctx) -> None:
    # Runs once, while Dynamo traces `_dispatch`, reading its own local.
    seam = ctx.get_local("seam").as_python_constant()
    _record(Decision(seam, "native", "traced"))


try:
    from torch._dynamo.comptime import comptime as _comptime
except Exception:  # noqa: BLE001 -- recording a traced decision is best effort
    _comptime = None


def _dispatch(seam, native, capability, inputs, kwargs):
    if torch.compiler.is_compiling():
        if _comptime is not None:
            _comptime(_record_traced)
        return native(*inputs, **kwargs)
    if _arm == "native":
        _record(Decision(seam, "native", "arm-native"))
        return native(*inputs, **kwargs)
    captured, bucket = _graph_state()
    routes = _routes(seam, captured, bucket)
    if not routes:
        _no_table(seam, captured, bucket)
        return native(*inputs, **kwargs)
    spec = SEAMS[seam]
    named = dict(zip(spec.inputs, inputs))
    key = (
        seam,
        captured,
        bucket,
        tuple(_key_of(named.get(k)) for k in spec.key),
        tuple((k, _key_of(v)) for k, v in sorted(kwargs.items())),
    )
    hit = _cache.get(key)
    if hit is None:
        static = capability(*inputs, **kwargs) if callable(capability) else capability
        static = static or {}
        cap = Capability(
            seam,
            quant=static.get("quant"),
            rows=_rows_of(named.get(spec.key[0])) if spec.key else 0,
            captured=captured,
            bucket=bucket,
            flags=MappingProxyType(dict(static.get("flags", {}))),
        )
        decision, route = _decide(seam, routes, cap, inputs, kwargs, _strict)
        if len(_cache) >= _CACHE_LIMIT:  # keys with per-step values grow it
            _cache.clear()
        hit = _cache[key] = (decision, route, cap)
    decision, route, cap = hit
    if route is None:
        _record(decision)
        if _on_decline == "error":
            raise PluginDeclined(f"{seam}: every owner declined ({decision.reason})")
        return native(*inputs, **kwargs)
    if _arm == "identity":
        _record(Decision(seam, "plugin", "identity", route.owner, decision.declined))
        return native(*inputs, **kwargs)
    _record(decision)
    result = route.impl(
        *inputs, capability=cap, valid_rows=_valid_rows(cap.rows), **kwargs
    )
    arity = len(spec.returns)
    if arity > 1 and not (isinstance(result, tuple) and len(result) == arity):
        raise PluginContractError(
            f"{route.owner} returned {type(result).__name__} for {seam}, "
            f"expected {spec.returns}"
        )
    return result


class Router:
    """A bound seam that has a planned owner (``bind`` returns native else)."""

    __slots__ = ("capability", "native", "seam")

    def __init__(self, seam, native, capability):
        self.seam, self.native, self.capability = seam, native, capability

    def __call__(self, *inputs: Any, **kwargs: Any) -> Any:
        return _dispatch(self.seam, self.native, self.capability, inputs, kwargs)

    def explain(self, *inputs: Any, **kwargs: Any) -> Decision:
        return explain(self.seam, *inputs, _capability=self.capability, **kwargs)


def bind(
    seam: str,
    native: Callable[..., Any],
    *,
    capability: Mapping[str, Any] | Callable[..., Any] | None = None,
    layer: str | None = None,
    fake: Callable[..., torch.Tensor] | None = None,
) -> Callable[..., Any]:
    """At model build (after ``configure``): how the call site runs ``seam``.

    No planned owner: ``native`` itself, so the call site is today's call.
    Else a ``Router``; an ``opaque_op`` seam given ``layer`` and ``fake`` is
    wrapped in one opaque op so a compiled graph decides per execution.
    ``capability`` is ``{"quant": ..., "flags": {...}}`` or a callable taking
    the call's inputs and returning one.
    """
    spec = SEAMS.get(seam)
    if spec is None or spec.form != "bind":
        raise KeyError(f"hook API {HOOK_API} has no bound seam {seam!r}")
    _bound.add(seam)
    if seam not in _owned:
        return native
    router = Router(seam, native, capability)
    if spec.graph == "opaque_op" and layer is not None and fake is not None:
        return span_op(
            seam, layer, lambda x, extra: router(x), lambda x, extra: fake(x)
        )
    return router


_call_capability: dict[str, Callable[..., Mapping[str, Any]]] = {
    "v41.moe.topk": lambda *a, score_func=None, **kw: {
        "flags": {"score_func": score_func}
    },
    "v41.op.quant_fp8_g32": lambda *a, **kw: {"quant": MX},
    "v41.attn.index_topk": lambda *a, **kw: {"flags": {"packed": True}},
    "v41.attn.attend": lambda *a, unified_kv_rope=None, **kw: {
        "flags": {"kv_layout": "bf16" if unified_kv_rope is None else "fp8_2buff"}
    },
}


def call(seam: str, native: Callable[..., Any], *inputs: Any, **kwargs: Any) -> Any:
    """``native(*inputs, **kwargs)``, unless a planned owner of ``seam`` serves."""
    if seam not in _owned:
        return native(*inputs, **kwargs)
    return _dispatch(seam, native, _call_capability.get(seam), inputs, kwargs)


def site(name: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Declare a whole module-level function as the native of call seam ``name``."""
    spec = SEAMS.get(name)
    if spec is None or spec.form != "call":
        raise KeyError(f"hook API {HOOK_API} has no call seam {name!r}")

    def decorate(native):
        @functools.wraps(native)
        def wrapper(*args, **kwargs):
            return call(name, native, *args, **kwargs)

        return wrapper

    return decorate


# -- opaque op ----------------------------------------------------------------

# Seam spans a compiled graph contains: layer -> (run, fake). See `span_op`.
_spans: dict[str, tuple[Callable[..., Any], Callable[..., Any]]] = {}
SPAN_OP = "atom_plugin::span"


def _span(x: torch.Tensor, extra: torch.Tensor | None, layer: str) -> torch.Tensor:
    # Through sys.modules: the op outlives a re-import of this module.
    return sys.modules[__name__]._spans[layer][0](x, extra)


def _span_fake(x: torch.Tensor, extra: torch.Tensor | None, layer: str) -> torch.Tensor:
    return sys.modules[__name__]._spans[layer][1](x, extra)


def _define_span_op() -> None:
    namespace, name = SPAN_OP.split("::")
    if hasattr(getattr(torch.ops, namespace), name):
        return
    library = torch.library.Library(namespace, "FRAGMENT")
    schema = torch.library.infer_schema(_span, mutates_args=())
    library.define(name + schema)
    library.impl(name, _span, "CompositeExplicitAutograd")
    library._register_fake(name, _span_fake)
    _define_span_op.library = library  # the registration lives as long as it


def span_op(
    seam: str,
    layer: str,
    run: Callable[..., torch.Tensor],
    fake: Callable[..., torch.Tensor],
) -> Callable[..., torch.Tensor]:
    """At model build: ``run(x, extra)`` as a compiled graph should see it.

    No planned owner of ``seam``: ``run`` itself, inline, the graph ATOM has
    without plugins. Else one opaque op (``atom_plugin::span``, shapes from
    ``fake``) whose body runs ``run`` eagerly, where the seam's owners are
    asked on every execution.
    """
    _bound.add(seam)
    if seam not in _owned:
        return run
    if layer in _spans:
        raise ValueError(f"span layer name {layer!r} is not unique")
    _define_span_op()
    _spans[layer] = (run, fake)
    op = torch.ops.atom_plugin.span.default
    return lambda x, extra=None: op(x, extra, layer)


# -- diagnostics --------------------------------------------------------------


def explain(seam: str, *inputs: Any, captured=False, bucket=None, **kwargs) -> Decision:
    """What a call would decide now, without running a kernel or recording."""
    capability = kwargs.pop("_capability", _call_capability.get(seam))
    routes = _routes(seam, captured, bucket)
    if not routes:
        reason = "kill-switch" if _kill_switch else "no-owner"
        return Decision(seam, "native", reason, captured=captured, bucket=bucket)
    spec = SEAMS[seam]
    named = dict(zip(spec.inputs, inputs))
    static = (
        capability(*inputs, **kwargs) if callable(capability) else capability
    ) or {}
    cap = Capability(
        seam,
        quant=static.get("quant"),
        rows=_rows_of(named.get(spec.key[0])) if spec.key else 0,
        captured=captured,
        bucket=bucket,
        flags=MappingProxyType(dict(static.get("flags", {}))),
    )
    return _decide(seam, routes, cap, inputs, kwargs, _strict)[0]


def owners(seam: str, table: Any = "eager") -> tuple[str, ...]:
    """The order a seam's owners are asked in, for a table (``eager``,
    ``any`` or a capture bucket) under the current arm."""
    if table == "eager":
        routes = _routes(seam, False, None)
    elif table == "any":
        routes = _active.get("any", {}).get(seam, ())
    else:
        routes = _routes(seam, True, table)
    return tuple(r.owner for r in routes)


def last_decision() -> Decision | None:
    return _last


def decision_counts() -> Counter:
    """Decisions so far, keyed ``(seam, route, reason, owner)``."""
    return Counter(_counts)


def route_proof(seam: str, owner: str, n: int = 1) -> int:
    """Assert ``owner`` served ``seam`` at least ``n`` times; the count."""
    served = _counts[(seam, "plugin", "selected", owner)]
    if served < n:
        raise AssertionError(f"{seam}: {owner} served {served} < {n} calls")
    return served


def status() -> dict[str, Any]:
    return {
        "hook_api": HOOK_API,
        "plan_id": None if _plan is None else _plan.plan_id,
        "strategy": None if _plan is None else _plan.strategy,
        "arm": _arm,
        "kill_switch": _kill_switch,
        "strict": _strict,
        "on_decline": _on_decline,
        "seams": {
            seam: {
                "tables": {
                    str(key): tuple(r.owner for r in table.get(seam, ()))
                    for key, table in _active.items()
                    if seam in table
                }
            }
            for seam in sorted(_owned)
        },
        # A captured pass routes by its padded rows; these are the ones planned.
        "capture_rows": _declared_rows(),
        "span_ops": tuple(_spans),
        "bound": tuple(sorted(_bound)),
        "refusals": tuple(_refusals),
    }


def overhead_probe(
    seam: str, native: Callable[..., Any], inputs: tuple, n: int = 10000
) -> dict[str, float]:
    """µs per call: native direct, and through a Router on the native and
    plugin arms (benchmark use; restores the arm)."""

    def timed(fn) -> float:
        start = time.perf_counter()
        for _ in range(n):
            fn()
        return (time.perf_counter() - start) * 1e6 / n

    router = Router(seam, native, _call_capability.get(seam))
    previous = _arm
    result = {"native_direct_us": timed(lambda: native(*inputs))}
    try:
        for arm in ("native", "plugin"):
            set_arm(arm)
            result[f"router_{arm}_arm_us"] = timed(lambda: router(*inputs))
    finally:
        set_arm(previous)
    return result


def _reset_for_testing() -> None:
    """Forget the plan, the binds and the record (tests only)."""
    global _plan, _owned, _last, _kill_switch, _strict, _on_decline, _requested
    with _lock:
        _bound.clear()
        _registered.clear()
        _tables.clear()
        _spans.clear()
        _refusals.clear()
        _counts.clear()
        _warned_misses.clear()
        _plan, _owned, _last, _requested = None, frozenset(), None, None
        _kill_switch, _strict, _on_decline = False, False, "native"
        set_arm("plugin")


def _configure_aiter(names: Iterable[str]) -> bool:
    """The AITER half of the plan; False when AITER has no ``aiter.plugin``
    (an older build), which leaves AITER routing untouched."""
    try:
        from aiter import plugin as aiter_plugin
    except Exception:  # noqa: BLE001 -- absent or broken: nothing to drive
        return False
    configure_aiter = getattr(aiter_plugin, "configure", None)
    if not callable(configure_aiter):
        return False
    configure_aiter(source=AITER_SOURCE, enable=list(names))
    return True
