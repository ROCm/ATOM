# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Per-process enablement of aiter's routed QSA scorer op.

aiter owns the op ``aiter.ops.qsa.qsa_paged_mqa_logits`` (contract
``qsa_paged_mqa_logits`` v1, routed through ``aiter.plugin``; the 12 leading
parameters of ATOM's own scorer): an enabled provider (e.g.
``qsa-index-scorer-provider``, the fused FlyDSL gfx1250 scorer) serves the call
classes it covers, aiter's native paths serve the rest. ATOM only:

* :func:`configure` -- once per worker process, from ``ModelRunner.__init__``
  after ``_setup_device_and_distributed()`` and before
  ``_build_and_load_model()`` (after :func:`atom.aiter_provider.
  configure_aiter_providers`): the indexer uses aiter's op only when an enabled
  provider declares ``qsa_paged_mqa_logits`` and this aiter routes it. Fail-open:
  otherwise ATOM's own scorer stays in place -- logged once, never raised.
* :func:`index_scorer` -- what ``Qwen4ExpAttention`` injects into the stock
  selection loop: aiter's op itself (no wrapper, no ATOM fallback), or None
  for ATOM's own scorer.
* :func:`prewarm` -- after KV allocation, before CUDA-graph capture: the
  providers' ``warmup(payload, device=...)`` once per distinct bound
  index-cache layout (compiles / loads every variant the cache can select; the
  per-bucket eager warm-up forward then decides every class before capture).
* The opt-in class observer (``--qsa-index-scorer-capture``,
  :class:`ClassObserver`): one ``aiter.plugin.explain()`` per call class,
  per-route call counters and the capture JSON. Not installed by default.

Nothing here imports aiter at module scope; with the switch off nothing of
this package is imported at all.
"""

from __future__ import annotations

import atexit
import collections
import importlib
import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger("atom")

#: aiter's routed op (contract ``qsa_paged_mqa_logits`` v1) and its module.
OP_NAME = "qsa_paged_mqa_logits"
AITER_QSA_MODULE = "aiter.ops.qsa"
#: Query-position dtype ATOM's QSA metadata carries (``logical_positions``).
QUERY_POSITION_DTYPES = ("int64",)

# Why ATOM's own scorer stayed in place (fail-open reasons, logged once).
NO_PROVIDER = "no_enabled_provider_for_op"  # providers off / none declares the op
AITER_UNAVAILABLE = "aiter_qsa_unavailable"  # import of aiter.ops.qsa failed
OP_NOT_ROUTED = "aiter_qsa_not_routed"  # this aiter's op is not routed
CAPTURE_ONLY = "capture_only"  # no provider: capture observes ATOM's scorer

#: Path recorded for ATOM's own scorer in the capture (not an aiter route).
ATOM_PATH = "atom"


@dataclass
class ScorerState:
    """Outcome of the per-process ``configure``."""

    #: enabled providers that declare the op (preference order)
    providers: tuple
    #: ``aiter.ops.qsa.qsa_paged_mqa_logits`` when usable, else None (ATOM's own)
    op: Callable | None = None
    plugin: Any = None
    reason: str = ""
    status: dict = field(default_factory=dict)

    @property
    def enabled(self) -> bool:
        return self.op is not None

    # the capture JSON still names its policy field "profile"
    @property
    def profile(self) -> str:
        return ",".join(self.providers)


_LOCK = threading.Lock()
_STATES: dict[tuple, ScorerState] = {}
_PREWARMED: dict[tuple, tuple | None] = {}
_OBSERVERS: dict[str, ClassObserver] = {}


def configure(config, rank_name: str = "") -> ScorerState | None:
    """Decide once per process whether the indexer scores through aiter's op.

    Returns None while no provider is enabled (nothing imported). Idempotent;
    never raises. Runs before the first scorer call of the process
    (``ModelRunner.__init__``, before the model is built).
    """
    enable = tuple(getattr(config, "aiter_provider_enable", ()) or ())
    if not enable:
        return None
    with _LOCK:
        state = _STATES.get(enable)
        if state is not None:
            return state
        state = _STATES[enable] = _configure(config, rank_name)
    who = rank_name or "-"
    if state.enabled:
        logger.info(
            "QSA index scorer [%s]: aiter op %s.%s (route %s) providers=%s",
            who,
            AITER_QSA_MODULE,
            OP_NAME,
            OP_NAME,
            list(state.providers),
        )
    else:
        logger.warning(
            "QSA index scorer [%s]: DISABLED (ATOM scorer) enabled=%s reason=%s",
            who,
            list(enable),
            state.reason,
        )
    return state


def _configure(config, rank_name: str) -> ScorerState:
    from atom.aiter_provider import configure_aiter_providers

    pstate = configure_aiter_providers(config, rank_name)
    providers = () if pstate is None else pstate.serves(OP_NAME)
    if not providers:
        reason = NO_PROVIDER if pstate is None else (pstate.reason or NO_PROVIDER)
        return ScorerState((), reason=reason)
    try:
        api = importlib.import_module(AITER_QSA_MODULE)
        plugin = importlib.import_module("aiter.plugin")
    except Exception as exc:  # noqa: BLE001 -- fail-open: ATOM's scorer
        return ScorerState(providers, reason=f"{AITER_UNAVAILABLE}: {exc!r}")
    op = getattr(api, "qsa_paged_mqa_logits", None)
    if not callable(op) or getattr(op, "__aiter_plugin_op__", None) is None:
        return ScorerState(providers, reason=OP_NOT_ROUTED)
    return ScorerState(
        providers, op=op, plugin=plugin, reason="routed", status=dict(pstate.status)
    )


def index_scorer(config) -> Callable | None:
    """The scorer ``Qwen4ExpAttention`` injects, or None for ATOM's own.

    * an enabled provider declares the op and this aiter routes it:
      ``aiter.ops.qsa.qsa_paged_mqa_logits`` itself -- the same object, so the
      call site costs nothing extra;
    * otherwise: None;
    * ``capture`` set: the scorer above (or ATOM's own) behind the opt-in
      :class:`ClassObserver`.

    Configures lazily when the process never ran ``ModelRunner.__init__``
    (still before the first scorer call: layers are built before any forward).
    """
    cfg = getattr(config, "qsa_index_scorer", None)
    if cfg is None:
        return None
    state = configure(config)
    op = None if state is None else state.op
    capture = str(getattr(cfg, "capture", "") or "")
    if not capture:
        return op
    if op is None:
        from atom.model_ops.qwen4_exp.ops.qsa import qsa_paged_mqa_logits

        return observer(capture, qsa_paged_mqa_logits, None, state)
    return observer(capture, op, state.plugin, state)


# ---------------------------------------------------------------------------
# prewarm (after KV allocation, before CUDA-graph capture)
# ---------------------------------------------------------------------------
def cache_dtype_name(cache, scale) -> str:
    """aiter's ``cache_dtype`` vocabulary (profile rows, ``prewarm``)."""
    name = str(cache.dtype).replace("torch.", "")
    if name == "bfloat16" and scale is None:
        return "bf16"
    if name.startswith("float8_e4m3") and scale is not None:
        if str(scale.dtype) == "torch.float32" and scale.ndim == 3:
            return "fp8"
        return "mxfp8"
    if name == "uint8":
        return "fp4"
    return name


def _layout(t) -> tuple | None:
    if t is None:
        return None
    return (tuple(t.shape), tuple(t.stride()), str(t.dtype), str(t.device))


def prewarm(caches, rank_name: str = "") -> dict:
    """The providers' ``warmup`` once per distinct bound index-cache layout.

    ``caches``: ``(compressed_k_cache, compressed_k_scale)`` pairs of the
    layers that call aiter's op. Each provider that declares the op compiles /
    loads what those caches can select (their page size and dtype, ATOM's int64
    positions) so no variant compiles inside the per-bucket capture warm-up or
    serving. A failing warm-up is logged once and not retried: variants then
    build on first use (eager, before capture). Never raises. Returns
    ``{layout: {provider: ok} or None}``.
    """
    out: dict = {}
    for cache, scale in caches:
        if cache is None or getattr(cache, "ndim", 0) != 4:
            continue
        state = next((s for s in _STATES.values() if s.enabled), None)
        if state is None:
            return out
        key = (_layout(cache), _layout(scale))
        if key in out:
            continue
        if key in _PREWARMED:
            out[key] = _PREWARMED[key]
            continue
        out[key] = _PREWARMED[key] = _prewarm_one(state, cache, scale, rank_name)
    return out


def _prewarm_one(state: ScorerState, cache, scale, rank_name: str):
    import torch

    from atom.aiter_provider import warmup_aiter_providers

    page = int(cache.shape[1])
    dtype = cache_dtype_name(cache, scale)
    payload = {
        "page_sizes": (page,),
        "cache_dtypes": (dtype,),
        "pos_dtypes": tuple(getattr(torch, n) for n in QUERY_POSITION_DTYPES),
        "compressed_k_cache": cache,
        "compressed_k_scale": scale,
    }
    t0 = time.perf_counter()
    try:
        warmed = warmup_aiter_providers(
            OP_NAME, payload, device=cache.device, rank_name=rank_name
        )
    except Exception as exc:  # noqa: BLE001 -- lazy build stays correct
        warmed = None
        err = repr(exc)
    else:
        err = ""
    level = logging.INFO if warmed and all(warmed.values()) else logging.WARNING
    logger.log(
        level,
        "QSA index scorer [%s]: provider warm-up page=%d cache=%s pos=%s -> %s in "
        "%.2f s%s",
        rank_name or "-",
        page,
        dtype,
        ",".join(QUERY_POSITION_DTYPES),
        warmed,
        time.perf_counter() - t0,
        f" ({err}; variants build on first use)" if err else "",
    )
    return warmed


# ---------------------------------------------------------------------------
# opt-in class observer: explain() per class, counters, capture JSON
# ---------------------------------------------------------------------------
SCORER_PARAMS = (
    "q",
    "compressed_k_cache",
    "page_table",
    "token_to_request",
    "query_positions",
    "context_lens",
    "compress_ratio",
    "score_divisor",
    "max_columns",
    "row_starts",
    "allow_flydsl",
    "compressed_k_scale",
)
_DEFAULTS = {
    "compress_ratio": 4,
    "score_divisor": None,
    "max_columns": None,
    "row_starts": None,
    "allow_flydsl": False,
    "compressed_k_scale": None,
}


def bind_args(args: tuple, kwargs: dict) -> tuple:
    """The 12 scorer arguments, in order, from a call's args/kwargs."""
    if len(args) > len(SCORER_PARAMS):
        raise TypeError(f"scorer takes {len(SCORER_PARAMS)} arguments")
    bound = dict(zip(SCORER_PARAMS, args))
    for k, v in kwargs.items():
        if k not in SCORER_PARAMS or k in bound:
            raise TypeError(f"unexpected scorer argument {k!r}")
        bound[k] = v
    return tuple(bound.get(n, _DEFAULTS.get(n)) for n in SCORER_PARAMS)


def _class_key(a: tuple) -> tuple:
    key = []
    for v in a:
        if hasattr(v, "stride") and hasattr(v, "shape"):
            key.append((tuple(v.shape), tuple(v.stride()), str(v.dtype)))
        else:
            key.append(v)
    # q's base alignment can steer aiter's choice of kernel.
    key.append(int(a[0].data_ptr()) % 16 if hasattr(a[0], "data_ptr") else 0)
    return tuple(key)


class ClassObserver:
    """Opt-in scorer wrapper: counts calls per class and asks aiter once.

    The first call of each class runs ``aiter.plugin.explain(op, *12 args)``
    (a host-side dry run of the routing decision; under CUDA graphs the first
    call of a class is the eager per-bucket warm-up, outside the recording; it
    calls the providers' ``select``, which decides and caches the class) and
    records the class in
    the capture JSON. Every call then goes to ``op`` unchanged -- the observer
    never re-runs or replaces a call. Python-level counts: a captured call
    counts once, replays run no Python.
    """

    def __init__(self, path: str, op: Callable, plugin, state: ScorerState | None):
        from .capture import get_capture

        self.op = op
        self.plugin = plugin
        self.profile = "" if state is None else state.profile
        self.reason = CAPTURE_ONLY if state is None else state.reason
        self.capture = get_capture(path)
        #: class key -> (``"path:reason"``, the capture entry of the class)
        self.classes: dict[tuple, tuple[str, dict]] = {}
        self.calls: collections.Counter = collections.Counter()

    def _explain(self, a: tuple) -> tuple[str, str]:
        if self.plugin is None:
            return ATOM_PATH, self.reason
        explain = getattr(self.plugin, "explain", None)
        if not callable(explain):
            return "aiter", "explain_unavailable"
        try:
            d = explain(OP_NAME, *a).decision
        except Exception as exc:  # noqa: BLE001 -- observability only
            return "aiter", f"explain_error:{type(exc).__name__}"
        if d is None:  # AITER_PLUGINS=0: routing off, native
            return "native", "kill_switch"
        route = str(getattr(d, "route", "aiter"))
        provider = getattr(d, "provider", None)
        path = f"{route}:{provider}" if route == "provider" and provider else route
        return path.replace(":", "/"), str(getattr(d, "reason", ""))

    def __call__(self, *args, **kwargs):
        a = bind_args(args, kwargs)
        key = _class_key(a)
        known = self.classes.get(key)
        if known is None:
            path, reason = self._explain(a)
            dk = f"{path}:{reason}"
            entry = self.capture.class_entry(a, self.profile)
            known = self.classes[key] = (dk, entry)
        dk, entry = known
        self.calls[dk] += 1
        self.capture.count(entry, dk)
        return self.op(*args, **kwargs)

    def by_path(self) -> dict:
        out: collections.Counter = collections.Counter()
        for dk, n in self.calls.items():
            out[dk.partition(":")[0]] += n
        return dict(out)


def observer(path: str, op: Callable, plugin, state) -> ClassObserver:
    """One observer per capture path and process (all layers share it)."""
    with _LOCK:
        obs = _OBSERVERS.get(path)
        if obs is None:
            obs = _OBSERVERS[path] = ClassObserver(path, op, plugin, state)
            if len(_OBSERVERS) == 1:
                atexit.register(log_decisions)
    return obs


def decisions() -> dict:
    """``{path: calls}`` over every observer of the process."""
    out: collections.Counter = collections.Counter()
    for obs in _OBSERVERS.values():
        out.update(obs.by_path())
    return dict(out)


def log_decisions() -> None:
    d = decisions()
    if d:
        logger.info(
            "QSA index scorer decisions: %s",
            ", ".join(f"{k}={v}" for k, v in sorted(d.items())),
        )


def _reset_for_testing() -> None:
    with _LOCK:
        _STATES.clear()
        _PREWARMED.clear()
        _OBSERVERS.clear()
