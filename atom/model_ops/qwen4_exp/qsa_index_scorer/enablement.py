# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Per-process enablement of aiter's neutral QSA scorer op.

aiter owns the op and its dispatch: ``aiter.ops.qsa.qsa_paged_mqa_logits``
(the 12 leading parameters of ATOM's own scorer) serves every valid call --
its fused FlyDSL kernel where the configured profile covers the call class,
aiter's native paths otherwise. ATOM only:

* :func:`configure` -- once per worker process, from ``ModelRunner.__init__``
  after ``_setup_device_and_distributed()`` and before
  ``_build_and_load_model()``: ``aiter.ops.qsa.configure(profile=...,
  source="atom:...")`` and one startup line from ``aiter.ops.qsa.status()``.
  Fail-open: an aiter without ``aiter.ops.qsa.configure`` (an image older
  than the neutral op), or one whose ``configure`` raises, leaves ATOM's own
  scorer in place -- logged once, never raised.
* :func:`index_scorer` -- what ``Qwen4ExpAttention`` injects into the stock
  selection loop: aiter's op itself (no wrapper, no ATOM fallback), or None
  for ATOM's own scorer.
* :func:`prewarm` -- after KV allocation, before CUDA-graph capture:
  ``aiter.ops.qsa.prewarm(...)`` once per distinct bound index-cache layout.
* The opt-in class observer (``--qsa-index-scorer-capture``,
  :class:`ClassObserver`): one ``aiter.ops.qsa.explain()`` per call class,
  per-path call counters and the capture JSON. Not installed by default.

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

#: The neutral op's module (aiter in-tree, contract ``qsa_paged_mqa_logits`` v1).
AITER_QSA_MODULE = "aiter.ops.qsa"
#: ``source`` passed to ``aiter.ops.qsa.configure``: provenance of the policy.
CONFIGURE_SOURCE = "atom:Config.qsa_index_scorer.profile"
#: Query-position dtype ATOM's QSA metadata carries (``logical_positions``).
QUERY_POSITION_DTYPES = ("int64",)

# Why ATOM's own scorer stayed in place (fail-open reasons, logged once).
AITER_UNAVAILABLE = "aiter_qsa_unavailable"  # import of aiter.ops.qsa failed
AITER_WITHOUT_CONFIGURE = "aiter_qsa_without_configure"  # pre-neutral-op aiter
CONFIGURE_FAILED = "aiter_qsa_configure_failed"  # configure() raised
CAPTURE_ONLY = "capture_only"  # no profile: capture observes ATOM's scorer

#: Path recorded for ATOM's own scorer in the capture (not an aiter path).
ATOM_PATH = "atom"


@dataclass
class ScorerState:
    """Outcome of the per-process ``configure`` for one profile."""

    profile: str
    #: ``aiter.ops.qsa.qsa_paged_mqa_logits`` when aiter accepted the profile,
    #: else None (ATOM's own scorer).
    op: Callable | None = None
    api: Any = None
    reason: str = ""
    status: dict = field(default_factory=dict)

    @property
    def enabled(self) -> bool:
        return self.op is not None


_LOCK = threading.Lock()
_STATES: dict[str, ScorerState] = {}
_PREWARMED: dict[tuple, tuple | None] = {}
_OBSERVERS: dict[str, ClassObserver] = {}


def _profile_of(config) -> str:
    cfg = getattr(config, "qsa_index_scorer", None)
    return "" if cfg is None else str(getattr(cfg, "profile", "") or "")


def _status_text(status: dict) -> str:
    return " ".join(f"{k}={status[k]}" for k in sorted(status, key=str))


def configure(config, rank_name: str = "") -> ScorerState | None:
    """``aiter.ops.qsa.configure(profile=..., source=...)`` once per process.

    Returns None while no profile is set (nothing imported). Idempotent per
    profile; never raises. Must run before the first scorer call of the
    process (``ModelRunner.__init__``, before the model is built), so the
    eager pre-capture warm-up already sees the configured policy.
    """
    profile = _profile_of(config)
    if not profile:
        return None
    with _LOCK:
        state = _STATES.get(profile)
        if state is not None:
            return state
        state = _STATES[profile] = _configure(profile)
    who = rank_name or "-"
    if state.enabled:
        st = state.status
        fused_on = bool(st.get("enabled", True))
        logger.info(
            "QSA index scorer [%s]: %s profile=%s source=%s op=%s.%s status: %s",
            who,
            "enabled" if fused_on else "configured (aiter reports fused off)",
            profile,
            CONFIGURE_SOURCE,
            AITER_QSA_MODULE,
            "qsa_paged_mqa_logits",
            _status_text(st),
        )
    else:
        logger.warning(
            "QSA index scorer [%s]: DISABLED (ATOM scorer) profile=%s reason=%s",
            who,
            profile,
            state.reason,
        )
    return state


def _configure(profile: str) -> ScorerState:
    try:
        api = importlib.import_module(AITER_QSA_MODULE)
    except Exception as exc:  # noqa: BLE001 -- fail-open: ATOM's scorer
        return ScorerState(profile, reason=f"{AITER_UNAVAILABLE}: {exc!r}")
    if not callable(getattr(api, "configure", None)) or not callable(
        getattr(api, "qsa_paged_mqa_logits", None)
    ):
        return ScorerState(profile, reason=AITER_WITHOUT_CONFIGURE)
    try:
        api.configure(profile=profile, source=CONFIGURE_SOURCE)
    except Exception as exc:  # noqa: BLE001 -- fail-open: ATOM's scorer
        return ScorerState(profile, reason=f"{CONFIGURE_FAILED}: {exc!r}")
    status: dict = {}
    status_fn = getattr(api, "status", None)
    if callable(status_fn):
        try:
            status = dict(status_fn())
        except Exception as exc:  # noqa: BLE001 -- observability only
            status = {"status_error": repr(exc)}
    return ScorerState(
        profile,
        op=api.qsa_paged_mqa_logits,
        api=api,
        reason="configured",
        status=status,
    )


def index_scorer(config) -> Callable | None:
    """The scorer ``Qwen4ExpAttention`` injects, or None for ATOM's own.

    * profile set, aiter configured: ``aiter.ops.qsa.qsa_paged_mqa_logits``
      itself -- the same object, so the call site costs nothing extra;
    * profile set, aiter without the neutral op (or configure failed): None;
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
    return observer(capture, op, state.api, state)


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
    """``aiter.ops.qsa.prewarm`` once per distinct bound index-cache layout.

    ``caches``: ``(compressed_k_cache, compressed_k_scale)`` pairs of the
    layers that call aiter's op. Builds what those caches can select (their
    page size and dtype, ATOM's int64 positions) so no variant compiles inside
    the per-bucket capture warm-up or serving. A raising prewarm is logged
    once and not retried: variants then build on first use (eager, before
    capture). Never raises. Returns ``{layout: built variants or None}``.
    """
    out: dict = {}
    for cache, scale in caches:
        if cache is None or getattr(cache, "ndim", 0) != 4:
            continue
        state = next((s for s in _STATES.values() if s.enabled), None)
        if state is None or not callable(getattr(state.api, "prewarm", None)):
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

    page = int(cache.shape[1])
    dtype = cache_dtype_name(cache, scale)
    pos_dtypes = tuple(getattr(torch, n) for n in QUERY_POSITION_DTYPES)
    t0 = time.perf_counter()
    try:
        built = tuple(
            state.api.prewarm(
                page_sizes=(page,),
                cache_dtypes=(dtype,),
                pos_dtypes=pos_dtypes,
                compressed_k_cache=cache,
                compressed_k_scale=scale,
                device=cache.device,
            )
            or ()
        )
    except Exception as exc:  # noqa: BLE001 -- lazy build stays correct
        logger.warning(
            "QSA index scorer [%s]: prewarm page=%d cache=%s failed after %.2f s "
            "(variants build on first use): %r",
            rank_name or "-",
            page,
            dtype,
            time.perf_counter() - t0,
            exc,
        )
        return None
    logger.info(
        "QSA index scorer [%s]: prewarm page=%d cache=%s pos=%s -> %d variant(s) "
        "in %.2f s: %s",
        rank_name or "-",
        page,
        dtype,
        ",".join(QUERY_POSITION_DTYPES),
        len(built),
        time.perf_counter() - t0,
        ", ".join(map(str, built)),
    )
    return built


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

    The first call of each class runs ``aiter.ops.qsa.explain(*12 args)`` (a
    host-side decision; under CUDA graphs the first call of a class is the
    eager per-bucket warm-up, outside the recording) and records the class in
    the capture JSON. Every call then goes to ``op`` unchanged -- the observer
    never re-runs or replaces a call. Python-level counts: a captured call
    counts once, replays run no Python.
    """

    def __init__(self, path: str, op: Callable, api, state: ScorerState | None):
        from .capture import get_capture

        self.op = op
        self.api = api
        self.profile = "" if state is None else state.profile
        self.reason = CAPTURE_ONLY if state is None else state.reason
        self.capture = get_capture(path)
        #: class key -> (``"path:reason"``, the capture entry of the class)
        self.classes: dict[tuple, tuple[str, dict]] = {}
        self.calls: collections.Counter = collections.Counter()

    def _explain(self, a: tuple) -> tuple[str, str]:
        if self.api is None:
            return ATOM_PATH, self.reason
        explain = getattr(self.api, "explain", None)
        if not callable(explain):
            return "aiter", "explain_unavailable"
        try:
            d = explain(*a)
        except Exception as exc:  # noqa: BLE001 -- observability only
            return "aiter", f"explain_error:{type(exc).__name__}"
        return str(getattr(d, "path", "aiter")), str(getattr(d, "reason", ""))

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


def observer(path: str, op: Callable, api, state) -> ClassObserver:
    """One observer per capture path and process (all layers share it)."""
    with _LOCK:
        obs = _OBSERVERS.get(path)
        if obs is None:
            obs = _OBSERVERS[path] = ClassObserver(path, op, api, state)
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
