# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Call-site route: aiter fused QSA index scorer, else ATOM's native scorer.

One :class:`QSAScorerRoute` per QSA layer (``site`` = the layer prefix). It is
called exactly like ATOM's ``qsa_paged_mqa_logits`` (same 12 parameters, see
:data:`NATIVE_SCORER_PARAMS`) and returns the same ``(logits, visible)``.

Decision protocol (cross-repo contract ``records/integration/contract.md``,
aiter ``aiter.ops.qsa_index_scorer`` API 1.x):

* construction -- import aiter's module (only here, only when a profile is
  configured), check ``API_VERSION`` major and the scorer signature; any
  failure pins the whole site native with an ``atom_*`` reason;
* :meth:`QSAScorerRoute.prepare` -- ``layout_eligibility`` once per bound
  compressed-key cache (prepare time: ``Qwen4ExpAttention.bind_caches``);
  a static failure pins every call on that cache native;
* per call class -- ``decide`` ONCE per class key (shapes, dtypes, strides,
  q 16-byte alignment, cache identity, bucket ``max_columns``, ...); the
  decision is cached and reused. ``decide`` reads no tensor *values*, so a
  class key fixes its answer: no device sync, no collective, and identical
  layouts on every TP rank give identical decisions (rank-uniform by
  construction);
* a covered call goes through aiter's public ``qsa_paged_mqa_logits(...,
  profile=...)``; if that still refuses (``QSAIndexScorerUnsupported``) the
  call runs native and the class is re-pinned with aiter's reason;
* every native fallback carries a reason, logged once per (site, reason) and
  counted in :attr:`QSAScorerRoute.counts`.

Pure Python + torch: importable without Triton or aiter (CPU tests inject a
fake ``api`` and ``native``).
"""

from __future__ import annotations

import collections
import importlib
import inspect
import logging
from collections.abc import Callable
from typing import Any

logger = logging.getLogger("atom")

AITER_MODULE = "aiter.ops.qsa_index_scorer"
#: API major this ATOM build is written against (contract section 1).
CONTRACT_API_MAJOR = "1"

_REQUIRED = "<required>"
#: ATOM ``atom/model_ops/qwen4_exp/ops/qsa.py::qsa_paged_mqa_logits`` (frozen
#: sha256 aaa78f43...): (name, default) in order. CPU tests AST-check qsa.py
#: against this tuple and aiter's ``ATOM_SCORER_SIGNATURE`` against it.
NATIVE_SCORER_PARAMS: tuple[tuple[str, Any], ...] = (
    ("q", _REQUIRED),
    ("compressed_k_cache", _REQUIRED),
    ("page_table", _REQUIRED),
    ("token_to_request", _REQUIRED),
    ("query_positions", _REQUIRED),
    ("context_lens", _REQUIRED),
    ("compress_ratio", 4),
    ("score_divisor", None),
    ("max_columns", None),
    ("row_starts", None),
    ("allow_flydsl", False),
    ("compressed_k_scale", None),
)

#: aiter ``Reason`` values (contract section 3, frozen order) -- static first.
CONTRACT_STATIC_REASONS = (
    "disabled",
    "profile_not_found",
    "profile_invalid",
    "profile_kernel_mismatch",
    "kernel_env_mismatch",
    "kernel_unavailable",
    "arch_unsupported",
    "cu_unsupported",
    "cache_format_unsupported",
    "cache_shape_unsupported",
    "cache_layout_unsupported",
    "cache_alignment_unsupported",
    "scale_layout_unsupported",
)
CONTRACT_CALL_REASONS = (
    "device_not_gpu",
    "device_mismatch",
    "compress_ratio_unsupported",
    "q_dtype_unsupported",
    "q_shape_unsupported",
    "q_layout_unsupported",
    "metadata_unsupported",
    "output_layout_unsupported",
    "empty_call",
    "divisor_uncovered",
    "max_columns_invalid",
    "plane_too_large",
    "rows_rule_uncovered",
    "columns_uncovered",
    "table_width_uncovered",
    "row_unverified",
    "route_mismatch",
)
CONTRACT_REASONS = frozenset(("ok",) + CONTRACT_STATIC_REASONS + CONTRACT_CALL_REASONS)

#: ATOM-side reasons (the site never reaches aiter's decision).
AITER_UNAVAILABLE = "atom_aiter_unavailable"  # import of aiter's module failed
API_VERSION_MISMATCH = "atom_api_version_mismatch"  # API_VERSION major != 1
SIGNATURE_DRIFT = "atom_signature_drift"  # aiter op != ATOM's scorer signature
CAPTURE_ONLY = "atom_capture_only"  # no profile: capture-only route
ATOM_REASONS = (AITER_UNAVAILABLE, API_VERSION_MISMATCH, SIGNATURE_DRIFT, CAPTURE_ONLY)

OK = "ok"
#: Decision-cache bound: classes are few (buckets x layers' caches); a flood
#: means something keys per step, so start over rather than grow.
MAX_CACHED_DECISIONS = 4096


def native_qsa_paged_mqa_logits():
    """ATOM's own scorer (imported lazily: it needs Triton and aiter)."""
    from atom.model_ops.qwen4_exp.ops.qsa import qsa_paged_mqa_logits

    return qsa_paged_mqa_logits


def regime_of(rows: int, requests: int) -> str | None:
    """aiter ``profiles.regime_of`` (same thresholds), for capture records."""
    rows, requests = int(rows), int(requests)
    if rows < 1 or requests < 1:
        return None
    if rows <= requests:
        return "draft"
    if rows >= 32 and rows >= 32 * requests:
        return "prefill"
    if rows % 4 == 0:
        return "decode"
    return None


def _tensor_key(t) -> tuple | None:
    if t is None:
        return None
    return (tuple(t.shape), t.dtype, tuple(t.stride()), t.device)


def layout_key(cache, scale) -> tuple:
    """Identity of a bound compressed-key cache (+ scale plane)."""
    return (
        cache.data_ptr(),
        tuple(cache.shape),
        cache.dtype,
        tuple(cache.stride()),
        cache.device,
        None if scale is None else (scale.data_ptr(),) + _tensor_key(scale),
    )


def call_class_key(
    q,
    compressed_k_cache,
    page_table,
    token_to_request,
    query_positions,
    context_lens,
    compress_ratio,
    score_divisor,
    max_columns,
    row_starts,
    compressed_k_scale,
) -> tuple:
    """Everything aiter's ``decide`` reads (it never reads tensor values).

    q's base matters only through its 16-byte alignment (``q_layout``); the
    cache is keyed by identity (its base alignment picks the kernel route).
    """
    return (
        _tensor_key(q),
        q.data_ptr() % 16 == 0,
        layout_key(compressed_k_cache, compressed_k_scale),
        _tensor_key(page_table),
        _tensor_key(token_to_request),
        _tensor_key(query_positions),
        _tensor_key(context_lens),
        _tensor_key(row_starts),
        int(compress_ratio),
        None if score_divisor is None else float(score_divisor),
        None if max_columns is None else int(max_columns),
    )


def _params_of(fn) -> tuple[tuple[str, Any], ...]:
    out = []
    for p in inspect.signature(fn).parameters.values():
        if p.kind not in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD):
            break
        out.append((p.name, _REQUIRED if p.default is p.empty else p.default))
    return tuple(out)


def signature_drift(api) -> str | None:
    """Why aiter's op is not a drop-in for ATOM's scorer, or None."""
    fn = getattr(api, "qsa_paged_mqa_logits", None)
    if fn is None:
        return "aiter module has no qsa_paged_mqa_logits"
    got = _params_of(fn)
    if got != NATIVE_SCORER_PARAMS:
        return f"positional params {got} != ATOM {NATIVE_SCORER_PARAMS}"
    sig = inspect.signature(fn)
    if "profile" not in sig.parameters or (
        sig.parameters["profile"].kind is not inspect.Parameter.KEYWORD_ONLY
    ):
        return "aiter op has no keyword-only `profile`"
    declared = getattr(api, "ATOM_SCORER_SIGNATURE", None)
    if declared is not None:
        norm = tuple(
            (n, _REQUIRED if d is inspect.Parameter.empty else d) for n, d in declared
        )
        if norm != NATIVE_SCORER_PARAMS:
            return f"ATOM_SCORER_SIGNATURE {norm} != ATOM {NATIVE_SCORER_PARAMS}"
    for name in ("decide", "layout_eligibility", "QSAIndexScorerUnsupported"):
        if not hasattr(api, name):
            return f"aiter module has no {name}"
    return None


class QSAScorerRoute:
    """The scorer the QSA layer calls; fused when covered, else native."""

    def __init__(
        self,
        profile: str = "",
        *,
        site: str = "qsa",
        capture=None,
        api=None,
        native: Callable | None = None,
    ):
        self.profile = profile or ""
        self.site = site
        self.capture = capture
        self._native = native
        self.api = None
        #: (reason, detail) pinning EVERY call native, or None.
        self.pinned: tuple[str, str] | None = None
        self._layout: dict[tuple, tuple[str, str]] = {}
        self._decisions: dict[tuple, tuple[str, str]] = {}
        self._logged: set[str] = set()
        #: (path, reason) -> calls; path is "fused" or "native".
        self.counts: collections.Counter = collections.Counter()
        self.decide_calls = 0
        self.layout_calls = 0
        if not self.profile:
            self._pin(CAPTURE_ONLY, "no profile configured; capture only")
            return
        if api is None:
            try:
                api = importlib.import_module(AITER_MODULE)
            except Exception as exc:  # noqa: BLE001 -- any failure => native
                self._pin(AITER_UNAVAILABLE, f"import {AITER_MODULE}: {exc!r}")
                return
        self.api = api
        version = str(getattr(api, "API_VERSION", ""))
        if version.split(".")[0] != CONTRACT_API_MAJOR:
            self._pin(
                API_VERSION_MISMATCH,
                f"aiter API_VERSION {version!r}, ATOM needs {CONTRACT_API_MAJOR}.x",
            )
            return
        drift = signature_drift(api)
        if drift is not None:
            self._pin(SIGNATURE_DRIFT, drift)

    # ------------------------------------------------------------------ #
    @property
    def native(self) -> Callable:
        if self._native is None:
            self._native = native_qsa_paged_mqa_logits()
        return self._native

    @property
    def fused_possible(self) -> bool:
        return self.pinned is None

    def _pin(self, reason: str, detail: str) -> None:
        self.pinned = (reason, detail)
        self._log_once(reason, detail, warn=reason != CAPTURE_ONLY)

    def _log_once(self, reason: str, detail: str, warn: bool = False) -> None:
        if reason in self._logged:
            return
        self._logged.add(reason)
        path = "fused" if reason == OK else "native"
        (logger.warning if warn else logger.info)(
            "QSA index scorer [%s] -> %s (%s)%s",
            self.site,
            path,
            reason,
            f": {detail}" if detail else "",
        )

    # ------------------------------------------------------------------ #
    def prepare(self, compressed_k_cache, compressed_k_scale=None) -> str | None:
        """Prepare-time layout check for a bound cache; returns its reason.

        Called from ``bind_caches`` (once per cache binding); calls on a cache
        never prepared are checked lazily on their first call.
        """
        if compressed_k_cache is None:
            return None
        return self._layout_decision(compressed_k_cache, compressed_k_scale)[0]

    def _layout_decision(self, cache, scale) -> tuple[str, str]:
        if self.pinned is not None:
            return self.pinned
        key = layout_key(cache, scale)
        dec = self._layout.get(key)
        if dec is None:
            self.layout_calls += 1
            d = self.api.layout_eligibility(cache, scale, profile=self.profile)
            dec = (d.reason.value, d.detail)
            if len(self._layout) >= MAX_CACHED_DECISIONS:
                self._layout.clear()
            self._layout[key] = dec
            if dec[0] != OK:
                self._log_once(dec[0], dec[1])
        return dec

    def _decide(self, args: tuple) -> tuple[str, str]:
        (
            q,
            cache,
            table,
            t2r,
            pos,
            ctx,
            ratio,
            divisor,
            cols,
            row_starts,
            flydsl,
            scale,
        ) = args
        layout = self._layout_decision(cache, scale)
        if layout[0] != OK:
            return layout
        self.decide_calls += 1
        d = self.api.decide(
            q,
            cache,
            table,
            t2r,
            pos,
            ctx,
            ratio,
            divisor,
            cols,
            row_starts,
            flydsl,
            scale,
            profile=self.profile,
        )
        return (d.reason.value, d.detail)

    def decision_for(self, *args, **kwargs) -> tuple[str, str]:
        """(reason, detail) this route would take for a call (cached)."""
        a = self._bind(args, kwargs)
        key = call_class_key(*a[:10], a[11])
        dec = self._decisions.get(key)
        if dec is None:
            dec = self._decide(a)
            if len(self._decisions) >= MAX_CACHED_DECISIONS:
                self._decisions.clear()
            self._decisions[key] = dec
        return dec

    @staticmethod
    def _bind(args, kwargs) -> tuple:
        if not kwargs and len(args) == len(NATIVE_SCORER_PARAMS):
            return tuple(args)
        values = dict(zip((n for n, _ in NATIVE_SCORER_PARAMS), args))
        for k, v in kwargs.items():
            if k in values:
                raise TypeError(f"multiple values for argument {k!r}")
            values[k] = v
        out = []
        for name, default in NATIVE_SCORER_PARAMS:
            if name in values:
                out.append(values.pop(name))
            elif default is _REQUIRED:
                raise TypeError(f"missing required argument {name!r}")
            else:
                out.append(default)
        if values:
            raise TypeError(f"unexpected arguments {sorted(values)}")
        return tuple(out)

    # ------------------------------------------------------------------ #
    def __call__(self, *args, **kwargs):
        """Drop-in for ATOM's ``qsa_paged_mqa_logits``."""
        a = self._bind(args, kwargs)
        reason, detail = self.decision_for(*a)
        path = "native"
        out = None
        if reason == OK:
            try:
                out = self.api.qsa_paged_mqa_logits(*a, profile=self.profile)
                path = "fused"
            except self.api.QSAIndexScorerUnsupported as exc:
                reason = getattr(exc.reason, "value", str(exc.reason))
                detail = str(exc)
                key = call_class_key(*a[:10], a[11])
                self._decisions[key] = (reason, detail)
        if self.capture is not None:
            self.capture.record(self.site, a, reason, path, self.profile)
        self.counts[(path, reason)] += 1
        self._log_once(reason, detail)
        if out is not None:
            return out
        return self.native(*a)

    def select_paged_tokens(self, *args, **kwargs):
        """``qsa_select_paged_tokens`` with this route as its scorer."""
        from .select import qsa_select_paged_tokens_routed

        return qsa_select_paged_tokens_routed(*args, scorer=self, **kwargs)

    def summary(self) -> dict:
        return {
            "site": self.site,
            "profile": self.profile,
            "pinned": None if self.pinned is None else list(self.pinned),
            "counts": {f"{p}:{r}": n for (p, r), n in sorted(self.counts.items())},
            "decide_calls": self.decide_calls,
            "layout_calls": self.layout_calls,
            "classes": len(self._decisions),
        }


def build_route(cfg, site: str, *, api=None, native=None) -> QSAScorerRoute | None:
    """The route a QSA layer installs for ``cfg`` (``QSAIndexScorerConfig``).

    None while the config is inactive (the caller keeps the stock path).
    """
    if cfg is None or not getattr(cfg, "active", False):
        return None
    capture = None
    if getattr(cfg, "capture", ""):
        from .capture import get_capture

        capture = get_capture(cfg.capture)
    return QSAScorerRoute(
        getattr(cfg, "profile", ""), site=site, capture=capture, api=api, native=native
    )
