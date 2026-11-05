# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Call-site route: aiter's in-tree QSA index scorer op, else ATOM's stock scorer.

One :class:`QSAScorerRoute` per QSA layer (``site`` = the layer prefix). It is
called exactly like ATOM's ``qsa_paged_mqa_logits`` (same 12 parameters,
:data:`SCORER_PARAMS` == :data:`NATIVE_SCORER_PARAMS`) and returns the same
``(logits, visible)``.

Every call takes one of three paths (:attr:`QSAScorerRoute.counts` keys):

* ``fused`` -- aiter's op served it with its FlyDSL kernel;
* ``aiter-native`` -- aiter's op served it on aiter's own native path (only
  with an aiter whose ``Decision`` carries ``path``; ATOM adds no fallback);
* ``stock`` -- ATOM's own scorer ran (flag-on interim behaviour for calls an
  aiter without a native path does not cover, and for every call of a site
  whose capability or cache layout failed).

Decision protocol (cross-repo contract ``records/integration/contract.md``,
aiter ``aiter.ops.qsa_index_scorer`` API 1.x):

* construction -- bind the process capability (:mod:`.capability`, checked
  once per process: aiter module present, ``API_VERSION`` major, scorer
  signature, profile loads); any failure pins the whole site to the stock
  scorer with an ``atom_*`` (or aiter static) reason;
* :meth:`QSAScorerRoute.prepare` -- ``layout_eligibility`` once per bound
  compressed-key cache (prepare time: ``Qwen4ExpAttention.bind_caches``);
  a static failure pins every call on that cache to the stock scorer. With
  aiter API >= 1.1 the first cache that passes also triggers aiter's
  ``prewarm`` ONCE PER PROCESS per (cache dtype, page size) (not per layer):
  every kernel variant that cache can select is built after KV allocation
  and before CUDA-graph capture, so no variant compiles inside serving;
* per call class -- ``decide`` ONCE per class key (shapes, dtypes, strides,
  devices, q 16-byte alignment, cache identity, bucket ``max_columns``,
  ...); the decision is cached and reused. ``decide`` reads no tensor
  *values*, so a class key fixes its answer: no device sync, no
  collective, and identical layouts on every TP rank give identical
  decisions (rank-uniform by construction);
* a covered call runs aiter's ``launch(cached_decision, ...)`` (API >= 1.1:
  no per-call ``decide``) or, with an API 1.0 aiter, the public
  ``qsa_paged_mqa_logits(..., profile=...)``; if either still refuses
  (``QSAIndexScorerUnsupported``, an O(1) guard before any launch) the call
  runs the stock scorer and the class is re-pinned with aiter's reason;
  any other exception propagates (no retry after aiter's launch began);
* every stock call carries a reason, logged once per (site, reason) and
  counted in :attr:`QSAScorerRoute.counts`; :func:`process_summary` sums
  every route of the process (dumped once at exit when enabled).

The per-call host path (:meth:`QSAScorerRoute.__call__`) is kept minimal:
Python binds the arguments, one flat class-key tuple of torch metadata (no
string formatting, no ``inspect``), one dict lookup, the cached aiter
callable, and the capture hook only when capture is on.

Pure Python + torch: importable without Triton or aiter (CPU tests inject a
fake ``api`` and ``native``).
"""

from __future__ import annotations

import collections
import inspect
import logging
import time
import weakref
from collections.abc import Callable
from typing import Any

logger = logging.getLogger("atom")

AITER_MODULE = "aiter.ops.qsa_index_scorer"
#: API major this ATOM build is written against (contract section 1).
CONTRACT_API_MAJOR = "1"
#: First aiter API with ``prewarm`` and ``launch`` (contract v1.1 addendum).
LAUNCH_API_VERSION = (1, 1)
#: dtype(s) of ``query_positions`` on ATOM's real call path: the QSA metadata's
#: ``logical_positions`` is int64 in every builder (``qwen4_exp_attn``
#: ``qsa_logical_positions``, the sglang bridge, the rtpllm context). Names,
#: resolved to torch dtypes at prewarm time.
QUERY_POSITION_DTYPES = ("int64",)

_REQUIRED = "<required>"
#: The contract scorer signature: aiter's ``qsa_paged_mqa_logits`` /
#: ``decide`` / ``eligibility`` and ``ATOM_SCORER_SIGNATURE``, i.e. the frozen
#: golden ATOM ``qsa.py::qsa_paged_mqa_logits`` (sha256 aaa78f43... CRLF /
#: c2e46e50... LF): (name, default) in order. The route binds every call to it.
SCORER_PARAMS: tuple[tuple[str, Any], ...] = (
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
#: This tree's ``atom/model_ops/qwen4_exp/ops/qsa.py::qsa_paged_mqa_logits``
#: (unchanged from ATOM 8a07708 through ab4ada40: the frozen golden scorer):
#: the same 12 parameters. CPU tests AST-check qsa.py against this tuple.
NATIVE_SCORER_PARAMS: tuple[tuple[str, Any], ...] = SCORER_PARAMS

#: Call paths (``counts`` keys, capture records, the exit summary).
FUSED = "fused"
AITER_NATIVE = "aiter-native"
STOCK = "stock"
PATHS = (FUSED, AITER_NATIVE, STOCK)


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
FUSED_OK = (FUSED, OK)
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


def api_version_tuple(version) -> tuple[int, ...]:
    """``"1.1"`` -> ``(1, 1)``; stops at the first non-integer component."""
    out = []
    for part in str(version).split("."):
        if not part.isdigit():
            break
        out.append(int(part))
    return tuple(out)


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


def _tensor_key(t) -> tuple | None:
    if t is None:
        return None
    # torch.Size is a hashable tuple; get_device() is an int (-1 on CPU) and
    # cheaper than materialising a torch.device per call.
    return (t.shape, t.stride(), t.dtype, t.get_device())


def layout_key(cache, scale) -> tuple:
    """Identity of a bound compressed-key cache (+ scale plane)."""
    return (
        cache.data_ptr(),
        cache.shape,
        cache.stride(),
        cache.dtype,
        cache.get_device(),
        None if scale is None else (scale.data_ptr(),) + _tensor_key(scale),
    )


#: id(tensor) -> (weakref to it, its key): keys of tensor OBJECTS reused
#: across scorer calls -- the bound index cache (every call of a layer) and
#: the step's ``page_table`` / ``context_lens`` (every layer of a step). The
#: weakref proves the id still names the same live tensor; a live tensor's
#: shape / stride / dtype / device / base never change on these paths (the
#: caches are allocated once, QSA metadata is rebuilt -- never ``resize_`` /
#: ``set_`` -- per step), so its key is fixed for its lifetime.
_STABLE_KEYS: dict[int, tuple] = {}
_STABLE_LAYOUT_KEYS: dict[int, tuple] = {}
MAX_STABLE_KEYS = 1024


def _stable_tensor_key(t) -> tuple:
    hit = _STABLE_KEYS.get(id(t))
    if hit is not None and hit[0]() is t:
        return hit[1]
    key = (t.shape, t.stride(), t.dtype, t.get_device())
    if len(_STABLE_KEYS) >= MAX_STABLE_KEYS:
        _STABLE_KEYS.clear()
    _STABLE_KEYS[id(t)] = (weakref.ref(t), key)
    return key


def _stable_layout_key(cache, scale) -> tuple:
    if scale is not None:
        return layout_key(cache, scale)
    hit = _STABLE_LAYOUT_KEYS.get(id(cache))
    if hit is not None and hit[0]() is cache:
        return hit[1]
    key = layout_key(cache, None)
    if len(_STABLE_LAYOUT_KEYS) >= MAX_STABLE_KEYS:
        _STABLE_LAYOUT_KEYS.clear()
    _STABLE_LAYOUT_KEYS[id(cache)] = (weakref.ref(cache), key)
    return key


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
    This runs on every scorer call, so it is one flat tuple of torch metadata
    (no ``tuple()`` copies, no ``torch.device`` objects, no strings), and the
    tensor objects reused across calls (cache, page_table, context_lens) are
    keyed once per object (:data:`_STABLE_KEYS`). Positions: 0-4 q (shape,
    stride, dtype, device, 16-byte aligned), 5 cache layout
    (:func:`layout_key`, :data:`CLASS_KEY_LAYOUT_INDEX`), 6 page_table,
    7-10 token_to_request (shape, stride, dtype, device), 11-14
    query_positions (same), 15 context_lens, 16 row_starts (None or its
    4-tuple), 17-19 compress_ratio, score_divisor, max_columns.
    """
    return (
        q.shape,
        q.stride(),
        q.dtype,
        q.get_device(),
        q.data_ptr() % 16 == 0,
        _stable_layout_key(compressed_k_cache, compressed_k_scale),
        _stable_tensor_key(page_table),
        token_to_request.shape,
        token_to_request.stride(),
        token_to_request.dtype,
        token_to_request.get_device(),
        query_positions.shape,
        query_positions.stride(),
        query_positions.dtype,
        query_positions.get_device(),
        _stable_tensor_key(context_lens),
        (
            None
            if row_starts is None
            else (
                row_starts.shape,
                row_starts.stride(),
                row_starts.dtype,
                row_starts.get_device(),
            )
        ),
        compress_ratio,
        score_divisor,
        max_columns,
    )


#: Index of the cache-layout component in :func:`call_class_key` (the only
#: component that differs between ranks: it holds the cache's address).
CLASS_KEY_LAYOUT_INDEX = 5

#: process-wide prewarm record: (profile, cache_dtypes, page size, pos_dtypes,
#: device) -> built variant ids (None when ``prewarm`` raised). ``prewarm``
#: runs once per key per process, not once per layer.
_PREWARMED: dict[tuple, tuple[str, ...] | None] = {}


def index_page_of(cache) -> int | None:
    """Index page size of a bound compressed-key cache
    ``[blocks, block_size / compress_ratio, heads, head_dim]``: KV block
    16 / 32 / 64 -> page 4 / 8 / 16."""
    return int(cache.shape[1]) if getattr(cache, "ndim", 0) == 4 else None


def prewarm_once(
    prewarm, profile: str, cache, scale, *, scoped: bool = False
) -> tuple[str, ...] | None:
    """aiter ``prewarm`` for this cache's dtype / page size / device, once
    per process.

    Builds the kernel variants this cache can select -- with ``scoped``
    (aiter accepts ``compressed_k_cache=``) only its page size and dtype,
    else every variant of the profile for its dtype -- for ATOM's
    :data:`QUERY_POSITION_DTYPES`, after KV allocation and before CUDA-graph
    capture, so the first call of a variant does not compile inside serving.
    Logs one line (built variants and seconds). A raising ``prewarm`` is
    logged and NOT retried; fused calls then build their variant lazily on
    first use (the API 1.0 behaviour) -- no decision changes, so rank
    uniformity holds. aiter's prewarm launches on a private zero problem and
    keeps no persistent device workspace.
    """
    import torch

    cache_dtypes = (cache_dtype_name(cache, scale),)
    pos_dtypes = tuple(getattr(torch, n) for n in QUERY_POSITION_DTYPES)
    device = cache.device
    page = index_page_of(cache)
    key = (profile, cache_dtypes, page, pos_dtypes, str(device))
    if key in _PREWARMED:
        return _PREWARMED[key]
    kwargs = {
        "profile": profile,
        "cache_dtypes": cache_dtypes,
        "pos_dtypes": pos_dtypes,
        "device": device,
    }
    if scoped:
        kwargs.update(compressed_k_cache=cache, compressed_k_scale=scale)
    t0 = time.perf_counter()
    try:
        built = tuple(prewarm(**kwargs))
    except Exception as exc:  # noqa: BLE001 -- lazy build stays correct
        _PREWARMED[key] = None
        logger.warning(
            "QSA index scorer prewarm [%s, cache %s, page %s, pos %s, %s] failed "
            "after %.2f s (variants build lazily on first use): %r",
            profile,
            ",".join(cache_dtypes),
            page,
            ",".join(QUERY_POSITION_DTYPES),
            device,
            time.perf_counter() - t0,
            exc,
        )
        return None
    _PREWARMED[key] = built
    logger.info(
        "QSA index scorer prewarm [%s, cache %s, page %s, pos %s, %s]: %d "
        "variant(s) in %.2f s: %s",
        profile,
        ",".join(cache_dtypes),
        page,
        ",".join(QUERY_POSITION_DTYPES),
        device,
        len(built),
        time.perf_counter() - t0,
        ", ".join(built),
    )
    return built


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
    if got != SCORER_PARAMS:
        return f"positional params {got} != ATOM {SCORER_PARAMS}"
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
        if norm != SCORER_PARAMS:
            return f"ATOM_SCORER_SIGNATURE {norm} != ATOM {SCORER_PARAMS}"
    for name in ("decide", "layout_eligibility", "QSAIndexScorerUnsupported"):
        if not hasattr(api, name):
            return f"aiter module has no {name}"
    return None


def launch_signature_drift(api) -> str | None:
    """Why aiter's 1.1 ``launch`` is not ``(decision, <SCORER_PARAMS>, *,
    logits, visible_groups)``, or None."""
    fn = getattr(api, "launch", None)
    if not callable(fn):
        return "aiter module has no launch"
    try:
        params = list(inspect.signature(fn).parameters.values())
    except (TypeError, ValueError) as exc:
        return f"launch signature unreadable: {exc!r}"
    if not params or params[0].kind not in (
        inspect.Parameter.POSITIONAL_ONLY,
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
    ):
        return "launch takes no leading decision argument"
    rest = tuple(
        (p.name, _REQUIRED if p.default is p.empty else p.default)
        for p in params[1:]
        if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
    )
    if rest != SCORER_PARAMS:
        return f"launch positional params {rest} != ATOM {SCORER_PARAMS}"
    return None


class _NeverRaised(Exception):
    """``except`` target while no aiter module is bound (never raised)."""


#: Every route of this process (weak: layers own them), for the exit summary.
_ROUTES: weakref.WeakSet[QSAScorerRoute] = weakref.WeakSet()
_SUMMARY_REGISTERED = False


def process_summary() -> dict:
    """Per-(path, reason) call counts summed over every route of the process,
    plus the per-site entries. Decisions are counted per scorer call made in
    Python: a CUDA-graph replay runs no Python, so a captured call counts once
    (at capture), eager calls count every time."""
    total: collections.Counter = collections.Counter()
    sites = []
    for route in list(_ROUTES):
        total.update(route.counts)
        sites.append(route.summary())
    return {
        "paths": {f"{p}:{r}": n for (p, r), n in sorted(total.items())},
        "by_path": {p: sum(n for (q, _), n in total.items() if q == p) for p in PATHS},
        "sites": sorted(sites, key=lambda s: s["site"]),
    }


def log_process_summary() -> None:
    """One line at exit (registered only while the feature is active)."""
    if not _ROUTES:
        return
    summary = process_summary()
    logger.info(
        "QSA index scorer decisions: %s; per reason: %s",
        ", ".join(f"{p}={n}" for p, n in summary["by_path"].items()),
        summary["paths"] or "{}",
    )


def _register(route: QSAScorerRoute) -> None:
    global _SUMMARY_REGISTERED
    _ROUTES.add(route)
    if not _SUMMARY_REGISTERED:
        import atexit

        atexit.register(log_process_summary)
        _SUMMARY_REGISTERED = True


class QSAScorerRoute:
    """The scorer the QSA layer calls: aiter's op when the process capability
    and the call class allow it, else ATOM's stock scorer."""

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
        #: (reason, detail) pinning EVERY call to the stock scorer, or None.
        self.pinned: tuple[str, str] | None = None
        self._layout: dict[tuple, tuple[str, str]] = {}
        #: class key -> (reason, detail, aiter Decision when aiter serves the
        #: class else None, counts key (path, reason)).
        self._decisions: dict[tuple, tuple] = {}
        self._logged: set[str] = set()
        #: (path, reason) -> calls; path in :data:`PATHS`.
        self.counts: collections.Counter = collections.Counter()
        self.decide_calls = 0
        self.layout_calls = 0
        #: aiter callables resolved once (bound here, not looked up per call).
        self._op: Callable | None = None
        self._unsupported: type = _NeverRaised
        #: aiter API >= 1.1: ``launch(decision, ...)`` / ``prewarm(...)``.
        self._launch: Callable | None = None
        self._prewarm: Callable | None = None
        self._prewarm_scoped = False
        #: aiter serves uncovered classes on its own native path
        #: (``Decision.path``); False = interim: they run the stock scorer.
        self.aiter_native_path = False
        #: variant ids aiter's prewarm built for this process (None: not run).
        self.prewarmed: tuple[str, ...] | None = None
        self._select: Callable | None = None
        _register(self)
        # Process capability (checked once per process; an injected `api` --
        # CPU tests -- is checked here and never cached).
        from .capability import check_capability

        cap = check_capability(self.profile, api=api)
        self.capability = cap
        if not cap.ok:
            self._pin(cap.reason, cap.detail)
            return
        self.api = api = cap.api
        self._op = api.qsa_paged_mqa_logits
        self._unsupported = api.QSAIndexScorerUnsupported
        self._launch = cap.launch
        self._prewarm = cap.prewarm
        self._prewarm_scoped = cap.prewarm_scoped
        self.aiter_native_path = cap.aiter_native_path

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
        path = FUSED if reason == OK else STOCK
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

        Called from ``bind_caches`` (once per cache binding, at model load,
        before warmup and CUDA-graph capture); calls on a cache never
        prepared are checked lazily on their first call. With aiter API >=
        1.1, a cache that passes the layout check also prewarms aiter's
        kernel variants -- once per process (:func:`prewarm_once`), not per
        layer -- so no variant compiles inside serving.
        """
        if compressed_k_cache is None:
            return None
        reason = self._layout_decision(compressed_k_cache, compressed_k_scale)[0]
        if reason == OK and self._prewarm is not None:
            self.prewarmed = prewarm_once(
                self._prewarm,
                self.profile,
                compressed_k_cache,
                compressed_k_scale,
                scoped=self._prewarm_scoped,
            )
        return reason

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

    def _decide(self, args: tuple) -> tuple[str, str, Any]:
        """(reason, detail, aiter Decision or None) for one call class."""
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
        if layout[0] != OK and not self.aiter_native_path:
            return (layout[0], layout[1], None)
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
        return (d.reason.value, d.detail, d)

    def _classify(self, key: tuple, args: tuple) -> tuple:
        """Decide a new call class once and cache its entry (cold path)."""
        reason, detail, d = self._decide(args)
        # aiter >= the native-path release says which path its op takes for
        # this class; older aiter: ok = fused, anything else = not served.
        path = getattr(d, "path", None) if d is not None else None
        if path == "native" and self._launch is not None:
            # aiter serves the class on its own native path (no ATOM fallback).
            entry = (reason, detail, d, (AITER_NATIVE, reason))
        elif reason == OK and path in (None, "fused"):
            entry = (reason, detail, d, FUSED_OK)
        else:
            entry = (reason, detail, None, (STOCK, reason))
        if entry[3] != FUSED_OK:
            # Logged here, once: every later call of the class is a cache hit.
            self._log_once(reason, detail)
        if len(self._decisions) >= MAX_CACHED_DECISIONS:
            self._decisions.clear()
        self._decisions[key] = entry
        return entry

    def _repin(self, key: tuple, exc: BaseException) -> tuple:
        """aiter's O(1) launch guard refused a decided class before launching:
        run it on the stock scorer from now on."""
        reason = getattr(exc.reason, "value", str(exc.reason))
        detail = str(exc)
        entry = (reason, detail, None, (STOCK, reason))
        self._decisions[key] = entry
        self._log_once(reason, detail)
        return entry

    def decision_for(self, *args, **kwargs) -> tuple[str, str]:
        """(reason, detail) this route would take for a call (cached)."""
        a = self._bind(args, kwargs)
        if self.pinned is not None:
            return self.pinned
        key = call_class_key(*a[:10], a[11])
        entry = self._decisions.get(key)
        if entry is None:
            entry = self._classify(key, a)
        return (entry[0], entry[1])

    @staticmethod
    def _bind(args, kwargs) -> tuple:
        if not kwargs and len(args) == len(SCORER_PARAMS):
            return tuple(args)
        values = dict(zip((n for n, _ in SCORER_PARAMS), args))
        for k, v in kwargs.items():
            if k in values:
                raise TypeError(f"multiple values for argument {k!r}")
            values[k] = v
        out = []
        for name, default in SCORER_PARAMS:
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
    def __call__(
        self,
        q,
        compressed_k_cache,
        page_table,
        token_to_request,
        query_positions,
        context_lens,
        compress_ratio=4,
        score_divisor=None,
        max_columns=None,
        row_starts=None,
        allow_flydsl=False,
        compressed_k_scale=None,
    ):
        """Drop-in for ATOM's ``qsa_paged_mqa_logits`` (hot path).

        Parameters are :data:`SCORER_PARAMS` (Python binds them: no
        ``*args``/``**kwargs`` re-binding). A cached class costs one key
        tuple, one dict lookup and the aiter call.
        """
        if self.pinned is not None:
            reason = self.pinned[0]
            count_key = (STOCK, reason)
        else:
            key = call_class_key(
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
            )
            entry = self._decisions.get(key)
            if entry is None:
                entry = self._classify(
                    key,
                    (
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
                        allow_flydsl,
                        compressed_k_scale,
                    ),
                )
            decision = entry[2]
            if decision is not None:
                try:
                    if self._launch is not None:
                        out = self._launch(
                            decision,
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
                            allow_flydsl,
                            compressed_k_scale,
                        )
                    else:
                        out = self._op(
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
                            allow_flydsl,
                            compressed_k_scale,
                            profile=self.profile,
                        )
                except self._unsupported as exc:
                    entry = self._repin(key, exc)
                    reason, count_key = entry[0], entry[3]
                else:
                    if out is not None:
                        count_key = entry[3]
                        if self.capture is not None:
                            self._record(
                                entry[0],
                                count_key[0],
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
                                allow_flydsl,
                                compressed_k_scale,
                            )
                        self.counts[count_key] += 1
                        if OK not in self._logged and count_key is FUSED_OK:
                            self._log_once(OK, entry[1])
                        return out
                    reason, count_key = OK, (STOCK, OK)
            else:
                reason, count_key = entry[0], entry[3]
        if self.capture is not None:
            self._record(
                reason,
                STOCK,
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
                allow_flydsl,
                compressed_k_scale,
            )
        self.counts[count_key] += 1
        return self.native(
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
            allow_flydsl,
            compressed_k_scale,
        )

    def _record(self, reason: str, path: str, *args) -> None:
        """Capture hook (only reached when capture is on)."""
        self.capture.record(self.site, args, reason, path, self.profile)

    def select_paged_tokens(self, *args, **kwargs):
        """``qsa_select_paged_tokens`` with this route as its scorer."""
        select = self._select
        if select is None:  # first call: import once, keep the function
            from .select import qsa_select_paged_tokens_routed

            select = self._select = qsa_select_paged_tokens_routed
        return select(*args, scorer=self, **kwargs)

    def summary(self) -> dict:
        return {
            "site": self.site,
            "profile": self.profile,
            "pinned": None if self.pinned is None else list(self.pinned),
            "counts": {f"{p}:{r}": n for (p, r), n in sorted(self.counts.items())},
            "decide_calls": self.decide_calls,
            "layout_calls": self.layout_calls,
            "classes": len(self._decisions),
            "fused_call": "launch" if self._launch is not None else "op",
            "uncovered": AITER_NATIVE if self.aiter_native_path else STOCK,
            "prewarmed": None if self.prewarmed is None else list(self.prewarmed),
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
