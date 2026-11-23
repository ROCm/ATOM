# SPDX-License-Identifier: MIT
"""Native-collective transition, contract 1, eager or whole request graph.

One owner registers both phases. A capture transaction resolves and agrees on the
route in the bucket warmup, before the current stream enters capture. Recording
reuses that decision; replay executes only captured GPU work. The native combine,
mHC, and cache ownership do not change. No Python handle crosses the model ABI.
"""

from collections.abc import Callable
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass

import torch

SEAM = "v41.block.transition"
WINDOW = "v41.moe_to_attn_qkv"


@dataclass(frozen=True)
class Phases:
    pairs: tuple[tuple[int, int], ...]
    begin: Callable
    end: Callable


@dataclass
class _Step:
    owner: str
    phases: Phases
    pending: dict
    completed: set
    identity: bool = False
    bucket: int | None = None


@dataclass
class _Capture:
    rows: int
    prepared: bool = False
    route: tuple | None = None
    # Keep the table object alive; a new arm/plan always publishes a new dict.
    table: object = None


_step = ContextVar("atom_v41_transition_step", default=None)
_capture = ContextVar("atom_v41_transition_capture", default=None)


@contextmanager
def capture_transaction(rows):
    """Scope exactly one request bucket's warmup and graph record."""
    if _capture.get() is not None:
        raise RuntimeError("nested transition capture transaction")
    prior = _capture.set(_Capture(rows))
    try:
        yield
    finally:
        _capture.reset(prior)


def register_window(
    *,
    window,
    contract,
    owner,
    pairs,
    begin,
    end,
    graph="eager",
    collectives="native",
    owns_between=(),
    requires_api=">=2,<3",
):
    from atom import plugin_host as host

    def refuse(reason):
        host._refuse(SEAM, owner, reason, f"window={window!r}")

    if window != WINDOW:
        refuse("unknown-window")
    if graph not in host.GRAPHS or collectives != "native" or tuple(owns_between):
        refuse("window-ownership-refused")
    if not callable(begin) or not callable(end):
        refuse("not-callable")
    pairs = tuple(tuple(p) for p in pairs)
    if (
        not pairs
        or any(
            len(p) != 2
            or any(type(i) is not int for i in p)
            or p[0] < 0
            or p[1] != p[0] + 1
            for p in pairs
        )
        or len(set(pairs)) != len(pairs)
    ):
        refuse("not-contiguous")
    return host.register(
        seam=SEAM,
        contract=contract,
        owner=owner,
        graph=graph,
        impl=Phases(tuple(sorted(pairs)), begin, end),
        requires_api=requires_api,
    )


def _agree_route(host, eligible, rows, agree, bucket):
    routes = host._routes(SEAM, bucket is not None, bucket)
    route = routes[0] if len(routes) == 1 else None
    arm = host._arm
    phases = route.impl if route is not None else None
    token = None
    if (
        isinstance(phases, Phases)
        and rows > 0
        and (bucket is None or route.graph == "capture_safe")
        and eligible(phases.pairs)
    ):
        token = (host._requested, arm, route.owner, phases.pairs, int(rows))
    strict = arm != "native" and bool(routes) and host._on_decline == "error"
    vote = (token, strict)
    votes = tuple(agree(vote))
    if token is not None and votes and all(v == vote for v in votes):
        return route.owner, phases, arm == "identity"
    why = (
        "rank-plan-mismatch"
        if any(v and v[0] is not None for v in votes)
        else "pair-declined"
    )
    host._record(
        host.Decision(SEAM, "native", why, captured=bucket is not None, bucket=bucket)
    )
    if any(v and v[1] for v in votes):
        raise host.PluginDeclined(f"{WINDOW}: {why}")
    return None


@contextmanager
def scope(*, eligible, rows, agree):
    """Pin one phase pair (or a whole native fallback) before native work starts."""
    from atom import plugin_host as host

    if _step.get() is not None:
        raise host.PluginContractError("nested transition scope")
    if torch.compiler.is_compiling():
        yield
        return
    captured, bucket = host._graph_state()
    recording = host._stream_capturing()
    txn = _capture.get()
    if captured or recording:
        if txn is None:
            host._record(
                host.Decision(
                    SEAM,
                    "native",
                    "no-capture-transaction",
                    captured=True,
                    bucket=bucket,
                )
            )
            yield
            return
        if rows != txn.rows or (bucket is not None and bucket != rows):
            raise host.PluginContractError(f"{WINDOW}: capture bucket changed")
        if not txn.prepared:
            if recording:
                raise host.PluginContractError(
                    f"{WINDOW}: capture requires a warmup vote"
                )
            txn.route = _agree_route(host, eligible, rows, agree, rows)
            txn.table = host._active
            txn.prepared = True
        if txn.table is not host._active:
            raise host.PluginContractError(
                f"{WINDOW}: plan changed between warmup and record"
            )
        chosen, bucket = txn.route, rows
    else:
        chosen = _agree_route(host, eligible, rows, agree, None)
        bucket = None
    if chosen is None:
        yield
        return
    owner, phases, identity = chosen
    step = _Step(owner, phases, {}, set(), identity, bucket)
    saved = _step.set(step)
    try:
        yield
        if step.pending or step.completed != set(step.phases.pairs):
            raise host.PluginContractError(f"{WINDOW}: incomplete phase pair")
    finally:
        _step.reset(saved)


def _compatible(out, ref):
    return (
        out is None
        if ref is None
        else isinstance(out, torch.Tensor)
        and (out.shape, out.dtype, out.device, out.stride())
        == (ref.shape, ref.dtype, ref.device, ref.stride())
    )


def begin(layer, routed, shared):
    from atom import plugin_host as host

    if torch.compiler.is_compiling():
        return routed, shared
    step = _step.get()
    if step is None or not isinstance(layer, int):
        return routed, shared
    pair = (layer, layer + 1)
    if pair not in step.phases.pairs:
        return routed, shared
    if pair in step.pending or pair in step.completed:
        raise host.PluginContractError(f"{WINDOW}: duplicate begin {pair}")
    handle = {}
    out = (
        (routed, shared)
        if step.identity
        else step.phases.begin(routed, shared, handle=handle)
    )
    if (
        not isinstance(out, tuple)
        or len(out) != 2
        or not _compatible(out[0], routed)
        or not _compatible(out[1], shared)
    ):
        raise host.PluginContractError(
            f"{WINDOW}: begin changed native combine contract"
        )
    step.pending[pair] = handle
    host._record(
        host.Decision(
            SEAM + ".begin",
            "plugin",
            "identity" if step.identity else "selected",
            step.owner,
            captured=step.bucket is not None,
            bucket=step.bucket,
        )
    )
    return out


def end(layer, project, hidden, scale, output_sizes):
    from atom import plugin_host as host

    if torch.compiler.is_compiling():
        return project(hidden, scale)
    step = _step.get()
    if step is None or not isinstance(layer, int):
        return project(hidden, scale)
    pair = (layer - 1, layer)
    if pair not in step.phases.pairs:
        return project(hidden, scale)
    if pair not in step.pending:
        raise host.PluginContractError(f"{WINDOW}: end without begin {pair}")
    handle = step.pending.pop(pair)
    # The native projector is supplied as a named callable, not a model object.
    # Providers may call it with equivalent inputs or replace it entirely.
    sizes = tuple(output_sizes)
    out = (
        project(hidden, scale)
        if step.identity
        else step.phases.end(
            hidden, scale, project=project, output_sizes=sizes, handle=handle
        )
    )
    if (
        len(sizes) != 2
        or not isinstance(out, tuple)
        or len(out) != 2
        or any(
            not isinstance(t, torch.Tensor)
            or t.device != hidden.device
            or t.shape != (*hidden.shape[:-1], size)
            for t, size in zip(out, sizes)
        )
    ):
        raise host.PluginContractError(
            f"{WINDOW}: end must return two projected tensors"
        )
    step.completed.add(pair)
    host._record(
        host.Decision(
            SEAM + ".end",
            "plugin",
            "identity" if step.identity else "selected",
            step.owner,
            captured=step.bucket is not None,
            bucket=step.bucket,
        )
    )
    return out
