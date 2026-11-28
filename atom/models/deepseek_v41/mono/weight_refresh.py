# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors
"""V4.1 mono under in-place weight updates: the weights its launches read
beside the parameters, rebuilt in place after an update, and the audit that no
launch reads any other copy.

A weight update (the rollout sync, ``weight_updater``; a weight-cache daemon's
``load``) rewrites the model's parameters IN PLACE, so a captured graph and a
mono pointer table keep their addresses. A V4.1 mono launch reads each weight
one of two ways, and only these two:

- **in place**: the pointer lies in a parameter's storage (a kernel reads the
  parameter as ATOM keeps it, a dtype converted in registers): the update is
  seen on the next launch;
- **derived**: the pointer lies in a buffer a registered ``DerivedWeight``
  writes (hi's re-laid-out MXFP8 packs, its head): after an update the registry
  rebuilds it IN PLACE, from the parameters it names as its sources, into the
  same storage.

Anything else -- a copy made at bind time that nothing refreshes -- would go on
serving the old weights with no shape or dtype check able to notice. A bound
path declares every weight pointer it hands a launch (its non-weight pointers
-- scratch, activations, mailboxes, the KV cache -- are the launch's other
arguments) and ``DerivedWeights.audit`` classifies each: a pointer that is
neither in place nor derived is a refusal naming it. Under a runner that takes
weight updates (``strict``) the refusal turns that path off at bind; under any
other the verdict is kept for ``weight_update_refusal``, which a writer asks
before it writes.

The routing wrapper (``dispatch.MonoDecodeModel``) answers the update path's
model hooks with this registry: ``weight_update_refusal()`` before a sync
writes, ``after_weight_update(changed)`` once it has (``weight_updater``).
Rebuilds run once per update, never per step: re-deriving every copy each step
would cost hi about 3 GB of traffic a step.
"""

from __future__ import annotations

import bisect
import logging
import time
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field

import torch

from atom.mono.runtime.consensus import MonoUnsupported

logger = logging.getLogger("atom")


def byte_span(t: torch.Tensor) -> tuple[int, int]:
    """The bytes ``t`` reads, [first, last + 1): its extent in its storage
    (a strided view's too: the farthest element its strides reach)."""
    start = t.data_ptr()
    if t.numel() == 0:
        return start, start
    last = sum((n - 1) * abs(s) for n, s in zip(t.shape, t.stride()))
    return start, start + (last + 1) * t.element_size()


def _storage_span(t: torch.Tensor) -> tuple[int, int]:
    storage = t.untyped_storage()
    return storage.data_ptr(), storage.data_ptr() + storage.nbytes()


class _Spans:
    """Byte ranges, each with a label: which one holds a given range."""

    def __init__(self) -> None:
        self._spans: list[tuple[int, int, str]] = []
        self._starts: list[int] | None = None
        self._longest = 0

    def add(self, span: tuple[int, int], label: str) -> None:
        if span[1] > span[0]:
            self._spans.append((*span, label))
            self._starts = None
            self._longest = max(self._longest, span[1] - span[0])

    def holder(self, span: tuple[int, int]) -> str | None:
        """The label of a range holding ``span`` entirely, or None."""
        if self._starts is None:
            self._spans.sort()
            self._starts = [s for s, _, _ in self._spans]
        i = bisect.bisect_right(self._starts, span[0])
        # ranges may nest or overlap (a view's beside its storage's): scan back
        # over every range that starts close enough to reach it
        while i > 0:
            i -= 1
            start, end, label = self._spans[i]
            if span[0] - start > self._longest:
                break
            if span[1] <= end:
                return label
        return None


def _overlaps(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return a[0] < b[1] and b[0] < a[1]


class ParameterStorage:
    """The storage of every parameter of ``models`` (None skipped): whether a
    tensor lies in one (``t in storage``), i.e. reads a weight in place."""

    def __init__(self, *models) -> None:
        self._spans = _Spans()
        for model in models:
            if model is not None:
                for p in model.parameters():
                    self._spans.add(_storage_span(p), "parameter")

    def __contains__(self, t: torch.Tensor) -> bool:
        return self._spans.holder(byte_span(t)) is not None


@dataclass(frozen=True)
class DerivedWeight:
    """One derived buffer set: ``rebuild()`` re-derives ``dests`` from
    ``sources`` (the model's parameters, or views of them), writing into the
    same storage -- the addresses a pointer table or a captured graph holds."""

    name: str
    sources: tuple[torch.Tensor, ...]
    dests: Mapping[str, torch.Tensor]
    rebuild: Callable[[], None]

    def reads_any(self, changed: list[tuple[int, int]]) -> bool:
        """Whether a source overlaps any of the ``changed`` byte ranges."""
        return any(
            _overlaps(byte_span(src), span) for src in self.sources for span in changed
        )


@dataclass(frozen=True)
class RebuildReport:
    """What one ``rebuild`` re-derived and its wall time (device work
    included)."""

    rebuilt: tuple[str, ...]
    seconds: float


@dataclass
class DerivedWeights:
    """A mono deployment's derived weights and its bound paths' audits.

    ``strict``: the deployment's runner takes weight updates (``deployment.
    _takes_weight_updates``), so an audit that fails refuses the path it audits
    (``MonoUnsupported``, raised from its bind) instead of only being kept."""

    strict: bool = False
    entries: list[DerivedWeight] = field(default_factory=list)
    # bound path -> its audit's refusal, None where it passed
    audits: dict[str, str | None] = field(default_factory=dict)
    # every registered destination's bytes
    _dests: _Spans = field(default_factory=_Spans, repr=False)

    def register(
        self,
        name: str,
        sources: Iterable[torch.Tensor],
        dests: Mapping[str, torch.Tensor],
        rebuild: Callable[[], None],
    ) -> DerivedWeight:
        """Register ``dests`` as derived from ``sources``, re-derived in place
        by ``rebuild()``. A destination inside a source would make the rebuild
        read what it writes."""
        sources, dests = tuple(sources), dict(dests)
        if not sources or not dests:
            raise ValueError(f"derived weight {name}: no sources or no destinations")
        for key, dest in dests.items():
            span = byte_span(dest)
            if any(_overlaps(span, byte_span(src)) for src in sources):
                raise ValueError(
                    f"derived weight {name}: destination {key} overlaps a source"
                )
        entry = DerivedWeight(name, sources, dests, rebuild)
        self.entries.append(entry)
        for key, dest in dests.items():
            self._dests.add(byte_span(dest), f"{name}:{key}")
        return entry

    def audit(
        self,
        path: str,
        weights: Mapping[str, torch.Tensor],
        params: Iterable[torch.Tensor],
    ) -> str | None:
        """Every weight pointer bound path ``path`` hands a launch
        (``weights``, by a name the refusal can quote) classified against
        ``params`` (the parameters of every model the path reads) and the
        registered destinations; the derived entries' sources must lie in
        ``params`` too. Returns the refusal (None: every pointer is in place or
        derived), recorded for ``refusal``; raised as ``MonoUnsupported`` when
        ``strict``."""
        in_params = _Spans()
        for i, p in enumerate(params):
            in_params.add(_storage_span(p), f"parameter {i}")
        derived = self._dests
        why = None
        counts = {"in place": 0, "derived": 0}
        for name, t in weights.items():
            span = byte_span(t)
            if in_params.holder(span) is not None:
                counts["in place"] += 1
            elif derived.holder(span) is not None:
                counts["derived"] += 1
            else:
                why = (
                    f"{path}: weight {name} ({t.dtype} {tuple(t.shape)} at "
                    f"{t.data_ptr():#x}) lies in no parameter's storage and no "
                    f"registered derived buffer, a copy an in-place weight "
                    f"update cannot reach"
                )
                break
        if why is None:
            for entry in self.entries:
                bad = next(
                    (
                        i
                        for i, src in enumerate(entry.sources)
                        if in_params.holder(byte_span(src)) is None
                    ),
                    None,
                )
                if bad is not None:
                    src = entry.sources[bad]
                    why = (
                        f"{path}: derived weight {entry.name} reads source {bad} "
                        f"({src.dtype} {tuple(src.shape)} at {src.data_ptr():#x}), "
                        f"which lies in no parameter's storage: its rebuild "
                        f"would re-derive it from a stale copy"
                    )
                    break
        self.audits[path] = why
        if why is not None:
            if self.strict:
                raise MonoUnsupported(f"weight-update audit: {why}")
            logger.info(
                "mono weight-update audit, kept for weight_update_refusal: %s", why
            )
        else:
            logger.info(
                "mono weight-update audit: %s: %d weight pointers, %d in place, "
                "%d in derived buffers (%d derived entries registered)",
                path,
                len(weights),
                counts["in place"],
                counts["derived"],
                len(self.entries),
            )
        return why

    def registered(self, t: torch.Tensor) -> bool:
        """Whether ``t`` lies in a registered destination."""
        return self._dests.holder(byte_span(t)) is not None

    def refuse(self, path: str, why: str) -> None:
        """Record bound path ``path`` as one an in-place weight update cannot
        reach for a reason no pointer shows (a weight it takes per step from
        a derived cache it does not own); raised as ``MonoUnsupported`` when
        ``strict``."""
        why = f"{path}: {why}"
        self.audits[path] = why
        if self.strict:
            raise MonoUnsupported(f"weight-update audit: {why}")
        logger.info("mono weight-update audit, kept for weight_update_refusal: %s", why)

    def refusal(self) -> str | None:
        """The first bound path's audit refusal, None when every one passed."""
        return next((why for why in self.audits.values() if why is not None), None)

    def rebuild(self, changed: Iterable[torch.Tensor] | None = None) -> RebuildReport:
        """Re-derive, in place, every entry that reads any of the ``changed``
        tensors (None: every entry). Each destination keeps its address."""
        spans = None if changed is None else [byte_span(t) for t in changed]
        selected = [e for e in self.entries if spans is None or e.reads_any(spans)]
        t0 = time.perf_counter()
        for entry in selected:
            addresses = {k: v.data_ptr() for k, v in entry.dests.items()}
            entry.rebuild()
            moved = [k for k, v in entry.dests.items() if v.data_ptr() != addresses[k]]
            if moved:
                raise RuntimeError(
                    f"derived weight {entry.name}: its rebuild moved {moved}; a "
                    f"pointer table or captured graph still holds the old address"
                )
        if selected and torch.cuda.is_available():
            torch.cuda.synchronize()
        return RebuildReport(tuple(e.name for e in selected), time.perf_counter() - t0)


def registries_of(model) -> list[DerivedWeights]:
    """The ``DerivedWeights`` of every V4.1 mono deployment in ``model``'s
    module tree (a routing wrapper's ``mono_derived``), each once."""
    found: list[DerivedWeights] = []
    modules = getattr(model, "modules", None)
    for module in modules() if callable(modules) else ():
        registry = module.__dict__.get("mono_derived")
        if isinstance(registry, DerivedWeights) and not any(
            registry is r for r in found
        ):
            found.append(registry)
    return found


def weight_update_refusal(model) -> str | None:
    """Why an in-place weight update of ``model`` would leave a mono launch
    reading stale weights (a bound path whose audit failed), or None. Ask it
    before writing anything."""
    for registry in registries_of(model):
        why = registry.refusal()
        if why is not None:
            return why
    return None


def rebuild_after_weight_update(
    model, changed: Iterable[torch.Tensor] | None = None
) -> RebuildReport:
    """Re-derive, in place, the V4.1 mono derived weights of ``model`` whose
    sources overlap the ``changed`` tensors (None: every one) -- what a weight
    writer has rebuilt once its writes are done (the rollout sync at its last
    bucket, through the wrapper's ``after_weight_update``; a weight-cache
    daemon's ``load``). The caller has waited for
    the device's readers (a replay still in flight would read a half-written
    buffer) and refused the update first if ``weight_update_refusal`` said so.
    Collective-free: every rank re-derives its own buffers from its own
    parameters."""
    changed = None if changed is None else list(changed)
    reports = [registry.rebuild(changed) for registry in registries_of(model)]
    return RebuildReport(
        tuple(name for r in reports for name in r.rebuilt),
        sum(r.seconds for r in reports),
    )
