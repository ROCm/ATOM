# SPDX-License-Identifier: MIT
"""Metadata interface and live capture engine for :func:`piecewise_core`."""

from collections.abc import Callable
from dataclasses import dataclass

import torch

from atom.utils.cuda_graph import CudagraphCaptureRunner, StableOutputs
from atom.utils.graph_holders import register_graph_holder


@dataclass
class PiecewiseMetadata:
    """Describe dynamic metadata without implementing a graph runner.

    ``inputs`` is an explicit dict/list/tuple tree of tensors and immutable
    scalar leaves. Tensor layouts, tree structure and scalar values are keyed
    automatically. ``key`` adds host branches and persistent addresses outside
    that tree (such as the KV pool). Do not put arbitrary request objects or
    workspaces in the tree; keep them in the model's reconstruction function.

    ``rebuild(staged_inputs)`` returns keyword overrides for the decorated core.
    It runs separately for warmup and recording, with fresh tree containers, and
    must create fresh derived Python state. Tensor inputs are read-only; device
    side effects belong in persistent state such as the KV pool.

    ``publish(overrides)`` copies only this core's Python results into the CURRENT
    live metadata. The engine supplies warmup results on the first call and
    recorded results on replay. It never retains this publication callback.

    Optional ``memo`` shares immutable-per-forward inputs across serial layers;
    its owner MUST clear it at each forward, including reused request objects.
    ``copy_each_call`` lists paths whose contents may change within a forward.
    Without a memo every tensor is refreshed on every call.
    """

    key: tuple
    inputs: dict
    rebuild: Callable[[dict], dict]
    publish: Callable[[dict], None] | None = None
    memo: dict | None = None
    copy_each_call: tuple[tuple, ...] = ()


def _signature(tensor: torch.Tensor) -> tuple:
    return (tuple(tensor.shape), tuple(tensor.stride()), tensor.dtype, tensor.device)


def _flatten(tree, leaves, path=()):
    """Only traverse the declared tree; never inspect arbitrary model objects."""
    if type(tree) is dict:
        return (
            dict,
            tuple((k, _flatten(v, leaves, (*path, k))) for k, v in tree.items()),
        )
    if type(tree) in (tuple, list):
        return (
            type(tree),
            tuple(_flatten(v, leaves, (*path, i)) for i, v in enumerate(tree)),
        )
    if not isinstance(tree, (torch.Tensor, str, int, float, bool, type(None))):
        raise TypeError(
            f"Piecewise metadata at {path!r} must be a tensor or scalar, "
            f"got {type(tree).__name__}"
        )
    leaves[path] = tree
    return None


def _unflatten(structure, leaves, path=("metadata",)):
    if structure is None:
        return leaves[path]
    kind, children = structure
    if kind is dict:
        return {k: _unflatten(v, leaves, (*path, k)) for k, v in children}
    return kind(_unflatten(v, leaves, (*path, i)) for i, v in enumerate(children))


class SharedGraphInputs(StableOutputs):
    """Stable input addresses, optionally copied once per forward across layers."""

    def __init__(self):
        super().__init__()
        self._generation = object()
        register_graph_holder(self)

    def release_graphs(self):
        self._slots.clear()
        self._generation = object()
        return 0

    def prepare(self, inputs, *, memo=None, copy_each_call=(), zero_copy=()):
        staged = {}
        for name, source in inputs.items():
            if not isinstance(source, torch.Tensor) or name in zero_copy:
                staged[name] = source
                continue
            # Scope the memo to this store, including after a graph release.
            key = (self._generation, name, _signature(source))
            pointer = source.data_ptr()
            memoize = memo is not None and name not in copy_each_call
            cached = memo.get(key) if memoize else None
            if cached is not None and cached[0] == pointer:
                staged[name] = cached[1]
                continue
            buffer = self.slot(key, source).copy_(source)
            if memoize:
                memo[key] = (pointer, buffer)
            staged[name] = buffer
        return staged


class PiecewiseAFRunner(CudagraphCaptureRunner):
    """Bound live recordings and manage metadata for the decorator.

    No subclass is needed. An explicit instance is optional; supply one when
    sharing input storage or using fixed-capacity output slots across layers.
    A full cache falls back before staging new inputs; existing keys replay.
    """

    max_entries = 32

    def __init__(self, outputs=None, inputs=None, *, token_dim=0):
        super().__init__()
        self.outputs = (
            outputs if outputs is not None else StableOutputs(token_dim=token_dim)
        )
        self._owns_inputs = inputs is None
        self.inputs = inputs if inputs is not None else SharedGraphInputs()
        self._captures: dict[tuple, dict] = {}

    def release_graphs(self):
        self._captures.clear()
        count = super().release_graphs()
        if self._owns_inputs:
            self.inputs.release_graphs()
        return count

    def run(self, key, tensors, metadata, core, out_slot, *, zero_copy, config_names):
        """Return ``(handled, result)``; only cache misses execute Python math."""
        sources = {("tensor", name): value for name, value in tensors.items()}
        structure = _flatten(metadata.inputs, sources, ("metadata",))
        direct = frozenset(("tensor", name) for name in zero_copy)
        signature = tuple(
            (
                name,
                (
                    (_signature(value), value.data_ptr() if name in direct else None)
                    if isinstance(value, torch.Tensor)
                    else (type(value), value)
                ),
            )
            for name, value in sources.items()
        )
        key = (key, metadata.key, structure, signature)
        hash(key)  # Reject an invalid host key before allocating or warming.
        first_call = not self.has_graph(key)
        if first_call and len(self._graphs) >= self.max_entries:
            return False, None
        staged = self.inputs.prepare(
            sources,
            memo=metadata.memo,
            copy_each_call=(
                *(("tensor", name) for name in tensors),
                *(("metadata", *path) for path in metadata.copy_each_call),
            ),
            zero_copy=direct,
        )
        if first_call:
            executions = []

            def compute(**values):
                overrides = metadata.rebuild(_unflatten(structure, staged))
                if not isinstance(overrides, dict) or overrides.keys() - config_names:
                    raise ValueError(
                        "Piecewise metadata rebuild must return only declared "
                        "non-tensor keyword arguments"
                    )
                result = core(**values, **overrides)
                executions.append(overrides)
                return result

            result = self.capture_on_demand(
                key,
                {name: staged["tensor", name] for name in tensors},
                {},
                compute,
                out_slot,
            )
            self._captures[key] = executions[-1]
            published = executions[0]
        else:
            result = self.replay(key, {})  # All refreshes happened above.
            published = self._captures[key]
        if metadata.publish is not None:
            metadata.publish(published)
        return True, result
