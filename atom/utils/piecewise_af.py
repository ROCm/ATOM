# SPDX-License-Identifier: MIT
"""Shared input staging and live capture state for ``piecewise_core`` adapters.

Models supply the host-branch key, named inputs and a state factory. The state
provides a ``bind`` context manager for warmup/recording; publishing its Python
results into the live request remains the adapter's responsibility.
"""

from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass
from typing import Protocol

import torch

from atom.utils.cuda_graph import CudagraphCaptureRunner, StableOutputs
from atom.utils.graph_holders import register_graph_holder


def _signature(tensor: torch.Tensor) -> tuple:
    return (tuple(tensor.shape), tuple(tensor.stride()), tensor.dtype, tensor.device)


class SharedGraphInputs(StableOutputs):
    """Stable input addresses, optionally copied once per forward across layers.

    The caller owns and clears ``memo`` at the start of every forward, even when
    reusing request objects and tensor allocations. Inputs whose values change
    between layers must be listed in ``copy_each_call``. ``zero_copy`` inputs
    are read directly; their producer addresses become part of the graph key.
    """

    def __init__(self):
        super().__init__()
        register_graph_holder(self)

    def release_graphs(self):
        self._slots.clear()
        return 0

    def prepare(self, inputs, *, memo=None, copy_each_call=(), zero_copy=()):
        staged = {}
        for name, source in inputs.items():
            if not isinstance(source, torch.Tensor) or name in zero_copy:
                staged[name] = source
                continue
            key = (name, _signature(source))
            pointer = source.data_ptr()
            memoize = memo is not None and name not in copy_each_call
            if memoize and key in memo and memo[key][0] == pointer:
                staged[name] = memo[key][1]
                continue
            buffer = self.slot(key, source).copy_(source)
            if memoize:
                memo[key] = (pointer, buffer)
            staged[name] = buffer
        return staged


class CaptureState(Protocol):
    def bind(self) -> AbstractContextManager[dict]:
        """Provide fresh Python arguments for each warmup/recording execution."""
        ...


@dataclass
class CaptureBinding:
    inputs: dict
    arguments: dict
    state: CaptureState
    first_call: bool


class PiecewiseAFRunner(CudagraphCaptureRunner):
    """Bound live AF recordings while preserving the model's state lifecycle.

    Used with ``@piecewise_core(capture_prefill=True)`` after the adapter checks
    eligibility. ``key`` must include every host branch and external persistent
    address (for example the KV pool); input layouts and direct-read addresses
    are added here. Non-tensor inputs must be immutable and hashable.

    A full cache rejects new keys before allocating staging buffers. Existing
    keys still replay. Graph release also drops the associated capture states.
    """

    max_entries = 32

    def __init__(self, outputs=None, inputs=None):
        super().__init__()
        self.outputs = outputs if outputs is not None else StableOutputs()
        self.inputs = inputs if inputs is not None else SharedGraphInputs()
        self._captures: dict[tuple, CaptureState] = {}

    def release_graphs(self):
        self._captures.clear()
        return super().release_graphs()

    @contextmanager
    def bind_capture(
        self,
        key: tuple,
        sources: dict,
        make_state: Callable[[dict], CaptureState],
        *,
        memo=None,
        copy_each_call=(),
        zero_copy=(),
    ) -> Iterator[CaptureBinding | None]:
        signature = tuple(
            (
                name,
                (
                    (_signature(value), value.data_ptr() if name in zero_copy else None)
                    if isinstance(value, torch.Tensor)
                    else value
                ),
            )
            for name, value in sources.items()
        )
        key = (key, signature)
        first_call = not self.has_graph(key)
        if first_call and len(self._graphs) >= self.max_entries:
            yield None
            return
        inputs = self.inputs.prepare(
            sources, memo=memo, copy_each_call=copy_each_call, zero_copy=zero_copy
        )
        state = make_state(inputs) if first_call else self._captures[key]
        yield CaptureBinding(
            inputs,
            {
                "capture": True,
                "capture_key": key,
                "prepare": state.bind,
                "runner": self,
                "outputs": self.outputs,
            },
            state,
            first_call,
        )
        # An exception or a decorator-level fallback must not install a state
        # for a graph that was never recorded.
        if first_call and self.has_graph(key):
            self._captures[key] = state
