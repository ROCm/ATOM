# SPDX-License-Identifier: MIT
"""Live piecewise graphs over builder-owned buffers and per-forward caches."""

from collections.abc import MutableMapping
from contextlib import contextmanager
from contextvars import ContextVar

import torch

from atom.utils.cuda_graph import CudagraphCaptureRunner, StableOutputs
from atom.utils.graph_holders import register_graph_holder

_recording = ContextVar("piecewise_recording", default=None)
_missing = object()


def buffer_key(value):
    """Key explicit buffer trees by address/layout and host values, never data.

    Builders call this once when publishing a view. Persistent tensors are read
    directly, just as in startup decode capture. This does not walk arbitrary
    model/request objects or rebuild metadata.
    """
    if isinstance(value, torch.Tensor):
        return (
            value.data_ptr(),
            tuple(value.shape),
            tuple(value.stride()),
            value.dtype,
            value.device,
        )
    if isinstance(value, dict):
        return tuple((k, buffer_key(v)) for k, v in value.items())
    if isinstance(value, (tuple, list)):
        return (type(value), tuple(buffer_key(v) for v in value))
    if isinstance(value, (str, int, float, bool, type(None))) or value is _missing:
        return (type(value), value)
    raise TypeError(f"Unsupported piecewise buffer/key value: {type(value).__name__}")


class PiecewiseBuffers(StableOutputs):
    """Builder-owned storage for views/results without an existing fixed buffer.

    Use distinct namespaces for simultaneously live views (including TBO).
    Existing builder buffers must be passed directly, without another copy.
    """

    def __init__(self):
        super().__init__()
        register_graph_holder(self)

    def release_graphs(self):
        self._slots.clear()
        return 0


class ReplayCache(MutableMapping):
    """A forward-local dictionary whose Python writes accompany graph replay.

    Reads are dependencies; only reads before the core's own write are keyed.
    Warmup and recording run against the SAME entry state. Writes are restored
    afterwards and published from warmup on the first call, recording on replay.
    Values are immutable scalar/tensor trees. Tensor contents must not be mutated
    by Python between layers; persistent GPU state belongs in builder buffers.

    ``stable=True`` delivers tensor results to builder-owned slots. This lets
    eager, newly captured and replayed producers share downstream graph inputs.
    """

    def __init__(self, values=(), *, stable=False):
        self._data = dict(values)
        self.stable = stable
        self._storage = None
        self._name = None

    def _read(self, key):
        record = _recording.get()
        if record is not None:
            record.read(self, key)

    def __getitem__(self, key):
        self._read(key)
        return self._data[key]

    def __iter__(self):
        self._read(_missing)
        return iter(self._data)

    def __len__(self):
        self._read(_missing)
        return len(self._data)

    def bind(self, storage, namespace):
        """Bind result slots before any producer (including an eager gather)."""
        self._storage, self._name = storage, namespace

    def __setitem__(self, key, value):
        record = _recording.get()
        if record is not None:
            record.write(self, key)
        if self.stable and self._storage is not None:
            if not isinstance(value, torch.Tensor):
                raise TypeError("Stable ReplayCache results must be tensors")
            slot = self._storage.slot((self._name, key), value)
            if slot.data_ptr() != value.data_ptr():
                slot.copy_(value)
            value = slot
        self._data[key] = value

    def __delitem__(self, key):
        record = _recording.get()
        if record is not None:
            record.write(self, key)
        del self._data[key]

    def clear(self):
        if _recording.get() is None:
            self._data.clear()
        else:
            super().clear()


class _Transaction:
    def __init__(self, caches):
        self.caches = caches
        self.names = {id(cache): name for name, cache in caches.items()}
        self.reads, self.before, self.writes, self.orders = {}, {}, {}, {}

    def _path(self, cache, key):
        if id(cache) not in self.names:
            raise ValueError("Core accessed a ReplayCache outside its capture state")
        return self.names[id(cache)], key

    def read(self, cache, key):
        path = self._path(cache, key)
        if key is _missing:
            self.reads[path] = self.orders.get(path[0], tuple(cache._data))
        elif path not in self.before:
            self.reads[path] = buffer_key(cache._data.get(key, _missing))

    def write(self, cache, key):
        path = self._path(cache, key)
        if path[0] not in self.orders:
            self.orders[path[0]] = tuple(cache._data)
        if path not in self.before:
            self.before[path] = cache._data.get(key, _missing)

    def finish(self):
        for (name, key), previous in self.before.items():
            data = self.caches[name]._data
            self.writes[name, key] = data.get(key, _missing)
            if previous is _missing:
                data.pop(key, None)
            else:
                data[key] = previous
        for name, order in self.orders.items():
            cache = self.caches[name]
            cache._data = {key: cache._data[key] for key in order}


class PiecewiseState:
    """The builder's public contract: a complete key, buffers, and result caches.

    Prepare once per forward/view, after publishing the buffers. ``key`` covers
    host branches; buffer addresses/layouts are added automatically. Capture
    binds the live objects unchanged. There are no reconstruction/publication
    callbacks, field selectors or model-specific graph classes.
    """

    def __init__(self, key, buffers, caches=(), *, storage=None, namespace=()):
        self.buffers = buffers  # Keep captured device allocations alive.
        self.key = (key, buffer_key(buffers))
        hash(self.key)
        self.caches = dict(caches)
        if len({id(v) for v in self.caches.values()}) != len(self.caches):
            raise ValueError("Name each shared ReplayCache only once")
        for name, cache in self.caches.items():
            if not isinstance(cache, ReplayCache):
                raise TypeError("PiecewiseState caches must be ReplayCache instances")
            cache.bind(storage, (namespace, name))

    @contextmanager
    def record(self):
        if _recording.get() is not None:
            raise RuntimeError("Nested piecewise state recording is unsupported")
        transaction = _Transaction(self.caches)
        token = _recording.set(transaction)
        try:
            yield transaction
        finally:
            _recording.reset(token)
            transaction.finish()

    def matches(self, reads):
        for (name, key), expected in reads.items():
            data = self.caches[name]._data
            actual = (
                tuple(data) if key is _missing else buffer_key(data.get(key, _missing))
            )
            if actual != expected:
                return False
        return True

    def publish(self, writes):
        # These copies already ran on the GPU. Rebind Python results without
        # launching another copy or overwriting unrelated inputs.
        for (name, key), value in writes.items():
            data = self.caches[name]._data
            if value is _missing:
                data.pop(key, None)
            else:
                data[key] = value


class PiecewiseAFRunner(CudagraphCaptureRunner):
    """Bound live recordings over stable builder metadata, with eager fallback."""

    max_entries = 32

    def __init__(self, outputs=None, *, token_dim=0):
        super().__init__()
        self.outputs = (
            outputs if outputs is not None else StableOutputs(token_dim=token_dim)
        )
        self._captures = {}

    def release_graphs(self):
        self._captures.clear()
        return super().release_graphs()

    def run(self, key, tensors, state, core, out_slot, *, zero_copy):
        direct = {name: value for name, value in tensors.items() if name in zero_copy}
        layouts = tuple(
            (name, tuple(value.shape), tuple(value.stride()), value.dtype, value.device)
            for name, value in tensors.items()
            if isinstance(value, torch.Tensor)
        )
        key = (key, state.key, layouts, buffer_key(direct))
        entries = self._captures.get(key, ())
        for graph_key, reads, writes in entries:
            if state.matches(reads):
                result = self.replay(graph_key, tensors)
                state.publish(writes)
                return True, result
        if len(self._graphs) >= self.max_entries:
            return False, None
        graph_key = len(self._graphs)
        executions = []

        def compute(**values):
            with state.record() as transaction:
                result = core(**values)
            executions.append(transaction)
            return result

        read_from, refresh = self.input_buffers(tensors, tuple(tensors), zero_copy)
        result = self.capture_on_demand(
            graph_key, read_from, refresh, compute, out_slot
        )
        warm, recorded = executions
        self._graphs[graph_key]["state"] = state
        # Retain non-tensor resources whose addresses were captured too.
        self._graphs[graph_key]["core"] = core
        self._captures.setdefault(key, []).append(
            (graph_key, recorded.reads, recorded.writes)
        )
        state.publish(warm.writes)
        return True, result
