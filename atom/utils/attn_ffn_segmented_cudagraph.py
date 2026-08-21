# SPDX-License-Identifier: MIT
"""Attn/FFN SEGMENTED cudagraph capture for AF_PIECEWISE (ROCm).

Capture a whole forward as a sequence of torch.cuda.CUDAGraph SEGMENTS (one per
model piece: dense projections/FFN and the attention core), ALL sharing ONE
mempool. This is the correct capture for AF_PIECEWISE:

  * ONE capture session gives the allocator a coordinated view of the whole
    forward's liveness, so a segment's output that a later segment consumes is
    NOT reclaimed -- unlike independently-captured per-piece graphs, where a
    dense output the attention core reads across the graph boundary could be
    reused/overwritten (the reason the old path needed a per-step input copy).
  * While a segment graph is alive it holds a use_count on the shared pool
    (beginAllocateToPool), pinning pool-allocated tensors so later segments read
    them ZERO-COPY across replays.

NOTE: this is SEGMENTED, not "breakable": the attention core is a CAPTURED
segment (still replayed), there is no eager break to CPU. Segments that are
structurally identical across ragged buckets (a dense piece at the same shape)
are shared via `HipGraphDedupRegistry` (one hipGraphExec for the group).
"""

import logging
import os
import threading
from contextvars import ContextVar

import torch

from atom.utils import weak_ref_tensor

logger = logging.getLogger("atom")

# Delayed weak-ref release of segment outputs (sgl parity, the real memory lever).
# A segment output is returned to the model as a weak-ref VIEW (from_blob, no
# owning block); the owning tensor is kept in the session until the LAST segment
# that reads it has captured, then dropped so the shared pool overlays the block
# on later segments. Without this every one of a bucket's ~62 dense outputs stays
# pinned (measured 69GB dense). Off -> keep every output owning (old 35GB path).
_AF_SEG_WEAKREF = os.environ.get("ATOM_AF_SEG_WEAKREF", "1") == "1"

__all__ = [
    "SegmentedCudaGraph",
    "SegmentedCudaGraphCapture",
    "active_segmented_session",
    "get_current_stream",
    "log_segment_mem_stats",
    "segmented_mem_stats",
]

# Diagnostic. Per category (group_key[0], e.g. "dense"/"attn"/"af_output") across
# ALL buckets we track [reserved_growth_bytes, num_segments]:
#   reserved_growth = sum of positive memory_reserved() deltas over each segment's
#   begin..end. memory_RESERVED is the real physical pool footprint; it only grows
#   when the pool must claim NEW physical memory, so if weak-ref release lets the
#   pool overlay/reuse blocks this stays ~0. (memory_ALLOCATED deltas, which we
#   used before, are just accounting and mislead once blocks are freed/reused.)
# Each category -> [reserved_growth, allocated_held, num_segments]:
#   reserved_growth = Σ positive memory_reserved() delta over begin..end (new
#     physical the pool had to claim -> ~0 if it overlays/reuses earlier blocks).
#   allocated_held  = Σ memory_allocated() delta over begin..capture_end BEFORE
#     the release step (what each segment leaves LIVE at capture_end). This
#     distinguishes "output/workspace held live" (allocated_held ~ reserved) from
#     "transient freed but pool won't reuse across graphs" (allocated_held small,
#     reserved large).
_SEG_MEM: dict = {}


def segmented_mem_stats() -> dict:
    """{category: (reserved_growth, allocated_held, num_segments)} since reset."""
    return {k: (v[0], v[1], v[2]) for k, v in _SEG_MEM.items()}


def log_segment_mem_stats(reset: bool = True) -> None:
    if not _SEG_MEM:
        return
    total_rsv = sum(v[0] for v in _SEG_MEM.values())
    total_held = sum(v[1] for v in _SEG_MEM.values())
    parts = ", ".join(
        f"{k}: rsv={v[0] / (1 << 30):.2f}GB held={v[1] / (1 << 30):.2f}GB/{v[2]}seg"
        for k, v in sorted(_SEG_MEM.items())
    )
    logger.info(
        "[af-seg] reserved_growth=%.2fGB allocated_held=%.2fGB | proc "
        "reserved=%.2fGB allocated=%.2fGB | per-cat: %s",
        total_rsv / (1 << 30),
        total_held / (1 << 30),
        torch.cuda.memory_reserved() / (1 << 30),
        torch.cuda.memory_allocated() / (1 << 30),
        parts,
    )
    if reset:
        _SEG_MEM.clear()


# The capture stream, and the side streams forked during the current segment
# (auto-joined before capture_end). Set only while a capture session is open.
_current_stream_var: ContextVar["torch.cuda.Stream | None"] = ContextVar(
    "afseg_current_stream", default=None
)
_forked_streams_var: ContextVar["set | None"] = ContextVar(
    "afseg_forked_streams", default=None
)
# The active segmented session, read by the per-piece hooks (CUDAGraphWrapper /
# v4_core_attention) to route their body through run_segment.
_active_session_var: ContextVar["SegmentedCudaGraphCapture | None"] = ContextVar(
    "afseg_active_session", default=None
)


def active_segmented_session():
    """The SegmentedCudaGraphCapture currently capturing, or None. Piece hooks
    call `session.run_segment(...)` when this is set."""
    return _active_session_var.get()


def get_current_stream(device=None) -> torch.cuda.Stream:
    stream = _current_stream_var.get()
    return stream if stream is not None else torch.cuda.current_stream(device)


def _is_stream_capturing(stream: torch.cuda.Stream) -> bool:
    # ROCm/HIP: the portable torch API maps to the HIP runtime and is reliable.
    with torch.cuda.stream(stream):
        return torch.cuda.is_current_stream_capturing()


# ---- wait_stream hook: track side-stream forks so we can auto-join before a
# segment's capture_end (capture_end fails if a forked side stream is still
# participating in the capture). AF forces single-stream, so this is defensive.
_original_wait_stream = None
_hook_lock = threading.Lock()
_hook_refcount = 0


def _hooked_wait_stream(self: torch.cuda.Stream, other: torch.cuda.Stream):
    assert _original_wait_stream is not None
    forked = _forked_streams_var.get()
    capturing = _current_stream_var.get()
    if forked is None or capturing is None:
        _original_wait_stream(self, other)
        return
    cap_ptr = capturing.cuda_stream
    is_self_cap = self is capturing or self.cuda_stream == cap_ptr
    is_other_cap = other is capturing or other.cuda_stream == cap_ptr
    if is_self_cap and not is_other_cap:
        if not _is_stream_capturing(other):
            return
        _original_wait_stream(self, other)
        forked.discard(other)
    elif is_other_cap and not is_self_cap:
        _original_wait_stream(self, other)
        forked.add(self)
    else:
        _original_wait_stream(self, other)


def _install_wait_stream_hook():
    global _original_wait_stream, _hook_refcount
    with _hook_lock:
        if _hook_refcount == 0:
            _original_wait_stream = torch.cuda.Stream.wait_stream
            torch.cuda.Stream.wait_stream = _hooked_wait_stream  # type: ignore
        _hook_refcount += 1


def _uninstall_wait_stream_hook():
    global _original_wait_stream, _hook_refcount
    with _hook_lock:
        _hook_refcount -= 1
        if _hook_refcount == 0:
            assert _original_wait_stream is not None
            torch.cuda.Stream.wait_stream = _original_wait_stream  # type: ignore
            _original_wait_stream = None


class SegmentedCudaGraph:
    """One captured segment (torch.cuda.CUDAGraph or a dedup handle) per model
    piece; `replay()` runs them in order."""

    def __init__(self) -> None:
        self._segments: list = []

    def replay(self) -> None:
        for seg in self._segments:
            seg.replay()

    def _append_segment(self, seg) -> None:
        self._segments.append(seg)


class SegmentedCudaGraphCapture:
    """Context manager: capture the enclosed forward as segments (one per
    `run_segment` call), ALL sharing `pool`. Pieces route their body through
    `run_segment` off the active-session var while this is open."""

    def __init__(
        self,
        cuda_graph: SegmentedCudaGraph,
        pool=None,
        stream: "torch.cuda.Stream | None" = None,
        capture_error_mode: str = "thread_local",
        dedup=None,
    ):
        assert isinstance(cuda_graph, SegmentedCudaGraph)
        self.cuda_graph = cuda_graph
        self._pool = pool if pool is not None else (0, 0)
        # Capture the segments on torch's default_capture_stream, NOT the passed
        # graph_capture() stream. Cross-graph cudagraph-pool memory reuse only
        # kicks in on that stream (verified in tools/ut_attn_qsa_share.py: same
        # pool + shared workspace reuses on default_capture_stream, but NOT on a
        # fresh/explicit stream). This is what lets the per-layer attn graphs
        # share one qkn workspace instead of each reserving its own (~22GB).
        self._stream = stream
        self._capture_error_mode = capture_error_mode
        self._dedup = dedup
        self._stream_ctx = None
        self._stream_token = None
        self._forked_token = None
        self._session_token = None
        self._current_graph: "torch.cuda.CUDAGraph | None" = None
        self._seg_group_key = None
        # Delayed weak-ref release bookkeeping (see _AF_SEG_WEAKREF):
        #  _owned: {data_ptr: owning tensor} for outputs handed out as weak views;
        #  _consumed: data_ptrs that have been read by some segment (their producer
        #  output can be freed once no later segment still reads them).
        self._owned: dict = {}
        self._consumed: set = set()

    def __enter__(self):
        _install_wait_stream_hook()
        if self._stream is not None:
            self._stream_ctx = torch.cuda.stream(self._stream)
            self._stream_ctx.__enter__()
        self._stream_token = _current_stream_var.set(
            self._stream or torch.cuda.current_stream()
        )
        self._forked_token = _forked_streams_var.set(set())
        self._session_token = _active_session_var.set(self)
        return self

    def __exit__(self, *exc):
        _active_session_var.reset(self._session_token)
        _forked_streams_var.reset(self._forked_token)
        _current_stream_var.reset(self._stream_token)
        if self._stream_ctx is not None:
            self._stream_ctx.__exit__(*exc)
            self._stream_ctx = None
        _uninstall_wait_stream_hook()
        return False

    def run_segment(self, group_key, fn, *args, **kwargs):
        """Capture `fn(*args, **kwargs)` as ONE segment tagged `group_key`.

        The recorded kernels go into a fresh segment sharing the session pool; it
        is registered with the dedup registry under `group_key`, so the same piece
        at the same shape shares one exec across captures.

        Memory (see _AF_SEG_WEAKREF): the output is handed to the model as a
        weak-ref VIEW while the owning tensor is retained in `_owned`, and released
        only once the last segment that reads it has captured. That lets the shared
        pool overlay a bucket's per-layer intermediates instead of pinning all ~62.
        """
        self._seg_group_key = group_key
        input_ptrs = self._tensor_ptrs(args, kwargs) if _AF_SEG_WEAKREF else None
        _rsv_before = torch.cuda.memory_reserved()
        _alloc_before = torch.cuda.memory_allocated()
        self._begin_new_segment()
        out = fn(*args, **kwargs)
        self._end_current_segment()
        # allocated_held: what this segment leaves LIVE at capture_end, measured
        # BEFORE the release step below (so releases of PRIOR segments don't skew).
        _alloc_held = max(0, torch.cuda.memory_allocated() - _alloc_before)
        if _AF_SEG_WEAKREF:
            # This segment has baked the addresses it reads, so any owned output
            # whose last reader has now passed (consumed earlier, not read here)
            # can be freed -> the pool reclaims the block for later segments.
            for p in list(self._owned):
                if p in self._consumed and p not in input_ptrs:
                    del self._owned[p]
                    self._consumed.discard(p)
            self._consumed |= input_ptrs
            # Hand this segment's NEW outputs out as weak views (keep owning here).
            out = self._weakify_own(out, input_ptrs)
        _cat = group_key[0] if isinstance(group_key, tuple) and group_key else "?"
        _rec = _SEG_MEM.setdefault(_cat, [0, 0, 0])
        # Positive reserved delta only: the physical memory this segment forced the
        # pool to claim NEW. ~0 means the pool reused/overlaid earlier blocks.
        _rec[0] += max(0, torch.cuda.memory_reserved() - _rsv_before)
        _rec[1] += _alloc_held
        _rec[2] += 1
        return out

    @staticmethod
    def _tensor_ptrs(args, kwargs) -> set:
        """data_ptr of every tensor reachable in args/kwargs (one level into
        tuples/lists)."""
        ptrs: set = set()

        def _add(x):
            if isinstance(x, torch.Tensor) and x.numel() > 0:
                ptrs.add(x.data_ptr())
            elif isinstance(x, (list, tuple)):
                for e in x:
                    if isinstance(e, torch.Tensor) and e.numel() > 0:
                        ptrs.add(e.data_ptr())

        for a in args:
            _add(a)
        for v in kwargs.values():
            _add(v)
        return ptrs

    def _weakify_own(self, out, input_ptrs: set):
        """Return `out` with each NEW tensor replaced by a weak-ref view, keeping
        the owning tensor in `_owned`. A tensor that is also a segment INPUT is a
        pass-through (the residual stream, `positions`, ...) and is left as-is so
        its producer/model keeps owning it -- that is the residual exemption."""
        if isinstance(out, torch.Tensor):
            if out.numel() == 0 or out.data_ptr() in input_ptrs:
                return out
            self._owned[out.data_ptr()] = out
            return weak_ref_tensor(out)
        if isinstance(out, tuple):
            return tuple(self._weakify_own(o, input_ptrs) for o in out)
        if isinstance(out, list):
            return [self._weakify_own(o, input_ptrs) for o in out]
        return out

    def _begin_new_segment(self) -> None:
        # keep_graph=True keeps the raw hipGraph_t so dedup can `raw_cuda_graph()`
        # it and share one exec; the plain path needs no raw handle.
        graph = (
            torch.cuda.CUDAGraph(keep_graph=True)
            if self._dedup is not None
            else torch.cuda.CUDAGraph()
        )
        graph.capture_begin(
            pool=self._pool, capture_error_mode=self._capture_error_mode
        )
        self._current_graph = graph

    def _end_current_segment(self) -> None:
        # Auto-join any side streams forked but not rejoined during this segment,
        # else capture_end fails.
        forked = _forked_streams_var.get()
        if forked:
            assert _original_wait_stream is not None
            main_stream = get_current_stream()
            for side in list(forked):
                if _is_stream_capturing(side):
                    _original_wait_stream(main_stream, side)
            forked.clear()
        graph = self._current_graph
        assert graph is not None
        graph.capture_end()
        if self._dedup is not None:
            # Register under the piece's group key; a matching segment from another
            # capture shares its exec (fail loud if incompatible).
            handle = self._dedup.register(
                int(graph.raw_cuda_graph()), self._seg_group_key, keepalive=graph
            )
            self.cuda_graph._append_segment(handle)
        else:
            self.cuda_graph._append_segment(graph)
        self._current_graph = None
