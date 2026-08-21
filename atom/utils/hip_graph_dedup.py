# SPDX-License-Identifier: MIT
"""HIP graph-exec DEDUP for ROCm (no cuda-python).

Many captured cudagraphs that are structurally identical and differ only in the
baked addresses -- e.g. a dense attention/FFN segment at the same layer+num_tokens
across different (bs, q_eff) ragged buckets -- can share ONE instantiated
hipGraphExec. Before launching a specific member we `hipGraphExecUpdate` the shared
exec to that member's raw graph (cheap; no re-instantiate). That is the memory win
that makes Option C's ragged path viable on ROCm.

Unlike sglang's NVIDIA path (cuda-python + a structural signature to GROUP
compatible graphs), we let the CALLER declare the group key: a dense segment at a
given (position, num_tokens) is identical across buckets by construction, so no
node-level signature introspection is needed. If a member is ever incompatible,
`hipGraphExecUpdate` says so and we assert -- loud, not silent.
"""

import ctypes

__all__ = ["HipGraphDedupRegistry", "hip_dedup_available"]

_HIP = None


def _hip():
    global _HIP
    if _HIP is None:
        lib = ctypes.CDLL("libamdhip64.so")
        cvp, cint, csz = ctypes.c_void_p, ctypes.c_int, ctypes.c_size_t
        lib.hipGraphInstantiate.restype = cint
        lib.hipGraphInstantiate.argtypes = [
            ctypes.POINTER(cvp),
            cvp,
            cvp,
            ctypes.c_char_p,
            csz,
        ]
        lib.hipGraphExecUpdate.restype = cint
        lib.hipGraphExecUpdate.argtypes = [
            cvp,
            cvp,
            ctypes.POINTER(cvp),
            ctypes.POINTER(cint),
        ]
        lib.hipGraphLaunch.restype = cint
        lib.hipGraphLaunch.argtypes = [cvp, cvp]
        lib.hipGraphExecDestroy.restype = cint
        lib.hipGraphExecDestroy.argtypes = [cvp]
        _HIP = lib
    return _HIP


def hip_dedup_available() -> bool:
    """Whether the HIP graph-exec dedup primitives load on this build."""
    try:
        _hip()
        return True
    except OSError:
        return False


def _instantiate(raw_graph: int) -> ctypes.c_void_p:
    hip = _hip()
    exec_ = ctypes.c_void_p()
    err = hip.hipGraphInstantiate(
        ctypes.byref(exec_), ctypes.c_void_p(raw_graph), None, None, 0
    )
    if err != 0:
        raise RuntimeError(f"hipGraphInstantiate failed (hipError={err})")
    return exec_


def _exec_update(exec_: ctypes.c_void_p, raw_graph: int) -> None:
    hip = _hip()
    err_node = ctypes.c_void_p()
    result = ctypes.c_int(-1)
    err = hip.hipGraphExecUpdate(
        exec_, ctypes.c_void_p(raw_graph), ctypes.byref(err_node), ctypes.byref(result)
    )
    # result 0 == hipGraphExecUpdateSuccess
    if err != 0 or result.value != 0:
        raise RuntimeError(
            "hipGraphExecUpdate failed: the captured graph is not compatible with "
            f"its dedup group (hipError={err}, updateResult={result.value}). Group "
            "members must be structurally identical (same kernels/topology, differ "
            "only in baked addresses)."
        )


def _launch(exec_: ctypes.c_void_p, stream: int) -> None:
    hip = _hip()
    err = hip.hipGraphLaunch(exec_, ctypes.c_void_p(stream))
    if err != 0:
        raise RuntimeError(f"hipGraphLaunch failed (hipError={err})")


class _DedupHandle:
    """One captured graph that belongs to a dedup group. `replay` retargets the
    group's shared exec to this graph (if needed) and launches it."""

    __slots__ = ("_registry", "group_key", "raw_graph", "_keepalive")

    def __init__(self, registry, group_key, raw_graph, keepalive):
        self._registry = registry
        self.group_key = group_key
        self.raw_graph = raw_graph
        self._keepalive = keepalive  # hold the torch CUDAGraph so raw stays valid

    def replay(self, stream: int | None = None) -> None:
        self._registry._replay(self, stream)


class HipGraphDedupRegistry:
    """One hipGraphExec per group_key, shared across all members of the group."""

    def __init__(self):
        self._groups: dict = {}  # group_key -> {"exec": c_void_p, "cur_raw": int}
        self._closed = False

    def register(self, raw_graph: int, group_key, keepalive=None) -> _DedupHandle:
        """Add a captured graph (raw hipGraph_t handle) to `group_key`. The first
        member instantiates the group's exec; later members are checked compatible
        via ExecUpdate at register time (fail loud)."""
        assert not self._closed
        grp = self._groups.get(group_key)
        if grp is None:
            self._groups[group_key] = {
                "exec": _instantiate(raw_graph),
                "cur_raw": raw_graph,
            }
        else:
            _exec_update(grp["exec"], raw_graph)  # verify compatible now
            grp["cur_raw"] = raw_graph
        return _DedupHandle(self, group_key, raw_graph, keepalive)

    def _replay(self, handle: _DedupHandle, stream: int | None) -> None:
        if stream is None:
            import torch

            stream = torch.cuda.current_stream().cuda_stream
        grp = self._groups[handle.group_key]
        if grp["cur_raw"] != handle.raw_graph:
            _exec_update(grp["exec"], handle.raw_graph)
            grp["cur_raw"] = handle.raw_graph
        _launch(grp["exec"], stream)

    def stats(self) -> tuple[int, int]:
        """(num groups == num execs, ...)."""
        return len(self._groups), len(self._groups)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        hip = _hip()
        for grp in self._groups.values():
            hip.hipGraphExecDestroy(grp["exec"])
        self._groups.clear()
