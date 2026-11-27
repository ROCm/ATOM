# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors
"""PP mailbox hops between hosts, over the GPU scale-up fabric (UALink).

``pp_mailbox.py`` moves a hop between two stages on one host off NCCL: a ring
of receive slots in the downstream GPU, a flag the upstream bumps after a slot,
a credit the downstream returns after reading it. A hop between hosts stayed on
NCCL, and on Kimi-K3's PP8 P it is the last place a send spins on half the GPU
waiting for its receiver (LLM-Booster #813: pp3's cross-tray send stalls its
compute by 17.7 ms per microbatch, and one such hop is enough to bring back
the pipeline's slow stretches).

This module carries the same protocol across hosts that share a UALink pod:

* **Mapping**: torch.symm_mem with mori's ``MORI`` backend (``mori.allocator``),
  the transport aiter's cross-tray custom all-reduce and Kimi-K3's gfx1250 mono
  kernels already use. ``symm_mem.empty`` is HIP VMM with a fabric handle;
  ``symm_mem.rendezvous`` maps the peer's buffer into this process, so a kernel
  reaches it with plain loads and stores. Each end of a hop allocates the same
  buffer (symmetric): a counter page, then ``slots`` ring slots.
* **Moving the payload**: mori's SDMA path is per local device pair (XGMI), so
  bytes cross the fabric through kernels. ``push`` (default): the sender's
  copy kernel stores the payload into the receiver's ring, makes it visible at
  system scope and raises the receiver's flag. ``pull``: the sender copies the
  payload into its own ring with the copy engine (no CUs) and raises the flag;
  the receiver's copy-in reads the sender's ring across the fabric. Either way
  no kernel waits on the peer's progress: a slot is written only once its
  credit has come back.
* **Waiting**: ``kernel`` (default): a one-thread kernel polls a local word with
  system-scope acquire loads (gfx1250 runs ``hipStreamWaitValue64`` as a kernel
  anyway). ``stream``: ``hipStreamWaitValue64`` on the word, if the command
  processor observes stores that arrive over the fabric.

Remote memory is only ever stored to (push) or loaded from (pull): no remote
atomics. Counters only grow; a word is written by exactly one side.

Validate on the hardware with ``tools/pp_fabric_probe.py`` first (two ranks,
one per tray), then ``tools/pp_mailbox_selftest.py`` across two nodes.
"""

from __future__ import annotations

import contextlib
import ctypes
import itertools
import logging
import os
from dataclasses import dataclass

import torch

logger = logging.getLogger("atom")

# The symmetric buffer: a counter page at offset 0, the ring at RING_OFFSET.
PAGE_BYTES = 4096
RING_OFFSET = 2 << 20
# Counter page words (uint64). FLAG and ROWS live on the receiver's page and are
# written by the sender; CREDIT lives on the sender's page and is written by the
# receiver; ARRIVE (uint32) is the sender's own, used by the push kernel.
FLAG_WORD = 0
ROWS_WORD = 1  # + slot
MAX_SLOTS = 16
CREDIT_WORD = 32
ARRIVE_WORD = 64

MODES = ("push", "pull")
WAITS = ("kernel", "stream")
PUSH_THREADS = 256

# HIP enums (hip_runtime_api.h)
_WAIT_GTE = 0x0
_MASK64 = (1 << 64) - 1


@dataclass(frozen=True)
class FabricOptions:
    """How fabric hops run (``ATOM_PP_MAILBOX_FABRIC_*``)."""

    mode: str = "push"
    wait: str = "kernel"
    ctas: int = 32

    def __post_init__(self):
        if self.mode not in MODES:
            raise ValueError(f"fabric mode must be one of {MODES}, got {self.mode!r}")
        if self.wait not in WAITS:
            raise ValueError(f"fabric wait must be one of {WAITS}, got {self.wait!r}")
        if self.ctas < 1:
            raise ValueError(f"fabric copy needs at least one CTA, got {self.ctas}")


@dataclass(frozen=True)
class StageEnd:
    """What a stage brings to the hops on either side of it."""

    host: str
    mailbox: bool  # ATOM_PP_MAILBOX on this stage
    fabric: bool  # fabric hops requested here, and this rank passed probe_fabric


def fabric_hop_ends(hosts: list[str]) -> set[int]:
    """The stages at either end of a hop between hosts: the only ones a fabric
    hop can involve, so the only ones that set mori up (``probe_fabric``)."""
    ends = set()
    for s, (a, b) in enumerate(itertools.pairwise(hosts)):
        if a != b:
            ends.update((s, s + 1))
    return ends


def plan_hop_transports(stages: list[StageEnd], fabric_allowed: bool) -> list[str]:
    """Transport of each hop ``s -> s + 1``: ``"ipc"`` (both ends on one host,
    mailbox on), ``"fabric"`` (different hosts, fabric usable on both ends and
    allowed for the group), else ``"nccl"``. Pure: every rank derives the same
    plan from the same gathered ``stages``."""
    kinds = []
    for a, b in itertools.pairwise(stages):
        if a.mailbox and b.mailbox and a.host == b.host:
            kinds.append("ipc")
        elif fabric_allowed and a.fabric and b.fabric and a.host != b.host:
            kinds.append("fabric")
        else:
            kinds.append("nccl")
    return kinds


# ---------------------------------------------------------------------------------------------------------------------
# Kernels: HIP C++ through hipRTC, for explicit system-scope ordering.

KERNEL_SOURCE = r"""
typedef int v4i __attribute__((ext_vector_type(4)));
typedef unsigned long long u64;

// The payload into a ring slot across the fabric: hidden rows, the closed AttnRes
// bank blocks of each row (the bank's leading `closed_v4` 16 B words of a row
// `bank_row_v4` long) and the DSpark aux rows, each region packed row-major at its
// own width. Then every CTA fences at system scope and counts itself in; the last
// one writes the rows word and raises the flag (release, system scope).
extern "C" __global__ void __launch_bounds__(256)
pp_fabric_push(const v4i* __restrict__ hidden, long long hidden_v4,
               const v4i* __restrict__ bank, long long bank_row_v4, long long closed_v4,
               const v4i* __restrict__ aux, long long aux_v4, long long rows,
               v4i* __restrict__ dst_hidden, v4i* __restrict__ dst_bank,
               v4i* __restrict__ dst_aux, u64* rows_word, u64* flag_word,
               u64 flag_value, unsigned int* arrive) {
  for (long long r = blockIdx.x; r < rows; r += gridDim.x) {
    const v4i* h = hidden + r * hidden_v4;
    v4i* dh = dst_hidden + r * hidden_v4;
    for (long long c = threadIdx.x; c < hidden_v4; c += blockDim.x) dh[c] = h[c];
    if (closed_v4 > 0) {
      const v4i* b = bank + r * bank_row_v4;
      v4i* db = dst_bank + r * closed_v4;
      for (long long c = threadIdx.x; c < closed_v4; c += blockDim.x) db[c] = b[c];
    }
    if (aux_v4 > 0) {
      const v4i* a = aux + r * aux_v4;
      v4i* da = dst_aux + r * aux_v4;
      for (long long c = threadIdx.x; c < aux_v4; c += blockDim.x) da[c] = a[c];
    }
  }
  __threadfence_system();
  __syncthreads();
  if (threadIdx.x == 0) {
    unsigned int prev =
        __hip_atomic_fetch_add(arrive, 1u, __ATOMIC_ACQ_REL, __HIP_MEMORY_SCOPE_AGENT);
    if (prev == gridDim.x - 1) {
      __threadfence_system();
      __hip_atomic_store(rows_word, (u64)rows, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
      __hip_atomic_store(flag_word, flag_value, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
      __hip_atomic_store(arrive, 0u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
    }
  }
}

// Hold the stream until a local word written over the fabric reaches `target`.
extern "C" __global__ void pp_fabric_wait_ge(u64* word, u64 target) {
  if (threadIdx.x == 0) {
    while (__hip_atomic_load(word, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM) < target) {
      __builtin_amdgcn_s_sleep(8);
    }
  }
}

// Publish a counter across the fabric (and, when `rows_word` is set, the rows of a
// slot first), after everything earlier on the stream.
extern "C" __global__ void pp_fabric_signal(u64* rows_word, u64 rows, u64* word, u64 value) {
  if (threadIdx.x == 0) {
    __threadfence_system();
    if (rows_word) __hip_atomic_store(rows_word, rows, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
    __hip_atomic_store(word, value, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
  }
}
"""
KERNEL_NAMES = ("pp_fabric_push", "pp_fabric_wait_ge", "pp_fabric_signal")

_HIPRTC = None


def _hiprtc():
    """libhiprtc, next to the HIP runtime this process loaded when not on the
    loader path (``ATOM_HIPRTC_LIB`` overrides)."""
    global _HIPRTC
    if _HIPRTC is not None:
        return _HIPRTC
    names = [
        os.environ.get("ATOM_HIPRTC_LIB", ""),
        "libhiprtc.so",
        "libhiprtc.so.7",
        "libhiprtc.so.6",
    ]
    with contextlib.suppress(OSError), open("/proc/self/maps") as maps:
        for line in maps:
            if "libamdhip64" in line:
                names.append(
                    os.path.join(os.path.dirname(line.split()[-1]), "libhiprtc.so")
                )
                break
    lib, tried = None, []
    for name in filter(None, names):
        try:
            lib = ctypes.CDLL(name)
            break
        except OSError as e:
            tried.append(f"{name}: {e}")
    if lib is None:
        raise RuntimeError("PP fabric: cannot load hipRTC; tried " + "; ".join(tried))
    vp, sz, i, cp = ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_char_p
    for name, args in {
        "hiprtcCreateProgram": [
            ctypes.POINTER(vp),
            cp,
            cp,
            i,
            ctypes.POINTER(cp),
            ctypes.POINTER(cp),
        ],
        "hiprtcCompileProgram": [vp, i, ctypes.POINTER(cp)],
        "hiprtcGetProgramLogSize": [vp, ctypes.POINTER(sz)],
        "hiprtcGetProgramLog": [vp, cp],
        "hiprtcGetCodeSize": [vp, ctypes.POINTER(sz)],
        "hiprtcGetCode": [vp, cp],
        "hiprtcDestroyProgram": [ctypes.POINTER(vp)],
    }.items():
        fn = getattr(lib, name)
        fn.argtypes, fn.restype = args, i
    _HIPRTC = lib
    return lib


def compile_kernels(arch: str) -> bytes:
    """``KERNEL_SOURCE`` built for ``arch`` (a gcnArchName such as gfx1250): the
    code object's bytes. Needs no GPU."""
    lib = _hiprtc()
    prog = ctypes.c_void_p()
    err = lib.hiprtcCreateProgram(
        ctypes.byref(prog), KERNEL_SOURCE.encode(), b"pp_fabric.hip", 0, None, None
    )
    if err:
        raise RuntimeError(f"PP fabric: hiprtcCreateProgram failed ({err})")
    try:
        opts = [f"--gpu-architecture={arch}".encode(), b"-O3", b"-std=c++17"]
        err = lib.hiprtcCompileProgram(
            prog, len(opts), (ctypes.c_char_p * len(opts))(*opts)
        )
        if err:
            size = ctypes.c_size_t()
            lib.hiprtcGetProgramLogSize(prog, ctypes.byref(size))
            log = ctypes.create_string_buffer(max(1, size.value))
            lib.hiprtcGetProgramLog(prog, log)
            raise RuntimeError(
                f"PP fabric: hipRTC could not build the kernels for {arch}:\n{log.value.decode(errors='replace')}"
            )
        size = ctypes.c_size_t()
        lib.hiprtcGetCodeSize(prog, ctypes.byref(size))
        code = ctypes.create_string_buffer(size.value)
        lib.hiprtcGetCode(prog, code)
        return code.raw
    finally:
        lib.hiprtcDestroyProgram(ctypes.byref(prog))


def _hip_modules():
    """The HIP runtime of ``atom.mono.runtime.hip_ipc`` with the module and
    stream-memory calls this module makes."""
    from atom.mono.runtime.hip_ipc import load_runtime

    lib = load_runtime()
    vp, u, i, u64 = ctypes.c_void_p, ctypes.c_uint, ctypes.c_int, ctypes.c_uint64
    for name, args in {
        "hipModuleLoadData": [ctypes.POINTER(vp), vp],
        "hipModuleGetFunction": [ctypes.POINTER(vp), vp, ctypes.c_char_p],
        "hipModuleLaunchKernel": [
            vp,
            u,
            u,
            u,
            u,
            u,
            u,
            u,
            vp,
            ctypes.POINTER(vp),
            ctypes.POINTER(vp),
        ],
        "hipStreamWaitValue64": [vp, vp, u64, u, u64],
    }.items():
        fn = getattr(lib, name)
        fn.argtypes, fn.restype = args, i
    return lib


def _check(err: int, what: str) -> None:
    if err != 0:
        from atom.mono.runtime.hip_ipc import check_hip_error

        check_hip_error(err, operation=f"PP fabric {what}")


class FabricKernels:
    """The fabric kernels loaded on one device."""

    def __init__(self, arch: str):
        self.hip = _hip_modules()
        image = compile_kernels(arch)
        self._image = ctypes.create_string_buffer(
            image, len(image)
        )  # alive with the module
        module = ctypes.c_void_p()
        _check(
            self.hip.hipModuleLoadData(ctypes.byref(module), self._image),
            "hipModuleLoadData",
        )
        self._fn = {}
        for name in KERNEL_NAMES:
            fn = ctypes.c_void_p()
            _check(
                self.hip.hipModuleGetFunction(ctypes.byref(fn), module, name.encode()),
                f"get {name}",
            )
            self._fn[name] = fn

    def _launch(
        self, name: str, grid: int, block: int, stream: torch.cuda.Stream, args: list
    ) -> None:
        params = (ctypes.c_void_p * len(args))(
            *[ctypes.cast(ctypes.byref(a), ctypes.c_void_p) for a in args]
        )
        err = self.hip.hipModuleLaunchKernel(
            self._fn[name],
            grid,
            1,
            1,
            block,
            1,
            1,
            0,
            ctypes.c_void_p(stream.cuda_stream),
            params,
            None,
        )
        _check(err, f"launch {name}")

    def push(self, stream, *, hidden, hidden_v4, bank, bank_row_v4, closed_v4, aux, aux_v4, rows,
             dst_hidden, dst_bank, dst_aux, rows_word, flag_word, flag_value, arrive, ctas) -> None:  # fmt: skip
        p, q, w = ctypes.c_void_p, ctypes.c_longlong, ctypes.c_ulonglong
        args = [p(hidden), q(hidden_v4), p(bank), q(bank_row_v4), q(closed_v4), p(aux), q(aux_v4), q(rows),
                p(dst_hidden), p(dst_bank), p(dst_aux), p(rows_word), p(flag_word), w(flag_value), p(arrive)]  # fmt: skip
        grid = max(1, min(ctas, rows))
        self._launch("pp_fabric_push", grid, PUSH_THREADS, stream, args)

    def wait_ge(self, stream, word: int, target: int) -> None:
        self._launch(
            "pp_fabric_wait_ge",
            1,
            1,
            stream,
            [ctypes.c_void_p(word), ctypes.c_ulonglong(target)],
        )

    def signal(
        self, stream, rows_word: int | None, rows: int, word: int, value: int
    ) -> None:
        args = [
            ctypes.c_void_p(rows_word or 0),
            ctypes.c_ulonglong(rows),
            ctypes.c_void_p(word),
            ctypes.c_ulonglong(value),
        ]
        self._launch("pp_fabric_signal", 1, 1, stream, args)


_KERNELS: dict[int, FabricKernels] = {}


def fabric_kernels(device: torch.device) -> FabricKernels:
    index = device.index if device.index is not None else torch.cuda.current_device()
    if index not in _KERNELS:
        with torch.cuda.device(index):
            _KERNELS[index] = FabricKernels(
                torch.cuda.get_device_properties(index).gcnArchName
            )
    return _KERNELS[index]


# ---------------------------------------------------------------------------------------------------------------------
# Capability, mapping


def probe_fabric(device: torch.device) -> str | None:
    """None when this rank can serve a fabric hop, else why not. Local, no
    collective: mori and torch.symm_mem import, the device exports fabric
    handles, the MORI backend can be selected and the kernels build. No test
    allocation: the hop's own allocation is voted on by both ends in
    ``FabricHop``, which keeps the hop on NCCL if either end fails. Leaves the
    MORI backend selected on success (process-global, as aiter's custom
    all-reduce does); restores the previous one on failure."""
    try:
        import mori.allocator as mori_allocator
        import torch.distributed._symmetric_memory as symm_mem
    except Exception as e:  # noqa: BLE001
        return f"mori / torch.symm_mem unavailable ({e})"
    index = device.index if device.index is not None else torch.cuda.current_device()
    try:
        kind = mori_allocator.handle_type(index)
    except Exception as e:  # noqa: BLE001
        return f"mori cannot tell the device's handle type ({e})"
    if kind != "fabric":
        return f"the device exports {kind} handles, not fabric"
    prev = symm_mem.get_backend(device)
    if prev not in (None, "CUDA", "MORI"):
        return f"torch.symm_mem already uses the {prev} backend"
    try:
        symm_mem.set_backend("MORI")
        fabric_kernels(device)
    except Exception as e:  # noqa: BLE001
        if prev is not None:
            with contextlib.suppress(Exception):
                symm_mem.set_backend(prev)
        return f"MORI symm_mem or the fabric kernels unusable ({e})"
    return None


class FabricHop:
    """One end of a mailbox hop between hosts: the symmetric buffer (counter page
    + ring) on both ends, each mapped into the other through torch.symm_mem.

    Collective over ``group`` (the hop's two ranks, gloo): both ends must build
    it. ``ok`` is the two ends' agreement that each maps the other; when False
    both ends keep the hop on NCCL. An exception inside mori's rendezvous is not
    caught: mori runs its own collectives there with no timeout, and only this
    process exiting frees a peer blocked in them (as aiter's ``_rendezvous``)."""

    def __init__(
        self, *, role: str, layout, group, device: torch.device, options: FabricOptions
    ):
        import torch.distributed as dist
        import torch.distributed._symmetric_memory as symm_mem

        if role not in ("send", "recv"):
            raise ValueError(f"fabric hop role must be send or recv, got {role!r}")
        if layout.slots > MAX_SLOTS:
            raise ValueError(
                f"fabric hop supports at most {MAX_SLOTS} slots, got {layout.slots}"
            )
        check_widths(layout)
        self.role, self.layout, self.options = role, layout, options
        self.kernels = fabric_kernels(device)
        self.ok, self.buffer, self.local, self.peer = False, None, 0, 0
        # Allocate before any mori collective: an allocation can fail on one end
        # alone, and both ends must know before the rendezvous, which would
        # otherwise leave the other end blocked inside mori.
        why = None
        try:
            self.buffer = symm_mem.empty(
                RING_OFFSET + layout.ring_bytes, dtype=torch.uint8, device=device
            )
            self.buffer.zero_()
        except Exception as e:  # noqa: BLE001
            self.buffer, why = None, repr(e)
        if not _agree(self.buffer is not None, group):
            logger.warning(
                "PP fabric: hop buffer not allocated on both ends (%s); the hop stays on NCCL",
                why or "the peer failed",
            )
            self.buffer = None
            return
        # zero_() is stream-ordered and the rendezvous runs on the host: drain first,
        # or the peer could map a page whose counters are not yet zero
        torch.cuda.synchronize(device)
        self._handle = symm_mem.rendezvous(self.buffer, group)
        ptrs = list(self._handle.buffer_ptrs)
        me = dist.get_rank(group)
        self.ok = _agree(len(ptrs) == 2 and all(ptrs), group)
        if not self.ok:
            logger.warning(
                "PP fabric: the hop's two ends could not map each other; the hop stays on NCCL"
            )
            return
        self.local = self.buffer.data_ptr()
        self.peer = int(ptrs[1 - me])

    # -- addresses ------------------------------------------------------------------------------------------------------
    def word(self, index: int, *, peer: bool = False) -> int:
        return (self.peer if peer else self.local) + 8 * index

    def slot_base(self, slot: int) -> int:
        """The ring slot this end moves the payload through, as it addresses it:
        the receiver's ring under push, the sender's under pull."""
        return ring_slot_address(
            self.role,
            self.options.mode,
            self.local,
            self.peer,
            slot,
            self.layout.slot_bytes,
        )

    # -- waiting --------------------------------------------------------------------------------------------------------
    def wait_ge(self, stream: torch.cuda.Stream, word: int, target: int) -> None:
        if self.options.wait == "stream":
            err = self.kernels.hip.hipStreamWaitValue64(
                ctypes.c_void_p(stream.cuda_stream),
                ctypes.c_void_p(word),
                target,
                _WAIT_GTE,
                _MASK64,
            )
            _check(err, "stream wait")
        else:
            self.kernels.wait_ge(stream, word, target)


def _agree(ok: bool, group) -> bool:
    """Both ends of the hop say yes (a CPU all-reduce over its gloo group)."""
    import torch.distributed as dist

    vote = torch.tensor([int(bool(ok))], dtype=torch.int32)
    dist.all_reduce(vote, op=dist.ReduceOp.MIN, group=group)
    return bool(vote.item())


def ring_slot_address(
    role: str, mode: str, local: int, peer: int, slot: int, slot_bytes: int
) -> int:
    """Push: the sender writes the receiver's ring, the receiver reads its own.
    Pull: the sender writes its own ring, the receiver reads the sender's."""
    own = (role == "recv") == (mode == "push")
    return (local if own else peer) + RING_OFFSET + slot * slot_bytes


def check_widths(layout) -> None:
    """The push kernel moves 16 B words: every region's row must be a whole number of them."""
    e = layout.elem_size
    for name, cols in (("hidden", layout.hidden), ("aux", layout.aux_cols)):
        if (cols * e) % 16:
            raise ValueError(
                f"PP fabric: a {name} row of {cols} x {e} B is not a multiple of 16 B"
            )
