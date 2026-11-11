# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# The exchange is adapted from ATOM atom/mono/runtime/peer_memory.py (MIT, Copyright (C) 2026 Advanced Micro
# Devices, Inc.).
"""A TP group's symmetric peer memory: one device buffer per rank, every rank holding every
rank's address (``plan.program``: tensor parallelism).

``PeerGroup(nbytes, group)`` is collective over the ``torch.distributed`` group: each rank allocates
``nbytes`` of uncached device memory (zeroed; ``hipExtMallocWithFlags`` uncached, every program's protocol was settled
on it), exports its HIP IPC handle, and maps every peer's (``atom.mono.runtime.hip_ipc``).
``MonoKernel.table(..., peers=group)`` adds the program's ``PEERS_ARG`` / ``RANK_ARG`` entries.

Launch order: a peer region has two copies by launch-tag parity, so consecutive launches on one
group must alternate tag parity (= the parity of ``layer + 1``: ``LAYER_SLOTS`` is even).
``MonoKernel.launch`` checks it through :meth:`PeerGroup.launched`. A graph replay launches no
Python: whoever captures a sequence keeps its layer ids alternating. A poll accepts any pair with
its exact tag, so no two launches on a group may carry the same tag: every ``Buffers`` launched on
the group shares the group's step counter (``PeerGroup.step``; ``runtime.kernel.Buffers``).

The step phase (shared programs, ``Program.peer_shared``): a step whose launches on the group are odd in number ends on
the parity its first launch had, so the next step's first launch would repeat it (two launches of one parity in a
row: the next step's early puts over this step's last slots). ``end_step`` after a step's last launch on the group adds
one to the group's step phase then (a device add, captured with the step's launches; the peer table's entry ``tp``),
which shared programs add to their tag's parity: consecutive launches alternate across steps too, whatever the steps'
widths.
"""

from __future__ import annotations

import warnings

import torch

from atom.mono.runtime import hip_ipc

_RETIRED: list = []  # closed groups kept mapped for reuse (``PeerGroup.close``)


def _pick_retired(retired: list, nbytes: int, group) -> int | None:
    """The index of the smallest retired group on TP group ``group`` (the same object) of at least ``nbytes``, the
    earliest of equals; None when none fits. Every rank retires in program order, so every rank picks alike
    (``PeerGroup._reuse`` checks)."""
    fits = [
        (g.nbytes, i)
        for i, g in enumerate(retired)
        if g.group is group and g.nbytes >= nbytes and g.local
    ]
    return min(fits)[1] if fits else None


class PeerGroup:
    """``nbytes`` of zeroed peer memory on every rank of ``group`` (module docstring)."""

    def __init__(self, nbytes: int, group=None):
        import torch.distributed as dist

        self.group = group
        self.rank = dist.get_rank(group)
        self.tp = dist.get_world_size(group)
        self.nbytes = nbytes
        self.last_parity: int | None = None
        self.step = None  # the group's step counter (``runtime.kernel.Buffers``: one counter a group)
        self.ids: set[int] = (
            set()
        )  # layer ids launched on the group since a bound Buffers' last bump
        self._opened: list[int] = []
        self.local = 0
        self._retired = False
        self.layouts: dict[tuple, str] = (
            {}
        )  # the peer layouts bound to the group (``bind``) -> a program's name
        self._warned = False
        self.since_step = 0  # launches since the last ``end_step``
        self._phase = (
            0  # the step phase as the host last set it (the launch order check's)
        )
        if self._reuse(
            nbytes, group
        ):  # a retired group's memory (``close``), zeroed again
            return
        self.local = hip_ipc.allocate_uncached(
            max(nbytes, 1)
        )  # (whole 4 KiB pages, zeroed)
        mine = hip_ipc.get_ipc_handle(self.local)
        handles = [None] * self.tp
        dist.all_gather_object(handles, mine, group=group)
        addresses, error = [], None
        for peer, h in enumerate(handles):
            if peer == self.rank:
                addresses.append(self.local)
                continue
            try:
                base = hip_ipc.open_ipc_handle(h)
                self._opened.append(base)
                addresses.append(base)
            except RuntimeError as e:  # every rank must still reach the agreement below
                error = f"rank {self.rank} mapping rank {peer}: {e}"
                break
        errors = [None] * self.tp
        dist.all_gather_object(errors, error, group=group)
        if any(errors):
            self.close(barrier=False, now=True)
            raise RuntimeError(
                "peer memory exchange failed: " + "; ".join(e for e in errors if e)
            )
        # every rank's buffer, then the step phase (``end_step``; shared programs read its low word)
        self.addresses = torch.tensor(addresses + [0], dtype=torch.int64, device="cuda")

    def bind(self, program) -> None:
        """Record ``program``'s peer layout (``MonoKernel.table``). Programs of different layouts on one group need
        ``Program.peer_shared`` (``plan.program``: a later launch's early put could land on a slot a slower peer
        still polls); a group holding unshared programs of two layouts warns once."""
        pairs = {r.name: r.pairs for r in program.regions}
        layout = (
            bool(program.peer_shared),
            tuple(sorted((n, o, pairs[n]) for n, o in program.peer.items())),
        )
        self.layouts.setdefault(layout, program.name)
        unshared = [name for (shared, _), name in self.layouts.items() if not shared]
        if len(self.layouts) > 1 and unshared and not self._warned:
            self._warned = True
            warnings.warn(f"dsv41hi PeerGroup: programs of {len(self.layouts)} peer layouts on one group, "
                          f"{len(unshared)} of them not shared ({', '.join(unshared[:4])}): a launch's early put may "
                          "land on a slot the previous launch still polls on a slower rank (Program.peer_shared, "
                          "plan.program.share_peer_groups)", stacklevel=3)  # fmt: skip

    def end_step(self) -> None:
        """A step's last launch on the group is issued (module docstring, "The step phase"): after an odd number of
        launches since the last call, the step phase flips (a device add on the current stream, graph-capturable), so
        the next launch has the other parity. Only shared programs read the phase."""
        if self.since_step % 2:
            self.addresses[self.tp :].add_(1)
            self._phase ^= 1
        self.since_step = 0

    def launched(self, layer: int, program_name: str) -> None:
        """Record a launch's tag parity; raise if it repeats the previous launch's (module docstring)."""
        parity = (layer + 1 + self._phase) & 1
        self.ids.add(layer)
        self.since_step += 1
        if self.last_parity == parity:
            raise ValueError(
                f"{program_name}: layer id {layer} has the previous peer launch's tag parity; "
                "consecutive launches on a peer group must alternate layer parity"
            )
        self.last_parity = parity

    def _reuse(self, nbytes: int, group) -> bool:
        """Take a retired group's memory (``close``) for this one when every rank picks the same one (collective):
        zeroed again, every rank past the zeroing before any launches."""
        import torch.distributed as dist

        pick = _pick_retired(_RETIRED, nbytes, group)
        every = [None] * self.tp
        dist.all_gather_object(
            every, None if pick is None else (pick, _RETIRED[pick].nbytes), group=group
        )
        if pick is None or any(e != every[0] for e in every):
            return False
        old = _RETIRED.pop(pick)
        self.nbytes, self.local, self._opened, self.addresses = (
            old.nbytes,
            old.local,
            old._opened,
            old.addresses,
        )
        self.addresses[self.tp :].zero_()  # (a fresh step phase)
        old.local, old._opened = 0, []
        hip_ipc.check_hip_error(
            hip_ipc.load_runtime().hipMemset(self.local, 0, self.nbytes),
            operation="hipMemset",
        )
        torch.cuda.synchronize()
        dist.barrier(group=group)
        return True

    def close(self, barrier: bool = True, now: bool = False) -> None:
        """Done with the group: kept mapped for a later ``PeerGroup`` of the same TP group to reuse (``_RETIRED``),
        freed at process exit. ``now``: unmap the peers and free the local buffer now,
        after every rank is done with it. Idempotent.

        Why not free: in a process that runs device collectives afterwards, closing a group (its peers' IPC mappings,
        its buffer) corrupted them (V4.1 TP8, a group closed before an NCCL all_gather in the same process: some
        ranks' slices arrived wrong at most receivers, deterministically and in every later gather; no close: clean).
        Reuse bounds the memory of evaluations that make a group per build."""
        if barrier:
            import torch.distributed as dist

            torch.cuda.synchronize()
            dist.barrier(group=self.group)
        if not now:
            if self.local and not self._retired:
                self._retired = True
                _RETIRED.append(self)
            return
        if self in _RETIRED:
            _RETIRED.remove(self)
        opened, self._opened = self._opened, []
        for base in opened:
            hip_ipc.close_ipc_handle(base)
        if self.local:
            hip_ipc.check_hip_error(
                hip_ipc.load_runtime().hipFree(self.local), operation="hipFree"
            )
            self.local = 0
