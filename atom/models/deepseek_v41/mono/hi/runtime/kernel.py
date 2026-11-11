# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""A built, checked, launchable monokernel."""

import copy
from dataclasses import dataclass, field

import torch

from ..device.gfx950.hazards import srcc_hazards
from ..emit.compile import ResourceError, compile_traced, device_cus, gate, resources
from ..emit.kernel import build
from ..plan.program import (
    DUAL_PAGE,
    LAYER_SLOTS,
    OWN_ARG,
    PEER_TOP_ARG,
    PEERS_ARG,
    RANK_ARG,
    Program,
    plan,
)


class TagError(ValueError):
    """A launch would reuse a tag: its pairs could be satisfied by an earlier launch's."""


def bind_group(bufs: "Buffers", peers, name: str, layer: int) -> None:
    """``bufs``' launches run on ``peers``: its step becomes the group's counter (``Buffers`` docstring). The group's
    counter is the first bound ``Buffers``' own tensor. A ``Buffers`` whose tensor must change is bound outside a graph
    capture, before any capture of its launches (the graph holds the old tensor): ``TagError`` otherwise. Its first
    launch, layer id ``layer``, starts a new step when the group launched that id since its last bump (an evaluation
    building a fresh ``Buffers`` an arm on one group: its step 0 would repeat the previous arm's tags).
    """
    if bufs.group is peers:
        return
    if bufs.group is not None:
        raise TagError(f"{name}: a Buffers bound to one peer group launched on another")
    if peers.step is None:
        peers.step = bufs.step
    elif bufs.step is not peers.step:
        if bufs.captured or (
            torch.cuda.is_available() and torch.cuda.is_current_stream_capturing()
        ):
            raise TagError(
                f"{name}: a Buffers with its own step counter joins a peer group inside or after a graph capture: "
                "launch it on the group once eagerly first (or build it on the group's step)"
            )
        if (
            bufs.step.numel() > peers.step.numel()
        ):  # (a dyn_rows build reads the live words after the counter)
            raise TagError(
                f"{name}: a Program.dyn_rows Buffers joins a peer group whose step counter has no live words (its "
                "first bound Buffers is a static program's): bind a dyn_rows Buffers to the group first"
            )
        bufs.step = (
            peers.step
        )  # (dyn_rows: the group's live words too, Buffers.set_live on any of its Buffers)
        if layer in peers.ids:
            bufs.step[:1].add_(1)  # (the counter only: not the live words after it)
            peers.ids.clear()
            bufs.launched.clear()
    bufs.group = peers


def _zeros(nbytes: int, device, uncached: bool = False) -> torch.Tensor:
    """A scratch allocation of ``nbytes`` zeroed bytes: ``torch.zeros``, or uncached device memory (``uncached``)."""
    if uncached:
        from . import uncached

        return uncached.zeros(nbytes, device)
    return torch.zeros(nbytes, dtype=torch.uint8, device=device)


@dataclass
class Buffers:
    """Per-launch-sequence state: scratch (zeroed once, never cleared again) and the
    step counter.

    A launch's tag is ``step * LAYER_SLOTS + layer + 1``; it must differ from the
    tags still in scratch (in practice: the previous launch's), so between two
    launches with the same ``layer`` the step must be bumped (:meth:`bump`,
    graph-capturable). Tags need not grow and may wrap. ``launched`` holds the
    layer ids launched (or captured) since the last host-side :meth:`bump`:
    ``MonoKernel.launch`` refuses one twice. A graph replay launches no Python,
    so whoever replays a graph bumps the step for it (inside or before it).

    **One counter a peer group:** a peer region's pairs are matched by their exact tag, so two launches on one
    ``PeerGroup`` must never carry the same tag, whichever ``Buffers`` they come from. The first launch of a
    ``Buffers`` on a group binds it (``group``): the group's step counter (``PeerGroup.step``, the first bound
    ``Buffers``' own) becomes its ``step``. Every bump adds 1, whichever of the group's ``Buffers`` it is called on:
    a driver that bumps several of them between two steps adds several, and the tags stay distinct. (No host-side
    "one add a step": a graph replay calls no Python, so such a flag would go stale and drop the replays' bumps.)
    """

    scratch: torch.Tensor
    step: torch.Tensor
    # the program's own buffer (``Program.private``: the dual-copy word, the atomics' regions); None: the program
    # has neither (``buffers`` sets it)
    private: torch.Tensor | None = None
    launched: set = field(default_factory=set)
    group: object = (
        None  # the PeerGroup this Buffers' launches run on (bound at its first launch on one)
    )
    captured: bool = (
        False  # launched inside a graph capture (its step tensor is in the graph)
    )

    def bump(self) -> None:
        """A new step: every layer id is fresh again (one device add, capturable)."""
        self.step[:1].add_(1)  # (not the live word after it, Program.dyn_rows)
        self.launched.clear()
        if self.group is not None:
            self.group.ids.clear()

    def set_step(self, step: int) -> None:
        """Jump the step counter (tests, harnesses); the caller keeps tags distinct."""
        self.step[0].fill_(step)
        self.launched.clear()
        if self.group is not None:
            self.group.ids.clear()

    def set_live(
        self, rows: int, spec: int | None = None, batch: int | None = None
    ) -> None:
        """``Program.dyn_rows``: the live rows of the next launches (the words after the step counter: rows, and for a
        batch x spec program its live spec and batch, ``rows = batch x spec``, the rows packed at the live spec's
        stride; device fills, capturable). Every rank of a TP group sets the same values. On a peer group the step
        tensor is the group's (``bind_group``): the live words are then every bound ``Buffers``' (one live shape a
        step), and a ``Buffers`` set before its first launch on the group takes the group's at that launch.
        """
        if self.step.numel() < 4:
            raise ValueError(
                "set_live: these buffers have no live words (a program without dyn_rows)"
            )
        if (spec is None) != (batch is None) or (
            spec is not None and spec * batch != rows
        ):
            raise ValueError(f"set_live: rows {rows} != batch {batch} x spec {spec}")
        self.step[1].fill_(rows)
        if spec is not None:
            self.step[2].fill_(spec)
            self.step[3].fill_(batch)


def _disassemble(elf: bytes, mcpu: str) -> str:
    import subprocess
    import tempfile

    from ..emit.isa import _objdump

    with tempfile.NamedTemporaryFile(suffix=".co") as f:
        f.write(elf)
        f.flush()
        cmd = [_objdump(), "-d", f"--mcpu={mcpu}", f.name]
        return subprocess.run(cmd, capture_output=True, text=True, check=True).stdout


_MEMO: list[dict] = []  # the active compile memos, innermost last


class MonoKernel:
    """``program`` planned, emitted, compiled, contract-checked and resource-gated, for its
    target (``Program.target``: LDS, grid, admission on a device of that arch).

    Construction raises ``ContractError`` (a hand-off could deadlock) or
    ``ResourceError`` (scratch, spills, LDS, residency) before anything runs. Inside an active compile memo
    (``_MEMO``) a program already built there returns a copy of that build.
    """

    def __new__(cls, program: Program | None = None):  # (None: copy.copy's call)
        if _MEMO and program is not None:
            hit = _MEMO[-1].get(plan(program).key())
            if hit is not None:
                k = copy.copy(hit)
                k.program, k._memo_hit = program, True
                return k
        return super().__new__(cls)

    def __init__(self, program: Program):
        if self.__dict__.pop(
            "_memo_hit", False
        ):  # a copy of a build in the active compile memo
            return
        self.program = plan(program)
        t = program.target
        self.launcher = build(program)
        self.record = compile_traced(self.launcher, program)
        self.resources = resources(self.launcher, t.arch)[0]
        self._gate()
        from ..emit.compile import _code_object

        # the built code's MFMA srcC hazards (``device.gfx950.hazards``)
        bad = srcc_hazards(
            _disassemble(
                _code_object(next(iter(self.launcher._mem_cache.values()))._ir_text),
                t.mcpu,
            )
        )
        if bad:
            raise ResourceError(
                f"{program.name}: {len(bad)} code hazard(s), e.g. {bad[0]}"
            )
        if _MEMO:
            _MEMO[-1][self.program.key()] = self

    def _gate(self) -> None:
        p, t = self.program, self.program.target
        # a build stays free of SGPR spills into VGPR lanes unless the composition says so (``Program.sgpr_spill_ok``)
        gate(
            self.resources,
            t.lds_bytes,
            sgpr_spill_ok=p.sgpr_spill_ok,
            grid=t.grid,
            cus=(
                device_cus()
                if self.resources["max_blocks_per_cu"] is not None
                else None
            ),
        )

    def buffers(self, device="cuda", scratch: torch.Tensor | None = None, uncached: bool = False,
                word: int | None = None) -> Buffers:  # fmt: skip
        """Scratch (zeroed), the private buffer and the step counter for launches. ``scratch``: a zeroed uint8 tensor
        of at least ``program.scratch_bytes`` to use instead of a new allocation (a slice of an arena other programs
        that never run at once share, ``emit.kernel``'s kernel ABI); the caller keeps it zero until the first launch.
        A dual-copy program's must start a 2 MiB page, and ``word`` is its copy word (``runtime.dual.scratch_word``
        of the arena; None: calibrated here). ``uncached``: new allocations are uncached device memory
        (``runtime.uncached``)."""
        p = self.program
        if scratch is not None:
            if (
                scratch.dtype != torch.uint8
                or scratch.numel() < p.scratch_bytes
                or not scratch.is_contiguous()
            ):
                raise ValueError(
                    f"{p.name}: scratch needs {p.scratch_bytes} contiguous uint8 bytes"
                )
            if p.dual_on and scratch.data_ptr() % DUAL_PAGE:
                raise ValueError(
                    f"{p.name}: a dual-copy program's scratch starts a {DUAL_PAGE} B page"
                )
        elif p.dual_on:  # the run area at the start of a 2 MiB page
            raw = _zeros(p.scratch_bytes + DUAL_PAGE, device, uncached)
            shift = -raw.data_ptr() % DUAL_PAGE
            scratch = raw[shift : shift + p.scratch_bytes]
        else:
            alloc = max(p.scratch_bytes, 256)
            scratch = _zeros(alloc, device, uncached)
        private = (
            _zeros(max(p.private_bytes, 256), device, uncached)
            if p.private or p.dual_on
            else None
        )
        if p.dual_on:  # the copy word of each XCD: word 0 of the private buffer
            if word is None:
                from .dual import scratch_word

                word = scratch_word(scratch, p.dual_bytes, p.name)
            word = (
                word - (1 << 32) if word >= 1 << 31 else word
            )  # (as int32 bits: page 3's XCD 7 is bit 31)
            private[:4].view(torch.int32)[0] = word
        # the step counter: one int32
        if (
            p.dyn_rows
        ):  # the step word, then live rows, spec, batch: all of them until set_live
            step = torch.zeros(4, dtype=torch.int32, device=device)
            step[1:] = torch.tensor(
                [p.scenario.rows, p.scenario.spec, p.scenario.batch], dtype=torch.int32
            )
            return Buffers(scratch=scratch, step=step, private=private)
        return Buffers(
            scratch=scratch,
            step=torch.zeros(1, dtype=torch.int32, device=device),
            private=private,
        )

    def table(self, pointers: dict, device="cuda", peers=None) -> torch.Tensor:
        """The pointer table: ``pointers[name]`` (a tensor or an int address) per
        ``program.args`` entry, int64, on ``device``. Keep it alive while launching.
        A program with peer regions takes its TP group's ``peers`` (``runtime.peer.PeerGroup``).
        """
        if self.program.peer:
            if peers is None or peers.tp != self.program.scenario.tp:
                raise ValueError(
                    f"{self.program.name} needs a PeerGroup of {self.program.scenario.tp} ranks"
                )
            if peers.nbytes < self.program.peer_bytes:
                raise ValueError(
                    f"{self.program.name} needs {self.program.peer_bytes} B of peer memory, "
                    f"the group has {peers.nbytes}"
                )
            pointers = {**pointers, PEERS_ARG: peers.addresses, RANK_ARG: peers.rank}
            if self.program.fast_entry:
                pointers[OWN_ARG] = peers.local
            if (
                self.program.peer_shared
            ):  # its parity-1 slots mirrored from the group's top (``Program.peer_shared``)
                pointers[PEER_TOP_ARG] = peers.nbytes
            peers.bind(
                self.program
            )  # (the group's layouts: a warning for unshared programs of different layouts)
        missing = set(self.program.args) - set(pointers)
        if missing:
            raise KeyError(f"pointer table needs {sorted(missing)}")
        addr = [
            p.data_ptr() if isinstance(p, torch.Tensor) else int(p)
            for p in (pointers[n] for n in self.program.args)
        ]
        table = torch.tensor(addr + [0], dtype=torch.int64, device=device)
        # the table holds raw addresses: keep the tensors alive as long as the table,
        # or a temporary's memory is reused (e.g. by buffers()) while the kernel writes it
        table._hi_refs = [p for p in pointers.values() if isinstance(p, torch.Tensor)]
        table._hi_peers = peers if self.program.peer else None
        return table

    def launch(
        self, table: torch.Tensor, bufs: Buffers, layer: int = 0, stream=None
    ) -> None:
        """One launch. Each launch needs a fresh tag: a new ``layer`` (0 ..
        ``LAYER_SLOTS - 1``) within a step, or ``bufs.bump()`` between steps;
        raises ``TagError`` on a layer id already launched since the last bump."""
        if not 0 <= layer < LAYER_SLOTS:
            raise TagError(
                f"layer id {layer} outside 0..{LAYER_SLOTS - 1} (it would alias another step's tag)"
            )
        peers = getattr(table, "_hi_peers", None)
        if peers is not None:
            bind_group(bufs, peers, self.program.name, layer)
        if layer in bufs.launched:
            raise TagError(
                f"{self.program.name}: layer id {layer} launched twice in one step (bump the step)"
            )
        bufs.launched.add(layer)
        if peers is not None:
            peers.launched(layer, self.program.name)
        if torch.cuda.is_current_stream_capturing():
            bufs.captured = True
        self.launcher(
            bufs.step.data_ptr(),
            layer,
            table.data_ptr(),
            bufs.scratch.data_ptr(),
            0 if bufs.private is None else bufs.private.data_ptr(),
            stream=stream or torch.cuda.current_stream(),
        )
