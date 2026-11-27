# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors
"""PP mailbox: hand a stage's output to the next stage without a send/recv rendezvous.

Why. NCCL point-to-point is a rendezvous. A stage's send kernel launches right
after its compute and spins until the next stage posts the matching recv,
holding 128 of the GPU's 256 CUs (Kimi-K3 PP8 P, LLM-Booster #813 / #815). The
next microbatch computes beside it at reduced speed, and a kernel that needs
every CU at once (persistent, cluster-barrier tiles such as q_b's ``_ps2``
GEMM) stalls until the send ends. The recv, in turn, is ordered after the
receiver's previous compute, so no transfer overlaps compute, and a CPU-side
gloo exchange of the tensor metadata makes each stage wait for its neighbour
once per microbatch. Capping RCCL's P2P channels is no way out: its bandwidth
scales with the blocks it holds (#813 run C: 8 blocks, transfers ~10x slower,
throughput -16%).

What. A hop between two stages on the same host gets a ring of ``slots``
receive buffers in the downstream GPU's memory, mapped into the upstream
process with a HIP IPC handle, and two 64-bit counters in fine-grained memory
(polled by the command processor, written by the peer):

* ``flag`` (downstream): microbatches written; the upstream bumps it after a slot.
* ``credit`` (upstream): microbatches read; the downstream bumps it after a slot.

Send (upstream, on its copy stream, after the compute that made the payload)::

    wait  credit >= seq - slots + 1      hipStreamWaitValue64: the slot is free
    copy  hidden, closed AttnRes blocks, DSpark aux rows -> slot seq % slots
    write rows[slot] = rows; flag = seq + 1   (stream order puts them after the copies)

Recv (downstream, on its recv stream, when the CPU reaches the microbatch,
normally while the GPU still runs the previous one)::

    wait  flag >= seq + 1                hipStreamWaitValue64
    copy-in kernel: slot -> fresh local tensors, ".cv" loads (a peer wrote the
          slot; its lines must not come from this GPU's caches)
    write credit = seq + 1 (upstream memory)
    the compute stream waits on an event after the copy-in

No kernel spins on a peer, the CPU never waits for a neighbour, and the
transfer overlaps the previous compute. ``slots`` bounds how far a stage runs
ahead of the next; two is enough (#813's pipeline model shows no gain past
depth 2 once the stages are balanced). The sender keeps at most
``_MAX_OUTSTANDING_SENDS`` sends' source tensors alive: past that, its compute
stream waits (an event, not a spin) for the oldest copy.

The payload also drops what the receiver never reads. Kimi-K3's IntermediateTensors
carry the whole AttnRes block bank ``[T, blocks, H]`` at every hop, but a stage
starting at layer ``L`` reads only the ``ceil(L / block_size)`` slots closed
before it and writes the rest itself (``AttnRes.window`` /
``AttnRes.maybe_close_block``). Only those are copied (one 2D copy).

Scope (v1): Kimi-K3 (``hidden_states`` + ``block_residual`` + DSpark
``pp_aux``), TP = 1, eager PP. A hop between hosts keeps NCCL; with
``ATOM_PP_TRIM_BANK=1`` it also sends only the closed bank blocks. Enable with
``ATOM_PP_MAILBOX=1``; validate on the hardware with
``scripts/pp_mailbox_selftest.py`` first.
"""

from __future__ import annotations

import ctypes
import logging
import socket
from collections import deque
from dataclasses import asdict, dataclass

import torch

logger = logging.getLogger("atom")

# HIP enums (hip_runtime_api.h)
_D2D = 3  # hipMemcpyDeviceToDevice
_D2H = 2  # hipMemcpyDeviceToHost
_WAIT_GTE = 0x0  # hipStreamWaitValueGte
_WRITE_DEFAULT = 0x0  # hipStreamWriteValueDefault
_MASK64 = (1 << 64) - 1
_COUNTER_PAGE = 4096
_SLOT_ALIGN = 2 << 20
# Counter page words (uint64): the counter (flag downstream, credit upstream), then rows per slot.
_COUNTER_WORD = 0
_ROWS_WORD = 1
_MAX_OUTSTANDING_SENDS = 2

_HIP = None


def _hip():
    """The HIP runtime of ``atom.mono.runtime.hip_ipc``, with the copy and
    stream-memory-op signatures this module calls on top of its own."""
    global _HIP
    if _HIP is not None:
        return _HIP
    from atom.mono.runtime.hip_ipc import load_runtime

    lib = load_runtime()
    vp, sz, u, i, u64 = (
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_uint,
        ctypes.c_int,
        ctypes.c_uint64,
    )
    for name, args in {
        "hipMalloc": [ctypes.POINTER(vp), sz],
        "hipMemcpy": [vp, vp, sz, i],
        "hipMemcpyAsync": [vp, vp, sz, i, vp],
        "hipMemcpy2DAsync": [vp, sz, vp, sz, sz, sz, i, vp],
        "hipStreamWaitValue64": [vp, vp, u64, u, u64],
        "hipStreamWriteValue64": [vp, vp, u64, u],
    }.items():
        fn = getattr(lib, name)
        fn.argtypes, fn.restype = args, i
    _HIP = lib
    return lib


def _check(err: int, what: str) -> None:
    if err != 0:
        from atom.mono.runtime.hip_ipc import check_hip_error

        check_hip_error(err, operation=f"PP mailbox {what}")


def _alloc_ring(nbytes: int) -> int:
    """Cached device memory for a receive ring (a raw ``hipMalloc``: its own
    allocation base, so ``hipIpcGetMemHandle`` exports it whole)."""
    p = ctypes.c_void_p()
    _check(_hip().hipMalloc(ctypes.byref(p), nbytes), "hipMalloc")
    return p.value


def _stream(stream: torch.cuda.Stream) -> ctypes.c_void_p:
    return ctypes.c_void_p(stream.cuda_stream)


def closed_bank_blocks(
    first_layer: int, block_size: int | None, total_blocks: int
) -> int:
    """AttnRes bank slots a stage starting at ``first_layer`` reads but did not
    write: those closed before it.

    Slot ``b`` is written at layer ``b * block_size`` (the attention site's
    ``maybe_close_block``) and read by every later site (``window``: slots below
    ``ceil(layer_idx / block_size)``).
    """
    if block_size is None or total_blocks == 0:
        return 0
    return min(total_blocks, -(-first_layer // block_size))


def fabric_options_from_env():
    """``ATOM_PP_MAILBOX_FABRIC*`` as ``pp_fabric.FabricOptions``, or None when
    fabric hops are off (or the mailbox is)."""
    from atom.distributed.pp_fabric import FabricOptions
    from atom.utils import envs

    if not (envs.ATOM_PP_MAILBOX and envs.ATOM_PP_MAILBOX_FABRIC):
        return None
    return FabricOptions(
        mode=envs.ATOM_PP_MAILBOX_FABRIC_MODE,
        wait=envs.ATOM_PP_MAILBOX_FABRIC_WAIT,
        ctas=envs.ATOM_PP_MAILBOX_FABRIC_CTAS,
    )


@dataclass(frozen=True)
class HopLayout:
    """One hop's slot layout. Each end derives it from the split and its relay
    taps; the setup checks the two agree."""

    max_rows: int  # rows a slot holds: a forward's tokens at most
    hidden: int  # H
    total_blocks: int  # slots of the full AttnRes bank
    closed_blocks: int  # bank slots that travel
    aux_cols: int  # DSpark aux columns: relayed taps x H
    elem_size: int  # bytes per element
    slots: int

    @property
    def row_elems(self) -> int:
        return self.hidden * (1 + self.closed_blocks) + self.aux_cols

    @property
    def slot_bytes(self) -> int:
        raw = self.max_rows * self.row_elems * self.elem_size
        return -(-raw // _SLOT_ALIGN) * _SLOT_ALIGN

    @property
    def ring_bytes(self) -> int:
        return self.slots * self.slot_bytes

    def offsets(self) -> dict[str, int]:
        """Byte offsets in a slot: hidden rows, closed-bank rows, aux rows; each
        region row-major at ``max_rows`` rows."""
        m, e = self.max_rows, self.elem_size
        bank = m * self.hidden * e
        aux = bank + m * self.closed_blocks * self.hidden * e
        return {"hidden": 0, "bank": bank, "aux": aux}


class _RawPtr:
    """A raw device address Triton takes as a pointer argument."""

    def __init__(self, addr: int, dtype: torch.dtype):
        self._addr, self.dtype = addr, dtype

    def data_ptr(self) -> int:
        return self._addr


_COPY_IN = None


def _copy_in_kernel():
    global _COPY_IN
    if _COPY_IN is None:
        import triton
        import triton.language as tl

        @triton.jit
        def _pp_mailbox_copy_in(
            src, dst, cols, src_stride, dst_stride, BLOCK: tl.constexpr
        ):
            row = tl.program_id(0)
            offs = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
            mask = offs < cols
            # ".cv": fetch from memory, not this GPU's caches (a peer wrote the slot)
            x = tl.load(src + row * src_stride + offs, mask=mask, cache_modifier=".cv")
            tl.store(dst + row * dst_stride + offs, x, mask=mask)

        _COPY_IN = _pp_mailbox_copy_in
    return _COPY_IN


def copy_in(
    src_addr: int,
    dst: torch.Tensor,
    rows: int,
    cols: int,
    src_stride: int,
    dst_stride: int,
) -> None:
    """``dst[r, :cols] = src[r, :cols]`` for ``r < rows``, in int64 words; ``dst``
    is an int64 view whose rows are ``dst_stride`` words apart."""
    if rows == 0 or cols == 0:
        return
    import triton

    block = 1024
    _copy_in_kernel()[(rows, triton.cdiv(cols, block))](
        _RawPtr(src_addr, torch.int64), dst, cols, src_stride, dst_stride, BLOCK=block
    )


class PPMailbox:
    """This stage's two ends: the ring it receives into (from the previous stage)
    and the peer ring it writes (the next stage's).

    ``recv_enabled`` / ``send_enabled`` say whether each hop uses the mailbox. A
    hop between hosts, or every hop with ``ATOM_PP_MAILBOX=0``, keeps NCCL, where
    only ``trim_for_nccl`` / ``expand_from_nccl`` apply.
    """

    def __init__(
        self,
        *,
        group,
        device: torch.device,
        dtype: torch.dtype,
        hidden: int,
        total_blocks: int,
        block_size: int | None,
        first_layer: int,
        end_layer: int,
        aux_cols_in: int,
        aux_cols_out: int,
        max_rows: int,
        slots: int = 2,
        use_mailbox: bool = True,
        fabric=None,
        trim_nccl_bank: bool = False,
        check: bool = False,
    ):
        if slots < 1:
            raise ValueError(f"PP mailbox needs at least one slot, got {slots}")
        self.device, self.dtype = device, dtype
        self.hidden, self.total_blocks = hidden, total_blocks
        self.trim_nccl_bank, self.check = trim_nccl_bank, check
        elem = torch.empty((), dtype=dtype).element_size()
        if (hidden * elem) % 8:
            raise ValueError(
                f"PP mailbox copies 8-byte words; a {hidden}-wide row of {elem}-byte elements is not"
            )
        first, last = group.is_first_rank, group.is_last_rank
        self.in_layout = None if first else HopLayout(
            max_rows, hidden, total_blocks,
            closed_bank_blocks(first_layer, block_size, total_blocks), aux_cols_in, elem, slots,
        )  # fmt: skip
        self.out_layout = None if last else HopLayout(
            max_rows, hidden, total_blocks,
            closed_bank_blocks(end_layer, block_size, total_blocks), aux_cols_out, elem, slots,
        )  # fmt: skip

        # Which hops the mailbox serves, before any allocation: one host and the
        # mailbox on at both ends -> "ipc" (HIP IPC ring); different hosts with
        # fabric hops on and usable at both ends -> "fabric" (pp_fabric.py: a
        # torch.symm_mem ring over the scale-up fabric); else "nccl", with no
        # ring (a ring is up to 2 x max_rows x 13 H-wide rows, ~5.7 GiB on
        # Kimi-K3's last stage, all of it KV cache otherwise).
        from atom.distributed import pp_fabric

        r = group.rank_in_group
        why_not_fabric = "off"
        if use_mailbox and fabric is not None:
            why_not_fabric = pp_fabric.probe_fabric(device)  # local; None: usable
        gathered: list = [None] * group.world_size
        torch.distributed.all_gather_object(
            gathered,
            (socket.gethostname(), use_mailbox, why_not_fabric is None, why_not_fabric),
            group=group.cpu_group,
        )
        stages = [pp_fabric.StageEnd(h, m, f) for h, m, f, _ in gathered]
        # A fabric hop builds a two-rank group, which every rank of the default
        # group must create: only when the PP group is the whole world.
        fabric_allowed = (
            fabric is not None
            and torch.distributed.get_world_size() == group.world_size
        )
        kinds = pp_fabric.plan_hop_transports(stages, fabric_allowed)
        self.recv_kind = None if first else kinds[r - 1]
        self.send_kind = None if last else kinds[r]
        self.fabric = fabric

        # Local memory of an "ipc" hop: the receive ring and its counter page (a
        # mailbox recv); the credit page of the ring this stage writes (a mailbox
        # send). Counter pages are fine-grained, polled by the command processor
        # and written by a peer; like every uncached / fine-grained peer page here,
        # never freed (atom/mono/runtime/uncached.py).
        from atom.mono.runtime.hip_ipc import (
            allocate_fine_grained,
            get_ipc_handle,
            open_ipc_handle,
        )

        self._ring = self._ring_page = self._credit_page = None
        if self.recv_kind == "ipc":
            self._ring = _alloc_ring(self.in_layout.ring_bytes)
            self._ring_page = allocate_fine_grained(_COUNTER_PAGE)
        if self.send_kind == "ipc":
            self._credit_page = allocate_fine_grained(_COUNTER_PAGE)
        torch.cuda.synchronize(device)

        mine = {
            "in_layout": None if self.in_layout is None else asdict(self.in_layout),
            "out_layout": None if self.out_layout is None else asdict(self.out_layout),
            "ring": None if self._ring is None else get_ipc_handle(self._ring),
            "ring_page": (
                None if self._ring_page is None else get_ipc_handle(self._ring_page)
            ),
            "credit_page": (
                None if self._credit_page is None else get_ipc_handle(self._credit_page)
            ),
        }
        everyone: list = [None] * group.world_size
        torch.distributed.all_gather_object(everyone, mine, group=group.cpu_group)
        prev = None if first else everyone[r - 1]
        nxt = None if last else everyone[r + 1]
        if nxt is not None and nxt["in_layout"] != mine["out_layout"]:
            raise RuntimeError(
                f"PP mailbox: stage {r} writes {mine['out_layout']}, stage {r + 1} reads {nxt['in_layout']}"
            )
        self._peer_ring = self._peer_ring_page = self._peer_credit_page = None
        if self.send_kind == "ipc":
            self._peer_ring = open_ipc_handle(nxt["ring"])
            self._peer_ring_page = open_ipc_handle(nxt["ring_page"])
        if self.recv_kind == "ipc":
            self._peer_credit_page = open_ipc_handle(prev["credit_page"])

        # "fabric" hops: every rank creates every such hop's two-rank group, in
        # hop order (new_group is collective over the world); the hop's two ends
        # then map each other's buffer. If either end cannot, both keep NCCL.
        self._fabric_in = self._fabric_out = None
        for h, kind in enumerate(kinds):
            if kind != "fabric":
                continue
            pair = torch.distributed.new_group(
                [group.ranks[h], group.ranks[h + 1]], backend="gloo"
            )
            if r in (h, h + 1):
                hop = pp_fabric.FabricHop(
                    role="send" if r == h else "recv",
                    layout=self.out_layout if r == h else self.in_layout,
                    group=pair, device=device, options=fabric,
                )  # fmt: skip
                if r == h:
                    self._fabric_out = hop if hop.ok else None
                    self.send_kind = "fabric" if hop.ok else "nccl"
                else:
                    self._fabric_in = hop if hop.ok else None
                    self.recv_kind = "fabric" if hop.ok else "nccl"

        self.send_enabled = self.send_kind in ("ipc", "fabric")
        self.recv_enabled = self.recv_kind in ("ipc", "fabric")
        self.send_seq = self.recv_seq = 0
        self._send_done: deque[torch.cuda.Event] = deque()
        self.send_stream = (
            torch.cuda.Stream(device=device) if self.send_enabled else None
        )
        self.recv_stream = (
            torch.cuda.Stream(device=device) if self.recv_enabled else None
        )

        def describe(lay, kind, peer) -> str:
            if peer is None:
                return "-"
            how = {
                "ipc": "mailbox",
                "fabric": f"mailbox over the fabric ({fabric.mode if fabric else ''}, wait {fabric.wait if fabric else ''})",
                "nccl": "nccl, closed blocks only" if trim_nccl_bank else "nccl",
            }[kind]
            return f"{how} ({lay.closed_blocks}/{total_blocks} bank blocks, {lay.aux_cols // hidden} aux taps)"

        rings = []
        if self._ring is not None:
            rings.append(f"ring {self.in_layout.ring_bytes / (1 << 30):.2f} GiB")
        for hop in (self._fabric_in, self._fabric_out):
            if hop is not None:
                rings.append(f"fabric buffer {hop.buffer.numel() / (1 << 30):.2f} GiB")
        logger.info(
            "PP mailbox: stage %d layers [%d, %d): recv %s, send %s; %d slots%s",
            r, first_layer, end_layer,
            describe(self.in_layout, self.recv_kind, prev),
            describe(self.out_layout, self.send_kind, nxt),
            slots,
            "".join(", " + x for x in rings),
        )  # fmt: skip
        if r == 0 and fabric is not None:
            for s, (_, _, ok, why) in enumerate(gathered):
                if not ok:
                    logger.warning(
                        "PP mailbox: fabric hops unavailable on stage %d: %s", s, why
                    )
            if not fabric_allowed:
                logger.warning(
                    "PP mailbox: fabric hops need the PP group to be the whole world "
                    "(%d of %d ranks); hops between hosts stay on NCCL",
                    group.world_size, torch.distributed.get_world_size(),
                )  # fmt: skip

    @classmethod
    def build(cls, config, model, relay, device) -> PPMailbox | None:
        """The mailbox of a Kimi-K3 PP stage under ``ATOM_PP_MAILBOX=1`` (or bank
        trimming alone under ``ATOM_PP_TRIM_BANK=1``); None otherwise or outside
        v1's scope. Collective over the PP group: every stage calls it at the same
        point, and every stage reaches the same verdict from the same config."""
        from atom.utils import envs

        if not (envs.ATOM_PP_MAILBOX or envs.ATOM_PP_TRIM_BANK):
            return None
        from aiter.dist.parallel_state import get_pp_group, get_tp_group

        pp = get_pp_group()
        if pp.world_size <= 1:
            return None
        inner = next(
            (
                m
                for m in model.modules()
                if hasattr(m, "block_bank")
                and hasattr(m, "start_layer")
                and hasattr(m, "end_layer")
            ),
            None,
        )
        why = None
        if inner is None:
            why = "the model has no AttnRes block bank (v1 handles Kimi-K3)"
        elif get_tp_group().world_size != 1:
            why = "TP > 1 (v1 handles TP = 1; the NCCL path splits the send across TP ranks)"
        if why is not None:
            logger.warning("PP mailbox off: %s", why)
            return None
        hf = inner.config
        block_size = getattr(hf, "attn_res_block_size", None)
        total_blocks = (
            0 if block_size is None else -(-hf.num_hidden_layers // block_size)
        )
        hidden = hf.hidden_size
        aux_in = 0 if relay is None else len(relay.incoming_ids) * hidden
        aux_out = (
            0 if relay is None or pp.is_last_rank else len(relay.outgoing_ids) * hidden
        )
        return cls(
            group=pp,
            device=device,
            dtype=getattr(config, "torch_dtype", None) or torch.bfloat16,
            hidden=hidden,
            total_blocks=total_blocks,
            block_size=block_size,
            first_layer=inner.start_layer,
            end_layer=inner.end_layer,
            aux_cols_in=aux_in,
            aux_cols_out=aux_out,
            max_rows=int(config.max_num_batched_tokens),
            slots=envs.ATOM_PP_MAILBOX_SLOTS,
            use_mailbox=envs.ATOM_PP_MAILBOX,
            fabric=fabric_options_from_env(),
            trim_nccl_bank=envs.ATOM_PP_TRIM_BANK,
            check=envs.ATOM_PP_MAILBOX_CHECK,
        )

    # -- mailbox hops -------------------------------------------------------------------------------------------------
    def send(self, tensors: dict[str, torch.Tensor], rows: int) -> None:
        """Queue this microbatch's payload for the next stage and return; the copy
        stream does any waiting."""
        lay, h = self.out_layout, self.hidden
        if rows > lay.max_rows:
            raise RuntimeError(
                f"PP mailbox: {rows} rows exceed a slot's {lay.max_rows}"
            )
        hidden = tensors["hidden_states"]
        bank = tensors.get("block_residual")
        aux = tensors.get("pp_aux")
        if not hidden.is_contiguous() or hidden.shape != (rows, h):
            raise RuntimeError(
                f"PP mailbox: hidden_states {tuple(hidden.shape)}, expected ({rows}, {h}) contiguous"
            )
        if lay.closed_blocks and (
            bank is None
            or not bank.is_contiguous()
            or bank.shape[0] != rows
            or bank.shape[1] < lay.closed_blocks
        ):
            raise RuntimeError(
                f"PP mailbox: block_residual {None if bank is None else tuple(bank.shape)} cannot carry {lay.closed_blocks} blocks"
            )
        if lay.aux_cols and (
            aux is None or not aux.is_contiguous() or aux.shape != (rows, lay.aux_cols)
        ):
            raise RuntimeError(
                f"PP mailbox: pp_aux {None if aux is None else tuple(aux.shape)}, expected ({rows}, {lay.aux_cols})"
            )
        compute = torch.cuda.current_stream(self.device)
        ready = torch.cuda.Event()
        ready.record(compute)
        s = self.send_stream
        seq, slot = self.send_seq, self.send_seq % lay.slots
        with torch.cuda.stream(s):
            s.wait_event(ready)
            if self.send_kind == "fabric":
                self._send_fabric(s, hidden, bank, aux, rows, seq, slot)
            else:
                self._send_ipc(s, hidden, bank, aux, rows, seq, slot)
            done = torch.cuda.Event()
            done.record(s)
        for t in (
            hidden,
            bank,
            aux,
        ):  # sources stay alive until the copies have read them
            if t is not None:
                t.record_stream(s)
        self._send_done.append(done)
        while len(self._send_done) > _MAX_OUTSTANDING_SENDS:
            compute.wait_event(self._send_done.popleft())
        self.send_seq += 1

    def _send_ipc(self, s, hidden, bank, aux, rows: int, seq: int, slot: int) -> None:
        """Same host: the copy engine writes the next stage's ring through IPC."""
        hip, sh, lay = _hip(), _stream(s), self.out_layout
        if seq >= lay.slots:  # the slot's previous payload must have been read
            credit = ctypes.c_void_p(self._credit_page + 8 * _COUNTER_WORD)
            _check(
                hip.hipStreamWaitValue64(
                    sh, credit, seq - lay.slots + 1, _WAIT_GTE, _MASK64
                ),
                "wait credit",
            )
        self._sdma_payload(
            sh, self._peer_ring + slot * lay.slot_bytes, hidden, bank, aux, rows
        )
        rows_word = ctypes.c_void_p(self._peer_ring_page + 8 * (_ROWS_WORD + slot))
        flag = ctypes.c_void_p(self._peer_ring_page + 8 * _COUNTER_WORD)
        _check(
            hip.hipStreamWriteValue64(sh, rows_word, rows, _WRITE_DEFAULT),
            "write rows",
        )
        _check(
            hip.hipStreamWriteValue64(sh, flag, seq + 1, _WRITE_DEFAULT), "write flag"
        )

    def _send_fabric(
        self, s, hidden, bank, aux, rows: int, seq: int, slot: int
    ) -> None:
        """Another host on the fabric (pp_fabric.py). Push: one kernel stores the
        payload into the next stage's ring and raises its flag. Pull: the copy
        engine fills this stage's own ring slot, then a kernel raises the flag."""
        from atom.distributed import pp_fabric as pf

        hop, lay = self._fabric_out, self.out_layout
        if seq >= lay.slots:
            hop.wait_ge(s, hop.word(pf.CREDIT_WORD), seq - lay.slots + 1)
        base = hop.slot_base(slot)
        rows_word = hop.word(pf.ROWS_WORD + slot, peer=True)
        flag = hop.word(pf.FLAG_WORD, peer=True)
        if self.fabric.mode == "pull":
            self._sdma_payload(_stream(s), base, hidden, bank, aux, rows)
            hop.kernels.signal(s, rows_word, rows, flag, seq + 1)
            return
        off, h = lay.offsets(), self.hidden
        v = 16 // lay.elem_size  # elements per 16 B word
        hop.kernels.push(
            s,
            hidden=hidden.data_ptr(), hidden_v4=h // v,
            bank=0 if bank is None else bank.data_ptr(),
            bank_row_v4=0 if bank is None else bank.shape[1] * h // v,
            closed_v4=lay.closed_blocks * h // v,
            aux=0 if aux is None else aux.data_ptr(), aux_v4=lay.aux_cols // v,
            rows=rows,
            dst_hidden=base + off["hidden"], dst_bank=base + off["bank"], dst_aux=base + off["aux"],
            rows_word=rows_word, flag_word=flag, flag_value=seq + 1,
            arrive=hop.word(pf.ARRIVE_WORD), ctas=self.fabric.ctas,
        )  # fmt: skip

    def _sdma_payload(self, sh, base: int, hidden, bank, aux, rows: int) -> None:
        """Hidden, the closed bank blocks (one 2D copy) and the aux rows into the
        slot at ``base``, on the copy engine."""
        hip, lay, h = _hip(), self.out_layout, self.hidden
        e, off = lay.elem_size, lay.offsets()
        dst = ctypes.c_void_p(base + off["hidden"])
        _check(
            hip.hipMemcpyAsync(
                dst, ctypes.c_void_p(hidden.data_ptr()), rows * h * e, _D2D, sh
            ),
            "copy hidden",
        )
        if lay.closed_blocks:
            width, pitch = lay.closed_blocks * h * e, bank.shape[1] * h * e
            dst = ctypes.c_void_p(base + off["bank"])
            src = ctypes.c_void_p(bank.data_ptr())
            _check(
                hip.hipMemcpy2DAsync(dst, width, src, pitch, width, rows, _D2D, sh),
                "copy bank",
            )
        if lay.aux_cols:
            dst = ctypes.c_void_p(base + off["aux"])
            src = ctypes.c_void_p(aux.data_ptr())
            _check(
                hip.hipMemcpyAsync(dst, src, rows * lay.aux_cols * e, _D2D, sh),
                "copy aux",
            )

    def recv(self, rows: int) -> dict[str, torch.Tensor]:
        """This microbatch's payload from the previous stage, as fresh tensors the
        current (compute) stream may use at once: it waits on the copy-in. The
        bank is full size, the closed blocks filled, the rest this stage's to write."""
        from atom.distributed import pp_fabric as pf

        lay, h = self.in_layout, self.hidden
        if rows > lay.max_rows:
            raise RuntimeError(
                f"PP mailbox: {rows} rows exceed a slot's {lay.max_rows}"
            )
        hip = _hip()
        compute = torch.cuda.current_stream(self.device)
        r = self.recv_stream
        rh = _stream(r)
        seq, slot = self.recv_seq, self.recv_seq % lay.slots
        hop = self._fabric_in if self.recv_kind == "fabric" else None
        with torch.cuda.stream(r):
            hidden = torch.empty(rows, h, dtype=self.dtype, device=self.device)
            bank = (
                torch.empty(
                    rows, self.total_blocks, h, dtype=self.dtype, device=self.device
                )
                if self.total_blocks
                else None
            )
            aux = (
                torch.empty(rows, lay.aux_cols, dtype=self.dtype, device=self.device)
                if lay.aux_cols
                else None
            )
            if hop is not None:
                hop.wait_ge(r, hop.word(pf.FLAG_WORD), seq + 1)
                rows_word, base = hop.word(pf.ROWS_WORD + slot), hop.slot_base(slot)
            else:
                flag = ctypes.c_void_p(self._ring_page + 8 * _COUNTER_WORD)
                _check(
                    hip.hipStreamWaitValue64(rh, flag, seq + 1, _WAIT_GTE, _MASK64),
                    "wait flag",
                )
                rows_word = self._ring_page + 8 * (_ROWS_WORD + slot)
                base = self._ring + slot * lay.slot_bytes
            if self.check:
                self._check_rows(r, rows_word, seq, rows)
            self._copy_in_payload(base, hidden, bank, aux, rows)
            if hop is not None:
                hop.kernels.signal(
                    r, None, 0, hop.word(pf.CREDIT_WORD, peer=True), seq + 1
                )
            else:
                credit = ctypes.c_void_p(self._peer_credit_page + 8 * _COUNTER_WORD)
                _check(
                    hip.hipStreamWriteValue64(rh, credit, seq + 1, _WRITE_DEFAULT),
                    "write credit",
                )
            done = torch.cuda.Event()
            done.record(r)
        compute.wait_event(done)
        out = {"hidden_states": hidden}
        if bank is not None:
            out["block_residual"] = bank
        if aux is not None:
            out["pp_aux"] = aux
        for t in out.values():
            t.record_stream(compute)
        self.recv_seq += 1
        return out

    def _copy_in_payload(self, base: int, hidden, bank, aux, rows: int) -> None:
        """The slot at ``base`` (local, or a peer's across the fabric) into this
        stage's tensors with ".cv" loads."""
        lay, h = self.in_layout, self.hidden
        off, w = lay.offsets(), 8 // lay.elem_size  # elements per int64 word
        copy_in(
            base + off["hidden"], hidden.view(torch.int64), rows, h // w, h // w, h // w
        )
        if lay.closed_blocks:
            cols = lay.closed_blocks * h // w
            dst = bank.view(rows, -1).view(torch.int64)
            copy_in(
                base + off["bank"], dst, rows, cols, cols, self.total_blocks * h // w
            )
        if lay.aux_cols:
            cols = lay.aux_cols // w
            copy_in(base + off["aux"], aux.view(torch.int64), rows, cols, cols, cols)

    def _check_rows(
        self, stream: torch.cuda.Stream, rows_word: int, seq: int, rows: int
    ) -> None:
        """``ATOM_PP_MAILBOX_CHECK``: the slot carries the rows this stage runs (a sync per microbatch)."""
        stream.synchronize()
        got = ctypes.c_uint64()
        src = ctypes.c_void_p(rows_word)
        _check(_hip().hipMemcpy(ctypes.byref(got), src, 8, _D2H), "read rows")
        if got.value != rows:
            raise RuntimeError(
                f"PP mailbox: microbatch {seq} carries {got.value} rows, this stage runs {rows}"
            )

    # -- NCCL hops ----------------------------------------------------------------------------------------------------
    def trim_for_nccl(self, tensors: dict[str, torch.Tensor]) -> tuple[str, ...]:
        """An NCCL hop under ``ATOM_PP_TRIM_BANK``: keep only the closed bank
        blocks. Returns the keys now holding fresh tensors (the send owns them)."""
        if not self.trim_nccl_bank or self.out_layout is None:
            return ()
        bank = tensors.get("block_residual")
        k = self.out_layout.closed_blocks
        if bank is None or k >= bank.shape[1]:
            return ()
        tensors["block_residual"] = bank[:, :k].contiguous()
        return ("block_residual",)

    def expand_from_nccl(self, tensors: dict[str, torch.Tensor]) -> None:
        """An NCCL hop under ``ATOM_PP_TRIM_BANK``: rebuild the full bank from the
        closed blocks the previous stage sent."""
        if not self.trim_nccl_bank or self.in_layout is None:
            return
        bank = tensors.get("block_residual")
        if bank is None or bank.shape[1] == self.total_blocks:
            return
        full = bank.new_empty(bank.shape[0], self.total_blocks, bank.shape[2])
        full[:, : bank.shape[1]].copy_(bank)
        tensors["block_residual"] = full
