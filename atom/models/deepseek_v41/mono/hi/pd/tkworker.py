# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Ports, for one rank of ATOM's one-process-a-GPU host, TileRT 0.1.7 (MIT, Copyright (c) TileRT-team)
# ``e2e_graph.DsV41Graph``'s request calls (``set_sampling``, ``begin_request``, ``import_kv``, ``chunks``' enqueue /
# wait) and ``receive_server.ReceiveServer``'s buffer and transport set-up.
"""One GPU's worker of the decode host (``fanout.RankGroup`` backend): a ``tk.TkRank`` with its parameters, token tables,
code objects (TileRT's ``.co``, or a swap table's FlyDSL builds) and captured mode-5 graphs, every op on one stream of
its own. Rank 0 also owns the receive buffer (``receive.alloc_receive_buffer`` on its GPU) and the Mooncake transport
that registers it; the other ranks map that buffer over HIP IPC at start and convert a landed request themselves
(``pd.convert`` on their own device, the same planes TileRT copies from cuda:0 to every rank).

Startup (collective over a gloo group at ``dist_addr``): ``TkRank`` (its ``sym:`` arena exchange), ``load_params`` (the
checkpoint, ``weights.ckpt.compute_engram_consts``), the token tables (``toktab_dir/rank<r>.pt``: TileRT's
``g.toktab[r]``, ``[q, kv, hc]`` of ``layers.0`` then ``mtp.0``; without it ``build_toktab``, a stub, raises), kernels
with ``host.ADMIT_SYMBOL``, ``warmup`` + ``capture``, the receive buffer (rank 0: allocate, Mooncake init + register;
broadcast its IPC handle; ranks 1..7 open it).

Ops: ``set_sampling`` (``on`` only: greedy needs mode-2 graphs) -> ``TkRank.set_sampling``; ``admit`` ->
``TkRank.admit_request``; ``load_kv`` -> ``pd.convert.convert`` + ``TkRank.import_planes``, synchronized (the receive
buffer is free when every rank answered); ``step(k, ticket)`` -> k graph replays and an event (at most 4 in flight, as
TileRT's ring); ``wait_read`` -> the event polled up to ``device_timeout_s`` (TileRT ``TILERT_DEVICE_TIMEOUT_SECONDS``,
120), then rank 0's ``read_logs``.

Not run on a GPU yet (written against the TkRank API on the CPU): the IPC mapping of the receive buffer onto another
device, and every op's device behaviour, are for the GPU check (m7/D_LIFECYCLE.md).
"""

from __future__ import annotations

import os
import time


def build_toktab(tk, rank: int):
    """A rank's DSpark token tables (``toktab:layers.0.*`` / ``toktab:mtp.0.*``). Stub."""
    raise NotImplementedError(
        "the DSpark token tables are not built yet (TileRT: toktab.build_table, its embed / mHC / q-kv / kv-extract "
        "ops over the vocabulary); give the worker toktab_dir (TileRT's g.toktab[r] saved as rank<r>.pt)"
    )


def load_toktab(tk, rank: int, toktab_dir: str | None) -> None:
    """``toktab_dir/rank<r>.pt``: 6 tensors (TileRT's order: q, kv, hc of ``layers.0``, then of ``mtp.0``) or
    ``{scope: [q, kv, hc]}``."""
    import torch

    if toktab_dir is None:
        build_toktab(tk, rank)
    tabs = torch.load(os.path.join(toktab_dir, f"rank{rank}.pt"), map_location="cpu")
    if isinstance(tabs, (list, tuple)):
        tabs = {"layers.0": tabs[0:3], "mtp.0": tabs[3:6]}
    for scope in ("layers.0", "mtp.0"):
        tk.buffers.toktab(scope, *tabs[scope])


class TkRankWorker:
    """The module docstring's worker; ``fanout.RankGroup(8, "...pd.tkworker:TkRankWorker", kwargs)``."""

    def __init__(
        self,
        rank: int,
        nranks: int,
        ckpt: str,
        co: str,
        toktab_dir: str | None = None,
        swap: str | None = None,
        dist_addr: str = "127.0.0.1:29561",
        mooncake_host: str | None = None,
        cached_sym: bool = False,
        device_timeout_s: float = 120.0,
    ):
        import torch
        import torch.distributed as dist

        from ..tk import alloc, codeobj, host
        from ..weights import ckpt as CK

        if nranks != alloc.NPES:
            raise ValueError(f"the plan is TP{alloc.NPES}, got {nranks} ranks")
        self.rank, self.nranks = rank, nranks
        self.device = torch.device("cuda", rank)
        torch.cuda.set_device(self.device)
        dist.init_process_group(
            "gloo", init_method=f"tcp://{dist_addr}", rank=rank, world_size=nranks
        )
        self.max_seq_len = alloc.MAX_SEQ_LEN
        self.device_timeout_s = device_timeout_s
        self.tk = host.TkRank(rank, None, device=self.device, cached_sym=cached_sym)
        consts = CK.compute_engram_consts(ckpt)
        self.tk.load_params(ckpt, consts)
        self.token_map = torch.as_tensor(consts["token_map"]).to(torch.int64)
        load_toktab(self.tk, rank, toktab_dir)
        table = codeobj.CodeTable.from_dir(co)
        if swap:
            table = table.swap(swap)
        self.tk.load_kernels(table, extra=(host.ADMIT_SYMBOL,))
        self.stream = torch.cuda.Stream(self.device)
        with torch.cuda.stream(self.stream):
            self.tk.warmup()
            self.tk.capture()
        torch.cuda.synchronize(self.device)
        self._map_receive_buffer(mooncake_host)
        self.events: dict = {}
        dist.barrier()

    def _map_receive_buffer(self, mooncake_host: str | None) -> None:
        import torch
        import torch.distributed as dist

        from atom.mono.runtime import hip_ipc
        from atom.mono.runtime.uncached import DeviceBytes

        from . import receive as R
        from . import wire as W

        nbytes = W.wire_bytes(self.max_seq_len)
        info: list = [None]
        if self.rank == 0:
            self.buffer = R.alloc_receive_buffer(self.max_seq_len, str(self.device))
            self.transport = R.make_transport("mooncake")
            self.transport.init(mooncake_host or W.local_ip())
            self.transport.register(self.buffer.data_ptr(), nbytes)
            ptr = self.buffer.data_ptr()
            base = hip_ipc.get_allocation_base(ptr)
            info = [
                {
                    "handle": hip_ipc.get_ipc_handle(base),
                    "offset": ptr - base,
                    "base_ptr": ptr,
                    "session_id": self.transport.session_id,
                }
            ]
        dist.broadcast_object_list(info, src=0)
        self.receive_info = {k: v for k, v in info[0].items() if k != "handle"}
        if self.rank != 0:
            mapped = hip_ipc.open_ipc_handle(info[0]["handle"]) + info[0]["offset"]
            self.buffer = torch.as_tensor(DeviceBytes(mapped, nbytes))

    # ---- the ops (fanout.OPS)
    def hello(self) -> dict:
        from . import session as S

        out = {
            "rank": self.rank,
            "pid": os.getpid(),
            "max_seq_len": self.max_seq_len,
            "num_mtp": S.NUM_MTP,
            "stop_token_ids": [S.EOS_ID],
            "greedy": False,
            "transport": "mooncake",
        }
        if self.rank == 0:
            out.update(self.receive_info)
        return out

    def set_sampling(self, params: list[int], on: bool) -> None:
        import torch

        if not on:
            raise NotImplementedError(
                "greedy sampling mode: TileRT's mode-2 graphs are not in this host's plan"
            )
        with torch.cuda.device(self.device), torch.cuda.stream(self.stream):
            self.tk.set_sampling(params)
        self.stream.synchronize()

    def admit(self, first: int, pos: int, mtp: bool, params) -> None:
        import torch

        with torch.cuda.device(self.device), torch.cuda.stream(self.stream):
            self.tk.admit_request(first, pos, mtp, params)

    def load_kv(self, seq_len: int, generation: int) -> None:
        import torch

        from . import convert as C
        from . import wire as W

        rec = W.ReceivedRequest(
            rid=f"generation-{generation}",
            seq_len=seq_len,
            last_prompt_token=0,
            first_token_id=None,
            sampling=None,
            generation=generation,
        )
        with torch.cuda.device(self.device), torch.cuda.stream(self.stream):
            conv = C.convert(
                self.buffer, self.max_seq_len, rec, self.token_map, device=self.device
            )
            self.tk.import_planes(conv.planes, seq_len)
        self.stream.synchronize()

    def step(self, k: int, ticket: int) -> None:
        import torch

        if len(self.events) >= 4:
            raise RuntimeError("more than 4 chunks in flight")
        with torch.cuda.device(self.device), torch.cuda.stream(self.stream):
            for _ in range(k):
                self.tk.step()
            ev = torch.cuda.Event()
            ev.record(self.stream)
        self.events[ticket] = ev

    def wait_read(self, ticket: int, acc_from: int, k: int, tok_from: int):
        ev = self.events.pop(ticket)
        deadline = time.monotonic() + self.device_timeout_s
        while not ev.query():
            if time.monotonic() > deadline:
                raise TimeoutError(
                    f"rank {self.rank}: chunk {ticket} not done in {self.device_timeout_s}s"
                )
        if self.rank == 0:
            return self.tk.read_logs(acc_from, k, tok_from)
        return None

    def sync(self) -> None:
        self.stream.synchronize()
        self.events.clear()

    def state(self) -> dict:
        return {
            "rank": self.rank,
            "steps": self.tk.steps,
            "in_flight": sorted(self.events),
        }

    def close(self) -> None:
        import torch.distributed as dist

        self.sync()
        if dist.is_initialized():
            dist.destroy_process_group()
