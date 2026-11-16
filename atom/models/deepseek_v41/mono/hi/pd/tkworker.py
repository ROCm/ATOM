# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Ports, for one rank of ATOM's one-process-a-GPU host, TileRT 0.1.7 (MIT, Copyright (c) TileRT-team)
# ``e2e_graph.DsV41Graph``'s request calls (``set_sampling``, ``begin_request``, ``import_kv``, ``chunks``' enqueue /
# wait) and ``receive_server.ReceiveServer``'s buffer and transport set-up.
"""One GPU's worker of the decode host (``fanout.RankGroup`` backend): a ``tk.TkRank`` with its parameters, token tables,
code objects (our FlyDSL builds from source; TileRT's ``.co`` or a swap table only as dev options) and captured mode-5
graphs, every op on one stream of its own. Rank 0 also owns the receive buffer (``receive.alloc_receive_buffer`` on its GPU) and the Mooncake transport
that registers it; the other ranks map that buffer over HIP IPC at start and convert a landed request themselves
(``pd.convert`` on their own device, the same planes TileRT copies from cuda:0 to every rank).

Startup (collective over a gloo group at ``dist_addr``): ``TkRank`` (its ``sym:`` arena exchange), ``load_params`` (the
checkpoint, ``weights.ckpt.compute_engram_consts``), the token tables (``load_toktab``: built from the checkpoint by
``tk.toktab``, or read from ``toktab_dir/rank<r>.pt`` when every rank has one there, a cache the build fills; ``[q,
kv, hc]`` of ``layers.0`` then ``mtp.0``), kernels (``codeobj.CodeTable.from_source``, ``admit_kernel`` included; dev:
``co`` = TileRT's ``.co``, ``swap`` a swap table over it), ``warmup`` + ``capture``, the receive buffer (rank 0:
allocate, Mooncake init + register; broadcast its IPC handle; ranks 1..7 open it).

Ops: ``set_sampling`` (``on`` only: greedy needs mode-2 graphs) -> ``TkRank.set_sampling``; ``admit`` ->
``TkRank.admit_request``; ``load_kv`` -> ``pd.convert.convert`` + ``TkRank.import_planes``, synchronized (the receive
buffer is free when every rank answered); ``step(k, ticket)`` -> k graph replays and an event (at most 4 in flight, as
TileRT's ring); ``wait_read`` -> the event polled up to ``device_timeout_s`` (TileRT ``TILERT_DEVICE_TIMEOUT_SECONDS``,
120), then rank 0's ``read_logs``.

Test options: ``receive=False`` (no receive buffer, no Mooncake), ``planes=<file>`` (``load_kv`` imports the planes of a
``torch.save``d ``{"n", "first", "planes"}``, as TileRT's ``restore()`` re-imports its prefill's snapshot, instead of
converting the receive buffer: ``load_planes_file``), ``progress_dir`` (``rank<r>.progress``: the startup phase
running, for naming a hang).

GPU status: admission and import reproduced TileRT's state on 049 (m7/RUN2.md, run_admit.py); this worker class itself
(the fan-out, the IPC mapping of the receive buffer, Mooncake in a worker) is not yet run on a GPU.
"""

from __future__ import annotations

import os
import time
from datetime import timedelta


def load_planes_file(path: str, seq_len: int | None = None) -> dict:
    """``torch.save``d ``{"n", "first", "planes"}`` (m7/x_tkdump2.py's ``snapshot.pt``: TileRT's ``g.kv_planes(n, 0)``),
    on the CPU; ``seq_len`` must equal its ``n`` when given."""
    import torch

    snap = torch.load(path, map_location="cpu")
    if not isinstance(snap, dict) or not {"n", "first", "planes"} <= set(snap):
        raise ValueError(f"{path}: not a planes file (want n, first, planes)")
    if seq_len is not None and int(snap["n"]) != int(seq_len):
        raise ValueError(
            f"{path}: planes of {snap['n']} positions, asked for {seq_len}"
        )
    planes = snap["planes"]
    for name in ("main_kv", "index_k", "win", "state", "hist"):
        if name not in planes:
            raise ValueError(f"{path}: planes without {name}")
    return snap


def _pack(o):
    """Tensors as (dtype name, shape, uint8 bytes): ``broadcast_object_list``'s pickling cannot carry float8."""
    import torch

    if isinstance(o, torch.Tensor):
        t = o.detach().contiguous().cpu()
        return (
            "tensor",
            str(t.dtype).rsplit(".", 1)[-1],
            tuple(t.shape),
            t.reshape(-1).view(torch.uint8),
        )
    if isinstance(o, dict):
        return {k: _pack(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return type(o)(_pack(v) for v in o)
    return o


def _unpack(o):
    import torch

    if (
        isinstance(o, tuple)
        and len(o) == 4
        and o[0] == "tensor"
        and isinstance(o[3], torch.Tensor)
    ):
        return o[3].view(getattr(torch, o[1])).reshape(o[2])
    if isinstance(o, dict):
        return {k: _unpack(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return type(o)(_unpack(v) for v in o)
    return o


def share_planes(planes, src: int = 0, group=None):
    """Rank ``src``'s converted planes on every rank of ``group`` (CPU tensors through gloo's
    ``broadcast_object_list``, as bytes; about 0.36 GB at a 400K request)."""
    import torch.distributed as dist

    box = [_pack(planes) if dist.get_rank(group) == src else None]
    dist.broadcast_object_list(box, src=src, group=group)
    return _unpack(box[0])


def _to_device(o, device):
    import torch

    if isinstance(o, torch.Tensor):
        return o.to(device)
    if isinstance(o, dict):
        return {k: _to_device(v, device) for k, v in o.items()}
    return type(o)(_to_device(v, device) for v in o)


def load_toktab(
    tk,
    rank: int,
    toktab_dir: str | None = None,
    ckpt: str | None = None,
    group=None,
    log=None,
    hw: str = "cpu",
) -> None:
    """This rank's token tables into ``tk``: ``toktab_dir/rank<r>.pt`` (6 tensors in TileRT's order: q, kv, hc of
    ``layers.0``, then of ``mtp.0``; or ``{scope: [q, kv, hc]}``) when it is there on every rank of ``group``, else
    built from ``ckpt`` (``tk.toktab.build_rank_tables``, collective over ``group``) and, with ``toktab_dir``, saved
    there. ``hw``: the build's hardware approximations, ``cpu`` (modelled) or ``device`` (this rank's GPU).
    """
    import torch
    import torch.distributed as dist

    path = os.path.join(toktab_dir, f"rank{rank}.pt") if toktab_dir else None
    have = path is not None and os.path.isfile(path)
    distributed = dist.is_available() and dist.is_initialized()
    if distributed:  # every rank reads, or every rank builds (the build is collective)
        seen = [None] * dist.get_world_size(group)
        dist.all_gather_object(seen, have, group=group)
        have = all(seen)
    if have:
        tabs = torch.load(path, map_location="cpu")
    else:
        if ckpt is None:
            raise ValueError(
                f"no token tables at {path} and no checkpoint to build them from"
            )
        from ..tk import toktab

        grp = (
            (group if group is not None else dist.group.WORLD) if distributed else None
        )
        provider = toktab.DeviceHw(tk.device) if hw == "device" else toktab.CpuHw()
        tabs = toktab.build_rank_tables(ckpt, rank, group=grp, log=log, hw=provider)
        if path is not None:
            os.makedirs(toktab_dir, exist_ok=True)
            tmp = f"{path}.tmp{os.getpid()}"
            torch.save(tabs, tmp)
            os.replace(tmp, path)
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
        co: str | None = None,
        toktab_dir: str | None = None,
        swap: str | None = None,
        fly_cache: str | None = None,
        toktab_hw: str = "cpu",
        dist_addr: str = "127.0.0.1:29561",
        mooncake_host: str | None = None,
        cached_sym: bool = False,
        device_timeout_s: float = 120.0,
        receive: bool = True,
        planes: str | None = None,
        progress_dir: str | None = None,
        accept: str = "real",
        kv_share: str = "gloo",
    ):
        import torch
        import torch.distributed as dist

        from ..tk import alloc, codeobj, host
        from ..weights import ckpt as CK

        if nranks != alloc.NPES:
            raise ValueError(f"the plan is TP{alloc.NPES}, got {nranks} ranks")
        self.rank, self.nranks = rank, nranks
        self.progress_dir = progress_dir
        self._progress("init")
        self.device = torch.device("cuda", rank)
        torch.cuda.set_device(self.device)
        # (a long timeout: without a token table cache, rank 0 builds the shared rows while the others wait)
        dist.init_process_group(
            "gloo",
            init_method=f"tcp://{dist_addr}",
            rank=rank,
            world_size=nranks,
            timeout=timedelta(hours=3),
        )
        self.max_seq_len = alloc.MAX_SEQ_LEN
        self.device_timeout_s = device_timeout_s
        if kv_share not in ("gloo", "ipc"):
            raise ValueError(f"kv_share: gloo | ipc, not {kv_share!r}")
        self.accept, self.kv_share = accept, kv_share
        self.tk = host.TkRank(
            rank, None, device=self.device, cached_sym=cached_sym, accept=accept
        )
        self._progress("params")
        consts = CK.compute_engram_consts(ckpt)
        self.tk.load_params(ckpt, consts)
        self.token_map = torch.as_tensor(consts["token_map"]).to(torch.int64)
        self._progress("toktab")
        load_toktab(self.tk, rank, toktab_dir, ckpt=ckpt, log=self._log, hw=toktab_hw)
        self._progress("kernels")
        # ours from source by default; TileRT's .co / a swap table only as dev options
        table = (
            codeobj.CodeTable.from_dir(co)
            if co
            else codeobj.CodeTable.from_source(fly_cache, log=self._log)
        )
        if swap:
            table = table.swap(swap)
        self.tk.load_kernels(table, extra=(host.ADMIT_SYMBOL,))
        self.stream = torch.cuda.Stream(self.device)
        dist.barrier()
        self._progress("warmup")
        with torch.cuda.stream(self.stream):
            self.tk.warmup()
        torch.cuda.synchronize(self.device)
        self._progress("capture")
        with torch.cuda.stream(self.stream):
            self.tk.capture()
        torch.cuda.synchronize(self.device)
        self.planes = None
        if planes:
            self._progress("planes")
            snap = load_planes_file(planes)
            self.planes_n = int(snap["n"])
            self.planes = _to_device(snap["planes"], self.device)
        self.buffer = None
        if receive:
            self._progress("receive")
            self._map_receive_buffer(mooncake_host)
        self.events: dict = {}
        dist.barrier()
        self._progress("ready")

    def _log(self, msg: str) -> None:
        if self.rank == 0:
            print(f"[tkworker r{self.rank}] {msg}", flush=True)

    def _progress(self, phase: str) -> None:
        if self.progress_dir:
            p = os.path.join(self.progress_dir, f"rank{self.rank}.progress")
            with open(p, "w") as f:
                f.write(f'{{"phase": "{phase}", "t": {time.time()}}}\n')

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
            info = [{"base_ptr": ptr, "session_id": self.transport.session_id}]
            if self.kv_share == "ipc":
                base = hip_ipc.get_allocation_base(ptr)
                info[0].update(handle=hip_ipc.get_ipc_handle(base), offset=ptr - base)
        dist.broadcast_object_list(info, src=0)
        self.receive_info = {k: v for k, v in info[0].items() if k != "handle"}
        if self.rank != 0 and self.kv_share == "ipc":
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
            "accept": self.accept,
            "kv_share": self.kv_share,
        }
        if self.rank == 0 and self.buffer is not None:
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
            if (
                self.planes is not None
            ):  # the test source: the saved planes (TileRT's restore())
                if seq_len != self.planes_n:
                    raise ValueError(
                        f"load_kv({seq_len}): the planes file holds {self.planes_n} positions"
                    )
                self.tk.import_planes(self.planes, seq_len)
            elif self.kv_share == "gloo":  # rank 0 converts, every rank imports
                snap = None
                if self.rank == 0:
                    if self.buffer is None:
                        raise RuntimeError("load_kv: rank 0 has no receive buffer")
                    conv = C.convert(self.buffer, self.max_seq_len, rec, self.token_map, device=self.device)  # fmt: skip
                    snap = _to_device(conv.planes, "cpu")
                planes = _to_device(share_planes(snap), self.device)
                self.tk.import_planes(planes, seq_len)
            elif self.buffer is None:
                raise RuntimeError(
                    "load_kv: this worker has neither a receive buffer nor a planes file"
                )
            else:
                conv = C.convert(
                    self.buffer,
                    self.max_seq_len,
                    rec,
                    self.token_map,
                    device=self.device,
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
