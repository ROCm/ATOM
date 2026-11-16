# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Issues the decode step of TileRT 0.1.7 (MIT, Copyright (c) TileRT-team) as its compiled host does: the launches of
# ``plan.decode_launches``, captured one HIP graph a parity and replayed alternately (``dsv41_e2e_prepare`` /
# ``fn_86a400``).
"""One TP8 rank of ATOM's own host for the TileRT-derived DSpark decode step (gfx950).

``TkRank`` holds, on its device:

- the buffers (``alloc.Buffers``: temps, caches, the .so's internal buffers, the token tables, the ``sym:`` peer arena
  and the 7 pointer tables);
- the packed parameters (``load_params``: ``weights.params.rank_params`` on the CPU, each copied once; the two Engram
  table shards streamed in row chunks);
- each graph parity's kernarg blobs (``blobs``: ``decode_launches.kernargs`` over ``address_map()``) and the code
  objects (``codeobj.Launcher``: TileRT's original ``.co`` files or our FlyDSL builds, ``codeobj.CodeTable``);
- ``launch_step(parity)``: the 118 launches, ``hipModuleLaunchKernel`` in params mode, on the current stream;
  ``capture()``: one ``torch.cuda.CUDAGraph`` a parity; ``step()``: the next graph (parity = steps so far & 1, as
  TileRT replays them, so every cross-rank flag word sees a new epoch each step); ``warmup()``: TileRT's prepare (one
  eager step at parity 2, then ``blk_pos`` / ``vpos`` back to -1).

Every rank of the TP group runs the same sequence: the kernels poll each other's ``sym:`` slots within a step.

A request's device work outside the step graphs, as TileRT's ``dsv41_e2e_set_sampling`` / ``dsv41_e2e_begin`` /
``dsv41_e2e_admit`` / ``DsV41Graph.import_kv`` do it (each on the current stream, stream-ordered after any step still
in flight): ``set_sampling(params)`` (``temp:samp_params`` <- 4 x int64, ``sym:sample`` zeroed), ``admit_request``
(``admit_kernel<<<64, 256>>>`` with ``admit_args``, the 176-byte ``AdmitArgs`` built from this rank's buffers),
``import_planes`` (a converted request's planes into the caches) and ``read_logs`` (``ar_num`` / ``ar_acc``).

Not here: the token tables' build (TileRT ``toktab.build_table``: the scope's first block for every token);
``Buffers.toktab`` takes tables built elsewhere.

How to validate on GPU (the CPU checks: ``tk.check``):

1. Record a reference in tkserve with tkshim (``TKSHIM_REC=all``: the graph launches and ``admit_kernel``'s), and have it
   dump, synchronized at a step boundary, every buffer this module allocates (by its plan name: ``temp:`` / ``cache:``
   / ``internal:`` / ``toktab:`` / ``sym:<family>``) on all 8 ranks before step k and after it, with the step's
   parity k & 1.
2. Here, on the 8 ranks: ``TkRank(rank, group)``, ``load_params(ckpt, consts)`` (or the snapshot's ``param:``s, to
   isolate the packers), ``Buffers.toktab`` from the snapshot, ``load_kernels(CodeTable.from_dir(<TileRT co>))``;
   copy the before-step contents into the buffers by name (the ``sym:`` arena's families at ``Buffers.sym_offsets``);
   ``launch_step(k & 1)`` eagerly on every rank; synchronize.
3. Compare with the after-step dump: ``tok_out``, ``ar_acc`` / ``ar_num`` / ``plain_acc``, ``logits``, ``draft_z``,
   ``blk_tok`` / ``blk_pos`` / ``vtok`` / ``vpos``, the caches and ``ctl`` bit for bit (same kernels, same inputs);
   then the same through ``capture()`` and a replay of graph k & 1; then with a ``CodeTable.swap`` of our FlyDSL
   builds (expected equal where a build is bit-exact, else the build's own tolerance); then the junk-free blobs
   (``junk="zero"``) to show the kernels ignore those bytes.
"""

from __future__ import annotations

import struct

from ..plan import decode_launches as DL
from . import alloc

# ---------------------------------------------------------------------------------------------------------------
# admit_kernel (outside the graphs; tkshim recording of 2026-11-16, m7/rec_admit.jsonl, checked by ``check.check_admit``)

ADMIT_SYMBOL = (
    "_ZN6tilert3lib13ds_v4_1_flash12_GLOBAL__N_112admit_kernelENS2_9AdmitArgsE"
)
ADMIT_GRID, ADMIT_BLOCK = (64, 1, 1), (256, 1, 1)
ADMIT_BYTES = 176
ADMIT_PAD = (
    0x0C,
    0x1C,
    0x2C,
    0x4C,
    0x64,
)  # the 4 bytes after each int32 count: TileRT leaves them uninitialized
VERIFY_ROWS = 6  # a DSpark step's target rows: the token and 5 drafts


def stream_len(cap: int) -> int:
    """The ``ar_acc`` / ``ar_num`` logs' length (TileRT ``e2e_graph.stream_len``)."""
    return cap + 64


def admit_args(
    addr: dict,
    first: int,
    pos: int,
    mtp: bool,
    params: list[int] | None = None,
    cap: int = alloc.MAX_SEQ_LEN,
) -> bytes:
    """``AdmitArgs`` (176 B) of one rank, ``addr`` its ``{buffer name: address}``. ``mtp`` is TileRT's flag (1 for a
    request, 0 for ``_reset_streams``); ``params`` the 4 x int64 ``samp_params`` (``dsv41_e2e_begin``: ``pack_params``'
    first three and the .so's sampling counter, which ``dsv41_e2e_set_sampling`` and ``dsv41_e2e_begin`` each bump),
    or None (``dsv41_e2e_admit``: ``samp_params`` untouched).

    The kernel (ISA, matching the recording): zeroes ``plain_acc``; ``ar_acc = [mtp, mtp ? first : 0, 0...]``,
    ``ar_num = [mtp, mtp, 0...]``; ``ctl[0] = pos - 1``, ``ctl[1] = first``, ``ctl[5] = 1`` when mtp, ``ctl[3] = cap``;
    when mtp: ``blk_tok`` / ``vtok`` = ``[first, 128799 (pad)...]``, ``blk_pos = [pos .. pos + 4, -1, -1, -1]``, ``vpos =
    [pos .. pos + 5]``, rows 0..7 of ``X`` the 4 mHC copies of the embedding of ``first`` / the pad; ``samp_params`` <-
    ``params`` when given."""
    b = bytearray(ADMIT_BYTES)
    n = stream_len(cap)
    for off, name, count in (
        (0x00, "temp:plain_acc", n + cap // 8),
        (0x10, "temp:ar_acc", n),
        (0x20, "temp:ar_num", n),
    ):
        struct.pack_into("<Qi", b, off, addr[name], count)
    struct.pack_into("<Q", b, 0x30, addr["temp:ctl"])
    struct.pack_into(
        "<QQi", b, 0x38, addr["temp:blk_tok"], addr["temp:blk_pos"], alloc.ROWS
    )
    struct.pack_into("<QQi", b, 0x50, addr["temp:vtok"], addr["temp:vpos"], VERIFY_ROWS)
    struct.pack_into("<QQ", b, 0x68, addr["param:embed"], addr["temp:X"])
    struct.pack_into("<iiii", b, 0x78, int(pos), int(first), cap, int(bool(mtp)))
    if params is not None:
        if len(params) != 4:
            raise ValueError(f"samp_params: 4 int64, got {len(params)}")
        struct.pack_into(
            "<Q4q", b, 0x88, addr["temp:samp_params"], *(int(p) for p in params)
        )
    return bytes(b)


def _zero_junk(spec):
    if isinstance(spec, list):
        if spec and spec[0] in ("j", "J"):
            return 0
        return [_zero_junk(x) for x in spec]
    return spec


# head_seed_fold's AcceptArgs word at +0x310: the pad id (low half) and, high half, TileRT's simulated acceptance length
# (f32, TILERT_SIMULATE_ACC_LEN, baked into the recorded graphs: 3.51); 0.0 = real acceptance (the kernel's accept
# count from the draft / target probabilities: kernels/gfx950/head_seed_fold.py, both branches diffed bit-exact)
ACCEPT_KERNEL = "head_seed_fold_kernel"
SIM_ACC_OFF = 0x314


def simulated_acceptance(blob: bytes) -> float:
    """The simulated acceptance length a head_seed_fold blob carries (0.0: real acceptance)."""
    return struct.unpack_from("<f", blob, SIM_ACC_OFF)[0]


def with_acceptance(blob: bytes, acc_len: float = 0.0) -> bytes:
    """A head_seed_fold blob with its simulated acceptance length set (0.0: real acceptance)."""
    b = bytearray(blob)
    struct.pack_into("<f", b, SIM_ACC_OFF, float(acc_len))
    return bytes(b)


def blobs(
    plan: dict,
    rank: int,
    parity: int,
    addr: list[dict],
    junk: str = "recorded",
    accept: str = "recorded",
) -> list[bytes]:
    """Rank ``rank``'s kernarg bytes, launch by launch, for graph ``parity`` over ``addr`` (``address_map()``, or a
    recording's ``decode_launches.recorded_addr``). ``junk``: ``"recorded"`` keeps the bytes TileRT's host left
    uninitialized as recorded (byte-identical blobs), ``"zero"`` zeroes them (as does a parity past the recorded two:
    the warmup's). ``accept``: ``"recorded"`` (the recording's simulated acceptance, 3.51) or ``"real"``.
    """
    out = []
    for launch in DL.rank_launches(plan, rank):
        if junk == "zero" or parity not in (0, 1):
            launch = {**launch, "words": [_zero_junk(w) for w in launch["words"]]}
        blob = DL.kernargs(launch, rank, parity, addr, plan["mode"])
        if accept == "real" and ACCEPT_KERNEL in launch["kernel"]:
            blob = with_acceptance(blob, 0.0)
        out.append(blob)
    return out


class TkRank:
    """One rank (module docstring). Collective over ``group`` at construction (the ``sym:`` arena)."""

    def __init__(self, rank: int, group=None, device=None, plan: dict | None = None, cached_sym: bool = False,
                 junk: str = "recorded", accept: str = "recorded"):  # fmt: skip
        self.plan = plan or DL.load()
        if accept not in ("recorded", "real"):
            raise ValueError(f"accept: recorded | real, not {accept!r}")
        self.rank, self.junk, self.accept = rank, junk, accept
        self.buffers = alloc.Buffers(rank, group, device, cached=cached_sym)
        self.device = self.buffers.device
        self.params: dict = {}
        self.launches = DL.rank_launches(self.plan, rank)
        self.launcher = None
        self._blobs: dict[int, list] = {}
        self.graphs: dict = {}
        self.steps = 0

    # ---- parameters
    def load_params(self, ckpt, consts: dict, chunk_rows: int = 1 << 22) -> None:
        """The packed parameters on the device (``weights.params``), every slot the plan names."""
        import torch

        from ..weights import params as P

        for name, t in P.rank_params(
            ckpt, self.rank, consts, max_seq_len=alloc.MAX_SEQ_LEN
        ):
            self.params[f"param:{name}"] = t.to(self.device)
        for layer in (1, 14):
            lo, hi = P.engram_row_range(consts, layer, self.rank)
            first = P.engram_table(ckpt, layer, self.rank, consts, rows=(0, 1))
            dst = {k: torch.empty((hi - lo, first[k].shape[1]), dtype=torch.uint8, device=self.device)
                   for k in ("table", "table_s")}  # fmt: skip
            for a in range(0, hi - lo, chunk_rows):
                part = P.engram_table(
                    ckpt,
                    layer,
                    self.rank,
                    consts,
                    rows=(a, min(hi - lo, a + chunk_rows)),
                )
                for k in dst:
                    dst[k][a : a + part[k].shape[0]].copy_(part[k])
            self.params[f"param:layers.{layer}.engram.table"] = dst["table"]
            self.params[f"param:layers.{layer}.engram.table_s"] = dst["table_s"]
            self.params[f"param:layers.{layer}.engram.row_range"] = first[
                "row_range"
            ].to(self.device)
        missing = [
            n
            for n in DL.buffers_used(self.plan)
            if n.startswith("param:") and n not in self.params
        ]
        if missing:
            raise KeyError(
                f"{len(missing)} parameter(s) the plan names were not produced, e.g. {missing[0]}"
            )

    # ---- kernargs
    def address_map(self) -> list[dict]:
        return self.buffers.address_map(self.params)

    def blobs(self, parity: int) -> list[bytes]:
        return blobs(
            self.plan, self.rank, parity, self.address_map(), self.junk, self.accept
        )

    # ---- launch
    def load_kernels(self, table, extra=()) -> None:
        """The step's code objects on this device (``codeobj.CodeTable``), and the kernels of ``extra`` (symbols:
        ``admit_kernel``'s)."""
        import torch

        from .codeobj import Launcher

        with torch.cuda.device(self.device):
            self.launcher = Launcher(
                table, [x["kernel"] for x in self.launches] + list(extra)
            )

    def _prepared(self, parity: int) -> list:
        """Per launch (symbol, grid, block, shm, kernelParams), the blobs kept alive with them."""
        import ctypes

        if parity not in self._blobs:
            out = []
            for launch, blob in zip(self.launches, self.blobs(parity)):
                # zero-padded to the loaded function's last argument (a swap's may end past TileRT's blob)
                buf = ctypes.create_string_buffer(
                    blob, max(len(blob), self.launcher.arg_bytes(launch["kernel"]), 1)
                )
                params = self.launcher.params(launch["kernel"], buf)
                out.append(
                    (
                        launch["kernel"],
                        launch["grid"],
                        launch["block"],
                        launch["shm"],
                        params,
                        buf,
                    )
                )
            self._blobs[parity] = out
        return self._blobs[parity]

    def launch_step(self, parity: int, stream=None) -> None:
        """The step's launches for graph ``parity`` on ``stream`` (default the current one)."""
        import torch

        s = stream or torch.cuda.current_stream(self.device)
        for sym, grid, block, shm, params, _ in self._prepared(parity):
            self.launcher.launch(sym, grid, block, shm, s.cuda_stream, params)

    def warmup(self) -> None:
        """TileRT's prepare before its captures: one eager step at parity 2, then ``blk_pos`` / ``vpos`` to -1."""
        import torch

        with torch.cuda.device(self.device):
            self.launch_step(2)
            torch.cuda.synchronize(self.device)
            for n in ("temp:blk_pos", "temp:vpos"):
                self.buffers.tensors[n].fill_(-1)
            torch.cuda.synchronize(self.device)
        self._blobs.pop(2, None)

    def capture(self) -> None:
        """One graph a parity (TileRT's 2 graphs of the mode)."""
        import torch

        with torch.cuda.device(self.device):
            for parity in (0, 1):
                self._prepared(
                    parity
                )  # the kernargs before the capture: nothing but the launches inside it
            for parity in (0, 1):
                g = torch.cuda.CUDAGraph()
                # thread_local: a host process (ATOM's ModelRunner) may have other threads issuing work meanwhile
                with torch.cuda.graph(g, capture_error_mode="thread_local"):
                    self.launch_step(parity)
                self.graphs[parity] = g

    def step(self) -> None:
        """The next step: graph ``steps & 1``."""
        self.graphs[self.steps & 1].replay()
        self.steps += 1

    def admit(
        self,
        blob: bytes,
        symbol: str = ADMIT_SYMBOL,
        grid=ADMIT_GRID,
        block=ADMIT_BLOCK,
    ) -> None:
        """``admit_kernel`` with an ``AdmitArgs`` blob (``admit_args``, or a recording's) on the current stream; the
        launcher needs ``ADMIT_SYMBOL`` (``load_kernels(table, extra=(ADMIT_SYMBOL,))``).
        """
        import ctypes

        import torch

        buf = ctypes.create_string_buffer(
            blob, max(len(blob), self.launcher.arg_bytes(symbol), 1)
        )
        params = self.launcher.params(symbol, buf)
        self.launcher.launch(
            symbol,
            grid,
            block,
            0,
            torch.cuda.current_stream(self.device).cuda_stream,
            params,
        )

    # ---- a request's device work outside the graphs
    def _addr(self) -> dict:
        if getattr(self, "_own_addr", None) is None:
            self._own_addr = self.address_map()[self.rank]
        return self._own_addr

    def admit_request(
        self, first: int, pos: int, mtp: bool = True, params: list[int] | None = None
    ) -> None:
        """TileRT's ``dsv41_e2e_begin`` (``params`` given) / ``dsv41_e2e_admit`` on this rank."""
        if mtp and not 0 < pos < alloc.MAX_SEQ_LEN:
            raise ValueError(f"admit: pos {pos} outside (0, {alloc.MAX_SEQ_LEN})")
        self.admit(admit_args(self._addr(), first, pos, mtp, params))

    def set_sampling(self, params: list[int]) -> None:
        """TileRT's ``set_sampling`` on this rank: ``sym:sample`` zeroed, ``temp:samp_params`` <- ``params`` (4 x
        int64, the counter included); the mode is the captured graphs' (mode 5, sampling).
        """
        import torch

        from atom.mono.runtime.uncached import DeviceBytes

        b = self.buffers
        base = b.peer_bases[self.rank] + b.sym_offsets["sample"]
        sym = torch.as_tensor(
            DeviceBytes(base, alloc.SYM_BYTES["sample"]), device=self.device
        )
        sym.zero_()
        src = torch.tensor([int(p) for p in params], dtype=torch.int64)
        b.tensors["temp:samp_params"].copy_(src.to(self.device, non_blocking=False))

    def import_planes(self, planes: dict, n: int) -> None:
        """TileRT's ``DsV41Graph.import_kv(planes, n, 0)`` on this rank (``pd.convert`` planes, on this device)."""
        t = self.buffers.tensors
        for s, ratio in alloc.KV_SOURCES.items():
            for name in ("main_kv", "index_k"):
                for j, src in enumerate(planes[name][s]):
                    t[f"cache:{name}.{s}.{j}"][: n // ratio].copy_(
                        src, non_blocking=True
                    )
        for i, (kv, sc) in enumerate(planes["win"]):
            t[f"cache:win.{i}.kv"].copy_(
                kv.view(t[f"cache:win.{i}.kv"].dtype), non_blocking=True
            )
            t[f"cache:win.{i}.s"].copy_(sc, non_blocking=True)
        for s, st in planes["state"].items():
            t[f"cache:state.{s}"].copy_(st, non_blocking=True)
        t["temp:hist"].copy_(planes["hist"], non_blocking=True)

    def read_logs(
        self, acc_from: int, k: int, tok_from: int
    ) -> tuple[list[int], list[int]]:
        """``ar_num[1 + acc_from :][:k]`` (each step's accepted count) and the accepted tokens after ``tok_from``."""
        t = self.buffers.tensors
        acc = t["temp:ar_num"][0, 1 + acc_from : 1 + acc_from + k].tolist()
        n = sum(acc)
        return acc, t["temp:ar_acc"][0, 1 + tok_from : 1 + tok_from + n].tolist()
