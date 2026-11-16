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

Not here (TileRT runs them outside the step graphs, and the recording has none of them): the per-request
``admit_kernel`` (``AdmitArgs``) that seeds the device state of a request, and the token tables' build
(``toktab.build_table``: the scope's first block for every token). ``admit(blob)`` takes a recorded ``AdmitArgs`` blob;
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

from ..plan import decode_launches as DL
from . import alloc


def _zero_junk(spec):
    if isinstance(spec, list):
        if spec and spec[0] in ("j", "J"):
            return 0
        return [_zero_junk(x) for x in spec]
    return spec


def blobs(
    plan: dict, rank: int, parity: int, addr: list[dict], junk: str = "recorded"
) -> list[bytes]:
    """Rank ``rank``'s kernarg bytes, launch by launch, for graph ``parity`` over ``addr`` (``address_map()``, or a
    recording's ``decode_launches.recorded_addr``). ``junk``: ``"recorded"`` keeps the bytes TileRT's host left
    uninitialized as recorded (byte-identical blobs), ``"zero"`` zeroes them (as does a parity past the recorded two:
    the warmup's)."""
    out = []
    for launch in DL.rank_launches(plan, rank):
        if junk == "zero" or parity not in (0, 1):
            launch = {**launch, "words": [_zero_junk(w) for w in launch["words"]]}
        out.append(DL.kernargs(launch, rank, parity, addr, plan["mode"]))
    return out


class TkRank:
    """One rank (module docstring). Collective over ``group`` at construction (the ``sym:`` arena)."""

    def __init__(self, rank: int, group=None, device=None, plan: dict | None = None, cached_sym: bool = False,
                 junk: str = "recorded"):  # fmt: skip
        self.plan = plan or DL.load()
        self.rank, self.junk = rank, junk
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
        return blobs(self.plan, self.rank, parity, self.address_map(), self.junk)

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
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g):
                    self.launch_step(parity)
                self.graphs[parity] = g

    def step(self) -> None:
        """The next step: graph ``steps & 1``."""
        self.graphs[self.steps & 1].replay()
        self.steps += 1

    def admit(
        self, blob: bytes, symbol: str, grid=(64, 1, 1), block=(256, 1, 1)
    ) -> None:
        """``admit_kernel`` with a recorded ``AdmitArgs`` blob (module docstring: not in the plan)."""
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
