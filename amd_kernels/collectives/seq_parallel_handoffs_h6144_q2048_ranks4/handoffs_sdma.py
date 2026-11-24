# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""``SeqParallelHandoffs`` with the bytes moved by SDMA pushes (``collectives/sdma_push_ranks4``): the same phases and
the same outputs bit for bit, with no collective on the CUs. Every phase method takes ``between``, called after a
push is armed and before its wait: a DBO microbatch passes its yield, so the other microbatch computes while the
engines copy; a single batch passes nothing and waits at once.

Per layer, as ``SeqParallelHandoffs`` (``handoffs.py``), with the same local kernels:

1. ``input_gather``: the quantizer on this rank's rows, pushed into every peer's codes region.
2. ``attention_moe`` / ``attention_dense``: the pack (``mxfp8_pack_q2048``) into a send buffer, block p pushed to
   peer p's all-to-all region; the unpack (``pull_unpack_q2048_shuffled`` without its publication, reading the local
   region; its flag reads see an always-ready word); o_proj_full, the norm (+ the MoE preparation, its compact payload
   pushed; dense: ``gemma_add_rmsnorm_mxfp8_row_major`` writes this rank's slot of the codes region, pushed from
   there).
3. ``ffn_norm_gather``: MoE: ``slots4_stage_rows`` (slot sum + shared) into a stage buffer, block p pushed to peer p's
   reduce-scatter region, this rank's block copied in, ``sum4_add_rmsnorm_mxfp8`` on the region writes this rank's
   slot of the codes region, pushed from there. Dense: the partial's blocks pushed, the four-pointer ``sum4``, the
   norm into the codes slot, pushed.
4. ``ffn_reduce_scatter`` and ``gather``: as ``handoffs.py``; ``gather`` returns a copy (its regions are reused).

Region reuse (``SdmaPush``'s contract): a region's consumer on this rank runs before this rank's next push, and
every peer waits for that push before it pushes into the region again, since every phase gathers from every rank.
Two pushes into one region need a phase in between (its push follows the first one's consumer): ``gather`` alternates
between two regions of its own, since two gathers come in a row (the final output and an aux hidden state) and the
last layer's reduce-scatter comes right before them (sharing its region raced: a peer's gather landed before this
rank's sum read it).

Memory per instance (``max_tokens`` 16384, 4096 rows per rank): ~0.86 GB of regions, plus the stage buffer of the
step. One instance per DBO microbatch (their regions must differ); every rank calls the phases of an instance in the
same order.
"""

import struct

import torch

from .handoffs import _COMPACT, H, MAX_TOKENS, MIN_ROWS, Q, ROW_ALIGN, WORLD, _round_up, pack, shard_compute, shard_rows
from .mxfp8_pack_q2048 import ROW_BYTES

_FP8 = torch.float8_e4m3fn
A2A, COMPACT, RS, CODES, GATHER0, GATHER1 = range(6)
_UNPACK_WGS = 32
_READY = 0x7FFFFFFF  # the unpack's flag words: always at or past any sequence number


def _field_bytes(cols, dtype) -> int:
    return cols * torch.empty((), dtype=dtype).element_size()


def _compact_offsets(rows: int) -> list[int]:
    offs, off = [], 0
    for cols, dtype in _COMPACT:
        offs.append(off)
        off = _round_up(off + WORLD * rows * _field_bytes(cols, dtype), 256)
    return offs + [off]


def _codes_offsets(rows: int) -> tuple[int, int]:
    """(scales offset, total) of the codes region at ``rows``: codes [4 rows, H] e4m3, scales [4 rows, H / 32]."""
    s = _round_up(WORLD * rows * H, 256)
    return s, s + WORLD * rows * (H // 32)


def _noop():
    pass


class SdmaSeqParallelHandoffs:
    """The phases of ``SeqParallelHandoffs`` (same signatures, plus ``between``) on SDMA pushes, sized for
    ``max_tokens`` tokens per call. Collective: every rank of the four-rank ``cpu_group`` constructs it at the same
    point; it compiles the local kernels the transport adds (no collective in their warm-up)."""

    def __init__(self, rank: int, cpu_group, device: torch.device, max_tokens: int = MAX_TOKENS):
        from amd_kernels.collectives.sdma_push_ranks4 import SdmaPush

        assert 0 < max_tokens <= MAX_TOKENS, max_tokens
        device = torch.device(device)
        if device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        self.device, self.rank = device, rank
        self.max_tokens = max_tokens
        self.rows = shard_rows(max_tokens)
        self.padded = WORLD * self.rows
        r = self.rows
        sizes = [0] * 6
        sizes[A2A] = WORLD * r * ROW_BYTES
        sizes[COMPACT] = _compact_offsets(r)[-1]
        sizes[RS] = WORLD * r * H * 2
        sizes[CODES] = _codes_offsets(r)[1]
        sizes[GATHER0] = sizes[GATHER1] = WORLD * r * H * 2
        self.t = SdmaPush(rank, cpu_group, device, sizes)
        self.ready = torch.full((64,), _READY, dtype=torch.int32, device=device)
        self.zero_counter = torch.zeros(1, dtype=torch.int32, device=device)
        self._gathers = 0
        self._warm_transport()
        torch.accelerator.synchronize()

    def supports(self, num_tokens: int) -> bool:
        return 0 < num_tokens <= self.max_tokens

    # -- local kernels -------------------------------------------------------------------------------------------

    @staticmethod
    def _kernels():
        from amd_kernels.collectives.pull_collectives_ranks4.host import _kernels
        from amd_kernels.collectives.pull_collectives_ranks4.pull_unpack_q2048_shuffled import _kernel
        from amd_kernels.collectives.pull_collectives_ranks4.slots4_host import _stage_rows, _sum_norm

        return _kernel(_UNPACK_WGS, False), _stage_rows(), _sum_norm(), _kernels()[-1]

    def _warm_transport(self) -> None:
        """Compile the local kernels the SDMA phases launch (zero payloads, no collective)."""
        from amd_kernels.norm.gemma_add_rmsnorm_h6144.add_rmsnorm import gemma_add_rmsnorm_mxfp8_row_major

        dev, rows = self.device, ROW_ALIGN
        send = torch.zeros((WORLD * rows, ROW_BYTES), dtype=torch.uint8, device=dev)
        self._unpack([send[p * rows : (p + 1) * rows].data_ptr() for p in range(WORLD)], rows)
        slots = torch.zeros((WORLD * rows, 4, H), dtype=torch.bfloat16, device=dev)
        stage = self._stage(slots, slots[:, 0].contiguous())
        res = torch.zeros((rows, H), dtype=torch.bfloat16, device=dev)
        w = torch.zeros(H, dtype=torch.bfloat16, device=dev)
        q = torch.empty((rows, H), dtype=_FP8, device=dev)
        s = torch.empty((rows, H // 32), dtype=torch.uint8, device=dev)
        self._sum_norm(stage.data_ptr(), res, w, 1e-6, q, s)
        out = self._sum4([stage.data_ptr() + p * rows * H * 2 for p in range(WORLD)], rows)
        gemma_add_rmsnorm_mxfp8_row_major(out, res, w, 1e-6)

    def _unpack(self, srcs: list[int], rows: int):
        """Codes ``[rows, 4 Q]`` + scales in o_proj_full's layout from the four ranks' packed blocks at ``srcs``."""
        from amd_kernels.collectives.pull_collectives_ranks4.host import _run_compiled

        unpack = self._kernels()[0]
        codes = torch.empty((rows, WORLD * Q), dtype=torch.uint8, device=self.device)
        scales = torch.empty((_round_up(rows, 32), WORLD * Q // 32), dtype=torch.uint8, device=self.device)
        f = self.ready.data_ptr()
        _run_compiled(unpack, codes.data_ptr(), scales.data_ptr(), *srcs, f, f, f, f, 0, 1, self.rank, rows,
                      torch.cuda.current_stream(self.device))  # fmt: skip
        return codes.view(_FP8), scales

    def _stage(self, slots, shared) -> torch.Tensor:
        """Slot sum + shared of every token -> a bf16 ``[4 rows, H]`` stage buffer (``slots4_stage_rows``)."""
        from amd_kernels.collectives.pull_collectives_ranks4.host import _run_compiled

        total = slots.shape[0]
        assert slots.shape == (total, 4, H) and shared.shape == (total, H) and total % WORLD == 0
        for x in (slots, shared):
            assert x.dtype == torch.bfloat16 and x.is_contiguous() and x.device == self.device
        stage = torch.empty((total, H), dtype=torch.bfloat16, device=self.device)
        _run_compiled(self._kernels()[1], slots.data_ptr(), shared.data_ptr(), stage.data_ptr(), self.zero_counter, 0,
                      total, torch.cuda.current_stream(self.device))  # fmt: skip
        return stage

    def _sum_norm(self, staging_ptr: int, residual, weight, eps, q, scale):
        """``sum4_add_rmsnorm_mxfp8`` on four contiguous bf16 slabs at ``staging_ptr`` -> codes into ``q``, scales
        into ``scale``; returns the new residual."""
        from amd_kernels.collectives.pull_collectives_ranks4.host import _run_compiled

        rows = residual.shape[0]
        assert residual.shape == (rows, H) and residual.dtype == torch.bfloat16 and residual.is_contiguous()
        assert weight.shape == (H,) and weight.dtype == torch.bfloat16
        res_out = torch.empty_like(residual)
        _run_compiled(self._kernels()[2], staging_ptr, residual.data_ptr(), weight.data_ptr(), q.data_ptr(),
                      scale.data_ptr(), res_out.data_ptr(), q.numel(), scale.numel(), rows, H, H,
                      struct.unpack("<i", struct.pack("<f", eps))[0], torch.cuda.current_stream(self.device))  # fmt: skip
        return res_out

    def _sum4(self, srcs: list[int], rows: int) -> torch.Tensor:
        """The fixed-order sum of four bf16 ``[rows, H]`` slabs at ``srcs`` (the reduce-scatter's ``sum4``)."""
        from amd_kernels.collectives.pull_collectives_ranks4.host import _run_compiled

        out = torch.empty((rows, H), dtype=torch.bfloat16, device=self.device)
        _run_compiled(self._kernels()[3], out.data_ptr(), *srcs, rows * H * 2, torch.cuda.current_stream(self.device))
        return out

    # -- pushes --------------------------------------------------------------------------------------------------

    def _region_view(self, k, off, shape, dtype) -> torch.Tensor:
        n = 1
        for d in shape:
            n *= d
        nbytes = n * _field_bytes(1, dtype)
        return self.t.region(k)[off : off + nbytes].view(dtype).view(shape)

    def _peers(self):
        return [p for p in range(WORLD) if p != self.rank]

    def _gather_push(self, k, fields, rows, between, keep=()):
        """Push this rank's ``fields`` = [(src address, field offset in region k, bytes per row)] to every peer's
        slot ``rank`` of each field, call ``between``, wait."""
        items = [(p, src, off + self.rank * rows * n, rows * n) for p in self._peers() for src, off, n in fields]
        seq = self.t.push(k, items, keep=keep)
        between()
        self.t.wait(k, seq)

    def _codes_slot(self, rows):
        """This rank's slot of the codes region: (codes [rows, H], scales [rows, H / 32])."""
        s_off, _ = _codes_offsets(rows)
        me = slice(self.rank * rows, (self.rank + 1) * rows)
        codes = self._region_view(CODES, 0, (WORLD * rows, H), _FP8)
        scales = self._region_view(CODES, s_off, (WORLD * rows, H // 32), torch.uint8)
        return codes[me], scales[me]

    # -- phases --------------------------------------------------------------------------------------------------

    def input_gather(self, x: torch.Tensor, between=_noop):
        rows = x.shape[0]
        q, s = shard_compute("input", x)
        return self._codes_gather_from(rows, q, s, between)

    def _codes_gather_from(self, rows, q, s, between):
        """Push this rank's codes ``q`` / scales ``s`` (a tensor elsewhere) as the codes region's slot."""
        s_off, _ = _codes_offsets(rows)
        codes = self._region_view(CODES, 0, (WORLD * rows, H), _FP8)
        scales = self._region_view(CODES, s_off, (WORLD * rows, H // 32), torch.uint8)
        me = slice(self.rank * rows, (self.rank + 1) * rows)
        codes[me].copy_(q)
        scales[me].copy_(s)
        self._gather_push(CODES, [(q.data_ptr(), 0, H), (s.data_ptr(), s_off, H // 32)], rows, between, keep=(q, s))
        return codes, scales

    def _codes_gather_slot(self, rows, between):
        """Push the codes region's own slot, already written by a local kernel."""
        s_off, _ = _codes_offsets(rows)
        mine_c, mine_s = self._codes_slot(rows)
        self._gather_push(CODES, [(mine_c.data_ptr(), 0, H), (mine_s.data_ptr(), s_off, H // 32)], rows, between)
        codes = self._region_view(CODES, 0, (WORLD * rows, H), _FP8)
        scales = self._region_view(CODES, s_off, (WORLD * rows, H // 32), torch.uint8)
        return codes, scales

    def _unpacked(self, attn_output, rows, between):
        t, d = attn_output.shape
        padded = WORLD * rows
        assert d == Q and t <= padded, (attn_output.shape, rows)
        send = torch.empty((padded, ROW_BYTES), dtype=torch.uint8, device=self.device)
        pack(attn_output, out=send[:t])
        send[t:].zero_()
        block = rows * ROW_BYTES
        items = [(p, send.data_ptr() + p * block, self.rank * block, block) for p in self._peers()]
        seq = self.t.push(A2A, items, keep=(send,))
        between()
        self.t.wait(A2A, seq)
        region = self.t.region(A2A).data_ptr()
        srcs = [send.data_ptr() + p * block if p == self.rank else region + p * block for p in range(WORLD)]
        return self._unpack(srcs, rows)

    def attention_moe(self, attn_output, o_weight, o_scale, residual, norm_weight, eps, moe_weights, between=_noop):
        from amd_kernels.moe.mxfp4_h6144_i768_e128_top4 import MoeInputCompact

        rows = residual.shape[0]
        xq, xs = self._unpacked(attn_output, rows, between)
        local, r = shard_compute("moe", xq, xs, o_weight, o_scale, residual, norm_weight, eps, moe_weights=moe_weights)
        fields = list(local.tensors())
        offs = _compact_offsets(rows)
        outs, items = [], []
        me = slice(self.rank * rows, (self.rank + 1) * rows)
        for (cols, dtype), off, x in zip(_COMPACT, offs, fields):
            assert x.shape == (rows, cols) and x.dtype == dtype and x.is_contiguous(), (x.shape, x.dtype)
            out = self._region_view(COMPACT, off, (WORLD * rows, cols), dtype)
            out[me].copy_(x)
            outs.append(out)
            items.append((x.data_ptr(), off, _field_bytes(cols, dtype)))
        self._gather_push(COMPACT, items, rows, between, keep=tuple(fields))
        return MoeInputCompact(*outs), r

    def attention_dense(self, attn_output, o_weight, o_scale, residual, norm_weight, eps, between=_noop):
        from amd_kernels.norm.gemma_add_rmsnorm_h6144.add_rmsnorm import gemma_add_rmsnorm_mxfp8_row_major

        rows = residual.shape[0]
        xq, xs = self._unpacked(attn_output, rows, between)
        h = shard_compute("dense", xq, xs, o_weight, o_scale)
        q, s = self._codes_slot(rows)
        _, _, r = gemma_add_rmsnorm_mxfp8_row_major(h, residual, norm_weight, eps, q_out=q, scale_out=s)
        codes, scales = self._codes_gather_slot(rows, between)
        return codes, scales, r

    def _reduce_scatter_push(self, full: torch.Tensor, rows, between):
        """Block p of this rank's bf16 ``[4 rows, H]`` ``full`` -> slot ``rank`` of peer p's reduce-scatter region;
        returns the region's four slabs' addresses with this rank's own block read from ``full``."""
        block = rows * H * 2
        items = [(p, full.data_ptr() + p * block, self.rank * block, block) for p in self._peers()]
        seq = self.t.push(RS, items, keep=(full,))
        between()
        self.t.wait(RS, seq)
        region = self.t.region(RS).data_ptr()
        return [full.data_ptr() + p * block if p == self.rank else region + p * block for p in range(WORLD)]

    def ffn_norm_gather(self, ffn_out, residual, norm_weight, eps, between=_noop):
        rows = residual.shape[0]
        q, s = self._codes_slot(rows)
        if isinstance(ffn_out, tuple):
            stage = self._stage(*ffn_out)
            self._reduce_scatter_push(stage, rows, between)
            # sum4_add_rmsnorm_mxfp8 reads the four slabs contiguously: this rank's block joins the region
            block = rows * H * 2
            region = self.t.region(RS)
            region[self.rank * block : (self.rank + 1) * block].copy_(stage.view(-1).view(torch.uint8)[self.rank * block : (self.rank + 1) * block])
            r = self._sum_norm(region.data_ptr(), residual, norm_weight, eps, q, s)
        else:
            from amd_kernels.norm.gemma_add_rmsnorm_h6144.add_rmsnorm import gemma_add_rmsnorm_mxfp8_row_major

            assert ffn_out.dtype == torch.bfloat16 and ffn_out.is_contiguous() and ffn_out.shape == (WORLD * rows, H)
            summed = self._sum4(self._reduce_scatter_push(ffn_out, rows, between), rows)
            _, _, r = gemma_add_rmsnorm_mxfp8_row_major(summed, residual, norm_weight, eps, q_out=q, scale_out=s)
        codes, scales = self._codes_gather_slot(rows, between)
        return codes, scales, r

    def ffn_reduce_scatter(self, ffn_out, between=_noop) -> torch.Tensor:
        if isinstance(ffn_out, tuple):
            full = self._stage(*ffn_out)
        else:
            full = ffn_out
            assert full.dtype == torch.bfloat16 and full.is_contiguous() and full.shape[1] == H
        rows = full.shape[0] // WORLD
        return self._sum4(self._reduce_scatter_push(full, rows, between), rows)

    def gather(self, x: torch.Tensor, between=_noop) -> torch.Tensor:
        rows, cols = x.shape
        assert cols == H and x.dtype == torch.bfloat16 and x.is_contiguous()
        k = GATHER0 if self._gathers % 2 == 0 else GATHER1
        self._gathers += 1
        out = self._region_view(k, 0, (WORLD * rows, H), torch.bfloat16)
        out[self.rank * rows : (self.rank + 1) * rows].copy_(x)
        self._gather_push(k, [(x.data_ptr(), 0, H * 2)], rows, between, keep=(x,))
        return out.clone()

    def warm_up(self, o_weight, o_scale, moe_weights=None) -> None:
        """The compute variants (``SeqParallelHandoffs.warm_up``: the same registered entries)."""
        from .handoffs import SeqParallelHandoffs

        SeqParallelHandoffs.warm_up(self, o_weight, o_scale, moe_weights)

    def drain(self) -> None:
        """Host: every push done; the held sources dropped."""
        self.t.drain()
