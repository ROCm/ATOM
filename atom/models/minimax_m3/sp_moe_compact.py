# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""ATOM_P1_SP_MOE_COMPACT: the SP prefill MoE input as amd_kernels' lossless
compact payload instead of bf16 rows (see sp_prefill.py).

``route_quant_local`` runs amd_kernels' prepare kernels (bf16x3 router + MXFP4
+ MXFP8 input quant) on this rank's rows with the router K split of the
full-M bf16 entry, so the routing and every quantized byte equal the entry's.
``CompactMoeHandoff`` packs, gathers (one carved buffer per side) and unpacks
with the fewest host calls; the first call goes through the public entry so
its warm-up loads every compact-mode variant.
"""

import torch

H = 6144
# local payload columns (uint8): MXFP4 codes, MXFP4 scales, MXFP8 codes, MXFP8 scales
_LOCAL = (H // 2, H // 32, H, H // 32)
# gathered compact fields: (columns, dtype, bytes per element)
_COMPACT = (
    (H // 32, torch.uint8, 1),  # routed (MXFP4) scales
    (4, torch.float32, 4),  # top-4 weights (routed scale included)
    (4, torch.int32, 4),  # top-4 ids
    (H, torch.uint8, 1),  # shared MXFP8 codes
    (H // 32, torch.uint8, 1),  # their scales
    (H // 8, torch.uint8, 1),  # one correction bit per value
)

# ATOM_P1_SP_SHARED_LOCAL: the routed-only gather (3296 B/token)
_ROUTED = (
    (H // 2, torch.uint8, 1),  # routed MXFP4 codes
    (H // 32, torch.uint8, 1),  # their scales
    (4, torch.float32, 4),  # top-4 weights (routed scale included)
    (4, torch.int32, 4),  # top-4 ids
)


def _carve(buf: torch.Tensor, rows: int, spec):
    """Consecutive row-major [rows, cols] views of the flat uint8 ``buf``."""
    out, off = [], 0
    for cols, dtype, size in spec:
        n = rows * cols * size
        out.append(buf[off : off + n].view(dtype).view(rows, cols))
        off += n
    return out


class CompactMoeHandoff:
    def __init__(self):
        self._warm = False
        self._k = None
        # ATOM_P1_SP_SHARED_LOCAL: router K-split warm-up done, the side stream
        self._ks_warm = False
        self._side = None

    def _kernels(self):
        if self._k is None:
            from amd_kernels.moe.mxfp4_h6144_i768_e128_top4 import chain, prepared_input
            from amd_kernels.moe.mxfp4_h6144_i768_e128_top4.common.launch import (
                _run_compiled,
            )
            from amd_kernels.moe.mxfp4_h6144_i768_e128_top4.router_mxfp4_mxfp8_quant_row_scales import (
                compile_router_mxfp4_mxfp8_quant_row_scales,
            )
            from amd_kernels.moe.mxfp4_h6144_i768_e128_top4.router_mxfp4_quant import (
                K_SPLIT_ABOVE,
                K_SPLITS,
                router_k_split,
            )
            from amd_kernels.moe.mxfp4_h6144_i768_e128_top4.sidecar_pack import (
                pack_mxfp4_sidecar,
            )
            from amd_kernels.moe.mxfp4_h6144_i768_e128_top4.sidecar_unpack import (
                unpack_mxfp4_sidecar,
            )

            self._k = dict(
                run=_run_compiled,
                router=compile_router_mxfp4_mxfp8_quant_row_scales,
                ks=router_k_split,
                all_ks=sorted({k for _, k in K_SPLITS} | {K_SPLIT_ABOVE}),
                route_parts=chain._route_parts,
                pack=pack_mxfp4_sidecar,
                unpack=unpack_mxfp4_sidecar,
                MoeInput=prepared_input.MoeInput,
                MoeInputCompact=prepared_input.MoeInputCompact,
                prepared=prepared_input._mxfp4_moe_prepared,
                experts=chain.mxfp4_moe_experts,
            )
        return self._k

    def route_quant_local(self, x: torch.Tensor, w, m_full: int):
        """bf16 [m, H] rows -> (q4, s4, q8, s8, topk weights, topk ids), the
        bf16 entry's bits at ``m_full`` rows."""
        k = self._kernels()
        m = x.shape[0]
        ks = k["ks"](m_full)
        buf = torch.empty((m * sum(_LOCAL),), dtype=torch.uint8, device=x.device)
        q4, s4, q8, s8 = _carve(buf, m, [(c, torch.uint8, 1) for c in _LOCAL])
        parts = torch.empty((ks, m, 128), dtype=torch.float32, device=x.device)
        k["run"](
            k["router"](ks),
            x.view(torch.int32).view(-1),
            w.router_weight_lines.view(-1),
            q4.view(-1),
            s4.view(-1),
            q8.view(-1),
            s8.view(-1),
            parts.view(-1),
            m,
            torch.cuda.current_stream(),
        )
        wt, ids = k["route_parts"](parts.permute(1, 0, 2), w)
        return q4, s4, q8, s8, wt, ids

    def __call__(self, x: torch.Tensor, w, world: int, all_gather):
        """This rank's normed bf16 rows -> (slots, shared) partials of every
        token. ``all_gather(outs, ins)``: the rank-order gather of row fields."""
        k = self._kernels()
        m = x.shape[0]
        if not self._warm:
            # every K split a served M can pick, loaded now rather than mid-serving
            for ks in k["all_ks"]:
                self.route_quant_local(x[:32].contiguous(), w, _rows_for(ks))
        q4, s4, q8, s8, wt, ids = self.route_quant_local(x, w, world * m)
        mask = k["pack"](q4, s4, q8, s8)
        rows = world * m
        gbuf = torch.empty(
            (rows * sum(c * s for c, _, s in _COMPACT),),
            dtype=torch.uint8,
            device=x.device,
        )
        outs = _carve(gbuf, rows, _COMPACT)
        all_gather(outs, [s4, wt, ids, q8, s8, mask])
        if not self._warm:
            from amd_kernels.moe.mxfp4_h6144_i768_e128_top4 import mxfp4_moe

            self._warm = True
            return mxfp4_moe(k["MoeInputCompact"](*outs), w, return_partials=True)
        s4g, wtg, idsg, q8g, s8g, maskg = outs
        q4g = k["unpack"](maskg, s4g, q8g, s8g)
        return k["prepared"](
            k["MoeInput"](q4g, s4g, wtg, idsg, q8g, s8g), w, return_partials=True
        )

    def local_shared(
        self, x: torch.Tensor, w, sfull, world: int, all_gather, side_rows: int = 0
    ):
        """ATOM_P1_SP_SHARED_LOCAL: this rank's normed bf16 rows -> (routed slots of
        every token [world m, 4, H], the FULL shared expert of this rank's own rows
        [m, H]) for ``p1_sls_shared.sum5.slots4_reduce_scatter_owner_shared``. The
        gather carries the routed operands only; their bytes, the routing and the
        slots equal the compact / bf16 entry's (same prepare kernels, router K split
        of the full M, same routed launches on the same operands)."""
        import p1_sls_shared

        k = self._kernels()
        m = x.shape[0]
        rows = world * m
        if not self._ks_warm:
            for ks in k["all_ks"]:
                self.route_quant_local(x[:32].contiguous(), w, _rows_for(ks))
            self._ks_warm = True
        q4, s4, q8, s8, wt, ids = self.route_quant_local(x, w, rows)
        current = torch.cuda.current_stream(x.device)
        side = side_rows and m >= side_rows
        if side:
            if self._side is None:
                self._side = torch.cuda.Stream(device=x.device)
            prepared = torch.cuda.Event()
            prepared.record(current)
        gbuf = torch.empty(
            (rows * sum(c * s for c, _, s in _ROUTED),), dtype=torch.uint8, device=x.device
        )
        outs = _carve(gbuf, rows, _ROUTED)
        all_gather(outs, [q4, s4, wt, ids])
        keep = []
        if side:
            # PR #5's order: forked after the preparation, enqueued after the gather
            self._side.wait_event(prepared)
            sh = p1_sls_shared.shared_expert_full(q8, s8, sfull, stream=self._side, keep=keep)
            done = torch.cuda.Event()
            done.record(self._side)
        else:
            sh = p1_sls_shared.shared_expert_full(q8, s8, sfull)
        q4g, s4g, wtg, idsg = outs
        slots = k["experts"]((q4g, s4g), wtg, idsg, w)
        if side:
            current.wait_event(done)
            keep.clear()
        return slots, sh


def _rows_for(ks: int) -> int:
    """A full-M row count whose router_k_split is ``ks``."""
    from amd_kernels.moe.mxfp4_h6144_i768_e128_top4.router_mxfp4_quant import (
        K_SPLIT_ABOVE,
        K_SPLITS,
    )

    for bound, k in K_SPLITS:
        if k == ks:
            return bound
    assert ks == K_SPLIT_ABOVE
    return K_SPLITS[-1][0] + 1
