# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors
"""An Engram layer's memory gate on a mono step in one launch
(``kernels.engram``): the owed post settled, the FP8 ``wkv`` and the gate, in
place of the column gather and the original ``EngramOp``. A column-sharded
step's on this rank's own columns of every row; a token-parallel step's on the
rank's own rows; TP 1 every row. A layer or a step it is not built for keeps
the original."""

import logging
import os

import torch
from aiter.ops.flydsl.kernels.tensor_shim import _run_compiled

from atom.models.deepseek_v41.mono.kernels import engram as eg
from atom.models.deepseek_v41.mono.kernels.dims import HC, HIDDEN
from atom.mono.runtime.uncached import uncached_bytes

logger = logging.getLogger("atom")
# ``ATOM_H_H2_CHECK=1`` (off by default; read once, at ``MonoEngram.bind``):
# on every eager step whose Engram projection a host layer ran, the host's
# partials are compared bitwise with a projection launch's on the same inputs,
# and (the negative control) with the other Engram layer's (``_check``)
CHECK_ENV = "ATOM_H_H2_CHECK"

# the FP8 linear attributes that decide the weight's stored layout: the
# Engram projection must keep the one K1's ``wqkv_a`` (which the mono GEMV
# reads) keeps
_LAYOUT_ATTRS = (
    "quant_type",
    "params_dtype",
    "native_a8_group_rows",
    "needs_preshuffled_weight",
)


def _engram_served(block) -> bool:
    op = block.engram
    if op is None:
        return False
    w, ref = op.wkv, block.attn.wqkv_a
    scale = getattr(w, "weight_scale", None)
    return (
        op.hc_mult == HC
        and op.hidden_size == HIDDEN
        and op.engram_hidden_size == eg.WIDTH
        and op.gate_weight is not None
        and op.gate_weight.dtype == torch.float32
        and w.weight.dtype == torch.float8_e4m3fn
        and tuple(w.weight.shape) == ((HC + 1) * HIDDEN, eg.WIDTH)
        and scale is not None
        and scale.dtype == torch.float8_e8m0fnu
        and all(getattr(w, a, None) == getattr(ref, a, None) for a in _LAYOUT_ATTRS)
    )


def _staged_rows(rows, layer, joined: bool):
    """The layer's looked-up rows. Once the caller's stream has waited on the
    staging's (``joined``) and taking them would only wait on the layer's
    ``done`` event (no main-stream gather of its own: ``fallback_done``
    empty), they are read in place: that wait, captured, adds a redundant
    edge from the side stream's all-gather to this launch, which a graph
    replay pays as ~4 us of idle before it (G368). Else ``get``."""
    staging = getattr(rows, "staging", None)
    if (
        joined
        and isinstance(rows, dict)
        and staging is not None
        and not staging.fallback_done
        and not getattr(staging, "_served", False)
    ):
        return dict.get(rows, layer)
    return rows.get(layer)


class MonoEngram:
    """The Engram layers the launch serves (layer id -> block), their shared
    buffers (the projection's partials, the gated residual) and an S-row
    step's build."""

    @classmethod
    def bind(cls, model, tp: int, max_rows: int, dev) -> "MonoEngram | None":
        """The Engram layers bound for the launch, None where no layer is one
        the launch is built for (or they disagree on the norm's epsilon)."""
        blocks = {
            spec.layer_id: block
            for spec, block in zip(model.topology, model.layers)
            if _engram_served(block)
        }
        eps = {float(b.engram.norm_eps) for b in blocks.values()}
        if len(eps) != 1:
            return None
        return cls(blocks, eps.pop(), tp, max_rows, dev)

    def __init__(self, blocks: dict, eps: float, tp: int, max_rows: int, dev) -> None:
        self.blocks, self.eps, self.tp = blocks, eps, tp
        # the hosted projection's bitwise check (``CHECK_ENV``), fixed at bind
        self.check = os.environ.get(CHECK_ENV) == "1"
        self.max_rows = max_rows
        self.parts = uncached_bytes(eg.parts_bytes(max_rows, tp), dev)
        # each layer's projection partials, where a host layer projects
        # it (``host_args``) and its gate alone reads them (``launch``)
        self.parts_of = {
            lid: uncached_bytes(eg.parts_bytes(max_rows, tp), dev) for lid in blocks
        }
        self.out = torch.empty(
            1, max_rows, HC, HIDDEN, dtype=torch.bfloat16, device=dev
        )

    def build(self, rows: int, split: bool):
        """The launcher of an S-row step's gate over ``rows`` rows (``split``:
        a column-sharded step, every rank its columns of every row; else the
        rows' every column)."""
        key = eg.EngramBuild(rows=rows, tp=self.tp, eps=self.eps, split=split)
        # both stages; the gate alone, after a host layer's projection
        out = {
            "full": eg.build_engram(key),
            "gate": eg.build_engram(eg.EngramBuild(**{**key.__dict__, "mode": 2})),
        }
        if self.check:
            # (``CHECK_ENV``) the projection launch alone, the reference
            out["proj"] = eg.build_engram(eg.EngramBuild(**{**key.__dict__, "mode": 1}))
        return out

    def host_args(self, layer_id, metadata, t0=0, joined=False):
        """The compressor-slot arguments of the layer hosting Engram layer
        ``layer_id``'s projection (``kernels.layer.engram_host_build``), None
        where its rows are not this launch's to read."""
        block = self.blocks.get(layer_id)
        if block is None or metadata.image_mask is not None:
            return None
        emb = _staged_rows(metadata.engram_embeddings, block.engram.layer_id, joined)
        if emb is None or emb.dtype != torch.bfloat16 or not emb.is_contiguous():
            return None
        op = block.engram
        return {
            "c_w": op.wkv.weight.data_ptr(), "c_wk": op.wkv.weight_scale.data_ptr(),
            "c_ape": emb.data_ptr() + t0 * eg.WIDTH * emb.element_size(),
            "c_parts": self.parts_of[layer_id].data_ptr(),
        }  # fmt: skip

    def launch(
        self, launcher, layer_id, state, metadata, peer_args, t0=0, joined=False,
        hosted=False,
    ) -> bool:
        """``state`` (owing its post) through the layer's gate into ``out``,
        the launch's rows from row ``t0`` (a token-parallel step's own); False
        where the launch does not serve this layer or step (the caller runs
        the original). ``joined``: the caller's stream has already waited on
        the staging's (``_staged_rows``). ``hosted``: a host layer projected
        it this step (``host_args``): the gate alone, on its partials."""
        block = self.blocks.get(layer_id)
        if launcher is None or block is None or state.pending is None:
            return False
        if metadata.image_mask is not None:
            return False
        emb = _staged_rows(metadata.engram_embeddings, block.engram.layer_id, joined)
        rows = state.residual.shape[1]
        if (
            emb is None
            or emb.dtype != torch.bfloat16
            or not emb.is_contiguous()
            or emb.shape[-2:] != (rows, eg.WIDTH)
        ):
            return False
        op = block.engram

        def at(t, row):
            """Row t0 of the [1, rows, ..] tensor ``t``."""
            return t.data_ptr() + t0 * row * t.element_size()

        args = {
            "res": at(state.residual, HC * HIDDEN), "sub": at(state.pending, HIDDEN),
            "post": at(state.post_mix, HC), "comb": at(state.combination, HC * HC),
            "emb": at(emb, eg.WIDTH), "w": op.wkv.weight.data_ptr(),
            "ws": op.wkv.weight_scale.data_ptr(),
            "gate_w": op.gate_weight.data_ptr(), "out": at(self.out, HC * HIDDEN),
            "parts": (self.parts_of[layer_id] if hosted else self.parts).data_ptr(),
            "sym": peer_args["sym"],
            "peers": peer_args["peers"], "rank": peer_args["rank"], "layer": layer_id,
        }  # fmt: skip
        if hosted and self.check and not torch.cuda.is_current_stream_capturing():
            self._check(launcher, layer_id, args, rows)
        build = launcher["gate" if hosted else "full"]
        _run_compiled(build, *eg.ABI.pack(args), torch.cuda.current_stream())
        return True

    def _check(self, launcher, layer_id, args, rows):
        """(``CHECK_ENV``, eager steps) The host layer's partials of Engram
        layer ``layer_id`` against a projection launch's (mode 1) on the same
        inputs, bitwise, and -- the negative control -- the other Engram
        layer's partials against the same reference (expected to differ); a
        tally of (layer, equal, control equal) logged at each new outcome and
        every 50 checks."""
        ref = self.__dict__.get("_ref")
        if ref is None:
            ref = self._ref = torch.zeros_like(self.parts)
        n = eg.parts_bytes(rows, self.tp)
        ref[:n].zero_()
        _run_compiled(
            launcher["proj"], *eg.ABI.pack({**args, "parts": ref.data_ptr()}),
            torch.cuda.current_stream(),
        )  # fmt: skip
        got = self.parts_of[layer_id][:n]
        same = bool(torch.equal(got, ref[:n]))
        other = [k for k in self.parts_of if k != layer_id][0]
        ctrl = bool(torch.equal(self.parts_of[other][:n], ref[:n]))
        cnt = self.__dict__.setdefault("_checks", {})
        key = (layer_id, same, ctrl)
        cnt[key] = cnt.get(key, 0) + 1
        k = sum(cnt.values())
        if cnt[key] == 1 or k <= 4 or k % 50 == 0:
            logger.warning(
                "H2 check: Engram layer %d host partials == projection launch's: %s "
                "(negative control, layer %d's: %s); tally %s", layer_id, same, other,
                ctrl, cnt,
            )  # fmt: skip

    def gated(self, rows: int) -> torch.Tensor:
        """The launch's output residual of an S-row step [1, rows, HC, dim]."""
        return self.out[:, :rows]
