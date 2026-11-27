# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors
"""The mono decode's compressor (``kernels.compress``) between K1 and K2 of
the layers that compress: a layer's weights bound once, a step's launch from
its plans and the cache. A layer or a step the launch is not built for keeps
the original path (``DeepseekV41Attention._compress_batch``)."""

from dataclasses import dataclass

import torch
from aiter.ops.flydsl.kernels.tensor_shim import _run_compiled

from atom.model_ops.deepseek_v41.index_plane import FP4_MFMA
from atom.models.deepseek_v41.mono import index_plan as ip
from atom.models.deepseek_v41.mono.kernels import compress as cp
from atom.models.deepseek_v41.mono.kernels.dims import HEAD_DIM, HIDDEN
from atom.mono.runtime.uncached import uncached_bytes


@dataclass(frozen=True)
class CompressWeights:
    """A compressing layer's tensors as the launch reads them."""

    ratio: int
    wkv: torch.Tensor  # [kv | score rows, hidden] bf16
    ape: torch.Tensor  # [ratio, head_dim] f32
    norm_w: torch.Tensor  # [head_dim] bf16, the model's own (read in place)
    wk: torch.Tensor  # [index dim, head_dim] bf16
    k_norm_w: torch.Tensor  # [index dim] bf16, the model's own (read in place)
    eps: float
    k_eps: float


def compress_weights(attn) -> CompressWeights | None:
    """``attn``'s compressor and index key projection, or None where the
    launch is not built for them (no compressor, another dtype or shape)."""
    comp, indexer = attn.compressor, attn.indexer
    if comp is None or indexer is None or comp.ratio not in (1, 2):
        return None
    wkv, wk = comp.wkv_gate.weight, getattr(indexer, "wk", None)
    if wk is None:
        return None
    wk = wk.weight
    shapes_ok = (
        wkv.dtype == torch.bfloat16
        and tuple(wkv.shape) == (HEAD_DIM * comp.ratio, HIDDEN)
        and wk.dtype == torch.bfloat16
        and tuple(wk.shape) == (ip.DIM, HEAD_DIM)
        and tuple(comp.ape.shape) == (comp.ratio, HEAD_DIM)
        and wkv.is_contiguous()
        and wk.is_contiguous()
        and comp.norm.weight.dtype == torch.bfloat16
        and tuple(comp.norm.weight.shape) == (HEAD_DIM,)
        and comp.norm.weight.is_contiguous()
        and indexer.k_norm.weight.dtype == torch.bfloat16
        and tuple(indexer.k_norm.weight.shape) == (ip.DIM,)
        and indexer.k_norm.weight.is_contiguous()
    )
    if not shapes_ok:
        return None
    return CompressWeights(
        ratio=comp.ratio,
        wkv=wkv,
        ape=comp.ape.float().contiguous(),
        norm_w=comp.norm.weight,
        wk=wk,
        k_norm_w=indexer.k_norm.weight,
        eps=float(comp.norm.eps),
        k_eps=float(indexer.k_norm.eps),
    )


class MonoCompressor:
    """The compressing layers' launches of an S-row step (``builds``,
    ``launch``) and the buffers they share: the projection's partials, the
    mailboxes (``scratch``, zeroed with the step's others) and the latent /
    key rows a check reads."""

    @classmethod
    def bind(cls, model, max_rows: int, kv_fp8: bool, dev) -> "MonoCompressor | None":
        """``model``'s layers the launch serves (``compress_weights``), None
        where it serves none."""
        bound = {
            spec.layer_id: compress_weights(block.attn)
            for spec, block in zip(model.topology, model.layers)
        }
        layers = {layer: w for layer, w in bound.items() if w is not None}
        return cls(layers, max_rows, kv_fp8, dev) if layers else None

    def __init__(self, layers: dict, max_rows: int, kv_fp8: bool, dev):
        """``layers``: layer id -> its ``CompressWeights``."""
        self.kv_fp8 = kv_fp8
        # one build a ratio: its layers' norms agree (a model's one epsilon)
        self.eps = {}
        for w in layers.values():
            self.eps.setdefault(w.ratio, (w.eps, w.k_eps))
        self.layers = {
            layer: w
            for layer, w in layers.items()
            if (w.eps, w.k_eps) == self.eps[w.ratio]
        }
        self.scratch = uncached_bytes(cp.scratch_bytes(max_rows), dev)
        self.parts = uncached_bytes(cp.parts_bytes(max_rows), dev)
        self.latent = torch.zeros(max_rows, HEAD_DIM, dtype=torch.bfloat16, device=dev)
        self.keys = torch.zeros(max_rows, ip.DIM, dtype=torch.bfloat16, device=dev)
        self.tl = torch.zeros(1, dtype=torch.int64, device=dev)

    def weight_pointers(self) -> dict[str, torch.Tensor]:
        """The weights the launches read, every one the model's in place
        (``ape`` is a constant buffer, not a weight)."""
        return {
            f"compress {layer} {name}": getattr(w, name)
            for layer, w in self.layers.items()
            for name in ("wkv", "norm_w", "wk", "k_norm_w")
        }

    def builds(self, rows: int) -> dict:
        """An S-row step's build a ratio its layers compress at."""
        return {
            ratio: cp.CompressBuild(
                rows=rows, ratio=ratio, eps=eps, k_eps=k_eps, kv_fp8=self.kv_fp8
            )
            for ratio, (eps, k_eps) in self.eps.items()
        }

    def launch(self, launchers, rows, spec, cache, step, rope, normed) -> bool:
        """Layer ``spec``'s compressor for this step on ``launchers`` (ratio ->
        launcher, the step's ``builds``), True if it ran; False leaves it to
        the original path (a layer or a step the launch is not built for, or
        a step with nothing to compress, which that path skips too)."""
        w = self.layers.get(spec.layer_id)
        if w is None or not step.plans or not step.requests:
            return False
        args = self._args(w, rows, spec, cache, step, rope, normed)
        if args is None:
            return False
        stream = torch.cuda.current_stream()
        _run_compiled(launchers[w.ratio], *cp.ABI.pack(args), stream)
        return True

    def ratio_of(self, layer_id: int) -> int:
        """The ratio the launch compresses layer ``layer_id`` at, 0 if it
        does not serve it."""
        w = self.layers.get(layer_id)
        return 0 if w is None else w.ratio

    def layer_args(self, rows, spec, cache, step, rope) -> dict | None:
        """Layer ``spec``'s compressor arguments inside the layer kernel
        (``kernels.layer.COMPRESS_ARGS``: its input, scratch, tag and timeline
        the launch's), or None where this step is not built for (the caller
        runs K1, the compressor and K2 apart)."""
        w = self.layers.get(spec.layer_id)
        if w is None or not step.plans or not step.requests:
            return None
        args = self._args(w, rows, spec, cache, step, rope, normed=None)
        if args is None:
            return None
        return {
            f"c_{name}": value
            for name, value in args.items()
            if name not in ("x", "scratch", "layer", "tl")
        }

    def _args(self, w, rows, spec, cache, step, rope, normed):
        """The launch's arguments, or None where it is not built for the
        step's plans or the cache's layout."""
        owner, ratio = spec.kv_owner, w.ratio
        plan = step.plans[ratio]
        kv_state, score_state = cache.compress_state(owner)
        ring = kv_state.shape[1]
        ncap, nwcap = plan.compress_plan_gpu.shape[0], plan.write_plan_gpu.shape[0]
        pages, scale, mode = cache.compressed_rows(owner, ratio)
        units = cache.index_units[owner]
        per_page = cache.geometry.rows_per_page(ratio)
        tables = step.block_tables
        served = (
            # the pool reads the ring before the step's rows, the update writes
            # the step's: never one slot (``fused_compress_attn``'s write plan)
            0 < step.max_q_len + ratio - 1 <= ring
            and ncap <= rows
            and nwcap <= rows
            and mode == ("planar_fp8" if self.kv_fp8 else "none")
            and pages.shape[1] == per_page
            and pages.stride()[1:] == (HEAD_DIM, 1)
            and (scale is None or scale.stride()[1:] == (HEAD_DIM // 32, 1))
            # the FP4 index cache's units only (an FP8 one has no scales)
            and cache.geometry.index_fp4
            and units.layout == FP4_MFMA
            and units.values.is_contiguous()
            and units.scales.is_contiguous()
            and rope.cos_cache.stride(0) == rope.sin_cache.stride(0) == cp.ROPE // 2
            and per_page % cp.FP4_PAGE == 0
            and kv_state.dtype == torch.float32
            and kv_state.stride()[1:] == (HEAD_DIM, 1)
            and score_state.stride() == kv_state.stride()
            and tables.dtype == torch.int32
            and tables.stride(1) == 1
        )
        if not served:
            return None
        if scale is None:
            scale = pages
        p = torch.Tensor.data_ptr
        return {
            # ``normed`` None: the layer kernel's (K1's, ``layer_args``)
            "x": 0 if normed is None else p(normed), "w": p(w.wkv),
            "parts": p(self.parts),
            "cplan": p(plan.compress_plan_gpu), "ncap": ncap,
            "wplan": p(plan.write_plan_gpu), "nwcap": nwcap, "slots": p(step.slots),
            "kv_state": p(kv_state), "score_state": p(score_state),
            "state_stride": kv_state.stride(0), "ring": ring, "ape": p(w.ape),
            "norm_w": p(w.norm_w), "cos": p(rope.cos_cache), "sin": p(rope.sin_cache),
            "pages": p(pages), "page_stride": pages.stride(0) * pages.element_size(),
            "page_scale": p(scale),
            "scale_stride": scale.stride(0) * scale.element_size(),
            "tables": p(tables), "table_stride": tables.stride(0),
            "per_page": per_page, "wk": p(w.wk), "k_norm_w": p(w.k_norm_w),
            "ivals": p(units.values), "iscales": p(units.scales),
            "latent": p(self.latent), "keys": p(self.keys),
            "scratch": p(self.scratch), "layer": spec.layer_id, "tl": p(self.tl),
        }  # fmt: skip
