# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors
"""The DSpark draft's context KV write of a mono step in one launch
(``kernels.draft_ctx``): the draft's ``main_proj``, ``main_norm``, the stages'
fused ``wkv`` and their ``kv_norm`` bound once, a step's launch from its taps,
positions and the cache's fp8 windows. A draft, a cache or a step the launch is
not built for keeps ``DeepseekV41DSpark.write_context_kv``."""

import logging
import weakref

import torch
from aiter.ops.flydsl.kernels.tensor_shim import _run_compiled

from atom.models.deepseek_v41.mono.kernels import draft_ctx as dc
from atom.models.deepseek_v41.mono.kernels.dims import HIDDEN, KV_DIM
from atom.mono.runtime.uncached import uncached_bytes

logger = logging.getLogger("atom")

FP8, E8M0 = torch.float8_e4m3fn, torch.float8_e8m0fnu


def _fp8_block_weight(w, ws, rows: int, cols: int) -> bool:
    """A (16, 16)-preshuffled FP8 weight [rows, cols], E8M0 a 32 x 32 block."""
    return (
        getattr(w, "is_shuffled", False)
        and w.dtype == FP8
        and tuple(w.shape) == (rows, cols)
        and w.is_contiguous()
        and ws.dtype == E8M0
        and tuple(ws.shape) == (rows // 32, cols // 32)
        and ws.is_contiguous()
    )


def _bf16_vector(t, n: int) -> bool:
    return t.dtype == torch.bfloat16 and tuple(t.shape) == (n,) and t.is_contiguous()


class MonoDraftContext:
    """The draft's context weights bound for the launch, its buffers and an
    S-row step's build. ``tp``: the attention TP group the projection's
    columns are split over."""

    @classmethod
    def bind(cls, draft, tp: int, max_rows: int, dev) -> "MonoDraftContext | None":
        """The draft's context write bound for the launch, None (its reason
        logged where there is a draft) where the launch is not built for it."""
        why = cls._refusal(draft)
        if why is None:
            return cls(draft, tp, max_rows, dev)
        if draft is not None:
            logger.info("V4.1 mono draft context off: %s", why)
        return None

    def __init__(self, draft, tp: int, max_rows: int, dev) -> None:
        stage0 = draft.mtp[0]
        proj, mnorm = stage0.main_proj, stage0.main_norm
        fused, tail = draft._context_kv_fusion(), draft._draft_kv_tail_fusion()
        layers = draft.context_layers
        kv_norms = [a.kv_norm.weight for a in layers]
        self.tp = tp
        self.proj, self.wkv = (proj.weight, proj.weight_scale), fused[:2]
        self.mnorm_w, self.mnorm_eps = mnorm.weight, float(mnorm.eps)
        self.kv_norms, self.kv_eps = kv_norms, float(tail[1])
        self.layer_ids = [a.spec.layer_id for a in layers]
        self.rope = draft.rope
        self.scratch = uncached_bytes(dc.scratch_bytes(max_rows, tp), dev)
        self.parts = uncached_bytes(dc.parts_bytes(max_rows, tp), dev)
        self.h8 = uncached_bytes(dc.h8_bytes(max_rows), dev)
        self.kvp = uncached_bytes(dc.kvp_bytes(max_rows), dev)
        # the caches whose window layer ids were checked (``window_target``'s
        # ids are the cache's for these layers): the cache objects themselves,
        # not their ``id``, which a rebuilt cache may take again
        self._checked = weakref.WeakSet()

    @staticmethod
    def _refusal(draft) -> str | None:
        """Why the launch does not serve ``draft`` (None: it does)."""
        if draft is None or not hasattr(draft, "_context_kv_fusion"):
            return "no DSpark V4.1 draft"
        stage0 = draft.mtp[0]
        proj, mnorm = stage0.main_proj, stage0.main_norm
        fused, tail = draft._context_kv_fusion(), draft._draft_kv_tail_fusion()
        layers = draft.context_layers
        rope = draft.rope
        checks = [
            (fused is not None, "the stages' wkv not fused (_context_kv_fusion)"),
            (tail is not None, "no fused KV tail (_draft_kv_tail_fusion)"),
            (len(layers) == dc.STAGES, f"{len(layers)} draft stages"),
            (getattr(proj, "bias", None) is None, "main_proj has a bias"),
            (
                _fp8_block_weight(proj.weight, proj.weight_scale, HIDDEN, dc.K_PROJ),
                (
                    f"main_proj {proj.weight.dtype} {tuple(proj.weight.shape)} "
                    f"shuffled={getattr(proj.weight, 'is_shuffled', False)}"
                ),
            ),
        ]
        if fused is not None:
            w, ws, group_rows, width = fused
            checks.append(
                (
                    group_rows == 32
                    and width == KV_DIM
                    and _fp8_block_weight(w, ws, dc.KV_ROWS, HIDDEN),
                    (
                        f"fused wkv {w.dtype} {tuple(w.shape)} group rows "
                        f"{group_rows} shuffled={getattr(w, 'is_shuffled', False)}"
                    ),
                )
            )
        checks += [
            (_bf16_vector(mnorm.weight, HIDDEN), "main_norm weight"),
            (
                all(_bf16_vector(a.kv_norm.weight, KV_DIM) for a in layers),
                "kv_norm weights",
            ),
            (
                all(
                    t.dtype == torch.bfloat16
                    and t.shape[-1] == dc.HALF
                    and t.stride(0) == dc.HALF
                    for t in (rope.cos_cache, rope.sin_cache)
                ),
                f"rope table {rope.cos_cache.dtype} {tuple(rope.cos_cache.shape)}",
            ),
        ]
        return next((why for ok, why in checks if not ok), None)

    def build(self, rows: int):
        """An S-row step's launcher."""
        return dc.build_draft_ctx(
            dc.DraftCtxBuild(
                rows=rows, tp=self.tp, mnorm_eps=self.mnorm_eps, kv_eps=self.kv_eps
            )
        )

    def launch(self, launcher, taps, positions, cache, step, peers: dict) -> bool:
        """The step's context rows into the draft's fp8 windows on
        ``launcher`` (the step's build), True if it ran; False leaves them to
        the original path."""
        args = self._args(taps, positions, cache, step, peers)
        if args is None:
            return False
        _run_compiled(launcher, *dc.ABI.pack(args), torch.cuda.current_stream())
        return True

    def _args(self, taps, positions, cache, step, peers):
        rows = positions.numel()
        window = cache.window_target(self.layer_ids, step)
        if window is None or step.replay is not None or len(taps) != dc.TAPS:
            return None
        ptrs, ids, ring, cached = window
        if cached is None:
            cached = step.swa_cached_until
        int_words = {torch.int32: 1, torch.int64: 2}
        widths = [int_words.get(t.dtype) for t in (positions, step.positions, cached)]
        served = (
            None not in widths
            and all(
                t.dtype == torch.bfloat16
                and t.shape == (rows, HIDDEN)
                and t.stride() == (HIDDEN, 1)
                for t in taps
            )
            and all(
                t.dtype == torch.int32
                for t in (step.batch_ids, step.cu_seqlens_q, step.slots)
            )
            and positions.is_contiguous()
            and step.positions.is_contiguous()
        )
        if not served:
            return None
        if cache not in self._checked:
            if ids.tolist() != self.layer_ids:
                return None
            self._checked.add(cache)
        p = torch.Tensor.data_ptr
        (wp, wps), (wk, wks) = self.proj, self.wkv
        kvn = self.kv_norms
        return {
            "tap0": p(taps[0]), "tap1": p(taps[1]), "tap2": p(taps[2]),
            "wp": p(wp), "wps": p(wps), "wk": p(wk), "wks": p(wks),
            "mnorm_w": p(self.mnorm_w), "kvn0": p(kvn[0]), "kvn1": p(kvn[1]),
            "kvn2": p(kvn[2]), "pos": p(positions), "pos_w": widths[0],
            "ring_pos": p(step.positions), "ring_pos_w": widths[1],
            "batch": p(step.batch_ids), "cu": p(step.cu_seqlens_q),
            "slots": p(step.slots), "cached": p(cached), "cached_w": widths[2],
            "wptrs": p(ptrs), "layer0": self.layer_ids[0],
            "layer1": self.layer_ids[1], "layer2": self.layer_ids[2],
            "ring": ring, "cos": p(self.rope.cos_cache),
            "sin": p(self.rope.sin_cache), "max_pos": self.rope.cos_cache.shape[0],
            "parts": p(self.parts), "h8": p(self.h8), "kvp": p(self.kvp),
            "scratch": p(self.scratch), "sym": peers["sym"],
            "peers": peers["peers"], "rank": peers["rank"], "epoch": peers["epoch"],
        }  # fmt: skip
