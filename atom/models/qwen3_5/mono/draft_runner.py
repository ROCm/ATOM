# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""The MTP draft's single full-attention decoder layer on the mono kernel.

Only that layer runs mono (the target's full-attention kernel over the draft's weights and KV cache);
embed, norms, ``fc`` and the final norm stay ATOM's. Step 0 runs the verify rows (``spec_steps + 1``
tokens a request), steps 1.. one row a request; each shape has its own ``WidthBuilds``.
"""

import logging

import torch
from aiter.dist.parallel_state import get_tp_group

from atom.models.qwen3_5.mono.config import MAX_TOKENS, Q35Config
from atom.models.qwen3_5.mono.layout import TL_POINTS, layout, shape_dims
from atom.models.qwen3_5.mono.op import Qwen35MonoKernel, Qwen35Variant
from atom.models.qwen3_5.mono.runner import (
    Qwen35MonoDecodeRunner,
    bf16_gemm_refusal,
    check_inputs,
    check_splits,
    flydsl_refusal,
    raise_on_given_up_waits,
)
from atom.models.qwen3_5.mono.weights import LayerWeights, rope_tables
from atom.mono.plan.execution import BLOCKS
from atom.mono.runtime.consensus import MonoUnsupported, bind_agreed
from atom.mono.runtime.lifecycle import owned_peer_buffer
from atom.mono.runtime.mailboxes import StepMailboxes
from atom.mono.runtime.timeline import LayerTimeline
from atom.mono.runtime.widths import WidthBuilds
from atom.utils import envs
from atom.utils.forward_context import get_forward_context

logger = logging.getLogger("atom")


class Qwen35MonoDraftRunner(Qwen35MonoDecodeRunner):
    """The draft (``Qwen3_5MTP``) decoder layer of up to ``MAX_TOKENS`` rows a step, bound (and
    agreed on) like the target's runner."""

    def __init__(self, draft, atom_config, spec_steps: int) -> None:
        tp = get_tp_group()
        self.group = tp.cpu_group
        self.tp, self.rank = tp.world_size, tp.rank_in_group
        self.T = spec_steps + 1
        self.debug = envs.ATOM_MONO_DEBUG
        self.max_len = atom_config.max_model_len
        bind_agreed(lambda: self._bind_draft(draft), self.group)
        # ATOM_MONO_TIMELINE: the draft's stamps go to <prefix>_draft_...
        tl_prefix = envs.ATOM_MONO_TIMELINE
        self.timeline = (
            LayerTimeline(f"{tl_prefix}_draft", 1, TL_POINTS, self.rank, self.dev)
            if tl_prefix
            else None
        )
        self.by_t = {
            t: WidthBuilds(
                lambda S, t=t: self._build_draft_width(S, t),
                self._width_kernels,
                self.group,
                f"Qwen3.5 mono draft (T {t})",
            )
            for t in (self.T, 1)
        }

    def _bind_draft(self, draft) -> None:
        self.mtp = draft.model
        if len(self.mtp.layers) != 1:
            raise MonoUnsupported(f"{len(self.mtp.layers)} MTP layers: one served")
        layer = self.mtp.layers[0]
        if layer.layer_type != "full_attention":
            raise MonoUnsupported(f"MTP layer {layer.layer_type}")
        self.dev = torch.device("cuda", torch.cuda.current_device())
        cus = torch.cuda.get_device_properties(self.dev).multi_processor_count
        if cus != BLOCKS:
            raise MonoUnsupported(
                f"{cus} CUs: the kernel needs all {BLOCKS} CTAs co-resident"
            )
        # the draft layer is built for the FP8 checkpoint only
        w13 = layer.mlp.experts.w13_weight
        if w13.dtype != torch.float8_e4m3fn:
            raise MonoUnsupported(
                f"MTP layer experts {w13.dtype}: the draft runs mono with the FP8 checkpoint"
            )
        # the kernel follows ATOM's unfused MTP layer (all-reduce, then add + norm); a fused layer's
        # input norm would quantize differently
        if getattr(layer, "fuse_allreduce_norm", True):
            raise MonoUnsupported(
                "the MTP layer folds its all-reduce into its norms; the kernel follows the unfused layer"
            )
        self.cfg = Q35Config(
            tp=self.tp,
            rank=self.rank,
            weights="fp8",
            qk_fusion=envs.ATOM_ENABLE_QK_NORM_ROPE_CACHE_QUANT_FUSION,
            pa_flydsl=envs.ATOM_PA_FLYDSL,
        )
        why = bf16_gemm_refusal(self.cfg) or (
            flydsl_refusal(self.cfg) if self.cfg.pa_flydsl else None
        )
        if why is not None:
            raise MonoUnsupported(why)
        self.full = [True]
        self.layers = [(LayerWeights(self.cfg, True), self._pack(0, layer))]
        self.layer_nums = [layer.self_attn.attn.layer_num]
        # the draft reaches up to 2k positions past the last kept token
        rope = rope_tables(
            layer.self_attn.rotary_emb, self.cfg, self.max_len, spare=2 * (self.T - 1)
        )
        if rope is None:
            raise MonoUnsupported(
                f"the RoPE caches do not cover {self.max_len} positions"
            )
        self.cos, self.sin = rope
        logger.info("Qwen3.5 mono draft: rank %d bound the MTP layer", self.rank)

    def _build_draft_width(self, S: int, T: int) -> dict:
        W, p = self.layers[0]
        dims = shape_dims(self.cfg)
        sym_bytes = layout(S, self.tp, True, tokens_per_seq=T, **dims)[1]["_bytes"]
        peers, free_peers = owned_peer_buffer(
            self, sym_bytes, self.group, self.rank, self.tp, self.dev
        )
        var = Qwen35Variant(
            self.cfg,
            S,
            True,
            self.rank,
            self.tp,
            self.group,
            tokens_per_seq=T,
            debug=self.debug,
            timeline=self.timeline is not None,
            peers=peers,
        )
        op = Qwen35MonoKernel(
            W, S, self.rank, self.tp, variant=var, packed=p, tokens_per_seq=T
        )
        mailboxes = StepMailboxes(peers, var.scratch, debug=self.debug)
        torch.cuda.synchronize()
        return {
            "variants": {True: var},
            "ops": [op],
            "mailboxes": mailboxes,
            "free_peers": free_peers,
        }

    def step_t(self, rows: int) -> int | None:
        """This draft step's tokens a request (T for step 0, 1 after), or None when mono does not serve
        it (a verify past MAX_TOKENS rows keeps every draft step ATOM's)."""
        bs = get_forward_context().context.scheduled_bs
        if bs * self.T > MAX_TOKENS:
            return None
        if rows == bs * self.T:
            return self.T
        if rows == bs:
            return 1
        return None

    def prepare(self, rows: int) -> bool:
        t = self.step_t(rows)
        return t is not None and self.by_t[t].prepare(rows)

    def prebuild(self) -> None:
        """Build every draft width now (collective): step 0 runs eagerly, so a width first met while
        serving would stall on its compile and peer handshake."""
        for bs in range(1, MAX_TOKENS // self.T + 1):
            self.by_t[self.T].prepare(bs * self.T)
            self.by_t[1].prepare(bs)

    def layer(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """The decoder layer on ``x`` [rows, hidden]; returns the f32 residual stream (final norm is
        the caller's)."""
        md = get_forward_context().attn_metadata
        S = x.shape[0]
        T = self.step_t(S)
        pos = (positions[0] if positions.dim() == 2 else positions)[:S].to(torch.int32)
        slots = md.slot_mapping[:S].to(torch.int32)
        # step 0 reuses the verify's metadata, whose context_lens a graph may have padded
        check_splits(md, S // T, T, self.cfg.pa_flydsl)
        bt = md.block_tables[: S // T]
        ctx = md.context_lens[: S // T].to(torch.int32)
        if T > 1:
            bt = bt.repeat_interleave(T, 0)
        lc = get_forward_context().kv_cache_data[f"layer_{self.layer_nums[0]}"]
        built = self.by_t[T][S]
        h = x.float().contiguous()
        ins = {
            "kv": (lc.k_cache, lc.v_cache, lc.k_scale, lc.v_scale, pos, ctx, bt, slots),
            "rope": (self.cos, self.sin),
        }
        check_inputs(built, get_forward_context().kv_cache_data, h, [ins])
        built["mailboxes"].begin_step()
        out = built["ops"][0].forward(
            h,
            **ins,
            layer=0,
            begin=False,
            tl=self.timeline.ptr(0) if self.timeline is not None else 0,
        )
        eager = not torch.cuda.is_current_stream_capturing()
        if self.debug and eager:
            raise_on_given_up_waits(
                built["variants"].values(), built["mailboxes"], self.rank, S
            )
        if self.timeline is not None and eager:
            self.timeline.step_done(S)
        return out
