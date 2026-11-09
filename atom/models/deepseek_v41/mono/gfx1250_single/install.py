# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Which deployments the single-launch step serves, and its routing wrapper.

The step kernel does the whole decode step -- target forward, LM head, sampling, verify, the draft's context KV, its
three stages, head and Markov -- so a served step bypasses ``ModelRunner``'s sampler, rejection sampler and the
DSpark proposer's ``DraftGraph`` as well as the target forward (``SingleStepModel.step_ready``; the runner's three
call sites ask it). Until the runner binds a width, every step keeps the production path.
"""

import logging
import os

import torch
from torch import nn

from atom.mono.runtime.deployment import common_refusals, first_refusal
from atom.mono.runtime.consensus import MonoUnsupported
from atom.mono.runtime.route import LazyRunner
from atom.utils.forward_context import get_forward_context

logger = logging.getLogger("atom")

# the one deployment the kernel is written for: TP4, DSpark5 (a verify of 6 rows a request, a draft block of 5),
# BF16 KV, FP8 index cache, up to 8 requests
SUPPORTED_TP = (4,)
SPEC_TOKENS = 5
MAX_REQUESTS = 8
MAX_ROWS = MAX_REQUESTS * (SPEC_TOKENS + 1)


def single_refusal(atom_config, dims) -> str | None:
    """Why this deployment cannot run the single-launch step, or None."""
    spec = atom_config.speculative_config
    method = None if spec is None else spec.method
    # the shared refusals but their arch check (gfx950 only: the per-layer kernels' wave64 code)
    shared = [
        c
        for c in common_refusals(atom_config, dims, SUPPORTED_TP)
        if not (isinstance(c[1], str) and c[1].startswith("arch "))
    ]
    checks = shared + [
        (
            atom_config.kv_cache_dtype == "bf16",
            f"kv cache {atom_config.kv_cache_dtype}",
        ),
        (
            atom_config.index_cache_dtype == "fp8",
            f"index cache {atom_config.index_cache_dtype}",
        ),
        (method == "dspark", f"speculative method {method}"),
        (
            spec is None or spec.num_speculative_tokens == SPEC_TOKENS,
            "DSpark block size",
        ),
    ]
    return first_refusal(checks)


class SingleStepModel(nn.Module):
    """The loaded V4.1 runtime model. Its forward is unchanged (a served step does not call it); ``step_ready(rows)``
    tells the model runner whether this step runs as the single launch."""

    def __init__(self, model: nn.Module, drafter, atom_config) -> None:
        super().__init__()
        self.model = model
        self._drafter = drafter

        def create():
            from atom.models.deepseek_v41.mono.gfx1250_single.runner import (
                SingleStepRunner,
            )

            return SingleStepRunner(model, drafter, atom_config)

        self._single = LazyRunner(create, "V4.1 single-launch step")

    def step_ready(self, rows: int) -> bool:
        return (
            0 < rows <= MAX_ROWS
            and rows % (SPEC_TOKENS + 1) == 0
            and self._single.ready(rows)
        )

    def _layer_hook(self, k, ref):
        """The production state after layer k - 1, collapsed and normed with layer k's attn_norm (eps as the kernel's
        front, 1e-20): what a k-layer launch's xh holds."""
        from atom.models.deepseek_v41.mono.gfx1250_single.runner import _base_model

        base = _base_model(self.model)
        w = base.layers[k].attn_norm.weight

        def hook(module, args, output):
            x = output.collapse().float()
            ref["x"] = (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-20) * w.float()).to(torch.bfloat16)

        last = base.layers[k - 1]
        hooks = [last.register_forward_hook(hook)]
        # the last layer's attention output (the kernel's y1: after wo_b and the TP all-reduce) and its FFN output (y2)
        attn_fwd = last.attention_forward

        def attention_forward(*a, **kw):
            out = attn_fwd(*a, **kw)
            ref["y1"] = out
            return out

        last.attention_forward = attention_forward
        prep = last.prepare_attention

        def prepare_attention(*a, **kw):
            out = prep(*a, **kw)
            normed = out[0]
            ref["axn"] = normed[0] if isinstance(normed, tuple) else normed  # the attention input (bf16 rows)
            return out

        last.prepare_attention = prepare_attention

        def ffn_hook(module, args, output):
            ref["y2"] = output

        hooks.append(last.ffn.register_forward_hook(ffn_hook))

        class _Undo:
            def remove(self_):
                for h in hooks:
                    h.remove()
                del last.attention_forward
                del last.prepare_attention

        return _Undo()

    def _shadow(self, input_ids):
        """Bring-up (ATOM_MONO_SINGLE_SHADOW=1, eager): the runner of a decode verify forward the single launch shadows,
        or None."""
        if not os.environ.get("ATOM_MONO_SINGLE_SHADOW") or torch.cuda.is_current_stream_capturing():
            return None
        fc = get_forward_context()
        rows = input_ids.numel()
        if fc.spec_decode_metadata is None or getattr(fc.context, "is_dummy_run", False) or not self.step_ready(rows):
            return None
        return self._single.runner

    def _shadow_launch(self, runner, input_ids):
        """The single launch on the production forward's inputs (False: refused, logged; the production forward goes
        on). Before that forward its KV writes are rewritten by it; after it (ATOM_MONO_SINGLE_SHADOW_AFTER=1) it reads
        the rows that forward wrote this step -- the compressed KV and index rows a kernel without COMP cannot."""
        try:
            runner.launch(input_ids, get_forward_context(), getattr(self, "_step_batch", None))
        except MonoUnsupported as why:
            logger.warning("single-launch shadow step skipped (%d rows): %s", input_ids.numel(), why)
            return False
        return True

    def set_step_batch(self, batch) -> None:
        """``ModelRunner.run_model``: this step's batch (the per-request sampling parameters the launch reads)."""
        self._step_batch = batch

    @property
    def single_runner(self):
        return self._single.runner

    def forward(self, input_ids, positions, inputs_embeds=None):
        if inputs_embeds is None:
            shadow = self._shadow(input_ids)
            after = os.environ.get("ATOM_MONO_SINGLE_SHADOW_AFTER") == "1"
            ctx_check = shadow is not None and shadow.shadow_layers == 40 and not after
            if ctx_check:  # the production's draft context rows of the step before, before the launch rewrites them
                shadow.arm_native_draft_capture()
                shadow.shadow_draft_hidden()
                shadow.shadow_ctx_compare(get_forward_context())
            if shadow is not None and not after and not self._shadow_launch(shadow, input_ids):
                shadow = None
            if shadow is not None and ctx_check:
                shadow.shadow_ctx_after_launch(get_forward_context(), input_ids.numel())
                shadow.shadow_draft_after_launch(input_ids.numel())
            ref, hook = {}, None
            if shadow is not None and shadow.shadow_layers < 40:
                hook = self._layer_hook(shadow.shadow_layers, ref)
            # (bring-up: the kernel has no Engram yet; ATOM_MONO_SINGLE_SHADOW_NOENGRAM=1 drops it from the production
            # forward too, so the layers after it compare -- that forward's tokens are then not the model's)
            no_engram = []
            if shadow is not None and os.environ.get("ATOM_MONO_SINGLE_SHADOW_NOENGRAM") == "1":
                for blk in shadow.tables.target_blocks:
                    if getattr(blk, "engram", None) is not None:
                        blk.engram_forward = lambda residual, embeddings, image_mask: residual.clone()
                        no_engram.append(blk)
            try:
                out = self.model(input_ids, positions)
            finally:
                if hook is not None:
                    hook.remove()
                for blk in no_engram:
                    del blk.engram_forward
            if shadow is not None and after and not self._shadow_launch(shadow, input_ids):
                shadow = None
            if shadow is not None:
                shadow.shadow_compare(input_ids.numel(), out, ref.get("x"), ref)
                shadow.shadow_selection(input_ids.numel(), get_forward_context())
                if shadow.shadow_layers == 40:
                    shadow.shadow_taps(input_ids.numel())
                    shadow.shadow_drafts(input_ids, get_forward_context())
            return out
        return self.model(input_ids, positions, inputs_embeds=inputs_embeds)

    def compute_logits(self, hidden_states):
        return self.model.compute_logits(hidden_states)

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)


def install_single_step(model: nn.Module, atom_config, drafter) -> nn.Module:
    from atom.models.deepseek_v41.mono.kernels.dims import Dims

    why = single_refusal(atom_config, Dims)
    if why is not None:
        logger.info("V4.1 single-launch step off: %s", why)
        return model
    return SingleStepModel(model, drafter, atom_config)
