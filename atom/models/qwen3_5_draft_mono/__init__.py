# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# ruff: noqa
# fmt: off
"""ATOM_MTP_MONO=1: the Qwen3.5/3.6 MTP drafter's post-attention block (o_proj, post-attention norm, MoE, final norm,
LM head, argmax) as one persistent gfx1250 FlyDSL kernel (post.py, host.py; the pass before the attention: ATOM_MTP_MONO_PRE=1, pre.py).

TP1 / DP only, at most 16 output rows a pass (else ATOM's own path runs). The kernel reads the MTP weights from the
checkpoint files (experts quantized to MXFP4 1x32 with aiter's reference quantizer) and an MXFP8 1x32 LM head from
ATOM_MTP_MONO_HEAD (a checkpoint directory with outside.safetensors lm_head.weight / weight_scale).
ATOM_MTP_MONO_CHECK=1 (eager): ATOM's path runs too and serves; the kernel's hidden rows and token ids are compared."""
from __future__ import annotations

import logging

import torch

logger = logging.getLogger("atom")
from atom.utils import envs

ENABLED = envs.ATOM_MTP_MONO
CHECK = envs.ATOM_MTP_MONO_CHECK
MAX_ROWS = 16


class DraftMonoHook:
    """One a drafter layer. `last` is the hidden tensor the kernel returned (the model's forward returns it as is;
    compute_draft_ids recognises it and copies the kernel's ids)."""

    def __init__(self, model_path: str):
        self.model_path = model_path
        self.impl = None
        self.failed = False
        self.last = None
        self.ids = None
        self.pending = None          # check mode: (hidden, ids) of the kernel, compared with ATOM's
        self.n_cmp = self.n_tok = self.n_match = 0
        self.max_rel = 0.0

    def _init(self) -> bool:
        if self.impl is not None:
            return True
        if self.failed or torch.cuda.is_current_stream_capturing():
            return False
        try:
            from atom.models.qwen3_5_draft_mono import host as draft_mono

            dev = torch.device("cuda", torch.cuda.current_device())
            impl = draft_mono.DraftMono(self.model_path, envs.ATOM_MTP_MONO_HEAD, device=dev)
            x = torch.zeros(1, 4096, dtype=torch.bfloat16, device=dev)
            impl.run(x, torch.zeros(1, 2048, dtype=torch.bfloat16, device=dev))   # compile + first launch
            torch.cuda.synchronize(dev)
            self.impl = impl
            logger.info("ATOM_MTP_MONO: drafter post-attention kernel ready on %s (check=%s)", dev, CHECK)
        except Exception:  # noqa: BLE001 - the drafter falls back to ATOM's path
            logger.exception("ATOM_MTP_MONO: init failed, ATOM's drafter path runs")
            self.failed = True
        return self.impl is not None

    def applies(self, attn, hidden_states, select_rows) -> bool:
        n = hidden_states.shape[0] if select_rows is None else select_rows.shape[0]
        if not (0 < n <= MAX_ROWS) or hidden_states.dim() != 2:
            return False
        if getattr(attn, "use_fused_sigmoid_mul_quant", False) or not getattr(attn, "attn_output_gate", True):
            return False
        if CHECK and torch.cuda.is_current_stream_capturing():
            return False             # (the comparison syncs: eager passes only)
        return self._init()

    def run(self, x0, residual, select_rows):
        rows = None if select_rows is None else select_rows.to(torch.int64)
        hidden, ids = self.impl.run(x0.contiguous(), residual.contiguous(), rows)
        if CHECK:
            self.pending = (hidden.clone(), ids.clone())
            return None
        self.last, self.ids = hidden, ids
        return hidden

    def check_hidden(self, native) -> None:
        if self.pending is not None and native.shape == self.pending[0].shape:
            a, b = self.pending[0].float(), native.float()
            self.max_rel = max(self.max_rel, float((a - b).abs().max() / b.abs().max().clamp_min(1e-9)))

    def check_ids(self, native_ids) -> None:
        if self.pending is None:
            return
        k = self.pending[1]
        nat = native_ids.reshape(-1)[: k.shape[0]].to(torch.int64)
        self.n_tok += int(k.shape[0])
        self.n_match += int((k == nat).sum())
        self.n_cmp += 1
        self.pending = None
        if self.n_cmp % 500 == 1:
            logger.info("ATOM_MTP_MONO_CHECK: draft ids equal %d / %d (%.2f%%), hidden max rel err %.3g over %d passes",
                        self.n_match, self.n_tok, 100.0 * self.n_match / max(self.n_tok, 1), self.max_rel, self.n_cmp)


def make_hook(model_path: str) -> DraftMonoHook | None:
    return DraftMonoHook(model_path) if ENABLED else None


PRE_ENABLED = envs.ATOM_MTP_MONO_PRE


class DraftPreHook:
    """ATOM_MTP_MONO_PRE=1: the pass up to the q/k/v projection (embedding rows, both pre-fc norms, fc, the input
    norm, q/k/v) as one kernel on the drafter's own weights; the attention's q/k norm + rotary + cache write and the
    attention stay ATOM's. At most 16 rows a pass."""

    def __init__(self, mtp):
        self.impl = None
        try:
            from atom.models.qwen3_5_draft_mono import host as draft_mono

            layer = mtp.layers[0]
            emb = mtp._local_embedding_weight()
            dev = emb.device
            self.impl = draft_mono.DraftPre(emb, mtp.pre_fc_norm_embedding.weight, mtp.pre_fc_norm_hidden.weight, mtp.fc.weight,
                                            layer.input_layernorm.weight, layer.self_attn.qkv_proj.weight, device=dev)
            self.impl.run(torch.zeros(1, dtype=torch.int64, device=dev), torch.zeros(1, 2048, dtype=torch.bfloat16, device=dev))
            torch.cuda.synchronize(dev)
            logger.info("ATOM_MTP_MONO_PRE: drafter pre-attention kernel ready on %s", dev)
        except Exception:  # noqa: BLE001 - ATOM's own path runs
            logger.exception("ATOM_MTP_MONO_PRE: init failed, ATOM's drafter path runs")
            self.impl = None

    def applies(self, input_ids, hidden_states) -> bool:
        ok = self._ok(input_ids, hidden_states)
        if not ok and not getattr(self, "_told", False):
            self._told = True
            logger.info("ATOM_MTP_MONO_PRE: not applied to ids %s %s, hidden %s %s", tuple(input_ids.shape), input_ids.dtype,
                        tuple(hidden_states.shape), hidden_states.dtype)
        return ok

    def _ok(self, input_ids, hidden_states) -> bool:
        return (self.impl is not None and input_ids.dim() == 1 and 0 < input_ids.shape[0] <= MAX_ROWS
                and input_ids.dtype in (torch.int32, torch.int64) and hidden_states.dim() == 2 and hidden_states.dtype == torch.bfloat16
                and hidden_states.shape[0] == input_ids.shape[0])

    def run(self, input_ids, hidden_states):
        return self.impl.run(input_ids.contiguous(), hidden_states.contiguous())


def make_pre_hook(mtp) -> DraftPreHook | None:
    if not PRE_ENABLED:
        return None
    hook = DraftPreHook(mtp)
    return hook if hook.impl is not None else None
