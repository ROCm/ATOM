# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors
"""The target's final norm and LM head of a mono step in one launch
(``kernels.lm_head``: this rank's vocab shard). A step of whole verify requests
(``rq`` rows each) gets each row's greedy id over every rank in the launch and
the verify's verdict on them (``model_ops.verdict``): the forward's head output
(output ids, verdict), which ``ModelRunner`` takes on a greedy step in place of
the logits; any other step at TP 1 its logits. Where the head is not the plain
vocab-sharded one the launch is built for (``MonoLmHead.bind``), or a step it
does not serve (``build``), the original ``compute_logits``."""

import torch
from aiter.ops.flydsl.kernels.tensor_shim import _run_compiled

from atom.model_ops.verdict import UniformVerdict
from atom.models.deepseek_v41.mono.kernels import lm_head as lh


class MonoLmHead:
    """``model``'s final norm and LM head bound for the launch, the buffers it
    writes (the normed rows, this rank's logits, the rows' ids) and an S-row
    step's build. ``rq``: a verify request's rows (None: no greedy head, no
    drafter). ``cond_rates`` / ``seed_base``: forced acceptance's verdict on
    the ids (``UniformVerdict``), None for exact match."""

    @classmethod
    def bind(
        cls, model, tp: int, max_rows: int, dev, rq=None, cond_rates=None, seed_base=0
    ) -> "MonoLmHead | None":
        """The head bound for the launch, None where it is not the plain
        vocab-sharded bf16 one the launch is built for."""
        head, norm = model.head, model.norm
        w = head.weight
        plain_norm = not (
            getattr(norm, "x_pad_to_multiple", 0)
            or getattr(norm, "use_fused_quant", False)
            or getattr(norm, "fused_allreduce", False)
        )
        served = (
            plain_norm
            and getattr(head, "bias", None) is None
            and w.dtype == torch.bfloat16
            and norm.weight.dtype == torch.bfloat16
            and w.is_contiguous()
            and w.shape[0] % lh.RG == 0
            and getattr(head, "tp_size", 1) == tp
        )
        if not served:
            return None
        return cls(model, tp, max_rows, dev, rq, cond_rates, seed_base)

    def __init__(
        self, model, tp: int, max_rows: int, dev, rq=None, cond_rates=None, seed_base=0
    ) -> None:
        head, norm = model.head, model.norm
        w = head.weight
        self.tp, self.rq, self.dev = tp, rq, dev
        self.cond_rates, self.seed_base = cond_rates, seed_base
        # forced acceptance's launch count, every width's verdicts' one
        self.launches = torch.zeros(1, dtype=torch.int64, device=dev)
        self.w, self.norm_w, self.eps = w, norm.weight, float(norm.eps)
        self.vocab, self.hidden = w.shape
        self.x = torch.empty(max_rows, self.hidden, dtype=torch.bfloat16, device=dev)
        self.logits = torch.empty(
            max_rows, self.vocab, dtype=torch.bfloat16, device=dev
        )
        self.ids = torch.empty(max_rows, dtype=torch.int32, device=dev)
        # a width's verdict buffers (fixed addresses: a captured graph's)
        self.verdicts = {}

    def greedy(self, rows: int) -> bool:
        """Whether an S-row step's launch picks its ids (whole verify
        requests)."""
        return self.rq is not None and rows % self.rq == 0

    def build(self, rows: int):
        """An S-row step's launcher, None where the launch does not serve: a
        width it gives logits at TP > 1, whose gather a decode graph does not
        record (``ModelRunner._head_output``: logits at TP 1)."""
        if self.tp > 1 and not self.greedy(rows):
            return None
        key = lh.LmHeadBuild(
            rows=rows, vocab=self.vocab, hidden=self.hidden, eps=self.eps,
            tp=self.tp, argmax=self.greedy(rows),
        )  # fmt: skip
        if key.argmax and rows not in self.verdicts:
            self.verdicts[rows] = UniformVerdict(
                rows // self.rq,
                self.rq,
                self.dev,
                self.cond_rates,
                self.seed_base,
                self.launches,
            )
        return lh.build_lm_head(key)

    def head_of(self, launcher, hidden: torch.Tensor, input_ids, peers: dict):
        """``hidden`` [rows, hidden] (the step's collapsed rows) and its input
        ids -> the head output: (output ids, verdict) of a greedy width, else
        (TP 1, ``build``) the logits [rows, vocab]."""
        rows = hidden.shape[0]
        p = torch.Tensor.data_ptr
        args = {
            "hidden": p(hidden), "norm_w": p(self.norm_w), "x": p(self.x),
            "w": p(self.w), "logits": p(self.logits), "ids": p(self.ids),
            "sym": peers["sym"], "peers": peers["peers"], "rank": peers["rank"],
        }  # fmt: skip
        _run_compiled(launcher, *lh.ABI.pack(args), torch.cuda.current_stream())
        if self.greedy(rows):
            return self.verdicts[rows].launch(self.ids, input_ids)
        return self.logits[:rows]
