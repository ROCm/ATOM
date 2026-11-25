# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors
"""The mono LM head launch (``kernel``) as a verify head's picks: a step of
whole verify requests (``rq`` rows each) gets each row's id over every rank in
one launch -- its greedy id, or with temperature T its draw of
softmax(logits / T), within its top-p where top_p < 1
(``verify_ids.HeadDraw``) -- and the verify on them
(``verify_head.ShardedPicks``).

``MonoLmHead`` is the launch over a vocab-parallel bf16 head's weights and,
optionally, a final RMSNorm before it; a model's mono decode binds it with its
own steps (DeepSeek-V4.1's ``MonoVerifyHead``). ``bind_verify_head`` binds it
for a drafted model whose logits are its ``ParallelLMHead`` of the final rows
(``ModelRunner._bind_verify_head``), ATOM's Triton head the fallback on the
steps it does not serve.
"""

from __future__ import annotations

import logging

import torch
from aiter.dist.parallel_state import get_tp_group
from aiter.ops.flydsl.kernels.tensor_shim import _run_compiled

from atom.distributed.group_agreement import group_agree
from atom.model_ops.embed_head import ParallelLMHead
from atom.model_ops.verify_head import ArgmaxVerifyHead, ShardedPicks, VerifyHead
from atom.model_ops.verify_ids import IdVerifiers, VerifyOutput
from atom.mono.device import race
from atom.mono.lm_head import kernel as lk
from atom.mono.lm_head.dot import KSTEP, RG, UNIT
from atom.mono.lm_head.serves import deployment_refusal
from atom.mono.plan.execution import THREADS
from atom.mono.runtime.lifecycle import owned_peer_buffer
from atom.utils.forward_context import get_forward_context

logger = logging.getLogger("atom")


class MonoLmHead(ShardedPicks):
    """The launch over this rank's head shard ``weight`` [vocab / tp, hidden]
    (bf16, contiguous) of a TP group of ``tp``, after the final RMSNorm
    ``norm`` ((weight, eps); None: the rows come normed), for steps of up to
    ``max_rows`` rows: its buffers, its builds a width, and (``open_peers``)
    its own peer buffer. ``verifiers`` / ``input_ids``: ``ShardedPicks``'.

    ``bind`` is local (no collective): the caller agrees with the other
    ranks that every one bound it, then opens the peers on every rank."""

    @staticmethod
    def refusal(weight: torch.Tensor, bias, norm, tp: int) -> str | None:
        """Why the launch does not serve this head (``bias``: the head's,
        which the launch does not add), None where it does: the deployment's
        (``serves.deployment_refusal``), then the head's shapes."""
        vocab, hidden = weight.shape
        why = deployment_refusal(tp)
        if why is not None:
            return why
        if bias is not None:
            return "a head bias"
        if weight.dtype != torch.bfloat16 or not weight.is_contiguous():
            return f"head weight {weight.dtype}, contiguous {weight.is_contiguous()}"
        if norm is not None and norm[0].dtype != torch.bfloat16:
            return f"final norm weight {norm[0].dtype}"
        if vocab % RG or hidden % (KSTEP * UNIT):
            return f"head shard [{vocab}, {hidden}]"
        # the dot's weight loads take Int32 byte offsets (``buffer_load``:
        # its word index times 4)
        if vocab * hidden * weight.element_size() >= 1 << 31:
            return f"head shard [{vocab}, {hidden}] past 2 GiB"
        if vocab * tp > race.MAX_VOCAB:
            return f"vocab {vocab * tp} past the draw's {race.MAX_VOCAB}"
        return None

    @classmethod
    def bind(
        cls, weight, bias, norm, tp: int, max_rows: int, verifiers: IdVerifiers,
        input_ids,
    ) -> MonoLmHead | None:  # fmt: skip
        """The launch bound, None where it does not serve this head
        (``refusal``, logged)."""
        refusal = cls.refusal(weight, bias, norm, tp)
        if refusal is not None:
            logger.info("mono LM head not bound: %s", refusal)
            return None
        return cls(weight, norm, tp, max_rows, verifiers, input_ids)

    def __init__(self, weight, norm, tp, max_rows, verifiers, input_ids) -> None:
        self.vocab, self.hidden = weight.shape
        self.tp, self.weight, self.norm = tp, weight, norm
        # a pick lane a (row, rank), a norm row a CTA
        self.max_rows = min(max_rows, THREADS // tp, lk.task_count(self.vocab))
        dev = weight.device
        super().__init__(verifiers, input_ids, self.max_rows, dev)
        # the normed rows the launch writes and reads back (``norm``)
        self.x = (
            None
            if norm is None
            else torch.empty(self.max_rows, self.hidden, dtype=weight.dtype, device=dev)
        )
        self.peers = None
        self.builds = {}

    def open_peers(self, group, rank: int) -> None:
        """Its own peer buffer, zeroed once (``kernel``: a launch's
        hand-offs carry its own tag). Collective over ``group``: every rank
        that bound the launch, and only those."""
        nbytes = lk.peer_bytes(
            self.max_rows, lk.task_count(self.vocab), self.tp, draws=True
        )
        self.peers, self._finalizer = owned_peer_buffer(
            self, nbytes, group, rank, self.tp, self.weight.device
        )
        self.peers.bytes.zero_()

    def _key(self, rows: int) -> lk.LmHeadBuild:
        draw = self.verifiers.draw()
        return lk.LmHeadBuild(
            rows=rows, vocab=self.vocab, hidden=self.hidden, tp=self.tp,
            max_rows=self.max_rows, norm=self.norm is not None,
            eps=0.0 if self.norm is None else float(self.norm[1]),
            rq=draw.rq, greedy_t=draw.greedy_t, seed=draw.seed,
        )  # fmt: skip

    def build(self, rows: int):
        """An S-row step's launcher (made once a width), None where the
        launch does not pick its ids (``picks``)."""
        if not self.picks(rows):
            return None
        launcher = self.builds.get(rows)
        if launcher is None:
            launcher = self.builds[rows] = lk.build_lm_head(self._key(rows))
            self.verifiers.of(rows)
        return launcher

    def launch(self, launcher, hidden: torch.Tensor) -> VerifyOutput:
        """``hidden`` [rows, hidden] (the step's final rows, contiguous) ->
        the rows' ids picked in the launch and the verify on them
        (``verify``), whose launch also advances the launch count the next
        launch's tag comes from: the two always run together. A top-p row's
        logits go to a buffer of the launch's own (``kernel.logit_put``),
        from the allocator: the logits this step's head would have made."""
        rows = hidden.shape[0]
        draw = self.verifiers.draw()
        peers = self.peers.kernel_args()
        lg = torch.empty(rows, self.vocab, dtype=torch.bfloat16, device=hidden.device)
        p = torch.Tensor.data_ptr
        args = {
            "hidden": p(hidden),
            "norm_w": 0 if self.norm is None else p(self.norm[0]),
            "x": 0 if self.x is None else p(self.x),
            "w": p(self.weight), "ids": p(self.row_ids),
            "temps": p(draw.temperatures), "top_ps": p(draw.top_ps), "lg": p(lg),
            "tag": p(draw.tag),
            "sym": peers["sym"], "peers": peers["peers"], "rank": peers["rank"],
            "tl": 0,
        }  # fmt: skip
        _run_compiled(launcher, *lk.ABI.pack(args), torch.cuda.current_stream())
        return self.verify(rows)


class MonoArgmaxVerifyHead(VerifyHead):
    """A drafted model's verify head over its ``ParallelLMHead``: the mono
    launch (``MonoLmHead``) on a decode step it serves, ATOM's Triton head
    (``fallback``, no top-p) on any other; both pick every row's id over
    every rank's vocab shard alike on every rank."""

    samples = ids_agree = True

    def __init__(self, head: MonoLmHead, fallback: ArgmaxVerifyHead) -> None:
        self.head, self.fallback = head, fallback

    def draws_top_p(self, rows: int) -> bool:
        return self.head.picks(rows)

    def sample_token_ids(self, hidden: torch.Tensor):
        context = get_forward_context().context
        if context is None or context.is_prefill:
            return None
        launcher = self.head.build(hidden.shape[0])
        if launcher is None:
            return self.fallback.sample_token_ids(hidden)
        return self.head.launch(launcher, hidden.contiguous())


def bind_verify_head(fallback: ArgmaxVerifyHead, max_rows: int) -> VerifyHead:
    """``fallback``'s head (ATOM's Triton picks on a ``ParallelLMHead``) with
    the mono launch in front where every TP rank binds it, else
    ``fallback``. Collective over the TP group: every rank reaches it."""
    lm_head: ParallelLMHead = fallback.lm_head
    tp = get_tp_group()
    head = MonoLmHead.bind(
        lm_head.weight, lm_head.bias, None, tp.world_size, max_rows,
        fallback.verifiers, fallback.input_ids,
    )  # fmt: skip
    if not group_agree(head is not None, tp.cpu_group):
        return fallback
    head.open_peers(tp.cpu_group, tp.rank_in_group)
    logger.info(
        "mono LM head bound: up to %d rows a step, ATOM's picks past them",
        head.max_rows,
    )
    return MonoArgmaxVerifyHead(head, fallback)
