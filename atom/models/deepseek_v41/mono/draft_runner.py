# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors
"""One TP rank's DSpark draft backbone on the mono kernels.

The draft's three stages are V4.1 blocks with window-only attention: the same
K1 / K2 as the target's layers (``runner``), S = requests x block width rows,
the draft's routing (``dspark_n_routed_experts`` / ``dspark_num_experts_per_tok``)
and no indexer. Two things differ from a target step, both in the step's
integers (``step_meta.write_draft_step_meta``), not in a kernel:

- K1 writes the block's keys into the stage's ring at the block's own positions
  (anchor + 1 ..). The ring holds window + width slots, so they never overwrite
  the window; only the draft reads these rings, and the target's context write
  replaces every position it commits.
- Every row's keys are its request's window up to the anchor and its whole
  block, both ways: one span, the same for every row of the block.

The key order is the paged decode's (window, then block), not the draft's
Triton ``sparse_attn`` indices order, so the attention's sums are not the
original draft's bit for bit.
"""

from copy import copy
from dataclasses import replace

import torch
from aiter.ops.flydsl.kernels.tensor_shim import _run_compiled

from atom.model_ops.deepseek_v41.dspark import build_block_state
from atom.model_ops.deepseek_v41.mhc import SinglePassHCState
from atom.models.deepseek_v41.mono import attention_plan as aplan
from atom.models.deepseek_v41.mono.config import MAX_REQUESTS
from atom.models.deepseek_v41.mono.kernels import attn_pre as k1
from atom.models.deepseek_v41.mono.kernels import draft_head as dh
from atom.models.deepseek_v41.mono.runner import V41MonoDecodeRunner
from atom.models.deepseek_v41.mono.step_meta import write_draft_step_meta
from atom.models.deepseek_v41.mono.weights import bind_layer
from atom.mono.runtime.consensus import MonoUnsupported
from atom.utils import envs
from atom.utils.forward_context import get_forward_context


class DraftMonoRunner(V41MonoDecodeRunner):
    """The draft (``models.deepseek_v41.dspark.DSparkDraftModel``) backbone of
    up to ``MAX_REQUESTS`` requests, a block of ``width`` rows each, bound (and
    agreed on) like the target's runner."""

    # the draft head (``_bind_head``), agreed on as the target's launches
    optional_launches = V41MonoDecodeRunner.optional_launches + ("heads",)
    heads = None
    # a full beginning every pass, recording the target epoch it followed
    # (``epoch_begin``: the target's slim condition)
    always_full_begin = True
    # () -> the target's runner or None (``dispatch.install_mono_decode``)
    epoch_target = None

    def __init__(self, draft) -> None:
        config = draft.config
        self.width = draft.block_size
        self.max_rows = self.width * MAX_REQUESTS
        self.experts = config.dspark_n_routed_experts
        self.topk = config.dspark_num_experts_per_tok
        # no indexer: the K2a build without it
        self.index_bound_max = 0
        self._bind_agreed(lambda: self._bind_draft(draft))

    def _bind_draft(self, draft) -> None:
        # a row's keys: its request's window up to the anchor and its whole
        # block, every one in a window slot (``attention_plan``)
        keys = draft.config.sliding_window + self.width
        if keys > aplan.window_slots(aplan.selection_slots(False)):
            raise MonoUnsupported(f"{keys} draft attention keys a row")
        self.model = draft
        self._bind_rank()
        config = copy(draft.config)
        config.n_routed_experts = self.experts
        config.num_experts_per_tok = self.topk
        self.layers = [
            bind_layer(block, config, self.dims, self.moe_tp) for block in draft.mtp
        ]
        for table in (draft.rope.cos_cache, draft.rope.sin_cache):
            if table.dtype != torch.bfloat16 or table.shape[-1] != k1.HALF:
                raise MonoUnsupported(f"draft rope table {table.dtype} {table.shape}")
        self.timelines = {}
        self.timeline = False
        self.check = False
        self.split_k2 = False
        # the backbone's output is read whole (no column gather): replicated
        self.shards = False
        self.debug = envs.ATOM_MONO_DEBUG
        self.selecting = []
        self._allocate(config)
        self.positions = torch.empty(
            self.max_rows, device=self.scratch.device, dtype=torch.int64
        )
        self._bind_head(draft)

    def _bind_head(self, draft) -> None:
        """The draft head (``kernels.draft_head``: the last stage's owed post,
        collapse and final norm, the LM head and the Markov sampler in one
        launch) at every request count; none where the draft's head is not
        the vocab-sharded greedy one or its final norm not the plain RMSNorm,
        and at a request count its build refuses (``DraftHeadBuild.refusal``)."""
        head, norm = draft.head, draft.mtp[-1].norm
        if head is None or not draft._sharded_markov_sample():
            return
        markov = draft.mtp[-1].markov_head
        w, w1, w2 = head.weight, markov.markov_w1.weight, markov.markov_w2.weight
        if head.vocab_start_idx != self.rank * w.shape[0] or not all(
            t.dtype == torch.bfloat16 and t.is_contiguous()
            for t in (w, w1, w2, norm.weight)
        ):
            return
        if norm.x_pad_to_multiple or norm.use_fused_quant or norm.fused_allreduce:
            return
        # the confidence head in the launch too: a bias-free bf16 [hidden + r]
        proj = draft.mtp[-1].confidence_head.proj
        cw = proj.weight
        confidence = (
            proj.bias is None
            and cw.dtype == torch.bfloat16
            and tuple(cw.shape) == (1, w.shape[1] + w1.shape[1])
            and cw.is_contiguous()
        )
        heads = {}
        for requests in range(1, MAX_REQUESTS + 1):
            key = dh.DraftHeadBuild(
                requests=requests,
                positions=self.width,
                vocab=w.shape[0],
                hidden=w.shape[1],
                rank_r=w1.shape[1],
                tp=self.tp,
                from_state=True,
                eps=norm.eps,
                confidence=confidence,
            )
            if key.refusal() is None:
                heads[requests] = dh.build_draft_head(key)
        if not heads:
            return
        self.heads = heads
        dev = self.scratch.device
        rows, hidden = self.max_rows, w.shape[1]
        # each row's confidence (``confidence``; else the caller's head)
        self.head_conf = (
            torch.empty(rows, dtype=torch.float32, device=dev) if confidence else None
        )
        self.head_ids = torch.empty(rows, dtype=torch.int32, device=dev)
        self.head_embeds = torch.empty(
            rows, w1.shape[1], dtype=torch.bfloat16, device=dev
        )
        # the head's rows: collapsed (the confidence head's) and normed
        self.head_hidden = torch.empty(rows, hidden, dtype=torch.bfloat16, device=dev)
        self.head_x = torch.empty(rows, hidden, dtype=torch.bfloat16, device=dev)
        p = torch.Tensor.data_ptr
        pre, post, comb = self.gates_ffn
        # the peer buffer's (``sym`` / ``peers`` / ``rank``) at the call: it is
        # created after this bind (``_bind_agreed``)
        self.head_args = {
            "x": p(self.head_x), "w": p(w), "w1": p(w1), "w2": p(w2),
            "ids": p(self.head_ids), "embeds": p(self.head_embeds),
            "res": p(self.res_ffn), "sub": p(self.ffn_out), "post": p(post),
            "comb": p(comb), "pre": p(pre), "norm_w": p(norm.weight),
            "hidden": p(self.head_hidden),
            "cw": p(cw) if confidence else 0,
            "conf": p(self.head_conf) if confidence else 0,
            "first": head.vocab_start_idx, "tl": 0,
        }  # fmt: skip

    def head(self, anchor_ids):
        """After ``backbone``: the block's ids [B, width], each position's
        ``W1[x_{j-1}]`` [B, width, r], the collapsed rows [B, width, dim] (the
        confidence head's two inputs) and the confidences [B, width] (fp32;
        None where the launch does not compute them), one launch on the last
        stage's state; None where no build serves ``anchor_ids``' requests
        (the caller finishes the state itself: ``normed_rows``)."""
        requests = anchor_ids.numel()
        heads = self.heads
        if heads is None or requests not in heads or anchor_ids.dtype != torch.int32:
            return None
        args = dict(self.head_args)
        args["anchor"] = anchor_ids.data_ptr()
        peers = self.peers.kernel_args()
        args["sym"], args["peers"], args["rank"] = (
            peers["sym"],
            peers["peers"],
            peers["rank"],
        )
        _run_compiled(heads[requests], *dh.ABI.pack(args), torch.cuda.current_stream())
        rows = requests * self.width
        conf = self.head_conf
        return (
            self.head_ids[:rows].view(requests, self.width),
            self.head_embeds[:rows].view(requests, self.width, -1),
            self.head_hidden[:rows].view(requests, self.width, -1),
            None if conf is None else conf[:rows].view(requests, self.width),
        )

    def normed_rows(self, state, requests):
        """``backbone``'s last-stage state finished as the original draft
        does (collapse, then the final norm): (normed [B width, dim], the
        collapsed rows [B, width, dim])."""
        hidden = state.collapse().view(requests, self.width, -1)
        return self.model.mtp[-1].norm(hidden).flatten(0, 1), hidden

    def _layer_blocks(self) -> list:
        return [(block.attn.spec, block) for block in self.model.mtp]

    def _layer_keys(self, s: int, numa: bool = True) -> list:
        # stage 0 starts from the block's embeddings: no owed post to fold, and
        # from the embedding rows themselves where the peer gather makes them
        # (``backbone``)
        return [
            k1.AttnPreBuild(
                tokens=s,
                fold=i != 0,
                tp=self.tp,
                timeline=self.timeline,
                diag_off=self.diag_off,
                kv_fp8=self.kv_fp8,
                embed_in=i == 0 and self.embed_peer is not None,
                # K2's dual table (``_post_key``): one layout in the layer kernel
                numa=self._numa(s) if numa else "",
            )
            for i in range(len(self.model.mtp))
        ]

    def _post_key(self, s: int, numa: bool = True):
        # the draft's attention: its window and its block, no selection
        # (``write_draft_step_meta``): every key slot the window's, every row
        # of a block seeing all of it
        return replace(
            super()._post_key(s, numa), attn_sel=False, block_rows=self.width
        )

    def _tok_par(self, s: int) -> bool:
        # the backbone gathers no rows: every K1 and K2 runs on all of them
        return False

    def _rope_positions(self, cache, step) -> torch.Tensor:
        return self.positions

    def backbone(self, anchor_ids, anchor_positions):
        """``DSparkDraftModel.block_backbone`` of ``anchor_ids``' requests at the
        full block width, up to the last stage's state (its owed post not yet
        settled): ``head`` finishes it, or ``normed_rows``."""
        draft = self.model
        metadata = get_forward_context().attn_metadata
        cache = metadata.cache
        if cache.packed:
            raise MonoUnsupported("packed KV cache")
        requests = anchor_ids.numel()
        rows = requests * self.width
        self._use_rows(rows)
        if self.mailboxes.link is None and self.epoch_target is not None:
            # the target runner's mailboxes, once it exists: a pass recorded
            # before then (a capture bakes no link) leaves the target's every
            # beginning full -- never a slim one without the record
            target = self.epoch_target()
            if target is not None:
                self.mailboxes.link = target.mailboxes
        self.mailboxes.begin_step(self._zero_extents(rows))
        # the requests' blocks as one step's rows, request after request
        if self.embed_peer is not None:
            # each block's rows (the anchor, then the noise token) straight
            # from the vocab shards; stage 0 writes their streams itself
            # (``embed_in``), its pre mix the constant one
            embedded = self.embed_peer(
                anchor_ids,
                self.embedded[:rows],
                self.width,
                draft.config.dspark_noise_token_id,
            )
            state = SinglePassHCState(embedded.view(1, rows, -1), self.pre_init)
        else:
            residual, pre_mix = build_block_state(
                draft.embed(anchor_ids),
                draft._noise_embedding(),
                self.width,
                draft.config.hc_mult,
            )
            state = SinglePassHCState(
                residual.flatten(0, 1).unsqueeze(0),
                pre_mix.flatten(0, 1).unsqueeze(0),
            )
        write_draft_step_meta(
            anchor_positions,
            metadata.state_slot_out,
            cache.num_slots,
            cache.geometry,
            cache.geometry.window(0, cache.num_pages),
            self.positions,
            self.ring_rel,
            self.key_meta,
        )
        for block, weights, key in zip(draft.mtp, self.layers, self.build.keys):
            spec = block.attn.spec
            # a stage's K1 and K2 in one launch (``kernels.layer``) where the
            # build has it, else apart -- the target's ``forward``
            if not (
                self._fuses(rows, spec, key, block)
                and self._layer(
                    spec, block, weights, key, state, cache, None, draft.rope
                )
            ):
                residual, gates = self._front(
                    spec, key, weights, state, cache, None, draft.rope
                )
                self._layer_post(
                    spec, block, weights, residual, gates, None, cache, draft.rope
                )
            pre, post, comb = self.gates_ffn
            state = SinglePassHCState(self.res_ffn, pre, self.ffn_out, post, comb)
        self._raise_on_given_up_waits(rows)
        return state
