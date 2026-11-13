# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Speculative rejection sampling for MTP drafts (ATOM_MTP_REJECTION_SAMPLING).

Without it a draft is the draft head's argmax, and a verify row keeps it only if
the target's own sample at that row is the same token: acceptance is p(d), at
most the target's top probability, however close the draft head is.

With it (Leviathan et al. 2023, Chen et al. 2023):

* each draft pass draws its token from the draft head's distribution q with the
  sampler the verify uses (aiter ``topk_topp_sample``: the request's temperature,
  the top ``draft_k`` logits, the request's top-p, the draft presence penalty in
  the kernel when ATOM_MTP_DRAFT_PRESENCE_PENALTY is on) and keeps that sampler's
  record of the row -- its k columns, their renormalized probabilities and its
  top-p threshold -- per (request, draft position);
* the verify draws every row with the official sampler, as without this, also
  keeping each row's record; a verify row accepts its draft d with probability
  min(1, p(d) / q(d)), p and q being exactly the distributions the sampler drew
  from (its top-k-first top-p rule, re-derived from the records); on a rejection
  it draws from the residual max(0, p - q), renormalized. The bonus rows keep the
  sampler's own draws.

The output distribution is the target's either way (both schemes are exact);
only how many drafts stand changes. The decision is written as the verify row's
"target token" (the draft when accepted, the residual draw when not, and a
residual draw is never the draft), so the code after sampling -- the fused verify
epilogue's "a draft stands while it equals the target token" -- keeps exactly the
drafts this accepted, unchanged.

A record whose stored token is not the draft being verified (none written, or a
recycled slot) falls back to q = one-hot(d): accept with p(d), else draw from p
without d -- the previous rule, also exact for any draft.

Requests keep a record slot by request id (batches reorder between steps), the
least recently scheduled recycled first, as the presence-penalty slots are. TP1
LM heads only: the draft draws are per rank.
"""

from collections import OrderedDict

import numpy as np
import torch
import triton
import triton.language as tl

_EPS = (
    1e-10  # the sampler's greedy floor (SAMPLER_EPS): temperatures arrive clamped to it
)
_PINNED_RING = 8  # host staging buffers for the per-step slot ids (async H2D)


def _kb(k: int) -> int:
    """The sampler's column block for top-k ``k`` (its debug row is 3 * KB + 2)."""
    return max(16, triton.next_power_of_2(int(k)))


@triton.jit
def _record_dist(rec, K, KB: tl.constexpr, LP: tl.constexpr, greedy):
    """``(columns, probabilities)`` of the distribution the sampler drew a row
    from, from its debug record ``[cols[KB], p[KB], pk[KB], top_p, Z]``: the k
    columns' renormalized probabilities kept while the mass strictly above them
    is below its top-p threshold (``keep = (gt < top_p)``, the select kernel's
    rule), renormalized; a greedy row is one-hot at column 0."""
    lane = tl.arange(0, LP)
    km = lane < K
    cols = tl.load(rec + lane, mask=km, other=-1)
    p = tl.load(rec + KB + lane, mask=km, other=0).to(tl.float32, bitcast=True)
    p = tl.where(km, p, 0.0)
    thr = tl.load(rec + 3 * KB).to(tl.float32, bitcast=True)
    gt = tl.sum(tl.where(p[None, :] > p[:, None], p[None, :], 0.0), axis=1)
    w = tl.where((gt < thr) & km, p, 0.0)
    w = w / tl.sum(w, 0)
    return cols, tl.where(greedy, tl.where(lane == 0, 1.0, 0.0), w)


@triton.jit
def _race(w, ids, noise):
    """A draw from the lanes in proportion to ``w`` (an exponential race:
    argmax of w / E, E ~ Exp(1)); lanes with no weight never win."""
    idx = tl.argmax(tl.where(w > 0, w / noise, -1.0), 0)
    lane = tl.arange(0, w.shape[0])
    return tl.sum(tl.where(lane == idx, ids, 0), 0)


@triton.jit(do_not_specialize=["num_spec_steps", "k_p"])
def _rs_verify_kernel(
    dbg_ptr,  # [n_verify + bs, 3 * KB_P + 2] int32: the verify sampler's records
    rec_ptr,  # [num_slots, num_spec_steps, 3 * KB_Q + 2] int32: the drafts' records
    rtok_ptr,  # [num_slots, num_spec_steps] int32: each record's drawn token
    slot_ptr,  # [bs] int64: each request's record slot
    cu_ptr,  # [bs] cumulative drafts
    tli_ptr,  # [n_verify] the logits row of each verify row
    input_ids_ptr,  # the step's input ids: draft r is input_ids[tli[r] + 1]
    temp_ptr,  # [bs]
    u_ptr,  # [bs, S_MAX, LP + 1] U[0, 1)
    out_v_ptr,  # [n_verify] int32: the decisions, as target tokens
    num_spec_steps,
    k_p,
    KB_P: tl.constexpr,
    K_Q: tl.constexpr,
    KB_Q: tl.constexpr,
    LP: tl.constexpr,
    S_MAX: tl.constexpr,
    EPS: tl.constexpr,
):
    b = tl.program_id(0)
    lane = tl.arange(0, LP)
    start = tl.load(cu_ptr + b - 1, mask=b > 0, other=0)
    nd = tl.load(cu_ptr + b) - start
    greedy = tl.load(temp_ptr + b) <= EPS
    slot = tl.load(slot_ptr + b)
    for pos in tl.static_range(S_MAX):
        if pos < nd:
            r = start + pos
            cp, p = _record_dist(dbg_ptr + r * (3 * KB_P + 2), k_p, KB_P, LP, greedy)
            d = tl.load(input_ids_ptr + tl.load(tli_ptr + r) + 1).to(tl.int32)
            at = slot * num_spec_steps + pos
            ok = tl.load(rtok_ptr + at) == d
            cq, q = _record_dist(rec_ptr + at * (3 * KB_Q + 2), K_Q, KB_Q, LP, greedy)
            # No record of d's draw: q = one-hot(d), the previous rule.
            q = tl.where(ok, q, tl.where(lane == 0, 1.0, 0.0))
            cq = tl.where(ok, cq, tl.where(lane == 0, d, -1))
            p_d = tl.sum(tl.where(cp == d, p, 0.0), 0)
            q_d = tl.sum(tl.where(cq == d, q, 0.0), 0)
            ub = u_ptr + (b * S_MAX + pos) * (LP + 1)
            accept = tl.load(ub + LP) * q_d < p_d
            # q's mass on each of p's columns, then the residual.
            q_at = tl.sum(tl.where(cp[:, None] == cq[None, :], q[None, :], 0.0), axis=1)
            res = tl.maximum(p - q_at, 0.0)
            w = tl.where(tl.sum(res, 0) > 1e-9, res, tl.where(cp == d, 0.0, p))
            noise = -tl.log(1.0 - tl.load(ub + lane))
            tl.store(out_v_ptr + r, tl.where(accept, d, _race(w, cp, noise)))


class SpecRejectionSampling:
    """The draft records, the draft passes' draws and the verify decisions (module
    docstring). One per runner, attached to the draft LM head as ``spec_rs``."""

    def __init__(
        self,
        *,
        max_rows: int,
        mtp_k: int,
        max_num_seqs: int,
        device,
        draft_k: int = 20,
        max_k: int = 64,
    ):
        from aiter.ops.triton.fusions.topk_topp_sample import MAX_K

        self.mtp_k = int(mtp_k)
        self.draft_k = int(draft_k)
        self.max_k = min(int(max_k), MAX_K)
        self.KB_Q = _kb(self.draft_k)
        self.DW = 3 * self.KB_Q + 2
        self.max_rows = int(max_rows)
        self.num_slots = 2 * int(max_num_seqs) + 8
        self.device = device
        i32 = {"dtype": torch.int32, "device": device}
        f32 = {"dtype": torch.float32, "device": device}
        self.records = torch.zeros(self.num_slots, self.mtp_k, self.DW, **i32)
        self.record_tok = torch.full((self.num_slots, self.mtp_k), -1, **i32)
        self.dbg_scratch = torch.zeros(self.max_rows, self.DW, **i32)
        self.tok_scratch = torch.zeros(self.max_rows, **i32)
        self.u_draft = torch.zeros(self.max_rows, **f32)
        self.row_temp = torch.ones(self.max_rows, **f32)
        self.row_topp = torch.ones(self.max_rows, **f32)
        self._topp_scalar = 1.0
        self.step_slots = torch.zeros(self.max_rows, dtype=torch.int64, device=device)
        self._pinned = [
            torch.zeros(self.max_rows, dtype=torch.int64).pin_memory()
            for _ in range(_PINNED_RING)
        ]
        self._pin_next = 0
        self._slot_of: OrderedDict = OrderedDict()
        self._free = list(range(self.num_slots - 1, -1, -1))
        self._presence = None
        self.active_bs = 0

    # ---- per step (eager) -------------------------------------------------
    def _slots_for(self, req_ids) -> np.ndarray:
        in_batch = set(req_ids)
        slots = np.empty(len(req_ids), dtype=np.int64)
        for i, rid in enumerate(req_ids):
            slot = self._slot_of.get(rid)
            if slot is None:
                if not self._free:
                    for old in self._slot_of:
                        if old not in in_batch:
                            self._free.append(self._slot_of.pop(old))
                            break
                slot = self._free.pop()
                self._slot_of[rid] = slot
            else:
                self._slot_of.move_to_end(rid)
            slots[i] = slot
        return slots

    def stage_step(self, req_ids, temperatures, top_ps) -> None:
        """This step's requests (batch order): their record slots, for the
        verify below and the draft passes after it, and the draft passes'
        sampling parameters (each draft row is the step's request of that
        index). A uniform top-p is refilled only when it changes."""
        bs = min(len(req_ids), self.max_rows)
        self.active_bs = bs
        if bs == 0:
            return
        host = self._pinned[self._pin_next]
        self._pin_next = (self._pin_next + 1) % _PINNED_RING
        host[:bs].copy_(torch.from_numpy(self._slots_for(list(req_ids[:bs]))))
        self.step_slots[:bs].copy_(host[:bs], non_blocking=True)
        self.row_temp[:bs].copy_(temperatures[:bs])
        if isinstance(top_ps, torch.Tensor) and top_ps.numel() > 1:
            self.row_topp[:bs].copy_(top_ps[:bs])
            self._topp_scalar = None
        else:
            tp = 1.0 if top_ps is None else float(top_ps)
            if tp != self._topp_scalar:
                self.row_topp.fill_(tp)
                self._topp_scalar = tp

    def store_step(self, i: int, scheduled_bs: int, context=None) -> None:
        """Draft pass ``i``'s records (the scratch the pass wrote) into each
        request's slot; after the pass, before the next one overwrites them."""
        if context is not None and getattr(context, "is_dummy_run", False):
            return
        bs = min(int(scheduled_bs), self.active_bs)
        if bs <= 0 or i >= self.mtp_k:
            return
        slots = self.step_slots[:bs]
        self.records[:, i].index_copy_(0, slots, self.dbg_scratch[:bs])
        self.record_tok[:, i].index_copy_(0, slots, self.tok_scratch[:bs])

    # ---- draft LM head (recorded with the draft passes) --------------------
    def _presence_args(self, draft_presence):
        """aiter's in-kernel presence penalty over the draft passes' buffers
        (`DraftPresencePenalty`): the request's marked tokens and this step's
        earlier drafts, as `DraftPresencePenalty.apply` penalizes them."""
        if draft_presence is None:
            return None
        if self._presence is None or self._presence[0] is not draft_presence:
            dp = draft_presence
            self._presence = (
                dp,
                {
                    "input_ids": dp.ctx,
                    "row_seq": dp.row_seq,
                    "row_pos": dp.pos,
                    "seq_start": dp.seq_start,
                    "slots": dp.slots,
                    "penalties": dp.penalties,
                    "mask": dp.state.mask,
                    "max_extra": dp.state.max_extra,
                },
            )
        return self._presence[1]

    def sample_draft(
        self, logits: torch.Tensor, out: torch.Tensor, draft_presence=None
    ) -> torch.Tensor:
        """Each draft row's token, drawn from its q by the verify's sampler, into
        ``out``; the sampler's records into the scratch."""
        from aiter.ops.triton.fusions import topk_topp_sample as tts

        rows = logits.shape[0]
        assert rows <= self.max_rows, f"{rows} draft rows > {self.max_rows}"
        u = self.u_draft[:rows]
        u.uniform_()  # in the recorded pass: a fresh draw every replay
        tts.topk_topp_sample(
            logits,
            self.row_temp,
            self.draft_k,
            self.row_topp,
            uniform=u,
            eps=_EPS,
            out=self.tok_scratch[:rows],
            debug=self.dbg_scratch[:rows],
            presence=self._presence_args(draft_presence),
            gluon_select=tts.topk_topp_sample_gluon_select_supported(
                logits, self.draft_k
            ),
        )
        out.view(-1).copy_(self.tok_scratch[:rows])
        return out

    # ---- verify (eager) -----------------------------------------------------
    def applies(self, logits, top_ks, top_ps) -> bool:
        from atom.model_ops.sampler import fused_topk_topp_applicable

        return (
            self.active_bs > 0
            and isinstance(logits, torch.Tensor)
            and isinstance(top_ks, int)
            and not isinstance(top_ks, bool)
            and 0 < top_ks <= self.max_k
            and fused_topk_topp_applicable(logits, top_ks, top_ps)
        )

    def verify(
        self,
        logits: torch.Tensor,
        metadata,
        temperatures: torch.Tensor,
        top_k: int,
        top_ps,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """``(target tokens of the verify rows [R], bonus tokens [bs])``: the
        official sampler's draw of every row (its records kept), the verify
        rows' rejection decisions written over its draws, the bonus draws as
        drawn."""
        from atom.model_ops.sampler import fused_topk_topp_sample

        cu = metadata.cu_num_draft_tokens
        tli = metadata.target_logits_indices
        bs = cu.numel()
        n_v = tli.numel()
        S = int(metadata.num_spec_steps)
        assert bs == self.active_bs, f"verify of {bs} requests, staged {self.active_bs}"
        dev = logits.device
        kb_p = _kb(top_k)
        dbg = torch.empty(n_v + bs, 3 * kb_p + 2, dtype=torch.int32, device=dev)
        tokens = fused_topk_topp_sample(
            logits,
            temperatures,
            int(top_k),
            top_ps,
            verify_rows=tli,
            bonus_rows=metadata.bonus_logits_indices,
            cu_num_draft_tokens=cu,
            debug=dbg,
        )
        lp = max(kb_p, self.KB_Q)
        # fp32 explicitly: the runner's default dtype is the model's (bf16).
        u = torch.rand(bs * S * (lp + 1), dtype=torch.float32, device=dev)
        out_v = torch.empty(n_v, dtype=torch.int32, device=dev)
        _rs_verify_kernel[(bs,)](
            dbg,
            self.records,
            self.record_tok,
            self.step_slots,
            cu,
            tli,
            metadata.input_ids,
            temperatures[:bs].float().contiguous(),
            u,
            out_v,
            S,
            int(top_k),
            KB_P=kb_p,
            K_Q=self.draft_k,
            KB_Q=self.KB_Q,
            LP=lp,
            S_MAX=S,
            EPS=_EPS,
            num_warps=1,
        )
        return out_v, tokens[n_v:]

    def warm(self, vocab: int, top_k: int | None = None, draft_presence=None) -> None:
        """Compile the draft draw and the verify (the verify at ``top_k``, the
        draft top-k by default, with a scalar and a per-request top-p) before
        serving and before the draft passes are recorded."""
        from types import SimpleNamespace

        dev = self.device
        S = self.mtp_k
        logits = torch.zeros(S + 1, vocab, dtype=torch.bfloat16, device=dev)
        out = torch.zeros(1, dtype=torch.int32, device=dev)
        self.active_bs = 1
        self.sample_draft(logits[:1], out, draft_presence)
        meta = SimpleNamespace(
            cu_num_draft_tokens=torch.full((1,), S, dtype=torch.int32, device=dev),
            target_logits_indices=torch.arange(S, dtype=torch.int64, device=dev),
            bonus_logits_indices=torch.full((1,), S, dtype=torch.int64, device=dev),
            input_ids=torch.zeros(S + 2, dtype=torch.int64, device=dev),
            num_spec_steps=S,
        )
        temps = torch.ones(1, dtype=torch.float32, device=dev)
        k = top_k or self.draft_k
        self.verify(logits, meta, temps, k, 0.95)
        self.verify(
            logits,
            meta,
            temps,
            k,
            torch.full((1,), 0.95, dtype=torch.float32, device=dev),
        )
        torch.cuda.synchronize(dev)
        self.record_tok.fill_(-1)
        self.active_bs = 0
