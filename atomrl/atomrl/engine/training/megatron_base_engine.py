# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""Shared training logic for Megatron-Core engines.

The concrete native engine owns model construction, topology, and checkpoint
format. This base provides common forward, packed batching, log-prob, policy
loss, optimizer, scheduler, and data-parallel helpers.
"""

from __future__ import annotations

import logging
import os
from contextlib import contextmanager, nullcontext
from typing import Any

import torch
import torch.distributed as dist

from atomrl.algorithms.loss_functions import (
    asymmetric_clip_loss,
    policy_gradient_loss,
    reduce_reported_loss,
    reference_kl_loss,
    reference_kl_tokens,
)
from atomrl.algorithms.rollout_correction import rollout_is_weights
from atomrl.core.protocol import DataProto
from atomrl.core.types import AlgorithmName
from atomrl.engine.training.base_engine import BaseEngine
from atomrl.engine.training.bridges.gpt import GPTDims

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("ATOMRL_LOGGING_LEVEL", "INFO"))


def _response_mask_is_token_indexed(tensors: dict[str, torch.Tensor]) -> bool:
    """Whether response_mask uses token columns [S], rather than log-prob columns [S-1]."""
    response_mask = tensors.get("response_mask")
    input_ids = tensors.get("input_ids")
    return bool(
        response_mask is not None
        and input_ids is not None
        and response_mask.shape[-1] >= input_ids.shape[-1]
    )


def moe_dispatcher_kwargs(
    engine_config: dict[str, Any],
    *,
    tp: int,
    cp: int,
    sp: bool,
    max_tokens_per_gpu: int = 0,
) -> dict[str, Any]:
    """Map ``megatron_cfg`` dispatcher fields onto ``TransformerConfig``.

    Megatron-LM's pretrain script derives ``moe_mori_max_tokens_per_rank`` in
    ``validate_args``. AtomRL constructs ``TransformerConfig`` directly, so a
    MORI flex dispatcher must set that heap size here. Packed RL forwards use
    ``max_tokens_per_gpu`` as the per-rank token ceiling (then CP / SP).
    """
    dispatcher = str(engine_config.get("moe_token_dispatcher_type", "alltoall"))
    out: dict[str, Any] = {"moe_token_dispatcher_type": dispatcher}
    if dispatcher != "flex":
        return out
    backend = str(engine_config.get("moe_flex_dispatcher_backend") or "mori")
    out["moe_flex_dispatcher_backend"] = backend
    if backend != "mori":
        return out
    mori_max = engine_config.get("moe_mori_max_tokens_per_rank")
    if mori_max is None:
        budget = int(max_tokens_per_gpu or 0) or 21504
        if cp > 1:
            budget = (budget + cp - 1) // cp
        if sp and tp > 1:
            budget = (budget + tp - 1) // tp
        mori_max = budget
    out["moe_mori_max_tokens_per_rank"] = int(mori_max)
    kernel = engine_config.get("moe_mori_kernel_type")
    if kernel:
        out["moe_mori_kernel_type"] = str(kernel)
    logger.info(
        "MoE flex dispatcher: backend=mori moe_mori_max_tokens_per_rank=%s",
        out["moe_mori_max_tokens_per_rank"],
    )
    return out


# Scratch buffer used by the merged native-engine log-prob gap diagnostic.
_GAP_ROWS: list = []
_GAP_SENTINEL = "/tmp/atomrl_gap_dump_dir"


def _gap_dump_dir() -> str | None:
    directory = os.environ.get("ATOMRL_DUMP_LOGPROB_GAP")
    if directory:
        return directory
    try:
        with open(_GAP_SENTINEL) as sentinel:
            return sentinel.read().strip() or None
    except OSError:
        return None


# Sum / token-count pairs that ``_row_policy_loss`` adds: under
# force_on_policy_ratio what the skipped old-log-prob pass would report and the
# in-loss importance weights, and with a reference policy its KL. Returned as
# pairs because the controller averages each key over workers, which keeps
# sum / count exact.
_ROW_STAT_NAMES = ("tmpe", "ent", "ris", "refkl")
_ON_POLICY_STATS = tuple(f"{n}_{p}" for n in _ROW_STAT_NAMES for p in ("sum", "tok"))


def on_policy_metrics(sums: dict[str, float]) -> dict[str, float]:
    out = {}
    for name in _ROW_STAT_NAMES:
        if sums.get(f"{name}_tok", 0.0) > 0:
            out[f"{name}_sum"] = sums[f"{name}_sum"]
            out[f"{name}_tok"] = sums[f"{name}_tok"]
    return out


class _FusedTokenLogProb(torch.autograd.Function):
    """Memory-efficient per-token log-prob: ``log p(target) = logit_target - logsumexp``.

    Retains a single ``[L, V]`` softmax buffer for backward instead of the
    several ``[L, V]`` tensors that ``log_softmax(logits).gather(...)`` keeps
    alive (the full log_softmax output plus its gradient). Values/gradients are
    exact. Backward uses ``grad_logits = (onehot(target) - softmax) * grad_lp``.
    """

    @staticmethod
    def forward(ctx, logits, target):
        logits = logits.float()
        m = logits.max(dim=-1, keepdim=True).values  # [L,1]
        shifted = logits.sub(m)  # new [L,V]
        exp = shifted.exp_()  # in-place -> exp
        Z = exp.sum(dim=-1, keepdim=True)  # [L,1]
        softmax = exp.div_(Z)  # in-place -> softmax
        logZ = Z.log_().add_(m)  # logsumexp [L,1]
        tgt_logit = logits.gather(-1, target.unsqueeze(-1))  # [L,1]
        log_prob = (tgt_logit - logZ).squeeze(-1)  # [L]
        ctx.save_for_backward(softmax, target)
        return log_prob

    @staticmethod
    def backward(ctx, grad_lp):
        softmax, target = ctx.saved_tensors  # softmax [L,V]
        grad = softmax.neg_()  # -softmax (reuse buffer)
        grad.scatter_add_(-1, target.unsqueeze(-1), torch.ones_like(grad[:, :1]))
        grad.mul_(grad_lp.unsqueeze(-1))
        return grad, None


class MegatronBaseEngine(BaseEngine):
    """Shared implementation for concrete Megatron-Core engines."""

    def __init__(
        self, model_config, engine_config, optimizer_config, model_name: str = ""
    ):
        super().__init__()
        self.model_config = (
            model_config if isinstance(model_config, dict) else vars(model_config)
        )
        self.engine_config = (
            engine_config if isinstance(engine_config, dict) else vars(engine_config)
        )
        self.optimizer_config = (
            optimizer_config
            if isinstance(optimizer_config, dict)
            else vars(optimizer_config)
        )
        self.model_name = model_name or self.model_config.get("local_path", "")
        self.module: torch.nn.Module | None = (
            None  # unwrapped GPTModel (eval fwd, save/load)
        )
        self._ddp: Any = None  # Megatron DistributedDataParallel wrapper
        self.optimizer: Any = None  # Megatron distributed optimizer
        self.lr_scheduler: Any = None  # Megatron OptimizerParamScheduler
        self._dims: GPTDims | None = None
        self.mode: str | None = None

    # -- offload (Ray path: never offload) --
    @property
    def is_param_offload_enabled(self) -> bool:
        return False

    @property
    def is_optimizer_offload_enabled(self) -> bool:
        return False

    def train_mode(self, **kwargs):
        return nullcontext()

    def eval_mode(self, **kwargs):
        return nullcontext()

    # ------------------------------------------------------------------
    def _rank(self) -> int:
        return dist.get_rank() if dist.is_initialized() else 0

    def get_data_parallel_size(self) -> int:
        try:
            from megatron.core import parallel_state as mpu

            return mpu.get_data_parallel_world_size()
        except (AssertionError, RuntimeError):
            return dist.get_world_size() if dist.is_initialized() else 1

    def get_data_parallel_rank(self) -> int:
        try:
            from megatron.core import parallel_state as mpu

            return mpu.get_data_parallel_rank()
        except (AssertionError, RuntimeError):
            return self._rank()

    def get_data_parallel_group(self):
        try:
            from megatron.core import parallel_state as mpu

            return mpu.get_data_parallel_group()
        except (AssertionError, RuntimeError):
            return dist.group.WORLD if dist.is_initialized() else None

    def to(
        self, device: str, model: bool = True, optimizer: bool = True, grad: bool = True
    ) -> None:
        return

    # ------------------------------------------------------------------
    def _forward_logits(self, ids: torch.Tensor, model=None) -> torch.Tensor:
        """Run the model on a single unpadded sequence -> logits [L, V] (float).

        ``model`` defaults to the unwrapped GPTModel (eval); pass ``self._ddp``
        during training so DDP grad hooks fire and grads land in the buffer.
        """
        m = model if model is not None else self.module
        L = ids.numel()
        inp = ids.view(1, L)
        pos = torch.arange(L, device=ids.device).view(1, L)
        out = m(input_ids=inp, position_ids=pos, attention_mask=None)
        logits = out.logits if hasattr(out, "logits") else out
        return logits.view(L, -1).float()

    def _forward_logits_packed(
        self, ids_list, model=None
    ) -> tuple[torch.Tensor, list[int]]:
        """Packed varlen forward: concat ``ids_list`` (per-sequence 1D token tensors)
        into one [1,T] stream and run a single GPTModel forward with thd
        ``PackedSeqParams``. Returns ``(logits [T,V] float, offsets)`` where
        ``offsets[i]:offsets[i+1]`` slices sequence ``i``'s logits.

        Rotary is applied per-segment by Megatron's thd path (via cu_seqlens);
        attention is isolated per-segment by flash_attn_varlen. So each sequence's
        logits are identical to a standalone forward (up to bf16 nondeterminism)."""
        from megatron.core.packed_seq_params import PackedSeqParams

        m = model if model is not None else self.module
        lens = [int(t.numel()) for t in ids_list]
        offsets = [0]
        for L in lens:
            offsets.append(offsets[-1] + L)
        T = offsets[-1]
        tokens = torch.cat([t.view(-1) for t in ids_list], dim=0).view(1, T)
        # per-segment position ids (0..L_i-1); ignored by thd rotary but kept correct.
        pos = torch.cat(
            [torch.arange(L, device=tokens.device) for L in lens], dim=0
        ).view(1, T)
        # Same token-count alignment the pipelined path applies (TP/SP scatter,
        # DSv4's compressor, and TE's assert_dim_for_fp8_exec under FP8). The pp
        # path padded for this and this one did not, so an FP8 run that happened
        # to take the dense path tripped TE instead.
        align = max(1, int(self._packed_token_align()))
        if align > 1 and T % align:
            pad = align - (T % align)
            # One extra cu_seqlens bin so attention stays isolated from the real
            # sequences, and the caller's `offsets` is left untouched so no layout
            # entry points into the pad and its logits are never read for loss.
            tokens = torch.cat([tokens.view(-1), tokens.new_zeros(pad)], dim=0).view(
                1, T + pad
            )
            pos = torch.cat(
                [pos.view(-1), torch.arange(pad, device=pos.device)], dim=0
            ).view(1, T + pad)
            padded_offsets = offsets + [offsets[-1] + pad]
        else:
            pad = 0
            padded_offsets = offsets
        cu = torch.tensor(padded_offsets, dtype=torch.int32, device=tokens.device)
        # Every bin, pad included. The pad bin can be longer than every real
        # sequence (a microbatch of short sequences with a large alignment pad),
        # and rotary/attention slice their tables to max_seqlen -- so taking the
        # max over real sequences only hands that bin a short slice.
        max_seqlen = self._packed_max_seqlen(padded_offsets, 1)  # cp == 1 on this path
        pp = PackedSeqParams(
            cu_seqlens_q=cu,
            cu_seqlens_kv=cu,
            max_seqlen_q=max_seqlen,
            max_seqlen_kv=max_seqlen,
            qkv_format="thd",
        )
        out = m(
            input_ids=tokens,
            position_ids=pos,
            attention_mask=None,
            packed_seq_params=pp,
        )
        logits = out.logits if hasattr(out, "logits") else out
        # Drop the pad bin's rows: callers index by `offsets`, which covers real tokens only.
        return logits.view(T + pad, -1)[:T].float(), offsets

    def _packed_token_align(self) -> int:
        """Token-count divisibility this packed forward must satisfy; 1 = no padding.

        Overridden by MegatronNativeEngine, which knows the run's TP/SP, DSv4 and
        FP8 constraints. A base hook rather than a getattr probe so the contract is
        visible here and a subclass cannot silently fail to provide it.
        """
        return 1

    @staticmethod
    def _packed_max_seqlen(local_offsets, cp: int) -> int:
        """``max_seqlen`` for PackedSeqParams: every cu_seqlens bin, pad included.

        Megatron sizes the RoPE table from this (``get_rotary_seq_len``), then
        ``_apply_rotary_pos_emb_thd`` slices ``freqs[:len]`` per bin. A trailing
        pad bin longer than ``max_seqlen`` therefore gets a short slice, and the
        packed freqs come out shorter than the activation. It only bites when the
        pad exceeds every real sequence, so a microbatch of short sequences is
        what exposes it.
        """
        bins = [
            (local_offsets[i + 1] - local_offsets[i]) * cp
            for i in range(len(local_offsets) - 1)
        ]
        return max(bins) if bins else 0

    def _build_bins(self, lengths: list[int], budget: int) -> list[list[int]]:
        """Greedy bin-packing of row indices into groups whose summed token length
        stays <= ``budget`` (a row longer than budget forms its own bin)."""
        if budget <= 0:
            budget = max(lengths) if lengths else 1
        order = sorted(range(len(lengths)), key=lambda i: -lengths[i])
        bins: list[list[int]] = []
        bin_tokens: list[int] = []
        for i in order:
            Li = lengths[i]
            placed = False
            for b in range(len(bins)):
                if bin_tokens[b] + Li <= budget:
                    bins[b].append(i)
                    bin_tokens[b] += Li
                    placed = True
                    break
            if not placed:
                bins.append([i])
                bin_tokens.append(Li)
        return bins

    @staticmethod
    def _real_block(mask_row: torch.Tensor) -> tuple[int, int]:
        idx = mask_row.nonzero(as_tuple=False).squeeze(-1)
        if idx.numel() == 0:
            return 0, 0
        return int(idx[0].item()), int(idx.numel())

    # ---- memory-efficient log-prob helpers ----
    def _token_logprob_train(
        self, logits: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """Per-token log-prob with grad. Uses the fused single-buffer CE (optionally
        chunked over the sequence) when ``log_probs_chunk_size>0``; otherwise the
        original ``log_softmax(...).gather(...)`` path (kept for the smoke config)."""
        cs = self._logprob_chunk_size
        if cs and cs > 0:
            outs = []
            for s in range(0, logits.shape[0], cs):
                outs.append(
                    _FusedTokenLogProb.apply(logits[s : s + cs], targets[s : s + cs])
                )
            return torch.cat(outs, dim=0)
        lp = torch.log_softmax(logits, dim=-1)
        return lp.gather(-1, targets.view(-1, 1)).squeeze(-1)

    def _logprob_entropy_nograd(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        want_entropy: bool,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """No-grad per-token log-prob (+ optional entropy), chunked over the
        sequence to bound the ``[chunk, V]`` softmax memory."""
        cs = (
            self._logprob_chunk_size
            if (self._logprob_chunk_size and self._logprob_chunk_size > 0)
            else logits.shape[0]
        )
        cs = max(1, cs)
        lps, ents = [], []
        for s in range(0, logits.shape[0], cs):
            lg = logits[s : s + cs]
            lsm = torch.log_softmax(lg, dim=-1)
            lps.append(lsm.gather(-1, targets[s : s + cs].view(-1, 1)).squeeze(-1))
            if want_entropy:
                ents.append(-(lsm.exp() * lsm).sum(-1))
        lp = torch.cat(lps, dim=0)
        ent = torch.cat(ents, dim=0) if want_entropy else None
        return lp, ent

    def _row_policy_loss(
        self,
        t,
        r,
        start,
        token_lp,
        algo_name,
        cfg_fn,
        bnt,
        dp,
        loss_agg_mode,
        global_batch_size,
        token_ent=None,
        in_loss_is=None,
    ):
        """DAPO/PG loss + PPO-KL metrics for one sequence, given its (grad-carrying)
        per-token log-prob ``token_lp`` [1, Lm]. Returns ``(loss_tensor|None, stats|None)``.
        Shared by the packed and per-row training paths.

        With ``force_on_policy_ratio`` the old log-probs are ``token_lp``
        detached, so this forward also reports what the skipped old-log-prob
        pass would have: NeMo-RL's token_mult_prob_error against the rollout and,
        given the no-grad ``token_ent`` aligned with ``token_lp``, the entropy.
        ``in_loss_is`` (the rollout-correction settings) then forms the
        importance weights from them as well, as NeMo-RL does."""
        Lm = token_lp.shape[-1]
        dev = token_lp.device
        force_on_policy = bool(cfg_fn("force_on_policy_ratio", False))

        def _col(name, shift):
            x = t.get(name)
            if x is None:
                return None
            x = x[r].to(dev)
            s0 = start + (1 if shift else 0)
            return x[s0:].reshape(1, -1)

        old_lp = None if force_on_policy else _col("old_log_probs", shift=False)
        if old_lp is None and not force_on_policy:
            return None, None
        # Trainer masks are normally width S-1 and already aligned to log-prob
        # columns. Preserve compatibility with token-indexed width-S masks.
        resp_mask = _col(
            "response_mask",
            shift=_response_mask_is_token_indexed(t),
        )
        adv_t = t.get("advantages")
        if adv_t is None:
            return None, None
        if adv_t.dim() == 1:
            adv = adv_t[r].to(dev).view(1, 1).expand(1, Lm).float()
        else:
            adv = adv_t[r].to(dev)[start:].reshape(1, -1).float()
        ris = _col("rollout_is_weights", shift=False)
        ref_lp0 = _col("ref_log_probs", shift=False)
        rlp0 = _col("rollout_log_probs", shift=False)

        cand = [token_lp, adv]
        for v in (old_lp, resp_mask, ris, ref_lp0, rlp0):
            if v is not None:
                cand.append(v)
        Le = min(v.shape[-1] for v in cand)
        token_lp = token_lp[..., :Le]
        old_lp = token_lp.detach() if force_on_policy else old_lp[..., :Le]
        adv = adv[..., :Le]
        mask = resp_mask[..., :Le].float() if resp_mask is not None else None
        ris = ris[..., :Le] if ris is not None else None
        ref_lp = ref_lp0[..., :Le] if ref_lp0 is not None else None
        rlp = rlp0[..., :Le] if rlp0 is not None else None
        if force_on_policy and in_loss_is and rlp is not None:
            ris, _ = rollout_is_weights(
                old_lp - rlp,
                mask if mask is not None else torch.ones_like(old_lp),
                in_loss_is["rollout_is"],
                in_loss_is["rollout_is_threshold"],
                in_loss_is.get("rollout_is_seq_reduction", "sum"),
            )

        if algo_name == AlgorithmName.DAPO.value:
            loss = asymmetric_clip_loss(
                token_lp,
                old_lp,
                adv,
                float(cfg_fn("clip_ratio_low", 0.2)),
                float(cfg_fn("clip_ratio_high", 0.28)),
                mask=mask,
                clip_ratio_c=float(cfg_fn("clip_ratio_c", 0.0)),
                batch_num_tokens=bnt,
                dp_size=dp,
                rollout_is_weights=ris,
            )
        elif algo_name == AlgorithmName.GRPO.value:
            loss = asymmetric_clip_loss(
                token_lp,
                old_lp,
                adv,
                float(cfg_fn("clip_ratio", 0.2)),
                float(cfg_fn("clip_ratio_high", 0.28)),
                mask=mask,
                batch_num_tokens=bnt,
                dp_size=dp,
                loss_agg_mode=loss_agg_mode,
                global_batch_size=global_batch_size,
                rollout_is_weights=ris,
            )
        else:
            loss = policy_gradient_loss(
                token_lp,
                old_lp,
                adv,
                float(cfg_fn("clip_ratio", 0.2)),
                mask=mask,
            )
        kl_c = float(cfg_fn("kl_coeff", 0.0))
        kl_type = str(cfg_fn("kl_penalty", "k3"))
        use_ref_kl = kl_c > 0.0 and ref_lp is not None
        if use_ref_kl:
            loss = loss + kl_c * reference_kl_loss(
                token_lp,
                ref_lp,
                mask=mask,
                kl_type=kl_type,
                loss_agg_mode=loss_agg_mode,
                dp_size=dp,
                batch_num_tokens=bnt,
                global_batch_size=global_batch_size,
            )

        stats = {
            "loss": float(loss.detach()),
            "ppo_kl_sum": 0.0,
            "ppo_kl_tok": 0.0,
            "rc_kl_sum": 0.0,
            "rc_kl_tok": 0.0,
            **dict.fromkeys(_ON_POLICY_STATS, 0.0),
        }
        if mask is not None:
            with torch.no_grad():
                tok = float(mask.sum())
                stats["ppo_kl_sum"] = float(((old_lp - token_lp) * mask).sum())
                stats["ppo_kl_tok"] = tok
                if use_ref_kl:
                    kl_tok = reference_kl_tokens(token_lp.detach(), ref_lp, kl_type)
                    stats["refkl_sum"] = float((kl_tok * mask).sum())
                    stats["refkl_tok"] = tok
                if force_on_policy and ris is not None:
                    stats["ris_sum"] = float((ris * mask).sum())
                    stats["ris_tok"] = tok
                if rlp is not None:
                    stats["rc_kl_sum"] = float(((rlp - token_lp) * mask).sum())
                    stats["rc_kl_tok"] = tok
                if force_on_policy and rlp is not None:
                    gap = (rlp.float() - token_lp.float()).abs()
                    stats["tmpe_sum"] = float((torch.exp(gap) * mask).sum())
                    stats["tmpe_tok"] = tok
                if force_on_policy and token_ent is not None:
                    ent = token_ent[..., :Le].to(dev).float()
                    stats["ent_sum"] = float((ent * mask).sum())
                    stats["ent_tok"] = tok
        return loss, stats

    def _row_entropy_nograd(self, logits, targets):
        """No-grad entropy [1, L] for one row's logits, for the metrics that the
        old-log-prob pass reports and ``force_on_policy_ratio`` skips."""
        with torch.no_grad():
            _, ent = self._logprob_entropy_nograd(logits.detach(), targets, True)
        return ent.view(1, -1)

    # ---- frozen reference policy held by the actor ----
    def snapshot_reference_weights(self) -> None:
        """Keep this rank's current weights, on the GPU, as the reference policy.

        Taken before the first update (and before a checkpoint resume), so the
        reference is the initial model. Refuses ``overlap_param_gather`` here,
        at setup, rather than at the first reference forward after a rollout."""
        self._refuse_overlap_param_gather()
        with torch.no_grad():
            self._reference_weights = {
                name: p.detach().clone() for name, p in self.module.named_parameters()
            }

    @contextmanager
    def reference_weights(self):
        """Run the enclosed forwards with the reference weights in the model.

        The weights are exchanged in place, so parameter storage (and the DDP
        buffers it is a view of) never moves, and they are exchanged back on the
        way out, bit for bit."""
        if getattr(self, "_reference_weights", None) is None:
            raise RuntimeError(
                "snapshot_reference_weights() must run before a reference forward."
            )
        self._refuse_overlap_param_gather()
        self._swap_reference_weights()
        try:
            yield
        finally:
            self._swap_reference_weights()

    def _refuse_overlap_param_gather(self) -> None:
        if getattr(self, "_overlap_param_gather", False):
            raise NotImplementedError(
                "a reference forward needs the updated weights gathered first; "
                "overlap_param_gather defers that into the next forward"
            )

    @torch.no_grad()
    def _swap_reference_weights(self) -> None:
        for name, p in self.module.named_parameters():
            ref = self._reference_weights[name]
            cur = p.data.clone()
            p.data.copy_(ref)
            ref.copy_(cur)

    # ---- engine-level compute_log_probs (actor delegates here) ----
    def engine_compute_log_probs(self, batch: DataProto) -> DataProto:
        seqs = batch["input_ids"]
        B, S = seqs.shape
        am = batch.tensors.get("attention_mask")
        if am is None:
            am = torch.ones_like(seqs)
        want_ent = bool(batch.meta.get("calculate_entropy", False))
        temperature = float(batch.meta.get("temperature", 1.0) or 1.0)

        lp_out = torch.zeros(B, S, dtype=torch.float32)
        ent_out = torch.zeros(B, S, dtype=torch.float32) if want_ent else None
        self.module.eval()

        def _emit(r, start, L, seg_logits, ids_row):
            tok_lp, ent = self._logprob_entropy_nograd(
                seg_logits[:-1], ids_row[1:], want_ent
            )  # [L-1]
            lp_out[r, start : start + L - 1] = tok_lp.cpu()
            if want_ent and ent is not None:
                ent_out[r, start : start + L - 1] = ent.cpu()

        with torch.no_grad():
            rows = []
            for r in range(B):
                start, L = self._real_block(am[r])
                if L >= 2:
                    rows.append((r, start, L))
            if self._dynamic_batch:
                budget = (
                    self._max_tokens_per_gpu if self._max_tokens_per_gpu > 0 else 21504
                )
                lengths = [L for (_, _, L) in rows]
                for bin_rows in self._build_bins(lengths, budget):
                    ids_list = [
                        seqs[rows[j][0], rows[j][1] : rows[j][1] + rows[j][2]].to(
                            "cuda"
                        )
                        for j in bin_rows
                    ]
                    logits_packed, offsets = self._forward_logits_packed(
                        ids_list, model=self.module
                    )
                    logits_packed = logits_packed / temperature
                    for k, j in enumerate(bin_rows):
                        r, start, L = rows[j]
                        _emit(
                            r,
                            start,
                            L,
                            logits_packed[offsets[k] : offsets[k + 1]],
                            ids_list[k],
                        )
            else:
                for r, start, L in rows:
                    ids = seqs[r, start : start + L].to("cuda")
                    logits = self._forward_logits(ids) / temperature  # [L,V]
                    _emit(r, start, L, logits, ids)
        tensors = {"log_probs": lp_out, "input_ids": batch["input_ids"]}
        if want_ent:
            tensors["entropy"] = ent_out
        return DataProto(tensors=tensors, meta=dict(batch.meta))

    # ---- engine-level DAPO/GRPO/SFT update (actor delegates here) ----
    def engine_update_policy(self, batch: DataProto) -> dict[str, float]:
        if batch.batch_size == 0:
            return {"loss": 0.0, "lr": self._cur_lr(), "grad_norm": 0.0}
        meta = dict(batch.meta)
        if meta.get("task_type") == "sft":
            return self._engine_update_sft(batch)
        algo_name = str(meta.get("algorithm", "dapo")).lower()
        temperature = float(meta.get("temperature", 1.0) or 1.0)
        bnt = meta.get("batch_num_tokens")
        dp = int(meta.get("dp_size", self.get_data_parallel_size()) or 1)
        algo_cfg_full = meta.get("algo_config", {}) or {}
        _sub = algo_cfg_full.get(algo_name)
        _sub = _sub if isinstance(_sub, dict) else {}

        def _cfg(key, default):
            v = _sub.get(key, algo_cfg_full.get(key, default))
            return default if v is None else v

        t = batch.tensors
        seqs = t["input_ids"]
        am = t.get("attention_mask")
        if am is None:
            am = torch.ones_like(seqs)
        B, _S = seqs.shape
        loss_agg_mode = str(_cfg("loss_agg_mode", "token-mean"))
        global_batch_size = int(meta.get("global_batch_size") or B * dp)

        self.module.train()
        self._ddp.zero_grad_buffer()
        self.optimizer.zero_grad()

        loss_accum = 0.0
        ppo_kl_sum = 0.0
        ppo_kl_tok = 0.0
        rc_kl_sum = 0.0
        rc_kl_tok = 0.0
        n_rows = 0
        force_on_policy = bool(_cfg("force_on_policy_ratio", False))
        on_policy_sums = dict.fromkeys(_ON_POLICY_STATS, 0.0)

        def _accum(stats):
            nonlocal loss_accum, ppo_kl_sum, ppo_kl_tok, rc_kl_sum, rc_kl_tok, n_rows
            loss_accum += stats["loss"]
            n_rows += 1
            ppo_kl_sum += stats["ppo_kl_sum"]
            ppo_kl_tok += stats["ppo_kl_tok"]
            rc_kl_sum += stats["rc_kl_sum"]
            rc_kl_tok += stats["rc_kl_tok"]
            for key in _ON_POLICY_STATS:
                on_policy_sums[key] += stats[key]

        # Collect valid (non-empty) rows.
        rows = []
        for r in range(B):
            start, L = self._real_block(am[r])
            if L >= 2:
                rows.append((r, start, L))

        if self._dynamic_batch:
            # ---- dynamic-batch packing: concat rows into varlen forwards ----
            budget = self._max_tokens_per_gpu if self._max_tokens_per_gpu > 0 else 21504
            lengths = [L for (_, _, L) in rows]
            for bin_rows in self._build_bins(lengths, budget):
                ids_list = [
                    seqs[rows[j][0], rows[j][1] : rows[j][1] + rows[j][2]].to("cuda")
                    for j in bin_rows
                ]
                logits_packed, offsets = self._forward_logits_packed(
                    ids_list, model=self._ddp
                )
                logits_packed = logits_packed / temperature  # [T,V] (grad)
                bin_loss = None
                for k, j in enumerate(bin_rows):
                    r, start, _L = rows[j]
                    seg = logits_packed[offsets[k] : offsets[k + 1]]  # [L,V]
                    token_lp = self._token_logprob_train(
                        seg[:-1], ids_list[k][1:]
                    ).view(1, -1)
                    token_ent = (
                        self._row_entropy_nograd(seg[:-1], ids_list[k][1:])
                        if force_on_policy
                        else None
                    )
                    loss, stats = self._row_policy_loss(
                        t,
                        r,
                        start,
                        token_lp,
                        algo_name,
                        _cfg,
                        bnt,
                        dp,
                        loss_agg_mode,
                        global_batch_size,
                        token_ent=token_ent,
                        in_loss_is=meta.get("in_loss_rollout_is"),
                    )
                    if loss is None:
                        continue
                    bin_loss = loss if bin_loss is None else bin_loss + loss
                    _accum(stats)
                if bin_loss is not None:
                    bin_loss.backward()
        else:
            # ---- per-sequence forward (original path) ----
            for r, start, L in rows:
                ids = seqs[r, start : start + L].to("cuda")
                logits = (
                    self._forward_logits(ids, model=self._ddp) / temperature
                )  # [L,V] (grad)
                token_lp = self._token_logprob_train(logits[:-1], ids[1:]).view(
                    1, -1
                )  # [1,L-1]
                token_ent = (
                    self._row_entropy_nograd(logits[:-1], ids[1:])
                    if force_on_policy
                    else None
                )
                loss, stats = self._row_policy_loss(
                    t,
                    r,
                    start,
                    token_lp,
                    algo_name,
                    _cfg,
                    bnt,
                    dp,
                    loss_agg_mode,
                    global_batch_size,
                    token_ent=token_ent,
                    in_loss_is=meta.get("in_loss_rollout_is"),
                )
                if loss is None:
                    continue
                loss.backward()
                _accum(stats)

        grad_norm = self._optimizer_step()
        lr = self._sched_step()
        metrics = {
            # ``loss_accum`` is the sum over rows, each already globally
            # normalized by agg_loss. See reduce_reported_loss for why this must
            # not be divided by the row count.
            "loss": reduce_reported_loss(loss_accum),
            "lr": lr,
            "grad_norm": grad_norm,
        }
        if ppo_kl_tok > 0:
            metrics["ppo_kl_sum"] = ppo_kl_sum
            metrics["ppo_kl_tok"] = ppo_kl_tok
        if rc_kl_tok > 0:
            metrics["rollout_corr_kl_sum"] = rc_kl_sum
            metrics["rollout_corr_kl_tok"] = rc_kl_tok
        metrics.update(on_policy_metrics(on_policy_sums))
        return metrics

    def _engine_update_sft(self, batch: DataProto) -> dict[str, float]:
        """SFT training update: forward → log_probs → NLL → backward.

        Uses global token-mean normalization with cross-DP all-reduce,
        matching the FSDP2 path's ``sft_loss()`` behavior.
        """
        meta = dict(batch.meta)
        t = batch.tensors
        seqs = t["input_ids"]
        am = t.get("attention_mask")
        if am is None:
            am = torch.ones_like(seqs)
        loss_masks = t["loss_mask"]
        B, _S = seqs.shape
        dp = int(meta.get("dp_size", self.get_data_parallel_size()) or 1)

        self.module.train()
        self._ddp.zero_grad_buffer()
        self.optimizer.zero_grad()

        rows = []
        for r in range(B):
            start, L = self._real_block(am[r])
            if L >= 2:
                rows.append((r, start, L))

        # Global token count for token-mean normalization (cross-DP all-reduce).
        local_tokens = sum(
            float(loss_masks[r, start + 1 : start + L].sum()) for r, start, L in rows
        )
        num_tokens_t = torch.tensor(local_tokens, device="cuda")
        dp_group = self.get_data_parallel_group()
        if dp_group is not None and dist.is_initialized():
            dist.all_reduce(num_tokens_t, group=dp_group)
        global_num_tokens = max(int(num_tokens_t.item()), 1)

        loss_accum = 0.0
        token_accum = local_tokens

        for r, start, L in rows:
            ids = seqs[r, start : start + L].to("cuda")
            logits = self._forward_logits(ids, model=self._ddp)
            token_lp = self._token_logprob_train(logits[:-1], ids[1:])

            mask = loss_masks[r, start + 1 : start + L].to(token_lp.device).float()
            Le = min(token_lp.shape[0], mask.shape[0])
            token_lp = token_lp[:Le]
            mask = mask[:Le]

            if mask.sum() < 1:
                del logits, token_lp
                continue
            loss = -(token_lp * mask).sum() / global_num_tokens * dp
            loss.backward()

            loss_accum += float(-(token_lp.detach() * mask).sum())
            del logits, token_lp, loss

        torch.cuda.empty_cache()
        grad_norm = self._optimizer_step()
        lr = self._sched_step()
        avg_loss = loss_accum / max(token_accum, 1)
        return {
            "loss": avg_loss,
            "sft_loss": avg_loss,
            "num_tokens": token_accum,
            "lr": lr,
            "grad_norm": grad_norm,
        }

    def _nan_scan(self, losses) -> None:
        """Report non-finite grads BEFORE the reduce-and-clip inside optimizer.step().

        ``optimizer.step()`` returns an already-global grad_norm, so a NaN there
        says only "somewhere on some rank". This runs per rank on the local
        gradients right after the backward, which distinguishes: a bad forward
        (non-finite microbatch loss), one bad layer, one bad rank, or everything.
        Opt-in via ATOMRL_NAN_SCAN=1 -- it walks every parameter.
        """
        if os.environ.get("ATOMRL_NAN_SCAN", "0") != "1":
            return
        rank = self._rank()
        bad_loss = []
        for i, l in enumerate(losses or []):
            for k, v in (l.items() if isinstance(l, dict) else [("loss", l)]):
                if torch.is_tensor(v) and not torch.isfinite(v).all():
                    bad_loss.append(f"mb{i}:{k}")
        offenders, n_params, n_with_grad = [], 0, 0
        for name, prm in self.module.named_parameters():
            n_params += 1
            g = getattr(prm, "main_grad", None)
            src = "main_grad"
            if g is None:
                g, src = prm.grad, "grad"
            if g is None:
                continue
            n_with_grad += 1
            if not torch.isfinite(g).all():
                offenders.append(
                    f"{name}[{src}] nan={int(torch.isnan(g).sum())} "
                    f"inf={int(torch.isinf(g).sum())}/{g.numel()}"
                )
        print(
            f"[NAN_SCAN rank={rank}] params={n_params} with_grad={n_with_grad} "
            f"bad_microbatch_loss={bad_loss or 'none'} "
            f"non_finite_grads={len(offenders)}"
            + (
                "".join(
                    "\n  [NAN_SCAN rank=" + str(rank) + "] " + o for o in offenders[:8]
                )
                if offenders
                else ""
            ),
            flush=True,
        )

    def _optimizer_step(self) -> float:
        """Reduce grads across DP (+reduce-scatter for the distributed optimizer),
        then step the Megatron distributed optimizer."""
        self._nan_scan(None)
        from megatron.core.distributed import finalize_model_grads

        finalize_model_grads([self._ddp])
        update_successful, grad_norm, _num_zeros = self.optimizer.step()
        if not update_successful:
            logger.warning("optimizer.step reported update_successful=False")
        return float(grad_norm) if grad_norm is not None else 0.0

    def _cur_lr(self) -> float:
        try:
            return float(self.optimizer.param_groups[0]["lr"])
        except (TypeError, ValueError):
            return 0.0

    def _sched_step(self) -> float:
        if self.lr_scheduler is not None:
            self.lr_scheduler.step(increment=1)
        return self._cur_lr()

    def lr_scheduler_step(self) -> float:
        return self._cur_lr()
