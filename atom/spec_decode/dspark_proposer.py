import logging

import torch
import torch.nn as nn
from atom.spec_decode.drafter import AuxCaptureSpec, Drafter
from atom.spec_decode.dspark_verify import VerifyScheduler
from atom.utils.forward_context import get_forward_context

logger = logging.getLogger("atom")


class DSparkProposer(Drafter):
    """DSpark block-parallel drafter (sibling of ``EagleProposer``).

    Unlike the serial Eagle/MTP loop (the draft model run ``mtp_k`` times),
    DSpark generates the whole block in a single ``forward_spec`` backbone
    pass; the sequential dependency lives in the lightweight Markov head. The
    verify length defaults to the checkpoint's ``dspark_block_size`` and may be
    driven by a confidence schedule (variable-length, Level B) verification.
    """

    def __init__(self, atom_config, device: torch.device, runner):
        super().__init__(atom_config, device, runner)
        # Confidence-scheduled verification (Level B, variable-length verify) is
        # DSpark-only. The ell (per-request verify length) machinery lives in a
        # reusable VerifyScheduler; propose() feeds it the confidence head and
        # the next step's calc_spec_decode_metadata consumes the ell map.
        self.dspark_confidence_schedule = bool(self.config.dspark.confidence_schedule)
        self._verify_scheduler = (
            VerifyScheduler(runner) if self.dspark_confidence_schedule else None
        )

    def _resolve_mtp_k(self) -> int:
        draft_cfg = self.speculative_config.draft_model_hf_config
        self.dspark_block_size = int(getattr(draft_cfg, "dspark_block_size"))
        # num_speculative_tokens may be unset for DSpark; default to the full
        # block (a static verify length == block size).
        return self.speculative_config.num_speculative_tokens or self.dspark_block_size

    # ---- Drafter capability surface ----
    @property
    def is_block_drafter(self) -> bool:
        return True

    @property
    def uses_confidence_schedule(self) -> bool:
        return self.dspark_confidence_schedule

    @property
    def verify_scheduler(self):
        return self._verify_scheduler

    # ---- aux-hidden-state ownership (declarative; base owns the hook machinery) ----
    def _aux_capture_spec(self, target_model: nn.Module) -> AuxCaptureSpec:
        """DSpark taps the configured target layers and, for each, reduces the
        layer's output mHC residual over the hc axis (mean(dim=1):
        [N, hc, dim] -> [N, dim]). The base registers the forward hooks."""
        draft_cfg = self.speculative_config.draft_model_hf_config
        layer_ids = tuple(
            int(i) for i in getattr(draft_cfg, "dspark_target_layer_ids", ())
        )
        if not layer_ids:
            raise ValueError(
                "DSpark requires dspark_target_layer_ids on the draft config."
            )
        return AuxCaptureSpec(
            layer_ids=layer_ids,
            hidden_size=self.config.hf_config.hidden_size,
            extract=self._extract_mhc_residual,
        )

    @staticmethod
    def _extract_mhc_residual(output, block: nn.Module):
        """output is the HCState returned by Block.forward. Synthesize the
        post-layer mHC residual [N, hc, dim] and reduce over hc -> [N, dim]."""
        residual = getattr(output, "residual", None)
        if residual is None:
            return None
        x_prev = getattr(output, "x_prev", None)
        post = getattr(output, "post_mix", None)
        comb = getattr(output, "comb_mix", None)
        if x_prev is not None and post is not None and comb is not None:
            out_res = block.hc_post(x_prev, residual, post, comb)
        else:
            out_res = residual
        return out_res.mean(dim=1)

    def propose(
        self,
        # [num_tokens] (unused: DSpark seeds from the verified anchor, not the
        # full target token stream)
        target_token_ids: torch.Tensor,
        # [num_tokens]
        target_positions: torch.Tensor,
        # [num_tokens, hidden_size] (unused: DSpark reads aux_hidden_states)
        target_hidden_states: torch.Tensor,
        # [batch] (unused on this path)
        num_reject_tokens: torch.Tensor,
        next_token_ids: torch.Tensor,  # [batch] verified anchor token x0
        last_token_indices: torch.Tensor,  # [batch] flat index of each anchor row
    ) -> torch.Tensor:
        """DSpark block drafting: ONE parallel backbone pass + Markov sampling.

        Unlike the serial Eagle/MTP path (a python loop running the draft model
        mtp_k times), DSpark generates the whole block in a single forward_spec
        call. The sequential dependency lives inside the lightweight Markov head,
        not in repeated heavyweight backbone passes.

        GPU-VERIFY: this path needs an MI3xx run against the reference DSpark to
        confirm (a) the rolling target-KV window is populated correctly across
        prefix-cache hits, and (b) the sampled block matches the reference.
        """
        forward_context = get_forward_context()
        context = forward_context.context
        attn_metadata = forward_context.attn_metadata
        context.is_draft = True
        bs = context.batch_size

        # Drafter-owned aux: our own forward-hook capture buffers, row-aligned to
        # the target hidden states.
        aux_hidden_states = self.aux_for(target_hidden_states)
        if aux_hidden_states is None:
            raise RuntimeError(
                "DSpark requires target auxiliary hidden states from "
                "dspark_target_layer_ids; none were captured."
            )
        # Concatenate the configured target layers -> [num_tokens, dim*L].
        main_hidden_all = torch.cat(aux_hidden_states, dim=-1)

        # Anchor token x0 per request = the just-verified target token, located
        # at last_token_indices in the flat batch.
        anchor_ids = next_token_ids
        anchor_positions = torch.index_select(target_positions, 0, last_token_indices)
        main_hidden = torch.index_select(main_hidden_all, 0, last_token_indices)
        state_slot = getattr(attn_metadata, "state_slot_mapping", None)
        if state_slot is not None:
            cache_indices = state_slot[:bs].to(torch.long)
        else:
            cache_indices = torch.arange(bs, device=anchor_ids.device, dtype=torch.long)

        # Prefill warmup: seed each request's rolling window with the last
        # min(seq_len, window) target tokens BEFORE drafting. Right after
        # prefill the window is otherwise empty (only the anchor would be
        # written), so the first draft block sees almost no target context and
        # rejects early. Writing the prefill tail lifts first-block acceptance
        # to the steady-state level. Decode steps skip this (the ring buffer is
        # already populated from prior steps).
        if context.is_prefill:
            cu_seqlens_q = getattr(attn_metadata, "cu_seqlens_q", None)
            if cu_seqlens_q is not None:
                window = int(self.model.model.mtp[0].window_size)
                seqlens = cu_seqlens_q[1 : bs + 1] - cu_seqlens_q[:bs]
                write_per_batch = int(min(int(seqlens.max().item()), window))
                self.model.precompute_context_kv(
                    main_hidden_all,
                    target_positions,
                    cache_indices,
                    cu_seqlens_q=cu_seqlens_q[: bs + 1],
                    write_per_batch=write_per_batch,
                )

        # Refresh the rolling target-KV window with the new anchor row, then
        # draft the block in a single backbone pass.
        self.model.precompute_context_kv(main_hidden, anchor_positions, cache_indices)
        # Draft width = the verify horizon mtp_k (num_speculative_tokens). This
        # may exceed dspark_block_size (the training default); DSpark weights are
        # draft-width-agnostic so the wider block is drafted in one pass, with
        # positions past block_size RoPE-extrapolated. Capped at the rolling
        # window so [window ++ draft] KV stays bounded.
        window = int(self.model.model.mtp[0].window_size)
        num_draft = min(self.mtp_k, window)
        self._refresh_dp_metadata(forward_context, bs * num_draft)
        draft_token_ids, confidence = self.model.forward_spec(
            anchor_ids,
            main_hidden,
            anchor_positions,
            cache_indices,
            num_draft=num_draft,
        )
        draft_token_ids = draft_token_ids[:, : self.mtp_k]
        # Confidence-scheduled verification. The hardware-aware prefix scheduler
        # consumes the confidence head to pick a per-request verify length
        # ell_r. We compute ell here and stash it; the actual variable-length
        # verification (Level B) is applied downstream by truncating each
        # request's scheduled spec tokens to ell_r, which frees batch capacity
        # instead of the no-op in-block masking of Level A.
        if self.verify_scheduler is not None and confidence is not None:
            self.verify_scheduler.set_last_ell(
                self.verify_scheduler.compute_ell(confidence[:, : self.mtp_k])
            )
        elif self.verify_scheduler is not None:
            self.verify_scheduler.set_last_ell(None)
        return draft_token_ids
