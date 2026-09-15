# SPDX-License-Identifier: MIT
"""Align draft confidence with the next actual verification by request and IDs."""

import torch

from atom.utils.forward_context import get_forward_context


class ConfidenceTrace:
    def __init__(self, runner):
        self.pending, self.records = {}, []
        self._head = runner.drafter.model.head_and_sample
        self._commit = runner.attn_metadata_builder.commit_speculative_state
        runner.drafter.model.head_and_sample = self.head
        runner.attn_metadata_builder.commit_speculative_state = self.commit

    def head(self, *args, **kwargs):
        tokens, confidence = self._head(*args, **kwargs)
        metadata = get_forward_context().attn_metadata
        for i, span in enumerate(metadata.step.requests):
            self.pending[span.request_id] = (
                int(metadata.cache.cursor[span.slot, 0]),
                tokens[i].detach().clone(),
                confidence[i].detach().clone(),
            )
        return tokens, confidence

    def commit(self, metadata, last_indices):
        step = metadata.step
        if step.tentative:
            spec = get_forward_context().spec_decode_metadata
            accepted = (last_indices - step.cu_seqlens_q[:-1]).tolist()
            start = 0
            for span, count in zip(step.requests, accepted):
                position, tokens, confidence = self.pending.pop(span.request_id)
                offered = span.length - 1
                actual = spec.draft_token_ids[start : start + offered]
                assert position == span.position, "Confidence refers to another cursor"
                assert torch.equal(
                    actual, tokens[:offered]
                ), "Confidence/draft ID mismatch"
                self.records.append(
                    {
                        "request_id": span.request_id,
                        "position": position,
                        "verified_drafts": offered,
                        "accepted_drafts": count,
                        "confidence": confidence.tolist(),
                    }
                )
                start += offered
        return self._commit(metadata, last_indices)

    def take(self):
        records, self.records = self.records, []
        # A finished request's final proposal is never verified. Do not turn
        # these unobserved proposals into rejection labels for calibration.
        self.pending.clear()
        return records
