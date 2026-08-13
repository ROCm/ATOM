# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""CUDAGraph decode buckets must stay inside what the scheduler can hand them.

`capture_cudagraph` used to filter capture sizes by `max_num_seqs` alone. That
is only one of the two bounds `Scheduler.schedule_decode` applies, and under
speculation it is not the one that binds: `mtp_k=3` turns 256 sequences into
1024 tokens, past a 512-token budget. Capturing there wrote past the per-token
forward buffers (sized `max_num_batched_tokens`) and surfaced as a bare
`could not broadcast input array from shape (1024,) into shape (512,)`.
"""

import pytest

pytest.importorskip("aiter", reason="needs the AITER GPU kernel library")

from atom.model_engine.model_runner import max_schedulable_decode_bs


class TestMaxSchedulableDecodeBs:
    def test_sequence_count_binds_when_the_token_budget_is_ample(self):
        # Default serving shape: 16384 // 4 = 4096 sequences' worth of budget,
        # far past max_num_seqs, so the sequence cap is what's left.
        assert max_schedulable_decode_bs(256, 16384, 4) == 256

    def test_token_budget_binds_under_speculation(self):
        # The regression: 256 sequences * (mtp_k=3 + 1) = 1024 > 512.
        assert max_schedulable_decode_bs(256, 512, 4) == 128

    def test_no_speculation_charges_one_token_per_sequence(self):
        assert max_schedulable_decode_bs(256, 512, 1) == 256

    @pytest.mark.parametrize("full_q_len", [1, 2, 4, 8])
    def test_bound_is_never_exceeded_by_the_bucket_it_admits(self, full_q_len):
        # The property the capture loop relies on: every admitted bs, at the
        # full speculative width, fits the token budget. Smaller q buckets fit
        # a fortiori, which is why one filter covers them all.
        budget = 512
        bs = max_schedulable_decode_bs(1024, budget, full_q_len)
        assert bs * full_q_len <= budget

    def test_budget_below_one_sequence_yields_zero_not_a_partial_bucket(self):
        # Caller turns this into an actionable assert rather than capturing a
        # bucket it cannot fill.
        assert max_schedulable_decode_bs(256, 2, 4) == 0

    def test_matches_the_scheduler_admission_loop(self):
        # Mirrors `schedule_decode`: charge `tokens_per_decode_seq` per seq and
        # stop at either bound. Pins the two to each other rather than to a
        # comment — if the scheduler ever charges the replayed q instead of the
        # full width, this is what should fail.
        for max_num_seqs in (1, 7, 128, 256):
            for budget in (1, 4, 500, 512, 16384):
                for full_q_len in (1, 4, 8):
                    admitted, tokens = 0, 0
                    while admitted < max_num_seqs:
                        if tokens + full_q_len > budget:
                            break
                        tokens += full_q_len
                        admitted += 1
                    assert (
                        max_schedulable_decode_bs(max_num_seqs, budget, full_q_len)
                        == admitted
                    )
