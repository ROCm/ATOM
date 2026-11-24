# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""Isolated unit test for AITER moe_sorting_fwd kernel.

This test bypasses Lumen's wrapper and calls AITER's moe_sorting_fwd
directly, to verify whether observed failures in
TestFusedPermuteCorrectness (test_sorted_weights_match_input and
SIGSEGV in test_permute_unpermute_round_trip) originate from the
AITER HIP kernel itself.

Run in a subprocess to protect the parent pytest session from SIGSEGV:
    pytest tests/ops/test_aiter_moe_sorting_isolation.py -v
"""

import subprocess
import sys
import textwrap

import pytest
import torch


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
class TestAiterMoeSortingIsolation:
    """Each test runs in a subprocess so SIGSEGV does not kill pytest."""

    def _run_script(self, script: str, timeout: int = 60):
        result = subprocess.run(
            [sys.executable, "-c", textwrap.dedent(script)],
            capture_output=True,
            check=False,
            text=True,
            timeout=timeout,
        )
        return result

    def test_kernel_does_not_crash(self):
        """moe_sorting_fwd should not SIGSEGV on basic inputs."""
        result = self._run_script("""
            import torch
            from aiter.ops.moe_sorting import moe_sorting_fwd

            num_tokens, k, num_experts = 16, 2, 4
            block_size = 32
            max_padded = num_tokens * k + num_experts * block_size - k
            max_blocks = (max_padded + block_size - 1) // block_size

            indices = torch.randint(0, num_experts, (num_tokens, k),
                                    device="cuda", dtype=torch.int32)
            weights = torch.softmax(
                torch.randn(num_tokens, k, device="cuda"), dim=-1
            ).float()

            sorted_ids = torch.empty(max_padded, dtype=torch.int32, device="cuda")
            sorted_weights = torch.empty(max_padded, dtype=torch.float32, device="cuda")
            sorted_expert_ids = torch.empty(max_blocks, dtype=torch.int32, device="cuda")
            num_valid_ids = torch.empty(2, dtype=torch.int32, device="cuda")
            moe_buf = torch.empty((num_tokens, 128), dtype=torch.bfloat16, device="cuda")

            moe_sorting_fwd(
                indices, weights,
                sorted_ids, sorted_weights, sorted_expert_ids,
                num_valid_ids, moe_buf,
                num_experts, block_size,
            )
            torch.cuda.synchronize()

            total_valid = num_valid_ids[0].item()
            # AITER returns the *padded* total (num_experts * block_size),
            # not the raw num_tokens * k.  This is expected behaviour.
            print(f"OK: total_valid={total_valid} (padded), "
                  f"raw={num_tokens * k}, "
                  f"max_padded={max_padded}")
            assert total_valid > 0, f"total_valid must be positive, got {total_valid}"
        """)
        if result.returncode == -11:
            pytest.fail(
                "AITER moe_sorting_fwd SIGSEGV — HIP kernel bug confirmed.\n"
                f"stderr: {result.stderr[-500:]}"
            )
        assert (
            result.returncode == 0
        ), f"exit={result.returncode}\nstdout: {result.stdout}\nstderr: {result.stderr[-500:]}"

    def test_sorted_ids_are_not_flat_slot_indices(self):
        """Confirm that AITER sorted_ids are NOT flat (token*k) indices.

        AITER's moe_sorting_fwd encodes sorted_ids in a packed format
        for consumption by downstream AITER GEMM kernels.  They are NOT
        indices into weights.reshape(-1).  This test documents and
        verifies that property so Lumen tests don't re-introduce the
        wrong assumption.
        """
        result = self._run_script("""
            import torch
            from aiter.ops.moe_sorting import moe_sorting_fwd

            num_tokens, k, num_experts = 16, 2, 4
            block_size = 32
            max_padded = num_tokens * k + num_experts * block_size - k
            max_blocks = (max_padded + block_size - 1) // block_size

            indices = torch.randint(0, num_experts, (num_tokens, k),
                                    device="cuda", dtype=torch.int32)
            weights = torch.softmax(
                torch.randn(num_tokens, k, device="cuda"), dim=-1
            ).float()

            sorted_ids = torch.empty(max_padded, dtype=torch.int32, device="cuda")
            sorted_weights = torch.empty(max_padded, dtype=torch.float32, device="cuda")
            sorted_expert_ids = torch.empty(max_blocks, dtype=torch.int32, device="cuda")
            num_valid_ids = torch.empty(2, dtype=torch.int32, device="cuda")
            moe_buf = torch.empty((num_tokens, 128), dtype=torch.bfloat16, device="cuda")

            moe_sorting_fwd(
                indices, weights,
                sorted_ids, sorted_weights, sorted_expert_ids,
                num_valid_ids, moe_buf,
                num_experts, block_size,
            )
            torch.cuda.synchronize()

            total_valid = num_valid_ids[0].item()
            ids_cpu = sorted_ids[:total_valid].long().cpu()
            max_idx = ids_cpu.max().item()
            flat_len = num_tokens * k

            # AITER sorted_ids exceed flat-weight bounds — they are packed
            # token IDs for the AITER GEMM pipeline, not array indices.
            print(f"max_sorted_id={max_idx}, flat_weight_len={flat_len}")
            print(f"total_valid={total_valid} (AITER padded count)")

            assert max_idx >= flat_len, (
                f"Expected sorted_ids to exceed flat bounds "
                f"(AITER packed format), but max_id={max_idx} < {flat_len}"
            )
        """)
        if result.returncode == -11:
            pytest.fail(
                "AITER moe_sorting_fwd SIGSEGV.\n" f"stderr: {result.stderr[-500:]}"
            )
        assert (
            result.returncode == 0
        ), f"exit={result.returncode}\nstdout: {result.stdout}\nstderr: {result.stderr[-500:]}"

    def test_repeated_calls_stability(self):
        """Call moe_sorting_fwd 10 times to check for accumulated state crashes.

        The existing test suite documents that moe_sorting_fwd can SIGABRT
        after many prior invocations. This test checks stability under
        repeated calls in a single process.
        """
        result = self._run_script("""
            import torch
            from aiter.ops.moe_sorting import moe_sorting_fwd

            num_tokens, k, num_experts = 16, 2, 4
            block_size = 32
            max_padded = num_tokens * k + num_experts * block_size - k
            max_blocks = (max_padded + block_size - 1) // block_size

            for iteration in range(10):
                indices = torch.randint(0, num_experts, (num_tokens, k),
                                        device="cuda", dtype=torch.int32)
                weights = torch.softmax(
                    torch.randn(num_tokens, k, device="cuda"), dim=-1
                ).float()

                sorted_ids = torch.empty(max_padded, dtype=torch.int32, device="cuda")
                sorted_weights = torch.empty(max_padded, dtype=torch.float32, device="cuda")
                sorted_expert_ids = torch.empty(max_blocks, dtype=torch.int32, device="cuda")
                num_valid_ids = torch.empty(2, dtype=torch.int32, device="cuda")
                moe_buf = torch.empty((num_tokens, 128), dtype=torch.bfloat16, device="cuda")

                moe_sorting_fwd(
                    indices, weights,
                    sorted_ids, sorted_weights, sorted_expert_ids,
                    num_valid_ids, moe_buf,
                    num_experts, block_size,
                )
                torch.cuda.synchronize()

            print(f"OK: 10 iterations completed without crash")
        """)
        if result.returncode == -11:
            pytest.fail(
                "AITER moe_sorting_fwd SIGSEGV after repeated calls — "
                "accumulated GPU state bug confirmed.\n"
                f"stderr: {result.stderr[-500:]}"
            )
        elif result.returncode == -6:
            pytest.fail(
                "AITER moe_sorting_fwd SIGABRT after repeated calls — "
                "accumulated GPU state bug confirmed.\n"
                f"stderr: {result.stderr[-500:]}"
            )
        assert (
            result.returncode == 0
        ), f"exit={result.returncode}\nstdout: {result.stdout}\nstderr: {result.stderr[-500:]}"
