# SPDX-License-Identifier: MIT
# CPP P2.1 — chunked prefill page alignment.
#
# Verifies that the page-alignment logic rounds middle-chunk sizes down to
# max(block_size, 64) boundaries while leaving final chunks untouched.
# Standalone — no torch/GPU/Scheduler dependencies.


def _prefill_chunk_for_budget(
    num_new_tokens: int,
    budget_remaining: int,
    block_size: int,
    enable_chunked_prefill: bool = True,
    num_batched_tokens: int = 0,
):
    """Mirror of Scheduler._prefill_chunk_for_budget for isolated testing."""
    if enable_chunked_prefill:
        chunk = min(num_new_tokens, budget_remaining)
        if chunk < num_new_tokens:
            align = max(block_size, 64)
            aligned = (chunk // align) * align
            if aligned > 0:
                chunk = aligned
        return chunk
    if num_new_tokens > budget_remaining and num_batched_tokens > 0:
        return None
    return num_new_tokens


class TestChunkPageAlignment:
    def test_middle_chunk_aligned_to_block_size(self):
        """Middle chunk rounds down to max(block_size, 64) boundary."""
        # 200 tokens, budget 100, block_size=16, align=64 -> 64 (100 // 64 * 64)
        chunk = _prefill_chunk_for_budget(200, 100, block_size=16)
        assert chunk == 64

    def test_final_chunk_not_aligned(self):
        """When chunk == num_new_tokens (final chunk), no alignment."""
        # 50 tokens, budget 200 -> chunk = 50 (fits entirely)
        chunk = _prefill_chunk_for_budget(50, 200, block_size=16)
        assert chunk == 50

    def test_alignment_uses_max_of_block_size_and_64(self):
        """Alignment granularity is max(block_size, 64)."""
        # block_size=4, alignment = max(4, 64) = 64
        chunk = _prefill_chunk_for_budget(200, 100, block_size=4)
        assert chunk == 64

    def test_large_block_size(self):
        """block_size > 64 uses block_size as alignment."""
        # 500 tokens, budget 300, block_size=128 -> 256 (300 // 128 * 128)
        chunk = _prefill_chunk_for_budget(500, 300, block_size=128)
        assert chunk == 256

    def test_budget_smaller_than_alignment_no_underflow(self):
        """If budget < alignment granularity, don't round to 0."""
        # 500 tokens, budget 64, block_size=128 -> aligned=0 -> fallback 64
        chunk = _prefill_chunk_for_budget(500, 64, block_size=128)
        assert chunk == 64

    def test_exact_multiple_unchanged(self):
        """Budget that is already an exact multiple stays the same."""
        # align = max(16, 64) = 64; 128 // 64 * 64 = 128
        chunk = _prefill_chunk_for_budget(200, 128, block_size=16)
        assert chunk == 128

    def test_token_conservation(self):
        """Simulated multi-chunk schedule covers all prompt tokens."""
        prompt_len = 250
        budget = 100
        block_size = 16
        align = max(block_size, 64)

        remaining = prompt_len
        chunks = []
        while remaining > 0:
            chunk = _prefill_chunk_for_budget(remaining, budget, block_size)
            chunks.append(chunk)
            remaining -= chunk

        assert sum(chunks) == prompt_len
        for c in chunks[:-1]:
            assert c % align == 0

    def test_chunked_prefill_disabled_unchanged(self):
        """With chunked prefill disabled, alignment is not applied."""
        chunk = _prefill_chunk_for_budget(
            50, 100, block_size=16, enable_chunked_prefill=False
        )
        assert chunk == 50

    def test_budget_equals_remaining(self):
        """When budget exactly matches remaining tokens, treat as final."""
        chunk = _prefill_chunk_for_budget(100, 100, block_size=16)
        assert chunk == 100

    def test_block_size_1(self):
        """block_size=1 uses alignment=64."""
        chunk = _prefill_chunk_for_budget(200, 100, block_size=1)
        assert chunk == 64
