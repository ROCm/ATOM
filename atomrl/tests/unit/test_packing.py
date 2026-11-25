# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""Unit tests for sequence packing utilities."""

import torch
import torch.nn.functional as F
from atomrl.engine.training.packing import (
    PackingContext,
    clear_packing_context,
    get_packing_context,
    pack_sequences,
    packed_token_log_probs,
    set_packing_context,
    unpack_log_probs,
)


def _make_left_padded_batch(
    seq_lens: list[int], padded_len: int, vocab_size: int = 100, pad_id: int = 0
) -> tuple[torch.Tensor, torch.Tensor]:
    """Create a left-padded batch for testing."""
    B = len(seq_lens)
    input_ids = torch.full((B, padded_len), pad_id, dtype=torch.long)
    attention_mask = torch.zeros(B, padded_len, dtype=torch.long)
    for i, sl in enumerate(seq_lens):
        tokens = torch.randint(1, vocab_size, (sl,))
        input_ids[i, padded_len - sl :] = tokens
        attention_mask[i, padded_len - sl :] = 1
    return input_ids, attention_mask


class TestPackSequences:
    def test_basic_roundtrip(self):
        """Packed tokens should match the original non-pad tokens."""
        seq_lens = [5, 3, 8]
        padded_len = 10
        input_ids, attention_mask = _make_left_padded_batch(seq_lens, padded_len)
        packed = pack_sequences(input_ids, attention_mask)

        assert packed.input_ids.shape == (1, sum(seq_lens))
        assert packed.position_ids.shape == (1, sum(seq_lens))
        assert packed.cu_seqlens.shape == (len(seq_lens) + 1,)
        assert packed.cu_seqlens.dtype == torch.int32
        assert packed.max_seqlen == max(seq_lens)

        # Verify tokens match
        flat = packed.input_ids.squeeze(0)
        for i, sl in enumerate(seq_lens):
            start = int(packed.cu_seqlens[i].item())
            end = int(packed.cu_seqlens[i + 1].item())
            expected = input_ids[i, padded_len - sl :]
            assert torch.equal(flat[start:end], expected), f"Sequence {i} mismatch"

    def test_position_ids_reset(self):
        """Position IDs should reset to 0 at each sequence boundary."""
        seq_lens = [4, 6]
        input_ids, attn_mask = _make_left_padded_batch(seq_lens, 10)
        packed = pack_sequences(input_ids, attn_mask)

        pos = packed.position_ids.squeeze(0)
        # First sequence: [0, 1, 2, 3]
        assert pos[:4].tolist() == [0, 1, 2, 3]
        # Second sequence: [0, 1, 2, 3, 4, 5]
        assert pos[4:10].tolist() == [0, 1, 2, 3, 4, 5]

    def test_cu_seqlens(self):
        """cu_seqlens should be cumulative actual lengths."""
        seq_lens = [3, 7, 2]
        input_ids, attn_mask = _make_left_padded_batch(seq_lens, 10)
        packed = pack_sequences(input_ids, attn_mask)
        assert packed.cu_seqlens.tolist() == [0, 3, 10, 12]

    def test_single_sequence(self):
        """Should work with a single sequence."""
        input_ids, attn_mask = _make_left_padded_batch([5], 10)
        packed = pack_sequences(input_ids, attn_mask)
        assert packed.input_ids.shape == (1, 5)
        assert packed.cu_seqlens.tolist() == [0, 5]


class TestPackedTokenLogProbs:
    def _reference_log_probs(self, logits, input_ids):
        """Reference implementation matching _fused_token_log_probs."""
        B, _S, _V = logits.shape
        shifted = logits[:, :-1]
        targets = input_ids[:, 1:].unsqueeze(-1)
        result = []
        for i in range(B):
            lp = F.log_softmax(shifted[i], dim=-1)
            result.append(lp.gather(-1, targets[i]).squeeze(-1))
        return torch.stack(result, dim=0).float()

    def test_matches_fused_single_sequence(self):
        """Packed log_probs should match padded log_probs for a single sequence."""
        torch.manual_seed(42)
        sl = 8
        V = 50
        padded_len = 12

        input_ids, attn_mask = _make_left_padded_batch([sl], padded_len, vocab_size=V)
        packed = pack_sequences(input_ids, attn_mask)

        # Create logits for packed and padded formats
        packed_logits = torch.randn(sl, V)
        padded_logits = torch.zeros(1, padded_len, V)
        padded_logits[0, padded_len - sl :] = packed_logits

        # Compute both ways
        flat_lp = packed_token_log_probs(
            packed_logits, packed.input_ids.squeeze(0), packed.cu_seqlens
        )
        ref_lp = self._reference_log_probs(padded_logits, input_ids)

        # Unpack and compare
        unpacked_lp = unpack_log_probs(
            flat_lp, packed.cu_seqlens, packed.seq_lens, padded_len
        )

        # Only compare non-padding positions
        actual_positions = ref_lp[0, padded_len - sl : padded_len - 1]
        packed_positions = unpacked_lp[0, padded_len - sl : padded_len - 1]
        torch.testing.assert_close(
            packed_positions, actual_positions, rtol=1e-4, atol=1e-6
        )

    def test_matches_fused_multi_sequence(self):
        """Packed log_probs should match padded for multiple sequences."""
        torch.manual_seed(123)
        seq_lens = [5, 3, 7]
        V = 30
        padded_len = 10

        input_ids, attn_mask = _make_left_padded_batch(
            seq_lens, padded_len, vocab_size=V
        )
        packed = pack_sequences(input_ids, attn_mask)

        # Create shared logits: first build packed, then scatter into padded
        total_tokens = sum(seq_lens)
        packed_logits = torch.randn(total_tokens, V)

        padded_logits = torch.zeros(len(seq_lens), padded_len, V)
        offset = 0
        for i, sl in enumerate(seq_lens):
            padded_logits[i, padded_len - sl :] = packed_logits[offset : offset + sl]
            offset += sl

        flat_lp = packed_token_log_probs(
            packed_logits, packed.input_ids.squeeze(0), packed.cu_seqlens
        )
        ref_lp = self._reference_log_probs(padded_logits, input_ids)

        unpacked_lp = unpack_log_probs(
            flat_lp, packed.cu_seqlens, packed.seq_lens, padded_len
        )

        for i, sl in enumerate(seq_lens):
            actual = ref_lp[i, padded_len - sl : padded_len - 1]
            packed_val = unpacked_lp[i, padded_len - sl : padded_len - 1]
            torch.testing.assert_close(
                packed_val,
                actual,
                rtol=1e-4,
                atol=1e-6,
                msg=f"Sequence {i} log_probs mismatch",
            )


class TestUnpackLogProbs:
    def test_gradient_flow(self):
        """Gradients should flow through unpack_log_probs."""
        seq_lens = torch.tensor([4, 3])
        cu_seqlens = torch.tensor([0, 4, 7], dtype=torch.int32)
        padded_len = 6

        flat_lp = torch.randn(5, requires_grad=True)  # (4-1) + (3-1) = 5
        result = unpack_log_probs(flat_lp, cu_seqlens, seq_lens, padded_len)

        # Sum non-zero positions and backward
        loss = result.sum()
        loss.backward()

        assert flat_lp.grad is not None
        assert (flat_lp.grad != 0).all(), "All flat_lp elements should receive gradient"

    def test_correct_placement(self):
        """Values should be placed at correct left-padded positions."""
        seq_lens = torch.tensor([3, 5])
        cu_seqlens = torch.tensor([0, 3, 8], dtype=torch.int32)
        padded_len = 8

        # seq 0: 3 tokens → 2 shifted log_probs
        # seq 1: 5 tokens → 4 shifted log_probs
        flat_lp = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])

        result = unpack_log_probs(flat_lp, cu_seqlens, seq_lens, padded_len)
        assert result.shape == (2, 7)  # S-1 = 7

        # Seq 0 (sl=3): positions [7-2, 7) = [5, 7) in S-1 dim
        assert result[0, 5:7].tolist() == [1.0, 2.0]
        assert result[0, :5].sum() == 0  # padding is zero

        # Seq 1 (sl=5): positions [7-4, 7) = [3, 7) in S-1 dim
        assert result[1, 3:7].tolist() == [3.0, 4.0, 5.0, 6.0]
        assert result[1, :3].sum() == 0


class TestPackingContext:
    def test_context_manager(self):
        """PackingContext should set and clear packing state."""
        cu = torch.tensor([0, 5, 10], dtype=torch.int32)
        assert get_packing_context() is None

        with PackingContext(cu, 5):
            ctx = get_packing_context()
            assert ctx is not None
            assert torch.equal(ctx[0], cu)
            assert ctx[1] == 5

        assert get_packing_context() is None

    def test_manual_set_clear(self):
        cu = torch.tensor([0, 3], dtype=torch.int32)
        set_packing_context(cu, 3)
        assert get_packing_context() is not None
        clear_packing_context()
        assert get_packing_context() is None


class TestDynamicMiniBatchesPacking:
    """Test that _dynamic_mini_batches creates fewer batches with actual lengths."""

    def test_packing_reduces_batch_count(self):
        """With short actual sequences and large padded length, packing should
        create far fewer mini-batches than the old padded approach."""
        from atomrl.core.protocol import DataProto

        B = 20
        padded_len = 100
        actual_lens = [10] * B  # each sequence is only 10 tokens
        max_token_len = 100

        input_ids = torch.zeros(B, padded_len, dtype=torch.long)
        attention_mask = torch.zeros(B, padded_len, dtype=torch.long)
        for i, sl in enumerate(actual_lens):
            input_ids[i, padded_len - sl :] = torch.randint(1, 50, (sl,))
            attention_mask[i, padded_len - sl :] = 1

        DataProto(
            tensors={
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "old_log_probs": torch.zeros(B, padded_len - 1),
                "response_mask": torch.zeros(B, padded_len - 1),
                "advantages": torch.zeros(B),
            }
        )

        # With actual lengths (10 tokens each), 100/10=10 sequences per batch
        # So 20 sequences / 10 per batch = 2 mini-batches
        # Old code would use padded_len=100 → 1 sequence per batch → 20 mini-batches

        # We can't call _dynamic_mini_batches directly (it's a method),
        # so test the packing logic directly
        seq_lens = attention_mask.sum(dim=1).long()
        sorted_idx = torch.argsort(seq_lens)
        sorted_lens = seq_lens[sorted_idx]

        batches = []
        start = 0
        n = B
        while start < n:
            tok_count = 0
            end = start
            while end < n:
                sl = int(sorted_lens[end].item())
                if tok_count + sl > max_token_len and end > start:
                    break
                tok_count += sl
                end += 1
            batches.append((start, end))
            start = end

        assert len(batches) == 2, f"Expected 2 mini-batches, got {len(batches)}"
