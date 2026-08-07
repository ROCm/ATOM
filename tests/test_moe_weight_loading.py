# SPDX-License-Identifier: MIT
# Tests for FusedMoE._load_w13 / _load_w2 – verifies correct TP sharding
# when expert_data is padded (as happens with MXFP4 quantised weights).

import sys

import pytest
import torch

# ---------------------------------------------------------------------------
# conftest.py stubs atom and atom.config with minimal placeholders that lack
# the symbols moe.py needs.  Remove those stubs so the real module loads.
# ---------------------------------------------------------------------------
for _k in list(sys.modules):
    if _k.startswith("atom"):
        del sys.modules[_k]

from atom.model_ops.moe import FusedMoE, FusedMoEParallelConfig  # noqa: E402

# ---------------------------------------------------------------------------
# Helper: build a minimal FusedMoE instance with only the attributes that
# _load_w13 / _load_w2 read, skipping the heavy __init__ path.
# ---------------------------------------------------------------------------


def _make_stub(tp_size: int) -> FusedMoE:
    stub = object.__new__(FusedMoE)
    stub.moe_parallel_config = FusedMoEParallelConfig(
        tp_size=tp_size,
        dp_size=1,
        ep_size=1,
        tp_rank=0,
        dp_rank=0,
        ep_rank=0,
        use_ep=False,
        local_ep_size=1,
    )
    return stub


# ===================================================================
# _load_w13  tests
# ===================================================================


class TestLoadW13:
    """Unit tests for FusedMoE._load_w13."""

    @pytest.mark.parametrize(
        "tp_size,tp_rank", [(1, 0), (2, 0), (2, 1), (4, 0), (4, 3), (8, 7)]
    )
    @pytest.mark.parametrize("shard_id", ["w1", "w3"])
    def test_unpadded_weight_correct_copy(self, tp_size, tp_rank, shard_id):
        """When expert_data is NOT padded the loaded shard must land in the
        correct half (w1 -> first, w3 -> second) of expert_data."""
        stub = _make_stub(tp_size)
        dim = 1
        full_size = 256
        shard = full_size // tp_size
        expert_data = torch.zeros(64, 2 * shard, dtype=torch.float32)
        loaded_weight = (
            torch.arange(full_size, dtype=torch.float32)
            .unsqueeze(0)
            .expand(64, -1)
            .contiguous()
        )

        stub._load_w13(expert_data, dim, shard_id, loaded_weight, tp_rank)

        expected_start = shard * tp_rank
        expected = loaded_weight[:, expected_start : expected_start + shard]
        if shard_id == "w1":
            actual = expert_data[:, :shard]
        else:
            actual = expert_data[:, shard:]
        torch.testing.assert_close(actual, expected)

    @pytest.mark.parametrize(
        "tp_size,tp_rank", [(2, 0), (2, 1), (4, 0), (4, 3), (8, 0), (8, 7)]
    )
    @pytest.mark.parametrize("shard_id", ["w1", "w3"])
    def test_padded_expert_data_no_crash(self, tp_size, tp_rank, shard_id):
        """When expert_data is padded (larger than loaded_weight per rank),
        .narrow() must NOT raise RuntimeError."""
        stub = _make_stub(tp_size)
        dim = 1
        full_loaded = 1536
        loaded_shard = full_loaded // tp_size
        pad = 64
        expert_shard = loaded_shard + pad

        expert_data = torch.zeros(64, 2 * expert_shard, dtype=torch.float32)
        loaded_weight = torch.ones(64, full_loaded, dtype=torch.float32)

        stub._load_w13(expert_data, dim, shard_id, loaded_weight, tp_rank)

    @pytest.mark.parametrize("tp_size,tp_rank", [(8, 0), (8, 5), (4, 3)])
    @pytest.mark.parametrize("shard_id", ["w1", "w3"])
    def test_padded_expert_data_correct_values(self, tp_size, tp_rank, shard_id):
        """Padded path: valid region has correct data, padding stays zero."""
        stub = _make_stub(tp_size)
        dim = 1
        full_loaded = 256
        loaded_shard = full_loaded // tp_size
        pad = 32
        expert_shard = loaded_shard + pad

        expert_data = torch.zeros(8, 2 * expert_shard, dtype=torch.float32)
        loaded_weight = (
            torch.arange(full_loaded, dtype=torch.float32)
            .unsqueeze(0)
            .expand(8, -1)
            .contiguous()
        )

        stub._load_w13(expert_data, dim, shard_id, loaded_weight, tp_rank)

        expected_start = loaded_shard * tp_rank
        expected_data = loaded_weight[:, expected_start : expected_start + loaded_shard]

        if shard_id == "w1":
            half = expert_data[:, :expert_shard]
        else:
            half = expert_data[:, expert_shard:]

        valid = half[:, :loaded_shard]
        torch.testing.assert_close(valid, expected_data)

        padding = half[:, loaded_shard:]
        assert padding.abs().max() == 0, "Padding region was overwritten"

    def test_tp1_is_identity(self):
        """TP=1 should copy the entire loaded_weight into the w1 half."""
        stub = _make_stub(1)
        dim = 1
        size = 128
        expert_data = torch.zeros(16, 2 * size, dtype=torch.float32)
        loaded_weight = torch.randn(16, size)

        stub._load_w13(expert_data, dim, "w1", loaded_weight, 0)
        torch.testing.assert_close(expert_data[:, :size], loaded_weight)


# ===================================================================
# _load_w2  tests
# ===================================================================


class TestLoadW2:
    """Unit tests for FusedMoE._load_w2."""

    @pytest.mark.parametrize(
        "tp_size,tp_rank", [(1, 0), (2, 0), (2, 1), (4, 0), (4, 3), (8, 7)]
    )
    def test_unpadded_weight_correct_copy(self, tp_size, tp_rank):
        """Unpadded w2: correct shard is copied into expert_data."""
        stub = _make_stub(tp_size)
        dim = 1
        full_size = 512
        shard = full_size // tp_size

        expert_data = torch.zeros(32, shard, dtype=torch.float32)
        loaded_weight = (
            torch.arange(full_size, dtype=torch.float32)
            .unsqueeze(0)
            .expand(32, -1)
            .contiguous()
        )

        stub._load_w2(expert_data, dim, loaded_weight, tp_rank)

        expected_start = shard * tp_rank
        expected = loaded_weight[:, expected_start : expected_start + shard]
        torch.testing.assert_close(expert_data, expected)

    @pytest.mark.parametrize(
        "tp_size,tp_rank", [(2, 0), (2, 1), (4, 0), (4, 3), (8, 0), (8, 7)]
    )
    def test_padded_expert_data_no_crash(self, tp_size, tp_rank):
        """Padded w2: .narrow() must NOT raise RuntimeError."""
        stub = _make_stub(tp_size)
        dim = 1
        full_loaded = 1536
        loaded_shard = full_loaded // tp_size
        pad = 64
        expert_shard = loaded_shard + pad

        expert_data = torch.zeros(32, expert_shard, dtype=torch.float32)
        loaded_weight = torch.ones(32, full_loaded, dtype=torch.float32)

        stub._load_w2(expert_data, dim, loaded_weight, tp_rank)

    @pytest.mark.parametrize("tp_size,tp_rank", [(8, 0), (8, 5), (4, 3)])
    def test_padded_expert_data_correct_values(self, tp_size, tp_rank):
        """Padded w2: valid region matches, padding stays zero."""
        stub = _make_stub(tp_size)
        dim = 1
        full_loaded = 256
        loaded_shard = full_loaded // tp_size
        pad = 32
        expert_shard = loaded_shard + pad

        expert_data = torch.zeros(8, expert_shard, dtype=torch.float32)
        loaded_weight = (
            torch.arange(full_loaded, dtype=torch.float32)
            .unsqueeze(0)
            .expand(8, -1)
            .contiguous()
        )

        stub._load_w2(expert_data, dim, loaded_weight, tp_rank)

        expected_start = loaded_shard * tp_rank
        expected = loaded_weight[:, expected_start : expected_start + loaded_shard]

        valid = expert_data[:, :loaded_shard]
        torch.testing.assert_close(valid, expected)

        padding = expert_data[:, loaded_shard:]
        assert padding.abs().max() == 0, "Padding region was overwritten"

    def test_load_full_skips_narrow(self):
        """When load_full=True the entire loaded_weight is copied as-is."""
        stub = _make_stub(4)
        dim = 1
        size = 128
        expert_data = torch.zeros(8, size, dtype=torch.float32)
        loaded_weight = torch.randn(8, size)

        stub._load_w2(expert_data, dim, loaded_weight, 0, load_full=True)
        torch.testing.assert_close(expert_data, loaded_weight)

    def test_tp1_is_identity(self):
        """TP=1 should copy the full loaded_weight into expert_data."""
        stub = _make_stub(1)
        dim = 1
        size = 128
        expert_data = torch.zeros(16, size, dtype=torch.float32)
        loaded_weight = torch.randn(16, size)

        stub._load_w2(expert_data, dim, loaded_weight, 0)
        torch.testing.assert_close(expert_data, loaded_weight)


# ===================================================================
# Regression test – exact Qwen3-235B MXFP4 TP=8 scenario
# ===================================================================


class TestQwen3Regression:
    """Reproduces the exact dimensions from the Qwen3-235B MXFP4 TP=8 crash:
    loaded_weight total dim=1536, TP=8 -> loaded_shard=192,
    padded expert_data shard=256 (MXFP4 block-aligned).
    """

    def test_w13_qwen3_tp8(self):
        stub = _make_stub(8)
        dim = 1
        full_loaded = 1536
        padded_shard = 256

        for tp_rank in range(8):
            for shard_id in ("w1", "w3"):
                expert_data = torch.zeros(512, 2 * padded_shard)
                loaded_weight = torch.randn(512, full_loaded)
                # Used to crash: start (1536) + length (256) exceeds dim (1536)
                stub._load_w13(expert_data, dim, shard_id, loaded_weight, tp_rank)

    def test_w2_qwen3_tp8(self):
        stub = _make_stub(8)
        dim = 1
        full_loaded = 1536
        padded_shard = 256

        for tp_rank in range(8):
            expert_data = torch.zeros(128, padded_shard)
            loaded_weight = torch.randn(128, full_loaded)
            stub._load_w2(expert_data, dim, loaded_weight, tp_rank)

    def test_w13_qwen3_tp4(self):
        stub = _make_stub(4)
        dim = 1
        full_loaded = 1536
        padded_shard = 384 + 128

        for tp_rank in range(4):
            for shard_id in ("w1", "w3"):
                expert_data = torch.zeros(512, 2 * padded_shard)
                loaded_weight = torch.randn(512, full_loaded)
                stub._load_w13(expert_data, dim, shard_id, loaded_weight, tp_rank)

    def test_w2_qwen3_tp4(self):
        stub = _make_stub(4)
        dim = 1
        full_loaded = 1536
        padded_shard = 384 + 128

        for tp_rank in range(4):
            expert_data = torch.zeros(128, padded_shard)
            loaded_weight = torch.randn(128, full_loaded)
            stub._load_w2(expert_data, dim, loaded_weight, tp_rank)


# ===================================================================
# Verify the OLD (buggy) logic DOES crash with padded dimensions
# ===================================================================


class TestOldLogicWouldCrash:
    """Confirm that WITHOUT the fix, the original .narrow() raises RuntimeError."""

    def test_old_w13_crashes_when_padded(self):
        """Original code: loaded_weight.narrow(dim, shard_size*tp_rank, shard_size)
        with shard_size from padded expert_data."""
        dim = 1
        full_loaded = 1536
        padded_shard = 256

        expert_data = torch.zeros(64, 2 * padded_shard)
        loaded_weight = torch.randn(64, full_loaded)

        shard_size = expert_data.shape[dim] // 2  # 256 (padded)
        tp_rank = 6  # 256 * 6 = 1536, 1536 + 256 > 1536

        with pytest.raises(RuntimeError, match="start.*length.*exceeds"):
            loaded_weight.narrow(dim, shard_size * tp_rank, shard_size)

    def test_old_w2_crashes_when_padded(self):
        """Same issue for w2 path."""
        dim = 1
        full_loaded = 1536
        padded_shard = 256

        expert_data = torch.zeros(64, padded_shard)
        loaded_weight = torch.randn(64, full_loaded)

        shard_size = expert_data.shape[dim]  # 256 (padded)
        tp_rank = 6

        with pytest.raises(RuntimeError, match="start.*length.*exceeds"):
            loaded_weight.narrow(dim, shard_size * tp_rank, shard_size)
