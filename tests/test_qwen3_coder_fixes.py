# SPDX-License-Identifier: MIT
# Tests for Qwen3-Coder MXFP4 loading fixes:
#   1. fp4x2-aware copy in LinearBase.weight_loader_process
#   2. prefix propagation in Qwen3MoeAttention.__init__

import ast
from pathlib import Path
from unittest.mock import MagicMock

import pytest

ATOM_ROOT = str(Path(__file__).resolve().parent.parent)


# ---------------------------------------------------------------------------
# 1. Tests for weight_loader_process fp4x2 copy fix
# ---------------------------------------------------------------------------
#
# We replicate the exact branch logic from LinearBase.weight_loader_process
# and test it in isolation, avoiding the heavy import chain.


_FP4X2 = "fake_fp4x2_dtype"
_UINT8 = "torch.uint8"


def _weight_loader_process_logic(param_data, loaded_weight, fp4x2_dtype, uint8):
    """Replicate the copy branch from LinearBase.weight_loader_process.

    This mirrors lines 335-338 of atom/model_ops/linear.py exactly.
    """
    if param_data.dtype != fp4x2_dtype:
        param_data.copy_(loaded_weight)
    else:
        param_data.view(uint8).copy_(loaded_weight.view(uint8))


def _make_mock_tensor(dtype):
    t = MagicMock()
    t.dtype = dtype
    t.view = MagicMock(return_value=MagicMock())
    return t


class TestWeightLoaderProcessFp4x2:
    """Verify the fp4x2-aware copy logic in weight_loader_process."""

    def test_non_fp4x2_uses_direct_copy(self):
        """Non-fp4x2 param should use param.data.copy_(loaded_weight)."""
        param_data = _make_mock_tensor(dtype="torch.bfloat16")
        loaded = _make_mock_tensor(dtype="torch.bfloat16")

        _weight_loader_process_logic(param_data, loaded, _FP4X2, _UINT8)

        param_data.copy_.assert_called_once_with(loaded)
        param_data.view.assert_not_called()

    def test_fp4x2_uses_uint8_view_copy(self):
        """fp4x2 param should view both tensors as uint8 before copy."""
        param_data = _make_mock_tensor(dtype=_FP4X2)
        loaded = _make_mock_tensor(dtype=_FP4X2)

        _weight_loader_process_logic(param_data, loaded, _FP4X2, _UINT8)

        # Should NOT use direct copy_
        param_data.copy_.assert_not_called()
        # Both should be viewed as uint8
        param_data.view.assert_called_once_with(_UINT8)
        loaded.view.assert_called_once_with(_UINT8)
        # The uint8 view of param should copy_ from uint8 view of loaded
        param_data.view.return_value.copy_.assert_called_once_with(
            loaded.view.return_value
        )

    def test_non_fp4x2_does_not_view_as_uint8(self):
        """Ensure the uint8 view path is NOT taken for regular dtypes."""
        for dtype in ("torch.bfloat16", "torch.float16", "mock_fp8"):
            param_data = _make_mock_tensor(dtype=dtype)
            loaded = _make_mock_tensor(dtype=dtype)
            _weight_loader_process_logic(param_data, loaded, _FP4X2, _UINT8)
            param_data.view.assert_not_called()
            loaded.view.assert_not_called()

    def test_source_has_fp4x2_branch(self):
        """Verify the actual source code contains the fp4x2 branch."""
        path = Path(ATOM_ROOT) / "atom" / "model_ops" / "linear.py"
        source = path.read_text()
        assert "dtypes.fp4x2" in source, "linear.py should reference dtypes.fp4x2"
        assert (
            ".view(torch.uint8).copy_" in source
        ), "linear.py should have uint8 view copy path"


# ---------------------------------------------------------------------------
# 2. Tests for Qwen3MoeAttention prefix propagation
# ---------------------------------------------------------------------------


class TestQwen3MoeAttentionPrefix:
    """Verify that Qwen3MoeAttention passes prefix to qkv_proj and o_proj."""

    @pytest.fixture(scope="class")
    def source(self):
        path = Path(ATOM_ROOT) / "atom" / "models" / "qwen3_moe.py"
        return path.read_text()

    def test_qkv_proj_receives_prefix(self, source):
        assert 'prefix=f"{prefix}.qkv_proj"' in source

    def test_o_proj_receives_prefix(self, source):
        assert 'prefix=f"{prefix}.o_proj"' in source

    def test_init_has_at_least_two_prefix_kwargs(self, source):
        """Qwen3MoeAttention.__init__ should pass prefix= at least twice."""
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == "Qwen3MoeAttention":
                for item in node.body:
                    if isinstance(item, ast.FunctionDef) and item.name == "__init__":
                        count = sum(
                            1
                            for n in ast.walk(item)
                            if isinstance(n, ast.keyword) and n.arg == "prefix"
                        )
                        assert count >= 2, f"Expected >=2 prefix= kwargs, found {count}"
                        return
        pytest.fail("Qwen3MoeAttention.__init__ not found")
