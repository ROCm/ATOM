"""Unit tests for config classes."""

import pytest
from atom.config import CompilationConfig, CompilationLevel, CUDAGraphMode


class TestCUDAGraphMode:

    def test_mode_values(self):
        assert CUDAGraphMode.NONE.value == 0
        assert CUDAGraphMode.PIECEWISE.value == 1
        assert CUDAGraphMode.FULL.value == 2
        assert CUDAGraphMode.FULL_DECODE_ONLY.value == (2, 0)
        assert CUDAGraphMode.FULL_AND_PIECEWISE.value == (2, 1)

    def test_separate_routine(self):
        assert CUDAGraphMode.NONE.separate_routine() is False
        assert CUDAGraphMode.FULL_DECODE_ONLY.separate_routine() is True

    def test_decode_and_mixed_mode(self):
        assert CUDAGraphMode.FULL_DECODE_ONLY.decode_mode() == CUDAGraphMode.FULL
        assert CUDAGraphMode.FULL_DECODE_ONLY.mixed_mode() == CUDAGraphMode.NONE
        assert CUDAGraphMode.FULL_AND_PIECEWISE.mixed_mode() == CUDAGraphMode.PIECEWISE

    def test_has_full_cudagraphs(self):
        assert CUDAGraphMode.NONE.has_full_cudagraphs() is False
        assert CUDAGraphMode.FULL.has_full_cudagraphs() is True

    def test_requires_piecewise_compilation(self):
        assert CUDAGraphMode.PIECEWISE.requires_piecewise_compilation() is True
        assert CUDAGraphMode.FULL.requires_piecewise_compilation() is False


class TestCompilationConfig:

    def test_default_values(self):
        config = CompilationConfig()
        assert config.level == 0
        assert config.use_cudagraph is True
        assert config.cuda_graph_sizes == [512]

    def test_invalid_level_raises_error(self):
        with pytest.raises(ValueError, match="level must in 0-3"):
            CompilationConfig(level=5)

    def test_compute_hash_consistency(self):
        config = CompilationConfig(level=1)
        assert config.compute_hash() == config.compute_hash()

    def test_set_splitting_ops_for_v1(self):
        config = CompilationConfig(level=CompilationLevel.PIECEWISE)
        config.set_splitting_ops_for_v1()
        assert "aiter.unified_attention_with_output" in config.splitting_ops
