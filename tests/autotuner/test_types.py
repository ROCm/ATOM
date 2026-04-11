"""Tests for autotuner core types."""

import tempfile
from pathlib import Path

from atom.autotuner.types import (
    BenchmarkResult,
    Experiment,
    ExperimentStatus,
    GPUInfo,
    InferenceConfig,
    KernelConfig,
    KernelType,
    TunerState,
)


class TestKernelConfig:
    def test_fingerprint_deterministic(self):
        cfg = KernelConfig(KernelType.GEMM, {"m": 1024, "n": 4096, "k": 4096, "dtype": "fp8"})
        assert cfg.fingerprint() == cfg.fingerprint()

    def test_fingerprint_different_for_different_params(self):
        c1 = KernelConfig(KernelType.GEMM, {"m": 1024, "n": 4096, "k": 4096})
        c2 = KernelConfig(KernelType.GEMM, {"m": 2048, "n": 4096, "k": 4096})
        assert c1.fingerprint() != c2.fingerprint()


class TestGPUInfo:
    def test_mi355x_factory(self):
        gpu = GPUInfo.mi355x(num_gpus=8)
        assert gpu.name == "mi355x"
        assert gpu.num_gpus == 8
        assert gpu.memory_gb == 288.0
        assert gpu.peak_tflops_fp8 > gpu.peak_tflops_fp16

    def test_mi300x_factory(self):
        gpu = GPUInfo.mi300x(num_gpus=4)
        assert gpu.name == "mi300x"
        assert gpu.num_gpus == 4
        assert gpu.memory_gb == 192.0


class TestInferenceConfig:
    def test_total_gpus_aggregated(self):
        cfg = InferenceConfig(model="test", tp=4, pp=2, dp=1)
        assert cfg.total_gpus_used() == 8

    def test_total_gpus_disaggregated(self):
        cfg = InferenceConfig(
            model="test", tp=2, pp=1, disagg=True,
            prefill_workers=2, decode_workers=3,
        )
        assert cfg.total_gpus_used() == 10  # (2+3) * 2

    def test_fingerprint_unique(self):
        c1 = InferenceConfig(model="a", tp=4, batch_size=32)
        c2 = InferenceConfig(model="a", tp=4, batch_size=64)
        assert c1.fingerprint() != c2.fingerprint()


class TestExperiment:
    def test_is_better_than_none(self):
        exp = Experiment(
            config=InferenceConfig(model="test"),
            result=BenchmarkResult(
                config=InferenceConfig(model="test"),
                throughput_per_gpu=100.0,
            ),
            status=ExperimentStatus.COMPLETED,
        )
        assert exp.is_better_than(None)

    def test_is_better_than_worse(self):
        cfg = InferenceConfig(model="test")
        e1 = Experiment(
            config=cfg,
            result=BenchmarkResult(config=cfg, throughput_per_gpu=200.0),
        )
        e2 = Experiment(
            config=cfg,
            result=BenchmarkResult(config=cfg, throughput_per_gpu=100.0),
        )
        assert e1.is_better_than(e2)
        assert not e2.is_better_than(e1)


class TestTunerState:
    def test_save_and_load(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "state.json"
            state = TunerState(model="test-model", system="mi355x")
            state.save(path)

            loaded = TunerState.load(path)
            assert loaded.model == "test-model"
            assert loaded.system == "mi355x"
            assert loaded.session_id == state.session_id
