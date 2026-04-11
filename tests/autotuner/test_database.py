"""Tests for the performance database layer."""

import tempfile
from pathlib import Path

from atom.autotuner.types import (
    GPUInfo,
    KernelBenchResult,
    KernelConfig,
    KernelType,
    DatabaseMode,
)
from atom.autotuner.database.storage import PerfStorage
from atom.autotuner.database.perf_model import PerformanceModel
from atom.autotuner.database.estimator import E2EEstimator, ModelArch


class TestPerfStorage:
    def setup_method(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.db_path = Path(self._tmp.name) / "test.db"
        self.storage = PerfStorage(self.db_path)

    def teardown_method(self):
        self.storage.close()
        self._tmp.cleanup()

    def test_insert_and_query(self):
        config = KernelConfig(KernelType.GEMM, {"m": 1024, "n": 4096, "k": 4096, "dtype": "fp16"})
        result = KernelBenchResult(config=config, latency_us=42.0, throughput_tflops=100.0)

        self.storage.insert("mi355x", result)
        results = self.storage.query("mi355x", KernelType.GEMM)
        assert len(results) == 1
        assert results[0].latency_us == 42.0

    def test_insert_batch(self):
        results = []
        for m in [128, 256, 512]:
            config = KernelConfig(KernelType.GEMM, {"m": m, "n": 4096, "k": 4096, "dtype": "fp8"})
            results.append(KernelBenchResult(config=config, latency_us=float(m) / 10))

        count = self.storage.insert_batch("mi355x", results)
        assert count == 3
        assert self.storage.count("mi355x") == 3
        assert self.storage.count("mi355x", KernelType.GEMM) == 3

    def test_query_with_filters(self):
        for dtype in ["fp16", "fp8"]:
            config = KernelConfig(KernelType.GEMM, {"m": 1024, "n": 4096, "k": 4096, "dtype": dtype})
            self.storage.insert("mi355x", KernelBenchResult(config=config, latency_us=10.0))

        fp8_results = self.storage.query("mi355x", KernelType.GEMM, dtype="fp8")
        assert len(fp8_results) == 1
        assert fp8_results[0].config.params["dtype"] == "fp8"

    def test_export_import_jsonl(self):
        config = KernelConfig(KernelType.ATTENTION, {"phase": "prefill", "batch_size": 4, "seq_len": 2048})
        self.storage.insert("mi355x", KernelBenchResult(config=config, latency_us=55.0))

        jsonl_path = Path(self._tmp.name) / "export.jsonl"
        self.storage.export_jsonl("mi355x", jsonl_path)

        storage2 = PerfStorage(Path(self._tmp.name) / "test2.db")
        imported = storage2.import_jsonl("mi355x", jsonl_path)
        assert imported == 1
        storage2.close()


class TestPerformanceModel:
    def setup_method(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.db_path = Path(self._tmp.name) / "test.db"
        self.storage = PerfStorage(self.db_path)
        self.gpu = GPUInfo.mi355x()

    def teardown_method(self):
        self.storage.close()
        self._tmp.cleanup()

    def test_sol_mode_no_data(self):
        model = PerformanceModel(self.storage, "mi355x", self.gpu, DatabaseMode.SOL)
        cfg = KernelConfig(KernelType.GEMM, {"m": 1024, "n": 4096, "k": 4096, "dtype": "fp16"})
        latency = model.predict(cfg)
        assert latency > 0

    def test_empirical_mode(self):
        model = PerformanceModel(self.storage, "mi355x", self.gpu, DatabaseMode.EMPIRICAL)
        cfg = KernelConfig(KernelType.GEMM, {"m": 1, "n": 4096, "k": 4096, "dtype": "fp16"})
        latency = model.predict(cfg)
        assert latency > 0

    def test_hybrid_fallback_to_empirical(self):
        model = PerformanceModel(self.storage, "mi355x", self.gpu, DatabaseMode.HYBRID)
        cfg = KernelConfig(KernelType.GEMM, {"m": 512, "n": 8192, "k": 8192, "dtype": "fp8"})
        latency = model.predict(cfg)
        assert latency > 0

    def test_prediction_with_uncertainty(self):
        model = PerformanceModel(self.storage, "mi355x", self.gpu, DatabaseMode.SOL)
        cfg = KernelConfig(KernelType.GEMM, {"m": 4096, "n": 4096, "k": 4096, "dtype": "fp16"})
        latency, uncertainty = model.predict_with_uncertainty(cfg)
        assert latency > 0
        assert uncertainty >= 0


class TestE2EEstimator:
    def setup_method(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.storage = PerfStorage(Path(self._tmp.name) / "test.db")
        self.gpu = GPUInfo.mi355x(num_gpus=8)
        self.perf_model = PerformanceModel(self.storage, "mi355x", self.gpu, DatabaseMode.SOL)
        self.estimator = E2EEstimator(self.perf_model, self.gpu)

    def teardown_method(self):
        self.storage.close()
        self._tmp.cleanup()

    def test_estimate_llama_70b(self):
        from atom.autotuner.types import InferenceConfig

        config = InferenceConfig(
            model="llama-70b", tp=8, pp=1, batch_size=32,
            kv_cache_dtype="fp8", quant_format="fp8",
            isl=4000, osl=1000,
        )
        arch = ModelArch.llama_70b()
        result = self.estimator.estimate(config, arch)

        assert result.ttft_ms > 0
        assert result.tpot_ms > 0
        assert result.throughput_per_gpu > 0
        assert result.throughput_per_user > 0

    def test_estimate_deepseek_v3_moe(self):
        from atom.autotuner.types import InferenceConfig

        config = InferenceConfig(
            model="deepseek-v3", tp=8, pp=1, ep=4, batch_size=64,
            kv_cache_dtype="fp8", quant_format="fp8",
            isl=4000, osl=1000,
        )
        arch = ModelArch.deepseek_v3()
        result = self.estimator.estimate(config, arch)

        assert result.ttft_ms > 0
        assert result.tpot_ms > 0

    def test_disagg_adds_kv_transfer(self):
        from atom.autotuner.types import InferenceConfig

        arch = ModelArch.llama_70b()
        agg_cfg = InferenceConfig(
            model="llama-70b", tp=4, batch_size=32,
            disagg=False, isl=4000, osl=1000,
        )
        disagg_cfg = InferenceConfig(
            model="llama-70b", tp=4, batch_size=32,
            disagg=True, prefill_workers=1, decode_workers=1,
            isl=4000, osl=1000,
        )

        agg_result = self.estimator.estimate(agg_cfg, arch)
        disagg_result = self.estimator.estimate(disagg_cfg, arch)

        assert disagg_result.ttft_ms > agg_result.ttft_ms


class TestModelArch:
    def test_llama_70b(self):
        arch = ModelArch.llama_70b()
        assert arch.num_layers == 80
        assert arch.hidden_dim == 8192
        assert not arch.is_moe

    def test_deepseek_v3(self):
        arch = ModelArch.deepseek_v3()
        assert arch.is_moe
        assert arch.num_experts == 256
        assert arch.top_k == 8

    def test_gpt_oss_120b(self):
        arch = ModelArch.gpt_oss_120b()
        assert arch.num_layers == 96
        assert arch.hidden_dim == 12288
