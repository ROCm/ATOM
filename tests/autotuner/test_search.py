"""Tests for configuration search and Pareto analysis."""

from atom.autotuner.types import (
    BenchmarkResult,
    GPUInfo,
    InferenceConfig,
)
from atom.autotuner.database.estimator import ModelArch
from atom.autotuner.search.space import ConfigSpace, SearchBounds
from atom.autotuner.search.pareto import ParetoAnalyzer
from atom.autotuner.search.strategies import GridSearch, AgentGuidedSearch


class TestConfigSpace:
    def test_basic_enumeration(self):
        arch = ModelArch.llama_70b()
        gpu = GPUInfo.mi355x(num_gpus=8)
        bounds = SearchBounds(
            tp_values=[4, 8],
            pp_values=[1],
            batch_sizes=[32],
            kv_cache_dtypes=["fp8"],
            quant_formats=["fp8"],
            disagg_modes=[False],
        )
        space = ConfigSpace(arch, gpu, total_gpus=8, bounds=bounds)
        configs = list(space.enumerate())
        assert len(configs) > 0
        for cfg in configs:
            assert cfg.tp in [4, 8]
            assert cfg.pp == 1

    def test_pruning_invalid_tp(self):
        arch = ModelArch("test", 32, 4096, 32, 8, 128, 11008, 32000)
        gpu = GPUInfo.mi355x(num_gpus=8)
        bounds = SearchBounds(
            tp_values=[3],  # 32 heads not divisible by 3
            pp_values=[1],
            batch_sizes=[32],
            kv_cache_dtypes=["fp8"],
            quant_formats=["fp8"],
            disagg_modes=[False],
        )
        space = ConfigSpace(arch, gpu, total_gpus=8, bounds=bounds)
        configs = list(space.enumerate())
        assert len(configs) == 0

    def test_disagg_enumeration(self):
        arch = ModelArch.llama_70b()
        gpu = GPUInfo.mi355x(num_gpus=8)
        bounds = SearchBounds(
            tp_values=[2],
            pp_values=[1],
            batch_sizes=[32],
            kv_cache_dtypes=["fp8"],
            quant_formats=["fp8"],
            disagg_modes=[True],
            prefill_worker_counts=[1, 2],
            decode_worker_counts=[1, 2],
        )
        space = ConfigSpace(arch, gpu, total_gpus=8, bounds=bounds)
        configs = list(space.enumerate())
        assert all(c.disagg for c in configs)
        assert len(configs) > 0

    def test_moe_has_ep(self):
        arch = ModelArch.deepseek_v3()
        gpu = GPUInfo.mi355x(num_gpus=8)
        bounds = SearchBounds(
            tp_values=[8],
            pp_values=[1],
            batch_sizes=[32],
            kv_cache_dtypes=["fp8"],
            quant_formats=["fp8"],
            disagg_modes=[False],
        )
        space = ConfigSpace(arch, gpu, total_gpus=8, bounds=bounds)
        configs = list(space.enumerate())
        assert all(c.ep >= 1 for c in configs)


class TestParetoAnalyzer:
    def test_simple_frontier(self):
        pa = ParetoAnalyzer()
        cfg = InferenceConfig(model="test")

        pa.add_result(BenchmarkResult(
            config=cfg, throughput_per_gpu=100, throughput_per_user=50,
            ttft_ms=100, tpot_ms=20,
        ))
        pa.add_result(BenchmarkResult(
            config=cfg, throughput_per_gpu=50, throughput_per_user=100,
            ttft_ms=50, tpot_ms=10,
        ))
        pa.add_result(BenchmarkResult(
            config=cfg, throughput_per_gpu=30, throughput_per_user=30,
            ttft_ms=200, tpot_ms=30,
        ))

        frontier = pa.compute_frontier()
        assert len(frontier) == 2  # dominated point excluded
        fps = {(p.throughput_per_gpu, p.throughput_per_user) for p in frontier}
        assert (100, 50) in fps
        assert (50, 100) in fps

    def test_sla_filtering(self):
        pa = ParetoAnalyzer(ttft_limit_ms=150)
        cfg = InferenceConfig(model="test")

        pa.add_result(BenchmarkResult(
            config=cfg, throughput_per_gpu=100, throughput_per_user=50,
            ttft_ms=100, tpot_ms=20,
        ))
        pa.add_result(BenchmarkResult(
            config=cfg, throughput_per_gpu=200, throughput_per_user=80,
            ttft_ms=300, tpot_ms=10,  # exceeds TTFT limit
        ))

        frontier = pa.compute_frontier()
        assert len(frontier) == 1
        assert frontier[0].ttft_ms == 100

    def test_format_frontier(self):
        pa = ParetoAnalyzer()
        cfg = InferenceConfig(model="test", tp=4, pp=1, batch_size=32, quant_format="fp8")
        pa.add_result(BenchmarkResult(
            config=cfg, throughput_per_gpu=100, throughput_per_user=50,
            ttft_ms=100, tpot_ms=20,
        ))
        output = pa.format_frontier()
        assert "100.00" in output

    def test_ascii_chart(self):
        pa = ParetoAnalyzer()
        cfg = InferenceConfig(model="test")
        for i in range(10):
            pa.add_result(BenchmarkResult(
                config=cfg,
                throughput_per_gpu=100 + i * 10,
                throughput_per_user=50 - i * 3,
                ttft_ms=100, tpot_ms=20,
            ))
        chart = pa.format_ascii_chart()
        assert "tokens/s" in chart


class TestGridSearch:
    def test_basic_search(self):
        arch = ModelArch.qwen3_32b()
        gpu = GPUInfo.mi355x(num_gpus=8)
        bounds = SearchBounds(
            tp_values=[4, 8],
            pp_values=[1],
            batch_sizes=[32, 64],
            kv_cache_dtypes=["fp8"],
            quant_formats=["fp8"],
            disagg_modes=[False],
        )
        space = ConfigSpace(arch, gpu, total_gpus=8, bounds=bounds)

        def dummy_eval(config):
            return BenchmarkResult(
                config=config,
                throughput_per_gpu=100.0 / config.tp * config.batch_size,
                throughput_per_user=50.0,
                ttft_ms=100.0,
                tpot_ms=10.0,
            )

        gs = GridSearch()
        results = gs.search(space, dummy_eval, budget=100)
        assert len(results) > 0
        assert all(r.throughput_per_gpu > 0 for r in results)


class TestAgentGuidedSearch:
    def test_basic_search(self):
        arch = ModelArch.llama_70b()
        gpu = GPUInfo.mi355x(num_gpus=8)
        bounds = SearchBounds(
            tp_values=[4, 8],
            pp_values=[1, 2],
            batch_sizes=[16, 32, 64, 128],
            kv_cache_dtypes=["fp8"],
            quant_formats=["fp8"],
            disagg_modes=[False],
        )
        space = ConfigSpace(arch, gpu, total_gpus=8, bounds=bounds)

        call_count = 0

        def eval_fn(config):
            nonlocal call_count
            call_count += 1
            score = config.batch_size * 10 / config.tp
            return BenchmarkResult(
                config=config,
                throughput_per_gpu=score,
                throughput_per_user=1000 / max(config.batch_size, 1),
                ttft_ms=100.0,
                tpot_ms=10.0,
            )

        ags = AgentGuidedSearch(seed=42)
        results = ags.search(space, eval_fn, budget=20)
        assert len(results) > 0
        assert call_count >= 2
