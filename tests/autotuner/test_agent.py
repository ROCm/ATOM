"""Tests for the agent loop and experiment tracking."""

import tempfile
from pathlib import Path

from atom.autotuner.types import (
    BenchmarkResult,
    DatabaseMode,
    ExperimentStatus,
    GPUInfo,
    InferenceConfig,
)
from atom.autotuner.agent.experiment import ExperimentTracker
from atom.autotuner.agent.loop import AgentLoop, EvalMode, LoopConfig
from atom.autotuner.database.estimator import ModelArch
from atom.autotuner.database.perf_model import PerformanceModel
from atom.autotuner.database.storage import PerfStorage


class TestExperimentTracker:
    def setup_method(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tracker = ExperimentTracker(Path(self._tmp.name))

    def teardown_method(self):
        self._tmp.cleanup()

    def test_create_and_complete(self):
        cfg = InferenceConfig(model="test", tp=4, batch_size=32)
        exp = self.tracker.create(cfg, mutation="initial")
        assert exp.status == ExperimentStatus.PENDING

        self.tracker.start(exp)
        assert exp.status == ExperimentStatus.RUNNING

        result = BenchmarkResult(config=cfg, throughput_per_gpu=100.0)
        self.tracker.complete(exp, result)
        assert exp.status == ExperimentStatus.COMPLETED
        assert self.tracker.best is not None
        assert self.tracker.best.id == exp.id

    def test_best_tracks_improvement(self):
        cfg = InferenceConfig(model="test")

        exp1 = self.tracker.create(cfg)
        self.tracker.start(exp1)
        self.tracker.complete(exp1, BenchmarkResult(config=cfg, throughput_per_gpu=50.0))

        exp2 = self.tracker.create(cfg, parent_id=exp1.id, mutation="increase_bs")
        self.tracker.start(exp2)
        self.tracker.complete(exp2, BenchmarkResult(config=cfg, throughput_per_gpu=100.0))

        assert self.tracker.best.id == exp2.id

    def test_checkpoint_save_load(self):
        cfg = InferenceConfig(model="test-model", tp=8)
        exp = self.tracker.create(cfg)
        self.tracker.start(exp)
        self.tracker.complete(exp, BenchmarkResult(config=cfg, throughput_per_gpu=75.0))

        cp_path = self.tracker.save_checkpoint()
        assert cp_path.exists()

        tracker2 = ExperimentTracker(Path(self._tmp.name))
        loaded = tracker2.load_checkpoint()
        assert loaded == 1
        assert tracker2.completed_count == 1

    def test_summary_format(self):
        cfg = InferenceConfig(model="test", tp=4, batch_size=32, quant_format="fp8", kv_cache_dtype="fp8")
        exp = self.tracker.create(cfg)
        self.tracker.start(exp)
        self.tracker.complete(exp, BenchmarkResult(
            config=cfg, throughput_per_gpu=100.0, throughput_per_user=50.0,
            ttft_ms=100.0, tpot_ms=10.0,
        ))

        summary = self.tracker.format_summary()
        assert "100.00" in summary
        assert "Experiment Summary" in summary


class TestAgentLoop:
    def test_model_only_run(self):
        tmp = tempfile.mkdtemp()
        try:
            gpu = GPUInfo.mi355x(num_gpus=8)
            storage = PerfStorage(Path(tmp) / "perf.db")
            perf_model = PerformanceModel(storage, "mi355x", gpu, DatabaseMode.SOL)

            loop_config = LoopConfig(
                budget_sec=60,
                max_experiments=10,
                eval_mode=EvalMode.MODEL_ONLY,
                strategy="agent_guided",
                log_dir=Path(tmp) / "results",
            )

            loop = AgentLoop(
                model_arch=ModelArch.qwen3_32b(),
                gpu_info=gpu,
                total_gpus=8,
                loop_config=loop_config,
                perf_model=perf_model,
            )

            tracker = loop.run()
            assert tracker.completed_count > 0
            assert tracker.best is not None
            assert tracker.best.result.throughput_per_gpu > 0

            storage.close()
        finally:
            import shutil
            shutil.rmtree(tmp, ignore_errors=True)

    def test_grid_strategy(self):
        tmp = tempfile.mkdtemp()
        try:
            gpu = GPUInfo.mi355x(num_gpus=8)
            storage = PerfStorage(Path(tmp) / "perf.db")
            perf_model = PerformanceModel(storage, "mi355x", gpu, DatabaseMode.SOL)

            loop_config = LoopConfig(
                budget_sec=30,
                max_experiments=5,
                eval_mode=EvalMode.MODEL_ONLY,
                strategy="grid",
                log_dir=Path(tmp) / "results",
            )

            loop = AgentLoop(
                model_arch=ModelArch.llama_70b(),
                gpu_info=gpu,
                total_gpus=8,
                loop_config=loop_config,
                perf_model=perf_model,
            )

            tracker = loop.run()
            assert tracker.completed_count > 0
            storage.close()
        finally:
            import shutil
            shutil.rmtree(tmp, ignore_errors=True)
