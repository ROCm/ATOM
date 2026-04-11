"""Tests for the kernel collectors (using analytical/SOL mode, no GPU needed)."""

from atom.autotuner.types import GPUInfo, KernelConfig, KernelType
from atom.autotuner.collector.gemm import GEMMCollector
from atom.autotuner.collector.attention import AttentionCollector
from atom.autotuner.collector.communication import CommunicationCollector
from atom.autotuner.collector.moe import MoECollector


class TestGEMMCollector:
    def test_analytical_estimate(self):
        gpu = GPUInfo.mi355x()
        collector = GEMMCollector(gpu, dtypes=["fp16"])
        config = KernelConfig(KernelType.GEMM, {"m": 1024, "n": 4096, "k": 4096, "dtype": "fp16"})
        result = collector._analytical_estimate(config, 1024, 4096, 4096, "fp16")
        assert result.latency_us > 0
        assert result.throughput_tflops > 0

    def test_sweep_configs_generated(self):
        gpu = GPUInfo.mi355x()
        collector = GEMMCollector(gpu, dtypes=["fp16"])
        configs = collector._build_sweep_configs()
        assert len(configs) > 0
        assert all(c.kernel_type == KernelType.GEMM for c in configs)

    def test_small_m_lower_efficiency(self):
        gpu = GPUInfo.mi355x()
        collector = GEMMCollector(gpu)
        small = collector._analytical_estimate(
            KernelConfig(KernelType.GEMM, {"m": 1, "n": 4096, "k": 4096, "dtype": "fp16"}),
            1, 4096, 4096, "fp16",
        )
        large = collector._analytical_estimate(
            KernelConfig(KernelType.GEMM, {"m": 4096, "n": 4096, "k": 4096, "dtype": "fp16"}),
            4096, 4096, 4096, "fp16",
        )
        assert small.throughput_tflops < large.throughput_tflops


class TestAttentionCollector:
    def test_analytical_prefill(self):
        gpu = GPUInfo.mi355x()
        collector = AttentionCollector(gpu)
        config = KernelConfig(KernelType.ATTENTION, {
            "phase": "prefill", "batch_size": 1, "seq_len": 2048,
            "context_len": 2048, "num_q_heads": 32, "num_kv_heads": 8,
            "head_dim": 128, "kv_dtype": "fp16",
        })
        result = collector._analytical_estimate(config)
        assert result.latency_us > 0

    def test_analytical_decode(self):
        gpu = GPUInfo.mi355x()
        collector = AttentionCollector(gpu)
        config = KernelConfig(KernelType.ATTENTION, {
            "phase": "decode", "batch_size": 64, "seq_len": 1,
            "context_len": 4096, "num_q_heads": 32, "num_kv_heads": 8,
            "head_dim": 128, "kv_dtype": "fp8",
        })
        result = collector._analytical_estimate(config)
        assert result.latency_us > 0


class TestCommunicationCollector:
    def test_modeled_allreduce(self):
        gpu = GPUInfo.mi355x(num_gpus=8)
        collector = CommunicationCollector(gpu)
        config = KernelConfig(KernelType.COMMUNICATION, {
            "op": "all_reduce", "tp_size": 8, "message_bytes": 1024 * 1024,
        })
        result = collector._modeled_estimate(config)
        assert result.latency_us > 0

    def test_single_gpu_zero_latency(self):
        gpu = GPUInfo.mi355x(num_gpus=1)
        collector = CommunicationCollector(gpu)
        config = KernelConfig(KernelType.COMMUNICATION, {
            "op": "all_reduce", "tp_size": 1, "message_bytes": 1024,
        })
        result = collector._modeled_estimate(config)
        assert result.latency_us == 0.0


class TestMoECollector:
    def test_analytical_estimate(self):
        gpu = GPUInfo.mi355x()
        collector = MoECollector(gpu)
        config = KernelConfig(KernelType.MOE, {
            "num_tokens": 128, "num_experts": 64, "top_k": 6,
            "hidden_dim": 7168, "intermediate_dim": 2048,
            "dtype": "fp16", "ep_size": 1, "arch": "deepseek-v3",
        })
        result = collector._analytical_estimate(config)
        assert result.latency_us > 0

    def test_sweep_configs_cover_architectures(self):
        gpu = GPUInfo.mi355x()
        collector = MoECollector(gpu, dtypes=["fp16"])
        configs = collector._build_sweep_configs()
        archs = {c.params["arch"] for c in configs}
        assert "deepseek-v3" in archs
        assert "mixtral-8x7b" in archs
