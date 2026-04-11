"""Core data types for the ROCm autotuner."""

from __future__ import annotations

import hashlib
import json
import time
import uuid
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Any, Optional


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class KernelType(Enum):
    GEMM = "gemm"
    ATTENTION = "attention"
    MOE = "moe"
    COMMUNICATION = "communication"
    ELEMENTWISE = "elementwise"
    EMBEDDING = "embedding"
    LAYERNORM = "layernorm"


class QuantFormat(Enum):
    FP16 = "fp16"
    BF16 = "bf16"
    FP8 = "fp8"
    FP8_BLOCK = "fp8_block"
    INT8 = "int8"
    INT4 = "int4"


class SearchStrategy(Enum):
    GRID = "grid"
    BAYESIAN = "bayesian"
    AGENT_GUIDED = "agent_guided"
    EVOLUTIONARY = "evolutionary"


class DatabaseMode(Enum):
    SILICON = "silicon"
    HYBRID = "hybrid"
    EMPIRICAL = "empirical"
    SOL = "sol"


class ExperimentStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    DISCARDED = "discarded"


# ---------------------------------------------------------------------------
# Kernel-level types
# ---------------------------------------------------------------------------

@dataclass
class KernelConfig:
    """Describes a single kernel invocation's parameters."""
    kernel_type: KernelType
    params: dict[str, Any]

    def fingerprint(self) -> str:
        blob = json.dumps(
            {"type": self.kernel_type.value, **self.params}, sort_keys=True
        )
        return hashlib.sha256(blob.encode()).hexdigest()[:16]


@dataclass
class KernelBenchResult:
    """Result of a single kernel micro-benchmark."""
    config: KernelConfig
    latency_us: float
    throughput_tflops: float = 0.0
    memory_bw_gbps: float = 0.0
    power_watts: float = 0.0
    gpu_util_pct: float = 0.0
    timestamp: float = field(default_factory=time.time)


# ---------------------------------------------------------------------------
# System-level types
# ---------------------------------------------------------------------------

@dataclass
class GPUInfo:
    """Hardware descriptor for the target GPU system."""
    name: str                      # e.g. "mi355x"
    compute_units: int = 0
    memory_gb: float = 0.0
    memory_bw_gbps: float = 0.0
    peak_tflops_fp16: float = 0.0
    peak_tflops_fp8: float = 0.0
    interconnect: str = ""         # "xgmi", "pcie"
    interconnect_bw_gbps: float = 0.0
    num_gpus: int = 1
    driver_version: str = ""
    rocm_version: str = ""

    @classmethod
    def mi355x(cls, num_gpus: int = 1) -> GPUInfo:
        return cls(
            name="mi355x",
            compute_units=304,
            memory_gb=288.0,
            memory_bw_gbps=8000.0,
            peak_tflops_fp16=1307.0,
            peak_tflops_fp8=2614.0,
            interconnect="xgmi",
            interconnect_bw_gbps=896.0,
            num_gpus=num_gpus,
        )

    @classmethod
    def mi325x(cls, num_gpus: int = 1) -> GPUInfo:
        return cls(
            name="mi325x",
            compute_units=304,
            memory_gb=256.0,
            memory_bw_gbps=6000.0,
            peak_tflops_fp16=1307.0,
            peak_tflops_fp8=2614.0,
            interconnect="xgmi",
            interconnect_bw_gbps=896.0,
            num_gpus=num_gpus,
        )

    @classmethod
    def mi300x(cls, num_gpus: int = 1) -> GPUInfo:
        return cls(
            name="mi300x",
            compute_units=304,
            memory_gb=192.0,
            memory_bw_gbps=5300.0,
            peak_tflops_fp16=1307.0,
            peak_tflops_fp8=2614.0,
            interconnect="xgmi",
            interconnect_bw_gbps=896.0,
            num_gpus=num_gpus,
        )


# ---------------------------------------------------------------------------
# Inference configuration
# ---------------------------------------------------------------------------

@dataclass
class InferenceConfig:
    """Full inference deployment configuration to be searched/tuned."""
    model: str
    tp: int = 1
    pp: int = 1
    dp: int = 1
    ep: int = 1
    batch_size: int = 1
    max_seq_len: int = 2048
    kv_cache_dtype: str = "fp8"
    quant_format: str = "fp8"
    compilation_level: int = 3
    cudagraph_mode: str = "piecewise"
    attention_backend: str = "aiter"
    enable_prefix_caching: bool = False
    moe_tp: int = 1
    moe_ep: int = 1
    disagg: bool = False
    prefill_workers: int = 1
    decode_workers: int = 1
    isl: int = 4000
    osl: int = 1000

    def total_gpus_used(self) -> int:
        if self.disagg:
            p_gpus = self.prefill_workers * self.tp * self.pp
            d_gpus = self.decode_workers * self.tp * self.pp
            return p_gpus + d_gpus
        return self.tp * self.pp * self.dp

    def fingerprint(self) -> str:
        blob = json.dumps(asdict(self), sort_keys=True)
        return hashlib.sha256(blob.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Benchmark results
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkResult:
    """End-to-end inference benchmark result."""
    config: InferenceConfig
    ttft_ms: float = 0.0
    tpot_ms: float = 0.0
    throughput_tokens_per_sec: float = 0.0
    throughput_per_gpu: float = 0.0
    throughput_per_user: float = 0.0
    request_latency_ms: float = 0.0
    memory_used_gb: float = 0.0
    power_watts: float = 0.0
    timestamp: float = field(default_factory=time.time)


# ---------------------------------------------------------------------------
# Experiment tracking (autoresearch-style)
# ---------------------------------------------------------------------------

@dataclass
class Experiment:
    """One iteration of the autoresearch loop."""
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    config: InferenceConfig = field(default_factory=lambda: InferenceConfig(model=""))
    result: Optional[BenchmarkResult] = None
    parent_id: Optional[str] = None
    mutation: str = ""
    status: ExperimentStatus = ExperimentStatus.PENDING
    created_at: float = field(default_factory=time.time)
    completed_at: Optional[float] = None
    error_message: Optional[str] = None

    def duration_sec(self) -> float:
        if self.completed_at and self.created_at:
            return self.completed_at - self.created_at
        return 0.0

    def is_better_than(self, other: Optional[Experiment]) -> bool:
        if other is None or other.result is None or self.result is None:
            return self.result is not None
        return self.result.throughput_per_gpu > other.result.throughput_per_gpu


# ---------------------------------------------------------------------------
# Pareto frontier
# ---------------------------------------------------------------------------

@dataclass
class ParetoPoint:
    """A point on the throughput-per-gpu vs throughput-per-user Pareto frontier."""
    config: InferenceConfig
    throughput_per_gpu: float
    throughput_per_user: float
    ttft_ms: float
    tpot_ms: float
    request_latency_ms: float = 0.0
    is_frontier: bool = False


# ---------------------------------------------------------------------------
# State snapshot (for crash recovery)
# ---------------------------------------------------------------------------

@dataclass
class TunerState:
    """Serializable snapshot of the full tuner state — allows crash recovery."""
    session_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    model: str = ""
    system: str = ""
    best_experiment: Optional[Experiment] = None
    all_experiments: list[Experiment] = field(default_factory=list)
    pareto_frontier: list[ParetoPoint] = field(default_factory=list)
    start_time: float = field(default_factory=time.time)
    last_checkpoint: float = field(default_factory=time.time)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self._serialize(), indent=2))

    def _serialize(self) -> dict:
        """Best-effort JSON-safe serialization."""
        def _conv(obj: Any) -> Any:
            if isinstance(obj, Enum):
                return obj.value
            if hasattr(obj, "__dataclass_fields__"):
                return {k: _conv(v) for k, v in asdict(obj).items()}
            if isinstance(obj, list):
                return [_conv(x) for x in obj]
            if isinstance(obj, dict):
                return {k: _conv(v) for k, v in obj.items()}
            return obj

        raw = {}
        for k, v in self.__dict__.items():
            raw[k] = _conv(v)
        return raw

    @classmethod
    def load(cls, path: Path) -> TunerState:
        raw = json.loads(path.read_text())
        state = cls()
        state.session_id = raw.get("session_id", state.session_id)
        state.model = raw.get("model", "")
        state.system = raw.get("system", "")
        state.start_time = raw.get("start_time", time.time())
        state.last_checkpoint = raw.get("last_checkpoint", time.time())
        return state
