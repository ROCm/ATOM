# ATOM Environment Variables

This document describes the environment variables used in the ATOM project.

---

## Data Parallelism

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| **ATOM_DP_RANK** | int | 0 | The rank ID for the current process in data parallelism. |
| **ATOM_DP_RANK_LOCAL** | int | 0 | The local rank ID for the current process (used in SPMD mode). |
| **ATOM_DP_SIZE** | int | 1 | Total number of data parallel ranks. |
| **ATOM_DP_MASTER_IP** | str | 127.0.0.1 | Master IP address for DP ranks coordination. |
| **ATOM_DP_MASTER_PORT** | int | 29500 | Master port for DP ranks coordination. |

---

## Execution Mode

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| **ATOM_ENFORCE_EAGER** | bool | 0 (false) | If set to `1`, use eager mode instead of CUDA Graph for kernel launch. |

---

## Kernel / Backend Selection

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| **ATOM_USE_TRITON_GEMM** | bool | 0 (false) | If set to `1`, use AITER Triton FP4 weight preshuffled GEMM. Otherwise use AITER ASM FP4 weight preshuffled GEMM. |
| **ATOM_USE_TRITON_MXFP4_BMM** | bool | 0 (false) | If set to `1`, use FP4 BMM in MLA attention module. |

---

## Fusion Passes

### TP AllReduce Fusion

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| **ATOM_ENABLE_ALLREDUCE_RMSNORM_FUSION** | bool | 1 (true) | If set to `1`, fuse allreduce with RMSNorm in tensor parallel mode. |

### DeepSeek-style

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| **ATOM_ENABLE_DS_INPUT_RMSNORM_QUANT_FUSION** | bool | 1 (true) | If set to `1`, fuse RMSNorm with quantization. |
| **ATOM_ENABLE_DS_QKNORM_QUANT_FUSION** | bool | 1 (true) | If set to `1`, fuse QK norm with quantization in MLA attention module. |

### Qwen3-MoE style

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| **ATOM_ENABLE_QK_NORM_ROPE_CACHE_QUANT_FUSION** | bool | 0 (false) | If set to `1`, fuse QK norm, RoPE, and cache quantization into one kernel. **Enable this for Qwen3-MoE models for better performance.** |

### Llama-style

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| **ATOM_LLAMA_ENABLE_AITER_TRITON_FUSED_RMSNORM_QUANT** | bool | 1 (true) | If set to `1`, use Triton kernel to fuse RMSNorm with quantization. |
| **ATOM_LLAMA_ENABLE_AITER_TRITON_FUSED_SILU_MUL_QUANT** | bool | 1 (true) | If set to `1`, use Triton kernel to fuse SiLU and mul with quantization in MLP module. |

---

## Profiling & Debugging

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| **ATOM_TORCH_PROFILER_DIR** | str | — | When set, enables PyTorch profiler and writes traces to this directory. Create subdirectories per rank (e.g., `rank_0`, `dp0_tp0`). |
| **ATOM_PROFILER_MORE** | bool | 0 (false) | When `ATOM_TORCH_PROFILER_DIR` is set and this is `1`, enables detailed profiling: `record_shapes`, `with_stack`, and `profile_memory`. |
| **ATOM_LOG_MORE** | bool | 0 (false) | If set to `1`, use verbose logging format (includes process name, PID, path, line number, function name). |

---

## Benchmarks (Optional)

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| **OPENAI_API_KEY** | str | — | API key for OpenAI-compatible benchmark requests. |
| **VLLM_USE_MODELSCOPE** | bool | false | If set to `true`, use ModelScope for model downloads in benchmarks. |
| **SAVE_TO_PYTORCH_BENCHMARK_FORMAT** | bool | false | If set, save benchmark results in PyTorch benchmark format. |

---

## Internal / Set by ATOM

The following variables are set internally by ATOM; users typically do not need to configure them:

| Variable | Description |
|----------|-------------|
| **AITER_QUICK_REDUCE_QUANTIZATION** | Set to `INT4` for Llama models with bf16/fp16. |
| **TORCHINDUCTOR_CACHE_DIR** | Set by compiler interface for inductor cache. |
| **TRITON_CACHE_DIR** | Set by compiler interface for Triton cache. |

---

## Reference

Environment variables are defined and accessed via `atom.utils.envs`:

```python
from atom.utils import envs

# Example: check data parallel size
dp_size = envs.ATOM_DP_SIZE
```

See `atom/utils/envs.py` for the full list of lazy-evaluated environment variables.
