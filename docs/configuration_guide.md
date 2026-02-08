# ATOM Configuration Guide

ATOM (AiTer Optimized Model) is AMD's lightweight LLM inference engine built on
[AITER](https://github.com/ROCm/aiter) kernels for ROCm/HIP GPUs. This guide
documents every configuration class, CLI flag, and environment variable that
controls ATOM's runtime behaviour.

---

## Quick Reference

| Config Class | Primary Purpose |
|---|---|
| `Config` | Master dataclass -- model path, memory, TP size, scheduler limits, KV cache, profiler, and references to all sub-configs |
| `CompilationConfig` | Compilation level (0-3), CUDA graph capture sizes, piecewise splitting ops, inductor settings |
| `CompilationLevel` | Integer constants for the four compilation levels |
| `CUDAGraphMode` | Enum controlling how CUDA graphs are captured (none / piecewise / full / hybrid) |
| `QuantizationConfig` | Quantization type, dtype, dynamic flag, method, excluded layers |
| `ParallelConfig` | Data-parallel size, rank, master IP/port |
| `SpeculativeConfig` | Speculative decoding method, draft model, number of speculative tokens |
| `KVCacheConfig` / `KVCacheTensor` | Per-layer KV cache tensor descriptors (k/v caches and scales) |
| `SamplingParams` | Temperature, max tokens, stop strings, ignore-EOS flag |
| `EngineArgs` | CLI argument parser that builds a `Config` for `LLMEngine` |

---

## 1. Master Configuration (`Config`)

Defined in `atom/config.py`. The root dataclass that the engine consumes.

| Field | Type | Default | Description |
|---|---|---|---|
| `model` | `str` | *(required)* | HuggingFace model name or local path |
| `trust_remote_code` | `bool` | `False` | Trust remote code when loading the model from HuggingFace |
| `max_num_batched_tokens` | `int` | `16384` | Maximum number of tokens batched together per scheduler step |
| `scheduler_delay_factor` | `float` | `0.0` | Multiplicative delay (factor x previous prompt latency) before scheduling the next prompt |
| `max_num_seqs` | `int` | `512` | Maximum number of sequences batched together |
| `max_model_len` | `int \| None` | `None` | Maximum context length; defaults to `hf_config.max_position_embeddings` (capped by it when set) |
| `gpu_memory_utilization` | `float` | `0.9` | Fraction of GPU memory available for KV cache and weights (0.0 -- 1.0) |
| `tensor_parallel_size` | `int` | `1` | Number of tensor-parallel GPUs (1 -- 8) |
| `enforce_eager` | `bool` | `False` | Disable compilation and CUDA graphs; run in eager mode |
| `parallel_config` | `ParallelConfig` | `ParallelConfig()` | Data-parallel configuration (see Section 4) |
| `kv_cache_block_size` | `int` | `16` | Block size for paged KV cache; must be a multiple of 16 or exactly 1 |
| `num_kvcache_blocks` | `int` | `-1` | Number of KV cache blocks (`-1` = auto) |
| `kv_cache_dtype` | `str` | `"bf16"` | KV cache data type (`"bf16"` or `"fp8"`) |
| `enable_prefix_caching` | `bool` | `False` | Enable prefix caching to reuse KV blocks across requests sharing the same prefix |
| `port` | `int` | `8006` | Engine internal communication port |
| `torch_profiler_dir` | `str \| None` | `os.getenv("ATOM_TORCH_PROFILER_DIR", None)` | Directory for saving PyTorch profiler traces; creates the directory if it does not exist |
| `compilation_config` | `CompilationConfig` | `CompilationConfig()` | Compilation and CUDA graph settings (see Section 2) |
| `quant_config` | `QuantizationConfig` | `QuantizationConfig()` | Quantization settings; auto-detected from HuggingFace config during `__post_init__` (see Section 3) |
| `asyncio_mode` | `bool` | `False` | Enable asyncio-based engine loop |
| `load_dummy` | `bool` | `False` | Skip loading model weights (for benchmarking / testing) |
| `enable_expert_parallel` | `bool` | `False` | Enable Expert Parallelism for MoE models |
| `master_addr` | `str` | `"127.0.0.1"` | Master address for distributed communication |
| `graph_bs` | `Optional[list[int]]` | `None` | Explicit list of batch sizes for CUDA graph capture; derived from `compilation_config` during init |
| `enable_dp_attention` | `bool` | `False` | Enable data-parallel attention |
| `torch_dtype` | `torch.dtype` | *(computed)* | Inferred from `hf_config.torch_dtype`; falls back to `torch.bfloat16` |
| `speculative_config` | `Optional[SpeculativeConfig]` | `None` | Speculative decoding configuration (see Section 5) |
| `bos_token_id` | `int` | `-1` | Beginning-of-sequence token ID (`-1` = use model default) |
| `eos_token_id` | `int` | `-1` | End-of-sequence token ID (`-1` = use model default) |
| `stop_token_ids` | `list[int]` | `[]` | Additional stop token IDs; populated from `GenerationConfig.eos_token_id` during init |

**Auto-derived fields** (set in `__post_init__`, not user-supplied):

| Field | Type | Description |
|---|---|---|
| `hf_config` | `PretrainedConfig` | Loaded automatically via `get_hf_config(model)` |
| `generation_config` | `GenerationConfig` | Loaded automatically via `get_generation_config(model)` |

---

## 2. Compilation Configuration (`CompilationConfig`)

Defined in `atom/config.py`. Controls torch.compile and CUDA graph behaviour.

### 2.1 Compilation Levels (`CompilationLevel`)

| Constant | Value | Description |
|---|---|---|
| `NO_COMPILATION` | `0` | No compilation -- pure eager execution |
| `DYNAMO_AS_IS` | `1` | Use torch.compile / TorchDynamo as-is |
| `DYNAMO_ONCE` | `2` | TorchDynamo with a single compilation pass |
| `PIECEWISE` | `3` | Piecewise compilation with CUDA graph capture (recommended for production) |

### 2.2 `CompilationConfig` Fields

| Field | Type | Default | Description |
|---|---|---|---|
| `level` | `int` | `0` | Compilation level (see table above); must be 0 -- 3 |
| `use_cudagraph` | `bool` | `True` | Whether to use CUDA graphs |
| `cudagraph_capture_sizes` | `Optional[list[int]]` | `None` | Explicit list of batch sizes for CUDA graph capture; overrides `cuda_graph_sizes` when set |
| `cuda_graph_sizes` | `list[int]` | `[]` (post-init: `[512]`) | CUDA graph sizing strategy: 1 value generates `[1,2,4,8] + range(16, N+1, 16)`; multiple values used as-is; empty defaults to `[512]` |
| `debug_dump_path` | `str` | `""` | Path to dump debug / compilation information |
| `cache_dir` | `str` | `""` | Directory for compilation caches |
| `use_inductor` | `bool` | `True` | Enable TorchInductor backend |
| `cudagraph_mode` | `Optional[CUDAGraphMode]` | `None` | CUDA graph capture mode (see below); set to `PIECEWISE` automatically at level 3 |
| `splitting_ops` | `Optional[list[str]]` | `None` | Ops that split the graph into sub-graphs for piecewise compilation; auto-populated at level 3 with `["aiter.unified_attention_with_output", "aiter.mla_attention"]` |
| `cudagraph_copy_inputs` | `bool` | `False` | Copy input tensors into internally managed buffers before CUDA graph replay; only effective in PIECEWISE mode |
| `compile_sizes` | `Optional[list[Union[int, str]]]` | `None` | Sizes to compile for inductor; accepts integers and the string `"cudagraph_capture_sizes"` |
| `inductor_compile_config` | `dict` | `{}` | Additional configuration passed to the inductor backend |

### 2.3 CUDA Graph Mode (`CUDAGraphMode`)

| Mode | Value | Description |
|---|---|---|
| `NONE` | `0` | No CUDA graph capture |
| `PIECEWISE` | `1` | Piecewise CUDA graphs -- attention ops stay outside the graph for flexibility (default at level 3) |
| `FULL` | `2` | Full CUDA graph capture for all batches; best for small models / short prompts |
| `FULL_DECODE_ONLY` | `(FULL, NONE)` | Full CUDA graphs for decode batches only; mixed prefill-decode runs without graphs (useful in P/D setups) |
| `FULL_AND_PIECEWISE` | `(FULL, PIECEWISE)` | Full graphs for decode, piecewise for prefill/mixed -- most performant mode for most models |

Helper methods on `CUDAGraphMode`:

- `decode_mode()` -- returns the mode used for pure decode batches.
- `mixed_mode()` -- returns the mode used for mixed prefill-decode batches.
- `requires_piecewise_compilation()` -- whether the mode needs piecewise compilation.
- `has_full_cudagraphs()` -- whether the mode includes full CUDA graph capture.
- `separate_routine()` -- whether decode and mixed batches use different routines.

---

## 3. Quantization Configuration (`QuantizationConfig`)

Defined in `atom/config.py`. Extends `dict` so fields are stored and accessed as
dictionary keys (e.g., `config["quant_type"]`).

### 3.1 `QuantizationConfig` Fields

| Field | Type | Default | Description |
|---|---|---|---|
| `quant_type` | `QuantType` | `QuantType.No` | Quantization granularity (see below) |
| `quant_dtype` | `torch.dtype` | `torch.bfloat16` | Data type for quantized weights |
| `is_dynamic` | `bool` | `True` | Use dynamic quantization (scales computed at runtime) |
| `quant_name` | `str` | `""` | Human-readable name for the quantization scheme |
| `quant_method` | `Optional[str]` | `None` | Quantization method from HuggingFace config (e.g., `"compressed-tensors"`, `"quark"`) |
| `exclude_layers` | `Optional[list[str]]` | `[]` | Layer names excluded from quantization |

### 3.2 `QuantType` Values (from AITER)

| Value | Description |
|---|---|
| `QuantType.No` | No quantization |
| `QuantType.per_Token` | Per-token / per-channel quantization |
| `QuantType.per_1x128` | Block quantization with group size 128 |
| `QuantType.per_1x32` | Block quantization with group size 32 |
| `QuantType.per_128x128` | Large 2D block quantization (remapped to `per_1x128` in MoE kernels) |
| `QuantType.per_Tensor` | Per-tensor quantization |

### 3.3 Supported Quantization Dtypes

| Dtype | AITER Key | Notes |
|---|---|---|
| FP8 (E4M3) | `"fp8"` | 8-bit floating point |
| MXFP4 | `"fp4x2"` | Microscaling FP4; forces `QuantType.per_1x32` |
| INT8 | `"i8"` | 8-bit integer |
| INT4 | `"i4x2"` | 4-bit integer (packed) |

### 3.4 Auto-Detection from HuggingFace (`get_quant_config`)

During `Config.__post_init__`, ATOM reads `hf_config.quantization_config` and
automatically determines `quant_type`, `quant_dtype`, and `is_dynamic`:

1. If `quantization_config` is absent, returns `QuantType.No` with `torch_dtype`.
2. If `quant_method == "compressed-tensors"` or channel quantization is detected, sets `per_Token`.
3. If `weight_block_size` or `group_size` is found: group size 128 maps to `per_1x128`, group size 32 maps to `per_1x32`.
4. Otherwise falls back to `per_Tensor`.
5. The dtype is parsed from fields like `dtype`, `weight_dtype`, or `quant_method` looking for `fp8`, `fp4`, `mxfp4`, `int8`, `int4`, or `num_bits`.
6. If `activation_scheme` is `"static"`, `is_dynamic` is set to `False`.
7. Excluded layers are read from the `"ignore"` key (compressed-tensors) or `"exclude"` key (quark).

---

## 4. Parallel Configuration (`ParallelConfig`)

Defined in `atom/config.py`. Controls data parallelism. Environment variables
(Section 8) override defaults when set.

| Field | Type | Default | Description |
|---|---|---|---|
| `data_parallel_size` | `int` | `1` | Number of data-parallel groups; overridden by `ATOM_DP_SIZE` env var |
| `data_parallel_size_local` | `int` | `1` | Number of local data-parallel groups |
| `data_parallel_rank` | `int` | `0` | Rank within the data-parallel group; overridden by `ATOM_DP_RANK` |
| `data_parallel_rank_local` | `Optional[int]` | `None` | Local rank within the data-parallel group (SPMD mode); overridden by `ATOM_DP_RANK_LOCAL` |
| `data_parallel_master_port` | `int` | `29500` | Port used by the data-parallel master for process group initialization |
| `data_parallel_base_port` | `int` | `get_open_port()` | Base port for data-parallel communication (dynamically assigned) |
| `data_parallel_master_ip` | `str` | `"127.0.0.1"` | IP address of the data-parallel master |

**Computed property:**

- `world_size` -- set during init, equals TP x PP.
- `world_size_across_dp` -- `world_size * data_parallel_size`.

---

## 5. Speculative Decoding Configuration (`SpeculativeConfig`)

Defined in `atom/config.py`. Currently only the Multi-Token Prediction (MTP)
method with `num_speculative_tokens=1` is supported.

| Field | Type | Default | Description |
|---|---|---|---|
| `method` | `Optional[str]` | `""` | Speculative decoding method; currently only `"mtp"` is accepted |
| `model` | `Optional[str]` | `None` | Draft model name or path (typically the same as the target model for MTP) |
| `num_speculative_tokens` | `Optional[int]` | `None` | Number of speculative tokens per iteration; **must be `1`** |
| `draft_model_hf_config` | `Optional[PretrainedConfig]` | `None` | HuggingFace config for the draft model; auto-loaded from `model` when `None` |

**Post-init behaviour:**

- Loads `draft_model_hf_config` from `model` if not provided.
- For DeepSeek V3 / MTP models: overrides `model_type` to `"deepseek_mtp"`, sets `n_predict=1` and `num_nextn_predict_layers=1`, and switches architectures to `["DeepSeekMTPModel"]`.
- `Config.__post_init__` raises `ValueError` if `num_speculative_tokens != 1`.

---

## 6. Sampling Parameters (`SamplingParams`)

Defined in `atom/sampling_params.py`. Passed per-request to control generation.

| Field | Type | Default | Description |
|---|---|---|---|
| `temperature` | `float` | `1.0` | Sampling temperature; lower values make output more deterministic |
| `max_tokens` | `int` | `64` | Maximum number of tokens to generate |
| `ignore_eos` | `bool` | `False` | Continue generating past the EOS token |
| `stop_strings` | `Optional[list[str]]` | `None` | List of strings that trigger generation to stop |

---

## 7. CLI Arguments (`EngineArgs`)

Defined in `atom/model_engine/arg_utils.py`. The `EngineArgs` dataclass exposes
all flags via `add_cli_args()` and converts them into a `Config` via
`create_engine()`.

| Flag | Short | Type | Default | Description |
|---|---|---|---|---|
| `--model` | | `str` | `"Qwen/Qwen3-0.6B"` | Model name or path |
| `--trust-remote-code` | | flag | `False` | Trust remote code when loading model |
| `--tensor-parallel-size` | `-tp` | `int` | `1` | Tensor parallel size |
| `--data-parallel-size` | `-dp` | `int` | `1` | Data parallel size |
| `--enforce-eager` | | flag | `False` | Enforce eager mode execution |
| `--enable_prefix_caching` | | flag | `False` | Enable prefix caching |
| `--port` | | `int` | `8006` | Engine internal port |
| `--kv_cache_dtype` | | `str` | `"bf16"` | KV cache dtype; choices: `bf16`, `fp8` |
| `--block-size` | | `int` | `16` | KV cache block size (maps to `kv_cache_block_size`) |
| `--max-model-len` | | `int` | `None` | Maximum model context length; defaults to `hf_config.max_position_embeddings` |
| `--cudagraph-capture-sizes` | | `str` | `"[1,2,4,8,16,32,48,64,128,256]"` | CUDA graph capture sizes as a Python list string |
| `--level` | | `int` | `3` | Compilation level (0 -- 3) |
| `--load_dummy` | | flag | `False` | Skip loading model weights |
| `--enable-expert-parallel` | | flag | `False` | Enable Expert Parallelism (EP MoE) |
| `--torch-profiler-dir` | | `str` | `None` | Directory for torch profiler traces |
| `--enable-dp-attention` | | flag | `False` | Enable DP attention |
| `--method` | | `str` | `None` | Speculative method; choices: `mtp` |
| `--num-speculative-tokens` | | `int` | `1` | Number of speculative tokens per iteration |
| `--max-num-batched-tokens` | | `int` | `16384` | Maximum number of tokens to batch in the async engine |
| `--max-num-seqs` | | `int` | `512` | Maximum number of sequences to batch together |
| `--gpu-memory-utilization` | | `float` | `0.9` | Fraction of GPU memory to use (0.0 -- 1.0) |
| `--scheduler-delay-factor` | | `float` | `0.0` | Delay factor multiplied by previous prompt latency before scheduling next prompt |

**Example:**

```bash
python -m atom.entrypoint \
    --model deepseek-ai/DeepSeek-R1 \
    --tensor-parallel-size 8 \
    --level 3 \
    --cudagraph-capture-sizes "[1,2,4,8,16,32,64,128,256]" \
    --kv_cache_dtype fp8 \
    --gpu-memory-utilization 0.92 \
    --max-num-seqs 256
```

---

## 8. Environment Variables

### 8.1 Variables Registered in `atom/utils/envs.py`

All variables use lazy evaluation. Boolean variables treat `"1"` as `True` and
anything else (including unset) as `False`, unless noted otherwise.

| Variable | Type | Default | Description |
|---|---|---|---|
| `ATOM_DP_RANK` | `int` | `0` | Data-parallel rank of this process |
| `ATOM_DP_RANK_LOCAL` | `int` | `0` | Local data-parallel rank (for SPMD mode) |
| `ATOM_DP_SIZE` | `int` | `1` | Total number of data-parallel groups |
| `ATOM_DP_MASTER_IP` | `str` | `"127.0.0.1"` | IP address of the data-parallel master |
| `ATOM_DP_MASTER_PORT` | `int` | `29500` | Port of the data-parallel master |
| `ATOM_ENFORCE_EAGER` | `bool` | `False` | Force eager mode globally (also set programmatically by `set_current_atom_config`) |
| `ATOM_ENABLE_QK_NORM_ROPE_CACHE_QUANT_FUSION` | `bool` | `False` | Enable QK-norm + RoPE + cache + quant fusion; enable for Qwen3-MoE models |
| `ATOM_USE_TRITON_GEMM` | `bool` | `False` | Use Triton-based GEMM kernels instead of default backends |
| `ATOM_USE_TRITON_MXFP4_BMM` | `bool` | `False` | Use Triton-based MXFP4 batched matrix multiply |
| `ATOM_ENABLE_DS_INPUT_RMSNORM_QUANT_FUSION` | `bool` | `True` | Enable fused input RMSNorm + quantization for DeepSeek models |
| `ATOM_ENABLE_DS_QKNORM_QUANT_FUSION` | `bool` | `True` | Enable fused QK-norm + quantization for DeepSeek models |
| `ATOM_ENABLE_ALLREDUCE_RMSNORM_FUSION` | `bool` | `True` | Enable fused all-reduce + RMSNorm kernel |
| `ATOM_LLAMA_ENABLE_AITER_TRITON_FUSED_RMSNORM_QUANT` | `bool` | `True` | Enable AITER Triton fused RMSNorm + quantization for LLaMA models |
| `ATOM_LLAMA_ENABLE_AITER_TRITON_FUSED_SILU_MUL_QUANT` | `bool` | `True` | Enable AITER Triton fused SiLU + multiply + quantization for LLaMA models |

### 8.2 Additional Environment Variables (Used Outside `envs.py`)

| Variable | Type | Default | Where Used | Description |
|---|---|---|---|---|
| `ATOM_TORCH_PROFILER_DIR` | `str` | `None` | `atom/config.py` (`Config.torch_profiler_dir`) | Directory for PyTorch profiler output; sets the default for `Config.torch_profiler_dir` |
| `ATOM_PROFILER_MORE` | `str` | `"0"` | `atom/model_engine/model_runner.py` | Set to `"1"` to enable detailed profiling (`record_shapes`, `with_stack`, `profile_memory`) |
| `HF_TOKEN` | `str` | `None` | `atom/config.py` (`get_hf_config`) | HuggingFace authentication token for gated model downloads |

---

## 9. Decision Tree -- Choosing a Compilation Level

```
Start
  |
  v
Is this a debugging / development run?
  |-- Yes --> Level 0 (NO_COMPILATION) or --enforce-eager
  |
  v
Do you need torch.compile but no graph splitting?
  |-- Yes, one-shot compile --> Level 2 (DYNAMO_ONCE)
  |-- Yes, keep Dynamo default --> Level 1 (DYNAMO_AS_IS)
  |
  v
Production inference on ROCm/HIP GPU?
  |-- Yes --> Level 3 (PIECEWISE) [default in EngineArgs]
              - Auto-sets CUDAGraphMode.PIECEWISE
              - Auto-populates splitting_ops for attention ops
              - Pair with --cudagraph-capture-sizes for your batch profile
  |
  v
Need maximum decode throughput?
  |-- Yes --> Level 3 + set cudagraph_mode to FULL_AND_PIECEWISE
              (full graphs for decode, piecewise for prefill)
```

**Rules of thumb:**

- **Level 3** is the default for `EngineArgs` and is recommended for most
  production workloads.
- **Level 0** / `--enforce-eager` is useful for debugging, profiling, or when
  CUDA graphs are incompatible with your model.
- Match `--cudagraph-capture-sizes` to your expected batch sizes for optimal
  memory usage and launch latency.
- When using `--enable-dp-attention` or Expert Parallelism (`--enable-expert-parallel`),
  level 3 is still recommended.

---

## Source Files

| File | Description |
|---|---|
| `atom/config.py` | `Config`, `CompilationConfig`, `CompilationLevel`, `CUDAGraphMode`, `QuantizationConfig`, `ParallelConfig`, `SpeculativeConfig`, `KVCacheTensor`, `KVCacheConfig`, `get_quant_config`, `get_hf_config` |
| `atom/utils/envs.py` | All `ATOM_*` environment variable definitions with lazy evaluation |
| `atom/model_engine/arg_utils.py` | `EngineArgs` dataclass and CLI argument parser |
| `atom/sampling_params.py` | `SamplingParams` dataclass |
| `atom/model_engine/model_runner.py` | Uses `ATOM_PROFILER_MORE` and `ATOM_TORCH_PROFILER_DIR` for profiling |
