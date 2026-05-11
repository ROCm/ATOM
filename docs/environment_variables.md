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

## Model Loading

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| **ATOM_DISABLE_MMAP** | bool | false | If set to `true`, disable memory-mapped file loading for model weights. Useful in containerized environments where mmap may cause issues. |

---

## Plugin Mode

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| **ATOM_DISABLE_VLLM_PLUGIN** | bool | 0 (false) | If set to `1`, disable the vLLM plugin registration entirely. |
| **ATOM_DISABLE_VLLM_PLUGIN_ATTENTION** | bool | 0 (false) | If set to `1`, disable only the vLLM attention plugin while keeping other plugins active. |

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
| **ATOM_ENABLE_DS_QKNORM_FUSION** | bool | 1 (true) | If set to `1`, use the fused Q/K RMSNorm path (`fused_qk_rmsnorm`) in the DeepSeek MLA attention module when Q-LoRA is enabled and QK norm+quant fusion is not used. If set to `0`, apply separate RMSNorm for the Q and KV branches instead. |
| **ATOM_ENABLE_DS_QKNORM_QUANT_FUSION** | bool | 1 (true) | If set to `1`, fuse QK norm with quantization in MLA attention module. |
| **ATOM_DUAL_STREAM_MOE_TOKEN_THRESHOLD** | int | 1024 | Upper bound on MoE token count (`num_tokens` in the MoE forward) for using the dual-stream path: shared experts on a secondary CUDA stream while routed experts run on the default stream. If `num_tokens` exceeds this value, that forward uses single-stream MoE instead. Set to `0` to disable dual-stream setup entirely (no alt stream, no `maybe_dual_stream_forward` registration). |

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

## V4 Attention Backend (Migration)

Selects between the legacy per-seq Python dispatch path in `atom/models/deepseek_v4.py`
and the new batched `V4AttentionBackend` (`atom/model_ops/v4_attention_backend.py`).
The new backend removes ~256 GPU→CPU `.item()` syncs per forward and is required
to enable CUDAGraph capture for V4. Legacy stays available during PR-A migration
for byte-equal A/B verification via dump-bisect; it is removed once all phases
land. See `atom/model_ops/v4_backend_gate.py` for the selector.

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| **ATOM_V4_BACKEND** | str | `legacy` | `legacy` keeps the per-seq dispatch loop. `new` routes through `V4AttentionBackend`. Layer-restricted by `ATOM_V4_BACKEND_LAYERS` if set. |
| **ATOM_V4_BACKEND_LAYERS** | csv int | "" (= all) | Comma-separated layer ids that use the new backend (others stay legacy). Empty means: apply `ATOM_V4_BACKEND` uniformly. Used for layer-by-layer bisect during migration (e.g. `0,3,15,30`). |

---

## Profiling & Debugging

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| **ATOM_TORCH_PROFILER_DIR** | str | — | When set, enables PyTorch profiler and writes traces to this directory. Create subdirectories per rank (e.g., `rank_0`, `dp0_tp0`). |
| **ATOM_PROFILER_MORE** | bool | 0 (false) | When `ATOM_TORCH_PROFILER_DIR` is set and this is `1`, enables detailed profiling: `record_shapes`, `with_stack`, and `profile_memory`. |
| **ATOM_LOG_MORE** | bool | 0 (false) | If set to `1`, use verbose logging format (includes process name, PID, path, line number, function name). |

### Debug Dump (`atom.utils.debug_helper`)

Env-gated dump / compare / monkey-patch primitives for forward bisect &
batch invariance investigation. All entries are **no-op when their
controlling `*_DIR` is unset**, so they are safe to leave wired into
production paths. See `.claude/skills/dump-bisect-debug.md` for the
methodology and `atom/utils/debug_helper/` for the implementation.

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| **ATOM_FWD_DUMP_DIR** | str | — | Enables `install_block_forward_hooks`. Per-Block hidden state is saved to `{DIR}/layer{LL}_{Cls}_rank{R}[_call{NNN}].pt`. |
| **ATOM_FWD_DUMP_LAYERS** | csv int | "" (= all) | Comma-separated layer ids to dump (e.g. `0,5,15,30`). Empty string means dump every layer. |
| **ATOM_FWD_DUMP_BLOCK_CLASS** | csv str | `Block` | Module class names to hook. Multiple values supported (e.g. `Block,DeepseekV4Attention,MoE,Compressor,Indexer`) for sub-stage bisect. Override per model. |
| **ATOM_FWD_DUMP_LAYER_ATTR** | str | `layer_id` | Attribute name on the block carrying its index. Some non-DeepSeek models use `layer_idx`. |
| **ATOM_FWD_DUMP_ONE_SHOT** | bool | 1 (true) | When `1`, only the first call per layer is dumped (typical: warmup). Set to `0` to enumerate every call (`_call000.pt`, `_call001.pt`, …) — required when bisecting per-seq dispatch loops. |
| **ATOM_WEIGHT_DUMP_DIR** | str | — | Enables `maybe_dump_weights_and_exit`. Per-rank params + buffers for selected layers dumped to `{DIR}/weight_rank{R}_layer{L}.pt`. Skips `.experts.*` (FP4 packed). |
| **ATOM_WEIGHT_DUMP_LAYERS** | csv int | `0` | Comma-separated layer ids to dump weights for. |
| **ATOM_WEIGHT_DUMP_EXIT** | bool | 1 (true) | When `1` (default), call `sys.exit(0)` after dumping. Set to `0` to continue inference after dump. |
| **ATOM_DEBUG_TOPK** | int | 0 | Set to `K > 0` to log top-K logits per row from `Sampler.forward` via `maybe_log_topk()`. Only rank 0 writes. |
| **ATOM_DEBUG_TOPK_PATH** | str | — | Optional output file for top-K logs. Writes to stderr if unset. |

CLI for comparing dumps:

```bash
python -m atom.utils.debug_helper.compare slot-invariance --dir DIR --n-slots 4
python -m atom.utils.debug_helper.compare ref-vs-target  --dir DIR
python -m atom.utils.debug_helper.compare layer-bisect   --dir DIR --threshold 0.99
python -m atom.utils.debug_helper.compare schema --a A.pt --b B.pt
```

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
