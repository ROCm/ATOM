# Local CLAUDE.md — ATOM (personal notes, not committed)

> Supplements the project-level `/workspace/ATOM/CLAUDE.md`.
> Add this to .gitignore if you don't want it tracked: `echo '.claude/CLAUDE.md' >> .gitignore`

---

## Supported Models (model_runner.py:51)

| HF Architecture | File | Notes |
|---|---|---|
| `Qwen3ForCausalLM` | `models/qwen3.py` | |
| `Qwen3MoeForCausalLM` | `models/qwen3_moe.py` | enable `ATOM_ENABLE_QK_NORM_ROPE_CACHE_QUANT_FUSION=1` for perf |
| `LlamaForCausalLM` | `models/llama.py` | |
| `MixtralForCausalLM` | `models/mixtral.py` | |
| `DeepseekV3ForCausalLM` | `models/deepseek_v2.py` | shared with V3.2 and GLM-5 |
| `DeepseekV32ForCausalLM` | `models/deepseek_v2.py` | |
| `GptOssForCausalLM` | `models/gpt_oss.py` | |
| `GlmMoeDsaForCausalLM` | `models/deepseek_v2.py` | GLM-5 (shares deepseek_v2) |
| `Glm4MoeForCausalLM` | `models/glm4_moe.py` | |
| `Qwen3NextForCausalLM` | `models/qwen3_next.py` | |
| `KimiK25ForConditionalGeneration` | `models/kimi_k25.py` | |
| `MiniMaxM2ForCausalLM` | `models/minimax_m2.py` | |

MTP (speculative decoding) models: `deepseek_mtp.py`, `qwen3_next_mtp.py`

---

## Config Quick Reference (config.py:705)

Key `Config` fields and their defaults:

```
max_num_batched_tokens = 16384
max_num_seqs          = 512
gpu_memory_utilization= 0.9
tensor_parallel_size  = 1   (max 8)
kv_cache_dtype        = "bf16"   (also: "fp8")
kv_cache_block_size   = 16
enable_prefix_caching = False
port                  = 8006
asyncio_mode          = False
enable_expert_parallel= False
enable_dp_attention   = False
```

### Speculative Decoding (MTP)
- `--speculative_config` with draft model path
- `num_speculative_tokens` max 4
- MTP layers: `num_nextn_predict_layers` (forced to 1, reused)
- Proposer: `atom/spec_decode/eagle.py` → `EagleProposer`
- **Do NOT instrument model files** — instrument at `ModelRunner.run_model()` or `EagleProposer.propose()`

### Compilation Levels (`CompilationConfig`)

| Level | Constant | Behavior |
|---|---|---|
| 0 | `NO_COMPILATION` | Eager |
| 1 | `DYNAMO_AS_IS` | torch.compile as-is |
| 2 | `DYNAMO_ONCE` | dynamo once |
| 3 | `PIECEWISE` | piecewise + CUDAGraph (default) |

### CUDAGraph Modes (`CUDAGraphMode`)
- `NONE` — no capture
- `PIECEWISE` — default v1
- `FULL` — full graph
- `FULL_DECODE_ONLY` — full decode, no prefill graph
- `FULL_AND_PIECEWISE` — both

Graph capture sizes: if one value given → `[1,2,4,8] + range(16, N+1, 16)`

---

## ATOM_* Environment Variables (atom/utils/envs.py)

Access via `from atom.utils import envs; envs.ATOM_FOO`

| Variable | Default | Purpose |
|---|---|---|
| `ATOM_DP_RANK` | 0 | Data parallelism rank |
| `ATOM_DP_RANK_LOCAL` | 0 | Local DP rank |
| `ATOM_DP_SIZE` | 1 | DP world size |
| `ATOM_DP_MASTER_IP` | 127.0.0.1 | DP master address |
| `ATOM_DP_MASTER_PORT` | 29500 | DP master port |
| `ATOM_USE_TRITON_GEMM` | 0 | Use Triton GEMM kernels |
| `ATOM_USE_TRITON_MXFP4_BMM` | 0 | Triton MXFP4 BMM |
| `ATOM_ENABLE_QK_NORM_ROPE_CACHE_QUANT_FUSION` | 0 | Qwen3-MoE perf fusion |
| `ATOM_ENABLE_DS_INPUT_RMSNORM_QUANT_FUSION` | **1** | DeepSeek input fusion |
| `ATOM_ENABLE_DS_QKNORM_QUANT_FUSION` | **1** | DeepSeek QK-norm quant |
| `ATOM_ENABLE_DS_QKNORM_FUSION` | **1** | DeepSeek QK-norm fusion |
| `ATOM_ENABLE_ALLREDUCE_RMSNORM_FUSION` | **1** | AllReduce+RMSNorm fusion |
| `ATOM_LLAMA_ENABLE_AITER_TRITON_FUSED_RMSNORM_QUANT` | **1** | Llama fused RMSNorm |
| `ATOM_LLAMA_ENABLE_AITER_TRITON_FUSED_SILU_MUL_QUANT` | **1** | Llama fused SiLU |
| `ATOM_TORCH_PROFILER_DIR` | None | Save torch profiler traces here |
| `ATOM_PROFILER_MORE` | 0 | Extra profiler metrics |
| `ATOM_LOG_MORE` | 0 | Verbose logging |
| `ATOM_DISABLE_MMAP` | false | Disable mmap for model loading |
| `ATOM_DISABLE_VLLM_PLUGIN` | 0 | Disable vLLM plugin mode |
| `ATOM_DISABLE_VLLM_PLUGIN_ATTENTION` | 0 | Disable plugin attention |
| `ATOM_USE_CUSTOM_ALL_GATHER` | **1** | Custom all-gather (vs NCCL) |
| `ATOM_DUAL_STREAM_MOE_TOKEN_THRESHOLD` | 1024 | Dual-stream MoE threshold (tokens) |

### Third-party vars (not managed in envs.py)
- `AITER_LOG_LEVEL=WARNING` — **always set before starting server**
- `AITER_QUICK_REDUCE_QUANTIZATION` — set by model_runner.py
- `TORCHINDUCTOR_CACHE_DIR`, `TRITON_CACHE_DIR` — set by compiler_interface.py
- `HF_TOKEN` — HuggingFace auth
- `FLA_*` — FLA ops library flags
- `VLLM_PP_LAYER_PARTITION` — pipeline layer partitioning (legacy, still active)

---

## Model Ops (atom/model_ops/)

| File | Purpose |
|---|---|
| `attention_mha.py` | Multi-head attention |
| `attention_mla.py` | Multi-latent attention (DeepSeek) |
| `attention_gdn.py` | GDN attention variant |
| `base_attention.py` | Shared attention base |
| `linear.py` | Linear layers (quantized-aware) |
| `moe.py` | MoE routing + dispatch |
| `fused_moe_triton.py` | Triton-based fused MoE |
| `layernorm.py` | RMSNorm / LayerNorm |
| `rotary_embedding.py` | RoPE |
| `paged_attention.py` | Paged KV attention |
| `radix_attention.py` | Radix-tree prefix cache attention |
| `sampler.py` | Token sampling |
| `rejection_sampler.py` | Speculative decoding rejection |
| `embed_head.py` | Embedding + LM head |
| `activation.py` | SiLU, GELU, etc. |
| `topK.py` | Top-K selection (MoE) |

---

## Engine & Scheduler (atom/model_engine/)

```
llm_engine.py       — public API, tokenizer, CoreManager init
engine_core_mgr.py  — ZMQ IPC management (CoreManager)
engine_core.py      — inference loop
scheduler.py        — batching (ScheduledBatch, Scheduler)
model_runner.py     — forward pass, KV cache, CUDAGraphs
block_manager.py    — physical KV block allocator
sequence.py         — Sequence, SequenceGroup state machine
request.py          — Request abstraction
arg_utils.py        — CLI argument parsing
async_proc.py       — async subprocess utilities
```

### Scheduler key classes
- `ScheduledBatch` (line 179) — output of one scheduling step
- `ScheduledBatchOutput` (line 265) — post-forward output
- `Scheduler` (line 292) — main scheduling logic
- `BlockManager` — can_allocate / allocate / may_append / deallocate

---

## Quantization

- Config parsed from HF `quantization_config` via `atom/quantization/quark/`
- Access per-layer config: `quant_config.get_layer_quant_config(prefix)`
- Quant types: `QuantType.No`, weight-only, W8A8, MXFP4, etc.
- KV cache quant: `--kv_cache_dtype fp8` (Config.kv_cache_dtype)
- FP8 skip: `weights_proj` in GLM-5 (GEMM crash workaround — see recent commits)

---

## Plugin Mode (atom/plugin/)

ATOM can run as a vLLM plugin (`ATOM_DISABLE_VLLM_PLUGIN=0`):
- `plugin/vllm/register.py` — patches vLLM attention, platform, TP group
- `plugin/vllm/mla_patch.py` — MLA attention forward override
- `plugin/vllm/model_wrapper.py` — wraps vLLM model in ATOM runner
- `plugin/attention_mla.py` — plugin-mode MLA implementation
- Disable attention plugin only: `ATOM_DISABLE_VLLM_PLUGIN_ATTENTION=1`

---

## Tests (tests/)

All tests run without GPU (AITER and torch.cuda are mocked via `conftest.py`):

| Test file | Coverage |
|---|---|
| `test_scheduler.py` | Scheduler batching logic |
| `test_block_manager.py` | KV block allocation |
| `test_sequence.py` | Sequence state machine |
| `test_request.py` | Request lifecycle |
| `test_envs.py` | Env var definitions |
| `test_quant_config.py` | Quantization config parsing |
| `test_sampling_params.py` | Sampling parameter validation |
| `test_prefix_cache_accuracy.py` | Prefix cache correctness |
| `test_profiler_regression.py` | HIP graph replay profiler guard |
| `test_kimi_k25.py` | Kimi K2.5 model |
| `test_mxfp4_moe_has_bias.py` | MoE MXFP4 with bias |
| `test_qwen3_coder_fixes.py` | Qwen3-Coder fixes |
| `plugin/` | Plugin mode config, env flags, status |

---

## Common Workflows

```bash
# Start server (production)
AITER_LOG_LEVEL=WARNING python -m atom.entrypoints.openai_server \
  --model <model> --kv_cache_dtype fp8 -tp 8

# Clear stale compile cache (do this after code changes)
rm -rf /root/.cache/atom/*

# Verify GPU is actually loaded (not just HTTP health)
rocm-smi --showmemuse   # VRAM% must be > 0

# Run all tests
python -m pytest tests/

# Format + lint
black . && ruff check .

# Profiling run
ATOM_TORCH_PROFILER_DIR=/tmp/traces ATOM_PROFILER_MORE=1 python -m atom.entrypoints.openai_server ...
```
