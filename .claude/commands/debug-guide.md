# ATOM Debugging Guide

## Quick Triage

| Symptom | Jump to |
|---------|---------|
| Server won't start / hangs on startup | [Server Setup](#server-setup), [Multi-Process](#multi-process--zmq) |
| GPU OOM | [GPU / CUDA](#gpu--cuda) |
| Server hangs during serving | [Multi-Process](#multi-process--zmq), [Scheduler](#scheduler), [NCCL](#nccl--distributed) |
| Wrong or garbled output | [Sampling & Output](#sampling--output), [Speculative Decoding / MTP](#speculative-decoding--mtp) |
| Low / zero MTP acceptance rate | [Speculative Decoding / MTP](#speculative-decoding--mtp), [Weight Loading & Quantization](#weight-loading--quantization) |
| Slow first request / cold start | [Compilation](#compilation) |
| Low throughput / decode starvation | [Scheduler](#scheduler) |
| NCCL hang / timeout | [NCCL](#nccl--distributed) |
| Weight loading failure | [Weight Loading & Quantization](#weight-loading--quantization) |

> **First step for any model execution issue:** Run with `--level 0 --enforce-eager` to disable both torch.compile and CUDA graphs. This eliminates an entire class of problems. (`--enforce-eager` alone only disables CUDA graphs; Dynamo tracing still runs.)

## General Debugging Methodology

1. **Isolate with minimal tests.** Don't repeatedly restart the server to test hypotheses. Write standalone Python scripts that call the specific kernel/function in question, comparing spec vs non-spec or different input shapes. This is faster and produces definitive results.
2. **Compare known-good vs broken.** Use the same input prompt with `--level 0 --enforce-eager`. Dump logits argmax and hidden-state norms at each stage. The first point of divergence is the bug.
3. **Check weight loading before checking model logic.** Print `module.weight.dtype` and `module.weight.float().norm()` at runtime. A `norm=0.0` or wrong `dtype` means loading issue, not inference bug.
4. **Read the safetensors index, not individual shards.** Weights may be spread across 90+ shards. Use `model.safetensors.index.json` to locate specific weights.
5. **Reference existing implementations.** Before assuming a bug, check if a similar model (e.g. Qwen3-Next for Qwen3.5, MiMo-V2-Flash for MHA MTP) works. If it does, diff the code paths.

## Server Setup

- **[CRITICAL] Verify server is truly running with GPU loaded** — `curl /health` returning OK is NOT sufficient. Always verify with:
  ```bash
  curl -sf http://localhost:8000/v1/models          # API responds with model name
  rocm-smi --showmemuse | grep "GPU Memory Allocated"  # VRAM% > 0
  ```
- **Set** `AITER_LOG_LEVEL=WARNING` to suppress aiter kernel-level log flooding
- **Kill old servers before starting new ones.** Check `rocm-smi` for VRAM usage — leftover processes hold GPU memory
- **Detect server hangs:** Use polling (`grep -q` in a loop with timeout), never blocking `tail -f | grep`
- **Always delete `gpucore.*` files** when GPU hang occurs

## Observability

- **Runtime profiling:** `curl -X POST http://127.0.0.1:8000/start_profile` / `stop_profile`. Traces saved to `ATOM_TORCH_PROFILER_DIR`
- **Detailed profiling:** Set `ATOM_PROFILER_MORE=1` for `record_shapes`, `with_stack`, `profile_memory`
- **Offline profiling:** `python -m atom.examples.profile_offline --model <model> --kv_cache_dtype fp8`
- **Accuracy validation:** Always run `lm_eval` with gsm8k after inference or MTP changes
- **ATOM logging:** Configure via `logging.getLogger("atom")`

## Instrumentation Rules

- **[CRITICAL] Never modify `@support_torch_compile` decorated models.** This breaks Dynamo tracing. Find all: `grep -rn "@support_torch_compile" atom/models/`
  - Currently decorated: `deepseek_v2.py`, `deepseek_mtp.py`, `qwen3.py`, `qwen3_moe.py`, `qwen3_next.py`, `llama.py`, `mixtral.py`, `glm4_moe.py`, `gpt_oss.py`
- **[CRITICAL] `--enforce-eager` and `--level 0` are independent.** `--enforce-eager` disables CUDA graphs only. `--level 0` disables torch.compile only. Debug instrumentation requires **both**: `--level 0 --enforce-eager`
- **[CRITICAL] Do NOT add debug code in `run_model()`.** It runs during CUDA graph warmup within a Dynamo-traced context. Class variables and `forward_context` attribute access trigger `InternalTorchDynamoError`. **Put debug code in `forward()` instead** (has `@torch.inference_mode()`), guarded by `not batch.is_dummy_run`
- **Instrument at call sites.** Add checks in `EagleProposer.propose()` for MTP, or `ModelRunner.forward()` for target model — never inside compiled model code
- **Clear compile cache after accidental modification:** `rm -rf ~/.cache/atom/torch_compile_cache/`

## GPU / CUDA

- **[COMMON] OOM during startup:** Reduce `max_num_seqs` or `max_num_batched_tokens`. Check `rocm-smi` for fragmentation
- **GPU memory fault (SIGBUS/SIGSEGV):** Trace the bad index backward — check `block_tables`, `kv_indices`, `slot_mapping` shapes and bounds
- **[COMMON] CUDA graph capture failures:** Use `--enforce-eager` to bypass. Sizes are `[1, 2, 4, 8, ..., max_num_seqs]`
- **CUDA graph replay mismatch:** Input tensor addresses must match between capture and replay. Check `allocate_forward_vars()` and `copy_and_call()` in attention backends
- **Stale graph pool:** After model changes, always restart the server; don't hot-reload

## Compilation

- **Levels:** 0=eager (no torch.compile), 1=dynamo-eager, 2=dynamo-inductor, 3=piecewise (default). `--level 0` disables torch.compile; `--enforce-eager` disables CUDA graphs. For full eager mode use both: `--level 0 --enforce-eager`
- **Cold start slow?** Cache key = config hash + traced code hash + compiler hash. **Any** change invalidates. Clear: `rm -rf ~/.cache/atom/torch_compile_cache/`
- **"vLLM failed to compile the model":** Usually corrupted cache. Clear and retry
- **[RARE] Piecewise compilation stall:** `PiecewiseBackend` lazily compiles new batch sizes on first encounter. Check logs for "Compiling a graph for shape" messages

## Multi-Process / ZMQ

- **Architecture:** `CoreManager` → ZMQ → `EngineCore` (per DP rank) → `AsyncIOProcManager` → `ModelRunner` (per TP rank)
- **[COMMON] DP hang on startup:** EngineCore sends `READY` only after model load + CUDA graph capture. If one rank OOMs, all hang. **Fix:** Check per-rank logs
- **Process crash silent failure:** `AsyncIOProcManager` monitors children. If `ModelRunner` dies, `EXECUTOR_FAILED` is sent
- **Multiprocessing start method:** Must be `spawn`. `fork` causes CUDA re-init issues

## KV Cache / Block Manager

- **[COMMON] "Failed to allocate kv cache":** `num_blocks` returned 0. Model + KV cache don't fit in VRAM. **Fix:** Reduce `max_num_seqs` or use `--kv_cache_dtype fp8`
- **Block exhaustion during serving:** Scheduler preempts last running sequence when blocks run out
- **Block leak (ref_count never reaches 0):** Verify `seq.block_table` is cleared on sequence completion
- **Mamba/GDN block table:** Hybrid models use separate `mamba_block_table` for linear attention state. No prefix caching support

## Scheduler

- **Prefill-first policy:** Prefill always runs before decode. High request rate → decode starvation
- **Preemption:** KV cache exhausted → scheduler preempts last running sequence. Excessive preemption = insufficient KV cache blocks
- **Batch size mismatch with CUDA graph:** Scheduler pads to nearest captured graph size. If batch exceeds largest captured size, falls back to eager

## ForwardContext

- **"Forward context is not set":** `set_forward_context()` was not called before model forward. Lifecycle bug — check `ModelRunner` forward path
- **[COMMON] DP metadata wrong:** `DPMetadata.make()` does CPU `all_reduce` via Gloo. If one rank has 0 tokens, mismatch causes NCCL hang
- **Stale context after exception:** If forward throws, `reset_forward_context()` may not be called. Wrap forward in try/finally

## Speculative Decoding / MTP

- **Low acceptance rate:** Check `SpecStats` in scheduler logs. Distribution shows how many draft tokens (0..k) were accepted per step
- **0% acceptance with correct output:** The MTP model produces all-zero hidden states → weight loading issue. **Diagnose:** Print `logits[0].float().norm()` in `EagleProposer.propose()`. If `0.0`:
  1. Check `fc.weight` dtype — BF16 checkpoint weight may be silently quantized to FP8 if `ColumnParallelLinear` lacks the correct `prefix` argument (see [Weight Loading](#weight-loading--quantization))
  2. Check `loaded_weights_record` from `load_model()` — is the weight present?
  3. Check `_share_if_not_loaded` log for `lm_head` sharing
- **Garbled output with MTP (GDN hybrid models):** Compare logits at position 0 between MTP (q_len=2) and non-MTP (q_len=1) using `--level 0 --enforce-eager`. If they differ → GDN kernel bug. Write standalone tests for `causal_conv1d_update` and `fused_recurrent_gated_delta_rule` comparing spec vs non-spec at token 0. **Do not suspect `pa_fwd_asm`** — it handles causal masking correctly with `max_qlen > 1` (verified)
- **MTP crash in propose loop (mtp_k > 1):** The propose loop updates attention metadata per draft step. MHA and MLA need different fields (`block_tables`/`context_lens` vs `kv_last_page_lens`). Check `use_mla` branching in `eagle.py`. Ref: PR #560 (MiMo-V2-Flash)
- **Rejection sampler issues:** NaN/Inf in logits causes silent rejection of all tokens

## Sampling & Output

- **Wrong/garbled output:** Check tokenizer BOS/EOS token IDs. Verify `InputOutputProcessor` in `llm_engine.py`
- **Unexpected EOS:** Check `SamplingParams.stop` and `stop_token_ids`
- **Temperature/top-p issues:** Check `Sampler` in `model_ops/sampler.py`

## Weight Loading & Quantization

- **Shape mismatch on load:** Verify HF `config.json` `architectures` field matches `support_model_arch_dict`
- **Quant format issues:** Auto-detected from `quantization_config`. Supported: FP8, INT8, MXFP4, Quark
- **BF16 weight silently quantized to FP8:** Happens when `ColumnParallelLinear` is created without the correct `prefix` — `get_layer_quant_config("")` returns the global FP8 config instead of checking `modules_to_not_convert`. **Diagnose:** Print `module.weight.dtype` at runtime. **Fix:** Pass `prefix=f"{prefix}.fc"`. Don't remove `quant_config`; the quant system handles BF16 correctly when prefix matches
- **Weight not found in checkpoint:** Check `model.safetensors.index.json` for weight locations across shards. Inspecting a single shard is insufficient
- **MTP weight remap:** Each MTP model class defines `remap_mtp_weight_name` and `weights_mapping`. Loader calls `model.remap_mtp_weight_name(name)` to filter, then applies `weights_mapping` for name transformation

## NCCL / Distributed

- **NCCL hang:** Set `NCCL_DEBUG=INFO` for verbose logs. Common cause: one rank diverges in collective op count
- **NCCL timeout:** Set `NCCL_TIMEOUT` env var (default 1800s)
- **TP NCCL init failure:** Check `init_dist_env()`. Verify `MASTER_ADDR` and `MASTER_PORT`
- **Expert parallelism hang:** Check MORI all-to-all communication layer. Verify `--enable-expert-parallel` flag

## Testing & Validation

- **Quick smoke test:** `python -m atom.examples.simple_inference --model <model> --kv_cache_dtype fp8` — runs 4 prompts with profiling, prints MTP stats
- **Unit tests:** `pytest tests/test_scheduler.py -v`
- **Accuracy:** `lm_eval --model local-completions --tasks gsm8k --num_fewshot 5`
- **Kernel isolation test:** To verify a specific kernel (e.g. `causal_conv1d_update`, `pa_fwd_asm`), write a standalone script that creates minimal inputs, calls the kernel with both spec and non-spec parameters, and compares outputs. This is the fastest way to isolate kernel bugs without server restarts

## Quick Reference

| Component | Key File | Debug Notes |
|-----------|----------|-------------|
| Server entry | `entrypoints/openai_server.py` | Sampling params, API handling |
| Core manager | `model_engine/engine_core_mgr.py` | ZMQ orchestration, DP dispatch |
| Engine core | `model_engine/engine_core.py` | Per-DP-rank busy loop |
| Model runner | `model_engine/model_runner.py` | CUDA graph capture/replay, forward dispatch |
| Scheduler | `model_engine/scheduler.py` | Batch scheduling, preemption, SpecStats |
| Block manager | `model_engine/block_manager.py` | KV cache block allocation |
| Forward context | `utils/forward_context.py` | Module-level global, attn/DP metadata |
| Compilation | `utils/backends.py` | VllmBackend, CompilerManager |
| CUDA graph | `utils/cuda_graph.py` | CUDAGraphWrapper |
| MTP proposer | `spec_decode/eagle.py` | Safe to instrument |
| MTP model (DeepSeek) | `models/deepseek_mtp.py` | **Compiled — do NOT modify** |
| MTP model (Qwen3.5) | `models/qwen3_5_mtp.py` | **Compiled — do NOT modify** |
| Target model | `models/deepseek_v2.py` | **Compiled — do NOT modify** |
| MoE | `model_ops/moe.py` | FusedMoE, expert parallelism |
| Sampler | `model_ops/sampler.py` | Token sampling |
| Rejection sampler | `model_ops/rejection_sampler.py` | Spec decode logit comparison |
| Simple inference | `examples/simple_inference.py` | Quick smoke test with profiling + MTP stats |
