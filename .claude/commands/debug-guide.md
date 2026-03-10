# ATOM Debugging Guide

## Quick Triage

| Symptom | Jump to |
|---------|---------|
| Server won't start / hangs on startup | [Server Setup](#server-setup), [Multi-Process](#multi-process--zmq) |
| GPU OOM | [GPU / CUDA](#gpu--cuda) |
| Server hangs during serving | [Multi-Process](#multi-process--zmq), [Scheduler](#scheduler), [NCCL](#nccl--distributed) |
| Wrong or garbled output | [Sampling & Output](#sampling--output) |
| Slow first request / cold start | [Compilation](#compilation) |
| Low throughput / decode starvation | [Scheduler](#scheduler) |
| Low speculative decode acceptance | [Speculative Decoding / MTP](#speculative-decoding--mtp) |
| NCCL hang / timeout | [NCCL](#nccl--distributed) |
| Weight loading failure | [Weight Loading & Quantization](#weight-loading--quantization) |

> **First step for any model execution issue:** Run with `--enforce-eager` to disable compilation and CUDA graphs. This eliminates an entire class of problems.

## Server Setup

- **Set** `AITER_LOG_LEVEL=WARNING` to suppress aiter kernel-level log flooding (this is an aiter library env var, not ATOM's)
- **Kill old servers before starting new ones.** Check `rocm-smi` for VRAM usage — leftover processes hold GPU memory
- **Detect server hangs:** Use polling (`grep -q` in a loop with timeout), never blocking `tail -f | grep` which hangs indefinitely if the process crashes
- **Always delete `gpucore.*` files** when GPU hang occurs

## Observability

- **Runtime profiling:** `curl -X POST http://127.0.0.1:8000/start_profile` / `stop_profile`. Traces saved to `ATOM_TORCH_PROFILER_DIR`
- **Detailed profiling:** Set `ATOM_PROFILER_MORE=1` for `record_shapes`, `with_stack`, `profile_memory`
- **Offline profiling:** `python -m atom.examples.profile_offline --model <model> --kv_cache_dtype fp8`
- **Accuracy validation:** Always run `lm_eval` with gsm8k after inference or MTP changes
- **ATOM logging:** Configure via `logging.getLogger("atom")`

## Instrumentation Rules

- **[CRITICAL] Never modify `@support_torch_compile` decorated models.** This breaks Dynamo tracing even with `--enforce-eager`. Find all affected models: `grep -rn "@support_torch_compile" atom/models/`
  - Currently decorated: `deepseek_v2.py`, `deepseek_mtp.py`, `qwen3.py`, `qwen3_moe.py`, `qwen3_next.py`, `llama.py`, `mixtral.py`, `glm4_moe.py`, `gpt_oss.py`
- **Instrument at call sites instead.** Add checks before/after `self.model()` in callers (e.g., `EagleProposer.propose()` for MTP, `ModelRunner.run_model()` for the target model)
- **Clear compile cache after accidental modification:** `rm -rf ~/.cache/atom/torch_compile_cache/`

## GPU / CUDA

- **[COMMON] OOM during startup:** Reduce `max_num_seqs` or `max_num_batched_tokens`. Check `rocm-smi` for fragmentation. Run `torch.cuda.empty_cache()` between attempts
- **GPU memory fault (SIGBUS/SIGSEGV):** Trace the bad index backward — check `block_tables`, `kv_indices`, `slot_mapping` shapes and bounds
- **[COMMON] CUDA graph capture failures:** Happen during `ModelRunner.capture_cudagraph()`. Sizes are `[1, 2, 4, 8, ..., max_num_seqs]`. If capture fails at a specific size, it may exceed KV cache capacity. **Fix:** Use `--enforce-eager` to bypass
- **CUDA graph replay mismatch:** `CUDAGraphWrapper` checks input tensor addresses match between capture and replay. Address mismatch means input buffers changed between capture and replay. **Fix:** Check `allocate_forward_vars()` and `copy_and_call()` in attention backends
- **Stale graph pool:** After model changes, old graphs reference freed memory. Always restart the server; don't hot-reload

## Compilation

- **Levels:** 0=eager, 1=dynamo-eager, 2=dynamo-inductor, 3=piecewise (default). Use `--enforce-eager` for level 0
- **Cold start slow?** Cache key = config hash + traced code hash + compiler hash. **Any** change invalidates. Clear: `rm -rf ~/.cache/atom/torch_compile_cache/`
- **"vLLM failed to compile the model":** Usually corrupted cache. **Fix:** `rm -rf ~/.cache/atom/torch_compile_cache/` and retry
- **PyTorch 2.8+** uses `InductorStandaloneAdaptor`; older uses `InductorAdaptor`. See `make_compiler()` in `backends.py`
- **[RARE] Piecewise compilation stall:** `PiecewiseBackend` lazily compiles new batch sizes on first encounter. **Diagnose:** Check logs for "Compiling a graph for shape" messages. **Mitigate:** Pre-populate `compile_sizes` with expected batch sizes

## Multi-Process / ZMQ

- **Architecture:** `CoreManager` → ZMQ → `EngineCore` (per DP rank) → `AsyncIOProcManager` → `ModelRunner` (per TP rank)
- **[COMMON] DP hang on startup:** EngineCore sends `READY` only after model load + CUDA graph capture. `CoreManager._wait_for_all_ready_signals()` blocks until all ranks report. If one rank OOMs, all hang. **Fix:** Check per-rank logs
- **ZMQ socket types:** Input: `ROUTER`→`DEALER`. Output: `PUSH`→`PULL`. Pickle-serialized `Sequence` objects
- **Process crash silent failure:** `AsyncIOProcManager` monitors children. If `ModelRunner` dies, `EXECUTOR_FAILED` is sent. Check if `runner_mgr.call_func()` returns or hangs
- **DP round-robin stuck:** `CoreManager._rr_counter` dispatches round-robin. If one EngineCore is slow, requests queue. Check per-rank scheduler state
- **Multiprocessing start method:** Must be `spawn` (set in `engine_core_mgr.py:58` and `:462`). `fork` causes CUDA re-init issues

## KV Cache / Block Manager

- **[COMMON] "Failed to allocate kv cache":** `num_blocks` returned 0. Model + KV cache don't fit in VRAM. **Fix:** Reduce `max_num_seqs` or use `--kv_cache_dtype fp8`
- **Block exhaustion during serving:** Scheduler preempts the last running sequence when blocks run out. Monitor `len(free_block_ids)` in `BlockManager`
- **[RARE] Prefix caching hash collision:** Uses `xxhash64` with chain hashing. Collisions cause cache miss (performance only, not correctness). **Diagnose:** Monitor `BlockManager.allocate()` for `token_ids` mismatch fallthrough
- **Block leak (ref_count never reaches 0):** If `deallocate()` is never called, blocks leak. **Fix:** Verify `seq.block_table` is cleared on sequence completion
- **Mamba block table:** Mamba-like models use separate `mamba_block_table`. No prefix caching support

## Scheduler

- **Prefill-first policy:** Prefill always runs before decode. High request rate → decode starvation. **Diagnose:** Monitor `scheduler.waiting` vs `scheduler.running` queue lengths
- **Preemption:** KV cache exhausted → scheduler preempts last running sequence. Excessive preemption = insufficient KV cache blocks
- **Batch size mismatch with CUDA graph:** Scheduler pads to nearest captured graph size. If batch exceeds largest captured size, falls back to eager. Check `graph_bs` in `ModelRunner`

## ForwardContext

- **"Forward context is not set":** `set_forward_context()` was not called before model forward. Lifecycle bug — check `ModelRunner` forward path. See CLAUDE.md "Forward Context Pattern" for lifecycle details
- **[COMMON] DP metadata wrong:** `DPMetadata.make()` does CPU `all_reduce` via Gloo. If one rank has 0 tokens, `local_size` is clamped to 1. Mismatch causes NCCL hang. **Diagnose:** Log `num_tokens_across_dp` before `DPMetadata.make()` at `forward_context.py`. Check if any rank shows 0 tokens unexpectedly
- **Stale context after exception:** If forward throws, `reset_forward_context()` may not be called. **Fix:** Always wrap forward in try/finally

## Speculative Decoding / MTP

- **Low acceptance rate:** Check `SpecStats` in scheduler — logs acceptance distribution every `log_interval * mtp_k` draft tokens (default `log_interval=1000`). Distribution shows how many draft tokens (0..k) were accepted per step
- **Draft model mismatch:** `EagleProposer` calls the MTP model. If draft weights are stale/mismatched, acceptance drops to near-zero. **Fix:** Verify `--speculative-model` path
- **Rejection sampler issues:** `RejectionSampler` compares target vs draft logits. NaN/Inf in logits causes silent rejection of all tokens. **Diagnose:** Add `torch.isnan(logits).any()` checks at `ModelRunner` call sites

## Sampling & Output

- **Wrong/garbled output:** Check tokenizer BOS/EOS token IDs (`config.bos_token_id`, `config.eos_token_id`). Verify `InputOutputProcessor` in `llm_engine.py`
- **Unexpected EOS:** Check `SamplingParams.stop` and `stop_token_ids` in `sampling_params.py`
- **Temperature/top-p issues:** Check `Sampler` in `model_ops/sampler.py`

## Weight Loading & Quantization

- **Shape mismatch on load:** Check `load_model()` in `model_loader/loader.py`. Verify HF `config.json` `architectures` field matches `support_model_arch_dict`
- **Quant format issues:** Auto-detected from HF `config.json` `quantization_config`. Supported: FP8 (per-tensor, per-token), INT8, MXFP4, Quark
- **FP8 KV cache:** Ensure `--kv_cache_dtype fp8` is set consistently

## NCCL / Distributed

- **NCCL hang:** Set `NCCL_DEBUG=INFO` for verbose logs. Common cause: one rank diverges in collective op count
- **NCCL timeout:** Set `NCCL_TIMEOUT` env var (default 1800s). Check rank synchronization
- **DP Gloo all_reduce hang:** Check `DPMetadata.num_tokens_across_dp()` in `forward_context.py`. Uses `get_dp_group().cpu_group`
- **TP NCCL init failure:** Check `init_dist_env()` call in `ModelRunner`. Verify `MASTER_ADDR` and `MASTER_PORT`
- **Expert parallelism hang:** Check MORI all-to-all communication layer. Verify `--enable-expert-parallel` flag. Check `FusedMoE` in `model_ops/moe.py` for expert load distribution

## Testing & Validation

- **Quick smoke test:** `python -m atom.examples.simple_inference --model <model> --kv_cache_dtype fp8` — runs 4 prompts with profiling, prints MTP stats. Good for verifying end-to-end inference, sampling, and speculative decoding in one shot
- **Unit tests:** `cd /app/ATOM && pytest tests/test_scheduler.py -v`
- **Accuracy:** `lm_eval --model local-completions --tasks gsm8k --num_fewshot 5` (see CLAUDE.md for full command)
- **Test stubs:** `conftest.py` mocks GPU/ZMQ/HF dependencies — unit tests run without a GPU

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
| MTP model | `models/deepseek_mtp.py` | **Compiled — do NOT modify** |
| Target model | `models/deepseek_v2.py` | **Compiled — do NOT modify** |
| MoE | `model_ops/moe.py` | FusedMoE, expert parallelism |
| Sampler | `model_ops/sampler.py` | Token sampling |
| Rejection sampler | `model_ops/rejection_sampler.py` | Spec decode logit comparison |
| Simple inference | `examples/simple_inference.py` | Quick smoke test with profiling + MTP stats |
