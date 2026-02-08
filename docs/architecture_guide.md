# ATOM Architecture Guide

> **Quick Reference**
>
> | Class | Import | Purpose |
> |-------|--------|---------|
> | `LLMEngine` | `from atom.model_engine.llm_engine import LLMEngine` | User-facing inference API |
> | `InputOutputProcessor` | `from atom.model_engine.llm_engine import InputOutputProcessor` | Tokenize/detokenize, TTFT/TPOT stats |
> | `CoreManager` | `from atom.model_engine.engine_core_mgr import CoreManager` | Multi-process orchestration via ZMQ |
> | `EngineCore` | `from atom.model_engine.engine_core import EngineCore` | Per-process engine loop |
> | `DPEngineCoreProc` | `from atom.model_engine.engine_core import DPEngineCoreProc` | Data-parallel engine core variant |
> | `ModelRunner` | `from atom.model_engine.model_runner import ModelRunner` | Per-GPU model execution |
> | `Scheduler` | `from atom.model_engine.scheduler import Scheduler` | Prefill-first request scheduling |
> | `BlockManager` | `from atom.model_engine.block_manager import BlockManager` | KV cache block allocation |
> | `Sequence` | `from atom.model_engine.sequence import Sequence` | Request state and token tracking |
> | `ForwardContext` | `from atom.utils.forward_context import ForwardContext` | Global forward pass metadata |
> | `Config` | `from atom.config import Config` | Master configuration dataclass |

---

## 1. System Overview

ATOM (AiTer Optimized Model) is AMD's lightweight LLM inference engine, inspired by vLLM's architecture and built on the [AITER](https://github.com/ROCm/aiter) kernel library for ROCm/HIP GPUs.

Key design principles:

- **Multi-process architecture** -- each engine core runs in its own process, with ZMQ-based IPC connecting the user-facing API to one or more GPU workers.
- **AITER-native execution** -- model forward passes use AITER's optimized attention, MoE, sampling, and communication kernels rather than generic PyTorch operators.
- **CUDA graph acceleration** -- decode batches are captured into CUDA graphs for replay, eliminating per-step kernel launch overhead.
- **Prefill-first scheduling** -- the scheduler prioritizes prompt prefills before decode steps, following vLLM's continuous batching strategy.
- **Speculative decoding** -- optional EAGLE/MTP (Multi-Token Prediction) draft models propose tokens that are verified via rejection sampling.

---

## 2. Component Architecture

```
LLMEngine (user-facing API)
├── InputOutputProcessor (tokenize/detokenize, TTFT/TPOT stats)
├── CoreManager (multi-process orchestration via ZMQ)
│   └── EngineCore (one per DP rank, runs in its own process)
│       ├── ModelRunner (per-GPU execution via AsyncIOProcManager)
│       │   ├── Model (Qwen3, Llama, DeepSeek, Mixtral, etc.)
│       │   ├── Sampler / RejectionSampler
│       │   └── EagleProposer (optional MTP draft)
│       └── Scheduler
│           └── BlockManager (KV cache block management)
└── Config (master configuration)
```

**Supported model architectures** (registered in `support_model_arch_dict`, a module-level dict in `model_runner.py`):

| Architecture key | Implementation |
|---|---|
| `Qwen3ForCausalLM` | `atom.models.qwen3.Qwen3ForCausalLM` |
| `Qwen3MoeForCausalLM` | `atom.models.qwen3_moe.Qwen3MoeForCausalLM` |
| `LlamaForCausalLM` | `atom.models.llama.LlamaForCausalLM` |
| `MixtralForCausalLM` | `atom.models.mixtral.MixtralForCausalLM` |
| `DeepseekV3ForCausalLM` | `atom.models.deepseek_v2.DeepseekV2ForCausalLM` |
| `DeepseekV32ForCausalLM` | `atom.models.deepseek_v2.DeepseekV2ForCausalLM` |
| `GptOssForCausalLM` | `atom.models.gpt_oss.GptOssForCausalLM` |
| `Glm4MoeForCausalLM` | `atom.models.glm4_moe.Glm4MoeForCausalLM` |

---

## 3. Request Lifecycle

A request flows through the system in ten steps:

1. **`LLMEngine.add_request()` / `generate()`** -- the user submits a list of prompts (strings or pre-tokenized token IDs) together with `SamplingParams`.

2. **`InputOutputProcessor.preprocess()`** -- each prompt is tokenized via the HuggingFace tokenizer. A `Sequence` object is created to track the request's state, timing, and block allocation. `arrive_time` is recorded.

3. **`CoreManager.add_request()`** -- the list of `Sequence` objects is serialized with `pickle` and sent over a ZMQ `ROUTER` socket. When multiple DP ranks are active, requests are distributed round-robin.

4. **`EngineCore.process_input_sockets()`** -- an I/O thread on the `EngineCore` process receives the serialized data on a ZMQ `DEALER` socket, deserializes it, and places the sequences into the `input_queue`.

5. **`EngineCore.busy_loop()`** -- the main execution loop pulls from `input_queue` via `pull_and_process_input_queue()`, feeds new sequences into the scheduler, and repeatedly calls `_process_engine_step()` until all work is done.

6. **`Scheduler.schedule()`** -- implements prefill-first scheduling. Waiting sequences are scheduled for prefill if they fit within `max_num_seqs` and `max_num_batched_tokens` and the `BlockManager` can allocate blocks. If no prefills are pending, running sequences are batched for decode. The scheduler returns a `ScheduledBatch` and the corresponding sequence map.

7. **`ModelRunner.forward()`** -- executes the three-phase forward pass:
   - `prepare_model()` -- assembles input IDs (handling deferred output from previous steps), builds attention metadata, and gathers sampling temperatures.
   - `run_model()` -- runs the model forward. Prefill and large batches run eagerly; decode batches replay captured CUDA graphs. Returns logits and hidden states.
   - `postprocess()` -- samples tokens (or runs rejection sampling for speculative decoding), prepares deferred output via `tokenIDProcessor`, and optionally proposes draft tokens through `EagleProposer`.

8. **`Scheduler.postprocess()`** -- appends sampled tokens to each `Sequence`, records `first_token_time`, checks stop conditions (EOS, stop token IDs, stop token sequences, `max_tokens`), and moves finished sequences out of the running queue. The `BlockManager` deallocates blocks for finished sequences.

9. **Output via ZMQ** -- finished sequences are placed on the `output_queue`. A dedicated output thread serializes them and sends them over a ZMQ `PUSH` socket back to the `CoreManager`, which receives them on a `PULL` socket and places them in `outputs_queue`.

10. **`InputOutputProcessor.postprocess()`** -- detokenizes completed sequences, computes TTFT (Time To First Token) and TPOT (Time Per Output Token), and returns structured output dictionaries.

---

## 4. Forward Context Pattern

ATOM uses a module-level global `ForwardContext` to pass metadata through CUDA graph boundaries without threading it as function parameters.

**Core dataclasses** (defined in `atom/utils/forward_context.py`):

- **`ForwardContext`** -- top-level container holding:
  - `attn_metadata` (`AttentionMetaData`) -- cumulative sequence lengths, block tables, slot mappings, and backend-specific metadata.
  - `context` (`Context`) -- positions, prefill flag, batch size, graph batch size, draft flag.
  - `dp_metadata` (`DPMetadata`) -- cross-DP-rank token counts and cumulative sums.
  - `spec_decode_metadata` (`SpecDecodeMetadata`) -- draft token IDs, target/bonus logits indices.
  - `kv_cache_data` (`dict[str, KVCacheTensor]`) -- per-layer KV cache tensor references.

- **`Context`** -- lightweight struct: `positions`, `is_prefill`, `batch_size`, `graph_bs`, `is_draft`.

- **`DPMetadata`** -- data parallel metadata with `num_tokens_across_dp()` (all-reduce), `max_tokens_across_dp`, and `chunked_sizes()` context manager.

**Global accessors:**

| Function | Purpose |
|---|---|
| `set_forward_context(attn_metadata, atom_config, context, ...)` | Set the global context before a forward pass |
| `get_forward_context()` | Retrieve the current context (used by attention backends) |
| `reset_forward_context()` | Clear after forward pass completes |
| `set_kv_cache_data(kv_cache_data)` | Register KV cache tensors at initialization |

This pattern enables stateless dispatch: attention backends and model operators call `get_forward_context()` to access metadata without requiring it as a function parameter, which is critical for CUDA graph compatibility.

---

## 5. Multi-Process Architecture

ATOM uses a multi-process design with ZMQ sockets for inter-process communication:

```
                        ┌──────────────────────────────────┐
                        │          LLMEngine               │
                        │  ┌────────────────────────────┐  │
                        │  │    CoreManager              │  │
                        │  │                             │  │
                        │  │  ROUTER ──────► DEALER      │  │
                        │  │  (input)        (per rank)  │  │
                        │  │                             │  │
                        │  │  PULL ◄─────── PUSH         │  │
                        │  │  (output)      (per rank)   │  │
                        │  └────────────────────────────┘  │
                        └──────────────────────────────────┘
                              │                    ▲
                     pickle   │                    │  pickle
                              ▼                    │
               ┌──────────────────────────────────────┐
               │         EngineCore (Process)          │
               │                                       │
               │  input_queue ──► busy_loop             │
               │                    │                   │
               │  ┌─────────────────▼───────────────┐  │
               │  │  AsyncIOProcManager              │  │
               │  │  ┌────────────────────────────┐  │  │
               │  │  │ ModelRunner (TP rank 0)     │  │  │
               │  │  │ ModelRunner (TP rank 1)     │  │  │
               │  │  │ ...                         │  │  │
               │  │  └────────────────────────────┘  │  │
               │  └──────────────────────────────────┘  │
               │                                       │
               │  Scheduler + BlockManager             │
               └──────────────────────────────────────┘
```

**Socket types:**

| Socket | Type | Direction | Purpose |
|---|---|---|---|
| Input | `ROUTER` (CoreManager) / `DEALER` (EngineCore) | CoreManager -> EngineCore | Send requests and control commands |
| Output | `PUSH` (EngineCore) / `PULL` (CoreManager) | EngineCore -> CoreManager | Return finished sequences and stream outputs |

**Process hierarchy:**

- **`CoreManager`** spawns one `EngineCore` process per DP rank using `multiprocessing.Process`.
- Each **`EngineCore`** creates an `AsyncIOProcManager`, which in turn spawns one subprocess per TP rank.
- Each **`ModelRunner`** subprocess initializes AITER's distributed environment via `init_dist_env()` from AITER, setting up NCCL communication across TP ranks.

**Data-parallel variant** (`DPEngineCoreProc`):

When `data_parallel_size > 1`, each EngineCore process is a `DPEngineCoreProc` that synchronizes with other DP ranks via `torch.distributed.all_reduce` on a Gloo process group. The `busy_loop()` override ensures all DP ranks stay in lockstep: if one rank has a prefill batch while another does not, the idle rank executes a dummy prefill (`dummy_prefill_execution()`) to keep NCCL collectives synchronized.

---

## 6. Sequence Lifecycle

The `Sequence` class (in `atom/model_engine/sequence.py`) is the central data structure tracking a single request through the engine.

**Key fields:**

| Field | Type | Purpose |
|---|---|---|
| `id` | `int` | Auto-incrementing unique identifier |
| `token_ids` | `list[int]` | Full token sequence (prompt + generated) |
| `num_prompt_tokens` | `int` | Length of the original prompt |
| `num_tokens` | `int` (property) | Total length including generated tokens |
| `block_table` | `list[int]` | KV cache block IDs allocated to this sequence |
| `status` | `SequenceStatus` | Current lifecycle state |
| `type` | `SequenceType` | Current execution type |
| `temperature` | `float` | Sampling temperature |
| `max_tokens` | `int` | Maximum completion length |
| `arrive_time` | `float` | Timestamp when request entered the system |
| `first_token_time` | `float` | Timestamp of first generated token (for TTFT) |
| `leave_time` | `float` | Timestamp when request finished (for TPOT) |
| `spec_token_ids` | `list[int]` | Speculative/draft token IDs for MTP |
| `stream_callback` | `Callable` | Optional per-token streaming callback |

**Status transitions:**

```
WAITING ──(scheduled for prefill)──► RUNNING ──(stop condition met)──► FINISHED
   ▲                                    │
   └────────(preempted by scheduler)────┘
```

- `SequenceStatus.WAITING` -- queued in the scheduler's waiting deque, awaiting block allocation.
- `SequenceStatus.RUNNING` -- actively being processed (prefill or decode).
- `SequenceStatus.FINISHED` -- stop condition met (EOS, stop token, stop sequence, or `max_tokens`). Blocks are deallocated.
- `SequenceStatus.EXIT_ENGINE` -- sentinel status used to signal engine shutdown.

**Execution types:**

- `SequenceType.DUMMY` -- initial state before scheduling.
- `SequenceType.PREFILL` -- prompt processing phase (all prompt tokens in one batch).
- `SequenceType.DECODE` -- autoregressive token generation (one or more tokens per step with MTP).

---

## Source Files

| File | Description |
|------|-------------|
| `atom/model_engine/llm_engine.py` | `LLMEngine` user-facing API, `InputOutputProcessor` for tokenization/detokenization and TTFT/TPOT statistics |
| `atom/model_engine/engine_core.py` | `EngineCore` main execution loop, `DPEngineCoreProc` data-parallel variant, `EngineCoreRequestType` message protocol |
| `atom/model_engine/engine_core_mgr.py` | `CoreManager` ZMQ orchestration, process launching, round-robin DP dispatch |
| `atom/model_engine/model_runner.py` | `ModelRunner` per-GPU execution (model loading, CUDA graph capture, forward pass), `tokenIDProcessor` deferred output handling |
| `atom/model_engine/scheduler.py` | `Scheduler` prefill-first scheduling, `ScheduledBatch` batch descriptor, `ScheduledBatchOutput` forward results |
| `atom/model_engine/sequence.py` | `Sequence` request state, `SequenceStatus` and `SequenceType` enums |
| `atom/model_engine/block_manager.py` | `BlockManager` KV cache block allocation with optional prefix caching |
| `atom/model_engine/request.py` | `RequestOutput` dataclass for streaming callbacks |
| `atom/model_engine/async_proc.py` | `AsyncIOProcManager` and `AsyncIOProc` for spawning and managing ModelRunner subprocesses |
| `atom/utils/forward_context.py` | `ForwardContext`, `Context`, `DPMetadata`, `SpecDecodeMetadata`, `AttentionMetaData` dataclasses and global accessors |
| `atom/config.py` | `Config` master configuration, `ParallelConfig`, `CompilationConfig`, `QuantizationConfig`, `SpeculativeConfig`, `KVCacheTensor` |
