# ATOM architecture guide

## Quick reference

| Class | Import | Purpose |
|-------|--------|---------|
| `LLMEngine` | `from atom.model_engine.llm_engine import LLMEngine` | User-facing inference API |
| `InputOutputProcessor` | `from atom.model_engine.llm_engine import InputOutputProcessor` | Tokenize/detokenize, TTFT/TPOT stats |
| `CoreManager` | `from atom.model_engine.engine_core_mgr import CoreManager` | Multi-process orchestration via ZMQ |
| `EngineCore` | `from atom.model_engine.engine_core import EngineCore` | Per-process engine loop |
| `DPEngineCoreProc` | `from atom.model_engine.engine_core import DPEngineCoreProc` | Data-parallel engine core variant |
| `ModelRunner` | `from atom.model_engine.model_runner import ModelRunner` | Per-GPU model execution |
| `Scheduler` | `from atom.model_engine.scheduler import Scheduler` | Prefill-first request scheduling |
| `BlockManager` | `from atom.model_engine.block_manager import BlockManager` | KV cache block allocation |
| `Sequence` | `from atom.model_engine.sequence import Sequence` | Request state and token tracking |
| `ForwardContext` | `from atom.utils.forward_context import ForwardContext` | Global forward pass metadata |
| `Config` | `from atom.config import Config` | Master configuration dataclass |

## System overview

ATOM (AiTer Optimized Model) is AMD's lightweight LLM inference engine, inspired by vLLM's architecture and built on the [AITER](https://github.com/ROCm/aiter) kernel library for ROCm/HIP GPUs.

Key design principles:

- **Multi-process architecture** — each engine core runs in its own process, with ZMQ-based IPC connecting the user-facing API to one or more GPU workers.
- **AITER-native execution** — model forward passes use AITER's optimized attention, MoE, sampling, and communication kernels rather than generic PyTorch operators.
- **CUDA graph acceleration** — decode batches are captured into CUDA graphs for replay, eliminating per-step kernel launch overhead.
- **Prefill-first scheduling** — the scheduler prioritizes prompt prefills before decode steps, following vLLM's continuous batching strategy.
- **Speculative decoding** — optional EAGLE/MTP (Multi-Token Prediction) draft models propose tokens that are verified via rejection sampling.

## Component architecture

```text
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

**Supported model architectures** (registered in `support_model_arch_dict`, a module-level dict in `atom/models/registry.py`):

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

## Request lifecycle

A request flows through the system in ten steps:

1. **`LLMEngine.add_request()` / `generate()`** — you submit a list of prompts (strings or pre-tokenized token IDs) together with `SamplingParams`.

2. **`InputOutputProcessor.preprocess()`** — each prompt is tokenized via the HuggingFace tokenizer. A `Sequence` object is created to track the request's state, timing, and block allocation. `arrive_time` is recorded.

3. **`CoreManager.add_request()`** — the list of `Sequence` objects is serialized with `pickle` and sent over a ZMQ `ROUTER` socket. When multiple DP ranks are active, requests are routed by a configurable load-balancing strategy (`least_requests` by default; `round_robin`, `least_tokens` and `prefix_spread` also available).

4. **`EngineCore.process_input_sockets()`** — an I/O thread on the `EngineCore` process receives the serialized data on a ZMQ `DEALER` socket, deserializes it, and places the sequences into the `input_queue`.

5. **`EngineCore.busy_loop()`** — the main execution loop pulls from `input_queue` via `pull_and_process_input_queue()`, feeds new sequences into the scheduler, and repeatedly calls `_process_engine_step()` until all work is done.

6. **`Scheduler.schedule()`** — implements prefill-first scheduling. Waiting sequences are scheduled for prefill if they fit within `max_num_seqs` and `max_num_batched_tokens` and the `BlockManager` can allocate blocks. If no prefills are pending, running sequences are batched for decode. The scheduler returns a `ScheduledBatch` and the corresponding sequence map.

7. **`ModelRunner.forward()`** — executes the three-phase forward pass:
   - `prepare_model()` — assembles input IDs (handling deferred output from previous steps), builds attention metadata, and gathers sampling temperatures.
   - `run_model()` — runs the model forward. Prefill and large batches run eagerly; decode batches replay captured CUDA graphs. Returns logits and hidden states.
   - `postprocess()` — samples tokens (or runs rejection sampling for speculative decoding), prepares deferred output via `tokenIDProcessor`, and optionally proposes draft tokens through `EagleProposer`.

8. **`Scheduler.postprocess()`** — appends sampled tokens to each `Sequence`, records `first_token_time`, checks stop conditions (EOS, stop token IDs, stop token sequences, `max_tokens`), and moves finished sequences out of the running queue. The `BlockManager` deallocates blocks for finished sequences.

9. **Output via ZMQ** — finished sequences are placed on the `output_queue`. A dedicated output thread serializes them and sends them over a ZMQ `PUSH` socket back to the `CoreManager`, which receives them on a `PULL` socket and places them in `outputs_queue`.

10. **`InputOutputProcessor.postprocess()`** — detokenizes completed sequences, computes TTFT (Time To First Token) and TPOT (Time Per Output Token), and returns structured output dictionaries.

## Forward context pattern

ATOM uses a module-level global `ForwardContext` to pass metadata through CUDA graph boundaries without threading it as function parameters.

**Core dataclasses** (defined in `atom/utils/forward_context.py`):

- **`ForwardContext`** — top-level container holding:
  - `attn_metadata` (`AttentionMetaData`) — cumulative sequence lengths, block tables, slot mappings, and backend-specific metadata.
  - `context` (`Context`) — positions, prefill flag, batch size, graph batch size, draft flag.
  - `dp_metadata` (`DPMetadata`) — cross-DP-rank token counts and cumulative sums.
  - `spec_decode_metadata` (`SpecDecodeMetadata`) — draft token IDs, target/bonus logits indices.
  - `kv_cache_data` (`dict[str, KVCacheTensor]`) — per-layer KV cache tensor references.

- **`Context`** — lightweight struct: `positions`, `is_prefill`, `batch_size`, `running_bs`, `running_tokens`, `is_draft`. The last two are the step's DP-unified padded shape: sequences and hidden_states rows respectively.

- **`DPMetadata`** — data parallel metadata holding every rank's running tokens (`running_tokens_per_dp_rank`, a numpy array), with `gather_running_tokens()` (all-reduce, for a caller that knows only its own count), `max_tokens_across_dp`, and `get_sizes_across_dp()`.

**Global accessors:**

| Function | Purpose |
|---|---|
| `set_forward_context(attn_metadata, atom_config, context, running_tokens_per_dp_rank, ...)` | Set the global context before a forward pass |
| `get_forward_context()` | Retrieve the current context (used by attention backends) |
| `reset_forward_context()` | Clear after forward pass completes |
| `set_kv_cache_data(kv_cache_data)` | Register KV cache tensors at initialization |

This pattern enables stateless dispatch: attention backends and model operators call `get_forward_context()` to access metadata without requiring it as a function parameter, which is critical for CUDA graph compatibility.

## Model hooks (head output, deferred status, decode round)

Two switches of `ModelRunner` let a model take over part of a step; neither
is specific to one model, and a model that uses neither runs exactly as before.

- **The in-graph head output.** A model may give its own head output,
  `compute_head(hidden_states)`, resolved once at install (`ModelRunner.compute_head`):
  the forward's logits, or a speculative verify's output token ids
  `[requests, num_spec_steps + 1]` and verdict `[3, requests]` (accepted,
  rejected, anchor row: the rejection sampler's layout), computed inside its
  forward; None leaves the head to the logits. `ModelRunner._head_output` takes
  it after an eager forward and records it at the tail of a decode graph's
  capture, as `logits_in_graph` records the logits at TP 1, so a replay returns
  it. `ModelRunner._step_head` keeps a verify's ids and verdict only where they
  stand for the step (`_head_ids_serve`: nothing reads the logits but each
  row's pick -- no logprobs, structured output or required channel, exact or
  forced acceptance -- every request verifies the full draft span, and the rows
  are all greedy or all sampled from temperature and top-p alone), else it
  computes the logits; `postprocess` then branches once, on whether the head
  output carries ids. As on every other step, the output ids are broadcast over
  TP and each request's next id (the next forward's ids, the drafter's anchor)
  is gathered from the broadcast ids. ATOM's own greedy argmax and rejection
  verdict could move into the graph the same way.
- **The deferrable status wait.** An attention builder may say a step consumes
  the previous verify's verdict on the device rather than its host-side status
  (`AttentionMetadataBuilder.consumes_device_verdict`, asked once a step in
  `prepare_model`): `tokenIDProcessor.prepare_input_ids` then builds the step
  from zero rejected / bonus counts and the runner takes the status
  (`take_deferred_spec_status`) right after the target's launch, where
  `postprocess` needs it.

DeepSeek-V4.1's metadata builder also takes a model's decode round
(`atom/model_engine/model_hooks.py`, `DecodeRound`, handed over by the model's
install: `DeepseekV41MetadataBuilder.use_decode_round`) and asks it at its own
points of a step: `consumes_device_verdict` (its answer to the runner),
`service` before the layout, `laid_out` after it and `committed` before the
speculative commit (at the anchors the runner commits). `DecodeRound.service`
is the step's one routing decision, `AttentionMetaData.served_by_round`
(`RoundService`): `NONE`, or `FORWARD` (the model's own forward runs every
layer, looks its Engram rows up itself and writes the DSpark draft's context
KV), `LEAN` (laid out without the compress plans and step planners nothing it
runs reads) and `DEVICE` (positions advanced on the device from the previous
verdict). The builder decides it once before the step's layout; the builder,
its cache, Engram staging (`EngramStep.served_by_round`), the DSpark proposer
and the model's forward all read it, so they agree on every step.

DeepSeek-V4.1's high-interactivity monokernels (`atom/models/deepseek_v41/mono/hi_*.py`,
`round/`, the FlyDSL runtime in `hi/`; `ATOM_DSV41_MONO_HI`, on by default where
`hi_gate` says they serve) use all three: `hi_dispatch` wraps the model
(`HiDecodeModel`, whose `compute_head` is the in-graph verify head and
post-verify, `round/post_verify.py`), and `round.HiRound` is its decode round
(the device round, `round/device_round.py`, consumes the device verdict).

## Multi-process architecture

ATOM uses a multi-process design with ZMQ sockets for inter-process communication:

```text
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
| Input | `ROUTER` (CoreManager) / `DEALER` (EngineCore) | CoreManager → EngineCore | Send requests and control commands |
| Control | `ROUTER` (CoreManager) / `DEALER` (EngineCore) | CoreManager → EngineCore | Serialized utility and collective-RPC commands |
| Output | `PUSH` (EngineCore) / `PULL` (CoreManager) | EngineCore → CoreManager | Return finished sequences and stream outputs |
| Worker RPC output | one `PUSH` / `PULL` pair per TP rank | ModelRunner → AsyncIOProcManager | One correlated collective-RPC reply per worker |

**Process hierarchy:**

- **`CoreManager`** spawns one `EngineCore` process per DP rank using `multiprocessing.Process`.
- Each **`EngineCore`** creates an `AsyncIOProcManager`, which in turn spawns one subprocess per TP rank.
- Each **`ModelRunner`** subprocess initializes AITER's distributed environment via `init_dist_env()` from AITER, setting up NCCL communication across TP ranks.

**Data-parallel variant** (`DPEngineCoreProc`):

When `data_parallel_size > 1`, each EngineCore process is a `DPEngineCoreProc` that synchronizes with other DP ranks via `torch.distributed.all_reduce` on a Gloo process group. The `busy_loop()` override ensures all DP ranks stay in lockstep: a rank with no request runs a step with none (`dummy_execution()`) in the group's phase, to keep NCCL collectives synchronized.

### Collective worker RPC

`AsyncLLMEngine.collective_rpc()` sends a small control call to every DP
engine and every TP worker, returning `RpcResult` entries in DP-major,
TP-rank order. The wire types and request-id router live in
`atom/model_engine/collective_rpc.py`; worker dispatch and per-rank reply
sockets live in `atom/model_engine/async_proc.py`; capability aggregation is
split between `atom/model_engine/capabilities.py` and
`atom/rollout/capabilities.py`.

Every call carries a request id and a finite deadline. Both ordinary utility
replies and collective replies are routed to the caller holding that id; a
late reply is dropped rather than becoming the next utility caller's answer.
Worker exceptions are returned as `RpcResult.error`, preserving successful
rank results as well.

The API is synchronous and intended for control messages, not bulk tensor
transfer. A target returns plain data -- `None`, `bool`, `int`, `float`, `str`
and `bytes`, in lists, tuples and dicts -- since its reply is unpickled in
EngineCore and again in the API server; anything else, a tensor included, is
returned as that rank's error. Only the runner methods listed in
`COLLECTIVE_RPC_METHODS` run on this generic path, and they are exactly what
`get_worker_capabilities` advertises: the rest of a runner -- weight updates,
memory release/resume, its profiler, KV allocation, graph capture, dummy
forwards -- has EngineCore state around it that a direct call would bypass.
Pipeline-parallel and prefill/decode-disaggregated engines are also rejected
because their engine processes are not one-to-one DP ranks.
RPC targets must not perform cross-DP collectives: each EngineCore dequeues
control work independently, so another DP engine may already be waiting in the
busy loop's state all-reduce.

Direct, SHM and IPC weight updates are fenced sessions. The first bucket stops
serving; CoreManager clears the fence only after every DP engine completes the
final bucket. Built-in senders abort from ``finally`` if iteration stops early,
and asynchronous weight-update utility sends are rejected because they have no
owner to finish or abort them. ``AsyncLLMEngine.get_weight_sync_status()``
reports a fence, an in-progress session and its last failure.

Every update carries the id of the sync it belongs to: one per
``weight_sync_session()`` until that sync finishes, and one per update sent
outside a session. A session whose update failed sends nothing more, and an
engine refuses the remaining updates and the finish of a failed or aborted
sync, so the rest of a failed sync cannot commit as a new one. Recovering takes
a new, complete sync. Weights may be resumed while fenced, since after
``sleep(level=2)`` only a resume gives released parameters storage again; the
fence keeps serving off until that complete sync commits.

## Sequence lifecycle

The `Sequence` class (in `atom/model_engine/sequence.py`) is the central data structure tracking a single request through the engine.

**Key fields:**

| Field | Type | Purpose |
|---|---|---|
| `id` | `int` | Auto-incrementing unique identifier |
| `token_ids` | `list[int]` | Full token sequence (prompt + generated) |
| `num_prompt_tokens` | `int` | Length of the original prompt |
| `num_tokens` | `int` (property) | Total length including generated tokens |
| `block_table` | `list[int]` | KV cache block IDs allocated to this sequence |
| `state_slots` | `list[int]` | Every per-request stateful-attention slot the sequence holds: `[0]` committed, `[1:]` speculation rollback (Qwen3-Next / Qwen3.5 GDN layers, Kimi-K3 KDA, the DeepSeek-V4 compressor ring; future stateful attentions plug in via the same mechanism). Empty if unallocated or not a stateful-attention model |
| `state_slot` | `int` (property) | `state_slots[0]`, or `-1` — the slot the forward reads and writes |
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

```text
WAITING ──(scheduled for prefill)──► RUNNING ──(stop condition met)──► FINISHED
   ▲                                    │
   └────────(preempted by scheduler)────┘
```

- `SequenceStatus.WAITING` — queued in the scheduler's waiting deque, awaiting block allocation.
- `SequenceStatus.RUNNING` — actively being processed (prefill or decode).
- `SequenceStatus.FINISHED` — stop condition met (EOS, stop token, stop sequence, or `max_tokens`). Blocks are deallocated.
- `SequenceStatus.EXIT_ENGINE` — sentinel status used to signal engine shutdown.

**Execution types:**

- `SequenceType.DUMMY` — initial state before scheduling.
- `SequenceType.PREFILL` — prompt processing phase (all prompt tokens in one batch).
- `SequenceType.DECODE` — autoregressive token generation (one or more tokens per step with MTP).

## Speculative decoding (MTP / EAGLE)

ATOM supports speculative decoding via `EagleProposer` (in `atom/spec_decode/eagle.py`), which uses Multi-Token Prediction (MTP) draft models to propose candidate tokens that are verified by the target model through rejection sampling.

### Supported MTP architectures

The `support_eagle_model_arch_dict` maps HuggingFace architecture keys to draft model implementations:

| Architecture key | Implementation | Used by |
|---|---|---|
| `DeepSeekMTPModel` | `atom.models.deepseek_mtp.DeepSeekMTP` | DeepSeek-R1, DeepSeek-V3 |
| `Qwen3NextMTPModel` | `atom.models.qwen3_next_mtp.Qwen3NextMTP` | Qwen3-Next |
| `Qwen3_5MTPModel` | `atom.models.qwen3_5_mtp.Qwen3_5MTP` | Qwen3.5 |

### Weight sharing

`EagleProposer.load_model()` shares weights between the draft and target models to save memory. It uses `_share_if_not_loaded()`, which checks whether a parameter key exists in the `loaded_weights_record` set returned by the model loader. If the key is absent (meaning the checkpoint did not contain that weight), the draft module's attribute is replaced with a reference to the target model's corresponding parameter. This avoids comparing tensor values and instead relies on the loader's bookkeeping of which weights were actually present in the checkpoint.

Two sharing patterns are handled:

- **Per-layer `shared_head.head`** (DeepSeek MTP) — each MTP layer has a `shared_head` whose `head` weight is shared with `target_base.lm_head` if not loaded from the checkpoint.
- **Top-level `lm_head`** (Qwen3.5 / Qwen3-Next MTP) — the draft model's `lm_head` is shared with the target if not loaded.

The `embed_tokens` layer is always shared when shapes match and pipeline parallelism is not used.

### Propose loop: metadata is the backend's, not the proposer's

`EagleProposer._propose()` (the flavor body `Drafter.propose` calls) iterates `mtp_k` draft steps. Every step calls
`runner.attn_metadata_builder.prepare_mtp_decode()` and merges the dict it
returns into `attn_metadata`, so *which* fields a step updates — `kv_indptr`
and `kv_indices` for MLA, `block_tables` and `context_lens` for MHA — is the
backend's answer and not a branch here. The proposer passes
`num_reject_tokens` on `i == 0` only, which is how the previous step's
rejected speculative tokens are taken back out.

One thing does branch, on a builder capability rather than a model family:
`fuse_mtp_decode_position_update`. Where a builder sets it, the per-step
position bump and the `context_lens` increment ride into its kernel through
`update_context_lens` / `positions_out`; where it does not, `_propose()` does
both itself before the call.

### `prepare_mtp_decode()`

Each attention backend provides its own `prepare_mtp_decode()` implementation:

- **`AiterMLAAttnMetadataBuilder`** (`atom/model_ops/attentions/aiter_mla.py`) — for MLA attention, increments `kv_indptr` by the cumulative query sequence lengths and regenerates `kv_indices` via `kv_indices_generate_triton`. Supports incremental updates (`only_update=True`) for persistent worker buffers used by AITER's paged attention.
- **`GDNAttnMetadataBuilder`** (`atom/model_ops/attentions/gdn_attn.py`) — for GDN hybrid models with paged KV cache, regenerates `kv_indices` for the new `max_seqlen_k` after each draft token. The `kv_indptr` (block count) stays unchanged since the page table is stable within a draft step. The `only_update` and `num_reject_tokens` parameters are unused — GDN always does a full `kv_indices` regeneration.

## Source files

| File | Description |
|------|-------------|
| `atom/model_engine/llm_engine.py` | `LLMEngine` user-facing API, `InputOutputProcessor` for tokenization/detokenization and TTFT/TPOT statistics |
| `atom/model_engine/engine_core.py` | `EngineCore` main execution loop, `DPEngineCoreProc` data-parallel variant, `EngineCoreRequestType` message protocol |
| `atom/model_engine/engine_core_mgr.py` | `CoreManager` ZMQ orchestration, process launching, load-balanced DP dispatch |
| `atom/model_engine/model_runner.py` | `ModelRunner` per-GPU execution (model loading, CUDA graph capture, forward pass), `tokenIDProcessor` deferred output handling |
| `atom/model_engine/model_hooks.py` | `DecodeRound`, the decode round a model's install may give DeepSeek-V4.1's metadata builder |
| `atom/model_engine/dp_routing/prefix_spread.py` | `PrefixSpreadRouter`: the `--dp-load-balance prefix_spread` prefix index and placement rule `CoreManager` calls under its load-balancing lock |
| `atom/model_engine/dp_routing/policies.py` | Stateless, session-affinity and prefix-spread placement policies with common request lifecycle hooks |
| `atom/model_engine/dp_routing/route.py` | Placement rank, prefix credit, moved tokens and routing counter/label names |
| `atom/model_engine/dp_routing/topology.py` | Effective prefix-routing topology used by the manager and MP deployment checks |
| `atom/model_engine/scheduler.py` | `Scheduler` prefill-first scheduling, `ScheduledBatch` batch descriptor, `ScheduledBatchOutput` forward results |
| `atom/model_engine/engine_stats.py` | `EngineStats` — MTP acceptance, prefix-cache hits, and the periodic engine-status line, each on its own cadence |
| `atom/model_engine/sequence.py` | `Sequence` request state, `SequenceStatus` and `SequenceType` enums |
| `atom/model_engine/block_manager.py` | `BlockManager` KV cache block allocation with optional prefix caching |
| `atom/model_engine/request.py` | `RequestOutput` dataclass for streaming callbacks |
| `atom/model_engine/async_proc.py` | `AsyncIOProcManager` and `AsyncIOProc` for spawning and managing ModelRunner subprocesses |
| `atom/utils/forward_context.py` | `ForwardContext`, `Context`, `DPMetadata`, `SpecDecodeMetadata`, `AttentionMetaData` dataclasses and global accessors |
| `atom/config.py` | `Config` master configuration, `ParallelConfig`, `CompilationConfig`, `QuantizationConfig`, `SpeculativeConfig`, `KVCacheTensor` |
