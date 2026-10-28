# CLAUDE.md — ATOM

ATOM is a lightweight LLM inference engine built on AITER GPU kernels for the AMD ROCm platform.

## Build & Test & Lint

```bash
pip install -e .                          # editable install
python -m pytest tests/                   # all tests (no GPU needed — mocks AITER and torch.cuda)
black . && ruff check .                   # format + lint (CI enforced)
```

## Running

```bash
# OpenAI-compatible serving
python -m atom.entrypoints.openai_server --model <model> --kv-cache-dtype fp8 -tp 8

# Offline inference
python -m atom.examples.simple_inference --model <model> --kv-cache-dtype fp8
```

- Accuracy validation: see `/ci-pr-guide` for `lm_eval` setup and CI thresholds
- Performance benchmark: see `/benchmark-guide` for full parameters

## Architecture (Quick Index)

```
openai_server.py → LLMEngine → CoreManager → EngineCore → Scheduler → ModelRunner
                   (tokenizer)   (ZMQ IPC)    (inference    (batching)  (forward pass,
                                               loop)                    KV cache, CUDAGraphs)
```

ATOM also has a **diffusion subsystem** (`atom/diffusion/`) for video+audio
generation, which is a sibling of `atom/model_engine/` rather than an extension
of it — no KV cache, one fixed N-step denoise loop per job, four heterogeneous
networks, and sequence parallelism across a *single* request:

```
diffusion_server.py → DiffusionEngine → DiffusionCoreManager → DiffusionEngineCore → PipelineRunner
   (/v1/videos)        (JobScheduler)      (ZMQ, one PUSH/rank)    (per-GPU worker)     (ComposedPipeline)
```

Key entry points:
- Server: `atom/entrypoints/openai_server.py`
- Engine: `atom/model_engine/llm_engine.py` → `engine_core.py` → `scheduler.py` → `model_runner.py`
- Models: `atom/models/` — registered in `atom/models/registry.py:support_model_arch_dict`
- Ops: `atom/model_ops/` — AITER kernel wrappers (linear, attention, fused_moe)
- Config: `atom/config.py` (Config, KVCacheConfig, CompilationConfig)
- Env vars: `atom/utils/envs.py` (all `ATOM_*` variable definitions)
- Pipeline parallel: `atom/model_engine/pp_engine_core.py` (head / downstream stage
  loops) over `atom/distributed/pp_transport.py`; dynamic chunked PP prefill in
  `atom/model_engine/dynamic_chunking.py` (latency model, pure) and
  `dynamic_chunking_worker.py` (startup sweep and runtime timing in the runner),
  sized by `Scheduler._dynamic_chunk_limit`
- Mono decode (fused per-layer decode kernels): the shared mechanisms in `atom/mono/`
  (`runtime/`: TP consensus and a runner's bind-then-agree construction
  (`consensus.bind_agreed`), peer memory and its owned buffer (`lifecycle`),
  the step's mailbox zeroing and peer fence, one FlyDSL launch
  (`mailboxes.StepMailboxes.begin_step` -> `step_begin`), per-width builds (`widths`), the routing wrapper's
  lazy runner (`route`), debug wait reports, kernel argument tables, compile-only
  builds and their traced hand-offs (`compile.trace` / `compile.check_traced`),
  the refusals every model shares (`deployment`), check-mode
  comparison (`compare`); `plan/`: the execution model and task placement, build
  keys, a TP rank's share of the widths (`shard`: a model's `Dims` and its TP
  refusal), mailbox region layout (`layout`), int / traced arithmetic (`arith`)
  and the hand-off contract a trace is held to (`check.check`); `device/`: the tagged
  mailbox and its `publish`, the TP group's peer bases and all-reduce sum
  (`ranks`), FP8 / MX codes, FP4 tile loads and the scaled MFMA (`mx`) and wave
  / math primitives), a model's
  kernels in `atom/models/<model>/mono/kernels/` (MiniMax-M3; DeepSeek-V4.1 and
  Kimi-K3, both wrapped after load through
  `models.registry.mono_decode_installers`; V4.1's per-step cell decisions live in
  `deepseek_v41/mono/{index,attention,moe}_plan.py`, each called by both the
  kernel and its CPU I3 test. Kimi-K3 runs K1 (the attention seam through
  the KDA recurrence) and K2 (`o_proj` + the latent MoE) on its KDA layers, and
  only on a uniform speculative verify of at most 16 rows. The MLA layers keep
  their original paged attention but also run K2; an MLA layer that cannot bind
  K2 stays on the production path alone. Which steps it serves and how a served step is
  keyed live in `kimi_k3/mono/plan.py`, apart from `runner.py` so the CPU test
  gate can reach them without AITER or FlyDSL. GLM-5.2's single `kernels/kernel.py` covers the
  whole layer, `kernels/op.py` is its host wrapper, and `kernels/geometry.py` /
  `kernels/geometry_config.py` hold its launch geometry and its own `LayerConfig`.
  What is NOT GLM-5.2's sits in `atom/mono/`: the storage vocabulary and host-side
  weight preparation in `mono/weights/` (`config` the enum vocabulary and
  `LayerConfig`, `formats` MX conversion, `packing`, `layer` the `LayerWeights`
  ABI), ATOM's MXFP4 index arithmetic in `plan/tiles.py`, and the HIP IPC wrapper
  in `runtime/hip_ipc.py`. `runtime/peer_memory.py` now holds BOTH peer-buffer
  implementations side by side — `PeerBuffer` (aiter's heap) and
  `SymmetricPeerBuffer` (HIP directly, GLM-5.2's) — because they are different
  implementations rather than two copies, and merging them is a GPU behaviour
  change, not a move. Where a model ships a
  `mono/contract.py` with a `__main__` (DeepSeek-V4.1, MiniMax-M3), check its
  hand-offs without a GPU with `FLYDSL_RUNTIME_ENABLE_CACHE=0 ARCH=gfx950
  python -m atom.models.<model>.mono.contract`; GLM-5.2's `contract.py` is the
  per-step input check only, and Kimi-K3 ships neither it nor a `check.py`
  yet)
- Diffusion: `atom/diffusion/` — framework at the top level (`pipeline.py`,
  `attention.py`, `ulysses.py`), engine in `engine/`, server in `entrypoints/`,
  and one package per model in `models/<family>/`. Extras: `pip install -e ".[diffusion]"`

## Critical Rules

- **Change `@support_torch_compile` decorated model files with care** — edits are allowed, but debug/instrumentation code inside traced functions breaks Dynamo tracing even with `--enforce-eager`. Instrument at call sites instead (e.g., `ModelRunner.run_model()`, `EagleProposer.propose()`, `DSparkProposer.propose()`), know which functions are traced before editing, and run the compiled path afterwards. `deepseek_v4_dspark.py` carries a COMPILE BOUNDARY block at the top naming exactly which functions are traced — read it before touching that file
- **Multiprocessing must use `spawn`** — `fork` causes CUDA re-initialization crashes
- **Set `AITER_LOG_LEVEL=WARNING` before starting server** — suppresses aiter kernel log flooding
- **Clear compile cache before server restart:** `rm -rf /root/.cache/atom/*` — stale cache causes silent failures after code changes
- **Verify server with GPU, not just HTTP:** `curl /health` can return OK even when model is not loaded. Always check `rocm-smi --showmemuse` (VRAM% > 0) to confirm
- **On any server/GPU error, run `/debug-guide` first** — do not blindly retry
- **Fix-then-sweep**: after fixing a bug, immediately grep for the same pattern across the codebase and fix all occurrences in one pass
- **Name-matches-function**: variable, function, and file names must accurately describe what they do. When behavior changes, rename immediately — stale names mislead future readers

## Key Development Patterns

- **Adding a model**: see `/add-model` for full guide
- **Model reuse**: DeepSeek V3/V3.2/GLM-5 share `deepseek_v2.py`; MTP models in `deepseek_mtp.py`, `qwen3_next_mtp.py`, and `qwen3_5_mtp.py`
- **Compilation levels**: `--level` 0=eager, 1=torch.compile, 2=dynamo once, 3=piecewise compile (default); CUDA graphs are `--cudagraph-mode`'s, independent of the level

## Dependencies

- **AITER** (`from aiter import ...`) — GPU compute kernels
- **MORI** — MoE expert-parallel all-to-all communication
- **RCCL** — collective communication primitives

## Detailed Guides

| Topic | Source |
|-------|--------|
| Architecture | `docs/architecture_guide.md` |
| Environment variables | `docs/environment_variables.md` |
| Compilation & CUDAGraph | `docs/compilation_cudagraph_guide.md` |
| Model support | `docs/model_support_guide.md` |
| Model ops (AITER kernels) | `docs/model_ops_guide.md` |
| Scheduling & KV cache | `docs/scheduling_kv_cache_guide.md` |
| Serving & benchmarking | `docs/serving_benchmarking_guide.md` |
| Configuration | `docs/configuration_guide.md` |
| Distributed | `docs/distributed_guide.md` |
| Dynamic chunked pipeline parallel | `docs/dynamic_chunking_guide.md` |
| Metrics & observability | `docs/metrics_guide.md` |
| CI/PR workflow | `/ci-pr-guide` |
| Performance benchmark | `/benchmark-guide` |
| Debugging | `/debug-guide` |
| Adding a model | `/add-model` |
| MiniMax-H3 (video+audio diffusion) | `recipes/MiniMax-H3.md` |
| DeepSeek-V4.1-Flash | `recipes/DeepSeek-V4.1-Flash.md` |
