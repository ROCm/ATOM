# ATOM Compilation & CUDA Graphs Guide

> **Quick Reference**
>
> | Concept | Key Class / Enum | Import |
> |---------|-----------------|--------|
> | Compilation Levels | `CompilationLevel` | `from atom.config import CompilationLevel` |
> | Compilation Config | `CompilationConfig` | `from atom.config import CompilationConfig` |
> | CUDA Graph Modes | `CUDAGraphMode` | `from atom.config import CUDAGraphMode` |
> | CUDA Graph Wrapper | `CUDAGraphWrapper` | `from atom.utils.cuda_graph import CUDAGraphWrapper` |
> | Forward Context | `ForwardContext` | `from atom.utils.forward_context import ForwardContext` |
> | Compiler Backend | `VllmBackend` | `from atom.utils.backends import VllmBackend` |
> | Compiler Manager | `CompilerManager` | `from atom.utils.backends import CompilerManager` |
> | Compiler Interface | `CompilerInterface` | `from atom.utils.compiler_inferface import CompilerInterface` |
> | Inductor Adaptor | `InductorAdaptor` | `from atom.utils.compiler_inferface import InductorAdaptor` |
> | Piecewise Backend | `PiecewiseBackend` | `from atom.utils.cuda_piecewise_backend import PiecewiseBackend` |
> | Compile Decorator | `@support_torch_compile` | `from atom.utils.decorators import support_torch_compile` |
> | Custom Op Registration | `direct_register_custom_op` | `from atom.utils.custom_register import direct_register_custom_op` |
>
> **Compilation Levels at a Glance**
>
> | Level | Name | Behavior |
> |-------|------|----------|
> | 0 | `NO_COMPILATION` | Pure eager execution, no `torch.compile` |
> | 1 | `DYNAMO_AS_IS` | `torch.compile` with `backend="eager"` |
> | 2 | `DYNAMO_ONCE` | `torch.compile` with Inductor |
> | 3 | `PIECEWISE` | Piecewise compilation with CUDA graph capture (production default) |
>
> **CUDA Graph Modes at a Glance**
>
> | Mode | Value | Behavior |
> |------|-------|----------|
> | `NONE` | `0` | No graph capture |
> | `PIECEWISE` | `1` | Per-subgraph capture (default for level 3) |
> | `FULL` | `2` | Whole-model capture |
> | `FULL_DECODE_ONLY` | `(FULL, NONE)` | Full for decode, none for mixed batches |
> | `FULL_AND_PIECEWISE` | `(FULL, PIECEWISE)` | Full for decode, piecewise for prefill |

---

## 1. Compilation Levels

ATOM provides four compilation levels via the `CompilationLevel` class in `atom/config.py`. The level is set through `CompilationConfig.level` and controls how `torch.compile` is applied to the model.

### Level 0 -- NO_COMPILATION

No `torch.compile` is applied. The model runs in pure eager mode. This is the simplest mode and is useful for debugging or when using models that are incompatible with `torch.compile`.

When `level=0`, the `@support_torch_compile` decorator sets `self.do_not_compile = True` and the model's `__call__` method bypasses compilation entirely, calling `self.forward()` directly.

### Level 1 -- DYNAMO_AS_IS

Uses `torch.compile` with `backend="eager"` and `fullgraph=True`. This runs Dynamo's bytecode analysis and graph capture but does not apply any compiler optimizations. It is useful as a quick check to verify that a model is compatible with Dynamo's tracing.

Like level 0, `DYNAMO_AS_IS` causes the decorator to set `self.do_not_compile = True`, since the model runner (rather than the decorator) handles the compilation at this level.

### Level 2 -- DYNAMO_ONCE

Uses `torch.compile` with the Inductor backend. The model graph is traced by Dynamo and compiled once through Inductor for optimized GPU kernel generation. The `@support_torch_compile` decorator's custom dispatcher is activated when `compilation_level >= DYNAMO_ONCE`, allowing compiled bytecode to be dispatched directly after the first compilation without repeated guard evaluation.

### Level 3 -- PIECEWISE (Production Default)

The most advanced level. When `Config.__post_init__` detects `level == PIECEWISE`, it:

1. Calls `CompilationConfig.set_splitting_ops_for_v1()` to configure the splitting operations (default: `["aiter.unified_attention_with_output", "aiter.mla_attention"]`).
2. Calls `Config._set_cudagraph_sizes()` to compute the graph batch sizes.
3. Sets `cudagraph_mode = CUDAGraphMode.PIECEWISE`.
4. Calls `CompilationConfig.init_with_cudagraph_sizes()` to finalize compile sizes.

The `VllmBackend` is then used as the `torch.compile` backend. It splits the model graph into subgraphs at the splitting operations and compiles each subgraph independently via `PiecewiseBackend`.

---

## 2. CUDA Graph Modes

The `CUDAGraphMode` enum in `atom/config.py` controls how CUDA graphs are captured and replayed. CUDA graphs record a sequence of GPU operations and replay them with minimal CPU overhead, which is critical for low-latency decode steps.

### NONE (value: 0)

No CUDA graph capture or replay. Every forward pass launches kernels individually. This mode is used during profiling, warmup, or when CUDA graphs are not supported.

### PIECEWISE (value: 1)

The default mode for level 3 compilation. CUDA graphs are captured per subgraph (one for each piecewise-compiled region). Attention operations, which are split out by `splitting_ops`, run outside CUDA graphs because they may need dynamic metadata that changes between steps.

The `CUDAGraphWrapper` class wraps each subgraph with `runtime_mode=CUDAGraphMode.PIECEWISE` for capture and replay.

### FULL (value: 2)

The entire model forward pass is captured as a single CUDA graph. This is suitable for small models or workloads with small, uniform batch sizes. Not all attention backends support full CUDA graph capture.

### FULL_DECODE_ONLY (value: (FULL, NONE))

A tuple mode that applies different strategies to different batch types:
- **Decode batches**: Captured with full CUDA graphs.
- **Mixed prefill-decode batches**: Run without CUDA graphs.

This is useful for prefill/decode disaggregated (P/D) setups where decode latency matters more than prefill performance.

### FULL_AND_PIECEWISE (value: (FULL, PIECEWISE))

A tuple mode combining both strategies:
- **Decode batches**: Captured with full CUDA graphs.
- **Prefill and mixed batches**: Captured with piecewise CUDA graphs.

This is described in the code as "the most performant mode for most models."

### Helper Methods

The `CUDAGraphMode` enum provides several helper methods for runtime dispatch:

| Method | Returns | Purpose |
|--------|---------|---------|
| `decode_mode()` | `CUDAGraphMode` | Returns the mode to use for decode batches. For tuple modes, returns the first element. |
| `mixed_mode()` | `CUDAGraphMode` | Returns the mode to use for mixed batches. For tuple modes, returns the second element. |
| `separate_routine()` | `bool` | Returns `True` if the mode is a tuple (different strategies for decode vs. mixed). |
| `has_full_cudagraphs()` | `bool` | Returns `True` if any part of the mode uses `FULL` capture. |
| `requires_piecewise_compilation()` | `bool` | Returns `True` if either decode or mixed mode uses `PIECEWISE`. |
| `max_cudagraph_mode()` | `CUDAGraphMode` | Returns the highest-valued mode across both decode and mixed modes. |

---

## 3. CUDA Graph Capture

CUDA graph capture is handled by `ModelRunner.capture_cudagraph()` in `atom/model_engine/model_runner.py`. This method is called at startup (under `@torch.inference_mode()`) to pre-capture graphs for a set of batch sizes.

### Capture Flow

```
capture_cudagraph()
  |
  +-- Determine graph_bs list
  |     |-- If cudagraph_capture_sizes is set: use directly
  |     |-- If cuda_graph_sizes has 1 value N: [1, 2, 4, 8, 16, 32, ..., N]
  |     +-- If cuda_graph_sizes has >1 values: use the provided list
  |
  +-- Sort graph_bs in descending order (largest batch first)
  |
  +-- Assert max batch size <= max_num_seqs
  |
  +-- Initialize graph storage: self.graphs = dict()
  |
  +-- For each batch size bs (with progress bar on rank 0):
  |     |
  |     +-- Compute max_q_len (= mtp_k + 1 if MTP drafter, else 1)
  |     +-- Compute num_tokens = bs * max_q_len
  |     +-- Prepare cu_seqlens_q, positions
  |     +-- Build attn_metadata and context via attn_metadata_builder
  |     +-- Handle DP padding via get_dp_padding()
  |     +-- Set forward context (set_forward_context)
  |     +-- Warmup run: model(input_ids[:num_tokens], positions[:num_tokens])
  |     +-- Capture: torch.cuda.graph(graph, self.graph_pool, stream=gc.stream)
  |     +-- Share graph_pool across captures (set on first capture)
  |     +-- Store: self.graphs[(bs, max_q_len)] = graph
  |     +-- torch.cuda.synchronize()
  |
  +-- Sort graph_bs back to ascending order
  +-- Return (elapsed_time, graph_bs)
```

### Graph Keying

Each captured graph is stored in a dictionary keyed by a `(graph_bs, max_q_len)` tuple:

```python
self.graphs: dict[tuple[int, int], torch.cuda.CUDAGraph] = dict()
```

- `graph_bs`: The padded batch size used during capture.
- `max_q_len`: The maximum query length per sequence. For standard decode, this is `1`. For MTP (Multi-Token Prediction) speculative decoding, this is `mtp_k + 1`.

### Graph Pool Sharing

The first captured graph creates a CUDA memory pool via `graph.pool()`. All subsequent captures share this pool through the `self.graph_pool` parameter, enabling memory reuse across different batch sizes.

```python
if self.graph_pool is None:
    self.graph_pool = graph.pool()
```

### Default Capture Sizes

When `cuda_graph_sizes` has a single value (e.g., `[512]`, the default), the capture sizes follow this pattern:

```python
[1, 2, 4, 8] + [i for i in range(16, cuda_graph_sizes[0] + 1, 16)]
# Example with default 512:
# [1, 2, 4, 8, 16, 32, 48, 64, ..., 496, 512]
```

### Graph Replay in run_model()

During inference, `ModelRunner.run_model()` decides whether to use eager execution or graph replay:

```python
def run_model(self, input_ids):
    forward_context = get_forward_context()
    context = forward_context.context
    bs = context.batch_size
    is_prefill = context.is_prefill
    positions = context.positions

    if is_prefill or self.enforce_eager or bs > self.graph_bs[-1]:
        # Eager path: prefills, enforce_eager mode, or oversized batches
        hidden_states = self.model(input_ids, positions)
    else:
        # Graph replay path: decode batches within captured range
        graph_bs = context.graph_bs
        max_q_len = forward_context.attn_metadata.max_seqlen_q
        graph_key = (graph_bs, max_q_len)
        self.graphs[graph_key].replay()
        num_tokens = context.batch_size * max_q_len
        hidden_states = self.forward_vars["outputs"][:num_tokens]

    return self.model.compute_logits(hidden_states), hidden_states
```

Key decisions:
- **Prefill**: Always eager (variable sequence lengths make CUDA graphs impractical).
- **Decode with bs <= max captured size**: Replay the pre-captured graph.
- **Decode with bs > max captured size**: Fall back to eager execution.
- **enforce_eager=True**: Always eager, regardless of batch size.

---

## 4. Piecewise Compilation

Piecewise compilation splits the model's computation graph at specified operations and compiles each subgraph independently. This enables CUDA graph capture for the compilable parts while leaving incompatible operations (primarily attention) to run eagerly.

### Splitting Operations

The `splitting_ops` field in `CompilationConfig` defines which operations split the graph. When `set_splitting_ops_for_v1()` is called (automatically at level 3), the default splitting ops are:

```python
["aiter.unified_attention_with_output", "aiter.mla_attention"]
```

These attention operations are split out because:
1. They require dynamic metadata (sequence lengths, block tables) that changes per step.
2. Some attention backends are not compatible with CUDA graph capture.
3. Attention kernels are already highly optimized, so Inductor compilation provides minimal additional benefit.

### Compilation Pipeline

The `VllmBackend.__call__` method orchestrates the piecewise compilation:

1. **Graph splitting**: `split_graph()` divides the traced model graph at the splitting operations into a sequence of `SplitItem` objects, each containing a subgraph.

2. **Submodule identification**: Subgraphs that are *not* splitting operations are identified as candidates for compilation.

3. **Dynamic-shape compilation**: `PiecewiseCompileInterpreter` runs the split graph with fake inputs and compiles each non-splitting subgraph via `CompilerManager.compile()` for a general (dynamic) shape.

4. **Backend creation**: For each compiled subgraph, a `PiecewiseBackend` instance is created. It holds:
   - `compiled_graph_for_general_shape`: The Inductor-compiled graph for dynamic shapes.
   - `concrete_size_entries`: A dictionary mapping specific runtime shapes to `ConcreteSizeEntry` objects for shape-specialized compilation.

5. **Runtime dispatch**: When `PiecewiseBackend.__call__` is invoked:
   - On the first run, it uses the general-shape compiled graph.
   - For subsequent runs, if the runtime shape is in `compile_sizes`, it lazily compiles a shape-specialized version via `CompilerManager.compile()` and caches it.
   - For shapes not in `compile_sizes`, it falls back to the general-shape compiled graph.

### Cache Management

The `CompilerManager` caches compiled graphs using a key of `(runtime_shape, graph_index, backend_name)`. The cache is stored in a Python file (`vllm_compile_cache.py`) at the local cache directory (`~/.cache/atom/torch_compile_cache/<hash>/rank_<i>/<prefix>/`).

On subsequent runs with the same model and configuration, compiled graphs are loaded from the cache, bypassing Inductor compilation entirely.

---

## 5. Forward Context & Stateless Dispatch

The `ForwardContext` dataclass in `atom/utils/forward_context.py` provides a module-level global mechanism for passing metadata to layers during the forward pass. This is critical for CUDA graphs because captured graphs cannot accept new arguments -- all dynamic metadata must be accessible through a side channel.

### ForwardContext Fields

| Field | Type | Purpose |
|-------|------|---------|
| `no_compile_layers` | `dict[int, Any]` | Layers that should skip compilation (from `static_forward_context`) |
| `attn_metadata` | `AttentionMetaData` or `dict` | Attention-specific metadata (sequence lengths, block tables, etc.) |
| `kv_cache_data` | `dict[str, KVCacheTensor]` | KV cache tensors for each layer |
| `context` | `Context` | Basic forward pass context (positions, is_prefill, batch_size, graph_bs) |
| `dp_metadata` | `DPMetadata` | Data-parallel metadata (token counts across DP ranks) |
| `spec_decode_metadata` | `SpecDecodeMetadata` | Speculative decoding metadata (draft tokens, logits indices) |

### Lifecycle

The forward context follows a set-use-reset lifecycle:

1. **Set**: Before each forward pass, `set_forward_context()` is called with attention metadata, the ATOM config, a `Context` object, and optional DP/speculative decoding metadata.

2. **Access**: During the forward pass, any layer can call `get_forward_context()` to retrieve the current metadata without needing it passed as a function argument. This is used by both eager execution and CUDA graph replay paths.

3. **Reset**: After the forward pass, `reset_forward_context()` replaces the global context with an empty `ForwardContext()`.

### Context Dataclass

The `Context` object carries the most frequently accessed per-step state:

```python
@dataclass
class Context:
    positions: torch.Tensor    # Token position IDs
    is_prefill: bool = False   # Whether this is a prefill step
    batch_size: int = 0        # Number of sequences in the batch
    graph_bs: int = 0          # Padded batch size for graph lookup
    is_draft: bool = False     # Whether this is a draft model forward
```

The `graph_bs` field is particularly important for CUDA graph dispatch: it holds the padded batch size that maps to a pre-captured graph key.

### Integration with CUDA Graphs

For ModelRunner's direct CUDA graph path (non-piecewise), the forward context is set before `run_model()` via `set_forward_context()`, and `run_model()` reads `context.graph_bs` and `attn_metadata.max_seqlen_q` to look up the correct pre-captured graph.

For the piecewise path, `CUDAGraphWrapper` (in `atom/utils/cuda_graph.py`) expects `batch_descriptor` and `cudagraph_runtime_mode` fields on the forward context to decide whether to capture, replay, or run eagerly:

```python
forward_context = get_forward_context()
batch_descriptor = forward_context.batch_descriptor
cudagraph_runtime_mode = forward_context.cudagraph_runtime_mode
```

> **Note:** The piecewise `CUDAGraphWrapper` integration is under development. The `batch_descriptor` and `cudagraph_runtime_mode` fields are expected by `CUDAGraphWrapper.__call__()` but are not currently defined on the `ForwardContext` dataclass. The per-subgraph wrapping in `backends.py` is also currently commented out. The direct CUDA graph path in `ModelRunner` is the active production path.

---

## 6. Compiler Backend

### CompilerManager

`CompilerManager` in `atom/utils/backends.py` manages the full compilation lifecycle:

- **Initialization**: Creates a `CompilerInterface` via `make_compiler()`. Uses `InductorStandaloneAdaptor` for PyTorch 2.8+ or `InductorAdaptor` for earlier versions.
- **Caching**: Maintains a dictionary mapping `(runtime_shape, graph_index, backend_name)` to compiler-specific handles. Caches are serialized to a Python file using `pprint`.
- **Compile-or-load**: On each call to `compile()`, first attempts `load()` from the cache. On miss, delegates to the compiler and stores the result.

### CompilerInterface

`CompilerInterface` in `atom/utils/compiler_inferface.py` (note the typo in the filename) defines the abstract interface that all compiler backends must implement:

| Method | Purpose |
|--------|---------|
| `initialize_cache(cache_dir, disable_cache, prefix)` | Set up cache directories for the compiler |
| `compute_hash(vllm_config)` | Generate a hash of compiler-specific state for cache invalidation |
| `compile(graph, example_inputs, compiler_config, runtime_shape, key)` | Compile a graph, returning `(compiled_callable, handle)` |
| `load(handle, graph, example_inputs, graph_index, runtime_shape)` | Load a previously compiled graph from the handle |

### InductorAdaptor

The default compiler for PyTorch < 2.8. Uses `torch._inductor.compile_fx.compile_fx` and monkey-patches several internal functions to:

- Extract the compilation hash for caching.
- Provide a dummy shape environment (`AlwaysHitShapeEnv`) so Inductor cache lookups succeed outside of Dynamo's tracing context.
- Force caching of graphs that Inductor would normally refuse to cache.

When `runtime_shape` is an integer (specific batch size), it enables `max_autotune` and `coordinate_descent_tuning` for Triton kernel parameter optimization.

### InductorStandaloneAdaptor

The preferred compiler for PyTorch 2.8+. Uses `torch._inductor.standalone_compile` which provides a cleaner interface without the monkey-patching required by `InductorAdaptor`. Compiled artifacts are saved to disk in "unpacked" format and can be loaded directly.

### VllmBackend

`VllmBackend` in `atom/utils/backends.py` serves as the `torch.compile` backend for level 3 (piecewise) compilation. When Dynamo calls it:

1. Computes a cache directory hash from config, traced files, and compiler state.
2. Splits the graph at `splitting_ops` using `split_graph()`.
3. Runs `PiecewiseCompileInterpreter` to compile each non-splitting subgraph.
4. Saves the computation graph to `computation_graph.py` for debugging.
5. Returns the stitching graph module (`split_gm`) as the callable.

If `cudagraph_copy_inputs` is `True`, it wraps the callable to copy input tensors into static buffers before each call, ensuring CUDA graph input address stability.

### @support_torch_compile Decorator

The `@support_torch_compile` decorator in `atom/utils/decorators.py` augments a model class to support `torch.compile`:

1. **Class modification**: Adds `TorchCompileWrapperWithCustomDispatcher` as a base class and overrides `__init__` and `__call__`.

2. **Dynamic shape marking**: On the first compilation, it inspects the `forward` method signature, identifies `torch.Tensor` arguments, and calls `torch._dynamo.mark_dynamic()` to mark their batch dimensions as dynamic.

3. **Custom dispatch**: After the first compilation, if `use_custom_dispatcher` is True (levels >= 2), subsequent calls bypass Dynamo's guard mechanism and dispatch directly to the compiled bytecode via `dispatch_to_code(0)`.

4. **Safety check**: The bytecode hook checks for `update` in the compiled code's `co_names`, raising an error if the model modifies `nn.Module` buffers during the forward pass (which would cause silent errors with CUDA graphs).

### Custom Op Registration

`direct_register_custom_op()` in `atom/utils/custom_register.py` registers custom operators with PyTorch's `torch.library` system:

```python
direct_register_custom_op(
    op_name="my_op",
    op_func=my_kernel,
    mutates_args=["output"],
    fake_impl=my_fake_impl,
)
```

This registers the op under the `"aiter"` library namespace (e.g., `aiter.my_op`), making it visible to Dynamo's tracing. The `fake_impl` is used during tracing to compute output shapes without executing the real kernel. The `dispatch_key` defaults to `"CUDA"` for GPU operations.

Registered custom ops can be used as `splitting_ops` in piecewise compilation (e.g., `"aiter.unified_attention_with_output"`).

---

## 7. Configuration Options

All compilation-related configuration fields from `CompilationConfig`:

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `level` | `int` | `0` | Compilation level (0-3). See Section 1. |
| `use_cudagraph` | `bool` | `True` | Whether CUDA graph capture is enabled. |
| `cudagraph_capture_sizes` | `Optional[list[int]]` | `None` | Explicit list of batch sizes to capture. Overrides `cuda_graph_sizes`. |
| `cuda_graph_sizes` | `list[int]` | `[512]` | Controls auto-generated capture sizes. 1 value = generate pattern; >1 values = use directly. |
| `cudagraph_mode` | `Optional[CUDAGraphMode]` | `None` | CUDA graph mode. Set to `PIECEWISE` automatically at level 3. |
| `splitting_ops` | `Optional[list[str]]` | `None` | Operations that split the graph for piecewise compilation. Auto-set at level 3. |
| `cudagraph_copy_inputs` | `bool` | `False` | Copy input tensors to static buffers for CUDA graph address stability. Only effective in `PIECEWISE` mode. |
| `use_inductor` | `bool` | `True` | Whether to use the Inductor compiler backend. |
| `compile_sizes` | `Optional[list[Union[int, str]]]` | `None` | Specific sizes to compile with Inductor. Supports `"cudagraph_capture_sizes"` string. |
| `inductor_compile_config` | `dict` | `{}` | Additional Inductor configuration (e.g., `max_autotune`). |
| `debug_dump_path` | `str` | `""` | Path to dump debug information (traced graphs, decompiled code). |
| `cache_dir` | `str` | `""` | Custom cache directory. Auto-generated if empty (`~/.cache/atom/torch_compile_cache/<hash>/`). |

Related fields on `Config`:

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `enforce_eager` | `bool` | `False` | Force eager execution, skip all compilation and CUDA graphs. |
| `graph_bs` | `Optional[list[int]]` | `None` | Final list of batch sizes for CUDA graph capture (computed from `CompilationConfig`). |
| `compilation_config` | `CompilationConfig` | `CompilationConfig()` | The compilation configuration dataclass. |

---

## 8. Decision Tree

Use this decision tree to select the right compilation level and CUDA graph mode for your workload:

```
Is the model supported by torch.compile?
|
+-- No --> Level 0 (NO_COMPILATION)
|          enforce_eager=True
|
+-- Yes
    |
    +-- Debugging / profiling?
    |   |
    |   +-- Yes --> Level 0 (NO_COMPILATION)
    |
    +-- Quick compatibility check?
    |   |
    |   +-- Yes --> Level 1 (DYNAMO_AS_IS)
    |
    +-- Want Inductor optimization without CUDA graphs?
    |   |
    |   +-- Yes --> Level 2 (DYNAMO_ONCE)
    |
    +-- Production deployment
        |
        +-- Level 3 (PIECEWISE) [recommended]
            |
            +-- Standard serving --> cudagraph_mode=PIECEWISE (default)
            |
            +-- Small model / uniform batches --> cudagraph_mode=FULL
            |
            +-- P/D disaggregated (decode instance) --> cudagraph_mode=FULL_DECODE_ONLY
            |
            +-- Maximum performance --> cudagraph_mode=FULL_AND_PIECEWISE
```

### Common Configurations

**Default production setup** (level 3, piecewise CUDA graphs):
```python
CompilationConfig(level=3)
# Automatically sets:
#   splitting_ops = ["aiter.unified_attention_with_output", "aiter.mla_attention"]
#   cudagraph_mode = CUDAGraphMode.PIECEWISE
#   cuda_graph_sizes = [512]
#   graph_bs = [1, 2, 4, 8, 16, 32, ..., 512]
```

**Custom capture sizes**:
```python
CompilationConfig(level=3, cudagraph_capture_sizes=[1, 2, 4, 8])
```

**Debugging with full eager execution**:
```python
Config(model="...", enforce_eager=True)
# or
CompilationConfig(level=0)
```

**Inductor with debug dump**:
```python
CompilationConfig(level=3, debug_dump_path="/tmp/atom_debug")
# Dumps traced graphs and decompiled code to /tmp/atom_debug/rank_0/
```

---

## Source Files

| File | Description |
|------|-------------|
| `atom/config.py` | `CompilationLevel`, `CompilationConfig`, `CUDAGraphMode`, `Config.__post_init__` (compilation setup) |
| `atom/utils/cuda_graph.py` | `CUDAGraphEntry`, `CUDAGraphOptions`, `CUDAGraphWrapper`, `BatchDescriptor` |
| `atom/utils/backends.py` | `CompilerManager`, `VllmBackend`, `SplitItem`, `split_graph()`, `PiecewiseCompileInterpreter` |
| `atom/utils/forward_context.py` | `ForwardContext`, `Context`, `AttentionMetaData`, `DPMetadata`, `set_forward_context()`, `get_forward_context()` |
| `atom/utils/compiler_inferface.py` | `CompilerInterface`, `InductorAdaptor`, `InductorStandaloneAdaptor`, `AlwaysHitShapeEnv` |
| `atom/utils/cuda_piecewise_backend.py` | `PiecewiseBackend`, `ConcreteSizeEntry` |
| `atom/utils/decorators.py` | `@support_torch_compile`, `TorchCompileWrapperWithCustomDispatcher`, `start_monitoring_torch_compile` |
| `atom/utils/custom_register.py` | `direct_register_custom_op()`, `aiter_lib` (Library instance) |
| `atom/model_engine/model_runner.py` | `ModelRunner.capture_cudagraph()`, `ModelRunner.run_model()` |
