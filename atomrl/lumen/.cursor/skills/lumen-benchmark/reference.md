# Lumen Benchmark Reference

Detailed patterns, error catalog, and per-benchmark conventions.

## conftest.py Patterns

### Markers

```python
from benchmarks.conftest import CUDA, AITER

@CUDA          # skip if no CUDA
@AITER         # skip if no CUDA or no AITER
```

### Warning Suppression

Global filters in `conftest.py` (also duplicated in `pyproject.toml` for `torchrun`):

```python
warnings.filterwarnings("ignore", message=".*Aiter backend is selected for fused RoPE.*")
warnings.filterwarnings("ignore", message=".*will be removed in.*megatron-core.*")
warnings.filterwarnings("ignore", message=".*megatron.core.transformer.custom_layers.*")
warnings.filterwarnings("ignore", message=".*No device id is provided via.*init_process_group.*")
warnings.filterwarnings("ignore", message=".*destroy_process_group.*was not called.*")
```

### `_cleanup_dist` Fixture

Session-scoped autouse fixture that calls `dist.destroy_process_group()` on exit.
**Prevents SIGSEGV** when NCCL is active and the process exits without cleanup.

## Error Catalog

### 1. `RuntimeError: Expected one of cpu, cuda... device type at start of device string: nccl`

**Cause**: Using `backend="nccl:gloo"` or similar invalid backend string.

**Fix**: Use `backend="cpu:gloo,cuda:nccl"` with `device_id=torch.device(f"cuda:{local_rank}")`.

### 2. SIGSEGV on Process Exit (Multi-GPU)

**Cause**: NCCL process group not destroyed before process terminates.

**Fix**: The `_cleanup_dist` fixture in `benchmarks/conftest.py` handles this automatically. Do NOT add manual `dist.destroy_process_group()` in individual benchmark files.

### 3. AITER CK JIT Hang / Stale Lock Files

**Cause**: AITER CK kernel compilation creates `FileBaton` lock files. If a previous run was killed, stale locks block subsequent runs.

**Symptoms**: Tests hang indefinitely during `aiter_csrc` attention backward.

**Fix**: Clean stale locks in `tests/ops/conftest.py`:

```python
@pytest.fixture(scope="session", autouse=True)
def _cleanup_stale_aiter_jit_locks():
    bd_dir = os.path.join(os.path.dirname(aiter.__file__), "jit_build")
    for pattern in ["lock_*", "*/build/lock"]:
        for lock in glob.glob(os.path.join(bd_dir, pattern)):
            os.remove(lock)
```

### 4. CK Native Kernel SIGSEGV in `mp.spawn` + NCCL P2P

**Cause**: CK (Composable Kernel) native `aiter_csrc` backend crashes in `mp.spawn` subprocess when NCCL P2P is active. Root cause is interaction between CK's HIP runtime state and the forked process environment.

**Workaround**: Force `backend_type="aiter_triton"` in the attention call. Do NOT change CK itself.

### 5. Low Speedup in Single-Layer FP8 vs BF16

**Cause**: FP8 gather is faster (half bandwidth) but dequantization adds latency. In a single-layer sequential pipeline, dequant overhead erases the communication savings.

**Solution**: Add a multi-layer pipelined test (`test_multi_layer_pipelined_gather_forward`) where allgather(layer i+1) on comm_stream overlaps with dequant+GEMM(layer i) on compute stream.

### 6. Megatron/Apex Warning Noise

**Cause**: Megatron-LM and Apex emit deprecation warnings when imported.

**Fix**: `filterwarnings` in both `conftest.py` (for `pytest`) and `pyproject.toml` (for `torchrun` which doesn't load conftest early enough).

## Per-Benchmark Details

### bench_kernel_launch.py

**Features**: `quantized_linear` (7 scaling modes), `fused_moe_triton`, `fp8_activation_store`, attention backends, norm+GEMM pipeline.

**Key pattern**: Use `cuda_timer_batch()` for very fast kernels where per-iteration event overhead matters.

**Scaling modes**: `["none", "delayed", "dynamic", "per_token", "blockwise", "blockwise2d", "mxfp8"]` — always include all 7.

### bench_rope_fusion.py

**Features**: `apply_rotary_pos_emb` (1D), `fused_rope` (Q+K), 2D vision, 3D video, GQA configs, NeoX vs GPT-J.

**Baseline**: Always include a pure-PyTorch reference implementation for comparison.

### bench_fp8_param_allgather.py

**Features**: `FP8ParamManager`, dequant hooks, param compression SNR, multi-GPU NCCL/SDMA gather, pipelined E2E.

**Key insight**: Single-layer FP8 E2E appears slower than BF16. Always include a multi-layer pipeline test to demonstrate overlap benefit.

**Pattern for pipelined tests**:
- BF16 sequential (baseline)
- FP8 sequential (shows dequant overhead)
- FP8 pipelined (shows overlap hiding dequant)

### bench_comm_overlap.py

**Features**: `SdmaTpComm` async allgather/reduce-scatter, column/row parallel, NCCL vs SDMA.

**Mock pattern**: For single-GPU simulation tests, use `unittest.mock.patch` with `_MockSdmaComm` that delegates to `torch.distributed`.

**SDMA fixture**: `os.environ["MORI_ENABLE_SDMA"] = "1"` in `_setup`.

### bench_wgrad_delay.py

**Features**: `_DeferredWgrad` API, stream overlap, multi-GPU NCCL/SDMA comm overlap, `gradient_accumulation_fusion`.

**Overlap interpretation**: When wgrad GEMM time << allreduce time, the wgrad is fully hidden. Metrics:
- `hidden_ms = t_seq - t_ovl` (absolute time saved)
- `speedup = t_seq / t_ovl`
- `overlap_ratio = 1 - (t_ovl / (t_compute + t_comm))` — can be low when compute << comm

**Simulated comm**: Use `torch.cuda._sleep()` to simulate communication latency without consuming compute SMs or HBM bandwidth.

## Utility Functions Quick Reference

| Function | Use |
|----------|-----|
| `cuda_timer(fn, ...)` | Per-iteration CUDA event timing |
| `cuda_timer_batch(fn, ...)` | Batch timing (single event pair, divide by iters) |
| `track_cuda_memory()` | Context manager yielding `{peak_bytes, peak_delta, ...}` |
| `print_report(title, results)` | Pretty-print detailed results |
| `print_report_with_table(title, results)` | Detailed + summary table |
| `print_overlap_summary(...)` | Visual bar chart of overlap metrics |
| `print_table(title, results)` | Just the summary table |
| `dump_csv(results, path)` | Export to CSV |
| `dump_json(results, path)` | Export to JSON |
| `trace_fn(fn, path, ...)` | Profile a callable, export Chrome trace |
| `trace_context(path, ...)` | Context manager for profiling multi-stream code |
| `format_bytes(n)` | Human-readable byte formatting |

## run_traces.py CLI

```bash
python -m benchmarks.run_traces              # all trace groups
python -m benchmarks.run_traces --only rope  # one group
python -m benchmarks.run_traces --run trace_fused_moe  # one function
python -m benchmarks.run_traces --list       # list available functions
```

Groups: `kernel`, `comm`, `wgrad`, `rope`, `fp8`.

## Adding a New Benchmark

1. Create `bench_<feature>.py` in `Lumen/benchmarks/`
2. Import from `bench_utils` and `conftest` — no inline imports
3. Define Llama 3.1 8B dimensions at top
4. Add docstring with run commands (single-GPU, multi-GPU)
5. Add expected-effect comments before each test
6. Use `_init_dist()` pattern for multi-GPU tests
7. Use `_DIST` / `_SDMA` markers for conditional skipping
8. Add tracing entries to `run_traces.py` if applicable
9. Update `README.md` with new benchmark entry
