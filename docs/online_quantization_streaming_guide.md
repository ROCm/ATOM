# Streaming online quantization guide

ATOM can quantize eligible modules while their checkpoint weights are being
loaded, then release their source weights early. This reduces the load-time peak
GPU memory caused by source and quantized weights coexisting.

This guide covers only the **streaming execution path**. For target formats,
layer-matching rules, and accuracy guidance, first read the
[online quantization guide](./online_quantization_guide.md) and
[online quantization best practices](./online_quantization_best_practices.md).

## 1. What streaming online quantization solves

Without streaming, online quantization runs in two consecutive phases:

1. Load all checkpoint weights onto the device.
2. Walk the model and call `process_weights_after_loading()` to create the
   quantized weights.

Until the second phase finishes, source weights and quantized outputs may occupy
device memory at the same time. For large MoE models, this temporary peak can
determine whether the model starts successfully.

Streaming reduces the processing unit to one Linear or fused MoE module:

```text
Read the module's checkpoint weights
    ↓
Confirm that all parameters for the module have arrived
    ↓
Move them to the target device and quantize
    ↓
Release source weights no longer needed by the module
    ↓
Continue loading later modules
```

Checkpoint reads, H2D transfers, quantization, and source-storage release can
overlap. The primary goal is to reduce load-time peak memory; the serving API
after startup is unchanged.

## 2. Activation conditions and defaults

The following environment variables control streaming:

| Environment variable | Default | Description |
|---|---:|---|
| `ATOM_ONLINE_QUANT_STREAMING` | `0` | Opts in to streaming online quantization. |
| `ATOM_ONLINE_QUANT_STREAMING_HOST_STAGING` | `1` | Assembles module weights in CPU host memory before moving them to the device. |
| `ATOM_ONLINE_QUANT_STREAMING_THREADS` | `4` | Number of tail workers for H2D, quantization, and source release. `0` finalizes synchronously in checkpoint-walker threads. |

Streaming is disabled by default so existing online-quant loads retain their
post-load behavior and full-weight quantization semantics. Set
`ATOM_ONLINE_QUANT_STREAMING=1` to opt in. After it is enabled, all of the
following must also hold:

1. The command provides a valid `--online_quant_config`.
2. The model contains online-quant modules that support streaming.
3. The load is not a dummy load.
4. The source checkpoint and target format are supported by online
   quantization.

Setting `ATOM_ONLINE_QUANT_STREAMING=1` without an
`--online_quant_config` does not quantize the model by itself.

## 3. Recommended launch configuration

The following example converts an FP8-block source model to FP8 non-expert
Linear layers and MXFP4 experts:

```bash
ATOM_ONLINE_QUANT_STREAMING=1 \
ATOM_ONLINE_QUANT_STREAMING_HOST_STAGING=1 \
ATOM_ONLINE_QUANT_STREAMING_THREADS=4 \
python -m atom.entrypoints.openai_server \
  --model deepseek-ai/DeepSeek-R1-0528 \
  -tp 8 \
  --online_quant_config '{
    "global_quant_config": "ptpc_fp8",
    "layer_quant_config": {"*expert*": "mxfp4"},
    "exclude_layer": ["lm_head", "*.gate.*"]
  }'
```

The environment variables and `--online_quant_config` syntax are identical
when using `atom.examples.simple_inference`.

### A/B against the non-streaming path

Unset the streaming variable, or set it explicitly to `0`, to quantize after the
complete model has loaded:

```bash
ATOM_ONLINE_QUANT_STREAMING=0 \
python -m atom.entrypoints.openai_server \
  ... \
  --online_quant_config '{...}'
```

When comparing load time or peak memory, keep the model, TP/EP configuration,
quantization JSON, application caches, and checkpoint page-cache state
identical. Cold storage and a warm page cache can differ by several times in
load duration.

## 4. Host staging and buffered replay

### Host staging (default when streaming is enabled, and recommended)

With `ATOM_ONLINE_QUANT_STREAMING_HOST_STAGING=1`:

1. Streamable parameters initially avoid allocating their complete GPU source
   storage.
2. Checkpoint arrivals write into module-level staging storage in CPU host
   memory.
3. After every parameter for a module has arrived, its parameters are moved to
   the target device.
4. Tail workers quantize and release source storage early. When
   `ATOM_ONLINE_QUANT_STREAMING_THREADS=0`, checkpoint-walker threads perform
   this finalization synchronously instead.

This mode allows the checkpoint walker to retain the concurrency configured by
`ATOM_LOADER_NUM_THREADS`. It usually provides better loading throughput, at
the cost of corresponding host-memory usage.

### Buffered replay

With `ATOM_ONLINE_QUANT_STREAMING_HOST_STAGING=0`, loader calls are recorded and
replayed on the target device when the module can be processed. The checkpoint
walk is forced to one thread for ordering and state safety. Tail quantization
can still use the workers configured by
`ATOM_ONLINE_QUANT_STREAMING_THREADS`; a value of `0` finalizes inline.

This mode is primarily useful for compatibility comparisons and debugging. It
is usually not the preferred performance configuration.

## 5. Detecting module completion

A weight loader may slice the destination, call `narrow`, or map expert shards
before writing. The checkpoint tensor shape alone is therefore not sufficient
to determine whether a module is complete.

ATOM counts the number of elements actually written through `aten.copy_` inside
the loading scope. Where available, fused MoE modules use a more explicit
`(expert, shard)` region-coverage protocol, preventing duplicate arrivals from
claiming a module too early.

Only after all parameters for a module are known to be complete does the
streamer:

1. Claim the module so later arrivals cannot overwrite quantized results.
2. Submit finalization to a tail worker, or run it inline when the tail-worker
   count is `0`.
3. Call `process_weights_after_loading()`.
4. Release stale source storage.

## 6. Tuning the worker count

`ATOM_ONLINE_QUANT_STREAMING_THREADS` controls the size of the tail-worker pool.

| Value | Use case | Cost |
|---:|---|---|
| `0` | Do not create tail workers | Checkpoint-walker threads quantize synchronously |
| `1` | Memory is tight, but some overlap is desirable | Low parallelism |
| `4` | Recommended default | Balance between throughput and in-flight memory |
| `>4` | H2D or quantization is the measured bottleneck and memory is available | More in-flight storage, streams, and scheduling overhead; not guaranteed to be faster |

A value of `0` does not imply strictly serial execution. With host staging
enabled, multiple checkpoint-walker threads can still finalize different
modules concurrently. Set `ATOM_LOADER_NUM_THREADS=1` as well when validating
strict ordering.

When the tail-worker count is greater than zero, a semaphore limits submitted
but incomplete tail tasks to `2 × worker count`, preventing an unbounded queue
when loading produces work faster than quantization consumes it.

Before increasing the worker count, measure load time, GPU peak memory, host
memory, and the number of fallback modules.

## 7. Verifying that streaming is active

Every rank logs phase timing:

```text
[rank_0] weight load phases (including streaming online quant):
read+queue 140.71s (...) | drain 2.01s | quant drain 0.00s |
staging flush 0.00s | threads 16
```

Here, `threads 16` is the checkpoint-walker thread count from
`ATOM_LOADER_NUM_THREADS`, not the tail-worker count from
`ATOM_ONLINE_QUANT_STREAMING_THREADS`. `staging flush` refers to the loader's
expert-staging flush, not the streamer's host-staging H2D time.

The runtime also logs streaming coverage and fallback counts. The numbers below
are one example, not expected values for every model:

```text
Online-quant streaming: 444/582 eligible modules quantized during load,
138 fell back to the post-load pass (no memory saving for those).
```

The final loading summary includes:

```text
Post-stream fallback and weight processing done: 0.55 seconds
Peak GPU memory during streaming weight loading and online quantization: ...
Model load done: ... (weights loaded in 145.67s)
```

Check the following:

- `eligible modules quantized during load` is greater than zero.
- The fallback list is consistent with the model structure.
- There are no warnings about parameters that never loaded and were
  zero-filled.
- There are no warnings about checkpoint arrivals dropped after a module was
  quantized.
- Layer counts and formats in `online_quant_info_*.json` match the requested
  quantization configuration.

## 8. What fallback means

Not every eligible module can necessarily be quantized during checkpoint
loading. The following cases stay in the standard post-load pass:

- A child module must retain source weights until a parent combines them.
- A custom or fused loader cannot provide reliable write coverage.
- Arrivals are incomplete or coverage tracking becomes invalid.
- A module needs the loader to finish before its final storage is known.

Fallback preserves compatibility, but those modules do not receive the
peak-memory benefit of streaming source release. A small, stable, and
explainable fallback set is usually normal. A sudden increase should prompt an
inspection of recent model-loader or module post-processing changes.

## 9. Performance and output semantics

The primary goal of streaming online quantization is **lower load-time peak
memory, not lower load time**. Compared with quantizing after the complete model
has loaded, it adds some fixed overhead:

1. In the default host-staging mode, checkpoint arrivals write into host staging
   rather than remaining directly in final device source storage. This adds CPU
   copies and staging allocation/zero-fill.
2. Every loader call records actual write coverage and updates module completion
   under a lock.
3. Module completion requires storage swaps, future submission, semaphore
   acquisition, and device-stream management when tail workers are enabled.
4. Workers synchronize after H2D and quantization. H2D, quantization kernels,
   and checkpoint reads may also compete for memory bandwidth.
5. Fallback modules pay streaming-tracking overhead and still enter the standard
   post-load pass.

Streaming does not reduce the required quantization computation; it attempts to
overlap that work with checkpoint I/O. Consequently:

- With cold storage or an I/O-bound load, checkpoint reads may hide
  quantization, making startup equal or faster.
- With a warm page cache, lightweight quantization, or many fallbacks, little
  work can be hidden and management overhead appears directly as several extra
  seconds of load time.
- More workers help only when H2D or quantization is the bottleneck. Too many
  workers increase in-flight storage, synchronization, and bandwidth
  contention, and can make loading slower.

Under TP, streaming quantizes the local shard on each rank directly. When
matching full-weight quantization would require a TP all-gather, quantization
scales and final weights can differ slightly from non-streaming or offline
full-weight quantization. Run an accuracy A/B before deployment.

The final runtime format is still selected by `--online_quant_config`.
Streaming primarily changes load scheduling and intermediate-storage lifetime,
with the TP output difference noted above.

## 10. Troubleshooting

### No streaming logs appear

Confirm that the command contains a non-empty `--online_quant_config`, and that
the source checkpoint supports online quantization. The streaming environment
variable alone does not create quantization work.

### Streaming is slower than non-streaming

First confirm that both runs use the same cache state. Then:

1. Keep host staging enabled.
2. Benchmark worker counts of `1`, `4`, and higher.
3. Check whether too many modules fall back.
4. Determine whether the bottleneck is checkpoint I/O, H2D, or a quantization
   kernel.

### Peak memory is still high

Reduce `ATOM_ONLINE_QUANT_STREAMING_THREADS` and inspect the fallback count.
Many fallback modules keep their source weights until the post-load phase.

### Disable streaming completely

```bash
ATOM_ONLINE_QUANT_STREAMING=0
```

Online quantization still runs, but only in the unified post-load pass after the
complete model has loaded.
