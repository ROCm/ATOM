# ROCm 7.2.4 ordering-edge runtime

The native ATOM image builds the paired HIP and HSA runtimes from ROCm commit
`b539bf7eebfd99ad0a69668caa1f4037034d501f`, the ROCm 7.2.4 backport used by
[vLLM PR #55099](https://github.com/vllm-project/vllm/pull/55099).
It adds device-resident ordering signals for cross-stream/queue dependencies.
The model's stream topology, draft model and kernels are unchanged.

`ROCM_ORDERING_EDGE=auto` (the default Docker build argument) installs this pair
only when the base SDK reports ROCm 7.2.4. Other versions, including ROCm 10,
retain their original runtime. Set it to `0` to build with stock libraries;
`1` requires a 7.2.4 base and fails on an unsupported version. Remove this
backport once the image's official runtime includes the fix.

```bash
docker build -f docker/Dockerfile --target atom_image \
  --build-arg BASE_IMAGE=rocm/pytorch:rocm7.2.4_ubuntu24.04_py3.12_pytorch_release_2.10.0 \
  -t atom:ordering-edge .

# Build without replacing HIP/HSA.
docker build -f docker/Dockerfile --target atom_image \
  --build-arg ROCM_ORDERING_EDGE=0 -t atom:stock .
```

The separate runtime builder installs development dependencies before the custom
RCCL stage. Its compiler, headers and source do not enter the final image. Both
libraries are installed into the common base, so native builds and the OOT /
SGLang images derived from that native image use the paired runtime.

## Selection and rollback

Original HIP/HSA libraries are saved under
`/opt/atom-rocm-runtime/stock/lib`, with SHA256 checksums. The compiled pair is
under `/opt/atom-rocm-runtime/patched/lib`; `build-info.json` records the commit.
All original versioned filenames and SONAMEs select the same pair. This also
covers consumers that use a hardcoded `/opt/rocm/lib/libamdhip64.so*` path with
`RTLD_NOLOAD`, such as older DMA-BUF registration shims.

Run as root in the container, with GPU applications stopped:

```bash
atom-rocm-runtime status
atom-rocm-runtime stock
ldconfig
# Restart the application to load the original pair.

# Select the backport again, then restart the application.
atom-rocm-runtime patched
ldconfig
```

A switch changes files for new processes; it cannot replace libraries already
mapped by a running worker. The original backup is retained across repeated
switches, and its checksums are verified before either selection. RCCL and other
SDK libraries are not selected or overwritten by this tool. There is no new
entrypoint, LD_PRELOAD requirement, or GPU_MAX_HW_QUEUES setting.

For an ordering-edge diagnostic using the patched pair, set
`DEBUG_CLR_DISABLE_ORDERING_EDGE=1` before starting the application.
This disables the feature; it is not a rollback to the original binaries.
For Torch Profiler on the tested stack, use
`ROCPROFILER_QUEUE_INTERPOSITION=0`, as in the upstream PR.

## Validation

CPU-only rollback/alias checks:

```bash
python3 tests/test_rocm_runtime_switch.py
```

The backport was tested with DeepSeek V4 Pro on MI355X, TP8, fixed BS16 decode,
FP8 KV / FP4 index and a fixed simulated speculative acceptance length of 3.01.
Four waves of 16 requests per configuration used the same 98,306-token prompt
and 2,048 output tokens; TPOT was measured only while all requests were decoding.
GPU_MAX_HW_QUEUES was unset, target multi-stream and draft were retained.

| Runtime | Mean TPOT (ms/token) | Four-wave standard deviation |
|---|---:|---:|
| Original ROCm 7.2.4 | 7.7675 | 0.2432 |
| Ordering-edge pair | 6.7484 | 0.1137 |

This is a 13.1% TPOT reduction in that experiment, not an accuracy result or a
full agentic benchmark. A same-build feature-off timing and a repeated stock
model timing were not completed; the result does not isolate every build or
time-drift effect. Multi-stream graph correctness was checked on all eight GPUs.

The Docker common-base target was rebuilt on a ROCm 7.2.4 ATOM development
image. In a disposable GPU container, patched -> stock -> patched each passed
128 changing-input, multi-stream graph replays on all eight GPUs. The checks
also verified loaded HIP/HSA file hashes, the new HSA symbol, and RTLD_NOLOAD
identity through the original HIP filenames. Full native/OOT/SGLang image
builds and model accuracy tests remain CI validation.
