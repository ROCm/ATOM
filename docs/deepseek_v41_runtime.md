# DeepSeek-V4.1 native paged runtime (P05)

P05 is complete on the user-accepted P04 arithmetic baseline. This enables
native ATOM text execution through ModelRunner and Scheduler with chunked
prefill, ragged batches, continuous decode and complete request-state recovery.
The existing V4 BF16 sparse attention and inverse RoPE kernels are unchanged.
The numerical differences from the mathematical reference remain documented in
[the P04 validation report](deepseek_v41_validation.md); P05 does not reclassify
them as passes of the original thresholds.

## Ownership and execution boundaries

- `models/deepseek_v41/model.py` and `attention.py` own model arithmetic.
  `runtime.py` adapts its input/output contract to ModelRunner. Q/KV and output
  projections run over the flat token batch; compression and index selection
  use request spans. The offline interface remains available as a comparison.
- `model_ops/attentions/deepseek_v41/` owns metadata, addresses, PAGE/STATE views
  and checkpoint copies. Geometry is declared separately in
  `pool_layout/v41_pool_geometry.py` without model or scheduler imports.
- `model_loader/deepseek_v41.py` owns native checkpoint loading and mapped-table
  lifetime. Native post-load processing runs exactly once, preserving W4A8/QAT.
- `model_engine/engram_runtime.py` prepares Engram rows after final GPU token IDs
  and restored state are available. There is no separate committed history map.
- Scheduler consumes the existing generic `StateTransfer.copy` capability.
  The only scheduling change fixes cancellation of requests with no sampled
  output, including a middle prefill chunk and the first deferred step.

Only Full owners 2, 8, 14 and 20 allocate global main/index storage. Reuse and
Reindex layers read those owner regions. Every request has all 40 SWA rings,
three ratio-2 FP32 compressor tails, the latest three compressed Engram IDs and
a committed position in its STATE entry. Prefill retains the old rings until
all queries have consumed their causal prefixes, including chunks wider than
the ring. Index reads gather a tile or candidate positions, never a full copy
of the historical main KV.

With the production geometry and block size 16, one PAGE costs 51,200 bytes and
one STATE entry costs 5,256,192 bytes. The complete image occupies 103 PAGE
units. Allocation and checkpoints use these same declarations. Positions and
Engram history advance after the model forward; checkpoints carry every state
field and padding byte. Images are versioned by geometry and index tie policy.

An exact prefix hit restores the entire state before preparing Engram inputs.
Without a matching image, the generic scheduler replays from a recoverable
boundary. Checkpoint relocation/fork and restored tentative suffixes cannot
retain stale compressor tails, window rows or Engram history.

## Acceptance

Validation used `ljin_dev`, GPUs 0–3, TP4 with whole-expert EP, and native weights
at `/mnt/DeepSeek-V4.1-Flash`. AITER was pinned to
`2039d2b96cd547ebc52f8d55f5f29ec1b8290796`. The runtime explicitly initializes
RCCL without AITER custom all-reduce, matching the accepted P04 policy.

- 655 combined tests passed, zero skipped, in 13.97 seconds: original V4.1
  tests, paging/state tests, generic scheduler/Engram regressions and package
  boundary checks.
- Real ModelRunner execution matched every P04 logit bitwise at 670 positions
  across 17 chunks of English, Chinese, code and a 525-token input. Chunks
  include odd ratio-2 boundaries, decode and a 129-token chunk crossing the
  128-row SWA ring. A real-model padded input also passed.
- 40 scheduled batches covered differing prompt lengths, batch reorder,
  continuous decode, two simultaneous prefix restores at position 80, missing
  image replay from zero, preemption at position 64 with restore from 32,
  cancellation during partial prefill and subsequent slot reuse. Completed
  generations matched P04's offline completions.
- CPU/GPU tests verify byte-exact multi-PAGE checkpoint scatter/gather,
  relocation swaps, rollback and slot recycling; paged candidate selection and
  odd-tail attention continuation pass both tie policies. Empty padded work
  does not mutate PAGE/STATE.
- Ruff passes on the changed implementation except five pre-existing findings
  in `atom/config.py`, verified identical to the accepted base. Black and
  `git diff --check` pass.

The real-checkpoint harness used one full model per rank and peaked at
71.83 GiB of PyTorch allocated memory on rank 0. This is a bounded correctness
run, not a throughput benchmark. Its total elapsed time includes weight loading,
P04 comparison forwards, synchronization and scheduler tests.

```bash
HIP_VISIBLE_DEVICES=0,1,2,3 \
PYTHONPATH=/tmp/atom-dsv41-flash:/tmp/aiter-dsv41-2039d2b96c \
AITER_META_DIR=/tmp/aiter-dsv41-2039d2b96c-meta \
AITER_JIT_DIR=/tmp/aiter-dsv41-2039d2b96c-jit \
AITER_REUSE_IDENTICAL_COMM_GROUPS=1 OMP_NUM_THREADS=4 \
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
torchrun --standalone --nproc_per_node=4 \
  -m tests.attentions.deepseek_v41.validate_runtime \
  --model /mnt/DeepSeek-V4.1-Flash --output /tmp/p05-runtime-tp4.json
```

The harness caps its PAGE allocation; normal ModelRunner uses its measured
memory budget. Reports and logs are retained in
`/app/logs_claude/atom_dsv41_flash_impl_20260912/p05_runtime/`.

## Current execution scope

Use `enforce_eager=True`, `enable_expert_parallel=True` for TP greater than one,
BF16 KV/index storage, and an even cache block size. The native architecture is
`DeepseekV41ForCausalLM`; `small_position` and `large_position` remain configurable
through `index_topk_tie_break` in the HF text config/overrides.

Packed cache, graph capture, speculative decoding, PP/CP/DP, TBO, KV transfer,
plugin execution and EPLB are rejected before loading. Vision and the V4.1
chat/tool protocol belong to later milestones. Host Engram lookup still reads
final GPU IDs on the CPU; request-specific compression/index selection remains
eager. HBM lookup, fused kernels and end-to-end throughput/latency optimization
remain later work, including P09. P05 establishes their tested state/lifecycle
contract without claiming those optimizations are complete.
