# Pull collectives over IPC-mapped pools, four ranks

## What it does

The SP-prefill collectives of MiniMax-M3.1-Flash TP4 in FlyDSL instead of RCCL: the all-to-all with the MXFP8 unpack
into o_proj_full's input, all-gathers of row-major fields, the reduce-scatter, and the SP norm tails fused into them
(callers: `seq_parallel_handoffs`, MoE `gather_mxfp4_moe`). IPC pools, two slots and uncached flags per rank; a
producer publishes once its bytes reach memory, persistent pullers spin on the peer's flag and read with
system-coherent 16-byte loads. Bit for bit (`reference.py`): gathers in rank order; `unpack_reference`; reduce-scatter
`bf16((((+0 + b0) + b1) + b2) + b3)` in fp32 (RCCL's ring: up to 0.25 off); slot combine `(((s0+s1)+s2)+s3)+shared`
in fp32; norms per `norm/gemma_add_rmsnorm_h6144` (quant of the fp32 norm).

## Entry points

All are built collectively (gloo, same sizes on every rank), warmed up collectively on the current stream before
capture, one per ordered stream or DBO microbatch, called in the same order on all ranks; pools live for the process.
`Pools`, `Collectives` and `FencedAllGatherComm` keep their sequence on the host (eager only); the others keep device
counters, so eager calls and graph replays interleave.

- `Pools`: the IPC data pools and flag words of a rank, one set per microbatch.
- `Collectives`: one phase's slot of a set, bound to its stream; `alloc` gives a slot view the producer writes and the
  collective reads in place.
- `FencedAllGatherComm.all_to_all_unpack_shuffled`: the attention rows' all-to-all into o_proj_full's MXFP8 codes and
  scales in `mxfp8_gemm`'s preshuffled layout.
- `AllGatherComm.all_gather`: gathers contiguous 16 B-aligned row-major fields in rank order.
- `SlotsReduceScatterComm.slots4_reduce_scatter_add_rmsnorm_mxfp8`: the MoE tail (slot combine, shared add,
  reduce-scatter, residual add, norm, quant) into row-major codes, shuffled scales and the residual;
  `slots4_reduce_scatter` and `reduce_scatter` stop after the sum. Rows a multiple of 4; 768 MiB at 32768 rows.
- `NormAllGatherComm.norm_all_gather`: the dense norm + quant fused into the gather (codes, scales, residual).

Each communicator's host file picks its kernels by rows (`slots4_host.py`, `gather_host.py`,
`all_gather_fence_h6144.py`, `norm_gather_h6144.py`).

## Performance

Eager, rank MAX, median us, 16 inputs; the MoE FFN tail: the slot sum, shared add, RCCL reduce-scatter and
norm as separate launches vs `slots4_reduce_scatter_add_rmsnorm_mxfp8`.

| tokens | MoE tail: separate | fused |
|---:|---:|---:|
| 4096 | 336.25 | 241.25 |
| 8192 | 634.33 | 465.58 |
| 16384 | 1308.35 | 915.65 |
| 32768 | 2595.59 | 1826.27 |

## Tests

```bash
python3 -m pytest tests/collectives/pull_collectives_ranks4 -q                        # 4 GPUs
python3 benchmarks/collectives/pull_collectives_ranks4/benchmark_slots4_reduce_scatter.py r.json
```
