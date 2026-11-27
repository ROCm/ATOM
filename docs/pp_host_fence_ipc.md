# Experimental PP transport using HIP IPC memory

`ATOM_PP_HOST_FENCE_IPC=1` replaces six same-host K3 pipeline sends with copies
into receiver-owned HIP IPC memory. The edge between the two hosts keeps the
existing RCCL path. The option is off by default because this public-base port
has not passed GPU or model-output validation.

The sender's CPU worker waits for a process-local producer event, submits copies
to its own stream, and observes stream completion before publishing `READY` on
the edge's Gloo channel. The receiver clones those fields into tensors it owns
and observes clone completion before publishing `CONSUMED`. A three-slot ring
cannot reuse a slot before that acknowledgement. This uses no cross-process GPU
event wait and makes no claim that HIP chose SDMA or that copies consume no CUs.

The model caller can continue computing while the outgoing worker waits. Its
existing send commit still bounds outstanding submissions. Source tensors stay
alive through the copy fence, including handed-over auxiliary views. Received
outputs stay valid after ring reuse. Zero-row messages preserve sequence order
without dereferencing the ring.

## Supported experimental contract

The implementation intentionally retains a narrow source and configuration
binding in `atom/distributed/pp_host_fence/SOURCE-BINDING.json`. It validates the
installed source, Torch/HIP versions, model and draft config bytes on every rank
before allocating rings or exchanging handles. Updating those pins establishes
a new reviewed source set, **not** GPU validation of that set.

- Native ATOM K3, gfx1250, BF16, eager PP8×TP1, with ranks 0–3 and 4–7 on two
  distinct hosts and eight distinct GPUs. Each host uses local devices 0–3.
- Layer partition `7,13,13,13,13,13,13,8`; the bound K3 config has hidden width
  7168, residual block size 12 and draft target taps `[2,23,47,71,89]`.
  On public main the residual bank grows at each block boundary: edges 0–6
  carry `[1,2,3,4,5,6,8]` blocks, unlike native531's fixed eight-slot bank.
- `max_num_batched_tokens=16384`, `max_num_seqs=256`,
  `gpu_memory_utilization=0.88`, no KV-transfer connector, NUMA balancing zero.
- The fields are exactly `hidden_states`, `block_residual` and `pp_aux`.
  Shapes, BF16 dtype, row capacity, source device, sequence, edge and slot are
  checked at their corresponding boundaries.
- IPC and early receive cannot be combined. Unsupported configurations fail
  before payload transfer; a failed IPC edge does not silently fall back to RCCL.

Enable the flag on **all eight ranks**, as a delta to an otherwise matching
validated PP recipe. Other source trees, model configs, runtimes and topologies
are refused. Do not use this option with the TP8 c128 recipe.

The receiving ring is allocated before warmup and KV sizing. Its bytes are
reported at initialization; the runner also reports the resulting non-Torch
memory and KV budget. These costs must remain in any comparison.

## Lifetime and failure limits

Normal shutdown drains outgoing acknowledgements, joins the worker, then closes
its peer mapping. Receiver allocations and failed or partially initialized
objects remain owned until process exit. In-process model reload and recovery
from an abruptly failed peer are unsupported. CPU-channel timeouts do not make
a driver-side stream synchronization interruptible; a GPU or peer failure needs
whole-job teardown by the existing execution owner.

## Validation status

The original native531 implementation has two-GPU field/lifetime evidence and
eight-rank model trace observations. Those runs used ATOM
`5310947e5b8e0977f6351b2d5a98cda5edb3afbe`, not this public base, and do not establish
model output preservation or a throughput improvement for this PR. The changed
public source binding and integration require their own validation.

CPU protocol tests run with:

```sh
python -m unittest discover -s tests -p 'test_pp_host_fence*.py' -v
```

Before the Draft is ready for merge, validate the public head's enabled and
disabled GPU paths, all field values and shapes through slot reuse, the six
same-host edges plus the RCCL boundary, and model output preservation. Compare
the same model, workload and base with the flag off/on; report stage time,
communication/copy cost, KV capacity and model performance separately.
