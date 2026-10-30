# MiniMax M3.1 PP4 to TP4 KV landing

This builds on the gather/landing protocol from ATOM PR #2431. The validated
PP4/TP4 base is `87133902f`; model computation and KV layouts are unchanged.
Enable `ATOM_PD_MHA_LANDING=1` on P and D. The default is disabled so the
existing chunk OFF/ON cases remain available as baselines.

## Data path

Each TP1 prefill stage gathers the exact head slices needed by one decode TP
rank into destination-PAGE order. This includes target K/V, NVFP4 scales,
FP4 index/scales, and the draft stage's FP8 K/V/scales. Index planes remain
replicated as specified by the existing layout. Packing copies raw bytes,
without quantization or dequantization.

Each decode rank has its own registered GPU landing pool and scatters into
its own KV allocations. A slot packs complete destination PAGEs, potentially
from several layers/planes, and uses one RDMA descriptor. No new cross-GPU
scatter or decode-side collective is introduced. A wire row is one whole
physical PAGE (`mha_pages_v1`), so page-16 SHUFFLE and page-128 NVFP4 do not need
a token-contiguous layout. The consumer's scatter uses the existing segment
copy kernel and waits for the compute-stream event recorded at admission.

All source/destination head geometry and semantic roles are checked for the
whole PP stage before sending. Source PAGEs and destination cursor offsets
are validated before gathering. Staged direct writes never cross a destination
PAGE, preserving existing memory-registration chunk boundaries.

## Chunking, completion and fallback

The landing switch is independent of `enable_chunked_transfer`. Chunked
requests keep READY sequence numbers across chunks and use offsets into the
full prefix-trimmed destination block table. Cached-prefix PAGEs are untouched.
A chunk's source event precedes gathering. Completion requires every PP stage's
write-done, every announced landing slot's scatter, and matching valid final
prefill handoff metadata. Duplicate and wrong-nonce results cannot advance the
handoff quorum.

The receiver reuses #2431's per-stage partitions, credits, lost-READY handling,
ack deadlines and late-message handling. A request needing fewer than two
landing slots uses staged PAGE writes. If a credit does not arrive within
10 ms, remaining PAGEs use that same gather followed by direct writes, without
waiting for more credits. A consumer without a landing offer can use this
staged path too; a producer without MHA staging keeps the original direct path.

A gather must finish before its source slot is reused, including exceptions.
A failed RDMA quarantines the remote credit and permanently retains the local
staging pool until connector teardown, because the NIC may still read it;
waiters are awakened with an error and later requests use the original direct
path. Successful RDMA is recorded in the write-done manifest before READY can
fail to send. Decode still observes the existing failed-write page hold.

## Memory and scope

P allocates one 8 MiB staging slot per send worker, up to 256 MiB per GPU.
D allocates 32 x 8 MiB landing slots plus a 4096-column int64 scatter table
(96 KiB) per GPU. Both are reserved through `KVConnectorFactory` before KV
cache sizing. Gathering uploads one int64 source block index per chunk; the
copy is synchronized before that metadata is released. The first version is
limited to MiniMax M3, TP1 P and TP-sharded D, without DCP or slot-state pools.
The existing MLA/DCP path and Store's persistent KV layout are unchanged.

## Validation and CI recipes

`tests/test_mha_landing.py` compares the real NVFP4/FP4/index/draft physical
views byte-for-byte for all four target ranks, including sparse and dense PP
stages, fragmented block IDs, prefix skips, multiple chunks, credit exhaustion,
failed writes, gather exceptions, and handoff disagreement. Its GPU tests run
the actual gather and scatter kernels with a synchronous simulated NIC and
repeated landing-slot reuse; they do not measure RDMA performance.

Run the following accuracy cases before performance:

- `minimax-m31-flash-mxfp4-1p1d-pp4-tp4-dspark7-nvfp4-gsm8k-5shot-landing-c32`
- `minimax-m31-flash-mxfp4-1p1d-pp4-tp4-dspark7-nvfp4-gsm8k-5shot-landing-chunked-c32`

They retain all 1319 GSM8K samples, five-shot prompts, the 94% flexible gate,
and real DSpark K7. Then compare these C32 cases against the existing cases
without `landing` on the same revision and, where possible, the same node:

- `minimax-m31-flash-mxfp4-1p1d-pp4-tp4-dspark7-nvfp4-mcstore-agentx-landing-c32`
- `minimax-m31-flash-mxfp4-1p1d-pp4-tp4-dspark7-nvfp4-mcstore-agentx-landing-chunked-c32`

All new cases use automatic placement. The C32 workload keeps P's 8192-token
budget, D's 32768-token budget, AgentX 1.0.12, synthetic AL4.04, 1 TiB Store
and one-hour measurements. Record the actual KV capacity, landed bytes/slots,
RDMA descriptor counts, gather/scatter cost, KV-ready wait, TTFT and ITL alongside
both p90 interactivity metrics and throughput. Reduced descriptor counts alone
do not establish improved model performance or explain the earlier PP OFF ITL.

Local validation on 2026-10-30: 608 tests passed across the MHA landing,
M3 physical layouts, chunk protocol, MLA landing/staging, model-runner fence,
connector topology, and PP suites. GPU tests ran on MI308X with ROCm 7.2.4;
the NIC was simulated. Black, Ruff and whitespace checks passed. The actual
CI matrix parser expanded all four cases for `atomesh-cicd`, with automatic
placement and identical inherited workload settings. Full-model MI355X
accuracy and C32 measurements for this optimization have not run yet.
