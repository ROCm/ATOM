# MiniMax-M3 stalled-length full-model replay

This is a **candidate**, not a confirmed minimal reproducer. Original HTTP bodies
and token IDs were not retained. The user authorized constructing synthetic text
from the observed lengths. Run the complete MiniMax-M3 target and EAGLE3 draft,
TP4 prefill + TP4 decode on one node, draft CUDA graphs, Mooncake RDMA and LMCache
256 GiB/rank, using the same server configuration as the original C32 case.

```sh
gh workflow run atomesh-benchmark.yaml --repo ROCm/ATOM \
  --ref mengqing/moe_scatter_reduce_bounds_ci \
  -f suite=nightly \
  -f case_names=minimax-m3-fp4-1p1d-tp4-eagle3-stalled-lengths-replay \
  -f atomesh_image=rocm/atom-dev:nightly_202610141644 \
  -f atomesh_slurm_account=amd-atom \
  -f atomesh_slurm_partition=amd-spur \
  -f run_model_benchmark=true -f publish_dashboard=false
```

Slurm allocation time is capped at **20 minutes**, including image/setup, model
loading, graph capture and requests; PENDING time does not count. The Actions
ceiling allows 24 hours for queueing. This remains a full eight-GPU node. It reduces
request workload, not the resources needed to load the model. Startup or allocation
timeout before the target requests is not a successful reproduction of the hang.

The client runs after the normal P/D workers and mesh router become healthy.
It sends eight sequential cache primers, then eight concurrent target chat
requests with max_tokens=1 and ignore_eos=true, matching the original warmup's
output length. There is no AgentX dataset reconstruction or performance phase.
The server capacity remains 80, and its batched-token budget remains 32768.

Source: run `38664663067`, attempt 1, prefill log physical line 3329, 01:37:53 UTC.

| Original request ID | Full prompt | Frontier at stalled step | New tokens at stalled step |
| --- | ---: | ---: | ---: |
| 68 | 142015 | 131072 | 10943 |
| 65 | 149072 | 146688 | 2384 |
| 69 | 165518 | 165376 | 142 |
| 70 | 178412 | 176000 | 2412 |
| 73 | 172112 | 171776 | 336 |
| 71 | 175618 | 171648 | 3970 |
| 72 | 203129 | 190720 | 12409 |
| 74 | 170398 | 164864 | 128 |

Request 74's 164992 tokens at the stalled step are only a partial prompt.
Request 68 originally started at a 104064-token cache hit and reached 131072
in the previous batch. This reduction primes directly to the stalled frontier.
The eight primers replace earlier conversations; they do not reproduce the
original request order, allocator history, offload backlog or MoE routing.

The tokenizer constructs deterministic varied-word chat text (seed 42 + original
request ID). It uses the same thinking_mode=enabled template and encode defaults
as the service, verifies each full prompt length and the shared 128-token block
boundary, and rejects cross-request sharing beyond the common system-template blocks
(the real MiniMax template shares 128 tokens). Text is
sent through `/v1/chat/completions`: mesh's current request schema drops an unknown
`prompt_token_ids` field. Chat template suffixes are excluded from the shared
prefix calculation. A client barrier starts eight prepared HTTP requests together;
server arrival order and batch packing are not forced. Actual cache hits may also
be lower due to eviction or connector scheduling.

Artifacts under `benchmark_results/stalled-lengths-replay/` include:

- Exact HTTP bodies and local prompt token IDs (gzip), token hashes and lengths.
- Per-request started/completed/failed events and streaming responses.
- `replay-summary.json`: observed prefill batches, maximum total KV, and whether
  the original ordered done/new vectors were seen (1350868 total KV tokens).

HTTP errors, missing stream terminators, wrong prompt lengths or wrong output
lengths fail the case. Requests have a 660-second client deadline; the 20-minute
allocation limit remains authoritative. There are no client retries. Normal
prefill, decode and router logs are uploaded by the existing CI artifact path.
A green job means these 16 requests completed without reproducing the hang;
it does not prove the original C32 issue is fixed. A red job must be classified
from the first error and actual batch shapes. A timeout alone does not establish
the same NCCL/offload failure.
