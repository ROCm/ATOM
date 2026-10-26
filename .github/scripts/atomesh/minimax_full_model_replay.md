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

## Deterministic admission and stage diagnostics

The follow-up case is
`minimax-m3-fp4-1p1d-tp4-eagle3-stalled-lengths-gated-replay`.
Use the same dispatch command above with this case name. It retains the same
model configuration and 20-minute allocation limit. The original case remains
available as the control.
Both full-model replay cases use the Mesh release binary already installed in
`rocm/atom-dev:nightly_202610141644` at
`/app/ATOM/atom/mesh/target/release/atomesh`. The image's ATOM revision
`6e9b3e1c5` and this replay branch have the same `atom/mesh` source tree
(`55ad15ac25935554c211538e0611bbca26bc567c`). The workflow therefore skips
Mesh compilation for `minimax_full_model_replay`, avoiding Rust installation
and the shared Cargo source cache that failed in run `38728945101`
(`aws-lc-rs-1.18.2/Cargo.toml` missing). No cache deletion is needed.
Recheck Mesh compatibility if the image or router source changes. Agentic
cells build their reviewed Mesh source unless they explicitly select
`benchmark.mesh_source: image`, as the C32 case does.

After all eight primers return, the client atomically creates `armed.json`.
The prefill EngineCore continues receiving requests and draining idle connector
work, but holds target scheduling until all eight distinct prompt lengths are
present. It then orders them as 68,65,69,70,73,71,72,74 and calls the normal
scheduler. Before dispatching any target GPU work it verifies request IDs,
`done`, and `new` against the original batch. The gate releases only once.
Unexpected requests, a 30-second admission timeout, or a shape mismatch fail the
case; `gate-batch.json` records the observed shape. No dummy GPU barrier is added.

LMCache host-memory preparation is enabled on the original C32 AgentX benchmark,
not on this diagnostic replay. See [the C32 setup](minimax_m3_c32_lmcache.md).

`prepare_minimax_replay_debug.py` copies the read-only mounted `atom/` package
into a fresh container-local `/tmp` directory and applies exact-anchor hooks
there, after validating all source anchors and Python syntax. Both services
launch with that directory as their working directory; the launcher verifies
package resolution before startup. The mounted and committed `atom/` sources
are unchanged. Artifacts include the source paths, runtime diff,
before/after source hashes, and the copied debug module. This is diagnostic code,
not a production bugfix. Startup and primer execution pass through the hooks
without enabling diagnostic GPU events.

Once armed, both service roles record per-process JSONL with rank, PID, thread,
request IDs and stage. Hooks cover model input preparation, target forward,
draft context KV, sampling/postprocess, sampled-token broadcast, draft propose,
result copy and previous-result wait, offload producer wait/store, and staging
allocation/pack/copy/final wait. They sit outside compiled model operator bodies.
CPU arrival is persisted before HIP/allocator calls. Independent GPU events are
recorded on the relevant stream and queried by a background thread, without any
new synchronize/wait dependency. Allocator allocated/reserved bytes, retry and
OOM counters are sampled at model-stage boundaries. This instrumentation can
still perturb timing, so a successful diagnostic run is not proof of absence.

The client requests Flight Recorder dumps after 30 seconds without completion
of the target group. Native faulthandler independently dumps all Python thread
stacks every 35 seconds while the target phase is active, including when an
event-query thread itself blocks. A 75-second target-request timeout leaves time
for diagnostics; primers retain their original timeout. Slurm's 20-minute limit
still overrides all of these. Source-stage tracebacks and Python stacks do not
replace GPU kernel traces or native C++ thread stacks.

Flight Recorder uses a buffer of 8192 entries, C++ stack capture and timing.
Trigger FIFOs live in a fresh container-local `/tmp` directory (not NFS or the
artifact tree); dumps go into `diagnostics/prefill/` and `diagnostics/decode/`.
If the installed PyTorch build does not expose the configured FIFO, the client
records an empty pipe list instead of claiming a successful dump. Normal run
completion writes `client-done.json`, stops the diagnostic watchdogs and permits
normal service cleanup. CPU unit tests cover gate ordering, premature release,
shape mismatches, bounded admission, atomic patch failure, event record/query,
and FIFO triggering; actual model execution remains CI-only.
