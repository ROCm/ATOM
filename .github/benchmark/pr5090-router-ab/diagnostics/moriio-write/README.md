# PP4 to DEP4, MoRIIO WRITE with CPU offload

This experiment retains the fixed M3.1 worker image, compute kernels, model,
P TP1×PP4 / D TP1×DP4+EP4, natural P device order, C80, D max-num-seqs20/rank,
DSpark7/capture160 and the SimpleCPUOffload sibling. MoRI uses RDMA WRITE,
four native workers and four QPs per transfer, as the existing M3.1 MoRI recipe.
MoRI native workers are not equivalent to Mooncake's Python sender threads.

The fixed image predates PP support in its installed MoRI connector. The three
production files from merged vLLM #64693 are backported without changing model
code. Both sides remain TP1, so heterogeneous-TP NVFP4 #68072 is not needed here.
The image build verifies original and patched source hashes, imports the actual
connector and offloader, and runs CPU checks of the real PP port and completion
methods. The old completion method incorrectly finishes after four duplicate
messages from one stage; the patched method waits for four distinct stages.

The router is the existing vllm-router v0.1.16 WRITE/discovery path. ATOMesh's
current native-vLLM MoRI adapter is READ-only; this experiment does not claim an
ATOMesh WRITE result or a router-only performance comparison.

The small srtctl overlay allows MoRI discovery to carry different P/D DP sizes
instead of applying a uniform static DP expansion. The pinned router's
`src/routers/http/vllm_service_discovery.rs` at v0.1.16 stores TP/DP sizes per HTTP
address and service type. A real SrtConfig PP4/DEP4 case fails before the overlay
and passes after it. The Windows frontend suite has64 passing tests and one
unrelated POSIX-path-separator assertion; only an unused fcntl import is stubbed.
Full Linux configuration and imports are checked again before GPU model startup.

Bounded Python observations record positive-byte writes only after the native
completion wait succeeds. Functional acceptance requires the same transfer ID
to complete on all four PP stages, then checks SSE/tools/disconnect and runs the
original AgentX warmup plus900s. The probe does not change native transfer status,
model output, cache addresses, or deadline. Source and observation hashes are
recorded separately in `manifest.json`.

Use one gpool lease for the test and release after exact job-label cleanup.
Outputs use a task-owned node-local directory because shared home quota is full;
the existing read-only collector retrieves the completed artifact through SSH.
