# ATOMesh PP4 to DEP4, MoRIIO WRITE with CPU offload

This experiment retains the fixed M3.1 worker image, compute kernels, model,
P TP1×PP4 / D TP1×DP4+EP4, natural P device order, C80, D max-num-seqs20/rank,
DSpark7/capture160 and the SimpleCPUOffload sibling on P. MoRI uses RDMA WRITE,
four native workers and four QPs per transfer. MoRI native workers are different
from Mooncake's Python sender threads.

The fixed image predates PP support in its installed MoRI connector. The three
production files from merged vLLM #64693 are backported without changing model
code. Both sides remain TP1, so heterogeneous-TP NVFP4 #68072 is not needed here.
The image build verifies original and patched source hashes, imports the actual
connector and offloader, and checks real PP port and completion methods. The old
completion method finishes after four duplicate messages from one stage; the
patched method waits for four distinct stages.

The router is ATOMesh, built at the exact source SHA recorded in the contract.
Its WRITE adapter injects static MoRI peer metadata before concurrent P/D HTTP
dispatch. D uses native internal DP load balancing; the selected rank notifies P.
The real-worker method test proves ranks0–3 can notify with WRITE metadata and
that reusing READ's rank override incorrectly suppresses ranks1–3.

The srtctl patch generates peer hosts/ports from the same resolver used for the
workers, preserves the MultiConnector offloader, and disables ZMQ discovery.
The runtime verifies the installed srtctl files and router binary/build hashes.
The retired vllm-router contract is retained only as cancelled historical input;
the current launcher has no dispatch entry for it.

Bounded observations record positive-byte writes after native completion wait
succeeds. Functional acceptance requires the same transfer ID to complete on
all four PP stages, then checks SSE/tools/disconnect and runs the original
AgentX warmup plus900s. The probe does not change native transfer status, model
output, cache addresses, or deadline. This is not a formal SA sweep or accuracy
claim. Source and observation hashes are recorded separately in manifest.json.

Use one gpool lease for the test and release after exact job-label cleanup.
Outputs and host caches use task-owned node-local directories because shared
home quota is full. The read-only collector retrieves the final artifact by SSH.

The follow-up after L25189d adds a one-line worker candidate: native IOEngine
binds port0 atomically instead of reusing an already-closed get_open_port probe.
The earlier run failed with native TCP EADDRINUSE before requests; it did not
record the conflicting port, so the exact competitor is not proven. The CPU
test demonstrates the stale-probe race using the real config method and real
OS sockets. Bounded startup observations record requested and bound ports on
each rank. GPU startup, actual engine-descriptor advertisement and four-stage
WRITE completion must validate this candidate. Model kernels, topology,
offloader and native RDMA worker/QP counts remain fixed.
