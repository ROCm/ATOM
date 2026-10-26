# MiniMax prefill hang reduction experiment

Run **only through ATOMesh benchmark CI**:

```sh
gh workflow run atomesh-benchmark.yaml --repo ROCm/ATOM \
  --ref mengqing/moe_scatter_reduce_bounds_ci \
  -f suite=minimax_prefill_repro \
  -f atomesh_image=rocm/atom-dev:nightly_202610141644 \
  -f atomesh_slurm_account=amd-atom \
  -f atomesh_slurm_partition=amd-spur \
  -f run_model_benchmark=false -f publish_dashboard=false
```

The job uses one Slurm node, TP4, a 20-minute allocation limit, and four sequential
subcases, each capped at 180 seconds (plus 15 seconds for termination). The 20-minute
Slurm limit starts after allocation, excluding time in PENDING. The Actions ceiling
is 24 hours, allowing long waits under `amd-atom-qos`. To clean up a previous
completed CI run before submitting, pass `-f repro_cleanup_run_id=<GitHub-run-ID>`;
only this user's reproduction jobs matching that exact run are cancelled. The
preflight log is uploaded before entering the queue. The existing
TW workflow uses an exclusive node allocation, so this does **not** promise a
half-node billing reduction. Savings come from removing model downloads/loading,
decode/router startup, and AgentX warmup. Each case repeats the two-batch transition
64 times. Initialization/JIT also counts toward the per-case deadline.

| Case | Offload streams | Resident allocation target |
| --- | --- | --- |
| `none-resident0` | none | workload only |
| `two-resident0` | pack + copy | workload only |
| `single-resident0` | shared pack/copy | workload only |
| `two-resident244` | pack + copy | 244 GiB before dense gather output |

All ranks run on one node. Bootstrap sockets use the exact loopback interface
(`NCCL_SOCKET_IFNAME==lo`, IPv4); RCCL still selects GPU P2P transport. Run
`38703620681` failed during its first barrier because automatic selection chose
`eno1:10.13.0.134`, returning `No route to host`. That initialization failure
occurred before long-KV replay/offload and is not evidence of the original hang.

The input reproduces the consecutive 7-request / 8-request cache-length batches
from run `38664663067`, attempt 1: total KV 794,109 -> 1,350,868. Gather uses the
checkout's verbatim fixed Triton kernel: 16 KV heads per rank, head dimension 128,
FP8 SHUFFLE cache, per-token scales and BF16 dense outputs (6.06 -> 10.31 GiB/rank).

The asynchronous save reuses the actual ATOM fused staging kernel with simplified
single-segment storage, 28,016,640 bytes per save, four chunks, pinned host memory,
producer-event host waits, and staging-buffer pack/copy reuse events. Sampled IDs
are broadcast as 7/8 int64 elements on a separate TP process group and copied to
CPU on another stream; the previous output is consumed on the next iteration.

This is a **candidate**, not an established minimal reproducer. It omits the
MiniMax target and EAGLE3 attention/MLP, aiter custom/PyNCCL collectives, draft CUDA
graphs, the full LMCache connector/worker scheduling, Mooncake, and decode. The
model allreduce is a small torch/RCCL stub. No aiter installation is needed by the
operations in this first reduction. The resident-memory case is an explicit
allocation-pressure experiment, not proof of an OOM in the original run.

Artifacts contain per-rank stage logs, RCCL logs, Flight Recorder dumps on timeout,
thread dumps, extracted gather sources/hashes, image digest, Slurm logs/status, and
one result JSON per successful rank. No dashboard data is published.

A green job with 16 completed rank results means **the reduced workload did not
reproduce the hang**. A red job needs classification: missing `warmup_complete`
means initialization/build/setup did not finish; a later timeout is only a hang
candidate until rank stages and collective evidence match the original failure.
A HIP memory fault, allocation error, or mismatched assertion is a different
failure. The launcher stops on the first failure to avoid scheduling further
experiments onto a possibly stalled device.
