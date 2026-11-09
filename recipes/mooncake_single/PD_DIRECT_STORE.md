# V4.1 P/D with direct shared Mooncake Store

This is the source version used by the 2026-11-05 P4+D4, C256, 900-second
AgentX experiment. It builds on PR #2755 source at
`0392e24daf7586cad7740d6db33247d66ae7df2e`, plus this branch's integration
changes. It does not include subsequent #2755 scheduler changes or the proposed
unpack optimization.

P queries shared prefix keys asynchronously after request admission to the
scheduler's waiting set. Store reads target registered GPU slots, then the codec
restores PAGE data to model KV pages. V4.1 state is reconstructed by bounded
replay. There is no LMCache CPU L1 in this path. D receives KV through the P/D
connector and does not run speculative Store prefetch into those same pages.
Each rank still owns its model KV allocations; Store provides shared serialized
KV, not coherent shared GPU memory. Compute/load overlap is not yet quantified.

## Environment and Store

Use Linux containers with the already-installed ATOM GPU dependencies and a
working ROCm-enabled Mooncake SDK. Tested on MI355X, ROCm 7.2.4, Torch 2.10,
Python 3.12, DeepSeek-V4.1-Flash-1017. No binary dependencies are distributed here.
Use the same compatible environment on P and D.

Start an existing Mooncake master/metadata service on ports 19861/19862 and a
DRAM owner before starting P. The experiment used 512 GiB on the P host. In the
owner, supply a valid segment location, e.g.
`allocate_and_mount_segment(segment_bytes, 'rdma', '*')`; an empty location
passed local tests but failed reads by a separate client. Verify a separate
process can put/get before loading the model.

PITT needed a process-local THP/prefault workaround and serial 1 GiB registrations
for the 512 GiB owner on g09. The same workaround did not succeed on g25. This
launcher assumes a working Store and does not configure host memory or RDMA.
Store is CPU DRAM, with RDMA into GPU staging; it is not GPU-resident storage.

## Start P and D

Run from the branch checkout inside each prepared container. Replace IPs, model
path, RDMA device and socket interface with your environment. P and D must use
different hosts with these default ports. `--dry-run` prints the generated
command; without it the launcher writes `launch.json`/`server.log` into a new
output directory and runs in the foreground.

```bash
export MODEL=/models/deepseek-ai/DeepSeek-V4.1-Flash-1017
export P_IP=10.19.0.109 D_IP=10.19.0.125 STORE_IP=10.19.0.109

# On P:
python recipes/mooncake_single/run_pd_direct.py --role prefill \
  --model "$MODEL" --host-ip "$P_IP" --store-ip "$STORE_IP" \
  --gpus 0,1,2,3 --nic rdma0 --socket-interface eno0 --output /tmp/direct-p-01

# On D (set the variables in this shell too):
python recipes/mooncake_single/run_pd_direct.py --role decode \
  --model "$MODEL" --host-ip "$D_IP" --store-ip "$STORE_IP" \
  --gpus 0,1,2,3 --socket-interface eno0 --output /tmp/direct-d-01
```

Both use TP4+DPA, no EP/MoRI, DSpark5, BF16 KV/FP8 index, GPU utilization
0.65, max 64 sequences per rank, and `--dp-load-balance least_requests`.
P has max batched tokens 16384 and explicit `--offload-prefetch`; D has 1024.
There is no session affinity. Do not substitute `prefix_spread` when reproducing
this particular run.

In an environment with `atomesh` installed, launch the router:

```bash
atomesh launch --host "$P_IP" --port 18000 --pd-disaggregation --dp-aware \
  --prefill "http://$P_IP:19300" --decode "http://$D_IP:19300" \
  --prefill-policy cache_aware --decode-policy cache_aware \
  --cache-threshold 1.0 --balance-abs-threshold 0 --balance-rel-threshold 1.0 \
  --atom-pd-rank-mapping-policy none --backend atom --model-path "$MODEL" \
  --disable-circuit-breaker --request-timeout-secs 1800 --prometheus-port 29100
```

Run the AgentX trace against the router (`http://$P_IP:18000`) from a client that
can reach the service, the model tokenizer and the public dataset, with a new
artifact directory:

```bash
export AIPERF_DATASET_WEKA_LIVE_ASSISTANT_RESPONSES=0
export AIPERF_TIMING_CANCEL_DRAIN_TIMEOUT=300
export AIPERF_HTTP_TCP_USER_TIMEOUT=900000

aiperf profile --scenario inferencex-agentx-mvp \
  --url "http://$P_IP:18000" --endpoint /v1/chat/completions \
  --endpoint-type chat --streaming \
  --model "$MODEL" --tokenizer "$MODEL" --tokenizer-trust-remote-code \
  --concurrency 256 --benchmark-duration 900 --stats-interval 30 --random-seed 42 \
  --failed-request-threshold 0.10 \
  --trajectory-start-min-ratio 0.25 --trajectory-start-max-ratio 0.75 \
  --warmup-requests-per-lane 10 --trace-idle-gap-cap-seconds 300 \
  --agentic-warmup-grace-period 1800 --use-server-token-count \
  --no-gpu-telemetry --num-dataset-entries 393 --slice-duration 1.0 \
  --public-dataset semianalysis_cc_traces_weka_062126 \
  --output-artifact-dir /tmp/agentx-pd-01 --no-server-metrics
```

Check actual generation and a cross-rank prefix restore before benchmarking;
health alone does not prove KV correctness.

## Observed result and limits

7983 completed requests, zero request errors, 23 boundary cancellations.
TTFT mean/P50/P90/P99: 1.712/1.295/2.759/10.464 seconds; mean ITL 14.89 ms;
output throughput 6060 tokens/s. AIPerf's throughput includes the 300-second
cancel drain in a 1230-second denominator. Cancelled credits did not all return
before that deadline, so the client forced phase completion despite exit code 0.

Compared with the historical sticky baseline, mean TTFT improved 8.2%, output
throughput declined 5.9%, and mean ITL increased 15.8%. D changed from g20 to
g25 and code/storage changed too: this is not an isolated routing A/B.

V4.1's 739840-byte chunks occupy 741376-byte aligned slots. The current
contiguous fast path rejects the padding, so loads use an extra GPU staging
copy. Windows are also fetched/restored serially. These optimizations are
pending, not part of the measured version. Real KV sharing across DP ranks on
one P host is verified; cross-host byte RDMA is verified, but cross-host real
KV reuse between two P instances is not yet covered by this experiment.
