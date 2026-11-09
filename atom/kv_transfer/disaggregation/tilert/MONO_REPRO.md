# V4.1 C1 decode / Mono reproduction handoff

Branch: `lirzhang/atom-tilert-pd` in ROCm/ATOM. The branch includes #3105
(`d46eb22`), which wakes idle KV waits on incoming requests and worker reports.
Goal: investigate the ATOM Mono decode latency gap relative to TileRT.

## GPU budget

**If 4P8D is unavailable, use 4P4D with concurrency1 for ATOM P/D diagnostics.**
Keep the D parallelism explicit and verify actual Mono kernel execution in
a trace. A smaller layout is useful for debugging, but its result must be
labelled separately from the TP8 result below.

**The TileRT0.1.7 V4.1 comparison itself requires D TP8.** Its Python
`model_args.py` defines `TP_CHOICES=(8,)`; four visible GPUs cannot run this
decoder. Use 4P8D for the exact ATOM-P + TileRT-D reproduction. P4/D8 here
means four/eight physical GPUs, not four/eight server instances.

## Measured ITL

All three runs use C1, DSpark5, fixed mean acceptance length3.51.
ITL is AIPerf `inter_token_latency`, in milliseconds, aggregated over requests.

| Configuration | Duration | Requests | ITL mean | P50 | P90 | P99 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Previous ATOM P4 + ATOM D TP8, Mono enabled, KV wait yield fix | 1800s | 209 | 1.375477 | 1.298533 | 1.475755 | 3.699538 |
| This branch: ATOM P4 + TileRT D TP8, #3105 included | 900s | 149 | 0.726997 | 0.718253 | 0.744409 | 0.916183 |
| SA: vLLM P TP4 + TileRT D TP8 | 3600s | 473 | 0.722746 | 0.720143 | 0.742346 | 0.783187 |

TileRT reduced mean ITL47.15% versus the previous ATOM decoder and P90
ITL49.56%. The new mean is0.59% above SA, with P90 only0.28% above SA.
This makes the ATOM decode implementation/launch schedule the main target
for Mono profiling. The runs cover different durations and request lengths;
these are historical comparisons, not a same-window controlled A/B.

SA source:
https://github.com/SemiAnalysisAI/InferenceX/actions/runs/42159378057/job/127760303692
This is vLLM prefill + TileRT decode, not a vLLM decode result.

## Start ATOM prefill (P4)

Inside an existing ATOM GPU image, mount the branch as `/workspace` and the
same DeepSeek-V4.1-Flash-1017 model as `$MODEL`. Install TileRT0.1.7's Python
wheel with `pip install --no-deps /path/to/tilert-0.1.7-...whl`; do not replace
ATOM's torch/aiter with the TileRT decoder environment.

```bash
export PYTHONPATH=/workspace
export HIP_VISIBLE_DEVICES=0,1,2,3
export ATOM_MONO_ENABLE=1 ATOM_KV_OFFLOAD=none
export MC_NUM_QP_PER_EP=1 MC_GID_INDEX=1
export HSA_ENABLE_IPC_MODE_LEGACY=1 RDMAV_FORK_SAFE=1
export MOONCAKE_DISABLE_HIP_DMABUF=1
python -m atom.entrypoints.openai_server \
  --model "$MODEL" --trust-remote-code -tp 4 \
  --enable-dp-attention --moe-backend standard --all2all-backend none \
  --dp-load-balance least_requests --server-port 19609 \
  --kv_cache_dtype bf16 --index-cache-dtype fp8 \
  --gpu-memory-utilization .65 --max-model-len 1048576 \
  --max-num-seqs 4 --max-num-batched-tokens 16384 \
  --attn-prefill-chunk-size 16384 --enable_prefix_caching --block-size 128 \
  --state-checkpoint-interval-tokens 8192 --level 0 --enforce-eager \
  --method dspark --num-speculative-tokens 5 \
  --kv-transfer-config "{\"kv_role\":\"kv_producer\",\"kv_connector\":\"tilert\",\"tilert_max_seq_len\":1048576,\"tilert_host\":\"$D_IP\",\"tilert_ctrl_port\":19641}"
```

Set DP master/base ports explicitly if sharing a host with other servers.
Clear inherited `MC_DISABLE_HIP`, Store/offload and profiling variables.
The producer DPA4 may be ineligible for Mono; the Mono comparison concerns
the previous **pure TP8 D, with DPA disabled**. Setting the env flag alone
does not prove execution: confirm the trace contains the Mono kernel.

## Start TileRT decode (D8)

Use the SA TileRT ROCm decode environment and TileRT0.1.7. The tested base is
`ghcr.io/tile-ai/tilert-rocm-decode@sha256:48c30d7636a561b4edf781ac3c7712e09fa9a0b2f1292863b72c2d8e4bcb492a`
with the0.1.7 wheel installed. This environment uses torch2.12/ROCm7.14.
Do not run it in the ATOM image. Pass `/dev/kfd`, `/dev/dri`,
`/dev/infiniband`, host networking/IPC and unlimited memlock to containers.
Use `nofile=65536:524288`. Reuse the cluster's known-working RDMA provider
mounts/configuration; GID index1 is the PITT setting and may differ elsewhere.

```bash
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export HSA_ENABLE_IPC_MODE_LEGACY=1 RDMAV_FORK_SAFE=1
export MC_NUM_QP_PER_EP=1 MC_GID_INDEX=1 MOONCAKE_DISABLE_HIP_DMABUF=1
export TILERT_PD_SHARDS=1 TILERT_DECODE_POLL_MS=2
export TILERT_SIMULATE_ACC_LEN=3.51 TILERT_SIMULATE_ACC_METHOD=match-expected
python -m tilert.pd_vllm.dsv41_decode_server --engine tilert \
  --model-weights-dir "$MODEL" --max-seq-len 1048576 \
  --ctrl-port 19641 --http-port 19642 \
  --transport mooncake --with-mtp --num-mtp 5
```

Wait for both `TileRT engine ready` and `buffer registered`, and check
`/health` on both servers. Remove diagnostic `LD_PRELOAD` hooks before timing.

## HTTP bridge and AgentX

The installed TileRT vLLM router needs three ATOM adaptations: remove the
prefill logprobs request, set `do_remote_decode=true`, and take the `rid` and
`first_token_id` from ATOM's `kv_transfer_params` response. The helper
`patch_router.py` in this directory produces a separate router file from
the installed TileRT0.1.7 router without modifying its package.

```bash
python /workspace/atom/kv_transfer/disaggregation/tilert/patch_router.py /tmp/atom_pd_router.py
python /tmp/atom_pd_router.py --vllm-url "http://$P_IP:19609" \
  --decode "$D_IP:19641:19642" --port 19643 --model dsv41 \
  --parser none --model-path "$MODEL" --queue-timeout 1800
```

Router tokenization may disable itself if the HF tokenizer has no chat
template; in that case ATOM P performs tokenization. Confirm a streaming
chat request finishes with `[DONE]` before starting the client.

Use the InferenceX AgentX AIPerf build used by SA (our runner records
commit `754356e9a39acc6cc6afb242d123bb57c3fb6f75`), not a generic stock build:

```bash
export AIPERF_DATASET_WEKA_LIVE_ASSISTANT_RESPONSES=0
aiperf profile --scenario inferencex-agentx-mvp \
  --url "http://$ROUTER_IP:19643" --endpoint /v1/chat/completions \
  --endpoint-type chat --streaming --model "$MODEL" \
  --tokenizer "$MODEL" --tokenizer-trust-remote-code \
  --concurrency 1 --benchmark-duration 900 --random-seed 42 \
  --trajectory-start-min-ratio .25 --trajectory-start-max-ratio .75 \
  --warmup-requests-per-lane 10 --trace-idle-gap-cap-seconds 300 \
  --agentic-warmup-grace-period 1800 --num-dataset-entries 393 \
  --public-dataset semianalysis_cc_traces_weka_062126 \
  --use-server-token-count --no-gpu-telemetry --no-server-metrics \
  --output-artifact-dir /tmp/atom-tilert-c1
```

Preserve `profile_export_aiperf.json` and per-request `profile_export.jsonl`.
Compare profiling records only, excluding warmup. For a controlled Mono
comparison use the same hardware, replay, duration and fixed acceptance;
record D TP/DPA and the actual kernel in the trace.

## Experimental caveats

The producer currently sends40 backbone windows; TileRT initializes missing
draft windows empty. Fixed acceptance3.51 is for performance experiments,
not correctness validation. Whole-request packing and transfer are
synchronous. The900s run completed149 requests with zero errors/cancellations,
but this does not establish production readiness or correctness without
simulated acceptance.
