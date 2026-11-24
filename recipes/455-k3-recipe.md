# Kimi-K3 1P1D on MI455X (gfx1250): P = PP8 × TP1, D = DP8 EP8

> [!WARNING]
> **Reconstructed, not re-run. These commands may be wrong; please correct this file when you run it.**
>
> The G13 trays this 1P1D ran on died (P on g13-03 + g13-11, D on g13-05 + g13-08). The launch kit went with them:
> `kit_pp` with `make_pd_launchers.py`, `make_dpep_launcher.py` and `run_pd.sh`.
> Everything below was pieced together from the run history and checked against the flags and environment variables
> this branch defines. Nobody has launched it as written.
>
> What has been checked, on a CPU host only:
> - This branch's own argument parser accepts both server commands.
> - Without DSpark, the engine `Config` builds for both: multi-node PP for P (four stages per tray), multi-node
>   DP with EP for D.
> - The DSpark flags parse. Their `Config` imports the GPU stack, so it was not built.
>
> Sources, all in wuhuikx/LLM-Booster:
> - #578: comment 1 (the 1P1D handbook, TP8 + DCP8 era); comments 4, 10, 15, 20 and 27 (the PP → DP8EP8 runs; 27 is
>   the 11-22 status with the best configuration per point).
> - #635: the P route, PQ4.
> - #714: the PP → DP+EP transfer.
> - #246.

## What it reproduces

16 MI455X GPUs on four Helios trays. Prefill (P) is one Kimi-K3 instance pipelined over 8 GPUs, PP8 × TP1, on two trays.
Decode (D) is one instance with DP attention and expert-parallel MoE over 8 GPUs, DP8 EP8 + MegaMoE, on the two other
trays. The MoRIIO connector writes KV and KDA state from P to D over the UALink fabric (`moriio_backend: fabric`, write
mode). The router is atomesh in PD mode.

Best measured results (11-22, MLPerf Endpoints agentic workload, natural DSpark acceptance, whole window):

| point | tok/s/GPU | tok/s/user | TTFT p50 / p90 | TPOT p50 | run |
|---|---|---|---|---|---|
| c1024 | 737 | 20.5 | 7.47 / 9.49 s | 27.4 ms | PD36 |
| c512 | 603 | 38.6 | 1.99 / 2.97 s | 19.8 ms | PD29 |
| c192 | 411 | 53.1 | 1.72 / 2.77 s | 14.4 ms | PD18b (PD29: 410) |

GSM8K 5-shot through the router: 0.966–0.968.

## Software

- **Code: this branch with ROCm/aiter `k3-455-dev`.**
  - This ATOM branch carries the PD code, one commit per PR. Those commits are the same code the runs used (internal
    tree `82385ab30`), rebased.
  - aiter `k3-455-dev` carries the MegaMoE S1F fix: #8189 + #8483.
  - No image built from the two branches exists yet (11-24).
  - The runs used `rocm/fw-bringup:gfx1250-atom-k3-20261119` (aiter face19800, mori 695d4c03, FlyDSL 0.5.0) with the PD
    tree first on `PYTHONPATH`.
  - Putting this ATOM on `PYTHONPATH` in such an image works the same way. The aiter in those images is much older than
    aiter `k3-455-dev` (hundreds of files, C++ included), so build an image from both branches. Without the S1F fix,
    D needs `AITER_MEGA_S1F=0`.
- **Model.** `Kimi-K3` and the DSpark draft `Kimi-K3-DSpark` (Inferact) in one directory, mounted at `/models`.

## Host preparation (all four trays)

- **Same vPOD.** All four trays must be in the same UALink vPOD, with the fabric ACTIVE.
- **Load the driver together.** Power-cycle the four trays in one batch and load the driver on all four together
  (`sudo modprobe amdgpu`). A tray from another power-cycle batch takes a fatal error when it imports memory from the
  others.
- **After every boot:** `echo 0 | sudo tee /proc/sys/kernel/numa_balancing`.
- **Before every run:**
  - Run `sync; echo 1 | sudo tee /proc/sys/vm/drop_caches`.
  - On 1 TB trays, start a memory guard: drop caches when free memory falls below 64 GB, and stop the local server on
    a KFD eviction storm.
  - The TP8 + DCP8-era kit has the helpers this setup still needs (`memguard.sh`, `fabric_pairs.sh`, `preflight.sh`).
    It is in `zejunchen-zejun/ATOM`, branch `zejun/k3-455-1p1d-ualink`, directory `recipes/k3-455-pd/`.

## Container (all four trays)

```bash
IMAGE=<image built from ROCm/ATOM + ROCm/aiter k3-455-dev>
docker run -d --name k3pd -w /tmp \
  --device=/dev/kfd --device=/dev/dri --network=host --ipc=host --pid=host \
  --group-add "$(getent group video | cut -d: -f3)" --group-add "$(getent group render | cut -d: -f3)" \
  --cap-add=SYS_PTRACE --cap-add=SYS_ADMIN --security-opt seccomp=unconfined --shm-size=128g --ulimit memlock=-1:-1 \
  -v <dir with Kimi-K3/ and Kimi-K3-DSpark/>:/models:ro \
  -v <work>/cache:/root/.cache -v <work>/triton:/root/.triton \
  --entrypoint sleep "$IMAGE" infinity
docker exec k3pd mkdir -p /root/.cache/logs
```

- Pass the group IDs as numbers: passing `render` by name fails.
- The KV goes over UALink, so `/dev/infiniband` is not needed.
- `--pid=host` means a plain `pkill` inside the container can reach other users' processes. Stop by PID (see
  [Stop](#stop)).

If the image's ATOM is not this branch, check out this branch and put it first on `PYTHONPATH`:

```bash
docker exec k3pd git clone -q -b k3-455-dev https://github.com/ROCm/ATOM.git /root/.cache/atom-k3-455-dev
# then add  -e PYTHONPATH=/root/.cache/atom-k3-455-dev  to the docker exec that starts each server
```

## Server script (all four trays)

Save this as `/root/.cache/k3pd_server.sh` in each container. It starts one tray's half of an instance and is
mostly the image's `k3_455_tp8dcp8_server.sh` environment. The P and D parts come from the runs.

- **P parts:** PQ4 plus the 11-22 P settings.
- **D parts:** the #712 / #722 DP8 EP8 launch plus the PD settings.

```bash
#!/bin/bash
# k3pd_server.sh <P|D> <0|4>
#   P 0 | P 4: PP stages 0-3 | 4-7 of the prefill instance. P0 serves the API on :8000.
#   D 0 | D 4: DP ranks 0-3 | 4-7 of the decode instance. D0 serves the API on :8000.
# Environment knobs:
#   POINT   c1024 | c512 | c192 (default c512): picks the per-point settings below
#   D_SEQS  max-num-seqs per D DP rank (default 96 at c1024, else 64)
#   AITER_MEGA_S1F  1 (default; needs aiter #8189 + #8483) | 0 (as the runs, on an aiter without the fix)
#   IFNAME  data-plane NIC that reaches the other trays (enp4s0np0 on the G13 / E15 trays)
set -euo pipefail
ROLE=${1:?usage: k3pd_server.sh <P|D> <0|4>} FIRST=${2:?usage: k3pd_server.sh <P|D> <0|4>}
P0=<P0_IP> P1=<P1_IP> D0=<D0_IP>     # data-plane IPs: P's two trays and D's first tray
POINT=${POINT:-c512}
IF=${IFNAME:-enp4s0np0}

# --- architecture (B0 runs the image's code objects)
export PYTORCH_ROCM_ARCH=gfx1250 GPU_ARCHS=gfx1250 GPU_ARCH_LIST=gfx1250 MORI_GPU_ARCHS=gfx1250 GPU_TARGETS=gfx1250
export HSA_OVERRIDE_GFX_VERSION=12.5.0 ENABLE_CK=0
export MORI_AMD_SMI_PATH=/usr/local/lib/python3.12/dist-packages/_rocm_sdk_core/lib/libamd_smi.so.27
# --- attention: MLA decode on aiter's persistent PS1 FP8 kernel, FP8 prefill attention on the ASM kernel
export ATOM_USE_TRITON_MLA=0 AITER_MLA_DECODE_PS1_FLYDSL=1 ATOM_USE_TRITON_MLA_SHUFFLE_KV=0 ATOM_USE_UNIFIED_ATTN=1
export ATOM_USE_ASM_FP8_PREFILL_ATTN=1
# --- MoE, GEMM, quantization
export ATOM_MOE_GU_ITLV=1 ATOM_USE_TRITON_MOE_DECODE=0 AITER_USE_GROUPED_GEMM=1 AITER_FLYDSL_USE_HERD=1
export ATOM_FP8_BLOCKSCALE_WEIGHT_PRESHUFFLE=1 ATOM_FP8_BLOCKSCALE_USE_E8M0_SCALE=1 ATOM_FP4_MXSCALE_USE_32X4_SCALE_SHUFFLE=1
export AITER_ROPE_TRITON_BACKEND=1 AITER_USE_SYSTEM_TRITON=1
# --- communication: RCCL over UALink (without NCCL_MNNVL_ENABLE it falls back to TCP). Neither instance has a TP group
# across trays, so aiter's cross-tray custom all-reduce stays off.
export NCCL_MNNVL_ENABLE=1 NCCL_IB_DISABLE=1 NCCL_P2P_DISABLE=0 NCCL_P2P_LEVEL=SYS NCCL_CUMEM_ENABLE=1
export MORI_SOCKET_IFNAME=$IF NCCL_SOCKET_IFNAME=$IF GLOO_SOCKET_IFNAME=$IF
export AITER_CUSTOM_AR_USE_SYMM_MEM=0 ATOM_USE_CUSTOM_ALL_GATHER=0 ATOM_DP_LM_HEAD_MODE=default
export NCCL_GRAPH_REGISTER=0     # DSpark: capturing the q_len k+1 graphs fails in RCCL without it
# --- runtime and loading. Mmap and prefetch off: a full page cache set off KFD eviction storms on the 1 TB trays.
export HSA_USE_SVM=1 HSA_ENABLE_SDMA=1
export HSA_XNACK=$([ "$(cat /sys/module/amdgpu/parameters/noretry 2>/dev/null)" = 1 ] && echo 0 || echo 1)
export ATOM_LOADER_PREFETCH=false ATOM_DISABLE_MMAP=true
export TRITON_CACHE_DIR=/root/.triton/cache HSA_COREDUMP_PATTERN=/root/.cache/gpucore.%p
export ATOM_MONO_ENABLE=0 ATOM_PROFILER_TIMEOUT=3600 ROCPROFILER_REGISTER_FORCE_LOAD=0 AITER_LOG_LEVEL=WARNING
# --- both ends: DSpark k=4 with ReplaySSM (one KDA state slot per request), KDA state in fp16
export ATOM_ENABLE_REPLAYSSM=1 ATOM_GDN_SSM_DTYPE=fp16
SPEC=(--method dspark --draft-model /models/Kimi-K3-DSpark --num-speculative-tokens 4)
ONLINE_QUANT='{"global_quant_config": "per_block_fp8", "layer_quant_config": {"*self_attn.kv_b_proj": "ptpc_fp8"}, "exclude_layer": ["*self_attn.[qkv]_conv1d*", "*block_sparse_moe.experts*", "*block_sparse_moe.routed_expert_*"]}'
COMMON=(--model /models/Kimi-K3 --served-model-name moonshotai/Kimi-K3 --trust-remote-code --server-port 8000
  --kv_cache_dtype fp8 --block-size 128 --max-num-batched-tokens 16384
  --enable_prefix_caching --state-checkpoint-interval-tokens -1
  --online_quant_config "$ONLINE_QUANT" --torch-profiler-dir /root/.cache/torch_profile --disable-uvicorn-access-log)

case $ROLE in
P)
  # PQ4 (#635): 256 sequences, 16 micro-batches in flight. Utilization 0.88, the 32K prefix-attention chunk and
  # HSA_NO_SCRATCH_RECLAIM avoid the TP1 long-prefix OOM. One loader thread: a TP1 stage pins 160-176 GiB of host
  # memory while it loads its experts.
  export ATOM_PP_MAX_INFLIGHT=16 HSA_NO_SCRATCH_RECLAIM=1 ATOM_LOADER_NUM_THREADS=1
  # KDA state images in a per-stage store taking 0.75 of the KV budget; a session's spent anchor goes first
  export ATOM_PP_STATE_CHECKPOINT_KV_SHARE=0.75 ATOM_STATE_CHECKPOINT_REFRESH_FIRST_RESTORE=0 \
    ATOM_STATE_CHECKPOINT_DEMAND=0 ATOM_STATE_CHECKPOINT_DEMOTE_FIRST_RESTORE=1
  if [ "$POINT" = c1024 ]; then
    # P is the bottleneck only at c1024: a lighter stage 0 (it holds the dense FFN layer; the DSpark drafter sits
    # on stage 7) and the P kernel switches of PD36
    export VLLM_PP_LAYER_PARTITION=7,13,13,13,13,13,13,8
    export AITER_SITUV2_A4W4=1 AITER_FLYDSL_STAGE2_FP8=1 ATOM_USE_ASM_FP8_PREFILL_ATTN=0 ATOM_USE_FLYDSL_FP8_PREFILL_ATTN=1
  else
    export ATOM_PP_LAYER_PARTITION=kv_balanced
  fi
  KV='{"kv_role": "kv_producer", "kv_connector": "moriio", "moriio_backend": "fabric", "backend": "fabric", "moriio_mode": "write", "handshake_port": 6301}'
  cd /tmp   # from /app, the image's aiter checkout would shadow the aiter package
  # PP runs eager (PP requires --enforce-eager)
  exec python3 -m atom.entrypoints.openai_server "${COMMON[@]}" \
    -tp 1 -pp 8 --pipeline-parallel-size-local 4 --pipeline-parallel-rank "$FIRST" \
    --pipeline-parallel-node-ips "$P0,$P1" \
    --data-parallel-master-ip "$P0" --data-parallel-master-port 29500 --data-parallel-base-port 29700 \
    --enforce-eager --max-num-seqs 256 --gpu-memory-utilization 0.88 --attn-prefill-chunk-size 32768 \
    "${SPEC[@]}" --kv-transfer-config "$KV" ;;
D)
  SEQS=${D_SEQS:-$([ "$POINT" = c1024 ] && echo 96 || echo 64)}
  # MegaMoE over mori v2, FP4 dispatch and combine wire (the #712 DP8 EP8 launch)
  export ATOM_MORI_V2=1 ATOM_MORI_V2_FUSED=1 MEGA_DISPATCH=flydsl ATOM_MEGA_STAGE1_FUSED=1 ATOM_MEGA_STAGE2_FUSED=1 \
    MEGA_DISPATCH_WIRE=fp4 ATOM_MEGA_COMBINE_WIRE=fp4 ATOM_MEGA_DECODE_FAST_PATH=0 \
    MORI_EP_TOKOFF_EXT=0 ATOM_UNFUSED_GATHER_KV_B_PROJ=1 ATOM_MLA_QH128_ASM_PAGE_SIZE=0
  export AITER_MEGA_S1F=${AITER_MEGA_S1F:-1}
  # The default-on PrefillDelayer deadlocks DP8 EP8 on main-based ATOM (#722): ranks block in different collectives
  export ATOM_ENABLE_PREFILL_DELAYER=0
  export ATOM_DP_SESSION_AFFINITY=1 ATOM_LOADER_NUM_THREADS=2
  KV='{"kv_role": "kv_consumer", "kv_connector": "moriio", "moriio_backend": "fabric", "backend": "fabric", "moriio_mode": "write", "handshake_port": 6301}'
  cd /tmp
  # DP must not run eager: an idle rank's dummy step runs an M=0 GEMM. --enable-dp-attention is required,
  # or the MoE runs with EP=1.
  exec python3 -m atom.entrypoints.openai_server "${COMMON[@]}" \
    -tp 1 --data-parallel-size 8 --data-parallel-size-local 4 --data-parallel-rank "$FIRST" \
    --data-parallel-master-ip "$D0" --data-parallel-master-port 29500 --data-parallel-base-port 29700 \
    --enable-expert-parallel --enable-dp-attention \
    --cudagraph-mode FULL --max-num-seqs "$SEQS" --cudagraph-capture-sizes "[$(seq -s, 1 "$SEQS")]" \
    --gpu-memory-utilization 0.90 "${SPEC[@]}" --kv-transfer-config "$KV" ;;
*) echo "usage: k3pd_server.sh <P|D> <0|4>" >&2; exit 2 ;;
esac
```

## Launch

Start one command per tray. Each instance's second tray waits for its first, so the order does not matter. The usual
order is D, then P.

```bash
PT=c1024   # or c512 / c192
ssh <D0> "docker exec -d -e POINT=$PT k3pd bash -c 'bash /root/.cache/k3pd_server.sh D 0 > /root/.cache/logs/D.log 2>&1'"
ssh <D1> "docker exec -d -e POINT=$PT k3pd bash -c 'bash /root/.cache/k3pd_server.sh D 4 > /root/.cache/logs/D.log 2>&1'"
ssh <P0> "docker exec -d -e POINT=$PT k3pd bash -c 'bash /root/.cache/k3pd_server.sh P 0 > /root/.cache/logs/P.log 2>&1'"
ssh <P1> "docker exec -d -e POINT=$PT k3pd bash -c 'bash /root/.cache/k3pd_server.sh P 4 > /root/.cache/logs/P.log 2>&1'"
until curl -sf -m 5 http://<P0_IP>:8000/health && curl -sf -m 5 http://<D0_IP>:8000/health; do sleep 30; done
```

- **Startup time.** Expect about 7–12 minutes to `/health` 200: weight load, then warmup and KV allocation, then
  graph capture on D. It takes longer after a cache wipe.
- **Check the GPUs too.** `/health` can answer 200 before the model is loaded, so on every tray check that
  `rocm-smi --showmemuse` shows VRAM in use on all four GPUs.
- **What to look for in the P log:**
  - `MoRIIO backend FABRIC (UALink), transfer_mode=write`
  - `KV pool and per-request state come from a fabric MemPool (UALink)`
  - The `PP state checkpoint store:` line (images per stage, KV blocks with and without the store).

## Router

Run the router on any host that reaches both API servers, for example D's second tray:

```bash
docker run -d --name k3pd_router --network=host --ulimit nofile=1048576:1048576 \
  --entrypoint atomesh "$IMAGE" launch --host 0.0.0.0 --port 8100 --backend atom --pd-disaggregation \
  --prefill http://<P0_IP>:8000 --decode http://<D0_IP>:8000 \
  --prefill-policy dp_sticky --decode-policy round_robin \
  --request-timeout-secs 7200 --disable-circuit-breaker --log-level info --prometheus-port 29100
curl -s http://<ROUTER_IP>:8100/health
```

## Check

```bash
# smoke: P prefill -> UALink -> D decode
curl -s http://<ROUTER_IP>:8100/v1/completions -H 'Content-Type: application/json' \
  -d '{"model": "moonshotai/Kimi-K3", "prompt": "The capital of France is", "max_tokens": 16, "temperature": 0}'
# the image's accuracy gate: the last line must be GATE PASS
docker exec k3pd python3 /opt/k3/gate.py http://<ROUTER_IP>:8100
# GSM8K 5-shot through the router; the runs got 0.966-0.968
docker exec k3pd bash /opt/k3/gsm8k.sh http://<ROUTER_IP>:8100 /root/.cache/logs/gsm8k
```

- To debug a transfer, `ATOM_MORIIO_VERIFY=1` on both ends checks every received PAGE region and state slot against
  the producer's digests, stage by stage.
- Leave it off for performance runs.

## Benchmark (MLPerf Endpoints, agentic)

Set up the client once:

```bash
git clone https://github.com/mlcommons/endpoints.git <client>/endpoints && cd <client>/endpoints
git checkout -q --detach 4b385eb && UV_FROZEN=1 uv sync --frozen
# dataset: <client>/data/mlperf_agentic_inference_dataset.jsonl
```

Use one YAML per point. This one is for c1024. For the other points, set `target_concurrency` and
`num_trajectories_to_issue`: ceil(C / 613) × 613 trajectories, so 1226 at c1024 and 613 at c512 and c192.

```yaml
name: "k3pd-c1024"
version: "1.0"
type: "online"
submission_ref:
  model: moonshotai/Kimi-K3
  ruleset: mlperf-endpoints-v1.0-2026-10-C1-A
model_params:
  name: "moonshotai/Kimi-K3"
  tokenizer_name: "<model dir>/Kimi-K3"
  temperature: 1.0
  top_p: 1.0
  max_new_tokens: 8192
  streaming: "on"
datasets:
  - name: agentic_combined
    type: performance
    path: <client>/data/mlperf_agentic_inference_dataset.jsonl
    accuracy_config:
      eval_method: agentic_inference_inline
      num_repeats: 1
    agentic_inference:
      turn_timeout_s: 86400.0
      enable_salt: true
      inject_tool_delay: true
      num_trajectories_to_issue: 1226
      stop_issuing_on_first_user_complete: true
settings:
  runtime:
    max_issue_duration_ms: 10m
  load_pattern:
    type: agentic_inference
    target_concurrency: 1024
  client:
    warmup_connections: 0
    max_idle_time: 0.5
endpoint_config:
  endpoints: ["http://<ROUTER_IP>:8100"]
  api_type: openai
report_dir: <results>/k3pd-c1024
```

For the warmup YAML, start from the same file and change three things:
- drop `submission_ref`;
- add `scheduler_random_seed: 1` and `dataloader_random_seed: 2` under `runtime`;
- rename `name` and `report_dir`.

```bash
cd <client>/endpoints && ulimit -n 524288
UV_FROZEN=1 uv run inference-endpoint benchmark from-config --config <results>/k3pd-c1024-warmup.yaml --mode perf
UV_FROZEN=1 uv run inference-endpoint benchmark from-config --config <results>/k3pd-c1024.yaml --mode perf
cat <results>/k3pd-c1024/report.txt   # divide system tok/s by 16 for tok/s/GPU
```

## Per-point settings

`POINT` selects these in the script.

| | c1024 (PD36) | c512 (PD29) | c192 (PD29) |
|---|---|---|---|
| D max-num-seqs per DP rank | 96 | 64 | 64 |
| P layer split | `VLLM_PP_LAYER_PARTITION=7,13,13,13,13,13,13,8` | `ATOM_PP_LAYER_PARTITION=kv_balanced` | kv_balanced |
| P kernel switches (`AITER_SITUV2_A4W4`, `AITER_FLYDSL_STAGE2_FP8`, FlyDSL FP8 prefill attention) | on | off | off |
| Everything else | same | same | same |

**Why the split and the P switches are c1024-only.** P is the bottleneck only at c1024; at c512 and c192, P does not
queue, so neither changes anything there. With the switches alone, the c1024 plateau gained +2.3%; on top of the new
split they added nothing (PD33 without them: 732).

## Ports

| what | where |
|---|---|
| P API / D API / router | `<P0>:8000` / `<D0>:8000` / router host `:8100` |
| P sockets (multi-node PP, from master port 29500) | 29600–29633: engine sockets on `<P0>`, stage channels on each stage's tray; TCPStore 29700 on `<P0>` |
| D sockets (multi-node DP, from master port 29500) | 29600–29623 on `<D0>`; TCPStore 29700 on `<D0>` |
| MoRIIO side channels | P stage s: `6301 + s` on its own tray; D DP rank r: `6301 + r` on its own tray |

- P and D sit on different trays, so equal port numbers do not collide.
- If they ever share a tray, move one instance's ports and its `handshake_port` out of the other's range.

## Stop

```bash
# each tray: SIGTERM only this container's server processes (--pid=host), SIGKILL after 45 s if needed
P=$(docker top k3pd -eo pid,args | awk '/openai_server|ATOM::/ {print $1}' | paste -sd' ')
[ -z "$P" ] || docker exec k3pd kill -TERM $P
docker rm -f k3pd_router
# after a crash or kill -9, clear half-written caches before the next start
docker exec k3pd sh -c 'rm -rf /root/.triton/cache /root/.cache/atom /root/.cache/comgr /tmp/torchinductor_root /tmp/aiter_configs'
```

After a code change, clear at least `/root/.cache/atom` before the restart: a stale compile cache fails silently.

## Where this differs from the runs, and what to watch

- **`AITER_MEGA_S1F`.** The runs had S1F off: the 1119 aiter crashed with `hipErrorInvalidValue` (#722).
  - aiter `k3-455-dev` carries the fix. S1F on is validated for aggregated DP8 EP8 (06 + 05: GSM8K equal, TPOT
    −1% to −6%), not yet behind P.
  - If D fails at startup with `hipErrorInvalidValue` or `compact plan needs ... LDS`, set `AITER_MEGA_S1F=0`.
- **`ATOM_ENABLE_PREFILL_DELAYER=0` on D** is new. On main-based ATOM, the default-on delayer deadlocks aggregated
  DP8 EP8. The runs' tree may not have had it.
- **`ATOM_MEGA_MASK_PAD_ROWS=1`** was set on D in the runs. Nothing on this branch reads it, so the script leaves it
  out.
- **The PP checkpoint store** is on by default here under PP, with a share of 0.5. The script sets 0.75, as the runs
  did. ROCm/ATOM#3694's head makes the store opt-in, so keep setting the share explicitly.
- **D at 96 sequences per rank** runs with fp16 KDA state (361 MB per sequence instead of 578 MB). If 96 does not fit,
  try `--gpu-memory-utilization 0.92` (queued as PD27 with 80) or fewer sequences.
- **The torch profiler on P** crashed the process after a few minutes (glibc heap corruption or SIGSEGV). Profile D
  only.
- **A cross-tray 4-rank group** (TP, DP or EP of 4 split over two trays) needs `ATOM_ENABLE_GLUON_FUSED_AR=0`.
  aiter otherwise builds a Gluon all-reduce whose IPC cannot cross trays. DP8 EP8 and PP8 × TP1 do not hit this.
- **The warmup and the measured run replay the same trajectories.** Use a different seed for the warmup (above).
- **32 GPUs** (2P1D, D = DP16 EP16) is not covered here.
