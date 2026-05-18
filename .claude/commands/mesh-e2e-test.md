# Mesh PD End-to-End Test

Drives a 2-host PD-disaggregated test of `atom-mesh` against gsm8k (accuracy)
and a `bench_serving` perf sweep. Use after any change in `atom/mesh/src/`
(routers, placement core, BackendAdapter, policies) before merging.

## Test matrix

| Variant | Backend | Wire | Scripts dir |
|---|---|---|---|
| sglang gRPC | sglang | grpc:// | `0509_ds_fp4_1p_tp8_1d_tp8_5833/scripts/` |
| sglang HTTP | sglang | http:// | derive `http_*.sh` from above |
| vllm HTTP   | vllm   | http:// | `0509_ds_fp4_vllm_1p_tp8_1d_tp8_5836/scripts/` |

vLLM has no gRPC mode. sglang HTTP variant is built by dropping `--grpc-mode`
from `prefill.sh`/`decode.sh` and rewriting `router.sh` to use `http://` URLs.

Scripts live under `/it-share/yajizhan/slurm_logs/<run_id>/scripts/` and are
mounted into both containers via `-v /it-share:/it-share`. They write logs to
`/workspace/logs/{prefill,decode,router}.log`.

## Topology

```
host A (mia1-p02-g42, 10.24.112.168)        host B (mia1-p02-g44, 10.24.112.170)
+-------------------------------------+      +-------------------------------------+
| docker: atom_sglang_mesh OR         |      | docker: atom_sglang_mesh OR         |
|         atom_vllm_mesh              |      |         atom_vllm_mesh              |
|                                     |      |                                     |
| prefill :8010 (TP8)  + bootstrap    |<====>| decode  :8020 (TP8)                 |
| router  :8000  -> http/grpc fanout  |      |                                     |
+-------------------------------------+      +-------------------------------------+
       ^                                              mooncake RDMA over rdma0..7
       |
   gsm8k.sh / benchmark.sh -> http://127.0.0.1:8000
```

Bootstrap port `8998` is sglang-specific; vLLM passes the same port via
`VLLM_MOONCAKE_BOOTSTRAP_PORT`. Mooncake config JSON lives under
`/workspace/mooncake_prefill.json` (sglang prefill writes it inline).

## Pre-flight (do these first, fail fast)

1. **Containers up on both hosts.** If `docker ps` shows nothing matching
   `atom_(sglang|vllm)_mesh`, start it:
   - sglang: `bash atom/mesh/scripts/docker_start.sh`
   - vLLM:   `bash atom/mesh/scripts/docker_start_vllm.sh`
   - Run on the **other** host too: `ssh <other-host> 'bash -lc "<same cmd>"'`

2. **Both hosts reachable from current host.** The harness can only `bash` on
   the local host; the remote one is driven via `ssh <hostname> '...'`. Verify:
   `ssh -o ConnectTimeout=5 <other-host> hostname`.

3. **GPUs free on both hosts.** Each TP8 server needs all 8 GPUs at ~85% mem.
   `ssh <host> 'rocm-smi --showpids'` — if a vLLM benchmark / other tenant is
   holding VRAM, ASK the user before stopping it. We hit this once: a stale
   `atom_vllm_oot_benchmark_*` container occupied all 8 GPUs and OOM'd decode.

4. **No stale atom-mesh / sglang / vllm processes** on either host. Clean with
   `docker exec <c> pkill -9 -f <pattern>` (run separately per pattern; do
   not chain them in one ssh+docker exec — the exec session dies if a pkill
   target overlaps it). Zombies (`<defunct>`) are OK, they don't hold GPU.

## Standard run order

Build (host A only, ~2 min cached, ~10 min cold) → install → start P+D in
parallel (~10 min model load, dominated by safetensors) → start router (~3 s)
→ smoke test → gsm8k (~1 min) → benchmark (~10 min sweep). Total ~25 min cold,
~15 min if model weights are in OS page cache.

### 1. Rebuild atom-mesh on host A

```bash
docker exec -w /it-share/yajizhan/code/ATOM/atom/mesh atom_sglang_mesh \
    cargo build --release --bin atom-mesh
docker exec atom_sglang_mesh \
    cp /it-share/yajizhan/code/ATOM/atom/mesh/target/release/atom-mesh \
       /usr/local/bin/atom-mesh
```

For the vLLM variant, swap container to `atom_vllm_mesh`. Both containers ship
with Rust 1.95 toolchain; the Cargo target dir is shared via `/it-share`, so
host A only needs to build once per branch.

**Verify the running router is the rebuild** (we got asked this — proves it):

```bash
md5sum /it-share/yajizhan/code/ATOM/atom/mesh/target/release/atom-mesh
docker exec <container> md5sum /usr/local/bin/atom-mesh
docker exec <container> bash -c \
    'pid=$(pgrep -f atom-mesh | head -1); md5sum /proc/$pid/exe'
```

All three MD5s must match. mtime should match `cargo build` completion.

### 2. Start prefill and decode in parallel

`docker exec -d` is the right primitive — `nohup ... &` inside a non-`-d`
exec session dies when the exec session ends. Run in parallel; both take
many minutes so serializing wastes wall time:

```bash
docker exec -d <container> bash -c \
    'cd <scripts_dir> && exec bash prefill.sh > /workspace/logs/prefill.console.log 2>&1'

ssh <host_B> "docker exec -d <container> bash -c \
    'cd <scripts_dir> && exec bash decode.sh > /workspace/logs/decode.console.log 2>&1'"
```

Watch for ready/fail with Monitor (one per host):

```bash
# host A
docker exec <container> bash -c 'tail -n 0 -F /workspace/logs/prefill.log' \
    | grep -E --line-buffered \
        "server is fired up|HIP out of memory|RuntimeError|Killed|Scheduler hit an exception"

# host B (same pattern wrapped in ssh ...)
```

`tail -n 0 -F` (not `tail -f`) is required — otherwise the monitor replays
old log lines from a prior run that wrote to the same path before
`tee` truncated it.

### 3. Start router on host A

```bash
docker exec -d <container> bash -c \
    'cd <scripts_dir> && exec bash router.sh > /workspace/logs/router.console.log 2>&1'
```

Confirm the right wire format and worker set in the log:

```
Router ready | workers: ["http://...:8010", "http://...:8020"]   # HTTP variant
Created single router with ID: http-pd                            # http-pd or grpc-pd
```

### 4. Smoke test before benchmarks

A single deterministic completion catches wire/dispatch breakage in seconds —
do not skip this:

```bash
docker exec <container> bash -c '
    curl -sS -X POST http://127.0.0.1:8000/v1/completions \
        -H "Content-Type: application/json" \
        -d "{\"model\":\"<MODEL_PATH>\",\"prompt\":\"The capital of France is\",\"max_tokens\":16,\"temperature\":0}"'
```

Expected: response body contains "Paris" and `finish_reason:"length"`.

### 5. gsm8k accuracy (1319 samples, ~1 min)

```bash
docker exec <container> bash -c \
    'cd <scripts_dir> && bash gsm8k.sh 2>&1' \
    | tee /tmp/gsm8k.out | grep -E "exact_match|gsm8k\|"
```

Reference (DeepSeek-R1-0528-MXFP4-MTP-MoEFP4, 3-shot):
- flexible-extract ≈ 0.84–0.86 (±0.01 stderr)
- strict-match    ≈ 0.83–0.85

Drops > 3 stderr (~3 pp) below this band are a regression — investigate before
shipping. Greedy decoding is batch-order sensitive, so 1–2 pp run-to-run drift
is expected.

### 6. Perf sweep (ISL=8192 OSL=1024, CONC ∈ {1,2,4,8,16}, ~10 min)

```bash
docker exec <container> bash -c \
    'rm -rf /workspace/benchmark_results && cd <scripts_dir> && bash benchmark.sh 2>&1'
```

Watch with Monitor for per-config result lines:

```bash
grep -E --line-buffered \
    "ISL=|Output token throughput|Mean TTFT|Mean ITL|Mean TPOT|Successful requests|Benchmark duration|ERROR|Traceback|HTTP/|failed"
```

Reference table on 1P+1D TP8 + DeepSeek R1 MXFP4 (validated 2026-05-14 on
v0.2 placement-core refactor, both gRPC and HTTP wire):

| CONC | TTFT mean (ms) | TPOT (ms) | tput (tok/s) |
|---:|---:|---:|---:|
| 1  | ~225 | ~7.2  | ~135 |
| 2  | ~240 | ~7.5  | ~256 |
| 4  | ~281 | ~8.2  | ~457 |
| 8  | ~370 | ~9.3  | ~805 |
| 16 | ~450 | ~10.9 | ~1358 |

HTTP TTFT runs ~10–15 ms above gRPC at CONC≥16 (HTTP body framing); decode
TPOT is wire-independent (router not on hot path post-prefill).

### 7. Final sanity check

`docker exec <container> bash -c 'grep -cE "ERROR|panic|Failed|failed" /workspace/logs/router.log'`

Must be 0. If not, dump the offending lines and bisect against the previous
known-good binary (kept in container at `/usr/local/bin/atom-mesh.bak` — back
this up before step 1 if you want bisect headroom).

## Variant: sglang HTTP

Drop `--grpc-mode` from `prefill.sh`/`decode.sh`; rewrite `router.sh`
`--prefill grpc://...` → `--prefill http://...`. We keep both variants so
HTTP and gRPC code paths in `routers/http_pd_router.rs` and
`routers/grpc/pd_router.rs` both get exercised. The check-in pattern is to
copy the three scripts to `http_prefill.sh` / `http_decode.sh` /
`http_router.sh` in the same scripts dir (already done for run 5833).

## Variant: vLLM HTTP

Swap container `atom_sglang_mesh` → `atom_vllm_mesh` everywhere; scripts dir
is `0509_ds_fp4_vllm_1p_tp8_1d_tp8_5836/scripts/`. Router uses
`--backend vllm`. KV transfer is mooncake but configured per-side via
`--kv-transfer-config '{"kv_connector": "MooncakeConnector", "kv_role":
"kv_producer|kv_consumer"}'` in the vllm serve commands. Bootstrap port
plumbed via `VLLM_MOONCAKE_BOOTSTRAP_PORT=8998` env. vLLM scripts call
`rm -rf /root/.cache` at start — first run after container restart is slower.

## Failure modes we hit (and the fix)

- **GPU OOM at decode start, all VRAM allocated by another process**: a stale
  benchmark container was holding 89% on every GPU. Stop it, recheck
  `rocm-smi --showpids` shows only `gpuagent`. The 50% VRAM number that
  rocm-smi reports right after a process dies is a stale readout, not a leak.
- **`docker exec ... pkill -9 -f atom-mesh` exits 137**: the exec session
  itself can be the SIGKILL target if the matched process tree includes the
  shell. Use `docker exec <c> pkill -9 -f <pattern>` directly (no nested
  `bash -c`), one pattern per call.
- **`nohup bash decode.sh &` from `docker exec` doesn't actually launch**:
  the exec session terminates on return and takes the backgrounded process
  with it. Use `docker exec -d <c> bash -c '... exec bash decode.sh > log
  2>&1'` instead.
- **Monitor replays old log content**: `tail -F` from start re-emits the
  whole file, including the previous run's crash trace. Use `tail -n 0 -F`.
- **Router up but workers not ready**: `Workers initialized: 2 total, 2
  healthy` in the log means registry-level handshake passed. It does NOT
  mean the underlying sglang/vllm servers can serve a request. The smoke
  test catches this gap.
