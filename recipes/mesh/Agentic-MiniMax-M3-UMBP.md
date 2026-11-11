# MiniMax-M3 with Mori UMBP KV Offload (`umbp_store`)

This recipe runs MiniMax-M3 on ATOM with a Mori UMBP standalone server as the
CPU KV tier, through the `umbp_store` offload connector. It covers two
deployments:

- **Aggregated TP4 + UMBP**: the same server as
  [MiniMax-M3-Agentic-InferenceX.md](../MiniMax-M3-Agentic-InferenceX.md), with
  `lmcache_offload` replaced by `umbp_store`.
- **PD 1P1D + UMBP on the prefill node**: the same topology as
  [Agentic-MiniMax-M3.md](Agentic-MiniMax-M3.md), with the prefill node's
  `multi` connector holding `mooncake` (P/D transfer) and `umbp_store`
  (offload).

The benchmark client and workload are the SemiAnalysis AgentX replay described
in [MiniMax-M3-Agentic-InferenceX.md](../MiniMax-M3-Agentic-InferenceX.md).

## How it fits

```text
                 ATOM server (TP4)
  scheduler process                     ModelRunner x4 (one per TP rank)
  UMBPStoreOffloadScheduler             UMBPStoreOffloadConnector
   prompt -> chunk hash chain -> keys    m3 PAGE codec pack / unpack
   lookup: batch_exists                  (K / V / scale / NSA index cache)
        |                                HBM staging pool, registered with UMBP
        |                                save / load threads
        |  gRPC over a unix socket              |  batch_put_from_ptr / batch_get_into_ptr
        v                                       v
  umbp_standalone_server (one per node): shared DRAM pool; it reads and
  writes the workers' staging buffers directly through HIP IPC
```

- The server owns the DRAM pool. ATOM does not start it; start it before the
  ATOM server and stop it after.
- Objects are immutable: a put of a key that already exists succeeds and
  keeps the old bytes. Keys carry a content hash of the prompt prefix, so
  this is the intended behaviour.
- MiniMax-M3 always uses the `m3` PAGE layout, which carries the NSA index
  cache. The connector reads `KVTransferTensors.block_regions`; nothing in the
  model changes.

## Prerequisites

- An ATOM container whose Mori build includes UMBP. Check both pieces:

  ```bash
  python3 -c "import mori.umbp; print('umbp ok')"
  python3 -c 'import mori.umbp,pathlib;print(pathlib.Path(mori.umbp.__file__).resolve().parents[1]/"umbp_standalone_server")'
  ```

- A Mori build that includes ROCm/mori#791 ("a lookup renews the keys it
  finds"): `amd_mori_nightly` 1.2.6.dev20261109 or later, or Mori main.
  Without it, a lookup does not refresh the LRU, so a prefix that keeps
  hitting in HBM (the shared system-prompt head chunk above all) ages out of
  UMBP first, and every later lookup of that conversation misses from chunk 0.

- The model and EAGLE3 draft as in
  [MiniMax-M3-Agentic-InferenceX.md](../MiniMax-M3-Agentic-InferenceX.md).
- For PD: RDMA between the prefill and decode GPUs (see
  [Agentic-MiniMax-M3.md](Agentic-MiniMax-M3.md) and the Mooncake notes in
  Troubleshooting below).

## Step 1: Start the UMBP standalone server

Run it in the same container as the ATOM server it serves. Size the pool for
the whole node: the ranks share it, unlike a per-rank LMCache CPU tier.

```bash
export UMBP_ADDR=unix:///tmp/umbp/server.sock
UMBP_BIN=$(python3 -c 'import mori.umbp,pathlib;print(pathlib.Path(mori.umbp.__file__).resolve().parents[1]/"umbp_standalone_server")')
mkdir -p /tmp/umbp && rm -f /tmp/umbp/server.sock*

# Use the same visible GPUs as the ATOM server (see "HIP IPC device mismatch").
ROCR_VISIBLE_DEVICES=0,1,4,5 \
UMBP_DRAM_CAPACITY=$((1024 * (1 << 30))) \
UMBP_DRAM_NUMA_NODE=0,1 \
UMBP_DRAM_USE_HUGEPAGES=0 \
UMBP_DRAM_PREFAULT_THREADS=16 \
UMBP_SSD_ENABLED=0 \
UMBP_STANDALONE_METRICS_PORT=9877 \
MORI_UMBP_LOG_LEVEL=INFO \
  nohup "$UMBP_BIN" "$UMBP_ADDR" > /workspace/logs/umbp_server.log 2>&1 &
```

The `UMBP_*` variables belong to the Mori server. ATOM adds none.

A socket file is not proof the server is ready; prefaulting a large pool takes
minutes. Wait until a real client connects:

```bash
until python3 - <<PY 2>/dev/null
import mori.umbp as u
c = u.UMBPConfig(); p = u.UMBPStandaloneProcessConfig()
p.address = "$UMBP_ADDR"; p.auto_start = False; p.startup_timeout_ms = 10000
c.standalone_process = p
u.UMBPClient(c)
PY
do sleep 5; done; echo "umbp ready"
```

Restart the server together with the ATOM server between benchmark points so
every point starts from an empty pool.

## Step 2a: Aggregated TP4 + UMBP

Take the server command from
[MiniMax-M3-Agentic-InferenceX.md](../MiniMax-M3-Agentic-InferenceX.md)
(the offload points) and change only the KV tier:

- drop the `LMCACHE_*` variables;
- keep `PYTHONHASHSEED=0`, `ATOM_PREFIX_CACHE_POLICY=slru`,
  `ATOM_PREFIX_CACHE_PROTECTED_RATIO=0.5`, `ATOM_M3_INDEXER_CP=1` and
  `ROCR_VISIBLE_DEVICES=0,1,4,5`;
- replace the `--kv-transfer-config`:

```bash
  --kv-transfer-config '{"kv_connector":"umbp_store","kv_role":"offload",
    "kv_connector_extra_config":{"umbp_store.address":"unix:///tmp/umbp/server.sock"}}'
```

## Step 2b: PD 1P1D with UMBP on the prefill node

Follow [Agentic-MiniMax-M3.md](Agentic-MiniMax-M3.md), and on the prefill
node replace `lmcache_offload` inside `multi` with `umbp_store`:

```bash
# Prefill (GPUs 0-3); start the UMBP server first with ROCR_VISIBLE_DEVICES=0,1,2,3
  --kv-transfer-config '{"kv_connector":"multi","connectors":[
    {"kv_connector":"mooncake","kv_role":"kv_producer","handshake_port":6301,"protocol":"rdma"},
    {"kv_connector":"umbp_store","kv_role":"offload",
     "kv_connector_extra_config":{"umbp_store.address":"unix:///tmp/umbp/server.sock"}}]}'
```

The decode node stays a plain Mooncake `kv_consumer` and needs no UMBP server.
The router is unchanged (`atomesh launch --pd-disaggregation --backend atom`).

## `umbp_store` settings

All settings live in `kv_connector_extra_config`; an unknown `umbp_store.*`
key is an error.

| Key | Default | Meaning |
|-----|:-------:|---------|
| `umbp_store.address` | required | The standalone server's address, e.g. `unix:///tmp/umbp/server.sock`. |
| `umbp_store.startup_timeout_ms` | 10000 | How long a client waits for the server at connect. |
| `umbp_store.chunk_tokens` | 256 | Tokens per UMBP object; a multiple of the KV block size. |
| `umbp_store.load_pool_mib` | 1024 | HBM load staging per worker, MiB. |
| `umbp_store.save_pool_mib` | 256 | HBM save staging per worker, MiB. |
| `umbp_store.save_abandon_timeout_s` | 300 | Seconds before the engine reclaims an unreported save's source. |

## Validation Checklist

- The ATOM log shows the `umbp_store` connector on every TP rank and a passing
  start-up probe.
- The UMBP server log shows client registrations and no errors; with the
  metrics port on, put/get counters grow during the run.
- Accuracy: run GSM8K **without** `--spec-decode-acceptance-length` (with it,
  every output is the synthetic token) and compare against the same server
  with `lmcache_offload`.
- KV read-back: the load path only runs once a prefix has left HBM. To check
  it, send a few long greedy prompts with facts spread through them, push them
  out of the HBM prefix cache with unrelated long requests (a lower
  `--gpu-memory-utilization` makes this quicker), resend the prompts, and
  confirm that the log shows a UMBP load for them and that the answers about
  every fact are still right.

## Troubleshooting

### HIP IPC device mismatch

The server opens the workers' staging buffers through HIP IPC. Start it with
the same `ROCR_VISIBLE_DEVICES` (and order) as the ATOM server it serves, so
the device ordinals agree and it does not touch other GPUs.

### Server not ready / connection refused

`umbp_store` refuses to start without a reachable server. Check the address,
wait for the readiness loop above, and look at the server log for prefault
progress.

### Behaviour that differs from Mooncake Store

- There is no per-key remove: the start-up probe object stays until evicted.
- A failed put that raised (rather than returned False) stops loads in that
  worker and is recorded in that process only; other processes and engines
  are not told.
- `deregister_memory` in standalone mode releases every region the client
  registered, so the connector deregisters only when it closes.
- Calls are synchronous; the connector runs them on its save and load threads.

### Mooncake silently using TCP (PD only)

If the container's `libionic` does not match the host ionic driver, Mooncake
finds no HCA and falls back to TCP while ATOM still logs `protocol=rdma`.
Bind-mount the host library over the container's copies, then confirm traffic
on the RDMA counters rather than in the log:

```bash
  -v /usr/lib/x86_64-linux-gnu/libionic.so.1.<ver>:/usr/lib/x86_64-linux-gnu/libionic.so.1:ro \
  -v /usr/lib/x86_64-linux-gnu/libionic.so.1.<ver>:/usr/lib/x86_64-linux-gnu/libibverbs/libionic-rdmav34.so:ro \
```

Keep `NCCL_IB_DISABLE=1 RCCL_IB_DISABLE=1` for single-node TP; they only
affect RCCL. On nodes where only same-NIC loopback works, use
`ATOM_MOONCAKE_MATCHED_RAILS=auto` with `MC_ENABLE_DEST_DEVICE_AFFINITY=1`.

### Releasing resources

```bash
pkill -f 'atom.entrypoints.openai_server' || true
pkill -f 'atomesh launch' || true
pkill -f umbp_standalone_server || true
```
