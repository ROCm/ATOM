# MiniMax-M3 PD 1P1D with a Mori UMBP tier on the prefill node

> **Status: not supported yet.** `umbp_store` is currently supported as an
> offload tier for aggregated serving only (see
> [MiniMax-M3-UMBP.md](../MiniMax-M3-UMBP.md)). This page describes the planned
> PD setup; it has not been validated and is not expected to work yet.

PD disaggregation for MiniMax-M3 with the prefill node's KV offload tier served by a
Mori UMBP standalone server (`umbp_store`). It is the topology of
[Agentic-MiniMax-M3.md](Agentic-MiniMax-M3.md), with `lmcache_offload` in the prefill
node's `multi` connector replaced by `umbp_store`.

- How to install, start and check UMBP: [UMBP-KV-Offload.md](../UMBP-KV-Offload.md)
- The same model aggregated on TP4 with UMBP: [MiniMax-M3-UMBP.md](../MiniMax-M3-UMBP.md)

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

- Mori with UMBP, including ROCm/mori#791: see
  [UMBP-KV-Offload.md § Requirements](../UMBP-KV-Offload.md#1-requirements).
- The model and EAGLE3 draft as in
  [MiniMax-M3-Agentic-InferenceX.md](../MiniMax-M3-Agentic-InferenceX.md).
- For PD: RDMA between the prefill and decode GPUs (see
  [Agentic-MiniMax-M3.md](Agentic-MiniMax-M3.md) and the Mooncake notes in
  Troubleshooting below).

## Step 1: Start the UMBP server on the prefill node

Follow [UMBP-KV-Offload.md § Start the server](../UMBP-KV-Offload.md#2-start-the-server),
with `ROCR_VISIBLE_DEVICES` set to the prefill server's GPUs (`0,1,2,3` here) and the
pool sized for the prefill node. Wait for the readiness check to pass. The decode node
needs no UMBP server.

## Step 2: PD 1P1D with UMBP on the prefill node

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

See the settings table in [UMBP-KV-Offload.md § Point ATOM at it](../UMBP-KV-Offload.md#3-point-atom-at-it).

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
