# Native KV connector (`kv_connector="native"`)

A fully in-tree prefill/decode (P/D) KV-cache connector built only on the HIP
Virtual Memory Management (VMM) API — **no third-party transport** (no MoRI, no
Mooncake). It covers **both**:

- **scale-up (single node / XGMI):** VMM buffers shared as **POSIX fd** handles
  over a UNIX socket; KV moved GPU→GPU with `hipMemcpy` peer over XGMI.
- **scale-out (cross node / IFOE, gfx1250):** the same VMM buffers shared as
  node-independent **fabric** handles over TCP; KV moved over the IFOE fabric.

## When to use it

| Scenario | Connector |
|---|---|
| Single node, P/D split across GPUs on the same box (XGMI) | **`native`** |
| Cross node, P/D over the IFOE fabric (gfx1250) | **`native`** |
| Cross node over RDMA NICs | `moriio` |

Requires GPUs with VMM support (queried per device); the fabric/scale-out path
additionally requires fabric-handle support (gfx1250) — on parts without it
(e.g. gfx950) only the same-node fd/XGMI path is used.

## How to select it

Pass `"kv_connector": "native"` in `--kv-transfer-config`. The transport is
**auto-selected by topology**: a producer on the same node → fd + UNIX socket +
XGMI; a producer on a different node (and fabric supported) → fabric handle +
TCP + IFOE. No `protocol` field; natural placement (`device == rank`), no
visibility reorder.

## Launch (4 prefill GPUs + 4 decode GPUs on one node)

```bash
# 1) proxy
python -m atom.kv_transfer.disaggregation.proxy --port 10001

# 2) prefill engine (producer) on GPUs 0-3
HIP_VISIBLE_DEVICES=0,1,2,3 python -m atom.entrypoints.openai_server \
  --model deepseek-ai/DeepSeek-V4-Pro --kv_cache_dtype bf16 -tp 4 \
  --gpu-memory-utilization 0.85 --max-num-seqs 128 \
  --host 0.0.0.0 --server-port 8003 \
  --kv-transfer-config '{"kv_connector":"native","kv_role":"kv_producer","proxy_ip":"127.0.0.1","proxy_ping_port":36367,"http_port":8003,"handshake_port":6501}'

# 3) decode engine (consumer) on GPUs 4-7
HIP_VISIBLE_DEVICES=4,5,6,7 python -m atom.entrypoints.openai_server \
  --model deepseek-ai/DeepSeek-V4-Pro --kv_cache_dtype bf16 -tp 4 \
  --gpu-memory-utilization 0.85 --max-num-seqs 128 \
  --host 0.0.0.0 --server-port 8004 \
  --kv-transfer-config '{"kv_connector":"native","kv_role":"kv_consumer","proxy_ip":"127.0.0.1","proxy_ping_port":36367,"http_port":8004,"handshake_port":6501}'

# 4) send requests to the proxy
curl -s http://127.0.0.1:10001/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"deepseek-ai/DeepSeek-V4-Pro","prompt":"The capital of France is","max_tokens":16}'
```

`kv_transfer_params.do_remote_prefill` in the response should be `true`.

## `kv-transfer-config` fields

| key | meaning |
|---|---|
| `kv_connector` | `"native"` |
| `kv_role` | `"kv_producer"` (prefill) or `"kv_consumer"` (decode) |
| `proxy_ip`, `proxy_ping_port` | proxy address for registration |
| `http_port` | this engine's OpenAI server port |
| `handshake_port` | base port for the UNIX side channel (per-rank offset added) |

## How it works

- Each worker allocates a VMM **staging** buffer and (consumer) exports its
  POSIX fd over a UNIX side channel (`SCM_RIGHTS`).
- The consumer sends its destination block ids + staging fd to the producer;
  the producer imports the staging (granting its own device access), gathers
  the request's KV blocks straight into it over XGMI, and replies `WRITE_DONE`.
- The consumer scatters from its staging into its local KV pool.
- One fd import per (producer, consumer) pair; subsequent transfers are direct
  device-to-device copies — no RDMA, no IPC-handle churn, no host staging.

## Status

v1 wires the VMM transport primitive (validated cross-process by
`tests/test_native_vmm_transfer.py`) into the KVConnector interface. Requests
within a scheduler step are transferred sequentially; a concurrent staging pool
and DeepSeek-V4 slot/index-region fast paths are follow-ups. See ROCm/ATOM#1483
for end-to-end 4P4D validation.
