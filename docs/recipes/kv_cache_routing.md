# KV cache routing rollout

The native PAGE namespace is version 4. It includes the complete PP layer
partition, TP size, DCP size and interleave, effective index dtype, HF cache
geometry and speculative configuration. Existing version 3 CPU/disk objects
remain isolated and are cold misses after upgrading.

Both frontends expose `/server_info`; the Python frontend also exposes
`/kv_transfer_info`. These include PP/TP/DCP geometry and the global hash span.
Placement is optional in `kv_transfer_config.routing_topology`:

```json
{
  "execution_id": "prefill-a/dp-0",
  "ranks": [
    {"pp": 0, "tp": 0, "host_id": "host-a", "device_id": 0},
    {"pp": 1, "tp": 0, "host_id": "host-a", "device_id": 1},
    {"pp": 2, "tp": 0, "host_id": "host-b", "device_id": 0},
    {"pp": 3, "tp": 0, "host_id": "host-b", "device_id": 1}
  ],
  "cache_bindings": [
    {"storage_domain_id": "native-prefill-a", "rank_set": [[0, 0], [1, 0], [2, 0], [3, 0]]}
  ]
}
```

Every PP x TP rank must appear exactly once when placement is supplied. DCP
reuses TP GPUs: TP4/DCP2 has token shards 0,1,0,1. Host placement is independent
of byte compatibility. The connector handshake still determines actual transfer
support and reachability; a placement manifest does not enable a new layout
conversion. P and D may have different physical namespaces.

For GLM-5.2, keep FP8 index cache and DCP interleave 1. Preserve the exact PP
partition used for each deployment (`20,20,20,18` and `18,20,20,20` differ).
Correctness runs must disable forced MTP acceptance.
