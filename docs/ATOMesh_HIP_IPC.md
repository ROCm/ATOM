# Single-node ATOMesh transfers over HIP IPC/xGMI

ATOMesh's `single_node` P/D layout selects `protocol: hip` for Mooncake when
launching prefill and decode. This also updates Mooncake inside a `multi`
connector while preserving LMCache/offload settings. Other layouts retain
their configured transport (RDMA by default).

Transport selection happens after runtime environment expansion, so the
prefill/decode handshake ports include the execution phase's service-port
offset. `${HANDSHAKE_PORT}` in a role's connector configuration resolves to
that worker's port, including decode's offset from prefill.

For a manually launched server, add `"protocol":"hip"` to its Mooncake KV
transfer configuration on **both** sides. The two processes must run on the
same host. The connector excludes RDMA HCAs using a nonempty device filter;
an empty filter would let Mooncake auto-select HCAs. HIP mode also clears
inherited `MC_FORCE_TCP` and `MC_DISABLE_HIP` overrides.

## Runtime requirements

- A Mooncake build with HIP support that installs the HIP transport for
  intra-node transfers. Initialization must report
  `HIP transport installed for intra-node GPU P2P` (or equivalent evidence
  for the installed version).
- ROCm peer access between the selected GPUs. Check the node's topology to
  distinguish xGMI from PCIe links.
- GPU devices exposed to the container, with IPC access between the workers.
  The ATOMesh container launcher uses host IPC. Each worker may have its own
  `HIP_VISIBLE_DEVICES` subset.

Setting `protocol: hip` alone is not proof of the data path: older Mooncake
builds can accept it and silently use TCP. A stale cached `rocm/atom-dev:latest`
was observed doing this during validation. Rebuild the ATOM image with the
HIP-capable Mooncake dependency or select a verified image, then run the
smoke check below before benchmarking.

## GPU smoke check

Run inside the ATOM runtime container on an idle node, with two distinct GPUs:

```bash
python3 .github/scripts/atomesh/hip_smoke.py --source-gpu 0 --target-gpu 4
```

The check uses the connector's transport-selection functions and two separate
processes, each seeing one GPU. It transfers a 64 MiB buffer, verifies every
byte for two different patterns, and reports bandwidth over 100 writes.
It fails if loopback traffic is large enough to indicate TCP payload fallback.
The check uses loopback for control messages; unrelated heavy loopback traffic
on the node can also make it fail. Success prints `HIP_TRANSFER_PASS` with
the measured bytes, duration, bandwidth and loopback counter delta.

For a full P/D run, use the existing ATOMesh benchmark with
`pd_worker_layout: single_node` and verify both workers initialize Mooncake
with `protocol=hip`, register their KV buffers, and complete benchmark requests.
Model throughput also depends on compute, topology and concurrency; the smoke
check's transfer bandwidth is not an inference-throughput speedup claim.
