# Experimental V4.1 TileRT producer

Based on #3105 (`d46eb22`), including event-driven idle KV wakeup.
This producer packs ATOM V4.1 PAGE/STATE caches into TileRT 0.1.7 wire
records, sends them over Mooncake, and retains source ownership until
the TP-wide completion reports arrive.

Use `--kv-transfer-config` with:

```json
{"kv_role":"kv_producer","kv_connector":"tilert","tilert_max_seq_len":1048576,"tilert_host":"DECODE_IP","tilert_ctrl_port":19641}
```

The tested producer uses TP4 with DP attention, eager execution, BF16 KV,
FP8 index cache, block size128, DSpark5 and the V4.1 checkpoint. Install the
TileRT0.1.7 wheel with `--no-deps` in the ATOM image to preserve torch/aiter.
The decoder uses TileRT's native TP8 image and `dsv41_decode_server`.
Keep `MC_NUM_QP_PER_EP` identical at both endpoints. The test environment
also needs `HSA_ENABLE_IPC_MODE_LEGACY=1`, `RDMAV_FORK_SAFE=1`,
`MC_GID_INDEX=1` and `MOONCAKE_DISABLE_HIP_DMABUF=1`.

The prefill request must generate exactly one token and set
`kv_transfer_params.do_remote_decode=true`. Its response supplies
`kv_transfer_params.rid` and `first_token_id`. Forward these directly to
TileRT `/pd/decode`; do not reconstruct the token ID from decoded text.

## Current verification and limitations

Three real cross-node requests (128/129/257 prompt tokens) completed RDMA
transfer, TileRT conversion and native decoding. Connector and idle KV
dispatch tests passed (18 tests). A 900s C1 AgentX performance run is underway.

This remains an experimental adapter: whole-request packing/sending is
synchronous, transfer failures propagate, and automatic recovery is absent.
ATOM currently exports40 backbone window layers; TileRT starts missing
draft windows empty. Correctness without simulated acceptance and complete
draft-state handoff still require verification before production use.
TileRT's vLLM HTTP router requires an ATOM bridge: omit the prefill logprobs
request, set `do_remote_decode`, and use the returned `rid`/`first_token_id`
instead of parsing vLLM's token-ID logprobs.
