# MiniMax M3.1 PP4 to DCP4 KV transfer

This implementation uses PP4 × TP1 for prefill and TP4 × DCP4 for decode.
Decode stores all four KV heads and a shard of the tokens on each rank.
The supported compute layout is `--block-size 128 --dcp-config '{"interleave_size":128}'`.
Set `ATOM_MONO_ENABLE=0` and `ATOM_M3_INDEXER_CP=0`; TBO is not supported by this path.
Target and DSpark attention use the same ownership and block IDs.

## Layout and protocol

For global token `p`, its owner is `(p // 128) % DCP`, and its local
position is `(p // (128 * DCP)) * 128 + p % 128`.
A Decode block therefore represents `DCP` Prefill blocks. Cached-prefix skips
and chunk publication must use that ratio before trimming the source block table.

Each PAGE describes its physical byte shape and ordered token axes. This covers
page-128 NVFP4 K/V and independent scales, FP4/FP8 index keys, and SHUFFLE K/V
whose logical block is physically eight page-16 subpages. Draft planes have
independent pools but use the target block IDs. PP stages retain their global
layer/field identities; only the stage holding the draft sends draft planes.

Both endpoints must declare identical all-head layouts. The complete stage is
validated before writing. Descriptors never cross a source or destination PAGE,
so PAGE-aligned memory-region boundaries remain valid. S=128 batches whole-page
address calculations (up to 4096 separate descriptors per batch);
the transfer planner also supports S=1/16 through byte runs, but M3 compute is
currently restricted to S=128.

## Compute

TP projection shards gather Q/K/V and index Q in head order; index K is already
replicated. Fused norm/RoPE writes only owned slots. Each rank scores its owned
index blocks, exchanges scores in global block order, and uses the native packed
top-k selector, including its init/local priorities and tie-breaking.

Attention reads only owned selected blocks and emits an output and natural-log
LSE. The distributed merge accumulates in FP32; empty shards contribute zero
output and negative-infinite LSE. The result is sliced back to the TP head share
before the output projection. Ragged cached prefill and q=8 verify use global
per-row causal lengths. DSpark uses a capped noncausal block window and preserves
the existing page-16 rounding of the 4096-token sliding window.

The first implementation uses portable Triton local attention and a full-score
gather. It does not yet use the specialized gfx950 NVFP4 kernel or a distributed
candidate-only exchange; long-context performance requires measurement.

## Bytes per token

The public `amd/MiniMax-M3.1-Flash-Quark-MXFP4` configuration has 60 sparse target
layers, 4 KV heads, head dimension 128, and 5 draft layers.

| Plane | Bytes per global token |
| --- | ---: |
| Target NVFP4 K/V plus per-16-element scales | `60 * 2 * 4 * (64 + 8) = 34,560` |
| Target FP4 index plus scales | `60 * (64 + 8) = 4,320` |
| Draft FP8 K/V plus FP32 per-token scales | `5 * 2 * 4 * (128 + 4) = 5,280` |
| Total | **44,160** |

This excludes page-tail padding and protocol overhead. The FP8 baseline is
76,320 B/token. DCP4's aggregate transferred payload remains 44,160 B/global token;
each rank receives approximately one quarter, subject to block tails.

## Validation

- `tests/test_m3_dcp_transfer.py`: bytewise real Mooncake CPU copies across PP
  stages, DCP2/4, S=1/16/128, physical KV formats, draft pools, malformed wire
  layouts, cached-prefix skips and incremental handoff.
- `tests/test_m3_dcp_attention.py`: GPU comparisons with an independent PyTorch
  reference, real 64-query-head geometry, independent K/V scales, empty shards,
  causal block boundaries, global top-k and graph replay.
- `tests/models/minimax_m3/validate_dcp_collectives.py`: four-GPU AITER projection
  gathers, global top-k, FP32 merge and full attention graph replay.
- `tests/models/minimax_m3/validate_dcp_module.py`: the real sparse layer on
  four GPUs, including projection and fused writers, compared with dense
  PyTorch attention over unsharded quantized KV. Checks byte equality and q=8
  whole-layer graph replay with changed inputs (short contexts, all blocks selected).
- CI accuracy cases: `minimax-m31-flash-mxfp4-1p1d-pp4-dcp4-dspark7-nvfp4-gsm8k-5shot-c32`
  and its `gsm8k-5shot-chunked-c32` counterpart. They retain the PP4 integration's
  full 1319-question, five-shot, real-acceptance GSM8K gate (0.94).

- `test_real_nvfp4_rope_writer_keeps_queries_for_unowned_tokens` exercises the
  real AITER fused norm/RoPE and NVFP4/FP4 writers with alternating unowned slots.
- Non-P/D accuracy baseline:
  `minimax-m31-flash-mxfp4-agg-tp4-dspark7-nvfp4-gsm8k-5shot-c32` uses the same
  model, KV formats, five-shot prompt, real DSpark K7 and TP4 without a connector.
- C56 performance cases:
  `minimax-m31-flash-mxfp4-1p1d-pp4-dcp4-dspark7-nvfp4-mcstore-agentx-c56`
  and its `agentx-chunked-c56` counterpart retain AgentX 1.0.12, the one-hour
  measurement window, 1 TiB Store and synthetic AL 4.04 performance recipe.
  Run these after real-acceptance accuracy passes. Use the profiling phase's
  `atom:pd_kv_transfer_seconds` sum/count deltas for the request-weighted mean;
  this includes dispatch, handshake and notification until all workers finish.
  Report warmup separately, and do not substitute TTFT or an unweighted average
  of the report's rolling means for the transfer wait.

GPU kernel/collective tests alone do not establish model-level accuracy or the
issue #314 C56 transfer-wait target. Those results must be collected separately.
