# Sequence-parallel layer hand-offs, hidden 6144, 2048 attention columns per rank, four ranks

## What it does

`SeqParallelHandoffs` runs the chains of a sequence-parallel (SP) decoder layer of MiniMax-M3.1-Flash TP4 prefill
between the head-sharded parts (qkv, attention, the FFN of every token) and the token-sharded residual stream, with the
four `pull_collectives_ranks4` communicators and the warm-up; the caller keeps the layer loop, attention, the first
and final norms and the DBO gating. o_proj_full MXFP8 `[6144, 8192]` (`gemm/mxfp8_common`), the MoE of
`moe/mxfp4_h6144_i768_e128_top4`. Rank r holds `shard_rows(t)` tokens (a multiple of 32), t <= 32768. Contract
(`reference.py`): `pack_reference` bit for bit (the Triton `mxfp8_quant_pack` but for blocks with a maximum below
2^-119); every phase output bit for bit its kernels' outputs on the rows the collectives deliver (gathers and fp32
reduce-scatter sums in rank order); no atomics, no quantized reduction. Eager only (host-side sequence state), on the
current stream, the same calls on every rank.

## Entry points

- `SeqParallelHandoffs`: a collective; builds the transport for a batch-token limit (clamped to 4096..32768) and
  compiles it; `supports` says whether a token count fits.
- `SeqParallelHandoffs.warm_up`: local, before serving: every compute variant at 4096-32768 tokens.
- `SeqParallelHandoffs.input_gather`: the first layer's normed bf16 shard -> MXFP8 codes + AITER-shuffled scales of
  every token.
- `SeqParallelHandoffs.attention_moe`: this rank's heads of every token -> the `MoeInputCompact` of every token and the
  new residual shard; `attention_dense`: the same o_proj, then the dense MLP's codes + scales of every token.
- `SeqParallelHandoffs.ffn_norm_gather`: the FFN output of every token (MoE slots + shared, or a dense partial) -> the
  next layer's codes, scales and residual shard.
- `SeqParallelHandoffs.ffn_reduce_scatter`: after the last layer, this rank's summed rows; `gather`: bf16 rows of
  every token.
- `shard`: a rank's rows of a full tensor, zero padded; `shard_rows`: their count (a multiple of 32).
- `mxfp8_pack_q2048` (also `pack`): bf16 `[M, 2048]` -> uint8 `[M, 2112]` (2048 codes, 64 scale bytes per row).
- `SdmaSeqParallelHandoffs`: the same phases and bits on SDMA pushes (`sdma_push_ranks4`); each phase takes a hook
  that runs after a push and before its wait (DBO: the yield); one instance per microbatch; `drain` waits on the host
  for every push.

The phase order and kernels of each chain are in `handoffs.py` and `handoffs_sdma.py`.

## Performance

One MI355X: the pack vs the Triton `mxfp8_quant_pack`, graph replay of 100 calls, faster of two rounds,
bit for bit; SDMA-form serving numbers: `sdma_push_ranks4`.

| rows | 4096 | 8885 | 15745 | 32768 |
|---|---:|---:|---:|---:|
| FlyDSL (us) | 3.2 | 8.7 | 15.5 | 30.0 |
| Triton (us) | 3.8 | 9.1 | 14.8 | 28.8 |

## Tests

```bash
python3 -m pytest tests/collectives/seq_parallel_handoffs_h6144_q2048_ranks4/test_pack.py -q
python3 -m pytest tests/collectives/seq_parallel_handoffs_h6144_q2048_ranks4/test_handoffs.py -q  # 4 GPUs
python3 -m pytest tests/collectives/seq_parallel_handoffs_h6144_q2048_ranks4/test_shard.py -q     # 4 GPUs
python3 -m pytest tests/collectives/seq_parallel_handoffs_h6144_q2048_ranks4/test_warm_up.py -q   # 4 GPUs, alone
```
