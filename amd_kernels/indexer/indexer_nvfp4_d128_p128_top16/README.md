# Lightning indexer over NVFP4 index pages, head dim 128, 128-token pages, top-16 (gfx950)

## What it does
The MiniMax-M3.1 sparse-attention indexer: per (index head, query row, 128-token block) the max of `q . k` over the
row's visible tokens, then each row's top-16 blocks (score desc, ties to the larger id, -1 padded) as ids or with the
attend's page-16 table and context bound. Query e4m3 `[rows, heads, 128]`; NVFP4 index pages `[blocks,128,72]` uint8
(per token 64 B e2m1 codes, 8 B e4m3 scales) scored as `e4m3_rn_satfinite(e2m1 x scale)` on fp8 MFMA; sentinels 1e30
(init) / 1e29 (local blocks); <= 8192 blocks (1M tokens) per row. One GPU at 1 / 2 / 4 heads, or four TP ranks at
decode (a head each) on a replicated or rank-owned cache (rank r holds blocks b % 4 == r, page p at slot p // 4).
Callers: the model's sparse attention layers, `attention/nvfp4_sparse_decode_gqa16_d128_p128_top16`. Contract
(`reference.py`): scores within 2e-4 of `max(|ref|, 1)` of fp64; sentinels, ids, tables, bounds exact (AITER's
`pa_sparse_block_topk`); routes bitwise equal.

## Entry points
- `decode_top16.IndexDecodeTop16`: a step's decode top-16 ids into `[heads, rows, 16]` int32; built with the four
  ranks' gloo group it is a collective and takes the four-GPU route by request count, else one GPU. A call can stay
  on one GPU (DBO micro-batches).
- `prefill_top16.index_prefill_top16`: the top-16 ids of every ragged prefill row in one call (one-GPU scorer, then the
  ids-only top-k); `index_prefill_top16_blocker`: why a model's indexer does not fit (None: it fits); `warm_up`: one
  small call that compiles every prefill variant.
- `score_decode.nvfp4_index_score_top16`: one-GPU decode scores + top-16, two launches; `nvfp4_index_score_rows`: the
  scores alone, for a consumer that selects in its prologue (`topk.top16_indices_p128_plain_wave4`);
  `nvfp4_index_top16_from_scores`: the selection alone; `nvfp4_index_score_decode`: scores at any head and row count
  (heads x rows <= 32).
- `score_prefill.nvfp4_index_score_prefill`: fp32 scores of ragged prefill rows (`host.nvfp4_e4m3_lut`: its key
  table); `supports_nvfp4_index_score_prefill`: its shape gate.
- `topk.index_topk_decode_indices`, `topk.index_topk_prefill_indices`: top-16 ids of score rows;
  `topk.index_topk_prefill`: ids + page table + context bounds, AITER's `pa_sparse_block_topk` outputs;
  `topk.supports_index_topk`: their shape gate.
- `split4.host.init_rank_owned`: the four-GPU buffers, one set per micro-batch, a collective before graph capture;
  `split4.host.get_rank_owned`: a micro-batch's set, whose `score_topk` gives the four-GPU scores + top-16 and
  `score_push` the scores alone in rank-owned order.

Decode calls read lengths on the device (graph-safe at max_model_len). `nvfp4_index_score_top16`, `_rows`,
`score_topk` and `index_prefill_top16` are `amd_kernels.warm_up` entries (the prefill entry warms one request per
top-k window tile); the scorers and top-k called alone are not. Dispatch: `decode_top16.py` picks the route,
`score_decode/host.py` the one-head scorer, `split4/host.py` the four-GPU kernels by rows, `score_prefill/host.py`
the tile by work, `topk/host.py` the window tile by `max_seq_len`.

## Performance
Graph replay, host 1M, cold pages, us per call, max over ranks; AgentX = request i of N at the (i + 0.5) / N quantile
of AgentX c16 + c32 inputs; AgentX decode = the decode-time contexts (Tests) of c64 runs at 4 rows, c16 + c32 at 8.

| Route | Rows per request x requests | Context | us | Baseline (us) |
|---|---|---|---|---|
| `nvfp4_index_score_top16` | 8 x 1 / 8 x 4 / 4 x 1 | 32K, 192K | 5.83 / 7.43 / 5.72, 7.87 / 14.75 / 7.78 | `nvfp4_index_score_decode` + `index_topk_decode_indices` 8.23 / 9.38 / 8.06, 10.49 / 20.16 / 10.25 |
| `index_prefill_top16`, 1 head, four GPUs at once | M 4096 / 8192 / 16384 x 1 | prefix 0 / 8K / 32K / 128K / 256K | 11.2 / 17.6 / 35.5 / 74.6 / 126.7, 22.0 / 29.3 / 50.2 / 123.2 / 226.5, 44.7 / 57.7 / 88.8 / 241.3 / 435.3 | vLLM's in-tree scorer + top-k: 11.1 / 17.7 / 42.9 / 91.3 / 146.2, 21.8 / 29.4 / 58.9 / 140.6 / 244.4, 44.7 / 61.3 / 98.2 / 261.1 / 453.8 |
| `IndexDecodeTop16` four GPUs | 4 x 6 / 12 / 24 / 32, 8 x 5 / 12 / 16 | AgentX decode | 13.57 / 20.67 / 26.70 / 28.50, 16.20 / 23.06 / 23.60 | a four-rank flag wait, then the selector: 14.99 / 21.54 / 27.59 / 28.92, 17.53 / 23.79 / 23.99 |

## Tests
A decode-time contexts file (`--lens-json` of the block benchmark) holds, per request count N, the (i + 0.5) / N
quantiles of the contexts of the requests decoding at sampled times of a workload's client records.

```bash
HIP_VISIBLE_DEVICES=0,1,2,3 python3 -m pytest tests/indexer/indexer_nvfp4_d128_p128_top16 -q  # four-GPU tests skip on fewer cards
python3 benchmarks/indexer/indexer_nvfp4_d128_p128_top16/benchmark_indexer.py [--prefill]
python3 benchmarks/indexer/indexer_nvfp4_d128_p128_top16/benchmark_rank_owned.py --ql 4 --decode '' --lens-json <your lengths.json> --reqs 8,32,64  # 4 GPUs
HIP_VISIBLE_DEVICES=0,1,2,3 python3 benchmarks/attention/nvfp4_sparse_decode_gqa16_d128_p128_top16/benchmark_index_decode_ranks4.py --ql 4 --lens-json <your decode-time contexts.json> --reqs 4,6,8,12  # the decode block on each route (SPLIT_MIN_REQS)
python3 benchmarks/indexer/indexer_nvfp4_d128_p128_top16/benchmark_decode_topk.py --selector rank_owned --ref today=<frozen copy> --lens <your lengths.json> --reqs 32,48 --ql 4
```
