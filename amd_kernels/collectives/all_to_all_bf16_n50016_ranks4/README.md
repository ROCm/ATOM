# bf16 owner-row all-to-all, 50016 columns per source, four ranks

## What it does

`RowAllToAll` moves the vocabulary-sharded logits rows of MiniMax-M3.1-Flash TP4 (DSpark 7 target and draft sampling)
to the rank that owns each request, straight into the final `[owned rows, 200064]` layout, in place of RCCL's uint8
`all_to_all_single` + permute of the target and the draft rows in the batch-sharded sampler; grammar mask, NaN
count and sampling run on the received rows in order.
Input: bf16 `[rows, 50016]` contiguous, rows <= `max_rows` <= 2048, sorted stably by owner. Contract (`reference.py`):
rank r receives rows `[sum(counts[:r]), sum(counts[:r + 1]))` of every source, concatenated along the vocabulary,
every bf16 bit kept (NaN payloads too). No atomic output adds.

## Entry points

- `RowAllToAll`: a collective; two IPC slots of `max_rows x 50016 x 2` B per rank (410 MB at 2048 rows).
- `RowAllToAll.warmup`: a collective before capture; compiles every stage and pull grid.
- `RowAllToAll.__call__`: stage + pull of owner-sorted rows into a preallocated `[>= own rows, 200064]` bf16 output;
  the counts are a CUDA int32 `[4]`, the same on every rank, summing to the rows (device values, not checked).
- `RowAllToAll.slot`: this rank's send slot (pool address, slot bytes, device counter, flag address) for a producer
  that writes and publishes it itself (`bf16_gemm_n50016_k6144`'s publish); `RowAllToAll.pull` then receives.
- `RowAllToAll.close`: a collective, after every graph that uses the communicator is retired.

All ranks call every exchange in the same order, empty receivers included; one communicator per stream. Counts may
change between graph replays (device sequence, no host state). `host.py` picks the stage and pull grids by rows.

## Performance

Four GPUs of one MI355X node. `benchmark_exchange.py`: graph replay, inputs rotating over 256 MB, five
alternating rounds, MAX over ranks, median us, dense owners; baseline RCCL uint8 all-to-all + permute.

| requests (7-8 rows each) | target: baseline | this entry | draft: baseline | this entry |
|---:|---:|---:|---:|---:|
| 10 | 72.32 | 57.97 | 67.04 | 56.41 |
| 32 | 169.19 | 133.08 | 152.04 | 117.49 |
| 256 | 1159.30 | 958.99 | 1015.33 | 845.02 |

The target exchange is 15% / 4% above the 55.5 GB/s-per-link floor at 32 / 256 requests.

## Tests

```bash
python3 -m pytest tests/collectives/all_to_all_bf16_n50016_ranks4 -q               # 4 GPUs
python3 benchmarks/collectives/all_to_all_bf16_n50016_ranks4/benchmark_exchange.py # 4 GPUs
```
