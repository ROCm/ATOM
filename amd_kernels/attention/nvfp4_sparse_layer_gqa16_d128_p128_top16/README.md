# One NVFP4 sparse attention layer, 16 query heads per KV head, head dim 128, 128-token pages, top-16

## What it does

The kernels of one NVFP4 sparse (MSA) attention layer of MiniMax-M3.1, as a server calls them: the decode indexer
(`IndexDecodeTop16`, one or four GPUs), the warm-up of every variant the layer launches, and the insert, decode and
prefill entries with the served compute dtypes bound (decode fp16, prefill fp16 with the lazy rescale: exact K / V;
`DECODE_COMPUTE`, `PREFILL_COMPUTE`, `PREFILL_LAZY`). The caller passes tensors and its limits (heads, rows per
request, requests, context). No kernel of its own. Contract: each call is the call of its entry with those dtypes,
bit for bit (the entries' `reference.py`).

## Entry points

- `init`: before serving, builds the decode indexer (with the four ranks' group a collective on every rank, in order)
  and runs the entries' warm-ups for the server's limits: insert, decode, prefill, prefill indexer. Returns the
  indexer.
- `sparse_decode`, `index_sparse_decode`, `sparse_prefill`: `nvfp4_sparse_decode`,
  `index_decode.nvfp4_index_sparse_decode` and `nvfp4_sparse_prefill` with the served dtypes bound.
- `quant_out_buffers`: the decode's MXFP8 output buffers in the layout of the MXFP8 GEMM it feeds; None below
  `QUANT_OUT_MIN_ROWS` = 8 rows, where the bf16 output and the GEMM's own quant are faster.
- `qkv_norm_rope_nvfp4_insert`, `mxfp8_gemm_qkv_norm_rope_nvfp4_insert`: re-exported from
  `norm_rope_nvfp4_insert_d128_rot64_p128`.

The entries dispatch; `layer.py` only binds dtypes and runs the warm-ups.

## Performance

Not timed as a block: the entries' READMEs hold the numbers (the decode's fp16 vs bf16 table included).

## Tests

```bash
python3 -m pytest tests/attention/nvfp4_sparse_layer_gqa16_d128_p128_top16/test_layer.py -q
python3 -m pytest tests/attention/nvfp4_sparse_layer_gqa16_d128_p128_top16/test_init_warm_up.py -q
```
