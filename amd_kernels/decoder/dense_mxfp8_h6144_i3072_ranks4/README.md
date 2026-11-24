# Dense decoder layers around their attention on four TP ranks, MXFP8 linears, hidden 6144, I 3072 per rank

## What it does

A dense decoder layer stack without its attention, on four TP ranks: the input fc split by N (5 taps of 6144 ->
6144), the embedding owner read + Gemma norm + MXFP8 quant, the all-reduce + add + Gemma norm (+ quant) tails, the
MLP (gate_up -> SwiGLU-OAI -> down, TP partial), the stack around the caller's attention and the context K / V chain
(Gemma norm + MXFP8 GEMM + multi-layer K norm, RoPE, fp8 insert). Caller: the MiniMax-M3.1 DSpark draft at TP4. Each
call picks its consumer GEMM's code layout (`gemm.mxfp8_common.gemm.prequant_code_tile_rows`) and row-count variant;
weights are preshuffled MXFP8, gate_up [6144, 6144] (gate | up), down [6144, 3072]. Contract (`reference.py`):
compositions of the per-kernel references, cosine ~1; each call equals its per-kernel chain bitwise.

## Entry points

- `DenseDecoderRanks4`: the layers' communicators; built collectively (every rank, the same order) after the weights
  load, before any capture: packs this rank's 1536 fc rows (the caller may free the full weight), maps this rank's
  embedding shard, builds and warms up three comms. Its calls:
  - `fc`, `fc_cat`: the split input fc of the five taps (unit column stride, row stride a multiple of 8; asserted) in
    pieces of up to 2048 rows; eager, every rank.
  - `embed_norm_quant`: the embedding owner read + Gemma norm + MXFP8 codes of up to 2048 ids, and the residual.
  - `ar_norm_quant`, `ar_norm`: the all-reduce + add + Gemma norm (+ quant) tails, up to 2048 rows; the residual stays
    on the owned rows.
  - `forward`: the stack on 1-2048 ids (one `DenseLayer` per layer) around the caller's attention callback, which
    returns a layer's attention + o_proj TP partial on the qkv GEMM's codes.
  - `store_context_kv`: every layer's context K / V into its fp8 cache (static shapes).
  - `capture`: a context around the graph capture of these calls; `close`: the collective teardown.
- `stack_blocker`, `context_kv_blocker`: why a model's layer stack or context K / V chain does not fit (None: it fits).
- `mxfp8_swiglu_oai_mlp`: the MLP's down output, bf16 [m, 6144], from the gate_up input codes, 1-2048 rows, in a
  `MXFP8_SWIGLU_FORMS` form (`bf16_chain` or `fp32`); `mlp_supported`: whether the MLP weights fit.
- `gemma_rmsnorm_mxfp8_linear`: Gemma norm -> MXFP8 codes in the GEMM's layout -> the GEMM, bf16 [m, N], any row count
  and N.

The MLP and the norm + GEMM are entries (`amd_kernels.warm_up`): the first call of each form or weight shape compiles
every row count's variants (MLP 1-2048 rows; norm + GEMM 1-256 rows and the tiers to 32768). No kernel of its own:
each call sequences library entries (`host.py`, `mlp.py` with the row-count dispatch, `norm_linear.py`).

## Performance

Not timed as a block. The launched entries' numbers are in their READMEs.

## Tests

```bash
python3 -m pytest tests/decoder/dense_mxfp8_h6144_i3072_ranks4 -q  # test_decoder_ranks4.py: 4 GPUs
```
