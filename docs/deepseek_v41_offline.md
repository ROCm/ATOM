# DeepSeek-V4.1 offline text development baseline

The eager text backbone executes all 40 layers, including Engram and the
checkpoint's FP8/FP4 quantization steps. ModelRunner registration remains gated:
the paged request lifecycle and model-quality acceptance are still in progress.
This entry point is for development and numerical comparison.

## Run

Use a ROCm environment with ATOM and AITER installed and the native checkpoint
available locally. The tested setup uses MI355X and TP8; a shorter full-model
probe also ran under TP4.

```bash
torchrun --standalone --nproc_per_node=8 \
  -m atom.examples.deepseek_v41_offline \
  --model /mnt/DeepSeek-V4.1-Flash \
  --prompt "The capital of France is" \
  --max-length 256 --max-new-tokens 16
```

The example accepts one raw text prefix. It owns the parallel group, prepares
Engram rows through the host provider, and maintains a private eager cache.
Chat/tool encoding, vision, serving batches, paging, graph execution and
speculative decoding are separate milestones.

## Equal-score index selection

`text_config.index_topk_tie_break` accepts `small_position` (default) or
`large_position`. The same policy applies to CSA2 top-k, candidate blocks and
Reindex. The newest visible candidate block is always retained; selected token
positions are returned in ascending order under either policy.

Use `--index-topk-tie-break large_position` to override the checkpoint config for
an offline run. The model captures the selected policy at construction time.

## Implementation boundaries

- `atom/models/deepseek_v41/` composes projections, mHC, attention and experts.
- `atom/model_ops/deepseek_v41/` owns rotary, compressor, indexer and MoE math.
- `atom/model_ops/attentions/deepseek_v41_state.py` owns the eager SWA,
  incomplete compression groups and one global main/index allocation per owner.
- `atom/examples/deepseek_v41_offline.py` owns checkpoint loading and request
  preparation. Token hashing and table lookup use the existing Engram runtime.

The four global cache owners are layers 2, 8, 14 and 20. Reuse layers share
their owner's values and top-k; Reindex layers score only the supplied compact
candidate blocks. All attention caches contain BF16 values after the original
FP8/FP4 quantization and dequantization steps. Attention calls the existing V4
BF16 paged prefill/decode entry points with
unchanged dispatch and kernels. The earlier fixed-64 tile override was removed.
The cache owns a fixed BF16 row pool: one SWA ring per layer and one global
region per owner. Prefill reads the prior ring and current chunk separately;
decode writes its current row before reading the pool. Both use V4's existing
ring writer. No forward step concatenates the complete global KV history.
`atom/model_ops/deepseek_v41/paged_indices.py` prepares the caller's CSR indices;
it contains no attention computation.

Weights retain native FP8 32x32 or FP4 1x32 storage. The correctness GEMM converts
register tiles for BF16 MFMA, retaining FP8 activation quantization. Routed
experts use whole-expert partitioning; the shared expert uses TP with FP32
partials and rounds after reduction. The eager dispatch still synchronizes
expert counts to the CPU. Packed caches, native FP8 MFMA, fused dispatch and
performance tuning remain pending.

## Validation and open gates

Reference model sources are pinned to revision
`dba1be0a40aa45a94ad051997016db3960a90277`; tests verify their hashes.

```bash
ATOM_DSV41_REFERENCE=/mnt/DeepSeek-V4.1-Flash \
  python -m pytest -q tests/models/deepseek_v41
```

Operator tests compare with independent PyTorch contracts and pinned upstream
methods. Attention integration fixtures retain the real 512-dimensional heads
and eight local heads. Cache tests cover Full/Reuse/Reindex ownership, causal
visibility, ring wraparound, separate requests and batch rows, and ratio-2
boundaries. The actual V4 BF16 prefill/decode tests cover a single sink across
SWA/global representations and 192/640 selected entries.

Quantized graph tests first compare each real backend's output with the upstream
attention output (relative L2 <= 0.003), then supply the upstream output to the
remaining projections. This controls accepted BF16 rounding before discontinuous
FP8 quantization and isolates model composition. These tests do not establish
end-to-end numerical parity. The production model has no such substitution.

An isolated run of the official TileLang HIP attention also checked the
unmodified V4 kernels at 5/192/640 entries. The maximum observed relative L2 was
0.002461 for decode and 0.001203 for prefill across those cases.

The current paged V4 path completed an independent TP8 full-weight comparison
using one loaded block at a time, without aligning GEMMs, attention, norms or
collective implementations. All 40 layers ran for five fixtures and 20 calls;
peak allocated HBM was 6.46 GiB per rank. Across 248 scored next-token positions,
top-1 agreement was 238/248, target mean NLL 1.329078 and reference mean NLL
1.333321. Per-call logits relative L2 ranged from 0.103 to 0.604. These results
leave the full-model numerical/quality gate open; aggregate NLL is not sufficient
to close it. Streaming is an external diagnostic, not a model execution feature.

Earlier full-weight results below describe assembly commit `446d5a03a`, before
the switch from the generic helper to paged V4 attention. They are historical
evidence, not acceptance of the current backend:

- Five fixtures (English, Chinese, code, Unicode and a 200-token window-boundary
  sequence) cover 20 prefill/decode calls. Every layer residual and final logit
  matches exactly when both model graphs use the same GEMM, attention, norm,
  shared-expert K partition and collective reduction order. This validates model
  composition and state flow; it is not an independent kernel or quality gate.
- With the original independent PyTorch GEMM/attention oracle, the same corpus
  has 248 scored next-token positions: target mean NLL 1.284154, reference
  mean NLL 1.333241, top-1 agreement 235/248. At reference margins
  above 0.1, agreement is 235/246. Cross-backend numerical
  acceptance remains open despite the lower aggregate target NLL.
- A temporary lm_eval adapter ran GSM8K, five-shot, greedy, on four examples.
  Both strict and flexible exact-match scores were 4/4. This is a smoke test;
  it does not establish a model-quality score or replace paired evaluation.

The earlier short real-weight probe isolated a one-element BF16 attention
difference and FP32 collective reduction differences that accumulate through
quantized layers. With paged attention, the reference and production kernels
also differ in KV tile and region accumulation order. Full-model numerical and
quality acceptance remains open. Long contexts that exercise top-512 pruning
and the complete serving state lifecycle still need their own acceptance checks.

A GPU graph microbenchmark on MI355X measured decode attention plus the former
full-history KV concatenation, with H=8, D=512 and 640 selected entries. At 8192
history rows, median latency changed from 157.8 to 7.8 microseconds for batch 1,
and from 245.7 to 9.3 microseconds for batch 16, over three measurements. The
baseline is the helper from `446d5a03a`; the new case reads the existing pool.
Index preparation, ring writes, projections, MoE and serving are excluded.
These measurements are not end-to-end latency or throughput results.
