# DeepSeek-V4.1 image requests

The native multimodal processor reuses the checkpoint message encoder. It applies
DeepSeek's resize/pad, normalization, patch ordering and image token budget. The
vision tower uses BF16, per-image bidirectional PyTorch SDPA, 2D RoPE and the
released 3x3 unfold aligner. The language model keeps V4 BF16 attention and 1D
positions. No language attention kernel changes are required.

Image spans are explicit: delimiter embeddings, VL routing and Engram DEAD
history all derive from the same token types. Prefix hashes include processed
image content and layout, so different images with identical placeholder token
IDs cannot share KV. Vision weights belong to a multimodal subclass and an
explicit checkpoint scope; offline text loading still excludes vision by default.

## Chunked images and embedding leases

The worker caches image embeddings under request leases and scatters only the
current chunk's explicit span intersections. CPU payloads remain available for
retry; after the first successful prefill, subsequent prefills carry descriptors
and decode carries no image payload. The last consumer releases the GPU entry.
CPU payload construction, GPU embedding ownership and scheduler policy live in
separate modules.

Small-row `wo_a` and mHC projection arithmetic is handled in the operator layer;
V4 BF16 attention and inverse RoPE are reused unchanged.

## Accepted chunking quality tradeoff

Chunked image prefill is **not** bitwise identical to an atomic one. The native
BF16 `wo_a` and FP32 mHC GEMMs change reduction order with row count, and
downstream quantization can amplify the difference — so a prompt split into
63-row chunks takes a different arithmetic path from the same prompt prefilled
whole. Fixed-input attention and inverse-RoPE checks on two text samples across
four ranks were bitwise equal, which does not cover every attention layout but
gives no reason to replace the V4 kernels.

The measured task difference, accepted on 2026-09-14:

| Chinese LogiQA, 651 documents | Atomic prefill | 63-row chunks |
|---|---:|---:|
| Raw accuracy | 475/651 | 471/651 |
| Normalized accuracy | 425/651 | 423/651 |

Raw has 19 gains / 23 losses and normalized 19 gains / 21 losses; the paired 95%
intervals include zero, which is not proof of equivalence. This is an accepted
tradeoff for chunking, not a claim of identical task quality, and it does not
waive quality checks for later changes to the projection path.

## Validation

- Nine independent image tests passed. Six resize/patch fixtures include tiny,
  tall, wide and token-budget-limited images; preprocessing is bitwise equal to
  the pinned released code. The full real-weight 32-layer ViT, aligner and learned
  delimiters are also bitwise equal, including padding and multiple image grids.
- Native TP4 ModelRunner/Scheduler with packed cache and PIECEWISE dense graphs
  passed single, multiple and interleaved image generation. All four ranks
  generated the same IDs; outputs were `Red`, `Blue, red` and `A: green, B: blue`.
  The same interleaved fixture under the independently loaded official full
  vision/text reference returned `A is green, B is blue`. These are smoke
  fixtures, not a visual benchmark score.
- The TP4 runtime validation covers two and three concurrent image requests,
  shared image encoding, batch reorder, preempt/checkpoint resume, prefix forks
  and final-request cancellation with packed KV and PIECEWISE graphs. Each group
  shares two image encodes and all leases are empty at completion; the 63-row
  chunk case includes 189-row prefill batches and three-request decode. Mixed
  prefill/decode batches and arbitrary-batch visual quality are not covered.

The interleaved green/blue fixture exists because of a concrete failure: an
expert backend that passed both the single-image and adjacent-image fixtures
answered it `A: Green. B: White`, where the eager model and the official
reference were both correct. That backend has since been replaced by V4's
`FusedMoE`, but the fixture is the reason a new expert path is not qualified on
single-image cases alone.
