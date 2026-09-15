# DeepSeek-V4.1 image requests (P07)

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

P07's accepted image path used atomic prefills bounded by max_num_batched_tokens.
P08 adds chunking and request-owned embedding lifetime, including preemption.
Its runtime contracts passed and the user accepted the recorded small-chunk
quality difference; see the validation and exact scores below.

## Validation

- Nine independent image tests passed. Six resize/patch fixtures include tiny,
  tall, wide and token-budget-limited images; preprocessing is bitwise equal to
  the pinned released code. The full real-weight 32-layer ViT, aligner and learned
  delimiters are also bitwise equal, including padding and multiple image grids.
- Scheduler, block manager, protocol and runtime regressions: 2,508 passed,
  56 skipped, 3 xfailed. Focused graph/vision/Engram/math: 66 passed.
- Native TP4 ModelRunner/Scheduler, packed cache, PIECEWISE dense graphs and the
  default eager experts passed single, multiple and interleaved image generation.
  All four ranks generated the same IDs. Outputs were Red; Blue, red; and
  A: green, B: blue. These are smoke fixtures, not a visual benchmark score.
- The same interleaved fixture under the independently loaded official full
  vision/text reference returned A is green, B is blue.

## Separate expert-backend observation

The optional P09 AITER MoE adapter passed the single and adjacent-image smoke
fixtures, but answered the interleaved green/blue fixture with A: Green. B:
White, where the eager model and the official reference both answered
correctly. Both of those expert paths have since been replaced by V4's
`FusedMoE`, so this stands only as the reason the fixtures exist: an expert
path can pass the single-image cases and still fail the interleaved one. The
vision results below were taken on the eager loop and have not been re-measured.

Logs, commands, fixed inputs and the paired diagnostic are archived under
/app/logs_claude/atom_dsv41_flash_impl_20260912/p07_vision/ in ljin_dev.
Run the production smoke with torchrun --nproc_per_node=4 -m
tests.attentions.deepseek_v41.validate_vision --output /tmp/v41-vision.json.
The test-only --expert-backend aiter option reproduces the separate observation.

## P08: chunked images and embedding leases

The worker caches image embeddings under request leases and scatters only the
current chunk's explicit span intersections. CPU payloads remain available for
retry; after the first successful prefill, subsequent prefills carry descriptors
and decode carries no image payload. The last consumer releases the GPU entry.
CPU payload construction, GPU embedding ownership and scheduler policy live in
separate modules.

The TP4 runtime validator covers two and three concurrent image requests,
shared image encoding, batch reorder, preempt/checkpoint resume, prefix forks
and final-request cancellation with packed KV and PIECEWISE graphs. Each group
shares two image encodes; all leases are empty at completion. All four ranks
produce the same outputs and each active request identifies green before blue.
The chunk63 test includes 189-row prefill batches and three-request decode.
Mixed prefill/decode batches and arbitrary-batch visual quality are not covered
by this smoke test. Related CPU multimodal/scheduler/block-manager regression:
211 passed; affected projection/model/vision regression: 44 passed.

```bash
HIP_VISIBLE_DEVICES=0,1,2,3 AITER_REUSE_IDENTICAL_COMM_GROUPS=1 OMP_NUM_THREADS=4 \
  torchrun --standalone --nproc_per_node=4 \
  -m tests.attentions.deepseek_v41.validate_vision_runtime \
  --model /mnt/DeepSeek-V4.1-Flash --chunk-size 63 --requests 2 3 \
  --output /tmp/v41-vision-runtime.json
```

Small-row wo_a/mHC projection arithmetic is handled in the operator layer.
V4 BF16 attention and inverse RoPE are reused. The accepted small-chunk quality
tradeoff, projection policy, numerical results and performance measurements are
documented in [Small-chunk projections](deepseek_v41_small_chunks.md). The user
accepted Chinese chunk63 raw471/651 versus475/651 and normalized423/651
versus425/651 on 2026-09-14; this is not a claim of identical task quality.

Detailed inputs, scripts, rejected candidates and results are in
/app/logs_claude/atom_dsv41_flash_impl_20260912/p08_multimodal/ in ljin_dev.
