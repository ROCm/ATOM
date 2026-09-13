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

At this stage image prefills remain atomic and bounded by max_num_batched_tokens.
P08 adds chunking and request-owned embedding lifetime, including preemption.
The generic input path uses explicit span intersections after a prefix hit.

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

## Separate AITER observation

The optional P09 AITER MoE passed the single and adjacent-image smoke fixtures,
but answered the interleaved green/blue fixture with A: Green. B: White.
The accepted eager model and the official reference both answered correctly.
This is a recorded visual regression of the optional expert backend, not a P07
pass and not a new numerical tolerance. Default expert_backend remains eager.
Do not infer full visual quality acceptance from these smoke fixtures or the
previous text-only acceptance of P09's numerical loss.

Logs, commands, fixed inputs and the paired diagnostic are archived under
/app/logs_claude/atom_dsv41_flash_impl_20260912/p07_vision/ in ljin_dev.
Run the production smoke with torchrun --nproc_per_node=4 -m
tests.attentions.deepseek_v41.validate_vision --output /tmp/v41-vision.json.
The test-only --expert-backend aiter option reproduces the separate observation.
