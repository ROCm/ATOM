# DeepSeek-V4.1 image requests

The native multimodal processor reuses the checkpoint message encoder for
resize/pad, normalization, patch ordering and the image token budget. The
vision tower uses BF16, per-image bidirectional PyTorch SDPA, 2D RoPE and the
released 3x3 unfold aligner. The language model uses V4 BF16 attention,
inverse RoPE and one-dimensional positions.

Image spans are explicit: delimiter embeddings, routing and Engram history
derive from the same token types. Prefix hashes include processed image
content and layout, so different images with identical placeholder token IDs
cannot share KV. Vision weights belong to the multimodal subclass and its
checkpoint scope; offline text loading excludes vision by default.

## API image input

Chat Completions, Responses and Anthropic Messages accept images and share
one path: each endpoint translates its image blocks into Chat `image_url`
parts (`atom/entrypoints/openai/images.py`), which are loaded, checked and
decoded before the processor sees them. No endpoint drops an image; a block it
cannot serve is HTTP 400 before generation.

| API | Image block | Roles |
|---|---|---|
| Chat | `image_url` (`url`, optional `detail`); `file` with `file_data` | `user` |
| Responses | `input_image` (`image_url`, optional `detail`) | `user`, `developer`, `function_call_output` |
| Messages | `image` with a `base64` (`media_type`) or `url` source | `user`, including `tool_result` |

Following DeepSeek's vision contract: JPEG, PNG, GIF (first frame) and WebP,
recognised from the bytes rather than the declared type; base64 data URLs or
http(s) URLs only, never a server path or `file://`; at most 32 MiB and 8192 px
a side per image, 600 images per request, a 60 s download and 8192-character
URLs. `detail: "low"` downscales to fit 512x512; `high`, `original` and `auto`
keep the image. There is no Files API, so `file_id` and Anthropic `file`
sources are refused. Images in `system` or `assistant` messages, P/D handoff
and a prompt with images longer than the context are refused as well; a text
prompt is still truncated as before.

URLs must be public: the host, and the host of every redirect, must resolve
only to global addresses, and the download connects to the address that was
checked. Loopback, private, link-local and other internal addresses are
refused, and no HTTP proxy is used. Set `ATOM_IMAGE_URL_ALLOW_PRIVATE=1`
when clients may fetch from the internal network, for example an intranet
image store or egress only through a proxy.

A V4.1 deployment that cannot run the vision path -- DSpark or DP attention --
refuses images on every endpoint. Serve images from a deployment without
`--method dspark`.

## Chunked images and embedding leases

The worker caches image embeddings under request leases and scatters the
current chunk's span intersections. CPU payloads remain available for retry.
After the first successful prefill, subsequent prefills carry descriptors;
decode carries no image payload. The last consumer releases the GPU entry.

CPU payload construction, GPU embedding ownership and scheduler policy live
in separate modules. Prefill chunking, preemption and prefix forks preserve
request ownership of image embeddings.

## Routing and Engram

Image tokens select experts with `gate.bias_vl`; text tokens use
`gate.e_score_correction_bias`. The shared `mm_topk` operator chooses the
per-token bias, experts and unbiased normalized weights in one kernel. V4.1
passes its explicit image mask rather than inferring modality from token IDs.

The routing hook is registered at construction and reads the active forward
metadata. Offline forward scopes that metadata for each call. Text-only calls
use V4's original router, and V4 owns expert execution. Engram excludes image
spans, including delimiters, through the backbone's existing token mask.

For 384 experts on wave64, expert-score ties follow AITER's lane priority and
six-element per-lane sorting network. This MoE expert-selection rule is
independent of the indexer's token-position tie rule.

## Current limitations

Visual quality is not yet stable across concurrent requests, chunk sizes and
execution modes. Both level-0 eager and FULL-graph execution can produce
incorrect image answers. Successful graph capture or request-state restoration
does not establish visual quality.

Chunked prefill need not produce the same outputs as whole-prompt prefill.
BF16 projection and FP32 mHC reductions can change with row count, and subsequent
quantization can amplify those differences. Evaluate the intended chunk size
and multi-image workload before deployment. Multimodal DSpark is not supported.

## Validation

The manual validator exercises two and three concurrent image requests,
shared image encoding, reorder, checkpoint restore, prefix forks and final
cancellation. It checks output content as well as embedding-lease cleanup.

```bash
HIP_VISIBLE_DEVICES=0,1,2,3 AITER_LOG_LEVEL=WARNING \
torchrun --standalone --nproc_per_node=4 \
  -m tests.attentions.deepseek_v41.validate_vision_runtime \
  --chunk-size 128 --output /tmp/v41-vision.json
```

It uses compilation level 0, FP4 main KV and the FP8 index plane. Add
`--graph` for FULL decode capture; image prefill remains eager. The checkpoint
interval is aligned to the prefix hash block, and the preemption scenario
waits until a checkpoint has been stored. This is a lifecycle and color smoke
test, not a general visual benchmark.
