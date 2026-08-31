# MiniMax-H3 Usage Guide (video + audio generation)

[MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3) is a unified
text/image/video/audio → **video + audio** diffusion model. It is the first
diffusion model supported by ATOM, and it runs on `atom/diffusion/` — a
subsystem separate from `atom/model_engine/`, because a denoise loop shares
almost nothing with autoregressive decoding (no KV cache, no continuous
batching, four heterogeneous networks instead of one, and sequence parallelism
across a *single* request).

Output contract: H.264 1344×768 @24fps plus one AAC stereo 32 kHz track, muxed
into a single MP4. **The audio track is half the model** — a video-only result
is not a valid H3 result.

| Hardware | Task | Partition | Parallelism | Validated |
| --- | --- | --- | --- | --- |
| MI308X (gfx942) | t2va | FL2VA | Ulysses-4 | ✅ 41.48 dB / SSIM 0.963 |
| MI308X (gfx942) | fl2va | FL2VA | Ulysses-4 | ✅ 40.66 dB / SSIM 0.970 |
| MI308X (gfx942) | ref2va | Ref2VA | Ulysses-4 | ✅ 41.52 dB / SSIM 0.969 |
| MI355X (gfx950) | t2va | FL2VA | Ulysses-8 | ✅ single-forward cos 1.0000 |

PSNR/SSIM are measured against the upstream sglang reference on the same box at
the same seed. See [Validation](#validation) for exactly what that number does
and does not cover.

## Layout

`atom/diffusion/` is **model-major**: the framework sits at the top level and
everything for one model lives in one package, so adding a model is a new
directory plus one line in `registry.py` rather than edits scattered across
`dits/`, `vaes/`, `encoders/` and `schedulers/`.

The pipeline is chosen the way the LLM side chooses a model class: from the
checkpoint. `registry.py` maps the `_class_name` in `model_index.json` to a
pipeline, so `--model /path/to/FL2VA` is enough. `--pipeline <dotted.path>`
overrides it, which is what an out-of-tree pipeline needs.

```
atom/diffusion/
  config.py request.py pipeline.py attention.py ulysses.py mux.py registry.py
  engine/        job scheduler, ZMQ workers, per-GPU runner
  entrypoints/   diffusion_server.py, video_api.py
  models/minimax_h3/
      arch.py          architecture config
      dit.py           the network
      components.py    both VAEs, text encoder, weight loading
      layout.py        geometry, packed sequence, patchify, initial latents
      conditioning.py  keyframes, references, noise aug, presentation
      denoise.py       the loop and its rectified-flow sampler
      pipeline.py      the 8 stages and the pipeline
```

Seven files per model, grouped by role. Deliberately *not* split further: the
generic-looking pieces (weight sharding, patchify, seeded noise,
rectified-flow Euler) each sit alongside H3 specifics in the same module, and
promoting them to a shared layer today would mean inventing an API with one
caller. A component graduates when a *second* model uses it, not in
anticipation.

## Preparing environment

```bash
docker pull rocm/atom-dev:latest
```

Everything below runs inside the container.

Weights are two independent ~135 GiB partitions. `t2va` and `fl2va` are served
by **FL2VA**; `ref2va` needs **Ref2VA**. They are separate replicas on separate
ports, not two branches of one load.

```bash
export HF_HOME=/data/hf_home
hf download MiniMaxAI/MiniMax-H3 --include "FL2VA/*" "tokenizer/*" "processor/*" \
  "scheduler/*" "audio_scheduler/*" "*.json" --local-dir /data/models/MiniMax-H3
# ref2va only:
hf download MiniMaxAI/MiniMax-H3 --include "Ref2VA/*" --local-dir /data/models/MiniMax-H3
```

## Launching the server

```bash
export AITER_LOG_LEVEL=WARNING

python -m atom.diffusion.entrypoints.diffusion_server \
  --model /data/models/MiniMax-H3 --model-variant FL2VA \
  --num-gpus 4 \
  --ulysses-degree 4 \
  --output-dir /data/outputs \
  --port 30010
```

Startup loads ~144 GiB and takes roughly 4–5 minutes; the server does not bind
until every rank reports ready, so a successful `/health` means the model is
actually resident.

The replica runs **one throwaway denoise step before reporting ready**. The
first DiT forward in a fresh process costs far more than the rest — on gfx950
at Ulysses-8, the token refiner's first attention forward alone is 8.9 s of
aiter kernel JIT — and paying it during a multi-minute load is free where
paying it inside the first generation is not. `--no-warmup` skips it.

| first request, gfx950 U-8 | `--no-warmup` | default |
| --- | ---: | ---: |
| warmup at load | — | 10.8 s |
| rope + `refine_prompt_embeds` | 8.9 s | 0.0 s |
| denoise step 1 | 1,373 ms | **552 ms** |
| steps 2–6 | ~552 ms | ~552 ms |
| **first denoise block** | **13.0 s** | **3.4 s** |

Warmup uses the released 1344×768 / 5.17 s geometry. A request at another
canvas still benefits — aiter JIT, allocator growth and RCCL setup are
shape-independent — but re-pays the shape-dependent part of GEMM selection.
A warmup failure is logged, not fatal: the same work reruns on the first
request, where the error is attributable to a job.

**Ulysses degree must divide both the head count (56) and the 64-aligned packed
sequence.** 1, 2, 4 and 8 all work. 7 divides the heads but not the sequence
and is rejected at config time rather than at the first all-to-all.

`--model-variant` names the partition under `--model`; `--model /path/FL2VA`
with no variant is equivalent. For `ref2va`, use `--model-variant Ref2VA` and a
different port -- the two partitions are separate replicas.

Install the extras once: `pip install -e ".[diffusion]"` (PyAV for the mux,
Pillow for image conditioning, torchaudio for reference audio).

## Generating

`task` is required and is not inferred from the conditions.

### t2va

```bash
curl -X POST http://127.0.0.1:30010/v1/videos \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "At night, three cats march in playing tiny brass instruments.",
    "task": "t2va",
    "seconds": 5.166667,
    "seed": 1101,
    "num_inference_steps": 50,
    "target": {"height": 768, "width": 1344, "fps": 24}
  }'
```

Returns `202` with a job id immediately — a generation takes minutes, so the
API is asynchronous by contract.

```bash
curl http://127.0.0.1:30010/v1/videos/<id>              # status + progress
curl -o out.mp4 http://127.0.0.1:30010/v1/videos/<id>/content
curl -X DELETE http://127.0.0.1:30010/v1/videos/<id>    # abort
```

`GET .../content` returns **409** (not 404) while the job is still running: the
job exists, the caller polled early, and that is a different fix from a bad id.
A full queue returns **429** — the scheduler rejects rather than queueing, since
with 4-minute jobs an unbounded queue is an unbounded invisible wait.

### fl2va (first/last-frame conditioning)

```bash
  "task": "fl2va",
  "conditions": [
    {"type": "image", "uri": "file:///data/keyframe.png", "frame_index": 0}
  ]
```

`frame_index` may be `0` (first), `-1` (last), or both images in that order.

The anchor conditions the model **twice** and both paths are load-bearing: the
Qwen3-VL vision tower folds it into the prompt sequence (1,010 of 1,029 tokens
for a 1344×768 anchor) and the video VAE encodes it into 1,008 packed rows.

### ref2va (reference image / audio / video)

```bash
  "task": "ref2va",
  "conditions": [
    {"type": "image", "uri": "file:///data/subject.png"},
    {"type": "audio", "uri": "file:///data/track.wav"}
  ]
```

References do **not** bind the target canvas — unlike an fl2va keyframe, a
reference image goes to its own 2048px short edge. Set `target` explicitly.

## Offline use

```python
from atom.diffusion.config import DiffusionConfig
from atom.diffusion.engine import DiffusionEngine
from atom.diffusion.request import DiffusionJob

config = DiffusionConfig(
    model_path="/data/models/MiniMax-H3/FL2VA",
    pipeline_class="atom.diffusion.models.minimax_h3.pipeline.MiniMaxH3Pipeline",
    num_gpus=4,
    ulysses_degree=4,
    output_dir="/data/outputs",
)
with DiffusionEngine(config) as engine:
    job = engine.submit(DiffusionJob(prompt="...", task="t2va", seed=1101))
    print(engine.wait(job.job_id).output_path)
```

## Attention backend

`--attn-backend` selects the packed varlen FMHA kernel, and the choice is a real
trade rather than a fallback ladder:

| backend | throughput | use |
| --- | --- | --- |
| `asm` (default) | 124.0 TFLOP/s | fastest on gfx942 |
| `triton` | 99.0 TFLOP/s | **reproduces the sglang reference bit-for-bit** |
| `sdpa` | — | CPU fallback and numerics anchor |

The kernels agree to ~1e-5 cosine per call, which is ordinary bf16 spread — but
over 50 denoise steps that compounds into a *different but equally valid*
sample. Nothing here claims one is more accurate. **Anyone diffing pixels
against sglang must select `triton`** or they will chase a phantom.

Do not reintroduce upstream's `USE_AITER_GFX942` Triton fallback as a default:
on MI308X the ASM varlen path matches the tuned fixed-length kernel (124.0 vs
123.9 TFLOP/s), so that workaround costs ~20% here for nothing.

## Memory

Measured on MI308X (192 GB/GPU), 1344×768 × 5.17 s, 50 steps, Ulysses-4:

| | |
| --- | --- |
| resident per rank | 66 GB (DiT) |
| rank 0, additionally | 10.4 GB video VAE + 0.6 GB audio VAE |
| peak, rank 0 | ~171 GB |
| denoise | ~395 s |
| decode + mux | ~28 s (Ulysses-4) / ~4 s (Ulysses-8) |

The video VAE decodes in **bf16**. It is transformer-based rather than
convolutional -- 39.7% of decode is `addmm`, 0.0% is convolution -- so the
checkpoint's fp32 weights make decode GEMM-bound for no benefit: measured 88.4 s
fp32 against 24.4 s bf16, agreeing to 51.4 dB. End-to-end parity is unchanged
(41.47 dB vs 41.48). Encode still runs fp32, which is the reference's recipe.

The 50 GiB Qwen3-VL text encoder is **staged on the host** and uploaded only
for the encode it performs once per request. Not an optimisation -- it is what
makes the replica fit: the first served request died with 182 GiB allocated
before the encoder was moved off the resident set.

Weights are read-only, so the host copy stays authoritative and releasing just
drops the device copy -- no copy back. With the host side pinned at load, the
per-request cost is **1.0 s** rather than the 12.7 s a naive round trip costs.

Video decode is **collective**. The checkpoint's bundled VAE already
implements tiled decode and the rank sharding for it, but seeds a
single-process parallel state, so every tile ran on rank 0. Pointing it at the
sequence-parallel group takes decode from 27.5 s to **4.1 s** at Ulysses-8,
output pixel-identical. The video VAE therefore lives on every rank (~5 GB in
bf16); the encoder and audio VAE stay on rank 0.

Which rank holds what is a declaration, not a code path -- `component_placement`
on the pipeline class, read by `ComposedPipeline` to guard `register_component`
and to derive `verify_components`:

```python
component_placement = {
    "transformer":  ComponentPlacement.ALL_RANKS,
    "video_vae":    ComponentPlacement.ALL_RANKS,   # tiled decode is collective
    "audio_vae":    ComponentPlacement.MAIN_RANK,
    "text_encoder": ComponentPlacement.MAIN_RANK,
}
```

Each rank then asserts exactly its own set at load: a video VAE that failed to
load on rank 3 is an error there rather than a hang in decode. This is a
different axis from `host_staged_components`, which is host-or-device; the text
encoder is both rank-0-only and staged.

Attention skips the trailing alignment padding. The ASM kernel's grid is
`(heads, num_segments, ceil(max_seqlen / 256))` and is sized from `max_seqlen`
rather than from each segment, so a 24-row padding segment gets a whole plane
of 2,072 workgroups of which one has work. Dropping those rows halves the grid:
93.0 -> 80.9 ms per layer, 30 s off a denoise, output pixel-identical. Triton
shows no benefit (104.78 vs 104.66), so this is ASM-only.

### Measured cost, t2va at Ulysses-4

| | |
|---|---:|
| text encode | ~23 s |
| encoder staging | 1.0 s |
| denoise | 394.6 s |
| decode | 27.7 s |
| **total** | **~446 s** |

### On MI355X (gfx950), Ulysses-8

Steady-state denoise is **563 ms/step**, of which the DiT forward is
essentially all: a kernel profile of one forward puts attention at 272.6 ms,
GEMM at 171.1 ms, norm at 23.9 ms and elementwise/cat at 55.9 ms. The CPU
queues the whole 50-layer forward in 29 ms and then waits 531 ms for the GPU,
so the loop is GPU-bound -- there is no launch overhead to reclaim.

The elementwise share is where a fused AdaLN-modulation and QKV-pack kernel
would pay off, worth roughly 13 ms/step. Small next to the warmup above, which
is why it is not the top priority.

Two profiler traps worth not re-learning: `key_averages()` reports the aten
scope *and* the kernel it launched, each with the same device time, so summing
both double-counts; and host API entries (`hipThreadExchangeStreamCaptureMode`
and friends) carry attributed device time without being GPU work.

## Validation

```bash
python -m pytest tests/diffusion/   # 235 tests, CPU only, no AITER needed
```

Against the sglang reference on the same box, same seed, `--attn-backend triton`:

| layer | evidence |
| --- | --- |
| DiT forward (steps 0 and 45) | max_rel_err **0.000e+00** |
| weight loading | 535/535 tensors |
| packed layout, all three tasks | value-exact, position grid maxdiff **0.000e+00** |
| 45 steps of the full denoise loop | max_rel_err **0.000e+00** |
| fl2va keyframe conditioning rows | mean \|diff\| **5.4e-7** |
| decode + mux | **40.7–41.5 dB**, SSIM 0.963–0.970 |

Those runs seed the loop from the reference's captured step-0 state, so RNG and
text-encoding semantics are held fixed and what is measured is ATOM's DiT,
sampler, conditioning layout, decode and mux.

A **fully self-contained** run — ATOM's own text encoder and its own seeded
noise — produces a valid sample at the same contract but a different
trajectory: 24.6 dB against the reference, in the same band as any two runs
whose latents genuinely differ. Two known contributors: ATOM uses transformers'
Qwen3-VL while the reference vendors its own (refined-embedding cosine 0.9913,
traced to the vision tower — inputs, M-RoPE positions and the token refiner are
all exact), and ATOM's seed→noise mapping has not been verified bit-for-bit
against upstream's. Neither is a correctness defect; both are open items.

## Known gotchas on ROCm

* **`tensor.is_cuda` is True for HIP tensors.** Upstream gates three separate
  CUDA-only JIT kernels on it (QK-Norm, RoPE, and the VAE's
  `apply_rotary_pos_emb_qk`); each fails to build under hipcc and the failure is
  fatal even though the correct eager fallback is the next statement. Grep for
  `is_cuda` in anything ported from sglang.
* Dispatch attention on `q.device.type`, **not** on whether `aiter` imports —
  aiter imports fine in a CPU-only process and then dies inside the kernel.
* The video VAE emits ImageNet-**normalized** pixels. Decode must finish with
  the checkpoint's `transform_rev` and clamp to [0, 1]. Skipping it is invisible
  to every structural check and costs ~22 dB.
* Decode via `decode_temporal()`, not `decode()`: only the former honours
  `clip_length=17` / `token_drop=3` and yields the 17n+5 frame lattice.
