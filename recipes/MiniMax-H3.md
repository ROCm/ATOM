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

PSNR/SSIM are measured against the upstream sglang reference on the same box at
the same seed. See [Validation](#validation) for exactly what that number does
and does not cover.

## Layout

`atom/diffusion/` is **model-major**: the framework sits at the top level and
everything for one model lives in one package, so adding a model is a new
directory plus a `--pipeline` path rather than edits scattered across
`dits/`, `vaes/`, `encoders/` and `schedulers/`.

```
atom/diffusion/
  config.py request.py pipeline.py attention.py ulysses.py mux.py
  engine/        job scheduler, ZMQ workers, per-GPU runner
  entrypoints/   diffusion_server.py, video_api.py
  models/minimax_h3/
      arch.py dit.py vae.py text_encoder.py scheduler.py loader.py
      pipeline.py                       the 8 stages and the pipeline
      geometry.py packed_sequence.py packed_tokens.py latent_prep.py
      keyframe.py condition_noise.py reference_encoding.py presentation.py
      denoise.py
```

A component graduates out of a model package into the shared layer when a
*second* model uses it, not in anticipation.

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
| denoise | ~450 s |
| decode + mux | ~28 s |

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

### Against the sglang reference, same box, t2va at Ulysses-4

| | reference | ATOM |
|---|---:|---:|
| text encode | ~23 s | ~23 s |
| encoder staging | — | 1.0 s |
| denoise | ~425 s | **424.7 s** |
| decode | 17.3 s | 27.7 s |
| **total** | **465.5 s** | **~476 s** |

Denoise is even. The reference is locked to the Triton attention kernel on
gfx942 by an upstream workaround for an ASM hang that does not reproduce on
this aiter build (14 configurations tested), so ATOM runs the faster ASM path
and gives that time back on decode, where sglang uses its own vendored VAE.

## Validation

```bash
python -m pytest tests/test_diffusion_*.py   # 247 tests, CPU only, no AITER needed
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
