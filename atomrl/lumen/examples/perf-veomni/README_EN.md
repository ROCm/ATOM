# VeOmni Qwen-Image / Wan2.1 training performance: configuration tuning + Lumen

> 中文版：[README.md](README.md)

A three-level performance ladder for two diffusion-model training workloads on VeOmni: **the example as
shipped → the best reachable by configuration alone → the best with Lumen optimizations on top of that
configuration**. The configuration levels change only VeOmni command-line arguments; reusable Lumen levels
are enabled through `LumenConfig`, with only VeOmni-specific workarounds left in `veomni_patches/`.
Neither VeOmni's nor AITER's installed source is modified. The current VAE
convolution route lives in `atomrl.lumen.ops.conv`.

> The performance tables below were recorded with the former experimental
> FlyDSL route. They are historical evidence, not validation numbers for the
> current in-package convolution, which must be re-measured.

---

## 1. Conclusion

| Workload | As shipped | Best configuration | Best with Lumen | From configuration | From Lumen, on top | Total |
|---|---:|---:|---:|---:|---:|---:|
| Qwen-Image 1024² online training | 1.855 s | 0.971 s | **0.680 s** | 1.91× | 1.43× | **2.73×** |
| Wan2.1 81×368×544 online training | 3.569 s | 2.654 s | **1.160 s** | 1.34× | 2.29× | **3.08×** |
| Wan2.1 video embedding (`offline_embedding`) | 2.027 s | 1.493 s | **0.340 s** | 1.36× | 4.39× | **5.96×** |

Seconds per step (8 GPUs, global batch 8, 10 steps; median with step 1 dropped; 3 repeats run round by round).

- **Configuration** gains come from two things. First, restoring the settings where the example yaml departs from
  VeOmni's defaults: `empty_cache_steps` 1 → 500, and for Wan `num_workers` 0 → 2 with `pin_memory` on.
  Second, trading memory for time: gradient checkpointing off, plus reshard off for Qwen.
- **Lumen** gains come mostly from two places: the attention backward moving to aiter's fused kernel
  (Qwen −12.5%, Wan −18.8%), and the VAE running in BF16 on the FlyDSL convolution (−44% on Wan training,
  −77% on video embedding).
- Switching to "embed once, then train from the embeddings" lowers the training step further: Qwen-Image
  **0.613 s** (−9.9% against the best online step), Wan2.1 **0.824 s** (−29.0%), paying back in about one
  epoch (§2.6).

Scope: 8× AMD Instinct MI350X (gfx950), ROCm 7.14 / torch 2.12, VeOmni `573848a`, FSDP2 full-parameter SFT,
Qwen-Image 20B DiT at 1024², Wan2.1-T2V-1.3B at 81 frames of 368×544.

---

## 2. Data

### 2.1 The ladder

Each level adds **exactly one** change to the level above it, so "vs previous" is that change's increment on top
of everything before it. A and B levels change VeOmni configuration only; L levels keep the best configuration
and add one `LUMEN_PATCH` entry at a time. p is a Mann-Whitney rank-sum test against the previous level (per-step
samples pooled across repeats); spread = range of the three per-repeat medians / median. Level numbers are shared
by both models, so a level that does not apply leaves a gap.

**Qwen-Image 1024² online training**

| Level | Change | Step | Spread | vs previous | p | Peak memory |
|---|---|---:|---:|---:|---:|---:|
| A0 | Example as shipped (`empty_cache_steps=1`, gradient checkpointing on) | 1.855 s | 0.4% | — | — | 38.8 GB |
| B1 | `--train.empty_cache_steps 500` | 1.205 s | 1.0% | −35.0% | 3e-9 | 38.8 GB |
| B2 | `--train.gradient_checkpointing.enable false` | 1.045 s | 0.8% | −13.3% | 3e-9 | 72.2 GB |
| B3 | `--train.accelerator.fsdp_config.reshard_after_forward false` | **0.971 s** | 0.6% | −7.1% | 8e-7 | 108.9 GB |
| L1 | `vae_bf16`: the VAE runs in BF16 | 0.889 s | 0.6% | −8.4% | 5e-7 | 108.7 GB |
| L2 | `vae_conv`: VAE convolutions on aiter FlyDSL | 0.881 s | 1.4% | −0.9%, within spread, not counted | 7e-4 | 108.8 GB |
| L4 | `sdpa_efficient`: attention backward on aiter's fused `fmha_bwd` | 0.771 s | 3.0% | −12.5% | 3e-8 | 108.8 GB |
| L5 | `attn_triton_fwd`: DiT attention forward on aiter Triton | 0.742 s | 0.8% | −3.8% | 4e-7 | 108.8 GB |
| L6 | `rmsnorm_fuse`: RMSNorm forward fused into one kernel | 0.702 s | 2.2% | −5.3% | 4e-7 | 100.4 GB |
| L7 | `local_adamw`: AdamW on the local shards | **0.680 s** | 2.7% | −3.1% | 3e-6 | 100.4 GB |

**Wan2.1 online training**

| Level | Change | Step | Spread | vs previous | p | Peak memory |
|---|---|---:|---:|---:|---:|---:|
| A0 | Example as shipped (also `num_workers 0`, `pin_memory false`) | 3.569 s | 0.5% | — | — | 21.2 GB |
| B1 | `--train.empty_cache_steps 500` | 3.101 s | 0.3% | −13.1% | 4e-8 | 21.2 GB |
| B2 | `--train.gradient_checkpointing.enable false` | 2.827 s | 0.5% | −8.8% | 5e-8 | 70.9 GB |
| B4 | `--data.dataloader.num_workers 2 --data.dataloader.pin_memory true` | **2.654 s** | 0.4% | −6.1% | 5e-8 | 70.9 GB |
| L1 | `vae_bf16` | 1.633 s | 0.3% | −38.5% | 3e-9 | 70.7 GB |
| L2 | `vae_conv_video`: convolutions including T>1 and the feature cache | 1.535 s | 1.0% | −6.0% | 3e-9 | 70.9 GB |
| L3 | `conv_pad_in_kernel`: causal-conv padding handed to the kernel | 1.495 s | 0.4% | −2.6% | 3e-9 | 70.9 GB |
| L4 | `sdpa_efficient` | 1.215 s | 0.9% | −18.8% | 8e-7 | 70.9 GB |
| L5 | `attn_triton_fwd` | 1.170 s | 1.0% | −3.7% | 6e-6 | 70.9 GB |
| L7 | `local_adamw` | **1.160 s** | 1.5% | −0.8%, within spread, not counted | 4e-3 | 70.9 GB |

**Wan2.1 video embedding** (`offline_embedding` runs only the VAE and the text encoder: no DiT, no backward, no optimizer)

| Level | Change | Step | Spread | vs previous | p | Peak memory |
|---|---|---:|---:|---:|---:|---:|
| A0 | Example as shipped | 2.027 s | 0.3% | — | — | 14.7 GB |
| B1 | `--train.empty_cache_steps 500` | 1.677 s | 0.3% | −17.2% | 7e-7 | 14.7 GB |
| B4 | data loading `num_workers 2`, `pin_memory` | **1.493 s** | 0.2% | −11.0% | 3e-9 | 14.7 GB |
| L1 | `vae_bf16` | 0.475 s | 0.5% | −68.2% | 3e-9 | 12.8 GB |
| L2 | `vae_conv_video` | 0.376 s | 4.3% | −20.7% | 5e-8 | 12.9 GB |
| L3 | `conv_pad_in_kernel` | **0.340 s** | 5.3% | −9.7% | 3e-9 | 12.8 GB |

Notes:

- Levels skipped on purpose:
  - B4 for Qwen: data loading is not Qwen's bottleneck.
  - B3 for Wan: reshard off was measured earlier to have no effect.
  - L6 for Wan: its DiT uses `torch.nn.RMSNorm`, which is already a fused kernel.
  - L3 for Qwen: every Qwen convolution takes the exact T=1 → conv2d rewrite, so there is no temporal padding to hand over.
  - B2 and L4–L7 for the embedding task: it has no backward and no optimizer.
- **Two increments are not counted as gains**: Qwen L2 (−0.9%) and Wan L7 (−0.8%) are both within spread.
  After L1, Qwen's VAE convolution has only about 13 ms of GPU time per step left (§2.2); FlyDSL pays off on
  Wan's video VAE.
- Step 1 (FlyDSL JIT, `torch.compile`, autotuning), reported separately:
  - Qwen: 32–34 s at the configuration levels, 36–39 s at the Lumen levels.
  - Wan: 29–32 s at every level.
  - Embedding task: 17–20 s.

  All were measured with a warm FlyDSL JIT disk cache. On a cold cache Wan's first step needs several more minutes to compile.
- Memory: gradient checkpointing off and reshard off trade memory for time (Qwen 38.8 → 108.9 GB, Wan 21.2 →
  70.9 GB). The RMSNorm fusion lowers the peak by 8.4 GB, because the compiled forward saves fewer intermediates
  for the backward.
- GPU 0 had been set to `perf_determinism` by another user. Sampled during a run, its busy-time mean clock was
  2180 MHz against 2155–2192 MHz on the other seven, so it was not throttled.

### 2.2 Kernel-level evidence for each change

These figures come from a separate profiler run of each level (steps 4–5, no Python stacks), in GPU ms per step on
rank 0, from `kernel_diff.py` between adjacent levels. The profiler lengthens the step slightly, so use these for
which kernel replaced which; step times are in §2.1.

| Change | Model | Before → after |
|---|---|---|
| B1 `empty_cache_steps 500` | Qwen | A0 has a ~660 ms GPU-idle pit every step (220–260 ms of it in `hipFree`); by B3 the largest pit is 19 ms |
| | Wan | A0 has a 450–465 ms pit every step; by B4 (with the loader change) the largest is 8 ms |
| B2 gradient checkpointing off | Qwen | GEMM calls 600 → 300, `attn_fwd` 149 → 89: the forward recompute is gone |
| B3 reshard off | Qwen | RCCL 312 → 203 ms (A0→B3): no second all-gather after the forward |
| L1 `vae_bf16` | Qwen | CK FP32 `grouped_conv` 20 calls, 79.3 ms → CK BF16 convolutions, ~12.8 ms |
| | Wan | CK FP32 `grouped_conv` 460 calls, 1078.3 ms → CK BF16 convolutions, 188.9 ms; 73.7 ms of FP32 transposes gone |
| L2 FlyDSL convolution | Qwen | CK BF16 convolutions ~12.8 ms → `conv3d_implicit` 22 calls, 3.95 ms, + transpose 0.69 ms |
| | Wan | CK BF16 convolutions 460 calls, 187.9 ms, + ~48 ms transposes → `conv3d_implicit` 525 calls, 146.5 ms, + transpose 19.2 ms |
| L3 padding in the kernel | Wan | `aten::pad` copies and fills 1131 calls, 40.7 ms → 295 calls, 6.6 ms; `conv3d_implicit` 146.5 → 140.7 ms (tuned table for these shapes, §2.5) |
| L4 `sdpa_efficient` | Qwen | AOTriton `bwd_kernel_dk_dv` + `bwd_kernel_dq` 202.8 ms → aiter `fmha_bwd_hd128_bf16_a32` + postprocess 77.2 ms |
| | Wan | the same, 607.7 ms → 313.9 ms |
| L5 `attn_triton_fwd` | Qwen | `attn_fwd` 61 calls, 72.2 ms → 1 call, 7.8 ms (the VAE's); aiter Triton `_attn_fwd` 60 calls, 23.5 ms |
| | Wan | `attn_fwd` 81 calls, 170.3 ms → 21 calls, 20.4 ms; Triton `_attn_fwd` 60 calls, 87.1 ms |
| L6 `rmsnorm_fuse` | Qwen | 120–240 fewer elementwise and reduction launches per step, one new fused Triton kernel (120 calls, 2.3 ms); GPU total −48.5 ms |
| L7 `local_adamw` | Qwen | no kernel change (host side); optimizer-phase host time 44.5 → 6.4 ms |
| | Wan | optimizer-phase host time 20.1 → 2.8 ms |

At the end of each run the example's entry point also prints what the patches actually did. This is evidence
separate from the "patch applied" message. At the best Lumen configuration, per step:

- Qwen: 22 aiter convolution table lookups, all exact hits; 60 SDPA calls routed to the Triton forward; 120 RMSNorm calls on the fused path.
- Wan: 315 of 525 convolutions hit the tuned table exactly. The other 210 fall on six shapes that did not clear the 3% bar and so stay on the heuristic.

### 2.3 Numerics

Three grades:

| Grade | Change | Evidence |
|---|---|---|
| **Bit-identical** | B1, B2, B3 | Qwen loss and grad_norm equal to A0's bit for bit on 10/10 steps |
| | B4 data loading | Wan: 80/80 samples on 8 ranks receive identical bytes in identical order (`LUMEN_BATCH_HASH=1`) |
| | L3 `conv_pad_in_kernel` | Wan VAE latents bit-identical to the pre-padded path at 81×368×544; L2 and L3 losses equal on 10/10 steps within the same round |
| | L7 `local_adamw` | on the deterministic B3, loss and grad_norm equal to stock AdamW's on 10/10 steps |
| **Changes the numbers, deterministically** | L1 `vae_bf16` + L2 FlyDSL convolution | SNR against an FP32 encode: Qwen 50.4 dB (stock BF16 50.3 dB), Wan 50.7 dB (stock BF16 50.4 dB); each level's 3 repeats bit-identical |
| | L5 `attn_triton_fwd` | out and dQ/dK/dV 52.9/52.5/52.6/52.6 dB against FP32, the same as the efficient backend; LSE within 1.9e-6 |
| | L6 `rmsnorm_fuse` | 55.6 dB against FP32 vs 52.6 dB stock (more accurate: the intermediate stays FP32) |
| **Nondeterministic** | L4 `sdpa_efficient` | the fused backward accumulates dQ with FP32 atomics (`a32` in the kernel name); Qwen repeats of one configuration differ by up to 0.66% in the loss, against 0 at every level from A0 to L2 |

Wan's loss is not reproducible to begin with: the example as shipped differs by up to 1.4% between repeats. Wan's
numerics are therefore judged only on latent comparisons, SNR and input hashes, never on the loss.

### 2.4 GPU time by category: where the gains come from

These figures come from the profiler runs (rank 0, ms per step). `trace_report.py` files CK convolutions under GEMM
and aiter's `fmha_bwd` under other; the table below uses the corrected categories.

| Category | Qwen A0 | Qwen B3 | Qwen L7 | Wan A0 | Wan B4 | Wan L7 | Embed B4 | Embed L3 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Step (under the profiler) | 1921 | 1001 | 746 | 3384 | 2650 | 1157 | 1500 | 377 |
| GPU busy | 57% | 87% | 83% | 85% | 98% | 96% | 98% | 88% |
| attention | 322 | 276 | 110 | 932 | 782 | 423 | 34 | 21 |
| VAE convolution | 79 | 79 | 5 | 1082 | 1082 | 160 | 1082 | 159 |
| GEMM | 330 | 251 | 252 | 240 | 193 | 154 | 51 | 14 |
| communication (RCCL) | 312 | 203 | 197 | 44 | 31 | 44 | 0 | 9 |
| elementwise and other | 301 | 215 | 166 | 509 | 448 | 277 | 279 | 106 |
| optimizer | 30 | 26 | 28 | 3 | 3 | 3 | — | — |

- **The two models' bottlenecks are almost complementary.** Qwen has a large DiT (20B) and a VAE that encodes
  one image, so its time goes to attention, GEMM and communication. Wan has a small DiT (1.3B) and a VAE that
  encodes 81 frames of video, so once the configuration is fixed the VAE convolution is its largest item (42%).
  That is why the same VAE work (L1–L3) is worth −44% on Wan and only −9% on Qwen.
- B1 shows up in the trace as an idle pit. At A0, Qwen has a ~660 ms GPU-idle pit every step, 89% of all idle
  time, and 220–260 ms of it is spent in `hipFree`: with `empty_cache_steps=1` every step hands the cached
  memory back to the driver block by block. From A0 to B3, GPU busy goes from 57% to 87%.
- What remains at the best Lumen configuration: for Qwen, communication (25% of GPU time, competing with compute
  for CUs); for Wan, attention (38%, of which `fmha_bwd` is 311 ms).

### 2.5 Convolution: Lumen's in-package integration

`LumenConfig(aiter_conv=True)` patches supported `nn.Conv2d` and `nn.Conv3d`
modules through `atomrl.lumen.ops.conv`. Lumen owns the autograd-safe routing;
the forward calls AITER's public convolution APIs and the backward uses PyTorch
gradient operators. The example does not copy files into AITER, alter
`AITER_CONFIGS`, or monkey-patch AITER modules.

### 2.6 The offline-embedding workflow (optional)

| | Qwen-Image | Wan2.1 |
|---|---:|---:|
| Online training (best with Lumen) | 0.680 s | 1.160 s |
| One-time embedding, 8 samples per step (VAE patches as in the best Lumen level) | 0.059 s | 0.340 s |
| Training from the embeddings (`offline_training`) | **0.613 s** (−9.9%, p 3e-9) | **0.824 s** (−29.0%, p 4e-8) |
| Break-even | 0.88 epoch | 1.01 epoch |

Break-even = embedding time / (online step − offline step): past about one epoch, embedding first and then
training is faster. This assumes training needs no online data augmentation. `offline_training` redistributes the
samples to different ranks and steps. Measured on the 80 samples, the stored embeddings are byte-identical to the
online ones, only in a different order, so the per-step loss cannot be compared with online training step by
step; the two are equivalent statistically. Qwen's `offline_training` requires `qwen_offline_fix`; without it the
run crashes on its first step (a VeOmni `573848a` defect, §3.7).

---

## 3. Runbook

Every command runs inside the container, with the default layout of `env.sh`: one host directory mounted at `/work`.

### 3.1 Container

This needs one node with 8× MI350X / MI355X (gfx950), the ROCm driver, and about 200 GB of space for the two
models and the data.

```bash
export ROOT=/path/to/fast-storage/perf-veomni
mkdir -p "$ROOT"
docker pull amdagi/veomni:rocm7.14_torch2.12_py3.12
docker run -d --name perf-veomni \
  --device /dev/kfd --device /dev/dri --group-add video --group-add render \
  --ipc host --shm-size 32g --cap-add SYS_PTRACE \
  --security-opt seccomp=unconfined --security-opt label=disable --ulimit memlock=-1:-1 \
  -e HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 -e CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  -e PYTORCH_ROCM_ARCH=gfx950 \
  -v "$ROOT":/work -w /work \
  amdagi/veomni:rocm7.14_torch2.12_py3.12 sleep infinity

dx() { docker exec -w /work perf-veomni bash -lc "$*"; }
export EX=/work/Lumen/examples/perf-veomni
```

Set `HIP_VISIBLE_DEVICES` and `CUDA_VISIBLE_DEVICES` together, single-GPU scripts included. If they disagree,
the run crashes.

### 3.2 Code and environment

```bash
dx 'git clone https://github.com/ROCm/ATOM.git /work/ATOM &&
    git -C /work/ATOM checkout main &&
    ln -s /work/ATOM/atomrl/lumen /work/Lumen'
dx 'git clone https://github.com/ByteDance-Seed/VeOmni.git /work/VeOmni &&
    git -C /work/VeOmni checkout 573848a00fcd7329c2411346c6f4a983e9f67e3f'
dx ". $EX/env.sh && bash \$EXAMPLE_DIR/setup_env.sh"
```

`setup_env.sh` installs VeOmni with `--no-deps` and verifies the Lumen
convolution entry point.

It should end with:

```text
Lumen convolution: available (must be available)
SETUP OK
```

Do not run `uv sync` or `pip install -e '.[gpu]'`, and do not install any torch, triton, `cu12`/`cu13` or
`nvidia-*` package.

### 3.3 Models and data

```bash
# Qwen-Image, and 80 samples (10 steps × 8 ranks)
dx ". $EX/env.sh && hf download Qwen/Qwen-Image --local-dir \$QWEN_IMAGE_DIR"
dx ". $EX/env.sh && python3 \$EXAMPLE_DIR/make_data.py 80 \$DATA_DIR/train_80.jsonl"

# Wan2.1-T2V-1.3B, and the first 400 clips of the public Tom-and-Jerry dataset
dx ". $EX/env.sh && hf download Wan-AI/Wan2.1-T2V-1.3B-Diffusers --local-dir \$WAN_DIR"
dx '. /work/Lumen/examples/perf-veomni/env.sh && RAW=$WORK/data/tom-and-jerry && mkdir -p "$RAW" &&
    hf download Wild-Heart/Tom-and-Jerry-VideoGeneration-Dataset --repo-type dataset \
      --include captions.txt --local-dir "$RAW" &&
    hf download Wild-Heart/Tom-and-Jerry-VideoGeneration-Dataset --repo-type dataset \
      --include videos.txt --local-dir "$RAW" &&
    head -n 400 "$RAW/videos.txt" | xargs -n 25 hf download Wild-Heart/Tom-and-Jerry-VideoGeneration-Dataset \
      --repo-type dataset --local-dir "$RAW" --quiet'
```

VeOmni's converter needs `captions.txt` and `videos.txt` to line up one to one. Build a subset from the clips
that actually downloaded, then convert it to parquet:

```bash
docker exec -i -w /work perf-veomni python3 - <<'PY'
from pathlib import Path
src, dst = Path('/work/data/tom-and-jerry'), Path('/work/data/tom-and-jerry-subset')
dst.mkdir(parents=True, exist_ok=True)
if not (dst / 'videos').exists():
    (dst / 'videos').symlink_to(src / 'videos', target_is_directory=True)
pairs = zip(src.joinpath('videos.txt').read_text().splitlines(), src.joinpath('captions.txt').read_text().splitlines())
keep = [(v, c) for v, c in list(pairs)[:400] if (src / v).exists()]
dst.joinpath('videos.txt').write_text(''.join(f'{v}\n' for v, _ in keep))
dst.joinpath('captions.txt').write_text(''.join(f'{c}\n' for _, c in keep))
assert len(keep) >= 80, len(keep)
PY
dx ". $EX/env.sh && python3 \$VEOMNI_DIR/scripts/multimodal/convert_data/tom-and-jerry.py \
  --dataset_path \$WORK/data/tom-and-jerry-subset --output_dir \$WAN_DATA_DIR"
```

### 3.4 Single-GPU acceptance (before any 8-GPU run)

```bash
G="HIP_VISIBLE_DEVICES=0 CUDA_VISIBLE_DEVICES=0 PYTHONPATH=\$LUMEN_PYTHONPATH"
dx ". $EX/env.sh && $G python3 \$EXAMPLE_DIR/verify_aiter_conv.py --model qwen"
dx ". $EX/env.sh && $G python3 \$EXAMPLE_DIR/verify_aiter_conv.py --model wan --padk"
dx ". $EX/env.sh && $G python3 \$EXAMPLE_DIR/verify_sdpa_triton_fwd.py"
dx ". $EX/env.sh && $G python3 \$EXAMPLE_DIR/verify_rmsnorm_fuse.py"
dx ". $EX/env.sh && cd \$EXAMPLE_DIR && $G python3 -m pytest -q test_patches.py"
```

Expected:

- Patched latents at the same SNR against FP32 as stock BF16 (Qwen 50.4 vs 50.3 dB, Wan 50.7 vs 50.4 dB), and
  `bit-identical` latents between pad-in-kernel and pre-pad.
- For Wan with L3, 315/525 tuned-table hits per encode.
- 52.5–52.9 dB on both attention paths.
- The RMSNorm forward going from 8 kernels to 1, at 55.6 dB against 52.6 dB stock.
- All tests passing.

### 3.5 The ladder

```bash
dx ". $EX/env.sh && bash \$EXAMPLE_DIR/ladder.sh qwen"      # 10 levels × 3, ~70 min
dx ". $EX/env.sh && bash \$EXAMPLE_DIR/ladder.sh wan"       # 10 levels × 3, ~50 min
dx ". $EX/env.sh && bash \$EXAMPLE_DIR/ladder.sh wanemb"    # 6 levels × 3, ~25 min
dx ". $EX/env.sh && bash \$EXAMPLE_DIR/ladder.sh offline"   # optional workflow; run wanemb first
dx ". $EX/env.sh && python3 \$EXAMPLE_DIR/summarize.py --control pvq-A0 --chain -- \
      pvq-A0 pvq-B1 pvq-B2 pvq-B3 pvq-L1 pvq-L2 pvq-L4 pvq-L5 pvq-L6 pvq-L7"
```

`ladder.sh` runs repeats round by round: every level once, then every level again. Slow drift in the machine
therefore lands on all levels alike. When it finishes it prints the matching `summarize.py` command. `--chain`
compares each row with the row above it and adds the cumulative speed-up over A0. `DRY_RUN=1` prints the commands
without running them, and `ladder.sh qwen A0 B3 L7` runs only the levels listed. Do not edit the script while it
runs, because bash reads a script as it executes it.

### 3.6 The full command for any level

Every level is this command with a different `LUMEN_PATCH` and different trailing arguments. For Wan, use
`run_wan_10steps.sh` with `WAN_TASK=online_training` or `WAN_TASK=offline_embedding`:

```bash
dx ". $EX/env.sh && RUN_NAME=<name> RUN_SUFFIX=r1 LUMEN_PATCH=<patches> \
      bash \$EXAMPLE_DIR/run_10steps.sh custom <VeOmni arguments...>"
```

| Workload | Level | `LUMEN_PATCH` | VeOmni arguments |
|---|---|---|---|
| all | A0 | (empty) | (none) |
| all | B1 | (empty) | `--train.empty_cache_steps 500` |
| Qwen / Wan | B2 | (empty) | B1 + `--train.gradient_checkpointing.enable false` |
| Qwen | B3 = best configuration | (empty) | B2 + `--train.accelerator.fsdp_config.reshard_after_forward false` |
| Wan | B4 = best configuration | (empty) | B2 + `--data.dataloader.num_workers 2 --data.dataloader.pin_memory true` |
| Embedding | B4 = best configuration | (empty) | B1 + the two data-loading flags |
| Qwen | L7 = best with Lumen | `vae_bf16,vae_conv,sdpa_efficient,attn_triton_fwd,rmsnorm_fuse,local_adamw` | as B3 |
| Wan | L7 = best with Lumen | `vae_bf16,vae_conv_video,conv_pad_in_kernel,sdpa_efficient,attn_triton_fwd,local_adamw` | as B4 |
| Embedding | L3 = best with Lumen | `vae_bf16,vae_conv_video,conv_pad_in_kernel` | as B4 |

The levels in between add one `LUMEN_PATCH` entry at a time, in the order of §2.1. For the offline workflow,
generate the embeddings with `--train.training_task offline_embedding --data.offline_embedding_save_dir <dir>`, then
train with `--train.training_task offline_training --data.train_path <dir>`. Training loads no VAE, so it takes no
VAE patch; Qwen also needs `qwen_offline_fix`.

### 3.7 `LUMEN_PATCH` values

| Value | Effect | Code |
|---|---|---|
| `vae_bf16` | The frozen VAE runs in BF16; `encode()` returns FP32 latents | `train_dit_lumen.py` |
| `vae_conv` / `vae_conv_video` | VAE convolutions use Lumen's AITER-backed Conv2d/Conv3d route. The first is for images (exact T=1 → conv2d); the second also covers T>1 and the feature cache | `lumen_vae_conv.py`, `atomrl.lumen.ops.conv` |
| `conv_pad_in_kernel` | Causal convolutions no longer `F.pad` a full copy; the kernel applies the symmetric padding with masks | `lumen_vae_conv.py` |
| `sdpa_efficient` | SDPA's flash backend off; on ROCm the backward runs aiter's fused `fmha_bwd`. **Nondeterministic** | `atomrl.lumen.ops.attention.hf_patch` |
| `attn_triton_fwd` | DiT attention only (no mask, non-causal, BF16, head_dim 128) gets aiter's Triton forward, with the efficient backward kept; requires `sdpa_efficient` | `atomrl.lumen.ops.attention.hf_patch` |
| `rmsnorm_fuse` | diffusers `RMSNorm` forward fused with `torch.compile`; weighted RMSNorm only, LayerNorm never touched | `atomrl.lumen.ops.normalization.rmsnorm` |
| `local_adamw` | Fused AdamW runs directly on the FSDP2 local shards, bypassing DTensor dispatch; bit-identical | `atomrl.lumen.quantize.local_adamw` |
| `qwen_offline_fix` | Works around a VeOmni `573848a` defect: Qwen-Image `offline_training` reads `self.vae.config` with no VAE loaded and crashes on step 1. Should be fixed upstream | `veomni_patches/qwen_offline.py` |

The last five are installed before VeOmni is imported and before the trainer builds anything. All are off by default and take effect
only when listed in `LUMEN_PATCH`. `LUMEN_BATCH_HASH=1` prints a hash of every condition-model input; it is for verification only and
must not be used for timed runs.

### 3.8 Collecting evidence

```bash
# A trace of any level (steps 4–5), written to $OUT_DIR/trace/<RUN_NAME>
dx ". $EX/env.sh && RUN_NAME=prof-w-L3 WAN_TASK=online_training \
      LUMEN_PATCH=vae_bf16,vae_conv_video,conv_pad_in_kernel bash \$EXAMPLE_DIR/profile.sh wan \
      --train.empty_cache_steps 500 --train.gradient_checkpointing.enable false \
      --data.dataloader.num_workers 2 --data.dataloader.pin_memory true"

python3 kernel_diff.py <A.json.gz> <B.json.gz>           # kernels that appeared, vanished or changed cost
python3 kernel_blame.py <trace> --category elementwise   # attribution to the launching aten op (F.pad copies)
python3 trace_report.py <trace>                          # GPU busy, category shares, idle pits
python3 opt_phase.py <trace dir> ...                     # host time in the optimizer step
python3 loss_equal.py <reference run> <run> ...          # loss / grad_norm bit-identical?
python3 determinism.py <level> ...                       # do a level's repeats agree?
python3 compare_batch_hash.py <log A> <log B>            # did two runs see identical inputs?
```

### 3.9 Acceptance criteria

- Every run's `.meta.txt` shows `EXIT_CODE : 0`, and both VeOmni and Lumen report 0 dirty files.
- A run with a VAE patch logs `Lumen convolution: AITER` and 0 `torch`
  calls in `aiter conv calls`. A `patched` line alone does not prove the
  kernel ran.
- Every L change in a conclusion needs kernel-level evidence (§2.2), not an inference from step time.
- An increment counts as a gain only if it exceeds the spread of both levels and p < 0.05.
- Only one 8-GPU job at a time; the runners enforce this with `$LOG_DIR/.run.lock`. Check
  `rocm-smi --showpids` for other users' work before starting.

### 3.10 Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| AITER convolution import fails | the installed AITER lacks the public Conv2d/Conv3d APIs | install a current, compatible AITER release |
| `IndexError` in the timestep lookup after a BF16 VAE | `encode()` output was not returned to FP32 | use this example's entry point and `vae_bf16` |
| fewer steps than asked for, yet exit 0 | fewer rows than steps × ranks | provide enough samples |
| `Conflicting visibility of agent-N` | the two GPU-visibility variables disagree | set both the HIP and CUDA variables |
| Qwen `offline_training` fails on step 1 with `'NoneType' ... 'config'` | VeOmni defect | add `qwen_offline_fix` to `LUMEN_PATCH` |
| Wan's first run sits on step 1 for a long time | FlyDSL JIT of 22 convolution shapes | wait; the disk cache is reused by later runs |

---

## 4. Files

| File | Purpose |
|---|---|
| `ladder.sh` | every ladder level, and repeats run round by round |
| `run_10steps.sh` / `run_wan_10steps.sh` | one 8-GPU 10-step run; `custom` mode reads `LUMEN_PATCH` and passes the rest through to VeOmni |
| `profile.sh` | a trace of any level with VeOmni's built-in profiler |
| `train_dit_lumen.py` | wrapper around VeOmni's training entry: applies the patches from `LUMEN_PATCH`, and reports backend routing counts |
| `lumen_vae_conv.py` | VAE convolution rewrites: exact T=1 → conv2d, kernel swap under T>1, padding in the kernel |
| `veomni_patches/` | VeOmni-only `qwen_offline.py`; convolution, attention, RMSNorm and optimizer optimizations call Lumen directly |
| `verify_aiter_conv.py` / `verify_sdpa_triton_fwd.py` / `verify_rmsnorm_fuse.py` | single-GPU numerics and speed checks |
| `test_patches.py` | unit tests for `veomni_patches/` |
| `summarize.py` | median step, spread, p, peak memory, loss deviation; `--chain` compares level by level |
| `kernel_diff.py` / `kernel_blame.py` / `trace_report.py` / `opt_phase.py` | trace analysis |
| `loss_equal.py` / `determinism.py` / `compare_batch_hash.py` | numerics and data identity |

All of the code is in this directory; the Lumen library itself is not modified.
