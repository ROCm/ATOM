# Reproduce Kimi-K3 gsm8k on MI450 (gfx1250 ×4, ATOM)

End-to-end steps to reproduce the accuracy of Kimi-K3 (`Kimi-K3` /
`Minimax-m3-xiaobing`) served by ATOM on **AMD MI450 (gfx1250)**, `-tp 4`.

**Validated (5-shot, base completions, tp4, CUDA-graph decode):**
- **serial (num_concurrent=1): flexible/strict ≈ 0.98** (100q) — on par with / above
  the MI355 gfx950 number (0.9378).
- **concurrent (num_concurrent=8): flexible/strict ≈ 0.92** (200q). The ~0.01–0.02
  gap vs MI355 conc=8 (0.93) is a **known gfx1250 concurrent-decode divergence**
  (see Known issues), not a core-accuracy bug.
- **full 1319 conc=8: flexible 0.9067 / strict 0.9060** (±0.008, ~64 min, 0 crashes).
  ~0.024 below MI355 conc=8 (0.93) — the gfx1250 concurrent-decode divergence (see
  Known issues); gfx1250 **serial 0.98** has no such gap.

> This is the gfx1250 counterpart of `REPRODUCE_MI355_GSM8K.md`. The model math is
> identical; the difference is the gfx1250 enablement (native CK kernels via a
> composable_kernel patch) plus a few gfx1250 work-arounds documented below.

Assumes: a host with visible **gfx1250** GPUs (this box: 8× gfx1250, 432 GB each;
we use `-tp 4`), `docker`, the model at **`/data/models/Kimi-K3`** (local NVMe),
and the source trees `ATOM-K3` + `aiter-k3` under `$HOME/xiaobing/` (mounted into
the container at the same path via `-v /home:/home`).

Model: KimiLinear hybrid — 24 MLA full-attn layers + 69 KDA linear-attn layers,
896-expert MXFP4 latent MoE (top-16, SiTU act, beta=4/linear_beta=25),
`first_k_dense_replace=1`, `attn_res_block_size=12`.

---

## 1. Create the container

Image: `rocm/fw-bringup:gfx1250-atom-dev-20260715-tp4_pro_flash` (gfx1250 dev image
with the gfx1250 Triton overlay at `/app/triton-mi450`). See `run_docker.sh`:

```bash
image=rocm/fw-bringup:gfx1250-atom-dev-20260715-tp4_pro_flash
docker rm -f zxb_kimi 2>/dev/null || true
docker run -dit --rm --name zxb_kimi \
  --device=/dev/kfd --device=/dev/dri --device=/dev/mem \
  --ulimit memlock=-1:-1 --cap-add=SYS_PTRACE --cap-add=SYS_NICE \
  --net host --ipc host --security-opt seccomp=unconfined \
  -v /shared:/shared -v /data:/data -v /home:/home \
  --entrypoint=/bin/bash "$image" -lc "sleep infinity"
```

Install the local `aiter-k3` + `flash-linear-attention` + `ATOM-K3` editable, and
`fastsafetensors` (missing from the image; without it the loader silently falls
back to the slow `safe_open` path):

```bash
docker exec zxb_kimi bash -lc '
python -m pip uninstall -y atom amd-aiter aiter || true
AITER_USE_SYSTEM_TRITON=1 python -m pip install --force-reinstall --no-deps -e /home/carhuang/xiaobing/aiter-k3
python -m pip install --force-reinstall --no-deps "git+https://github.com/fla-org/flash-linear-attention.git"
python -m pip install --force-reinstall --no-deps -e /home/carhuang/xiaobing/ATOM-K3
python -m pip install --no-deps fastsafetensors
'
```

Required source revisions (must contain the gfx1250 fixes):
- `ATOM-K3` @ `xiaobing/k3` (≥ commit `fix(kimi-k3): gfx1250 accuracy, stability,
  CUDA-graph`): MoE sub-batch, warmup cap, cudagraph-capture-safe reshape_and_cache,
  `run_models.sh`.
- `aiter-k3` @ `xiaoing/k3` (≥ commit `feat(gfx1250): enable native CK kernels via
  composable_kernel patch`): the CK patch + grouped-MoE contiguous-M-off default.

### 1a. Apply the CK gfx1250 patch (REQUIRED for native CK kernels)

Stock composable_kernel does not know gfx1250, so its CK-backed JIT modules
(`module_quant/cache/rmsnorm_quant/moe_asm/custom_all_reduce/sample/moe_sorting_opus`)
fail to build. Apply the patch that registers gfx1250 as a gfx12-family CK target
(wavefront 32):

```bash
docker exec zxb_kimi bash -lc '
cd /home/carhuang/xiaobing/aiter-k3
git -C 3rdparty/composable_kernel apply patches/ck_gfx1250_enable.patch || \
  git -C 3rdparty/composable_kernel apply --check patches/ck_gfx1250_enable.patch  # already applied?
'
```

Import sanity (run from the ATOM dir — from `/app` the image-bundled `aiter`/`triton`
shadow the editable ones):

```bash
docker exec zxb_kimi bash -lc '
cd /home/carhuang/xiaobing/ATOM-K3
python - <<PY
import torch
print("cuda", torch.cuda.is_available(), "devices", torch.cuda.device_count())
import aiter, atom, fla, lm_eval
print("aiter", aiter.__file__)   # -> /home/carhuang/xiaobing/aiter-k3/aiter/...
print("atom",  atom.__file__)    # -> /home/carhuang/xiaobing/ATOM-K3/atom/...
PY'
```

---

## 2. Start the server (tp4 + CUDA graph, native CK)

```bash
docker exec -d zxb_kimi bash -lc '
bash /home/carhuang/xiaobing/ATOM-K3/my_script/run_models.sh \
  > /home/carhuang/xiaobing/ATOM-K3/my_script/server.log 2>&1'

docker exec zxb_kimi bash -lc '
until curl -sf -m3 http://127.0.0.1:8000/health >/dev/null; do sleep 5; done; echo READY'
```

Load takes **~20 min** (see Known issues — load speed). `run_models.sh` uses:
- `-tp 4`, `--level 0`, CUDA-graph decode (no `--enforce-eager`), `gmu 0.93`,
  `--max-model-len 4096 --max-num-seqs 8 --max-num-batched-tokens 2048`.
- `--no-enable_prefix_caching --no-enable_chunked_prefill` — **whole-prompt prefill**
  (the K3 MLA prefill is per-seq SDPA over in-batch Q/K/V, so a chunked/cached
  prefix would be missed → wrong MLA).
- `ATOM_K3_MOE_CHUNK=128` — sub-batch the MoE block (see Known issues).
- `ATOM_WARMUP_MAX_TOKENS=256` — cap the dummy warmup (see Known issues).
- native aiter **CK** kernels on (quant/cache/rmsnorm/moe_asm/custom_all_reduce/
  sampler) — needs the CK patch from step 1a. Fall back with `AITER_USE_TRITON_QUANT=1
  AITER_USE_OPUS_RMSNORM=1 AITER_USE_TORCH_TOPK=1 ATOM_USE_TORCH_SAMPLER=1
  ATOM_USE_TORCH_CACHE=1 AITER_DISABLE_CUSTOM_ALL_REDUCE=1` if the patch is absent.
- `ATOM_KDA_FORCE_RECURRENT=1` — fla `chunk_kda` NaNs on short seqs on gfx1250.

---

## 3. Accuracy test

Uses `lm_eval --model local-completions` (BASE completions, NOT chat) + 5-shot —
matching how Kimi models are scored (chat/thinking template otherwise tanks it).

```bash
# concurrent (deployment mode), full 1319, conc=8 — ~60 min
docker exec zxb_kimi bash -lc '
K3_MODEL=/data/models/Kimi-K3 K3_NUM_CONCURRENT=8 K3_NUM_FEWSHOT=5 \
bash /home/carhuang/xiaobing/ATOM-K3/my_script/eval_k3_gsm8k.sh 2>&1 | tee /home/carhuang/xiaobing/ATOM-K3/my_script/gsm8k_full_c8.log'

# quick smoke: 30q serial (~6 min, ~0.97) or 50q conc=8 (~3 min, ~0.92)
docker exec zxb_kimi bash -lc 'K3_MODEL=/data/models/Kimi-K3 K3_LIMIT=30 K3_NUM_CONCURRENT=1 bash .../eval_k3_gsm8k.sh'
docker exec zxb_kimi bash -lc 'K3_MODEL=/data/models/Kimi-K3 K3_LIMIT=50 K3_NUM_CONCURRENT=8 bash .../eval_k3_gsm8k.sh'
```

Expected (this session):
```
serial  100q  : flexible-extract ~0.98   / strict-match ~0.98
conc=8  200q  : flexible-extract  0.92   / strict-match  0.92   (± 0.019)
conc=8  1319  : flexible-extract  0.9067 / strict-match  0.9060 (± 0.008)
```
vs MI355 gfx950 conc=8 = 0.93. The gfx1250 core/serial accuracy is on par or better
(0.98); the ~0.024 conc=8 shortfall is the concurrent-decode divergence below.

---

## gfx1250-specific fixes (why they exist)

1. **Native CK kernels via `patches/ck_gfx1250_enable.patch`.** CK never registered
   gfx1250 (its `__gfx12__` macro is gated on `__gfx1200__/__gfx1201__/__gfx12_generic__`,
   and its target-id / `amdgcn_compiler_target_state` list had no GFX1250) → all
   CK-backed JIT modules failed to build (undefined `CK_TILE_BUFFER_RESOURCE_3RD_DWORD`,
   "only one target arch" static_assert). gfx1250 is warp-32 gfx12-family, so the
   patch registers it as such. This removes the whole torch/opus/triton fallback
   cluster (same accuracy, ~13% faster decode).
2. **`ATOM_K3_MOE_CHUNK=128` (required for correctness).** On gfx1250 the grouped
   MoE + MXFP4 latent projections are only numerically correct at small M: at large
   prefill M the DeepGEMM contiguous-M path OOB-crashes, and the non-contiguous path
   returns coherent-but-wrong values (gsm8k 0.0). The MoE block is per-token
   independent, so sub-batching it to ≤128 tokens is numerically identical and keeps
   whole-prompt prefill (correct MLA) working. Repro: `my_script/repro_moe.py`.
3. **`ATOM_WARMUP_MAX_TOKENS=256` (warmup-only).** A large all-zero warmup batch
   samples garbage logits over all positions and faults the sampler on gfx1250; real
   inference only samples the last token per seq, so it is unaffected.
4. **Whole-prompt prefill** (`--no-enable_chunked_prefill --no-enable_prefix_caching`).

## Known issues / gotchas

- **Load is ~20 min.** `/data` is local NVMe; the model is 1.5 TB (96×16 GB) and the
  host has ~1 TB RAM, so the OS page cache can't hold it and each of the 4 TP ranks
  re-reads every shard (4× amplification). `fastsafetensors` (device=cuda) is used;
  `ATOM_FASTSAFETENSORS_DIST_LOAD=1` (read 1/4 per rank) would ~halve it but currently
  hangs. Practical rule: load once, keep the server alive, iterate by request.
- **Concurrent decode divergence (conc>1 ≈ 0.92 vs serial 0.98).** The flydsl MoE
  per-token output drifts ~2e-3 by batch position → near-tie argmax flips in the
  recurrent KDA state → answer flips. Same root cause noted in the MI355 handoff;
  more visible on gfx1250. Serial is clean.
- **Clean GPUs only** — a competing job stealing VRAM mid-run corrupts the score.
- If you change model code, clear `/root/.cache/atom/*` before restarting.
- Full background: `my_script/K3_ACCURACY_HANDOFF.md`,
  `my_script/KIMI_K3_PRECISION_DEBUG_NOTES.md`.
