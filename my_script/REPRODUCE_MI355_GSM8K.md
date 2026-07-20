# Reproduce Kimi-K3 gsm8k 0.9378 on MI355 (gfx950 ×8, ATOM)

End-to-end steps to reproduce the validated accuracy of Kimi-K3
(`Minimax-m3-xiaobing`) served by ATOM on 8× AMD MI355 (gfx950):

**gsm8k (full 1319, 5-shot, base completions, tp8, CUDA-graph):
`flexible-extract 0.9378 / strict-match 0.9371`** — on par with Kimi-K2-Thinking (0.9363).

Assumes: host with 8 visible gfx950 GPUs, `podman` (or `docker`), the model at
`/shared/data/amd_int/models/xiaobing/Minimax-m3-xiaobing`, and the source trees
`ATOM-K3` + `aiter-k3` under your `$HOME` (this guide uses `$HOME/xiaobing/{ATOM-K3,aiter-k3}`,
which mounts to `/workdir/xiaobing/...` in the container).

---

## 1. Create the container

Image: `docker.io/rocm/atom-dev:vllm-latest`. `$HOME` is mounted at `/workdir`, and
`/shared` (the NFS with the model) is mounted through.

```bash
podman pull docker.io/rocm/atom-dev:vllm-latest

podman rm -f zxb_kimi_k3 2>/dev/null || true
podman run -d \
  --privileged --group-add keep-groups \
  --network=host --ipc=host --security-opt seccomp=unconfined \
  --device=/dev/kfd --device=/dev/dri \
  -v "$HOME:/workdir" -v /shared:/shared \
  --workdir /workdir \
  --name zxb_kimi_k3 \
  docker.io/rocm/atom-dev:vllm-latest \
  sleep infinity
```

Install the local `aiter-k3` (with the MoE patches), `flash-linear-attention` (KDA
kernels), and `ATOM-K3` (with the accuracy fixes) into the container venv. The `-e`
editable installs point at the mounted source so all fixes are picked up:

```bash
podman exec zxb_kimi_k3 bash -lc '
python - <<PY
from pathlib import Path
import shutil
pth = Path("/opt/venv/lib/python3.12/site-packages/easy-install.pth")
if pth.exists():
    lines = [l for l in pth.read_text().splitlines() if l.strip() != "/app/aiter"]
    pth.write_text("\n".join(lines) + ("\n" if lines else ""))
shutil.rmtree("/opt/venv/lib/python3.12/site-packages/aiter", ignore_errors=True)
PY
python -m pip uninstall -y atom amd-aiter >/tmp/pip_uninstall.log 2>&1 || true
AITER_USE_SYSTEM_TRITON=1 python -m pip install --force-reinstall --no-deps -e /workdir/xiaobing/aiter-k3 >/tmp/aiter_install.log 2>&1
python -m pip install --force-reinstall --no-deps "git+https://github.com/fla-org/flash-linear-attention.git" >/tmp/fla_install.log 2>&1
python -m pip install --force-reinstall --no-deps -e /workdir/xiaobing/ATOM-K3 >/tmp/atom_install.log 2>&1
'
```

Sanity check:
```bash
podman exec zxb_kimi_k3 bash -lc '
python - <<PY
import torch
print("cuda", torch.cuda.is_available(), "devices", torch.cuda.device_count())
for m in ["fla","aiter","atom","lm_eval"]:
    mod=__import__(m); print(m, getattr(mod,"__file__",None))
PY'
```
Expect `device_count 8`, and `atom`/`aiter` importing from `/workdir/xiaobing/...`.

Required source revisions (branches, must contain the fixes):
- `ATOM-K3` @ `xiaobing/k3` (commits: prefill KV-cache write + latent-MoE norm,
  triton MoE situ activation, MLA flash KV-cache layout, CUDA-graph launch script).
- `aiter-k3` @ `xiaoing/k3` (commit: `shuffle_scale_moe` return_layout + flat `topk` sigmoid).

---

## 2. Start the server (tp8 + CUDA graph)

```bash
podman exec -d zxb_kimi_k3 bash -lc '
bash /workdir/xiaobing/ATOM-K3/my_script/run_k3_mi355_tp8_cudagraph.sh \
  > /workdir/xiaobing/server_k3.log 2>&1'
```
Loads in ~3-4 min. Wait until ready:
```bash
podman exec zxb_kimi_k3 bash -lc '
until curl -sf -m3 http://127.0.0.1:8000/health >/dev/null; do sleep 5; done; echo READY'
```
The script uses `-tp 8`, CUDA-graph decode (no `--enforce-eager`, `--level 0`),
`--gpu-memory-utilization 0.93`, `--no-enable_prefix_caching --no-enable_chunked_prefill`,
and the gfx950 kernel env. Decode is ~20 tok/s (~4x over eager). See the script header
for details. (Optional +9% decode: `ATOM_USE_TRITON_MOE=0` uses the aiter FlyDSL
Situv2 MoE kernel — ~0.955 on a 200q spot-check; the 0.9378 number below is the
default triton path.)

---

## 3. Accuracy test

```bash
# full 1319 (reproduces 0.9378) — ~15-20 min
podman exec zxb_kimi_k3 bash -lc '
bash /workdir/xiaobing/ATOM-K3/my_script/eval_k3_gsm8k.sh 2>&1 | tee /workdir/xiaobing/gsm8k_result.log'

# quick 50-question smoke test (~3 min)
podman exec zxb_kimi_k3 bash -lc 'K3_LIMIT=50 bash /workdir/xiaobing/ATOM-K3/my_script/eval_k3_gsm8k.sh'
```
Uses `lm_eval --model local-completions` (BASE completions, NOT chat) + `--num_fewshot 5`
— matching how Kimi models are scored (see `recipes/Kimi-K2*.md`); the chat/thinking
template otherwise burns the token budget and tanks the score.

Expected:
```
|gsm8k| flexible-extract | 5 | exact_match | 0.9378 | ± 0.0067 |
|     | strict-match     | 5 | exact_match | 0.9371 | ± 0.0067 |
```

---

## Gotchas
- **Clean GPUs only.** A competing job stealing VRAM mid-run silently corrupts the score
  (a one-off 0.53 / 0.12 was traced to a shared-GPU sglang CI container). Before running:
  `podman ps` and stop any competing container; `rocm-smi --showmeminfo vram` should be ~0
  before the server loads; re-check the eval output has no `ServerDisconnected`.
- `-tp 8` is required (tp4 OOMs: MoE weights ~175GB/GPU).
- If you change model code, clear `/root/.cache/atom/*` before restarting the server.
- Full background/context of the fixes: `my_script/K3_ACCURACY_HANDOFF.md`.
