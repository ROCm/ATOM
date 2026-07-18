# ATOM Kimi-K3 Docker Runbook

This document records the working setup used on `/home/xiaobing` to create the Docker container and run the ATOM OpenAI-compatible server for `/mnt/models/Kimi-K3` on 4 AMD `gfx1250` GPUs.

## 1. Host Requirements

Expected host state:

- Linux host with AMD GPUs exposed through ROCm/KFD.
- 4 visible `gfx1250` GPUs for `-tp 4`.
- Docker runtime available as `docker`.
- Model files available at `/mnt/models/Kimi-K3`.
- Source trees available on host:
  - `/home/xiaobing/ATOM-K3`
  - `/home/xiaobing/aiter-k3`
  - `/home/xiaobing/my_test/run_models.sh`

Current working image:

```bash
docker.io/rocm/fw-bringup:gfx1250-atom-dev-20260715-tp4_pro_flash
```

Do not use the older `rocm/atom-dev:nightly_202607171835` image for `gfx1250`; it used an older ROCm stack and did not support this GPU correctly.

## 2. Prepare Model Mount

On this machine the model comes from NFS:

```bash
sudo mount -t nfs 172.16.1.6:/helios_msft_shared /mnt
test -f /mnt/models/Kimi-K3/config.json
```

If `/mnt/models/Kimi-K3/config.json` is missing, fix the mount before creating the container.

## 3. Prepare AMD GPU Devices

Check devices:

```bash
ls -l /dev/kfd /dev/dri
```

If `/dev/kfd` is missing, load the driver:

```bash
sudo modprobe amdgpu
sleep 5
ls -l /dev/kfd /dev/dri
```

On this host `amdgpu` had previously been blacklisted, so after a reboot this may need to be done again unless the blacklist is removed.

Recommended host setting:

```bash
if [ -w /proc/sys/kernel/numa_balancing ]; then
  echo 0 | sudo tee /proc/sys/kernel/numa_balancing
fi
```

## 4. Create Container

The current working script is `/home/xiaobing/run_docker.sh`. It removes any old `zxb_kimi3` container, recreates it, then reinstalls local `aiter-k3`, FLA, and `ATOM-K3` inside the container.

Equivalent command:

```bash
CONTAINER_NAME=zxb_kimi3
IMAGE=docker.io/rocm/fw-bringup:gfx1250-atom-dev-20260715-tp4_pro_flash

docker rm -f "${CONTAINER_NAME}" 2>/dev/null || true

docker run \
  -d \
  --privileged \
  --network=host \
  --ipc=host \
  --security-opt seccomp=unconfined \
  --add-host "$(hostname):127.0.0.1" \
  --ulimit memlock=-1:-1 \
  --ulimit stack=67108864 \
  --device=/dev/kfd \
  --device=/dev/dri \
  -e HIP_VISIBLE_DEVICES=0,1,2,3 \
  -e GLOO_SOCKET_IFNAME=lo \
  -e NCCL_SOCKET_IFNAME=lo \
  -v /home/xiaobing:/workdir \
  -v /mnt/models:/mnt/models \
  --workdir /workdir \
  --name "${CONTAINER_NAME}" \
  "${IMAGE}"
```

Then reinstall local code inside the container:

```bash
docker exec zxb_kimi3 bash -lc '
python - <<'"'"'PY'"'"'
from pathlib import Path
import shutil

pth = Path("/opt/venv/lib/python3.12/site-packages/easy-install.pth")
if pth.exists():
    lines = [line for line in pth.read_text().splitlines()
             if line.strip() != "/app/aiter"]
    pth.write_text("\n".join(lines) + ("\n" if lines else ""))
shutil.rmtree("/opt/venv/lib/python3.12/site-packages/aiter", ignore_errors=True)
PY

python -m pip uninstall -y atom amd-aiter >/tmp/pip_uninstall.log 2>&1 || true
AITER_USE_SYSTEM_TRITON=1 python -m pip install --force-reinstall --no-deps -e /workdir/aiter-k3 >/tmp/aiter_install.log 2>&1
python -m pip install --force-reinstall --no-deps "git+https://github.com/fla-org/flash-linear-attention.git" >/tmp/fla_install.log 2>&1
python -m pip install --force-reinstall --no-deps -e /workdir/ATOM-K3 >/tmp/atom_install.log 2>&1
'
```

If the host has `/home/xiaobing/run_docker.sh`, this is the preferred one-liner:

```bash
sudo bash /home/xiaobing/run_docker.sh
```

## 5. Verify Container

Check the container and image:

```bash
docker ps -a --filter name=zxb_kimi3
docker inspect zxb_kimi3 --format '{{.Config.Image}}'
```

Check GPU and Python environment:

```bash
docker exec zxb_kimi3 bash -lc '
ls -l /dev/kfd /dev/dri
python --version
python - <<'"'"'PY'"'"'
import torch
print("torch", torch.__version__)
print("cuda available", torch.cuda.is_available())
print("device count", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(i, torch.cuda.get_device_name(i))
PY
'
```

Expected:

- `torch.cuda.is_available()` is `True`.
- `torch.cuda.device_count()` is `4`.
- `atom` imports from `/workdir/ATOM-K3`.
- `aiter` imports from `/workdir/aiter-k3`.

Check Python packages:

```bash
docker exec zxb_kimi3 bash -lc '
python - <<'"'"'PY'"'"'
for m in ["fastsafetensors", "fla", "atom", "aiter"]:
    try:
        mod = __import__(m)
        print(m, "OK", getattr(mod, "__file__", None))
    except Exception as e:
        print(m, "ERR", repr(e))
PY
'
```

## 6. Run ATOM Server

Server script path:

```bash
/home/xiaobing/my_test/run_models.sh
```

Inside the container this is:

```bash
/workdir/my_test/run_models.sh
```

Important defaults in the script:

- `--model /mnt/models/Kimi-K3`
- `-tp 4`
- `--kv_cache_dtype bf16`
- `--max-model-len 8192`
- `--max-num-seqs 16`
- `--max-num-batched-tokens 7168`
- `--gpu-memory-utilization 0.9`
- `ATOM_USE_FASTSAFETENSORS=1`
- `ATOM_LOADER_USE_THREADPOOL=0`
- `ATOM_USE_UNIFIED_ATTN=1`
- `ATOM_FORCE_ATTN_TRITON=1`
- `ATOM_SYNC_AFTER_LOAD=1`

Start the server in the background:

```bash
docker exec -d zxb_kimi3 bash -lc '
cd /workdir/ATOM-K3
printf "\n===== real weights start $(date -Is) =====\n" >> /workdir/my_test/server.log
ATOM_USE_FASTSAFETENSORS=1 \
ATOM_FASTSAFETENSORS_NOGDS=1 \
ATOM_LOADER_USE_THREADPOOL=0 \
bash /workdir/my_test/run_models.sh >> /workdir/my_test/server.log 2>&1
'
```

Watch the log:

```bash
less +F /home/xiaobing/my_test/server.log
```

The external server listens on:

```text
http://127.0.0.1:8000
```

Health check:

```bash
curl -fsS http://127.0.0.1:8000/health
```

Expected response:

```json
{"status":"ok"}
```

Completion smoke test:

```bash
curl -sS http://127.0.0.1:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"/mnt/models/Kimi-K3","prompt":"Hello, my name is","max_tokens":16,"temperature":0}'
```

## 7. Stop or Restart Server

Stop only the server process inside the container:

```bash
docker exec zxb_kimi3 bash -lc \
  'pkill -TERM -f "^python -m atom\.entrypoints\.openai_server" || true'
```

Restart container if GPU device nodes changed after driver load:

```bash
docker restart zxb_kimi3
```

If `/dev/kfd` was created after the container was created, recreate the container with `/home/xiaobing/run_docker.sh`.

## 8. Common Issues

### `/dev/kfd` missing in container

Cause: container was created before `amdgpu` loaded, or device mapping is missing.

Fix:

```bash
sudo modprobe amdgpu
sudo bash /home/xiaobing/run_docker.sh
```

### `torch` or `rocminfo` crashes

Most likely the image ROCm stack does not support `gfx1250`. Use:

```bash
docker.io/rocm/fw-bringup:gfx1250-atom-dev-20260715-tp4_pro_flash
```

Avoid:

```bash
docker.io/rocm/atom-dev:nightly_202607171835
```

### Model not found

Check:

```bash
test -f /mnt/models/Kimi-K3/config.json
docker exec zxb_kimi3 bash -lc 'test -f /mnt/models/Kimi-K3/config.json'
```

If missing, remount `/mnt`.

### Slow or stuck loading

Use the current loader settings:

```bash
ATOM_USE_FASTSAFETENSORS=1
ATOM_FASTSAFETENSORS_NOGDS=1
ATOM_LOADER_USE_THREADPOOL=0
```

These avoid the earlier many-small-copy/threadpool issues during real-weight loading.

### Server says ready but completion quality is bad

This runbook covers environment and server startup. Accuracy debugging is separate. Current investigation found that the first full-attention/MLA layer is the first place where hidden-state stats can become non-finite, so keep `ATOM_USE_UNIFIED_ATTN=1` and `ATOM_FORCE_ATTN_TRITON=1` in `run_models.sh` while debugging Kimi-K3.

