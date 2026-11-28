# DeepSeek-V4-Pro: server fails at startup in the MegaMoE S1F stage-1 builder (455-CI 2026-11-28)

## Environment
- Image: `rocm/fw-bringup-mlperf:gfx1250-atom-ci455-20261128` (digest `sha256:58d7808f5fbfdf020616ebfc421327e095e839d2b59d1b296861ff09dca8d2fd`), ROCm 10.2.0 base
- ATOM `main@380aeb8a0`, aiter `main@9da592818`; last good session: 2026-11-24, ATOM `dc6cd23bb`, aiter `2989e550f` (GSM8K 0.9462)
- Machine: one MI455X node, 4 GPUs; weights: `deepseek-ai/DeepSeek-V4-Pro` @ `b5968e9190ef611bbf34a7229255be88a0e937c1`

## Steps
1. Start the test container. The ROCm 10.2 base activates its venv only in the entrypoint, so put it on `PATH` for `docker exec`:
   ```bash
   IMG=rocm/fw-bringup-mlperf:gfx1250-atom-ci455-20261128
   V=/root/venv/atom-venv-gfx1250
   P=$(docker image inspect $IMG --format '{{range .Config.Env}}{{println .}}{{end}}' | sed -n 's/^PATH=//p')
   docker run -d --name v4pro --network host --ipc host --shm-size 128g \
     --device /dev/kfd --device /dev/dri --group-add video --security-opt seccomp=unconfined \
     -e PATH=$V/bin:$P -e VIRTUAL_ENV=$V -e GPU_ARCHS=gfx1250 -e GPU_TARGETS=gfx1250 -e PYTORCH_ROCM_ARCH=gfx1250 \
     -v <DeepSeek-V4-Pro dir>:/models/DeepSeek-V4-Pro:ro $IMG
   ```
2. Start the server with the 455-CI recipe's DeepSeek-V4-Pro command, as run: [server_cmd.sh](server_cmd.sh).
3. No client is needed: the server exits before it serves.

## Expected vs actual
- Expected: the server starts (2026-11-24 aiter `2989e550f`: ready, GSM8K 0.9462).
- Actual: every ModelRunner raises during warmup and the server exits after 381 s; reproduced 1 of 1 runs.

## Logs
[error.log](error.log). Core:
```
File "/app/ATOM/atom/model_ops/fused_moe/mori_v2_prepare_finalize.py", line 797, in forward
File "/app/aiter/aiter/ops/flydsl/kernels/mega_moe_gfx1250/mega_moe.py", line 532, in _compute
File "/app/aiter/aiter/fused_moe.py", line 1474, in _fused_moe_impl
File "/app/aiter/aiter/ops/flydsl/grouped_moe_gfx1250.py", line 1446, in _grouped_a8w4_tdm_moe
File "/app/aiter/aiter/ops/flydsl/kernels/mega_moe_gfx1250/s1f_ctl.py", line 733, in gemm1
File "/app/aiter/aiter/ops/flydsl/kernels/mega_moe_gfx1250/s1f_stage1.py", line 667, in make_s1f_stage1
ValueError: an op's stores overflow the storecnt wait field
```

## What we checked
- The check that raises (`s1f_stage1.py:667`) came with ROCm/aiter#7921 ("[FlyDSL] [gfx1250] grouped MoE GEMM1 opts into mega", merged 2026-11-26), the only commit since `2989e550f` that touches `s1f_stage1.py` / `s1f_ctl.py`.
- The same image with only #7921 reverted (`git revert 4120f9839`, editable reinstall) starts in 761 s and scores GSM8K 0.9507 (2026-11-21: 0.9507).
- The same commit lowers MiniMax-M3's GSM8K strict-match from about 0.940 to 0.923 (aiter bisect, each point run on two nodes; see [findings](../findings.md)): the new GEMM1 path also changes M3's MoE numerics on gfx1250.
- PRs searched (ROCm/aiter, open and merged since 2026-11-27): `storecnt`, `s1f stage1`, `mega GEMM1`, `7921`. Open S1F fixes (#8731 pre_ready, #8483 LDS overflow) are about other failures; none about this one.
