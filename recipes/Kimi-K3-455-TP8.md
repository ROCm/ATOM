# Kimi-K3 TP8 on gfx1250 (MI455) — TP8 + DCP8 on two trays

Full-size **Kimi-K3** on **two gfx1250 B0 trays, 4 GPUs each (432 GiB/GPU)**, served as **one tensor-parallel group
of 8**:

- **Model:** 2.78T parameters. 93 layers: 24 full-attention MLA and 69 KDA linear. 896 routed experts, top-16.
  MXFP4 routed experts, BF16 everything else; the launch quantizes the BF16 layers to FP8 at load.
- **Parallelism:** `-tp 8` across the two trays (4 ranks per tray), decode context parallel 8. The MoE runs over the
  same 8 GPUs, either expert-parallel (`K3_MOE=ep`, the default: 112 experts per GPU) or tensor-parallel
  (`K3_MOE=tp`: all 896 experts on every GPU, each split 8 ways, inter_dim 384).
- **Checkpoint:** about 1.42 TiB, needed **in full on both trays**.
- **Image:** `rocm/fw-bringup:gfx1250-atom-k3-20261022`. It is the 20261016 image of the
  [wide-EP recipe](https://github.com/ROCm/ATOM/blob/recipe-k3-455-wideep/recipes/Kimi-K3-455-wideEP.md), with ATOM
  and aiter switched for TP8, aiter's K3 TP8 GEMM and MoE tables, and the launcher in `/opt/k3`. See [Image](#image).

Validated on CT2 (node0) and CT17, 2026-10-22, with `K3_MOE=tp` and full GSM8K. The throughput figures come from the
same launcher and GEMM tables on the 20261016 image, with `K3_MOE=ep` (CT15 + CT13, 2026-10-21). The newer
[20261027 image](#the-20261027-image) does not start without two ATOM overlays; with them it gave GSM8K strict-match
0.9651 on E15's Helios-R trays 05 and 07 (2026-10-27, `K3_MOE=ep`).

| | |
|---|---|
| **GSM8K**, 5-shot, all 1319 | `K3_MOE=tp`: **strict-match 0.9568 ±0.0056** with RCCL all-reduce, and 0.9492 ±0.0060 with the default custom all-reduce. The gap is within run-to-run noise. `K3_MOE=ep` on the 20261016 image, RCCL: 0.9613–0.9629. `K3_MOE=ep` on the [20261027 image](#the-20261027-image) with its two overlays, custom all-reduce: 0.9651 ±0.0051 |
| **Throughput**, 14k/1k, concurrency 32 | `K3_MOE=ep`, RCCL: **893.1 output tok/s**, median TPOT 34.5 ms. Without the TP8 GEMM tables: 787.9 tok/s, 38.8 ms. Not run on this image |
| **Memory** | `peak_torch` 190 GB and `available_for_kv` about 143 GB per GPU at `--gpu-memory-utilization 0.90` |
| Gate | B0 text; chat `17 * 23` answers `391`; decode matches prefill |
| Cold start | 12–13 min in a fresh container, mostly aiter JIT builds; about 4 min warm; 22–24 min when one tray has 1 TB of RAM |
| AgentX | `K3_MODE=agentx` in the launcher, and ATOM CI's aiperf client, as in the wide-EP recipe. Not run on this image |

Contents:

- [Image](#image)
- [The 20261027 image](#the-20261027-image)
- [Before you start](#before-you-start)
- [Launch](#launch)
- [Accuracy](#accuracy)
- [Throughput](#throughput)
- [AgentX](#agentx)
- [Profiling](#profiling)
- [Known issues](#known-issues)

---

## Image

`rocm/fw-bringup:gfx1250-atom-k3-20261022`, digest
`sha256:0dac2559dc9e3817a0aaa940048b8cc4b18ffc1beb1f3c52847041f6cc780404`. It is about 151 GB unpacked.

- **Private repository.** `docker login` with a Docker Hub account that can read `rocm/fw-bringup`.
- **Layers.** A tray that already has the 20261016 image, or its base, pulls only the top layers.

| | commit | contents |
|---|---|---|
| ATOM, `/app/ATOM` | `6c442b7a` ([`gyu/k3-455-tp8`](https://github.com/ROCm/ATOM/tree/gyu/k3-455-tp8)) | The 20261016 image's ATOM: `main` `d82f906b` + [#2557](https://github.com/ROCm/ATOM/pull/2557) + [#2554](https://github.com/ROCm/ATOM/pull/2554). Plus one TP group across nodes and DCP decode on gfx1250's PS1 MLA kernel (both now in [#2647](https://github.com/ROCm/ATOM/pull/2647)), and DSpark with DCP |
| aiter, `/app/aiter` | `3145069b` ([`zejun/k3-tp8-image-20261022`](https://github.com/ROCm/aiter/tree/zejun/k3-tp8-image-20261022)) | The 20261016 image's aiter: `main` `dee019f6` + [#6525](https://github.com/ROCm/aiter/pull/6525) + [#6530](https://github.com/ROCm/aiter/pull/6530). Plus the grouped-MoE k-tile fit (`cc8d4ff12`) and the K3 TP8 tables below |
| mori, `/app/mori` | `695d4c03` | the base's |

FlyDSL 0.4.0, torch, Triton and ROCm 10.1 (TheRock wheels, Python 3.12) come from the base. So does aiperf 0.12.0
(SemiAnalysis aiperf `754356e9`, in `/opt/aiperf-venv`, on `PATH`); see [Installing aiperf](#installing-aiperf).

**The K3 TP8 tables** are in `/app/aiter/aiter/configs/model_configs/`.
- aiter merges them with its own tables at first use.
- Never set an `AITER_CONFIG_*` variable: it replaces that merge and drops these tables.

| file | rows | used by |
|---|---|---|
| `kimik3tp8_a8w8_blockscale_mxscale_bpreshuffle_tuned_gemm.csv` | 94 | the online FP8 GEMMs ([#6763](https://github.com/ROCm/aiter/pull/6763)) |
| `a8w8_bpreshuffle_tuned_gemm_kimik3tp8.csv` | 16 | `kv_b_proj` in PTPC FP8 (#6763) |
| `kimik3_bf16_tuned_gemm.csv` | 28 retuned | the MoE router gate and the LM head (#6763) |
| `kimik3tp8_tuned_grouped_fmoe.csv` | 15 | the grouped MoE with `K3_MOE=tp`, 1–16384 tokens ([#6768](https://github.com/ROCm/aiter/pull/6768)) |

**`/opt/k3`** holds:
- **`k3_455_tp8dcp8_server.sh`, the launcher.**
  - It is ATOM `ee7efc6f`'s
    [`scripts/k3_455_tp8dcp8_server.sh`](https://github.com/ROCm/ATOM/blob/ee7efc6fb2b8279486b8f9060656de2460531e1f/scripts/k3_455_tp8dcp8_server.sh),
    unchanged. The copy in `/app/ATOM/scripts/` is older; use this one.
  - The image sets `K3_ATOM_DIR=/app/ATOM`, so the launcher serves the image's ATOM and fetches nothing.
- **The clients** `gate.py`, `gsm8k.sh` and `bench.sh`.
- **`verify_inside.sh`**, static checks that need no GPU.
- **`README.md`**, plus `manifest.json` with the exact commits.

`/opt/k3-20261016` is the base image's dp8ep8 kit, kept for reference.

---

## The 20261027 image

`rocm/fw-bringup:gfx1250-atom-k3-20261027`, digest
`sha256:5eefef68bd4d0d95ba73096f5df50a64f569ddefba3a83c76d3c10eb73c938f3`. It is the 20261022 image with ATOM and aiter
moved on:

| | commit | contents |
|---|---|---|
| ATOM, `/app/ATOM` | `d8877f1c` | A newer `main`, with two PRs merged in: [#2647](https://github.com/ROCm/ATOM/pull/2647) (one TP group across nodes) and [#2737](https://github.com/ROCm/ATOM/pull/2737) (prefetch the next MLA layer's cached-prefix KV all-gathers, opt-in). It does **not** carry #2554 and #2557 |
| aiter, `/app/aiter` | `7061cbd8` | `main`, which holds the K3 TP8 tables. #6763's GEMM rows are there: [#6530](https://github.com/ROCm/aiter/pull/6530) moved its 94 mxscale rows into `kimik3_a8w8_blockscale_mxscale_bpreshuffle_tuned_gemm.csv`, now 214 rows. [#6768](https://github.com/ROCm/aiter/pull/6768)'s grouped-MoE table is there, retuned. In place of `cc8d4ff12`, aiter has the upstream K-tail fix [#6663](https://github.com/ROCm/aiter/pull/6663); the launcher prints `cc8d4ff12 not needed` |

- **Image files.**
  - `/opt/k3/build_info.json` holds the commits.
  - `manifest.json` and `verify_inside.sh` moved to `/opt/k3/base-20261022/` and describe the base image, so the
    `verify_inside.sh` step of [Launch](#launch) does not apply.
  - `/opt/k3/README.md` is still the 20261022 image's.
  - The launcher differs from the 20261022 image's only in `ATOM_COMMIT` and in that aiter message.
- **As shipped, it does not start.** Post-load weight processing fails on every rank with
  `AssertionError: 6284 % 16 == 12`. The weight is not an expert's, so `K3_MOE` does not matter:
  - KDA's `in_proj` is fused with `b_proj` and `f_a_proj`: N = 4 × 1536 + 12 + 128 = 6284 per rank.
  - Online quantization makes it a per_1x128 FP8 weight, and aiter's preshuffle needs N % 16 == 0.
  - [#2554](https://github.com/ROCm/ATOM/pull/2554) (E8M0 block scales for online per_1x128 FP8 on gfx1250) pads such a
    weight's N to a multiple of 128 first.
  - [#2557](https://github.com/ROCm/ATOM/pull/2557) plans the persistent MLA decode over only the rows that q and o have.
  - Both are still open. The 20261016 and 20261022 images carried both as overlays; this image dropped them.

**Workaround: serve the image's ATOM with both overlays.** In each tray's container, before the first start:

```bash
P=https://raw.githubusercontent.com/ROCm/ATOM/recipe-k3-455-tp8/recipes/Kimi-K3-455-TP8-patches
curl -sSfL -o /tmp/atom-2554.diff $P/atom-2554.diff && curl -sSfL -o /tmp/atom-2557.diff $P/atom-2557.diff
D=/root/.cache/atom-overlay; rm -rf $D && cp -a /app/ATOM $D && cd $D
git -c safe.directory='*' apply /tmp/atom-2554.diff /tmp/atom-2557.diff
# fixed metadata, so both trays get the same commit
export GIT_AUTHOR_NAME=k3 GIT_AUTHOR_EMAIL=k3@localhost GIT_AUTHOR_DATE=2026-10-27T06:00:00Z \
  GIT_COMMITTER_NAME=k3 GIT_COMMITTER_EMAIL=k3@localhost GIT_COMMITTER_DATE=2026-10-27T06:00:00Z
git -c safe.directory='*' commit -qam "ATOM #2554 + #2557 on d8877f1c"
sed "s/^ATOM_COMMIT=.*/ATOM_COMMIT=$(git -c safe.directory='*' rev-parse HEAD)/" \
  /opt/k3/k3_455_tp8dcp8_server.sh > /root/.cache/k3_server.sh
# only on a tray whose NIC is not enp1s0f1 (see Before you start), e.g. E15's Helios-R trays:
#   sed -i 's/enp1s0f1/enp4s0np0/g' /root/.cache/k3_server.sh
rm -rf /root/.cache/atom/* /tmp/torchinductor_root
```

- **Starting it.** Start that copy, with `-e K3_ATOM_DIR=/root/.cache/atom-overlay`, as in [Launch](#launch). On node0:
  `docker exec -d -e K3_ATOM_DIR=/root/.cache/atom-overlay k3tp8 bash -c 'bash /root/.cache/k3_server.sh 0 <node0 IP> > /root/.cache/tp8.log 2>&1'`.
  On node1, the same with `4`.
- **The patch files.**
  - `atom-2554.diff` is #2554 at `e16b3dd4`.
  - `atom-2557.diff` is the 20261022 image's overlay: #2557 at `8a24a910`. The PR has moved on since.
  - The block makes commit `6c960542` on every tray, with the same tree (`161af75e`) as the validation run below.
- **Validated with the workaround** on E15's Helios-R trays, 2026-10-27:
  - trays 05 (node0) and 07, NIC `enp4s0np0`;
  - the launcher defaults: `K3_MOE=ep`, custom all-reduce;
  - 607 layers online-quantized, healthy in 10 min, `GATE PASS`;
  - GSM8K strict-match 0.9651 ±0.0051, flexible-extract 0.9636 ±0.0052.

---

## Before you start

On both trays:

```bash
ls /sys/class/kfd/kfd/proc | wc -l              # 0: nobody is using the GPUs
for u in /sys/bus/pci/devices/*/ualink; do      # four "active" lines, with the same ppod/vpod on both trays
  echo "$(cat $u/accel_id) $(cat $u/accel_state) $(cat $u/ppod_id)/vpod$(cat $u/vpod_id)"
done
ip -br addr show enp1s0f1                       # node0's address here is <node0 IP> below
```

- **Both trays must be in one active UALink vpod.**
  - Every TP all-reduce crosses the trays.
  - A tray in state `ppod_configured`, or two trays in different vpods, hang in the first RCCL all-reduce.
  - `ualink/vpod_active_accels` lists the vpod's accelerators; `accel_id` is 4 × (tray − 1).
- **The data-plane NIC.** The launcher sets `MORI_SOCKET_IFNAME`, `NCCL_SOCKET_IFNAME` and `GLOO_SOCKET_IFNAME` to
  `enp1s0f1`, the NIC of the CT trays.
  - Elsewhere, use the NIC that routes to the other tray (`ip route get <other tray's IP>`), e.g. `enp4s0np0` on
    E15's Helios-R trays.
  - Once the container is up (see [Launch](#launch)), make a copy of the launcher with that name, and start
    `bash /root/.cache/k3_server.sh` in place of `bash /opt/k3/k3_455_tp8dcp8_server.sh`:
    `docker exec k3tp8 bash -c "sed 's/enp1s0f1/enp4s0np0/g' /opt/k3/k3_455_tp8dcp8_server.sh > /root/.cache/k3_server.sh"`.
- **2 TB of host RAM per tray is recommended.**
  - A 1 TB tray cannot keep the checkpoint in page cache, so it loads its weights 30–45× slower.
  - The server then takes 22–24 min to come up.
- **Driver:** `amdgpu` loaded.
  - The launcher was validated with a plain driver load, without perf setup.
  - For reboots and driver loads, see the wide-EP recipe's
    [Tray operations](https://github.com/ROCm/ATOM/blob/recipe-k3-455-wideep/recipes/Kimi-K3-455-wideEP.md#tray-operations).
- **B0 silicon.** On A0 without translation, every token is `!`. See
  [A0 and B0](https://github.com/ROCm/ATOM/blob/recipe-k3-455-wideep/recipes/Kimi-K3-455-wideEP.md#a0-and-b0-silicon).
- **The checkpoint**, on a data disk of both trays:
  - Download [`moonshotai/Kimi-K3`](https://huggingface.co/moonshotai/Kimi-K3) with
    `hf download moonshotai/Kimi-K3 --local-dir /data/k3/Kimi-K3`. These runs used revision `f831ab66`
    (`--revision f831ab66814297da540d832a5235f8e904f29d06`), 1.561 TB in 113 files.
  - Check that the copy is complete:
    `cd /data/k3/Kimi-K3 && find . -maxdepth 1 -type f -printf "%f %s\n" | LC_ALL=en_US.UTF-8 sort | md5sum`
    must print `fa7ad4b64b18668b2bbf0db320232a8d`.
    - The sort order is the locale's. Under `C` or `C.UTF-8`, the default in most containers, the same files print
      `2773d6074935cda0399f6cafd717f86c`.
  - DSpark also needs [`Inferact/Kimi-K3-DSpark`](https://huggingface.co/Inferact/Kimi-K3-DSpark), at
    `/data/k3/Kimi-K3-DSpark`.

---

## Launch

Run one container per tray. Mount the directory that holds `Kimi-K3/` at `/models`. `--network=host` is required,
because the ranks meet on data-plane IPs.

```bash
WORK=/data/k3tp8; mkdir -p $WORK/cache $WORK/triton    # logs, compile caches, traces
docker run -d --name k3tp8 -w /tmp \
  --device=/dev/kfd --device=/dev/dri --network=host --ipc=host --pid=host \
  --group-add $(getent group video | cut -d: -f3) --group-add $(getent group render | cut -d: -f3) \
  --cap-add=SYS_PTRACE --cap-add=SYS_ADMIN --security-opt seccomp=unconfined --shm-size=128g \
  -v /data/k3:/models:ro -v $WORK/cache:/root/.cache -v $WORK/triton:/root/.triton \
  --entrypoint sleep rocm/fw-bringup:gfx1250-atom-k3-20261022 infinity
docker exec k3tp8 bash /opt/k3/verify_inside.sh | tail -1     # == RESULT: PASS
```

Start both trays, in either order. Each one waits for the other.

```bash
# node0: TP ranks 0-3, and the API on :8000
docker exec -d k3tp8 bash -c 'bash /opt/k3/k3_455_tp8dcp8_server.sh 0 <node0 IP> > /root/.cache/tp8.log 2>&1'
# node1: TP ranks 4-7
docker exec -d k3tp8 bash -c 'bash /opt/k3/k3_455_tp8dcp8_server.sh 4 <node0 IP> > /root/.cache/tp8.log 2>&1'
until curl -sf -m 5 http://<node0 IP>:8000/health; do sleep 15; done
```

Knobs are environment variables. Give them the same values on both trays, e.g. `docker exec -d -e K3_MOE=tp k3tp8 …`:

| variable | default | meaning |
|---|---|---|
| `K3_MOE` | `ep` | `ep`: `--enable-expert-parallel`. `tp`: every expert split 8 ways, as on MI355X; the grouped-MoE table is for this layout |
| `K3_MODE` | `accuracy` | `accuracy`: no prefix caching, 16384 batched tokens. `agentx`: see [AgentX](#agentx) |
| `K3_MAX_SEQS` | 64 | `--max-num-seqs` |
| `AITER_CUSTOM_AR_USE_SYMM_MEM` | 1 | 1: aiter's custom all-reduce across the trays (torch.symm_mem over mori) for decode-sized all-reduces. 0: RCCL for all of them |
| `K3_DSPARK` | 0 | N: DSpark with N draft tokens; the draft is read from `K3_DRAFT` (default `/models/Kimi-K3-DSpark`) |
| `K3_MODEL`, `K3_PORT` | `/models/Kimi-K3`, 8000 | the checkpoint, and the API port on node0 |

**What the launcher runs.** Use this to launch without the script. `RANK0` is 0 on node0 and 4 on node1.

```bash
export PYTORCH_ROCM_ARCH=gfx1250 GPU_ARCHS=gfx1250 GPU_ARCH_LIST=gfx1250 MORI_GPU_ARCHS=gfx1250 GPU_TARGETS=gfx1250
export HSA_OVERRIDE_GFX_VERSION=12.5.0 ENABLE_CK=0
export MORI_AMD_SMI_PATH=/usr/local/lib/python3.12/dist-packages/_rocm_sdk_core/lib/libamd_smi.so.27
export ATOM_USE_TRITON_MLA=0 AITER_MLA_DECODE_PS1_FLYDSL=1 ATOM_USE_TRITON_MLA_SHUFFLE_KV=0 ATOM_USE_UNIFIED_ATTN=1
export ATOM_MOE_GU_ITLV=1 ATOM_USE_TRITON_MOE_DECODE=0 AITER_USE_GROUPED_GEMM=1
export ATOM_FP8_BLOCKSCALE_WEIGHT_PRESHUFFLE=1 ATOM_FP8_BLOCKSCALE_USE_E8M0_SCALE=1
export AITER_ROPE_TRITON_BACKEND=1 AITER_USE_SYSTEM_TRITON=1
export NCCL_MNNVL_ENABLE=1 NCCL_IB_DISABLE=1 NCCL_P2P_DISABLE=0 NCCL_P2P_LEVEL=SYS NCCL_CUMEM_ENABLE=1
export MORI_SOCKET_IFNAME=enp1s0f1 NCCL_SOCKET_IFNAME=enp1s0f1 GLOO_SOCKET_IFNAME=enp1s0f1   # the image says lo
export AITER_CUSTOM_AR_USE_SYMM_MEM=1 ATOM_USE_CUSTOM_ALL_GATHER=0 ATOM_DP_LM_HEAD_MODE=default
export HSA_USE_SVM=1 HSA_ENABLE_SDMA=1 ATOM_LOADER_NUM_THREADS=4 TRITON_CACHE_DIR=/root/.triton/cache
export HSA_XNACK=$([ "$(cat /sys/module/amdgpu/parameters/noretry)" = 1 ] && echo 0 || echo 1)
export HSA_COREDUMP_PATTERN=/root/.cache/gpucore.%p ATOM_PROFILER_TIMEOUT=3600 ROCPROFILER_REGISTER_FORCE_LOAD=0
ONLINE_QUANT='{"global_quant_config": "per_block_fp8", "layer_quant_config": {"*self_attn.kv_b_proj": "ptpc_fp8"}, "exclude_layer": ["*self_attn.[qkv]_conv1d*", "*block_sparse_moe.experts*", "*block_sparse_moe.routed_expert_*"]}'
cd /tmp   # from /app, the aiter source tree would shadow the aiter package
python3 -m atom.entrypoints.openai_server \
  --model /models/Kimi-K3 --served-model-name moonshotai/Kimi-K3 --trust-remote-code --server-port 8000 \
  -tp 8 --tensor-parallel-size-local 4 --tensor-parallel-rank $RANK0 --decode-context-parallel-size 8 \
  --data-parallel-master-ip <node0 IP> --data-parallel-base-port 29700 --enable-expert-parallel \
  --kv_cache_dtype fp8 --cudagraph-mode FULL --max-num-seqs 64 --gpu-memory-utilization 0.90 \
  --max-num-batched-tokens 16384 --no-enable_prefix_caching --online_quant_config "$ONLINE_QUANT" \
  --torch-profiler-dir /root/.cache/torch_profile --disable_uvicorn_access_log
```

How the knobs change this command:
- `K3_MOE=tp` drops `--enable-expert-parallel`.
- `K3_MODE=agentx` replaces the two batching flags with `--max-num-batched-tokens 8192 --enable_prefix_caching
  --state-checkpoint-interval-tokens -1 --cudagraph-capture-sizes "[1,2,…,K3_MAX_SEQS]"`.
- `K3_DSPARK=N` adds `--method dspark --draft-model /models/Kimi-K3-DSpark --num-speculative-tokens N`, and sets
  `NCCL_GRAPH_REGISTER=0`.

Notes:
- **The `--data-parallel-*` flags** are the cross-node TP rendezvous. The DP size stays 1.
- **The dp8ep8 variables.** The launcher also exports the 20261016 dp8ep8 kit's MegaMoE and mori variables,
  unchanged, so that it reproduces the validated runs exactly. Nothing reads them at DP 1.
- **Online quantization:**
  - the BF16 layers become FP8 with E8M0 block scales;
  - `kv_b_proj` becomes per-token/per-channel FP8;
  - the KDA short convolutions and the MXFP4 experts stay as they are.

Check node0's log (`$WORK/cache/tp8.log`) for these lines:

| log line | what it confirms |
|---|---|
| `All 4 TP ranks on other nodes subscribed` | node1 joined |
| `Custom allreduce: using torch.symm_mem transport (mori backend)` | the custom all-reduce runs across the trays (the default) |
| `Custom allreduce is disabled because this process group spans across nodes` | every all-reduce runs on RCCL (`AITER_CUSTOM_AR_USE_SYMM_MEM=0`) |
| `607 layers online-quantized` | online quantization ran |
| no `no grouped CSV config matched` | with `K3_MOE=tp`: the MoE table has a row for every call |

**Stop** each tray by killing only this container's processes. `--pid=host` lets a plain `pkill` reach other
people's.

```bash
docker top k3tp8 -eo pid,args | awk '/openai_server|ATOM::/ {print $1}' | xargs -r kill
```

**After changing ATOM or aiter**, clear the compile caches before the next start:
- `rm -rf /root/.cache/atom/* /tmp/torchinductor_root`;
- after changing aiter, also its JIT modules: `rm -rf /app/aiter/aiter/jit/*.so /app/aiter/aiter/jit/build`.

---

## Accuracy

Inside node0's container, once `/health` answers:

```bash
python3 /opt/k3/gate.py http://127.0.0.1:8000                                     # prints GATE PASS
bash /opt/k3/gsm8k.sh http://127.0.0.1:8000 /root/.cache/results/gsm8k_<label>    # full GSM8K, about 8 min
```

`gsm8k.sh` runs lm_eval's `local-chat-completions`: 5-shot, with the chat template, `num_concurrent=32` and
`max_tokens=12288`. K3 reasons before it answers. With a smaller budget, `content` comes back empty and scores 0.

| server | trays, date | strict-match |
|---|---|---|
| this image, `K3_MOE=tp`, `AITER_CUSTOM_AR_USE_SYMM_MEM=0` (RCCL) | CT2 + CT17, 2026-10-22 | **0.9568 ±0.0056** |
| this image, `K3_MOE=tp`, custom all-reduce (the default) | CT2 + CT17, 2026-10-22 | 0.9492 ±0.0060 |
| the launcher on the 20261016 image, `K3_MOE=tp`, RCCL | CT11 + CT10, 2026-10-19 | 0.9621 ±0.0053 |
| the 20261016 image with this image's GEMM tables, `K3_MOE=ep`, RCCL | CT15 + CT13, 2026-10-21 | 0.9613–0.9629 |
| the [20261027 image](#the-20261027-image) with ATOM #2554 and #2557 overlaid, `K3_MOE=ep`, custom all-reduce | E15 05 + 07, 2026-10-27 | 0.9651 ±0.0051 |

The first two rows differ by less than run-to-run noise:
- Paired over the 1319 questions, 31 were wrong only with the custom all-reduce and 21 only with RCCL. The McNemar
  exact p-value is 0.21.
- About 4% of the answers change between any two runs.

---

## Throughput

This is ATOM's serving benchmark, run the way `scripts/run_benchmark.sh` runs it. Run it inside node0's container:

```bash
ISL=14336 OSL=1024 CONC=32
mkdir -p /root/.cache/results/bench && cd /tmp
python3 -m atom.benchmarks.benchmark_serving \
  --model=/models/Kimi-K3 --tokenizer=/models/Kimi-K3 --served-model-name=moonshotai/Kimi-K3 --trust-remote-code \
  --backend=vllm --base-url=http://127.0.0.1:8000 \
  --dataset-name=random --random-input-len=$ISL --random-output-len=$OSL --random-range-ratio=0.8 \
  --max-concurrency=$CONC --num-prompts=$((CONC * 10)) --num-warmups=$((CONC * 2)) --request-rate=inf --ignore-eos \
  --percentile-metrics=ttft,tpot,itl,e2el \
  --save-result --result-dir=/root/.cache/results/bench --result-filename=k3tp8_${ISL}-${OSL}-c${CONC}.json
# the same, wrapped: K3_LOGDIR=/root/.cache/results bash /opt/k3/bench.sh <label> $ISL $OSL $CONC
```

Measured with TP8 + DCP8, `K3_MOE=ep` and RCCL, at 14336/1024 and concurrency 32. These runs used the 20261016
image (CT15 + CT13, 2026-10-21):

| aiter tables | output tok/s | median TPOT | median ITL |
|---|---|---|---|
| none | 787.9 | 38.8 ms | 25.1 ms |
| this image's GEMM tables | **893.1** (+13.4%) | 34.5 ms | 21.2 ms |

Not measured yet on this image, with `K3_MOE=tp`, or with the custom all-reduce.

---

## AgentX

**Method: ATOM CI's agentic benchmark, unchanged.** It is the same method as the wide-EP recipe's
[Agentic (AgentX)](https://github.com/ROCm/ATOM/blob/recipe-k3-455-wideep/recipes/Kimi-K3-455-wideEP.md#agentic-agentx):
- **The script** is `.github/scripts/aiperf_agentic.sh`, run the way `.github/scripts/atom_test.sh` runs it. The
  image's copy, `/app/ATOM/.github/scripts/aiperf_agentic.sh`, is ATOM `main`'s, unchanged since `dcf790be` (#2387).
- **The client** is SemiAnalysis aiperf at the pinned commit `754356e9` (aiperf 0.12.0). See
  [Installing aiperf](#installing-aiperf).
- **The workload** is `semianalysis_cc_traces_weka_062126`: real Claude Code sessions, with a median ISL of about
  100k tokens. It is replayed with the scenario `inferencex-agentx-mvp`. The first run downloads the dataset (public,
  1.85 GB) into `HF_HOME`.

Defaults:
- **3600 s of profiling.**
  - `AIPERF_BENCHMARK_DURATION=1800` gives 30 min, as in the wide-EP recipe's 2026-10-16 points.
  - Under the scenario's 900 s floor, the script adds `--unsafe-override` and the run is stamped
    `submission_valid=false`: a smoke test, not a result.
- **10 warm-up requests per lane.**
- **No session headers.** `agentic_prepare` turns them on only when `SERVER_ARGS` contains `--enable-dp-attention`,
  and TP8 runs one engine.

### AgentX server

**What `K3_MODE=agentx` sets.** It runs the TP8 + DCP8 server options of [Agentic-Kimi-K3.md](Agentic-Kimi-K3.md),
which was written for MI355X:
- prefix caching;
- 8192 batched tokens;
- KDA state checkpoints only where a request resumes;
- a CUDA graph for every batch size up to `K3_MAX_SEQS`.

It leaves out the rest of that recipe: LMCache, `--block-size 128`, ReplaySSM, FlyDSL FP8 prefill attention, the
PrefillDelayer and synthetic acceptance.

**Start a fresh server for every concurrency,** with `K3_MAX_SEQS` = 2 × concurrency. The wide-EP recipe runs
concurrency 16, 32 and 64.

```bash
CONC=32   # 16, 32 or 64
docker exec -d -e K3_MODE=agentx -e K3_MAX_SEQS=$((2 * CONC)) k3tp8 \
  bash -c 'bash /opt/k3/k3_455_tp8dcp8_server.sh <0|4> <node0 IP> > /root/.cache/tp8_agentx.log 2>&1'
```

- **DSpark:** add `-e K3_DSPARK=<N>`, with the draft length that Agentic-Kimi-K3.md lists for the concurrency.
  - At concurrency 1–4 that recipe also runs DCP 1, which this launcher does not.
  - The launcher on the 20261016 image, with `K3_MOE=tp` and N=7, gave GSM8K strict-match 0.9545 and 4.75 tokens
    per forward.

### Installing aiperf

- **This image has it.**
  - It is in `/opt/aiperf-venv`, and on `PATH` as `/usr/local/bin/aiperf`. `aiperf --version` prints 0.12.0.
  - `aiperf_agentic.sh` uses any aiperf on `PATH` that is 0.12.0 or newer, and then installs nothing.
- **Without it,** `aiperf_agentic.sh` installs aiperf itself:
  - it clones `SemiAnalysisAI/aiperf` into `AIPERF_DIR` and checks out `754356e9`;
  - it pip-installs that checkout into a new venv at `AIPERF_VENV`.
  - The defaults are `/tmp/atom-aiperf` and `/tmp/atom-aiperf-venv`. Set both under `/root/.cache` to keep the
    install across containers, as the client command below does.
  - This needs a Python with `ensurepip`.
- **By hand,** the way the K3 images install it. Their Python has the venv module but no `ensurepip`, so pip comes
  from get-pip:

```bash
git clone https://github.com/SemiAnalysisAI/aiperf.git /root/.cache/aiperf-src
git -C /root/.cache/aiperf-src checkout --detach 754356e9a39acc6cc6afb242d123bb57c3fb6f75
python3 -m venv --without-pip /root/.cache/aiperf-venv
curl -sSfL https://bootstrap.pypa.io/get-pip.py | /root/.cache/aiperf-venv/bin/python - -q
/root/.cache/aiperf-venv/bin/pip install -q /root/.cache/aiperf-src
ln -sf /root/.cache/aiperf-venv/bin/aiperf /usr/local/bin/aiperf
aiperf --version     # 0.12.0
```

### AgentX client

Run this inside node0's container once `/health` answers. It is the wide-EP recipe's command, with TP8's
`SERVER_ARGS`:

```bash
export MODEL_PATH=/models/Kimi-K3 AIPERF_MODEL=moonshotai/Kimi-K3 HF_HOME=/root/.cache/huggingface
CONC=32   # the server's concurrency; it runs with K3_MAX_SEQS = 2 x CONC
# What the launcher runs in agentx mode; with K3_MOE=tp, drop --enable-expert-parallel
export SERVER_ARGS="--model /models/Kimi-K3 --served-model-name moonshotai/Kimi-K3 --trust-remote-code -tp 8 \
--tensor-parallel-size-local 4 --decode-context-parallel-size 8 --enable-expert-parallel \
--kv_cache_dtype fp8 --cudagraph-mode FULL --max-num-seqs $((2 * CONC)) --gpu-memory-utilization 0.90 \
--max-num-batched-tokens 8192 --enable_prefix_caching --state-checkpoint-interval-tokens -1 \
--disable_uvicorn_access_log"
export AIPERF_DIR=/root/.cache/aiperf-src AIPERF_VENV=/root/.cache/aiperf-venv   # used only without aiperf >= 0.12.0
OUT=/root/.cache/results/agentic/c${CONC}_$(date +%m%d-%H%M%S); mkdir -p $OUT; cd /tmp
source /app/ATOM/.github/scripts/aiperf_agentic.sh
agentic_prepare
run_aiperf_agentic http://127.0.0.1:8000 $CONC $OUT http://127.0.0.1:8000/metrics
python3 /app/ATOM/.github/scripts/atomesh/interactivity.py $OUT     # both P90 interactivities
```

- **`SERVER_ARGS`.** `agentic_prepare` reads it only to decide the session headers. It is set here to what the
  launcher runs.
- **Results, in `$OUT`:**
  - `profile_export_aiperf.json`: aiperf's summary (throughput, TTFT and ITL percentiles). `run_aiperf_agentic`
    returns non-zero when it is missing. That is the only reliable sign that a run produced nothing: aiperf itself
    exits 0 even when every request failed;
  - `profile_export.jsonl`: one record per request;
  - `aiperf.log`: the live stats.
- **Validity.** aiperf discards a run unless its last first token or its last completed response lands in the final
  5% of the window (`ProfileMetricCoverageError`, `submission_valid=False`).
- **Interactivity** follows ATOM's definitions in `.github/scripts/atomesh/interactivity.py`, computed per request
  from `profile_export.jsonl`:
  - E2E-normalized, the dashboard default: `1 / p90(ITL + TTFT/OSL)`;
  - `1 / p90(ITL)`, which is InferenceX's `p90_intvty`.

  Neither is aiperf's own per-user p90.
- **Throughput per GPU** is aiperf's `total_token_throughput` (input plus output tokens) over the 8 GPUs.
- **Not run end to end on this image yet.** On an earlier TP8 + DCP8 build with LMCache MP, a concurrency-128 run hung
  after about an hour; the cause is unknown.

---

## Profiling

**Server.** Start it with `-e ATOM_PROFILER_MORE=1` on both trays. That records shapes, Python stacks and memory.

**Client.** Run the [throughput](#throughput) command with three changes:
- add `--profile`;
- `--num-prompts=$CONC`, so the trace holds one wave of requests;
- `OSL=16` or `32`. A long OSL makes the trace too large to open.

The warm-ups run before the profiler starts.

**Traces:**
- Each TP rank writes its own trace on its own tray: `/root/.cache/torch_profile/rank_<r>/…pt.trace.json.gz`.
  Ranks 0–3 are on node0, ranks 4–7 on node1.
- Each tray's log prints the paths: `grep "trace exported to" /root/.cache/tp8.log`.
- Open the traces in [Perfetto](https://ui.perfetto.dev). Serve files over about 1 GB with `trace_processor --httpd`.

Restart without `ATOM_PROFILER_MORE` before you benchmark again.

---

## Known issues

| symptom | cause and fix |
|---|---|
| At startup, both trays hang in the first RCCL all-reduce (py-spy: `communicator_pynccl` → `stream.synchronize`) | The trays are not in one active UALink vpod. Check before every start; see [Before you start](#before-you-start) |
| With `K3_MOE=tp`, one tray or both go down at start, during load or under load | Seen eight times between 2026-10-19 and 10-22, with RCCL and with the custom all-reduce, on pairs inside and outside one vpod. The same configuration ran clean on CT2 + CT17 and on CT12 + CT2. Cause unknown. Watch the first minutes of a start |
| Pure TP8 (`--decode-context-parallel-size 1`) fails at warm-up with `store->get('0') got error: wait timeout after 300000ms` | One tray loaded more than 300 s later than the other, e.g. a 1 TB tray. The faster tray's first LM-head all-gather creates the TP communicator, and waits for its id with torch's default 300 s store timeout. Fixed in ATOM [#2647](https://github.com/ROCm/ATOM/pull/2647) (`c3cc2367`, `ATOM_MULTINODE_TP_TIMEOUT_S`), which the 20261022 image does not have and the [20261027 image](#the-20261027-image) does. The launcher always runs DCP 8: its post-load DCP all-gather creates the communicator while the ranks are still together |
| Custom all-reduce (`AITER_CUSTOM_AR_USE_SYMM_MEM=1`) | Correct (see [Accuracy](#accuracy)), but 2.4–4.5 µs slower per call than RCCL at decode sizes in a micro-benchmark. Twice, before 2026-10-21, both trays took a fatal CPU error at startup with it on. `=0` goes back to RCCL |
| `mxscale bpreshuffle (2d) M:…, N:1536, K:128 is untuned` | These DCP8 GEMMs, at decode M ≤ 64 and prefill M about 1100–1350, run a heuristic kernel. They are not in the tables yet |
| `K3_MOE=tp` at 2048–16384 tokens per MoE call | The image's grouped-MoE table has 15 rows. End to end, its 2048–16384 rows run at 0.97–1.03× the speed of the default tile. [#6768](https://github.com/ROCm/aiter/pull/6768) retunes all 15 buckets against the default tiles of #6663: 1.03–1.43× in a `fused_moe` timing with cold weights. It was merged into aiter `main` on 2026-10-27 and is in the [20261027 image](#the-20261027-image). The output is the same either way |
| GSM8K about 0.89 with `K3_MOE=tp` on another aiter | That aiter lacks `cc8d4ff12`: the default tile_k of 256 does not divide inter_dim 384, so the grouped GEMM drops the last 128 of K. This image's aiter has the fix |
| The [20261027 image](#the-20261027-image) fails in post-load weight processing: `AssertionError: 6284 % 16 == 12` | Its ATOM lacks #2554 and #2557, which the earlier images carried as overlays. Overlay both; see that section |
| The checkpoint's file-list md5 is `2773d607…`, not `fa7ad4b6…` | `sort` ran in the `C` locale. Use `LC_ALL=en_US.UTF-8 sort`; see [Before you start](#before-you-start) |
| Every token is `!` | A0 silicon without translation. See the wide-EP recipe's [A0 and B0](https://github.com/ROCm/ATOM/blob/recipe-k3-455-wideep/recipes/Kimi-K3-455-wideEP.md#a0-and-b0-silicon) |
| The ranks on the two trays cannot connect | The image's ENV sets `NCCL_SOCKET_IFNAME=lo GLOO_SOCKET_IFNAME=lo`. Set all three `*_SOCKET_IFNAME` to `enp1s0f1`, as the launcher does |
