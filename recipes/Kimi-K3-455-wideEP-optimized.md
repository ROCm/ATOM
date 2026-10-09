# Kimi-K3 MI455 B0 wide-EP optimized recipe

This is the reproducible optimization companion to
[Kimi-K3-455-wideEP.md](Kimi-K3-455-wideEP.md). It targets four MI455 B0
nodes, four GPUs per node, `TP=1`, `DP=16`, `EP=16`, and DP attention. DSpark
is not used.

The default recipe applies the formal PR stack and retained ATOM candidates.
It uses stock `ATOM_MORI_V2_FUSED=1`; that switch is existing infrastructure
and is not an optimization or gain claimed by this recipe. The separate
MegaMoE TDM experiment is opt-in only.

## Validation tiers

### A — correctness reference

Use the default patches, but disable AITER #6121 at runtime:

```bash
export ATOM_USE_FLYDSL_GATHER_KV_B_PROJ=0
export ATOM_UNFUSED_GATHER_KV_B_PROJ=0
```

This uses the upstream Triton 3.9 gather path. AITER #6120 is not present:
the exact production case passes `3/3` strict on current upstream Triton 3.9.

The preceding clean stack passed non-fake text `10/10` and full five-shot
GSM8K (`1319` samples): strict `0.9583 +/- 0.0055`. An earlier ptpc-only gate
scored strict `0.9598 +/- 0.0054`, flexible `0.9606 +/- 0.0054`.

Those runs did not contain AITER #6121 or the optional MegaMoE overlay. The
current bundle also replaces ATOM #2447 with AITER #6078 and the closed ATOM
top-k candidate with AITER #6131. Those replacements have focused correctness
coverage, but this exact rebuilt SHA stack still requires the gate below before
production.

### B — maximum-performance experimental

Enable all required patches, including AITER #6121:

```bash
export ATOM_USE_FLYDSL_GATHER_KV_B_PROJ=1
export ATOM_UNFUSED_GATHER_KV_B_PROJ=0
```

This tier has no valid full-stack GSM8K plus stable AgentX E2E result. It must
pass `10/10` non-fake text and full GSM8K before any performance run. Isolated
kernel gains must not be added together or presented as an E2E result.

The optional `--experimental-mori` overlay may be tested after Tier B passes,
but is not part of the recommended stack.

## Topology and health gate

All four nodes must share one `PPOD_ID` and `VPOD_ID`. Every GPU must report an
active UALink state, the rack must have no other containers or GPU processes,
and `/dev/kfd` must exist.

```bash
for n in <node0> <node1> <node2> <node3>; do
  ssh "$n" '
    test -e /dev/kfd
    sudo cat /sys/class/drm/card*/device/ualink/accel_state
    rocm-smi --showpids
    docker ps --format "{{.Names}}"
  '
done
```

Require all 16 `accel_state` values to be `active`. Also run the `ualoe_p2p`
and bandwidth checks described in the base recipe; an active state alone does
not prove cross-node reachability. Do not start if another user's container
exists, even when VRAM is still empty.

## Prepare exact sources

The manifest pins the fetched 2026-10-09 heads. Do not substitute a newer
`main`.

```bash
git clone git@github.com:ROCm/ATOM.git ATOM-base
git -C ATOM-base checkout --detach a526f0d557eeed08396670249af58bd85fa57338

git clone git@github.com:ROCm/aiter.git AITER-base
git -C AITER-base checkout --detach e6ded2168ef43111c57d591dd310eccdfc27aaf1

git clone --branch xiaobingsuper/kimi-k3-mi455-b0-recipe \
  git@github.com:ROCm/ATOM.git recipe
BUNDLE=$PWD/recipe/experiments/kimi_k3_b0/all_optimizations
"$BUNDLE/apply.sh" "$PWD/ATOM-base" "$PWD/AITER-base"
```

`apply.sh` first prints the imported Triton version and module path. It requires
a PEP 440/`packaging.version`-comparable version of at least 3.9, then verifies
both base commits and artifact checksums, runs `git apply --check`, applies the
patches, and installs the production-safe BF16 CSV. Re-running it is safe. If
the preflight reports Triton 3.8, upgrade Triton; do not restore #6120.

## Included formal PRs

- [ATOM #2458](https://github.com/ROCm/ATOM/pull/2458): multi-row K3 gated
  RMSNorm pipeline.
- [ATOM #2505](https://github.com/ROCm/ATOM/pull/2505): bind MXFP4
  producers and GEMMs through an explicit backend/layout contract.
- [AITER #6294](https://github.com/ROCm/aiter/pull/6294): define the
  row-major, AITER E8M0, and Opus F4 scale-layout ABI consumed by #2505.
- [AITER #6097](https://github.com/ROCm/aiter/pull/6097): already in the pinned
  AITER base; fixes gfx1250 MXFP4 TDM fusion stability.
- [AITER #6078](https://github.com/ROCm/aiter/pull/6078): use gfx1250's real
  large LDS capacity for K3 `D=33792` SiTUv2 quantization.
- [AITER #6115](https://github.com/ROCm/aiter/pull/6115): extend the fused G2L
  routing LUT to K3's 896 experts.
- [AITER #6121](https://github.com/ROCm/aiter/pull/6121): optional Tier-B
  FlyDSL cached-prefix gather/projection.
- [AITER #6130](https://github.com/ROCm/aiter/pull/6130): publish the gfx1250
  FlashKDA K2 schedule.
- [AITER #6131](https://github.com/ROCm/aiter/pull/6131): specialize the exact
  gfx1250 sigmoid top-k family.

ATOM #2447, the closed #2466 compatibility bridge, and the closed ATOM
grouped-top-k candidate are deliberately absent.
AITER #6120 is also absent: it was closed after Triton 3.9 passed the exact
production case `3/3` strict.

## Retained local ATOM candidates

`atom.patch` also carries KDA W8 launch geometry, causal-conv decode,
causal-conv prefill/fork metadata, guarded MLA split8, and AttentionResidual
long-prefill `BL=1`. Their focused tests are included in the patch.

## ptpc and BF16 configuration

The bundled `ptpc_online_experimental.json` enables ptpc for attention, dense
MLP, and shared experts. It excludes q/k/v convolution projections,
`f_b_proj` (`K=128`), the router, routed projections, and grouped experts.
Because #6121 is present in Tier B, `kv_b_proj` is intentionally not excluded.

```bash
export ATOM_USE_TRITON_GEMM=0
export AITER_CONFIG_GEMM_BF16="$PWD/AITER-base/aiter/configs/k3_bf16_hot_gfx1250_production_safe.csv"
export ONLINE_QUANT_CONFIG="$(tr -d '\n' < "$BUNDLE/ptpc_online_experimental.json")"
```

The K2 JSON, #6131 top-k dispatch, and required grouped-MoE behavior are in the
patched source/config paths and load automatically. `ATOM_MORI_V2_FUSED=1`
selects the stock fused Mori infrastructure; it is held equal across A/B and
is not counted as a gain.

## Exact server launch

Run the same command on each node with `DPRANK=0,4,8,12` respectively:

```bash
export DPRANK=<0|4|8|12>
export MASTER_IP=<node0-data-plane-ip>
export PYTORCH_ROCM_ARCH=gfx1250 AITER_RUNTIME_GPU_ARCH=gfx1250
export GPU_ARCHS=gfx1250 GPU_ARCH_LIST=gfx1250 MORI_GPU_ARCHS=gfx1250
export HSA_OVERRIDE_GFX_VERSION=12.5.0 ENABLE_CK=0

export ATOM_USE_TRITON_MLA=1 ATOM_USE_TRITON_MLA_SHUFFLE_KV=0
export ATOM_USE_AITER_TRITON_ATTN=1 ATOM_USE_UNIFIED_ATTN=1
export ATOM_USE_TRITON_GEMM=0 ATOM_WO_A_USE_FLYDSL=1
export ATOM_USE_FLYDSL_GATHER_KV_B_PROJ=1
export ATOM_UNFUSED_GATHER_KV_B_PROJ=0
export ATOM_FP8_BLOCKSCALE_USE_E8M0_SCALE=1

export ATOM_MOE_GU_ITLV=1 ATOM_USE_TRITON_MOE_DECODE=0
export MEGA_DISPATCH=mori MEGA_DISPATCH_WIRE=fp4
export ATOM_MORI_V2=1 ATOM_MORI_V2_FUSED=1
export AITER_USE_GROUPED_GEMM=1 AITER_USE_OPUS_MOE_SORTING=1

export NCCL_MNNVL_ENABLE=1 NCCL_IB_DISABLE=1 NCCL_P2P_DISABLE=0
export NCCL_P2P_LEVEL=SYS NCCL_CUMEM_ENABLE=1
export ATOM_DP_LM_HEAD_MODE=allgather
export ATOM_USE_CUSTOM_ALL_GATHER=1 AITER_CUSTOM_AR_USE_SYMM_MEM=1
export MORI_SOCKET_IFNAME=enp1s0f1
export NCCL_SOCKET_IFNAME=enp1s0f1 GLOO_SOCKET_IFNAME=enp1s0f1
export ATOM_DP_SESSION_AFFINITY=1
export HSA_XNACK=1 HSA_USE_SVM=1 HSA_ENABLE_SDMA=1
export ATOM_LOADER_USE_THREADPOOL=1 ATOM_LOADER_NUM_THREADS=4

python3 -m atom.entrypoints.openai_server \
  --model /models/Kimi-K3 \
  --served-model-name moonshotai/Kimi-K3 \
  --trust-remote-code -tp 1 \
  --data-parallel-size 16 --data-parallel-size-local 4 \
  --data-parallel-rank "$DPRANK" \
  --data-parallel-master-ip "$MASTER_IP" \
  --data-parallel-master-port 29500 --data-parallel-base-port 29700 \
  --enable-expert-parallel --enable-dp-attention \
  --kv_cache_dtype fp8 --index-cache-dtype fp8 \
  --cudagraph-mode FULL \
  --max-num-seqs 8 --max-num-batched-tokens 16384 \
  --gpu-memory-utilization 0.94 --enable_prefix_caching \
  --online_quant_config "$ONLINE_QUANT_CONFIG" \
  --disable_uvicorn_access_log
```

There is no `--method dspark`, draft model, or speculative-token argument.

## Mandatory accuracy gate

First run ten deterministic non-fake prompts and reject repeated-token,
non-finite, or empty answers. Then run the complete five-shot GSM8K gate:

```bash
lm_eval --model local-chat-completions --apply_chat_template \
  --tasks gsm8k --num_fewshot 5 \
  --model_args "model=moonshotai/Kimi-K3,base_url=http://<node0>:8000/v1/chat/completions,api_key=EMPTY,eos_string=</s>,max_retries=5,num_concurrent=32,timeout=1800,tokenized_requests=False,max_length=16384" \
  --gen_kwargs max_tokens=12288,temperature=0,top_p=1 \
  --output_path <gsm8k-output> --log_samples
```

Do not run AgentX unless both gates pass.

## Fixed AgentX con32 harness

Use `experiments/kimi_k3_b0/run_agentx.sh`. It fixes concurrency 32,
900 seconds, warmup 3/lane, seed 42, the public 393-trace dataset, and the
same cache policy. Compare A/B in the same rack window; do not report a result
that fails metric coverage.

## Optional MegaMoE TDM experiment

The five-file `experimental_mori_aiter.patch` changes AITER's MegaMoE
TDM tile selection, direct EP route, and counter reset behavior. It does not
modify Mori core and is not needed for stock fused Mori:

```bash
"$BUNDLE/apply.sh" --experimental-mori "$PWD/ATOM-base" "$PWD/AITER-base"
```

Evidence is provisional EP4/operator-level only. There is no stable 16-GPU
accuracy plus E2E result; prior testing included page-fault history under
overlap/high-concurrency conditions. Treat it as a fault-risk experiment,
capture dmesg watermarks, and never include it in the correctness reference,
formal PR stack, or default performance claim.
