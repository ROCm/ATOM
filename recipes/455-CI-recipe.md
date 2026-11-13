# 455-CI recipe

各模型在 455-CI 镜像（`rocm/fw-bringup-mlperf:gfx1250-atom-ci455-<YYYYMMDD>`，ATOM main + aiter main）上的 server、GSM8K 和 bench 命令。流程、指标口径和记录方式见 [`455-CI-guide.md`](455-CI-guide.md)；这里只写命令。

目前收录 Kimi-K3 (2.8T)、GLM-5.2、MiniMax-M3 (428B)、DeepSeek R1、DeepSeek-V4.1-Flash、DeepSeek-V4-Pro 六个模型，其余 2 个见 [`455-CI-guide.md`](455-CI-guide.md) §1。

## 通用约定

- 下面的命令都在测试容器里执行，路径都是容器内的路径。容器参数：

  ```bash
  docker run -d --name ci455-<slug> --network host --ipc host --shm-size 128g \
    --device /dev/kfd --device /dev/dri --group-add video --cap-add SYS_PTRACE \
    --security-opt seccomp=unconfined -v <模型目录>:/models:ro -v <结果目录>:/results \
    "$NEW_IMAGE"    # 镜像入口是 sleep infinity
  ```

- `MODEL_PATH` 是容器内的模型路径，也是 server 的 `--model`。没有加 `--served-model-name` 的模型，请求里的模型名就是它（ATOM 会校验这个名字）。
- 8K/1K 的 bench 都是 ATOM 的 `benchmark_serving`：随机数据，输入 8192、输出 1024，`--random-range-ratio 0.8`，请求数 `10 × CONC`，warmup `2 × CONC`（不计入结果），`--ignore-eos`。每卡吞吐按指南 §4.1 取 `total_token_throughput ÷ GPU 数`。
- "参考值"是 2026-11-11 摸底时测的，用的不是 455-CI 镜像，GSM8K 也不是 §3.3 的口径。只用来判断量级，不是 baseline。

---

## Kimi-K3 (2.8T)

- 场景：MLPerf Endpoint；GPU：8（2 node, 2 tray x 1P4G），两台在同一个 UALink vpod 里
- 并行：注意力 TP8 + DCP8，MoE TP8；一个 TP 组跨两台，node0 跑 TP rank 0–3 并对外服务，node1 跑 rank 4–7
- MODEL_PATH：`/models/Kimi-K3`（[`moonshotai/Kimi-K3`](https://huggingface.co/moonshotai/Kimi-K3) @ `f831ab66814297da540d832a5235f8e904f29d06`）；DSpark 草稿：`/models/Kimi-K3-DSpark`（[`Inferact/Kimi-K3-DSpark`](https://huggingface.co/Inferact/Kimi-K3-DSpark) @ `cf6b8244620e7ea4b0651d214f28e89eac75bed6`）。两台都要有，路径相同
- 对外模型名：`moonshotai/Kimi-K3`；PORT：`8000`（node0）

**当前 main 上跑不起来，预期记 N/A**：
- 一个 TP 组跨节点要 [ROCm/ATOM#2647](https://github.com/ROCm/ATOM/pull/2647)。合入前 main 的 argparse 不认识 `--tensor-parallel-size-local` 和 `--tensor-parallel-rank`，报 `unrecognized arguments`。
- gfx1250 上 DSpark 和 DCP 同时开要 [ROCm/ATOM#2760](https://github.com/ROCm/ATOM/pull/2760)（叠在 #2647 上）。合入前 `atom/config.py` 会断言 `Speculative decode + DCP is only supported on gfx950`。

命令取自 `rocm/fw-bringup:gfx1250-atom-k3-20261110` 镜像里的 K3 MLPerf 复现手册（`/opt/k3/k3-mlperf-repro/recipe.md`），去掉了只在 DP > 1 时生效的 `ATOM_MEGA_*`、`MORI_*`、`ATOM_DP_*` 和采 profile 用的变量。c4 和 c64 用同一个 server，按 c64 的点配置：max-num-seqs 64，DSpark 4 个草稿 token。手册的点是 c8 到 c512，c4 不在其中，用这个配置跑 c4 还没有验证过。

### server

两台都先关 NUMA balancing（`echo 0 > /proc/sys/kernel/numa_balancing`，在宿主机上执行）。1 TB 内存的 tray 起 server 前再执行 `sync; echo 1 > /proc/sys/vm/drop_caches`，并打开下面的 `ATOM_LOADER_PREFETCH=false`。

```bash
# 两台相同，只有 RANK0 不同：node0 填 0，node1 填 4
RANK0=0
NODE0_IP=<node0 的数据面 IP>
NIC=<两台之间的数据面网卡>            # 镜像 ENV 里的 lo 只适合单机，必须改

export PYTORCH_ROCM_ARCH=gfx1250 GPU_ARCHS=gfx1250 GPU_ARCH_LIST=gfx1250 MORI_GPU_ARCHS=gfx1250 GPU_TARGETS=gfx1250
export HSA_OVERRIDE_GFX_VERSION=12.5.0 ENABLE_CK=0
export HSA_USE_SVM=1 HSA_ENABLE_SDMA=1 HSA_XNACK=1 ATOM_LOADER_NUM_THREADS=4
# export ATOM_LOADER_PREFETCH=false    # 1 TB 内存的 tray
export ATOM_USE_TRITON_MLA=0 AITER_MLA_DECODE_PS1_FLYDSL=1 ATOM_USE_TRITON_MLA_SHUFFLE_KV=0 ATOM_USE_UNIFIED_ATTN=1
export ATOM_USE_ASM_FP8_PREFILL_ATTN=1
export ATOM_TOKEN_GRAPH_SIZES=256,512,1024,1536,2048,3072,4096,5120,6144,7168,8192,10240,12288,14336,16384
export ATOM_MOE_GU_ITLV=1 ATOM_USE_TRITON_MOE_DECODE=0 AITER_USE_GROUPED_GEMM=1 AITER_FLYDSL_USE_HERD=1 ATOM_MONO_ENABLE=0
export ATOM_FP8_BLOCKSCALE_WEIGHT_PRESHUFFLE=1 ATOM_FP8_BLOCKSCALE_USE_E8M0_SCALE=1 ATOM_FP4_MXSCALE_USE_32X4_SCALE_SHUFFLE=1
export AITER_ROPE_TRITON_BACKEND=1 AITER_USE_SYSTEM_TRITON=1
export NCCL_MNNVL_ENABLE=1 NCCL_IB_DISABLE=1 NCCL_P2P_DISABLE=0 NCCL_P2P_LEVEL=SYS NCCL_CUMEM_ENABLE=1
export MORI_SOCKET_IFNAME=$NIC NCCL_SOCKET_IFNAME=$NIC GLOO_SOCKET_IFNAME=$NIC
export AITER_CUSTOM_AR_USE_SYMM_MEM=1 AITER_CUSTOM_AR_MAX_SIZE=8388608
export RCCL_DDA_LL_ONESHOT_THRESHOLD=2097152 RCCL_DDA_LL_TWOSHOT_THRESHOLD=134217728
export ATOM_USE_CUSTOM_ALL_GATHER=0
export NCCL_GRAPH_REGISTER=0          # 必需：否则 decode CUDA graph 录制时 RCCL 报 invalid argument

M=64
cd /tmp
python3 -m atom.entrypoints.openai_server \
  --model /models/Kimi-K3 --served-model-name moonshotai/Kimi-K3 --trust-remote-code --server-port 8000 \
  -tp 8 --tensor-parallel-size-local 4 --tensor-parallel-rank $RANK0 --decode-context-parallel-size 8 \
  --data-parallel-master-ip $NODE0_IP --data-parallel-base-port 29700 \
  --kv_cache_dtype fp8 --block-size 128 --cudagraph-mode FULL --gpu-memory-utilization 0.90 \
  --max-num-seqs $M --cudagraph-capture-sizes "[$(seq -s, 1 $M)]" \
  --max-num-batched-tokens 16384 --enable_prefix_caching --state-checkpoint-interval-tokens -1 \
  --method dspark --draft-model /models/Kimi-K3-DSpark --num-speculative-tokens 4 \
  --online_quant_config '{"global_quant_config": "per_block_fp8", "layer_quant_config": {"*self_attn.kv_b_proj": "ptpc_fp8"}, "exclude_layer": ["*self_attn.[qkv]_conv1d*", "*block_sparse_moe.experts*", "*block_sparse_moe.routed_expert_*"]}' \
  --disable_uvicorn_access_log
```

两台先起哪台都行，会互相等。只在 node0 上查 `curl -sf http://$NODE0_IP:8000/v1/models`。

### gsm8k

和指南 §3.3 不同的地方有三处：
- 请求里的模型名是 `moonshotai/Kimi-K3`，tokenizer 指向本地 checkpoint；
- K3 先思考再回答，要走 chat 接口，`max_tokens` 至少 12288，否则思考没写完，`content` 是空的，这题记 0 分；
- 5-shot，和 K3 复现手册的 baseline（strict-match 0.9659）一致。

```bash
lm_eval --model local-chat-completions --apply_chat_template --tasks gsm8k --num_fewshot 5 \
  --model_args "model=moonshotai/Kimi-K3,tokenizer=/models/Kimi-K3,base_url=http://127.0.0.1:8000/v1/chat/completions,api_key=EMPTY,eos_string=</s>,max_retries=5,num_concurrent=32,timeout=1800,tokenized_requests=False,max_length=16384" \
  --gen_kwargs max_tokens=12288,temperature=0,top_p=1 --output_path "$ACC_DIR" > "$ACC_DIR/lm_eval.log" 2>&1
```

accuracy.csv 的 `num_fewshot` 写 `5`。

### bench

client 是官方的 [`mlcommons/endpoints`](https://github.com/mlcommons/endpoints) @ `4b385eb`，不改代码，装在 node0 的宿主机上（不需要 GPU）。数据集按 sha256 校验：

```bash
CLIENT=<node0 上的目录>
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | env UV_NO_MODIFY_PATH=1 sh
git clone -q https://github.com/mlcommons/endpoints.git $CLIENT/endpoints
git -C $CLIENT/endpoints checkout -q --detach 4b385eb
(cd $CLIENT/endpoints && UV_FROZEN=1 uv sync --frozen)
mkdir -p $CLIENT/data
curl -fsSL -o $CLIENT/data/mlperf_agentic_inference_dataset.jsonl \
  https://endpoints.mlcommons-storage.org/mlperf_agentic_inference_dataset/mlperf_agentic_inference_dataset.jsonl
echo "1beb24c882122df96571cf11b390acbea388944038bc55c78b891475459014ae  $CLIENT/data/mlperf_agentic_inference_dataset.jsonl" | sha256sum -c -
```

每个并发先 warmup，再正式测 10 分钟。warmup 用另一组 seed：C < 16 时 2C 条轨迹，否则 C 条，第一个 user 跑完就停。正式测试用 seed set A，613 条轨迹，发 10 分钟请求。`TOKENIZER` 是 node0 宿主机上 `Kimi-K3` 目录的路径。

```bash
# point_yaml <name> <并发> <轨迹数> <cap 或 -> <submission|warmup> <report_dir>
point_yaml() {
  cat << YAML
name: "$1"
version: "1.0"
type: "online"
YAML
  [ "$5" = submission ] && printf 'submission_ref:\n  model: moonshotai/Kimi-K3\n  ruleset: mlperf-endpoints-v1.0-2026-10-C1-A\n'
  cat << YAML
model_params:
  name: "moonshotai/Kimi-K3"
  tokenizer_name: "$TOKENIZER"
  temperature: 1.0
  top_p: 1.0
  max_new_tokens: 8192
  streaming: "on"
datasets:
  - name: agentic_combined
    type: performance
    path: $CLIENT/data/mlperf_agentic_inference_dataset.jsonl
    accuracy_config:
      eval_method: agentic_inference_inline
      num_repeats: 1
    agentic_inference:
      turn_timeout_s: 86400.0
      enable_salt: true
      inject_tool_delay: true
      num_trajectories_to_issue: $3
      stop_issuing_on_first_user_complete: true
settings:
YAML
  if [ "$4" != - ] || [ "$5" = warmup ]; then
    echo "  runtime:"
    [ "$4" != - ] && echo "    max_issue_duration_ms: $4"
    [ "$5" = warmup ] && printf '    scheduler_random_seed: 1\n    dataloader_random_seed: 2\n'
  fi
  cat << YAML
  load_pattern:
    type: agentic_inference
    target_concurrency: $2
  client:
    warmup_connections: 0
    max_idle_time: 0.5
endpoint_config:
  endpoints: ["http://$NODE0_IP:8000"]
  api_type: openai
report_dir: $6
YAML
}
run_point() { (ulimit -n 524288 && cd $CLIENT/endpoints && UV_FROZEN=1 uv run inference-endpoint benchmark from-config --config "$1" --mode perf); }

NW=$(( CONC < 16 ? 2 * CONC : CONC ))
point_yaml k3-c${CONC}-warmup $CONC $NW - warmup $RESULT_DIR/warmup > $RESULT_DIR/warmup.yaml
point_yaml k3-c${CONC} $CONC 613 10m submission $RESULT_DIR > $RESULT_DIR/run.yaml
run_point $RESULT_DIR/warmup.yaml > $RESULT_DIR/warmup.log 2>&1
run_point $RESULT_DIR/run.yaml > $RESULT_DIR/bench.log 2>&1
```

结果在 `$RESULT_DIR/performance/result_summary.json`。

---

## GLM-5.2

- 场景：8K/1K；GPU：4（1 tray x 1P4G）
- 并行：TP4；FP8 KV cache；非 expert 层在线量化成 ptpc_fp8，不开 MTP
- MODEL_PATH：`/models/GLM-5.2-MXFP4`（[`amd/GLM-5.2-MXFP4`](https://huggingface.co/amd/GLM-5.2-MXFP4) @ `386bd0e4ec821f7b07975701cec3c3b953a5576a`）；PORT：`8000`

**当前 main 上跑不起来，预期记 N/A**：CUDA graph 录制时 4 个 rank 都报 `ValueError: unexpected KV cache shape (<blocks>, 16, 1, 144)`。ATOM 把 FP8 index 行补齐到 144 字节，aiter 的 gfx1250 FP8 paged MQA logits kernel 只接受 132 字节。修复是 [ROCm/aiter#7812](https://github.com/ROCm/aiter/pull/7812)，合入前加 `--index_cache_dtype fp4` 也绕不开：gfx1250 上 ATOM 会退回 FP8。

命令取自 ATOM main 的 [`recipes/GLM-5.md`](GLM-5.md)（GLM-5.2 MXFP4 Server）。gfx1250 另加两项选 kernel 的环境变量：
- `ATOM_MOE_GU_ITLV=1`：gfx1250 的 fused_moe 只接受 gate/up 交错的权重布局；
- `AITER_MLA_DECODE_PS1_FLYDSL=1`：gfx1250 的 ASM MLA decode 只支持 non-persistent，要换成 FlyDSL PS1 decode。

### server

```bash
export AITER_QUICK_REDUCE_QUANTIZATION=INT4 AITER_USE_FLYDSL_MOE_SORTING=1
export PYTORCH_ROCM_ARCH=gfx1250 GPU_ARCHS=gfx1250 HSA_OVERRIDE_GFX_VERSION=12.5.0
export ATOM_MOE_GU_ITLV=1 AITER_MLA_DECODE_PS1_FLYDSL=1 AITER_ROPE_TRITON_BACKEND=1 AITER_USE_SYSTEM_TRITON=1
export ATOM_LOADER_NUM_THREADS=4         # 默认 16；曾有 tray 在加载权重时整机报 CPU ProcessorError，降到 4 后没再出现
cd /tmp
python3 -m atom.entrypoints.openai_server \
  --model /models/GLM-5.2-MXFP4 --server-port 8000 -tp 4 \
  --kv_cache_dtype fp8 --no-enable_prefix_caching \
  --online_quant_config '{"global_quant_config": "ptpc_fp8", "exclude_layer": ["lm_head", "model.embed_tokens", "*.mlp.gate", "*expert*"]}'
```

`--max-num-seqs`（512）和 CUDA graph 尺寸用 ATOM 默认值，c4 和 c64 用同一个 server。

### gsm8k

按指南 §3.3，`MODEL_PATH=/models/GLM-5.2-MXFP4`，`PORT=8000`。

### bench

```bash
python3 -m atom.benchmarks.benchmark_serving \
  --model=/models/GLM-5.2-MXFP4 --backend=vllm --base-url=http://127.0.0.1:8000 \
  --dataset-name=random --random-input-len=8192 --random-output-len=1024 --random-range-ratio=0.8 \
  --max-concurrency=$CONC --num-prompts=$((CONC * 10)) --num-warmups=$((CONC * 2)) \
  --request-rate=inf --ignore-eos --trust-remote-code \
  --save-result --result-dir=$RESULT_DIR --result-filename=raw.json \
  --percentile-metrics="ttft,tpot,itl,e2el"
```

### 已知问题

- Triton 低于 3.9 时还有两处会挂。`gfx1250-atom-20261109-dsv41` 带的是 3.9.0.dev20261022；`gfx1250-atom-20261111` 没有逐个核对，按同一系列推断也是 3.9。
  - ATOM main 默认在 gfx1250 TP4 上打开 aiter 的 Gluon all-reduce（`ATOM_ENABLE_GLUON_FUSED_AR=1`），Triton 3.8 编译它会报 `CompilationError`；
  - GSM8K 一开始的大 prefill 会在 Triton 3.8 上触发 LLVM 断言 `PHINode ... same type`，rank 0 退出，API 进程还在，lm_eval 一直等到超时。
- `recipes/GLM-5.md` 里那条 GSM8K 命令照抄跑不通：`//v1` 返回 404，`local-completions` 发到 chat 接口返回 422。用指南 §3.3 的命令。

### 参考值

2026-11-11，K3 1110 镜像换成 main，加 aiter#7812，`ATOM_ENABLE_GLUON_FUSED_AR=0`，Triton 3.9：
- GSM8K：chat、5-shot，strict 0.9644 / flexible 0.9636；
- 8K/1K 每卡吞吐：c64 3,504 tok/s/GPU，c4 516 tok/s/GPU；
- MI355X 上同一 recipe 的 c64 是 2,934 tok/s/GPU（11,734 ÷ 4，来自 `recipes/GLM-5.md`）。

---

## MiniMax-M3 (428B)

- 场景：8K/1K；GPU：4（1 tray x 1P4G）
- 并行：TP4 + EAGLE3（3 个草稿 token，真实接受率）；FP8 KV cache；非 MoE 层在线量化成 ptpc_fp8
- MODEL_PATH：`/models/MiniMax-M3-MXFP4`（`amd/MiniMax-M3-MXFP4`）；草稿：`/models/MiniMax-M3-EAGLE3-GQA`（`Inferact/MiniMax-M3-EAGLE3-GQA`）；PORT：`8890`（API）、`8896`（engine）

命令取自 `rocm/fw-bringup:gfx1250-minimax-m3-20261110` 镜像里的 `/app/endpoints-recipe-minimax-m3/server.sh`，参数原样，按并发 64 配置：max-num-seqs 128，开 indexer CP。c4 和 c64 用同一个 server。main 上没有 M3 的 prefill token graph，`ATOM_MINIMAX_M3_PREFILL_TOKEN_GRAPH` 不设。

### server

```bash
export NCCL_IB_DISABLE=1 RCCL_IB_DISABLE=1 AITER_QUICK_REDUCE_QUANTIZATION=INT4 ATOM_FORCE_ATTN_TRITON=1
export ATOM_GC_THRESHOLD=20000,50,50 ATOM_PA_FLYDSL=1
export PYTORCH_ROCM_ARCH=gfx1250 GPU_ARCHS=gfx1250 HSA_OVERRIDE_GFX_VERSION=12.5.0
export ATOM_MOE_GU_ITLV=1 AITER_ROPE_TRITON_BACKEND=1 AITER_USE_SYSTEM_TRITON=1 AITER_QUICK_REDUCE_CAST_BF16_TO_FP16=1
export ATOM_MONO_ENABLE=0                  # gfx1250 上开 mono 时 decode 一步要几百 ms
export ATOM_M3_INDEXER_CP=1                # 并发 15 及以上用
# ATOM main 的 loader 通过全局符号表调用 omp_set_num_threads，不预加载 libgomp 时加载权重会让每个 rank 退出
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libgomp.so.1
cd /tmp
python3 -u -m atom.entrypoints.openai_server \
  --model /models/MiniMax-M3-MXFP4 --served-model-name /models/MiniMax-M3-MXFP4 \
  --host 0.0.0.0 --port 8896 --server-port 8890 --tensor-parallel-size 4 --trust-remote-code \
  --kv_cache_dtype fp8 --gpu-memory-utilization 0.95 --block-size 128 \
  --max-num-batched-tokens 32768 --attn-prefill-chunk-size 16384 --max-num-seqs 128 \
  --online_quant_config '{"global_quant_config": "ptpc_fp8", "exclude_layer": ["lm_head", "model.embed_tokens", "vision_tower", "multi_modal_projector", "patch_merge_mlp", "*block_sparse_moe"]}' \
  --default-chat-template-kwargs '{"thinking_mode": "enabled"}' \
  --enable-prefix-caching \
  --method eagle3 --draft-model /models/MiniMax-M3-EAGLE3-GQA --num-speculative-tokens 3 \
  --cudagraph-capture-sizes '[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,20,22,24,26,28,30,32,34,36,40,48,56,64]'
```

### gsm8k

按指南 §3.3，`MODEL_PATH=/models/MiniMax-M3-MXFP4`，`PORT=8890`。

### bench

```bash
python3 -m atom.benchmarks.benchmark_serving \
  --model=/models/MiniMax-M3-MXFP4 --tokenizer=/models/MiniMax-M3-MXFP4 \
  --served-model-name=/models/MiniMax-M3-MXFP4 --backend=vllm --base-url=http://127.0.0.1:8890 \
  --dataset-name=random --random-input-len=8192 --random-output-len=1024 --random-range-ratio=0.8 \
  --max-concurrency=$CONC --num-prompts=$((CONC * 10)) --num-warmups=$((CONC * 2)) \
  --request-rate=inf --ignore-eos --trust-remote-code \
  --save-result --result-dir=$RESULT_DIR --result-filename=raw.json \
  --percentile-metrics="ttft,tpot,itl,e2el"
```

结果 JSON 里的 `acceptance_rate` 是 server 启动以来的累计值（读的是 `/debug/mtp_stats`），会把 GSM8K 和前一个并发都算进去。要看单个并发的接受率，就在 bench 前后各读一次 `curl -s http://127.0.0.1:8890/debug/mtp_stats`，用 `total_accepted_tokens / total_draft_tokens` 的差值算。

### 参考值

2026-11-11，M3 1110 镜像换成 main（不打补丁）：
- GSM8K：chat、5-shot，strict 0.9469 / flexible 0.9568；
- 8K/1K 每卡吞吐：c64 11,237 tok/s/GPU，c4 3,331 tok/s/GPU；
- 随机数据下 EAGLE3 接受率：c64 68.8%，c4 62.8%。

---

## DeepSeek R1

- 场景：MLPerf Endpoint；GPU：4（1 tray x 1P4G）
- 并行：注意力 DP4（DP attention），MoE EP4；FP8 KV cache，block 64；不开 prefix cache 和 chunked prefill；`q_b_proj` 在线量化成 MXFP4
- MODEL_PATH：`/models/Deepseek-S3_sq_a05_v2_mlperf6_1/`（[`amd/Deepseek-S3_sq_a05_v2_mlperf6_1`](https://huggingface.co/amd/Deepseek-S3_sq_a05_v2_mlperf6_1) @ `1415311f5a898bc5fb0579612edacf4d4eb75bed`，MLPerf v6.1 closed division 的 Quark MXFP4 权重，455-CI 固定用它；结尾的 `/` 是模型名的一部分，lm_eval 和 bench 要写成一样）；PORT：`8000`

**旧权重 `DeepSeek-R1-0528-MXFP4-MTP-MoEFP4` 在 main 上能跑**（2026-11-13 复测，换 MLPerf v6.1 权重之前）。偶发问题：2026-11-12 在 main 上第一次跑时，GSM8K 开始后 rank 0 退出过一次，报 `RuntimeError: Triton Error [HIP]: Code: 200, Message: device kernel image is invalid`。出错的是 o_proj 的 aiter Triton `gemm_a8w8`（`atom/model_ops/linear.py:615`）现场 JIT 出的一个新 M 档位 kernel，在加载（`hipModuleLoadDataEx`）时失败。之后同样的代码和命令又跑了两遍 GSM8K 和一轮完整性能测试，都没再出现，同一个 kernel 也能正常加载。这个 recipe 用 `--gpu-memory-utilization 0.975`，c64 时每卡空闲显存只剩 1–2 GiB，推断是新 kernel 加载时正好显存不够，没有证实。再遇到时按指南 §3.4 记失败，`note` 写明是这个问题。旧权重的 o_proj 是 8-bit（走这个 `gemm_a8w8`），MLPerf v6.1 权重的 o_proj 是 MXFP4，新权重上还没验证过。

命令取自 `gfx1250-atom-2026111x` 镜像里的 `/app/scripts/dsr1/serve_dsr1fp4_a8w8_dp4ep4_realEPLB.sh`，模型换成了上面的 MLPerf v6.1 权重，其余参数原样。这些参数是按旧权重整理的，第一次用新权重时先确认 server 起得来、GSM8K 正常。不用同目录的 `serve_dsr1fp4_a8w8_dp4ep4.sh`：它默认 `FAKE_EPLB=1`，加 `--fake-eplb --hf-overrides '{"num_hash_layers": 0}'`，假路由输出乱码，GSM8K 没法测。c4 和 c64 用同一个 server。

### server

```bash
env ATOM_MLA_ABSORB_BF16_FUSE_QOUT=1 ATOM_MLA_KV_SHUFFLED=1 ATOM_MEGA_MASK_PAD_ROWS=1 MEGA_COMBINE_WIRE=bf16 \
  AITER_DISABLE_KERNARG_PRELOAD=1 HSA_ENABLE_SDMA=1 HSA_USE_SVM=1 HSA_XNACK=1 ATOM_DP_LM_HEAD_MODE=allgather \
  ATOM_USE_UNIFIED_ATTN=1 ATOM_USE_TRITON_MLA=0 ATOM_MLA_PAGE_SIZE=1 AITER_MLA_DECODE_PS1_FLYDSL=1 \
  AITER_ROPE_TRITON_BACKEND=1 ATOM_ENABLE_DS_QKNORM_FUSION=0 ATOM_LOADER_NUM_THREADS=4 \
  ATOM_ENABLE_DS_INPUT_RMSNORM_QUANT_FUSION=0 ATOM_ENABLE_DS_QKNORM_QUANT_FUSION=0 ATOM_USE_TRITON_GEMM=1 \
  ATOM_USE_TRITON_MLA_SHUFFLE_KV=0 ENABLE_CK=0 AITER_USE_GROUPED_GEMM=1 AITER_FORCE_A8W4=0 \
  AITER_USE_OPUS_MOE_SORTING=1 ATOM_MOE_GU_ITLV=1 ATOM_MORI_V2=1 ATOM_MORI_V2_FUSED=1 ATOM_MEGA_STAGE1_FUSED=0 \
  MEGA_DISPATCH=mori MORI_GPU_ARCHS=gfx1250 FLYDSL_GPU_ARCH=gfx1250 AITER_FLYDSL_GATHER_REDUCE_WAVE_MIN_TOKENS=-1 \
  ATOM_GC_THRESHOLD=20000,50,50 MEGA_DISPATCH_WIRE=fp4 \
python3 -m atom.entrypoints.openai_server -tp 4 --model /models/Deepseek-S3_sq_a05_v2_mlperf6_1/ \
  --kv_cache_dtype fp8 --block-size 64 --no-enable_prefix_caching --no-enable-chunked-prefill \
  --enable-expert-parallel --enable-dp-attention \
  --online_quant_config '{"layer_quant_config":{"*.self_attn.q_b_proj":"mxfp4"}}' \
  --max-num-seqs 2048 --gpu-memory-utilization 0.975 \
  --cudagraph-capture-sizes '[1,2,4,8,16,32,48,64,128,256,512,1024,1536,2048]'
```

### gsm8k

和指南 §3.3 不同的地方：
- 加 `max_gen_toks=2048`，和镜像里的 `/app/scripts/dsr1/eval_dsr1fp4.sh` 一致；
- 镜像 ENV 里有 `HF_HUB_OFFLINE=1`，lm_eval 拉不到 `openai/gsm8k`，要加 `HF_HUB_OFFLINE=0`。

```bash
HF_HUB_OFFLINE=0 lm_eval --model local-completions \
  --model_args "model=/models/Deepseek-S3_sq_a05_v2_mlperf6_1/,base_url=http://127.0.0.1:8000/v1/completions,num_concurrent=65,max_retries=3,max_gen_toks=2048,tokenized_requests=False,trust_remote_code=True" \
  --tasks gsm8k --num_fewshot 3 --output_path "$ACC_DIR" > "$ACC_DIR/lm_eval.log" 2>&1
```

### bench

client 用镜像里的 `/app/mlcommons-endpoints`（`mlcommons/endpoints` @ `d3c57f2`，自带 venv）。数据集是 MLPerf DeepSeek-R1 的 eval parquet（4388 条）。镜像里那份 `examples/07_DeepSeekR1_Example/data/deepseek_r1_eval.parquet` 只是 git-lfs 指针，要另外下载并校验：

```bash
EP=/app/mlcommons-endpoints
DSR1=<数据目录>/deepseek_r1_eval.parquet
curl -fsSL -o $DSR1 https://media.githubusercontent.com/media/mlcommons/endpoints/d3c57f2/examples/07_DeepSeekR1_Example/data/deepseek_r1_eval.parquet
echo "36facfae18f42587b7e4863e16a0e96446470761017d207791e609537d0f6f02  $DSR1" | sha256sum -c -
```

固定并发，流式，发 10 分钟请求。命令取自镜像 `/app/endpoints-recipe/06_bench_concurrency.sh`，另加 10 分钟上限。warmup 发 `CONC` 条、输出上限 2048，不计入结果。

```bash
export LEGACY_MLPERF_DEEPSEEK_R1_DATASET=$DSR1
rm -rf $EP/dataset_cache/legacy_mlperf_deepseek_r1    # 缓存不跟数据集路径走，每次清
cd $EP
run_r1() {  # run_r1 <样本数> <输出上限> <report_dir>
  $EP/venv/bin/inference-endpoint benchmark online --endpoints http://127.0.0.1:8000 \
    --model /models/Deepseek-S3_sq_a05_v2_mlperf6_1/ --tokenizer /models/Deepseek-S3_sq_a05_v2_mlperf6_1/ \
    --api-type openai_completions --dataset legacy_mlperf_deepseek_r1 --mode perf \
    --load-pattern concurrency --concurrency $CONC --num-samples $1 --streaming on \
    --max-output-tokens $2 --runtime.max-issue-duration-ms 600000 --timeout 14400 --report-dir $3
}
run_r1 $CONC 2048 $RESULT_DIR/warmup > $RESULT_DIR/warmup.log 2>&1
run_r1 4388 20000 $RESULT_DIR/report > $RESULT_DIR/bench.log 2>&1
```

结果在 `$RESULT_DIR/report/performance/result_summary.json`。R1 的回答很长，c4 十分钟只能完成二十来条，发请求停止后还要等在途请求跑完，c64 一个点约 15 分钟。

### 参考值

旧权重 `DeepSeek-R1-0528-MXFP4-MTP-MoEFP4`，同一台机器，其余参数和上面的命令相同。换成 MLPerf v6.1 权重后数字会变，只能看量级：

| | GSM8K flexible / strict | c4 tok/s/GPU | c4 interactivity | c64 tok/s/GPU | c64 interactivity |
|---|---|---|---|---|---|
| 2026-11-13，main（ATOM `f7b8962fd`、aiter `65714259c`） | 0.9538 / 0.9538 | 69.47（277.86 ÷ 4） | 74.57 | 702.71（2,810.84 ÷ 4） | 55.91 |
| 2026-11-12，`gfx1250-atom-20261111` 原样（私有分支 `jiaolyu/duty1111-main-rebase`） | 0.9522 / 0.9522 | 66.90（267.59 ÷ 4） | 71.20 | 679.99（2,719.95 ÷ 4） | 53.76 |

- tok/s/GPU 是 client 的 `tps` ÷ 4，interactivity 是 `e2e_avg_interactivity`（tok/s/user）。
- c4 两边各完成 20 条；c64 main 362 条，镜像 353 条。每个点只跑了一次。
- main 上 GSM8K 另一次复测是 0.9545。

---

## DeepSeek-V4.1-Flash

- 场景：MLPerf Endpoint；GPU：4（1 tray x 1P4G）
- MODEL_PATH：`/models/DeepSeek-V4.1-Flash`（`deepseek-ai/DeepSeek-V4.1-Flash` @ `dba1be0a40aa45a94ad051997016db3960a90277`）；PORT：`8000`
- c4 和 c64 用两个不同的 server，和镜像里 `/app/endpoints-recipe-dsv41/run_ep.py` 的分法一致：c4 用 TP4（它 c8 的配置），c64 用 DEP4（EP4 + DP attention，`--max-num-seqs 64`）。GSM8K 在两个 server 上各测一次，accuracy.csv 记两行，`note` 写 TP4 / DEP4。
- 宿主机要设 `echo Y > /sys/module/amdgpu/parameters/no_system_mem_limit`（DEP4 的 `ATOM_ENGRAM_UVA=1` 依赖它，机器重启后复位）。

**当前 main 上 c64 跑不起来，预期记 N/A**：DEP4 用 FP4 KV cache，main 的 argparse 直接退出：`invalid choice: 'fp4' (choose from 'bf16', 'fp8')`。FP4 KV 在 [ROCm/ATOM#2989](https://github.com/ROCm/ATOM/pull/2989)，gfx1250 上的 A8W4 attention 还要 [ROCm/aiter#7407](https://github.com/ROCm/aiter/pull/7407)，两个都没合。另外 DEP4 用的下面这些环境变量在 main 上没有代码读，等于没开：`ATOM_DSPARK_PROB_DRAFT`、`ATOM_DSPARK_BLOCK_VERIFY`（ROCm/ATOM#2724）、`ATOM_DSPARK_PROB_FUSED_CHAIN`（#2877）、`ATOM_DP_LOGITS_PEER_EXCHANGE`（#2882）、`AITER_MOE_LOC`（ROCm/aiter#7332 + ROCm/ATOM#3144），以及 `AITER_S1F_*`、`AITER_TDM_COMPACT_ROUTE_*`、`ATOM_PREFILL_ADMIT*`、`ATOM_ENGRAM_MLOCK`、`ATOM_HYI_*`、`ATOM_STEP_TIMER`（只在镜像的私有分支里）。c4（TP4）在 main 上能跑。

### server，c4（TP4）

命令取自镜像 `/app/endpoints-recipe-dsv41/serve_tp4.sh`，`TP4_CFG=c8`：

```bash
cd /app/ATOM
export HIP_VISIBLE_DEVICES=0,1,2,3
export ATOM_DSPARK_PROB_DRAFT=1 ATOM_DSPARK_BLOCK_VERIFY=1
export AITER_CUSTOM_AR_MIN_SIZE=131072 AITER_CUSTOM_AR_MAX_SIZE=8388608 ATOM_GC_THRESHOLD=20000,50,50 HSA_XNACK=1
export HSA_ENABLE_SCRATCH_ASYNC_RECLAIM=1
export ATOM_MONO_ENABLE=1
python3 -u -m atom.entrypoints.openai_server --model /models/DeepSeek-V4.1-Flash --trust-remote-code \
  --host 0.0.0.0 --server-port 8000 -tp 4 \
  --kv_cache_dtype bf16 --index-cache-dtype fp8 \
  --max-num-seqs 64 --max-num-batched-tokens 16384 --attn-prefill-chunk-size 16384 --block-size 16 \
  --level 0 --method dspark --num-speculative-tokens 5 \
  --cudagraph-capture-sizes "[1,2,3,4,5,6,7,8,12,16,24,32,40,48,56,64]" \
  --disable-uvicorn-access-log
```

### server，c64（DEP4）

命令取自镜像 `/app/endpoints-recipe-dsv41/serve_dep4.sh`，`MAXSEQ=64`。三个 `AITER_CONFIG_*` 是调参表，只影响性能，指向 aiter 里的 CSV 和镜像 recipe 目录 `$R/configs` 下的 4 个 CSV；换成 aiter main 后某个文件不存在或 shape 重复时，把那一项删掉。2026-11-12 的 aiter main（`65714259c`）上这些文件都在。

```bash
cd /app/ATOM
R=/app/endpoints-recipe-dsv41
export HIP_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=4 ATOM_DISABLE_MMAP=true AITER_LOG_LEVEL=WARNING
export AITER_REUSE_IDENTICAL_COMM_GROUPS=1 HF_HUB_OFFLINE=1 PYTHONFAULTHANDLER=1 HSA_DISABLE_COREDUMP_ON_EXCEPTION=1
export MORI_AMD_SMI_PATH=/opt/rocm/lib/libamd_smi.so.27 MORI_GPU_ARCHS=gfx1250 FLYDSL_GPU_ARCH=gfx1250
export HIP_FORCE_DEV_KERNARG=1 HSA_ENABLE_SCRATCH_ASYNC_RECLAIM=0 HSA_ENABLE_SDMA=1 HSA_USE_SVM=1 HSA_XNACK=1
export ATOM_GC_THRESHOLD=20000,50,50 AITER_CUSTOM_AR_MIN_SIZE=131072 AITER_CUSTOM_AR_MAX_SIZE=8388608
export MORI_SHMEM_HEAP_SIZE=17179869184
export ATOM_DSPARK_PROB_DRAFT=1 ATOM_DSPARK_BLOCK_VERIFY=1
export ATOM_DP_SESSION_AFFINITY=1
export ATOM_ENGRAM_UVA=1 ATOM_ENGRAM_UVA_COPY=0
export ATOM_PREFILL_DECODE_INTERVAL=24
export ATOM_MORI_V2=1 MEGA_DISPATCH_WIRE=fp4
export ATOM_DP_LM_HEAD_MODE=allgather
export ATOM_STEP_TIMER=1 ATOM_STEP_TIMER_EVERY=500
export MAX_JOBS=16
export ATOM_DSV41_ROPE_HEAD_BLOCK=1
export ATOM_DECODER_GRAPH_SIZES=1,2,3,4,8,16
export ATOM_TOKEN_GRAPH_SIZES=256,512,1024,2048,3072,4096,5120,6144
export AITER_CONFIG_GEMM_BF16="/app/aiter/aiter/configs/bf16_tuned_gemm.csv:/app/aiter/aiter/configs/model_configs/dsr1_bf16_tuned_gemm.csv:/app/aiter/aiter/configs/model_configs/dsv3_bf16_tuned_gemm.csv:/app/aiter/aiter/configs/model_configs/dsv41_bf16_tuned_gemm.csv:/app/aiter/aiter/configs/model_configs/dsv4_bf16_tuned_gemm.csv:/app/aiter/aiter/configs/model_configs/flux2_bf16_tuned_gemm.csv:/app/aiter/aiter/configs/model_configs/glm47_bf16_tuned_gemm.csv:/app/aiter/aiter/configs/model_configs/glm53_bf16_tuned_gemm.csv:/app/aiter/aiter/configs/model_configs/glm53_flash_bf16_tuned_gemm.csv:/app/aiter/aiter/configs/model_configs/glm5_bf16_tuned_gemm.csv:/app/aiter/aiter/configs/model_configs/gptoss_bf16_tuned_gemm.csv:/app/aiter/aiter/configs/model_configs/inkling_bf16_tuned_gemm.csv:/app/aiter/aiter/configs/model_configs/kimi_bf16_tuned_gemm.csv:/app/aiter/aiter/configs/model_configs/kimik2_bf16_tuned_gemm.csv:/app/aiter/aiter/configs/model_configs/kimik3_bf16_tuned_gemm.csv:/app/aiter/aiter/configs/model_configs/llama405B_bf16_tuned_gemm.csv:/app/aiter/aiter/configs/model_configs/llama70B_bf16_tuned_gemm.csv:/app/aiter/aiter/configs/model_configs/minimax_m3_eagle_bf16_tuned_gemm.csv:/app/aiter/aiter/configs/model_configs/minimax_m3_mxfp4_bf16_tuned_gemm.csv:/app/aiter/aiter/configs/model_configs/muse_glimmer_bf16_tuned_gemm.csv:/app/aiter/aiter/configs/model_configs/q3vl_bf16_tuned_gemm.csv:/app/aiter/aiter/configs/model_configs/qwen32B_bf16_tuned_gemm.csv:/app/aiter/aiter/configs/model_configs/qwen38_flash_next_bf16_tuned_gemm.csv:/app/aiter/aiter/configs/model_configs/qwen3_5_397b_bf16_tuned_gemm.csv:/app/aiter/aiter/configs/model_configs/qwen3_8_27b_bf16_tuned_gemm.csv:/app/aiter/aiter/configs/model_configs/qwen3_8_2p4t_a95b_bf16_tuned_gemm.csv:/app/aiter/aiter/configs/model_configs/qwen3_8b_bf16_tuned_gemm.csv:/app/aiter/aiter/configs/model_configs/wan22_bf16_tuned_gemm.csv:$R/configs/dsv41_bf16_all_tuned.csv"
export AITER_CONFIG_GEMM_A8W8_BLOCKSCALE_MXSCALE_BPRESHUFFLE="/app/aiter/aiter/configs/a8w8_blockscale_mxscale_bpreshuffle_tuned_gemm.csv:$R/configs/dsv41_a8w8_blockscale_mxscale_bpreshuffle_tuned_gemm.minus_recipe9.csv:/app/aiter/aiter/configs/model_configs/dsv4_a8w8_blockscale_mxscale_bpreshuffle_tuned_gemm.csv:/app/aiter/aiter/configs/model_configs/minimax_m3_a8w8_blockscale_mxscale_bpreshuffle_tuned_gemm.csv:/app/aiter/aiter/configs/model_configs/kimik3_a8w8_blockscale_mxscale_bpreshuffle_tuned_gemm.csv:$R/configs/dsv41_dep4_dense_mx32_tuned.csv"
export AITER_CONFIG_BATCHED_GEMM_A8W8_BLOCKSCALE_MXSCALE_BPRESHUFFLE="/app/aiter/aiter/configs/batched_gemm_a8w8_blockscale_mxscale_bpreshuffle_tuned.csv:/app/aiter/aiter/configs/model_configs/dsv4_batched_gemm_a8w8_blockscale_mxscale_bpreshuffle_tuned.csv:$R/configs/dsv41_dep4_bmm_mx32_tuned.csv"
export AITER_MEGA_S1F=1 AITER_S1F_BLOCK_RUNS=4,2,8,8 AITER_S1F_MAX_RECV=4096 AITER_S1F_ANY_TILE=1
export AITER_TDM_COMPACT_ROUTE_UPTO=1024 AITER_TDM_COMPACT_ROUTE_SPEC=384,16
export ATOM_PREFILL_ADMIT=adaptive ATOM_PREFILL_ADMIT_VERSION=2
export ATOM_ENGRAM_MLOCK=1 ATOM_HYI_TPL_EXECUTOR=1 ATOM_HYI_DROP_FINISHED_SEQS=1 ATOM_HYI_ENC_MEMO=1
export ATOM_DSPARK_PROB_FUSED_CHAIN=1
export ATOM_DP_LOGITS_PEER_EXCHANGE=1
export AITER_MOE_LOC=1
python3 -u -m atom.entrypoints.openai_server --model /models/DeepSeek-V4.1-Flash --trust-remote-code \
  --host 0.0.0.0 --server-port 8000 -tp 4 \
  --kv_cache_dtype fp4 --index-cache-dtype fp4 --gpu-memory-utilization 0.9 \
  --max-num-batched-tokens 16384 --attn-prefill-chunk-size 16384 --block-size 16 \
  --level 0 --max-num-seqs 64 --cudagraph-capture-sizes "[1,2,3,4,5,6,7,8,12,16,24,32,40,48,56,64]" \
  --enable-expert-parallel --enable-dp-attention --all2all-backend high-throughput \
  --method dspark --num-speculative-tokens 5 \
  --disable-uvicorn-access-log --api-server-count 4
```

### gsm8k

按指南 §3.3，`MODEL_PATH=/models/DeepSeek-V4.1-Flash`，`PORT=8000`；前面加 `HF_HUB_OFFLINE=0`（镜像 ENV 里是 1）。

### bench

client 用镜像里的 `/app/mlcommons-endpoints`（`mlcommons/endpoints` @ `d3c57f2`，自带 venv），agentic 数据集和 Kimi-K3 一节相同（sha256 `1beb24c8…`）。client 配置照镜像 `run_ep.py` / `mkcfg.py` 生成：以 `examples/10_Agentic_Inference/dsv4_agentic_benchmark.yaml` 为底，只保留 performance 数据集，613 条轨迹，第一个 user 跑完就停，发 10 分钟请求。

```bash
EP=/app/mlcommons-endpoints
DATASET=<数据目录>/mlperf_agentic_inference_dataset.jsonl
$EP/venv/bin/python - $EP/examples/10_Agentic_Inference/dsv4_agentic_benchmark.yaml $DATASET /models/DeepSeek-V4.1-Flash $CONC $RESULT_DIR/report > $RESULT_DIR/client.yaml << 'PY'
import sys, yaml
base, dataset, model, c, report = sys.argv[1:]
cfg = yaml.safe_load(open(base))
cfg["model_params"].update(name=model, tokenizer_name=model)
perf = next(d for d in cfg["datasets"] if d.get("type") == "performance")
perf["path"] = dataset
perf["accuracy_config"] = {"eval_method": "agentic_inference_inline", "num_repeats": 1}
perf["agentic_inference"].update(num_trajectories_to_issue=613, enable_salt=True, inject_tool_delay=True,
                                 stop_issuing_on_first_user_complete=True)
cfg["datasets"] = [perf]
cfg["settings"]["load_pattern"]["target_concurrency"] = int(c)
cfg["settings"].setdefault("runtime", {})["max_issue_duration_ms"] = 600000
cfg["settings"].setdefault("timeouts", {})["run_timeout_s"] = 14400
cfg["endpoint_config"]["endpoints"] = ["http://127.0.0.1:8000"]
cfg["report_dir"] = report
print(yaml.safe_dump(cfg, sort_keys=False))
PY
# warmup（run_ep.py 的做法）：发一条带工具调用的 chat 请求，再清 prefix cache
curl -sf -m 900 http://127.0.0.1:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{"model": "/models/DeepSeek-V4.1-Flash", "max_tokens": 2048, "temperature": 1.0, "top_p": 0.95, "top_k": 0, "chat_template_kwargs": {"thinking": true, "reasoning_effort": "max"}, "tools": [{"type": "function", "function": {"name": "get_order", "description": "Look up an order", "parameters": {"type": "object", "properties": {"order_id": {"type": "string"}}, "required": ["order_id"]}}}], "messages": [{"role": "user", "content": "Where is my order 43E33A3CAA? Use the tool."}]}' > $RESULT_DIR/smoke.json
for i in $(seq 12); do      # 刚结束的 smoke 请求还在清理时返回 409，和 run_ep.py 一样每 5 秒重试
  code=$(curl -s -o $RESULT_DIR/prefix_cache_reset.json -w '%{http_code}' -X POST http://127.0.0.1:8000/reset_prefix_cache -H 'Content-Type: application/json' -d '{}')
  [ "$code" = 409 ] || break; sleep 5
done
cd $EP && HF_HUB_OFFLINE=1 $EP/venv/bin/inference-endpoint benchmark from-config --config $RESULT_DIR/client.yaml --mode perf > $RESULT_DIR/bench.log 2>&1
```

结果在 `$RESULT_DIR/report/performance/result_summary.json`，inline 精度在 `$RESULT_DIR/report/scores.json`。

- main 上没有 `/reset_prefix_cache`（返回 404，在 [ROCm/ATOM#2763](https://github.com/ROCm/ATOM/pull/2763)），这一步等于跳过。

### 参考值

2026-11-12，同一台机器：

| | GSM8K flexible / strict | tok/s/GPU | E2E avg interactivity | inline 精度 |
|---|---|---|---|---|
| c4 TP4，main（ATOM `f7b8962fd`、aiter `65714259c`） | 0.9196 / 0.9196 | 274.51（1098.06 ÷ 4） | 502.76 | 0.5061 |
| c4 TP4，`gfx1250-atom-20261111` 原样 | 0.9219 / 0.9212 | 283.92（1135.67 ÷ 4） | 531.07 | 0.5106 |
| c64 DEP4，`gfx1250-atom-20261111` 原样 | 0.9196 / 0.9196 | 3,098.08（12,392.34 ÷ 4） | 317.68 | 0.5477 |

每个点只跑了一次。inline 精度是 client 自带的 agentic 打分，`run_ep.py` 的门槛是 0.5236；c4 十分钟只评到约 700 个 turn。

---

## DeepSeek-V4-Pro

- 场景：8K/1K；GPU：4（1 tray x 1P4G）
- 并行：注意力 DP4（DP attention），MoE EP4（MegaMoE）；FP8 KV cache，FP4 index cache；不开 prefix cache
- MODEL_PATH：`/models/DeepSeek-V4-Pro`；PORT：`8000`。2026-11-12 测的是 `DeepSeek-V4-Pro-0813` 这份 checkpoint（config 比另一份 `DeepSeek-V4-Pro` 多 `dspark_*` 字段，attention 权重的类型和 shape 相同）

**当前 main 上跑不起来，预期记 N/A；`gfx1250-atom-20261111` 原样镜像同样跑不起来**：加载权重后录 CUDA graph 时，4 个 rank 都在 `layers.1.attn` 的 `wqkv_a` 上报 `IndexError: tuple index out of range`，位置是 aiter 的 FlyDSL gfx1250 mxfp8 GEMM（`aiter/ops/flydsl/mxfp8_bpreshuffle_gemm_gfx1250.py:411` → `kernels/tensor_shim.py:336`）。[ROCm/aiter#7178](https://github.com/ROCm/aiter/pull/7178)（已合入 main）给 `launch_gemm_a8w8` 加了带默认值的参数 `arg_scale_out`，dense mxfp8 的调用方没跟着改，少传参数，同一个 kernel 第二次走缓存时越界。镜像里带的是这个 PR 的预合入版，所以也崩。脚本默认的 `FAKE_EPLB=1` 和 `FAKE_EPLB=0` 结果一样。

命令取自镜像 `/app/scripts/dsv4/serve_dsv4_ep4.sh`（默认 `FUSED=1 STAGE1=1 TRITON_MOE=0`），去掉 `FAKE_EPLB=1` 加的 `--fake-eplb --hf-overrides '{"num_hash_layers": 0}'`：假路由输出乱码，GSM8K 没法测。c4 和 c64 用同一个 server。

### server

```bash
env ATOM_MEGA_COMBINE_WIRE=fp4 PYTHONPATH=/app/aiter HSA_ENABLE_SDMA=1 HSA_USE_SVM=1 HSA_XNACK=1 \
  AITER_DISABLE_KERNARG_PRELOAD=0 ATOM_USE_TRITON_MOE=0 ATOM_USE_TRITON_MOE_DECODE=0 ATOM_USE_TRITON_MOE_A4W4=0 \
  ATOM_USE_UNIFIED_ATTN=1 ATOM_FORCE_ATTN_TRITON=0 ATOM_USE_V4_PREFILL_ASM_FOR_DECODE=1 ATOM_LOADER_USE_THREADPOOL=1 \
  AITER_ROPE_TRITON_BACKEND=1 ATOM_ENABLE_DS_QKNORM_FUSION=0 ATOM_ENABLE_DS_INPUT_RMSNORM_QUANT_FUSION=0 \
  ATOM_ENABLE_DS_QKNORM_QUANT_FUSION=0 ATOM_USE_TRITON_GEMM=1 ATOM_LOADER_NUM_THREADS=4 ENABLE_CK=0 \
  AITER_USE_OPUS_MOE_SORTING=1 AITER_FORCE_A8W4=0 AITER_USE_GROUPED_GEMM=1 ATOM_USE_AITER_TRITON_ATTN=1 \
  AITER_TRITON_USE_HERD=1 ATOM_FP8_BLOCKSCALE_USE_E8M0_SCALE=1 ATOM_DSV4_USE_GFX1250_MXFP8_ASM_GEMM=1 \
  AITER_CUSTOM_AR_MIN_SIZE=131072 ATOM_MOE_GU_ITLV=1 ATOM_MORI_V2=1 MEGA_DISPATCH=mori ATOM_WO_A_USE_FLYDSL=1 \
  MORI_GPU_ARCHS=gfx1250 FLYDSL_GPU_ARCH=gfx1250 ATOM_MEGA_STAGE2_FUSED=1 ATOM_MEGA_STAGE1_FUSED=1 \
  AITER_FLYDSL_GATHER_REDUCE_WAVE_MIN_TOKENS=-1 ATOM_GC_THRESHOLD=20000,50,50 AITER_CUSTOM_AR_MAX_SIZE=8388608 \
  ATOM_DP_LM_HEAD_MODE=allgather MEGA_DISPATCH_WIRE=fp4 \
python3 -m atom.entrypoints.openai_server -tp 4 --model /models/DeepSeek-V4-Pro \
  --kv_cache_dtype fp8 --index-cache-dtype fp4 --no-enable_prefix_caching \
  --enable-expert-parallel --enable-dp-attention --disable_uvicorn_access_log
```

### gsm8k

和指南 §3.3 不同的地方：加 `max_gen_toks=2048`（镜像里的 `/app/scripts/dsv4/eval_dsv4.sh`），前面加 `HF_HUB_OFFLINE=0`。

```bash
HF_HUB_OFFLINE=0 lm_eval --model local-completions \
  --model_args "model=/models/DeepSeek-V4-Pro,base_url=http://127.0.0.1:8000/v1/completions,num_concurrent=65,max_retries=3,max_gen_toks=2048,tokenized_requests=False,trust_remote_code=True" \
  --tasks gsm8k --num_fewshot 3 --output_path "$ACC_DIR" > "$ACC_DIR/lm_eval.log" 2>&1
```

### bench

```bash
python3 -m atom.benchmarks.benchmark_serving \
  --model=/models/DeepSeek-V4-Pro --backend=vllm --base-url=http://127.0.0.1:8000 \
  --dataset-name=random --random-input-len=8192 --random-output-len=1024 --random-range-ratio=0.8 \
  --max-concurrency=$CONC --num-prompts=$((CONC * 10)) --num-warmups=$((CONC * 2)) \
  --request-rate=inf --ignore-eos --trust-remote-code \
  --save-result --result-dir=$RESULT_DIR --result-filename=raw.json \
  --percentile-metrics="ttft,tpot,itl,e2el"
```

### 参考值

还没有：2026-11-12 在 main 和 `gfx1250-atom-20261111` 上都没起来（见上）。
