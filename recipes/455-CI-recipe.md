# 455-CI recipe

各模型在 455-CI 镜像（`rocm/fw-bringup-mlperf:gfx1250-atom-ci455-<YYYYMMDD>`，ATOM main + aiter main）上的 server、GSM8K 和 bench 命令。流程、指标口径和记录方式见 [`455-CI-guide.md`](455-CI-guide.md)；这里只写命令。

目前收录 Kimi-K3 (2.8T)、GLM-5.2、MiniMax-M3 (428B) 三个模型，其余 5 个见 [`455-CI-guide.md`](455-CI-guide.md) §1。

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
