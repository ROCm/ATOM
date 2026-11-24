> [Examples README](../README_cn.md) > 排障

# 6. 排障

**FP8 训练发散**（entropy ~0.04 / `grad_norm` 1e4+ / `rollout_corr/kl` 1e4+）：
基本只有两个原因——`FP8_PARAM_MANAGER` 没设成 0（它与 native FSDP2 的 fp32-master 冲突），
或 [vLLM RMSNorm patch](02-dependencies_cn.md#27-vllm-aiter-rmsnorm-patch例子-2-3-4-必需) 没打（新容器要重打）。

**显存回退（OOM）**：
- 降 `policy.max_response_length=8192` + `max_total_sequence_length=9216` + `max_token_len_per_gpu=9216`；
- 或降 `train_global_batch_size` / `gen_batch_size`；
- Ray 路径 **不要**开 `fsdp_cfg.param_offload` / `optimizer_offload`，会报
  `parameters should be materialized on CPU`。

**例子 7 的 OOM 有个反直觉的点**：`max_tokens_per_gpu: 22528` 时会在 step 14 死在 actor
backward，但**崩溃不是 allocated 峰值的问题**（改前后都是约 130 GB），而是**碎片**——
ROCm 没有 `expandable_segments`，每步约 7 个打满 22.5k 的 bin 反复申请释放巨块，reserved
比 allocated 多出 42 GB。压到 8192 之后碎片间隙塌到 4-11 GB，峰值 reserved 从 177 GB 降到
134 GB。**所以那个 8192 不能随手调回去。**

**`weight sync (colocate-ipc) left N/M rollout parameters untouched`**：同步漏了参数。
异常会列出前 8 个名字；若是 `...experts.w13_weight` / `w2_weight`，说明融合 MoE 路由没生效，
要么代码没拉到最新，要么 vLLM/transformers 升级后布局假设失效了。

**例子 7 在 `init_model` 死于 `TypeError: 'NoneType' object is not callable`**
（上一帧是 `get_gpt_layer_with_transformer_engine_spec`）：TransformerEngine 没装。
没有它 `megatron.core` 也能正常 import，所以 [依赖](02-dependencies_cn.md) §2.5 的导入
检查会通过，只有 layer spec 那条检查才能抓到。按 §2.3 编译 TE；只 `pip install
megatron-core` 是不够的。

**例子 5（`MODE=atombf16`）可能死在 ATOM 引擎初始化**，driver 侧报
`RuntimeError: Engine Core Mgr: Received unexpected SHUTDOWN signal from DP rank 0
during initialization`。这条只是症状，**一定要往上读 replica 侧的真实异常**，因为有两个
不同的原因共用这个症状：

- `rope_cache` / `fused_qk_rope_reshape_and_cache` 下的
  `ValueError: too many values to unpack (expected 4)`：这是 `ATOM_FORCE_ATTN_TRITON=1`。
  ATOM 分配的是 5 维 SHUFFLE V-cache，而 aiter 的 triton kernel 在非 flash 布局下要 4 维，
  两者不能靠 `view()` 互转。把这个变量取消掉；AtomRL 侧绕不过去。
- `torch/_functorch/_aot_autograd/runtime_wrappers.py` 里的
  `IndexError: list index out of range`，由 ATOM 的 `warmup_model` 触发：**是例子 4
  留下的编译缓存。** 两个例子都让 ATOM 跑在 `enforce_eager=false` +
  `compilation_config.level=3` 下，但例子 4 编译的是 FP8 模型、例子 5 编译的是 BF16 模型，
  而 torch 的 inductor 缓存（`/tmp/torchinductor_root`）**不像 `ATOM_ISOLATE_TORCH_COMPILE_CACHE`
  那样按运行隔离**。于是例子 5 载入了一份输入签名不匹配的图，死在 AOTAutograd。

  实测，同一容器同一套栈：

  | 顺序 | 例子 5 |
  |---|---|
  | 清缓存后单独跑例子 5 | 通过（冷、热各一次） |
  | 清缓存后先跑例子 4、再跑例子 5 | 71 秒失败 |

  所以两次精度不同的 ATOM 运行之间要清缓存：

  ```bash
  sudo docker exec "$CONTAINER" bash -lc \
    'rm -rf /tmp/aiter_configs /tmp/atom_torch_compile_cache /tmp/torchinductor_root'
  ```

  `/tmp/aiter_configs` 本身也要清：不清的话 aiter 会静默沿用上一次的合并调优配置。

**例子 5 能跑起来但 `rollout_corr/kl` 明显偏高。** 这与上面的崩溃是两件事，而且不触发
失败判据——exit 0，生成侧健康（`ppo_kl` ≈ 0、零次 `kept 0/`、零次
`finished with reason max`），只有训练与 rollout 的 log-prob 差距不对。

例子 5 存在的意义正是回答「差距来自 FP8 还是 ATOM 对齐」：它**高于**例子 4 就说明答案是
对齐而不是量化。这时不必再去核 `use_model_sensitive_rmsnorm`——本发布钉的 ATOM 在
`atom/model_ops/layernorm.py` 的两个调用点都已经传了它。问题在别处，值得向上游反馈。

**ATOM rollout 退化**（`MODE=atomfp8` / `atombf16` 时 `filter_groups: kept 0/96` +
`Rollout reward: accuracy=0.0000` + 日志大量 `finished with reason max`、无 `eos`）：
rollout 生成崩坏。优先检查 ATOM `atom/model_ops/layernorm.py` 的 plain RMSNorm 有没有传
`use_model_sensitive_rmsnorm=1`；未对齐会先表现为 `rollout_corr/kl` 偏大（~0.007 而非 ~0.004）。
**用例子 5 定位更快**：它把量化关掉了，kl 若仍偏大，问题一定在 ATOM 与训练侧的对齐上，
而不在 FP8。

**`TORCHDYNAMO_DISABLE` 不要手工设。** 脚本全局保持 `=1`（训练 actor 关 dynamo）；
例子 4、5 的 no-eager level=3 rollout 所需的 dynamo 由 `ATOMReplicaManager` 在创建 ATOM Ray
actor 时通过 `runtime_env` 注入 `TORCHDYNAMO_DISABLE=0`，**只作用于 rollout 进程**。
顶层 `export TORCHDYNAMO_DISABLE=0` 会让训练 actor 一并继承，对训练侧纯属副作用。

**例子 4、5 的另外两个前提**（`MODE=atomfp8` / `atombf16` 都会自动设，手工覆盖时别丢）：
`ATOM_ISOLATE_TORCH_COMPILE_CACHE=1`（否则 8 个单卡 replica 并发写同一个 torch compile
cache，触发 Inductor `write_atomic -> rename` 的 `FileNotFoundError`）、
`enable_sleep_mode=true` 且 `sleep_level=2`（rollout 后释放 KV cache / weights / CUDA graph，
否则训练 backward 容易 `HSA_STATUS_ERROR_OUT_OF_RESOURCES`）。

**跑完/中断后显存不释放**（`rocm-smi` 每卡仍 ~90 GB，但 `ps` 里已无 trainer）：
`run_dapo.sh` 只在**启动前**清理进程，收尾不清，所以 ATOM EngineCore 的 `spawn_main`
子进程（及其 inductor compile worker）会变成孤儿继续占显存。手动清理时注意用
`spawn[_]main` 这类写法，否则 `pkill -f` 会匹配到自己的命令行而自杀：

Ray 的 actor worker 也会以同样方式变孤儿，只杀 launcher 的话它们会继续抱着模型。
把这些一起覆盖掉：

```bash
sudo docker exec "$CONTAINER" bash -lc '
  ray stop --force >/dev/null 2>&1
  for p in "[a]tomrl.trainer.main" "[r]ay::AtomActorWorker" "[r]ay::VLLMRayServer" \
           "[r]ay::ATOMRayServer" "[V]LLMRayServer" "[E]ngineCore" "[r]aylet" \
           "compile_[w]orker" "spawn[_]main" "resource[_]tracker"; do
    pkill -9 -f "$p" || true
  done
  sleep 10; rocm-smi --showmeminfo vram | grep -i used | head -3'
```

> ⚠️ **只按进程名匹配，绝不要按"占了很多显存"来杀。** 这些节点经常是共享的，按显存
> 大小做 `pkill` 会把别人的作业一起带走。如果上面这些模式跑完显存仍被占着，那剩下的
> 就不是你的——动手之前先确认归属：
>
> ```bash
> # 只读：卡上到底是什么，属于哪个容器
> rocm-smi --showpids
> docker ps --format '{{.Names}}  {{.Status}}'
> docker top <container>            # 按容器归属某个 PID
> ```
>
> 命令行与 AtomRL 无关的进程属于同租户，别碰它，
> 见下一条。

**同租户占了卡，而 OOM 报错会掩盖这一点。** 调度器可能把 8 张卡分给你，但节点上已经有
别人在算。特征是 OOM 里两个数字对不上：

```text
torch.OutOfMemoryError: HIP out of memory. Tried to allocate 1.50 GiB.
GPU 0 has a total capacity of 287.98 GiB of which 0 bytes is free.
Of the allocated memory 47.66 GiB is allocated by PyTorch, ...
```

PyTorch 只占了 47 GiB 却已经 `0 bytes free`，说明另外约 240 GiB 不是我们的。开跑**前**
就查基线，别等失败了再回头诊断：

```bash
sudo docker exec "$CONTAINER" bash -lc \
  'rocm-smi --showmeminfo vram | grep -i used'
```

每张卡应该读到约 298 MB。预算要按**剩余量**算而不是按整卡容量算：rollout 引擎预留的是
`gpu_memory_utilization x 288 GiB`——示例配置用 0.30，即 **86 GiB**——这是整卡的比例而不是
空闲部分的比例，训练侧还要再占 44 GiB（8B）到 115 GiB（MoE）。同租户占了 230 GiB 就只剩
58 GiB，什么都放不下。这种情况请等独占节点，不要靠调低 `gpu_memory_utilization` 硬塞：
那样指标就不能和 [启动](04-launching_cn.md) §4.7 的基线对比了。

**换 `DATA_ROOT` 要无条件覆盖。** 起容器时的 `docker run -e DATA_ROOT=...` 把值烤进了容器环境，
自建 wrapper 里写 `export DATA_ROOT="${DATA_ROOT:-/new/disk}"` **不会生效**（变量已存在），
ckpt 会静默写回旧盘。直接 `export DATA_ROOT=/new/disk`。

**vLLM worker 里的 `logger.info` 不进 driver 日志。** 所以例子 7 看不到那行
`weight sync coverage`。**不能**据此认为断言没跑——判断断言是否触发，看它有没有抛异常。

多节点 RDMA 排障见 [多节点 RDMA](05-multinode-rdma_cn.md)。
