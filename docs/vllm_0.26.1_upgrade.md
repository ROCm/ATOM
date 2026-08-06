# vLLM 0.26.1 upgrade

本文记录 ATOM OOT plugin 从 vLLM `568afb3a13806beb53bb2e6bd518269357b237c0`
升级到指定提交 `adbf08d977fb3fa26c4f19826745a02abd6dd7ca` 的镜像构建、兼容修改和
PR CI 模型验证结果。

## 源码与镜像

- 上一版 vLLM 提交：`568afb3a13806beb53bb2e6bd518269357b237c0`
  - 实际 wheel：`0.26.1.dev0+g568afb3a1.d20260803.rocm724`
- 目标 vLLM 提交：`adbf08d977fb3fa26c4f19826745a02abd6dd7ca`
  - `git describe`：`v0.26.1rc0-305-gadbf08d97`
  - 实际 wheel：`0.26.1rc1.dev305+gadbf08d97.d20260804.rocm724`
- 版本入口：`atom/plugin/vllm/vllm-version.env`
- 最终本地镜像：`localhost/atom-vllm:v0.26.1`
- 镜像 ID：`8c164cd3024252b16e58768bdf97054764bb4113f1293f75842cfb918dc34e41`
- vLLM commit label：
  `com.rocm.atom.vllm_commit=adbf08d977fb3fa26c4f19826745a02abd6dd7ca`

## 构建与依赖处理

目标提交的上游构建依赖将 PyTorch 从 `2.11.0` 提升到 `2.13.0`。
按照升级要求，ATOM 不安装该 PyTorch 依赖：

- 继续使用 `setup.py bdist_wheel` 直接构建，避免 pyproject build
  isolation 拉取 `torch==2.13.0`。
- Dockerfile 在构建前显式删除 pyproject 中的 `torch == 2.13.0` build
  requirement。
- vLLM wheel 继续使用 `pip install --no-deps` 安装。
- ROCm runtime requirements 中没有新增 torch、Triton 或 transformers
  强制替换项。
- 上游 `requirements/common.txt` 的 `mistral_common[image]` 最低版本从
  `1.11.5` 提升到 `1.11.6`；镜像中已有 `1.11.7`，无需变更。
- release Dockerfile 不安装或升级 AITER/FlyDSL；二者继续由
  `OOT_BASE_IMAGE` 或 CI overlay 管理。

升级后保留的核心运行栈：

- torch：`2.10.0+rocm7.2.4.lw.git3d3aa833`
- torchvision：`0.25.0+rocm7.2.4.git82df5f59`
- torchaudio：`2.10.0+rocm7.2.4.git5047768f`
- Triton：`3.7.0+amd.rocm7.2.0.git89002410`
- transformers：`5.12.1`
- mistral-common：`1.11.7`
- amd-quark：`0.12.post1`
- xgrammar：`0.2.3`
- compressed-tensors：`0.17.0`
- amd-aiter：`0.0.0`（继承自 `rocm/atom-dev:latest`）
- FlyDSL：`0.2.4`（继承自 `rocm/atom-dev:latest`）

镜像对比中实际发生的包版本变化：

- vLLM：
  `0.26.1.dev0+g568afb3a1.d20260803.rocm724` ->
  `0.26.1rc1.dev305+gadbf08d97.d20260804.rocm724`
- openai：`2.52.0` -> `2.53.0`（由评测/runtime 依赖解析更新）

## Plugin 兼容修改

### CacheConfig 字段删除

目标 vLLM 删除了 `CacheConfig.calculate_kv_scales`。MHA、MLA 和
MiniMax-M3 attention 初始化改为 capability-safe 的 `getattr(..., False)`；
新版本保持关闭已删除的 runtime KV scale 计算，旧版本仍可读取该字段。

### KV cache 绑定协议

vLLM 现在统一调用 attention layer 的 `bind_kv_cache()`，不再直接赋值
`layer.kv_cache`。为 ATOM 的 `DeepseekV32IndexerCache` decorator 补充
`bind_kv_cache()`，并保留 ATOM 所需的单元素 list 包装。

### SchedulerConfig 字段删除

目标 vLLM 删除了 `max_num_partial_prefills`。ATOM cudagraph capture metadata
在字段不存在时使用等价的默认值 `1`。

### DeepSeek-V4 block 复用精度回退

完整 DeepSeek-V4-Flash 初跑从上一版 `0.839272` 降到 `0.620925`。
受控对比排除了 AITER 和 cudagraph：

- 目标 vLLM + 旧 AITER，200 题：`0.635`
- 目标 vLLM + `--enforce-eager`，200 题：`0.655`
- 上一版 vLLM，同一当前 ATOM 代码，20 题：`0.85`
- 目标 vLLM，20 题：`0.50`

对 `568afb3a1..adbf08d97` 的 Python 源码做 10 轮二分后，首个回退提交是
`a82f1b388fe625038502eaa593690ed055fc4dd1`：
`Skip LRU hash-split in free_blocks when prefix caching is off`。该提交在关闭
prefix cache 时立即复用刚释放的 block。

ATOM V4 proxy 把固定 SWA 区和 block-indexed CSA/HCA 区组织在全局 arena
中；ATOM GDN 也保持由 Mamba block-table slot 索引的递归状态。为这两类
stateful cache pool 保留 vLLM `a82f` 之前的 free-queue 顺序，普通 MHA/MLA
模型仍使用上游的新 locality 优化。补丁在 general-plugin hook 中安装，
确保拥有 Scheduler/KVCacheManager 的 EngineCore 进程生效。

修复后：

- 20 题确定性样本：`0.90`
- 完整 1,319 题：`0.838514`

### GDN FULL replay

目标 vLLM 已在 FULL replay 时用清空后的 dummy block-table 重新构造
Mamba/GDN metadata。移除 ATOM 针对旧版 vLLM 的二次 compaction，避免把
上游生成的 PAD slot 覆盖为 live state index。

### AITER 基线

`rocm/atom-dev:latest` 中原有 AITER package metadata 为 `0.0.0`，且缺少
ATOM 当前代码需要的 `ActivationType.Situv2`。该问题不在 release Dockerfile
中修复：本地 PR CI 验证按照 `.github/workflows/atom-vllm-test.yaml` 的
overlay 流程安装 AITER `0.1.1.dev1+gaf02117fb` 与 FlyDSL `0.3.0`；最终
release 镜像则原样继承基础镜像中的 AITER/FlyDSL。

## 构建与基础验证

- `vllm-torch210-compat.patch` 对目标提交可干净应用。
- release Dockerfile 构建成功。
- 镜像 commit label 与 `/app/vllm` HEAD 均为目标提交。
- GPU torch 运算和 `vllm._custom_ops` 加载通过。
- plugin 兼容回归：`7 passed`。
- `pip check` 仅保留基础镜像中已知的 LMCache 可选依赖缺失及
  torch/Triton 定制组合告警。

## PR CI 精度结果

来源：`.github/workflows/atom-vllm-test.yaml`。

- DeepSeek-V4-Flash TP4
  - flexible-extract：`0.838514`
  - strict-match：`0.751327`
  - 配置阈值：`0.93`
  - 状态：完整运行成功；未达到 catalog 阈值，但与上一版完整结果
    `0.839272` 一致，升级回退已修复。该 checkpoint/catalog 阈值不匹配是
    已知基线问题。
- GLM-5.2-MXFP4-MTP TP4
  - flexible-extract：`0.915845`
  - strict-match：`0.916603`
  - 阈值：`0.91`
  - MTP overall acceptance：`0.752255`
  - MTP per-position：`0.911720`, `0.762246`, `0.582799`
  - 状态：通过。
- MiniMax-M3-MXFP4 TP4
  - flexible-extract：`0.944655`
  - strict-match：`0.945413`
  - 阈值：`0.93`
  - 状态：通过。
- Kimi-K2.7-MXFP4 TP4
  - flexible-extract：`0.953753`
  - strict-match：`0.952995`
  - 阈值：`0.92`
  - 状态：通过。
  - 本机 checkpoint 位于
    `/data/amd_int/models/Kimi-K2.7-Code-MXFP4`，与 workflow 中的
    `amd/` catalog 前缀不同。
- Qwen3.5-397B-A17B-MXFP4 TP4
  - 默认 65 并发 flexible-extract：`0.834723`
  - 默认 65 并发 strict-match：`0.818802`
  - 阈值：`0.83`
  - 状态：通过。
  - 修复前 65/8 并发均在 GDN state churn 时触发 HIP illegal-address；
    stateful Mamba block 复用修复后，8 并发先得到 `0.830933`，随后原始
    65 并发完整复验通过，因此无需降低 CI 并发。CI overlay 使用
    FlyDSL 0.3；清空显存、独占 GPU0-3 后复验得到上述结果。
  - 本机 checkpoint 位于
    `/data/amd_int/models/Qwen/Qwen3.5-397B-A17B-MXFP4`。

统计：

- CI catalog case：5
- 完整运行成功：5
- 达到配置阈值：4
- 未达到阈值但与升级前基线一致：1（DeepSeek-V4-Flash）
- 运行失败：0

## Kimi-K3 补充验证

使用 `/data/amd_int/models/Kimi-K3`，按照
`recipes/atom_vllm/Kimi-K3.md` 运行 TP8、FP8 KV cache、
`FULL_AND_PIECEWISE` CUDA Graph、GSM8K 5-shot：

- vLLM 0.26.1 首次启动暴露 `process_weights_after_loading(act_dtype)` 新
  调用协议；plugin KDA layer 吸收该参数后继续执行原有 post-load folding。
- Kimi-K3 plugin 单测：`6 passed`。
- 固定 ATOM `a82c25760`、AITER `5c9f6431ec` 的升级前完整基线：
  - flexible-extract：`0.950720`
  - strict-match：`0.949204`
- 升级后 64 并发完整 1,319 题：
  - flexible-extract：`0.366945`
  - strict-match：`0.366945`
- 并发探测：
  - 8 并发前 100 题：`0.96`
  - 32 并发前 100 题：`0.95`
- 保持客户端 64 outstanding、将服务端 `max-num-seqs` 限制为 32 后，
  完整 1,319 题：
  - flexible-extract：`0.952995`
  - strict-match：`0.952237`
  - 总评测时间：`416.8s`（客户端 32 的安全对照为 `411.3s`）

使用 ATOM `guanbao/vllm_0.26_vision`
（`3aed484e47257d2f78166640a416c1a01915d078`）在同一升级后 vLLM、
AITER `5c9f6431ec` 环境中重测真正的 client64/server-active64：

- flexible-extract：`0.952995`
- strict-match：`0.950720`
- 总评测时间：`327.5s`

逐提交和最小化实验确认：

1. `b502148f` 将 language model 的 instance-level KDA packed/weight
   mapping 暴露到 conditional-generation 顶层，避免 253 个 KDA 参数未加载；
   其父提交只补 post-load 签名时前 100 题为 `0.0`，加入该提交后为 `1.0`。
   这是 vision conditional-generation wrapper 必需的加载修复，但主线
   text-only wrapper 已直接暴露 instance mapping，不需要移植该功能栈。
2. vLLM 0.26.1 已提供 `KimiK3KDAMetadata` 和专用 builder；它虽然继承
   `GDNAttentionMetadata`，但具有 KDA 的 state-slot 对齐、graph padding 和
   speculative-decode 语义。ATOM 原先仍注册通用 GDN backend，契约不完整。
3. 直接切换上游 KDA builder 时 active64 前 100 题为 `0.71/0.70`，因为上游
   packed-decode kernel 不消费 compacted `query_start_loc`，而 ATOM KDA
   kernel 仍需要该请求索引。增加只属于 KDA backend 的 FULL-graph adapter
   后恢复至 `0.99/0.99`；通用 GDN backend 不再包含 Kimi 特判。

最终主线返回 `KimiK3KDAMetadata`，使用 KDA state shape/dtype/copy contract，
并在 KDA builder 后补充 ATOM decode 所需的 request-index adapter。无需
`max-num-seqs 32` workaround，client64/server-active64 全量结果：

- flexible-extract：`0.949204`
- strict-match：`0.947688`
- 总评测时间：`329.1s`

该结果与升级前基线均仅相差 2 题，低于单次评测标准误差，并比
server-active32 workaround（`416.8s`）快约 21%。通用 GDN builder 保持
vLLM 0.26.1 原生行为，因此不重新引入 Qwen3.5 的旧 metadata 问题。

## 结果文件

所有日志、二分记录和结果 JSON 位于：
`/shared/amdgpu/home/perzhang_qle/vllm_0261_upgrade_results`。

最终结果：

- DeepSeek-V4-Flash：
  `ci/deepseek-v4-flash/final-fixed/results/20260804065730_DeepSeek-V4-Flash.json`
- GLM-5.2-MXFP4-MTP：
  `ci/glm-5.2-mxfp4-mtp/final-flydsl03/results/20260804085507_GLM-5.2-MXFP4-MTP.json`
- MiniMax-M3-MXFP4：
  `ci/minimax-m3-mxfp4/results/20260804072612_MiniMax-M3-MXFP4.json`
- Kimi-K2.7-MXFP4：
  `ci/kimi-k2.7-mxfp4/results/20260804073620_Kimi-K2.7-Code-MXFP4.json`
- Qwen3.5-397B-A17B-MXFP4：
  `ci/qwen3.5-mxfp4/final-flydsl03-solo/results/20260804090417_Qwen3.5-397B-A17B-MXFP4.json`
- Kimi-K3（最终主线，active64）：
  `ci/kimi-k3-gsm8k/kda-metadata-adapter-active64-full/results/20260805070651_Kimi-K3-kda-adapter-full.json`
- Kimi-K3 (`guanbao/vllm_0.26_vision`, active64)：
  `ci/kimi-k3-gsm8k/vision-branch-full/results/20260805040551_Kimi-K3-vision-branch.json`
