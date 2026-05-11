# MTP 入口投影 BF16/量化失配 bugfix 说明

本文记录 `fix/mtp-entry-proj-quant-bf16-mismatch` 分支修复的一组
MTP accept rate 约等于 0% 的静默 corruption 问题。核心问题有两个：

1. MTP 入口投影的 checkpoint 权重是 BF16，但模型构造阶段错误地按 FP8
   量化层创建了参数。
2. 端到端验证 Qwen3-Next-FP8 时，又发现两个与入口投影无关、但同样会把
   MTP accept rate 打到 0% 的加载路径 bug。

这些问题的共同危险点是：server 可以正常启动，单 prompt 输出也可能看起来正常，
但 speculative decoding 完全失效，只有 MTP accept rate 或加载日志才能暴露。

## 1. TL;DR

| 项 | 结论 |
| --- | --- |
| 主症状 | `openai_server` 启动、推理都正常，但 MTP accept rate 约为 0%，`Average toks/fwd` 约为 1.00 |
| 主根因 | BF16 entry projection 被错误构造成 FP8 block-scale 量化层，checkpoint 又没有对应 `weight_scale` |
| 直接后果 | BF16 权重被静默 cast 到 FP8；`weight_scale` 保持 `torch.empty` 脏值；AITER GEMM 用错误 scale 计算 |
| 影响模型 | DeepSeek R1/R1-0528/V3/V3.2 FP8；Qwen3-Next FP8；潜在影响 MiMo-V2-Flash MTP |
| 已验证修复 | Qwen3-Next-FP8 TP4 e2e smoke：accept rate 从 0% 恢复到 60%+，吞吐从约 172 tok/s 提升到约 604 tok/s |
| 修复 commits | `5519463`、`e49474f`、`10eb9f5`、`f5bbf4c` |
| 关键教训 | MTP 修复必须跑端到端 accept rate；`tools/repro_hidden_gap.py` 不覆盖 MTP drafter/shared-expert 路径 |

## 2. 线上表现

坏状态通常长这样：

- `python -m atom.entrypoints.openai_server ... --method mtp` 能启动；
- `/health` 返回 OK；
- GPU VRAM 确实加载了模型；
- 单 prompt completion 语义看起来没坏；
- 但 MTP stats 显示：
  - `Acceptance rate: 0.00%` 或接近 0%；
  - `Average toks/fwd: 1.00`；
  - speculative decoding 没有收益，甚至因为多跑了 MTP forward 而变慢。

这类问题不容易靠 crash 发现：

- BF16 到 FP8 的 dtype cast 是合法操作，不会抛异常；
- tensor shape 能对上；
- `weight_scale` 是合法 `nn.Parameter`，只是没有从 checkpoint 加载，仍是
  `torch.empty(...)` 的未初始化值；
- GEMM kernel 不知道这个 scale 是脏数据。

## 3. 共同 corruption 链路

入口投影层如果被错误地按 FP8 量化层构造，会走到同一条静默损坏链路。

### 3.1 构造阶段：按全局 FP8 spec 创建参数

相关代码在 [`atom/model_ops/linear.py`](../atom/model_ops/linear.py)：

```python
layer_quant_config = quant_config.get_layer_quant_config(prefix)
quant_type = layer_quant_config.quant_type
params_dtype = layer_quant_config.quant_dtype

self.weight = atom_parameter(torch.empty((out, in), dtype=params_dtype))
self.weight_scale = atom_parameter(torch.empty((out // 128, in // 128), dtype=torch.float32))
```

当 `prefix` 没命中 exclude，或者 checkpoint metadata 漏写 exclude 时，
`get_layer_quant_config(...)` 会回退到全局 FP8 量化配置：

- `self.weight` 被创建成 FP8；
- `self.weight_scale` 被注册出来；
- forward 的 dispatch 也固定成 FP8 block-scale 路径。

### 3.2 加载阶段：BF16 权重被静默 cast 到 FP8

`weight_loader_process` 遇到 dtype 不一致时会做转换：

```python
if param.data.dtype != loaded_weight.dtype:
    loaded_weight = loaded_weight.to(param.data.dtype)
```

因此 checkpoint 里的 BF16 entry projection 会被直接 cast 到 FP8 槽位。
这一步不会报错，但会丢掉 BF16 数值范围与精度。

更关键的是：checkpoint 没有 `*.weight_scale`，所以 loader 不会写入
`self.weight_scale`。这个参数保持未初始化值，后续 GEMM 会直接消费它。

### 3.3 Forward 阶段：GEMM 使用脏 scale

FP8 block-scale 路径会把未初始化的 `weight_scale` 传入 AITER GEMM：

```python
y = gemm_a8w8_blockscale(x, self.weight, x_scale, self.weight_scale, dtype=otype)
```

结果是 MTP draft logits 被污染，验证阶段几乎不接受 draft token。

## 4. Bug A：DeepSeek MTP `eh_proj` 的 checkpoint metadata 漏列 exclude

### 4.1 触发条件

DeepSeek R1/R1-0528/V3/V3.2 的 FP8 checkpoint 中，MTP 入口投影实际是
BF16：

```text
model.layers.61.eh_proj.weight        bf16, shape=(7168, 14336)
model.layers.61.eh_proj.weight_scale  不存在
```

但 HF `quantization_config.modules_to_not_convert` 没有列出 `eh_proj`。

commit `5d34e9e` 之后，
[`DeepSeekMultiTokenPredictorLayer.eh_proj`](../atom/models/deepseek_mtp.py)
开始通过 `ReplicatedLinear(..., quant_config=atom_config.quant_config, prefix=...)`
构造。由于 metadata 没有 exclude，`eh_proj` 会回退到全局 FP8 spec，触发
第 3 节的 corruption 链路。

### 4.2 为什么这不是普通 dtype mismatch

`eh_proj` 本身是小参数量、高精度敏感的入口投影：

- 输入是 token embedding 和 base model hidden state 的拼接；
- 输出是 MTP block 的起始 hidden state；
- 一旦第一步 draft hidden 被污染，accept rate 会直接塌到 0% 附近。

因此不能简单相信全局量化配置；必须识别 checkpoint 里这层到底有没有
`weight_scale`。

## 5. Bug B：Qwen3-Next MTP `fc` 构造时漏传 `prefix`

### 5.1 触发条件

Qwen3-Next-FP8 checkpoint 的入口投影是：

```text
mtp.fc.weight        bf16, shape=(2048, 4096)
mtp.fc.weight_scale  不存在
```

这次 HF metadata 是正确的，`modules_to_not_convert` 里有 `mtp.fc`。问题在
ATOM：`Qwen3NextMultiTokenPredictor.fc` 构造 `ColumnParallelLinear` 时没有传
`prefix=f"{prefix}.fc"`。

修复前：

```python
self.fc = ColumnParallelLinear(
    self.config.hidden_size * 2,
    self.config.hidden_size,
    bias=False,
    quant_config=quant_config,
)
```

`ColumnParallelLinear` 默认 `prefix=""`，因此 exclude 匹配器拿空字符串去匹配
`mtp.fc`，不会命中，最终又回退到全局 FP8 spec。

### 5.2 修复方式

commit `5519463` 补上 prefix：

```python
self.fc = ColumnParallelLinear(
    self.config.hidden_size * 2,
    self.config.hidden_size,
    bias=False,
    quant_config=quant_config,
    prefix=f"{prefix}.fc",
)
```

这与 `qwen3_5_mtp.py` 的已有写法一致。修复后，
`get_layer_quant_config("mtp.fc")` 能命中 checkpoint metadata 中的 exclude，
`fc` 会按 BF16 层构造。

## 6. 修复设计

### 6.1 Qwen3-Next：传正确 prefix

Qwen3-Next 的问题是 ATOM 构造层时查错了 key。修复就是传入与 checkpoint
权重名一致的 prefix，让现有 exclude 机制生效。

影响范围：

- 修复 Qwen3-Next-FP8 `mtp.fc`；
- 与 Qwen3.5 MTP 现有实现对齐；
- 不改变其他模型。

### 6.2 DeepSeek/MiMo：构造前做 checkpoint 内省

DeepSeek 的问题是 checkpoint metadata 漏写 exclude。只靠
`modules_to_not_convert` 不可靠，因此 commit `e49474f` 新增
`ckpt_has_tensor_suffix(model_path, suffix)`：

```python
def ckpt_has_tensor_suffix(model_path: str, suffix: str) -> bool:
    """Return True if any checkpoint tensor key ends with suffix."""
```

MTP 构造前检查 checkpoint 里是否有 `eh_proj.weight_scale`：

- 如果没有，说明入口投影不是量化权重，给 quant config 追加默认 exclude：
  `*.eh_proj`；
- 如果有，说明 checkpoint 确实提供了量化 scale，保持量化路径。

这个判断被接在 `DeepSeekMTP.__init__` 和 `MiMoV2FlashMTP.__init__` 上。

### 6.3 为什么不在 weight loader 里修

weight loader 能看到 `param.dtype=fp8` 和 `loaded_weight.dtype=bf16`，但这个时机
已经太晚：

- `weight` 已经按 FP8 dtype 分配；
- `weight_scale` 已经注册；
- `quant_type` 已经固定；
- forward dispatch 已经选择 FP8 路径。

在 loader 中“修复”需要对 `nn.Module` 做手术：重建参数、删除
`weight_scale`、改 `quant_type`、换 forward 路径。这不仅复杂，也容易和
`torch.compile`/CUDAGraph 生命周期冲突。

更合理的边界是 MTP 构造阶段：MTP 类知道 `eh_proj` 是入口投影，也知道这类层
常常与主干量化策略不同。用 checkpoint 里是否存在 `*.weight_scale` 作为信号，
可以在分配参数之前决定正确 dtype。

## 7. 受影响 checkpoint 矩阵

| Checkpoint | entry projection | checkpoint 权重 | 是否有 scale | 修复前 | 修复后 |
| --- | --- | --- | --- | --- | --- |
| DeepSeek R1/R1-0528/V3/V3.2 FP8 | `model.layers.N.eh_proj` | BF16 | 否 | 被误构造成 FP8，accept 约 0% | checkpoint 内省补 exclude，按 BF16 构造 |
| DeepSeek R1-0528 Quark/MXFP4 MTP | `model.layers.N.eh_proj` | BF16 | 否 | Quark exclude 已覆盖 | 去重 no-op |
| GLM-5/GLM-5.1 FP8 | `model.layers.N.eh_proj` | BF16 | 否 | HF exclude 已覆盖 | 去重 no-op |
| GLM-4.7 compressed-tensors | `model.layers.N.eh_proj` | BF16 | 否 | `ignore` 已覆盖 | 去重 no-op |
| GLM-5.1 MXFP4 | `model.layers.N.eh_proj` | 量化权重 | 是 | 正确量化 | 内省发现有 scale，不加 exclude |
| Qwen3-Next-80B-A3B-Instruct-FP8 | `mtp.fc` | BF16 | 否 | 漏 prefix，误构造成 FP8 | 补 prefix，命中 `mtp.fc` exclude |
| Qwen3.5 FP8/MXFP4 | `mtp.fc` | BF16 | 否 | 已正确传 prefix | 不变 |
| MiMo-V2-Flash MTP | `eh_proj` | 本地未验证 | 未知 | 潜在同类风险 | 同 DeepSeek 的内省逻辑覆盖 |

结论：

- 多数观察到的 MTP entry projection 都是 BF16 + 无 scale；
- GLM-5.1 MXFP4 是少数真的量化 entry projection 的例子；
- metadata 是否声明 exclude 不一致，因此磁盘上是否存在 `*.weight_scale` 是更可靠信号。

## 8. 端到端验证时新增发现的两个独立 bug

在验证 entry-proj 修复时，Qwen3-Next-FP8 仍然出现 accept rate 约 0%。继续查
server 启动日志和 MTP stats，又发现两个与入口投影无关的问题。

### 8.1 Bug C：`loader.py` 直接访问 `hf_config.n_routed_experts`

触发位置：

```python
name = name.replace(
    maybe_matching_name,
    f"mlp.experts.{hf_config.n_routed_experts}.",
)
```

`Qwen3NextConfig` 没有 `n_routed_experts` 字段。shared expert fusion 路径启用时，
这里会在 MTP drafter 加载阶段抛 `AttributeError`，导致 EngineCore shutdown。

commit `10eb9f5` 改为与同文件后续代码一致的 `getattr` guard：

```python
name = name.replace(
    maybe_matching_name,
    f"mlp.experts.{getattr(hf_config, 'n_routed_experts', 0)}.",
)
```

DeepSeek 系 config 有该字段，行为不变；非 DeepSeek config fallback 到 0，避免
错误访问。

### 8.2 Bug D：`Qwen3NextMTP.packed_modules_mapping` 漏掉 shared expert gate

Bug C 修复后，server 能启动，但日志仍有 dropped tensor：

```text
mtp.layers.0.mlp.shared_expert_gate.weight
  -> model.layers.0.mlp.shared_expert_gate.weight
```

Qwen3-Next base model 中，`shared_expert_gate` 与 `.gate.` 被 packed 到同一个
`gate` 参数的不同 shard：

```python
packed_modules_mapping = {
    ".gate.": (".gate.", 0),
    "shared_expert_gate": ("gate", 1),
}
```

但 `Qwen3NextMTP.packed_modules_mapping` 修复前只拷了 qkv 和 gate/up projection，
漏掉这两项。结果 shared expert gate 权重被 silently dropped，MTP 里对应参数保持
未初始化值，单独就能让 accept rate 归零。

commit `f5bbf4c` 补上：

```python
packed_modules_mapping = {
    "q_proj": ("qkv_proj", "q"),
    "k_proj": ("qkv_proj", "k"),
    "v_proj": ("qkv_proj", "v"),
    "gate_proj": ("gate_up_proj", 0),
    "up_proj": ("gate_up_proj", 1),
    ".gate.": (".gate.", 0),
    "shared_expert_gate": ("gate", 1),
}
```

## 9. 验证方法

### 9.1 启动前清编译缓存

按仓库规则，server 重启前先清 ATOM compile cache：

```bash
rm -rf /root/.cache/atom/*
```

### 9.2 启动 openai_server

Qwen3-Next-FP8 TP4：

```bash
HIP_VISIBLE_DEVICES=4,5,6,7 AITER_LOG_LEVEL=WARNING \
  python -m atom.entrypoints.openai_server \
    --model /data/amd_int/models/Qwen3-Next-80B-A3B-Instruct-FP8 \
    --kv_cache_dtype fp8 -tp 4 --method mtp \
  > server.log 2>&1
```

DeepSeek-R1-0528 TP8：

```bash
HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 AITER_LOG_LEVEL=WARNING \
  python -m atom.entrypoints.openai_server \
    --model /data/amd_int/models/DeepSeek-R1-0528 \
    --kv_cache_dtype fp8 -tp 8 --method mtp \
  > server.log 2>&1
```

### 9.3 先检查加载日志

accept rate 之前，必须先确认没有加载期 warning：

```bash
grep -E "load_model:.*(NOT loaded|silently dropped)" server.log
```

期望输出为空。只要还有 unloaded/dropped tensor，MTP accept rate 仍可能是 0%，
这时继续看 accept 指标意义不大。

### 9.4 跑 smoke 或 lm_eval

smoke 需要触发足够 draft token，才会打印 MTP stats。示例：

```bash
python repro_main_logs/smoke.py /data/amd_int/models/Qwen3-Next-80B-A3B-Instruct-FP8
```

CI/合并前建议跑 `lm_eval` 的 MTP accept rate 回归，至少覆盖：

- DeepSeek-R1-0528 FP8 + MTP；
- Qwen3-Next-FP8 + MTP。

### 9.5 判定标准

| 信号 | 坏状态 | 好状态 |
| --- | --- | --- |
| `load_model: NOT loaded` | 存在 | 不应存在 |
| `load_model: silently dropped` | 存在 | 不应存在 |
| `Average toks/fwd` | 约 1.00 | 大于 1.0 |
| `Acceptance rate` | 约 0% | 明显大于 0%，Qwen3-Next smoke 约 60%+ |

Qwen3-Next-FP8 TP4 实测：

| 配置 | NOT loaded | silently dropped | MTP accept | avg toks/fwd | 32x384 smoke |
| --- | --- | --- | --- | --- | --- |
| main (`679422d`) | `model.fc.weight_scale` | `shared_expert_gate.weight` | 0.00% | 1.00 | 约 172 tok/s |
| 仅 entry-proj 修复 | 无 | 仍存在 | 0.00% | 1.00 | 约 273 tok/s |
| 四个修复齐全 | 无 | 无 | 60.14% | 1.60 | 约 604 tok/s |

## 10. 为什么 `tools/repro_hidden_gap.py` 不够

`tools/repro_hidden_gap.py` 只比较 base model MLA hidden state，不覆盖：

- MTP drafter 权重加载；
- Qwen3-Next MTP 的 shared expert gate；
- MTP forward；
- speculative verification 和 accept/reject 统计。

因此它可以作为 base model hidden gap 的定位工具，但不能作为 MTP 修复的唯一验证。
任何 MTP 相关修复都必须补端到端 accept rate 或 `lm_eval` MTP metrics。

## 11. 新模型/新 MTP 架构 checklist

添加或迁移 MTP 架构时，至少检查这些点：

1. 入口投影构造是否传了与 checkpoint key 对齐的 `prefix`。
2. checkpoint 里的入口投影是否真的量化：是否存在对应 `*.weight_scale`。
3. HF/quark/compressed-tensors metadata 的 exclude 字段是否与磁盘 dtype 一致。
4. MTP 子类的 `packed_modules_mapping` 是否完整继承或同步 base model 的 packing 规则。
5. 启动日志中是否完全没有 `NOT loaded` 或 `silently dropped`。
6. 端到端 MTP accept rate 是否恢复，不能只看单 prompt 语义输出。

## 12. 后续建议

这组修复解决了当前已知的 MTP entry projection 和 Qwen3-Next loading 问题。为了让
未来类似 bug 更早暴露，可以继续做两个通用增强：

- **加载完成后的 sanity assert**：遍历量化 `LinearBase`，确认需要 scale 的层确实
  从 checkpoint 加载了 scale，而不是保持未初始化参数。
- **checkpoint metadata audit 工具**：对比 `modules_to_not_convert`/`exclude`/`ignore`
  与实际 tensor dtype、`weight_scale` 是否存在，提前发现发布包 metadata 漏项。

