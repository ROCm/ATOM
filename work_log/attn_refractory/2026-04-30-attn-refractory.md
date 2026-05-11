# 2026-04-30 ATOM Attention Refactory Session Summary

## 1. 本次会话的核心目标

本次连续讨论的主线是围绕 `ATOM plugin` 中 attention backend 的重构方向展开，重点包括：

- 调研当前 `ATOM plugin` attention backend 的架构问题
- 评估 `vLLM plugin` 与 `SGLang plugin` 的耦合程度
- 分析 SGLang plugin 在支持新模型、特别是非传统 MHA/MLA 类型 backend 时会遇到的困难
- 讨论如何在 plugin 层把 `vLLM` 与 `SGLang` 解耦
- 讨论 `SGLang` 中 `hybrid/composite backend` 的接入方式
- 评估三种模式共享 ATOM attention 实现的难度：
  - `ATOM native/server`
  - `ATOM vLLM plugin`
  - `ATOM SGLang plugin`

## 2. 对当前 ATOM plugin attention 架构的主要判断

### 2.1 当前不是一个真正统一的 attention backend 架构

本次调研得到的一个核心结论是：

- `vLLM plugin` 和 `SGLang plugin` 虽然都叫 “ATOM attention”
- 但它们并不是通过同一套干净的 backend contract 接进来的
- 更准确地说，是两套宿主接入模型外加多层 patch / adapter / runtime glue 共同构成的系统

### 2.2 当前的主要问题不是底层 kernel，而是 runtime / ownership / patch 层次

现有问题主要集中在：

- 全局状态过重
  - `_CURRENT_FRAMEWORK`
  - `current_atom_config`
  - `ops.Attention`
- vLLM 与 SGLang 的接入模型差异很大，但还共享 `plugin` 根目录里一批伪公共逻辑
- 很多扩展点依赖 monkey patch 和 runtime override
- SGLang 侧对 DeepSeek MLA 已经上卷到 model-level patch，而不是单纯 backend 替换

### 2.3 attention 文件结构的拆分轴不统一

本次对 `plugin` 目录下 attention 相关文件的解读认为，目前至少混杂了三层不同对象：

- host/runtime glue
- backend runtime core
- model-specific specialization

尤其在 `SGLang` 侧：

- `radix_attention.py` 更像 adapter
- `sgl_attn_backend.py` 更像 full-attn runtime backend core
- `sgl_attention_mla.py` 更像 DeepSeek MLA specialization

而在根目录：

- `attention.py`
- `attention_mha.py`
- `attention_mla.py`
- `attention_mla_sparse.py`

虽然放在 `plugin/` 根目录，但职责上明显更接近 `vLLM plugin` 的 glue / patch 层，而不是真正共享的 plugin runtime core。

## 3. 关于 vLLM plugin 与 SGLang plugin 的耦合

### 3.1 结论：耦合度高

这次讨论中对二者耦合的判断是：

- 不是轻量共享几个工具函数
- 而是共享了一套 runtime state、attention 抽象、selector 和 plugin-mode 判断方式

关键耦合点包括：

- `atom/plugin/prepare.py`
- `atom/plugin/register.py`
- `atom/plugin/config.py`
- `atom.plugin.prepare.is_vllm()/is_sglang()/is_plugin_mode()`
- `ops.Attention` 的全局切换

### 3.2 关键判断

真正的问题不是 “目录没有拆开”，而是：

**host runtime 差异没有被限制在 adapter 边界，而是泄漏进了 backend 选择、metadata 组织和全局状态模型。**

## 4. 关于 plugin 层解耦的共识

### 4.1 plugin 层不应该继续保留共享 runtime

讨论最终倾向于一个更激进但更干净的方向：

- `atom/plugin/vllm/` 是一个完整子系统
- `atom/plugin/sglang/` 是另一个完整子系统
- `atom/plugin/` 根目录不再承担共享 runtime 中心的角色

共享只保留给更底层的 `ATOM core`，例如：

- `model_ops`
- kernels
- loader
- metadata helpers
- model-family specialization 中真正 host-agnostic 的部分

### 4.2 “bootstrap” 的定义

本次还明确了 `bootstrap` 在本讨论语境中的含义：

- 不是核心执行逻辑
- 而是插件被宿主框架加载时的最早期接线/初始化层

也就是：

- 注册扩展点
- 安装 patch
- 选择 wrapper / adapter / backend
- 建立 host 与 plugin 的连接关系

## 5. 已经做过的代码原型

本次会话中，为了验证“按 host 拆入口”的思路，已经做了一批原型代码修改：

### 5.1 新增的 bootstrap / prepare / register 原型

- `atom/plugin/sglang/bootstrap.py`
- `atom/plugin/vllm/bootstrap.py`
- `atom/plugin/sglang/prepare.py`
- `atom/plugin/sglang/register.py`

### 5.2 `prepare.py` 已部分收缩为兼容层

- 实际的 SGLang prepare 逻辑已经移动到 `atom/plugin/sglang/prepare.py`
- `atom/plugin/prepare.py` 现在更像一个 legacy shim
- 但它仍保留了 framework-state helper（因为仓库里很多地方还在依赖）

### 5.3 SGLang register 逻辑已开始下沉

SGLang 独有的几块逻辑已经迁入：

- `register_ops_to_sglang`
- `set_sglang_attn_cls`
- `init_aiter_dist_for_sglang`
- `bootstrap_sglang_runtime`

共享 `atom/plugin/register.py` 目前更多是兼容 facade。

### 5.4 说明

这些改动的定位是：

- 用来显式化未来结构
- 不是完整重构
- 还保留了较多兼容层

## 6. 关于 SGLang attention backend 的重要判断

### 6.1 `ATOMAttnBackendForSgl` 与 public `AiterAttnBackend`

在一次反复讨论后，本次对它们的关系收敛为：

- 在 backend 角色位置上，两者基本是 apple-to-apple
- `ATOMAttnBackendForSgl` 不是“更高层的新东西”
- 而是 `SGLang full-attention runtime backend` 的 ATOM 对位实现

因此更合适的重构思路不是怀疑它的存在合法性，而是：

- 保留它作为 full-attn backend core
- 再去拆它内部的 metadata / kv_cache / decode / graph 等职责

### 6.2 DeepSeek MLA 的特殊性

`sgl_attention_mla.py` 暴露出一个很重要的事实：

- 对 `MLA`，尤其是 `DeepSeek MLA`
- SGLang plugin 已经不是单纯换 backend
- 而是 patch 了模型级 `forward`

这说明：

**MLA 在 SGLang plugin 里已经进入了 model specialization 维度。**

## 7. 关于 GDN / KDA / Lightning / Mamba2 等非传统 “attention”

### 7.1 调研结论

public SGLang 已经证明，系统里不止 full-attention backend，还存在：

- linear attention backend
  - `GDNAttnBackend`
  - `KDAAttnBackend`
  - `LightningAttentionBackend`
  - `Mamba2AttnBackend`
- hybrid/composite backend
  - `HybridLinearAttnBackend`
  - `HybridAttnBackend`
- wrapper/composition backend
  - `TboAttnBackend`

### 7.2 对当前 ATOM plugin 设计造成的挑战

当前 ATOM plugin 的主抽象仍偏向：

- MHA
- MLA
- full attention runtime

这会带来几个结构性问题：

1. 还没有把 `linear backend` 当成一等公民
2. 还没有给 `hybrid/composite backend` 预留独立宿主位置
3. 容易把 algorithm backend、kernel backend、host glue 混成一层
4. 容易继续把 `GDN` 等路径硬塞回 full-attn backend 体系

### 7.3 当前共识

后续不应该继续只谈 “attention backend 重构”，而应该升级为：

**sequence backend / mixer backend 体系重构**

建议至少在设计上并列三类：

- `full_attn`
- `linear_attn`
- `hybrid/composite`

## 8. hybrid/composite backend 在 plugin 侧的接入思路

### 8.1 核心判断

由于 ATOM SGLang plugin 的启动方式是：

- 启动 `sglang server`
- 再通过 plugin runtime override 接管 model / attention

所以要接 `hybrid/composite backend`，不可避免需要 patch 某个 seam。

### 8.2 最好的 seam

本次讨论认为，public SGLang 里最好的 seam 是：

- `model_runner.py` 中导入并调用的 `attn_backend_wrapper`

注意：

- 它不是 `ModelRunner` 的成员方法
- 而是 `model_runner.py` 模块级导入的符号

所以如果要 patch，更稳的是 patch：

- `sglang.srt.model_executor.model_runner.attn_backend_wrapper`

而不是仅 patch `attention_registry.attn_backend_wrapper`。

### 8.3 最推荐的方式

对 hybrid/composite backend 的接入方式，本次最终倾向于：

**plugin-own 一个薄的 composite wrapper/factory，只在 backend 构造 seam 做单点 patch。**

不推荐：

- 深度 monkey patch public hybrid backend 实现
- 一开始就复制整套 public linear/full backend 逻辑

更推荐：

- full side 先用 ATOM full backend
- linear side 初期先复用 public SGLang 的 `GDNAttnBackend` / `KDAAttnBackend` / `LightningAttentionBackend` / `Mamba2AttnBackend`
- composite wrapper 由 plugin 侧拥有

## 9. 关于三种模式共享 ATOM attention 实现的难度

本次对：

- `ATOM native/server`
- `ATOM vLLM plugin`
- `ATOM SGLang plugin`

三种模式共享 ATOM attention 实现，得到的判断如下。

### 9.1 MHA

难度：`Medium-High`

原因：

- `ATOM native` 与 `ATOM vLLM plugin` 已经在 `PagedAttentionImpl` 等层共享较多
- 真正难的是 `SGLang plugin` 这一侧
- 它不走 `PagedAttention` 的 ATOM 主路径，而走：
  - `RadixAttention`
  - `ATOMAttnBackendForSgl`
  - `ForwardBatch`

所以 MHA 共享的主要困难在于：

**runtime orchestration 差异**

### 9.2 MLA

难度：`High`

原因：

- native/server 与 vLLM plugin 仍然较多共享 `MLAAttention`
- 但 SGLang plugin 下的 DeepSeek MLA 已经上卷到 model specialization
- 不只是 backend 不同，连 forward 组织方式都不同

所以 MLA 的困难在于：

**runtime orchestration + model specialization 双重叠加**

## 10. 已产出的文档与图

本次会话过程中，已额外产出下列文档/图，用于不同角度说明问题。

### 10.1 代码目录内 Markdown

- `atom/plugin/decoupling-diagram.md`
- `atom/plugin/vllm-integration-architecture.md`
- `atom/plugin/sglang-attention-backend-survey.md`

### 10.2 Canvas / 可视化分析

- `atom-attention-backend-architecture-review.canvas.tsx`
- `atom-plugin-coupling-risk-analysis.canvas.tsx`
- `atom-plugin-decoupling-diagram.canvas.tsx`
- `atom-vllm-plugin-architecture.canvas.tsx`
- `atom-attention-sharing-modes.canvas.tsx`

### 10.3 这些产物对应的主题

- 当前 attention backend 架构缺陷
- vLLM / SGLang plugin 耦合与风险
- plugin 解耦方向与 bootstrap 理解
- vLLM plugin 集成架构
- public SGLang backend survey
- 三种模式共享 ATOM attention 的难度评估

## 11. 当前最值得继续推进的方向

如果延续本次会话的结论，后续工作最值得按下面顺序推进：

1. 继续收缩 `atom/plugin/` 根目录的 shared runtime 语义
2. 把 `full_attn / linear_attn / hybrid` 三层作为 plugin 后续结构设计的一等公民
3. 在 `SGLang plugin` 侧先补出 `hybrid/composite` 组合层
4. 初期复用 public SGLang linear backend，优先验证 runtime 结构是否成立
5. 然后再评估哪些 linear backend 需要逐步替换成 ATOM 自己的实现
6. 对 `MLA` 尽早拆开：
   - generic MLA runtime
   - DeepSeek specialization

## 12. 一句话总结

本次会话最终把问题收敛为：

**当前 ATOM plugin attention 的关键矛盾，不是底层 kernel 能不能共享，而是 vLLM / SGLang / native 三种模式在 runtime、metadata、model specialization 和 backend ownership 上没有对齐；后续重构应从“统一 attention backend”升级为“重新定义 plugin 的 sequence backend / host-owned runtime 结构”。**
