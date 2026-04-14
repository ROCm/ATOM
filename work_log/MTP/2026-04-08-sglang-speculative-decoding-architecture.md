# 2026-04-08 SGLang Speculative Decoding 架构笔记

## 文档目的

这份文档用于从 **SGLang 整体架构** 的角度梳理 speculative decoding
（推测解码）的实现方式，重点回答下面几个问题：

- SGLang 在启动阶段如何决定是否进入 speculative decoding
- target model 和 draft model 是如何被构造和组织的
- scheduler、worker、batch、attention backend 分别承担什么职责
- EAGLE / EAGLE3 / NEXTN / STANDALONE / NGRAM 在 SGLang 里是怎样映射的
- v1 与 v2（overlap/spec v2）在控制流和数据结构上有何差异
- 遇到问题时，应该优先看哪些代码位置

本文会尽量从“系统设计”和“源码位置”两个维度同时展开，方便后续复盘。


## 一句话总结

从架构上看，SGLang 的 speculative decoding 可以概括成：

- **配置层** 先在 `ServerArgs` 中解析算法与 draft 参数
- **模型配置层** 决定 draft model 应该映射成哪一个架构名
- **调度层** 通过 `Scheduler` 把执行入口从普通 `TpModelWorker` 切到
  speculative orchestrator
- **worker 层** 维护 target worker 与 draft worker 的协作关系
- **batch 层** 用 `ScheduleBatch -> ModelWorkerBatch -> ForwardBatch` 三层结构
  将调度语义转为 GPU 执行语义
- **attention/backend 层** 再按 `ForwardMode` 和 `SpecInput` 区分
  `decode / target_verify / draft_extend`

也就是说，speculative decoding 在 SGLang 里不是“加一个 draft model”这么简单，
而是一整套跨：

- 配置
- 调度
- 模型加载
- batch 编排
- attention metadata

的系统设计。


## 1. 启动入口：ServerArgs 如何决定 speculative decoding

### 1.1 关键配置项

核心入口文件：

- `sglang/python/sglang/srt/server_args.py`

关键字段位置：

- `speculative_algorithm`：约 `480`
- `speculative_draft_model_path`：约 `481`
- `speculative_num_steps`：约 `484`
- `speculative_num_draft_tokens`：约 `486`

这些字段决定：

- 用哪种 speculative 算法
- draft model 从哪里加载
- 每轮 draft 几步
- 每轮最多提议多少 draft token


### 1.2 `NEXTN` 在 SGLang 里的真实含义

很多人第一次看会以为 `NEXTN` 是一条完全独立的 speculative runtime。
实际上不是。

在：

- `sglang/python/sglang/srt/server_args.py`
- `_handle_speculative_decoding()` 逻辑中

有一个关键规范化：

- `NEXTN -> EAGLE`

对应代码位置：

- `server_args.py` 约 `2680-2681`

这意味着：

- 用户在命令行里写 `--speculative-algorithm NEXTN`
- 进入运行时后，SGLang 会把它归并到 `EAGLE` 这套 speculative worker 流程里

也就是说：

- `NEXTN` 更像是“draft model 形态 / 语义”
- `EAGLE` 更像是“runtime orchestration 机制”


### 1.3 spec v2 与 overlap scheduler

仍然是在：

- `server_args.py`
- `_handle_speculative_decoding()`

关键逻辑位置：

- `2696-2716`

SGLang 会做一件重要的系统级决策：

- 如果 speculative 算法属于 `EAGLE / EAGLE3 / STANDALONE`
- 且环境变量 `SGLANG_ENABLE_SPEC_V2=True`
- 则开启 overlap schedule（即 spec v2）

否则：

- 会退回到不带 overlap 的传统路径（可以理解为 spec v1）

同时还有一些额外约束：

- spec v2 目前只支持 `topk = 1`
- 使用 speculative 时会关闭 mixed chunked prefill


### 1.4 DeepSeek / MTP 与 `speculative_draft_model_path`

同一段逻辑里还有一个对 DeepSeek 很关键的行为：

- 对 `DeepseekV3ForCausalLM`、`DeepseekV32ForCausalLM`、`GlmMoeDsaForCausalLM`
  等架构
- 如果没有显式传 `speculative_draft_model_path`
- 会自动把它设成主模型路径

关键位置：

- `server_args.py` 约 `2725-2748`

这就是为什么日志里会有类似：

- `DeepSeek MTP does not require setting speculative_draft_model_path.`

的提示。

这说明 SGLang 把 DeepSeek MTP / NextN 看成是某种“和 target 模型强绑定”的
draft 形态，而不是完全独立的小模型。


## 2. 算法层：SpeculativeAlgorithm 与 worker 工厂

核心文件：

- `sglang/python/sglang/srt/speculative/spec_info.py`

### 2.1 算法枚举

关键枚举：

- `SpeculativeAlgorithm`

包含：

- `EAGLE`
- `EAGLE3`
- `STANDALONE`
- `NGRAM`
- `NONE`

关键位置：

- `spec_info.py` 约 `15-23`


### 2.2 worker 工厂

最关键的方法：

- `SpeculativeAlgorithm.create_worker()`

关键位置：

- `spec_info.py` 约 `52-105`

这个函数负责把：

- 算法类型
- overlap 是否开启
- multi-layer eagle 是否开启

映射成具体 worker 类。

典型映射关系：

- `EAGLE + overlap` -> `EAGLEWorkerV2`
- `EAGLE + no overlap` -> `EAGLEWorker`
- `STANDALONE + overlap` -> `StandaloneWorkerV2`
- `STANDALONE + no overlap` -> `StandaloneWorker`
- `NGRAM` -> `NGRAMWorker`


### 2.3 什么叫 “supports_spec_v2”

还有个很重要的方法：

- `supports_spec_v2()`

关键位置：

- `spec_info.py` 约 `49-50`

含义是：

- 当前算法是否支持 overlap/spec v2 抽象

目前只有：

- `EAGLE`
- `STANDALONE`

对应为真。


## 3. 调度层：Scheduler 如何把普通模型调度切成 speculative 调度

核心文件：

- `sglang/python/sglang/srt/managers/scheduler.py`

### 3.1 初始化顺序

关键位置：

- `maybe_init_draft_worker()`：约 `527-554`
- `init_model_worker()`：约 `556-564`

逻辑顺序是：

1. 先建 `tp_worker`
2. 如果 speculative 开启，再建 `draft_worker`
3. 决定 `self.model_worker` 指向谁

代码语义：

- 没开 speculative：
  - `self.model_worker = self.tp_worker`
- 开了 speculative：
  - `self.model_worker = self.draft_worker`


### 3.2 为什么 `self.model_worker = self.draft_worker`

这里名字非常容易误导。

`scheduler.draft_worker` 并不一定是一个“纯 draft model worker”，它更像是：

- speculative orchestrator

例如：

- `EAGLEWorker`
- `EAGLEWorkerV2`

也就是说：

- scheduler 并不是“把 target worker 替换掉了”
- 而是把执行入口切到了一个能同时协调 target + draft 的总控 worker


### 3.3 `run_batch()` 的差异

关键位置：

- `scheduler.py` 约 `2360-2426`

这里能看出 v1 与 v2 在 batch 抽象上的差异：

- 开 overlap/spec v2 时：
  - `worker_batch_or_batch = batch.get_model_worker_batch()`
  - 下游主要处理 `ModelWorkerBatch`
- 非 overlap 的传统 speculative v1：
  - 会直接把 `ScheduleBatch` 传给 `model_worker.forward_batch_generation()`

这也是为什么你有时候会看到：

- 有的 speculative worker 收的是 `ScheduleBatch`
- 有的 speculative worker 收的是 `ModelWorkerBatch`

这不是 bug，而是新老抽象并存。


## 4. Worker 层：target worker、draft worker 与 orchestrator 的关系

### 4.1 普通 target worker

核心文件：

- `sglang/python/sglang/srt/managers/tp_worker.py`

`TpModelWorker` 是普通模型执行单元，负责：

- 初始化 `ModelConfig`
- 初始化 `ModelRunner`
- 提供 `forward_batch_generation()`

关键位置：

- `_init_model_config()`：约 `320-336`
- `_init_model_runner()`：约 `338-358`
- `forward_batch_generation()`：约 `442+`


### 4.2 target 和 draft 的分流在哪里发生

在 `TpModelWorker._init_model_config()` 中：

- 如果 `is_draft_worker=False`，用主模型路径
- 如果 `is_draft_worker=True`，用 `speculative_draft_model_path`

关键位置：

- `tp_worker.py` 约 `323-336`

这就是 target 和 draft 最底层模型配置分流的地方。


### 4.3 `EAGLEWorker`（v1）

核心文件：

- `sglang/python/sglang/srt/speculative/eagle_worker.py`

`EAGLEWorker` 的特点：

- 自己继承自 `TpModelWorker`
- 运行时同时持有：
  - target worker
  - 自己这套 draft model runner

其 `forward_batch_generation()` 的大致逻辑是：

- 如果是 extend：
  - 先 `forward_target_extend`
  - 再 `forward_draft_extend`
- 如果是 decode：
  - 先 `draft()`
  - 再 `verify()`
  - 再 `forward_draft_extend_after_decode()`

关键位置：

- `eagle_worker.py` 约 `278-337`


### 4.4 `EAGLEWorkerV2`（spec v2 / overlap）

核心文件：

- `sglang/python/sglang/srt/speculative/eagle_worker_v2.py`

v2 与 v1 最大的结构差异是：

- 外层 `EAGLEWorkerV2` 是 orchestrator
- 内层还有一个 `EagleDraftWorker`
- `EagleDraftWorker` 再内嵌一个真正的 draft `TpModelWorker`

关键类：

- `EagleDraftWorker`：约 `82`
- `EAGLEWorkerV2`：约 `607`

这层设计的意义是：

- 把 draft 逻辑进一步模块化
- 更方便做 overlap 和独立的 draft graph / backend 管理


### 4.5 `StandaloneWorkerV2`

核心文件：

- `sglang/python/sglang/srt/speculative/standalone_worker_v2.py`

它和 `EAGLEWorkerV2` 的主要区别不是调度框架，而是：

- draft model 不再共享 target 的 embedding / lm_head

在源码里可以看到：

- `StandaloneDraftWorker.init_lm_head()` 明确覆写为空实现

也就是：

- standalone draft 用自己的一套 embedding/head
- 不走与 target 的共享逻辑


## 5. 模型配置与 draft 架构改写

核心文件：

- `sglang/python/sglang/srt/configs/model_config.py`

### 5.1 `ModelConfig.from_server_args()`

这是 target / draft `ModelConfig` 的统一入口。

关键位置：

- `from_server_args()`：约 `238+`


### 5.2 `_config_draft_model()`

最关键的方法：

- `_config_draft_model()`

关键位置：

- `model_config.py` 约 `277-340`

对于 DeepSeek：

- 若 `is_draft_model=True`
- 且原始架构是 `DeepseekV3ForCausalLM`

就会改写为：

- `DeepseekV3ForCausalLMNextN`

这是 draft 侧为什么会变成 NextN 壳子的核心原因。


### 5.3 这和 ATOM plugin 的关系

这也是当前 `ATOM plugin` 只接管 target、没接管 draft 的根源：

- ATOM external model package 只导出了 `DeepseekV3ForCausalLM`
- 并没有导出 `DeepseekV3ForCausalLMNextN`

所以最终结果是：

- target `DeepseekV3ForCausalLM` 被 external package 覆盖
- draft `DeepseekV3ForCausalLMNextN` 仍走 upstream SGLang native


## 6. 模型实例化链路

### 6.1 `ModelRunner.load_model()`

核心文件：

- `sglang/python/sglang/srt/model_executor/model_runner.py`

关键位置：

- `load_model()`：约 `901-991`

这里完成：

- 构造 `LoadConfig`
- 选择 model loader
- 调 `loader.load_model(...)`


### 6.2 `ModelRunner._get_attention_backend()`

关键位置：

- `model_runner.py` 约 `1736-1746`

这里会根据：

- 是否是 draft worker
- 是否设置了 `speculative_draft_attention_backend`

来决定 draft 用哪种 attention backend。

这是 speculative 与 attention backend 结合的一个重要入口。


### 6.3 `_initialize_model()`

核心文件：

- `sglang/python/sglang/srt/model_loader/loader.py`

关键位置：

- `_initialize_model()`：约 `257-277`

这是底层真正 `return model_class(**kwargs)` 的地方。

也就是说：

- target 和 draft 在上层是两个不同 worker / model config
- 但底层最终都汇合到同一个模型实例化函数


### 6.4 `get_model_architecture()`

核心文件：

- `sglang/python/sglang/srt/model_loader/utils.py`

关键位置：

- `get_model_architecture()`：约 `89-119`

这个函数负责：

- 看 `hf_config.architectures`
- 查 `ModelRegistry`
- 最终选出要实例化的 model class

如果 external package 覆盖了同名架构，就会优先拿 external package 的类。


## 7. 三层 batch 数据结构

核心文件：

- `sglang/python/sglang/srt/model_executor/forward_batch_info.py`
- `sglang/python/sglang/srt/managers/schedule_batch.py`

### 7.1 三层抽象

`forward_batch_info.py` 文件开头就写得很清楚：

- `ScheduleBatch -> ModelWorkerBatch -> ForwardBatch`

含义：

- `ScheduleBatch`
  - scheduler 侧高层调度状态
  - CPU 语义更强
- `ModelWorkerBatch`
  - 给 worker 的中间态
- `ForwardBatch`
  - 最接近 kernel / backend 执行的低层态


### 7.2 `ForwardMode`

关键位置：

- `forward_batch_info.py` 约 `74-179`

推测相关最重要的几个 mode：

- `TARGET_VERIFY`
- `DRAFT_EXTEND`
- `DRAFT_EXTEND_V2`
- `DECODE`
- `EXTEND`

这里有个容易踩坑的点：

- `TARGET_VERIFY` 在 `is_extend()` 里返回真

所以如果 backend 只按 “decode vs extend” 粗暴分流，很容易把 verify 错当普通 extend。


### 7.3 `ScheduleBatch.get_model_worker_batch()`

核心文件：

- `sglang/python/sglang/srt/managers/schedule_batch.py`

关键位置：

- `get_model_worker_batch()`：约 `2175-2228`

这一步负责把 scheduler 层状态打包成 `ModelWorkerBatch`。

关键理解：

- 对 `decode_or_idle()`，`extend_seq_lens` 会被设成 `None`
- 对其他 extend 类路径，`extend_seq_lens` 来自 `self.extend_lens`

这也是后面 verify 路径里经常出现 `extend_seq_lens=None` 的背景。


## 8. speculative 的核心数据结构：SpecInput

核心文件：

- `sglang/python/sglang/srt/speculative/spec_info.py`

关键抽象：

- `SpecInput`
- `SpecInputType`

类型包括：

- `EAGLE_DRAFT`
- `EAGLE_VERIFY`
- `NGRAM_VERIFY`

也就是说，speculative 不只是“多传几个 tensor”，而是有一套专门的数据结构协议。


### 8.1 DeepSeek / EAGLE 相关具体实现

主要文件：

- `sglang/python/sglang/srt/speculative/eagle_info.py`
- `sglang/python/sglang/srt/speculative/eagle_info_v2.py`

这些文件负责：

- draft 输入构造
- verify 输入构造
- draft token / hidden state / custom mask / positions 等 speculative 元数据


## 9. EAGLE v1 的主执行流程

核心文件：

- `sglang/python/sglang/srt/speculative/eagle_worker.py`

### 9.1 extend / prefill 阶段

关键位置：

- `forward_batch_generation()`：约 `278-309`

流程：

1. target 先跑 extend / prefill
2. target 产出 hidden state
3. draft 用 target hidden state 再做 draft extend


### 9.2 decode 阶段

关键位置：

- `forward_batch_generation()`：约 `310-337`

流程：

1. draft 先 propose
2. target 再 verify
3. draft 根据 verify 结果再 extend，为下一轮准备

这是一个典型的：

- `draft -> target verify -> draft extend`

链式协作过程。


### 9.3 why target and draft share embed/head

关键位置：

- `eagle_worker.py` 约 `157-183`

这里会显式调用：

- `target_worker.model_runner.model.get_embed_and_head()`
- `draft_model_runner.model.set_embed_and_head(...)`

说明：

- upstream 的 EAGLE/NextN draft 设计默认依赖 target 的 embedding 和 lm_head


## 10. EAGLE v2 的主执行流程

核心文件：

- `sglang/python/sglang/srt/speculative/eagle_worker_v2.py`

### 10.1 prefill / extend

关键位置：

- `forward_batch_generation()`：约 `673-697`

流程：

1. target prefill
2. draft prefill
3. 返回 `next_draft_input`


### 10.2 decode / verify

关键位置：

- `forward_batch_generation()`：约 `698-722`
- `verify()`：约 `724-780`

流程：

1. `draft_worker.draft()` 生成 `EagleVerifyInput`
2. `verify()` 内部构造 verify forward batch
3. target 执行 verify 前向
4. draft 再做 `_draft_extend_for_decode()`


### 10.3 spec v2 的一个核心特征

它不再直接围绕 `ScheduleBatch` 做所有 speculative 逻辑，而是更偏向：

- `ModelWorkerBatch`
- `next_draft_input`
- `future_indices`
- overlap plan stream

这也是它和 v1 最大的结构差异。


## 11. attention backend 如何感知 speculative

最典型的文件：

- `sglang/python/sglang/srt/layers/attention/aiter_backend.py`

### 11.1 `init_forward_metadata()`

关键位置：

- `aiter_backend.py` 约 `435+`

这段代码是理解 speculative attention 路径最关键的入口之一。

它不是简单区分：

- decode
- extend

而是细分：

- `decode_or_idle`
- `draft_extend`
- `target_verify`
- 普通 extend


### 11.2 `draft_extend`

关键位置：

- `aiter_backend.py` 约 `526-606`

特点：

- 通过 `spec_info.generate_attn_arg_prefill(...)`
  来生成 draft extend 所需的 attention 参数
- 对 MLA 与非 MLA 路径分别处理


### 11.3 `target_verify`

关键位置：

- `aiter_backend.py` 约 `607+`

特点：

- 不依赖普通 extend 的 `extend_seq_lens`
- 直接根据：
  - `spec_info.draft_token_num`
  - `forward_batch.seq_lens`

构造 verify 所需的：

- `qo_indptr`
- `kv_indptr`
- `kv_indices`

这是后续排查 `ATOM plugin verify` 问题时最值得对照的一段。


## 12. v1 与 v2 的差异总结

### 12.1 v1

特点：

- 更偏 `ScheduleBatch`
- speculative worker 逻辑更集中在一个大类里
- 以串行 orchestrate 为主


### 12.2 v2

特点：

- 依赖 overlap scheduler
- 更偏 `ModelWorkerBatch`
- 引入 `next_draft_input`
- 更明显地区分 draft worker 与 orchestrator worker
- 更容易做 plan stream / overlap


### 12.3 对调试的实际影响

如果你调试 speculative 问题，一定要先分清：

- 当前是 v1 还是 v2
- 当前 `model_worker.forward_batch_generation(...)`
  收到的是 `ScheduleBatch` 还是 `ModelWorkerBatch`

否则很容易误判字段来源和生命周期。


## 13. 一份推荐阅读顺序

如果之后需要重新从头理解 SGLang speculative decoding，建议按下面顺序读：

1. `server_args.py`
   - 看 speculative 参数、NEXTN 规范化、spec v2 开关
2. `spec_info.py`
   - 看算法枚举和 worker 工厂
3. `scheduler.py`
   - 看 `maybe_init_draft_worker()` / `init_model_worker()` / `run_batch()`
4. `tp_worker.py`
   - 看 target 与 draft `ModelConfig` 的分流
5. `model_config.py`
   - 看 `_config_draft_model()`
6. `deepseek_nextn.py`
   - 看 `DeepseekV3ForCausalLMNextN` 到底长什么样
7. `eagle_worker.py`
   - 看传统 speculative v1 主流程
8. `eagle_worker_v2.py`
   - 看 overlap/spec v2 主流程
9. `schedule_batch.py`
   - 看 `ScheduleBatch -> ModelWorkerBatch`
10. `forward_batch_info.py`
   - 看 `ForwardMode` 和 `ForwardBatch`
11. `aiter_backend.py`
   - 看 speculative attention metadata 怎么初始化


## 14. 对 ATOM plugin 调试的启发

这份背景知识对 `ATOM + SGLang plugin` 的调试最直接的启发有三点：

### 启发 1

不能只盯 model class，还要盯：

- `ServerArgs`
- `ModelConfig`
- `Scheduler`
- `TpModelWorker`

因为 draft/target 的分流在这些层已经决定了。


### 启发 2

如果 plugin 只覆盖了：

- `DeepseekV3ForCausalLM`

但没有覆盖：

- `DeepseekV3ForCausalLMNextN`

那么最终一定会形成：

- target 走 plugin
- draft 走 upstream

的混合运行形态。


### 启发 3

如果 attention backend 只按：

- decode
- extend

粗暴分流，而没有补：

- `target_verify`
- `draft_extend`

这类 speculative 专有 metadata 路径，
那么在 speculative 模式下一定迟早会在 verify / draft_extend 里出错。


## 15. 最终总结

SGLang 的 speculative decoding 并不是一个局部 feature，而是一套完整的运行时体系：

- 在配置层决定算法和 draft model 语义
- 在 model config 层改写 draft 架构
- 在 scheduler 层切换到 speculative orchestrator
- 在 worker 层维护 target / draft 协作
- 在 batch 层用三层结构管理状态转换
- 在 attention backend 层按 `ForwardMode` 和 `SpecInput`
  细化 metadata 初始化

如果后续要把 `ATOM MTP` 真正接到 plugin 路径上，最重要的不是先改单个 kernel，
而是先把这张图看清楚：

- 谁负责 target
- 谁负责 draft
- 谁负责调度
- 哪些数据结构在层层转换
- speculative 特有的 `target_verify` / `draft_extend`
  在 attention/backend 层是如何被建模的

只有在这个架构认知稳定之后，后面的接入和调试才会高效。
