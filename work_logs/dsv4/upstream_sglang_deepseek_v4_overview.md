# Upstream SGLang DeepSeek V4 学习概览

本文面向 ATOM SGLang plugin 适配 DeepSeek V4 的工程调研，重点解释 upstream SGLang 中 DeepSeek V4 的 attention、attention kernel path、KV cache 管理，以及它们和 DeepSeek V4 technical report 中混合注意力设计的关系。

文件位置：

- Markdown：`ATOM/work_logs/dsv4/upstream_sglang_deepseek_v4_overview.md`
- HTML 可视化页：`ATOM/work_logs/dsv4/upstream_sglang_deepseek_v4_overview.html`

源码位置：

- Upstream SGLang 根目录：`/shared/amdgpu/home/qichu_qle/zhiwei/dsv4/sglang`
- ATOM 根目录：`/shared/amdgpu/home/qichu_qle/zhiwei/dsv4/ATOM`
- ATOM 当前 V4 plugin 相关代码：`ATOM/atom/plugin/sglang/`

参考资料：

- DeepSeek V4 technical report: [DeepSeek_V4.pdf](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/main/DeepSeek_V4.pdf)
- Hugging Face 模型页给出的 SGLang serving 入口: [DeepSeek-V4-Pro](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/main/DeepSeek_V4.pdf)
- 本地 upstream SGLang 代码：`/shared/amdgpu/home/qichu_qle/zhiwei/dsv4/sglang`

配套 HTML 可视化页面：`ATOM/work_logs/dsv4/upstream_sglang_deepseek_v4_overview.html`

## 1. 一句话总览

Upstream SGLang 的 DeepSeek V4 路径不是把 V3/MLA backend 简单扩展一下，而是围绕 V4 的混合注意力形态重做了一套 attention backend 和 KV cache pool：

- 模型层：`sglang/python/sglang/srt/models/deepseek_v4.py`
- Attention backend：`sglang/python/sglang/srt/layers/attention/deepseek_v4_backend.py`
- ROCm/HIP backend：`sglang/python/sglang/srt/layers/attention/deepseek_v4_backend_hip_radix.py`
- DSV4 kernel helpers：`sglang/python/sglang/srt/layers/attention/dsv4/`
- KV pool：`sglang/python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py`
- Compress state pool：`sglang/python/sglang/srt/mem_cache/deepseek_v4_compress_state.py`
- Pool sizing：`sglang/python/sglang/srt/model_executor/pool_configurator.py`

核心概念是：**每层根据 `compress_ratios[layer_id]` 选择 Dense/SWA-only、C4、C128 三类注意力路径；cache 也随之拆成 SWA pool、C4 pool、C128 pool、C4 indexer pool、compress state pool。**

### AMD/ROCm 实现补充

AMD 路径上需要额外关注 `deepseek_v4_backend_hip_radix.py`、`hip_flash_mla`、`aiter` 和 `SGLANG_HACK_FLASHMLA_BACKEND`。Upstream SGLang 的 AMD 实现仍复用 DSV4 metadata/cache 抽象，但最终 core attention dispatch 会走 HIP FlashMLA entrypoint、Triton fallback 或 AITER fused op。调试时不要只看 `deepseek_v4_backend.py`，要同步看 HIP backend 的分支。

## 2. DeepSeek V4 report 到 SGLang 实现的映射

Technical report 中 DeepSeek V4 的注意力设计强调利用局部窗口与压缩历史来降低长上下文 KV 成本。SGLang 把这个设计落成可 serving 的工程系统时，主要拆成以下几层。

### 2.1 模型结构层

DeepSeek V4 attention 在 SGLang 模型中对应 `MQALayer`：

- `MQALayer.compress_ratio` 来自 HF config 的 `compress_ratios[layer_id]`
- 允许值是 `0`、`4`、`128`
- `0` 表示只使用 SWA 近邻窗口
- `4` 表示 C4 压缩层，带 C4 indexer
- `128` 表示 C128 压缩层，走高压缩比历史

代码入口：

- `MQALayer.__init__`: `sglang/python/sglang/srt/models/deepseek_v4.py`
- `MQALayer.forward`: `sglang/python/sglang/srt/models/deepseek_v4.py`
- `Compressor`: `sglang/python/sglang/srt/layers/attention/dsv4/compressor.py`
- `C4Indexer`: `sglang/python/sglang/srt/layers/attention/dsv4/indexer.py`

#### AMD/ROCm 实现补充

Report 描述的是算法结构；AMD 实现还需要满足硬件友好的 packed layout。重点看 `DeepSeekV4SingleKVPool.create_buffer()` 中的 bytes-per-token 断言：默认 V4 KV layout 是 `qk_nope_head_dim FP8(448) + qk_rope_head_dim BF16(64*2) + scale/pad(8)`。这个 layout 会影响 fused store、HIP FlashMLA 读取和 cache transfer。

### 2.2 Runtime metadata 层

SGLang 不在模型 forward 里现场推导所有 page/index，而是在 attention backend 的 `init_forward_metadata()` 中提前构造 `DSV4Metadata`。

重要对象：

- `DSV4AttnMetadata`: 核心 page/index/length metadata
- `PagedIndexerMetadata`: C4 indexer 需要的 paged metadata
- `FusedCompressMetadata`: C4/C128 compressor 写入计划

关键字段：

- `page_table`: 从 `req_to_token` 按 `page_size` 抽样得到的 full KV page 表
- `raw_out_loc`: 当前 forward 新 token 的 full KV location
- `swa_page_indices`: 每个 query 需要看的 SWA 窗口页索引
- `swa_topk_lengths`: 每个 query 的 SWA 可见长度，上限 128
- `c4_sparse_page_indices`: C4 indexer 选出的稀疏压缩历史
- `c4_sparse_topk_lengths`: 每个 query 的 C4 top-k 长度
- `c128_page_indices`: C128 压缩历史页索引
- `c128_topk_lengths_clamp1`: C128 可见长度
- `swa_out_cache_loc`: 将 full `out_cache_loc` 映射到 SWA pool 后的写入位置

#### AMD/ROCm 实现补充

AMD backend 依赖同一类 metadata，但更敏感于 graph/capture 友好的张量生命周期。打开 graph 或 multi-stream 后，metadata 的准备位置可能从 Python eager 迁移到 graph 内或 stream overlap 中。ROCm correctness 调试初期建议关闭 graph 和 multi-stream，先确认 `swa_page_indices`、C4/C128 extra indices、`swa_out_cache_loc` 在 eager 路径下正确。

### 2.3 KV cache 层

SGLang V4 使用专用 `DeepSeekV4TokenToKVPool`，不是普通 `TokenToKVPool`，也不是 V2/V3 MLA KV pool。

它内部包含：

- `swa_kv_pool`: 保存最近窗口，窗口大小对应 V4 SWA
- `c4_kv_pool`: C4 压缩 KV pool
- `c128_kv_pool`: C128 压缩 KV pool
- `c4_indexer_kv_pool`: C4 indexer 的 key/scale pool
- `compress_state_pools`: attention compressor 的 rolling state
- `indexer_compress_state_pools`: C4 indexer compressor 的 rolling state
- `full_to_swa_index_mapping`: full KV slot 到 SWA slot 的映射

这一点非常重要：**SGLang 的 full `out_cache_loc` 并不直接等于 SWA 写入地址，必须经 `translate_loc_from_full_to_swa()` 转换。**

#### AMD/ROCm 实现补充

AMD 上 KV cache layout 不是普通 BF16 K/V tensor。`DeepSeekV4SingleKVPool` 使用 packed FP8/BF16/scale layout，`kv_cache_total_dim`、page size、alignment 会直接影响 HIP FlashMLA 和 fused store。若出现 token 错误或 kernel assert，优先检查 `page_size=256`、`kv_cache_dtype=fp8_e4m3`、`swa_page_indices` 64 对齐和 `full_to_swa_index_mapping`。

## 3. 请求到 Attention Kernel 的完整路径

### 3.1 启动期

启动 server 后，SGLang 会先配置模型和 cache：

1. `ModelRunner` 读取 HF config，识别 DeepSeek V4。
2. `DSV4PoolConfigurator` 根据可用显存、`swa_full_tokens_ratio`、`compress_ratios`、`page_size` 计算各 pool 大小。
3. `model_runner_kv_cache_mixin.py` 创建 `DeepSeekV4TokenToKVPool`。
4. Attention backend 创建 `DeepseekV4AttnBackend` 或 HIP 路径 `DeepseekV4HipRadixBackend`。
5. 模型层 `MQALayer` 绑定 `compress_ratio`，初始化 compressor/indexer。

`DSV4PoolConfigurator` 的 sizing 思路：

- full token 数是外部看到的总 token budget
- SWA token 数约等于 `full_token * swa_full_tokens_ratio`
- C4 pool 约等于 `full_token / 4`
- C128 pool 约等于 `full_token / 128`
- C4/C128 state pool 按 ring size 和 SWA page size 计算
- HiSparse 打开时，C4 device pool 可进一步按 host/device ratio 缩小

#### AMD/ROCm 实现补充

在 AMD 环境中，pool sizing 不只是显存容量问题，还会决定 HIP kernel 看到的物理 layout。`DSV4PoolConfigurator` 计算出的 SWA/C4/C128/state pool size 会进入 `DeepSeekV4TokenToKVPool`，随后被 `DeepseekV4HipRadixBackend` 读取。建议 launch 初期固定 `--page-size 256` 和保守的 `--swa-full-tokens-ratio`，避免 sizing 与 kernel layout 同时变化。

### 3.2 每个 forward 前

每个 forward batch 进入模型前，attention backend 会做 metadata 初始化：

1. `init_forward_metadata(forward_batch)`
2. 根据 forward mode 分支：
   - decode/idle -> `init_forward_metadata_decode`
   - prefill/extend -> `init_forward_metadata_prefill`
   - target verify -> `init_forward_metadata_target_verify`
   - draft extend -> `init_forward_metadata_draft_extend`
3. 调 `make_core_attn_metadata(...)`
4. 生成 `DSV4AttnMetadata`
5. 如果需要压缩历史，调用：
   - `core_attn_metadata.init_compression_metadata()`
   - `core_attn_metadata.init_flashmla_related()`
6. 生成 C4/C128 compressor metadata：
   - `create_paged_compressor_data(..., compress_ratio=4)`
   - `create_paged_compressor_data(..., compress_ratio=128)`

#### AMD/ROCm 实现补充

ROCm 上如果怀疑 metadata 问题，建议按 decode、短 prefill、长 prefill、target verify/MTP 分开验证。不要一开始就打开 graph capture 或 multi-stream overlap，否则 `init_forward_metadata_out_graph()` 与 `init_forward_metadata_in_graph()` 的职责边界会更难排查。

### 3.3 模型层 attention forward

`MQALayer.forward()` 的主路径：

1. 从 `get_attn_backend()` 取 DSV4 backend。
2. 调 `_forward_prepare(...)` 或 multi-stream prepare：
   - Q projection
   - KV projection
   - Q/K norm
   - RoPE
   - SWA KV cache 写入，通常融合进 K kernel
   - C4/C128 compressor 写入
   - C4 indexer 写入
3. 调 `attn_backend.forward(...)`
4. backend 取出对应 layer 的 SWA cache 和 extra compressed cache。
5. 选择 FlashMLA / HIP FlashMLA / sparse prefill kernel。
6. 输出 attention result 后做 inverse RoPE 和 output projection。

注意这里 `save_kv_cache=False` 是刻意的：`MQALayer` 的 prepare 阶段已经把 cache 写入融合进 kernel 或 compressor 路径，backend forward 只负责读 cache 和算 attention。

#### AMD/ROCm 实现补充

AMD 上 `MQALayer` 的 prepare 阶段可能走 AITER/Triton fused path，例如 fused Q/K norm + RoPE + SWA store。调试 attention 输出错误时，不要只查 `attn_backend.forward()`，还要查 `_forward_prepare*` 是否已经把 SWA/C4/C128 cache 写错。

## 4. Attention Kernel Path

### 4.1 Q/KV prepare path

V4 的 Q/KV prepare 不是单一 kernel，而是由环境变量和平台决定：

- `fused_q_norm_rope`: Q norm + RoPE
- `fused_norm_rope_inplace`: KV norm + RoPE
- `fused_qk_norm_rope_swa_store`: Q/KV norm + RoPE + SWA store 融合路径
- `fused_k_norm_rope_flashmla`: SWA KV 写入的 fused norm/rope/store 路径
- HIP 下可能使用 AITER 或 Triton fused path

模型层关键点：

- `MQALayer.use_fused_qk_norm_rope`
- `SGLANG_OPT_USE_FUSED_QK_NORM_ROPE`
- `SGLANG_OPT_USE_MULTI_STREAM_OVERLAP`
- `SGLANG_USE_AITER`

#### AMD/ROCm 实现补充

AMD 上 Q/KV prepare 的关键开关包括 `SGLANG_USE_AITER`、`SGLANG_OPT_USE_FUSED_QK_NORM_ROPE`、`SGLANG_OPT_USE_MULTI_STREAM_OVERLAP` 和 gfx 支持判断。基础正确性验证阶段建议先关闭 multi-stream，必要时固定 Triton fallback，再逐步打开 AITER/fused path。

### 4.2 Compressor path

Compressor path 对 C4/C128 生效：

- C4: `compress_ratio=4`，overlap compress，且有 C4 indexer
- C128: `compress_ratio=128`，高压缩比 history

核心入口：

- `CompressorBackendMixin.forward_compress`
- `CompressorBackendMixin.forward_core_compressor`
- `CompressorBackendMixin.forward_indexer_compressor`
- HIP: `hip_compress_forward`
- HIP fused norm/rope: `hip_compress_fused_norm_rope_inplace`

写入目标：

- C4/C128 compressed KV 写入 `token_to_kv_pool.set_extra_key_buffer*`
- C4 indexer 写入 `token_to_kv_pool.set_index_k*`

#### AMD/ROCm 实现补充

HIP compressor 路径会进入 `hip_compress_forward`、`hip_compress_fused_norm_rope_inplace` 或相关 fused Triton kernel。若 C4/C128 layer 才出错，而 dense/SWA layer 正常，优先检查 compressor metadata、`c4_out_loc/c128_out_loc` 和 state pool ring size。

### 4.3 Indexer path

C4 层需要 indexer，因为 C4 历史不是全量 dense 读取，而是先用 indexer 选稀疏位置。

核心入口：

- `C4Indexer`
- `C4IndexerBackendMixin`
- `PagedIndexerMetadata`
- `topk_transform_512` / `topk_transform_512_v2`
- `fp8_paged_mqa_logits_torch`
- `_aiter_fp8_paged_mqa_logits`

输入来自：

- C4 indexer KV pool
- `page_table`
- `seq_lens`
- query/indexer projection

输出是：

- `c4_sparse_page_indices`
- `c4_sparse_topk_lengths`

这些字段最终传给 FlashMLA，作为 extra compressed KV 的 sparse indices。

#### AMD/ROCm 实现补充

AMD 上 C4 indexer 可能走 AITER 或 Triton top-k 路径。相关开关包括 `SGLANG_OPT_USE_AITER_INDEXER`、`SGLANG_OPT_USE_TILELANG_INDEXER`、`SGLANG_OPT_USE_TOPK_V2`。如果 C4 sparse indices 异常，先固定一个 indexer backend，避免 top-k kernel 与 attention kernel 同时变化。

### 4.4 Core attention path

`DeepseekV4AttnBackend.forward()` 或 HIP backend 最终会选择：

- NVIDIA/非 HIP：`sgl_kernel.flash_mla.flash_mla_with_kvcache`
- SM120 特化：`flash_mla_with_kvcache_sm120`
- HIP：`hip_flash_mla.flash_mla_with_kvcache_entrypoint`
- large prefill / sparse prefill：`flash_mla_sparse_fwd`

传入 FlashMLA 的核心参数：

- `q`
- `k_cache`: SWA cache
- `indices`: SWA page indices
- `topk_length`: SWA visible lengths
- `extra_k_cache`: C4 或 C128 compressed KV cache
- `extra_indices_in_kvcache`: C4/C128 indices
- `extra_topk_length`: C4/C128 visible lengths
- `attn_sink`
- `tile_scheduler_metadata`
- `is_fp8_kvcache=True`

这解释了为什么 V4 attention backend 与普通 paged attention 不兼容：它不是单一 K/V cache，而是 SWA cache + optional compressed cache + per-ratio sparse index 的组合。

#### AMD/ROCm 实现补充

AMD core attention 重点看 `DeepseekV4HipRadixBackend.forward()`，它最终调用 `flash_mla_with_kvcache_entrypoint(..., backend=...)`。`SGLANG_HACK_FLASHMLA_BACKEND=triton` 是很有用的 debug baseline；等 correctness 过后再切换到 kernel/AITER 路径比较性能和结果。

## 5. KV Cache 管理细节

### 5.1 为什么要拆多个 pool

DeepSeek V4 的层级压缩让不同 layer 的历史访问模式不同：

- SWA 部分只保留最近窗口，所有 layer 都需要。
- C4 层需要较密的压缩历史和 indexer。
- C128 层需要很稀的压缩历史。
- Compressor 需要状态池来跨 token 累积压缩窗口。

如果用普通 KV pool，会浪费显存，也无法表达 C4 indexer 和 compressor state。

#### AMD/ROCm 实现补充

AMD 上 cache 管理还会影响 transfer/offload。`get_contiguous_buf_infos()` 和 `get_state_buf_infos()` 将 C4/C128/indexer/state pool 暴露为连续区域，后续 HiCache/HiSparse 或传输逻辑会依赖这些区域划分。

### 5.2 `DeepSeekV4TokenToKVPool` 的物理布局

`DeepSeekV4TokenToKVPool` 会根据 `compress_ratios` 为当前 PP stage 统计：

- C4 layer 数
- C128 layer 数
- Dense/SWA-only layer 数

然后创建：

- `DeepSeekV4SingleKVPool` 用于 SWA/C4/C128 key storage
- `DeepSeekV4IndexerPool` 用于 C4 indexer key
- `CompressStatePool` 用于 compressor rolling state

`DeepSeekV4SingleKVPool` 的单 token layout 不是常规 BF16 K/V：

- qk_nope FP8
- qk_rope BF16
- scale
- padding

代码中对默认 V4 layout 有断言：`448 + 64 * 2 + 8 = 584 bytes/token`。

#### AMD/ROCm 实现补充

这个 584 bytes/token layout 是 AMD debug 的核心线索之一。HIP FlashMLA 读取的是 packed cache，不是普通 `[num_blocks, block_size, num_heads, head_dim]` BF16 KV。如果 `kv_cache_total_dim`、page size 或 dtype 不一致，通常会表现为 kernel assert 或 silent wrong token。

### 5.3 Full slot 与 SWA slot

SGLang scheduler 分配的是 full KV slot，写在 `forward_batch.out_cache_loc`。V4 SWA pool 是一个更小的 sliding window pool，所以需要：

```text
full out_cache_loc -> full_to_swa_index_mapping -> swa_out_cache_loc
```

这一步在 `DeepseekV4AttnBackend.init_forward_metadata_in_graph()` 和 `get_swa_out_cache_loc()` 中完成。

#### AMD/ROCm 实现补充

AMD fused SWA store 依赖 `swa_out_cache_loc` 已经转换到 SWA address space。若直接用 `out_cache_loc` 写 SWA pool，短 decode 可能偶然不炸，但长上下文或 batch 下会读写错位。

### 5.4 Page size 约束

SGLang V4 路径当前有强约束：

- model runner 初始化 V4 pool 时要求 `page_size == 256`
- backend 中也有 `assert self.page_size == 256`
- SWA window 常量是 128
- C4 page size = `page_size // 4`
- C128 page size = `page_size // 128`

因此任何 ATOM plugin 或自定义 launch 想复用 upstream DSV4 backend，都应该先固定：

```bash
--page-size 256
--swa-full-tokens-ratio <合理值>
--kv-cache-dtype fp8_e4m3
```

#### AMD/ROCm 实现补充

在 AMD 上这些参数是 correctness 约束。`page_size` 会派生 C4/C128 page size，FP8 dtype 会决定 cache packing，indices 64 对齐会影响 HIP FlashMLA。不要把它们当成单纯性能调优项。

### 5.5 Decode 与 Prefill 的差异

Decode：

- 每个 request 通常一个新 token
- `req_pool_indices`、`seq_lens`、`out_cache_loc` shape 对齐 batch size
- 直接生成 `DSV4AttnMetadata`
- `need_compress=True`

Prefill/extend：

- 一个 request 可能有多个 extend token
- 需要把 request 展开成 token-level casual metadata
- `expand_prefill_casually()` 生成每个 query token 的 causal seq len
- 大 prefill 可走 `flash_mla_sparse_fwd`

Target verify / draft extend：

- 为 MTP/speculative decode 服务
- 会处理 `speculative_num_draft_tokens`
- 当前代码对 MTP topk 有限制：`MTP Topk > 1 not supported for DeepSeek V4`

#### AMD/ROCm 实现补充

ROCm 调试建议按 mode 分开验证：先 decode，再短 prefill，再长 prefill，最后 target verify/MTP。HIP graph 和 multi-stream 会让 metadata 的准备位置从 Python eager 迁移到 graph 内或 stream overlap 中，问题定位难度会明显上升。

## 6. 与 ATOM Plugin 适配相关的结论

如果 ATOM SGLang plugin 想使用 upstream SGLang 的 DSV4 cache/backend，有两条路线。

### 路线 A：模型也走 upstream SGLang V4

这是 upstream 默认 serving 路线。优点是 cache、metadata、kernel、model forward 都在同一套契约内。缺点是无法使用 ATOM V4 模型实现和 ATOM kernel。

路径：

```text
SGLang scheduler
  -> DeepSeekV4TokenToKVPool
  -> DeepseekV4HipRadixBackend / DeepseekV4AttnBackend
  -> sglang.srt.models.deepseek_v4.DeepseekV4ForCausalLM
```

### 路线 B：ATOM V4 模型 + SGLang scheduler

这是当前 ATOM plugin 想做的路线。难点是必须桥接两套契约：

```text
SGLang ForwardBatch / DeepSeekV4TokenToKVPool / DSV4Metadata
  -> ATOM AttentionMetaData_DSV4 / state_slot_mapping / compress_plans / unified_kv
  -> atom.models.deepseek_v4.DeepseekV4ForCausalLM
```

这不是简单注册一个 attention backend 能解决的，因为 ATOM V4 attention 读取的是 ATOM 的 `AttentionMetaData_DSV4` 字段，而 upstream SGLang V4 backend 生成的是 `DSV4Metadata`。

#### AMD/ROCm 实现补充

ATOM 在 AMD 上接 V4 时，最容易混合两套“AMD 实现”：upstream SGLang 的 HIP DSV4 backend 与 ATOM 自己的 V4 attention/kernel。前者围绕 `DeepSeekV4TokenToKVPool + DSV4Metadata`，后者围绕 ATOM 的 `AttentionMetaData_DSV4` 和 ATOM kernel。二者都面向 AMD，但 metadata/cache ownership 不同，不能因为都是 ROCm kernel 就直接拼接。

### 对 ATOM enable DeepSeek V4 的启发

最小正确路径应按这个顺序验证：

1. 模型注册：确认 external model package 命中 ATOM V4 wrapper。
2. Config 映射：确认 `compress_ratios`、`window_size`、`index_topk`、`page_size`。
3. Weight load：确认 V4 权重映射与 `prefix="model."` 对齐。
4. Logits path：V4 必须走 `compute_logits()`，不能走 `lm_head`。
5. Attention metadata bridge：构造 ATOM V4 所需 metadata。
6. KV cache ownership：决定复用 SGLang pool 还是 ATOM 自管 unified pool。
7. Kernel alignment：固定 page size、FP8 layout、compress ratio、SWA window。

#### AMD/ROCm 实现补充

AMD enable 顺序建议额外加入：先固定 `SGLANG_HACK_FLASHMLA_BACKEND=triton` 跑通，再切换 HIP kernel/AITER；先关闭 `SGLANG_ROCM_USE_MULTI_STREAM` 和 graph capture，再逐步打开；每次只动一个 kernel/backend 开关。

## 7. 推荐阅读顺序

建议按以下顺序读代码：

1. `configs/deepseek_v4.py`  
   先了解 HF config 字段：`compress_ratios`、`qk_nope_head_dim`、`qk_rope_head_dim`、`index_head_dim`、`window_size`。

2. `model_executor/pool_configurator.py`  
   看 `DSV4PoolConfigurator` 如何把显存预算分到 SWA/C4/C128/state pools。

3. `mem_cache/deepseek_v4_memory_pool.py`  
   看 `DeepSeekV4TokenToKVPool` 如何拆分物理 pool。

4. `layers/attention/deepseek_v4_backend.py`  
   看 `DSV4AttnMetadata` 和 metadata 初始化流程。

5. `models/deepseek_v4.py`  
   看 `MQALayer.forward()` 如何 prepare Q/KV、写 cache、调用 backend。

6. `layers/attention/dsv4/compressor.py` 和 `indexer.py`  
   看 C4/C128 compressor 和 C4 indexer 的具体 kernel path。

7. `layers/attention/deepseek_v4_backend_hip_radix.py`  
   看 AMD/HIP 下 FlashMLA entrypoint 和环境变量分支。

## 8. 常见误区

### 误区 1：DeepSeek V4 可以复用 DeepSeek V3 MLA cache

不可以。V3 MLA 的 cache 语义和 V4 SWA/C4/C128 混合 cache 不同。V4 有 C4 indexer、C128 compressed pool、compress state pool，普通 MLA KV pool 无法表达。

### 误区 2：注册 `dsv4` attention backend 就等于 ATOM V4 能跑

不等于。注册 backend 只决定 SGLang 的 attention backend class。ATOM V4 模型本身还需要 ATOM 的 forward context 和 `AttentionMetaData_DSV4`。

### 误区 3：`out_cache_loc` 可以直接写 SWA pool

不可以。`out_cache_loc` 是 full KV slot，SWA pool 有单独缩小后的 slot 空间，必须通过 `full_to_swa_index_mapping` 转换。

### 误区 4：page size 可以随便调

当前 SGLang V4 路径强依赖 `page_size == 256`，并派生 C4 page size 64、C128 page size 2。改 page size 会影响 metadata、FlashMLA indices、compressor 写入计划。

### AMD/ROCm 常见误区

- 误以为 HIP backend 只是普通 backend 的轻量 wrapper。实际上它会改变 core attention entrypoint 和部分 fused path。
- 误以为 AITER/Triton/kernel backend 可以随意混用。正确做法是固定一个 backend 做 correctness baseline，再切换比较。
- 误以为 token 错误一定来自 FlashMLA。AMD 上 cache write 的 fused norm/rope/store、compressor 和 indexer 都可能更早写错 cache。

## 9. Launch 参数建议

学习和 debug 阶段建议先用保守组合：

```bash
python3 -m sglang.launch_server \
  --model-path <DeepSeek-V4-Pro-or-Flash> \
  --trust-remote-code \
  --tensor-parallel-size 8 \
  --kv-cache-dtype fp8_e4m3 \
  --page-size 256 \
  --swa-full-tokens-ratio 0.1 \
  --chunked-prefill-size 8192 \
  --attention-backend dsv4 \
  --disable-radix-cache \
  --disable-cuda-graph
```

ROCm/HIP debug 时可额外固定：

```bash
export SGLANG_HACK_FLASHMLA_BACKEND=triton
export SGLANG_OPT_USE_FUSED_COMPRESS=true
export SGLANG_OPT_USE_FUSED_COMPRESS_TRITON=true
export SGLANG_ROCM_USE_MULTI_STREAM=false
```

AMD baseline 建议从 `SGLANG_HACK_FLASHMLA_BACKEND=triton` 开始，关闭 multi-stream 和 graph capture，先验证单请求 decode 与短 prefill。等 correctness 稳定后，再单独评估 fused compress、AITER indexer、HIP kernel backend、multi-stream overlap 和 graph capture。

等基础 decode/prefill 正确后，再逐步打开：

1. radix cache
2. cuda graph
3. multi-stream overlap
4. MTP
5. HiSparse/HiCache

## 10. 最小验证矩阵

建议每次只打开一个新能力：

- 32 input / 1 output，单请求
- 256 input / 16 output，跨 SWA window
- 1024 input / 16 output，触发 C4 路径
- 8192 input / 128 output，触发 chunked prefill 和 C128
- concurrency 4
- concurrency 16

每个 case 观察：

- 是否进入 DSV4 backend
- metadata 是否完整
- `swa_page_indices` 和 extra indices 是否 64 对齐
- KV pool 是否稳定
- token 是否能连续生成
- 是否出现 `page_size`、`compress_ratio`、`out_cache_loc` 相关断言

