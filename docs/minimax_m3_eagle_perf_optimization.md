# MiniMax-M3 EAGLE3 性能优化分析与计划

> 分支:`zejun/m3_eagle_in_graph_0619`
> 依据:`MiniMax-M3-MXFP4_ts_20260619_081706_316.pt.trace.json.gz`(PyTorch profiler trace,MXFP4 + EAGLE3,TP4)
> 日期:2026-06-19

## 1. 问题

MiniMax-M3 本体性能已达标,但合入 EAGLE3 投机解码后,相对非投机基线的吞吐提升不明显——小并发只有 20% 多,达不到普遍 ~50% 的预期。本文基于 profile 定位真正瓶颈,并给出优化计划。

接受率本身是健康的(每次 forward 产出 ~3.06 token,接受率 ~68.7%),所以**理论上 decode 吞吐应接近 3×**,但实测远低于此 → 说明开销吃掉了收益。问题是开销在哪。

## 2. 排除:CUDA graph 不是瓶颈

最初怀疑 draft 跑 eager(无 CUDA graph)导致 kernel-launch / Python 开销主导小并发。profile 否定了这一点:

- 在整个 spec-decode 的墙钟区间(verify + draft 合并 = **857.4 ms**)内,GPU 计算流忙碌 **853.4 ms = 99.5%**。
- 即 **GPU 时间线几乎没有空泡**。CUDA graph 的作用是消除 launch 间隙 / 空泡,在这里最多只能回收 ~0.5%,无意义。

结论:瓶颈是**密集的 GPU 工作**本身,而其中很大一部分是**被浪费的**。

## 3. 根因:draft attention 每步拷贝整块 KV cache

### 3.1 区域耗时(GPU 时间,可信)

> CPU 侧标注因开了 python profiling(trace 里 150 万个 `python_function` 事件)被严重放大,不可信;以 GPU 标注为准。

| 区域 | GPU 时间 | 次数 | 每步 |
|---|---|---|---|
| verify (`decode[...]`) | 462 ms | 24 | ~19.3 ms |
| **draft (`draft[i/3]`)** | 391 ms | 87 (=29×3) | ~4.5 ms/iter → **~16 ms/步** |

一个**单层** draft,每步 GPU 竟≈整个 MoE target verify 的 85% —— 明显异常。

### 3.2 draft GPU 时间的构成

拆解 draft 区域内的 GPU kernel:

| kernel | draft 内耗时 | 说明 |
|---|---|---|
| **`elementwise_kernel_manual_unroll`** | **285.6 ms (73%)** | 不是 GEMM,parent op = `aten::copy_` |
| `Cijk_...MT256x256x64`(GEMM) | 23.3 ms | lm_head 等 |
| `allreduce_twoshot` | 16.5 ms | TP 通信 |
| `Cijk_...MT256x16x128` | 14.8 ms | 小 GEMM |
| `paged_attention_decode` | 10.4 ms | draft 注意力 |
| `reduce`(argmax) | 4.8 ms | |

**73% 的 draft GPU 时间是一个 `copy_`。**

### 3.3 被拷贝的是整块 draft KV cache

`copy_` 的张量形状(按 input dims 聚合):

| 形状 (bf16) | 总耗时 | 次数 |
|---|---|---|
| `[24068, 128, 16, 128]` | 213.5 ms | 72 |
| `[24068, 128, 16, 16, 8]` | 71.2 ms | 72 |

其中 `24068 = draft KV cache 的 num_blocks`,`128 = block_size`,`16 = per-rank kv_heads`,`128 = head_dim`。

→ 这是**整块 draft K cache 和 V cache(合计 ~12 GB)在每个 draft iter 各拷一遍**(72 = 24 步 × 3 iter)。

### 3.4 精确定位(Python 调用栈)

```
eagle.py:343 propose
  → eagle3_llama.py forward → Eagle3LlamaAttention
    → paged_attention → attention_mha.py:148 forward_impl
      → rope_cache → attention_mha.py:366 _gather_prefix_and_concat_kv
        → Tensor.contiguous()        ← 全池拷贝
```

代码:`atom/model_ops/attention_mha.py:421-428`(NHD 分支)

```python
else:  # V is in ASM/NHD format [n, nh, hd, bs], convert to [n, bs, nh, hd]
    k_cache_gather = (
        k_cache.permute(0, 3, 1, 2, 4).contiguous().view(n, block_size, nh, head_dim)
    )                                  # 对整池 K 做 contiguous
    v_cache_gather = v_cache.permute(0, 3, 1, 2).contiguous()   # 对整池 V 做 contiguous
```

形状完全吻合:
- K:`[24068,16,16,128,8] --permute(0,3,1,2,4)--> [24068,128,16,16,8]`
- V:`[24068,16,128,128] --permute(0,3,1,2)--> [24068,128,16,128]`

`permute(...).contiguous()` 把**整个分页 KV 池**(全部 24068 个 block)做格式转换,而真正需要的只是 `block_tables` 引用的少数 block。

### 3.5 为什么只有 draft 中招

1. draft 是 Eagle3 MHA,`num_heads == num_kv_heads`(64==64)→ `attention_mha.py:194` 强制走 **triton 路径**(`use_triton_attn=True`);
2. draft KV 写成 **NHD** 格式(`_cache_format != "SHUFFLE"`,`use_shuffle=False`)→ 进上面那个 `else` 分支;
3. decode 时 `attn_metadata.has_cached=True` → 每步都进 `_gather_prefix_and_concat_kv`。

target 主注意力是 sparse/自定义路径,不经过这里(per-region 统计:verify 区域内该 elementwise ≈ 0)。所以这是 **draft 独有** 的浪费,正好对应 EAGLE 收益问题。

### 3.6 为什么尤其拖累小并发

被拷贝的是**整块 cache(~12GB),大小固定**,与 batch size / 序列长度无关。

- 小并发:真实计算量很小,这个固定 ~12GB 拷贝占比最大 → 收益被吃掉最多。
- 大并发:真实计算变大,固定拷贝相对占比下降。

这正好解释了"**小并发提升只有 20% 多**"的现象。

## 4. 优化计划

### P0 —(主攻,已实现)decode 不做 prefix-gather

#### 更深的根因:gather 的产物在 decode 被丢弃

顺着 dispatch 查清后发现,问题比"拷贝太大"更彻底——**这次 gather 在 decode 里压根没用**:

- draft 是 MHA 且 `num_heads==num_kv_heads` → `attention_mha.py:194` 强制 triton → decode 走 `paged_attention_triton`;其内部无论 `unified_attention` 还是 `run_pa_decode_gluon`,都**直接读分页 `k_cache/v_cache`(按 `block_tables`)**,把 gather 产出的 `k_full/v_full` 整个丢弃。`paged_attention_asm/persistent_asm` 同理。
- `_gather_prefix_and_concat_kv` 的本意是 **prefill 前缀缓存命中**时,为 varlen flash 物化完整 K/V(`has_cached` 本质是 prefill-prefix 概念,`backends.py:351` state=`PREFILL_PREFIX`)。decode 的分页注意力天然能读缓存,从不需要它。
- `rope_cache` 只看 `has_cached` 就 gather,没看"接下来选的 backend 用不用结果"——于是 draft decode 每步白做一次 O(整池) 的 permute+contiguous(~12GB),产物即弃。

#### 实现

`atom/model_ops/attention_mha.py` `rope_cache`,把 gather 的触发从

```python
if attn_metadata.has_cached:
```
改为
```python
if attn_metadata.has_cached and fwd_ctx.context.is_prefill:
```

与 `dispatch_backend` 用 `is_prefill` 选 backend 完全一致:prefill 走 varlen backend(确实消费 `k_full/v_full`)→ 保留 gather;decode 走分页 backend(直接读缓存)→ 跳过。

- **为什么有收益 / 不改数值**:decode backend 本就忽略 `k_full/v_full`,且 gather 不修改 `k_cache/v_cache`(缓存在更早的 `fused_qk_rope_reshape_and_cache` 已写好),所以跳过它**输出逐位不变**,只删掉 285.6ms 纯浪费。
  - draft:~16 ms/步 → ~4.4 ms/步(≈ verify 的 23%,回到"单层 draft 应该很便宜");
  - spec-decode 总 GPU:853 → ~570 ms(**−33%**);
  - 固定拷贝占比在小并发最高 → **小并发收益最大**,正中问题要害。
- **附带收益(fix-then-sweep)**:这是通用修复——任何"decode + has_cached"的 MHA 路径(如带前缀缓存的解码)都不再白做 gather。MLA 路径在 `attention_mla.py`,不受影响。
- **风险**:`attention_mha.py` 共享文件;已确认 `_gather_prefix_and_concat_kv` 仅此一处调用,所有 decode backend 均不消费其产物。需 GPU 验证 GSM8K/accept 不变(预期逐位等价)。

### P1 —(P0 后量化)draft decode 避免"gather 全量 KV + varlen"

- 即便消了 contiguous,draft 每步仍用 `cp_mha_gather_cache` 把整条序列的 KV gather 出来跑 varlen 注意力(因 `num_heads==num_kv_heads` 被强制 triton)。q=1 decode 本可用分页 decode kernel 在 kernel 内直接读 block,免去每步 O(seq_len) 的 gather。
- **为什么有收益**:把每步的 gather 从 O(序列长度) 降到 O(1 token) 的分页读取。
- **做法**:评估给 draft decode 走 paged-decode kernel(或放宽 `use_triton_attn` 条件)。属调研项,等 P0 后看残余热点再决定是否值得。

### P2 —(P0 后)spec-verify(q>1)路径优化

- P0 之后 verify(462 ms)将成为 spec 的最大头:`_index_block_score` 119 ms + `ck_tile` 115 ms + `_gqa_sparse_fwd` 77 ms + `reduce_scatter` 33 ms。
- 已合入的 commit 注释指出:q>1 的稀疏 spec-verify 走的是 ASM 的 **on-device fallback** 路径。
- **为什么可能有收益**:fallback 通常非最优;替换为正经的 batched-verify 稀疏 kernel 有望降低 verify。
- **做法**:核查 q=k+1 verify 的稀疏 indexer/attention 是否落在次优路径,需进一步定位。调研项。

### 非 EAGLE 旁注(本次不动)

- TP 通信类 kernel(allreduce / reduce_scatter / rmsnorm-fusion)合计 693 ms / 总 GPU 3233 ms,主要在 prefill/verify,属 TP4 固有,部分可融合,但与 EAGLE 收益关系不大。

## 5. 验证方法

- **正确性**:这些优化均**不改数值**(只去掉冗余拷贝 / 换等价 kernel),accept 分布与 GSM8K 必须与现状一致——
  - GSM8K MXFP4 TP4:strict-match = 0.9447、flexible = 0.9439(= 现 EAGLE 基线);
  - accept 率 ~68.7% 不变。
- **性能**:用 **output tok/s / TPOT / inter-token latency** 复测(不要用 8k/1k 的 total tok/s,会被未加速的 prefill 稀释),重点看小并发(con4/con16)。
- **profile 回归**:重抓 trace,确认 `elementwise_kernel_manual_unroll` 的 285ms 消失、draft 区域 GPU 时间显著下降。

## 6. 一句话总结

EAGLE 收益不达预期的主因 **不是** CUDA graph、**不是** 200k lm_head、**不是** MoE 固有成本,而是 draft attention 在 `_gather_prefix_and_concat_kv` 里**每个 draft iter 把整块 ~12GB KV cache permute+contiguous 一遍**(占 spec-decode GPU 的 33%,小并发占比更高),而且这次 gather 的产物在 decode 路径里**根本被丢弃**。P0 已实现:`rope_cache` 把 gather 触发加 `and fwd_ctx.context.is_prefill`,decode 直接跳过——逐位等价、零浪费。

## 7. 状态

- **P0:已实现**(`atom/model_ops/attention_mha.py` `rope_cache`,1 行 gate)。lint 通过;mocked 测试套件与基线一致(33 failed / 406 passed / 33 skipped,失败均为预存在的缺依赖项,与本改动无关,零回归)。**待 GPU 验证**:GSM8K MXFP4 TP4 strict=0.9447 / flexible=0.9439 与 accept~68.7% 应不变;profile 复抓确认 285ms 消失、draft GPU 显著下降。
- **P1 / P2:待 P0 GPU 验证后按残余热点推进。**
