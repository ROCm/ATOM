# Phase 0 findings — 统一 superblock 池的两个证伪测试

计划见 `docs/unified_superblock_pool_plan.md`。Phase 0 的规则:两个数没出来之前不写生产代码。

## 0.1 strided state view — **PASS**

问题:统一池给 KDA 层的是 `as_strided` view,slot stride 是**一整个 54 MiB superblock**,
而不是今天 `mamba_v_cache[layer]` 的一个 state。任何一个算子假设了连续,计划就死了。

脚本:`_phase0_strided_state.py`(gather/scatter)、`_phase0_conv_decode.py`(原地写)。
形状取自真实 config,K3 TP8:

```
shape_k = (3, 4608)        bf16
shape_v = (12, 128, 128)   fp32      <- kimi_linear 的 fp32 v 侧
per_layer   =    814,080 B
state       = 56,171,520 B = 53.57 MiB   (与计划里的假设精确匹配)
block       =  1,769,472 B =  1.69 MiB
super       = 56,623,104 B = 54.00 MiB   blocks/super = 32, 浪费 0.80%

view stride: conv_state (96,3,4608)      -> (28311552, 4608, 1)
             ssm_state  (96,12,128,128)  -> (14155776, 16384, 128, 1)
两者 is_contiguous() == False
```

| 算子 | 结果 | 代价 |
|---|---|---|
| `ssm_state[idx] = last_state`(scatter) | bit-exact | 1.01x |
| scatter 隔离性(只碰 idx 对应的 superblock) | bit-exact | — |
| `ssm_state[idx]`(gather) | bit-exact | 1.05x |
| `gather_kda_initial_state`(fused) | bit-exact | — |
| `causal_conv1d_update`(**原地写 conv_state**) | bit-exact | 1.03x |
| `fused_sigmoid_gating_delta_rule_update`(**原地写 ssm_state**) | bit-exact | 1.00x |

未触碰的 slot 逐字节比对未变——两个原地写算子都验了。

### 为什么能过:ATOM 的 wrapper 读 stride,aiter 的硬算

这不是运气。两个实现的地址算术不一样:

- **aiter** `_triton_kernels/gated_delta_rule/decode/fused_sigmoid_gating_recurrent.py:115`
  ```python
  p_h0 = h0_source + idx * HV * K * V + ...
  ```
  slot stride 从**形状**硬算。这个实现**会**在 strided view 上算错地址。

- **ATOM** `atom/model_ops/fla_ops/fused_sigmoid_gating.py:259`
  ```python
  stride_init_state_token = initial_state.stride(0)
  ```
  从**张量**读 stride。K3 走的是这条(`kimi_k3.py:34-35` import 的就是它)。

**这是一个未被记录的依赖。** 计划成立的前提是 K3 一直用 ATOM 这个 wrapper;
哪天有人把它换成 aiter 的版本图省事,统一池会静默读错 state,不报错。
Phase 3 必须加一条断言把这件事钉住。

同理 `causal_conv1d_update` 只要求 `x.stride(1)==1`,`kimi_k3.py:1092-1094` 的注释
已经写明了它读张量自己的 stride。

### 推翻的假设

上一轮认为"要把 pool 指针 + superblock index 传进 kernel、改 kernel 接口"。
**不需要。** `kimi_k3.py:1070-1071` 的

```python
conv_state = cache.k_cache
ssm_state  = cache.v_cache
```

一行都不用改,变的只是 `build_kv_cache_tensor` 递给它什么张量。
`kimi_k3.py` 带 `@support_torch_compile`,CLAUDE.md 明令不得修改——这个结果让我们不必碰它。
工作量比原估计小一个数量级。

## 0.2 4096-token KV 粒度 — 未测

**这是剩下的唯一致命风险。** 之前那个 94–98% drain rate 是**模拟**,不是硬件,
而且第一次报的 3% 还是度量 bug(按 eviction 而非 cohort 算,给出 1/32 = 3.1%)。

要在真实 BlockPool 上用 agentic trace 量:以 32 block 为单位分配/淘汰,
有效利用率多少,命中率掉几个点。

## 0.2 4096-token KV 粒度 — **PASS**

脚本 `_phase0_kv_granularity.py`。用真实 `BlockPool` + 真实 trace
(`semianalysis_cc_traces_weka_062126`,agentic_benchmark.sh 驱动的同一个数据集,
`hash_ids` 就是 trace 自带的前缀身份),两个 arm 唯一的差别是 `superblocks=` 参数。

700 个请求 / 40 个 session / conc=8,每请求 2–1374 个 block(中位 445)。

```
    pool      MiB   plain hit  super hit    delta  partial stranded   oom
   20160   34020M      82.14%     82.03%   -0.11p    55.8%    0.00%     0
   16384   27648M      77.71%     77.37%   -0.35p    55.7%    0.00%     0
   12288   20736M      70.21%     69.97%   -0.24p    55.8%    0.00%     0
    8192   13824M      54.40%     54.07%   -0.33p    54.0%    0.00%     0
    4096    6912M      24.24%     24.27%   +0.03p    48.6%    0.00%   138
```

**最差 -0.35 点。** 20160 blocks 是 K3 的真实池(33.27 GiB / 1.6875 MiB),
那一行命中率 82.03%,落在硬件实测的 79–93% 区间内,OOM=0。

### 两个 waste 指标,只有一个是真的

- `partial` 55.8% — 未满的 superblock 里有多少 block 不是 live。**这个数字没有意义**,
  因为那些 block 并没有被锁住:`_take_free` 照常把它们发出去。
- `stranded` 0.00% — 全池范围内,有多少 block 因为待在已定型的 superblock 里而
  **谁也用不了**。五个池大小全是 0。

**没有内存被 superblock 扣住。** 之前模拟报的 "drain rate 94–98%" 问错了问题。

### 第一版跑出来的结果是错的,记录在此

第一次跑报 "VERDICT: PASS, waste 3.1%",三个信号同时指向它不可信:

1. 池最大只有 4096 blocks = 6.75 GiB,**是 K3 真实 33.27 GiB 的 1/5**
2. 命中率 24%、OOM 138 次——测的是一个饿死的池,不是 K3 的工况
3. waste 3.1% 恰好等于 1/32 = 3.125%,**正是把全满 superblock 算进分母的指纹**——
   跟之前那个报 3% 的度量 bug 是同一个错误,第二次犯

修法:池大小改成真实的 20160,waste 分母只算未满的 superblock,
再加一个 `stranded` 指标回答"到底有没有内存被扣住"。

## 结论

两关都过。Phase 0.1 还把工作量砍掉一个数量级(不需要改 kernel 接口)。
可以进 Phase 1。

**但边界不变**:这套改造解决的不是命中率。82.03 vs 82.14 说明 superblock
在命中率上是**中性**的(-0.11 点),它买到的是结构——一个池、可移动的分界、
checkpoint 从 69 次拷贝变 1 次。命中率的账仍然要靠 `--state-checkpoint-slots 64`
单独验。

## 待办

- [x] Phase 0.1 — PASS
- [x] Phase 0.2 — PASS
- [ ] Phase 3 加断言:KDA state 张量非连续时,禁止走 aiter 那个硬算 stride 的实现
