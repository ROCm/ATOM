# SparseKV Phase 0 — 验收报告

> 日期: 2026-07-31 · 分支: Jasen/cpp-dev · 平台: MI355X (ROCm 7.2.4) · 模型: GLM-5.2-MXFP4
> 拓扑: PP4×TP1 Prefill + TP4 Decode (PD disaggregation, mooncake)

## 1. 结论

SparseKV Phase 0 **功能完整、精度无损、换入(swap)正确**,端到端跑通 GLM-5.2 PD decode。

| 验证项 | 配置 | 结果 |
|--------|------|------|
| Swap kernel 单元测试 | pinned host → GPU hot buffer gather | ✅ `hot[row] == cold[src]` 逐元素一致 |
| 精度回归 (GSM8K) | 50 样本, fewshot=5, 短上下文(全驻留) | ✅ EM **0.94 ± 0.034** vs 基线 **0.9348** |
| 长上下文换入压测 (needle) | 15777 token 上下文, hot buffer **4096 (26%)** | ✅ 精确召回 needle, **0 fault** |

三项叠加证明: SparseKV 在**有/无 swap 两种情况下都数值正确**。

## 2. 测试详情

### 2.1 Swap kernel 单元测试(GPU 直测)

独立验证 HIP gather kernel(`atom/sparsekv/swap_kernel.py`)从 pinned host cold pool
按散列 `src_locs` gather 到 GPU hot buffer 的 `dst_locs`,`hipHostGetDevicePointer`
转换后逐元素比对一致。确认 xnack- 环境下 GPU kernel 可直接跨 PCIe/XGMI 读 pinned host。

### 2.2 精度回归 — GSM8K

```
flexible-extract  exact_match = 0.94 ± 0.0339
strict-match      exact_match = 0.94 ± 0.0339
```

在基线 0.9348 的噪声带内。注意:GSM8K prompt 短(< hot buffer),top-k 全驻留,
**只验证正确性,不产生 swap**;swap 压测见 2.3。

### 2.3 长上下文换入压测 — Needle-in-a-Haystack(关键)

- 上下文 **15,777 token**,hot buffer **4,096**(仅 26% 驻留)。
- 密语放在 ~第 52 行(约 1.2K token 深处),**位于初始 hot set(最近 4096 token)之外**。
- 唯一能召回的路径: indexer 选中该位置 → miss → coordinator 从 CPU cold pool
  经 HIP kernel 换入 → attention 读到正确 KV。
- 结果: 模型精确输出 `aurora-7731-zephyr`,**~500 decode step × 78 层持续 swap,0 fault**。

这是换入正确性的决定性证据:被换入的 KV 是对的。

> 注意: GLM-5.2 是 reasoning 模型,先出思维 token,needle 探针需 `max_tokens >= 512`
> (32 会在思考中途被截断,是探针问题,非 SparseKV 问题)。

## 3. 脚本

| 脚本 | 作用 |
|------|------|
| `scripts/start_glm52_pp4pd_sparsekv.sh` | 启动 PP4 prefill + TP4 decode,decode 端开 SparseKV |
| `scripts/test_sparsekv_needle.py` | 长上下文换入压测(needle 召回) |
| `tests/test_sparsekv_coordinator.py` | coordinator 纯逻辑单测(14 项,CPU,无需 GPU) |

### 3.1 启动(decode 端 SparseKV)

启动脚本相对 `start_glm52_pp4pd.sh` 的差异 **全部在 decode 端**:
- `--level 0`(eager):Phase-0 的 CPU miss-detect 每步同步 top-k D2H,CUDAGraph 内非法。
- 小 `--max-num-seqs` / `--max-model-len`:cold pool 按 `max_num_seqs × max_model_len ×
  num_layers × 576B` 分配,默认值会 OOM。
- `ATOM_SPARSEKV_ENABLE=1`(仅 decode 端)。

三个旋钮可用环境变量覆盖:

```bash
# 精度回归(短上下文,全驻留)
docker exec -it atom_pp4pd_test bash scripts/start_glm52_pp4pd_sparsekv.sh
# 默认: HS_MAX_NUM_SEQS=16 HS_MAX_MODEL_LEN=4096 HS_HOT_BUFFER_SIZE=8192

# 长上下文换入压测(hot << context,强制 swap)
docker exec -it \
  -e HS_MAX_NUM_SEQS=4 -e HS_MAX_MODEL_LEN=24576 -e HS_HOT_BUFFER_SIZE=4096 \
  atom_pp4pd_test bash scripts/start_glm52_pp4pd_sparsekv.sh
```

约束(coordinator 构造时 assert):`HS_HOT_BUFFER_SIZE >= index_topk(2048)`;
`ATOM_MLA_PAGE_SIZE==1` 且非 triton MLA。

### 3.2 精度回归

```bash
# 50 样本快检(fewshot=5)
docker exec -w /it-share/yajizhan/code/ATOM atom_pp4pd_test \
  bash -c 'LIMIT=50 bash scripts/run_gsm8k_eval.sh /mnt/models/GLM-5.2-MXFP4 30000 5'
# 去掉 LIMIT 跑全量 1319
```

### 3.3 长上下文换入压测

```bash
# python scripts/test_sparsekv_needle.py [PORT] [FILLER_LINES] [DEPTH_FRAC]
docker exec -w /it-share/yajizhan/code/ATOM atom_pp4pd_test \
  python scripts/test_sparsekv_needle.py 30000 650 0.08
# 650 行 ≈ 15.8K token;needle 在 8% 深度;PASS = 精确召回密语
```

## 4. 已知边界 / 下一步

- **MTP path**(`max_seqlen_q > 1`)已实现但未验证。
- **性能是 Phase 1**:当前每层 per-token Python miss-detect + 每层 D2H 同步,
  正确但慢(所以 Phase 0 强制 eager)。Phase 1: 异步 DMA + compute overlap、
  把 miss-detect/LRU fuse 进 HIP kernel(对标 SGLang `sparsekv.cuh`)、CUDAGraph 兼容。
- 更大规模精度(全量 1319)与更长上下文(62K,对标设计文档附录 A)可进一步压测。

## 5. 实现落点(参见 `.claude/plans/sparsekv_phase0_plan.md`)

```
atom/sparsekv/{__init__,swap_kernel,coordinator}.py   # 新增: kernel + coordinator
atom/utils/envs.py                                     # ATOM_SPARSEKV_ENABLE / _HOT_BUFFER_SIZE
atom/models/deepseek_v2.py                             # logical top-k 侧信道(custom op body)
atom/model_ops/attention_mla.py                        # decode intercept + 每层当前 token backup
atom/model_ops/attentions/aiter_mla.py                 # coordinator 分配/接线 + 每 token slot metadata
atom/model_engine/{model_runner,sequence}.py           # worker 侧 staging(按 req_id)
```
