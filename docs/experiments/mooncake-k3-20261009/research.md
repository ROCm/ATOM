# vLLM Mooncake direct PD + Mooncake Store：ROCm / K3 / DSpark 源码与发布现状

调研时间：2026-10-09 UTC。研究范围为官方源码、文档、PR 和 GitHub Actions；本文不把其他 GPU 的测试结果当成 ROCm 测试，也不把构建通过当作模型正确性通过。远端 C16 实验由主任务单独执行、记录。

固定基线：

| 项目 | 固定版本 | 本地证据 |
|---|---|---|
| vLLM upstream main | `bacbbe187885db62859f5a4a1443f8ec3006e987` | `/app/vllm-k3-mooncake-rocm-20261009`；独立分支由主任务创建 |
| Mooncake upstream main | `b140c1a904d53d24ed211d01ce9a87fc161cb432` | `/app/research/Mooncake-20261009` |
| 目标工作负载 | Kimi K3 hybrid MLA/KDA，P8/D8，各 TP8/DCP8，DSpark 3，C16，FP8 KV，block 128，1M context，FULL_AND_PIECEWISE | 主任务沿用历史实验口径；不等于下面官方 smoke 的配置 |

## 结论

这条路线已经有真实 ROCm 实现、ROCm wheel 和专用 GPU CI 配置，不能再以“Mooncake 只支持 NVIDIA”概括。当前 vLLM main 的两个 connector 都支持 HMA；Mooncake Store 已具备 hybrid + DCP 的分组缓存几何处理，`mamba_cache_mode=align`、PCP=1 的对称 TP8/DCP8 不存在 Store 初始化层面的硬拒绝。Direct connector 通过 TP rank 配对传整页，具备 Mamba group、N−1 状态和多个 attention backend 的处理，因此先跑固定 main 加配置，是有依据的实验起点。[V1][V2][V3][V4]

但“ROCm 上可安装/可传输”与“K3 + DSpark + C16 agentic 稳定且能有效命中 CPU prefix cache”之间还有明显证据缺口：

1. Mooncake 官方 ROCm vLLM GPU lane 实际只跑 Qwen3-8B、TP2、32k、关闭 prefix caching、`num_workers=1` 的串行 direct PD smoke。没有 K3、DSpark、Store 或 C16 验证。[M1][M2]
2. vLLM main 的 Store partial-tail 写入仍缺少 EAGLE/DSpark 查找所需的 attention proof。已有未合并 PR 明确复现 K3 TP8/DCP8 + DSpark 在 GPU prefix cache reset 后外部缓存完全未命中或只部分命中。必须测 CPU tier 的 cold → GPU reset → exact resend，不能只看普通热缓存命中率或 TPS。[V5][P1][P2]
3. direct PD 的一般 DCP/不同 CP 拓扑支持尚不能由当前实现推断。对称 TP8/DCP8 的 rank 配对恰好一致；共享 core 的 finish-time block clipping 仍用物理 `spec.block_size`，存在带 speculative lookahead 时多传未计算尾块的上游已知风险。[V6][V7][P3]
4. PyPI 最新 ROCm wheel `0.3.13.post1` 不满足当前官方 multi-protocol verifier 的二进制判据。本次应使用已下载、已固定 SHA256 的官方 main CI wheel，不能只按包名或版本号安装。[M3][M4]

## 可直接用于本次 CI 的 Mooncake 制品

已从 **Mooncake 官方仓库的 main push CI** 下载合适制品，避免远端重复源码构建：

| 字段 | 值 |
|---|---|
| 源码 SHA | `b140c1a904d53d24ed211d01ce9a87fc161cb432` |
| 成功 workflow | <https://github.com/kvcache-ai/Mooncake/actions/runs/37877673963>，事件 `push`，仓库 `kvcache-ai/Mooncake`，branch `main` |
| ROCm cp312 成功 job | <https://github.com/kvcache-ai/Mooncake/actions/runs/37877673963/job/113650773454> |
| Artifact | `11593557642`，`mooncake-wheel-rocm-ubuntu-py312` |
| Artifact 下载 API | <https://api.github.com/repos/kvcache-ai/Mooncake/actions/artifacts/11593557642/zip>（GitHub Actions artifact，可用 `gh api` 下载） |
| Wheel 本地路径 | `/app/research/mooncake_transfer_engine_rocm-0.3.13-cp312-cp312-manylinux_2_35_x86_64.whl` |
| Wheel SHA256 | `7dfe1f9acec16843868dde28dcb1d3903aafeaf8b92a657239055581a8c44df6` |
| Wheel 自报版本 | `0.3.13`；**用源码 SHA + wheel hash 标识实验，不依赖这个版本号排序** |
| 构建环境 | 官方 workflow 使用 `rocm/dev-ubuntu-22.04:7.2.3-complete`；cp312；`USE_HIP=ON`、`USE_CUDA=OFF`、`ENABLE_MULTI_PROTOCOL=ON`、`WITH_EP=OFF`。[M4] |

成功 job 原始日志保存为 `/app/research/mooncake-b140c1a9-rocm-build.log`。日志 5548–5552 行显示 `Mooncake ROCm distribution: 0.3.13`、`Mooncake multi-protocol support: enabled`，且 wheel RECORD 校验覆盖 `mooncake/engine.so`、`mooncake/store.so`、`mooncake/mooncake_master`。本地又校验了整个 wheel 的 SHA256，并确认 engine binary 包含 `SUPPORT_HIP`、`SUPPORT_MULTI_PROTOCOL`、`MC_DISABLE_HIP`。这能证明官方构建和 package smoke 通过；目标机器上的 import、GPU DMA-BUF 注册和双机传输仍需实际验证。

建议将本地固定 wheel 随远端实验制品上传，再在实验 Python 环境安装。该 wheel 与 CUDA variants 共用 `mooncake` import namespace，环境应只保留正确 ROCm variant。设置 `LD_LIBRARY_PATH` 包含 `/opt/rocm/lib` 后，用固定 Mooncake 源码的 `scripts/e2e/python/verify_rocm_wheel.py` 检查。该脚本校验 wheel RECORD、active import 路径、Store/master 是否包含，以及 multi-protocol 能力。[M3][M4]

### 为什么不选 PyPI latest / 最新 nightly

PyPI 在调研时的 latest 是 `0.3.13.post1`（2026-08-31）。cp312 wheel URL：

<https://files.pythonhosted.org/packages/4c/53/e269f99feaa90d4c9a1364a5288c95f5eddf45424e733d3168a9ccdbbe12/mooncake_transfer_engine_rocm-0.3.13.post1-cp312-cp312-manylinux_2_35_x86_64.whl>

SHA256 为 `2c426b67417dfe925e717f04958b71ec755557b18201a1395548ec87e170bbb5`。本地下载核验后，`engine.so` 虽包含 HIP、`libamdhip64.so` 和 direct transfer API，却既不含 `SUPPORT_MULTI_PROTOCOL`，也不含当前 verifier 对老版本接受的 `MC_DISABLE_HIP` marker。因此它会被当前官方 verifier 拒绝；这不是在目标 GPU 上实测到损坏，不能据此宣称已复现通信故障。[M3][M5]

2026-10-08 nightly `37807302484` 的 ROCm cp312 job 在 `real_client.cpp` 编译时因 `mooncake::Environ` incomplete type 失败，ROCm integration 被跳过。日志在 `/app/research/mooncake-latest-rocm-build.log`。随后上面 `b140c1a9` 的 main push build 已成功，所以不必为这个旧 nightly 失败回退整个路线。[M6]

## 平台、拓扑和配置边界

| 项目 | 固定 main 的证据与边界 | 本次做法 |
|---|---|---|
| ROCm RDMA | Mooncake 有 HIP transport、HIP DMA-BUF 注册和自动跨主机选择 RDMA 的 multi-protocol 路径。[M7][M8] | 使用 ROCm multi-protocol wheel；双机实测 RDMA。 |
| HMA | direct / Store 都继承 `SupportsHMA`；MultiConnector 要求所有 children 都支持。[V1][V2][V3] | 保持 hybrid manager；不靠关闭 HMA 绕过问题。 |
| KDA/Mamba Store | 校验要求 `mamba_cache_mode='align'`；多组 hybrid 拒绝 PCP>1，没有 DCP>1 拒绝。[V2] | `--mamba-cache-mode align`，PCP1。 |
| DCP Store | 每个 group 用 `resolve_dcp_kv_cache_spec`；key 有 TP/PCP/DCP/PP/group identity；DCP 和 Mamba replication factor 为 1。[V4] | 相同 TP8/DCP8；不启用 hetero-TP Store 共享选项。 |
| Heterogeneous Store TP | 仅特定单 full-attention group、PCP/DCP disabled、cross-layer blocks disabled 的布局支持 `store_tp_size`；不适用于此 K3 配套。[V8] | 不设置 `store_tp_size` / LCM 共享。 |
| Direct Mamba | P 截去最后 prompt token 计算 h(N−1)，D 复算最后 token；源码说明测试覆盖 GDN，未验证 Mamba2。KDA 的具体模型组合仍需真实输出验证。[V1] | 保留标准 direct proxy 的 transfer params；检查流式 EOF、首 token、greedy 对照。 |
| Direct DCP | `TransferTopology` 有 DCP 能力，但 Mooncake constructor 没传 dcp_size，握手也没传 peer DCP；对称同 TP 可维持同 rank 配对。[V6] | 对称 TP8/DCP8 作为研究范围，不推广到不对称 CP。 |
| DSpark backend/block layout | direct `_sync_block_size_with_kernel` 取 target/draft backend 共同物理 block size；Mamba block ID 不做 attention kernel block 展开。[V1] | 记录实际 resolved block size 与每组布局，不能只记 CLI block128。 |
| CUDA graph | direct/Store 没有覆盖 base 的 `requires_piecewise_for_cudagraph=False`；Multi 会逐 child 检查。[V3][V9] | 没有 connector 声明的 graph 启动拒绝；FULL_AND_PIECEWISE 正确性仍靠远端。 |
| Direct + Store 同进程 | Store 明确初始化自己的 TransferEngine，direct 另有独立 TransferEngine。[V10] | 两套 RDMA registrations、线程、端口和 CPU segment 都计入资源观察。 |
| Pool 容量 | embedded 的 `global_segment_size` 是**每 GPU rank**贡献量，`local_buffer_size` 也是每 GPU。[V8] | TP8 的总 CPU pool = 8 × per-rank size；不要把总量直接填 per-rank 参数。 |
| 键隔离 | main key 有 `cache_prefix`、模型 basename、ranks/groups；完整 dtype/quantization fingerprint 仍是 #56971 提案的一部分。[V11][P4] | 每个实验用唯一 `cache_prefix`，同一复用实验保持不变。 |

### 官方 ROCm CI 的环境可借鉴项

实际 ROCm vLLM smoke 使用 `ROCR_VISIBLE_DEVICES`、`HIP_VISIBLE_DEVICES`、`MC_MAX_CONCURRENT_REG_MR=1`、`MC_TE_FILTERS=<选定 RoCE NIC>`；connector `num_workers=1`。容器传入 `/dev/kfd` 与选择的 render/RDMA devices，memlock unlimited，设置 `MC_GID_INDEX`、`MC_FORCE_HCA=1`，并根据 runner NIC 设置 `NCCL_IB_HCA` 和 `NCCL_SOCKET_IFNAME`。[M1][M9]

这些是该 MI350X/Pensando runner 的配置证据，NIC 名、GID、device provider 不能原样复制到 g10/g12。`num_workers=1` 限制的是 sender worker pool，客户端依然可以 C16；因此可以作为保守的 first baseline，随后若做 `num_workers=10` 对照，应分开报告。官方注释明确因为 #44238 不覆盖并发 sender；该 issue 的原始故障来自 Qwen3-Omni，评论中的 H20/不同模型未复现，不能把它当 ROCm 已确认缺陷。[M1][P5]

### 与历史 P-only CPU offload 对齐的配置

Prefill 用：

```json
{
  "kv_connector": "MultiConnector",
  "kv_role": "kv_producer",
  "kv_connector_extra_config": {
    "connectors": [
      {
        "kv_connector": "MooncakeConnector",
        "kv_role": "kv_producer",
        "kv_connector_extra_config": {"mooncake_protocol": "rdma", "num_workers": 1}
      },
      {
        "kv_connector": "MooncakeStoreConnector",
        "kv_role": "kv_both",
        "kv_connector_extra_config": {"cache_prefix": "k3-c16-bacbbe187-mooncake-b140c1a9"}
      }
    ]
  }
}
```

Decode 用 `MooncakeConnector` / `kv_consumer`。P 在没有 remote prefill 参数时 direct 返回零匹配，Store 能接管 prefix lookup；Multi 选择第一个有可用 token 的 connector，save 向全部 children 转发。[V3] 这是保持历史 P-only offload 的公平起点。若后续把 Store 加到 D 并开 `save_decode_cache`，那是额外实验：main 仍丢弃超过 prefill_end 的 Mamba boundary，不能把 full-attention decode blocks 已存储误称完整 hybrid conversation prefix 已复用。[P4][V12]

## 尚未合并的相关修复：与本次 baseline 的关系

| PR（2026-10-09 查询均 open） | 具体作用 | 本次处理判断 |
|---|---|---|
| [#52271](https://github.com/vllm-project/vllm/pull/52271) `e9575296f26b01beb8798a4dd99a2f4a74769528` | 去掉 hybrid+DCP 拒绝，Store attention groups 按 DCP 缩放。 | main 已有这两个关键效果及通用 `resolve_dcp_kv_cache_spec`，不是启动前置。不要因 PR 仍 open 就重复集成。 |
| [#53730](https://github.com/vllm-project/vllm/pull/53730) `c286a3321c0224b77e40ea619f84181792bfc98d` | K3 hybrid DCP Store external-hit 几何、失败恢复、PUT waves。作者报告 no-draft DCP8 GSM8K 1270/1319，external get 成功。 | PR 自己声明不验证 drafted stack；描述没有可确认的 ROCm平台证据，不当作本次 ROCm/DSpark validation。 |
| [#45340](https://github.com/vllm-project/vllm/pull/45340) `a3c8805d858ba3a16b598828819c621c7aee93a0` | 对齐 CP 的 scheduler block math，尤其 SWA clip。 | 当前 direct raw block size 主要用于 SW clipping，K3 MLA/KDA 无 SWA，不据此认定本场景必失败。 |
| [#49965](https://github.com/vllm-project/vllm/pull/49965) `85cc3579009096ffe3dbe8e3b91e90fc1d441e6e` | computed block clipping 用 manager 有效 span，排除 DCP speculative lookahead 未计算尾块。 | main 风险仍在；若 direct 边界失败，应优先对照这点，不能自动批量合入不相关 patch。 |
| [#56615](https://github.com/vllm-project/vllm/pull/56615) `6dbf99e98cbe96744cd6fbb38a80e7704f0c77aa` | Store Mamba@B + attention@B+H pairing，修复 DSpark/EAGLE drop 后外部 miss。 | 与 next PR 同一关键问题的不同实现；不能叠加 cherry-pick。作者 K3 TP8/DCP8/DSpark 外部 reset 复用结果未标为 ROCm。 |
| [#60695](https://github.com/vllm-project/vllm/pull/60695) `3ac5312ac1d82f6cf9710b9bdf02788e0011291e` | 从 #59826 拆出的 Mooncake attention proof + async pinning 修复；core #60533 已合并。 | 最新且直接命中本场景。main baseline 后若测到 Store miss，可做独立有记录的集成 A/B；PR自身未跑新的 K3 GPU e2e，引用的 C128 性能同时包括 core 改动。 |
| [#56971](https://github.com/vllm-project/vllm/pull/56971) `da225215151a5698385b09dbd471bffc736e3343` | Store decode-phase Mamba states + key fingerprint。 | P-only offload 首轮不需要为了 decode保存集成；若追求 agentic D生成内容直接回写Store则相关。其端到端证据为 H20、TCP，不是 ROCm/RDMA。 |

## C16 实验要回答什么

建议按现有 C16 harness 执行，不在研究阶段预判成功。最少应取得以下互相可区分的证据：

1. **运行路径成立**：固定 vLLM/Mooncake SHA 和 wheel hash；HIP import、Store/master启动、双机 RDMA registration 成功；P direct transfer 字节/descriptor/latency 有成功计数，D输出完整。direct 成功 transfer 统计在 P，D主要记失败，不要只抓 D 指标。[V1]
2. **模型正确性**：有完整输出/无坏 EOF，并有与 standalone 或既有正确 baseline 的固定提示 greedy/accuracy 对照；C16 请求成功不自动证明 KDA 状态正确。
3. **CPU offload 真正生效**：同一个精确 token prefix 做 cold、hot、只重置 GPU cache后复用三步；记录 P 本地 prefix hits、external prefix hits、Store put/get byte/key/failure counts。用 7449 等非齐块长度与 128边界相邻长度，能覆盖上游 DSpark partial-tail 已知失配；精确 token length 应由 tokenizer/token-ID输入控制。[P1][P2]
4. **Agentic延续**：与历史同一 C16 trace、同一 turn/context/output 分布；记录 TTFT、ITL、end-to-end延迟、吞吐、acceptance、重算token、external命中。若只有 HBM热命中，不能说明替换LMCache的CPU层有效。
5. **解释边界**：先报告纯 main+固定wheel结果；如果观察到与上游open PR吻合的失败，后续 patched run明确列出patch与SHA。对称TP8/DCP8结果不推广到heterogeneous TP/CP，也不将NVIDIA来源数据作为ROCm收益。

当前 vLLM 自带 K3 prefix-cache test 在 main 被 SM100 family 条件 skip，部署矩阵是 NIXL/SimpleCPUOffload而非Mooncake；DSpark+DCP测试还标注 FlashInfer ragged-batch xfail。这说明现有 K3 test 文件本身不能替本次 ROCm Mooncake C16提供覆盖；FlashInfer issue也不能直接外推为ROCm失败。[V13]

## 一手来源

以下源代码链接均固定到本次 SHA，避免 `main` 漂移；PR状态与API信息为调研时快照。

- [V1] [Mooncake direct connector：SupportsHMA、Mamba N−1、spec backend、页映射与指标](https://github.com/vllm-project/vllm/blob/bacbbe187885db62859f5a4a1443f8ec3006e987/vllm/distributed/kv_transfer/kv_connector/v1/mooncake/mooncake_connector.py#L703)。关键行：703、845–852、897–899、946–990、1347–1366、1713–1738、1775–1817、1942–1976。
- [V2] [Store connector validation](https://github.com/vllm-project/vllm/blob/bacbbe187885db62859f5a4a1443f8ec3006e987/vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/connector.py#L127)，127–156。
- [V3] [MultiConnector HMA、graph、first-hit load和多child update](https://github.com/vllm-project/vllm/blob/bacbbe187885db62859f5a4a1443f8ec3006e987/vllm/distributed/kv_transfer/kv_connector/v1/multi_connector.py#L135)，135–218、434–481。
- [V4] [Store per-group DCP resolution、key ranks和replication](https://github.com/vllm-project/vllm/blob/bacbbe187885db62859f5a4a1443f8ec3006e987/vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/worker.py#L1606)，1606–1644、1791–1843。
- [V5] [main partial-tail只存到boundary的实现](https://github.com/vllm-project/vllm/blob/bacbbe187885db62859f5a4a1443f8ec3006e987/vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/worker.py#L613)，613–684；[EAGLE lookup](https://github.com/vllm-project/vllm/blob/bacbbe187885db62859f5a4a1443f8ec3006e987/vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/coordinator.py#L389)，389–410。
- [V6] [TransferTopology默认dcp_size=1](https://github.com/vllm-project/vllm/blob/bacbbe187885db62859f5a4a1443f8ec3006e987/vllm/distributed/kv_transfer/kv_connector/utils.py#L429)，429–440、544–588；[Mooncake构造未传dcp_size](https://github.com/vllm-project/vllm/blob/bacbbe187885db62859f5a4a1443f8ec3006e987/vllm/distributed/kv_transfer/kv_connector/v1/mooncake/mooncake_connector.py#L1331)，1331–1340、1495–1515。
- [V7] [KVCacheManager computed block clipping](https://github.com/vllm-project/vllm/blob/bacbbe187885db62859f5a4a1443f8ec3006e987/vllm/v1/core/kv_cache_manager.py#L761)，761–779。
- [V8] [Store官方使用指南：容量、Multi、decode保存、hetero TP边界](https://github.com/vllm-project/vllm/blob/bacbbe187885db62859f5a4a1443f8ec3006e987/docs/features/mooncake_store_connector_usage.md#L35)，35–63、86–152、275–296。
- [V9] [base graph requirement默认为False](https://github.com/vllm-project/vllm/blob/bacbbe187885db62859f5a4a1443f8ec3006e987/vllm/distributed/kv_transfer/kv_connector/v1/base.py#L695)。
- [V10] [Store独立TransferEngine](https://github.com/vllm-project/vllm/blob/bacbbe187885db62859f5a4a1443f8ec3006e987/vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/worker.py#L1508)；[direct独立TransferEngine](https://github.com/vllm-project/vllm/blob/bacbbe187885db62859f5a4a1443f8ec3006e987/vllm/distributed/kv_transfer/kv_connector/v1/mooncake/mooncake_connector.py#L1189)。
- [V11] [Store KeyMetadata / PoolKey](https://github.com/vllm-project/vllm/blob/bacbbe187885db62859f5a4a1443f8ec3006e987/vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/data.py#L101)，101–167。
- [V12] [Store scheduler prefill_end过滤Mamba boundary](https://github.com/vllm-project/vllm/blob/bacbbe187885db62859f5a4a1443f8ec3006e987/vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/scheduler.py#L556)，556–609。
- [V13] [main K3 prefix-cache tests的平台和部署矩阵](https://github.com/vllm-project/vllm/blob/bacbbe187885db62859f5a4a1443f8ec3006e987/tests/models/kimi_k3/test_prefix_cache.py#L48)，48–75、140–163。
- [M1] [ROCm vLLM smoke实际模型、num_workers、prefix关闭、TP2、32k](https://github.com/kvcache-ai/Mooncake/blob/b140c1a904d53d24ed211d01ce9a87fc161cb432/scripts/e2e/scripts/test_vllm_1p1d_erdma.sh#L5)，5–9、39–66。
- [M2] [ROCm MI350X双机集成workflow](https://github.com/kvcache-ai/Mooncake/blob/b140c1a904d53d24ed211d01ce9a87fc161cb432/.github/workflows/integration-test-rocm.yml#L17)，17–26、64–68；[ROCm suite列表](https://github.com/kvcache-ai/Mooncake/blob/b140c1a904d53d24ed211d01ce9a87fc161cb432/scripts/rocm_tests/scripts/run_test.sh#L59)。
- [M3] [ROCm wheel provenance/multi-protocol verifier](https://github.com/kvcache-ai/Mooncake/blob/b140c1a904d53d24ed211d01ce9a87fc161cb432/scripts/e2e/python/verify_rocm_wheel.py#L34)，34–56、59–108。
- [M4] [ROCm build与smoke workflow](https://github.com/kvcache-ai/Mooncake/blob/b140c1a904d53d24ed211d01ce9a87fc161cb432/.github/workflows/ci_rocm.yml#L28)，28–40、131–143、184–207；[固定成功job](https://github.com/kvcache-ai/Mooncake/actions/runs/37877673963/job/113650773454)。
- [M5] [PyPI ROCm package JSON](https://pypi.org/pypi/mooncake-transfer-engine-rocm/json)，快照 `/app/research/mooncake-rocm-pypi-20261009.json`；本文记录了下载URL、时间、whole-wheel hash和实际binary检查。
- [M6] [10月8日失败ROCm nightly job](https://github.com/kvcache-ai/Mooncake/actions/runs/37807302484/job/113415006019)，错误日志本地4357–4365、4434–4444行；不是本次选用的成功wheel。
- [M7] [HIP DMA-BUF registration](https://github.com/kvcache-ai/Mooncake/blob/b140c1a904d53d24ed211d01ce9a87fc161cb432/mooncake-transfer-engine/src/transport/rdma_transport/rdma_context.cpp#L652)，652–740。
- [M8] [multi-protocol跨主机排除HIP IPC并走RDMA](https://github.com/kvcache-ai/Mooncake/blob/b140c1a904d53d24ed211d01ce9a87fc161cb432/mooncake-transfer-engine/src/multi_transport.cpp#L756)，756–811。
- [M9] [ROCm CI container/RDMA环境](https://github.com/kvcache-ai/Mooncake/blob/b140c1a904d53d24ed211d01ce9a87fc161cb432/scripts/rocm_tests/scripts/common.sh#L52)，52–98。
- [P1] [#56615：K3 TP8/DCP8 DSpark外部cache reset复现和B+H方案](https://github.com/vllm-project/vllm/pull/56615)。
- [P2] [#60695：EAGLE attention proof、async pinning、测试范围与未测项](https://github.com/vllm-project/vllm/pull/60695)。
- [P3] [#49965：DCP computed block list clipping复现](https://github.com/vllm-project/vllm/pull/49965)。
- [P4] [#56971：decode Mamba states、key fingerprint和H20/TCP验证范围](https://github.com/vllm-project/vllm/pull/56971)。
- [P5] [#44238：并发direct transfer问题与H20未复现评论](https://github.com/vllm-project/vllm/issues/44238)。
