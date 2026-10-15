# PR #2586：V4 mono 使能、CI、整网精度与性能验证方案

## 1. 结论与范围

本文定义 V4 mono 的使能范围、验证门禁与发布步骤，不是单轮实验结果报告。
每次执行应在仓外 artifacts 中冻结实际代码、依赖、配置和运行状态；不要把
临时路径、逐轮结果和过期 CI 快照提交到本方案中。

**现有 V4 mono 执行器与 native engine 的安装/graph 接入已存在；第一阶段不需要
再写一套 kernel。缺的是合格配置、可复现依赖、真正执行 mono 的验证门禁，以及
同配置整网精度和性能证据。** 不能以绿色 CI、安装成功或少量 decode 尾部命中
代替这些证据。

第一批使能目标限定为单机 gfx950、256 CU/设备、纯 TP8、FP8 KV、FP4 index：

| 目标 | index 配置 | auto 的实际请求宽度 | 当前资格 |
| --- | --- | --- | --- |
| 普通 greedy decode | `use_index_cache=true, index_topk_freq=4` | 1 / 4 / 8，q=1 | 代码支持，需整网资格验证 |
| MTP3 target verify | `use_index_cache=false, index_topk_freq=1` | 1，q=4 | 代码支持，需真实 MTP3 整网验证 |
| 普通 C2/C16 | 同普通档 | auto 不接受；force 可探针 | 不自动扩围，不计生产收益 |
| MTP3 C2 | 同 MTP 档 | auto 不接受；force 可探针 | S8 vs 双 S4 结构探针，需独立正向证据 |

`C` 是客户端并发，准入判断的是 engine 的 `running_bs`/verify q 与真实行契约，
两者不能简单等同。C4/C8 排队、prefill 插入、EOS、补入请求会改变实际 decode
形状，必须查看命中/回退统计。普通 auto 目前还要求 greedy 且不返回 logprobs；
采样/logprob 请求应回退，并有兼容性测试。

不在本 PR V4 mono 范围：DSpark、prefix caching、EP/DPA/DP/PP/PCP/DCP、TBO、
RapidServe/分离式 decode、ATOM 的 vLLM/SGLang plugin、非合格硬件和 cache 格式。
保留这些路径的 native 兼容性测试，不为“让 dashboard 都变成 mono”放宽准入。
V4.1/其他模型已有的 mono 和 V4 DSpark 适配是不同事项。

S1 是一层一个 persistent launch；S2/S4/S8 当前是一层 attention + MoE 两阶段，
仍是 fused mono 路径，但不能宣传为所有宽度“一层严格只有一次 launch”。

## 2. CI 检查方法：已放行不等于已覆盖 mono

检查 PR 非 draft、具有 `ci:atom` 标签，并核对当前 HEAD 的 native heavy gate、
offline smoke、catalog、Pre Checkin、AITER wheel 和 GPU accuracy jobs。
Black、Ruff、schema 和 non-GPU tests 不能替代最终 GPU accuracy 结果。
后续 push 可能取消旧 run；必须区分 `cancelled`、排队和真正的失败。
`ci:atom` 只使 native heavy CI 放行；
本 PR 的 V4 mono 安装点也只在 native engine，不靠增加 plugin CI 标签启用它。

独立 mono accuracy 条目必须设置 `--no-enable_prefix_caching` 和合格并发。
默认高并发 native 条目保留兼容性回归职责；只补 no-prefix 不能把主体 C65
decode 算作 mono 精度覆盖。普通档还必须固定 true/4 index override。

CPU 环境未安装 FlyDSL 时，build-key 的两个 compile case 用 `importorskip` 跳过，
这是 CPU 分层允许的行为，不是 gfx950 编译证据。旧测试容器缺 AITER V4 导入符号
时跳过的共享 V4/DSpark 测试，同样不是通过；GPU V4 专属门禁必须对此失败退出。

## 3. ATOM 自己的 dashboard：落地点与缺口

### 3.1 原生 ATOM Benchmark / gh-pages performance dashboard

入口：`.github/workflows/atom-benchmark.yaml` → `.github/benchmark/models.json`
→ `benchmark-tmpl.yml` → `plugin_benchmark_to_dashboard.py`
→ `benchmark-dashboard` gh-pages。
现有转换已保留 **Total Tput、TTFT、TPOT**，也有 GPU/image 信息，不需重建 dashboard。
现有定时场景是 nightly 8192/1024、weekly 1024/1024，range ratio 0.8。

| ATOM dashboard 条目 | 当前能否运行本 PR V4 mono | 要做什么 / 如何认定收益 |
| --- | --- | --- |
| V4-Pro 普通档 C1/C4/C8 | 使用专属 ordinary mono/off variant、true/4 index 和包含实现的镜像 | 核对 resolved 配置与代码 SHA，验证命中后做 ABABAB |
| V4-Pro MTP3 C1 | 使用专属 MTP3 mono/off variant、no-prefix 和 false/1 index | 显式固定真实 MTP3、chat template、依赖和命中门禁；ABABAB 后才报告收益 |
| 普通 C16+ / MTP3 C4+ | 不是 auto 主体合格宽度；尾部可能命中 | 保留 native/混合兼容数据，标明覆盖率，不宣称高并发 mono 收益 |
| V4 TBO/DPA/DPA-MTP3/DPA-DSpark/EPLB MegaMoE | 不支持 | 保留原路径，不修改成强行使能 mono |
| V4.1-Flash DSpark TP4 | 有独立的已有 V4.1 mono 路径 | 目标最多 48 token rows，draft 最多 8 requests；不把它的结果当作 #2586 V4 证据 |
| MiniMax-M3 native TP4 FP8-KV/index | 有独立的已有 MiniMax mono 路径 | 最多 16 token rows、block128 等自身准入；不把它的收益归给 #2586 |

结构重构的 before/after TPOT 对照不是 native-off vs mono-on A/B，不能挪作
mono 加速比。实测收益必须由独立 artifacts 支撑，不在本方案中维护逐轮结果。

实施方式：保留现有无 suffix baseline，新增唯一 suffix，例如
`-mono-ordinary-off` / `-mono-ordinary-auto`、`-mono-mtp3-off` / `-mono-mtp3-auto`。
variant 自己携带 index override 和专属 scenarios；**不要把 ordinary true/4 放进
所有 variant 共用的 config**，否则会把 MTP3 改成不合格配置。
同一模型 prefix 下增 variant 不需增 workflow model checkbox；若增加顶层 prefix，
必须同步 dispatch inputs 和对应测试。

结果继续用 `{prefix}{suffix}-{isl}-{osl}-{conc}-{ratio}`，不要破坏历史 key。
新 series 首次没有上一晚 baseline 是预期行为；需显式 A/B compare，不能把无
baseline 当“无回归”。转换及 bundle metadata 增加 mono policy/profile、实际执行
次数、覆盖率、回退原因、SHA、依赖标识、配对实验 ID，dashboard 的 extra 显示
这些并链接完整证据。未命中/不完整的 cell 不进入 mono 收益结论。

手动 PR 验证使用本 PR ref/SHA 和 immutable image，`publish_to_dashboard=false`。
workflow 既要读取新增 catalog，也要运行对应代码：不能只给旧 main workflow 一个
`atom_commit` 就假定它已经读到了分支上的新 variant。先验证 checkout/catalog 的
resolved SHA 与 server `atom.__file__` 一致。ABABAB 通过后，合并/发布包含该代码
的镜像并记录 digest，再启用定时数据。正式 dashboard 发布是后续明确动作。

### 3.2 ATOM accuracy dashboard

`.github/workflows/atom-test.yaml` 及 `models_accuracy.json` 是精度入口。
给 V4 ordinary/MTP3 增加单独、可识别的 mono qualification 条目，配套 off 对照；
保留 C65 native 高并发和 DEP/DSpark 回归覆盖。新条目显示 mono execution coverage，
不要把不合格配置的 GSM8K pass 展示成 mono accuracy pass。

### 3.3 ATOM Agentic 与 Atomesh dashboard/结果页

`models_agentic.json` 当前是 V4.1-Flash DSpark5 TP4 C2/C4，走已有 V4.1 路径，
输出版本化 artifacts 供外部 viewer，不等同于原生 random performance dashboard。
`models_atomesh.yaml` 的 V4 1P1D（含 ordinary/MTP）在 decode 侧使用分离式 binding；
V4-Pro-0813 AgentX 又有 DPA、DSpark、prefix cache，均不是本 PR V4 mono 场景。
这些不属于“还差几个开关”的工作，不能为本 PR 自动扩展。

### 3.4 ATOM vLLM/SGLang plugin dashboard

`oot_benchmark_models.json` / `sglang_benchmark_models.json` 的 V4 项没有 native
`_v4_mono` 生命周期接入。插件若以后支持，需要自己的安装、cache/slot binding、
graph publication 和清理 bridge，属于另一项实现；本轮不在 plugin catalog 中加
native mono 参数冒充使能。

## 4. InferenceX 哪些能跑到

每次核对 InferenceX 的 `inferencex-e2e/configs/amd-master.yaml` 与 recipes，
在仓外记录实际 revision、resolved 参数和镜像 digest。下表描述配置类型的
可达性，不是实时状态或实测收益排名；条目变化后应重新核对。

| InferenceX arm | 配置事实 | 本 PR V4 mono |
| --- | --- | --- |
| `dsv4-fp4-mi355x-atom-agentic-mtp` 的 DSpark 配置 | DeepSeek-V4-Pro-0813、DSpark K6、prefix=true；高 C 又有 DEP/offload | 不支持，不能将其当作 MTP3 C1 |
| V4 ATOM disagg agentic LMCache DSpark | 多机分离式、DSpark、prefix/offload，旧镜像 | 不支持 |
| V4 vLLM/SGLang arms | 各框架自己的实现 | 不运行本 PR native ATOM V4 mono |
| V4.1 ATOM single-node TP2/TP4 agentic DSpark5 | bf16 KV、fp4 index、FULL，存在已有 V4.1 mono | 合格 decode shape 可运行已有 V4.1 mono；不是 #2586 |
| V4.1 vLLM MI355X TP2/TP4 | recipe 明确 `monokernel: true`，最多 48 decode rows | 是 vLLM 自己的 V4.1 mono，不是 ATOM V4 mono |

若要 InferenceX 验证 #2586，需要**新增而非改名复用 DSpark** 的单机纯 TP8
ordinary C1/C4/C8 或真实 MTP3 C1 arm，固定 no-prefix 和各自 index 配置，镜像
必须含本 PR。先做 ATOM 整网资格验证；再决定增加合格非 prefix workload，或另立
项目适配现有 prefix/agentic workload。不要为了接入当前 AgentX 曲线删掉它的
prefix/DSpark 然后声称保持了同一实验。
任何 synthetic acceptance 只用于明确标注的性能研究，整网精度必须真实 acceptance；
本 PR MTP3 初始收益验证也使用真实 acceptance。

## 5. 还差多少：按提交拆分的实施清单

第一阶段的可估范围是下面 **5 个独立工作包**；P0/P1 是先让 CI 真实跑到，
P2/P3 是精度修复与收益资格，P4 是发布。尚未完成的全量精度可能产生额外 kernel
修复，不能在数据出来前保证“只改配置即可完成”或给出确定 GPU 小时数。

| 工作包 | 主要落点 | 完成定义 |
| --- | --- | --- |
| P0：准入/配置契约 | `mono/dispatch.py`、`profile.py`、`tests/test_dsv4_mono_publication.py`、catalog tests | 合格普通/MTP3 配置确定；不合格路径明确拒绝；检查独立 `enable_tbo_decode` 拒绝是否需补齐，不能依赖 CLI 联动 |
| P1：CI 与观测 | `models_accuracy.json`、`atom-test.yaml`、`atom_test.sh`、`session.py`/model_runner 统计、benchmark bundle/模板 | ROCm import/compile smoke 无 skip；真实整网 mono 执行计数；client/timeout/fault/缺结果真失败；日志和退出统计可靠保存 |
| P2：精度资格与必要修复 | 精度 replay/check 工具、最小回归测试、真正分歧的 model/session/cache/kernel 文件 | 完整 GSM8K 与扩展任务通过，修复有根因和最小复现，不改低门槛 |
| P3：ATOM dashboard 配置与顺序 A/B | `models.json`、catalog tests、benchmark execution/summary、metadata converter | 4 个候选 × 标准场景顺序 ABABAB；可复现收益、覆盖率、尾延迟和稳定性报告 |
| P4：生产/外部同步 | image provenance、main schedule、InferenceX 新 eligible recipe | 镜像含代码、精度/收益门禁通过；再发布并同步，不提前宣传 |

提交建议分别为 `fix(dsv4): qualify mono deployments`、`test(ci): require V4 mono
execution`、实际发现问题后的 `fix(dsv4): ...`、`bench(dsv4): qualify mono gains`。
避免把数值修复、CI 绕行和 dashboard 修改揉在一个 commit。发布 InferenceX
变更应在 ATOM 验证之后，不夹入本 PR 以掩盖收益证据缺失。

### 5.1 依赖与 compiler/GPU fail-fast

- 固定 image digest、ROCm/torch/Triton、AITER wheel URL/artifact + SHA256 或源码
  SHA、FlyDSL 版本。AITER 当前 main requirements 已 pin `flydsl==0.4.0`，native
  Docker build 间接安装它；这不能证明历史 CI image/wheel 替换后的运行环境正确。
  先 smoke 与本 kernel 兼容的版本，再 pin；不武断再安装另一个版本。
- 记录 `atom`/`aiter`/`flydsl` 的实际 import 路径和版本；验证 ATOM 为 checked-out
  PR SHA，AITER 不是旧 source checkout 遮蔽 wheel。CPU 不强装 ROCm 依赖。
- 专属 GPU job 直接 import V4 model、MTP3 model、mono kernel，以及所需 AITER
  V4 符号（含 `cp_gather_indexer_k_quant_cache`、`rope_rotate_activation`、
  `flydsl_pa_mqa_logits_fp4` 等真实模块 import）；缺符号 **fail**，不 importorskip。
- FlyDSL compile-only build-key 测试在带 FlyDSL 的 compiler gate 必须 2 case
  全执行；再 fresh-cache 编译实际 S1/S4/S8、ordinary refresh/reuse 和 MTP q4
  variants。build-key 两 case 不等于完整生产 kernel 编译覆盖。
- 硬件预检 + 实际编译准入：gfx950/256 CU、spill scratch=0、LDS≤163840，
  `require_resident_grid(launch, 256, 512)` 实测 occupancy 通过。persistent mailbox
  kernel 必须全 grid resident，不能只看泛用 skill 的 LDS 经验值或只看编译成功。
- 旧 wheel 缺 V4 符号时修 image/依赖；若只有 DSpark confidence-head 既有容差
  测试失败，独立 matched baseline 复现并分类，不降容差、不伪装 mono 故障。

### 5.2 配置落地：普通与 MTP3 分开

以下是待接入现有启动脚本的配置，不是本轮已执行命令。两档共同设置：

```text
AITER_BF16_FP8_MOE_BOUND=0
ATOM_MOE_GU_ITLV=1
AITER_LOG_LEVEL=WARNING

-tp 8 --kv_cache_dtype fp8 --index_cache_dtype fp4
--no-enable_prefix_caching --cudagraph-mode FULL --level 0
```

先用 level0+FULL 隔离 compiler 组合风险；正式当前 dashboard 用 level3，需要
另做 level3+FULL 精度/性能资格，不能 level0 通过后直接声称 level3 已验证。
两者 A/B 都要用相同 level。eager NONE 是定位/兼容轨道，不混入 FULL 收益。

```text
ordinary:
  --hf-overrides '{"use_index_cache":true,"index_topk_freq":4}'
  --cudagraph-capture-sizes '[1,4,8]'
  --max-num-seqs 8

mtp3:
  --method mtp --num-speculative-tokens 3
  --hf-overrides '{"use_index_cache":false,"index_topk_freq":1}'
  --cudagraph-capture-sizes '[1]'
  --max-num-seqs 1
```

auto/off 对照只改 `--mono-decode auto/off`。`max-num-seqs` 是初始可控 harness
设置，不是默认生产配置；生产 512 等容量要另测准入、混合调度和 KV 预算。
MTP3 graph size 是 request 数 1，verify q=4，不把图列表 `[4]` 当作 C1。
最终以 resolved config/运行形状为准。prefill chunk、max-model-len、memory
utilization、block size 必须显式记录并在 A/B 相同。

快速 qualification 用 force 证明实际合格 live decode 没有偷偷 fallback；
正式 mixed/性能用 auto。force 的非自动宽度仅研究，不自动扩围。
普通/MTP 分别明确 greedy/temperature0、不请求 logprobs；MTP 性能 client 加
`--use-chat-template`，accuracy 的 completions/chat template 按对照任务合同一致。

### 5.3 “CI 不报错”的正确实现

1. 不改 heavy gate 绕开权限。当前标签已足够 native CI；GPU runner 排队不是
   代码错误。每次代码 push 后检查该 SHA 的最新 run，区分旧 cancelled run。
2. 优先新增 V4 mono ordinary C8 / MTP3 C1 accuracy 条目，显式 no-prefix、各自
   HF overrides、mono auto、正确图/level。保留 native C65/DEP/DSpark 回归条目。
   使用已支持的 `client_command` 指定 `num_concurrent=8/1`、GSM8K 3-shot、
   temperature0、明确 max_gen_toks、`--log_samples` 和输出路径占位符。
   不添加 schema 不认识的字段；若增加 `mono_required` 等结构字段，要一起改
   schema、matrix loader、workflow 传递与测试。不要以为 schema 中已有的
   `lm_eval_num_concurrent` 字段就一定被 native shell client 消费。
3. 30min 默认 accuracy budget 可能不足 C1 全1319 MTP：先测 pilot 的真实
   tokens/s/耗时，按全量量级 + 裕量设 `accuracy_timeout_minutes`，并同步 workflow
   step/job timeout。不能缩减样本却叫 full GSM8K。
4. 收集 workload 前后 per-rank stats **delta**：`decode_attempts`、`selected_steps`、
   `graph_replays`、`eager_forwards`、fallback reasons、profile/index policy。
   `selected_steps` 只是成功 publication，不是 device launch；还要执行计数和
   一次单独短 trace 验证真实 persistent kernel。
5. 当前 stats 在 model_runner 退出才打印 `ATOM_V4_MONO_SERVING_STATS`。
   benchmark bundle 的 EXIT trap 只停止 client/collector，并在 server stop 前打包；
   workflow server log dump/普通清理同样不足以可靠获得这份退出统计。
   accuracy 也先取结果再 teardown。实现已有 metrics/控制面上的无 GPU 同步
   stats snapshot（分 rank，且 off/disabled 有明确状态），在 drain 后、打包前抓取；
   或可靠 graceful stop→等待 worker stats→再打包，不能依赖固定2s后 SIGKILL。
   GPU job 必须上传 server log、resolved config、mono-stats.json、每题记录，
   失败也保留 artifacts。
6. 合格 fixed-width smoke：每个 TP rank 真实执行>0，FULL 时 replay>0，形状/计数
   一致且无未解释 refusal；B 的受控 steady decode 覆盖率目标≥95%，低于目标先
   定位而不是发布收益。整网 mixed coverage 单独报告；预期 prefill不算decode分母，
   尾部/native fallback 列原因，不把 token数/pub数/launch数混成同一指标。
7. 修 client error propagation：accuracy drain 的非0处理应像 benchmark 一样
   配 `set +e`/捕获状态，不能被 `set -e` 提前退出绕过诊断；不能 `wait CLIENT_PID
   || true` 后沿用旧结果。只接受本次独立 output目录的结果和正确样本数量。
   错误/超时/缺统计/NaN/空结果应失败，日志打包成功不能把原失败洗成绿。
8. coverage gate 单元测试至少包含：缺依赖、compile skip、只 capture没 replay、
   只有 warmup命中、disabled、rank不一致、过低覆盖率、期望 fallback、client非0、
   缺当前结果、drain fault/timeout。完整性只检查 completed==num_prompts 不够。

## 6. 验证顺序：先快后全，不并行精度与性能

按仓内 `atom-patterns` 的所有权边界组织修复，按 `run-atom-workload` 分别调用
现有 stop → tracked start → `wait_server_ready.sh` → tracked workload →
`wait_infer_drain.sh` → stop。启动/长 client 用工具跟踪的后台任务，不写临时
shell `&` wrapper；每个阶段可中止。ready 失败不得启动 client。
不执行宽泛 pkill/cache删除，不触碰别人的 GPU job；清理和 cold-cache 实验只针对
明确的本次进程、容器和独立 cache目录。

| 阶段 | 必测内容 | 下一步条件 |
| --- | --- | --- |
| Q0 CPU | publication/contract/build-key digest、共享 mono runtime/plan、V4.1 相关回归、catalog/schema/style | 全通过；CPU compiler skip 有明确分层解释 |
| Q1 compiler/import | 5.1 的依赖/全部生产形状编译与资源门禁 | 不 skip，不用 native fallback 代替编译 |
| Q2 快速整网 | 普通 C1/C4/C8、MTP3 C1；8个固定问题/短生成 + 少量128/256 token decode；force证明执行，auto证明分流 | 无 fault/卡住/异常重复、真实 mono 执行；短样本只作 smoke |
| Q3 精度 pilot | 固定50/100题 paired A/B + 边界/replay；完整输出足够 tokens/EOS | 根因清楚、无明显掉点，不能以 pilot 代全量 |
| Q4 全量精度 | 下节完整任务/所有合格档、level0/3对照、真实 MTP acceptance | 绝对/相对门禁通过，问题已修 |
| Q5 性能 | 标准场景先 ABABAB，再扩展矩阵；不带 profiling/check/debug 开销 | 配对收益、覆盖率、尾延迟、稳定性都合格 |
| Q6 发布资格 | main镜像/生产容量 canary、ATOM dashboard、合格 InferenceX arm | 可复现 artifact + 回滚开关，之后发布 |

Q0 可用现有 tests：`test_dsv4_mono_publication.py`、`test_dsv4_mono_build_key.py`、
`test_mono_runtime.py`、`test_mono_plan.py`、相关 V4.1 mono tests、
`test_benchmark_catalog.py`。后续新增 stats/配置/数值回归测试也进入对应 tier。
不靠重复全文件 DSpark GPU 测试取代本 PR 的四个合格档覆盖。

## 7. 全量整网精度方案

### 7.1 三路拆分：先分离 index 优化与 mono 数值

同一个 SHA/image/checkpoint，定义：

- **N**：native off、ordinary 模型原始 index 配置 false/1（生产原始参照）。
- **A**：native off、ordinary true/4（与 mono 相同 index 优化）。
- **B**：mono auto、ordinary true/4。

`N→A` 检查 index reuse 本身的精度影响；`A→B` 检查 fused mono 的影响。
不能把 N→B 的性能收益全归给 mono，也不能只比较 A/B 就忽略 index reuse
对原始模型质量的影响。MTP3 A/B 都是 false/1，另比较同条件非 MTP target，
确认 verifier/采样语义与真实 acceptance，不能用 ordinary true/4 混作它的 target。

### 7.2 GSM8K 与更广任务

1. GSM8K **全部1319题、3-shot**：普通 C1/C4/C8、MTP3 C1，各档同题、同顺序、
   同 harness/version/task prompt、同 seed/temperature/stop/context cap、同
   generation budget。先 completions 对齐现有 CI baseline，chat 轨道单独固定
   template/tokenizer；不拿3-shot与5-shot、原始与0813 checkpoint混比。
2. 每档 A/B 都完整评测，N 的普通控制轨道也完整评测；有波动/临界结果至少重复
   3组完整 matched runs，不能只补跑一个有利 B。保存 flexible/strict score、
   每题 token/text、finish_reason、生成长度、首次差异、retry/error/timeout。
3. 现有 catalog V4 absolute bar `accuracy_threshold=0.94` 保留，MTP acceptance
   fraction bar `0.622` 保留；CI 已正确把记录的百分数除100比较，不需为过关改单位。
   baseline 元数据0.96与备注0.9522不是同一次现场对照，paired A 才是本轮基线。
4. 建议在看结果前约定 paired non-inferiority margin **0.5个百分点**，使用按题
   paired bootstrap/McNemar 观察 win/loss与95%置信区间；B−A 置信下界需不低于
   −0.005。GSM8K 单组可能没有足够统计功效：结果不确定就增加预定 matched runs
   或扩大任务证据，不降低门槛、不把不显著当作证明等价。
5. 扩展整网：数学 MATH-500；代码 HumanEval/MBPP（隔离执行生成代码）；知识/推理
   MMLU-Pro或仓内已有同类任务；长上下文 needle/retrieval与跨窗口事实；JSON/stop/EOS
   协议与截断。固定 dataset revision、样本数、评分器、模板及已有质量bar；无现有
   bar的任务先登记 matched-native baseline及非劣规则，再运行，不事后挑任务。
6. 长上下文检查 1k/8k/32k/64k，128k及更长在模型、KV预算和真实 adapter bounds
   支持时继续，unsupported记录拒绝，不越界硬跑。生成预算应覆盖 reasoning；
   pilot确定最大token cap，A/B相同；被cap截断不能当正常错误或自动排除。
7. MTP记录每位置接受率、平均 tokens/forward、实际 verify q、拒绝后state推进、
   EOS附近行为。target质量过关不代表draft质量过关；acceptance下降即使GSM8K不掉
   也要定位。准确率轨道禁止 `spec_decode_acceptance_length` 等 synthetic override。

### 7.3 数值、状态与请求生命周期

- 固定 teacher-forced tokens，比 logits/top1/logit margin、各层 hidden、router
  expert IDs/weights、index top-k、attention/LSE、MHC、KV/压缩状态，定位真实误差。
  保留原算子 rtol/atol；bitwise text一致不是唯一要求，端到端过分也不是单层
  数值测试替代品。越过 argmax margin 的近并列分歧要统计，不能统称噪声。
- ordinary reuse phase0/1/2/3、refresh/reuse交界、CSA/SWA、压缩边界、block frontier
  追加、rolling window wrap、context接近capacity都覆盖。
- 请求补入/移除/重排、C8→C4→C1及非资格形状、取消/超时/EOS、slot重用及generation、
  state reset、cache rebind、weight replacement、graph recapture覆盖；prefix/fork
  不支持时在写cache前正确退回native，不能写一半再重试。
- MTP q4每行position/slot、next_token_locs、aux hidden taps、verify拒绝后回滚/推进
  独立测试。检查TP8各rank一致、epoch每stack只advance一次、两阶段不提前advance、
  microbucket不同lane无mailbox覆盖。
- FULL与NONE对照、level0与level3、fresh/reused compile cache、长时间多轮请求覆盖。
  短文本一致性和单轮耗时不能代替整网质量与稳定性证明。

## 8. 整网精度不通过：定位并修代码的闭环

**不调低 accuracy bar/数值容差，不靠隐藏 fallback、关闭 failing shape或改题目
掩盖失败。** 如需临时安全限制，明确是未完成资格、记录原因与待修项，不能宣布
该场景已使能成功。先冻结失败 artifact与SHA，然后按以下分支处理：

| 观测 | 优先判断 | 定位/应修改的代码边界 |
| --- | --- | --- |
| N通过，A和B都掉 | ordinary index reuse语义问题 | native `model.py` 的reuse与 mono `runner.index_policy`、indexer phase/tag；校正producer/consumer层、每请求refresh、slot/frontier失效，不用回退false/1冒充修复 |
| A通过，B掉，FULL/NONE都复现 | mono计算/packing差异 | `weights.py` 的lossless packing和scale布局；`kernels/attn_pre.py`/`indexer.py`/`attention.py`/`moe.py`/`mhc.py` 中最早分歧处 |
| NONE通过，FULL掉 | graph捕获/发布/陈旧buffer | `session.py`、`cache.py`、`contract.py`、model_runner replay/terminal；修owner/slot/graph generation、稳定地址、每步publish与输出引用 |
| level0通过，level3掉 | torch/Triton/Inductor组合或graph边界 | pin已验证组合、最小compiler复现、修compile边界；不能未经定位把mono算子全换掉 |
| C1通过，C4/C8掉 | 多行layout/TP/mailbox/lane问题 | bucket行映射、同请求q不拆分、padding live mask、lane独立peer buffer、epoch与barrier协议 |
| 首次通过，slot重用/长上下文掉 | 生命周期/rolling state/frontier失效 | `cache.py`/`contract.py`/`session.py` generation与压缩状态，不清空别人的live cache |
| ordinary通过，MTP3掉 | verifier q4、aux taps、采样/回滚契约 | native MTP3 model/drafter与mono session/cache衔接，核对每行metadata与接受后的状态 |
| GSM8K通过但acceptance/tokens-fwd下降 | draft/verify性能回归，不一定target掉点 | 对齐native acceptance、draft hidden/输出/next-token映射；保留真实acceptance门禁 |
| 偶发重复token/非有限值/TP卡住 | stale cache、竞争或数值溢出 | 最早非有限层、指针layout、scale、mailbox visibility、rank-uniform协议；修复后同时做稳定性和数值回归 |

具体执行步骤：

1. 保存失败题的完整prompt/token、参考生成、B生成、SHA/依赖/config、per-rank
   stats和log。先检查token budget、模板、错误重试、是否真的执行mono，排除harness
   截断/漂移，不能凭总分猜kernel。
2. 从首个分歧 token 之前的共同 prefix 开始 **teacher-forced replay**，固定metadata
   和state。A/B用两个独立engine/cache实例，或完整可靠restore，不在同一生产cache
   上先native写入再mono写入。不能将V4.1的shadow checker原样套V4而污染状态。
3. 做一次step的层级二分/探针，在最早分歧层保存输入、量化scale、top-k路由/索引、
   attention/LSE、输出。debug instrumentation默认关闭，单独运行，不进入性能。
   可借鉴V4.1 checker的组织方式，但保留V4自己的算术与cache协议。
4. 把失败降到最小operator/单层/少请求，分别对照native与高精度oracle；检查
   FP8 E4M3格式、MXFP4 E8M0 scale/byte offset、round/clamp、累加顺序、RoPE、
   MoE activation边界、MHC权重归一化。只在最早错误处修，不顺手重构全kernel。
5. 先加失败回归（修前红、修后绿），覆盖相邻width/phase/q/边界。device helper
   改动保持 `SOURCES` digest依赖完整；更新受影响UPSTREAM integrated hash/
   provenance，不用旧compile cache验证新设备代码。
6. 重新跑最小repro → Q0/Q1 → 所有4合格档smoke → 原失败子集 → 全1319及扩展任务。
   graph/cache修复增加多轮slot生命周期；MTP修复加真实acceptance；TP协议修复加
   多rank长时间稳定性。最后重新ABABAB，防止为了精度把热路径拖慢。
7. 若matched A也失败：独立报告native/环境/harness回归，按真实根因修复或升级
   依赖；不能把不合格baseline算成B通过。若统计仍不确定，保持资格pending继续
   按预定重复方案取证。

## 9. 全量性能：同机顺序 A/B/A/B/A/B

### 9.1 实验定义与排程

**无 MTP 的纯 TP8 ordinary 是独立必验主轨，不由 MTP 结果替代。** 首轮先完成
ordinary C1 的精度与性能，再完成 ordinary C4/C8；MTP3 C1 单独完成。这里的
“纯 TP”指 TP8、PP1、DP1、无 EP/DPA/上下文并行：MTP3 也可使用纯 TP 拓扑，
但它增加 draft/verify/rejection 算法，不能证明无 MTP 网络的性能或精度。
ordinary 必须关闭 MTP/DSpark，固定 true/4 index；MTP3 固定 false/1 index。
ordinary C16 的历史负收益不构成跳过 C1/C4/C8 的理由，也不自动扩大 auto 宽度。
执行下限为每个核心 cell 完成 A1→B1→A2→B2；默认完整资格仍追加 A3→B3。
只完成两对时明确报告重复次数和不确定性，不称完成六轮资格。ordinary C1/C4/C8
同时执行同配置完整 GSM8K native/mono 配对（至少两组独立启动），保持3-shot、
1319题、相同生成预算与scorer；性能与精度不同时占用GPU。收益目标未达到或精度
未通过时，定位、修复并重跑受影响 cell，不因缩短轮次或只检出正收益宣告目标达成。

性能的 A=同SHA `mono off`，B=同SHA `mono auto`，各档index/cache/graph配置完全
相同。ordinary额外N→A实验单独命名，拆出index reuse收益。不要比较旧镜像vs新
镜像、不同index档、不同TP、不同acceptance，再将差异全归mono。

每个 `(profile, C, ISL, OSL, level, workload)` cell 在**同一物理8卡节点/同一GPU
分配**依次 A1→B1→A2→B2→A3→B3。每run全新server，单独执行ready/warmup/workload/
drain/stop，全部记录。既有benchmark matrix把不同concurrency/variant分成不同
runner job，不能当作同机ABABAB；实现专门顺序paired block，复用已有启动/测量/
证据模块。不能用6个独立workflow dispatch假定得到同机顺序配对。

固定 checkpoint/tokenizer revision、image digest、ATOM SHA、AITER/FlyDSL/
ROCm/torch/Triton、TP/NUMA/CPU affinity、GPU clock/power条件、cachedtype/index/
blocksize、memory预算、model context/prefill chunk、seq cap、capture sizes、seed、
prompt/token列表及到达模式。节点无其他负载，accuracy与perf不同时运行。

首次cold import/compile、model load、graph capture及峰值内存另行记账；steady
测量前双方使用已经预热的编译缓存，不能A冷B热。fresh-cache compile实验在独立
目录另跑，不删除共享cache。每run至少2C warmup及≥10C正式请求；为稳定统计，
建议正式区间≥120s（不足就增请求），tail需足够样本。长场景用pilot估时，不为
满足时长删超时请求。perf正常不打开trace/check、rocm debug agent或同步探针；
另做一个短diagnostic run证明kernel路径。

### 9.2 矩阵与运行量

| 轨道 | 场景 | profiles / C | 用途 |
| --- | --- | --- | --- |
| 核心 dashboard资格 | ISL/OSL 1024/1024、8192/1024，ratio0.8 | ordinary C1/4/8，MTP3 C1 | 4×2=8 cells；每cell6run，共48个server测量run |
| 全量扩展 | ISL1k/8k/32k/64k × OSL256/1k/8k，ratio0.8 | 同4档 | 12×4=48cells，含核心重合；完整每cell6run共288run，不与核心重复计数 |
| 边界/production | 支持的128k+、接近adapter/model/context上界、真实EOS及短输出 | 同合格档 | pilot后按容量加测，unsupported明确标注 |
| 调度/兼容 | C2/16/32/64、混合长短prompt、请求率梯度、join/leave/cancel/logprob/sampling | ordinary/MTP及nativeunsupported代表档 | 检查fallback和host overhead，不当作auto高并发收益 |
| 结构探针 | 普通force C2/C16、MTP3 force C2 | 非自动资格 | 独立编号，只有正向端到端证据才另提扩围 |
| 执行模式 | level0 FULL、生产level3 FULL、NONE诊断 | 同档配对 | 分开报告，不混用结果 |

初轮先8个核心cell；扩展矩阵在精度与核心收益通过后执行，GPU小时数由pilot的
startup+warmup+正式请求耗时估算并给排程。整网生成不一定与microbenchmark同收益；
prefill占比高场景、真实EOS与低请求率要专门报告。rate sweep至少含低负载、约50%
及接近饱和；与封闭并发饱和测试分轨。

### 9.3 指标、统计与预登记门槛

#### 分层参考基线与目标

以下是固定的性能参考和目标依据，不是当前 HEAD 的通过声明。原始证据保留在
仓外 reference artifacts，不将逐轮输出、临时路径或 CI 快照提交到本方案。
参考来源为 `DSV4_MONOKERNEL_IMPLEMENTATION_PLAN.md` 链接的
`pr1234_baseline_NpRdsEly/BASELINE_COMPARISON.md` §4.2–4.3、同目录
`comparison.csv` 的 `remote_main_pinned` 行，以及
`dsv4_mono_p0_20261012/MTP_MONO_ABBA_305_310_311_312.json`。

| 口径 | 历史 native → Mono/PR | 比值或收益 | 证据等级 / 验收使用方式 |
| --- | --- | --- | --- |
| 单层 S1 HCA / CSA | 185.62→61.15 / 283.38→88.29 µs | 3.04× / 3.21× | TP8、position8192、三启动；未成对耗时比，不是等精度加速证明 |
| 单层 S2 HCA / CSA | 191.05→73.80 / 295.03→100.02 µs | 2.59× / 2.95× | 同上 |
| 单层 S4 HCA / CSA | 198.66→95.55 / 304.56→125.19 µs | **2.08× / 2.43×** | “算子层约2×”的确切参考；同上，不可直接外推整网 |
| 单层 S8 HCA / CSA | 210.05→145.27 / 320.59→184.77 µs | 1.45× / 1.74× | 同上；更宽 shape 不承诺统一2× |
| 无 MTP TP8，8k/1k，C1/C4/C8 TPOT | 作者公开 serving 声明 | **2.31× / 1.67× / 1.23×** | 未在本地复现；作为 ordinary 各档复现目标，不标为已有本地收益 |
| MTP3 TP8，C1，context8192，q4 target GPU | 15.396→9.897 ms | **1.556×** | 本地 fixed-prompt ABBA，8步warmup+每段24步；不是 random serving 或整网精度验收 |
| 同一 MTP3 C1，完整 speculative step GPU | 17.204→11.667 ms | **1.475×** | 包含 native draft/其他成本；同上 |
| 同一 MTP3 C1，有效 output throughput | 194.99→292.65 tok/s | +50.09% | committed length 均值并非完全相同；不能替代固定工作量 Total Tput 主指标 |

单层分母是受控 A8W4 materialized 完整层，PR 分子含 epoch advance；权重/cache
数值 oracle 未闭合，且 attention 数学与 native 不逐步相同。因此它们只给出
性能潜力和定位参考，不能作当前生产 FP8 KV/FP4 index、two-launch binary 的
硬时延预算。复测必须使用当前完整边界、同权重/cache 初态与已验精度的二进制。

预登记时区分三种判定，避免用“超过3%”冒充“达到预期目标”：

- **功能/质量放行**：完整精度、数值/状态安全、真实 acceptance、覆盖率和延迟
  门禁先通过；不以性能补偿精度下降。
- **可靠正收益下限**：继续使用下述 ≥3% 一致 Total Tput 收益与稳定性要求，
  仅判定是否有可辨认收益，不是性能目标。
- **目标达成**：ordinary C1/C4/C8 分别对照 TPOT **2.31×/1.67×/1.23×** 的
  复现目标，同时检查 Total Tput 与 TTFT/尾延迟；MTP3 C1 对照历史 target
  **1.556×**、完整step **1.475×**。必须分别记录“达到目标 / 有收益但未达目标 /
  无可靠收益 / 退化”，未达目标继续定位，不写成目标验收通过。历史绝对时延
  仅适用于原 fixed-prompt probe；新环境以新冻结 A 的同口径倍率验收。

旧方案的 C1 **1.8–2.2×** 是 DSpark K6 完整 draft+verify 的评估目标，不是
本 PR ordinary 或 MTP3 的已测收益，也不适用于本 PR 未使能的 DSpark 场景。
算子2×到整网的损失需用 prefill、61层decode、head/sampler、metadata/TP、
draft/rejection 的完整时间账本解释，不能直接把单层倍率或节省时间相加。

- 主指标 **Total token throughput=(输入+输出tokens)/正式测量时间**；另报
  Output Tput、请求/s，以及 `tput_per_gpu=total/实际参与GPU数`，本PR TP8为8。
  不用输出throughput代总throughput，不用配置文件名代真实GPU数。
- 报告mean/p50/p95/p99 TTFT、TPOT、ITL、E2EL；spec每token算法固定；MTP真实
  acceptance/平均tokens-forward；errors/retries/timeouts/completed/dropped/cap-EOS。
  记录GPU利用率/频率/功耗/温度、峰值HBM、KV capacity、startup/compile耗时。
- 每pair计算 `g_i=T_Bi/T_Ai−1` 与TPOT/尾延迟比，展示全部6个原始结果及3对变化，
  汇总配对几何均值/中位数与离散度。request-level bootstrap不能代替独立server
  repeat的不确定性；仅3对时95%区间很弱，临界/小收益追加预登记配对轮数。
- 建议预登记：A自身CV≤3%，A1→A3漂移≤3%；不满足则该完整cell判环境不稳定并
  原样保留，再整块重跑，不挑选有利pair。至少≥3%一致Total Tput收益且误差证据
  支持才称有收益；小于3%或区间跨0记“待定/无可靠收益”，不是强行加速结论。
- 非目标延迟/兼容场景建议不劣于3%，p95/p99不劣于5%，无新增error/fault/hang，
  精度先过门禁；具体SLA在首次测量前确认。这些是建议验收线，不是仓内既有硬bar。
- B必须在正式区间有真实mono执行；固定steady资格区间目标≥95% coverage，
  低覆盖先定位。mixed结果即使变快也要注明mono覆盖率和fallback原因，不能叫
  全程mono收益。全栈61层路径用独立trace抽样验证，不逐层开同步影响正式数据。

性能不通过时依次区分：没命中→准入/metadata；命中但host时间高→publication/
consensus/greedy terminal；device时间高→S1/S4/S8资源与两阶段、MoE/attention；
TTFT/HBM退化→初始化packing/KV预算/过多variant；C4/C8长尾→lanes/mailbox与rank
负载。先短trace证实再改对应所有权模块，修复后同一cell重新完整ABABAB，精度
相关改动同时回Q4。扩展auto宽度只能在新资格证据后另立修改。

## 10. 每个cell的交付物与最终放行标准

沿用 `atom.benchmarks.results` 的完整/summary bundle，新增而不覆盖字段；若schema
限制额外字段，同步版本/validator测试，或用明确的versioned mono sidecar。

```text
experiment/<cell>/<A1|B1|A2|B2|A3|B3>/
  manifest.json          # SHA/image/checkpoint/deps/node/GPU/seed/阶段/状态
  resolved-config.json  # 最终server/client配置，不仅原始CLI
  server.log / client.log
  mono-stats.json       # 各rank开始/结束与delta、policy/profile/拒绝原因
  result.json           # 全指标、完成数、paired id、覆盖资格
  requests.jsonl        # perf每请求；accuracy每题prompt/output/评分
  telemetry/            # GPU与server采样
  diagnostics/          # 单独trace/compile资源/失败最小repro（非正式perf）
```

结果主表每行包含dashboard/框架、checkpoint、profile、TP、C、ISL/OSL、level、
A/B total与per-GPU、gain、TPOT/p95/p99、accuracy/acceptance、coverage、image/SHA、
artifact链接。没跑/不支持/skip/环境失败/质量失败/收益待定分别标记，不能填0充数。

最终完成定义：

1. 四个合格档真正执行mono、CI不静默skip/fallback；GPU依赖与fresh compile资源门禁通过。
2. 完整精度及状态边界通过；每个发现的问题有修前失败/修后通过的回归与根因。
3. 核心与约定扩展ABABAB、生产level/容量通过；无收益cell如实记录，不强行默认使能。
4. ATOM原生performance与accuracy dashboard能正确标识mono，artifact可复现。
5. 合并后镜像含正确代码/依赖；canary可用 `--mono-decode off` 回滚，不影响其他 native路径。
6. InferenceX只同步已完成资格的新合格arm；现有DSpark/PD/plugin不冒充本PR覆盖。

依据：仓内 `.claude/skills/atom-patterns/SKILL.md`、`review-pr/SKILL.md`、
`run-atom-workload/SKILL.md`，以及 `ci-pr-guide.md`、`benchmark-guide.md`、
`inferencex-sync.md`。旧指南的并发/矩阵/路径与实际代码有漂移时，以本SHA workflow/
catalog/dispatch为准；继承其独立阶段、精度/性能隔离、Total Tput及MTP模板约束。
