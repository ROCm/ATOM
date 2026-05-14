# Mesh Router 重构 — 测试单元 Plan

> 关联: `router-refactor-plan.md` §4 / §11
> 总数: **94** 个单元（落在 §4 计划的 82-102 区间）
> 目标: 覆盖 placement core 100%；I/O/router 集成层覆盖切换路径
> 约定: `[ ]` = 未做；`[x]` = 已做；`[~]` = 进行中
> 行号: 引用基于 atom_mesh 分支当下 HEAD（取自 `router-refactor-plan.md` 撰写时的 grep 输出）
>
> ## 编码原则
>
> 所有代码变更必须遵守 **Karpathy Guidelines**，执行前加载: `/karpathy-guidelines`
>
> 额外约定（项目硬性要求）:
> - **不写无关注释**: 默认不加注释；只在 WHY 非显然时写一行。禁止 plan 引用、phase 标签（"P1 将实现"）、"覆盖 §X"、设计决策段落、过渡说明（"from pd_router.rs:1199"）写进代码——这些归 plan / PR description / commit message。
> - **TDD + Subagent review**: 写完 production code 必须启动 `rust-reviewer` subagent 独立 review，不要自审通过。
>
> ## 开发环境
>
> - Host 编辑代码，Container (`atom_sglang_mesh`) 编译执行，详见 `router-refactor-plan.md` 顶部约定
> - **严禁在 Container 内编辑/创建文件**（root 权限会破坏 Host 用户文件 ownership）
> - 运行测试: `docker exec -w /it-share/yajizhan/code/ATOM/atom/mesh atom_sglang_mesh cargo test --package atom-mesh`

---

## 测试夹具（在 P0b 实现）

- `MockWorkerSource`: 实现 `WorkerSource` trait，构造 `(model_id → Vec<Arc<dyn Worker>>)`、`hash_ring(model_id) → Option<Arc<HashRing>>`，可注入 unhealthy / 不同 worker_type / 不同 connection_mode
- `MockPolicySource`: 实现 `PolicySource` trait，可分别注入 regular / prefill / decode policy
- `RecordingPolicy`: 包一个真实 policy，记录每次 `select_worker` 调用的 `SelectWorkerInfo`（验证 text/tokens/headers/hash_ring 透传）
- `make_worker(url, worker_type, healthy, dp_rank?, bootstrap_port?, api_key?)` 工厂
- `make_descriptor(model_id, text, tokens, headers, stream, return_logprob)` 工厂

---

## A. Candidate 过滤 — `filter_candidates()` (18 个)

> 覆盖：`http/router.rs:131-147`、`grpc/.../worker_selection.rs:136-150 + 195-225`、`http/pd_router.rs:1199-1201, 1262-1266`

- [ ] **A01** `model_id=Some("m1")` 仅返回 m1 的 worker（D2 bug fix 核心证据 —— 不再静默落到 None）
- [ ] **A02** `model_id=None` 返回所有 worker（向后兼容兜底）
- [ ] **A03** `model_id=Some("unknown")` 且 registry 内无该 model → 返回空集合（错误分类由 planner 决定，候选层只负责过滤）
- [ ] **A04** 两个 model 都有 worker，请求 model A 不会污染到 model B 的 worker
- [ ] **A05** `worker_type=Some(Regular)` 过滤掉 Prefill / Decode
- [ ] **A06** `worker_type=Some(Prefill { .. })` 过滤掉 Regular / Decode
- [ ] **A07** `worker_type=Some(Decode)` 过滤掉 Regular / Prefill
- [ ] **A08** `connection_mode=Some(Http)` 过滤掉 gRPC worker
- [ ] **A09** `connection_mode=Some(Grpc { port: None })` 过滤掉 HTTP worker
- [ ] **A10** 全部 worker healthy → 全部进入候选
- [ ] **A11** 全部 worker unhealthy → 候选为空（caller 决定 NoAvailableWorkers）
- [ ] **A12** 健康/不健康混合 → 仅健康进入候选
- [ ] **A13** Registry 为空 → 候选为空，不 panic
- [ ] **A14** Prefill 带 `bootstrap_port=Some(8998)`（SGLang）与 `None`（vLLM）共存 → 都被 `WorkerType::Prefill { .. }` 匹配，不因 port 值丢候选
- [ ] **A15** 组合：`model_id + worker_type + connection_mode + 健康` 全部生效
- [ ] **A16** 组合：`model_id=Some + worker_type=None`（HTTP PD 旧调用形态）正确返回所有该 model 的 worker
- [ ] **A17** DP-aware: 同 URL 不同 `dp_rank` 的两个 worker 都进入候选（不去重）
- [ ] **A18** 大量 worker（≥256）下过滤性能不退化（断言 ≤ N×O(1)，仅 sanity）

---

## B. Policy 应用 — `apply_policy()` (14 个)

> 覆盖：`http/router.rs:149-178`、`grpc/.../worker_selection.rs:152-186, 227-267`、`http/pd_router.rs:1247-1280`

- [ ] **B01** `RoundRobin` policy 被调用一次，返回 idx 在 `0..candidates.len()` 内
- [ ] **B02** `Random` policy 被调用一次，返回有效 idx
- [ ] **B03** `PowerOfTwo` policy 被调用一次（由 `policies/power_of_two.rs` 决定 idx）
- [ ] **B04** `CacheAware` policy 被调用，且 `SelectWorkerInfo.request_text` 必须是 `Some` 否则触发 `needs_request_text=true` 路径
- [ ] **B05** `PrefixHash` policy 被调用，且 `SelectWorkerInfo.tokens` 必须为 `Some(non_empty)` 才进入哈希分支
- [ ] **B06** **D2 修复证据**: 请求 model "m1" 时 `apply_policy` 收到的 `hash_ring` 是 `worker_source.hash_ring("m1")` 的结果，**不是** `hash_ring(UNKNOWN_MODEL_ID)`
- [ ] **B07** `model_id` 在 registry 中无对应 hash_ring → 注入 `hash_ring=None`，policy 仍可工作（fallback 到非 ring 分支）
- [ ] **B08** `request_text=Some("hello")` 透传至 policy 收到的 `SelectWorkerInfo.request_text`
- [ ] **B09** `tokens=Some(&[1,2,3])` 透传至 policy 收到的 `SelectWorkerInfo.tokens`
- [ ] **B10** `headers=Some(&hm)` 透传至 policy 收到的 `SelectWorkerInfo.headers`
- [ ] **B11** Policy `select_worker` 返回 `None` → `apply_policy` 返回 `Err(PolicyReturnedNone)` 而不是 panic
- [ ] **B12** `candidates.is_empty()` → 调用方在进入 `apply_policy` 前应已 short-circuit；本测试断言 `apply_policy` 在空 slice 上不调用 policy（防御）
- [ ] **B13** `cache_aware` policy 的 `needs_request_text()=true` 在 `PolicySource::pd_needs_request_text()` 中被正确聚合（覆盖 §11.4）
- [ ] **B14** `apply_policy` 返回的 idx 用于 `available[idx].clone()` 时不越界（使用上面 B01-B05 任一断言 idx < len）

---

## C. Regular planning — `DefaultPlanner::plan()` Single 分支 (11 个)

> 覆盖：HTTP `select_worker_for_model` 全函数、gRPC `select_single_worker` 全函数

- [ ] **C01** `model_id=Some("m1")`，registry 内 m1 有 1 个健康 Regular HTTP worker → 返回 `PlacementPlan::Single { worker.url == m1_worker_url }`
- [ ] **C02** `model_id=None`，registry 内有任意 model 的健康 worker → fallback 到 `get_default_policy`，返回任一 worker
- [ ] **C03** `model_id=Some("m_unknown")`，registry 无该 model → 返回 `Err(PlacementError::ModelNotFound { model_id: "m_unknown" })`（§11.7）
- [ ] **C04** **D2 跨 model 隔离证据**: registry 内 m1 / m2 各有 worker，`model_id=Some("m1")` 永远不返回 m2 的 worker（重复跑 100 次断言 URL 全部命中 m1）
- [ ] **C05** **D2 hash_ring 证据**: 配置 `PrefixHash` policy + 多 model，断言 `WorkerSource::hash_ring` 的调用参数等于请求的 `model_id`（不是 `UNKNOWN_MODEL_ID`）
- [ ] **C06** gRPC 单 worker：`connection_mode` 自动选 `Grpc { port: None }`，断言不会返回 HTTP worker
- [ ] **C07** gRPC + `model_id=None` → fallback 到 default policy（与 HTTP 行为一致）
- [ ] **C08** 全部 unhealthy → `Err(NoAvailableWorkers)`
- [ ] **C09** Registry 空 + `model_id=None` → `Err(NoWorkers)`
- [ ] **C10** Registry 非空但全部是 Prefill / Decode（无 Regular） + `model_id=None` → `Err(NoAvailableWorkers)`（候选过滤后为空）
- [ ] **C11** Policy 强制返回 None → `Err(PolicyReturnedNone)` 透传到顶层

---

## D. PD planning — `DefaultPlanner::plan()` Pair 分支 (14 个)

> 覆盖：HTTP `select_pd_pair`（pd_router.rs:1191-1245）、gRPC `select_pd_pair`（worker_selection.rs:188-268）

- [ ] **D01** 1P + 1D normal（HTTP）→ `PlacementPlan::Pair { prefill, decode, prefill_policy, decode_policy }`
- [ ] **D02** 1P + 1D normal（gRPC）→ 同上
- [ ] **D03** **D2 跨 model 隔离 PD 路径**: m1 与 m2 各有 P/D，请求 m1 永远不混入 m2 worker
- [ ] **D04** **D2 hash_ring 在 PD 路径**: PrefixHash 配置下，`hash_ring` 调用使用真实 model_id 而非 UNKNOWN_MODEL_ID（覆盖 pd_router.rs:1207 与 worker_selection.rs:236 的 bug）
- [ ] **D05** 0 prefill + n decode → `Err(NoPrefillWorkers)`
- [ ] **D06** n prefill + 0 decode → `Err(NoDecodeWorkers)`
- [ ] **D07** **D4 行为变更证据（gRPC PD）**: 注入不同的 `prefill_policy`（RoundRobin）和 `decode_policy`（Random），验证两次 `select_worker` 调用分别走两个 policy 实例 —— 不再像今天那样用同一个 per-model policy（§11.1）
- [ ] **D08** **D4 行为对齐（HTTP PD）**: 同 D07，验证 HTTP PD 行为不退化（HTTP 今天已是分离）
- [ ] **D09** Prefill policy 返回 None → `Err(PolicyReturnedNone)`，且 decode policy 不被调用（短路）
- [ ] **D10** Decode policy 返回 None → `Err(PolicyReturnedNone)`，且 prefill 已选定的事实在 trace 中可见（防止悄无声息）
- [ ] **D11** `tokens=Some(&[..])` 同时传到 prefill_policy 和 decode_policy 的 SelectWorkerInfo
- [ ] **D12** `text=Some("...")` 同样传到两个 policy
- [ ] **D13** `headers=Some(&hm)` 同样传到两个 policy
- [ ] **D14** Prefill worker 携带 `bootstrap_port=Some(8998)` 时 `PlacementPlan::Pair.prefill` 的 worker 引用保留该字段（adapter 层依赖）

---

## E. BackendAdapter — `backend/sglang.rs` + `backend/vllm.rs` (12 个)

> 覆盖：`http/pd_router.rs:348-449`（SGLang bootstrap + vLLM kv_transfer_params）

### E.1 SGLang
- [ ] **E01** `inject_prefill_fields` 单请求：`bootstrap_host` / `bootstrap_port` / `bootstrap_room` 三个 key 写入；值与 prefill worker 字段一致
- [ ] **E02** `inject_prefill_fields` 单请求 prefill 的 `bootstrap_port=None` → `bootstrap_port` 字段写为 `Value::Null`（非缺失）
- [ ] **E03** `inject_decode_fields` 在 SGLang 模式下应该是 no-op（SGLang 走 dual-dispatch，不在 decode 上注入）—— 文档化
- [ ] **E04** `inject_batch_prefill_fields(batch_size=3)`: 三个 key 都是长度 3 的数组
- [ ] **E05** `inject_batch_prefill_fields(batch_size=3)`: 三个 `bootstrap_room` 互不相同（验证 `generate_room_id()` per-element 调用）
- [ ] **E06** `inject_*` 在非 JSON object 的 body 上 → `Err(AdapterError::BodyNotObject)`，不修改 body

### E.2 vLLM
- [ ] **E07** `inject_prefill_fields`: `kv_transfer_params.do_remote_decode=true`、`do_remote_prefill=false`、`transfer_id` 形如 `xfer-{uuid}`；同时 force_prefill 改写：`stream=false`、`max_tokens=1`、`max_completion_tokens=1`（如 body 已含此 key）
- [ ] **E08** `inject_decode_fields`: `kv_transfer_params.do_remote_prefill=true`、`remote_bootstrap_addr` 取自 `VllmPrefillInfo.bootstrap_addrs[prefill_url]`、`remote_engine_id` 取自 `engine_ids[prefill_url][dp_rank]`、`transfer_id == prefill 侧 transfer_id`（**§11.2 证据：PairCtx 共享 transfer_id**）
- [ ] **E09** `inject_prefill_fields` force_prefill 时若 body 没有 `max_completion_tokens` 字段则不强加（只在已存在时改写，覆盖 pd_router.rs:443-445）
- [ ] **E10** `inject_prefill_fields` force_prefill 时移除 `stream_options` key（pd_router.rs:446）
- [ ] **E11** `VllmPrefillInfo.bootstrap_addrs` 缺 prefill_url → `prepare_pair` 返回 `Err(AdapterError::BootstrapAddrMissing { prefill_url })`
- [ ] **E12** `VllmPrefillInfo.engine_ids[prefill_url]` 缺当前 `dp_rank` → `prepare_pair` 返回 `Err(AdapterError::EngineIdMissing { prefill_url, dp_rank })`

---

## F. PlacementTrace (6 个)

> 覆盖：`core/placement/trace.rs`

- [ ] **F01** trace.model_id 等于请求的 `model_id`（`Some / None` 都覆盖）
- [ ] **F02** trace.candidate_count_before / after 反映过滤前后数量（健康 + 过滤 unhealthy 后变化）
- [ ] **F03** trace.selected_urls：Single 模式 1 个；Pair 模式 2 个（prefill / decode 顺序固定）
- [ ] **F04** Single 模式 trace.policy_name 等于实际调用的 policy name（如 `"round_robin"`）
- [ ] **F05** Pair 模式 trace 同时含 prefill_policy_name 与 decode_policy_name（验证 §11.1 不再共用同一名称）
- [ ] **F06** trace.hash_ring_key：当 `model_id=Some("m1")` 时为 `Some("m1")`；当 `model_id=None` 时为 `None`（不写 UNKNOWN_MODEL_ID）

---

## G. Error 路径 — 每个 PlacementError variant (9 个)

> 覆盖：`core/placement/types.rs::PlacementError`

- [ ] **G01** `NoWorkers` 触发：registry 完全为空 + `model_id=None`
- [ ] **G02** `NoAvailableWorkers` 触发：registry 有 worker 但全部 unhealthy
- [ ] **G03** `NoPrefillWorkers` 触发：PD 路径，0 prefill + n decode
- [ ] **G04** `NoDecodeWorkers` 触发：PD 路径，n prefill + 0 decode
- [ ] **G05** `PolicyReturnedNone` 触发：mock policy 强制返回 None
- [ ] **G06** `ModelNotFound { model_id }` 触发：`model_id=Some("m_x")` + registry 内无 m_x（按 §11.7 规则）
- [ ] **G07** `NoAvailableWorkers` 在 PD 路径：prefill / decode 都存在但全部 unhealthy（验证不会被错分类成 NoPrefillWorkers / NoDecodeWorkers）
- [ ] **G08** `AdapterError` 各 variant（BodyNotObject / BootstrapAddrMissing / EngineIdMissing）能被 router 层捕获并转 5xx Response
- [ ] **G09** Error Display / Debug 包含 model_id / dp_rank / prefill_url 等关键字段（grep-able 到日志）

---

## H. Router 集成（端到端，最少必要） (10 个)

> 覆盖：HTTP router / HTTP PD router / gRPC stage 切到 planner 后的契约

- [ ] **H01** HTTP regular: `route_typed_request` → planner 返回 `Single` → 请求成功 POST 到选中 worker URL（用 mock client 验证 URL）
- [ ] **H02** HTTP regular: planner `Err(NoWorkers)` → 503 `service_unavailable("no_workers", ...)`
- [ ] **H03** HTTP regular: planner `Err(ModelNotFound)` → 503 且 body 含 model name（覆盖 §11.7 错误路径，用户可调试）
- [ ] **H04** gRPC `WorkerSelectionStage` Regular 模式 → planner 返回 `Single` → `ctx.state.workers = WorkerSelection::Single { .. }`
- [ ] **H05** gRPC stage PD 模式 → planner 返回 `Pair` → `ctx.state.workers = WorkerSelection::Dual { prefill, decode }`
- [ ] **H06** gRPC stage planner Err → stage 返回 `Err(error::service_unavailable(...))`，错误信息含 model_id
- [ ] **H07** HTTP PD SGLang: planner Pair → `BackendAdapter::prepare_pair` → `inject_prefill_fields` 写入 bootstrap_* → `execute_dual_dispatch` 同时 POST 到 prefill 和 decode（mock client 断言两个 URL 都被命中）
- [ ] **H08** HTTP PD vLLM: planner Pair → `prepare_pair` 产出 `transfer_id` → prefill request 含 `do_remote_decode=true`、decode request 含 `do_remote_prefill=true` 且 `transfer_id` 一致（断言 mock client 拦截到的两个 body）
- [ ] **H09** HTTP PD batch（SGLang completions batch）: planner Pair + `inject_batch_prefill_fields(n)` → body 中 `bootstrap_host/port/room` 是长度 n 数组
- [ ] **H10** HTTP PD retry: 第一次 attempt planner 选中 (P1, D1)；mock 让 P1 返 503；第二次 attempt 重新调 planner 时 `text/headers/tokens` 与首次一致（验证 `Arc<HeaderMap>` 路径，§11.6 顺手验证 worker.api_key() 注入若选 B）

---

## 总计

| 类别 | 数量 | 覆盖 |
|------|------|------|
| A. Candidate 过滤 | 18 | 4 处分散过滤代码 |
| B. Policy 应用 | 14 | HTTP `pick_worker_by_policy_arc` + gRPC stage 内联 |
| C. Regular planning | 11 | `select_worker_for_model` + `select_single_worker` |
| D. PD planning | 14 | HTTP `select_pd_pair` + gRPC `select_pd_pair` |
| E. BackendAdapter | 12 | `inject_bootstrap_into_value` + `inject_kv_transfer_params` + VllmPrefillInfo |
| F. PlacementTrace | 6 | 新模块 |
| G. Error 路径 | 9 | PlacementError × 7 + AdapterError × 2 |
| H. Router 集成 | 10 | 切换路径契约 |
| **合计** | **94** | |

---

## P0b 完成标准

- 上述 94 条全部转成 `cargo test --package atom-mesh` 可识别的 `#[tokio::test]` / `#[test]`
- 每条都 `panic!("not implemented")` 或 `assert_eq!(true, false)` 占位 → `cargo test` 全部 fail，且 fail 原因符合预期（不是 build 错误）
- `MockWorkerSource` / `MockPolicySource` / `RecordingPolicy` 已实现并能在测试间复用
- `cargo build` 干净通过；`cargo clippy` 无新 warning

## P1-P5 推进时的勾选规则

- 实现某个测试覆盖的代码 → 把对应测试改成真实断言并跑绿 → 把 `[ ]` 改成 `[x]`
- 测试发现 plan §11 中某条决策需要返工 → 在本文件对应测试上方加 `> ⚠️ 阻塞：见 §11.X`
- P6 cleanup 阶段不允许新增测试，只允许把 `[ ]` 改 `[x]`；新增意味着覆盖漏洞需要返工 P0b

---

## 不在范围（提前声明，避免 review 时 scope creep）

- KV cost model / pair-native placement 的测试（plan §7 已排除）
- `WorkerRegistry` / `PolicyRegistry` 内部实现单测（只通过 trait mock 用，不重测）
- HTTP PD dispatch 拓扑（dual / fire-and-forget）的端到端测试（拓扑不变，复用 H07/H08 sanity check 即可）
- gRPC pipeline 框架本身的测试
- 性能 / 压测（独立 benchmark suite）
