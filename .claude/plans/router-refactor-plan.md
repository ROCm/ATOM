# Mesh Router 统一重构 — v0.2 (决策锁定版)

> 状态: 决策已锁定，待启动 P0
> 范围: `atom/mesh/src/routers/` + 新增 `core/placement/`
> 时间: ~10.5 工作日 / 2 周
> 上一版: `unified-router-refactor.md` (v0.1, 讨论版)
>
> ## 编码原则
>
> 所有代码变更必须遵守 **Karpathy Guidelines**，执行前加载: `/karpathy-guidelines`
>
> 额外约定（项目硬性要求）:
> - **不写无关注释**: 默认不加注释；只在 WHY 非显然时写一行（隐藏约束、不变量、为修某 bug 的 workaround）。禁止把 plan 引用、phase 标签、"P1 将实现"、"覆盖 §X"、设计决策段落写进代码——这些属于 plan / PR description / commit message，代码里只留代码本身能解释清楚的内容。
> - **不写元注释**: 不要用 `// from pd_router.rs:1199`、`// 此函数即将被替换` 之类的过渡说明；refactor 进度通过 git 历史和 plan 跟踪，不在代码里。
> - **TDD + Subagent review**: 写完 production code 必须启动 subagent（`rust-reviewer`）做独立 review，不要自审通过。
>
> ## 开发环境约定
>
> - **编辑区（Host）**: 当前用户环境，Claude Code 在此编辑 `.rs` 文件、执行 git 操作
> - **执行区（Docker）**: `atom_sglang_mesh` 容器，Rust 1.95 toolchain，仅用于编译/测试
> - 代码路径 `/it-share/yajizhan/code/ATOM` 通过 `-v /it-share:/it-share` 挂载，Host 与 Container 共享同一路径
> - **⚠️ 严禁在 Container 内编辑代码** — Container 以 root 运行，修改文件会改变 owner 导致 Host 用户丧失写权限
> - 启动容器: `bash atom/mesh/scripts/docker_start.sh`
> - 编译/测试命令模板:
>   ```bash
>   docker exec -w /it-share/yajizhan/code/ATOM/atom/mesh atom_sglang_mesh cargo build
>   docker exec -w /it-share/yajizhan/code/ATOM/atom/mesh atom_sglang_mesh cargo test --package atom-mesh
>   docker exec -w /it-share/yajizhan/code/ATOM/atom/mesh atom_sglang_mesh cargo clippy --package atom-mesh
>   ```

本版相对 v0.1 的变化：

- 删除所有防御性设计（compat flag / legacy mode / parity test / 行为零 diff）—— 项目未上线
- 锁定架构：placement 提升为 `core/` 一等公民（B 路线），不是 router 内 helper（A 路线）
- 锁定 backend adapter 范围：只管 wire 注入，dispatch 拓扑保留在 router（C1 方案）
- 锁定 PD policy 模型：分离（prefill ≠ decode）但**全局**，不做 per-model PD policy
- 锁定测试策略：TDD 路径，placement core 100% 单测；`WorkerRegistry` 和 `PolicyRegistry` 抽 trait 给 mock
- 删除 §4.10 unified vs legacy 整节
- 修 D2 bug（HTTP 候选按 model_id 过滤）作为本次必修项

---

## 1. 决策板

| ID | 决策 | 选定 |
|----|------|------|
| D1 | placement 架构 | **B**: `core/placement/` 一等公民，router 退化为协议适配器 |
| D2 | HTTP 不按 model_id 过滤候选 | **bug**，本次修复 |
| D3 | BackendAdapter trait scope | **C1**: 只管 wire 字段注入；dispatch 拓扑保留在 router 内 `match strategy` 分支 |
| D4 | PD policy 分离 | **是**: prefill_policy ≠ decode_policy，HTTP/gRPC 都分离 |
| D4.1 | PD policy 粒度 | **α 全局分离**: 不做 per-model PD policy |
| D5 | WorkerRegistry mock | **抽 trait**: `WorkerSource` |
| D5.1 | PolicyRegistry mock | **抽 trait**: `PolicySource` |
| D6 | 测试边界 | **分层**: candidate / policy_apply / planner / adapter 各自单测 + 少量集成测 |
| D7 | 测试覆盖范围 | **placement core 100%**；I/O 层维持现状（HTTP 40 测试，gRPC 0 测试） |
| D8 | 防御性设计 | **全部砍**: 无 compat flag / legacy mode / parity test |

---

## 2. 目标目录结构

```
atom/mesh/src/
├── core/
│   ├── (现有: worker_registry, worker_manager, policies/, ...)
│   └── placement/                         ← 新增
│       ├── mod.rs
│       ├── types.rs                       — RequestDescriptor, PlacementPlan, PlacementTrace, PlacementError
│       ├── traits.rs                      — WorkerSource, PolicySource (mock-friendly)
│       ├── candidate.rs                   — filter_candidates()
│       ├── policy_apply.rs                — apply_policy()
│       ├── planner.rs                     — PdPlanner trait + DefaultPlanner impl
│       ├── trace.rs                       — PlacementTrace 构造与 emit
│       └── backend/                       — wire 字段注入
│           ├── mod.rs                     — BackendAdapter trait
│           ├── sglang.rs                  — bootstrap 注入
│           └── vllm.rs                    — kv_transfer_params 注入 + VllmPrefillInfo cache
└── routers/
    ├── shared/                            ← 新增
    │   └── metrics_utils.rs               — 从 grpc/utils.rs 迁出 route_to_endpoint, error_type_from_status
    ├── http_router.rs                     ← 从 http/router.rs 上调 + 瘦身 (~500 行)
    ├── http_pd_router.rs                  ← 从 http/pd_router.rs 上调 + 瘦身 (~1200 行)
    ├── grpc/                              (保留目录，仍有 client/pipeline/etc.)
    │   ├── router.rs                      (~250 行)
    │   ├── pd_router.rs                   (~220 行)
    │   └── common/stages/worker_selection.rs (~50 行，瘦身后只剩 stage 包装)
    ├── (delete: http/, pd_types.rs)
    └── (其他不变)
```

---

## 3. 类型签名 (sketch，P0a 锁定具体形状)

```rust
// core/placement/types.rs
pub enum Protocol { Http, Grpc }

pub struct RequestDescriptor<'a> {
    pub model_id: Option<&'a str>,
    pub protocol: Option<Protocol>,  // optional hint, 后续 DynamicRouter 阶段去掉
    pub text: Option<&'a str>,
    pub tokens: Option<&'a [u32]>,
    pub headers: Option<&'a HeaderMap>,
    pub stream: bool,
    pub return_logprob: bool,
    // role 不作为输入 — planner 看 worker pool 有无 Prefill+Decode 自动判断
}

pub enum PlacementPlan {
    Single { worker: Arc<dyn Worker>, policy_name: &'static str, trace: PlacementTrace },
    Pair {
        prefill: Arc<dyn Worker>,
        decode: Arc<dyn Worker>,
        prefill_policy: &'static str,
        decode_policy: &'static str,
        trace: PlacementTrace,
    },
}

pub struct PlacementTrace { /* model_id, candidate counts, hash_ring_key, notes, ... */ }

pub enum PlacementError {
    NoWorkers, NoAvailableWorkers, NoPrefillWorkers, NoDecodeWorkers,
    PolicyReturnedNone, ModelNotFound,
}

// core/placement/traits.rs (mock-friendly)
pub trait WorkerSource: Send + Sync {
    fn workers_filtered(&self, model_id: Option<&str>, worker_type: Option<WorkerType>,
                        connection_mode: Option<ConnectionMode>) -> Vec<Arc<dyn Worker>>;
    fn hash_ring(&self, model_id: &str) -> Option<Arc<HashRing>>;
}
pub trait PolicySource: Send + Sync {
    fn regular_policy(&self, model_id: Option<&str>) -> Arc<dyn LoadBalancingPolicy>;
    fn prefill_policy(&self) -> Arc<dyn LoadBalancingPolicy>;
    fn decode_policy(&self) -> Arc<dyn LoadBalancingPolicy>;
}

// core/placement/planner.rs
#[async_trait]
pub trait PdPlanner: Send + Sync {
    async fn plan(&self, req: &RequestDescriptor<'_>) -> Result<PlacementPlan, PlacementError>;
}

// core/placement/backend/mod.rs
// 锁定 (§11.2 + §11.13): trait 必须 object-safe，PairCtx 在 trait 层是 type-erased
// `Box<dyn Any + Send + Sync>`，每个具体 adapter 在 prepare_pair 里 box 自己的
// concrete ctx (SglangPairCtx / VllmPairCtx)，在 inject_* 里向下 cast，cast 失败
// 返回 AdapterError::CtxTypeMismatch（理论不会发生 — 同一个 router 持同一个 adapter
// + ctx，cast 失败说明调用方串了 adapter）。
// vLLM 的 force_prefill 协议必需的请求形状强制（stream=false / max_tokens=1 /
// max_completion_tokens / 删除 stream_options，pd_router.rs:440-447）由
// inject_prefill_fields 顺带完成（D3 描述放宽，§11.3）。
pub type PairCtx = Box<dyn std::any::Any + Send + Sync>;

pub trait BackendAdapter: Send + Sync {
    fn prepare_pair(&self, prefill: &dyn Worker, decode: &dyn Worker) -> Result<PairCtx, AdapterError>;
    fn inject_prefill_fields(&self, body: &mut Value, ctx: &PairCtx) -> Result<(), AdapterError>;
    fn inject_decode_fields(&self, body: &mut Value, ctx: &PairCtx) -> Result<(), AdapterError>;
    fn inject_batch_prefill_fields(&self, body: &mut Value, ctx: &PairCtx, batch_size: usize) -> Result<(), AdapterError>;
}
```

---

## 4. 测试用例分类与计数 (P0a 展开样本，P0b 全部展开)

| # | 类别 | 测试数 | 覆盖 |
|---|------|--------|------|
| A | Candidate 过滤 | 15-20 | model_id 过滤（含 D2 bug fix）/ worker_type / connection_mode / 健康过滤 / DP-aware / 组合 |
| B | Policy 应用 | 12-15 | 5 个 policy 调用契约 / hash_ring 按 model 注入 / text+tokens+headers 透传 |
| C | Regular planning | 10-12 | 单 model / 跨 model 隔离（D2 fix）/ model_id=None fallback / 全部 unhealthy / 空 registry |
| D | PD planning | 12-15 | 1P+1D / 跨 model 隔离 / 0P 0D 报错 / prefill_policy ≠ decode_policy 验证 / hash_ring keyed by model / tokens 传递 |
| E | BackendAdapter | 10-12 | SGLang bootstrap / vLLM kv_transfer_params / batch / missing metadata 报错 / VllmPrefillInfo cache |
| F | PlacementTrace | 5-6 | 字段完整 / candidate 数前后 / selected URLs / policy_name / hash_ring_key |
| G | Error 路径 | 8-10 | 每个 PlacementError variant 一个触发用例 |
| H | Router 集成 | 10-12 | HTTP→planner / gRPC stage→planner / HTTP PD dispatch+adapter |
| | **合计** | **82-102** | |

---

## 5. Phase 计划

| Phase | 内容 | 工时 | 状态 | 完成标准 |
|-------|------|------|------|---------|
| P0a | spec 定稿: 决策 + type signatures + 12 个样本测试 + 测试分类总表 | 0.5-1 天 | ✅ done | 文档评审通过 |
| P0b | 完整 82-102 测试用例表（setup/input/expected）+ 转成 cargo test 全红骨架 | 1 天 | ✅ done（94 个 placeholder + 10 个 fixture smoke 全绿） | `cargo test` 全部 fail 且原因符合预期 |
| P1 | 实现 placement core (candidate / policy_apply / planner / trace) | 2 天 | ✅ done | A/B/C/D/F/G 类测试全绿 |
| P2 | 实现 BackendAdapter (SGLang + vLLM) | 1.5 天 (含 §11.13 +0.5) | ✅ done | E 类测试全绿 |
| P3 | gRPC 切换 (WorkerSelectionStage 调 planner) | 1 天 | pending | gRPC 走新 planner，H 类 gRPC 集成测全绿 |
| P4 | HTTP Regular 切换 | 1 天 | pending | http_router.rs 走 planner，`routers/http/router.rs` 现有 15 个测试全绿 |
| P5 | HTTP PD 切换 (含 BackendAdapter wiring) | 2 天 | pending | http_pd_router.rs 走 planner + adapter，HTTP PD 25 测试全绿 |
| P6 | 清理: 死代码删除 / 反向依赖修 / 文件扁平化 / clippy clean | 1 天 | pending | clippy 无 warning，cargo test 全绿 |
| **合计** | | **10-10.5 天** | | |

### 5.2 P1 实际产出（已落地）

实现：

| 文件 | 改动 | 关键决策 |
|------|------|---------|
| `candidate.rs` | `filter_candidates()` = `WorkerSource::workers_filtered` + `is_available()` 过滤 | trait 文档锁定 Prefill 按 variant 松匹配 |
| `policy_apply.rs` | `apply_policy()` 返回 `Arc<dyn Worker>`；空候选返 `NoAvailableWorkers`，policy 返 None 返 `PolicyReturnedNone` | 调用方在 planner 已短路；这里是防御 |
| `trace.rs` | `for_single` / `for_pair` 纯结构装配 | Pair 的 `candidate_count_before/after` 用 P+D 总和 |
| `planner.rs` | `DefaultPlanner` 自动检测模式：regular_pool 非空 → Single；P+D 都非空 → Pair；其他 → 类型化 error | `plan_single`/`plan_pair` 复用 `plan()` 已 fetch 的 raw pool，不二次查 source |
| `traits.rs` | `WorkerSource::workers_filtered` 加 doc：Prefill 松匹配；P3 wrapping `WorkerRegistry` 必须改用 `get_prefill_workers()` | — |
| `test_support.rs` | `RecordingPolicy.calls.hash_ring` 改为 `Option<Arc<HashRing>>`（支持 ptr_eq）；新增 `make_prefill_grpc` / `make_decode_grpc` | — |

测试覆盖 (72 个，全绿):
- A01-A18 (candidate filter)
- B01-B14 (policy apply)
- C01-C11 (regular planning)
- D01-D14 (PD planning)
- F01-F06 (trace)
- G01-G09 (error variants)

**当下 baseline** (P2 启动前必须保持):
- `cargo build --package atom-mesh` 干净
- `cargo test --package atom-mesh --lib core::placement::` → 82 passed (10 fixture smoke + 72 P1) / 22 failed (E12 + H10，P2/P3-P5 scope)
- `cargo clippy --package atom-mesh --lib --tests` → 3 pre-existing warning，placement 0 新 warning

**Subagent review 累计 5 轮**（A / B / F+C / D / 整体），全部解决。

**遗留进入 P2 跟踪**:
- §11.6 PD path api_key 转发：本期决定 A 路线（删字段 + tracking issue），P5 wire 阶段不顺手补
- `PlacementError::NoPrefill/NoDecodeWorkers` 仅 plan() 派遣层使用，post-health 一律返 NoAvailableWorkers（§11.7 + G07 spec），观察性可在 P2 引入区分
- `PlacementTrace` 的 `candidate_count_before/after` 在 Pair 模式合并 P+D 总数；如观察性需要拆分需扩 schema

### 5.3 P2 实际产出（已落地）

实现：

| 文件 | 改动 | 关键决策 |
|------|------|---------|
| `backend/sglang.rs` | `prepare_pair`(host+port) / `inject_prefill_fields` / `inject_decode_fields` (no-op + ctx type check) / `inject_batch_prefill_fields`；inline 1 行 `generate_room_id()`；3 个 `const KEY_*` 静态键避免 per-request 分配 | bootstrap_room 在 inject 调用时生成（不是 prepare_pair），与 reference pd_router.rs:402 一致 |
| `backend/vllm.rs` | `prepare_pair` 查 `bootstrap_addr` / `engine_id[dp_rank]`，typed `BootstrapAddrMissing` / `EngineIdMissing`；`inject_prefill_fields` 写 `kv_transfer_params` + force_prefill body 形状（stream=false / max_tokens=1 / max_completion_tokens=1 仅 if exists / 删 stream_options）；`inject_decode_fields` 写 decode 侧 kv_transfer_params 共享 transfer_id；`inject_batch_prefill_fields` 委托单请求并 `debug_assert_eq!(batch_size, 1)`（vLLM Mooncake 不批） | transfer_id 在 prepare_pair 生成一次、prefill+decode 共享（§11.2） |
| `tests_adapter.rs` | E01-E12 + 2 个 vLLM body-not-object 补测（review H2 反馈） | — |

测试覆盖 (14 个新增 — E01-E12 全绿 + 2 vllm body-not-object 补测):
- E01-E06 (SGLang)
- E07-E12 (vLLM)

**当下 baseline** (P3 启动前必须保持):
- `cargo build --package atom-mesh` 干净
- `cargo test --package atom-mesh --lib core::placement::` → 96 passed (10 fixture smoke + 72 P1 + 12 E + 2 补测) / 10 failed (H 类 P3-P5 scope)
- `cargo clippy --package atom-mesh --lib --tests` → 3 pre-existing warning（不动），placement 0 新 warning

**Subagent review 累计 1 轮**（rust-reviewer 0 BLOCK / 2 HIGH / 3 MEDIUM / 3 LOW；HIGH H2 已修，MEDIUM M1-M3 全采纳，LOW L1 折入 M3 修复，L2/L3 按 surgical-changes 原则不动）。

**遗留进入 P3 跟踪**:
- §11.6 PD path api_key 转发：本期决定 A 路线（删字段 + tracking issue），P5 wire 阶段不顺手补
- `PlacementError::NoPrefill/NoDecodeWorkers` 仅 plan() 派遣层使用，post-health 一律返 NoAvailableWorkers（§11.7 + G07 spec），观察性可在 P2 引入区分
- `PlacementTrace` 的 `candidate_count_before/after` 在 Pair 模式合并 P+D 总数；如观察性需要拆分需扩 schema

### 5.1 P0b 实际产出（已落地，后续 phase 直接消费）

新建目录 `atom/mesh/src/core/placement/`，文件清单：

| 文件 | 类别 | 状态 |
|------|------|------|
| `mod.rs` | 模块声明 | 完成 |
| `types.rs` | `RequestDescriptor` / `PlacementPlan` / `PlacementTrace` / `PlacementError`(6) / `AdapterError`(4 含 `CtxTypeMismatch`) / `Protocol` | 完成 |
| `traits.rs` | `WorkerSource` / `PolicySource`（含 `pd_needs_request_text` default）/ `PdPlanner` | 完成 |
| `candidate.rs` | `filter_candidates()` 签名锁定，body `todo!()` | 等 P1 实现 |
| `policy_apply.rs` | `apply_policy()` 返回 `Arc<dyn Worker>`（不是 idx），body `todo!()` | 等 P1 实现 |
| `planner.rs` | `DefaultPlanner` + `PdPlanner` impl，body `todo!()` | 等 P1 实现 |
| `trace.rs` | `PlacementTrace::for_single` / `for_pair`，body `todo!()` | 等 P1 实现 |
| `backend/mod.rs` | `BackendAdapter` object-safe trait + `pub type PairCtx = Box<dyn Any + Send + Sync>` | 完成 |
| `backend/sglang.rs` | `SglangAdapter` + `SglangPairCtx` + `downcast()` 助手；`inject_decode_fields` 已是 no-op `Ok(())` | 等 P2 实现其余方法 |
| `backend/vllm.rs` | `VllmAdapter` + `VllmPairCtx` + `VllmPrefillInfo` + `downcast()` 助手 | 等 P2 实现 |
| `test_support.rs` | `MockWorkerSource` / `MockPolicySource` / `RecordingPolicy` / `AlwaysNonePolicy` / `StaticNeedsTextPolicy` / 6 个 worker/descriptor 工厂 + 10 个 fixture smoke 测试（含 `backend_adapter_is_object_safe` 编译期断言） | 完成 |
| `tests_candidate.rs` | A01-A18 共 18 个 `todo!()` placeholder | 等 P1 转绿 |
| `tests_policy_apply.rs` | B01-B14 共 14 个 placeholder | 等 P1 转绿 |
| `tests_regular_planning.rs` | C01-C11 共 11 个 placeholder | 等 P1 转绿 |
| `tests_pd_planning.rs` | D01-D14 共 14 个 placeholder | 等 P1 转绿 |
| `tests_adapter.rs` | E01-E12 共 12 个 placeholder | 等 P2 转绿 |
| `tests_trace.rs` | F01-F06 共 6 个 placeholder | 等 P1 转绿 |
| `tests_error.rs` | G01-G09 共 9 个 placeholder | 等 P1/P2 转绿 |
| `tests_integration.rs` | H01-H10 共 10 个 placeholder | 等 P3-P5 转绿 |

**当下 baseline**（任何 P1+ 改动后必须保持或好于）:
- `cargo build --package atom-mesh` 干净
- `cargo test --package atom-mesh --lib core::placement::` → **94 failed + 10 passed**（94 placeholder 全 fail + 10 fixture smoke 全绿）
- `cargo clippy --package atom-mesh --lib --tests` → 0 placement 新 warning（pre-existing 3 个不动）

---

## 6. 工作清单（按目标架构组织）

### 6.1 新增 `core/placement/`

| 子模块 | 内容 | 来源 |
|--------|------|------|
| types.rs | RequestDescriptor, PlacementPlan, PlacementTrace, PlacementError | 全新 |
| traits.rs | WorkerSource, PolicySource | 全新 |
| candidate.rs | filter_candidates() | 整合 4 处分散的过滤逻辑 |
| policy_apply.rs | apply_policy() | 整合 HTTP `pick_worker_by_policy_arc` + gRPC stage 内联调用 |
| planner.rs | PdPlanner trait + DefaultPlanner | 整合 4 个 select 函数 |
| trace.rs | PlacementTrace 构造 | 全新 |
| backend/mod.rs | BackendAdapter trait | 全新 |
| backend/sglang.rs | inject_bootstrap_into_value 等 | 从 `http/pd_router.rs:348-406` 搬过来 |
| backend/vllm.rs | inject_kv_transfer_params + VllmPrefillInfo + fetch_vllm_prefill_info | 从 `http/pd_router.rs:431-449, 后续` 搬过来 |

### 6.2 Router 改造

| 文件 | 操作 |
|------|------|
| `routers/http/router.rs` → `routers/http_router.rs` | 上调 + 删除 `select_worker_for_model`，改调 planner |
| `routers/http/pd_router.rs` → `routers/http_pd_router.rs` | 上调 + 删除 `select_pd_pair`/`pick_worker_by_policy_arc`/`inject_*`，改调 planner + adapter；dispatch 拓扑（dual / fire-and-forget）原地保留 |
| `routers/http/pd_types.rs` | 删除（死代码） |
| `routers/grpc/common/stages/worker_selection.rs` | 删除内部 `select_single_worker` / `select_pd_pair`，stage 退化为 planner 调用包装 |
| `routers/grpc/router.rs` `routers/grpc/pd_router.rs` | 通过 stage 间接走 planner，自身改动小 |

### 6.3 共享 + 反向依赖修复

| 内容 | 来源 → 目标 |
|------|------------|
| 新增 `routers/shared/metrics_utils.rs` | 从 `grpc/utils.rs:1022-1041` 迁出 `route_to_endpoint`, `error_type_from_status` |
| `grpc/utils.rs` | 原位置改为 re-export，gRPC 内部 import 不变 |
| HTTP `use grpc::utils::{...}` | 改为 `use shared::metrics_utils::{...}` |

### 6.4 死代码删除

| 内容 | 位置 |
|------|------|
| `PDSelectionPolicy` enum | `pd_types.rs:40-57` |
| `RequestWithBootstrap` | `pd_types.rs:66-71` |
| `BatchRequestWithBootstrap` | `pd_types.rs:71-75` |
| `PDRouter.api_key` | `pd_router.rs:67/194` — 已确认全程未读，**直接删字段**；同时按 §11.6 决定是否顺手补 `worker.api_key()` 注入修真 bug |

### 6.5 配套 bug 修复

| Bug | 位置 | 修法 |
|-----|------|------|
| HTTP 不按 model_id 过滤（D2） | `http/router.rs:132`, `http/pd_router.rs:1199` | 走新 planner，自动按 model 过滤 |
| HTTP hash ring 写死 UNKNOWN_MODEL_ID | `http/router.rs:156`, `http/pd_router.rs:1207` | 走新 planner，用真实 model_id |
| HTTP PD 4xx/5xx 状态码吞掉 | `http/pd_router.rs:139-164, 979-1019, 1455-1481` 等 4 处 | 抽 `map_status_to_error()`，透传上游 status |
| HTTP PD headers retry 时被 clone | `pd_router.rs` 多处 | `Option<HeaderMap>` → `Option<Arc<HeaderMap>>` |

---

## 7. 不在范围

- 不重写 `WorkerRegistry` / `PolicyRegistry`（只抽 trait 给 mock，实现不动）
- 不引入 KV cost model
- 不引入新 placement 策略（pair-native 等）
- 不重构 gRPC pipeline 框架
- 不引入 per-model PD policy
- 不重构 HTTP PD dispatch 拓扑（dual / fire-and-forget 原地保留）
- 不动其他 `core/` 子模块
- 不引入 placement metrics surface（trace 仅 debug log）
- 不合并 4 个 router 为 DynamicRouter（后续独立工作，详见 `dynamic-router-plan.md`）

---

## 8. 验收标准

- `cargo test` 全绿
- `cargo clippy` 无新 warning
- `core/placement/` 单测覆盖 82-102 个 case，全绿
- HTTP router 不再 `use ...grpc::...`
- HTTP PD router 不再直接 `match self.backend`
- HTTP PD 状态码对 4xx/5xx 透传上游真实 status
- `pd_types.rs` 删除
- 4 个 router 各自的 select 函数删除，统一调 planner
- 文件扁平化完成

---

## 9. 风险

| 风险 | 缓解 |
|------|------|
| gRPC 0 单元测试 | placement 单测覆盖原 stage 内逻辑；切换时 gRPC stage 只剩"调 planner"，最小化变更面 |
| HTTP PD dispatch 复杂 | 不重构 dispatch，只换 placement 调用 + wire 注入 |
| 抽象设计错 → 推倒重做 | P0a 评审锁 type signature；最大代价是重做 P0 (~1.5 天) |
| 时间超预算 | TDD 路径下 cargo test 全绿即完成；review 轮次多则 2.5 周 |
| **gRPC PD 的 per-model policy 能力静默丢失**（§11.1） | P0a 文档 + release note 显式声明；如需保留则 D4.1 调整为β（per-model PD policy） |
| **vLLM adapter 启动期 I/O 失败时 PDRouter 无法构造**（§11.5） | adapter `init` 返回 Result，PDRouter 构造期向上抛 + 顶层启动逻辑明确 fail-fast 语义 |

---

## 10. 下一步

1. 评审本文档 + `router-refactor-proposal.md`（同事 + 老板）
2. 评审通过 → 启动 **P0a**（4-6 小时），P0a 必须解决 §11 中标 ⛔ 的项
3. P0a 产出 review 通过 → 启动 **P0b**（1 天）
4. P0b 测试骨架全红 → 进入 P1-P6 执行

---

## 11. P0a 待锁定事项（代码核查后补，2026-05-13）

> 下文每条都标注源文件:行号；查证后撰写。⛔ = P0a 必须落地决策；🟡 = P0a 建议落地，可推迟到对应 phase 启动前。

### 11.1 ⛔ gRPC PD 的"单 policy 同时跑 P 和 D + per-model"今天就是这样，D4 是行为变更

**证据**：`routers/grpc/common/stages/worker_selection.rs:228-245`
```rust
let policy = match model_id {
    Some(model) => self.policy_registry.get_policy_or_default(model),
    None => self.policy_registry.get_default_policy(),
};
// ...
let prefill_idx = policy.select_worker(&available_prefill, &info).await?;
let decode_idx  = policy.select_worker(&available_decode,  &info).await?;
```
HTTP PD 已经是分离（`routers/http/pd_router.rs:1186-1188`：`get_prefill_policy/get_decode_policy`），但 gRPC PD 不是。
切到 D4 + D4.1（α 全局分离）后，gRPC PD 同时丢两个能力：
1. P 和 D 用同一个 policy（routing 局部性可能依赖此）
2. per-model PD policy（gRPC 用户的现有 model→policy 配置在 PD 路径上失效）

**决策**：保持 D4 + D4.1（α），但 P0a 必须在文档里**显式记录 gRPC PD 的两点行为变更**，并加进 release note。如果用户/老板不接受，立刻把 D4.1 升到 β（per-model PD policy），本期对应工作量 +0.5 天。

### 11.2 ⛔ BackendAdapter trait sketch 的签名不够，P0a 必须收紧

**证据**：`routers/http/pd_router.rs:566-636`，vLLM 注入需要的 `bootstrap_addr`/`engine_id` 来自 `Arc<VllmPrefillInfo>` cache（pd_router.rs:46-58, 202-296），`transfer_id` 是 per-request `Uuid::new_v4()` 且 **prefill_kv 与 decode_kv 必须共用同一个**（pd_router.rs:611-625）。从 `(prefill: &dyn Worker, decode: &dyn Worker)` 两个引用拿不到任何一个。

**决策**：§3 已修订为 `prepare_pair(...) -> PairCtx` + `inject_*_fields(body, ctx)`。VllmAdapter 持有 `Arc<VllmPrefillInfo>`；`PairCtx` 内含 `bootstrap_addr / engine_id / transfer_id`。SglangAdapter 的 `PairCtx` 只携带 `bootstrap_room`（batch 时是 Vec）。P0a 锁这个形状。

### 11.3 ⛔ vLLM 注入还会改请求 shape，D3 描述要放宽

**证据**：`routers/http/pd_router.rs:431-449`，`force_prefill=true` 时强制 `stream=false`、`max_tokens=1`、`max_completion_tokens=1`（如果存在）、删除 `stream_options`。这超出"wire 字段注入"的字面承诺。

**决策**：D3 描述放宽为 **"wire 字段注入 + 协议必需的请求形状强制"**，由 `inject_prefill_fields` 顺带完成。不引入额外 hook。

### 11.4 🟡 `policies_need_request_text()` 的归宿

**证据**：`routers/http/pd_router.rs:1185-1189` 定义、`:1721/1751/1793` 三处使用。决定 retry loop 之前是否预提取 `request_text`，避免每次重试重复 tokenize。trait 方法在 `policies/mod.rs:62`，`cache_aware.rs:371` override。

**决策**：在 `PolicySource` trait 上加一个零成本默认方法 `fn pd_needs_request_text(&self) -> bool { self.prefill_policy().needs_request_text() || self.decode_policy().needs_request_text() }`；router 层在 retry loop 外查询一次。P1 实现 PolicySource 时一并落。

### 11.5 ⛔ vLLM adapter 的启动期 I/O 没规划

**证据**：`routers/http/pd_router.rs:183, 202-296`，`fetch_vllm_prefill_info` 在 `PDRouter::new()` 里同步去访问每个 prefill worker 的 `/query`，失败则 PDRouter 构造失败。

**决策**：`backend/vllm.rs` 暴露 `async fn init(worker_registry: &WorkerRegistry, client: &Client) -> Result<VllmAdapter, AdapterError>`。PDRouter 构造序列：① 读 backend 类型 → ② 若 vLLM 调 `VllmAdapter::init().await?` → ③ 把 `Arc<dyn BackendAdapter>` 注入 PDRouter。失败 fail-fast，与今天行为一致。

### 11.6 ⛔ `PDRouter.api_key` 决策必须从"或"改成单选

**证据**：`routers/http/pd_router.rs:67, 194` 写入；全文件无任何 `self.api_key` 读路径（grep 已确认）。非 PD 路径走 `worker.api_key()`：`routers/http/router.rs:365, 481` per-worker 注入。所以 PD 路径事实上**不向后端转发任何 auth header**。

**决策（推荐 A）**：本期只删字段；同时建一个跟踪 issue "PD path: forward worker-level api_key in Authorization header"。理由：保持本次 refactor 范围干净，不混入新行为。
**决策（备选 B）**：顺手在 P5 wire 注入阶段对齐 non-PD 行为，调 `worker.api_key()` 注入 Authorization。+0.25 天。
P0a 必须在 A/B 之间选定一个。

### 11.7 ⛔ `PlacementError::ModelNotFound` vs `NoWorkers` 的判别规则

**当前 §3 列了两个 variant 但未定义边界**。证据：`worker_registry.rs:466` 的 `get_workers_filtered` 不区分"该 model 没注册"与"全局没 worker"。

**决策**：P0a 锁定如下规则：
- `model_id.is_some()` 且 registry 内该 model 无任何 worker（不论健康） → `ModelNotFound { model_id }`
- `model_id.is_none()` 且 registry 整体为空 → `NoWorkers`
- candidate 集合非空但全 unhealthy → `NoAvailableWorkers`
- PD 拆出后 prefill/decode 任一为空 → `NoPrefillWorkers` / `NoDecodeWorkers`
- policy 返回 None → `PolicyReturnedNone`

### 11.8 🟡 D2 bug 实际是两层（candidate 过滤 + hash_ring keying）

**证据**：
- `routers/http/router.rs:132` `get_workers_filtered(None, ...)` ← 不按 model 过滤；`:156` `get_hash_ring(UNKNOWN_MODEL_ID)` ← hash_ring 也没按 model
- `routers/http/pd_router.rs:1199-1207` 同样两处错（`get_prefill_workers/get_decode_workers` 全局拉 + `get_hash_ring(UNKNOWN_MODEL_ID)`）

**决策**：§4 的 C 类 / D 类测试都拆成两组：
- C1/D1 "candidate 按 model 过滤"
- C2/D2 "hash_ring 按 model keyed"（覆盖 prefix-hash 在多 model 下不退化）

测试总数仍在 82-102 区间内，不另外加预算。

### 11.9 🟡 candidate 过滤要覆盖 `WorkerType::Prefill { bootstrap_port }` 维度

**证据**：`core/worker_registry.rs:377-388` 与 `WorkerType::Prefill { .. }` 的负载形状。SGLang 必须 `bootstrap_port = Some(_)`，vLLM Mooncake 不依赖该值。

**决策**：A 类（candidate 过滤）追加 1-2 个用例覆盖 prefill bootstrap_port 有无的两种场景；不影响测试总数上限。

### 11.10 🟡 §6.3 反向依赖修复要先扫一遍完整清单

**已扫描确认（grep 输出）**：HTTP → gRPC 的反向依赖只有 `route_to_endpoint` + `error_type_from_status` 两个符号：
- 来源：`routers/grpc/utils.rs:1022, 1033`
- HTTP 引用：`routers/http/router.rs:35, 192, 253, 315`、`routers/http/pd_router.rs:39, 463, 548, 734, 780, 855, 904`
- gRPC 内部消费：`routers/grpc/pipeline.rs:15, 177, 278`

清单封闭，迁去 `routers/shared/metrics_utils.rs` 后只需 (a) 在原位置 re-export 给 gRPC，(b) HTTP 改 import 路径。**无其他隐藏跨包符号**。

### 11.11 🟡 §5 测试计数与现实对齐

**实测**（`grep -c "#\[(tokio::)?test\]"`）：
- `routers/http/router.rs`: **15** 个测试（plan §5 P4 写"HTTP 现有 40 测试"应理解为 router 15 + pd 25 = 40）
- `routers/http/pd_router.rs`: **25** 个测试 ✓
- `routers/grpc/`: 共 **9** 个测试（utils.rs:6 + regular/responses/conversions.rs:3，全部是工具/转换层；plan §5 写"gRPC 0 测试"指的是 stage/router 层为 0，需在文档里明确以避免歧义）

**决策**：P4 完成标准改为"`routers/http/router.rs` 现有 15 个测试全绿"；P5 不变；§9 风险表里"gRPC 0 单元测试"加括号注 "(stage/router 层；utils 层有 9 个测试)"。

### 11.13 ⛔ BackendAdapter object-safety 锁定为 type-erased ctx

**证据**: §3 旧 sketch 用 `type PairCtx`（关联类型），导致 `BackendAdapter` 不是 object-safe，router 层无法 `Arc<dyn BackendAdapter>` 做运行时 backend 分发。Subagent review (P0b) 标 BLOCK。

**决策**: trait 改为 object-safe — `PairCtx = Box<dyn Any + Send + Sync>` 在 trait 层就是类型擦除的：
- `prepare_pair` 返回 `Box<dyn Any + Send + Sync>`（具体 adapter 内部 `Box::new(SglangPairCtx { .. })` 或 `Box::new(VllmPairCtx { .. })`）。
- `inject_*` 接 `&PairCtx`，方法体头一行 `ctx.downcast_ref::<XxxPairCtx>().ok_or(AdapterError::CtxTypeMismatch)?`。
- 新增 `AdapterError::CtxTypeMismatch` variant（理论不会触发，是防御 — 调用方持同一个 adapter 配同一个 ctx 不会失配；触发 = 调用方串了 adapter，5xx 直接打到日志）。

**为什么不选 enum BackendKind**: 那条路把"backend 分发"硬编码进枚举，加新 backend（如 TensorRT-LLM）要改 enum + 4 处 match。Object-safe trait 加新 backend 只需新增 impl，零侵入。

**工时**: P2 +0.5 天（每个 adapter 1 行 downcast 助手 + 多处 cast；新增 1 个 error variant；plan §3 sketch 更新）。

### 11.12 🟡 HTTP PD 现有 9 处直接调用 `select_pd_pair` 的测试需迁移

**证据**（grep）：`routers/http/pd_router.rs` 内 `select_pd_pair(` 出现 9 次（含函数定义 + 8 个测试调用 / 业务调用），`select_worker_for_model(` 在 `routers/http/router.rs` 出现 4 次。

**决策**：P5 完成标准追加 "`select_pd_pair` 直接调用的旧测试已迁移到 planner 层（或删除并以 placement core 单测覆盖）"；P4 同样追加 `select_worker_for_model`。工时不另加（包含在 P4/P5 各 1-2 天里）。

---

## 12. P1-P6 开发执行流程（每个 phase 都按此走）

### 12.1 TDD 循环（一条测试一个循环）

1. **挑下一条 placeholder 测试**（按 ID 顺序：A01→A02→…）
2. **Read 测试 ID 对应的子项**——回到 `router-refactor-tests.md` 看完整描述（plan 的真理来源就在那里，不要从代码 todo!() 反推）
3. **写最小测试体**：构造 mock fixture → 调用 production 函数 → assert。**先看测试 fail**，确认 fail 原因符合预期（typed error / 错误数值），不是 build 错误或 mock 问题
4. **写最小 production 改动让单条测试转绿**——不顺手实现下一条
5. **再跑一次 cargo test 这条**：必须绿
6. **本地三件套**：`cargo build` + `cargo test --lib core::placement::` + `cargo clippy --lib --tests`
7. **进入下一条**

每完成 8-10 条或一个测试类别（A/B/C/...），进入 §12.3 subagent review。

### 12.2 跑命令（在 host 编辑，container 里跑）

```bash
# 编译
docker exec -w /it-share/yajizhan/code/ATOM/atom/mesh atom_sglang_mesh \
  cargo build --package atom-mesh

# 跑全部 placement 测试（看哪些转绿了）
docker exec -w /it-share/yajizhan/code/ATOM/atom/mesh atom_sglang_mesh \
  cargo test --package atom-mesh --lib core::placement::

# 跑单条
docker exec -w /it-share/yajizhan/code/ATOM/atom/mesh atom_sglang_mesh \
  cargo test --package atom-mesh --lib core::placement::tests_candidate::a01_

# Clippy（必须 0 placement 新 warning）
docker exec -w /it-share/yajizhan/code/ATOM/atom/mesh atom_sglang_mesh \
  cargo clippy --package atom-mesh --lib --tests
```

⚠️ **绝对不在 container 内编辑文件**——root ownership 会破坏 host 用户的写权限。

### 12.3 Subagent review（每个 phase 收尾 + 任何超过 ~50 行的改动）

**时机**:
- 每完成一个测试类别（A 全绿 / B 全绿 / ...）
- 任何 phase（P1/P2/...）声称完成之前
- 任何超过 50 行的 production 改动（不算 test placeholder）

**调用模板**（直接复用，按需替换 `{...}`）:

```
Agent({
  description: "Review {phase} placement {category} implementation",
  subagent_type: "rust-reviewer",
  prompt: """
Quick review of {phase} changes. Files changed:

1. /it-share/yajizhan/code/ATOM/atom/mesh/src/core/placement/{file1.rs} — {what changed}
2. ...

**Context**: This is {phase} of the router refactor (see .claude/plans/router-refactor-plan.md).
Previous review feedback already applied: no meta comments (no plan refs / phase labels /
"P1 will implement" / coverage notes), object-safe BackendAdapter (§11.13), apply_policy
returns Arc<dyn Worker>.

**Validate specifically**:
1. {phase-specific concern 1}
2. {phase-specific concern 2}
3. Any non-obvious bug, broken invariant, or signature that won't survive next phase?
4. Any new comment that's a meta / plan ref / phase label that should be deleted?

Report under 300 words.
"""
})
```

**应用结果**:
- BLOCK 级别：必须修，再跑一轮 review，直到无 BLOCK
- HIGH 级别：默认修；如有理由不修，在本次 commit message 里写明
- MEDIUM/LOW：按 surgical-changes 原则评估 — 与本 phase scope 强相关才动

### 12.4 注释红线（与 §编码原则呼应）

**禁止写进 .rs 源码**（违反则 review 必删）:
- ❌ `// see plan §X` / `// 覆盖 §A`
- ❌ `// P1 will implement` / `// TODO P2: ...` / `// will be removed in P6`
- ❌ `// from pd_router.rs:1199` / `// 此函数即将被替换`
- ❌ 文件顶部 `//!` 说"Locked decisions"、"Per plan §11.X"、"This module will eventually..."
- ❌ 整段设计 rationale 段落（去 plan / PR description / commit message）
- ❌ 测试函数上的 `(D2 fix evidence)` / `(plan §11.7)` 类括号注

**允许保留**（WHY 非显然时一行）:
- ✅ 隐藏约束（"Tonic 0.14 requires this header to be lowercase"）
- ✅ 不变量（"caller holds the lock when calling this"）
- ✅ Workaround（"upstream bug X / linked issue: ..."）
- ✅ no-op 契约一行说明（如 `/// No-op: SGLang dual-dispatch does not inject on the decode side.`）

**自检方法**: 写完一个 phase 后，`grep -nE 'P[0-9]|plan §|TODO|will be|D[0-9] fix|covers' atom/mesh/src/core/placement/**/*.rs` 应该返回空（或只剩 `dp_rank` 之类误匹配）。

### 12.5 每个 phase 启动前的 checklist

- [ ] Read `router-refactor-plan.md` 对应 phase 的"完成标准"列
- [ ] Read `router-refactor-tests.md` 对应类别的全部子项（不要凭记忆）
- [ ] 跑当前 baseline (`cargo test core::placement::`) 记录"现在多少 passed/failed"
- [ ] 进入 §12.1 TDD 循环
- [ ] 收尾按 §12.3 跑 subagent review
- [ ] 跑最后三件套，passed 数应严格大于 baseline，failed 严格小于
- [ ] 在本 plan §5 把当前 phase 改 ✅ done，写一行实际产出（仿照 §5.1）

---

### §11 速查清单

| § | 类型 | 决策必须落地于 | 阻塞 |
|---|------|---------------|------|
| 11.1 | ⛔ 行为变更披露 | P0a 文档 + release note | gRPC 用户兼容性 |
| 11.2 | ⛔ trait 形状 | P0a §3 | 已落 |
| 11.3 | ⛔ D3 描述放宽 | P0a §1 决策板 | P2 实现 |
| 11.4 | 🟡 PolicySource API | P1 启动前 | P5 retry loop |
| 11.5 | ⛔ adapter init 序列 | P0a 文档 | P2 实现 |
| 11.6 | ⛔ api_key A/B 选边 | P0a 文档 | P5 wire 注入 |
| 11.7 | ⛔ Error variant 边界 | P0a §3 + §4 G 类 | P0b 测试骨架 |
| 11.8 | 🟡 测试拆分 | P0b 启动前 | 测试覆盖 |
| 11.9 | 🟡 测试维度 | P0b 启动前 | 测试覆盖 |
| 11.10 | 🟡 已确认 | — | — |
| 11.11 | 🟡 文档计数订正 | P0a 文档 | — |
| 11.12 | 🟡 完成标准追加 | P4/P5 启动前 | — |
| 11.13 | ⛔ BackendAdapter object-safe | P0b 已落（§3 sketch + AdapterError::CtxTypeMismatch + 2 个 adapter downcast） | P2 实现 |
