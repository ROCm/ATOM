# `ProcessingState` + `PipelineStage` 过度设计分析

**日期**: 2026-05-19
**聚焦**: `atom/mesh/src/routers/grpc/` 中 `ProcessingState`（god-bag）+ `Vec<Box<dyn PipelineStage>>`（运行时 stage 分发）的设计问题

---

## 1. 论点

当前 pipeline 设计用了一个**可扩展框架**来解决一个**线性 4 步流程**的问题。框架的成本（动态分发、运行时状态校验、18 个文件 1469 行 stage 代码）没有对应的收益——整个代码库只存在两种 pipeline 配置（regular/PD），且它们之间只差两个参数。

这不是一个风格偏好。`ProcessingState` + `PipelineStage` 引入了三类可量化的工程负债：编译器无法保护的运行时失败路径、可审计性下降、以及对未来重构的阻力。

---

## 2. 问题一：`ProcessingState` 用 `Option<T>` 模拟了编译器本该保证的状态

### 2.1 现状

```rust
// context.rs
#[derive(Default)]
pub(crate) struct ProcessingState {
    pub preparation: Option<PreparationOutput>,    // Stage 1 填
    pub tokenizer: Option<Arc<dyn Tokenizer>>,     // Stage 1 填
    pub workers: Option<WorkerSelection>,          // Stage 2 填
    pub clients: Option<ClientSelection>,          // Stage 3 填
    pub proto_request: Option<ProtoRequest>,       // Stage 4 填
    pub dispatch: Option<DispatchMetadata>,         // Stage 5 填
    pub load_guards: Option<LoadGuards>,           // Stage 6 填
    pub response: ResponseState,                   // Stage 7 填
}
```

所有字段初始为 `None`，由各个 stage 逐步填充。

### 2.2 运行时 unwrap 是唯一安全网

每个 stage 在访问前一个 stage 的输出时，必须 `ok_or_else` 或 `.unwrap()`：

```rust
// dispatch_metadata.rs:24 — Stage 5 读 Stage 4 的输出
let proto_request = ctx.state.proto_request.as_ref().ok_or_else(|| {
    error!("Proto request not built");
    error::internal_error("proto_request_not_built", "Proto request not built")
})?;

// request_execution.rs:48 — Stage 6 读 Stage 3 的输出
let clients = ctx.state.clients.as_mut().ok_or_else(|| {
    error!("Client acquisition not completed");
    error::internal_error("client_acquisition_not_completed", ...)
})?;

// request_execution.rs:61 — Stage 6 读 Stage 2 的输出
let workers = ctx.state.workers.as_ref().ok_or_else(|| {
    error!("Worker selection not completed");
    error::internal_error("worker_selection_not_completed", ...)
})?;
```

仅 `request_execution.rs` 一个文件就有 **3 处** runtime 检查来验证前置 stage 已运行。这些检查只有在 stage 顺序被错误调整时才触发——而 stage 顺序在编译期就已经固定了。

### 2.3 编译器本可以保证这一点

显式函数调用天然地通过值传递保证顺序：

```rust
let (payload, resp_ctx) = prepare_chat(req, ...)?;       // 不存在 "prepare 没跑" 的可能
let placement = planner.plan(&payload.descriptor())?;      // payload 是上一行的返回值
let stream = engine.dispatch(&placement, &payload)?;       // placement 和 payload 都有值
render::chat_aggregator::process(stream, resp_ctx).await   // stream 和 resp_ctx 都有值
```

没有 `Option`，没有 `unwrap`，没有"stage 没运行"的运行时错误路径。**编译器拒绝编译顺序错误的代码**，而不是在运行时 panic。

### 2.4 量化对比

| 指标 | `ProcessingState` | 显式函数调用 |
|---|---|---|
| 运行时状态校验（`ok_or_else` / `unwrap`） | **≥ 8 处**（每个 stage 至少 1 处） | **0 处** |
| "stage 未完成"错误变体 | 8 种字符串（`proto_request_not_built` 等） | 不存在 |
| 编译器能否检测 stage 顺序错误 | 否 | 是 |
| 加一个 stage 间数据依赖的方式 | 给 `ProcessingState` 加 `Option<T>` 字段 + 加 `ok_or_else` 检查 | 在函数参数上加类型 |

---

## 3. 问题二：`PipelineStage` trait 是 single-impl indirection

### 3.1 trait 定义

```rust
// common/stages/mod.rs
#[async_trait]
pub trait PipelineStage: Send + Sync {
    async fn execute(&self, ctx: &mut RequestContext) -> Result<Option<Response>, Response>;
    fn name(&self) -> &'static str;
}
```

### 3.2 实际使用：13 个 impl，但只有 2 种固定组合

代码库中有 13 个 `impl PipelineStage`，分布在 18 个文件、1469 行代码中。但这些 impl **从未被动态组合**——整个代码库只存在两种 pipeline 配置：

```rust
// pipeline.rs:78-86 — Regular
let stages: Vec<Box<dyn PipelineStage>> = vec![
    Box::new(PreparationStage::new()),
    Box::new(WorkerSelectionStage::new(planner)),
    Box::new(ClientAcquisitionStage),
    Box::new(RequestBuildingStage::new(false)),    // ← 唯一差异 1
    Box::new(DispatchMetadataStage),
    Box::new(RequestExecutionStage::new(ExecutionMode::Single)),  // ← 唯一差异 2
    Box::new(ResponseProcessingStage::new(processor, streaming_processor)),
];

// pipeline.rs:123-131 — PD
let stages: Vec<Box<dyn PipelineStage>> = vec![
    Box::new(PreparationStage::new()),
    Box::new(WorkerSelectionStage::new(planner)),
    Box::new(ClientAcquisitionStage),
    Box::new(RequestBuildingStage::new(true)),      // ← 差异 1: true vs false
    Box::new(DispatchMetadataStage),
    Box::new(RequestExecutionStage::new(ExecutionMode::DualDispatch)),  // ← 差异 2
    Box::new(ResponseProcessingStage::new(processor, streaming_processor)),
];
```

两种配置之间的差异：**两个布尔参数**。其余 5 个 stage 完全相同。

### 3.3 额外的 wrapper stage 层

部分 stage 本身只是 dispatcher，内部再 `match RequestType` 分发到 chat/generate 子 stage：

```rust
// regular/stages/preparation.rs — 整个文件的唯一作用是 match 分发
impl PipelineStage for PreparationStage {
    async fn execute(&self, ctx: &mut RequestContext) -> Result<Option<Response>, Response> {
        match &ctx.input.request_type {
            RequestType::Chat(_) => self.chat_stage.execute(ctx).await,
            RequestType::Generate(_) => self.generate_stage.execute(ctx).await,
        }
    }
}
```

同样的 wrapper 模式在 `RequestBuildingStage` 和 `ResponseProcessingStage` 中重复出现——3 个 wrapper stage，每个只做 `match + delegate`，共 132 行代码。在显式 pipeline 中，这个分支就是 `Pipeline::execute_chat` vs `Pipeline::execute_generate` 两个方法，wrapper 层完全不需要。

### 3.4 dynamic dispatch 的成本

`Vec<Box<dyn PipelineStage>>` 引入的不只是 vtable 调用开销（在 IO-bound 的 LLM 推理场景下可忽略），更重要的是：

- **可读性成本**：要理解 pipeline 流程，必须跳转 7 个文件，而不是读一个函数体
- **重构摩擦**：想改 stage 顺序或合并两个 stage，需要考虑 trait 签名兼容性
- **错误处理的三态返回值**：`Result<Option<Response>, Response>` 区分"继续"/"正常完成"/"错误"三种状态，仅仅因为 for 循环需要这个协议。显式函数调用中，`?` 操作符天然处理错误，正常路径就是下一行代码

### 3.5 框架适用场景 vs 当前场景

| 特征 | 框架适用 | 当前代码 |
|---|---|---|
| Pipeline 配置数量 | 多种（≥ 3），运行时决定 | 2 种，编译期固定 |
| Stage 是否可独立增删 | 是，stage 集合运行时可变 | 否，一年来始终是同样 7 个 stage |
| Chat/Generate 是否需要不同 stage 序列 | 是，序列本身不同 | 否，序列相同，差异在 stage 内部的 match 分支 |
| Stage 间数据依赖 | 非线性（DAG 或黑板） | 严格线性（1→2→3→...→7） |

当所有四个条件都不满足时，`Vec<Box<dyn PipelineStage>>` 是 indirection without polymorphism。

---

## 4. 问题三：黑板模式的灵活性在线性流中没有被使用

### 4.1 数据流图

追踪 `ProcessingState` 中每个字段的写入者和读取者：

| 字段 | 写入 stage | 读取 stage | 读写模式 |
|---|---|---|---|
| `preparation` | 1 (Preparation) | 4 (RequestBuilding) | 一写一读 |
| `tokenizer` | 1 (Preparation) | 7 (ResponseProcessing) | 一写一读 |
| `workers` | 2 (WorkerSelection) | 3 (ClientAcquisition), 6 (Execution) | 一写两读 |
| `clients` | 3 (ClientAcquisition) | 6 (Execution) | 一写一读 |
| `proto_request` | 4 (RequestBuilding) | 5 (DispatchMetadata), 6 (Execution) | 一写两读 |
| `dispatch` | 5 (DispatchMetadata) | 6 (Execution), 7 (Response) | 一写两读 |
| `load_guards` | 6 (Execution) | Drop 时自动释放 | 一写零读 |
| `execution_result` | 6 (Execution) | 7 (ResponseProcessing) | 一写一读 |

**全部都是 "一写 N 读" 的线性传递**。没有出现：
- 多个 stage 写同一字段
- 条件性跳过某 stage 后由后续 stage 补填
- 运行时动态决定哪些 stage 读哪些字段

这正是函数调用链的 `let a = f(); let b = g(a); let c = h(a, b);` 能精确表达的模式。黑板模式（shared mutable state）在这里引入了读写不透明性，却没有使用到任何非线性数据流的能力。

### 4.2 黑板的真实代价

因为所有 stage 共享 `&mut RequestContext`，编译器无法约束：
- Stage 3 只应该读 `workers` 和写 `clients`，但它**技术上**可以读写任何字段
- 一个 stage 意外覆盖另一个 stage 的输出不会有编译警告
- Code review 是唯一防线

显式值传递中，每个函数只看到它参数里的数据，不可能意外修改不属于它的状态。

---

## 5. Pipeline 模式的真实优点——以及为什么在当前场景下不成立

Pipeline + 黑板不是一个坏模式。它在特定条件下有四个真实优点，这里逐一承认，然后用代码证据解释为什么当前代码没有满足这些条件。

### 5.1 优点：Stage 隔离——每个 stage 可独立开发、独立 review

Pipeline 模式将流程拆分为独立的 struct，每个 stage 有自己的文件、自己的 `impl`。新人加一个 stage 不需要读完整条流水线，只需要知道"我从 `ctx.state` 读什么、写什么"。Code review 时也可以只看一个 stage 的 diff。

**为什么在当前场景下不成立**：

这种隔离是**形式上的**——所有 stage 共享 `&mut RequestContext`，改 `PreparationOutput` 加一个字段，下游的 `RequestBuildingStage`、`ResponseProcessingStage` 都可能受影响。你不能只 review 一个 stage 的 diff 而不检查上下游。

新方案的 `prepare/`、`engine/`、`render/` 三个模块同样可以独立开发和 review，且隔离更彻底：它们之间**只通过函数签名的类型通信**，编译器强制禁止跨模块直接访问内部状态。改 `GenerationPayload` 加一个字段，只影响 `prepare/`（生产者）和 `engine/`（消费者），`render/` 完全无感——因为 `render/` 的函数签名里根本没有 `GenerationPayload`。

### 5.2 优点：横切关注点——for 循环天然支持 stage 级 metrics / tracing / 超时

Pipeline 的 for 循环是一个天然的 AOP 拦截点：

```rust
for stage in self.stages.iter() {
    let start = Instant::now();
    let result = stage.execute(&mut ctx).await;
    metrics::record_stage_latency(stage.name(), start.elapsed());  // 一行搞定所有 stage
    // ...
}
```

加 stage 级 tracing、超时、或统一错误处理，只需要在循环体里加几行，不需要改每个 stage。

**为什么在当前场景下不成立**：

当前代码的 for 循环里**并没有实现 stage 级 metrics**。实际只有两处 metrics：

- 整条请求的总延迟（`pipeline.rs:168-176`，在 stage 循环外）
- 错误时的 `stage.name()`（仅用于 error log，不是 histogram）

也就是说，这个优点是**理论上的**——pipeline 模式提供了这个能力，但当前代码没有使用。如果未来确实需要 stage 级 metrics，显式函数调用中加 tracing span 同样简单且更精确：

```rust
let (payload, resp_ctx) = {
    let _span = info_span!("prepare").entered();
    prepare::prepare_chat(req, ...)?
};
```

### 5.3 优点：配置式组合——改参数而非改代码切换 pipeline 行为

Regular 和 PD 的差异通过构造参数表达，不需要 if/else 分支：

```rust
// Regular: 传 false + Single
Box::new(RequestBuildingStage::new(false)),
Box::new(RequestExecutionStage::new(ExecutionMode::Single)),

// PD: 传 true + DualDispatch
Box::new(RequestBuildingStage::new(true)),
Box::new(RequestExecutionStage::new(ExecutionMode::DualDispatch)),
```

如果将来出现第三种模式（如 speculative decoding），只需创建新的 stage 实例。

**为什么在当前场景下不成立**：

这个优点的前提是"pipeline 配置数量会增长"。从代码引入至今（一年），配置数量始终是 2，差异是 2 个参数。7 个 stage 中有 5 个完全相同——**这不是组合，是复制**。

在显式函数调用中，Regular vs PD 的差异变成 `engine.dispatch()` 内部的一个 `match PlacementPlan`——一个分支，而非两套 `vec![]` 构造。如果第三种模式出现，加一个 `match` arm 的成本和加一套 stage 配置相当，但不需要维护 trait 框架。

更重要的是：**从显式调用提取 trait 是机械操作**（提取函数签名为 trait method）；**从 trait 框架回退到显式调用需要拆除整个框架**。选择简单方案的回退成本更低。

### 5.4 优点：黑板灵活性——跨 stage 加数据流零成本

黑板模式下，stage A 想给 stage D 传值：

```rust
// 1. ProcessingState 加一个字段
pub some_new_field: Option<SomeType>,

// 2. Stage A 写
ctx.state.some_new_field = Some(value);

// 3. Stage D 读
let v = ctx.state.some_new_field.as_ref().unwrap();
```

不需要改中间 stage B、C 的签名。成本是一个字段 + 一个 unwrap。

**为什么在当前场景下不成立**：

追踪实际数据流（见 §4.1 表），**所有 8 个字段都是"一写 N 读"的线性传递**，没有出现：

- 跨越多个 stage 的非顺序依赖（如 stage 1 写、stage 3 跳过、stage 5 补填）
- 多个 stage 写同一字段
- 运行时动态决定读哪些字段

在线性流中，函数调用的值传递 `let a = f(); let b = g(a);` 同样零成本传递数据，且不引入 `Option` unwrap 风险。"跨 stage 加数据流"的灵活性在一年中没有被使用过。

### 5.5 小结

| 优点 | 成立前提 | 当前代码是否满足 |
|---|---|---|
| Stage 隔离 | Stage 之间无共享状态依赖 | 否——全部共享 `&mut RequestContext` |
| 横切关注点 | 需要 stage 级 metrics/tracing/超时 | 否——当前只有请求级总延迟 |
| 配置式组合 | ≥ 3 种配置，stage 集合运行时可变 | 否——2 种固定配置，差异 2 个参数 |
| 黑板灵活性 | 非线性数据依赖，多 stage 交叉读写 | 否——全部线性一写 N 读 |

**四个优点的前提条件全部不成立。** 这不是说 pipeline 模式本身不好——在配置多变、stage 可动态增删、数据依赖复杂的场景中（如 CI/CD pipeline 引擎、消息处理中间件），它是正确选择。但当前代码是一个 2 配置、7 固定 stage、线性数据流的场景，为不存在的变化轴付出了框架成本。

---

## 6. 预期挑战与回应

以下挑战与 §5 的优点分析有部分重叠，但侧重不同：§5 是从设计角度分析优点是否成立，§6 是从 review 对话角度准备回应。

### 挑战 1："Stage pipeline 支持独立开发和测试"

**回应**：新方案的职责域模块（`prepare/`、`engine/`、`render/`）同样支持独立开发，且解耦更彻底：

- Stage pipeline 的独立性是**假的**——所有 stage 都依赖 `&mut RequestContext`，改 `PreparationOutput` 的字段，下游 stage 全部受影响
- 新方案的模块间**只通过类型签名通信**（`GenerationPayload`、`TokenChunk`），且这种隔离是**编译器强制的**（`prepare/` 和 `render/` 禁止 import `mesh_grpc::*`）

### 挑战 2："Pipeline 方便加 stage 级 metrics/tracing"

**回应**：

- 当前代码中 for 循环里的 metrics **实际上没有 stage 级粒度**——只有 `stage.name()` 出现在 error log 中，没有 per-stage latency histogram
- 显式函数调用中加 tracing span 同样简单：`let _span = info_span!("prepare").entered();`，且更精确（span 范围与函数生命周期一致，而非 stage 的 `execute` 返回时才结束）

### 挑战 3："未来可能需要更多 pipeline 配置"

**回应**：

- 从代码引入至今，pipeline 配置数量始终是 2（regular + PD），且没有出现第三种的迹象
- 即使未来需要第三种，从显式函数调用提取 trait 是机械操作（提取函数签名为 trait method）；反过来，从 trait pipeline 回退到显式调用则需要删除整个框架
- YAGNI 原则：为尚不存在的需求付出已经确定的复杂度代价，是过度设计的定义

### 挑战 4："黑板模式方便加跨 stage 数据流"

**回应**：

- 跨 stage 加一条数据流的场景在一年内没有出现过
- 当前所有数据流都是线性的（见 §4.1 表），这意味着函数参数就能精确表达
- 如果确实出现了非线性数据流需求，可以把相关值加到 `GenerationPayload` 或 `ResponseContext` 的字段上——成本是改一个 struct 定义 + 改一个构造点，与 `ProcessingState` 加 `Option<T>` 的成本相当，但不引入 unwrap 风险

### 挑战 5："这只是风格偏好"

**回应**：这是一个可量化的工程决策，不是风格问题：

| 维度 | Stage pipeline | 显式函数调用 |
|---|---|---|
| Stage 相关文件数 | 18 个文件 | 0（pipeline 是一个函数体） |
| Stage 相关代码行数 | 1469 行 | 0 |
| 运行时状态校验 | ≥ 8 处 `ok_or_else` | 0 |
| 编译器可检测的错误类 | stage 顺序错误 → **运行时才发现** | stage 顺序错误 → **编译失败** |
| 阅读完整 pipeline 需跳转的文件 | 7+ | 1 |

---

## 7. 为什么必须改——不改会持续产生的问题

这不是"技术债但能忍"的情况。`ProcessingState` + `PipelineStage` 是本次 gRPC 解耦重构的**阻塞依赖**，不改它，后续的解耦目标无法实现。

### 7.1 阻塞解耦：transport-neutral 代码无法从 `grpc/` 中抽离

重构的核心目标是把 `prepare/`（请求构造）和 `render/`（响应渲染）从 `grpc/` 中抽出，使其不依赖 `mesh_grpc::*`。但当前架构下：

- 所有 stage 都通过 `&mut RequestContext` 传递数据
- `RequestContext` 包含 `ProcessingState`
- `ProcessingState` 包含 `ProtoRequest`（`mesh_grpc` 类型）和 `ExecutionResult`（包含 `ProtoStream`，也是 `mesh_grpc` 类型）

也就是说，**每一个 stage 都被迫依赖 `mesh_grpc::*`**——即使它的业务逻辑（如 chat template 解析、SSE 编码）与 gRPC 完全无关。只要 `ProcessingState` 存在，transport-neutral 代码就无法脱离 `grpc/` 的 import 链。

把 stage 搬到 `grpc/` 外面但保留 `ProcessingState`？不行——`ProcessingState` 自身就 import 了 `ProtoRequest` 和 `ProtoStream`，搬出去的 stage 仍然通过 `&mut RequestContext` 间接依赖 `mesh_grpc::*`。

### 7.2 阻塞可测试性：协议逻辑无法脱离 tonic 单测

目标是 `prepare/` 和 `render/` 可以用纯 Rust 单元测试验证，不需要 tonic server、proto 编解码、或 worker mock。但：

- `PreparationStage` 的输出写入 `ctx.state.preparation`（`Option<PreparationOutput>`），而不是作为函数返回值
- 要测试 preparation 逻辑，必须构造一个完整的 `RequestContext`（包含 `SharedComponents`、`ProcessingState`），然后调用 `stage.execute(&mut ctx)`，再从 `ctx.state.preparation` 中取出结果
- `RequestContext` 通过 `ProcessingState` 依赖 `ProtoRequest` → 依赖 `mesh_grpc::*` → 测试需要 proto 编译环境

显式函数调用中，`prepare_chat()` 返回 `(GenerationPayload, ResponseContext)`——测试只需要构造输入参数，断言返回值，无需 `RequestContext` 或 proto 依赖。

### 7.3 持续的维护成本

即使不做解耦重构，保留当前设计也有持续成本：

| 场景 | 当前成本 |
|---|---|
| **理解 pipeline 流程** | 跳转 7+ 个文件（stage 定义）+ 1 个文件（pipeline 构造）+ 1 个文件（context 定义），而非读一个函数体 |
| **追踪一个字段的生命周期** | 在 `ProcessingState` 找字段 → grep 写入 stage → grep 读取 stage → 确认中间 stage 没有意外覆盖 |
| **加一个新的请求参数** | 给 `PreparationOutput` 加字段 → 在 Preparation stage 填充 → 在 RequestBuilding stage 读取 → 处理 `Option` unwrap → 可能需要改 wrapper stage 的分发逻辑 |
| **Debug "field not set" 运行时错误** | 回溯 stage 执行顺序，确认哪个 stage 没跑或跑的顺序不对——一类在显式调用中根本不存在的 bug |

### 7.4 对新人的额外认知负担

新人阅读 pipeline 时需要理解：
1. `PipelineStage` trait 及其三态返回值 `Result<Option<Response>, Response>`
2. `ProcessingState` 的 8 个 `Option` 字段分别由哪个 stage 填充
3. Wrapper stage 的 `match RequestType` 分发模式（3 处重复）
4. Regular vs PD pipeline 的差异在哪里（需要 diff 两段几乎相同的 `vec![]` 构造）

这些都是**框架知识**，不是**业务知识**。显式函数调用中，新人读一个 `execute_chat` 函数体就能看到完整流程，不需要先学框架。

---

## 8. 结论

`ProcessingState` + `PipelineStage` 解决了一个不存在的问题（动态可组合的 pipeline 配置），同时引入了一个真实的问题（运行时状态校验替代编译器保证）。更关键的是，它是本次 gRPC 解耦重构的**阻塞点**——不拆掉 `ProcessingState`，`mesh_grpc::*` 就无法被隔离到 `engine/` 内，transport-neutral 代码就无法独立单测。

重构为 4 步显式函数调用不丢失任何能力，同时消除所有 `Option` unwrap 路径，将 1469 行 stage 框架代码归零。

如果 pipeline 配置确实需要动态组合（≥ 3 种配置、stage 集合运行时可变、非线性数据依赖），那 `PipelineStage` trait 是正确设计。但当前代码的证据不支持这些前提中的任何一条。
