# gRPC Router 重构方案 — Review 版

**日期**: 2026-05-19  
**范围**: `atom/mesh/src/routers/grpc/` (9280 行 Rust 代码)

---

## 1. 为什么要重构

`routers/grpc/` 目录包含 9280 行代码，但**真正与 gRPC 传输相关的只有约 1100 行（~15%）**。其余代码——协议解析、SSE 编码、stop 解码、工具/推理 parser、响应渲染、OpenAI Responses API——与 gRPC 无关，却因为历史原因被放在了 `grpc/` 下。

根本原因是**耦合方向错误**：transport-neutral 的代码直接 import 了 `mesh_grpc::sglang_proto::*` 类型（哪怕只在函数签名层面），导致无法从 `grpc/` 中抽离。

---

## 2. 当前方案的问题

### 2.1 import 耦合：transport-neutral 逻辑被 gRPC 类型"钉死"

`streaming.rs`（1326 行）和 `processor.rs`（465 行）的核心逻辑是 SSE 编码和响应组装，不涉及 gRPC 语义。但它们的函数签名直接接收 proto stream 类型，导致：
- 无法被未来的 HTTP-native 后端复用
- 无法在单元测试中脱离 tonic/proto 依赖独立测试

### 2.2 上帝对象 `ProcessingState`：`Option<T>` 组成的状态袋

```rust
// context.rs — 当前
pub(crate) struct ProcessingState {
    pub preparation: Option<PreparationOutput>,
    pub tokenizer: Option<Arc<dyn Tokenizer>>,
    pub workers: Option<WorkerSelection>,
    pub clients: Option<ClientSelection>,
    pub proto_request: Option<ProtoRequest>,
    pub dispatch: Option<DispatchMetadata>,
    pub execution: Option<ExecutionResult>,
    pub response: Option<FinalResponse>,
}
```

每个字段在特定阶段被填入，但编译器无法保证访问顺序——runtime unwrap 是唯一的安全网。

### 2.3 流水线过度抽象：运行时 stage 分发

```rust
// pipeline.rs — 当前
pub(crate) struct RequestPipeline {
    stages: Arc<Vec<Box<dyn PipelineStage>>>,
}
```

7 个 stage 通过 `Vec<Box<dyn PipelineStage>>` 运行时索引执行。Chat 和 Generate 两条路径在同一个循环内通过 `match RequestType` 分叉。实际上只有一种 pipeline 实现，trait 动态分发是 single-impl indirection。

### 2.4 `utils.rs`：1214 行万能工具箱

chat template、tool constraints、stop decoder、parser factory lookup、finish reason mapping、gRPC client 获取、logprob proto 适配器……全部混在一个文件里，职责完全看不出来。

### 2.5 OpenAI Responses API 位置错误

`grpc/{common,regular}/responses/`（2059 行）**没有任何 `mesh_grpc::*` import，也没有 PD 模式代码**（PD router 对 `/v1/responses` 直接返回 501）。放在 `grpc/` 下纯粹是因为 `Pipeline` 历史上在这个目录。

---

## 3. 重构目标

1. **`mesh_grpc::*` import 仅出现在 `grpc/engine/` 内**——协议逻辑和响应渲染可独立于 gRPC 进行单元测试
2. **消除 `ProcessingState` 上帝对象**——用类型安全的值传递（tuple）替代 `Option` 状态袋
3. **消除运行时 stage 分发**——pipeline 变成可读的 4 步函数调用
4. **每个文件名即职责**——消灭 `utils.rs`、`common.rs` 等伞状命名
5. **OpenAI Responses API 迁移至 `routers/openai/responses/`**——路径反映所属协议

---

## 4. 重构前后对比

### 4.1 Pipeline：从运行时 stage 列表到显式函数调用

**重构前**：
```rust
// 7 个 stage 通过 trait object 动态分发
let pipeline = RequestPipeline {
    stages: vec![
        Box::new(PreparationStage),
        Box::new(WorkerSelectionStage),
        Box::new(ClientAcquisitionStage),
        Box::new(RequestBuildingStage),
        Box::new(DispatchMetadataStage),
        Box::new(RequestExecutionStage),
        Box::new(ResponseProcessingStage),
    ],
};
// 通过循环运行，ProcessingState 在 stage 间传递
```

**重构后**：
```rust
impl Pipeline {
    pub async fn execute_chat(&self, req, headers, model_id) -> Response {
        // 1. prepare — 构造 transport-neutral 请求
        let (payload, resp_ctx) = prepare::prepare_chat(req, headers, model_id, &self.components)?;
        // 2. select — 选择 worker
        let placement = self.planner.plan(&payload.descriptor()).await?;
        // 3. dispatch — engine 只看 GenerationPayload，不碰 ResponseContext
        let stream = self.engine.dispatch(&placement, &payload).await?;
        // 4. render — 将 TokenChunk stream 渲染为 HTTP 响应
        if resp_ctx.original.is_streaming() {
            render::chat_streaming::process(stream, resp_ctx)
        } else {
            render::chat_aggregator::process(stream, resp_ctx).await
        }
    }
}
```

### 4.2 数据流：从 `Option<T>` 状态袋到类型安全的值传递

**重构前**：
```
ChatCompletionRequest → ProcessingState { Option<A>, Option<B>, ... Option<H> }
                         ↕ (每个 stage 填一个 Option 字段，runtime unwrap)
                        Response
```

**重构后**：
```
ChatCompletionRequest
        │
        ▼
  prepare_chat() ──► (GenerationPayload, ResponseContext)   ← tuple，编译期确定
        │                    │              │
        ▼ (payload)          │              └── 只给 render 用
  engine.dispatch()          │
        │                    │
        ▼                    │
  WorkerStream<TokenChunk>   │
        │                    │
        ▼                    ▼
  render::chat_*::process(stream, resp_ctx) → axum::Response
```

### 4.3 目录结构：从 gRPC 单体到职责分离

**重构前**（全部在 `routers/grpc/` 下）：
```
routers/grpc/               # 9280 行，85% 与 gRPC 无关
├── utils.rs                 # 1214 行万能工具箱
├── context.rs               # ProcessingState 上帝对象
├── pipeline.rs              # 7-stage 动态分发
├── regular/streaming.rs     # 1326 行（SSE 编码，与 gRPC 无关）
├── regular/processor.rs     # 465 行（响应组装，与 gRPC 无关）
├── {common,regular}/responses/  # 2059 行（OpenAI Responses API，无 gRPC import）
├── client.rs, proto_wrapper.rs  # 真正的 gRPC 传输层
└── ...
```

**重构后**：
```
routers/
├── grpc/                    # ~1800 行 — 只剩 HTTP 入口 + pipeline + gRPC engine
│   ├── http_router.rs
│   ├── http_router_pd.rs
│   ├── pipeline.rs          # 4 步显式函数调用
│   └── engine/              # 唯一允许 import mesh_grpc::* 的地方
│       ├── payload_to_proto.rs   # GenerationPayload → proto request
│       ├── proto_to_chunk.rs     # proto response → TokenChunk
│       ├── pd_stream_merge.rs    # PD 流合并状态机
│       └── worker_client_cache.rs
│
├── prepare/                 # ~900 行 — HTTP 请求 → (GenerationPayload, ResponseContext)
│   ├── chat_template.rs
│   ├── tool_constraints.rs
│   ├── stop_sequence_decoder.rs
│   └── generation_payload.rs
│
├── worker_stream/           # ~150 行 — engine 边界类型
│   ├── token_chunk.rs       # TokenChunk（transport-neutral 工作单元）
│   ├── worker_stream.rs
│   └── engine_error.rs
│
├── render/                  # ~2100 行 — Stream<TokenChunk> → axum::Response
│   ├── chat_streaming.rs
│   ├── chat_aggregator.rs
│   ├── generate_streaming.rs
│   └── generate_aggregator.rs
│
└── openai/responses/        # ~2059 行 — OpenAI Responses API（纯搬迁）
```

---

## 5. 核心设计决策

| 决策 | 理由 |
|------|------|
| **引入 `GenerationPayload` + `TokenChunk` 作为对称边界类型** | 请求侧和响应侧各一个 transport-neutral 类型，把 `mesh_grpc::*` 隔离在 `engine/` 内。协议逻辑两侧均可独立单测。 |
| **`prepare_*()` 返回 tuple 而非 umbrella struct** | `(GenerationPayload, ResponseContext)` 的两个值分别流向不同消费者（engine / render），不需要也不应该绑在一起。 |
| **不引入 `Engine` trait** | 只有一个实现（`GrpcEngine`）。单一实现的 trait 是 indirection 而非 polymorphism，需要时再机械提取。 |
| **`openai/responses/` 而非裸 `responses/`** | 文件夹名来自 OpenAI endpoint 名（`/v1/responses`），放在 `openai/` 下避免与 `render/` 产生视觉歧义。 |
| **Strangler-fig 迁移，非大爆炸** | 8 步渐进，每步独立可发布，e2e 门控。 |

---

## 6. 迁移策略概述

采用 **Strangler-fig**（绞杀者模式），共 8 步，每步独立可发布、有明确的 e2e 门控：

| 步骤 | 内容 | 门控 |
|------|------|------|
| 1 | 搬迁 `utils.rs` 中 transport-neutral helpers → `prepare/` 和 `render/` | 现有测试通过 |
| 2a | 单场景 proto byte-equality spike：验证 `GenerationPayload → proto` 可行性 | byte 对齐确认 |
| 2b | 完整实现 `GenerationPayload` + `payload_to_proto` | 4 场景 proto snapshot 通过 |
| 3 | 添加 `worker_stream/` 类型 + `engine/` 子目录 + PD 流合并 | 单元测试 + PD merge T1-T7 |
| 4 | 实现 `render/` 四个文件 | 合成 chunk 输入的单元测试通过 |
| 5 | 新 `Pipeline`（4 步显式调用），Regular 模式切换 | e2e regular 矩阵通过 |
| 6 | PD 模式 + Responses 切换到新 pipeline | e2e PD + SSE 字节快照对齐 |
| 7 | OpenAI Responses API 搬迁至 `routers/openai/responses/` | 分层检查 + smoke test |
| 8 | 删除旧代码：`ProcessingState`、`PipelineStage` trait、`stages/`、空 `utils.rs` | 编译 + 全量 e2e + 无孤儿 |

---

## 7. 预期收益

- **`grpc/` 从 9280 行降至 ~1800 行**，职责清晰：HTTP 入口 + pipeline 编排 + gRPC 传输
- **协议逻辑可独立单测**：`prepare` 和 `render` 不依赖 tonic/proto，测试速度和覆盖率显著提升
- **未来 HTTP-native 后端可直接复用** `prepare/`、`render/`、`worker_stream/`，只需实现新的 `engine`
- **消灭 runtime unwrap 风险**：编译器保证数据在正确阶段可用
- **新人可读性**：pipeline 是一个 4 步函数，不需要理解 stage trait + dynamic dispatch

---

*详细设计参见 `2026-05-19-grpc-engine-extraction-design.md`；PD 流合并状态机参见 `2026-05-19-grpc-pd-merge-spec.md`。*
