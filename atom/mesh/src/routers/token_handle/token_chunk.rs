//! Transport-neutral token-stream chunk produced by the gRPC engine and
//! consumed by the render layer.
//!
//! No `mesh_grpc::*` types appear here; backend protos are converted into
//! these neutral structs inside `grpc/engine/proto_to_chunk.rs`.

#[derive(Debug, Clone)]
pub enum TokenChunk {
    Partial {
        token_ids: Vec<u32>,
        logprobs: Option<TokenLogprobs>,
    },
    /// In PD mode, `input_logprobs` is injected by the merger from the prefill
    /// worker's Complete; all other fields come from the decode worker.
    Complete {
        token_ids: Vec<u32>,
        finish_reason: FinishReason,
        matched_stop: Option<MatchedStop>,
        usage: Usage,
        logprobs: Option<TokenLogprobs>,
        input_logprobs: Option<InputLogprobs>,
        meta: WorkerMeta,
    },
}

#[derive(Debug, Clone)]
pub enum FinishReason {
    Stop,
    Length,
    ContentFilter,
    ToolCalls,
    Abort,
    Other(String),
}

#[derive(Debug, Clone)]
pub enum MatchedStop {
    Str(String),
    TokenId(u32),
}

#[derive(Debug, Clone, Default)]
pub struct TokenLogprobs {
    pub items: Vec<TokenLogprob>,
}

#[derive(Debug, Clone, Default)]
pub struct InputLogprobs {
    pub items: Vec<TokenLogprob>,
}

#[derive(Debug, Clone)]
pub struct TokenLogprob {
    pub token_id: u32,
    pub logprob: f32,
    pub decoded_text: Option<String>,
    pub top: Vec<(u32, f32, Option<String>)>,
}

#[derive(Debug, Clone, Default)]
pub struct Usage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
}

#[derive(Debug, Clone, Default)]
pub struct WorkerMeta {
    pub request_id: String,
    pub weight_version: Option<String>,
    pub cached_tokens: u32,
}
