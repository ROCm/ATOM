//! Transport-neutral response context: everything the render layer needs to
//! turn `WorkerStream<TokenChunk>` into an HTTP `Response`. Built by the
//! `prepare_chat` / `prepare_generate` helpers alongside `GenerationPayload`.

use std::sync::Arc;

use http::HeaderMap;

use crate::{
    protocols::{chat::ChatCompletionRequest, generate::GenerateRequest},
    routers::prepare::chat_template::ProcessedMessages,
    tokenizer::{traits::Tokenizer, StopSequenceDecoder},
};

/// The original protocol request, in `Arc` form so the render layer can hand
/// it back to downstream callers (e.g. Responses-API echoes the chat request)
/// without cloning.
pub enum ProtocolRequest {
    Chat(Arc<ChatCompletionRequest>),
    Generate(Arc<GenerateRequest>),
}

impl ProtocolRequest {
    pub fn is_streaming(&self) -> bool {
        match self {
            ProtocolRequest::Chat(r) => r.stream,
            ProtocolRequest::Generate(r) => r.stream,
        }
    }
}

pub struct ResponseContext {
    pub original: ProtocolRequest,
    pub model_id: Option<String>,
    pub headers: Option<HeaderMap>,
    pub original_text: Option<String>,
    pub processed_messages: Option<ProcessedMessages>,
    pub tokenizer: Arc<dyn Tokenizer>,
    pub stop_decoder: StopSequenceDecoder,
}
