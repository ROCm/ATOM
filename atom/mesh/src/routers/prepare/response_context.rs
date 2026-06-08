//! Transport-neutral response context: everything the render layer needs to
//! turn `TokenHandle<TokenChunk>` into an HTTP `Response`. Built by the
//! `prepare_chat` / `prepare_generate` helpers alongside `GenerationPayload`.

use std::sync::Arc;

use http::HeaderMap;

use crate::{
    protocols::{chat::ChatCompletionRequest, generate::GenerateRequest},
    reasoning_parser::ParserFactory as ReasoningParserFactory,
    routers::prepare::chat_template::ProcessedMessages,
    tokenizer::{traits::Tokenizer, StopSequenceDecoder},
    tool_parser::ParserFactory as ToolParserFactory,
};

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
    pub request_id: String,
    pub created: u64,
    pub tool_parser_factory: Option<ToolParserFactory>,
    pub reasoning_parser_factory: Option<ReasoningParserFactory>,
    pub configured_tool_parser: Option<String>,
    pub configured_reasoning_parser: Option<String>,
}
