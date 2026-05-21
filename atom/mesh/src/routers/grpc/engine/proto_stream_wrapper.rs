//! Unified enums wrapping SGLang and vLLM proto types so the router can work
//! with either backend transparently.

use futures_util::StreamExt;
use mesh_grpc::{
    sglang_proto::{self as sglang, generate_complete::MatchedStop},
    sglang_scheduler::AbortOnDropStream as SglangStream,
    vllm_engine::AbortOnDropStream as VllmStream,
    vllm_proto as vllm,
};

#[derive(Clone)]
pub enum ProtoRequest {
    Generate(ProtoGenerateRequest),
}

impl ProtoRequest {
    pub fn request_id(&self) -> &str {
        match self {
            Self::Generate(req) => req.request_id(),
        }
    }
}

#[derive(Clone)]
pub enum ProtoGenerateRequest {
    Sglang(Box<sglang::GenerateRequest>),
    Vllm(Box<vllm::GenerateRequest>),
}

impl ProtoGenerateRequest {
    pub fn as_sglang(&self) -> &sglang::GenerateRequest {
        match self {
            Self::Sglang(req) => req,
            Self::Vllm(_) => panic!("Expected SGLang GenerateRequest, got vLLM"),
        }
    }

    pub fn as_sglang_mut(&mut self) -> &mut sglang::GenerateRequest {
        match self {
            Self::Sglang(req) => req,
            Self::Vllm(_) => panic!("Expected SGLang GenerateRequest, got vLLM"),
        }
    }

    pub fn as_vllm(&self) -> &vllm::GenerateRequest {
        match self {
            Self::Vllm(req) => req,
            Self::Sglang(_) => panic!("Expected vLLM GenerateRequest, got SGLang"),
        }
    }

    pub fn as_vllm_mut(&mut self) -> &mut vllm::GenerateRequest {
        match self {
            Self::Vllm(req) => req,
            Self::Sglang(_) => panic!("Expected vLLM GenerateRequest, got SGLang"),
        }
    }

    pub fn is_sglang(&self) -> bool {
        matches!(self, Self::Sglang(_))
    }

    pub fn is_vllm(&self) -> bool {
        matches!(self, Self::Vllm(_))
    }

    pub fn clone_inner(&self) -> Self {
        self.clone()
    }

    pub fn request_id(&self) -> &str {
        match self {
            Self::Sglang(req) => &req.request_id,
            Self::Vllm(req) => &req.request_id,
        }
    }
}

pub enum ProtoGenerateResponse {
    Sglang(Box<sglang::GenerateResponse>),
    Vllm(vllm::GenerateResponse),
}

impl ProtoGenerateResponse {
    /// Consumes self to avoid cloning large proto messages on the streaming hot path.
    pub fn into_response(self) -> ProtoResponseVariant {
        match self {
            Self::Sglang(resp) => match resp.response {
                Some(sglang::generate_response::Response::Chunk(chunk)) => {
                    ProtoResponseVariant::Chunk(ProtoGenerateStreamChunk::Sglang(chunk))
                }
                Some(sglang::generate_response::Response::Complete(complete)) => {
                    ProtoResponseVariant::Complete(ProtoGenerateComplete::Sglang(complete))
                }
                Some(sglang::generate_response::Response::Error(error)) => {
                    ProtoResponseVariant::Error(ProtoGenerateError::Sglang(error))
                }
                None => ProtoResponseVariant::None,
            },
            Self::Vllm(resp) => match resp.response {
                Some(vllm::generate_response::Response::Chunk(chunk)) => {
                    ProtoResponseVariant::Chunk(ProtoGenerateStreamChunk::Vllm(chunk))
                }
                Some(vllm::generate_response::Response::Complete(complete)) => {
                    ProtoResponseVariant::Complete(ProtoGenerateComplete::Vllm(complete))
                }
                // vLLM proto has no Error variant; errors flow via gRPC status.
                None => ProtoResponseVariant::None,
            },
        }
    }
}

pub enum ProtoResponseVariant {
    Chunk(ProtoGenerateStreamChunk),
    Complete(ProtoGenerateComplete),
    Error(ProtoGenerateError),
    None,
}

#[derive(Clone)]
pub enum ProtoGenerateStreamChunk {
    Sglang(sglang::GenerateStreamChunk),
    Vllm(vllm::GenerateStreamChunk),
}

impl ProtoGenerateStreamChunk {
    pub fn as_sglang(&self) -> &sglang::GenerateStreamChunk {
        match self {
            Self::Sglang(chunk) => chunk,
            Self::Vllm(_) => panic!("Expected SGLang GenerateStreamChunk, got vLLM"),
        }
    }

    pub fn as_vllm(&self) -> &vllm::GenerateStreamChunk {
        match self {
            Self::Vllm(chunk) => chunk,
            Self::Sglang(_) => panic!("Expected vLLM GenerateStreamChunk, got SGLang"),
        }
    }

    pub fn is_sglang(&self) -> bool {
        matches!(self, Self::Sglang(_))
    }

    pub fn is_vllm(&self) -> bool {
        matches!(self, Self::Vllm(_))
    }

    pub fn token_ids(&self) -> &[u32] {
        match self {
            Self::Sglang(c) => &c.token_ids,
            Self::Vllm(c) => &c.token_ids,
        }
    }

    /// vLLM does not support `n>1`; always returns 0.
    pub fn index(&self) -> u32 {
        match self {
            Self::Sglang(c) => c.index,
            Self::Vllm(_) => 0,
        }
    }

    /// SGLang only; returns `None` for vLLM.
    pub fn output_logprobs(&self) -> Option<&sglang::OutputLogProbs> {
        match self {
            Self::Sglang(c) => c.output_logprobs.as_ref(),
            Self::Vllm(_) => None,
        }
    }

    pub fn prompt_tokens(&self) -> i32 {
        match self {
            Self::Sglang(c) => c.prompt_tokens,
            Self::Vllm(c) => c.prompt_tokens as i32,
        }
    }

    pub fn completion_tokens(&self) -> i32 {
        match self {
            Self::Sglang(c) => c.completion_tokens,
            Self::Vllm(c) => c.completion_tokens as i32,
        }
    }

    pub fn cached_tokens(&self) -> i32 {
        match self {
            Self::Sglang(c) => c.cached_tokens,
            Self::Vllm(c) => c.cached_tokens as i32,
        }
    }
}

#[derive(Clone)]
pub enum ProtoGenerateComplete {
    Sglang(sglang::GenerateComplete),
    Vllm(vllm::GenerateComplete),
}

impl ProtoGenerateComplete {
    pub fn as_sglang(&self) -> &sglang::GenerateComplete {
        match self {
            Self::Sglang(complete) => complete,
            Self::Vllm(_) => panic!("Expected SGLang GenerateComplete, got vLLM"),
        }
    }

    pub fn as_sglang_mut(&mut self) -> &mut sglang::GenerateComplete {
        match self {
            Self::Sglang(complete) => complete,
            Self::Vllm(_) => panic!("Expected SGLang GenerateComplete, got vLLM"),
        }
    }

    pub fn as_vllm(&self) -> &vllm::GenerateComplete {
        match self {
            Self::Vllm(complete) => complete,
            Self::Sglang(_) => panic!("Expected vLLM GenerateComplete, got SGLang"),
        }
    }

    pub fn is_sglang(&self) -> bool {
        matches!(self, Self::Sglang(_))
    }

    pub fn is_vllm(&self) -> bool {
        matches!(self, Self::Vllm(_))
    }

    pub fn token_ids(&self) -> &[u32] {
        match self {
            Self::Sglang(c) => &c.output_ids,
            Self::Vllm(c) => &c.output_ids,
        }
    }

    pub fn prompt_tokens(&self) -> i32 {
        match self {
            Self::Sglang(c) => c.prompt_tokens,
            Self::Vllm(c) => c.prompt_tokens as i32,
        }
    }

    pub fn completion_tokens(&self) -> i32 {
        match self {
            Self::Sglang(c) => c.completion_tokens,
            Self::Vllm(c) => c.completion_tokens as i32,
        }
    }

    pub fn finish_reason(&self) -> &str {
        match self {
            Self::Sglang(c) => &c.finish_reason,
            Self::Vllm(c) => &c.finish_reason,
        }
    }

    /// vLLM does not support `n>1`; always returns 0.
    pub fn index(&self) -> u32 {
        match self {
            Self::Sglang(c) => c.index,
            Self::Vllm(_) => 0,
        }
    }

    /// SGLang only; vLLM has no matched_stop and returns `None`.
    pub fn matched_stop(&self) -> Option<&MatchedStop> {
        match self {
            Self::Sglang(c) => c.matched_stop.as_ref(),
            Self::Vllm(_) => None,
        }
    }

    pub fn output_ids(&self) -> &[u32] {
        match self {
            Self::Sglang(c) => &c.output_ids,
            Self::Vllm(c) => &c.output_ids,
        }
    }

    pub fn cached_tokens(&self) -> i32 {
        match self {
            Self::Sglang(c) => c.cached_tokens,
            Self::Vllm(c) => c.cached_tokens as i32,
        }
    }

    /// SGLang only; returns `None` for vLLM.
    pub fn input_logprobs(&self) -> Option<&sglang::InputLogProbs> {
        match self {
            Self::Sglang(c) => c.input_logprobs.as_ref(),
            Self::Vllm(_) => None,
        }
    }

    /// SGLang only; returns `None` for vLLM.
    pub fn output_logprobs(&self) -> Option<&sglang::OutputLogProbs> {
        match self {
            Self::Sglang(c) => c.output_logprobs.as_ref(),
            Self::Vllm(_) => None,
        }
    }
}

/// vLLM proto has no GenerateError variant; vLLM errors flow via gRPC status.
#[derive(Clone)]
pub enum ProtoGenerateError {
    Sglang(sglang::GenerateError),
}

impl ProtoGenerateError {
    pub fn message(&self) -> &str {
        match self {
            Self::Sglang(e) => &e.message,
        }
    }
}

pub enum ProtoStream {
    Sglang(SglangStream),
    Vllm(VllmStream),
}

impl ProtoStream {
    pub async fn next(&mut self) -> Option<Result<ProtoGenerateResponse, tonic::Status>> {
        match self {
            Self::Sglang(stream) => stream
                .next()
                .await
                .map(|result| result.map(|r| ProtoGenerateResponse::Sglang(Box::new(r)))),
            Self::Vllm(stream) => stream
                .next()
                .await
                .map(|result| result.map(ProtoGenerateResponse::Vllm)),
        }
    }

    pub fn mark_completed(&mut self) {
        match self {
            Self::Sglang(stream) => stream.mark_completed(),
            Self::Vllm(stream) => stream.mark_completed(),
        }
    }
}

#[derive(Clone)]
pub enum ProtoEmbedRequest {
    Sglang(Box<sglang::EmbedRequest>),
}

impl ProtoEmbedRequest {
    pub fn as_sglang(&self) -> &sglang::EmbedRequest {
        match self {
            Self::Sglang(req) => req,
        }
    }

    pub fn as_sglang_mut(&mut self) -> &mut sglang::EmbedRequest {
        match self {
            Self::Sglang(req) => req,
        }
    }

    pub fn is_sglang(&self) -> bool {
        matches!(self, Self::Sglang(_))
    }

    pub fn clone_inner(&self) -> Self {
        self.clone()
    }

    pub fn request_id(&self) -> &str {
        match self {
            Self::Sglang(req) => &req.request_id,
        }
    }
}

pub enum ProtoEmbedResponse {
    Sglang(sglang::EmbedResponse),
}

impl ProtoEmbedResponse {
    pub fn into_response(self) -> ProtoEmbedResponseVariant {
        match self {
            Self::Sglang(resp) => match resp.response {
                Some(sglang::embed_response::Response::Complete(complete)) => {
                    ProtoEmbedResponseVariant::Complete(ProtoEmbedComplete::Sglang(complete))
                }
                Some(sglang::embed_response::Response::Error(error)) => {
                    ProtoEmbedResponseVariant::Error(ProtoEmbedError::Sglang(error))
                }
                None => ProtoEmbedResponseVariant::None,
            },
        }
    }
}

pub enum ProtoEmbedResponseVariant {
    Complete(ProtoEmbedComplete),
    Error(ProtoEmbedError),
    None,
}

#[derive(Clone)]
pub enum ProtoEmbedComplete {
    Sglang(sglang::EmbedComplete),
}

impl ProtoEmbedComplete {
    pub fn embedding(&self) -> &[f32] {
        match self {
            Self::Sglang(c) => &c.embedding,
        }
    }

    pub fn prompt_tokens(&self) -> i32 {
        match self {
            Self::Sglang(c) => c.prompt_tokens,
        }
    }

    pub fn cached_tokens(&self) -> i32 {
        match self {
            Self::Sglang(c) => c.cached_tokens,
        }
    }

    pub fn embedding_dim(&self) -> i32 {
        match self {
            Self::Sglang(c) => c.embedding_dim,
        }
    }
}

#[derive(Clone)]
pub enum ProtoEmbedError {
    Sglang(sglang::EmbedError),
}

impl ProtoEmbedError {
    pub fn message(&self) -> &str {
        match self {
            Self::Sglang(e) => &e.message,
        }
    }

    pub fn code(&self) -> &str {
        match self {
            Self::Sglang(e) => &e.code,
        }
    }
}
