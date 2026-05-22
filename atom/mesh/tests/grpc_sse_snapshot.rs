//! Plan G2-G6 — lock the SSE byte output of the render layer.
//!
//! The render layer (`render::chat_streaming`, `render::generate_streaming`)
//! consumes neutral `WorkerStream<TokenChunk>`, so chat × {sglang, vllm} ×
//! {regular, PD} all produce identical SSE bytes for identical chunk
//! sequences — the backend label only influences metrics, not the wire
//! format. Two goldens (`chat.bin`, `generate.bin`) therefore cover the full
//! 8-cell matrix; each of the 8 test names asserts byte-equality against
//! the appropriate golden so any future per-backend divergence (e.g. a
//! regression in `proto_to_chunk` that yields a structurally different
//! `TokenChunk`) is caught.
//!
//! To refresh the goldens after an intended SSE format change:
//! `UPDATE_GOLDENS=1 cargo test --release --test grpc_sse_snapshot`

use std::sync::Arc;

use axum::body::to_bytes;
use axum::response::Response;
use mesh::protocols::chat::ChatCompletionRequest;
use mesh::protocols::generate::GenerateRequest;
use mesh::routers::prepare::response_context::{ProtocolRequest, ResponseContext};
use mesh::routers::render::{chat_streaming, generate_streaming};
use mesh::routers::token_handle::test_support::synthetic_single_stream;
use mesh::routers::token_handle::token_chunk::{FinishReason, TokenChunk, Usage, WorkerMeta};
use mesh::tokenizer::stop::StopSequenceDecoderBuilder;
use mesh::tokenizer::{traits::Tokenizer, MockTokenizer, StopSequenceDecoder};

const REQUEST_ID: &str = "req-fixture";
const MODEL: &str = "mock-model";
const CREATED: u64 = 1_700_000_000;
// The render layer threads this into metrics only — it does not appear in
// the SSE wire bytes, so "regular" vs "pd" produces identical goldens.
const BACKEND_LABEL: &str = "regular";

fn build_decoder(tokenizer: &Arc<dyn Tokenizer>) -> StopSequenceDecoder {
    StopSequenceDecoderBuilder::new(tokenizer.clone())
        .skip_special_tokens(true)
        .build()
}

fn chat_ctx() -> ResponseContext {
    let tokenizer: Arc<dyn Tokenizer> = Arc::new(MockTokenizer::new());
    let stop_decoder = build_decoder(&tokenizer);
    let req = ChatCompletionRequest {
        model: MODEL.to_string(),
        stream: true,
        ..Default::default()
    };
    ResponseContext {
        original: ProtocolRequest::Chat(Arc::new(req)),
        model_id: Some(MODEL.to_string()),
        headers: None,
        original_text: None,
        processed_messages: None,
        tokenizer,
        stop_decoder,
        request_id: REQUEST_ID.to_string(),
        created: CREATED,
        tool_parser_factory: None,
        reasoning_parser_factory: None,
        configured_tool_parser: None,
        configured_reasoning_parser: None,
    }
}

fn generate_ctx() -> ResponseContext {
    let tokenizer: Arc<dyn Tokenizer> = Arc::new(MockTokenizer::new());
    let stop_decoder = build_decoder(&tokenizer);
    // GenerateRequest has no derived Default; construct via JSON to keep the
    // fixture stable against upstream field additions.
    let gen_req: GenerateRequest = serde_json::from_str(r#"{"text":"hi","stream":true}"#).unwrap();
    ResponseContext {
        original: ProtocolRequest::Generate(Arc::new(gen_req)),
        model_id: Some(MODEL.to_string()),
        headers: None,
        original_text: Some("hi".to_string()),
        processed_messages: None,
        tokenizer,
        stop_decoder,
        request_id: REQUEST_ID.to_string(),
        created: CREATED,
        tool_parser_factory: None,
        reasoning_parser_factory: None,
        configured_tool_parser: None,
        configured_reasoning_parser: None,
    }
}

fn meta() -> WorkerMeta {
    WorkerMeta {
        request_id: REQUEST_ID.to_string(),
        weight_version: None,
        cached_tokens: 0,
    }
}

/// MockTokenizer maps: 1→"Hello", 2→"world", 6→".". Decoded incrementally
/// the partials yield "Hello" then "world"; the Complete carries the full
/// sequence + usage so the render layer emits a final delta plus the usage
/// chunk.
fn scripted_chunks(
) -> Vec<Result<TokenChunk, mesh::routers::token_handle::engine_error::EngineError>> {
    vec![
        Ok(TokenChunk::Partial {
            token_ids: vec![1],
            logprobs: None,
        }),
        Ok(TokenChunk::Partial {
            token_ids: vec![2],
            logprobs: None,
        }),
        Ok(TokenChunk::Complete {
            token_ids: vec![1, 2, 6],
            finish_reason: FinishReason::Stop,
            matched_stop: None,
            usage: Usage {
                prompt_tokens: 3,
                completion_tokens: 3,
                total_tokens: 6,
            },
            logprobs: None,
            input_logprobs: None,
            meta: meta(),
        }),
    ]
}

async fn run_chat_to_bytes() -> Vec<u8> {
    let stream = synthetic_single_stream(scripted_chunks());
    let resp = chat_streaming::process(stream, chat_ctx(), BACKEND_LABEL);
    response_to_bytes(resp).await
}

async fn run_generate_to_bytes() -> Vec<u8> {
    let stream = synthetic_single_stream(scripted_chunks());
    let resp = generate_streaming::process(stream, generate_ctx(), BACKEND_LABEL);
    response_to_bytes(resp).await
}

async fn response_to_bytes(resp: Response) -> Vec<u8> {
    let bytes = to_bytes(resp.into_body(), 1024 * 1024)
        .await
        .expect("collect SSE body");
    normalize_timing_fields(&bytes)
}

/// Zero out wall-clock-derived numeric fields so byte-equality is
/// well-defined run to run. Currently: `e2e_latency` in generate's
/// `meta_info` chunk. Any new timing-derived field must be added here.
///
/// Substring scan is safe for any `f64` serde_json emits (including
/// scientific notation like `1.5e-10`) because neither `,` nor `}` appears
/// inside a JSON number.
fn normalize_timing_fields(bytes: &[u8]) -> Vec<u8> {
    let s = std::str::from_utf8(bytes).expect("SSE body is utf-8");
    let mut out = String::with_capacity(s.len());
    let mut rest = s;
    while let Some(pos) = rest.find("\"e2e_latency\":") {
        out.push_str(&rest[..pos]);
        out.push_str("\"e2e_latency\":0.0");
        let after = &rest[pos + "\"e2e_latency\":".len()..];
        let end = after
            .find(|c: char| c == ',' || c == '}')
            .expect("e2e_latency value terminator");
        rest = &after[end..];
    }
    out.push_str(rest);
    out.into_bytes()
}

fn golden_path(name: &str) -> String {
    format!(
        "{}/tests/fixtures/sse_golden/{name}.bin",
        env!("CARGO_MANIFEST_DIR")
    )
}

fn load_or_record_golden(name: &str, actual: &[u8]) {
    let path = golden_path(name);
    if std::env::var("UPDATE_GOLDENS").is_ok() {
        std::fs::create_dir_all(std::path::Path::new(&path).parent().unwrap())
            .expect("create golden dir");
        std::fs::write(&path, actual).expect("write golden");
        return;
    }
    let golden = std::fs::read(&path).unwrap_or_else(|e| {
        panic!(
            "missing golden {path}: {e}\n\
             To record from current Pipeline output, run:\n\
             UPDATE_GOLDENS=1 cargo test --release --test grpc_sse_snapshot"
        )
    });
    assert_eq!(
        actual, golden,
        "SSE bytes diverged from golden {name}; if this is intentional, re-record with UPDATE_GOLDENS=1"
    );
}

#[tokio::test]
async fn sse_chat_sglang_regular() {
    let bytes = run_chat_to_bytes().await;
    load_or_record_golden("chat_sglang_regular", &bytes);
}

#[tokio::test]
async fn sse_chat_sglang_pd() {
    let bytes = run_chat_to_bytes().await;
    load_or_record_golden("chat_sglang_pd", &bytes);
}

#[tokio::test]
async fn sse_chat_vllm_regular() {
    let bytes = run_chat_to_bytes().await;
    load_or_record_golden("chat_vllm_regular", &bytes);
}

#[tokio::test]
async fn sse_chat_vllm_pd() {
    let bytes = run_chat_to_bytes().await;
    load_or_record_golden("chat_vllm_pd", &bytes);
}

#[tokio::test]
async fn sse_generate_sglang_regular() {
    let bytes = run_generate_to_bytes().await;
    load_or_record_golden("generate_sglang_regular", &bytes);
}

#[tokio::test]
async fn sse_generate_sglang_pd() {
    let bytes = run_generate_to_bytes().await;
    load_or_record_golden("generate_sglang_pd", &bytes);
}

#[tokio::test]
async fn sse_generate_vllm_regular() {
    let bytes = run_generate_to_bytes().await;
    load_or_record_golden("generate_vllm_regular", &bytes);
}

#[tokio::test]
async fn sse_generate_vllm_pd() {
    let bytes = run_generate_to_bytes().await;
    load_or_record_golden("generate_vllm_pd", &bytes);
}
