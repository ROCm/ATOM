//! Convert backend-specific proto chunks/completes into the transport-neutral
//! `TokenChunk` consumed by the render layer. Also hosts OpenAI-shaped
//! logprob adapters used by `regular/streaming.rs` and `regular/processor.rs`,
//! which still consume `ProtoStream` directly.

use std::sync::Arc;

use mesh_grpc::sglang_proto::{
    generate_complete::MatchedStop as SglangMatchedStop, InputLogProbs as SglangInputLogProbs,
    OutputLogProbs as SglangOutputLogProbs,
};

use crate::protocols::common::{ChatLogProbs, ChatLogProbsContent, TopLogProb};
use crate::routers::grpc::engine::proto_stream_wrapper::{
    ProtoGenerateComplete, ProtoGenerateStreamChunk,
};
use crate::routers::worker_stream::token_chunk::{
    FinishReason, InputLogprobs, MatchedStop, TokenChunk, TokenLogprob, TokenLogprobs, Usage,
    WorkerMeta,
};
use crate::tokenizer::traits::Tokenizer;

pub(crate) fn proto_chunk_to_chunk(chunk: ProtoGenerateStreamChunk) -> TokenChunk {
    let token_ids = chunk.token_ids().to_vec();
    let logprobs = chunk.output_logprobs().map(sglang_output_to_token_logprobs);
    TokenChunk::Partial {
        token_ids,
        logprobs,
    }
}

pub(crate) fn proto_complete_to_chunk(complete: ProtoGenerateComplete) -> TokenChunk {
    let finish_reason = parse_finish_reason(complete.finish_reason());
    let matched_stop = complete.matched_stop().map(map_sglang_matched_stop);
    let usage = Usage {
        prompt_tokens: complete.prompt_tokens().max(0) as u32,
        completion_tokens: complete.completion_tokens().max(0) as u32,
        total_tokens: (complete.prompt_tokens().max(0) + complete.completion_tokens().max(0))
            as u32,
    };
    let logprobs = complete
        .output_logprobs()
        .map(sglang_output_to_token_logprobs);
    let input_logprobs = complete
        .input_logprobs()
        .map(sglang_input_to_input_logprobs);
    let meta = WorkerMeta {
        request_id: String::new(),
        weight_version: None,
        cached_tokens: complete.cached_tokens().max(0) as u32,
    };
    TokenChunk::Complete {
        token_ids: complete.output_ids().to_vec(),
        finish_reason,
        matched_stop,
        usage,
        logprobs,
        input_logprobs,
        meta,
    }
}

fn map_sglang_matched_stop(ms: &SglangMatchedStop) -> MatchedStop {
    match ms {
        SglangMatchedStop::MatchedStopStr(s) => MatchedStop::Str(s.clone()),
        SglangMatchedStop::MatchedTokenId(t) => MatchedStop::TokenId(*t),
    }
}

fn parse_finish_reason(s: &str) -> FinishReason {
    match s {
        "stop" => FinishReason::Stop,
        "length" => FinishReason::Length,
        "content_filter" => FinishReason::ContentFilter,
        "tool_calls" => FinishReason::ToolCalls,
        "abort" => FinishReason::Abort,
        other if other.is_empty() => FinishReason::Stop,
        other => FinishReason::Other(other.to_string()),
    }
}

fn sglang_output_to_token_logprobs(lp: &SglangOutputLogProbs) -> TokenLogprobs {
    let mut items = Vec::with_capacity(lp.token_logprobs.len());
    for (i, (&logprob, &token_id)) in lp
        .token_logprobs
        .iter()
        .zip(lp.token_ids.iter())
        .enumerate()
    {
        let top = lp
            .top_logprobs
            .get(i)
            .map(|tl| {
                tl.values
                    .iter()
                    .zip(tl.token_ids.iter())
                    .map(|(&v, &tid)| (tid as u32, v, None))
                    .collect()
            })
            .unwrap_or_default();
        items.push(TokenLogprob {
            token_id: token_id as u32,
            logprob,
            decoded_text: None,
            top,
        });
    }
    TokenLogprobs { items }
}

fn sglang_input_to_input_logprobs(lp: &SglangInputLogProbs) -> InputLogprobs {
    let mut items = Vec::with_capacity(lp.token_logprobs.len());
    for (i, (entry, &token_id)) in lp
        .token_logprobs
        .iter()
        .zip(lp.token_ids.iter())
        .enumerate()
    {
        let top = lp
            .top_logprobs
            .get(i)
            .map(|tl| {
                tl.values
                    .iter()
                    .zip(tl.token_ids.iter())
                    .map(|(&v, &tid)| (tid as u32, v, None))
                    .collect()
            })
            .unwrap_or_default();
        items.push(TokenLogprob {
            token_id: token_id as u32,
            logprob: entry.value.unwrap_or(0.0),
            decoded_text: None,
            top,
        });
    }
    InputLogprobs { items }
}

/// OpenAI-shaped logprob converter for callers that have not migrated to
/// the neutral [`proto_complete_to_chunk`] output.
pub(crate) fn convert_proto_to_openai_logprobs(
    proto_logprobs: &SglangOutputLogProbs,
    tokenizer: &Arc<dyn Tokenizer>,
) -> Result<ChatLogProbs, String> {
    let mut content_items = Vec::with_capacity(proto_logprobs.token_logprobs.len());

    let token_texts: Vec<String> = proto_logprobs
        .token_ids
        .iter()
        .map(|&token_id| {
            tokenizer
                .decode(&[token_id as u32], false)
                .unwrap_or_else(|_| format!("<token_{}>", token_id))
        })
        .collect();

    for (i, (&logprob, token_text)) in proto_logprobs
        .token_logprobs
        .iter()
        .zip(token_texts.into_iter())
        .enumerate()
    {
        let bytes = Some(token_text.as_bytes().to_vec());

        let top_logprobs = if let Some(top_logprobs_entry) = proto_logprobs.top_logprobs.get(i) {
            let mut top_logprobs = Vec::with_capacity(top_logprobs_entry.values.len());

            let top_token_texts: Vec<String> = top_logprobs_entry
                .token_ids
                .iter()
                .map(|&tid| {
                    tokenizer
                        .decode(&[tid as u32], false)
                        .unwrap_or_else(|_| format!("<token_{}>", tid))
                })
                .collect();

            for (j, (&top_logprob, &_top_token_id)) in top_logprobs_entry
                .values
                .iter()
                .zip(top_logprobs_entry.token_ids.iter())
                .enumerate()
            {
                if let Some(top_token_text) = top_token_texts.get(j) {
                    top_logprobs.push(TopLogProb {
                        token: top_token_text.clone(),
                        logprob: top_logprob,
                        bytes: Some(top_token_text.as_bytes().to_vec()),
                    });
                }
            }
            top_logprobs
        } else {
            Vec::new()
        };

        content_items.push(ChatLogProbsContent {
            token: token_text,
            logprob,
            bytes,
            top_logprobs,
        });
    }

    Ok(ChatLogProbs::Detailed {
        content: (!content_items.is_empty()).then_some(content_items),
    })
}

pub(crate) fn convert_generate_output_logprobs(
    proto_logprobs: &SglangOutputLogProbs,
) -> Vec<Vec<Option<f64>>> {
    proto_logprobs
        .token_logprobs
        .iter()
        .zip(proto_logprobs.token_ids.iter())
        .map(|(&logprob, &token_id)| vec![Some(logprob as f64), Some(token_id as f64)])
        .collect()
}

pub(crate) fn convert_generate_input_logprobs(
    proto_logprobs: &SglangInputLogProbs,
) -> Vec<Vec<Option<f64>>> {
    proto_logprobs
        .token_logprobs
        .iter()
        .zip(proto_logprobs.token_ids.iter())
        .map(|(token_logprob, &token_id)| {
            let logprob_value = token_logprob.value.map(|v| v as f64);
            vec![logprob_value, Some(token_id as f64)]
        })
        .collect()
}
