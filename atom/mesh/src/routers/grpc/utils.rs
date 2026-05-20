//! Shared utilities for gRPC routers

use std::sync::Arc;

use axum::response::Response;
use mesh_grpc::sglang_proto::{InputLogProbs, OutputLogProbs};
use tracing::error;

use super::{
    client::GrpcClient,
    context::RequestContext,
    proto_wrapper::{ProtoGenerateComplete, ProtoStream},
};
use crate::{
    core::Worker,
    protocols::common::{ChatLogProbs, ChatLogProbsContent, TopLogProb},
    routers::{error, grpc::proto_wrapper::ProtoResponseVariant},
    tokenizer::traits::Tokenizer,
};

/// Resolve tokenizer from registry and cache it in request context.
///
/// This is a helper to avoid duplicating tokenizer resolution logic across
/// preparation stages (chat, generate, embedding).
///
/// Returns the tokenizer Arc, which is also cached in `ctx.state.tokenizer`.
pub(crate) fn resolve_tokenizer(
    ctx: &mut RequestContext,
    stage_name: &str,
) -> Result<Arc<dyn Tokenizer>, Box<Response>> {
    let model_id = ctx.input.model_id.as_deref().ok_or_else(|| {
        error!(
            function = %stage_name,
            "model_id not set in request context"
        );
        Box::new(error::internal_error(
            "model_id_not_set",
            "model_id not set in request context - this is a bug in request routing",
        ))
    })?;

    let tokenizer = ctx
        .components
        .tokenizer_registry
        .get(model_id)
        .ok_or_else(|| {
            error!(
                function = %stage_name,
                model = %model_id,
                "Tokenizer not found for model"
            );
            Box::new(error::internal_error(
                "tokenizer_not_found",
                format!("Tokenizer not found for model: {}", model_id),
            ))
        })?;

    // Cache tokenizer in context for reuse in response processing stage
    ctx.state.tokenizer = Some(tokenizer.clone());

    Ok(tokenizer)
}

/// Get gRPC client from worker, returning appropriate error response on failure
pub(crate) async fn get_grpc_client_from_worker(
    worker: &Arc<dyn Worker>,
) -> Result<GrpcClient, Response> {
    // Get cached client from worker (or create one if not cached yet)
    let client_arc = worker
        .get_grpc_client()
        .await
        .map_err(|e| {
            error!(
                function = "get_grpc_client_from_worker",
                error = %e,
                "Failed to get gRPC client from worker"
            );
            error::internal_error(
                "get_grpc_client_failed",
                format!("Failed to get gRPC client: {}", e),
            )
        })?
        .ok_or_else(|| {
            error!(
                function = "get_grpc_client_from_worker",
                "Selected worker not configured for gRPC"
            );
            error::internal_error(
                "worker_not_configured_for_grpc",
                "Selected worker is not configured for gRPC",
            )
        })?;

    Ok((*client_arc).clone())
}

/// Collect responses from a gRPC stream
///
/// This helper processes a gRPC GenerateResponse stream and collects all Complete responses.
/// Used by both regular and PD routers for non-streaming requests.
///
/// # Arguments
/// * `stream` - The gRPC response stream to consume
/// * `worker_name` - Name for logging (e.g., "Prefill", "Decode", "Worker")
///
/// # Returns
/// * `Ok(Vec<GenerateComplete>)` - All complete responses collected from the stream
/// * `Err(Response)` - Error response if the stream fails or returns an error
pub(crate) async fn collect_stream_responses(
    stream: &mut ProtoStream,
    worker_name: &str,
) -> Result<Vec<ProtoGenerateComplete>, Response> {
    let mut all_responses = Vec::new();

    while let Some(response) = stream.next().await {
        match response {
            Ok(gen_response) => {
                match gen_response.into_response() {
                    ProtoResponseVariant::Complete(complete) => {
                        all_responses.push(complete);
                    }
                    ProtoResponseVariant::Error(err) => {
                        error!(function = "collect_stream_responses", worker = %worker_name, error = %err.message(), "Worker generation error");
                        // Don't mark as completed - let Drop send abort for error cases
                        return Err(error::internal_error(
                            "worker_generation_failed",
                            format!("{} generation failed: {}", worker_name, err.message()),
                        ));
                    }
                    ProtoResponseVariant::Chunk(_chunk) => {
                        // Streaming chunk - no action needed
                    }
                    ProtoResponseVariant::None => {
                        // Empty response - no action needed
                    }
                }
            }
            Err(e) => {
                error!(function = "collect_stream_responses", worker = %worker_name, error = ?e, "Worker stream error");
                // Don't mark as completed - let Drop send abort for error cases
                return Err(error::internal_error(
                    "worker_stream_failed",
                    format!("{} stream failed: {}", worker_name, e),
                ));
            }
        }
    }

    Ok(all_responses)
}

/// Convert OutputLogProbs to OpenAI ChatLogProbs format
///
/// This function decodes token IDs using the tokenizer and builds the logprobs structure
/// expected by the OpenAI API format.
pub(crate) fn convert_proto_to_openai_logprobs(
    proto_logprobs: &OutputLogProbs,
    tokenizer: &Arc<dyn Tokenizer>,
) -> Result<ChatLogProbs, String> {
    let mut content_items = Vec::with_capacity(proto_logprobs.token_logprobs.len());

    // Decode token IDs to text (always with skip_special_tokens=false for logprobs)
    let token_texts: Vec<String> = proto_logprobs
        .token_ids
        .iter()
        .map(|&token_id| {
            tokenizer
                .decode(&[token_id as u32], false)
                .unwrap_or_else(|_| format!("<token_{}>", token_id))
        })
        .collect();

    // Build ChatLogProbsContent for each token (consume iterator to avoid clones)
    for (i, (&logprob, token_text)) in proto_logprobs
        .token_logprobs
        .iter()
        .zip(token_texts.into_iter())
        .enumerate()
    {
        let bytes = Some(token_text.as_bytes().to_vec());

        // Build top_logprobs for this position
        let top_logprobs = if let Some(top_logprobs_entry) = proto_logprobs.top_logprobs.get(i) {
            let mut top_logprobs = Vec::with_capacity(top_logprobs_entry.values.len());

            // Decode top token IDs (always with skip_special_tokens=false)
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

/// Convert OutputLogProbs to Generate format Vec<Vec<Option<f64>>>
///
/// Generate format: [[logprob, token_id, ...], [logprob, token_id, ...], ...]
/// Each inner vec contains [logprob (f64), token_id (i32), ...]
pub(crate) fn convert_generate_output_logprobs(
    proto_logprobs: &OutputLogProbs,
) -> Vec<Vec<Option<f64>>> {
    proto_logprobs
        .token_logprobs
        .iter()
        .zip(proto_logprobs.token_ids.iter())
        .map(|(&logprob, &token_id)| vec![Some(logprob as f64), Some(token_id as f64)])
        .collect()
}

/// Convert InputLogProbs to Generate format Vec<Vec<Option<f64>>>
///
/// Generate format: [[logprob, token_id, ...], [logprob, token_id, ...], ...]
/// First token has null logprob: [[null, token_id], [logprob, token_id], ...]
pub(crate) fn convert_generate_input_logprobs(
    proto_logprobs: &InputLogProbs,
) -> Vec<Vec<Option<f64>>> {
    proto_logprobs
        .token_logprobs
        .iter()
        .zip(proto_logprobs.token_ids.iter())
        .map(|(token_logprob, &token_id)| {
            // InputTokenLogProb has optional value field
            let logprob_value = token_logprob.value.map(|v| v as f64);
            vec![logprob_value, Some(token_id as f64)]
        })
        .collect()
}

pub(crate) use crate::routers::shared::metrics_utils::error_type_from_status;
