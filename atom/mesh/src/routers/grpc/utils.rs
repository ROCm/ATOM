//! Shared utilities for gRPC routers

use std::sync::Arc;

use axum::response::Response;
use tracing::error;

use super::context::RequestContext;
use super::engine::proto_stream_wrapper::{ProtoGenerateComplete, ProtoStream};
use crate::{
    routers::{error, grpc::engine::proto_stream_wrapper::ProtoResponseVariant},
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

pub(crate) use crate::routers::shared::metrics_utils::error_type_from_status;
