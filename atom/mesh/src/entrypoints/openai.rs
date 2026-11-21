//! OpenAI handlers on top of prepare, [`Backend`], and render.

use std::sync::Arc;

use axum::response::Response;
use http::HeaderMap;

use super::backend::{Backend, TokenRequest, TokenStream};
use super::prepare::{self, response_context::ResponseContext};
use super::render;
use crate::{
    app_context::AppContext,
    core::AttachedBody,
    protocols::{chat::ChatCompletionRequest, generate::GenerateRequest},
};

pub async fn run_chat(
    backend: &dyn Backend,
    req: Arc<ChatCompletionRequest>,
    headers: Option<HeaderMap>,
    model_id: Option<String>,
    components: &AppContext,
    backend_label: &'static str,
) -> Result<Response, Response> {
    let (payload, mut ctx) = prepare::prepare_chat(req, headers, model_id, components)?;
    let stream = generate(backend, payload, &mut ctx).await?;
    Ok(render_chat(stream, ctx, backend_label).await)
}

pub async fn run_generate(
    backend: &dyn Backend,
    req: Arc<GenerateRequest>,
    headers: Option<HeaderMap>,
    model_id: Option<String>,
    components: &AppContext,
    backend_label: &'static str,
) -> Result<Response, Response> {
    let (payload, mut ctx) = prepare::prepare_generate(req, headers, model_id, components)?;
    let stream = generate(backend, payload, &mut ctx).await?;
    Ok(render_generate(stream, ctx, backend_label).await)
}

async fn generate(
    backend: &dyn Backend,
    payload: prepare::generation_payload::GenerationPayload,
    ctx: &mut ResponseContext,
) -> Result<TokenStream, Response> {
    backend
        .generate(TokenRequest::from_prepared(payload, ctx))
        .await
        .map_err(|error| error.into_response())
}

async fn render_chat(
    stream: TokenStream,
    ctx: ResponseContext,
    backend_label: &'static str,
) -> Response {
    let streaming = ctx.original.is_streaming();
    let (handle, finish) = stream.into_parts();
    let response = if streaming {
        render::chat_streaming::process(handle, ctx, backend_label)
    } else {
        render::chat_aggregator::process(handle, ctx).await
    };
    attach(response, finish)
}

async fn render_generate(
    stream: TokenStream,
    ctx: ResponseContext,
    backend_label: &'static str,
) -> Response {
    let streaming = ctx.original.is_streaming();
    let (handle, finish) = stream.into_parts();
    let response = if streaming {
        render::generate_streaming::process(handle, ctx, backend_label)
    } else {
        render::generate_aggregator::process(handle, ctx).await
    };
    attach(response, finish)
}

fn attach(response: Response, finish: Option<Box<dyn Send + 'static>>) -> Response {
    match finish {
        Some(finish) => AttachedBody::wrap_response(response, finish),
        None => response,
    }
}
