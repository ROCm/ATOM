//! `AtomEngineRouter`: the ATOM standalone frontend.
//!
//! prepare (template, tokenize) → `AtomEngine` (submit to the EngineCores) →
//! render (detokenize, parsers, SSE). Prepare and render are shared with the
//! gRPC router; ATOM's own parts turn a payload into a `Submission`
//! (`request`) and a sequence's events into `TokenChunk`s (`token_source`).

mod chat;
mod request;
mod token_source;

use std::sync::Arc;

use async_trait::async_trait;
use axum::{
    body::Body,
    extract::Request,
    http::{HeaderMap, StatusCode},
    response::{IntoResponse, Response},
    Json,
};
use serde_json::{json, Value};

use crate::{
    app_context::AppContext,
    atom_engine::AtomEngine,
    core::{prepare_pool::PrepareError, UNKNOWN_MODEL_ID},
    protocols::{
        chat::ChatCompletionRequest, completion::CompletionRequest, generate::GenerateRequest,
    },
    routers::{
        comm::error::{self, IngressError},
        grpc::completion_adapter::completion_to_generate,
        ingress::InferenceEnvelope,
        prepare::{
            self, generation_payload::GenerationPayload, inference::ParsedInference,
            response_context::ResponseContext, thread_tokenizer,
        },
        render::{self, completion::CompletionRender},
        RouterTrait,
    },
    tokenizer::{factory, registry::TokenizerRegistry},
};

/// Label for the render layer's streaming metrics.
const BACKEND_LABEL: &str = "atom_engine";

pub struct AtomEngineRouter {
    engine: Arc<AtomEngine>,
    ctx: Arc<AppContext>,
}

impl AtomEngineRouter {
    /// Load the model's tokenizer under its served name, which is the
    /// model id every request is prepared with, and the preparation
    /// workers' copies of it.
    pub async fn new(ctx: &Arc<AppContext>) -> Result<Self, String> {
        let engine = ctx
            .atom_engine
            .clone()
            .ok_or("ATOM standalone requires a started ATOM engine")?;
        let model = &engine.model;
        let source = model.tokenizer_path.clone();
        let chat_template = model.chat_template.clone();
        ctx.tokenizer_registry
            .load(
                &TokenizerRegistry::generate_id(),
                &model.served_name,
                &model.tokenizer_path,
                || async move {
                    factory::create_tokenizer_async_with_chat_template(
                        &source,
                        chat_template.as_deref(),
                    )
                    .await
                    .map_err(|e| format!("Failed to load tokenizer: {e}"))
                },
            )
            .await
            .map_err(|e| e.to_string())?;
        let entry = ctx
            .tokenizer_registry
            .get_by_name(&model.served_name)
            .ok_or("tokenizer vanished after loading")?;
        let workers = ctx.router_config.resolved_prepare_workers();
        tokio::task::spawn_blocking(move || thread_tokenizer::preload(&entry, workers))
            .await
            .map_err(|e| e.to_string())?;
        Ok(Self {
            engine,
            ctx: ctx.clone(),
        })
    }

    async fn chat(
        &self,
        headers: Option<&HeaderMap>,
        body: &ChatCompletionRequest,
        thinking: Option<&Value>,
    ) -> Result<Response, Response> {
        let admission = self.chat_request(body, thinking)?;
        let request = Arc::new(admission.request);
        let (headers, model_id, app) = (headers.cloned(), self.model_id(), self.ctx.clone());
        let (payload, ctx) = self
            .on_prepare_worker(move || prepare::prepare_chat(request, headers, model_id, &app))
            .await?;
        let stream = self.submit(&payload, &ctx, admission.required_tool_call)?;
        Ok(if payload.stream {
            render::chat_streaming::process(stream, ctx, BACKEND_LABEL)
        } else {
            render::chat_aggregator::process(stream, ctx).await
        })
    }

    async fn generate(
        &self,
        headers: Option<&HeaderMap>,
        body: &GenerateRequest,
    ) -> Result<Response, Response> {
        self.check_model(body.model.as_deref())?;
        let (payload, ctx) = self.prepare_generate(headers, body.clone()).await?;
        let stream = self.submit(&payload, &ctx, false)?;
        Ok(if payload.stream {
            render::generate_streaming::process(stream, ctx, BACKEND_LABEL)
        } else {
            render::generate_aggregator::process(stream, ctx).await
        })
    }

    async fn completion(
        &self,
        headers: Option<&HeaderMap>,
        body: &CompletionRequest,
    ) -> Result<Response, Response> {
        let generate = completion_to_generate(body)
            .map_err(|msg| error::bad_request("completion_unsupported_field", msg))?;
        self.check_model(generate.model.as_deref())?;
        let (payload, ctx) = self.prepare_generate(headers, generate).await?;
        let stream = self.submit(&payload, &ctx, false)?;
        let render = CompletionRender::new(body, self.engine.model.served_name.clone());
        Ok(if payload.stream {
            render.stream(stream, ctx)
        } else {
            render.aggregate(stream, ctx).await
        })
    }

    async fn prepare_generate(
        &self,
        headers: Option<&HeaderMap>,
        request: GenerateRequest,
    ) -> Result<(GenerationPayload, ResponseContext), Response> {
        let (headers, model_id, app) = (headers.cloned(), self.model_id(), self.ctx.clone());
        self.on_prepare_worker(move || {
            prepare::prepare_generate(Arc::new(request), headers, model_id, &app)
        })
        .await
    }

    /// Run `prepare` (template rendering, tokenization) on a preparation
    /// worker rather than an async one, with that worker's tokenizer copy.
    async fn on_prepare_worker<T: Send + 'static>(
        &self,
        prepare: impl FnOnce() -> Result<T, Response> + Send + 'static,
    ) -> Result<T, Response> {
        let pool = &self.ctx.prepare_pool;
        let refused = |e: PrepareError| {
            let status =
                StatusCode::from_u16(e.status_code()).unwrap_or(StatusCode::SERVICE_UNAVAILABLE);
            error::create_error(status, e.code(), e.to_string())
        };
        let ticket = pool
            .submit(pool.prepare_deadline(), move |_| prepare())
            .await
            .map_err(refused)?;
        ticket.wait().await.map_err(refused)?
    }

    fn check_model(&self, requested: Option<&str>) -> Result<(), Response> {
        let served = &self.engine.model.served_name;
        match requested {
            None => Ok(()),
            Some(name) if name == UNKNOWN_MODEL_ID || name == served => Ok(()),
            Some(name) => Err(error::bad_request(
                "model_not_found",
                format!("requested model `{name}` does not match loaded model `{served}`"),
            )),
        }
    }

    /// A chat body's `thinking`, which the typed request has no field for.
    /// The body is read again only when it names one.
    fn thinking(body: &[u8]) -> Result<Option<Value>, Response> {
        #[derive(serde::Deserialize)]
        struct Thinking {
            #[serde(default)]
            thinking: Option<Value>,
        }
        if memchr::memmem::find(body, b"\"thinking\"").is_none() {
            return Ok(None);
        }
        serde_json::from_slice::<Thinking>(body)
            .map(|t| t.thinking)
            .map_err(|e| error::bad_request("invalid_request", e.to_string()))
    }

    fn model_id(&self) -> Option<String> {
        Some(self.engine.model.served_name.clone())
    }

    fn model_card(&self) -> Value {
        json!({
            "id": self.engine.model.served_name,
            "object": "model",
            "owned_by": "atom",
            "max_model_len": self.engine.model.max_model_len,
        })
    }
}

impl std::fmt::Debug for AtomEngineRouter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AtomEngineRouter")
            .field("engine", &self.engine)
            .finish()
    }
}

#[async_trait]
impl RouterTrait for AtomEngineRouter {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    /// The typed routes, with the chat body's `thinking`; Messages and
    /// Responses are refused, not yet served.
    async fn route_inference(
        &self,
        request: InferenceEnvelope,
        _app: &Arc<AppContext>,
    ) -> Response {
        let thinking = match &request.parsed {
            ParsedInference::Chat(_) => Self::thinking(&request.body),
            _ => Ok(None),
        };
        let (uri, headers, parsed, _) = request.into_native_parts();
        let headers = Some(&headers);
        let routed = match parsed {
            ParsedInference::Chat(body) => match thinking {
                Ok(thinking) => self.chat(headers, &body, thinking.as_ref()).await,
                Err(e) => Err(e),
            },
            ParsedInference::Completion(body) => self.completion(headers, &body).await,
            ParsedInference::Generate(body) => self.generate(headers, &body).await,
            ParsedInference::Messages(_) | ParsedInference::Responses(_) => Err(IngressError::new(
                StatusCode::NOT_IMPLEMENTED,
                "unsupported_api",
                format!(
                    "{} is not supported by the ATOM standalone server yet",
                    uri.path()
                ),
            )
            .response(uri.path())),
        };
        routed.unwrap_or_else(|e| e)
    }

    async fn health_generate(&self, _req: Request<Body>) -> Response {
        let client = &self.engine.client;
        if client.alive_core_count() == client.core_count() {
            (StatusCode::OK, "All EngineCores are alive").into_response()
        } else {
            error::service_unavailable("engine_core_lost", "an EngineCore is gone")
        }
    }

    async fn get_server_info(&self, _req: Request<Body>) -> Response {
        let client = &self.engine.client;
        Json(json!({
            "router_type": self.router_type(),
            "model": self.engine.model.served_name,
            "engine_cores": client.core_count(),
            "alive_engine_cores": client.alive_core_count(),
            "max_pool_tokens": client.max_pool_tokens(),
            "inflight_requests_per_rank": client.rank_loads().0,
            "queued_prefill_tokens_per_rank": client.rank_loads().1,
        }))
        .into_response()
    }

    async fn get_models(&self, _req: Request<Body>) -> Response {
        Json(json!({ "object": "list", "data": [self.model_card()] })).into_response()
    }

    async fn get_model_info(&self, _req: Request<Body>) -> Response {
        Json(self.model_card()).into_response()
    }

    async fn route_generate(
        &self,
        headers: Option<&HeaderMap>,
        body: &GenerateRequest,
        _model_id: Option<&str>,
    ) -> Response {
        self.generate(headers, body).await.unwrap_or_else(|e| e)
    }

    async fn route_chat(
        &self,
        headers: Option<&HeaderMap>,
        body: &ChatCompletionRequest,
        _model_id: Option<&str>,
    ) -> Response {
        self.chat(headers, body, None).await.unwrap_or_else(|e| e)
    }

    async fn route_completion(
        &self,
        headers: Option<&HeaderMap>,
        body: &CompletionRequest,
        _model_id: Option<&str>,
    ) -> Response {
        self.completion(headers, body).await.unwrap_or_else(|e| e)
    }

    fn router_type(&self) -> &'static str {
        "atom_standalone"
    }

    fn is_pd_mode(&self) -> bool {
        false
    }
}
