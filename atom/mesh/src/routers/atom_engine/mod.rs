//! `AtomEngineRouter`: the ATOM standalone frontend.
//!
//! prepare (template, tokenize) → `AtomEngine` (submit to the EngineCores) →
//! render (detokenize, parsers, SSE). Prepare and render are shared with the
//! gRPC router; ATOM's own parts turn a payload into a `Submission`
//! (`request`) and a sequence's events into `TokenChunk`s (`token_source`).

mod chat;
mod request;
mod token_source;

use request::Admission;

use std::{
    sync::Arc,
    time::{SystemTime, UNIX_EPOCH},
};

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
        extras: &RawExtras,
    ) -> Result<Response, Response> {
        let chat = self.chat_request(body, extras.thinking.as_ref())?;
        let admission = Admission {
            required_tool_call: chat.required_tool_call,
            rank: extras.rank()?,
            ignore_logprobs: false,
        };
        let request = Arc::new(chat.request);
        let (headers, model_id, app) = (headers.cloned(), self.model_id(), self.ctx.clone());
        let (payload, ctx) = self
            .on_prepare_worker(move || prepare::prepare_chat(request, headers, model_id, &app))
            .await?;
        let stream = self.submit(&payload, &ctx, admission)?;
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
        let admission = Admission {
            rank: dp_rank(body.data_parallel_rank.map(i64::from))?,
            ..Default::default()
        };
        let (payload, ctx) = self.prepare_generate(headers, body.clone(), false).await?;
        let stream = self.submit(&payload, &ctx, admission)?;
        Ok(if payload.stream {
            render::generate_streaming::process(stream, ctx, BACKEND_LABEL)
        } else {
            render::generate_aggregator::process(stream, ctx).await
        })
    }

    /// `/v1/completions`, as the Python server serves it: the prompt gets the
    /// tokenizer's special tokens (BOS), and `logprobs` is ignored.
    async fn completion(
        &self,
        headers: Option<&HeaderMap>,
        body: &CompletionRequest,
        extras: &RawExtras,
    ) -> Result<Response, Response> {
        let mut generate = completion_to_generate(body).map_err(|msg| {
            // The adapter is the gRPC router's too, and says so.
            let msg = msg.replace("gRPC /v1/completions path", "ATOM standalone server");
            error::bad_request("completion_unsupported_field", msg)
        })?;
        self.check_model(generate.model.as_deref())?;
        let render = CompletionRender::new(self.engine.model.served_name.clone());
        generate.rid = Some(render.id().to_string());
        let admission = Admission {
            rank: extras.rank()?,
            ignore_logprobs: true,
            ..Default::default()
        };
        let (payload, ctx) = self.prepare_generate(headers, generate, true).await?;
        let stream = self.submit(&payload, &ctx, admission)?;
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
        add_special_tokens: bool,
    ) -> Result<(GenerationPayload, ResponseContext), Response> {
        let (headers, model_id, app) = (headers.cloned(), self.model_id(), self.ctx.clone());
        self.on_prepare_worker(move || {
            let request = Arc::new(request);
            prepare::prepare_generate_with(request, headers, model_id, &app, add_special_tokens)
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

    /// `validate_model`: the served name, a trailing `/` aside on either side.
    fn check_model(&self, requested: Option<&str>) -> Result<(), Response> {
        let served = &self.engine.model.served_name;
        match requested {
            None => Ok(()),
            Some(name)
                if name == UNKNOWN_MODEL_ID
                    || name.trim_end_matches('/') == served.trim_end_matches('/') =>
            {
                Ok(())
            }
            Some(name) => Err(error::bad_request(
                "model_not_found",
                format!("requested model `{name}` does not match loaded model `{served}`"),
            )),
        }
    }

    fn model_id(&self) -> Option<String> {
        Some(self.engine.model.served_name.clone())
    }

    /// api_server's `ModelCard`: the deployment's metadata under its name.
    fn model_card(&self) -> Value {
        let created = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or_default();
        let mut card = json!({
            "id": self.engine.model.served_name,
            "object": "model",
            "created": created,
            "owned_by": "atom",
        });
        if let Value::Object(card) = &mut card {
            card.extend(self.engine.model.model_card.clone());
        }
        card
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
        let extras = match &request.parsed {
            ParsedInference::Chat(_) | ParsedInference::Completion(_) => {
                RawExtras::read(&request.body)
            }
            _ => Ok(RawExtras::default()),
        };
        let (uri, headers, parsed, _) = request.into_native_parts();
        let headers = Some(&headers);
        let routed = match (parsed, extras) {
            (_, Err(e)) => Err(e),
            (ParsedInference::Chat(body), Ok(extras)) => self.chat(headers, &body, &extras).await,
            (ParsedInference::Completion(body), Ok(extras)) => {
                self.completion(headers, &body, &extras).await
            }
            (ParsedInference::Generate(body), _) => self.generate(headers, &body).await,
            (ParsedInference::Messages(_) | ParsedInference::Responses(_), _) => {
                Err(IngressError::new(
                    StatusCode::NOT_IMPLEMENTED,
                    "unsupported_api",
                    format!(
                        "{} is not supported by the ATOM standalone server yet",
                        uri.path()
                    ),
                )
                .response(uri.path()))
            }
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
        self.chat(headers, body, &RawExtras::default())
            .await
            .unwrap_or_else(|e| e)
    }

    async fn route_completion(
        &self,
        headers: Option<&HeaderMap>,
        body: &CompletionRequest,
        _model_id: Option<&str>,
    ) -> Response {
        self.completion(headers, body, &RawExtras::default())
            .await
            .unwrap_or_else(|e| e)
    }

    fn router_type(&self) -> &'static str {
        "atom_standalone"
    }

    fn is_pd_mode(&self) -> bool {
        false
    }
}

/// Request fields the typed chat / completion requests have no field for,
/// read from the body only when it names one of them.
#[derive(Debug, Default, serde::Deserialize)]
struct RawExtras {
    #[serde(default)]
    thinking: Option<Value>,
    #[serde(default)]
    data_parallel_rank: Option<Value>,
}

impl RawExtras {
    fn read(body: &[u8]) -> Result<Self, Response> {
        let names = |key: &[u8]| memchr::memmem::find(body, key).is_some();
        if !names(b"\"thinking\"") && !names(b"\"data_parallel_rank\"") {
            return Ok(Self::default());
        }
        serde_json::from_slice(body)
            .map_err(|e| error::bad_request("invalid_request", e.to_string()))
    }

    fn rank(&self) -> Result<Option<usize>, Response> {
        match &self.data_parallel_rank {
            None | Some(Value::Null) => Ok(None),
            Some(Value::Number(n)) if n.is_i64() => dp_rank(n.as_i64()),
            Some(_) => Err(error::bad_request(
                "invalid_request",
                "data_parallel_rank must be an integer",
            )),
        }
    }
}

/// A request's `data_parallel_rank`, refused when negative; the client
/// checks it against the engine count.
fn dp_rank(rank: Option<i64>) -> Result<Option<usize>, Response> {
    match rank {
        None => Ok(None),
        Some(r) if r >= 0 => Ok(Some(r as usize)),
        Some(r) => Err(error::bad_request(
            "invalid_request",
            format!("data_parallel_rank must be >= 0, got {r}"),
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn raw_extras_are_read_only_when_named_and_ranks_are_checked() {
        let none = RawExtras::read(br#"{"model": "m", "prompt": "data"}"#).unwrap();
        assert!(none.thinking.is_none() && none.data_parallel_rank.is_none());
        let extras =
            RawExtras::read(br#"{"data_parallel_rank": 2, "thinking": {"type": "enabled"}}"#)
                .unwrap();
        assert_eq!(extras.rank().unwrap(), Some(2));
        assert!(extras.thinking.is_some());
        let negative = RawExtras::read(br#"{"data_parallel_rank": -1}"#).unwrap();
        assert!(negative.rank().is_err());
        let text = RawExtras::read(br#"{"data_parallel_rank": "1"}"#).unwrap();
        assert!(text.rank().is_err());
    }
}
