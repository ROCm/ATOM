//! Transport-neutral 4-step gRPC pipeline: prepare → plan → engine.dispatch → render.

use std::{sync::Arc, time::Instant};

use axum::response::Response;
use http::HeaderMap;
use tracing::error;

#[cfg(feature = "grpc-client")]
use super::engine::atom::pair::AtomPairs;
use super::engine::{Dispatcher, GrpcEngine};
use crate::{
    app_context::AppContext,
    core::{
        placement::{
            planner::DefaultPlanner,
            registry_adapters::{PolicyRegistryAdapter, WorkerRegistryAdapter},
            traits::PdPlanner,
            types::{PlacementError, PlacementPlan, Protocol, RequestDescriptor},
        },
        AttachedBody, WorkerLoadGuard, WorkerRegistry, UNKNOWN_MODEL_ID,
    },
    observability::metrics::{bool_to_static_str, metrics_labels, MeshMetrics},
    policies::PolicyRegistry,
    protocols::{
        chat::{ChatCompletionRequest, ChatCompletionResponse},
        generate::GenerateRequest,
    },
    routers::{
        comm::{
            error, metrics_utils::error_type_from_status,
            placement_response::placement_err_to_response,
        },
        prepare::{
            self,
            generation_payload::GenerationPayload,
            response_context::{ProtocolRequest, ResponseContext},
            stop_decoder_builder::create_stop_decoder,
        },
        render,
        token_handle::engine_error::EngineError,
    },
    tokenizer::{traits::Tokenizer, StopSequenceDecoder},
};

#[derive(Clone)]
pub(crate) struct Pipeline {
    planner: Arc<dyn PdPlanner>,
    engine: Arc<dyn Dispatcher>,
    backend_label: &'static str,
    /// The ATOM bridge's: its output read as ATOM's Python server reads it
    /// (`AtomOutput`). gRPC routing keeps its own reading.
    atom_output: Option<Arc<AtomOutput>>,
}

/// How ATOM's Python server reads the engine's output, where the router's
/// own reading differs:
/// - a chat the router renders has the channel opener its prompt ends on
///   read before the reply (`ResponseContext::prompt_opener`);
/// - the engine ends an output where a stop string's ids match and the
///   server returns all of it, so the router cuts at none of the stops;
/// - the server's decode leaves the model's end-of-sequence ids out of the
///   text, which the tiktoken decoder would not do by `skip_special_tokens`.
pub(crate) struct AtomOutput {
    end_token_ids: Vec<u32>,
}

impl AtomOutput {
    pub(crate) fn new(end_token_ids: Vec<u32>) -> Self {
        Self { end_token_ids }
    }

    /// The request's decoder: none of its stops, and the end-of-sequence ids
    /// hidden unless it asked to run past them.
    fn stop_decoder(&self, ctx: &ResponseContext) -> StopSequenceDecoder {
        let (skip_special_tokens, ignore_eos) = match &ctx.original {
            ProtocolRequest::Chat(req) => (req.skip_special_tokens, req.ignore_eos),
            ProtocolRequest::Generate(req) => {
                let params = req.sampling_params.as_ref();
                (
                    params.and_then(|p| p.skip_special_tokens).unwrap_or(true),
                    params.and_then(|p| p.ignore_eos).unwrap_or(false),
                )
            }
        };
        self.decoder(&ctx.tokenizer, skip_special_tokens, ignore_eos)
    }

    fn decoder(
        &self,
        tokenizer: &Arc<dyn Tokenizer>,
        skip_special_tokens: bool,
        ignore_eos: bool,
    ) -> StopSequenceDecoder {
        let end_token_ids = (!ignore_eos).then_some(&self.end_token_ids);
        create_stop_decoder(tokenizer, None, end_token_ids, skip_special_tokens, false)
    }
}

impl Pipeline {
    pub(crate) fn with_injected(
        planner: Arc<dyn PdPlanner>,
        engine: Arc<dyn Dispatcher>,
        backend_label: &'static str,
    ) -> Self {
        Self {
            planner,
            engine,
            backend_label,
            atom_output: None,
        }
    }

    /// Read the output as ATOM's Python server reads it (`AtomOutput`).
    pub(crate) fn reading_output_as_atom(mut self, atom_output: AtomOutput) -> Self {
        self.atom_output = Some(Arc::new(atom_output));
        self
    }

    /// Give a prepared request the decoder `AtomOutput` reads with, if any.
    fn read_as_atom(&self, ctx: &mut ResponseContext) {
        if let Some(atom) = &self.atom_output {
            ctx.stop_decoder = atom.stop_decoder(ctx);
        }
    }
}

impl Pipeline {
    pub fn new_regular(
        worker_registry: Arc<WorkerRegistry>,
        policy_registry: Arc<PolicyRegistry>,
    ) -> Self {
        Self::with_label(
            worker_registry,
            policy_registry,
            metrics_labels::BACKEND_REGULAR,
        )
    }

    pub fn new_pd(
        worker_registry: Arc<WorkerRegistry>,
        policy_registry: Arc<PolicyRegistry>,
    ) -> Self {
        Self::with_label(worker_registry, policy_registry, metrics_labels::BACKEND_PD)
    }

    /// Runs ATOM pairs through `pairs` rather than the dual dispatch.
    #[cfg(feature = "grpc-client")]
    pub(crate) fn with_atom_pairs(self, pairs: AtomPairs) -> Self {
        Self {
            engine: Arc::new(GrpcEngine::with_atom_pairs(pairs)),
            ..self
        }
    }

    fn with_label(
        worker_registry: Arc<WorkerRegistry>,
        policy_registry: Arc<PolicyRegistry>,
        backend_label: &'static str,
    ) -> Self {
        let planner: Arc<dyn PdPlanner> = Arc::new(DefaultPlanner::new(
            Arc::new(WorkerRegistryAdapter::new(worker_registry)),
            Arc::new(PolicyRegistryAdapter::new(policy_registry)),
        ));
        Self {
            planner,
            engine: Arc::new(GrpcEngine::new()),
            backend_label,
            atom_output: None,
        }
    }

    pub async fn execute_chat(
        &self,
        req: Arc<ChatCompletionRequest>,
        headers: Option<HeaderMap>,
        model_id: Option<String>,
        components: Arc<AppContext>,
    ) -> Response {
        self.execute_chat_with(req, headers, model_id, components, None)
            .await
    }

    /// `execute_chat`, rendering the prompt with the chat's body as `sent`
    /// where the caller has it (`prepare::prepare_chat_with`).
    pub async fn execute_chat_with(
        &self,
        req: Arc<ChatCompletionRequest>,
        headers: Option<HeaderMap>,
        model_id: Option<String>,
        components: Arc<AppContext>,
        sent: Option<Arc<serde_json::Map<String, serde_json::Value>>>,
    ) -> Response {
        let start = Instant::now();
        let model = req.model.clone();
        let streaming = req.stream;
        MeshMetrics::record_router_request(
            metrics_labels::ROUTER_GRPC,
            self.backend_label,
            metrics_labels::CONNECTION_GRPC,
            &model,
            metrics_labels::ENDPOINT_CHAT,
            bool_to_static_str(streaming),
        );

        let (mut payload, mut resp_ctx) = match prepare::prepare_chat_with(
            req,
            headers,
            model_id,
            &components,
            sent.as_deref(),
            self.atom_output.is_some(),
        ) {
            Ok(t) => t,
            Err(e) => return self.record_chat_err(&model, start, e),
        };
        self.read_as_atom(&mut resp_ctx);
        let placement = match self.plan_for(&payload, &resp_ctx).await {
            Ok(p) => p,
            Err(e) => return self.record_chat_err(&model, start, e),
        };
        let guards = self.load_guards(&placement, resp_ctx.headers.as_ref());
        let stream = match self
            .engine
            .dispatch_with_tokenizer(&placement, &mut payload, &resp_ctx.tokenizer)
            .await
        {
            Ok(s) => s,
            Err(e) => return self.record_chat_err(&model, start, engine_err_to_response(e)),
        };

        let is_stream = resp_ctx.original.is_streaming();
        let response = if is_stream {
            render::chat_streaming::process(stream, resp_ctx, self.backend_label)
        } else {
            render::chat_aggregator::process(stream, resp_ctx).await
        };
        let response = AttachedBody::wrap_response(response, guards);

        MeshMetrics::record_router_duration(
            metrics_labels::ROUTER_GRPC,
            self.backend_label,
            metrics_labels::CONNECTION_GRPC,
            &model,
            metrics_labels::ENDPOINT_CHAT,
            start.elapsed(),
        );
        response
    }

    pub async fn execute_generate(
        &self,
        req: Arc<GenerateRequest>,
        headers: Option<HeaderMap>,
        model_id: Option<String>,
        components: Arc<AppContext>,
    ) -> Response {
        let start = Instant::now();
        let model_for_metric = model_id
            .clone()
            .unwrap_or_else(|| UNKNOWN_MODEL_ID.to_string());
        let streaming = req.stream;
        MeshMetrics::record_router_request(
            metrics_labels::ROUTER_GRPC,
            self.backend_label,
            metrics_labels::CONNECTION_GRPC,
            &model_for_metric,
            metrics_labels::ENDPOINT_GENERATE,
            bool_to_static_str(streaming),
        );

        let (mut payload, mut resp_ctx) =
            match prepare::prepare_generate(req, headers, model_id, &components) {
                Ok(t) => t,
                Err(e) => return self.record_generate_err(&model_for_metric, start, e),
            };
        self.read_as_atom(&mut resp_ctx);
        let placement = match self.plan_for(&payload, &resp_ctx).await {
            Ok(p) => p,
            Err(e) => return self.record_generate_err(&model_for_metric, start, e),
        };
        let guards = self.load_guards(&placement, resp_ctx.headers.as_ref());
        let stream = match self
            .engine
            .dispatch_with_tokenizer(&placement, &mut payload, &resp_ctx.tokenizer)
            .await
        {
            Ok(s) => s,
            Err(e) => {
                return self.record_generate_err(
                    &model_for_metric,
                    start,
                    engine_err_to_response(e),
                )
            }
        };

        let is_stream = resp_ctx.original.is_streaming();
        let response = if is_stream {
            render::generate_streaming::process(stream, resp_ctx, self.backend_label)
        } else {
            render::generate_aggregator::process(stream, resp_ctx).await
        };
        let response = AttachedBody::wrap_response(response, guards);

        MeshMetrics::record_router_duration(
            metrics_labels::ROUTER_GRPC,
            self.backend_label,
            metrics_labels::CONNECTION_GRPC,
            &model_for_metric,
            metrics_labels::ENDPOINT_GENERATE,
            start.elapsed(),
        );
        response
    }

    pub async fn execute_chat_for_responses(
        &self,
        req: Arc<ChatCompletionRequest>,
        headers: Option<HeaderMap>,
        model_id: Option<String>,
        components: Arc<AppContext>,
    ) -> Result<ChatCompletionResponse, Response> {
        if req.stream {
            return Err(error::bad_request(
                "streaming_not_supported",
                "Streaming is not supported in this context".to_string(),
            ));
        }

        let (mut payload, resp_ctx) = prepare::prepare_chat(req, headers, model_id, &components)?;
        let placement = self.plan_for(&payload, &resp_ctx).await?;
        let _guards = self.load_guards(&placement, resp_ctx.headers.as_ref());
        let stream = self
            .engine
            .dispatch_with_tokenizer(&placement, &mut payload, &resp_ctx.tokenizer)
            .await
            .map_err(engine_err_to_response)?;

        render::chat_aggregator::process_typed(stream, resp_ctx).await
    }

    /// The loads held until the response ends; none when dispatch holds them.
    fn load_guards(
        &self,
        plan: &PlacementPlan,
        headers: Option<&HeaderMap>,
    ) -> Vec<WorkerLoadGuard> {
        if self.engine.holds_load(plan) {
            return Vec::new();
        }
        make_load_guards(plan, headers)
    }

    async fn plan_for(
        &self,
        payload: &GenerationPayload,
        resp_ctx: &ResponseContext,
    ) -> Result<PlacementPlan, Response> {
        let tokens = if payload.token_ids.is_empty() {
            None
        } else {
            Some(payload.token_ids.as_slice())
        };
        let descriptor = RequestDescriptor {
            model_id: resp_ctx.model_id.as_deref(),
            protocol: Some(Protocol::Grpc),
            text: resp_ctx.original_text.as_deref(),
            tokens,
            headers: resp_ctx.headers.as_ref(),
            stream: resp_ctx.original.is_streaming(),
            // Image routing keys are HTTP chat only: this path routes on
            // the text and tokens alone.
            media_count: 0,
            chat_template: payload.chat_template.as_ref(),
        };
        self.planner
            .plan(&descriptor)
            .await
            .map_err(|e| placement_err_to_response_log(e, resp_ctx.model_id.as_deref()))
    }

    fn record_chat_err(&self, model: &str, _start: Instant, resp: Response) -> Response {
        MeshMetrics::record_router_error(
            metrics_labels::ROUTER_GRPC,
            self.backend_label,
            metrics_labels::CONNECTION_GRPC,
            model,
            metrics_labels::ENDPOINT_CHAT,
            error_type_from_status(resp.status()),
        );
        resp
    }

    fn record_generate_err(&self, model: &str, _start: Instant, resp: Response) -> Response {
        MeshMetrics::record_router_error(
            metrics_labels::ROUTER_GRPC,
            self.backend_label,
            metrics_labels::CONNECTION_GRPC,
            model,
            metrics_labels::ENDPOINT_GENERATE,
            error_type_from_status(resp.status()),
        );
        resp
    }
}

fn make_load_guards(plan: &PlacementPlan, headers: Option<&HeaderMap>) -> Vec<WorkerLoadGuard> {
    match plan {
        PlacementPlan::Single { worker, .. } => {
            vec![WorkerLoadGuard::new(worker.clone(), headers)]
        }
        PlacementPlan::Pair {
            prefill, decode, ..
        } => vec![
            WorkerLoadGuard::new(prefill.clone(), headers),
            WorkerLoadGuard::new(decode.clone(), headers),
        ],
    }
}

fn placement_err_to_response_log(err: PlacementError, model_id: Option<&str>) -> Response {
    error!(
        function = "Pipeline::plan_for",
        model_id = %model_id.unwrap_or(UNKNOWN_MODEL_ID),
        error = %err,
        "Placement planner returned error"
    );
    placement_err_to_response(err, model_id)
}

fn engine_err_to_response(err: EngineError) -> Response {
    match err {
        EngineError::Transport(s) => {
            error::service_unavailable("engine_transport_error", format!("transport error: {}", s))
        }
        EngineError::Prefill(m) => {
            error::internal_error("engine_prefill_error", format!("prefill error: {}", m))
        }
        EngineError::DecodeError(m) => {
            error::internal_error("engine_decode_error", format!("decode error: {}", m))
        }
        EngineError::PrefillEarlyClose => error::internal_error(
            "engine_prefill_early_close",
            "prefill stream closed without Complete",
        ),
        EngineError::DecodeIncomplete => error::internal_error(
            "engine_decode_incomplete",
            "decode stream closed without Complete",
        ),
        EngineError::ConnectionAcquireFailed(r) => {
            error::service_unavailable("engine_connection_failed", r)
        }
        EngineError::RequestBuildFailed(r) => error::bad_request("engine_request_build_failed", r),
        EngineError::Http { status, message } => {
            error::create_error(status, "engine_error", message)
        }
    }
}

#[cfg(test)]
mod atom_output_tests {
    use super::*;
    use crate::{
        protocols::generate::GenerateRequest,
        routers::test_mocks::{
            dispatcher::MockDispatcher, planner::MockPdPlanner, tokenizer::byte_level_tokenizer_dir,
        },
        tokenizer::{huggingface::HuggingFaceTokenizer, stop::SequenceDecoderOutput},
    };

    /// A `/generate` of `body`, prepared as the router prepares it: its own
    /// stops in its decoder.
    fn generate_ctx(tokenizer: &Arc<dyn Tokenizer>, body: &str) -> ResponseContext {
        let req: GenerateRequest = serde_json::from_str(body).unwrap();
        let params = req.sampling_params.as_ref();
        let stop_decoder = create_stop_decoder(
            tokenizer,
            params.and_then(|p| p.stop.as_ref()),
            None,
            true,
            false,
        );
        ResponseContext {
            original: ProtocolRequest::Generate(Arc::new(req)),
            model_id: None,
            headers: None,
            original_text: None,
            processed_messages: None,
            tokenizer: Arc::clone(tokenizer),
            stop_decoder,
            request_id: "gen-1".to_string(),
            created: 0,
            tool_parser_factory: None,
            reasoning_parser_factory: None,
            configured_tool_parser: None,
            configured_reasoning_parser: None,
            prompt_opener: None,
        }
    }

    /// The text `decoder` makes of `ids`.
    fn decode(decoder: &mut StopSequenceDecoder, ids: &[u32]) -> String {
        let mut text = String::new();
        for &id in ids {
            match decoder.process_token(id).unwrap() {
                SequenceDecoderOutput::Text(t) | SequenceDecoderOutput::StoppedWithText(t) => {
                    text.push_str(&t)
                }
                SequenceDecoderOutput::Stopped | SequenceDecoderOutput::Held => {}
            }
        }
        if let SequenceDecoderOutput::Text(t) = decoder.flush() {
            text.push_str(&t);
        }
        text
    }

    #[test]
    fn atom_output_keeps_the_text_past_a_stop_and_hides_the_end_of_sequence() {
        let dir = byte_level_tokenizer_dir(&[], false, serde_json::json!({}));
        let path = dir.path().join("tokenizer.json");
        let tokenizer: Arc<dyn Tokenizer> =
            Arc::new(HuggingFaceTokenizer::from_file(path.to_str().unwrap()).unwrap());
        // "</s>", the second token added after the 256 bytes, ends the output.
        let eos = 257;
        let ids: Vec<u32> = "ab. c".bytes().map(u32::from).collect();
        let output = [ids.as_slice(), &[eos]].concat();
        let pipeline = Pipeline::with_injected(
            Arc::new(MockPdPlanner::new(vec![])),
            Arc::new(MockDispatcher::new(vec![])),
            "regular",
        );

        let mut ctx = generate_ctx(
            &tokenizer,
            r#"{"text": "hi", "sampling_params": {"stop": ["."]}}"#,
        );
        // The router's own reading cuts at the stop.
        assert_eq!(decode(&mut ctx.stop_decoder, &output), "ab");

        let atom = pipeline.reading_output_as_atom(AtomOutput::new(vec![eos]));
        let mut ctx = generate_ctx(
            &tokenizer,
            r#"{"text": "hi", "sampling_params": {"stop": ["."]}}"#,
        );
        atom.read_as_atom(&mut ctx);
        // The engine stopped where the stop's ids matched; the text is all of it.
        assert_eq!(decode(&mut ctx.stop_decoder, &output), "ab. c");
    }
}
