//! Transport-neutral 4-step gRPC pipeline: prepare → plan → engine.dispatch → render.

use std::{
    pin::Pin,
    sync::Arc,
    task::{ready, Context, Poll},
    time::Instant,
};

use axum::response::Response;
use futures::Stream;
use http::HeaderMap;
use tracing::error;

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
        worker::RuntimeType,
        AttachedBody, Worker, WorkerLoadGuard, WorkerRegistry, UNKNOWN_MODEL_ID,
    },
    observability::{
        metrics::{bool_to_static_str, metrics_labels, MeshMetrics},
        route_trace,
    },
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
        token_handle::{
            engine_error::EngineError,
            token_chunk::TokenChunk,
            token_handle::{SourceClock, TokenHandle, TokenSource},
        },
    },
    tokenizer::StopSequenceDecoder,
};

#[derive(Clone)]
pub(crate) struct Pipeline {
    planner: Arc<dyn PdPlanner>,
    engine: Arc<dyn Dispatcher>,
    backend_label: &'static str,
    /// Every output read as ATOM's servers read it (`AtomOutput`), whatever produced
    /// it; otherwise only an ATOM worker's.
    atom_output: Option<Arc<AtomOutput>>,
}

/// The label an ATOM worker lists its end-of-sequence ids under, GetModelInfo's.
const EOS_TOKEN_IDS_LABEL: &str = "eos_token_ids";
/// The label, "true" or absent, of GetModelInfo's `usage_on_finish`.
const USAGE_ON_FINISH_LABEL: &str = "usage_on_finish";

/// ATOM's servers' reading of an output: the prompt's opener read first, no stop cut
/// (ATOM ends where a stop's ids match and returns it all), special tokens skipped and
/// end-of-sequence ids hidden.
pub(crate) struct AtomOutput {
    end_token_ids: Vec<u32>,
    /// A streamed chat's usage rides on its finish chunk.
    usage_on_finish: bool,
}

impl AtomOutput {
    pub(crate) fn new(end_token_ids: Vec<u32>) -> Self {
        Self {
            end_token_ids,
            usage_on_finish: false,
        }
    }

    /// An ATOM worker's: the end-of-sequence ids and usage placement its GetModelInfo gave.
    fn of_worker(worker: &dyn Worker) -> Self {
        let labels = &worker.metadata().labels;
        let listed = labels.get(EOS_TOKEN_IDS_LABEL);
        Self {
            usage_on_finish: labels
                .get(USAGE_ON_FINISH_LABEL)
                .is_some_and(|v| v == "true"),
            ..Self::new(
                listed
                    .and_then(|ids| serde_json::from_str(ids).ok())
                    .unwrap_or_default(),
            )
        }
    }

    /// The request's decoder: none of its stops, special tokens skipped whatever it asks
    /// (as ATOM's HTTP routes decode), and the end-of-sequence ids hidden unless it asked
    /// to run past them.
    fn stop_decoder(&self, ctx: &ResponseContext) -> StopSequenceDecoder {
        let ignore_eos = match &ctx.original {
            ProtocolRequest::Chat(req) => req.ignore_eos,
            ProtocolRequest::Generate(req) => req
                .sampling_params
                .as_ref()
                .and_then(|p| p.ignore_eos)
                .unwrap_or(false),
        };
        let end_token_ids = (!ignore_eos).then_some(&self.end_token_ids);
        create_stop_decoder(&ctx.tokenizer, None, end_token_ids, true, false)
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

    /// Read every output as ATOM's servers read it (`AtomOutput`).
    #[cfg_attr(not(test), allow(dead_code))] // for an in-process ATOM front end
    pub(crate) fn reading_output_as_atom(mut self, atom_output: AtomOutput) -> Self {
        self.atom_output = Some(Arc::new(atom_output));
        self
    }

    /// Give a placed request the decoder `AtomOutput` reads with, if its producer's
    /// output is ATOM's (D's for a pair); any other keeps the router's reading.
    fn read_as_atom(&self, plan: &PlacementPlan, ctx: &mut ResponseContext) {
        let producer = match plan {
            PlacementPlan::Single { worker, .. } => worker,
            PlacementPlan::Pair { decode, .. } => decode,
        };
        let atom = match &self.atom_output {
            Some(atom) => Arc::clone(atom),
            None if producer.metadata().runtime_type == RuntimeType::Atom => {
                take_parsers(producer.as_ref(), ctx);
                Arc::new(AtomOutput::of_worker(producer.as_ref()))
            }
            None => {
                ctx.prompt_opener = None;
                return;
            }
        };
        ctx.stop_decoder = atom.stop_decoder(ctx);
        ctx.reads_as_atom = true;
        ctx.usage_on_finish = atom.usage_on_finish;
    }
}

/// The parsers an ATOM `worker` names (GetModelInfo's, or its registration's) where the
/// router names none, so the reply is not parsed by the model name it was requested under.
fn take_parsers(worker: &dyn Worker, ctx: &mut ResponseContext) {
    if ctx.configured_reasoning_parser.is_none() {
        ctx.configured_reasoning_parser = worker.reasoning_parser().map(str::to_string);
    }
    if ctx.configured_tool_parser.is_none() {
        ctx.configured_tool_parser = worker.tool_parser().map(str::to_string);
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

    /// Dispatches through `engine`, which the router built from its config.
    #[cfg(feature = "grpc-client")]
    pub(crate) fn with_engine(self, engine: GrpcEngine) -> Self {
        Self {
            engine: Arc::new(engine),
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
        let (guards, stream) = match self.dispatch(&mut payload, &mut resp_ctx).await {
            Ok(dispatched) => dispatched,
            Err(e) => return self.record_chat_err(&model, start, e),
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
        let (guards, stream) = match self.dispatch(&mut payload, &mut resp_ctx).await {
            Ok(dispatched) => dispatched,
            Err(e) => return self.record_generate_err(&model_for_metric, start, e),
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

        let (mut payload, mut resp_ctx) =
            prepare::prepare_chat(req, headers, model_id, &components)?;
        let (_guards, stream) = self.dispatch(&mut payload, &mut resp_ctx).await?;

        render::chat_aggregator::process_typed(stream, resp_ctx).await
    }

    /// Plan the request, read its output as its producer's, and dispatch it: the loads
    /// held until the response ends, and its stream, whose cached tokens are reconciled
    /// with the estimate the plan's selection made on the worker computing the prompt.
    async fn dispatch(
        &self,
        payload: &mut GenerationPayload,
        ctx: &mut ResponseContext,
    ) -> Result<(Vec<WorkerLoadGuard>, TokenHandle), Response> {
        let (dispatched, estimate) = route_trace::estimated(async {
            let placement = self.plan_for(payload, ctx).await?;
            self.read_as_atom(&placement, ctx);
            let guards = self.load_guards(&placement, ctx.headers.as_ref());
            let stream = self
                .engine
                .dispatch_with_tokenizer(&placement, payload, &ctx.tokenizer)
                .await
                .map_err(engine_err_to_response)?;
            Ok::<_, Response>((placement, guards, stream))
        })
        .await;
        let (placement, guards, stream) = dispatched?;
        // A pair's cached tokens are the ones its prefill reused.
        let pool = match placement {
            PlacementPlan::Single { .. } => metrics_labels::WORKER_REGULAR,
            PlacementPlan::Pair { .. } => metrics_labels::WORKER_PREFILL,
        };
        let reconciled = Reconciled {
            inner: stream,
            pool,
            estimate,
            done: false,
        };
        Ok((guards, TokenHandle::new(reconciled)))
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

/// A stream that records its first Complete's cached prompt tokens against `pool`, and
/// against the router's `estimate` of them, as the HTTP routers reconcile a response's usage.
struct Reconciled {
    inner: TokenHandle,
    pool: &'static str,
    estimate: Option<f64>,
    done: bool,
}

impl Stream for Reconciled {
    type Item = Result<TokenChunk, EngineError>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let item = ready!(Pin::new(&mut self.inner).poll_next(cx));
        if let Some(Ok(TokenChunk::Complete { usage, meta, .. })) = &item {
            // n siblings share one prompt, as their first reports it.
            if !std::mem::replace(&mut self.done, true) {
                let (prompt, cached) = (usage.prompt_tokens.into(), meta.cached_tokens.into());
                MeshMetrics::record_route_cache_tokens(self.pool, prompt, cached);
                if let Some(ratio) = self.estimate {
                    let estimated = (ratio * prompt as f64).round() as u64;
                    MeshMetrics::record_route_cache_estimate(self.pool, prompt, estimated, cached);
                }
            }
        }
        Poll::Ready(item)
    }
}

impl TokenSource for Reconciled {
    fn mark_completed(&mut self) {
        self.inner.mark_completed();
    }

    fn clock(&self) -> Option<SourceClock> {
        self.inner.clock()
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
        core::BasicWorkerBuilder,
        protocols::generate::GenerateRequest,
        routers::test_mocks::{
            dispatcher::MockDispatcher, planner::MockPdPlanner, tokenizer::byte_level_tokenizer_dir,
        },
        tokenizer::{
            huggingface::HuggingFaceTokenizer, stop::SequenceDecoderOutput, traits::Tokenizer,
        },
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
            params.and_then(|p| p.skip_special_tokens).unwrap_or(true),
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
            prompt_opener: Some("<think>".to_string()),
            reads_as_atom: false,
            usage_on_finish: false,
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

    fn pipeline() -> Pipeline {
        Pipeline::with_injected(
            Arc::new(MockPdPlanner::new(vec![])),
            Arc::new(MockDispatcher::new(vec![])),
            "regular",
        )
    }

    fn single(runtime: RuntimeType, labels: &[(&str, &str)]) -> PlacementPlan {
        let mut builder = BasicWorkerBuilder::new("grpc://w:1").runtime_type(runtime);
        for (key, value) in labels {
            builder = builder.label(*key, *value);
        }
        PlacementPlan::Single {
            worker: Arc::new(builder.build()),
            policy_name: "random",
            prefill_tokens: 0,
        }
    }

    fn byte_level() -> Arc<dyn Tokenizer> {
        let dir = byte_level_tokenizer_dir(&[], false, serde_json::json!({}));
        let path = dir.path().join("tokenizer.json");
        Arc::new(HuggingFaceTokenizer::from_file(path.to_str().unwrap()).unwrap())
    }

    // "<tool>", an added token that is not special, ends the output: only the end
    // ids' hiding keeps it out.
    const EOS: u32 = 260;
    // "</s>", a special token.
    const SPECIAL: u32 = 257;
    const STOPPED_AT_DOT: &str =
        r#"{"text": "hi", "sampling_params": {"stop": ["."], "skip_special_tokens": false}}"#;

    fn output() -> Vec<u32> {
        let ids: Vec<u32> = "ab. c".bytes().map(u32::from).collect();
        [ids.as_slice(), &[EOS]].concat()
    }

    #[test]
    fn atom_output_keeps_the_text_past_a_stop_and_hides_the_end_of_sequence() {
        let tokenizer = byte_level();
        let output = output();
        let plan = single(RuntimeType::Sglang, &[]);

        let mut ctx = generate_ctx(&tokenizer, STOPPED_AT_DOT);
        pipeline().read_as_atom(&plan, &mut ctx);
        // The router's own reading cuts at the stop, and keeps no opener.
        assert_eq!(decode(&mut ctx.stop_decoder, &output), "ab");
        assert_eq!(ctx.prompt_opener, None);
        assert!(!ctx.reads_as_atom);

        let atom = pipeline().reading_output_as_atom(AtomOutput::new(vec![EOS]));
        let mut ctx = generate_ctx(&tokenizer, STOPPED_AT_DOT);
        atom.read_as_atom(&plan, &mut ctx);
        // The engine stopped where the stop's ids matched; the text is all of it.
        assert_eq!(decode(&mut ctx.stop_decoder, &output), "ab. c");
        assert_eq!(ctx.prompt_opener.as_deref(), Some("<think>"));
        assert!(ctx.reads_as_atom);
    }

    #[test]
    fn an_atom_reading_skips_special_tokens_as_atom_s_http_routes_do() {
        // Both requests ask for special tokens kept; ATOM's HTTP routes skip them.
        let tokenizer = byte_level();
        let output = [u32::from(b'a'), SPECIAL, u32::from(b'b')];
        let chat = || {
            let mut ctx = chat_ctx(&tokenizer, "m", None);
            let req = ChatCompletionRequest {
                skip_special_tokens: false,
                ..Default::default()
            };
            ctx.original = ProtocolRequest::Chat(Arc::new(req));
            ctx.stop_decoder = create_stop_decoder(&tokenizer, None, None, false, false);
            ctx
        };
        let generate = || generate_ctx(&tokenizer, STOPPED_AT_DOT);
        for request in [&chat as &dyn Fn() -> ResponseContext, &generate] {
            let mut ctx = request();
            pipeline().read_as_atom(&single(RuntimeType::Sglang, &[]), &mut ctx);
            assert_eq!(decode(&mut ctx.stop_decoder, &output), "a</s>b");
            let mut ctx = request();
            pipeline().read_as_atom(&single(RuntimeType::Atom, &[]), &mut ctx);
            assert_eq!(decode(&mut ctx.stop_decoder, &output), "ab");
        }
    }

    #[test]
    fn an_atom_producer_is_read_with_the_end_ids_its_worker_lists() {
        let tokenizer = byte_level();
        let output = output();
        let eos = EOS.to_string();
        let listed = format!("[{eos}]");
        let atom = single(RuntimeType::Atom, &[(EOS_TOKEN_IDS_LABEL, &listed)]);

        let mut ctx = generate_ctx(&tokenizer, STOPPED_AT_DOT);
        pipeline().read_as_atom(&atom, &mut ctx);
        assert_eq!(decode(&mut ctx.stop_decoder, &output), "ab. c");
        assert_eq!(ctx.prompt_opener.as_deref(), Some("<think>"));

        // ignore_eos runs past the end ids, so they stay in the text.
        let body = r#"{"text": "hi", "sampling_params": {"ignore_eos": true, "skip_special_tokens": false}}"#;
        let mut ctx = generate_ctx(&tokenizer, body);
        pipeline().read_as_atom(&atom, &mut ctx);
        assert_eq!(decode(&mut ctx.stop_decoder, &output), "ab. c<tool>");

        // A pair's output is its decode worker's.
        let PlacementPlan::Single { worker, .. } = atom else {
            unreachable!()
        };
        let PlacementPlan::Single {
            worker: prefill, ..
        } = single(RuntimeType::Sglang, &[])
        else {
            unreachable!()
        };
        let pair = PlacementPlan::Pair {
            prefill,
            decode: worker,
            prefill_policy: "random",
            decode_policy: "random",
            prefill_tokens: 0,
        };
        let mut ctx = generate_ctx(&tokenizer, STOPPED_AT_DOT);
        pipeline().read_as_atom(&pair, &mut ctx);
        assert_eq!(decode(&mut ctx.stop_decoder, &output), "ab. c");
    }

    #[test]
    fn an_atom_producer_places_streamed_usage_as_its_worker_says() {
        let tokenizer = byte_level();
        for (labels, usage_on_finish) in [
            (&[(USAGE_ON_FINISH_LABEL, "true")][..], true),
            (&[][..], false),
        ] {
            let mut ctx = generate_ctx(&tokenizer, STOPPED_AT_DOT);
            pipeline().read_as_atom(&single(RuntimeType::Atom, labels), &mut ctx);
            assert_eq!(ctx.usage_on_finish, usage_on_finish);
        }
    }

    /// A chat to `model` whose reply is read with the router's `configured` parser.
    fn chat_ctx(
        tokenizer: &Arc<dyn Tokenizer>,
        model: &str,
        configured: Option<&str>,
    ) -> ResponseContext {
        let req = ChatCompletionRequest {
            model: model.to_string(),
            separate_reasoning: true,
            ..Default::default()
        };
        ResponseContext {
            original: ProtocolRequest::Chat(Arc::new(req)),
            model_id: Some(model.to_string()),
            headers: None,
            original_text: None,
            processed_messages: None,
            tokenizer: Arc::clone(tokenizer),
            stop_decoder: create_stop_decoder(tokenizer, None, None, true, false),
            request_id: "chat-1".to_string(),
            created: 0,
            tool_parser_factory: Some(crate::tool_parser::ParserFactory::new()),
            reasoning_parser_factory: Some(reasoning_parser::ParserFactory::new()),
            configured_tool_parser: None,
            configured_reasoning_parser: configured.map(str::to_string),
            prompt_opener: None,
            reads_as_atom: false,
            usage_on_finish: false,
        }
    }

    #[tokio::test]
    async fn an_atom_reply_is_parsed_by_its_worker_s_parsers_under_any_model_name() {
        use crate::routers::{
            render::chat_aggregator,
            token_handle::test_support::synthetic_single_stream,
            token_handle::token_chunk::{FinishReason, TokenChunk, Usage, WorkerMeta},
        };
        let tokenizer = byte_level();
        let reply = tokenizer
            .encode("<think>plan</think>answer", false)
            .unwrap();
        let answer = |ctx: ResponseContext| {
            let complete = TokenChunk::Complete {
                token_ids: reply.token_ids().to_vec(),
                finish_reason: FinishReason::Stop,
                matched_stop: None,
                usage: Usage::default(),
                logprobs: None,
                input_logprobs: None,
                meta: WorkerMeta::default(),
            };
            let stream = synthetic_single_stream(vec![Ok(complete)]);
            async move {
                let resp = chat_aggregator::process_typed(stream, ctx).await.unwrap();
                let message = &resp.choices[0].message;
                (message.reasoning_content.clone(), message.content.clone())
            }
        };
        let worker = BasicWorkerBuilder::new("grpc://w:1")
            .runtime_type(RuntimeType::Atom)
            .reasoning_parser("qwen3")
            .build();
        let plan = PlacementPlan::Single {
            worker: Arc::new(worker),
            policy_name: "random",
            prefill_tokens: 0,
        };

        // Served under an alias that names no parser, the worker's splits the reply.
        let mut ctx = chat_ctx(&tokenizer, "served-alias", None);
        pipeline().read_as_atom(&plan, &mut ctx);
        assert_eq!(ctx.configured_reasoning_parser.as_deref(), Some("qwen3"));
        let split = (Some("plan".to_string()), Some("answer".to_string()));
        assert_eq!(answer(ctx).await, split);

        // The router's own parser wins.
        let mut ctx = chat_ctx(&tokenizer, "served-alias", Some("deepseek_r1"));
        pipeline().read_as_atom(&plan, &mut ctx);
        assert_eq!(
            ctx.configured_reasoning_parser.as_deref(),
            Some("deepseek_r1")
        );

        // Another runtime's worker leaves the model name to choose.
        let sglang = single(RuntimeType::Sglang, &[]);
        let mut ctx = chat_ctx(&tokenizer, "served-alias", None);
        pipeline().read_as_atom(&sglang, &mut ctx);
        assert_eq!(ctx.configured_reasoning_parser, None);
        let unsplit = (None, Some("<think>plan</think>answer".to_string()));
        assert_eq!(answer(ctx).await, unsplit);
    }
}
