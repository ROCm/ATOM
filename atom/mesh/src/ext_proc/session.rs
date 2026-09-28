use std::{future::Future, pin::Pin, sync::Arc, time::Duration};

use tokio::{
    sync::{mpsc, watch},
    time::{timeout, Instant},
};

use crate::app_context::AppContext;

use super::{
    admission::{Admission, AdmissionLease},
    error::ProcessingError,
    executor::PdExecutor,
    lifecycle::{Outcome, RequestLifecycle},
    mutation::Mutation,
    pb,
    request::{RequestEnvelope, RequestParser},
    routing::{EndpointRouter, RoutingDecision},
};

#[derive(Debug, PartialEq)]
enum Phase {
    Headers,
    RequestBody,
    Deciding,
    Dispatching,
    ResponseHeaders,
    ResponseBody,
    Complete,
}

struct PreparedRequest {
    request: RequestEnvelope,
    admission: AdmissionLease,
    decision: RoutingDecision,
    trailers: bool,
    observation: crate::observability::request::RequestMetrics,
}

type PendingDecision =
    Pin<Box<dyn Future<Output = Result<PreparedRequest, ProcessingError>> + Send>>;

type PendingAdmission =
    Pin<Box<dyn Future<Output = Result<AdmissionLease, ProcessingError>> + Send>>;

type PendingDispatch = Pin<Box<dyn Future<Output = Result<(), ProcessingError>> + Send>>;

#[derive(Clone)]
struct ResponseSender {
    channel: mpsc::Sender<Result<pb::ProcessingResponse, tonic::Status>>,
    timeout: Duration,
}

pub(super) struct Session {
    app: Arc<AppContext>,
    admission: Arc<Admission>,
    router: Arc<EndpointRouter>,
    phase: Phase,
    protocol: Option<pb::ProtocolConfiguration>,
    early_response: bool,
    parser: Arc<RequestParser>,
    request: Option<RequestEnvelope>,
    request_id: Option<String>,
    request_id_headers: Vec<String>,
    admitted: Option<AdmissionLease>,
    request_end: Option<bool>,
    lifecycle: RequestLifecycle,
    output: ResponseSender,
}

impl Session {
    pub fn new(
        app: Arc<AppContext>,
        admission: Arc<Admission>,
        output: mpsc::Sender<Result<pb::ProcessingResponse, tonic::Status>>,
        executor: Option<Arc<PdExecutor>>,
        parser: Arc<RequestParser>,
    ) -> Self {
        let output = ResponseSender {
            channel: output,
            timeout: Duration::from_secs(app.router_config.ext_proc.idle_timeout_secs),
        };
        Self {
            router: Arc::new(EndpointRouter::new(app.clone(), executor)),
            request_id_headers: crate::observability::request_id::header_names(
                app.router_config.request_id_headers.as_deref(),
            ),
            app,
            parser,
            admission,
            output,
            phase: Phase::Headers,
            protocol: None,
            early_response: false,
            request: None,
            request_id: None,
            admitted: None,
            request_end: None,
            lifecycle: RequestLifecycle::new(),
        }
    }

    pub async fn run(
        mut self,
        mut input: tonic::Streaming<pb::ProcessingRequest>,
        mut force_stop: watch::Receiver<bool>,
    ) {
        let output = self.output.channel.clone();
        tokio::select! {
            _ = output.closed() => {},
            _ = async { let _ = force_stop.wait_for(|stop| *stop).await; } => { self.lifecycle.finish(Outcome::Drained); },
            result = self.process(&mut input) => {
                if let Err(error) = result {
                    self.lifecycle.finish(if error.code == "proxy_disconnected" { Outcome::ProxyDisconnected } else { Outcome::Failed(error.code) });
                    metrics::counter!("mesh_ext_proc_errors_total", "reason" => error.code).increment(1);
                    tracing::warn!(code = error.code, phase = ?self.phase, error = %error, "ext-proc processing failed");
                    let response = if error.code == "unsupported_observability_mode" {
                        // Envoy ignores ProcessingResponses in observability mode.
                        Err(tonic::Status::failed_precondition(error.to_string()))
                    } else if self.lifecycle.response_started {
                        Err(tonic::Status::internal(error.to_string()))
                    } else { Ok(error.response(self.request_id.as_deref())) };
                    let _ = timeout(Duration::from_secs(1), output.send(response)).await;
                }
            }
        }
    }

    async fn process(
        &mut self,
        input: &mut tonic::Streaming<pb::ProcessingRequest>,
    ) -> Result<(), ProcessingError> {
        let body_deadline =
            Instant::now() + Duration::from_secs(self.app.router_config.ext_proc.body_timeout_secs);
        let mut admission: Option<PendingAdmission> = None;
        let mut pending: Option<PendingDecision> = None;
        let mut dispatch: Option<PendingDispatch> = None;
        loop {
            let wait = if matches!(self.phase, Phase::Headers | Phase::RequestBody) {
                body_deadline.saturating_duration_since(Instant::now())
            } else {
                Duration::from_secs(self.app.router_config.ext_proc.idle_timeout_secs)
            };
            let next = tokio::select! {
                // An already available local reply takes precedence over a
                // placement result. Dropping pending also drops its guards.
                biased;
                result = async { admission.as_mut().unwrap().await }, if admission.is_some() => {
                    admission = None;
                    self.admitted = Some(result?);
                    if let Some(trailers) = self.request_end { pending = Some(self.prepare(trailers)); }
                    continue;
                }
                _ = self.lifecycle.expiration() => {
                    return Err(ProcessingError::new(504, "execution_lease_timeout", "executor did not accept the reserved request before its deadline"));
                }
                next = timeout(wait, input.message()) => next,
                result = async { pending.as_mut().unwrap().await }, if pending.is_some() => {
                    pending = None;
                    dispatch = Some(self.dispatch(result?));
                    continue;
                }
                result = async { dispatch.as_mut().unwrap().await }, if dispatch.is_some() => {
                    dispatch = None;
                    result?;
                    self.phase = Phase::ResponseHeaders;
                    continue;
                }
            }
            .map_err(|_| {
                ProcessingError::new(
                    408,
                    "stream_timeout",
                    "external processing stream timed out",
                )
            })?
            .map_err(|_| {
                ProcessingError::new(502, "proxy_disconnected", "proxy stream disconnected")
            })?;
            let Some(message) = next else {
                return Err(ProcessingError::protocol(
                    "stream ended before the HTTP response completed",
                ));
            };
            use pb::processing_request::Request;
            // Once Envoy has a response, never fabricate an ImmediateResponse
            // over it, even if validating that response subsequently fails.
            if matches!(&message.request, Some(Request::ResponseHeaders(_))) {
                self.lifecycle.response_started = true;
            }
            if self.request_id.is_none() {
                if let Some(Request::RequestHeaders(headers)) = &message.request {
                    self.request_id = Some(RequestEnvelope::request_id(
                        headers,
                        &self.request_id_headers,
                    ));
                }
            }
            self.validate_protocol(&message)?;
            let attributes = message.attributes;
            if matches!(&message.request, Some(Request::ResponseHeaders(_))) {
                self.lifecycle.from_upstream = self.phase == Phase::ResponseHeaders;
            }
            // Preserve evidence even if acknowledging this message fails.
            self.lifecycle.observe_attributes(&attributes);
            match (message.request, &self.phase) {
                (Some(Request::RequestHeaders(headers)), Phase::Headers) => {
                    let end = headers.end_of_stream;
                    let mut request =
                        RequestEnvelope::new(headers, self.request_id.clone().unwrap())?;
                    request.set_budget(self.parser.budget.clone());
                    request.metadata(message.metadata_context)?;
                    self.request = Some(request);
                    if end {
                        return Err(ProcessingError::invalid(
                            "inference request requires a JSON body",
                        ));
                    }
                    self.phase = Phase::RequestBody;
                    let gate = self.admission.clone();
                    admission = Some(Box::pin(async move { gate.acquire().await }));
                }
                (Some(Request::RequestBody(body)), Phase::RequestBody) => {
                    let request = self.request.as_mut().unwrap();
                    request.metadata(message.metadata_context)?;
                    if self.request_end.is_some() {
                        return Err(ProcessingError::protocol("request body already ended"));
                    }
                    let config = &self.app.router_config.ext_proc;
                    if self.admitted.is_none()
                        && request.raw.len().saturating_add(body.body.len())
                            > config.max_message_bytes
                    {
                        return Err(ProcessingError::new(
                            429,
                            "admission_buffer_full",
                            "queued upload buffer exhausted",
                        ));
                    }
                    request.append(&body.body, config.max_body_bytes)?;
                    if body.end_of_stream {
                        self.request_end = Some(false);
                        if self.admitted.is_some() {
                            pending = Some(self.prepare(false));
                        }
                    }
                }
                (Some(Request::RequestTrailers(trailers)), Phase::RequestBody) => {
                    if self.request_end.is_some() {
                        return Err(ProcessingError::protocol("request body already ended"));
                    }
                    self.request.as_mut().unwrap().trailer_mutation =
                        Some(Mutation::filter_request_trailers(trailers));
                    self.request_end = Some(true);
                    if self.admitted.is_some() {
                        pending = Some(self.prepare(true));
                    }
                }
                (
                    Some(Request::ResponseHeaders(headers)),
                    Phase::Headers
                    | Phase::RequestBody
                    | Phase::Deciding
                    | Phase::Dispatching
                    | Phase::ResponseHeaders,
                ) => {
                    self.early_response = self.phase != Phase::ResponseHeaders;
                    admission = None;
                    self.admitted = None;
                    pending = None;
                    // Cancel an upload blocked by gRPC flow control before
                    // acknowledging Envoy's local response.
                    dispatch = None;
                    self.request = None;
                    for header in headers.headers.unwrap_or_default().headers {
                        let bytes = RequestEnvelope::header_bytes(&header);
                        if header.key == ":status" {
                            self.lifecycle.status =
                                std::str::from_utf8(bytes).ok().and_then(|v| v.parse().ok());
                        }
                        if header.key.eq_ignore_ascii_case("content-type") {
                            self.lifecycle.streaming = bytes.starts_with(b"text/event-stream");
                        }
                    }
                    if self.lifecycle.status.is_none() {
                        return Err(ProcessingError::protocol("response is missing :status"));
                    }
                    self.respond(
                        self.output
                            .send(Mutation::response_headers(self.request_id.as_deref())),
                    )
                    .await?;
                    self.phase = if headers.end_of_stream {
                        Phase::Complete
                    } else {
                        Phase::ResponseBody
                    };
                }
                (Some(Request::ResponseBody(body)), Phase::ResponseBody) => {
                    self.lifecycle.body(&body.body);
                    self.respond(self.output.send_body(&body.body, body.end_of_stream, false))
                        .await?;
                    if body.end_of_stream {
                        self.phase = Phase::Complete;
                    }
                }
                (Some(Request::ResponseTrailers(_)), Phase::ResponseBody) => {
                    self.respond(self.output.send(Mutation::trailers(false, None)))
                        .await?;
                    self.phase = Phase::Complete;
                }
                (
                    Some(Request::RequestBody(_) | Request::RequestTrailers(_)),
                    Phase::ResponseBody,
                ) if self.early_response => {
                    // Full-duplex request data can already be in flight when
                    // Envoy generates a local reply. It must not restart routing
                    // or replace that reply with a processing-sequence error.
                }
                (None, _) => {
                    return Err(ProcessingError::protocol(
                        "ProcessingRequest.request is missing",
                    ));
                }
                (Some(request), phase) => {
                    let kind = match request {
                        Request::RequestHeaders(_) => "request_headers",
                        Request::RequestBody(_) => "request_body",
                        Request::RequestTrailers(_) => "request_trailers",
                        Request::ResponseHeaders(_) => "response_headers",
                        Request::ResponseBody(_) => "response_body",
                        Request::ResponseTrailers(_) => "response_trailers",
                    };
                    return Err(ProcessingError::protocol(format!(
                        "unexpected {kind} in {phase:?}"
                    )));
                }
            }
            if self.phase == Phase::Complete {
                self.lifecycle.finish(Outcome::Completed);
                return Ok(());
            }
        }
    }

    async fn respond(
        &self,
        send: impl Future<Output = Result<(), ProcessingError>>,
    ) -> Result<(), ProcessingError> {
        // A local response can arrive while a reserved upload is blocked by
        // flow control. Its acknowledgment must obey the reservation deadline too.
        tokio::select! {
            _ = self.lifecycle.expiration() => Err(ProcessingError::new(504, "execution_lease_timeout", "executor did not accept the reserved request before its deadline")),
            result = send => result,
        }
    }

    fn validate_protocol(
        &mut self,
        message: &pb::ProcessingRequest,
    ) -> Result<(), ProcessingError> {
        use super::proto::envoy::extensions::filters::http::ext_proc::v3::processing_mode::BodySendMode;

        if message.observability_mode {
            return Err(ProcessingError::new(
                500,
                "unsupported_observability_mode",
                "observability_mode must be false for inference routing",
            ));
        }
        let Some(config) = &message.protocol_config else {
            // The protocol sends this field only on the first message.
            return if self.protocol.is_some() {
                Ok(())
            } else {
                Err(ProcessingError::new(500, "protocol_config_missing",
                    "first ProcessingRequest must include protocol_config; use an Envoy version that reports FULL_DUPLEX_STREAMED modes"))
            };
        };
        if config.request_body_mode != BodySendMode::FullDuplexStreamed as i32
            || config.response_body_mode != BodySendMode::FullDuplexStreamed as i32
        {
            let mode = |value| {
                BodySendMode::try_from(value)
                    .map(|mode| mode.as_str_name().to_owned())
                    .unwrap_or_else(|_| format!("UNKNOWN({value})"))
            };
            let request_mode = mode(config.request_body_mode);
            let response_mode = mode(config.response_body_mode);
            return Err(ProcessingError::new(
                500,
                "unsupported_processing_mode",
                format!(
                    "request_body_mode={request_mode} and response_body_mode={response_mode}; \
                     both must be FULL_DUPLEX_STREAMED"
                ),
            ));
        }
        if self
            .protocol
            .as_ref()
            .is_some_and(|previous| previous != config)
        {
            return Err(ProcessingError::new(
                500,
                "protocol_config_changed",
                "protocol_config must not change during an ext-proc stream",
            ));
        }
        self.protocol = Some(*config);
        Ok(())
    }

    fn prepare(&mut self, trailers: bool) -> PendingDecision {
        let decision_timeout =
            Duration::from_secs(self.app.router_config.ext_proc.decision_timeout_secs);
        let request = self.request.take().unwrap();
        let parser = self.parser.clone();
        let admission = self
            .admitted
            .take()
            .expect("request admitted before parsing");
        let router = self.router.clone();
        let ingress_started = self.lifecycle.started;
        let backend = if self.app.router_config.mode.is_pd_mode() {
            "pd"
        } else {
            "regular"
        };
        self.phase = Phase::Deciding;
        Box::pin(async move {
            let started = Instant::now();
            let prepared = timeout(decision_timeout, async {
                let (mut request, input) = parser.parse(request).await?;
                let mut observation = crate::observability::request::RequestMetrics::new(
                    "ext_proc",
                    backend,
                    input
                        .model
                        .as_deref()
                        .unwrap_or(crate::core::UNKNOWN_MODEL_ID),
                    input.route,
                    input.stream,
                )
                .started_at(ingress_started);
                let decision = match router.select(&mut request, &input).await {
                    Ok(decision) => decision,
                    Err(error) => {
                        observation.finish(Some(error.status), Some(error.code));
                        return Err(error);
                    }
                };
                Ok(PreparedRequest {
                    request,
                    admission,
                    decision,
                    trailers,
                    observation,
                })
            })
            .await
            .map_err(|_| {
                ProcessingError::new(504, "decision_timeout", "routing decision timed out")
            })?;
            metrics::histogram!("mesh_ext_proc_decision_seconds")
                .record(started.elapsed().as_secs_f64());
            prepared
        })
    }

    fn dispatch(&mut self, prepared: PreparedRequest) -> PendingDispatch {
        let PreparedRequest {
            mut request,
            admission,
            decision,
            trailers,
            observation,
        } = prepared;
        self.lifecycle.observation = Some(observation);
        let headers = Mutation::request_headers(
            &request.headers,
            &decision.address.to_string(),
            &request.id,
            decision.authorization.as_deref(),
            decision.target.execution_id(),
        );
        self.lifecycle.admit(admission);
        self.lifecycle.bind(decision);
        self.phase = Phase::Dispatching;
        let output = self.output.clone();
        Box::pin(async move {
            output.send(headers).await?;
            output.send_body(&request.raw, !trailers, true).await?;
            if trailers {
                output
                    .send(Mutation::trailers(true, request.trailer_mutation.take()))
                    .await?;
            }
            Ok(())
        })
    }
}

impl ResponseSender {
    async fn send_body(
        &self,
        bytes: &[u8],
        end: bool,
        request: bool,
    ) -> Result<(), ProcessingError> {
        if bytes.is_empty() {
            return self.send(Mutation::body(Vec::new(), end, request)).await;
        }
        let count = bytes.len().div_ceil(Mutation::CHUNK_BYTES);
        for (index, chunk) in bytes.chunks(Mutation::CHUNK_BYTES).enumerate() {
            self.send(Mutation::body(
                chunk.to_vec(),
                end && index + 1 == count,
                request,
            ))
            .await?;
        }
        Ok(())
    }

    async fn send(&self, response: pb::ProcessingResponse) -> Result<(), ProcessingError> {
        timeout(self.timeout, self.channel.send(Ok(response)))
            .await
            .map_err(|_| {
                ProcessingError::new(
                    504,
                    "send_timeout",
                    "proxy is not reading processing responses",
                )
            })?
            .map_err(|_| {
                ProcessingError::new(502, "proxy_disconnected", "proxy response stream closed")
            })
    }
}
