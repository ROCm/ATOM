use std::{sync::Arc, time::Duration};

use tokio::{
    sync::{mpsc, watch},
    time::{timeout, Instant},
};

use crate::app_context::AppContext;

use super::{
    admission::Admission,
    error::ProcessingError,
    executor::PdExecutor,
    lifecycle::RequestLifecycle,
    mutation::Mutation,
    pb,
    request::{RequestEnvelope, RequestParser},
    routing::EndpointRouter,
};

#[derive(PartialEq)]
enum Phase {
    Headers,
    RequestBody,
    ResponseHeaders,
    ResponseBody,
    Complete,
}

pub(super) struct Session {
    app: Arc<AppContext>,
    admission: Arc<Admission>,
    router: EndpointRouter,
    phase: Phase,
    parser: Arc<RequestParser>,
    request: Option<RequestEnvelope>,
    lifecycle: RequestLifecycle,
    output: mpsc::Sender<Result<pb::ProcessingResponse, tonic::Status>>,
}

impl Session {
    pub fn new(
        app: Arc<AppContext>,
        admission: Arc<Admission>,
        output: mpsc::Sender<Result<pb::ProcessingResponse, tonic::Status>>,
        executor: Option<Arc<PdExecutor>>,
        parser: Arc<RequestParser>,
    ) -> Self {
        Self {
            router: EndpointRouter::new(app.clone(), executor),
            app,
            parser,
            admission,
            output,
            phase: Phase::Headers,
            request: None,
            lifecycle: RequestLifecycle::new(),
        }
    }

    pub async fn run(
        mut self,
        mut input: tonic::Streaming<pb::ProcessingRequest>,
        mut force_stop: watch::Receiver<bool>,
    ) {
        let output = self.output.clone();
        tokio::select! {
            _ = output.closed() => {},
            _ = async { let _ = force_stop.wait_for(|stop| *stop).await; } => { self.lifecycle.finish("drain_timeout"); },
            result = self.process(&mut input) => {
                if let Err(error) = result {
                    self.lifecycle.finish(error.code);
                    metrics::counter!("mesh_ext_proc_errors_total", "reason" => error.code).increment(1);
                    let response = if self.lifecycle.response_started {
                        Err(tonic::Status::internal(error.to_string()))
                    } else { Ok(error.response()) };
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
        loop {
            let wait = if matches!(self.phase, Phase::Headers | Phase::RequestBody) {
                body_deadline.saturating_duration_since(Instant::now())
            } else {
                Duration::from_secs(self.app.router_config.ext_proc.idle_timeout_secs)
            };
            let next = timeout(wait, input.message())
                .await
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
            if message.observability_mode {
                return Err(ProcessingError::protocol(
                    "observability mode cannot route inference requests",
                ));
            }
            if let Some(config) = message.protocol_config {
                use super::proto::envoy::extensions::filters::http::ext_proc::v3::processing_mode::BodySendMode;
                if config.request_body_mode != BodySendMode::FullDuplexStreamed as i32
                    || config.response_body_mode != BodySendMode::FullDuplexStreamed as i32
                {
                    return Err(ProcessingError::protocol(
                        "request and response body modes must be FULL_DUPLEX_STREAMED",
                    ));
                }
            }
            use pb::processing_request::Request;
            match (message.request, &self.phase) {
                (Some(Request::RequestHeaders(headers)), Phase::Headers) => {
                    let end = headers.end_of_stream;
                    let mut request = RequestEnvelope::new(headers)?;
                    request.metadata(message.metadata_context)?;
                    self.request = Some(request);
                    if end {
                        return Err(ProcessingError::invalid(
                            "inference request requires a JSON body",
                        ));
                    }
                    self.phase = Phase::RequestBody;
                }
                (Some(Request::RequestBody(body)), Phase::RequestBody) => {
                    let request = self.request.as_mut().unwrap();
                    request.metadata(message.metadata_context)?;
                    request.append(&body.body, self.app.router_config.ext_proc.max_body_bytes)?;
                    if body.end_of_stream {
                        self.dispatch(false).await?;
                    }
                }
                (Some(Request::RequestTrailers(_)), Phase::RequestBody) => {
                    self.dispatch(true).await?;
                }
                (Some(Request::ResponseHeaders(headers)), Phase::ResponseHeaders) => {
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
                    self.send(Mutation::response_headers()).await?;
                    self.lifecycle.response_started = true;
                    self.phase = if headers.end_of_stream {
                        Phase::Complete
                    } else {
                        Phase::ResponseBody
                    };
                }
                (Some(Request::ResponseBody(body)), Phase::ResponseBody) => {
                    self.lifecycle.body(&body.body);
                    self.send_body(&body.body, body.end_of_stream, false)
                        .await?;
                    if body.end_of_stream {
                        self.phase = Phase::Complete;
                    }
                }
                (Some(Request::ResponseTrailers(_)), Phase::ResponseBody) => {
                    self.send(Mutation::trailers(false)).await?;
                    self.phase = Phase::Complete;
                }
                _ => return Err(ProcessingError::protocol("unexpected ext-proc phase")),
            }
            if self.phase == Phase::Complete {
                self.lifecycle.finish("completed");
                return Ok(());
            }
        }
    }

    async fn dispatch(&mut self, trailers: bool) -> Result<(), ProcessingError> {
        let started = Instant::now();
        let decision_timeout =
            Duration::from_secs(self.app.router_config.ext_proc.decision_timeout_secs);
        let request = self.request.take().unwrap();
        let (request, decision) = timeout(decision_timeout, async {
            let (mut request, input) = self.parser.parse(request).await?;
            let lease = self.admission.acquire().await?;
            self.lifecycle.admit(lease);
            let decision = self.router.select(&mut request, &input).await?;
            Ok::<_, ProcessingError>((request, decision))
        })
        .await
        .map_err(|_| {
            ProcessingError::new(504, "decision_timeout", "routing decision timed out")
        })??;
        metrics::histogram!("mesh_ext_proc_decision_seconds")
            .record(started.elapsed().as_secs_f64());
        let headers = Mutation::request_headers(
            &decision.address.to_string(),
            &request.id,
            request.raw.len(),
            decision.authorization.as_deref(),
            decision.target.execution_id(),
        );
        self.lifecycle.bind(decision);
        self.send(headers).await?;
        self.send_body(&request.raw, !trailers, true).await?;
        if trailers {
            self.send(Mutation::trailers(true)).await?;
        }
        self.phase = Phase::ResponseHeaders;
        Ok(())
    }

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
        timeout(
            Duration::from_secs(self.app.router_config.ext_proc.idle_timeout_secs),
            self.output.send(Ok(response)),
        )
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
