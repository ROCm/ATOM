//! One business observation per request, shared by HTTP and ext-proc.
//! The owner lives through body completion or cancellation, outside backend retries.
use super::{
    metrics::{bool_to_static_str, metrics_labels, MeshMetrics},
    ttft::FirstOutputSse,
    usage::UsageObserver,
};
use crate::{
    core::UNKNOWN_MODEL_ID,
    routers::comm::metrics_utils::{error_type_from_status, route_to_endpoint},
};
use axum::{body::Body, response::Response};
use http::StatusCode;
use http_body::{Body as HttpBody, Frame, SizeHint};
use std::{
    pin::Pin,
    task::{Context, Poll},
    time::Instant,
};

pub(crate) struct RequestMetrics {
    router: &'static str,
    backend: &'static str,
    model: String,
    endpoint: &'static str,
    pub streaming: bool,
    started: Instant,
    status: Option<u16>,
    error: Option<&'static str>,
    detector: FirstOutputSse,
    usage: UsageObserver,
}
impl RequestMetrics {
    pub fn new(
        router: &'static str,
        backend: &'static str,
        model: &str,
        route: &str,
        streaming: bool,
    ) -> Self {
        let model = if model.is_empty() {
            UNKNOWN_MODEL_ID
        } else {
            model
        };
        let endpoint = route_to_endpoint(route);
        MeshMetrics::record_router_request(
            router,
            backend,
            metrics_labels::CONNECTION_HTTP,
            model,
            endpoint,
            bool_to_static_str(streaming),
        );
        Self {
            router,
            backend,
            model: model.to_owned(),
            endpoint,
            streaming,
            started: Instant::now(),
            status: None,
            error: Some("canceled"),
            detector: FirstOutputSse::default(),
            usage: UsageObserver::default(),
        }
    }
    #[cfg(feature = "ext-proc")]
    pub fn started_at(mut self, started: Instant) -> Self {
        self.started = started;
        self
    }
    pub fn body(&mut self, bytes: &[u8]) {
        self.usage.feed(bytes, self.streaming);
        // HTTP TTFT is measured by the outer ingress middleware, which includes
        // admission and upload time. Only ext-proc records it here.
        if self.router == "ext_proc" && self.streaming && self.detector.feed(bytes) {
            MeshMetrics::record_router_ttft(
                self.router,
                self.backend,
                &self.model,
                self.endpoint,
                self.started.elapsed(),
            );
            metrics::histogram!("mesh_ext_proc_ttft_seconds")
                .record(self.started.elapsed().as_secs_f64());
        }
    }
    pub fn finish(&mut self, status: Option<u16>, error: Option<&'static str>) {
        self.status = status;
        self.error = error.or_else(|| {
            status
                .and_then(|s| StatusCode::from_u16(s).ok())
                .filter(|s| s.is_client_error() || s.is_server_error())
                .map(error_type_from_status)
        });
    }
    pub fn wrap_response(mut self, response: Response) -> Response {
        self.status = Some(response.status().as_u16());
        if let Some(content_type) = response.headers().get(http::header::CONTENT_TYPE) {
            self.streaming = content_type.as_bytes().starts_with(b"text/event-stream");
        } else if !response.status().is_success() {
            self.streaming = false;
        }
        let (parts, inner) = response.into_parts();
        // Empty responses may never be polled by hyper.
        if inner.is_end_stream() {
            self.finish(self.status, None);
        }
        Response::from_parts(
            parts,
            Body::new(ObservedBody {
                inner,
                observation: Some(self),
            }),
        )
    }
}
impl Drop for RequestMetrics {
    fn drop(&mut self) {
        MeshMetrics::record_router_duration(
            self.router,
            self.backend,
            metrics_labels::CONNECTION_HTTP,
            &self.model,
            self.endpoint,
            self.started.elapsed(),
        );
        if let Some(error) = self.error {
            MeshMetrics::record_router_error(
                self.router,
                self.backend,
                metrics_labels::CONNECTION_HTTP,
                &self.model,
                self.endpoint,
                error,
            );
        }
        if let Some((prompt, completion)) = self.usage.usage(self.streaming) {
            for (kind, token_type, count) in [
                ("prompt", metrics_labels::TOKEN_INPUT, prompt),
                ("completion", metrics_labels::TOKEN_OUTPUT, completion),
            ] {
                MeshMetrics::record_router_tokens(
                    self.router,
                    self.backend,
                    &self.model,
                    self.endpoint,
                    token_type,
                    count,
                );
                if self.router == "ext_proc" {
                    metrics::counter!("mesh_ext_proc_tokens_total", "kind" => kind)
                        .increment(count);
                }
            }
        }
    }
}
struct ObservedBody {
    inner: Body,
    observation: Option<RequestMetrics>,
}
impl HttpBody for ObservedBody {
    type Data = bytes::Bytes;
    type Error = axum::Error;
    fn poll_frame(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
    ) -> Poll<Option<Result<Frame<Self::Data>, Self::Error>>> {
        let this = self.get_mut();
        let frame = Pin::new(&mut this.inner).poll_frame(cx);
        if let Some(observation) = &mut this.observation {
            let terminal = match &frame {
                Poll::Ready(Some(Ok(frame))) => {
                    if let Some(bytes) = frame.data_ref() {
                        observation.body(bytes);
                    }
                    this.inner.is_end_stream().then_some(None)
                }
                Poll::Ready(Some(Err(_))) => Some(Some("body_error")),
                Poll::Ready(None) => Some(None),
                Poll::Pending => None,
            };
            if let Some(error) = terminal {
                observation.finish(observation.status, error);
                this.observation.take();
            }
        }
        frame
    }
    fn is_end_stream(&self) -> bool {
        self.inner.is_end_stream()
    }
    fn size_hint(&self) -> SizeHint {
        self.inner.size_hint()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use http_body_util::BodyExt;

    fn total(rendered: &str, name: &str, labels: &[&str]) -> f64 {
        rendered
            .lines()
            .filter(|line| {
                line.starts_with(&format!("{name}{{"))
                    && labels.iter().all(|label| line.contains(label))
            })
            .map(|line| line.rsplit_once(' ').unwrap().1.parse::<f64>().unwrap())
            .sum()
    }

    #[tokio::test]
    async fn both_ingresses_record_usage_once_even_when_the_body_is_canceled_or_fails() {
        let recorder = metrics_exporter_prometheus::PrometheusBuilder::new().build_recorder();
        let handle = recorder.handle();
        let _recorder = metrics::set_default_local_recorder(&recorder);
        let chunk = bytes::Bytes::from_static(b"data: {\"choices\":[{\"delta\":{\"content\":\"hi\"}}],\"usage\":{\"prompt_tokens\":12,\"completion_tokens\":3}}\n\n");
        for ingress in ["http", "ext_proc"] {
            for outcome in ["completed", "canceled", "body_error"] {
                let observation =
                    RequestMetrics::new(ingress, "pd", "m", "/v1/chat/completions", true);
                let mut chunks = vec![Ok::<_, std::io::Error>(chunk.clone())];
                if outcome != "completed" {
                    chunks.push(Err(std::io::Error::other("upstream reset")));
                }
                let stream = futures_util::stream::iter(chunks);
                let mut response =
                    observation.wrap_response(Response::new(Body::from_stream(stream)));
                assert_eq!(
                    response
                        .body_mut()
                        .frame()
                        .await
                        .unwrap()
                        .unwrap()
                        .into_data()
                        .unwrap(),
                    chunk
                );
                if outcome == "body_error" {
                    assert!(response.body_mut().frame().await.unwrap().is_err());
                } else if outcome == "completed" {
                    // A finite stream reaches EOF without requiring the response to drop.
                    assert!(response.body_mut().frame().await.is_none());
                }
                drop(response);
            }
            let response = RequestMetrics::new(ingress, "pd", "m", "/v1/chat/completions", false)
                .wrap_response(Response::new(Body::from(
                    r#"{"usage":{"prompt_tokens":12,"completion_tokens":3}}"#,
                )));
            axum::body::to_bytes(response.into_body(), 4096)
                .await
                .unwrap();
        }
        let rendered = handle.render();
        for ingress in ["http", "ext_proc"] {
            let router_label = format!("router_type=\"{ingress}\"");
            assert_eq!(
                total(&rendered, "mesh_router_requests_total", &[&router_label]),
                4.0,
                "{rendered}"
            );
            assert_eq!(
                total(
                    &rendered,
                    "mesh_router_request_duration_seconds_count",
                    &[&router_label]
                ),
                4.0,
                "{rendered}"
            );
            assert_eq!(
                total(
                    &rendered,
                    "mesh_router_tokens_total",
                    &[&router_label, "token_type=\"input\""]
                ),
                48.0,
                "{rendered}"
            );
            assert_eq!(
                total(
                    &rendered,
                    "mesh_router_tokens_total",
                    &[&router_label, "token_type=\"output\""]
                ),
                12.0,
                "{rendered}"
            );
            assert_eq!(
                total(
                    &rendered,
                    "mesh_router_request_errors_total",
                    &[&router_label, "error_type=\"body_error\""]
                ),
                1.0,
                "{rendered}"
            );
        }
        assert_eq!(
            total(
                &rendered,
                "mesh_router_ttft_seconds_count",
                &["router_type=\"http\""]
            ),
            0.0
        );
        assert_eq!(
            total(
                &rendered,
                "mesh_router_ttft_seconds_count",
                &["router_type=\"ext_proc\""]
            ),
            3.0
        );
        assert_eq!(
            total(
                &rendered,
                "mesh_ext_proc_tokens_total",
                &["kind=\"prompt\""]
            ),
            48.0
        );
    }
}
