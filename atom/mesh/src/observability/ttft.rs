//! HTTP ingress to first generated SSE output, measured on the Mesh clock.

use std::{
    pin::Pin,
    task::{Context, Poll},
    time::Instant,
};

use axum::{
    body::Body,
    extract::{Request, State},
    http::{header::CONTENT_TYPE, Method},
    middleware::Next,
    response::Response,
};
use bytes::Bytes;
use http_body::Frame;
use serde_json::Value;

use crate::{
    config::{RouterConfig, RoutingMode},
    core::{ConnectionMode, UNKNOWN_MODEL_ID},
    observability::metrics::{metrics_labels, MeshMetrics},
    routers::comm::metrics_utils::route_to_endpoint,
};

/// The server holds a RouterManager, so resolve the backend from its configuration.
pub fn http_backend_type(config: &RouterConfig) -> &'static str {
    if config.atom_standalone || !matches!(config.connection_mode, ConnectionMode::Http) {
        return "unsupported";
    }
    match config.mode {
        RoutingMode::Regular { .. } => "regular",
        RoutingMode::PrefillDecode { .. } => "pd",
    }
}

/// Installed outside the concurrency queue, so retries and queueing retain t0.
pub async fn track_http_ttft(
    State(backend_type): State<&'static str>,
    request: Request,
    next: Next,
) -> Response {
    // Other router types already collect their own generation metrics.
    if !matches!(backend_type, "regular" | "pd")
        || request.method() != Method::POST
        || !matches!(
            request.uri().path(),
            "/v1/chat/completions" | "/v1/completions"
        )
    {
        return next.run(request).await;
    }
    let started_at = Instant::now();
    let endpoint = route_to_endpoint(request.uri().path());
    let response = next.run(request).await;
    wrap_response(response, started_at, backend_type, endpoint)
}

fn wrap_response(
    response: Response,
    started_at: Instant,
    backend_type: &'static str,
    endpoint: &'static str,
) -> Response {
    if !response.status().is_success()
        || !response
            .headers()
            .get(CONTENT_TYPE)
            .is_some_and(|value| value.as_bytes().starts_with(b"text/event-stream"))
    {
        // A buffered JSON response cannot reveal when the first token existed.
        return response;
    }
    let (parts, body) = response.into_parts();
    Response::from_parts(
        parts,
        Body::new(FirstOutputBody {
            inner: body,
            detector: FirstOutputSse::default(),
            started_at,
            backend_type,
            endpoint,
        }),
    )
}

struct FirstOutputBody {
    inner: Body,
    detector: FirstOutputSse,
    started_at: Instant,
    backend_type: &'static str,
    endpoint: &'static str,
}

impl http_body::Body for FirstOutputBody {
    type Data = Bytes;
    type Error = axum::Error;

    fn poll_frame(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
    ) -> Poll<Option<Result<Frame<Self::Data>, Self::Error>>> {
        let this = self.get_mut();
        let result = Pin::new(&mut this.inner).poll_frame(cx);
        if let Poll::Ready(Some(Ok(frame))) = &result {
            if let Some(data) = frame.data_ref() {
                if this.detector.feed(data) {
                    MeshMetrics::record_router_ttft(
                        metrics_labels::ROUTER_HTTP,
                        this.backend_type,
                        this.detector.model.as_deref().unwrap_or(UNKNOWN_MODEL_ID),
                        this.endpoint,
                        this.started_at.elapsed(),
                    );
                }
            }
        }
        // Forward the original frame, preserving bytes, errors, trailers and backpressure.
        result
    }

    fn is_end_stream(&self) -> bool {
        self.inner.is_end_stream()
    }

    fn size_hint(&self) -> http_body::SizeHint {
        self.inner.size_hint()
    }
}

#[derive(Default)]
struct SseFrames {
    pending: Vec<u8>,
    start: usize,
    search: usize,
    oversized: bool,
}

impl SseFrames {
    const MAX_FRAME_BYTES: usize = 1024 * 1024;

    fn append(&mut self, chunk: &[u8]) {
        // Amortize compaction over the consumed bytes, not over frames.
        if self.start > 0 && self.start >= self.pending.len() / 2 {
            self.pending.drain(..self.start);
            self.search -= self.start;
            self.start = 0;
        }
        self.pending.extend_from_slice(chunk);
    }

    fn next_frame(&mut self) -> Option<&[u8]> {
        while let Some(offset) = memchr::memchr(b'\n', &self.pending[self.search..]) {
            let end = self.search + offset;
            self.search = end + 1;
            // Inspect preceding bytes so delimiters spanning appends require no
            // rescan of the buffered prefix. Each newline is visited once.
            let frame_end = if end > self.start && self.pending[end - 1] == b'\n' {
                Some(end - 1)
            } else if end >= self.start + 3 && &self.pending[end - 3..end] == b"\r\n\r" {
                Some(end - 3)
            } else {
                None
            };
            if let Some(frame_end) = frame_end {
                if frame_end - self.start > Self::MAX_FRAME_BYTES {
                    self.oversized = true;
                    return None;
                }
                let start = self.start;
                self.start = self.search;
                return Some(&self.pending[start..frame_end]);
            }
        }
        self.search = self.pending.len();
        self.oversized = self.pending.len() - self.start > Self::MAX_FRAME_BYTES;
        None
    }
}

#[derive(Default)]
struct FirstOutputSse {
    frames: SseFrames,
    model: Option<String>,
    done: bool,
}

impl FirstOutputSse {
    fn feed(&mut self, chunk: &[u8]) -> bool {
        if self.done {
            return false;
        }
        self.frames.append(chunk);
        while let Some(frame) = self.frames.next_frame() {
            let Ok(frame) = std::str::from_utf8(frame) else {
                continue;
            };
            let data = frame
                .lines()
                .filter_map(|line| {
                    line.strip_prefix("data:")
                        .map(|s| s.trim_start_matches(' '))
                })
                .collect::<Vec<_>>()
                .join("\n");
            if data == "[DONE]" {
                self.finish();
                return false;
            }
            let Ok(payload) = serde_json::from_str::<Value>(&data) else {
                continue;
            };
            if payload.get("error").is_some() {
                self.finish();
                return false;
            }
            if let Some(model) = payload.get("model").and_then(Value::as_str) {
                self.model = Some(model.to_owned());
            }
            if has_generated_output(&payload) {
                self.finish();
                return true;
            }
        }
        if self.frames.oversized {
            self.finish();
        }
        false
    }

    fn finish(&mut self) {
        self.done = true;
        self.frames = SseFrames::default();
    }
}

fn nonempty_string(value: &Value) -> bool {
    value.as_str().is_some_and(|s| !s.is_empty())
}

fn has_function_output(value: &Value) -> bool {
    nonempty_string(&value["name"]) || nonempty_string(&value["arguments"])
}

fn has_generated_output(payload: &Value) -> bool {
    payload["choices"].as_array().is_some_and(|choices| {
        choices.iter().any(|choice| {
            let delta = &choice["delta"];
            nonempty_string(&choice["text"])
                || ["content", "reasoning_content", "reasoning"]
                    .iter()
                    .any(|key| nonempty_string(&delta[key]))
                || has_function_output(&delta["function_call"])
                || delta["tool_calls"].as_array().is_some_and(|calls| {
                    calls
                        .iter()
                        .any(|call| has_function_output(&call["function"]))
                })
        })
    })
}

#[cfg(test)]
mod tests {
    use std::{convert::Infallible, time::Duration};

    use axum::{
        http::{HeaderMap, StatusCode},
        routing::post,
        Router,
    };
    use http_body_util::{BodyExt, StreamBody};
    use metrics_exporter_prometheus::PrometheusBuilder;
    use serde_json::json;
    use tower::ServiceExt;

    use super::*;

    #[test]
    fn managed_http_backends_use_config_and_other_transports_are_excluded() {
        let mut config = RouterConfig::default();
        assert_eq!(http_backend_type(&config), "regular");
        config.mode = RoutingMode::PrefillDecode {
            prefill_urls: vec![],
            decode_urls: vec![],
            prefill_policy: None,
            decode_policy: None,
        };
        assert_eq!(http_backend_type(&config), "pd");
        config.connection_mode = ConnectionMode::Grpc { port: None };
        assert_eq!(http_backend_type(&config), "unsupported");
        config.connection_mode = ConnectionMode::Http;
        config.atom_standalone = true;
        assert_eq!(http_backend_type(&config), "unsupported");
    }

    fn sse(value: Value) -> Vec<u8> {
        format!("data: {value}\r\n\r\n").into_bytes()
    }

    #[test]
    fn fragmented_unicode_and_metadata_are_handled_without_duplicate_samples() {
        for payload in [
            json!({"choices": [{"delta": {"content": "你好"}}]}),
            json!({"choices": [{"delta": {"reasoning_content": "thinking"}}]}),
            json!({"choices": [{"delta": {"tool_calls": [{"function": {"name": "search"}}]}}]}),
            json!({"choices": [{"delta": {"function_call": {"arguments": "{"}}}]}),
            json!({"choices": [{"text": " "}]}),
        ] {
            let mut detector = FirstOutputSse::default();
            assert!(!detector.feed(b": keepalive\n\n"));
            assert!(!detector.feed(&sse(json!({"model": "test-model", "choices": [{"delta": {"role": "assistant", "content": ""}}]}))));
            let bytes = sse(payload);
            for (index, byte) in bytes.iter().enumerate() {
                assert_eq!(detector.feed(&[*byte]), index == bytes.len() - 1);
            }
            assert_eq!(detector.model.as_deref(), Some("test-model"));
            assert!(!detector.feed(&bytes));
        }
    }

    #[test]
    fn invalid_terminal_and_oversized_events_do_not_fabricate_ttft() {
        let valid = sse(json!({"choices": [{"text": "ok"}]}));
        for terminal in [
            b"data: [DONE]\n\n".to_vec(),
            sse(json!({"error": "failed"})),
        ] {
            let mut detector = FirstOutputSse::default();
            assert!(!detector.feed(&terminal));
            assert!(!detector.feed(&valid));
        }
        let mut detector = FirstOutputSse::default();
        for invalid in [
            json!(null),
            json!({"choices": 42}),
            json!({"choices": [null]}),
            json!({"choices": [{"delta": {"tool_calls": 42}}]}),
            json!({"usage": {"completion_tokens": 3}}),
        ] {
            assert!(!detector.feed(&sse(invalid)));
        }
        assert!(detector.feed(&valid));
        let mut detector = FirstOutputSse::default();
        assert!(!detector.feed(&vec![b'x'; SseFrames::MAX_FRAME_BYTES + 1]));
        assert!(detector.done && detector.frames.pending.is_empty());
    }

    fn sample(rendered: &str, suffix: &str) -> f64 {
        rendered
            .lines()
            .find(|line| line.starts_with(&format!("mesh_router_ttft_seconds_{suffix}{{")))
            .unwrap_or_else(|| panic!("missing {suffix}: {rendered}"))
            .rsplit_once(' ')
            .unwrap()
            .1
            .parse()
            .unwrap()
    }

    #[test]
    fn many_metadata_frames_and_large_fragmented_output() {
        let mut payload = b": keepalive\n\ndata: {}\r\n\r\n".repeat(100);
        payload.extend(sse(json!({"choices": [{"text": "x".repeat(65536)}]})));
        for chunk_size in [1, 3, 1024, 65536] {
            let mut detector = FirstOutputSse::default();
            let count = payload.chunks(chunk_size).len();
            for (index, chunk) in payload.chunks(chunk_size).enumerate() {
                assert_eq!(detector.feed(chunk), index + 1 == count);
            }
            assert!(detector.frames.pending.is_empty());
        }
    }

    #[test]
    fn frame_limit_applies_after_consumed_metadata() {
        let mut payload = b"data: {}\r\n\r\ndata: ".to_vec();
        payload.extend(vec![b'x'; SseFrames::MAX_FRAME_BYTES]);
        payload.extend_from_slice(b"\n\n");
        let mut detector = FirstOutputSse::default();
        assert!(!detector.feed(&payload));
        assert!(detector.done);
    }

    #[tokio::test(flavor = "current_thread")]
    async fn preserves_body_frames_and_trailers_and_records_once_at_first_content() {
        let recorder = PrometheusBuilder::new().build_recorder();
        let handle = recorder.handle();
        let _guard = metrics::set_default_local_recorder(&recorder);
        let role = Bytes::from(sse(
            json!({"model": "test-model", "choices": [{"delta": {"role": "assistant"}}]}),
        ));
        let content = Bytes::from(sse(
            json!({"choices": [{"delta": {"content": "four tokens"}}]}),
        ));
        let mut trailers = HeaderMap::new();
        trailers.insert("x-test", "retained".parse().unwrap());
        let frames: Vec<Result<_, Infallible>> = vec![
            Ok(Frame::data(role.clone())),
            Ok(Frame::data(content.clone())),
            Ok(Frame::data(content.clone())),
            Ok(Frame::trailers(trailers.clone())),
        ];
        let response = Response::builder()
            .header(CONTENT_TYPE, "text/event-stream")
            .body(Body::new(StreamBody::new(futures_util::stream::iter(
                frames,
            ))))
            .unwrap();
        let mut body = wrap_response(
            response,
            Instant::now() - Duration::from_secs(1),
            "pd",
            "chat",
        )
        .into_body();
        assert_eq!(
            body.frame().await.unwrap().unwrap().into_data().unwrap(),
            role
        );
        assert!(!handle.render().contains("mesh_router_ttft_seconds_count"));
        assert_eq!(
            body.frame().await.unwrap().unwrap().into_data().unwrap(),
            content
        );
        let rendered = handle.render();
        assert_eq!(sample(&rendered, "count"), 1.0);
        assert!(sample(&rendered, "sum") >= 1.0);
        assert!(rendered.contains("model=\"test-model\""));
        assert_eq!(
            body.frame().await.unwrap().unwrap().into_data().unwrap(),
            content
        );
        assert_eq!(
            body.frame()
                .await
                .unwrap()
                .unwrap()
                .into_trailers()
                .unwrap(),
            trailers
        );
        assert!(body.frame().await.is_none());
        assert_eq!(sample(&handle.render(), "count"), 1.0);
    }

    #[tokio::test(flavor = "current_thread")]
    async fn ingress_timing_includes_handler_wait_and_skips_buffered_or_failed_responses() {
        let recorder = PrometheusBuilder::new().build_recorder();
        let handle = recorder.handle();
        let _guard = metrics::set_default_local_recorder(&recorder);
        let app = Router::new()
            .route(
                "/v1/chat/completions",
                post(|| async {
                    tokio::time::sleep(Duration::from_millis(10)).await;
                    Response::builder()
                        .header(CONTENT_TYPE, "text/event-stream")
                        .body(Body::from(sse(
                            json!({"model": "test-model", "choices": [{"text": "ok"}]}),
                        )))
                        .unwrap()
                }),
            )
            .layer(axum::middleware::from_fn_with_state(
                http_backend_type(&RouterConfig::default()),
                track_http_ttft,
            ));
        let response = app
            .oneshot(
                Request::post("/v1/chat/completions")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert!(!handle.render().contains("mesh_router_ttft_seconds_count"));
        response.into_body().collect().await.unwrap();
        assert_eq!(sample(&handle.render(), "count"), 1.0);
        assert!(sample(&handle.render(), "sum") >= 0.01);
        for (status, content_type) in [
            (StatusCode::OK, "application/json"),
            (StatusCode::BAD_GATEWAY, "text/event-stream"),
        ] {
            let response = Response::builder()
                .status(status)
                .header(CONTENT_TYPE, content_type)
                .body(Body::from(sse(json!({"choices": [{"text": "ignored"}]}))))
                .unwrap();
            wrap_response(response, Instant::now(), "pd", "chat")
                .into_body()
                .collect()
                .await
                .unwrap();
        }
        assert_eq!(sample(&handle.render(), "count"), 1.0);
    }
}
