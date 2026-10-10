use std::sync::Arc;

use crate::core::Worker;
use axum::{body::Body, response::Response};
use bytes::Bytes;
use http::StatusCode;

pub(crate) struct ProxyBody {
    inner: Body,
    worker: Arc<dyn Worker>,
    policy: Arc<dyn crate::policies::LoadBalancingPolicy>,
    status: StatusCode,
    finished: bool,
    remaining: Option<u64>,
    _load: crate::core::WorkerLoadGuard,
}
impl ProxyBody {
    pub(crate) fn response(
        response: reqwest::Response,
        worker: Arc<dyn Worker>,
        policy: Arc<dyn crate::policies::LoadBalancingPolicy>,
        load: crate::core::WorkerLoadGuard,
    ) -> Response {
        let status = response.status();
        let headers = super::header_utils::preserve_response_headers(response.headers());
        let remaining = response.content_length();
        let mut body = ProxyBody {
            inner: Body::from_stream(response.bytes_stream()),
            worker,
            policy,
            status,
            finished: false,
            remaining,
            _load: load,
        };
        if remaining == Some(0) {
            body.finish(true);
        }
        let mut response = Response::new(Body::new(body));
        *response.status_mut() = status;
        *response.headers_mut() = headers;
        response
    }

    fn finish(&mut self, body_ok: bool) {
        if self.finished {
            return;
        }
        self.finished = true;
        let success = body_ok && self.status.is_success();
        if success || !body_ok || self.status.is_server_error() {
            self.worker.record_outcome(success);
        }
        self.policy.on_request_complete(self.worker.url(), success);
    }
}
impl Drop for ProxyBody {
    fn drop(&mut self) {
        if !self.finished {
            // Count discarded 5xx responses, but do not penalize client cancellation.
            if self.status.is_server_error() {
                self.worker.record_outcome(false);
            }
            self.policy.on_request_complete(self.worker.url(), false);
        }
    }
}
impl http_body::Body for ProxyBody {
    type Data = Bytes;
    type Error = axum::Error;
    fn poll_frame(
        self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Option<Result<http_body::Frame<Bytes>, axum::Error>>> {
        let this = self.get_mut();
        let result = std::pin::Pin::new(&mut this.inner).poll_frame(cx);
        match &result {
            std::task::Poll::Ready(Some(Ok(frame))) => {
                if let (Some(remaining), Some(data)) = (&mut this.remaining, frame.data_ref()) {
                    *remaining = remaining.saturating_sub(data.len() as u64);
                }
                // Hyper may drop fixed-length bodies without an EOF poll.
                if this.remaining == Some(0) || this.inner.is_end_stream() {
                    this.finish(true);
                }
            }
            std::task::Poll::Ready(None) => this.finish(true),
            std::task::Poll::Ready(Some(Err(_))) => this.finish(false),
            _ => {}
        }
        result
    }
}

#[cfg(test)]
#[path = "../../../tests/routing/http_proxy_body_tests.rs"]
mod tests;
