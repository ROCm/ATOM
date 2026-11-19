//! Routing tokens for a chat as one of its workers renders it.
//!
//! `POST /v1/chat/completions/render`, which ATOM and vLLM serve, takes a chat
//! request and returns, among other fields, the `token_ids` the worker's engine
//! runs for it. Those are the tokens its KV events key blocks by, whatever
//! template kwargs, reasoning rules or encoder the worker resolved at startup.
use std::{
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    },
    time::{Duration, Instant},
};

use bytes::Bytes;
use dashmap::DashMap;
use http::{header::CONTENT_TYPE, HeaderMap, HeaderValue};
use reqwest::{Client, StatusCode};
use serde::Deserialize;
use tracing::info;

use crate::{
    core::Worker,
    routers::comm::{error::IngressError, header_utils},
};

pub(crate) const CHAT_RENDER_PATH: &str = "/v1/chat/completions/render";
/// The longest the router waits for a worker's render of a chat it can place
/// without it.
pub(crate) const RENDER_TIMEOUT: Duration = Duration::from_secs(2);
/// How long a worker that serves no render route goes unasked.
const UNSERVED_FOR: Duration = Duration::from_secs(60);

/// The workers chats render on, shared by every request's routing.
#[derive(Debug, Default)]
pub struct WorkerRenders {
    /// Round-robin position among a chat's workers.
    next: AtomicUsize,
    /// Workers, by base URL, that answered a render with 404 or 405, and when.
    unserved: DashMap<String, Instant>,
}

/// Why a worker's render gave no routing tokens.
#[derive(Debug)]
pub(crate) enum RenderMiss {
    /// The worker refused the chat with a 4xx, as its chat route does, in the
    /// answer it sent.
    Refused(IngressError),
    /// The worker has no render route, as backends before it do not.
    Unserved,
    /// The render failed, or its response had no token IDs.
    Failed,
    /// No response in the time the request had left.
    TimedOut,
}

impl RenderMiss {
    pub(crate) fn label(&self) -> &'static str {
        match self {
            Self::Refused(_) => "refused",
            Self::Unserved => "unserved",
            Self::Failed => "failed",
            Self::TimedOut => "timed_out",
        }
    }
}

#[derive(Deserialize)]
struct Rendered {
    token_ids: Vec<u32>,
}

impl WorkerRenders {
    /// The worker of `workers` to render a chat on: the next available one
    /// that has not lately answered that it serves no render route.
    pub(crate) fn pick(&self, workers: &[Arc<dyn Worker>]) -> Option<Arc<dyn Worker>> {
        let now = Instant::now();
        let ready: Vec<&Arc<dyn Worker>> = workers
            .iter()
            .filter(|worker| worker.is_available() && !self.unserved(worker.base_url(), now))
            .collect();
        if ready.is_empty() {
            return None;
        }
        let at = self.next.fetch_add(1, Ordering::Relaxed) % ready.len();
        Some(ready[at].clone())
    }

    fn unserved(&self, url: &str, now: Instant) -> bool {
        self.unserved
            .remove_if(url, |_, since| now.duration_since(*since) >= UNSERVED_FOR);
        self.unserved.contains_key(url)
    }

    /// The prompt token IDs `worker` renders `body`, a chat request as the
    /// client sent it, to, waiting no longer than `budget`. The client's
    /// `headers` are forwarded as for the chat itself.
    pub(crate) async fn render(
        &self,
        client: &Client,
        worker: &dyn Worker,
        headers: &HeaderMap,
        body: Bytes,
        budget: Duration,
    ) -> Result<Vec<u32>, RenderMiss> {
        let mut headers = header_utils::inference_request_headers(
            headers,
            CHAT_RENDER_PATH,
            worker.api_key().as_deref(),
        )
        .map_err(|_| RenderMiss::Failed)?;
        headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
        let response = client
            .post(worker.endpoint_url(CHAT_RENDER_PATH))
            .headers(headers)
            .body(body)
            .timeout(budget)
            .send()
            .await
            .map_err(miss)?;
        match response.status() {
            status if status.is_success() => {}
            StatusCode::NOT_FOUND | StatusCode::METHOD_NOT_ALLOWED => {
                info!(
                    worker = worker.url(),
                    "The worker serves no {}, so the router renders chats itself instead of asking it",
                    CHAT_RENDER_PATH
                );
                self.unserved
                    .insert(worker.base_url().to_owned(), Instant::now());
                return Err(RenderMiss::Unserved);
            }
            // The worker is busy, not refusing the chat.
            StatusCode::REQUEST_TIMEOUT | StatusCode::TOO_MANY_REQUESTS => {
                return Err(RenderMiss::Failed)
            }
            status if status.is_client_error() => {
                let content_type = response
                    .headers()
                    .get(CONTENT_TYPE)
                    .cloned()
                    .unwrap_or_else(|| HeaderValue::from_static("application/json"));
                let body = response.bytes().await.map_err(miss)?;
                return Err(RenderMiss::Refused(IngressError::from_worker(
                    status,
                    content_type,
                    body,
                )));
            }
            _ => return Err(RenderMiss::Failed),
        }
        let body = response.bytes().await.map_err(miss)?;
        serde_json::from_slice::<Rendered>(&body)
            .map(|rendered| rendered.token_ids)
            .map_err(|_| RenderMiss::Failed)
    }
}

fn miss(error: reqwest::Error) -> RenderMiss {
    if error.is_timeout() {
        RenderMiss::TimedOut
    } else {
        RenderMiss::Failed
    }
}
