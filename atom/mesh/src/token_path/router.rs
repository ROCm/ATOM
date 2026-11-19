//! Route a token request, run ATOM P/D, and forward decode events unchanged.

use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};

use dashmap::DashMap;
use futures::Stream;
use http::{HeaderMap, HeaderValue};
use serde_json::{json, Value};
use tracing::warn;

use super::engine::AtomEngine;
use super::types::{TokenGenerateEvent, TokenGenerateRequest};
use super::{AbortState, TokenPathError};
use crate::config::{AtomPdRankMappingPolicy, BackendType};
use crate::core::placement::backend::atom::{AtomAdapter, AtomPairCtx, AtomPrefillInfo};
use crate::core::placement::backend::BackendAdapter;
use crate::core::placement::planner::DefaultPlanner;
use crate::core::placement::registry_adapters::{PolicyRegistryAdapter, WorkerRegistryAdapter};
use crate::core::placement::traits::PdPlanner;
use crate::core::placement::{PlacementPlan, Protocol, RequestDescriptor};
use crate::core::{Worker, WorkerLoadGuard, WorkerRegistry, WorkerType};
use crate::observability::metrics::MeshMetrics;
use crate::policies::PolicyRegistry;

pub struct TokenStream {
    inner: Pin<Box<dyn Stream<Item = Result<TokenGenerateEvent, TokenPathError>> + Send>>,
}

impl std::fmt::Debug for TokenStream {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("TokenStream")
    }
}

impl TokenStream {
    pub fn into_inner(
        self,
    ) -> Pin<Box<dyn Stream<Item = Result<TokenGenerateEvent, TokenPathError>> + Send>> {
        self.inner
    }
}

impl Stream for TokenStream {
    type Item = Result<TokenGenerateEvent, TokenPathError>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        self.get_mut().inner.as_mut().poll_next(cx)
    }
}

struct ForwardStream {
    inner: super::engine::EngineStream,
    abort: Arc<AbortState>,
    _decode_load: Option<WorkerLoadGuard>,
    started: std::time::Instant,
    last_event: Option<std::time::Instant>,
}

impl Drop for ForwardStream {
    fn drop(&mut self) {
        self.abort.cancel("client_disconnect");
    }
}

impl Stream for ForwardStream {
    type Item = Result<TokenGenerateEvent, TokenPathError>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = self.get_mut();
        match this.inner.as_mut().poll_next(cx) {
            Poll::Ready(Some(Ok(event))) => {
                let now = std::time::Instant::now();
                let since = this.last_event.unwrap_or(this.started);
                MeshMetrics::record_token_forward(now.saturating_duration_since(since));
                this.last_event = Some(now);
                if event.finish_reason.is_some() {
                    // Engine finished. Drop must not abort a completed request.
                    this.abort.cancel_finished();
                }
                Poll::Ready(Some(Ok(event)))
            }
            Poll::Ready(None) => {
                this.abort.cancel_finished();
                Poll::Ready(None)
            }
            other => other,
        }
    }
}

pub struct TokenRouter {
    planner: Arc<DefaultPlanner>,
    registry: Arc<WorkerRegistry>,
    engine: Arc<dyn AtomEngine>,
    prefill_info: Arc<tokio::sync::OnceCell<Arc<AtomPrefillInfo>>>,
    http: reqwest::Client,
    rank_policy: AtomPdRankMappingPolicy,
    backend: BackendType,
    inflight: Arc<DashMap<String, Arc<AbortState>>>,
}

impl TokenRouter {
    pub fn new(
        registry: Arc<WorkerRegistry>,
        policies: Arc<PolicyRegistry>,
        engine: Arc<dyn AtomEngine>,
        prefill_info: Arc<AtomPrefillInfo>,
        rank_policy: AtomPdRankMappingPolicy,
        backend: BackendType,
    ) -> Self {
        let planner = Arc::new(DefaultPlanner::new(
            Arc::new(WorkerRegistryAdapter::new(registry.clone())),
            Arc::new(PolicyRegistryAdapter::new(policies)),
        ));
        let loaded = Arc::new(tokio::sync::OnceCell::new());
        let _ = loaded.set(prefill_info);
        Self {
            planner,
            registry,
            engine,
            prefill_info: loaded,
            http: reqwest::Client::new(),
            rank_policy,
            backend,
            inflight: Arc::new(DashMap::new()),
        }
    }

    pub fn from_parts(
        registry: Arc<WorkerRegistry>,
        policies: Arc<PolicyRegistry>,
        http: reqwest::Client,
        rank_policy: AtomPdRankMappingPolicy,
        backend: BackendType,
    ) -> Self {
        let planner = Arc::new(DefaultPlanner::new(
            Arc::new(WorkerRegistryAdapter::new(registry.clone())),
            Arc::new(PolicyRegistryAdapter::new(policies)),
        ));
        Self {
            planner,
            registry,
            engine: Arc::new(super::GrpcAtomEngine::new()),
            prefill_info: Arc::new(tokio::sync::OnceCell::new()),
            http,
            rank_policy,
            backend,
            inflight: Arc::new(DashMap::new()),
        }
    }

    pub async fn generate(
        &self,
        request: TokenGenerateRequest,
    ) -> Result<TokenStream, TokenPathError> {
        if let Err(error) = request.validate() {
            MeshMetrics::record_token_request("token", "error");
            return Err(error);
        }
        if self.backend != BackendType::Atom {
            MeshMetrics::record_token_request("token", "error");
            return Err(TokenPathError::UnsupportedBackend);
        }
        match self.dispatch(request).await {
            Ok(stream) => Ok(stream),
            Err(error) => {
                MeshMetrics::record_token_request("token", "error");
                Err(error)
            }
        }
    }

    async fn dispatch(&self, request: TokenGenerateRequest) -> Result<TokenStream, TokenPathError> {
        let mut headers = HeaderMap::new();
        if let Some(session_id) = request.session_id.as_deref() {
            let value =
                HeaderValue::from_str(session_id).map_err(|_| TokenPathError::InvalidSession)?;
            headers.insert("x-session-id", value);
        }
        let descriptor = RequestDescriptor {
            model_id: Some(request.model.as_str()),
            protocol: Some(Protocol::Grpc),
            text: None,
            tokens: Some(request.token_ids.as_slice()),
            headers: Some(&headers),
            stream: request.stream,
            carries_media: false,
        };
        let plan = self.planner.plan(&descriptor).await?;
        match plan {
            PlacementPlan::Single { worker, .. } => self.forward_one(request, worker).await,
            PlacementPlan::Pair {
                prefill, decode, ..
            } => self.forward_pd(request, prefill, decode).await,
        }
    }

    async fn forward_one(
        &self,
        mut request: TokenGenerateRequest,
        worker: Arc<dyn Worker>,
    ) -> Result<TokenStream, TokenPathError> {
        apply_rank(&mut request, worker.dp_rank());
        let abort = self.reserve(request.request_id.clone(), vec![worker.clone()])?;
        let load = WorkerLoadGuard::new(worker.clone(), None);
        let stream = match self.engine.generate(worker.as_ref(), request.clone()).await {
            Ok(stream) => stream,
            Err(error) => {
                abort.release();
                return Err(error);
            }
        };
        MeshMetrics::record_token_request("regular", "ok");
        Ok(self.wrap_stream(abort, stream, Some(load)))
    }

    async fn forward_pd(
        &self,
        request: TokenGenerateRequest,
        prefill: Arc<dyn Worker>,
        decode: Arc<dyn Worker>,
    ) -> Result<TokenStream, TokenPathError> {
        let prefill = self.map_prefill_idx2idx(prefill, &decode);
        let adapter = AtomAdapter::new(self.prefill_info().await?);
        let ctx = adapter.prepare_pair(prefill.as_ref(), decode.as_ref())?;
        let abort = self.reserve(
            request.request_id.clone(),
            vec![prefill.clone(), decode.clone()],
        )?;
        let prefill_load = WorkerLoadGuard::new(prefill.clone(), None);
        let decode_load = WorkerLoadGuard::new(decode.clone(), None);

        let mut prefill_request = request.clone();
        prefill_request.stream = false;
        prefill_request.max_tokens = Some(1);
        set_max_tokens(&mut prefill_request.sampling_params, 1);
        prefill_request.kv_transfer_params = Some(json!({
            "do_remote_decode": true,
            "do_remote_prefill": false,
        }));
        apply_rank(&mut prefill_request, prefill.dp_rank());

        let kv = match self.read_prefill_kv(&prefill, prefill_request).await {
            Ok(kv) => kv,
            Err(error) => {
                self.fail_reserved(&abort, std::slice::from_ref(&prefill))
                    .await;
                return Err(error);
            }
        };
        drop(prefill_load);
        if abort.is_finished() {
            return Err(aborted_error());
        }

        let mut kv = kv;
        if let Err(error) = adapter.enrich_decode_kv(&mut kv, &ctx) {
            self.fail_reserved(&abort, std::slice::from_ref(&prefill))
                .await;
            return Err(error.into());
        }
        let carried = AtomAdapter::carry_prompt_token_ids(
            &json!({"prompt_token_ids": request.token_ids}),
            &mut kv,
        );
        if carried == 0 {
            self.fail_reserved(&abort, std::slice::from_ref(&prefill))
                .await;
            return Err(TokenPathError::Engine {
                role: "decode",
                message: "prefill kv_transfer_params has no prompt_token_ids".into(),
            });
        }
        let _ = ctx.downcast_ref::<AtomPairCtx>();

        let mut decode_request = request.clone();
        decode_request.kv_transfer_params = Some(kv);
        apply_rank(&mut decode_request, decode.dp_rank());
        if abort.is_finished() {
            return Err(aborted_error());
        }
        let stream = match self.engine.generate(decode.as_ref(), decode_request).await {
            Ok(stream) => stream,
            Err(error) => {
                self.fail_reserved(&abort, &[prefill, decode]).await;
                return Err(error);
            }
        };
        MeshMetrics::record_token_request("pd", "ok");
        Ok(self.wrap_stream(abort, stream, Some(decode_load)))
    }

    async fn fail_reserved(&self, abort: &AbortState, workers: &[Arc<dyn Worker>]) {
        if abort.is_finished() {
            return;
        }
        for worker in workers {
            self.abort_logged(worker.as_ref(), &abort.request_id).await;
        }
        abort.release();
    }

    async fn abort_logged(&self, worker: &dyn Worker, request_id: &str) {
        if let Err(error) = self.engine.abort(worker, &[request_id.to_string()]).await {
            warn!(
                request_id,
                worker = worker.url(),
                %error,
                "token-path abort failed"
            );
        }
    }

    async fn read_prefill_kv(
        &self,
        prefill: &Arc<dyn Worker>,
        request: TokenGenerateRequest,
    ) -> Result<Value, TokenPathError> {
        let mut stream = self.engine.generate(prefill.as_ref(), request).await?;
        let mut kv = None;
        // Read the prefill RPC to completion. Stopping at the first KV event
        // drops the HTTP/2 stream and the engine treats that as a cancel.
        while let Some(item) = futures::StreamExt::next(&mut stream).await {
            let event = item?;
            if let Some(error) = event.error {
                return Err(TokenPathError::Engine {
                    role: "prefill",
                    message: error.to_string(),
                });
            }
            if kv.is_none() {
                if let Some(params) = event.kv_transfer_params {
                    kv = Some(params);
                }
            }
        }
        kv.ok_or(TokenPathError::MissingKv)
    }

    fn reserve(
        &self,
        request_id: String,
        workers: Vec<Arc<dyn Worker>>,
    ) -> Result<Arc<AbortState>, TokenPathError> {
        let abort = Arc::new(AbortState {
            finished: std::sync::atomic::AtomicBool::new(false),
            workers,
            request_id: request_id.clone(),
            engine: self.engine.clone(),
            inflight: self.inflight.clone(),
        });
        match self.inflight.entry(request_id) {
            dashmap::mapref::entry::Entry::Occupied(_) => Err(TokenPathError::DuplicateRequest),
            dashmap::mapref::entry::Entry::Vacant(slot) => {
                slot.insert(abort.clone());
                Ok(abort)
            }
        }
    }

    fn wrap_stream(
        &self,
        abort: Arc<AbortState>,
        stream: super::engine::EngineStream,
        decode_load: Option<WorkerLoadGuard>,
    ) -> TokenStream {
        TokenStream {
            inner: Box::pin(ForwardStream {
                inner: stream,
                abort,
                _decode_load: decode_load,
                started: std::time::Instant::now(),
                last_event: None,
            }),
        }
    }

    async fn prefill_info(&self) -> Result<Arc<AtomPrefillInfo>, TokenPathError> {
        if let Some(info) = self.prefill_info.get() {
            return Ok(info.clone());
        }
        self.prefill_info
            .get_or_try_init(|| load_atom_prefill_info(&self.registry, &self.http))
            .await
            .map(|info| info.clone())
            .map_err(|message| TokenPathError::Engine {
                role: "prefill",
                message,
            })
    }

    pub async fn abort(&self, request_ids: &[String]) -> Result<(), TokenPathError> {
        for request_id in request_ids {
            if let Some((_, state)) = self.inflight.remove(request_id) {
                state.cancel("explicit");
            }
        }
        Ok(())
    }

    fn map_prefill_idx2idx(
        &self,
        prefill: Arc<dyn Worker>,
        decode: &Arc<dyn Worker>,
    ) -> Arc<dyn Worker> {
        if self.rank_policy != AtomPdRankMappingPolicy::Idx2Idx {
            return prefill;
        }
        let (Some(prefill_dp_size), Some(decode_dp_rank)) = (prefill.dp_size(), decode.dp_rank())
        else {
            return prefill;
        };
        if decode_dp_rank >= prefill_dp_size || prefill.dp_rank() == Some(decode_dp_rank) {
            return prefill;
        }
        let mapped_url = format!("{}@{}", prefill.base_url(), decode_dp_rank);
        match self.registry.get_by_url(&mapped_url) {
            Some(mapped)
                if mapped.is_available()
                    && mapped.model_id() == prefill.model_id()
                    && matches!(mapped.worker_type(), WorkerType::Prefill { .. })
                    && mapped.connection_mode().matches(prefill.connection_mode()) =>
            {
                mapped
            }
            _ => prefill,
        }
    }
}

async fn load_atom_prefill_info(
    registry: &WorkerRegistry,
    http: &reqwest::Client,
) -> Result<Arc<AtomPrefillInfo>, String> {
    let prefills = registry.get_prefill_workers();
    if prefills.is_empty() {
        return Err("ATOM PD mode requires at least one prefill worker".to_string());
    }
    let mut info = AtomPrefillInfo::default();
    let mut seen = std::collections::HashSet::new();
    for worker in prefills {
        let base = worker.base_url().trim_end_matches('/').to_string();
        if !seen.insert(base.clone()) {
            if let Some(tp) = info.tp_sizes.get(&base).copied() {
                info.tp_sizes.insert(worker.url().to_string(), tp);
            }
            continue;
        }
        let info_url = kv_transfer_info_url(worker.as_ref());
        let response = http
            .get(&info_url)
            .send()
            .await
            .map_err(|error| format!("GET {info_url} failed: {error}"))?;
        if !response.status().is_success() {
            return Err(format!("{info_url} returned {}", response.status()));
        }
        let data: Value = response
            .json()
            .await
            .map_err(|error| format!("parse {info_url}: {error}"))?;
        let tp = data
            .get("tp_size")
            .and_then(Value::as_u64)
            .ok_or_else(|| format!("missing tp_size in {info_url}"))? as usize;
        if data.get("kv_role").and_then(Value::as_str) != Some("kv_producer") {
            return Err(format!("{base} is not a prefill"));
        }
        info.tp_sizes.insert(base, tp);
        info.tp_sizes.insert(worker.url().to_string(), tp);
    }
    Ok(Arc::new(info))
}

fn aborted_error() -> TokenPathError {
    TokenPathError::Engine {
        role: "engine",
        message: "request aborted".into(),
    }
}

fn kv_transfer_info_url(worker: &dyn Worker) -> String {
    format!(
        "{}/kv_transfer_info",
        super::grpc_endpoint(worker).trim_end_matches('/')
    )
}

fn apply_rank(request: &mut TokenGenerateRequest, rank: Option<usize>) {
    if let Some(rank) = rank {
        request.data_parallel_rank = Some(rank as u32);
    }
}

fn set_max_tokens(params: &mut Value, max_tokens: u32) {
    if !params.is_object() {
        *params = json!({});
    }
    if let Some(object) = params.as_object_mut() {
        object.insert("max_tokens".to_string(), json!(max_tokens));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;
    use futures::StreamExt;
    use serde_json::json;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::Mutex;

    struct Ending {
        events: std::vec::IntoIter<Result<TokenGenerateEvent, TokenPathError>>,
        on_end: Option<Arc<AtomicBool>>,
    }

    impl Stream for Ending {
        type Item = Result<TokenGenerateEvent, TokenPathError>;

        fn poll_next(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
            let this = self.get_mut();
            match this.events.next() {
                Some(item) => Poll::Ready(Some(item)),
                None => {
                    if let Some(flag) = &this.on_end {
                        flag.store(true, Ordering::SeqCst);
                    }
                    Poll::Ready(None)
                }
            }
        }
    }

    use crate::config::PolicyConfig;
    use crate::core::placement::test_support::make_regular_grpc;
    use crate::core::{ConnectionMode, DPAwareWorkerBuilder, WorkerType};

    struct ScriptedEngine {
        calls: Mutex<Vec<TokenGenerateRequest>>,
        urls: Mutex<Vec<String>>,
        aborts: Mutex<Vec<(String, Vec<String>)>>,
        fail_prefill: bool,
        prefill_completed: Arc<AtomicBool>,
    }

    impl Default for ScriptedEngine {
        fn default() -> Self {
            Self {
                calls: Mutex::new(Vec::new()),
                urls: Mutex::new(Vec::new()),
                aborts: Mutex::new(Vec::new()),
                fail_prefill: false,
                prefill_completed: Arc::new(AtomicBool::new(false)),
            }
        }
    }

    #[async_trait]
    impl AtomEngine for ScriptedEngine {
        async fn generate(
            &self,
            worker: &dyn Worker,
            request: TokenGenerateRequest,
        ) -> Result<super::super::engine::EngineStream, TokenPathError> {
            self.urls.lock().unwrap().push(worker.url().to_string());
            self.calls.lock().unwrap().push(request.clone());
            if self.fail_prefill && request.max_tokens == Some(1) {
                return Err(TokenPathError::Engine {
                    role: "prefill",
                    message: "prefill down".to_string(),
                });
            }
            let remote_decode = request
                .kv_transfer_params
                .as_ref()
                .and_then(|v| v.get("do_remote_decode"))
                .and_then(Value::as_bool)
                .unwrap_or(false);
            let events = if remote_decode {
                vec![Ok(TokenGenerateEvent {
                    token_ids: vec![],
                    finish_reason: Some("stop".into()),
                    kv_transfer_params: Some(json!({"dp_rank": 0})),
                    usage: None,
                    error: None,
                })]
            } else {
                vec![
                    Ok(TokenGenerateEvent {
                        token_ids: vec![11],
                        finish_reason: None,
                        kv_transfer_params: None,
                        usage: None,
                        error: None,
                    }),
                    Ok(TokenGenerateEvent {
                        token_ids: vec![12],
                        finish_reason: Some("stop".into()),
                        kv_transfer_params: None,
                        usage: None,
                        error: None,
                    }),
                ]
            };
            let on_end = remote_decode.then(|| Arc::clone(&self.prefill_completed));
            Ok(Box::pin(Ending {
                events: events.into_iter(),
                on_end,
            }))
        }

        async fn abort(
            &self,
            worker: &dyn Worker,
            request_ids: &[String],
        ) -> Result<(), TokenPathError> {
            self.aborts
                .lock()
                .unwrap()
                .push((worker.url().to_string(), request_ids.to_vec()));
            Ok(())
        }
    }

    fn request() -> TokenGenerateRequest {
        TokenGenerateRequest {
            request_id: "req-1".into(),
            model: "m".into(),
            token_ids: vec![1, 2, 3, 4],
            sampling_params: json!({"temperature": 0}),
            stream: true,
            kv_transfer_params: None,
            data_parallel_rank: None,
            session_id: Some("sess-1".into()),
            max_tokens: Some(16),
        }
    }

    fn router(engine: Arc<ScriptedEngine>, workers: Vec<Arc<dyn Worker>>) -> TokenRouter {
        let registry = Arc::new(WorkerRegistry::new());
        for worker in workers {
            registry.register(worker);
        }
        let policies = Arc::new(PolicyRegistry::new(PolicyConfig::RoundRobin));
        let mut info = AtomPrefillInfo::default();
        info.tp_sizes.insert("http://prefill:8000".into(), 2);
        info.tp_sizes.insert("http://prefill:8000@0".into(), 2);
        info.tp_sizes.insert("http://prefill:8000@1".into(), 2);
        TokenRouter::new(
            registry,
            policies,
            engine,
            Arc::new(info),
            AtomPdRankMappingPolicy::Idx2Idx,
            BackendType::Atom,
        )
    }

    #[tokio::test]
    async fn pd_forwards_decode_events_one_by_one_and_keeps_request_id() {
        let engine = Arc::new(ScriptedEngine::default());
        let prefill = dp_prefill("http://prefill:8000", 0, 2);
        let prefill_rank_1 = dp_prefill("http://prefill:8000", 1, 2);
        let decode = dp_decode("http://decode:8000", 1, 2);
        let router = router(engine.clone(), vec![prefill, prefill_rank_1, decode]);
        let mut stream = router.generate(request()).await.unwrap();
        assert!(
            engine.prefill_completed.load(Ordering::SeqCst),
            "prefill stream was dropped before the RPC completed"
        );
        let first = stream.next().await.unwrap().unwrap();
        let second = stream.next().await.unwrap().unwrap();
        assert_eq!(first.token_ids, vec![11]);
        assert_eq!(second.token_ids, vec![12]);
        assert!(stream.next().await.is_none());

        let calls = engine.calls.lock().unwrap();
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].request_id, "req-1");
        assert_eq!(calls[0].token_ids, vec![1, 2, 3, 4]);
        assert_eq!(calls[0].max_tokens, Some(1));
        assert_eq!(calls[1].request_id, "req-1");
        assert_eq!(calls[1].token_ids, vec![1, 2, 3, 4]);
        assert_eq!(
            calls[1].kv_transfer_params.as_ref().unwrap()["prompt_token_ids"],
            json!([1, 2, 3, 4])
        );
        assert!(calls[1].kv_transfer_params.as_ref().unwrap()["remote_tp_size"].is_number());
        assert_eq!(calls[1].data_parallel_rank, Some(1));
        let urls = engine.urls.lock().unwrap();
        assert!(urls[0].ends_with("@1"), "idx2idx prefill url {}", urls[0]);
        drop(urls);
        drop(calls);
        drop(stream);
        for _ in 0..30 {
            tokio::task::yield_now().await;
        }
        assert!(
            engine.aborts.lock().unwrap().is_empty(),
            "a finished request must not be aborted"
        );
    }

    #[tokio::test]
    async fn prefill_error_does_not_call_decode() {
        let engine = Arc::new(ScriptedEngine {
            fail_prefill: true,
            ..ScriptedEngine::default()
        });
        let router = router(
            engine.clone(),
            vec![
                dp_prefill("http://prefill:8000", 1, 2),
                dp_decode("http://decode:8000", 1, 2),
            ],
        );
        let error = router.generate(request()).await.unwrap_err();
        assert_eq!(error.status_code(), http::StatusCode::BAD_GATEWAY);
        assert_eq!(error.code(), "engine_error");
        assert_eq!(engine.urls.lock().unwrap().len(), 1);
    }

    #[tokio::test]
    async fn dropping_the_stream_aborts_prefill_and_decode() {
        let engine = Arc::new(ScriptedEngine::default());
        let router = router(
            engine.clone(),
            vec![
                dp_prefill("http://prefill:8000", 1, 2),
                dp_decode("http://decode:8000", 1, 2),
            ],
        );
        let mut stream = router.generate(request()).await.unwrap();
        let _ = stream.next().await;
        drop(stream);
        for _ in 0..30 {
            tokio::task::yield_now().await;
        }
        let aborts = engine.aborts.lock().unwrap();
        assert!(
            aborts
                .iter()
                .any(|(url, ids)| url.contains("prefill") && ids == &vec!["req-1".to_string()]),
            "prefill abort missing: {aborts:?}"
        );
        assert!(
            aborts
                .iter()
                .any(|(url, ids)| url.contains("decode") && ids == &vec!["req-1".to_string()]),
            "decode abort missing: {aborts:?}"
        );
    }

    #[tokio::test]
    async fn explicit_abort_cancels_inflight_request() {
        let engine = Arc::new(ScriptedEngine::default());
        let router = router(
            engine.clone(),
            vec![make_regular_grpc("grpc://engine:50051", "m")],
        );
        let _stream = router.generate(request()).await.unwrap();
        router.abort(&["req-1".into()]).await.unwrap();
        for _ in 0..30 {
            tokio::task::yield_now().await;
        }
        let aborts = engine.aborts.lock().unwrap();
        assert_eq!(aborts.len(), 1);
        assert_eq!(aborts[0].1, vec!["req-1".to_string()]);
    }

    #[tokio::test]
    async fn same_token_ids_select_the_same_worker() {
        let engine = Arc::new(ScriptedEngine::default());
        let registry = Arc::new(WorkerRegistry::new());
        registry.register(make_regular_grpc("grpc://a:50051", "m"));
        registry.register(make_regular_grpc("grpc://b:50051", "m"));
        let policies = Arc::new(PolicyRegistry::new(PolicyConfig::PrefixHash {
            prefix_token_count: 256,
            load_factor: 1.25,
        }));
        let router = TokenRouter::new(
            registry,
            policies,
            engine.clone(),
            Arc::new(AtomPrefillInfo::default()),
            AtomPdRankMappingPolicy::None,
            BackendType::Atom,
        );
        let mut first = request();
        first.request_id = "a".into();
        first.session_id = None;
        let mut second = request();
        second.request_id = "b".into();
        second.session_id = None;
        let first_stream = router.generate(first).await.unwrap();
        drop(first_stream);
        let _ = router.generate(second).await.unwrap();
        let urls = engine.urls.lock().unwrap();
        assert_eq!(urls.len(), 2);
        assert_eq!(urls[0], urls[1]);
    }

    #[tokio::test]
    async fn empty_tokens_are_rejected_before_routing() {
        let engine = Arc::new(ScriptedEngine::default());
        let router = router(
            engine.clone(),
            vec![make_regular_grpc("grpc://a:50051", "m")],
        );
        let mut body = request();
        body.token_ids.clear();
        let error = router.generate(body).await.unwrap_err();
        assert_eq!(error.status_code(), http::StatusCode::BAD_REQUEST);
        assert!(engine.urls.lock().unwrap().is_empty());
    }

    #[tokio::test]
    async fn duplicate_request_id_is_rejected_while_the_first_is_inflight() {
        let engine = Arc::new(ScriptedEngine::default());
        let router = router(
            engine.clone(),
            vec![make_regular_grpc("grpc://engine:50051", "m")],
        );
        let _first = router.generate(request()).await.unwrap();
        let error = router.generate(request()).await.unwrap_err();
        assert_eq!(error.code(), "duplicate_request_id");
        assert_eq!(error.status_code(), http::StatusCode::CONFLICT);
        assert_eq!(engine.urls.lock().unwrap().len(), 1);
    }

    #[tokio::test]
    async fn invalid_session_id_is_rejected_before_routing() {
        let engine = Arc::new(ScriptedEngine::default());
        let router = router(
            engine.clone(),
            vec![make_regular_grpc("grpc://engine:50051", "m")],
        );
        let mut body = request();
        body.session_id = Some("sess\n1".into());
        let error = router.generate(body).await.unwrap_err();
        assert_eq!(error.code(), "invalid_session_id");
        assert_eq!(error.status_code(), http::StatusCode::BAD_REQUEST);
        assert!(engine.urls.lock().unwrap().is_empty());
    }

    #[test]
    fn kv_transfer_info_uses_http_on_the_grpc_worker() {
        let worker = make_regular_grpc("grpc://engine:50051", "m");
        assert_eq!(
            kv_transfer_info_url(worker.as_ref()),
            "http://engine:50051/kv_transfer_info"
        );
    }

    #[test]
    fn grpc_endpoint_strips_scheme_and_dp_rank() {
        let worker = dp_prefill("grpc://engine:50051", 3, 4);
        assert_eq!(
            super::super::grpc_endpoint(worker.as_ref()),
            "http://engine:50051"
        );
        let _ = ConnectionMode::Grpc { port: None };
    }

    fn dp_prefill(base: &str, rank: usize, size: usize) -> Arc<dyn Worker> {
        Arc::new(
            DPAwareWorkerBuilder::new(base, rank, size)
                .worker_type(WorkerType::Prefill {
                    bootstrap_port: None,
                })
                .connection_mode(ConnectionMode::Grpc { port: None })
                .model_id("m")
                .build(),
        )
    }

    fn dp_decode(base: &str, rank: usize, size: usize) -> Arc<dyn Worker> {
        Arc::new(
            DPAwareWorkerBuilder::new(base, rank, size)
                .worker_type(WorkerType::Decode)
                .connection_mode(ConnectionMode::Grpc { port: None })
                .model_id("m")
                .build(),
        )
    }
}
