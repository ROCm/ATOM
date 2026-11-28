// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// Copyright (c) 2026 ATOM Project Contributors
//! ATOM's own P/D flow over `atom.grpc.v1`: `AtomAdapter` fills a JSON view of the
//! request fields it reads and writes, which each GenerateRequest then takes.

use std::collections::{HashMap, HashSet};
use std::pin::Pin;
use std::sync::{Arc, Mutex, RwLock, Weak};
use std::task::{Context, Poll};
use std::time::{Duration, Instant};

use futures::{Stream, StreamExt};
use http::StatusCode;
use serde_json::{json, Value};
use tracing::error;

use super::request::generate_request;
use super::stream::{http_error, AtomStream};
use super::AtomClient;
use crate::atom_engine::grpc::{
    generate_response::Response, GenerateComplete, GenerateRequest, GenerateResponse,
};
use crate::config::types::AtomPdRankMappingPolicy;
use crate::core::placement::backend::atom::{
    apply_atom_pd_rank_mapping_policy, parse_atom_prefill_info, AtomAdapter, AtomPrefillInfo,
};
use crate::core::placement::backend::BackendAdapter;
use crate::core::placement::PlacementPlan;
use crate::core::worker::RuntimeType;
use crate::core::{Worker, WorkerLoadGuard, WorkerRegistry};
use crate::observability::events::{Event, RequestPDSentEvent};
use crate::routers::grpc::engine::worker_client_cache::{get_grpc_client_from_worker, GrpcClient};
use crate::routers::prepare::generation_payload::GenerationPayload;
use crate::routers::token_handle::engine_error::EngineError;
use crate::routers::token_handle::token_chunk::TokenChunk;
use crate::routers::token_handle::token_handle::{SourceClock, TokenHandle, TokenSource};
use crate::tokenizer::traits::Tokenizer;

/// Runs ATOM pairs as ATOM's HTTP P/D router does: relay, or chunked when P's
/// instance reports a chunked-transfer template.
pub(crate) struct AtomPairs {
    pub(super) info: RwLock<Arc<AtomPrefillInfo>>,
    /// The worker each URL's info was read for, and each instance's (its base URL):
    /// a worker registered again at that URL is another, whose info is read afresh.
    /// Taken after `info`, as a read publishes both. Each with its instance's base URL.
    pub(super) read_for: RwLock<HashMap<String, ReadFor>>,
    /// Per instance, held while its GetServerInfo is read, so concurrent first pairs of
    /// its ranks read it once; it keeps a failed read's error, and when.
    reading: Mutex<HashMap<String, Arc<tokio::sync::Mutex<Option<Failed>>>>>,
    mapping: AtomPdRankMappingPolicy,
    workers: Arc<WorkerRegistry>,
}

/// The worker a URL's info was read for, and its instance's base URL.
type ReadFor = (Weak<dyn Worker>, String);

/// A failed GetServerInfo read: when, and the error each pair that waited for it gets.
type Failed = (Instant, StatusCode, String);

impl AtomPairs {
    /// Reads each registered ATOM prefill's `kv_transfer_info` from GetServerInfo,
    /// once per instance, as the HTTP router reads `/kv_transfer_info`.
    pub(crate) async fn new(
        workers: Arc<WorkerRegistry>,
        mapping: AtomPdRankMappingPolicy,
    ) -> Result<Self, String> {
        let mut info = AtomPrefillInfo::default();
        let mut read_for = HashMap::new();
        for worker in workers.get_prefill_workers().iter().filter(|w| is_atom(w)) {
            learn(&mut info, worker).await.map_err(|e| e.to_string())?;
            let read = (Arc::downgrade(worker), base_url(worker));
            read_for.insert(worker.url().to_string(), read.clone());
            read_for.insert(base_url(worker), read);
        }
        Ok(Self {
            info: RwLock::new(Arc::new(info)),
            read_for: RwLock::new(read_for),
            reading: Mutex::default(),
            mapping,
            workers,
        })
    }

    /// The prefills' info, `prefill`'s read on its first pair when it registered later
    /// or again. Waiting for another pair's read and reading take at most the prefill's
    /// health-check timeout together; pairs that waited for a read that failed get its
    /// error.
    pub(super) async fn adapter_for(
        &self,
        prefill: &Arc<dyn Worker>,
    ) -> Result<AtomAdapter, EngineError> {
        let url = prefill.url();
        let this = Arc::downgrade(prefill);
        // The info this prefill was read into, taken with the check that it was, so no
        // read publishing meanwhile can have pruned it.
        let cached = || {
            let info = self.info.read().unwrap();
            let read_for = self.read_for.read().unwrap();
            let read = read_for
                .get(url)
                .is_some_and(|(known, _)| known.ptr_eq(&this));
            read.then(|| AtomAdapter::new(info.clone()))
        };
        if let Some(adapter) = cached() {
            return Ok(adapter);
        }
        let base = base_url(prefill);
        let gate = self
            .reading
            .lock()
            .unwrap()
            .entry(base.clone())
            .or_default()
            .clone();
        let started = Instant::now();
        let deadline = Duration::from_secs(prefill.metadata().health_config.timeout_secs);
        let read = async {
            let mut failed = gate.lock().await;
            if let Some(adapter) = cached() {
                return Ok(adapter);
            }
            if let Some((_, status, message)) = failed.as_ref().filter(|f| f.0 >= started) {
                let (status, message) = (*status, message.clone());
                return Err(EngineError::Http { status, message });
            }
            let read = self.read(prefill, &base, this.clone()).await;
            if let Err(e) = &read {
                *failed = Some(match e {
                    EngineError::Http { status, message } => {
                        (Instant::now(), *status, message.clone())
                    }
                    e => (Instant::now(), StatusCode::BAD_GATEWAY, e.to_string()),
                });
            }
            read
        };
        tokio::time::timeout(deadline, read)
            .await
            .map_err(|_| EngineError::Http {
                status: StatusCode::GATEWAY_TIMEOUT,
                message: format!("GetServerInfo on {url} timed out after {deadline:?}"),
            })?
    }

    /// Reads `prefill`'s info, or its instance's from a sibling rank still registered,
    /// and publishes it.
    async fn read(
        &self,
        prefill: &Arc<dyn Worker>,
        base: &str,
        this: Weak<dyn Worker>,
    ) -> Result<AtomAdapter, EngineError> {
        let (url, base) = (prefill.url(), base.to_string());
        let mut learned = AtomPrefillInfo::default();
        // A rank still registered read this instance: its info serves this sibling too,
        // unless this rank replaced one, whose siblings may be the old instance's.
        let sibling = {
            let read_for = self.read_for.read().unwrap();
            let replaced = read_for.contains_key(url);
            let sibling = read_for.get(&base).and_then(|(w, _)| w.upgrade());
            sibling.filter(|_| !replaced)
        };
        let registered = |w: &Arc<dyn Worker>| {
            self.workers
                .get_by_url(w.url())
                .is_some_and(|r| Arc::ptr_eq(&r, w))
        };
        if sibling.is_some_and(|w| registered(&w)) {
            let info = self.info.read().unwrap();
            if let Some(tp) = info.tp_sizes.get(&base) {
                learned.tp_sizes.insert(base.clone(), *tp);
            }
            if let Some(template) = info.chunked_transfer.get(&base) {
                learned
                    .chunked_transfer
                    .insert(base.clone(), template.clone());
            }
        }
        let read_instance = !learned.tp_sizes.contains_key(&base);
        learn(&mut learned, prefill).await?;
        let mut info = self.info.write().unwrap();
        let mut merged = AtomPrefillInfo::clone(&info);
        let mut read_for = self.read_for.write().unwrap();
        // A URL no registered worker holds any more is forgotten, with its info; a removed
        // rank's instance too, so a rank registered there again reads it afresh.
        let mut removed = HashSet::new();
        read_for.retain(|url, (w, base)| {
            let kept = w.upgrade().is_some_and(|w| registered(&w));
            if !kept && url != base {
                removed.insert(base.clone());
            }
            kept
        });
        read_for.retain(|url, (_, base)| !(url == base && removed.contains(base)));
        if read_instance {
            // What was read replaces what every rank of this instance took before.
            let ranks: Vec<String> = read_for
                .iter()
                .filter(|(_, (_, rank_base))| *rank_base == base)
                .map(|(rank, _)| rank.clone())
                .collect();
            for rank in ranks {
                learned
                    .tp_sizes
                    .insert(rank.clone(), learned.tp_sizes[&base]);
                if let Some(template) = learned.chunked_transfer.get(&base).cloned() {
                    learned.chunked_transfer.insert(rank, template);
                }
            }
        }
        for url in learned.tp_sizes.keys() {
            merged.chunked_transfer.remove(url);
        }
        merged.tp_sizes.extend(learned.tp_sizes);
        merged.chunked_transfer.extend(learned.chunked_transfer);
        if read_instance {
            read_for.insert(base.clone(), (this.clone(), base.clone()));
        }
        read_for.insert(url.to_string(), (this, base));
        merged.tp_sizes.retain(|url, _| read_for.contains_key(url));
        merged
            .chunked_transfer
            .retain(|url, _| read_for.contains_key(url));
        *info = Arc::new(merged);
        // The gates no first pair is waiting on.
        self.reading
            .lock()
            .unwrap()
            .retain(|_, gate| Arc::strong_count(gate) > 1);
        Ok(AtomAdapter::new(info.clone()))
    }

    /// One attempt on the planned pair, P picked by the rank mapping. Both loads
    /// are taken first, as `ReservedPair` takes them; P's ends when P completes.
    pub(crate) async fn dispatch(
        &self,
        prefill: &Arc<dyn Worker>,
        decode: &Arc<dyn Worker>,
        payload: &GenerationPayload,
        tokenizer: Option<&dyn Tokenizer>,
    ) -> Result<TokenHandle, EngineError> {
        let started = Instant::now();
        let prefill =
            apply_atom_pd_rank_mapping_policy(self.mapping, &self.workers, prefill.clone(), decode);
        let prefill_load = WorkerLoadGuard::new(prefill.clone(), None);
        let decode_load = WorkerLoadGuard::new(decode.clone(), None);
        RequestPDSentEvent {
            prefill_url: prefill.url(),
            decode_url: decode.url(),
        }
        .emit();
        let stream = self
            .run(&prefill, decode, payload, tokenizer, prefill_load)
            .await
            .inspect_err(|e| {
                error!(prefill = prefill.url(), decode = decode.url(), error = %e, "ATOM P/D failed")
            })?;
        Ok(TokenHandle::new(DecodeStream {
            stream,
            _load: decode_load,
            started,
        }))
    }

    async fn run(
        &self,
        prefill: &Arc<dyn Worker>,
        decode: &Arc<dyn Worker>,
        payload: &GenerationPayload,
        tokenizer: Option<&dyn Tokenizer>,
        prefill_load: WorkerLoadGuard,
    ) -> Result<AtomStream, EngineError> {
        let adapter = &self.adapter_for(prefill).await?;
        let ctx = adapter
            .prepare_pair(prefill.as_ref(), decode.as_ref())
            .map_err(internal)?;
        let view = json!({
            "stream": payload.stream,
            "max_tokens": payload.sampling.max_new_tokens,
            "n": payload.sampling.n,
        });
        let (mut prefill_view, mut decode_view) = (view.clone(), view);
        adapter
            .inject_prefill_fields(&mut prefill_view, &ctx)
            .map_err(internal)?;
        adapter
            .inject_decode_fields(&mut decode_view, &ctx)
            .map_err(internal)?;
        let mut prefill_request = generate_request(payload, prefill.as_ref(), tokenizer)?;
        let mut decode_request = generate_request(payload, decode.as_ref(), tokenizer)?;
        take_view(&mut prefill_request, &prefill_view);
        take_view(&mut decode_request, &decode_view);
        let prefill_client = atom_client(prefill)
            .await
            .ok_or_else(|| no_client(prefill))?;
        let decode_client = atom_client(decode).await.ok_or_else(|| no_client(decode))?;

        if decode_view["kv_transfer_params"]["chunked_transfer"] == true {
            // D's stream is returned only after P completes; either failing drops,
            // and so cancels, the other's call.
            let (_, stream) = tokio::try_join!(
                async {
                    run_prefill(&prefill_client, prefill_request).await?;
                    drop(prefill_load);
                    Ok::<_, EngineError>(())
                },
                open_decode(&decode_client, decode_request),
            )?;
            return Ok(stream);
        }
        let complete = run_prefill(&prefill_client, prefill_request).await?;
        // P has completed; D's stream must not keep P loaded.
        drop(prefill_load);
        let kv = complete
            .kv_transfer_params_json
            .ok_or_else(|| bad_gateway("Prefill response missing kv_transfer_params".into()))?;
        let mut kv: Value = serde_json::from_str(&kv)
            .map_err(|e| bad_gateway(format!("Prefill kv_transfer_params parse error: {e}")))?;
        adapter.enrich_decode_kv(&mut kv, &ctx).map_err(internal)?;
        decode_request.kv_transfer_params_json = Some(kv.to_string());
        open_decode(&decode_client, decode_request).await
    }
}

/// The pair `placement` plans when both of its workers are ATOM's.
pub(crate) fn atom_pair(placement: &PlacementPlan) -> Option<[&Arc<dyn Worker>; 2]> {
    match placement {
        PlacementPlan::Pair {
            prefill, decode, ..
        } if is_atom(prefill) && is_atom(decode) => Some([prefill, decode]),
        _ => None,
    }
}

/// The refusal of a pair `placement` plans from an ATOM and a non-ATOM worker, between
/// which neither P/D flow hands the KV over.
pub(crate) fn mixed_pair_refusal(placement: &PlacementPlan) -> Option<EngineError> {
    match placement {
        PlacementPlan::Pair {
            prefill, decode, ..
        } if is_atom(prefill) != is_atom(decode) => Some(internal(format!(
            "P/D pair mixes runtimes: prefill {} is {:?}, decode {} is {:?}; an ATOM worker \
             pairs only with an ATOM worker",
            prefill.url(),
            prefill.metadata().runtime_type,
            decode.url(),
            decode.metadata().runtime_type,
        ))),
        _ => None,
    }
}

/// The URL of `worker`'s instance, which its ranks share.
fn base_url(worker: &Arc<dyn Worker>) -> String {
    worker.base_url().trim_end_matches('/').to_string()
}

fn is_atom(worker: &Arc<dyn Worker>) -> bool {
    worker.metadata().runtime_type == RuntimeType::Atom
}

async fn atom_client(worker: &Arc<dyn Worker>) -> Option<AtomClient> {
    match get_grpc_client_from_worker(worker).await {
        Ok(GrpcClient::Atom(client)) => Some(client),
        _ => None,
    }
}

/// Adds `worker`'s TP size and chunked template to `info`, its instance's GetServerInfo
/// read once.
async fn learn(info: &mut AtomPrefillInfo, worker: &Arc<dyn Worker>) -> Result<(), EngineError> {
    let base_url = base_url(worker);
    if !info.tp_sizes.contains_key(&base_url) {
        let data = kv_transfer_info(worker).await?;
        let source = format!("{base_url} GetServerInfo");
        let (tp, chunked) =
            parse_atom_prefill_info(&data, &source, &base_url).map_err(bad_gateway)?;
        info.tp_sizes.insert(base_url.clone(), tp);
        if let Some(template) = chunked {
            info.chunked_transfer.insert(base_url.clone(), template);
        }
    }
    let url = worker.url().to_string();
    info.tp_sizes.insert(url.clone(), info.tp_sizes[&base_url]);
    if let Some(template) = info.chunked_transfer.get(&base_url).cloned() {
        info.chunked_transfer.insert(url, template);
    }
    Ok(())
}

/// `worker`'s instance's `/kv_transfer_info` body, as GetServerInfo carries it. Connecting
/// and reading it take at most the worker's health-check timeout together.
async fn kv_transfer_info(worker: &Arc<dyn Worker>) -> Result<Value, EngineError> {
    let url = worker.url();
    let deadline = Duration::from_secs(worker.metadata().health_config.timeout_secs);
    let read = async {
        let client = atom_client(worker)
            .await
            .ok_or_else(|| bad_gateway(format!("Failed to connect to ATOM prefill {url}")))?;
        client
            .get_server_info()
            .await
            .map_err(|e| bad_gateway(format!("GetServerInfo on {url} failed: {e}")))
    };
    let info = tokio::time::timeout(deadline, read)
        .await
        .map_err(|_| EngineError::Http {
            status: StatusCode::GATEWAY_TIMEOUT,
            message: format!("GetServerInfo on {url} timed out after {deadline:?}"),
        })??;
    serde_json::from_str(&info.kv_transfer_info_json)
        .map_err(|e| bad_gateway(format!("Parse {url} GetServerInfo kv_transfer_info: {e}")))
}

/// The fields `AtomAdapter` wrote into `view`, onto `request`.
fn take_view(request: &mut GenerateRequest, view: &Value) {
    request.kv_transfer_params_json = view.get("kv_transfer_params").map(Value::to_string);
    request.stream = view["stream"] == true;
    if let Some(sampling) = request.sampling.as_mut() {
        sampling.max_tokens = view["max_tokens"].as_i64();
    }
}

/// P's call up to its Complete. Failures that are not ATOM's own GenerateError
/// take the 502 ATOM's HTTP P/D router answers them with.
async fn run_prefill(
    client: &AtomClient,
    request: GenerateRequest,
) -> Result<GenerateComplete, EngineError> {
    let mut stream = client
        .generate(request)
        .await
        .map_err(|s| bad_gateway(format!("Prefill server error: {s}")))?;
    loop {
        match stream.message().await {
            Ok(Some(GenerateResponse { response })) => match response {
                Some(Response::Complete(complete)) => return Ok(complete),
                Some(Response::Error(error)) => return Err(http_error(error)),
                Some(Response::Chunk(_)) | None => {}
            },
            Ok(None) => return Err(bad_gateway("Prefill stream ended without Complete".into())),
            Err(s) => return Err(bad_gateway(format!("Prefill read error: {s}"))),
        }
    }
}

async fn open_decode(
    client: &AtomClient,
    request: GenerateRequest,
) -> Result<AtomStream, EngineError> {
    AtomStream::open(client, request)
        .await
        .map_err(|e| match e {
            EngineError::Transport(s) => bad_gateway(format!("Decode server error: {s}")),
            e => e,
        })
}

fn no_client(worker: &Arc<dyn Worker>) -> EngineError {
    bad_gateway(format!("Failed to connect to {}", worker.url()))
}

fn bad_gateway(message: String) -> EngineError {
    EngineError::Http {
        status: StatusCode::BAD_GATEWAY,
        message,
    }
}

fn internal(error: impl std::fmt::Display) -> EngineError {
    EngineError::Http {
        status: StatusCode::INTERNAL_SERVER_ERROR,
        message: error.to_string(),
    }
}

/// D's stream, holding D's load until the handle drops.
struct DecodeStream {
    stream: AtomStream,
    _load: WorkerLoadGuard,
    /// When the pair began, so the TTFT a renderer measures includes P's part.
    started: Instant,
}

impl Stream for DecodeStream {
    type Item = Result<TokenChunk, EngineError>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        self.stream.poll_next_unpin(cx)
    }
}

impl TokenSource for DecodeStream {
    fn mark_completed(&mut self) {
        self.stream.mark_completed();
    }

    fn clock(&self) -> Option<SourceClock> {
        let clock = self.stream.clock()?;
        Some(SourceClock {
            started: self.started,
            ..clock
        })
    }
}
