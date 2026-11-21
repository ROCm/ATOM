//! ATOM's own P/D flow over `atom.grpc.v1`: `AtomAdapter` fills a JSON view of the
//! request fields it reads and writes, which each GenerateRequest then takes.

use std::collections::HashSet;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};

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
use crate::routers::token_handle::token_handle::{TokenHandle, TokenSource};
use crate::tokenizer::traits::Tokenizer;

/// Runs ATOM pairs as ATOM's HTTP P/D router does: relay, or chunked when P's
/// instance reports a chunked-transfer template.
pub(crate) struct AtomPairs {
    pub(super) adapter: AtomAdapter,
    mapping: AtomPdRankMappingPolicy,
    workers: Arc<WorkerRegistry>,
}

impl AtomPairs {
    /// Reads each registered ATOM prefill's `kv_transfer_info` from GetServerInfo,
    /// once per instance, as the HTTP router reads `/kv_transfer_info`.
    pub(crate) async fn new(
        workers: Arc<WorkerRegistry>,
        mapping: AtomPdRankMappingPolicy,
    ) -> Result<Self, String> {
        let mut info = AtomPrefillInfo::default();
        let mut queried = HashSet::new();
        for worker in workers.get_prefill_workers().iter().filter(|w| is_atom(w)) {
            let base_url = worker.base_url().trim_end_matches('/').to_string();
            if queried.insert(base_url.clone()) {
                let data = kv_transfer_info(worker).await?;
                let source = format!("{base_url} GetServerInfo");
                let (tp, chunked) = parse_atom_prefill_info(&data, &source, &base_url)?;
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
        }
        Ok(Self {
            adapter: AtomAdapter::new(Arc::new(info)),
            mapping,
            workers,
        })
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
        let adapter = &self.adapter;
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

fn is_atom(worker: &Arc<dyn Worker>) -> bool {
    worker.metadata().runtime_type == RuntimeType::Atom
}

pub(super) async fn atom_client(worker: &Arc<dyn Worker>) -> Option<AtomClient> {
    match get_grpc_client_from_worker(worker).await {
        Ok(GrpcClient::Atom(client)) => Some(client),
        _ => None,
    }
}

/// `worker`'s instance's `/kv_transfer_info` body, as GetServerInfo carries it.
async fn kv_transfer_info(worker: &Arc<dyn Worker>) -> Result<Value, String> {
    let url = worker.url();
    let client = atom_client(worker)
        .await
        .ok_or_else(|| format!("Failed to connect to ATOM prefill {url}"))?;
    let info = client
        .get_server_info()
        .await
        .map_err(|e| format!("GetServerInfo on {url} failed: {e}"))?;
    serde_json::from_str(&info.kv_transfer_info_json)
        .map_err(|e| format!("Parse {url} GetServerInfo kv_transfer_info: {e}"))
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
}
