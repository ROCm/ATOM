use std::{
    collections::{BTreeMap, HashMap, HashSet},
    pin::Pin,
    sync::{
        atomic::{AtomicBool, AtomicI64, Ordering},
        mpsc as std_mpsc, Arc,
    },
    task::{Context, Poll},
    thread,
    thread::JoinHandle,
    time::{Duration, Instant},
};

use blake2::{digest::consts::U8, Blake2b, Digest};
use dashmap::DashMap;
use futures::Stream;
use parking_lot::Mutex;
use prost::Message;
use prost_types::{value::Kind, ListValue, Struct, Value};
use tokio::sync::mpsc;
use tokio_stream::wrappers::UnboundedReceiverStream;

use crate::{
    proto::engine::{
        engine_core_envelope::Payload, EngineCoreEnvelope, Sequence, UtilityCommand,
        UtilityResponse,
    },
    routers::{
        prepare::generation_payload::{GenerationPayload, DEFAULT_MAX_OUTPUT_TOKENS},
        token_handle::{
            engine_error::EngineError,
            token_chunk::{FinishReason, TokenChunk, Usage, WorkerMeta},
            token_handle::{TokenHandle, TokenSource},
        },
    },
};

use super::{
    encode_add_requests_configured,
    transport::{EngineCoreRankSockets, EngineCoreTransportError},
    EngineCoreTransport, ENGINE_CORE_WIRE_VERSION,
};

struct SequenceRoute {
    sender: mpsc::UnboundedSender<Result<TokenChunk, EngineError>>,
    request_id: String,
    prompt_tokens: u32,
    output_tokens: Vec<u32>,
    cached_tokens: u32,
    dp_rank: usize,
    token_cost: u64,
    deadline: Instant,
    sibling_index: usize,
    completion_group: Arc<Mutex<CompletionGroup>>,
}

struct CompletionGroup {
    expected: usize,
    completed: BTreeMap<usize, TokenChunk>,
    sender: mpsc::UnboundedSender<Result<TokenChunk, EngineError>>,
}

enum RankCommand {
    Input(EngineCoreEnvelope),
    Control(EngineCoreEnvelope),
    Shutdown,
}

type UtilityResponses = BTreeMap<String, BTreeMap<usize, UtilityResponse>>;

fn failure_domain_ranks(
    dp_attention_enabled: bool,
    failed_rank: usize,
    all_ranks: &[usize],
) -> Vec<usize> {
    if dp_attention_enabled {
        all_ranks.to_vec()
    } else {
        vec![failed_rank]
    }
}

fn deployment_is_ready(
    dp_attention_enabled: bool,
    healthy_ranks: usize,
    total_ranks: usize,
) -> bool {
    healthy_ranks > 0 && (!dp_attention_enabled || healthy_ranks == total_ranks)
}

pub struct EngineCoreClient {
    _transport_context: Arc<Mutex<EngineCoreTransport>>,
    rank_commands: BTreeMap<usize, std_mpsc::SyncSender<RankCommand>>,
    routes: Arc<DashMap<i64, SequenceRoute>>,
    next_sequence_id: AtomicI64,
    stopped: Arc<AtomicBool>,
    collective_shutdown_started: AtomicBool,
    admission_lock: Mutex<()>,
    scheduler: Mutex<DpScheduler>,
    latest_metrics: DashMap<usize, crate::proto::engine::MetricsSnapshot>,
    rank_health: DashMap<usize, bool>,
    rank_timeout_strikes: DashMap<usize, u32>,
    utility_responses: Mutex<UtilityResponses>,
    poisoned_utility_commands: Mutex<HashSet<String>>,
    utility_execution_lock: tokio::sync::Mutex<()>,
    request_timeout: Duration,
    num_draft_tokens: i32,
    has_per_req_cache: bool,
    max_model_len: Option<i64>,
    max_pool_tokens: Option<i64>,
    shutdown_timeout: Duration,
    dp_attention_enabled: bool,
    worker_handles: Mutex<Vec<JoinHandle<()>>>,
}

struct DpScheduler {
    strategy: String,
    ranks: Vec<usize>,
    requests: BTreeMap<usize, u64>,
    tokens: BTreeMap<usize, u64>,
    round_robin_cursor: usize,
    disabled_ranks: HashSet<usize>,
    session_owners: HashMap<String, usize>,
    session_prompt_tokens: HashMap<String, u64>,
    session_parents: HashMap<String, String>,
    session_last_used: HashMap<String, u64>,
    session_use_counter: u64,
    request_equivalent: u64,
}

struct DpReservation {
    rank: usize,
    token_cost: u64,
    request_cost: u64,
    session_id: Option<String>,
    previous_session_owner: Option<usize>,
    previous_session_prompt_tokens: Option<u64>,
    previous_session_parent: Option<String>,
    previous_session_last_used: Option<u64>,
    evicted_session: Option<(String, usize, Option<u64>, Option<String>, Option<u64>)>,
}

impl DpScheduler {
    fn new(ranks: Vec<usize>, strategy: &str, request_equivalent: u64) -> Self {
        Self {
            strategy: strategy.to_string(),
            requests: ranks.iter().map(|&rank| (rank, 0)).collect(),
            tokens: ranks.iter().map(|&rank| (rank, 0)).collect(),
            ranks,
            round_robin_cursor: 0,
            disabled_ranks: HashSet::new(),
            session_owners: HashMap::new(),
            session_prompt_tokens: HashMap::new(),
            session_parents: HashMap::new(),
            session_last_used: HashMap::new(),
            session_use_counter: 0,
            request_equivalent,
        }
    }

    fn select_and_charge(
        &mut self,
        token_cost: u64,
        request_cost: u64,
        preferred_rank: Option<usize>,
        session_id: Option<&str>,
        parent_session_id: Option<&str>,
    ) -> Result<DpReservation, EngineError> {
        const MAX_INFLIGHT_PER_RANK: u64 = 1_024;
        let pinned_rank = preferred_rank.or_else(|| {
            session_id.and_then(|session_id| self.session_owners.get(session_id).copied())
        });
        let rank = if let Some(rank) = pinned_rank {
            if !self.requests.contains_key(&rank) {
                return Err(EngineError::RequestBuildFailed(format!(
                    "requested EngineCore DP rank {rank} does not exist"
                )));
            }
            if self.disabled_ranks.contains(&rank) {
                return Err(EngineError::ConnectionAcquireFailed(format!(
                    "requested EngineCore DP rank {rank} is unavailable"
                )));
            }
            (self.requests[&rank].saturating_add(request_cost) <= MAX_INFLIGHT_PER_RANK)
                .then_some(rank)
        } else if let Some(session_id) = session_id {
            self.ranks
                .iter()
                .copied()
                .filter(|rank| {
                    !self.disabled_ranks.contains(rank)
                        && self.requests[rank].saturating_add(request_cost) <= MAX_INFLIGHT_PER_RANK
                })
                .min_by_key(|rank| {
                    (
                        self.tokens[rank]
                            .saturating_add(self.requests[rank] * self.request_equivalent),
                        std::cmp::Reverse(session_rank_score(session_id, *rank)),
                    )
                })
        } else if self.strategy == "round_robin" {
            let count = self.ranks.len();
            let mut selected = None;
            for _ in 0..count {
                let rank = self.ranks[self.round_robin_cursor % count];
                self.round_robin_cursor = self.round_robin_cursor.wrapping_add(1);
                if !self.disabled_ranks.contains(&rank)
                    && self.requests[&rank].saturating_add(request_cost) <= MAX_INFLIGHT_PER_RANK
                {
                    selected = Some(rank);
                    break;
                }
            }
            selected
        } else {
            let count = self.ranks.len();
            let start = self.round_robin_cursor % count;
            self.round_robin_cursor = self.round_robin_cursor.wrapping_add(1);
            (0..count)
                .map(|offset| self.ranks[(start + offset) % count])
                .filter(|rank| {
                    !self.disabled_ranks.contains(rank)
                        && self.requests[rank].saturating_add(request_cost) <= MAX_INFLIGHT_PER_RANK
                })
                .min_by_key(|rank| {
                    if self.strategy == "least_requests" {
                        (self.requests[rank], self.tokens[rank])
                    } else {
                        (
                            self.tokens[rank]
                                .saturating_add(self.requests[rank] * self.request_equivalent),
                            self.requests[rank],
                        )
                    }
                })
        }
        .ok_or_else(|| {
            EngineError::ConnectionAcquireFailed(
                "all EngineCore rank queues are at capacity".to_string(),
            )
        })?;
        let previous_session_owner = session_id.and_then(|id| self.session_owners.get(id).copied());
        let effective_token_cost = if previous_session_owner == Some(rank) {
            session_id
                .and_then(|id| self.session_prompt_tokens.get(id).copied())
                .map_or(token_cost, |previous| token_cost.saturating_sub(previous))
        } else {
            token_cost
        };
        *self.requests.get_mut(&rank).expect("rank initialized") += request_cost;
        *self.tokens.get_mut(&rank).expect("rank initialized") +=
            effective_token_cost.saturating_mul(request_cost);
        let previous_session_prompt_tokens =
            session_id.and_then(|id| self.session_prompt_tokens.get(id).copied());
        let previous_session_parent =
            session_id.and_then(|id| self.session_parents.get(id).cloned());
        let previous_session_last_used =
            session_id.and_then(|id| self.session_last_used.get(id).copied());
        let mut evicted_session = None;
        if let Some(session_id) = session_id {
            const MAX_SESSION_OWNERS: usize = 4_096;
            if !self.session_owners.contains_key(session_id)
                && self.session_owners.len() >= MAX_SESSION_OWNERS
            {
                if let Some(expired) = self
                    .session_owners
                    .keys()
                    .min_by_key(|id| self.session_last_used.get(*id).copied().unwrap_or(0))
                    .cloned()
                {
                    if let Some(owner) = self.session_owners.remove(&expired) {
                        evicted_session = Some((
                            expired.clone(),
                            owner,
                            self.session_prompt_tokens.remove(&expired),
                            self.session_parents.remove(&expired),
                            self.session_last_used.remove(&expired),
                        ));
                    }
                }
            }
            self.session_owners.insert(session_id.to_string(), rank);
            self.session_prompt_tokens
                .insert(session_id.to_string(), token_cost);
            if let Some(parent_session_id) = parent_session_id {
                self.session_parents
                    .insert(session_id.to_string(), parent_session_id.to_string());
            }
            self.session_use_counter = self.session_use_counter.wrapping_add(1);
            self.session_last_used
                .insert(session_id.to_string(), self.session_use_counter);
        }
        Ok(DpReservation {
            rank,
            token_cost: effective_token_cost,
            request_cost,
            session_id: session_id.map(str::to_string),
            previous_session_owner,
            previous_session_prompt_tokens,
            previous_session_parent,
            previous_session_last_used,
            evicted_session,
        })
    }

    fn rollback(&mut self, reservation: &DpReservation) {
        if let Some(requests) = self.requests.get_mut(&reservation.rank) {
            *requests = requests.saturating_sub(reservation.request_cost);
        }
        if let Some(tokens) = self.tokens.get_mut(&reservation.rank) {
            *tokens = tokens.saturating_sub(
                reservation
                    .token_cost
                    .saturating_mul(reservation.request_cost),
            );
        }
        if let Some(session_id) = reservation.session_id.as_ref() {
            match reservation.previous_session_owner {
                Some(owner) => {
                    self.session_owners.insert(session_id.clone(), owner);
                }
                None => {
                    self.session_owners.remove(session_id);
                }
            }
            match reservation.previous_session_prompt_tokens {
                Some(tokens) => {
                    self.session_prompt_tokens
                        .insert(session_id.clone(), tokens);
                }
                None => {
                    self.session_prompt_tokens.remove(session_id);
                }
            }
            match reservation.previous_session_parent.as_ref() {
                Some(parent) => {
                    self.session_parents
                        .insert(session_id.clone(), parent.clone());
                }
                None => {
                    self.session_parents.remove(session_id);
                }
            }
            match reservation.previous_session_last_used {
                Some(last_used) => {
                    self.session_last_used.insert(session_id.clone(), last_used);
                }
                None => {
                    self.session_last_used.remove(session_id);
                }
            }
        }
        if let Some((session_id, owner, prompt_tokens, parent, last_used)) =
            reservation.evicted_session.as_ref()
        {
            self.session_owners.insert(session_id.clone(), *owner);
            if let Some(tokens) = prompt_tokens {
                self.session_prompt_tokens
                    .insert(session_id.clone(), *tokens);
            }
            if let Some(parent) = parent {
                self.session_parents
                    .insert(session_id.clone(), parent.clone());
            }
            if let Some(last_used) = last_used {
                self.session_last_used
                    .insert(session_id.clone(), *last_used);
            }
        }
    }

    fn disable_rank(&mut self, rank: usize) {
        self.disabled_ranks.insert(rank);
        let sessions: Vec<_> = self
            .session_owners
            .iter()
            .filter_map(|(session_id, owner)| (*owner == rank).then(|| session_id.clone()))
            .collect();
        for session_id in sessions {
            self.session_owners.remove(&session_id);
            self.session_prompt_tokens.remove(&session_id);
            self.session_parents.remove(&session_id);
            self.session_last_used.remove(&session_id);
        }
    }

    fn release(&mut self, rank: usize, token_cost: u64) {
        if let Some(requests) = self.requests.get_mut(&rank) {
            *requests = requests.saturating_sub(1);
        }
        if let Some(tokens) = self.tokens.get_mut(&rank) {
            *tokens = tokens.saturating_sub(token_cost);
        }
    }

    fn release_tokens(&mut self, rank: usize, token_cost: u64) {
        if let Some(tokens) = self.tokens.get_mut(&rank) {
            *tokens = tokens.saturating_sub(token_cost);
        }
    }
}

impl std::fmt::Debug for EngineCoreClient {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("EngineCoreClient")
            .field("active_routes", &self.routes.len())
            .finish_non_exhaustive()
    }
}

impl EngineCoreClient {
    pub fn new(
        transport: Arc<Mutex<EngineCoreTransport>>,
        dp_load_balance: &str,
        dp_lb_request_equivalent: u64,
        request_timeout: Duration,
        num_draft_tokens: i32,
        has_per_req_cache: bool,
        max_model_len: Option<i64>,
        max_pool_tokens: Option<i64>,
        shutdown_timeout: Duration,
        dp_attention_enabled: bool,
    ) -> Result<Arc<Self>, EngineError> {
        let rank_sockets = transport
            .lock()
            .take_rank_sockets()
            .map_err(|error| EngineError::ConnectionAcquireFailed(error.to_string()))?;
        let ranks: Vec<_> = rank_sockets.keys().copied().collect();
        let mut rank_commands = BTreeMap::new();
        let mut rank_receivers = Vec::new();
        for (dp_rank, sockets) in rank_sockets {
            let (sender, receiver) = std_mpsc::sync_channel(1_024);
            rank_commands.insert(dp_rank, sender);
            rank_receivers.push((dp_rank, sockets, receiver));
        }
        let client = Arc::new(Self {
            _transport_context: transport,
            rank_commands,
            routes: Arc::new(DashMap::new()),
            next_sequence_id: AtomicI64::new(1),
            stopped: Arc::new(AtomicBool::new(false)),
            collective_shutdown_started: AtomicBool::new(false),
            admission_lock: Mutex::new(()),
            scheduler: Mutex::new(DpScheduler::new(
                ranks.clone(),
                dp_load_balance,
                dp_lb_request_equivalent,
            )),
            latest_metrics: DashMap::new(),
            rank_health: ranks.iter().map(|&rank| (rank, true)).collect(),
            rank_timeout_strikes: ranks.into_iter().map(|rank| (rank, 0)).collect(),
            utility_responses: Mutex::new(BTreeMap::new()),
            poisoned_utility_commands: Mutex::new(HashSet::new()),
            utility_execution_lock: tokio::sync::Mutex::new(()),
            request_timeout,
            num_draft_tokens,
            has_per_req_cache,
            max_model_len,
            max_pool_tokens,
            shutdown_timeout,
            dp_attention_enabled,
            worker_handles: Mutex::new(Vec::new()),
        });
        for (dp_rank, sockets, receiver) in rank_receivers {
            if let Err(error) = Self::start_rank_worker(&client, dp_rank, sockets, receiver) {
                let _ = client.shutdown();
                return Err(error);
            }
        }
        Ok(client)
    }

    fn start_rank_worker(
        client: &Arc<Self>,
        dp_rank: usize,
        sockets: EngineCoreRankSockets,
        receiver: std_mpsc::Receiver<RankCommand>,
    ) -> Result<(), EngineError> {
        let weak = Arc::downgrade(client);
        let handle = thread::Builder::new()
            .name(format!("EngineCore-DP{dp_rank}"))
            .spawn(move || {
                while let Some(client) = weak.upgrade() {
                    let mut received = false;
                    loop {
                        match receiver.try_recv() {
                            Ok(RankCommand::Input(envelope)) => {
                                received = true;
                                if let Err(error) = sockets.send_input(dp_rank, &envelope) {
                                    client.fail_rank(dp_rank, error.to_string());
                                }
                            }
                            Ok(RankCommand::Control(envelope)) => {
                                received = true;
                                if let Err(error) = sockets.send_control(dp_rank, &envelope) {
                                    client.fail_rank(
                                        dp_rank,
                                        format!(
                                            "EngineCore DP rank {dp_rank} control send failed: {error}"
                                        ),
                                    );
                                    tracing::warn!(
                                        "EngineCore DP rank {dp_rank} control send failed: {error}"
                                    );
                                }
                            }
                            Ok(RankCommand::Shutdown) => {
                                let envelope = EngineCoreEnvelope {
                                    wire_version: ENGINE_CORE_WIRE_VERSION,
                                    payload: Some(Payload::Shutdown(())),
                                };
                                if let Err(error) = sockets.send_control(dp_rank, &envelope) {
                                    tracing::warn!(
                                        "EngineCore DP rank {dp_rank} shutdown failed: {error}"
                                    );
                                    return;
                                }
                                let deadline = Instant::now() + client.shutdown_timeout;
                                while Instant::now() < deadline {
                                    match sockets.receive_output_nonblocking(dp_rank) {
                                        Ok(Some(EngineCoreEnvelope {
                                            payload: Some(Payload::Shutdown(_)),
                                            ..
                                        })) => return,
                                        Ok(Some(envelope)) => {
                                            client.handle_envelope(dp_rank, envelope);
                                        }
                                        Ok(None) => thread::sleep(Duration::from_millis(1)),
                                        Err(error) => {
                                            client.rank_health.insert(dp_rank, false);
                                            tracing::warn!(
                                                "EngineCore DP rank {dp_rank} shutdown acknowledgement failed: {error}"
                                            );
                                            return;
                                        }
                                    }
                                }
                                tracing::warn!(
                                    "EngineCore DP rank {dp_rank} did not acknowledge shutdown"
                                );
                                client.rank_health.insert(dp_rank, false);
                                return;
                            }
                            Err(std_mpsc::TryRecvError::Empty) => break,
                            Err(std_mpsc::TryRecvError::Disconnected) => return,
                        }
                    }
                    match sockets.receive_output_nonblocking(dp_rank) {
                        Ok(Some(envelope)) => {
                            received = true;
                            client.handle_envelope(dp_rank, envelope);
                        }
                        Ok(None) => {}
                        Err(error @ EngineCoreTransportError::InvalidProtobuf { .. }) => {
                            client.fail_rank(
                                dp_rank,
                                format!(
                                    "EngineCore DP rank {dp_rank} emitted malformed output: {error}"
                                ),
                            );
                        }
                        Err(error) => client.fail_rank(
                            dp_rank,
                            format!("EngineCore DP rank {dp_rank} output failed: {error}"),
                        ),
                    }
                    for sequence_id in client.expire_rank(dp_rank) {
                        if let Err(error) =
                            sockets.send_control(dp_rank, &abort_envelope(sequence_id))
                        {
                            tracing::warn!(
                                "EngineCore DP rank {dp_rank} timeout abort failed: {error}"
                            );
                        }
                    }
                    if !received {
                        thread::sleep(Duration::from_millis(1));
                    }
                }
            })
            .map_err(|error| {
                EngineError::ConnectionAcquireFailed(format!(
                    "failed to start EngineCore DP rank {dp_rank} worker: {error}"
                ))
            })?;
        client.worker_handles.lock().push(handle);
        Ok(())
    }

    pub fn submit(
        self: &Arc<Self>,
        payload: &GenerationPayload,
        block_size: i32,
    ) -> Result<TokenHandle, EngineError> {
        self.submit_routed(payload, block_size, None, None, None, &[])
    }

    pub fn submit_routed(
        self: &Arc<Self>,
        payload: &GenerationPayload,
        block_size: i32,
        preferred_rank: Option<usize>,
        session_id: Option<&str>,
        parent_session_id: Option<&str>,
        encoded_stop_sequences: &[Vec<u32>],
    ) -> Result<TokenHandle, EngineError> {
        let _admission_guard = self.admission_lock.lock();
        if self.stopped.load(Ordering::Acquire) {
            return Err(EngineError::ConnectionAcquireFailed(
                "EngineCore client is shutting down".to_string(),
            ));
        }
        if payload.sampling.n <= 0 {
            return Err(EngineError::RequestBuildFailed(
                "sampling n must be positive".to_string(),
            ));
        }
        let requested_output_tokens = payload
            .sampling
            .max_new_tokens
            .unwrap_or(DEFAULT_MAX_OUTPUT_TOKENS);
        if requested_output_tokens < 0 {
            return Err(EngineError::RequestBuildFailed(
                "max_tokens must be non-negative".to_string(),
            ));
        }
        let prompt_tokens = i64::try_from(payload.token_ids.len())
            .map_err(|_| EngineError::RequestBuildFailed("prompt is too long".to_string()))?;
        let encoded_dp_rank = preferred_rank
            .map(|rank| {
                i32::try_from(rank).map_err(|_| {
                    EngineError::RequestBuildFailed(format!(
                        "requested EngineCore DP rank {rank} exceeds int32 range"
                    ))
                })
            })
            .transpose()?;
        let total_tokens = prompt_tokens.saturating_add(i64::from(requested_output_tokens));
        if let Some(max_model_len) = self.max_model_len {
            if total_tokens > max_model_len {
                return Err(EngineError::RequestBuildFailed(format!(
                    "this model's maximum context length is {max_model_len} tokens, but the \
                     request contains {prompt_tokens} prompt tokens and requests \
                     {requested_output_tokens} output tokens"
                )));
            }
        }
        if let Some(max_pool_tokens) = self.max_pool_tokens {
            if total_tokens > max_pool_tokens {
                return Err(EngineError::RequestBuildFailed(format!(
                    "the KV cache holds at most {max_pool_tokens} tokens for one request, but the \
                     request needs {prompt_tokens} prompt tokens plus \
                     {requested_output_tokens} output tokens"
                )));
            }
        }
        let sequence_count = i64::from(payload.sampling.n);
        let first_sequence_id = self
            .next_sequence_id
            .fetch_add(sequence_count, Ordering::Relaxed);
        let sequence_ids: Vec<_> =
            (first_sequence_id..first_sequence_id + sequence_count).collect();
        let prompt_tokens = u32::try_from(prompt_tokens)
            .map_err(|_| EngineError::RequestBuildFailed("prompt is too long".to_string()))?;
        let envelope = encode_add_requests_configured(
            payload,
            &sequence_ids,
            block_size,
            encoded_stop_sequences,
            self.num_draft_tokens,
            self.has_per_req_cache,
            encoded_dp_rank,
        )
        .map_err(EngineError::RequestBuildFailed)?;
        let token_cost = u64::from(prompt_tokens);
        let reservation = self.scheduler.lock().select_and_charge(
            token_cost,
            sequence_ids.len() as u64,
            preferred_rank,
            session_id,
            parent_session_id,
        )?;
        let dp_rank = reservation.rank;
        let token_cost = reservation.token_cost;
        let (sender, receiver) = mpsc::unbounded_channel();
        let completion_group = Arc::new(Mutex::new(CompletionGroup {
            expected: sequence_ids.len(),
            completed: BTreeMap::new(),
            sender: sender.clone(),
        }));
        for (sibling_index, &sequence_id) in sequence_ids.iter().enumerate() {
            self.routes.insert(
                sequence_id,
                SequenceRoute {
                    sender: sender.clone(),
                    request_id: payload.request_id.clone(),
                    prompt_tokens,
                    output_tokens: Vec::new(),
                    cached_tokens: 0,
                    dp_rank,
                    token_cost,
                    deadline: Instant::now() + self.request_timeout,
                    sibling_index,
                    completion_group: completion_group.clone(),
                },
            );
        }

        let send_result = self.rank_commands[&dp_rank].try_send(RankCommand::Input(envelope));
        if let Err(error) = send_result {
            for sequence_id in &sequence_ids {
                self.routes.remove(sequence_id);
            }
            self.scheduler.lock().rollback(&reservation);
            let message = match error {
                std_mpsc::TrySendError::Full(_) => {
                    format!("EngineCore DP rank {dp_rank} command queue is full")
                }
                std_mpsc::TrySendError::Disconnected(_) => {
                    format!("EngineCore DP rank {dp_rank} worker stopped")
                }
            };
            return Err(EngineError::ConnectionAcquireFailed(message));
        }

        Ok(TokenHandle::new(EngineCoreTokenSource {
            receiver: UnboundedReceiverStream::new(receiver),
            client: Arc::clone(self),
            sequence_ids,
            completed: false,
        }))
    }

    fn handle_envelope(&self, dp_rank: usize, envelope: EngineCoreEnvelope) {
        match envelope.payload {
            Some(Payload::Stream(chunk)) => {
                self.rank_timeout_strikes.insert(dp_rank, 0);
                for output in chunk.outputs {
                    let Some(output_value) = output.output else {
                        self.fail_route(
                            output.sequence_id,
                            "EngineCore STREAM output is missing RequestOutput".to_string(),
                        );
                        continue;
                    };
                    self.handle_stream_output(output.sequence_id, output_value);
                }
            }
            Some(Payload::AddResponse(response)) => {
                self.rank_timeout_strikes.insert(dp_rank, 0);
                for sequence in response.sequences {
                    self.handle_terminal_sequence(sequence);
                }
            }
            Some(Payload::Metrics(metrics)) => {
                self.latest_metrics.insert(dp_rank, metrics);
            }
            Some(Payload::UtilityResponse(response)) => {
                if response.command != "abort_request" {
                    let mut responses = self.utility_responses.lock();
                    responses
                        .entry(response.command.clone())
                        .or_default()
                        .insert(dp_rank, response);
                }
            }
            Some(Payload::Shutdown(_)) => {
                self.fail_rank(dp_rank, "EngineCore shut down".to_string());
            }
            Some(payload) => {
                tracing::warn!("Ignoring unexpected EngineCore output payload: {payload:?}");
            }
            None => tracing::warn!("Ignoring empty EngineCore output envelope"),
        }
    }

    fn handle_stream_output(&self, sequence_id: i64, output: crate::proto::engine::RequestOutput) {
        let tokens = match convert_tokens(&output.output_tokens) {
            Ok(tokens) => tokens,
            Err(error) => {
                self.fail_route(sequence_id, error);
                return;
            }
        };
        let Some(mut route) = self.routes.get_mut(&sequence_id) else {
            tracing::debug!("Dropping output for unknown EngineCore sequence {sequence_id}");
            return;
        };
        if route.token_cost > 0 {
            self.scheduler
                .lock()
                .release_tokens(route.dp_rank, route.token_cost);
            route.token_cost = 0;
        }
        route.cached_tokens = output.num_cached_tokens.max(0) as u32;
        route.output_tokens.extend_from_slice(&tokens);

        if output.finished {
            let complete =
                complete_chunk(&route, parse_finish_reason(output.finish_reason.as_deref()));
            let sender = route.sender.clone();
            if !tokens.is_empty() {
                let _ = sender.send(Ok(TokenChunk::Partial {
                    token_ids: tokens,
                    logprobs: None,
                }));
            }
            let dp_rank = route.dp_rank;
            let token_cost = route.token_cost;
            drop(route);
            if let Some((_, route)) = self.routes.remove(&sequence_id) {
                self.scheduler.lock().release(dp_rank, token_cost);
                Self::emit_complete(&route, complete);
            }
        } else {
            let _ = route.sender.send(Ok(TokenChunk::Partial {
                token_ids: tokens,
                logprobs: None,
            }));
        }
    }

    fn handle_terminal_sequence(&self, sequence: Sequence) {
        let Some((_, mut route)) = self.routes.remove(&sequence.id) else {
            return;
        };
        let tokens = match convert_tokens(&sequence.output_tokens) {
            Ok(tokens) => tokens,
            Err(error) => {
                self.scheduler
                    .lock()
                    .release(route.dp_rank, route.token_cost);
                let _ = route.sender.send(Err(EngineError::DecodeError(error)));
                return;
            }
        };
        route.output_tokens = tokens;
        let reason = parse_finish_reason(
            (!sequence.leave_reason.is_empty()).then_some(sequence.leave_reason.as_str()),
        );
        self.scheduler
            .lock()
            .release(route.dp_rank, route.token_cost);
        let complete = complete_chunk(&route, reason);
        Self::emit_complete(&route, complete);
    }

    fn emit_complete(route: &SequenceRoute, complete: TokenChunk) {
        let mut group = route.completion_group.lock();
        group.completed.insert(route.sibling_index, complete);
        if group.completed.len() != group.expected {
            return;
        }
        let sender = group.sender.clone();
        let completed = std::mem::take(&mut group.completed);
        drop(group);
        for (_, complete) in completed {
            let _ = sender.send(Ok(complete));
        }
    }

    fn fail_route(&self, sequence_id: i64, message: String) {
        if let Some((_, route)) = self.routes.remove(&sequence_id) {
            self.scheduler
                .lock()
                .release(route.dp_rank, route.token_cost);
            let _ = route.sender.send(Err(EngineError::DecodeError(message)));
        }
    }

    fn fail_rank(&self, dp_rank: usize, message: String) {
        let _admission_guard = self.admission_lock.lock();
        if self.stopped.load(Ordering::Acquire) {
            return;
        }
        let all_ranks = self
            .rank_health
            .iter()
            .map(|entry| *entry.key())
            .collect::<Vec<_>>();
        let failed_ranks = failure_domain_ranks(self.dp_attention_enabled, dp_rank, &all_ranks);
        {
            let mut scheduler = self.scheduler.lock();
            for &rank in &failed_ranks {
                self.rank_health.insert(rank, false);
                scheduler.disable_rank(rank);
            }
        }
        let message = if self.dp_attention_enabled {
            format!(
                "DP-attention collective failed because EngineCore rank {dp_rank} \
                 became unavailable: {message}"
            )
        } else {
            message
        };
        let sequence_ids: Vec<_> = self
            .routes
            .iter()
            .filter(|route| self.dp_attention_enabled || route.dp_rank == dp_rank)
            .map(|route| *route.key())
            .collect();
        for sequence_id in sequence_ids {
            self.fail_route(sequence_id, message.clone());
        }
        if self.dp_attention_enabled
            && !self
                .collective_shutdown_started
                .swap(true, Ordering::AcqRel)
        {
            for (&rank, sender) in &self.rank_commands {
                match sender.try_send(RankCommand::Shutdown) {
                    Ok(()) => {}
                    Err(std_mpsc::TrySendError::Full(command)) => {
                        let sender = sender.clone();
                        if let Err(error) = thread::Builder::new()
                            .name(format!("EngineCore-DP{rank}-ShutdownDispatch"))
                            .spawn(move || {
                                if let Err(error) = sender.send(command) {
                                    tracing::warn!(
                                        "Failed to queue DP-attention collective shutdown \
                                         for EngineCore rank {rank}: {error}"
                                    );
                                }
                            })
                        {
                            tracing::warn!(
                                "Failed to start DP-attention shutdown dispatcher for \
                                 EngineCore rank {rank}: {error}"
                            );
                        }
                    }
                    Err(std_mpsc::TrySendError::Disconnected(_)) => {
                        tracing::warn!(
                            "EngineCore rank {rank} worker stopped before DP-attention \
                             collective shutdown"
                        );
                    }
                }
            }
        }
    }

    fn expire_rank(&self, dp_rank: usize) -> Vec<i64> {
        let now = Instant::now();
        let expired: Vec<_> = self
            .routes
            .iter()
            .filter(|route| route.dp_rank == dp_rank && route.deadline <= now)
            .map(|route| *route.key())
            .collect();
        for &sequence_id in &expired {
            self.fail_route(sequence_id, "EngineCore request timed out".to_string());
        }
        if !expired.is_empty() {
            let strike_limit_reached = {
                let strikes = self
                    .rank_timeout_strikes
                    .entry(dp_rank)
                    .and_modify(|strikes| *strikes = strikes.saturating_add(1))
                    .or_insert(1);
                *strikes >= 3
            };
            if strike_limit_reached {
                self.fail_rank(
                    dp_rank,
                    "EngineCore request timeout strike limit reached".to_string(),
                );
            }
        }
        expired
    }

    pub fn broadcast_utility(
        &self,
        command: impl Into<String>,
        arguments: Option<Struct>,
    ) -> Result<(), EngineError> {
        let command = command.into();
        self.ensure_utility_command_usable(&command)?;
        self.ensure_all_ranks_healthy()?;
        let envelope = EngineCoreEnvelope {
            wire_version: ENGINE_CORE_WIRE_VERSION,
            payload: Some(Payload::UtilityCommand(UtilityCommand {
                command: command.clone(),
                arguments,
            })),
        };
        let mut failures = Vec::new();
        for (&dp_rank, sender) in &self.rank_commands {
            if let Err(error) = sender.try_send(RankCommand::Control(envelope.clone())) {
                failures.push(format!("rank {dp_rank}: {error}"));
            }
        }
        if !failures.is_empty() {
            self.poison_utility_command(&command);
            return Err(EngineError::ConnectionAcquireFailed(format!(
                "EngineCore utility send failed for {}",
                failures.join(", ")
            )));
        }
        Ok(())
    }

    pub fn send_control_frame(&self, dp_rank: usize, frame: &[u8]) -> Result<String, EngineError> {
        let envelope = decode_control_frame(frame)?;
        let command = control_command(&envelope)?.to_string();
        self.ensure_utility_command_usable(&command)?;
        self.ensure_rank_healthy(dp_rank)?;
        self.take_utility_responses(&command);
        let sender = self.rank_commands.get(&dp_rank).ok_or_else(|| {
            EngineError::RequestBuildFailed(format!(
                "requested EngineCore DP rank {dp_rank} does not exist"
            ))
        })?;
        sender
            .try_send(RankCommand::Control(envelope))
            .map_err(|error| {
                EngineError::ConnectionAcquireFailed(format!(
                    "EngineCore DP rank {dp_rank} control queue failed: {error}"
                ))
            })?;
        Ok(command)
    }

    pub fn broadcast_control_frame(&self, frame: &[u8]) -> Result<String, EngineError> {
        let envelope = decode_control_frame(frame)?;
        let command = control_command(&envelope)?.to_string();
        self.ensure_utility_command_usable(&command)?;
        self.take_utility_responses(&command);
        let mut failures = Vec::new();
        let mut sent = 0;
        for (&dp_rank, sender) in &self.rank_commands {
            if !self
                .rank_health
                .get(&dp_rank)
                .is_some_and(|healthy| *healthy)
            {
                continue;
            }
            if let Err(error) = sender.try_send(RankCommand::Control(envelope.clone())) {
                failures.push(format!("rank {dp_rank}: {error}"));
            } else {
                sent += 1;
            }
        }
        if sent == 0 && failures.is_empty() {
            return Err(EngineError::ConnectionAcquireFailed(
                "no healthy EngineCore ranks are available for control command".to_string(),
            ));
        }
        if !failures.is_empty() {
            self.poison_utility_command(&command);
            return Err(EngineError::ConnectionAcquireFailed(format!(
                "EngineCore control send failed for {}",
                failures.join(", ")
            )));
        }
        Ok(command)
    }

    pub fn take_utility_response_frames(&self, command: &str) -> Vec<(usize, Vec<u8>)> {
        self.take_utility_responses(command)
            .into_iter()
            .map(|(rank, response)| {
                let envelope = EngineCoreEnvelope {
                    wire_version: ENGINE_CORE_WIRE_VERSION,
                    payload: Some(Payload::UtilityResponse(response)),
                };
                (rank, envelope.encode_to_vec())
            })
            .collect()
    }

    pub fn execute_control_frame_all_blocking(
        &self,
        frame: &[u8],
        expected_count: Option<usize>,
        timeout: Duration,
    ) -> Result<Vec<(usize, Vec<u8>)>, EngineError> {
        let envelope = decode_control_frame(frame)?;
        let command = control_command(&envelope)?.to_string();
        let expected = expected_count.unwrap_or_else(|| self.total_rank_count());
        if expected != self.total_rank_count() {
            return Err(EngineError::RequestBuildFailed(format!(
                "broadcast utility response count must equal the EngineCore rank count {}; \
                 got {expected}",
                self.total_rank_count()
            )));
        }
        let deadline = Instant::now().checked_add(timeout).ok_or_else(|| {
            EngineError::RequestBuildFailed("utility timeout is too large".into())
        })?;
        let _execution_guard = loop {
            match self.utility_execution_lock.try_lock() {
                Ok(guard) => break guard,
                Err(_) if Instant::now() < deadline => thread::sleep(Duration::from_millis(5)),
                Err(_) => {
                    return Err(EngineError::ConnectionAcquireFailed(format!(
                        "timed out waiting to execute EngineCore utility command {command:?}"
                    )));
                }
            }
        };

        self.ensure_all_ranks_healthy()?;
        self.broadcast_control_frame(frame)?;
        let mut responses = BTreeMap::new();
        loop {
            responses.extend(self.take_utility_responses(&command));
            if responses.len() >= expected {
                return Ok(responses
                    .into_iter()
                    .map(|(rank, response)| {
                        let envelope = EngineCoreEnvelope {
                            wire_version: ENGINE_CORE_WIRE_VERSION,
                            payload: Some(Payload::UtilityResponse(response)),
                        };
                        (rank, envelope.encode_to_vec())
                    })
                    .collect());
            }
            if Instant::now() >= deadline {
                self.poison_utility_command(&command);
                return Err(EngineError::ConnectionAcquireFailed(format!(
                    "timed out waiting for {expected} EngineCore responses to {command:?}; \
                     got ranks {:?}",
                    responses.keys().collect::<Vec<_>>()
                )));
            }
            thread::sleep(Duration::from_millis(5));
        }
    }

    pub fn poison_utility_command(&self, command: &str) {
        self.poisoned_utility_commands
            .lock()
            .insert(command.to_string());
        self.take_utility_responses(command);
    }

    fn ensure_utility_command_usable(&self, command: &str) -> Result<(), EngineError> {
        if self.poisoned_utility_commands.lock().contains(command) {
            return Err(EngineError::ConnectionAcquireFailed(format!(
                "EngineCore utility command {command:?} previously timed out and cannot be safely reused"
            )));
        }
        Ok(())
    }

    fn ensure_rank_healthy(&self, dp_rank: usize) -> Result<(), EngineError> {
        if self.stopped.load(Ordering::Acquire) {
            return Err(EngineError::ConnectionAcquireFailed(
                "EngineCore client is shutting down".to_string(),
            ));
        }
        match self.rank_health.get(&dp_rank) {
            Some(healthy) if *healthy => Ok(()),
            Some(_) => Err(EngineError::ConnectionAcquireFailed(format!(
                "EngineCore DP rank {dp_rank} is unavailable"
            ))),
            None => Err(EngineError::RequestBuildFailed(format!(
                "requested EngineCore DP rank {dp_rank} does not exist"
            ))),
        }
    }

    fn ensure_all_ranks_healthy(&self) -> Result<(), EngineError> {
        if self.stopped.load(Ordering::Acquire) {
            return Err(EngineError::ConnectionAcquireFailed(
                "EngineCore client is shutting down".to_string(),
            ));
        }
        let unavailable = self
            .rank_health
            .iter()
            .filter_map(|entry| (!*entry.value()).then_some(*entry.key()))
            .collect::<Vec<_>>();
        if unavailable.is_empty() {
            Ok(())
        } else {
            Err(EngineError::ConnectionAcquireFailed(format!(
                "EngineCore utility command requires every rank, but ranks \
                 {unavailable:?} are unavailable"
            )))
        }
    }

    pub fn metrics_snapshot(&self) -> BTreeMap<usize, crate::proto::engine::MetricsSnapshot> {
        self.latest_metrics
            .iter()
            .map(|entry| (*entry.key(), entry.value().clone()))
            .collect()
    }

    pub fn metrics_snapshot_json(&self) -> BTreeMap<usize, serde_json::Value> {
        self.latest_metrics
            .iter()
            .map(|entry| {
                let values = entry
                    .value()
                    .values
                    .as_ref()
                    .map(prost_struct_to_json)
                    .unwrap_or_else(|| serde_json::Value::Object(serde_json::Map::new()));
                (*entry.key(), values)
            })
            .collect()
    }

    pub fn healthy_rank_count(&self) -> usize {
        if self.stopped.load(Ordering::Acquire) {
            return 0;
        }
        self.rank_health
            .iter()
            .filter(|entry| *entry.value())
            .count()
    }

    pub fn total_rank_count(&self) -> usize {
        self.rank_health.len()
    }

    pub fn serving_ready(&self) -> bool {
        let healthy = self.healthy_rank_count();
        deployment_is_ready(self.dp_attention_enabled, healthy, self.total_rank_count())
    }

    pub fn mark_rank_failed(&self, dp_rank: usize, message: String) -> Result<(), EngineError> {
        if !self.rank_health.contains_key(&dp_rank) {
            return Err(EngineError::RequestBuildFailed(format!(
                "requested EngineCore DP rank {dp_rank} does not exist"
            )));
        }
        if self.stopped.load(Ordering::Acquire) {
            return Ok(());
        }
        self.fail_rank(dp_rank, message);
        Ok(())
    }

    pub fn load_snapshot(&self) -> BTreeMap<usize, (u64, u64)> {
        let scheduler = self.scheduler.lock();
        scheduler
            .ranks
            .iter()
            .map(|&rank| (rank, (scheduler.requests[&rank], scheduler.tokens[&rank])))
            .collect()
    }

    pub fn take_utility_responses(&self, command: &str) -> Vec<(usize, UtilityResponse)> {
        self.utility_responses
            .lock()
            .remove(command)
            .unwrap_or_default()
            .into_iter()
            .collect()
    }

    pub async fn execute_utility_all(
        &self,
        command: &str,
        arguments: Option<Struct>,
        timeout: Duration,
    ) -> Result<Vec<(usize, UtilityResponse)>, EngineError> {
        let _execution_guard = self.utility_execution_lock.lock().await;
        self.take_utility_responses(command);
        self.broadcast_utility(command, arguments)?;
        let expected = self.rank_commands.len();
        let deadline = tokio::time::Instant::now() + timeout;
        let mut responses = BTreeMap::new();
        loop {
            responses.extend(self.take_utility_responses(command));
            if responses.len() >= expected {
                return Ok(responses.into_iter().collect());
            }
            if tokio::time::Instant::now() >= deadline {
                self.poison_utility_command(command);
                return Err(EngineError::ConnectionAcquireFailed(format!(
                    "timed out waiting for {expected} EngineCore responses to {command:?}; got {}",
                    responses.len()
                )));
            }
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    }

    pub async fn execute_utility_all_json(
        &self,
        command: &str,
        arguments: Option<Struct>,
        timeout: Duration,
    ) -> Result<BTreeMap<usize, serde_json::Value>, EngineError> {
        Ok(self
            .execute_utility_all(command, arguments, timeout)
            .await?
            .into_iter()
            .map(|(rank, response)| {
                let value = response
                    .result
                    .as_ref()
                    .map(prost_value_to_json)
                    .unwrap_or(serde_json::Value::Null);
                (rank, value)
            })
            .collect())
    }

    pub fn shutdown(&self) -> Result<(), EngineError> {
        let admission_guard = self.admission_lock.lock();
        if self.stopped.swap(true, Ordering::AcqRel) {
            return Ok(());
        }
        let sequence_ids: Vec<_> = self.routes.iter().map(|route| *route.key()).collect();
        for sequence_id in sequence_ids {
            self.fail_route(
                sequence_id,
                "EngineCore client is shutting down".to_string(),
            );
        }
        drop(admission_guard);
        let mut failed_ranks = Vec::new();
        for (&dp_rank, sender) in &self.rank_commands {
            if sender.send(RankCommand::Shutdown).is_err() {
                failed_ranks.push(dp_rank);
            }
        }
        let handles = std::mem::take(&mut *self.worker_handles.lock());
        for handle in handles {
            if handle.join().is_err() {
                tracing::warn!("EngineCore rank worker panicked during shutdown");
            }
        }
        if !failed_ranks.is_empty() && !self.collective_shutdown_started.load(Ordering::Acquire) {
            return Err(EngineError::ConnectionAcquireFailed(format!(
                "EngineCore DP rank workers stopped before shutdown: {failed_ranks:?}"
            )));
        }
        Ok(())
    }

    fn abort(&self, sequence_id: i64) {
        let Some((_, route)) = self.routes.remove(&sequence_id) else {
            return;
        };
        self.scheduler
            .lock()
            .release(route.dp_rank, route.token_cost);
        let envelope = abort_envelope(sequence_id);
        if let Err(error) =
            self.rank_commands[&route.dp_rank].try_send(RankCommand::Control(envelope))
        {
            tracing::warn!("Failed to abort EngineCore sequence {sequence_id}: {error}");
        }
    }
}

fn decode_control_frame(frame: &[u8]) -> Result<EngineCoreEnvelope, EngineError> {
    let envelope = EngineCoreEnvelope::decode(frame).map_err(|error| {
        EngineError::RequestBuildFailed(format!("invalid EngineCore control frame: {error}"))
    })?;
    if envelope.wire_version != ENGINE_CORE_WIRE_VERSION {
        return Err(EngineError::RequestBuildFailed(format!(
            "unsupported EngineCore wire version {}; expected {}",
            envelope.wire_version, ENGINE_CORE_WIRE_VERSION
        )));
    }
    control_command(&envelope)?;
    Ok(envelope)
}

fn session_rank_score(session_id: &str, rank: usize) -> u64 {
    let mut hasher = Blake2b::<U8>::new();
    hasher.update(session_id.as_bytes());
    hasher.update(&(rank as u32).to_le_bytes());
    let digest: [u8; 8] = hasher.finalize().into();
    u64::from_le_bytes(digest)
}

fn control_command(envelope: &EngineCoreEnvelope) -> Result<&str, EngineError> {
    match envelope.payload.as_ref() {
        Some(Payload::UtilityCommand(command)) if !command.command.is_empty() => {
            Ok(command.command.as_str())
        }
        Some(Payload::UtilityCommand(_)) => Err(EngineError::RequestBuildFailed(
            "EngineCore utility command must not be empty".to_string(),
        )),
        _ => Err(EngineError::RequestBuildFailed(
            "raw EngineCore control bridge only accepts UtilityCommand payloads".to_string(),
        )),
    }
}

fn abort_envelope(sequence_id: i64) -> EngineCoreEnvelope {
    let arguments = Struct {
        fields: [
            (
                "__atom_ipc_type__".to_string(),
                Value {
                    kind: Some(Kind::StringValue("dict".to_string())),
                },
            ),
            (
                "value".to_string(),
                Value {
                    kind: Some(Kind::ListValue(ListValue {
                        values: vec![Value {
                            kind: Some(Kind::StructValue(Struct {
                                fields: [
                                    (
                                        "key".to_string(),
                                        Value {
                                            kind: Some(Kind::StringValue("req_id".to_string())),
                                        },
                                    ),
                                    (
                                        "value".to_string(),
                                        Value {
                                            kind: Some(Kind::StructValue(Struct {
                                                fields: [
                                                    (
                                                        "__atom_ipc_type__".to_string(),
                                                        Value {
                                                            kind: Some(Kind::StringValue(
                                                                "int".to_string(),
                                                            )),
                                                        },
                                                    ),
                                                    (
                                                        "value".to_string(),
                                                        Value {
                                                            kind: Some(Kind::StringValue(
                                                                sequence_id.to_string(),
                                                            )),
                                                        },
                                                    ),
                                                ]
                                                .into_iter()
                                                .collect(),
                                            })),
                                        },
                                    ),
                                ]
                                .into_iter()
                                .collect(),
                            })),
                        }],
                    })),
                },
            ),
        ]
        .into_iter()
        .collect(),
    };
    EngineCoreEnvelope {
        wire_version: ENGINE_CORE_WIRE_VERSION,
        payload: Some(Payload::UtilityCommand(UtilityCommand {
            command: "abort_request".to_string(),
            arguments: Some(arguments),
        })),
    }
}

impl Drop for EngineCoreClient {
    fn drop(&mut self) {
        self.stopped.store(true, Ordering::Release);
    }
}

struct EngineCoreTokenSource {
    receiver: UnboundedReceiverStream<Result<TokenChunk, EngineError>>,
    client: Arc<EngineCoreClient>,
    sequence_ids: Vec<i64>,
    completed: bool,
}

impl Stream for EngineCoreTokenSource {
    type Item = Result<TokenChunk, EngineError>;

    fn poll_next(mut self: Pin<&mut Self>, context: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        Pin::new(&mut self.receiver).poll_next(context)
    }
}

impl TokenSource for EngineCoreTokenSource {
    fn mark_completed(&mut self) {
        self.completed = true;
    }
}

impl Drop for EngineCoreTokenSource {
    fn drop(&mut self) {
        if !self.completed {
            for &sequence_id in &self.sequence_ids {
                self.client.abort(sequence_id);
            }
        }
    }
}

fn prost_struct_to_json(value: &Struct) -> serde_json::Value {
    serde_json::Value::Object(
        value
            .fields
            .iter()
            .map(|(key, value)| (key.clone(), prost_value_to_json(value)))
            .collect(),
    )
}

fn prost_value_to_json(value: &Value) -> serde_json::Value {
    match value.kind.as_ref() {
        Some(Kind::NullValue(_)) | None => serde_json::Value::Null,
        Some(Kind::NumberValue(number)) => serde_json::Number::from_f64(*number)
            .map(serde_json::Value::Number)
            .unwrap_or(serde_json::Value::Null),
        Some(Kind::StringValue(string)) => serde_json::Value::String(string.clone()),
        Some(Kind::BoolValue(boolean)) => serde_json::Value::Bool(*boolean),
        Some(Kind::StructValue(object)) => prost_struct_to_json(object),
        Some(Kind::ListValue(list)) => {
            serde_json::Value::Array(list.values.iter().map(prost_value_to_json).collect())
        }
    }
}

fn convert_tokens(tokens: &[i32]) -> Result<Vec<u32>, String> {
    tokens
        .iter()
        .map(|&token| {
            u32::try_from(token).map_err(|_| format!("EngineCore returned negative token {token}"))
        })
        .collect()
}

fn complete_chunk(route: &SequenceRoute, finish_reason: FinishReason) -> TokenChunk {
    let completion_tokens = route.output_tokens.len() as u32;
    TokenChunk::Complete {
        token_ids: route.output_tokens.clone(),
        finish_reason,
        matched_stop: None,
        usage: Usage {
            prompt_tokens: route.prompt_tokens,
            completion_tokens,
            total_tokens: route.prompt_tokens.saturating_add(completion_tokens),
        },
        logprobs: None,
        input_logprobs: None,
        meta: WorkerMeta {
            request_id: route.request_id.clone(),
            weight_version: None,
            cached_tokens: route.cached_tokens,
        },
    }
}

fn parse_finish_reason(reason: Option<&str>) -> FinishReason {
    let reason = reason.unwrap_or("stop").to_ascii_lowercase();
    if reason == "stop_sequence" || reason.starts_with("stop_") {
        return FinishReason::Stop;
    }
    match reason.as_str() {
        "stop" | "eos" | "finished" => FinishReason::Stop,
        "length" | "max_tokens" => FinishReason::Length,
        "abort" | "aborted" => FinishReason::Abort,
        "content_filter" => FinishReason::ContentFilter,
        "tool_calls" => FinishReason::ToolCalls,
        other => FinishReason::Other(other.to_string()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dp_attention_uses_collective_failure_domain() {
        assert_eq!(
            failure_domain_ranks(true, 1, &[0, 1, 2, 3]),
            vec![0, 1, 2, 3]
        );
        assert_eq!(failure_domain_ranks(false, 1, &[0, 1, 2, 3]), vec![1]);
    }

    #[test]
    fn dp_attention_readiness_requires_every_rank() {
        assert!(deployment_is_ready(true, 4, 4));
        assert!(!deployment_is_ready(true, 3, 4));
        assert!(deployment_is_ready(false, 3, 4));
        assert!(!deployment_is_ready(false, 0, 4));
    }

    #[test]
    fn round_robin_spreads_and_releases_load() {
        let mut scheduler = DpScheduler::new(vec![0, 1], "round_robin", 256);
        assert_eq!(
            scheduler
                .select_and_charge(10, 1, None, None, None)
                .unwrap()
                .rank,
            0
        );
        assert_eq!(
            scheduler
                .select_and_charge(20, 1, None, None, None)
                .unwrap()
                .rank,
            1
        );
        scheduler.release(0, 10);
        assert_eq!(scheduler.requests[&0], 0);
        assert_eq!(scheduler.tokens[&0], 0);
    }

    #[test]
    fn least_tokens_and_session_affinity_are_stable() {
        let mut scheduler = DpScheduler::new(vec![0, 1], "least_tokens", 256);
        assert_eq!(
            scheduler
                .select_and_charge(100, 1, Some(1), Some("session-a"), None)
                .unwrap()
                .rank,
            1
        );
        scheduler.release(1, 100);
        scheduler
            .select_and_charge(10_000, 1, Some(1), None, None)
            .unwrap();
        assert_eq!(
            scheduler
                .select_and_charge(1, 1, None, Some("session-a"), None)
                .unwrap()
                .rank,
            1
        );
        assert_eq!(
            scheduler
                .select_and_charge(1, 1, None, None, None)
                .unwrap()
                .rank,
            0
        );
    }

    #[test]
    fn rejects_unknown_explicit_rank() {
        let mut scheduler = DpScheduler::new(vec![0], "least_requests", 256);
        assert!(matches!(
            scheduler.select_and_charge(1, 1, Some(4), None, None),
            Err(EngineError::RequestBuildFailed(_))
        ));
    }

    #[test]
    fn session_charges_only_prompt_growth() {
        let mut scheduler = DpScheduler::new(vec![0, 1], "least_tokens", 256);
        let first = scheduler
            .select_and_charge(100, 1, None, Some("session-a"), Some("parent"))
            .unwrap();
        assert_eq!(first.token_cost, 100);
        scheduler.release(first.rank, first.token_cost);

        let second = scheduler
            .select_and_charge(140, 1, None, Some("session-a"), Some("parent"))
            .unwrap();
        assert_eq!(second.rank, first.rank);
        assert_eq!(second.token_cost, 40);
        assert_eq!(scheduler.tokens[&second.rank], 40);
    }

    #[test]
    fn new_session_uses_python_rendezvous_tie_break() {
        assert_eq!(
            session_rank_score("session-a", 0),
            11_165_587_363_488_860_042
        );
        assert_eq!(
            session_rank_score("session-a", 1),
            6_104_279_702_633_409_475
        );
        let mut scheduler = DpScheduler::new(vec![0, 1], "round_robin", 256);
        assert_eq!(
            scheduler
                .select_and_charge(1, 1, None, Some("session-a"), None)
                .unwrap()
                .rank,
            0
        );
    }

    #[test]
    fn rollback_restores_session_and_load_state() {
        let mut scheduler = DpScheduler::new(vec![0], "least_tokens", 256);
        let reservation = scheduler
            .select_and_charge(100, 2, None, Some("session-a"), None)
            .unwrap();
        scheduler.rollback(&reservation);
        assert_eq!(scheduler.requests[&0], 0);
        assert_eq!(scheduler.tokens[&0], 0);
        assert!(!scheduler.session_owners.contains_key("session-a"));
        assert!(!scheduler.session_prompt_tokens.contains_key("session-a"));
    }

    #[test]
    fn raw_control_bridge_rejects_non_utility_payloads() {
        let shutdown = EngineCoreEnvelope {
            wire_version: ENGINE_CORE_WIRE_VERSION,
            payload: Some(Payload::Shutdown(())),
        }
        .encode_to_vec();
        assert!(matches!(
            decode_control_frame(&shutdown),
            Err(EngineError::RequestBuildFailed(_))
        ));
    }
}
