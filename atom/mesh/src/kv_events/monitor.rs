//! Follows the KV event stream of every registered worker rank, one task each on a thread
//! of the monitor's own, so that decoding never holds up the request path. A task's SUB
//! socket connects once the engine binds, and reconnects after the engine restarts.
//! Batches the stream missed are replayed from the engine, if it serves a replay.
//! Admitted batches go to the writer of the rank's index partition, which has a thread of
//! its own. What each stream carried is what `/kv_events/sources` shows. A registered
//! worker without exact events is also predicted to hold the prompts routed to it.

use std::{
    collections::{BTreeMap, VecDeque},
    fmt,
    sync::Arc,
    time::{Duration, SystemTime, UNIX_EPOCH},
};

use bytes::Bytes;
use dashmap::DashMap;
use futures::StreamExt;
use parking_lot::Mutex;
use serde::Serialize;
use tokio::{
    runtime::{Handle, Runtime},
    task::JoinHandle,
};
use tracing::{info, warn};
use zeromq::{
    DealerSocket, Socket, SocketEvent, SocketOptions, SocketRecv, SocketSend, SubSocket, ZmqMessage,
};

use super::{
    model::{DropReason, KvEvent},
    rank_state::{Admit, RankState, Trust},
    source::KvEventSource,
    wire::{self, DecodedBatch},
};
use crate::{
    core::{worker::PrefillCharge, Worker},
    kv_index::{match_lanes, Holder, Lane, Partition, Predictor, PrefixMatchProvider},
    observability::metrics::schema::names,
};

/// How long a routed prompt is predicted to stay cached: the TTL Dynamo gives the blocks
/// it routes when it has no KV events.
const PREDICTION_TTL: Duration = Duration::from_secs(120);
/// Tokens per predicted block of a worker that reports no KV events, ATOM's default.
const PREDICTED_BLOCK_SIZE: u32 = 16;
/// How long a replay may take, from its request to its end.
const REPLAY_DEADLINE: Duration = Duration::from_secs(5);
/// Live batches held while the ones before them are replayed; more abandon the replay.
const HELD_BATCHES: usize = 1024;
/// The seq of ATOM's last replay message: `[b"", b"", REPLAY_DONE, window]`.
const REPLAY_DONE: [u8; 8] = [0xff; 8];

/// A batch and its seq.
type Batch = (u64, DecodedBatch);
/// What a socket reports of its connections.
type SocketEvents = futures::channel::mpsc::Receiver<SocketEvent>;

/// The subscriptions and prediction lanes by worker URL, and the index partitions by
/// model, pool and block size.
pub struct KvEventMonitor {
    subscriptions: DashMap<String, Subscription>,
    /// Every worker from its registration to its detach; none until it is routed to, or
    /// while its partition has no lane left.
    predictors: DashMap<String, Option<Predictor>>,
    partitions: DashMap<(String, &'static str, u32), Arc<Partition>>,
    threads: Handle,
    runtime: Option<Runtime>,
}

impl fmt::Debug for KvEventMonitor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("KvEventMonitor")
            .field("subscriptions", &self.subscriptions.len())
            .finish_non_exhaustive()
    }
}

impl Drop for KvEventMonitor {
    fn drop(&mut self) {
        // Dropping a runtime waits for its threads, which async code must not do.
        if let Some(runtime) = self.runtime.take() {
            runtime.shutdown_background();
        }
    }
}

struct Subscription {
    status: Arc<Mutex<SourceStatus>>,
    partition: Arc<Partition>,
    /// The rank's lane, when its partition had one left. The task holds it too, and the
    /// lane is freed once both let go of it.
    holder: Option<Arc<Holder>>,
    task: JoinHandle<()>,
}

impl Subscription {
    /// The rank's lane while its stream is followed, and whether its events are exact.
    fn events(&self) -> Option<(Lane, bool)> {
        // A stream that could not start tells nothing of the rank's cache.
        let holder = self.holder.as_ref().filter(|_| !self.task.is_finished())?;
        Some((holder.lane(), self.status.lock().trust == Trust::Exact))
    }
}

impl Drop for Subscription {
    fn drop(&mut self) {
        self.task.abort();
    }
}

/// A subscription as `/kv_events/sources` shows it.
#[derive(Debug, Clone, Default, Serialize)]
pub struct SourceStatus {
    pub worker: String,
    pub model_id: String,
    pub dp_rank: u32,
    #[serde(flatten)]
    pub source: KvEventSource,
    pub connected: bool,
    /// Why the subscription stopped, if it did: its socket could not connect or subscribe.
    pub error: Option<String>,
    pub trust: Trust,
    pub last_seq: Option<u64>,
    /// Seconds from the newest batch's `ts` to its arrival.
    pub lag_secs: Option<f64>,
    pub batches: u64,
    pub stored_blocks: u64,
    pub removed_blocks: u64,
    pub clears: u64,
    /// Holder clears after lost batches, a lost connection or an engine restart.
    pub resets: u64,
    /// Replays of missed batches, and those that could not get them all.
    pub replays: u64,
    pub failed_replays: u64,
    pub dropped_events: BTreeMap<DropReason, u64>,
    /// Messages dropped whole: not `[topic, seq, payload]`, a seq that is not 8 bytes, a
    /// payload that does not decode, or a batch of another DP rank.
    pub dropped_batches: u64,
    /// The rank's blocks in the router's index; null when its partition had no lane left.
    pub indexed_blocks: Option<u64>,
}

impl KvEventMonitor {
    pub fn start() -> std::io::Result<Self> {
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(1)
            .thread_name("kv-events")
            .enable_all()
            .build()?;
        Ok(Self {
            subscriptions: DashMap::new(),
            predictors: DashMap::new(),
            partitions: DashMap::new(),
            threads: runtime.handle().clone(),
            runtime: Some(runtime),
        })
    }

    /// Follows `source` as `worker`'s stream, in place of any earlier one, and indexes it
    /// with the other ranks of the model in the worker's pool.
    pub fn attach(
        &self,
        worker: &str,
        model_id: &str,
        pool: &'static str,
        dp_rank: u32,
        source: KvEventSource,
    ) {
        info!(worker, dp_rank, endpoint = %source.endpoint, "Following KV events");
        let status = Arc::new(Mutex::new(SourceStatus {
            worker: worker.into(),
            model_id: model_id.into(),
            dp_rank,
            source: source.clone(),
            ..Default::default()
        }));
        let partition = self.partition(model_id, pool, source.block_size);
        let holder = partition.hold(status.clone()).map(Arc::new);
        match &holder {
            Some(_) => status.lock().indexed_blocks = Some(0),
            None => warn!(
                worker,
                model_id,
                pool,
                "Not indexing the rank's KV events: the 64 lanes of its model and pool are taken"
            ),
        }
        let task = self
            .threads
            .spawn(follow(source, dp_rank, status.clone(), holder.clone()));
        let subscription = Subscription {
            status,
            partition,
            holder,
            task,
        };
        self.subscriptions.insert(worker.into(), subscription);
        self.register(worker);
    }

    /// Predicts the prompts routed to `worker` until it is detached.
    pub fn register(&self, worker: &str) {
        self.predictors.entry(worker.into()).or_default();
    }

    /// Stops following `worker`'s stream, if any, and predicting its cache.
    pub fn detach(&self, worker: &str) {
        self.predictors.remove(worker);
        self.subscriptions.remove(worker);
    }

    /// Releases `charge` when `worker`'s followed rank stores the last full block of
    /// `tokens`. Nothing waits without a followed rank or a full block.
    pub fn watch_prefill(&self, worker: &str, tokens: &[u32], charge: &Arc<PrefillCharge>) {
        // Under the entry's lock, so the rank keeps its lane until the charge waits on it, and
        // the lane's release drops the charge with the rank's other watches.
        let Some(subscription) = self.subscriptions.get(worker) else {
            return;
        };
        if let Some(((partition, lane), _)) = subscription.events() {
            partition.watch_prefill(lane, tokens, charge);
        }
    }

    fn partition(&self, model_id: &str, pool: &'static str, block_size: u32) -> Arc<Partition> {
        self.partitions
            .entry((model_id.into(), pool, block_size))
            .or_insert_with(|| Partition::start(block_size, PREDICTION_TTL, &self.threads))
            .clone()
    }

    pub fn sources(&self) -> Vec<SourceStatus> {
        let mut sources: Vec<_> = self
            .subscriptions
            .iter()
            .map(|subscription| subscription.status.lock().clone())
            .collect();
        sources.sort_by(|a, b| a.worker.cmp(&b.worker));
        sources
    }
}

impl PrefixMatchProvider for KvEventMonitor {
    fn match_workers(&self, tokens: &[u32], workers: &[Arc<dyn Worker>]) -> Vec<Option<u32>> {
        let (events, predicted): (Vec<_>, Vec<_>) = workers
            .iter()
            .map(|worker| {
                let events = self
                    .subscriptions
                    .get(worker.url())
                    .and_then(|s| s.events());
                let predicted = match events {
                    Some((_, true)) => None,
                    _ => self
                        .predictors
                        .get(worker.url())
                        .and_then(|p| p.as_ref().map(Predictor::lane)),
                };
                (events.map(|(lane, _)| lane), predicted)
            })
            .unzip();
        match_lanes(tokens, &events, &predicted)
    }

    fn on_routed(&self, tokens: &[u32], worker: &dyn Worker) {
        let url = worker.url();
        let subscribed = match self.subscriptions.get(url) {
            Some(s) if s.events().is_some_and(|(_, exact)| exact) => return,
            subscription => subscription.map(|s| s.partition.clone()),
        };
        // A followed rank's predictions are on its partition.
        let placed = |p: &Predictor| subscribed.as_ref().is_none_or(|s| p.is_in(s));
        match self.predictors.get(url).as_deref() {
            // Not registered, or detached.
            None => return,
            Some(Some(predictor)) if placed(predictor) => {
                predict(predictor, tokens, worker);
                return;
            }
            Some(_) => {}
        }
        let partition = subscribed.unwrap_or_else(|| {
            let pool = worker.worker_type().as_metric_label();
            self.partition(worker.model_id(), pool, PREDICTED_BLOCK_SIZE)
        });
        // Under the entry's lock, so that concurrent routings make one predictor, and a
        // worker detached since makes none.
        let Some(mut entry) = self.predictors.get_mut(url) else {
            return;
        };
        if !entry.as_ref().is_some_and(|p| p.is_in(&partition)) {
            *entry = partition.predict(url);
        }
        if let Some(predictor) = entry.as_ref() {
            predict(predictor, tokens, worker);
        }
    }
}

/// Predicts that `worker` holds `tokens`; a prompt dropped as the writer is behind is
/// counted, and warned of once per partition.
fn predict(predictor: &Predictor, tokens: &[u32], worker: &dyn Worker) {
    if predictor.record(tokens) {
        return;
    }
    let pool = worker.worker_type().as_metric_label();
    metrics::counter!(
        names::KV_EVENTS_DROPPED_PREDICTIONS_TOTAL,
        "model" => worker.model_id().to_string(),
        "pool" => pool,
    )
    .increment(1);
    if predictor.first_drop() {
        warn!(
            worker = worker.url(),
            model_id = worker.model_id(),
            pool,
            "Dropping routed prompts from the KV index's predictions: its writer is behind"
        );
    }
}

async fn follow(
    source: KvEventSource,
    dp_rank: u32,
    status: Arc<Mutex<SourceStatus>>,
    holder: Option<Arc<Holder>>,
) {
    let mut options = SocketOptions::default();
    options.no_connect_timeout();
    let mut socket = SubSocket::with_options(options);
    let mut events = socket.monitor();
    let subscribed = async {
        socket.connect(&source.endpoint).await?;
        socket.subscribe(&source.topic).await
    };
    if let Err(e) = subscribed.await {
        warn!(endpoint = %source.endpoint, error = %e, "Cannot follow KV events");
        status.lock().error = Some(e.to_string());
        return;
    }
    status.lock().connected = true;
    let replay = source.replay_endpoint.as_deref();
    let mut rank = Rank {
        dp_rank,
        block_size: source.block_size,
        status,
        admission: RankState::default(),
        holder,
    };
    let mut recv_failed = false;
    loop {
        let message = tokio::select! {
            // A lost connection counts before the batches of the next one.
            biased;
            Some(event) = events.next() => {
                if rank.lost_connection(event) {
                    rank.disconnected();
                }
                continue;
            }
            message = socket.recv() => message,
        };
        let message = match message {
            Ok(message) => {
                recv_failed = false;
                message
            }
            // The socket reconnects by itself.
            Err(e) => {
                if !std::mem::replace(&mut recv_failed, true) {
                    warn!(endpoint = %source.endpoint, error = %e, "Cannot receive KV events; retrying");
                }
                tokio::time::sleep(Duration::from_millis(100)).await;
                continue;
            }
        };
        let Some(batch) = rank.decode(&message.into_vec()) else {
            continue;
        };
        let missing = rank.admit(batch, replay.is_some()).await;
        if let (Some(endpoint), Some((from, batch))) = (replay, missing) {
            rank.recover(endpoint, from, batch, &mut socket, &mut events)
                .await;
        }
    }
}

/// A followed rank: what its stream carried, and where its batches go.
struct Rank {
    dp_rank: u32,
    block_size: u32,
    status: Arc<Mutex<SourceStatus>>,
    admission: RankState,
    holder: Option<Arc<Holder>>,
}

impl Rank {
    /// Shows a socket event; true when the connection was lost, and batches with it.
    fn lost_connection(&self, event: SocketEvent) -> bool {
        let connected = match event {
            SocketEvent::Connected(..) => true,
            SocketEvent::Disconnected(_) => false,
            _ => return false,
        };
        let mut status = self.status.lock();
        status.connected = connected;
        if !connected {
            status.trust = Trust::Partial;
        }
        !connected
    }

    /// Starts the rank over at its next batch, the connection lost since the last.
    fn disconnected(&mut self) {
        self.admission.disconnected();
        self.status.lock().trust = self.admission.trust();
    }

    /// The batch in `[topic, seq, payload]`, or None, counted, if there is none.
    fn decode(&self, frames: &[Bytes]) -> Option<Batch> {
        match batch(frames, self.block_size, self.dp_rank) {
            Ok(batch) => Some(batch),
            Err(e) => {
                let mut status = self.status.lock();
                status.dropped_batches += 1;
                if status.dropped_batches == 1 {
                    warn!(worker = %status.worker, error = %e, "Dropping KV event batches");
                }
                None
            }
        }
    }

    /// Applies the batch unless it was already. A batch with batches missing before it
    /// is returned, with the first missing seq, when `replay`; else they are given up.
    async fn admit(&mut self, (seq, batch): Batch, replay: bool) -> Option<(u64, Batch)> {
        let clears = batch.events.contains(&KvEvent::Cleared);
        let reset = match self.admission.admit(seq, batch.ts, clears) {
            Admit::Apply => false,
            Admit::ClearAndApply => true,
            Admit::Duplicate => return None,
            Admit::Missing { from } if replay => return Some((from, (seq, batch))),
            Admit::Missing { .. } => self.admission.lose(seq, batch.ts) == Admit::ClearAndApply,
        };
        record(
            &mut self.status.lock(),
            self.admission.trust(),
            seq,
            &batch,
            reset,
        );
        if let Some(holder) = &self.holder {
            holder.write(reset, batch.events).await;
        }
        None
    }

    /// Replays the batches from `from` on, holding the live ones, `first` the earliest,
    /// until the replay ends; then applies those. Batches a failed replay leaves out are
    /// given up.
    async fn recover(
        &mut self,
        endpoint: &str,
        from: u64,
        first: Batch,
        live: &mut SubSocket,
        events: &mut SocketEvents,
    ) {
        let seq = first.0;
        // None: the connection was lost after the batches before it.
        let mut held = VecDeque::from([Some(first)]);
        let replayed = self
            .replay(endpoint, from, seq, live, events, &mut held)
            .await;
        {
            let mut status = self.status.lock();
            status.replays += 1;
            if let Err(e) = &replayed {
                status.failed_replays += 1;
                warn!(worker = %status.worker, from, error = %e, "Cannot replay KV event batches");
            }
        }
        for batch in held {
            match batch {
                Some(batch) => {
                    self.admit(batch, false).await;
                }
                None => self.disconnected(),
            }
        }
    }

    async fn replay(
        &mut self,
        endpoint: &str,
        from: u64,
        first: u64,
        live: &mut SubSocket,
        events: &mut SocketEvents,
        held: &mut VecDeque<Option<Batch>>,
    ) -> Result<(), String> {
        let deadline = tokio::time::Instant::now() + REPLAY_DEADLINE;
        let late = || format!("not done in {REPLAY_DEADLINE:?}");
        let mut dealer = DealerSocket::new();
        let mut request = ZmqMessage::from(Bytes::new());
        request.push_back(Bytes::copy_from_slice(&from.to_be_bytes()));
        let requested = async {
            dealer.connect(endpoint).await?;
            dealer.send(request).await
        };
        tokio::time::timeout_at(deadline, requested)
            .await
            .map_err(|_| late())?
            .map_err(|e| e.to_string())?;
        let mut window_start = None;
        let mut recv_failed = false;
        loop {
            tokio::select! {
                // The live batches of a new connection follow its loss, and drain before replies.
                biased;
                () = tokio::time::sleep_until(deadline) => return Err(late()),
                Some(event) = events.next() => {
                    if self.lost_connection(event) {
                        held.push_back(None);
                    }
                }
                message = live.recv() => {
                    let message = match message {
                        Ok(message) => {
                            recv_failed = false;
                            message
                        }
                        Err(e) => {
                            if !std::mem::replace(&mut recv_failed, true) {
                                let status = self.status.lock();
                                warn!(worker = %status.worker, error = %e, "Cannot receive KV events during a replay");
                            }
                            continue;
                        }
                    };
                    let Some(batch) = self.decode(&message.into_vec()) else {
                        continue;
                    };
                    if held.len() == HELD_BATCHES {
                        return Err(format!("{HELD_BATCHES} live batches arrived first"));
                    }
                    held.push_back(Some(batch));
                }
                reply = dealer.recv() => {
                    let frames = reply.map_err(|e| e.to_string())?.into_vec();
                    let frames = frames.get(1..).unwrap_or_default();
                    if frames.get(1).is_some_and(|seq| seq.as_ref() == REPLAY_DONE) {
                        break;
                    }
                    let Some(batch) = self.decode(frames) else { continue };
                    if let Some((_, batch)) = self.admit(batch, true).await {
                        window_start.get_or_insert(batch.0);
                        self.admit(batch, false).await;
                    }
                }
            }
        }
        if let Some(start) = window_start {
            return Err(format!("the engine kept batches from {start} on"));
        }
        match self.admission.next() {
            Some(next) if next > first => Ok(()),
            _ => Err(format!("batch {first} was not replayed")),
        }
    }
}

/// A message's seq and batch: `[topic, seq as 8 big-endian bytes, payload]`.
fn batch(frames: &[Bytes], block_size: u32, dp_rank: u32) -> Result<Batch, String> {
    let [_topic, seq, payload] = frames else {
        return Err(format!(
            "{} frames, not [topic, seq, payload]",
            frames.len()
        ));
    };
    let seq = seq.as_ref().try_into().map_err(|_| "seq is not 8 bytes")?;
    let batch = wire::decode(payload, block_size).map_err(|e| e.to_string())?;
    match batch.dp_rank {
        Some(rank) if rank != dp_rank => Err(format!("a batch of DP rank {rank}")),
        _ => Ok((u64::from_be_bytes(seq), batch)),
    }
}

/// Counts an applied batch; `reset`: the rank's holder was cleared first.
fn record(status: &mut SourceStatus, trust: Trust, seq: u64, batch: &DecodedBatch, reset: bool) {
    if reset {
        info!(worker = %status.worker, seq, last_seq = ?status.last_seq, "KV event stream restarted or lost batches");
        status.resets += 1;
    }
    status.trust = trust;
    status.last_seq = Some(seq);
    let now = SystemTime::now().duration_since(UNIX_EPOCH).ok();
    status.lag_secs = now.map(|now| now.as_secs_f64() - batch.ts);
    status.batches += 1;
    for event in &batch.events {
        match event {
            KvEvent::Stored { blocks, .. } => status.stored_blocks += blocks.len() as u64,
            KvEvent::Removed { hashes } => status.removed_blocks += hashes.len() as u64,
            KvEvent::Cleared => status.clears += 1,
        }
    }
    for &reason in &batch.dropped {
        *status.dropped_events.entry(reason).or_default() += 1;
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use zeromq::{util::PeerIdentity, PubSocket, RouterSocket};

    use super::*;
    use crate::{core::BasicWorkerBuilder, kv_events::wire::tests::encode};

    const WORKER: &str = "http://127.0.0.1:8000@1";
    const CLEAR: &str = r#"[{"type": "AllBlocksCleared"}]"#;

    /// A batch of DP rank `rank` made now.
    fn payload(events: &str, rank: u32) -> Bytes {
        let now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap();
        encode(&format!("[{:?}, {events}, {rank}]", now.as_secs_f64())).into()
    }

    fn message(frames: impl IntoIterator<Item = Bytes>) -> ZmqMessage {
        ZmqMessage::try_from(frames.into_iter().collect::<Vec<_>>()).unwrap()
    }

    fn seq_frame(seq: u64) -> Bytes {
        Bytes::copy_from_slice(&seq.to_be_bytes())
    }

    async fn send(socket: &mut PubSocket, topic: &str, seq: u64, payload: Bytes) {
        let topic = Bytes::from(topic.to_string());
        socket
            .send(message([topic, seq_frame(seq), payload]))
            .await
            .unwrap();
    }

    async fn publish(socket: &mut PubSocket, topic: &str, seq: u64, events: &str, rank: u32) {
        send(socket, topic, seq, payload(events, rank)).await;
    }

    /// The peer of the next replay request, which must be from `from`.
    async fn request(router: &mut RouterSocket, from: u64) -> Bytes {
        let request = router.recv().await.unwrap().into_vec();
        let [peer, _, start] = &request[..] else {
            panic!("replay request {request:?}");
        };
        assert_eq!(start.as_ref(), from.to_be_bytes());
        peer.clone()
    }

    /// Answers the next replay request, which must be from `from`, with `batches`.
    async fn replay(router: &mut RouterSocket, from: u64, batches: &[(u64, Bytes)]) {
        let peer = request(router, from).await;
        answer(router, &peer, batches).await;
    }

    /// Replays `batches` to `peer`, then ends the replay.
    async fn answer(router: &mut RouterSocket, peer: &Bytes, batches: &[(u64, Bytes)]) {
        for (seq, payload) in batches {
            let frames = [
                peer.clone(),
                Bytes::new(),
                "kv".into(),
                seq_frame(*seq),
                payload.clone(),
            ];
            router.send(message(frames)).await.unwrap();
        }
        let done = [
            peer.clone(),
            Bytes::new(),
            Bytes::new(),
            Bytes::from_static(&REPLAY_DONE),
            Bytes::new(),
        ];
        router.send(message(done)).await.unwrap();
    }

    /// An engine's publisher, bound once its earlier one has let go of the port.
    async fn bind(endpoint: &str) -> PubSocket {
        let mut publisher = PubSocket::new();
        for _ in 0..50 {
            if publisher.bind(endpoint).await.is_ok() {
                return publisher;
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
        panic!("cannot bind {endpoint}");
    }

    fn status(monitor: &KvEventMonitor) -> SourceStatus {
        monitor.sources().remove(0)
    }

    async fn status_once(
        monitor: &KvEventMonitor,
        done: impl Fn(&SourceStatus) -> bool,
    ) -> SourceStatus {
        for _ in 0..500 {
            if done(&status(monitor)) {
                return status(monitor);
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
        panic!("timed out at {:?}", status(monitor));
    }

    /// Waits for the writer, which applies a batch after the batch is counted, to leave the
    /// rank with `blocks` indexed, the first `hit` tokens of 1..=10 among them.
    async fn indexed(monitor: &KvEventMonitor, blocks: u64, hit: u32) {
        let worker: Arc<dyn Worker> = Arc::new(BasicWorkerBuilder::new(WORKER).build());
        let tokens: Vec<u32> = (1..=10).collect();
        for _ in 0..500 {
            let matched = monitor.match_workers(&tokens, std::slice::from_ref(&worker));
            if status(monitor).indexed_blocks == Some(blocks) && matched == [Some(hit)] {
                return;
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
        panic!("not indexed at {:?}", status(monitor));
    }

    /// Publishes the startup clear, as every publisher does at seq 0, until `done` holds:
    /// what is published before a subscription reaches the publisher is lost.
    async fn clear_until(
        publisher: &mut PubSocket,
        monitor: &KvEventMonitor,
        done: impl Fn(&SourceStatus) -> bool,
    ) -> SourceStatus {
        for _ in 0..500 {
            publish(publisher, "kv", 0, CLEAR, 1).await;
            tokio::time::sleep(Duration::from_millis(20)).await;
            if done(&status(monitor)) {
                // Lets a clear still in flight arrive.
                tokio::time::sleep(Duration::from_millis(100)).await;
                return status(monitor);
            }
        }
        panic!("no KV events at {:?}", status(monitor));
    }

    fn free_endpoint() -> String {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        format!("tcp://{}", listener.local_addr().unwrap())
    }

    /// Follows rank 1 of `worker`, alone on `monitor`, from a publisher of blocks of 4
    /// tokens, once its stream is joined.
    pub(crate) async fn followed(monitor: &KvEventMonitor, worker: &str) -> PubSocket {
        let endpoint = free_endpoint();
        let source = KvEventSource {
            endpoint: endpoint.clone(),
            topic: "kv".into(),
            block_size: 4,
            ..Default::default()
        };
        monitor.attach(worker, "m", "regular", 1, source);
        let mut publisher = bind(&endpoint).await;
        clear_until(&mut publisher, monitor, |s| s.batches > 0).await;
        publisher
    }

    /// Publishes rank 1's store of the full blocks of `tokens` from the root, each block
    /// hashed as its last token.
    pub(crate) async fn publish_stored(publisher: &mut PubSocket, seq: u64, tokens: &[u32]) {
        let blocks = &tokens[..tokens.len() / 4 * 4];
        let hashes: Vec<u32> = blocks.chunks(4).map(|block| block[3]).collect();
        let stored = format!(
            r#"[{{"type": "BlockStored", "block_hashes": {hashes:?}, "token_ids": {blocks:?}, "block_size": 4, "medium": "GPU"}}]"#
        );
        publish(publisher, "kv", seq, &stored, 1).await;
    }

    /// Waits for the writers to apply the prompts routed, until `workers` match `hits`.
    async fn matched(
        monitor: &KvEventMonitor,
        prompt: &[u32],
        workers: &[Arc<dyn Worker>],
        hits: &[Option<u32>],
    ) {
        for _ in 0..500 {
            if monitor.match_workers(prompt, workers) == hits {
                return;
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
        assert_eq!(monitor.match_workers(prompt, workers), hits);
    }

    /// Rank 1 of a connected worker, its batches counted but not indexed.
    fn rank() -> Rank {
        let status = SourceStatus {
            worker: WORKER.into(),
            connected: true,
            ..Default::default()
        };
        Rank {
            dp_rank: 1,
            block_size: 4,
            status: Arc::new(Mutex::new(status)),
            admission: RankState::default(),
            holder: None,
        }
    }

    fn decoded(seq: u64, payload: &Bytes) -> Batch {
        batch(&[Bytes::new(), seq_frame(seq), payload.clone()], 4, 1).unwrap()
    }

    #[tokio::test]
    async fn a_rank_stream_is_followed_from_attach_to_detach() {
        let endpoint = free_endpoint();
        let monitor = KvEventMonitor::start().unwrap();
        let source = KvEventSource {
            endpoint: endpoint.clone(),
            topic: "kv".into(),
            block_size: 4,
            ..Default::default()
        };
        // Registered before its engine binds, as a worker whose HTTP server is up first.
        monitor.attach(WORKER, "model", "regular", 1, source);
        tokio::time::sleep(Duration::from_millis(200)).await;
        assert!(!status(&monitor).connected);
        let mut publisher = bind(&endpoint).await;
        let base = clear_until(&mut publisher, &monitor, |s| s.batches > 0).await;
        assert_eq!((base.trust, base.last_seq), (Trust::Exact, Some(0)));

        let stored = r#"[{"type": "BlockStored", "block_hashes": [11, 12], "token_ids": [1, 2, 3, 4, 5, 6, 7, 8], "block_size": 4, "medium": "GPU"}]"#;
        let removed = r#"[{"type": "BlockRemoved", "block_hashes": [11], "medium": "GPU"},
            {"type": "BlockRemoved", "block_hashes": [12], "medium": "CPU"}]"#;
        publish(&mut publisher, "kv", 1, stored, 1).await;
        publish(&mut publisher, "other", 2, stored, 1).await;
        indexed(&monitor, 2, 8).await;
        // Without its first block, the chain matches nothing.
        publish(&mut publisher, "kv", 2, removed, 1).await;
        indexed(&monitor, 1, 0).await;
        // The gap before seq 5 clears the holder before the blocks are stored again.
        publish(&mut publisher, "kv", 3, stored, 7).await;
        publish(&mut publisher, "kv", 5, stored, 1).await;
        indexed(&monitor, 2, 8).await;
        let status = status_once(&monitor, |s| s.last_seq == Some(5)).await;
        assert_eq!(status.batches - base.batches, 3);
        assert_eq!((status.stored_blocks, status.removed_blocks), (4, 1));
        assert_eq!(
            (status.dropped_batches, status.resets - base.resets),
            (1, 1)
        );
        assert_eq!(
            status.dropped_events,
            BTreeMap::from([(DropReason::Tier, 1)])
        );
        assert_eq!(status.trust, Trust::Partial);

        // A restarted engine binds again and starts over at seq 0.
        drop(publisher);
        status_once(&monitor, |s| !s.connected).await;
        let mut publisher = bind(&endpoint).await;
        let restarted = clear_until(&mut publisher, &monitor, |s| s.last_seq == Some(0)).await;
        assert!(restarted.connected && restarted.resets > status.resets);
        assert_eq!(restarted.trust, Trust::Exact);
        // Batches may be lost from the moment the connection is.
        drop(publisher);
        let lost = status_once(&monitor, |s| !s.connected).await;
        assert_eq!(lost.trust, Trust::Partial);

        monitor.detach(WORKER);
        assert!(monitor.sources().is_empty());
    }

    #[tokio::test]
    async fn missed_batches_are_replayed_while_the_engine_keeps_them() {
        let (endpoint, replay_endpoint) = (free_endpoint(), free_endpoint());
        let mut router = RouterSocket::new();
        router.bind(&replay_endpoint).await.unwrap();
        let mut publisher = bind(&endpoint).await;
        let monitor = KvEventMonitor::start().unwrap();
        let source = KvEventSource {
            endpoint,
            topic: "kv".into(),
            block_size: 4,
            replay_endpoint: Some(replay_endpoint),
            ..Default::default()
        };
        monitor.attach(WORKER, "model", "regular", 1, source);
        let base = clear_until(&mut publisher, &monitor, |s| s.batches > 0).await;
        let first = r#"[{"type": "BlockStored", "block_hashes": [11], "token_ids": [1, 2, 3, 4], "block_size": 4, "medium": "GPU"}]"#;
        let second = r#"[{"type": "BlockStored", "block_hashes": [12], "parent_block_hash": 11, "token_ids": [5, 6, 7, 8], "block_size": 4, "medium": "GPU"}]"#;
        publish(&mut publisher, "kv", 1, first, 1).await;
        indexed(&monitor, 1, 4).await;

        // seq 2 never arrives live; seq 3 waits for its replay, then is a duplicate.
        let third = payload("[]", 1);
        send(&mut publisher, "kv", 3, third.clone()).await;
        replay(&mut router, 2, &[(2, payload(second, 1)), (3, third)]).await;
        indexed(&monitor, 2, 8).await;
        let filled = status_once(&monitor, |s| s.replays == 1 && s.last_seq == Some(3)).await;
        assert_eq!(filled.batches - base.batches, 3);
        assert_eq!((filled.failed_replays, filled.resets), (0, base.resets));
        assert_eq!(filled.trust, Trust::Exact);

        // The engine no longer keeps seq 4: the rank starts over from what it kept.
        let sixth = payload("[]", 1);
        send(&mut publisher, "kv", 6, sixth.clone()).await;
        replay(&mut router, 4, &[(5, payload(first, 1)), (6, sixth)]).await;
        indexed(&monitor, 1, 4).await;
        let lost = status_once(&monitor, |s| s.replays == 2 && s.last_seq == Some(6)).await;
        assert_eq!((lost.failed_replays, lost.resets), (1, base.resets + 1));
        assert_eq!(lost.trust, Trust::Partial);
    }

    /// A replay the engine never ends is given up at its deadline, with the batches it was
    /// to fill in.
    #[tokio::test]
    async fn a_replay_that_never_ends_is_given_up_at_its_deadline() {
        let endpoint = free_endpoint();
        let mut router = RouterSocket::new();
        router.bind(&endpoint).await.unwrap();
        let mut rank = rank();
        rank.admit(decoded(0, &payload(CLEAR, 1)), true).await;
        let (from, first) = rank
            .admit(decoded(2, &payload("[]", 1)), true)
            .await
            .unwrap();
        let mut live = SubSocket::new();
        let (_lost, mut events) = futures::channel::mpsc::channel(1);
        let recovered = rank.recover(&endpoint, from, first, &mut live, &mut events);
        let expired = async {
            request(&mut router, 1).await;
            // Paused, the clock skips to the deadline once nothing else is due.
            tokio::time::pause();
        };
        tokio::join!(recovered, expired);
        let status = rank.status.lock().clone();
        assert_eq!(
            (status.replays, status.failed_replays, status.resets),
            (1, 1, 1)
        );
        assert_eq!((status.last_seq, status.trust), (Some(2), Trust::Partial));
    }

    /// A connection lost during a replay leaves the rank partial once the replay is
    /// applied, and its next batch clears its holder.
    #[tokio::test]
    async fn a_connection_lost_during_a_replay_leaves_the_rank_partial() {
        let endpoint = free_endpoint();
        let mut router = RouterSocket::new();
        router.bind(&endpoint).await.unwrap();
        let mut rank = rank();
        rank.admit(decoded(0, &payload(CLEAR, 1)), true).await;
        let second = payload("[]", 1);
        let (from, first) = rank.admit(decoded(2, &second), true).await.unwrap();
        let mut live = SubSocket::new();
        let (mut lost, mut events) = futures::channel::mpsc::channel(1);
        let recovered = rank.recover(&endpoint, from, first, &mut live, &mut events);
        let answered = async {
            let peer = request(&mut router, 1).await;
            let disconnected = SocketEvent::Disconnected(PeerIdentity::new());
            lost.try_send(disconnected).unwrap();
            let batches = [(1, payload("[]", 1)), (2, second.clone())];
            answer(&mut router, &peer, &batches).await;
        };
        tokio::join!(recovered, answered);
        let status = rank.status.lock().clone();
        assert_eq!(
            (status.replays, status.failed_replays, status.resets),
            (1, 0, 0)
        );
        assert_eq!(
            (status.last_seq, status.trust, status.connected),
            (Some(2), Trust::Partial, false)
        );
        rank.admit(decoded(3, &payload("[]", 1)), false).await;
        assert_eq!(rank.status.lock().resets, 1);
    }

    #[tokio::test]
    async fn a_stream_that_cannot_start_keeps_its_lane_and_matches_nothing() {
        const OTHER: &str = "http://127.0.0.1:8001@1";
        let monitor = KvEventMonitor::start().unwrap();
        let source = |endpoint: &str| KvEventSource {
            endpoint: endpoint.into(),
            topic: "kv".into(),
            block_size: 4,
            ..Default::default()
        };
        monitor.attach(WORKER, "model", "regular", 1, source("not an endpoint"));
        status_once(&monitor, |s| s.error.is_some()).await;
        // Time for a lane given up too early to be freed by the writer.
        tokio::time::sleep(Duration::from_millis(100)).await;
        monitor.attach(OTHER, "model", "regular", 1, source("tcp://127.0.0.1:1"));
        let lane = |worker: &str| {
            let subscription = monitor.subscriptions.get(worker).unwrap();
            subscription.holder.as_ref().unwrap().lane().1
        };
        assert_ne!(lane(WORKER), lane(OTHER));
        let workers: Vec<Arc<dyn Worker>> = [WORKER, OTHER]
            .map(|url| Arc::new(BasicWorkerBuilder::new(url).build()) as Arc<dyn Worker>)
            .into();
        assert_eq!(
            monitor.match_workers(&[1, 2, 3, 4], &workers),
            [None, Some(0)]
        );
    }

    /// Exact events answer alone, partial ones are raised to the prompts routed to the
    /// worker, and those alone match a worker without events.
    #[tokio::test]
    async fn routed_prompts_raise_partial_events_and_stand_in_for_missing_ones() {
        const OTHER: &str = "http://127.0.0.1:8001@1";
        const UNFOLLOWED: &str = "http://127.0.0.1:8002";
        let monitor = KvEventMonitor::start().unwrap();
        let source = |endpoint: &str| KvEventSource {
            endpoint: endpoint.into(),
            topic: "kv".into(),
            block_size: 4,
            ..Default::default()
        };
        // One stream cannot start; the other waits for its engine, its trust partial.
        monitor.attach(WORKER, "model", "regular", 1, source("not an endpoint"));
        monitor.attach(OTHER, "model", "regular", 1, source("tcp://127.0.0.1:1"));
        monitor.register(UNFOLLOWED);
        status_once(&monitor, |s| s.error.is_some()).await;
        let workers: Vec<Arc<dyn Worker>> = [WORKER, OTHER, UNFOLLOWED]
            .map(|url| Arc::new(BasicWorkerBuilder::new(url).build()) as Arc<dyn Worker>)
            .into();
        let prompt: Vec<u32> = (1..=40).collect();
        for worker in &workers {
            monitor.on_routed(&prompt, worker.as_ref());
        }
        // Blocks of 4 short of the last token, and of 16 for the worker without events.
        matched(&monitor, &prompt, &workers, &[Some(36), Some(36), Some(32)]).await;
        monitor
            .subscriptions
            .get(OTHER)
            .unwrap()
            .status
            .lock()
            .trust = Trust::Exact;
        let exact = [Some(36), Some(0), Some(32)];
        assert_eq!(monitor.match_workers(&prompt, &workers), exact);

        monitor.detach(UNFOLLOWED);
        let removed = [Some(36), Some(0), None];
        assert_eq!(monitor.match_workers(&prompt, &workers), removed);
    }

    /// A worker is predicted from its registration to its detach: in the partition of its
    /// model and pool until its rank is followed, then in the rank's.
    #[tokio::test]
    async fn predictions_follow_the_worker_from_registration_to_detach() {
        let monitor = KvEventMonitor::start().unwrap();
        let workers: Vec<Arc<dyn Worker>> = vec![Arc::new(BasicWorkerBuilder::new(WORKER).build())];
        let prompt: Vec<u32> = (1..=40).collect();
        let route = || monitor.on_routed(&prompt, workers[0].as_ref());
        route();
        assert!(monitor.predictors.is_empty());
        monitor.register(WORKER);
        route();
        matched(&monitor, &prompt, &workers, &[Some(32)]).await;
        // Activated before it is followed, as the two run in parallel.
        let source = KvEventSource {
            endpoint: "tcp://127.0.0.1:1".into(),
            topic: "kv".into(),
            block_size: 4,
            ..Default::default()
        };
        monitor.attach(WORKER, "model", "regular", 1, source);
        route();
        matched(&monitor, &prompt, &workers, &[Some(36)]).await;
        monitor.detach(WORKER);
        route();
        assert!(monitor.predictors.is_empty());
    }

    /// A prompt the writer has no room for is counted by model and pool.
    #[tokio::test]
    async fn a_prompt_the_writer_has_no_room_for_is_counted() {
        let recorder = metrics_exporter_prometheus::PrometheusBuilder::new().build_recorder();
        let _recorder = metrics::set_default_local_recorder(&recorder);
        let monitor = KvEventMonitor::start().unwrap();
        let worker = BasicWorkerBuilder::new(WORKER).model_id("model").build();
        monitor.register(WORKER);
        // More than the 64 MB of tokens the writer queues.
        monitor.on_routed(&vec![0; (1 << 24) + 1], &worker);
        let dropped = r#"mesh_kv_events_dropped_predictions_total{model="model",pool="regular"} 1"#;
        assert!(recorder.handle().render().contains(dropped));
    }

    #[test]
    fn a_message_is_a_batch_of_the_rank_or_dropped() {
        let payload = Bytes::from(encode(&format!("[1.0, {CLEAR}, 1]")));
        let seq = Bytes::copy_from_slice(&7u64.to_be_bytes());
        let frames = |frames: &[&Bytes]| frames.iter().map(|&f| f.clone()).collect::<Vec<_>>();
        let topic = Bytes::new();
        let (seq_no, decoded) = batch(&frames(&[&topic, &seq, &payload]), 4, 1).unwrap();
        assert_eq!((seq_no, decoded.events), (7, vec![KvEvent::Cleared]));
        assert!(batch(&frames(&[&topic, &seq, &payload]), 4, 2).is_err());
        assert!(batch(&frames(&[&seq, &payload]), 4, 1).is_err());
        assert!(batch(&frames(&[&topic, &payload, &payload]), 4, 1).is_err());
        assert!(batch(&frames(&[&topic, &seq, &seq]), 4, 1).is_err());
    }
}
