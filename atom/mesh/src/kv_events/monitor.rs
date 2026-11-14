//! Follows the KV event stream of every registered worker rank, one task each on a thread
//! of the monitor's own, so that decoding never holds up the request path. A task's SUB
//! socket connects once the engine binds, and reconnects after the engine restarts.
//! What each stream carried is what `/kv_events/sources` shows.

use std::{
    collections::BTreeMap,
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
use zeromq::{Socket, SocketEvent, SocketOptions, SocketRecv, SubSocket};

use super::{
    model::{DropReason, KvEvent},
    rank_state::{Admit, RankState, Trust},
    source::KvEventSource,
    wire::{self, DecodedBatch},
};

/// The subscriptions by worker URL.
pub struct KvEventMonitor {
    subscriptions: DashMap<String, Subscription>,
    threads: Handle,
    runtime: Option<Runtime>,
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
    task: JoinHandle<()>,
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
    /// Why the subscription stopped, if it did.
    pub error: Option<String>,
    pub trust: Trust,
    pub last_seq: Option<u64>,
    /// Seconds from the newest batch's `ts` to its arrival.
    pub lag_secs: Option<f64>,
    pub batches: u64,
    pub stored_blocks: u64,
    pub removed_blocks: u64,
    pub clears: u64,
    /// Holder clears after lost batches or an engine restart.
    pub resets: u64,
    pub dropped_events: BTreeMap<DropReason, u64>,
    /// Payloads that are not a batch of this rank.
    pub dropped_batches: u64,
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
            threads: runtime.handle().clone(),
            runtime: Some(runtime),
        })
    }

    /// Follows `source` as `worker`'s stream, in place of any earlier one.
    pub fn attach(&self, worker: &str, model_id: &str, dp_rank: u32, source: KvEventSource) {
        info!(worker, dp_rank, endpoint = %source.endpoint, "Following KV events");
        let status = Arc::new(Mutex::new(SourceStatus {
            worker: worker.into(),
            model_id: model_id.into(),
            dp_rank,
            source: source.clone(),
            ..Default::default()
        }));
        let task = self.threads.spawn(follow(source, dp_rank, status.clone()));
        let subscription = Subscription { status, task };
        self.subscriptions.insert(worker.into(), subscription);
    }

    /// Stops following `worker`'s stream, if any.
    pub fn detach(&self, worker: &str) {
        self.subscriptions.remove(worker);
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

async fn follow(source: KvEventSource, dp_rank: u32, status: Arc<Mutex<SourceStatus>>) {
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
    let mut admission = RankState::default();
    loop {
        tokio::select! {
            // A lost connection counts before the batches of the next one.
            biased;
            Some(event) = events.next() => match event {
                SocketEvent::Connected(..) => status.lock().connected = true,
                SocketEvent::Disconnected(_) => {
                    status.lock().connected = false;
                    admission.disconnected();
                }
                _ => {}
            },
            message = socket.recv() => match message {
                Ok(message) => {
                    let batch = batch(&message.into_vec(), source.block_size, dp_rank);
                    record(&mut status.lock(), &mut admission, batch);
                }
                // The socket reconnects by itself.
                Err(_) => tokio::time::sleep(Duration::from_millis(100)).await,
            },
        }
    }
}

/// A message's seq and batch: `[topic, seq as 8 big-endian bytes, payload]`.
fn batch(frames: &[Bytes], block_size: u32, dp_rank: u32) -> Result<(u64, DecodedBatch), String> {
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

fn record(
    status: &mut SourceStatus,
    admission: &mut RankState,
    batch: Result<(u64, DecodedBatch), String>,
) {
    let (seq, batch) = match batch {
        Ok(batch) => batch,
        Err(e) => {
            status.dropped_batches += 1;
            if status.dropped_batches == 1 {
                warn!(worker = %status.worker, error = %e, "Dropping KV event batches");
            }
            return;
        }
    };
    let clears = batch.events.contains(&KvEvent::Cleared);
    if admission.admit(seq, clears) == Admit::ClearAndApply {
        info!(worker = %status.worker, seq, last_seq = ?status.last_seq, "KV event stream restarted or lost batches");
        status.resets += 1;
    }
    status.trust = admission.trust();
    status.last_seq = Some(seq);
    let now = SystemTime::now().duration_since(UNIX_EPOCH).ok();
    status.lag_secs = now.map(|now| now.as_secs_f64() - batch.ts);
    status.batches += 1;
    for event in &batch.events {
        match event {
            KvEvent::Stored { hashes } => status.stored_blocks += hashes.len() as u64,
            KvEvent::Removed { hashes } => status.removed_blocks += hashes.len() as u64,
            KvEvent::Cleared => status.clears += 1,
        }
    }
    for reason in batch.dropped {
        *status.dropped_events.entry(reason).or_default() += 1;
    }
}

#[cfg(test)]
mod tests {
    use bytes::Bytes;
    use zeromq::{PubSocket, SocketSend, ZmqMessage};

    use super::*;
    use crate::kv_events::wire::tests::encode;

    const WORKER: &str = "http://127.0.0.1:8000@1";
    const CLEAR: &str = r#"[{"type": "AllBlocksCleared"}]"#;

    async fn publish(socket: &mut PubSocket, topic: &str, seq: u64, events: &str, rank: u32) {
        let mut message = ZmqMessage::from(Bytes::from(topic.to_string()));
        message.push_back(Bytes::copy_from_slice(&seq.to_be_bytes()));
        message.push_back(encode(&format!("[1.0, {events}, {rank}]")).into());
        socket.send(message).await.unwrap();
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

    #[tokio::test]
    async fn a_rank_stream_is_followed_from_attach_to_detach() {
        let port = std::net::TcpListener::bind("127.0.0.1:0")
            .unwrap()
            .local_addr()
            .unwrap()
            .port();
        let endpoint = format!("tcp://127.0.0.1:{port}");
        let monitor = KvEventMonitor::start().unwrap();
        let source = KvEventSource {
            endpoint: endpoint.clone(),
            topic: "kv".into(),
            block_size: 4,
        };
        // Registered before its engine binds, as a worker whose HTTP server is up first.
        monitor.attach(WORKER, "model", 1, source);
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
        publish(&mut publisher, "kv", 2, removed, 1).await;
        publish(&mut publisher, "kv", 3, stored, 7).await;
        publish(&mut publisher, "kv", 5, stored, 1).await;
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

        monitor.detach(WORKER);
        assert!(monitor.sources().is_empty());
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
