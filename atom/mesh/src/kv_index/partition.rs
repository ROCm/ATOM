//! A partition of the index: the ranks of one model, pool and block size, each on a lane,
//! and the writer task that alone applies their admitted batches, in order. The prompts
//! routed to each worker are predictions on lanes of a second index, kept apart from the
//! events, each block until a TTL after it was last routed.

use std::{
    collections::HashMap,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
    time::Duration,
};

use parking_lot::Mutex;
use tokio::{
    runtime::Handle,
    sync::mpsc::{self, error::TrySendError},
    time::Instant,
};
use tracing::warn;

use super::{
    hash,
    index::{HolderTable, Op, PrefixIndex},
    HolderId, PrefixHash,
};
use crate::kv_events::{model::KvEvent, SourceStatus};

/// Admitted batches a partition's writer may fall behind by before its ranks wait.
const QUEUED_BATCHES: usize = 1024;
/// Routed prompts the writer may fall behind by before more are dropped, each its prefix
/// hashes, 8 bytes a block.
const QUEUED_PREDICTIONS: usize = 256;
/// Blocks one worker's predictions may hold, as many as a rank's events.
const MAX_PREDICTED_BLOCKS: usize = 1 << 22;

pub(crate) struct Partition {
    block_size: u32,
    pub(super) index: PrefixIndex,
    /// Lanes taken, by bit.
    lanes: Mutex<u64>,
    writes: mpsc::Sender<Write>,
    /// Where the writer runs, and a release waits for room in its queue.
    runtime: Handle,
    pub(super) predicted: PrefixIndex,
    predicting: Mutex<u64>,
    /// Set once a worker found no prediction lane left.
    crowded: AtomicBool,
    /// Apart from `writes`, so that routing never delays a rank's batches.
    predictions: mpsc::Sender<Write>,
}

enum Write {
    /// A rank's admitted batch; `reset` clears what the rank held first.
    Batch {
        holder: HolderId,
        reset: bool,
        events: Vec<KvEvent>,
        status: Arc<Mutex<SourceStatus>>,
    },
    /// The rank is no longer followed: its blocks leave the index and its lane is free.
    Release(HolderId),
    /// The prefix hashes of a prompt routed to the worker on the prediction lane.
    Predict(HolderId, Vec<PrefixHash>),
    /// The worker's predictions leave the index and its lane is free.
    Unpredict(HolderId),
}

impl Partition {
    /// A partition whose writer runs on `runtime` and keeps predictions for `ttl`.
    pub(crate) fn start(block_size: u32, ttl: Duration, runtime: &Handle) -> Arc<Self> {
        let (writes, batches) = mpsc::channel(QUEUED_BATCHES);
        let (predictions, routed) = mpsc::channel(QUEUED_PREDICTIONS);
        let partition = Arc::new(Self {
            block_size,
            index: PrefixIndex::default(),
            lanes: Mutex::new(0),
            writes,
            runtime: runtime.clone(),
            predicted: PrefixIndex::default(),
            predicting: Mutex::new(0),
            crowded: AtomicBool::new(false),
            predictions,
        });
        runtime.spawn(write(partition.clone(), batches, routed, ttl));
        partition
    }

    /// A lane for a rank whose batches `status` counts; None once all 64 are taken.
    pub(crate) fn hold(self: &Arc<Self>, status: Arc<Mutex<SourceStatus>>) -> Option<Holder> {
        take(&self.lanes).map(|id| Holder {
            id,
            partition: self.clone(),
            status,
        })
    }

    /// A prediction lane for `worker`; None once all 64 are taken, which warns once.
    pub(crate) fn predict(self: &Arc<Self>, worker: &str) -> Option<Predictor> {
        let predictor = take(&self.predicting).map(|id| Predictor {
            id,
            partition: self.clone(),
        });
        if predictor.is_none() && !self.crowded.swap(true, Ordering::Relaxed) {
            warn!(
                worker,
                "Not predicting the worker's cache: the 64 prediction lanes of its model and pool are taken"
            );
        }
        predictor
    }
}

fn take(lanes: &Mutex<u64>) -> Option<HolderId> {
    let mut lanes = lanes.lock();
    let free = lanes.trailing_ones();
    (free < 64).then(|| {
        *lanes |= 1 << free;
        HolderId(free as u8)
    })
}

/// A rank's lane, held while the rank is followed.
pub(crate) struct Holder {
    id: HolderId,
    partition: Arc<Partition>,
    status: Arc<Mutex<SourceStatus>>,
}

impl Holder {
    pub(crate) fn lane(&self) -> (Arc<Partition>, HolderId) {
        (self.partition.clone(), self.id)
    }

    /// Queues an admitted batch, waiting while the writer is behind.
    pub(crate) async fn write(&self, reset: bool, events: Vec<KvEvent>) {
        let batch = Write::Batch {
            holder: self.id,
            reset,
            events,
            status: self.status.clone(),
        };
        // The writer runs as long as the partition does.
        let _ = self.partition.writes.send(batch).await;
    }
}

/// The release follows the rank's last batch, and the lane stays taken until the writer
/// has removed the rank's blocks, so a rank given the lane next starts empty.
impl Drop for Holder {
    fn drop(&mut self) {
        let partition = &self.partition;
        release(
            &partition.runtime,
            &partition.writes,
            Write::Release(self.id),
        );
    }
}

/// A worker's prediction lane, held while it may be routed to.
pub(crate) struct Predictor {
    id: HolderId,
    partition: Arc<Partition>,
}

impl Predictor {
    pub(crate) fn lane(&self) -> (Arc<Partition>, HolderId) {
        (self.partition.clone(), self.id)
    }

    /// Predicts the full blocks of a prompt routed to the worker; dropped while the writer
    /// is behind, so that routing never waits for it.
    pub(crate) fn record(&self, tokens: &[u32]) {
        let prefix = hash::prefix_hashes(tokens, self.partition.block_size);
        let _ = self
            .partition
            .predictions
            .try_send(Write::Predict(self.id, prefix));
    }
}

/// Like a holder's, the lane stays taken until its predictions are gone.
impl Drop for Predictor {
    fn drop(&mut self) {
        let partition = &self.partition;
        release(
            &partition.runtime,
            &partition.predictions,
            Write::Unpredict(self.id),
        );
    }
}

/// Queues a lane's release, waiting on the writer's runtime while the queue is full.
fn release(runtime: &Handle, writes: &mpsc::Sender<Write>, release: Write) {
    match writes.try_send(release) {
        Err(TrySendError::Full(release)) => {
            let writes = writes.clone();
            runtime.spawn(async move { writes.send(release).await });
        }
        // Closed only once the monitor's runtime, and the index with it, is gone.
        Ok(()) | Err(TrySendError::Closed(_)) => {}
    }
}

/// Applies the ranks' batches before any prediction. Expired predictions leave at the next
/// sweep, due every eighth of the TTL.
async fn write(
    partition: Arc<Partition>,
    mut writes: mpsc::Receiver<Write>,
    mut predictions: mpsc::Receiver<Write>,
    ttl: Duration,
) {
    let mut tables: HashMap<HolderId, HolderTable> = HashMap::new();
    let mut predicted: HashMap<HolderId, HashMap<PrefixHash, Instant>> = HashMap::new();
    let mut sweeps = tokio::time::interval(ttl / 8);
    let mut ops = Vec::new();
    loop {
        let write = tokio::select! {
            biased;
            Some(write) = writes.recv() => write,
            Some(prediction) = predictions.recv() => prediction,
            now = sweeps.tick() => {
                for (&lane, blocks) in &mut predicted {
                    blocks.retain(|&hash, &mut until| {
                        let live = until > now;
                        if !live {
                            ops.push(Op::Remove(hash));
                        }
                        live
                    });
                    partition.predicted.apply(lane, &mut ops);
                }
                continue;
            }
        };
        match write {
            Write::Batch {
                holder,
                reset,
                events,
                status,
            } => {
                let table = tables.entry(holder).or_default();
                if reset {
                    table.clear(&mut ops);
                }
                let dropped: Vec<_> = events
                    .iter()
                    .filter_map(|event| table.apply(event, &mut ops).err())
                    .collect();
                partition.index.apply(holder, &mut ops);
                let mut status = status.lock();
                status.indexed_blocks = Some(table.len() as u64);
                for reason in dropped {
                    *status.dropped_events.entry(reason).or_default() += 1;
                }
            }
            Write::Release(holder) => {
                if let Some(mut table) = tables.remove(&holder) {
                    table.clear(&mut ops);
                    partition.index.apply(holder, &mut ops);
                }
                *partition.lanes.lock() &= !(1 << holder.0);
            }
            Write::Predict(lane, prefix) => {
                let until = Instant::now() + ttl;
                let blocks = predicted.entry(lane).or_default();
                for hash in prefix {
                    if let Some(last) = blocks.get_mut(&hash) {
                        *last = until;
                    } else if blocks.len() < MAX_PREDICTED_BLOCKS {
                        blocks.insert(hash, until);
                        ops.push(Op::Store(hash));
                    }
                }
                partition.predicted.apply(lane, &mut ops);
            }
            Write::Unpredict(lane) => {
                if let Some(blocks) = predicted.remove(&lane) {
                    ops.extend(blocks.into_keys().map(Op::Remove));
                    partition.predicted.apply(lane, &mut ops);
                }
                *partition.predicting.lock() &= !(1 << lane.0);
            }
        }
    }
}

/// A rank's lane in the events of its partition, or a worker's in the predictions.
pub(crate) type Lane = (Arc<Partition>, HolderId);

/// The leading tokens of `tokens` each worker holds, the more of what its rank's events
/// and its predictions hold, in the order of the lanes: one hashing per partition, and one
/// walk of each of its indexes.
pub(crate) fn match_lanes(
    tokens: &[u32],
    events: &[Option<Lane>],
    predicted: &[Option<Lane>],
) -> Vec<Option<u32>> {
    let mut hits = vec![None; events.len().max(predicted.len())];
    let mut walked: Vec<&Arc<Partition>> = Vec::new();
    for (partition, _) in events.iter().chain(predicted).flatten() {
        if walked.iter().any(|walked| Arc::ptr_eq(walked, partition)) {
            continue;
        }
        walked.push(partition);
        // The engine computes the block of the prompt's last token, so it never reports
        // that block as cached.
        let reusable = &tokens[..tokens.len().saturating_sub(1)];
        let prefix = hash::prefix_hashes(reusable, partition.block_size);
        for (lanes, index) in [
            (events, &partition.index),
            (predicted, &partition.predicted),
        ] {
            let walk: Vec<(usize, HolderId)> = lanes
                .iter()
                .enumerate()
                .filter_map(|(j, lane)| match lane {
                    Some((other, id)) if Arc::ptr_eq(other, partition) => Some((j, *id)),
                    _ => None,
                })
                .collect();
            if walk.is_empty() {
                continue;
            }
            let ids: Vec<HolderId> = walk.iter().map(|&(_, id)| id).collect();
            for ((j, _), blocks) in walk.into_iter().zip(index.find_matches(&prefix, &ids)) {
                hits[j] = hits[j].max(Some(blocks * partition.block_size));
            }
        }
    }
    hits
}
