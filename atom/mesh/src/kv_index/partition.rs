//! A partition of the index: the ranks of one model, pool and block size, each on a lane,
//! and the writer thread that alone applies their admitted batches, in order. The prompts
//! routed to each worker are predictions on lanes of a second index, kept apart from the
//! events. A predicted block matches until the first sweep after a TTL from its last
//! routing, so for up to TTL + TTL/8. A non-streaming prefill's charge waits on its rank's
//! lane for the store of its last full prompt block.

use std::{
    collections::{HashMap, VecDeque},
    sync::{
        atomic::{AtomicBool, AtomicUsize, Ordering},
        Arc, Weak,
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
    index::{find_matches, HolderTable, Op, PrefixIndex},
    HolderId, PrefixHash,
};
use crate::{
    core::worker::PrefillCharge,
    kv_events::{
        model::{KvEvent, StoredBlock},
        SourceStatus,
    },
};

/// Admitted batches a partition's writer may fall behind by before its ranks wait.
const QUEUED_BATCHES: usize = 1024;
/// Routed prompts the writer may fall behind by before more are dropped.
const QUEUED_PREDICTIONS: usize = 256;
/// Their tokens, 4 bytes each, the writer may fall behind by: 64 MB.
const QUEUED_PREDICTION_TOKENS: usize = 1 << 24;
/// Blocks one worker's predictions may hold, as many as a rank's events.
const MAX_PREDICTED_BLOCKS: usize = 1 << 22;

pub(crate) struct Partition {
    block_size: u32,
    pub(super) index: PrefixIndex,
    /// Lanes taken, by bit.
    lanes: Mutex<u64>,
    writes: mpsc::Sender<Write>,
    /// Where a release waits for room in its queue.
    runtime: Handle,
    pub(super) predicted: PrefixIndex,
    predicting: Mutex<u64>,
    /// Set once a worker found no prediction lane left.
    crowded: AtomicBool,
    /// Set once a routed prompt was dropped as the writer was behind.
    behind: AtomicBool,
    /// Apart from `writes`, so that routing never delays a rank's batches.
    predictions: mpsc::Sender<Write>,
    /// Tokens of the prompts queued in `predictions`.
    queued_tokens: AtomicUsize,
    /// Prefill charges by the lane and last full prompt block they wait for, oldest first.
    pub(super) watches: Mutex<HashMap<(HolderId, PrefixHash), VecDeque<Weak<PrefillCharge>>>>,
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
    /// The tokens of a prompt routed to the worker on the prediction lane.
    Predict(HolderId, Vec<u32>),
    /// The worker's predictions leave the index and its lane is free.
    Unpredict(HolderId),
}

impl Partition {
    /// A partition whose ranks write from `runtime`, and whose writer keeps predictions for
    /// `ttl`. The writer has its own thread, as it waits for walks to leave the shards and
    /// then lets them run, and ends with the partition.
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
            behind: AtomicBool::new(false),
            predictions,
            queued_tokens: AtomicUsize::new(0),
            watches: Mutex::default(),
        });
        let writer = Arc::downgrade(&partition);
        std::thread::Builder::new()
            .name("kv-index-writer".into())
            .spawn(move || {
                tokio::runtime::Builder::new_current_thread()
                    .enable_time()
                    .build()
                    .expect("the KV index writer's runtime starts")
                    .block_on(write(writer, batches, routed, ttl))
            })
            .expect("the KV index writer thread starts");
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

    /// Releases `charge` when the rank on `lane` stores the last full block of `tokens`;
    /// a prompt shorter than a block has none to wait for.
    pub(crate) fn watch_prefill(
        &self,
        lane: HolderId,
        tokens: &[u32],
        charge: &Arc<PrefillCharge>,
    ) {
        if let Some(&last) = hash::prefix_hashes(tokens, self.block_size).last() {
            let mut watches = self.watches.lock();
            let waiting = watches.entry((lane, last)).or_default();
            waiting.push_back(Arc::downgrade(charge));
        }
    }

    /// Each block stored on `lane` releases the oldest charge waiting for it that is still
    /// held, as each prefill that computes the block stores it once.
    fn release_prefills(&self, lane: HolderId, stored: &[PrefixHash]) {
        let mut watches = self.watches.lock();
        for &hash in stored {
            let Some(waiting) = watches.get_mut(&(lane, hash)) else {
                continue;
            };
            while let Some(watch) = waiting.pop_front() {
                if watch.upgrade().is_some_and(|charge| charge.release()) {
                    break;
                }
            }
            if waiting.is_empty() {
                watches.remove(&(lane, hash));
            }
        }
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

    pub(crate) fn is_in(&self, partition: &Arc<Partition>) -> bool {
        Arc::ptr_eq(&self.partition, partition)
    }

    /// Predicts the full blocks of a prompt routed to the worker, which the writer hashes;
    /// false, the prompt dropped, while the writer is behind, so that routing never waits
    /// for it.
    pub(crate) fn record(&self, tokens: &[u32]) -> bool {
        let partition = &self.partition;
        let queued = partition
            .queued_tokens
            .fetch_add(tokens.len(), Ordering::Relaxed);
        let sent = queued + tokens.len() <= QUEUED_PREDICTION_TOKENS
            && partition
                .predictions
                .try_send(Write::Predict(self.id, tokens.to_vec()))
                .is_ok();
        if !sent {
            partition
                .queued_tokens
                .fetch_sub(tokens.len(), Ordering::Relaxed);
        }
        sent
    }

    /// Marks the partition as having dropped a prompt; true the first time.
    pub(crate) fn first_drop(&self) -> bool {
        !self.partition.behind.swap(true, Ordering::Relaxed)
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

/// Queues a lane's release, waiting on `runtime` while the queue is full.
fn release(runtime: &Handle, writes: &mpsc::Sender<Write>, release: Write) {
    match writes.try_send(release) {
        Err(TrySendError::Full(release)) => {
            let writes = writes.clone();
            runtime.spawn(async move { writes.send(release).await });
        }
        // The writer ends with the partition, which the lane keeps, or by panicking.
        Ok(()) | Err(TrySendError::Closed(_)) => {}
    }
}

/// Applies the ranks' batches before any prediction. Expired predictions leave at the next
/// sweep, due every eighth of the TTL, and the first sweep after the partition ends the
/// writer.
async fn write(
    partition: Weak<Partition>,
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
                let Some(partition) = partition.upgrade() else {
                    return;
                };
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
                partition.watches.lock().retain(|_, waiting| {
                    waiting.retain(|watch| watch.strong_count() > 0);
                    !waiting.is_empty()
                });
                continue;
            }
        };
        let Some(partition) = partition.upgrade() else {
            return;
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
                // Each stored block's prefix, held already or new.
                let mut stored = Vec::new();
                let dropped: Vec<_> = events
                    .iter()
                    .filter_map(|event| {
                        let applied = table.apply(event, &mut ops);
                        if let KvEvent::Stored { blocks, .. } = event {
                            let prefix =
                                |block: &StoredBlock| table.prefixes.get(&block.hash).copied();
                            stored.extend(blocks.iter().filter_map(prefix));
                        }
                        applied.err()
                    })
                    .collect();
                partition.index.apply(holder, &mut ops);
                partition.release_prefills(holder, &stored);
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
                partition
                    .watches
                    .lock()
                    .retain(|&(lane, _), _| lane != holder);
                *partition.lanes.lock() &= !(1 << holder.0);
            }
            Write::Predict(lane, tokens) => {
                partition
                    .queued_tokens
                    .fetch_sub(tokens.len(), Ordering::Relaxed);
                let until = Instant::now() + ttl;
                let blocks = predicted.entry(lane).or_default();
                for hash in hash::prefix_hashes(&tokens, partition.block_size) {
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
/// and its predictions hold, in the order of the lanes: one hashing and one walk of both
/// indexes per partition.
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
        let here = |lanes: &[Option<Lane>], j: usize| match lanes.get(j) {
            Some(Some((other, id))) if Arc::ptr_eq(other, partition) => Some(*id),
            _ => None,
        };
        let (workers, lanes): (Vec<usize>, Vec<_>) = (0..hits.len())
            .map(|j| (j, [here(events, j), here(predicted, j)]))
            .filter(|(_, lanes)| lanes != &[None, None])
            .unzip();
        // The engine computes the block of the prompt's last token, so it never reports
        // that block as cached.
        let reusable = &tokens[..tokens.len().saturating_sub(1)];
        let prefix = hash::prefix_hashes(reusable, partition.block_size);
        let found = if lanes.iter().all(|[_, predicted]| predicted.is_none()) {
            let holders: Vec<_> = lanes.iter().filter_map(|[events, _]| *events).collect();
            partition.index.find_matches(&prefix, &holders)
        } else {
            find_matches([&partition.index, &partition.predicted], &lanes, &prefix)
        };
        for (j, blocks) in workers.into_iter().zip(found) {
            hits[j] = hits[j].max(Some(blocks * partition.block_size));
        }
    }
    hits
}

#[cfg(test)]
mod tests {
    use std::time::{Duration, Instant};

    use super::*;

    /// A writer woken after its partition is gone ends, which closes the partition's queue.
    #[test]
    fn writer_ends_with_its_partition() {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .build()
            .unwrap();
        let partition = Partition::start(16, Duration::from_secs(3600), runtime.handle());
        let writes = partition.writes.clone();
        drop(partition);
        writes.blocking_send(Write::Release(HolderId(0))).unwrap();
        let deadline = Instant::now() + Duration::from_secs(10);
        while !writes.is_closed() {
            assert!(
                Instant::now() < deadline,
                "the writer outlived its partition"
            );
            std::thread::sleep(Duration::from_millis(1));
        }
    }

    /// A prompt the writer has no room for is dropped, and gives its room back.
    #[tokio::test]
    async fn prompts_are_dropped_while_the_writer_is_behind() {
        let partition = Partition::start(4, Duration::from_secs(3600), &Handle::current());
        let predictor = partition.predict("w").unwrap();
        let queued = || partition.queued_tokens.load(Ordering::Relaxed);
        partition
            .queued_tokens
            .store(QUEUED_PREDICTION_TOKENS, Ordering::Relaxed);
        assert!(!predictor.record(&[1, 2, 3, 4]));
        assert_eq!(queued(), QUEUED_PREDICTION_TOKENS);
        assert!(predictor.first_drop());
        assert!(!predictor.first_drop());
        partition.queued_tokens.store(0, Ordering::Relaxed);
        assert!(predictor.record(&[1, 2, 3, 4]));
    }
}
