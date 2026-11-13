//! A partition of the index: the ranks of one model, pool and block size, each on a lane,
//! and the writer task that alone applies their admitted batches, in order.

use std::{collections::HashMap, sync::Arc};

use parking_lot::Mutex;
use tokio::{
    runtime::Handle,
    sync::mpsc::{self, error::TrySendError},
};

use super::{
    hash,
    index::{HolderTable, PrefixIndex},
    HolderId,
};
use crate::kv_events::{model::KvEvent, SourceStatus};

/// Admitted batches a partition's writer may fall behind by before its ranks wait.
const QUEUED_BATCHES: usize = 1024;

pub(crate) struct Partition {
    block_size: u32,
    pub(super) index: PrefixIndex,
    /// Lanes taken, by bit.
    lanes: Mutex<u64>,
    writes: mpsc::Sender<Write>,
    /// Where the writer runs, and a release waits for room in `writes`.
    runtime: Handle,
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
}

impl Partition {
    /// A partition whose writer runs on `runtime`.
    pub(crate) fn start(block_size: u32, runtime: &Handle) -> Arc<Self> {
        let (writes, batches) = mpsc::channel(QUEUED_BATCHES);
        let partition = Arc::new(Self {
            block_size,
            index: PrefixIndex::default(),
            lanes: Mutex::new(0),
            writes,
            runtime: runtime.clone(),
        });
        runtime.spawn(write(partition.clone(), batches));
        partition
    }

    /// A lane for a rank whose batches `status` counts; None once all 64 are taken.
    pub(crate) fn hold(self: &Arc<Self>, status: Arc<Mutex<SourceStatus>>) -> Option<Holder> {
        let mut lanes = self.lanes.lock();
        let free = lanes.trailing_ones();
        (free < 64).then(|| {
            *lanes |= 1 << free;
            Holder {
                id: HolderId(free as u8),
                partition: self.clone(),
                status,
            }
        })
    }
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
        let writes = &self.partition.writes;
        match writes.try_send(Write::Release(self.id)) {
            Err(TrySendError::Full(release)) => {
                let writes = writes.clone();
                self.partition
                    .runtime
                    .spawn(async move { writes.send(release).await });
            }
            // Closed only once the monitor's runtime, and the index with it, is gone.
            Ok(()) | Err(TrySendError::Closed(_)) => {}
        }
    }
}

async fn write(partition: Arc<Partition>, mut writes: mpsc::Receiver<Write>) {
    let mut tables: HashMap<HolderId, HolderTable> = HashMap::new();
    let mut ops = Vec::new();
    while let Some(write) = writes.recv().await {
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
        }
    }
}

/// The leading tokens of `tokens` each lane holds, in their order; one walk per partition.
pub(crate) fn match_holders(
    tokens: &[u32],
    lanes: &[Option<(Arc<Partition>, HolderId)>],
) -> Vec<Option<u32>> {
    let mut hits = vec![None; lanes.len()];
    for (i, lane) in lanes.iter().enumerate() {
        let Some((partition, _)) = lane.as_ref().filter(|_| hits[i].is_none()) else {
            continue;
        };
        let walk: Vec<(usize, HolderId)> = lanes
            .iter()
            .enumerate()
            .skip(i)
            .filter_map(|(j, lane)| match lane {
                Some((other, id)) if Arc::ptr_eq(other, partition) => Some((j, *id)),
                _ => None,
            })
            .collect();
        let ids: Vec<HolderId> = walk.iter().map(|&(_, id)| id).collect();
        let prefix = hash::prefix_hashes(tokens, partition.block_size);
        for ((j, _), blocks) in walk
            .into_iter()
            .zip(partition.index.find_matches(&prefix, &ids))
        {
            hits[j] = Some(blocks * partition.block_size);
        }
    }
    hits
}
