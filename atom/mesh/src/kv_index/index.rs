//! The prefix index of one partition, and the per-holder tables its writer translates
//! events with.

use std::{
    collections::HashMap,
    sync::atomic::{AtomicBool, AtomicU64, Ordering},
    time::Instant,
};

use parking_lot::{Mutex, RwLock, RwLockReadGuard, RwLockWriteGuard};

use super::{hash, HolderId, PrefixHash};
use crate::kv_events::model::{DropReason, EngineHash, KvEvent, StoredBlock};

const SHARDS: usize = 64;

/// Blocks a walk prefetches ahead of its lookup.
const PREFETCH: usize = 16;

/// Ops the writer applies per hold of the shards they touch.
const HOLD_OPS: usize = 512;

/// Blocks a walk looks up between giving up the shards the writer waits for.
const YIELD_BLOCKS: usize = 16;

/// Spins of a wait before it yields the thread instead.
const SPINS: u32 = 100;

/// Blocks one holder may index, more than a device holds: a stream that stores without
/// ever removing stops growing the index here.
const MAX_HOLDER_BLOCKS: usize = 1 << 22;

/// The shard takes the top bits and a slot the low bits.
fn shard_of(hash: PrefixHash) -> usize {
    (hash.0 >> 58) as usize
}

/// Prefix hashes are XXH3 digests, so they are their own slot hash. 0 marks an empty slot:
/// hash 0 shares the key of hash 1, as rare as any other 64-bit collision.
fn key(hash: PrefixHash) -> u64 {
    hash.0.max(1)
}

fn lane(holder: HolderId) -> u64 {
    1 << holder.0
}

/// A holder's store or remove of one block, by its prefix hash.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Op {
    Store(PrefixHash),
    Remove(PrefixHash),
}

impl Op {
    fn shard(self) -> usize {
        let (Op::Store(hash) | Op::Remove(hash)) = self;
        shard_of(hash)
    }
}

/// Linear-probing table of lane masks, grown at 3/4 full and shrunk at 1/4. A key's
/// first slot is known before the lookup, so a walk can prefetch it.
#[derive(Default)]
struct Shard {
    slots: Vec<(u64, u64)>,
    len: usize,
}

impl Shard {
    fn with_slots(slots: usize) -> Self {
        Self {
            slots: vec![(0, 0); slots],
            len: 0,
        }
    }

    fn resize(&mut self, slots: usize) {
        let mut resized = Shard::with_slots(slots);
        for (key, lanes) in self.slots.drain(..).filter(|slot| slot.0 != 0) {
            *resized.upsert(key) = lanes;
        }
        *self = resized;
    }

    fn find(&self, key: u64) -> Option<usize> {
        let mask = self.slots.len().checked_sub(1)?;
        let mut i = key as usize & mask;
        loop {
            match self.slots[i].0 {
                0 => return None,
                found if found == key => return Some(i),
                _ => i = (i + 1) & mask,
            }
        }
    }

    fn get(&self, key: u64) -> u64 {
        self.find(key).map_or(0, |i| self.slots[i].1)
    }

    fn apply(&mut self, op: Op, lane: u64) {
        match op {
            Op::Store(hash) => *self.upsert(key(hash)) |= lane,
            Op::Remove(hash) => {
                if let Some(i) = self.find(key(hash)) {
                    self.slots[i].1 &= !lane;
                    if self.slots[i].1 == 0 {
                        self.remove(i);
                    }
                }
            }
        }
    }

    fn upsert(&mut self, key: u64) -> &mut u64 {
        if (self.len + 1) * 4 > self.slots.len() * 3 {
            self.resize((self.slots.len() * 2).max(16));
        }
        let mask = self.slots.len() - 1;
        let mut i = key as usize & mask;
        while self.slots[i].0 != key {
            if self.slots[i].0 == 0 {
                self.slots[i].0 = key;
                self.len += 1;
                break;
            }
            i = (i + 1) & mask;
        }
        &mut self.slots[i].1
    }

    /// Backward-shift deletion: later slots of the probe run move into the hole when their
    /// own first slot is not after it.
    fn remove(&mut self, mut hole: usize) {
        let mask = self.slots.len() - 1;
        let mut i = hole;
        loop {
            i = (i + 1) & mask;
            let key = self.slots[i].0;
            if key == 0 {
                break;
            }
            if i.wrapping_sub(key as usize) & mask >= i.wrapping_sub(hole) & mask {
                self.slots[hole] = self.slots[i];
                hole = i;
            }
        }
        self.slots[hole] = (0, 0);
        self.len -= 1;
        if self.slots.len() > 16 && self.len * 4 < self.slots.len() {
            self.resize(self.slots.len() / 2);
        }
    }

    fn prefetch(&self, key: u64) {
        let Some(mask) = self.slots.len().checked_sub(1) else {
            return;
        };
        let slot = self.slots.as_ptr().wrapping_add(key as usize & mask);
        #[cfg(target_arch = "x86_64")]
        // SAFETY: a prefetch is only a hint; it neither dereferences nor faults.
        unsafe {
            use std::arch::x86_64::{_mm_prefetch, _MM_HINT_T0};
            _mm_prefetch::<_MM_HINT_T0>(slot.cast());
        }
        #[cfg(not(target_arch = "x86_64"))]
        let _ = slot;
    }
}

/// For each prefix hash, the lanes of the holders whose device cache holds it.
pub(super) struct PrefixIndex {
    shards: Box<[RwLock<Shard>]>,
    /// The shards the writer waits for, by bit.
    wanted: AtomicU64,
    /// A walk waited for the writer.
    waited: AtomicBool,
    /// One writer at a time: walks waiting behind two could wait for each other.
    writer: Mutex<()>,
}

impl Default for PrefixIndex {
    fn default() -> Self {
        Self {
            shards: (0..SHARDS).map(|_| RwLock::default()).collect(),
            wanted: AtomicU64::new(0),
            waited: AtomicBool::new(false),
            writer: Mutex::new(()),
        }
    }
}

impl PrefixIndex {
    /// Applies and drains one holder's ops, each shard's in order, `HOLD_OPS` at most per
    /// hold of the shards they touch, so a walk may see part of the batch. Walks soon give
    /// up the shards the writer waits for, and a writer that kept a walk waiting then
    /// pauses as long as it held the shards, so that even a writer without pause leaves
    /// the walks half the time.
    pub(super) fn apply(&self, holder: HolderId, ops: &mut Vec<Op>) {
        let _writer = self.writer.lock();
        let lane = lane(holder);
        ops.sort_by_key(|op| op.shard());
        for held in ops.chunks(HOLD_OPS) {
            let started = Instant::now();
            let touched = held.iter().fold(0, |shards, op| shards | 1 << op.shard());
            let mut shards = self.write(touched);
            for &op in held {
                if let Some(shard) = shards[op.shard()].as_mut() {
                    shard.apply(op, lane);
                }
            }
            drop(shards);
            if self.waited.swap(false, Ordering::Relaxed) {
                let held = started.elapsed();
                spin(|| (started.elapsed() >= held * 2).then_some(()));
            }
        }
        ops.clear();
    }

    /// The write locks of the shards in `touched`, by bit. While the writer waits for any,
    /// walks give them up, and take them again only after the writer.
    fn write(&self, touched: u64) -> [Option<RwLockWriteGuard<'_, Shard>>; SHARDS] {
        let mut shards = [const { None }; SHARDS];
        let mut busy = 0;
        for i in bits(touched) {
            shards[i] = self.shards[i].try_write();
            busy |= u64::from(shards[i].is_none()) << i;
        }
        if busy != 0 {
            self.wanted.fetch_or(busy, Ordering::Relaxed);
            for i in bits(busy) {
                let shard = &self.shards[i];
                shards[i] = Some(spin(|| match shard.is_locked() {
                    true => None,
                    false => shard.try_write(),
                }));
            }
            self.wanted.fetch_and(!busy, Ordering::Relaxed);
        }
        shards
    }

    /// The leading blocks of `prefix` each of `holders` holds, in their order. One walk,
    /// which stops when no holder is left.
    pub(super) fn find_matches(&self, prefix: &[PrefixHash], holders: &[HolderId]) -> Vec<u32> {
        let mut open = holders
            .iter()
            .fold(0, |lanes, &holder| lanes | lane(holder));
        let mut depth = [prefix.len() as u32; 64];
        let mut walk = Walk::new(self);
        for &hash in prefix.iter().take(PREFETCH) {
            walk.prefetch(hash);
        }
        for (i, &hash) in prefix.iter().enumerate() {
            if open == 0 {
                break;
            }
            if i % YIELD_BLOCKS == 0 {
                walk.yield_to_writer();
            }
            if let Some(&ahead) = prefix.get(i + PREFETCH) {
                walk.prefetch(ahead);
            }
            let mut gone = open & !walk.get(hash);
            open &= !gone;
            while gone != 0 {
                depth[gone.trailing_zeros() as usize] = i as u32;
                gone &= gone - 1;
            }
        }
        holders
            .iter()
            .map(|holder| depth[holder.0 as usize])
            .collect()
    }

    /// The leading blocks of `prefix` the lanes of each of `groups` hold between them, in
    /// their order: a block counts for a group while any of its lanes holds it. One walk,
    /// which stops when no group is left.
    pub(super) fn find_shared(&self, prefix: &[PrefixHash], groups: &[u64]) -> Vec<u32> {
        let mut depth: Vec<u32> = groups.iter().map(|_| prefix.len() as u32).collect();
        let mut open: Vec<usize> = (0..groups.len()).collect();
        let mut walk = Walk::new(self);
        for &hash in prefix.iter().take(PREFETCH) {
            walk.prefetch(hash);
        }
        for (i, &hash) in prefix.iter().enumerate() {
            if open.is_empty() {
                break;
            }
            if i % YIELD_BLOCKS == 0 {
                walk.yield_to_writer();
            }
            if let Some(&ahead) = prefix.get(i + PREFETCH) {
                walk.prefetch(ahead);
            }
            let held = walk.get(hash);
            open.retain(|&group| {
                let holds = held & groups[group] != 0;
                if !holds {
                    depth[group] = i as u32;
                }
                holds
            });
        }
        depth
    }

    /// Prefixes, blocks held over all lanes, and slot bytes.
    #[cfg(test)]
    pub(super) fn stats(&self) -> (usize, u32, usize) {
        let (mut entries, mut blocks, mut slots) = (0, 0, 0);
        for shard in &self.shards {
            let shard = shard.read();
            slots += shard.slots.len();
            for (_, lanes) in shard.slots.iter().filter(|slot| slot.0 != 0) {
                entries += 1;
                blocks += lanes.count_ones();
            }
        }
        (entries, blocks, slots * size_of::<(u64, u64)>())
    }
}

/// The leading blocks of `prefix` each worker holds, in their order, the more of what its
/// lanes `[events, predicted]` hold in the two indexes. One walk, which looks a block up
/// in the predictions only for the workers whose events miss it: both hold whole prefixes,
/// so that a worker's predictions count only past the end of its events.
pub(super) fn find_matches(
    [events, predicted]: [&PrefixIndex; 2],
    workers: &[[Option<HolderId>; 2]],
    prefix: &[PrefixHash],
) -> Vec<u32> {
    let blocks = prefix.len() as u32;
    // By index and lane, the first block the lane misses, and by lane of the events, the
    // worker's lane of the predictions, which goes on from there.
    let mut depth = [[blocks; 64]; 2];
    let mut then = [None; 64];
    let (mut open, mut predicting) = (0, 0);
    for &[on_events, on_predicted] in workers {
        match on_events {
            Some(holder) => {
                open |= lane(holder);
                then[holder.0 as usize] = on_predicted;
            }
            None => predicting |= on_predicted.map_or(0, lane),
        }
    }
    let mut walk = Walk::new(events);
    let mut predictions: Option<Walk<'_>> = None;
    for &hash in prefix.iter().take(PREFETCH) {
        walk.prefetch(hash);
    }
    for (i, &hash) in prefix.iter().enumerate() {
        if open | predicting == 0 {
            break;
        }
        if i % YIELD_BLOCKS == 0 {
            walk.yield_to_writer();
            if let Some(predictions) = &mut predictions {
                predictions.yield_to_writer();
            }
        }
        if open != 0 {
            if let Some(&ahead) = prefix.get(i + PREFETCH) {
                walk.prefetch(ahead);
            }
            let mut gone = open & !walk.get(hash);
            open &= !gone;
            while gone != 0 {
                let holder = gone.trailing_zeros() as usize;
                depth[0][holder] = i as u32;
                predicting |= then[holder].map_or(0, lane);
                gone &= gone - 1;
            }
        }
        if predicting != 0 {
            let predictions = predictions.get_or_insert_with(|| Walk::new(predicted));
            if let Some(&ahead) = prefix.get(i + PREFETCH) {
                predictions.prefetch(ahead);
            }
            let mut gone = predicting & !predictions.get(hash);
            predicting &= !gone;
            while gone != 0 {
                depth[1][gone.trailing_zeros() as usize] = i as u32;
                gone &= gone - 1;
            }
        }
    }
    workers
        .iter()
        .map(|&[events, predicted]| {
            let events = events.map_or(0, |holder| depth[0][holder.0 as usize]);
            match predicted {
                Some(holder) if events < blocks => events.max(depth[1][holder.0 as usize]),
                _ => events,
            }
        })
        .collect()
}

/// A walk's read locks, each shard's taken once and past a waiting writer, so that a walk
/// never waits behind it for the walks under way. Every `YIELD_BLOCKS` blocks, and while
/// it waits, the walk gives up the shards the writer waits for, and to take one again it
/// waits for the writer.
struct Walk<'a> {
    index: &'a PrefixIndex,
    shards: [Option<RwLockReadGuard<'a, Shard>>; SHARDS],
}

impl<'a> Walk<'a> {
    fn new(index: &'a PrefixIndex) -> Self {
        let wanted = index.wanted.load(Ordering::Relaxed);
        let shards = std::array::from_fn(|i| match wanted >> i & 1 {
            0 => index.shards[i].try_read(),
            _ => None,
        });
        Self { index, shards }
    }

    fn yield_to_writer(&mut self) {
        let wanted = self.index.wanted.load(Ordering::Relaxed);
        if wanted != 0 {
            self.give_up(wanted);
        }
    }

    #[cold]
    fn give_up(&mut self, wanted: u64) {
        for i in bits(wanted) {
            self.shards[i] = None;
        }
    }

    fn prefetch(&self, hash: PrefixHash) {
        if let Some(shard) = &self.shards[shard_of(hash)] {
            shard.prefetch(key(hash));
        }
    }

    fn get(&mut self, hash: PrefixHash) -> u64 {
        match &self.shards[shard_of(hash)] {
            Some(shard) => shard.get(key(hash)),
            None => self.take(hash),
        }
    }

    /// Takes the shard of `hash` again, behind the writer, and looks it up.
    #[cold]
    fn take(&mut self, hash: PrefixHash) -> u64 {
        let (index, i) = (self.index, shard_of(hash));
        let take = || match index.wanted.load(Ordering::Relaxed) >> i & 1 {
            0 => index.shards[i].try_read(),
            _ => None,
        };
        let shard = take().unwrap_or_else(|| {
            index.waited.store(true, Ordering::Relaxed);
            spin(|| {
                self.yield_to_writer();
                take()
            })
        });
        let found = shard.get(key(hash));
        self.shards[i] = Some(shard);
        found
    }
}

/// The set bits of `mask`, lowest first.
fn bits(mut mask: u64) -> impl Iterator<Item = usize> {
    std::iter::from_fn(move || {
        let i = mask.trailing_zeros() as usize;
        mask &= mask.wrapping_sub(1);
        (i < 64).then_some(i)
    })
}

/// Spins until `ready` gives a value, yielding the thread once it takes long.
fn spin<T>(mut ready: impl FnMut() -> Option<T>) -> T {
    let mut spins = 0;
    loop {
        if let Some(value) = ready() {
            return value;
        }
        if spins < SPINS {
            std::hint::spin_loop();
            spins += 1;
        } else {
            std::thread::yield_now();
        }
    }
}

/// One holder's prefix hash of each engine hash it holds, owned by its partition's
/// writer. Within a holder a prefix has one engine hash, as SMG assumes.
#[derive(Debug)]
pub(super) struct HolderTable {
    pub(super) prefixes: HashMap<EngineHash, PrefixHash>,
    pub(super) max_blocks: usize,
}

impl Default for HolderTable {
    fn default() -> Self {
        Self {
            prefixes: HashMap::new(),
            max_blocks: MAX_HOLDER_BLOCKS,
        }
    }
}

impl HolderTable {
    pub(super) fn len(&self) -> usize {
        self.prefixes.len()
    }

    /// Translates one event of the holder into `ops`. A block that cannot be placed drops
    /// itself and the rest of the event; the blocks before it stay.
    pub(super) fn apply(&mut self, event: &KvEvent, ops: &mut Vec<Op>) -> Result<(), DropReason> {
        match event {
            KvEvent::Stored { parent, blocks } => return self.store(*parent, blocks, ops),
            KvEvent::Removed { hashes } => {
                ops.extend(
                    hashes
                        .iter()
                        .filter_map(|hash| self.prefixes.remove(hash))
                        .map(Op::Remove),
                );
                if self.prefixes.capacity() > 4 * self.prefixes.len().max(1024) {
                    self.prefixes.shrink_to_fit();
                }
            }
            KvEvent::Cleared => self.clear(ops),
        }
        Ok(())
    }

    /// A block stored again is held once, so its first remove drops it: an engine that
    /// holds a hash twice, or re-reports hits, under-reports rather than keeps a stale hit.
    fn store(
        &mut self,
        parent: Option<EngineHash>,
        blocks: &[StoredBlock],
        ops: &mut Vec<Op>,
    ) -> Result<(), DropReason> {
        let mut prev = match parent {
            None => None,
            Some(parent) => Some(
                *self
                    .prefixes
                    .get(&parent)
                    .ok_or(DropReason::UnknownParent)?,
            ),
        };
        for block in blocks {
            let prefix = match self.prefixes.get(&block.hash) {
                Some(&prefix) => prefix,
                None if self.prefixes.len() >= self.max_blocks => return Err(DropReason::Capacity),
                None => {
                    let prefix = hash::chain(prev, &block.tokens);
                    self.prefixes.insert(block.hash, prefix);
                    ops.push(Op::Store(prefix));
                    prefix
                }
            };
            prev = Some(prefix);
        }
        Ok(())
    }

    /// Translates into `ops` a batch for the lane of the blocks a rank shares, which holds
    /// only blocks the rank does: `events`, of the rank's device tier, already applied to
    /// its table `rank`, remove blocks here too, and `shared`, of its shared tier, name
    /// blocks by hash, placed where `rank` holds them. A shared block the rank does not
    /// hold is dropped as one of an unknown parent.
    pub(super) fn share(
        &mut self,
        rank: &HolderTable,
        events: &[KvEvent],
        shared: &[KvEvent],
        ops: &mut Vec<Op>,
    ) -> Vec<DropReason> {
        let mut dropped = Vec::new();
        let removals = events
            .iter()
            .filter(|event| !matches!(event, KvEvent::Stored { .. }));
        for event in removals.chain(shared) {
            let KvEvent::Stored { blocks, .. } = event else {
                let _ = self.apply(event, ops);
                continue;
            };
            for block in blocks {
                match rank.prefixes.get(&block.hash) {
                    Some(&prefix) => {
                        if self.prefixes.insert(block.hash, prefix).is_none() {
                            ops.push(Op::Store(prefix));
                        }
                    }
                    None => dropped.push(DropReason::UnknownParent),
                }
            }
        }
        dropped
    }

    /// Removes the holder's own blocks, so a clear costs its entries, not the partition's.
    pub(super) fn clear(&mut self, ops: &mut Vec<Op>) {
        ops.extend(
            std::mem::take(&mut self.prefixes)
                .into_values()
                .map(Op::Remove),
        );
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::Shard;

    /// Keys start their probes at both ends of the table, so runs collide, wrap around and
    /// shift back on removal; emptied, the table shrinks back.
    #[test]
    fn shard_matches_a_hash_map() {
        let mut state = 0x2545_f491_4f6c_dd1d_u64;
        let mut below = move |n: u64| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state % n
        };
        let mut shard = Shard::default();
        let mut reference = HashMap::new();
        for step in 1..20_000 {
            let low = match below(2) {
                0 => below(8),
                _ => u64::from(u32::MAX) - below(8),
            };
            let key = (below(64) + 1) << 32 | low;
            if below(3) < 2 {
                *shard.upsert(key) = step;
                reference.insert(key, step);
            } else {
                if let Some(i) = shard.find(key) {
                    shard.remove(i);
                }
                reference.remove(&key);
            }
            assert_eq!(shard.len, reference.len());
            if step % 100 == 0 {
                for (&key, &value) in &reference {
                    assert_eq!(shard.get(key), value);
                }
            }
            assert_eq!(shard.get(key), reference.get(&key).copied().unwrap_or(0));
        }
        let grown = shard.slots.len();
        for key in reference.keys() {
            shard.remove(shard.find(*key).unwrap());
        }
        assert!(grown >= 256 && shard.slots.len() == 16, "{grown} slots");
    }
}
