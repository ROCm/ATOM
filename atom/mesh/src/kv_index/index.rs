//! The prefix index of one partition, and the per-holder tables its writer translates
//! events with.

use std::collections::HashMap;

use parking_lot::{RwLock, RwLockReadGuard, RwLockWriteGuard};

use super::{hash, HolderId, PrefixHash};
use crate::kv_events::model::{DropReason, EngineHash, KvEvent, StoredBlock};

const SHARDS: usize = 64;

/// Blocks a walk prefetches ahead of its lookup.
const PREFETCH: usize = 16;

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
}

impl Default for PrefixIndex {
    fn default() -> Self {
        Self {
            shards: (0..SHARDS).map(|_| RwLock::default()).collect(),
        }
    }
}

impl PrefixIndex {
    /// Applies and drains one holder's ops, in order, under the write locks of the shards
    /// they touch. Writers and walks lock shards in ascending order, so neither waits on
    /// the other in a cycle: a writer waits for the walks under way, and a walk for at
    /// most one writer's ops.
    pub(super) fn apply(&self, holder: HolderId, ops: &mut Vec<Op>) {
        let lane = lane(holder);
        let touched = ops.iter().fold(0u64, |shards, op| shards | 1 << op.shard());
        let mut shards: [Option<RwLockWriteGuard<'_, Shard>>; SHARDS] = [const { None }; SHARDS];
        for (i, shard) in shards.iter_mut().enumerate() {
            if touched >> i & 1 == 1 {
                *shard = Some(self.shards[i].write());
            }
        }
        for op in ops.drain(..) {
            let Some(shard) = shards[op.shard()].as_mut() else {
                continue;
            };
            match op {
                Op::Store(hash) => *shard.upsert(key(hash)) |= lane,
                Op::Remove(hash) => {
                    if let Some(i) = shard.find(key(hash)) {
                        shard.slots[i].1 &= !lane;
                        if shard.slots[i].1 == 0 {
                            shard.remove(i);
                        }
                    }
                }
            }
        }
    }

    /// The leading blocks of `prefix` each of `holders` holds, in their order. The walk
    /// read-locks every shard, in ascending order, once, and stops when no holder is left.
    pub(super) fn find_matches(&self, prefix: &[PrefixHash], holders: &[HolderId]) -> Vec<u32> {
        let mut open = holders
            .iter()
            .fold(0, |lanes, &holder| lanes | lane(holder));
        let mut depth = [prefix.len() as u32; 64];
        let shards: [RwLockReadGuard<'_, Shard>; SHARDS] =
            std::array::from_fn(|i| self.shards[i].read());
        for &hash in prefix.iter().take(PREFETCH) {
            shards[shard_of(hash)].prefetch(key(hash));
        }
        for (i, &hash) in prefix.iter().enumerate() {
            if open == 0 {
                break;
            }
            if let Some(&ahead) = prefix.get(i + PREFETCH) {
                shards[shard_of(ahead)].prefetch(key(ahead));
            }
            let mut gone = open & !shards[shard_of(hash)].get(key(hash));
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
