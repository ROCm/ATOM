//! The prefix index of one partition, and the per-holder tables its writer translates
//! events with.

use std::collections::HashMap;

use parking_lot::{RwLock, RwLockReadGuard};

use super::{hash, HolderId, HolderMatch, IndexOp, IndexStats, KvIndex, MatchQuery, PrefixHash};
use crate::kv_events::{DropReason, EngineHash, GroupKind, KvEvent, Namespace, StoredBlock, Tier};

/// Holders per index: a holder id is its lane in the per-tier bit masks.
pub const MAX_HOLDERS: u32 = 64;

const SHARDS: usize = 64;

/// Blocks a walk prefetches ahead of its lookup.
const PREFETCH: usize = 16;

/// The shard takes the top bits and a slot the low bits.
fn shard_of(hash: PrefixHash) -> usize {
    (hash.0 >> 58) as usize
}

/// Prefix hashes are XXH3 digests, so they are their own slot hash. 0 marks an empty slot:
/// hash 0 shares the key of hash 1, as rare as any other 64-bit collision.
fn key(hash: PrefixHash) -> u64 {
    hash.0.max(1)
}

/// Linear-probing table of lane masks. A key's first slot is known before the lookup, so
/// a walk can prefetch it; one slot is 32 bytes, half a cache line.
#[derive(Default)]
struct Shard {
    slots: Vec<(u64, [u64; 3])>,
    len: usize,
}

impl Shard {
    fn with_slots(slots: usize) -> Self {
        Self {
            slots: vec![(0, [0; 3]); slots],
            len: 0,
        }
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

    fn get(&self, key: u64) -> [u64; 3] {
        self.find(key).map_or([0; 3], |i| self.slots[i].1)
    }

    fn upsert(&mut self, key: u64) -> &mut [u64; 3] {
        if (self.len + 1) * 4 > self.slots.len() * 3 {
            let mut grown = Shard::with_slots((self.slots.len() * 2).max(16));
            for (key, lanes) in self.slots.drain(..).filter(|slot| slot.0 != 0) {
                *grown.upsert(key) = lanes;
            }
            *self = grown;
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
        self.slots[hole] = (0, [0; 3]);
        self.len -= 1;
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

/// Tiers in column order.
const TIERS: [Tier; 3] = [Tier::Device, Tier::Host, Tier::Disk];

/// Copies counted per tier, as in SMG: vLLM's `kv_cache_report_mode: "full"` re-reports
/// reused blocks as stored, with no matching remove.
const MAX_COPIES: u16 = 8;

/// Column of a tier in `HolderMatch::blocks`.
fn column(tier: Tier) -> usize {
    match tier {
        Tier::Device => 0,
        Tier::Host => 1,
        Tier::Disk => 2,
    }
}

/// Columns a clear covers: one tier, or all when `tier` is None.
fn columns(tier: Option<Tier>) -> std::ops::Range<usize> {
    tier.map_or(0..3, |tier| column(tier)..column(tier) + 1)
}

fn lane(holder: HolderId) -> u64 {
    1u64.checked_shl(holder.0).unwrap_or(0)
}

/// Flat index: for each prefix hash, the lanes holding it at Device, Host and Disk.
pub struct PrefixIndex {
    shards: Box<[RwLock<Shard>]>,
}

impl Default for PrefixIndex {
    fn default() -> Self {
        Self {
            shards: (0..SHARDS).map(|_| RwLock::default()).collect(),
        }
    }
}

impl KvIndex for PrefixIndex {
    fn apply(&self, holder: HolderId, ops: &[IndexOp]) {
        let lane = lane(holder);
        if lane == 0 {
            return;
        }
        for op in ops {
            let (IndexOp::Store { hash, tier } | IndexOp::Remove { hash, tier }) = *op;
            let column = column(tier);
            let mut shard = self.shards[shard_of(hash)].write();
            if matches!(op, IndexOp::Store { .. }) {
                shard.upsert(key(hash))[column] |= lane;
            } else if let Some(i) = shard.find(key(hash)) {
                let lanes = &mut shard.slots[i].1;
                lanes[column] &= !lane;
                if *lanes == [0; 3] {
                    shard.remove(i);
                }
            }
        }
    }

    /// Rebuilds every shard. A holder table clears through its own entries instead.
    fn clear(&self, holder: HolderId, tier: Option<Tier>) {
        let (columns, lane @ 1..) = (columns(tier), lane(holder)) else {
            return;
        };
        for shard in &self.shards {
            let mut shard = shard.write();
            let mut kept = Shard::with_slots(shard.slots.len());
            for (key, mut lanes) in shard.slots.drain(..).filter(|slot| slot.0 != 0) {
                for held in &mut lanes[columns.clone()] {
                    *held &= !lane;
                }
                if lanes != [0; 3] {
                    *kept.upsert(key) = lanes;
                }
            }
            *shard = kept;
        }
    }

    fn remove_holder(&self, holder: HolderId) {
        self.clear(holder, None);
    }

    fn find_matches(&self, query: &MatchQuery<'_>) -> Vec<HolderMatch> {
        let wanted = query
            .holders
            .iter()
            .fold(0, |lanes, &holder| lanes | lane(holder));
        // Lanes still matching for Device, Host or above, Disk or above, and the lower tiers.
        let mut open = [wanted; 4];
        let mut depth = [[0u32; 4]; MAX_HOLDERS as usize];
        let mut record = |lanes: u64, column: usize, blocks: usize| {
            let mut lanes = lanes;
            while lanes != 0 {
                depth[lanes.trailing_zeros() as usize][column] = blocks as u32;
                lanes &= lanes - 1;
            }
        };
        // The walk read-locks every shard, in order, once, so lookups take no atomics.
        // Writers lock one shard at a time; `read_recursive` keeps walks from queuing
        // behind them.
        let guards: [RwLockReadGuard<'_, Shard>; SHARDS] =
            std::array::from_fn(|i| self.shards[i].read_recursive());
        let shards: [&Shard; SHARDS] = std::array::from_fn(|i| &*guards[i]);
        for &hash in query.prefix.iter().take(PREFETCH) {
            shards[shard_of(hash)].prefetch(key(hash));
        }
        for (i, &hash) in query.prefix.iter().enumerate() {
            if open == [0; 4] {
                break;
            }
            if let Some(&ahead) = query.prefix.get(i + PREFETCH) {
                shards[shard_of(ahead)].prefetch(key(ahead));
            }
            let [device, host, disk] = shards[shard_of(hash)].get(key(hash));
            let present = [device, device | host, device | host | disk, host | disk];
            for (column, lanes) in open.iter_mut().enumerate() {
                record(*lanes & !present[column], column, i);
                *lanes &= present[column];
            }
        }
        for (column, lanes) in open.into_iter().enumerate() {
            record(lanes, column, query.prefix.len());
        }
        query
            .holders
            .iter()
            .filter(|holder| lane(**holder) != 0)
            .filter_map(|&holder| {
                let [device, host, disk, offload] = depth[holder.0 as usize];
                (disk > 0 || offload > 0).then_some(HolderMatch {
                    holder,
                    blocks: [device, host, disk],
                    offload,
                })
            })
            .collect()
    }

    fn stats(&self) -> IndexStats {
        let mut stats = IndexStats::default();
        let mut holders = 0;
        let mut slots = 0;
        for shard in &self.shards {
            let shard = shard.read();
            slots += shard.slots.len();
            for (_, masks) in shard.slots.iter().filter(|slot| slot.0 != 0) {
                stats.entries += 1;
                for (blocks, lanes) in stats.blocks.iter_mut().zip(masks) {
                    *blocks += u64::from(lanes.count_ones());
                    holders |= lanes;
                }
            }
        }
        stats.holders = holders.count_ones();
        stats.approx_bytes = (slots * size_of::<(u64, [u64; 3])>()) as u64;
        stats
    }
}

/// One holder's translation table, owned by the partition's single writer: engine hash to
/// prefix hash, with the holder's reference count per tier. Within a holder a prefix has
/// one engine hash, as SMG assumes.
#[derive(Debug, Default)]
pub struct HolderTable {
    slots: HashMap<EngineHash, Slot>,
    /// Groups whose stores named another kind; removes carry no kind, only the group.
    other_groups: u64,
}

#[derive(Debug)]
struct Slot {
    prefix: PrefixHash,
    counts: [u16; 3],
}

impl HolderTable {
    /// Applies one event of `holder`. A block that cannot be placed drops itself and the
    /// rest of the event; the blocks before it stay. Only full-attention groups are indexed.
    pub fn apply(
        &mut self,
        index: &dyn KvIndex,
        holder: HolderId,
        event: &KvEvent,
    ) -> Result<(), DropReason> {
        let mut ops = Vec::new();
        let group_bit = |idx: u32| 1u64.checked_shl(idx).unwrap_or(0);
        let result = match event {
            KvEvent::Stored { group, .. } if group.kind != GroupKind::Full => {
                self.other_groups |= group_bit(group.idx);
                Ok(())
            }
            KvEvent::Removed { group, .. }
                if group.kind != GroupKind::Full
                    || self.other_groups & group_bit(group.idx) != 0 =>
            {
                Ok(())
            }
            KvEvent::Stored {
                parent,
                blocks,
                tier,
                ns,
                ..
            } => self.store(*parent, blocks, *tier, ns, &mut ops),
            KvEvent::Removed { hashes, tier, .. } => {
                self.remove(hashes, *tier, &mut ops);
                Ok(())
            }
            KvEvent::Cleared { tier } => {
                self.clear(*tier, &mut ops);
                Ok(())
            }
        };
        index.apply(holder, &ops);
        result
    }

    fn store(
        &mut self,
        parent: Option<EngineHash>,
        blocks: &[StoredBlock],
        tier: Tier,
        ns: &Namespace,
        ops: &mut Vec<IndexOp>,
    ) -> Result<(), DropReason> {
        let column = column(tier);
        if ns.opaque {
            return Err(DropReason::Namespace);
        }
        let mut prev = match parent {
            None => None,
            Some(parent) => Some(
                self.slots
                    .get(&parent)
                    .ok_or(DropReason::UnknownParent)?
                    .prefix,
            ),
        };
        let mut buf = Vec::new();
        for block in blocks {
            // A known engine hash keeps its prefix hash: a copy to a lower tier may come
            // without tokens, or without the salt of its first block.
            let prefix = match (self.slots.get(&block.hash), &block.tokens) {
                (Some(slot), _) => slot.prefix,
                (None, Some(tokens)) => {
                    let seed = prev.map_or_else(|| hash::namespace_seed(ns), |_| hash::XXH3_SEED);
                    hash::chain(prev, tokens, seed, &mut buf)
                }
                (None, None) => return Err(DropReason::Unresolved),
            };
            let slot = self.slots.entry(block.hash).or_insert(Slot {
                prefix,
                counts: [0; 3],
            });
            slot.counts[column] = (slot.counts[column] + 1).min(MAX_COPIES);
            if slot.counts[column] == 1 {
                ops.push(IndexOp::Store { hash: prefix, tier });
            }
            prev = Some(prefix);
        }
        Ok(())
    }

    /// Hashes the holder never stored here were stored before the router subscribed.
    fn remove(&mut self, hashes: &[EngineHash], tier: Tier, ops: &mut Vec<IndexOp>) {
        let column = column(tier);
        for hash in hashes {
            let Some(slot) = self.slots.get_mut(hash) else {
                continue;
            };
            match slot.counts[column] {
                0 => continue,
                1 => ops.push(IndexOp::Remove {
                    hash: slot.prefix,
                    tier,
                }),
                _ => {}
            }
            slot.counts[column] -= 1;
            if slot.counts == [0; 3] {
                self.slots.remove(hash);
            }
        }
    }

    /// Removes the holder's own blocks, so a clear costs its entries, not the partition.
    fn clear(&mut self, tier: Option<Tier>, ops: &mut Vec<IndexOp>) {
        let columns = columns(tier);
        self.slots.retain(|_, slot| {
            for column in columns.clone() {
                if slot.counts[column] > 0 {
                    slot.counts[column] = 0;
                    ops.push(IndexOp::Remove {
                        hash: slot.prefix,
                        tier: TIERS[column],
                    });
                }
            }
            slot.counts != [0; 3]
        });
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::Shard;

    /// Keys start their probes at both ends of the table, so runs collide, wrap around and
    /// shift back on removal.
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
                shard.upsert(key)[0] = step;
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
                    assert_eq!(shard.get(key)[0], value);
                }
            }
            assert_eq!(shard.get(key)[0], reference.get(&key).copied().unwrap_or(0));
        }
    }
}
