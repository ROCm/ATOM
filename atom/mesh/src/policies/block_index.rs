//! Router-side estimate of the prompt blocks each worker holds, keyed by chained block hashes.
//! It cannot see engine evictions; over capacity, the least recently placed entries go first.

use std::{
    collections::HashMap,
    hash::{BuildHasherDefault, Hasher},
};

use parking_lot::RwLock;

/// Chained keys of `items`, one per full block of `block_len` items.
pub(crate) fn chain_keys<T: bytemuck::Pod>(items: &[T], block_len: usize, seed: u64) -> Vec<u64> {
    let mut parent = seed;
    items
        .chunks_exact(block_len.max(1))
        .map(|block| {
            parent = xxhash_rust::xxh3::xxh3_64_with_seed(bytemuck::cast_slice(block), parent);
            parent
        })
        .collect()
}

/// Keys are already uniform hashes, so they need no further hashing.
#[derive(Default)]
struct KeyHasher(u64);

impl Hasher for KeyHasher {
    fn finish(&self) -> u64 {
        self.0
    }
    fn write(&mut self, bytes: &[u8]) {
        for &byte in bytes {
            self.0 = self.0.rotate_left(8) ^ u64::from(byte);
        }
    }
    fn write_u64(&mut self, value: u64) {
        self.0 = value;
    }
}

type KeyMap<V> = HashMap<u64, V, BuildHasherDefault<KeyHasher>>;

/// One (block, worker) pair as a single key.
fn entry_key(block: u64, worker: u32) -> u64 {
    block ^ (u64::from(worker) + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15)
}

#[derive(Default)]
struct State {
    /// (block, worker) -> placement tick.
    entries: KeyMap<u64>,
    workers: HashMap<String, u32>,
    clock: u64,
}

impl State {
    /// Keep the newest three quarters of `capacity`.
    fn evict(&mut self, capacity: usize) {
        let target = capacity - capacity / 4;
        let mut ticks: Vec<u64> = self.entries.values().copied().collect();
        let drop_count = ticks.len().saturating_sub(target);
        if drop_count == 0 {
            return;
        }
        let (_, &mut cutoff, _) = ticks.select_nth_unstable(drop_count - 1);
        self.entries.retain(|_, tick| *tick > cutoff);
    }
}

#[derive(Default)]
pub(crate) struct BlockIndex {
    /// Most (block, worker) entries kept.
    capacity: usize,
    state: RwLock<State>,
}

impl std::fmt::Debug for BlockIndex {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("BlockIndex")
            .field("capacity", &self.capacity)
            .field("entries", &self.len())
            .finish()
    }
}

impl BlockIndex {
    pub(crate) fn new(capacity: usize) -> Self {
        Self {
            capacity: capacity.max(1),
            state: RwLock::default(),
        }
    }

    pub(crate) fn len(&self) -> usize {
        self.state.read().entries.len()
    }

    /// Leading blocks of `keys` that each of `workers` holds, in the same order.
    pub(crate) fn matched_blocks(&self, keys: &[u64], workers: &[&str]) -> Vec<usize> {
        let state = self.state.read();
        let mut matched = vec![0; workers.len()];
        let mut alive: Vec<(usize, u32)> = workers
            .iter()
            .enumerate()
            .filter_map(|(position, url)| state.workers.get(*url).map(|&slot| (position, slot)))
            .collect();
        for (depth, &key) in keys.iter().enumerate() {
            alive.retain(|&(position, slot)| {
                let held = state.entries.contains_key(&entry_key(key, slot));
                if held {
                    matched[position] = depth + 1;
                }
                held
            });
            if alive.is_empty() {
                break;
            }
        }
        matched
    }

    /// Record that `worker` holds every block of `keys`.
    pub(crate) fn insert(&self, worker: &str, keys: &[u64]) {
        if keys.is_empty() {
            return;
        }
        let mut state = self.state.write();
        state.clock += 1;
        let tick = state.clock;
        let next_slot = state.workers.len() as u32;
        let slot = *state.workers.entry(worker.to_string()).or_insert(next_slot);
        for &key in keys {
            state.entries.insert(entry_key(key, slot), tick);
        }
        if state.entries.len() > self.capacity {
            state.evict(self.capacity);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chain_keys_identify_whole_prefixes() {
        let a = chain_keys(&[1u32, 2, 3, 4, 5, 6, 7], 2, 0);
        let b = chain_keys(&[1u32, 2, 3, 4, 9, 9], 2, 0);
        let c = chain_keys(&[0u32, 2, 3, 4], 2, 0);
        assert_eq!(a.len(), 3, "the partial tail block is not keyed");
        assert_eq!(a[..2], b[..2]);
        assert_ne!(a[2], b[2]);
        // Same block content after a different parent is a different key.
        assert_ne!(a[1], c[1]);
        assert_ne!(chain_keys(&[1u32, 2], 2, 0), chain_keys(&[1u32, 2], 2, 1));
    }

    #[test]
    fn matched_blocks_counts_each_workers_leading_run() {
        let index = BlockIndex::new(1024);
        let keys = chain_keys(&(0..64u32).collect::<Vec<_>>(), 16, 0);
        index.insert("w0", &keys);
        index.insert("w1", &keys[..2]);
        assert_eq!(
            index.matched_blocks(&keys, &["w1", "unknown", "w0"]),
            vec![2, 0, 4]
        );
        let other = chain_keys(&(100..164u32).collect::<Vec<_>>(), 16, 0);
        assert_eq!(index.matched_blocks(&other, &["w0", "w1"]), vec![0, 0]);
        assert_eq!(index.matched_blocks(&[], &["w0"]), vec![0]);
    }

    #[test]
    fn least_recently_placed_entries_are_evicted_first() {
        let index = BlockIndex::new(7);
        let old = chain_keys(&(0..64u32).collect::<Vec<_>>(), 16, 0);
        let shared = chain_keys(&(0..32u32).collect::<Vec<_>>(), 16, 0);
        let new = chain_keys(&(500..564u32).collect::<Vec<_>>(), 16, 0);
        index.insert("w0", &old);
        // Re-placing the shared prefix keeps it fresh.
        index.insert("w0", &shared);
        index.insert("w1", &new);
        assert_eq!(index.len(), 6);
        assert_eq!(index.matched_blocks(&old, &["w0"]), vec![2]);
        assert_eq!(index.matched_blocks(&new, &["w1"]), vec![4]);
    }
}
