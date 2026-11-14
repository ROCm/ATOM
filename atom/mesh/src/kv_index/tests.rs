use std::{
    collections::{BTreeMap, HashSet},
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
    time::{Duration, Instant},
};

use parking_lot::Mutex;
use xxhash_rust::xxh3::xxh3_64_with_seed;

use super::{
    hash,
    index::{HolderTable, Op, PrefixIndex},
    match_holders, HolderId, Partition, PrefixHash,
};
use crate::kv_events::{
    model::{DropReason, EngineHash, KvEvent, StoredBlock},
    SourceStatus,
};

/// Golden values from python-xxhash's `xxh3_64_intdigest` with SMG's formulas.
#[test]
fn hashes_follow_smg() {
    let tokens: Vec<u32> = (1..=9).collect();
    assert_eq!(
        hash::prefix_hashes(&tokens, 4),
        [
            PrefixHash(0xcb38e4d7fb2dca5c),
            PrefixHash(0x44a2b2379d6d3c3d)
        ]
    );
}

struct Rng(u64);

impl Rng {
    fn below(&mut self, n: usize) -> usize {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 % n as u64) as usize
    }
}

/// An engine's own block hash: of the whole prefix, as the router's, by another function.
fn engine_hash(holder: usize, prefix: &[u32]) -> EngineHash {
    EngineHash(xxh3_64_with_seed(
        bytemuck::cast_slice(prefix),
        7 + holder as u64,
    ))
}

/// The store of `blocks` of `seq`, continuing the chain of the block before them.
fn stored(holder: usize, seq: &[u32], blocks: std::ops::Range<usize>, bs: usize) -> KvEvent {
    KvEvent::Stored {
        parent: (blocks.start > 0).then(|| engine_hash(holder, &seq[..blocks.start * bs])),
        blocks: blocks
            .map(|b| StoredBlock {
                hash: engine_hash(holder, &seq[..(b + 1) * bs]),
                tokens: seq[b * bs..(b + 1) * bs].into(),
            })
            .collect(),
    }
}

/// Applies one event of `holder`, as its partition's writer does.
fn apply(
    index: &PrefixIndex,
    table: &mut HolderTable,
    holder: HolderId,
    event: &KvEvent,
) -> Result<(), DropReason> {
    let mut ops = Vec::new();
    let result = table.apply(event, &mut ops);
    index.apply(holder, &mut ops);
    result
}

/// Random stores, evictions anywhere in a chain and clears over three holders sharing
/// prefixes; after every step each holder's match of every sequence equals the reference,
/// and once every holder clears, the index is empty again.
#[test]
fn index_matches_the_naive_reference() {
    const HOLDERS: usize = 3;
    const BLOCKS: usize = 12;
    for bs in [1, 4, 16] {
        let mut rng = Rng(0x2545_f491_4f6c_dd1d ^ bs as u64);
        let mut seqs: Vec<Vec<u32>> = Vec::new();
        for _ in 0..6 {
            let shared = match seqs.len() {
                0 => Vec::new(),
                n => seqs[rng.below(n)][..rng.below(BLOCKS) * bs].to_vec(),
            };
            let fresh = (shared.len()..BLOCKS * bs).map(|_| rng.below(1000) as u32);
            seqs.push(shared.iter().copied().chain(fresh).collect());
        }
        let index = PrefixIndex::default();
        let mut tables: Vec<HolderTable> = (0..HOLDERS).map(|_| HolderTable::default()).collect();
        // The token prefixes each holder holds.
        let mut held = vec![HashSet::<Vec<u32>>::new(); HOLDERS];
        let holders: Vec<HolderId> = (0..HOLDERS as u8).map(HolderId).collect();
        let mut deepest = 0;
        for _ in 0..2000 {
            let h = rng.below(HOLDERS);
            let seq = &seqs[rng.below(seqs.len())];
            let event = match rng.below(20) {
                0 => {
                    held[h].clear();
                    KvEvent::Cleared
                }
                1..=8 => {
                    let mut keys: Vec<&Vec<u32>> = held[h].iter().collect();
                    keys.sort();
                    let Some(key) = keys
                        .get(rng.below(keys.len().max(1)))
                        .map(|key| (*key).clone())
                    else {
                        continue;
                    };
                    held[h].remove(&key);
                    KvEvent::Removed {
                        hashes: vec![engine_hash(h, &key)],
                    }
                }
                _ => {
                    // An engine names a parent it holds and stores blocks it does not.
                    let start = rng.below(BLOCKS);
                    if start > 0 && !held[h].contains(&seq[..start * bs]) {
                        continue;
                    }
                    let mut end = start;
                    while end < BLOCKS.min(start + 4) && !held[h].contains(&seq[..(end + 1) * bs]) {
                        end += 1;
                    }
                    if end == start {
                        continue;
                    }
                    for b in start..end {
                        held[h].insert(seq[..(b + 1) * bs].to_vec());
                    }
                    stored(h, seq, start..end, bs)
                }
            };
            apply(&index, &mut tables[h], holders[h], &event).unwrap();
            for seq in &seqs {
                let found = index.find_matches(&hash::prefix_hashes(seq, bs as u32), &holders);
                for (h, held) in held.iter().enumerate() {
                    let expected = (1..=BLOCKS)
                        .take_while(|&b| held.contains(&seq[..b * bs]))
                        .count();
                    assert_eq!(found[h], expected as u32, "block size {bs}, holder {h}");
                    deepest = deepest.max(found[h]);
                }
            }
        }
        assert!(deepest > 1, "{deepest}");
        let blocks: usize = held.iter().map(HashSet::len).sum();
        assert_eq!(index.stats().1 as usize, blocks);
        for (h, table) in tables.iter_mut().enumerate() {
            apply(&index, table, holders[h], &KvEvent::Cleared).unwrap();
        }
        let (entries, blocks, bytes) = index.stats();
        assert_eq!((entries, blocks), (0, 0));
        assert!(bytes <= 64 * 16 * 16, "{bytes} bytes of slots");
    }
}

#[test]
fn concurrent_walks_and_writes_finish() {
    let index = PrefixIndex::default();
    let tokens: Vec<u32> = (0..1024).collect();
    let prefix = hash::prefix_hashes(&tokens, 1);
    let holders: Vec<HolderId> = (0..8).map(HolderId).collect();
    std::thread::scope(|scope| {
        for writer in 0..2u8 {
            let (index, prefix) = (&index, &prefix);
            scope.spawn(move || {
                for round in 0..100u8 {
                    let holder = HolderId(writer * 4 + round % 4);
                    for op in [Op::Store, Op::Remove] {
                        index.apply(holder, &mut prefix.iter().copied().map(op).collect());
                    }
                }
            });
        }
        for _ in 0..4 {
            let (index, prefix, holders) = (&index, &prefix, &holders);
            scope.spawn(move || (0..100).for_each(|_| drop(index.find_matches(prefix, holders))));
        }
    });
    assert_eq!(index.stats().0, 0);
}

/// A block stored again is held once, so that one remove drops it: re-reports, as with
/// vLLM's `kv_cache_report_mode: "full"`, would otherwise keep it indexed for good. A
/// store needs its parent held, and a holder at its cap stores no more.
#[test]
fn stores_are_idempotent_and_bounded() {
    let (index, id) = (PrefixIndex::default(), HolderId(0));
    let mut table = HolderTable::default();
    let seq: Vec<u32> = (0..8).collect();
    let prefix = hash::prefix_hashes(&seq, 4);
    let held = |holder| index.find_matches(&prefix, &[holder])[0];
    let child = stored(0, &seq, 1..2, 4);
    assert_eq!(
        apply(&index, &mut table, id, &child),
        Err(DropReason::UnknownParent)
    );
    for _ in 0..3 {
        apply(&index, &mut table, id, &stored(0, &seq, 0..2, 4)).unwrap();
    }
    assert_eq!(held(id), 2);
    let tail = KvEvent::Removed {
        hashes: vec![engine_hash(0, &seq)],
    };
    apply(&index, &mut table, id, &tail).unwrap();
    assert_eq!((held(id), table.len()), (1, 1));
    let mut capped = HolderTable {
        max_blocks: 1,
        ..HolderTable::default()
    };
    let both = stored(1, &seq, 0..2, 4);
    assert_eq!(
        apply(&index, &mut capped, HolderId(1), &both),
        Err(DropReason::Capacity)
    );
    assert_eq!(held(HolderId(1)), 1);
}

async fn settles(
    tokens: &[u32],
    lanes: &[Option<(Arc<Partition>, HolderId)>],
    want: &[Option<u32>],
) {
    for _ in 0..200 {
        if match_holders(tokens, lanes) == want {
            return;
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    panic!("{:?}, not {want:?}", match_holders(tokens, lanes));
}

/// Batches reach the index through the partition's writer, in order, a reset clearing
/// the rank first. Lanes run out at 64; one given back is free once its blocks are gone.
#[tokio::test]
async fn the_writer_applies_batches_and_frees_released_lanes() {
    let partition = Partition::start(4, &tokio::runtime::Handle::current());
    let status = Arc::new(Mutex::new(SourceStatus::default()));
    let mut holders: Vec<_> = (0..64)
        .map(|_| partition.hold(status.clone()).unwrap())
        .collect();
    assert!(partition.hold(status.clone()).is_none());
    let seq: Vec<u32> = (0..10).collect();
    let lanes: Vec<_> = holders[..2].iter().map(|h| Some(h.lane())).collect();
    holders[0]
        .write(false, vec![stored(0, &seq, 0..2, 4)])
        .await;
    holders[1]
        .write(false, vec![stored(1, &seq, 0..1, 4)])
        .await;
    settles(&seq, &lanes, &[Some(8), Some(4)]).await;
    assert_eq!(status.lock().indexed_blocks, Some(1));
    holders[0].write(true, vec![stored(0, &seq, 0..1, 4)]).await;
    settles(&seq, &lanes, &[Some(4), Some(4)]).await;
    let first = KvEvent::Removed {
        hashes: vec![engine_hash(0, &seq[..4])],
    };
    holders[0]
        .write(false, vec![first, stored(0, &seq, 1..2, 4)])
        .await;
    settles(&seq, &lanes, &[Some(0), Some(4)]).await;
    let dropped = BTreeMap::from([(DropReason::UnknownParent, 1)]);
    assert_eq!(status.lock().dropped_events, dropped);

    drop(holders.remove(1));
    let holder = loop {
        if let Some(holder) = partition.hold(status.clone()) {
            break holder;
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    };
    assert_eq!(holder.lane().1, HolderId(1));
    assert_eq!(match_holders(&seq, &[Some(holder.lane())]), [Some(0)]);
}

fn resident_bytes() -> u64 {
    let statm = std::fs::read_to_string("/proc/self/statm").unwrap();
    statm
        .split_whitespace()
        .nth(1)
        .unwrap()
        .parse::<u64>()
        .unwrap()
        * 4096
}

/// A bench's size: `name` from the environment, `default` without it.
fn size(name: &str, default: usize) -> usize {
    std::env::var(name).map_or(default, |value| value.parse().unwrap())
}

/// `KV_BENCH_WORKERS` (16) holders sharing `KV_BENCH_ENTRIES` (1M) entries of other
/// prefixes, a deterministic prompt of `KV_BENCH_TOKENS` (100k) tokens and its prefix
/// hashes in blocks of 16.
fn filled(index: &PrefixIndex) -> (Vec<HolderId>, Vec<u32>, Vec<PrefixHash>) {
    let workers = size("KV_BENCH_WORKERS", 16);
    let holders: Vec<HolderId> = (0..workers as u8).map(HolderId).collect();
    let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
    for holder in &holders {
        let other = (0..size("KV_BENCH_ENTRIES", 1_000_000) / workers)
            .map(|_| Op::Store(PrefixHash(rng.below(usize::MAX) as u64)));
        index.apply(*holder, &mut other.collect());
    }
    let tokens: Vec<u32> = (0..size("KV_BENCH_TOKENS", 100_000))
        .map(|i| (i * 7919 % 150_000) as u32)
        .collect();
    let prefix = hash::prefix_hashes(&tokens, 16);
    (holders, tokens, prefix)
}

fn percentiles(mut samples: Vec<Duration>) -> String {
    samples.sort();
    let at = |q: f64| samples[((samples.len() as f64 * q) as usize).min(samples.len() - 1)];
    format!("p50 {:?}, p99 {:?}", at(0.5), at(0.99))
}

/// `cargo test --release --lib kv_index::tests::bench -- --ignored --nocapture --test-threads=1`;
/// `filled` reads the sizes, and `KV_BENCH_SAMPLES` (2000) the samples.
#[test]
#[ignore]
fn bench_query_16_workers_100k_tokens() {
    let index = PrefixIndex::default();
    let (holders, tokens, prefix) = filled(&index);
    let (workers, count) = (holders.len(), size("KV_BENCH_SAMPLES", 2000));
    let hashing = (0..count)
        .map(|_| {
            let started = Instant::now();
            std::hint::black_box(hash::prefix_hashes(&tokens, 16));
            started.elapsed()
        })
        .collect();
    println!(
        "prefix hashes of {} tokens: {}",
        tokens.len(),
        percentiles(hashing)
    );
    let mut held = vec![0; workers];
    for (name, full) in [
        ("staggered, worker k holds (k+1)/n of the prefix", false),
        ("worst, all hold all of it", true),
    ] {
        for (k, holder) in holders.iter().enumerate() {
            let target = if full {
                prefix.len()
            } else {
                prefix.len() * (k + 1) / workers
            };
            index.apply(
                *holder,
                &mut prefix[held[k]..target]
                    .iter()
                    .copied()
                    .map(Op::Store)
                    .collect(),
            );
            held[k] = target;
        }
        for hashing in [false, true] {
            let samples = (0..count)
                .map(|_| {
                    let started = Instant::now();
                    let fresh = hashing.then(|| hash::prefix_hashes(&tokens, 16));
                    let found = index.find_matches(fresh.as_deref().unwrap_or(&prefix), &holders);
                    assert_eq!(found.len(), workers);
                    started.elapsed()
                })
                .collect();
            println!("{name}, hashing {hashing}: {}", percentiles(samples));
        }
    }
}

/// `KV_BENCH_BLOCKS` (1M) blocks: the memory they take, and applying them, hashing
/// included.
#[test]
#[ignore]
fn bench_memory_1m_blocks() {
    let wanted = size("KV_BENCH_BLOCKS", 1_000_000);
    let before = resident_bytes();
    let index = PrefixIndex::default();
    let mut tables: Vec<HolderTable> = (0..16).map(|_| HolderTable::default()).collect();
    let mut applying = Duration::ZERO;
    // 16 holders, chains of 64 blocks of 16 tokens.
    for chain in 0..wanted / 64 {
        let h = chain % 16;
        let seq: Vec<u32> = (0..64 * 16).map(|i| (chain * 1024 + i) as u32).collect();
        let event = stored(h, &seq, 0..64, 16);
        let started = Instant::now();
        apply(&index, &mut tables[h], HolderId(h as u8), &event).unwrap();
        applying += started.elapsed();
    }
    let after = resident_bytes();
    let (entries, blocks, bytes) = index.stats();
    println!(
        "{entries} entries, {blocks} blocks: resident +{} MB (index slots {} MB); applied in {applying:?}, {:?} a block",
        (after - before) >> 20,
        bytes >> 20,
        applying / blocks as u32
    );
    assert!(after - before < (200 << 20) * wanted as u64 / 1_000_000);
}

/// What `cache_aware` asks, hashing the request included: 16 workers, worker k holding
/// (k+1)/16 of a 100k-token prompt, in one partition with 1M other entries.
#[tokio::test]
#[ignore]
async fn bench_match_workers_16_workers() {
    let partition = Partition::start(16, &tokio::runtime::Handle::current());
    let (_, tokens, prefix) = filled(&partition.index);
    let status = Arc::new(Mutex::new(SourceStatus::default()));
    let holders: Vec<_> = (0..16)
        .map(|_| partition.hold(status.clone()).unwrap())
        .collect();
    for (k, holder) in holders.iter().enumerate() {
        let held = &prefix[..prefix.len() * (k + 1) / 16];
        partition.index.apply(
            holder.lane().1,
            &mut held.iter().copied().map(Op::Store).collect(),
        );
    }
    let lanes: Vec<_> = holders.iter().map(|holder| Some(holder.lane())).collect();
    let samples = (0..2000)
        .map(|_| {
            let started = Instant::now();
            let hits = match_holders(&tokens, &lanes);
            assert_eq!(hits[15], Some(100_000));
            started.elapsed()
        })
        .collect();
    println!(
        "match_workers with hashing, 16 workers: {}",
        percentiles(samples)
    );
}

/// Both sides of the shard locks: a prefill step's store of 512 blocks applied to a
/// 1M-entry index, alone and while walkers query a 100k-token prompt without pause, and
/// those walks meanwhile. `KV_BENCH_WALKERS` (0,4) lists the walker counts.
#[test]
#[ignore]
fn bench_writer_with_walkers() {
    let index = PrefixIndex::default();
    let (holders, _, prefix) = filled(&index);
    for holder in &holders {
        index.apply(
            *holder,
            &mut prefix.iter().copied().map(Op::Store).collect(),
        );
    }
    let counts = std::env::var("KV_BENCH_WALKERS").map_or(vec![0, 4], |list| {
        list.split(',')
            .map(|walkers| walkers.parse().unwrap())
            .collect()
    });
    for walkers in counts {
        let stop = AtomicBool::new(false);
        let walks = Mutex::new(Vec::new());
        std::thread::scope(|scope| {
            for _ in 0..walkers {
                scope.spawn(|| {
                    let mut samples = Vec::new();
                    while !stop.load(Ordering::Relaxed) {
                        let started = Instant::now();
                        drop(index.find_matches(&prefix, &holders));
                        samples.push(started.elapsed());
                    }
                    walks.lock().extend(samples);
                });
            }
            let samples = (0..500u32)
                .map(|round| {
                    let seq: Vec<u32> = (0..512 * 16).map(|i| round << 20 | i).collect();
                    let stores = hash::prefix_hashes(&seq, 16);
                    let started = Instant::now();
                    index.apply(
                        HolderId(16),
                        &mut stores.iter().copied().map(Op::Store).collect(),
                    );
                    let elapsed = started.elapsed();
                    index.apply(
                        HolderId(16),
                        &mut stores.into_iter().map(Op::Remove).collect(),
                    );
                    elapsed
                })
                .collect();
            stop.store(true, Ordering::Relaxed);
            println!(
                "apply 512 stores, {walkers} walkers: {}",
                percentiles(samples)
            );
        });
        if walkers > 0 {
            println!("walks meanwhile: {}", percentiles(walks.into_inner()));
        }
    }
}
