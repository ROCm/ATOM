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
    match_lanes, HolderId, Lane, Partition, PrefixHash,
};
use crate::{
    core::{BasicWorkerBuilder, Worker, WorkerLoadGuard},
    kv_events::{
        model::{DropReason, EngineHash, KvEvent, StoredBlock},
        SourceStatus,
    },
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

/// Walks without pause starve no writer, and two writers deadlock with no walk.
#[test]
fn concurrent_walks_and_writes_finish() {
    let index = PrefixIndex::default();
    let tokens: Vec<u32> = (0..4096).collect();
    let prefix = hash::prefix_hashes(&tokens, 1);
    let holders: Vec<HolderId> = (0..8).map(HolderId).collect();
    let written = AtomicBool::new(false);
    std::thread::scope(|scope| {
        for _ in 0..4 {
            let (index, prefix, holders, written) = (&index, &prefix, &holders, &written);
            scope.spawn(move || {
                while !written.load(Ordering::Relaxed) {
                    drop(index.find_matches(prefix, holders));
                }
            });
        }
        let writers: Vec<_> = (0..2u8)
            .map(|writer| {
                let (index, prefix) = (&index, &prefix);
                scope.spawn(move || {
                    for round in 0..100u8 {
                        let holder = HolderId(writer * 4 + round % 4);
                        for op in [Op::Store, Op::Remove] {
                            index.apply(holder, &mut prefix.iter().copied().map(op).collect());
                        }
                    }
                })
            })
            .collect();
        for writer in writers {
            writer.join().unwrap();
        }
        written.store(true, Ordering::Relaxed);
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

type Lanes = [Option<Lane>];

fn match_holders(tokens: &[u32], lanes: &Lanes) -> Vec<Option<u32>> {
    match_lanes(tokens, lanes, &[])
}

fn match_predicted(tokens: &[u32], lanes: &Lanes) -> Vec<Option<u32>> {
    match_lanes(tokens, &[], lanes)
}

async fn settles(
    matches: fn(&[u32], &Lanes) -> Vec<Option<u32>>,
    tokens: &[u32],
    lanes: &Lanes,
    want: &[Option<u32>],
) {
    for _ in 0..200 {
        if matches(tokens, lanes) == want {
            return;
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    panic!("{:?}, not {want:?}", matches(tokens, lanes));
}

/// Batches reach the index through the partition's writer, in order, a reset clearing
/// the rank first. Lanes run out at 64; one given back is free once its blocks are gone.
#[tokio::test]
async fn the_writer_applies_batches_and_frees_released_lanes() {
    let ttl = Duration::from_secs(120);
    let partition = Partition::start(4, ttl, &tokio::runtime::Handle::current());
    let status = Arc::new(Mutex::new(SourceStatus::default()));
    let mut holders: Vec<_> = (0..64)
        .map(|_| partition.hold(status.clone()).unwrap())
        .collect();
    assert!(partition.hold(status.clone()).is_none());
    // Predictions take lanes of their own.
    let predictors: Vec<_> = (0..64).map(|_| partition.predict("w").unwrap()).collect();
    assert!(partition.predict("w").is_none());
    drop(predictors);
    let seq: Vec<u32> = (0..10).collect();
    let lanes: Vec<_> = holders[..2].iter().map(|h| Some(h.lane())).collect();
    holders[0]
        .write(false, vec![stored(0, &seq, 0..2, 4)])
        .await;
    holders[1]
        .write(false, vec![stored(1, &seq, 0..1, 4)])
        .await;
    settles(match_holders, &seq, &lanes, &[Some(8), Some(4)]).await;
    assert_eq!(status.lock().indexed_blocks, Some(1));
    holders[0].write(true, vec![stored(0, &seq, 0..1, 4)]).await;
    settles(match_holders, &seq, &lanes, &[Some(4), Some(4)]).await;
    let first = KvEvent::Removed {
        hashes: vec![engine_hash(0, &seq[..4])],
    };
    holders[0]
        .write(false, vec![first, stored(0, &seq, 1..2, 4)])
        .await;
    settles(match_holders, &seq, &lanes, &[Some(0), Some(4)]).await;
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

/// A prediction holds the full blocks of each prompt routed, each until a TTL after its
/// last routing, apart from the events. A lane given back is free once they are gone.
#[tokio::test]
async fn predictions_last_a_ttl_from_their_last_routing() {
    let ttl = Duration::from_millis(800);
    let partition = Partition::start(4, ttl, &tokio::runtime::Handle::current());
    let status = Arc::new(Mutex::new(SourceStatus::default()));
    let holder = partition.hold(status).unwrap();
    let predictor = partition.predict("w").unwrap();
    let lanes = [Some(predictor.lane())];
    let seq: Vec<u32> = (0..10).collect();
    predictor.record(&seq[..8]);
    settles(match_predicted, &seq, &lanes, &[Some(8)]).await;
    assert_eq!(match_holders(&seq, &[Some(holder.lane())]), [Some(0)]);
    tokio::time::sleep(ttl * 3 / 4).await;
    predictor.record(&seq[..7]);
    // The second block expires a TTL after the first routing, the first after the second.
    settles(match_predicted, &seq, &lanes, &[Some(4)]).await;
    settles(match_predicted, &seq, &lanes, &[Some(0)]).await;

    predictor.record(&seq);
    settles(match_predicted, &seq, &lanes, &[Some(8)]).await;
    drop(predictor);
    let predictor = loop {
        if let Some(predictor) = partition.predict("w") {
            break predictor;
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    };
    assert_eq!(match_predicted(&seq, &[Some(predictor.lane())]), [Some(0)]);
}

/// A worker holds the more of what its rank's events and its predictions hold: the
/// predictions go on where the events end, and alone stand in for missing events.
#[tokio::test]
async fn predictions_go_on_where_the_events_end() {
    let ttl = Duration::from_secs(120);
    let partition = Partition::start(4, ttl, &tokio::runtime::Handle::current());
    let status = Arc::new(Mutex::new(SourceStatus::default()));
    let seq: Vec<u32> = (0..17).collect();
    let prefix = hash::prefix_hashes(&seq[..16], 4);
    // Blocks of the events and of the predictions, by worker; `None` without a lane.
    let held = [
        (Some(2), Some(4)),
        (Some(3), Some(1)),
        (None, Some(2)),
        (Some(1), None),
    ];
    let holders: Vec<_> = held
        .iter()
        .map(|_| partition.hold(status.clone()).unwrap())
        .collect();
    let predictors: Vec<_> = held
        .iter()
        .map(|_| partition.predict("w").unwrap())
        .collect();
    let store = |index: &PrefixIndex, lane: Lane, blocks: usize| {
        index.apply(
            lane.1,
            &mut prefix[..blocks].iter().copied().map(Op::Store).collect(),
        );
        lane
    };
    let (events, predicted): (Vec<_>, Vec<_>) = held
        .iter()
        .zip(holders.iter().zip(&predictors))
        .map(|(&(on_events, on_predicted), (holder, predictor))| {
            (
                on_events.map(|blocks| store(&partition.index, holder.lane(), blocks)),
                on_predicted.map(|blocks| store(&partition.predicted, predictor.lane(), blocks)),
            )
        })
        .unzip();
    let want = [Some(16), Some(12), Some(8), Some(4)];
    assert_eq!(match_lanes(&seq, &events, &predicted), want);
}

/// A guard charging `tokens` of prefill to a worker of its own.
fn charged(tokens: usize) -> (Arc<dyn Worker>, WorkerLoadGuard) {
    let worker: Arc<dyn Worker> = Arc::new(BasicWorkerBuilder::new("http://w:1").build());
    let guard = WorkerLoadGuard::new(worker.clone(), None).with_prefill_tokens(tokens);
    (worker, guard)
}

async fn until(done: impl Fn() -> bool) {
    for _ in 0..200 {
        if done() {
            return;
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    panic!("timed out");
}

/// Identical prompts on one rank wait for one key: each store of the last full block
/// releases the oldest charge still held, skipping one its first byte released. A store
/// on another lane, or a prompt shorter than a block, releases nothing.
#[tokio::test]
async fn each_stored_block_releases_the_oldest_held_prefill() {
    let partition = Partition::start(
        4,
        Duration::from_secs(120),
        &tokio::runtime::Handle::current(),
    );
    let status = Arc::new(Mutex::new(SourceStatus::default()));
    let holders: Vec<_> = (0..2)
        .map(|_| partition.hold(status.clone()).unwrap())
        .collect();
    let lane = holders[0].lane().1;
    let seq: Vec<u32> = (0..8).collect();
    let charges: Vec<_> = (0..3).map(|_| charged(8)).collect();
    for (_, guard) in &charges {
        partition.watch_prefill(lane, &seq, &guard.prefill_charge());
    }
    let (short, short_guard) = charged(3);
    partition.watch_prefill(lane, &seq[..3], &short_guard.prefill_charge());
    let held = || {
        charges
            .iter()
            .map(|(w, _)| w.prefill_tokens())
            .collect::<Vec<_>>()
    };

    holders[1]
        .write(false, vec![stored(1, &seq, 0..2, 4)])
        .await;
    let other = [Some(holders[1].lane())];
    // The match leaves out the prompt's last token, so one more matches both blocks.
    let probe: Vec<u32> = (0..9).collect();
    settles(match_holders, &probe, &other, &[Some(8)]).await;
    assert_eq!(held(), [8, 8, 8]);
    charges[1].1.prefill_charge().release();
    holders[0]
        .write(false, vec![stored(0, &seq, 0..2, 4)])
        .await;
    until(|| held() == [0, 0, 8]).await;
    // Stored again by a second prefill, while the rank still holds it.
    holders[0]
        .write(false, vec![stored(0, &seq, 1..2, 4)])
        .await;
    until(|| held() == [0, 0, 0]).await;
    assert_eq!(short.prefill_tokens(), 3);
    assert!(partition.watches.lock().is_empty());
}

/// A lane given back drops its watches, so the rank given it next releases none of them;
/// a watch whose charge is gone leaves at the next sweep.
#[tokio::test]
async fn a_released_lane_and_a_sweep_drop_prefill_watches() {
    let partition = Partition::start(
        4,
        Duration::from_millis(800),
        &tokio::runtime::Handle::current(),
    );
    let status = Arc::new(Mutex::new(SourceStatus::default()));
    let holder = partition.hold(status.clone()).unwrap();
    let seq: Vec<u32> = (0..8).collect();
    let lane = holder.lane().1;
    let (worker, guard) = charged(8);
    partition.watch_prefill(lane, &seq, &guard.prefill_charge());
    drop(holder);
    // The lowest free lane is taken, so the released one once the writer frees it.
    let holder = loop {
        let holder = partition.hold(status.clone()).unwrap();
        if holder.lane().1 == lane {
            break holder;
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    };
    assert!(partition.watches.lock().is_empty());
    holder.write(false, vec![stored(0, &seq, 0..2, 4)]).await;
    let probe: Vec<u32> = (0..9).collect();
    settles(match_holders, &probe, &[Some(holder.lane())], &[Some(8)]).await;
    assert_eq!(worker.prefill_tokens(), 8);

    let (_, dropped) = charged(8);
    partition.watch_prefill(holder.lane().1, &seq[..4], &dropped.prefill_charge());
    drop(dropped);
    until(|| partition.watches.lock().is_empty()).await;
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
        applying / blocks
    );
    assert!(after - before < (200 << 20) * wanted as u64 / 1_000_000);
}

/// What `cache_aware` asks, hashing the request included: 16 workers, worker k holding
/// (k+1)/16 of a 100k-token prompt, in one partition with 1M other entries. `filled` reads
/// the sizes, and `KV_BENCH_SAMPLES` (2000) the samples.
#[tokio::test]
#[ignore]
async fn bench_match_workers_16_workers() {
    let ttl = Duration::from_secs(120);
    let partition = Partition::start(16, ttl, &tokio::runtime::Handle::current());
    let (background, tokens, prefix) = filled(&partition.index);
    let (workers, count) = (background.len(), size("KV_BENCH_SAMPLES", 2000));
    let status = Arc::new(Mutex::new(SourceStatus::default()));
    let holders: Vec<_> = (0..workers)
        .map(|_| partition.hold(status.clone()).unwrap())
        .collect();
    for (k, holder) in holders.iter().enumerate() {
        let held = &prefix[..prefix.len() * (k + 1) / workers];
        partition.index.apply(
            holder.lane().1,
            &mut held.iter().copied().map(Op::Store).collect(),
        );
    }
    let lanes: Vec<_> = holders.iter().map(|holder| Some(holder.lane())).collect();
    // Short of the block of the prompt's last token.
    let all = Some(((tokens.len() - 1) / 16 * 16) as u32);
    let samples = (0..count)
        .map(|_| {
            let started = Instant::now();
            let hits = match_holders(&tokens, &lanes);
            assert_eq!(hits[workers - 1], all);
            started.elapsed()
        })
        .collect();
    println!(
        "match_workers with hashing, {workers} workers: {}",
        percentiles(samples)
    );

    // Every rank's events partial, and each worker predicted to hold its share too.
    let predictors: Vec<_> = (0..workers)
        .map(|_| partition.predict("w").unwrap())
        .collect();
    for (k, predictor) in predictors.iter().enumerate() {
        let held = &prefix[..prefix.len() * (k + 1) / workers];
        partition.predicted.apply(
            predictor.lane().1,
            &mut held.iter().copied().map(Op::Store).collect(),
        );
    }
    let predicted: Vec<_> = predictors.iter().map(|p| Some(p.lane())).collect();
    let samples = (0..count)
        .map(|_| {
            let started = Instant::now();
            let hits = match_lanes(&tokens, &lanes, &predicted);
            assert_eq!(hits[workers - 1], all);
            started.elapsed()
        })
        .collect();
    println!(
        "match_workers with hashing, {workers} partial workers with predictions: {}",
        percentiles(samples)
    );
}

/// One routing of a 100k-token prompt, routed again and again: the request's time, each
/// once the writer is done with the one before, and the writer's, hashing included, for
/// routings queued at once. `KV_BENCH_TOKENS` sizes the prompt.
#[tokio::test]
#[ignore]
async fn bench_predict_100k_tokens() {
    const ROUTINGS: u32 = 32;
    let ttl = Duration::from_secs(120);
    let partition = Partition::start(16, ttl, &tokio::runtime::Handle::current());
    let predictor = partition.predict("w").unwrap();
    let lanes = [Some(predictor.lane())];
    let prompt: Vec<u32> = (0..size("KV_BENCH_TOKENS", 100_000) as u32).collect();
    let mut request = Duration::ZERO;
    for round in 0..=ROUTINGS {
        let last: Vec<u32> = (0..17).map(|i| (round << 20) | i).collect();
        let started = Instant::now();
        predictor.record(&prompt);
        // The first only warms the allocator up.
        if round > 0 {
            request += started.elapsed();
        }
        predictor.record(&last);
        while match_predicted(&last, &lanes) != [Some(16)] {
            tokio::task::yield_now().await;
        }
    }
    let last: Vec<u32> = (0..17).rev().collect();
    let started = Instant::now();
    for _ in 1..ROUTINGS {
        predictor.record(&prompt);
    }
    predictor.record(&last);
    let queued = started.elapsed();
    while match_predicted(&last, &lanes) != [Some(16)] {
        tokio::task::yield_now().await;
    }
    println!(
        "predict, {} tokens: request {:?}, writer {:?}",
        prompt.len(),
        request / ROUTINGS,
        (started.elapsed() - queued) / ROUTINGS
    );
}

/// Both sides of the shard locks: a prefill step's store of 512 blocks applied to a
/// 1M-entry index, alone and while walkers query a 100k-token prompt without pause, and
/// those walks, alone and meanwhile. `KV_BENCH_WALKERS` (0,4) lists the walker counts;
/// `KV_BENCH_WRITE_EVERY_US` (0) paces the writer, which otherwise writes without pause.
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
    // The walks of `walkers` walkers while `work` runs.
    let walking = |walkers: usize, work: &mut dyn FnMut()| {
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
            work();
            stop.store(true, Ordering::Relaxed);
        });
        walks.into_inner()
    };
    for walkers in counts {
        if walkers > 0 {
            let walks = walking(walkers, &mut || {
                std::thread::sleep(Duration::from_millis(500))
            });
            println!("walks alone, {walkers} walkers: {}", percentiles(walks));
        }
        let mut samples = Vec::new();
        let pause = Duration::from_micros(size("KV_BENCH_WRITE_EVERY_US", 0) as u64);
        let walks = walking(walkers, &mut || {
            samples = (0..500u32)
                .map(|round| {
                    std::thread::sleep(pause);
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
        });
        println!(
            "apply 512 stores, {walkers} walkers: {}",
            percentiles(samples)
        );
        if walkers > 0 {
            println!("walks meanwhile: {}", percentiles(walks));
        }
    }
}
