use std::{
    collections::HashMap,
    time::{Duration, Instant},
};

use xxhash_rust::xxh3::xxh3_64_with_seed;

use super::*;
use crate::kv_events::{
    DropReason, EngineHash, Group, GroupKind, KvEvent, Namespace, StoredBlock, Tier,
};

const TIERS: [Tier; 3] = [Tier::Device, Tier::Host, Tier::Disk];

/// Golden values from python-xxhash's `xxh3_64_intdigest` with SMG's formulas.
#[test]
fn hashes_follow_smg() {
    let tokens: Vec<u32> = (1..=9).collect();
    let plain = hash::prefix_hashes(&tokens, 4, &Namespace::default());
    assert_eq!(
        plain,
        [
            PrefixHash(0xcb38e4d7fb2dca5c),
            PrefixHash(0x44a2b2379d6d3c3d)
        ]
    );
    let ns = Namespace {
        lora: Some("adapter".into()),
        salt: Some("salty".into()),
        opaque: false,
    };
    assert_eq!(hash::namespace_seed(&ns), 0x6afe0c57a29d587c);
    let namespaced = hash::prefix_hashes(&tokens, 4, &ns);
    assert_eq!(
        namespaced,
        [
            PrefixHash(0x6491b83975a725f1),
            PrefixHash(0x8ba69fcccbbce881)
        ]
    );
    let empty = Namespace {
        lora: Some("".into()),
        ..Namespace::default()
    };
    assert_eq!(hash::namespace_seed(&empty), hash::XXH3_SEED);
    let opaque = Namespace {
        opaque: true,
        ..Namespace::default()
    };
    assert!(hash::prefix_hashes(&tokens, 4, &opaque).is_empty());
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

/// An engine's own block hash: chained like the router's, but a different function.
fn engine_hash(holder: usize, prefix: &[u32]) -> EngineHash {
    let bytes: Vec<u8> = prefix
        .iter()
        .flat_map(|token| token.to_le_bytes())
        .collect();
    EngineHash(xxh3_64_with_seed(&bytes, 7 + holder as u64))
}

/// The naive reference: the token prefixes one holder holds, per tier.
type Held = HashMap<Vec<u32>, [bool; 3]>;

fn expected(held: &Held, seq: &[u32], block_size: usize) -> ([u32; 3], u32) {
    let mut open = [true; 4];
    let mut depth = [0; 4];
    for end in (block_size..=seq.len()).step_by(block_size) {
        let [device, host, disk] = held.get(&seq[..end]).copied().unwrap_or_default();
        let present = [device, device || host, device || host || disk, host || disk];
        for (column, present) in present.into_iter().enumerate() {
            open[column] &= present;
            if open[column] {
                depth[column] = (end / block_size) as u32;
            }
        }
    }
    ([depth[0], depth[1], depth[2]], depth[3])
}

fn stored(
    holder: usize,
    seq: &[u32],
    blocks: std::ops::Range<usize>,
    bs: usize,
    tier: Tier,
    tokens: bool,
) -> KvEvent {
    KvEvent::Stored {
        parent: (blocks.start > 0).then(|| engine_hash(holder, &seq[..blocks.start * bs])),
        blocks: blocks
            .map(|b| StoredBlock {
                hash: engine_hash(holder, &seq[..(b + 1) * bs]),
                tokens: tokens.then(|| seq[b * bs..(b + 1) * bs].into()),
            })
            .collect(),
        block_size: bs as u32,
        tier,
        group: Group::default(),
        ns: Namespace::default(),
    }
}

/// Random stores (with and without tokens), evictions anywhere in a chain and clears over
/// three holders sharing prefixes; after every step each holder's match of every sequence
/// equals the reference.
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
        let mut held: Vec<Held> = vec![Held::new(); HOLDERS];
        let holders: Vec<HolderId> = (0..HOLDERS as u32).map(HolderId).collect();
        let mut deepest = [0; 4];
        for _ in 0..2000 {
            let h = rng.below(HOLDERS);
            let seq = &seqs[rng.below(seqs.len())];
            let tier = rng.below(3);
            let event = match rng.below(20) {
                0 => {
                    let tier = (rng.below(2) == 0).then_some(tier);
                    held[h].retain(|_, tiers| {
                        for (t, flag) in tiers.iter_mut().enumerate() {
                            *flag &= tier.is_some_and(|tier| tier != t);
                        }
                        tiers.contains(&true)
                    });
                    KvEvent::Cleared {
                        tier: tier.map(|t| TIERS[t]),
                    }
                }
                1..=8 => {
                    let mut keys: Vec<&Vec<u32>> = held[h]
                        .iter()
                        .filter(|(_, tiers)| tiers[tier])
                        .map(|(key, _)| key)
                        .collect();
                    keys.sort();
                    let Some(key) = keys
                        .get(rng.below(keys.len().max(1)))
                        .map(|key| (*key).clone())
                    else {
                        continue;
                    };
                    let tiers = held[h].get_mut(&key).unwrap();
                    tiers[tier] = false;
                    if !tiers.contains(&true) {
                        held[h].remove(&key);
                    }
                    KvEvent::Removed {
                        hashes: vec![engine_hash(h, &key)],
                        tier: TIERS[tier],
                        group: Group::default(),
                    }
                }
                _ => {
                    // An engine names a parent it holds, and copies to a lower tier without
                    // tokens only blocks it holds.
                    let known = |b: usize| held[h].contains_key(&seq[..(b + 1) * bs]);
                    let start = rng.below(BLOCKS);
                    let tokens = tier == 0 || rng.below(2) == 0;
                    if start > 0 && !known(start - 1) {
                        continue;
                    }
                    let mut end = start;
                    while end < BLOCKS.min(start + 4)
                        && !held[h]
                            .get(&seq[..(end + 1) * bs])
                            .is_some_and(|tiers| tiers[tier])
                        && (tokens || known(end))
                    {
                        end += 1;
                    }
                    if end == start {
                        continue;
                    }
                    for b in start..end {
                        held[h].entry(seq[..(b + 1) * bs].to_vec()).or_default()[tier] = true;
                    }
                    stored(h, seq, start..end, bs, TIERS[tier], tokens)
                }
            };
            tables[h].apply(&index, holders[h], &event).unwrap();
            for seq in &seqs {
                let prefix = hash::prefix_hashes(seq, bs, &Namespace::default());
                let found = index.find_matches(&MatchQuery {
                    prefix: &prefix,
                    holders: &holders,
                });
                for (h, held) in held.iter().enumerate() {
                    let actual = found
                        .iter()
                        .find(|m| m.holder == holders[h])
                        .map_or(([0; 3], 0), |m| (m.blocks, m.offload));
                    assert_eq!(
                        actual,
                        expected(held, seq, bs),
                        "block size {bs}, holder {h}"
                    );
                    let ([device, host, disk], offload) = actual;
                    for (deepest, depth) in deepest.iter_mut().zip([device, host, disk, offload]) {
                        *deepest = (*deepest).max(depth);
                    }
                }
            }
        }
        assert!(deepest.iter().all(|&depth| depth > 1), "{deepest:?}");
        let stats = index.stats();
        for (t, blocks) in stats.blocks.iter().enumerate() {
            let reference: usize = held
                .iter()
                .map(|held| held.values().filter(|tiers| tiers[t]).count())
                .sum();
            assert_eq!(*blocks, reference as u64);
        }
        index.remove_holder(holders[0]);
        let prefix = hash::prefix_hashes(&seqs[0], bs, &Namespace::default());
        let found = index.find_matches(&MatchQuery {
            prefix: &prefix,
            holders: &holders[..1],
        });
        assert!(found.is_empty());
    }
}

#[test]
fn concurrent_walks_and_writes_finish() {
    let index = PrefixIndex::default();
    let tokens: Vec<u32> = (0..1024).collect();
    let prefix = hash::prefix_hashes(&tokens, 1, &Namespace::default());
    let holders: Vec<HolderId> = (0..8).map(HolderId).collect();
    let ops: Vec<IndexOp> = prefix
        .iter()
        .map(|&hash| IndexOp::Store {
            hash,
            tier: Tier::Device,
        })
        .collect();
    std::thread::scope(|scope| {
        for writer in 0..2 {
            let (index, ops) = (&index, &ops);
            scope.spawn(move || {
                for round in 0..100 {
                    let holder = HolderId(writer * 4 + round % 4);
                    index.apply(holder, ops);
                    index.clear(holder, None);
                }
            });
        }
        for _ in 0..4 {
            let query = MatchQuery {
                prefix: &prefix,
                holders: &holders,
            };
            let index = &index;
            scope.spawn(move || (0..100).for_each(|_| drop(index.find_matches(&query))));
        }
    });
}

#[test]
fn token_less_blocks_need_a_known_engine_hash() {
    let index = PrefixIndex::default();
    let mut table = HolderTable::default();
    let seq: Vec<u32> = (0..8).collect();
    let lower = stored(0, &seq, 0..2, 4, Tier::Host, false);
    assert_eq!(
        table.apply(&index, HolderId(0), &lower),
        Err(DropReason::Unresolved)
    );
    let child = stored(0, &seq, 1..2, 4, Tier::Device, true);
    assert_eq!(
        table.apply(&index, HolderId(0), &child),
        Err(DropReason::UnknownParent)
    );
    table
        .apply(
            &index,
            HolderId(0),
            &stored(0, &seq, 0..2, 4, Tier::Device, true),
        )
        .unwrap();
    table.apply(&index, HolderId(0), &lower).unwrap();
    assert_eq!(index.stats().blocks, [2, 2, 0]);
}

/// vLLM stores a hybrid model's blocks once per group but removes them by group alone;
/// `kv_cache_report_mode: "full"` re-reports held blocks as stored.
#[test]
fn holder_tables_skip_other_groups_cap_copies_and_drop_opaque_blocks() {
    let (index, mut table, id) = (PrefixIndex::default(), HolderTable::default(), HolderId(0));
    let seq: Vec<u32> = (0..8).collect();
    let full = stored(0, &seq, 0..2, 4, Tier::Device, true);
    let (mut window, mut opaque) = (full.clone(), full.clone());
    if let (KvEvent::Stored { group, .. }, KvEvent::Stored { ns, .. }) = (&mut window, &mut opaque)
    {
        (group.idx, group.kind) = (1, GroupKind::SlidingWindow(128));
        ns.opaque = true;
    }
    let removed = |idx| KvEvent::Removed {
        hashes: vec![engine_hash(0, &seq[..4])],
        tier: Tier::Device,
        group: Group {
            idx,
            kind: GroupKind::Full,
        },
    };
    // Event, times applied, Device blocks held after.
    for (event, times, device) in [
        (&full, 1, 2),
        (&window, 1, 2),
        (&removed(1), 1, 2),
        (&full, 20, 2),
        (&removed(0), 7, 2),
        (&removed(0), 1, 1),
    ] {
        (0..times).for_each(|_| table.apply(&index, id, event).unwrap());
        assert_eq!(index.stats().blocks[0], device);
    }
    assert_eq!(table.apply(&index, id, &opaque), Err(DropReason::Namespace));
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

/// `cargo test --release --lib kv_index::tests::bench -- --ignored --nocapture --test-threads=1`
#[test]
#[ignore]
fn bench_query_16_workers_100k_tokens() {
    const BS: usize = 16;
    let index = PrefixIndex::default();
    let holders: Vec<HolderId> = (0..16).map(HolderId).collect();
    // Other prefixes fill the index to 1M entries.
    let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
    for i in 0..1_000_000u32 {
        let op = IndexOp::Store {
            hash: PrefixHash(rng.below(usize::MAX) as u64),
            tier: Tier::Device,
        };
        index.apply(holders[i as usize % 16], &[op]);
    }
    let tokens: Vec<u32> = (0..100_000).map(|i| (i * 7919 % 150_000) as u32).collect();
    let (ns, started) = (Namespace::default(), Instant::now());
    let prefix = hash::prefix_hashes(&tokens, BS, &ns);
    println!("prefix hashes of 100k tokens: {:?}", started.elapsed());
    let mut held = [0; 16];
    for (name, full) in [
        ("staggered, worker k holds (k+1)/16 of the prefix", false),
        ("worst, all 16 hold all of it", true),
    ] {
        for (k, holder) in holders.iter().enumerate() {
            let target = if full {
                prefix.len()
            } else {
                prefix.len() * (k + 1) / 16
            };
            let ops: Vec<IndexOp> = prefix[held[k]..target]
                .iter()
                .map(|&hash| IndexOp::Store {
                    hash,
                    tier: Tier::Device,
                })
                .collect();
            index.apply(*holder, &ops);
            held[k] = target;
        }
        for hashing in [false, true] {
            let mut samples: Vec<Duration> = (0..2000)
                .map(|_| {
                    let started = Instant::now();
                    let fresh = hashing.then(|| hash::prefix_hashes(&tokens, BS, &ns));
                    let found = index.find_matches(&MatchQuery {
                        prefix: fresh.as_deref().unwrap_or(&prefix),
                        holders: &holders,
                    });
                    assert_eq!(found.len(), 16);
                    started.elapsed()
                })
                .collect();
            samples.sort();
            let (p50, p99) = (samples[1000], samples[1980]);
            println!("{name}, hashing {hashing}: p50 {p50:?}, p99 {p99:?}");
        }
    }
}

#[test]
#[ignore]
fn bench_memory_1m_blocks() {
    let before = resident_bytes();
    let index = PrefixIndex::default();
    let mut tables: Vec<HolderTable> = (0..16).map(|_| HolderTable::default()).collect();
    // 16 holders, chains of 64 blocks of 16 tokens, 1M blocks in all.
    for chain in 0..15_625usize {
        let h = chain % 16;
        let seq: Vec<u32> = (0..64 * 16).map(|i| (chain * 1024 + i) as u32).collect();
        let event = stored(h, &seq, 0..64, 16, Tier::Device, true);
        tables[h].apply(&index, HolderId(h as u32), &event).unwrap();
    }
    let after = resident_bytes();
    let stats = index.stats();
    println!(
        "{} entries, {} blocks: resident +{} MB (index estimate {} MB)",
        stats.entries,
        stats.blocks[0],
        (after - before) >> 20,
        stats.approx_bytes >> 20
    );
    assert!(after - before < 200 << 20);
}
