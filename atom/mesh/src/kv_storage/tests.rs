use std::time::Duration;

use serde::Deserialize;

use super::{
    keys::{chain_check, STORE_TAINT_KEY},
    *,
};

const VECTORS: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/kv_storage/atom_keys.jsonl"
));
const NS: &str = "atomkv1-b3fd64e0ee0549784107510e";

#[derive(Deserialize)]
struct Vector {
    name: String,
    len: u64,
    mul: u64,
    add: u64,
    vocab: u64,
    chunk_tokens: u32,
    namespace: String,
    world: u32,
    rank: u32,
    keys: Vec<String>,
    chain_check: String,
}

#[test]
fn atom_chunk_keys_match_atom_key_vectors() {
    for line in VECTORS.lines() {
        let case: Vector = serde_json::from_str(line).unwrap();
        let tokens: Vec<u32> = (0..case.len)
            .map(|i| ((case.mul * i + case.add) % case.vocab) as u32)
            .collect();
        let rule = AtomChunk {
            namespace: case.namespace.clone(),
            chunk_tokens: case.chunk_tokens,
            world: case.world,
            probe: RankProbe::All,
        };
        let keys: Vec<String> = rule
            .keys(&tokens)
            .into_iter()
            .skip(case.rank as usize)
            .step_by(case.world as usize)
            .collect();
        assert_eq!(keys, case.keys, "{}", case.name);
        assert_eq!(
            chain_check(case.chunk_tokens),
            case.chain_check,
            "{}",
            case.name
        );
        if case.rank == 0 {
            let first = AtomChunk {
                probe: RankProbe::First,
                ..rule
            };
            assert_eq!(first.keys(&tokens), case.keys, "{}", case.name);
        }
    }
}

/// World 2, 4-token chunks.
fn rule(probe: RankProbe) -> AtomChunk {
    AtomChunk {
        namespace: NS.into(),
        chunk_tokens: 4,
        world: 2,
        probe,
    }
}

/// Pools separated by spaces: `M`, `S`, `-` or `?` per chunk for every rank's object in
/// memory, on SSD only, absent or not answered for; `x` is a pool that fails, and a
/// leading `t` marks a pool holding the taint marker.
fn pools(tokens: &[u32], spec: &str) -> Vec<MockPool> {
    let keys = rule(RankProbe::All).keys(tokens);
    spec.split(' ')
        .map(|spec| {
            if spec == "x" {
                return MockPool {
                    fail: Some(FailReason::Transport),
                    ..MockPool::default()
                };
            }
            let mut pool = MockPool::default();
            let states = spec.trim_start_matches('t').chars();
            for (ranks, state) in keys.chunks(2).zip(states) {
                let state = match state {
                    'M' => KeyState::Memory,
                    'S' => KeyState::SsdOnly,
                    '?' => KeyState::Unknown,
                    _ => KeyState::Absent,
                };
                for key in ranks {
                    pool.keys.insert(key.clone(), state);
                }
            }
            if spec.starts_with('t') {
                pool.keys.insert(STORE_TAINT_KEY.into(), KeyState::Memory);
            }
            pool
        })
        .collect()
}

fn hit(shared: u32, disk: u32) -> StorageLookup {
    StorageLookup::Hit { shared, disk }
}

async fn fold(rule: &AtomChunk, store: &MockStore, tokens: &[u32]) -> StorageLookup {
    match store.query(&rule.keys(tokens)).await {
        Ok(states) => rule.fold(&states),
        Err(reason) => StorageLookup::Unavailable(reason),
    }
}

#[test]
fn ranks_merge_to_the_slowest_and_pools_answering_other_counts_fail() {
    use KeyState::*;
    // A chunk is as slow as its slowest rank, and missing if any rank misses it.
    let lookup = rule(RankProbe::All).fold(&[Memory, SsdOnly, Absent, Unknown]);
    assert_eq!(lookup, hit(0, 4));
    let answer = |states: Vec<KeyState>| {
        Ok(PoolAnswer {
            states,
            tainted: false,
        })
    };
    let short = answer(vec![Memory]);
    let full = answer(vec![SsdOnly; 2]);
    assert_eq!(merge_pools(2, &[short.clone(), full]), Ok(vec![SsdOnly; 2]));
    assert_eq!(merge_pools(2, &[short]), Err(FailReason::BadResponse));
}

#[tokio::test]
async fn mock_pools_fold_to_leading_hit_tokens() {
    let tokens: Vec<u32> = (0..24).collect();
    let unknown = StorageLookup::Unavailable(FailReason::Transport);
    let cases = [
        ("MMMSS", hit(12, 20)),
        // Memory in any pool wins, then SSD in any pool.
        ("MM SSSS", hit(8, 16)),
        // As in ATOM, a tainted pool holds nothing and a failed pool is left out; a pool
        // that answered for a key beats one that could not.
        ("tMMMMMM SS", hit(0, 8)),
        ("tMM x", hit(0, 0)),
        ("MM x", hit(8, 8)),
        ("MM MM?M", hit(8, 8)),
        // A chunk no pool answered for decides the hit only before the first miss.
        ("MM?M", unknown),
        ("MM-?", hit(8, 8)),
        ("x x", unknown),
    ];
    for (spec, expected) in cases {
        let store = MockStore::new(pools(&tokens, spec));
        let lookup = fold(&rule(RankProbe::First), &store, &tokens).await;
        assert_eq!(lookup, expected, "{spec}");
    }

    // Rank 1 lacks chunk 2: only probing every rank sees it.
    let store = MockStore::new(pools(&tokens, "MMMMMM"));
    store.pools.lock()[0]
        .keys
        .remove(&rule(RankProbe::All).keys(&tokens)[2 * 2 + 1]);
    assert_eq!(
        fold(&rule(RankProbe::First), &store, &tokens).await,
        hit(24, 24)
    );
    assert_eq!(
        fold(&rule(RankProbe::All), &store, &tokens).await,
        hit(8, 8)
    );
    assert_eq!(MockStore::new(pools(&tokens, "tMM SS")).tainted_pools(), 1);
}

#[test]
#[ignore]
fn bench_atom_chunk_keys_100k_tokens() {
    let tokens: Vec<u32> = (0..100_000u32)
        .map(|i| i.wrapping_mul(2_654_435_761) % 151_936)
        .collect();
    let time = |run: &dyn Fn()| {
        let mut samples: Vec<Duration> = (0..1000)
            .map(|_| {
                let start = std::time::Instant::now();
                run();
                start.elapsed()
            })
            .collect();
        samples.sort();
        (samples[500], samples[990])
    };
    let bytes: Vec<u8> = tokens
        .iter()
        .flat_map(|token| token.to_le_bytes())
        .collect();
    let (p50, p99) = time(&|| {
        let bytes = std::hint::black_box(&bytes);
        std::hint::black_box(blake2b_simd::Params::new().hash_length(16).hash(bytes));
    });
    println!("blake2b of the same 400 KB in one call: p50 {p50:?}, p99 {p99:?}");
    for (chunk_tokens, probe) in [
        (256, RankProbe::First),
        (256, RankProbe::All),
        (16, RankProbe::First),
    ] {
        let rule = AtomChunk {
            namespace: NS.into(),
            chunk_tokens,
            world: 8,
            probe,
        };
        let (p50, p99) = time(&|| {
            std::hint::black_box(rule.keys(std::hint::black_box(&tokens)));
        });
        let keys = rule.keys(&tokens).len();
        println!("AtomChunk keys, 100k tokens, chunk {chunk_tokens}: {keys} keys, p50 {p50:?}, p99 {p99:?}");
    }
}
