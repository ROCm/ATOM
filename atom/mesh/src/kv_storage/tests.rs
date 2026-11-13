use std::{sync::Arc, time::Duration};

use serde::Deserialize;

use super::{
    hub::Connect,
    keys::{chain_check, STORE_TAINT_KEY},
    merge::merge,
    *,
};
use crate::{
    config::KvStorageConfig,
    core::{placement::test_support::make_regular_http, Worker},
    kv_events::{
        discovery::{OffloadBackend, OffloadReport},
        Namespace,
    },
    kv_index::{
        CacheHit, HitSource, IndexProvider, LoadPath, LowerHit, PrefixMatchProvider, PrefixRequest,
    },
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

fn workers(names: &[&str]) -> Vec<Arc<dyn Worker>> {
    names
        .iter()
        .map(|name| make_regular_http(&format!("http://{name}:8000"), "m"))
        .collect()
}

/// A DP replica of world 2 storing 4-token chunks and loading from 8 tokens on.
fn report(namespace: &str, chain_check: Option<String>) -> OffloadReport {
    OffloadReport {
        backend: OffloadBackend::MooncakeStore,
        masters: vec!["10.0.0.1:50051".into(), "10.0.0.3:50051".into()],
        namespace: Some(namespace.into()),
        chunk_tokens: 4,
        world: Some(2),
        min_load_tokens: 8,
        chain_check,
    }
}

fn hub(store: &Arc<MockStore>, config: KvStorageConfig) -> Hub {
    let store = store.clone();
    let connect: Connect =
        Arc::new(move |_: &StorageBackendSpec| Some(store.clone() as Arc<dyn StorageBackend>));
    Hub::new(config, connect)
}

/// A hub with one source, which worker `a` reads and `pools` serve.
async fn reading(pools: Vec<MockPool>, config: KvStorageConfig) -> (Arc<MockStore>, Hub) {
    let store = Arc::new(MockStore::new(pools));
    let hub = hub(&store, config);
    hub.on_worker_added("http://a:8000", "m", Some(&report(NS, None)))
        .await;
    (store, hub)
}

async fn lookup(hub: &Hub, tokens: &[u32], candidates: &[Arc<dyn Worker>]) -> StorageLookup {
    let req = PrefixRequest::new("m", tokens, Namespace::default());
    let mut lookups = hub.lookup_all(&req, candidates).await;
    assert_eq!(lookups.len(), 1);
    lookups.pop().unwrap().1
}

#[tokio::test]
async fn chain_check_gates_sources_and_readers() {
    let workers = workers(&["a", "b", "c", "d"]);
    let hub = hub(&Arc::new(MockStore::default()), KvStorageConfig::default());
    let passed = report(NS, Some(chain_check(4)));
    // A rule that differs from the router's, here over 8-token chunks.
    let changed = report(NS, Some(chain_check(8)));
    let other = report("atomkv1-0123456789abcdef01234567", None);
    for (worker, report) in workers.iter().zip([&passed, &passed, &changed, &other]) {
        hub.on_worker_added(worker.url(), "m", Some(report)).await;
    }

    let status = hub.status();
    let ids: Vec<&str> = status.iter().map(|source| source.id.as_str()).collect();
    let checks: Vec<ChainCheck> = status.iter().map(|source| source.chain_check).collect();
    // The worker whose rule differs reads a store of its own, unreadable to the router.
    let failed = "atom-mooncake-b3fd64e0-2";
    assert_eq!(
        ids,
        ["atom-mooncake-b3fd64e0", failed, "atom-mooncake-01234567"]
    );
    use ChainCheck::*;
    assert_eq!(checks, [Passed, Failed, NotReported]);
    let endpoints = ["http://10.0.0.1:9003", "http://10.0.0.3:9003"];
    assert_eq!(status[0].endpoints, endpoints);
    assert_eq!(status[0].readers, [workers[0].url(), workers[1].url()]);
    let reads = |id: &str| {
        let source = SourceId(id.into());
        ReaderInfo {
            sources: vec![source.clone()],
            load_path: Some(LoadPath::Storage(source)),
        }
    };
    assert_eq!(hub.reader(workers[1].url()), reads(ids[0]));
    assert_eq!(hub.reader(workers[2].url()), reads(failed));
    let tokens: Vec<u32> = (0..24).collect();
    let req = PrefixRequest::new("m", &tokens, Namespace::default());
    let unreadable = StorageLookup::Unavailable(FailReason::Unreadable);
    assert_eq!(hub.lookup_all(&req, &workers[2..3]).await[1].1, unreadable);

    // Another model gets its own source; a worker registered again keeps only its new role.
    hub.on_worker_added("http://e:8000", "other", Some(&passed))
        .await;
    hub.on_worker_added(workers[2].url(), "m", None).await;
    hub.on_worker_removed(workers[0].url()).await;
    hub.on_worker_removed(workers[1].url()).await;
    let ids: Vec<String> = hub.status().into_iter().map(|source| source.id).collect();
    assert_eq!(ids, ["atom-mooncake-01234567", "atom-mooncake-b3fd64e0-3"]);
}

#[tokio::test]
async fn umbp_without_a_master_leaves_lower_unknown_and_multimodal_requests_skip() {
    let tokens: Vec<u32> = (0..24).collect();
    let workers = workers(&["a", "b"]);
    let (store, hub) = reading(pools(&tokens, "MM"), KvStorageConfig::default()).await;
    let umbp = OffloadReport {
        backend: OffloadBackend::UmbpStore,
        masters: Vec::new(),
        ..report("atomumbp1-0123456789abcdef01234567", None)
    };
    hub.on_worker_added(workers[1].url(), "m", Some(&umbp))
        .await;
    let hub = Arc::new(hub);
    let provider = WithStorage::new(IndexProvider::new(Duration::from_secs(60), 4), hub.clone());
    let mut req = PrefixRequest::new("m", &tokens, Namespace::default());
    req.set_storage(hub.lookup_all(&req, &workers).await);
    let umbp = SourceId("atom-umbp-01234567".into());
    let unknown = CacheHit {
        tokens: [0; 4],
        lower: LowerHit::Unknown(umbp.clone()),
        source: HitSource::Storage,
        exact: false,
    };
    assert_eq!(provider.match_workers(&req, &workers)[1], Some(unknown));
    let status = &hub.status()[1];
    assert_eq!((status.id.as_str(), status.endpoints.len()), (&*umbp.0, 0));

    req.multimodal = true;
    let skipped = StorageLookup::Skipped(GateReason::Multimodal);
    let lookups = hub.lookup_all(&req, &workers).await;
    assert!(lookups.iter().all(|(_, lookup)| *lookup == skipped));
    assert_eq!(store.queries(), 1);
}

#[tokio::test(start_paused = true)]
async fn gate_and_deadline() {
    let tokens: Vec<u32> = (0..24).collect();
    let workers = workers(&["a", "b"]);
    let (store, hub) = reading(pools(&tokens, "MM"), KvStorageConfig::default()).await;

    let too_short = StorageLookup::Skipped(GateReason::TooShort);
    assert_eq!(lookup(&hub, &tokens[..7], &workers).await, too_short);
    let no_reader = StorageLookup::Skipped(GateReason::NoReaderInCandidates);
    assert_eq!(lookup(&hub, &tokens, &workers[1..]).await, no_reader);
    assert_eq!(store.queries(), 0);
    assert_eq!(lookup(&hub, &tokens, &workers).await, hit(8, 8));

    store.pools.lock()[0].delay = Duration::from_millis(10);
    let timeout = StorageLookup::Unavailable(FailReason::Timeout);
    assert_eq!(lookup(&hub, &tokens, &workers).await, timeout);
    // A request for another model looks nothing up.
    let req = PrefixRequest::new("other", &tokens, Namespace::default());
    assert!(hub.lookup_all(&req, &workers).await.is_empty());
    assert_eq!(store.queries(), 2);
}

#[tokio::test(start_paused = true)]
async fn breaker_opens_after_five_failures_and_lets_one_probe_through() {
    let tokens: Vec<u32> = (0..24).collect();
    let workers = workers(&["a"]);
    let (store, hub) = reading(pools(&tokens, "x"), KvStorageConfig::default()).await;

    let failed = StorageLookup::Unavailable(FailReason::Transport);
    for _ in 0..5 {
        assert_eq!(lookup(&hub, &tokens, &workers).await, failed);
    }
    let open = StorageLookup::Skipped(GateReason::BreakerOpen);
    assert_eq!(lookup(&hub, &tokens, &workers).await, open);
    assert_eq!(store.queries(), 5);
    assert_eq!(hub.status()[0].breaker, BreakerState::Open);

    tokio::time::advance(Duration::from_secs(10)).await;
    assert_eq!(hub.status()[0].breaker, BreakerState::HalfOpen);
    *store.pools.lock() = pools(&tokens, "MM");
    store.pools.lock()[0].delay = Duration::from_millis(1);
    let (probe, other) = tokio::join!(
        lookup(&hub, &tokens, &workers),
        lookup(&hub, &tokens, &workers)
    );
    assert_eq!((probe, other), (hit(8, 8), open));
    assert_eq!(hub.status()[0].breaker, BreakerState::Closed);
}

#[tokio::test(start_paused = true)]
async fn lookups_beyond_the_inflight_cap_skip_the_source() {
    let tokens: Vec<u32> = (0..24).collect();
    let workers = workers(&["a"]);
    let mut slow = pools(&tokens, "MM");
    slow[0].delay = Duration::from_millis(1);
    let config = KvStorageConfig {
        max_inflight: 1,
        ..KvStorageConfig::default()
    };
    let (_, hub) = reading(slow, config).await;
    let (first, second) = tokio::join!(
        lookup(&hub, &tokens, &workers),
        lookup(&hub, &tokens, &workers)
    );
    let overloaded = StorageLookup::Skipped(GateReason::Overloaded);
    assert_eq!((first, second), (hit(8, 8), overloaded));
    assert_eq!(lookup(&hub, &tokens, &workers).await, hit(8, 8));
}

#[test]
fn storage_hits_extend_reader_tiers_and_set_lower() {
    let source = SourceId("s".into());
    let reader = ReaderInfo {
        sources: vec![source.clone()],
        load_path: Some(LoadPath::Storage(source.clone())),
    };
    let events = |tokens| CacheHit {
        tokens,
        lower: LowerHit::None,
        source: HitSource::Events,
        exact: true,
    };
    let stored = [(source.clone(), hit(16, 24))];
    let lower = LowerHit::Hit {
        tokens: 24,
        from: LoadPath::Storage(source.clone()),
    };

    let merged = merge(Some(events([8; 4])), &reader, &stored).unwrap();
    assert_eq!((merged.tokens, &merged.lower), ([8, 8, 16, 24], &lower));
    let merged = merge(Some(events([8, 20, 20, 20])), &reader, &stored).unwrap();
    assert_eq!(merged.tokens, [8, 20, 20, 24]);
    let stored_only = merge(None, &reader, &stored).unwrap();
    let expected = (stored_only.tokens, stored_only.source, stored_only.exact);
    assert_eq!(expected, ([0, 0, 16, 24], HitSource::Storage, false));
    assert_eq!(merge(None, &reader, &[(source.clone(), hit(0, 0))]), None);

    let failed = [(
        source.clone(),
        StorageLookup::Unavailable(FailReason::Timeout),
    )];
    let merged = merge(Some(events([8; 4])), &reader, &failed).unwrap();
    assert_eq!(
        (merged.tokens, &merged.lower),
        ([8; 4], &LowerHit::Unknown(source.clone()))
    );
    let unknown = merge(None, &reader, &failed).unwrap();
    assert_eq!(
        (unknown.tokens, unknown.lower),
        ([0; 4], LowerHit::Unknown(source.clone()))
    );
    let open = [(
        source.clone(),
        StorageLookup::Skipped(GateReason::BreakerOpen),
    )];
    assert!(merge(None, &reader, &open).is_some());
    let short = [(source, StorageLookup::Skipped(GateReason::TooShort))];
    assert_eq!(merge(None, &reader, &short), None);
    let other = ReaderInfo::default();
    assert_eq!(
        merge(Some(events([8; 4])), &other, &stored),
        Some(events([8; 4]))
    );
}

#[tokio::test]
async fn the_planner_lookup_reaches_readers_only() {
    let tokens: Vec<u32> = (0..24).collect();
    let workers = workers(&["a", "b"]);
    let (_, hub) = reading(pools(&tokens, "MMS"), KvStorageConfig::default()).await;
    let hub = Arc::new(hub);
    let provider = WithStorage::new(IndexProvider::new(Duration::from_secs(60), 4), hub.clone());

    let mut req = PrefixRequest::new("m", &tokens, Namespace::default());
    assert_eq!(provider.match_workers(&req, &workers), [None, None]);
    let lookups = hub.lookup_all(&req, &workers).await;
    req.set_storage(lookups);
    let lower = LowerHit::Hit {
        tokens: 12,
        from: LoadPath::Storage(SourceId("atom-mooncake-b3fd64e0".into())),
    };
    let expected = CacheHit {
        tokens: [0, 0, 8, 12],
        lower,
        source: HitSource::Storage,
        exact: false,
    };
    assert_eq!(
        provider.match_workers(&req, &workers),
        [Some(expected), None]
    );

    let none: Connect =
        Arc::new(|_: &StorageBackendSpec| -> Option<Arc<dyn StorageBackend>> { None });
    let empty = Hub::new(KvStorageConfig::default(), none);
    let req = PrefixRequest::new("m", &tokens, Namespace::default());
    assert!(empty.lookup_all(&req, &workers).await.is_empty());
}
