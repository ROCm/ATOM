//! Where the policies read the tiers a worker's engine takes a prompt's KV from, with
//! `--kv-events`: the KV event index, for its device cache, and the stores the workers
//! read, which a policy asks itself for what its engine will load.

use std::sync::Arc;

use crate::{
    core::{Worker, WorkerType},
    kv_index::PrefixMatchProvider,
    kv_storage::StorageSources,
    kv_tiers::{TierWeights, TieredHit},
};

/// The KV event index and the stores the workers read, if any.
#[derive(Debug, Clone)]
pub(crate) struct KvTiers {
    index: Arc<dyn PrefixMatchProvider>,
    stores: Option<Arc<StorageSources>>,
}

impl KvTiers {
    pub(crate) fn new(
        index: Arc<dyn PrefixMatchProvider>,
        stores: Option<Arc<StorageSources>>,
    ) -> Self {
        Self { index, stores }
    }

    /// What each worker's engine will take the leading tokens of `tokens` from, in the
    /// order of `workers`; None for a worker the index knows nothing of. A tier `weights`
    /// counts for nothing is not asked, and holds no tokens: the stores are asked, once a
    /// call, only where what they hold counts.
    pub(crate) async fn tiered_hits(
        &self,
        tokens: &[u32],
        workers: &[Arc<dyn Worker>],
        weights: &TierWeights,
    ) -> Vec<Option<TieredHit>> {
        let mut hits = self.index.tiered_hits(tokens, workers);
        if let (true, Some(stores)) = (weights.stored > 0.0, &self.stores) {
            ask_stores(stores, tokens, workers, &mut hits).await;
        }
        hits
    }

    /// Predicts that `worker` will hold `tokens`, routed to it.
    pub(crate) fn on_routed(&self, tokens: &[u32], worker: &dyn Worker) {
        self.index.on_routed(tokens, worker);
    }
}

/// Sets what each worker's engine will load of `tokens` from the store it reads, past what
/// its device cache holds by `hits`: None where that store does not answer.
async fn ask_stores(
    stores: &StorageSources,
    tokens: &[u32],
    workers: &[Arc<dyn Worker>],
    hits: &mut [Option<TieredHit>],
) {
    // A decode worker takes the prompt's KV from its prefill worker.
    let (readers, asked): (Vec<usize>, Vec<(&str, u32)>) = (workers.iter().zip(hits.iter()))
        .enumerate()
        .filter(|(_, (worker, _))| !matches!(worker.worker_type(), WorkerType::Decode))
        .filter_map(|(i, (worker, hit))| Some((i, (worker.url(), (*hit)?.device))))
        .unzip();
    let cached = stores.cached_tokens(tokens, &asked).await;
    for ((i, (_, device)), cached) in readers.into_iter().zip(asked).zip(cached) {
        if let Some(hit) = &mut hits[i] {
            hit.stored = cached.map(|cached| cached.saturating_sub(device));
        }
    }
}

/// A Mooncake Store for the tests of what the policies read of it.
#[cfg(test)]
pub(crate) mod testing {
    use std::{
        collections::HashMap,
        sync::atomic::{AtomicBool, AtomicUsize, Ordering},
        time::Duration,
    };

    use axum::{
        extract::{Query, State},
        http::StatusCode,
        response::{IntoResponse, Response},
        routing::get,
        Json, Router,
    };
    use serde_json::{json, Map, Value};

    use super::*;
    use crate::kv_storage::{keys, OffloadReport};

    /// The one pool of a store: its master's admin service holds the first `held` chunks
    /// of a prompt asked for in one batch, or fails, and counts the requests it gets.
    #[derive(Debug, Default)]
    pub(crate) struct Pool {
        pub held: AtomicUsize,
        pub fail: AtomicBool,
        pub asked: AtomicUsize,
    }

    async fn answer(
        State(pool): State<Arc<Pool>>,
        Query(query): Query<HashMap<String, String>>,
    ) -> Response {
        pool.asked.fetch_add(1, Ordering::Relaxed);
        if pool.fail.load(Ordering::Relaxed) {
            return StatusCode::INTERNAL_SERVER_ERROR.into_response();
        }
        let held = pool.held.load(Ordering::Relaxed);
        let state = |(i, key): (usize, &str)| {
            let replicas = match i < held && key != keys::TAINT_KEY {
                true => json!({"ok": true, "values": [{}]}),
                false => json!({"ok": false, "error": "OBJECT_NOT_FOUND"}),
            };
            (key.to_string(), replicas)
        };
        let data: Map<String, Value> = query["keys"].split(',').enumerate().map(state).collect();
        Json(json!({"success": true, "data": data})).into_response()
    }

    /// The store of `pool`, read by `readers` in chunks of `chunk_tokens`, each loading no
    /// fewer than `min_load_tokens`.
    pub(crate) async fn stores(
        pool: Arc<Pool>,
        readers: &[&str],
        chunk_tokens: u32,
        min_load_tokens: u32,
    ) -> Arc<StorageSources> {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();
        let service = Router::new()
            .route("/batch_query_keys", get(answer))
            .with_state(pool);
        tokio::spawn(async move { axum::serve(listener, service).await });
        let stores = StorageSources::new(port, Duration::from_secs(10), reqwest::Client::new());
        let report = OffloadReport {
            backend: "mooncake_store".into(),
            masters: vec!["127.0.0.1:50051".into()],
            namespace: "atomkv1-0123456789abcdef01234567".into(),
            chunk_tokens,
            world: 1,
            min_load_tokens,
            chain_check: keys::chain_check(chunk_tokens),
        };
        for reader in readers {
            stores.add(reader, "model", &report);
        }
        Arc::new(stores)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::Ordering;

    use super::{
        testing::{stores, Pool},
        *,
    };
    use crate::core::BasicWorkerBuilder;

    /// An index that knows each worker's tiers.
    #[derive(Debug)]
    struct Index(Vec<Option<TieredHit>>);

    impl PrefixMatchProvider for Index {
        fn tiered_hits(&self, _: &[u32], _: &[Arc<dyn Worker>]) -> Vec<Option<TieredHit>> {
            self.0.clone()
        }
    }

    const URLS: [&str; 4] = [
        "http://d:8000",
        "http://n:8000",
        "http://r:8000",
        "http://h:8000",
    ];

    /// A decode worker, one the index knows nothing of, one with nothing on its device,
    /// and one whose device cache holds 8 tokens.
    fn workers_and_hits() -> (Vec<Arc<dyn Worker>>, Vec<Option<TieredHit>>) {
        let decode = BasicWorkerBuilder::new(URLS[0]).worker_type(WorkerType::Decode);
        let workers: Vec<Arc<dyn Worker>> = vec![
            Arc::new(decode.build()),
            Arc::new(BasicWorkerBuilder::new(URLS[1]).build()),
            Arc::new(BasicWorkerBuilder::new(URLS[2]).build()),
            Arc::new(BasicWorkerBuilder::new(URLS[3]).build()),
        ];
        let hit = |device| {
            Some(TieredHit {
                device,
                stored: Some(0),
            })
        };
        (workers, vec![hit(0), None, hit(0), hit(8)])
    }

    /// A decode worker takes the prompt's KV from its prefill worker, and a worker whose
    /// device cache the index knows nothing of is not asked for: of four readers of a
    /// store, it answers for the last two, each by what its own device cache holds.
    #[tokio::test]
    async fn a_store_answers_for_its_readers_the_index_knows_but_decode_workers() {
        let (workers, hits) = workers_and_hits();
        let pool = Arc::new(Pool::default());
        // The store holds the 6 chunks of 12 tokens, of which the engine takes 10. With 8 on
        // the device, the 2 more are under the 4 it loads at least.
        pool.held.store(6, Ordering::Relaxed);
        let stores = stores(pool.clone(), &URLS, 2, 4).await;
        let tiers = KvTiers::new(Arc::new(Index(hits.clone())), Some(stores));
        let weights = TierWeights { stored: 1.0 };
        let prompt: Vec<u32> = (0..12).collect();
        let got = tiers.tiered_hits(&prompt, &workers, &weights).await;
        let loaded = |hit: Option<TieredHit>, stored| hit.map(|hit| TieredHit { stored, ..hit });
        let want = vec![hits[0], None, loaded(hits[2], Some(10)), hits[3]];
        assert_eq!(got, want);
        let cached: Vec<_> = got
            .iter()
            .map(|hit| hit.and_then(|hit| hit.cached()))
            .collect();
        assert_eq!(cached, [Some(0), None, Some(10), Some(8)]);
        // One store, asked once for all its readers.
        assert_eq!(pool.asked.load(Ordering::Relaxed), 1);

        // A store that does not answer leaves its readers' loads unknown, not the others'.
        pool.fail.store(true, Ordering::Relaxed);
        let got = tiers.tiered_hits(&prompt, &workers, &weights).await;
        let want = vec![hits[0], None, loaded(hits[2], None), loaded(hits[3], None)];
        assert_eq!(got, want);
    }

    /// At a weight of 0 a tier is not asked: no request goes to a store.
    #[tokio::test]
    async fn a_tier_that_counts_for_nothing_is_not_asked() {
        let recorder = metrics_exporter_prometheus::PrometheusBuilder::new().build_recorder();
        let _recorder = metrics::set_default_local_recorder(&recorder);
        let (workers, hits) = workers_and_hits();
        let pool = Arc::new(Pool::default());
        pool.held.store(6, Ordering::Relaxed);
        let stores = stores(pool.clone(), &URLS, 2, 4).await;
        let index = Arc::new(Index(hits.clone()));
        let tiers = KvTiers::new(index.clone(), Some(stores));
        let prompt: Vec<u32> = (0..12).collect();
        let lookups = || {
            recorder
                .handle()
                .render()
                .contains("mesh_kv_storage_lookups_total")
        };

        let unweighed = TierWeights { stored: 0.0 };
        assert_eq!(tiers.tiered_hits(&prompt, &workers, &unweighed).await, hits);
        assert_eq!(pool.asked.load(Ordering::Relaxed), 0);
        assert!(!lookups());

        let weighed = TierWeights { stored: 0.5 };
        let got = tiers.tiered_hits(&prompt, &workers, &weighed).await;
        assert_eq!(got[2].and_then(|hit| hit.stored), Some(10));
        assert_eq!(pool.asked.load(Ordering::Relaxed), 1);
        assert!(lookups());

        // With no stores to ask, the index answers alone.
        let alone = KvTiers::new(index, None);
        assert_eq!(alone.tiered_hits(&prompt, &workers, &weighed).await, hits);
        assert_eq!(pool.asked.load(Ordering::Relaxed), 1);
    }
}
