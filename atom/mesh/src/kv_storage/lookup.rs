//! A prompt looked up on the request path in the store a worker reads: ATOM's keys of
//! its chunks, asked of every pool within a deadline, and what the engine loads of the
//! prefix the store holds. A source that keeps failing is skipped for a while, and so are
//! the lookups past a cap on those in flight.

use std::{
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    },
    time::{Duration, Instant},
};

use parking_lot::Mutex;

use super::{
    keys,
    mooncake::{self, KeyState},
    OffloadReport,
};

const MAX_IN_FLIGHT: usize = 64;
/// Failures in a row that open a source's breaker, which a lookup probes again after
/// `BREAKER_OPEN`.
const BREAKER_FAILURES: u32 = 5;
const BREAKER_OPEN: Duration = Duration::from_secs(10);

/// The lookups of one source.
#[derive(Debug, Default)]
pub struct Lookups {
    in_flight: AtomicUsize,
    breaker: Mutex<Breaker>,
}

#[derive(Debug, Default)]
struct Breaker {
    failures: u32,
    open_until: Option<Instant>,
    last: Option<Duration>,
}

impl Lookups {
    pub fn breaker_open(&self) -> bool {
        let open_until = self.breaker.lock().open_until;
        open_until.is_some_and(|until| Instant::now() < until)
    }

    /// How long the last lookup took, from its first request to its last answer.
    pub fn last_lookup_us(&self) -> Option<u64> {
        let last = self.breaker.lock().last;
        last.map(|took| u64::try_from(took.as_micros()).unwrap_or(u64::MAX))
    }

    /// Whether a lookup may go ahead: the breaker is closed, or it has been open long
    /// enough for this lookup to probe the source while it keeps the others out.
    fn admit(&self) -> bool {
        let mut breaker = self.breaker.lock();
        match breaker.open_until {
            Some(until) if Instant::now() < until => false,
            Some(_) => {
                breaker.open_until = Some(Instant::now() + BREAKER_OPEN);
                true
            }
            None => true,
        }
    }

    fn record(&self, ok: bool, took: Duration) {
        let mut breaker = self.breaker.lock();
        breaker.last = Some(took);
        if ok {
            (breaker.failures, breaker.open_until) = (0, None);
        } else {
            breaker.failures += 1;
            if breaker.failures >= BREAKER_FAILURES {
                breaker.open_until = Some(Instant::now() + BREAKER_OPEN);
            }
        }
    }
}

/// A lookup in flight, counted until it is dropped.
struct Flight<'a>(&'a AtomicUsize);

impl<'a> Flight<'a> {
    fn board(in_flight: &'a AtomicUsize) -> Option<Self> {
        let flight = Self(in_flight);
        (in_flight.fetch_add(1, Ordering::AcqRel) < MAX_IN_FLIGHT).then_some(flight)
    }
}

impl Drop for Flight<'_> {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::AcqRel);
    }
}

/// What a lookup reads of a source, out of its lock.
pub(super) struct Read {
    pub id: String,
    pub report: OffloadReport,
    pub admins: Vec<String>,
    pub readable: bool,
    pub lookups: Arc<Lookups>,
    /// From the first request to the last answer; later answers are dropped.
    pub deadline: Duration,
}

impl Read {
    /// The tokens of `tokens` the engine reports cached when its device cache holds
    /// `device` of them, after it loads what it will of what the store holds; None when
    /// the store does not answer.
    pub async fn cached_tokens(
        &self,
        client: &reqwest::Client,
        tokens: &[u32],
        device: u32,
    ) -> Option<u32> {
        let prompt = u32::try_from(tokens.len()).unwrap_or(u32::MAX);
        let (chunk, min_load) = (self.report.chunk_tokens, self.report.min_load_tokens);
        let loaded = |hit| {
            let hit = loadable_hit(hit, prompt, chunk);
            (load_tokens(device, hit, chunk, min_load) > 0).then_some(hit)
        };
        if loaded(prompt).is_none() {
            self.count("skipped");
            return Some(device);
        }
        let held = self.lookup(client, tokens).await;
        self.count(match held {
            Ok(0) => "miss",
            Ok(_) => "hit",
            Err(reason) => reason,
        });
        Some(loaded(held.ok()?).unwrap_or(device))
    }

    fn count(&self, result: &'static str) {
        let source = self.id.clone();
        metrics::counter!("mesh_kv_storage_lookups_total", "source" => source, "result" => result)
            .increment(1);
    }

    /// The tokens from the prompt's start some pool holds, in memory or on disk.
    async fn lookup(&self, client: &reqwest::Client, tokens: &[u32]) -> Result<u32, &'static str> {
        if !self.readable {
            return Err("unreadable");
        }
        let Some(_flight) = Flight::board(&self.lookups.in_flight) else {
            return Err("overloaded");
        };
        if !self.lookups.admit() {
            return Err("breaker_open");
        }
        let chunk = self.report.chunk_tokens;
        let chain = keys::chunk_chain(tokens, chunk as usize);
        let keys = keys::rank0_keys(&self.report.namespace, self.report.world, &chain);
        let started = Instant::now();
        let asked = mooncake::query_pools(client, &self.admins, &keys);
        let held = match tokio::time::timeout(self.deadline, asked).await {
            Ok(states) => held_chunks(&states).ok_or("error"),
            Err(_) => Err("timeout"),
        };
        let took = started.elapsed();
        self.lookups.record(held.is_ok(), took);
        let source = self.id.clone();
        metrics::histogram!("mesh_kv_storage_lookup_duration_seconds", "source" => source)
            .record(took);
        held.map(|chunks| chunks * chunk)
    }
}

/// The chunks from the start that some pool holds, unless no pool answered for the
/// first one after them, which might be held too.
fn held_chunks(states: &[KeyState]) -> Option<u32> {
    let held = states.iter().take_while(|state| state.is_held()).count();
    match states.get(held) {
        Some(KeyState::Unknown) => None,
        _ => u32::try_from(held).ok(),
    }
}

/// What ATOM's scheduler can load of a store's `hit` for a prompt of `prompt` tokens:
/// a hit of the whole prompt leaves its last token to compute, and loads go by chunks.
fn loadable_hit(hit: u32, prompt: u32, chunk: u32) -> u32 {
    let hit = if hit == prompt {
        hit.saturating_sub(1)
    } else {
        hit
    };
    hit / chunk * chunk
}

/// The tokens ATOM's scheduler loads of a `loadable` hit, from the first chunk boundary
/// at or after the `device` tokens it holds; none when they are fewer than `min_load`.
fn load_tokens(device: u32, loadable: u32, chunk: u32, min_load: u32) -> u32 {
    let moved = loadable.saturating_sub(device.div_ceil(chunk).saturating_mul(chunk));
    if moved > 0 && moved >= min_load {
        moved
    } else {
        0
    }
}

#[cfg(test)]
mod tests {
    use std::{
        collections::HashMap,
        sync::atomic::{AtomicBool, AtomicU64},
    };

    use axum::{
        extract::{Query, State},
        http::StatusCode,
        response::{IntoResponse, Response},
        routing::get,
        Json, Router,
    };
    use serde_json::{json, Map};

    use super::*;

    #[test]
    fn the_engine_loads_a_hit_that_adds_enough_past_its_device_cache() {
        // A dense model's prompt of 20000 tokens in chunks of 256 with 3000 on the device,
        // loading no fewer than 8192: ATOM loads 16384 from a chunk boundary, not 10240.
        let cached = |device, hit| {
            let hit = loadable_hit(hit, 20_000, 256);
            if load_tokens(device, hit, 256, 8192) > 0 {
                hit
            } else {
                device
            }
        };
        assert_eq!((cached(3000, 16_384), cached(3000, 10_240)), (16_384, 3000));
        // A load starts at the device's next chunk boundary: 11264 − 3072 is 8192.
        assert_eq!(load_tokens(3000, 11_264, 256, 8192), 8192);
        assert_eq!(load_tokens(3000, 11_008, 256, 8192), 0);
        assert_eq!(load_tokens(0, 0, 256, 0), 0);
        // A hit of the whole prompt leaves its last token, and its last chunk, to compute.
        assert_eq!(loadable_hit(512, 512, 256), 256);
        assert_eq!(loadable_hit(500, 511, 256), 256);
    }

    #[test]
    fn a_held_prefix_ends_at_the_first_chunk_no_pool_holds() {
        use KeyState::*;
        for (states, want) in [
            (&[Memory, SsdOnly, Absent, Memory][..], Some(2)),
            (&[Memory, Memory], Some(2)),
            (&[Absent, Unknown], Some(0)),
            (&[Memory, Unknown, Memory], None),
            (&[], Some(0)),
        ] {
            assert_eq!(held_chunks(states), want, "{states:?}");
        }
    }

    /// The admin service of a pool holding the first `held` chunks of a prompt asked for
    /// in one batch, answering after `delay_ms`, or failing.
    #[derive(Default)]
    struct Admin {
        held: AtomicUsize,
        delay_ms: AtomicU64,
        fail: AtomicBool,
        asked: AtomicUsize,
    }

    async fn answer(
        State(admin): State<Arc<Admin>>,
        Query(query): Query<HashMap<String, String>>,
    ) -> Response {
        admin.asked.fetch_add(1, Ordering::Relaxed);
        let delay = admin.delay_ms.load(Ordering::Relaxed);
        tokio::time::sleep(Duration::from_millis(delay)).await;
        if admin.fail.load(Ordering::Relaxed) {
            return StatusCode::INTERNAL_SERVER_ERROR.into_response();
        }
        let held = admin.held.load(Ordering::Relaxed);
        let replicas = |i| match i < held {
            true => json!({"ok": true, "values": [{}]}),
            false => json!({"ok": false, "error": "OBJECT_NOT_FOUND"}),
        };
        let keys = query["keys"].split(',');
        let held = |(i, key): (usize, &str)| (key.to_string(), replicas(i));
        let data: Map<String, serde_json::Value> = keys.enumerate().map(held).collect();
        Json(json!({"success": true, "data": data})).into_response()
    }

    async fn source(admin: Arc<Admin>) -> Read {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        let service = Router::new()
            .route("/batch_query_keys", get(answer))
            .with_state(admin);
        tokio::spawn(async move { axum::serve(listener, service).await });
        let report = OffloadReport {
            backend: "mooncake_store".into(),
            namespace: "atomkv1-0123456789abcdef01234567".into(),
            chunk_tokens: 16,
            world: 1,
            min_load_tokens: 16,
            ..Default::default()
        };
        Read {
            id: "atom-mooncake-01234567".into(),
            report,
            admins: vec![url],
            readable: true,
            lookups: Arc::default(),
            deadline: Duration::from_millis(500),
        }
    }

    #[tokio::test]
    async fn a_prompt_is_looked_up_within_the_deadline_until_the_breaker_opens() {
        let recorder = metrics_exporter_prometheus::PrometheusBuilder::new().build_recorder();
        let handle = recorder.handle();
        let _recorder = metrics::set_default_local_recorder(&recorder);
        let admin = Arc::new(Admin::default());
        let read = source(admin.clone()).await;
        let client = reqwest::Client::new();
        let prompt: Vec<u32> = (0..64).collect();
        let cached = |device| read.cached_tokens(&client, &prompt, device);

        // All 4 chunks held: the engine loads 3 of them, past what its device holds.
        admin.held.store(4, Ordering::Relaxed);
        assert_eq!((cached(0).await, cached(32).await), (Some(48), Some(48)));
        assert!(read.lookups.last_lookup_us().is_some());
        admin.held.store(1, Ordering::Relaxed);
        assert_eq!(cached(0).await, Some(16));
        admin.held.store(0, Ordering::Relaxed);
        assert_eq!(cached(16).await, Some(16));
        // The store could not add a chunk past 40 tokens of 64: it is not asked.
        assert_eq!(cached(40).await, Some(40));
        assert_eq!(admin.asked.load(Ordering::Relaxed), 4);

        // Late or failed answers are unknown, and 5 in a row open the breaker.
        admin.delay_ms.store(1000, Ordering::Relaxed);
        assert_eq!(cached(0).await, None);
        admin.delay_ms.store(0, Ordering::Relaxed);
        admin.fail.store(true, Ordering::Relaxed);
        for _ in 0..4 {
            assert_eq!(cached(0).await, None);
        }
        assert!(read.lookups.breaker_open());
        assert_eq!(cached(0).await, None);
        assert_eq!(admin.asked.load(Ordering::Relaxed), 9);
        // Once it has been open long enough, a lookup probes the store, and closes it.
        admin.fail.store(false, Ordering::Relaxed);
        read.lookups.breaker.lock().open_until = Some(Instant::now());
        assert_eq!(cached(0).await, Some(0));
        assert!(!read.lookups.breaker_open());

        // Past the cap on lookups in flight, or of a store whose keys the router does not
        // compute, nothing is asked.
        read.lookups
            .in_flight
            .store(MAX_IN_FLIGHT, Ordering::Relaxed);
        assert_eq!(cached(0).await, None);
        assert_eq!(
            read.lookups.in_flight.load(Ordering::Relaxed),
            MAX_IN_FLIGHT
        );
        read.lookups.in_flight.store(0, Ordering::Relaxed);
        let unreadable = Read {
            readable: false,
            lookups: Arc::default(),
            ..read
        };
        let cached = |device| unreadable.cached_tokens(&client, &prompt, device);
        assert_eq!(cached(0).await, None);
        assert_eq!(admin.asked.load(Ordering::Relaxed), 10);

        let rendered = handle.render();
        for (result, count) in [
            ("hit", 3),
            ("miss", 2),
            ("skipped", 1),
            ("timeout", 1),
            ("error", 4),
            ("breaker_open", 1),
            ("overloaded", 1),
            ("unreadable", 1),
        ] {
            let line = format!(
                r#"mesh_kv_storage_lookups_total{{source="atom-mooncake-01234567",result="{result}"}} {count}"#
            );
            assert!(rendered.contains(&line), "missing {line} in\n{rendered}");
        }
        let lookups =
            r#"mesh_kv_storage_lookup_duration_seconds_count{source="atom-mooncake-01234567"} 10"#;
        assert!(
            rendered.contains(lookups),
            "missing {lookups} in\n{rendered}"
        );
    }
}
