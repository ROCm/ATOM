//! Time estimates from measured curves; missing calibration gives no cache score.
use super::{
    protocol::{content_keys, plan_reuse, ReusePlan},
    CacheIndexService, Source,
};
use crate::core::Worker;
use serde_json::Value;
use std::{
    collections::HashSet,
    sync::Arc,
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

#[derive(Debug)]
pub struct ReservationEntry {
    p: String,
    d: String,
    work: usize,
    p_pages: usize,
    d_pages: usize,
    accepted: HashSet<String>,
    created: Instant,
}

#[derive(Debug)]
pub struct Reservation {
    index: Arc<CacheIndexService>,
    pub id: String,
}
impl Drop for Reservation {
    fn drop(&mut self) {
        self.index.reservations.lock().remove(&self.id);
    }
}

pub struct PairSelection {
    pub prefill: Arc<dyn Worker>,
    pub decode: Arc<dyn Worker>,
    pub load_policy: &'static str,
    pub reservation: Reservation,
}

fn positive(value: &Value) -> Option<f64> {
    value.as_f64().filter(|v| v.is_finite() && *v >= 0.0)
}

pub fn curve(points: &Value, tokens: usize) -> Option<f64> {
    if tokens == 0 {
        return Some(0.0);
    }
    let points = points.as_array()?;
    let mut previous = (0.0, 0.0);
    for point in points {
        let x = positive(&point[0])?;
        let y = positive(&point[1])?;
        if x <= previous.0 || y < previous.1 {
            return None;
        }
        if tokens as f64 <= x {
            return Some(
                previous.1 + (y - previous.1) * (tokens as f64 - previous.0) / (x - previous.0),
            );
        }
        previous = (x, y);
    }
    None // No extrapolation past measured context/work ranges.
}

fn prefill(source: &Source, context: usize, tokens: usize) -> Option<f64> {
    if tokens == 0 {
        return Some(0.0);
    }
    let bucket = source.info["costs"]["prefill_curves"]
        .as_array()?
        .iter()
        .filter(|b| b["max_context_tokens"].as_u64().unwrap_or(0) >= context as u64)
        .min_by_key(|b| b["max_context_tokens"].as_u64().unwrap_or(u64::MAX))?;
    curve(&bucket["points"], tokens)
}

fn load_fresh(source: &Source) -> bool {
    let sampled = source.load["sample_time_unix_ns"]
        .as_str()
        .and_then(|s| s.parse::<u128>().ok());
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .ok()
        .map(|d| d.as_nanos());
    matches!((sampled, now), (Some(s), Some(n)) if n >= s && n - s < 3_000_000_000)
}

fn pages(n: usize, hit: usize, span: usize, reserve: usize) -> Option<usize> {
    if span == 0 {
        return None;
    }
    n.checked_add(reserve)?
        .div_ceil(span)
        .checked_sub(hit / span)
}

fn hbm(source: &Source, keys: &[String], n: usize) -> usize {
    let span = source.info["hash_block_size_tokens"].as_u64().unwrap_or(0) as usize;
    if span == 0 {
        return 0;
    }
    source.prefix(keys, "HBM").min(n.saturating_sub(1)) / span * span
}

fn p_cost(source: &Source, keys: &[String], n: usize, overlay: usize) -> Option<(f64, ReusePlan)> {
    if !load_fresh(source) {
        return None;
    }
    let h = hbm(source, keys, n);
    let chunk = source.info["lmcache_chunk_size_tokens"]
        .as_u64()
        .unwrap_or(256) as usize;
    let min_load = source.info["min_load_tokens"].as_u64()? as usize;
    let base = plan_reuse(n, h, 0, chunk, min_load, false)?;
    let waiting = source.load["pending_prefill_tokens"].as_u64()? as usize;
    let queue = prefill(source, n, waiting.checked_add(overlay)?)?;
    let mut best = (queue + prefill(source, n, n - h)?, base);
    if source.info["capabilities"]["cache_load_policy_hint"] != true {
        return None;
    }
    let cpu = plan_reuse(n, h, source.prefix(keys, "CPU"), chunk, min_load, true)?;
    if cpu.eligible {
        if let Some(copy) = curve(
            &source.info["costs"]["h2d_curve"],
            cpu.transfer_end - cpu.load_start,
        ) {
            let copy_queue = if source.load["offload"]["loads_pending"].as_u64() == Some(0) {
                Some(0.0)
            } else {
                positive(&source.load["copy_queue_ms"])
            };
            let Some(copy_queue) = copy_queue else {
                return Some(best);
            };
            let cost = queue
                + prefill(source, n, cpu.precompute_end - cpu.precompute_start)?
                + copy_queue
                + copy
                + prefill(source, n, n - cpu.reuse_end)?;
            if cost < best.0 {
                best = (cost, cpu);
            }
        }
    }
    Some(best)
}

impl CacheIndexService {
    pub fn select_pair(
        self: &Arc<Self>,
        prefill_workers: &[Arc<dyn Worker>],
        decode_workers: &[Arc<dyn Worker>],
        tokens: &[u32],
        namespace: &str,
        block: usize,
        model: Option<&str>,
        idx2idx: bool,
        decode_reserve: usize,
    ) -> Option<PairSelection> {
        let keys = content_keys(namespace, tokens, block)?;
        let sources = self.sources.read();
        let mut reservations = self.reservations.lock();
        reservations.retain(|_, r| r.created.elapsed() < Duration::from_secs(30));
        for (id, reservation) in reservations.iter_mut() {
            for url in [&reservation.p, &reservation.d] {
                if sources
                    .get(url)
                    .and_then(|s| s.load["accepted_dispatch_ids"].as_array())
                    .is_some_and(|ids| ids.iter().any(|v| v.as_str() == Some(id)))
                {
                    reservation.accepted.insert(url.clone());
                }
            }
        }
        let overlay = |url: &str| -> (usize, usize) {
            reservations
                .values()
                .filter(|r| !r.accepted.contains(url))
                .fold((0, 0), |(work, pages), r| {
                    (
                        work + if r.p == url { r.work } else { 0 },
                        pages
                            + if r.p == url {
                                r.p_pages
                            } else if r.d == url {
                                r.d_pages
                            } else {
                                0
                            },
                    )
                })
        };
        let healthy = |w: &Arc<dyn Worker>| {
            w.is_healthy()
                && w.circuit_breaker().can_execute()
                && model.is_none_or(|m| {
                    w.model_id() == m || w.model_id() == crate::core::UNKNOWN_MODEL_ID
                })
        };
        let mut best = None;
        for p in prefill_workers.iter().filter(|w| healthy(w)) {
            let Some(ps) = sources.get(p.url()).filter(|s| {
                s.ready
                    && s.fresh.elapsed() < super::STALE
                    && s.info["content_namespace"] == namespace
                    && s.info["canonical_block_size_tokens"] == block as u64
            }) else {
                continue;
            };
            let (work, reserved_p) = overlay(p.url());
            let Some((p_ms, plan)) = p_cost(ps, &keys, tokens.len(), work) else {
                continue;
            };
            let p_span = ps.info["hash_block_size_tokens"].as_u64().unwrap_or(0) as usize;
            let Some(p_pages) = pages(tokens.len(), hbm(ps, &keys, tokens.len()), p_span, 0) else {
                continue;
            };
            if ps.load["kv_blocks_free"].as_u64().unwrap_or(0) < (p_pages + reserved_p) as u64 {
                continue;
            }
            for d in decode_workers.iter().filter(|w| healthy(w)) {
                if idx2idx && p.dp_size().unwrap_or(1) > 1 && p.dp_rank() != d.dp_rank() {
                    continue;
                }
                let Some(ds) = sources.get(d.url()).filter(|s| {
                    s.ready
                        && s.fresh.elapsed() < super::STALE
                        && s.info["content_namespace"] == namespace
                        && s.info["canonical_block_size_tokens"] == block as u64
                        && load_fresh(s)
                }) else {
                    continue;
                };
                let Some(paths) = ps.info["transfer_paths"].as_array() else {
                    continue;
                };
                let Some(path) = paths.iter().find(|path| {
                    path["verified"] == true
                        && path["destination_execution_id"] == ds.info["execution_id"]
                        && path["source_layout_id"] == ps.info["layout_id"]
                        && path["destination_layout_id"] == ds.info["layout_id"]
                        && path["protocol"] == "mooncake"
                }) else {
                    continue;
                };
                let d_hit = if ds.info["capabilities"]["pd_delta_receive"] == true {
                    hbm(ds, &keys, tokens.len())
                } else {
                    0
                };
                let d_span = ds.info["hash_block_size_tokens"].as_u64().unwrap_or(0) as usize;
                let Some(d_pages) = pages(tokens.len(), d_hit, d_span, decode_reserve) else {
                    continue;
                };
                if ds.load["kv_blocks_free"].as_u64().unwrap_or(0)
                    < (d_pages + overlay(d.url()).1) as u64
                {
                    continue;
                }
                let Some(transfer) = curve(&path["transfer_curve"], tokens.len() - d_hit) else {
                    continue;
                };
                let Some(step) = positive(&ds.info["costs"]["decode_step_ms"]) else {
                    continue;
                };
                let queue = ds.load["requests_waiting"].as_u64().unwrap_or(0) as f64 * step;
                let score = p_ms + (queue - p_ms).max(0.0) + transfer + step;
                if best
                    .as_ref()
                    .is_none_or(|(cost, _, _, _, _, _)| score < *cost)
                {
                    best = Some((score, p.clone(), d.clone(), plan.clone(), p_pages, d_pages));
                }
            }
        }
        let (_, p, d, plan, p_pages, d_pages) = best?;
        let id = uuid::Uuid::new_v4().to_string();
        reservations.insert(
            id.clone(),
            ReservationEntry {
                p: p.url().into(),
                d: d.url().into(),
                work: tokens.len() - plan.reuse_end + plan.precompute_end - plan.precompute_start,
                p_pages,
                d_pages,
                accepted: HashSet::new(),
                created: Instant::now(),
            },
        );
        Some(PairSelection {
            prefill: p,
            decode: d,
            load_policy: if plan.eligible { "auto" } else { "skip" },
            reservation: Reservation {
                index: self.clone(),
                id,
            },
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        cache_index::protocol::content_keys,
        core::{BasicWorkerBuilder, WorkerType},
    };
    use serde_json::json;
    use std::collections::HashMap;

    const NS: &str = "0101010101010101010101010101010101010101010101010101010101010101";

    fn source(id: &str, role: &str) -> Source {
        let layout = if role == "prefill" {
            "P-layout"
        } else {
            "D-layout"
        };
        Source {
            info: json!({"execution_id": id, "content_namespace": NS,
            "canonical_block_size_tokens": 16, "hash_block_size_tokens": if role == "prefill" { 16 } else { 64 },
            "lmcache_chunk_size_tokens": 256, "min_load_tokens": 0,
            "storage_domain_id": id, "layout_id": layout, "cpu_layout_id": layout,
            "capabilities": {"cache_load_policy_hint": true, "pd_delta_receive": true},
            "costs": {"prefill_curves": [{"max_context_tokens": 16384, "points": [[16384,160.0]]}],
                "h2d_curve": [[16384,20.0]], "decode_step_ms": 2.0},
            "transfer_paths": [
                {"verified": true, "protocol": "mooncake", "destination_execution_id": "d1", "source_layout_id": "P-layout", "destination_layout_id": "D-layout", "transfer_curve": [[16384,80.0]]},
                {"verified": true, "protocol": "mooncake", "destination_execution_id": "d2", "source_layout_id": "P-layout", "destination_layout_id": "D-layout", "transfer_curve": [[16384,80.0]]}
            ]}),
            load: json!({"sample_time_unix_ns": SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos().to_string(),
                "pending_prefill_tokens": 0, "requests_waiting": 0, "kv_blocks_free": 65536,
                "offload": {"loads_pending": 0}, "accepted_dispatch_ids": []}),
            epoch: "boot".into(),
            seq: 0,
            fresh: Instant::now(),
            ready: true,
            entries: HashMap::new(),
        }
    }

    fn add_prefix(source: &mut Source, tier: &str, count: usize) {
        let span = source.info[if tier == "HBM" {
            "hash_block_size_tokens"
        } else {
            "lmcache_chunk_size_tokens"
        }]
        .as_u64()
        .unwrap() as usize;
        let keys = content_keys(NS, &(0..count as u32).collect::<Vec<_>>(), 16).unwrap();
        for start in (0..count).step_by(span) {
            if start + span > count {
                break;
            }
            let entry = json!({"tier": tier, "layout_id": source.info["layout_id"], "execution_id": source.info["execution_id"],
                "storage_domain_id": source.info["storage_domain_id"], "content_namespace": NS,
                "token_start": start, "token_end": start+span, "content_keys": &keys[start/16..(start+span)/16]});
            source
                .entries
                .insert((tier.into(), keys[(start + span) / 16 - 1].clone()), entry);
        }
    }

    fn fixture() -> (
        Arc<CacheIndexService>,
        Vec<Arc<dyn Worker>>,
        Vec<Arc<dyn Worker>>,
    ) {
        let index = Arc::new(CacheIndexService::default());
        let make = |id: &str, role| {
            Arc::new(
                BasicWorkerBuilder::new(format!("http://{id}"))
                    .worker_type(role)
                    .build(),
            ) as Arc<dyn Worker>
        };
        let ps = vec![
            make(
                "p1",
                WorkerType::Prefill {
                    bootstrap_port: None,
                },
            ),
            make(
                "p2",
                WorkerType::Prefill {
                    bootstrap_port: None,
                },
            ),
        ];
        let ds = vec![
            make("d1", WorkerType::Decode),
            make("d2", WorkerType::Decode),
        ];
        for (url, role) in [
            ("p1", "prefill"),
            ("p2", "prefill"),
            ("d1", "decode"),
            ("d2", "decode"),
        ] {
            index
                .sources
                .write()
                .insert(format!("http://{url}"), source(url, role));
        }
        (index, ps, ds)
    }

    #[test]
    fn cpu_prefill_and_hbm_decode_choose_independent_pair() {
        let (index, ps, ds) = fixture();
        add_prefix(
            index.sources.write().get_mut("http://p2").unwrap(),
            "CPU",
            1024,
        );
        add_prefix(
            index.sources.write().get_mut("http://d2").unwrap(),
            "HBM",
            1024,
        );
        let choice = index
            .select_pair(
                &ps,
                &ds,
                &(0..1024).collect::<Vec<_>>(),
                NS,
                16,
                None,
                false,
                64,
            )
            .unwrap();
        assert_eq!(choice.prefill.url(), "http://p2");
        assert_eq!(choice.decode.url(), "http://d2");
        assert_eq!(choice.load_policy, "auto");
        assert_eq!(index.reservations.lock().len(), 1);
        drop(choice);
        assert!(index.reservations.lock().is_empty());
    }

    #[test]
    fn cpu_holes_and_decode_cpu_never_invent_reuse() {
        let (index, ps, ds) = fixture();
        {
            let mut sources = index.sources.write();
            let p = sources.get_mut("http://p2").unwrap();
            add_prefix(p, "CPU", 1024);
            p.entries.retain(|_, e| e["token_start"] != 0);
            add_prefix(sources.get_mut("http://d2").unwrap(), "CPU", 1024);
        }
        let choice = index
            .select_pair(
                &ps,
                &ds,
                &(0..1024).collect::<Vec<_>>(),
                NS,
                16,
                None,
                false,
                64,
            )
            .unwrap();
        assert_eq!(choice.prefill.url(), "http://p1");
        assert_eq!(choice.decode.url(), "http://d1");
        assert_eq!(choice.load_policy, "skip");
    }

    #[test]
    fn stale_source_missing_calibration_and_invalid_path_give_no_benefit() {
        let (index, ps, ds) = fixture();
        for source in index.sources.write().values_mut() {
            source.fresh = Instant::now() - Duration::from_secs(4);
        }
        assert!(index
            .select_pair(&ps, &ds, &[1; 1024], NS, 16, None, false, 64)
            .is_none());
        for source in index.sources.write().values_mut() {
            source.fresh = Instant::now();
            source.info["transfer_paths"] = json!([]);
        }
        assert!(index
            .select_pair(&ps, &ds, &[1; 1024], NS, 16, None, false, 64)
            .is_none());
        assert_eq!(curve(&json!([[16, 1.0], [32, 2.0]]), 64), None);
    }

    #[test]
    fn reservations_prevent_burst_herding_and_release_on_cancel() {
        let (index, ps, ds) = fixture();
        let first = index
            .select_pair(&ps, &ds, &[1; 1024], NS, 16, None, false, 64)
            .unwrap();
        let second = index
            .select_pair(&ps, &ds, &[1; 1024], NS, 16, None, false, 64)
            .unwrap();
        assert_ne!(first.prefill.url(), second.prefill.url());
        drop(first);
        drop(second);
        assert!(index.reservations.lock().is_empty());
    }
}
