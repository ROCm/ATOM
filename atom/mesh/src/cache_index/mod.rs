//! Background catalog recovery. Selection reads local state and never probes engines.
pub mod protocol;
pub mod scoring;

use crate::core::{Worker, WorkerRegistry};
use parking_lot::RwLock;
use reqwest::Client;
use serde_json::Value;
use std::{
    collections::{HashMap, HashSet},
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc, LazyLock,
    },
    time::{Duration, Instant},
};

const MAX_ENTRIES: usize = 200_000;
const MAX_PAGE_BYTES: u64 = 16 * 1024 * 1024;
const STALE: Duration = Duration::from_secs(3);

#[derive(Debug)]
pub struct Source {
    pub info: Value,
    pub load: Value,
    pub epoch: String,
    pub seq: u64,
    pub fresh: Instant,
    pub ready: bool,
    pub entries: HashMap<(String, String), Value>,
}

impl Source {
    pub fn prefix(&self, keys: &[String], tier: &str) -> usize {
        if !self.ready || self.fresh.elapsed() > STALE {
            return 0;
        }
        let block = self.info["canonical_block_size_tokens"]
            .as_u64()
            .unwrap_or(0) as usize;
        let span = self.info[if tier == "HBM" {
            "hash_block_size_tokens"
        } else {
            "lmcache_chunk_size_tokens"
        }]
        .as_u64()
        .unwrap_or(0) as usize;
        if block == 0 || span == 0 || span % block != 0 {
            return 0;
        }
        let width = span / block;
        let mut end = 0;
        while end + width <= keys.len() {
            let Some(entry) = self
                .entries
                .get(&(tier.to_string(), keys[end + width - 1].clone()))
            else {
                break;
            };
            let expected_layout = &self.info[if tier == "HBM" {
                "layout_id"
            } else {
                "cpu_layout_id"
            }];
            let identity = if tier == "HBM" {
                "execution_id"
            } else {
                "storage_domain_id"
            };
            if entry["layout_id"] != *expected_layout
                || entry[identity] != self.info[identity]
                || entry["content_namespace"] != self.info["content_namespace"]
                || entry["token_start"].as_u64() != Some((end * block) as u64)
                || entry["token_end"].as_u64() != Some(((end + width) * block) as u64)
                || entry["content_keys"] != serde_json::json!(&keys[end..end + width])
            {
                break;
            }
            end += width;
        }
        end * block
    }
}

#[derive(Debug, Default)]
pub struct CacheIndexService {
    pub sources: RwLock<HashMap<String, Source>>,
    pub reservations: parking_lot::Mutex<HashMap<String, scoring::ReservationEntry>>,
    started: AtomicBool,
}

pub static CACHE_INDEX: LazyLock<Arc<CacheIndexService>> =
    LazyLock::new(|| Arc::new(CacheIndexService::default()));

// Calibration is independent of model startup. Restarting only the router can
// install new measurements; execution/layout binding prevents using old curves
// after a model, PP partition or DCP geometry changes.
static CALIBRATION: LazyLock<Result<Value, String>> =
    LazyLock::new(|| match std::env::var("ATOM_CACHE_ROUTING_CALIBRATION") {
        Ok(path) => std::fs::read_to_string(path)
            .map_err(|error| error.to_string())
            .and_then(|data| serde_json::from_str(&data).map_err(|error| error.to_string())),
        Err(std::env::VarError::NotPresent) => Ok(Value::Null),
        Err(error) => Err(error.to_string()),
    });

fn apply_calibration(info: &mut Value, calibration: &Value) -> Result<(), String> {
    let id = info["execution_id"]
        .as_str()
        .ok_or("missing execution identity")?;
    let Some(entry) = calibration["executions"].get(id) else {
        return Ok(());
    };
    if entry["layout_id"] != info["layout_id"] || entry["layout_id"].as_str().is_none() {
        return Err("calibration layout differs from execution".into());
    }
    info["costs"] = entry["costs"].clone();
    info["transfer_paths"] = entry["transfer_paths"].clone();
    Ok(())
}

fn cursor(value: &Value, key: &str) -> Result<u64, String> {
    value[key]
        .as_str()
        .and_then(|s| s.parse().ok())
        .ok_or_else(|| format!("invalid {key}"))
}

fn entry_key(entry: &Value) -> Result<(String, String), String> {
    let tier = entry["tier"].as_str().ok_or("missing tier")?;
    if !matches!(tier, "HBM" | "CPU") {
        return Err("invalid tier".into());
    }
    let key = entry["content_keys"]
        .as_array()
        .and_then(|a| a.last())
        .and_then(Value::as_str)
        .ok_or("missing content key")?;
    if key.len() != 64 {
        return Err("invalid content key".into());
    }
    Ok((tier.to_string(), key.to_string()))
}

async fn get(client: &Client, worker: &dyn Worker, url: &str) -> Result<Value, String> {
    let mut request = client.get(url).timeout(Duration::from_secs(1));
    if let Some(key) = worker.api_key() {
        request = request.bearer_auth(key);
    }
    let response = request
        .send()
        .await
        .map_err(|e| e.to_string())?
        .error_for_status()
        .map_err(|e| e.to_string())?;
    if response.content_length().unwrap_or(MAX_PAGE_BYTES + 1) > MAX_PAGE_BYTES {
        return Err("catalog page exceeds budget".into());
    }
    response.json().await.map_err(|e| e.to_string())
}

impl CacheIndexService {
    pub fn start(self: &Arc<Self>, registry: &Arc<WorkerRegistry>, client: Client) {
        if self.started.swap(true, Ordering::AcqRel) {
            return;
        }
        let index = Arc::downgrade(self);
        let registry = Arc::downgrade(registry);
        tokio::spawn(async move {
            loop {
                let (Some(index), Some(registry)) = (index.upgrade(), registry.upgrade()) else {
                    break;
                };
                let workers = registry.get_all();
                drop(registry);
                let urls: HashSet<_> = workers.iter().map(|w| w.url().to_string()).collect();
                index.sources.write().retain(|url, _| urls.contains(url));
                // Network work is concurrent and entirely outside selection locks.
                futures::future::join_all(workers.iter().map(|worker| async {
                    if let Err(error) = index.refresh(&client, worker.as_ref()).await {
                        if let Some(source) = index.sources.write().get_mut(worker.url()) { source.ready = false; }
                        tracing::debug!(worker = worker.url(), %error, "cache catalog unavailable; using load routing");
                    }
                })).await;
                tokio::time::sleep(Duration::from_millis(100)).await;
            }
            if let Some(index) = index.upgrade() {
                index.started.store(false, Ordering::Release);
            }
        });
    }

    async fn refresh(&self, client: &Client, worker: &dyn Worker) -> Result<(), String> {
        let server = get(
            client,
            worker,
            &format!("{}/server_info", worker.base_url()),
        )
        .await?;
        let base = server["cache_routing"]["catalog_http"]
            .as_str()
            .ok_or("catalog discovery absent")?;
        let mut info = get(client, worker, &format!("{base}/info")).await?;
        apply_calibration(&mut info, CALIBRATION.as_ref().map_err(Clone::clone)?)?;
        if info["capabilities"]["exact_prefix_reuse"] != true
            || info["canonical_hash"] != "sha256-prefix-u32le-v1"
        {
            return Err("exact cache capability absent".into());
        }
        let epoch = info["source_epoch"]
            .as_str()
            .ok_or("missing epoch")?
            .to_string();
        let current = self
            .sources
            .read()
            .get(worker.url())
            .filter(|s| s.ready && s.epoch == epoch)
            .map(|s| s.seq);
        if let Some(after) = current {
            let page = get(
                client,
                worker,
                &format!("{base}/events?source_epoch={epoch}&after_seq={after}"),
            )
            .await?;
            let events = Self::validate_events(&page, &epoch, after)?;
            let load = get(client, worker, &format!("{base}/load")).await?;
            let mut sources = self.sources.write();
            let source = sources.get_mut(worker.url()).ok_or("source removed")?;
            if source.epoch != epoch || source.seq != after {
                return Err("concurrent source change".into());
            }
            Self::apply(&mut source.entries, events)?;
            source.seq = cursor(&page, "cut_seq")?;
            source.info = info;
            source.load = load;
            source.fresh = Instant::now();
        } else {
            if let Some(source) = self.sources.write().get_mut(worker.url()) {
                source.ready = false;
            }
            let mut page = get(client, worker, &format!("{base}/snapshot")).await?;
            let id = page["snapshot_id"]
                .as_str()
                .ok_or("missing snapshot id")?
                .to_string();
            let cut = cursor(&page, "cut_seq")?;
            let mut entries = HashMap::new();
            let started = Instant::now();
            loop {
                if page["source_epoch"] != epoch
                    || page["snapshot_id"] != id
                    || cursor(&page, "cut_seq")? != cut
                    || page["complete"] != true
                {
                    return Err("inconsistent snapshot".into());
                }
                for entry in page["entries"].as_array().ok_or("missing entries")? {
                    entries.insert(entry_key(entry)?, entry.clone());
                }
                if entries.len() > MAX_ENTRIES || started.elapsed() > Duration::from_secs(8) {
                    return Err("snapshot budget exceeded".into());
                }
                let Some(next) = page["next_page_token"].as_str() else {
                    break;
                };
                page = get(
                    client,
                    worker,
                    &format!("{base}/snapshot?snapshot_id={id}&page_token={next}"),
                )
                .await?;
            }
            let page = get(
                client,
                worker,
                &format!("{base}/events?source_epoch={epoch}&after_seq={cut}"),
            )
            .await?;
            Self::apply(&mut entries, Self::validate_events(&page, &epoch, cut)?)?;
            let load = get(client, worker, &format!("{base}/load")).await?;
            let source = Source {
                info,
                load,
                epoch,
                seq: cursor(&page, "cut_seq")?,
                entries,
                fresh: Instant::now(),
                ready: true,
            };
            self.sources
                .write()
                .insert(worker.url().to_string(), source);
        }
        Ok(())
    }

    fn validate_events<'a>(
        page: &'a Value,
        epoch: &str,
        after: u64,
    ) -> Result<Vec<&'a Value>, String> {
        if page["source_epoch"] != epoch || page["complete"] != true {
            return Err("stale catalog epoch".into());
        }
        let mut seq = after;
        let mut events = Vec::new();
        for batch in page["events"].as_array().ok_or("missing events")? {
            seq += 1;
            if batch["source_epoch"] != epoch || cursor(batch, "seq")? != seq {
                return Err("catalog sequence gap".into());
            }
            events.extend(batch["events"].as_array().ok_or("missing batch events")?);
        }
        if cursor(page, "cut_seq")? != seq {
            return Err("missing final catalog mutation".into());
        }
        Ok(events)
    }

    fn apply(
        entries: &mut HashMap<(String, String), Value>,
        events: Vec<&Value>,
    ) -> Result<(), String> {
        for event in events {
            match event["type"].as_str() {
                Some("residency_upsert") => {
                    entries.insert(entry_key(event)?, event.clone());
                }
                Some("residency_remove") => {
                    entries.remove(&entry_key(event)?);
                }
                Some("heartbeat" | "content_define") => {}
                _ => return Err("unsupported catalog mutation".into()),
            }
            if entries.len() > MAX_ENTRIES {
                return Err("catalog memory budget exceeded".into());
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn replay_rejects_gap_old_epoch_and_missing_last_mutation() {
        let page = json!({"source_epoch":"boot", "cut_seq":"2", "complete":true,
            "events":[{"source_epoch":"boot","seq":"2","events":[]}]});
        assert!(CacheIndexService::validate_events(&page, "boot", 0).is_err());
        assert!(CacheIndexService::validate_events(&page, "old", 1).is_err());
        assert!(CacheIndexService::validate_events(&page, "boot", 1).is_ok());
        let mut missing = page;
        missing["cut_seq"] = json!("3");
        assert!(CacheIndexService::validate_events(&missing, "boot", 1).is_err());
    }

    #[test]
    fn calibration_is_bound_to_execution_layout() {
        let calibration =
            json!({"executions":{"p":{"layout_id":"layout-v1","costs":{"decode_step_ms":2}}}});
        let mut info = json!({"execution_id":"p","layout_id":"layout-v2"});
        assert!(apply_calibration(&mut info, &calibration).is_err());
        info["layout_id"] = json!("layout-v1");
        apply_calibration(&mut info, &calibration).unwrap();
        assert_eq!(info["costs"]["decode_step_ms"], 2);
    }
}
