//! A Mooncake master's admin service (its `--metrics_port`), whose
//! `GET /batch_query_keys?keys=k1,k2` lists each key's replicas without leasing them.

use std::{collections::HashMap, time::Duration};

use futures::future::join_all;
use serde::{de::IgnoredAny, Deserialize};

use super::keys::TAINT_KEY;

/// Keys a request asks for at most, besides the taint key: they go in its URL.
const BATCH_KEYS: usize = 64;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KeyState {
    Memory,
    /// Replicas on disk only: `disk_values`, `local_disk_values` or `nof_values`.
    SsdOnly,
    /// Not found, or not written yet.
    Absent,
    /// Left out of the answer.
    Unknown,
}

impl KeyState {
    pub fn is_held(self) -> bool {
        matches!(self, Self::Memory | Self::SsdOnly)
    }

    /// A key's state by two pools' answers: held by either, in memory first.
    fn or(self, other: Self) -> Self {
        use KeyState::*;
        match (self, other) {
            (Memory, _) | (_, Memory) => Memory,
            (SsdOnly, _) | (_, SsdOnly) => SsdOnly,
            (Absent, _) | (_, Absent) => Absent,
            _ => Unknown,
        }
    }
}

/// The state of each of `keys` in the pools whose admin services are `admins`, asked
/// concurrently in batches, for as long as the caller waits: unknown only where no pool
/// answered. A pool that holds the taint key holds nothing, as the engine loads nothing
/// from it, and an answer that leaves the taint key out says nothing.
pub async fn query_pools(
    client: &reqwest::Client,
    admins: &[String],
    keys: &[String],
) -> Vec<KeyState> {
    let batches: Vec<Vec<String>> = keys
        .chunks(BATCH_KEYS)
        .map(|batch| batch.iter().cloned().chain([TAINT_KEY.into()]).collect())
        .collect();
    let asked = admins.iter().flat_map(|admin| {
        let batches = batches.iter();
        batches.map(move |batch| query(client, admin, batch, None))
    });
    let answers = join_all(asked).await;
    let mut merged = vec![KeyState::Unknown; keys.len()];
    for pool in answers.chunks(batches.len().max(1)) {
        let taint = |states: &Vec<KeyState>| states.last().is_some_and(|s| s.is_held());
        let tainted = pool.iter().flatten().any(taint);
        for (answer, first) in pool.iter().zip((0..).step_by(BATCH_KEYS)) {
            let answered = answer.as_ref().ok().and_then(|s| s.split_last());
            let Some((KeyState::Memory | KeyState::SsdOnly | KeyState::Absent, states)) = answered
            else {
                continue;
            };
            for (state, merged) in states.iter().zip(&mut merged[first..]) {
                let state = if tainted { KeyState::Absent } else { *state };
                *merged = merged.or(state);
            }
        }
    }
    merged
}

/// The admin service of the master at `master`, its RPC `host:port`.
pub fn admin_url(master: &str, port: u16) -> String {
    let host = master.rsplit_once(':').map_or(master, |(host, _)| host);
    format!("http://{host}:{port}")
}

/// The state of each of `keys`, in their order, as the service at `admin` answers.
pub async fn query(
    client: &reqwest::Client,
    admin: &str,
    keys: &[String],
    timeout: Option<Duration>,
) -> Result<Vec<KeyState>, String> {
    let url = format!("{admin}/batch_query_keys");
    let mut request = client.get(&url).query(&[("keys", keys.join(","))]);
    if let Some(timeout) = timeout {
        request = request.timeout(timeout);
    }
    let response = request
        .send()
        .await
        .and_then(reqwest::Response::error_for_status)
        .map_err(|e| format!("{url}: {e}"))?;
    let answer = response.json().await.map_err(|e| format!("{url}: {e}"))?;
    states(answer, keys).map_err(|e| format!("{url}: {e}"))
}

#[derive(Deserialize)]
struct Answer {
    success: bool,
    #[serde(default)]
    data: HashMap<String, Replicas>,
    #[serde(default)]
    error: Option<String>,
}

#[derive(Deserialize)]
struct Replicas {
    ok: bool,
    #[serde(default)]
    values: Option<Vec<IgnoredAny>>,
    #[serde(default)]
    disk_values: Option<Vec<IgnoredAny>>,
    #[serde(default)]
    local_disk_values: Option<Vec<IgnoredAny>>,
    #[serde(default)]
    nof_values: Option<Vec<IgnoredAny>>,
}

impl Replicas {
    fn state(&self) -> KeyState {
        let any =
            |replicas: &Option<Vec<IgnoredAny>>| replicas.as_ref().is_some_and(|r| !r.is_empty());
        if !self.ok {
            KeyState::Absent
        } else if any(&self.values) {
            KeyState::Memory
        } else if any(&self.disk_values) || any(&self.local_disk_values) || any(&self.nof_values) {
            KeyState::SsdOnly
        } else {
            KeyState::Absent
        }
    }
}

fn states(answer: Answer, keys: &[String]) -> Result<Vec<KeyState>, String> {
    if !answer.success {
        return Err(answer.error.unwrap_or_else(|| "no success".into()));
    }
    let state = |key: &String| {
        answer
            .data
            .get(key)
            .map_or(KeyState::Unknown, Replicas::state)
    };
    Ok(keys.iter().map(state).collect())
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use axum::{extract::Query, routing::get, Json, Router};
    use parking_lot::Mutex;
    use serde_json::{json, Map, Value};

    use super::*;

    /// A master's admin service whose pool holds `held`, each key's state, and that
    /// records how many keys each request asks for.
    async fn admin(held: HashMap<String, KeyState>, asked: Arc<Mutex<Vec<usize>>>) -> String {
        let answer = move |Query(query): Query<HashMap<String, String>>| {
            let keys: Vec<String> = query["keys"].split(',').map(String::from).collect();
            asked.lock().push(keys.len());
            let replicas = |key: &String| match held.get(key).copied() {
                Some(KeyState::Memory) => Some(json!({"ok": true, "values": [{}]})),
                Some(KeyState::SsdOnly) => Some(json!({"ok": true, "disk_values": [{}]})),
                Some(KeyState::Unknown) => None,
                _ => Some(json!({"ok": false, "error": "OBJECT_NOT_FOUND"})),
            };
            let data: Map<String, Value> = (keys.iter())
                .filter_map(|key| Some((key.clone(), replicas(key)?)))
                .collect();
            async move { Json(json!({"success": true, "data": data})) }
        };
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        let service = Router::new().route("/batch_query_keys", get(answer));
        tokio::spawn(async move { axum::serve(listener, service).await });
        url
    }

    #[tokio::test]
    async fn pools_are_asked_in_batches_and_merged() {
        let key = |i: usize| format!("k{i}");
        let pool = |held: &[(usize, KeyState)]| -> HashMap<String, KeyState> {
            held.iter().map(|&(i, state)| (key(i), state)).collect()
        };
        use KeyState::*;
        let asked: Vec<_> = (0..4).map(|_| Arc::new(Mutex::new(Vec::new()))).collect();
        let mut tainted = pool(&[(3, Memory)]);
        tainted.insert(TAINT_KEY.into(), Memory);
        let mut mute = pool(&[(4, Memory)]);
        mute.insert(TAINT_KEY.into(), Unknown);
        let closed = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let dead = format!("http://{}", closed.local_addr().unwrap());
        drop(closed);
        let admins = [
            admin(
                pool(&[(0, Memory), (1, SsdOnly), (65, Memory), (66, Unknown)]),
                asked[0].clone(),
            )
            .await,
            admin(pool(&[(1, Memory), (2, SsdOnly)]), asked[1].clone()).await,
            admin(tainted, asked[2].clone()).await,
            admin(mute, asked[3].clone()).await,
            dead.clone(),
        ];
        let keys: Vec<String> = (0..70).map(key).collect();
        let client = reqwest::Client::new();
        let got = query_pools(&client, &admins, &keys).await;
        let mut want = vec![Absent; 70];
        (want[0], want[1], want[2], want[65]) = (Memory, Memory, SsdOnly, Memory);
        assert_eq!(got, want);
        for asked in asked {
            let mut asked = asked.lock().clone();
            asked.sort();
            assert_eq!(asked, [7, 65]);
        }
        let none = query_pools(&client, &[dead], &keys).await;
        assert_eq!(none, vec![Unknown; 70]);
    }

    #[test]
    fn an_admin_service_is_at_the_masters_host() {
        for (master, want) in [
            ("10.0.0.9:50051", "http://10.0.0.9:9003"),
            ("[fd00::1]:50051", "http://[fd00::1]:9003"),
            ("master-0", "http://master-0:9003"),
        ] {
            assert_eq!(admin_url(master, 9003), want);
        }
    }

    // As mooncake_master 0.3.14 answers, with replicas on disk added.
    #[test]
    fn each_key_is_in_memory_on_disk_absent_or_unknown() {
        let replica = json!({"size_": 4096, "buffer_address_": 1, "protocol_": "tcp",
            "transport_endpoint_": "127.0.0.1:13874"});
        let none = json!({"ok": false, "error": "OBJECT_NOT_FOUND", "values": null,
            "disk_values": null, "local_disk_values": null, "nof_values": null});
        let answer = json!({"success": true, "data": {
            "memory": {"ok": true, "error": null, "values": [replica], "disk_values": null},
            "disk": {"ok": true, "values": [], "disk_values": [{"file_path": "/d", "object_size": 1}]},
            "local_disk": {"ok": true, "values": [], "local_disk_values": [{}]},
            "nof": {"ok": true, "values": [], "nof_values": [replica]},
            "none": none,
            "writing": {"ok": false, "error": "REPLICA_IS_NOT_READY"},
            "empty": {"ok": true, "values": []},
        }});
        let keys: Vec<String> = [
            "memory",
            "disk",
            "local_disk",
            "nof",
            "none",
            "writing",
            "empty",
            "unasked",
        ]
        .map(String::from)
        .into();
        let got = states(serde_json::from_value(answer).unwrap(), &keys).unwrap();
        use KeyState::*;
        assert_eq!(
            got,
            [Memory, SsdOnly, SsdOnly, SsdOnly, Absent, Absent, Absent, Unknown]
        );
        let failed =
            json!({"success": false, "error": "No keys provided. Use ?keys=key1,key2,..."});
        let failed = states(serde_json::from_value(failed).unwrap(), &keys).unwrap_err();
        assert!(failed.starts_with("No keys"), "{failed}");
    }
}
