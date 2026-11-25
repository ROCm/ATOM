//! A Mooncake master's admin service (its `--metrics_port`), whose
//! `GET /batch_query_keys?keys=k1,k2` lists each key's replicas without leasing them.

use std::{collections::HashMap, time::Duration};

use serde::{de::IgnoredAny, Deserialize};

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
    timeout: Duration,
) -> Result<Vec<KeyState>, String> {
    let url = format!("{admin}/batch_query_keys");
    let response = client
        .get(&url)
        .query(&[("keys", keys.join(","))])
        .timeout(timeout)
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
    use serde_json::json;

    use super::*;

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
