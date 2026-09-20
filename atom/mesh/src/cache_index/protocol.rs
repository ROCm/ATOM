use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

pub const PLANNER_VERSION: &str = "page-reuse-v1";

pub fn content_keys(namespace: &str, tokens: &[u32], block: usize) -> Option<Vec<String>> {
    if namespace.len() != 64 || block == 0 || block > u32::MAX as usize {
        return None;
    }
    let namespace: Vec<u8> = (0..64)
        .step_by(2)
        .map(|i| u8::from_str_radix(&namespace[i..i + 2], 16))
        .collect::<Result<_, _>>()
        .ok()?;
    let mut hash = Sha256::new();
    hash.update(b"atom-kv-content-v1\0");
    hash.update(namespace);
    hash.update((block as u32).to_le_bytes());
    let mut parent = hash.finalize().to_vec();
    let mut result = Vec::with_capacity(tokens.len() / block);
    for chunk in tokens.chunks_exact(block) {
        let mut hash = Sha256::new();
        hash.update(b"block\0");
        hash.update(&parent);
        for token in chunk {
            hash.update(token.to_le_bytes());
        }
        parent = hash.finalize().to_vec();
        result.push(parent.iter().map(|b| format!("{b:02x}")).collect());
    }
    Some(result)
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ReusePlan {
    pub kind: String,
    pub precompute_start: usize,
    pub precompute_end: usize,
    pub load_start: usize,
    pub transfer_end: usize,
    pub reuse_end: usize,
    pub eligible: bool,
    pub reject_reason: String,
    pub planner_version: String,
}

pub fn plan_reuse(
    n: usize,
    h: usize,
    cpu: usize,
    chunk: usize,
    minimum: usize,
    allow_cpu: bool,
) -> Option<ReusePlan> {
    if chunk == 0 {
        return None;
    }
    let h = h.min(n.saturating_sub(1));
    let end = cpu.min(n.saturating_sub(1)) / chunk * chunk;
    let boundary = h.div_ceil(chunk).checked_mul(chunk)?;
    let reason = if !allow_cpu {
        "cpu_disabled"
    } else if end <= boundary {
        "no_loadable_suffix"
    } else if end - boundary < minimum {
        "too_small"
    } else {
        ""
    };
    let eligible = reason.is_empty();
    Some(ReusePlan {
        kind: if !eligible {
            "hbm"
        } else if boundary == h {
            "cpu"
        } else {
            "precompute_cpu"
        }
        .into(),
        precompute_start: h,
        precompute_end: if eligible { boundary } else { h },
        load_start: if eligible { boundary } else { h },
        transfer_end: if eligible { end } else { h },
        reuse_end: if eligible { end } else { h },
        eligible,
        reject_reason: reason.into(),
        planner_version: PLANNER_VERSION.into(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn python_golden_vectors() {
        let data: serde_json::Value = serde_json::from_str(include_str!(
            "../../../../tests/fixtures/cache_routing_golden.json"
        ))
        .unwrap();
        for row in data["hashes"].as_array().unwrap() {
            let tokens: Vec<u32> = serde_json::from_value(row["tokens"].clone()).unwrap();
            let keys = content_keys(
                row["namespace"].as_str().unwrap(),
                &tokens,
                row["block_size"].as_u64().unwrap() as usize,
            )
            .unwrap();
            assert_eq!(serde_json::to_value(keys).unwrap(), row["keys"]);
        }
        for row in data["plans"].as_array().unwrap() {
            let n = |key: &str| row[key].as_u64().unwrap() as usize;
            let plan = plan_reuse(
                n("prompt_tokens"),
                n("hbm_prefix"),
                n("cpu_prefix"),
                n("chunk"),
                n("min_load"),
                true,
            )
            .unwrap();
            assert_eq!(serde_json::to_value(plan).unwrap(), row["plan"]);
        }
    }
}
