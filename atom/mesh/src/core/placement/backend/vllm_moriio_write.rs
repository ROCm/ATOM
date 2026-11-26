//! Native MoRIIO WRITE: both peers are known before concurrent P/D dispatch.
use std::collections::HashMap;

use serde_json::{json, Value};
use uuid::Uuid;

use super::super::types::AdapterError;
use super::{vllm::inject_prefill, BackendAdapter, PairCtx};
use crate::config::VllmMoriioPeer;
use crate::core::Worker;

#[derive(Debug)]
pub struct VllmMoriioWriteAdapter {
    peers: HashMap<String, VllmMoriioPeer>,
}

#[derive(Debug)]
struct WritePair {
    transfer_id: String,
    prefill: VllmMoriioPeer,
    decode: VllmMoriioPeer,
}

fn pair(ctx: &PairCtx) -> Result<&WritePair, AdapterError> {
    ctx.downcast_ref().ok_or(AdapterError::CtxTypeMismatch)
}

impl VllmMoriioWriteAdapter {
    pub fn new(peers: Vec<VllmMoriioPeer>) -> Self {
        Self {
            peers: peers
                .into_iter()
                .map(|peer| (peer.url.trim_end_matches('/').to_string(), peer))
                .collect(),
        }
    }

    fn peer(&self, worker: &dyn Worker) -> Result<VllmMoriioPeer, AdapterError> {
        self.peers
            .get(worker.base_url().trim_end_matches('/'))
            .cloned()
            .ok_or_else(|| {
                AdapterError::UnsupportedTopology(format!(
                    "Missing MoRIIO WRITE peer metadata for {}",
                    worker.base_url()
                ))
            })
    }

    fn metadata(peer: &VllmMoriioPeer, transfer_id: &str, producer: bool) -> Value {
        json!({
            "do_remote_decode": producer,
            "do_remote_prefill": !producer,
            "transfer_id": transfer_id,
            "remote_engine_id": format!("{}:{}", peer.host, peer.handshake_port),
            "remote_block_ids": Value::Null,
            "remote_host": peer.host,
            "remote_handshake_port": peer.handshake_port,
            "remote_notify_port": peer.notify_port,
            "remote_tp_size": peer.tp_size,
            "tp_size": peer.tp_size,
            "remote_dp_size": peer.dp_size,
            "remote_dp_size_local": peer.dp_size,
            "remote_dp_rank": 0,
            // P has one DP rank. D's own scheduler reports its actual allocation
            // rank; comparing it with P's rank would suppress DP1..N notifications.
            "remote_dp_rank_override": false,
        })
    }
}

impl BackendAdapter for VllmMoriioWriteAdapter {
    fn prepare_pair(
        &self,
        prefill: &dyn Worker,
        decode: &dyn Worker,
    ) -> Result<PairCtx, AdapterError> {
        let prefill = self.peer(prefill)?;
        let decode = self.peer(decode)?;
        if prefill.dp_size != 1 || prefill.tp_size != decode.tp_size {
            return Err(AdapterError::UnsupportedTopology(
                "MoRIIO WRITE requires prefill DP=1 and matching TP; decode may use a native local DP pool".into(),
            ));
        }
        Ok(Box::new(WritePair {
            transfer_id: format!("tx-{}", Uuid::new_v4()),
            prefill,
            decode,
        }))
    }

    fn inject_prefill_fields(&self, body: &mut Value, ctx: &PairCtx) -> Result<(), AdapterError> {
        if ["n", "best_of"].iter().any(|key| {
            body.get(key)
                .is_some_and(|v| !v.is_null() && v.as_u64() != Some(1))
        }) || body["prompt"]
            .as_array()
            .is_some_and(|items| items.iter().any(|item| item.as_u64().is_none()))
        {
            return Err(AdapterError::UnsupportedTopology(
                "MoRIIO WRITE requires one prompt and one completion".into(),
            ));
        }
        let ctx = pair(ctx)?;
        inject_prefill(body, &ctx.transfer_id)?;
        body["request_id"] = ctx.transfer_id.clone().into();
        body["min_tokens"] = 1.into();
        body["ignore_eos"] = true.into();
        let obj = body.as_object_mut().ok_or(AdapterError::BodyNotObject)?;
        for key in ["stop", "stop_token_ids", "data_parallel_rank"] {
            obj.remove(key);
        }
        obj.insert(
            "kv_transfer_params".into(),
            Self::metadata(&ctx.decode, &ctx.transfer_id, true),
        );
        Ok(())
    }

    fn inject_decode_fields(&self, body: &mut Value, ctx: &PairCtx) -> Result<(), AdapterError> {
        let ctx = pair(ctx)?;
        let obj = body.as_object_mut().ok_or(AdapterError::BodyNotObject)?;
        obj.remove("data_parallel_rank");
        obj.insert("request_id".into(), ctx.transfer_id.clone().into());
        obj.insert(
            "kv_transfer_params".into(),
            Self::metadata(&ctx.prefill, &ctx.transfer_id, false),
        );
        Ok(())
    }

    fn inject_batch_prefill_fields(
        &self,
        body: &mut Value,
        ctx: &PairCtx,
        batch_size: usize,
    ) -> Result<(), AdapterError> {
        if batch_size != 1 {
            return Err(AdapterError::UnsupportedTopology(
                "MoRIIO WRITE requires one prompt per request".into(),
            ));
        }
        self.inject_prefill_fields(body, ctx)
    }

    fn correlation_id(&self, ctx: &PairCtx) -> Option<String> {
        pair(ctx).ok().map(|ctx| ctx.transfer_id.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn peer(url: &str, host: &str, port: u16, dp_size: usize) -> VllmMoriioPeer {
        VllmMoriioPeer {
            url: url.into(),
            host: host.into(),
            handshake_port: port,
            notify_port: port + 100,
            tp_size: 1,
            dp_size,
        }
    }

    #[test]
    fn write_supplies_both_peers_before_either_request_and_keeps_decode_generation() {
        let ctx: PairCtx = Box::new(WritePair {
            transfer_id: "tx-test".into(),
            prefill: peer("http://p:8000", "p", 6300, 1),
            decode: peer("http://d:8000", "d", 6400, 4),
        });
        let adapter = VllmMoriioWriteAdapter::new(Vec::new());
        let original = json!({"messages":[{"role":"user","content":"hello"}],
            "stream":true,"stream_options":{"include_usage":true},
            "tools":[{"type":"function"}],"max_tokens":64,"stop":["end"],
            "data_parallel_rank":99,"kv_transfer_params":{"remote_host":"untrusted"}});
        let mut p = original.clone();
        let mut d = original.clone();
        adapter.inject_prefill_fields(&mut p, &ctx).unwrap();
        adapter.inject_decode_fields(&mut d, &ctx).unwrap();
        assert_eq!(p["kv_transfer_params"]["remote_host"], "d");
        assert_eq!(p["kv_transfer_params"]["remote_dp_size"], 4);
        assert_eq!(p["kv_transfer_params"]["remote_dp_size_local"], 4);
        assert_eq!(d["kv_transfer_params"]["remote_host"], "p");
        assert_eq!(d["kv_transfer_params"]["remote_dp_size"], 1);
        assert_eq!(d["kv_transfer_params"]["remote_dp_rank"], 0);
        assert_eq!(d["kv_transfer_params"]["remote_dp_rank_override"], false);
        assert!(p["kv_transfer_params"]["remote_block_ids"].is_null());
        assert_eq!(p["request_id"], d["request_id"]);
        assert_eq!(p["request_id"], "tx-test");
        assert_eq!(p["max_tokens"], 1);
        assert_eq!(p["stream"], false);
        assert!(p.get("data_parallel_rank").is_none());
        assert!(d.get("data_parallel_rank").is_none());
        for key in [
            "messages",
            "stream",
            "stream_options",
            "tools",
            "max_tokens",
            "stop",
        ] {
            assert_eq!(d[key], original[key]);
        }
        assert!(adapter
            .inject_batch_prefill_fields(&mut p, &ctx, 2)
            .is_err());
    }
}
