//! Native vLLM MoRI-IO READ using explicit, producer-returned metadata.
use serde_json::Value;
use uuid::Uuid;

use super::super::types::AdapterError;
use super::{vllm::inject_prefill, BackendAdapter, PairCtx};
use crate::core::Worker;

#[derive(Debug)]
pub struct VllmMoriioAdapter;

#[derive(Debug)]
struct MoriioPairCtx {
    transfer_id: String,
}

fn downcast(ctx: &PairCtx) -> Result<&MoriioPairCtx, AdapterError> {
    ctx.downcast_ref().ok_or(AdapterError::CtxTypeMismatch)
}

impl VllmMoriioAdapter {
    /// Fail closed before D can fall back to local prefill or read the wrong peer.
    /// Keep the complete object, including grouped/hybrid block IDs and extension fields.
    pub fn validate_decode_kv(kv: &Value, ctx: &PairCtx) -> Result<(), String> {
        let ctx = downcast(ctx).map_err(|e| e.to_string())?;
        let obj = kv
            .as_object()
            .ok_or("kv_transfer_params must be an object")?;
        if kv["transfer_id"].as_str() != Some(&ctx.transfer_id)
            || kv["do_remote_prefill"] != true
            || kv["do_remote_decode"] != false
        {
            return Err(
                "MoRI-IO READ requires matching transfer_id and remote-prefill flags".into(),
            );
        }
        for key in ["remote_engine_id", "remote_host"] {
            if !obj
                .get(key)
                .and_then(Value::as_str)
                .is_some_and(|s| !s.trim().is_empty())
            {
                return Err(format!("MoRI-IO READ requires nonempty {key}"));
            }
        }
        for key in ["remote_handshake_port", "remote_notify_port"] {
            // vLLM returns configured ports verbatim; srt-slurm supplies strings.
            let port = kv[key].as_u64().or_else(|| kv[key].as_str()?.parse().ok());
            if !port.is_some_and(|p| (1..=65535).contains(&p)) {
                return Err(format!("invalid MoRI-IO {key}"));
            }
        }
        if !kv["tp_size"].as_u64().is_some_and(|size| size > 0)
            || kv["remote_dp_size"].as_u64() != Some(1)
            || !matches!(kv["remote_dp_size_local"].as_u64(), Some(0 | 1))
            || kv["remote_dp_rank"].as_u64() != Some(0)
            || kv["remote_dp_rank_override"] != true
        {
            return Err(
                "MoRI-IO READ requires explicit TP metadata and DP=1 rank ownership".into(),
            );
        }
        let groups = kv["remote_block_ids"]
            .as_array()
            .ok_or("remote_block_ids must be grouped")?;
        if groups.is_empty()
            || !groups.iter().all(|group| {
                group
                    .as_array()
                    .is_some_and(|ids| ids.iter().all(|id| id.as_u64().is_some()))
            })
        {
            return Err("remote_block_ids must contain groups of nonnegative integers".into());
        }
        Ok(())
    }
}

impl BackendAdapter for VllmMoriioAdapter {
    fn prepare_pair(
        &self,
        prefill: &dyn Worker,
        decode: &dyn Worker,
    ) -> Result<PairCtx, AdapterError> {
        if [prefill, decode]
            .iter()
            .any(|w| w.dp_rank().unwrap_or(0) != 0 || w.dp_size().unwrap_or(1) != 1)
        {
            return Err(AdapterError::UnsupportedTopology(
                "MoRI-IO READ requires DP=1".into(),
            ));
        }
        Ok(Box::new(MoriioPairCtx {
            transfer_id: format!("xfer-{}", Uuid::new_v4()),
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
                "MoRI-IO READ requires one prompt and one completion per request".into(),
            ));
        }
        inject_prefill(body, &downcast(ctx)?.transfer_id)?;
        body["request_id"] = downcast(ctx)?.transfer_id.clone().into();
        // The producer returns transferable blocks only on FINISHED_LENGTH_CAPPED.
        body["min_tokens"] = 1.into();
        body["ignore_eos"] = true.into();
        body.as_object_mut().unwrap().remove("stop");
        body.as_object_mut().unwrap().remove("stop_token_ids");
        Ok(())
    }

    fn inject_decode_fields(&self, body: &mut Value, ctx: &PairCtx) -> Result<(), AdapterError> {
        let ctx = downcast(ctx)?;
        let obj = body.as_object_mut().ok_or(AdapterError::BodyNotObject)?;
        obj.remove("kv_transfer_params");
        obj.insert("request_id".into(), ctx.transfer_id.clone().into());
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
                "MoRI-IO READ requires one prompt per request".into(),
            ));
        }
        self.inject_prefill_fields(body, ctx)
    }

    fn correlation_id(&self, ctx: &PairCtx) -> Option<String> {
        downcast(ctx).ok().map(|ctx| ctx.transfer_id.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn context() -> PairCtx {
        Box::new(MoriioPairCtx {
            transfer_id: "xfer-test".into(),
        })
    }

    fn metadata() -> Value {
        json!({"transfer_id": "xfer-test", "do_remote_prefill": true,
            "do_remote_decode": false, "remote_engine_id": "producer",
            "remote_host": "10.0.0.1", "remote_handshake_port": "7000",
            "remote_notify_port": 7100, "tp_size": 4, "remote_dp_size": 1,
            "remote_dp_size_local": 1, "remote_dp_rank": 0,
            "remote_dp_rank_override": true, "remote_block_ids": [[1, 2], [], [9]],
            "extension": {"preserve": true}})
    }

    #[test]
    fn moriio_preserves_grouped_metadata_and_empty_prefix_hit_groups() {
        let mut kv = metadata();
        let before = kv.clone();
        VllmMoriioAdapter::validate_decode_kv(&kv, &context()).unwrap();
        assert_eq!(kv, before);
        kv["remote_block_ids"] = json!([[], []]);
        kv["remote_dp_size_local"] = json!(0);
        VllmMoriioAdapter::validate_decode_kv(&kv, &context()).unwrap();
    }

    #[test]
    fn moriio_rejects_missing_or_malformed_fields() {
        let original = metadata();
        for key in original
            .as_object()
            .unwrap()
            .keys()
            .filter(|k| *k != "extension")
        {
            let mut kv = original.clone();
            kv.as_object_mut().unwrap().remove(key);
            assert!(
                VllmMoriioAdapter::validate_decode_kv(&kv, &context()).is_err(),
                "{key}"
            );
        }
        for (key, bad) in [
            ("transfer_id", json!("another")),
            ("remote_host", json!("")),
            ("remote_handshake_port", json!(65536)),
            ("remote_notify_port", json!(0)),
            ("remote_block_ids", json!([1, 2])),
            ("remote_block_ids", json!([[-1]])),
            ("remote_block_ids", json!([])),
            ("remote_block_ids", json!([[true]])),
            ("remote_dp_rank", json!(1)),
            ("remote_dp_size", json!(2)),
            ("remote_dp_size_local", json!(2)),
            ("tp_size", json!(0)),
            ("remote_dp_rank_override", json!(false)),
            ("do_remote_prefill", json!(false)),
        ] {
            let mut kv = original.clone();
            kv[key] = bad;
            assert!(
                VllmMoriioAdapter::validate_decode_kv(&kv, &context()).is_err(),
                "{key}"
            );
        }
    }

    #[test]
    fn moriio_prefill_caps_generation_without_changing_decode() {
        let adapter = VllmMoriioAdapter;
        let original = json!({"messages": [{"role":"user","content":"test"}],
            "tools": [{"type":"function"}], "stream": true, "stream_options": {"include_usage":true},
            "max_tokens": 100, "max_completion_tokens": 100, "min_tokens": 10,
            "stop": ["end"], "stop_token_ids": [2], "kv_transfer_params": {"untrusted":true}});
        let mut p = original.clone();
        let mut d = original.clone();
        adapter.inject_prefill_fields(&mut p, &context()).unwrap();
        adapter.inject_decode_fields(&mut d, &context()).unwrap();
        assert_eq!(p["max_tokens"], 1);
        assert_eq!(p["max_completion_tokens"], 1);
        assert_eq!(p["min_tokens"], 1);
        assert_eq!(p["ignore_eos"], true);
        assert_eq!(p["stream"], false);
        assert!(p.get("stream_options").is_none());
        assert!(p.get("stop").is_none());
        assert!(p.get("stop_token_ids").is_none());
        assert_eq!(p["kv_transfer_params"]["transfer_id"], "xfer-test");
        let mut expected = original;
        expected["request_id"] = json!("xfer-test");
        expected
            .as_object_mut()
            .unwrap()
            .remove("kv_transfer_params");
        assert_eq!(d, expected);
        assert!(adapter
            .inject_batch_prefill_fields(&mut p, &context(), 2)
            .is_err());
        for mut invalid in [
            json!({"n": 2}),
            json!({"best_of": 2}),
            json!({"prompt": ["a", "b"]}),
        ] {
            assert!(adapter
                .inject_prefill_fields(&mut invalid, &context())
                .is_err());
        }
    }
}
