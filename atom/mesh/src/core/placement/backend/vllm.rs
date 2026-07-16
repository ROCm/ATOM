use std::collections::HashMap;
use std::sync::Arc;

use serde_json::{json, Value};
use uuid::Uuid;

use super::super::types::AdapterError;
use super::{BackendAdapter, PairCtx};
use crate::core::Worker;

#[derive(Default)]
pub struct VllmPrefillInfo {
    pub bootstrap_addrs: HashMap<String, String>,
    pub engine_ids: HashMap<String, HashMap<usize, String>>,
}

impl std::fmt::Debug for VllmPrefillInfo {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VllmPrefillInfo")
            .field("prefill_count", &self.bootstrap_addrs.len())
            .finish()
    }
}

#[derive(Debug)]
pub struct VllmAdapter {
    pub prefill_info: Arc<VllmPrefillInfo>,
}

impl VllmAdapter {
    pub fn new(prefill_info: Arc<VllmPrefillInfo>) -> Self {
        Self { prefill_info }
    }
}

#[derive(Debug, Clone)]
pub struct VllmPairCtx {
    pub prefill_url: String,
    pub bootstrap_addr: String,
    pub engine_id: String,
    pub transfer_id: String,
    pub dp_rank: usize,
}

fn downcast(ctx: &PairCtx) -> Result<&VllmPairCtx, AdapterError> {
    ctx.downcast_ref::<VllmPairCtx>()
        .ok_or(AdapterError::CtxTypeMismatch)
}

impl BackendAdapter for VllmAdapter {
    fn prepare_pair(
        &self,
        prefill: &dyn Worker,
        _decode: &dyn Worker,
    ) -> Result<PairCtx, AdapterError> {
        let prefill_url = prefill.url().to_string();
        let bootstrap_addr = self
            .prefill_info
            .bootstrap_addrs
            .get(&prefill_url)
            .cloned()
            .ok_or_else(|| AdapterError::BootstrapAddrMissing {
                prefill_url: prefill_url.clone(),
            })?;
        let dp_rank = prefill.dp_rank().unwrap_or(0);
        let engine_id = self
            .prefill_info
            .engine_ids
            .get(&prefill_url)
            .and_then(|m| m.get(&dp_rank))
            .cloned()
            .ok_or_else(|| AdapterError::EngineIdMissing {
                prefill_url: prefill_url.clone(),
                dp_rank,
            })?;
        Ok(Box::new(VllmPairCtx {
            prefill_url,
            bootstrap_addr,
            engine_id,
            transfer_id: format!("xfer-{}", Uuid::new_v4()),
            dp_rank,
        }))
    }

    fn inject_prefill_fields(&self, body: &mut Value, ctx: &PairCtx) -> Result<(), AdapterError> {
        let ctx = downcast(ctx)?;
        let obj = body.as_object_mut().ok_or(AdapterError::BodyNotObject)?;
        obj.insert(
            "kv_transfer_params".to_string(),
            json!({
                "do_remote_decode": true,
                "do_remote_prefill": false,
                "transfer_id": ctx.transfer_id,
            }),
        );
        obj.insert("stream".to_string(), Value::Bool(false));
        obj.insert("max_tokens".to_string(), json!(1));
        if obj.contains_key("max_completion_tokens") {
            obj.insert("max_completion_tokens".to_string(), json!(1));
        }
        obj.remove("stream_options");
        Ok(())
    }

    fn inject_decode_fields(&self, body: &mut Value, ctx: &PairCtx) -> Result<(), AdapterError> {
        let ctx = downcast(ctx)?;
        let obj = body.as_object_mut().ok_or(AdapterError::BodyNotObject)?;
        obj.insert(
            "kv_transfer_params".to_string(),
            json!({
                "do_remote_decode": false,
                "do_remote_prefill": true,
                "remote_bootstrap_addr": ctx.bootstrap_addr,
                "remote_engine_id": ctx.engine_id,
                "transfer_id": ctx.transfer_id,
            }),
        );
        Ok(())
    }

    fn inject_batch_prefill_fields(
        &self,
        body: &mut Value,
        ctx: &PairCtx,
        batch_size: usize,
    ) -> Result<(), AdapterError> {
        debug_assert_eq!(batch_size, 1, "vLLM Mooncake fires per-request");
        self.inject_prefill_fields(body, ctx)
    }

    fn correlation_id(&self, ctx: &PairCtx) -> Option<String> {
        downcast(ctx).ok().map(|c| c.transfer_id.clone())
    }
}
