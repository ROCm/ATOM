use std::collections::HashMap;
use std::sync::Arc;

use serde_json::{json, Value};
use uuid::Uuid;

use super::super::types::AdapterError;
use super::{BackendAdapter, PairCtx};
use crate::core::Worker;

#[derive(Default)]
pub struct AtomPrefillInfo {
    pub tp_sizes: HashMap<String, usize>,
}

impl std::fmt::Debug for AtomPrefillInfo {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AtomPrefillInfo")
            .field("prefill_count", &self.tp_sizes.len())
            .finish()
    }
}

#[derive(Debug)]
pub struct AtomAdapter {
    pub prefill_info: Arc<AtomPrefillInfo>,
}

impl AtomAdapter {
    pub fn new(prefill_info: Arc<AtomPrefillInfo>) -> Self {
        Self { prefill_info }
    }

    /// Fill in fields the prefill response omits but decode's ReqMeta needs:
    /// remote_dp_size/remote_tp_size, and remote_dp_rank (renamed from the
    /// prefill's `dp_rank`). Mirrors proxy.py.
    pub fn enrich_decode_kv(&self, kv: &mut Value, ctx: &PairCtx) -> Result<(), AdapterError> {
        let ctx = downcast(ctx)?;
        let obj = kv.as_object_mut().ok_or(AdapterError::BodyNotObject)?;
        let tp_size = self
            .prefill_info
            .tp_sizes
            .get(&ctx.prefill_url)
            .copied()
            .ok_or_else(|| AdapterError::TpSizeMissing {
                prefill_url: ctx.prefill_url.clone(),
            })?;
        obj.insert("remote_dp_size".to_string(), json!(ctx.prefill_dp_size));
        obj.insert("remote_tp_size".to_string(), json!(tp_size));
        // Only a numeric rank; else leave unset so decode defaults to 0.
        if let Some(dp_rank) = obj.get("dp_rank").filter(|v| v.is_number()).cloned() {
            obj.insert("remote_dp_rank".to_string(), dp_rank);
        }
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct AtomPairCtx {
    pub transfer_id: String,
    pub prefill_url: String,
    pub prefill_dp_size: usize,
}

fn downcast(ctx: &PairCtx) -> Result<&AtomPairCtx, AdapterError> {
    ctx.downcast_ref::<AtomPairCtx>()
        .ok_or(AdapterError::CtxTypeMismatch)
}

impl BackendAdapter for AtomAdapter {
    fn prepare_pair(
        &self,
        prefill: &dyn Worker,
        _decode: &dyn Worker,
    ) -> Result<PairCtx, AdapterError> {
        Ok(Box::new(AtomPairCtx {
            transfer_id: format!("xfer-{}", Uuid::new_v4()),
            prefill_url: prefill.url().to_string(),
            prefill_dp_size: prefill.dp_size().unwrap_or(1),
        }))
    }

    fn inject_prefill_fields(&self, body: &mut Value, ctx: &PairCtx) -> Result<(), AdapterError> {
        downcast(ctx)?;
        let obj = body.as_object_mut().ok_or(AdapterError::BodyNotObject)?;
        obj.insert(
            "kv_transfer_params".to_string(),
            json!({
                "do_remote_decode": true,
                "do_remote_prefill": false,
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

    /// No-op: the kv_transfer_params for decode comes from the prefill response,
    /// not from a static ctx. Mesh injects it in execute_atom_relay after enriching.
    fn inject_decode_fields(&self, _body: &mut Value, ctx: &PairCtx) -> Result<(), AdapterError> {
        downcast(ctx)?;
        Ok(())
    }

    fn inject_batch_prefill_fields(
        &self,
        body: &mut Value,
        ctx: &PairCtx,
        batch_size: usize,
    ) -> Result<(), AdapterError> {
        debug_assert_eq!(batch_size, 1, "ATOM Mooncake fires per-request");
        self.inject_prefill_fields(body, ctx)
    }

    fn correlation_id(&self, ctx: &PairCtx) -> Option<String> {
        downcast(ctx).ok().map(|c| c.transfer_id.clone())
    }
}
