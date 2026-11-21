use std::collections::HashMap;
use std::sync::Arc;

use serde_json::{json, Value};
use tracing::{debug, info, warn};
use uuid::Uuid;

use super::super::types::AdapterError;
use super::{BackendAdapter, PairCtx};
use crate::config::types::AtomPdRankMappingPolicy;
use crate::core::{Worker, WorkerRegistry};

#[derive(Default)]
pub struct AtomPrefillInfo {
    pub tp_sizes: HashMap<String, usize>,
    pub chunked_transfer: HashMap<String, Value>,
}

impl std::fmt::Debug for AtomPrefillInfo {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AtomPrefillInfo")
            .field("prefill_count", &self.tp_sizes.len())
            .finish()
    }
}

/// Read one ATOM prefill's `kv_transfer_info`: its TP size and chunked-transfer template.
/// A missing `tp_size` errs naming `source`, a role other than `kv_producer` naming `base_url`.
pub fn parse_atom_prefill_info(
    data: &Value,
    source: &str,
    base_url: &str,
) -> Result<(usize, Option<Value>), String> {
    let tp = data
        .get("tp_size")
        .and_then(|v| v.as_u64())
        .ok_or_else(|| format!("Missing tp_size in {} response", source))? as usize;
    let kv_role = data.get("kv_role").and_then(|v| v.as_str());
    if kv_role != Some("kv_producer") {
        return Err(format!(
            "{} is not a prefill (kv_role={:?}, expected kv_producer)",
            base_url, kv_role
        ));
    }
    info!("ATOM prefill {} tp_size={}", base_url, tp);
    let chunked = data.get("chunked_transfer").filter(|v| v.is_object());
    Ok((tp, chunked.cloned()))
}

#[derive(Debug)]
pub struct AtomAdapter {
    pub prefill_info: Arc<AtomPrefillInfo>,
}

impl AtomAdapter {
    pub fn new(prefill_info: Arc<AtomPrefillInfo>) -> Self {
        Self { prefill_info }
    }

    fn can_chunk_request(body: &Value) -> bool {
        if body.get("n").and_then(Value::as_u64).unwrap_or(1) != 1
            || body.get("best_of").and_then(Value::as_u64).unwrap_or(1) != 1
        {
            return false;
        }
        // One transfer ID identifies one sequence. Batched completion prompts
        // keep using the existing response relay.
        body.get("prompt")
            .and_then(Value::as_array)
            .is_none_or(|items| items.iter().all(Value::is_number))
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
        let remote_dp_rank = obj
            .get("dp_rank")
            .filter(|v| v.is_number())
            .cloned()
            .or_else(|| ctx.prefill_dp_rank.map(|r| json!(r)));
        if let Some(dp_rank) = remote_dp_rank {
            obj.insert("remote_dp_rank".to_string(), dp_rank);
        }
        Ok(())
    }

    /// Copy prompt IDs into decode KV metadata using vLLM's PD format.
    /// Return the number copied, or zero when no usable IDs are available.
    pub fn carry_prompt_token_ids(prefill_body: &Value, kv: &mut Value) -> usize {
        let Some(ids) = prefill_body.get("prompt_token_ids") else {
            return 0;
        };
        if ids.is_null() {
            return 0;
        }
        let Some(len) = ids.as_array().map(|a| a.len()).filter(|n| *n > 0) else {
            warn!(
                "prefill returned an unusable prompt_token_ids ({}); decode will tokenize",
                ids
            );
            return 0;
        };
        let Some(obj) = kv.as_object_mut() else {
            warn!("decode kv_transfer_params is not an object; dropping prompt_token_ids");
            return 0;
        };
        obj.insert("prompt_token_ids".to_string(), ids.clone());
        len
    }
}

#[derive(Debug, Clone)]
pub struct AtomPairCtx {
    pub transfer_id: String,
    pub prefill_url: String,
    pub prefill_dp_size: usize,
    pub prefill_dp_rank: Option<usize>,
    pub decode_dp_rank: Option<usize>,
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
            prefill_dp_rank: prefill.dp_rank(),
            decode_dp_rank: _decode.dp_rank(),
        }))
    }

    fn inject_prefill_fields(&self, body: &mut Value, ctx: &PairCtx) -> Result<(), AdapterError> {
        let pair = downcast(ctx)?;
        let chunked = Self::can_chunk_request(body)
            && self
                .prefill_info
                .chunked_transfer
                .contains_key(&pair.prefill_url);
        let obj = body.as_object_mut().ok_or(AdapterError::BodyNotObject)?;
        obj.insert(
            "kv_transfer_params".to_string(),
            json!({
                "do_remote_decode": true,
                "do_remote_prefill": false,
            }),
        );
        if chunked {
            obj["kv_transfer_params"]["chunked_transfer"] = json!(true);
            obj["kv_transfer_params"]["transfer_id"] = json!(pair.transfer_id);
        }
        obj.insert("stream".to_string(), Value::Bool(false));
        obj.insert("max_tokens".to_string(), json!(1));
        if obj.contains_key("max_completion_tokens") {
            obj.insert("max_completion_tokens".to_string(), json!(1));
        }
        obj.remove("stream_options");
        // Request prompt IDs for decode reuse; older servers may omit them.
        obj.insert("return_token_ids".to_string(), Value::Bool(true));
        Ok(())
    }

    /// Chunked mode establishes D before P finishes; legacy mode receives its
    /// metadata from the completed prefill response in execute_atom_relay.
    fn inject_decode_fields(&self, body: &mut Value, ctx: &PairCtx) -> Result<(), AdapterError> {
        let pair = downcast(ctx)?;
        if !Self::can_chunk_request(body) {
            return Ok(());
        }
        if let Some(template) = self.prefill_info.chunked_transfer.get(&pair.prefill_url) {
            let mut kv = template.clone();
            kv["chunked_transfer"] = json!(true);
            kv["do_remote_prefill"] = json!(true);
            kv["do_remote_decode"] = json!(false);
            kv["transfer_id"] = json!(pair.transfer_id);
            self.enrich_decode_kv(&mut kv, ctx)?;
            body.as_object_mut()
                .ok_or(AdapterError::BodyNotObject)?
                .insert("kv_transfer_params".to_string(), kv);
        }
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

/// The prefill worker that the ATOM P/D rank-mapping `policy` pairs with `decode`.
pub fn apply_atom_pd_rank_mapping_policy(
    policy: AtomPdRankMappingPolicy,
    worker_registry: &WorkerRegistry,
    prefill: Arc<dyn Worker>,
    decode: &Arc<dyn Worker>,
) -> Arc<dyn Worker> {
    match policy {
        AtomPdRankMappingPolicy::None => prefill,
        AtomPdRankMappingPolicy::Idx2Idx => {
            map_atom_prefill_idx2idx(worker_registry, prefill, decode)
        }
    }
}

fn map_atom_prefill_idx2idx(
    worker_registry: &WorkerRegistry,
    prefill: Arc<dyn Worker>,
    decode: &Arc<dyn Worker>,
) -> Arc<dyn Worker> {
    let (Some(prefill_dp_size), Some(decode_dp_rank)) = (prefill.dp_size(), decode.dp_rank())
    else {
        debug!(
            "ATOM PD rank mapping policy=idx2idx skipped: prefill={} prefill_dp_size={:?} decode={} decode_dp_rank={:?}",
            prefill.url(),
            prefill.dp_size(),
            decode.url(),
            decode.dp_rank()
        );
        return prefill;
    };

    if decode_dp_rank >= prefill_dp_size {
        warn!(
            "ATOM PD rank mapping policy=idx2idx skipped: decode rank {} is outside prefill dp_size {} (prefill={}, decode={})",
            decode_dp_rank,
            prefill_dp_size,
            prefill.url(),
            decode.url()
        );
        return prefill;
    }

    if prefill.dp_rank() == Some(decode_dp_rank) {
        return prefill;
    }

    let mapped_url = format!("{}@{}", prefill.base_url(), decode_dp_rank);
    match worker_registry.get_by_url(&mapped_url) {
        Some(mapped)
            if mapped.is_available()
                && mapped.model_id() == prefill.model_id()
                && matches!(
                    mapped.worker_type(),
                    crate::core::WorkerType::Prefill { .. }
                )
                && mapped.connection_mode().matches(prefill.connection_mode()) =>
        {
            info!(
                "ATOM PD rank mapping policy=idx2idx: prefill {} -> {}, decode={}",
                prefill.url(),
                mapped.url(),
                decode.url()
            );
            mapped
        }
        Some(mapped) => {
            warn!(
                "ATOM PD rank mapping policy=idx2idx target {} is unhealthy; keeping prefill {} (decode={})",
                mapped.url(),
                prefill.url(),
                decode.url()
            );
            prefill
        }
        None => {
            warn!(
                "ATOM PD rank mapping policy=idx2idx target {} not found; keeping prefill {} (decode={})",
                mapped_url,
                prefill.url(),
                decode.url()
            );
            prefill
        }
    }
}
