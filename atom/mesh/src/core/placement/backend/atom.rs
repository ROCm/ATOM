use std::collections::HashMap;
use std::sync::{Arc, LazyLock};

use dashmap::DashSet;

use serde_json::{json, Value};
use tracing::{debug, info, warn};
use uuid::Uuid;

use super::super::planner::{idx2idx, Idx2Idx};
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

    /// Requests with media take the response relay: decode needs the prompt
    /// ids prefill expands from the media and the handoff's `cache_seed`,
    /// which chunked mode starts decode without. ATOM's multimodal prefill
    /// hands them off only once ROCm/ATOM#3410 lands; before it, prefill
    /// answers `kv_transfer_params: null` and the relay answers the request
    /// 502 `prefill_missing_kv_transfer_params`.
    fn can_chunk_request(body: &Value, carries_media: bool) -> bool {
        if carries_media
            || body.get("n").and_then(Value::as_u64).unwrap_or(1) != 1
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
    pub carries_media: bool,
}

fn downcast(ctx: &PairCtx) -> Result<&AtomPairCtx, AdapterError> {
    ctx.downcast_ref::<AtomPairCtx>()
        .ok_or(AdapterError::CtxTypeMismatch)
}

impl BackendAdapter for AtomAdapter {
    fn prepare_pair(
        &self,
        prefill: &dyn Worker,
        decode: &dyn Worker,
    ) -> Result<PairCtx, AdapterError> {
        self.prepare_pair_with_media(prefill, decode, 0)
    }

    fn prepare_pair_with_media(
        &self,
        prefill: &dyn Worker,
        _decode: &dyn Worker,
        media_count: usize,
    ) -> Result<PairCtx, AdapterError> {
        Ok(Box::new(AtomPairCtx {
            transfer_id: format!("xfer-{}", Uuid::new_v4()),
            prefill_url: prefill.url().to_string(),
            prefill_dp_size: prefill.dp_size().unwrap_or(1),
            prefill_dp_rank: prefill.dp_rank(),
            decode_dp_rank: _decode.dp_rank(),
            carries_media: media_count > 0,
        }))
    }

    fn inject_prefill_fields(&self, body: &mut Value, ctx: &PairCtx) -> Result<(), AdapterError> {
        let pair = downcast(ctx)?;
        let chunked = Self::can_chunk_request(body, pair.carries_media)
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
        if !Self::can_chunk_request(body, pair.carries_media) {
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
    let decode_dp_rank = match idx2idx(prefill.as_ref(), decode.as_ref()) {
        Idx2Idx::Move(rank) => rank,
        Idx2Idx::Keep => {
            debug!(
                "ATOM PD rank mapping policy=idx2idx keeps prefill={} decode={}",
                prefill.url(),
                decode.url()
            );
            return prefill;
        }
        Idx2Idx::OutOfRange { rank, size } => {
            // A deployment fault rather than a request's: warn once per prefill.
            static WARNED: LazyLock<DashSet<String>> = LazyLock::new(DashSet::new);
            if WARNED.insert(prefill.base_url().to_string()) {
                warn!(
                    "ATOM PD rank mapping policy=idx2idx: decode rank {} is outside prefill {} dp_size {}, so its ranks are not mapped; check the prefill and decode DP sizes",
                    rank,
                    prefill.base_url(),
                    size
                );
            }
            return prefill;
        }
    };

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

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use serde_json::{json, Value};

    use super::{AtomAdapter, AtomPrefillInfo};
    use crate::core::placement::backend::BackendAdapter;
    use crate::core::placement::test_support::{make_decode_http, make_prefill_http};
    use crate::routers::prepare::inference::{InferenceRequest, ParsedInference};

    /// The prefill and decode `kv_transfer_params` of a chat, with media as
    /// ingress finds it, between a prefill that offers chunked transfer and a decode.
    fn transfer_params(body: Value) -> (Value, Value) {
        let route = "/v1/chat/completions";
        let parsed = ParsedInference::parse(route, body.to_string().as_bytes()).unwrap();
        let media_count = parsed.metadata_with_text(false).media_count;
        let prefill = make_prefill_http("http://p:8000", "m", Some(8998));
        let decode = make_decode_http("http://d:8000", "m");
        let mut info = AtomPrefillInfo::default();
        info.tp_sizes.insert(prefill.url().to_string(), 1);
        let template = json!({"remote_host": "10.0.0.1", "remote_handshake_port": 6301});
        info.chunked_transfer
            .insert(prefill.url().to_string(), template);
        let adapter = AtomAdapter::new(Arc::new(info));
        let ctx = adapter
            .prepare_pair_with_media(prefill.as_ref(), decode.as_ref(), media_count)
            .unwrap();
        let (mut prefill_body, mut decode_body) = (body.clone(), body);
        adapter
            .inject_prefill_fields(&mut prefill_body, &ctx)
            .unwrap();
        adapter
            .inject_decode_fields(&mut decode_body, &ctx)
            .unwrap();
        (
            prefill_body["kv_transfer_params"].clone(),
            decode_body["kv_transfer_params"].clone(),
        )
    }

    #[test]
    fn a_chat_ingress_finds_media_in_takes_the_response_relay() {
        let chat = |content: Value| {
            json!({"model": "m", "messages": [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": content},
            ]})
        };
        let text = json!({"type": "text", "text": "Which color is this?"});
        for media in [
            json!({"type": "image_url", "image_url": {"url": "https://example.com/red.png"}}),
            json!({"type": "video_url", "video_url": {"url": "https://example.com/v.mp4"}}),
        ] {
            let (prefill, decode) = transfer_params(chat(json!([text, media])));
            assert_eq!(prefill["chunked_transfer"], Value::Null, "{prefill}");
            assert_eq!(decode, Value::Null, "{decode}");
        }
        for content in [json!([text, text]), json!("Which color is this?")] {
            let (prefill, decode) = transfer_params(chat(content));
            assert_eq!(prefill["chunked_transfer"], true, "{prefill}");
            assert_eq!(decode["chunked_transfer"], true, "{decode}");
        }
    }
}
