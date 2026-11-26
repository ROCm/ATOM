use std::collections::{HashMap, HashSet};
use std::sync::{Arc, LazyLock};

use serde_json::{json, Value};
use tracing::warn;
use uuid::Uuid;

use super::super::types::AdapterError;
use super::{BackendAdapter, PairCtx};
use crate::core::Worker;

/// Router kill switch for early handoff; anything but "0" (or unset) keeps it on.
pub const EARLY_HANDOFF_ENV: &str = "ATOMESH_ATOM_PD_EARLY_HANDOFF";

/// The early-handoff wire contract version this router speaks.
pub const EARLY_HANDOFF_VERSION: u64 = 1;

static EARLY_HANDOFF_ENABLED: LazyLock<bool> =
    LazyLock::new(|| early_handoff_enabled_from(std::env::var(EARLY_HANDOFF_ENV).ok().as_deref()));

pub fn early_handoff_enabled_from(value: Option<&str>) -> bool {
    value.is_none_or(|v| v.trim() != "0")
}

/// What a decode's `/kv_transfer_info` says about receiving a chunked transfer
/// from a prefill that answered early (`chunked_receive`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AtomChunkedReceive {
    /// D-local verdict: covers PP=1 and DCP-decode constraints.
    pub accepts: bool,
    pub tp_size: usize,
}

impl AtomChunkedReceive {
    /// Parse a decode's `/kv_transfer_info`; None for anything but a v1
    /// `chunked_receive` from a `kv_consumer` (that decode stays on the legacy relay).
    pub fn from_kv_transfer_info(data: &Value) -> Option<Self> {
        if data.get("kv_role").and_then(Value::as_str) != Some("kv_consumer") {
            return None;
        }
        let receive = data.get("chunked_receive")?;
        if receive.get("version").and_then(Value::as_u64) != Some(EARLY_HANDOFF_VERSION) {
            return None;
        }
        Some(Self {
            accepts: receive.get("accepts").and_then(Value::as_bool)?,
            tp_size: receive
                .get("tp_size")
                .and_then(Value::as_u64)
                .filter(|tp| *tp >= 1)? as usize,
        })
    }
}

/// Whether a prefill's `/kv_transfer_info` offers a v1 early handoff.
pub fn offers_early_handoff(data: &Value) -> bool {
    data.get("early_handoff")
        .and_then(|v| v.get("version"))
        .and_then(Value::as_u64)
        == Some(EARLY_HANDOFF_VERSION)
}

#[derive(Default)]
pub struct AtomPrefillInfo {
    pub tp_sizes: HashMap<String, usize>,
    pub chunked_transfer: HashMap<String, Value>,
    /// Prefill URLs (worker and base) that offer a v1 early handoff.
    pub early_handoff: HashSet<String>,
    /// Decode URLs (worker and base) with a v1 `chunked_receive`. A decode
    /// missing here never gets an early-handoff prefill.
    pub decode_receive: HashMap<String, AtomChunkedReceive>,
}

impl std::fmt::Debug for AtomPrefillInfo {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AtomPrefillInfo")
            .field("prefill_count", &self.tp_sizes.len())
            .field("early_handoff_prefills", &self.early_handoff.len())
            .field("decode_receive_count", &self.decode_receive.len())
            .finish()
    }
}

#[derive(Debug)]
pub struct AtomAdapter {
    pub prefill_info: Arc<AtomPrefillInfo>,
    early_handoff_enabled: bool,
}

impl AtomAdapter {
    pub fn new(prefill_info: Arc<AtomPrefillInfo>) -> Self {
        Self {
            prefill_info,
            early_handoff_enabled: *EARLY_HANDOFF_ENABLED,
        }
    }

    /// Override the `ATOMESH_ATOM_PD_EARLY_HANDOFF` kill switch.
    pub fn with_early_handoff(mut self, enabled: bool) -> Self {
        self.early_handoff_enabled = enabled;
        self
    }

    /// Whether this request's prefill may answer before its prefill ends
    /// (early handoff, wire contract v1). Router-chunked prefills keep their
    /// own path; a decode that did not say it accepts a chunked receive, or
    /// whose TP is not a multiple of the prefill's, keeps the legacy relay.
    pub fn early_handoff(&self, body: &Value, pair: &AtomPairCtx) -> bool {
        let info = &self.prefill_info;
        if !self.early_handoff_enabled
            || !Self::can_chunk_request(body, pair.carries_media)
            || info.chunked_transfer.contains_key(&pair.prefill_url)
            || !info.early_handoff.contains(&pair.prefill_url)
        {
            return false;
        }
        let Some(prefill_tp) = info
            .tp_sizes
            .get(&pair.prefill_url)
            .copied()
            .filter(|tp| *tp >= 1)
        else {
            return false;
        };
        info.decode_receive
            .get(&pair.decode_url)
            .is_some_and(|d| d.accepts && d.tp_size % prefill_tp == 0)
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
    pub decode_url: String,
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
        decode: &dyn Worker,
        media_count: usize,
    ) -> Result<PairCtx, AdapterError> {
        Ok(Box::new(AtomPairCtx {
            transfer_id: format!("xfer-{}", Uuid::new_v4()),
            prefill_url: prefill.url().to_string(),
            prefill_dp_size: prefill.dp_size().unwrap_or(1),
            prefill_dp_rank: prefill.dp_rank(),
            decode_url: decode.url().to_string(),
            decode_dp_rank: decode.dp_rank(),
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
        let early_handoff = self.early_handoff(body, pair);
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
        } else if early_handoff {
            // P may answer with the handoff right after DP dispatch; the
            // relay then holds P's load until P reports the prefill's end.
            obj["kv_transfer_params"]["early_handoff"] = json!(true);
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
