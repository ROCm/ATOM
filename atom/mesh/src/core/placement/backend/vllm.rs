use std::collections::HashMap;
use std::sync::Arc;

use serde_json::Value;

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
        decode: &dyn Worker,
    ) -> Result<PairCtx, AdapterError> {
        let _ = (prefill, decode);
        todo!()
    }

    fn inject_prefill_fields(
        &self,
        body: &mut Value,
        ctx: &PairCtx,
    ) -> Result<(), AdapterError> {
        let ctx = downcast(ctx)?;
        let _ = (body, ctx);
        todo!()
    }

    fn inject_decode_fields(
        &self,
        body: &mut Value,
        ctx: &PairCtx,
    ) -> Result<(), AdapterError> {
        let ctx = downcast(ctx)?;
        let _ = (body, ctx);
        todo!()
    }

    fn inject_batch_prefill_fields(
        &self,
        body: &mut Value,
        ctx: &PairCtx,
        batch_size: usize,
    ) -> Result<(), AdapterError> {
        let ctx = downcast(ctx)?;
        let _ = (body, ctx, batch_size);
        todo!()
    }
}
