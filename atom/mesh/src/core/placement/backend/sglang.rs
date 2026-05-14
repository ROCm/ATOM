use serde_json::Value;

use super::super::types::AdapterError;
use super::{BackendAdapter, PairCtx};
use crate::core::Worker;

pub struct SglangAdapter;

#[derive(Debug, Clone)]
pub struct SglangPairCtx {
    pub bootstrap_host: String,
    pub bootstrap_port: Option<u16>,
}

fn downcast(ctx: &PairCtx) -> Result<&SglangPairCtx, AdapterError> {
    ctx.downcast_ref::<SglangPairCtx>()
        .ok_or(AdapterError::CtxTypeMismatch)
}

impl BackendAdapter for SglangAdapter {
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

    /// No-op: SGLang dual-dispatch does not inject on the decode side.
    /// Still validates ctx type so a wrong-adapter call surfaces as CtxTypeMismatch.
    fn inject_decode_fields(
        &self,
        body: &mut Value,
        ctx: &PairCtx,
    ) -> Result<(), AdapterError> {
        let _ = (body, downcast(ctx)?);
        Ok(())
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
