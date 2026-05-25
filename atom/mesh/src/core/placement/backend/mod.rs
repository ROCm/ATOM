pub mod atom;
pub mod sglang;
pub mod vllm;

use std::any::Any;

use serde_json::Value;

use super::types::AdapterError;
use crate::core::Worker;

pub type PairCtx = Box<dyn Any + Send + Sync>;

pub trait BackendAdapter: Send + Sync {
    fn prepare_pair(
        &self,
        prefill: &dyn Worker,
        decode: &dyn Worker,
    ) -> Result<PairCtx, AdapterError>;

    fn inject_prefill_fields(&self, body: &mut Value, ctx: &PairCtx) -> Result<(), AdapterError>;

    fn inject_decode_fields(&self, body: &mut Value, ctx: &PairCtx) -> Result<(), AdapterError>;

    fn inject_batch_prefill_fields(
        &self,
        body: &mut Value,
        ctx: &PairCtx,
        batch_size: usize,
    ) -> Result<(), AdapterError>;

    fn correlation_id(&self, ctx: &PairCtx) -> Option<String>;
}
