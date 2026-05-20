use serde_json::Value;

use super::super::types::AdapterError;
use super::{BackendAdapter, PairCtx};
use crate::core::Worker;

pub struct SglangAdapter;

#[derive(Debug, Clone)]
pub struct SglangPairCtx {
    pub bootstrap_host: String,
    pub bootstrap_port: Option<u16>,
    pub bootstrap_room: u64,
}

fn downcast(ctx: &PairCtx) -> Result<&SglangPairCtx, AdapterError> {
    ctx.downcast_ref::<SglangPairCtx>()
        .ok_or(AdapterError::CtxTypeMismatch)
}

fn generate_room_id() -> u64 {
    rand::random::<u64>() & (i64::MAX as u64)
}

fn port_to_value(port: Option<u16>) -> Value {
    match port {
        Some(v) => Value::from(v),
        None => Value::Null,
    }
}

impl BackendAdapter for SglangAdapter {
    fn prepare_pair(
        &self,
        prefill: &dyn Worker,
        _decode: &dyn Worker,
    ) -> Result<PairCtx, AdapterError> {
        Ok(Box::new(SglangPairCtx {
            bootstrap_host: prefill.bootstrap_host().to_string(),
            bootstrap_port: prefill.bootstrap_port(),
            bootstrap_room: generate_room_id(),
        }))
    }

    fn inject_prefill_fields(&self, body: &mut Value, ctx: &PairCtx) -> Result<(), AdapterError> {
        let ctx = downcast(ctx)?;
        let obj = body.as_object_mut().ok_or(AdapterError::BodyNotObject)?;
        obj.insert(
            "bootstrap_host".to_string(),
            Value::from(ctx.bootstrap_host.as_str()),
        );
        obj.insert(
            "bootstrap_port".to_string(),
            port_to_value(ctx.bootstrap_port),
        );
        obj.insert(
            "bootstrap_room".to_string(),
            Value::from(ctx.bootstrap_room),
        );
        Ok(())
    }

    /// No-op: SGLang dual-dispatch does not inject on the decode side.
    /// Still validates ctx type so a wrong-adapter call surfaces as CtxTypeMismatch.
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
        let ctx = downcast(ctx)?;
        let obj = body.as_object_mut().ok_or(AdapterError::BodyNotObject)?;
        let host_val = Value::from(ctx.bootstrap_host.as_str());
        let port_val = port_to_value(ctx.bootstrap_port);
        let hosts: Vec<Value> = (0..batch_size).map(|_| host_val.clone()).collect();
        let ports: Vec<Value> = (0..batch_size).map(|_| port_val.clone()).collect();
        let rooms: Vec<Value> = (0..batch_size)
            .map(|_| Value::from(generate_room_id()))
            .collect();
        obj.insert("bootstrap_host".to_string(), Value::Array(hosts));
        obj.insert("bootstrap_port".to_string(), Value::Array(ports));
        obj.insert("bootstrap_room".to_string(), Value::Array(rooms));
        Ok(())
    }

    fn correlation_id(&self, ctx: &PairCtx) -> Option<String> {
        downcast(ctx).ok().map(|c| c.bootstrap_room.to_string())
    }
}
