"""Experimental final-state fast path with replay only for partial acceptance."""
import os
import logging
from types import MethodType
import torch
import triton
import triton.language as tl

@triton.jit
def _commit(R,PTR,READ,WRITE,ACCEPT,HV:tl.constexpr,D:tl.constexpr,Q:tl.constexpr,CAP:tl.constexpr,BV:tl.constexpr,STATE_STRIDE:tl.constexpr):
    n=tl.program_id(0);lh=tl.program_id(1);tile=tl.program_id(2)
    layer=lh//HV;h=lh%HV
    count=tl.load(ACCEPT+n);src=tl.load(READ+n);last=tl.load(WRITE+n*Q+Q-1)
    if (count>0)&(src>=0)&((count<Q)|(src==last)):
        state=tl.load(PTR+layer).to(tl.pointer_type(tl.float32))
        vv=tile*BV+tl.arange(0,BV);kk=tl.arange(0,D)
        value=tl.load(state+src.to(tl.int64)*STATE_STRIDE+h*D*D+vv[:,None]*D+kk[None,:],vv[:,None]<D,0)
        for t in tl.static_range(Q):
            base=(((layer*CAP+n)*Q+t)*HV+h)*3*D
            k=tl.load(R+base+kk);u=tl.load(R+base+D+vv,vv<D,0);decay=tl.load(R+base+2*D)
            value=tl.where(t<count,value*decay+u[:,None]*k[None,:],value)
        dst=tl.load(WRITE+n*Q+count-1)
        tl.store(state+dst.to(tl.int64)*STATE_STRIDE+h*D*D+vv[:,None]*D+kk[None,:],value,(vv[:,None]<D)&(dst>=0))

def enabled(runtime):
    return bool(getattr(runtime, '_hybrid_enabled', False))

def prepare_metadata(runtime,metadata,caches,layers):
    if not enabled(runtime) or metadata is None or not getattr(metadata,'rtp_flydsl_mtp',False):return
    keys=tuple(sorted(layers));writes=metadata.spec_state_indices_tensor;reads=metadata.rtp_flydsl_read_indices
    n,q=writes.shape;state=caches[f'layer_{keys[0]}'].v_cache
    if state.dtype!=torch.float32 or state.shape[-2:]!=(128,128):raise RuntimeError('Hybrid replay supports FP32 128x128 state only')
    hv=state.shape[1];key=(n,q)
    if key not in runtime._hybrid_stages:
        if torch.cuda.is_current_stream_capturing():raise RuntimeError('Hybrid replay buffers not warmed')
        for layer in keys:
            st=caches[f'layer_{layer}'].v_cache
            if st.shape!=state.shape or st.stride()!=state.stride() or st.stride(-2)!=1:raise RuntimeError('Hybrid replay state layout differs')
        records=torch.empty(len(keys),n,q,hv,3,128,device=state.device,dtype=torch.float32)
        ptrs=torch.tensor([caches[f'layer_{i}'].v_cache.data_ptr() for i in keys],device=state.device,dtype=torch.int64)
        runtime._hybrid_stages[key]=(records,ptrs)
    records,ptrs=runtime._hybrid_stages[key]
    metadata.rtp_hybrid_records={layer:records[i] for i,layer in enumerate(keys)}
    runtime._hybrid_active=(records,ptrs,reads,writes,state.stride(0))

def _clear(self):self._hybrid_active=None

def _save(self,key):
    if self._hybrid_active is None:raise RuntimeError('Missing hybrid capture stage')
    self._hybrid_graphs[int(key)]=self._hybrid_active

def _activate(self,key):self._hybrid_active=self._hybrid_graphs[int(key)]

def _prepare(self,accept):
    stage=self._hybrid_active
    if stage is None or self._hybrid_pending is not None:raise RuntimeError('Invalid hybrid transaction')
    if accept.ndim!=1 or accept.dtype!=torch.int32 or not 0<accept.numel()<=stage[2].numel() or accept.device!=stage[0].device:raise RuntimeError('Invalid accept vector')
    self._hybrid_pending=accept;self._hybrid_modified=False

def _finish(self,commit):
    if not commit:
        if self._hybrid_modified:raise RuntimeError('Hybrid partial commit requires fail-stop')
        self._hybrid_pending=None;return
    accept=self._hybrid_pending
    if accept is None:raise RuntimeError('Hybrid commit not prepared')
    records,ptrs,reads,writes,state_stride=self._hybrid_active
    layers,cap,q,hv,_,d=records.shape
    self._hybrid_modified=True
    _commit[(accept.numel(),layers*hv,4)](records,ptrs,reads,writes,accept,hv,d,q,cap,32,state_stride,num_warps=4)

def _finalize(self):
    if self._hybrid_pending is None:raise RuntimeError('Hybrid finalize without prepare')
    self._hybrid_pending=None;self._hybrid_modified=False

def _has(self):return self._hybrid_reducer is not None

def _reduce(self,ok):return self._hybrid_reducer.reduce(ok)

def install_runtime(runtime):
    from atom.plugin.rtpllm.utils.replay_policy import replay_mode
    runtime._hybrid_enabled = False
    if (not torch.version.hip or replay_mode() != "1"
            or not getattr(runtime._rtp_model_config, "atom_qwen4_exp_mtp_target", False)):
        return
    from rtp_llm.models_py.modules.qwen4_exp.cpu_target_status import prepare_cpu_target_status
    runtime._hybrid_reducer=prepare_cpu_target_status(runtime._rtp_model_config,runtime.parallelism_config)
    if runtime._hybrid_reducer is None:
        # Keep the supported FlyDSL path when this topology lacks CPU consensus.
        return
    runtime._hybrid_enabled = True
    logging.getLogger(__name__).info("RTP GDN default route: FlyDSL + hybrid Replay; ATOM_ENABLE_REPLAYSSM=%s", os.getenv("ATOM_ENABLE_REPLAYSSM", "<unset>"))
    runtime._hybrid_stages={};runtime._hybrid_graphs={};runtime._hybrid_active=None;runtime._hybrid_pending=None;runtime._hybrid_modified=False
    for name,fn in [('has_speculative_target_status_reducer',_has),('reduce_speculative_target_prepare_status',_reduce),('clear_speculative_target_graph_capture',_clear),('save_speculative_target_graph_capture',_save),('activate_speculative_target_graph_replay',_activate),('prepare_speculative_target_commit',_prepare),('finish_speculative_target_commit',_finish),('finalize_speculative_target_commit',_finalize)]:setattr(runtime,name,MethodType(fn,runtime))
