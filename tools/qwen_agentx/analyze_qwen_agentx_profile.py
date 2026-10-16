#!/usr/bin/env python3
"""Summarize per-rank GPU kernels and CPU target/draft scopes in Chrome traces."""
import collections
import gzip
import json
import pathlib
import re

root = pathlib.Path('/home/yifehuan/data/qwen38_agentx_native_20261016')
out = []
for path in sorted((root/'tp2_k3_c4_formal/profiles').rglob('*.json.gz')):
    with gzip.open(path,'rt') as f:
        trace = json.load(f)
    kernels = collections.defaultdict(lambda:[0,0.0])
    scopes = collections.defaultdict(lambda:[0,0.0])
    categories = collections.defaultdict(float)
    gpu = []
    def category(name):
        name=name.lower()
        if any(s in name for s in ['allreduce','all_reduce','reduce_scatter','allgather','all_gather','rccl','msccl','cross_device_reduce']):return 'communication'
        if any(s in name for s in ['gdn','delta_rule','replayssm','causal_conv','recurrent']):return 'gdn_recurrence_conv'
        if any(s in name for s in ['qsa','flash','attention','attn','topk_select']):return 'qsa_attention_index'
        if any(s in name for s in ['moe','fmoe','expert','sorting','softmax']):return 'moe_routing_sampling'
        if any(s in name for s in ['gemm','matmul','cijk','rocblas','hipblas']):return 'dense_and_moe_gemm'
        if any(s in name for s in ['hyper','hc_','_hc','rms','layernorm','layer_norm','quant','silu']):return 'hc_norm_quant_activation'
        return 'other'
    for e in trace['traceEvents']:
        if e.get('ph')!='X':continue
        name=e.get('name','');duration=e.get('dur',0)
        cat=e.get('cat','')
        if cat=='kernel':
            kernels[name][0]+=1;kernels[name][1]+=duration
            categories[category(name)]+=duration
            gpu.append((e['ts'],e['ts']+duration))
        elif cat=='user_annotation' and ('propose_eagle' in name or name.startswith(('decode[','prefill['))):
            scopes[name][0]+=1;scopes[name][1]+=duration
    gpu.sort();busy=0;end=-1
    for start,finish in gpu:
        busy+=max(0,finish-max(start,end));end=max(end,finish)
    total=sum(v[1] for v in kernels.values())
    out.append({'trace':str(path),'kernel_total_ms':total/1000,'gpu_busy_union_ms':busy/1000,
        'window_ms':(gpu[-1][1]-gpu[0][0])/1000 if gpu else 0,
        'kernel_categories_ms':{k:v/1000 for k,v in categories.items()},
        'top_kernels':[{'name':k,'calls':v[0],'total_ms':v[1]/1000,'avg_us':v[1]/v[0]}
                       for k,v in sorted(kernels.items(),key=lambda x:-x[1][1])[:30]],
        'cpu_scopes':[{'name':k,'calls':v[0],'total_ms':v[1]/1000} for k,v in scopes.items()],
        'note':'Kernel categories are name-based approximations; GEMMs can belong to target or draft. CPU scopes are inclusive and can overlap; profiler run is diagnostic only.'})
result={'ranks':out}
(root/'tp2_k3_c4_profile/profile_analysis.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
