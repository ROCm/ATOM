#!/usr/bin/env python3
"""Summarize AgentX exports without mixing warmup and profiling requests."""
import argparse
import json
import pathlib
import statistics

p = argparse.ArgumentParser()
p.add_argument('run', type=pathlib.Path)
a = p.parse_args()
artifact = a.run/'aiperf_artifacts'
export = json.loads((artifact/'profile_export_aiperf.json').read_text())
rows = [json.loads(line) for line in (artifact/'profile_export.jsonl').read_text().splitlines()]
phases = {}
cancelled_by_phase = {}
for r in rows:
    phase = r['metadata'].get('benchmark_phase', 'unknown')
    phases[phase] = phases.get(phase, 0) + 1
    if r['metadata'].get('was_cancelled'):
        cancelled_by_phase[phase] = cancelled_by_phase.get(phase, 0) + 1
rows = [r for r in rows if r['metadata'].get('benchmark_phase') == 'profiling']

def metric(r, name):
    return r.get('metrics', {}).get(name, {}).get('value')

def stats(values):
    values = sorted(v for v in values if v is not None)
    if not values:
        return {'n': 0}
    def q(f):
        i = (len(values)-1)*f
        lo = int(i)
        hi = min(lo+1, len(values)-1)
        return values[lo] + (values[hi]-values[lo])*(i-lo)
    return {'n': len(values), 'mean': statistics.mean(values),
            'p50': q(.5), 'p95': q(.95), 'p99': q(.99)}

result = {'metadata': export['metadata'], 'phases': phases, 'cancelled_by_phase': cancelled_by_phase,
          'metrics': {key: export[key] for key in [
              'request_count', 'error_request_count', 'completed_request_count',
              'request_throughput', 'output_token_throughput', 'time_to_first_token',
              'inter_token_latency', 'request_latency', 'input_sequence_length',
              'output_sequence_length', 'output_token_throughput_per_user',
              'active_decode_throughput', 'active_prefill_throughput',
              'effective_decode_throughput', 'osl_mismatch_count'] if key in export},
          'context_bins': {}}
for low, high in [(0,32768),(32768,65536),(65536,131072),(131072,262145)]:
    subset = [r for r in rows if low <= (metric(r, 'input_sequence_length') or 0) < high]
    result['context_bins'][f'{low}-{high-1}'] = {
        key: stats([metric(r,key) for r in subset])
        for key in ['time_to_first_token', 'inter_token_latency', 'request_latency',
                    'output_sequence_length']}
before = json.loads((a.run/'before.mtp.json').read_text())
after = json.loads((a.run/'after.mtp.json').read_text())
dist = {k: after['distribution'][k]-before['distribution'][k] for k in after['distribution']}
steps = sum(dist.values())
accepted = sum(int(k)*v for k,v in dist.items())
drafted = after['total_draft_tokens']-before['total_draft_tokens']
result['mtp_whole_client_run_including_warmup_and_drain'] = {
    'distribution':dist, 'steps':steps, 'accepted_drafts':accepted, 'drafted':drafted,
    'average_tokens_per_forward':1+accepted/steps if steps else None,
    'acceptance_rate':accepted/drafted if drafted else None}
samples = []
for path in sorted(a.run.glob('sample_*.mtp.json')):
    samples.append({'utc_epoch_seconds': int(path.name.split('_')[1].split('.')[0]),
                    'stats': json.loads(path.read_text())})
result['mtp_samples'] = samples
server_export = json.loads((artifact/'server_metrics_export.json').read_text())
window = server_export['summary']['phase_time_ranges']['profiling']
result['server_export_phase'] = server_export.get('metrics_phase')
result['server_metric_export_stats'] = {
    name: [series['stats'] for series in server_export['metrics'].get(name, {}).get('series', [])]
    for name in ['atom:mtp_accepted_tokens', 'atom:mtp_draft_tokens', 'atom:decode_batch_size',
                 'atom:requests_running', 'atom:kv_cache_usage_ratio', 'atom:preemptions']}
result['server_metric_export_note'] = 'Retain upstream stats as exported; cumulative histogram/counter stats may include boundary or warmup effects. Interior independent samples below give an explicit interval.'
inside = [s for s in samples if window['start_ns']/1e9 <= s['utc_epoch_seconds'] <= window['end_ns']/1e9]
if len(inside) >= 2:
    first, last = inside[0], inside[-1]
    interval_dist = {k: last['stats']['distribution'][k]-first['stats']['distribution'][k]
                     for k in last['stats']['distribution']}
    interval_steps = sum(interval_dist.values())
    interval_accepted = sum(int(k)*v for k,v in interval_dist.items())
    result['mtp_sampled_profiling_interval'] = {
        'start_epoch': first['utc_epoch_seconds'], 'end_epoch': last['utc_epoch_seconds'],
        'distribution': interval_dist, 'steps': interval_steps,
        'average_tokens_per_forward': 1+interval_accepted/interval_steps if interval_steps else None,
        'note': 'Interior samples exclude warmup, with up to 30s unsampled at each edge; may include drain.'}
(a.run/'summary.json').write_text(json.dumps(result,indent=2))
print(json.dumps({k:v for k,v in result.items() if k!='mtp_samples'},indent=2))
