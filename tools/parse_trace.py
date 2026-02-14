#!/usr/bin/env python3
"""
Parse PyTorch profiler trace JSON to extract kernel information.

Usage:
    python parse_trace.py <trace.json> [output.csv]
"""

import csv
import json
import gzip
import sys
from typing import List, Dict, Any

# Modules to filter out (no corresponding GPU kernel in decode)
FILTER_OUT = ['fill_']

# Sampling-related modules and low-level ops to filter out in prefill
FILTER_OUT_PREFILL = ['aten::', 'aiter::gemm_a16w16', 'aiter::mixed_sample']


# =============================================================================
# Utility Functions
# =============================================================================

def load_trace(filepath: str) -> Dict[str, Any]:
    """Load trace JSON file (supports .gz)."""
    opener = gzip.open if filepath.endswith('.gz') else open
    with opener(filepath, 'rt', encoding='utf-8') as f:
        return json.load(f)


def is_within(child_ts: float, child_dur: float, parent_ts: float, parent_dur: float) -> bool:
    """Check if child event is within parent's time range."""
    return child_ts >= parent_ts and (child_ts + child_dur) <= (parent_ts + parent_dur)


def is_kernel_launch(name: str) -> bool:
    """Check if name is a kernel launch (contains 'launch' and 'kernel')."""
    n = name.lower()
    return 'launch' in n and 'kernel' in n


def should_filter(name: str) -> bool:
    """Check if module should be filtered out."""
    return any(f in name for f in FILTER_OUT)


def should_filter_prefill(name: str) -> bool:
    """Check if module should be filtered out in prefill (sampling ops)."""
    return any(f in name for f in FILTER_OUT_PREFILL)


# =============================================================================
# Event Query Functions
# =============================================================================

def find_events(events: List[Dict], name: str, prefix: bool = False) -> List[Dict]:
    """Find all duration events (ph='X') with given name, sorted by time.
    
    Args:
        prefix: If True, match names starting with 'name'; if False, exact match.
    """
    if prefix:
        result = [e for e in events if e.get('name', '').startswith(name) and e.get('ph') == 'X']
    else:
        result = [e for e in events if e.get('name') == name and e.get('ph') == 'X']
    return sorted(result, key=lambda x: x['ts'])


def get_gpu_kernels(events: List[Dict], start_ts: float) -> List[Dict]:
    """Get GPU kernels (cat='kernel') starting from given timestamp."""
    result = [e for e in events if e.get('cat') == 'kernel' and e['ts'] >= start_ts]
    return sorted(result, key=lambda x: x['ts'])


def get_direct_children(parent: Dict, events: List[Dict]) -> List[Dict]:
    """Get direct children of parent event (excluding nested children)."""
    p_ts, p_dur = parent['ts'], parent.get('dur', 0)
    
    # Find all events within parent
    candidates = [
        e for e in events
        if e.get('ph') == 'X' and e is not parent
        and is_within(e.get('ts', 0), e.get('dur', 0), p_ts, p_dur)
    ]
    
    # Filter to direct children only
    direct = []
    for c in candidates:
        c_ts, c_dur = c['ts'], c.get('dur', 0)
        # Check if c is nested inside any other candidate
        is_direct = not any(
            is_within(c_ts, c_dur, o['ts'], o.get('dur', 0))
            for o in candidates if o is not c
        )
        if is_direct:
            direct.append(c)
    
    return sorted(direct, key=lambda x: x['ts'])


def count_kernel_launches(event: Dict, events: List[Dict]) -> int:
    """Count kernel launches within event's subtree."""
    e_ts, e_dur = event['ts'], event.get('dur', 0)
    return sum(
        1 for e in events
        if e.get('ph') == 'X' and is_kernel_launch(e.get('name', ''))
        and is_within(e.get('ts', 0), e.get('dur', 0), e_ts, e_dur)
    )


def has_kernel_launch(event: Dict, events: List[Dict]) -> bool:
    """Check if event's subtree contains any kernel launch."""
    return count_kernel_launches(event, events) > 0


# =============================================================================
# Parse Functions
# =============================================================================

def parse_prefill(events: List[Dict], output_csv: str) -> None:
    """
    Parse prefill phase: find kernel launch modules between capture_graph end
    and the first decode_step.
    """
    # Find capture_graph end time (use prefix match for capture_graph_bs_X)
    capture_graphs = find_events(events, 'capture_graph', prefix=True)
    if not capture_graphs:
        print("No capture_graph events found.")
        return
    
    # Use the LAST capture_graph (latest end time)
    cg = max(capture_graphs, key=lambda e: e['ts'] + e.get('dur', 0))
    cg_end_ts = cg['ts'] + cg.get('dur', 0)
    print(f"Last capture_graph ({cg.get('name')}) ends at: {cg_end_ts}")
    
    # Find all CompiledFxGraph events after capture_graph, before first decode_step
    decode_steps = find_events(events, 'decode_step', prefix=True)
    decode_start = decode_steps[0]['ts'] if decode_steps else float('inf')
    
    compiled_fx = [
        e for e in events
        if 'CompiledFxGraph' in e.get('name', '') and e.get('ph') == 'X'
        and e['ts'] >= cg_end_ts and e['ts'] < decode_start
    ]
    compiled_fx = sorted(compiled_fx, key=lambda x: x['ts'])
    
    if not compiled_fx:
        print("No CompiledFxGraph events found in prefill range.")
        return
    
    # Use first decode_step as prefill end
    prefill_start = cg_end_ts
    prefill_end = decode_start
    
    print(f"Prefill phase: {prefill_start} ~ {prefill_end}")
    print(f"CompiledFxGraph count: {len(compiled_fx)}")
    
    # Pre-filter events to prefill time range only (for speed)
    prefill_events = [
        e for e in events
        if e.get('ph') == 'X'
        and e['ts'] >= prefill_start
        and e['ts'] < prefill_end
    ]
    print(f"Events in prefill range: {len(prefill_events)}")
    
    # Find kernel launches in prefill range
    kernel_launches = [
        e for e in prefill_events
        if is_kernel_launch(e.get('name', ''))
    ]
    print(f"Kernel launches in prefill: {len(kernel_launches)}")
    
    # For each CompiledFxGraph, find direct children with kernel launches
    # Format: (timestamp, name, kernel_names)
    kernel_modules = []
    for cfg in compiled_fx:
        cfg_ts, cfg_dur = cfg['ts'], cfg.get('dur', 0)
        
        # Find events within this cfg (use pre-filtered list)
        cfg_events = [
            e for e in prefill_events
            if is_within(e['ts'], e.get('dur', 0), cfg_ts, cfg_dur)
        ]
        
        # Count kernel launches in this cfg
        cfg_kernel_count = sum(1 for e in cfg_events if is_kernel_launch(e.get('name', '')))
        if cfg_kernel_count == 0:
            continue
        
        # Find direct children of cfg
        direct = []
        for e in cfg_events:
            if e is cfg:
                continue
            e_ts, e_dur = e['ts'], e.get('dur', 0)
            # Check if nested inside another cfg_event
            is_direct = not any(
                is_within(e_ts, e_dur, o['ts'], o.get('dur', 0))
                for o in cfg_events if o is not e and o is not cfg
            )
            if is_direct:
                direct.append(e)
        
        # Filter to those with kernel launches in subtree
        for child in direct:
            child_name = child.get('name', '')
            # Skip GPU kernels (cat='kernel'), kernel launches, CompiledFxGraph, and filtered modules
            if child.get('cat') == 'kernel':
                continue
            if is_kernel_launch(child_name):
                continue
            if 'CompiledFxGraph' in child_name:
                continue
            if should_filter(child_name) or should_filter_prefill(child_name):
                continue
            
            # Count kernel launches in child's subtree
            c_ts, c_dur = child['ts'], child.get('dur', 0)
            kernel_count = sum(
                1 for e in cfg_events
                if is_kernel_launch(e.get('name', '')) and is_within(e['ts'], e.get('dur', 0), c_ts, c_dur)
            )
            
            if kernel_count > 0:
                kernel_modules.append((child['ts'], child_name, kernel_count))
    
    # Also find modules parallel to CompiledFxGraph (e.g., unified_attention_with_output_base)
    # These are top-level modules with kernel launches that are NOT inside any CompiledFxGraph
    PARALLEL_MODULE_PATTERNS = ['unified_attention', 'attention_base']
    
    for e in prefill_events:
        e_name = e.get('name', '')
        # Only process modules matching specific patterns
        if not any(p in e_name.lower() for p in PARALLEL_MODULE_PATTERNS):
            continue
        if e.get('cat') == 'kernel':
            continue
        if is_kernel_launch(e_name):
            continue
        
        # Check if inside any CompiledFxGraph
        e_ts, e_dur = e['ts'], e.get('dur', 0)
        inside_cfg = any(
            is_within(e_ts, e_dur, cfg['ts'], cfg.get('dur', 0))
            for cfg in compiled_fx
        )
        if inside_cfg:
            continue
        
        # Count kernel launches in this event's subtree
        kernel_count = sum(
            1 for kl in prefill_events
            if is_kernel_launch(kl.get('name', '')) and is_within(kl['ts'], kl.get('dur', 0), e_ts, e_dur)
        )
        
        if kernel_count > 0:
            kernel_modules.append((e_ts, e_name, kernel_count))
    
    # Sort by timestamp
    kernel_modules = sorted(kernel_modules, key=lambda x: x[0])
    
    # Get GPU kernels in prefill range, sorted by time
    gpu_kernels = [
        e for e in events
        if e.get('cat') == 'kernel' and prefill_start <= e['ts'] <= prefill_end
    ]
    gpu_kernels = sorted(gpu_kernels, key=lambda x: x['ts'])
    
    # Build CSV rows
    rows = []
    gpu_idx = 0
    last_mod_name = None
    
    for ts, mod_name, kernel_count in kernel_modules:
        for i in range(kernel_count):
            # Only show module name on first kernel AND if different from last module
            if i == 0 and mod_name != last_mod_name:
                display_name = mod_name
                last_mod_name = mod_name
            else:
                display_name = ''
            
            # Get GPU kernel info
            if gpu_idx < len(gpu_kernels):
                gpu = gpu_kernels[gpu_idx]
                rows.append([display_name, gpu.get('name', 'N/A'), gpu.get('dur', 0)])
                gpu_idx += 1
            else:
                rows.append([display_name, 'N/A', 0])
    
    # Write CSV
    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['cpu_module', 'gpu_kernel', 'duration_us'])
        writer.writerows(rows)
    
    print(f"\nPrefill kernel launch modules: {len(kernel_modules)}")
    print(f"GPU kernels used: {gpu_idx}/{len(gpu_kernels)}")
    print(f"CSV written to: {output_csv} ({len(rows)} rows)")


def parse_decode(events: List[Dict], output_csv: str) -> None:
    """
    Parse decode phase: map capture_graph modules to GPU kernels.
    
    Output CSV columns: cpu_module, gpu_kernel, duration_us
    """
    # Find decode_step events (with bs suffix like decode_step_bs_1)
    decode_steps = [
        e for e in events
        if e.get('name', '').startswith('decode_step') and e.get('ph') == 'X'
    ]
    decode_steps = sorted(decode_steps, key=lambda x: x['ts'])
    if not decode_steps:
        print("No decode_step events found.")
        return
    
    # Group by name and find the first occurrence of each batch size
    # Then select the one with longer duration (GPU thread has longer duration)
    from collections import defaultdict
    by_name = defaultdict(list)
    for ds in decode_steps:
        by_name[ds.get('name', '')].append(ds)
    
    # For each name, prefer the one with longer duration (GPU thread)
    first_ds_candidates = []
    for name, dss in by_name.items():
        # Sort by ts and get earliest
        dss_sorted = sorted(dss, key=lambda x: x['ts'])
        # Find the longest duration among first few
        earliest_ts = dss_sorted[0]['ts']
        candidates = [d for d in dss_sorted if d['ts'] < earliest_ts + 10000]  # within 10ms
        if candidates:
            best = max(candidates, key=lambda x: x.get('dur', 0))
            first_ds_candidates.append(best)
    
    if not first_ds_candidates:
        first_ds = decode_steps[0]
    else:
        # Pick the one with longest duration
        first_ds = max(first_ds_candidates, key=lambda x: x.get('dur', 0))
    
    first_ds_name = first_ds.get('name', '')
    if '_bs_' in first_ds_name:
        bs = first_ds_name.split('_bs_')[-1]
        target_cg_name = f'capture_graph_bs_{bs}'
    else:
        target_cg_name = 'capture_graph'
    
    print(f"First decode_step: {first_ds_name}")
    print(f"Looking for: {target_cg_name}")
    
    # Find matching capture_graph
    capture_graphs = [
        e for e in events
        if e.get('name') == target_cg_name and e.get('ph') == 'X'
    ]
    if not capture_graphs:
        # Fallback: find any capture_graph
        capture_graphs = [
            e for e in events
            if e.get('name', '').startswith('capture_graph') and e.get('ph') == 'X'
        ]
        capture_graphs = sorted(capture_graphs, key=lambda x: x['ts'])
        print(f"No exact match, using first capture_graph")
    
    if not capture_graphs:
        print("No capture_graph events found.")
        return
    
    # Get GPU kernels from first decode_step (within its duration)
    ds1_start = first_ds['ts']
    ds1_end = ds1_start + first_ds.get('dur', 0)
    
    gpu_kernels = [
        e for e in events
        if e.get('cat') == 'kernel' and ds1_start <= e['ts'] <= ds1_end
    ]
    gpu_kernels = sorted(gpu_kernels, key=lambda x: x['ts'])
    print(f"First decode_step (tid={first_ds.get('tid')}): {first_ds_name}")
    print(f"  Range: {ds1_start:.0f} ~ {ds1_end:.0f} (dur={first_ds.get('dur', 0):.0f})")
    print(f"  GPU kernels: {len(gpu_kernels)}")
    
    cg = capture_graphs[0]
    print(f"Using: {cg.get('name')}")
    direct_children = get_direct_children(cg, events)
    kernel_children = [c for c in direct_children if has_kernel_launch(c, events)]
    
    # Build CSV rows
    rows = []
    gpu_idx = 0
    
    for child in kernel_children:
        child_name = child.get('name', '')
        if should_filter(child_name):
            continue
        
        # Get sub-children (actual module names)
        sub_children = get_direct_children(child, events)
        sub_kernel_children = [sc for sc in sub_children if has_kernel_launch(sc, events)]
        
        # Determine modules to process
        modules = sub_kernel_children if sub_kernel_children else [child]
        
        for mod in modules:
            mod_name = mod.get('name', '<unknown>')
            kernel_count = count_kernel_launches(mod, events)
            
            for i in range(kernel_count):
                # Only show module name on first kernel, rest use empty string
                display_name = mod_name if i == 0 else ''
                if gpu_idx < len(gpu_kernels):
                    gpu = gpu_kernels[gpu_idx]
                    rows.append([display_name, gpu.get('name', 'N/A'), gpu.get('dur', 0)])
                    gpu_idx += 1
                else:
                    rows.append([display_name, 'N/A', 0])
    
    # Write CSV
    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['cpu_module', 'gpu_kernel', 'duration_us'])
        writer.writerows(rows)
    
    print(f"GPU kernels used: {gpu_idx}/{len(gpu_kernels)}")
    print(f"CSV written to: {output_csv} ({len(rows)} rows)")


# =============================================================================
# Main
# =============================================================================

def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    
    filepath = sys.argv[1]
    
    print(f"Loading: {filepath}")
    trace = load_trace(filepath)
    events = trace.get('traceEvents', [])
    print(f"Loaded {len(events)} events\n")
    
    print("=" * 60)
    print("PREFILL ANALYSIS")
    print("=" * 60)
    parse_prefill(events, 'prefill_breakdown.csv')
    
    print("\n" + "=" * 60)
    print("DECODE ANALYSIS")
    print("=" * 60)
    parse_decode(events, 'decode_breakdown.csv')


if __name__ == '__main__':
    main()
