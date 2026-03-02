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
import bisect
from typing import List, Dict, Any, Tuple, Optional

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
# Optimized Event Index for fast time-range queries
# =============================================================================

class EventIndex:
    """Pre-indexed events for fast time-range queries."""
    
    def __init__(self, events: List[Dict]):
        # Filter duration events only
        self.duration_events = [e for e in events if e.get('ph') == 'X']
        self.duration_events.sort(key=lambda x: x['ts'])
        self.ts_list = [e['ts'] for e in self.duration_events]
        
        # Pre-compute kernel launch flags and prefix sum
        self._is_kernel_launch = [is_kernel_launch(e.get('name', '')) for e in self.duration_events]
        self._kernel_prefix_sum = [0]
        for is_kl in self._is_kernel_launch:
            self._kernel_prefix_sum.append(self._kernel_prefix_sum[-1] + (1 if is_kl else 0))
    
    def events_in_range(self, start_ts: float, end_ts: float) -> List[Dict]:
        """Get all duration events within [start_ts, end_ts]."""
        left = bisect.bisect_left(self.ts_list, start_ts)
        right = bisect.bisect_right(self.ts_list, end_ts)
        return [
            e for e in self.duration_events[left:right]
            if e['ts'] + e.get('dur', 0) <= end_ts
        ]
    
    def count_kernel_launches_in_range(self, start_ts: float, end_ts: float) -> int:
        """Count kernel launches within time range (fast using prefix sum)."""
        left = bisect.bisect_left(self.ts_list, start_ts)
        right = bisect.bisect_right(self.ts_list, end_ts)
        count = 0
        for i in range(left, right):
            e = self.duration_events[i]
            if e['ts'] + e.get('dur', 0) <= end_ts and self._is_kernel_launch[i]:
                count += 1
        return count
    
    def get_direct_children(self, parent: Dict) -> List[Dict]:
        """Get direct children of parent event (optimized)."""
        p_ts = parent['ts']
        p_end = p_ts + parent.get('dur', 0)
        
        # Get candidates in parent's time range
        candidates = [
            e for e in self.events_in_range(p_ts, p_end)
            if e is not parent
        ]
        
        if not candidates:
            return []
        
        # Filter to direct children only (not nested in other candidates)
        # Sort by duration descending - larger events are potential parents
        candidates_sorted = sorted(candidates, key=lambda x: -x.get('dur', 0))
        
        direct = []
        for i, c in enumerate(candidates_sorted):
            c_ts, c_dur = c['ts'], c.get('dur', 0)
            c_end = c_ts + c_dur
            # Check if c is nested inside any larger candidate
            is_nested = False
            for j in range(i):  # Only check larger (earlier in sorted list)
                o = candidates_sorted[j]
                o_ts = o['ts']
                o_end = o_ts + o.get('dur', 0)
                if c_ts >= o_ts and c_end <= o_end:
                    is_nested = True
                    break
            if not is_nested:
                direct.append(c)
        
        return sorted(direct, key=lambda x: x['ts'])
    
    def count_kernel_launches(self, event: Dict) -> int:
        """Count kernel launches within event's time range."""
        e_ts = event['ts']
        e_end = e_ts + event.get('dur', 0)
        return self.count_kernel_launches_in_range(e_ts, e_end)
    
    def has_kernel_launch(self, event: Dict) -> bool:
        """Check if event contains any kernel launch."""
        return self.count_kernel_launches(event) > 0


# =============================================================================
# Legacy functions (for prefill compatibility)
# =============================================================================

def find_events(events: List[Dict], name: str, prefix: bool = False) -> List[Dict]:
    """Find all duration events (ph='X') with given name, sorted by time."""
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
    
    candidates = [
        e for e in events
        if e.get('ph') == 'X' and e is not parent
        and is_within(e.get('ts', 0), e.get('dur', 0), p_ts, p_dur)
    ]
    
    direct = []
    for c in candidates:
        c_ts, c_dur = c['ts'], c.get('dur', 0)
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


def clean_module_name(name: str) -> str:
    """Clean and simplify module name."""
    # Remove 'aiter::' prefix if present
    if name.startswith('aiter::'):
        name = name[7:]  # len('aiter::') == 7
    
    # Rename based on keywords (rope takes priority)
    name_lower = name.lower()
    if 'rope' in name_lower:
        return 'rope'
    if 'cache' in name_lower:
        return 'kv_cache'
    
    return name


def process_moe_module(
    mod_name: str,
    kernel_count: int,
    start_gpu_idx: int,
    gpu_kernels: List[Dict]
) -> List[List]:
    """
    Process moe_forward module: categorize kernels by name.
    
    - 'moesort' in kernel name -> moe_sort
    - 'topk' in kernel name -> moe_topk
    - others -> keep original mod_name
    
    Returns list of [display_name, gpu_kernel_name, gpu_dur] rows.
    """
    rows = []
    prev_category = None
    clean_mod_name = clean_module_name(mod_name)
    
    for i in range(kernel_count):
        gpu_idx = start_gpu_idx + i
        gpu_kernel_name = 'N/A'
        gpu_dur = 0
        if gpu_idx < len(gpu_kernels):
            gpu = gpu_kernels[gpu_idx]
            gpu_kernel_name = gpu.get('name', 'N/A')
            gpu_dur = gpu.get('dur', 0)
        
        # Determine category based on kernel name
        kernel_lower = gpu_kernel_name.lower()
        if 'moesort' in kernel_lower:
            category = 'moe_sort'
        elif 'topk' in kernel_lower:
            category = 'moe_topk'
        else:
            category = clean_mod_name
        
        # Show name only on first kernel of each category
        display_name = category if category != prev_category else ''
        prev_category = category
        rows.append([display_name, gpu_kernel_name, gpu_dur])
    
    return rows


def process_regular_module(
    mod_name: str,
    kernel_count: int,
    start_gpu_idx: int,
    gpu_kernels: List[Dict]
) -> List[List]:
    """
    Process regular module: show name on first kernel only.
    
    Returns list of [display_name, gpu_kernel_name, gpu_dur] rows.
    """
    rows = []
    clean_mod_name = clean_module_name(mod_name)
    for i in range(kernel_count):
        gpu_idx = start_gpu_idx + i
        gpu_kernel_name = 'N/A'
        gpu_dur = 0
        if gpu_idx < len(gpu_kernels):
            gpu = gpu_kernels[gpu_idx]
            gpu_kernel_name = gpu.get('name', 'N/A')
            gpu_dur = gpu.get('dur', 0)
        display_name = clean_mod_name if i == 0 else ''
        rows.append([display_name, gpu_kernel_name, gpu_dur])
    return rows


def parse_decode(events: List[Dict], output_csv: str) -> None:
    """
    Parse decode phase: map capture_graph modules to GPU kernels.
    
    Output CSV columns: cpu_module, gpu_kernel, duration_us
    """
    print("Building event index...")
    
    # Find GPU-annotated decode_step events (cat='gpu_user_annotation')
    decode_steps = [
        e for e in events
        if e.get('name', '').startswith('decode_step') 
        and e.get('ph') == 'X'
        and e.get('cat') == 'gpu_user_annotation'
    ]
    decode_steps = sorted(decode_steps, key=lambda x: x['ts'])
    
    if not decode_steps:
        print("No decode_step (gpu_user_annotation) events found.")
        return
    
    # Skip warmup: find first gap > 100ms (warmup/run boundary)
    # Normal decode gaps are < 5ms, so 100ms is safe threshold
    WARMUP_GAP_THRESHOLD = 100000  # 100ms in microseconds
    actual_run_idx = 0
    found_warmup_boundary = False
    for i in range(1, len(decode_steps)):
        gap = decode_steps[i]['ts'] - (decode_steps[i-1]['ts'] + decode_steps[i-1].get('dur', 0))
        if gap > WARMUP_GAP_THRESHOLD:
            actual_run_idx = i
            found_warmup_boundary = True
            print(f"Warmup/run boundary at [{i-1}]->[{i}], gap={gap/1000:.1f}ms")
            break
    
    if not found_warmup_boundary:
        print("No warmup detected (no gap > 100ms), using first decode_step")
    
    first_ds = decode_steps[actual_run_idx]
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
    
    cg = capture_graphs[0]
    print(f"Using: {cg.get('name')}")
    
    # Build optimized index only for capture_graph's time range
    cg_start = cg['ts']
    cg_end = cg_start + cg.get('dur', 0)
    cg_events = [
        e for e in events
        if e.get('ph') == 'X' and e.get('ts', 0) >= cg_start 
        and e.get('ts', 0) + e.get('dur', 0) <= cg_end
    ]
    print(f"Events in capture_graph: {len(cg_events)}")
    idx = EventIndex(cg_events)
    
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
    
    # Use optimized index for children lookup
    direct_children = idx.get_direct_children(cg)
    kernel_children = [c for c in direct_children if idx.has_kernel_launch(c)]
    print(f"Direct children with kernels: {len(kernel_children)}")
    
    # Collect all modules with their kernel info
    all_modules = []  # list of (mod_name, kernel_count, start_gpu_idx)
    gpu_idx = 0
    
    for child in kernel_children:
        child_name = child.get('name', '')
        if should_filter(child_name):
            continue
        
        # Get sub-children (actual module names)
        sub_children = idx.get_direct_children(child)
        sub_kernel_children = [sc for sc in sub_children if idx.has_kernel_launch(sc)]
        
        # Determine modules to process
        modules = sub_kernel_children if sub_kernel_children else [child]
        
        for mod in modules:
            mod_name = mod.get('name', '<unknown>')
            kernel_count = idx.count_kernel_launches(mod)
            all_modules.append((mod_name, kernel_count, gpu_idx))
            gpu_idx += kernel_count
    
    # Find norm positions (rmsnorm in name)
    norm_indices = [i for i, (name, _, _) in enumerate(all_modules) if 'rmsnorm' in name.lower()]
    print(f"Found {len(norm_indices)} norm modules")
    
    # Extract layer 3 (4th layer, 0-indexed)
    # Each layer has 2 norms, so layer N starts at norm index 2*N
    TARGET_LAYER = 3
    norm_start_idx = TARGET_LAYER * 2  # 6 (7th norm, 0-indexed)
    norm_end_idx = (TARGET_LAYER + 1) * 2  # 8 (9th norm, 0-indexed)
    
    if norm_start_idx >= len(norm_indices):
        print(f"Not enough norms for layer {TARGET_LAYER}")
        return
    
    # Module range for layer 3: from norm_indices[6] to norm_indices[8] (exclusive)
    mod_start = norm_indices[norm_start_idx]
    mod_end = norm_indices[norm_end_idx] if norm_end_idx < len(norm_indices) else len(all_modules)
    
    print(f"Layer {TARGET_LAYER}: modules [{mod_start}:{mod_end}] (norms at indices {norm_start_idx}, {norm_start_idx+1})")
    
    # Build CSV rows for layer 3 only
    rows = []
    for mod_name, kernel_count, start_gpu_idx in all_modules[mod_start:mod_end]:
        if 'moe_forward' in mod_name.lower():
            rows.extend(process_moe_module(mod_name, kernel_count, start_gpu_idx, gpu_kernels))
        else:
            rows.extend(process_regular_module(mod_name, kernel_count, start_gpu_idx, gpu_kernels))
    
    # Write CSV
    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['cpu_module', 'gpu_kernel', 'duration_us'])
        writer.writerows(rows)
    
    print(f"Layer {TARGET_LAYER} modules: {mod_end - mod_start}")
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
    
    # print("=" * 60)
    # print("PREFILL ANALYSIS")
    # print("=" * 60)
    # parse_prefill(events, 'prefill_breakdown.csv')
    
    print("\n" + "=" * 60)
    print("DECODE ANALYSIS")
    print("=" * 60)
    parse_decode(events, 'decode_breakdown.csv')


if __name__ == '__main__':
    main()
