"""Independent CPU MoE oracle for TP8 width-1 checkpoint packets.

Unshuffles FP8 weights and computes explicit FP32 matrix products, BF16
expert contributions and rank sums without importing ATOM/FlyDSL kernels.
"""
import argparse
import json
from pathlib import Path

import torch


def error(actual, reference):
    actual, reference = actual.float().flatten(), reference.float().flatten()
    delta = actual - reference
    return dict(rel_l2=float(delta.norm() / reference.norm().clamp_min(1e-12)),
                max_abs=float(delta.abs().max()), exact=bool(torch.equal(actual, reference)))


def unshuffle(weight):
    experts, rows, columns = weight.shape
    return (weight.view(torch.uint8).reshape(experts, rows // 16, columns // 32, 2, 16, 16)
            .permute(0, 1, 4, 2, 3, 5).contiguous()
            .reshape(experts, rows, columns).view(weight.dtype).float())


def analyze(path):
    data = torch.load(path, weights_only=True, map_location='cpu')
    scratch, layout = data['scratch'], data['layout']

    def field(name, dtype):
        offset, size = layout[name]
        return scratch[offset:offset + size].view(dtype)

    mixed = data['mixed'].float().reshape(-1)
    quantized = field('xq', torch.float8_e4m3fn).float().reshape(20, 128)
    scales = field('xq_scale', torch.float32)
    logits = data['router_logits'].float().reshape(-1)
    ids = field('route_ids', torch.int32)
    route_weights = field('route_weights', torch.float32)
    expected_ids = logits[:512].argsort(descending=True, stable=True)[:10]
    expected_weights = torch.softmax(logits[:512], dim=-1)[expected_ids]
    expected_weights /= expected_weights.sum()
    mapped_ids = data['expert_map'][expected_ids.long()].int()
    active = route_weights > 0
    if not torch.equal(ids[active], mapped_ids[active]):
        raise ValueError('Local route mapping differs from independent top-k')
    if not torch.equal(active, mapped_ids >= 0):
        raise ValueError('Local active routes differ from expert ownership')
    report = dict(rank=path.stem, router=error(field('router_logits', torch.bfloat16), logits),
                  routes=ids.tolist(), global_routes=expected_ids.tolist(),
                  route_weights=error(route_weights[active], expected_weights[active])
                  if bool(active.any()) else None)
    dequantized = (quantized * scales[:, None]).reshape(-1)
    w13, w2 = unshuffle(data['w13_weight']), unshuffle(data['w2_weight'])
    scales13 = data['w13_weight_scale'].float().reshape(-1, 10, 20)
    scales2 = data['w2_weight_scale'].float().reshape(-1, 20, 5)
    middle = field('routed_mid', torch.bfloat16).reshape(10, 640)
    middle_q = field('midq', torch.float8_e4m3fn).float().reshape(10, 5, 128)
    middle_scale = field('midq_scale', torch.float32).reshape(10, 5)
    output = torch.zeros(2560)
    details = []
    for slot, expert in enumerate(ids.tolist()):
        if not active[slot]:
            continue
        weight = w13[expert].reshape(10, 128, 20, 128) * scales13[expert, :, None, :, None]
        gate, up = weight.reshape(1280, 2560).mv(dequantized).chunk(2)
        expected = torch.nn.functional.silu(gate) * up
        details.append(dict(slot=slot, mid=error(middle[slot], expected.bfloat16())))
        down_weight = w2[expert].reshape(20, 128, 5, 128) * scales2[expert, :, None, :, None]
        down_input = (middle_q[slot] * middle_scale[slot, :, None]).reshape(-1)
        down = down_weight.reshape(2560, 640).mv(down_input)
        contribution = (down * route_weights[slot]).bfloat16().float()
        output = (output + contribution).bfloat16().float()
    report['g1'] = details

    shared_w13 = data['shared_gate_up_proj.weight'].float()
    shared_w2 = data['shared_down_proj.weight'].float()
    gate, up = shared_w13.mv(mixed).bfloat16().float().chunk(2)
    shared_middle = (torch.nn.functional.silu(gate) * up).bfloat16().float()
    report['shared_mid'] = error(field('shared_mid', torch.bfloat16)[:shared_middle.numel()], shared_middle)
    shared = shared_w2.mv(shared_middle).bfloat16().float()
    shared_gate = torch.sigmoid(logits[-1]).bfloat16().float()
    combined = (output + (shared * shared_gate).bfloat16().float()).bfloat16()
    return report, combined, data['mono_moe'].flatten(), data['native_moe'].flatten()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('input', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    analyzed = [analyze(path) for path in sorted(args.input.glob('*.pt'))]
    expected_ranks = {f'moe_layer0_rank{rank}' for rank in range(8)}
    if {row[0]['rank'] for row in analyzed} != expected_ranks or len(analyzed) != 8:
        raise ValueError('TP8 reference requires exactly one diagnostic packet per rank')
    if any(not torch.equal(row[2], analyzed[0][2]) or
           not torch.equal(row[3], analyzed[0][3]) for row in analyzed):
        raise ValueError('Saved reduced MoE endpoints differ across ranks')
    # Each rank produces its BF16 local contribution before the rank sum.
    oracle = torch.stack([row[1].float() for row in analyzed]).sum(0).bfloat16()
    endpoint = dict(oracle_vs_mono=error(oracle, analyzed[0][2]),
                    oracle_vs_native=error(oracle, analyzed[0][3]))
    report = dict(endpoint=endpoint, ranks=[row[0] for row in analyzed])
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(endpoint, indent=2))
