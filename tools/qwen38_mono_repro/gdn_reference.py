"""Independent CPU oracle for checkpoint GDN diagnostic packets.

Uses native projections and weights without importing ATOM or FlyDSL kernels.
Stored state is [head, K, V]; the recurrence uses [head, V, K].
"""
import argparse
import json
from pathlib import Path

import torch


def error(actual, reference):
    actual = actual.float().flatten()
    reference = reference.float().flatten()
    delta = actual - reference
    return dict(rel_l2=float(delta.norm() / reference.norm().clamp_min(1e-12)),
                max_abs=float(delta.abs().max()), exact=bool(torch.equal(actual, reference)))


def analyze(path):
    data = torch.load(path, weights_only=True, map_location='cpu')
    raw = data['native_projection'].flatten()
    history = data['initial_conv']
    conv_dim = history.shape[-1]
    weights = data['conv_weight'].reshape(conv_dim, 4)
    value_heads, head_dim = data['initial_state'].shape[:2]
    key_heads = (conv_dim // head_dim - value_heads) // 2
    if key_heads <= 0 or value_heads % key_heads:
        raise ValueError('Invalid GDN head layout')
    value_start = 2 * key_heads * head_dim
    beta_start = conv_dim + value_heads * head_dim

    accumulator = torch.zeros(conv_dim)
    for tap in range(4):
        operand = history[tap] if tap < 3 else raw[:conv_dim]
        accumulator += (operand.float() * weights[:, tap].float()).bfloat16().float()
    conv = torch.nn.functional.silu(accumulator).bfloat16()
    projected = raw.clone()
    projected[:conv_dim] = conv
    report = dict(
        rank=path.stem,
        projection=error(data['mono_projected'][conv_dim:], raw[conv_dim:]),
        conv=error(data['mono_projected'][:conv_dim], conv),
        core=error(data['mono_core'], data['native_core']),
        norm=error(data['mono_normed'], data['native_normed']),
    )
    query = projected[:key_heads * head_dim].float().reshape(key_heads, head_dim)
    key = projected[key_heads * head_dim:value_start].float().reshape(key_heads, head_dim)
    value = projected[value_start:conv_dim].float().reshape(value_heads, head_dim)
    query = query * torch.rsqrt((query * query).sum(-1, keepdim=True) + 1e-6) / head_dim**.5
    key = key * torch.rsqrt((key * key).sum(-1, keepdim=True) + 1e-6)
    b = projected[beta_start:beta_start + value_heads].float()
    a = projected[beta_start + value_heads:beta_start + 2 * value_heads].float()

    for rounded in (False, True):
        state = data['initial_state'].transpose(-1, -2).contiguous().clone()
        core = []
        for head in range(value_heads):
            key_head = head // (value_heads // key_heads)
            decay = torch.exp(-torch.exp(data['a_log'][head].float()) *
                              torch.nn.functional.softplus(a[head] + data['dt_bias'][head].float()))
            beta = torch.sigmoid(b[head])
            if rounded:
                beta = beta.bfloat16().float()
            recurrent = state[head] * decay
            update = (value[head] - recurrent.mv(key[key_head])) * beta
            recurrent = recurrent + torch.outer(update, key[key_head])
            state[head] = recurrent
            core.append(recurrent.mv(query[key_head]).bfloat16())
        tag = 'rounded_beta' if rounded else 'fp32_beta'
        report[tag] = dict(
            state_vs_native=error(state.transpose(-1, -2), data['native_state']),
            state_vs_mono=error(state.transpose(-1, -2), data['mono_state']),
            core_vs_native=error(torch.stack(core), data['native_core']),
            core_vs_mono=error(torch.stack(core), data['mono_core']),
        )
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('input', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    reports = [analyze(path) for path in sorted(args.input.glob('*.pt'))]
    if not reports:
        raise ValueError('No diagnostic packets found')
    args.output.write_text(json.dumps(reports, indent=2))
    print(json.dumps(dict(ranks=len(reports), rounded_beta_state_relative_l2=[
        row['rounded_beta']['state_vs_native']['rel_l2'] for row in reports]), indent=2))
