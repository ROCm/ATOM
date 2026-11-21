"""What a HIP kernel in FLOAT_DENORM_MODE_32 0 does with denormals in
``v_pk_add_f32`` + ``v_cvt_pk_bf16_f32``: add_rmsnorm_quant's residual add on
crafted BF16 bit patterns, against four host models."""

import json

import torch
from aiter import QuantType, dtypes

from atom.model_ops.layernorm import _aiter_rms_quant

dev = torch.device("cuda", torch.cuda.current_device())
HIDDEN = 7168
TINY = 2.0**-126


def i16(v):
    return v - 65536 if v >= 32768 else v


CASES = [
    ("denorm + 0", 0x0003, 0x0000),
    ("-denorm + 0", 0x8003, 0x0000),
    ("big denorm + 0", 0x007F, 0x0000),
    ("1.5m - 1.25m (denormal sum)", 0x00C0, 0x80A0),
    ("denorm + min normal", 0x0040, 0x0080),
    ("denorm + denorm (normal sum)", 0x0040, 0x0040),
    ("denorm + denorm (denormal sum)", 0x0040, 0x0020),
    ("-denorm - denorm", 0x8003, 0x8003),
    ("1 + 2", 0x3F80, 0x4000),
    ("m - m", 0x0080, 0x8080),
    ("inf - inf", 0x7F80, 0xFF80),
    ("nan + 1", 0x7FC0, 0x3F80),
    ("-nan + 1", 0xFFC0, 0x3F80),
    ("snan + 1", 0x7FA0, 0x3F80),
    ("1 + nan", 0x3F80, 0x7FC1),
    ("inf + 1", 0x7F80, 0x3F80),
    ("-0 + -0", 0x8000, 0x8000),
    ("-0 + 0", 0x8000, 0x0000),
    ("-denorm + -0", 0x8003, 0x8000),
]
a = torch.zeros(2, HIDDEN, dtype=torch.bfloat16, device=dev)
b = torch.zeros(2, HIDDEN, dtype=torch.bfloat16, device=dev)
av = torch.tensor([i16(c[1]) for c in CASES], dtype=torch.int16, device=dev)
bv = torch.tensor([i16(c[2]) for c in CASES], dtype=torch.int16, device=dev)
a.view(torch.int16)[0, : len(CASES)] = av
b.view(torch.int16)[0, : len(CASES)] = bv
weight = torch.ones(HIDDEN, dtype=torch.bfloat16, device=dev)
_, _, got = _aiter_rms_quant(
    a, weight, 1e-6, QuantType.per_1x128.value, True, b, dtypes.fp8, True
)
torch.cuda.synchronize()
got = got.view(torch.int16)[0, : len(CASES)].cpu()
af = a.cpu()[0, : len(CASES)].float()
bf = b.cpu()[0, : len(CASES)].float()


def flush(v):
    return torch.where(v.abs() < TINY, torch.copysign(torch.zeros_like(v), v), v)


models = {
    "ieee": af + bf,
    "flush_in_out": flush(flush(af) + flush(bf)),
    "flush_out": flush(af + bf),
    "flush_in": flush(af) + flush(bf),
}
bits = {name: v.to(torch.bfloat16).view(torch.int16) for name, v in models.items()}
for i, (name, x, y) in enumerate(CASES):
    row = {"case": name, "a": hex(x), "b": hex(y), "hip": hex(int(got[i]) & 0xFFFF)}
    row.update({m: hex(int(v[i]) & 0xFFFF) for m, v in bits.items()})
    print(json.dumps(row), flush=True)
