"""MONO_INSTPREF=<lines>: set the instruction-prefetch size of the fused MoE kernel's descriptor.

LLVM sets COMPUTE_PGM_RSRC3.INST_PREF_SIZE (bits 11:4 on gfx12) from the code size, at most 255 lines: every wave
launch then prefetches the kernel's first ~32 KB into its WGP's instruction cache, whatever part of the code the
wave runs. With the fused kernel's roles split by WGP, that prefetch would refill each WGP with code it never
executes. This wraps FlyDSL's MlirCompiler.compile and rewrites the field in the code object of every kernel
whose name contains ``name_part``.
"""
import struct

_INSTALLED = {}


def _patch_elf(data: bytearray, name_part: str, lines: int) -> int:
    """Patch INST_PREF_SIZE of every ``*.kd`` symbol containing ``name_part``; returns how many."""
    if data[:4] != b"\x7fELF":
        return 0
    e_shoff = struct.unpack_from("<Q", data, 0x28)[0]
    e_shentsize, e_shnum = struct.unpack_from("<HH", data, 0x3A)
    secs = []
    for i in range(e_shnum):
        o = e_shoff + i * e_shentsize
        _name, sh_type, _flags, sh_addr, sh_offset, sh_size, sh_link = struct.unpack_from("<IIQQQQI", data, o)
        secs.append((sh_type, sh_addr, sh_offset, sh_size, sh_link))
    n = 0
    for sh_type, _a, off, size, link in secs:
        if sh_type not in (2, 11):  # SYMTAB, DYNSYM
            continue
        stroff = secs[link][2]
        for j in range(size // 24):
            st_name, _info, _other, st_shndx, st_value, _st_size = struct.unpack_from("<IBBHQQ", data, off + 24 * j)
            end = data.index(b"\0", stroff + st_name)
            sym = data[stroff + st_name:end].decode(errors="replace")
            if not sym.endswith(".kd") or name_part not in sym or st_shndx >= len(secs):
                continue
            _t, s_addr, s_off, _s, _l = secs[st_shndx]
            kd = s_off + (st_value - s_addr)
            rsrc3 = struct.unpack_from("<I", data, kd + 44)[0]
            rsrc3 = (rsrc3 & ~(0xFF << 4)) | ((lines & 0xFF) << 4)
            struct.pack_into("<I", data, kd + 44, rsrc3)
            n += 1
    return n


def install(name_part: str, lines: int) -> None:
    """Rewrite INST_PREF_SIZE to ``lines`` in kernels whose names contain ``name_part`` (idempotent)."""
    if _INSTALLED.get(name_part) == lines:
        return
    from flydsl._mlir import ir
    from flydsl._mlir.dialects import gpu
    from flydsl.compiler import jit_function as jf

    orig = jf.MlirCompiler.__dict__["compile"].__func__ if not _INSTALLED else jf.MlirCompiler._mono_orig_compile
    jf.MlirCompiler._mono_orig_compile = orig
    _INSTALLED[name_part] = lines

    def compile(cls, module, **kw):
        out = orig(cls, module, **kw)
        for op in out.body.operations:
            if op.operation.name != "gpu.binary":
                continue
            objs = op.operation.attributes["objects"]
            new = []
            changed = 0
            for a in objs:
                oa = gpu.ObjectAttr(a)
                data = bytearray(oa.object)
                c = 0
                for part, ln in _INSTALLED.items():
                    c += _patch_elf(data, part, ln)
                changed += c
                new.append(gpu.ObjectAttr.get(oa.target, oa.format, bytes(data), oa.properties, oa.kernels) if c else a)
            if changed:
                op.operation.attributes["objects"] = ir.ArrayAttr.get(new, context=out.context)
        return out

    jf.MlirCompiler.compile = classmethod(compile)


def read_inst_pref(data: bytes, name_part: str):
    """INST_PREF_SIZE of the first matching ``*.kd`` in a code object (for checks)."""
    buf = bytearray(data)
    before = bytes(buf)
    _patch_elf(buf, name_part, 0)
    for i in range(0, len(before) - 4, 4):
        if before[i:i + 4] != bytes(buf[i:i + 4]):
            return (struct.unpack_from("<I", before, i)[0] >> 4) & 0xFF
    return None
