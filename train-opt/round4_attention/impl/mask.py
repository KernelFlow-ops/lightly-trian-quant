"""128->32 bilinear compact mask construction, preserves torch RNG externally."""
import torch
import triton
import triton.language as tl

@triton.jit
def _build(X, DISABLE, OUT, TOTAL: tl.constexpr, P: tl.constexpr, HAS_DISABLE: tl.constexpr,
           BLOCK: tl.constexpr):
    t = tl.program_id(0)*BLOCK + tl.arange(0, BLOCK)
    row, pixel = t // P, t % P
    # 128/32=4: align_corners=False centers are 4*y+1.5, 4*x+1.5.
    iy, ix = (pixel // 32)*4+1, (pixel % 32)*4+1
    off = row*128*128 + iy*128+ix
    a = tl.load(X+off, t<TOTAL, 0).to(tl.float32)
    b = tl.load(X+off+1, t<TOTAL, 0).to(tl.float32)
    c = tl.load(X+off+128, t<TOTAL, 0).to(tl.float32)
    d = tl.load(X+off+129, t<TOTAL, 0).to(tl.float32)
    # ATen bilinear combines horizontally first and rounds to input dtype.
    interp = ((a*0.5+b*0.5)*0.5+(c*0.5+d*0.5)*0.5).to(X.dtype.element_ty)
    allow = interp > 0
    if HAS_DISABLE:
        allow |= tl.load(DISABLE+row, t<TOTAL, 0)
    tl.store(OUT+t, allow, t<TOTAL)

def build_compact(mask_logits, disabled=None):
    assert mask_logits.is_contiguous() and mask_logits.shape[-2:] == (128,128)
    b,q=mask_logits.shape[:2]
    out=torch.empty((b,q,1024),device=mask_logits.device,dtype=torch.bool)
    _build[(triton.cdiv(b*q*1024,256),)](mask_logits, mask_logits if disabled is None else disabled,
         out,b*q*1024,1024,disabled is not None,256)
    return out
