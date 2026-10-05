"""Current MXFP8 E4M3 1x32 quantization into cuBLAS's blocked UE8M0 layout.

No unsupported SM90 WGMMA/SM100 TMEM instructions are assumed. Hardware
GEMM is torch._scaled_mm's verified sm_120 path. Awaiting operator A gate.
"""
import torch
import triton
import triton.language as tl

@triton.jit
def _quant_mx(X,Q,S,ROWS:tl.constexpr,K:tl.constexpr,STRIDE0:tl.constexpr,
              STRIDE1:tl.constexpr,GROUPS:tl.constexpr,SCALE_COL_TILES:tl.constexpr,
              BM:tl.constexpr):
    r=tl.program_id(0)*BM+tl.arange(0,BM)
    group=tl.program_id(1)
    c=group*32+tl.arange(0,32)
    if STRIDE0==1 and STRIDE1!=1:
        # Backward transposes x/dy. Load their physical contiguous channel
        # dimension first, then transpose on chip before the block reduction.
        xt=tl.load(X+c[:,None]*STRIDE1+r[None,:]*STRIDE0,
                   (c[:,None]<K)&(r[None,:]<ROWS),0).to(tl.float32)
        x=tl.trans(xt)
    else:
        x=tl.load(X+r[:,None]*STRIDE0+c[None,:]*STRIDE1,(r[:,None]<ROWS)&(c[None,:]<K),0).to(tl.float32)
    exponent=tl.ceil(tl.log2(tl.maximum(tl.max(tl.abs(x),1)/448.,1e-30))).to(tl.int32)
    scale=tl.exp2(exponent.to(tl.float32))
    q=tl.minimum(tl.maximum(x/scale[:,None],-448.),448.)
    tl.store(Q+r[:,None]*K+c[None,:],q,(r[:,None]<ROWS)&(c[None,:]<K))
    # Swizzle: [row_tile128,col_tile4,32,row_quarter4,col_inner4].
    off=((r//128*SCALE_COL_TILES+group//4)*32+r%32)*16+(r%128//32)*4+group%4
    tl.store(S+off,exponent+127,r<ROWS)

def quantize_blocks(x):
    assert x.ndim==2 and x.shape[1]%32==0
    rows,k=x.shape;groups=k//32;col_tiles=triton.cdiv(groups,4)
    q=torch.empty((rows,k),device=x.device,dtype=torch.float8_e4m3fn)
    scales=torch.zeros(triton.cdiv(rows,128)*col_tiles*512,device=x.device,dtype=torch.uint8)
    _quant_mx[(triton.cdiv(rows,16),groups)](x,q,scales,rows,k,x.stride(0),x.stride(1),groups,col_tiles,16,
                                           num_warps=4,enable_fp_fusion=False)
    return q,scales.view(torch.float8_e8m0fnu)

class MXScalingState:
    reduction_multiple=32
    @torch.no_grad()
    def quantize(self,key,x,dtype=torch.float8_e4m3fn):
        if dtype!=torch.float8_e4m3fn:raise ValueError('MXFP8 implementation uses E4M3 for all GEMMs')
        return quantize_blocks(x)
