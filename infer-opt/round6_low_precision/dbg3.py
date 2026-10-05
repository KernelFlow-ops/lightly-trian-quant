import torch
torch.manual_seed(0)
for K in (1024, 4096):
    M,N=2058,1024
    a=torch.randn(M,K,device='cuda').to(torch.float8_e4m3fn); w=torch.randn(N,K,device='cuda').to(torch.float8_e4m3fn)
    ref=(a.double()@w.double().t())
    one=torch.ones((),device='cuda')
    o=torch._scaled_mm(a,w.t(),scale_a=one,scale_b=one,out_dtype=torch.float32)
    ob=torch._scaled_mm(a,w.t(),scale_a=one,scale_b=one,out_dtype=torch.bfloat16)
    rel=lambda x:((x.double()-ref).abs().mean()/ref.abs().mean()).item()
    print(K,'fp32-out rel err',rel(o),'bf16-out rel err',rel(ob),'bf16 rounding of exact',rel(ref.bfloat16()), 'fp32 matmul', rel(a.float()@w.float().t()))
