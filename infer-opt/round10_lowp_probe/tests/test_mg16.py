"""Gate A for the mg_f16acc prototype: fp16-accumulate GEMM + R5 epilogues vs an fp64 reference of the same (fp16-rounded) operands,
and vs the original r5 bf16 kernel (bf16-rounded operands) for scale.  Also the P1 rule: error vs the *fp32* data <= 1.5 x the bf16 GEMM error.

  python tests/test_mg16.py [--model vitl16] [--M 2058]
"""
import argparse
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE / "impl"))
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

import bench10 as b  # noqa: E402
import cands_base as cb  # noqa: E402
import mg16  # noqa: E402

F16, BF = torch.float16, torch.bfloat16


def err(out, ref):
    d = (out.double() - ref.double())
    return dict(max_abs=d.abs().max().item(), rel_l2=(d.norm() / ref.double().norm()).item(),
                cos=F.cosine_similarity(out.double().flatten(), ref.double().flatten(), dim=0).item())


def rope_ref(y, sin, cos, B, Nt, d, H, prefix):
    """y [B*Nt, 3d] fp32 (bias added) -> [3,B,Nt,H,64] with RoPE on q,k for tokens >= prefix."""
    y = y.view(B, Nt, 3, H, 64).permute(2, 0, 1, 3, 4).clone()
    pp = torch.arange(Nt, device=y.device) - prefix
    ok = pp >= 0
    c = cos[pp.clamp(min=0)][None, :, None, :]
    s = sin[pp.clamp(min=0)][None, :, None, :]
    for w in (0, 1):
        x = y[w]
        lo, hi = x[..., :32], x[..., 32:]
        rot = torch.cat((-hi, lo), -1)
        xr = x * c + rot * s
        y[w] = torch.where(ok[None, :, None, None], xr, x)
    return y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="vitl16")
    ap.add_argument("--M", type=int, default=2058)
    ap.add_argument("--kinds", default="qkv,proj,fc1,fc2")
    a = ap.parse_args()
    torch.manual_seed(0)
    Mi = a.M
    B, Nt = cb._batch(Mi)
    d, H = b.MODELS[a.model]["d"], b.MODELS[a.model]["H"]
    prefix = Nt - 1024
    out = {}
    for kind in a.kinds.split(","):
        N, K = b.kind_nk(a.model, kind)
        # realistic-ish statistics: unit-ish activations with a few large channels, small weights
        X = torch.randn(Mi, K, device="cuda") * 0.7
        X[:, :4] *= 8
        W = torch.randn(N, K, device="cuda") * 0.03
        bias = (torch.randn(N, device="cuda") * 0.1).to(BF)
        res = torch.randn(B, Nt, N, device="cuda")
        gamma = torch.rand(N, device="cuda") + 0.1
        sin = torch.randn(1024, 64, device="cuda"); cos = torch.randn(1024, 64, device="cuda")
        # ---------- references on the exact fp32 data
        y_ref32 = X.double() @ W.double().t() + bias.double()
        if kind == "fc1":
            ref32 = F.gelu(y_ref32)
        elif kind in ("proj", "fc2"):
            ref32 = res.view(-1, N).double() + gamma.double() * y_ref32
        else:
            ref32 = rope_ref(y_ref32.float(), sin, cos, B, Nt, d, H, prefix).double()
        # ---------- bf16 r5 kernel (current)
        Xb, Wb = X.to(BF), W.to(BF)
        o = cb.r5()
        if kind == "qkv":
            ob = o.qkv_rope(Xb.view(B, Nt, K), Wb, bias, sin, cos, prefix, H).transpose(2, 3).float()
        elif kind == "fc1":
            ob = o.gemm_bias_gelu(Xb, Wb, bias).float()
        else:
            ob = o.gemm_bias_residual(Xb, Wb, bias, res.view(B, Nt, N), gamma).view(-1, N)
        # ---------- fp16 prototype (no pre-scale needed at these magnitudes), promotion every k-tile
        X16, W16 = X.to(F16), W.to(F16)
        aux = dict(sin=sin, cos=cos, prefix=prefix, res=res.view(B, Nt, N), gamma=gamma)
        of = mg16.run_kind(kind, X16, W16, bias, aux=aux, B=B, heads=H)
        of = of.transpose(2, 3).float() if kind == "qkv" else of.float().view(-1, N) if kind != "qkv" else of
        # reference of the SAME rounded fp16 operands in fp64 (isolates accumulation + output rounding)
        y16 = X16.double() @ W16.double().t() + bias.double()
        if kind == "fc1":
            r16 = F.gelu(y16)
        elif kind in ("proj", "fc2"):
            r16 = res.view(-1, N).double() + gamma.double() * y16
        else:
            r16 = rope_ref(y16.float(), sin, cos, B, Nt, d, H, prefix).double()
        if kind == "qkv":
            of_cmp, ob_cmp = of, ob
        else:
            of_cmp, ob_cmp = of.view(r16.shape), ob.view(r16.shape)
        e_acc = err(of_cmp, r16)           # fp16-acc kernel vs exact product of its own operands
        e_f16 = err(of_cmp, ref32)         # vs exact fp32 data
        e_b16 = err(ob_cmp, ref32)         # bf16 r5 kernel vs exact fp32 data
        # for the residual epilogues the output is dominated by the fp32 residual -> also compare the GEMM branch
        if kind in ("proj", "fc2"):
            g_f = (of_cmp.double() - res.view(-1, N).double()) / gamma.double()
            g_b = (ob_cmp.double() - res.view(-1, N).double()) / gamma.double()
            e_f16["branch_rel_l2"] = ((g_f - y_ref32).norm() / y_ref32.norm()).item()
            e_b16["branch_rel_l2"] = ((g_b - y_ref32).norm() / y_ref32.norm()).item()
        out[kind] = dict(vs_own_operands=e_acc, f16acc_vs_fp32=e_f16, bf16_r5_vs_fp32=e_b16)
        ratio = (e_f16.get("branch_rel_l2", e_f16["rel_l2"])) / (e_b16.get("branch_rel_l2", e_b16["rel_l2"]))
        print(f"{a.model} {kind:5s} M={Mi}: f16acc vs own-operand fp64: rel_l2={e_acc['rel_l2']:.2e} max_abs={e_acc['max_abs']:.2e} cos={e_acc['cos']:.8f} | "
              f"vs fp32 data: f16acc={e_f16.get('branch_rel_l2', e_f16['rel_l2']):.2e} bf16={e_b16.get('branch_rel_l2', e_b16['rel_l2']):.2e} ratio={ratio:.2f}", flush=True)
    b.merge_results(HERE / "results.json", {"mg16_test": {f"{a.model}|M{Mi}": out}})


if __name__ == "__main__":
    main()
