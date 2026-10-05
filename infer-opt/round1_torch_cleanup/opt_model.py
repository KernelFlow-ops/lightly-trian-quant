"""Round 1: PyTorch-level cleanup of the DINOv3-EoMT semantic-segmentation forward.

`OptEoMT` is built from an already loaded `DINOv3EoMTSemanticSegmentation` (fp32) and
computes exactly the math of `forward_train(..., return_logits_per_layer=False)` with

  1. bf16-resident weights, no autocast (autocast re-casts every fp32 weight per call),
  2. RoPE sin/cos precomputed once ([1024, 64] fp32, same values as backbone.rope_embed),
  3. LinearKMaskedBias' masked bias precomputed,
  4. LayerScale gamma folded into proj / fc2 (W and b, folded in fp32 then rounded),
  5. a single in-place RoPE on the (q,k) slice of the qkv tensor with a signed sin table
     (rotate_half(x)*sin == roll(x, hd/2)*sin_signed): roll, mul, addcmul, copy = 4 kernels,
     no cat / neg / prefix copies; q,k,v are strided views of the qkv tensor,
  6. selectable SDPA backend,
  7. head: optional channels_last / batched-GEMM formulation.

Interfaces
  forward_lowres(x)  -> (mask_logits [B,100,128,128] bf16, class_logits [B,100,K+1] bf16)
  forward_backend(x) -> (mask_logits bilinear-upsampled to x's HxW, class_logits)
"""
from __future__ import annotations

import contextlib

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.attention import SDPBackend, sdpa_kernel

_BACKENDS = {
    "auto": None,
    "flash": SDPBackend.FLASH_ATTENTION,
    "efficient": SDPBackend.EFFICIENT_ATTENTION,
    "cudnn": SDPBackend.CUDNN_ATTENTION,
    "math": SDPBackend.MATH,
}


def _tl_ops():
    import sys
    from pathlib import Path

    p = str(Path(__file__).resolve().parents[1] / "round4_attention" / "impl")
    if p not in sys.path:
        sys.path.insert(0, p)
    import ops  # round4_attention/impl/ops.py (TileLang attention)

    return ops


class _Block(nn.Module):
    """One transformer block with everything pre-folded (bf16)."""

    def __init__(self, blk, dtype: torch.dtype, num_heads: int, res_fp32: bool, attn_impl: str = "sdpa") -> None:
        super().__init__()
        self.attn_impl = attn_impl
        self.res_fp32 = res_fp32
        self.dtype = dtype
        self.h = num_heads
        d = blk.norm1.weight.numel()
        self.d = d
        f32 = torch.float32
        qkv = blk.attn.qkv
        b_qkv = qkv.bias.to(f32) * qkv.bias_mask.to(f32)  # LinearKMaskedBias
        g1 = blk.ls1.gamma.detach().to(f32)
        g2 = blk.ls2.gamma.detach().to(f32)
        w_proj = blk.attn.proj.weight.detach().to(f32) * g1[:, None]
        b_proj = blk.attn.proj.bias.detach().to(f32) * g1
        w_fc2 = blk.mlp.fc2.weight.detach().to(f32) * g2[:, None]
        b_fc2 = blk.mlp.fc2.bias.detach().to(f32) * g2

        def P(t):
            return nn.Parameter(t.detach().to(dtype).contiguous(), requires_grad=False)

        def PL(t):  # LN affine params follow the residual-stream dtype
            return nn.Parameter(t.detach().to(f32 if res_fp32 else dtype).contiguous(), requires_grad=False)

        self.ln1_w, self.ln1_b = PL(blk.norm1.weight), PL(blk.norm1.bias)
        self.ln2_w, self.ln2_b = PL(blk.norm2.weight), PL(blk.norm2.bias)
        self.eps1, self.eps2 = blk.norm1.eps, blk.norm2.eps
        self.w_qkv, self.b_qkv = P(qkv.weight), P(b_qkv)
        self.w_proj, self.b_proj = P(w_proj), P(b_proj)
        self.w_fc1, self.b_fc1 = P(blk.mlp.fc1.weight), P(blk.mlp.fc1.bias)
        self.w_fc2, self.b_fc2 = P(w_fc2), P(b_fc2)

    def forward(self, x: Tensor, rope: tuple[Tensor, Tensor] | None) -> Tensor:
        B, N, d = x.shape
        H = self.h
        hd = d // H
        h = F.layer_norm(x, (d,), self.ln1_w, self.ln1_b, self.eps1)
        if self.res_fp32:
            h = h.to(self.dtype)
        qkv = F.linear(h, self.w_qkv, self.b_qkv).view(B, N, 3, H, hd)
        if self.attn_impl == "tl_rope":  # RoPE fused into the TileLang attention's Q/K tile loads
            o = _tl_ops().attention_rope_qkv(qkv, rope[2], rope[3], N - rope[2].shape[0])
        else:
            if rope is not None:
                sin_s, cos = rope[0], rope[1]  # [hw,1,1,hd] fp32 (sin has the rotate_half sign folded in)
                hw = sin_s.shape[0]
                qk = qkv[:, N - hw :, :2]  # [B,hw,2,H,hd] view (q and k together)
                xr = torch.roll(qk, hd // 2, -1)
                y = torch.addcmul(qk * cos, xr, sin_s)  # fp32 (promotion), like the reference
                qk.copy_(y)  # cast back to bf16 in place
            if self.attn_impl == "tl":
                o = _tl_ops().attention_qkv(qkv)
            else:
                q, k, v = qkv.unbind(2)
                o = F.scaled_dot_product_attention(
                    q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
                )
                o = o.transpose(1, 2).reshape(B, N, d)
        x = x + F.linear(o, self.w_proj, self.b_proj)
        h = F.layer_norm(x, (d,), self.ln2_w, self.ln2_b, self.eps2)
        if self.res_fp32:
            h = h.to(self.dtype)
        h = F.gelu(F.linear(h, self.w_fc1, self.b_fc1))
        return x + F.linear(h, self.w_fc2, self.b_fc2)


class OptEoMT(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        dtype: torch.dtype = torch.bfloat16,
        sdpa_backend: str = "auto",
        head_channels_last: bool = False,
        head_convt_as_gemm: bool = False,
        residual_fp32: bool = True,
        attn_impl: str = "sdpa",  # "sdpa" | "tl" (TileLang) | "tl_rope" (TileLang with fused RoPE)
    ) -> None:
        super().__init__()
        self.residual_fp32 = residual_fp32
        self.dtype = dtype
        self.sdpa_backend = sdpa_backend
        self.head_channels_last = head_channels_last
        self.head_convt_as_gemm = head_convt_as_gemm
        bb = model.backbone
        self.num_queries = model.num_queries
        self.num_joint = model.num_joint_blocks
        self.n_storage = bb.n_storage_tokens
        self.patch = bb.patch_size
        self.d = bb.embed_dim
        heads = bb.blocks[0].attn.num_heads
        dev = next(model.parameters()).device
        f32 = torch.float32

        def P(t):
            return nn.Parameter(t.detach().to(dtype).contiguous(), requires_grad=False)

        with torch.no_grad():
            # patch embed + token prefix (cls + 0*mask_token, registers)
            self.pe_w, self.pe_b = P(bb.patch_embed.proj.weight), P(bb.patch_embed.proj.bias)
            self.prefix = P(
                torch.cat([bb.cls_token + 0 * bb.mask_token, bb.storage_tokens], dim=1)
            )  # [1, 5, d]
            self.queries = P(model.queries.weight)
            self.blocks = nn.ModuleList(_Block(b, dtype, heads, residual_fp32, attn_impl) for b in bb.blocks)
            rdt = f32 if residual_fp32 else dtype
            self.norm_w = nn.Parameter(bb.norm.weight.detach().to(rdt).contiguous(), requires_grad=False)
            self.norm_b = nn.Parameter(bb.norm.bias.detach().to(rdt).contiguous(), requires_grad=False)
            self.norm_eps = bb.norm.eps
            self.cls_w, self.cls_b = P(model.class_head.weight), P(model.class_head.bias)
            mh = model.mask_head
            self.mh = nn.ParameterList(
                [P(mh[i].weight) for i in (0, 2, 4)] + [P(mh[i].bias) for i in (0, 2, 4)]
            )
            # head upscale
            self.up = nn.ParameterList()
            self.up_eps = []
            for sb in model.upscale:
                w = sb.conv1.weight.detach()  # [Cin, Cout, 2, 2]
                self.up.append(P(w))
                self.up.append(P(sb.conv1.bias))
                # GEMM formulation: [Cin, Cout*4] with columns ordered (o, i, j)
                self.up.append(P(w.reshape(w.shape[0], -1).t()))  # [Cout*4, Cin]
                self.up.append(P(sb.conv1.bias.repeat_interleave(4)))
                self.up.append(P(sb.conv2.weight))
                self.up.append(P(sb.norm.weight))
                self.up.append(P(sb.norm.bias))
                self.up_eps.append(sb.norm.eps)
            # RoPE tables (fp32, identical to backbone.rope_embed at 32x32 tokens).
            g = 512 // self.patch
            sin, cos = bb.rope_embed(H=g, W=g)
            hd = sin.shape[-1]
            sign = torch.ones(hd, device=dev, dtype=f32)
            sign[: hd // 2] = -1.0
            self.register_buffer("rope_sin_s", (sin.to(f32) * sign)[:, None, None, :].contiguous(), persistent=False)
            self.register_buffer("rope_cos", cos.to(f32)[:, None, None, :].contiguous(), persistent=False)
            self.register_buffer("rope_sin2d", sin.to(f32).contiguous(), persistent=False)
            self.register_buffer("rope_cos2d", cos.to(f32).contiguous(), persistent=False)
        self.grid = g
        self.eval()

    # -------------------------------------------------------------- head
    def _upscale(self, x: Tensor, B: int) -> Tensor:
        """x: [B, hw, d] patch tokens -> [B, d, 4g, 4g] (logically), possibly channels_last."""
        g = self.grid
        d = self.d
        up = self.up
        if self.head_convt_as_gemm:
            # ConvTranspose2d(k=2,s=2) == GEMM + pixel shuffle; result kept NHWC.
            t = x
            hh = g
            for s in range(len(self.up_eps)):
                w_g, b_g, dw, lw, lb = up[7 * s + 2], up[7 * s + 3], up[7 * s + 4], up[7 * s + 5], up[7 * s + 6]
                y = F.linear(t.reshape(B * hh * hh, d), w_g, b_g)  # [B*hh*hh, d*4] cols (o,i,j)
                y = y.view(B, hh, hh, d, 2, 2).permute(0, 1, 4, 2, 5, 3).reshape(B, 2 * hh, 2 * hh, d)
                y = F.gelu(y)
                y = F.conv2d(y.permute(0, 3, 1, 2), dw, padding=1, groups=d)  # NCHW view of NHWC mem
                hh *= 2
                t = F.layer_norm(y.permute(0, 2, 3, 1), (d,), lw, lb, self.up_eps[s])
            return t  # [B, hh, hh, d]  (NHWC)
        z = x.transpose(1, 2).reshape(B, d, g, g)
        if self.head_channels_last:
            z = z.contiguous(memory_format=torch.channels_last)
        for s in range(len(self.up_eps)):
            w, b, dw, lw, lb = up[7 * s], up[7 * s + 1], up[7 * s + 4], up[7 * s + 5], up[7 * s + 6]
            z = F.conv_transpose2d(z, w, b, stride=2)
            z = F.gelu(z)
            z = F.conv2d(z, dw, padding=1, groups=d)
            z = F.layer_norm(z.permute(0, 2, 3, 1), (d,), lw, lb, self.up_eps[s]).permute(0, 3, 1, 2)
        return z  # [B,d,4g,4g]

    def _predict(self, x: Tensor):
        B = x.shape[0]
        d = self.d
        nq = self.num_queries
        x = F.layer_norm(x, (d,), self.norm_w, self.norm_b, self.norm_eps)
        if self.residual_fp32:
            x = x.to(self.dtype)
        q = x[:, :nq]
        cls = F.linear(q, self.cls_w, self.cls_b)
        m = F.gelu(F.linear(q, self.mh[0], self.mh[3]))
        m = F.gelu(F.linear(m, self.mh[1], self.mh[4]))
        m = F.linear(m, self.mh[2], self.mh[5])  # [B,nq,d]
        pt = x[:, nq + 1 + self.n_storage :]
        z = self._upscale(pt, B)
        if self.head_convt_as_gemm:
            hh = z.shape[1]
            mask = torch.bmm(m, z.reshape(B, hh * hh, d).transpose(1, 2)).view(B, nq, hh, hh)
        else:
            hh = z.shape[-1]
            mask = torch.einsum("bqc,bchw->bqhw", m, z)
        return mask, cls

    # -------------------------------------------------------------- forward
    def _ctx(self):
        b = _BACKENDS[self.sdpa_backend]
        return contextlib.nullcontext() if b is None else sdpa_kernel(b)

    def forward_lowres(self, x: Tensor):
        if x.dtype != self.dtype:
            x = x.to(self.dtype)
        B = x.shape[0]
        d = self.d
        t = F.conv2d(x, self.pe_w, self.pe_b, stride=self.patch)  # [B,d,g,g]
        t = t.flatten(2).transpose(1, 2)  # [B,hw,d]
        t = torch.cat([self.prefix.expand(B, -1, -1), t], dim=1)
        if self.residual_fp32:
            t = t.float()
        rope = (self.rope_sin_s, self.rope_cos, self.rope_sin2d, self.rope_cos2d)
        first_joint = len(self.blocks) - self.num_joint
        with self._ctx():
            for i, blk in enumerate(self.blocks):
                if i == first_joint:
                    q0 = self.queries[None].expand(B, -1, -1)
                    t = torch.cat([q0.to(t.dtype), t], dim=1)
                t = blk(t, rope)
        return self._predict(t)

    def forward_backend(self, x: Tensor):
        mask, cls = self.forward_lowres(x)
        mask = F.interpolate(mask, x.shape[-2:], mode="bilinear")
        return mask, cls

    forward = forward_lowres
