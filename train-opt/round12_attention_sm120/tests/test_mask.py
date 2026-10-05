"""Bitwise test of the fused mask builder against the source construction (F.interpolate bilinear > 0, random row disable),
plus the dense-conversion round trip and the host / device-prob RNG-consumption contract."""
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import benchlib as bl  # noqa: E402

R12 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R12.parent))
from round12_attention_sm120.impl import attn_ext  # noqa: E402
from round12_attention_sm120.impl.bitmask import QueryBitmask, build_query_bitmask, make_build_mask  # noqa: E402


def source_mask(logits, n_tokens, prob, rand):
    """Exact copy of common.ops default build_mask (dense bool [B,N,N]) without the ctx."""
    q = 100
    storage = 4
    mask = torch.ones(logits.shape[0], n_tokens, n_tokens, dtype=torch.bool, device=logits.device)
    interp = F.interpolate(logits, size=(32, 32), mode="bilinear")
    interp = interp.view(interp.size(0), interp.size(1), -1)
    mask[:, :q, q + 1 + storage :] = interp > 0
    if rand is not None:
        random_queries = rand > prob
        mask[:, :q, q + 1 + storage :][random_queries] = True
    return mask


def main():
    attn_ext.load_ext()
    results = {}
    g = torch.Generator(device="cuda").manual_seed(0)
    B = 7
    cases = {
        "randn": torch.randn(B, 100, 128, 128, device="cuda", generator=g),
        "scale1e-3": torch.randn(B, 100, 128, 128, device="cuda", generator=g) * 1e-3,
        "tiny_cancel": None,
    }
    x = torch.randn(B, 100, 128, 128, device="cuda", generator=g)
    x[..., 1::4, :] = torch.where(torch.rand_like(x[..., 1::4, :]) < 0.3, -x[..., 2::4, :], x[..., 1::4, :])  # many exact cancellations
    x[..., ::2] = 0.0
    cases["tiny_cancel"] = x
    cases["bf16_randn"] = torch.randn(B, 100, 128, 128, device="cuda", generator=g).to(torch.bfloat16)
    cases["bf16_scale1e-3"] = (torch.randn(B, 100, 128, 128, device="cuda", generator=g) * 1e-3).to(torch.bfloat16)
    all_ok = True
    for name, logits in cases.items():
        for prob_v in (1.0, 0.5, 0.0):
            prob = torch.tensor(prob_v, device="cuda")
            rand = torch.rand(B, 100, device="cuda", generator=g) if prob_v < 1 else None
            ref = source_mask(logits, 1129, prob, rand)
            qb = build_query_bitmask(logits, 105, rand, prob if rand is not None else None)
            got = qb.to_dense(1129)
            ok = bool(torch.equal(ref, got))
            all_ok &= ok
            results[f"{name}/prob{prob_v}"] = {"equal": ok, "mismatches": int((ref != got).sum())}
            print(f"{name:12s} prob={prob_v}: bit-exact={ok} (mismatching entries {int((ref != got).sum())}), shape {tuple(qb.bits.shape)} dtype {qb.bits.dtype}", flush=True)
    # RNG-consumption contract: host mode draws iff host prob < 1 (name mask/j{j}, shape (B,100)); device mode always draws
    from common.rng import RngTape

    class Ctx:  # minimal stand-in for the harness context
        pass

    def make_ctx(host_probs, host_known):
        ctx = Ctx()
        ctx.rng = RngTape(mode="record")
        ctx.host_probs = list(host_probs)
        ctx.meta = {"round1_annealing_host": True} if host_known else {}
        m = Ctx()
        m.num_queries = 100
        m.backbone = Ctx()
        m.backbone.n_storage_tokens = 4
        m.attn_mask_probs = [torch.tensor(float(p), device="cuda") for p in host_probs]
        ctx.model = m
        return ctx

    logits = cases["randn"]
    fallback = lambda *a, **k: "fallback"
    for probs, host_known, device_prob in (((1.0, 0.5, 0.0), True, False), ((1.0, 0.5, 0.0), False, False), ((1.0, 0.5, 0.0), True, True)):
        ctx = make_ctx(probs, host_known)
        bm = make_build_mask(fallback, device_prob=device_prob)
        outs = [bm(ctx, j, logits, (32, 32), 1129) for j in range(3)]
        drawn = sorted(ctx.rng.values)
        kinds = [type(o).__name__ for o in outs]
        results[f"rng/host_known={host_known}/device_prob={device_prob}"] = {"drawn": drawn, "outputs": kinds}
        print(f"host_known={host_known} device_prob={device_prob}: drawn {drawn}, outputs {kinds}", flush=True)
    # fallback for unsupported shapes
    ctx = make_ctx((1.0, 1.0, 1.0), True)
    assert make_build_mask(fallback)(ctx, 0, logits[:, :, :64, :64].contiguous(), (32, 32), 1129) == "fallback"
    (R12 / "profile" / "test_mask.json").write_text(json.dumps({"all_bit_exact": all_ok, "cases": results}, indent=1))
    print("ALL BIT-EXACT" if all_ok else "MISMATCH")


if __name__ == "__main__":
    main()
