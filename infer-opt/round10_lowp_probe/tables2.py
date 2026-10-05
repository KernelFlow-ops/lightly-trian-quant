"""more table generators (decision / robustness); imported by tables.py GEN registration at the bottom."""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
NL = chr(10)


def J(p):
    p = HERE / p
    return json.loads(p.read_text()) if p.exists() else {}


def dec_groups(arg):
    """arg = '<model>_<preset>_<variant>' -> candidate groups with the go rules."""
    d = J(f"profile/final_{arg}.json")
    if not d:
        return "(not available)"
    out = ["| GEMM 种类 | 档位 / 格式 | 每帧节省 ms（central / pess） | kernel 对 r5_v1 | 门稳健性（通过 / 次数） | 判定 |", "|---|---|---|---|---|---|"]
    for g in d["backbone_groups"]:
        sp = g["spec"]
        fmt = sp["fmt"] + (f"+Had{sp['had']}" if sp.get("had") else "")
        rb = "未测" if g["robust_n"] is None else f"{g['robust_MH_pass']} (M+H) / {g['robust_M_pass']} (M) / {g['robust_n']}"
        sc, pp = g["saving_ms_central"], g["saving_ms_pess"]
        ratio = g["kernel_ratio_vs_r5_v1"]
        out.append(f"| {g['kind']} | {g['tier']} / {fmt} | {'-' if sc is None else f'{sc:.3f}'} / {'-' if pp is None else f'{pp:.3f}'} | {'-' if ratio is None else f'{ratio:.2f}x'} | {rb} | {'**go**' if g['go'] else 'no-go：' + g['no_go_reason']} |")
    for h in d["head_groups"]:
        sc, pp = h["saving_ms_central"], h["saving_ms_pess"]
        out.append(f"| head {h['kind']} | {h['tier']} / mx8 | {'-' if sc is None else f'{sc:.3f}'} / {'-' if pp is None else f'{pp:.3f}'} | - | G5 单次（ViT-L / ViT-B 均过） | {'**go**' if h['go'] else 'no-go：' + h['no_go_reason']} |")
    return NL.join(out)


def dec_summary(_=None):
    out = ["| 模型 | 预设 | 变体 | 每种 GEMM 的档位（qkv / proj / fc1 / fc2） | head | 每帧节省 ms（central / pess） | 估计每帧 GPU ms（nsys 基线 → 估计） |", "|---|---|---|---|---|---|---|"]
    for m in ("vitl16", "vitb16"):
        for preset in ("safe", "fast"):
            for v in ("strict", "recommended"):
                d = J(f"profile/final_{m}_{preset}_{v}.json")
                if not d:
                    continue
                pk = d["per_kind_tier"]
                sp = d["per_kind_spec"]
                tiers = " / ".join(pk[k] + ("" if pk[k] == "P0" else f"({sp[k]['fmt']}{'+Had' + str(sp[k]['had']) if sp[k].get('had') else ''})") for k in ("qkv", "proj", "fc1", "fc2"))
                head = ", ".join(f"{k}:{s['fmt']}" for k, s in d["head_spec"].items()) or "P0"
                out.append(f"| {m} | {preset} | {v} | {tiers} | {head} | {d['saving_ms_per_frame_central']:.3f} / {d['saving_ms_per_frame_pess']:.3f} | {d['frame_gpu_ms_r5_v1_nsys']:.2f} → {d['frame_gpu_ms_estimated_central']:.2f}（{100 * d['saving_ms_per_frame_central'] / d['frame_gpu_ms_r5_v1_nsys']:.1f}%） |")
    return NL.join(out)


def robust_table(model):
    d = J(f"profile/robust_{model}.json")
    d2 = {**{f"[fp16 tuned tiles] {k}": v for k, v in J(f"profile/robust_{model}_fp16_tuned.json").items()}, **{f"[fp16 pkt1 tiles] {k}": v for k, v in J(f"profile/robust_{model}_fp16_pkt1.json").items()}}
    out = ["| 候选 | 路径 | M 通过 | M+H 通过 | 门 M conf 最大 % / 每个 seed | 门 H conf 最大 % / 每个 seed | H 池化 IoU 最低 | H logit MAE 中位数 |", "|---|---|---|---|---|---|---|---:|"]
    for k, v in list(d.items()) + list(d2.items()):
        name, _, path = k.rpartition("|")
        out.append(f"| {name} | {path} | {v['pass_M']}/{v['n_seeds']} | {v['pass_M_and_H']}/{v['n_seeds']} | {v['M_conf_max_pct_per_seed']} | {v['H_conf_max_pct_per_seed']} | {min(v['H_iou_per_seed']):.4f} | {v['H_logit_mae_median']:.2e} |")
    return NL.join(out)


def verify2(_=None):
    out = ["| 方案 | 仿真方式 | 门 M：IoU / min-frame / conf 最大 | 门 H（110 帧）：IoU / min-frame IoU / conf 最大 / conf>3% 帧数 / logit MAE | M+H |", "|---|---|---|---|---|"]
    for f in sorted((HERE / "profile").glob("verify_*.json")):
        name = f.stem[len("verify_"):]
        if "_bare" in name or "_margin" in name:
            continue
        v = json.loads(f.read_text())
        m, h = v["M"], v["H"]
        out.append(f"| {name} | {'真实 mg_i8 kernel' if 'realkernel' in name else '反量化操作数 / mg_f16acc kernel'} | {m['pooled_iou']:.4f} / {m['min_iou']:.3f} / {m['conf_max'] * 100:.2f}% | {h['pooled_iou']:.4f} / {h['min_iou']:.3f} / {h['conf_max'] * 100:.2f}% / {h['n_conf_gt3']} / {h['logit_mae']:.2e} | {'通过' if (m['pass'] and h['pass']) else '不过'} |")
    return NL.join(out)
