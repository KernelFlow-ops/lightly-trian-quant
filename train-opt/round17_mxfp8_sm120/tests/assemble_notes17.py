"""Assemble round17/NOTES.md from NOTES_part{1,2,3}.md (static text) + measured tables.  Run on the server:  python3 tests/assemble_notes17.py"""
import subprocess
from pathlib import Path

R = Path("/root/lightly-train/train-opt/round17_mxfp8_sm120")
T = Path("/root/autodl-tmp/train-opt-tmp/round17_mxfp8_sm120/notes_parts")
p1, p2, p3 = [(T / f"NOTES_part{i}.md").read_text(encoding="utf-8") for i in (1, 2, 3)]


def sh(cmd):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd=R).stdout


steptable = sh("python3 tests/step_table.py")

p1 = p1.replace("[[NSYS_EXTRA]]", "vitl16:MX8_fc+proj+qkv_dw vitb16:MX8_fc+proj+qkv_dw vitb16:TB1_R13   # 第二次调用补采")
for k, v in {
    "REC_COL": "MX8_fc+proj+qkv_dw（推荐）", "REC_QUACK": "5.1（qkv fwd，bf16）", "REC_MX": "25.4", "REC_CUB": "6.2", "REC_Q": "**5.06**", "REC_HEAD": "22.3", "REC_EW": "19.6",
    "REC_ATT": "16.4", "REC_OPT": "10.7", "REC_LN": "9.2", "REC_LOSS": "0.2", "REC_TOT": "**120.4**", "REC_N": "3292", "REC_BUSY": "0.794", "REC_GAP": "329 个 / 17.6 ms", "REC_SYNC": "8",
}.items():
    p1 = p1.replace("[[" + k + "]]", v)
p1 = p1.replace("[[RECOMMEND]]", (
    "**推荐 `MX8_fc+proj+qkv_dw`（`r17:fc+proj+qkv_dw`）；稳妥备选 `MX8_fc`。** 两档：(a) 稳妥档 `r17:fc`（只有 fc1/fc2 走 MXFP8，全部块含 joint 块）：ViT-L 1.145× / ViT-B 1.097×（相对 TB1_R13），单步全局梯度余弦 40 批里 38 批 ≥0.99，最低 0.9897；"
    "(b) 性能档 `r17:fc+proj+qkv_dw`（再加 proj 的三个 GEMM，以及 qkv 的 dgrad/wgrad——qkv 的**前向保持 bf16**）：ViT-L 1.179× / ViT-B 1.093×，16 批里 13 批 ≥0.99，最低 0.9880（loss_rel ≤4e-3，Gate S 全过）。"
    "**不推荐** `all`（qkv 前向也走 MXFP8）：ViT-L 1.191× / ViT-B 1.115×，但 qkv 前向的量化噪声在部分批次把全局梯度余弦拉到 0.96–0.99（§5）。"
    "eager 下整步已被 host 限制，(b) 相对 (a) 的 GPU 收益要等 r16 的整步 graph 才兑现（块级 graph 回放 fc+proj 2.63 ms vs fc 2.72 ms，再加 qkv 的 dgrad/wgrad 预计再省 ~0.24 ms/块，未单独实测）"))

gateb = """### Gate B-lite（改精度准则，Amendment 1 §9.4）

`tests/gate_b_multi.py`：同一进程内用同一权重、同一 batch、同一 RngTape（R0 记录，其余回放，`assert_complete` 通过）依次跑 R0（fp32）、TB1（bf16 参考）和若干候选各一步，比较全部梯度/更新。准则：loss_rel ≤2e-2 且全局 grad 余弦 ≥0.99，**同时对 R0 和对 TB1** 成立；逐参数余弦只记录（要求 `≥ cos(TB1,R0) − 0.05`，违例参数列在 JSON 的 `per_param_offenders_vs_r0`，不单独卡）。每次进程调用的“batch”不同（loader 在进程间不是逐位可复现的，同一进程内所有栈共享同一批），所以 3 个 seed 不是固定的 3 批；下表按“批”计数（来源：`profile/gateB_*_mx8*.json`，共 4 轮：v1 的 3 批、policy bisect 的 3 批、seeds 3–10 的 8 批、推荐策略的 6 批、qkv 种类实验的 4 批，视候选而定）。

[[POLICYTABLE]]

**更新余弦与损失**：所有候选、所有批：update 余弦 1.0000（Adam 第一步的更新≈符号），loss_rel 最大 5.3e-3（全部 Linear，ViT-L）/ 4.8e-3（ViT-B），远低于 2e-2；全局 grad_norm_rel ≤0.038。

**读法**：(1) 只有 fc1/fc2：40 批里 38 批通过，两个未通过的批全局余弦分别是 0.9901 对 TB1 / 0.9899 对 R0（ViT-L）、0.9914 对 TB1 / 0.9897 对 R0（ViT-B），都在门槛的 3e-3 内；(2) 加 proj 和 qkv 的 dgrad/wgrad：16 批里 13 批通过，最低 0.9880；(3) 加 qkv 前向（`all`、`attn`、`fc+qkv`）：ViT-B `all` 11 批只有 7 批过（最低 0.9833），ViT-L 在某一批里所有含 qkv 前向的策略都掉到 0.96–0.985（`all+nofirst` 0.9607、`all+nofirst+nolast+nojoint` 0.9621、`all+nojoint` 0.9789、`attn` 0.9783、`fc+qkv` 0.9854），而同一批里 fc / fc+proj 是 0.9995 / 0.9946；(4) 保留首块、末块或 joint 块为 bf16（`nofirst / nolast / nojoint`）不能修复这些批次，噪声不是集中在首末块；同一批里每个块的梯度余弦都只有 0.93–0.97，`queries.weight`、`mask_head` 也只有 0.92–0.96，也就是噪声不集中在某一层，而是整条链上的。

**qkv 按 GEMM 种类的二分**（ViT-L，4 个新批；基线 `fc+proj`；`qkv_f/d/w` = 只有 qkv 的前向/dgrad/wgrad 走 MXFP8）：

[[QKVTABLE]]

**结论**：qkv 的 dgrad 和 wgrad 用 MXFP8 对全局梯度余弦**没有可见影响**（与基线相差 ≤1e-4），而 qkv 的前向（LN1 输出和 qkv 权重的量化）使余弦系统性下降 0.002–0.007。这意味着：前向激活的量化噪声才是这套策略里精度的主要代价，反向 MXFP8 几乎是免费的；这也是 `fc+proj+qkv_dw` 比 `all` 更稳、而 GEMM 节省只少 ~0.12 ms/块的原因（qkv 前向 bf16 195 µs → MXFP8 74 µs）。未验证但值得做的推论：proj / fc 的前向也可能是主要噪声源，“前向 bf16、反向 MXFP8”的策略应该更接近 bf16（GEMM 只省 2/3 的 FLOP），本轮没有实现（`_MXProjRes/_MXFC1/_MXFC2Res` 还没有按种类拆分）。

### Gate S（50 步轨迹，2 个真实 batch，`gate_c(interval=10, max_rel=0.05)`，无 NaN/Inf）

`tests/gate_s_multi.py`：R0、TB1、TB1_R13 和各 MXFP8 策略在同一进程、同两批上各跑 50 步（真实掩码调度，固定 mid / late 相位）。**第一次运行我用了缓存的 R0/TB1 曲线（来自另一进程的另一批数据）与新进程的候选比较，得到 3% 的“偏差”，这是无效比较（连 TB1_R13 对 TB1 都 3%），已删除（文件改名 `*_INVALID_mixed_batches.json` 留作记录），并把缓存键改成所有栈都带同一个 tag 后重跑。**

[[GATESTABLE]]

结果：所有 MXFP8 策略对 R0 和对 TB1 的 EMA 损失最大相对偏差 ≤4.1e-3（ViT-L 全部 Linear）/ ≤2.6e-3（ViT-B），门槛 5e-2；梯度范数比（检查点 10…50 上的单步比值）的范围：ViT-L 全部 Linear [0.917, 1.018]，ViT-B [0.67, 1.36]（门槛 [0.1,10]；对照：bf16 TB1 对 R0 为 [0.995, 1.031]，TB1_R13 对 TB1 为 [0.99, 1.00]），也就是 MXFP8 的单步梯度范数在个别检查点上会偏 ±30%，但损失曲线不受影响；全部有限，全部 PASS。推荐策略 `fc+proj+qkv_dw`：schedule + 固定 mid + 固定 late（late 最后补跑），全 PASS。
"""
qk = sh("python3 - <<'EOF'\nimport json\nd=json.load(open('profile/gateB_vitl16_mx8_qkvkinds.json'))\nseeds=sorted(d['seeds'],key=int)\ncands=list(d['seeds'][seeds[0]]['cands'])\nprint('| policy (all contain fc+proj) | '+' | '.join('batch %s'%s for s in seeds)+' | mean |')\nprint('|---|'+'---|'*(len(seeds)+1))\nfor c in cands:\n    g=[d['seeds'][s]['cands'][c]['vs_ref']['grad_cos_global'] for s in seeds]\n    name=c.replace('MX8_fc+proj','fc+proj').replace('+qkv_',' + qkv_')\n    print('| %s | '%name+' | '.join('%.4f'%x for x in g)+' | %.4f |'%(sum(g)/len(g)))\nEOF")
pol = sh("""cd tests && python3 - <<'EOF'
import finalize_results as F
gb=F.gate_b_stats()
order=['MX8_fc','MX8_fc+proj','MX8_fc+proj+qkv_dw','MX8_all','MX8_attn','MX8_fc+qkv','MX8_all+nojoint','MX8_all+nofirst','MX8_all+nofirst+nolast+nojoint']
for m in ('vitl16','vitb16'):
    print('\\n**%s**\\n'%m)
    print('| policy | batches | pass | grad cos vs TB1: min / mean | loss_rel max (vs TB1) | grad_norm_rel max |')
    print('|---|---|---|---|---|---|')
    for c in order:
        e=gb[m].get(c)
        if not e: continue
        print('| %s | %d | %d / %d | %.4f / %.4f | %.1e | %.3f |'%(c,e['n'],e['pass_n'],e['n'],min(e['gcos']),sum(e['gcos'])/len(e['gcos']),max(e['loss_rel_ref']),max(e['gn_rel'])))
EOF""")
gs = ""
for m in ("vitl16", "vitb16"):
    gs += sh(f"python3 tests/gate_tables.py S profile/gateS_{m}_mx8.json | grep -v 'TB1_R13\\|TB1 vs'")
    gs += sh(f"python3 tests/gate_tables.py S profile/gateS_{m}_mx8_rec.json | grep -v 'TB1 vs' | grep 'MX8'")
    gs += sh(f"python3 tests/gate_tables.py S profile/gateS_{m}_mx8_rec_late.json | grep -v 'TB1 vs' | grep 'MX8'")
gateb = gateb.replace("[[POLICYTABLE]]", pol).replace("[[QKVTABLE]]", qk).replace("[[GATESTABLE]]", gs)
p2 = p2.replace("[[STEPTABLE]]", steptable).replace("[[GATEB]]", gateb)
p2 = p2.replace("完整矩阵（含 joint M=7903、ViT-B M=14406/15806、N=1029 单样本尾块、scale8 数据）由 `test_gemm_mx8.py` 不带 `--quick` 运行（耗时长，仅 quick 在最终代码上重跑；完整矩阵在 v1 代码上未单独跑，见 §7 风险）。",
                "其余形状（joint M=7903、ViT-B M=14406/15806）没有单独跑 GEMM 矩阵（`test_gemm_mx8.py` 不带 `--quick` 的完整矩阵没有运行），而是由下面的块测试覆盖：块测试在这些 M 上执行全部 12 个 GEMM 与 epilogue，并与 fp32 块比较。")
p2 = p2.replace("qkv 按 GEMM 种类（f/d/w/fd/dw/fw）的块级检查也全部 ≥0.9999。", "qkv 按 GEMM 种类（f/d/w/fd/dw/fw，ViT-L plain spiky）的块级检查：全局梯度余弦 ≥0.99994，单个参数 ≥0.9996。")

concl = """**结论**
- **做成了**：自研双输出 MXFP8 量化器（逐位等于 QuACK rceil）+ QuACK block-scaled GEMM（原生 MN-major 操作数，满速）+ round13 的融合 epilogue，整块 fwd+bwd 在 graph 回放下 ViT-L plain 3.63 → 2.30 ms（全 4 个 Linear，1.58×）/ 2.63（fc+proj，1.38×）/ 2.72（fc，1.33×）；整步（eager，三相加权）ViT-L 43.72 → 50.06 / 50.44 / 51.56 / 52.09 img/s（fc / fc+proj / fc+proj+qkv_dw / all，1.145–1.191×），ViT-B 100.67 → 110.46 / 108.91 / 110.03 / 112.29（1.082–1.115×）。相对 R0（16.2 / 37.2）3.0–3.2×，相对 TB1（41.4 / 96.0）1.15–1.26×。
- **量化开销**：每块 203 µs（8 个激活/梯度双输出 + 4 个权重），整步 5.1 ms（ViT-L，kernel 总时间的 4.2–4.4%）/ 4.1 ms（ViT-B）；h / dz 两个最大的张量各 ~64 µs（DRAM 受限），占一半。**未做**生产者融合（GELU epilogue 里直接出量化的 h、LN 输出、γ⊙dX、权重量化进优化器）：eager 下整步已是 host 受限，GPU 侧再省几 ms 看不到；这 5.1 ms 要在 graph 之后才有意义。
- **精度**：单步梯度余弦对 bf16 参考是边缘的——fc 38/40 批、fc+proj+qkv_dw 13/16 批、全部 Linear ViT-B 7/11 批 ≥0.99；所有批 loss_rel ≤5.3e-3、update 余弦 1.0000；Gate S 所有策略全过（EMA 损失偏差 ≤4.1e-3）。qkv 的**前向**是精度代价的主要来源，qkv 的 dgrad/wgrad 免费。
- **峰值显存**：ViT-L 17.48 → 17.07–17.25 GB，ViT-B 15.31 → 14.80–14.88 GB（只省 0.2–0.5 GB，峰值不在块的激活上）。

**什么限制了剩下的收益**（ViT-L 全部 Linear：kernel 117.0 ms，wall 134.5 ms）
1. **host launch**：GPU 忙碌率 0.888 → 0.805，>20 µs 间隙 126 → 275 个（7.5 → 14.7 ms）；MXFP8 把块的 GPU 时间压到 2.3 ms，已经低于 eager 下块的 host enqueue（~2.3 ms，QuACK epilogue/GEMM 的 Python 派发 100–165 µs/次 + 12 次量化 33 µs/次 + autograd）。每块 +12 次量化 kernel 和更贵的 block-scaled 派发是新增的 host 开销。整步 kernel 总时间到 117 ms 的潜力只有 CUDA graph 能兑现：块级 graph 回放已验证可捕获（全部变体、plain/joint、两个模型）；r16 做整步 graph 后预期 wall ≈ kernel 总时间 + 约 5–7 ms 间隙（估计，未测）。
2. head 卷积 22.1 ms（ViT-B 30.4 ms，27%）、elementwise 19.6 ms（ViT-B 28.9 ms）、attention 16.4 ms、优化器 10.4 ms、LN/行 kernel 9.2 ms、量化 5.1 ms、bf16 的 head GEMM 6.2 ms：现在骨干 GEMM（MXFP8 27.6 ms，23.6%）已经不是最大块，ViT-B 上 head 卷积 + elementwise 占一半以上。

**下一步（按预期收益）**
1. 整步 CUDA graph（r16）里验证 MXFP8 块：预期把 ViT-L 从 ~135 ms 推向 ~122 ms；这时才值得做 2 和 3。
2. 把 `fc+proj+qkv_dw` 里没拆的部分按 GEMM 种类拆开，试“前向 bf16 / 反向 MXFP8”（proj、fc1、fc2 的 `kinds`）以及只对 fc 的前向保持 MXFP8：用 Gate B-lite 的批次分布（而不是 3 批）选策略。
3. 生产者融合量化（按大小）：fc1 的 GELU epilogue 直接出 h 的行副本（QuACK 的 SFD 量化 epilogue 对 aux 输出只支持行方向，列副本仍需单独一次读 z 重算 GELU）、dgrad 的 GELU' epilogue 出 dz、LN 输出、`γ⊙dX`；权重量化接到 r15 的 `register_weight_writer`（每步量化 12.6M×24 个权重 ~1 ms）。预期合计 2–3 ms/步（graph 之后）。
4. proj / qkv 的 wgrad（314–421 TFLOPS，输出 tile 数不足 170 个 SM）：试 split-K（`R17_WGRAD_SPLITK=auto`，未整步对照）。

**契约变更请求**
1. CONTRACTS §4 的 `MX8Tensor` 建议改为本轮的 `MX8`：`qr/sfr`（行副本，K-major，scale 沿 C）+ `qc/sfc`（列副本，行主序，scale 沿 token 轴，rows 填充到 128 的倍数且 pad 行为 0），scale 为 QuACK/cuBLAS 的 128×4 swizzle `(rm, rk, 32, 4, 4)` UE8M0，rceil 公式已对 QuACK `to_mx` 逐位验证。
2. 门 B 的改精度准则“全局 grad 余弦 ≥0.99”在单步上对 MXFP8 是边缘的（见 §5，批间波动 0.988–0.9995）；建议改成多批统计准则（例如 ≥8 批、中位数 ≥0.995、最低 ≥0.98）或加 Level C/D 判据，而不是单步 3 个 seed。
3. `common/bench.assert_gpu_exclusive` 对“别的代理留下的空闲 CUDA 上下文”会直接中止测量/nsys（本轮多次发生，r13 的 TB1 整步 nsys 就是这样丢的）；建议所有 GPU 进程都用同一个锁并在持锁区间外释放上下文，或让该检查只看锁持有者。
4. `gate_s` 的 R0/TB1 曲线不可跨进程缓存（批数据在进程间不同）——`check.curve` 文档里应写明。
"""
p3 = p3.replace("[[CONCLUSION]]", concl)
(R / "NOTES.md").write_text(p1 + p2 + p3, encoding="utf-8")
print("NOTES.md written", len(p1 + p2 + p3))
