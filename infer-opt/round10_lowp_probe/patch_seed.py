p = "impl/quant_ref.py"
s = open(p, encoding="utf-8").read()
old = '''def block_hadamard(x: torch.Tensor, b: int) -> torch.Tensor:
    """orthonormal: <H a, H c> == <a, c>.  fp32 math."""
    return fwht(x.float(), b) * (1.0 / math.sqrt(b))'''
new = '''def hadamard_signs(K: int, seed: int, device="cuda") -> torch.Tensor:
    """fixed +-1 vector [K] (fp32) for the randomised block Hadamard (seed 0 = all ones = the plain block Hadamard)."""
    if seed == 0:
        return torch.ones(K, device=device)
    g = torch.Generator(device="cpu").manual_seed(7919 * seed + K)
    return (torch.randint(0, 2, (K,), generator=g).float() * 2 - 1).to(device)


def block_hadamard(x: torch.Tensor, b: int, signs: Optional[torch.Tensor] = None) -> torch.Tensor:
    """orthonormal: <H a, H c> == <a, c>.  fp32 math.  signs [K] (+-1): randomised Hadamard H S x (same signs on activations and weights keep the product)."""
    xf = x.float()
    if signs is not None:
        xf = xf * signs
    return fwht(xf, b) * (1.0 / math.sqrt(b))'''
assert old in s
s = s.replace(old, new)
open(p, "w", encoding="utf-8").write(s)

p = "impl/emu.py"
s = open(p).read()


def rep(old, new, cnt=1):
    global s
    assert s.count(old) == cnt, (old[:70], s.count(old))
    s = s.replace(old, new)


rep('''    pw: int = 0           # fp16 tier: W pre-scale 2^-pw

    def key(self):
        return (self.fmt, self.sq, self.had, self.pfx, self.gptq, self.wfmt or self.fmt, self.pa, self.pw)''',
    '''    pw: int = 0           # fp16 tier: W pre-scale 2^-pw
    seed: int = 0         # randomised block Hadamard: 0 = plain, k > 0 = +-1 sign vector from quant_ref.hadamard_signs(K, k) (noise-realisation robustness checks)

    def key(self):
        return (self.fmt, self.sq, self.had, self.pfx, self.gptq, self.wfmt or self.fmt, self.pa, self.pw, self.seed)''')
rep('''            if spec.had:
                Wm = qr.block_hadamard(Wm, spec.had)
            if spec.pfx:''', '''            if spec.had:
                Wm = qr.block_hadamard(Wm, spec.had, qr.hadamard_signs(Wm.shape[1], spec.seed, Wm.device) if spec.seed else None)
            if spec.pfx:''')
rep('''        if spec.had:
            Xf = qr.block_hadamard(Xf, spec.had)
        Wq = ml["wq"][kind]''', '''        if spec.had:
            Xf = qr.block_hadamard(Xf, spec.had, qr.hadamard_signs(K, spec.seed, Xf.device) if spec.seed else None)
        Wq = ml["wq"][kind]''')
open(p, "w").write(s)
print("patched")
