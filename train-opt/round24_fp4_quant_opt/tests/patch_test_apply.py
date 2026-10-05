s = open('test_fp4q2.py').read()


def rep(old, new):
    global s
    assert old in s, old[:80]
    s = s.replace(old, new, 1)


rep("\nimport fp4q\n", "\nimport fp4q2 as fp4q\n")
rep('"""R21a gate A: fused FP4 quantizer (impl/fp4q.py, impl/fp4_kernels.cu) against the frozen bit-exact reference round20_fp4_probe/impl/fp4_ref.py.',
    '"""R24 gate A (copy of round21 tests/test_fp4q.py with the module under test replaced by round24 impl/fp4q2.py, plus the parts "fast" and "philox7"):\nfused FP4 quantizer (impl/fp4q2.py, impl/fp4_kernels2.cu) against the frozen bit-exact reference round20_fp4_probe/impl/fp4_ref.py and against round21 fp4q.')
# fold the spread amax publication before the tests read the next slot
rep('    checks["next_row"] = bool(torch.equal(st.amax[1, 1, 0], a_row))', '    st.fold()  # round24: the fast kernels publish the amax spread over NPART sectors; roll() / fold() move it to amax[1]\n    checks["next_row"] = bool(torch.equal(st.amax[1, 1, 0], a_row))')
rep('    checks["w_next"] = bool(torch.equal(st.amax[1, 2, 0], xw.abs().amax()))', '    st.fold()\n    checks["w_next"] = bool(torch.equal(st.amax[1, 2, 0], xw.abs().amax()))')
rep('    checks["step1_next_row"] = bool(torch.equal(st.amax[1, 1, 0], x2.float().abs().amax()))', '    st.fold()\n    checks["step1_next_row"] = bool(torch.equal(st.amax[1, 1, 0], x2.float().abs().amax()))')
rep('    checks["graph_state_equals_eager"] = bool(torch.equal(stg.amax, ste.amax) and torch.equal(stg.seed, ste.seed) and torch.equal(stg.step, ste.step))',
    '    checks["graph_state_equals_eager"] = bool(torch.equal(stg.amax, ste.amax) and torch.equal(stg.parts, ste.parts) and torch.equal(stg.seed, ste.seed) and torch.equal(stg.step, ste.step))')
rep('    eq_state = torch.equal(stg.amax, ste.amax) and torch.equal(stg.seed, ste.seed) and torch.equal(stg.step, ste.step)',
    '    eq_state = torch.equal(stg.amax, ste.amax) and torch.equal(stg.parts, ste.parts) and torch.equal(stg.seed, ste.seed) and torch.equal(stg.step, ste.step)')

new_parts = open('new_parts.inc').read()
rep('\nPARTS = {"rn": part_rn,', new_parts + '\nPARTS = {"rn": part_rn,')
rep('"headroom": part_headroom}', '"headroom": part_headroom, "fast": part_fast, "philox7": part_philox7}')
open('test_fp4q2.py', 'w').write(s)
import ast
ast.parse(s)
print('ok')
