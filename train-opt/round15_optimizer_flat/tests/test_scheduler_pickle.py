"""Regression: the r15 scheduler wrappers must not leak into state_dict (torch.save must work)."""
import io
import torch


def test_scheduler_state_dict_is_picklable_and_roundtrips():
    from round15_optimizer_flat.impl.patch import install_checkpoint_compat

    class FakeFlat:
        class flat:  # noqa: N801
            params = []
            device = torch.device("cpu")
        dead = []
        ctl = [0, 0]

        def device_counters(self):
            return 0, 0

    p = torch.nn.Parameter(torch.zeros(2))
    opt = torch.optim.AdamW([p], lr=1e-3)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 1.0)
    install_checkpoint_compat(FakeFlat(), opt, sched)
    state = sched.state_dict()
    assert "state_dict" not in state and "load_state_dict" not in state
    buf = io.BytesIO()
    torch.save(state, buf)
    buf.seek(0)
    sched.load_state_dict(torch.load(buf, weights_only=False))
