import torch
from copy import deepcopy


def test():
    a = torch.randn(100, 30).type(torch.float64).to("cpu")
    a.requires_grad_(True)

    b = 3.0 * a
    b.retain_grad()

    c = 4.0 * b
    c.retain_grad()

    d = 5.0 * torch.sum(c)
    d.retain_grad()

    d.backward(retain_graph=True)

    assert a.grad is not None
    assert b.grad is not None
    assert c.grad is not None
    assert d.grad is not None

    a_grad = deepcopy(a.grad.data)
    b_grad = deepcopy(b.grad.data)
    c_grad = deepcopy(c.grad.data)
    d_grad = deepcopy(d.grad.data)

    a.grad.zero_()
    b.grad.zero_()

    for _ in range(10):
        c = 4.0 * b
        c.retain_grad()

        d = 5.0 * torch.sum(c)
        d.retain_grad()

        d.backward(retain_graph=True)

        assert a.grad is not None
        assert b.grad is not None
        assert c.grad is not None
        assert d.grad is not None

        assert (a.grad.data == a_grad).all()
        assert (b.grad.data == b_grad).all()
        assert (c.grad.data == c_grad).all()
        assert (d.grad.data == d_grad).all()

        a.grad.zero_()
        b.grad.zero_()
    return True
