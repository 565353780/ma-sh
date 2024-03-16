import torch
from copy import deepcopy


def testDetachGrad():
    a = torch.ones(100, 30).type(torch.float64).to("cpu") * 3
    a.requires_grad_(True)

    b = a.detach()

    c = torch.mean(a * 2)
    d = torch.mean(b * 2)
    e = c * d

    e.backward()

    a2 = torch.ones(100, 30).type(torch.float64).to("cpu") * 3
    a2.requires_grad_(True)

    b2 = torch.ones(100, 30).type(torch.float64).to("cpu") * 3

    c2 = torch.mean(a2 * 2)
    d2 = torch.mean(b2 * 2)
    e2 = c2 * d2

    e2.backward()

    assert (a.grad == a2.grad).all()
    return True


def testCommonTensorGrad():
    a = torch.ones(100, 30).type(torch.float64).to("cpu") * 3
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


def test():
    testDetachGrad()
    testCommonTensorGrad()
    return True
