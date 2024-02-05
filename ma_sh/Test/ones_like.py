import torch


def test():
    a = torch.randn([40, 40]).type(torch.float32)
    try:
        a = a.to("cuda")
    except:
        try:
            a = a.to("mps")
        except:
            pass

    b = torch.ones_like(a)
    assert a.dtype == b.dtype
    assert a.device == b.device
    return True
