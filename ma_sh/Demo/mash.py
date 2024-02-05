import torch

from ma_sh.Model.mash import MASH


def demo():
    params = torch.randn([40, 40])

    mash = MASH()
    pts = mash.getPts(params)
    print(pts.shape)
    return True
