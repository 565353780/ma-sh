import os
import torch
from torch import nn
from torch.optim import AdamW

from ma_sh.Method.io import loadMashFileParamsTensor

def test():
    mash_file_path = os.environ['HOME'] + '/Dataset/MashV4/ShapeNet/03636649/583a5a163e59e16da523f74182db8f2.npy'
    assert os.path.exists(mash_file_path)

    gt_mash_params = loadMashFileParamsTensor(mash_file_path, torch.float32, 'cuda')
    assert gt_mash_params is not None

    param = nn.Parameter(torch.randn_like(gt_mash_params))

    optimizer = AdamW([param], lr=1e-3)

    loss_fn = nn.MSELoss()

    step = 0
    while True:
        optimizer.zero_grad()
        loss = loss_fn(param, gt_mash_params)
        loss.backward()
        optimizer.step()

        step += 1

        if step % 1000 == 0:
            print(f"Step {step:03d}: Loss = {loss.item()}")
