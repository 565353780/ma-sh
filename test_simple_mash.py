import os
import sys

sys.path.append("../wn-nc")

os.environ["CUDA_VISIBLE_DEVICES"] = "7"

import torch
from tqdm import trange

from ma_sh.Data.simple_mash import SimpleMash
from ma_sh.Model.simple_mash import SimpleMash as SMash
from ma_sh.Method.render import renderPoints

if __name__ == "__main__":
    mash = SimpleMash(
        anchor_num=4000,
        mask_degree_max=3,
        sh_degree_max=2,
        sample_phi_num=40,
        sample_theta_num=40,
        dtype=torch.float32,
        device="cuda:0",
    )

    mash.setGradState(True)

    smash = SMash(
        anchor_num=4000,
        mask_degree_max=3,
        sh_degree_max=2,
        sample_phi_num=40,
        sample_theta_num=40,
        use_inv=False,
        dtype=torch.float32,
        device="cuda:0",
    )

    smash.setGradState(True)

    for _ in trange(100):
        sample_points = mash.toSamplePoints()

    for _ in trange(100):
        sample_points2 = smash.toSamplePoints()

    mean = sample_points.mean()
    mean.backward()

    mean2 = sample_points2.mean()
    mean2.backward()

    # renderPoints(sample_points)
