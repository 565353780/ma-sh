import os
import sys

sys.path.append("../wn-nc")

os.environ["CUDA_VISIBLE_DEVICES"] = "7"

import torch
from tqdm import trange

from ma_sh.Data.simple_mash import SimpleMash
from ma_sh.Model.simple_mash import SimpleMash as SMash
from ma_sh.Method.rotate import toOrthoPosesFromRotateVectors
from ma_sh.Method.render import renderPoints

if __name__ == "__main__":
    device = "cuda:0"
    iter_num = 20

    smash = SMash(
        anchor_num=4000,
        mask_degree_max=3,
        sh_degree_max=2,
        sample_phi_num=40,
        sample_theta_num=40,
        use_inv=False,
        dtype=torch.float32,
        device=device,
    )

    smash.rotate_vectors[:, 0] = 3.0
    smash.rotate_vectors[:, 0] = 2.0
    smash.rotate_vectors[:, 0] = 1.0

    smash.positions[:, 0] = 1.0
    smash.positions[:, 1] = 2.0
    smash.positions[:, 2] = 3.0

    smash.setGradState(True)

    for _ in trange(iter_num):
        sample_points2 = smash.toSimpleSamplePoints()

    mean2 = sample_points2.mean()
    mean2.backward()

    mash = SimpleMash(
        anchor_num=4000,
        mask_degree_max=3,
        sh_degree_max=2,
        sample_phi_num=40,
        sample_theta_num=40,
        dtype=torch.float32,
        device=device,
    )

    mash.mask_params = smash.mask_params.detach().clone()
    mash.sh_params = smash.sh_params.detach().clone()
    mash.ortho_poses = (
        toOrthoPosesFromRotateVectors(smash.rotate_vectors).detach().clone()
    )
    mash.positions = smash.positions.detach().clone()

    mash.setGradState(True)

    for _ in trange(iter_num):
        sample_points = mash.toSamplePoints()

    mean = sample_points.mean()
    mean.backward()

    merge_points = torch.vstack([sample_points2, sample_points.reshape(-1, 3)])

    if device == "cpu":
        renderPoints(sample_points2)
        renderPoints(sample_points)

        renderPoints(merge_points)
