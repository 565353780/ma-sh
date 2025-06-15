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
    test_speed = True

    if test_speed:
        anchor_num = 4000
        mask_degree_max = 3
        sh_degree_max = 2
        sample_phi_num = 40
        sample_theta_num = 40
        dtype = torch.float32
        device = "cuda:0"
        iter_num = 20
        print_grad = False
        render = False
    else:
        anchor_num = 4
        mask_degree_max = 3
        sh_degree_max = 2
        sample_phi_num = 100
        sample_theta_num = 100
        dtype = torch.float32
        device = "cpu"
        iter_num = 1
        print_grad = True
        render = True

    smash = SMash(
        anchor_num=anchor_num,
        mask_degree_max=mask_degree_max,
        sh_degree_max=sh_degree_max,
        sample_phi_num=sample_phi_num,
        sample_theta_num=sample_theta_num,
        use_inv=False,
        dtype=dtype,
        device=device,
    )

    smash.rotate_vectors[:, 0] = 3.0
    smash.rotate_vectors[:, 0] = 2.0
    smash.rotate_vectors[:, 0] = 1.0

    smash.positions[:, 0] = 1.0
    smash.positions[:, 1] = 2.0
    smash.positions[:, 2] = 3.0

    smash.setGradState(True)

    if test_speed:
        for _ in trange(iter_num):
            sample_points2 = smash.toSimpleSamplePoints()

        mean2 = sample_points2.mean()
        mean2.backward()
    else:
        sample_points2 = smash.toSimpleSamplePoints()

    mash = SimpleMash(
        anchor_num=anchor_num,
        mask_degree_max=mask_degree_max,
        sh_degree_max=sh_degree_max,
        sample_phi_num=sample_phi_num,
        sample_theta_num=sample_theta_num,
        dtype=dtype,
        device=device,
    )

    # mash = torch.jit.script(mash)

    with torch.no_grad():
        mash.mask_params.copy_(smash.mask_params.detach().clone())
        mash.sh_params.copy_(smash.sh_params.detach().clone())
        mash.ortho_poses.copy_(
            toOrthoPosesFromRotateVectors(smash.rotate_vectors).detach().clone()
        )
        mash.positions.copy_(smash.positions.detach().clone())

    mash.setGradState(True)

    for _ in trange(iter_num):
        sample_points = mash()

    mean = sample_points.mean()
    mean.backward()

    if print_grad:
        print(mash.mask_params.grad)
        print(mash.sh_params.grad)
        print(mash.ortho_poses.grad)
        print(mash.positions.grad)

    if device == "cpu" and render:
        renderPoints(sample_points2)
        renderPoints(sample_points)

        merge_points = torch.vstack([sample_points2, sample_points.reshape(-1, 3)])
        renderPoints(merge_points)
