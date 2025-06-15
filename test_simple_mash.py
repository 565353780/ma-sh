import os

os.environ["CUDA_VISIBLE_DEVICES"] = "7"

import torch
from tqdm import trange

from ma_sh.Model.simple_mash import SimpleMash
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

    mash = SimpleMash(
        anchor_num=anchor_num,
        mask_degree_max=mask_degree_max,
        sh_degree_max=sh_degree_max,
        sample_phi_num=sample_phi_num,
        sample_theta_num=sample_theta_num,
        dtype=dtype,
        device=device,
    )

    with torch.no_grad():
        mash.mask_params[:, 0] = -0.4
        mash.sh_params[:, 0] = 1.0

        mash.ortho_poses[:, 0] = 3.0
        mash.ortho_poses[:, 2] = 2.0
        mash.ortho_poses[:, 4] = 1.0

        mash.positions[:, 0] = 1.0
        mash.positions[:, 1] = 2.0
        mash.positions[:, 2] = 3.0

    mash.setGradState(True)

    for _ in range(20):
        sample_points = mash()
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
        renderPoints(sample_points)
