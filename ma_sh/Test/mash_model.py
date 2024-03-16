import torch
from tqdm import trange
from torchviz import make_dot

from ma_sh.Model.mash import Mash


def test():
    anchor_num = 40
    mask_degree_max = 4
    sh_degree_max = 3
    mask_boundary_sample_num = 100
    sample_polar_num = 1000
    sample_point_scale = 0.5
    idx_dtype = torch.int64
    dtype = torch.float64
    device = "cuda:0"

    mash = Mash(
        anchor_num,
        mask_degree_max,
        sh_degree_max,
        mask_boundary_sample_num,
        sample_polar_num,
        sample_point_scale,
        idx_dtype,
        dtype,
        device,
    )

    for i in range(anchor_num):
        mash.mask_params.data[i, 0] = i + 1.0

    for i in range(anchor_num):
        mash.sh_params.data[i, 0] = i + 1.0

    for i in range(anchor_num):
        mash.rotate_vectors.data[i, 0] = i

    for i in range(anchor_num):
        mash.positions.data[i, 0] = i

    mash.setGradState(True)

    sh_points = mash.toSamplePoints()

    mean_point_value = torch.mean(sh_points)

    g = make_dot(
        mean_point_value,
        params={
            "mask_params": mash.mask_params,
            "sh_params": mash.sh_params,
            "rotate_vectors": mash.rotate_vectors,
            "positions": mash.positions,
        },
    )

    g.render("./output/Mash.gv", view=False)

    for _ in trange(100):
        sh_points = mash.toSamplePoints()

    print(sh_points)
    # mash.renderSamplePoints()
    return True
