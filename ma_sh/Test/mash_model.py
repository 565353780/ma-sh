import torch
from tqdm import trange
from torchviz import make_dot

from ma_sh.Model.mash import Mash
from ma_sh.Test.init_values import initValues


def test():
    anchor_num = 100
    mask_degree_max = 1
    sh_degree_max = 3
    mask_boundary_sample_num = 10
    sample_polar_num = 2000
    sample_point_scale = 0.5
    use_inv = True
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
        use_inv,
        idx_dtype,
        dtype,
        device,
    )

    initValues(mash.mask_params, mash.sh_params, mash.rotate_vectors, mash.positions, 2)

    mash.setGradState(True)

    sh_points = mash.toSamplePoints()

    if True:
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

        g.render("./output/grad_graph/Mash.gv", view=False)

        mean_point_value.backward()

    for _ in trange(20):
        sh_points = mash.toSamplePoints()

        if False:
            sh_mean = torch.mean(sh_points)

            sh_mean.backward()

            print(mash.mask_params.grad[0])
            print(mash.sh_params.grad[0])
            print(mash.rotate_vectors.grad[0])
            print(mash.positions.grad[0])

    for _ in trange(20):
        sh_points2 = mash.toSamplePointsUnit()

    print(sh_points == sh_points2)
    print(sh_points.shape)
    print(sh_points2.shape)
    # mash.renderSamplePoints()
    return True
