import torch
from tqdm import trange
from torchviz import make_dot

from ma_sh.Model.mash import Mash


def test():
    anchor_num = 4
    mask_degree_max = 2
    sh_degree_max = 2
    mask_boundary_sample_num = 100
    sample_polar_num = 1000
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

    for i in range(anchor_num):
        mash.mask_params.data[i, 0] = i + 10.0

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

    mean_point_value.backward()

    print(mash.mask_params.grad)
    print(mash.sh_params.grad)
    print(mash.rotate_vectors.grad)
    print(mash.positions.grad)

    for _ in trange(20):
        if False:
            mash.mask_params.data = (
                torch.randn(mash.mask_params.shape, dtype=dtype).to(device) * 100.0
            )
            mash.sh_params.data = (
                torch.randn(mash.sh_params.shape, dtype=dtype).to(device) * 100.0
            )
            mash.rotate_vectors.data = (
                torch.randn(mash.rotate_vectors.shape, dtype=dtype).to(device) * 100.0
            )
            mash.positions.data = (
                torch.randn(mash.positions.shape, dtype=dtype).to(device) * 100.0
            )

        sh_points = mash.toSamplePoints()

        if False:
            sh_mean = torch.mean(sh_points)

            sh_mean.backward()

            print(mash.mask_params.grad[0])
            print(mash.sh_params.grad[0])
            print(mash.rotate_vectors.grad[0])
            print(mash.positions.grad[0])

    print(sh_points)
    # mash.renderSamplePoints()
    return True
