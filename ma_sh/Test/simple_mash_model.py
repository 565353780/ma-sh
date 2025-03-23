import torch
from tqdm import trange
from torchviz import make_dot

from ma_sh.Model.simple_mash import SimpleMash
from ma_sh.Test.init_values import initValues

def test():
    anchor_num = 800
    mask_degree_max = 3
    sh_degree_max = 2
    sample_phi_num = 10
    sample_theta_num = 10
    use_inv = True
    idx_dtype = torch.int64
    dtype = torch.float32
    device = "cuda:0"

    mash = SimpleMash(
        anchor_num,
        mask_degree_max,
        sh_degree_max,
        sample_phi_num,
        sample_theta_num,
        use_inv,
        idx_dtype,
        dtype,
        device,
    )

    initValues(mash.mask_params, mash.sh_params, mash.rotate_vectors, mash.positions, 2)

    mash.setGradState(True)

    boundary_pts, inner_pts, inner_idxs = mash.toSamplePoints()
    sh_points = torch.vstack([boundary_pts, inner_pts])

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

    for _ in trange(2):
        boundary_pts, inner_pts, inner_idxs = mash.toSamplePoints()
        sh_points = torch.vstack([boundary_pts, inner_pts])

        if False:
            sh_mean = torch.mean(sh_points)

            sh_mean.backward()

            print(mash.mask_params.grad[0])
            print(mash.sh_params.grad[0])
            print(mash.rotate_vectors.grad[0])
            print(mash.positions.grad[0])

    # mash.renderSamplePoints()
    return True
