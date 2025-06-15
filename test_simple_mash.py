from ma_sh.Data.simple_mash import SimpleMash
from ma_sh.Method.render import renderPoints

if __name__ == "__main__":
    mash = SimpleMash(
        anchor_num=1,
        mask_degree_max=3,
        sh_degree_max=2,
        sample_phi_num=100,
        sample_theta_num=100,
    )

    mash.setGradState(True)

    sample_points = mash.toSamplePoints()
    print("sample_points")
    print(sample_points.shape)

    mean = sample_points.mean()
    mean.backward()

    renderPoints(sample_points)

    print(mash.mask_params.grad)
    print(mash.sh_params.grad)
    print(mash.ortho_poses.grad)
    print(mash.positions.grad)
